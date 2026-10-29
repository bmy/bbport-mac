// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The texture cache: the registry of images by page, the guest pages each image watches, and
// what guest and GPU writes to them do.
#include "textures/texture_cache.h"
#include "shim/bb_env.h"

#include <cstdio>
#include <cstdlib>

#include <xxhash.h>

#include "common/assert.h"
#include "memory/fault_sites.h"
#include "memory/page_watch.h"
#include "render/scheduler.h"

namespace VideoCore {

void TextureCache::RegisterImage(ImageId image_id) {
    BbStats::images_registered.fetch_add(1, std::memory_order_relaxed);
    Image& image = slot_images[image_id];
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "image registered twice");
    image.flags |= ImageFlagBits::Registered;
    ++registry_generation;
    const u64 bytes = (u64(image.info.guest_size) + 1023) & ~u64(1023);
    total_used_memory += bytes;
    BbStats::live_image_bytes.fetch_add(bytes, std::memory_order_relaxed);
    BbStats::live_images.fetch_add(1, std::memory_order_relaxed);
    image.lru_id = lru_cache.Insert(image_id, gc_tick);
    image.lru_touched_tick = gc_tick;
    ImageRegistry::ForEachPageIndex(image.info.guest_address, image.info.guest_size,
                                    [&](u64 page) { registry.At(page).push_back(image_id); });
}

void TextureCache::UnregisterImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(True(image.flags & ImageFlagBits::Registered), "image not registered");
    image.flags &= ~ImageFlagBits::Registered;
    ++registry_generation;
    lru_cache.Free(image.lru_id);
    const u64 bytes = (u64(image.info.guest_size) + 1023) & ~u64(1023);
    total_used_memory -= bytes;
    BbStats::live_image_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    BbStats::live_images.fetch_sub(1, std::memory_order_relaxed);
    ImageRegistry::ForEachPageIndex(
        image.info.guest_address, image.info.guest_size, [&](u64 page) {
            ImageRegistry::Ids* ids = registry.Find(page);
            ASSERT_MSG(ids, "page {:#x} of an image has no images", page << ImageRegistry::PageBits);
            const auto it = std::ranges::find(*ids, image_id);
            ASSERT_MSG(it != ids->end(), "image missing from page {:#x}",
                       page << ImageRegistry::PageBits);
            ids->erase(it);
        });
}

// Each image watches the guest pages under it while it is clean: [track_addr, track_addr_end).
// A guest write fault on an edge page that the image only partly covers gives that page up
// (head or tail) without marking the image dirty: its other pages would show a write into it.

void TextureCache::TrackImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::Registered)) {
        return;
    }
    const VAddr begin = image.info.guest_address;
    const VAddr end = begin + image.info.guest_size;
    if (image.track_addr == begin && image.track_addr_end == end) {
        return;
    }
    if (!image.IsTracked()) {
        image.track_addr = begin;
        image.track_addr_end = end;
        page_watch.WatchImage<true>(begin, image.info.guest_size);
        return;
    }
    if (begin < image.track_addr) {
        TrackImageHead(image_id);
    }
    if (image.track_addr_end < end) {
        TrackImageTail(image_id);
    }
}

void TextureCache::TrackImageHead(ImageId image_id) {
    Image& image = slot_images[image_id];
    const VAddr begin = image.info.guest_address;
    if (False(image.flags & ImageFlagBits::Registered) || image.track_addr == begin) {
        return;
    }
    ASSERT(image.track_addr != 0 && begin < image.track_addr);
    const u64 size = image.track_addr - begin;
    image.track_addr = begin;
    page_watch.WatchImage<true>(begin, size);
}

void TextureCache::TrackImageTail(ImageId image_id) {
    Image& image = slot_images[image_id];
    const VAddr end = image.info.guest_address + image.info.guest_size;
    if (False(image.flags & ImageFlagBits::Registered) || image.track_addr_end == end) {
        return;
    }
    ASSERT(image.track_addr_end != 0 && image.track_addr_end < end);
    const VAddr from = image.track_addr_end;
    image.track_addr_end = end;
    page_watch.WatchImage<true>(from, end - from);
}

void TextureCache::UntrackImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    if (!image.IsTracked()) {
        return;
    }
    const VAddr from = image.track_addr;
    const u64 size = image.track_addr_end - image.track_addr;
    image.track_addr = 0;
    image.track_addr_end = 0;
    if (size != 0) {
        page_watch.WatchImage<false>(from, size);
    }
}

void TextureCache::UntrackImageHead(ImageId image_id) {
    Image& image = slot_images[image_id];
    const VAddr begin = image.info.guest_address;
    if (!image.IsTracked() || begin < image.track_addr) {
        return;
    }
    const VAddr next = BbMem::NextPage(begin);
    image.track_addr = next;
    if (image.track_addr == image.track_addr_end) {
        // Two pages, both written, the image itself maybe not: its hash will tell.
        MarkAsMaybeDirty(image_id, image);
    }
    page_watch.WatchImage<false>(begin, next - begin);
}

void TextureCache::UntrackImageTail(ImageId image_id) {
    Image& image = slot_images[image_id];
    const VAddr end = image.info.guest_address + image.info.guest_size;
    if (!image.IsTracked() || image.track_addr_end < end) {
        return;
    }
    ASSERT(image.track_addr_end != 0);
    const VAddr last = BbMem::PageStart(end);
    image.track_addr_end = last;
    if (image.track_addr == image.track_addr_end) {
        MarkAsMaybeDirty(image_id, image);
    }
    page_watch.WatchImage<false>(last, end - last);
}

void TextureCache::MarkAsMaybeDirty(ImageId image_id, Image& image) {
    if (image.hash == 0) {
        image.hash = XXH3_64bits(std::bit_cast<const u8*>(image.info.guest_address),
                                 image.info.guest_size);
    }
    image.flags |= ImageFlagBits::MaybeCpuDirty;
    UntrackImage(image_id);
}

void TextureCache::TouchImage(const Image& image) {
    if (image.lru_touched_tick == gc_tick) {
        return; // touched this period already (the LRU list is a cache miss)
    }
    image.lru_touched_tick = gc_tick;
    lru_cache.Touch(image.lru_id, gc_tick);
}

void TextureCache::DeleteImage(ImageId image_id) {
    Image& image = slot_images[image_id];
    ASSERT_MSG(!image.IsTracked(), "image still watched");
    ASSERT_MSG(False(image.flags & ImageFlagBits::Registered), "image still registered");
    const auto& meta = image.info.meta_info;
    for (const VAddr address : {meta.cmask_addr, meta.fmask_addr, meta.htile_addr}) {
        if (address) {
            surface_metas.erase(address);
        }
    }
    {
        std::scoped_lock lk{download_mutex};
        download_images.erase(image_id);
    }
    // The image and its views go once the GPU is done with them.
    scheduler.DeferOperation([this, image_id] {
        Image& dead = slot_images[image_id];
        for (const auto& backing : dead.backing_images) {
            for (const ImageViewId view_id : backing.image_view_ids) {
                slot_image_views.erase(view_id);
            }
        }
        slot_images.erase(image_id);
    });
}

void TextureCache::InvalidateMemory(VAddr addr, size_t size) {
    std::scoped_lock lock{mutex};
    const VAddr pages_begin = BbMem::PageStart(addr);
    const VAddr pages_end = BbMem::NextPage(addr + size - 1);
    ForEachImageInRegion(pages_begin, pages_end - pages_begin, [&](ImageId image_id, Image& image) {
        const VAddr begin = image.info.guest_address;
        const VAddr end = begin + image.info.guest_size;
        if (image.Overlaps(addr, size)) {
            // The write is into the image: dirty, and its pages free for the guest.
            image.flags |= ImageFlagBits::CpuDirty;
            BbMem::FaultSites::NoteImage(begin, image.info.guest_size, image.info.size.width,
                                         image.info.size.height, u32(image.info.pixel_format),
                                         u32(image.info.tile_mode));
            UntrackImage(image_id);
        } else if (pages_end < end) {
            // A write on its first page, before it: that page is given up.
            UntrackImageHead(image_id);
        } else if (begin < pages_begin) {
            // On its last page, after it.
            UntrackImageTail(image_id);
        } else {
            // Within these pages but not written: no other page can tell, the hash will.
            MarkAsMaybeDirty(image_id, image);
        }
    });
}

void TextureCache::WatchImage(const Image& image, const char* what) {
    static const u64 watched = [] {
        const char* env = bb_env("BB_IMAGE_WATCH");
        return env ? std::strtoull(env, nullptr, 0) : 0ull;
    }();
    if (!watched || image.info.guest_address != watched) {
        return;
    }
    static std::atomic<u32> printed{0};
    if (printed.fetch_add(1) < 200000) {
        std::printf("Image watch %#llx: frame %llu %s, flags %#x, %ux%u fmt %u, image %p\n",
                    (unsigned long long)watched,
                    (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed), what,
                    u32(image.flags), image.info.size.width, image.info.size.height,
                    u32(image.info.pixel_format), static_cast<const void*>(&image));
    }
}

void TextureCache::InvalidateMemoryFromGPU(VAddr address, size_t max_size) {
    std::scoped_lock lock{mutex};
    // Only fully overwritten images starting at `address` can be refreshed wholesale.
    // A partial overwrite must preserve the image's other GPU-only texels.
    ForEachImageInRegion(address, 1, [&](ImageId, Image& image) {
        static const bool log = bb_env("BB_IMAGE_SYNC_LOG") != nullptr;
        if (log) {
            static std::atomic<u32> printed{0};
            if (printed.fetch_add(1) < 200) {
                std::printf("Image sync: GPU write %#llx+%#llx over image %#llx %ux%u fmt %u "
                            "flags %#x%s\n",
                            (unsigned long long)address, (unsigned long long)max_size,
                            (unsigned long long)image.info.guest_address, image.info.size.width,
                            image.info.size.height, u32(image.info.pixel_format), u32(image.flags),
                            image.info.guest_address == address ? "" : " (not at its start: kept)");
            }
        }
        if (image.info.guest_address != address || image.info.guest_size > max_size) {
            return;
        }
        WatchImage(image, "GPU write (marked for refresh)");
        image.flags |= ImageFlagBits::GpuDirty; // uploaded again from memory on next use
    });
}

void TextureCache::UnmapMemory(VAddr cpu_addr, size_t size) {
    std::scoped_lock lock{mutex};
    ImageIds gone;
    ForEachImageInRegion(cpu_addr, size, [&](ImageId id, Image&) { gone.push_back(id); });
    for (const ImageId id : gone) {
        FreeImage(id);
    }
}

u64 TextureCache::FreeDeadImages(VAddr addr, size_t size, u64* gpu_made_bytes) {
    std::scoped_lock lock{mutex};
    ImageIds gone;
    u64 bytes = 0;
    ForEachImageInRegion(addr, size, [&](ImageId id, Image& image) {
        if (image.info.guest_address >= addr &&
            image.info.guest_address + image.info.guest_size <= addr + size) {
            gone.push_back(id);
            bytes += image.info.guest_size;
            if (gpu_made_bytes && True(image.flags & ImageFlagBits::GpuModified)) {
                *gpu_made_bytes += image.info.guest_size;
            }
        }
    });
    for (const ImageId id : gone) {
        FreeImage(id); // the memory is the game's no more: nothing to write back
    }
    return bytes;
}

} // namespace VideoCore
