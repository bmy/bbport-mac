// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The texture cache: uploads of images the guest (or the GPU, through buffers) changed.
#include <cstring>
#include "shim/bb_env.h"
#include "textures/texture_cache.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>

#include <xxhash.h>

#include "bbport_button_prompts.h"
#include "bbport_toggles.h"
#include "buffers/buffer_cache.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "vulkan/instance.h"

namespace VideoCore {

namespace {

/// BB_TEXTURE_UPLOAD_MB_PER_FRAME (32; 0: off): the image bytes uploaded a frame before new
/// textures start with their small levels only, the large ones following in later frames
/// (SrShadowy's suggestion; the GTX 1050 laptop uploaded 100-230 MB of textures in one frame on
/// entering an area, 150-500 ms frames). Their views start at the levels there (min LOD) meanwhile.
u64 UploadBudgetBytes() {
    static const u64 bytes = [] {
        const char* env = bb_env("BB_TEXTURE_UPLOAD_MB_PER_FRAME");
        return u64(env && *env ? std::strtoull(env, nullptr, 10) : 32) << 20;
    }();
    return bytes;
}

/// This frame's image upload bytes (RefreshImage runs under the texture cache's lock).
struct UploadBudget {
    u64 frame = ~0ull;
    u64 spent = 0;
    void Roll() {
        const u64 now = BbStats::gpu_frames.load(std::memory_order_relaxed);
        if (now != frame) {
            frame = now;
            spent = 0;
        }
    }
    u64 Remaining() const {
        return spent >= UploadBudgetBytes() ? 0 : UploadBudgetBytes() - spent;
    }
};
UploadBudget upload_budget;

/// Partial uploads: started, completed, bytes left for later (Texture uploads lines).
std::atomic<u64> partial_started{0}, partial_completed{0}, partial_deferred_bytes{0};
u64 partial_started_total = 0, partial_completed_total = 0;

} // namespace

void ReportPartialUploads() {
    static std::atomic<u32> printed{0};
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed.load(std::memory_order_relaxed) < 10 ||
        (partial_started.load() == 0 && partial_completed.load() == 0)) {
        return;
    }
    printed.store(now, std::memory_order_relaxed);
    const u64 started = partial_started.exchange(0), completed = partial_completed.exchange(0);
    partial_started_total += started;
    partial_completed_total += completed;
    std::printf("Texture uploads (10 s): %llu textures started with their small levels, %llu "
                "completed (%llu and %llu in all), %llu MiB of large levels left for later "
                "frames (budget %llu MiB a frame)\n",
                (unsigned long long)started, (unsigned long long)completed,
                (unsigned long long)partial_started_total,
                (unsigned long long)partial_completed_total,
                (unsigned long long)(partial_deferred_bytes.exchange(0) >> 20),
                (unsigned long long)(UploadBudgetBytes() >> 20));
}

namespace {

/// The copy of one level of `image` from its guest layout.
vk::BufferImageCopy LevelCopy(const Image& image, u32 level) {
    const ImageInfo& info = image.info;
    const auto& layout = info.mips_layout[level];
    const u32 width = std::max(info.size.width >> level, 1u);
    const u32 height = std::max(info.size.height >> level, 1u);
    const u32 depth = info.props.is_volume ? std::max(info.size.depth >> level, 1u) : 1u;
    return {
        .bufferOffset = layout.offset,
        .bufferRowLength = layout.pitch,
        .bufferImageHeight = layout.height,
        .imageSubresource{
            .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
            .mipLevel = level,
            .baseArrayLayer = 0,
            .layerCount = info.resources.layers,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {layout.pitch ? std::min(layout.pitch, width) : width,
                        layout.height ? std::min(layout.height, height) : height, depth},
    };
}

/// BB_PROMPT_TRACE=1: the small images the game uploads (which are its key guide icons).
void TracePromptCandidate(const Image& image) {
    static const bool trace = bb_env_on("BB_PROMPT_TRACE");
    if (!trace) {
        return;
    }
    // Any block-compressed image: blocks of the cross icon (KG_Cancel) in any layout.
    if (image.info.props.is_block) {
        if (const u8* cancel = BbPrompts::IconBytes(2)) {
            const auto* bytes = std::bit_cast<const u8*>(image.info.guest_address);
            u32 hits = 0, first = 0;
            for (u32 offset = 0; offset + 16 <= image.info.guest_size; offset += 16) {
                for (u32 b : {27u, 28u, 35u, 36u}) {
                    if (std::memcmp(bytes + offset, cancel + b * 16, 16) == 0 && !hits++) {
                        first = offset;
                    }
                }
            }
            if (hits) {
                std::printf("Prompt trace: %u cross blocks in uploaded %#llx %ux%u levels %u fmt %s "
                            "tile %u from %#x\n",
                            hits, (unsigned long long)image.info.guest_address,
                            image.info.size.width, image.info.size.height,
                            image.info.resources.levels,
                            vk::to_string(image.info.pixel_format).c_str(),
                            u32(image.info.tile_mode), first);
            }
        }
    }
    // Larger block-compressed images: a key guide icon (one 1024-byte micro tile) copied in?
    if (image.info.props.is_block && image.info.size.width * image.info.size.height > 32 * 32) {
        const auto* bytes = std::bit_cast<const u8*>(image.info.guest_address);
        for (u32 offset = 0; offset + 1024 <= image.info.guest_size; offset += 1024) {
            if (const u32 icon = BbPrompts::IconOf(bytes + offset, 1024)) {
                std::printf("Prompt trace: icon %u inside image %#llx %ux%u fmt %s at %#x (tile %u)\n",
                            icon, (unsigned long long)image.info.guest_address,
                            image.info.size.width, image.info.size.height,
                            vk::to_string(image.info.pixel_format).c_str(), offset,
                            u32(image.info.tile_mode));
            }
        }
        return;
    }
    if (image.info.size.width != 32 || image.info.size.height != 32) {
        return;
    }
    static std::atomic<u32> traced{0};
    if (traced.fetch_add(1, std::memory_order_relaxed) >= 2000) {
        return;
    }
    const auto* bytes = std::bit_cast<const u8*>(image.info.guest_address);
    std::printf("Prompt trace: image %#llx %ux%u levels %u layers %u fmt %s block %d size %#x "
                "tile %u hash %016llx: icon %u\n",
                (unsigned long long)image.info.guest_address, image.info.size.width,
                image.info.size.height, image.info.resources.levels, image.info.resources.layers,
                vk::to_string(image.info.pixel_format).c_str(), int(image.info.props.is_block),
                image.info.guest_size, u32(image.info.tile_mode),
                (unsigned long long)XXH3_64bits(bytes, std::min<u32>(image.info.guest_size, 1024)),
                u32(image.prompt_icon));
}

/// BB_IMAGE_SYNC_LOG: the images refreshed most per 5 s (any reason), with their flags.
void NoteRefresh(const Image& image) {
    static const bool log = bb_env("BB_IMAGE_SYNC_LOG") != nullptr;
    if (!log) {
        return;
    }
    // Render-target-sized images: every refresh, with the GPU frame (flicker hunting).
    if (image.info.size.width >= 640 && image.info.size.height >= 360) {
        static std::mutex big_mutex;
        static std::unordered_map<VAddr, u64> last_frame;
        static u32 big = 0;
        const u64 frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
        std::scoped_lock big_lock{big_mutex};
        const u64 last = std::exchange(last_frame[image.info.guest_address], frame);
        if (last + 1 < frame && big++ < 3000) {
            std::printf("Image sync: big refresh %#llx %ux%u fmt %u tile %u flags %#x frame %llu\n",
                        (unsigned long long)image.info.guest_address, image.info.size.width,
                        image.info.size.height, u32(image.info.pixel_format),
                        u32(image.info.tile_mode), u32(image.flags),
                        (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed));
        }
    }
    if (True(image.flags & ImageFlagBits::GpuDirty)) {
        static std::atomic<u32> printed{0};
        if (printed.fetch_add(1) < 100) {
            std::printf("Image sync: refresh of %#llx (%ux%u) from memory the GPU wrote, flags "
                        "%#x\n",
                        (unsigned long long)image.info.guest_address, image.info.size.width,
                        image.info.size.height, u32(image.flags));
        }
    }
    struct Refreshes {
        u64 count = 0, bytes = 0;
        u32 width = 0, height = 0, format = 0, tiling = 0, flags = 0;
    };
    static std::mutex mutex;
    static std::unordered_map<VAddr, Refreshes> refreshes;
    static auto window_start = std::chrono::steady_clock::now();
    std::scoped_lock lk{mutex};
    auto& r = refreshes[image.info.guest_address];
    r = {r.count + 1, r.bytes + image.info.guest_size, image.info.size.width,
         image.info.size.height, u32(image.info.pixel_format), u32(image.info.tile_mode),
         r.flags | u32(image.flags)};
    const auto now = std::chrono::steady_clock::now();
    if (now - window_start < std::chrono::seconds(5)) {
        return;
    }
    std::vector<std::pair<VAddr, Refreshes>> top(refreshes.begin(), refreshes.end());
    std::ranges::sort(top, std::greater{}, [](const auto& e) { return e.second.bytes; });
    std::string text;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 5); ++i) {
        const auto& [address, e] = top[i];
        text += fmt::format(" {:#x} {}x{} fmt {} tile {}: {}x {:.0f} MB flags {:#x};", address,
                            e.width, e.height, e.format, e.tiling, e.count, e.bytes / 1e6, e.flags);
    }
    std::printf("Image refreshes (5 s, %zu images):%s\n", refreshes.size(), text.c_str());
    refreshes.clear();
    window_start = now;
}

} // namespace

u32 TextureCache::PartialUploadStart(const Image& image) {
    const ImageInfo& info = image.info;
    const u32 levels = info.resources.levels;
    // Sampled color textures with levels to leave out, the view clamp there to keep them unseen.
    if (UploadBudgetBytes() == 0 || !instance.IsImageViewMinLodSupported() || levels < 2 ||
        levels > 15 || info.num_samples > 1 || info.props.is_depth || image.usage.render_target ||
        image.usage.depth_target || image.usage.storage || info.guest_size < (64u << 10)) {
        return 0;
    }
    upload_budget.Roll();
    const u64 remaining = upload_budget.Remaining();
    u64 total = 0;
    for (u32 level = 0; level < levels; ++level) {
        total += info.mips_layout[level].size;
    }
    if (static const bool trace = bb_env("BB_PARTIAL_TRACE") != nullptr; trace) {
        static std::atomic<u32> n{0};
        if (n.fetch_add(1) < 60) {
            std::printf("Partial budget: frame %llu spent %llu KiB total %llu KiB\n",
                        (unsigned long long)upload_budget.frame,
                        (unsigned long long)(upload_budget.spent >> 10),
                        (unsigned long long)(total >> 10));
        }
    }
    if (total <= remaining) {
        return 0;
    }
    // The smallest levels that fit what is left of the budget (16 KiB at least, the smallest
    // level always).
    const u64 allowance = std::max<u64>(remaining, 16u << 10);
    u64 bytes = 0;
    u32 first = levels;
    while (first > 0 && bytes + info.mips_layout[first - 1].size <= allowance) {
        bytes += info.mips_layout[--first].size;
    }
    return std::min(first, levels - 1);
}

void TextureCache::UploadRemainingLevels(Image& image, bool partial_ok) {
    const ImageInfo& info = image.info;
    const u32 to = std::min<u32>(image.resident_level, info.resources.levels);
    u32 from = 0;
    if (partial_ok) {
        upload_budget.Roll();
        const u64 remaining = upload_budget.Remaining();
        if (remaining == 0) {
            return; // next frame
        }
        // As many of the next larger levels as fit (the next one at least, unless others took
        // part of the budget already and it does not fit).
        u64 bytes = 0;
        from = to;
        while (from > 0 && (bytes + info.mips_layout[from - 1].size <= remaining ||
                            (from == to && upload_budget.spent == 0))) {
            bytes += info.mips_layout[--from].size;
        }
        if (from == to) {
            return;
        }
    }
    boost::container::small_vector<vk::BufferImageCopy, 14> copies;
    for (u32 level = from; level < to; ++level) {
        copies.push_back(LevelCopy(image, level));
    }
    WatchImage(image, "refresh: more levels of a partial upload");
    UploadImageLevels(image, from, copies);
    image.resident_level = u8(from);
    ++registry_generation; // bindings find the view of the levels there now
    if (from == 0) {
        partial_completed.fetch_add(1, std::memory_order_relaxed);
    }
    ReportPartialUploads();
}

void TextureCache::UploadImageLevels(Image& image, u32 first_level,
                                     std::span<vk::BufferImageCopy> copies) {
    const ImageInfo& info = image.info;
    u64 bytes = 0;
    for (const auto& copy : copies) {
        bytes += info.mips_layout[copy.imageSubresource.mipLevel].size;
    }
    upload_budget.Roll();
    upload_budget.spent += bytes;
    BbStats::image_upload_bytes.fetch_add(bytes, std::memory_order_relaxed);

    scheduler.EndRendering();
    const auto [source, source_offset] =
        buffer_cache.ObtainBufferForImage(info.guest_address, info.guest_size);
    // The guest's whole description (texture_quality: `info` may leave its largest levels out):
    // those levels are not detiled, nor read over the bus (BB_DETILE_SKIPPED=1: they are, as
    // before), nor the ones a partial upload leaves for later. The staging copy then starts at
    // the first level detiled.
    static const bool detile_skipped = bb_env_on("BB_DETILE_SKIPPED");
    const u32 guest_first = detile_skipped && first_level == 0 ? 0u
                                                                : u32(info.skip_levels) + first_level;
    const auto [linear, linear_offset] =
        detiler.DetileImage(source, source_offset, image.guest_info, guest_first);
    const u64 skipped = image.guest_info.props.is_tiled && guest_first != 0
                            ? image.guest_info.mips_layout[guest_first].offset
                            : 0;
    for (auto& copy : copies) {
        copy.bufferOffset += linear_offset;
        copy.bufferOffset -= skipped;
    }
    runtime.UploadImage(&image, linear, copies);
    ReportPartialUploads();
}

void TextureCache::RefreshImage(Image& image, bool partial_ok) {
    WatchImage(image, "refresh entered");
    // The rest of a partial upload (the guest's copy unchanged since).
    if (False(image.flags & ImageFlagBits::Dirty) && image.resident_level != 0) {
        UploadRemainingLevels(image, partial_ok);
        return;
    }
    if (False(image.flags & ImageFlagBits::Dirty) || image.info.num_samples > 1) {
        return;
    }
    BbStats::Timer timer{BbStats::t_refresh};
    const ImageInfo& info = image.info;

    // Maybe written (a page under it was, not it): compared by its whole contents, as
    // MarkAsMaybeDirty hashes it. bbport (macOS port): its first 8x8 texels alone missed writes
    // elsewhere in it: an icon's corner stays transparent while the rest is filled in (the item
    // picture on the loading screen stayed empty until loading ended).
    if (True(image.flags & ImageFlagBits::MaybeCpuDirty) &&
        False(image.flags & ImageFlagBits::CpuDirty)) {
        const u64 hash =
            XXH3_64bits(std::bit_cast<const u8*>(info.guest_address), info.guest_size);
        if (image.hash == hash) {
            image.flags &= ~ImageFlagBits::MaybeCpuDirty;
            return;
        }
        image.hash = hash;
    }

    const u32 layers = info.resources.layers;
    const u32 levels = info.resources.levels;
    // The game's key guide icons (32x32, one level, known by their bytes): the port may sample
    // its own for the device in use.
    if (info.size.width == 32 && info.size.height == 32 && levels == 1 && layers == 1 &&
        info.props.is_block && info.guest_size >= 1024) {
        image.prompt_icon =
            u8(BbPrompts::IconOf(std::bit_cast<const u8*>(info.guest_address), 1024));
    }
    // The HUD's atlases with key guide icons packed in: the port may sample a copy with its own.
    if (levels == 1 && layers == 1 && info.props.is_block) {
        static std::atomic<u32> uploads{0};
        image.prompt_atlas = u8(BbPrompts::AtlasOf(std::bit_cast<const u8*>(info.guest_address),
                                                   info.guest_size, info.size.width,
                                                   info.size.height));
        image.prompt_upload = image.prompt_atlas ? ++uploads : 0;
    }
    TracePromptCandidate(image);
    NoteRefresh(image);
    // A GPU-written image is reuploaded only where the guest's copy changed (by level hashes):
    // not over what the GPU rendered with a stale copy.
    // Every upload of such an image keeps the hashes, the first one too (a new image is
    // GpuDirty): else the first refresh after the GPU wrote it (a CPU write elsewhere in its
    // page) found no hash and uploaded unchanged memory over the GPU's data. The character
    // creator's exposure (1x1, written by a compute shader) went to 0: no character (#178).
    const bool gpu_modified = True(image.flags & ImageFlagBits::GpuModified);
    const bool gpu_written = gpu_modified && False(image.flags & ImageFlagBits::GpuDirty);
    // Past the frame's upload budget a sampled texture starts with its small levels.
    const u32 first = !gpu_modified && partial_ok ? PartialUploadStart(image) : 0u;
    const bool was_partial = image.resident_level != 0;
    boost::container::small_vector<vk::BufferImageCopy, 14> copies;
    for (u32 level = first; level < levels; ++level) {
        const auto& layout = info.mips_layout[level];
        if (gpu_modified) {
            const u64 hash = XXH3_64bits(
                std::bit_cast<const u8*>(info.guest_address) + layout.offset, layout.size);
            if (gpu_written && image.mip_hashes[level] == hash) {
                continue;
            }
            image.mip_hashes[level] = hash;
        }
        copies.push_back(LevelCopy(image, level));
    }
    WatchImage(image, copies.empty() ? "refresh: nothing to copy" : "refresh: uploading");
    if (copies.empty()) {
        image.flags &= ~ImageFlagBits::Dirty;
        image.resident_level = 0;
        return;
    }
    UploadImageLevels(image, first, copies);
    image.resident_level = u8(first);
    if (first != 0) {
        u64 deferred = 0;
        for (u32 level = 0; level < first; ++level) {
            deferred += info.mips_layout[level].size;
        }
        partial_started.fetch_add(1, std::memory_order_relaxed);
        partial_deferred_bytes.fetch_add(deferred, std::memory_order_relaxed);
        ReportPartialUploads();
    }
    if ((first != 0) != was_partial) {
        ++registry_generation; // bindings find the view of the levels there now
    }
}

} // namespace VideoCore
