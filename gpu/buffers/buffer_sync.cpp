// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport: BufferCache, guest memory to the GPU: synchronizing CPU-modified ranges (uploads,
// chunk copies), memory from images, and the arena's pending sparse binds. Split out of
// buffer_cache.cpp (2026-10-26); shared state in buffer_cache_internal.h.


#include <chrono>
#include "shim/bb_env.h"
#include <array>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <magic_enum/magic_enum.hpp>
#include "shader/profile.h"
#include "bbport_copy.h"
#include "bbport_toggles.h"
#include "bbport_sections.h"
#include "bbport_free_check.h"
#include "bbport_guest_memory.h"
#include "game_profile.h"
#include "bblayer_gpu_memory.h"
#include "bblayer_write_traps.h"
#include "common/alignment.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "cp/command_processor.h"
#include "buffers/buffer.h"
#include "buffers/buffer_cache.h"
#include "memory/dirty_tracker.h"
#include "render/pipeline.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"
#include "buffers/buffer_cache_internal.h"

#include <mutex>
#include <pthread.h>
#include "bbport_cpu.h" // bbport: not <x86intrin.h> (arm64: the native GPU process)
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <fmt/format.h>
#include <sys/mman.h>
#ifdef __APPLE__
#include <sys/ucontext.h> // bbport: <ucontext.h> requires _XOPEN_SOURCE on macOS
#else
#include <ucontext.h>
#endif
#include "common/signal_context.h"
#include "core/signals.h"
#include <vk_mem_alloc.h>
#include "buffers/buffer_cache_internal.h"

namespace VideoCore {

bool BufferCache::IsInPlace(VAddr addr, u64 size) const {
    if (size == 0) {
        return false;
    }
    if (LayerMode()) {
        return BbLayer::GpuMemory::Get().Resolve(addr, size).kind ==
               BbLayer::Resolution::Kind::InPlace;
    }
    const u64 first = addr >> block_shift, last = (addr + size - 1) >> block_shift;
    // bbport: asked for every binding; the answer holds while the residency is unchanged.
    thread_local std::array<ResidencyMemo, 256> in_place_memo{};
    auto& memo = in_place_memo[first & (in_place_memo.size() - 1)];
    const u64 generation = ResidencyGeneration();
    if (memo.first != first || memo.last != last || memo.generation != generation) {
        memo = {first, last, generation, false, in_place_blocks.Contains(first, last + 1)};
    }
    return memo.in_place;
}

bool BufferCache::SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size,
                                    bool is_written, bool is_texel_buffer) {
    boost::container::small_vector<std::pair<vk::Buffer, vk::BufferCopy>, 4> chunk_copies;
    // bbport: the arena is the game's memory there: nothing to upload, no
    // write tracking, and GPU writes need no readback. Images aliasing it still sync.
    if (!LayerMode() && IsInPlace(device_addr, size)) {
        return is_texel_buffer && !is_written &&
               SynchronizeMemoryFromImage(arena, arena->Offset(device_addr), device_addr, size);
    }
    // Partly in place: only the parts in VRAM are synced. An upload over a block in place would
    // write an older copy of the game's memory back into it after the game wrote it again.
    if (in_place_blocks.Overlaps(device_addr >> block_shift,
                                                  ((device_addr + size - 1) >> block_shift) + 1)) {
        const VAddr end = device_addr + size;
        bool image = false;
        in_place_blocks.ForEachGap(
            device_addr >> block_shift, ((end - 1) >> block_shift) + 1, [&](u64 first, u64 last) {
                const VAddr from = std::max<VAddr>(device_addr, first << block_shift);
                const VAddr to = std::min<VAddr>(end, last << block_shift);
                if (from < to) {
                    image |= SynchronizeMemory(arena, from, static_cast<u32>(to - from),
                                               is_written, is_texel_buffer);
                }
            });
        return image;
    }
    boost::container::small_vector<vk::BufferCopy, 4> copies;
    size_t total_size_bytes{};
    const Buffer* src_buffer{};
    memory_tracker->ForEachUploadRange(
        device_addr, size, is_written,
        [&](u64 addr, u64 size) {
            copies.emplace_back(total_size_bytes, addr, size);
            total_size_bytes += size;
        },
        [&] {
            // bbport (BB_GPU_WRITE_TWINS): bytes whose guest copy still matches its twin keep the
            // GPU's data in the arena; twin bytes the guest wrote since are uploaded and no
            // longer GPU-modified.
            if (TwinsEnabled() && !copies.empty()) {
                std::scoped_lock lk{twins_mutex};
                if (!twins.empty()) {
                    boost::container::small_vector<vk::BufferCopy, 4> kept;
                    boost::container::small_vector<std::pair<VAddr, VAddr>, 4> changed;
                    u64 offset = 0;
                    for (const auto& copy : copies) {
                        ForEachTwinRun(copy.dstOffset, copy.dstOffset + copy.size,
                                       [&](VAddr from, VAddr to, TwinRun run) {
                                           if (run == TwinRun::Unchanged) {
                                               twin_kept_bytes.fetch_add(to - from);
                                               return;
                                           }
                                           if (run == TwinRun::Changed) {
                                               twin_guest_bytes.fetch_add(to - from);
                                               gpu_modified_ranges.Subtract(from, to - from);
                                               changed.emplace_back(from, to);
                                           }
                                           kept.push_back({offset, from, to - from});
                                           offset += to - from;
                                       });
                    }
                    // Changed parts are uploaded now: their twins are done.
                    for (const auto& [from, to] : changed) {
                        EraseTwins(from, to);
                    }
                    copies.assign(kept.begin(), kept.end());
                    total_size_bytes = offset;
                }
            }
            // bbport: what lies in a guest memory chunk is copied by the GPU
            // straight from it (no CPU copy into staging); the rest goes through staging.
            if (ChunkUploads() && !copies.empty()) {
                boost::container::small_vector<vk::BufferCopy, 4> staged;
                u64 staged_bytes = 0;
                for (const auto& copy : copies) {
                    const VAddr address = copy.dstOffset;
                    u64 phys = 0;
                    uintptr_t mapping_end = 0;
                    const BbGuestMemory::Chunk* chunk = nullptr;
                    if (runtime_memory_direct_phys(address, &phys, &mapping_end) &&
                        address + copy.size <= mapping_end &&
                        (chunk = BbGuestMemory::Find(phys)) != nullptr &&
                        phys + copy.size <= chunk->phys + chunk->size) {
                        chunk_copies.push_back({chunk->buffer,
                                                {phys - chunk->phys, address - arena->cpu_addr,
                                                 copy.size}});
                        continue;
                    }
                    staged.push_back({staged_bytes, copy.dstOffset, copy.size});
                    staged_bytes += copy.size;
                }
                copies.assign(staged.begin(), staged.end());
                total_size_bytes = staged_bytes;
            }
            src_buffer = UploadCopies(arena, copies, total_size_bytes);
        });

    if (src_buffer) {
        runtime.CopyBuffer(src_buffer, arena, copies);
    }
    for (const auto& [chunk_buffer, copy] : chunk_copies) {
        BbStats::buffer_upload_bytes.fetch_add(copy.size, std::memory_order_relaxed);
        // bbport BB_LAYER_MEMORY: the GPU writes the game's memory too (through the chunk's
        // buffer wrapper): a copy from it waits for those writes like any buffer copy.
        if (LayerMode()) {
            const VAddr address = arena->cpu_addr + copy.dstOffset;
            if (const auto source = LayerInPlace(address, copy.size)) {
                const vk::BufferCopy wrapped{source->second, copy.dstOffset, copy.size};
                runtime.CopyBuffer(source->first, arena, std::span{&wrapped, 1});
                continue;
            }
        }
        runtime.CopyFromGuestChunk(chunk_buffer, arena, std::span{&copy, 1});
    }
    if (is_texel_buffer && !is_written) {
        return SynchronizeMemoryFromImage(arena, arena->Offset(device_addr), device_addr, size);
    }
    return false;
}

const Buffer* BufferCache::UploadCopies(const Buffer* arena, std::span<vk::BufferCopy> copies,
                                        size_t total_size_bytes) {
    if (copies.empty()) {
        return nullptr;
    }
    BbStats::buffer_upload_bytes.fetch_add(total_size_bytes, std::memory_order_relaxed);
    const auto staging = staging_pool.Request(total_size_bytes, MemoryType::HostUncached);
    // bbport: the guest memory is copied into staging on the copy threads, started now in
    // groups of about 1 MiB; submission and guest-visible fences wait for them
    // (Scheduler::WaitHostCopies).
    if (!BbToggle::Disabled(BbToggle::DeferredUploads) && total_size_bytes < 1_MB) {
        // Small uploads join the calling thread's batch.
        for (auto& copy : copies) {
            SmallGuestCopy({
                .run = &RunGuestCopy,
                .context = this,
                .source = copy.dstOffset,
                .destination = reinterpret_cast<u64>(staging.mapped + copy.srcOffset),
                .size = copy.size,
                .extra = reinterpret_cast<u64>(staging.buffer),
            });
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
        }
        return staging.buffer;
    }
    if (!BbToggle::Disabled(BbToggle::DeferredUploads)) {
        struct HostCopy {
            VAddr source;
            u8* destination;
            u64 size;
        };
        using Group = boost::container::small_vector<HostCopy, 8>;
        auto group = std::make_shared<Group>();
        u64 group_bytes = 0;
        const auto launch = [&] {
            BbCopy::Async([group, memory = memory, staging] {
                for (const auto& copy : *group) {
                    memory->CopySparseMemory(copy.source, copy.destination, copy.size);
                }
                staging.Flush();
            });
            for (const auto& copy : *group) {
                scheduler.NoteHostCopySource(copy.source, copy.size);
            }
        };
        for (auto& copy : copies) {
            group->push_back({copy.dstOffset, staging.mapped + copy.srcOffset, copy.size});
            group_bytes += copy.size;
            copy.srcOffset += staging.offset;
            copy.dstOffset -= arena->cpu_addr;
            if (group_bytes >= 1_MB) {
                launch();
                group = std::make_shared<Group>();
                group_bytes = 0;
            }
        }
        if (!group->empty()) {
            launch();
        }
        return staging.buffer;
    }
    const auto copy = [&](std::size_t i) {
        memory->CopySparseMemory(copies[i].dstOffset, staging.mapped + copies[i].srcOffset,
                                 copies[i].size);
    };
    if (total_size_bytes >= 2_MB && copies.size() > 1) {
        BbCopy::ParallelFor(copies.size(), copy);
    } else {
        for (std::size_t i = 0; i < copies.size(); ++i) {
            copy(i);
        }
    }
    for (auto& copy : copies) {
        copy.srcOffset += staging.offset;
        copy.dstOffset -= arena->cpu_addr;
    }
    staging.Flush();
    return staging.buffer;
}

bool BufferCache::SynchronizeMemoryFromImage(const Buffer* arena, u64 arena_offset,
                                             VAddr device_addr, u32 size,
                                             std::pair<const Buffer*, u64>* vram_copy) {
    if (auto type = texture_cache.IsMeta(device_addr)) {
        if (*type == TextureCache::MetaType::HTile) {
            static constexpr u32 ZmaskUncompressed = 0xf;
            runtime.FillBuffer(arena, arena_offset, size, ZmaskUncompressed);
            return true;
        } else {
            LOG_WARNING(Render_Vulkan, "Unhandled metadata type {}", magic_enum::enum_name(*type));
        }
    }
    // bbport: most texel buffers alias no image; remember misses until images change.
    const u64 generation = texture_cache.RegistryGeneration();
    auto& miss = image_miss_cache[((device_addr >> 6) ^ size * 0x9E3779B1u) % image_miss_cache.size()];
    if (miss.address == device_addr && miss.size == size && miss.generation == generation &&
        !BbToggle::Disabled(BbToggle::TextureBindingMemo)) {
        return false;
    }
    const ImageId image_id = texture_cache.FindImageFromRange(device_addr, size);
    if (!image_id) {
        // Only where no image is there at all: an image not yet rendered to (or written over by
        // the CPU) is found once the GPU draws into it, which changes no generation. Memoized,
        // the buffer read stale memory until the next image came or went (#308: frozen motion
        // blur after a lamp or a warp, gone on respawn).
        if (!texture_cache.FindImageFromRange(device_addr, size, false)) {
            miss = {device_addr, size, generation};
        } else if (bb_env("BB_TILE_BACK_LOG")) {
            static std::atomic<u32> printed{0};
            if (printed.fetch_add(1) < 100) {
                std::printf("Tile back: image at %#llx+%u not written by the GPU yet\n",
                            (unsigned long long)device_addr, size);
            }
        }
        return false;
    }
    // bbport: the lookups above only read what the texture binding helper leaves alone; the
    // copy below changes image state, so the helper finishes first.
    runtime.BeforeImageAccess();
    Image& image = texture_cache.GetImage(image_id);
    ASSERT_MSG(device_addr == image.info.guest_address,
               "Texel buffer aliases image subresources {:x} : {:x}", device_addr,
               image.info.guest_address);
    boost::container::small_vector<vk::BufferImageCopy, 8> buffer_copies;
    for (u32 mip = 0; mip < image.info.resources.levels; mip++) {
        const auto& mip_info = image.info.mips_layout[mip];
        const u32 width = std::max(image.info.size.width >> mip, 1u);
        const u32 height = std::max(image.info.size.height >> mip, 1u);
        const u32 depth = std::max(image.info.size.depth >> mip, 1u);
        if (arena_offset + mip_info.offset + mip_info.size > arena->size_bytes) {
            break;
        }
        buffer_copies.push_back(vk::BufferImageCopy{
            .bufferOffset = mip_info.offset,
            .bufferRowLength = mip_info.pitch,
            .bufferImageHeight = mip_info.height,
            .imageSubresource{
                .aspectMask = image.aspect_mask & ~vk::ImageAspectFlagBits::eStencil,
                .mipLevel = mip,
                .baseArrayLayer = 0,
                .layerCount = image.info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {width, height, depth},
        });
    }
    if (buffer_copies.empty()) {
        return false;
    }
    // bbport BB_TILE_BACK_LOG=1 (diagnostics): each image written back into the game's memory
    // because a texel buffer reads it, with the age of its last GPU write in submissions.
    static const bool tile_back_log = bb_env("BB_TILE_BACK_LOG") != nullptr;
    if (tile_back_log) {
        static std::atomic<u32> printed{0};
        if (printed.fetch_add(1) < 400) {
            std::printf("Tile back: image %#llx %ux%u fmt %u tile %u size %llu, read %#llx+%u; last "
                        "GPU write %lld submissions ago, flags %#x\n",
                        (unsigned long long)image.info.guest_address, image.info.size.width,
                        image.info.size.height, u32(image.info.pixel_format),
                        u32(image.info.tile_mode), (unsigned long long)image.info.guest_size,
                        (unsigned long long)device_addr, size,
                        image.gpu_write_tick ? (long long)(scheduler.CurrentTick() -
                                                           image.gpu_write_tick)
                                             : -1ll,
                        u32(image.flags));
        }
    }
    auto& tile_manager = texture_cache.GetTileManager();
    // bbport: tiled in VRAM on its way to host memory (BB_LAYER_MEMORY): the shader reads that
    // copy, not the game's memory over the bus (uncached for the GPU on NVIDIA). Measured small
    // on its own (3.6 -> 3.5 ms a frame of copy dispatches with BB_GUEST_UNCACHED on an RX 7800
    // XT: their writes cost, see IsComputeImageCopy). The bytes past the image come from where
    // they are.
    // BB_TILE_BIND_VRAM=0: read in place.
    static const bool bind_vram = !bb_env_off("BB_TILE_BIND_VRAM");
    const bool want_vram = vram_copy && bind_vram && device_addr == image.info.guest_address;
    const auto tiled = tile_manager.TileImage(image, buffer_copies, arena, arena_offset,
                                              want_vram ? size : 0);
    if (want_vram && tiled.first) {
        // A binding may end at the end of its chunk's buffer (LayerInPlace).
        const u64 end = std::min<u64>(size, arena->SizeBytes() - arena_offset);
        if (end > image.info.guest_size) {
            const vk::BufferCopy tail{.srcOffset = arena_offset + image.info.guest_size,
                                      .dstOffset = tiled.second + image.info.guest_size,
                                      .size = end - image.info.guest_size};
            runtime.CopyBuffer(arena, tiled.first, {&tail, 1});
        }
        *vram_copy = tiled;
    }
    DropShadows(device_addr, size);
    // A GPU write like any other: what the GPU copies there each frame (a render target read as a
    // texel buffer) goes to VRAM instead of crossing PCIe twice (written there, read back).
    NoteGpuWrite(device_addr, size);
    return true;
}

void BufferCache::SubmitPendingArenaBinds(Vulkan::SubmitInfo& info) {
    // The submission going out has the tick before the current one.
    const u64 submitted = scheduler.CurrentTick() - 1;
    if (pending_binds.empty()) {
        QueueCopyBacks(submitted);
        QueuePromotions(submitted);
        return;
    }

    // bbport: blocks moved in place wait for the submission that may still
    // upload into their VRAM copy (ProcessDemotions).
    const u64 wait_tick = bind_wait_tick;
    const auto wait_sema = scheduler.GetWorkSemaphore()->Handle();
    bind_wait_tick = 0;
    const u64 signal_tick = memory_semaphore.NextTick();
    const auto signal_sema = memory_semaphore.Handle();
    info.AddWait(signal_sema, signal_tick, "the memory binds before it");

    // The binds go into the queue right before the submission (BB_ASYNC_SUBMIT: later, from a
    // recording thread), after the submissions still on their way there.
    auto binds = std::make_shared<std::vector<ArenaBinds>>(std::move(pending_binds));
    pending_binds.clear();
    info.before_submit.push_back([this, binds, wait_tick, wait_sema, signal_tick, signal_sema] {
        std::vector<vk::SparseBufferMemoryBindInfo> buffer_binds;
        buffer_binds.reserve(binds->size());
        u64 ranges = 0;
        for (const auto& arena_binds : *binds) {
            ranges += arena_binds.binds.size();
            buffer_binds.emplace_back(vk::SparseBufferMemoryBindInfo{
                .buffer = arena_binds.arena->Handle(),
                .bindCount = static_cast<u32>(arena_binds.binds.size()),
                .pBinds = arena_binds.binds.data(),
            });
        }
        const vk::TimelineSemaphoreSubmitInfo timeline_si = {
            .waitSemaphoreValueCount = wait_tick ? 1u : 0u,
            .pWaitSemaphoreValues = wait_tick ? &wait_tick : nullptr,
            .signalSemaphoreValueCount = 1u,
            .pSignalSemaphoreValues = &signal_tick,
        };
        const vk::BindSparseInfo sparse_info = {
            .pNext = &timeline_si,
            .waitSemaphoreCount = wait_tick ? 1u : 0u,
            .pWaitSemaphores = wait_tick ? &wait_sema : nullptr,
            .bufferBindCount = static_cast<u32>(buffer_binds.size()),
            .pBufferBinds = buffer_binds.data(),
            .signalSemaphoreCount = 1u,
            .pSignalSemaphores = &signal_sema,
        };
        const auto bind_start = std::chrono::steady_clock::now();
        auto submit_result = instance.GetGraphicsQueue().bindSparse(sparse_info);
        const double bind_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - bind_start)
                                   .count();
        arena_bind_calls.fetch_add(1, std::memory_order_relaxed);
        arena_bind_ranges.fetch_add(ranges, std::memory_order_relaxed);
        arena_bind_signaled.store(signal_tick, std::memory_order_relaxed);
        if (bind_ms >= 50.0) {
            std::printf("Guest memory: vkQueueBindSparse of %llu ranges took %.0f ms in the driver\n",
                        (unsigned long long)ranges, bind_ms);
        }
        if (submit_result == vk::Result::eErrorDeviceLost) {
            Vulkan::Breadcrumbs::ReportDeviceLost("submit");
        }
        ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");
    });
    QueueCopyBacks(submitted); // bound with the next submission
    QueuePromotions(submitted);
}

void BufferCache::ReportArenaBinds() {
    const u64 signaled = arena_bind_signaled.load(std::memory_order_relaxed);
    const auto [result, done] =
        instance.GetDevice().getSemaphoreCounterValue(memory_semaphore.Handle());
    std::printf("Arena binds: %llu vkQueueBindSparse calls (%llu ranges); the last one signals %llu, "
                "the GPU is at %llu%s\n",
                (unsigned long long)arena_bind_calls.load(), (unsigned long long)arena_bind_ranges.load(),
                (unsigned long long)signaled,
                result == vk::Result::eSuccess ? (unsigned long long)done : 0ull,
                result == vk::Result::eSuccess && done < signaled
                    ? ": the submissions after it wait for a bind not done yet"
                    : "");
}

} // namespace VideoCore
