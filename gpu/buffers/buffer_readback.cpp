// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport: BufferCache, the way back to the CPU: guest reads of memory the GPU wrote (readbacks,
// downloads on the readback queue), the layer's copy-backs, and the twins of small GPU-written
// bytes. Split out of buffer_cache.cpp (2026-10-26); shared state in buffer_cache_internal.h.


#include <chrono>
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

namespace {
thread_local const char* readback_requester = "";
thread_local bool readback_is_write = false;

/// bbport: copies finished GPU data out of an arena on the readback queue (Instance), waiting
/// only for those writes (the timeline semaphore at their tick), not for the queued frames.
class FastReadback {
public:
    explicit FastReadback(const Vulkan::Instance& instance)
        : device{instance.GetDevice()}, queue{instance.GetReadbackQueue()},
          allocator{instance.GetAllocator()} {
        if (!queue) {
            return;
        }
        pool = Vulkan::Check(device.createCommandPoolUnique({
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = instance.GetReadbackQueueFamilyIndex(),
        }));
        cmdbuf = Vulkan::Check(device.allocateCommandBuffers({
            .commandPool = *pool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        }))[0];
        fence = Vulkan::Check(device.createFenceUnique({}));
    }
    ~FastReadback() {
        if (buffer) {
            vmaDestroyBuffer(allocator, buffer, allocation);
        }
    }

    [[nodiscard]] bool Available() const noexcept {
        return bool(queue);
    }

    /// Copies `copies` (destination offsets from 0) of `arena` once `semaphore` reaches `tick`;
    /// returns the host copy, or null on failure (the caller takes the regular path).
    const u8* Copy(const Buffer* arena, std::span<const vk::BufferCopy> copies, u64 total,
                   vk::Semaphore semaphore, u64 tick) {
        if (!Ensure(total)) {
            return nullptr;
        }
        Vulkan::Check(cmdbuf.reset());
        Vulkan::Check(cmdbuf.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
        cmdbuf.copyBuffer(arena->Handle(), vk::Buffer{buffer}, copies);
        const vk::MemoryBarrier2 to_host{
            .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
        };
        cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_host});
        Vulkan::Check(cmdbuf.end());
        const vk::SemaphoreSubmitInfo wait{
            .semaphore = semaphore,
            .value = tick,
            .stageMask = vk::PipelineStageFlagBits2::eTransfer,
        };
        const vk::CommandBufferSubmitInfo command{.commandBuffer = cmdbuf};
        const vk::SubmitInfo2 submit{
            .waitSemaphoreInfoCount = 1,
            .pWaitSemaphoreInfos = &wait,
            .commandBufferInfoCount = 1,
            .pCommandBufferInfos = &command,
        };
        if (queue.submit2(submit, *fence) != vk::Result::eSuccess) {
            return nullptr;
        }
        const auto result = device.waitForFences(*fence, vk::True, UINT64_MAX);
        Vulkan::Check(device.resetFences(*fence));
        if (result != vk::Result::eSuccess) {
            return nullptr;
        }
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        return mapped;
    }

private:
    bool Ensure(u64 size) {
        if (size <= capacity) {
            return true;
        }
        if (buffer) {
            vmaDestroyBuffer(allocator, buffer, allocation);
            buffer = VK_NULL_HANDLE;
            capacity = 0;
        }
        const u64 new_capacity = std::max<u64>(std::bit_ceil(size), 1_MB);
        const VkBufferCreateInfo buffer_ci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = new_capacity,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        };
        const VmaAllocationCreateInfo alloc_ci{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        };
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(allocator, &buffer_ci, &alloc_ci, &buffer, &allocation, &info) !=
            VK_SUCCESS) {
            buffer = VK_NULL_HANDLE;
            return false;
        }
        mapped = static_cast<u8*>(info.pMappedData);
        capacity = new_capacity;
        return true;
    }

    vk::Device device;
    vk::Queue queue;
    VmaAllocator allocator;
    vk::UniqueCommandPool pool;
    vk::CommandBuffer cmdbuf;
    vk::UniqueFence fence;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = nullptr;
    u8* mapped = nullptr;
    u64 capacity = 0;
};
std::mutex fast_readback_mutex;
/// One readback queue user for the old model's downloads and the module's read traps.
FastReadback& SharedReadback(const Vulkan::Instance& instance) {
    static FastReadback readback{instance};
    return readback;
}
} // namespace

void BufferCache::ReadMemory(VAddr device_addr, u64 size, bool is_write, bool assume_locks) {
    BbStats::readbacks.fetch_add(1, std::memory_order_relaxed);
    std::array<char, 16> requester{};
    if (Readbacks()) {
        pthread_getname_np(pthread_self(), requester.data(), requester.size());
    }
    const auto flush_request = [this, device_addr, size, is_write, requester] {
        readback_requester = requester.data();
        readback_is_write = is_write;
        const u32 first_block = device_addr >> block_shift;
        const u32 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);

        // GPU-modified ranges come as many small scattered islands,
        // so the download is widened to a window around the request
        constexpr u64 WindowSize = 512_KB;
        const VAddr arena_end = arena->cpu_addr + arena->size_bytes;
        const VAddr window_start =
            std::max<VAddr>(Common::AlignDown(device_addr, WindowSize), arena->cpu_addr);
        const VAddr window_end = std::min<VAddr>(
            std::max<VAddr>(window_start + WindowSize, device_addr + size), arena_end);
        // bbport (BB_GPU_WRITE_TWINS): bytes the GPU is still writing on the faulting page are
        // twinned instead of waited for (see TwinsEnabled).
        if (is_write && TwinsEnabled()) {
            const VAddr page_begin = Common::AlignDown(device_addr, BbMem::PageSize);
            const VAddr page_end = Common::AlignUp(device_addr + size, BbMem::PageSize);
            u64 tick = 0, gpu_bytes = 0;
            gpu_modified_ranges.ForEachInRange(page_begin, page_end - page_begin,
                                               [&](VAddr start, VAddr end) {
                                                   gpu_bytes += end - start;
                                                   tick = std::max(tick, NewestGpuWrite(
                                                                             start, end - start));
                                               });
            if (gpu_bytes != 0 && gpu_bytes <= TwinsMaxBytes() && tick != 0 &&
                !scheduler.IsFree(tick)) {
                {
                    std::scoped_lock lk{twins_mutex};
                    gpu_modified_ranges.ForEachInRange(
                        page_begin, page_end - page_begin, [&](VAddr start, VAddr end) {
                            // Parts with a twin keep their older snapshot: the guest bytes may
                            // have been written since it was taken.
                            ForEachTwinRun(start, end, [&](VAddr from, VAddr to, TwinRun run) {
                                if (run == TwinRun::None) {
                                    // The backing view: the page may be read-protected.
                                    std::vector<u8> bytes(to - from);
                                    Core::Memory::Instance()->ReadBacking(from, bytes.data(),
                                                                          to - from);
                                    twins[from] = std::move(bytes);
                                }
                            });
                        });
                }
                memory_tracker->MarkRegionAsCpuModified(device_addr, size);
                twin_faults.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
        // bbport (BB_GPU_WRITE_TWINS): the window around the fault is downloaded only when the GPU has
        // done its writes there; else just the faulting page (the other pages stay GPU-modified
        // and protected), instead of waiting for GPU writes that have nothing to do with it.
        VAddr download_start = window_start, download_end = window_end;
        if (is_write && TwinsEnabled()) {
            u64 window_tick = 0;
            gpu_modified_ranges.ForEachInRange(
                window_start, window_end - window_start, [&](VAddr start, VAddr end) {
                    window_tick = std::max(window_tick, NewestGpuWrite(start, end - start));
                });
            if (window_tick != 0 && !scheduler.IsFree(window_tick)) {
                download_start = Common::AlignDown(device_addr, BbMem::PageSize);
                download_end = Common::AlignUp(device_addr + size, BbMem::PageSize);
            }
        }
        DownloadMemory(arena, download_start, download_end - download_start);
        if (is_write) {
            memory_tracker->MarkRegionAsCpuModified(device_addr, size);
        }
    };
    if (assume_locks) {
        flush_request();
    } else {
        liverpool->SendCommand<true>(std::move(flush_request));
    }
}

bool BufferCache::LayerReadTrapHit(VAddr address, TrapCaller caller) {
    if (!(BbLayer::WriteTraps::Reasons(address) & BbLayer::WriteTraps::VramData)) {
        return false;
    }
    layer_read_trap_hits.fetch_add(1, std::memory_order_relaxed);
    switch (caller) {
    case TrapCaller::Guest:
        // Commands run with the draw pipe drained (RunCommands): flushes and copies are safe.
        liverpool->SendCommand<true>([this, address] { LayerCopyBackForCpu(address, true); });
        break;
    case TrapCaller::CommandThread:
        LayerCopyBackForCpu(address, true); // the caller drained the draw pipe
        break;
    case TrapCaller::RecordingThread:
        LayerCopyBackForCpu(address, false);
        break;
    }
    return true;
}

void BufferCache::LayerCopyBackForCpu(VAddr address, bool may_record) {
    if (!(BbLayer::WriteTraps::Reasons(address) & BbLayer::WriteTraps::VramData)) {
        return; // copied back meanwhile
    }
    // The GPU's data in the block's VRAM copy (one readback for the block, not for each page),
    // once the GPU has written it: on the readback queue, without waiting for the queued frames.
    const VAddr from = Common::AlignDown(address, block_size);
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    boost::container::small_vector<VAddr, 8> guests;
    const Buffer* mirror_buffer = nullptr;
    u64 total = 0, tick = 0;
    if (const auto mirror = BbLayer::GpuMemory::Get().MirrorAt(from)) {
        mirror_buffer = static_cast<const Buffer*>(mirror->owner);
        gpu_modified_ranges.ForEachInRange(from, block_size, [&](VAddr start, VAddr end) {
            copies.push_back({mirror_buffer->Offset(start), total, end - start});
            guests.push_back(start);
            total += Common::AlignUp(end - start, 64);
            tick = std::max(tick, NewestGpuWrite(start, end - start));
        });
    }
    bool copied = copies.empty();
    if (!copies.empty() && !may_record && tick != 0 && !scheduler.IsSubmitted(tick)) {
        // The recording thread itself, before its own GPU writes are submitted: they cannot be
        // waited for here. The page is read as it is in the game's memory (as without read
        // traps); the GPU's data stays marked for later copies back.
        layer_read_trap_stale.fetch_add(1, std::memory_order_relaxed);
        std::scoped_lock lk{layer_read_trap_mutex};
        BbLayer::WriteTraps::Set(from, block_size, BbLayer::WriteTraps::VramData, false);
        layer_read_trapped.Subtract(from, block_size);
        return;
    }
    if (!copies.empty()) {
        if (tick != 0 && !scheduler.IsFree(tick)) {
            if (may_record) {
                scheduler.Wait(tick); // submitted, and done
            } else {
                scheduler.GetWorkSemaphore()->Wait(tick); // submitted already: done
            }
        }
        auto& readback = SharedReadback(instance);
        if (readback.Available() && !BbToggle::Disabled(BbToggle::ReadbackQueue)) {
            std::scoped_lock lk{fast_readback_mutex};
            if (const u8* data = readback.Copy(mirror_buffer, copies, total,
                                               scheduler.GetWorkSemaphore()->Handle(), tick)) {
                for (size_t i = 0; i < copies.size(); ++i) {
                    memory->TryWriteBacking(std::bit_cast<u8*>(guests[i]),
                                            data + copies[i].dstOffset, copies[i].size);
                }
                copied = true;
            }
        }
        if (copied) {
            gpu_modified_ranges.Subtract(from, block_size);
        } else if (may_record) {
            // No readback queue: a copy in stream order, waited for.
            LayerCopyBack(from >> block_shift, (from >> block_shift) + 1);
            scheduler.Finish();
        } else {
            layer_read_trap_stale.fetch_add(1, std::memory_order_relaxed);
        }
        layer_read_trap_bytes.fetch_add(total, std::memory_order_relaxed);
    } else if (layer_copy_back_tick != 0 && !scheduler.IsFree(layer_copy_back_tick)) {
        // A copy back recorded is not done yet.
        if (may_record) {
            scheduler.Wait(layer_copy_back_tick);
        } else if (scheduler.IsSubmitted(layer_copy_back_tick)) {
            scheduler.GetWorkSemaphore()->Wait(layer_copy_back_tick);
        } else {
            layer_read_trap_stale.fetch_add(1, std::memory_order_relaxed);
        }
    }
    std::scoped_lock lk{layer_read_trap_mutex};
    BbLayer::WriteTraps::Set(from, block_size, BbLayer::WriteTraps::VramData, false);
    layer_read_trapped.Subtract(from, block_size);
}

void BufferCache::DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size) {
    boost::container::small_vector<vk::BufferCopy, 1> copies;
    u64 total_size_bytes = 0;
    const VAddr arena_base = arena->cpu_addr;
    memory_tracker->ForEachDownloadRange<false>(device_addr, size, [&](u64 address, u64 size) {
        const auto add_download = [&](VAddr start, VAddr end) {
            const u64 new_offset = start - arena_base;
            const u64 new_size = end - start;
            copies.push_back(vk::BufferCopy{
                .srcOffset = new_offset,
                .dstOffset = total_size_bytes,
                .size = new_size,
            });
            // Align up to avoid cache conflicts
            constexpr u64 align = 64ULL;
            constexpr u64 mask = ~(align - 1ULL);
            total_size_bytes += (new_size + align - 1) & mask;
        };
        gpu_modified_ranges.ForEachInRange(address, size, add_download);
        gpu_modified_ranges.Subtract(address, size);
    });
    if (total_size_bytes == 0) {
        return;
    }
    const auto write_range = [&](VAddr guest, const u8* src, u64 bytes) {
        BbFreeCheck::Check(guest, bytes, src, BbFreeCheck::BufferDownload);
        memory->TryWriteBacking(std::bit_cast<u8*>(guest), src, bytes);
    };
    const auto write_back = [&](const u8* data, u64 data_offset) {
        std::unique_lock twins_lock{twins_mutex, std::defer_lock};
        if (TwinsEnabled()) {
            twins_lock.lock();
        }
        for (const auto& copy : copies) {
            const VAddr guest = arena_base + copy.srcOffset;
            const u8* src = data + (copy.dstOffset - data_offset);
            if (!TwinsEnabled() || twins.empty()) {
                write_range(guest, src, copy.size);
                continue;
            }
            // bbport (BB_GPU_WRITE_TWINS): guest bytes written since their twin was made stay;
            // the page is CPU-modified, so its next upload carries them to the arena.
            ForEachTwinRun(guest, guest + copy.size, [&](VAddr from, VAddr to, TwinRun run) {
                if (run != TwinRun::Changed) {
                    write_range(from, src + (from - guest), to - from);
                }
            });
            EraseTwins(guest, guest + copy.size);
        }
    };
    const auto wait_start = std::chrono::steady_clock::now();
    const u64 tick_before = scheduler.CurrentTick();
    // bbport: the readback queue copies the data once the GPU has done the newest write into
    // it (waiting for that submission only when it is still queued); the graphics queue would
    // first run all queued work (~a frame: 20-30 ms stutters whenever the game wrote into
    // memory a GPU pass had written).
    u64 write_tick = 0;
    for (const auto& copy : copies) {
        write_tick = std::max(write_tick, NewestGpuWrite(arena_base + copy.srcOffset, copy.size));
    }
    const bool write_done = write_tick != 0 && scheduler.IsFree(write_tick);
    bool fast = false;
    auto& readback = SharedReadback(instance);
    if (write_tick != 0 && readback.Available() && !BbToggle::Disabled(BbToggle::ReadbackQueue)) {
        if (!write_done) {
            scheduler.Wait(write_tick);
        }
        {
            std::scoped_lock lk{fast_readback_mutex};
            if (const u8* data = readback.Copy(arena, copies, total_size_bytes,
                                               scheduler.GetWorkSemaphore()->Handle(),
                                               write_tick)) {
                write_back(data, 0);
                fast = true;
            }
        }
    }
    if (!fast) {
        const auto download =
            staging_pool.Request(total_size_bytes, VideoCore::MemoryType::HostCached);
        for (auto& copy : copies) {
            copy.dstOffset += download.offset;
        }
        runtime.CopyBuffer(arena, download.buffer, copies);
        scheduler.Finish();
        download.buffer->Invalidate(download.offset, download.size);
        write_back(download.mapped, download.offset);
    }
    if (auto* log = Readbacks()) {
        const auto now = std::chrono::steady_clock::now();
        std::scoped_lock lk{log->mutex};
        std::fprintf(log->file, "%.4f,%s,%#llx,%llu,%llu,%zu,%d,%.3f,%lld,%d,%d\n",
                     std::chrono::duration<double>(now - log->start).count(), readback_requester,
                     (unsigned long long)device_addr, (unsigned long long)(size >> 10),
                     (unsigned long long)(total_size_bytes >> 10), copies.size(),
                     readback_is_write ? 1 : 0,
                     std::chrono::duration<double, std::milli>(now - wait_start).count(),
                     write_tick ? (long long)(tick_before - write_tick) : -1LL, write_done ? 1 : 0,
                     fast ? 1 : 0);
        if (!write_done && write_tick != 0) {
            // The GPU-written bindings this readback waited for (newest first).
            const u64 n = written_head.load();
            u32 shown = 0;
            for (u64 i = n; i-- > (n > written_bindings.size() ? n - written_bindings.size() : 0) &&
                            shown < 4;) {
                const WrittenBinding b = written_bindings[i % written_bindings.size()];
                bool overlaps = false;
                u64 cpu_lo = ~0ull, cpu_hi = 0;
                for (const auto& copy : copies) {
                    const u64 a = arena_base + copy.srcOffset;
                    overlaps |= b.address < a + copy.size && a < b.address + b.size;
                    cpu_lo = std::min(cpu_lo, a);
                    cpu_hi = std::max(cpu_hi, a + copy.size);
                }
                if (overlaps) {
                    std::fprintf(log->file,
                                 "# waited for GPU write %#llx+%llu (%lld ticks before), downloaded "
                                 "%#llx..%#llx\n",
                                 (unsigned long long)b.address, (unsigned long long)b.size,
                                 (long long)(tick_before - b.tick), (unsigned long long)cpu_lo,
                                 (unsigned long long)cpu_hi);
                    ++shown;
                }
            }
        }
        std::fflush(log->file);
    }
    memory_tracker->UnmarkRegionAsGpuModified(device_addr, size);
}

void BufferCache::DropTwins(VAddr addr, u64 size) {
    if (TwinsEnabled()) {
        std::scoped_lock lk{twins_mutex};
        EraseTwins(addr, addr + size);
    }
}

} // namespace VideoCore
