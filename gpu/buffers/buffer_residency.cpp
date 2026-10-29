// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport: BufferCache, where the game's memory lives for the GPU: VRAM residency (allocation,
// use, idle eviction, demotion), binding in place, and the layer's in-place lookups. Split out of
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

namespace {
/// bbport: BB_RESIDENCY_CHUNKS=0: each request gets device memory of its own at offset 0, as in
/// 0.2, and no slot is reused (diagnostics: sparse binds into shared chunks on drivers that may
/// mishandle them).
bool ResidencyChunks() {
    static const bool on = [] {
        if (bb_env_off("BB_RESIDENCY_CHUNKS")) {
            std::printf("Buffer cache: residency without 64 MiB chunks (BB_RESIDENCY_CHUNKS=0)\n");
            return false;
        }
        return true;
    }();
    return on;
}
} // namespace

void BufferCache::EnsureResident(const Buffer* arena, u64 first_block, u64 last_block,
                                 bool in_place) {
    BB_SECTION(EnsureResident);
    NoteUse(first_block << block_shift, (last_block - first_block + 1) << block_shift);
    if (maintained_epoch != packet_epoch) {
        Maintain();
    }
    // bbport: the blocks were all resident the last time and nothing changed since: no interval
    // lookup (it ran for every buffer binding).
    thread_local std::array<ResidencyMemo, 256> resident_memo{};
    auto& memo = resident_memo[first_block & (resident_memo.size() - 1)];
    if (memo.first == first_block && memo.last == last_block &&
        memo.generation == ResidencyGeneration()) {
        return;
    }
    IntervalList bind_ranges;
    resident_ranges.ForEachGap(first_block, last_block + 1,
                               [&](u64 start, u64 end) { bind_ranges.Add({start, end}); });
    if (bind_ranges.Empty()) {
        memo = {first_block, last_block, ResidencyGeneration()};
        return;
    }
    BbStats::Timer timer{BbStats::t_resident};

    // bbport: blocks of the game's direct memory are bound to the memory it lives in; the rest
    // gets VRAM.
    std::vector<ResidentBind> binds;
    {
        IntervalList rest;
        for (const auto& range : bind_ranges) {
            BindInPlace(range.start, range.end, binds, rest, in_place);
        }
        bind_ranges = std::move(rest);
    }
    u64 vram_blocks = 0;
    for (const auto& range : bind_ranges) {
        vram_blocks += range.end - range.start;
    }

    if (vram_blocks != 0) {
        // bbport: the slots of blocks that left VRAM first, block by block. A request of several
        // blocks only ever took new space, and the free slots piled up while new 64 MiB chunks
        // were allocated.
        const auto into_slots = [&](const auto& ranges) {
            std::remove_cvref_t<decltype(ranges)> rest;
            for (const auto& range : ranges) {
                u64 block = range.start;
                for (; block < range.end && free_slot_count != 0; ++block) {
                    const auto [memory, memory_offset] = AllocateResidency(block_size);
                    auto* last = binds.empty() ? nullptr : &binds.back();
                    if (last && last->end == block && last->memory == memory &&
                        last->offset + ((block - last->start) << block_shift) == memory_offset) {
                        last->end = block + 1;
                    } else {
                        binds.push_back({block, block + 1, memory, memory_offset});
                    }
                    --vram_blocks;
                }
                if (block < range.end) {
                    rest.Add({block, range.end});
                }
            }
            return rest;
        };
        auto rest_ranges = into_slots(bind_ranges);
        if (vram_blocks != 0) {
            auto [memory, memory_offset] = AllocateResidency(vram_blocks << block_shift);
            for (const auto& range : rest_ranges) {
                binds.push_back({range.start, range.end, memory, memory_offset});
                memory_offset += (range.end - range.start) << block_shift;
            }
        }
    }

    u64 total_blocks = 0;
    for (const auto& bind : binds) {
        total_blocks += bind.end - bind.start;
    }
    boost::container::small_vector<vk::BufferCopy, 8> copies;
    const auto staging =
        staging_pool.Request(total_blocks * sizeof(vk::DeviceAddress), MemoryType::HostUncached);
    ArenaBinds* arena_binds = BindsForArena(arena);
    auto* bda_addrs = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    u64 offset = staging.offset;
    for (const auto& bind : binds) {
        Backing backing;
        backing.start = bind.start;
        backing.end = bind.end;
        backing.memory = bind.memory;
        backing.offset = bind.offset >> block_shift;
        resident_ranges.Add(backing);

        const auto& sparse = arena_binds->binds.emplace_back(vk::SparseMemoryBind{
            .resourceOffset = (bind.start << block_shift) - arena->cpu_addr,
            .size = (bind.end - bind.start) << block_shift,
            .memory = bind.memory,
            .memoryOffset = bind.offset,
        });
        for (u64 block = 0; block < sparse.size; block += block_size) {
            *(bda_addrs++) = arena->BufferDeviceAddress() + sparse.resourceOffset + block;
        }
        const u64 copy_size = (bind.end - bind.start) * sizeof(vk::DeviceAddress);
        // Only inside the page table's window (Shader::PageTableBase): the arena holds no other.
        // bbport: through Shader::PageTableOffset (two halves on macOS).
        if (Shader::InPageTableWindow(bind.start << block_shift,
                                      (bind.end - bind.start) << block_shift)) {
            copies.emplace_back(offset,
                                (Shader::PageTableOffset(bind.start << block_shift) >> block_shift) *
                                    sizeof(vk::DeviceAddress),
                                copy_size);
        }
        offset += copy_size;
    }

    staging.Flush();
    // With the layer's memory module the page table is the module's (LayerWriteEntries): arena
    // blocks written into it (bindings not whole in one import of lazy guest memory) sent paged
    // reads to VRAM copies, stale ones, and once the arena let the blocks go, to unbound memory
    // (GPUVM faults, a GPU hang in a full-screen pass, 2026-10-23).
    if (!LayerMode()) {
        runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
    }
}

std::pair<vk::DeviceMemory, u64> BufferCache::AllocateResidency(u64 bytes) {
    if (bytes == block_size && free_slot_count != 0) {
        // bbport: a slot of the fullest chunk: the emptier ones drain and go back to the driver.
        auto best = residency_chunks.end();
        for (auto it = residency_chunks.begin(); it != residency_chunks.end(); ++it) {
            if (!it->second.free_offsets.empty() &&
                (best == residency_chunks.end() || it->second.used > best->second.used)) {
                best = it;
            }
        }
        if (best != residency_chunks.end()) {
            const u64 offset = best->second.free_offsets.back();
            best->second.free_offsets.pop_back();
            best->second.used += block_size;
            --free_slot_count;
            NoteResidencyUnused();
            return {vk::DeviceMemory{best->first}, offset};
        }
    }
    // bbport: residency comes out of 64 MiB blocks (a larger request gets its own): a streaming
    // burst bound hundreds of small ranges, each its own vkAllocateMemory (~8 ms a burst). At most
    // one block is partly unused. The next block is allocated in the background once half of the
    // current one is used: the kernel clears new memory, on a Steam Deck with the CPU (16-22 ms a
    // burst). At most one spare block.
    constexpr u64 ResidencyBlock = 64_MB;
    // bbport: VRAM full (a 4 GB card: "ErrorOutOfDeviceMemory", the game stopped): the block
    // comes from system memory the GPU reads over the bus, as a PC game's data spills there.
    // Slower, not fatal. BB_RESIDENCY_VRAM_LIMIT_MB=N (tests): VRAM "full" after N MiB of blocks.
    const auto allocate = [this](u64 size) {
        static const u64 test_limit = [] {
            const char* env = bb_env("BB_RESIDENCY_VRAM_LIMIT_MB");
            return env ? std::strtoull(env, nullptr, 10) << 20 : 0;
        }();
        static std::atomic<u64> vram_bytes{0};
        const auto device = instance.GetDevice();
        vk::Result result = vk::Result::eErrorOutOfDeviceMemory;
        vk::DeviceMemory memory{};
        // FixedArena: no VRAM in the arena (system memory, as the guest chunks it is bound with).
        // Out of memory here too returns none (AllocateResidency waits and tries again):
        // Vulkan::Check stopped the game (#173: "Check: ErrorOutOfDeviceMemory"; PR #206).
        if (FixedArena() && arena_fallback_type_index) {
            auto host = device.allocateMemory(
                {.allocationSize = size, .memoryTypeIndex = *arena_fallback_type_index});
            if (host.result != vk::Result::eSuccess) {
                return vk::DeviceMemory{};
            }
            memory = host.value;
            BbStats::device_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
            BbStats::residency_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
            return memory;
        }
        if (!test_limit || vram_bytes.load(std::memory_order_relaxed) + size <= test_limit) {
            auto vram = device.allocateMemory({.allocationSize = size,
                                               .memoryTypeIndex = arena_memory_type_index});
            result = vram.result;
            memory = vram.value;
        }
        if (result == vk::Result::eSuccess) {
            vram_bytes.fetch_add(size, std::memory_order_relaxed);
        } else if ((result == vk::Result::eErrorOutOfDeviceMemory ||
                    result == vk::Result::eErrorOutOfHostMemory) &&
                   arena_fallback_type_index) {
            static std::atomic<bool> reported{false};
            if (!reported.exchange(true)) {
                std::printf("Guest memory: VRAM is full (%s): further copies of the game's memory "
                            "go to system memory (slower)\n",
                            vk::to_string(result).c_str());
            }
            auto host = device.allocateMemory(
                {.allocationSize = size, .memoryTypeIndex = *arena_fallback_type_index});
            if (host.result != vk::Result::eSuccess) {
                return vk::DeviceMemory{};
            }
            memory = host.value;
        } else {
            // No system-memory type for the arena (NVIDIA's sparse buffers take VRAM only):
            // AllocateResidency tries a smaller block and the other memory types.
            return vk::DeviceMemory{};
        }
        BbStats::device_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
        BbStats::residency_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
        return memory;
    };
    if (!ResidencyChunks()) {
        const auto memory = allocate(bytes);
        ASSERT_MSG(memory, "Buffer cache: no device memory for {} bytes (BB_RESIDENCY_CHUNKS=0)",
                   bytes);
        residency_chunks[static_cast<VkDeviceMemory>(memory)] = {bytes, bytes, 0, 0};
        NoteResidencyUnused();
        return {memory, 0};
    }
    if (residency_used + bytes > residency_size) {
        u64 size = std::max(bytes, ResidencyBlock);
        vk::DeviceMemory memory =
            size == ResidencyBlock && spare_residency.valid() ? spare_residency.get() : vk::DeviceMemory{};
        // bbport: VRAM full and no system memory to spill to (NVIDIA's sparse buffers take VRAM
        // only; RTX 3050 4 GB: "Failed to allocate 64 MiB for guest memory copies", the game
        // stopped): the texture collector evicts by submissions again for a while, and this
        // request gets just its size, then any other memory type the arena can be bound to.
        if (!memory) {
            memory = allocate(size);
        }
        if (!memory) {
            TextureCache::NoteVramShort();
            static std::atomic<u32> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::printf("Guest memory: VRAM is full (%llu MiB wanted): textures go sooner\n",
                            (unsigned long long)(size >> 20));
            }
            size = bytes;
            memory = allocate(size);
        }
        // Just the request's size, then any other memory type the arena can be bound to.
        const auto other_types = [&] {
            const auto& types = instance.GetMemoryProperties();
            vk::DeviceMemory any{};
            for (u32 i = 0; !any && i < types.memoryTypeCount; ++i) {
                if (((arena_memory_type_bits >> i) & 1) && i != arena_memory_type_index) {
                    auto other = instance.GetDevice().allocateMemory({.allocationSize = size, .memoryTypeIndex = i});
                    if (other.result == vk::Result::eSuccess) {
                        any = other.value;
                        BbStats::device_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
                        BbStats::residency_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
                    }
                }
            }
            return any;
        };
        if (!memory) {
            memory = other_types();
        }
        if (!memory) {
            // Still full (#163, #173: 4 GB cards on area loads; PR #206): the images the
            // collector evicted are destroyed only once the GPU is past them (DeleteImage), so
            // their VRAM was still taken. Wait for the submitted work (not the one being
            // recorded), run those deletions, try again.
            scheduler.WaitSubmittedWork();
            scheduler.PopPendingOperations();
            memory = allocate(size);
            if (!memory) {
                memory = other_types();
            }
            static std::atomic<u32> waited{0};
            if (memory && waited.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::printf("Guest memory: VRAM is full: waited for the GPU to free evicted "
                            "textures (%llu MiB)\n",
                            (unsigned long long)(size >> 20));
            }
        }
        ASSERT_MSG(memory, "Failed to allocate {} MiB for guest memory copies: VRAM is full", size >> 20);
        residency_memory = memory;
        residency_size = size;
        residency_used = 0;
        residency_chunks[static_cast<VkDeviceMemory>(residency_memory)] = {size, 0, 0, 0};
    }
    const u64 offset = residency_used;
    residency_used += bytes;
    residency_chunks[static_cast<VkDeviceMemory>(residency_memory)].used += bytes;
    if (!spare_residency.valid() && residency_used * 2 >= residency_size) {
        spare_residency = std::async(std::launch::async, allocate, ResidencyBlock);
    }
    NoteResidencyUnused();
    return {residency_memory, offset};
}

void BufferCache::NoteResidencyUnused() {
    const u64 unused = free_slot_count * block_size + (residency_size - residency_used) +
                       (spare_residency.valid() ? 64_MB : 0);
    BbStats::residency_unused_bytes.store(unused, std::memory_order_relaxed);
    BbStats::residency_chunk_count.store(residency_chunks.size(), std::memory_order_relaxed);
}

// VRAM copies of data not read for that long go back in place (mirror compaction frees them). 30 s
// was tried on a 2 GB card: 1.8 GB went and came back over the bus within minutes.
u32 BufferCache::VramIdleSeconds() {
    static const u32 seconds = [] {
        const char* env = bb_env("BB_VRAM_IDLE_SECONDS");
        return env ? u32(std::strtoul(env, nullptr, 10)) : 60u;
    }();
    return seconds;
}

void BufferCache::NoteResidencyMix(u64 first_block, u64 last_block, u64 size, bool is_written) {
    // Per class (0 in place, 1 VRAM, 2 mixed): bindings and bytes; mixed by read/write too.
    static std::array<std::atomic<u64>, 3> count{}, bytes{};
    static std::atomic<u64> mixed_written{0}, mixed_gpu_data{0};
    static std::atomic<u32> printed_second{0};
    u64 in_place = 0;
    in_place_blocks.ForEachInRange(first_block, last_block + 1, [&](const Interval& iv) {
        in_place += std::min(iv.end, last_block + 1) - std::max(iv.start, first_block);
    });
    const u64 blocks = last_block - first_block + 1;
    const int cls = in_place == blocks ? 0 : in_place == 0 ? 1 : 2;
    count[cls].fetch_add(1, std::memory_order_relaxed);
    bytes[cls].fetch_add(size, std::memory_order_relaxed);
    if (cls == 2) {
        if (is_written) {
            mixed_written.fetch_add(1, std::memory_order_relaxed);
        }
        if (gpu_modified_ranges.Intersects(first_block << block_shift, blocks << block_shift)) {
            mixed_gpu_data.fetch_add(1, std::memory_order_relaxed);
        }
    }
    // Segment sizes for VRAM mirrors aligned to guest addresses (MEMORY_MODULE_PLAN): bindings
    // all in VRAM (and under 64 MiB) that cross a segment of 2^k MiB.
    constexpr std::array<u32, 6> SegmentMiB = {2, 4, 8, 16, 32, 64};
    static std::array<std::atomic<u64>, 6> crossing{};
    static std::atomic<u64> vram_small{0};
    const VAddr address = first_block << block_shift;
    if (cls == 1 && size < 64_MB) {
        vram_small.fetch_add(1, std::memory_order_relaxed);
        for (size_t i = 0; i < SegmentMiB.size(); ++i) {
            const u64 seg = u64(SegmentMiB[i]) << 20;
            if (address / seg != (address + size - 1) / seg) {
                crossing[i].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed_second.load(std::memory_order_relaxed) >= 2) {
        printed_second.store(now, std::memory_order_relaxed);
        const auto take = [](std::atomic<u64>& v) { return (unsigned long long)v.exchange(0); };
        std::string cross_text;
        for (size_t i = 0; i < SegmentMiB.size(); ++i) {
            cross_text += fmt::format(" {}M:{}", SegmentMiB[i], take(crossing[i]));
        }
        // Distinct segments holding VRAM blocks now (VRAM a mirror of segments would take).
        std::string seg_text;
        for (const u32 mib : SegmentMiB) {
            const u64 seg_blocks = (u64(mib) << 20) >> block_shift;
            u64 segments = 0, last = ~0ULL;
            for (const auto& range : resident_ranges) {
                if (!residency_chunks.contains(static_cast<VkDeviceMemory>(range.memory))) {
                    continue;
                }
                for (u64 s = range.start / seg_blocks; s <= (range.end - 1) / seg_blocks; ++s) {
                    segments += s != last;
                    last = s;
                }
            }
            seg_text += fmt::format(" {}M:{}MiB", mib, segments * mib);
        }
        std::printf("Residency mix: VRAM bindings under 64 MiB %llu, crossing a segment of%s; "
                    "VRAM as segments of%s\n",
                    take(vram_small), cross_text.c_str(), seg_text.c_str());
        const auto c0 = take(count[0]), c1 = take(count[1]), c2 = take(count[2]);
        const auto b0 = take(bytes[0]) >> 20, b1 = take(bytes[1]) >> 20, b2 = take(bytes[2]) >> 20;
        std::printf("Residency mix (2 s): in place %llu bindings %llu MiB, VRAM %llu / %llu MiB, "
                    "mixed %llu / %llu MiB (%llu written, %llu over GPU-written data)\n",
                    c0, b0, c1, b1, c2, b2, take(mixed_written), take(mixed_gpu_data));
    }
}

void BufferCache::NoteUse(VAddr address, u64 size) {
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    // The same range noted this second already: its groups are marked (a group demoted since
    // is looked at again next second). Paged bindings span nearly all memory: thousands of
    // groups each, for every draw.
    struct Noted {
        VAddr address = 0;
        u64 size = 0;
        u32 second = ~0u;
    };
    thread_local std::array<Noted, 256> noted{};
    auto& memo = noted[((address >> 12) ^ (size * 0x9E3779B97F4A7C15ull >> 40)) & 255];
    if (memo.address == address && memo.size == size && memo.second == now) {
        return;
    }
    memo = {address, size, now};
    const u64 last = std::min<u64>((address + size - 1) >> USE_GROUP_BITS, group_use.size() - 1);
    for (u64 group = address >> USE_GROUP_BITS; group <= last; ++group) {
        if (group_use[group] != now) {
            group_use[group] = now;
        }
        if (group_demoted[group]) [[unlikely]] {
            group_demoted[group] = 0;
            const u64 first = (group << USE_GROUP_BITS) >> block_shift;
            const u64 end = ((group + 1) << USE_GROUP_BITS) >> block_shift;
            idle_demoted.ForEachInRange(first, end, [&](const Interval& range) {
                promote_candidates.Add({std::max(first, range.start), std::min(end, range.end)});
            });
            idle_demoted.Subtract(first, end);
        }
    }
}

void BufferCache::ReleaseResidencySlot(vk::DeviceMemory memory, u64 offset) {
    if (const auto it = residency_chunks.find(static_cast<VkDeviceMemory>(memory));
        it != residency_chunks.end()) {
        if (ResidencyChunks()) {
            it->second.free_offsets.push_back(offset);
            ++free_slot_count;
        }
        it->second.used -= block_size;
        if (it->second.used == 0) {
            it->second.empty_second = BbStats::coarse_second.load(std::memory_order_relaxed);
            it->second.empty_tick = scheduler.CurrentTick();
        }
    }
    NoteResidencyUnused();
}

void BufferCache::EvictVramBlock(u64 block) {
    const auto old = resident_ranges.Find(block);
    if (old == resident_ranges.end() || in_place_blocks.Contains(block) ||
        !residency_chunks.contains(static_cast<VkDeviceMemory>(old->memory))) {
        return;
    }
    const vk::DeviceMemory memory = old->memory;
    const u64 offset = (old->offset + (block - old->start)) << block_shift;
    const VAddr address = block << block_shift;
    for (const auto& arena : arenas) {
        if (arena.cpu_addr <= address && address - arena.cpu_addr < arena.size_bytes) {
            // One unbind for neighbouring blocks: NVIDIA's vkQueueBindSparse is slow per range
            // (PR #206).
            auto& binds = BindsForArena(&arena)->binds;
            const u64 offset_in_arena = address - arena.cpu_addr;
            if (!binds.empty() && !binds.back().memory &&
                binds.back().resourceOffset + binds.back().size == offset_in_arena) {
                binds.back().size += block_size;
                continue;
            }
            binds.emplace_back(vk::SparseMemoryBind{
                .resourceOffset = offset_in_arena,
                .size = block_size,
                .memory = {},
                .memoryOffset = 0,
            });
        }
    }
    resident_ranges.Subtract(block, block + 1);
    memory_tracker->UnmarkRegionAsGpuModified(address, block_size);
    gpu_modified_ranges.Subtract(address, block_size);
    memory_tracker->MarkRegionAsCpuModified(address, block_size);
    ReleaseResidencySlot(memory, offset);
    // After the work submitted so far: it may still read the block.
    bind_wait_tick = std::max(bind_wait_tick, scheduler.CurrentTick() - 1);
}

void BufferCache::ProcessIdleBlocks() {
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now == idle_scan_second || VramIdleSeconds() == 0) {
        return;
    }
    idle_scan_second = now;
    if (LayerMode()) {
        LayerMaintain();
        LayerProcessIdle();
        return;
    }
    // Blocks in VRAM whose 2 MiB were not bound for the idle time go back in place
    // (ProcessDemotions: only blocks of the game's direct memory without GPU-written data).
    const u64 tick = scheduler.CurrentTick();
    for (const auto& range : resident_ranges) {
        if (!residency_chunks.contains(static_cast<VkDeviceMemory>(range.memory))) {
            continue; // in place
        }
        for (u64 block = range.start; block < range.end; ++block) {
            if (now - group_use[(block << block_shift) >> USE_GROUP_BITS] >= VramIdleSeconds()) {
                demotions_waiting.push_back({block, tick, false, true});
            }
        }
    }
    // Over 128 MiB of free slots: the blocks of the emptiest chunk go back in place too, idle or
    // not, when the free slots of the others can take them; the ones bound again return to VRAM
    // into fuller chunks (AllocateResidency), and the chunk empties. Free slots spread evenly
    // (half of 1.3 GiB) left no chunk empty. One chunk every 10 s, each once a minute at most
    // (blocks with GPU-written data stay).
    if (free_slot_count * block_size >= 128_MB && now - last_evacuation >= 10) {
        auto sparsest = residency_chunks.end();
        for (auto it = residency_chunks.begin(); it != residency_chunks.end(); ++it) {
            const ResidencyChunk& chunk = it->second;
            if (it->first == static_cast<VkDeviceMemory>(residency_memory) || chunk.used == 0 ||
                now - chunk.evacuated_second < 60) {
                continue;
            }
            if (sparsest == residency_chunks.end() || chunk.used < sparsest->second.used) {
                sparsest = it;
            }
        }
        const u64 free_elsewhere =
            sparsest == residency_chunks.end()
                ? 0
                : (free_slot_count - sparsest->second.free_offsets.size()) * block_size;
        if (sparsest != residency_chunks.end() && sparsest->second.used <= free_elsewhere) {
            sparsest->second.evacuated_second = now;
            last_evacuation = now;
            BbStats::evacuations.fetch_add(1, std::memory_order_relaxed);
            for (const auto& range : resident_ranges) {
                if (static_cast<VkDeviceMemory>(range.memory) != sparsest->first) {
                    continue;
                }
                for (u64 block = range.start; block < range.end; ++block) {
                    demotions_waiting.push_back({block, tick, false, true, true});
                }
            }
        }
    }
    FreeEmptyChunks(now);
}

void BufferCache::FreeEmptyChunks(u32 now) {
    // Chunks without blocks for 5 s, the rebinds away from them long done: back to the driver.
    for (auto it = residency_chunks.begin(); it != residency_chunks.end();) {
        const ResidencyChunk& chunk = it->second;
        const VkDeviceMemory memory = it->first;
        if (chunk.used != 0 || memory == static_cast<VkDeviceMemory>(residency_memory) ||
            now - chunk.empty_second < 5 || !pending_binds.empty() ||
            !scheduler.GetWorkSemaphore()->IsFree(chunk.empty_tick + 1)) {
            ++it;
            continue;
        }
        free_slot_count -= chunk.free_offsets.size();
        instance.GetDevice().freeMemory(vk::DeviceMemory{memory});
        BbStats::residency_alloc_bytes.fetch_sub(chunk.size, std::memory_order_relaxed);
        BbStats::device_free_bytes.fetch_add(chunk.size, std::memory_order_relaxed);
        BbStats::vram_chunks_freed_bytes.fetch_add(chunk.size, std::memory_order_relaxed);
        it = residency_chunks.erase(it);
        NoteResidencyUnused();
    }
}

bool BufferCache::VramEligible(u64 block) {
    if (!GarlicInVram() || dynamic_blocks.Contains(block)) {
        return false;
    }
    // Diagnostics: BB_GARLIC_VRAM_RANGE=begin,end keeps VRAM copies to guest addresses in
    // [begin, end) (finding the block whose copy goes stale by halving the range).
    static const std::pair<u64, u64> vram_range = [] {
        const char* env = bb_env("BB_GARLIC_VRAM_RANGE");
        char* end = nullptr;
        const u64 begin = env ? std::strtoull(env, &end, 0) : 0;
        return std::pair<u64, u64>{begin, end && *end == ',' ? std::strtoull(end + 1, nullptr, 0) : ~0ull};
    }();
    if ((block << block_shift) < vram_range.first || (block << block_shift) >= vram_range.second) {
        return false;
    }
    if (LayerMode()) {
        return LayerEligible(block);
    }
    // Every byte last written by a loader or by the GPU.
    bool known = true;
    asset_bytes.ForEachGap(block << block_shift, (block + 1) << block_shift,
                           [&](u64 start, u64 end) {
                               known = known && gpu_written_bytes.Contains(start, end);
                           });
    // Any memory type: what decides is that every write there is one we hear of (Onion data
    // the CPU keeps writing is written by code we do not hear of, so it is not known data).
    return known;
}

void BufferCache::BindInPlace(u64 start, u64 end, std::vector<ResidentBind>& out,
                              IntervalList<>& rest, bool any_type) {
    static std::atomic<u64> in_place_bytes{0}, vram_bytes{0}, reports{0};
    for (u64 block = start; block < end;) {
        const VAddr address = block << block_shift;
        u64 phys = 0;
        uintptr_t mapping_end = 0;
        const BbGuestMemory::Chunk* chunk = nullptr;
        u64 run = 0;
        // PS4 memory types as PC heaps: Onion (CPU-cached; what the CPU writes often: constants,
        // dynamic buffers) stays in place, like an upload heap the GPU reads over PCIe; Garlic
        // (what the GPU reads a lot and the CPU rarely writes: geometry, textures) gets a VRAM
        // copy kept current like a default heap (BB_GARLIC_VRAM=0: in place too).
        // Without write tracking only data whose writes we hear of may keep a VRAM copy.
        const bool garlic = !any_type && VramEligible(block);
        // LayerMode: what reaches the arena is not bound in place (no chunk memory in sparse).
        if (!garlic && !LayerMode() && runtime_memory_direct_phys(address, &phys, &mapping_end) &&
            (phys & (block_size - 1)) == 0 && (chunk = BbGuestMemory::Find(phys)) != nullptr) {
            // Whole blocks within the mapping, the chunk and the range.
            const u64 bytes = std::min<u64>({mapping_end - address, chunk->phys + chunk->size - phys,
                                             (end - block) << block_shift});
            run = bytes >> block_shift;
            // The run ends where data may keep a VRAM copy.
            if (!any_type && GarlicInVram()) {
                for (u64 i = 1; i < run; ++i) {
                    if (VramEligible(block + i)) {
                        run = i;
                        break;
                    }
                }
            }
        }
        if (run == 0) {
            rest.Add({block, block + 1});
            vram_bytes.fetch_add(block_size, std::memory_order_relaxed);
            ++block;
            continue;
        }
        out.push_back({block, block + run, chunk->memory, phys - chunk->phys});
        in_place_blocks.Add({block, block + run});
        in_place_bytes.fetch_add(run << block_shift, std::memory_order_relaxed);
        block += run;
    }
    if (const u64 n = reports.fetch_add(1, std::memory_order_relaxed); (n & (n + 1)) == 0) {
        std::printf("Guest memory: %llu MiB of the arena bound in place, %llu MiB in VRAM "
                    "(Garlic, not direct memory in a guest chunk, or not block-aligned)\n",
                    (unsigned long long)(in_place_bytes.load() >> 20),
                    (unsigned long long)(vram_bytes.load() >> 20));
    }
}

void BufferCache::NoteCpuWrite(VAddr address, u64 guest_rip) {
    constexpr u32 Frames = 3, Window = 60;
    // Guest code the game's profile knows to write per-frame data: what it writes is an upload
    // heap at its first write (others after Frames writes in Window frames).
    const bool dynamic_writer = Game::IsDynamicWriter(guest_rip);
    const u64 block = address >> block_shift;
    const u32 frame = static_cast<u32>(BbStats::frame_number.load(std::memory_order_relaxed));
    std::scoped_lock lk{dynamic_mutex};
    auto& [last, frames] = cpu_write_frames[block];
    if (dynamic_writer && frames < Frames) {
        frames = Frames;
        last = frame;
        demote_requests.push_back(block);
        demotes_requested.store(true, std::memory_order_release);
        return;
    }
    if (last == frame && frames != 0) {
        return;
    }
    frames = frame - last > Window ? 1 : frames + 1;
    last = frame;
    if (frames == Frames) {
        demote_requests.push_back(block);
        demotes_requested.store(true, std::memory_order_release);
    }
}

void BufferCache::ProcessDemotions() {
    // bbport: the waiting requests become ready only once their tick is submitted: rescanned
    // when a new tick began or a request came (it ran for every draw and dispatch).
    const bool requested = demotes_requested.load(std::memory_order_acquire);
    if (!requested && demotions_scanned_tick == scheduler.CurrentTick()) {
        return;
    }
    demotions_scanned_tick = scheduler.CurrentTick();
    if (requested) {
        std::scoped_lock lk{dynamic_mutex};
        for (const u64 block : demote_requests) {
            demotions_waiting.push_back({block, scheduler.CurrentTick(), true});
        }
        demote_requests.clear();
        demotes_requested.store(false, std::memory_order_release);
    }
    if (demotions_waiting.empty()) {
        return;
    }
    static std::atomic<u64> moved_bytes{0}, kept{0};
    const u64 submitted_below = scheduler.CurrentTick(); // ticks under it are submitted
    std::erase_if(demotions_waiting, [&](const Demotion& request) {
        const auto [block, tick, cpu_written, idle, force] = request;
        if (tick >= submitted_below) {
            return false; // its uploads are still being recorded
        }
        const VAddr address = block << block_shift;
        if (cpu_written) {
            dynamic_blocks.Add({block, block + 1}); // bound in place when it is next used
            if (LayerMode()) {
                layer_cpu_hot.Add({block, block + 1}); // the CPU's, not the GPU's: may be watched
            }
        }
        if (LayerMode()) {
            // Its mirror's copy goes (written by code we do not hear of, or handed out again
            // and not reloaded): in place from now on.
            if (!cpu_written && !idle && VramEligible(block)) {
                return true; // loaded again since
            }
            LayerDemote(block, block + 1, true, 3);
            return true;
        }
        if (in_place_blocks.Contains(block) || !resident_ranges.Contains(block)) {
            return true;
        }
        if (idle) {
            if (!force &&
                BbStats::coarse_second.load(std::memory_order_relaxed) -
                        group_use[address >> USE_GROUP_BITS] <
                    VramIdleSeconds()) {
                return true; // bound again since
            }
        } else if (!cpu_written && VramEligible(block)) {
            return true; // written by a loader since it was handed out again
        }
        // Data only the GPU has (in the VRAM copy) would be lost: such blocks stay.
        if (IsRegionGpuModified(address, block_size) ||
            gpu_modified_ranges.Intersects(address, block_size)) {
            kept.fetch_add(1, std::memory_order_relaxed);
            if (force) {
                BbStats::evac_kept_gpu.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
        u64 phys = 0;
        uintptr_t mapping_end = 0;
        const BbGuestMemory::Chunk* chunk = nullptr;
        if (!runtime_memory_direct_phys(address, &phys, &mapping_end) ||
            (phys & (block_size - 1)) != 0 || mapping_end - address < block_size ||
            (chunk = BbGuestMemory::Find(phys)) == nullptr) {
            if (force) {
                BbStats::evac_kept_other.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
        if (!address_space[block >> blocks_per_arena_page_shift]) {
            return true;
        }
        Backing backing;
        backing.start = block;
        backing.end = block + 1;
        backing.memory = chunk->memory;
        backing.offset = (phys - chunk->phys) >> block_shift;
        // Its VRAM slot is reused by a later move to VRAM (bound once this rebind is done).
        if (const auto old = resident_ranges.Find(block); old != resident_ranges.end()) {
            ReleaseResidencySlot(old->memory, (old->offset + (block - old->start)) << block_shift);
        }
        resident_ranges.Subtract(block, block + 1);
        resident_ranges.Add(backing);
        in_place_blocks.Add({block, block + 1});
        // Every arena over it: one replaced by a wider one (GetArena) still serves the shaders
        // whose device addresses (the BDA page table) were taken from it.
        for (const auto& arena : arenas) {
            if (arena.cpu_addr <= address && address - arena.cpu_addr < arena.size_bytes) {
                BindsForArena(&arena)->binds.emplace_back(vk::SparseMemoryBind{
                    .resourceOffset = address - arena.cpu_addr,
                    .size = block_size,
                    .memory = chunk->memory,
                    .memoryOffset = backing.offset << block_shift,
                });
            }
        }
        // Its uploads up to `tick` land in the VRAM copy first; the rebind waits for them.
        bind_wait_tick = std::max(bind_wait_tick, tick);
        // Never uploaded or protected again: writes go to the memory the GPU reads.
        memory_tracker->MarkRegionAsCpuModified(address, block_size);
        if (idle) {
            BbStats::vram_idle_bytes.fetch_add(block_size, std::memory_order_relaxed);
            idle_demoted.Add({block, block + 1});
            group_demoted[address >> USE_GROUP_BITS] = 1;
        }
        const u64 total = moved_bytes.fetch_add(block_size, std::memory_order_relaxed) + block_size;
        if (!idle && (total >> 20) != ((total - block_size) >> 20)) {
            std::printf("Guest memory: %llu MiB of CPU-written blocks moved in place from VRAM, "
                        "%llu blocks kept (GPU-written)\n",
                        (unsigned long long)(total >> 20), (unsigned long long)kept.load());
        }
        return true;
    });
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::GuestChunkSource(VAddr address, u64 size) {
    u64 phys = 0;
    uintptr_t mapping_end = 0;
    const BbGuestMemory::Chunk* chunk = nullptr;
    if (size == 0 || !runtime_memory_direct_phys(address, &phys, &mapping_end) ||
        address + size > mapping_end || (chunk = BbGuestMemory::Find(phys)) == nullptr ||
        phys + size > chunk->phys + chunk->size) {
        return std::nullopt;
    }
    auto& buffer = chunk_buffers[chunk->index % chunk_buffers.size()];
    if (!buffer) {
        buffer = std::make_unique<Buffer>(instance, chunk->size, chunk->memory,
                                          fmt::format("bbport guest memory {:#x}", chunk->phys));
    }
    return std::pair<const Buffer*, u64>{buffer.get(), phys - chunk->phys};
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::LayerInPlace(VAddr address, u64 size,
                                                                      bool clamp_huge) {
    // Where a range is in place changes only when the game unmaps memory (counted when queued,
    // UnmapInPlace); each answer took a walk of its mappings (~2% of the draw recording thread).
    // Only ranges found in place: a mapping made later at a range not mapped before counts.
    struct Memo {
        VAddr address = 0;
        u64 size = 0, unmaps = 0;
        bool clamp_huge = false;
        std::pair<const Buffer*, u64> result{};
    };
    thread_local auto memo = std::make_unique<std::array<Memo, 1024>>(); // heap: see LayerBind
    auto& m = (*memo)[((address * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full)) >> 54];
    const u64 unmaps = unmaps_queued.load(std::memory_order_acquire) + 1;
    if (m.address == address && m.size == size && m.unmaps == unmaps &&
        m.clamp_huge == clamp_huge) {
        return m.result;
    }
    const auto result = LayerInPlaceUncached(address, size, clamp_huge);
    if (result) {
        m = {address, size, unmaps, clamp_huge, *result};
    }
    return result;
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::LayerInPlaceUncached(VAddr address,
                                                                              u64 size,
                                                                              bool clamp_huge) {
    auto span = BbLayer::GpuMemory::Get().ResolvePrefix(address, size);
    if (!span) {
        return std::nullopt;
    }
    if (span->size != size) {
        // Shaders given (nearly) all memory as one buffer: the part the module has whole. Until
        // the paged path (MEMORY_MODULE_PLAN step 3) a range across chunks is not one buffer.
        if (!clamp_huge || size < 64_MB) {
            return std::nullopt;
        }
        // Lazy guest memory: the largest import from its start (its granule ended it early).
        u64 phys = 0;
        uintptr_t mapping_end = 0;
        if (runtime_memory_direct_phys(address, &phys, &mapping_end) &&
            BbGuestMemory::Widen(phys, std::min<u64>(size, mapping_end - address))) {
            if (const auto wider = BbLayer::GpuMemory::Get().ResolvePrefix(address, size);
                wider && wider->size > span->size) {
                span = wider;
            }
        }
        static std::atomic<u32> reported{0};
        if (span->size < size && reported.fetch_add(1, std::memory_order_relaxed) < 8) {
            std::printf("Layer memory: binding %#llx+%#llx cut to %#llx (the rest is in another "
                        "chunk)\n",
                        (unsigned long long)address, (unsigned long long)size,
                        (unsigned long long)span->size);
        }
    }
    const auto* chunk = static_cast<const BbGuestMemory::Chunk*>(span->source->owner);
    auto& buffer = chunk_buffers[chunk->index % chunk_buffers.size()];
    if (!buffer) {
        buffer = std::make_unique<Buffer>(instance, chunk->size, chunk->memory,
                                          fmt::format("bbport guest memory {:#x}", chunk->phys));
    }
    return std::pair<const Buffer*, u64>{buffer.get(), span->offset};
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::LayerStitch(VAddr address, u64 size) {
    // Not served by the arena: its VRAM copy of these bytes went stale where the GPU had written
    // them in place since (skinned vertices across two imports: exploding vertices, 2026-10-23).
    // Also without lazy memory: heap growth imports pieces of its own (Windows laptop, Hunter's
    // Dream railings stretched near the camera, 2026-10-25). BB_LAYER_STITCH=0: the arena as before.
    static const bool enabled = !bb_env_off("BB_LAYER_STITCH");
    if (!enabled || size == 0 || size > 16_MB) {
        return std::nullopt;
    }
    struct Piece {
        const Buffer* buffer;
        u64 offset, size;
    };
    boost::container::small_vector<Piece, 4> pieces;
    for (VAddr at = address; at < address + size;) {
        const auto span = BbLayer::GpuMemory::Get().ResolvePrefix(at, address + size - at);
        if (!span || span->size == 0 || pieces.size() == pieces.capacity()) {
            return std::nullopt;
        }
        const auto* chunk = static_cast<const BbGuestMemory::Chunk*>(span->source->owner);
        auto& buffer = chunk_buffers[chunk->index % chunk_buffers.size()];
        if (!buffer) {
            buffer = std::make_unique<Buffer>(instance, chunk->size, chunk->memory,
                                              fmt::format("bbport guest memory {:#x}", chunk->phys));
        }
        pieces.push_back({buffer.get(), span->offset, span->size});
        at += span->size;
    }
    const auto offset = stream_buffer.Reserve(size, instance.StorageMinAlignment());
    if (!offset) {
        return std::nullopt;
    }
    u64 to = *offset;
    for (const auto& piece : pieces) {
        const vk::BufferCopy copy{piece.offset, to, piece.size};
        runtime.CopyBuffer(piece.buffer, &stream_buffer, {&copy, 1});
        to += piece.size;
    }
    NoteBlocksBound(address >> block_shift, ((address + size - 1) >> block_shift) + 1);
    static std::atomic<u64> stitched{0}, bytes{0};
    static std::atomic<u32> printed_second{0};
    stitched.fetch_add(1, std::memory_order_relaxed);
    bytes.fetch_add(size, std::memory_order_relaxed);
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed_second.load(std::memory_order_relaxed) >= 10) {
        printed_second.store(now, std::memory_order_relaxed);
        std::printf("Layer memory (10 s): %llu reads across imports stitched by GPU copies, %llu MiB\n",
                    (unsigned long long)stitched.exchange(0),
                    (unsigned long long)(bytes.exchange(0) >> 20));
    }
    return std::pair<const Buffer*, u64>{&stream_buffer, *offset};
}

void BufferCache::NoteLayerMiss(VAddr address, u64 size, bool is_written) {
    // Per 2 s: bindings not whole in one chunk (served by the arena: VRAM copies).
    static std::atomic<u64> misses{0}, bytes{0}, written{0};
    static std::atomic<u32> printed_second{0};
    misses.fetch_add(1, std::memory_order_relaxed);
    bytes.fetch_add(size, std::memory_order_relaxed);
    if (is_written) {
        written.fetch_add(1, std::memory_order_relaxed);
    }
    static std::atomic<u32> examples{0};
    if (examples.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::printf("Layer memory: %#llx+%#llx%s not whole in one guest chunk\n",
                    (unsigned long long)address, (unsigned long long)size,
                    is_written ? " (written)" : "");
    }
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed_second.load(std::memory_order_relaxed) >= 2) {
        printed_second.store(now, std::memory_order_relaxed);
        std::printf("Layer memory (2 s): %llu bindings, %llu MiB not whole in one guest chunk "
                    "(%llu written)\n",
                    (unsigned long long)misses.exchange(0), (unsigned long long)(bytes.exchange(0) >> 20),
                    (unsigned long long)written.exchange(0));
    }
}

} // namespace VideoCore
