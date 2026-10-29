// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport: BufferCache, once per packet and frame: promotions to VRAM, unmaps, dead images,
// the in-place report, and the VRAM access traps (diagnostics). Split out of buffer_cache.cpp
// (2026-10-26); shared state in buffer_cache_internal.h.


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
// bbport BB_VRAM_ACCESS_TRAP=1 (diagnostics): blocks moved to VRAM are closed to the CPU
// (PROT_NONE); the first CPU access to each is recorded (code site, read or write) and the block
// opened again. Finds CPU readers of data the GPU writes in VRAM, where the CPU sees a stale copy.
struct VramTrap {
    std::atomic<u64> address{0}, rip{0}, caller{0};
    std::atomic<bool> write{false};
};
std::array<VramTrap, 256> vram_traps;
std::atomic<u32> vram_trap_count{0}, vram_trap_reported{0};
std::mutex vram_trapped_mutex;
std::unordered_map<u64, u64> vram_trapped; // block address -> block size
u64 vram_trap_block_size = 0;
// BB_VRAM_TRAP_RANGE=addr,size: only blocks over that range, re-armed for every packet, every
// access counted by site.
std::pair<u64, u64> VramTrapRange() {
    static const std::pair<u64, u64> range = [] {
        const char* env = bb_env("BB_VRAM_TRAP_RANGE");
        if (!env) {
            return std::pair<u64, u64>{0, 0};
        }
        char* end = nullptr;
        const u64 addr = std::strtoull(env, &end, 0);
        return std::pair<u64, u64>{addr, end && *end == ',' ? std::strtoull(end + 1, nullptr, 0) : 0};
    }();
    return range;
}
struct SiteCount {
    std::atomic<u64> key{0}, count{0}, caller{0};
};
std::array<SiteCount, 128> vram_trap_sites;
std::vector<u64> vram_rearm;

bool VramTrapsEnabled() {
    static const bool enabled = bb_env("BB_VRAM_ACCESS_TRAP") != nullptr;
    return enabled;
}

bool VramTrapHandler(void* context, void* fault_address) {
    const u64 address = reinterpret_cast<u64>(fault_address);
    const u64 block = vram_trap_block_size ? address & ~(vram_trap_block_size - 1) : 0;
    {
        std::scoped_lock lk{vram_trapped_mutex};
        if (!vram_trapped.erase(block)) {
            return false;
        }
    }
    const auto* g = Common::Gregs(context);
    const u64 rip = u64(g[REG_RIP]);
    constexpr u64 Image = 0x800000000ull, ImageEnd = 0x810000000ull;
    u64 caller = 0;
#if defined(__APPLE__) && !defined(__x86_64__)
    // bbport (the native GPU process): the guest's stack isn't this one; no caller.
    const u64* stack = nullptr;
#else
    const auto* stack = reinterpret_cast<const u64*>(g[REG_RSP]);
#endif
    for (u32 i = 0; stack && i < 48; ++i) {
        if (stack[i] >= Image && stack[i] < ImageEnd) {
            caller = stack[i] - Image;
            break;
        }
    }
    const bool write = Common::IsWriteError(context);
    const u64 site = (rip >= Image && rip < ImageEnd ? rip - Image : rip | (1ull << 63)) ^
                     (write ? 1ull << 62 : 0);
    if (VramTrapRange().second) {
        // Counted by site; the block is closed again at the next packet.
        for (u32 i = 0, slot = u32((site * 0x9E3779B97F4A7C15ull) >> 57); i < vram_trap_sites.size();
             ++i) {
            auto& entry = vram_trap_sites[(slot + i) % vram_trap_sites.size()];
            u64 expected = 0;
            if (entry.key.load() == site || entry.key.compare_exchange_strong(expected, site)) {
                entry.count.fetch_add(1);
                entry.caller = caller;
                break;
            }
        }
        std::scoped_lock lk{vram_trapped_mutex};
        vram_rearm.push_back(block);
    } else {
        const u32 slot = vram_trap_count.fetch_add(1);
        if (slot < vram_traps.size()) {
            auto& trap = vram_traps[slot];
            trap.rip = site & ~(1ull << 62);
            trap.caller = caller;
            trap.write = write;
            trap.address = address;
        }
    }
    mprotect(reinterpret_cast<void*>(block), vram_trap_block_size, PROT_READ | PROT_WRITE);
    return true;
}

std::string VramTrapSiteName(u64 rip) {
    if (!(rip >> 63)) {
        return fmt::format("+{:#x}", rip);
    }
    Dl_info info{};
    const u64 address = rip & ~(1ull << 63);
    dladdr(reinterpret_cast<void*>(address), &info);
    const char* name = info.dli_fname ? std::strrchr(info.dli_fname, '/') : nullptr;
    return fmt::format("host {}+{:#x}", name ? name + 1 : "?",
                       address - reinterpret_cast<u64>(info.dli_fbase));
}

void ReportVramTraps() {
    if (VramTrapRange().second) {
        // Re-arm the blocks touched since the last packet.
        std::vector<u64> blocks;
        {
            std::scoped_lock lk{vram_trapped_mutex};
            blocks.swap(vram_rearm);
            for (const u64 block : blocks) {
                vram_trapped[block] = vram_trap_block_size;
            }
        }
        for (const u64 block : blocks) {
            mprotect(reinterpret_cast<void*>(block), vram_trap_block_size, PROT_NONE);
        }
        static auto last = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - last < std::chrono::seconds(3)) {
            return;
        }
        last = std::chrono::steady_clock::now();
        std::string line;
        for (auto& entry : vram_trap_sites) {
            const u64 key = entry.key.load(), count = entry.count.exchange(0);
            if (key && count) {
                line += fmt::format(" {} {} x{} (guest caller +{:#x});",
                                    key & (1ull << 62) ? "write" : "read",
                                    VramTrapSiteName(key & ~(1ull << 62)), count,
                                    entry.caller.load());
            }
        }
        std::printf("VRAM trap sites (3 s):%s\n", line.c_str());
        return;
    }
    const u32 count = std::min<u32>(vram_trap_count.load(), vram_traps.size());
    for (u32 i = vram_trap_reported.load(); i < count; ++i) {
        const auto& trap = vram_traps[i];
        std::printf("VRAM trap: CPU %s at %#llx by %s (guest caller +%#llx)\n",
                    trap.write ? "write" : "read", (unsigned long long)trap.address.load(),
                    VramTrapSiteName(trap.rip.load()).c_str(),
                    (unsigned long long)trap.caller.load());
    }
    vram_trap_reported = count;
}
} // namespace

void BufferCache::ReportInPlaceBlocks() {
    // BB_RESIDENCY_REPORT=<file>: when the file appears (checked once a second), the blocks bound
    // in place that were used in the last seconds, and why they have no VRAM copy.
    static const char* const trigger = bb_env("BB_RESIDENCY_REPORT");
    static s64 checked = -1;
    const s64 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (!trigger || now == checked) {
        return;
    }
    checked = now;
    if (access(trigger, F_OK) != 0) {
        return;
    }
    unlink(trigger);
    u64 used = 0, idle = 0, dynamic = 0, unknown = 0, eligible = 0, candidates = 0, whole = 0;
    std::map<int, u64> unknown_by_type;
    std::vector<std::pair<u64, u64>> samples; // block address, unknown bytes
    for (const auto& range : in_place_blocks) {
        for (u64 block = range.start; block < range.end; ++block) {
            const VAddr address = block << block_shift;
            const u64 group = address >> USE_GROUP_BITS;
            if (group >= group_use.size() || now - s64(group_use[group]) > 2) {
                ++idle;
                continue;
            }
            ++used;
            if (dynamic_blocks.Contains(block)) {
                ++dynamic;
                continue;
            }
            u64 missing = 0;
            asset_bytes.ForEachGap(address, address + block_size, [&](u64 start, u64 end) {
                gpu_written_bytes.ForEachGap(start, end, [&](u64 s, u64 e) { missing += e - s; });
            });
            if (missing == 0) {
                ++eligible;
                candidates += promote_candidates.Contains(block);
                continue;
            }
            ++unknown;
            whole += missing == block_size;
            int prot = 0, type = -1;
            uintptr_t vma_end = 0;
            runtime_memory_vma_info(address, &prot, &type, &vma_end);
            ++unknown_by_type[type];
            if (samples.size() < 16) {
                samples.emplace_back(address, missing);
            }
        }
    }
    const auto mib = [&](u64 blocks) { return (unsigned long long)((blocks * block_size) >> 20); };
    std::printf("Residency report: blocks of %u KiB bound in place: %llu MiB used in the last 2 s, "
                "%llu MiB idle; of the used: %llu MiB dynamic (CPU-written), %llu MiB with bytes "
                "nobody announced (%llu MiB entirely), %llu MiB eligible (%llu MiB queued)\n",
                block_size >> 10, mib(used), mib(idle), mib(dynamic), mib(unknown), mib(whole),
                mib(eligible), mib(candidates));
    for (const auto& [type, count] : unknown_by_type) {
        std::printf("Residency report: unannounced bytes, memory type %d: %llu MiB\n", type,
                    mib(count));
    }
    for (const auto& [address, missing] : samples) {
        std::printf("Residency report:   block 0x%llx: %llu KiB unannounced\n",
                    (unsigned long long)address, (unsigned long long)(missing >> 10));
    }
}

void BufferCache::QueuePromotions(u64 submitted) {
    if (VramTrapsEnabled()) {
        ReportVramTraps();
    }
    // bbport BB_LAYER_MEMORY: mirrors replaced by bigger ones go as soon as the GPU is past them,
    // not once a second (LayerMaintain): loading a level replaced hundreds within a second.
    if (LayerMode() && !layer_retired.empty()) {
        std::erase_if(layer_retired, [&](auto& retired) {
            if (!scheduler.GetWorkSemaphore()->IsFree(retired.second)) {
                return false;
            }
            LayerReleaseMirror(std::move(retired.first));
            return true;
        });
        LayerDrainMirrorFrees();
    }
    ReportInPlaceBlocks();
    if (promote_candidates.Empty()) {
        BbStats::mirror_copies_waiting.store(false, std::memory_order_relaxed);
        return;
    }
    if (VramPromotionsPaused()) {
        BbStats::mirror_copies_waiting.store(true, std::memory_order_relaxed);
        return;
    }
    BbStats::mirror_copies_waiting.store(false, std::memory_order_relaxed);
    std::vector<u64> blocks;
    for (const auto& range : promote_candidates) {
        for (u64 block = range.start; block < range.end; ++block) {
            blocks.push_back(block);
        }
    }
    promote_candidates.Clear();
    if (LayerMode()) {
        LayerPromote(blocks); // mirrors in VRAM (buffer_cache_layer.cpp)
        return;
    }
    u64 moved = 0;
    for (const u64 block : blocks) {
        const VAddr address = block << block_shift;
        // What the GPU wrote in place is in the game's memory: the copy is filled from there.
        if (!in_place_blocks.Contains(block) || !resident_ranges.Contains(block) ||
            !VramEligible(block)) {
            continue;
        }
        memory_tracker->UnmarkRegionAsGpuModified(address, block_size);
        gpu_modified_ranges.Subtract(address, block_size);
        const auto [memory, offset] = AllocateResidency(block_size);
        Backing backing;
        backing.start = block;
        backing.end = block + 1;
        backing.memory = memory;
        backing.offset = offset >> block_shift;
        resident_ranges.Subtract(block, block + 1);
        resident_ranges.Add(backing);
        in_place_blocks.Subtract(block, block + 1);
        // Every arena over it (as ProcessDemotions). The commands recorded so far read it in place;
        // the bind waits for them, and the next submission fills the VRAM copy before reading it.
        for (const auto& arena : arenas) {
            if (arena.cpu_addr <= address && address - arena.cpu_addr < arena.size_bytes) {
                BindsForArena(&arena)->binds.emplace_back(vk::SparseMemoryBind{
                    .resourceOffset = address - arena.cpu_addr,
                    .size = block_size,
                    .memory = memory,
                    .memoryOffset = offset,
                });
            }
        }
        memory_tracker->MarkRegionAsCpuModified(address, block_size);
        moved += block_size;
        const auto trap_range = VramTrapRange();
        if (VramTrapsEnabled() &&
            (trap_range.second ? address < trap_range.first + trap_range.second &&
                                     trap_range.first < address + block_size
                               : gpu_written_bytes.Overlaps(address, address + block_size))) {
            static const bool registered = [] {
                Core::Signals::Instance()->RegisterAccessViolationHandler(VramTrapHandler, 0);
                return true;
            }();
            vram_trap_block_size = block_size;
            std::scoped_lock lk{vram_trapped_mutex};
            vram_trapped[address] = block_size;
            mprotect(reinterpret_cast<void*>(address), block_size, PROT_NONE);
        }
    }
    if (moved == 0) {
        return;
    }
    bind_wait_tick = std::max(bind_wait_tick, submitted);
    if (((promoted_bytes + moved) >> 20) != (promoted_bytes >> 20)) {
        std::printf("Guest memory: %llu MiB of loaded blocks moved from in place to VRAM\n",
                    (unsigned long long)((promoted_bytes + moved) >> 20));
    }
    promoted_bytes += moved;
}

void BufferCache::UnmapInPlace(VAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    unmaps_queued.fetch_add(1, std::memory_order_release);
    std::scoped_lock lk{pending_unmaps_mutex};
    pending_unmaps.emplace_back(addr, size);
    unmaps_pending.store(true, std::memory_order_release);
}

void BufferCache::Maintain() {
    maintained_epoch = packet_epoch;
    // A heap's pieces given back 3 s ago: their chunk buffers go, then their memory.
    if (const auto retired = BbGuestMemory::RetiredChunks(); !retired.empty()) {
        for (const std::uint32_t index : retired) {
            chunk_buffers[index % chunk_buffers.size()].reset();
        }
        BbGuestMemory::DestroyRetiredChunks();
    }
    if (VramTrapsEnabled()) {
        ReportVramTraps();
    }
    ProcessPendingUnmaps();
    ProcessIdleBlocks();
    ProcessDemotions();
    ProcessPendingAssets();
    ProcessLateWrites();
}

void BufferCache::ProcessDeadImages(u64 started_stamp) {
    if (!dead_images_pending.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<DeadRange> ranges;
    {
        std::scoped_lock lk{pending_unmaps_mutex};
        ranges.swap(dead_image_ranges);
        dead_images_pending.store(false, std::memory_order_release);
    }
    static u64 freed = 0, images_since = 0, late = 0, late_gpu_bytes = 0;
    static u32 printed = 0;
    for (const auto& range : ranges) {
        // Late: a submission made after the free was decoded before this (the block may hold the
        // game's next allocation and the images the GPU made in it, lost when dropped).
        u64 gpu_bytes = 0;
        freed += texture_cache.FreeDeadImages(range.addr, range.size, &gpu_bytes);
        if (started_stamp > range.stamp) {
            ++late;
            late_gpu_bytes += gpu_bytes;
        }
    }
    images_since += ranges.size();
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if ((freed != 0 || late != 0) && now - printed >= 5) {
        std::printf("Texture cache: %llu MiB of images in memory the game freed dropped (%llu "
                    "ranges, %llu late, %llu KiB of GPU-made images in late ones)\n",
                    (unsigned long long)(freed >> 20), (unsigned long long)images_since,
                    (unsigned long long)late, (unsigned long long)(late_gpu_bytes >> 10));
        printed = now;
        freed = images_since = late = late_gpu_bytes = 0;
    }
}

void BufferCache::ProcessPendingUnmaps() {
    if (!unmaps_pending.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<std::pair<VAddr, u64>> unmaps;
    {
        std::scoped_lock lk{pending_unmaps_mutex};
        unmaps.swap(pending_unmaps);
        unmaps_pending.store(false, std::memory_order_release);
    }
    // The blocks bound to memory the game no longer maps there: unbound, and bound again (to
    // whatever is mapped then) when next used. Partly unmapped blocks go too.
    for (const auto& [addr, size] : unmaps) {
        const u64 first = addr >> block_shift, end = (addr + size + block_size - 1) >> block_shift;
        IntervalList<> unbind;
        if (LayerMode()) {
            LayerDemote(first, end, false, 5); // the memory there is gone: no copy back
            LayerRepublish(first, end);        // paged bindings read zero there now
            layer_published.Subtract(first, end);
            ++layer_generation; // the mapping changed: memoized resolutions (LayerBind) go
        }
        asset_bytes.Subtract(first << block_shift, end << block_shift);
        gpu_written_bytes.Subtract(first << block_shift, end << block_shift);
        // bbport: blocks wholly unmapped hold nobody's data: whatever kept them in place (the
        // CPU wrote them, the GPU wrote them in place, their watch failed) is over. Memory mapped
        // there again starts as new. BB_LAYER_UNMAP_RESET=0: as before.
        static const bool unmap_reset = !bb_env_off("BB_LAYER_UNMAP_RESET");
        static std::atomic<u64> unmap_reset_bytes{0};
        if (const u64 whole_first = (addr + block_size - 1) >> block_shift,
            whole_end = (addr + size) >> block_shift;
            unmap_reset && whole_first < whole_end) {
            const u64 total = unmap_reset_bytes.fetch_add((whole_end - whole_first) << block_shift) +
                              ((whole_end - whole_first) << block_shift);
            if ((total >> 26) != ((total - ((whole_end - whole_first) << block_shift)) >> 26)) {
                std::printf("Guest memory: %llu MiB of unmapped blocks started over\n",
                            (unsigned long long)(total >> 20));
            }
            dynamic_blocks.Subtract(whole_first, whole_end);
            if (LayerMode()) {
                layer_cpu_blocks.Subtract(whole_first, whole_end);
                layer_cpu_hot.Subtract(whole_first, whole_end);
                layer_watch_failed.Subtract(whole_first, whole_end);
                layer_write_bounces.Subtract(whole_first, whole_end);
                layer_watched.Subtract(whole_first, whole_end);
                layer_edge_blocks.Subtract(whole_first, whole_end);
                layer_volatile.Subtract(whole_first, whole_end);
                for (u64 block = whole_first; block < whole_end; ++block) {
                    layer_watch_strikes.erase(block);
                }
            }
            std::scoped_lock dynamic_lock{dynamic_mutex};
            for (u64 block = whole_first; block < whole_end; ++block) {
                cpu_write_frames.erase(block);
            }
        }
        copy_back_blocks.Subtract(first, end);
        promote_candidates.Subtract(first, end);
        idle_demoted.Subtract(first, end);
        in_place_blocks.ForEachInRange(first, end, [&](const Interval& iv) {
            unbind.Add({std::max(first, iv.start), std::min(end, iv.end)});
        });
        for (const auto& range : unbind) {
            const VAddr from = range.start << block_shift, to = range.end << block_shift;
            for (const auto& arena : arenas) {
                const VAddr a = std::max<VAddr>(from, arena.cpu_addr);
                const VAddr b = std::min<VAddr>(to, arena.cpu_addr + arena.size_bytes);
                if (a < b) {
                    BindsForArena(&arena)->binds.emplace_back(vk::SparseMemoryBind{
                        .resourceOffset = a - arena.cpu_addr,
                        .size = b - a,
                        .memory = {},
                        .memoryOffset = 0,
                    });
                }
            }
            resident_ranges.Subtract(range.start, range.end);
            in_place_blocks.Subtract(range.start, range.end);
            asset_bytes.Subtract(range.start << block_shift, range.end << block_shift);
            gpu_written_bytes.Subtract(range.start << block_shift, range.end << block_shift);
            promote_candidates.Subtract(range.start, range.end);
        }
        // bbport: blocks wholly unmapped that have a VRAM copy leave VRAM too (their slots go
        // to the next blocks moved there; uploaded again if mapped and used).
        for (u64 block = (addr + block_size - 1) >> block_shift; block < (addr + size) >> block_shift;
             ++block) {
            if (resident_ranges.Contains(block)) {
                BbStats::vram_unmapped_bytes.fetch_add(block_size, std::memory_order_relaxed);
            }
            EvictVramBlock(block);
        }
    }
}

} // namespace VideoCore
