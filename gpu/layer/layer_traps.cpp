// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport BB_LAYER_MEMORY: blocks with a VRAM copy are write-protected (BB_LAYER_WRITE_TRAPS=0:
// off). The game's own code writes some of them without telling us (its heap's headers, parts of
// its resource loaders), and a stale copy then gives exploding vertices and garbage textures. A
// write there is let through at once; the block goes back in place before the next binding (the
// writers seen often stay there: NoteCpuWrite). Also: LayerMirrored (is a range in VRAM) and the
// mirror verification (BB_LAYER_VERIFY). Split out of buffer_cache_layer.cpp (2026-10-26).
#include <algorithm>
#include "shim/bb_env.h"
#include <bit>
#include <array>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <sys/mman.h>
#ifdef __APPLE__
#include <sys/ucontext.h> // bbport: <ucontext.h> requires _XOPEN_SOURCE on macOS
#else
#include <ucontext.h>
#endif
#include <map>
#include <tuple>
#include <string>
#include <cstdio>
#include <fmt/format.h>
#include <unordered_map>
#include <xxhash.h>
#include "shader/profile.h"
#include "bbport_free_check.h"
#include "bbport_guest_memory.h"
#include "bbport_write_log.h"
#include "bbport_sections.h"
#include "bbport_toggles.h"
#include "buffers/buffer_cache.h"
#include "memory/dirty_tracker.h"
#include "common/signal_context.h"
#include "core/signals.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"
// After the Vulkan headers of the video core (their configuration).
#include "bblayer_gpu_memory.h"
#include "bblayer_write_traps.h"
#include "layer/buffer_cache_layer_internal.h"

namespace VideoCore {

namespace {
std::mutex layer_trap_mutex;
std::unordered_set<u64> layer_trap_armed;              // blocks protected now
std::vector<std::pair<u64, u64>> layer_trap_hits;      // block, guest instruction (0: announced)
std::atomic<bool> layer_trap_pending{false};
std::atomic<u64> layer_trap_faults{0};
u32 layer_trap_shift = 16;

void LayerUnprotect(u64 block) {
    // Writable again unless another owner (an image there) still traps it (bblayer_write_traps.h).
    BbLayer::WriteTraps::Set(block << layer_trap_shift, u64{1} << layer_trap_shift,
                             BbLayer::WriteTraps::Mirror, false);
}

bool LayerTrapHandler(void* context, void* fault_address) {
    if (!Common::IsWriteError(context)) {
        return false;
    }
    const u64 block = reinterpret_cast<u64>(fault_address) >> layer_trap_shift;
    {
        std::scoped_lock lk{layer_trap_mutex};
        if (!layer_trap_armed.erase(block)) {
            return false;
        }
        // Writable before the hit can be seen: LayerProcessTraps sets the trap again for a
        // watched block (LayerArmTraps, under this lock). Unprotected after the lock, that new
        // trap was taken away again while the block still counted as armed: the game's next
        // writes there went unseen, its VRAM copy stale for good (another game: exploding vertices
        // everywhere, 2026-10-14).
        LayerUnprotect(block); // the write runs again, now through
        layer_trap_hits.emplace_back(block, u64(Common::Gregs(context)[REG_RIP]));
    }
    layer_trap_faults.fetch_add(1, std::memory_order_relaxed);
    layer_trap_pending.store(true, std::memory_order_release);
    return true;
}
} // namespace

void BufferCache::LayerArmTraps(u64 first, u64 end) {
    if (!LayerTrapsOn()) {
        return;
    }
    static const bool registered = [this] {
        layer_trap_shift = static_cast<u32>(block_shift);
        Core::Signals::Instance()->RegisterAccessViolationHandler(LayerTrapHandler, 0);
        return true;
    }();
    (void)registered;
    // One mprotect per run of blocks, not per block: each call interrupts every core running the
    // game's threads (TLB shootdown), and a mirror of 100 MiB made 1600 of them (the GPU thread
    // 250 ms in the kernel while an area streamed in).
    std::scoped_lock lk{layer_trap_mutex};
    u64 run_first = 0, run_end = 0;
    const auto flush = [&] {
        if (run_end > run_first) {
            BbStats::Timer timer{BbStats::t_layer[0]};
            BbStats::layer_mprotects.fetch_add(1, std::memory_order_relaxed);
            BbLayer::WriteTraps::Set(run_first << block_shift, (run_end - run_first) << block_shift,
                                     BbLayer::WriteTraps::Mirror, true);
        }
        run_first = run_end = 0;
    };
    int prot = 0, type = -1;
    uintptr_t vma_start = 0, vma_end = 0;
    for (u64 block = first; block < end; ++block) {
        if (layer_volatile.Contains(block, block + 1)) {
            flush(); // refreshed for every binding anyway
            continue;
        }
        const VAddr address = block << block_shift;
        // Only what the game's CPU may write (SCE_KERNEL_PROT_CPU_WRITE), whole in one mapping.
        if (address < vma_start || address >= vma_end) {
            if (!runtime_memory_vma_info(address, &prot, &type, &vma_end)) {
                vma_start = vma_end = 0;
                flush();
                continue;
            }
            vma_start = address;
        }
        if (!(prot & 0x2) || vma_end < address + block_size || !layer_trap_armed.insert(block).second) {
            flush();
            continue;
        }
        if (run_end != block) {
            flush();
            run_first = block;
        }
        run_end = block + 1;
    }
    flush();
}

void BufferCache::LayerDisarmTraps(u64 first, u64 end, bool demote) {
    std::scoped_lock lk{layer_trap_mutex};
    if (layer_trap_armed.empty()) {
        return;
    }
    // As LayerArmTraps: one mprotect per run of blocks that were armed.
    u64 run_first = 0, run_end = 0;
    const auto flush = [&] {
        if (run_end > run_first) {
            BbStats::Timer timer{BbStats::t_layer[0]};
            BbStats::layer_mprotects.fetch_add(1, std::memory_order_relaxed);
            BbLayer::WriteTraps::Set(run_first << layer_trap_shift,
                                     (run_end - run_first) << layer_trap_shift,
                                     BbLayer::WriteTraps::Mirror, false);
        }
        run_first = run_end = 0;
    };
    for (u64 block = first; block < end; ++block) {
        if (!layer_trap_armed.erase(block)) {
            flush();
            continue;
        }
        if (run_end != block) {
            flush();
            run_first = block;
        }
        run_end = block + 1;
        if (demote) {
            layer_trap_hits.emplace_back(block, 0);
            layer_trap_pending.store(true, std::memory_order_release);
        }
    }
    flush();
}

void BufferCache::LayerProcessTraps() {
    LayerForgetGpuData(); // before any demotion copies the GPU's data back
    if (!layer_trap_pending.load(std::memory_order_acquire)) {
        return; // every binding asks: no clock read when there is nothing to do
    }
    BbStats::Timer timer{BbStats::t_layer[3]};
    std::vector<std::pair<u64, u64>> hits;
    {
        std::scoped_lock lk{layer_trap_mutex};
        hits.swap(layer_trap_hits);
        layer_trap_pending.store(false, std::memory_order_release);
    }
    for (const auto& [block, rip] : hits) {
        const bool watched = layer_watched.Contains(block, block + 1);
        // An edge block of a range the GPU writes in VRAM: the CPU wrote beside that range (its
        // bytes are the GPU's): they are copied back below before it goes in place.
        const bool edge = layer_edge_blocks.Contains(block, block + 1) && Layer().AnyValid(block, block + 1);
        ++layer_trap_kinds[rip == 0 ? 0 : (watched ? 1 : 2) + (IsGuestCode(rip) ? 0 : 2)];
        // The CPU uses this block: GPU writes there stay in place from now on (until it is handed
        // out again), so a VRAM copy never holds GPU data the CPU's data could meet.
        LayerMarkCpuBlocks(block, block + 1);
        if (MarkLogged(block << block_shift, 1)) {
            std::printf("Layer mark: trap block %#llx rip %#llx%s\n",
                        (unsigned long long)(block << block_shift), (unsigned long long)rip,
                        rip && IsGuestCode(rip) ? " (game)" : "");
        }
        // A watched block the CPU wrote (the game's code or ours: HLE command buffers): it stays in
        // VRAM, uploaded again; with BB_LAYER_WATCH_HOT=1 however often (as the model of 0.3
        // uploads what the CPU wrote), else until it counts as often written (NoteCpuWrite: then
        // a request sends it in place). An announced write may still be under way (a file read):
        // in place as below, no trap set meanwhile. The runtime's file reads retry once their
        // pages fault again (a trap set between its touch and the read).
        if (rip != 0 && !(watched && LayerWatchHot())) {
            NoteCpuWrite(block << block_shift, rip); // often written: in place for good
        }
        if (rip != 0 && watched && Layer().AnyValid(block, block + 1)) {
            LayerRefreshWatched(block);
            continue;
        }
        // Written by the CPU since its copy was made: in place from now on, until it is known
        // data again. An announced write: the GPU's data beside its bytes is copied back first
        // (LayerForgetGpuData took its bytes out). A write nobody announced: the CPU reuses the
        // memory, the GPU's data there is dead (copying it back would land over the CPU's).
        LayerDemote(block, block + 1, rip == 0 || edge, 3);
    }
}

u64 BufferCache::LayerTrapFaults() {
    return layer_trap_faults.load(std::memory_order_relaxed);
}

bool BufferCache::LayerMirrored(VAddr addr, u64 size) {
    if (size == 0 || !LayerActive()) {
        return false;
    }
    // Asked for every small binding by two threads: the answer memoized per thread by the
    // blocks the range covers while no block gained or lost its copy (the module's lock taken
    // only then). By blocks, not by address: the game's constants move through a ring, a new
    // address every draw within the same few blocks.
    struct Memo {
        u64 first = 0, end = 0, version = ~0ULL;
        bool mirrored = false;
    };
    thread_local std::array<Memo, 1024> memo{};
    const u32 shift = Layer().BlockShift();
    const u64 first = addr >> shift, end = ((addr + size - 1) >> shift) + 1;
    auto& m = memo[(first ^ (end << 7)) & (memo.size() - 1)];
    const u64 version = Layer().ValidVersion();
    if (m.first == first && m.end == end && m.version == version) {
        return m.mirrored;
    }
    const bool mirrored = Layer().AnyValid(first, end);
    m = {first, end, version, mirrored};
    return mirrored;
}
bool BufferCache::LayerVerifyOn() {
    static const bool on = [] {
        const char* env = bb_env("BB_LAYER_VERIFY");
        const bool enabled = env && (env[0] == '1' || env[0] == '2');
        if (enabled) {
            std::printf("Layer verify: VRAM copies checked against the game's memory at every "
                        "binding (BB_LAYER_VERIFY=1, slow)\n");
        }
        return enabled;
    }();
    return on;
}

namespace {
/// BB_LAYER_VERIFY: the game's bytes of each mirrored block as last bound (its copy in VRAM then
/// matched them, unless the GPU wrote it there).
struct VerifiedBlock {
    u64 hash = 0;
    u64 frame = 0;
    std::unique_ptr<u8[]> bytes;
};
std::mutex layer_verify_mutex;
std::unordered_map<u64, VerifiedBlock> layer_verified;
u64 layer_verify_reports = 0;
} // namespace

void BufferCache::LayerVerifyForget(u64 first, u64 end) {
    std::scoped_lock lk{layer_verify_mutex};
    for (u64 block = first; block < end; ++block) {
        layer_verified.erase(block);
    }
}

void BufferCache::LayerVerifyMirror(VAddr address, u64 size, bool is_texel_buffer) {
    const u64 first = address >> block_shift;
    const u64 end = ((address + size - 1) >> block_shift) + 1;
    const u64 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    std::scoped_lock lk{layer_verify_mutex};
    for (u64 block = first; block < end; ++block) {
        const VAddr a = block << block_shift;
        // The GPU's data (newer than the game's memory) and blocks refreshed at every binding
        // are not compared.
        if (!Layer().AnyValid(block, block + 1) || layer_volatile.Contains(block, block + 1) ||
            gpu_modified_ranges.Intersects(a, block_size)) {
            layer_verified.erase(block);
            continue;
        }
        auto& entry = layer_verified[block];
        if (entry.bytes && entry.frame == frame) {
            continue; // once a frame
        }
        const auto* now = reinterpret_cast<const u8*>(a);
        const u64 hash = XXH3_64bits(now, block_size);
        const bool announced = memory_tracker->IsRegionCpuModified(a, block_size);
        bool armed;
        {
            std::scoped_lock trap_lk{layer_trap_mutex};
            armed = layer_trap_armed.contains(block);
        }
        u32 open_pages = 0;
        for (VAddr page = a; page < a + block_size; page += 4096) {
            open_pages += (BbLayer::WriteTraps::Reasons(page) & BbLayer::WriteTraps::Mirror) ? 0 : 1;
        }
        if (entry.bytes && hash != entry.hash && !announced && layer_verify_reports < 300) {
            // Changed with no announcement while its trap counts as set but pages of it are
            // writable: the copy in VRAM is stale (the trap handler race fixed 2026-10-14).
            u64 changed = 0, first_changed = ~0ull, last_changed = 0;
            for (u64 i = 0; i < block_size; ++i) {
                if (entry.bytes[i] != now[i]) {
                    ++changed;
                    first_changed = std::min(first_changed, i);
                    last_changed = i;
                }
            }
            bool was_zero = true;
            for (u64 i = first_changed; i <= last_changed && was_zero; ++i) {
                was_zero = entry.bytes[i] == 0;
            }
            int prot = 0, type = -1;
            uintptr_t vma_end = 0;
            const bool vma = runtime_memory_vma_info(a, &prot, &type, &vma_end);
            // Only a trap that should have seen the write: armed with pages open (a refresh
            // uploaded meanwhile also changes the bytes with nothing pending; not reported).
            // BB_LAYER_VERIFY=2: every unannounced change of a mirrored block (refreshes too).
            static const bool verify_all = [] {
                const char* env = bb_env("BB_LAYER_VERIFY");
                return env && env[0] == '2';
            }();
            if ((armed && open_pages != 0) || verify_all) {
                ++layer_verify_reports;
                std::printf(
                    "Layer verify: STALE %#llx (binding %#llx+%#llx%s): %llu bytes changed at "
                    "+%#llx..+%#llx (before: %s); trap %s, %u pages without it, page traps %#x, vma %s prot %#x type "
                    "%d; asset %s, GPU-written %s, watched %d, CPU block %d, dynamic %d, "
                    "frames since last check %llu\n",
                    (unsigned long long)a, (unsigned long long)address, (unsigned long long)size,
                    is_texel_buffer ? " texel" : "", (unsigned long long)changed,
                    (unsigned long long)first_changed, (unsigned long long)last_changed,
                    was_zero ? "zeros" : "data", armed ? "armed" : "not armed", open_pages,
                    BbLayer::WriteTraps::Reasons(a), vma ? "yes" : "no", prot, type,
                    asset_bytes.Contains(a, a + block_size) ? "all"
                    : asset_bytes.Overlaps(a, a + block_size) ? "part" : "none",
                    gpu_written_bytes.Contains(a, a + block_size) ? "all"
                    : gpu_written_bytes.Overlaps(a, a + block_size) ? "part" : "none",
                    layer_watched.Contains(block, block + 1) ? 1 : 0,
                    layer_cpu_blocks.Contains(block, block + 1) ? 1 : 0,
                    dynamic_blocks.Contains(block) ? 1 : 0,
                    (unsigned long long)(frame - entry.frame));
                if (LayerProbeOn()) {
                    LayerArmProbe(a, block_size); // the next CPU write there names its code
                }
            }
        }
        if (!entry.bytes) {
            entry.bytes = std::make_unique<u8[]>(block_size);
        }
        std::memcpy(entry.bytes.get(), now, block_size);
        entry.hash = hash;
        entry.frame = frame;
    }
}

} // namespace VideoCore
