// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport BB_LAYER_MEMORY: GPU data back into the game's memory (copy-back), the CPU read
// probes and read traps, queued promotions and volatile blocks. Split out of
// buffer_cache_layer.cpp (2026-10-26); shared helpers in buffer_cache_layer_internal.h.
#include <algorithm>
#include "shim/bb_env.h"
#include <bit>
#include <array>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <sys/mman.h>
#include <ucontext.h>
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

void BufferCache::LayerCopyBack(u64 first, u64 end) {
    LayerForgetGpuData();
    // The GPU's data in mirrors over [first, end) into the game's memory; the blocks stay valid
    // in VRAM (both copies equal now).
    // Each part of a valid run from the mirror holding it (a run may go on into a neighbour).
    std::vector<std::tuple<u64, u64, Buffer*>> pieces;
    ForEachValidPiece(Layer(), first, end, block_shift,
                      [&](const BbLayer::MemorySource& mirror, u64 a, u64 b) {
                          pieces.emplace_back(a, b, static_cast<Buffer*>(mirror.owner));
                      });
    for (const auto& [a, b, buffer] : pieces) {
        const VAddr from = a << block_shift;
        const u64 bytes = (b - a) << block_shift;
        if (!gpu_modified_ranges.Intersects(from, bytes)) {
            continue;
        }
        gpu_modified_ranges.ForEachInRange(from, bytes, [&](VAddr w_from, VAddr w_to) {
            if (const auto target = LayerInPlace(w_from, w_to - w_from)) {
                const vk::BufferCopy copy{buffer->Offset(w_from), target->second, w_to - w_from};
                BbWriteLog::NoteMirrorCopy(w_from, w_to - w_from);
                BbFreeCheck::NoteGpuWrite(w_from, w_to - w_from, 0, BbFreeCheck::MirrorCopyBack);
                runtime.CopyBuffer(buffer, target->first, std::span{&copy, 1});
                layer_copied_back_bytes += w_to - w_from;
                layer_copy_back_tick = scheduler.CurrentTick();
            }
        });
        gpu_modified_ranges.Subtract(from, bytes);
    }
}

// BB_LAYER_PROBE=1 (diagnostics): a GPU write kept in place because the CPU once wrote the block
// makes its pages no access; the CPU's first access after it is counted (by code address, read or
// write, GPU side or not) and lifts the trap. The data stays where it is, so nothing changes for
// the game but the faults: it shows which of these blocks the CPU really uses after the GPU.
namespace {
struct ProbeSlot {
    std::atomic<u64> key{0}; ///< block << 20 ^ code address, written once
    std::atomic<u32> reads{0}, writes{0}, gpu_side{0};
};
std::array<ProbeSlot, 1024> probe_slots;
std::atomic<u64> probe_hits{0}, probe_lost{0}, probe_armed_bytes{0};
} // namespace

bool BufferCache::LayerProbeOn() {
    static const bool on = bb_env_on("BB_LAYER_PROBE");
    return on;
}

void BufferCache::LayerArmProbe(VAddr address, u64 size) {
    if (!LayerProbeOn() || size == 0) {
        return;
    }
    constexpr u64 Page = 4096;
    const VAddr from = Common::AlignUp(address, Page); // whole pages only: never the CPU's edges
    const VAddr to = Common::AlignDown(address + size, Page);
    if (to <= from) {
        return;
    }
    for (VAddr page = from; page < to;) {
        // Runs of pages not armed yet (one call each).
        if (BbLayer::WriteTraps::Reasons(page) & BbLayer::WriteTraps::Probe) {
            page += Page;
            continue;
        }
        VAddr end = page + Page;
        while (end < to && !(BbLayer::WriteTraps::Reasons(end) & BbLayer::WriteTraps::Probe)) {
            end += Page;
        }
        BbLayer::WriteTraps::Set(page, end - page, BbLayer::WriteTraps::Probe, true);
        probe_armed_bytes.fetch_add(end - page, std::memory_order_relaxed);
        page = end;
    }
}

bool BufferCache::LayerProbeHit(VAddr address, u64 rip, bool write, bool gpu_thread) {
    if (!(BbLayer::WriteTraps::Reasons(address) & BbLayer::WriteTraps::Probe)) {
        return false;
    }
    probe_hits.fetch_add(1, std::memory_order_relaxed);
    const u64 key = ((address >> block_shift) << 20) ^ (rip & 0xFFFFF) ^ (rip >> 20 << 44);
    for (u64 i = 0; i < probe_slots.size(); ++i) {
        auto& slot = probe_slots[(key * 0x9E3779B97F4A7C15ull >> 54) + i & (probe_slots.size() - 1)];
        u64 expected = 0;
        if (slot.key.load(std::memory_order_relaxed) == key ||
            slot.key.compare_exchange_strong(expected, key, std::memory_order_relaxed) ||
            expected == key) {
            (write ? slot.writes : slot.reads).fetch_add(1, std::memory_order_relaxed);
            if (gpu_thread) {
                slot.gpu_side.fetch_add(1, std::memory_order_relaxed);
            }
            BbLayer::WriteTraps::Set(Common::AlignDown(address, 4096), 4096,
                                     BbLayer::WriteTraps::Probe, false);
            return true;
        }
    }
    probe_lost.fetch_add(1, std::memory_order_relaxed);
    BbLayer::WriteTraps::Set(Common::AlignDown(address, 4096), 4096, BbLayer::WriteTraps::Probe,
                             false);
    return true;
}

void BufferCache::LayerPrintProbe() {
    if (!LayerProbeOn()) {
        return;
    }
    struct Row {
        u64 key;
        u32 reads, writes, gpu_side;
    };
    std::vector<Row> rows;
    for (auto& slot : probe_slots) {
        if (const u64 key = slot.key.load(std::memory_order_relaxed)) {
            rows.push_back({key, slot.reads.exchange(0), slot.writes.exchange(0),
                            slot.gpu_side.exchange(0)});
        }
    }
    std::ranges::sort(rows, [](const Row& a, const Row& b) {
        return a.reads + a.writes > b.reads + b.writes;
    });
    std::printf("Layer probe: %llu first accesses in 10 s (%llu not recorded), %llu MiB armed\n",
                (unsigned long long)probe_hits.exchange(0),
                (unsigned long long)probe_lost.exchange(0),
                (unsigned long long)(probe_armed_bytes.exchange(0) >> 20));
    for (size_t i = 0; i < rows.size() && i < 12 && rows[i].reads + rows[i].writes; ++i) {
        const u64 block = rows[i].key >> 20;
        std::printf("  block %#llx: reads %u, writes %u (GPU side %u), code ..%05llx\n",
                    (unsigned long long)(block << block_shift), rows[i].reads, rows[i].writes,
                    rows[i].gpu_side, (unsigned long long)(rows[i].key & 0xFFFFF));
    }
}

// BB_LAYER_READ_TRAPS=1: what the GPU writes into a VRAM copy only is no access for the CPU until
// it is copied back (LayerReadTrapHit): the game or the translator reading it on the CPU would
// see stale data (on the PS4 the memory is one). Off by default for now.
bool BufferCache::LayerReadTrapsOn() {
    static const bool on = bb_env_on("BB_LAYER_READ_TRAPS");
    return on;
}

void BufferCache::LayerArmReadTraps(VAddr address, u64 size) {
    if (!LayerReadTrapsOn() || size == 0) {
        return;
    }
    constexpr u64 Page = 4096;
    const VAddr from = Common::AlignDown(address, Page);
    const VAddr to = Common::AlignUp(address + size, Page);
    std::scoped_lock lk{layer_read_trap_mutex};
    if (layer_read_trapped.Contains(from, to - from)) {
        return;
    }
    layer_read_trapped.ForEachNotInRange(from, to - from, [&](VAddr gap, u64 bytes) {
        BbLayer::WriteTraps::Set(gap, bytes, BbLayer::WriteTraps::VramData, true);
    });
    layer_read_trapped.Add(from, to - from);
}

void BufferCache::LayerQueuePromotion(u64 first, u64 end) {
    // Hashed by the whole range: indexed by its first block alone, ranges starting in the same
    // block (vertex streams of one buffer) took each other's slot, 1 call in 8 walked its blocks
    // again (~18, the module's lock each), ~4% of the draw recording thread. Looked at again
    // each second too: blocks become known data without anything moving (loads, GPU writes).
    struct Memo {
        u64 first = ~0ULL, end = 0, generation = 0;
        u32 second = 0;
    };
    thread_local auto memo = std::make_unique<std::array<Memo, 4096>>(); // see LayerBind
    auto& m = (*memo)[((first * 0x9E3779B97F4A7C15ull) ^ (end * 0xC2B2AE3D27D4EB4Full)) >> 52];
    const u32 second = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (m.first == first && m.end == end && m.generation == layer_generation &&
        m.second == second) {
        return;
    }
    m = {first, end, layer_generation, second};
    if (!GarlicInVramForLayer()) {
        return;
    }
    // And by block: ranges of data the game writes each frame at new addresses (its rings) are
    // new every time, their blocks are not.
    struct BlockMemo {
        u64 block = ~0ULL, generation = 0;
        u32 second = 0;
    };
    thread_local auto block_memo = std::make_unique<std::array<BlockMemo, 4096>>();
    for (u64 block = first; block < end; ++block) {
        auto& b = (*block_memo)[(block * 0x9E3779B97F4A7C15ull) >> 52];
        if (b.block == block && b.generation == layer_generation && b.second == second) {
            continue;
        }
        b = {block, layer_generation, second};
        if (!Layer().AnyValid(block, block + 1) && VramEligible(block)) {
            promote_candidates.Add({block, block + 1});
        }
    }
}

bool BufferCache::IsInVramLayer(VAddr addr, u64 size) const {
    return Layer().Resolve(addr, size).kind == BbLayer::Resolution::Kind::Mirror;
}

bool BufferCache::LayerAllKnown(u64 first, u64 end) {
    for (u64 block = first; block < end; ++block) {
        if (!Layer().AnyValid(block, block + 1) && !LayerEligible(block)) {
            return false;
        }
    }
    return true;
}

void BufferCache::LayerMakeVolatile(const std::vector<u64>& blocks) {
    for (const u64 block : blocks) {
        if (Layer().AnyValid(block, block + 1) || !Layer().ResolveInPlace(block << block_shift, block_size)) {
            continue; // in VRAM already, or not a whole block of a chunk (no source to refresh from)
        }
        // Volatile before the page table entries are rewritten (they keep it in place).
        layer_volatile.Add({block, block + 1});
        LayerPromoteRun(block, block + 1);
        if (!Layer().AnyValid(block, block + 1)) {
            layer_volatile.Subtract(block, block + 1);
        }
    }
}

void BufferCache::LayerRefreshVolatile(VAddr address, u64 size) {
    BbStats::Timer timer{BbStats::t_layer[5]};
    // The binding's bytes in volatile blocks: uploaded from the game's memory by the
    // SynchronizeMemory that follows (as CPU-modified: a GPU copy from its chunk, in stream
    // order). Only reading bindings use them (LayerBind).
    const u64 first = address >> block_shift;
    const u64 end = ((address + size - 1) >> block_shift) + 1;
    layer_volatile.ForEachInRange(first, end, [&](const Interval& iv) {
        const VAddr from = std::max<VAddr>(address, iv.start << block_shift);
        const VAddr to = std::min<VAddr>(address + size, iv.end << block_shift);
        if (from >= to) {
            return;
        }
        memory_tracker->UnmarkRegionAsGpuModified(from, to - from);
        gpu_modified_ranges.Subtract(from, to - from);
        memory_tracker->MarkRegionAsCpuModified(from, to - from);
        layer_volatile_refresh_bytes += to - from;
    });
}

} // namespace VideoCore
