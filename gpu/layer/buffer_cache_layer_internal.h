// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: small helpers the parts of the layer's BufferCache side share (buffer_cache_layer.cpp,
// layer_bind.cpp, layer_paged.cpp, layer_copyback.cpp, layer_traps.cpp). Only those files include
// it; everything is `inline` (one instance program-wide, inlinable without LTO).
#pragma once

#include <atomic>
#include "shim/bb_env.h"
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>
#include "bblayer_gpu_memory.h"
#include "common/types.h"

extern "C" int runtime_memory_vma_info(uintptr_t address, int* prot, int* type, uintptr_t* end);

namespace VideoCore {

inline BbLayer::GpuMemory& Layer() {
    return BbLayer::GpuMemory::Get();
}


/// Guest code (the game's image with its linked modules): the writer behind a trap hit. A hit
/// from our own code (the runtime touching pages before a file read into them) must not set the
/// trap again until that write is done.
inline bool IsGuestCode(u64 rip) {
    constexpr u64 Image = 0x800000000ull, ImageSize = 128ull << 20;
    return rip >= Image && rip < Image + ImageSize;
}

/// The valid blocks of [first, end) by the mirror holding them: f(mirror, a, b) for each piece.
/// A run of valid blocks may go on into a neighbouring mirror (mirrors touch: per game allocation
/// with BB_LAYER_DOMAINS, or after a compaction): taken whole from the first one's, a copy went
/// past its end (a CP write into another allocation, a GPU page fault and hang, 2026-10-23).
template <typename F>
void ForEachValidPiece(const BbLayer::GpuMemory& layer, u64 first, u64 end, u32 shift, F&& f) {
    std::vector<std::pair<u64, u64>> runs;
    layer.ForEachValid(first, end, [&](u64 a, u64 b) { runs.emplace_back(a, b); });
    for (const auto& [a, b] : runs) {
        for (u64 at = a; at < b;) {
            const auto mirror = layer.MirrorAt(at << shift);
            if (!mirror) {
                ++at; // valid without a mirror: nothing to copy from or to
                continue;
            }
            const u64 piece_end = std::min<u64>(b, (mirror->base + mirror->size) >> shift);
            if (piece_end <= at) {
                ++at;
                continue;
            }
            f(*mirror, at, piece_end);
            at = piece_end;
        }
    }
}

// Volatile blocks (BB_LAYER_VOLATILE=1; off by default): a range with a few unannounced blocks
// goes into one mirror anyway, those blocks refreshed for every binding that reads them. Off, the
// range stays mixed (in place). Their refreshes were ~300 MB/s of copies and, in render passes,
// up to ~100 pass breaks a frame; on a recorded route off was 4.29-4.31 ms of GPU time a frame,
// on 4.31-4.53 even with the in-pass reads in place, 5.4-7.3 before. They were ~0.1 % of the
// bytes bound.
inline bool LayerVolatileAllowed() {
    static const bool on = bb_env_on("BB_LAYER_VOLATILE");
    return on;
}

inline bool PagedVerifyOn() {
    static const bool on = [] {
        const bool enabled = (bb_env_on("BB_PAGED_VERIFY")) || (bb_env_on("BB_LAYER_CHECK"));
        if (enabled) {
            std::printf("Layer memory: paged bindings verified against the mappings (BB_PAGED_VERIFY=1)\n");
        }
        return enabled;
    }();
    return on;
}

/// BB_LAYER_MARK_LOG=lo-hi (diagnostics, hex guest addresses): who marks blocks there as the CPU's
/// (layer_cpu_blocks) and which allocations the game hands out over them.
inline bool MarkLogged(u64 address, u64 size) {
    static const std::pair<u64, u64> range = [] {
        const char* env = bb_env("BB_LAYER_MARK_LOG");
        if (!env || !*env) return std::pair<u64, u64>{0, 0};
        char* end = nullptr;
        const u64 lo = std::strtoull(env, &end, 16);
        const u64 hi = end && *end == '-' ? std::strtoull(end + 1, nullptr, 16) : 0;
        return std::pair<u64, u64>{lo, hi};
    }();
    static std::atomic<int> lines{0};
    return range.second > range.first && address < range.second && address + size > range.first &&
           lines.fetch_add(1, std::memory_order_relaxed) < 400;
}

inline bool LayerTrapsOn() {
    static const bool on = !bb_env_off("BB_LAYER_WRITE_TRAPS");
    return on;
}

} // namespace VideoCore
