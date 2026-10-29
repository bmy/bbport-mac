// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: state the parts of BufferCache share (buffer_cache.cpp, buffer_readback.cpp,
// buffer_residency.cpp, buffer_write_tracking.cpp, buffer_maintain.cpp, buffer_sync.cpp). Only
// those files include it. Everything here is `inline`: one instance in the whole program, and the
// small hot helpers stay inlinable without LTO (the Windows build has none).
#pragma once

#include <array>
#include "shim/bb_env.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>
#include "bbport_toggles.h"
#include "common/types.h"
#include "core/emulator_settings.h"
#include "core/memory.h"

extern "C" int runtime_memory_vma_info(uintptr_t address, int* prot, int* type, uintptr_t* end);
extern "C" int runtime_memory_direct_phys(uintptr_t address, uint64_t* phys, uintptr_t* end);

namespace VideoCore {

// bbport: the tick of the newest GPU write per 64 KiB of guest memory, per 1 MiB for writes over
// 16 MiB (ObtainBuffer with is_written: the only place ranges become GPU-modified). A readback
// waits for that submission only and copies on the readback queue, not behind the queued
// frames; a coarser grain let a buffer the GPU writes every frame hold up its neighbours.
// Read for every small constant binding of every draw (the GPU command thread, thousands of times
// a frame) while the draw recording thread notes writes: lock-free sparse page tables (a mutex and
// hash maps here took a third of that thread). Guest addresses below 2^40.
struct GpuWriteTicks {
    static constexpr u64 FineShift = 16, CoarseShift = 20;
    static constexpr u64 FineLimit = 16_MB;
    static constexpr u64 AddressBits = 40, LeafBits = 12, LeafMask = (1ull << LeafBits) - 1;
    struct Table {
        explicit Table(u64 shift_)
            : shift{shift_}, leaves((1ull << (AddressBits - shift_)) >> LeafBits) {}
        u64 shift;
        /// Page -> tick of the newest write; a leaf of 4096 pages is made at its first write.
        std::vector<std::atomic<std::atomic<u64>*>> leaves;
        std::atomic<u64>* Leaf(u64 page) const {
            return leaves[page >> LeafBits].load(std::memory_order_acquire);
        }
    };
    Table fine{FineShift};   ///< 64 KiB pages
    Table coarse{CoarseShift}; ///< 1 MiB pages (writes over FineLimit)
    std::mutex mutex;          ///< leaf creation
    void Note(u64 address, u64 size, u64 tick) {
        if (size == 0 || (address + size - 1) >> AddressBits) {
            return;
        }
        auto& table = size <= FineLimit ? fine : coarse;
        for (u64 page = address >> table.shift; page <= (address + size - 1) >> table.shift;
             ++page) {
            std::atomic<u64>* leaf = table.Leaf(page);
            if (!leaf) {
                std::scoped_lock lk{mutex};
                leaf = table.Leaf(page);
                if (!leaf) {
                    leaf = new std::atomic<u64>[1ull << LeafBits]{};
                    table.leaves[page >> LeafBits].store(leaf, std::memory_order_release);
                }
            }
            leaf[page & LeafMask].store(tick, std::memory_order_release);
        }
    }
    /// The newest write tick in the range, 0 when none is known.
    u64 Newest(u64 address, u64 size) const {
        if (size == 0 || (address + size - 1) >> AddressBits) {
            return 0;
        }
        u64 newest = 0;
        for (const Table* table : {&fine, &coarse}) {
            const u64 last = (address + size - 1) >> table->shift;
            for (u64 page = address >> table->shift; page <= last; ++page) {
                if (const std::atomic<u64>* leaf = table->Leaf(page)) {
                    newest = std::max(newest, leaf[page & LeafMask].load(std::memory_order_acquire));
                } else {
                    page |= LeafMask; // nothing written in this leaf
                }
            }
        }
        return newest;
    }
};
inline GpuWriteTicks& WriteTicks() {
    static GpuWriteTicks write_ticks;
    return write_ticks;
}

// bbport: GPU writes known to the byte (the copy shader's control list, CopyShaderDestinations): a
// few KiB at a time anywhere in 3.5 GiB, ~4000 times a second. In 64 KiB pages they marked the
// constants beside them as written by the GPU: bound in place instead of copied, a later constant
// RAM dump changed them before the GPU read them (single frames with another draw's quad).
// Small queries look at 256-byte granules (hashed; a collision only says "written" more often),
// larger ones at 64 KiB pages of their own.
struct PreciseWriteTicks {
    static constexpr u64 Shift = 8, Bits = 20;
    static constexpr u64 SmallLimit = 16_KB;
    std::vector<std::atomic<u64>> granules = std::vector<std::atomic<u64>>(1ull << Bits);
    GpuWriteTicks pages;
    static u64 Slot(u64 granule) {
        return (granule * 0x9E3779B97F4A7C15ull) >> (64 - Bits);
    }
    void Note(u64 address, u64 size, u64 tick) {
        if (size == 0) {
            return;
        }
        for (u64 g = address >> Shift; g <= (address + size - 1) >> Shift; ++g) {
            auto& slot = granules[Slot(g)];
            if (slot.load(std::memory_order_relaxed) < tick) {
                slot.store(tick, std::memory_order_release);
            }
        }
        pages.Note(address, size, tick);
    }
    u64 Newest(u64 address, u64 size) const {
        if (size == 0) {
            return 0;
        }
        // Every write is noted in its 64 KiB pages too: nothing there, nothing in the granules (most
        // constant bindings: one lookup instead of up to 64 scattered ones, 2 % of the recording
        // thread). A granule's tick above its page's is another address's (a hash collision).
        const u64 page = pages.Newest(address, size);
        if (page == 0 || size > SmallLimit) {
            return page;
        }
        u64 newest = 0;
        for (u64 g = address >> Shift; g <= (address + size - 1) >> Shift; ++g) {
            newest = std::max(newest, granules[Slot(g)].load(std::memory_order_acquire));
        }
        return std::min(newest, page);
    }
};
inline PreciseWriteTicks& PreciseTicks() {
    static PreciseWriteTicks ticks;
    return ticks;
}
/// A GPU command write (WRITE_DATA, DMA): small ones to the byte. In 64 KiB pages the 192 bytes the
/// game writes each frame beside its constant RAM dump ring marked the ring as written by the GPU:
/// the final pass's constants were bound in place, the next dump replaced them before the GPU read
/// them (single frames with the scene shifted or cut to a quarter; docs/Flicker_video_2026-10-17.ru.md).
inline void WatchGpuWrite(u64 address, u64 size, int line);
inline void NoteCommandWriteTick(u64 address, u64 size, u64 tick) {
    WatchGpuWrite(address, size, __LINE__);
    if (size <= PreciseWriteTicks::SmallLimit) {
        PreciseTicks().Note(address, size, tick);
    } else {
        WriteTicks().Note(address, size, tick);
    }
}

/// BB_GPU_WRITE_WATCH=address (diagnostics): GPU writes noted over the 64 KiB page holding it.
inline void WatchGpuWrite(u64 address, u64 size, int line) {
    static const u64 watched = [] {
        const char* env = bb_env("BB_GPU_WRITE_WATCH");
        return env ? std::strtoull(env, nullptr, 16) & ~0xFFFFull : 0ull;
    }();
    static std::atomic<u32> printed{0};
    if (watched && address < watched + 0x10000 && address + size > watched &&
        printed.fetch_add(1, std::memory_order_relaxed) < 2000) {
        std::printf("GPU write watch: line %d %#llx+%llu frame %llu\n", line,
                    (unsigned long long)address, (unsigned long long)size,
                    (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed));
    }
}
/// The newest GPU write tick in the range: page-granular writes and those known to the byte.
inline u64 NewestGpuWrite(u64 address, u64 size) {
    return std::max(WriteTicks().Newest(address, size), PreciseTicks().Newest(address, size));
}

// bbport (BB_READBACK_LOG): the newest GPU-written bindings, to show which writes a readback that
// waited for the GPU overlapped (a binding wider than what the shader writes would show here).
struct WrittenBinding {
    u64 address, size, tick;
};
inline std::array<WrittenBinding, 4096> written_bindings;
inline std::atomic<u64> written_head{0};

// bbport: BB_READBACK_LOG=path — every download of GPU-written buffer memory back to guest
// memory (a CPU write to a range the GPU wrote): who asked, where, how much, how long it took,
// how old the GPU's newest write there was, and whether the readback queue served it.
struct ReadbackLog {
    FILE* file = nullptr;
    std::mutex mutex;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};
inline ReadbackLog* Readbacks() {
    static ReadbackLog* const log = [] () -> ReadbackLog* {
        const char* path = bb_env("BB_READBACK_LOG");
        if (!path || !*path) {
            return nullptr;
        }
        auto* readbacks = new ReadbackLog;
        readbacks->file = std::fopen(path, "w");
        if (!readbacks->file) {
            delete readbacks;
            return nullptr;
        }
        std::fprintf(readbacks->file, "t_s,requester,address,window_kb,downloaded_kb,copies,is_write,"
                                      "wait_ms,ticks_since_write,write_done,readback_queue\n");
        return readbacks;
    }();
    return log;
}

// bbport: BB_GPU_WRITE_TWINS=1 — a guest write into a page that holds bytes the GPU is still
// writing (counters, small compute outputs rewritten every frame) does not wait for the GPU: the
// guest's copy of those bytes is kept as a "twin" and the page becomes CPU-modified at once.
// Uploads leave the GPU's bytes in the arena while the guest bytes still match their twin (the
// guest did not write them); downloads do not overwrite guest bytes that no longer match it (the
// guest wrote them since). Only pages with at most BB_GPU_WRITE_TWINS_MAX (1024) GPU-written
// bytes: larger outputs (vertices the CPU reads back, FaceGen) keep waiting for the GPU.
// Not with BB_READBACKS=2: a twinned page stays GPU-modified (read-protected) while the guest may
// write it, a write-only page the page manager cannot make (it stopped at the first cutscene).
inline bool TwinsEnabled() {
    static const bool enabled = [] {
        return bb_env_on("BB_GPU_WRITE_TWINS") &&
               EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Precise;
    }();
    return enabled;
}
inline u64 TwinsMaxBytes() {
    static const u64 max_bytes = [] {
        const char* env = bb_env("BB_GPU_WRITE_TWINS_MAX");
        return env ? std::strtoull(env, nullptr, 10) : 1024ull;
    }();
    return max_bytes;
}
inline std::mutex twins_mutex;
inline std::map<VAddr, std::vector<u8>> twins; ///< start -> guest bytes when the twin was made
inline std::atomic<u64> twin_faults{0}, twin_kept_bytes{0}, twin_guest_bytes{0};

enum class TwinRun { None, Unchanged, Changed };

/// Calls func(start, end, TwinRun) over [begin, end) in order (twins_mutex held).
template <typename Func>
void ForEachTwinRun(VAddr begin, VAddr end, Func&& func) {
    VAddr cursor = begin;
    auto it = twins.upper_bound(begin);
    if (it != twins.begin()) {
        --it;
    }
    for (; it != twins.end() && it->first < end; ++it) {
        const VAddr start = it->first, stop = it->first + it->second.size();
        if (stop <= begin) {
            continue;
        }
        const VAddr from = std::max(start, begin), to = std::min(stop, end);
        if (cursor < from) {
            func(cursor, from, TwinRun::None);
        }
        // bbport: through the backing view: with BB_READBACKS=2 the page is read-protected, and a
        // fault here (GPU thread, twins_mutex held) waited for a readback needing twins_mutex:
        // the game froze at the first cutscene (a movie frame written into GPU-written memory).
        thread_local std::vector<u8> guest_bytes;
        guest_bytes.resize(to - from);
        Core::Memory::Instance()->ReadBacking(from, guest_bytes.data(), to - from);
        const u8* guest = guest_bytes.data();
        const u8* twin = it->second.data() + (from - start);
        VAddr run = from;
        bool same = guest[0] == twin[0];
        for (VAddr at = from + 1; at <= to; ++at) {
            const bool now_same = at < to && guest[at - from] == twin[at - from];
            if (at == to || now_same != same) {
                func(run, at, same ? TwinRun::Unchanged : TwinRun::Changed);
                run = at;
                same = now_same;
            }
        }
        cursor = to;
    }
    if (cursor < end) {
        func(cursor, end, TwinRun::None);
    }
}

/// Drops the twins' parts in [begin, end), keeping the parts outside it (twins_mutex held).
inline void EraseTwins(VAddr begin, VAddr end) {
    auto it = twins.upper_bound(begin);
    if (it != twins.begin()) {
        --it;
    }
    while (it != twins.end() && it->first < end) {
        const VAddr start = it->first, stop = it->first + it->second.size();
        if (stop <= begin) {
            ++it;
            continue;
        }
        std::vector<u8> bytes = std::move(it->second);
        it = twins.erase(it);
        if (start < begin) {
            twins[start] = std::vector<u8>(bytes.begin(), bytes.begin() + (begin - start));
        }
        if (stop > end) {
            twins[end] = std::vector<u8>(bytes.begin() + (end - start), bytes.end());
        }
    }
}

// Defined in buffer_cache.cpp (its configuration from the environment).
extern bool integrated_gpu;
bool ChunkUploads();
bool ChunkTextures();
bool InPlaceTextures();
bool FixedArena();
bool MixStats();
bool LayerMode();
bool GarlicInVram();

} // namespace VideoCore
