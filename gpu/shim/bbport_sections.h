// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport BB_SECTIONS=1 (diagnostics): time the draw recording thread (and, "A:" parts, the GPU
// command thread) spends per part of a draw, inclusive (nested parts count in their parent too),
// printed with the draw pipe statistics.
#pragma once

#include <algorithm>
#include "shim/bb_env.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "bbport_cpu.h"

namespace BbSections {
enum Id : std::uint32_t {
    Packet,          ///< a whole packet of the draw pipe
    Task,            ///< ordered tasks (fences, signals, DMA, flips)
    DrawRecord,
    PrepareRenderState,
    BindResources,
    BindBuffers,
    BindTextures,
    BeginRendering,
    ResolveVertexBuffers,
    ResolveIndexBuffer,
    EmitVertexBuffers,
    UpdateDynamicState,
    DispatchRecord,
    ShaderHle,
    ObtainBuffer,
    Flush,
    Motion,          ///< object motion (bone palettes, index ranges)
    PipelineBind,    ///< Pipeline::BindResources (descriptors, push constants)
    Kick,            ///< ResetBindings and the hand-over to the Vulkan recording threads
    EnsureResident,
    ObtainStream,  ///< ObtainBuffer's copy of a small read-only buffer into the stream buffer
    ObtainVram,    ///< ObtainBuffer's VRAM path (SynchronizeMemory)
    LayerBind,     ///< BB_LAYER_MEMORY: BufferCache::LayerBind
    MotionHash,    ///< object motion: its buffers read and hashed (palettes, instances)
    MotionGate,    ///< object motion: whether a gated draw moved
    MotionIndex,   ///< object motion: the index range
    MotionPrepare, ///< object motion: the draw's history slot
    // The GPU command thread (stage A), per draw handed over.
    ADraw,         ///< Rasterizer::Draw: the pipeline selection and the hand-over
    APipeline,     ///< PipelineCache::GetGraphicsPipeline
    APrepared,     ///< its prepared-draw check (TryPreparedPipeline)
    APostDraw,     ///< packing the draw for the recording thread
    ARing,         ///< constants copied into the ring (CollectRingBindings)
    ARingChecks,   ///< its checks of each buffer (GPU-written, pending writes, VRAM copies)
    ARingCopy,     ///< its copies (memcpy and flush)
    ARingSharp,    ///< its V# reads and range clamps
    ARingNote,     ///< its pending read notes (NotePendingRead)
    ADecode,       ///< decoding a whole submission (everything of stage A)
    Count,
};
inline constexpr const char* Names[Count] = {
    "packet",       "tasks",       "draw",          "render state", "bind",
    "buffers",      "textures",    "begin render",  "vertex",       "index",
    "emit vertex",  "dynamic",     "dispatch",      "HLE shader",   "ObtainBuffer",
    "flush",        "motion",      "descriptors",   "kick",
    "EnsureResident", "stream copy", "VRAM path", "layer bind",
    "motion hash", "motion gate", "motion index", "motion prepare",
    "A: draw", "A: pipeline", "A: prepared", "A: post", "A: ring", "A: ring checks",
    "A: ring copy", "A: ring sharp", "A: ring note", "A: decode",
};
inline std::array<std::atomic<std::uint64_t>, Count> cycles{};
/// Constant buffers stage A looked at and bytes it copied into the ring (with the sections).
inline std::atomic<std::uint64_t> ring_buffers{0}, ring_copies{0}, ring_bytes{0};

inline bool Enabled() {
    static const bool on = bb_env_on("BB_SECTIONS");
    return on;
}
/// Set on the draw recording thread (DrawPipe::Run) and the GPU command thread: only their time
/// is counted.
inline thread_local bool recording_thread = false;
inline thread_local bool command_thread = false;

struct Scope {
    explicit Scope(Id id_)
        : id{id_}, start{Enabled() && (recording_thread || command_thread) ? BbCpu::Ticks() : 0} {}
    ~Scope() {
        if (start) {
            cycles[id].fetch_add(BbCpu::Ticks() - start, std::memory_order_relaxed);
        }
    }
    Id id;
    std::uint64_t start;
};

/// BB_SECTIONS_SLOW=1 (with BB_SECTIONS=1): at each frame's flip, a frame whose recording took
/// 1.5x the running average prints its costliest parts (which work makes a slow frame).
/// `tsc_hz`: the time stamp counter's frequency.
inline void FrameEnd(std::uint64_t tsc_hz) {
    static const bool on = [] {
        return Enabled() && bb_env_on("BB_SECTIONS_SLOW");
    }();
    if (!on || tsc_hz == 0) {
        return;
    }
    static std::array<std::uint64_t, Count> last{};
    static double average = 0.0;
    static std::uint64_t frames = 0;
    std::array<std::uint64_t, Count> delta{};
    for (std::uint32_t i = 0; i < Count; ++i) {
        const std::uint64_t now = cycles[i].load(std::memory_order_relaxed);
        delta[i] = now - last[i];
        last[i] = now;
    }
    const double packet_ms = double(delta[Packet]) * 1e3 / double(tsc_hz);
    const bool slow = ++frames > 60 && average > 0.0 && packet_ms > 1.5 * average;
    average = average == 0.0 ? packet_ms : average * 0.95 + packet_ms * 0.05;
    if (!slow) {
        return;
    }
    std::array<std::uint32_t, Count> order{};
    for (std::uint32_t i = 0; i < Count; ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](std::uint32_t a, std::uint32_t b) { return delta[a] > delta[b]; });
    std::printf("Slow frame: recording %.1f ms (average %.1f):", packet_ms, average);
    for (std::uint32_t k = 0, shown = 0; k < Count && shown < 10; ++k) {
        const std::uint32_t i = order[k];
        if (i == Packet || delta[i] == 0) {
            continue;
        }
        std::printf(" %s %.1f", Names[i], double(delta[i]) * 1e3 / double(tsc_hz));
        ++shown;
    }
    std::printf("\n");
}
} // namespace BbSections

#define BB_SECTION_JOIN2(a, b) a##b
#define BB_SECTION_JOIN(a, b) BB_SECTION_JOIN2(a, b)
#define BB_SECTION(id) BbSections::Scope BB_SECTION_JOIN(bb_section_scope_, __LINE__){BbSections::id}
