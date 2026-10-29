// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: what the Rasterizer's source files share (render/raster_*.cpp): includes, the
// register accessors of the draw being recorded, small helpers.
#pragma once

#include <dlfcn.h>

#include <bit>
#include <csetjmp>
#include <map>
#include <unordered_set>

#include <magic_enum/magic_enum.hpp>
#include <xxhash.h>

#include "bbport_ce_stats.h"
#include "bbport_cpu.h"
#include "bbport_free_check.h"
#include "bbport_guest_memory.h"
#include "bbport_sections.h"
#include "bbport_threads.h"
#include "bbport_timeline.h"
#include "bbport_toggles.h"
#include "bbport_write_log.h"
#include "buffers/buffer.h"
#include "buffers/buffer_cache.h"
#include "common/hash.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "cp/command_processor.h"
#include "cp/cp_diagnostics.h"
#include "game_profile.h"
#include "render/gcn_to_vk.h"
#include "render/pipeline_cache.h"
#include "render/rasterizer.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/shader_hle.h"
#include "render/vk_button_prompts.h"
#include "render/vk_frame_capture.h"
#include "render/vk_gpu_labels.h"
#include "render/vk_gpu_profiler.h"
#include "render/vk_indirect_guard.h"
#include "render/vk_occlusion.h"
#include "render/vk_timestamps.h"
#include "shader/runtime_info.h"
#include "textures/image_view.h"
#include "textures/texture_cache.h"
#include "upscale/ui_composition.h"
#include "vulkan/instance.h"

namespace Vulkan {

inline const Gcn::Regs& Rasterizer::Regs() const {
    return stage_regs ? *stage_regs : liverpool->regs;
}

inline Gcn::CbDbExtent Rasterizer::CbExtent(u32 index) const {
    return stage_regs ? pipe_cb_extent[index] : liverpool->last_cb_extent[index];
}

inline const Gcn::ComputeProgram& Rasterizer::CsRegs() const {
    return stage_cs ? *stage_cs : liverpool->GetCsRegs();
}

inline Gcn::CbDbExtent Rasterizer::DbExtent() const {
    return stage_regs ? pipe_db_extent : liverpool->last_db_extent;
}

/// Whether the formats are ones the hardware defines (magic_enum::enum_contains from a table:
/// that scans every enumerator, per texture per draw).
/// BB_VERTEX_IN_PLACE (diagnostics): also what draws read of the vertex data bound in place.
inline bool VertexInPlaceDiag() {
    static const bool on = bb_env("BB_VERTEX_IN_PLACE") &&
                           std::strtoull(bb_env("BB_VERTEX_IN_PLACE"), nullptr, 0) != 0;
    return on;
}

inline bool IsKnownFormat(Gcn::DataFormat data_fmt, Gcn::NumberFormat num_fmt) {
    static const auto tables = [] {
        std::pair<std::array<bool, 256>, std::array<bool, 256>> t{};
        for (const auto value : magic_enum::enum_values<Gcn::DataFormat>()) {
            if (static_cast<u32>(value) < 256) {
                t.first[static_cast<u32>(value)] = true;
            }
        }
        for (const auto value : magic_enum::enum_values<Gcn::NumberFormat>()) {
            if (static_cast<u32>(value) < 256) {
                t.second[static_cast<u32>(value)] = true;
            }
        }
        return t;
    }();
    const u32 d = static_cast<u32>(data_fmt), n = static_cast<u32>(num_fmt);
    return d < 256 && n < 256 && tables.first[d] && tables.second[n];
}

/// BB_INPLACE_READ_CHECK=1 (diagnostics): reads the GPU makes of the game's memory when it runs a
/// draw (stage B bindings not copied by stage A), with the tick they run in; a constant RAM dump
/// checks after its waits that none over its bytes is still pending (a missed wait).
namespace InPlaceReads {
bool Enabled();
void Note(VAddr address, u64 size, u64 tick, u64 hash, u32 kind);
void CheckDump(VAddr address, u64 size, const std::function<bool(u64)>& tick_done);
} // namespace InPlaceReads

} // namespace Vulkan
