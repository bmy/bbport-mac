// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: the exports to Position1..3, which carry system outputs (clip and
// cull distances, point size, render target and viewport index) as the stage's output map says.
#pragma once

#include <magic_enum/magic_enum.hpp>

#include "common/logging/log.h"
#include "shader/ir/ir_emitter.h"
#include "shader/runtime_info.h"

namespace Shader::IR {

template <typename StageRuntimeInfo>
inline void ExportPosition(IREmitter& ir, const StageRuntimeInfo& stage,
                           bool tess_emulated_primitive, Attribute attribute, u32 comp,
                           const F32& value) {
    if (attribute == Attribute::Position0) {
        ir.SetAttribute(attribute, value, comp);
        return;
    }
    const u32 index = u32(attribute) - u32(Attribute::Position1);
    const auto output = stage.outputs[index][comp];
    if constexpr (std::is_same_v<StageRuntimeInfo, HwVertexRuntimeInfo>) {
        // These belong to the last pre-rasterization stage: with rect and quad lists drawn by a
        // tessellation stage after this one they are not passed on (not done yet).
        const bool last_stage_only = output == Output::PointSize ||
                                     output == Output::RenderTargetIndex ||
                                     output == Output::ViewportIndex;
        if (tess_emulated_primitive && last_stage_only) {
            LOG_WARNING(Render, "{} exported by a vertex shader while primitives are emulated by "
                                "tessellation: not passed on",
                        magic_enum::enum_name(output));
            return;
        }
    }
    if (output >= Output::ClipDist0 && output <= Output::ClipDist7) {
        ir.SetAttribute(Attribute::ClipDistance, value, u32(output) - u32(Output::ClipDist0));
        return;
    }
    if (output >= Output::CullDist0 && output <= Output::CullDist7) {
        ir.SetAttribute(Attribute::CullDistance, value, u32(output) - u32(Output::CullDist0));
        return;
    }
    switch (output) {
    case Output::PointSize:
        ir.SetAttribute(Attribute::PointSize, value);
        break;
    case Output::RenderTargetIndex:
        ir.SetAttribute(Attribute::RenderTargetIndex, value);
        break;
    case Output::ViewportIndex:
        ir.SetAttribute(Attribute::ViewportIndex, value);
        break;
    case Output::None:
        LOG_WARNING(Render_Recompiler, "The {} component of {} isn't mapped, skipping",
                    "xyzw"[comp], NameOf(attribute));
        break;
    default:
        UNREACHABLE_MSG("Unhandled output {} on attribute {}", u32(output), NameOf(attribute));
    }
}

} // namespace Shader::IR
