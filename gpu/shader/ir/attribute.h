// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: shader inputs and outputs by meaning: export targets (render
// targets, depth, positions, parameters) and system values.
#pragma once

#include <string>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/types.h"

namespace Shader::IR {

enum class Attribute : u64 {
    // Export targets (their GCN export numbers).
    RenderTarget0 = 0,
    RenderTarget1 = 1,
    RenderTarget2 = 2,
    RenderTarget3 = 3,
    RenderTarget4 = 4,
    RenderTarget5 = 5,
    RenderTarget6 = 6,
    RenderTarget7 = 7,
    Depth = 8,
    Null = 9,
    Position0 = 12,
    Position1 = 13,
    Position2 = 14,
    Position3 = 15,
    Param0 = 32,
    Param31 = Param0 + 31,
    // System values.
    ClipDistance = 64,
    CullDistance,
    RenderTargetIndex,
    ViewportIndex,
    VertexId,
    PrimitiveId,
    InstanceId,
    IsFrontFace,
    SampleIndex,
    GlobalInvocationId,
    WorkgroupId,
    WorkgroupIndex,
    LocalInvocationId,
    LocalInvocationIndex,
    FragCoord,
    BaryCoordNoPersp,
    BaryCoordNoPerspCentroid,
    BaryCoordNoPerspSample,
    BaryCoordSmooth,
    BaryCoordSmoothCentroid,
    BaryCoordSmoothSample,
    BaryCoordPullModel,
    InvocationId, ///< the control point in a patch, or the instance of a geometry shader
    PatchVertices,
    TessellationEvaluationPointU,
    TessellationEvaluationPointV,
    PackedHullInvocationInfo, ///< patch id in the VGT and invocation id
    OffChipLdsBase,
    TessFactorsBufferBase,
    PointSize,
    StencilRef,
    SampleMask,
    PackedAncillary,
    SubgroupLtMask,
    BaseVertex,
    BaseInstance,
    IsHelperInvocation,
    Max,
};
static_assert(u64(Attribute::IsHelperInvocation) == 100);

constexpr size_t NumAttributes = static_cast<size_t>(Attribute::Max);
constexpr size_t NumRenderTargets = 8;
constexpr size_t NumParams = 32;

constexpr bool IsPosition(Attribute attribute) noexcept {
    return attribute >= Attribute::Position0 && attribute <= Attribute::Position3;
}
constexpr bool IsTessCoord(Attribute attribute) noexcept {
    return attribute >= Attribute::TessellationEvaluationPointU &&
           attribute <= Attribute::TessellationEvaluationPointV;
}
constexpr bool IsParam(Attribute attribute) noexcept {
    return attribute >= Attribute::Param0 && attribute <= Attribute::Param31;
}
constexpr bool IsMrt(Attribute attribute) noexcept {
    return attribute >= Attribute::RenderTarget0 && attribute <= Attribute::RenderTarget7;
}
constexpr bool IsBarycentricCoord(Attribute attribute) noexcept {
    return attribute >= Attribute::BaryCoordNoPersp &&
           attribute <= Attribute::BaryCoordSmoothSample;
}

[[nodiscard]] std::string NameOf(Attribute attribute);

/// The export target `num` places on (within render targets, positions and parameters).
[[nodiscard]] constexpr Attribute operator+(Attribute attr, int num) {
    const int result = int(attr) + num;
    ASSERT_MSG(result >= int(Attribute::RenderTarget0) && result <= int(Attribute::Param31),
               "attribute {} + {} out of range", int(attr), num);
    return Attribute(result);
}

} // namespace Shader::IR

template <>
struct fmt::formatter<Shader::IR::Attribute> : fmt::formatter<std::string> {
    auto format(Shader::IR::Attribute attribute, format_context& ctx) const {
        return fmt::formatter<std::string>::format(Shader::IR::NameOf(attribute), ctx);
    }
};
