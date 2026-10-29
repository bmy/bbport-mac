// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/attribute.h"

#include <array>
#include <string_view>

namespace Shader::IR {

std::string NameOf(Attribute attribute) {
    const u64 value = u64(attribute);
    if (IsMrt(attribute)) {
        return fmt::format("RenderTarget{}", value);
    }
    if (IsPosition(attribute)) {
        return fmt::format("Position{}", value - u64(Attribute::Position0));
    }
    if (IsParam(attribute)) {
        return fmt::format("Param{}", value - u64(Attribute::Param0));
    }
    switch (attribute) {
    case Attribute::Depth:
        return "Depth";
    case Attribute::Null:
        return "Null";
    default:
        break;
    }
    static constexpr std::array<std::string_view, 37> system_values{
        "ClipDistance",
        "CullDistance",
        "RenderTargetIndex",
        "ViewportIndex",
        "VertexId",
        "PrimitiveId",
        "InstanceId",
        "IsFrontFace",
        "SampleIndex",
        "GlobalInvocationId",
        "WorkgroupId",
        "WorkgroupIndex",
        "LocalInvocationId",
        "LocalInvocationIndex",
        "FragCoord",
        "BaryCoordNoPersp",
        "BaryCoordNoPerspCentroid",
        "BaryCoordNoPerspSample",
        "BaryCoordSmooth",
        "BaryCoordSmoothCentroid",
        "BaryCoordSmoothSample",
        "BaryCoordPullModel",
        "InvocationId",
        "PatchVertices",
        "TessellationEvaluationPointU",
        "TessellationEvaluationPointV",
        "PackedHullInvocationInfo",
        "OffChipLdsBase",
        "TessFactorsBufferBase",
        "PointSize",
        "StencilRef",
        "SampleMask",
        "PackedAncillary",
        "SubgroupLtMask",
        "BaseVertex",
        "BaseInstance",
        "IsHelperInvocation",
    };
    static_assert(system_values.size() == u64(Attribute::Max) - u64(Attribute::ClipDistance));
    const u64 index = value - u64(Attribute::ClipDistance);
    if (value >= u64(Attribute::ClipDistance) && index < system_values.size()) {
        return std::string{system_values[index]};
    }
    return fmt::format("<reserved attribute {}>", value);
}

} // namespace Shader::IR
