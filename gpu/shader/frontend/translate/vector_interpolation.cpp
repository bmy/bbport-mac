// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// VINTRP: attribute interpolation. Normally the host interpolates (P1 is dropped, P2 reads the
// attribute and records the qualifier its barycentric VGPR stands for). Hosts that need it
// interpolate manually from the per-vertex values; V_INTERP_MOV reads one vertex's value (with
// explicit vertex parameters or barycentric support), or the flat value.
#include "shader/frontend/translate/translate.h"
#include "shader/profile.h"

namespace Shader::Gcn {

namespace {

using Interpolation = Info::Interpolation;

Interpolation InterpolationOf(IR::Attribute barycentrics) {
    switch (barycentrics) {
    case IR::Attribute::BaryCoordNoPersp:
        return {Qualifier::NoPerspective, Qualifier::None};
    case IR::Attribute::BaryCoordNoPerspCentroid:
        return {Qualifier::NoPerspective, Qualifier::Centroid};
    case IR::Attribute::BaryCoordNoPerspSample:
        return {Qualifier::NoPerspective, Qualifier::Sample};
    case IR::Attribute::BaryCoordSmooth:
    case IR::Attribute::BaryCoordPullModel:
        return {Qualifier::Smooth, Qualifier::None};
    case IR::Attribute::BaryCoordSmoothCentroid:
        return {Qualifier::Smooth, Qualifier::Centroid};
    case IR::Attribute::BaryCoordSmoothSample:
        return {Qualifier::Smooth, Qualifier::Sample};
    default:
        UNREACHABLE_MSG("Unhandled barycentric attribute {}", NameOf(barycentrics));
    }
}

/// The values of inputs the vertex shader does not write (SPI_PS_INPUT_CNTL default).
constexpr std::array DefaultValues = {
    std::array{0.0f, 0.0f, 0.0f, 0.0f},
    std::array{0.0f, 0.0f, 0.0f, 1.0f},
    std::array{1.0f, 1.0f, 1.0f, 0.0f},
    std::array{1.0f, 1.0f, 1.0f, 1.0f},
};

} // namespace

void Translator::EmitVectorInterpolation(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::V_INTERP_P1_F32:
        return V_INTERP_P1_F32(inst);
    case Opcode::V_INTERP_P2_F32:
        return V_INTERP_P2_F32(inst);
    case Opcode::V_INTERP_MOV_F32:
        return V_INTERP_MOV_F32(inst);
    default:
        LogMissingOpcode(inst);
    }
}

/// VDST = (P1 - P0) * I + P0 (manual interpolation only).
void Translator::V_INTERP_P1_F32(const GcnInst& inst) {
    if (!profile.needs_manual_interpolation) {
        return;
    }
    const u32 index = inst.control.vintrp.attr;
    if (runtime_info.hw.fs.inputs[index].IsDefault()) {
        return;
    }
    const IR::Attribute attribute = IR::Attribute::Param0 + index;
    const u32 chan = inst.control.vintrp.chan;
    const IR::F32 p0 = ir.GetAttribute(attribute, chan, 0);
    const IR::F32 p1 = ir.GetAttribute(attribute, chan, 1);
    const IR::F32 i = GetSrc<IR::F32>(inst.src[0]);
    SetDst(inst.dst[0], ir.FPFma(ir.FPSub(p1, p0), i, p0));
}

/// VDST = (P2 - P0) * J + VDST, or the host-interpolated attribute.
void Translator::V_INTERP_P2_F32(const GcnInst& inst) {
    const u32 index = inst.control.vintrp.attr;
    const IR::Attribute attribute = IR::Attribute::Param0 + index;
    const u32 chan = inst.control.vintrp.chan;
    const auto& input = runtime_info.hw.fs.inputs[index];
    auto& interp = info.fs_interpolation[index];
    if (input.IsDefault()) {
        SetDst(inst.dst[0], ir.Imm32(DefaultValues[input.default_value][chan]));
        return;
    }
    ASSERT(!input.is_flat);
    if (!profile.needs_manual_interpolation) {
        interp = InterpolationOf(vgpr_to_interp[inst.src[0].code]);
        SetDst(inst.dst[0], ir.GetAttribute(attribute, chan));
        return;
    }
    const IR::F32 p0 = ir.GetAttribute(attribute, chan, 0);
    const IR::F32 p2 = ir.GetAttribute(attribute, chan, 2);
    const IR::F32 j = GetSrc<IR::F32>(inst.src[0]);
    const IR::F32 result = ir.FPFma(ir.FPSub(p2, p0), j, GetSrc<IR::F32>(inst.dst[0]));
    interp.primary = Qualifier::PerVertex;
    SetDst(inst.dst[0], result);
}

/// One vertex's value: selector 0 = P10, 1 = P20 (relative to P0), 2 = P0.
void Translator::V_INTERP_MOV_F32(const GcnInst& inst) {
    const u32 index = inst.control.vintrp.attr;
    const IR::Attribute attribute = IR::Attribute::Param0 + index;
    const u32 chan = inst.control.vintrp.chan;
    const auto& input = runtime_info.hw.fs.inputs[index];
    auto& interp = info.fs_interpolation[index];
    const u32 select = inst.src[0].code;
    ASSERT_MSG(select < 3, "Invalid V_INTERP_MOV_F32 selector {}", select);
    const bool per_vertex = profile.supports_amd_shader_explicit_vertex_parameter ||
                            profile.supports_fragment_shader_barycentric;
    ASSERT(input.is_flat || select == 2 || per_vertex);
    if (!per_vertex) {
        interp.primary = Qualifier::Flat;
        SetDst(inst.dst[0], ir.GetAttribute(attribute, chan));
        return;
    }
    interp.primary = Qualifier::PerVertex;
    IR::F32 result = ir.GetAttribute(attribute, chan, (select + 1) % 3);
    if (select != 2 && !input.IsPassthrough()) {
        const IR::F32 p0 = ir.GetAttribute(attribute, chan, 0);
        result = ir.FPSub(result, p0);
    }
    SetDst(inst.dst[0], result);
}

} // namespace Shader::Gcn
