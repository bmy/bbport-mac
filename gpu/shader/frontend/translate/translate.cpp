// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The translator core: the stage prologue (what GCN hands a wave in SGPRs and VGPRs at start),
// operand reads with their modifiers (32-bit, 16-bit halves, mixed precision, 64-bit pairs,
// packed pairs), result writes with output modifiers, the fetch shader call, and dispatch by
// instruction category.
#include <algorithm>
#include <numbers>
#include <optional>

#include <magic_enum/magic_enum.hpp>

#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "gcn/sharps.h"
#include "shader/frontend/translate/translate.h"
#include "shader/ir/attribute.h"
#include "shader/ir/condition.h"
#include "shader/ir/reg.h"
#include "shader/ir/reinterpret.h"
#include "shader/frontend/fetch_shader.h"
#include "shader/info.h"
#include "shader/profile.h"
#include "shader/runtime_info.h"

namespace Shader::Gcn {

namespace {

/// Calls set(vgpr, attribute, component) for the barycentrics a fragment shader receives in
/// its first VGPRs (SPI_PS_INPUT_ADDR); the first VGPR after them.
template <typename Set>
IR::VectorReg IterateBarycentrics(const RuntimeInfo& runtime_info, Set&& set) {
    if (runtime_info.hw_stage != HwStage::Fragment) {
        return IR::VectorReg::V0;
    }
    using enum IR::Attribute;
    u32 vgpr{};
    const auto pair = [&](IR::Attribute attribute) {
        set(vgpr++, attribute, 0); // I
        set(vgpr++, attribute, 1); // J
    };
    const auto flags = runtime_info.hw.fs.addr_flags;
    if (flags.persp_sample_ena) {
        pair(BaryCoordSmoothSample);
    }
    if (flags.persp_center_ena) {
        pair(BaryCoordSmooth);
    }
    if (flags.persp_centroid_ena) {
        pair(BaryCoordSmoothCentroid);
    }
    if (flags.persp_pull_model_ena) {
        set(vgpr++, BaryCoordPullModel, 0); // I/W
        set(vgpr++, BaryCoordPullModel, 1); // J/W
        set(vgpr++, BaryCoordPullModel, 2); // 1/W
    }
    if (flags.linear_sample_ena) {
        pair(BaryCoordNoPerspSample);
    }
    if (flags.linear_center_ena) {
        pair(BaryCoordNoPersp);
    }
    if (flags.linear_centroid_ena) {
        pair(BaryCoordNoPerspCentroid);
    }
    if (flags.line_stipple_tex_ena) {
        ++vgpr;
    }
    return IR::VectorReg(vgpr);
}

/// The user data SGPR the driver writes base vertex/instance to (an SPI_SHADER_USER_DATA
/// register offset of the stage); -1 for none.
s8 UserDataRegOf(u16 sgpr_offset, HwStage hw_stage) {
    static constexpr std::array StageUserDataBase{0u, 0x4cu, 0u, 0xccu, 0u, 0x14cu};
    if (!sgpr_offset) {
        return -1;
    }
    const u32 reg = sgpr_offset - StageUserDataBase[u32(hw_stage)];
    ASSERT_MSG(reg < 16, "Out of bounds indirect SGPR copy");
    return reg;
}

/// The 32-bit pattern of an inline constant or literal; nullopt for other operands.
std::optional<u32> ConstantBits(const InstOperand& operand) {
    switch (operand.field) {
    case OperandField::ConstZero:
        return 0U;
    case OperandField::SignedConstIntPos:
        return std::bit_cast<u32>(operand.code - SignedConstIntPosMin + 1);
    case OperandField::SignedConstIntNeg:
        return std::bit_cast<u32>(-s32(operand.code) + SignedConstIntNegMin - 1);
    case OperandField::LiteralConst:
        return operand.code;
    case OperandField::ConstFloatPos_1_0:
        return std::bit_cast<u32>(1.f);
    case OperandField::ConstFloatPos_0_5:
        return std::bit_cast<u32>(0.5f);
    case OperandField::ConstFloatPos_2_0:
        return std::bit_cast<u32>(2.0f);
    case OperandField::ConstFloatPos_4_0:
        return std::bit_cast<u32>(4.0f);
    case OperandField::ConstFloatNeg_0_5:
        return std::bit_cast<u32>(-0.5f);
    case OperandField::ConstFloatNeg_1_0:
        return std::bit_cast<u32>(-1.0f);
    case OperandField::ConstFloatNeg_2_0:
        return std::bit_cast<u32>(-2.0f);
    case OperandField::ConstFloatNeg_4_0:
        return std::bit_cast<u32>(-4.0f);
    case OperandField::Inv2Pi:
        return std::bit_cast<u32>(static_cast<float>(1.0f / (2.0f * std::numbers::pi)));
    default:
        return std::nullopt;
    }
}

/// The same for 64-bit operands: float constants as doubles, literals as the high dword of a
/// double (or a zero-extended integer).
std::optional<u64> ConstantBits64(const InstOperand& operand, bool is_float) {
    switch (operand.field) {
    case OperandField::ConstZero:
        return 0ULL;
    case OperandField::SignedConstIntPos:
        return std::bit_cast<u64>(s64(operand.code) - SignedConstIntPosMin + 1);
    case OperandField::SignedConstIntNeg:
        return std::bit_cast<u64>(-s64(operand.code) + SignedConstIntNegMin - 1);
    case OperandField::LiteralConst:
        return is_float ? u64(operand.code) << 32 : u64(operand.code);
    case OperandField::ConstFloatPos_1_0:
        return std::bit_cast<u64>(1.0);
    case OperandField::ConstFloatPos_0_5:
        return std::bit_cast<u64>(0.5);
    case OperandField::ConstFloatPos_2_0:
        return std::bit_cast<u64>(2.0);
    case OperandField::ConstFloatPos_4_0:
        return std::bit_cast<u64>(4.0);
    case OperandField::ConstFloatNeg_0_5:
        return std::bit_cast<u64>(-0.5);
    case OperandField::ConstFloatNeg_1_0:
        return std::bit_cast<u64>(-1.0);
    case OperandField::ConstFloatNeg_2_0:
        return std::bit_cast<u64>(-2.0);
    case OperandField::ConstFloatNeg_4_0:
        return std::bit_cast<u64>(-4.0);
    default:
        return std::nullopt;
    }
}

} // namespace

Translator::Translator(Info& info_, const RuntimeInfo& runtime_info_, const Profile& profile_)
    : info{info_}, runtime_info{runtime_info_}, profile{profile_},
      next_vgpr_num{runtime_info.props.num_allocated_vgprs} {
    IterateBarycentrics(runtime_info, [this](u32 vgpr, IR::Attribute attribute, u32) {
        vgpr_to_interp[vgpr] = attribute;
    });
}

void Translator::EmitPrologue(IR::Block* first_block) {
    ir = IR::IREmitter(*first_block, first_block->begin());
    ir.Prologue();
    ir.SetExec(ir.Imm1(true));

    IR::ScalarReg dst_sreg = IR::ScalarReg::S0;
    for (u32 i = 0; i < runtime_info.props.num_user_data; i++) {
        ir.SetScalarReg(dst_sreg, ir.GetUserData(dst_sreg));
        ++dst_sreg;
    }

    IR::VectorReg dst_vreg = IR::VectorReg::V0;
    switch (info.sw_stage) {
    case SwStage::Vertex: {
        const auto& vs = runtime_info.sw.vs;
        const s8 base_vertex_sgpr = UserDataRegOf(vs.vertex_sgpr_offset, info.hw_stage);
        if (base_vertex_sgpr != -1) {
            ir.SetScalarReg(IR::ScalarReg(base_vertex_sgpr),
                            ir.GetAttributeU32(IR::Attribute::BaseVertex));
        }
        const s8 base_instance_sgpr = UserDataRegOf(vs.instance_sgpr_offset, info.hw_stage);
        if (base_instance_sgpr != -1) {
            ir.SetScalarReg(IR::ScalarReg(base_instance_sgpr),
                            ir.GetAttributeU32(IR::Attribute::BaseInstance));
        }
        // v0: vertex index (without the base vertex, unless the fetch shader adds it).
        IR::U32 vertex_id = ir.GetAttributeU32(IR::Attribute::VertexId);
        if (base_vertex_sgpr != -1) {
            if (!fetch_data || fetch_data->vertex_offset_sgpr == -1) {
                vertex_id = ir.ISub(vertex_id, ir.GetAttributeU32(IR::Attribute::BaseVertex));
            } else {
                ASSERT_MSG(fetch_data->vertex_offset_sgpr == base_vertex_sgpr,
                           "Fetch shader in indirect draw uses wrong base vertex");
            }
        }
        ir.SetVectorReg(dst_vreg++, vertex_id);

        const u32 input_vgprs = runtime_info.props.num_input_vgprs;
        if (info.hw_stage == HwStage::Local) {
            // v1: relative patch id; v2: unknown.
            if (input_vgprs > 0) {
                ir.SetVectorReg(dst_vreg++, ir.Imm32(0));
            }
            if (input_vgprs > 1) {
                ++dst_vreg;
            }
        } else {
            // v1, v2: instance id divided by step rate 0 and 1.
            const auto stepped_instance = [&](u32 step_rate) {
                if (step_rate != 0) {
                    ir.SetVectorReg(dst_vreg++,
                                    ir.IDiv(ir.GetAttributeU32(IR::Attribute::InstanceId),
                                            ir.Imm32(step_rate)));
                } else {
                    ir.SetVectorReg(dst_vreg++, ir.Imm32(0));
                }
            };
            if (input_vgprs > 0) {
                stepped_instance(vs.step_rate_0);
            }
            if (input_vgprs > 1) {
                stepped_instance(vs.step_rate_1);
            }
        }
        // v3: instance index (without the base instance, unless the fetch shader adds it).
        if (input_vgprs > 2) {
            IR::U32 instance_id = ir.GetAttributeU32(IR::Attribute::InstanceId);
            if (base_instance_sgpr != -1) {
                if (!fetch_data || fetch_data->instance_offset_sgpr == -1) {
                    instance_id =
                        ir.ISub(instance_id, ir.GetAttributeU32(IR::Attribute::BaseInstance));
                } else {
                    ASSERT_MSG(fetch_data->instance_offset_sgpr == base_instance_sgpr,
                               "Fetch shader in indirect draw uses wrong base instance");
                }
            }
            ir.SetVectorReg(dst_vreg++, instance_id);
        }
        break;
    }
    case SwStage::Fragment: {
        // Barycentrics (read by the shader only with per-vertex attribute access), then the
        // inputs SPI_PS_INPUT_ADDR allocates, 0 where SPI_PS_INPUT_ENA does not enable them.
        dst_vreg = IterateBarycentrics(runtime_info, [this](u32 vgpr, IR::Attribute attribute,
                                                            u32 comp) {
            if (profile.supports_amd_shader_explicit_vertex_parameter ||
                profile.supports_fragment_shader_barycentric) {
                ir.SetVectorReg(IR::VectorReg(vgpr), ir.GetAttribute(attribute, comp));
            }
        });
        const auto& fs = runtime_info.hw.fs;
        const auto addr = fs.addr_flags;
        const auto en = fs.en_flags;
        const auto input = [&](bool allocated, bool enabled, auto&& value, IR::Value zero) {
            if (!allocated) {
                return;
            }
            if (enabled) {
                ir.SetVectorReg(dst_vreg++, value());
            } else {
                ir.SetVectorReg(dst_vreg++, IR::U32F32{zero});
            }
        };
        const IR::Value zero_f = ir.Imm32(0.0f);
        const IR::Value zero_u = ir.Imm32(0);
        for (u32 i = 0; i < 3; i++) {
            const bool allocated = i == 0 ? addr.pos_x_float_ena
                                   : i == 1 ? addr.pos_y_float_ena
                                            : addr.pos_z_float_ena;
            const bool enabled = i == 0   ? en.pos_x_float_ena
                                 : i == 1 ? en.pos_y_float_ena
                                          : en.pos_z_float_ena;
            input(allocated, enabled,
                  [&] { return ir.GetAttribute(IR::Attribute::FragCoord, i); }, zero_f);
        }
        input(addr.pos_w_float_ena, en.pos_w_float_ena,
              [&] { return ir.FPRecip(ir.GetAttribute(IR::Attribute::FragCoord, 3)); }, zero_f);
        input(addr.front_face_ena, en.front_face_ena, [&] -> IR::U32F32 {
            const IR::U1 front_face = ir.GetAttributeU1(IR::Attribute::IsFrontFace);
            if (fs.front_face_all_bits) {
                return IR::U32{ir.Select(front_face, ir.Imm32(1), ir.Imm32(0))};
            }
            return IR::F32{ir.Select(front_face, ir.Imm32(1.0f), ir.Imm32(-1.0f))};
        }, zero_u);
        input(addr.ancillary_ena, en.ancillary_ena,
              [&] { return ir.GetAttributeU32(IR::Attribute::PackedAncillary); }, zero_u);
        input(addr.sample_coverage_ena, en.sample_coverage_ena, [&] {
            // The sample mask, 0 for helper invocations.
            const u32 num_samples = std::clamp<u32>(fs.num_samples, 1U, 16U);
            IR::U32 coverage = ir.Imm32(1U);
            if (num_samples > 1) {
                const u32 valid_samples = (1U << num_samples) - 1U;
                coverage = ir.BitwiseAnd(ir.GetAttributeU32(IR::Attribute::SampleMask),
                                         ir.Imm32(valid_samples));
            }
            const IR::U1 is_helper = ir.GetAttributeU1(IR::Attribute::IsHelperInvocation);
            return IR::U32{ir.Select(is_helper, ir.Imm32(0U), coverage)};
        }, zero_u);
        break;
    }
    case SwStage::TessellationControl:
        ir.SetVectorReg(IR::VectorReg::V0, ir.GetAttributeU32(IR::Attribute::PrimitiveId));
        // v1: patch id within the VGT group [7:0], output control point id [12:8].
        ir.SetVectorReg(IR::VectorReg::V1,
                        ir.GetAttributeU32(IR::Attribute::PackedHullInvocationInfo));
        if (runtime_info.sw.tcs.offchip_lds_enable) {
            // (Off-chip tessellation not seen yet.)
            ir.SetScalarReg(dst_sreg++, ir.GetAttributeU32(IR::Attribute::OffChipLdsBase));
        }
        ir.SetScalarReg(dst_sreg++, ir.GetAttributeU32(IR::Attribute::TessFactorsBufferBase));
        break;
    case SwStage::TessellationEval:
        ir.SetVectorReg(IR::VectorReg::V0,
                        ir.GetAttribute(IR::Attribute::TessellationEvaluationPointU));
        ir.SetVectorReg(IR::VectorReg::V1,
                        ir.GetAttribute(IR::Attribute::TessellationEvaluationPointV));
        // v2: the patch id within the VGT group, only used in LDS addresses: 0 (see the hull
        // shader transform). v3: the primitive id.
        ir.SetVectorReg(IR::VectorReg::V2, ir.Imm32(0u));
        ir.SetVectorReg(IR::VectorReg::V3, ir.GetAttributeU32(IR::Attribute::PrimitiveId));
        break;
    case SwStage::Compute:
        for (u32 i = 0; i < 3; i++) {
            ir.SetVectorReg(dst_vreg++, ir.GetAttributeU32(IR::Attribute::LocalInvocationId, i));
        }
        for (u32 i = 0; i < 3; i++) {
            if (runtime_info.hw.cs.tgid_enable[i]) {
                ir.SetScalarReg(dst_sreg++, ir.GetAttributeU32(IR::Attribute::WorkgroupId, i));
            }
        }
        break;
    case SwStage::Geometry:
        // One ES vertex offset per input primitive vertex in v0-v6 (up to 6 with adjacency),
        // the primitive id in v2.
        switch (runtime_info.hw.gs.in_primitive) {
        case ::Gcn::PrimitiveType::AdjTriangleList:
        case ::Gcn::PrimitiveType::AdjTriangleStrip:
            ir.SetVectorReg(IR::VectorReg::V6, ir.Imm32(5u));
            ir.SetVectorReg(IR::VectorReg::V5, ir.Imm32(4u));
            [[fallthrough]];
        case ::Gcn::PrimitiveType::AdjLineList:
        case ::Gcn::PrimitiveType::AdjLineStrip:
            ir.SetVectorReg(IR::VectorReg::V4, ir.Imm32(3u));
            [[fallthrough]];
        case ::Gcn::PrimitiveType::TriangleList:
        case ::Gcn::PrimitiveType::TriangleStrip:
        case ::Gcn::PrimitiveType::RectList:
            ir.SetVectorReg(IR::VectorReg::V3, ir.Imm32(2u));
            [[fallthrough]];
        case ::Gcn::PrimitiveType::LineList:
        case ::Gcn::PrimitiveType::LineStrip:
            ir.SetVectorReg(IR::VectorReg::V1, ir.Imm32(1u));
            [[fallthrough]];
        default:
            ir.SetVectorReg(IR::VectorReg::V0, ir.Imm32(0u));
            break;
        }
        ir.SetVectorReg(IR::VectorReg::V2, ir.GetAttributeU32(IR::Attribute::PrimitiveId));
        break;
    default:
        UNREACHABLE_MSG("Unknown shader stage");
    }
}

/// A VGPR standing for fragment shader LDS at `offset` (per-lane scratch).
IR::VectorReg Translator::GetScratchVgpr(u32 offset) {
    const auto [it, is_new] = vgpr_map.try_emplace(offset);
    if (is_new) {
        ASSERT_MSG(next_vgpr_num < 256, "Out of VGPRs");
        it->second = static_cast<IR::VectorReg>(next_vgpr_num++);
    }
    return it->second;
}

template <typename T>
T Translator::GetSrc(const InstOperand& operand) {
    constexpr bool is_float = std::is_same_v<T, IR::F32>;
    const auto as_t = [&](const IR::U32& value) -> T {
        if constexpr (is_float) {
            return ir.BitCast<IR::F32>(value);
        } else {
            return value;
        }
    };
    T value{};
    if (const auto bits = ConstantBits(operand)) {
        if constexpr (is_float) {
            value = ir.Imm32(std::bit_cast<float>(*bits));
        } else {
            value = ir.Imm32(*bits);
        }
    } else {
        switch (operand.field) {
        case OperandField::ScalarGPR:
            value = ir.GetScalarReg<T>(IR::ScalarReg(operand.code));
            break;
        case OperandField::VectorGPR:
            value = ir.GetVectorReg<T>(IR::VectorReg(operand.code));
            break;
        case OperandField::VccLo:
            value = as_t(ir.GetVccLo());
            break;
        case OperandField::VccHi:
            value = as_t(ir.GetVccHi());
            break;
        case OperandField::M0:
            value = as_t(ir.GetM0());
            break;
        case OperandField::Scc:
            if constexpr (is_float) {
                UNREACHABLE();
            } else {
                value = IR::U32{ir.Select(ir.GetScc(), ir.Imm32(1u), ir.Imm32(0u))};
            }
            break;
        case OperandField::ExecLo:
            value = as_t(
                IR::U32{ir.CompositeExtract(ir.UnpackUint2x32(ir.Ballot(ir.GetExec())), 0)});
            break;
        case OperandField::ExecHi:
            value = as_t(
                IR::U32{ir.CompositeExtract(ir.UnpackUint2x32(ir.Ballot(ir.GetExec())), 1)});
            break;
        case OperandField::Sdwa:
            UNREACHABLE_MSG("unhandled SDWA");
        case OperandField::Dpp:
            UNREACHABLE_MSG("unhandled DPP");
        default:
            UNREACHABLE_MSG("Unexpected operand: {}", std::to_underlying(operand.field));
        }
    }
    // Input modifiers: on the sign bit for integers.
    if constexpr (is_float) {
        if (operand.input_modifier.abs) {
            value = ir.FPAbs(value);
        }
        if (operand.input_modifier.neg) {
            value = ir.FPNeg(value);
        }
    } else {
        if (operand.input_modifier.abs) {
            value = ir.BitwiseAnd(value, ir.Imm32(0x7FFFFFFFu));
        }
        if (operand.input_modifier.neg) {
            value = ir.BitwiseXor(value, ir.Imm32(0x80000000u));
        }
    }
    return value;
}

template IR::U32 Translator::GetSrc<IR::U32>(const InstOperand&);
template IR::F32 Translator::GetSrc<IR::F32>(const InstOperand&);

/// One 16-bit half (OP_SEL) of a register operand, as F32 or zero/sign-extended U32; constants
/// as 32-bit values.
template <typename T, bool is_signed>
T Translator::GetSrc16(const InstOperand& operand) {
    constexpr bool is_float = std::is_same_v<T, IR::F32>;
    constexpr auto number_format = is_float ? ::Gcn::NumberFormat::Float : ::Gcn::NumberFormat::Uint;
    const auto imm = [&](u32 bits) -> T {
        if constexpr (is_float) {
            return ir.Imm32(std::bit_cast<float>(bits));
        } else {
            return ir.Imm32(bits);
        }
    };
    const auto to_u32 = [&](auto value) -> IR::U32 {
        if constexpr (is_float) {
            return ir.BitCast<IR::U32>(value);
        } else {
            return value;
        }
    };
    const auto widen = [&](auto value) -> T {
        if constexpr (is_float) {
            return value;
        } else {
            return ir.BitFieldExtract(ir.BitCast<IR::U32>(value), ir.Imm32(0), ir.Imm32(16),
                                      is_signed);
        }
    };
    const u32 half = operand.op_sel.op_sel ? 1 : 0;
    T value{};
    if (const auto bits = ConstantBits(operand)) {
        value = imm(*bits);
    } else {
        switch (operand.field) {
        case OperandField::ScalarGPR: {
            const auto reg = ir.GetScalarReg<T>(IR::ScalarReg(operand.code));
            value = widen(
                IR::F32{ir.CompositeExtract(ir.Unpack2x16(number_format, to_u32(reg)), half)});
            break;
        }
        case OperandField::VectorGPR: {
            const auto reg = ir.GetVectorReg<T>(IR::VectorReg(operand.code));
            value = widen(
                IR::F32{ir.CompositeExtract(ir.Unpack2x16(number_format, to_u32(reg)), half)});
            break;
        }
        case OperandField::VccLo:
            value = widen(
                IR::F32{ir.CompositeExtract(ir.Unpack2x16(number_format, ir.GetVccLo()), half)});
            break;
        case OperandField::Sdwa:
            LOG_ERROR(Render_Recompiler, "unhandled SDWA");
            value = imm(0U);
            break;
        case OperandField::Dpp:
            LOG_ERROR(Render_Recompiler, "unhandled DPP");
            value = imm(0U);
            break;
        default:
            UNREACHABLE_MSG("Unexpected operand: {}", std::to_underlying(operand.field));
        }
    }
    if constexpr (is_float) {
        if (operand.input_modifier.abs) {
            value = ir.FPAbs(value);
        }
        if (operand.input_modifier.neg) {
            value = ir.FPNeg(value);
        }
    } else {
        if (operand.input_modifier.abs) {
            value = ir.BitwiseAnd(value, ir.Imm32(0x7FFFFFFFu));
        }
        if (operand.input_modifier.neg) {
            value = ir.BitwiseXor(value, ir.Imm32(0x80000000u));
        }
    }
    return value;
}

template IR::U32 Translator::GetSrc16<IR::U32, false>(const InstOperand&);
template IR::U32 Translator::GetSrc16<IR::U32, true>(const InstOperand&);
template IR::F32 Translator::GetSrc16<IR::F32, false>(const InstOperand&);

/// A V_MAD_MIX operand: F32, or (OP_SEL_HI) one F16 half selected by OP_SEL. NEG_HI is abs.
IR::F32 Translator::GetSrcMix(const InstOperand& operand) {
    const bool is_f16 = operand.op_sel.op_sel_hi;
    const u32 half = operand.op_sel.op_sel ? 1 : 0;
    const auto from_reg = [&](auto reg) -> IR::F32 {
        constexpr bool is_sgpr = std::same_as<decltype(reg), IR::ScalarReg>;
        IR::U32 raw;
        if constexpr (is_sgpr) {
            raw = ir.GetScalarReg<IR::U32>(reg);
        } else {
            raw = ir.GetVectorReg<IR::U32>(reg);
        }
        if (!is_f16) {
            // (The raw read above stays unused.)
            if constexpr (is_sgpr) {
                return ir.GetScalarReg<IR::F32>(reg);
            } else {
                return ir.GetVectorReg<IR::F32>(reg);
            }
        }
        return IR::F32{ir.CompositeExtract(ir.Unpack2x16(::Gcn::NumberFormat::Float, raw), half)};
    };
    IR::F32 value{};
    if (const auto bits = ConstantBits(operand)) {
        value = ir.Imm32(std::bit_cast<float>(*bits));
    } else {
        switch (operand.field) {
        case OperandField::ScalarGPR:
            value = from_reg(IR::ScalarReg(operand.code));
            break;
        case OperandField::VectorGPR:
            value = from_reg(IR::VectorReg(operand.code));
            break;
        case OperandField::VccLo:
            if (!is_f16) {
                value = ir.BitCast<IR::F32>(ir.GetVccLo());
            } else {
                value = IR::F32{ir.CompositeExtract(
                    ir.Unpack2x16(::Gcn::NumberFormat::Float, ir.GetVccLo()), half)};
            }
            break;
        default:
            UNREACHABLE_MSG("Unexpected operand: {}", std::to_underlying(operand.field));
        }
    }
    if (operand.input_modifier.neg_hi) {
        value = ir.FPAbs(value);
    }
    if (operand.input_modifier.neg) {
        value = ir.FPNeg(value);
    }
    return value;
}

/// A register pair (lo, hi) or constant as U64/F64. Integer abs/neg modifiers act on bit 63.
template <typename T>
T Translator::GetSrc64(const InstOperand& operand) {
    constexpr bool is_float = std::is_same_v<T, IR::F64>;
    const auto pack = [&](const IR::Value& pair) -> T {
        if constexpr (is_float) {
            return ir.PackDouble2x32(pair);
        } else {
            return ir.PackUint2x32(pair);
        }
    };
    T value{};
    if (const auto bits = ConstantBits64(operand, is_float)) {
        if constexpr (is_float) {
            value = ir.Imm64(std::bit_cast<double>(*bits));
        } else {
            value = ir.Imm64(*bits);
        }
    } else {
        switch (operand.field) {
        case OperandField::ScalarGPR: {
            const auto lo = ir.GetScalarReg(IR::ScalarReg(operand.code));
            const auto hi = ir.GetScalarReg(IR::ScalarReg(operand.code + 1));
            value = pack(ir.CompositeConstruct(lo, hi));
            break;
        }
        case OperandField::VectorGPR: {
            const auto lo = ir.GetVectorReg(IR::VectorReg(operand.code));
            const auto hi = ir.GetVectorReg(IR::VectorReg(operand.code + 1));
            value = pack(ir.CompositeConstruct(lo, hi));
            break;
        }
        case OperandField::VccLo:
            value = pack(ir.CompositeConstruct(ir.GetVccLo(), ir.GetVccHi()));
            break;
        case OperandField::ExecLo:
            if constexpr (is_float) {
                UNREACHABLE();
            } else {
                value = ir.Ballot(ir.GetExec());
            }
            break;
        default:
            UNREACHABLE_MSG("Unexpected operand: {}", std::to_underlying(operand.field));
        }
    }
    if constexpr (is_float) {
        if (operand.input_modifier.abs) {
            value = ir.FPAbs(value);
        }
        if (operand.input_modifier.neg) {
            value = ir.FPNeg(value);
        }
    } else {
        const auto modify_high = [&](auto&& op) {
            const auto halves = ir.UnpackUint2x32(value);
            const auto lo = IR::U32{ir.CompositeExtract(halves, 0)};
            const auto hi = IR::U32{ir.CompositeExtract(halves, 1)};
            const auto new_hi = op(hi);
            value = ir.PackUint2x32(ir.CompositeConstruct(lo, new_hi));
        };
        if (operand.input_modifier.abs) {
            modify_high([&](const IR::U32& hi) { return ir.BitwiseAnd(hi, ir.Imm32(0x7FFFFFFFu)); });
        }
        if (operand.input_modifier.neg) {
            modify_high([&](const IR::U32& hi) { return ir.BitwiseXor(hi, ir.Imm32(0x80000000u)); });
        }
    }
    return value;
}

template IR::U64 Translator::GetSrc64<IR::U64>(const InstOperand&);
template IR::F64 Translator::GetSrc64<IR::F64>(const InstOperand&);

/// A VOP3P operand: the halves OP_SEL and OP_SEL_HI select for the low and high lanes. A
/// constant goes to a lane whose select is 0 (the other gets 0). NEG/NEG_HI negate each lane.
template <typename T, bool is_signed>
pk_type<T> Translator::GetSrcPk(const InstOperand& operand) {
    constexpr bool is_float = std::is_same_v<T, IR::F32>;
    constexpr auto number_format = is_float ? ::Gcn::NumberFormat::Float : ::Gcn::NumberFormat::Uint;
    const auto imm = [&](u32 bits) -> pk_type<T> {
        if constexpr (is_float) {
            const auto value = ir.Imm32(std::bit_cast<float>(bits));
            return {operand.op_sel.op_sel ? ir.Imm32(0.f) : value,
                    operand.op_sel.op_sel_hi ? ir.Imm32(0.f) : value};
        } else {
            const auto value = ir.Imm32(bits);
            return {operand.op_sel.op_sel ? ir.Imm32(0U) : value,
                    operand.op_sel.op_sel_hi ? ir.Imm32(0U) : value};
        }
    };
    const auto widen = [&](auto value) -> T {
        if constexpr (is_float) {
            return value;
        } else {
            return ir.BitFieldExtract(ir.BitCast<IR::U32>(value), ir.Imm32(0), ir.Imm32(16),
                                      is_signed);
        }
    };
    const auto halves = [&](auto value) -> pk_type<T> {
        const auto unpacked = ir.Unpack2x16(number_format, value);
        return {widen(IR::F32{ir.CompositeExtract(unpacked, operand.op_sel.op_sel)}),
                widen(IR::F32{ir.CompositeExtract(unpacked, operand.op_sel.op_sel_hi)})};
    };
    pk_type<T> value{};
    if (operand.field == OperandField::Inv2Pi) {
        // (In float arithmetic here.)
        value = imm(std::bit_cast<u32>(1.0f / (2.0f * std::numbers::pi_v<float>)));
    } else if (const auto bits = ConstantBits(operand)) {
        value = imm(*bits);
    } else {
        switch (operand.field) {
        case OperandField::ScalarGPR:
            value = halves(ir.GetScalarReg<IR::U32>(IR::ScalarReg(operand.code)));
            break;
        case OperandField::VectorGPR:
            value = halves(ir.GetVectorReg<IR::U32>(IR::VectorReg(operand.code)));
            break;
        case OperandField::VccLo:
            value = halves(ir.GetVccLo());
            break;
        default:
            UNREACHABLE_MSG("Unexpected operand: {}", std::to_underlying(operand.field));
        }
    }
    if constexpr (is_float) {
        if (operand.input_modifier.neg) {
            value.first = ir.FPNeg(value.first);
        }
        if (operand.input_modifier.neg_hi) {
            value.second = ir.FPNeg(value.second);
        }
    } else {
        if (operand.input_modifier.neg) {
            value.first = ir.INeg(value.first);
        }
        if (operand.input_modifier.neg_hi) {
            value.second = ir.INeg(value.second);
        }
    }
    return value;
}

template pk_type<IR::U32> Translator::GetSrcPk<IR::U32, true>(const InstOperand&);
template pk_type<IR::U32> Translator::GetSrcPk<IR::U32, false>(const InstOperand&);
template pk_type<IR::F32> Translator::GetSrcPk<IR::F32, false>(const InstOperand&);

/// Writes a result; floats get the output modifiers (multiplier, clamp).
void Translator::SetDst(const InstOperand& operand, const IR::U32F32& value) {
    IR::U32F32 result = value;
    if (value.Type() == IR::Type::F32) {
        if (operand.output_modifier.multiplier != 0.f) {
            result = ir.FPMul(result, ir.Imm32(operand.output_modifier.multiplier));
        }
        if (operand.output_modifier.clamp) {
            result = ir.FPSaturate(result);
        }
    }
    switch (operand.field) {
    case OperandField::ScalarGPR:
        return ir.SetScalarReg(IR::ScalarReg(operand.code), result);
    case OperandField::VectorGPR:
        return ir.SetVectorReg(IR::VectorReg(operand.code), result);
    case OperandField::VccLo:
        return ir.SetVccLo(result);
    case OperandField::VccHi:
        return ir.SetVccHi(result);
    case OperandField::M0:
        return ir.SetM0(result);
    default:
        UNREACHABLE_MSG("Unknown field {}", u32(operand.field));
    }
}

/// Writes one 16-bit half (OP_SEL) of the destination, keeping the other; floats are converted
/// to F16, integer clamps saturate to 16 bits.
template <bool is_signed>
void Translator::SetDst16(const InstOperand& operand, const IR::U32F32& value) {
    IR::U32F32 result = value;
    if (value.Type() == IR::Type::F32) {
        if (operand.output_modifier.multiplier != 0.f) {
            result = ir.FPMul(result, ir.Imm32(operand.output_modifier.multiplier));
        }
        if (operand.output_modifier.clamp) {
            result = ir.FPSaturate(result);
        }
    } else if (operand.output_modifier.clamp) {
        if constexpr (is_signed) {
            result = ir.SClamp(result, ir.Imm32(-32768), ir.Imm32(32767));
        } else {
            result = ir.UMin(result, ir.Imm32(0xFFFF));
        }
    }
    const auto to_bits16 = [&](auto v) -> IR::U32 {
        if (v.Type() == IR::Type::F32) {
            return ir.UConvert(32, ir.BitCast<IR::U16>(IR::F16{ir.FPConvert(16, v)}));
        }
        if (v.Type() == IR::Type::U32) {
            return v;
        }
        UNREACHABLE();
    };
    const IR::U32 offset = ir.Imm32(operand.op_sel.op_sel ? 16 : 0);
    switch (operand.field) {
    case OperandField::ScalarGPR: {
        const auto previous = ir.GetScalarReg<IR::U32>(IR::ScalarReg(operand.code));
        const auto bits = to_bits16(result);
        return ir.SetScalarReg(IR::ScalarReg(operand.code),
                               ir.BitFieldInsert(previous, bits, offset, ir.Imm32(16)));
    }
    case OperandField::VectorGPR: {
        const auto previous = ir.GetVectorReg<IR::U32>(IR::VectorReg(operand.code));
        const auto bits = to_bits16(result);
        return ir.SetVectorReg(IR::VectorReg(operand.code),
                               ir.BitFieldInsert(previous, bits, offset, ir.Imm32(16)));
    }
    default:
        UNREACHABLE_MSG("Unexpected operand: {}", std::to_underlying(operand.field));
    }
}

template void Translator::SetDst16<false>(const InstOperand&, const IR::U32F32& value);
template void Translator::SetDst16<true>(const InstOperand&, const IR::U32F32& value);

/// Writes a 64-bit result to a register pair (high first), VCC, or EXEC (as lane mask).
void Translator::SetDst64(const InstOperand& operand, const IR::U64F64& value_raw) {
    IR::U64F64 value = value_raw;
    const bool is_float = value_raw.Type() == IR::Type::F64 || value_raw.Type() == IR::Type::F32;
    if (is_float) {
        if (operand.output_modifier.multiplier != 0.f) {
            value = ir.FPMul(value, ir.Imm64(f64(operand.output_modifier.multiplier)));
        }
        if (operand.output_modifier.clamp) {
            value = ir.FPSaturate(value);
        }
    }
    const auto split = [&] -> std::pair<IR::U32, IR::U32> {
        const IR::Value halves{is_float ? ir.UnpackDouble2x32(IR::F64{value})
                                        : ir.UnpackUint2x32(IR::U64{value})};
        const IR::U32 lo{ir.CompositeExtract(halves, 0U)};
        const IR::U32 hi{ir.CompositeExtract(halves, 1U)};
        return {lo, hi};
    };
    switch (operand.field) {
    case OperandField::ScalarGPR: {
        const auto [lo, hi] = split();
        ir.SetScalarReg(IR::ScalarReg(operand.code + 1), hi);
        ir.SetScalarReg(IR::ScalarReg(operand.code), lo);
        break;
    }
    case OperandField::VectorGPR: {
        const auto [lo, hi] = split();
        ir.SetVectorReg(IR::VectorReg(operand.code + 1), hi);
        ir.SetVectorReg(IR::VectorReg(operand.code), lo);
        break;
    }
    case OperandField::VccLo: {
        const auto [lo, hi] = split();
        ir.SetVccLo(lo);
        ir.SetVccHi(hi);
        break;
    }
    case OperandField::ExecLo:
        ir.SetExec(ir.InverseBallot(value));
        break;
    default:
        UNREACHABLE_MSG("Unexpected operand: {}", std::to_underlying(operand.field));
    }
}

/// Packs two 16-bit lanes into the destination (clamped per lane if asked).
template <typename T, bool is_signed>
void Translator::SetDstPk(const InstOperand& operand, const pk_type<T>& value) {
    pk_type<T> v = value;
    if (operand.output_modifier.clamp) {
        if constexpr (std::is_same_v<T, IR::F32>) {
            v = {ir.FPSaturate(v.first), ir.FPSaturate(v.second)};
        } else if constexpr (is_signed) {
            const auto lower = ir.Imm32(-32768);
            const auto upper = ir.Imm32(32767);
            v = {ir.SClamp(v.first, lower, upper), ir.SClamp(v.second, lower, upper)};
        } else {
            const auto max = ir.Imm32(0xFFFF);
            v = {ir.UMin(v.first, max), ir.UMin(v.second, max)};
        }
    }
    IR::U32 packed{};
    if constexpr (std::is_same_v<T, IR::F32>) {
        packed = ir.Pack2x16(::Gcn::NumberFormat::Float, ir.CompositeConstruct(v.first, v.second));
    } else {
        packed = ir.Pack2x16(::Gcn::NumberFormat::Uint,
                             ir.CompositeConstruct(ir.BitCast<IR::F32, IR::U32>(v.first),
                                                   ir.BitCast<IR::F32, IR::U32>(v.second)));
    }
    SetDst(operand, packed);
}

template void Translator::SetDstPk<IR::U32, false>(const InstOperand& operand,
                                                   const pk_type<IR::U32>& value);
template void Translator::SetDstPk<IR::U32, true>(const InstOperand& operand,
                                                  const pk_type<IR::U32>& value);
template void Translator::SetDstPk<IR::F32, false>(const InstOperand& operand,
                                                   const pk_type<IR::F32>& value);

/// S_SWAPPC_B64 to the fetch shader: the vertex attributes it loads become input attributes
/// (converted and swizzled by their V#s) written to its destination VGPRs.
void Translator::EmitFetch(const GcnInst& inst) {
    const auto code_sgpr_base = inst.src[0].code;
    info.has_fetch_shader = true;
    info.fetch_shader_sgpr_base = code_sgpr_base;
    fetch_data = ParseFetchShader(info);
    ASSERT(fetch_data.has_value());

    if (EmulatorSettings.IsDumpShaders()) {
        using namespace Common::FS;
        const auto dir = GetUserPath(PathType::ShaderDir) / "dumps";
        std::filesystem::create_directories(dir);
        const IOFile file{dir / fmt::format("vs_{:#018x}.fetch.bin", info.pgm_hash),
                          FileAccessMode::Create};
        file.WriteRaw<u8>(GetFetchShaderCode(info, code_sgpr_base), fetch_data->size);
    }

    for (const auto& attrib : fetch_data->attributes) {
        const IR::Attribute attribute{IR::Attribute::Param0 + attrib.semantic};
        IR::VectorReg dst{attrib.dest_vgpr};
        const auto buffer = attrib.GetSharp(info);
        const auto values =
            ir.CompositeConstruct(ir.GetAttribute(attribute, 0), ir.GetAttribute(attribute, 1),
                                  ir.GetAttribute(attribute, 2), ir.GetAttribute(attribute, 3));
        const auto converted =
            IR::ApplyReadNumberConversionVec4(ir, values, buffer.GetNumberConversion());
        const auto swizzled = ApplySwizzle(ir, converted, buffer.DstSelect());
        for (u32 i = 0; i < 4; i++) {
            ir.SetVectorReg(dst++, IR::F32{ir.CompositeExtract(swizzled, i)});
        }
    }
}

void Translator::LogMissingOpcode(const GcnInst& inst) {
    LOG_ERROR(Render_Recompiler, "Unknown opcode {} ({}, category = {})",
              GcnIsa::NameOf(inst.opcode), u32(inst.opcode), magic_enum::enum_name(inst.category));
    info.translation_failed = true;
}

void Translator::Translate(IR::Block* block, u32 start_pc, IR::Condition cond,
                           std::span<const GcnInst> inst_list) {
    if (inst_list.empty()) {
        return;
    }
    ir = IR::IREmitter{*block, block->begin()};
    pc = start_pc;
    for (const auto& inst : inst_list) {
        pc += inst.length;
        if (inst.opcode == Opcode::S_SWAPPC_B64) {
            ASSERT(info.hw_stage == HwStage::Vertex || info.hw_stage == HwStage::Export ||
                   info.hw_stage == HwStage::Local);
            EmitFetch(inst);
            continue;
        }
        TranslateInstruction(inst);
    }
    if (cond != IR::Condition::True && cond != IR::Condition::False) {
        block->branch_cond = ir.ConditionRef(ir.Condition(cond));
    }
}

void Translator::TranslateInstruction(const GcnInst& inst) {
    switch (inst.category) {
    case InstCategory::DataShare:
        return EmitDataShare(inst);
    case InstCategory::VectorInterpolation:
        return EmitVectorInterpolation(inst);
    case InstCategory::ScalarMemory:
        return EmitScalarMemory(inst);
    case InstCategory::VectorMemory:
        return EmitVectorMemory(inst);
    case InstCategory::Export:
        return EmitExport(inst);
    case InstCategory::FlowControl:
        return EmitFlowControl(inst);
    case InstCategory::ScalarALU:
        return EmitScalarAlu(inst);
    case InstCategory::VectorALU:
        return EmitVectorAlu(inst);
    case InstCategory::DebugProfile:
        return;
    default:
        UNREACHABLE();
    }
}

} // namespace Shader::Gcn
