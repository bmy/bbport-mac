// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Finds every instruction that uses a resource and the dwords its sharps were read from. Some
// games edit a sharp in the shader before use (a stride bit forced on, a cube map read as a 2D
// array, sampler clamps or anisotropy changed); those edits are recognized here and recorded as
// post operations, so the binding reproduces them from the original sharp.
#include <optional>

#include "common/assert.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/passes/resource_pass.h"
#include "shader/ir/program.h"
#include "shader/profile.h"
#include "shader/resource.h"

namespace Shader::Optimization {

namespace {

using IR::Opcode;

/// The instruction producing `value` when it is an `op`.
IR::Inst* Producer(const IR::Value& value, Opcode op) {
    IR::Inst* const inst = value.TryInst();
    return inst && inst->GetOpcode() == op ? inst : nullptr;
}

bool IsImm(const IR::Value& value, u32 imm) {
    return value.IsImmediate() && value.U32() == imm;
}

bool IsSharpSource(const IR::Inst* inst) {
    const Opcode op = inst->GetOpcode();
    return op == Opcode::GetUserData || op == Opcode::ReadConst || op == Opcode::ReadConstBuffer;
}

/// s_or_b32 s37, s101, imm on V# dword 1 (typically forcing the swizzle/stride bits):
/// {original dword 1, imm}.
std::optional<std::pair<IR::Value, u32>> StridePatch(const IR::Value& dw1) {
    const IR::Inst* const inst = Producer(dw1, Opcode::BitwiseOr32);
    if (!inst || !inst->Arg(1).IsImmediate()) {
        return std::nullopt;
    }
    return std::pair{inst->Arg(0), inst->Arg(1).U32()};
}

/// A constant buffer inline in the program: s_getpc_b64, then s_add_u32 of a constant to the low
/// half. The program-relative address of the data.
std::optional<IR::Value> InlineConstantBuffer(const IR::Value& dw0) {
    const IR::Inst* const add = Producer(dw0, Opcode::IAdd32);
    if (!add || add->Arg(0).IsImmediate() == add->Arg(1).IsImmediate()) {
        return std::nullopt;
    }
    const bool imm_first = add->Arg(0).IsImmediate();
    const u32 offset = add->Arg(imm_first ? 0 : 1).U32();
    const IR::Inst* const pc = add->Arg(imm_first ? 1 : 0).Inst();
    if (pc->GetOpcode() != Opcode::GetPcLo) {
        return std::nullopt;
    }
    return IR::Value{pc->Arg(0).U32() + offset};
}

/// A cube map T# rewritten as a 2D array: dword 3's type set to 0xd (or with 0xd0000000 after
/// masking with 0x0fffffff) and dword 4's depth replaced with depth * 6 + 5 (or into the bits
/// masked with 0xffffe000). The original dwords 3 and 4.
std::optional<std::pair<IR::Value, IR::Value>> CubeAs2DArray(const IR::Value& dw3,
                                                              const IR::Value& dw4) {
    const IR::Inst* const type_or = Producer(dw3, Opcode::BitwiseOr32);
    if (!type_or || type_or->Arg(1).IsImmediate() || !IsImm(type_or->Arg(0), 0xd0000000u)) {
        return std::nullopt;
    }
    const IR::Inst* const type_and = type_or->Arg(1).Inst();
    if (type_and->GetOpcode() != Opcode::BitwiseAnd32 || !IsImm(type_and->Arg(1), 0xfffffffu)) {
        return std::nullopt;
    }
    const IR::Inst* const depth_or = Producer(dw4, Opcode::BitwiseOr32);
    if (!depth_or || depth_or->Arg(0).IsImmediate() || depth_or->Arg(1).IsImmediate()) {
        return std::nullopt;
    }
    const IR::Inst* const depth_and = depth_or->Arg(1).Inst();
    if (depth_and->GetOpcode() != Opcode::BitwiseAnd32 || !IsImm(depth_and->Arg(1), 0xffffe000u)) {
        return std::nullopt;
    }
    return std::pair{type_and->Arg(0), depth_and->Arg(0)};
}

/// Anisotropy disabled when the texture has one LOD:
///   s_bfe_u32 s0, s7, 0x0008000c (T# dword 3, the LOD bits); s_and_b32 s1, s12, 0xfffff1ff;
///   s_cmp_eq_u32 s0, 0; s_cselect_b32 s12, s1, s12.
/// {original S# dword 0, T# dword 3}.
std::optional<std::pair<IR::Value, IR::Value>> AnisoOffAtLod0(const IR::Value& dw0) {
    const IR::Inst* const select = Producer(dw0, Opcode::SelectU32);
    if (!select) {
        return std::nullopt;
    }
    const IR::Inst* const is_zero = select->Arg(0).Inst();
    if (is_zero->GetOpcode() != Opcode::IEqual32 || !IsImm(is_zero->Arg(1), 0u)) {
        return std::nullopt;
    }
    const IR::Inst* const lods = is_zero->Arg(0).Inst();
    ASSERT(lods->GetOpcode() != Opcode::Phi);
    // (Offset and count as constant propagation leaves them.)
    if (lods->GetOpcode() != Opcode::BitFieldUExtract || !IsImm(lods->Arg(1), 12) ||
        !IsImm(lods->Arg(2), 8)) {
        return std::nullopt;
    }
    const IR::Inst* const masked = select->Arg(1).Inst();
    if (masked->GetOpcode() != Opcode::BitwiseAnd32 || masked->Arg(1).U32() != 0xfffff1ff) {
        return std::nullopt;
    }
    ASSERT(select->Arg(2).Inst()->GetOpcode() != Opcode::Phi);
    return std::pair{select->Arg(2), lods->Arg(0)};
}

/// s_and_b32 s12, 0xfffffe00, s12: clamp_x/y/z forced to wrap.
std::optional<IR::Value> ClampToWrap(const IR::Value& dw0) {
    const IR::Inst* const inst = Producer(dw0, Opcode::BitwiseAnd32);
    if (!inst || !IsImm(inst->Arg(0), 0xfffffe00u)) {
        return std::nullopt;
    }
    return inst->Arg(1);
}

/// The wrap mask above, then s_or_b32 with 0x12: clamp_x/y forced to the last texel.
std::optional<IR::Value> ClampToLastTexel(const IR::Value& dw0) {
    const IR::Inst* const inst = Producer(dw0, Opcode::BitwiseOr32);
    if (!inst || inst->Arg(1).IsImmediate() || !IsImm(inst->Arg(0), 0x12u)) {
        return std::nullopt;
    }
    const IR::Inst* const mask = inst->Arg(1).Inst();
    if (mask->GetOpcode() != Opcode::BitwiseAnd32 || !IsImm(mask->Arg(0), 0xfffffe00u)) {
        return std::nullopt;
    }
    return mask->Arg(1);
}

/// s_and_b32 s12, s12, 0xfff8f1ff: anisotropy ratio and threshold cleared.
std::optional<IR::Value> ClearAniso(const IR::Value& dw0) {
    const IR::Inst* const inst = Producer(dw0, Opcode::BitwiseAnd32);
    if (!inst || !IsImm(inst->Arg(1), 0xfff8f1ffu)) {
        return std::nullopt;
    }
    return inst->Arg(0);
}

/// Sharps read from a constant buffer (bindless) mark their reads, for the extended user data
/// flattening to include them.
void MarkSharpReads(const SharpReference& sharp) {
    for (size_t i = 0; i < sharp.num_dwords; i++) {
        IR::Inst* const source = sharp.dwords[i].TryInst();
        if (!source) {
            continue;
        }
        ASSERT(IsSharpSource(source));
        if (source->GetOpcode() == Opcode::ReadConstBuffer) {
            auto flags = source->Flags<IR::BufferInstInfo>();
            flags.sharp_source.Assign(1u);
            source->SetFlags(flags);
        }
    }
}

void DiscoverBuffer(IR::Inst& inst, ResourceDiscoveryList& resources) {
    const IR::Inst* const handle = inst.Arg(0).Inst();
    SharpReference& vsharp = resources.emplace_back(&inst).sharps[0];
    vsharp.num_dwords = handle->NumArgs();
    for (size_t i = 0; i < handle->NumArgs(); ++i) {
        vsharp.dwords[i] = handle->Arg(i);
        // A sharp in VGPRs is read through its first lane.
        if (const IR::Inst* const first = Producer(vsharp.dwords[i], Opcode::ReadFirstLane)) {
            vsharp.dwords[i] = first->Arg(0);
        }
    }
    if (const auto address = InlineConstantBuffer(vsharp.dwords[0])) {
        vsharp.post_op = SharpFetchPostOp::OffsetByProgramBase;
        vsharp.dwords[0] = *address;
        vsharp.dwords[1] = IR::Value{0U};
    } else if (const auto patch = StridePatch(handle->Arg(1))) {
        vsharp.post_op = SharpFetchPostOp::BitwiseOrDw1WithImm;
        vsharp.post_op_data.dw1_mask = patch->second;
        vsharp.dwords[1] = patch->first;
    }
    MarkSharpReads(vsharp);
}

void DiscoverImage(IR::Inst& inst, ResourceDiscoveryList& resources) {
    const IR::Inst* const handle = inst.Arg(0).Inst();
    ASSERT(handle->GetOpcode() == Opcode::ImageHandle);
    ResourceDiscovery& resource = resources.emplace_back(&inst);

    // T#: four dwords, or eight with the high half.
    SharpReference& tsharp = resource.sharps[0];
    const IR::Inst* const low = handle->Arg(0).Inst();
    for (size_t i = 0; i < low->NumArgs(); ++i) {
        tsharp.dwords[tsharp.num_dwords++] = low->Arg(i);
    }
    if (const IR::Inst* const high = handle->Arg(1).TryInst()) {
        for (size_t i = 0; i < high->NumArgs(); ++i) {
            tsharp.dwords[tsharp.num_dwords++] = high->Arg(i);
        }
    }
    if (const auto dwords = CubeAs2DArray(tsharp.dwords[3], tsharp.dwords[4])) {
        tsharp.post_op = SharpFetchPostOp::ConvertCubeTo2DArray;
        tsharp.dwords[3] = dwords->first;
        tsharp.dwords[4] = dwords->second;
    }
    MarkSharpReads(tsharp);

    if (inst.GetOpcode() != Opcode::ImageSampleRaw) {
        return;
    }
    SharpReference& ssharp = resource.sharps[1];
    const IR::Inst* const sampler = inst.Arg(1).Inst();
    for (size_t i = 0; i < sampler->NumArgs(); ++i) {
        ssharp.dwords[ssharp.num_dwords++] = sampler->Arg(i);
    }
    const IR::Value dw0 = ssharp.dwords[0];
    if (const auto aniso = AnisoOffAtLod0(dw0)) {
        ssharp.post_op = SharpFetchPostOp::DisableAnisoIfSingleLod;
        ssharp.post_op_data.lod_prod = aniso->second;
        ssharp.dwords[0] = aniso->first;
    } else if (const auto wrap = ClampToWrap(dw0)) {
        ssharp.post_op = SharpFetchPostOp::ForceRepeatXyzClamp;
        ssharp.dwords[0] = *wrap;
    } else if (const auto last_texel = ClampToLastTexel(dw0)) {
        ssharp.post_op = SharpFetchPostOp::ForceLastTexelXyClamp;
        ssharp.dwords[0] = *last_texel;
    } else if (const auto clear = ClearAniso(dw0)) {
        ssharp.post_op = SharpFetchPostOp::ClearAnisoRatioAndThreshold;
        ssharp.dwords[0] = *clear;
    }
    MarkSharpReads(ssharp);
}

} // namespace

ResourceDiscoveryList ResourceDiscoverPass(IR::Program& program, const Profile& profile) {
    ResourceDiscoveryList resources;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (IsBufferInstruction(inst)) {
                DiscoverBuffer(inst, resources);
            } else if (IsImageInstruction(inst)) {
                DiscoverImage(inst, resources);
            } else if (IsDataRingInstruction(inst)) {
                resources.emplace_back(&inst);
            }
        }
    }
    return resources;
}

} // namespace Shader::Optimization
