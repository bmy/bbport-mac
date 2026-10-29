// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <array>

#include "common/assert.h"
#include "gcn/decoder.h"
#include "shader/frontend/copy_shader.h"
#include "shader/frontend/instruction.h"

namespace Shader {

namespace {

/// An inline positive integer constant's value.
s32 PositiveConstant(const GcnIsa::InstOperand& operand) {
    ASSERT(operand.field == GcnIsa::OperandField::SignedConstIntPos);
    return s32(operand.code - GcnIsa::OperandFieldRange::SignedConstIntPosMin + 1);
}

} // namespace

/// The copy shader loads each component with buffer_load_dword (offset, plus an SGPR that
/// scalar code set to a constant) and exports it.
CopyShaderData ParseCopyShader(std::span<const u32> code) {
    using GcnIsa::Opcode;
    constexpr u32 MovVccHiLiteral = 0xBEEB03FF; // s_mov_b32 vcc_hi, #imm
    ASSERT_MSG(code[0] == MovVccHiLiteral, "First instruction is not s_mov_b32 vcc_hi, #imm");

    std::array<s32, 64> vgpr_offsets{}; // ring offset loaded into each VGPR
    vgpr_offsets.fill(-1);
    std::array<s32, 256> sgpr_values{}; // constants in SGPRs
    sgpr_values.fill(-1);

    CopyShaderData data{};
    auto last_attr{IR::Attribute::Position0};
    const u32* pc = code.data();
    const u32* const end = pc + code.size();
    while (pc < end) {
        const auto inst = GcnIsa::Decode(pc);
        switch (inst.opcode) {
        case Opcode::S_LSHL_B32:
            sgpr_values[inst.dst[0].code] = PositiveConstant(inst.src[0])
                                            << PositiveConstant(inst.src[1]);
            break;
        case Opcode::S_MOVK_I32:
            sgpr_values[inst.dst[0].code] = inst.control.sopk.simm;
            break;
        case Opcode::S_MOV_B32:
            sgpr_values[inst.dst[0].code] = inst.src[0].code;
            break;
        case Opcode::S_ADDK_I32:
            sgpr_values[inst.dst[0].code] += inst.control.sopk.simm;
            break;
        case Opcode::S_BFM_B32: {
            const s32 width = PositiveConstant(inst.src[0]);
            const s32 offset = PositiveConstant(inst.src[1]);
            sgpr_values[inst.dst[0].code] = ((1 << width) - 1) << offset;
            break;
        }
        case Opcode::BUFFER_LOAD_DWORD: {
            const u32 vgpr = inst.src[1].code;
            ASSERT_MSG(vgpr < vgpr_offsets.size(),
                       "offsets array for geometry shaders is too short");
            vgpr_offsets[vgpr] = inst.control.mubuf.offset;
            if (inst.src[3].field != GcnIsa::OperandField::ConstZero) {
                const u32 sgpr = inst.src[3].code;
                ASSERT(sgpr_values[sgpr] != -1);
                vgpr_offsets[vgpr] += sgpr_values[sgpr];
            }
            data.num_comps++;
            break;
        }
        case Opcode::EXP: {
            const auto& exp = inst.control.exp;
            const auto attribute = static_cast<IR::Attribute>(exp.target);
            for (int i = 0; i < inst.src_count; ++i) {
                if ((exp.en & (1 << i)) == 0) {
                    continue;
                }
                const s32 offset = vgpr_offsets[inst.src[i].code];
                if (offset != -1) {
                    data.attr_map[offset] = {attribute, i};
                    if (attribute > last_attr) {
                        last_attr = attribute;
                    }
                }
            }
            break;
        }
        default:
            break;
        }
    }
    if (!IsPosition(last_attr)) {
        data.num_attrs = static_cast<u32>(last_attr) - static_cast<u32>(IR::Attribute::Param0) + 1;
    }
    // Consecutive components lie output_vertices * 64 apart in the ring.
    if (data.attr_map.size() > 1) {
        const auto first = data.attr_map.begin();
        data.output_vertices = (std::next(first)->first - first->first) / 64;
    }
    return data;
}

} // namespace Shader
