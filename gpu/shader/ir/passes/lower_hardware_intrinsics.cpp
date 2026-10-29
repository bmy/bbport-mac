// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Lowers operations that mirror GCN hardware: float class tests, the fields of the packed
// ancillary VGPR (sample index, render target index), and masked bit counts (MBCNT).
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"

namespace Shader::Optimization {

namespace {

IR::IREmitter EmitterBefore(IR::Block& block, IR::Inst& inst) {
    return IR::IREmitter{block, IR::Block::InstructionList::s_iterator_to(inst)};
}

void LowerCmpClass(IR::Block& block, IR::Inst& inst) {
    ASSERT_MSG(inst.Arg(1).IsImmediate(), "float class test with a variable class mask");
    using IR::FloatClassFunc;
    const auto classes = static_cast<FloatClassFunc>(inst.Arg(1).U32());
    const auto all_of = [classes](FloatClassFunc set) { return (classes & set) == set; };
    if (all_of(FloatClassFunc::NaN)) {
        inst.ReplaceOpcode(IR::Opcode::FPIsNan32);
    } else if (all_of(FloatClassFunc::Infinity)) {
        inst.ReplaceOpcode(IR::Opcode::FPIsInf32);
    } else if (all_of(FloatClassFunc::Finite)) {
        auto ir = EmitterBefore(block, inst);
        const IR::F32 value{inst.Arg(0)};
        inst.ReplaceUsesWithAndRemove(
            ir.LogicalNot(ir.LogicalOr(ir.FPIsNan(value), ir.FPIsInf(value))));
    } else {
        UNREACHABLE_MSG("float class mask {:#x}", u32(classes));
    }
}

/// A field of the ancillary VGPR: bits 8..11 the sample index, 16..26 the render target index.
void LowerPackedAncillary(IR::Block& block, IR::Inst& inst) {
    if (inst.Arg(0).IsImmediate() || !inst.Arg(1).IsImmediate() || !inst.Arg(2).IsImmediate()) {
        return;
    }
    IR::Inst* const ancillary = inst.Arg(0).Inst();
    if (ancillary->GetOpcode() != IR::Opcode::GetAttributeU32 ||
        ancillary->Arg(0).Attribute() != IR::Attribute::PackedAncillary) {
        return;
    }
    const u32 offset = inst.Arg(1).U32();
    const u32 bits = inst.Arg(2).U32();
    auto ir = EmitterBefore(block, inst);
    const auto field = [&](IR::Attribute attribute, u32 first, u32 width) {
        const auto value = ir.GetAttributeU32(attribute);
        if (offset == first && bits == width) {
            inst.ReplaceUsesWithAndRemove(value);
        } else {
            inst.ReplaceUsesWithAndRemove(
                ir.BitFieldExtract(value, ir.Imm32(offset - first), ir.Imm32(bits)));
        }
    };
    if (offset >= 8 && offset + bits <= 12) {
        field(IR::Attribute::SampleIndex, 8, 4);
    } else if (offset >= 16 && offset + bits <= 27) {
        field(IR::Attribute::RenderTargetIndex, 16, 11);
    } else {
        UNREACHABLE_MSG("ancillary VGPR field at {}, {} bits", offset, bits);
    }
    ancillary->ReplaceUsesWithAndRemove(ir.Imm32(0U));
}

/// The bits set in the value below this lane (of the low or high 32 lanes), plus the addend.
void LowerMaskedBitCount(IR::Block& block, IR::Inst& inst) {
    auto ir = EmitterBefore(block, inst);
    const IR::U32 lanes_below{
        ir.GetAttributeU32(IR::Attribute::SubgroupLtMask, inst.Arg(2).U1() ? 1 : 0)};
    const IR::U32 count{ir.BitCount(ir.BitwiseAnd(IR::U32{inst.Arg(0)}, lanes_below))};
    inst.ReplaceUsesWithAndRemove(ir.IAdd(count, IR::U32{inst.Arg(1)}));
}

} // namespace

void LowerHardwareIntrinsics(IR::Program& program) {
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case IR::Opcode::FPCmpClass32:
                LowerCmpClass(*block, inst);
                break;
            case IR::Opcode::BitFieldUExtract:
                LowerPackedAncillary(*block, inst);
                break;
            case IR::Opcode::MaskedBitCount32:
                LowerMaskedBitCount(*block, inst);
                break;
            default:
                break;
            }
        }
    }
}

} // namespace Shader::Optimization
