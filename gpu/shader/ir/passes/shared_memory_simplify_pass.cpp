// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Without aliased shared memory (VK_KHR_workgroup_memory_explicit_layout) shared memory has one
// element type: 16- and 64-bit loads and stores of compute shaders become 32-bit ones, unless
// atomics of that size need the type.
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/profile.h"

namespace Shader::Optimization {

namespace {

bool Is64BitSharedAtomic(IR::Opcode op) {
    switch (op) {
    case IR::Opcode::SharedAtomicIAdd64:
    case IR::Opcode::SharedAtomicISub64:
    case IR::Opcode::SharedAtomicSMin64:
    case IR::Opcode::SharedAtomicUMin64:
    case IR::Opcode::SharedAtomicSMax64:
    case IR::Opcode::SharedAtomicUMax64:
    case IR::Opcode::SharedAtomicInc64:
    case IR::Opcode::SharedAtomicDec64:
    case IR::Opcode::SharedAtomicAnd64:
    case IR::Opcode::SharedAtomicOr64:
    case IR::Opcode::SharedAtomicXor64:
    case IR::Opcode::SharedAtomicCmpSwap64:
        return true;
    default:
        return false;
    }
}

/// The types shared atomics need (there are no 16-bit ones).
IR::Type AtomicTypes(const IR::Program& program) {
    IR::Type types = IR::Type::Void;
    for (const IR::Block* block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            if (Is64BitSharedAtomic(inst.GetOpcode())) {
                types |= IR::Type::U64;
            }
        }
    }
    return types;
}

} // namespace

void SharedMemorySimplifyPass(IR::Program& program, const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute ||
        profile.supports_workgroup_explicit_memory_layout) {
        return;
    }
    const IR::Type atomic_types = AtomicTypes(program);
    const bool keep_16 = True(atomic_types & IR::Type::U16);
    const bool keep_64 = True(atomic_types & IR::Type::U64);
    if (keep_16 && keep_64) {
        return;
    }
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const IR::Opcode op = inst.GetOpcode();
            if (op != IR::Opcode::LoadSharedU16 && op != IR::Opcode::LoadSharedU64 &&
                op != IR::Opcode::WriteSharedU16 && op != IR::Opcode::WriteSharedU64) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const IR::U32 offset{inst.Arg(0)};
            if (!keep_16 && op == IR::Opcode::LoadSharedU16) {
                // The half of the aligned dword.
                const IR::U32 dword_offset{ir.BitwiseAnd(offset, ir.Imm32(~3U))};
                const IR::U32 dword{ir.LoadShared(32, false, dword_offset)};
                const IR::U32 bit_offset{ir.IMul(ir.BitwiseAnd(offset, ir.Imm32(2U)), ir.Imm32(8U))};
                const IR::U32 value{ir.BitFieldExtract(dword, bit_offset, ir.Imm32(16U))};
                inst.ReplaceUsesWithAndRemove(ir.UConvert(16, value));
                continue;
            }
            if (!keep_16 && op == IR::Opcode::WriteSharedU16) {
                // Read, insert the half, write back.
                const IR::U32 value{ir.UConvert(32, IR::U16{inst.Arg(1)})};
                const IR::U32 bit_offset{ir.IMul(ir.BitwiseAnd(offset, ir.Imm32(2U)), ir.Imm32(8U))};
                const IR::U32 dword_offset{ir.BitwiseAnd(offset, ir.Imm32(~3U))};
                const IR::U32 dword{ir.LoadShared(32, false, ir.BitwiseAnd(offset, dword_offset))};
                const IR::U32 merged{ir.BitFieldInsert(dword, value, bit_offset, ir.Imm32(16U))};
                ir.WriteShared(32, merged, dword_offset);
                inst.Invalidate();
                continue;
            }
            if (!keep_64 && op == IR::Opcode::LoadSharedU64) {
                const IR::U32 lo{ir.LoadShared(32, false, offset)};
                const IR::U32 hi{ir.LoadShared(32, false, ir.IAdd(offset, ir.Imm32(4U)))};
                const IR::Value value{ir.PackUint2x32(ir.CompositeConstruct(lo, hi))};
                inst.ReplaceUsesWithAndRemove(value);
                continue;
            }
            if (!keep_64 && op == IR::Opcode::WriteSharedU64) {
                const IR::Value halves{ir.UnpackUint2x32(IR::U64{inst.Arg(1)})};
                const IR::U32 lo{ir.CompositeExtract(halves, 0)};
                const IR::U32 hi{ir.CompositeExtract(halves, 1)};
                ir.WriteShared(32, lo, offset);
                ir.WriteShared(32, hi, ir.IAdd(offset, ir.Imm32(4U)));
                inst.Invalidate();
            }
        }
    }
}

} // namespace Shader::Optimization
