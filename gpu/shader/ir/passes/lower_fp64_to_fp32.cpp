// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Without 64-bit floats doubles are computed as floats: 64-bit operations take their 32-bit
// forms, and the packing of doubles to and from dword pairs converts between the encodings.
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"

namespace Shader::Optimization {

namespace {

constexpr s32 F64ToF32Exp = +1023 - 127;
constexpr s32 F32ToF64Exp = +127 - 1023;

/// A double given as two dwords, as a float (the mantissa truncated, the exponent rebiased;
/// zero/denormal and infinity/NaN exponents kept as such).
IR::F32 PackedF64ToF32(IR::IREmitter& ir, const IR::Value& packed) {
    const IR::U32 lo{ir.CompositeExtract(packed, 0)};
    const IR::U32 hi{ir.CompositeExtract(packed, 1)};
    const IR::U32 sign{ir.BitFieldExtract(hi, ir.Imm32(31), ir.Imm32(1))};
    const IR::U32 exp{ir.BitFieldExtract(hi, ir.Imm32(20), ir.Imm32(11))};
    const IR::U32 mantissa_hi{ir.BitFieldExtract(hi, ir.Imm32(0), ir.Imm32(20))};
    const IR::U32 mantissa_lo{ir.BitFieldExtract(lo, ir.Imm32(29), ir.Imm32(3))};
    const IR::U32 mantissa{
        ir.BitwiseOr(ir.ShiftLeftLogical(mantissa_hi, ir.Imm32(3)), mantissa_lo)};
    const IR::U32 exp_if_subnorm{
        ir.Select(ir.IEqual(exp, ir.Imm32(0)), ir.Imm32(0), ir.IAdd(exp, ir.Imm32(F64ToF32Exp)))};
    const IR::U32 exp_if_infnan{
        ir.Select(ir.IEqual(exp, ir.Imm32(0x7ff)), ir.Imm32(0xff), exp_if_subnorm)};
    const IR::U32 result{
        ir.BitwiseOr(ir.ShiftLeftLogical(sign, ir.Imm32(31)),
                     ir.BitwiseOr(ir.ShiftLeftLogical(exp_if_infnan, ir.Imm32(23)), mantissa))};
    return ir.BitCast<IR::F32>(result);
}

/// A float as the two dwords of the double.
IR::Value F32ToPackedF64(IR::IREmitter& ir, const IR::Value& raw) {
    const IR::U32 value{ir.BitCast<IR::U32>(IR::F32(raw))};
    const IR::U32 sign{ir.BitFieldExtract(value, ir.Imm32(31), ir.Imm32(1))};
    const IR::U32 exp{ir.BitFieldExtract(value, ir.Imm32(23), ir.Imm32(8))};
    const IR::U32 mantissa{ir.BitFieldExtract(value, ir.Imm32(0), ir.Imm32(23))};
    const IR::U32 mantissa_hi{ir.BitFieldExtract(mantissa, ir.Imm32(3), ir.Imm32(20))};
    const IR::U32 mantissa_lo{ir.BitFieldExtract(mantissa, ir.Imm32(0), ir.Imm32(3))};
    const IR::U32 exp_if_subnorm{
        ir.Select(ir.IEqual(exp, ir.Imm32(0)), ir.Imm32(0), ir.IAdd(exp, ir.Imm32(F32ToF64Exp)))};
    const IR::U32 exp_if_infnan{
        ir.Select(ir.IEqual(exp, ir.Imm32(0xff)), ir.Imm32(0x7ff), exp_if_subnorm)};
    const IR::U32 lo{ir.ShiftLeftLogical(mantissa_lo, ir.Imm32(29))};
    const IR::U32 hi{
        ir.BitwiseOr(ir.ShiftLeftLogical(sign, ir.Imm32(31)),
                     ir.BitwiseOr(ir.ShiftLeftLogical(exp_if_infnan, ir.Imm32(20)), mantissa_hi))};
    return ir.CompositeConstruct(lo, hi);
}

/// The 32-bit form of a 64-bit float operation (others unchanged).
IR::Opcode As32Bit(IR::Opcode op) {
    using IR::Opcode;
    switch (op) {
#define BB_FP64(name)                                                                             \
    case Opcode::name##64:                                                                        \
        return Opcode::name##32;
        BB_FP64(FPAbs)
        BB_FP64(FPAdd)
        BB_FP64(FPFma)
        BB_FP64(FPMax)
        BB_FP64(FPMin)
        BB_FP64(FPMul)
        BB_FP64(FPDiv)
        BB_FP64(FPNeg)
        BB_FP64(FPRecip)
        BB_FP64(FPRecipSqrt)
        BB_FP64(FPSaturate)
        BB_FP64(FPClamp)
        BB_FP64(FPRoundEven)
        BB_FP64(FPFloor)
        BB_FP64(FPCeil)
        BB_FP64(FPTrunc)
        BB_FP64(FPFract)
        BB_FP64(FPFrexpSig)
        BB_FP64(FPFrexpExp)
        BB_FP64(FPOrdEqual)
        BB_FP64(FPUnordEqual)
        BB_FP64(FPOrdNotEqual)
        BB_FP64(FPUnordNotEqual)
        BB_FP64(FPOrdLessThan)
        BB_FP64(FPUnordLessThan)
        BB_FP64(FPOrdGreaterThan)
        BB_FP64(FPUnordGreaterThan)
        BB_FP64(FPOrdLessThanEqual)
        BB_FP64(FPUnordLessThanEqual)
        BB_FP64(FPOrdGreaterThanEqual)
        BB_FP64(FPUnordGreaterThanEqual)
        BB_FP64(FPIsNan)
        BB_FP64(FPIsInf)
#undef BB_FP64
    case Opcode::ConvertS32F64:
        return Opcode::ConvertS32F32;
    case Opcode::ConvertF64S32:
        return Opcode::ConvertF32S32;
    case Opcode::ConvertF64U32:
        return Opcode::ConvertF32U32;
    default:
        return op;
    }
}

} // namespace

void LowerFp64ToFp32(IR::Program& program) {
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            switch (inst.GetOpcode()) {
            case IR::Opcode::PackDouble2x32:
                inst.ReplaceUsesWithAndRemove(PackedF64ToF32(ir, inst.Arg(0)));
                break;
            case IR::Opcode::UnpackDouble2x32:
                inst.ReplaceUsesWithAndRemove(F32ToPackedF64(ir, inst.Arg(0)));
                break;
            case IR::Opcode::ConvertF64F32:
            case IR::Opcode::ConvertF32F64:
                inst.ReplaceUsesWithAndRemove(inst.Arg(0)); // same in 32 bits
                break;
            default:
                inst.ReplaceOpcode(As32Bit(inst.GetOpcode()));
                break;
            }
        }
    }
}

} // namespace Shader::Optimization
