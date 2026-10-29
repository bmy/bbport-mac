// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// SOP2/SOP1/SOPK/SOPC: scalar arithmetic, logic, shifts, bit fields and compares, with SCC.
// 64-bit masks (EXEC, VCC and their copies) are logic on lane masks: their SCC ("any bit
// set") is InverseBallot of the result.
#include "common/assert.h"
#include "shader/frontend/translate/translate.h"

namespace Shader::Gcn {

void Translator::EmitScalarAlu(const GcnInst& inst) {
    if (inst.encoding == InstEncoding::SOPC) {
        return EmitSOPC(inst);
    }
    if (inst.encoding == InstEncoding::SOPK) {
        return EmitSOPK(inst);
    }
    switch (inst.opcode) {
    // SOP2
    case Opcode::S_ADD_U32:
        return S_ADD_U32(inst);
    case Opcode::S_SUB_U32:
        return S_SUB_U32(inst);
    case Opcode::S_ADD_I32:
        return S_ADD_I32(inst);
    case Opcode::S_SUB_I32:
        return S_SUB_I32(inst);
    case Opcode::S_ADDC_U32:
        return S_ADDC_U32(inst);
    case Opcode::S_SUBB_U32:
        return S_SUBB_U32(inst);
    case Opcode::S_MIN_I32:
        return S_MIN_U32(true, inst);
    case Opcode::S_MIN_U32:
        return S_MIN_U32(false, inst);
    case Opcode::S_MAX_I32:
        return S_MAX_U32(true, inst);
    case Opcode::S_MAX_U32:
        return S_MAX_U32(false, inst);
    case Opcode::S_CSELECT_B32:
        return S_CSELECT_B32(inst);
    case Opcode::S_CSELECT_B64:
        return S_CSELECT_B64(inst);
    case Opcode::S_AND_B32:
        return S_AND_B32(NegateMode::None, inst);
    case Opcode::S_ANDN2_B32:
        return S_AND_B32(NegateMode::Src1, inst);
    case Opcode::S_NAND_B32:
        return S_AND_B32(NegateMode::Result, inst);
    case Opcode::S_AND_B64:
        return S_AND_B64(NegateMode::None, inst);
    case Opcode::S_ANDN2_B64:
        return S_AND_B64(NegateMode::Src1, inst);
    case Opcode::S_NAND_B64:
        return S_AND_B64(NegateMode::Result, inst);
    case Opcode::S_OR_B32:
        return S_OR_B32(inst);
    case Opcode::S_OR_B64:
        return S_OR_B64(NegateMode::None, false, inst);
    case Opcode::S_ORN2_B64:
        return S_OR_B64(NegateMode::Src1, false, inst);
    case Opcode::S_NOR_B64:
        return S_OR_B64(NegateMode::Result, false, inst);
    case Opcode::S_XOR_B32:
        return S_XOR_B32(inst);
    case Opcode::S_XOR_B64:
        return S_OR_B64(NegateMode::None, true, inst);
    case Opcode::S_XNOR_B64:
        return S_OR_B64(NegateMode::Result, true, inst);
    case Opcode::S_LSHL_B32:
        return S_LSHL_B32(inst);
    case Opcode::S_LSHL_B64:
        return S_LSHL_B64(inst);
    case Opcode::S_LSHR_B32:
        return S_LSHR_B32(inst);
    case Opcode::S_ASHR_I32:
        return S_ASHR_I32(inst);
    case Opcode::S_ASHR_I64:
        return S_ASHR_I64(inst);
    case Opcode::S_BFM_B32:
        return S_BFM_B32(inst);
    case Opcode::S_MUL_I32:
        return S_MUL_I32(inst);
    case Opcode::S_BFE_I32:
        return S_BFE(inst, true);
    case Opcode::S_BFE_U32:
        return S_BFE(inst, false);
    case Opcode::S_ABSDIFF_I32:
        return S_ABSDIFF_I32(inst);
    // SOP1
    case Opcode::S_MOV_B32:
        return S_MOV(inst);
    case Opcode::S_MOV_B64:
        return S_MOV_B64(inst);
    case Opcode::S_NOT_B32:
        return S_NOT_B32(inst);
    case Opcode::S_NOT_B64:
        return S_NOT_B64(inst);
    case Opcode::S_WQM_B64:
        return; // (whole quad mode is implied by the host)
    case Opcode::S_BREV_B32:
        return S_BREV_B32(inst);
    case Opcode::S_BCNT1_I32_B32:
        return S_BCNT1_I32_B32(inst);
    case Opcode::S_BCNT1_I32_B64:
        return S_BCNT1_I32_B64(inst);
    case Opcode::S_FF1_I32_B32:
        return S_FF1_I32_B32(inst);
    case Opcode::S_FF1_I32_B64:
        return S_FF1_I32_B64(inst);
    case Opcode::S_FLBIT_I32_B32:
        return S_FLBIT_I32_B32(inst);
    case Opcode::S_FLBIT_I32_B64:
        return S_FLBIT_I32_B64(inst);
    case Opcode::S_BITSET0_B32:
        return S_BITSET_B32(inst, 0);
    case Opcode::S_BITSET1_B32:
        return S_BITSET_B32(inst, 1);
    case Opcode::S_BITSET0_B64:
        return S_BITSET_B64(inst, 0);
    case Opcode::S_BITSET1_B64:
        return S_BITSET_B64(inst, 1);
    case Opcode::S_AND_SAVEEXEC_B64:
        return S_SAVEEXEC_B64(NegateMode::None, false, inst);
    case Opcode::S_ORN2_SAVEEXEC_B64:
        return S_SAVEEXEC_B64(NegateMode::Src1, true, inst);
    case Opcode::S_ABS_I32:
        return S_ABS_I32(inst);
    default:
        LogMissingOpcode(inst);
    }
}

void Translator::EmitSOPC(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::S_CMP_EQ_I32:
        return S_CMP(ConditionOp::EQ, true, inst);
    case Opcode::S_CMP_LG_I32:
        return S_CMP(ConditionOp::LG, true, inst);
    case Opcode::S_CMP_GT_I32:
        return S_CMP(ConditionOp::GT, true, inst);
    case Opcode::S_CMP_GE_I32:
        return S_CMP(ConditionOp::GE, true, inst);
    case Opcode::S_CMP_LT_I32:
        return S_CMP(ConditionOp::LT, true, inst);
    case Opcode::S_CMP_LE_I32:
        return S_CMP(ConditionOp::LE, true, inst);
    case Opcode::S_CMP_EQ_U32:
        return S_CMP(ConditionOp::EQ, false, inst);
    case Opcode::S_CMP_LG_U32:
        return S_CMP(ConditionOp::LG, false, inst);
    case Opcode::S_CMP_GT_U32:
        return S_CMP(ConditionOp::GT, false, inst);
    case Opcode::S_CMP_GE_U32:
        return S_CMP(ConditionOp::GE, false, inst);
    case Opcode::S_CMP_LT_U32:
        return S_CMP(ConditionOp::LT, false, inst);
    case Opcode::S_CMP_LE_U32:
        return S_CMP(ConditionOp::LE, false, inst);
    case Opcode::S_BITCMP0_B32:
        return S_BITCMP(false, 32, inst);
    case Opcode::S_BITCMP1_B32:
        return S_BITCMP(true, 32, inst);
    case Opcode::S_BITCMP0_B64:
        return S_BITCMP(false, 64, inst);
    case Opcode::S_BITCMP1_B64:
        return S_BITCMP(true, 64, inst);
    default:
        LogMissingOpcode(inst);
    }
}

void Translator::EmitSOPK(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::S_MOVK_I32:
        return S_MOVK(inst, false);
    case Opcode::S_CMOVK_I32:
        return S_MOVK(inst, true);
    case Opcode::S_CMPK_EQ_I32:
        return S_CMPK(ConditionOp::EQ, true, inst);
    case Opcode::S_CMPK_LG_I32:
        return S_CMPK(ConditionOp::LG, true, inst);
    case Opcode::S_CMPK_GT_I32:
        return S_CMPK(ConditionOp::GT, true, inst);
    case Opcode::S_CMPK_GE_I32:
        return S_CMPK(ConditionOp::GE, true, inst);
    case Opcode::S_CMPK_LT_I32:
        return S_CMPK(ConditionOp::LT, true, inst);
    case Opcode::S_CMPK_LE_I32:
        return S_CMPK(ConditionOp::LE, true, inst);
    case Opcode::S_CMPK_EQ_U32:
        return S_CMPK(ConditionOp::EQ, false, inst);
    case Opcode::S_CMPK_LG_U32:
        return S_CMPK(ConditionOp::LG, false, inst);
    case Opcode::S_CMPK_GT_U32:
        return S_CMPK(ConditionOp::GT, false, inst);
    case Opcode::S_CMPK_GE_U32:
        return S_CMPK(ConditionOp::GE, false, inst);
    case Opcode::S_CMPK_LT_U32:
        return S_CMPK(ConditionOp::LT, false, inst);
    case Opcode::S_CMPK_LE_U32:
        return S_CMPK(ConditionOp::LE, false, inst);
    case Opcode::S_ADDK_I32:
        return S_ADDK_I32(inst);
    case Opcode::S_MULK_I32:
        return S_MULK_I32(inst);
    default:
        LogMissingOpcode(inst);
    }
}

namespace {

IR::U1 Compare(IR::IREmitter& ir, ConditionOp cond, const IR::U32& lhs, const IR::U32& rhs,
               bool is_signed) {
    switch (cond) {
    case ConditionOp::EQ:
        return ir.IEqual(lhs, rhs);
    case ConditionOp::LG:
        return ir.INotEqual(lhs, rhs);
    case ConditionOp::GT:
        return ir.IGreaterThan(lhs, rhs, is_signed);
    case ConditionOp::GE:
        return ir.IGreaterThanEqual(lhs, rhs, is_signed);
    case ConditionOp::LT:
        return ir.ILessThan(lhs, rhs, is_signed);
    case ConditionOp::LE:
        return ir.ILessThanEqual(lhs, rhs, is_signed);
    default:
        UNREACHABLE();
    }
}

/// Signed overflow of a = b +/- c from the sign bits: for an add, b and c agree and the result
/// differs; for a subtract, b and c differ and the result differs from b.
IR::U1 SignedOverflow(IR::IREmitter& ir, const IR::U32& src0, const IR::U32& src1,
                      const IR::U32& result, bool is_sub) {
    const IR::U32 shift{ir.Imm32(31)};
    const IR::U32 sign0{ir.ShiftRightLogical(src0, shift)};
    const IR::U32 sign1{ir.ShiftRightLogical(src1, shift)};
    const IR::U32 signr{ir.ShiftRightLogical(result, shift)};
    if (is_sub) {
        return ir.LogicalAnd(ir.INotEqual(sign0, sign1), ir.INotEqual(sign0, signr));
    }
    return ir.LogicalAnd(ir.IEqual(sign0, sign1), ir.INotEqual(sign0, signr));
}

} // namespace

/// D = result, SCC = result != 0.
#define SET_DST_SCC_NONZERO(result)                                                                \
    SetDst(inst.dst[0], result);                                                                   \
    ir.SetScc(ir.INotEqual(result, ir.Imm32(0)))

// SOP2

void Translator::S_ADD_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.IAdd(src0, src1)};
    SetDst(inst.dst[0], result);
    // Carry out: the truncated sum is below an operand.
    ir.SetScc(ir.ILessThan(result, src0, false));
}

void Translator::S_SUB_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ISub(src0, src1));
    // Borrow: S1 > S0.
    ir.SetScc(ir.IGreaterThan(src1, src0, false));
}

void Translator::S_SUBB_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 borrow{ir.Select(ir.GetScc(), ir.Imm32(1U), ir.Imm32(0U))};
    const IR::U32 difference{ir.ISub(src0, src1)};
    SetDst(inst.dst[0], ir.ISub(difference, borrow));
    // S1 + SCC > S0 as a 33-bit compare.
    const IR::U1 underflow{ir.IGreaterThan(src1, src0, false)};
    const IR::U1 borrow_underflow{ir.IGreaterThan(borrow, difference, false)};
    ir.SetScc(ir.LogicalOr(underflow, borrow_underflow));
}

void Translator::S_ADD_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.IAdd(src0, src1)};
    SetDst(inst.dst[0], result);
    ir.SetScc(SignedOverflow(ir, src0, src1, result, false));
}

void Translator::S_SUB_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.ISub(src0, src1)};
    SetDst(inst.dst[0], result);
    ir.SetScc(SignedOverflow(ir, src0, src1, result, true));
}

void Translator::S_ADDC_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 carry{ir.Select(ir.GetScc(), ir.Imm32(1U), ir.Imm32(0U))};
    const IR::U32 sum{ir.IAdd(src0, src1)};
    const IR::U32 result{ir.IAdd(sum, carry)};
    const IR::U1 carry_out1{ir.ILessThan(sum, src0, false)};
    const IR::U1 carry_out2{ir.ILessThan(result, sum, false)};
    SetDst(inst.dst[0], result);
    ir.SetScc(ir.LogicalOr(carry_out1, carry_out2));
}

/// SCC = the result is S0.
void Translator::S_MIN_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result = ir.IMin(src0, src1, is_signed);
    SetDst(inst.dst[0], result);
    ir.SetScc(ir.IEqual(result, src0));
}

void Translator::S_MAX_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result = ir.IMax(src0, src1, is_signed);
    SetDst(inst.dst[0], result);
    ir.SetScc(ir.IEqual(result, src0));
}

void Translator::S_CSELECT_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], IR::U32{ir.Select(ir.GetScc(), src0, src1)});
}

void Translator::S_CSELECT_B64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U64 src1{GetSrc64(inst.src[1])};
    SetDst64(inst.dst[0], IR::U64{ir.Select(ir.GetScc(), src0, src1)});
}

void Translator::S_AND_B32(NegateMode negate, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    IR::U32 src1{GetSrc(inst.src[1])};
    if (negate == NegateMode::Src1) {
        src1 = ir.BitwiseNot(src1);
    }
    IR::U32 result{ir.BitwiseAnd(src0, src1)};
    if (negate == NegateMode::Result) {
        result = ir.BitwiseNot(result);
    }
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_AND_B64(NegateMode negate, const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    IR::U64 src1{GetSrc64(inst.src[1])};
    if (negate == NegateMode::Src1) {
        src1 = ir.BitwiseNot(src1);
    }
    IR::U64 result = ir.BitwiseAnd(src0, src1);
    if (negate == NegateMode::Result) {
        result = ir.BitwiseNot(result);
    }
    ir.SetScc(ir.InverseBallot(result));
    SetDst64(inst.dst[0], result);
}

void Translator::S_OR_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.BitwiseOr(src0, src1)};
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_OR_B64(NegateMode negate, bool is_xor, const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    IR::U64 src1{GetSrc64(inst.src[1])};
    if (negate == NegateMode::Src1) {
        src1 = ir.BitwiseNot(src1);
    }
    IR::U64 result = is_xor ? ir.BitwiseXor(src0, src1) : ir.BitwiseOr(src0, src1);
    if (negate == NegateMode::Result) {
        result = ir.BitwiseNot(result);
    }
    ir.SetScc(ir.InverseBallot(result));
    SetDst64(inst.dst[0], result);
}

void Translator::S_XOR_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.BitwiseXor(src0, src1)};
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_NOT_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 result{ir.BitwiseNot(src0)};
    SET_DST_SCC_NONZERO(result);
}

// Shift amounts use the low 5 (6) bits.

void Translator::S_LSHL_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result = ir.ShiftLeftLogical(src0, ir.BitwiseAnd(src1, ir.Imm32(0x1F)));
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_LSHL_B64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U64 src1{GetSrc64(inst.src[1])};
    const IR::U64 result = ir.ShiftLeftLogical(src0, ir.BitwiseAnd(src1, ir.Imm64(u64(0x3F))));
    SetDst64(inst.dst[0], result);
    ir.SetScc(ir.INotEqual(result, ir.Imm64(u64(0))));
}

void Translator::S_LSHR_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 amount = ir.BitwiseAnd(src1, ir.Imm32(0x1F));
    const IR::U32 result = ir.ShiftRightLogical(src0, amount);
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_ASHR_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.ShiftRightArithmetic(src0, ir.BitwiseAnd(src1, ir.Imm32(0x1F)))};
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_ASHR_I64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U64 src1{GetSrc64(inst.src[1])};
    const IR::U64 result{ir.ShiftRightArithmetic(src0, ir.BitwiseAnd(src1, ir.Imm64(u64(0x3F))))};
    SetDst64(inst.dst[0], result);
    ir.SetScc(ir.INotEqual(result, ir.Imm64(u64(0))));
}

/// D = ((1 << S0[4:0]) - 1) << S1[4:0].
void Translator::S_BFM_B32(const GcnInst& inst) {
    const IR::U32 width{ir.BitwiseAnd(GetSrc(inst.src[0]), ir.Imm32(0x1F))};
    const IR::U32 offset{ir.BitwiseAnd(GetSrc(inst.src[1]), ir.Imm32(0x1F))};
    const IR::U32 mask{ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1u), width), ir.Imm32(1))};
    SetDst(inst.dst[0], ir.ShiftLeftLogical(mask, offset));
}

void Translator::S_MUL_I32(const GcnInst& inst) {
    SetDst(inst.dst[0], ir.IMul(GetSrc(inst.src[0]), GetSrc(inst.src[1])));
}

/// S1: offset [4:0], width [22:16].
void Translator::S_BFE(const GcnInst& inst, bool is_signed) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 offset{ir.BitwiseAnd(src1, ir.Imm32(0x1F))};
    const IR::U32 count{ir.BitFieldExtract(src1, ir.Imm32(16), ir.Imm32(7))};
    const IR::U32 result{ir.BitFieldExtract(src0, offset, count, is_signed)};
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_ABSDIFF_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.IAbs(ir.ISub(src0, src1))};
    SET_DST_SCC_NONZERO(result);
}

// SOPK (16-bit signed immediate)

void Translator::S_MOVK(const GcnInst& inst, bool is_conditional) {
    const s16 simm16 = inst.control.sopk.simm;
    IR::U32 value = ir.Imm32(static_cast<s32>(simm16));
    if (is_conditional) {
        value = IR::U32{ir.Select(ir.GetScc(), value, GetSrc(inst.dst[0]))};
    }
    SetDst(inst.dst[0], value);
}

void Translator::S_CMPK(ConditionOp cond, bool is_signed, const GcnInst& inst) {
    const s32 simm16 = inst.control.sopk.simm;
    const IR::U32 lhs = GetSrc(inst.dst[0]);
    const IR::U32 rhs = ir.Imm32(simm16);
    ir.SetScc(Compare(ir, cond, lhs, rhs, is_signed));
}

void Translator::S_ADDK_I32(const GcnInst& inst) {
    const s32 simm16 = inst.control.sopk.simm;
    const IR::U32 src0{GetSrc(inst.dst[0])};
    const IR::U32 src1{ir.Imm32(simm16)};
    const IR::U32 result{ir.IAdd(src0, src1)};
    SetDst(inst.dst[0], result);
    ir.SetScc(SignedOverflow(ir, src0, src1, result, false));
}

void Translator::S_MULK_I32(const GcnInst& inst) {
    const s32 simm16 = inst.control.sopk.simm;
    SetDst(inst.dst[0], ir.IMul(GetSrc(inst.dst[0]), ir.Imm32(simm16)));
}

// SOP1

void Translator::S_MOV(const GcnInst& inst) {
    SetDst(inst.dst[0], GetSrc(inst.src[0]));
}

void Translator::S_MOV_B64(const GcnInst& inst) {
    SetDst64(inst.dst[0], GetSrc64(inst.src[0]));
}

void Translator::S_NOT_B64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U64 result = ir.BitwiseNot(src0);
    ir.SetScc(ir.InverseBallot(result));
    SetDst64(inst.dst[0], result);
}

void Translator::S_BREV_B32(const GcnInst& inst) {
    SetDst(inst.dst[0], ir.BitReverse(GetSrc(inst.src[0])));
}

void Translator::S_BCNT1_I32_B32(const GcnInst& inst) {
    const IR::U32 result = ir.BitCount(GetSrc(inst.src[0]));
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_BCNT1_I32_B64(const GcnInst& inst) {
    const IR::U32 result = ir.BitCount(GetSrc64(inst.src[0]));
    SET_DST_SCC_NONZERO(result);
}

void Translator::S_FF1_I32_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    SetDst(inst.dst[0], ir.FindILsb(src0));
}

void Translator::S_FF1_I32_B64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    SetDst(inst.dst[0], ir.BallotFindLsb(src0));
}

/// The MSB counted from the left (FindUMsb counts from the right); -1 for 0.
void Translator::S_FLBIT_I32_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 msb = ir.FindUMsb(src0);
    const IR::U32 from_left = ir.ISub(ir.Imm32(31), msb);
    const IR::U1 nonzero = ir.INotEqual(src0, ir.Imm32(0));
    SetDst(inst.dst[0], IR::U32{ir.Select(nonzero, from_left, ir.Imm32(~0U))});
}

void Translator::S_FLBIT_I32_B64(const GcnInst& inst) {
    const IR::U64 src0{GetSrc64(inst.src[0])};
    const IR::U32 msb = ir.FindUMsb(src0);
    const IR::U32 from_left = ir.ISub(ir.Imm32(63), msb);
    const IR::U1 nonzero = ir.INotEqual(src0, ir.Imm64(u64(0u)));
    SetDst(inst.dst[0], IR::U32{ir.Select(nonzero, from_left, ir.Imm32(~0U))});
}

void Translator::S_BITSET_B32(const GcnInst& inst, u32 bit_value) {
    const IR::U32 old_value{GetSrc(inst.dst[0])};
    const IR::U32 offset{ir.BitFieldExtract(GetSrc(inst.src[0]), ir.Imm32(0U), ir.Imm32(5U))};
    SetDst(inst.dst[0], ir.BitFieldInsert(old_value, ir.Imm32(bit_value), offset, ir.Imm32(1U)));
}

void Translator::S_BITSET_B64(const GcnInst& inst, u32 bit_value) {
    const IR::U64 old_value{GetSrc64(inst.dst[0])};
    const IR::U32 offset{ir.BitwiseAnd(GetSrc(inst.src[0]), ir.Imm32(0x3F))};
    SetDst64(inst.dst[0],
             ir.BitFieldInsert(old_value, ir.Imm64(u64(bit_value)), offset, ir.Imm32(1U)));
}

/// D = EXEC; EXEC = op(EXEC, S0); SCC = EXEC != 0.
void Translator::S_SAVEEXEC_B64(NegateMode negate, bool is_or, const GcnInst& inst) {
    IR::U64 exec{ir.Ballot(ir.GetExec())};
    const IR::U64 src{GetSrc64(inst.src[0])};
    SetDst64(inst.dst[0], exec);
    if (negate == NegateMode::Src1) {
        exec = ir.BitwiseNot(exec);
    }
    IR::U64 result = is_or ? ir.BitwiseOr(exec, src) : ir.BitwiseAnd(exec, src);
    if (negate == NegateMode::Result) {
        result = ir.BitwiseNot(result);
    }
    const IR::U1 lanes = ir.InverseBallot(result);
    ir.SetExec(lanes);
    ir.SetScc(lanes);
}

void Translator::S_ABS_I32(const GcnInst& inst) {
    const IR::U32 result = ir.IAbs(GetSrc(inst.src[0]));
    SET_DST_SCC_NONZERO(result);
}

// SOPC

void Translator::S_CMP(ConditionOp cond, bool is_signed, const GcnInst& inst) {
    const IR::U32 lhs = GetSrc(inst.src[0]);
    const IR::U32 rhs = GetSrc(inst.src[1]);
    ir.SetScc(Compare(ir, cond, lhs, rhs, is_signed));
}

/// SCC = bit S1 of S0 is `compare_mode`.
void Translator::S_BITCMP(bool compare_mode, u32 bits, const GcnInst& inst) {
    const IR::U32 src1 = GetSrc(inst.src[1]);
    const IR::U32 bitpos{ir.BitwiseAnd(src1, ir.Imm32(bits == 64 ? 0x3f : 0x1f))};
    const IR::U32 bit = [&] -> IR::U32 {
        if (bits == 64) {
            const IR::U64 src0 = GetSrc64(inst.src[0]);
            return ir.UConvert(
                32, IR::U64{ir.BitwiseAnd(ir.ShiftRightLogical(src0, bitpos), ir.Imm64(u64(1)))});
        }
        ASSERT(bits == 32);
        const IR::U32 src0 = GetSrc(inst.src[0]);
        return ir.BitwiseAnd(ir.ShiftRightLogical(src0, bitpos), ir.Imm32(1));
    }();
    ir.SetScc(ir.IEqual(bit, ir.Imm32(compare_mode ? 1 : 0)));
}

#undef SET_DST_SCC_NONZERO

} // namespace Shader::Gcn
