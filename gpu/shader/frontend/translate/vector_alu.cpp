// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// VOP1/VOP2/VOP3/VOPC/VOP3P dispatch. Plain arithmetic is written here as the operation on
// its sources (read in operand order) through the Vop* templates; instructions with more to
// them are in vector_alu_ops.cpp.
#include <type_traits>

#include "shader/frontend/translate/translate.h"
#include "shader/ir/attribute.h"
#include "shader/profile.h"

namespace Shader::Gcn {

template <typename T>
T Translator::ReadSrc(const InstOperand& operand) {
    if constexpr (std::is_same_v<T, IR::U64> || std::is_same_v<T, IR::F64>) {
        return GetSrc64<T>(operand);
    } else {
        return GetSrc<T>(operand);
    }
}

#define WRITE_DST(value)                                                                           \
    if constexpr (dst64) {                                                                         \
        SetDst64(inst.dst[0], value);                                                              \
    } else {                                                                                       \
        SetDst(inst.dst[0], value);                                                                \
    }

template <typename T, bool dst64, typename Op>
void Translator::VopUnary(const GcnInst& inst, Op&& op) {
    const T a = ReadSrc<T>(inst.src[0]);
    WRITE_DST(op(a));
}

template <typename T, bool dst64, typename Op>
void Translator::VopBinary(const GcnInst& inst, Op&& op) {
    const T a = ReadSrc<T>(inst.src[0]);
    const T b = ReadSrc<T>(inst.src[1]);
    WRITE_DST(op(a, b));
}

template <typename T, bool dst64, typename Op>
void Translator::VopTernary(const GcnInst& inst, Op&& op) {
    const T a = ReadSrc<T>(inst.src[0]);
    const T b = ReadSrc<T>(inst.src[1]);
    const T c = ReadSrc<T>(inst.src[2]);
    WRITE_DST(op(a, b, c));
}

#undef WRITE_DST

template <typename T, bool is_signed, typename Op>
void Translator::Vop16(const GcnInst& inst, Op&& op) {
    const T a = GetSrc16<T, is_signed>(inst.src[0]);
    const T b = GetSrc16<T, is_signed>(inst.src[1]);
    if constexpr (std::is_invocable_v<Op&, T, T>) {
        SetDst16<is_signed>(inst.dst[0], op(a, b));
    } else {
        const T c = GetSrc16<T, is_signed>(inst.src[2]);
        SetDst16<is_signed>(inst.dst[0], op(a, b, c));
    }
}

template <typename T, bool is_signed, typename Op>
void Translator::VopPacked(const GcnInst& inst, Op&& op) {
    const auto a = GetSrcPk<T, is_signed>(inst.src[0]);
    const auto b = GetSrcPk<T, is_signed>(inst.src[1]);
    if constexpr (std::is_invocable_v<Op&, T, T>) {
        const auto lo = op(a.first, b.first);
        const auto hi = op(a.second, b.second);
        SetDstPk<T, is_signed>(inst.dst[0], {lo, hi});
    } else {
        const auto c = GetSrcPk<T, is_signed>(inst.src[2]);
        const auto lo = op(a.first, b.first, c.first);
        const auto hi = op(a.second, b.second, c.second);
        SetDstPk<T, is_signed>(inst.dst[0], {lo, hi});
    }
}

/// VOPC: D (VCC or an SGPR pair) = the lanes' compare results; V_CMPX also writes EXEC, and
/// both get EXEC & result (inactive lanes clear).
template <typename T>
void Translator::V_CMP(ConditionOp op, bool is_signed, bool set_exec, const GcnInst& inst) {
    constexpr bool is_float = std::is_same_v<T, IR::F32> || std::is_same_v<T, IR::F64>;
    const T src0 = ReadSrc<T>(inst.src[0]);
    const T src1 = ReadSrc<T>(inst.src[1]);
    const IR::U1 result = [&] -> IR::U1 {
        switch (op) {
        case ConditionOp::F:
            return ir.Imm1(false);
        case ConditionOp::TRU:
            return ir.Imm1(true);
        default:
            break;
        }
        if constexpr (is_float) {
            switch (op) {
            case ConditionOp::EQ:
                return ir.FPEqual(src0, src1);
            case ConditionOp::LG:
                return ir.FPNotEqual(src0, src1);
            case ConditionOp::GT:
                return ir.FPGreaterThan(src0, src1);
            case ConditionOp::LT:
                return ir.FPLessThan(src0, src1);
            case ConditionOp::LE:
                return ir.FPLessThanEqual(src0, src1);
            case ConditionOp::GE:
                return ir.FPGreaterThanEqual(src0, src1);
            case ConditionOp::U:
                return ir.LogicalOr(ir.FPIsNan(src0), ir.FPIsNan(src1));
            default:
                UNREACHABLE();
            }
        } else {
            switch (op) {
            case ConditionOp::EQ:
                return ir.IEqual(src0, src1);
            case ConditionOp::LG:
                return ir.INotEqual(src0, src1);
            case ConditionOp::GT:
                return ir.IGreaterThan(src0, src1, is_signed);
            case ConditionOp::LT:
                return ir.ILessThan(src0, src1, is_signed);
            case ConditionOp::LE:
                return ir.ILessThanEqual(src0, src1, is_signed);
            case ConditionOp::GE:
                return ir.IGreaterThanEqual(src0, src1, is_signed);
            default:
                UNREACHABLE();
            }
        }
    }();
    if (set_exec) {
        const IR::U1 masked{ir.LogicalAnd(ir.GetExec(), result)};
        ir.SetExec(masked);
        SetDst64(inst.dst[1], ir.Ballot(masked));
        return;
    }
    SetDst64(inst.dst[1], ir.Ballot(result));
}

void Translator::EmitVectorAlu(const GcnInst& inst) {
    using F32 = IR::F32;
    using F64 = IR::F64;
    using U32 = IR::U32;
    using U64 = IR::U64;
    using enum ConditionOp;
    constexpr bool Wide = true; // a 64-bit result
    const auto sh5 = [&](const U32& amount) { return ir.BitwiseAnd(amount, ir.Imm32(0x1F)); };
    const auto sh6 = [&](const U64& amount) { return ir.BitwiseAnd(amount, ir.Imm64(u64(0x3F))); };
    const auto sh4 = [&](const U32& amount) { return ir.BitwiseAnd(amount, ir.Imm32(0xF)); };

    switch (inst.opcode) {
    // VOP2
    case Opcode::V_CNDMASK_B32:
        return V_CNDMASK_B32(inst);
    case Opcode::V_READLANE_B32:
        return V_READLANE_B32(inst);
    case Opcode::V_WRITELANE_B32:
        return V_WRITELANE_B32(inst);
    case Opcode::V_ADD_F32:
        return VopBinary<F32>(inst, [&](F32 a, F32 b) { return ir.FPAdd(a, b); });
    case Opcode::V_SUB_F32:
        return VopBinary<F32>(inst, [&](F32 a, F32 b) { return ir.FPSub(a, b); });
    case Opcode::V_SUBREV_F32:
        return VopBinary<F32>(inst, [&](F32 a, F32 b) { return ir.FPSub(b, a); });
    case Opcode::V_MAC_LEGACY_F32:
    case Opcode::V_MAC_F32:
        return V_MAC_F32(inst);
    case Opcode::V_MUL_LEGACY_F32:
        return V_MUL_LEGACY_F32(inst);
    case Opcode::V_MUL_F32:
        return V_MUL_F32(inst);
    case Opcode::V_MUL_I32_I24:
        return V_MUL_I32_I24(inst, true);
    case Opcode::V_MUL_U32_U24:
        return V_MUL_I32_I24(inst, false);
    case Opcode::V_MIN_LEGACY_F32:
        return V_MIN_F32(inst, true);
    case Opcode::V_MAX_LEGACY_F32:
        return V_MAX_F32(inst, true);
    case Opcode::V_MIN_F32:
        return V_MIN_F32(inst, false);
    case Opcode::V_MAX_F32:
        return V_MAX_F32(inst);
    case Opcode::V_MIN_I32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.SMin(a, b); });
    case Opcode::V_MAX_I32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.IMax(a, b, true); });
    case Opcode::V_MIN_U32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.IMin(a, b, false); });
    case Opcode::V_MAX_U32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.IMax(a, b, false); });
    case Opcode::V_LSHR_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.ShiftRightLogical(a, sh5(b)); });
    case Opcode::V_LSHRREV_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.ShiftRightLogical(b, sh5(a)); });
    case Opcode::V_ASHR_I32:
        return VopBinary<U32>(inst,
                              [&](U32 a, U32 b) { return ir.ShiftRightArithmetic(a, sh5(b)); });
    case Opcode::V_ASHRREV_I32:
        return VopBinary<U32>(inst,
                              [&](U32 a, U32 b) { return ir.ShiftRightArithmetic(b, sh5(a)); });
    case Opcode::V_LSHL_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.ShiftLeftLogical(a, sh5(b)); });
    case Opcode::V_LSHLREV_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.ShiftLeftLogical(b, sh5(a)); });
    case Opcode::V_AND_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.BitwiseAnd(a, b); });
    case Opcode::V_OR_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return U32{ir.BitwiseOr(a, b)}; });
    case Opcode::V_XOR_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return U32{ir.BitwiseXor(a, b)}; });
    case Opcode::V_BFM_B32:
        return V_BFM_B32(inst);
    case Opcode::V_MADMK_F32: // D = S0 * K + S1
        return VopTernary<F32>(
            inst, [&](F32 a, F32 b, F32 k) { return ir.FPAdd(ir.FPMul(a, k), b); });
    case Opcode::V_MADAK_F32: // D = S0 * S1 + K
    case Opcode::V_FMA_F32:
        return VopTernary<F32>(inst, [&](F32 a, F32 b, F32 c) { return ir.FPFma(a, b, c); });
    case Opcode::V_BCNT_U32_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.IAdd(ir.BitCount(a), b); });
    case Opcode::V_MBCNT_LO_U32_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.MaskedBitCount(a, b, false); });
    case Opcode::V_MBCNT_HI_U32_B32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.MaskedBitCount(a, b, true); });
    case Opcode::V_ADD_I32:
        return V_ADD_I32(inst);
    case Opcode::V_SUB_I32:
        return V_SUB_I32(inst);
    case Opcode::V_SUBREV_I32:
        return V_SUBREV_I32(inst);
    case Opcode::V_ADDC_U32:
        return V_ADDC_U32(inst);
    case Opcode::V_SUBB_U32:
        return V_SUBB_U32(inst);
    case Opcode::V_SUBBREV_U32:
        return V_SUBBREV_U32(inst);
    case Opcode::V_LDEXP_F32:
        return V_LDEXP_F32(inst);
    case Opcode::V_CVT_PKNORM_U16_F32:
        return V_CVT_PKNORM_U16_F32(inst);
    case Opcode::V_CVT_PKNORM_I16_F32:
        return V_CVT_PKNORM_I16_F32(inst);
    case Opcode::V_CVT_PKRTZ_F16_F32:
        return V_CVT_PKRTZ_F16_F32(inst);
    case Opcode::V_ADD_F16:
        return Vop16<F32>(inst, [&](F32 a, F32 b) { return ir.FPAdd(a, b); });
    case Opcode::V_SUB_F16:
        return Vop16<F32>(inst, [&](F32 a, F32 b) { return ir.FPSub(a, b); });
    case Opcode::V_MUL_F16:
        return Vop16<F32>(inst, [&](F32 a, F32 b) { return ir.FPMul(a, b); });
    case Opcode::V_MAX_F16:
        return Vop16<F32>(inst, [&](F32 a, F32 b) { return ir.FPMax(a, b); });
    case Opcode::V_MIN_F16:
        return Vop16<F32>(inst, [&](F32 a, F32 b) { return ir.FPMin(a, b); });

    // VOP1
    case Opcode::V_MOV_B32:
        return V_MOV(inst);
    case Opcode::V_READFIRSTLANE_B32:
        return V_READFIRSTLANE_B32(inst);
    case Opcode::V_CVT_I32_F64:
        return VopUnary<F64>(inst, [&](F64 a) { return ir.ConvertFToS(32, a); });
    case Opcode::V_CVT_F64_I32:
        return VopUnary<U32, Wide>(inst, [&](U32 a) { return ir.ConvertSToF(64, 32, a); });
    case Opcode::V_CVT_F64_U32:
        return VopUnary<U32, Wide>(inst, [&](U32 a) { return ir.ConvertUToF(64, 32, a); });
    case Opcode::V_CVT_F32_I32:
        return VopUnary<U32>(inst, [&](U32 a) { return ir.ConvertSToF(32, 32, a); });
    case Opcode::V_CVT_F32_U32:
        return VopUnary<U32>(inst, [&](U32 a) { return ir.ConvertUToF(32, 32, a); });
    case Opcode::V_CVT_U32_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.ConvertFToU(32, a); });
    case Opcode::V_CVT_I32_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.ConvertFToS(32, a); });
    case Opcode::V_CVT_F16_F32:
        return V_CVT_F16_F32(inst);
    case Opcode::V_CVT_F32_F16:
        return V_CVT_F32_F16(inst);
    case Opcode::V_CVT_FLR_I32_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.ConvertFToI(32, true, ir.FPFloor(a)); });
    case Opcode::V_CVT_RPI_I32_F32: // round half up
        return VopUnary<F32>(inst, [&](F32 a) {
            return ir.ConvertFToI(32, true, ir.FPFloor(ir.FPAdd(a, ir.Imm32(0.5f))));
        });
    case Opcode::V_CVT_F32_F64:
        return VopUnary<F64>(inst, [&](F64 a) { return ir.FPConvert(32, a); });
    case Opcode::V_CVT_F64_F32:
        return VopUnary<F32, Wide>(inst, [&](F32 a) { return ir.FPConvert(64, a); });
    case Opcode::V_CVT_OFF_F32_I4:
        return V_CVT_OFF_F32_I4(inst);
    case Opcode::V_CVT_F32_UBYTE0:
        return V_CVT_F32_UBYTE(0, inst);
    case Opcode::V_CVT_F32_UBYTE1:
        return V_CVT_F32_UBYTE(1, inst);
    case Opcode::V_CVT_F32_UBYTE2:
        return V_CVT_F32_UBYTE(2, inst);
    case Opcode::V_CVT_F32_UBYTE3:
        return V_CVT_F32_UBYTE(3, inst);
    case Opcode::V_FLOOR_F64:
        return VopUnary<F64, Wide>(inst, [&](F64 a) { return ir.FPFloor(a); });
    case Opcode::V_FRACT_F64:
        return VopUnary<F64, Wide>(inst, [&](F64 a) { return ir.FPFract(a); });
    case Opcode::V_RCP_F64:
        return VopUnary<F64, Wide>(inst, [&](F64 a) { return ir.FPRecip(a); });
    case Opcode::V_FREXP_MANT_F64:
        return VopUnary<F64, Wide>(inst, [&](F64 a) { return ir.FPFrexpSig(a); });
    case Opcode::V_FREXP_EXP_I32_F64:
        return VopUnary<F64>(inst, [&](F64 a) { return ir.FPFrexpExp(a); });
    case Opcode::V_FRACT_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPFract(a); });
    case Opcode::V_TRUNC_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPTrunc(a); });
    case Opcode::V_CEIL_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPCeil(a); });
    case Opcode::V_RNDNE_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPRoundEven(a); });
    case Opcode::V_FLOOR_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPFloor(a); });
    case Opcode::V_EXP_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPExp2(a); });
    case Opcode::V_LOG_CLAMP_F32:
    case Opcode::V_LOG_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPLog2(a); });
    case Opcode::V_RCP_F32:
    case Opcode::V_RCP_IFLAG_F32:
    case Opcode::V_RCP_CLAMP_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPRecip(a); });
    case Opcode::V_RCP_LEGACY_F32:
        return V_RCP_LEGACY_F32(inst);
    case Opcode::V_RSQ_F32:
    case Opcode::V_RSQ_CLAMP_F32:
    case Opcode::V_RSQ_LEGACY_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPRecipSqrt(a); });
    case Opcode::V_SQRT_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPSqrt(a); });
    case Opcode::V_SIN_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPSin(a); });
    case Opcode::V_COS_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPCos(a); });
    case Opcode::V_FREXP_EXP_I32_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPFrexpExp(a); });
    case Opcode::V_FREXP_MANT_F32:
        return VopUnary<F32>(inst, [&](F32 a) { return ir.FPFrexpSig(a); });
    case Opcode::V_NOT_B32:
        return VopUnary<U32>(inst, [&](U32 a) { return ir.BitwiseNot(a); });
    case Opcode::V_BFREV_B32:
        return VopUnary<U32>(inst, [&](U32 a) { return ir.BitReverse(a); });
    case Opcode::V_FFBL_B32:
        return VopUnary<U32>(inst, [&](U32 a) { return ir.FindILsb(a); });
    case Opcode::V_FFBH_U32:
        return V_FFBH_U32(inst);
    case Opcode::V_FFBH_I32:
        return V_FFBH_I32(inst);
    case Opcode::V_MOVRELD_B32:
        return V_MOVRELD_B32(inst);
    case Opcode::V_MOVRELS_B32:
        return V_MOVRELS_B32(inst);
    case Opcode::V_MOVRELSD_B32:
        return V_MOVRELSD_B32(inst);

    // VOPC (the N* forms are the negated compares, which equal the plain ones but for NaN)
    case Opcode::V_CMP_F_F32:
        return V_CMP<F32>(F, false, false, inst);
    case Opcode::V_CMP_LT_F32:
    case Opcode::V_CMP_NGE_F32:
        return V_CMP<F32>(LT, false, false, inst);
    case Opcode::V_CMP_EQ_F32:
        return V_CMP<F32>(EQ, false, false, inst);
    case Opcode::V_CMP_LE_F32:
    case Opcode::V_CMP_NGT_F32:
        return V_CMP<F32>(LE, false, false, inst);
    case Opcode::V_CMP_GT_F32:
    case Opcode::V_CMP_NLE_F32:
        return V_CMP<F32>(GT, false, false, inst);
    case Opcode::V_CMP_LG_F32:
    case Opcode::V_CMP_NEQ_F32:
        return V_CMP<F32>(LG, false, false, inst);
    case Opcode::V_CMP_GE_F32:
    case Opcode::V_CMP_NLT_F32:
        return V_CMP<F32>(GE, false, false, inst);
    case Opcode::V_CMP_U_F32:
        return V_CMP<F32>(U, false, false, inst);
    case Opcode::V_CMPX_F_F32:
        return V_CMP<F32>(F, false, true, inst);
    case Opcode::V_CMPX_LT_F32:
    case Opcode::V_CMPX_NGE_F32:
        return V_CMP<F32>(LT, false, true, inst);
    case Opcode::V_CMPX_EQ_F32:
    case Opcode::V_CMPX_NLG_F32:
        return V_CMP<F32>(EQ, false, true, inst);
    case Opcode::V_CMPX_LE_F32:
    case Opcode::V_CMPX_NGT_F32:
        return V_CMP<F32>(LE, false, true, inst);
    case Opcode::V_CMPX_GT_F32:
    case Opcode::V_CMPX_NLE_F32:
        return V_CMP<F32>(GT, false, true, inst);
    case Opcode::V_CMPX_LG_F32:
    case Opcode::V_CMPX_NEQ_F32:
        return V_CMP<F32>(LG, false, true, inst);
    case Opcode::V_CMPX_GE_F32:
    case Opcode::V_CMPX_NLT_F32:
        return V_CMP<F32>(GE, false, true, inst);
    case Opcode::V_CMPX_TRU_F32:
        return V_CMP<F32>(TRU, false, true, inst);
    case Opcode::V_CMP_F_F64:
        return V_CMP<F64>(F, false, false, inst);
    case Opcode::V_CMP_LT_F64:
    case Opcode::V_CMP_NGE_F64:
        return V_CMP<F64>(LT, false, false, inst);
    case Opcode::V_CMP_EQ_F64:
        return V_CMP<F64>(EQ, false, false, inst);
    case Opcode::V_CMP_LE_F64:
    case Opcode::V_CMP_NGT_F64:
        return V_CMP<F64>(LE, false, false, inst);
    case Opcode::V_CMP_GT_F64:
    case Opcode::V_CMP_NLE_F64:
        return V_CMP<F64>(GT, false, false, inst);
    case Opcode::V_CMP_LG_F64:
    case Opcode::V_CMP_NEQ_F64:
        return V_CMP<F64>(LG, false, false, inst);
    case Opcode::V_CMP_GE_F64:
    case Opcode::V_CMP_NLT_F64:
        return V_CMP<F64>(GE, false, false, inst);
    case Opcode::V_CMP_U_F64:
        return V_CMP<F64>(U, false, false, inst);
    case Opcode::V_CMP_LT_I32:
        return V_CMP<U32>(LT, true, false, inst);
    case Opcode::V_CMP_EQ_I32:
        return V_CMP<U32>(EQ, true, false, inst);
    case Opcode::V_CMP_LE_I32:
        return V_CMP<U32>(LE, true, false, inst);
    case Opcode::V_CMP_GT_I32:
        return V_CMP<U32>(GT, true, false, inst);
    case Opcode::V_CMP_NE_I32:
        return V_CMP<U32>(LG, true, false, inst);
    case Opcode::V_CMP_GE_I32:
        return V_CMP<U32>(GE, true, false, inst);
    case Opcode::V_CMPX_LT_I32:
        return V_CMP<U32>(LT, true, true, inst);
    case Opcode::V_CMPX_EQ_I32:
        return V_CMP<U32>(EQ, true, true, inst);
    case Opcode::V_CMPX_LE_I32:
        return V_CMP<U32>(LE, true, true, inst);
    case Opcode::V_CMPX_GT_I32:
        return V_CMP<U32>(GT, true, true, inst);
    case Opcode::V_CMPX_LG_I32:
        return V_CMP<U32>(LG, true, true, inst);
    case Opcode::V_CMPX_GE_I32:
        return V_CMP<U32>(GE, true, true, inst);
    case Opcode::V_CMP_F_U32:
        return V_CMP<U32>(F, false, false, inst);
    case Opcode::V_CMP_LT_U32:
        return V_CMP<U32>(LT, false, false, inst);
    case Opcode::V_CMP_EQ_U32:
        return V_CMP<U32>(EQ, false, false, inst);
    case Opcode::V_CMP_LE_U32:
        return V_CMP<U32>(LE, false, false, inst);
    case Opcode::V_CMP_GT_U32:
        return V_CMP<U32>(GT, false, false, inst);
    case Opcode::V_CMP_NE_U32:
        return V_CMP<U32>(LG, false, false, inst);
    case Opcode::V_CMP_GE_U32:
        return V_CMP<U32>(GE, false, false, inst);
    case Opcode::V_CMP_TRU_U32:
        return V_CMP<U32>(TRU, false, false, inst);
    case Opcode::V_CMPX_F_U32:
        return V_CMP<U32>(F, false, true, inst);
    case Opcode::V_CMPX_LT_U32:
        return V_CMP<U32>(LT, false, true, inst);
    case Opcode::V_CMPX_EQ_U32:
        return V_CMP<U32>(EQ, false, true, inst);
    case Opcode::V_CMPX_LE_U32:
        return V_CMP<U32>(LE, false, true, inst);
    case Opcode::V_CMPX_GT_U32:
        return V_CMP<U32>(GT, false, true, inst);
    case Opcode::V_CMPX_NE_U32:
        return V_CMP<U32>(LG, false, true, inst);
    case Opcode::V_CMPX_GE_U32:
        return V_CMP<U32>(GE, false, true, inst);
    case Opcode::V_CMPX_TRU_U32:
        return V_CMP<U32>(TRU, false, true, inst);
    case Opcode::V_CMP_F_I64:
        return V_CMP<U64>(F, true, false, inst);
    case Opcode::V_CMP_LT_I64:
        return V_CMP<U64>(LT, true, false, inst);
    case Opcode::V_CMP_EQ_I64:
        return V_CMP<U64>(EQ, true, false, inst);
    case Opcode::V_CMP_LE_I64:
        return V_CMP<U64>(LE, true, false, inst);
    case Opcode::V_CMP_GT_I64:
        return V_CMP<U64>(GT, true, false, inst);
    case Opcode::V_CMP_NE_I64:
        return V_CMP<U64>(LG, true, false, inst);
    case Opcode::V_CMP_GE_I64:
        return V_CMP<U64>(GE, true, false, inst);
    case Opcode::V_CMP_TRU_I64:
        return V_CMP<U64>(TRU, true, false, inst);
    case Opcode::V_CMPX_EQ_I64:
        return V_CMP<U64>(EQ, true, true, inst);
    case Opcode::V_CMP_EQ_U64:
        return V_CMP<U64>(EQ, false, false, inst);
    case Opcode::V_CMP_NE_U64:
        return V_CMP<U64>(LG, false, false, inst);
    case Opcode::V_CMP_GT_U64:
        return V_CMP<U64>(GT, false, false, inst);
    case Opcode::V_CMP_LT_U64:
        return V_CMP<U64>(LT, false, false, inst);
    case Opcode::V_CMPX_EQ_U64:
        return V_CMP<U64>(EQ, false, true, inst);
    case Opcode::V_CMPX_LG_U64:
        return V_CMP<U64>(LG, false, true, inst);
    case Opcode::V_CMP_CLASS_F32:
        return V_CMP_CLASS_F32(inst);

    // VOP3a
    case Opcode::V_MAD_LEGACY_F32:
    case Opcode::V_MAD_F32:
        return VopTernary<F32>(
            inst, [&](F32 a, F32 b, F32 c) { return ir.FPAdd(ir.FPMul(a, b), c); });
    case Opcode::V_MAD_I32_I24:
        return V_MAD_I32_I24(inst, true);
    case Opcode::V_MAD_U32_U24:
        return V_MAD_I32_I24(inst, false);
    case Opcode::V_CUBEID_F32:
        return V_CUBEID_F32(inst);
    case Opcode::V_CUBESC_F32:
        return V_CUBESC_F32(inst);
    case Opcode::V_CUBETC_F32:
        return V_CUBETC_F32(inst);
    case Opcode::V_CUBEMA_F32:
        return V_CUBEMA_F32(inst);
    case Opcode::V_BFE_U32:
        return V_BFE_U32(false, inst);
    case Opcode::V_BFE_I32:
        return V_BFE_U32(true, inst);
    case Opcode::V_BFI_B32: // (S0 & S1) | (~S0 & S2)
        return VopTernary<U32>(inst, [&](U32 a, U32 b, U32 c) {
            return ir.BitwiseOr(ir.BitwiseAnd(a, b), ir.BitwiseAnd(ir.BitwiseNot(a), c));
        });
    case Opcode::V_FMA_F64:
        return VopTernary<F64, Wide>(inst,
                                     [&](F64 a, F64 b, F64 c) { return ir.FPFma(a, b, c); });
    case Opcode::V_MIN3_F32:
        return VopTernary<F32>(inst, [&](F32 a, F32 b, F32 c) { return ir.FPMinTri(a, b, c); });
    case Opcode::V_MIN3_I32:
        return VopTernary<U32>(inst,
                               [&](U32 a, U32 b, U32 c) { return ir.IMinTri(a, b, c, true); });
    case Opcode::V_MIN3_U32:
        return VopTernary<U32>(inst,
                               [&](U32 a, U32 b, U32 c) { return ir.IMinTri(a, b, c, false); });
    case Opcode::V_MAX3_F32:
        return VopTernary<F32>(inst, [&](F32 a, F32 b, F32 c) { return ir.FPMaxTri(a, b, c); });
    case Opcode::V_MAX3_I32:
        return VopTernary<U32>(inst,
                               [&](U32 a, U32 b, U32 c) { return ir.IMaxTri(a, b, c, true); });
    case Opcode::V_MAX3_U32:
        return VopTernary<U32>(inst,
                               [&](U32 a, U32 b, U32 c) { return ir.IMaxTri(a, b, c, false); });
    case Opcode::V_MED3_F32:
        return VopTernary<F32>(inst, [&](F32 a, F32 b, F32 c) { return ir.FPMedTri(a, b, c); });
    case Opcode::V_MED3_I32:
        return VopTernary<U32>(inst,
                               [&](U32 a, U32 b, U32 c) { return ir.IMedTri(a, b, c, true); });
    case Opcode::V_MED3_U32:
        return VopTernary<U32>(inst,
                               [&](U32 a, U32 b, U32 c) { return ir.IMedTri(a, b, c, false); });
    case Opcode::V_SAD_U32:
        return V_SAD_U32(inst);
    case Opcode::V_CVT_PK_U16_U32:
        return V_CVT_PK_U16_U32(inst);
    case Opcode::V_CVT_PK_I16_I32:
        return V_CVT_PK_I16_I32(inst);
    case Opcode::V_CVT_PK_U8_F32:
        return V_CVT_PK_U8_F32(inst);
    case Opcode::V_LSHL_B64:
        return VopBinary<U64, Wide>(inst,
                                    [&](U64 a, U64 b) { return ir.ShiftLeftLogical(a, sh6(b)); });
    case Opcode::V_LSHR_B64:
        return VopBinary<U64, Wide>(inst,
                                    [&](U64 a, U64 b) { return ir.ShiftRightLogical(a, sh6(b)); });
    case Opcode::V_ASHR_I64:
        return VopBinary<U64, Wide>(
            inst, [&](U64 a, U64 b) { return ir.ShiftRightArithmetic(a, sh6(b)); });
    case Opcode::V_ADD_F64:
        return VopBinary<F64, Wide>(inst, [&](F64 a, F64 b) { return ir.FPAdd(a, b); });
    case Opcode::V_MUL_F64:
        return VopBinary<F64, Wide>(inst, [&](F64 a, F64 b) { return ir.FPMul(a, b); });
    case Opcode::V_MAX_F64:
        return VopBinary<F64, Wide>(inst, [&](F64 a, F64 b) { return ir.FPMax(a, b); });
    case Opcode::V_ALIGNBIT_B32:
        return V_ALIGNBIT_B32(inst);
    case Opcode::V_ALIGNBYTE_B32:
        return V_ALIGNBYTE_B32(inst);
    case Opcode::V_MUL_LO_U32:
    case Opcode::V_MUL_LO_I32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.IMul(a, b); });
    case Opcode::V_MUL_HI_U32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.IMulHi(a, b, false); });
    case Opcode::V_MUL_HI_I32:
        return VopBinary<U32>(inst, [&](U32 a, U32 b) { return ir.IMulHi(a, b, true); });
    case Opcode::V_MAD_U64_U32:
        return V_MAD_U64_U32(inst);
    case Opcode::V_LSHLREV_B16:
        return Vop16<U32>(inst, [&](U32 s, U32 v) { return ir.ShiftLeftLogical(v, sh4(s)); });
    case Opcode::V_LSHRREV_B16:
        return Vop16<U32>(inst, [&](U32 s, U32 v) { return ir.ShiftRightLogical(v, sh4(s)); });
    case Opcode::V_ASHRREV_I16:
        return Vop16<U32, true>(inst,
                                [&](U32 s, U32 v) { return ir.ShiftRightArithmetic(v, sh4(s)); });
    case Opcode::V_ADD_LSHL_U32: // (S0 + S1) << S2
        return VopTernary<U32>(inst, [&](U32 a, U32 b, U32 c) {
            const auto shift = sh5(c);
            return ir.ShiftLeftLogical(ir.IAdd(a, b), shift);
        });
    case Opcode::V_LSHL_ADD_U32: // (S0 << S1) + S2
        return VopTernary<U32>(inst, [&](U32 a, U32 b, U32 c) {
            const auto shift = sh5(b);
            return ir.IAdd(ir.ShiftLeftLogical(a, shift), c);
        });
    case Opcode::V_LSHL_OR_B32: // (S0 << S1) | S2
        return VopTernary<U32>(inst, [&](U32 a, U32 b, U32 c) {
            const auto shift = sh5(b);
            return ir.BitwiseOr(ir.ShiftLeftLogical(a, shift), c);
        });
    case Opcode::V_AND_OR_B32:
        return VopTernary<U32>(
            inst, [&](U32 a, U32 b, U32 c) { return ir.BitwiseOr(ir.BitwiseAnd(a, b), c); });
    case Opcode::V_OR3_B32:
        return VopTernary<U32>(
            inst, [&](U32 a, U32 b, U32 c) { return ir.BitwiseOr(ir.BitwiseOr(a, b), c); });
    case Opcode::V_ADD3_U32:
        return VopTernary<U32>(inst,
                               [&](U32 a, U32 b, U32 c) { return ir.IAdd(a, ir.IAdd(b, c)); });
    case Opcode::V_MIN3_F16:
        return Vop16<F32>(inst, [&](F32 a, F32 b, F32 c) { return ir.FPMinTri(a, b, c); });
    case Opcode::V_MAX3_F16:
        return Vop16<F32>(inst, [&](F32 a, F32 b, F32 c) { return ir.FPMaxTri(a, b, c); });
    case Opcode::V_MED3_F16:
        return Vop16<F32>(inst, [&](F32 a, F32 b, F32 c) { return ir.FPMedTri(a, b, c); });
    case Opcode::V_NOP:
        return;

    // VOP3P
    case Opcode::V_PK_MUL_LO_U16:
        return VopPacked<U32>(inst, [&](U32 a, U32 b) { return ir.IMul(a, b); });
    case Opcode::V_PK_ADD_I16:
        return VopPacked<U32, true>(inst, [&](U32 a, U32 b) { return ir.IAdd(a, b); });
    case Opcode::V_PK_SUB_I16:
        return VopPacked<U32, true>(inst, [&](U32 a, U32 b) { return ir.ISub(a, b); });
    case Opcode::V_PK_LSHLREV_B16:
        return VopPacked<U32>(inst, [&](U32 s, U32 v) { return ir.ShiftLeftLogical(v, s); });
    case Opcode::V_PK_LSHRREV_B16:
        return VopPacked<U32>(inst, [&](U32 s, U32 v) { return ir.ShiftRightLogical(v, s); });
    case Opcode::V_PK_MAD_U16:
        return VopPacked<U32>(
            inst, [&](U32 a, U32 b, U32 c) { return ir.IAdd(ir.IMul(a, b), c); });
    case Opcode::V_PK_ADD_U16:
        return VopPacked<U32>(inst, [&](U32 a, U32 b) { return ir.IAdd(a, b); });
    case Opcode::V_PK_SUB_U16:
        return VopPacked<U32>(inst, [&](U32 a, U32 b) { return ir.ISub(a, b); });
    case Opcode::V_PK_MAX_U16:
        return VopPacked<U32>(inst, [&](U32 a, U32 b) { return ir.UMax(a, b); });
    case Opcode::V_PK_MIN_U16:
        return VopPacked<U32>(inst, [&](U32 a, U32 b) { return ir.UMin(a, b); });
    case Opcode::V_PK_FMA_F16:
        return VopPacked<F32>(inst, [&](F32 a, F32 b, F32 c) { return ir.FPFma(a, b, c); });
    case Opcode::V_PK_ADD_F16:
        return VopPacked<F32>(inst, [&](F32 a, F32 b) { return ir.FPAdd(a, b); });
    case Opcode::V_PK_MUL_F16:
        return VopPacked<F32>(inst, [&](F32 a, F32 b) { return ir.FPMul(a, b); });
    case Opcode::V_PK_MIN_F16:
        return VopPacked<F32>(inst, [&](F32 a, F32 b) { return ir.FPMin(a, b); });
    case Opcode::V_PK_MAX_F16:
        return VopPacked<F32>(inst, [&](F32 a, F32 b) { return ir.FPMax(a, b); });
    case Opcode::V_MAD_MIX_F32:
        return V_MAD_MIX_F32(inst);
    case Opcode::V_MAD_MIXLO_F16:
        return V_MAD_MIXLO_F16(inst);
    case Opcode::V_MAD_MIXHI_F16:
        return V_MAD_MIXHI_F16(inst);
    default:
        LogMissingOpcode(inst);
    }
}

} // namespace Shader::Gcn
