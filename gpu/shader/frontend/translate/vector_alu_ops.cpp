// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Vector ALU instructions with their own semantics: carries through VCC or an SGPR pair, legacy
// float rules, 24-bit multiplies, F16 conversions with and without host F16, cube map
// coordinates, relative VGPR addressing (M0), packing and mixed precision.
#include "shader/frontend/translate/translate.h"
#include "shader/profile.h"

namespace Shader::Gcn {

// Carries: VOP3 forms name an SGPR pair (src2 / dst1), VOP2 forms use VCC.

IR::U32 Translator::GetCarryIn(const GcnInst& inst) {
    IR::U64 carry;
    if (inst.src_count == 3) {
        carry = GetSrc64(inst.src[2]);
    } else {
        carry = ir.PackUint2x32(ir.CompositeConstruct(ir.GetVccLo(), ir.GetVccHi()));
    }
    return IR::U32{ir.Select(ir.InverseBallot(carry), ir.Imm32(1), ir.Imm32(0))};
}

void Translator::SetCarryOut(const GcnInst& inst, const IR::U1& carry) {
    if (inst.dst_count == 2) {
        SetDst64(inst.dst[1], ir.Ballot(carry));
        return;
    }
    const auto halves = ir.UnpackUint2x32(ir.Ballot(carry));
    const IR::U32 lo{ir.CompositeExtract(halves, 0U)};
    const IR::U32 hi{ir.CompositeExtract(halves, 1U)};
    ir.SetVccLo(lo);
    ir.SetVccHi(hi);
}

// VOP2

/// D = mask ? S1 : S0, the mask from VCC (VOP2) or an SGPR pair (VOP3).
void Translator::V_CNDMASK_B32(const GcnInst& inst) {
    IR::U64 mask;
    if (inst.src[2].field == OperandField::Undefined) {
        mask = ir.PackUint2x32(ir.CompositeConstruct(ir.GetVccLo(), ir.GetVccHi()));
    } else {
        mask = GetSrc64(inst.src[2]);
    }
    const IR::Value result = ir.Select(ir.InverseBallot(mask), GetSrc<IR::F32>(inst.src[1]),
                                       GetSrc<IR::F32>(inst.src[0]));
    SetDst(inst.dst[0], IR::U32F32{result});
}

void Translator::V_MUL_F32(const GcnInst& inst) {
    SetDst(inst.dst[0], ir.FPMul(GetSrc<IR::F32>(inst.src[0]), GetSrc<IR::F32>(inst.src[1])));
}

/// Legacy multiply: 0 times anything (even Inf or NaN) is +0.
void Translator::V_MUL_LEGACY_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 zero{ir.Imm32(0.0f)};
    const IR::U1 src0_zero{ir.FPEqual(src0, zero)};
    const IR::U1 src1_zero{ir.FPEqual(src1, zero)};
    const IR::U1 either_zero{ir.LogicalOr(src0_zero, src1_zero)};
    SetDst(inst.dst[0], IR::F32{ir.Select(either_zero, zero, ir.FPMul(src0, src1))});
}

void Translator::V_MUL_I32_I24(const GcnInst& inst, bool is_signed) {
    const IR::U32 src0{
        ir.BitFieldExtract(GetSrc(inst.src[0]), ir.Imm32(0), ir.Imm32(24), is_signed)};
    const IR::U32 src1{
        ir.BitFieldExtract(GetSrc(inst.src[1]), ir.Imm32(0), ir.Imm32(24), is_signed)};
    SetDst(inst.dst[0], ir.IMul(src0, src1));
}

/// Legacy min/max (DX9 rules): a NaN operand gives S1.
void Translator::V_MIN_F32(const GcnInst& inst, bool is_legacy) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 min = ir.FPMin(src0, src1);
    const IR::F32 result =
        is_legacy
            ? IR::F32{ir.Select(ir.LogicalOr(ir.FPIsNan(src0), ir.FPIsNan(src1)), src1, min)}
            : min;
    SetDst(inst.dst[0], result);
}

void Translator::V_MAX_F32(const GcnInst& inst, bool is_legacy) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::F32 src1{GetSrc<IR::F32>(inst.src[1])};
    const IR::F32 max = ir.FPMax(src0, src1);
    const IR::F32 result =
        is_legacy
            ? IR::F32{ir.Select(ir.LogicalOr(ir.FPIsNan(src0), ir.FPIsNan(src1)), src1, max)}
            : max;
    SetDst(inst.dst[0], result);
}

/// D = ((1 << S0[4:0]) - 1) << S1[4:0].
void Translator::V_BFM_B32(const GcnInst& inst) {
    const IR::U32 width{ir.BitFieldExtract(GetSrc(inst.src[0]), ir.Imm32(0), ir.Imm32(5))};
    const IR::U32 offset{ir.BitFieldExtract(GetSrc(inst.src[1]), ir.Imm32(0), ir.Imm32(5))};
    const IR::U32 ones = ir.ISub(ir.ShiftLeftLogical(ir.Imm32(1), width), ir.Imm32(1));
    SetDst(inst.dst[0], ir.ShiftLeftLogical(ones, offset));
}

/// D = S0 * S1 + D.
void Translator::V_MAC_F32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::F32>(inst.src[0]);
    const auto src1 = GetSrc<IR::F32>(inst.src[1]);
    const auto dst = GetSrc<IR::F32>(inst.dst[0]);
    SetDst(inst.dst[0], ir.FPAdd(ir.FPMul(src0, src1), dst));
}

void Translator::V_ADD_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 result{ir.IAdd(src0, src1)};
    SetDst(inst.dst[0], result);
    SetCarryOut(inst, ir.ILessThan(result, src0, false));
}

void Translator::V_SUB_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], IR::U32{ir.ISub(src0, src1)});
    const IR::U1 borrow{ir.IGreaterThan(src1, src0, false)};
    SetCarryOut(inst, borrow);
}

void Translator::V_SUBREV_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], IR::U32{ir.ISub(src1, src0)});
    const IR::U1 borrow{ir.IGreaterThan(src0, src1, false)};
    SetCarryOut(inst, borrow);
}

void Translator::V_ADDC_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 carry{GetCarryIn(inst)};
    const IR::U32 sum{ir.IAdd(src0, src1)};
    const IR::U32 result{ir.IAdd(sum, carry)};
    const IR::U1 carry_out1{ir.ILessThan(sum, src0, false)};
    const IR::U1 carry_out2{ir.ILessThan(result, sum, false)};
    SetDst(inst.dst[0], result);
    const IR::U1 carry_out{ir.LogicalOr(carry_out1, carry_out2)};
    SetCarryOut(inst, carry_out);
}

/// D = A - B - carry, borrow out: B > A or carry > A - B.
static void SubtractWithBorrow(IR::IREmitter& ir, const IR::U32& a, const IR::U32& b,
                               const IR::U32& carry, auto&& set_dst, auto&& set_carry) {
    set_dst(IR::U32{ir.ISub(ir.ISub(a, b), carry)});
    const IR::U1 underflow{ir.IGreaterThan(b, a, false)};
    const IR::U32 difference{ir.ISub(a, b)};
    const IR::U1 borrow_underflow{ir.IGreaterThan(carry, difference, false)};
    set_carry(ir.LogicalOr(underflow, borrow_underflow));
}

void Translator::V_SUBB_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 carry{GetCarryIn(inst)};
    SubtractWithBorrow(
        ir, src0, src1, carry, [&](const IR::U32& v) { SetDst(inst.dst[0], v); },
        [&](const IR::U1& c) { SetCarryOut(inst, c); });
}

void Translator::V_SUBBREV_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 carry{GetCarryIn(inst)};
    SubtractWithBorrow(
        ir, src1, src0, carry, [&](const IR::U32& v) { SetDst(inst.dst[0], v); },
        [&](const IR::U1& c) { SetCarryOut(inst, c); });
}

void Translator::V_LDEXP_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.FPLdexp(src0, src1));
}

void Translator::V_CVT_PKNORM_U16_F32(const GcnInst& inst) {
    const IR::Value pair =
        ir.CompositeConstruct(GetSrc<IR::F32>(inst.src[0]), GetSrc<IR::F32>(inst.src[1]));
    SetDst(inst.dst[0], ir.Pack2x16(::Gcn::NumberFormat::Unorm, pair));
}

void Translator::V_CVT_PKNORM_I16_F32(const GcnInst& inst) {
    const IR::Value pair =
        ir.CompositeConstruct(GetSrc<IR::F32>(inst.src[0]), GetSrc<IR::F32>(inst.src[1]));
    SetDst(inst.dst[0], ir.Pack2x16(::Gcn::NumberFormat::Snorm, pair));
}

void Translator::V_CVT_PKRTZ_F16_F32(const GcnInst& inst) {
    const IR::Value pair =
        ir.CompositeConstruct(GetSrc<IR::F32>(inst.src[0]), GetSrc<IR::F32>(inst.src[1]));
    SetDst(inst.dst[0], ir.Pack2x16(::Gcn::NumberFormat::Float, pair));
}

// VOP1

void Translator::V_MOV(const GcnInst& inst) {
    SetDst(inst.dst[0], GetSrc<IR::F32>(inst.src[0]));
}

/// F32 to an F16 in the low half: with host F16 a conversion, otherwise a half-float pack.
void Translator::V_CVT_F16_F32(const GcnInst& inst) {
    const IR::F32 src0 = GetSrc<IR::F32>(inst.src[0]);
    IR::U32 half;
    if (profile.support_float16) {
        const IR::F16 converted = ir.FPConvert(16, src0);
        half = ir.UConvert(32, ir.BitCast<IR::U16>(converted));
    } else {
        const IR::U32 packed =
            ir.Pack2x16(::Gcn::NumberFormat::Float, ir.CompositeConstruct(src0, ir.Imm32(0.f)));
        half = ir.BitFieldExtract(packed, ir.Imm32(0U), ir.Imm32(16U));
    }
    SetDst(inst.dst[0], half);
}

void Translator::V_CVT_F32_F16(const GcnInst& inst) {
    const IR::U32 src0 = GetSrc(inst.src[0]);
    IR::F32 value;
    if (profile.support_float16) {
        const IR::U16 low = ir.UConvert(16, src0);
        value = ir.FPConvert(32, ir.BitCast<IR::F16>(low));
    } else {
        const IR::Value unpacked = ir.Unpack2x16(::Gcn::NumberFormat::Float, src0);
        value = IR::F32{ir.CompositeExtract(unpacked, 0)};
    }
    SetDst(inst.dst[0], value);
}

/// A 4-bit signed offset (two's complement, in sixteenths): only constants seen.
void Translator::V_CVT_OFF_F32_I4(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    ASSERT(src0.IsImmediate());
    static constexpr std::array Offsets = {
        0.0f,     0.0625f,  0.1250f,  0.1875f,  0.2500f,  0.3125f,  0.3750f,  0.4375f,
        -0.5000f, -0.4375f, -0.3750f, -0.3125f, -0.2500f, -0.1875f, -0.1250f, -0.0625f};
    SetDst(inst.dst[0], ir.Imm32(Offsets[src0.U32() & 0xF]));
}

void Translator::V_CVT_F32_UBYTE(u32 index, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 byte = ir.BitFieldExtract(src0, ir.Imm32(8 * index), ir.Imm32(8));
    SetDst(inst.dst[0], ir.ConvertUToF(32, 32, byte));
}

/// Legacy reciprocal: infinities become signed zero.
void Translator::V_RCP_LEGACY_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const auto result = ir.FPRecip(src0);
    const auto is_inf = ir.FPIsInf(result);
    const auto raw = ir.ConvertFToU(32, result);
    const auto sign = ir.ShiftRightLogical(raw, ir.Imm32(31u));
    const auto negative = ir.INotEqual(sign, ir.Imm32(0u));
    const IR::F32 zero{ir.Select(negative, ir.Imm32(-0.0f), ir.Imm32(0.0f))};
    SetDst(inst.dst[0], IR::F32{ir.Select(is_inf, zero, result)});
}

/// The first set bit from the MSB (FindUMsb counts from the LSB); -1 for 0.
void Translator::V_FFBH_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 msb = ir.FindUMsb(src0);
    const IR::U32 from_left = ir.ISub(ir.Imm32(31), msb);
    const IR::U1 nonzero = ir.INotEqual(src0, ir.Imm32(0));
    SetDst(inst.dst[0], IR::U32{ir.Select(nonzero, from_left, ir.Imm32(~0U))});
}

/// The first bit differing from the sign from the MSB; -1 for 0 and -1.
void Translator::V_FFBH_I32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 msb = ir.FindSMsb(src0);
    const IR::U32 from_left = ir.ISub(ir.Imm32(31), msb);
    const IR::U32 minus_one = ir.Imm32(~0U);
    const IR::U1 valid =
        ir.LogicalAnd(ir.INotEqual(src0, ir.Imm32(0)), ir.INotEqual(src0, minus_one));
    SetDst(inst.dst[0], IR::U32{ir.Select(valid, from_left, minus_one)});
}

// V_MOVREL*: VGPRs indexed by M0, as a select over every candidate VGPR (reads past the
// allocated VGPRs give V0).

IR::U32 Translator::VMovRelSHelper(u32 src_vgprno, const IR::U32 m0) {
    IR::U32 value = ir.GetVectorReg<IR::U32>(IR::VectorReg::V0);
    for (u32 i = src_vgprno; i < runtime_info.props.num_allocated_vgprs; i++) {
        const IR::U1 selected = ir.IEqual(m0, ir.Imm32(i - src_vgprno));
        value = IR::U32{ir.Select(selected, ir.GetVectorReg<IR::U32>(IR::VectorReg::V0 + i), value)};
    }
    return value;
}

void Translator::VMovRelDHelper(u32 dst_vgprno, const IR::U32 src_val, const IR::U32 m0) {
    for (u32 i = dst_vgprno; i < runtime_info.props.num_allocated_vgprs; i++) {
        const IR::U1 selected = ir.IEqual(m0, ir.Imm32(i - dst_vgprno));
        const IR::U32 value =
            IR::U32{ir.Select(selected, src_val, ir.GetVectorReg<IR::U32>(IR::VectorReg::V0 + i))};
        ir.SetVectorReg(IR::VectorReg::V0 + i, value);
    }
}

void Translator::V_MOVRELD_B32(const GcnInst& inst) {
    const IR::U32 value{GetSrc(inst.src[0])};
    const u32 dst_vgpr = inst.dst[0].code - static_cast<u32>(IR::VectorReg::V0);
    const IR::U32 m0 = ir.GetM0();
    VMovRelDHelper(dst_vgpr, value, m0);
}

void Translator::V_MOVRELS_B32(const GcnInst& inst) {
    const u32 src_vgpr = inst.src[0].code - static_cast<u32>(IR::VectorReg::V0);
    const IR::U32 m0 = ir.GetM0();
    SetDst(inst.dst[0], VMovRelSHelper(src_vgpr, m0));
}

void Translator::V_MOVRELSD_B32(const GcnInst& inst) {
    const u32 src_vgpr = inst.src[0].code - static_cast<u32>(IR::VectorReg::V0);
    const u32 dst_vgpr = inst.dst[0].code - static_cast<u32>(IR::VectorReg::V0);
    const IR::U32 m0 = ir.GetM0();
    const IR::U32 value = VMovRelSHelper(src_vgpr, m0);
    VMovRelDHelper(dst_vgpr, value, m0);
}

// VOPC

/// The class test with a constant mask resolved here (NaN, infinity, negative); a dynamic mask
/// is resolved once the type is known.
void Translator::V_CMP_CLASS_F32(const GcnInst& inst) {
    using IR::FloatClassFunc;
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    IR::U1 value;
    if (!src1.IsImmediate()) {
        value = ir.FPCmpClass32(src0, src1);
    } else {
        const auto mask = static_cast<FloatClassFunc>(src1.U32());
        if ((mask & FloatClassFunc::NaN) == FloatClassFunc::NaN) {
            value = ir.FPIsNan(src0);
        } else if ((mask & FloatClassFunc::Infinity) == FloatClassFunc::Infinity) {
            value = ir.FPIsInf(src0);
        } else if ((mask & FloatClassFunc::Negative) == FloatClassFunc::Negative) {
            value = ir.FPLessThanEqual(src0, ir.Imm32(-0.f));
        } else {
            UNREACHABLE_MSG("Unsupported float class mask: {:#x}", static_cast<u32>(mask));
        }
    }
    SetDst64(inst.dst[1], ir.Ballot(value));
}

// VOP3a

void Translator::V_MAD_I32_I24(const GcnInst& inst, bool is_signed) {
    const IR::U32 src0{
        ir.BitFieldExtract(GetSrc(inst.src[0]), ir.Imm32(0), ir.Imm32(24), is_signed)};
    const IR::U32 src1{
        ir.BitFieldExtract(GetSrc(inst.src[1]), ir.Imm32(0), ir.Imm32(24), is_signed)};
    const IR::U32 src2{GetSrc(inst.src[2])};
    SetDst(inst.dst[0], ir.IAdd(ir.IMul(src0, src1), src2));
}

/// The result for the cube face of (x, y, z): the major axis is z, then y, then x.
IR::F32 Translator::SelectCubeResult(const IR::F32& x, const IR::F32& y, const IR::F32& z,
                                     const IR::F32& x_res, const IR::F32& y_res,
                                     const IR::F32& z_res) {
    const auto abs_x = ir.FPAbs(x);
    const auto abs_y = ir.FPAbs(y);
    const auto abs_z = ir.FPAbs(z);
    const auto z_major{
        ir.LogicalAnd(ir.FPGreaterThanEqual(abs_z, abs_x), ir.FPGreaterThanEqual(abs_z, abs_y))};
    const auto y_major{ir.FPGreaterThanEqual(abs_y, abs_x)};
    return IR::F32{ir.Select(z_major, z_res, ir.Select(y_major, y_res, x_res))};
}

/// The face index: +x 0, -x 1, +y 2, -y 3, +z 4, -z 5.
void Translator::V_CUBEID_F32(const GcnInst& inst) {
    const auto x = GetSrc<IR::F32>(inst.src[0]);
    const auto y = GetSrc<IR::F32>(inst.src[1]);
    const auto z = GetSrc<IR::F32>(inst.src[2]);
    IR::F32 result;
    if (profile.supports_native_cube_calc) {
        result = ir.CubeFaceIndex(ir.CompositeConstruct(x, y, z));
    } else {
        const auto x_neg{ir.FPLessThan(x, ir.Imm32(0.f))};
        const auto y_neg{ir.FPLessThan(y, ir.Imm32(0.f))};
        const auto z_neg{ir.FPLessThan(z, ir.Imm32(0.f))};
        const IR::F32 x_face{ir.Select(x_neg, ir.Imm32(1.f), ir.Imm32(0.f))};
        const IR::F32 y_face{ir.Select(y_neg, ir.Imm32(3.f), ir.Imm32(2.f))};
        const IR::F32 z_face{ir.Select(z_neg, ir.Imm32(5.f), ir.Imm32(4.f))};
        result = SelectCubeResult(x, y, z, x_face, y_face, z_face);
    }
    SetDst(inst.dst[0], result);
}

/// The face's s coordinate (before the division by the major axis).
void Translator::V_CUBESC_F32(const GcnInst& inst) {
    const auto x = GetSrc<IR::F32>(inst.src[0]);
    const auto y = GetSrc<IR::F32>(inst.src[1]);
    const auto z = GetSrc<IR::F32>(inst.src[2]);
    const auto x_neg{ir.FPLessThan(x, ir.Imm32(0.f))};
    const auto z_neg{ir.FPLessThan(z, ir.Imm32(0.f))};
    const IR::F32 x_sc{ir.Select(x_neg, z, ir.FPNeg(z))};
    const IR::F32 y_sc{x};
    const IR::F32 z_sc{ir.Select(z_neg, ir.FPNeg(x), x)};
    SetDst(inst.dst[0], SelectCubeResult(x, y, z, x_sc, y_sc, z_sc));
}

/// The face's t coordinate.
void Translator::V_CUBETC_F32(const GcnInst& inst) {
    const auto x = GetSrc<IR::F32>(inst.src[0]);
    const auto y = GetSrc<IR::F32>(inst.src[1]);
    const auto z = GetSrc<IR::F32>(inst.src[2]);
    const auto y_neg{ir.FPLessThan(y, ir.Imm32(0.f))};
    const IR::F32 x_z_tc{ir.FPNeg(y)};
    const IR::F32 y_tc{ir.Select(y_neg, ir.FPNeg(z), z)};
    SetDst(inst.dst[0], SelectCubeResult(x, y, z, x_z_tc, y_tc, x_z_tc));
}

/// Twice the major axis.
void Translator::V_CUBEMA_F32(const GcnInst& inst) {
    const auto x = GetSrc<IR::F32>(inst.src[0]);
    const auto y = GetSrc<IR::F32>(inst.src[1]);
    const auto z = GetSrc<IR::F32>(inst.src[2]);
    const auto two{ir.Imm32(2.f)};
    const IR::F32 x_major{ir.FPMul(x, two)};
    const IR::F32 y_major{ir.FPMul(y, two)};
    const IR::F32 z_major{ir.FPMul(z, two)};
    SetDst(inst.dst[0], SelectCubeResult(x, y, z, x_major, y_major, z_major));
}

/// Offset and width use their low 5 bits.
void Translator::V_BFE_U32(bool is_signed, const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    IR::U32 offset{GetSrc(inst.src[1])};
    IR::U32 count{GetSrc(inst.src[2])};
    if (!offset.IsImmediate()) {
        offset = ir.BitwiseAnd(offset, ir.Imm32(0x1F));
    }
    if (!count.IsImmediate()) {
        count = ir.BitwiseAnd(count, ir.Imm32(0x1F));
    }
    SetDst(inst.dst[0], ir.BitFieldExtract(src0, offset, count, is_signed));
}

/// D = |S0 - S1| + S2 (unsigned).
void Translator::V_SAD_U32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{GetSrc(inst.src[2])};
    IR::U32 difference;
    if (src0.IsImmediate() && src0.U32() == 0U) {
        difference = src1;
    } else if (src1.IsImmediate() && src1.U32() == 0U) {
        difference = src0;
    } else {
        const IR::U32 max{ir.IMax(src0, src1, false)};
        const IR::U32 min{ir.IMin(src0, src1, false)};
        difference = ir.ISub(max, min);
    }
    SetDst(inst.dst[0], ir.IAdd(difference, src2));
}

void Translator::V_CVT_PK_U16_U32(const GcnInst& inst) {
    const IR::Value pair =
        ir.CompositeConstruct(ir.BitCast<IR::F32>(GetSrc<IR::U32>(inst.src[0])),
                              ir.BitCast<IR::F32>(GetSrc<IR::U32>(inst.src[1])));
    SetDst(inst.dst[0], ir.Pack2x16(::Gcn::NumberFormat::Uint, pair));
}

void Translator::V_CVT_PK_I16_I32(const GcnInst& inst) {
    const IR::Value pair =
        ir.CompositeConstruct(ir.BitCast<IR::F32>(GetSrc<IR::U32>(inst.src[0])),
                              ir.BitCast<IR::F32>(GetSrc<IR::U32>(inst.src[1])));
    SetDst(inst.dst[0], ir.Pack2x16(::Gcn::NumberFormat::Sint, pair));
}

/// S0 rounded and clamped to a byte, inserted at byte S1[1:0] of S2 (NaN gives 0).
void Translator::V_CVT_PK_U8_F32(const GcnInst& inst) {
    const IR::F32 src0{GetSrc<IR::F32>(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 src2{GetSrc(inst.src[2])};
    const IR::F32 clamped{ir.FPClamp(ir.FPRoundEven(src0), ir.Imm32(0.0f), ir.Imm32(255.0f))};
    const IR::F32 value{ir.Select(ir.FPIsNan(src0), ir.Imm32(0.0f), clamped)};
    const IR::U32 byte{ir.ConvertFToU(32, value)};
    const IR::U32 offset{ir.ShiftLeftLogical(ir.BitwiseAnd(src1, ir.Imm32(3)), ir.Imm32(3))};
    SetDst(inst.dst[0], ir.BitFieldInsert(src2, byte, offset, ir.Imm32(8)));
}

/// D = ({S0, S1} >> S2[4:0])[31:0].
void Translator::V_ALIGNBIT_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 shift{ir.BitwiseAnd(GetSrc(inst.src[2]), ir.Imm32(0x1F))};
    const IR::U32 lo{ir.ShiftRightLogical(src1, shift)};
    const IR::U32 hi{
        ir.ShiftLeftLogical(ir.ShiftLeftLogical(src0, ir.Imm32(1)), ir.ISub(ir.Imm32(31), shift))};
    SetDst(inst.dst[0], ir.BitwiseOr(lo, hi));
}

/// D = ({S0, S1} >> (8 * S2[1:0]))[31:0].
void Translator::V_ALIGNBYTE_B32(const GcnInst& inst) {
    const IR::U32 src0{GetSrc(inst.src[0])};
    const IR::U32 src1{GetSrc(inst.src[1])};
    const IR::U32 bytes{ir.BitwiseAnd(GetSrc(inst.src[2]), ir.Imm32(0x3))};
    const IR::U32 shift{ir.ShiftLeftLogical(bytes, ir.Imm32(3))};
    const IR::U32 lo{ir.ShiftRightLogical(src1, shift)};
    const IR::U32 hi{
        ir.ShiftLeftLogical(ir.ShiftLeftLogical(src0, ir.Imm32(1)), ir.ISub(ir.Imm32(31), shift))};
    SetDst(inst.dst[0], ir.BitwiseOr(lo, hi));
}

/// D = S0 * S1 + S2 in 64 bits; the carry out to VCC.
void Translator::V_MAD_U64_U32(const GcnInst& inst) {
    const auto src0 = GetSrc<IR::U32>(inst.src[0]);
    const auto src1 = GetSrc<IR::U32>(inst.src[1]);
    const auto src2 = GetSrc64<IR::U64>(inst.src[2]);
    const IR::U64 product = ir.IMul(ir.UConvert(64, src0), ir.UConvert(64, src1));
    const IR::U64 sum = ir.IAdd(product, src2);
    SetDst64(inst.dst[0], sum);
    const IR::U1 below_product = ir.ILessThan(sum, product, false);
    const IR::U1 below_addend = ir.ILessThan(sum, src2, false);
    const IR::U1 overflow = ir.LogicalOr(below_product, below_addend);
    const auto halves = ir.UnpackUint2x32(ir.Ballot(overflow));
    const IR::U32 lo{ir.CompositeExtract(halves, 0U)};
    const IR::U32 hi{ir.CompositeExtract(halves, 1U)};
    ir.SetVccLo(lo);
    ir.SetVccHi(hi);
}

// VOP3P mixed precision: D = S0 * S1 + S2 with F16 or F32 sources, to F32 or an F16 half.

void Translator::V_MAD_MIX_F32(const GcnInst& inst) {
    const auto src0 = GetSrcMix(inst.src[0]);
    const auto src1 = GetSrcMix(inst.src[1]);
    const auto src2 = GetSrcMix(inst.src[2]);
    SetDst(inst.dst[0], IR::F32{ir.FPAdd(ir.FPMul(src0, src1), src2)});
}

void Translator::V_MAD_MIXLO_F16(const GcnInst& inst) {
    const auto src0 = GetSrcMix(inst.src[0]);
    const auto src1 = GetSrcMix(inst.src[1]);
    const auto src2 = GetSrcMix(inst.src[2]);
    const IR::F32 result = ir.FPAdd(ir.FPMul(src0, src1), src2);
    const IR::F16 half = ir.FPConvert(16, result);
    const IR::U16 bits = ir.BitCast<IR::U16, IR::F16>(half);
    const IR::U32 old_value{GetSrc(inst.dst[0])};
    SetDst(inst.dst[0],
           IR::U32{ir.BitFieldInsert(old_value, ir.UConvert(32, bits), ir.Imm32(0U), ir.Imm32(16U))});
}

void Translator::V_MAD_MIXHI_F16(const GcnInst& inst) {
    const auto src0 = GetSrcMix(inst.src[0]);
    const auto src1 = GetSrcMix(inst.src[1]);
    const auto src2 = GetSrcMix(inst.src[2]);
    const IR::F32 result = ir.FPAdd(ir.FPMul(src0, src1), src2);
    const IR::F16 half = ir.FPConvert(16, result);
    const IR::U16 bits = ir.BitCast<IR::U16, IR::F16>(half);
    const IR::U32 old_value{GetSrc(inst.dst[0])};
    SetDst(inst.dst[0], IR::U32{ir.BitFieldInsert(old_value, ir.UConvert(32, bits), ir.Imm32(16U),
                                                  ir.Imm32(16U))});
}

} // namespace Shader::Gcn
