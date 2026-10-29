// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Integer arithmetic, bit operations, min/max/clamp and compares. 64-bit bit counts and scans
// are done on the 32-bit halves (Vulkan allows them on 32 bits only).
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

#define UNARY(name, op, type)                                                                      \
    Id Emit##name(EmitContext& ctx, Id value) {                                                    \
        return ctx.op(type, value);                                                                \
    }
#define BINARY(name, op, type)                                                                     \
    Id Emit##name(EmitContext& ctx, Id a, Id b) {                                                  \
        return ctx.op(type, a, b);                                                                 \
    }
/// Instructions the translator gives their IR::Inst (unused here).
#define BINARY_INST(name, op, type)                                                                \
    Id Emit##name(EmitContext& ctx, IR::Inst*, Id a, Id b) {                                       \
        return ctx.op(type, a, b);                                                                 \
    }

BINARY_INST(IAdd32, OpIAdd, ctx.U32[1])
BINARY(IAdd64, OpIAdd, ctx.U64)
BINARY(IAddCarry32, OpIAddCarry, ctx.full_result_u32x2)
BINARY(ISub32, OpISub, ctx.U32[1])
BINARY(ISub64, OpISub, ctx.U64)
BINARY(IMul32, OpIMul, ctx.U32[1])
BINARY(IMul64, OpIMul, ctx.U64)
BINARY(SDiv32, OpSDiv, ctx.U32[1])
BINARY(UDiv32, OpUDiv, ctx.U32[1])
BINARY(SMod32, OpSMod, ctx.U32[1])
BINARY(UMod32, OpUMod, ctx.U32[1])
UNARY(INeg32, OpSNegate, ctx.U32[1])
UNARY(INeg64, OpSNegate, ctx.U64)
UNARY(IAbs32, OpSAbs, ctx.U32[1])
BINARY(ShiftLeftLogical32, OpShiftLeftLogical, ctx.U32[1])
BINARY(ShiftLeftLogical64, OpShiftLeftLogical, ctx.U64)
BINARY(ShiftRightLogical32, OpShiftRightLogical, ctx.U32[1])
BINARY(ShiftRightLogical64, OpShiftRightLogical, ctx.U64)
BINARY(ShiftRightArithmetic32, OpShiftRightArithmetic, ctx.U32[1])
BINARY(ShiftRightArithmetic64, OpShiftRightArithmetic, ctx.U64)
BINARY_INST(BitwiseAnd32, OpBitwiseAnd, ctx.U32[1])
BINARY_INST(BitwiseAnd64, OpBitwiseAnd, ctx.U64)
BINARY_INST(BitwiseOr32, OpBitwiseOr, ctx.U32[1])
BINARY_INST(BitwiseOr64, OpBitwiseOr, ctx.U64)
BINARY_INST(BitwiseXor32, OpBitwiseXor, ctx.U32[1])
BINARY_INST(BitwiseXor64, OpBitwiseXor, ctx.U64)
UNARY(BitwiseNot32, OpNot, ctx.U32[1])
UNARY(BitwiseNot64, OpNot, ctx.U64)
UNARY(BitReverse32, OpBitReverse, ctx.U32[1])
UNARY(BitCount32, OpBitCount, ctx.U32[1])
UNARY(FindSMsb32, OpFindSMsb, ctx.U32[1])
UNARY(FindUMsb32, OpFindUMsb, ctx.U32[1])
UNARY(FindILsb32, OpFindILsb, ctx.U32[1])
BINARY(SMin32, OpSMin, ctx.U32[1])
BINARY(UMin32, OpUMin, ctx.U32[1])
BINARY(SMax32, OpSMax, ctx.U32[1])
BINARY(UMax32, OpUMax, ctx.U32[1])

Id EmitSMulHi(EmitContext& ctx, Id a, Id b) {
    const auto signed_a{ctx.OpBitcast(ctx.S32[1], a)};
    const auto signed_b{ctx.OpBitcast(ctx.S32[1], b)};
    const auto product{ctx.OpSMulExtended(ctx.full_result_i32x2, signed_a, signed_b)};
    const auto signed_hi{ctx.OpCompositeExtract(ctx.S32[1], product, 1)};
    return ctx.OpBitcast(ctx.U32[1], signed_hi);
}

Id EmitUMulHi(EmitContext& ctx, Id a, Id b) {
    const auto product{ctx.OpUMulExtended(ctx.full_result_u32x2, a, b)};
    return ctx.OpCompositeExtract(ctx.U32[1], product, 1);
}

Id EmitBitFieldInsert32(EmitContext& ctx, Id base, Id insert, Id offset, Id count) {
    return ctx.OpBitFieldInsert(ctx.U32[1], base, insert, offset, count);
}

Id EmitBitFieldInsert64(EmitContext& ctx, Id base, Id insert, Id offset, Id count) {
    return ctx.OpBitFieldInsert(ctx.U64, base, insert, offset, count);
}

Id EmitBitFieldSExtract(EmitContext& ctx, IR::Inst*, Id base, Id offset, Id count) {
    return ctx.OpBitFieldSExtract(ctx.U32[1], base, offset, count);
}

Id EmitBitFieldUExtract(EmitContext& ctx, IR::Inst*, Id base, Id offset, Id count) {
    return ctx.OpBitFieldUExtract(ctx.U32[1], base, offset, count);
}

Id EmitBitCount64(EmitContext& ctx, Id value) {
    const Id halves{ctx.OpBitcast(ctx.U32[2], value)};
    const Id lo{ctx.OpCompositeExtract(ctx.U32[1], halves, 0U)};
    const Id hi{ctx.OpCompositeExtract(ctx.U32[1], halves, 1U)};
    const Id lo_count{ctx.OpBitCount(ctx.U32[1], lo)};
    const Id hi_count{ctx.OpBitCount(ctx.U32[1], hi)};
    return ctx.OpIAdd(ctx.U32[1], lo_count, hi_count);
}

/// The high half's MSB + 32 if it has one, else the low half's (0 is handled by the IR).
Id EmitFindUMsb64(EmitContext& ctx, Id value) {
    const Id halves{ctx.OpBitcast(ctx.U32[2], value)};
    const Id hi{ctx.OpCompositeExtract(ctx.U32[1], halves, 1U)};
    const Id lo{ctx.OpCompositeExtract(ctx.U32[1], halves, 0U)};
    const Id hi_msb{ctx.OpFindUMsb(ctx.U32[1], hi)};
    const Id lo_msb{ctx.OpFindUMsb(ctx.U32[1], lo)};
    const Id found_hi{ctx.OpINotEqual(ctx.U1[1], hi_msb, ctx.ConstU32(u32(-1)))};
    const Id shifted_hi{ctx.OpIAdd(ctx.U32[1], hi_msb, ctx.ConstU32(32u))};
    return ctx.OpSelect(ctx.U32[1], found_hi, shifted_hi, lo_msb);
}

/// The low half's LSB if it has one, else the high half's.
Id EmitFindILsb64(EmitContext& ctx, Id value) {
    const Id halves{ctx.OpBitcast(ctx.U32[2], value)};
    const Id lo{ctx.OpCompositeExtract(ctx.U32[1], halves, 0U)};
    const Id hi{ctx.OpCompositeExtract(ctx.U32[1], halves, 1U)};
    const Id lo_lsb{ctx.OpFindILsb(ctx.U32[1], lo)};
    const Id hi_lsb{ctx.OpFindILsb(ctx.U32[1], hi)};
    const Id found_lo{ctx.OpINotEqual(ctx.U1[1], lo_lsb, ctx.ConstU32(u32(-1)))};
    return ctx.OpSelect(ctx.U32[1], found_lo, lo_lsb, hi_lsb);
}

// Three-operand min/max/median: AMD's instructions where available.

#define TRI_MINMAX(name, amd_op, op)                                                               \
    Id Emit##name(EmitContext& ctx, Id a, Id b, Id c) {                                            \
        if (ctx.profile.supports_trinary_minmax) {                                                 \
            return ctx.amd_op(ctx.U32[1], a, b, c);                                                \
        }                                                                                          \
        return ctx.op(ctx.U32[1], a, ctx.op(ctx.U32[1], b, c));                                    \
    }

TRI_MINMAX(SMinTri32, OpSMin3AMD, OpSMin)
TRI_MINMAX(UMinTri32, OpUMin3AMD, OpUMin)
TRI_MINMAX(SMaxTri32, OpSMax3AMD, OpSMax)
TRI_MINMAX(UMaxTri32, OpUMax3AMD, OpUMax)

#undef TRI_MINMAX

Id EmitSMedTri32(EmitContext& ctx, Id a, Id b, Id c) {
    if (ctx.profile.supports_trinary_minmax) {
        return ctx.OpSMid3AMD(ctx.U32[1], a, b, c);
    }
    const Id upper{ctx.OpSMin(ctx.U32[1], ctx.OpSMax(ctx.U32[1], a, b), c)};
    return ctx.OpSMax(ctx.U32[1], ctx.OpSMin(ctx.U32[1], a, b), upper);
}

Id EmitUMedTri32(EmitContext& ctx, Id a, Id b, Id c) {
    if (ctx.profile.supports_trinary_minmax) {
        return ctx.OpUMid3AMD(ctx.U32[1], a, b, c);
    }
    const Id upper{ctx.OpUMin(ctx.U32[1], ctx.OpUMax(ctx.U32[1], a, b), c)};
    return ctx.OpUMax(ctx.U32[1], ctx.OpUMin(ctx.U32[1], a, b), upper);
}

/// Clamps; drivers with broken clamp instructions get min/max (signed on signed types).
Id EmitSClamp32(EmitContext& ctx, IR::Inst*, Id value, Id min, Id max) {
    if (!ctx.profile.has_broken_spirv_clamp) {
        return ctx.OpSClamp(ctx.U32[1], value, min, max);
    }
    value = ctx.OpBitcast(ctx.S32[1], value);
    min = ctx.OpBitcast(ctx.S32[1], min);
    max = ctx.OpBitcast(ctx.S32[1], max);
    const Id result = ctx.OpSMax(ctx.S32[1], ctx.OpSMin(ctx.S32[1], value, max), min);
    return ctx.OpBitcast(ctx.U32[1], result);
}

Id EmitUClamp32(EmitContext& ctx, IR::Inst*, Id value, Id min, Id max) {
    if (!ctx.profile.has_broken_spirv_clamp) {
        return ctx.OpUClamp(ctx.U32[1], value, min, max);
    }
    return ctx.OpUMax(ctx.U32[1], ctx.OpUMin(ctx.U32[1], value, max), min);
}

#define COMPARE(name, op)                                                                          \
    BINARY(name##32, op, ctx.U1[1])                                                                \
    BINARY(name##64, op, ctx.U1[1])

COMPARE(SLessThan, OpSLessThan)
COMPARE(ULessThan, OpULessThan)
COMPARE(IEqual, OpIEqual)
COMPARE(SLessThanEqual, OpSLessThanEqual)
COMPARE(ULessThanEqual, OpULessThanEqual)
COMPARE(SGreaterThan, OpSGreaterThan)
COMPARE(UGreaterThan, OpUGreaterThan)
COMPARE(INotEqual, OpINotEqual)
COMPARE(SGreaterThanEqual, OpSGreaterThanEqual)
COMPARE(UGreaterThanEqual, OpUGreaterThanEqual)

/// (Lowered by the hardware intrinsics pass.)
Id EmitMaskedBitCount32(EmitContext& ctx, Id value, Id addend, bool hi) {
    UNREACHABLE_MSG("Unreachable instruction");
}

#undef COMPARE
#undef BINARY_INST
#undef BINARY
#undef UNARY

} // namespace Shader::Backend::SPIRV
