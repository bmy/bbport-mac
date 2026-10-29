// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Float arithmetic, transcendentals and compares. Adds, multiplies, divides and FMAs are
// NoContraction: the host compiler must not fuse what GCN computed separately.
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {

Id Exact(EmitContext& ctx, Id op) {
    ctx.Decorate(op, spv::Decoration::NoContraction);
    return op;
}

} // namespace

#define UNARY(name, op, type)                                                                      \
    Id Emit##name(EmitContext& ctx, Id value) {                                                    \
        return ctx.op(type, value);                                                                \
    }
#define BINARY(name, op, type)                                                                     \
    Id Emit##name(EmitContext& ctx, Id a, Id b) {                                                  \
        return ctx.op(type, a, b);                                                                 \
    }
#define BINARY_EXACT(name, op, type)                                                               \
    Id Emit##name(EmitContext& ctx, IR::Inst*, Id a, Id b) {                                       \
        return Exact(ctx, ctx.op(type, a, b));                                                     \
    }
/// Float types in pairs: 32 and 64 bits.
#define UNARY_32_64(name, op)                                                                      \
    UNARY(name##32, op, ctx.F32[1])                                                                \
    UNARY(name##64, op, ctx.F64[1])

UNARY_32_64(FPAbs, OpFAbs)
UNARY_32_64(FPNeg, OpFNegate)
UNARY_32_64(FPRecipSqrt, OpInverseSqrt)
UNARY_32_64(FPRoundEven, OpRoundEven)
UNARY_32_64(FPFloor, OpFloor)
UNARY_32_64(FPCeil, OpCeil)
UNARY_32_64(FPTrunc, OpTrunc)
UNARY_32_64(FPFract, OpFract)
UNARY(FPExp2, OpExp2, ctx.F32[1])
UNARY(FPLog2, OpLog2, ctx.F32[1])
UNARY(FPSqrt, OpSqrt, ctx.F32[1])

BINARY_EXACT(FPAdd32, OpFAdd, ctx.F32[1])
BINARY_EXACT(FPAdd64, OpFAdd, ctx.F64[1])
BINARY_EXACT(FPSub32, OpFSub, ctx.F32[1])
BINARY_EXACT(FPMul32, OpFMul, ctx.F32[1])
BINARY_EXACT(FPMul64, OpFMul, ctx.F64[1])
BINARY_EXACT(FPDiv32, OpFDiv, ctx.F32[1])
BINARY_EXACT(FPDiv64, OpFDiv, ctx.F64[1])
BINARY(FPMax32, OpFMax, ctx.F32[1])
BINARY(FPMax64, OpFMax, ctx.F64[1])
BINARY(FPMin32, OpFMin, ctx.F32[1])
BINARY(FPMin64, OpFMin, ctx.F64[1])
BINARY(FPPow, OpPow, ctx.F32[1])
BINARY(FPLdexp, OpLdexp, ctx.F32[1])

Id EmitFPFma32(EmitContext& ctx, IR::Inst*, Id a, Id b, Id c) {
    return Exact(ctx, ctx.OpFma(ctx.F32[1], a, b, c));
}

Id EmitFPFma64(EmitContext& ctx, IR::Inst*, Id a, Id b, Id c) {
    return Exact(ctx, ctx.OpFma(ctx.F64[1], a, b, c));
}

// Three-operand min/max/median: AMD's instructions where available.

Id EmitFPMinTri32(EmitContext& ctx, Id a, Id b, Id c) {
    if (ctx.profile.supports_trinary_minmax) {
        return ctx.OpFMin3AMD(ctx.F32[1], a, b, c);
    }
    return ctx.OpFMin(ctx.F32[1], a, ctx.OpFMin(ctx.F32[1], b, c));
}

Id EmitFPMaxTri32(EmitContext& ctx, Id a, Id b, Id c) {
    if (ctx.profile.supports_trinary_minmax) {
        return ctx.OpFMax3AMD(ctx.F32[1], a, b, c);
    }
    return ctx.OpFMax(ctx.F32[1], a, ctx.OpFMax(ctx.F32[1], b, c));
}

/// med3(a, b, c) = max(min(a, b), min(max(a, b), c)).
Id EmitFPMedTri32(EmitContext& ctx, Id a, Id b, Id c) {
    if (ctx.profile.supports_trinary_minmax) {
        return ctx.OpFMid3AMD(ctx.F32[1], a, b, c);
    }
    const Id upper{ctx.OpFMin(ctx.F32[1], ctx.OpFMax(ctx.F32[1], a, b), c)};
    return ctx.OpFMax(ctx.F32[1], ctx.OpFMin(ctx.F32[1], a, b), upper);
}

/// GCN sin/cos take revolutions.
Id EmitFPSin(EmitContext& ctx, Id value) {
    return ctx.OpSin(ctx.F32[1], ctx.OpFMul(ctx.F32[1], ctx.pi_x2, value));
}

Id EmitFPCos(EmitContext& ctx, Id value) {
    return ctx.OpCos(ctx.F32[1], ctx.OpFMul(ctx.F32[1], ctx.pi_x2, value));
}

Id EmitFPRecip32(EmitContext& ctx, Id value) {
    return ctx.OpFDiv(ctx.F32[1], ctx.ConstF32(1.0f), value);
}

Id EmitFPRecip64(EmitContext& ctx, Id value) {
    return ctx.OpFDiv(ctx.F64[1], ctx.Constant(ctx.F64[1], f64{1.0}), value);
}

Id EmitFPSaturate32(EmitContext& ctx, Id value) {
    const Id zero{ctx.ConstF32(f32{0.0})};
    const Id one{ctx.ConstF32(f32{1.0})};
    return ctx.OpFClamp(ctx.F32[1], value, zero, one);
}

Id EmitFPSaturate64(EmitContext& ctx, Id value) {
    const Id zero{ctx.Constant(ctx.F64[1], f64{0.0})};
    const Id one{ctx.Constant(ctx.F64[1], f64{1.0})};
    return ctx.OpFClamp(ctx.F64[1], value, zero, one);
}

Id EmitFPClamp32(EmitContext& ctx, Id value, Id min_value, Id max_value) {
    return ctx.OpFClamp(ctx.F32[1], value, min_value, max_value);
}

Id EmitFPClamp64(EmitContext& ctx, Id value, Id min_value, Id max_value) {
    return ctx.OpFClamp(ctx.F64[1], value, min_value, max_value);
}

// frexp: the significand, or the exponent as a 32-bit integer.

Id EmitFPFrexpSig32(EmitContext& ctx, Id value) {
    const auto frexp = ctx.OpFrexpStruct(ctx.frexp_result_f32, value);
    return ctx.OpCompositeExtract(ctx.F32[1], frexp, 0);
}

Id EmitFPFrexpSig64(EmitContext& ctx, Id value) {
    const auto frexp = ctx.OpFrexpStruct(ctx.frexp_result_f64, value);
    return ctx.OpCompositeExtract(ctx.F64[1], frexp, 0);
}

Id EmitFPFrexpExp32(EmitContext& ctx, Id value) {
    const auto frexp = ctx.OpFrexpStruct(ctx.frexp_result_f32, value);
    return ctx.OpBitcast(ctx.U32[1], ctx.OpCompositeExtract(ctx.S32[1], frexp, 1));
}

Id EmitFPFrexpExp64(EmitContext& ctx, Id value) {
    const auto frexp = ctx.OpFrexpStruct(ctx.frexp_result_f64, value);
    return ctx.OpBitcast(ctx.U32[1], ctx.OpCompositeExtract(ctx.S32[1], frexp, 1));
}

// Compares (16, 32 and 64 bits alike).

#define COMPARE(name, op)                                                                          \
    BINARY(name##16, op, ctx.U1[1])                                                                \
    BINARY(name##32, op, ctx.U1[1])                                                                \
    BINARY(name##64, op, ctx.U1[1])

COMPARE(FPOrdEqual, OpFOrdEqual)
COMPARE(FPUnordEqual, OpFUnordEqual)
COMPARE(FPOrdNotEqual, OpFOrdNotEqual)
COMPARE(FPUnordNotEqual, OpFUnordNotEqual)
COMPARE(FPOrdLessThan, OpFOrdLessThan)
COMPARE(FPUnordLessThan, OpFUnordLessThan)
COMPARE(FPOrdGreaterThan, OpFOrdGreaterThan)
COMPARE(FPUnordGreaterThan, OpFUnordGreaterThan)
COMPARE(FPOrdLessThanEqual, OpFOrdLessThanEqual)
COMPARE(FPUnordLessThanEqual, OpFUnordLessThanEqual)
COMPARE(FPOrdGreaterThanEqual, OpFOrdGreaterThanEqual)
COMPARE(FPUnordGreaterThanEqual, OpFUnordGreaterThanEqual)
UNARY(FPIsNan16, OpIsNan, ctx.U1[1])
UNARY(FPIsNan32, OpIsNan, ctx.U1[1])
UNARY(FPIsNan64, OpIsNan, ctx.U1[1])
UNARY(FPIsInf32, OpIsInf, ctx.U1[1])
UNARY(FPIsInf64, OpIsInf, ctx.U1[1])

/// (Resolved by the hardware intrinsics lowering.)
void EmitFPCmpClass32(EmitContext&) {
    UNREACHABLE();
}

#undef COMPARE
#undef UNARY_32_64
#undef BINARY_EXACT
#undef BINARY
#undef UNARY

} // namespace Shader::Backend::SPIRV
