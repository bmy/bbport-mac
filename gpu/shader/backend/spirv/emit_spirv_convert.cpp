// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Conversions and composites (vectors): one SPIR-V instruction each, written as tables.
#include "shader/ir/microinstruction.h"
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

/// Emit<name>(value) = op(type, value).
#define UNARY_OP(name, op, type)                                                                   \
    Id Emit##name(EmitContext& ctx, Id value) {                                                    \
        return ctx.op(type, value);                                                                \
    }

// Float to integer; 16-bit results are widened back to 32 bits.
UNARY_OP(ConvertS32F32, OpConvertFToS, ctx.U32[1])
UNARY_OP(ConvertS32F64, OpConvertFToS, ctx.U32[1])
UNARY_OP(ConvertS64F32, OpConvertFToS, ctx.U64)
UNARY_OP(ConvertS64F64, OpConvertFToS, ctx.U64)
UNARY_OP(ConvertU32F32, OpConvertFToU, ctx.U32[1])
UNARY_OP(ConvertU32F64, OpConvertFToU, ctx.U32[1])
UNARY_OP(ConvertU64F32, OpConvertFToU, ctx.U64)
UNARY_OP(ConvertU64F64, OpConvertFToU, ctx.U64)

Id EmitConvertS16F32(EmitContext& ctx, Id value) {
    return ctx.OpSConvert(ctx.U32[1], ctx.OpConvertFToS(ctx.U16, value));
}

Id EmitConvertS16F64(EmitContext& ctx, Id value) {
    return ctx.OpSConvert(ctx.U32[1], ctx.OpConvertFToS(ctx.U16, value));
}

Id EmitConvertU16F32(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U32[1], ctx.OpConvertFToU(ctx.U16, value));
}

Id EmitConvertU16F64(EmitContext& ctx, Id value) {
    return ctx.OpUConvert(ctx.U32[1], ctx.OpConvertFToU(ctx.U16, value));
}

// Integer widths.
UNARY_OP(ConvertU64U32, OpUConvert, ctx.U64)
UNARY_OP(ConvertU32U64, OpUConvert, ctx.U32[1])
UNARY_OP(ConvertU16U32, OpUConvert, ctx.U16)
UNARY_OP(ConvertU32U16, OpUConvert, ctx.U32[1])
UNARY_OP(ConvertU8U32, OpUConvert, ctx.U8)
UNARY_OP(ConvertU32U8, OpUConvert, ctx.U32[1])
UNARY_OP(ConvertS32S8, OpSConvert, ctx.U32[1])
UNARY_OP(ConvertS32S16, OpSConvert, ctx.U32[1])

// Float widths.
UNARY_OP(ConvertF16F32, OpFConvert, ctx.F16[1])
UNARY_OP(ConvertF32F16, OpFConvert, ctx.F32[1])
UNARY_OP(ConvertF32F64, OpFConvert, ctx.F32[1])
UNARY_OP(ConvertF64F32, OpFConvert, ctx.F64[1])

// Integer to float; 8/16-bit sources are narrowed first.
UNARY_OP(ConvertF32S32, OpConvertSToF, ctx.F32[1])
UNARY_OP(ConvertF32S64, OpConvertSToF, ctx.F32[1])
UNARY_OP(ConvertF32U32, OpConvertUToF, ctx.F32[1])
UNARY_OP(ConvertF32U64, OpConvertUToF, ctx.F32[1])
UNARY_OP(ConvertF64S32, OpConvertSToF, ctx.F64[1])
UNARY_OP(ConvertF64S64, OpConvertSToF, ctx.F64[1])
UNARY_OP(ConvertF64U32, OpConvertUToF, ctx.F64[1])
UNARY_OP(ConvertF64U64, OpConvertUToF, ctx.F64[1])

#define NARROW_TO_FLOAT(name, op, float_type, narrow, narrow_type)                                 \
    Id Emit##name(EmitContext& ctx, Id value) {                                                    \
        return ctx.op(float_type, ctx.narrow(narrow_type, value));                                 \
    }

NARROW_TO_FLOAT(ConvertF32S8, OpConvertSToF, ctx.F32[1], OpSConvert, ctx.S8)
NARROW_TO_FLOAT(ConvertF32S16, OpConvertSToF, ctx.F32[1], OpSConvert, ctx.S16)
NARROW_TO_FLOAT(ConvertF32U8, OpConvertUToF, ctx.F32[1], OpUConvert, ctx.U8)
NARROW_TO_FLOAT(ConvertF32U16, OpConvertUToF, ctx.F32[1], OpUConvert, ctx.U16)
NARROW_TO_FLOAT(ConvertF64S8, OpConvertSToF, ctx.F64[1], OpSConvert, ctx.S8)
NARROW_TO_FLOAT(ConvertF64S16, OpConvertSToF, ctx.F64[1], OpSConvert, ctx.S16)
NARROW_TO_FLOAT(ConvertF64U8, OpConvertUToF, ctx.F64[1], OpUConvert, ctx.U8)
NARROW_TO_FLOAT(ConvertF64U16, OpConvertUToF, ctx.F64[1], OpUConvert, ctx.U16)

#undef NARROW_TO_FLOAT
#undef UNARY_OP

// Composites: constructs of immediates are constants.

namespace {

template <typename... Args>
Id Construct(EmitContext& ctx, IR::Inst* inst, Args&&... args) {
    return inst->AreAllArgsImmediates() ? ctx.ConstantComposite(args...)
                                        : ctx.OpCompositeConstruct(args...);
}

} // namespace

#define COMPOSITES(T, scalar_type)                                                                 \
    Id EmitCompositeConstruct##T##x2(EmitContext& ctx, IR::Inst* inst, Id e1, Id e2) {             \
        return Construct(ctx, inst, ctx.scalar_type[2], e1, e2);                                   \
    }                                                                                              \
    Id EmitCompositeConstruct##T##x3(EmitContext& ctx, IR::Inst* inst, Id e1, Id e2, Id e3) {      \
        return Construct(ctx, inst, ctx.scalar_type[3], e1, e2, e3);                               \
    }                                                                                              \
    Id EmitCompositeConstruct##T##x4(EmitContext& ctx, IR::Inst* inst, Id e1, Id e2, Id e3,        \
                                     Id e4) {                                                      \
        return Construct(ctx, inst, ctx.scalar_type[4], e1, e2, e3, e4);                           \
    }                                                                                              \
    /* A vec4 from two vec2s. */                                                                   \
    Id EmitCompositeConstruct##T##x2x2(EmitContext& ctx, IR::Inst* inst, Id e1, Id e2) {           \
        return Construct(ctx, inst, ctx.scalar_type[4], e1, e2);                                   \
    }                                                                                              \
    Id EmitCompositeExtract##T##x2(EmitContext& ctx, Id composite, u32 index) {                    \
        return ctx.OpCompositeExtract(ctx.scalar_type[1], composite, index);                       \
    }                                                                                              \
    Id EmitCompositeExtract##T##x3(EmitContext& ctx, Id composite, u32 index) {                    \
        return ctx.OpCompositeExtract(ctx.scalar_type[1], composite, index);                       \
    }                                                                                              \
    Id EmitCompositeExtract##T##x4(EmitContext& ctx, Id composite, u32 index) {                    \
        return ctx.OpCompositeExtract(ctx.scalar_type[1], composite, index);                       \
    }                                                                                              \
    Id EmitCompositeInsert##T##x2(EmitContext& ctx, Id composite, Id object, u32 index) {          \
        return ctx.OpCompositeInsert(ctx.scalar_type[2], object, composite, index);                \
    }                                                                                              \
    Id EmitCompositeInsert##T##x3(EmitContext& ctx, Id composite, Id object, u32 index) {          \
        return ctx.OpCompositeInsert(ctx.scalar_type[3], object, composite, index);                \
    }                                                                                              \
    Id EmitCompositeInsert##T##x4(EmitContext& ctx, Id composite, Id object, u32 index) {          \
        return ctx.OpCompositeInsert(ctx.scalar_type[4], object, composite, index);                \
    }                                                                                              \
    Id EmitCompositeShuffle##T##x2(EmitContext& ctx, Id composite1, Id composite2, u32 comp0,      \
                                   u32 comp1) {                                                    \
        return ctx.OpVectorShuffle(ctx.scalar_type[2], composite1, composite2, comp0, comp1);      \
    }                                                                                              \
    Id EmitCompositeShuffle##T##x3(EmitContext& ctx, Id composite1, Id composite2, u32 comp0,      \
                                   u32 comp1, u32 comp2) {                                         \
        return ctx.OpVectorShuffle(ctx.scalar_type[3], composite1, composite2, comp0, comp1,       \
                                   comp2);                                                         \
    }                                                                                              \
    Id EmitCompositeShuffle##T##x4(EmitContext& ctx, Id composite1, Id composite2, u32 comp0,      \
                                   u32 comp1, u32 comp2, u32 comp3) {                              \
        return ctx.OpVectorShuffle(ctx.scalar_type[4], composite1, composite2, comp0, comp1,       \
                                   comp2, comp3);                                                  \
    }

COMPOSITES(U32, U32)
COMPOSITES(F32, F32)

#undef COMPOSITES

} // namespace Shader::Backend::SPIRV
