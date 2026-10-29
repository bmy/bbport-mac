// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bit casts and the packed formats: GLSL pack/unpack instructions where SPIR-V has them,
// otherwise bit field extracts and inserts (integer formats, 10:11:11 floats through helper
// functions of the context, 2:10:10:10).
#include <array>
#include <span>

#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {

struct Field {
    u32 offset;
    u32 size;
};

constexpr std::array<Field, 2> Fields16x2{{{0, 16}, {16, 16}}};
constexpr std::array<Field, 4> Fields8x4{{{0, 8}, {8, 8}, {16, 8}, {24, 8}}};
constexpr std::array<Field, 3> Fields10_11_11{{{0, 11}, {11, 11}, {22, 10}}};
constexpr std::array<Field, 4> Fields2_10_10_10{{{0, 10}, {10, 10}, {20, 10}, {30, 2}}};

/// The fields of a 32-bit value (sign-extended if signed).
template <size_t N>
std::array<Id, N> ExtractFields(EmitContext& ctx, Id value, const std::array<Field, N>& fields,
                                bool is_signed) {
    const auto op = is_signed ? &EmitContext::OpBitFieldSExtract : &EmitContext::OpBitFieldUExtract;
    std::array<Id, N> result{};
    for (size_t i = 0; i < N; i++) {
        result[i] = (ctx.*op)(ctx.U32[1], value, ctx.ConstU32(fields[i].offset),
                              ctx.ConstU32(fields[i].size));
    }
    return result;
}

/// The values packed into the fields of one 32-bit value (the first unmasked: it is placed
/// at offset 0 and the others are inserted over it).
template <size_t N>
Id InsertFields(EmitContext& ctx, const std::array<Id, N>& values,
                const std::array<Field, N>& fields) {
    Id result = values[0];
    for (size_t i = 1; i < N; i++) {
        result = ctx.OpBitFieldInsert(ctx.U32[1], result, values[i], ctx.ConstU32(fields[i].offset),
                                      ctx.ConstU32(fields[i].size));
    }
    return result;
}

template <size_t N>
std::array<Id, N> Components(EmitContext& ctx, const VectorIds& type, Id value) {
    std::array<Id, N> result{};
    for (u32 i = 0; i < N; i++) {
        result[i] = ctx.OpCompositeExtract(type[1], value, i);
    }
    return result;
}

/// An integer vector (as float bits) packed into fields.
template <size_t N>
Id PackFields(EmitContext& ctx, Id value, const std::array<Field, N>& fields) {
    const Id integers{ctx.OpBitcast(ctx.U32[N], value)};
    return InsertFields(ctx, Components<N>(ctx, ctx.U32, integers), fields);
}

/// Fields unpacked into an integer vector (returned as float bits).
template <size_t N>
Id UnpackFields(EmitContext& ctx, Id value, const std::array<Field, N>& fields, bool is_signed) {
    const auto values = ExtractFields(ctx, value, fields, is_signed);
    const Id vector = [&] {
        if constexpr (N == 2) {
            return ctx.OpCompositeConstruct(ctx.U32[2], values[0], values[1]);
        } else {
            return ctx.OpCompositeConstruct(ctx.U32[4], values[0], values[1], values[2], values[3]);
        }
    }();
    return ctx.OpBitcast(ctx.F32[N], vector);
}

/// A vec4 constant.
Id Vec4(EmitContext& ctx, float x, float y, float z, float w) {
    return ctx.ConstantComposite(ctx.F32[4], ctx.ConstF32(x), ctx.ConstF32(y), ctx.ConstF32(z),
                                 ctx.ConstF32(w));
}

} // namespace

#define BITCAST(name, type)                                                                        \
    Id Emit##name(EmitContext& ctx, Id value) {                                                    \
        return ctx.OpBitcast(type, value);                                                         \
    }

BITCAST(BitCastU16F16, ctx.U16)
BITCAST(BitCastU32F32, ctx.U32[1])
BITCAST(BitCastF16U16, ctx.F16[1])
BITCAST(BitCastF32U32, ctx.F32[1])
BITCAST(PackUint2x32, ctx.U64)
BITCAST(UnpackUint2x32, ctx.U32[2])
BITCAST(PackDouble2x32, ctx.F64[1])
BITCAST(UnpackDouble2x32, ctx.U32[2])

#undef BITCAST

#define GLSL_PACK(name, op, type)                                                                  \
    Id Emit##name(EmitContext& ctx, Id value) {                                                    \
        return ctx.op(type, value);                                                                \
    }

GLSL_PACK(PackUnorm2x16, OpPackUnorm2x16, ctx.U32[1])
GLSL_PACK(UnpackUnorm2x16, OpUnpackUnorm2x16, ctx.F32[2])
GLSL_PACK(PackSnorm2x16, OpPackSnorm2x16, ctx.U32[1])
GLSL_PACK(UnpackSnorm2x16, OpUnpackSnorm2x16, ctx.F32[2])
GLSL_PACK(PackHalf2x16, OpPackHalf2x16, ctx.U32[1])
GLSL_PACK(UnpackHalf2x16, OpUnpackHalf2x16, ctx.F32[2])
GLSL_PACK(PackUnorm4x8, OpPackUnorm4x8, ctx.U32[1])
GLSL_PACK(UnpackUnorm4x8, OpUnpackUnorm4x8, ctx.F32[4])
GLSL_PACK(PackSnorm4x8, OpPackSnorm4x8, ctx.U32[1])
GLSL_PACK(UnpackSnorm4x8, OpUnpackSnorm4x8, ctx.F32[4])

#undef GLSL_PACK

// Integer formats (signed and unsigned pack alike).

Id EmitPackUint2x16(EmitContext& ctx, Id value) {
    return PackFields(ctx, value, Fields16x2);
}

Id EmitPackSint2x16(EmitContext& ctx, Id value) {
    return PackFields(ctx, value, Fields16x2);
}

Id EmitUnpackUint2x16(EmitContext& ctx, Id value) {
    return UnpackFields(ctx, value, Fields16x2, false);
}

Id EmitUnpackSint2x16(EmitContext& ctx, Id value) {
    return UnpackFields(ctx, value, Fields16x2, true);
}

Id EmitPackUint4x8(EmitContext& ctx, Id value) {
    return PackFields(ctx, value, Fields8x4);
}

Id EmitPackSint4x8(EmitContext& ctx, Id value) {
    return PackFields(ctx, value, Fields8x4);
}

Id EmitUnpackUint4x8(EmitContext& ctx, Id value) {
    return UnpackFields(ctx, value, Fields8x4, false);
}

Id EmitUnpackSint4x8(EmitContext& ctx, Id value) {
    return UnpackFields(ctx, value, Fields8x4, true);
}

Id EmitPackUint2_10_10_10(EmitContext& ctx, Id value) {
    return PackFields(ctx, value, Fields2_10_10_10);
}

Id EmitPackSint2_10_10_10(EmitContext& ctx, Id value) {
    return PackFields(ctx, value, Fields2_10_10_10);
}

Id EmitUnpackUint2_10_10_10(EmitContext& ctx, Id value) {
    return UnpackFields(ctx, value, Fields2_10_10_10, false);
}

Id EmitUnpackSint2_10_10_10(EmitContext& ctx, Id value) {
    return UnpackFields(ctx, value, Fields2_10_10_10, true);
}

// Unsigned 11/11/10-bit floats.

Id EmitPackUfloat10_11_11(EmitContext& ctx, Id value) {
    const auto [x, y, z] = Components<3>(ctx, ctx.F32, value);
    const auto cvt_x{ctx.OpFunctionCall(ctx.U32[1], ctx.f32_to_uf11, x)};
    const auto cvt_y{ctx.OpFunctionCall(ctx.U32[1], ctx.f32_to_uf11, y)};
    const auto cvt_z{ctx.OpFunctionCall(ctx.U32[1], ctx.f32_to_uf10, z)};
    return InsertFields(ctx, std::array{cvt_x, cvt_y, cvt_z}, Fields10_11_11);
}

Id EmitUnpackUfloat10_11_11(EmitContext& ctx, Id value) {
    const auto [x, y, z] = ExtractFields(ctx, value, Fields10_11_11, false);
    const auto cvt_x{ctx.OpFunctionCall(ctx.F32[1], ctx.uf11_to_f32, x)};
    const auto cvt_y{ctx.OpFunctionCall(ctx.F32[1], ctx.uf11_to_f32, y)};
    const auto cvt_z{ctx.OpFunctionCall(ctx.F32[1], ctx.uf10_to_f32, z)};
    return ctx.OpCompositeConstruct(ctx.F32[3], cvt_x, cvt_y, cvt_z);
}

// Normalized 2:10:10:10: clamp, scale, round, pack (and back).

Id EmitPackUnorm2_10_10_10(EmitContext& ctx, Id value) {
    const auto min{Vec4(ctx, 0.f, 0.f, 0.f, 0.f)};
    const auto max{Vec4(ctx, 1.f, 1.f, 1.f, 1.f)};
    const auto clamped{ctx.OpFClamp(ctx.F32[4], value, min, max)};
    const auto scale{Vec4(ctx, 1023.f, 1023.f, 1023.f, 3.f)};
    const auto scaled{ctx.OpFMul(ctx.F32[4], clamped, scale)};
    const auto as_uint{ctx.OpConvertFToU(ctx.U32[4], ctx.OpRoundEven(ctx.F32[4], scaled))};
    return EmitPackUint2_10_10_10(ctx, ctx.OpBitcast(ctx.F32[4], as_uint));
}

Id EmitUnpackUnorm2_10_10_10(EmitContext& ctx, Id value) {
    const auto unpacked{ctx.OpBitcast(ctx.U32[4], EmitUnpackUint2_10_10_10(ctx, value))};
    const auto as_float{ctx.OpConvertUToF(ctx.F32[4], unpacked)};
    const auto scale{Vec4(ctx, 1023.f, 1023.f, 1023.f, 3.f)};
    return ctx.OpFDiv(ctx.F32[4], as_float, scale);
}

Id EmitPackSnorm2_10_10_10(EmitContext& ctx, Id value) {
    const auto min{Vec4(ctx, -1.f, -1.f, -1.f, -1.f)};
    const auto max{Vec4(ctx, 1.f, 1.f, 1.f, 1.f)};
    const auto clamped{ctx.OpFClamp(ctx.F32[4], value, min, max)};
    const auto scale{Vec4(ctx, 511.f, 511.f, 511.f, 1.f)};
    const auto scaled{ctx.OpFMul(ctx.F32[4], clamped, scale)};
    const auto as_sint{ctx.OpConvertFToS(ctx.U32[4], ctx.OpRoundEven(ctx.F32[4], scaled))};
    return EmitPackSint2_10_10_10(ctx, ctx.OpBitcast(ctx.F32[4], as_sint));
}

Id EmitUnpackSnorm2_10_10_10(EmitContext& ctx, Id value) {
    const auto unpacked{ctx.OpBitcast(ctx.U32[4], EmitUnpackSint2_10_10_10(ctx, value))};
    const auto as_float{ctx.OpConvertSToF(ctx.F32[4], unpacked)};
    const auto scale{Vec4(ctx, 511.f, 511.f, 511.f, 1.f)};
    return ctx.OpFDiv(ctx.F32[4], as_float, scale);
}

} // namespace Shader::Backend::SPIRV
