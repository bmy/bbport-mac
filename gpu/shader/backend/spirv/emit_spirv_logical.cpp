// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Boolean logic and selects.
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

Id EmitLogicalOr(EmitContext& ctx, Id a, Id b) {
    return ctx.OpLogicalOr(ctx.U1[1], a, b);
}

Id EmitLogicalAnd(EmitContext& ctx, Id a, Id b) {
    return ctx.OpLogicalAnd(ctx.U1[1], a, b);
}

Id EmitLogicalXor(EmitContext& ctx, Id a, Id b) {
    return ctx.OpLogicalNotEqual(ctx.U1[1], a, b);
}

Id EmitLogicalNot(EmitContext& ctx, Id value) {
    return ctx.OpLogicalNot(ctx.U1[1], value);
}

Id EmitSelectU1(EmitContext& ctx, Id cond, Id true_value, Id false_value) {
    return ctx.OpSelect(ctx.U1[1], cond, true_value, false_value);
}

Id EmitSelectU32(EmitContext& ctx, Id cond, Id true_value, Id false_value) {
    return ctx.OpSelect(ctx.U32[1], cond, true_value, false_value);
}

Id EmitSelectF32(EmitContext& ctx, Id cond, Id true_value, Id false_value) {
    return ctx.OpSelect(ctx.F32[1], cond, true_value, false_value);
}

Id EmitSelectU64(EmitContext& ctx, Id cond, Id true_value, Id false_value) {
    return ctx.OpSelect(ctx.U64, cond, true_value, false_value);
}

// Undefined values (the 8/16/64-bit ones are never made).

Id EmitUndefU1(EmitContext& ctx) {
    return ctx.OpUndef(ctx.U1[1]);
}

Id EmitUndefU8(EmitContext&) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitUndefU16(EmitContext&) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitUndefU32(EmitContext& ctx) {
    return ctx.OpUndef(ctx.U32[1]);
}

Id EmitUndefU32x2(EmitContext& ctx) {
    return ctx.OpUndef(ctx.U32[2]);
}

Id EmitUndefU32x3(EmitContext& ctx) {
    return ctx.OpUndef(ctx.U32[3]);
}

Id EmitUndefU32x4(EmitContext& ctx) {
    return ctx.OpUndef(ctx.U32[4]);
}

Id EmitUndefF32(EmitContext& ctx) {
    return ctx.OpUndef(ctx.F32[1]);
}

Id EmitUndefU64(EmitContext&) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

} // namespace Shader::Backend::SPIRV
