// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Atomics on LDS, buffers and images (device scope, relaxed). Buffers reached through the page
// table (BB_LAYER_MEMORY) get their pointer from it, out of range accesses hitting a trash area
// with a zero result. Float min/max without host support compare the bits as integers by sign.
#include <bit>

#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {

using PointerType = EmitContext::PointerType;
using PointerSize = EmitContext::PointerSize;
using AtomicOp = Id (Sirit::Module::*)(Id, Id, Id, Id, Id);
using AtomicIncDecOp = Id (Sirit::Module::*)(Id, Id, Id, Id);
using AtomicCmpSwapOp = Id (Sirit::Module::*)(Id, Id, Id, Id, Id, Id, Id);

/// Device scope, no ordering.
std::pair<Id, Id> AtomicArgs(EmitContext& ctx) {
    const Id scope{ctx.ConstU32(static_cast<u32>(spv::Scope::Device))};
    const Id semantics{ctx.u32_zero_value};
    return {scope, semantics};
}

/// The LDS element of 2^shift bytes at byte `offset`.
Id SharedPointer(EmitContext& ctx, Id offset, bool is_64) {
    const Id shift_id{ctx.ConstU32(is_64 ? 3U : 2U)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    return is_64 ? ctx.EmitSharedMemoryAccess(ctx.shared_u64, ctx.shared_memory_u64, index)
                 : ctx.EmitSharedMemoryAccess(ctx.shared_u32, ctx.shared_memory_u32, index);
}

Id SharedAtomic(EmitContext& ctx, Id offset, Id value, bool is_64, AtomicOp op) {
    const Id pointer{SharedPointer(ctx, offset, is_64)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*op)(is_64 ? ctx.U64 : ctx.U32[1], pointer, scope, semantics, value);
}

Id SharedAtomicIncDec(EmitContext& ctx, Id offset, bool is_64, AtomicIncDecOp op) {
    const Id pointer{SharedPointer(ctx, offset, is_64)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*op)(is_64 ? ctx.U64 : ctx.U32[1], pointer, scope, semantics);
}

Id SharedAtomicCmpSwap(EmitContext& ctx, Id offset, Id value, Id cmp_value, bool is_64) {
    const Id pointer{SharedPointer(ctx, offset, is_64)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.OpAtomicCompareExchange(is_64 ? ctx.U64 : ctx.U32[1], pointer, scope, semantics,
                                       semantics, value, cmp_value);
}

/// A buffer atomic's pointer (and, for paged buffers, whether it is in range).
struct AtomicTarget {
    Id pointer;
    Id guard; ///< invalid for bound buffers
};

AtomicTarget BufferPointer(EmitContext& ctx, const EmitContext::BufferDefinition& buffer,
                           PointerType alias, PointerSize size, Id address, u32 bytes, Id type) {
    if (const Id offset = buffer.Offset(size); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    if (buffer.paged) {
        const Id byte_offset = ctx.OpShiftLeftLogical(
            ctx.U32[1], address, ctx.ConstU32(u32(std::countr_zero(bytes))));
        const auto access = ctx.PagedPointer(buffer, byte_offset, bytes, type);
        return {access.pointer, access.guard};
    }
    const auto [id, pointer_type] = buffer.Alias(alias);
    return {ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, address), Id{}};
}

Id Guarded(EmitContext& ctx, const AtomicTarget& target, Id type, Id result, Id zero) {
    return Sirit::ValidId(target.guard) ? ctx.OpSelect(type, target.guard, result, zero) : result;
}

template <bool is_float = false>
Id BufferAtomicU32(EmitContext& ctx, u32 handle, Id address, Id value, AtomicOp op) {
    const Id type = is_float ? ctx.F32[1] : ctx.U32[1];
    const auto target =
        BufferPointer(ctx, ctx.buffers[handle], is_float ? PointerType::F32 : PointerType::U32,
                      PointerSize::B32, address, 4, type);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return Guarded(ctx, target, type, (ctx.*op)(type, target.pointer, scope, semantics, value),
                   is_float ? ctx.f32_zero_value : ctx.u32_zero_value);
}

Id BufferAtomicU32IncDec(EmitContext& ctx, u32 handle, Id address, AtomicIncDecOp op) {
    const auto target = BufferPointer(ctx, ctx.buffers[handle], PointerType::U32,
                                      PointerSize::B32, address, 4, ctx.U32[1]);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return Guarded(ctx, target, ctx.U32[1], (ctx.*op)(ctx.U32[1], target.pointer, scope, semantics),
                   ctx.u32_zero_value);
}

Id BufferAtomicU32CmpSwap(EmitContext& ctx, u32 handle, Id address, Id value, Id cmp_value) {
    const auto target = BufferPointer(ctx, ctx.buffers[handle], PointerType::U32,
                                      PointerSize::B32, address, 4, ctx.U32[1]);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return Guarded(ctx, target, ctx.U32[1],
                   ctx.OpAtomicCompareExchange(ctx.U32[1], target.pointer, scope, semantics,
                                               semantics, value, cmp_value),
                   ctx.u32_zero_value);
}

Id BufferAtomicU64(EmitContext& ctx, u32 handle, Id address, Id value, AtomicOp op) {
    const auto target = BufferPointer(ctx, ctx.buffers[handle], PointerType::U64,
                                      PointerSize::B64, address, 8, ctx.U64);
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return Guarded(ctx, target, ctx.U64, (ctx.*op)(ctx.U64, target.pointer, scope, semantics, value),
                   ctx.u64_zero_value);
}

Id ImageTexel(EmitContext& ctx, u32 handle, Id coords, Id pointer_type) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    return ctx.OpImageTexelPointer(pointer_type, texture.id, coords, ctx.ConstU32(0U));
}

Id ImageAtomic(EmitContext& ctx, u32 handle, Id coords, Id value, bool is_float, AtomicOp op) {
    const Id pointer{ImageTexel(ctx, handle, coords, is_float ? ctx.image_f32 : ctx.image_u32)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return (ctx.*op)(is_float ? ctx.F32[1] : ctx.U32[1], pointer, scope, semantics, value);
}

/// The sign bit of a float's bits.
Id IsNegative(EmitContext& ctx, Id bits) {
    return ctx.OpINotEqual(
        ctx.U1[1], ctx.OpBitFieldUExtract(ctx.U32[1], bits, ctx.ConstU32(31u), ctx.ConstU32(1u)),
        ctx.u32_zero_value);
}

} // namespace

// LDS

#define SHARED_ATOMIC(name, op)                                                                    \
    Id EmitSharedAtomic##name##32(EmitContext& ctx, Id offset, Id value) {                         \
        return SharedAtomic(ctx, offset, value, false, &Sirit::Module::op);                        \
    }                                                                                              \
    Id EmitSharedAtomic##name##64(EmitContext& ctx, Id offset, Id value) {                         \
        return SharedAtomic(ctx, offset, value, true, &Sirit::Module::op);                         \
    }

SHARED_ATOMIC(IAdd, OpAtomicIAdd)
SHARED_ATOMIC(ISub, OpAtomicISub)
SHARED_ATOMIC(UMax, OpAtomicUMax)
SHARED_ATOMIC(SMax, OpAtomicSMax)
SHARED_ATOMIC(UMin, OpAtomicUMin)
SHARED_ATOMIC(SMin, OpAtomicSMin)
SHARED_ATOMIC(And, OpAtomicAnd)
SHARED_ATOMIC(Or, OpAtomicOr)
SHARED_ATOMIC(Xor, OpAtomicXor)

#undef SHARED_ATOMIC

Id EmitSharedAtomicCmpSwap32(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    return SharedAtomicCmpSwap(ctx, offset, value, cmp_value, false);
}

Id EmitSharedAtomicCmpSwap64(EmitContext& ctx, Id offset, Id value, Id cmp_value) {
    return SharedAtomicCmpSwap(ctx, offset, value, cmp_value, true);
}

Id EmitSharedAtomicInc32(EmitContext& ctx, Id offset) {
    return SharedAtomicIncDec(ctx, offset, false, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitSharedAtomicInc64(EmitContext& ctx, Id offset) {
    return SharedAtomicIncDec(ctx, offset, true, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitSharedAtomicDec32(EmitContext& ctx, Id offset) {
    return SharedAtomicIncDec(ctx, offset, false, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitSharedAtomicDec64(EmitContext& ctx, Id offset) {
    return SharedAtomicIncDec(ctx, offset, true, &Sirit::Module::OpAtomicIDecrement);
}

// Buffers

#define BUFFER_ATOMIC(name, width, impl, op)                                                       \
    Id EmitBufferAtomic##name##width(EmitContext& ctx, IR::Inst*, u32 handle, Id address,          \
                                     Id value) {                                                   \
        return impl(ctx, handle, address, value, &Sirit::Module::op);                              \
    }

BUFFER_ATOMIC(IAdd, 32, BufferAtomicU32, OpAtomicIAdd)
BUFFER_ATOMIC(IAdd, 64, BufferAtomicU64, OpAtomicIAdd)
BUFFER_ATOMIC(ISub, 32, BufferAtomicU32, OpAtomicISub)
BUFFER_ATOMIC(SMin, 32, BufferAtomicU32, OpAtomicSMin)
BUFFER_ATOMIC(SMin, 64, BufferAtomicU64, OpAtomicSMin)
BUFFER_ATOMIC(UMin, 32, BufferAtomicU32, OpAtomicUMin)
BUFFER_ATOMIC(UMin, 64, BufferAtomicU64, OpAtomicUMin)
BUFFER_ATOMIC(SMax, 32, BufferAtomicU32, OpAtomicSMax)
BUFFER_ATOMIC(SMax, 64, BufferAtomicU64, OpAtomicSMax)
BUFFER_ATOMIC(UMax, 32, BufferAtomicU32, OpAtomicUMax)
BUFFER_ATOMIC(UMax, 64, BufferAtomicU64, OpAtomicUMax)
BUFFER_ATOMIC(And, 32, BufferAtomicU32, OpAtomicAnd)
BUFFER_ATOMIC(Or, 32, BufferAtomicU32, OpAtomicOr)
BUFFER_ATOMIC(Xor, 32, BufferAtomicU32, OpAtomicXor)
BUFFER_ATOMIC(Swap, 32, BufferAtomicU32, OpAtomicExchange)

#undef BUFFER_ATOMIC

Id EmitBufferAtomicInc32(EmitContext& ctx, IR::Inst*, u32 handle, Id address) {
    return BufferAtomicU32IncDec(ctx, handle, address, &Sirit::Module::OpAtomicIIncrement);
}

Id EmitBufferAtomicDec32(EmitContext& ctx, IR::Inst*, u32 handle, Id address) {
    return BufferAtomicU32IncDec(ctx, handle, address, &Sirit::Module::OpAtomicIDecrement);
}

Id EmitBufferAtomicCmpSwap32(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value,
                             Id cmp_value) {
    return BufferAtomicU32CmpSwap(ctx, handle, address, value, cmp_value);
}

Id EmitBufferAtomicFCmpSwap32(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value,
                              Id cmp_value) {
    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto u32_cmp = ctx.OpBitcast(ctx.U32[1], cmp_value);
    const auto result = BufferAtomicU32CmpSwap(ctx, handle, address, u32_value, u32_cmp);
    return ctx.OpBitcast(ctx.F32[1], result);
}

// Float min/max without host support, on the bits: non-negative floats order like signed
// integers, negative ones in reverse like unsigned ones. (Both atomics are issued.)

Id EmitBufferAtomicFMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    if (ctx.profile.supports_buffer_fp32_atomic_min_max) {
        return BufferAtomicU32<true>(ctx, handle, address, value, &Sirit::Module::OpAtomicFMin);
    }
    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto negative = IsNegative(ctx, u32_value);
    return ctx.OpSelect(
        ctx.F32[1], negative,
        EmitBitCastF32U32(ctx, EmitBufferAtomicUMax32(ctx, inst, handle, address, u32_value)),
        EmitBitCastF32U32(ctx, EmitBufferAtomicSMin32(ctx, inst, handle, address, u32_value)));
}

Id EmitBufferAtomicFMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    if (ctx.profile.supports_buffer_fp32_atomic_min_max) {
        return BufferAtomicU32<true>(ctx, handle, address, value, &Sirit::Module::OpAtomicFMax);
    }
    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto negative = IsNegative(ctx, u32_value);
    return ctx.OpSelect(
        ctx.F32[1], negative,
        EmitBitCastF32U32(ctx, EmitBufferAtomicUMin32(ctx, inst, handle, address, u32_value)),
        EmitBitCastF32U32(ctx, EmitBufferAtomicSMax32(ctx, inst, handle, address, u32_value)));
}

// Images

#define IMAGE_ATOMIC(name, op)                                                                     \
    Id EmitImageAtomic##name##32(EmitContext& ctx, IR::Inst*, u32 handle, Id coords, Id value) {   \
        return ImageAtomic(ctx, handle, coords, value, false, &Sirit::Module::op);                 \
    }

IMAGE_ATOMIC(IAdd, OpAtomicIAdd)
IMAGE_ATOMIC(SMin, OpAtomicSMin)
IMAGE_ATOMIC(UMin, OpAtomicUMin)
IMAGE_ATOMIC(SMax, OpAtomicSMax)
IMAGE_ATOMIC(UMax, OpAtomicUMax)
IMAGE_ATOMIC(And, OpAtomicAnd)
IMAGE_ATOMIC(Or, OpAtomicOr)
IMAGE_ATOMIC(Xor, OpAtomicXor)
IMAGE_ATOMIC(Exchange, OpAtomicExchange)

#undef IMAGE_ATOMIC

Id EmitImageAtomicCmpSwap32(EmitContext& ctx, IR::Inst*, u32 handle, Id coords, Id value,
                            Id cmp_value) {
    const Id pointer{ImageTexel(ctx, handle, coords, ctx.image_u32)};
    const auto [scope, semantics]{AtomicArgs(ctx)};
    return ctx.OpAtomicCompareExchange(ctx.U32[1], pointer, scope, semantics, semantics, value,
                                       cmp_value);
}

Id EmitImageAtomicFMax32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (ctx.profile.supports_image_fp32_atomic_min_max) {
        return ImageAtomic(ctx, handle, coords, value, true, &Sirit::Module::OpAtomicFMax);
    }
    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto negative = IsNegative(ctx, u32_value);
    return ctx.OpSelect(
        ctx.F32[1], negative,
        EmitBitCastF32U32(ctx, EmitImageAtomicUMin32(ctx, inst, handle, coords, u32_value)),
        EmitBitCastF32U32(ctx, EmitImageAtomicSMax32(ctx, inst, handle, coords, u32_value)));
}

Id EmitImageAtomicFMin32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id value) {
    if (ctx.profile.supports_image_fp32_atomic_min_max) {
        return ImageAtomic(ctx, handle, coords, value, true, &Sirit::Module::OpAtomicFMin);
    }
    const auto u32_value = ctx.OpBitcast(ctx.U32[1], value);
    const auto negative = IsNegative(ctx, u32_value);
    return ctx.OpSelect(
        ctx.F32[1], negative,
        EmitBitCastF32U32(ctx, EmitImageAtomicUMax32(ctx, inst, handle, coords, u32_value)),
        EmitBitCastF32U32(ctx, EmitImageAtomicSMin32(ctx, inst, handle, coords, u32_value)));
}

Id EmitImageAtomicInc32(EmitContext&, IR::Inst*, u32, Id, Id) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitImageAtomicDec32(EmitContext&, IR::Inst*, u32, Id, Id) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

/// (GDS counters are lowered to buffer atomics by the resource patching.)
Id EmitDataAppend(EmitContext& ctx, Id gds_dw_offset, Id exec) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

Id EmitDataConsume(EmitContext& ctx, Id gds_dw_offset, Id exec) {
    UNREACHABLE_MSG("SPIR-V Instruction");
}

} // namespace Shader::Backend::SPIRV
