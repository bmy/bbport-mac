// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Barriers, LDS accesses and subgroup operations.
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {

void MemoryBarrier(EmitContext& ctx, spv::Scope scope) {
    const auto semantics{
        spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::UniformMemory |
        spv::MemorySemanticsMask::WorkgroupMemory | spv::MemorySemanticsMask::AtomicCounterMemory |
        spv::MemorySemanticsMask::ImageMemory};
    ctx.OpMemoryBarrier(ctx.ConstU32(static_cast<u32>(scope)),
                        ctx.ConstU32(static_cast<u32>(semantics)));
}

/// The pointer to the LDS element an access of 2^shift bytes at byte `offset` reads.
Id SharedPointer(EmitContext& ctx, Id offset, u32 shift, Id element_pointer, Id array) {
    const Id shift_id{ctx.ConstU32(shift)};
    const Id index{ctx.OpShiftRightLogical(ctx.U32[1], offset, shift_id)};
    return ctx.EmitSharedMemoryAccess(element_pointer, array, index);
}

Id SubgroupScope(EmitContext& ctx) {
    return ctx.ConstU32(static_cast<u32>(spv::Scope::Subgroup));
}

/// A 64-bit lane mask as the uvec4 ballot form.
Id BallotVector(EmitContext& ctx, Id mask) {
    return ctx.OpCompositeConstruct(ctx.U32[4], ctx.OpBitcast(ctx.U32[2], mask),
                                    ctx.u32_zero_value, ctx.u32_zero_value);
}

} // namespace

/// s_barrier: workgroup execution barrier; tessellation control shaders (one invocation per
/// output control point) synchronize without memory semantics.
void EmitBarrier(EmitContext& ctx) {
    const auto execution{spv::Scope::Workgroup};
    spv::Scope memory;
    spv::MemorySemanticsMask memory_semantics;
    if (ctx.sw_stage == Shader::SwStage::TessellationControl) {
        memory = spv::Scope::Invocation;
        memory_semantics = spv::MemorySemanticsMask::MaskNone;
    } else {
        memory = spv::Scope::Workgroup;
        memory_semantics =
            spv::MemorySemanticsMask::AcquireRelease | spv::MemorySemanticsMask::WorkgroupMemory;
    }
    ctx.OpControlBarrier(ctx.ConstU32(static_cast<u32>(execution)),
                         ctx.ConstU32(static_cast<u32>(memory)),
                         ctx.ConstU32(static_cast<u32>(memory_semantics)));
}

void EmitWorkgroupMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Workgroup);
}

void EmitDeviceMemoryBarrier(EmitContext& ctx) {
    MemoryBarrier(ctx, spv::Scope::Device);
}

Id EmitLoadSharedU16(EmitContext& ctx, Id offset) {
    const Id pointer = SharedPointer(ctx, offset, 1U, ctx.shared_u16, ctx.shared_memory_u16);
    return ctx.OpLoad(ctx.U16, pointer);
}

Id EmitLoadSharedU32(EmitContext& ctx, Id offset) {
    const Id pointer = SharedPointer(ctx, offset, 2U, ctx.shared_u32, ctx.shared_memory_u32);
    return ctx.OpLoad(ctx.U32[1], pointer);
}

Id EmitLoadSharedU64(EmitContext& ctx, Id offset) {
    const Id pointer = SharedPointer(ctx, offset, 3U, ctx.shared_u64, ctx.shared_memory_u64);
    return ctx.OpLoad(ctx.U64, pointer);
}

void EmitWriteSharedU16(EmitContext& ctx, Id offset, Id value) {
    ctx.OpStore(SharedPointer(ctx, offset, 1U, ctx.shared_u16, ctx.shared_memory_u16), value);
}

void EmitWriteSharedU32(EmitContext& ctx, Id offset, Id value) {
    ctx.OpStore(SharedPointer(ctx, offset, 2U, ctx.shared_u32, ctx.shared_memory_u32), value);
}

void EmitWriteSharedU64(EmitContext& ctx, Id offset, Id value) {
    ctx.OpStore(SharedPointer(ctx, offset, 3U, ctx.shared_u64, ctx.shared_memory_u64), value);
}

// Subgroup operations (a GCN wave maps to a subgroup).

Id EmitWarpId(EmitContext& ctx) {
    UNREACHABLE();
}

Id EmitLaneId(EmitContext& ctx) {
    return ctx.OpLoad(ctx.U32[1], ctx.subgroup_local_invocation_id);
}

Id EmitQuadBroadcast(EmitContext& ctx, Id value, Id index) {
    return ctx.OpGroupNonUniformQuadBroadcast(ctx.U32[1], SubgroupScope(ctx), value, index);
}

Id EmitReadFirstLane(EmitContext& ctx, Id value) {
    return ctx.OpGroupNonUniformBroadcastFirst(ctx.U32[1], SubgroupScope(ctx), value);
}

Id EmitShuffle(EmitContext& ctx, Id value, Id index) {
    return ctx.OpGroupNonUniformShuffle(ctx.U32[1], SubgroupScope(ctx), value, index);
}

Id EmitShuffleXor(EmitContext& ctx, Id value, Id mask) {
    return ctx.OpGroupNonUniformShuffleXor(ctx.U32[1], SubgroupScope(ctx), value, mask);
}

Id EmitReadLane(EmitContext& ctx, Id value, Id lane) {
    return ctx.OpGroupNonUniformBroadcast(ctx.U32[1], SubgroupScope(ctx), value, lane);
}

/// (Lane writes are resolved by the read lane elimination; any left are dead.)
Id EmitWriteLane(EmitContext& ctx, Id value, Id write_value, u32 lane) {
    return ctx.u32_zero_value;
}

Id EmitBallot(EmitContext& ctx, Id bit) {
    const Id ballot{ctx.OpGroupNonUniformBallot(ctx.U32[4], SubgroupScope(ctx), bit)};
    return ctx.OpBitcast(ctx.U64, ctx.OpVectorShuffle(ctx.U32[2], ballot, ballot, 0, 1));
}

Id EmitBallotFindLsb(EmitContext& ctx, Id mask) {
    const Id value{BallotVector(ctx, mask)};
    return ctx.OpGroupNonUniformBallotFindLSB(ctx.U32[1], SubgroupScope(ctx), value);
}

Id EmitInverseBallot(EmitContext& ctx, Id mask) {
    const Id value{BallotVector(ctx, mask)};
    return ctx.OpGroupNonUniformInverseBallot(ctx.U1[1], SubgroupScope(ctx), value);
}

Id EmitGroupAny(EmitContext& ctx, Id bit) {
    return ctx.OpGroupNonUniformAny(ctx.U1[1], SubgroupScope(ctx), bit);
}

} // namespace Shader::Backend::SPIRV
