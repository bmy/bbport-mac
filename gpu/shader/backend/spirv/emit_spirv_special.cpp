// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Prologue and epilogue code (per-vertex attributes, workgroup index, buffer properties; the
// depth range and clip space conversions, object motion vectors, the UI edge stretch), discard,
// geometry emission, debug printf and the clock.
#include "shader/ir/debug_print.h"
#include "shader/ir/microinstruction.h"
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

void EmitPrologue(EmitContext& ctx) {
    if (ctx.hw_stage == HwStage::Fragment) {
        ctx.DefineAmdPerVertexAttribs();
    }
    if (ctx.info.loads.Get(IR::Attribute::WorkgroupIndex)) {
        ctx.DefineWorkgroupIndex();
    }
    ctx.DefineBufferProperties();
    ctx.DefinePagedRecords();
}

namespace {

/// [-1, 1] depth (the PS4 range when set) to Vulkan's [0, 1]: z = (z + w) / 2.
void ConvertDepthMode(EmitContext& ctx) {
    const Id type{ctx.F32[1]};
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id screen_depth{ctx.OpFMul(type, ctx.OpFAdd(type, z, w), ctx.Constant(type, 0.5f))};
    const Id vector{ctx.OpCompositeInsert(ctx.F32[4], screen_depth, position, 2u)};
    ctx.OpStore(ctx.output_position, vector);
}

Id LoadPushFloat(EmitContext& ctx, Id type, u32 index) {
    const Id pointer{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                       ctx.push_data_block, ctx.ConstU32(index))};
    return ctx.OpLoad(type, pointer);
}

/// With clipping disabled the shader outputs window coordinates: back to clip space through
/// the viewport transform in the push data (the viewport is the maximum the host allows).
void ConvertPositionToClipSpace(EmitContext& ctx) {
    ASSERT_MSG(!ctx.info.stores.GetAny(IR::Attribute::ViewportIndex),
               "Multi-viewport with shader clip space conversion not yet implemented.");
    const Id type{ctx.F32[1]};
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id x{ctx.OpCompositeExtract(type, position, 0u)};
    const Id y{ctx.OpCompositeExtract(type, position, 1u)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id xoffset{LoadPushFloat(ctx, type, PushData::XOffsetIndex)};
    const Id yoffset{LoadPushFloat(ctx, type, PushData::YOffsetIndex)};
    const Id xscale{LoadPushFloat(ctx, type, PushData::XScaleIndex)};
    const Id yscale{LoadPushFloat(ctx, type, PushData::YScaleIndex)};
    const Id vport_w =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_width / 2, 8_KB)));
    const Id wnd_x = ctx.OpFAdd(type, ctx.OpFMul(type, x, xscale), xoffset);
    const Id ndc_x = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_x, vport_w), ctx.Constant(type, 1.f));
    const Id vport_h =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_height / 2, 8_KB)));
    const Id wnd_y = ctx.OpFAdd(type, ctx.OpFMul(type, y, yscale), yoffset);
    const Id ndc_y = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_y, vport_h), ctx.Constant(type, 1.f));
    const Id vector{ctx.OpCompositeConstruct(ctx.F32[4], std::array<Id, 4>({ndc_x, ndc_y, z, w}))};
    ctx.OpStore(ctx.output_position, vector);
}

// Object motion vectors (runtime_info.h, MotionVectors). The vertex shader stores its
// clip position for this frame and loads the one of the previous frame (same draw, same
// vertex), both by buffer device address. Disabled accesses are branched around: a shared
// scratch element would race between all the inactive vertex invocations.
void EmitVertexMotion(EmitContext& ctx) {
    const Id u32_type = ctx.U32[1];
    const Id position = ctx.OpLoad(ctx.F32[4], ctx.output_position);
    const Id param_ptr = ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, u32_type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::MotionParamIndex));
    const Id param_index = ctx.OpLoad(u32_type, param_ptr);
    const auto address = [&](u64 base, Id index, u32 stride) {
        return ctx.OpIAdd(ctx.U64, ctx.Constant(ctx.U64, base),
                          ctx.OpIMul(ctx.U64, ctx.OpUConvert(ctx.U64, index),
                                     ctx.Constant(ctx.U64, u64(stride))));
    };
    const Id u32x4_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, ctx.U32[4]);
    const Id f32x4_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, ctx.F32[4]);
    const Id params = ctx.OpLoad(
        ctx.U32[4],
        ctx.OpConvertUToPtr(u32x4_ptr, address(MotionVectors::params_address, param_index, 32)),
        spv::MemoryAccessMask::Aligned, 16u);
    const Id store_base = ctx.OpCompositeExtract(u32_type, params, 0u);
    const Id load_base = ctx.OpCompositeExtract(u32_type, params, 1u);
    const Id vertices = ctx.OpCompositeExtract(u32_type, params, 2u);
    const Id flags = ctx.OpCompositeExtract(u32_type, params, 3u);
    const Id offsets = ctx.OpLoad(
        ctx.U32[4], ctx.OpConvertUToPtr(u32x4_ptr,
            address(MotionVectors::params_address + 16, param_index, 32)),
        spv::MemoryAccessMask::Aligned, 16u);
    const Id first_vertex = ctx.OpCompositeExtract(u32_type, offsets, 0u);
    const Id first_instance = ctx.OpCompositeExtract(u32_type, offsets, 1u);
    const Id instances = ctx.OpCompositeExtract(u32_type, offsets, 2u);
    const Id vertex = ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.vertex_index), first_vertex);
    const Id instance = ctx.OpISub(u32_type, ctx.OpLoad(u32_type, ctx.instance_id), first_instance);
    const Id bool_type = ctx.U1[1];
    const Id in_range = ctx.OpLogicalAnd(bool_type, ctx.OpULessThan(bool_type, vertex, vertices),
                                         ctx.OpULessThan(bool_type, instance, instances));
    const Id slot = ctx.OpIAdd(u32_type, vertex, ctx.OpIMul(u32_type, instance, vertices));
    const auto flag = [&](u32 bit) {
        return ctx.OpLogicalAnd(
            bool_type, in_range,
            ctx.OpINotEqual(bool_type, ctx.OpBitwiseAnd(u32_type, flags, ctx.ConstU32(bit)),
                            ctx.u32_zero_value));
    };
    const Id do_store = flag(MotionVectors::FlagStore);
    const Id do_load = flag(MotionVectors::FlagLoad);
    const Id store_label = ctx.OpLabel();
    const Id store_merge = ctx.OpLabel();
    ctx.OpSelectionMerge(store_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(do_store, store_label, store_merge);
    ctx.AddLabel(store_label);
    const Id store_address = address(MotionVectors::positions_address,
                                     ctx.OpIAdd(u32_type, store_base, slot), 16);
    // Indexed draws may invoke the same vertex more than once. Atomic component stores
    // avoid write/write races; all these invocations produce the same clip position.
    const Id scalar_ptr = ctx.TypePointer(spv::StorageClass::PhysicalStorageBuffer, u32_type);
    const Id scope = ctx.ConstU32(static_cast<u32>(spv::Scope::Device));
    for (u32 i = 0; i < 4; ++i) {
        const Id ptr = ctx.OpConvertUToPtr(scalar_ptr,
            ctx.OpIAdd(ctx.U64, store_address, ctx.Constant(ctx.U64, u64(i * 4))));
        const Id bits = ctx.OpBitcast(u32_type, ctx.OpCompositeExtract(ctx.F32[1], position, i));
        ctx.OpAtomicExchange(u32_type, ptr, scope, ctx.u32_zero_value, bits);
    }
    ctx.OpBranch(store_merge);
    ctx.AddLabel(store_merge);

    const Id load_label = ctx.OpLabel();
    const Id load_merge = ctx.OpLabel();
    ctx.OpSelectionMerge(load_merge, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(do_load, load_label, load_merge);
    ctx.AddLabel(load_label);
    const Id loaded = ctx.OpLoad(
        ctx.F32[4], ctx.OpConvertUToPtr(f32x4_ptr, address(MotionVectors::positions_address,
            ctx.OpIAdd(u32_type, load_base, slot), 16)), spv::MemoryAccessMask::Aligned, 16u);
    ctx.OpBranch(load_merge);
    ctx.AddLabel(load_merge);
    Id previous = ctx.OpPhi(ctx.F32[4], position, store_merge, loaded, load_label);
    const Id valid = ctx.OpSelect(ctx.F32[1], do_load, ctx.Constant(ctx.F32[1], 1.0f),
                                  ctx.Constant(ctx.F32[1], 0.0f));
    previous = ctx.OpCompositeInsert(ctx.F32[4], valid, previous, 2u);
    ctx.OpStore(ctx.motion_out_cur, position);
    ctx.OpStore(ctx.motion_out_prev, previous);
}

// Screen-space motion (previous minus current, pixels) through the viewport scale; z = valid.
void EmitFragmentMotion(EmitContext& ctx) {
    const Id f32_type = ctx.F32[1];
    const Id current = ctx.OpLoad(ctx.F32[4], ctx.motion_in_cur);
    const Id previous = ctx.OpLoad(ctx.F32[4], ctx.motion_in_prev);
    const auto push = [&](u32 index) {
        return ctx.OpLoad(f32_type, ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant,
                                                                 f32_type),
                                                 ctx.push_data_block, ctx.ConstU32(index)));
    };
    const Id xscale = push(PushData::XScaleIndex);
    const Id yscale = push(PushData::YScaleIndex);
    const auto ndc = [&](Id clip, u32 component) {
        return ctx.OpFDiv(f32_type, ctx.OpCompositeExtract(f32_type, clip, component),
                          ctx.OpCompositeExtract(f32_type, clip, 3u));
    };
    const Id dx = ctx.OpFMul(f32_type, ctx.OpFSub(f32_type, ndc(previous, 0u), ndc(current, 0u)), xscale);
    const Id dy = ctx.OpFMul(f32_type, ctx.OpFSub(f32_type, ndc(previous, 1u), ndc(current, 1u)), yscale);
    const Id zero = ctx.Constant(f32_type, 0.0f);
    const Id epsilon = ctx.Constant(f32_type, 1e-5f);
    const Id front = ctx.OpLogicalAnd(ctx.U1[1],
        ctx.OpFOrdGreaterThan(ctx.U1[1], ctx.OpCompositeExtract(f32_type, previous, 3u), epsilon),
        ctx.OpFOrdGreaterThan(ctx.U1[1], ctx.OpCompositeExtract(f32_type, current, 3u), epsilon));
    const Id valid = ctx.OpSelect(f32_type, front,
                                  ctx.OpCompositeExtract(f32_type, previous, 2u), zero);
    // Uninstrumented draws can overwrite the final scene depth after this pixel was
    // written. Preserve depth so the compose pass can reject an occluded vector.
    const Id depth = ctx.OpCompositeExtract(f32_type,
                                            ctx.OpLoad(ctx.F32[4], ctx.frag_coord), 2u);
    // R32G32_UINT (8 bytes a pixel, not 16): the vector as two halves; the depth, full
    // precision, with the sign bit set where a vector was written (a cleared target has none).
    const Id vector = ctx.OpCompositeConstruct(ctx.F32[2], ctx.OpSelect(f32_type, front, dx, zero),
                                               ctx.OpSelect(f32_type, front, dy, zero));
    const Id u32_type = ctx.U32[1];
    const Id written = ctx.OpFOrdGreaterThan(ctx.U1[1], valid, ctx.Constant(f32_type, 0.99f));
    const Id depth_bits = ctx.OpBitwiseOr(
        u32_type, ctx.OpBitwiseAnd(u32_type, ctx.OpBitcast(u32_type, depth), ctx.ConstU32(0x7fffffffu)),
        ctx.OpSelect(u32_type, written, ctx.ConstU32(0x80000000u), ctx.ConstU32(0u)));
    ctx.OpStore(ctx.motion_frag_out,
                ctx.OpCompositeConstruct(ctx.U32[2], ctx.OpPackHalf2x16(u32_type, vector),
                                         depth_bits));
}

// The game's UI on an output of another aspect ratio (UiComposition::EdgeParam): with
// motion_param set, a vertex in a corner of the game's 16:9 screen or past it (on or beyond its
// edges in both x and y: normalized edges, 16 bits each) goes to that corner of the output, so
// that full-screen fills and backgrounds (the game draws some a little larger than its screen)
// cover the output; and (subtitles' backdrops, lower screen) a vertex on a side edge. What the
// HUD moved past one edge only stays where it is.
void EmitUiEdgeStretch(EmitContext& ctx) {
    const Id f32 = ctx.F32[1];
    const Id u32 = ctx.U32[1];
    const Id b1 = ctx.U1[1];
    const Id param = ctx.OpLoad(u32, ctx.OpAccessChain(
        ctx.TypePointer(spv::StorageClass::PushConstant, u32), ctx.push_data_block,
        ctx.ConstU32(PushData::MotionParamIndex)));
    const Id on = ctx.OpINotEqual(b1, param, ctx.u32_zero_value);
    const Id unit = ctx.Constant(f32, 1.0f / 65535.0f);
    const Id ex = ctx.OpFMul(f32, ctx.OpConvertUToF(f32, ctx.OpShiftRightLogical(u32, param, ctx.ConstU32(16u))), unit);
    const Id ey = ctx.OpFMul(f32, ctx.OpConvertUToF(f32, ctx.OpShiftRightLogical(u32,
        ctx.OpBitwiseAnd(u32, param, ctx.ConstU32(65534u)), ctx.ConstU32(1u))), ctx.Constant(f32, 1.0f / 32767.0f));
    const Id y_down = ctx.OpINotEqual(b1, ctx.OpBitwiseAnd(u32, param, ctx.ConstU32(1u)), ctx.u32_zero_value);
    const Id position = ctx.OpLoad(ctx.F32[4], ctx.output_position);
    const Id x = ctx.OpCompositeExtract(f32, position, 0u);
    const Id y = ctx.OpCompositeExtract(f32, position, 1u);
    const Id w = ctx.OpFAbs(f32, ctx.OpCompositeExtract(f32, position, 3u));
    // On the edge (within 0.002: a pixel of the 1080 lines, two of the 1920 columns) or past it.
    const auto out = [&](Id v, Id edge) {
        return ctx.OpFOrdGreaterThanEqual(b1, ctx.OpFAbs(f32, v),
            ctx.OpFMul(f32, w, ctx.OpFSub(f32, edge, ctx.Constant(f32, 0.002f))));
    };
    // On the edge only (up to two pixels past it: Scaleform's anti-aliasing fringe): a band the
    // movie draws from edge to edge of its screen (the subtitles' backdrop) has its inner vertices
    // there. What the port moved to the edges (the HUD's groups) lies further out.
    const auto at = [&](Id v, Id edge) {
        return ctx.OpLogicalAnd(b1, out(v, edge),
            ctx.OpFOrdLessThanEqual(b1, ctx.OpFAbs(f32, v),
                ctx.OpFMul(f32, w, ctx.OpFAdd(f32, edge, ctx.Constant(f32, 0.004f)))));
    };
    // The band's vertices on a side edge only in the lower part of the screen (y past 0.4 of
    // its half height: the subtitles), not anywhere: the HUD's groups moved toward the edges put
    // some of their vertices on the 16:9 screen's edge by chance (a glyph, a frame), which went
    // to the output's edge (white streaks across the HUD at 21:9). Top and bottom edges: corners
    // only, for the same reason (the HUD moved up and down on taller outputs).
    const Id towards_bottom = ctx.OpSelect(f32, y_down, y, ctx.OpFNegate(f32, y));
    const Id lower = ctx.OpFOrdGreaterThan(b1, towards_bottom, ctx.OpFMul(f32, w, ctx.Constant(f32, 0.4f)));
    const Id corner = ctx.OpLogicalAnd(b1, out(x, ex), out(y, ey));
    const Id snap_x = ctx.OpLogicalAnd(b1, on, ctx.OpLogicalOr(b1, corner,
                                                               ctx.OpLogicalAnd(b1, lower, at(x, ex))));
    const Id snap_y = ctx.OpLogicalAnd(b1, on, corner);
    // v * |w| / |v|: the output's edge on v's side.
    const auto snap = [&](Id v, Id when) {
        return ctx.OpSelect(f32, when, ctx.OpFDiv(f32, ctx.OpFMul(f32, v, w), ctx.OpFAbs(f32, v)), v);
    };
    ctx.OpStore(ctx.output_position, ctx.OpCompositeInsert(ctx.F32[4], snap(y, snap_y),
        ctx.OpCompositeInsert(ctx.F32[4], snap(x, snap_x), position, 0u), 1u));
}

} // namespace

void EmitEpilogue(EmitContext& ctx) {
    if (Sirit::ValidId(ctx.motion_out_cur)) {
        EmitVertexMotion(ctx);
    }
    if (Sirit::ValidId(ctx.motion_frag_out)) {
        EmitFragmentMotion(ctx);
    }
    if (ctx.hw_stage == HwStage::Vertex &&
        ctx.runtime_info.hw.vs.emulate_depth_negative_one_to_one) {
        ConvertDepthMode(ctx);
    }
    if (ctx.hw_stage == HwStage::Vertex && ctx.runtime_info.hw.vs.clip_disable) {
        ConvertPositionToClipSpace(ctx);
    } else if (ctx.hw_stage == HwStage::Vertex && !Sirit::ValidId(ctx.motion_out_cur)) {
        // Any vertex shader can draw the UI (Scaleform's, the HUD's gauges and icons); outside
        // the UI motion_param is 0 and this does nothing.
        EmitUiEdgeStretch(ctx);
    }
}

/// Discard demotes to a helper invocation (derivatives stay defined).
void EmitDiscard(EmitContext& ctx) {
    ctx.OpDemoteToHelperInvocationEXT();
}

void EmitDiscardCond(EmitContext& ctx, Id condition) {
    const Id kill_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, kill_label, merge_label);
    ctx.AddLabel(kill_label);
    ctx.OpDemoteToHelperInvocationEXT();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitEmitVertex(EmitContext& ctx) {
    ctx.OpEmitVertex();
}

void EmitEmitPrimitive(EmitContext& ctx) {
    ctx.OpEndPrimitive();
}

void EmitEmitVertex(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitDebugPrint(EmitContext& ctx, IR::Inst* inst, Id fmt, Id arg0, Id arg1, Id arg2, Id arg3) {
    IR::DebugPrintFlags flags = inst->Flags<IR::DebugPrintFlags>();
    std::array<Id, IR::DEBUGPRINT_NUM_FORMAT_ARGS> fmt_args = {arg0, arg1, arg2, arg3};
    auto fmt_args_span = std::span<Id>(fmt_args.begin(), fmt_args.begin() + flags.num_args);
    ctx.OpDebugPrintf(fmt, fmt_args_span);
}

/// s_memtime: the subgroup clock where supported, otherwise a constant.
Id EmitMemtime(EmitContext& ctx) {
    if (ctx.profile.supports_shader_subgroup_clock) {
        return ctx.OpReadClockKHR(ctx.U64, ctx.ConstU32(std::to_underlying(spv::Scope::Subgroup)));
    } else {
        return ctx.Constant(ctx.U64, 1U);
    }
}

} // namespace Shader::Backend::SPIRV
