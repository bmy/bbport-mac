// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Reads and writes of the shader's interface: user data (push constants), the flattened user
// data, constant buffers, input/output attributes and builtins, tessellation attributes and
// patches, and buffer loads/stores (bound buffers, or the page table for paged ones).
#include <bit>

#include <magic_enum/magic_enum.hpp>

#include "common/assert.h"
#include "core/emulator_settings.h"
#include "shader/ir/attribute.h"
#include "shader/ir/microinstruction.h"
#include "shader/ir/patch.h"
#include "shader/runtime_info.h"
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

using PointerType = EmitContext::PointerType;
using PointerSize = EmitContext::PointerSize;

/// The scalar type of an output attribute, and whether stores bitcast to it.
static std::pair<Id, bool> OutputAttrComponentType(EmitContext& ctx, IR::Attribute attr) {
    if (IR::IsParam(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::Param0)};
        const auto& info{ctx.output_params.at(index)};
        return {info.component_type, info.is_integer};
    }
    if (IR::IsMrt(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::RenderTarget0)};
        const auto& info{ctx.frag_outputs.at(index)};
        return {info.component_type, info.is_integer};
    }
    switch (attr) {
    case IR::Attribute::Position0:
    case IR::Attribute::ClipDistance:
    case IR::Attribute::CullDistance:
    case IR::Attribute::Depth:
    case IR::Attribute::PointSize:
        return {ctx.F32[1], false};
    case IR::Attribute::RenderTargetIndex:
    case IR::Attribute::ViewportIndex:
    case IR::Attribute::SampleMask:
        return {ctx.U32[1], true};
    case IR::Attribute::StencilRef:
        return {ctx.S32[1], true};
    default:
        UNREACHABLE_MSG("Write attribute {}", attr);
    }
}

/// The user data registers the shader reads are packed into the push data (uvec4s).
Id EmitGetUserData(EmitContext& ctx, IR::ScalarReg reg) {
    const u32 index = ctx.binding.user_data + ctx.info.ud_mask.Index(reg);
    const u32 half = PushData::UdRegsIndex + (index >> 2);
    const Id ud_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1]),
                                      ctx.push_data_block, ctx.ConstU32(half),
                                      ctx.ConstU32(index & 3))};
    const Id ud_reg{ctx.OpLoad(ctx.U32[1], ud_ptr)};
    ctx.Name(ud_reg, fmt::format("ud_{}", u32(reg)));
    return ud_reg;
}

/// Flattened reads come from the flat buffer; without a slot (and DMA enabled) a dynamic read
/// through the page table.
Id EmitReadConst(EmitContext& ctx, IR::Inst* inst, Id addr, Id offset) {
    const u32 flatbuf_off_dw = inst->Flags<u32>();
    if (!EmulatorSettings.IsDirectMemoryAccessEnabled()) {
        return ctx.EmitFlatbufferLoad(ctx.ConstU32(flatbuf_off_dw));
    }
    if (flatbuf_off_dw == 0) {
        return ctx.OpFunctionCall(ctx.U32[1], ctx.read_const_dynamic, addr, offset);
    } else {
        return ctx.EmitFlatbufferLoad(ctx.ConstU32(flatbuf_off_dw));
    }
}

Id EmitReadConstBuffer(EmitContext& ctx, u32 handle, Id index) {
    const auto& buffer = ctx.buffers[handle];
    if (const Id offset = buffer.Offset(PointerSize::B32); Sirit::ValidId(offset)) {
        index = ctx.OpIAdd(ctx.U32[1], index, offset);
    }
    const auto [id, pointer_type] = buffer.Alias(PointerType::U32);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index)};
    const Id result{ctx.OpLoad(ctx.U32[1], ptr)};
    return result;
}

Id EmitGetAttribute(EmitContext& ctx, IR::Attribute attr, u32 comp, u32 index) {
    if (IR::IsParam(attr)) {
        const u32 param_index{u32(attr) - u32(IR::Attribute::Param0)};
        const auto& param{ctx.input_params.at(param_index)};
        const Id value = [&] {
            if (param.is_array) {
                ASSERT(param.num_components > 1);
                if (param.is_loaded) {
                    return ctx.OpCompositeExtract(param.component_type, param.id_array[index],
                                                  comp);
                } else {
                    return ctx.OpLoad(param.component_type,
                                      ctx.OpAccessChain(param.pointer_type, param.id,
                                                        ctx.ConstU32(index), ctx.ConstU32(comp)));
                }
            } else {
                ASSERT(!param.is_loaded);
                if (param.num_components > 1) {
                    return ctx.OpLoad(
                        param.component_type,
                        ctx.OpAccessChain(param.pointer_type, param.id, ctx.ConstU32(comp)));
                } else {
                    return ctx.OpLoad(param.component_type, param.id);
                }
            }
        }();
        return param.is_integer ? ctx.OpBitcast(ctx.F32[1], value) : value;
    }
    if (IR::IsBarycentricCoord(attr) && attr != IR::Attribute::BaryCoordPullModel &&
        ctx.profile.supports_fragment_shader_barycentric) {
        ++comp;
    }
    switch (attr) {
    case IR::Attribute::Position0:
        ASSERT(ctx.sw_stage == SwStage::Geometry);
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.gl_in, ctx.ConstU32(index),
                                            ctx.ConstU32(0U), ctx.ConstU32(comp)));
    case IR::Attribute::FragCoord: {
        const Id value = ctx.OpLoad(ctx.F32[1],
            ctx.OpAccessChain(ctx.input_f32, ctx.frag_coord, ctx.ConstU32(comp)));
        if (comp >= 2) return value;
        // Shaders address native-size textures: on a reduced framebuffer, pixel coordinates
        // are scaled back to the native size.
        const Id packed = ctx.OpLoad(ctx.U32[1], ctx.OpAccessChain(
            ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1]),
            ctx.push_data_block, ctx.ConstU32(PushData::SceneSizeIndex)));
        const Id dimension = comp == 0
            ? ctx.OpBitwiseAnd(ctx.U32[1], packed, ctx.ConstU32(65535u))
            : ctx.OpShiftRightLogical(ctx.U32[1], packed, ctx.ConstU32(16u));
        const Id native = ctx.ConstU32(comp == 0 ? 1920u : 1080u);
        const Id safe = ctx.OpSelect(ctx.U32[1],
            ctx.OpIEqual(ctx.U1[1], dimension, ctx.u32_zero_value), native, dimension);
        return ctx.OpFMul(ctx.F32[1], value, ctx.OpFDiv(ctx.F32[1],
            ctx.OpConvertUToF(ctx.F32[1], native), ctx.OpConvertUToF(ctx.F32[1], safe)));
    }
    case IR::Attribute::TessellationEvaluationPointU:
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.tess_coord, ctx.u32_zero_value));
    case IR::Attribute::TessellationEvaluationPointV:
        return ctx.OpLoad(ctx.F32[1],
                          ctx.OpAccessChain(ctx.input_f32, ctx.tess_coord, ctx.ConstU32(1U)));
    case IR::Attribute::BaryCoordSmooth:
        if (ctx.profile.supports_amd_shader_explicit_vertex_parameter) {
            return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth,
                                                            ctx.ConstU32(comp)));
        } else {
            return ctx.OpCompositeExtract(ctx.F32[1], ctx.OpLoad(ctx.F32[3], ctx.bary_coord), comp);
        }
    case IR::Attribute::BaryCoordSmoothCentroid:
        if (ctx.profile.supports_amd_shader_explicit_vertex_parameter) {
            return ctx.OpLoad(ctx.F32[1],
                              ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth_centroid,
                                                ctx.ConstU32(comp)));
        } else {
            return ctx.OpCompositeExtract(
                ctx.F32[1], ctx.OpInterpolateAtCentroid(ctx.F32[3], ctx.bary_coord), comp);
        }
    case IR::Attribute::BaryCoordSmoothSample:
        if (ctx.profile.supports_amd_shader_explicit_vertex_parameter) {
            return ctx.OpLoad(
                ctx.F32[1],
                ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_smooth_sample, ctx.ConstU32(comp)));
        } else {
            return ctx.OpCompositeExtract(
                ctx.F32[1],
                ctx.OpInterpolateAtSample(ctx.F32[3], ctx.bary_coord,
                                          ctx.OpLoad(ctx.U32[1], ctx.sample_index)),
                comp);
        }
    case IR::Attribute::BaryCoordNoPersp:
        return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_nopersp,
                                                        ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordNoPerspSample:
        return ctx.OpLoad(
            ctx.F32[1],
            ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_nopersp_sample, ctx.ConstU32(comp)));
    case IR::Attribute::BaryCoordPullModel: {
        ASSERT_MSG(comp < 3, "Invalid BaryCoordPullModel component {}", comp);

        if (ctx.profile.supports_amd_shader_explicit_vertex_parameter) {
            return ctx.OpLoad(
                ctx.F32[1],
                ctx.OpAccessChain(ctx.input_f32, ctx.bary_coord_pull_model, ctx.ConstU32(comp)));
        }

        // Pull model: I/W, J/W and 1/W. FragCoord.w is 1/W, and the perspective barycentrics
        // times it give the first two.
        const Id inv_w = ctx.OpLoad(
            ctx.F32[1], ctx.OpAccessChain(ctx.input_f32, ctx.frag_coord, ctx.ConstU32(3U)));
        if (comp == 2) {
            return inv_w;
        }
        const Id ij =
            ctx.OpCompositeExtract(ctx.F32[1], ctx.OpLoad(ctx.F32[3], ctx.bary_coord), comp + 1);
        return ctx.OpFMul(ctx.F32[1], ij, inv_w);
    }
    default:
        UNREACHABLE_MSG("Read attribute {}", attr);
    }
}

Id EmitGetAttributeU1(EmitContext& ctx, IR::Attribute attr, u32 comp) {
    ASSERT(comp == 0);
    switch (attr) {
    case IR::Attribute::IsFrontFace:
        return ctx.OpLoad(ctx.U1[1], ctx.front_facing);
    case IR::Attribute::IsHelperInvocation:
        return ctx.OpLoad(ctx.U1[1], ctx.helper_invocation);
    default:
        UNREACHABLE_MSG("Unsupported U1 attribute {}", attr);
    }
}

Id EmitGetAttributeU32(EmitContext& ctx, IR::Attribute attr, u32 comp) {
    switch (attr) {
    case IR::Attribute::VertexId:
        return ctx.OpLoad(ctx.U32[1], ctx.vertex_index);
    case IR::Attribute::InstanceId:
        return ctx.OpLoad(ctx.U32[1], ctx.instance_id);
    case IR::Attribute::BaseVertex:
        return ctx.OpLoad(ctx.U32[1], ctx.base_vertex);
    case IR::Attribute::BaseInstance:
        return ctx.OpLoad(ctx.U32[1], ctx.base_instance);
    case IR::Attribute::WorkgroupIndex:
        return ctx.workgroup_index_id;
    case IR::Attribute::WorkgroupId:
        return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.workgroup_id), comp);
    case IR::Attribute::LocalInvocationId:
        return ctx.OpCompositeExtract(ctx.U32[1], ctx.OpLoad(ctx.U32[3], ctx.local_invocation_id),
                                      comp);
    case IR::Attribute::LocalInvocationIndex:
        return ctx.OpLoad(ctx.U32[1], ctx.local_invocation_index);
    case IR::Attribute::SampleIndex:
        return ctx.OpLoad(ctx.U32[1], ctx.sample_index);
    case IR::Attribute::SampleMask:
        return ctx.OpLoad(ctx.U32[1],
                          ctx.OpAccessChain(ctx.input_u32, ctx.sample_mask_in, ctx.u32_zero_value));
    case IR::Attribute::RenderTargetIndex:
        return ctx.OpLoad(ctx.U32[1], ctx.output_layer);
    case IR::Attribute::PrimitiveId:
        return ctx.OpLoad(ctx.U32[1], ctx.primitive_id);
    case IR::Attribute::InvocationId:
        ASSERT(ctx.info.sw_stage == SwStage::Geometry ||
               ctx.info.sw_stage == SwStage::TessellationControl);
        return ctx.OpLoad(ctx.U32[1], ctx.invocation_id);
    case IR::Attribute::SubgroupLtMask:
        return ctx.OpLoad(
            ctx.U32[1], ctx.OpAccessChain(ctx.input_u32, ctx.subgroup_lt_mask, ctx.ConstU32(comp)));
    case IR::Attribute::PatchVertices:
        ASSERT(ctx.info.sw_stage == SwStage::TessellationControl);
        return ctx.OpLoad(ctx.U32[1], ctx.patch_vertices);
    case IR::Attribute::PackedHullInvocationInfo: {
        ASSERT(ctx.info.sw_stage == SwStage::TessellationControl);
        // Patch id in the VGT group [7:0] (0 here: addresses are within the patch), output
        // control point [12:8]. A passthrough shader runs once per patch on GCN but once per
        // control point here: all of them see 0.
        if (ctx.runtime_info.sw.tcs.IsPassthrough()) {
            return ctx.u32_zero_value;
        } else {
            const Id invocation_id = ctx.OpLoad(ctx.U32[1], ctx.invocation_id);
            return ctx.OpShiftLeftLogical(ctx.U32[1], invocation_id, ctx.ConstU32(8u));
        }
    }
    default:
        UNREACHABLE_MSG("Read U32 attribute {}", attr);
    }
}

void EmitSetAttribute(EmitContext& ctx, IR::Attribute attr, Id value, u32 element) {
    const auto op_store = [&](Id pointer) {
        const auto [component_type, is_integer] = OutputAttrComponentType(ctx, attr);
        if (is_integer) {
            ctx.OpStore(pointer, ctx.OpBitcast(component_type, value));
        } else {
            ctx.OpStore(pointer, value);
        }
    };
    if (IR::IsParam(attr)) {
        const u32 attr_index{u32(attr) - u32(IR::Attribute::Param0)};
        if (ctx.hw_stage == HwStage::Local) {
            const auto component_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
            return op_store(ctx.OpAccessChain(component_ptr, ctx.output_attr_array,
                                              ctx.ConstU32(attr_index), ctx.ConstU32(element)));
        } else {
            const auto& info{ctx.output_params.at(attr_index)};
            ASSERT(info.num_components > 0);
            if (info.num_components == 1) {
                return op_store(info.id);
            } else {
                return op_store(
                    ctx.OpAccessChain(info.pointer_type, info.id, ctx.ConstU32(element)));
            }
        }
    }
    if (IR::IsMrt(attr)) {
        const u32 index{u32(attr) - u32(IR::Attribute::RenderTarget0)};
        const auto& info{ctx.frag_outputs.at(index)};
        if (element < 3 && ctx.runtime_info.hw.fs.color_buffers[index].blend_self_scale) {
            // GCN's factor-scaled min/max blend: min/max(src*src, dst*dst).
            value = ctx.OpFMul(ctx.F32[1], value, value);
        }
        if (info.num_components == 1) {
            return op_store(info.id);
        } else {
            return op_store(ctx.OpAccessChain(info.pointer_type, info.id, ctx.ConstU32(element)));
        }
    }
    switch (attr) {
    case IR::Attribute::Position0:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.output_position, ctx.ConstU32(element)));
    case IR::Attribute::ClipDistance:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.clip_distances, ctx.ConstU32(element)));
    case IR::Attribute::CullDistance:
        return op_store(
            ctx.OpAccessChain(ctx.output_f32, ctx.cull_distances, ctx.ConstU32(element)));
    case IR::Attribute::PointSize:
        return op_store(ctx.output_point_size);
    case IR::Attribute::RenderTargetIndex:
        return op_store(ctx.output_layer);
    case IR::Attribute::ViewportIndex:
        return op_store(ctx.output_viewport_index);
    case IR::Attribute::Depth:
        return op_store(ctx.frag_depth);
    case IR::Attribute::SampleMask:
        return op_store(ctx.OpAccessChain(ctx.output_u32, ctx.sample_mask, ctx.u32_zero_value));
    case IR::Attribute::StencilRef:
        if (ctx.profile.supports_shader_stencil_export) {
            return op_store(ctx.stencil_ref);
        }
        return;
    default:
        UNREACHABLE_MSG("Write attribute {}", attr);
    }
}

Id EmitGetTessGenericAttribute(EmitContext& ctx, Id vertex_index, Id attr_index, Id comp_index) {
    const auto attr_comp_ptr = ctx.TypePointer(spv::StorageClass::Input, ctx.F32[1]);
    return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(attr_comp_ptr, ctx.input_attr_array,
                                                    vertex_index, attr_index, comp_index));
}

Id EmitReadTcsGenericOuputAttribute(EmitContext& ctx, Id vertex_index, Id attr_index,
                                    Id comp_index) {
    const auto attr_comp_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
    return ctx.OpLoad(ctx.F32[1], ctx.OpAccessChain(attr_comp_ptr, ctx.output_attr_array,
                                                    vertex_index, attr_index, comp_index));
}

void EmitSetTcsGenericAttribute(EmitContext& ctx, Id value, Id attr_index, Id comp_index) {
    // The control point is the invocation's.
    const auto component_ptr = ctx.TypePointer(spv::StorageClass::Output, ctx.F32[1]);
    Id pointer =
        ctx.OpAccessChain(component_ptr, ctx.output_attr_array,
                          ctx.OpLoad(ctx.U32[1], ctx.invocation_id), attr_index, comp_index);
    ctx.OpStore(pointer, value);
}

Id EmitGetPatch(EmitContext& ctx, IR::Patch patch) {
    const u32 index{IR::GenericPatchIndex(patch)};
    const Id element{ctx.ConstU32(IR::GenericPatchElement(patch))};
    const Id type{ctx.sw_stage == SwStage::TessellationControl ? ctx.output_f32 : ctx.input_f32};
    const Id pointer{ctx.OpAccessChain(type, ctx.patches.at(index), element)};
    return ctx.OpLoad(ctx.F32[1], pointer);
}

void EmitSetPatch(EmitContext& ctx, IR::Patch patch, Id value) {
    const Id pointer{[&] {
        if (IR::IsGeneric(patch)) {
            const u32 index{IR::GenericPatchIndex(patch)};
            const Id element{ctx.ConstU32(IR::GenericPatchElement(patch))};
            return ctx.OpAccessChain(ctx.output_f32, ctx.patches.at(index), element);
        }
        switch (patch) {
        case IR::Patch::TessellationLodLeft:
        case IR::Patch::TessellationLodRight:
        case IR::Patch::TessellationLodTop:
        case IR::Patch::TessellationLodBottom: {
            const u32 index{static_cast<u32>(patch) - u32(IR::Patch::TessellationLodLeft)};
            const Id index_id{ctx.ConstU32(index)};
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_outer, index_id);
        }
        case IR::Patch::TessellationLodInteriorU:
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_inner,
                                     ctx.u32_zero_value);
        case IR::Patch::TessellationLodInteriorV:
            return ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_inner, ctx.ConstU32(1u));
        default:
            UNREACHABLE_MSG("Patch {}", u32(patch));
        }
    }()};
    ctx.OpStore(pointer, value);
}

/// The buffer index with the binding's offset below the alignment (push data) added.
static Id WithBufferOffset(EmitContext& ctx, const EmitContext::BufferDefinition& buffer,
                           PointerSize size, Id address) {
    if (const Id offset = buffer.Offset(size); Sirit::ValidId(offset)) {
        address = ctx.OpIAdd(ctx.U32[1], address, offset);
    }
    return address;
}

// BB_LAYER_MEMORY: accesses of paged buffers (EmitContext::PagedPointer). `index` is in
// units of the access (as for the bound buffers); zero is read where the guard fails.
static Id PagedLoad(EmitContext& ctx, const EmitContext::BufferDefinition& buffer, Id index,
                    u32 bytes, Id type, Id zero) {
    const Id byte_offset =
        ctx.OpShiftLeftLogical(ctx.U32[1], index, ctx.ConstU32(u32(std::countr_zero(bytes))));
    const auto access = ctx.PagedPointer(buffer, byte_offset, bytes, type);
    const Id value = ctx.OpLoad(type, access.pointer, spv::MemoryAccessMask::Aligned, bytes);
    return ctx.OpSelect(type, access.guard, value, zero);
}

/// The game's buffer copy shader: its destinations include data the CPU reads, so the guest
/// bytes are written as well as the VRAM view (other shaders keep the mirror bookkeeping).
static bool IsBufferCopyShader(const EmitContext& ctx) {
    return ctx.profile.buffer_copy_shader_hash != 0 &&
           ctx.info.pgm_hash == ctx.profile.buffer_copy_shader_hash;
}

static void PagedStore(EmitContext& ctx, const EmitContext::BufferDefinition& buffer, Id index,
                       u32 bytes, Id type, Id value) {
    const Id byte_offset =
        ctx.OpShiftLeftLogical(ctx.U32[1], index, ctx.ConstU32(u32(std::countr_zero(bytes))));
    const auto access = ctx.PagedPointer(buffer, byte_offset, bytes, type);
    ctx.OpStore(access.pointer, value, spv::MemoryAccessMask::Aligned, bytes);
    if (IsBufferCopyShader(ctx)) {
        const auto guest = ctx.PagedPointer(buffer, byte_offset, bytes, type, true);
        ctx.OpStore(guest.pointer, value, spv::MemoryAccessMask::Aligned, bytes);
    }
}

/// 32-bit loads of N consecutive dwords.
template <u32 N, PointerType alias>
static Id EmitLoadBufferB32xN(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    const auto& spv_buffer = ctx.buffers[handle];
    address = WithBufferOffset(ctx, spv_buffer, PointerSize::B32, address);
    const auto& data_types = alias == PointerType::U32 ? ctx.U32 : ctx.F32;
    const auto [id, pointer_type] = spv_buffer.Alias(alias);

    boost::container::static_vector<Id, N> ids;
    if (spv_buffer.paged) {
        // One page table lookup for the dwords of a page (PagedDwords).
        const Id zero = alias == PointerType::U32 ? ctx.u32_zero_value : ctx.f32_zero_value;
        const Id byte_offset = ctx.OpShiftLeftLogical(ctx.U32[1], address, ctx.ConstU32(2U));
        for (const auto& access : ctx.PagedDwords(spv_buffer, byte_offset, N, data_types[1])) {
            const Id value =
                ctx.OpLoad(data_types[1], access.pointer, spv::MemoryAccessMask::Aligned, 4U);
            ids.push_back(ctx.OpSelect(data_types[1], access.guard, value, zero));
        }
        return N == 1 ? ids[0] : ctx.OpCompositeConstruct(data_types[N], ids);
    }
    for (u32 i = 0; i < N; i++) {
        const Id index_i = i == 0 ? address : ctx.OpIAdd(ctx.U32[1], address, ctx.ConstU32(i));
        const Id ptr_i = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index_i);
        const Id result_i = ctx.OpLoad(data_types[1], ptr_i);
        ids.push_back(result_i);
    }

    const Id result = N == 1 ? ids[0] : ctx.OpCompositeConstruct(data_types[N], ids);
    return result;
}

/// An 8/16/64-bit load; `zero` makes the value of failed paged reads, `first_index` is the
/// buffer struct's member index constant.
template <typename MakeZero>
static Id LoadScalar(EmitContext& ctx, u32 handle, Id address, PointerSize size, PointerType alias,
                     u32 bytes, Id type, MakeZero&& zero, Id first_index) {
    const auto& buffer = ctx.buffers[handle];
    address = WithBufferOffset(ctx, buffer, size, address);
    if (buffer.paged) {
        return PagedLoad(ctx, buffer, address, bytes, type, zero());
    }
    const auto [id, pointer_type] = buffer.Alias(alias);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, first_index, address)};
    return ctx.OpLoad(type, ptr);
}

static void StoreScalar(EmitContext& ctx, u32 handle, Id address, PointerSize size,
                        PointerType alias, u32 bytes, Id type, Id value, Id first_index) {
    const auto& buffer = ctx.buffers[handle];
    address = WithBufferOffset(ctx, buffer, size, address);
    if (buffer.paged) {
        PagedStore(ctx, buffer, address, bytes, type, value);
        return;
    }
    const auto [id, pointer_type] = buffer.Alias(alias);
    const Id ptr{ctx.OpAccessChain(pointer_type, id, first_index, address)};
    ctx.OpStore(ptr, value);
}

Id EmitLoadBufferU8(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return LoadScalar(ctx, handle, address, PointerSize::B8, PointerType::U8, 1, ctx.U8,
                      [&] { return ctx.Constant(ctx.U8, 0U); }, ctx.u32_zero_value);
}

Id EmitLoadBufferU16(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return LoadScalar(ctx, handle, address, PointerSize::B16, PointerType::U16, 2, ctx.U16,
                      [&] { return ctx.Constant(ctx.U16, 0U); }, ctx.u32_zero_value);
}

Id EmitLoadBufferU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<1, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<2, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<3, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<4, PointerType::U32>(ctx, inst, handle, address);
}

Id EmitLoadBufferU64(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return LoadScalar(ctx, handle, address, PointerSize::B64, PointerType::U64, 8, ctx.U64,
                      [&] { return ctx.u64_zero_value; }, ctx.u64_zero_value);
}

Id EmitLoadBufferF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<1, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<2, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<3, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferF32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    return EmitLoadBufferB32xN<4, PointerType::F32>(ctx, inst, handle, address);
}

Id EmitLoadBufferFormatF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address) {
    UNREACHABLE_MSG("SPIR-V instruction");
}

template <u32 N, PointerType alias>
static void EmitStoreBufferB32xN(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address,
                                 Id value) {
    const auto& spv_buffer = ctx.buffers[handle];
    address = WithBufferOffset(ctx, spv_buffer, PointerSize::B32, address);
    const auto& data_types = alias == PointerType::U32 ? ctx.U32 : ctx.F32;
    const auto [id, pointer_type] = spv_buffer.Alias(alias);

    if (spv_buffer.paged) {
        // One page table lookup for the dwords of a page (PagedDwords), the VRAM view's and
        // (the game's buffer copy shader) the guest bytes'.
        const Id byte_offset = ctx.OpShiftLeftLogical(ctx.U32[1], address, ctx.ConstU32(2U));
        const auto value_at = [&](u32 i) {
            return N == 1 ? value : ctx.OpCompositeExtract(data_types[1], value, i);
        };
        const auto store_all = [&](bool guest_write) {
            const auto accesses =
                ctx.PagedDwords(spv_buffer, byte_offset, N, data_types[1], guest_write);
            for (u32 i = 0; i < N; i++) {
                ctx.OpStore(accesses[i].pointer, value_at(i), spv::MemoryAccessMask::Aligned, 4U);
            }
        };
        store_all(false);
        if (IsBufferCopyShader(ctx)) {
            store_all(true);
        }
        return;
    }
    for (u32 i = 0; i < N; i++) {
        const Id index_i = i == 0 ? address : ctx.OpIAdd(ctx.U32[1], address, ctx.ConstU32(i));
        const Id value_i = N == 1 ? value : ctx.OpCompositeExtract(data_types[1], value, i);
        const Id ptr_i = ctx.OpAccessChain(pointer_type, id, ctx.u32_zero_value, index_i);
        ctx.OpStore(ptr_i, value_i);
    }
}

void EmitStoreBufferU8(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    StoreScalar(ctx, handle, address, PointerSize::B8, PointerType::U8, 1, ctx.U8, value,
                ctx.u32_zero_value);
}

void EmitStoreBufferU16(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    StoreScalar(ctx, handle, address, PointerSize::B16, PointerType::U16, 2, ctx.U16, value,
                ctx.u32_zero_value);
}

void EmitStoreBufferU32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<1, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<2, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<3, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<4, PointerType::U32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferU64(EmitContext& ctx, IR::Inst*, u32 handle, Id address, Id value) {
    StoreScalar(ctx, handle, address, PointerSize::B64, PointerType::U64, 8, ctx.U64, value,
                ctx.u64_zero_value);
}

void EmitStoreBufferF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<1, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x2(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<2, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x3(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<3, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferF32x4(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    EmitStoreBufferB32xN<4, PointerType::F32>(ctx, inst, handle, address, value);
}

void EmitStoreBufferFormatF32(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address, Id value) {
    UNREACHABLE_MSG("SPIR-V instruction");
}

// Register accesses are gone after SSA construction.

void EmitGetScalarRegister(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetScalarRegister(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetVectorRegister(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetVectorRegister(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetVirtualRegister(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetVirtualRegister(EmitContext& ctx) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitSetGotoVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

void EmitGetGotoVariable(EmitContext&) {
    UNREACHABLE_MSG("Unreachable instruction");
}

} // namespace Shader::Backend::SPIRV
