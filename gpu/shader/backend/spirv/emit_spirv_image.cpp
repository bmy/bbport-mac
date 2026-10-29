// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Image sampling, gathers, fetches, storage reads/writes and queries. The handle is the image
// binding (low 16 bits) and the sampler binding (high 16 bits); integer images are read as
// their float bits.
#include <boost/container/static_vector.hpp>

#include "shader/ir/microinstruction.h"
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {

/// The optional operands of an image instruction.
struct ImageOperands {
    void Add(spv::ImageOperandsMask new_mask, Id value) {
        if (!Sirit::ValidId(value)) {
            return;
        }
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value);
    }

    void Add(spv::ImageOperandsMask new_mask, Id value1, Id value2) {
        mask = static_cast<spv::ImageOperandsMask>(static_cast<u32>(mask) |
                                                   static_cast<u32>(new_mask));
        operands.push_back(value1);
        operands.push_back(value2);
    }

    /// Constant offsets as ConstOffset; dynamic ones only where SPIR-V allows them (gathers).
    void AddOffset(EmitContext& ctx, const IR::Value& offset, bool can_use_runtime_offsets = false) {
        if (offset.IsEmpty()) {
            return;
        }
        if (offset.IsImmediate()) {
            Add(spv::ImageOperandsMask::ConstOffset, ctx.ConstS32(s32(offset.U32())));
            return;
        }
        const IR::Inst* const inst{offset.Inst()};
        if (inst->AreAllArgsImmediates()) {
            switch (inst->GetOpcode()) {
            case IR::Opcode::CompositeConstructU32x2:
                Add(spv::ImageOperandsMask::ConstOffset,
                    ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                 static_cast<s32>(inst->Arg(1).U32())));
                return;
            case IR::Opcode::CompositeConstructU32x3:
                Add(spv::ImageOperandsMask::ConstOffset,
                    ctx.ConstS32(static_cast<s32>(inst->Arg(0).U32()),
                                 static_cast<s32>(inst->Arg(1).U32()),
                                 static_cast<s32>(inst->Arg(2).U32())));
                return;
            default:
                break;
            }
        }
        if (can_use_runtime_offsets) {
            Add(spv::ImageOperandsMask::Offset, ctx.Def(offset));
        } else {
            LOG_WARNING(Render_Vulkan,
                        "Runtime offset provided to unsupported image sample instruction");
        }
    }

    void AddDerivatives(Id dx, Id dy) {
        if (Sirit::ValidId(dx) && Sirit::ValidId(dy)) {
            Add(spv::ImageOperandsMask::Grad, dx, dy);
        }
    }

    spv::ImageOperandsMask mask{};
    boost::container::static_vector<Id, 4> operands;
};

/// An image with its sampler, and the result type of `num_components`.
struct SampledImage {
    const EmitContext::TextureDefinition& texture;
    Id result_type;
    Id sampled;
};

SampledImage Sample(EmitContext& ctx, u32 handle, u32 num_components) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id result_type = texture.data_types->Get(num_components);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    return {texture, result_type, sampled};
}

/// Integer texels as float bits.
Id AsFloat(EmitContext& ctx, const EmitContext::TextureDefinition& texture, Id type, Id texel) {
    return texture.is_integer ? ctx.OpBitcast(type, texel) : texel;
}

/// A depth compare result as a vec4 (r = result).
Id DepthResult(EmitContext& ctx, const EmitContext::TextureDefinition& texture, Id sample) {
    const Id value = AsFloat(ctx, texture, ctx.F32[1], sample);
    return ctx.OpCompositeConstruct(ctx.F32[4], value, ctx.f32_zero_value, ctx.f32_zero_value,
                                    ctx.f32_zero_value);
}

} // namespace

Id EmitImageHandle(EmitContext& ctx, Id, Id) {
    UNREACHABLE_MSG("Unreachable instruction");
}

Id EmitImageSampleRaw(EmitContext& ctx, IR::Inst* inst, u32 handle, Id address1, Id address2,
                      Id address3, Id address4) {
    UNREACHABLE_MSG("Unreachable instruction");
}

Id EmitImageSampleImplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id bias,
                              const IR::Value& offset) {
    const SampledImage image = Sample(ctx, handle, 4);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleImplicitLod(image.result_type, image.sampled, coords,
                                                   operands.mask, operands.operands);
    return AsFloat(ctx, image.texture, ctx.F32[4], sample);
}

Id EmitImageSampleExplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod,
                              const IR::Value& offset) {
    const SampledImage image = Sample(ctx, handle, 4);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(image.result_type, image.sampled, coords,
                                                   operands.mask, operands.operands);
    return AsFloat(ctx, image.texture, ctx.F32[4], sample);
}

Id EmitImageSampleDrefImplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id dref,
                                  Id bias, const IR::Value& offset) {
    const SampledImage image = Sample(ctx, handle, 1);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Bias, bias);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleDrefImplicitLod(image.result_type, image.sampled, coords,
                                                       dref, operands.mask, operands.operands);
    return DepthResult(ctx, image.texture, sample);
}

Id EmitImageSampleDrefExplicitLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id dref,
                                  Id lod, const IR::Value& offset) {
    const SampledImage image = Sample(ctx, handle, 1);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Lod, lod);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleDrefExplicitLod(image.result_type, image.sampled, coords,
                                                       dref, operands.mask, operands.operands);
    return DepthResult(ctx, image.texture, sample);
}

Id EmitImageGather(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords,
                   const IR::Value& offset) {
    const SampledImage image = Sample(ctx, handle, 4);
    const u32 comp = inst->Flags<IR::TextureInstInfo>().gather_comp.Value();
    ImageOperands operands;
    operands.AddOffset(ctx, offset, true);
    const Id texels = ctx.OpImageGather(image.result_type, image.sampled, coords,
                                        ctx.ConstU32(comp), operands.mask, operands.operands);
    return AsFloat(ctx, image.texture, ctx.F32[4], texels);
}

Id EmitImageGatherDref(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords,
                       const IR::Value& offset, Id dref) {
    const SampledImage image = Sample(ctx, handle, 4);
    ImageOperands operands;
    operands.AddOffset(ctx, offset, true);
    const Id texels = ctx.OpImageDrefGather(image.result_type, image.sampled, coords, dref,
                                            operands.mask, operands.operands);
    return AsFloat(ctx, image.texture, ctx.F32[4], texels);
}

Id EmitImageGradient(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id derivatives_dx,
                     Id derivatives_dy, const IR::Value& offset, const IR::Value& lod_clamp) {
    const SampledImage image = Sample(ctx, handle, 4);
    ImageOperands operands;
    operands.AddDerivatives(derivatives_dx, derivatives_dy);
    operands.AddOffset(ctx, offset);
    const Id sample = ctx.OpImageSampleExplicitLod(image.result_type, image.sampled, coords,
                                                   operands.mask, operands.operands);
    return AsFloat(ctx, image.texture, ctx.F32[4], sample);
}

/// (width, height, depth/layers, levels) as GCN returns them.
Id EmitImageQueryDimensions(EmitContext& ctx, IR::Inst* inst, u32 handle, Id lod, bool has_mips) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id zero = ctx.u32_zero_value;
    const auto mips{[&] { return has_mips ? ctx.OpImageQueryLevels(ctx.U32[1], image) : zero; }};
    const bool uses_lod{texture.view_type != ::Gcn::ImageType::Color2DMsaa && !texture.is_storage};
    const auto query{[&](Id type) {
        return uses_lod ? ctx.OpImageQuerySizeLod(type, image, lod)
                        : ctx.OpImageQuerySize(type, image);
    }};
    switch (texture.view_type) {
    case ::Gcn::ImageType::Color1D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[1]), zero, zero, mips());
    case ::Gcn::ImageType::Color1DArray:
    case ::Gcn::ImageType::Color2D:
    case ::Gcn::ImageType::Color2DMsaa:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[2]), zero, mips());
    case ::Gcn::ImageType::Color2DArray:
    case ::Gcn::ImageType::Color3D:
        return ctx.OpCompositeConstruct(ctx.U32[4], query(ctx.U32[3]), mips());
    default:
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
}

Id EmitImageQueryLod(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id image = ctx.OpLoad(texture.image_type, texture.id);
    const Id sampler = ctx.OpLoad(ctx.sampler_type, ctx.samplers[handle >> 16]);
    const Id sampled_image = ctx.OpSampledImage(texture.sampled_type, image, sampler);
    return ctx.OpImageQueryLod(ctx.F32[2], sampled_image, coords);
}

/// Texel fetches (sampled images; MSAA sample indices wrap like on GCN) and storage reads.
Id EmitImageRead(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod, Id ms) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    const Id color_type = texture.data_types->Get(4);
    ImageOperands operands;
    Id texel;
    if (!texture.is_storage) {
        const Id image = ctx.OpLoad(texture.image_type, texture.id);
        if (texture.view_type == ::Gcn::ImageType::Color2DMsaa) {
            if (Sirit::ValidId(ms)) {
                const Id sample_count = ctx.OpImageQuerySamples(ctx.U32[1], image);
                const Id wrapped_ms = ctx.OpUMod(ctx.U32[1], ms, sample_count);
                operands.Add(spv::ImageOperandsMask::Sample, wrapped_ms);
            }
        } else {
            if (Sirit::ValidId(ms)) {
                LOG_ERROR(Render_Recompiler, "image is not MS but ms operand is provided");
            }
            operands.Add(spv::ImageOperandsMask::Lod, lod);
            operands.Add(spv::ImageOperandsMask::Sample, ms);
        }
        texel = ctx.OpImageFetch(color_type, image, coords, operands.mask, operands.operands);
    } else {
        if (ctx.profile.supports_image_load_store_lod) {
            operands.Add(spv::ImageOperandsMask::Lod, lod);
        } else if (Sirit::ValidId(lod)) {
            // (image_load_mip of sampled images is a fetch; a storage read with a LOD has not
            // been seen.)
            UNREACHABLE_MSG("Unsupported ImageRead with Lod");
        }
        const Id image = ctx.OpLoad(texture.image_type, texture.id);
        texel = ctx.OpImageRead(color_type, image, coords, operands.mask, operands.operands);
    }
    return AsFloat(ctx, texture, ctx.F32[4], texel);
}

/// Storage writes; without host LOD support the level is an index into the bound views.
void EmitImageWrite(EmitContext& ctx, IR::Inst* inst, u32 handle, Id coords, Id lod, Id ms,
                    Id color) {
    const auto& texture = ctx.images[handle & 0xFFFF];
    Id image_ptr = texture.id;
    const Id color_type = texture.data_types->Get(4);
    ImageOperands operands;
    operands.Add(spv::ImageOperandsMask::Sample, ms);
    if (ctx.profile.supports_image_load_store_lod) {
        operands.Add(spv::ImageOperandsMask::Lod, lod);
    } else if (Sirit::ValidId(lod)) {
        LOG_WARNING(Render, "Fallback for ImageWrite with LOD");
        ASSERT(texture.mip_fallback_mode == MipStorageFallbackMode::DynamicIndex);
        const Id single_image_ptr_type =
            ctx.TypePointer(spv::StorageClass::UniformConstant, texture.image_type);
        image_ptr = ctx.OpAccessChain(single_image_ptr_type, image_ptr, std::array{lod});
    }
    const Id image = ctx.OpLoad(texture.image_type, image_ptr);
    const Id texel = AsFloat(ctx, texture, color_type, color);
    ctx.OpImageWrite(image, coords, texel, operands.mask, operands.operands);
}

Id EmitCubeFaceIndex(EmitContext& ctx, IR::Inst* inst, Id cube_coords) {
    if (!ctx.profile.supports_native_cube_calc) {
        UNREACHABLE_MSG("SPIR-V Instruction");
    }
    return ctx.OpCubeFaceIndexAMD(ctx.F32[1], cube_coords);
}

} // namespace Shader::Backend::SPIRV
