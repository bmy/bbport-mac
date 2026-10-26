// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <atomic>
#include <utility>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "common/serdes.h"
#include "render/gcn_to_vk.h"
#include "render/pipeline.h"
#include "render/scheduler.h"
#include "shader/backend/spirv/emit_spirv_discard_frag.h"
#include "shader/backend/spirv/emit_spirv_quad_rect.h"
#include "shader/info.h"
#include "shader/resource.h"
#include "render/pipeline_cache.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"
#include "metal/vk_metal_shadow.h" // bbport (macOS): no-ops elsewhere

namespace Vulkan {

namespace {

using Shader::Backend::SPIRV::AuxShaderType;

/// Vulkan stage bits by software stage.
constexpr std::array SwStageBits = {
    vk::ShaderStageFlagBits::eFragment,
    vk::ShaderStageFlagBits::eTessellationControl,
    vk::ShaderStageFlagBits::eTessellationEvaluation,
    vk::ShaderStageFlagBits::eVertex,
    vk::ShaderStageFlagBits::eGeometry,
    vk::ShaderStageFlagBits::eCompute,
};

/// Wave64 where the host can (GCN's lane masks are 64 bits).
const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo Wave64{
    .requiredSubgroupSize = 64,
};

/// A cached pipeline the driver rejects is a damaged cache entry: the cache is rebuilt
/// (PipelineCache::WarmUp) instead of stopping the game at every start.
template <typename Result>
vk::UniquePipeline CheckPipeline(Result result, bool preloading, const char* kind) {
    if (preloading && result.result != vk::Result::eSuccess) {
        throw Serialization::CorruptData{fmt::format("cached {} pipeline rejected by the driver: {}",
                                                     kind, vk::to_string(result.result))};
    }
    ASSERT_MSG(result.result == vk::Result::eSuccess, "Failed to create {} pipeline: {}", kind,
               vk::to_string(result.result));
    return std::move(result.value);
}

} // namespace

// Pipeline

Pipeline::Pipeline(const Instance& instance_, Scheduler& scheduler_, DescriptorHeap& desc_heap_,
                   const Shader::Profile& profile_, bool is_compute_)
    : instance{instance_}, scheduler{scheduler_}, desc_heap{desc_heap_}, profile{profile_},
      is_compute{is_compute_} {}

Pipeline::~Pipeline() {
    if (MetalShadow::Enabled() && pipeline) { // bbport (macOS): its handle may come back
        MetalShadow::Forget(*pipeline);
    }
}

void Pipeline::CreateLayouts(vk::ShaderStageFlags push_stages, const char* kind,
                             const std::string& debug_name) {
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    u32 binding{};
    for (const Shader::Info* stage : stages) {
        if (!stage) {
            continue;
        }
        const auto stage_bit = SwStageBits[u32(stage->sw_stage)];
        const auto add = [&](vk::DescriptorType type, u32 count) {
            bindings.push_back({
                .binding = binding,
                .descriptorType = type,
                .descriptorCount = count,
                .stageFlags = stage_bit,
            });
            binding += count;
        };
        for (size_t i = 0; i < stage->buffers.size(); ++i) {
            add(vk::DescriptorType::eStorageBuffer, 1);
        }
        for (const auto& image : stage->images) {
            add(image.is_written ? vk::DescriptorType::eStorageImage
                                 : vk::DescriptorType::eSampledImage,
                image.NumBindings(*stage));
        }
        for (size_t i = 0; i < stage->samplers.size(); ++i) {
            add(vk::DescriptorType::eSampler, 1);
        }
    }
    uses_push_descriptors = binding < instance.MaxPushDescriptors();
    const auto device = instance.GetDevice();
    auto [set_result, set_layout] = device.createDescriptorSetLayoutUnique({
        .flags = uses_push_descriptors ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                                       : vk::DescriptorSetLayoutCreateFlagBits{},
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });
    ASSERT_MSG(set_result == vk::Result::eSuccess, "Failed to create {} descriptor set layout: {}",
               kind, vk::to_string(set_result));
    desc_layout = std::move(set_layout);

    const vk::PushConstantRange push_constants{
        .stageFlags = push_stages,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };
    const vk::DescriptorSetLayout set = *desc_layout;
    auto [layout_result, layout] = device.createPipelineLayoutUnique({
        .setLayoutCount = 1U,
        .pSetLayouts = &set,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    });
    ASSERT_MSG(layout_result == vk::Result::eSuccess, "Failed to create {} pipeline layout: {}",
               kind, vk::to_string(layout_result));
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "{} PipelineLayout {}", kind, debug_name);
}

void Pipeline::BindResources(DescriptorWrites& set_writes, const Shader::PushData& push_data,
                             std::span<const vk::DescriptorImageInfo> image_infos,
                             std::span<const vk::DescriptorBufferInfo> buffer_infos) const {
    const auto bind_point =
        IsCompute() ? vk::PipelineBindPoint::eCompute : vk::PipelineBindPoint::eGraphics;
    const auto stage_flags = IsCompute() ? vk::ShaderStageFlagBits::eCompute : AllGraphicsStageBits;
    const vk::PipelineLayout layout = *pipeline_layout;
    scheduler.Record([layout, stage_flags, push_data](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushConstants(layout, stage_flags, 0u, sizeof(push_data), &push_data);
    });
    if (set_writes.empty()) {
        return;
    }
    if (!uses_push_descriptors) {
        // Descriptor set updates are device calls: they stay on this thread.
        const auto desc_set = desc_heap.Commit(*desc_layout);
        for (auto& write : set_writes) {
            write.dstSet = desc_set;
        }
        instance.GetDevice().updateDescriptorSets(set_writes, {});
        scheduler.Record([bind_point, layout, desc_set](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindDescriptorSets(bind_point, layout, 0, desc_set, {});
        });
        return;
    }
    if (!scheduler.IsRecordingDeferred()) {
        scheduler.Record([&](vk::CommandBuffer cmdbuf) {
            cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, set_writes);
        });
        return;
    }
    // Deferred recording: the writes and the infos they point to are copied into the recording
    // chunk with the pointers aimed at the copies. When every write points into the caller's
    // info arrays (draws and dispatches), each array is copied once and the pointers rebased
    // (a copy per descriptor cost the recording thread ~3% of its time).
    const auto within = [](const auto* p, u32 count, const auto& array) {
        return p >= array.data() && p + count <= array.data() + array.size();
    };
    const bool bulk =
        (!image_infos.empty() || !buffer_infos.empty()) &&
        std::ranges::all_of(set_writes, [&](const vk::WriteDescriptorSet& write) {
            return !write.pTexelBufferView &&
                   (!write.pImageInfo ||
                    within(write.pImageInfo, write.descriptorCount, image_infos)) &&
                   (!write.pBufferInfo ||
                    within(write.pBufferInfo, write.descriptorCount, buffer_infos));
        });
    std::span<const vk::WriteDescriptorSet> writes;
    if (bulk) {
        scheduler.ReserveRecordData(set_writes.size() * sizeof(vk::WriteDescriptorSet) +
                                    image_infos.size_bytes() + buffer_infos.size_bytes() + 256);
        writes = scheduler.RecordData(std::span<const vk::WriteDescriptorSet>{set_writes});
        const auto images = scheduler.RecordData(image_infos);
        const auto buffers = scheduler.RecordData(buffer_infos);
        for (auto& write : std::span{const_cast<vk::WriteDescriptorSet*>(writes.data()),
                                     writes.size()}) {
            if (write.pImageInfo) {
                write.pImageInfo = images.data() + (write.pImageInfo - image_infos.data());
            }
            if (write.pBufferInfo) {
                write.pBufferInfo = buffers.data() + (write.pBufferInfo - buffer_infos.data());
            }
        }
    } else {
        size_t bytes = set_writes.size() * sizeof(vk::WriteDescriptorSet) + 64;
        for (const auto& write : set_writes) {
            bytes += write.descriptorCount *
                         (sizeof(vk::DescriptorBufferInfo) + sizeof(vk::DescriptorImageInfo) +
                          sizeof(vk::BufferView)) +
                     32;
        }
        scheduler.ReserveRecordData(bytes);
        writes = scheduler.RecordData(std::span<const vk::WriteDescriptorSet>{set_writes});
        for (auto& write : std::span{const_cast<vk::WriteDescriptorSet*>(writes.data()),
                                     writes.size()}) {
            if (write.pBufferInfo) {
                write.pBufferInfo =
                    scheduler.RecordData(std::span{write.pBufferInfo, write.descriptorCount}).data();
            }
            if (write.pImageInfo) {
                write.pImageInfo =
                    scheduler.RecordData(std::span{write.pImageInfo, write.descriptorCount}).data();
            }
            if (write.pTexelBufferView) {
                write.pTexelBufferView =
                    scheduler.RecordData(std::span{write.pTexelBufferView, write.descriptorCount})
                        .data();
            }
        }
    }
    scheduler.Record([bind_point, layout, writes](vk::CommandBuffer cmdbuf) {
        cmdbuf.pushDescriptorSetKHR(bind_point, layout, 0, writes);
    });
}

std::string Pipeline::GetDebugString() const {
    std::string names;
    for (const Shader::Info* stage : stages) {
        if (stage) {
            const auto name = PipelineCache::GetShaderName(stage->hw_stage, stage->pgm_hash);
            names = names.empty() ? name : fmt::format("{},{}", names, name);
        }
    }
    return names;
}

// ComputePipeline

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info, vk::ShaderModule module,
                                 SerializationSupport& sdata, bool preloading)
    : Pipeline{instance, scheduler, desc_heap, profile, true}, compute_key{compute_key_} {
    stages[u32(Shader::SwStage::Compute)] = &info;
    const auto debug_name = GetDebugString();
    CreateLayouts(vk::ShaderStageFlagBits::eCompute, "Compute", debug_name);
    const vk::ComputePipelineCreateInfo pipeline_ci{
        .stage =
            {
                .pNext = instance.IsSubgroupSize64Supported() ? &Wave64 : nullptr,
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = module,
                .pName = "main",
            },
        .layout = *pipeline_layout,
    };
    {
        // bbport (macOS): the game's pipeline (MetalShadow::Compute below), not a generic one.
        [[maybe_unused]] MetalShadow::GameBuildScope game_build;
        pipeline = CheckPipeline(instance.GetDevice().createComputePipelineUnique(pipeline_cache,
                                                                                  pipeline_ci),
                                 preloading, "compute");
    }
    SetObjectName(instance.GetDevice(), *pipeline, "Compute Pipeline {}", debug_name);
    if (MetalShadow::Enabled()) { // bbport (macOS): docs/MACOS_METAL_RENDERER.md, stage 2
        MetalShadow::Compute(compute_key, module, *pipeline);
    }
}

ComputePipeline::~ComputePipeline() = default;

// GraphicsPipeline

namespace {

/// The dynamic state: everything Vulkan allows except what the key holds.
boost::container::static_vector<vk::DynamicState, 32> DynamicStates(const Instance& instance,
                                                                    bool has_vertex_bindings) {
    boost::container::static_vector<vk::DynamicState, 32> states = {
        vk::DynamicState::eViewportWithCount,  vk::DynamicState::eScissorWithCount,
        vk::DynamicState::eBlendConstants,     vk::DynamicState::eDepthTestEnable,
        vk::DynamicState::eDepthWriteEnable,   vk::DynamicState::eDepthCompareOp,
        vk::DynamicState::eDepthBiasEnable,    vk::DynamicState::eDepthBias,
        vk::DynamicState::eStencilTestEnable,  vk::DynamicState::eStencilReference,
        vk::DynamicState::eStencilCompareMask, vk::DynamicState::eStencilWriteMask,
        vk::DynamicState::eStencilOp,          vk::DynamicState::eCullMode,
        vk::DynamicState::eFrontFace,          vk::DynamicState::eRasterizerDiscardEnable,
        vk::DynamicState::eLineWidth,          vk::DynamicState::ePrimitiveRestartEnable,
    };
    if (instance.IsDepthBoundsSupported()) {
        states.push_back(vk::DynamicState::eDepthBoundsTestEnable);
        states.push_back(vk::DynamicState::eDepthBounds);
    }
    if (instance.IsDynamicColorWriteMaskSupported()) {
        states.push_back(vk::DynamicState::eColorWriteMaskEXT);
    }
    if (instance.IsVertexInputDynamicState()) {
        states.push_back(vk::DynamicState::eVertexInputEXT);
    } else if (has_vertex_bindings) {
        states.push_back(vk::DynamicState::eVertexInputBindingStride);
    }
    return states;
}

/// The blend state of color attachment `i`, with GCN's differences from Vulkan emulated.
vk::PipelineColorBlendAttachmentState BlendAttachment(const Instance& instance,
                                                      const GraphicsPipelineKey& key, u32 i) {
    const auto& control = key.blend_controls[i];
    const auto src_color = GcnToVk::BlendFactor(control.color_src_factor);
    const auto dst_color = GcnToVk::BlendFactor(control.color_dst_factor);
    const auto color_blend = GcnToVk::BlendOp(control.color_func);
    const bool separate = control.separate_alpha_blend;
    const auto src_alpha = separate ? GcnToVk::BlendFactor(control.alpha_src_factor) : src_color;
    const auto dst_alpha = separate ? GcnToVk::BlendFactor(control.alpha_dst_factor) : dst_color;
    const auto alpha_blend = separate ? GcnToVk::BlendOp(control.alpha_func) : color_blend;

    // Vulkan ignores factors for min/max, but a factor zeroing one operand makes the operation
    // a selection (min(s, 0) = 0, max(s, 0) = s for normalized alpha): an equivalent add.
    auto eff_src_alpha = src_alpha;
    auto eff_dst_alpha = dst_alpha;
    auto eff_alpha_blend = alpha_blend;
    if (alpha_blend == vk::BlendOp::eMin || alpha_blend == vk::BlendOp::eMax) {
        const bool takes_max = alpha_blend == vk::BlendOp::eMax;
        if (src_alpha == vk::BlendFactor::eOne && dst_alpha == vk::BlendFactor::eZero) {
            eff_alpha_blend = vk::BlendOp::eAdd;
            eff_src_alpha = takes_max ? vk::BlendFactor::eOne : vk::BlendFactor::eZero;
            eff_dst_alpha = vk::BlendFactor::eZero;
        } else if (src_alpha == vk::BlendFactor::eZero && dst_alpha == vk::BlendFactor::eOne) {
            eff_alpha_blend = vk::BlendOp::eAdd;
            eff_src_alpha = vk::BlendFactor::eZero;
            eff_dst_alpha = takes_max ? vk::BlendFactor::eOne : vk::BlendFactor::eZero;
        }
    }
    const bool self_scale = key.color_buffers[i].blend_self_scale;
    const auto scaled_min_max = [](vk::BlendOp op, vk::BlendFactor src, vk::BlendFactor dst) {
        return (op == vk::BlendOp::eMin || op == vk::BlendOp::eMax) &&
               (src != vk::BlendFactor::eOne || dst != vk::BlendFactor::eOne);
    };
    if ((scaled_min_max(color_blend, src_color, dst_color) && !self_scale) ||
        scaled_min_max(eff_alpha_blend, eff_src_alpha, eff_dst_alpha)) {
        LOG_WARNING(Render_Vulkan,
                    "Unimplemented use of min/max blend op with blend factor not equal to one.");
    }

    constexpr auto AllComponents = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                   vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    vk::PipelineColorBlendAttachmentState state{
        .blendEnable = control.enable,
        .srcColorBlendFactor = src_color,
        .dstColorBlendFactor = dst_color,
        .colorBlendOp = color_blend,
        .srcAlphaBlendFactor = eff_src_alpha,
        .dstAlphaBlendFactor = eff_dst_alpha,
        .alphaBlendOp = eff_alpha_blend,
        .colorWriteMask =
            instance.IsDynamicColorWriteMaskSupported() ? AllComponents : key.write_masks[i],
    };
    // The shader squares its output for this attachment (PsColorBuffer::blend_self_scale): the
    // factors must not scale again.
    if (self_scale) {
        LOG_WARNING(Render_Vulkan,
                    "Emulating scaled min/max blend with squared shader output on attachment {}",
                    i);
        state.srcColorBlendFactor = vk::BlendFactor::eOne;
        state.dstColorBlendFactor = vk::BlendFactor::eOne;
    }
    // With alpha masked out of the export (CB_SHADER_MASK), GCN blends with source alpha 1;
    // the shader still writes 0 there, so alpha-based color factors are resolved here.
    const bool alpha_masked_out =
        (key.cb_shader_mask.GetMask(i) & Gcn::ColorBufferMask::ComponentA) == 0;
    const auto source_alpha_as_one = [](vk::BlendFactor factor) {
        return factor == vk::BlendFactor::eSrcAlpha ? vk::BlendFactor::eOne
                                                    : vk::BlendFactor::eZero; // 1 - A
    };
    const auto uses_source_alpha = [](vk::BlendFactor factor) {
        return factor == vk::BlendFactor::eSrcAlpha || factor == vk::BlendFactor::eOneMinusSrcAlpha;
    };
    if (alpha_masked_out && uses_source_alpha(src_color)) {
        state.srcColorBlendFactor = source_alpha_as_one(src_color);
    }
    if (alpha_masked_out && uses_source_alpha(dst_color)) {
        state.dstColorBlendFactor = source_alpha_as_one(dst_color);
    }
    return state;
}

} // namespace

GraphicsPipeline::GraphicsPipeline(
    const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
    const Shader::Profile& profile, const GraphicsPipelineKey& key_,
    vk::PipelineCache pipeline_cache, std::span<const Shader::Info*, MaxShaderStages> infos,
    std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
    std::optional<const Shader::Gcn::FetchShaderData> fetch_shader_,
    std::span<const vk::ShaderModule> modules, SerializationSupport& sdata, bool preloading)
    : Pipeline{instance, scheduler, desc_heap, profile, false}, key{key_},
      fetch_shader{std::move(fetch_shader_)} {
    const vk::Device device = instance.GetDevice();
    std::ranges::copy(infos, stages.begin());
    const auto debug_name = GetDebugString();
    CreateLayouts(AllGraphicsStageBits, "Graphics", debug_name);
    const auto& fs_info = runtime_infos[u32(Shader::SwStage::Fragment)].hw.fs;

    // Vertex input (static only without dynamic vertex input).
    if (!preloading && !instance.IsVertexInputDynamicState()) {
        VertexInputs<Gcn::Buffer> guest_buffers;
        const auto& vs_info = runtime_infos[u32(Shader::SwStage::Vertex)].sw.vs;
        GetVertexInputs(sdata.vertex_attributes, sdata.vertex_bindings, sdata.divisors,
                        guest_buffers, vs_info.step_rate_0, vs_info.step_rate_1);
    }
    const vk::PipelineVertexInputDivisorStateCreateInfo divisor_state{
        .vertexBindingDivisorCount = static_cast<u32>(sdata.divisors.size()),
        .pVertexBindingDivisors = sdata.divisors.data(),
    };
    const vk::PipelineVertexInputStateCreateInfo vertex_input{
        .pNext = sdata.divisors.empty() ? nullptr : &divisor_state,
        .vertexBindingDescriptionCount = static_cast<u32>(sdata.vertex_bindings.size()),
        .pVertexBindingDescriptions = sdata.vertex_bindings.data(),
        .vertexAttributeDescriptionCount = static_cast<u32>(sdata.vertex_attributes.size()),
        .pVertexAttributeDescriptions = sdata.vertex_attributes.data(),
    };

    // Rect and quad lists are drawn as patches (auxiliary tessellation shaders).
    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = GcnToVk::PrimitiveType(key.prim_type),
    };
    const bool is_rect_list = key.prim_type == Gcn::PrimitiveType::RectList;
    const bool is_quad_list = key.prim_type == Gcn::PrimitiveType::QuadList;
    const vk::PipelineTessellationStateCreateInfo tessellation_state{
        .patchControlPoints = is_rect_list ? 3U : (is_quad_list ? 4U : key.patch_control_points),
    };

    vk::StructureChain raster_chain = {
        vk::PipelineRasterizationStateCreateInfo{
            .depthClampEnable = key.depth_clamp_enable &&
                                (!key.depth_clip_enable || instance.IsDepthClipEnableSupported()),
            .rasterizerDiscardEnable = false,
            .polygonMode = GcnToVk::PolygonMode(key.polygon_mode),
            .lineWidth = 1.0f,
        },
        vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT{
            .provokingVertexMode = key.provoking_vtx_last == Gcn::ProvokingVtxLast::First
                                       ? vk::ProvokingVertexModeEXT::eFirstVertex
                                       : vk::ProvokingVertexModeEXT::eLastVertex,
        },
        vk::PipelineRasterizationDepthClipStateCreateInfoEXT{
            .depthClipEnable = key.depth_clip_enable,
        },
    };
    if (!instance.IsProvokingVertexSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT>();
    }
    if (!instance.IsDepthClipEnableSupported()) {
        raster_chain.unlink<vk::PipelineRasterizationDepthClipStateCreateInfoEXT>();
    }

    if (!preloading) {
        sdata.multisampling = {
            .rasterizationSamples = GcnToVk::NumSamples(
                key.num_samples, instance.GetColorSampleCounts() & instance.GetDepthSampleCounts()),
            .sampleShadingEnable =
                fs_info.addr_flags.persp_sample_ena || fs_info.addr_flags.linear_sample_ena,
        };
    }
    const vk::PipelineViewportDepthClipControlCreateInfoEXT clip_control{
        .negativeOneToOne = key.clip_space == Gcn::ClipSpace::MinusWToW,
    };
    const vk::PipelineViewportStateCreateInfo viewport_info{
        .pNext = instance.IsDepthClipControlSupported() ? &clip_control : nullptr,
    };
    const auto dynamic_states = DynamicStates(instance, !sdata.vertex_bindings.empty());
    const vk::PipelineDynamicStateCreateInfo dynamic_info{
        .dynamicStateCount = static_cast<u32>(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };

    // Stages: the translated ones, or generated ones where the primitive needs them.
    boost::container::static_vector<vk::PipelineShaderStageCreateInfo, MaxShaderStages> shaders;
    const auto add_stage = [&](vk::ShaderStageFlagBits stage, vk::ShaderModule module) {
        shaders.push_back({
            .pNext = instance.IsSubgroupSize64Supported() ? &Wave64 : nullptr,
            .stage = stage,
            .module = module,
            .pName = "main",
        });
    };
    const auto translated = [&](Shader::SwStage stage) -> vk::ShaderModule {
        return infos[u32(stage)] ? modules[u32(stage)] : vk::ShaderModule{};
    };
    if (const auto module = translated(Shader::SwStage::Vertex)) {
        add_stage(vk::ShaderStageFlagBits::eVertex, module);
    }
    if (const auto module = translated(Shader::SwStage::Geometry)) {
        add_stage(vk::ShaderStageFlagBits::eGeometry, module);
    }
    if (const auto module = translated(Shader::SwStage::TessellationControl)) {
        add_stage(vk::ShaderStageFlagBits::eTessellationControl, module);
    } else if (is_rect_list || is_quad_list) {
        if (!preloading) {
            sdata.tcs = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
                is_quad_list ? AuxShaderType::QuadListTCS : AuxShaderType::RectListTCS, fs_info);
        }
        add_stage(vk::ShaderStageFlagBits::eTessellationControl, CompileSPV(sdata.tcs, device));
    }
    if (const auto module = translated(Shader::SwStage::TessellationEval)) {
        add_stage(vk::ShaderStageFlagBits::eTessellationEvaluation, module);
    } else if (is_rect_list || is_quad_list) {
        if (!preloading) {
            sdata.tes = Shader::Backend::SPIRV::EmitAuxilaryTessShader(
                AuxShaderType::PassthroughTES, fs_info);
        }
        add_stage(vk::ShaderStageFlagBits::eTessellationEvaluation, CompileSPV(sdata.tes, device));
    }
    if (const auto module = translated(Shader::SwStage::Fragment)) {
        add_stage(vk::ShaderStageFlagBits::eFragment, module);
    } else if (preloading ? !sdata.fragment.empty() : fs_info.clip_distance_emulation) {
        if (!preloading) {
            const auto& vs = runtime_infos[u32(Shader::SwStage::Vertex)].hw.vs;
            sdata.fragment = Shader::Backend::SPIRV::EmitDiscardFragmentShader(vs.outputs);
        }
        add_stage(vk::ShaderStageFlagBits::eFragment, CompileSPV(sdata.fragment, device));
    }

    // Attachments.
    const auto depth_format =
        instance.GetSupportedFormat(GcnToVk::DepthFormat(key.z_format, key.stencil_format),
                                    vk::FormatFeatureFlagBits2::eDepthStencilAttachment);
    std::array<vk::Format, Shader::IR::NumRenderTargets> color_formats;
    for (u32 i = 0; i < key.num_color_attachments; ++i) {
        const auto& color = key.color_buffers[i];
        const auto format = instance.GetSupportedFormat(
            GcnToVk::SurfaceFormat(color.data_format, color.num_format),
            vk::FormatFeatureFlagBits2::eColorAttachment);
        if (!instance.IsFormatSupported(format, vk::FormatFeatureFlagBits2::eColorAttachment)) {
            LOG_WARNING(Render_Vulkan, "color buffer format {} does not support COLOR_ATTACHMENT_BIT",
                        vk::to_string(format));
        }
        color_formats[i] = format;
    }
    std::array<vk::SampleCountFlagBits, Gcn::NUM_COLOR_BUFFERS> color_samples;
    std::ranges::transform(key.color_samples, color_samples.begin(), [&instance](u8 samples) {
        return samples ? GcnToVk::NumSamples(samples, instance.GetColorSampleCounts())
                       : vk::SampleCountFlagBits::e1;
    });
    const vk::AttachmentSampleCountInfoAMD mixed_samples{
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentSamples = color_samples.data(),
        .depthStencilAttachmentSamples =
            GcnToVk::NumSamples(key.depth_samples, instance.GetDepthSampleCounts()),
    };
    const vk::PipelineRenderingCreateInfo rendering_ci{
        .pNext = instance.IsMixedDepthSamplesSupported() ? &mixed_samples : nullptr,
        .colorAttachmentCount = key.num_color_attachments,
        .pColorAttachmentFormats = color_formats.data(),
        .depthAttachmentFormat = key.z_format != Gcn::DepthBuffer::ZFormat::Invalid
                                     ? depth_format
                                     : vk::Format::eUndefined,
        .stencilAttachmentFormat = key.stencil_format != Gcn::DepthBuffer::StencilFormat::Invalid
                                       ? depth_format
                                       : vk::Format::eUndefined,
    };
    std::array<vk::PipelineColorBlendAttachmentState, Gcn::NUM_COLOR_BUFFERS> attachments;
    for (u32 i = 0; i < key.num_color_attachments; i++) {
        attachments[i] = BlendAttachment(instance, key, i);
    }
    const vk::PipelineColorBlendStateCreateInfo color_blending{
        .logicOpEnable =
            instance.IsLogicOpSupported() && key.logic_op != Gcn::ColorControl::LogicOp::Copy,
        .logicOp = GcnToVk::LogicOp(key.logic_op),
        .attachmentCount = key.num_color_attachments,
        .pAttachments = attachments.data(),
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };
    // Required without extended dynamic state 3 (it is all dynamic in practice).
    constexpr vk::PipelineDepthStencilStateCreateInfo depth_stencil_info{};

    const vk::GraphicsPipelineCreateInfo pipeline_ci{
        .pNext = &rendering_ci,
        .stageCount = static_cast<u32>(shaders.size()),
        .pStages = shaders.data(),
        .pVertexInputState = !instance.IsVertexInputDynamicState() ? &vertex_input : nullptr,
        .pInputAssemblyState = &input_assembly,
        .pTessellationState = &tessellation_state,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_chain.get(),
        .pMultisampleState = &sdata.multisampling,
        .pDepthStencilState =
            !instance.IsExtendedDynamicState3Supported() ? &depth_stencil_info : nullptr,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = *pipeline_layout,
    };
    {
        // bbport (macOS): the game's pipeline (MetalShadow::Graphics below), not a generic one.
        [[maybe_unused]] MetalShadow::GameBuildScope game_build;
        pipeline = CheckPipeline(device.createGraphicsPipelineUnique(pipeline_cache, pipeline_ci),
                                 preloading, "graphics");
    }
    SetObjectName(device, *pipeline, "Graphics Pipeline {}", debug_name);
    // bbport (macOS): docs/MACOS_METAL_RENDERER.md, stage 2. Not with dynamic vertex input: the
    // key then holds no vertex formats (never so on KosmicKrisp).
    if (MetalShadow::Enabled() && !instance.IsVertexInputDynamicState()) {
        // The fetch shader's attributes in key.vertex_buffer_formats' order (GetVertexInputs).
        // Step rates: the static vertex input's when there is one (also when preloading, where
        // the runtime infos are not filled in), else the vertex stage's runtime info.
        boost::container::static_vector<MetalShadow::VertexInput, MaxVertexBufferCount> inputs;
        if (fetch_shader) {
            using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
            const auto& vs = runtime_infos[u32(Shader::SwStage::Vertex)].sw.vs;
            const bool from_sdata = !sdata.vertex_bindings.empty();
            for (const auto& attrib : fetch_shader->attributes) {
                const auto step_rate = attrib.GetStepRate();
                u32 divisor = 1;
                if (from_sdata) {
                    for (const auto& d : sdata.divisors) {
                        if (d.binding == attrib.semantic) {
                            divisor = d.divisor;
                        }
                    }
                } else if (!preloading) {
                    divisor = step_rate == InstanceIdType::OverStepRate0   ? vs.step_rate_0
                              : step_rate == InstanceIdType::OverStepRate1 ? vs.step_rate_1
                                                                           : 1;
                }
                inputs.push_back({
                    .location = attrib.semantic,
                    .per_instance = step_rate != InstanceIdType::None,
                    .divisor = divisor,
                });
            }
        }
        std::array<bool, MaxShaderStages> has_stage{};
        for (u32 i = 0; i < MaxShaderStages; ++i) {
            has_stage[i] = infos[i] != nullptr;
        }
        const bool aux_fragment = !infos[u32(Shader::SwStage::Fragment)] && !sdata.fragment.empty();
        MetalShadow::Graphics(key, modules, has_stage, std::span{inputs.data(), inputs.size()},
                              std::span{attachments.data(), key.num_color_attachments},
                              aux_fragment, *pipeline);
    }
}

GraphicsPipeline::~GraphicsPipeline() = default;

template <typename Attribute, typename Binding>
void GraphicsPipeline::GetVertexInputs(
    VertexInputs<Attribute>& attributes, VertexInputs<Binding>& bindings,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
    VertexInputs<Gcn::Buffer>& guest_buffers, u32 step_rate_0, u32 step_rate_1,
    std::span<const Gcn::Buffer> sharps) const {
    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
    if (!fetch_shader || fetch_shader->attributes.empty()) {
        return;
    }
    const auto& vs_info = GetStage(Shader::SwStage::Vertex);
    const bool given = sharps.size() == fetch_shader->attributes.size();
    for (u32 index = 0; const auto& attrib : fetch_shader->attributes) {
        const auto step_rate = attrib.GetStepRate();
        const auto buffer = given ? sharps[index] : attrib.GetSharp(vs_info);
        ++index;
        // An impossible format is a V# read while the game rewrote it (seen with BBSC on,
        // 2026-10-14: data format 9 with number format 7): a float format for this draw instead
        // of ending the session.
        vk::Format format = GcnToVk::TrySurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        if (format == vk::Format::eUndefined) {
            static std::atomic<u32> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 4) {
                LOG_WARNING(Render_Vulkan, "vertex attribute {} with data format {} and number "
                            "format {}: read as 32-bit floats", attrib.semantic,
                            u32(buffer.GetDataFmt()), u32(buffer.GetNumberFmt()));
            }
            format = vk::Format::eR32G32B32A32Sfloat;
        }
        attributes.push_back(Attribute{
            .location = attrib.semantic,
            .binding = attrib.semantic,
            .format = format,
            .offset = 0,
        });
        bindings.push_back(Binding{
            .binding = attrib.semantic,
            .stride = buffer.GetStride(),
            .inputRate = step_rate == InstanceIdType::None ? vk::VertexInputRate::eVertex
                                                           : vk::VertexInputRate::eInstance,
        });
        const u32 divisor = step_rate == InstanceIdType::OverStepRate0   ? step_rate_0
                            : step_rate == InstanceIdType::OverStepRate1 ? step_rate_1
                                                                         : 1;
        if constexpr (std::is_same_v<Binding, vk::VertexInputBindingDescription2EXT>) {
            bindings.back().divisor = divisor;
        } else if (step_rate != InstanceIdType::None) {
            divisors.push_back({.binding = attrib.semantic, .divisor = divisor});
        }
        guest_buffers.emplace_back(buffer);
    }
}

template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription>&,
    VertexInputs<vk::VertexInputBindingDescription>&,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>&, VertexInputs<Gcn::Buffer>&, u32,
    u32, std::span<const Gcn::Buffer>) const;
template void GraphicsPipeline::GetVertexInputs(
    VertexInputs<vk::VertexInputAttributeDescription2EXT>&,
    VertexInputs<vk::VertexInputBindingDescription2EXT>&,
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>&, VertexInputs<Gcn::Buffer>&, u32,
    u32, std::span<const Gcn::Buffer>) const;

} // namespace Vulkan
