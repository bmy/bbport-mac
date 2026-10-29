// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "textures/blitter.h"

#include <algorithm>
#include <array>

#include "common/assert.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"
#include "vulkan/shader_module.h"

#include "bbgpu/shaders/depth_from_color_frag.h"
#include "bbgpu/shaders/fullscreen_vert.h"
#include "bbgpu/shaders/texel_copy_frag.h"
#include "bbgpu/shaders/texel_copy_msaa_frag.h"

namespace VideoCore {

namespace {

vk::SampleCountFlagBits SampleCount(u32 samples) {
    ASSERT_MSG(samples >= 1 && samples <= 16 && std::has_single_bit(samples), "{} samples",
               samples);
    return vk::SampleCountFlagBits(samples);
}

vk::UniqueShaderModule Module(std::span<const u32> code, vk::Device device) {
    return vk::UniqueShaderModule{Vulkan::CompileSPV(code, device), device};
}

} // namespace

ImageBlitter::ImageBlitter(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    const auto device = instance.GetDevice();
    vertex = Module(FULLSCREEN_VERT, device);
    depth_from_color = Module(DEPTH_FROM_COLOR_FRAG, device);
    copy = Module(TEXEL_COPY_FRAG, device);
    copy_msaa = Module(TEXEL_COPY_MSAA_FRAG, device);

    const vk::DescriptorSetLayoutBinding texture{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eSampledImage,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };
    auto layout = device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = 1,
        .pBindings = &texture,
    });
    ASSERT_MSG(layout.result == vk::Result::eSuccess, "blitter set layout: {}",
               vk::to_string(layout.result));
    set_layout = std::move(layout.value);
    const vk::DescriptorSetLayout set = *set_layout;
    auto pl_layout = device.createPipelineLayoutUnique({.setLayoutCount = 1, .pSetLayouts = &set});
    ASSERT_MSG(pl_layout.result == vk::Result::eSuccess, "blitter pipeline layout: {}",
               vk::to_string(pl_layout.result));
    pipeline_layout = std::move(pl_layout.value);
    Vulkan::SetObjectName(device, *pipeline_layout, "blitter layout");
}

ImageBlitter::~ImageBlitter() = default;

vk::Pipeline ImageBlitter::GetPipeline(Kind kind, u32 num_samples, vk::Format format) {
    for (const auto& p : pipelines) {
        if (p.kind == kind && p.num_samples == num_samples && p.format == format) {
            return *p.pipeline;
        }
    }
    const bool to_depth = kind == Kind::DepthFromColor;
    const std::array stages = {
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = *vertex,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = to_depth ? *depth_from_color : kind == Kind::CopyMsaa ? *copy_msaa : *copy,
            .pName = "main",
        },
    };
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = to_depth ? 0u : 1u,
        .pColorAttachmentFormats = to_depth ? nullptr : &format,
        .depthAttachmentFormat = to_depth ? format : vk::Format::eUndefined,
    };
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };
    const vk::PipelineViewportStateCreateInfo viewport{};
    const vk::PipelineRasterizationStateCreateInfo raster{.lineWidth = 1.0f};
    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = SampleCount(num_samples),
    };
    const vk::PipelineDepthStencilStateCreateInfo depth{
        .depthTestEnable = to_depth,
        .depthWriteEnable = to_depth,
        .depthCompareOp = vk::CompareOp::eAlways,
    };
    const vk::PipelineColorBlendAttachmentState color_write{
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo blend{
        .attachmentCount = to_depth ? 0u : 1u,
        .pAttachments = to_depth ? nullptr : &color_write,
    };
    const std::array dynamic_states = {vk::DynamicState::eViewportWithCount,
                                       vk::DynamicState::eScissorWithCount};
    const vk::PipelineDynamicStateCreateInfo dynamic{
        .dynamicStateCount = u32(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    auto [result, pipeline] = instance.GetDevice().createGraphicsPipelineUnique(
        VK_NULL_HANDLE, vk::GraphicsPipelineCreateInfo{
                            .pNext = &rendering,
                            .stageCount = u32(stages.size()),
                            .pStages = stages.data(),
                            .pVertexInputState = &vertex_input,
                            .pInputAssemblyState = &assembly,
                            .pViewportState = &viewport,
                            .pRasterizationState = &raster,
                            .pMultisampleState = &multisample,
                            .pDepthStencilState = &depth,
                            .pColorBlendState = &blend,
                            .pDynamicState = &dynamic,
                            .layout = *pipeline_layout,
                        });
    ASSERT_MSG(result == vk::Result::eSuccess, "blitter pipeline: {}", vk::to_string(result));
    Vulkan::SetObjectName(instance.GetDevice(), *pipeline, "blit {} to {} samples",
                          to_depth ? "color as depth" : "texels", num_samples);
    const vk::Pipeline handle = *pipeline;
    pipelines.push_back({kind, num_samples, format, std::move(pipeline)});
    return handle;
}

vk::ImageView ImageBlitter::View(vk::Image image, vk::Format format, vk::ImageAspectFlags aspect,
                                 vk::ImageUsageFlags usage) {
    const vk::ImageViewUsageCreateInfo usage_info{.usage = usage};
    const auto [result, view] = instance.GetDevice().createImageView({
        .pNext = &usage_info,
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{aspect, 0, 1, 0, 1},
    });
    ASSERT_MSG(result == vk::Result::eSuccess, "blitter view: {}", vk::to_string(result));
    return view;
}

void ImageBlitter::Draw(vk::Pipeline pipeline, vk::ImageView source_view, u32 width, u32 height) {
    const auto cmd = scheduler.CommandBuffer();
    const Vulkan::Breadcrumbs::Scope crumb{cmd, scheduler.CrumbStream(), "image blit"};
    const vk::DescriptorImageInfo source{
        .imageView = source_view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0,
                             vk::WriteDescriptorSet{
                                 .dstBinding = 0,
                                 .descriptorCount = 1,
                                 .descriptorType = vk::DescriptorType::eSampledImage,
                                 .pImageInfo = &source,
                             });
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    cmd.setViewportWithCount(vk::Viewport{0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f});
    cmd.setScissorWithCount(vk::Rect2D{{0, 0}, {width, height}});
    if (Vulkan::DynCheck::Enabled()) {
        Vulkan::DynCheck::Set(cmd, {vk::Rect2D{{0, 0}, {width, height}}}, "blitter");
        Vulkan::DynCheck::SetViewports(cmd, {vk::Viewport{0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f}}, "blitter");
    }
    cmd.draw(3, 1, 0, 0);
    scheduler.EndRendering();
    scheduler.GetDynamicState().Invalidate();
}

void ImageBlitter::ColorAsMsDepth(u32 width, u32 height, u32 num_samples, vk::Format src_format,
                                  vk::Format dst_format, vk::Image source, vk::Image dest) {
    const auto color = View(source, src_format, vk::ImageAspectFlagBits::eColor,
                            vk::ImageUsageFlagBits::eSampled);
    const auto depth = View(dest, dst_format, vk::ImageAspectFlagBits::eDepth,
                            vk::ImageUsageFlagBits::eDepthStencilAttachment);
    scheduler.DeferOperation([device = instance.GetDevice(), color, depth] {
        device.destroyImageView(color);
        device.destroyImageView(depth);
    });
    Vulkan::RenderState state{};
    state.width = width;
    state.height = height;
    state.num_layers = 1;
    state.depth_stencil_attachment.image_view = depth;
    state.depth_stencil_attachment.image_layout = vk::ImageLayout::eDepthAttachmentOptimal;
    state.depth_stencil_attachment.has_depth = true;
    state.depth_stencil_attachment.depth_clear = true;
    scheduler.BeginRendering(state);
    Draw(GetPipeline(Kind::DepthFromColor, num_samples, dst_format), color, width, height);
}

void ImageBlitter::CopyMsImage(u32 width, u32 height, u32 num_samples, vk::Format format,
                               bool src_msaa, vk::Image source, vk::Image dest) {
    const auto src = View(source, format, vk::ImageAspectFlagBits::eColor,
                          vk::ImageUsageFlagBits::eSampled);
    const auto dst = View(dest, format, vk::ImageAspectFlagBits::eColor,
                          vk::ImageUsageFlagBits::eColorAttachment);
    scheduler.DeferOperation([device = instance.GetDevice(), src, dst] {
        device.destroyImageView(src);
        device.destroyImageView(dst);
    });
    Vulkan::RenderState state{};
    state.width = width;
    state.height = height;
    state.num_layers = 1;
    state.num_color_attachments = 1;
    state.color_attachments[0].image_view = dst;
    state.color_attachments[0].image_layout = vk::ImageLayout::eColorAttachmentOptimal;
    state.color_attachments[0].is_clear = true;
    scheduler.BeginRendering(state);
    Draw(GetPipeline(src_msaa ? Kind::CopyMsaa : Kind::Copy, num_samples, format), src, width,
         height);
}

} // namespace VideoCore
