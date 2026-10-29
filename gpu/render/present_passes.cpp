// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/present_passes.h"

#include <array>

#include "vulkan/platform.h"
#include "vulkan/shader_module.h"

#include "bbgpu/shaders/fullscreen_vert.h"
#include "bbgpu/shaders/present_encode_frag.h"

namespace Vulkan {

namespace {

constexpr vk::ImageSubresourceRange ColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

vk::UniqueSampler LinearClampSampler(vk::Device device) {
    return Check<"create present sampler">(device.createSamplerUnique({
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
        .addressModeW = vk::SamplerAddressMode::eClampToEdge,
        .maxAnisotropy = 1.0f,
        .minLod = -1000.0f,
        .maxLod = 1000.0f,
    }));
}

} // namespace

EncodePass::EncodePass(vk::Device device, vk::Format output_format)
    : sampler{LinearClampSampler(device)} {
    const vk::DescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };
    set_layout = Check<"create encode set layout">(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = 1,
        .pBindings = &binding,
    }));
    const vk::PushConstantRange constants{
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .size = sizeof(Settings),
    };
    const vk::DescriptorSetLayout set = *set_layout;
    layout = Check<"create encode layout">(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &set,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &constants,
    }));

    const vk::ShaderModule vertex = CompileSPV(FULLSCREEN_VERT, device);
    const vk::ShaderModule fragment = CompileSPV(PRESENT_ENCODE_FRAG, device);
    ASSERT_MSG(vertex && fragment, "encode pass shaders did not compile");
    const std::array stages{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vertex,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fragment,
            .pName = "main",
        },
    };
    const vk::PipelineRenderingCreateInfo rendering{
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &output_format,
    };
    const vk::PipelineVertexInputStateCreateInfo vertex_input{};
    const vk::PipelineInputAssemblyStateCreateInfo assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };
    const vk::PipelineViewportStateCreateInfo viewport{.viewportCount = 1, .scissorCount = 1};
    const vk::PipelineRasterizationStateCreateInfo raster{.lineWidth = 1.0f};
    const vk::PipelineMultisampleStateCreateInfo multisample{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };
    const vk::PipelineColorBlendAttachmentState write_all{
        .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                          vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
    };
    const vk::PipelineColorBlendStateCreateInfo blend{
        .attachmentCount = 1,
        .pAttachments = &write_all,
    };
    const std::array dynamic_states{vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    const vk::PipelineDynamicStateCreateInfo dynamic{
        .dynamicStateCount = u32(dynamic_states.size()),
        .pDynamicStates = dynamic_states.data(),
    };
    pipeline = Check<"create encode pipeline">(device.createGraphicsPipelineUnique(
        {}, vk::GraphicsPipelineCreateInfo{
                .pNext = &rendering,
                .stageCount = u32(stages.size()),
                .pStages = stages.data(),
                .pVertexInputState = &vertex_input,
                .pInputAssemblyState = &assembly,
                .pViewportState = &viewport,
                .pRasterizationState = &raster,
                .pMultisampleState = &multisample,
                .pColorBlendState = &blend,
                .pDynamicState = &dynamic,
                .layout = *layout,
            }));
    device.destroyShaderModule(vertex);
    device.destroyShaderModule(fragment);
    SetObjectName(device, *pipeline, "present encode");
}

EncodePass::~EncodePass() = default;

void EncodePass::Render(vk::CommandBuffer cmdbuf, vk::ImageView input, vk::ImageView output,
                        vk::Extent2D output_size, const Settings& settings) {
    const vk::RenderingAttachmentInfo attachment{
        .imageView = output,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eDontCare,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    const vk::DescriptorImageInfo source{
        .sampler = *sampler,
        .imageView = input,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::WriteDescriptorSet write{
        .dstBinding = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &source,
    };
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
    cmdbuf.setViewport(0, vk::Viewport{0.0f, 0.0f, float(output_size.width),
                                       float(output_size.height), 0.0f, 1.0f});
    cmdbuf.setScissor(0, vk::Rect2D{{0, 0}, output_size});
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, *layout, 0, write);
    cmdbuf.pushConstants(*layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(Settings),
                         &settings);
    cmdbuf.beginRendering({
        .renderArea = {{0, 0}, output_size},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    });
    cmdbuf.draw(3, 1, 0, 0);
    cmdbuf.endRendering();
}

} // namespace Vulkan
