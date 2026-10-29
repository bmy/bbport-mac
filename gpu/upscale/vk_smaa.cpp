// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_smaa.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <span>

#include <vk_mem_alloc.h>

#include "bbgpu/shaders/smaa_blend_comp.h"
#include "bbgpu/shaders/smaa_edges_comp.h"
#include "bbgpu/shaders/smaa_weights_comp.h"
#include "render/scheduler.h"
#include "third_party/smaa/AreaTex.h"
#include "third_party/smaa/SearchTex.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

namespace Vulkan {

namespace {

constexpr vk::ImageSubresourceRange ColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

void Barrier(vk::CommandBuffer cmdbuf, vk::Image image, vk::ImageLayout old_layout,
             vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
             vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
             vk::ImageLayout new_layout = vk::ImageLayout::eGeneral) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = src_stage,
        .srcAccessMask = src_access,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = old_layout,
        .newLayout = new_layout,
        .image = image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

} // namespace

Smaa::Smaa(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    const auto device = instance.GetDevice();
    const auto sampler = [&](vk::Filter filter) {
        return Check(device.createSamplerUnique({
            .magFilter = filter,
            .minFilter = filter,
            .mipmapMode = vk::SamplerMipmapMode::eNearest,
            .addressModeU = vk::SamplerAddressMode::eClampToEdge,
            .addressModeV = vk::SamplerAddressMode::eClampToEdge,
            .addressModeW = vk::SamplerAddressMode::eClampToEdge,
            .maxLod = 0.0f,
        }));
    };
    linear_sampler = sampler(vk::Filter::eLinear);
    point_sampler = sampler(vk::Filter::eNearest);
    // 0: the pass's input, 1: its output, 2: the area texture or the weights, 3: the search
    // texture (point sampled, as SMAA wants).
    const std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute,
         .pImmutableSamplers = &*linear_sampler},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute,
         .pImmutableSamplers = &*linear_sampler},
        {.binding = 3,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute,
         .pImmutableSamplers = &*point_sampler},
    }};
    desc_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(std::array<float, 4>)};
    pipeline_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*desc_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto create = [&](std::span<const u32> code) {
        const auto module = CompileSPV(code, device);
        auto result = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = module,
                              .pName = "main"},
                    .layout = *pipeline_layout,
                }));
        device.destroyShaderModule(module);
        return result;
    };
    edges_pipeline = create(SMAA_EDGES_COMP);
    weights_pipeline = create(SMAA_WEIGHTS_COMP);
    blend_pipeline = create(SMAA_BLEND_COMP);
}

Smaa::~Smaa() = default;

bool Smaa::EnsureImages(u32 width, u32 height) {
    if (output_image && width == image_width && height == image_height) {
        return true;
    }
    // Submitted passes may still use the old images: they go once the GPU is past them.
    scheduler.DeferOperation([edges = std::move(edges_image), weights = std::move(weights_image),
                              output = std::move(output_image), ev = std::move(edges_view),
                              wv = std::move(weights_view),
                              ov = std::move(output_view)]() mutable {});
    const auto device = instance.GetDevice();
    const auto make = [&](VideoCore::HostImage& image, vk::UniqueImageView& view,
                          vk::ImageUsageFlags extra) {
        image = VideoCore::HostImage(device, instance.GetAllocator());
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled | extra,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        view = Check(device.createImageViewUnique({
            .image = vk::Image(image),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = ColorRange,
        }));
    };
    make(edges_image, edges_view, {});
    make(weights_image, weights_view, {});
    make(output_image, output_view, vk::ImageUsageFlagBits::eTransferSrc);
    image_width = width;
    image_height = height;
    return bool(output_image);
}

void Smaa::UploadLookupTextures() {
    const auto device = instance.GetDevice();
    const auto make = [&](VideoCore::HostImage& image, vk::UniqueImageView& view, vk::Format format,
                          u32 width, u32 height) {
        image = VideoCore::HostImage(device, instance.GetAllocator());
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        view = Check(device.createImageViewUnique({
            .image = vk::Image(image),
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = ColorRange,
        }));
    };
    make(area_image, area_view, vk::Format::eR8G8Unorm, AREATEX_WIDTH, AREATEX_HEIGHT);
    make(search_image, search_view, vk::Format::eR8Unorm, SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT);

    // Both in one host buffer: the area texture, then the search texture.
    constexpr VkDeviceSize area_bytes = sizeof(areaTexBytes);
    constexpr VkDeviceSize total = area_bytes + sizeof(searchTexBytes);
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = total,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    const VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                        &info) != VK_SUCCESS) {
        return;
    }
    std::memcpy(info.pMappedData, areaTexBytes, area_bytes);
    std::memcpy(static_cast<u8*>(info.pMappedData) + area_bytes, searchTexBytes,
                sizeof(searchTexBytes));
    vmaFlushAllocation(instance.GetAllocator(), allocation, 0, VK_WHOLE_SIZE);
    scheduler.RecordCrumb({.name = "SMAA lookup textures"},
                          [buffer = vk::Buffer(buffer), area = vk::Image(area_image),
                           search = vk::Image(search_image)](vk::CommandBuffer cmdbuf) {
        constexpr auto none = vk::PipelineStageFlagBits2::eNone;
        for (const vk::Image image : {area, search}) {
            Barrier(cmdbuf, image, vk::ImageLayout::eUndefined, none, {},
                    vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                    vk::ImageLayout::eTransferDstOptimal);
        }
        const vk::BufferImageCopy area_copy{
            .bufferOffset = 0,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageExtent = {AREATEX_WIDTH, AREATEX_HEIGHT, 1},
        };
        const vk::BufferImageCopy search_copy{
            .bufferOffset = area_bytes,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
            .imageExtent = {SEARCHTEX_WIDTH, SEARCHTEX_HEIGHT, 1},
        };
        cmdbuf.copyBufferToImage(buffer, area, vk::ImageLayout::eTransferDstOptimal, area_copy);
        cmdbuf.copyBufferToImage(buffer, search, vk::ImageLayout::eTransferDstOptimal, search_copy);
        for (const vk::Image image : {area, search}) {
            Barrier(cmdbuf, image, vk::ImageLayout::eTransferDstOptimal,
                    vk::PipelineStageFlagBits2::eCopy, vk::AccessFlagBits2::eTransferWrite,
                    vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead,
                    vk::ImageLayout::eShaderReadOnlyOptimal);
        }
    });
    scheduler.DeferOperation([allocator = instance.GetAllocator(), buffer, allocation] {
        vmaDestroyBuffer(allocator, buffer, allocation);
    });
    uploaded = true;
}

bool Smaa::Record(vk::ImageView color_view, vk::ImageLayout color_layout, u32 width,
                  u32 height) {
    if (!color_view || width == 0 || height == 0 || !EnsureImages(width, height)) {
        return false;
    }
    if (!uploaded) {
        UploadLookupTextures();
        if (!uploaded) {
            return false;
        }
        std::printf("SMAA: on (1x, high preset), %ux%u\n", width, height);
    }
    const std::array<float, 4> metrics{1.0f / float(width), 1.0f / float(height), float(width),
                                       float(height)};
    scheduler.RecordCrumb(
        {.name = "SMAA"},
        [metrics, color_view, color_layout, edges = vk::Image(edges_image),
         edges_view = *edges_view, weights = vk::Image(weights_image), weights_view = *weights_view,
         output = vk::Image(output_image), output_view = *output_view, area_view = *area_view,
         search_view = *search_view, edges_pipeline = *edges_pipeline,
         weights_pipeline = *weights_pipeline, blend_pipeline = *blend_pipeline,
         layout = *pipeline_layout](vk::CommandBuffer cmdbuf) {
            const auto push = [&](vk::ImageView input, vk::ImageLayout input_layout,
                                  vk::ImageView storage, vk::ImageView second,
                                  vk::ImageLayout second_layout, vk::ImageView third) {
                const vk::DescriptorImageInfo infos[4] = {
                    {.imageView = input, .imageLayout = input_layout},
                    {.imageView = storage, .imageLayout = vk::ImageLayout::eGeneral},
                    {.imageView = second, .imageLayout = second_layout},
                    {.imageView = third, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal},
                };
                std::array<vk::WriteDescriptorSet, 4> writes{};
                u32 count = 0;
                for (u32 i = 0; i < 4; ++i) {
                    if (!infos[i].imageView) {
                        continue;
                    }
                    writes[count++] = {.dstBinding = i,
                                       .descriptorCount = 1,
                                       .descriptorType = i == 1
                                           ? vk::DescriptorType::eStorageImage
                                           : vk::DescriptorType::eCombinedImageSampler,
                                       .pImageInfo = &infos[i]};
                }
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0,
                                            std::span(writes.data(), count));
                cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0,
                                     sizeof(metrics), &metrics);
                cmdbuf.dispatch((u32(metrics[2]) + 7) / 8, (u32(metrics[3]) + 7) / 8, 1);
            };
            constexpr auto compute = vk::PipelineStageFlagBits2::eComputeShader;
            constexpr auto all = vk::PipelineStageFlagBits2::eAllCommands;
            constexpr auto any =
                vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
            constexpr auto write = vk::AccessFlagBits2::eShaderStorageWrite;
            constexpr auto read = vk::AccessFlagBits2::eShaderSampledRead;
            // Every pass rewrites its whole image: the old contents go.
            for (const vk::Image image : {edges, weights, output}) {
                Barrier(cmdbuf, image, vk::ImageLayout::eUndefined, all, any, compute, write);
            }
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, edges_pipeline);
            push(color_view, color_layout, edges_view, {}, vk::ImageLayout::eGeneral, {});
            Barrier(cmdbuf, edges, vk::ImageLayout::eGeneral, compute, write, compute, read);
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, weights_pipeline);
            push(edges_view, vk::ImageLayout::eGeneral, weights_view, area_view,
                 vk::ImageLayout::eShaderReadOnlyOptimal, search_view);
            Barrier(cmdbuf, weights, vk::ImageLayout::eGeneral, compute, write, compute, read);
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, blend_pipeline);
            push(color_view, color_layout, output_view, weights_view, vk::ImageLayout::eGeneral,
                 {});
            Barrier(cmdbuf, output, vk::ImageLayout::eGeneral, compute, write, all, any);
        });
    return true;
}

} // namespace Vulkan
