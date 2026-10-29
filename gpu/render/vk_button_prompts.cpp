// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_button_prompts.h"

#include <algorithm>
#include <array>
#include <cstdio>

#include "bbgpu/shaders/atlas_decode_comp.h"
#include "bbport_button_prompts.h"
#include "buffers/buffer_cache.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "vulkan/shader_module.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/image_view.h"

namespace Vulkan {
namespace {

bool IsSrgb(vk::Format format) {
    switch (format) {
    case vk::Format::eBc1RgbaSrgbBlock:
    case vk::Format::eBc1RgbSrgbBlock:
    case vk::Format::eBc2SrgbBlock:
    case vk::Format::eBc3SrgbBlock:
    case vk::Format::eBc7SrgbBlock:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eA8B8G8R8SrgbPack32:
        return true;
    default:
        return false;
    }
}

} // namespace

ButtonPrompts::ButtonPrompts(const Instance& instance_, Scheduler& scheduler_,
                             VideoCore::BufferCache& buffer_cache_)
    : instance{instance_}, scheduler{scheduler_}, buffer_cache{buffer_cache_},
      image{instance_.GetDevice(), instance_.GetAllocator()} {
    BbPrompts::Start();
}

ButtonPrompts::~ButtonPrompts() = default;

void ButtonPrompts::Update() {
    BbPrompts::Set set;
    if (!BbPrompts::Update(generation, set)) {
        return;
    }
    generation = set.generation;
    drawn = set.drawn;
    active = std::find(drawn.begin(), drawn.end(), true) != drawn.end();
    if (!active) {
        return; // the game's icons: nothing to upload
    }
    if (!image) {
        layers = set.count;
        image.Create(vk::ImageCreateInfo{
            .flags = vk::ImageCreateFlagBits::eMutableFormat,
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {BbPrompts::IconSize, BbPrompts::IconSize, 1},
            .mipLevels = 1,
            .arrayLayers = layers,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
    }
    if (set.count != layers) {
        active = false;
        return;
    }
    // The icons at the game's size for the atlases' cells: means of premultiplied texels.
    constexpr u32 Small = BbPrompts::GameIconSize, Factor = BbPrompts::IconSize / Small;
    small.assign(std::size_t(layers) * Small * Small, 0);
    for (u32 layer = 0; layer < layers; ++layer) {
        const u32* icon = set.pixels.data() + std::size_t(layer) * BbPrompts::IconSize * BbPrompts::IconSize;
        for (u32 y = 0; y < Small; ++y) {
            for (u32 x = 0; x < Small; ++x) {
                u32 sum[4]{};
                for (u32 dy = 0; dy < Factor; ++dy) {
                    for (u32 dx = 0; dx < Factor; ++dx) {
                        const u32 texel = icon[(y * Factor + dy) * BbPrompts::IconSize + x * Factor + dx];
                        for (u32 c = 0; c < 4; ++c) {
                            sum[c] += (texel >> (8 * c)) & 0xff;
                        }
                    }
                }
                u32 out = 0;
                for (u32 c = 0; c < 4; ++c) {
                    out |= (sum[c] + Factor * Factor / 2) / (Factor * Factor) << (8 * c);
                }
                small[(std::size_t(layer) * Small + y) * Small + x] = out;
            }
        }
    }
    const u64 bytes = u64(set.pixels.size()) * sizeof(u32);
    auto& stream = buffer_cache.GetStreamBuffer();
    const u64 offset = stream.Copy(set.pixels.data(), bytes, 16);
    // In stream order, ahead of the draws that sample it; the draws before it are done reading it
    // (the barrier's source stages).
    scheduler.EndRendering();
    scheduler.Record([image = vk::Image{image.image}, buffer = stream.Handle(), offset,
                      layers = layers](vk::CommandBuffer cmdbuf) {
        const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, layers};
        const vk::ImageMemoryBarrier2 to_copy{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = image,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                   .pImageMemoryBarriers = &to_copy});
        const vk::BufferImageCopy copy{
            .bufferOffset = offset,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, layers},
            .imageOffset = {0, 0, 0},
            .imageExtent = {BbPrompts::IconSize, BbPrompts::IconSize, 1},
        };
        cmdbuf.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, copy);
        const vk::ImageMemoryBarrier2 to_sample{
            .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = image,
            .subresourceRange = range,
        };
        cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                   .pImageMemoryBarriers = &to_sample});
    });
}

vk::ImageView ButtonPrompts::View(u32 icon, const VideoCore::ImageViewInfo& info) {
    if (!active || icon == 0 || icon > drawn.size() || !drawn[icon - 1]) {
        return {};
    }
    const bool array = info.type == Gcn::ImageType::Color2DArray;
    if (!array && info.type != Gcn::ImageType::Color2D) {
        return {};
    }
    const bool srgb = IsSrgb(info.format);
    const u32 mapping = u32(info.mapping.r) | u32(info.mapping.g) << 8 | u32(info.mapping.b) << 16 |
                        u32(info.mapping.a) << 24;
    auto& view = views[{icon, srgb, array, mapping}];
    if (!view) {
        auto [result, created] = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
            .image = image.image,
            .viewType = array ? vk::ImageViewType::e2DArray : vk::ImageViewType::e2D,
            .format = srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
            .components = info.mapping,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, icon - 1, 1},
        });
        if (result != vk::Result::eSuccess) {
            std::printf("Button prompts: no image view (%s)\n", vk::to_string(result).c_str());
            return {};
        }
        view = std::move(created);
    }
    return *view;
}

vk::ImageView ButtonPrompts::AtlasView(Runtime& runtime, VideoCore::Image& image,
                                       const VideoCore::ImageViewInfo& info) {
    if (!active || image.prompt_atlas == 0 || info.type != Gcn::ImageType::Color2D) {
        return {};
    }
    const auto cells = BbPrompts::AtlasCells(image.prompt_atlas);
    const auto is_drawn = [&](const BbPrompts::AtlasCell& cell) {
        return cell.icon <= drawn.size() && drawn[cell.icon - 1] && cell.icon <= layers;
    };
    if (std::none_of(cells.begin(), cells.end(), is_drawn)) {
        return {};
    }
    // The game's atlas (BC7) is decoded by a compute pass sampling it through an UNORM view (its
    // stored values), the port's icons copied into the cells after. A blit from the BC7 image
    // decoded it on RADV but left the HUD's other icons (vials, bullets, frames) out on AMD's
    // Windows driver (2026-10-27); a blit from an sRGB format into a UNORM one also converts to
    // linear light, which the sRGB view then took again.
    const vk::Format source_format = image.backing->image.image_ci.format;
    const u32 width = image.info.size.width, height = image.info.size.height;
    if ((source_format != vk::Format::eBc7UnormBlock && source_format != vk::Format::eBc7SrgbBlock) ||
        !DecodePipeline()) {
        return {};
    }
    Atlas& atlas = atlases[image.image_uid];
    if (!atlas.image) {
        atlas.image = VideoCore::HostImage{instance.GetDevice(), instance.GetAllocator()};
        atlas.image.Create(vk::ImageCreateInfo{
            .flags = vk::ImageCreateFlagBits::eMutableFormat,
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .extent = {width, height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eStorage,
            .sharingMode = vk::SharingMode::eExclusive,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        auto [result, created] = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
            .image = atlas.image.image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR8G8B8A8Unorm,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (result != vk::Result::eSuccess) {
            std::printf("Button prompts: no atlas storage view (%s)\n", vk::to_string(result).c_str());
            atlases.erase(image.image_uid);
            return {};
        }
        atlas.storage_view = std::move(created);
    }
    if (atlas.source != image.GetImage()) {
        const vk::ImageViewUsageCreateInfo sampled{.usage = vk::ImageUsageFlagBits::eSampled};
        auto [result, created] = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
            .pNext = &sampled,
            .image = image.GetImage(),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eBc7UnormBlock,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (result != vk::Result::eSuccess) {
            std::printf("Button prompts: no atlas source view (%s)\n", vk::to_string(result).c_str());
            return {};
        }
        atlas.source = image.GetImage();
        atlas.source_view = std::move(created);
        atlas.upload = ~0u; // decoded again from this image
    }
    if (atlas.upload != image.prompt_upload || atlas.generation != generation) {
        atlas.upload = image.prompt_upload;
        atlas.generation = generation;
        constexpr u32 Small = BbPrompts::GameIconSize;
        std::vector<vk::BufferImageCopy> copies;
        std::vector<u32> pixels;
        for (const auto& cell : cells) {
            if (!is_drawn(cell) || cell.x + Small > width || cell.y + Small > height) {
                continue;
            }
            copies.push_back(vk::BufferImageCopy{
                .bufferOffset = pixels.size() * sizeof(u32),
                .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .imageOffset = {s32(cell.x), s32(cell.y), 0},
                .imageExtent = {Small, Small, 1},
            });
            const u32* icon = small.data() + std::size_t(cell.icon - 1) * Small * Small;
            pixels.insert(pixels.end(), icon, icon + Small * Small);
        }
        auto& stream = buffer_cache.GetStreamBuffer();
        const u64 offset = stream.Copy(pixels.data(), pixels.size() * sizeof(u32), 16);
        for (auto& copy : copies) {
            copy.bufferOffset += offset;
        }
        scheduler.EndRendering();
        runtime.Transit(&image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderSampledRead);
        runtime.FlushBarriers();
        scheduler.Record([target = vk::Image{atlas.image.image}, target_view = *atlas.storage_view,
                          source_view = *atlas.source_view, pipeline = *decode_pipeline,
                          layout = *decode_layout, buffer = stream.Handle(), copies = std::move(copies), width,
                          height](vk::CommandBuffer cmdbuf) {
            const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            const vk::ImageMemoryBarrier2 to_decode{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = target,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &to_decode});
            const vk::DescriptorImageInfo source_info{.imageView = source_view,
                                                      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
            const vk::DescriptorImageInfo target_info{.imageView = target_view,
                                                      .imageLayout = vk::ImageLayout::eGeneral};
            const std::array<vk::WriteDescriptorSet, 2> writes{{
                {.dstBinding = 0, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &source_info},
                {.dstBinding = 1, .descriptorCount = 1, .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &target_info},
            }};
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
            cmdbuf.dispatch((width + 7) / 8, (height + 7) / 8, 1);
            const vk::ImageMemoryBarrier2 to_copy{
                .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eCopy,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eTransferDstOptimal,
                .image = target,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &to_copy});
            if (!copies.empty()) {
                cmdbuf.copyBufferToImage(buffer, target, vk::ImageLayout::eTransferDstOptimal, copies);
            }
            const vk::ImageMemoryBarrier2 to_sample{
                .srcStageMask = vk::PipelineStageFlagBits2::eCopy,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eTransferDstOptimal,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = target,
                .subresourceRange = range,
            };
            cmdbuf.pipelineBarrier2(vk::DependencyInfo{.imageMemoryBarrierCount = 1,
                                                       .pImageMemoryBarriers = &to_sample});
        });
    }
    const bool srgb = IsSrgb(info.format);
    const u32 mapping = u32(info.mapping.r) | u32(info.mapping.g) << 8 | u32(info.mapping.b) << 16 |
                        u32(info.mapping.a) << 24;
    auto& view = atlas.views[{srgb, false, mapping}];
    if (!view) {
        // Sampled only (the copy is a storage image for the decode; sRGB views cannot be).
        const vk::ImageViewUsageCreateInfo sampled{.usage = vk::ImageUsageFlagBits::eSampled};
        auto [result, created] = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
            .pNext = &sampled,
            .image = atlas.image.image,
            .viewType = vk::ImageViewType::e2D,
            .format = srgb ? vk::Format::eR8G8B8A8Srgb : vk::Format::eR8G8B8A8Unorm,
            .components = info.mapping,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        });
        if (result != vk::Result::eSuccess) {
            std::printf("Button prompts: no atlas view (%s)\n", vk::to_string(result).c_str());
            return {};
        }
        view = std::move(created);
    }
    return *view;
}

bool ButtonPrompts::DecodePipeline() {
    if (decode_pipeline || decode_failed) {
        return bool(decode_pipeline);
    }
    decode_failed = true; // made once; a failure keeps the game's own atlas
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 2> bindings{{
        {.binding = 0, .descriptorType = vk::DescriptorType::eSampledImage, .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1, .descriptorType = vk::DescriptorType::eStorageImage, .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    decode_set_layout = Check(device.createDescriptorSetLayoutUnique(
        {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
         .bindingCount = u32(bindings.size()),
         .pBindings = bindings.data()}));
    decode_layout = Check(device.createPipelineLayoutUnique({.setLayoutCount = 1, .pSetLayouts = &*decode_set_layout}));
    const auto module = CompileSPV(ATLAS_DECODE_COMP, device);
    decode_pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute, .module = module, .pName = "main"},
                .layout = *decode_layout}));
    device.destroyShaderModule(module);
    decode_failed = !decode_pipeline;
    return bool(decode_pipeline);
}

} // namespace Vulkan
