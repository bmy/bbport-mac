// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_vbao.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <utility>

#include "bbport_settings.h"

#include "bbgpu/shaders/vbao_comp.h"
#include "bbgpu/shaders/vbao_blur_comp.h"
#include "bbgpu/shaders/vbao_blur_rgba_comp.h"
#include "bbgpu/shaders/vbao_view_comp.h"
#include "upscale/vk_camera_motion.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "vulkan/shader_module.h"
#include "textures/texture_cache.h"

namespace Vulkan {

namespace {

struct PushConstants {
    std::array<std::array<float, 4>, 3> view_rows; // world to view rotation
    std::array<float, 4> proj;         // x scale, y scale, z scale, z offset (ndc +y down)
    std::array<float, 2> size;         // output pixels
    std::array<float, 2> normal_scale; // normal texels per output pixel
    float depth_scale;                 // depth pixels per output pixel
    float radius;                      // view units (metres)
    float thickness;                   // assumed thickness of the depth layer, view units
    float intensity;                   // exponent on the visibility
    float max_radius_px;
    u32 frame;       // sample pattern rotation; 0 when nothing accumulates frames
    u32 normal_mode; // 1: G-buffer normals through view_rows
};

struct BlurConstants {
    std::array<float, 2> size;
    s32 radius;
    float tolerance;
    float normal_power;
};

} // namespace

Vbao::Vbao(const Instance& instance_, Scheduler& scheduler_,
           VideoCore::TextureCache& texture_cache_, Runtime& runtime_, const CameraMotion& camera_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_}, runtime{runtime_},
      camera{camera_} {
    const auto device = instance.GetDevice();
    const std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {{
        {.binding = 0,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 1,
         .descriptorType = vk::DescriptorType::eStorageImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
        {.binding = 2,
         .descriptorType = vk::DescriptorType::eSampledImage,
         .descriptorCount = 1,
         .stageFlags = vk::ShaderStageFlagBits::eCompute},
    }};
    desc_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(PushConstants)};
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
    pipeline = create(VBAO_COMP);
    blur_pipeline = create(VBAO_BLUR_COMP);
    blur_rgba_pipeline = create(VBAO_BLUR_RGBA_COMP);
    view_pipeline = create(VBAO_VIEW_COMP);
}

Vbao::~Vbao() = default;

bool Vbao::Enabled() const {
    return BbSettings::Get().vbao && camera.Enabled();
}

void Vbao::OnAoPass(VideoCore::ImageId target, VideoCore::ImageId normals_id, bool temporal) {
    const auto proj = camera.SceneProjection();
    const VideoCore::ImageId depth_id = camera.Depth();
    if (!target || !depth_id || proj[0] == 0.0f || proj[1] == 0.0f) {
        return;
    }
    auto& depth = texture_cache.GetImage(depth_id);
    auto& ao = texture_cache.GetImage(target);
    const auto depth_format = depth.info.pixel_format;
    const auto ao_format = ao.info.pixel_format;
    // The half-size path: an RGBA8 target of half the depth's size (rounded up).
    const bool half = ao_format == vk::Format::eR8G8B8A8Unorm;
    if ((depth_format == vk::Format::eD32Sfloat || depth_format == vk::Format::eD32SfloatS8Uint) &&
        (ao_format == vk::Format::eR8Unorm || half)) {
        if (instance.IsCompressibleImages()) { // bbport (macOS): written as storage
            runtime.UpgradeUsage(&ao, "VBAO output");
        }
    }
    const u32 scale = half ? 2 : 1;
    if ((depth_format != vk::Format::eD32Sfloat && depth_format != vk::Format::eD32SfloatS8Uint) ||
        (ao_format != vk::Format::eR8Unorm && !half) ||
        !(ao.usage_flags & vk::ImageUsageFlagBits::eStorage) ||
        ao.info.size.width != (depth.info.size.width + scale - 1) / scale ||
        ao.info.size.height != (depth.info.size.height + scale - 1) / scale) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::printf("VBAO: skipped (depth %s %ux%u, AO target %s %ux%u)\n",
                        vk::to_string(depth_format).c_str(), depth.info.size.width,
                        depth.info.size.height, vk::to_string(ao.info.pixel_format).c_str(),
                        ao.info.size.width, ao.info.size.height);
        }
        return;
    }
    // The scene's viewport; the targets may be allocated larger (aligned height).
    auto [width, height] = camera.RenderSize();
    width = width ? std::min((width + scale - 1) / scale, ao.info.size.width) : ao.info.size.width;
    height =
        height ? std::min((height + scale - 1) / scale, ao.info.size.height) : ao.info.size.height;
    frame_ao = target;
    frame_ao_width = width;
    frame_ao_height = height;
    if (!Enabled()) {
        return;
    }
    {
        static bool logged[2] = {};
        if (!logged[half]) {
            logged[half] = true;
            std::printf("VBAO: on, %ux%u into the game's %s AO (%s %ux%u)\n", width, height,
                        half ? "half-size" : "upsampled", vk::to_string(ao_format).c_str(),
                        ao.info.size.width, ao.info.size.height);
        }
    }

    // BB_VBAO_RADIUS / BB_VBAO_THICKNESS (metres), BB_VBAO_POWER: tuning without a rebuild.
    static const float radius = bb_env_float("BB_VBAO_RADIUS", 1.5f);
    static const float thickness = bb_env_float("BB_VBAO_THICKNESS", 0.4f);
    static const float power = bb_env_float("BB_VBAO_POWER", 1.5f);
    // The G-buffer normals: a colour image covering the depth (BB_VBAO_NORMALS=0: from depth
    // only, A/B).
    static const bool use_normals = bb_env_float("BB_VBAO_NORMALS", 1.0f) != 0.0f;
    VideoCore::Image* normals = nullptr;
    if (use_normals && normals_id) {
        auto& image = texture_cache.GetImage(normals_id);
        const bool fits = !image.info.props.is_depth && image.backing &&
                          image.info.size.width >= depth.info.size.width / 2 &&
                          image.info.size.height >= depth.info.size.height / 2;
        static bool logged = false;
        if (!logged) {
            logged = true;
            std::printf("VBAO: G-buffer normals %s %ux%u%s\n",
                        vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                        image.info.size.height, fits ? "" : ": not used");
        }
        if (fits) {
            normals = &image;
        }
    }
    const auto view = camera.SceneView();
    const PushConstants push{
        .view_rows = {{{view[0], view[1], view[2], 0.0f},
                       {view[4], view[5], view[6], 0.0f},
                       {view[8], view[9], view[10], 0.0f}}},
        .proj = proj,
        .size = {float(width), float(height)},
        .normal_scale = {normals ? float(normals->info.size.width) / float(ao.info.size.width) : 1.0f,
                         normals ? float(normals->info.size.height) / float(ao.info.size.height)
                                 : 1.0f},
        .depth_scale = float(scale),
        .radius = radius,
        .thickness = thickness,
        .intensity = power,
        .max_radius_px = 0.1f * float(height),
        .frame = temporal ? ++frame : 0u,
        .normal_mode = normals ? 1u : 0u,
    };

    const auto device = instance.GetDevice();
    // The raw AO and view depth, denoised into the game's target.
    if (!raw_image || raw_width != ao.info.size.width || raw_height != ao.info.size.height) {
        scheduler.Finish();
        raw_view.reset();
        raw_width = ao.info.size.width;
        raw_height = ao.info.size.height;
        raw_image = VideoCore::HostImage(device, instance.GetAllocator());
        raw_image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .extent = {raw_width, raw_height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        raw_view = Check(device.createImageViewUnique({
            .image = vk::Image(raw_image),
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
    }
    // BB_VBAO_BLUR: denoise taps per side (2: 5x5; 0: the raw AO). BB_VBAO_BLUR_DEPTH: relative
    // depth difference that stops the blur.
    static const s32 blur = s32(bb_env_float("BB_VBAO_BLUR", 2.0f));
    static const float blur_depth = bb_env_float("BB_VBAO_BLUR_DEPTH", 0.05f);
    // BB_VBAO_BLUR_NORMAL: how sharply the blur stops at normal changes (0: ignores normals).
    static const float blur_normal = bb_env_float("BB_VBAO_BLUR_NORMAL", 8.0f);
    const BlurConstants blur_push{
        .size = push.size,
        .radius = std::clamp(blur, 0, 4),
        .tolerance = blur_depth,
        .normal_power = blur_normal,
    };

    const auto depth_view = Check(device.createImageView({
        .image = vk::Image(depth.backing->image),
        .viewType = vk::ImageViewType::e2D,
        .format = depth_format,
        .subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1},
    }));
    const auto ao_view = Check(device.createImageView({
        .image = vk::Image(ao.backing->image),
        .viewType = vk::ImageViewType::e2D,
        .format = ao_format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));

    // Without the G-buffer normals the depth stands in for binding 2 (not read).
    const auto normal_view = Check(device.createImageView({
        .image = vk::Image(normals ? normals->backing->image : depth.backing->image),
        .viewType = vk::ImageViewType::e2D,
        .format = normals ? normals->info.pixel_format : depth_format,
        .subresourceRange = {normals ? vk::ImageAspectFlagBits::eColor
                                     : vk::ImageAspectFlagBits::eDepth,
                             0, 1, 0, 1},
    }));

    scheduler.EndRendering();
    if (normals) {
        runtime.Transit(normals, vk::ImageLayout::eGeneral,
                        vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
    }
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&ao, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    half ? vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite
                         : vk::AccessFlagBits2::eShaderWrite);
    runtime.FlushBarriers();

    scheduler.RecordCrumb({.name = "vbao"}, [push, blur_push, depth_view, ao_view, normal_view,
                                              raw = vk::Image(raw_image), raw_view = *raw_view,
                                              pipeline = *pipeline,
                                              blur_pipeline = half ? *blur_rgba_pipeline
                                                                   : *blur_pipeline,
                                              layout = *pipeline_layout](vk::CommandBuffer cmdbuf) {
        const auto push_images = [&](vk::ImageView sampled, vk::ImageView storage,
                                     vk::ImageView extra = {}) {
            const vk::DescriptorImageInfo sampled_info{.imageView = sampled,
                                                       .imageLayout = vk::ImageLayout::eGeneral};
            const vk::DescriptorImageInfo storage_info{.imageView = storage,
                                                       .imageLayout = vk::ImageLayout::eGeneral};
            const vk::DescriptorImageInfo extra_info{.imageView = extra,
                                                     .imageLayout = vk::ImageLayout::eGeneral};
            const std::array<vk::WriteDescriptorSet, 3> writes = {{
                {.dstBinding = 0,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &sampled_info},
                {.dstBinding = 1,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eStorageImage,
                 .pImageInfo = &storage_info},
                {.dstBinding = 2,
                 .descriptorCount = 1,
                 .descriptorType = vk::DescriptorType::eSampledImage,
                 .pImageInfo = &extra_info},
            }};
            cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0,
                                        std::span(writes.data(), extra ? 3 : 2));
        };
        const u32 groups_x = (u32(push.size[0]) + 7) / 8, groups_y = (u32(push.size[1]) + 7) / 8;
        // The raw image is rewritten whole each frame: its old contents are not kept.
        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = raw,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        push_images(depth_view, raw_view, normal_view);
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        cmdbuf.dispatch(groups_x, groups_y, 1);

        barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
        barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
        barrier.oldLayout = vk::ImageLayout::eGeneral;
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, blur_pipeline);
        push_images(raw_view, ao_view);
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(blur_push),
                             &blur_push);
        cmdbuf.dispatch(groups_x, groups_y, 1);
    });

    scheduler.DeferOperation([device, depth_view, ao_view, normal_view] {
        device.destroyImageView(depth_view);
        device.destroyImageView(ao_view);
        device.destroyImageView(normal_view);
    });
}

void Vbao::OnUpscaled(vk::Image output, u32 width, u32 height) {
    if (BbSettings::Get().debug_view != BbSettings::DebugAo || !frame_ao) {
        return;
    }
    // The upscaler's own output image (RGBA8 storage, General layout): the UI and the display
    // read this one, not the game's frame.
    DrawView(frame_ao, output, width, height, nullptr);
    view_drawn = true;
}

void Vbao::OnDisplayPass(VideoCore::ImageId frame) {
    const VideoCore::ImageId ao_id = frame_ao;
    const bool drawn = std::exchange(view_drawn, false);
    frame_ao = {};
    if (BbSettings::Get().debug_view != BbSettings::DebugAo || drawn) {
        return;
    }
    if (!frame || !ao_id) {
        return;
    }
    auto& color = texture_cache.GetImage(frame);
    if (color.info.pixel_format == vk::Format::eR8G8B8A8Unorm) {
        if (instance.IsCompressibleImages()) { // bbport (macOS): written as storage
            runtime.UpgradeUsage(&color, "VBAO view");
        }
    }
    if (color.info.pixel_format != vk::Format::eR8G8B8A8Unorm ||
        !(color.usage_flags & vk::ImageUsageFlagBits::eStorage)) {
        return;
    }
    DrawView(ao_id, vk::Image(color.backing->image), color.info.size.width,
             color.info.size.height, &color);
}

void Vbao::DrawView(VideoCore::ImageId ao_id, vk::Image output, u32 width, u32 height,
                    VideoCore::Image* guest) {
    auto& ao = texture_cache.GetImage(ao_id);
    const auto ao_format = ao.info.pixel_format;
    if (ao_format != vk::Format::eR8Unorm && ao_format != vk::Format::eR8G8B8A8Unorm) {
        return;
    }
    const struct {
        std::array<float, 2> frame_size;
        std::array<float, 2> ao_size;
    } push{
        .frame_size = {float(width), float(height)},
        .ao_size = {float(frame_ao_width), float(frame_ao_height)},
    };

    const auto device = instance.GetDevice();
    const auto ao_view = Check(device.createImageView({
        .image = vk::Image(ao.backing->image),
        .viewType = vk::ImageViewType::e2D,
        .format = ao_format,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    const auto color_view = Check(device.createImageView({
        .image = output,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR8G8B8A8Unorm,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));

    scheduler.EndRendering();
    runtime.Transit(&ao, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead);
    if (guest) {
        runtime.Transit(guest, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderWrite);
    }
    runtime.FlushBarriers();

    scheduler.RecordCrumb({.name = "vbao view"}, [push, ao_view, color_view, output,
                                                   own_barriers = guest == nullptr,
                                                   pipeline = *view_pipeline,
                                                   layout = *pipeline_layout](vk::CommandBuffer cmdbuf) {
        // An image outside the texture cache: its barriers are recorded here.
        vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderWrite,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = output,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        if (own_barriers) {
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        }
        const vk::DescriptorImageInfo ao_info{.imageView = ao_view,
                                              .imageLayout = vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo color_info{.imageView = color_view,
                                                 .imageLayout = vk::ImageLayout::eGeneral};
        const std::array<vk::WriteDescriptorSet, 2> writes = {{
            {.dstBinding = 0,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .pImageInfo = &ao_info},
            {.dstBinding = 1,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .pImageInfo = &color_info},
        }};
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
        cmdbuf.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push), &push);
        cmdbuf.dispatch((u32(push.frame_size[0]) + 7) / 8, (u32(push.frame_size[1]) + 7) / 8, 1);
        if (own_barriers) {
            barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
            barrier.srcAccessMask = vk::AccessFlagBits2::eShaderWrite;
            barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
            barrier.dstAccessMask =
                vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        }
    });

    scheduler.DeferOperation([device, ao_view, color_view] {
        device.destroyImageView(ao_view);
        device.destroyImageView(color_view);
    });
}

} // namespace Vulkan
