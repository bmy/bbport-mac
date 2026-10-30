// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport: the upscalers of AMD and NVIDIA behind the temporal upscaler: FSR 4 (and its reactive
// mask) and DLSS. Split out of vk_temporal_upscaler.cpp (2026-10-26).

#include "upscale/vk_temporal_upscaler.h"
#include "render/vk_gpu_profiler.h"
#include "upscale/motion_history.h"
#include "upscale/vk_scene_resolution.h"
#include "upscale/ui_composition.h"

#include <array>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <vk_mem_alloc.h>

#include "bbport_guest_hooks.h"
#include "game_profile.h"
#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "ffx_vk_portable.h"
#include "bbgpu/shaders/upscale_merge_comp.h"
#include "bbgpu/shaders/upscale_reactive_comp.h"
#include "bbgpu/shaders/taa_comp.h"
#include "bbgpu/shaders/taa_sharpen_comp.h"
#include "bbgpu/shaders/taa_sharpen_ldr_comp.h"
#include "bbgpu/shaders/fsr1_easu_comp.h"
#include "bbgpu/shaders/fsr4_decode_comp.h"
#include "bbgpu/shaders/fsr4_encode_comp.h"
#include "bbgpu/shaders/fsr4_reactive_comp.h"
#include "vulkan/shader_module.h"
#include "upscale/vk_camera_motion.h"
#include "upscale/vk_frame_generation.h"
#include "upscale/vk_smaa.h"
#include "render/vk_frame_capture.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "metal/mtl_layer.h"
#include "textures/texture_cache.h"
#include "upscale/upscaler_internal.h"

namespace Vulkan {

void TemporalUpscaler::RecordFsr4Reactive(vk::CommandBuffer cmdbuf, vk::ImageView color, u32 w,
                                          u32 h, u32 ow, u32 oh) {
    const auto all = vk::PipelineStageFlagBits2::eAllCommands;
    const vk::MemoryBarrier2 before{
        .srcStageMask = all, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite};
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &before});
    const vk::DescriptorImageInfo current{.sampler = *fsr4_linear_sampler, .imageView = color,
                                          .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo mask{.sampler = *fsr4_linear_sampler, .imageView = *reactive_view,
                                       .imageLayout = vk::ImageLayout::eGeneral};
    const vk::DescriptorImageInfo out{.imageView = *output_view,
                                      .imageLayout = vk::ImageLayout::eGeneral};
    const std::array<vk::WriteDescriptorSet, 3> writes{{
        {.dstBinding = 0, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &current},
        {.dstBinding = 1, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &mask},
        {.dstBinding = 2, .descriptorCount = 1,
         .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &out},
    }};
    const std::array<float, 5> push{float(w), float(h), jitter[0], jitter[1], 1.0f};
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *fsr4_reactive_pipeline);
    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *fsr4_reactive_pipeline_layout, 0,
                                writes);
    cmdbuf.pushConstants(*fsr4_reactive_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
                         sizeof(push), push.data());
    cmdbuf.dispatch((ow + 7) / 8, (oh + 7) / 8, 1);
    const vk::MemoryBarrier2 after{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderWrite,
        .dstStageMask = all,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite};
    cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &after});
}

bool TemporalUpscaler::UseFsr4() const {
    const int selected = BbSettings::Get().upscaler;
    const bool supported = selected == BbSettings::UpscalerFsr411   ? instance.IsFsr411Supported()
                           : selected == BbSettings::UpscalerFsrAmd ? fsr4->AmdAvailable()
                                                                    : instance.IsFsr4Int8Supported();
    // bbport (macOS): not in the Metal command path's frame mode (it records around the layer).
    return BbSettings::IsFsr4(selected) && supported && !fsr4_failed &&
           !fsr4_waiting && !MetalLayer::FrameActive();
}

bool TemporalUpscaler::RecordFsr4(vk::CommandBuffer cmdbuf, Fsr4Upscaler::Image color,
                                  Fsr4Upscaler::Image depth, u32 w, u32 h, u32 ow, u32 oh,
                                  float frame_ms) {
    const auto& settings = BbSettings::Get();
    // Same jitter convention as FSR 3; the menu (or toggle 1 << 26) flips it for tests.
    const float sign =
        BbToggle::Disabled(1u << 26) != settings.fsr4_invert_jitter.load() ? -1.0f : 1.0f;
    const bool ok = fsr4->Record({
        .cmdbuf = cmdbuf,
        .color = color,
        .depth = depth,
        .motion = {vk::Image(motion_image), *motion_view, w, h, vk::Format::eR16G16Sfloat},
        .output = {vk::Image(output_image), *output_view, ow, oh, vk::Format::eR16G16B16A16Sfloat},
        .render_width = w,
        .render_height = h,
        .preset = applied_preset,
        .jitter = {sign * jitter[0], sign * jitter[1]},
        .frame_ms = frame_ms,
        .near_plane = camera_motion.Near(),
        .far_plane = 3000.0f,
        .vertical_fov = camera_motion.VerticalFov(),
        .sharpness = std::min(settings.sharpness.load(), 1.0f),
        .sharpen = settings.sharpen,
        .reset = reset,
        .auto_exposure = settings.fsr4_auto_exposure,
    });
    // The menu shows the reason; it outlives this frame (FSR 4 keeps its last message).
    static std::string shown;
    const char* problem = fsr4->Problem();
    if (!problem) {
        BbSettings::Get().fsr4_problem = nullptr;
    } else if (shown != problem) {
        shown = problem;
        static std::array<std::string, 8> kept;
        static u32 next = 0;
        kept[next] = shown;
        BbSettings::Get().fsr4_problem = kept[next].c_str();
        next = (next + 1) % kept.size();
    }
    if (!ok && fsr4->Fatal()) {
        std::printf("Upscaler: falling back to FSR 3.1\n");
        BbSettings::Get().upscaler = BbSettings::UpscalerFsr3;
        fsr4_failed = true; // EnsureResources creates the FSR 3 context next frame
        reset = true;
    }
    return ok;
}

bool TemporalUpscaler::UseDlss() const {
    const Dlss* dlss = Dlss::Get();
    return BbSettings::Get().upscaler == BbSettings::UpscalerDlss && dlss && dlss->Available() &&
           !dlss_failed;
}

bool TemporalUpscaler::RecordDlss(vk::CommandBuffer cmdbuf, const Dlss::Resource& color,
                                  const Dlss::Resource& depth, u32 w, u32 h, u32 ow, u32 oh,
                                  float frame_ms, bool hdr) {
    Dlss* dlss = Dlss::Get();
    const Dlss::FeatureDesc desc{w, h, ow, oh, Dlss::QualityForScale(float(ow) / float(w)), hdr};
    bool ok = true;
    if (!dlss->HasFeature(desc)) {
        // The previous feature may still be in use by submitted work; `cmdbuf` stays open.
        scheduler.WaitSubmittedWork();
        dlss->ReleaseFeature();
        ok = dlss->CreateFeature(cmdbuf, desc);
        reset = true;
    }
    if (ok) {
        // The jitter and motion vectors FSR 3 gets: render pixels, current to previous.
        const float sign = BbToggle::Disabled(1u << 26) ? -1.0f : 1.0f;
        const auto& settings = BbSettings::Get();
        ok = dlss->Evaluate(cmdbuf, {
            .color = color,
            .depth = depth,
            .motion = {vk::Image(motion_image), *motion_view, vk::Format::eR16G16Sfloat,
                       vk::ImageAspectFlagBits::eColor, w, h},
            .output = {vk::Image(output_image), *output_view, vk::Format::eR16G16B16A16Sfloat,
                       vk::ImageAspectFlagBits::eColor, ow, oh},
            .jitter_x = sign * jitter[0],
            .jitter_y = sign * jitter[1],
            .reset = reset,
            .frame_ms = frame_ms,
            .sharpness = settings.sharpen ? std::min(settings.sharpness.load(), 1.0f) : 0.0f,
        });
    }
    if (!ok) {
        std::printf("Upscaler: DLSS failed; falling back to FSR 3.1\n");
        BbSettings::Get().upscaler = BbSettings::UpscalerFsr3;
        dlss_failed = true; // EnsureResources creates the FSR 3 context next frame
        reset = true;
    }
    return ok;
}

} // namespace Vulkan
