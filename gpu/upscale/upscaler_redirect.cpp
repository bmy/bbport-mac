// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport: the scene drawn at the render resolution: color, depth and sampled targets redirected
// to the reduced-size proxies, the display pass, and frame generation's capture and interpolation.
// Split out of vk_temporal_upscaler.cpp (2026-10-26).

#include "upscale/vk_temporal_upscaler.h"
#include "shim/bb_env.h"
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
#include "textures/texture_cache.h"
#include "upscale/upscaler_internal.h"

namespace Vulkan {

vk::ImageView TemporalUpscaler::CachedView(const VideoCore::Image& image, vk::Format format,
                                           vk::ImageAspectFlags aspect) {
    const vk::Image handle = image.GetImage();
    ViewEntry* oldest = &view_cache[0];
    for (auto& entry : view_cache) {
        if (entry.view && entry.image == handle && entry.uid == image.image_uid &&
            entry.format == format && entry.aspect == aspect) {
            entry.last_use = ++view_uses;
            return entry.view;
        }
        if (entry.last_use < oldest->last_use) {
            oldest = &entry;
        }
    }
    const auto device = instance.GetDevice();
    if (oldest->view) {
        scheduler.DeferOperation([device, view = oldest->view] { device.destroyImageView(view); });
    }
    *oldest = {
        .image = handle,
        .uid = image.image_uid,
        .format = format,
        .aspect = aspect,
        .view = Check(device.createImageView({
            .image = handle,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = {aspect, 0, 1, 0, 1},
        })),
        .last_use = ++view_uses,
    };
    return oldest->view;
}

void TemporalUpscaler::OnColorTarget(VideoCore::ImageId color) {
    if (ui_phase || !camera_motion.Depth()) {
        return;
    }
    const auto& image = texture_cache.GetImage(color);
    const auto& depth = texture_cache.GetImage(camera_motion.Depth());
    if (image.info.pixel_format == vk::Format::eR8G8B8A8Unorm &&
        image.info.size.width == depth.info.size.width &&
        image.info.size.height == depth.info.size.height) {
        ldr_target = color;
    }
}

void TemporalUpscaler::OnDraw(u64 vs_hash, VideoCore::ImageId color,
                              VideoCore::ImageId depth, bool native_viewport) {
    if (!Scaled() && vs_hash == ui_trigger_vs) {
        // The game's own size: the UI draws into the scene's image. anti_aliasing=smaa: SMAA's
        // result goes back into it before the frame's first UI draw. That draw may write only the
        // stencil (no colour target): the scene is the colour target the frame drew last.
        if (!smaa_done && ldr_target && (!color || color == ldr_target) && SmaaWanted()) {
            if (const vk::Image smoothed = RunSmaa(ldr_target)) {
                auto& image = texture_cache.GetImage(ldr_target);
                runtime.Transit(&image, vk::ImageLayout::eTransferDstOptimal,
                                vk::PipelineStageFlagBits2::eCopy,
                                vk::AccessFlagBits2::eTransferWrite);
                runtime.FlushBarriers();
                const vk::ImageCopy region{
                    .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                    .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                    .extent = {image.info.size.width, image.info.size.height, 1},
                };
                scheduler.Record([smoothed, target = image.GetImage(),
                                  region](vk::CommandBuffer cmd) {
                    cmd.copyImage(smoothed, vk::ImageLayout::eGeneral, target,
                                  vk::ImageLayout::eTransferDstOptimal, region);
                });
            }
        }
        smaa_done = true;
        done_this_frame = true;
    }
    // bbport (issue #67, patch by bmy): a UI draw without a color target (a Scaleform mask, color
    // writes off) writes the stencil the UI's next draws test, so it must go into the UI's
    // output-size depth like them. Left in the guest depth, the mask was lost when the next UI
    // draw switched to the UI depth (PrepareUiDepth clears its stencil): masked images (the
    // loading screen's item picture, cut out of a sheet) vanished.
    if (ui_phase && !color && depth && depth != ui_depth && ui_color) {
        const auto& depth_image = texture_cache.GetImage(depth);
        const auto& ui_target = texture_cache.GetImage(ui_color);
        if (depth_image.info.size.width == ui_target.info.size.width &&
            depth_image.info.size.height == ui_target.info.size.height) {
            EnsureUiResources(ui_width, ui_height, ui_format, depth_image.info.pixel_format);
            PrepareUiDepth(depth);
            ui_depth = depth;
        }
        return;
    }
    if (!Scaled() || !color) {
        return;
    }
    if (ui_phase) {
        // A UI movie can start without stencil and enable it for later text/masks.
        if (color == ui_color && depth && depth != ui_depth) {
            EnsureUiResources(ui_width, ui_height, ui_format,
                              texture_cache.GetImage(depth).info.pixel_format);
            PrepareUiDepth(depth);
            ui_depth = depth;
        }
        return;
    }
    const auto& image = texture_cache.GetImage(color);
    const bool movie = vs_hash == ui_trigger_vs || UiComposition::MovieShader(vs_hash) ||
                       (scaled_session && native_viewport);
    if (ui_trace_on && ui_trace.size() < 1500) {
        ui_trace += fmt::format(" [vs {:x} {}x{} fmt {} movie {} display {}]", vs_hash,
                                image.info.size.width, image.info.size.height,
                                u32(image.info.pixel_format), movie ? 1 : 0,
                                FrameCapture::IsDisplayBuffer(image.info.guest_address) ? 1 : 0);
    }
    const bool ui_draw = movie &&
        (image.info.pixel_format == vk::Format::eR8G8B8A8Unorm ||
         image.info.pixel_format == vk::Format::eR8G8B8A8Srgb) &&
        RenderTarget(image.info.size.width, image.info.size.height) &&
        !FrameCapture::IsDisplayBuffer(image.info.guest_address);
    switch (UiComposition::Choose(Scaled(), ui_draw,
                                  camera_motion.Ready() && ldr_target == color,
                                  Active() && !failed)) {
    case UiComposition::Background::None:
        return;
    case UiComposition::Background::Temporal:
        RunScaled();
        if (ui_phase) {
            if (on_upscaled) {
                on_upscaled(vk::Image(ui_image), ui_width, ui_height);
            }
            return;
        }
        // A failed FSR dispatch must not force UI text back to the scene resolution.
        [[fallthrough]];
    case UiComposition::Background::Copy:
        RunUiOnly(color, depth);
        return;
    }
}

vk::ImageView TemporalUpscaler::Mirror(vk::Image image, std::vector<MirrorView>& views,
                                       vk::Format format, vk::ComponentMapping mapping) {
    for (const auto& entry : views) {
        if (entry.format == format && entry.mapping == mapping) {
            return *entry.view;
        }
    }
    const vk::ImageViewUsageCreateInfo usage{
        .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eColorAttachment,
    };
    auto view = Check(instance.GetDevice().createImageViewUnique({
        .pNext = &usage,
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .components = mapping,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    const vk::ImageView handle = *view;
    views.push_back({format, mapping, std::move(view)});
    return handle;
}

bool TemporalUpscaler::RedirectColor(VideoCore::ImageId color,
                                     const VideoCore::ImageViewInfo& view_info, Target& target) {
    if (ui_phase && color == ui_color) {
        // The first draw over the finished scene (after the game's full-screen post processing,
        // compute shaders): what is there now is the frame without its UI.
        if (!hudless_taken) {
            hudless_taken = true;
            if (frame_generation) {
                frame_generation->CaptureHudless(vk::Image(ui_image));
            }
        }
        target = {Mirror(vk::Image(ui_image), ui_views, view_info.format, {}),
                  vk::ImageLayout::eGeneral, ui_width, ui_height, true};
        return true;
    }
    const auto& image = texture_cache.GetImage(color);
    const VAddr address = image.info.guest_address;
    if (!FrameCapture::IsDisplayBuffer(address)) {
        display_redirect = false;
        return false;
    }
    std::scoped_lock lock{display_mutex};
    auto& display = displays[address];
    if (!display_redirect) {
        display.valid = false; // not upscaled: the presenter shows the guest buffer
        return false;
    }
    if (!display.image || display.format != image.info.pixel_format ||
        display.width != ui_width || display.height != ui_height) {
        const auto device = instance.GetDevice();
        scheduler.Finish();
        display.views.clear();
        std::printf("Display: host buffer %ux%u (live)\n", ui_width, ui_height);
        display.format = image.info.pixel_format;
        display.width = ui_width;
        display.height = ui_height;
        display.image = VideoCore::HostImage(device, instance.GetAllocator());
        display.image.Create(vk::ImageCreateInfo{
            .flags = vk::ImageCreateFlagBits::eMutableFormat,
            .imageType = vk::ImageType::e2D,
            .format = display.format,
            .extent = {ui_width, ui_height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                     vk::ImageUsageFlagBits::eTransferSrc,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        display.views.clear();
    }
    scheduler.EndRendering();
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eNone,
        .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        // The guest render pass uses LOAD, even for a full-screen display copy.
        // Its attachment load must wait for this layout transition as well.
        .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
                         vk::AccessFlagBits2::eColorAttachmentWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = vk::Image(display.image),
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    scheduler.Record([b](vk::CommandBuffer cmd) {
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
    });
    display.valid = true;
    target = {Mirror(vk::Image(display.image), display.views, view_info.format, {}),
              vk::ImageLayout::eGeneral, ui_width, ui_height};
    return true;
}

bool TemporalUpscaler::RedirectDepth(VideoCore::ImageId depth, Target& target) {
    if (!ui_phase || depth != ui_depth) {
        return false;
    }
    target = {*ui_depth_view, vk::ImageLayout::eGeneral, ui_width, ui_height, true};
    return true;
}

bool TemporalUpscaler::IsUiColorAddress(VAddr address) const {
    return ui_color && texture_cache.GetImage(ui_color).info.guest_address == address;
}

bool TemporalUpscaler::RedirectSampled(VideoCore::ImageId image,
                                       const VideoCore::ImageViewInfo& info, vk::ImageView& view,
                                       vk::ImageLayout& layout) {
    if (!display_redirect || image != ui_color) {
        return false;
    }
    if (!ui_read_barrier) {
        ui_read_barrier = true;
        scheduler.EndRendering();
        const vk::ImageMemoryBarrier2 b{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eGeneral,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = vk::Image(ui_image),
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        scheduler.Record([b](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
        });
    }
    view = Mirror(vk::Image(ui_image), ui_views, info.format, info.mapping);
    layout = vk::ImageLayout::eGeneral;
    return true;
}

void TemporalUpscaler::CaptureFrameGenScene(vk::Image depth_image, vk::Format depth_format, u32 w,
                                            u32 h, u32 ow, u32 oh, float frame_ms, bool frame_reset) {
    if (!frame_generation || !frame_generation->Wanted()) {
        return;
    }
    const float sign = BbToggle::Disabled(1u << 26) ? -1.0f : 1.0f;
    FrameGeneration::Scene scene{
        .color = vk::Image(ui_image),
        .color_format = ui_format,
        .width = ow,
        .height = oh,
        .depth = depth_image,
        .depth_format = depth_format,
        .motion = vk::Image(motion_image),
        .motion_view = *motion_view,
        .render_width = w,
        .render_height = h,
        .jitter_x = sign * jitter[0],
        .jitter_y = sign * jitter[1],
        .frame_ms = frame_ms,
        .camera_near = camera_motion.Near(),
        .camera_far = 3000.0f,
        .vertical_fov = camera_motion.VerticalFov(),
        .reset = frame_reset,
        .camera_position = {},
        .camera_up = {},
        .camera_right = {},
        .camera_forward = {},
    };
    camera_motion.CameraVectors(scene.camera_position, scene.camera_up, scene.camera_right,
                                scene.camera_forward);
    // The cameras as clip-space matrices (DLSS Frame Generation separates the camera's motion
    // from the objects' with them; a camera that stood still smeared the generated frames).
    scene.has_matrices = camera_motion.ClipTransforms(scene.view_to_clip, scene.clip_to_prev_clip,
                                                      scene.prev_clip_to_clip);
    frame_generation->CaptureScene(scene);
}

void TemporalUpscaler::OnDisplayPassStart(VAddr address) {
    // Periodically clean up stale proxy render targets to prevent VRAM leak during level transitions.
    static u64 last_cleanup_frame = 0;
    const u64 current_frame = BbStats::frame_number.load(std::memory_order_relaxed);
    if (current_frame - last_cleanup_frame >= 60) { // ~1 second at 60 FPS
        scene_targets.CleanupStaleEntries();
        last_cleanup_frame = current_frame;
    }
    // Cinematics can finish the LDR scene without drawing a Scaleform movie/HUD. The camera
    // already uses the output's aspect, so presenting the native 16:9 buffer would squeeze it.
    // Finish that scene at the display-copy boundary, while its camera and depth still exist.
    // The display pass then reads the same output-size image as frames which have UI.
    static const bool finish_without_ui = !bb_env_off("BB_SCENE_WITHOUT_UI");
    if (finish_without_ui && Scaled() && !ui_phase && ldr_target && camera_motion.Ready()) {
        if (Active() && !failed) {
            RunScaled();
        }
        if (!ui_phase) {
            RunUiOnly(ldr_target, camera_motion.Depth());
        }
        if (ui_phase) {
            static u64 count = 0;
            if (++count <= 3 || (count & (count - 1)) == 0) {
                std::printf("Scene output: frame without UI completed at %ux%u (%llu so far)\n",
                            ui_width, ui_height, static_cast<unsigned long long>(count));
            }
        }
    }
    if (frame_generation && ui_phase && ui_image) {
        // After the display pass's draw (RunPendingInterpolation): it reads the same image.
        pending_interpolation = vk::Image(ui_image);
        pending_display = address;
    }
}

bool TemporalUpscaler::FrameGenerationWanted() const {
    return frame_generation && frame_generation->Wanted();
}

void TemporalUpscaler::RunPendingInterpolation() {
    const vk::Image image = pending_interpolation;
    pending_interpolation = vk::Image{};
    if (image && frame_generation) {
        frame_generation->Interpolate(image, pending_display);
    }
}

bool TemporalUpscaler::DisplayOverride(VAddr address, Display& display) {
    std::scoped_lock lock{display_mutex};
    const auto it = displays.find(address);
    if (it == displays.end() || !it->second.valid) {
        return false;
    }
    display = {vk::Image(it->second.image), it->second.format, it->second.width, it->second.height};
    if (PresentDumpDue()) {
        DumpPresented(display.image, display.width, display.height, display.format);
    }
    return true;
}

} // namespace Vulkan
