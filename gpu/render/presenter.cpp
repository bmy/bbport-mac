// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/presenter.h"
#include "metal/metal_interop_test.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <limits>

#include <vk_mem_alloc.h>

#include "bbport_game_menu.h"
#include "bbport_overlay.h"
#include "bbport_timeline.h"
#include "bbport_toggles.h"
#include "core/emulator_settings.h"
#include "sdl_window.h"
#include "textures/image.h"
#include "textures/texture_cache.h"
#include "upscale/vk_frame_generation.h"
#include "upscale/vk_temporal_upscaler.h"
#include "render/rasterizer.h"
#include "vulkan/platform.h"

namespace Vulkan {

namespace {

constexpr vk::ImageSubresourceRange ColorRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

/// How the display buffer is read: its format as a view (the 10-bit sRGB one has no sRGB
/// Vulkan format: it is decoded in the encode pass).
/// The shown frame's aspect (bbgpu_frame_aspect): any thread.
std::atomic<float> shown_aspect{16.0f / 9.0f};

vk::Format FrameViewFormat(BbVideoOut::PixelFormat format) {
    switch (format) {
    case BbVideoOut::PixelFormat::A8B8G8R8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case BbVideoOut::PixelFormat::A8R8G8B8Srgb:
        return vk::Format::eB8G8R8A8Srgb;
    case BbVideoOut::PixelFormat::A2R10G10B10:
    case BbVideoOut::PixelFormat::A2R10G10B10Srgb:
    case BbVideoOut::PixelFormat::A2R10G10B10Bt2020Pq:
        return vk::Format::eA2R10G10B10UnormPack32;
    default:
        UNREACHABLE_MSG("display buffer format {:#x}", u32(format));
    }
}

/// `frame` fitted into `target` keeping its aspect, centered.
vk::Rect2D Letterbox(vk::Extent2D frame, vk::Extent2D target) {
    const float frame_aspect = float(frame.width) / float(frame.height);
    const float target_aspect = float(target.width) / float(target.height);
    vk::Extent2D size = target;
    if (frame_aspect > target_aspect) {
        size.height = u32(float(target.width) / frame_aspect);
    } else {
        size.width = u32(float(target.height) * frame_aspect);
    }
    return {{s32(target.width - size.width) / 2, s32(target.height - size.height) / 2}, size};
}

/// BB_FRAMES_AHEAD (default 1): how many guest frames the command processor thread may run
/// ahead of the GPU (0: unbounded).
u32 FramesAhead() {
    static const u32 value = [] {
        const char* env = std::getenv("BB_FRAMES_AHEAD");
        return env ? u32(std::max(0, std::atoi(env))) : 1u;
    }();
    return value;
}

void WaitFence(vk::Device device, vk::Fence fence) {
    while (true) {
        const vk::Result result = device.waitForFences(fence, true, std::numeric_limits<u64>::max());
        if (result == vk::Result::eSuccess) {
            return;
        }
        if (result != vk::Result::eTimeout) {
            Check<"wait for a frame">(result);
        }
    }
}

} // namespace

Presenter::Presenter(Frontend::WindowSDL& window_, BbCp::CommandProcessor* cp_)
    : window{window_},
      instance{window, EmulatorSettings.GetGpuId(), EmulatorSettings.IsVkValidationEnabled(),
               EmulatorSettings.IsVkCrashDiagnosticEnabled()},
      cp{cp_}, draw_scheduler{instance, true}, present_scheduler{instance},
      swapchain{instance, window}, runtime{instance, draw_scheduler},
      rasterizer{std::make_unique<Rasterizer>(instance, draw_scheduler, runtime, cp)},
      texture_cache{rasterizer->GetTextureCache()},
      encode_pass{instance.GetDevice(), swapchain.GetSurfaceFormat().format} {
    const vk::Device device = instance.GetDevice();
    const u32 count = swapchain.GetImageCount();
    frames.resize(count);
    for (u32 i = 0; i < count; ++i) {
        frames[i].id = u8(i);
        frames[i].present_done = Check<"create a frame fence">(
            device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}));
        free_frames.push_back(&frames[i]);
    }
    BbOverlay::Init(instance, swapchain.GetSurfaceFormat().format, count);
    BbMetal::RunInteropTestIfAsked(instance); // BB_METAL_INTEROP_TEST=1 (macOS)
    BbMetal::RunFrameTestIfAsked(instance);   // BB_METAL_FRAME_TEST=1 (macOS)
}

Presenter::~Presenter() {
    draw_scheduler.Finish();
    present_scheduler.Finish();
    const vk::Device device = instance.GetDevice();
    for (auto& frame : frames) {
        device.destroyImageView(frame.image_view);
        if (frame.image) {
            vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
        }
        device.destroyFence(frame.present_done);
    }
}

void Presenter::SetHDR(bool enable) {
    if (!IsHDRSupported()) {
        return;
    }
    if (swapchain.GetHDR() != enable) {
        // Changing HDR recreates the swapchain and waits for the device: exclude queue submits.
        std::scoped_lock lock{Scheduler::submit_mutex};
        swapchain.SetHDR(enable);
    }
    hdr.store(enable, std::memory_order_relaxed);
}

VideoCore::Image& Presenter::RegisterVideoOutSurface(
    const BbVideoOut::BufferAttributeGroup& attribute, VAddr cpu_address) {
    auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
    const auto image_id = texture_cache.FindImage(desc);
    auto& image = texture_cache.GetImage(image_id);
    image.usage.vo_surface = 1u;
    return image;
}

void Presenter::RecreateFrame(Frame& frame, u32 width, u32 height) {
    const vk::Device device = instance.GetDevice();
    device.destroyImageView(frame.image_view);
    if (frame.image) {
        vmaDestroyImage(instance.GetAllocator(), frame.image, frame.allocation);
    }
    const vk::Format format = swapchain.GetSurfaceFormat().format;
    const VkImageCreateInfo image_info = vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
    };
    const VmaAllocationCreateInfo alloc_info{
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    VkImage image{};
    const VkResult result = vmaCreateImage(instance.GetAllocator(), &image_info, &alloc_info,
                                           &image, &frame.allocation, nullptr);
    Check<"allocate a frame image">(vk::Result(result));
    frame.image = image;
    frame.image_view = Check<"create a frame view">(device.createImageView({
        .image = frame.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = ColorRange,
    }));
    SetObjectName(device, frame.image, "frame {}", frame.id);
    // Frame generation's frame (made with the first one generated: most sessions never need it).
    device.destroyImageView(frame.interp_view);
    if (frame.interp_image) {
        vmaDestroyImage(instance.GetAllocator(), frame.interp_image, frame.interp_allocation);
    }
    frame.interp_image = vk::Image{};
    frame.interp_view = vk::ImageView{};
    frame.has_interp = false;
    frame.width = width;
    frame.height = height;
    frame.is_hdr = swapchain.GetHDR();
}

Frame* Presenter::GetRenderFrame() {
    Frame* frame;
    {
        std::unique_lock lock{free_mutex};
        free_cv.wait(lock, [this] { return !free_frames.empty(); });
        frame = free_frames.front();
        free_frames.pop_front();
    }
    // Its last present has finished: the image, its FSR slot and the fence are free.
    WaitFence(instance.GetDevice(), frame->present_done);
    const u32 width = expected_width.load(std::memory_order_relaxed);
    const u32 height = expected_height.load(std::memory_order_relaxed);
    if (frame->width != width || frame->height != height ||
        frame->is_hdr != swapchain.GetHDR()) {
        RecreateFrame(*frame, width, height);
    }
    return frame;
}

void Presenter::ReleaseFrame(Frame* frame) {
    if (frame == &blank_frame) {
        return;
    }
    {
        std::scoped_lock lock{free_mutex};
        free_frames.push_back(frame);
    }
    free_cv.notify_one();
}

Frame* Presenter::PrepareFrame(const BbVideoOut::BufferAttributeGroup& attribute,
                               VAddr cpu_address) {
    BbTimeline::Note(BbTimeline::PipeTask, 4, draw_scheduler.CurrentTick());
    // Scaled upscaler presets: the output-size display buffer drawn by the port.
    TemporalUpscaler::Display display{};
    auto& upscaler = rasterizer->GetUpscaler();
    const bool upscaled = upscaler.DisplayOverride(cpu_address, display);
    VideoCore::ImageId image_id{};
    if (!upscaled) {
        auto desc = VideoCore::TextureCache::ImageDesc{attribute, cpu_address};
        image_id = texture_cache.FindImage(desc);
        texture_cache.UpdateImage(image_id);
    }

    Frame* frame = GetRenderFrame();

    // Recorded like draws, on the recording threads: CommandBuffer() would first wait for them
    // to record everything handed over before and then submit from this thread.
    draw_scheduler.EndRendering();

    VideoCore::ImageViewInfo view_info{};
    view_info.format = FrameViewFormat(attribute.attrib.pixel_format);
    // The frame is opaque: alpha would blend it with the overlay.
    view_info.mapping.a = vk::ComponentSwizzle::eOne;
    vk::ImageView source{};
    vk::Extent2D source_size{};
    if (upscaled) {
        const vk::Device device = instance.GetDevice();
        source = Check(device.createImageView({
            .image = display.image,
            .viewType = vk::ImageViewType::e2D,
            .format = view_info.format,
            .components = view_info.mapping,
            .subresourceRange = ColorRange,
        }));
        draw_scheduler.DeferOperation([device, source] { device.destroyImageView(source); });
        source_size = vk::Extent2D{display.width, display.height};
        draw_scheduler.Record([image = display.image](vk::CommandBuffer cmdbuf) {
            const vk::ImageMemoryBarrier2 to_read{
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = image,
                .subresourceRange = ColorRange,
            };
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_read});
        });
    } else {
        auto& image = texture_cache.GetImage(image_id);
        source = *image.FindView(view_info).image_view;
        source_size = vk::Extent2D{image.info.size.width, image.info.size.height};
        // BB_PRESENT_DUMP_TRIGGER for the game's own display buffer too.
        if (upscaler.PresentDumpDue()) {
            runtime.Transit(&image, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eTransfer,
                            vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
            upscaler.DumpPresented(image.GetImage(), source_size.width, source_size.height,
                                   view_info.format);
        }
        runtime.Transit(&image, vk::ImageLayout::eShaderReadOnlyOptimal,
                        vk::PipelineStageFlagBits2::eFragmentShader |
                            vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
    }
    expected_ratio.store(float(source_size.width) / float(source_size.height),
                         std::memory_order_relaxed);
    shown_aspect.store(float(source_size.width) / float(source_size.height),
                       std::memory_order_relaxed);

    const EncodePass::Settings encode{
        .gamma = gamma.load(std::memory_order_relaxed),
        .hdr = hdr.load(std::memory_order_relaxed),
        .srgb_input = attribute.attrib.pixel_format == BbVideoOut::PixelFormat::A2R10G10B10Srgb,
    };
    // Numbered here, in command order; marked where the commands are recorded.
    const u32 stream = draw_scheduler.CrumbStream();
    const u32 crumb = Breadcrumbs::Enabled()
                          ? Breadcrumbs::Note(stream, {.name = "presenter encode"})
                          : 0;
    // The encode pass holds no state: frames of different submissions may be recorded on two
    // threads at once.
    draw_scheduler.Record([this, frame, source, encode, stream, crumb](vk::CommandBuffer cmdbuf) {
        if (crumb) {
            Breadcrumbs::Mark(cmdbuf, stream, crumb, false);
        }
        const vk::Extent2D size{frame->width, frame->height};
        // The frame's last reader (the present blit) is done: the fence was waited for.
        const vk::ImageMemoryBarrier2 to_attachment{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = frame->image,
            .subresourceRange = ColorRange,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_attachment});
        encode_pass.Render(cmdbuf, source, frame->image_view, size, encode);
        // Read by the present blit (or a dump) in the general layout, after a semaphore wait.
        const vk::ImageMemoryBarrier2 to_general{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = frame->image,
            .subresourceRange = ColorRange,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_general});
        if (crumb) {
            Breadcrumbs::Mark(cmdbuf, stream, crumb, true);
        }
    });

    // Frame generation: the frame interpolated before this one, encoded the same way.
    frame->has_interp = false;
    FrameGeneration::Output interp{};
    if (upscaled && upscaler.GetFrameGeneration() &&
        upscaler.GetFrameGeneration()->TakeOutput(cpu_address, interp) &&
        interp.width == display.width && interp.height == display.height) {
        EncodeInterpolated(frame, interp.image, interp.format, interp.width, interp.height,
                           view_info.format, encode);
    }

    BbStats::frame_number.fetch_add(1, std::memory_order_relaxed);
    frame->ready_semaphore = draw_scheduler.GetWorkSemaphore()->Handle();
    frame->ready_tick = draw_scheduler.CurrentTick();
    SubmitInfo info{};
    draw_scheduler.NoteFrameEndSubmission();
    draw_scheduler.Flush(info);
    BbTimeline::Note(BbTimeline::PipeTask, 5, frame->ready_tick);

    // The command processor thread runs at most FramesAhead() guest frames ahead of the GPU: it
    // waits here for the frame that many flips back. When the GPU is the bottleneck it finishes
    // frames at an even rate; unbounded, the thread ran ahead and then blocked wherever a
    // resource ran out, so flips (and the guest's frame timing) came in bursts.
    if (const u32 ahead = FramesAhead()) {
        recent_frame_ticks.push_back(frame->ready_tick);
        if (recent_frame_ticks.size() > ahead) {
            const u64 tick = recent_frame_ticks.front();
            recent_frame_ticks.pop_front();
            BbStats::WaitTimer timer{BbStats::present_wait_ns};
            draw_scheduler.Wait(tick);
        }
    }
    return frame;
}

Frame* Presenter::PrepareBlankFrame(bool present_thread) {
    // Presented as the black bars alone: nothing to render. A blank flip of the guest still
    // ends its submission, as any flip does: the guest may wait for work recorded before it.
    if (!present_thread) {
        draw_scheduler.EndRendering();
        SubmitInfo info{};
        draw_scheduler.Flush(info);
    }
    return &blank_frame;
}

void Presenter::WaitRendered(const Frame* frame) {
    if (!frame || !frame->ready_semaphore) {
        return;
    }
    const vk::SemaphoreWaitInfo info{
        .semaphoreCount = 1,
        .pSemaphores = &frame->ready_semaphore,
        .pValues = &frame->ready_tick,
    };
    // A lost device ends the wait (the presenter reports it); a timeout waits again.
    while (instance.GetDevice().waitSemaphores(&info, 1'000'000'000) == vk::Result::eTimeout) {
    }
}

void Presenter::SetExpectedSize(u32 width, u32 height) {
    if (width == 0 || height == 0) {
        return; // minimised: keep the last size (a 0x0 frame image cannot be made, issue #31)
    }
    const float ratio = expected_ratio.load(std::memory_order_relaxed);
    if (float(width) / float(height) > ratio) {
        width = u32(float(height) * ratio);
    } else {
        height = u32(float(width) / ratio);
    }
    expected_width.store(width, std::memory_order_relaxed);
    expected_height.store(height, std::memory_order_relaxed);
}

void Presenter::EncodeInterpolated(Frame* frame, vk::Image image, vk::Format format, u32 width,
                                   u32 height, vk::Format display_view_format,
                                   const EncodePass::Settings& encode) {
    const vk::Device device = instance.GetDevice();
    if (!frame->interp_image) {
        const VkImageCreateInfo image_info = vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = swapchain.GetSurfaceFormat().format,
            .extent = {frame->width, frame->height, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        };
        const VmaAllocationCreateInfo alloc_info{
            .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        };
        VkImage created{};
        if (vmaCreateImage(instance.GetAllocator(), &image_info, &alloc_info, &created,
                           &frame->interp_allocation, nullptr) != VK_SUCCESS) {
            return;
        }
        frame->interp_image = created;
        frame->interp_view = Check<"create a frame view">(device.createImageView({
            .image = frame->interp_image,
            .viewType = vk::ImageViewType::e2D,
            .format = swapchain.GetSurfaceFormat().format,
            .subresourceRange = ColorRange,
        }));
    }
    // Read as the display buffer is, with its view format: the game's display pass copies the
    // scene image's bytes as they are into a BGRA buffer, so the scene image (R8G8B8A8 by its
    // format) holds them in that order too; its own format swapped red and blue. 8-bit images are
    // made mutable (vk_frame_generation), 32-bit formats view each other.
    const bool rgba8 = format == vk::Format::eR8G8B8A8Unorm || format == vk::Format::eB8G8R8A8Unorm;
    const bool display8 = display_view_format == vk::Format::eB8G8R8A8Srgb ||
                          display_view_format == vk::Format::eR8G8B8A8Srgb;
    const vk::Format view_format = rgba8 && display8 ? display_view_format : format;
    vk::ComponentMapping mapping{};
    mapping.a = vk::ComponentSwizzle::eOne;
    const vk::ImageView source = Check(device.createImageView({
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = view_format,
        .components = mapping,
        .subresourceRange = ColorRange,
    }));
    draw_scheduler.DeferOperation([device, source] { device.destroyImageView(source); });
    draw_scheduler.Record([this, frame, image, source, encode](vk::CommandBuffer cmdbuf) {
        const std::array<vk::ImageMemoryBarrier2, 2> before{{
            {
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
                .oldLayout = vk::ImageLayout::eGeneral,
                .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .image = image,
                .subresourceRange = ColorRange,
            },
            {
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .image = frame->interp_image,
                .subresourceRange = ColorRange,
            },
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = before.data()});
        encode_pass.Render(cmdbuf, source, frame->interp_view, {frame->width, frame->height}, encode);
        const std::array<vk::ImageMemoryBarrier2, 2> after{{
            {
                .srcStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
                .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .oldLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = image,
                .subresourceRange = ColorRange,
            },
            {
                .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
                .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = frame->interp_image,
                .subresourceRange = ColorRange,
            },
        }};
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 2, .pImageMemoryBarriers = after.data()});
    });
    frame->has_interp = true;
}

void Presenter::RecordPresent(vk::CommandBuffer cmdbuf, const Frame* frame, vk::Image target,
                              vk::Extent2D extent, vk::Image source) {
    const vk::Rect2D rect =
        frame ? Letterbox({frame->width, frame->height}, extent) : vk::Rect2D{{0, 0}, {0, 0}};
    const bool bars = rect.extent != extent;
    const vk::ImageMemoryBarrier2 to_transfer{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eTransferDstOptimal,
        .image = target,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_transfer});
    if (bars) {
        const vk::ClearColorValue black{std::array{0.0f, 0.0f, 0.0f, 1.0f}};
        cmdbuf.clearColorImage(target, vk::ImageLayout::eTransferDstOptimal, black, ColorRange);
    }
    if (frame) {
        if (bars) {
            const vk::MemoryBarrier2 cleared{
                .srcStageMask = vk::PipelineStageFlagBits2::eClear,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eBlit,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            };
            cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &cleared});
        }
        const vk::ImageSubresourceLayers layers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        const vk::ImageBlit region{
            .srcSubresource = layers,
            .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                     vk::Offset3D{s32(frame->width), s32(frame->height), 1}},
            .dstSubresource = layers,
            .dstOffsets = std::array{vk::Offset3D{rect.offset.x, rect.offset.y, 0},
                                     vk::Offset3D{rect.offset.x + s32(rect.extent.width),
                                                  rect.offset.y + s32(rect.extent.height), 1}},
        };
        cmdbuf.blitImage(source ? source : frame->image, vk::ImageLayout::eGeneral, target,
                         vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);
    }

    // The port's menu and FPS counter over the frame, at display resolution.
    BbGameMenu::Poll();
    const bool overlay = BbOverlay::Visible();
    const vk::ImageMemoryBarrier2 after_transfer{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = overlay ? vk::PipelineStageFlagBits2::eColorAttachmentOutput
                                : vk::PipelineStageFlagBits2::eNone,
        .dstAccessMask = overlay ? vk::AccessFlagBits2::eColorAttachmentRead |
                                       vk::AccessFlagBits2::eColorAttachmentWrite
                                 : vk::AccessFlagBits2::eNone,
        .oldLayout = vk::ImageLayout::eTransferDstOptimal,
        .newLayout = overlay ? vk::ImageLayout::eColorAttachmentOptimal
                             : vk::ImageLayout::ePresentSrcKHR,
        .image = target,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &after_transfer});
    if (overlay) {
        BbOverlay::Render(cmdbuf, swapchain.ImageView(), extent);
        const vk::ImageMemoryBarrier2 to_present{
            .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
            .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .newLayout = vk::ImageLayout::ePresentSrcKHR,
            .image = target,
            .subresourceRange = ColorRange,
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_present});
    }
}

void Presenter::Present(Frame* frame, bool is_reusing_frame, bool) {
    const auto release = [&] {
        if (!is_reusing_frame) {
            ReleaseFrame(frame);
        }
    };
    const bool blank = frame == &blank_frame;
    // Frame generation: the interpolated frame first, the real one half a frame later (the
    // interval between real frames, smoothed).
    using Clock = std::chrono::steady_clock;
    static Clock::time_point last_real{};
    static double interval_ms = 16.667;
    const auto now = Clock::now();
    if (last_real.time_since_epoch().count() != 0) {
        const double ms = std::chrono::duration<double, std::milli>(now - last_real).count();
        if (ms > 2.0 && ms < 100.0) {
            interval_ms = interval_ms * 0.9 + ms * 0.1;
        }
    }
    last_real = now;
    // BB_FRAME_GEN_ONLY=1 (diagnostics): only the interpolated frames are shown.
    static const bool only_interp = std::getenv("BB_FRAME_GEN_ONLY") != nullptr;
    if (only_interp && !blank && frame->has_interp && frame->interp_image) {
        frame->has_interp = false;
        PresentImage(frame, frame->interp_image, true);
        release();
        return;
    }
    if (!blank && frame->has_interp && frame->interp_image) {
        frame->has_interp = false;
        static std::atomic<u64> shown{0};
        if (const u64 n = ++shown; n == 1 || n % 1000 == 0) {
            std::printf("Frame generation: %llu interpolated frames shown\n", (unsigned long long)n);
        }
        if (PresentImage(frame, frame->interp_image, false)) {
            std::this_thread::sleep_until(now + std::chrono::duration_cast<Clock::duration>(
                                                    std::chrono::duration<double, std::milli>(
                                                        std::clamp(interval_ms * 0.5, 1.0, 50.0))));
        }
    }
    PresentImage(frame, {}, true);
    release();
}

bool Presenter::PresentImage(Frame* frame, vk::Image source, bool last) {
    const u32 width = u32(window.GetWidth());
    const u32 height = u32(window.GetHeight());
    // Nothing to present to while the window is minimised (a 0x0 surface, issue #31).
    if (width == 0 || height == 0) {
        return false;
    }
    // Recreate waits for the device. Recording threads must not submit to its queues meanwhile
    // (#211); the recreation after a failed present below already uses this same mutex.
    const auto recreate = [&] {
        std::scoped_lock lock{Scheduler::submit_mutex};
        swapchain.Recreate(width, height);
    };
    // A resized window (or one minimised at the last try) needs a new swapchain.
    if (width != swapchain.GetWidth() || height != swapchain.GetHeight() ||
        !swapchain.IsPresentable()) {
        recreate();
        if (!swapchain.IsPresentable()) {
            return false;
        }
    }
    if (!swapchain.AcquireNextImage()) {
        recreate();
        if (!swapchain.AcquireNextImage()) {
            // Resized faster than the swapchain can follow: this frame is skipped.
            LOG_WARNING(Render_Vulkan, "Skipping a frame: no swapchain image");
            return false;
        }
    }

    const bool blank = frame == &blank_frame;
    const vk::Extent2D extent = swapchain.GetExtent();
    SetExpectedSize(extent.width, extent.height);
    // Reset here, not when the frame is taken: a frame skipped above keeps its fence signaled.
    if (!blank && last) {
        Check<"reset a frame fence">(instance.GetDevice().resetFences(frame->present_done));
    }

    const auto cmdbuf = present_scheduler.CommandBuffer();
    if (!blank && last) {
        DumpFinalFrameIfDue(instance, present_scheduler, cmdbuf, frame->image, frame->width,
                            frame->height, swapchain.GetSurfaceFormat().format);
    }
    // BB_SEQUENCE_DUMP_TRIGGER=<file> (frame generation checks): the next 16 images shown, the
    // interpolated ones ("interp") and the game's ("real") in the order they are presented.
    {
        static const char* trigger = std::getenv("BB_SEQUENCE_DUMP_TRIGGER");
        static int left = 0, index = 0;
        if (trigger && left == 0 && std::remove(trigger) == 0) {
            left = 16;
            index = 0;
        }
        if (left > 0 && !blank) {
            --left;
            DumpPresented(instance, present_scheduler, cmdbuf, source ? source : frame->image,
                          frame->width, frame->height, swapchain.GetSurfaceFormat().format, index++,
                          source ? "interp" : "real");
        }
    }
    RecordPresent(cmdbuf, blank ? nullptr : frame, swapchain.Image(), extent, source);

    // The frame's submission goes out from a recording thread (BB_ASYNC_SUBMIT); the present
    // one, waiting for it on the same queue, must not get there first.
    if (!blank && frame->ready_semaphore == draw_scheduler.GetWorkSemaphore()->Handle()) {
        draw_scheduler.WaitSubmitted(frame->ready_tick);
    }
    SubmitInfo info{};
    info.AddWait(swapchain.GetImageAcquiredSemaphore(), "the swapchain image");
    if (!blank) {
        info.AddWait(frame->ready_semaphore, frame->ready_tick, "the frame's draws");
        if (last) {
            info.AddSignal(frame->present_done);
        }
    }
    info.AddSignal(swapchain.GetPresentReadySemaphore());
    present_scheduler.Flush(info);
    {
        std::scoped_lock lock{Scheduler::submit_mutex};
        if (!swapchain.Present()) {
            swapchain.Recreate(width, height);
            return false;
        }
    }
    return true;
}

} // namespace Vulkan

extern "C" float bbgpu_frame_aspect(void) {
    return Vulkan::shown_aspect.load(std::memory_order_relaxed);
}
