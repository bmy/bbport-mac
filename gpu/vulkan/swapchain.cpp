// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "vulkan/swapchain.h"
#include "shim/bb_env.h"
#include "render/vk_low_latency.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <limits>

#include "core/emulator_settings.h"
#include "sdl_window.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "bbport_timeline.h"

namespace Vulkan {

namespace {

/// Rec.2020 with the PQ curve: what HDR games output. Other HDR spaces would need conversions.
constexpr vk::SurfaceFormatKHR HdrFormat{
    .format = vk::Format::eA2B10G10R10UnormPack32,
    .colorSpace = vk::ColorSpaceKHR::eHdr10St2084EXT,
};

vk::PresentModeKHR RequestedPresentMode() {
    const auto name = EmulatorSettings.GetPresentMode();
    if (name == "Fifo") {
        return vk::PresentModeKHR::eFifo;
    }
    if (name == "Immediate") {
        return vk::PresentModeKHR::eImmediate;
    }
    if (name == "FifoRelaxed") {
        return vk::PresentModeKHR::eFifoRelaxed;
    }
    if (name != "Mailbox") {
        LOG_ERROR(Render_Vulkan, "Unknown present mode {}, using Mailbox", name);
    }
    return vk::PresentModeKHR::eMailbox;
}

} // namespace

Swapchain::Swapchain(const Instance& instance_, const Frontend::WindowSDL& window)
    : instance{instance_}, surface{CreateSurface(instance.GetInstance(), window)} {
    ChooseFormat();
    ChoosePresentMode();
    Create(u32(window.GetWidth()), u32(window.GetHeight()));
}

Swapchain::~Swapchain() {
    DestroyResources();
    if (swapchain) {
        LowLatency::Get().SwapchainDestroying(swapchain);
        instance.GetDevice().destroySwapchainKHR(swapchain);
    }
    instance.GetInstance().destroySurfaceKHR(surface);
}

void Swapchain::ChooseFormat() {
    const auto formats =
        Check<"query surface formats">(instance.GetPhysicalDevice().getSurfaceFormatsKHR(surface));
    supports_hdr = EmulatorSettings.IsHdrAllowed() && std::ranges::contains(formats, HdrFormat);
    // A single undefined format: the surface takes any.
    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
        sdr_format = {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
        return;
    }
    const auto it = std::ranges::find_if(formats, [](const vk::SurfaceFormatKHR& f) {
        return f.format == vk::Format::eR8G8B8A8Unorm || f.format == vk::Format::eB8G8R8A8Unorm;
    });
    ASSERT_MSG(it != formats.end(), "the surface has no RGBA8 or BGRA8 format");
    sdr_format = *it;
}

void Swapchain::ChoosePresentMode() {
    const auto [result, modes] = instance.GetPhysicalDevice().getSurfacePresentModesKHR(surface);
    present_mode = RequestedPresentMode();
#ifdef _WIN32
    // NVIDIA on Windows: Mailbox held vkAcquireNextImageKHR ~100 ms whenever the GPU was busy,
    // then showed three frames at once (GTX 1050 laptop: presents 3 in a row, gaps of 100-160 ms);
    // FIFO showed every frame as it came at the same frame rate. BB_NVIDIA_MAILBOX=1 keeps Mailbox.
    if (present_mode == vk::PresentModeKHR::eMailbox && instance.GetVendorID() == 0x10DE &&
        !bb_env("BB_NVIDIA_MAILBOX")) {
        std::printf("Swapchain: Mailbox -> FIFO on NVIDIA (Windows: Mailbox shows frames in "
                    "bursts; BB_NVIDIA_MAILBOX=1 keeps it)\n");
        present_mode = vk::PresentModeKHR::eFifo;
    }
#endif
    if (result != vk::Result::eSuccess || !std::ranges::contains(modes, present_mode)) {
        // FIFO is always there. FIFO Relaxed (not on Mesa's Wayland surfaces) falls back to
        // Mailbox, which also presents a late frame at once instead of locking to half the
        // refresh rate as FIFO does.
        const bool mailbox = result == vk::Result::eSuccess &&
                             present_mode == vk::PresentModeKHR::eFifoRelaxed &&
                             std::ranges::contains(modes, vk::PresentModeKHR::eMailbox);
        const auto fallback = mailbox ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
        LOG_WARNING(Render_Vulkan, "Present mode {} is not supported, using {}",
                    vk::to_string(present_mode), vk::to_string(fallback));
        present_mode = fallback;
    }
}

void Swapchain::QuerySurface() {
    const auto caps = Check<"query surface capabilities">(
        instance.GetPhysicalDevice().getSurfaceCapabilitiesKHR(surface));
    extent = caps.currentExtent;
    if (extent.width == std::numeric_limits<u32>::max()) {
        // The surface takes the swapchain's size.
        extent.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    // One image more than the minimum: one to draw into while the others are shown.
    // BB_SWAPCHAIN_IMAGES=N: N images (within the surface's limits).
    static const u32 wanted_images = [] {
        const char* env = bb_env("BB_SWAPCHAIN_IMAGES");
        return env ? u32(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    image_count = wanted_images ? std::max(wanted_images, caps.minImageCount) : caps.minImageCount + 1;
    if (caps.maxImageCount > 0) {
        image_count = std::min(image_count, caps.maxImageCount);
    }
    transform = caps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
                    ? vk::SurfaceTransformFlagBitsKHR::eIdentity
                    : caps.currentTransform;
    composite_alpha = caps.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
                          ? vk::CompositeAlphaFlagBitsKHR::eOpaque
                          : vk::CompositeAlphaFlagBitsKHR::eInherit;
}

void Swapchain::Create(u32 width_, u32 height_) {
    width = width_;
    height = height_;
    needs_recreation = false;

    // A minimised window has a 0x0 surface (issue #31) and no swapchain can have that extent:
    // the old one stays (presents are skipped) until the window is back.
    QuerySurface();
    zero_extent = extent.width == 0 || extent.height == 0;
    if (zero_extent) {
        needs_recreation = true;
        return;
    }

    // The old swapchain is retired through oldSwapchain, not destroyed first: NVIDIA's Wayland
    // WSI keeps a tearing controller per swapchain and rejected a fresh one on the same surface
    // ("Surface already has a tearing controller", SurfaceLost).
    DestroyResources();
    const vk::Device device = instance.GetDevice();
    const std::array families = {instance.GetGraphicsQueueFamilyIndex(),
                                 instance.GetPresentQueueFamilyIndex()};
    const bool exclusive = families[0] == families[1];
    const auto format = use_hdr ? HdrFormat : sdr_format;
    const vk::SwapchainKHR old = swapchain;
    // Low latency (NVIDIA Reflex, render/vk_low_latency): the swapchain takes the driver's
    // latency mode when the device has VK_NV_low_latency2.
    const vk::SwapchainLatencyCreateInfoNV latency{.latencyModeEnable = true};
    swapchain = Check<"create swapchain">(device.createSwapchainKHR({
        .pNext = instance.HasLowLatency() ? &latency : nullptr,
        .surface = surface,
        .minImageCount = image_count,
        .imageFormat = format.format,
        .imageColorSpace = format.colorSpace,
        .imageExtent = extent,
        .imageArrayLayers = 1,
        .imageUsage = vk::ImageUsageFlagBits::eColorAttachment |
                      vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .imageSharingMode = exclusive ? vk::SharingMode::eExclusive : vk::SharingMode::eConcurrent,
        .queueFamilyIndexCount = exclusive ? 1u : 2u,
        .pQueueFamilyIndices = families.data(),
        .preTransform = transform,
        .compositeAlpha = composite_alpha,
        .presentMode = present_mode,
        .clipped = true,
        .oldSwapchain = old,
    }));
    if (old) {
        LowLatency::Get().SwapchainDestroying(old);
        device.destroySwapchainKHR(old);
    }
    LowLatency::Get().SwapchainCreated(swapchain);

    images = Check<"get swapchain images">(device.getSwapchainImagesKHR(swapchain));
    image_count = u32(images.size());
    static vk::PresentModeKHR logged_mode{};
    static u32 logged_images = 0;
    if (logged_mode != present_mode || logged_images != image_count) {
        logged_mode = present_mode;
        logged_images = image_count;
        std::printf("Swapchain: %s, %u images, %ux%u\n", vk::to_string(present_mode).c_str(),
                    image_count, extent.width, extent.height);
    }
    views.resize(image_count);
    acquired.resize(image_count);
    present_ready.resize(image_count);
    for (u32 i = 0; i < image_count; ++i) {
        views[i] = Check<"create swapchain view">(device.createImageView({
            .image = images[i],
            .viewType = vk::ImageViewType::e2D,
            .format = format.format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        }));
        acquired[i] = Check<"create swapchain semaphore">(device.createSemaphore({}));
        present_ready[i] = Check<"create swapchain semaphore">(device.createSemaphore({}));
        SetObjectName(device, images[i], "swapchain image {}", i);
        SetObjectName(device, views[i], "swapchain view {}", i);
        SetObjectName(device, acquired[i], "swapchain acquired {}", i);
        SetObjectName(device, present_ready[i], "swapchain present ready {}", i);
    }
    frame_index = 0;
}

void Swapchain::DestroyResources() {
    const vk::Device device = instance.GetDevice();
    // The semaphores and views may still be in use by the last presents.
    if (const auto result = device.waitIdle(); result != vk::Result::eSuccess) {
        LOG_WARNING(Render_Vulkan, "Device idle wait before swapchain changes: {}",
                    vk::to_string(result));
    }
    for (const auto view : views) {
        device.destroyImageView(view);
    }
    for (const auto semaphore : acquired) {
        device.destroySemaphore(semaphore);
    }
    for (const auto semaphore : present_ready) {
        device.destroySemaphore(semaphore);
    }
    views.clear();
    acquired.clear();
    present_ready.clear();
}

void Swapchain::Recreate(u32 width_, u32 height_) {
    LOG_DEBUG(Render_Vulkan, "Recreating the swapchain: {}x{}, HDR {}", width_, height_, use_hdr);
    Create(width_, height_);
}

void Swapchain::SetHDR(bool hdr) {
    if (use_hdr == hdr) {
        return;
    }
    use_hdr = hdr;
    Recreate(width, height);
}

bool Swapchain::AcquireNextImage() {
    const vk::Result result = instance.GetDevice().acquireNextImageKHR(
        swapchain, std::numeric_limits<u64>::max(), acquired[frame_index], VK_NULL_HANDLE,
        &image_index);
    switch (result) {
    case vk::Result::eSuccess:
        break;
    case vk::Result::eSuboptimalKHR:
    case vk::Result::eErrorSurfaceLostKHR:
    case vk::Result::eErrorOutOfDateKHR:
    case vk::Result::eErrorUnknown:
        needs_recreation = true;
        break;
    default:
        Check<"acquire a swapchain image">(result);
    }
    return !needs_recreation;
}

bool Swapchain::Present(u64 present_id) {
    // The frame's ID (VK_KHR_present_id): the driver pairs the present with the frame's
    // submissions and markers (low latency). Zero: none (generated and blank frames).
    const vk::PresentIdKHR id_info{.swapchainCount = 1, .pPresentIds = &present_id};
    const vk::PresentInfoKHR info{
        .pNext = present_id != 0 && instance.HasPresentId() && instance.HasLowLatency() ? &id_info
                                                                                        : nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &present_ready[image_index],
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &image_index,
    };
    const auto present_start = BbTimeline::Now();
    const vk::Result result = instance.GetPresentQueue().presentKHR(info);
    BbTimeline::Note(BbTimeline::Presented, 0, u64(BbTimeline::Now() - present_start));
    if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR) {
        needs_recreation = true;
    } else {
        Check<"present">(result);
    }
    frame_index = (frame_index + 1) % image_count;
    return !needs_recreation;
}

} // namespace Vulkan
