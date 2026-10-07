// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_metalfx.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

#ifdef __APPLE__
// Only the C types of VK_EXT_external_memory_metal: vulkan.hpp is included without the Metal
// platform define everywhere, and defining it here would change its inline classes (ODR).
#include <vulkan/vulkan_metal.h>
#include "bbport_metalfx.h"
#else
namespace BbMetalFx {
class Scaler {};
} // namespace BbMetalFx
#endif

namespace Vulkan {

#if defined(__APPLE__) && defined(VK_EXT_external_memory_metal)

namespace {

constexpr VkExternalMemoryHandleTypeFlagBits HeapHandle =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT;

bool EnvOn(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}

} // namespace

MetalFxUpscaler::MetalFxUpscaler(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    if (!instance.IsExternalMemoryMetalEnabled()) {
        Fail("VK_EXT_external_memory_metal is missing (MoltenVK?); MetalFX needs KosmicKrisp",
             true);
        return;
    }
    const vk::PhysicalDeviceExternalBufferInfo info{
        .usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        .handleType = static_cast<vk::ExternalMemoryHandleTypeFlagBits>(HeapHandle),
    };
    const auto properties = instance.GetPhysicalDevice().getExternalBufferProperties(info);
    if (!(properties.externalMemoryProperties.externalMemoryFeatures &
          vk::ExternalMemoryFeatureFlagBits::eExportable)) {
        Fail("the driver cannot export buffer memory as an MTLHeap", true);
        return;
    }
    get_metal_handle =
        reinterpret_cast<void*>(instance.GetDevice().getProcAddr("vkGetMemoryMetalHandleEXT"));
    if (!get_metal_handle) {
        Fail("vkGetMemoryMetalHandleEXT is missing", true);
        return;
    }
    // A small exported buffer gives KosmicKrisp's MTLDevice, the one MetalFX must run on.
    Staging probe;
    if (!CreateStaging(probe, 64 * 1024)) {
        Fail("cannot allocate exportable memory", true);
        return;
    }
    std::string reason;
    supported = BbMetalFx::Supported(probe.heap, reason);
    DestroyStaging(probe);
    if (!supported) {
        Fail(reason, true);
        return;
    }
    std::printf("Upscaler: MetalFX available (temporal scaler on KosmicKrisp's MTLDevice)\n");
    SetUpEvents();
}

void MetalFxUpscaler::SetUpEvents() {
    if (EnvOn("BB_METALFX_SYNC")) {
        std::printf("Upscaler: MetalFX synchronous (BB_METALFX_SYNC=1)\n");
        return;
    }
    if (!instance.IsMetalObjectsEnabled()) {
        std::printf("Upscaler: MetalFX synchronous (the driver lacks VK_EXT_metal_objects: "
                    "rebuild KosmicKrisp with the port's patches)\n");
        return;
    }
    const auto device = instance.GetDevice();
    const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        device.getProcAddr("vkExportMetalObjectsEXT"));
    if (!export_objects) {
        std::printf("Upscaler: MetalFX synchronous (vkExportMetalObjectsEXT is missing)\n");
        return;
    }
    const vk::SemaphoreTypeCreateInfo timeline{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0,
    };
    const auto created = device.createSemaphore({.pNext = &timeline});
    if (created.result != vk::Result::eSuccess) {
        return;
    }
    done_semaphore = created.value;
    const auto event_of = [&](vk::Semaphore semaphore) -> void* {
        VkExportMetalSharedEventInfoEXT event_info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT,
            .semaphore = static_cast<VkSemaphore>(semaphore),
        };
        VkExportMetalObjectsInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
            .pNext = &event_info,
        };
        export_objects(static_cast<VkDevice>(device), &info);
        return reinterpret_cast<void*>(event_info.mtlSharedEvent);
    };
    done_event = event_of(done_semaphore);
    work_event = event_of(scheduler.GetWorkSemaphore()->Handle());
    if (!done_event || !work_event) {
        std::printf("Upscaler: MetalFX synchronous (no shared events from the driver)\n");
        device.destroySemaphore(done_semaphore);
        done_semaphore = vk::Semaphore{};
        done_event = work_event = nullptr;
        return;
    }
    async = true;
    std::printf("Upscaler: MetalFX without CPU waits (Vulkan timelines as Metal shared events)\n");
}

MetalFxUpscaler::~MetalFxUpscaler() {
    if (scaler || staging[0].buffer) {
        scheduler.Finish();
    }
    scaler.reset();
    for (auto& s : staging) {
        DestroyStaging(s);
    }
    if (done_semaphore) {
        instance.GetDevice().destroySemaphore(done_semaphore);
    }
}

void MetalFxUpscaler::Fail(std::string reason, bool permanent) {
    if (problem != reason) {
        std::printf("Upscaler: MetalFX unavailable: %s\n", reason.c_str());
    }
    problem = std::move(reason);
    fatal |= permanent;
}

bool MetalFxUpscaler::CreateStaging(Staging& s, vk::DeviceSize size) {
    const auto device = instance.GetDevice();
    const VkExternalMemoryBufferCreateInfo external{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = HeapHandle,
    };
    const auto buffer = device.createBuffer(vk::BufferCreateInfo{
        .pNext = &external,
        .size = size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    });
    if (buffer.result != vk::Result::eSuccess) {
        return false;
    }
    s.buffer = buffer.value;
    s.size = size;
    const auto requirements = device.getBufferMemoryRequirements(s.buffer);
    const auto memory_properties = instance.GetPhysicalDevice().getMemoryProperties();
    u32 type = ~0u;
    for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if (!(requirements.memoryTypeBits & (1u << i))) {
            continue;
        }
        if (type == ~0u || (memory_properties.memoryTypes[i].propertyFlags &
                            vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            type = i;
            if (memory_properties.memoryTypes[i].propertyFlags &
                vk::MemoryPropertyFlagBits::eDeviceLocal) {
                break;
            }
        }
    }
    // One allocation per buffer at offset 0: the MTLHeap's buffer view starts there.
    const VkMemoryDedicatedAllocateInfo dedicated{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .buffer = static_cast<VkBuffer>(s.buffer),
    };
    const VkExportMemoryAllocateInfo export_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .handleTypes = HeapHandle,
    };
    if (type == ~0u) {
        DestroyStaging(s);
        return false;
    }
    const auto memory = device.allocateMemory(vk::MemoryAllocateInfo{
        .pNext = &export_info,
        .allocationSize = requirements.size,
        .memoryTypeIndex = type,
    });
    if (memory.result != vk::Result::eSuccess) {
        DestroyStaging(s);
        return false;
    }
    s.memory = memory.value;
    if (device.bindBufferMemory(s.buffer, s.memory, 0) != vk::Result::eSuccess) {
        DestroyStaging(s);
        return false;
    }
    const VkMemoryGetMetalHandleInfoEXT get_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_METAL_HANDLE_INFO_EXT,
        .memory = static_cast<VkDeviceMemory>(s.memory),
        .handleType = HeapHandle,
    };
    const auto get = reinterpret_cast<PFN_vkGetMemoryMetalHandleEXT>(get_metal_handle);
    if (get(static_cast<VkDevice>(device), &get_info, &s.heap) != VK_SUCCESS || !s.heap) {
        DestroyStaging(s);
        return false;
    }
    return true;
}

void MetalFxUpscaler::DestroyStaging(Staging& s) {
    const auto device = instance.GetDevice();
    if (s.buffer) {
        device.destroyBuffer(s.buffer);
    }
    if (s.memory) {
        device.freeMemory(s.memory);
    }
    s = {};
}

bool MetalFxUpscaler::EnsureResources(const Frame& frame) {
    if (scaler && frame.width == width && frame.height == height &&
        frame.out_width == out_width && frame.out_height == out_height &&
        frame.color.format == color_format && frame.hdr == hdr) {
        return true;
    }
    // The old buffers may still be in a submitted copy.
    scheduler.Finish();
    scaler.reset();
    for (auto& s : staging) {
        DestroyStaging(s);
    }
    width = frame.width;
    height = frame.height;
    out_width = frame.out_width;
    out_height = frame.out_height;
    color_format = frame.color.format;
    hdr = frame.hdr;
    const bool rgba16 = color_format == vk::Format::eR16G16B16A16Sfloat;
    const vk::DeviceSize pixels = vk::DeviceSize(width) * height;
    const std::array<vk::DeviceSize, 4> sizes{pixels * (rgba16 ? 8 : 4), pixels * 4, pixels * 4,
                                              vk::DeviceSize(out_width) * out_height * 8};
    for (u32 i = 0; i < staging.size(); ++i) {
        if (!CreateStaging(staging[i], sizes[i])) {
            Fail("cannot allocate exportable staging buffers", true);
            return false;
        }
    }
    const BbMetalFx::Config config{
        .in_width = width,
        .in_height = height,
        .out_width = out_width,
        .out_height = out_height,
        .color = rgba16 ? BbMetalFx::ColorFormat::Rgba16Float : BbMetalFx::ColorFormat::Rgba8Unorm,
        .hdr = hdr,
    };
    std::string error;
    scaler = BbMetalFx::Scaler::Create(
        config, {staging[0].heap, staging[1].heap, staging[2].heap, staging[3].heap}, error);
    if (!scaler) {
        Fail(error, true);
        return false;
    }
    std::printf("Upscaler: MetalFX %ux%u -> %ux%u (%s)\n", width, height, out_width, out_height,
                hdr ? "HDR scene color" : "tonemapped frame");
    return true;
}

bool MetalFxUpscaler::Run(const Frame& frame) {
    if (!supported || fatal) {
        return false;
    }
    const auto color = frame.color.format;
    const auto depth = frame.depth.format;
    if ((color != vk::Format::eR16G16B16A16Sfloat && color != vk::Format::eR8G8B8A8Unorm &&
         color != vk::Format::eR8G8B8A8Srgb) ||
        (depth != vk::Format::eD32Sfloat && depth != vk::Format::eD32SfloatS8Uint) ||
        frame.motion.format != vk::Format::eR16G16Sfloat) {
        Fail("unsupported input formats " + vk::to_string(color) + " / " + vk::to_string(depth),
             true);
        return false;
    }
    if (!EnsureResources(frame)) {
        return false;
    }
    auto cmd = scheduler.CommandBuffer();
    const vk::MemoryBarrier2 before_copy{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
    };
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &before_copy});
    const auto download = [&](const Input& input, const Staging& target) {
        const vk::BufferImageCopy region{
            .imageSubresource = {input.aspect, 0, 0, 1},
            .imageExtent = {frame.width, frame.height, 1},
        };
        cmd.copyImageToBuffer(input.image, vk::ImageLayout::eGeneral, target.buffer, region);
    };
    download(frame.color, staging[0]);
    download(frame.depth, staging[1]);
    download(frame.motion, staging[2]);
    const vk::MemoryBarrier2 to_metal{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_metal});
    // An earlier asynchronous frame that failed: its event was still set, so nothing hangs.
    if (std::string error; async && scaler->TakeError(error)) {
        Fail(error, true);
        return false;
    }
    // Asynchronous: this submission's timeline value is what Metal waits for. Synchronous: the
    // GPU finishes everything first.
    const u64 copies_tick = scheduler.CurrentTick();
    if (async) {
        scheduler.Flush();
    } else {
        scheduler.Finish();
    }

    // Diagnostics for the conventions (A/B on the Mac): jitter sign, motion sign, reversed Z.
    static const bool invert_jitter = EnvOn("BB_METALFX_INVERT_JITTER");
    static const bool invert_motion = EnvOn("BB_METALFX_INVERT_MOTION");
    static const bool depth_reversed = EnvOn("BB_METALFX_DEPTH_REVERSED");
    const float jitter_sign = invert_jitter ? -1.0f : 1.0f;
    const float motion_sign = invert_motion ? -1.0f : 1.0f;
    std::string error;
    if (!scaler->Encode({.jitter_x = jitter_sign * frame.jitter[0],
                         .jitter_y = jitter_sign * frame.jitter[1],
                         .motion_scale_x = motion_sign,
                         .motion_scale_y = motion_sign,
                         .reset = frame.reset,
                         .depth_reversed = depth_reversed,
                         .wait_event = async ? work_event : nullptr,
                         .wait_value = copies_tick,
                         .signal_event = async ? done_event : nullptr,
                         .signal_value = async ? done_value + 1 : 0},
                        error)) {
        Fail(error, true);
        return false;
    }

    cmd = scheduler.CommandBuffer();
    const auto output_barrier = [&](vk::ImageLayout old_layout, vk::PipelineStageFlags2 src,
                                    vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst,
                                    vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = src,
            .srcAccessMask = src_access,
            .dstStageMask = dst,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = frame.output,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    };
    constexpr auto rw = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    // Every pixel is overwritten: the previous contents are not needed.
    output_barrier(vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits2::eAllCommands, rw,
                   vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    const vk::BufferImageCopy region{
        .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageExtent = {frame.out_width, frame.out_height, 1},
    };
    cmd.copyBufferToImage(staging[3].buffer, frame.output, vk::ImageLayout::eGeneral, region);
    output_barrier(vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eTransfer,
                   vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
                   rw);
    if (async) {
        // The copy back (and, in queue order, everything after it) waits for MetalFX on the GPU.
        ++done_value;
        SubmitInfo info{};
        info.AddWait(done_semaphore, done_value);
        scheduler.Flush(info);
    }
    return true;
}

#else // not macOS (or Vulkan headers without VK_EXT_external_memory_metal)

MetalFxUpscaler::MetalFxUpscaler(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
#ifdef __APPLE__
    problem = "Vulkan headers without VK_EXT_external_memory_metal (rerun setup_deps.sh)";
    std::printf("Upscaler: MetalFX unavailable: %s\n", problem.c_str());
#else
    problem = "MetalFX is macOS-only";
#endif
    fatal = true;
}

MetalFxUpscaler::~MetalFxUpscaler() = default;

bool MetalFxUpscaler::Run(const Frame&) {
    return false;
}

#endif

} // namespace Vulkan
