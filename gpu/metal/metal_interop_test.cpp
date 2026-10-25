// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): interop test for the Metal command path (metal_interop_test.h).

#include "metal/metal_interop_test.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "render/scheduler.h"
#include "vulkan/instance.h"

#include <vk_mem_alloc.h>

#if defined(__APPLE__)
// Only the C types of VK_EXT_metal_objects (as vk_metalfx.cpp): vulkan.hpp stays without the
// Metal platform define.
#include <vulkan/vulkan_metal.h>
#include "metal/metal_interop.h"
#endif

namespace BbMetal {

#if defined(__APPLE__) && defined(VK_EXT_metal_objects)

namespace {

constexpr uint32_t Size = 64;            // the test image: Size x Size RGBA8
constexpr uint32_t Bytes = Size * Size * 4;
constexpr uint32_t Rounds = 32;

/// The value every byte of channel `c` holds after a clear to `rgba` (UNORM, exact for k/255).
uint8_t Byte(const float rgba[4], int c) {
    return uint8_t(rgba[c] * 255.0f + 0.5f);
}

/// First pixel of `data` (Bytes bytes) that is not `rgba`, or -1.
int Mismatch(const uint8_t* data, const float rgba[4]) {
    for (uint32_t i = 0; i < Size * Size; ++i) {
        for (int c = 0; c < 4; ++c) {
            if (data[i * 4 + c] != Byte(rgba, c)) {
                return int(i);
            }
        }
    }
    return -1;
}

struct Test {
    const Vulkan::Instance& instance;
    vk::Device device;
    PFN_vkExportMetalObjectsEXT export_objects = nullptr;
    vk::Image image{};
    VmaAllocation image_allocation{};
    vk::Buffer metal_buffer{}; ///< Metal copies the image into it (device memory, exported)
    VmaAllocation metal_allocation{};
    VmaAllocationInfo metal_info{};
    vk::Buffer readback{}; ///< host-visible: [0, Bytes) the image, [Bytes, 2 Bytes) metal_buffer
    VmaAllocation readback_allocation{};
    VmaAllocationInfo readback_info{};
    vk::CommandPool pool{};
    vk::Fence fence{};

    explicit Test(const Vulkan::Instance& instance_)
        : instance{instance_}, device{instance_.GetDevice()} {}

    ~Test() {
        if (fence) {
            device.destroyFence(fence);
        }
        if (pool) {
            device.destroyCommandPool(pool);
        }
        if (readback) {
            vmaDestroyBuffer(instance.GetAllocator(), readback, readback_allocation);
        }
        if (metal_buffer) {
            vmaDestroyBuffer(instance.GetAllocator(), metal_buffer, metal_allocation);
        }
        if (image) {
            vmaDestroyImage(instance.GetAllocator(), image, image_allocation);
        }
    }

    void Export(VkBaseOutStructure* what) {
        VkExportMetalObjectsInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
            .pNext = what,
        };
        export_objects(static_cast<VkDevice>(device), &info);
    }

    bool Create(std::string& error) {
        export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
            device.getProcAddr("vkExportMetalObjectsEXT"));
        if (!instance.IsMetalObjectsEnabled() || !export_objects) {
            error = "VK_EXT_metal_objects is missing (rebuild KosmicKrisp with the port's patches)";
            return false;
        }
        const VkImageCreateInfo image_info{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R8G8B8A8_UNORM,
            .extent = {Size, Size, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        const VmaAllocationCreateInfo device_local{.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkImage raw_image{};
        if (vmaCreateImage(instance.GetAllocator(), &image_info, &device_local, &raw_image,
                           &image_allocation, nullptr) != VK_SUCCESS) {
            error = "cannot create the test image";
            return false;
        }
        image = raw_image;
        const VkBufferCreateInfo metal_buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = Bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        VkBuffer raw_buffer{};
        if (vmaCreateBuffer(instance.GetAllocator(), &metal_buffer_info, &device_local,
                            &raw_buffer, &metal_allocation, &metal_info) != VK_SUCCESS) {
            error = "cannot create the Metal-written buffer";
            return false;
        }
        metal_buffer = raw_buffer;
        const VkBufferCreateInfo readback_buffer_info{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = 2 * Bytes,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        };
        const VmaAllocationCreateInfo host{
            .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO,
        };
        if (vmaCreateBuffer(instance.GetAllocator(), &readback_buffer_info, &host, &raw_buffer,
                            &readback_allocation, &readback_info) != VK_SUCCESS ||
            !readback_info.pMappedData) {
            error = "cannot create the readback buffer";
            return false;
        }
        readback = raw_buffer;
        pool = device
                   .createCommandPool({.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                       .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex()})
                   .value;
        fence = device.createFence({}).value;
        return pool && fence;
    }

    vk::CommandBuffer Begin() {
        const auto buffers = device.allocateCommandBuffers(
            {.commandPool = pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1});
        const vk::CommandBuffer cmd = buffers.value[0];
        (void)cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        return cmd;
    }

    static void Barrier(vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 all{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        };
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &all});
    }

    bool Submit(vk::CommandBuffer cmd, vk::Fence signal) {
        (void)cmd.end();
        const vk::SubmitInfo submit{.commandBufferCount = 1, .pCommandBuffers = &cmd};
        return instance.GetGraphicsQueue().submit(submit, signal) == vk::Result::eSuccess;
    }
};

} // namespace

void RunInteropTestIfAsked(const Vulkan::Instance& instance) {
    const char* env = std::getenv("BB_METAL_INTEROP_TEST");
    if (!env || env[0] != '1') {
        return;
    }
    Test test{instance};
    std::string error;
    if (!test.Create(error)) {
        std::printf("Metal interop test: FAIL (%s)\n", error.c_str());
        return;
    }
    // KosmicKrisp's queue, the image's texture and the buffer memory's heap-wide MTLBuffer
    // (patches 0007, 0008).
    VkExportMetalCommandQueueInfoEXT queue_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT,
        .queue = static_cast<VkQueue>(instance.GetGraphicsQueue()),
    };
    VkExportMetalTextureInfoEXT texture_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
        .pNext = &queue_info,
        .image = static_cast<VkImage>(test.image),
        .plane = VK_IMAGE_ASPECT_PLANE_0_BIT,
    };
    VkExportMetalBufferInfoEXT buffer_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_BUFFER_INFO_EXT,
        .pNext = &texture_info,
        .memory = test.metal_info.deviceMemory,
    };
    test.Export(reinterpret_cast<VkBaseOutStructure*>(&buffer_info));
    void* queue = reinterpret_cast<void*>(queue_info.mtlCommandQueue);
    void* texture = reinterpret_cast<void*>(texture_info.mtlTexture);
    void* buffer = reinterpret_cast<void*>(buffer_info.mtlBuffer);
    if (!queue || !texture || !buffer) {
        std::printf("Metal interop test: FAIL (the driver exported queue %p, texture %p, buffer %p: "
                    "KosmicKrisp without the port's patches 0007 and 0008?)\n",
                    queue, texture, buffer);
        return;
    }
    const auto* data = static_cast<const uint8_t*>(test.readback_info.pMappedData);
    u32 passed = 0;
    for (u32 round = 0; round < Rounds; ++round) {
        // Vulkan clears to `before`; Metal must read `before` and then clear to `after`; Vulkan
        // must read `after`. Values change every round, so a stale result shows.
        const float before[4] = {float(8 * round + 1) / 255.0f, 0.0f, float(round) / 255.0f, 1.0f};
        const float after[4] = {0.0f, float(8 * round + 2) / 255.0f, 1.0f, float(255 - round) / 255.0f};
        {
            // Scheduler::submit_mutex: every use of the VkQueue (and so of its Metal queue)
            // is serialised with the renderer's own.
            std::scoped_lock lock{Vulkan::Scheduler::submit_mutex};
            vk::CommandBuffer first = test.Begin();
            const vk::ImageMemoryBarrier2 to_transfer{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .oldLayout = vk::ImageLayout::eUndefined,
                .newLayout = vk::ImageLayout::eGeneral,
                .image = test.image,
                .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
            };
            first.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_transfer});
            const vk::ClearColorValue color{std::array<float, 4>{before[0], before[1], before[2], before[3]}};
            const vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
            first.clearColorImage(test.image, vk::ImageLayout::eGeneral, color, range);
            Test::Barrier(first);
            if (!test.Submit(first, {})) {
                std::printf("Metal interop test: FAIL (round %u: the first Vulkan submission)\n", round);
                return;
            }
            if (!InteropRound(queue, texture, buffer, test.metal_info.offset, Size, Size, after,
                              error)) {
                std::printf("Metal interop test: FAIL (round %u: %s)\n", round, error.c_str());
                return;
            }
            vk::CommandBuffer second = test.Begin();
            Test::Barrier(second);
            const vk::BufferImageCopy image_copy{
                .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .imageExtent = {Size, Size, 1},
            };
            second.copyImageToBuffer(test.image, vk::ImageLayout::eGeneral, test.readback, image_copy);
            const vk::BufferCopy buffer_copy{.srcOffset = 0, .dstOffset = Bytes, .size = Bytes};
            second.copyBuffer(test.metal_buffer, test.readback, buffer_copy);
            const vk::MemoryBarrier2 to_host{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            };
            second.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_host});
            if (!test.Submit(second, test.fence)) {
                std::printf("Metal interop test: FAIL (round %u: the second Vulkan submission)\n", round);
                return;
            }
        }
        if (test.device.waitForFences(test.fence, true, 5'000'000'000ull) != vk::Result::eSuccess) {
            std::printf("Metal interop test: FAIL (round %u: the GPU did not finish in 5 s)\n", round);
            return;
        }
        (void)test.device.resetFences(test.fence);
        (void)test.device.resetCommandPool(test.pool);
        vmaInvalidateAllocation(instance.GetAllocator(), test.readback_allocation, 0, VK_WHOLE_SIZE);
        const int image_bad = Mismatch(data, after);           // Vulkan read Metal's clear
        const int buffer_bad = Mismatch(data + Bytes, before); // Metal read Vulkan's clear
        if (image_bad >= 0 || buffer_bad >= 0) {
            const int at = image_bad >= 0 ? image_bad : buffer_bad;
            const uint8_t* px = (image_bad >= 0 ? data : data + Bytes) + at * 4;
            const float* want = image_bad >= 0 ? after : before;
            std::printf("Metal interop test: FAIL (round %u: %s, pixel %d is %u,%u,%u,%u, expected "
                        "%u,%u,%u,%u)\n",
                        round,
                        image_bad >= 0 ? "Vulkan did not see Metal's clear"
                                       : "Metal did not see Vulkan's clear",
                        at, px[0], px[1], px[2], px[3], Byte(want, 0), Byte(want, 1),
                        Byte(want, 2), Byte(want, 3));
            return;
        }
        ++passed;
    }
    std::printf("Metal interop test: PASS (%u rounds: Metal on KosmicKrisp's queue saw each Vulkan "
                "write before it, and Vulkan saw each Metal write after it)\n",
                passed);
    std::fflush(stdout);
}

#else

void RunInteropTestIfAsked(const Vulkan::Instance&) {}

#endif

} // namespace BbMetal
