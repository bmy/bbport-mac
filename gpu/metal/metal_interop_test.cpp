// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): interop test for the Metal command path (metal_interop_test.h).

#include "metal/metal_interop_test.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include "core/emulator_settings.h"
#include "render/scheduler.h"
#include "shim/bb_env.h"
#include "vulkan/instance.h"

#include <vk_mem_alloc.h>

#if defined(__APPLE__)
// Only the C types of VK_EXT_metal_objects (as vk_metalfx.cpp): vulkan.hpp stays without the
// Metal platform define.
#include <vulkan/vulkan_metal.h>
#include "metal/kk_exports.h"
#include "metal/metal_interop.h"
#include "metal/mtl_layer.h"
#endif

namespace BbMetal {

#if defined(__APPLE__) && defined(VK_EXT_metal_objects)

namespace {

__attribute__((format(printf, 1, 2))) void Report(const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
    std::fflush(stdout);
}

// The test image: Size x Size RGBA8, large enough (16 MB) that work out of order shows.
constexpr uint32_t Size = 2048;
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
    vk::ImageView view{};
    bool submitted = false; ///< work may be on the GPU: the destructor waits for the queue

    explicit Test(const Vulkan::Instance& instance_)
        : instance{instance_}, device{instance_.GetDevice()} {}

    ~Test() {
        if (submitted && !QueueIdle()) {
            // Work still on the GPU after 5 s (a hang): its objects stay allocated rather than go
            // from under it, and startup goes on.
            Report("Metal interop test: the queue did not go idle; the test's objects are left "
                   "allocated\n");
            return;
        }
        if (view) {
            device.destroyImageView(view);
        }
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
        view = device
                   .createImageView({.image = image,
                                     .viewType = vk::ImageViewType::e2D,
                                     .format = vk::Format::eR8G8B8A8Unorm,
                                     .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}})
                   .value;
        return pool && fence && view;
    }

    /// Everything submitted so far has finished: a fenced empty submission after it, waited for
    /// at most 5 s (without holding the submission lock while waiting).
    bool QueueIdle() {
        const auto marker = device.createFence({});
        if (marker.result != vk::Result::eSuccess) {
            return false;
        }
        bool done = false;
        {
            std::scoped_lock lock{Vulkan::Scheduler::submit_mutex};
            done = instance.GetGraphicsQueue().submit(vk::SubmitInfo{}, marker.value) ==
                   vk::Result::eSuccess;
        }
        done = done && device.waitForFences(marker.value, true, 5'000'000'000ull) ==
                           vk::Result::eSuccess;
        if (done) {
            device.destroyFence(marker.value);
        }
        return done;
    }

    vk::CommandBuffer Begin() {
        const auto buffers = device.allocateCommandBuffers(
            {.commandPool = pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1});
        if (buffers.result != vk::Result::eSuccess || buffers.value.empty()) {
            return {};
        }
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
        if (!cmd) {
            return false;
        }
        submitted = true;
        (void)cmd.end();
        const vk::SubmitInfo submit{.commandBufferCount = 1, .pCommandBuffers = &cmd};
        return instance.GetGraphicsQueue().submit(submit, signal) == vk::Result::eSuccess;
    }
};

} // namespace

void RunInteropTestIfAsked(const Vulkan::Instance& instance) {
    const char* env = bb_env("BB_METAL_INTEROP_TEST");
    if (!env || (env[0] != '1' && env[0] != '2')) {
        return;
    }
    // 2: a negative control. The render pass has no barriers of its own; a FAIL then shows the
    // test can catch work out of order (a PASS says only that it ran in order anyway).
    const bool barriers = env[0] == '1';
    if (EmulatorSettings.IsVkValidationEnabled()) {
        // This port's own export entries (patch 0008) need the layers off (an unknown sType).
        Report("Metal interop test: skipped (the Vulkan validation layers are on)\n");
        return;
    }
    Test test{instance};
    std::string error;
    if (!test.Create(error)) {
        Report("Metal interop test: FAIL (%s)\n", error.c_str());
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
    // This port's own exports (patch 0008): the test buffer's MTLBuffer and offset, and the test
    // view's render and sampled textures.
    ExportBufferObject buffer_object{.buffer = static_cast<VkBuffer>(test.metal_buffer)};
    ExportViewTexture render_texture{.pNext = &buffer_object,
                                     .imageView = static_cast<VkImageView>(test.view),
                                     .use = ViewTextureRender};
    ExportViewTexture sampled_texture{.pNext = &render_texture,
                                      .imageView = static_cast<VkImageView>(test.view),
                                      .use = ViewTextureSampled};
    test.Export(reinterpret_cast<VkBaseOutStructure*>(&sampled_texture));
    if (!queue || !texture || !buffer || !buffer_object.mtlBuffer) {
        Report("Metal interop test: FAIL (the driver exported queue %p, texture %p, buffer %p, "
               "buffer object %p: KosmicKrisp without the port's patches 0007 and 0008?)\n",
               queue, texture, buffer, reinterpret_cast<void*>(buffer_object.mtlBuffer));
        return;
    }
    // The exports as KosmicKrisp makes them: the heap-backed test buffer has a sub-buffer of its
    // own at offset 0 (aliasing the memory's heap-wide buffer); a view with the image's format
    // renders into the image's own texture at level 0, slice 0.
    if (buffer_object.mtlBuffer == buffer_info.mtlBuffer || buffer_object.offset != 0 ||
        render_texture.mtlTexture != texture_info.mtlTexture || render_texture.level != 0 ||
        render_texture.slice != 0 || !sampled_texture.mtlTexture) {
        Report("Metal interop test: FAIL (this port's exports: buffer object %p at %llu (the "
               "memory's %p), the view's render texture %p at level %u slice %u (the image's %p), "
               "sampled %p)\n",
               reinterpret_cast<void*>(buffer_object.mtlBuffer),
               (unsigned long long)buffer_object.offset, reinterpret_cast<void*>(buffer),
               reinterpret_cast<void*>(render_texture.mtlTexture), render_texture.level,
               render_texture.slice, texture, reinterpret_cast<void*>(sampled_texture.mtlTexture));
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
            if (!first) {
                Report("Metal interop test: FAIL (round %u: no command buffer)\n", round);
                return;
            }
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
                Report("Metal interop test: FAIL (round %u: the first Vulkan submission)\n", round);
                return;
            }
            // The copy goes into the test buffer through its own MTLBuffer (this port's buffer
            // export, patch 0008), so that export is tested by use.
            if (!InteropRound(queue, texture, buffer_object.mtlBuffer, buffer_object.offset, Size,
                              Size, after, barriers, error)) {
                Report("Metal interop test: FAIL (round %u: %s)\n", round, error.c_str());
                return;
            }
            vk::CommandBuffer second = test.Begin();
            if (!second) {
                Report("Metal interop test: FAIL (round %u: no command buffer)\n", round);
                return;
            }
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
                Report("Metal interop test: FAIL (round %u: the second Vulkan submission)\n", round);
                return;
            }
        }
        if (test.device.waitForFences(test.fence, true, 5'000'000'000ull) != vk::Result::eSuccess) {
            Report("Metal interop test: FAIL (round %u: the GPU did not finish in 5 s)\n", round);
            return;
        }
        // The command buffer's feedback comes on a thread of Metal's own, not with the fence:
        // wait for this round's before reading it.
        for (int i = 0; i < 1000 && InteropCompleted() < round + 1; ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (InteropCompleted() < round + 1) {
            Report("Metal interop test: FAIL (round %u: no feedback for the Metal command buffer "
                   "1 s after the GPU finished)\n",
                   round);
            return;
        }
        if (InteropTakeError(error)) {
            Report("Metal interop test: FAIL (round %u: the Metal command buffer failed: %s)\n",
                   round, error.c_str());
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
            Report("Metal interop test: FAIL (round %u: %s, pixel %d is %u,%u,%u,%u, expected "
                        "%u,%u,%u,%u)\n",
                        round,
                        image_bad >= 0 ? "Vulkan did not see Metal's clear"
                                       : "Metal did not see Vulkan's clear",
                        at, unsigned(px[0]), unsigned(px[1]), unsigned(px[2]), unsigned(px[3]),
                        unsigned(Byte(want, 0)), unsigned(Byte(want, 1)), unsigned(Byte(want, 2)),
                        unsigned(Byte(want, 3)));
            return;
        }
        ++passed;
    }
    Report("Metal interop test: PASS (%u rounds of a %ux%u image%s: Metal on KosmicKrisp's queue "
           "read each Vulkan write before it, and Vulkan read each Metal write after it; KosmicKrisp "
           "orders its own work before ours itself, so the first direction does not test our "
           "barriers; this port's exports as expected)\n",
           passed, Size, Size,
           barriers ? "" : ", the render pass's barriers off (a negative control)");
}

void RunFrameTestIfAsked(const Vulkan::Instance& instance) {
    if (!bb_env_on("BB_METAL_FRAME_TEST")) {
        return;
    }
    const bool frame_on = MetalLayer::FrameActive();
    if (EmulatorSettings.IsVkValidationEnabled()) {
        Report("Metal frame test: skipped (the Vulkan validation layers are on)\n");
        return;
    }
    Test test{instance};
    std::string error;
    if (!test.Create(error)) {
        Report("Metal frame test: FAIL (%s)\n", error.c_str());
        return;
    }
    // Fills with bytes that differ go to KosmicKrisp (Metal fills one byte value), fills of one
    // byte value and copies to Metal: each round alternates the two queues on one 16 MB buffer,
    // then reads it back.
    constexpr u32 Quarter = Bytes / 4;
    const auto* data = static_cast<const uint8_t*>(test.readback_info.pMappedData);
    u32 passed = 0;
    const unsigned long long metal_before = MetalLayer::MetalTransfers();
    unsigned long long metal_expected = 0;
    for (u32 round = 0; round < Rounds; ++round) {
        const u32 x = 0x11223344u ^ (round << 8);                // KosmicKrisp
        const u8 y = u8(round * 7 + 1);                          // Metal
        const u32 z = 0x55667700u | (round & 0xffu);             // KosmicKrisp
        const bool split = round % 2 == 1; // the second half in a submission of its own
        metal_expected += 2;               // the y fill and the copy
        {
            std::scoped_lock lock{Vulkan::Scheduler::submit_mutex};
            vk::CommandBuffer cmd = test.Begin();
            if (!cmd) {
                Report("Metal frame test: FAIL (round %u: no command buffer)\n", round);
                return;
            }
            Test::Barrier(cmd);
            cmd.fillBuffer(test.metal_buffer, 0, Bytes, x);
            Test::Barrier(cmd);
            cmd.fillBuffer(test.metal_buffer, 0, Bytes, u32(y) * 0x01010101u);
            Test::Barrier(cmd);
            if (split) {
                if (!test.Submit(cmd, {})) {
                    Report("Metal frame test: FAIL (round %u: the first submission)\n", round);
                    return;
                }
                cmd = test.Begin();
                if (!cmd) {
                    Report("Metal frame test: FAIL (round %u: no command buffer)\n", round);
                    return;
                }
                Test::Barrier(cmd);
            }
            cmd.fillBuffer(test.metal_buffer, 0, Quarter, z);
            Test::Barrier(cmd);
            const vk::BufferCopy copy{.srcOffset = 0, .dstOffset = 0, .size = Bytes};
            cmd.copyBuffer(test.metal_buffer, test.readback, copy);
            const vk::MemoryBarrier2 to_host{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                .dstAccessMask = vk::AccessFlagBits2::eHostRead,
            };
            cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_host});
            if (!test.Submit(cmd, test.fence)) {
                Report("Metal frame test: FAIL (round %u: the submission)\n", round);
                return;
            }
        }
        if (const vk::Result r = test.device.waitForFences(test.fence, true, 5'000'000'000ull);
            r != vk::Result::eSuccess) {
            Report("Metal frame test: FAIL (round %u: %s)\n", round,
                   r == vk::Result::eTimeout
                       ? "the GPU did not finish in 5 s: a wait between the queues that is never "
                         "signalled?"
                       : vk::to_string(r).c_str());
            return;
        }
        (void)test.device.resetFences(test.fence);
        (void)test.device.resetCommandPool(test.pool);
        vmaInvalidateAllocation(instance.GetAllocator(), test.readback_allocation, 0, VK_WHOLE_SIZE);
        const auto word = [&](u32 i) {
            u32 w;
            std::memcpy(&w, data + u64(i) * 4, 4);
            return w;
        };
        for (u32 i = 0; i < Bytes / 4; ++i) {
            const u32 want = i < Quarter / 4 ? z : u32(y) * 0x01010101u;
            const u32 got = word(i);
            if (got != want) {
                const char* what =
                    got == x ? (i < Quarter / 4
                                    ? "the first fill (KosmicKrisp) last: the later fills lost, "
                                      "or the copy before them (Metal after KosmicKrisp)"
                                    : "the first fill (KosmicKrisp) after the second (Metal), or "
                                      "the copy before the second")
                    : i < Quarter / 4 && got == u32(y) * 0x01010101u
                        ? "the Metal fill after the third (KosmicKrisp), or the copy (Metal) "
                          "before the third"
                        : "a value of no fill of this round (the copy before the fills, or the "
                          "readback before the copy)";
                Report("Metal frame test: FAIL (round %u%s: word %u is 0x%08x, expected 0x%08x: "
                       "%s)\n",
                       round, split ? ", split" : "", i, got, want, what);
                return;
            }
        }
        ++passed;
    }
    const unsigned long long in_metal = MetalLayer::MetalTransfers() - metal_before;
    if (frame_on && in_metal < metal_expected) {
        Report("Metal frame test: FAIL (the data came back right, but only %llu of its %llu Metal "
               "transfers were encoded in Metal: BB_METAL_TRANSFERS, or the buffers not exported)\n",
               in_metal, metal_expected);
        return;
    }
    Report("Metal frame test: PASS (%u rounds on a 16 MB buffer: fills alternating between "
           "KosmicKrisp and %s, a copy and the readback, in one submission and split in two)\n",
           passed,
           frame_on ? "our Metal queue (the frame encoder)"
                    : "KosmicKrisp again (the frame encoder is off: the test then checks nothing "
                      "of the Metal path)");
}

#else

void RunInteropTestIfAsked(const Vulkan::Instance&) {}
void RunFrameTestIfAsked(const Vulkan::Instance&) {}

#endif

} // namespace BbMetal
