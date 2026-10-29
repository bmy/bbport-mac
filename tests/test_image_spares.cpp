// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: spare images (gpu/textures/image_spares.h) on a real GPU. The GPU is kept busy (buffer
// fills, as during an area load), and the texture cache's kind of images is made on this thread:
// first each created, allocated and bound here, then taken from the spares the spare thread made.
// Prints the time spent here per image both ways, and how long this thread's vkQueueSubmit took
// while the spare thread was binding. Exit code 0 when every image was made.
//   image-spares-test [images per shape, default 40]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include <vector>

#include "vulkan/vk_common.h"
#include <vk_mem_alloc.h>

#include "textures/image_spares.h"
#include "common/elf_info.h"
#include "common/path_util.h"
#include "common/thread.h"

// The few things image_spares.cpp takes from the rest of the GPU library.
namespace Common {
void SetCurrentThreadName(const char*) {}
void SetCurrentThreadPriority(ThreadPriority) {}
bool AccurateSleep(std::chrono::nanoseconds duration, std::chrono::nanoseconds*, bool) {
    std::this_thread::sleep_for(duration);
    return true;
}
namespace FS {
const std::filesystem::path& GetUserPath(PathType) {
    static const std::filesystem::path dir = std::filesystem::temp_directory_path() / "bbport-image-spares-test";
    return dir;
}
} // namespace FS
} // namespace Common
u32 VideoCore::ImageSpares::MemoryFlags(vk::Device, const vk::ImageCreateInfo&) {
    return VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
}

namespace {
using Clock = std::chrono::steady_clock;
double Ms(Clock::duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

struct Gpu {
    vk::Instance instance;
    vk::PhysicalDevice physical;
    vk::Device device;
    vk::Queue queue;
    u32 family = 0;
    VmaAllocator allocator{};
};

bool MakeGpu(Gpu& g) {
    static vk::detail::DynamicLoader loader;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));
    const vk::ApplicationInfo app{.pApplicationName = "image-spares-test", .apiVersion = VK_API_VERSION_1_3};
    auto [r, instance] = vk::createInstance({.pApplicationInfo = &app});
    if (r != vk::Result::eSuccess) {
        return false;
    }
    g.instance = instance;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(g.instance);
    auto [r2, physicals] = g.instance.enumeratePhysicalDevices();
    // A discrete GPU when there is one (the laptops' NVIDIA next to an Intel iGPU).
    for (const auto& p : physicals) {
        if (!g.physical || p.getProperties().deviceType == vk::PhysicalDeviceType::eDiscreteGpu) {
            g.physical = p;
        }
    }
    if (!g.physical) {
        return false;
    }
    const auto props = g.physical.getProperties();
    std::printf("GPU: %s (driver %#x)\n", props.deviceName.data(), props.driverVersion);
    const auto families = g.physical.getQueueFamilyProperties();
    for (u32 i = 0; i < families.size(); ++i) {
        if (families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            g.family = i;
            break;
        }
    }
    const float priority = 1.0f;
    const vk::DeviceQueueCreateInfo qci{.queueFamilyIndex = g.family, .queueCount = 1, .pQueuePriorities = &priority};
    auto [r3, device] = g.physical.createDevice({.queueCreateInfoCount = 1, .pQueueCreateInfos = &qci});
    if (r3 != vk::Result::eSuccess) {
        return false;
    }
    g.device = device;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(g.device);
    g.queue = g.device.getQueue(g.family, 0);
    VmaVulkanFunctions functions{
        .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
    };
    const VmaAllocatorCreateInfo aci{.physicalDevice = g.physical, .device = g.device,
                                     .pVulkanFunctions = &functions, .instance = g.instance,
                                     .vulkanApiVersion = VK_API_VERSION_1_3};
    return vmaCreateAllocator(&aci, &g.allocator) == VK_SUCCESS;
}

/// Keeps the GPU busy: fills of a large buffer, two submissions in flight.
struct Load {
    Gpu& g;
    VkBuffer buffer{};
    VmaAllocation allocation{};
    vk::CommandPool pool;
    std::vector<vk::CommandBuffer> cmds;
    std::vector<vk::Fence> fences;
    size_t next = 0;
    double submit_ms_max = 0, submit_ms_sum = 0;
    u32 submits = 0;

    explicit Load(Gpu& gpu) : g{gpu} {
        const VkBufferCreateInfo bci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 64ull << 20,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        const VmaAllocationCreateInfo aci{.usage = VMA_MEMORY_USAGE_GPU_ONLY};
        vmaCreateBuffer(g.allocator, &bci, &aci, &buffer, &allocation, nullptr);
        pool = g.device.createCommandPool({.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                           .queueFamilyIndex = g.family}).value;
        cmds = g.device.allocateCommandBuffers({.commandPool = pool, .commandBufferCount = 2}).value;
        for (int i = 0; i < 2; ++i) {
            fences.push_back(g.device.createFence({.flags = vk::FenceCreateFlagBits::eSignaled}).value);
        }
    }
    ~Load() {
        (void)g.device.waitIdle();
        for (auto f : fences) g.device.destroyFence(f);
        g.device.destroyCommandPool(pool);
        vmaDestroyBuffer(g.allocator, buffer, allocation);
    }
    /// One more batch of fills once a slot is free (does not wait for it).
    void Pump() {
        const size_t i = next % 2;
        if (g.device.getFenceStatus(fences[i]) != vk::Result::eSuccess) {
            return;
        }
        (void)g.device.resetFences(fences[i]);
        auto cmd = cmds[i];
        (void)cmd.reset();
        (void)cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
        for (int k = 0; k < 40; ++k) {
            cmd.fillBuffer(vk::Buffer{buffer}, 0, VK_WHOLE_SIZE, u32(k));
        }
        (void)cmd.end();
        const vk::SubmitInfo si{.commandBufferCount = 1, .pCommandBuffers = &cmd};
        const auto t = Clock::now();
        (void)g.queue.submit(si, fences[i]);
        const double ms = Ms(Clock::now() - t);
        submit_ms_max = std::max(submit_ms_max, ms);
        submit_ms_sum += ms;
        ++submits;
        ++next;
    }
};

/// The texture cache's kinds of images (image.cpp UsageOf): BC7 textures with mips, an RGBA8 one,
/// and the 480x272 depth image a world entry makes.
std::vector<vk::ImageCreateInfo> Shapes() {
    using U = vk::ImageUsageFlagBits;
    const auto flags = vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage;
    const auto copy = U::eTransferSrc | U::eTransferDst | U::eSampled;
    return {
        {.flags = flags, .imageType = vk::ImageType::e2D, .format = vk::Format::eBc7UnormBlock,
         .extent = {1024, 1024, 1}, .mipLevels = 11, .arrayLayers = 1, .usage = copy | U::eStorage},
        {.flags = flags, .imageType = vk::ImageType::e2D, .format = vk::Format::eBc1RgbaUnormBlock,
         .extent = {512, 512, 1}, .mipLevels = 10, .arrayLayers = 1, .usage = copy | U::eStorage},
        {.flags = flags, .imageType = vk::ImageType::e2D, .format = vk::Format::eR8G8B8A8Unorm,
         .extent = {256, 256, 1}, .mipLevels = 1, .arrayLayers = 1,
         .usage = copy | U::eColorAttachment | U::eStorage},
        {.flags = flags, .imageType = vk::ImageType::e2D, .format = vk::Format::eD32Sfloat,
         .extent = {480, 272, 1}, .mipLevels = 1, .arrayLayers = 1, .usage = copy | U::eDepthStencilAttachment},
    };
}

struct Made {
    VkImage image;
    VmaAllocation allocation;
};

/// Creates, allocates and binds on this thread (HostImage::Create's path without a spare).
bool MakeHere(Gpu& g, const vk::ImageCreateInfo& ci, Made& out, double& bind_ms, vk::DeviceSize& size) {
    VkImage h{};
    if (VkResult(g.device.createImage(&ci, nullptr, reinterpret_cast<vk::Image*>(&h))) != VK_SUCCESS) {
        return false;
    }
    const VmaAllocationCreateInfo aci{.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
                                      .usage = VMA_MEMORY_USAGE_GPU_ONLY};
    VmaAllocation a{};
    VmaAllocationInfo info{};
    if (vmaAllocateMemoryForImage(g.allocator, h, &aci, &a, &info) != VK_SUCCESS) {
        g.device.destroyImage(vk::Image{h});
        return false;
    }
    const auto t = Clock::now();
    vmaBindImageMemory(g.allocator, a, h);
    bind_ms = Ms(Clock::now() - t);
    out = {h, a};
    size = info.size;
    return true;
}
} // namespace

int main(int argc, char** argv) {
    const int per_shape = argc > 1 ? std::max(1, std::atoi(argv[1])) : 40;
    Gpu g;
    if (!MakeGpu(g)) {
        std::printf("SKIP: no Vulkan device\n");
        return 0;
    }
    std::filesystem::remove_all(Common::FS::GetUserPath(Common::FS::PathType::CacheDir));
    const auto shapes = Shapes();
    int failures = 0;

    // 1. Every image made here, the GPU busy.
    double here_total = 0, bind_max = 0, here_bind = 0;
    {
        Load load{g};
        std::vector<Made> made;
        for (int i = 0; i < per_shape; ++i) {
            for (const auto& ci : shapes) {
                load.Pump();
                Made m{};
                double bind_ms = 0;
                vk::DeviceSize size = 0;
                const auto t = Clock::now();
                if (!MakeHere(g, ci, m, bind_ms, size)) {
                    ++failures;
                    continue;
                }
                here_total += Ms(Clock::now() - t);
                here_bind += bind_ms;
                bind_max = std::max(bind_max, bind_ms);
                made.push_back(m);
            }
        }
        const int n = int(made.size());
        std::printf("made here:   %d images, %.3f ms each (bind %.3f ms each, %.2f ms at most); "
                    "submit %.3f ms avg, %.2f ms max\n",
                    n, here_total / std::max(n, 1), here_bind / std::max(n, 1), bind_max,
                    load.submit_ms_sum / std::max(load.submits, 1u), load.submit_ms_max);
        for (const auto& m : made) {
            vmaDestroyImage(g.allocator, m.image, m.allocation);
        }
    }

    // 2. The same through the spares: their shapes made here once each (as a first session
    // would), then the spare thread keeps them ready while this thread takes them.
    namespace Spares = VideoCore::ImageSpares;
#ifdef _WIN32
    _putenv_s("BB_IMAGE_SPARES", "1");
#else
    setenv("BB_IMAGE_SPARES", "1", 1);
#endif
    Spares::Start(g.device, g.allocator, 2048ull << 20);
    for (const auto& ci : shapes) {
        for (int k = 0; k < 40; ++k) { // a session's demand: 4 spares of each
            Spares::NoteMade(ci, 0);
        }
    }
    std::this_thread::sleep_for(std::chrono::seconds(2)); // the first spares made (a title screen)
    double spare_total = 0, take_max = 0;
    int taken = 0, missed = 0;
    {
        Load load{g};
        std::vector<Made> made;
        for (int i = 0; i < per_shape; ++i) {
            for (const auto& ci : shapes) {
                load.Pump();
                vk::Image image;
                VmaAllocation allocation{};
                vk::DeviceSize size = 0;
                const auto t = Clock::now();
                if (Spares::Take(g.allocator, ci, image, allocation, size)) {
                    const double ms = Ms(Clock::now() - t);
                    spare_total += ms;
                    take_max = std::max(take_max, ms);
                    made.push_back({VkImage(image), allocation});
                    ++taken;
                    continue;
                }
                Made m{};
                double bind_ms = 0;
                if (!MakeHere(g, ci, m, bind_ms, size)) {
                    ++failures;
                    continue;
                }
                spare_total += Ms(Clock::now() - t);
                Spares::NoteMade(ci, size);
                made.push_back(m);
                ++missed;
            }
            // A frame's worth of time between the images of a load (the spare thread catches up).
            std::this_thread::sleep_for(std::chrono::milliseconds(4));
        }
        const int n = taken + missed;
        std::printf("with spares: %d images, %.3f ms each here (%d taken, %.3f ms at most; %d made "
                    "here); submit %.3f ms avg, %.2f ms max\n",
                    n, spare_total / std::max(n, 1), taken, take_max, missed,
                    load.submit_ms_sum / std::max(load.submits, 1u), load.submit_ms_max);
        for (const auto& m : made) {
            vmaDestroyImage(g.allocator, m.image, m.allocation);
        }
    }
    Spares::Stop();
    auto shapes_file = Common::FS::GetUserPath(Common::FS::PathType::CacheDir);
    if (const auto serial = Common::ElfInfo::Instance().GameSerial(); !serial.empty()) {
        shapes_file /= std::string{serial};
    }
    const bool saved = std::filesystem::exists(shapes_file / "image_shapes.txt");
    std::printf("shapes file %s\n", saved ? "written" : "MISSING");
    vmaDestroyAllocator(g.allocator);
    g.device.destroy();
    g.instance.destroy();
    if (failures || !saved || taken == 0) {
        std::printf("FAIL: %d images not made, %d taken from spares\n", failures, taken);
        return 1;
    }
    std::printf("PASS\n");
    return 0;
}
