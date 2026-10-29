// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// PCIe load (diagnostics): keeps the bus between system memory and the GPU busy with copies from
// host memory into VRAM on a transfer queue, so a game running beside it gets a narrower bus
// (an RTX 5060 runs at x8: on a PCIe 4.0 board half a x16 card's bandwidth, a quarter on 3.0).
// Which GPU passes slow down with it on is what reads the game's memory over the bus.
//   pcie_hog.exe [SECONDS] [MIB_PER_COPY] [DEVICE_INDEX]
// Prints the bandwidth it takes every 2 s. Build (MSYS2 CLANG64):
//   clang -O2 pcie_hog.c -o pcie_hog.exe -lvulkan-1
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#define CHECK(x)                                                     \
    do {                                                             \
        VkResult r_ = (x);                                           \
        if (r_ != VK_SUCCESS) {                                      \
            fprintf(stderr, "%s failed: %d\n", #x, (int)r_);         \
            return 1;                                                \
        }                                                            \
    } while (0)

static double now_s(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
}

static int find_type(const VkPhysicalDeviceMemoryProperties* mp, uint32_t bits, VkMemoryPropertyFlags want,
                     VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < mp->memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = mp->memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (f & want) == want && !(f & avoid)) return (int)i;
    }
    return -1;
}

int main(int argc, char** argv) {
    const double seconds = argc > 1 ? atof(argv[1]) : 60;
    const VkDeviceSize size = (VkDeviceSize)(argc > 2 ? atoi(argv[2]) : 64) << 20;
    const uint32_t want_device = argc > 3 ? (uint32_t)atoi(argv[3]) : 0;

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "pcie_hog", 1, NULL, 0, VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance instance;
    CHECK(vkCreateInstance(&ici, NULL, &instance));
    uint32_t count = 8;
    VkPhysicalDevice devices[8];
    CHECK(vkEnumeratePhysicalDevices(instance, &count, devices));
    if (want_device >= count) {
        fprintf(stderr, "no device %u\n", want_device);
        return 1;
    }
    VkPhysicalDevice gpu = devices[want_device];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(gpu, &props);

    // A transfer-only queue family if there is one (the copy engine, beside the game's queues).
    uint32_t nfam = 16;
    VkQueueFamilyProperties fams[16];
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nfam, fams);
    uint32_t family = 0;
    for (uint32_t i = 0; i < nfam; ++i) {
        const VkQueueFlags f = fams[i].queueFlags;
        if ((f & VK_QUEUE_TRANSFER_BIT) && !(f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
            family = i;
            break;
        }
    }
    const float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, family, 1, &prio};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qci};
    VkDevice device;
    CHECK(vkCreateDevice(gpu, &dci, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);

    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu, &mp);
    VkBuffer buffers[2];
    VkDeviceMemory memories[2];
    for (int i = 0; i < 2; ++i) {
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                                  i == 0 ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT : VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        CHECK(vkCreateBuffer(device, &bci, NULL, &buffers[i]));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, buffers[i], &req);
        // 0: system memory (host visible, not device local); 1: VRAM.
        const int type = i == 0 ? find_type(&mp, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                                : find_type(&mp, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
        if (type < 0) {
            fprintf(stderr, "no memory type for buffer %d\n", i);
            return 1;
        }
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, (uint32_t)type};
        CHECK(vkAllocateMemory(device, &mai, NULL, &memories[i]));
        CHECK(vkBindBufferMemory(device, buffers[i], memories[i], 0));
    }

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, family};
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(device, &pci, NULL, &pool));
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool,
                                       VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmd;
    CHECK(vkAllocateCommandBuffers(device, &cai, &cmd));
    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    CHECK(vkBeginCommandBuffer(cmd, &cbi));
    const VkBufferCopy copy = {0, 0, size};
    for (int i = 0; i < 4; ++i) vkCmdCopyBuffer(cmd, buffers[0], buffers[1], 1, &copy);
    CHECK(vkEndCommandBuffer(cmd));
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    CHECK(vkCreateFence(device, &fci, NULL, &fence));

    printf("pcie_hog on %s, queue family %u, %llu MiB copies host -> VRAM for %.0f s\n", props.deviceName,
           family, (unsigned long long)(size >> 20), seconds);
    fflush(stdout);
    const double start = now_s();
    double window = start, bytes = 0;
    while (now_s() - start < seconds) {
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmd};
        CHECK(vkQueueSubmit(queue, 1, &si, fence));
        CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
        CHECK(vkResetFences(device, 1, &fence));
        bytes += 4.0 * (double)size;
        const double t = now_s();
        if (t - window >= 2.0) {
            printf("pcie_hog: %.1f GB/s\n", bytes / (t - window) / 1e9);
            fflush(stdout);
            window = t;
            bytes = 0;
        }
    }
    vkDeviceWaitIdle(device);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
