#include <mach/mach.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(call) do { \
    VkResult result = (call); \
    if (result != VK_SUCCESS) { \
        fprintf(stderr, "%s failed: %d\n", #call, result); \
        exit(1); \
    } \
} while (0)

static uint64_t footprint(void) {
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) != KERN_SUCCESS)
        exit(1);
    if (info.phys_footprint > (UINT64_C(2) << 30)) {
        fprintf(stderr, "Stopped at the standalone test's 2 GiB memory ceiling.\n");
        exit(2);
    }
    return info.phys_footprint >> 20;
}

int main(void) {
    enum { BUFFER_COUNT = 128, COPY_COUNT = 512, ROUNDS = 20, BUFFER_BYTES = 1024 * 1024 };
    const char *extension = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "bbport command memory check",
                            .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .pApplicationInfo = &app, .enabledExtensionCount = 1, .ppEnabledExtensionNames = &extension};
    VkInstance instance;
    CHECK(vkCreateInstance(&instance_info, NULL, &instance));
    uint32_t count = 1;
    VkPhysicalDevice physical;
    CHECK(vkEnumeratePhysicalDevices(instance, &count, &physical));
    if (count != 1) return 1;
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    printf("Device: %s; driver 0x%x\n", properties.deviceName, properties.driverVersion);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
    VkQueueFamilyProperties *queues = calloc(count, sizeof(*queues));
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, queues);
    uint32_t family = 0;
    while (family < count && !(queues[family].queueFlags & VK_QUEUE_GRAPHICS_BIT)) ++family;
    if (family == count) return 1;
    free(queues);
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};
    VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info};
    VkDevice device;
    CHECK(vkCreateDevice(physical, &device_info, NULL, &device));
    VkQueue queue;
    vkGetDeviceQueue(device, family, 0, &queue);
    VkBuffer buffers[2];
    VkDeviceMemory memories[2];
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);
    for (unsigned i = 0; i < 2; ++i) {
        VkBufferCreateInfo buffer_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = BUFFER_BYTES, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        CHECK(vkCreateBuffer(device, &buffer_info, NULL, &buffers[i]));
        VkMemoryRequirements requirements;
        vkGetBufferMemoryRequirements(device, buffers[i], &requirements);
        uint32_t type = 0;
        const VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        while (type < memory_properties.memoryTypeCount &&
            (!(requirements.memoryTypeBits & (1u << type)) ||
             (memory_properties.memoryTypes[type].propertyFlags & wanted) != wanted)) ++type;
        if (type == memory_properties.memoryTypeCount) return 1;
        VkMemoryAllocateInfo allocation = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = requirements.size, .memoryTypeIndex = type};
        CHECK(vkAllocateMemory(device, &allocation, NULL, &memories[i]));
        CHECK(vkBindBufferMemory(device, buffers[i], memories[i], 0));
        void *bytes;
        CHECK(vkMapMemory(device, memories[i], 0, BUFFER_BYTES, 0, &bytes));
        memset(bytes, i ? 0 : 0x5a, BUFFER_BYTES);
        vkUnmapMemory(device, memories[i]);
    }
    VkCommandPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family};
    VkCommandPool pool;
    CHECK(vkCreateCommandPool(device, &pool_info, NULL, &pool));
    VkCommandBuffer commands[BUFFER_COUNT];
    VkCommandBufferAllocateInfo allocation = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    for (unsigned i = 0; i < BUFFER_COUNT; ++i) {
        CHECK(vkAllocateCommandBuffers(device, &allocation, &commands[i]));
        footprint();
    }
    printf("Allocated %u Vulkan command buffers: %llu MiB footprint\n", BUFFER_COUNT, (unsigned long long)footprint());
    fflush(stdout);
    for (unsigned round = 0; round < ROUNDS; ++round) {
        for (unsigned i = 0; i < BUFFER_COUNT; ++i) {
            VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
            CHECK(vkBeginCommandBuffer(commands[i], &begin));
            for (unsigned j = 0; j < COPY_COUNT; ++j) {
                VkBufferCopy copy = {.srcOffset = j * 256, .dstOffset = j * 256, .size = 256};
                vkCmdCopyBuffer(commands[i], buffers[0], buffers[1], 1, &copy);
            }
            VkMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT};
            vkCmdPipelineBarrier(commands[i], VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
            CHECK(vkEndCommandBuffer(commands[i]));
            VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1, .pCommandBuffers = &commands[i]};
            CHECK(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
            CHECK(vkQueueWaitIdle(queue));
            footprint();
        }
        printf("Round %u: %llu MiB footprint\n", round + 1, (unsigned long long)footprint());
        fflush(stdout);
    }
    void *output;
    CHECK(vkMapMemory(device, memories[1], 0, BUFFER_BYTES, 0, &output));
    for (unsigned i = 0; i < COPY_COUNT * 256; ++i) {
        if (((unsigned char *)output)[i] != 0x5a) {
            fprintf(stderr, "Copy verification failed at byte %u\n", i);
            return 1;
        }
    }
    vkUnmapMemory(device, memories[1]);
    vkDestroyCommandPool(device, pool, NULL);
    printf("After pool destruction: %llu MiB footprint; copied bytes verified\n", (unsigned long long)footprint());
    for (unsigned i = 0; i < 2; ++i) {
        vkDestroyBuffer(device, buffers[i], NULL);
        vkFreeMemory(device, memories[i], NULL);
    }
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
