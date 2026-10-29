// Host memory import of 320 MiB resident pages by memory type and allocate flags: which of the
// game's choices (host-cached type, device address) makes the import 10x slower than the plain one.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <vulkan/vulkan.h>

static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "imp", 1, "imp", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    vkCreateInstance(&ici, NULL, &inst);
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    const char* exts[] = {"VK_KHR_external_memory", "VK_EXT_external_memory_host"};
    VkPhysicalDeviceVulkan12Features f12 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f12, 0, 1, &q, 0, NULL, 2, exts};
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { printf("no device\n"); return 1; }
    PFN_vkGetMemoryHostPointerPropertiesEXT getprops =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    const size_t size = 320ull << 20;
    for (int variant = 0; variant < 4; ++variant) {
        const int cached = variant & 1, address = variant >> 1;
        HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_COMMIT, 0, (DWORD)size, NULL);
        unsigned char* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size);
        memset(view, 1, size);
        VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        getprops(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view, &hp);
        uint32_t type = ~0u;
        for (uint32_t i = 0; i < mp.memoryTypeCount && type == ~0u; ++i) {
            const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
            if (!(hp.memoryTypeBits & (1u << i)) || !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ||
                (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
            if (!!(f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != cached) continue;
            type = i;
        }
        if (type == ~0u) { printf("no %s type\n", cached ? "cached" : "uncached"); continue; }
        VkMemoryAllocateFlagsInfo flags = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, NULL,
                                           VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
        VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                                address ? &flags : NULL,
                                                VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view};
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, size, type};
        VkDeviceMemory mem;
        const double t0 = now_ms();
        const VkResult r = vkAllocateMemory(dev, &mai, NULL, &mem);
        const double t1 = now_ms();
        printf("320 MiB resident, type %u %-8s %-15s: import %.1f ms (%d)\n", type, cached ? "cached" : "uncached",
               address ? "device address" : "", t1 - t0, r);
        vkFreeMemory(dev, mem, NULL);
        UnmapViewOfFile(view);
        CloseHandle(section);
    }
    return 0;
}
