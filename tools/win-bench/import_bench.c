// Cost of importing host memory into Vulkan (VK_EXT_external_memory_host) by size, untouched and
// touched pages, as the runtime's heap pieces (320 MiB) are imported when the game's heap grows.
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
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    const char* exts[] = {"VK_KHR_external_memory", "VK_EXT_external_memory_host"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q, 0, NULL, 2, exts};
    VkDevice dev;
    vkCreateDevice(pd, &dci, NULL, &dev);
    PFN_vkGetMemoryHostPointerPropertiesEXT getprops =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    const size_t sizes[] = {16ull << 20, 64ull << 20, 320ull << 20};
    for (int touched = 0; touched < 3; ++touched) {
        for (int s = 0; s < 3; ++s) {
            const size_t size = sizes[s];
            HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_COMMIT,
                                                (DWORD)(size >> 32), (DWORD)size, NULL);
            unsigned char* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size);
            double touch_ms = 0.0;
            if (touched == 1) memset(view, 1, size);
            if (touched == 2) { // read one byte of every page: resident, nothing written
                const double t = now_ms();
                volatile unsigned char sum = 0;
                for (size_t at = 0; at < size; at += 4096) sum += view[at];
                touch_ms = now_ms() - t;
            }
            VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
            getprops(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view, &hp);
            uint32_t type = 0;
            while (type < 32 && !(hp.memoryTypeBits & (1u << type))) ++type;
            VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, NULL,
                                                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view};
            VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, size, type};
            VkDeviceMemory mem;
            const double t0 = now_ms();
            const VkResult r = vkAllocateMemory(dev, &mai, NULL, &mem);
            const double t1 = now_ms();
            vkFreeMemory(dev, mem, NULL);
            const double t2 = now_ms();
            static const char* kinds[] = {"untouched", "written  ", "read     "};
            printf("%s %4zu MiB: import %.1f ms (%d), free %.1f ms (read touch %.1f ms)\n", kinds[touched],
                   size >> 20, t1 - t0, r, t2 - t1, touch_ms);
            UnmapViewOfFile(view);
            CloseHandle(section);
        }
    }
    return 0;
}
