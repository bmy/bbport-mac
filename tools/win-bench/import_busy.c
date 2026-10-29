// Host memory import of 320 MiB resident pages while the GPU is busy (long copies in flight):
// does the import wait for the GPU's queued work (WDDM paging), as the game's frames suggest?
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

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties mp;

static uint32_t pick(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want && !(mp.memoryTypes[i].propertyFlags & avoid))
            return i;
    return ~0u;
}

static VkBuffer buffer_of(VkDeviceSize size, VkMemoryPropertyFlags want) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer b;
    vkCreateBuffer(dev, &bci, NULL, &b);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b, &req);
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, pick(req.memoryTypeBits, want, 0)};
    VkDeviceMemory m;
    vkAllocateMemory(dev, &mai, NULL, &m);
    vkBindBufferMemory(dev, b, m, 0);
    return b;
}

static double import_ms(PFN_vkGetMemoryHostPointerPropertiesEXT getprops) {
    const size_t size = 320ull << 20;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_COMMIT, 0, (DWORD)size, NULL);
    unsigned char* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size);
    memset(view, 1, size);
    VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    getprops(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view, &hp);
    VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, NULL,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view};
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, size,
                                pick(hp.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory mem;
    const double t0 = now_ms();
    vkAllocateMemory(dev, &mai, NULL, &mem);
    return now_ms() - t0;
}

int main(void) {
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "imp", 1, "imp", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    vkCreateInstance(&ici, NULL, &inst);
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    const char* exts[] = {"VK_KHR_external_memory", "VK_EXT_external_memory_host"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q, 0, NULL, 2, exts};
    vkCreateDevice(pd, &dci, NULL, &dev);
    VkQueue queue;
    vkGetDeviceQueue(dev, 0, 0, &queue);
    PFN_vkGetMemoryHostPointerPropertiesEXT getprops =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");

    printf("GPU idle: import %.1f ms\n", import_ms(getprops));

    // ~1 s of VRAM-to-VRAM copies queued.
    VkBuffer a = buffer_of(256 << 20, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), b = buffer_of(256 << 20, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, 0};
    VkCommandPool pool;
    vkCreateCommandPool(dev, &cpi, NULL, &pool);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev, &cai, &cb);
    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &cbi);
    VkBufferCopy copy = {0, 0, 256 << 20};
    for (int i = 0; i < 600; ++i) vkCmdCopyBuffer(cb, i & 1 ? b : a, i & 1 ? a : b, 1, &copy);
    vkEndCommandBuffer(cb);
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(dev, &fci, NULL, &fence);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    const double s0 = now_ms();
    vkQueueSubmit(queue, 1, &si, fence);
    Sleep(20);
    const double during = import_ms(getprops);
    const double at = now_ms() - s0;
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    printf("GPU busy (%.0f ms of copies queued): import %.1f ms, done %.0f ms after the submission\n",
           now_ms() - s0, during, at);
    return 0;
}
