// Why VirtualProtect takes ~10 ms per call in the game but 0.01 ms in vp_bench: the game's guest
// memory is a multi-GiB section view placed into a placeholder, imported by the GPU in pieces,
// read by the GPU while the CPU changes protections, with protections already fragmented.
// Each case: 1 MiB pieces made read-only then writable again, at spread offsets.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

typedef PVOID(WINAPI* VirtualAlloc2Fn)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, MEM_EXTENDED_PARAMETER*, ULONG);
typedef PVOID(WINAPI* MapViewOfFile3Fn)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG,
                                        MEM_EXTENDED_PARAMETER*, ULONG);

static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

static void bench(const char* what, unsigned char* base, size_t size) {
    const size_t piece = 1 << 20;
    const int reps = 24;
    double total = 0.0, worst = 0.0, ro_total = 0.0;
    for (int i = 0; i < reps; ++i) {
        unsigned char* at = base + ((size_t)i * 7919 % (size / piece)) * piece;
        DWORD old;
        const double t0 = now_ms();
        VirtualProtect(at, piece, PAGE_READONLY, &old);
        const double t1 = now_ms();
        VirtualProtect(at, piece, PAGE_READWRITE, &old);
        const double t2 = now_ms();
        ro_total += t1 - t0;
        total += t2 - t0;
        if (t1 - t0 > worst) worst = t1 - t0;
    }
    printf("%-52s 1 MiB: to read-only %.3f ms, back %.3f ms (worst read-only %.3f)\n", what, ro_total / reps,
           (total - ro_total) / reps, worst);
}

static VkDevice dev;
static VkQueue queue;
static uint32_t family;

static VkBuffer import_buffer(VkPhysicalDevice pd, void* at, size_t size, VkDeviceMemory* memory) {
    PFN_vkGetMemoryHostPointerPropertiesEXT getprops =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    getprops(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, at, &hp);
    uint32_t type = 0;
    while (type < 32 && !(hp.memoryTypeBits & (1u << type))) ++type;
    VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, NULL,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, at};
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, size, type};
    if (vkAllocateMemory(dev, &mai, NULL, memory) != VK_SUCCESS) { printf("import failed\n"); exit(1); }
    VkExternalMemoryBufferCreateInfo ext = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, NULL,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, &ext, 0, size,
                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer buffer;
    vkCreateBuffer(dev, &bci, NULL, &buffer);
    vkBindBufferMemory(dev, buffer, *memory, 0);
    return buffer;
}

int main(void) {
    VirtualAlloc2Fn va2 = (VirtualAlloc2Fn)GetProcAddress(GetModuleHandleA("kernelbase.dll"), "VirtualAlloc2");
    MapViewOfFile3Fn mv3 = (MapViewOfFile3Fn)GetProcAddress(GetModuleHandleA("kernelbase.dll"), "MapViewOfFile3");
    const size_t size = 3ull << 30;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_EXECUTE_READWRITE | SEC_COMMIT,
                                        (DWORD)(size >> 32), (DWORD)size, NULL);
    // As the runtime: a reserved placeholder arena, the section's view replacing part of it.
    unsigned char* arena = va2(GetCurrentProcess(), NULL, size * 2, MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                               PAGE_NOACCESS, NULL, 0);
    VirtualFree(arena, size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    unsigned char* view = mv3(section, GetCurrentProcess(), arena, 0, size, MEM_REPLACE_PLACEHOLDER,
                              PAGE_READWRITE, NULL, 0);
    if (!view) { printf("placeholder view failed %lu\n", GetLastError()); return 1; }
    memset(view, 1, size);
    bench("3 GiB placeholder view, not imported", view, size);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vp", 1, "vp", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    vkCreateInstance(&ici, NULL, &inst);
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    float prio = 1.0f;
    family = 0;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, family, 1, &prio};
    const char* exts[] = {"VK_KHR_external_memory", "VK_EXT_external_memory_host"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q, 0, NULL, 2, exts};
    vkCreateDevice(pd, &dci, NULL, &dev);
    vkGetDeviceQueue(dev, family, 0, &queue);

    // Imported in 512 MiB pieces, as the runtime's chunks.
    const size_t chunk = 512ull << 20;
    VkDeviceMemory memories[8];
    VkBuffer buffers[8];
    const int chunks = (int)(size / chunk);
    for (int i = 0; i < chunks; ++i) buffers[i] = import_buffer(pd, view + (size_t)i * chunk, chunk, &memories[i]);
    bench("imported in 512 MiB chunks, GPU idle", view, size);

    // The GPU reads the imported memory (copies into a device-local buffer) while protections change.
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 64 << 20, VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VkBuffer local;
    vkCreateBuffer(dev, &bci, NULL, &local);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, local, &req);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    uint32_t lt = 0;
    while (!((req.memoryTypeBits >> lt) & 1) || !(mp.memoryTypes[lt].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ++lt;
    VkMemoryAllocateInfo lai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, req.size, lt};
    VkDeviceMemory lmem;
    vkAllocateMemory(dev, &lai, NULL, &lmem);
    vkBindBufferMemory(dev, local, lmem, 0);
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, family};
    VkCommandPool pool;
    vkCreateCommandPool(dev, &cpi, NULL, &pool);
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, pool,
                                       VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev, &cai, &cb);
    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cb, &cbi);
    for (int r = 0; r < 40; ++r) {
        for (int i = 0; i < chunks; ++i) {
            VkBufferCopy copy = {(VkDeviceSize)((r * 64ull << 20) % chunk), 0, 64 << 20};
            vkCmdCopyBuffer(cb, buffers[i], local, 1, &copy);
        }
    }
    vkEndCommandBuffer(cb);
    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(dev, &fci, NULL, &fence);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cb};
    const double s0 = now_ms();
    vkQueueSubmit(queue, 1, &si, fence);
    bench("imported, GPU reading it (copies in flight)", view, size);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    printf("(the GPU's copies took %.0f ms)\n", now_ms() - s0);

    // Fragmented protections: every other 64 KiB read-only, as write traps leave them.
    DWORD old;
    for (size_t at = 0; at < size; at += 128 << 10) VirtualProtect(view + at, 64 << 10, PAGE_READONLY, &old);
    bench("imported, protections fragmented (64 KiB)", view, size);
    vkResetFences(dev, 1, &fence);
    vkQueueSubmit(queue, 1, &si, fence);
    bench("fragmented + GPU reading it", view, size);
    vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX);
    return 0;
}
