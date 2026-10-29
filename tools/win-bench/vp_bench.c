// VirtualProtect cost on a section view: plain, then imported by the GPU
// (VK_EXT_external_memory_host), as bbport's guest memory on Windows.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static double now_ms(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}

static volatile int stop_threads = 0;
static unsigned char* g_view;
static size_t g_size;

static DWORD WINAPI writer(LPVOID arg) {
    // Busy threads elsewhere in the process (the game's 16): a protection change must reach them.
    size_t at = (size_t)(uintptr_t)arg * 4096;
    volatile unsigned long long x = 0;
    while (!stop_threads) {
        x += at;
    }
    return 0;
}

static void bench(const char* what, unsigned char* view, size_t size) {
    const size_t pieces[] = {64 << 10, 1 << 20, 16 << 20, 128 << 20};
    for (int p = 0; p < 4; ++p) {
        const size_t piece = pieces[p];
        if (piece > size) continue;
        const int reps = (int)(size / piece) < 32 ? (int)(size / piece) : 32;
        double total = 0.0, worst = 0.0;
        for (int i = 0; i < reps; ++i) {
            unsigned char* at = view + (size_t)i * piece;
            DWORD old;
            const double t0 = now_ms();
            VirtualProtect(at, piece, PAGE_READONLY, &old);
            const double t1 = now_ms();
            VirtualProtect(at, piece, PAGE_READWRITE, &old);
            const double t2 = now_ms();
            total += (t1 - t0) + (t2 - t1);
            if (t1 - t0 > worst) worst = t1 - t0;
            if (t2 - t1 > worst) worst = t2 - t1;
        }
        printf("%-34s piece %7zu KiB: %.3f ms per call (worst %.3f), %.1f ns per page\n", what,
               piece >> 10, total / (2 * reps), worst, total / (2 * reps) * 1e6 / (piece / 4096));
    }
}

int main(int argc, char** argv) {
    const size_t size = 512ull << 20;
    const int threads = argc > 1 ? atoi(argv[1]) : 0;
    HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE | SEC_COMMIT,
                                        (DWORD)(size >> 32), (DWORD)size, NULL);
    unsigned char* view = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size);
    if (!view) { printf("map failed\n"); return 1; }
    memset(view, 1, size);
    g_view = view; g_size = size;
    for (int i = 0; i < threads; ++i) CreateThread(NULL, 0, writer, (LPVOID)(uintptr_t)i, 0, NULL);
    printf("busy threads: %d\n", threads);
    bench("section view, not imported", view, size);

    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "vp", 1, "vp", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &app};
    VkInstance inst;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { printf("no instance\n"); return 1; }
    uint32_t n = 1;
    VkPhysicalDevice pd;
    vkEnumeratePhysicalDevices(inst, &n, &pd);
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    printf("GPU: %s\n", props.deviceName);
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, 0, 1, &prio};
    const char* exts[] = {"VK_KHR_external_memory", "VK_EXT_external_memory_host"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &q, 0, NULL, 2, exts};
    VkDevice dev;
    if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { printf("no device\n"); return 1; }
    PFN_vkGetMemoryHostPointerPropertiesEXT getprops =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev, "vkGetMemoryHostPointerPropertiesEXT");
    VkMemoryHostPointerPropertiesEXT hp = {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    getprops(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view, &hp);
    uint32_t type = 0;
    while (type < 32 && !(hp.memoryTypeBits & (1u << type))) ++type;
    VkImportMemoryHostPointerInfoEXT imp = {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, NULL,
                                            VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, view};
    VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &imp, size, type};
    VkDeviceMemory mem;
    VkResult r = vkAllocateMemory(dev, &mai, NULL, &mem);
    printf("import: %d (type %u of bits %#x)\n", r, type, hp.memoryTypeBits);
    if (r != VK_SUCCESS) return 1;
    bench("section view, imported by the GPU", view, size);
    vkFreeMemory(dev, mem, NULL);
    bench("section view, import freed", view, size);
    stop_threads = 1;
    return 0;
}
