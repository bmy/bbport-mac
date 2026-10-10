/* Check whether native-size depth/stencil images can be blitted to reduced
 * renderer targets. The renderer needs both directions for live presets.
 * --gamepads: the connected gamepads, "GUID<tab>name" per line (the launcher's controller list,
 * BB_GAMEPAD). --displays: the monitors, "name<tab>WxH<tab>primary (1 or 0)" per line in SDL's
 * order (the launcher's monitor list, BB_DISPLAY; issue #69). --read-input: one key or button for the
 * launcher's controls (below). --device: "vendorID<tab>name" of the GPU the game takes (run.sh:
 * memory model and driver workarounds by vendor and chip). --memory-import: whether the GPU can
 * use memory the program already has in place (VK_EXT_external_memory_host, what the new memory
 * model needs off Linux's dma-buf), tried for real on anonymous and shared memory. */
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include <sys/mman.h>
#include <unistd.h>

static VkDeviceSize largest_local_heap(VkPhysicalDevice device) {
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    VkDeviceSize largest = 0;
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) &&
            memory.memoryHeaps[i].size > largest) {
            largest = memory.memoryHeaps[i].size;
        }
    }
    return largest;
}

static int better_device(VkPhysicalDevice candidate, VkPhysicalDevice current) {
    VkPhysicalDeviceProperties next, old;
    vkGetPhysicalDeviceProperties(candidate, &next);
    vkGetPhysicalDeviceProperties(current, &old);
    const int next_api = next.apiVersion >= VK_API_VERSION_1_3;
    const int old_api = old.apiVersion >= VK_API_VERSION_1_3;
    if (next_api != old_api) return next_api;
    const int next_discrete = next.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    const int old_discrete = old.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    if (next_discrete != old_discrete) return next_discrete;
    const int next_cpu = next.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    const int old_cpu = old.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
    if (next_cpu != old_cpu) return !next_cpu;
    return largest_local_heap(candidate) > largest_local_heap(current);
}

/* --live-resolution: prints 1 when live resolution changes suit the GPU (run.sh, setting
 * live_resolution=auto), else 0. Live scaling keeps the game's post-processing at 1080p and
 * copies scene targets every frame: fine on a strong discrete GPU, 7-8 FPS on the Steam Deck
 * and a GTX 1060. Rule: discrete, at least 8 GB of device memory, and not an NVIDIA GPU older
 * than Turing (no fragment shader barycentrics; its depth/stencil copies take nine draws). */
static int has_extension(VkPhysicalDevice device, const char *name) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, NULL, &count, NULL) != VK_SUCCESS) return 0;
    VkExtensionProperties *list = calloc(count ? count : 1, sizeof(*list));
    int found = 0;
    if (list && vkEnumerateDeviceExtensionProperties(device, NULL, &count, list) == VK_SUCCESS)
        for (uint32_t i = 0; i < count && !found; ++i) found = !strcmp(list[i].extensionName, name);
    free(list);
    return found;
}

static int live_resolution_suits(VkPhysicalDevice device) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(device, &props);
    const int discrete = props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    const VkDeviceSize memory = largest_local_heap(device);
    const int old_nvidia = props.vendorID == 0x10de &&
                           !has_extension(device, "VK_KHR_fragment_shader_barycentric");
    const int suits = discrete && memory >= (VkDeviceSize)7680 << 20 && !old_nvidia;
    fprintf(stderr, "GPU: %s, %s, %llu MiB: live resolution changes %s\n", props.deviceName,
            discrete ? "discrete" : "integrated or other", (unsigned long long)(memory >> 20),
            suits ? "on" : "off (startup resolution patch)");
    return suits;
}

static int list_gamepads(void) {
    if (!SDL_Init(SDL_INIT_GAMEPAD)) {
        fprintf(stderr, "gamepads: %s\n", SDL_GetError());
        return 1;
    }
    // Some devices (HIDAPI) show up only after events are pumped.
    for (int i = 0; i < 5; ++i) {
        SDL_PumpEvents();
        SDL_Delay(40);
    }
    int count = 0;
    SDL_JoystickID *ids = SDL_GetGamepads(&count);
    for (int i = 0; ids && i < count; ++i) {
        char guid[33];
        SDL_GUIDToString(SDL_GetGamepadGUIDForID(ids[i]), guid, sizeof guid);
        const char *name = SDL_GetGamepadNameForID(ids[i]);
        printf("%s\t%s\n", guid, name ? name : "?");
    }
    SDL_free(ids);
    SDL_Quit();
    return 0;
}

static int list_displays(void) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "displays: %s\n", SDL_GetError());
        return 1;
    }
    const SDL_DisplayID primary = SDL_GetPrimaryDisplay();
    int count = 0;
    SDL_DisplayID *ids = SDL_GetDisplays(&count);
    for (int i = 0; ids && i < count; ++i) {
        const char *name = SDL_GetDisplayName(ids[i]);
        const SDL_DisplayMode *mode = SDL_GetDesktopDisplayMode(ids[i]);
        printf("%s\t%dx%d\t%d\n", name && *name ? name : "?", mode ? mode->w : 0,
               mode ? mode->h : 0, ids[i] == primary);
    }
    SDL_free(ids);
    SDL_Quit();
    return 0;
}

/* --read-input key|pad: a small window; prints "key <SDL key name>" ("Mouse Left", "Wheel Up", ...
 * for the mouse) or "pad <SDL button name>"|pad: a small window; prints "key <SDL key name>" or "pad <SDL button name>"
 * (lefttrigger/righttrigger for the triggers) for the first key or gamepad button pressed, the
 * names bbport.ini's key.* and pad.* lines take. Escape, closing it or 15 s: nothing. */
static int read_input(const char *kind) {
    const int want_key = strcmp(kind, "pad") != 0, want_pad = strcmp(kind, "key") != 0;
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
        fprintf(stderr, "read-input: %s\n", SDL_GetError());
        return 1;
    }
    SDL_Window *window = NULL;
    SDL_Renderer *renderer = NULL;
    const char *prompt = want_key && want_pad ? "Press a key, a mouse button or a gamepad button"
                         : want_key           ? "Press a key or a mouse button"
                                              : "Press a gamepad button";
    if (!SDL_CreateWindowAndRenderer("bbport", 520, 90, 0, &window, &renderer)) {
        fprintf(stderr, "read-input: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }
    int count = 0;
    SDL_JoystickID *ids = SDL_GetGamepads(&count);
    for (int i = 0; ids && i < count; ++i) SDL_OpenGamepad(ids[i]);
    SDL_free(ids);
    const Uint64 end = SDL_GetTicks() + 15000;
    int done = 0;
    while (!done && SDL_GetTicks() < end) {
        SDL_SetRenderDrawColor(renderer, 24, 24, 28, 255);
        SDL_RenderClear(renderer);
        SDL_SetRenderDrawColor(renderer, 230, 230, 230, 255);
        SDL_RenderDebugText(renderer, 16, 30, prompt);
        SDL_RenderDebugText(renderer, 16, 50, "Escape: cancel");
        SDL_RenderPresent(renderer);
        SDL_Event e;
        while (!done && SDL_WaitEventTimeout(&e, 50)) {
            switch (e.type) {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                done = 1;
                break;
            case SDL_EVENT_GAMEPAD_ADDED:
                SDL_OpenGamepad(e.gdevice.which);
                break;
            case SDL_EVENT_KEY_DOWN:
                if (e.key.scancode == SDL_SCANCODE_ESCAPE) {
                    done = 1;
                } else if (want_key) {
                    printf("key %s\n", SDL_GetScancodeName(e.key.scancode));
                    done = 1;
                }
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN: /* the names runtime_pad.c's key.* lines take */
                if (want_key && e.button.button >= SDL_BUTTON_LEFT && e.button.button <= SDL_BUTTON_X2) {
                    static const char *const names[] = {"", "Mouse Left", "Mouse Middle", "Mouse Right",
                                                        "Mouse X1", "Mouse X2"};
                    printf("key %s\n", names[e.button.button]);
                    done = 1;
                }
                break;
            case SDL_EVENT_MOUSE_WHEEL:
                if (want_key && e.wheel.y != 0) {
                    const float y = e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -e.wheel.y : e.wheel.y;
                    printf("key %s\n", y > 0 ? "Wheel Up" : "Wheel Down");
                    done = 1;
                }
                break;
            case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
                if (want_pad) {
                    printf("pad %s\n", SDL_GetGamepadStringForButton((SDL_GamepadButton)e.gbutton.button));
                    done = 1;
                }
                break;
            case SDL_EVENT_GAMEPAD_AXIS_MOTION:
                if (want_pad && e.gaxis.value > 16000 &&
                    (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER || e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER)) {
                    printf("pad %s\n", e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER ? "lefttrigger" : "righttrigger");
                    done = 1;
                }
                break;
            default:
                break;
            }
        }
    }
    fflush(stdout);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}


/* ---- --memory-import ---------------------------------------------------------------------- */

static const char *yes_no(int value) { return value ? "yes" : "no"; }

static void print_memory_types(VkPhysicalDevice device) {
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(device, &memory);
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i)
        printf("  heap %u: %llu MiB%s\n", i, (unsigned long long)(memory.memoryHeaps[i].size >> 20),
               memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT ? ", device-local" : "");
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        const VkMemoryPropertyFlags f = memory.memoryTypes[i].propertyFlags;
        printf("  type %u (heap %u):%s%s%s%s\n", i, memory.memoryTypes[i].heapIndex,
               f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT ? " device-local" : "",
               f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT ? " host-visible" : "",
               f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT ? " coherent" : "",
               f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT ? " cached" : "");
    }
}

/* Imports `size` bytes at `pointer` as Vulkan memory, binds a buffer to it, has the GPU fill part
 * of it and checks the program sees the GPU's writes there and the GPU sees the program's. */
static void try_import(VkPhysicalDevice physical, VkDevice device, uint32_t queue_family,
                       const char *what, void *pointer, VkDeviceSize size) {
    PFN_vkGetMemoryHostPointerPropertiesEXT get_props =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(
            device, "vkGetMemoryHostPointerPropertiesEXT");
    const VkExternalMemoryHandleTypeFlagBits handle =
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    VkMemoryHostPointerPropertiesEXT host = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult r = get_props ? get_props(device, handle, pointer, &host) : VK_ERROR_EXTENSION_NOT_PRESENT;
    if (r != VK_SUCCESS || !host.memoryTypeBits) {
        printf("%s: the driver won't take this memory (result %d, memory types %#x)\n", what, r,
               host.memoryTypeBits);
        return;
    }
    VkPhysicalDeviceMemoryProperties memory;
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount && type == UINT32_MAX; ++i)
        if (host.memoryTypeBits & (1u << i)) type = i;
    const VkExternalMemoryBufferCreateInfo external_buffer = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO, .handleTypes = handle};
    const VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &external_buffer, .size = size,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buffer = VK_NULL_HANDLE, other = VK_NULL_HANDLE;
    VkDeviceMemory bound = VK_NULL_HANDLE, other_memory = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if ((r = vkCreateBuffer(device, &buffer_info, NULL, &buffer)) != VK_SUCCESS) {
        printf("%s: memory types %#x, but no buffer for it (result %d)\n", what, host.memoryTypeBits, r);
        return;
    }
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, buffer, &reqs);
    const VkImportMemoryHostPointerInfoEXT import = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT, .handleType = handle,
        .pHostPointer = pointer};
    const VkMemoryAllocateInfo allocate = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                           .pNext = &import, .allocationSize = size,
                                           .memoryTypeIndex = type};
    if (!(reqs.memoryTypeBits & host.memoryTypeBits)) {
        printf("%s: the buffer can't use the imported memory's types (%#x vs %#x)\n", what,
               reqs.memoryTypeBits, host.memoryTypeBits);
        goto done;
    }
    if ((r = vkAllocateMemory(device, &allocate, NULL, &bound)) != VK_SUCCESS) {
        printf("%s: import failed (result %d)\n", what, r);
        goto done;
    }
    if ((r = vkBindBufferMemory(device, buffer, bound, 0)) != VK_SUCCESS) {
        printf("%s: imported, but binding a buffer failed (result %d)\n", what, r);
        goto done;
    }
    /* The GPU writes the first half; the program's pattern in the second half is copied by the
     * GPU into a buffer of the driver's own memory and compared. */
    uint32_t *words = pointer;
    const VkDeviceSize half = size / 2, count = half / 4;
    for (VkDeviceSize i = 0; i < count; ++i) words[count + i] = (uint32_t)(i * 2654435761u);
    const VkBufferCreateInfo other_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = half,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    vkCreateBuffer(device, &other_info, NULL, &other);
    VkMemoryRequirements other_reqs;
    vkGetBufferMemoryRequirements(device, other, &other_reqs);
    uint32_t visible = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount && visible == UINT32_MAX; ++i)
        if ((other_reqs.memoryTypeBits & (1u << i)) &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            visible = i;
    const VkMemoryAllocateInfo other_alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                              .allocationSize = other_reqs.size,
                                              .memoryTypeIndex = visible};
    if (visible == UINT32_MAX || vkAllocateMemory(device, &other_alloc, NULL, &other_memory) != VK_SUCCESS ||
        vkBindBufferMemory(device, other, other_memory, 0) != VK_SUCCESS) {
        printf("%s: imported, but no memory for the comparison buffer\n", what);
        goto done;
    }
    const VkCommandPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                               .queueFamilyIndex = queue_family};
    vkCreateCommandPool(device, &pool_info, NULL, &pool);
    const VkCommandBufferAllocateInfo cb_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb;
    vkAllocateCommandBuffers(device, &cb_info, &cb);
    const VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &begin);
    vkCmdFillBuffer(cb, buffer, 0, half, 0x5eedf00du);
    const VkBufferCopy copy = {.srcOffset = half, .dstOffset = 0, .size = half};
    vkCmdCopyBuffer(cb, buffer, other, 1, &copy);
    const VkMemoryBarrier to_host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                     .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                     .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &to_host, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cb);
    const VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(device, &fence_info, NULL, &fence);
    VkQueue queue;
    vkGetDeviceQueue(device, queue_family, 0, &queue);
    const VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
                                 .pCommandBuffers = &cb};
    if ((r = vkQueueSubmit(queue, 1, &submit, fence)) != VK_SUCCESS ||
        (r = vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ull)) != VK_SUCCESS) {
        printf("%s: imported, but the GPU test didn't finish (result %d)\n", what, r);
        goto done;
    }
    VkDeviceSize gpu_wrote = 0, gpu_read = 0;
    for (VkDeviceSize i = 0; i < count; ++i) gpu_wrote += words[i] == 0x5eedf00du;
    uint32_t *copied = NULL;
    if (vkMapMemory(device, other_memory, 0, half, 0, (void **)&copied) == VK_SUCCESS) {
        for (VkDeviceSize i = 0; i < count; ++i) gpu_read += copied[i] == (uint32_t)(i * 2654435761u);
        vkUnmapMemory(device, other_memory);
    }
    printf("%s: IMPORTED (memory type %u). The program sees the GPU's writes: %s; the GPU sees "
           "the program's data: %s\n", what, type, gpu_wrote == count ? "yes" : "NO",
           gpu_read == count ? "yes" : "NO");
done:
    if (fence) vkDestroyFence(device, fence, NULL);
    if (pool) vkDestroyCommandPool(device, pool, NULL);
    if (other) vkDestroyBuffer(device, other, NULL);
    if (other_memory) vkFreeMemory(device, other_memory, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    if (bound) vkFreeMemory(device, bound, NULL);
}

static int memory_import(VkPhysicalDevice physical) {
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);
    VkPhysicalDeviceFeatures features;
    vkGetPhysicalDeviceFeatures(physical, &features);
    printf("GPU: %s (vendor %#06x), Vulkan %u.%u.%u, driver %#x\n", props.deviceName,
           props.vendorID, VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion),
           props.driverVersion);
    printf("Program: %s, page size %ld\n",
#if defined(__x86_64__)
           "x86-64 (under Rosetta on Apple Silicon)",
#else
           "native",
#endif
           sysconf(_SC_PAGESIZE));
    const int host_import = has_extension(physical, "VK_EXT_external_memory_host");
    printf("VK_EXT_external_memory_host (host memory import): %s\n", yes_no(host_import));
    printf("VK_KHR_external_memory_fd: %s; VK_EXT_external_memory_dma_buf: %s; "
           "VK_EXT_external_memory_metal: %s\n",
           yes_no(has_extension(physical, "VK_KHR_external_memory_fd")),
           yes_no(has_extension(physical, "VK_EXT_external_memory_dma_buf")),
           yes_no(has_extension(physical, "VK_EXT_external_memory_metal")));
    printf("Sparse binding: %s; sparse buffers: %s; buffer device address: %s\n",
           yes_no(features.sparseBinding), yes_no(features.sparseResidencyBuffer),
           yes_no(has_extension(physical, "VK_KHR_buffer_device_address") ||
                  props.apiVersion >= VK_API_VERSION_1_2));
    printf("Memory:\n");
    print_memory_types(physical);
    if (!host_import) {
        printf("Result: no host memory import; the new memory model has no way in on this "
               "driver.\n");
        return 0;
    }
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT host = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                          .pNext = &host};
    vkGetPhysicalDeviceProperties2(physical, &props2);
    const VkDeviceSize align = host.minImportedHostPointerAlignment;
    printf("Imported pointers must be aligned to %llu bytes\n", (unsigned long long)align);

    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, NULL);
    VkQueueFamilyProperties *family = calloc(families ? families : 1, sizeof(*family));
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &families, family);
    uint32_t queue_family = 0;
    for (uint32_t i = 0; i < families; ++i)
        if (family[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT)) {
            queue_family = i;
            break;
        }
    free(family);
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                           .queueFamilyIndex = queue_family, .queueCount = 1,
                                           .pQueuePriorities = &priority};
    const char *extensions[] = {"VK_KHR_external_memory", "VK_EXT_external_memory_host",
                                "VK_KHR_portability_subset"};
    const uint32_t extension_count = has_extension(physical, "VK_KHR_portability_subset") ? 3 : 2;
    const VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                            .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue,
                                            .enabledExtensionCount = extension_count,
                                            .ppEnabledExtensionNames = extensions};
    VkDevice device = VK_NULL_HANDLE;
    VkResult r = vkCreateDevice(physical, &device_info, NULL, &device);
    if (r != VK_SUCCESS) {
        printf("Result: the extension is listed, but a device with it can't be created (result %d)\n", r);
        return 0;
    }
    /* 64 MiB, as the runtime's direct memory chunks come; anonymous memory, and a shared memory
     * object like the runtime's own (shm_open), the kind the game's memory actually is. */
    const VkDeviceSize size = 64ull << 20;
    const size_t page = (size_t)sysconf(_SC_PAGESIZE);
    void *anon = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (anon != MAP_FAILED) {
        try_import(physical, device, queue_family, "Anonymous memory", anon, size);
        munmap(anon, size);
    }
    char name[64];
    snprintf(name, sizeof name, "/bbport-import-%ld", (long)getpid());
    const int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    if (fd >= 0) {
        shm_unlink(name);
        void *shared = MAP_FAILED;
        if (ftruncate(fd, (off_t)size) == 0)
            shared = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (shared != MAP_FAILED) {
            try_import(physical, device, queue_family, "Shared memory (as the game's)", shared, size);
            /* The game maps its memory at 4 KiB granularity: a pointer that is only 4 KiB aligned. */
            if (align <= page && page > 4096) {
                printf("(4 KiB-aligned pointers can't be tried: the page size here is %zu)\n", page);
            } else if (align <= 4096) {
                try_import(physical, device, queue_family, "Shared memory at a 4 KiB offset",
                           (char *)shared + 4096, size - 8192);
            } else {
                printf("Shared memory at a 4 KiB offset: not allowed (the driver needs %llu-byte "
                       "alignment; the game maps 4 KiB pieces)\n", (unsigned long long)align);
            }
            munmap(shared, size);
        }
        close(fd);
    }
    vkDestroyDevice(device, NULL);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "--read-input")) {
        return read_input(argc > 2 ? argv[2] : "any");
    }
    if (argc > 1 && !strcmp(argv[1], "--gamepads")) {
        return list_gamepads();
    }
    if (argc > 1 && !strcmp(argv[1], "--displays")) {
        return list_displays();
    }
    const int live_mode = argc > 1 && !strcmp(argv[1], "--live-resolution");
    const int device_mode = argc > 1 && !strcmp(argv[1], "--device");
    const int import_mode = argc > 1 && !strcmp(argv[1], "--memory-import");
    const VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "bbport scene scaling probe",
        .apiVersion = VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo create = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create, NULL, &instance) != VK_SUCCESS) {
        fputs("GPU scene scaling: cannot create Vulkan instance\n", stderr);
        return 1;
    }
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, NULL) != VK_SUCCESS || !count) {
        fputs("GPU scene scaling: no Vulkan device\n", stderr);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (!devices || vkEnumeratePhysicalDevices(instance, &count, devices) != VK_SUCCESS) {
        fputs("GPU scene scaling: cannot enumerate Vulkan devices\n", stderr);
        free(devices);
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    /* Match vk_instance.cpp's default ranking or its explicit BB_GPU_ID index. */
    VkPhysicalDevice selected = devices[0];
    const char *gpu_id = getenv("BB_GPU_ID");
    if (gpu_id && atoi(gpu_id) >= 0) {
        const unsigned long index = strtoul(gpu_id, NULL, 10);
        if (index >= count) {
            fputs("GPU scene scaling: BB_GPU_ID is outside the device list\n", stderr);
            free(devices);
            vkDestroyInstance(instance, NULL);
            return 1;
        }
        selected = devices[index];
    } else {
        for (uint32_t i = 1; i < count; ++i)
            if (better_device(devices[i], selected)) selected = devices[i];
    }
    if (device_mode) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(selected, &props);
        printf("%#06x\t%s\n", props.vendorID, props.deviceName);
        free(devices);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    if (import_mode) {
        const int result = memory_import(selected);
        free(devices);
        vkDestroyInstance(instance, NULL);
        return result;
    }
    if (live_mode) {
        printf("%d\n", live_resolution_suits(selected));
        free(devices);
        vkDestroyInstance(instance, NULL);
        return 0;
    }
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(selected, &props);
    const struct { VkFormat format; const char *name; } formats[] = {
        {VK_FORMAT_R8G8B8A8_UNORM, "RGBA8"},
        {VK_FORMAT_R8G8B8A8_SRGB, "RGBA8 sRGB"},
        {VK_FORMAT_B10G11R11_UFLOAT_PACK32, "B10G11R11"},
        {VK_FORMAT_R16G16B16A16_SFLOAT, "RGBA16F"},
        {VK_FORMAT_D32_SFLOAT_S8_UINT, "D32S8"},
    };
    int supported = 1;
    for (size_t i = 0; i < sizeof(formats) / sizeof(formats[0]); ++i) {
        VkFormatProperties features;
        vkGetPhysicalDeviceFormatProperties(selected, formats[i].format, &features);
        const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                              VK_FORMAT_FEATURE_BLIT_DST_BIT;
        if ((features.optimalTilingFeatures & required) != required) {
            fprintf(stderr, "GPU scene scaling: %s lacks blit support for %s\n",
                    props.deviceName, formats[i].name);
            supported = 0;
        }
    }
    if (supported) fprintf(stderr, "GPU scene scaling: %s supports live presets\n", props.deviceName);
    free(devices);
    vkDestroyInstance(instance, NULL);
    return supported ? 0 : 1;
}
