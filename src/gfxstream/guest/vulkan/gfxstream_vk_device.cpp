/*
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <string.h>

#include "GfxStreamConnectionManager.h"
#include "GfxStreamRenderControl.h"
#include "GfxStreamVulkanConnection.h"
#include "ResourceTracker.h"
#include "Resources.h"
#include "VkEncoder.h"
#include "gfxstream_vk_entrypoints.h"
#include "gfxstream_vk_private.h"
#include "util/detect_os.h"
#include "util/perf/cpu_trace.h"
#include "vk_common_entrypoints.h"
#include "vk_sync_dummy.h"
#include "vk_util.h"

uint32_t gSeqno = 0;
uint32_t gNoRenderControlEnc = 0;

static gfxstream::vk::VkEncoder* getVulkanEncoder(GfxStreamConnectionManager* mgr) {
    if (!gNoRenderControlEnc) {
        int32_t ret = renderControlInit(mgr, nullptr);
        if (ret) {
            mesa_loge("Failed to initialize renderControl when getting VK encoder");
            return nullptr;
        }
    }

    gfxstream::vk::VkEncoder* vkEncoder =
        (gfxstream::vk::VkEncoder*)mgr->getEncoder(GFXSTREAM_CONNECTION_VULKAN);

    if (vkEncoder == nullptr) {
        auto stream = mgr->getStream();
        int32_t ret = mgr->addConnection(GFXSTREAM_CONNECTION_VULKAN,
                                         std::make_unique<GfxStreamVulkanConnection>(stream));
        if (ret) {
            return nullptr;
        }

        vkEncoder = (gfxstream::vk::VkEncoder*)mgr->getEncoder(GFXSTREAM_CONNECTION_VULKAN);
    }

    return vkEncoder;
}

static GfxStreamConnectionManager* getConnectionManager(void) {
    auto transport = renderControlGetTransport();
    return GfxStreamConnectionManager::getThreadLocalInstance(transport, kCapsetGfxStreamVulkan);
}

namespace {

static bool instance_extension_table_initialized = false;
static struct vk_instance_extension_table gfxstream_vk_instance_extensions_supported = {};

// Always provided by guest driver only; never encoded/decoded to/from host
static const char* const kGuestEmulatedInstanceExtensions[] = {
    VK_KHR_SURFACE_EXTENSION_NAME,
    VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME,
#if defined(GFXSTREAM_VK_WAYLAND)
    VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME,
#endif
#if defined(GFXSTREAM_VK_X11)
    VK_KHR_XCB_SURFACE_EXTENSION_NAME,
#endif
#if defined(GFXSTREAM_VK_METAL)
    VK_EXT_METAL_SURFACE_EXTENSION_NAME,
#endif
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
};

static bool isGuestEmulatedInstanceExtension(const char* name) {
    for (auto mesaExt : kGuestEmulatedInstanceExtensions) {
        if (!strncmp(mesaExt, name, VK_MAX_EXTENSION_NAME_SIZE)) return true;
    }
    return false;
}

static VkResult SetupInstanceForProcess(void) {
    auto mgr = getConnectionManager();
    if (!mgr) {
        mesa_logd("vulkan: Failed to get host connection\n");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    gfxstream::vk::ResourceTracker::get()->setupCaps(gNoRenderControlEnc);
    gfxstream::vk::ResourceTracker::get()->setupPlatformHelpers();
    // Legacy goldfish path: could be deleted once goldfish not used guest-side.
    if (!gNoRenderControlEnc) {
        struct GfxStreamVkFeatureInfo features = {};
        int32_t ret = renderControlInit(mgr, &features);
        if (ret) {
            mesa_loge("Failed to initialize renderControl ");
            return VK_ERROR_DEVICE_LOST;
        }

        gfxstream::vk::ResourceTracker::get()->setupFeatures(&features);
    }

    gfxstream::vk::ResourceTracker::get()->setThreadingCallbacks({
        .hostConnectionGetFunc = getConnectionManager,
        .vkEncoderGetFunc = getVulkanEncoder,
    });
    gfxstream::vk::ResourceTracker::get()->setSeqnoPtr(&gSeqno);
    gfxstream::vk::VkEncoder* vkEnc = getVulkanEncoder(mgr);
    if (!vkEnc) {
        mesa_loge("vulkan: Failed to get Vulkan encoder\n");
        return VK_ERROR_DEVICE_LOST;
    }

    return VK_SUCCESS;
}

// Filtered extension names for encoding
static std::vector<const char*> filteredInstanceExtensionNames(uint32_t count,
                                                               const char* const* extNames) {
    std::vector<const char*> retList;
    for (uint32_t i = 0; i < count; ++i) {
        auto extName = extNames[i];
        if (!isGuestEmulatedInstanceExtension(extName)) {
            retList.push_back(extName);
        }
    }
    return retList;
}

// Always provided by guest driver only; never encoded/decoded to/from host
static const char* const kGuestEmulatedDeviceExtensions[] = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
    VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
};

static bool isGuestEmulatedDeviceExtension(const char* name) {
    for (auto mesaExt : kGuestEmulatedDeviceExtensions) {
        if (!strncmp(mesaExt, name, VK_MAX_EXTENSION_NAME_SIZE)) return true;
    }
    return false;
}

static std::vector<const char*> filteredDeviceExtensionNames(
    gfxstream_vk_physical_device* physical_device, uint32_t count, const char* const* extNames) {
    std::vector<const char*> retList;
    for (uint32_t i = 0; i < count; ++i) {
        auto extName = extNames[i];
        // VK_EXT_image_drm_format_modifier
        if (!strncmp(extName, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
                     VK_MAX_EXTENSION_NAME_SIZE)) {
            if (physical_device->doImageDrmFormatModifierEmulation) {
                // If emulated, drop this exension from the filtered list
            } else {
                retList.push_back(extName);
            }
        } else if (!isGuestEmulatedDeviceExtension(extName)) {
            retList.push_back(extName);
        }
    }
    return retList;
}

static void get_device_extensions(VkPhysicalDevice physDevInternal,
                                  struct vk_device_extension_table* deviceExts) {
    VkResult result = (VkResult)0;
    auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
    auto resources = gfxstream::vk::ResourceTracker::get();
    uint32_t numDeviceExts = 0;
    result = resources->on_vkEnumerateDeviceExtensionProperties(vkEnc, VK_SUCCESS, physDevInternal,
                                                                NULL, &numDeviceExts, NULL);
    if (result == VK_SUCCESS) {
        std::vector<VkExtensionProperties> extProps(numDeviceExts);
        result = resources->on_vkEnumerateDeviceExtensionProperties(
            vkEnc, VK_SUCCESS, physDevInternal, NULL, &numDeviceExts, extProps.data());
        if (result == VK_SUCCESS) {
            // Enable device extensions from the host's physical device
            for (uint32_t i = 0; i < numDeviceExts; i++) {
                for (uint32_t j = 0; j < VK_DEVICE_EXTENSION_COUNT; j++) {
                    if (0 == strncmp(extProps[i].extensionName,
                                     vk_device_extensions[j].extensionName,
                                     VK_MAX_EXTENSION_NAME_SIZE)) {
                        deviceExts->extensions[j] = true;
                        break;
                    }
                }
            }
            // Make sure all guest-emulated device extensions are enabled
            for (uint32_t j = 0; j < VK_DEVICE_EXTENSION_COUNT; j++) {
                if (isGuestEmulatedDeviceExtension(vk_device_extensions[j].extensionName)) {
                    deviceExts->extensions[j] = true;
                }
            }
        }
    }
}

static VkResult gfxstream_vk_physical_device_init(
    struct gfxstream_vk_physical_device* physical_device, struct gfxstream_vk_instance* instance) {
    struct vk_device_extension_table supported_extensions = {};
    get_device_extensions(gfxstream_vk_physical_device_to_handle(physical_device),
                          &supported_extensions);

    // VK_EXT_image_drm_format_modifier support is either emulated, or passthrough using
    // host functionality
    if (!supported_extensions.EXT_image_drm_format_modifier) {
        physical_device->doImageDrmFormatModifierEmulation = true;
        supported_extensions.EXT_image_drm_format_modifier = true;
    } else {
        physical_device->doImageDrmFormatModifierEmulation = false;
    }

    struct vk_physical_device_dispatch_table dispatch_table;
    memset(&dispatch_table, 0, sizeof(struct vk_physical_device_dispatch_table));
    vk_physical_device_dispatch_table_from_entrypoints(
        &dispatch_table, &gfxstream_vk_physical_device_entrypoints, false);
#if !DETECT_OS_FUCHSIA
    vk_physical_device_dispatch_table_from_entrypoints(&dispatch_table,
                                                       &wsi_physical_device_entrypoints, false);
#endif

    // Initialize the mesa object
    VkResult result = vk_physical_device_init(&physical_device->base, &instance->base,
                                              &supported_extensions, NULL, NULL, &dispatch_table);

    if (result == VK_SUCCESS) {
        physical_device->instance = instance;
        // Note: Must use dummy_sync for correct sync object path in WSI operations
        physical_device->sync_types[0] = &vk_sync_dummy_type;
        physical_device->sync_types[1] = NULL;
        physical_device->base.supported_sync_types = physical_device->sync_types;

        result = gfxstream_vk_wsi_init(physical_device);
    }

    return result;
}

static void gfxstream_vk_destroy_physical_device(struct vk_physical_device* physical_device) {
    auto* gfxstream_physical_device = (struct gfxstream_vk_physical_device*)physical_device;
    VkPhysicalDevice handle = gfxstream_vk_physical_device_to_handle(gfxstream_physical_device);
    gfxstream_vk_wsi_finish(gfxstream_physical_device);
    gfxstream::vk::ResourceTracker::get()->unregister_VkPhysicalDevice(handle);
    delete_gfxstream_vk_physical_device(handle);
}

static VkResult gfxstream_vk_enumerate_devices(struct vk_instance* vk_instance) {
    VkResult result = VK_SUCCESS;
    gfxstream_vk_instance* gfxstream_instance = (gfxstream_vk_instance*)vk_instance;
    VkInstance instance = gfxstream_vk_instance_to_handle(gfxstream_instance);

    if (gfxstream_instance->init_failed) {
        return VK_SUCCESS;
    }

    uint32_t deviceCount = 0;
    auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
    auto resources = gfxstream::vk::ResourceTracker::get();
    result =
        resources->on_vkEnumeratePhysicalDevices(vkEnc, VK_SUCCESS, instance, &deviceCount, NULL);
    if (result != VK_SUCCESS) return result;
    std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
    result = resources->on_vkEnumeratePhysicalDevices(vkEnc, VK_SUCCESS, instance, &deviceCount,
                                                      physicalDevices.data());

    if (result == VK_SUCCESS) {
        for (uint32_t i = 0; i < deviceCount; i++) {
            VK_FROM_HANDLE(gfxstream_vk_physical_device, gfxstream_physicalDevice,
                           physicalDevices[i]);
            result =
                gfxstream_vk_physical_device_init(gfxstream_physicalDevice, gfxstream_instance);
            if (result == VK_SUCCESS) {
                list_addtail(&gfxstream_physicalDevice->base.link,
                             &gfxstream_instance->base.physical_devices.list);
            } else {
                break;
            }
        }
    }

    return result;
}

static struct vk_instance_extension_table* get_instance_extensions() {
    struct vk_instance_extension_table* const retTablePtr =
        &gfxstream_vk_instance_extensions_supported;
    if (!instance_extension_table_initialized) {
        VkResult result = SetupInstanceForProcess();
        if (result == VK_SUCCESS) {
            auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
            auto resources = gfxstream::vk::ResourceTracker::get();
            uint32_t numInstanceExts = 0;
            result = resources->on_vkEnumerateInstanceExtensionProperties(vkEnc, VK_SUCCESS, NULL,
                                                                          &numInstanceExts, NULL);
            if (result == VK_SUCCESS) {
                std::vector<VkExtensionProperties> extProps(numInstanceExts);
                result = resources->on_vkEnumerateInstanceExtensionProperties(
                    vkEnc, VK_SUCCESS, NULL, &numInstanceExts, extProps.data());
                if (result == VK_SUCCESS) {
                    // Enable instance extensions from gfxstream
                    for (uint32_t i = 0; i < numInstanceExts; i++) {
                        for (uint32_t j = 0; j < VK_INSTANCE_EXTENSION_COUNT; j++) {
                            if (0 == strncmp(extProps[i].extensionName,
                                             vk_instance_extensions[j].extensionName,
                                             VK_MAX_EXTENSION_NAME_SIZE)) {
                                gfxstream_vk_instance_extensions_supported.extensions[j] = true;
                                break;
                            }
                        }
                    }
                    // Make sure all guest-emulated instance extensions are enabled
                    for (uint32_t j = 0; j < VK_INSTANCE_EXTENSION_COUNT; j++) {
                        if (isGuestEmulatedInstanceExtension(
                                vk_instance_extensions[j].extensionName)) {
                            gfxstream_vk_instance_extensions_supported.extensions[j] = true;
                        }
                    }
                    instance_extension_table_initialized = true;
                }
            }
        }
    }
    return retTablePtr;
}

}  // namespace

VkResult gfxstream_vk_CreateInstance(const VkInstanceCreateInfo* pCreateInfo,
                                     const VkAllocationCallbacks* pAllocator,
                                     VkInstance* pInstance) {
    MESA_TRACE_SCOPE("vkCreateInstance");

    VkResult result = VK_SUCCESS;
    bool init_failed = (SetupInstanceForProcess() == VK_ERROR_INITIALIZATION_FAILED);
    auto extensions =
        init_failed ? &gfxstream_vk_instance_extensions_supported : get_instance_extensions();

    if (!init_failed) {
        // Full local copy of pCreateInfo
        VkInstanceCreateInfo localCreateInfo = *pCreateInfo;
        std::vector<const char*> filteredExts = filteredInstanceExtensionNames(
            localCreateInfo.enabledExtensionCount, localCreateInfo.ppEnabledExtensionNames);
        localCreateInfo.enabledExtensionCount = static_cast<uint32_t>(filteredExts.size());
        localCreateInfo.ppEnabledExtensionNames = filteredExts.data();

        auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
        result = vkEnc->vkCreateInstance(&localCreateInfo, nullptr, pInstance, true /* do lock */);
        if (result != VK_SUCCESS) {
            return vk_error(NULL, result);
        }
    } else {
        *pInstance = create_gfxstream_vk_instance(0);
        if (!*pInstance) {
            return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);
        }
    }

    VK_FROM_HANDLE(gfxstream_vk_instance, instance, *pInstance);
    instance->init_failed = init_failed;

    struct vk_instance_dispatch_table dispatch_table;
    memset(&dispatch_table, 0, sizeof(struct vk_instance_dispatch_table));
    vk_instance_dispatch_table_from_entrypoints(&dispatch_table, &gfxstream_vk_instance_entrypoints,
                                                false);
#if !DETECT_OS_FUCHSIA
    vk_instance_dispatch_table_from_entrypoints(&dispatch_table, &wsi_instance_entrypoints, false);
#endif

    pAllocator = pAllocator ?: vk_default_allocator();
    result =
        vk_instance_init(&instance->base, extensions, &dispatch_table, pCreateInfo, pAllocator);
    if (result != VK_SUCCESS) {
        list_inithead(&instance->base.physical_devices.list);
        list_inithead(&instance->base.debug_utils.callbacks);
        list_inithead(&instance->base.debug_utils.instance_callbacks);
        if (!instance->init_failed) {
            auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
            vkEnc->vkDestroyInstance(*pInstance, pAllocator, true /* do lock */);
        } else {
            delete_gfxstream_vk_instance(*pInstance);
        }
        *pInstance = VK_NULL_HANDLE;
        return vk_error(NULL, result);
    }

    // Note: Do not support try_create_for_drm. virtio_gpu DRM device opened in
    // init_renderer above, which can still enumerate multiple physical devices on the host.
    instance->base.physical_devices.enumerate = gfxstream_vk_enumerate_devices;
    instance->base.physical_devices.destroy = gfxstream_vk_destroy_physical_device;

    return VK_SUCCESS;
}

void gfxstream_vk_DestroyInstance(VkInstance _instance, const VkAllocationCallbacks* pAllocator) {
    MESA_TRACE_SCOPE("vkDestroyInstance");
    if (_instance == VK_NULL_HANDLE) return;

    VK_FROM_HANDLE(gfxstream_vk_instance, instance, _instance);

    if (!instance->init_failed) {
        auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
        vkEnc->vkDestroyInstance(_instance, pAllocator, true /* do lock */);
    } else {
        delete_gfxstream_vk_instance(_instance);
    }

    // To make End2EndTests happy, since now the host connection is statically linked to
    // libvulkan_ranchu.so [separate HostConnections now].
#if defined(END2END_TESTS)
    GfxStreamConnectionManager::resetThreadLocalInstance();
    VirtGpuDevice::resetInstance();
    gSeqno = 0;
#endif
}

VkResult gfxstream_vk_EnumerateInstanceExtensionProperties(const char* pLayerName,
                                                           uint32_t* pPropertyCount,
                                                           VkExtensionProperties* pProperties) {
    MESA_TRACE_SCOPE("vkvkEnumerateInstanceExtensionProperties");
    (void)pLayerName;

    return vk_enumerate_instance_extension_properties(get_instance_extensions(), pPropertyCount,
                                                      pProperties);
}

VkResult gfxstream_vk_EnumerateDeviceExtensionProperties(VkPhysicalDevice physicalDevice,
                                                         const char* pLayerName,
                                                         uint32_t* pPropertyCount,
                                                         VkExtensionProperties* pProperties) {
    MESA_TRACE_SCOPE("vkEnumerateDeviceExtensionProperties");
    (void)pLayerName;
    VK_FROM_HANDLE(vk_physical_device, pdevice, physicalDevice);

    VK_OUTARRAY_MAKE_TYPED(VkExtensionProperties, out, pProperties, pPropertyCount);

    for (int i = 0; i < VK_DEVICE_EXTENSION_COUNT; i++) {
        if (!pdevice->supported_extensions.extensions[i]) continue;

        vk_outarray_append_typed(VkExtensionProperties, &out, prop) {
            *prop = vk_device_extensions[i];
        }
    }

    return vk_outarray_status(&out);
}

static void gfxstream_vk_queue_fini(struct gfxstream_vk_queue* gfxstream_queue) {
    VkQueue queue = gfxstream_vk_queue_to_handle(gfxstream_queue);
    gfxstream::vk::ResourceTracker::get()->unregister_VkQueue(queue);
    delete_gfxstream_vk_queue(queue);
}

static VkResult gfxstream_vk_queue_init(struct gfxstream_vk_device* dev,
                                        const VkDeviceQueueCreateInfo* queue_info,
                                        uint32_t queue_index) {
    const VkDeviceQueueInfo2 device_queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
        .pNext = NULL,
        .flags = queue_info->flags,
        .queueFamilyIndex = queue_info->queueFamilyIndex,
        .queueIndex = queue_index,
    };

    VkQueue queue = VK_NULL_HANDLE;
    auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
    vkEnc->vkGetDeviceQueue2(gfxstream_vk_device_to_handle(dev), &device_queue_info, &queue,
                             true /* do lock */);
    if (queue == VK_NULL_HANDLE) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    VK_FROM_HANDLE(gfxstream_vk_queue, gfxstream_queue, queue);
    VkResult result = vk_queue_init(&gfxstream_queue->base, &dev->base, queue_info, queue_index);
    if (result != VK_SUCCESS) {
        gfxstream::vk::ResourceTracker::get()->unregister_VkQueue(queue);
        list_del(&gfxstream_queue->base.link);
        vk_object_base_finish(&gfxstream_queue->base.base);
        free(gfxstream_queue);
        return result;
    }

    gfxstream_queue->device = dev;

    return VK_SUCCESS;
}

static VkResult gfxstream_vk_device_init_queues(struct gfxstream_vk_device* dev,
                                                const VkDeviceCreateInfo* create_info) {
    for (uint32_t i = 0; i < create_info->queueCreateInfoCount; i++) {
        const VkDeviceQueueCreateInfo* queue_info = &create_info->pQueueCreateInfos[i];
        for (uint32_t j = 0; j < queue_info->queueCount; j++) {
            VkResult result = gfxstream_vk_queue_init(dev, queue_info, j);
            if (result != VK_SUCCESS) {
                vk_foreach_queue_safe(queue, &dev->base) {
                    gfxstream_vk_queue_fini((struct gfxstream_vk_queue*)queue);
                }
                return result;
            }
        }
    }

    return VK_SUCCESS;
}

static VkResult gfxstream_vk_device_queue_family_init(struct gfxstream_vk_device* dev,
                                                      const VkDeviceCreateInfo* create_info) {
    const VkAllocationCallbacks* alloc = &dev->base.alloc;
    uint32_t* queue_families = NULL;
    uint32_t count = 0;

    queue_families =
        (uint32_t*)vk_zalloc(alloc, sizeof(*queue_families) * create_info->queueCreateInfoCount,
                             GFXSTREAM_DEFAULT_ALIGN, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
    if (!queue_families) return VK_ERROR_OUT_OF_HOST_MEMORY;

    for (uint32_t i = 0; i < create_info->queueCreateInfoCount; i++) {
        const uint32_t index = create_info->pQueueCreateInfos[i].queueFamilyIndex;
        bool new_index = true;

        for (uint32_t j = 0; j < count; j++) {
            if (queue_families[j] == index) {
                new_index = false;
                break;
            }
        }
        if (new_index) queue_families[count++] = index;
    }

    dev->queue_families = queue_families;
    dev->queue_family_count = count;

    return VK_SUCCESS;
}

static inline void gfxstream_vk_device_queue_family_fini(struct gfxstream_vk_device* dev) {
    vk_free(&dev->base.alloc, dev->queue_families);
}

VkResult gfxstream_vk_CreateDevice(VkPhysicalDevice physicalDevice,
                                   const VkDeviceCreateInfo* pCreateInfo,
                                   const VkAllocationCallbacks* pAllocator, VkDevice* pDevice) {
    MESA_TRACE_SCOPE("vkCreateDevice");
    VK_FROM_HANDLE(gfxstream_vk_physical_device, gfxstream_physicalDevice, physicalDevice);
    VkResult result = (VkResult)0;

    /*
     * Android's libvulkan implements VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT, but
     * passes it to the underlying driver anyways.  See:
     *
     * https://android-review.googlesource.com/c/platform/hardware/google/gfxstream/+/2839438
     *
     * and associated bugs. Mesa VK runtime also checks this, so we have to filter out before
     * reaches it.
     */
    VkPhysicalDeviceSwapchainMaintenance1FeaturesEXT* mutableSwapchainMaintenance1Features =
        vk_find_struct(const_cast<VkDeviceCreateInfo*>(pCreateInfo),
                       PHYSICAL_DEVICE_SWAPCHAIN_MAINTENANCE_1_FEATURES_EXT);
    if (mutableSwapchainMaintenance1Features) {
        mutableSwapchainMaintenance1Features->swapchainMaintenance1 = VK_FALSE;
    }

    const VkAllocationCallbacks* pMesaAllocator =
        pAllocator ?: &gfxstream_physicalDevice->instance->base.alloc;


    // Full local copy of pCreateInfo
    VkDeviceCreateInfo localCreateInfo = *pCreateInfo;

    std::vector<const char*> filteredExts = filteredDeviceExtensionNames(
        gfxstream_physicalDevice, localCreateInfo.enabledExtensionCount,
        localCreateInfo.ppEnabledExtensionNames);
    localCreateInfo.enabledExtensionCount = static_cast<uint32_t>(filteredExts.size());
    localCreateInfo.ppEnabledExtensionNames = filteredExts.data();

    auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
    result = vkEnc->vkCreateDevice(physicalDevice, &localCreateInfo, pAllocator, pDevice,
                                   true /* do lock */);
    if (result != VK_SUCCESS) {
        return result;
    }

    VK_FROM_HANDLE(gfxstream_vk_device, gfxstream_device, *pDevice);

    struct vk_device_dispatch_table dispatch_table;
    memset(&dispatch_table, 0, sizeof(struct vk_device_dispatch_table));
    vk_device_dispatch_table_from_entrypoints(&dispatch_table, &gfxstream_vk_device_entrypoints,
                                              false);
#if !DETECT_OS_FUCHSIA
    vk_device_dispatch_table_from_entrypoints(&dispatch_table, &wsi_device_entrypoints, false);
#endif

    result = vk_device_init(&gfxstream_device->base, &gfxstream_physicalDevice->base,
                            &dispatch_table, pCreateInfo, pMesaAllocator);
    if (result != VK_SUCCESS) {
        list_inithead(&gfxstream_device->base.queues);
        vkEnc->vkDestroyDevice(*pDevice, pAllocator, true /* do lock */);
        *pDevice = VK_NULL_HANDLE;
        return result;
    }

    gfxstream_device->physical_device = gfxstream_physicalDevice;
    gfxstream_device->base.command_dispatch_table = &gfxstream_device->cmd_dispatch;

    result = gfxstream_vk_device_queue_family_init(gfxstream_device, pCreateInfo);
    if (result != VK_SUCCESS) {
        vkEnc->vkDestroyDevice(*pDevice, pAllocator, true /* do lock */);
        *pDevice = VK_NULL_HANDLE;
        return result;
    }

    result = gfxstream_vk_device_init_queues(gfxstream_device, pCreateInfo);
    if (result != VK_SUCCESS) {
        gfxstream_vk_device_queue_family_fini(gfxstream_device);
        vkEnc->vkDestroyDevice(*pDevice, pAllocator, true /* do lock */);
        *pDevice = VK_NULL_HANDLE;
        return result;
    }

    return VK_SUCCESS;
}

void gfxstream_vk_DestroyDevice(VkDevice device, const VkAllocationCallbacks* pAllocator) {
    MESA_TRACE_SCOPE("vkDestroyDevice");
    if (device == VK_NULL_HANDLE) return;
    VK_FROM_HANDLE(gfxstream_vk_device, gfxstream_device, device);

    vk_foreach_queue_safe(queue, &gfxstream_device->base) {
        gfxstream_vk_queue_fini((struct gfxstream_vk_queue*)queue);
    }

    gfxstream_vk_device_queue_family_fini(gfxstream_device);

    auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
    vkEnc->vkDestroyDevice(device, pAllocator, true /* do lock */);
}

/* The loader wants us to expose a second GetInstanceProcAddr function
 * to work around certain LD_PRELOAD issues seen in apps.
 */
extern "C" PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char* pName);

extern "C" PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char* pName) {
    return gfxstream_vk_GetInstanceProcAddr(instance, pName);
}

PFN_vkVoidFunction gfxstream_vk_GetInstanceProcAddr(VkInstance _instance, const char* pName) {
    VK_FROM_HANDLE(gfxstream_vk_instance, instance, _instance);
    return vk_instance_get_proc_addr(&instance->base, &gfxstream_vk_instance_entrypoints, pName);
}

PFN_vkVoidFunction gfxstream_vk_GetDeviceProcAddr(VkDevice _device, const char* pName) {
    MESA_TRACE_SCOPE("vkGetDeviceProcAddr");
    VK_FROM_HANDLE(gfxstream_vk_device, device, _device);
    return vk_device_get_proc_addr(&device->base, pName);
}

VkResult gfxstream_vk_AllocateMemory(VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo,
                                     const VkAllocationCallbacks* pAllocator,
                                     VkDeviceMemory* pMemory) {
    MESA_TRACE_SCOPE("vkAllocateMemory");
    VkResult vkAllocateMemory_VkResult_return = (VkResult)0;
    {
        auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
        auto resources = gfxstream::vk::ResourceTracker::get();
        vkAllocateMemory_VkResult_return = resources->on_vkAllocateMemory(
            vkEnc, VK_SUCCESS, device, pAllocateInfo, pAllocator, pMemory);
    }
    return vkAllocateMemory_VkResult_return;
}

VkResult gfxstream_vk_EnumerateInstanceLayerProperties(uint32_t* pPropertyCount,
                                                       VkLayerProperties* pProperties) {
    MESA_TRACE_SCOPE("vkEnumerateInstanceLayerProperties");
    auto result = SetupInstanceForProcess();
    if (result != VK_SUCCESS) {
        return vk_error(NULL, result);
    }

    VkResult vkEnumerateInstanceLayerProperties_VkResult_return = (VkResult)0;
    {
        auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
        vkEnumerateInstanceLayerProperties_VkResult_return =
            vkEnc->vkEnumerateInstanceLayerProperties(pPropertyCount, pProperties,
                                                      true /* do lock */);
    }
    return vkEnumerateInstanceLayerProperties_VkResult_return;
}

VkResult gfxstream_vk_EnumerateInstanceVersion(uint32_t* pApiVersion) {
    MESA_TRACE_SCOPE("vkEnumerateInstanceVersion");
    auto result = SetupInstanceForProcess();
    if (result != VK_SUCCESS) {
        return vk_error(NULL, result);
    }

    VkResult vkEnumerateInstanceVersion_VkResult_return = (VkResult)0;
    {
        auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
        vkEnumerateInstanceVersion_VkResult_return =
            vkEnc->vkEnumerateInstanceVersion(pApiVersion, true /* do lock */);
    }
    return vkEnumerateInstanceVersion_VkResult_return;
}

static bool vk_descriptor_type_has_descriptor_buffer(VkDescriptorType type) {
    switch (type) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            return true;
        default:
            return false;
    }
}

static std::vector<VkWriteDescriptorSet> transformDescriptorSetList(
    const VkWriteDescriptorSet* pDescriptorSets, uint32_t descriptorSetCount,
    std::vector<std::vector<VkDescriptorBufferInfo>>& bufferInfos) {
    std::vector<VkWriteDescriptorSet> outDescriptorSets(descriptorSetCount);
    bufferInfos.resize(descriptorSetCount);
    for (uint32_t i = 0; i < descriptorSetCount; ++i) {
        const auto& srcDescriptorSet = pDescriptorSets[i];
        const uint32_t descriptorCount = srcDescriptorSet.descriptorCount;

        VkWriteDescriptorSet& outDescriptorSet = outDescriptorSets[i];
        outDescriptorSet = srcDescriptorSet;

        std::vector<VkDescriptorBufferInfo>& bufferInfo = bufferInfos[i];
        bufferInfo.resize(descriptorCount);
        for (uint32_t j = 0; j < descriptorCount; ++j) {
            const auto* srcBufferInfo = srcDescriptorSet.pBufferInfo;
            if (srcBufferInfo) {
                bufferInfo[j] = srcBufferInfo[j];
                bufferInfo[j].buffer = VK_NULL_HANDLE;
                if (vk_descriptor_type_has_descriptor_buffer(srcDescriptorSet.descriptorType) &&
                    srcBufferInfo[j].buffer) {
                    bufferInfo[j].buffer = srcBufferInfo[j].buffer;
                }
            } else {
                memset(&bufferInfo[j], 0, sizeof(VkDescriptorBufferInfo));
            }
        }
        outDescriptorSet.pBufferInfo = bufferInfos[i].data();
    }
    return outDescriptorSets;
}

void gfxstream_vk_UpdateDescriptorSets(VkDevice device, uint32_t descriptorWriteCount,
                                       const VkWriteDescriptorSet* pDescriptorWrites,
                                       uint32_t descriptorCopyCount,
                                       const VkCopyDescriptorSet* pDescriptorCopies) {
    MESA_TRACE_SCOPE("vkUpdateDescriptorSets");
    {
        auto vkEnc = gfxstream::vk::ResourceTracker::getThreadLocalEncoder();
        std::vector<std::vector<VkDescriptorBufferInfo>> descriptorBufferInfoStorage;
        std::vector<VkWriteDescriptorSet> internal_pDescriptorWrites = transformDescriptorSetList(
            pDescriptorWrites, descriptorWriteCount, descriptorBufferInfoStorage);
        auto resources = gfxstream::vk::ResourceTracker::get();
        resources->on_vkUpdateDescriptorSets(vkEnc, device, descriptorWriteCount,
                                             internal_pDescriptorWrites.data(), descriptorCopyCount,
                                             pDescriptorCopies);
    }
}

/*
 * Override vk_common_SetDebugUtilsObjectNameEXT.
 *
 * Mesa's common implementation assumes every Vulkan object handle is a pointer
 * to a vk_object_base. In gfxstream, only a subset of objects carry a Mesa
 * wrapper; all other non-dispatchable handles are opaque host handles. Passing
 * such a handle to the common path causes a null-pointer dereference.
 *
 * For wrapped objects we delegate to the common implementation. For unwrapped
 * objects we return VK_SUCCESS as a no-op; the names would only become useful
 * once guest-to-host forwarding is implemented.
 */
VkResult gfxstream_vk_SetDebugUtilsObjectNameEXT(VkDevice device,
                                                 const VkDebugUtilsObjectNameInfoEXT* pNameInfo) {
    MESA_TRACE_SCOPE("vkSetDebugUtilsObjectNameEXT");
    /*
     * These object types have a gfxstream Mesa wrapper containing a
     * vk_object_base, so the common implementation can safely dereference
     * the handle. This list corresponds to the gfxstream_vk_* structs
     * defined in gfxstream_vk_private.h.
     */
    switch (pNameInfo->objectType) {
        case VK_OBJECT_TYPE_INSTANCE:
        case VK_OBJECT_TYPE_PHYSICAL_DEVICE:
        case VK_OBJECT_TYPE_DEVICE:
        case VK_OBJECT_TYPE_QUEUE:
        case VK_OBJECT_TYPE_COMMAND_BUFFER:
        case VK_OBJECT_TYPE_COMMAND_POOL:
        case VK_OBJECT_TYPE_BUFFER:
        case VK_OBJECT_TYPE_FENCE:
        case VK_OBJECT_TYPE_SEMAPHORE:
            return vk_common_SetDebugUtilsObjectNameEXT(device, pNameInfo);
        default:
            return VK_SUCCESS;
    }
}
