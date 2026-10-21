/*
 * Copyright 2023 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "gfxstream_vk_private.h"

#include "vk_sync_dummy.h"
#include "vulkan/vulkan_core.h"

/* Under the assumption that Mesa VK runtime queue submission is used, WSI flow
 * sets this temporary state to a dummy sync type (when no explicit dma-buf
 * synchronization is available). For gfxstream, ignore this sync object when
 * this is the case. Synchronization will be done on the host.
 */

static bool isNoopFence(gfxstream_vk_fence* fence) {
    return (fence && fence->base.temporary && vk_sync_type_is_dummy(fence->base.temporary->type));
}

static bool isNoopSemaphore(gfxstream_vk_semaphore* semaphore) {
    return (semaphore && semaphore->base.temporary &&
            vk_sync_type_is_dummy(semaphore->base.temporary->type));
}

std::vector<VkFence> FilterNoopFences(const VkFence* pFences, uint32_t fenceCount) {
    std::vector<VkFence> outFences;
    for (uint32_t j = 0; j < fenceCount; ++j) {
        VK_FROM_HANDLE(gfxstream_vk_fence, gfxstream_fence, pFences[j]);
        if (!isNoopFence(gfxstream_fence)) {
            outFences.push_back(pFences[j]);
        }
    }
    return outFences;
}

std::vector<VkSemaphore> FilterNoopSemaphores(const VkSemaphore* pSemaphores,
                                              uint32_t semaphoreCount) {
    std::vector<VkSemaphore> outSemaphores;
    for (uint32_t j = 0; j < semaphoreCount; ++j) {
        VK_FROM_HANDLE(gfxstream_vk_semaphore, gfxstream_semaphore, pSemaphores[j]);
        if (!isNoopSemaphore(gfxstream_semaphore)) {
            outSemaphores.push_back(pSemaphores[j]);
        }
    }
    return outSemaphores;
}

std::vector<VkSemaphoreSubmitInfo> FilterNoopSemaphoreSubmitInfos(
    const VkSemaphoreSubmitInfo* pSemaphoreSubmitInfos, uint32_t semaphoreSubmitInfoCount) {
    std::vector<VkSemaphoreSubmitInfo> outSemaphoreSubmitInfo;
    for (uint32_t j = 0; j < semaphoreSubmitInfoCount; ++j) {
        VkSemaphoreSubmitInfo outInfo = pSemaphoreSubmitInfos[j];
        VK_FROM_HANDLE(gfxstream_vk_semaphore, gfxstream_semaphore, outInfo.semaphore);
        if (!isNoopSemaphore(gfxstream_semaphore)) {
            outSemaphoreSubmitInfo.push_back(outInfo);
        }
    }
    return outSemaphoreSubmitInfo;
}

float linearChannelToSRGB(float cl) {
    if (cl <= 0.0f)
        return 0.0f;
    else if (cl < 0.0031308f)
        return 12.92f * cl;
    else if (cl < 1.0f)
        return 1.055f * pow(cl, 0.41666f) - 0.055f;
    else
        return 1.0f;
}

float srgbFormatNeedsConversionForClearColor(const VkFormat& format) {
    return format == VK_FORMAT_R8G8B8A8_SRGB;
}

extern "C" {

#define GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_type, vk_type, vk_object_type, finish_member,    \
                                   finish_func)                                               \
    /* Note: create_gfxstream_vk_* calls vk_object_base_init() early during handle            \
     * unmarshaling in VkEncoder so that ResourceTracker can immediately use the              \
     * gfxstream_vk_*_from_handle() functions (which assert on base.type). Entry points for   \
     * non-trivial objects (e.g. gfxstream_vk_CreateDevice,                                   \
     * gfxstream_vk_AllocateCommandBuffers) are still expected to call the corresponding Mesa \
     * runtime vk_*_init() (e.g. vk_device_init) after VkEncoder returns.                     \
     */                                                                                       \
    vk_type create_##gfxstream_type(uint64_t underlying) {                                    \
        struct gfxstream_type* res =                                                          \
            static_cast<gfxstream_type*>(calloc(1, sizeof(gfxstream_type)));                  \
        vk_object_base_init(nullptr, reinterpret_cast<struct vk_object_base*>(res),           \
                            vk_object_type);                                                  \
        res->common.underlying = underlying;                                                  \
        return reinterpret_cast<vk_type>(res);                                                \
    }                                                                                         \
    void delete_##gfxstream_type(vk_type toDelete) {                                          \
        if (!toDelete) return;                                                                \
        auto* obj = reinterpret_cast<gfxstream_type*>(toDelete);                              \
        finish_func(&obj->finish_member);                                                     \
        free(obj);                                                                            \
    }                                                                                         \
    uint64_t gfxstream_type##_to_host_u64(const vk_type obj) {                                \
        if (!obj) return 0;                                                                   \
        return reinterpret_cast<const struct gfxstream_type*>(obj)->common.underlying;        \
    }

GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_instance, VkInstance, VK_OBJECT_TYPE_INSTANCE, base,
                           vk_instance_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_physical_device, VkPhysicalDevice,
                           VK_OBJECT_TYPE_PHYSICAL_DEVICE, base, vk_physical_device_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_device, VkDevice, VK_OBJECT_TYPE_DEVICE, base,
                           vk_device_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_queue, VkQueue, VK_OBJECT_TYPE_QUEUE, base, vk_queue_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_command_pool, VkCommandPool, VK_OBJECT_TYPE_COMMAND_POOL,
                           base, vk_command_pool_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_command_buffer, VkCommandBuffer,
                           VK_OBJECT_TYPE_COMMAND_BUFFER, base, vk_command_buffer_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_buffer, VkBuffer, VK_OBJECT_TYPE_BUFFER, base,
                           vk_buffer_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_fence, VkFence, VK_OBJECT_TYPE_FENCE, base.base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_semaphore, VkSemaphore, VK_OBJECT_TYPE_SEMAPHORE, base.base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_device_memory, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY,
                           base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_image, VkImage, VK_OBJECT_TYPE_IMAGE, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_descriptor_update_template, VkDescriptorUpdateTemplate,
                           VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_descriptor_pool, VkDescriptorPool,
                           VK_OBJECT_TYPE_DESCRIPTOR_POOL, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_descriptor_set, VkDescriptorSet,
                           VK_OBJECT_TYPE_DESCRIPTOR_SET, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_descriptor_set_layout, VkDescriptorSetLayout,
                           VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_sampler, VkSampler, VK_OBJECT_TYPE_SAMPLER, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_private_data_slot, VkPrivateDataSlot,
                           VK_OBJECT_TYPE_PRIVATE_DATA_SLOT, base, vk_object_base_finish)
#ifdef VK_USE_PLATFORM_FUCHSIA
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_buffer_collection_fuchsia, VkBufferCollectionFUCHSIA,
                           VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA, base, vk_object_base_finish)
#endif
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_buffer_view, VkBufferView, VK_OBJECT_TYPE_BUFFER_VIEW, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_image_view, VkImageView, VK_OBJECT_TYPE_IMAGE_VIEW, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_shader_module, VkShaderModule, VK_OBJECT_TYPE_SHADER_MODULE,
                           base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_pipeline, VkPipeline, VK_OBJECT_TYPE_PIPELINE, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_pipeline_cache, VkPipelineCache,
                           VK_OBJECT_TYPE_PIPELINE_CACHE, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_pipeline_layout, VkPipelineLayout,
                           VK_OBJECT_TYPE_PIPELINE_LAYOUT, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_render_pass, VkRenderPass, VK_OBJECT_TYPE_RENDER_PASS, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_framebuffer, VkFramebuffer, VK_OBJECT_TYPE_FRAMEBUFFER,
                           base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_event, VkEvent, VK_OBJECT_TYPE_EVENT, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_query_pool, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_sampler_ycbcr_conversion, VkSamplerYcbcrConversion,
                           VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_surface_khr, VkSurfaceKHR, VK_OBJECT_TYPE_SURFACE_KHR, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_swapchain_khr, VkSwapchainKHR, VK_OBJECT_TYPE_SWAPCHAIN_KHR,
                           base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_display_khr, VkDisplayKHR, VK_OBJECT_TYPE_DISPLAY_KHR, base,
                           vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_display_mode_khr, VkDisplayModeKHR,
                           VK_OBJECT_TYPE_DISPLAY_MODE_KHR, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_validation_cache_ext, VkValidationCacheEXT,
                           VK_OBJECT_TYPE_VALIDATION_CACHE_EXT, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_debug_report_callback_ext, VkDebugReportCallbackEXT,
                           VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_debug_utils_messenger_ext, VkDebugUtilsMessengerEXT,
                           VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_micromap_ext, VkMicromapEXT, VK_OBJECT_TYPE_MICROMAP_EXT,
                           base, vk_object_base_finish)
#ifdef VK_NVX_binary_import
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_cu_module_nvx, VkCuModuleNVX, VK_OBJECT_TYPE_CU_MODULE_NVX,
                           base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_cu_function_nvx, VkCuFunctionNVX,
                           VK_OBJECT_TYPE_CU_FUNCTION_NVX, base, vk_object_base_finish)
#endif
#ifdef VK_NVX_device_generated_commands
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_object_table_nvx, VkObjectTableNVX,
                           VK_OBJECT_TYPE_OBJECT_TABLE_NVX, base, vk_object_base_finish)
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_indirect_commands_layout_nvx, VkIndirectCommandsLayoutNVX,
                           VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NVX, base, vk_object_base_finish)
#endif
#ifdef VK_NV_device_generated_commands
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_indirect_commands_layout_nv, VkIndirectCommandsLayoutNV,
                           VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV, base, vk_object_base_finish)
#endif
#ifdef VK_NV_ray_tracing
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_acceleration_structure_nv, VkAccelerationStructureNV,
                           VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV, base, vk_object_base_finish)
#endif
#ifdef VK_KHR_acceleration_structure
GFXSTREAM_DEFINE_VK_OBJECT(gfxstream_vk_acceleration_structure_khr, VkAccelerationStructureKHR,
                           VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR, base, vk_object_base_finish)
#endif

}  // extern "C"