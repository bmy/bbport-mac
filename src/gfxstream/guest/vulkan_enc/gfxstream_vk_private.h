/*
 * Copyright © 2023 Google Inc.
 *
 * derived from panvk_private.h driver which is:
 * Copyright © 2021 Collabora Ltd.
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 * Copyright © 2015 Intel Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef GFXSTREAM_VK_PRIVATE_H
#define GFXSTREAM_VK_PRIVATE_H

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vk_icd.h>
#include <vulkan/vulkan.h>

#include <vector>

#include "gfxstream_vk_entrypoints.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_device.h"
#include "vk_extensions.h"
#include "vk_fence.h"
#include "vk_image.h"
#include "vk_instance.h"
#include "vk_log.h"
#include "vk_object.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_semaphore.h"
#include "vulkan/wsi/wsi_common.h"

#define GFXSTREAM_DEFAULT_ALIGN 8

namespace gfxstream {
namespace guest {
class IOStream;
}  // namespace guest
namespace vk {
class VkEncoder;
struct DescriptorPoolAllocationInfo;
struct ReifiedDescriptorSet;
struct DescriptorSetLayoutInfo;
}  // namespace vk
}  // namespace gfxstream

#define GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_type, vk_type)         \
    extern "C" vk_type create_##gfxstream_type(uint64_t underlying); \
    extern "C" void delete_##gfxstream_type(vk_type toDelete);       \
    extern "C" uint64_t gfxstream_type##_to_host_u64(const vk_type obj);

struct gfxstream_vk_object_common {
    // The untyped host handle.
    uint64_t underlying;
};

struct gfxstream_vk_instance {
    // Must be first member.
    struct vk_instance base;

    struct gfxstream_vk_object_common common;

    uint32_t api_version;
    bool init_failed;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_instance, base.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_instance, VkInstance)

struct gfxstream_vk_physical_device {
    // Must be first member.
    struct vk_physical_device base;

    struct gfxstream_vk_object_common common;

    struct wsi_device wsi_device;
    const struct vk_sync_type* sync_types[2];
    struct gfxstream_vk_instance* instance;
    bool doImageDrmFormatModifierEmulation;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_physical_device, base.base, VkPhysicalDevice,
                       VK_OBJECT_TYPE_PHYSICAL_DEVICE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_physical_device, VkPhysicalDevice)

struct gfxstream_vk_device {
    // Must be first member.
    struct vk_device base;

    struct gfxstream_vk_object_common common;

    struct vk_device_dispatch_table cmd_dispatch;
    struct gfxstream_vk_physical_device* physical_device;

    /* unique queue family indices in which to create the device queues */
    uint32_t* queue_families;
    uint32_t queue_family_count;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_device, base.base, VkDevice, VK_OBJECT_TYPE_DEVICE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_device, VkDevice)

struct gfxstream_vk_queue {
    // Must be first member.
    struct vk_queue base;

    struct gfxstream_vk_object_common common;

    struct gfxstream_vk_device* device;

    gfxstream::vk::VkEncoder* lastUsedEncoder;
    uint32_t sequenceNumber;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_queue, base.base, VkQueue, VK_OBJECT_TYPE_QUEUE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_queue, VkQueue)

struct gfxstream_vk_buffer {
    // Must be first member.
    struct vk_buffer base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_buffer, base.base, VkBuffer, VK_OBJECT_TYPE_BUFFER)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_buffer, VkBuffer)

struct gfxstream_vk_object_list {
    void* obj;
    struct gfxstream_vk_object_list* next;
};

struct gfxstream_vk_command_pool {
    // Must be first member.
    struct vk_command_pool base;

    struct gfxstream_vk_object_common common;

    // The command buffers allocated from this pool.
    struct gfxstream_vk_object_list* subObjects;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_command_pool, base.base, VkCommandPool,
                               VK_OBJECT_TYPE_COMMAND_POOL)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_command_pool, VkCommandPool)

struct gfxstream_vk_command_buffer {
    // Must be first member.
    struct vk_command_buffer base;

    struct gfxstream_vk_object_common common;

    gfxstream::vk::VkEncoder* lastUsedEncoder;
    uint32_t sequenceNumber;
    gfxstream::vk::VkEncoder* privateEncoder;
    gfxstream::guest::IOStream* privateStream;
    uint32_t flags;
    struct gfxstream_vk_object_list* poolObjects;
    struct gfxstream_vk_object_list* subObjects;
    struct gfxstream_vk_object_list* superObjects;
    void* userPtr;
    bool isSecondary;
    VkDevice device;
};
VK_DEFINE_HANDLE_CASTS(gfxstream_vk_command_buffer, base.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_command_buffer, VkCommandBuffer)

struct gfxstream_vk_fence {
    // Must be first member.
    struct vk_fence base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_fence, base.base, VkFence, VK_OBJECT_TYPE_FENCE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_fence, VkFence)

struct gfxstream_vk_semaphore {
    // Must be first member.
    struct vk_semaphore base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_semaphore, base.base, VkSemaphore,
                               VK_OBJECT_TYPE_SEMAPHORE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_semaphore, VkSemaphore)

struct gfxstream_vk_descriptor_pool {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;

    gfxstream::vk::DescriptorPoolAllocationInfo* allocInfo;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_descriptor_pool, base, VkDescriptorPool,
                               VK_OBJECT_TYPE_DESCRIPTOR_POOL)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_descriptor_pool, VkDescriptorPool)

struct gfxstream_vk_descriptor_set {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;

    gfxstream::vk::ReifiedDescriptorSet* reified;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_descriptor_set, base, VkDescriptorSet,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_descriptor_set, VkDescriptorSet)

struct gfxstream_vk_descriptor_set_layout {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
    gfxstream::vk::DescriptorSetLayoutInfo* layoutInfo;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_descriptor_set_layout, base, VkDescriptorSetLayout,
                               VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_descriptor_set_layout, VkDescriptorSetLayout)

struct gfxstream_vk_device_memory {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_device_memory, base, VkDeviceMemory,
                               VK_OBJECT_TYPE_DEVICE_MEMORY)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_device_memory, VkDeviceMemory)

struct gfxstream_vk_image {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_image, base, VkImage, VK_OBJECT_TYPE_IMAGE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_image, VkImage)

struct gfxstream_vk_descriptor_update_template {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_descriptor_update_template, base,
                               VkDescriptorUpdateTemplate,
                               VK_OBJECT_TYPE_DESCRIPTOR_UPDATE_TEMPLATE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_descriptor_update_template, VkDescriptorUpdateTemplate)

struct gfxstream_vk_sampler {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_sampler, base, VkSampler, VK_OBJECT_TYPE_SAMPLER)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_sampler, VkSampler)

struct gfxstream_vk_private_data_slot {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_private_data_slot, base, VkPrivateDataSlot,
                               VK_OBJECT_TYPE_PRIVATE_DATA_SLOT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_private_data_slot, VkPrivateDataSlot)

#ifdef VK_USE_PLATFORM_FUCHSIA
struct gfxstream_vk_buffer_collection_fuchsia {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_buffer_collection_fuchsia, base,
                               VkBufferCollectionFUCHSIA, VK_OBJECT_TYPE_BUFFER_COLLECTION_FUCHSIA)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_buffer_collection_fuchsia, VkBufferCollectionFUCHSIA)
#endif

struct gfxstream_vk_buffer_view {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_buffer_view, base, VkBufferView,
                               VK_OBJECT_TYPE_BUFFER_VIEW)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_buffer_view, VkBufferView)

struct gfxstream_vk_image_view {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_image_view, base, VkImageView,
                               VK_OBJECT_TYPE_IMAGE_VIEW)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_image_view, VkImageView)

struct gfxstream_vk_shader_module {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_shader_module, base, VkShaderModule,
                               VK_OBJECT_TYPE_SHADER_MODULE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_shader_module, VkShaderModule)

struct gfxstream_vk_pipeline {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_pipeline, base, VkPipeline, VK_OBJECT_TYPE_PIPELINE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_pipeline, VkPipeline)

struct gfxstream_vk_pipeline_cache {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_pipeline_cache, base, VkPipelineCache,
                               VK_OBJECT_TYPE_PIPELINE_CACHE)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_pipeline_cache, VkPipelineCache)

struct gfxstream_vk_pipeline_layout {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_pipeline_layout, base, VkPipelineLayout,
                               VK_OBJECT_TYPE_PIPELINE_LAYOUT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_pipeline_layout, VkPipelineLayout)

struct gfxstream_vk_render_pass {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_render_pass, base, VkRenderPass,
                               VK_OBJECT_TYPE_RENDER_PASS)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_render_pass, VkRenderPass)

struct gfxstream_vk_framebuffer {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_framebuffer, base, VkFramebuffer,
                               VK_OBJECT_TYPE_FRAMEBUFFER)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_framebuffer, VkFramebuffer)

struct gfxstream_vk_event {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_event, base, VkEvent, VK_OBJECT_TYPE_EVENT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_event, VkEvent)

struct gfxstream_vk_query_pool {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_query_pool, base, VkQueryPool,
                               VK_OBJECT_TYPE_QUERY_POOL)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_query_pool, VkQueryPool)

struct gfxstream_vk_sampler_ycbcr_conversion {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_sampler_ycbcr_conversion, base,
                               VkSamplerYcbcrConversion, VK_OBJECT_TYPE_SAMPLER_YCBCR_CONVERSION)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_sampler_ycbcr_conversion, VkSamplerYcbcrConversion)

struct gfxstream_vk_surface_khr {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_surface_khr, base, VkSurfaceKHR,
                               VK_OBJECT_TYPE_SURFACE_KHR)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_surface_khr, VkSurfaceKHR)

struct gfxstream_vk_swapchain_khr {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_swapchain_khr, base, VkSwapchainKHR,
                               VK_OBJECT_TYPE_SWAPCHAIN_KHR)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_swapchain_khr, VkSwapchainKHR)

struct gfxstream_vk_display_khr {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_display_khr, base, VkDisplayKHR,
                               VK_OBJECT_TYPE_DISPLAY_KHR)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_display_khr, VkDisplayKHR)

struct gfxstream_vk_display_mode_khr {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_display_mode_khr, base, VkDisplayModeKHR,
                               VK_OBJECT_TYPE_DISPLAY_MODE_KHR)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_display_mode_khr, VkDisplayModeKHR)

struct gfxstream_vk_validation_cache_ext {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_validation_cache_ext, base, VkValidationCacheEXT,
                               VK_OBJECT_TYPE_VALIDATION_CACHE_EXT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_validation_cache_ext, VkValidationCacheEXT)

struct gfxstream_vk_debug_report_callback_ext {
    // Must be first member.
    struct vk_object_base base;
    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_debug_report_callback_ext, base,
                               VkDebugReportCallbackEXT, VK_OBJECT_TYPE_DEBUG_REPORT_CALLBACK_EXT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_debug_report_callback_ext, VkDebugReportCallbackEXT)

struct gfxstream_vk_debug_utils_messenger_ext {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_debug_utils_messenger_ext, base,
                               VkDebugUtilsMessengerEXT, VK_OBJECT_TYPE_DEBUG_UTILS_MESSENGER_EXT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_debug_utils_messenger_ext, VkDebugUtilsMessengerEXT)

struct gfxstream_vk_micromap_ext {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_micromap_ext, base, VkMicromapEXT,
                               VK_OBJECT_TYPE_MICROMAP_EXT)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_micromap_ext, VkMicromapEXT)

#ifdef VK_NVX_binary_import
struct gfxstream_vk_cu_module_nvx {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_cu_module_nvx, base, VkCuModuleNVX,
                               VK_OBJECT_TYPE_CU_MODULE_NVX)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_cu_module_nvx, VkCuModuleNVX)

struct gfxstream_vk_cu_function_nvx {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_cu_function_nvx, base, VkCuFunctionNVX,
                               VK_OBJECT_TYPE_CU_FUNCTION_NVX)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_cu_function_nvx, VkCuFunctionNVX)
#endif

#ifdef VK_NVX_device_generated_commands
struct gfxstream_vk_object_table_nvx {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_object_table_nvx, base, VkObjectTableNVX,
                               VK_OBJECT_TYPE_OBJECT_TABLE_NVX)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_object_table_nvx, VkObjectTableNVX)

struct gfxstream_vk_indirect_commands_layout_nvx {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_indirect_commands_layout_nvx, base,
                               VkIndirectCommandsLayoutNVX,
                               VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NVX)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_indirect_commands_layout_nvx, VkIndirectCommandsLayoutNVX)
#endif

#ifdef VK_NV_device_generated_commands
struct gfxstream_vk_indirect_commands_layout_nv {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_indirect_commands_layout_nv, base,
                               VkIndirectCommandsLayoutNV,
                               VK_OBJECT_TYPE_INDIRECT_COMMANDS_LAYOUT_NV)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_indirect_commands_layout_nv, VkIndirectCommandsLayoutNV)
#endif

#ifdef VK_NV_ray_tracing
struct gfxstream_vk_acceleration_structure_nv {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_acceleration_structure_nv, base,
                               VkAccelerationStructureNV, VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_NV)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_acceleration_structure_nv, VkAccelerationStructureNV)
#endif

#ifdef VK_KHR_acceleration_structure
struct gfxstream_vk_acceleration_structure_khr {
    // Must be first member.
    struct vk_object_base base;

    struct gfxstream_vk_object_common common;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(gfxstream_vk_acceleration_structure_khr, base,
                               VkAccelerationStructureKHR,
                               VK_OBJECT_TYPE_ACCELERATION_STRUCTURE_KHR)
GFXSTREAM_DECLARE_VK_OBJECT(gfxstream_vk_acceleration_structure_khr, VkAccelerationStructureKHR)
#endif

VkResult gfxstream_vk_wsi_init(struct gfxstream_vk_physical_device* physical_device);

void gfxstream_vk_wsi_finish(struct gfxstream_vk_physical_device* physical_device);

std::vector<VkSemaphore> FilterNoopSemaphores(const VkSemaphore* pSemaphores,
                                              uint32_t semaphoreCount);

std::vector<VkFence> FilterNoopFences(const VkFence* pFences, uint32_t fenceCount);

std::vector<VkSemaphoreSubmitInfo> FilterNoopSemaphoreSubmitInfos(
    const VkSemaphoreSubmitInfo* pSemaphoreSubmitInfos, uint32_t semaphoreSubmitInfoCount);

float linearChannelToSRGB(float cl);
float srgbFormatNeedsConversionForClearColor(const VkFormat& format);

#endif /* GFXSTREAM_VK_PRIVATE_H */
