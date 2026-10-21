/*
 * Copyright 2018 Google
 * SPDX-License-Identifier: MIT
 */
#pragma once
#include "util/detect_os.h"

#if DETECT_OS_ANDROID
#include <hardware/hwvulkan.h>
#elif DETECT_OS_LINUX
#include <vulkan/vk_icd.h>
#endif
#include <inttypes.h>
#include <vulkan/vulkan.h>

#include <functional>

#include "VulkanHandles.h"
#include "gfxstream_vk_private.h"

namespace gfxstream {
namespace guest {
class IOStream;
}  // namespace guest
}  // namespace gfxstream

namespace gfxstream {
namespace vk {
class VkEncoder;
struct DescriptorPoolAllocationInfo;
struct ReifiedDescriptorSet;
struct DescriptorSetLayoutInfo;
}  // namespace vk
}  // namespace gfxstream

extern "C" {

// TODO: remove these after fully consolidating to gfxstream_vk_* structs.
#define goldfish_VkBuffer gfxstream_vk_buffer
#define goldfish_VkFence gfxstream_vk_fence
#define goldfish_VkSemaphore gfxstream_vk_semaphore
#define goldfish_VkCommandPool gfxstream_vk_command_pool
#define goldfish_VkCommandBuffer gfxstream_vk_command_buffer
#define goldfish_VkQueue gfxstream_vk_queue
#define goldfish_VkInstance gfxstream_vk_instance
#define goldfish_VkPhysicalDevice gfxstream_vk_physical_device
#define goldfish_VkDevice gfxstream_vk_device
#define goldfish_VkDescriptorPool gfxstream_vk_descriptor_pool
#define goldfish_VkDescriptorSet gfxstream_vk_descriptor_set
#define goldfish_VkDescriptorSetLayout gfxstream_vk_descriptor_set_layout
#define goldfish_VkDeviceMemory gfxstream_vk_device_memory
#define goldfish_VkImage gfxstream_vk_image
#define goldfish_VkDescriptorUpdateTemplate gfxstream_vk_descriptor_update_template
#define goldfish_VkSampler gfxstream_vk_sampler
#define goldfish_VkPrivateDataSlot gfxstream_vk_private_data_slot
#define goldfish_VkBufferCollectionFUCHSIA gfxstream_vk_buffer_collection_fuchsia
#define goldfish_VkBufferView gfxstream_vk_buffer_view
#define goldfish_VkImageView gfxstream_vk_image_view
#define goldfish_VkShaderModule gfxstream_vk_shader_module
#define goldfish_VkPipeline gfxstream_vk_pipeline
#define goldfish_VkPipelineCache gfxstream_vk_pipeline_cache
#define goldfish_VkPipelineLayout gfxstream_vk_pipeline_layout
#define goldfish_VkRenderPass gfxstream_vk_render_pass
#define goldfish_VkFramebuffer gfxstream_vk_framebuffer
#define goldfish_VkEvent gfxstream_vk_event
#define goldfish_VkQueryPool gfxstream_vk_query_pool
#define goldfish_VkSamplerYcbcrConversion gfxstream_vk_sampler_ycbcr_conversion
#define goldfish_VkSurfaceKHR gfxstream_vk_surface_khr
#define goldfish_VkSwapchainKHR gfxstream_vk_swapchain_khr
#define goldfish_VkDisplayKHR gfxstream_vk_display_khr
#define goldfish_VkDisplayModeKHR gfxstream_vk_display_mode_khr
#define goldfish_VkValidationCacheEXT gfxstream_vk_validation_cache_ext
#define goldfish_VkDebugReportCallbackEXT gfxstream_vk_debug_report_callback_ext
#define goldfish_VkDebugUtilsMessengerEXT gfxstream_vk_debug_utils_messenger_ext
#define goldfish_VkMicromapEXT gfxstream_vk_micromap_ext
#define goldfish_VkCuModuleNVX gfxstream_vk_cu_module_nvx
#define goldfish_VkCuFunctionNVX gfxstream_vk_cu_function_nvx
#define goldfish_VkObjectTableNVX gfxstream_vk_object_table_nvx
#define goldfish_VkIndirectCommandsLayoutNVX gfxstream_vk_indirect_commands_layout_nvx
#define goldfish_VkIndirectCommandsLayoutNV gfxstream_vk_indirect_commands_layout_nv
#define goldfish_VkAccelerationStructureNV gfxstream_vk_acceleration_structure_nv
#define goldfish_VkAccelerationStructureKHR gfxstream_vk_acceleration_structure_khr

}  // extern "C"

namespace gfxstream {
namespace vk {

void appendObject(struct gfxstream_vk_object_list** begin, void* val);
void eraseObject(struct gfxstream_vk_object_list** begin, void* val);
void eraseObjects(struct gfxstream_vk_object_list** begin);
void forAllObjects(struct gfxstream_vk_object_list* begin, std::function<void(void*)> func);

}  // namespace vk
}  // namespace gfxstream
