// SPDX-License-Identifier: GPL-2.0-or-later
// bbport vulkan: the Vulkan instance, the chosen GPU and the logical device with what the port
// enables on it: queues (graphics and present, a readback queue of a compute family), the VMA
// allocator, and the device's capabilities as the renderer and the shader translator ask them.
#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "vulkan/platform.h"

namespace Frontend {
class WindowSDL;
struct WindowSystemInfo;
} // namespace Frontend

VK_DEFINE_HANDLE(VmaAllocator)

namespace Vulkan {

class Instance {
public:
    /// The device for the window (`device_index` < 0: the best GPU).
    Instance(Frontend::WindowSDL& window, s32 device_index, bool validation = false,
             bool crash_diagnostic = false);
    /// A device without a window, for the renderer's regression tests.
    Instance(s32 device_index, bool validation);
    ~Instance();
    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    vk::Instance GetInstance() const {
        return *instance;
    }
    vk::PhysicalDevice GetPhysicalDevice() const {
        return physical_device;
    }
    vk::Device GetDevice() const {
        return *device;
    }
    VmaAllocator GetAllocator() const {
        return allocator;
    }

    // Queues. Presenting goes through the graphics queue.
    static constexpr u32 NoFamily = ~0u;
    u32 GetGraphicsQueueFamilyIndex() const {
        return graphics_family;
    }
    u32 GetPresentQueueFamilyIndex() const {
        return graphics_family;
    }
    vk::Queue GetGraphicsQueue() const {
        return graphics_queue;
    }
    vk::Queue GetPresentQueue() const {
        return graphics_queue;
    }
    /// A queue of a compute family without graphics (null when there is none or with
    /// BB_READBACK_QUEUE=0), for reading back finished GPU writes without waiting behind frames.
    vk::Queue GetReadbackQueue() const {
        return readback_queue;
    }
    u32 GetReadbackQueueFamilyIndex() const {
        return readback_family;
    }

    // Identity.
    u32 GetVendorID() const {
        return properties.vendorID;
    }
    vk::DriverId GetDriverID() const {
        return driver_id;
    }
    u32 ApiVersion() const {
        return properties.apiVersion;
    }
    /// AMD GCN 3/4 (GFX8: Polaris, Tonga, Fiji, Carrizo, Stoney): RADV puts the chip in the device
    /// name ("... (RADV POLARIS10)"). The proprietary and PAL drivers name the card only.
    bool IsAmdGfx8() const {
        if (properties.vendorID != 0x1002) {
            return false;
        }
        const std::string_view name{properties.deviceName.data()};
        for (const std::string_view chip :
             {"POLARIS", "TONGA", "FIJI", "ICELAND", "CARRIZO", "STONEY", "VEGAM"}) {
            if (name.find(chip) != std::string_view::npos) {
                return true;
            }
        }
        return false;
    }
    bool IsIntegrated() const {
        return properties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu;
    }

    // Limits.
    vk::DeviceSize UniformMinAlignment() const {
        return properties.limits.minUniformBufferOffsetAlignment;
    }
    vk::DeviceSize StorageMinAlignment() const {
        return properties.limits.minStorageBufferOffsetAlignment;
    }
    vk::DeviceSize NonCoherentAtomSize() const {
        return properties.limits.nonCoherentAtomSize;
    }
    u32 SubgroupSize() const {
        return vk11_props.subgroupSize;
    }
    u32 MaxComputeSharedMemorySize() const {
        return properties.limits.maxComputeSharedMemorySize;
    }
    float MaxSamplerLodBias() const {
        return properties.limits.maxSamplerLodBias;
    }
    float MaxSamplerAnisotropy() const {
        return properties.limits.maxSamplerAnisotropy;
    }
    u32 MaxPushDescriptors() const {
        return max_push_descriptors;
    }
    u32 GetMaxViewportWidth() const {
        return properties.limits.maxViewportDimensions[0];
    }
    u32 GetMaxViewportHeight() const {
        return properties.limits.maxViewportDimensions[1];
    }
    u32 GetMaxFramebufferWidth() const {
        return properties.limits.maxFramebufferWidth;
    }
    u32 GetMaxFramebufferHeight() const {
        return properties.limits.maxFramebufferHeight;
    }
    u32 GetMaxSamplerAllocationCount() const {
        // KosmicKrisp fails sampler creation past 1024 unique samplers whatever it reports.
        return driver_id == vk::DriverId::eMesaKosmickrisp
                   ? 1024u
                   : properties.limits.maxSamplerAllocationCount;
    }
    vk::SampleCountFlags GetColorSampleCounts() const {
        return properties.limits.framebufferColorSampleCounts;
    }
    vk::SampleCountFlags GetDepthSampleCounts() const {
        return properties.limits.framebufferDepthSampleCounts &
               properties.limits.framebufferStencilSampleCounts;
    }
    const vk::PhysicalDeviceVulkan12Properties& GetVk12Properties() const noexcept {
        return vk12_props;
    }
    const vk::PhysicalDeviceMemoryProperties& GetMemoryProperties() const noexcept {
        return memory_properties;
    }

    // Core features.
    bool IsStorageImageWriteWithoutFormatEnabled() const {
        return features.shaderStorageImageWriteWithoutFormat;
    }
    bool IsAnisotropicFilteringSupported() const {
        return features.samplerAnisotropy;
    }
    bool IsDepthBoundsSupported() const {
        return features.depthBounds;
    }
    bool IsShaderFloat64Supported() const {
        return features.shaderFloat64;
    }
    bool IsShaderInt64Supported() const {
        return features.shaderInt64;
    }
    bool IsShaderInt16Supported() const {
        return features.shaderInt16;
    }
    bool IsOcclusionQueryPrecise() const {
        return features.occlusionQueryPrecise;
    }
    bool IsGeometryStageSupported() const {
        return features.geometryShader;
    }
    bool IsTessellationSupported() const {
        return features.tessellationShader;
    }
    bool IsLogicOpSupported() const {
        return features.logicOp;
    }
    bool IsShaderFloat16Supported() const {
        return vk12_features.shaderFloat16;
    }
    bool IsShaderFloat16Enabled() const {
        return vk12_features.shaderFloat16;
    }
    bool IsShaderInt8Supported() const {
        return vk12_features.shaderInt8;
    }
    bool IsBufferInt64AtomicsSupported() const {
        return vk12_features.shaderBufferInt64Atomics;
    }
    bool IsSharedInt64AtomicsSupported() const {
        return vk12_features.shaderSharedInt64Atomics;
    }
    bool IsSubgroupSizeControlEnabled() const {
        return vk13_features.subgroupSizeControl;
    }
    bool IsSubgroupSize64Supported() const {
        return vk13_features.subgroupSizeControl && vk13_props.maxSubgroupSize >= 64;
    }

    // Extensions.
    bool IsMaintenance8Supported() const {
        return ext.maintenance_8;
    }
    bool IsBufferMarkerSupported() const {
        return ext.buffer_marker;
    }
    /// bbport (macOS): images are made so that Metal can compress them losslessly (KosmicKrisp;
    /// BB_TEXTURE_COMPRESSION=0 turns it off). Depth images are made without mutable formats
    /// (their views always use the image's own format).
    bool IsCompressibleImages() const {
        return compressible_images;
    }
    bool IsAttachmentFeedbackLoopLayoutSupported() const {
        return ext.feedback_loop;
    }
    bool IsCustomBorderColorSupported() const {
        return ext.custom_border_color;
    }
    bool IsShaderStencilExportSupported() const {
        return ext.stencil_export;
    }
    bool IsDepthClipControlSupported() const {
        return ext.depth_clip_control;
    }
    bool IsDepthClipEnableSupported() const {
        return ext.depth_clip_enable;
    }
    bool IsDepthRangeUnrestrictedSupported() const {
        return ext.depth_range_unrestricted;
    }
    bool IsExtendedDynamicState3Supported() const {
        return ext.dynamic_state_3;
    }
    bool IsDynamicColorWriteMaskSupported() const {
        return ext.dynamic_state_3 && dynamic_color_write_mask;
    }
    bool IsVertexInputDynamicState() const {
        return ext.vertex_input_dynamic_state;
    }
    bool IsFragmentShaderBarycentricSupported() const {
        return ext.barycentric;
    }
    bool IsAmdShaderExplicitVertexParameterSupported() const {
        return ext.explicit_vertex_parameter;
    }
    bool IsListRestartSupported() const {
        return ext.list_restart && list_restart;
    }
    bool IsPatchListRestartSupported() const {
        return ext.list_restart && patch_list_restart;
    }
    bool IsProvokingVertexSupported() const {
        return ext.provoking_vertex;
    }
    bool IsImageLoadStoreLodSupported() const {
        return ext.image_load_store_lod;
    }
    bool IsAmdGcnShaderSupported() const {
        return ext.amd_gcn_shader;
    }
    bool IsAmdShaderTrinaryMinMaxSupported() const {
        return ext.trinary_minmax;
    }
    bool IsShaderAtomicFloatBuffer32MinMaxSupported() const {
        return ext.atomic_float2 && atomic_float_buffer_minmax;
    }
    bool IsShaderAtomicFloatImage32MinMaxSupported() const {
        return ext.atomic_float2 && atomic_float_image_minmax;
    }
    bool IsWorkgroupMemoryExplicitLayoutSupported() const {
        return ext.workgroup_layout && workgroup_layout_16bit;
    }
    bool IsMixedDepthSamplesSupported() const {
        return ext.nv_mixed_samples || ext.amd_mixed_samples;
    }
    bool IsMixedAnySamplesSupported() const {
        return ext.amd_mixed_samples;
    }
    bool Is2dViewOf3dSupported() const {
        return ext.view_2d_of_3d && view_2d_of_3d;
    }
    bool IsImageViewMinLodSupported() const {
        return ext.image_view_min_lod;
    }
    bool IsBlockTexelViewSupported() const {
        return block_texel_view;
    }
    bool IsShaderSubgroupClockSupported() const {
        return ext.shader_clock && subgroup_clock;
    }
    /// Guest memory can be exported as a dma-buf (BbGuestMemory).
    bool IsGuestMemoryExportSupported() const {
        return ext.memory_fd && ext.dma_buf;
    }
    /// Host memory can be imported (VK_EXT_external_memory_host; BbGuestMemory).
    bool IsHostMemoryImportSupported() const {
        return ext.memory_host;
    }
    /// BB_GUEST_UNCACHED=1 (AMD, diagnostic): the GPU reads the game's memory past its caches
    /// (VK_AMD_device_coherent_memory), as NVIDIA reads imported host memory.
    bool IsGuestUncached() const {
        return ext.guest_uncached;
    }

    // Upscalers.
    /// FSR 4 v07 INT8/DOT4: Float16, Int8, Int16, int8 dot products, linear compute derivatives,
    /// extended storage image formats.
    bool IsFsr4Int8Supported() const {
        return vk12_features.shaderFloat16 && vk12_features.shaderInt8 && features.shaderInt16 &&
               vk13_features.shaderIntegerDotProduct && ext.compute_derivatives &&
               derivatives_linear && features.shaderStorageImageExtendedFormats;
    }
    /// FSR 4.1.1: INT8 model passes and VK_VALVE_shader_mixed_float_dot_product.
    bool IsFsr411Supported() const {
        return IsFsr4Int8Supported() && ext.mixed_float_dot;
    }
    /// FSR 4.1.1 with FP16 matrices (the FP8 variant emulated, for testing).
    bool IsFsr411MatrixSupported() const {
        return IsFsr411Supported() && ext.cooperative_matrix && vk12_features.vulkanMemoryModel &&
               vk13_features.subgroupSizeControl && vk13_features.computeFullSubgroups &&
               vk12_features.storageBuffer8BitAccess;
    }
    /// FSR 4.1.1's FP8 matrix variant (RDNA4).
    bool IsFsr411Fp8Supported() const {
        return IsFsr411MatrixSupported() && ext.float8;
    }

    /// bbport (macOS): VK_EXT_external_memory_metal is enabled (the MetalFX upscaler).
    bool IsExternalMemoryMetalEnabled() const {
        return ext.external_memory_metal;
    }
    /// bbport (macOS): VK_EXT_metal_objects is enabled (shared events of semaphores; MetalFX).
    bool IsMetalObjectsEnabled() const {
        return ext.metal_objects;
    }

    // Memory.
    bool CanReportMemoryUsage() const {
        return ext.memory_budget;
    }
    /// Bytes in use in the heaps the port counts (device local; all on integrated GPUs).
    [[nodiscard]] u64 GetDeviceMemoryUsage() const;
    /// The driver's current budget of those heaps (other processes' use taken out).
    [[nodiscard]] u64 GetDeviceMemoryBudgetNow() const;
    /// What the port plans to use at most (taken at startup).
    [[nodiscard]] u64 GetTotalMemoryBudget() const {
        return total_memory_budget;
    }

    // Formats.
    [[nodiscard]] bool IsFormatSupported(vk::Format format, vk::FormatFeatureFlags2 flags) const;
    /// `format`, or a stand-in with the features when it lacks them.
    [[nodiscard]] vk::Format GetSupportedFormat(vk::Format format,
                                                vk::FormatFeatureFlags2 flags) const;

private:
    Instance(const Frontend::WindowSystemInfo& window_info, s32 device_index, bool validation,
             bool crash_diagnostic);
    void PickPhysicalDevice(s32 device_index);
    void ReadProperties();
    void CreateDevice();
    void CreateAllocator();
    void ComputeMemoryBudget();
    void ReadFormats();
    vk::FormatFeatureFlags2 FormatFeatures(vk::Format format) const;

    /// Extensions enabled on the device.
    struct Extensions {
        bool maintenance_5, maintenance_8, buffer_marker, feedback_loop, depth_range_unrestricted,
            dynamic_state_3, custom_border_color, depth_clip_control, depth_clip_enable,
            vertex_input_dynamic_state, list_restart, explicit_vertex_parameter, barycentric,
            provoking_vertex, stencil_export, image_load_store_lod, amd_gcn_shader,
            trinary_minmax, nv_mixed_samples, amd_mixed_samples, atomic_float, atomic_float2,
            workgroup_layout, view_2d_of_3d, image_view_min_lod, memory_budget, memory_fd,
            dma_buf, memory_host, compute_derivatives, mixed_float_dot, cooperative_matrix,
            float8, shader_clock, guest_uncached,
            external_memory_metal, // bbport (macOS): KosmicKrisp memory as MTLHeaps (MetalFX)
            metal_objects;         // bbport (macOS): semaphores' MTLSharedEvents (MetalFX)
    };

    vk::UniqueInstance instance;
    vk::UniqueDebugUtilsMessengerEXT debug_callback;
    vk::PhysicalDevice physical_device;
    vk::UniqueDevice device;
    VmaAllocator allocator{};
    vk::Queue graphics_queue;
    vk::Queue readback_queue;
    u32 graphics_family = 0;
    u32 readback_family = NoFamily;

    vk::PhysicalDeviceProperties properties;
    vk::PhysicalDeviceMemoryProperties memory_properties;
    vk::PhysicalDeviceVulkan11Properties vk11_props;
    vk::PhysicalDeviceVulkan12Properties vk12_props;
    vk::PhysicalDeviceVulkan13Properties vk13_props;
    vk::DriverId driver_id{};
    u32 max_push_descriptors = 0;
    vk::PhysicalDeviceFeatures features;
    vk::PhysicalDeviceVulkan12Features vk12_features;
    vk::PhysicalDeviceVulkan13Features vk13_features;
    Extensions ext{};
    // Features of extensions the renderer checks one by one.
    bool dynamic_color_write_mask = false;
    bool compressible_images = false;
    bool list_restart = false;
    bool patch_list_restart = false;
    bool atomic_float_buffer_minmax = false;
    bool atomic_float_image_minmax = false;
    bool workgroup_layout_16bit = false;
    bool view_2d_of_3d = false;
    bool subgroup_clock = false;
    bool derivatives_linear = false;
    bool block_texel_view = false;

    std::vector<std::string> available_extensions;
    std::unordered_map<vk::Format, vk::FormatFeatureFlags2> format_features;
    std::vector<u32> counted_heaps; ///< the heaps of the usage and budget figures
    u64 total_memory_budget = 0;
};

/// Memory statistics: device-local VMA blocks and the allocations in them (the difference is
/// free space VMA keeps).
void VmaDeviceUsage(u64& block_bytes, u64& allocation_bytes);

} // namespace Vulkan
