// SPDX-License-Identifier: GPL-2.0-or-later
#include "vulkan/instance.h"
#include "metal/mtl_layer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fmt/ranges.h>
#include <vk_mem_alloc.h>

#ifdef __APPLE__
#include <atomic>
#include <chrono>
#include <mach/mach.h>

#include "platform/macos/memory_macos.h"
#include "remote/bb_remote.h"
#endif

#include "bbport_toggles.h"
#include "render/gcn_to_vk.h"
#include "sdl_window.h"
#include "upscale/vk_dlss.h"

namespace Vulkan {

namespace {

/// BB_VRAM_LIMIT_MB=N (tests): the driver's VRAM budget taken as at most N MiB, at startup and
/// while the game runs, so the texture collector and VRAM copies behave as on a smaller card
/// (a 6 GB GTX 1660 Ti reports ~5600). Allocations themselves are not limited.
u64 VramLimit() {
    static const u64 limit = [] {
        const char* env = std::getenv("BB_VRAM_LIMIT_MB");
        const u64 mib = env && *env ? std::strtoull(env, nullptr, 10) : 0;
        if (mib != 0) {
            std::printf("GPU: VRAM budget limited to %llu MiB (BB_VRAM_LIMIT_MB)\n",
                        (unsigned long long)mib);
        }
        return mib << 20;
    }();
    return limit;
}

u64 LargestDeviceLocalHeap(const vk::PhysicalDeviceMemoryProperties& memory) {
    u64 largest = 0;
    for (u32 i = 0; i < memory.memoryHeapCount; ++i) {
        if (memory.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            largest = std::max<u64>(largest, memory.memoryHeaps[i].size);
        }
    }
    return largest;
}

/// How good a GPU is for the port: the API version first, then discrete over integrated, real
/// hardware over software renderers (they report system memory as device local), then VRAM.
std::tuple<bool, bool, bool, u64> Rank(vk::PhysicalDevice device) {
    const auto props = device.getProperties();
    return {props.apiVersion >= TargetVulkanApiVersion,
            props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu,
            props.deviceType != vk::PhysicalDeviceType::eCpu,
            LargestDeviceLocalHeap(device.getMemoryProperties())};
}

std::string VersionString(u32 version) {
    return fmt::format("{}.{}.{}", VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version),
                       VK_VERSION_PATCH(version));
}

/// The driver version as its vendor writes it (NVIDIA and Intel's Windows driver pack it their
/// own way).
std::string DriverVersionString(vk::DriverId driver, u32 version) {
    if (driver == vk::DriverId::eNvidiaProprietary) {
        return fmt::format("{}.{}.{}.{}", version >> 22, (version >> 14) & 0xff,
                           (version >> 6) & 0xff, version & 0x3f);
    }
    if (driver == vk::DriverId::eIntelProprietaryWindows) {
        return fmt::format("{}.{}", version >> 14, version & 0x3fff);
    }
    return VersionString(version);
}

/// Appends `features` to the pNext chain ending at `*tail`.
template <typename T>
void Link(void**& tail, T& features) {
    *tail = &features;
    tail = &features.pNext;
}

const Instance* stats_instance = nullptr;

} // namespace

Instance::Instance(Frontend::WindowSDL& window, s32 device_index, bool validation,
                   bool crash_diagnostic)
    : Instance(window.GetWindowInfo(), device_index, validation, crash_diagnostic) {}

Instance::Instance(s32 device_index, bool validation)
    : Instance(Frontend::WindowSystemInfo{}, device_index, validation, false) {}

Instance::Instance(const Frontend::WindowSystemInfo& window_info, s32 device_index,
                   bool validation, bool crash_diagnostic)
    : instance{CreateInstance(window_info.type, validation, crash_diagnostic)} {
    if (validation) {
        debug_callback = CreateDebugCallback(*instance);
    }
    PickPhysicalDevice(device_index);
    ReadProperties();
    ASSERT_MSG(properties.apiVersion >= TargetVulkanApiVersion,
               "the GPU supports Vulkan {}.{}; the port needs {}.{}",
               VK_VERSION_MAJOR(properties.apiVersion), VK_VERSION_MINOR(properties.apiVersion),
               VK_VERSION_MAJOR(TargetVulkanApiVersion), VK_VERSION_MINOR(TargetVulkanApiVersion));
    CreateDevice();
    CreateAllocator();
    ComputeMemoryBudget();
    ReadFormats();
}

Instance::~Instance() {
    if (stats_instance == this) {
        stats_instance = nullptr;
    }
    vmaDestroyAllocator(allocator);
}

void Instance::PickPhysicalDevice(s32 device_index) {
    const auto devices = Check<"enumerate GPUs">(instance->enumeratePhysicalDevices());
    ASSERT_MSG(!devices.empty(), "no GPU with Vulkan found");
    // The launcher found the host's NVIDIA driver (bbport_vulkan.py), yet it gave no device: it
    // failed to load, and the game would quietly run on another GPU (#205).
    const char* drivers = std::getenv("VK_DRIVER_FILES");
    if (drivers && std::strstr(drivers, "/vulkan/nvidia/") &&
        std::ranges::none_of(devices, [](vk::PhysicalDevice device) {
            return device.getProperties().vendorID == 0x10de;
        })) {
        std::printf("GPU: WARNING: the host's NVIDIA Vulkan driver did not load, another GPU is "
                    "used (see the loader's errors above; --vulkan-info lists the NVIDIA "
                    "libraries it could not find)\n");
    }
    if (device_index >= 0) {
        ASSERT_MSG(u32(device_index) < devices.size(), "GPU {} asked for; there are {}",
                   device_index, devices.size());
        physical_device = devices[device_index];
        return;
    }
    physical_device = *std::ranges::max_element(
        devices, [](vk::PhysicalDevice a, vk::PhysicalDevice b) { return Rank(a) < Rank(b); });
}

void Instance::ReadProperties() {
    const auto chain = physical_device.getProperties2<
        vk::PhysicalDeviceProperties2, vk::PhysicalDeviceVulkan11Properties,
        vk::PhysicalDeviceVulkan12Properties, vk::PhysicalDeviceVulkan13Properties,
        vk::PhysicalDevicePushDescriptorPropertiesKHR, vk::PhysicalDeviceDriverProperties>();
    properties = chain.get().properties;
    vk11_props = chain.get<vk::PhysicalDeviceVulkan11Properties>();
    vk12_props = chain.get<vk::PhysicalDeviceVulkan12Properties>();
    vk13_props = chain.get<vk::PhysicalDeviceVulkan13Properties>();
    max_push_descriptors =
        chain.get<vk::PhysicalDevicePushDescriptorPropertiesKHR>().maxPushDescriptors;
    const auto& driver = chain.get<vk::PhysicalDeviceDriverProperties>();
    driver_id = driver.driverID;
    if (driver_id == vk::DriverId::eMesaKosmickrisp) {
        const char* env = std::getenv("BB_TEXTURE_COMPRESSION");
        compressible_images = !(env && env[0] == '0');
        std::printf("Vulkan: compressible images %s (BB_TEXTURE_COMPRESSION)\n",
                    compressible_images ? "on" : "off");
    }
    memory_properties = physical_device.getMemoryProperties();

    const auto [result, extensions] = physical_device.enumerateDeviceExtensionProperties();
    Check<"list the GPU's extensions">(result);
    available_extensions.reserve(extensions.size());
    for (const auto& extension : extensions) {
        available_extensions.emplace_back(extension.extensionName.data());
    }

    // In every log: reports of hangs and lost devices need it.
    std::printf("GPU: %s%s, %llu MiB VRAM; %s %s (%s), Vulkan %s\n",
                properties.deviceName.data(), IsIntegrated() ? " (integrated)" : "",
                (unsigned long long)(LargestDeviceLocalHeap(memory_properties) >> 20),
                driver.driverName.data(),
                DriverVersionString(driver_id, properties.driverVersion).c_str(),
                driver.driverInfo.data(), VersionString(properties.apiVersion).c_str());
    LOG_INFO(Render_Vulkan, "Subgroup size {}; extensions: {}", vk11_props.subgroupSize,
             fmt::join(available_extensions, ", "));
}

void Instance::CreateDevice() {
    const auto query = physical_device.getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
        vk::PhysicalDeviceVulkan12Features, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceRobustness2FeaturesEXT,
        vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT,
        vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT,
        vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT,
        vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR,
        vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT, vk::PhysicalDeviceShaderClockFeaturesKHR,
        vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR,
        vk::PhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE,
        vk::PhysicalDeviceCooperativeMatrixFeaturesKHR, vk::PhysicalDeviceShaderFloat8FeaturesEXT>();
    features = query.get().features;
    // bbport: BB_GEOMETRY_SHADERS=0 hides the geometry stage (its draws are skipped, as before
    // KosmicKrisp had it), for A/B runs: KosmicKrisp emulates it with compute passes that end the
    // render pass.
    if (const char* gs = std::getenv("BB_GEOMETRY_SHADERS"); gs && gs[0] == '0') {
        features.geometryShader = false;
        std::printf("Vulkan: geometry shaders hidden (BB_GEOMETRY_SHADERS=0)\n");
    }
    const auto& vk11 = query.get<vk::PhysicalDeviceVulkan11Features>();
    vk12_features = query.get<vk::PhysicalDeviceVulkan12Features>();
    vk13_features = query.get<vk::PhysicalDeviceVulkan13Features>();

    std::vector<const char*> enabled;
    const auto has = [&](std::string_view name) {
        return std::ranges::contains(available_extensions, name);
    };
    const auto enable = [&](const char* name) {
        if (!has(name)) {
            LOG_INFO(Render_Vulkan, "Extension {} unavailable", name);
            return false;
        }
        enabled.push_back(name);
        return true;
    };

    for (const char* required :
         {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
          VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME, VK_EXT_ROBUSTNESS_2_EXTENSION_NAME}) {
        ASSERT_MSG(enable(required), "the GPU lacks the Vulkan extension {}", required);
    }
    const auto& robustness = query.get<vk::PhysicalDeviceRobustness2FeaturesEXT>();
    ASSERT_MSG(robustness.robustBufferAccess2 && robustness.robustImageAccess2 &&
                   robustness.nullDescriptor,
               "the GPU lacks robustBufferAccess2, robustImageAccess2 or nullDescriptor");

    ext.maintenance_5 = enable(VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    ext.maintenance_8 = enable(VK_KHR_MAINTENANCE_8_EXTENSION_NAME);
    ext.buffer_marker = enable(VK_AMD_BUFFER_MARKER_EXTENSION_NAME); // GPU breadcrumbs
    // The feedback loop layout only with its dynamic state.
    ext.feedback_loop = has(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
                        has(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
    if (ext.feedback_loop) {
        enable(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME);
        enable(VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME);
    }
    ext.depth_range_unrestricted = enable(VK_EXT_DEPTH_RANGE_UNRESTRICTED_EXTENSION_NAME);
    ext.dynamic_state_3 = enable(VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
    dynamic_color_write_mask = query.get<vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT>()
                                   .extendedDynamicState3ColorWriteMask;
    ext.custom_border_color = enable(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    ext.depth_clip_control = enable(VK_EXT_DEPTH_CLIP_CONTROL_EXTENSION_NAME);
    ext.depth_clip_enable = enable(VK_EXT_DEPTH_CLIP_ENABLE_EXTENSION_NAME);
    ext.vertex_input_dynamic_state = enable(VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    // bbport (macOS, from shadPS4 #5161): games leave restart enabled on list topologies without
    // needing it, and KosmicKrisp then "unrolls" every such indexed draw with an extra compute
    // dispatch and encoder switch. BB_LIST_RESTART=1 keeps it on (A/B testing).
    const bool kk_skip_list_restart =
        driver_id == vk::DriverId::eMesaKosmickrisp &&
        !(std::getenv("BB_LIST_RESTART") && std::getenv("BB_LIST_RESTART")[0] == '1');
    ext.list_restart =
        !kk_skip_list_restart && enable(VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME);
    if (kk_skip_list_restart) {
        LOG_WARNING(Render_Vulkan, "List primitive restart disabled on KosmicKrisp (speed)");
    }
    const auto& restart = query.get<vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT>();
    list_restart = restart.primitiveTopologyListRestart;
    patch_list_restart = restart.primitiveTopologyPatchListRestart;
    // Barycentrics from AMD's explicit vertex parameters, else the KHR extension.
    ext.explicit_vertex_parameter = enable(VK_AMD_SHADER_EXPLICIT_VERTEX_PARAMETER_EXTENSION_NAME);
    ext.barycentric = !ext.explicit_vertex_parameter &&
                      enable(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME);
    ext.provoking_vertex = enable(VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME);
    ext.stencil_export = enable(VK_EXT_SHADER_STENCIL_EXPORT_EXTENSION_NAME);
    ext.image_load_store_lod = enable(VK_AMD_SHADER_IMAGE_LOAD_STORE_LOD_EXTENSION_NAME);
    ext.amd_gcn_shader = enable(VK_AMD_GCN_SHADER_EXTENSION_NAME);
    ext.trinary_minmax = enable(VK_AMD_SHADER_TRINARY_MINMAX_EXTENSION_NAME);
    ext.nv_mixed_samples = enable(VK_NV_FRAMEBUFFER_MIXED_SAMPLES_EXTENSION_NAME);
    ext.amd_mixed_samples = enable(VK_AMD_MIXED_ATTACHMENT_SAMPLES_EXTENSION_NAME);
    ext.atomic_float = enable(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    ext.atomic_float2 = enable(VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    const auto& atomics = query.get<vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT>();
    atomic_float_buffer_minmax = atomics.shaderBufferFloat32AtomicMinMax;
    atomic_float_image_minmax = atomics.shaderImageFloat32AtomicMinMax;
    ext.workgroup_layout = enable(VK_KHR_WORKGROUP_MEMORY_EXPLICIT_LAYOUT_EXTENSION_NAME);
    const auto& layout = query.get<vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR>();
    workgroup_layout_16bit = layout.workgroupMemoryExplicitLayout16BitAccess;
    ext.view_2d_of_3d = enable(VK_EXT_IMAGE_2D_VIEW_OF_3D_EXTENSION_NAME);
    const auto& views = query.get<vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT>();
    view_2d_of_3d = views.image2DViewOf3D && views.sampler2DViewOf3D;
    ext.image_view_min_lod = enable(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
    ext.memory_budget = enable(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    // Guest direct memory allocated here and mapped by the runtime (BbGuestMemory), or host
    // memory imported where dma-bufs do not fit (NVIDIA).
    ext.memory_fd = enable(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    ext.dma_buf = ext.memory_fd && enable(VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
    ext.memory_host = enable(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    if (const char* env = std::getenv("BB_GUEST_UNCACHED"); env && env[0] == '1') {
        ext.guest_uncached = enable(VK_AMD_DEVICE_COHERENT_MEMORY_EXTENSION_NAME);
    }
    // FSR 4: quad derivatives in compute shaders; FSR 4.1.1: mixed float dot products, and
    // FP8 cooperative matrices for its RDNA4 variant (FP16 ones run it emulated, for tests).
    ext.compute_derivatives = enable(VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME);
    const auto& derivatives = query.get<vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR>();
    derivatives_linear = derivatives.computeDerivativeGroupLinear;
    ext.mixed_float_dot = query.get<vk::PhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE>()
                              .shaderMixedFloatDotProductFloat16AccFloat32 &&
                          enable(VK_VALVE_SHADER_MIXED_FLOAT_DOT_PRODUCT_EXTENSION_NAME);
    ext.cooperative_matrix =
        query.get<vk::PhysicalDeviceCooperativeMatrixFeaturesKHR>().cooperativeMatrix &&
        enable(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    ext.float8 = ext.cooperative_matrix &&
                 query.get<vk::PhysicalDeviceShaderFloat8FeaturesEXT>()
                     .shaderFloat8CooperativeMatrix &&
                 enable(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
    ext.shader_clock = enable(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
#ifdef __APPLE__
    // bbport: the MetalFX upscaler (upscale/vk_metalfx): KosmicKrisp exports memory as MTLHeaps
    // and, with this port's KosmicKrisp patch 0002, timeline semaphores as MTLSharedEvents (an
    // unpatched driver lacks it and MetalFX waits on the CPU). The name macros live in
    // vulkan_metal.h, outside this file's platform defines.
    ext.external_memory_metal = enable("VK_EXT_external_memory_metal");
    ext.metal_objects = enable("VK_EXT_metal_objects");
#endif
    subgroup_clock = query.get<vk::PhysicalDeviceShaderClockFeaturesKHR>().shaderSubgroupClock;

    // Queues: the first graphics family; a compute family without graphics for readbacks.
    const auto families = physical_device.getQueueFamilyProperties();
    const auto graphics = std::ranges::find_if(families, [](const auto& family) {
        return bool(family.queueFlags & vk::QueueFlagBits::eGraphics);
    });
    ASSERT_MSG(graphics != families.end(), "the GPU has no graphics queue");
    graphics_family = u32(graphics - families.begin());
    static constexpr float priority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queues{{
        .queueFamilyIndex = graphics_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    }};
    if (const char* env = std::getenv("BB_READBACK_QUEUE"); !env || env[0] != '0') {
        const auto compute = std::ranges::find_if(families, [](const auto& family) {
            return !(family.queueFlags & vk::QueueFlagBits::eGraphics) &&
                   (family.queueFlags & vk::QueueFlagBits::eCompute) && family.queueCount > 0;
        });
        if (compute != families.end()) {
            readback_family = u32(compute - families.begin());
            queues.push_back({
                .queueFamilyIndex = readback_family,
                .queueCount = 1,
                .pQueuePriorities = &priority,
            });
        }
    }

    // The features the port uses, of those the GPU has.
    vk::PhysicalDeviceFeatures2 core{
        .features{
            .robustBufferAccess = features.robustBufferAccess,
            .imageCubeArray = features.imageCubeArray,
            .independentBlend = features.independentBlend,
            .geometryShader = features.geometryShader,
            .tessellationShader = features.tessellationShader,
            .sampleRateShading = features.sampleRateShading,
            .dualSrcBlend = features.dualSrcBlend,
            .logicOp = features.logicOp,
            .multiDrawIndirect = features.multiDrawIndirect,
            .depthClamp = features.depthClamp,
            .depthBiasClamp = features.depthBiasClamp,
            .fillModeNonSolid = features.fillModeNonSolid,
            .depthBounds = features.depthBounds,
            .wideLines = features.wideLines,
            .multiViewport = features.multiViewport,
            .samplerAnisotropy = features.samplerAnisotropy,
            .occlusionQueryPrecise = features.occlusionQueryPrecise, // the game's queries
            .vertexPipelineStoresAndAtomics = features.vertexPipelineStoresAndAtomics,
            .fragmentStoresAndAtomics = features.fragmentStoresAndAtomics,
            .shaderImageGatherExtended = features.shaderImageGatherExtended,
            .shaderStorageImageExtendedFormats = features.shaderStorageImageExtendedFormats,
            .shaderStorageImageMultisample = features.shaderStorageImageMultisample,
            .shaderStorageImageWriteWithoutFormat =
                features.shaderStorageImageWriteWithoutFormat, // FSR 3
            .shaderClipDistance = features.shaderClipDistance,
            .shaderFloat64 = features.shaderFloat64,
            .shaderInt64 = features.shaderInt64,
            .shaderInt16 = features.shaderInt16,
            .sparseBinding = features.sparseBinding,
            .sparseResidencyBuffer = features.sparseResidencyBuffer,
        },
    };
    vk::PhysicalDeviceVulkan11Features enable11{
        .storageBuffer16BitAccess = vk11.storageBuffer16BitAccess,
        .shaderDrawParameters = vk11.shaderDrawParameters,
    };
    vk::PhysicalDeviceVulkan12Features enable12{
        .samplerMirrorClampToEdge = vk12_features.samplerMirrorClampToEdge,
        .drawIndirectCount = vk12_features.drawIndirectCount,
        .storageBuffer8BitAccess = vk12_features.storageBuffer8BitAccess,
        .shaderBufferInt64Atomics = vk12_features.shaderBufferInt64Atomics,
        .shaderSharedInt64Atomics = vk12_features.shaderSharedInt64Atomics,
        .shaderFloat16 = vk12_features.shaderFloat16,
        .shaderInt8 = vk12_features.shaderInt8,
        .scalarBlockLayout = vk12_features.scalarBlockLayout,
        .uniformBufferStandardLayout = vk12_features.uniformBufferStandardLayout,
        .separateDepthStencilLayouts = vk12_features.separateDepthStencilLayouts,
        .hostQueryReset = vk12_features.hostQueryReset,
        .timelineSemaphore = vk12_features.timelineSemaphore,
        .bufferDeviceAddress = vk12_features.bufferDeviceAddress,
        .vulkanMemoryModel = vk12_features.vulkanMemoryModel,
        .vulkanMemoryModelDeviceScope = vk12_features.vulkanMemoryModelDeviceScope,
        .shaderOutputLayer = vk12_features.shaderOutputLayer,
    };
    vk::PhysicalDeviceVulkan13Features enable13{
        .robustImageAccess = vk13_features.robustImageAccess,
        .shaderDemoteToHelperInvocation = vk13_features.shaderDemoteToHelperInvocation,
        .subgroupSizeControl = vk13_features.subgroupSizeControl,
        .computeFullSubgroups = vk13_features.computeFullSubgroups,
        .synchronization2 = vk13_features.synchronization2,
        .dynamicRendering = vk13_features.dynamicRendering,
        .shaderIntegerDotProduct = vk13_features.shaderIntegerDotProduct,
        .maintenance4 = vk13_features.maintenance4,
    };
    vk::PhysicalDeviceRobustness2FeaturesEXT robust{
        .robustBufferAccess2 = true, .robustImageAccess2 = true, .nullDescriptor = true};
    vk::PhysicalDeviceVertexAttributeDivisorFeatures divisor{
        .vertexAttributeInstanceRateDivisor = true};
    vk::PhysicalDeviceCustomBorderColorFeaturesEXT border{
        .customBorderColors = true, .customBorderColorWithoutFormat = true};
    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT dynamic3{
        .extendedDynamicState3ColorWriteMask = dynamic_color_write_mask};
    vk::PhysicalDeviceDepthClipControlFeaturesEXT clip_control{.depthClipControl = true};
    vk::PhysicalDeviceDepthClipEnableFeaturesEXT clip_enable{.depthClipEnable = true};
    vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT vertex_input{
        .vertexInputDynamicState = true};
    vk::PhysicalDevicePrimitiveTopologyListRestartFeaturesEXT list_restart_enable{
        .primitiveTopologyListRestart = list_restart,
        .primitiveTopologyPatchListRestart = patch_list_restart};
    vk::PhysicalDeviceFragmentShaderBarycentricFeaturesKHR barycentric{
        .fragmentShaderBarycentric = true};
    vk::PhysicalDeviceProvokingVertexFeaturesEXT provoking{.provokingVertexLast = true};
    vk::PhysicalDeviceMaintenance5FeaturesKHR maintenance5{.maintenance5 = true};
    vk::PhysicalDeviceMaintenance8FeaturesKHR maintenance8{.maintenance8 = true};
    vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT feedback_layout{
        .attachmentFeedbackLoopLayout = true};
    vk::PhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT feedback_state{
        .attachmentFeedbackLoopDynamicState = true};
    vk::PhysicalDeviceShaderAtomicFloat2FeaturesEXT atomic2{
        .shaderBufferFloat32AtomicMinMax = atomic_float_buffer_minmax,
        .shaderImageFloat32AtomicMinMax = atomic_float_image_minmax};
    vk::PhysicalDeviceWorkgroupMemoryExplicitLayoutFeaturesKHR workgroup{
        .workgroupMemoryExplicitLayout = layout.workgroupMemoryExplicitLayout,
        .workgroupMemoryExplicitLayoutScalarBlockLayout =
            layout.workgroupMemoryExplicitLayoutScalarBlockLayout,
        .workgroupMemoryExplicitLayout16BitAccess = workgroup_layout_16bit};
    vk::PhysicalDeviceImage2DViewOf3DFeaturesEXT view3d{
        .image2DViewOf3D = views.image2DViewOf3D, .sampler2DViewOf3D = views.sampler2DViewOf3D};
    vk::PhysicalDeviceImageViewMinLodFeaturesEXT min_lod{.minLod = true};
    vk::PhysicalDeviceShaderClockFeaturesKHR clock{.shaderSubgroupClock = subgroup_clock};
    vk::PhysicalDeviceComputeShaderDerivativesFeaturesKHR derivatives_enable{
        .computeDerivativeGroupQuads = derivatives.computeDerivativeGroupQuads,
        .computeDerivativeGroupLinear = derivatives_linear};
    vk::PhysicalDeviceShaderMixedFloatDotProductFeaturesVALVE mixed_dot{
        .shaderMixedFloatDotProductFloat16AccFloat32 = true};
    vk::PhysicalDeviceCooperativeMatrixFeaturesKHR matrix{.cooperativeMatrix = true};
    vk::PhysicalDeviceShaderFloat8FeaturesEXT float8{.shaderFloat8 = true,
                                                      .shaderFloat8CooperativeMatrix = true};
    vk::PhysicalDeviceCoherentMemoryFeaturesAMD coherent{.deviceCoherentMemory = true};

    vk::DeviceCreateInfo create_info{
        .queueCreateInfoCount = u32(queues.size()),
        .pQueueCreateInfos = queues.data(),
    };
    void** tail = const_cast<void**>(&create_info.pNext);
    Link(tail, core);
    Link(tail, enable11);
    Link(tail, enable12);
    Link(tail, enable13);
    Link(tail, robust);
    Link(tail, divisor);
    const auto link_if = [&](bool on, auto& features) {
        if (on) {
            Link(tail, features);
        }
    };
    link_if(ext.custom_border_color, border);
    link_if(ext.dynamic_state_3, dynamic3);
    link_if(ext.depth_clip_control, clip_control);
    link_if(ext.depth_clip_enable, clip_enable);
    link_if(ext.vertex_input_dynamic_state, vertex_input);
    link_if(ext.list_restart, list_restart_enable);
    link_if(ext.barycentric, barycentric);
    link_if(ext.provoking_vertex, provoking);
    link_if(ext.maintenance_5, maintenance5);
    link_if(ext.maintenance_8, maintenance8);
    link_if(ext.feedback_loop, feedback_layout);
    link_if(ext.feedback_loop, feedback_state);
    link_if(ext.atomic_float2, atomic2);
    link_if(ext.workgroup_layout, workgroup);
    link_if(ext.view_2d_of_3d, view3d);
    link_if(ext.image_view_min_lod, min_lod);
    link_if(ext.shader_clock, clock);
    link_if(ext.compute_derivatives, derivatives_enable);
    link_if(ext.mixed_float_dot, mixed_dot);
    link_if(ext.cooperative_matrix, matrix);
    link_if(ext.float8, float8);
    link_if(ext.guest_uncached, coherent);

    // DLSS (optional bridge library) needs its own device extensions on NVIDIA GPUs.
    const size_t own_extensions = enabled.size();
    Dlss* dlss = Dlss::Get();
    if (dlss) {
        std::vector<const char*> dlss_extensions;
        dlss->AppendDeviceExtensions(*instance, physical_device, dlss_extensions);
        for (const char* name : dlss_extensions) {
            if (std::ranges::none_of(enabled,
                                     [&](const char* e) { return std::string_view{e} == name; })) {
                enabled.push_back(name);
            }
        }
    }
    const auto create = [&] {
        create_info.enabledExtensionCount = u32(enabled.size());
        create_info.ppEnabledExtensionNames = enabled.data();
        return physical_device.createDeviceUnique(create_info);
    };
    auto created = create();
    if (created.result != vk::Result::eSuccess && enabled.size() > own_extensions) {
        // The driver advertises NGX's extensions, but they need what the system may lack
        // (libcuda.so.1 where the AppImage's loader looks, the NVIDIA UVM module):
        // ErrorInitializationFailed. DLSS is optional: again without them (#135, #147).
        LOG_WARNING(Render_Vulkan, "Device creation with the DLSS extensions failed ({}): "
                    "again without them", vk::to_string(created.result));
        if (dlss) {
            dlss->Disable("the Vulkan device does not start with NGX's extensions: NVIDIA's "
                          "libcuda.so.1 or the nvidia-uvm module is missing");
        }
        enabled.resize(own_extensions);
        created = create();
    }
    device = Check<"create the Vulkan device">(std::move(created));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*device);
    // An extension the driver lists and enables but whose commands it does not resolve counts
    // as missing: calling them jumped to address 0 (NVIDIA 595, #161; Intel ANV, #177).
    const auto& dispatch = VULKAN_HPP_DEFAULT_DISPATCHER;
    const auto require = [](bool& flag, const void* command, const char* name) {
        if (flag && !command) {
            LOG_WARNING(Render_Vulkan, "{} is enabled but its commands are not resolved: off",
                        name);
            flag = false;
        }
    };
    require(ext.buffer_marker, reinterpret_cast<const void*>(dispatch.vkCmdWriteBufferMarkerAMD),
            VK_AMD_BUFFER_MARKER_EXTENSION_NAME);
    require(ext.vertex_input_dynamic_state,
            reinterpret_cast<const void*>(dispatch.vkCmdSetVertexInputEXT),
            VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME);
    require(ext.dynamic_state_3,
            reinterpret_cast<const void*>(dispatch.vkCmdSetColorWriteMaskEXT),
            VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME);
    require(ext.memory_fd, reinterpret_cast<const void*>(dispatch.vkGetMemoryFdKHR),
            VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
    ext.dma_buf = ext.dma_buf && ext.memory_fd;
    require(ext.memory_host,
            reinterpret_cast<const void*>(dispatch.vkGetMemoryHostPointerPropertiesEXT),
            VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    ASSERT_MSG(dispatch.vkCmdPushDescriptorSetKHR,
               "the driver does not resolve vkCmdPushDescriptorSetKHR (VK_KHR_push_descriptor)");
    MetalLayer::InstallIfAsked(); // bbport: BB_METAL_TRACE=1 (docs/MACOS_METAL_RENDERER.md)
    if (dlss) {
        dlss->Initialize(*instance, physical_device, *device);
    }
    graphics_queue = device->getQueue(graphics_family, 0);
    if (readback_family != NoFamily) {
        readback_queue = device->getQueue(readback_family, 0);
        LOG_INFO(Render_Vulkan, "Readback queue: family {}", readback_family);
    }
}

void Instance::CreateAllocator() {
    const VmaVulkanFunctions functions{
        .vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
        .vkGetDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr,
    };
    // Device memory taken and given back per frame (BB_FRAME_LOG alloc_mb).
    static const VmaDeviceMemoryCallbacks callbacks{
        .pfnAllocate = [](VmaAllocator, uint32_t, VkDeviceMemory, VkDeviceSize size, void*) {
            BbStats::device_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
        },
        .pfnFree = [](VmaAllocator, uint32_t, VkDeviceMemory, VkDeviceSize size, void*) {
            BbStats::device_free_bytes.fetch_add(size, std::memory_order_relaxed);
        },
    };
    // 64 MiB blocks (VMA's default is 256): a block goes back to the driver only once empty,
    // and images freed here and there (the texture collector) left big blocks mostly empty
    // and held. BB_VMA_BLOCK_MB=N.
    static const VkDeviceSize block_size = [] {
        const char* env = std::getenv("BB_VMA_BLOCK_MB");
        return VkDeviceSize(env ? std::max(1ul, std::strtoul(env, nullptr, 10)) : 64ul) << 20;
    }();
    const VmaAllocatorCreateInfo info{
        .flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice = physical_device,
        .device = *device,
        .preferredLargeHeapBlockSize = block_size,
        .pDeviceMemoryCallbacks = &callbacks,
        .pVulkanFunctions = &functions,
        .instance = *instance,
        .vulkanApiVersion = TargetVulkanApiVersion,
    };
    Check<"create the memory allocator">(vk::Result(vmaCreateAllocator(&info, &allocator)));
    stats_instance = this;
}

void VmaDeviceUsage(u64& block_bytes, u64& allocation_bytes) {
    block_bytes = allocation_bytes = 0;
    if (!stats_instance) {
        return;
    }
    const auto& memory = stats_instance->GetMemoryProperties();
    std::array<VmaBudget, VK_MAX_MEMORY_HEAPS> budgets{};
    vmaGetHeapBudgets(stats_instance->GetAllocator(), budgets.data());
    for (u32 heap = 0; heap < memory.memoryHeapCount; ++heap) {
        if (memory.memoryHeaps[heap].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
            block_bytes += budgets[heap].statistics.blockBytes;
            allocation_bytes += budgets[heap].statistics.allocationBytes;
        }
    }
}

#ifdef __APPLE__
static u64 UnifiedMemoryBudget(u64 limit, u64 gpu_usage, u64 game_floor);
#endif

void Instance::ComputeMemoryBudget() {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props{.pNext = ext.memory_budget ? &budget : nullptr};
    physical_device.getMemoryProperties2(&props);
    const auto& heaps = props.memoryProperties;
    u64 initial_usage = 0;
    u64 local_memory = 0;
    for (u32 i = 0; i < heaps.memoryHeapCount; ++i) {
        const bool local = bool(heaps.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal);
        if (!local && !IsIntegrated()) {
            continue; // system memory of discrete GPUs does not count
        }
        counted_heaps.push_back(i);
        local_memory += local ? heaps.memoryHeaps[i].size : 0;
        if (ext.memory_budget) {
            initial_usage += budget.heapUsage[i];
            total_memory_budget += budget.heapBudget[i];
        } else {
            total_memory_budget += heaps.memoryHeaps[i].size;
        }
    }
    if (VramLimit() != 0) {
        total_memory_budget = std::min(total_memory_budget, VramLimit());
    }
    if (!IsIntegrated()) {
        // Some for the system.
        total_memory_budget -= std::min<u64>(total_memory_budget / 8, 1_GB);
        return;
    }
#ifdef __APPLE__
    // bbport (macOS): the reserve below was no reserve on a Mac: KosmicKrisp's heapBudget is the
    // usage plus 90 % of the few free pages, and the max() with the heap size (Metal's limit)
    // always won up to ~24 GB of RAM. The startup budget is the one the collector follows
    // (UnifiedMemoryBudget), with the game process taken at least as large as it gets (its
    // direct memory and Rosetta's share: ~5 GB; it is still loading now). BB_MEM_BUDGET=metal:
    // as before.
    u64 heap_bytes = 0;
    for (const u32 heap : counted_heaps) {
        heap_bytes += heaps.memoryHeaps[heap].size;
    }
    std::printf("GPU (macOS): %llu MiB of RAM, Metal's limit %llu MiB\n",
                (unsigned long long)(BbMacMemory::PhysicalBytes() >> 20),
                (unsigned long long)(heap_bytes >> 20));
    if (!BbMacMemory::MachineBudget()) {
        std::printf("GPU (macOS): memory budget within Metal's limit only (BB_MEM_BUDGET=metal)\n");
    } else if (heap_bytes != 0) {
        std::printf("GPU (macOS): memory budget of the whole Mac: macOS keeps %llu MiB, the game "
                    "process its footprint (BB_MEM_BUDGET=metal: Metal's limit only)\n",
                    (unsigned long long)(BbMacMemory::OSReserveBytes() >> 20));
        // The game process (still loading) as at least 5 GB; in one process it is in ours.
        total_memory_budget = UnifiedMemoryBudget(heap_bytes, initial_usage,
                                                  BbRemote::BackActive() ? 5_GB : 0);
        if (VramLimit() != 0) {
            total_memory_budget = std::min(total_memory_budget, VramLimit());
        }
        return;
    }
#endif
    // At least 8 GiB left for the system on integrated GPUs.
    const s64 available = s64(total_memory_budget - initial_usage);
    total_memory_budget = u64(std::max<s64>(available - s64(8_GB), s64(local_memory)));
}

u64 Instance::GetDeviceMemoryUsage() const {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props{.pNext = &budget};
    physical_device.getMemoryProperties2(&props);
    u64 usage = 0;
    for (const u32 heap : counted_heaps) {
        usage += budget.heapUsage[heap];
    }
    return usage;
}

#ifdef __APPLE__
/// bbport (macOS): the Mac's memory is shared by the CPU and the GPU, and Metal's limit (the
/// heap size, recommendedMaxWorkingSetSize, ~75% of the Mac's memory) is for everything this
/// process has, not for the GPU alone. The game's direct memory (5-9 GB), its heap, Rosetta and
/// the runtime came on top of a GPU budget of the whole limit (issue #3). The GPU's share is the
/// limit less what the process uses outside graphics memory (macOS's graphics ledger), at least
/// 2 GB. Not less the whole footprint beyond the renderer's own allocations: KosmicKrisp holds
/// graphics memory of its own the renderer cannot see (15 GB and growing in a 5-minute run,
/// issue #6), and counting it shrank the texture cache until it evicted images in use without
/// freeing any of that.
/// bbport (macOS): nor does the limit know of the game process (bb-probe) or of macOS: 16 GB Macs
/// swapped with the limit as the budget. So (unless BB_MEM_BUDGET=metal) the limit is taken as at
/// most the Mac's RAM less what macOS keeps (BbMacMemory::OSReserveBytes) and the game process's
/// footprint, sampled with the rest (that process grows as the game loads); `game_floor`: the
/// game process taken as at least that (at startup).
static u64 UnifiedMemoryBudget(u64 limit, u64 gpu_usage, u64 game_floor) {
    // Sampled at most every 250 ms (this runs at every texture collector tick).
    static std::atomic<u64> other_cached{0}, graphics_cached{0}, game_cached{0};
    static std::atomic<s64> sampled_ns{-1};
    const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch())
                        .count();
    const s64 sampled = sampled_ns.load(std::memory_order_relaxed);
    if (sampled < 0 || now - sampled >= 250'000'000) {
        u64 footprint = 0, graphics = 0;
        if (!BbMacMemory::OwnFootprint(footprint, graphics)) {
            return limit;
        }
        other_cached.store(footprint > graphics ? footprint - graphics : 0,
                           std::memory_order_relaxed);
        graphics_cached.store(graphics, std::memory_order_relaxed);
        if (BbMacMemory::MachineBudget()) {
            game_cached.store(BbMacMemory::GameProcessFootprint(), std::memory_order_relaxed);
        }
        sampled_ns.store(now, std::memory_order_relaxed);
    }
    const u64 other = other_cached.load(std::memory_order_relaxed);
    const u64 graphics = graphics_cached.load(std::memory_order_relaxed);
    const u64 game = std::max(game_cached.load(std::memory_order_relaxed), game_floor);
    // bbport (macOS): at most the Mac's RAM less macOS's reserve and the game process.
    u64 cap = limit;
    if (const u64 ram = BbMacMemory::PhysicalBytes(); BbMacMemory::MachineBudget() && ram != 0) {
        const u64 taken = BbMacMemory::OSReserveBytes() + game;
        cap = std::min(limit, ram > taken ? ram - taken : 0);
    }
    constexpr u64 Floor = u64{2} << 30;
    const u64 budget = cap > other + Floor ? cap - other : Floor;
    // Printed when the budget or the graphics memory moves by 1 GiB or more: the second line
    // tracks the driver's own graphics memory (graphics less the renderer's allocations).
    static std::atomic<u64> printed_budget{0}, printed_graphics{0};
    const auto moved = [](u64 a, u64 b) { return (a > b ? a - b : b - a) >= (u64{1} << 30); };
    u64 last_budget = printed_budget.load(std::memory_order_relaxed);
    u64 last_graphics = printed_graphics.load(std::memory_order_relaxed);
    const bool budget_moved =
        moved(last_budget, budget) &&
        printed_budget.compare_exchange_strong(last_budget, budget, std::memory_order_relaxed);
    const bool graphics_moved =
        moved(last_graphics, graphics) &&
        printed_graphics.compare_exchange_strong(last_graphics, graphics, std::memory_order_relaxed);
    if (budget_moved || graphics_moved) {
        std::printf("GPU: memory budget %llu MiB (Metal's limit %llu MiB, less %llu MiB outside "
                    "graphics memory); graphics memory %llu MiB, of it %llu MiB the renderer's "
                    "and %llu MiB the driver's own\n",
                    (unsigned long long)(budget >> 20), (unsigned long long)(limit >> 20),
                    (unsigned long long)(other >> 20), (unsigned long long)(graphics >> 20),
                    (unsigned long long)(gpu_usage >> 20),
                    (unsigned long long)((graphics > gpu_usage ? graphics - gpu_usage : 0) >> 20));
        // bbport (macOS): what the whole Mac left of the limit.
        if (BbMacMemory::MachineBudget()) {
            std::printf("GPU: memory budget of the whole Mac: %llu MiB of RAM less %llu MiB for "
                        "macOS and %llu MiB for the game process (footprint%s): at most %llu MiB "
                        "of Metal's %llu MiB\n",
                        (unsigned long long)(BbMacMemory::PhysicalBytes() >> 20),
                        (unsigned long long)(BbMacMemory::OSReserveBytes() >> 20),
                        (unsigned long long)(game >> 20),
                        game == game_floor && game_floor != 0 ? ", estimated at startup" : "",
                        (unsigned long long)(cap >> 20), (unsigned long long)(limit >> 20));
        }
    }
    return budget;
}
#endif

u64 Instance::GetDeviceMemoryBudgetNow() const {
    vk::PhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    vk::PhysicalDeviceMemoryProperties2 props{.pNext = &budget};
    physical_device.getMemoryProperties2(&props);
    u64 total = 0;
#ifdef __APPLE__
    // bbport: KosmicKrisp's budget is the usage plus 90% of the free and inactive pages, and macOS
    // keeps few of those (file cache, compressed memory): it stayed ~1 GB above the usage whatever
    // the Mac had, so the texture cache sat at its pressure mark and evicted hundreds of images
    // every few seconds, only to upload them again. The heap size is Metal's
    // recommendedMaxWorkingSetSize (a share of the Mac's memory): the real limit, less what the
    // rest of the process uses (UnifiedMemoryBudget).
    u64 gpu_usage = 0;
    for (const u32 heap : counted_heaps) {
        total += props.memoryProperties.memoryHeaps[heap].size;
        gpu_usage += budget.heapUsage[heap];
    }
    total = UnifiedMemoryBudget(total, gpu_usage, 0);
#else
    for (const u32 heap : counted_heaps) {
        total += budget.heapBudget[heap];
    }
#endif
    return VramLimit() != 0 ? std::min(total, VramLimit()) : total;
}

void Instance::ReadFormats() {
    const auto read = [&](vk::Format format) {
        if (!format_features.contains(format)) {
            vk::FormatProperties3 props3{};
            vk::FormatProperties2 props2{.pNext = &props3};
            physical_device.getFormatProperties2(format, &props2);
            format_features.emplace(format,
                                    props3.optimalTilingFeatures | props3.bufferFeatures);
        }
    };
    for (const auto& format : GcnToVk::SurfaceFormats()) {
        read(format.vk_format);
    }
    for (const auto& format : GcnToVk::DepthFormats()) {
        read(format.vk_format);
    }
    // Stand-ins and the presenter's formats.
    for (const auto format : {vk::Format::eA2R10G10B10UnormPack32, vk::Format::eB8G8R8A8Unorm,
                              vk::Format::eB8G8R8A8Srgb, vk::Format::eD24UnormS8Uint,
                              vk::Format::eD32SfloatS8Uint, vk::Format::eR8Unorm,
                              vk::Format::eR8G8Unorm}) {
        read(format);
    }

    // Views of other formats on block compressed images.
    const auto block_view = physical_device.getImageFormatProperties2(vk::PhysicalDeviceImageFormatInfo2{
        .format = vk::Format::eBc1RgbaUnormBlock,
        .type = vk::ImageType::e2D,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eSampled,
        .flags = vk::ImageCreateFlagBits::eBlockTexelViewCompatible,
    });
    block_texel_view = block_view.result == vk::Result::eSuccess;

    for (const auto& format : GcnToVk::SurfaceFormats()) {
        if (!IsFormatSupported(format.vk_format, format.flags)) {
            LOG_WARNING(Render_Vulkan, "Format {} (data {}, number {}) lacks {}",
                        vk::to_string(format.vk_format), u32(format.data_format),
                        u32(format.number_format),
                        vk::to_string(format.flags & ~FormatFeatures(format.vk_format)));
        }
    }
    for (const auto& format : GcnToVk::DepthFormats()) {
        if (!IsFormatSupported(format.vk_format, format.flags)) {
            LOG_WARNING(Render_Vulkan, "Depth format {} (z {}, stencil {}) lacks {}",
                        vk::to_string(format.vk_format), u32(format.z_format),
                        u32(format.stencil_format),
                        vk::to_string(format.flags & ~FormatFeatures(format.vk_format)));
        }
    }
}

vk::FormatFeatureFlags2 Instance::FormatFeatures(vk::Format format) const {
    if (const auto it = format_features.find(format); it != format_features.end()) {
        return it->second;
    }
    // Not among the GCN formats: asked of the driver (rare; not cached, the table is shared).
    vk::FormatProperties3 props3{};
    vk::FormatProperties2 props2{.pNext = &props3};
    physical_device.getFormatProperties2(format, &props2);
    return props3.optimalTilingFeatures | props3.bufferFeatures;
}

bool Instance::IsFormatSupported(vk::Format format, vk::FormatFeatureFlags2 flags) const {
    return format == vk::Format::eUndefined || (FormatFeatures(format) & flags) == flags;
}

vk::Format Instance::GetSupportedFormat(vk::Format format, vk::FormatFeatureFlags2 flags) const {
    if (IsFormatSupported(format, flags)) [[likely]] {
        return format;
    }
    const auto first_supported = [&](std::initializer_list<vk::Format> candidates) {
        for (const auto candidate : candidates) {
            if (IsFormatSupported(candidate, flags)) {
                return candidate;
            }
        }
        return format;
    };
    switch (format) {
    case vk::Format::eD16UnormS8Uint:
        return first_supported({vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint});
    case vk::Format::eR8Srgb:
        return first_supported({vk::Format::eR8Unorm});
    case vk::Format::eR8G8Srgb:
        return first_supported({vk::Format::eR8G8Unorm});
    default:
        return format;
    }
}

} // namespace Vulkan
