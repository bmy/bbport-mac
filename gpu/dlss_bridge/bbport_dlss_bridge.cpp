// SPDX-FileCopyrightText: Copyright 2026 IFreemz (shadps4_dlss bridge), bbport contributors
// SPDX-License-Identifier: MIT
//
// libbbport_dlss.so / bbport_dlss.dll: the only part of the project that uses the NVIDIA DLSS
// (NGX) SDK; the port loads it at run time. Linux: build.sh with DLSS_SDK_ROOT; Windows: MSVC
// (the SDK's Windows libraries are MSVC only), see CMakeLists.txt.
// DLSS Super Resolution (libnvidia-ngx-dlss.so / nvngx_dlss.dll) and DLSS Frame Generation
// (libnvidia-ngx-dlssg.so / nvngx_dlssg.dll, GeForce RTX 40 and newer); each is offered only
// when NGX finds its library and the GPU and driver support it.
// Adapted from IFreemz/shadPS4-Bloodborne-DLSS-FSR (dlss_bridge).

#include <algorithm>
#include <atomic>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "bbport_dlss_bridge.h" // Vulkan first: the NGX headers expect its types

#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssg_vk.h>

namespace {

constexpr char ProjectId[] = "42359704-c9f3-4806-9fb5-d469000fdb8a";
#ifdef _WIN32
constexpr char EngineVersion[] = "bbport-windows";
#else
constexpr char EngineVersion[] = "bbport-linux";
#endif

struct State {
    BbDlssLogFn log{};
    std::vector<std::wstring> dll_directories;
    std::vector<const wchar_t*> dll_paths; // into dll_directories (NGX keeps the pointers)
    std::wstring data_directory;
    NVSDK_NGX_FeatureCommonInfo common{};
    NVSDK_NGX_FeatureDiscoveryInfo discovery{};    // Super Resolution
    NVSDK_NGX_FeatureDiscoveryInfo discovery_fg{}; // Frame Generation
    NVSDK_NGX_Parameter* capabilities{};
    NVSDK_NGX_Parameter* parameters{};
    NVSDK_NGX_Handle* feature{};
    BbDlssFeature feature_desc{};
    // Frame Generation: its own parameter map (NVIDIA's advice with two features).
    NVSDK_NGX_Parameter* fg_parameters{};
    NVSDK_NGX_Handle* fg_feature{};
    BbDlssgFeature fg_desc{};
    bool fg_device_supported{};
    int32_t fg_max_count{};
    // The extensions of both features, without duplicates.
    std::vector<VkExtensionProperties> instance_extensions, device_extensions;
    VkDevice device{};
    bool configured{};
    bool initialized{};
} state;

template <typename... Args>
void Log(int warning, const char* format, Args... args) {
    if (!state.log)
        return;
    if constexpr (sizeof...(Args) == 0) {
        state.log(warning, format); // a plain message (no format string without arguments)
    } else {
        std::array<char, 1024> text{};
        std::snprintf(text.data(), text.size(), format, args...);
        state.log(warning, text.data());
    }
}

bool Check(const char* operation, NVSDK_NGX_Result result) {
    if (NVSDK_NGX_FAILED(result)) {
        Log(1, "%s failed: 0x%08x", operation, static_cast<unsigned>(result));
        return false;
    }
    return true;
}

void NVSDK_CONV NgxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature) {
    // NGX's own lines (BB_NGX_LOG): the first few thousand, a line repeated every frame would
    // fill the log otherwise.
    static std::atomic<unsigned> lines{0};
    const unsigned n = lines.fetch_add(1, std::memory_order_relaxed);
    if (n < 4000) {
        Log(0, "NGX: %s", message ? message : "");
    } else if (n == 4000) {
        Log(0, "NGX: (more lines left out)");
    }
}

/// BB_NGX_LOG=1: NGX's log into the game's (its warnings about the resources and parameters
/// DLSS and DLSS Frame Generation get); 2: verbose. Off by default.
NVSDK_NGX_Logging_Level NgxLogLevel() {
    const char* env = std::getenv("BB_NGX_LOG");
    return !env || !*env || env[0] == '0' ? NVSDK_NGX_LOGGING_LEVEL_OFF
           : env[0] == '2'                ? NVSDK_NGX_LOGGING_LEVEL_VERBOSE
                                          : NVSDK_NGX_LOGGING_LEVEL_ON;
}

int32_t ConfigureDirectories(const wchar_t* const* dll_directories, uint32_t count,
                             const wchar_t* data_directory, BbDlssLogFn log) {
    state.log = log;
    state.dll_directories.clear();
    for (uint32_t i = 0; dll_directories && i < count; ++i) {
        if (dll_directories[i] && dll_directories[i][0])
            state.dll_directories.emplace_back(dll_directories[i]);
    }
    if (state.dll_directories.empty())
        state.dll_directories.emplace_back(L".");
    state.dll_paths.clear();
    for (const auto& directory : state.dll_directories)
        state.dll_paths.push_back(directory.c_str());
    state.data_directory = data_directory ? data_directory : L".";
    state.common.PathListInfo = {state.dll_paths.data(),
                                 static_cast<unsigned int>(state.dll_paths.size())};
    state.common.LoggingInfo = {NgxLog, NgxLogLevel(), false};
    state.discovery.SDKVersion = NVSDK_NGX_Version_API;
    state.discovery.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    state.discovery.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    state.discovery.Identifier.v.ProjectDesc = {ProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                                                EngineVersion};
    state.discovery.ApplicationDataPath = state.data_directory.c_str();
    state.discovery.FeatureInfo = &state.common;
    state.discovery_fg = state.discovery;
    state.discovery_fg.FeatureID = NVSDK_NGX_Feature_FrameGeneration;
    state.configured = true;
    return 1;
}

int32_t Configure(const wchar_t* dll_directory, const wchar_t* data_directory, BbDlssLogFn log) {
    return ConfigureDirectories(&dll_directory, dll_directory ? 1u : 0u, data_directory, log);
}

void Merge(std::vector<VkExtensionProperties>& into, const VkExtensionProperties* add,
           uint32_t count) {
    for (uint32_t i = 0; add && i < count; ++i) {
        const auto same = std::find_if(into.begin(), into.end(), [&](const auto& have) {
            return std::strcmp(have.extensionName, add[i].extensionName) == 0;
        });
        if (same == into.end()) {
            into.push_back(add[i]);
        } else {
            same->specVersion = std::max(same->specVersion, add[i].specVersion);
        }
    }
}

int32_t InstanceExtensions(uint32_t* count, const VkExtensionProperties** extensions) {
    VkExtensionProperties* required{};
    uint32_t required_count{};
    if (!state.configured ||
        !Check("Instance extension query",
               NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
                   &state.discovery, &required_count, &required)))
        return 0;
    state.instance_extensions.clear();
    Merge(state.instance_extensions, required, required_count);
    // Frame Generation's own needs; without its library only Super Resolution is offered.
    required = nullptr;
    required_count = 0;
    if (NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
            &state.discovery_fg, &required_count, &required)))
        Merge(state.instance_extensions, required, required_count);
    *count = static_cast<uint32_t>(state.instance_extensions.size());
    *extensions = state.instance_extensions.data();
    return 1;
}

int32_t DeviceExtensions(VkInstance instance, VkPhysicalDevice physical, uint32_t* count,
                         const VkExtensionProperties** extensions) {
    if (!state.configured)
        return 0;
    NVSDK_NGX_FeatureRequirement support{};
    if (!Check("Feature requirement query",
               NVSDK_NGX_VULKAN_GetFeatureRequirements(instance, physical, &state.discovery,
                                                       &support)))
        return 0;
    if (support.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
        Log(1, "DLSS is not supported on this GPU or driver (code 0x%x)",
            static_cast<unsigned>(support.FeatureSupported));
        return 0;
    }
    VkExtensionProperties* required{};
    uint32_t required_count{};
    if (!Check("Device extension query",
               NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
                   instance, physical, &state.discovery, &required_count, &required)))
        return 0;
    state.device_extensions.clear();
    Merge(state.device_extensions, required, required_count);
    // Frame Generation: only when this GPU and driver support it (its extensions would make
    // the device creation fail elsewhere).
    NVSDK_NGX_FeatureRequirement fg_support{};
    required = nullptr;
    required_count = 0;
    state.fg_device_supported =
        NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureRequirements(
            instance, physical, &state.discovery_fg, &fg_support)) &&
        fg_support.FeatureSupported == NVSDK_NGX_FeatureSupportResult_Supported &&
        NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            instance, physical, &state.discovery_fg, &required_count, &required));
    if (state.fg_device_supported) {
        Merge(state.device_extensions, required, required_count);
    } else {
        Log(0, "Frame Generation is not supported on this GPU or driver (code 0x%x)",
            static_cast<unsigned>(fg_support.FeatureSupported));
    }
    *count = static_cast<uint32_t>(state.device_extensions.size());
    *extensions = state.device_extensions.data();
    return 1;
}

int32_t Initialize(VkInstance instance, VkPhysicalDevice physical, VkDevice device,
                   PFN_vkGetInstanceProcAddr get_instance_proc,
                   PFN_vkGetDeviceProcAddr get_device_proc) {
    if (!state.configured || state.initialized)
        return 0;
    if (!Check("NGX initialization",
               NVSDK_NGX_VULKAN_Init_with_ProjectID(
                   ProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, EngineVersion,
                   state.data_directory.c_str(), instance, physical, device, get_instance_proc,
                   get_device_proc, &state.common)))
        return 0;
    state.initialized = true;
    state.device = device;
    if (!Check("Capability query", NVSDK_NGX_VULKAN_GetCapabilityParameters(&state.capabilities)))
        return 0;
    int available{}, needs_driver{};
    NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_SuperSampling_Available,
                             &available);
    NVSDK_NGX_Parameter_GetI(state.capabilities,
                             NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needs_driver);
    if (needs_driver) {
        Log(1, "DLSS needs a newer NVIDIA driver");
        return 0;
    }
    if (!available) {
        Log(1, "DLSS Super Resolution is not available on this system");
        return 0;
    }
    // Frame Generation: a library found in the path list, a GPU and a driver that run it.
    int fg_available{}, fg_needs_driver{};
    unsigned fg_max{};
    NVSDK_NGX_Parameter_GetI(state.capabilities, NVSDK_NGX_Parameter_FrameGeneration_Available,
                             &fg_available);
    NVSDK_NGX_Parameter_GetI(state.capabilities,
                             NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver,
                             &fg_needs_driver);
    if (NVSDK_NGX_FAILED(NVSDK_NGX_Parameter_GetUI(
            state.capabilities, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax, &fg_max)))
        fg_max = 1;
    if (state.fg_device_supported && fg_available) {
        state.fg_max_count = static_cast<int32_t>(std::clamp(fg_max, 1u, 3u));
        Log(0, "Frame Generation available (up to %dx)", state.fg_max_count + 1);
    } else if (fg_needs_driver) {
        Log(1, "Frame Generation needs a newer NVIDIA driver");
    } else {
        Log(0, "Frame Generation unavailable (its library, a GeForce RTX 40 or newer and a "
               "current driver are needed)");
    }
    return 1;
}

void ReleaseFeature() {
    if (state.feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(state.feature);
        state.feature = nullptr;
    }
    if (state.parameters) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.parameters);
        state.parameters = nullptr;
    }
}

int32_t CreateFeature(VkCommandBuffer command, const BbDlssFeature* desc) {
    if (!state.initialized || !state.capabilities || !desc)
        return 0;
    ReleaseFeature();
    if (!Check("Parameter allocation", NVSDK_NGX_VULKAN_AllocateParameters(&state.parameters)))
        return 0;
    for (const char* key : {NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
                            NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality})
        NVSDK_NGX_Parameter_SetUI(state.parameters, key, desc->preset);
    static constexpr std::array qualities{
        NVSDK_NGX_PerfQuality_Value_DLAA, NVSDK_NGX_PerfQuality_Value_MaxQuality,
        NVSDK_NGX_PerfQuality_Value_Balanced, NVSDK_NGX_PerfQuality_Value_MaxPerf,
        NVSDK_NGX_PerfQuality_Value_UltraPerformance};
    NVSDK_NGX_DLSS_Create_Params settings{};
    settings.Feature.InWidth = desc->input_width;
    settings.Feature.InHeight = desc->input_height;
    settings.Feature.InTargetWidth = desc->output_width;
    settings.Feature.InTargetHeight = desc->output_height;
    settings.Feature.InPerfQualityValue = qualities[std::clamp(desc->quality, 0, 4)];
    settings.InFeatureCreateFlags =
        NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
        (desc->depth_inverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0) |
        (desc->hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_AutoExposure
                   : 0);
    const auto result = NGX_VULKAN_CREATE_DLSS_EXT1(state.device, command, 1, 1, &state.feature,
                                                   state.parameters, &settings);
    if (!Check("DLSS feature creation", result) || !state.feature) {
        state.feature = nullptr;
        return 0;
    }
    state.feature_desc = *desc;
    Log(0, "DLSS %ux%u -> %ux%u, quality %d, preset %u%s", desc->input_width, desc->input_height,
        desc->output_width, desc->output_height, desc->quality, desc->preset,
        desc->hdr ? ", HDR input" : "");
    return 1;
}

NVSDK_NGX_Resource_VK Wrap(const BbDlssImage& image, bool writable) {
    return NVSDK_NGX_Create_ImageView_Resource_VK(image.view, image.image, image.range,
                                                  image.format, image.width, image.height,
                                                  writable);
}

int32_t Evaluate(VkCommandBuffer command, const BbDlssEvaluate* evaluate) {
    if (!state.feature || !state.parameters || !evaluate)
        return 0;
    auto color = Wrap(evaluate->color, false);
    auto depth = Wrap(evaluate->depth, false);
    auto motion = Wrap(evaluate->motion, false);
    auto output = Wrap(evaluate->output, true);
    NVSDK_NGX_VK_DLSS_Eval_Params parameters{};
    parameters.Feature.pInColor = &color;
    parameters.Feature.pInOutput = &output;
    parameters.Feature.InSharpness = evaluate->sharpness;
    parameters.pInDepth = &depth;
    parameters.pInMotionVectors = &motion;
    parameters.InRenderSubrectDimensions = {state.feature_desc.input_width,
                                            state.feature_desc.input_height};
    parameters.InJitterOffsetX = evaluate->jitter_x;
    parameters.InJitterOffsetY = evaluate->jitter_y;
    parameters.InReset = evaluate->reset;
    parameters.InMVScaleX = parameters.InMVScaleY = 1.0f;
    parameters.InPreExposure = parameters.InExposureScale = 1.0f;
    parameters.InFrameTimeDeltaInMsec = evaluate->frame_ms;
    return Check("DLSS evaluation", NGX_VULKAN_EVALUATE_DLSS_EXT(command, state.feature,
                                                                state.parameters, &parameters))
               ? 1
               : 0;
}

// --- Frame Generation ----------------------------------------------------------------------

int32_t FrameGenMaxCount() {
    return state.initialized ? state.fg_max_count : 0;
}

void FrameGenRelease() {
    if (state.fg_feature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(state.fg_feature);
        state.fg_feature = nullptr;
    }
    if (state.fg_parameters) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.fg_parameters);
        state.fg_parameters = nullptr;
    }
}

int32_t FrameGenCreate(VkCommandBuffer command, const BbDlssgFeature* desc) {
    if (!state.initialized || state.fg_max_count < 1 || !desc)
        return 0;
    FrameGenRelease();
    if (!Check("Frame Generation parameter allocation",
               NVSDK_NGX_VULKAN_AllocateParameters(&state.fg_parameters)))
        return 0;
    // Depth and motion vectors come at the scene's render size, smaller than the frame.
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_Width, desc->width);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_Height, desc->height);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_InternalWidth,
                              desc->render_width);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_InternalHeight,
                              desc->render_height);
    NVSDK_NGX_Parameter_SetUI(state.fg_parameters, NVSDK_NGX_DLSSG_Parameter_DynamicResolution, 0);
    NVSDK_NGX_DLSSG_Create_Params settings{};
    settings.Width = desc->width;
    settings.Height = desc->height;
    settings.NativeBackbufferFormat = static_cast<unsigned>(desc->format);
    settings.RenderWidth = desc->render_width;
    settings.RenderHeight = desc->render_height;
    settings.DynamicResolutionScaling = false;
    const auto result = NGX_VK_CREATE_DLSSG(command, 1, 1, &state.fg_feature,
                                            state.fg_parameters, &settings);
    if (!Check("Frame Generation feature creation", result) || !state.fg_feature) {
        state.fg_feature = nullptr;
        return 0;
    }
    state.fg_desc = *desc;
    Log(0, "Frame Generation %ux%u (depth and motion %ux%u, format %u)", desc->width,
        desc->height, desc->render_width, desc->render_height,
        static_cast<unsigned>(desc->format));
    return 1;
}

// Row-major, row vectors (D3D style: clip = view * m), depth 0 at the near plane.
void Perspective(float (&m)[4][4], float fov, float aspect, float near_z, float far_z) {
    std::memset(m, 0, sizeof(m));
    const float y = 1.0f / std::tan(fov * 0.5f);
    m[0][0] = y / aspect;
    m[1][1] = y;
    m[2][2] = far_z / (far_z - near_z);
    m[2][3] = 1.0f;
    m[3][2] = -near_z * far_z / (far_z - near_z);
}

// The inverse of a projection of the shape above (the game's: x and y scales, z scale and
// offset, w = z).
void InverseProjection(float (&out)[4][4], const float (&m)[4][4]) {
    std::memset(out, 0, sizeof(out));
    out[0][0] = 1.0f / m[0][0];
    out[1][1] = 1.0f / m[1][1];
    out[2][3] = 1.0f / m[3][2];
    out[3][2] = 1.0f;
    out[3][3] = -m[2][2] / m[3][2];
}

void Identity(float (&m)[4][4]) {
    std::memset(m, 0, sizeof(m));
    for (int i = 0; i < 4; ++i)
        m[i][i] = 1.0f;
}

/// A 0/1 environment switch (read every call: cheap, and changeable between runs only).
bool Flag(const char* name, bool fallback) {
    const char* env = std::getenv(name);
    return env && *env ? env[0] == '1' : fallback;
}

int32_t FrameGenEvaluate(VkCommandBuffer command, const BbDlssgEvaluate* evaluate) {
    if (!state.fg_feature || !state.fg_parameters || !evaluate)
        return 0;
    auto backbuffer = Wrap(evaluate->backbuffer, false);
    auto depth = Wrap(evaluate->depth, false);
    auto motion = Wrap(evaluate->motion, false);
    auto output = Wrap(evaluate->output, true);
    NVSDK_NGX_VK_DLSSG_Eval_Params resources{};
    resources.pBackbuffer = &backbuffer;
    resources.pDepth = &depth;
    resources.pMVecs = &motion;
    resources.pOutputInterpFrame = &output;
    NVSDK_NGX_Resource_VK hudless{};
    if (evaluate->hudless.image != VK_NULL_HANDLE) {
        hudless = Wrap(evaluate->hudless, false);
        resources.pHudless = &hudless;
    }
    NVSDK_NGX_DLSSG_Opt_Eval_Params options{};
    options.multiFrameCount = std::max(1u, evaluate->multi_frame_count);
    options.multiFrameIndex = std::clamp(evaluate->multi_frame_index, 1u, options.multiFrameCount);
    const float aspect = state.fg_desc.render_height
                             ? float(state.fg_desc.render_width) / float(state.fg_desc.render_height)
                             : 16.0f / 9.0f;
    Identity(options.clipToLensClip);
    if (evaluate->has_matrices && evaluate->view_to_clip[0][0] != 0.0f &&
        evaluate->view_to_clip[1][1] != 0.0f && evaluate->view_to_clip[3][2] != 0.0f) {
        // The game's cameras: DLSS separates the camera's motion from the objects' with them
        // (the vectors hold both). A camera that stood still contradicts the vectors on every
        // turn and smears the generated frames.
        std::memcpy(options.cameraViewToClip, evaluate->view_to_clip,
                    sizeof(options.cameraViewToClip));
        std::memcpy(options.clipToPrevClip, evaluate->clip_to_prev_clip,
                    sizeof(options.clipToPrevClip));
        std::memcpy(options.prevClipToClip, evaluate->prev_clip_to_clip,
                    sizeof(options.prevClipToClip));
    } else {
        // Unknown camera: a plausible projection that stood still.
        Perspective(options.cameraViewToClip, evaluate->camera_fov, aspect,
                    evaluate->camera_near, evaluate->camera_far);
        Identity(options.clipToPrevClip);
        Identity(options.prevClipToClip);
    }
    InverseProjection(options.clipToCameraView, options.cameraViewToClip);
    options.jitterOffset[0] = evaluate->jitter_x;
    options.jitterOffset[1] = evaluate->jitter_y;
    // 1.0, as PR #260 had it when it was verified on an RTX 5090 (Linux, 2026-10-18: frame
    // generation 2x-4x right). 1 / render size (2026-10-26) did not end the black frames on
    // Windows; BB_DLSSG_MVEC_SCALE=<f> to compare.
    options.mvecScale[0] = options.mvecScale[1] = 1.0f;
    if (const char* env = std::getenv("BB_DLSSG_MVEC_SCALE"); env && *env) {
        options.mvecScale[0] = options.mvecScale[1] = float(std::atof(env)); // to compare
    }
    for (int i = 0; i < 3; ++i) {
        options.cameraPos[i] = evaluate->camera_position[i];
        options.cameraUp[i] = evaluate->camera_up[i];
        options.cameraRight[i] = evaluate->camera_right[i];
        options.cameraFwd[i] = evaluate->camera_forward[i];
    }
    const auto length = [](const float v[3]) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    };
    if (length(evaluate->camera_up) < 0.5f || length(evaluate->camera_right) < 0.5f ||
        length(evaluate->camera_forward) < 0.5f) {
        // No camera this frame: the identity axes.
        std::memset(options.cameraUp, 0, sizeof(options.cameraUp));
        std::memset(options.cameraRight, 0, sizeof(options.cameraRight));
        std::memset(options.cameraFwd, 0, sizeof(options.cameraFwd));
        options.cameraUp[1] = 1.0f;
        options.cameraRight[0] = 1.0f;
        options.cameraFwd[2] = 1.0f;
    }
    options.cameraNear = evaluate->camera_near;
    options.cameraFar = evaluate->camera_far;
    options.cameraFOV = evaluate->camera_fov;
    options.cameraAspectRatio = aspect;
    options.colorBuffersHDR = false;
    options.depthInverted = evaluate->depth_inverted != 0;
    options.cameraMotionIncluded = true;
    // No vector is marked invalid (0, the zero-initialised value, would mark every still pixel).
    options.motionVectorsInvalidValue = std::numeric_limits<float>::max();
    options.motionVectorsDilated = false;
    options.menuDetectionEnabled = false;
    options.reset = evaluate->reset != 0 && options.multiFrameIndex == 1;
    // Depth and vectors at the render size, smaller than the frame: their extents given
    // (a zero size means the whole texture, and without dynamic resolution DLSS-G may take them at
    // the backbuffer's size). BB_DLSSG_SUBRECTS=0: left to it.
    static const bool subrects = Flag("BB_DLSSG_SUBRECTS", true);
    if (subrects) {
        options.mvecsSubrectSize = {evaluate->motion.width, evaluate->motion.height};
        options.depthSubrectSize = {evaluate->depth.width, evaluate->depth.height};
        options.backbufferSubrectSize = {evaluate->backbuffer.width, evaluate->backbuffer.height};
        options.outputInterpSubrectSize = {evaluate->output.width, evaluate->output.height};
        if (evaluate->hudless.image != VK_NULL_HANDLE) {
            options.hudLessSubrectSize = {evaluate->hudless.width, evaluate->hudless.height};
        }
    }
    // Diagnostics for a GPU this was not tried on: BB_DLSSG_MATRICES=0 (a still camera: the
    // vectors alone), BB_DLSSG_DEPTH_INVERTED=1, BB_DLSSG_CAMERA_MOTION=0, BB_DLSSG_HUDLESS=0
    // (FrameGenEvaluate's caller).
    if (!Flag("BB_DLSSG_MATRICES", true)) {
        Perspective(options.cameraViewToClip, evaluate->camera_fov, aspect, evaluate->camera_near,
                    evaluate->camera_far);
        InverseProjection(options.clipToCameraView, options.cameraViewToClip);
        Identity(options.clipToPrevClip);
        Identity(options.prevClipToClip);
    }
    if (Flag("BB_DLSSG_DEPTH_INVERTED", false)) {
        options.depthInverted = !options.depthInverted;
    }
    options.cameraMotionIncluded = Flag("BB_DLSSG_CAMERA_MOTION", true);
    static unsigned logged = 0;
    if (logged++ == 0) {
        Log(0, "Frame Generation evaluation: backbuffer %ux%u (format %u), depth %ux%u (format %u), "
               "motion %ux%u (scale %.6f %.6f), output %ux%u, hudless %s, camera matrices %s",
            evaluate->backbuffer.width, evaluate->backbuffer.height,
            unsigned(evaluate->backbuffer.format), evaluate->depth.width,
            evaluate->depth.height, unsigned(evaluate->depth.format), evaluate->motion.width,
            evaluate->motion.height, options.mvecScale[0], options.mvecScale[1],
            evaluate->output.width, evaluate->output.height,
            evaluate->hudless.image != VK_NULL_HANDLE ? "yes" : "no",
            evaluate->has_matrices ? "the game's" : "made up");
    }
    return Check("Frame Generation evaluation",
                 NGX_VK_EVALUATE_DLSSG(command, state.fg_feature, state.fg_parameters,
                                       &resources, &options))
               ? 1
               : 0;
}

void Shutdown() {
    FrameGenRelease();
    ReleaseFeature();
    if (state.capabilities) {
        NVSDK_NGX_VULKAN_DestroyParameters(state.capabilities);
        state.capabilities = nullptr;
    }
    if (state.initialized) {
        NVSDK_NGX_VULKAN_Shutdown1(state.device);
        state.initialized = false;
        state.device = VK_NULL_HANDLE;
        state.fg_max_count = 0;
        state.fg_device_supported = false;
    }
}

const BbDlssApi api{BBPORT_DLSS_BRIDGE_ABI,
                    Configure,
                    InstanceExtensions,
                    DeviceExtensions,
                    Initialize,
                    CreateFeature,
                    Evaluate,
                    ReleaseFeature,
                    Shutdown,
                    ConfigureDirectories,
                    FrameGenMaxCount,
                    FrameGenCreate,
                    FrameGenEvaluate,
                    FrameGenRelease};

} // namespace

#ifdef _WIN32
#define BBPORT_DLSS_EXPORT __declspec(dllexport)
#else
#define BBPORT_DLSS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" BBPORT_DLSS_EXPORT const BbDlssApi* BbDlssGetApi() {
    return &api;
}
