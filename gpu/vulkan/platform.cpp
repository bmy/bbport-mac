// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#ifdef _WIN32
#define VK_USE_PLATFORM_WIN32_KHR
#elif defined(__APPLE__)
#define VK_USE_PLATFORM_METAL_EXT // bbport (macOS)
#else
#define VK_USE_PLATFORM_WAYLAND_KHR
#define VK_USE_PLATFORM_XLIB_KHR
#endif

#include "vulkan/platform.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fmt/ranges.h>

#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "sdl_window.h"
#include "upscale/vk_dlss.h"
#ifdef _WIN32
#include "platform/win32/nvidia_profile.h"
#endif

#ifdef __APPLE__
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <mach-o/dyld.h>
#endif

namespace Vulkan {

namespace {

constexpr const char* ValidationLayer = "VK_LAYER_KHRONOS_validation";
constexpr const char* CrashDiagnosticLayer = "VK_LAYER_LUNARG_crash_diagnostic";

VKAPI_ATTR VkBool32 VKAPI_CALL OnDebugMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                              vk::DebugUtilsMessageTypeFlagsEXT,
                                              const vk::DebugUtilsMessengerCallbackDataEXT* data,
                                              void*) {
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    const auto level = severity == Severity::eError      ? Common::Log::Level::err
                       : severity == Severity::eInfo ||
                               severity == Severity::eVerbose ? Common::Log::Level::debug
                                                              : Common::Log::Level::info;
    LOG_GENERIC(Common::Log::Class::Render_Vulkan, level, "{}: {}",
                data->pMessageIdName ? data->pMessageIdName : "<null>",
                data->pMessage ? data->pMessage : "<null>");
    return VK_FALSE;
}

bool Has(const auto& available, const char* name, auto member) {
    return std::ranges::any_of(available,
                               [&](const auto& p) { return std::strcmp(name, p.*member) == 0; });
}

/// The layers asked for that are installed.
std::vector<const char*> Layers(bool validation, bool crash_diagnostic) {
    const auto [result, available] = vk::enumerateInstanceLayerProperties();
    if (result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Failed to query layer properties: {}", vk::to_string(result));
        return {};
    }
    std::vector<const char*> layers;
    for (const auto& [wanted, name] : {std::pair{validation, ValidationLayer},
                                       std::pair{crash_diagnostic, CrashDiagnosticLayer}}) {
        if (!wanted) {
            continue;
        }
        if (Has(available, name, &vk::LayerProperties::layerName)) {
            layers.push_back(name);
        } else {
            LOG_ERROR(Render_Vulkan, "Requested layer {} is not available", name);
        }
    }
    return layers;
}

/// Surface extensions of the window system, color spaces for HDR, debug utils, and layer settings
/// when a layer takes them; those the loader lacks left out.
std::vector<const char*> Extensions(Frontend::WindowSystemType window_type,
                                    const std::vector<const char*>& layers) {
    const auto [result, available] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess) {
        LOG_ERROR(Render_Vulkan, "Failed to query extension properties: {}",
                  vk::to_string(result));
        return {};
    }
    std::vector<const char*> extensions;
#ifdef _WIN32
    if (window_type == Frontend::WindowSystemType::Windows) {
        extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
    }
#elif defined(__APPLE__)
    if (window_type == Frontend::WindowSystemType::Metal) {
        extensions.push_back(VK_EXT_METAL_SURFACE_EXTENSION_NAME);
    }
#else
    if (window_type == Frontend::WindowSystemType::X11) {
        extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
    } else if (window_type == Frontend::WindowSystemType::Wayland) {
        extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
    }
#endif
    // The headless renderer's device enables swapchains too, which need this.
    extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
    if (EmulatorSettings.IsHdrAllowed()) {
        extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    }
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    std::erase_if(extensions, [&](const char* name) {
        const bool present = Has(available, name, &vk::ExtensionProperties::extensionName);
        if (!present) {
            LOG_INFO(Render_Vulkan, "Candidate instance extension {} is not available", name);
        }
        return !present;
    });
    bool layer_settings = false;
    for (const char* layer : layers) {
        const auto [layer_result, layer_extensions] =
            vk::enumerateInstanceExtensionProperties(std::string(layer));
        if (Has(layer_extensions, VK_EXT_LAYER_SETTINGS_EXTENSION_NAME,
                &vk::ExtensionProperties::extensionName)) {
            layer_settings = true;
        } else {
            LOG_ERROR(Render_Vulkan, "Settings for layer {} not available.", layer);
        }
    }
    if (layer_settings) {
        extensions.push_back(VK_EXT_LAYER_SETTINGS_EXTENSION_NAME);
    }
    return extensions;
}

} // namespace

vk::UniqueInstance CreateInstance(Frontend::WindowSystemType window_type, bool enable_validation,
                                  bool enable_crash_diagnostic) {
    LOG_INFO(Render_Vulkan, "Creating vulkan instance");
#ifdef _WIN32
    // Before the driver loads (it reads its profile then): NVIDIA cards kept low clocks.
    BbNvidiaProfile::Apply();
#endif
    // AMD Anti-Lag 2 on Linux (render/vk_low_latency): Mesa's anti-lag layer gives
    // VK_AMD_anti_lag on any Mesa driver. Its manifest is implicit but enabled only by this
    // variable, and the layer does nothing for a device created without the antiLag feature.
    // The package carries its own copy (BB_BUNDLED_VK_LAYER_PATH, found before the host's: a host
    // layer is built against the host's libc). Only with BB_ANTI_LAG=1 (issue #289): it gained ~0 in
    // our measurements, and a frame generation layer (lsfg-vk, mako) adds presents the layer does
    // not expect, which crashes it (anti_lag_QueuePresentKHR, signal 11).
    if (bb_env_on("BB_ANTI_LAG")) {
        setenv("ENABLE_LAYER_MESA_ANTI_LAG", "1", 0);
        if (const char* bundled = bb_env("BB_BUNDLED_VK_LAYER_PATH"); bundled && *bundled) {
            const char* added = std::getenv("VK_ADD_IMPLICIT_LAYER_PATH");
            const std::string path =
                added && *added ? std::string{bundled} + ":" + added : std::string{bundled};
            setenv("VK_ADD_IMPLICIT_LAYER_PATH", path.c_str(), 1);
        }
    }
#ifdef __APPLE__
    // bbport (macOS): the driver manifest next to the executable unless the environment names one
    // (tools/macos env.sh and the app name KosmicKrisp).
    static const auto icd_path = [] {
        char path[PATH_MAX];
        u32 length = PATH_MAX;
        _NSGetExecutablePath(path, &length);
        return std::filesystem::path(path).parent_path();
    }();
    setenv("VK_DRIVER_FILES", icd_path.c_str(), false);
    // BB_VULKAN_LIBRARY names the Vulkan loader by path. The release app (no DYLD_LIBRARY_PATH:
    // macOS drops it when bash starts) loads its own copy this way; the default search only tries
    // leaf names.
    static vk::detail::DynamicLoader loader = [] {
        const char* library = bb_env("BB_VULKAN_LIBRARY");
        if (library && library[0]) {
            return vk::detail::DynamicLoader(library);
        }
        return vk::detail::DynamicLoader();
    }();
#else
    static vk::detail::DynamicLoader loader;
#endif
    VULKAN_HPP_DEFAULT_DISPATCHER.init(
        loader.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr"));

    u32 version = VK_API_VERSION_1_0;
    if (VULKAN_HPP_DEFAULT_DISPATCHER.vkEnumerateInstanceVersion) {
        version = Check<"query the Vulkan version">(vk::enumerateInstanceVersion());
    }
    ASSERT_MSG(version >= TargetVulkanApiVersion,
               "Vulkan {}.{} is required, but only {}.{} is supported by instance!",
               VK_VERSION_MAJOR(TargetVulkanApiVersion), VK_VERSION_MINOR(TargetVulkanApiVersion),
               VK_VERSION_MAJOR(version), VK_VERSION_MINOR(version));

    const auto layers = Layers(enable_validation, enable_crash_diagnostic);
    auto extensions = Extensions(window_type, layers);
    if (Dlss* dlss = Dlss::Get()) {
        dlss->AppendInstanceExtensions(extensions); // the optional DLSS bridge needs its own
    }
    LOG_INFO(Render_Vulkan, "Enabled instance extensions: {}", fmt::join(extensions, ", "));
    LOG_INFO(Render_Vulkan, "Enabled instance layers: {}", fmt::join(layers, ", "));

    // Layer settings: the validation checks chosen, crash dumps into the log directory.
    const vk::Bool32 core = EmulatorSettings.IsVkValidationCoreEnabled();
    const vk::Bool32 sync = EmulatorSettings.IsVkValidationSyncEnabled();
    const vk::Bool32 gpu_assisted = EmulatorSettings.IsVkValidationGpuEnabled();
    const vk::Bool32 on = vk::True;
    static const std::string crash_dir = Common::FS::GetUserPathString(Common::FS::PathType::LogDir);
    const char* crash_path = crash_dir.c_str();
    const auto setting = [](const char* layer, const char* name, const auto* value) {
        return vk::LayerSettingEXT{
            .pLayerName = layer,
            .pSettingName = name,
            .type = std::is_same_v<std::remove_cvref_t<decltype(*value)>, const char*>
                        ? vk::LayerSettingTypeEXT::eString
                        : vk::LayerSettingTypeEXT::eBool32,
            .valueCount = 1,
            .pValues = value,
        };
    };
    const std::array settings = {
        setting(ValidationLayer, "validate_core", &core),
        setting(ValidationLayer, "validate_sync", &sync),
        setting(ValidationLayer, "syncval_submit_time_validation", &sync),
        setting(ValidationLayer, "gpuav_enable", &gpu_assisted),
        setting(ValidationLayer, "gpuav_descriptor_checks", &gpu_assisted),
        setting(ValidationLayer, "gpuav_buffers_validation", &gpu_assisted),
        setting(ValidationLayer, "gpuav_indirect_draws_buffers", &gpu_assisted),
        setting(ValidationLayer, "gpuav_indirect_dispatches_buffers", &gpu_assisted),
        setting(ValidationLayer, "gpuav_indirect_trace_rays_buffers", &gpu_assisted),
        setting(ValidationLayer, "gpuav_buffer_copies", &gpu_assisted),
        setting("lunarg_crash_diagnostic", "output_path", &crash_path),
        setting("lunarg_crash_diagnostic", "sync_after_commands", &on),
    };
    const vk::ApplicationInfo app{
        .pApplicationName = "bbport",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "bbport",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = version,
    };
    const vk::StructureChain<vk::InstanceCreateInfo, vk::LayerSettingsCreateInfoEXT> create_info = {
        vk::InstanceCreateInfo{
            .pApplicationInfo = &app,
            .enabledLayerCount = u32(layers.size()),
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = u32(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        },
        vk::LayerSettingsCreateInfoEXT{
            .settingCount = u32(settings.size()),
            .pSettings = settings.data(),
        },
    };
    auto instance = Check<"create the Vulkan instance">(vk::createInstanceUnique(create_info.get()));
    VULKAN_HPP_DEFAULT_DISPATCHER.init(*instance);
    return instance;
}

vk::SurfaceKHR CreateSurface(vk::Instance instance, const Frontend::WindowSDL& window) {
    const auto info = window.GetWindowInfo();
    vk::SurfaceKHR surface{};
    vk::Result result = vk::Result::eErrorInitializationFailed;
#ifdef _WIN32
    if (info.type == Frontend::WindowSystemType::Windows) {
        const vk::Win32SurfaceCreateInfoKHR create_info{
            .hinstance = static_cast<HINSTANCE>(info.display_connection),
            .hwnd = static_cast<HWND>(info.render_surface),
        };
        result = instance.createWin32SurfaceKHR(&create_info, nullptr, &surface);
    }
#elif defined(__APPLE__)
    if (info.type == Frontend::WindowSystemType::Metal) {
        const vk::MetalSurfaceCreateInfoEXT create_info{
            .pLayer = static_cast<const CAMetalLayer*>(info.render_surface),
        };
        result = instance.createMetalSurfaceEXT(&create_info, nullptr, &surface);
    }
#else
    if (info.type == Frontend::WindowSystemType::X11) {
        const vk::XlibSurfaceCreateInfoKHR create_info{
            .dpy = static_cast<Display*>(info.display_connection),
            .window = reinterpret_cast<Window>(info.render_surface),
        };
        result = instance.createXlibSurfaceKHR(&create_info, nullptr, &surface);
    } else if (info.type == Frontend::WindowSystemType::Wayland) {
        const vk::WaylandSurfaceCreateInfoKHR create_info{
            .display = static_cast<wl_display*>(info.display_connection),
            .surface = static_cast<wl_surface*>(info.render_surface),
        };
        result = instance.createWaylandSurfaceKHR(&create_info, nullptr, &surface);
    }
#endif
    ASSERT_MSG(result == vk::Result::eSuccess && surface, "no window surface ({})",
               vk::to_string(result));
    return surface;
}

vk::UniqueDebugUtilsMessengerEXT CreateDebugCallback(vk::Instance instance) {
    using Severity = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
    return Check<"create the debug messenger">(instance.createDebugUtilsMessengerEXTUnique({
        .messageSeverity =
            Severity::eInfo | Severity::eError | Severity::eWarning | Severity::eVerbose,
        .messageType = Type::eGeneral | Type::eValidation | Type::ePerformance,
        .pfnUserCallback = OnDebugMessage,
    }));
}

} // namespace Vulkan
