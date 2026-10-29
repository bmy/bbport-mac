// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport vulkan: the Vulkan instance (with validation and crash diagnostic layers on request),
// window surfaces (X11 and Wayland), debug names of objects, and checked results.
#pragma once

#include <string>
#include <string_view>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/string_literal.h"
#include "common/types.h"
#include "render/vk_breadcrumbs.h"
#include "vulkan/vk_common.h"

namespace Frontend {
enum class WindowSystemType : u8;
class WindowSDL;
} // namespace Frontend

namespace Vulkan {

constexpr u32 TargetVulkanApiVersion = VK_API_VERSION_1_3;

vk::UniqueInstance CreateInstance(Frontend::WindowSystemType window_type, bool enable_validation,
                                  bool enable_crash_diagnostic);
vk::SurfaceKHR CreateSurface(vk::Instance instance, const Frontend::WindowSDL& window);
/// Validation and driver messages into the log.
vk::UniqueDebugUtilsMessengerEXT CreateDebugCallback(vk::Instance instance);

template <typename T>
concept VulkanHandleType = vk::isVulkanHandleType<T>::value;

/// Names the object for debuggers and validation messages (when debug utils are there).
template <VulkanHandleType Handle>
void SetObjectName(vk::Device device, const Handle& handle, std::string_view name) {
    const std::string terminated{name};
    const vk::Result result = device.setDebugUtilsObjectNameEXT({
        .objectType = Handle::objectType,
        .objectHandle = reinterpret_cast<u64>(static_cast<typename Handle::NativeType>(handle)),
        .pObjectName = terminated.c_str(),
    });
    if (result != vk::Result::eSuccess) {
        LOG_DEBUG(Render_Vulkan, "Could not set object debug name: {}", vk::to_string(result));
    }
}
template <VulkanHandleType Handle, typename... Args>
void SetObjectName(vk::Device device, const Handle& handle, const char* format,
                   const Args&... args) {
    SetObjectName(device, handle, std::string_view{fmt::vformat(format, fmt::make_format_args(args...))});
}

/// Stops the port on a failed call (a lost device with a report of the GPU's last steps).
template <StringLiteral what = "">
void Check(vk::Result result) {
    if (result == vk::Result::eErrorDeviceLost) {
        Breadcrumbs::ReportDeviceLost("a Vulkan call");
    }
    if constexpr (what.len <= 1) {
        ASSERT_MSG(result == vk::Result::eSuccess, "vk::Result={}", vk::to_string(result));
    } else {
        ASSERT_MSG(result == vk::Result::eSuccess, "Failed to {}: vk::Result={}", what.value,
                   vk::to_string(result));
    }
}
template <StringLiteral what = "", typename T>
T Check(vk::ResultValue<T> result) {
    Check<what>(result.result);
    return std::move(result.value);
}

} // namespace Vulkan
