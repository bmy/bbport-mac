// SPDX-FileCopyrightText: Copyright 2026 IFreemz, bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "shim/bb_env.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string_view>

#include "upscale/vk_dlss.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <cstring>
#include <fstream>
#include <iterator>
#include <fmt/format.h>
#include "../../../dlss_bridge/bbport_dlss_bridge.h"
#elif defined(__linux__)
#include <dlfcn.h>
#include "../../../dlss_bridge/bbport_dlss_bridge.h"
#endif

namespace Vulkan {

int Dlss::QualityForScale(float scale) {
    // DLAA only for equal sizes: any real upscale needs at least the Quality mode.
    return scale >= 2.9f ? 4 : scale >= 1.95f ? 3 : scale >= 1.65f ? 2 : scale > 1.01f ? 1 : 0;
}

#if defined(_WIN32) || defined(__linux__)

namespace {
#ifdef _WIN32
constexpr wchar_t BridgeName[] = L"bbport_dlss.dll";
constexpr wchar_t NgxName[] = L"nvngx_dlss.dll";
constexpr wchar_t NgxFrameGenName[] = L"nvngx_dlssg.dll";
using Module = HMODULE;

std::filesystem::path ExecutableDirectory() {
    std::wstring path(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    path.resize(length);
    return std::filesystem::path{path}.parent_path();
}

bool HasNgx(const std::filesystem::path& directory) {
    std::error_code error;
    return std::filesystem::is_regular_file(directory / NgxName, error);
}

bool HasNgxFrameGen(const std::filesystem::path& directory) {
    std::error_code error;
    return std::filesystem::is_regular_file(directory / NgxFrameGenName, error);
}

Module LoadModule(const std::filesystem::path& path) {
    return LoadLibraryW(path.c_str());
}

/// The FileVersion of a DLL's version resource ("310.9.16.0"), read from the file itself; empty
/// when it has none. Which NVIDIA library a player runs decides what DLSS Frame Generation
/// understands (a game's older copy, e.g. War Thunder's, against this SDK's parameters).
std::string FileVersion(const std::filesystem::path& path) {
    std::ifstream file{path, std::ios::binary};
    const std::string bytes{std::istreambuf_iterator<char>(file), {}};
    // VS_FIXEDFILEINFO follows the UTF-16 key "VS_VERSION_INFO" (aligned): its signature.
    const std::string signature{"\xbd\x04\xef\xfe", 4};
    const std::string key{"V\0S\0_\0V\0E\0R\0S\0I\0O\0N\0_\0I\0N\0F\0O\0", 30};
    const auto resource = bytes.rfind(key);
    const auto at = resource == std::string::npos ? std::string::npos : bytes.find(signature, resource);
    if (at == std::string::npos || at + 16 > bytes.size()) {
        return {};
    }
    u32 ms = 0, ls = 0;
    std::memcpy(&ms, bytes.data() + at + 8, 4);
    std::memcpy(&ls, bytes.data() + at + 12, 4);
    return fmt::format("{}.{}.{}.{}", ms >> 16, ms & 0xffff, ls >> 16, ls & 0xffff);
}

/// Hardware-accelerated GPU scheduling (Windows 10 2004+): NVIDIA's DLSS Frame Generation needs
/// it (RTX Remix checks it before offering frame generation). -1 unknown, 0 off, 1 on.
int HardwareScheduling() {
    DWORD mode = 0, size = sizeof(mode);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers",
                     L"HwSchMode", RRF_RT_REG_DWORD, nullptr, &mode, &size) != ERROR_SUCCESS) {
        return -1;
    }
    return mode == 2 ? 1 : 0;
}

void* FindSymbol(Module module, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(module, name));
}
#else
// bbport (Linux): libbbport_dlss.so (gpu/dlss_bridge) and NVIDIA's libnvidia-ngx-dlss.so.<version>
// (and libnvidia-ngx-dlssg.so.<version> for Frame Generation) next to bb-probe or in the player's
// DLSS folder; NGX finds them through the bridge's path list.
constexpr char BridgeName[] = "libbbport_dlss.so";
constexpr std::string_view NgxPrefix = "libnvidia-ngx-dlss.so.";
constexpr std::string_view NgxFrameGenPrefix = "libnvidia-ngx-dlssg.so.";
using Module = void*;

std::filesystem::path ExecutableDirectory() {
    std::error_code error;
    return std::filesystem::read_symlink("/proc/self/exe", error).parent_path();
}

bool HasLibrary(const std::filesystem::path& directory, std::string_view prefix) {
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        if (entry.path().filename().string().starts_with(prefix)) {
            return true;
        }
    }
    return false;
}

bool HasNgx(const std::filesystem::path& directory) {
    return HasLibrary(directory, NgxPrefix);
}

bool HasNgxFrameGen(const std::filesystem::path& directory) {
    return HasLibrary(directory, NgxFrameGenPrefix);
}

Module LoadModule(const std::filesystem::path& path) {
    Module module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        std::printf("DLSS: %s\n", dlerror());
    }
    return module;
}

void* FindSymbol(Module module, const char* name) {
    return dlsym(module, name);
}
#endif

/// The player's DLSS folder (<user>/dlss, BB_GPU_USER_DIR; beside bb-probe without it): NGX's
/// logs, and NVIDIA's library when the player chose one in the launcher (newer than the package's,
/// or the only one: the package may come without it).
std::filesystem::path UserDlssDirectory() {
    if (const char* user = bb_env("BB_GPU_USER_DIR"); user && user[0]) {
        std::error_code error;
        const auto absolute = std::filesystem::absolute(std::filesystem::path{user} / "dlss", error);
        return error ? std::filesystem::path{user} / "dlss" : absolute;
    }
    return ExecutableDirectory() / "dlss";
}

/// Where NVIDIA's DLSS library is: the player's own first, then beside bb-probe; empty: nowhere.
std::filesystem::path NgxDirectory() {
    for (const auto& directory : {UserDlssDirectory(), ExecutableDirectory()}) {
        if (HasNgx(directory)) {
            return directory;
        }
    }
    return {};
}

/// The directories NGX searches for NVIDIA's libraries, in order: the player's own folder (a
/// newer Super Resolution library, or the Frame Generation one added to a package without it),
/// then beside bb-probe. Only those holding one of the libraries.
std::vector<std::filesystem::path> NgxDirectories() {
    std::vector<std::filesystem::path> found;
    for (const auto& directory : {UserDlssDirectory(), ExecutableDirectory()}) {
        if (HasNgx(directory) || HasNgxFrameGen(directory)) {
            found.push_back(directory);
        }
    }
    return found;
}

bool HasNgxFrameGenAnywhere() {
    for (const auto& directory : {UserDlssDirectory(), ExecutableDirectory()}) {
        if (HasNgxFrameGen(directory)) {
            return true;
        }
    }
    return false;
}

/// Where the bridge is: beside bb-probe (the package, a build with DLSS_SDK_ROOT), else the
/// player's DLSS folder; empty: nowhere.
std::filesystem::path BridgeDirectory() {
    for (const auto& directory : {ExecutableDirectory(), UserDlssDirectory()}) {
        std::error_code error;
        if (std::filesystem::is_regular_file(directory / BridgeName, error)) {
            return directory;
        }
    }
    return {};
}

void BridgeLog(int warning, const char* message) {
    std::printf("DLSS: %s%s\n", warning ? "warning: " : "", message);
}

bool CollectExtensions(const VkExtensionProperties* required, u32 count,
                       const std::vector<vk::ExtensionProperties>& available,
                       std::vector<std::string>& storage, std::vector<const char*>& enabled) {
    storage.clear();
    for (u32 i = 0; i < count; ++i) {
        const auto& request = required[i];
        const auto match = std::ranges::find_if(available, [&](const auto& extension) {
            return std::strcmp(extension.extensionName.data(), request.extensionName) == 0 &&
                   extension.specVersion >= request.specVersion;
        });
        if (match == available.end()) {
            std::printf("DLSS: missing Vulkan extension %s\n", request.extensionName);
            return false;
        }
        storage.emplace_back(request.extensionName);
    }
    for (const auto& name : storage) {
        if (std::ranges::none_of(enabled, [&](const char* current) { return name == current; })) {
            enabled.push_back(name.c_str());
        }
    }
    return true;
}
} // namespace

struct Dlss::Impl {
    Module module{};
    const BbDlssApi* api{};
    std::vector<std::string> instance_extensions, device_extensions;
    std::optional<FeatureDesc> feature;
    std::optional<FrameGenDesc> frame_gen;
    int frame_gen_max{};
    std::string problem, frame_gen_problem;
    bool eligible{true}, instance_ready{}, device_ready{}, initialized{}, available{};

    void Disable(std::string_view reason) {
        eligible = available = false;
        problem = reason;
        std::printf("DLSS: unavailable (%.*s)\n", int(reason.size()), reason.data());
    }
};

Dlss* Dlss::Get() {
    static Dlss* const dlss = []() -> Dlss* {
        if (bb_env_off("BB_DLSS")) {
            return nullptr;
        }
        if (BridgeDirectory().empty()) {
            return nullptr;
        }
        auto* created = new Dlss;
        // The process may end through quick_exit/_exit paths: release NGX first when it can.
        std::atexit([] {
            if (Dlss* live = Get()) {
                live->Shutdown();
            }
        });
        return created;
    }();
    return dlss;
}

Dlss::Dlss() : impl{std::make_unique<Impl>()} {
    impl->module = LoadModule(BridgeDirectory() / BridgeName);
    const auto get_api =
        impl->module ? reinterpret_cast<BbDlssGetApiFn>(FindSymbol(impl->module, "BbDlssGetApi"))
                     : nullptr;
    impl->api = get_api ? get_api() : nullptr;
    // ABI 1 bridges (Super Resolution only) still work; Frame Generation needs ABI 2.
    if (!impl->api || impl->api->abi < 1 || impl->api->abi > BBPORT_DLSS_BRIDGE_ABI) {
        impl->Disable("the DLSS bridge library is missing or from another version");
        return;
    }
    const auto ngx = NgxDirectory();
    if (ngx.empty()) {
        impl->Disable("NVIDIA's DLSS library is missing: choose it in the launcher (Upscaler)");
        return;
    }
    // NGX writes its logs and model updates here: beside the saves and shader caches.
    const auto data = UserDlssDirectory();
    std::error_code error;
    std::filesystem::create_directories(data, error);
    std::printf("DLSS: NVIDIA's library from %s\n", ngx.string().c_str());
#ifdef _WIN32
    for (const auto& directory : NgxDirectories()) {
        for (const wchar_t* name : {NgxName, NgxFrameGenName}) {
            if (std::filesystem::is_regular_file(directory / name, error)) {
                const auto version = FileVersion(directory / name);
                std::printf("DLSS: %s %s in %s\n", std::filesystem::path{name}.string().c_str(),
                            version.empty() ? "(no version)" : version.c_str(),
                            directory.string().c_str());
            }
        }
    }
    if (const int hags = HardwareScheduling(); hags == 0) {
        std::printf("DLSS: hardware-accelerated GPU scheduling is OFF (Windows settings, Display, "
                    "Graphics): DLSS Frame Generation needs it on\n");
    } else if (hags == 1) {
        std::printf("DLSS: hardware-accelerated GPU scheduling is on\n");
    }
#endif
    bool configured = false;
    if (impl->api->abi >= 2) {
        // Every folder with one of NVIDIA's libraries: the Frame Generation library may be in
        // the player's folder while the package brought Super Resolution's.
        const auto directories = NgxDirectories();
        std::vector<std::wstring> storage;
        std::vector<const wchar_t*> pointers;
        for (const auto& directory : directories) {
            storage.push_back(directory.wstring());
        }
        for (const auto& text : storage) {
            pointers.push_back(text.c_str());
        }
        configured = impl->api->ConfigureDirectories(pointers.data(), u32(pointers.size()),
                                                     data.wstring().c_str(), BridgeLog) != 0;
    } else {
        configured =
            impl->api->Configure(ngx.wstring().c_str(), data.wstring().c_str(), BridgeLog) != 0;
    }
    if (!configured) {
        impl->Disable("bridge configuration failed");
    }
}

Dlss::~Dlss() {
    Shutdown();
}

void Dlss::AppendInstanceExtensions(std::vector<const char*>& enabled) {
    if (!impl->eligible) {
        return;
    }
    u32 count{};
    const VkExtensionProperties* required{};
    if (!impl->api->InstanceExtensions(&count, &required)) {
        return impl->Disable("instance extension query failed");
    }
    const auto [result, extensions] = vk::enumerateInstanceExtensionProperties();
    if (result != vk::Result::eSuccess ||
        !CollectExtensions(required, count, extensions, impl->instance_extensions, enabled)) {
        return impl->Disable("required Vulkan instance extensions are missing");
    }
    impl->instance_ready = true;
}

void Dlss::AppendDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                                  std::vector<const char*>& enabled) {
    if (!impl->eligible || !impl->instance_ready) {
        return;
    }
    const auto properties = physical.getProperties();
    if (properties.vendorID != 0x10de) {
        return impl->Disable("not an NVIDIA GPU");
    }
    u32 count{};
    const VkExtensionProperties* required{};
    if (!impl->api->DeviceExtensions(instance, physical, &count, &required)) {
        return impl->Disable("this GPU or driver does not support DLSS (GeForce RTX needed)");
    }
    const auto [result, extensions] = physical.enumerateDeviceExtensionProperties();
    if (result != vk::Result::eSuccess ||
        !CollectExtensions(required, count, extensions, impl->device_extensions, enabled)) {
        return impl->Disable("required Vulkan device extensions are missing");
    }
    impl->device_ready = true;
}

void Dlss::Disable(std::string_view reason) {
    impl->Disable(reason);
}

void Dlss::Initialize(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device) {
    if (!impl->eligible || !impl->device_ready || impl->initialized) {
        return;
    }
    impl->initialized = true;
    if (!impl->api->Initialize(instance, physical, device,
                               VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                               VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr)) {
        return impl->Disable("NVIDIA NGX initialization failed (update the GPU driver)");
    }
    impl->available = true;
    std::printf("DLSS: ready\n");
    if (impl->api->abi >= 2) {
        impl->frame_gen_max = std::max(0, impl->api->FrameGenMaxCount());
        if (impl->frame_gen_max < 1) {
            impl->frame_gen_problem =
                HasNgxFrameGenAnywhere()
                    ? "this GPU or driver has no DLSS Frame Generation (a GeForce RTX 40 or "
                      "newer and a current driver are needed)"
                    : "NVIDIA's Frame Generation library (libnvidia-ngx-dlssg.so) is missing: "
                      "choose it in the launcher (Upscaler)";
        }
    } else {
        impl->frame_gen_problem = "the DLSS bridge predates Frame Generation: rebuild it "
                                  "(DLSS_SDK_ROOT=... bash build.sh)";
    }
    std::printf("DLSS: Frame Generation %s\n",
                impl->frame_gen_max > 0 ? (impl->frame_gen_max > 1 ? "up to 4x" : "2x")
                                        : impl->frame_gen_problem.c_str());
}

bool Dlss::Available() const {
    return impl->available;
}

std::string Dlss::Problem() const {
    return impl->problem;
}

bool Dlss::HasFeature(const FeatureDesc& desc) const {
    return impl->feature && *impl->feature == desc;
}

bool Dlss::CreateFeature(vk::CommandBuffer command, const FeatureDesc& desc) {
    if (!impl->available) {
        return false;
    }
    impl->feature.reset();
    const BbDlssFeature feature{desc.input_width,  desc.input_height, desc.output_width,
                                desc.output_height, desc.quality,      0,
                                0,                  desc.hdr ? 1 : 0};
    if (!impl->api->CreateFeature(command, &feature)) {
        return false;
    }
    impl->feature = desc;
    return true;
}

bool Dlss::Evaluate(vk::CommandBuffer command, const Frame& frame) {
    if (!impl->available || !impl->feature) {
        return false;
    }
    const auto image = [](const Resource& r) {
        const vk::ImageSubresourceRange range{r.aspect, 0, 1, 0, 1};
        return BbDlssImage{r.image,
                           r.view,
                           static_cast<VkImageSubresourceRange>(range),
                           static_cast<VkFormat>(r.format),
                           r.width,
                           r.height};
    };
    const BbDlssEvaluate parameters{image(frame.color),  image(frame.depth), image(frame.motion),
                                    image(frame.output), frame.jitter_x,     frame.jitter_y,
                                    frame.reset ? 1 : 0, frame.frame_ms,     frame.sharpness};
    return impl->api->Evaluate(command, &parameters) != 0;
}

void Dlss::ReleaseFeature() {
    impl->feature.reset();
    if (impl->api) {
        impl->api->ReleaseFeature();
    }
}

int Dlss::FrameGenMaxCount() const {
    return impl->available ? impl->frame_gen_max : 0;
}

std::string Dlss::FrameGenProblem() const {
    if (!impl->problem.empty()) {
        return impl->problem;
    }
    return impl->available ? impl->frame_gen_problem : std::string{};
}

bool Dlss::HasFrameGen(const FrameGenDesc& desc) const {
    return impl->frame_gen && *impl->frame_gen == desc;
}

bool Dlss::CreateFrameGen(vk::CommandBuffer command, const FrameGenDesc& desc) {
    if (FrameGenMaxCount() < 1 || impl->api->abi < 2) {
        return false;
    }
    impl->frame_gen.reset();
    const BbDlssgFeature feature{desc.width, desc.height, static_cast<VkFormat>(desc.format),
                                 desc.render_width, desc.render_height};
    if (!impl->api->FrameGenCreate(command, &feature)) {
        return false;
    }
    impl->frame_gen = desc;
    return true;
}

bool Dlss::EvaluateFrameGen(vk::CommandBuffer command, const FrameGenFrame& frame) {
    if (!impl->frame_gen) {
        return false;
    }
    const auto image = [](const Resource& r) {
        const vk::ImageSubresourceRange range{r.aspect, 0, 1, 0, 1};
        return BbDlssImage{r.image,
                           r.view,
                           static_cast<VkImageSubresourceRange>(range),
                           static_cast<VkFormat>(r.format),
                           r.width,
                           r.height};
    };
    BbDlssgEvaluate parameters{};
    parameters.backbuffer = image(frame.backbuffer);
    parameters.depth = image(frame.depth);
    parameters.motion = image(frame.motion);
    parameters.output = image(frame.output);
    parameters.hudless = frame.hudless.image ? image(frame.hudless) : BbDlssImage{};
    parameters.jitter_x = frame.jitter_x;
    parameters.jitter_y = frame.jitter_y;
    parameters.camera_near = frame.camera_near;
    parameters.camera_far = frame.camera_far;
    parameters.camera_fov = frame.camera_fov;
    for (int i = 0; i < 3; ++i) {
        parameters.camera_position[i] = frame.camera_position[i];
        parameters.camera_up[i] = frame.camera_up[i];
        parameters.camera_right[i] = frame.camera_right[i];
        parameters.camera_forward[i] = frame.camera_forward[i];
    }
    parameters.depth_inverted = frame.depth_inverted ? 1 : 0;
    parameters.reset = frame.reset ? 1 : 0;
    parameters.multi_frame_count = frame.count;
    parameters.multi_frame_index = frame.index;
    parameters.has_matrices = frame.has_matrices ? 1 : 0;
    std::memcpy(parameters.view_to_clip, frame.view_to_clip.data(), sizeof(parameters.view_to_clip));
    std::memcpy(parameters.clip_to_prev_clip, frame.clip_to_prev_clip.data(),
                sizeof(parameters.clip_to_prev_clip));
    std::memcpy(parameters.prev_clip_to_clip, frame.prev_clip_to_clip.data(),
                sizeof(parameters.prev_clip_to_clip));
    return impl->api->FrameGenEvaluate(command, &parameters) != 0;
}

void Dlss::ReleaseFrameGen() {
    impl->frame_gen.reset();
    if (impl->api && impl->api->abi >= 2 && impl->initialized) {
        impl->api->FrameGenRelease();
    }
}

void Dlss::Shutdown() {
    impl->available = false;
    impl->feature.reset();
    impl->frame_gen.reset();
    impl->frame_gen_max = 0;
    if (impl->api && impl->initialized) {
        impl->api->Shutdown();
        impl->initialized = false;
    }
}

#else

struct Dlss::Impl {};
Dlss::Dlss() = default;
Dlss::~Dlss() = default;
Dlss* Dlss::Get() {
    return nullptr;
}
void Dlss::AppendInstanceExtensions(std::vector<const char*>&) {}
void Dlss::AppendDeviceExtensions(vk::Instance, vk::PhysicalDevice, std::vector<const char*>&) {}
void Dlss::Initialize(vk::Instance, vk::PhysicalDevice, vk::Device) {}
void Dlss::Disable(std::string_view) {}
void Dlss::Shutdown() {}
bool Dlss::Available() const {
    return false;
}
std::string Dlss::Problem() const {
    return {};
}
bool Dlss::HasFeature(const FeatureDesc&) const {
    return false;
}
bool Dlss::CreateFeature(vk::CommandBuffer, const FeatureDesc&) {
    return false;
}
bool Dlss::Evaluate(vk::CommandBuffer, const Frame&) {
    return false;
}
void Dlss::ReleaseFeature() {}
int Dlss::FrameGenMaxCount() const {
    return 0;
}
std::string Dlss::FrameGenProblem() const {
    return {};
}
bool Dlss::HasFrameGen(const FrameGenDesc&) const {
    return false;
}
bool Dlss::CreateFrameGen(vk::CommandBuffer, const FrameGenDesc&) {
    return false;
}
bool Dlss::EvaluateFrameGen(vk::CommandBuffer, const FrameGenFrame&) {
    return false;
}
void Dlss::ReleaseFrameGen() {}

#endif

} // namespace Vulkan
