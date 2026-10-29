// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_fsr_amd.h"

#ifdef _WIN32

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <thread>
#include <vector>

#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "render/scheduler.h"
#include "shim/bb_env.h"
#include "third_party/ffx-api-2.4/api/include/vk/ffx_api_vk.h"
#include "third_party/ffx-api-2.4/framegeneration/include/ffx_framegeneration.h"
#include "third_party/ffx-api-2.4/upscalers/include/ffx_upscale.h"
#include "vulkan/instance.h"

namespace Vulkan {

namespace {

constexpr wchar_t LoaderName[] = L"amd_fidelityfx_loader_vk.dll";
constexpr wchar_t UpscalerName[] = L"amd_fidelityfx_upscaler_vk.dll";
/// The backend keeps FFX_MAX_QUEUED_FRAMES (4) frames of views and constants: at most this many
/// dispatches may be on the GPU.
constexpr size_t FramesInFlight = 3;

std::filesystem::path ExecutableDirectory() {
    std::wstring path(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    path.resize(length);
    return std::filesystem::path{path}.parent_path();
}

/// Where the player put AMD's DLLs: BB_FFX_DIR, <user>/fsr, beside bb-probe.
std::filesystem::path FindDirectory() {
    std::vector<std::filesystem::path> dirs;
    if (const char* dir = bb_env("BB_FFX_DIR"); dir && dir[0]) dirs.emplace_back(dir);
    if (const char* user = bb_env("BB_GPU_USER_DIR"); user && user[0]) {
        dirs.push_back(std::filesystem::path{user} / "fsr");
    }
    dirs.push_back(ExecutableDirectory());
    for (const auto& dir : dirs) {
        std::error_code error;
        if (std::filesystem::is_regular_file(dir / LoaderName, error)) return dir;
    }
    return {};
}

void Message(uint32_t type, const wchar_t* message) {
    std::string text;
    for (; message && *message; ++message) {
        text += *message < 128 ? char(*message) : '?';
    }
    std::printf("Upscaler: AMD FSR %s: %s\n",
                type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", text.c_str());
}

bool IsStencil(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint ||
           format == vk::Format::eD16UnormS8Uint;
}

bool IsDepth(vk::Format format) {
    return IsStencil(format) || format == vk::Format::eD32Sfloat ||
           format == vk::Format::eD16Unorm || format == vk::Format::eX8D24UnormPack32;
}

// bbport: the FSR 4 model. The DLL (and the driver's amdvkffx64.dll) holds four FSR 4 providers and
// takes the first, in this order, whose device check (the provider's third virtual function)
// accepts the GPU. A model chosen in the menu answers yes there and the others no, in the loaded
// module's vtables (as tools/fsr4cap/driver_model.py does in a copy of the file); "auto" puts the
// checks back. Indexed by BbSettings::Fsr4Model - 1.
constexpr const char* ProviderTypes[] = {
    ".?AVffxProvider_FSR4@@",            // fp8 (RDNA4)
    ".?AVffxProvider_FSR4_Int8@@",       // int8 (RDNA3)
    ".?AVffxProvider_FSR4_Int8_Navi2@@", // int8_rdna2
    ".?AVffxProvider_FSR4Lite_Int8@@",   // int8_lite (Steam Deck, RDNA 3.5 APUs)
};
constexpr size_t DeviceCheckSlot = 2;

bool __cdecl CheckYes(void*, void*, void*, void*) {
    return true;
}
bool __cdecl CheckNo(void*, void*, void*, void*) {
    return false;
}

/// The vtable entry of the device check of the class with RTTI name `type` in `module`, or null:
/// type descriptor (name at +16) -> complete object locator (signature 1, descriptor at +12, its
/// own RVA at +20) -> the vtable after the pointer to it.
void** DeviceCheck(HMODULE module, const char* type) {
    auto* const base = reinterpret_cast<u8*>(module);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    std::vector<std::pair<u8*, size_t>> sections;
    const IMAGE_SECTION_HEADER* section = IMAGE_FIRST_SECTION(nt);
    for (u32 i = 0; i < nt->FileHeader.NumberOfSections; ++i, ++section) {
        sections.emplace_back(base + section->VirtualAddress,
                              std::min<size_t>(section->Misc.VirtualSize, section->SizeOfRawData));
    }
    const size_t length = std::strlen(type) + 1; // with its terminator
    u8* name = nullptr;
    for (const auto& [begin, size] : sections) {
        for (size_t at = 16; !name && at + length <= size; ++at) {
            if (begin[at] == '.' && std::memcmp(begin + at, type, length) == 0) name = begin + at;
        }
    }
    if (!name) return nullptr;
    const u32 descriptor = u32(name - 16 - base);
    for (const auto& [begin, size] : sections) {
        for (size_t at = 12; at + 12 <= size; at += 4) {
            u32 value;
            std::memcpy(&value, begin + at, 4);
            if (value != descriptor) continue;
            u8* const locator = begin + at - 12;
            u32 signature, self;
            std::memcpy(&signature, locator, 4);
            std::memcpy(&self, locator + 20, 4);
            if (signature != 1 || self != u32(locator - base)) continue;
            const u64 pointer = u64(locator);
            for (const auto& [vbegin, vsize] : sections) {
                for (size_t v = 0; v + 8 * (DeviceCheckSlot + 2) <= vsize; v += 8) {
                    u64 entry;
                    std::memcpy(&entry, vbegin + v, 8);
                    if (entry == pointer) {
                        return reinterpret_cast<void**>(vbegin + v + 8) + DeviceCheckSlot;
                    }
                }
            }
        }
    }
    return nullptr;
}

FfxApiResource Resource(const Fsr4Upscaler::Image& image, uint32_t usage, uint32_t state) {
    FfxApiResource r{};
    r.resource = reinterpret_cast<void*>(VkImage(image.image));
    r.state = state;
    r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    r.description.format = ffxApiGetSurfaceFormatVK(VkFormat(image.format));
    r.description.width = image.width;
    r.description.height = image.height;
    r.description.depth = 1;
    r.description.mipCount = 1;
    r.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    if (IsDepth(image.format)) usage |= FFX_API_RESOURCE_USAGE_DEPTHTARGET;
    if (IsStencil(image.format)) usage |= FFX_API_RESOURCE_USAGE_STENCILTARGET;
    r.description.usage = usage;
    return r;
}

} // namespace

struct FsrAmd::Impl {
    const Instance& instance;
    Scheduler& scheduler;
    HMODULE module = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    /// The descriptions a context keeps pointers to until it is destroyed.
    VkDeviceContext device_context{};
    ffxCreateBackendVKDesc backend{};
    ffxOverrideVersion override_version{};
    bool override_on = false;

    /// The device checks of the FSR 4 providers found (ProviderTypes index), with their own code.
    struct Check {
        void** slot;
        void* original;
        int provider;
    };
    std::vector<Check> checks;
    std::vector<HMODULE> scanned;

    struct Context {
        ffxContext context = nullptr;
        u32 width = 0, height = 0;
        bool auto_exposure = false;
        int model = BbSettings::Fsr4Auto;
        std::string provider;
        std::string error;
    };
    Context current;
    std::deque<u64> ticks; ///< scheduler ticks of the dispatches on the GPU
    std::unique_ptr<Context> staging;
    std::thread staging_thread;
    std::atomic<bool> staging_done{false};

    Impl(const Instance& instance_, Scheduler& scheduler_)
        : instance{instance_}, scheduler{scheduler_} {
        const auto dir = FindDirectory();
        if (dir.empty()) {
            std::printf("Upscaler: AMD's FSR DLLs not found (BB_FFX_DIR, <user>/fsr, beside the "
                        "game)\n");
            return;
        }
        std::error_code error;
        if (!std::filesystem::is_regular_file(dir / UpscalerName, error)) {
            std::printf("Upscaler: %s has the FSR loader but not amd_fidelityfx_upscaler_vk.dll\n",
                        dir.string().c_str());
        }
        module = LoadLibraryExW((dir / LoaderName).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!module) {
            std::printf("Upscaler: loading %s failed (%lu)\n", (dir / LoaderName).string().c_str(),
                        GetLastError());
            return;
        }
        create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
        destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
        query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
        dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
        if (!create || !destroy || !query || !dispatch) {
            std::printf("Upscaler: %s lacks the FSR API\n", (dir / LoaderName).string().c_str());
            FreeLibrary(module);
            module = nullptr;
            return;
        }
        device_context.vkDevice = VkDevice(instance.GetDevice());
        device_context.vkPhysicalDevice = VkPhysicalDevice(instance.GetPhysicalDevice());
        device_context.vkDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.deviceContext = device_context;
        ListVersions(dir);
        FindChecks();
        unsigned models = 0;
        for (const Check& c : checks) {
            if (c.provider != 0 || instance.IsFloat8MatrixSupported()) models |= 2u << c.provider;
        }
        BbSettings::Get().fsr_amd_models.store(models);
    }

    /// The providers' device checks in the loaded upscaler DLL and the driver's (each scanned once).
    void FindChecks() {
        for (const wchar_t* name : {UpscalerName, L"amdvkffx64.dll"}) {
            const HMODULE m = GetModuleHandleW(name);
            if (!m || std::find(scanned.begin(), scanned.end(), m) != scanned.end()) continue;
            scanned.push_back(m);
            std::string found;
            for (int i = 0; i < int(std::size(ProviderTypes)); ++i) {
                if (void** slot = DeviceCheck(m, ProviderTypes[i])) {
                    checks.push_back({slot, *slot, i});
                    found += std::string{found.empty() ? "" : ", "} +
                             BbSettings::Fsr4ModelName(i + 1);
                }
            }
            std::printf("Upscaler: AMD FSR 4 models in %ls: %s\n", name,
                        found.empty() ? "none" : found.c_str());
        }
    }

    /// Makes the providers accept only `model` (Fsr4Model), or all as they would (auto).
    void ApplyModel(int model) {
        FindChecks();
        for (const Check& c : checks) {
            void* const want = model == BbSettings::Fsr4Auto ? c.original
                               : c.provider + 1 == model
                                   ? reinterpret_cast<void*>(&CheckYes)
                                   : reinterpret_cast<void*>(&CheckNo);
            if (*c.slot == want) continue;
            DWORD old = 0;
            if (VirtualProtect(c.slot, sizeof(void*), PAGE_READWRITE, &old)) {
                *c.slot = want;
                VirtualProtect(c.slot, sizeof(void*), old, &old);
            }
        }
    }

    ~Impl() {
        if (staging_thread.joinable()) staging_thread.join();
        if (staging && staging->context) destroy(&staging->context, nullptr);
        if (current.context) {
            scheduler.Finish();
            destroy(&current.context, nullptr);
        }
        // The module stays loaded: its threads may outlive the context.
    }

    /// The upscalers the DLLs (and the driver) offer, into the log; BB_FFX_VERSION picks one.
    void ListVersions(const std::filesystem::path& dir) {
        uint64_t count = 0;
        ffxQueryDescGetVersions versions{};
        versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        versions.device = &device_context;
        versions.outputCount = &count;
        if (query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0) {
            std::printf("Upscaler: AMD FSR DLLs (%s) offer no upscaler for this GPU\n",
                        dir.string().c_str());
            return;
        }
        std::vector<uint64_t> ids(count);
        std::vector<const char*> names(count);
        versions.versionIds = ids.data();
        versions.versionNames = names.data();
        query(nullptr, &versions.header);
        const char* wanted = bb_env("BB_FFX_VERSION");
        std::string list;
        for (uint64_t i = 0; i < count; ++i) {
            const char* name = names[i] ? names[i] : "?";
            list += (i ? ", " : "") + std::string{name};
            if (wanted && wanted[0] && std::strstr(name, wanted) && !override_on) {
                override_version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
                override_version.versionId = ids[i];
                override_on = true;
                std::printf("Upscaler: AMD FSR version %s chosen (BB_FFX_VERSION)\n", name);
            }
        }
        std::printf("Upscaler: AMD FSR DLLs from %s: %s\n", dir.string().c_str(), list.c_str());
    }

    /// Creates a context (any thread); the error is kept in `c`.
    void Create(Context& c) {
        ffxCreateContextDescUpscale desc{};
        desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        desc.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
                     (c.auto_exposure ? FFX_UPSCALE_ENABLE_AUTO_EXPOSURE : 0u);
        // Any preset's render size fits: the context is not recreated for a preset change.
        desc.maxRenderSize = {c.width, c.height};
        desc.maxUpscaleSize = {c.width, c.height};
        desc.fpMessage = Message;
        desc.header.pNext = &backend.header;
        backend.header.pNext = override_on ? &override_version.header : nullptr;
        const auto start = std::chrono::steady_clock::now();
        if (const auto result = create(&c.context, &desc.header, nullptr);
            result != FFX_API_RETURN_OK) {
            c.context = nullptr;
            c.error = "ffxCreateContext failed (" + std::to_string(result) + ")";
            std::printf("Upscaler: AMD FSR %s\n", c.error.c_str());
            return;
        }
        ffxQueryGetProviderVersion version{};
        version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        if (query(&c.context, &version.header) == FFX_API_RETURN_OK && version.versionName) {
            c.provider = version.versionName;
        }
        if (c.model != BbSettings::Fsr4Auto) {
            c.provider += std::string{" "} + BbSettings::Fsr4ModelName(c.model);
        }
        std::printf("Upscaler: AMD FSR %s, output %ux%u, context in %.0f ms\n", c.provider.c_str(),
                    c.width, c.height,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                        .count());
    }

    /// The current context goes; `recording`: inside a pass (its earlier frames are submitted).
    void DestroyCurrent(bool recording) {
        if (!current.context) return;
        if (recording) {
            scheduler.WaitSubmittedWork();
        } else {
            scheduler.Finish();
        }
        destroy(&current.context, nullptr);
        current = {};
        ticks.clear();
    }

    void Install(Context&& c) {
        static std::string provider; // read by the menu
        provider = c.provider.empty() ? "AMD FSR" : "AMD " + c.provider;
        BbSettings::Get().fsr_amd_provider.store(provider.c_str());
        current = std::move(c);
    }

    /// The menu's model if the DLLs can run it, else auto.
    static int WantedModel() {
        const auto& s = BbSettings::Get();
        const int model = s.fsr4_model.load(std::memory_order_relaxed);
        return s.fsr_amd_models.load(std::memory_order_relaxed) >> model & 1u ? model
                                                                              : BbSettings::Fsr4Auto;
    }

    [[nodiscard]] bool Matches(const Context& c, u32 w, u32 h, bool ae) const {
        return c.width == w && c.height == h && c.auto_exposure == ae && c.model == WantedModel();
    }

    bool Prepare(u32 w, u32 h, bool ae) {
        if (!module || (current.context && Matches(current, w, h, ae))) return true;
        if (staging_thread.joinable()) {
            if (!staging_done.load(std::memory_order_acquire)) return false;
            staging_thread.join();
            if (Matches(*staging, w, h, ae)) {
                if (!staging->context) {
                    current.error = staging->error; // Record() reports it
                    staging.reset();
                    return true;
                }
                DestroyCurrent(false);
                Install(std::move(*staging));
                staging.reset();
                return true;
            }
            if (staging->context) destroy(&staging->context, nullptr); // never recorded with
            staging.reset();
        }
        staging = std::make_unique<Context>();
        staging->width = w;
        staging->height = h;
        staging->auto_exposure = ae;
        staging->model = WantedModel();
        ApplyModel(staging->model); // the current context's model is chosen already
        staging_done.store(false, std::memory_order_relaxed);
        staging_thread = std::thread([this] {
            Create(*staging);
            staging_done.store(true, std::memory_order_release);
        });
        std::printf("Upscaler: AMD FSR context %ux%u being created in the background (FSR 3.1 "
                    "meanwhile)\n", w, h);
        return false;
    }

    bool Record(const Fsr4Upscaler::Frame& f, std::string& problem, bool& fatal) {
        if (!module) {
            problem = "AMD's FSR DLLs are not installed";
            fatal = true;
            return false;
        }
        const u32 ow = f.output.width, oh = f.output.height;
        if (!current.context || !Matches(current, ow, oh, f.auto_exposure)) {
            // Prepare() makes it in the background; here only when it was not called first.
            DestroyCurrent(true);
            Context c;
            c.width = ow;
            c.height = oh;
            c.auto_exposure = f.auto_exposure;
            c.model = WantedModel();
            ApplyModel(c.model);
            Create(c);
            if (!c.context) {
                problem = c.error;
                fatal = true;
                return false;
            }
            Install(std::move(c));
        }
        while (ticks.size() >= FramesInFlight) {
            scheduler.Wait(ticks.front());
            ticks.pop_front();
        }
        constexpr uint32_t read = FFX_API_RESOURCE_STATE_GENERIC_READ; // General layout
        ffxDispatchDescUpscale d{};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = reinterpret_cast<void*>(VkCommandBuffer(f.cmdbuf));
        d.color = Resource(f.color, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
        d.depth = Resource(f.depth, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
        d.motionVectors = Resource(f.motion, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
        d.output = Resource(f.output, FFX_API_RESOURCE_USAGE_UAV,
                            FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = {f.jitter[0], f.jitter[1]};
        // Vectors are in render pixels (previous - current).
        d.motionVectorScale = {1.0f, BbToggle::Disabled(BbToggle::Fsr4MotionYFlip) ? -1.0f : 1.0f};
        d.renderSize = {f.render_width, f.render_height};
        d.upscaleSize = {ow, oh};
        d.enableSharpening = f.sharpen && f.sharpness > 0.0f;
        d.sharpness = f.sharpness;
        d.frameTimeDelta = f.frame_ms;
        d.preExposure = 1.0f;
        d.reset = f.reset;
        d.cameraNear = f.near_plane;
        d.cameraFar = f.far_plane;
        d.cameraFovAngleVertical = f.vertical_fov;
        d.viewSpaceToMetersFactor = 1.0f;
        if (const auto result = dispatch(&current.context, &d.header);
            result != FFX_API_RETURN_OK) {
            problem = "ffxDispatch failed (" + std::to_string(result) + ")";
            return false;
        }
        ticks.push_back(scheduler.CurrentTick());
        return true;
    }

    void Reset() {
        DestroyCurrent(false);
    }
};

FsrAmd::FsrAmd(const Instance& instance, Scheduler& scheduler)
    : impl{std::make_unique<Impl>(instance, scheduler)} {}

FsrAmd::~FsrAmd() = default;

bool FsrAmd::Available() const noexcept {
    return impl->module != nullptr;
}

bool FsrAmd::Prepare(u32 output_width, u32 output_height, bool auto_exposure) {
    return impl->Prepare(output_width, output_height, auto_exposure);
}

bool FsrAmd::Record(const Fsr4Upscaler::Frame& frame, std::string& problem, bool& fatal) {
    return impl->Record(frame, problem, fatal);
}

void FsrAmd::Reset() {
    impl->Reset();
}

namespace {

constexpr wchar_t FrameGenDllName[] = L"amd_fidelityfx_framegeneration_vk.dll";
/// The frame generation providers, as BB_FFX_FG_ML forces them: ML (4.0.x), FSR 3.1.
constexpr const char* FgProviderTypes[] = {
    ".?AVffxProvider_MLFrameGeneration@@",
    ".?AVffxProvider_Fsr3FrameGeneration@@",
};

} // namespace

struct FgAmd::Impl {
    const Instance& instance;
    Scheduler& scheduler;
    HMODULE module = nullptr;
    PfnFfxCreateContext create = nullptr;
    PfnFfxDestroyContext destroy = nullptr;
    PfnFfxQuery query = nullptr;
    PfnFfxDispatch dispatch = nullptr;
    PfnFfxConfigure configure = nullptr;
    VkDeviceContext device_context{};
    ffxCreateBackendVKDesc backend{};
    ffxCreateContextDescFrameGenerationVersion api_version{};
    ffxOverrideVersion override_version{};
    bool override_on = false;
    u32 debug_flags = 0; ///< BB_FFX_FG_DEBUG: AMD's tear lines and reset marks

    ffxContext context = nullptr;
    u32 render_w = 0, render_h = 0, width = 0, height = 0;
    vk::Format format{};
    std::deque<u64> ticks;

    Impl(const Instance& instance_, Scheduler& scheduler_)
        : instance{instance_}, scheduler{scheduler_} {
        const auto dir = FindDirectory();
        std::error_code error;
        if (dir.empty() || !std::filesystem::is_regular_file(dir / FrameGenDllName, error)) {
            std::printf("Frame generation: AMD's amd_fidelityfx_framegeneration_vk.dll not found "
                        "(BB_FFX_DIR, <user>/fsr, beside the game)\n");
            return;
        }
        module = LoadLibraryExW((dir / LoaderName).c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!module) {
            std::printf("Frame generation: loading %s failed (%lu)\n",
                        (dir / LoaderName).string().c_str(), GetLastError());
            return;
        }
        create = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(module, "ffxCreateContext"));
        destroy = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(module, "ffxDestroyContext"));
        query = reinterpret_cast<PfnFfxQuery>(GetProcAddress(module, "ffxQuery"));
        dispatch = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(module, "ffxDispatch"));
        configure = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(module, "ffxConfigure"));
        if (!create || !destroy || !query || !dispatch || !configure) {
            std::printf("Frame generation: %s lacks the FSR API\n",
                        (dir / LoaderName).string().c_str());
            FreeLibrary(module);
            module = nullptr;
            return;
        }
        device_context.vkDevice = VkDevice(instance.GetDevice());
        device_context.vkPhysicalDevice = VkPhysicalDevice(instance.GetPhysicalDevice());
        device_context.vkDeviceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.deviceContext = device_context;
        api_version.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
        api_version.version = FFX_FRAMEGENERATION_VERSION;
        if (bb_env_on("BB_FFX_FG_DEBUG")) {
            debug_flags = FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_TEAR_LINES |
                          FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_RESET_INDICATORS;
        }
        ListVersions(dir);
        ForceProvider();
    }

    ~Impl() {
        Destroy(); // the module stays loaded: its threads may outlive the context
    }

    /// The frame generations the DLLs (and the driver) offer, into the log; BB_FFX_FG_VERSION
    /// picks one by a part of its name.
    void ListVersions(const std::filesystem::path& dir) {
        uint64_t count = 0;
        ffxQueryDescGetVersions versions{};
        versions.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
        versions.createDescType = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        versions.device = &device_context;
        versions.outputCount = &count;
        if (query(nullptr, &versions.header) != FFX_API_RETURN_OK || count == 0) {
            std::printf("Frame generation: AMD's DLLs (%s) offer none for this GPU\n",
                        dir.string().c_str());
            return;
        }
        std::vector<uint64_t> ids(count);
        std::vector<const char*> names(count);
        versions.versionIds = ids.data();
        versions.versionNames = names.data();
        query(nullptr, &versions.header);
        const char* wanted = bb_env("BB_FFX_FG_VERSION");
        std::string list;
        for (uint64_t i = 0; i < count; ++i) {
            const char* name = names[i] ? names[i] : "?";
            list += (i ? ", " : "") + std::string{name};
            if (wanted && wanted[0] && std::strstr(name, wanted) && !override_on) {
                override_version.header.type = FFX_API_DESC_TYPE_OVERRIDE_VERSION;
                override_version.versionId = ids[i];
                override_on = true;
                std::printf("Frame generation: AMD %s chosen (BB_FFX_FG_VERSION)\n", name);
            }
        }
        std::printf("Frame generation: AMD's DLLs from %s: %s\n", dir.string().c_str(),
                    list.c_str());
    }

    /// BB_FFX_FG_ML: =1 the ML provider's device check says yes (RDNA3 and older: AMD ships it
    /// for RDNA4 only), =0 no; in the DLL and in the driver's amdvkffx64.dll.
    void ForceProvider() {
        const char* force = bb_env("BB_FFX_FG_ML");
        if (!force || !force[0]) return;
        const bool ml = force[0] != '0';
        for (const wchar_t* name : {FrameGenDllName, L"amdvkffx64.dll"}) {
            const HMODULE m = GetModuleHandleW(name);
            if (!m) continue;
            for (int i = 0; i < int(std::size(FgProviderTypes)); ++i) {
                void** const slot = DeviceCheck(m, FgProviderTypes[i]);
                if (!slot) continue;
                void* const want = (i == 0) == ml ? reinterpret_cast<void*>(&CheckYes)
                                                  : reinterpret_cast<void*>(&CheckNo);
                DWORD old = 0;
                if (VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old)) {
                    *slot = want;
                    VirtualProtect(slot, sizeof(void*), old, &old);
                }
                std::printf("Frame generation: %ls %s: device check forced to %s (BB_FFX_FG_ML)\n",
                            name, FgProviderTypes[i], want == &CheckYes ? "yes" : "no");
            }
        }
    }

    void Destroy() {
        if (!context) return;
        scheduler.Finish();
        destroy(&context, nullptr);
        context = nullptr;
        ticks.clear();
    }

    bool Ensure(u32 rw, u32 rh, u32 w, u32 h, vk::Format f, bool& created, std::string& problem) {
        created = false;
        if (!module) {
            problem = "AMD's frame generation DLL is not installed";
            return false;
        }
        if (context && render_w == rw && render_h == rh && width == w && height == h && format == f) {
            return true;
        }
        Destroy();
        ffxCreateContextDescFrameGeneration desc{};
        desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        desc.flags = 0; // non-inverted finite depth, LDR colour (as FSR 3.1's)
        desc.displaySize = {w, h};
        desc.maxRenderSize = {rw, rh};
        desc.backBufferFormat = ffxApiGetSurfaceFormatVK(VkFormat(f));
        desc.header.pNext = &api_version.header;
        api_version.header.pNext = &backend.header;
        backend.header.pNext = override_on ? &override_version.header : nullptr;
        const auto start = std::chrono::steady_clock::now();
        if (const auto result = create(&context, &desc.header, nullptr);
            result != FFX_API_RETURN_OK) {
            context = nullptr;
            problem = "AMD's ffxCreateContext failed (" + std::to_string(result) + ")";
            return false;
        }
        static std::string provider; // read by the menu
        provider = "AMD";
        ffxQueryGetProviderVersion version{};
        version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        if (query(&context, &version.header) == FFX_API_RETURN_OK && version.versionName) {
            provider += std::string{" "} + version.versionName;
        }
        BbSettings::Get().fg_amd_provider.store(provider.c_str());
        std::printf("Frame generation: %s %ux%u from %ux%u scenes (%s), context in %.0f ms\n",
                    provider.c_str(), w, h, rw, rh, vk::to_string(f).c_str(),
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                        .count());
        render_w = rw;
        render_h = rh;
        width = w;
        height = h;
        format = f;
        created = true;
        return true;
    }

    bool Record(const Frame& f, std::string& problem) {
        if (!context) {
            problem = "no context";
            return false;
        }
        while (ticks.size() >= FramesInFlight) {
            scheduler.Wait(ticks.front());
            ticks.pop_front();
        }
        constexpr uint32_t read = FFX_API_RESOURCE_STATE_GENERIC_READ; // General layout
        const uint32_t flags = FFX_FRAMEGENERATION_FLAG_NO_SWAPCHAIN_CONTEXT_NOTIFY | debug_flags;
        const FfxApiRect2D rect{0, 0, int32_t(width), int32_t(height)};
        void* const cmdbuf = reinterpret_cast<void*>(VkCommandBuffer(f.cmdbuf));

        // The order 4.0.x requires: configure, prepare, generate.
        ffxConfigureDescFrameGeneration config{};
        config.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        config.frameGenerationEnabled = true;
        config.allowAsyncWorkloads = false;
        if (f.hudless.image) {
            config.HUDLessColor = Resource(f.hudless, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
        }
        config.flags = flags;
        config.generationRect = rect;
        config.frameID = f.frame_id;
        if (const auto result = configure(&context, &config.header); result != FFX_API_RETURN_OK) {
            problem = "ffxConfigure failed (" + std::to_string(result) + ")";
            return false;
        }

        ffxDispatchDescFrameGenerationPrepareV2 prepare{};
        prepare.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
        prepare.frameID = f.frame_id;
        prepare.flags = flags;
        prepare.commandList = cmdbuf;
        prepare.renderSize = {f.depth.width, f.depth.height};
        prepare.jitterOffset = {f.jitter[0], f.jitter[1]};
        prepare.motionVectorScale = {1.0f, 1.0f}; // render pixels (previous - current)
        prepare.frameTimeDelta = f.frame_ms;
        prepare.reset = f.reset;
        prepare.cameraNear = f.near_plane;
        prepare.cameraFar = f.far_plane;
        prepare.cameraFovAngleVertical = f.vertical_fov;
        prepare.viewSpaceToMetersFactor = 1.0f;
        prepare.depth = Resource(f.depth, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
        prepare.motionVectors = Resource(f.motion, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
        std::memcpy(prepare.cameraPosition, f.camera_position, sizeof(prepare.cameraPosition));
        std::memcpy(prepare.cameraUp, f.camera_up, sizeof(prepare.cameraUp));
        std::memcpy(prepare.cameraRight, f.camera_right, sizeof(prepare.cameraRight));
        std::memcpy(prepare.cameraForward, f.camera_forward, sizeof(prepare.cameraForward));
        if (const auto result = dispatch(&context, &prepare.header); result != FFX_API_RETURN_OK) {
            problem = "prepare failed (" + std::to_string(result) + ")";
            return false;
        }

        ffxDispatchDescFrameGeneration generate{};
        generate.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION;
        generate.commandList = cmdbuf;
        generate.presentColor = Resource(f.color, FFX_API_RESOURCE_USAGE_READ_ONLY, read);
        generate.outputs[0] = Resource(f.output, FFX_API_RESOURCE_USAGE_UAV,
                                       FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        generate.numGeneratedFrames = 1;
        generate.reset = f.reset;
        generate.backbufferTransferFunction = FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
        generate.minMaxLuminance[0] = 0.0f;
        generate.minMaxLuminance[1] = 1.0f;
        generate.generationRect = rect;
        generate.frameID = f.frame_id;
        if (const auto result = dispatch(&context, &generate.header); result != FFX_API_RETURN_OK) {
            problem = "generation failed (" + std::to_string(result) + ")";
            return false;
        }
        ticks.push_back(scheduler.CurrentTick());
        return true;
    }
};

FgAmd::FgAmd(const Instance& instance, Scheduler& scheduler)
    : impl{std::make_unique<Impl>(instance, scheduler)} {}

FgAmd::~FgAmd() = default;

bool FgAmd::Available() const noexcept {
    return impl->module != nullptr;
}

bool FgAmd::Ready() const noexcept {
    return impl->context != nullptr;
}

bool FgAmd::Ensure(u32 render_width, u32 render_height, u32 width, u32 height, vk::Format format,
                   bool& created, std::string& problem) {
    return impl->Ensure(render_width, render_height, width, height, format, created, problem);
}

void FgAmd::Destroy() {
    impl->Destroy();
}

bool FgAmd::Record(const Frame& frame, std::string& problem) {
    return impl->Record(frame, problem);
}

} // namespace Vulkan

#else // AMD's DLLs are Windows libraries.

namespace Vulkan {

struct FsrAmd::Impl {};

FsrAmd::FsrAmd(const Instance&, Scheduler&) {}
FsrAmd::~FsrAmd() = default;
bool FsrAmd::Available() const noexcept {
    return false;
}
bool FsrAmd::Prepare(u32, u32, bool) {
    return true;
}
bool FsrAmd::Record(const Fsr4Upscaler::Frame&, std::string& problem, bool& fatal) {
    problem = "AMD's FSR DLLs need Windows";
    fatal = true;
    return false;
}
void FsrAmd::Reset() {}

struct FgAmd::Impl {};

FgAmd::FgAmd(const Instance&, Scheduler&) {}
FgAmd::~FgAmd() = default;
bool FgAmd::Available() const noexcept {
    return false;
}
bool FgAmd::Ready() const noexcept {
    return false;
}
bool FgAmd::Ensure(u32, u32, u32, u32, vk::Format, bool& created, std::string& problem) {
    created = false;
    problem = "AMD's frame generation DLL needs Windows";
    return false;
}
void FgAmd::Destroy() {}
bool FgAmd::Record(const Frame&, std::string& problem) {
    problem = "AMD's frame generation DLL needs Windows";
    return false;
}

} // namespace Vulkan

#endif
