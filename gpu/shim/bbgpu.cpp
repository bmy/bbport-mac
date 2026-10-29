// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_threads.h"
#include "bbport_affinity.h"
#include "shim/bb_env.h"
#include "bbport_write_log.h"
#include "bbport_frame_rate.h"
#include "bbport_game_menu.h"
#include "bbport_gnm_hooks.h"
// bbport: glue between the C loader and the vendored shadPS4 video core.
#include "bbport_overlay.h"
#include "bbport_guest_hooks.h"
#include "bbport_settings.h"
#include "game_profile.h"
#include "bbport_copy.h"
#include <sys/resource.h>
#include "bbport_free_check.h"
#include "bbport_toggles.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include "common/polyfill_thread.h"
#include <cstring>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <vector>
#include <SDL3/SDL.h>
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif
#include "../bbgpu.h"
#include "platform/kernel/event_queue.h"
#include "platform/videoout/videoout.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "platform/errors.h"
#include "core/libraries/libs.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/signals.h"
#include "sdl_window.h"
#include "bbport_portable.h"
#include "shader/ir/passes/srt.h"
#include "render/rasterizer.h"
#include "remote/bb_remote.h"
#include "games/game_profile.h"

extern "C" {
void bbgpu_read_sample_start(); // memory/read_sample.cpp (Linux; BB_READ_SAMPLE)
// runtime_memory.c
int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
int runtime_memory_write_backing(uintptr_t address, const void* data, uint64_t size);
void runtime_memory_read_backing(uintptr_t address, void* data, uint64_t size);
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size);
int runtime_memory_region(uintptr_t address, uintptr_t* start, uintptr_t* end, int* mapped);
const uint64_t* runtime_memory_generation(void);
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write);
typedef void (*RuntimeGpuRange)(uintptr_t address, uint64_t size);
// runtime_kernel.c: one clock for guest and GPU timestamps
uint64_t runtime_process_time_us(void);
uint64_t runtime_process_time_counter(void);
uint64_t runtime_tsc_frequency(void);
int32_t* runtime_errno(void);
void runtime_memory_set_gpu_hooks(RuntimeGpuRange map, RuntimeGpuRange unmap, RuntimeGpuRange invalidate);
void runtime_memory_set_note_write_hook(RuntimeGpuRange note);
void runtime_memory_set_cpu_write_hook(RuntimeGpuRange hook);
void runtime_memory_set_host_only_allocations(uint64_t mask);
}

Frontend::WindowSDL* g_window = nullptr;

namespace BbGnm { void RegisterLib(Core::Loader::SymbolsResolver* sym); }
namespace BbAvPlayer { void RegisterLib(Core::Loader::SymbolsResolver* sym); }

namespace {
struct Symbol {
    std::string nid, library, module;
    u64 address;
};
std::vector<Symbol> g_symbols;
u32 g_sdk_version;
} // namespace

namespace Core::Loader {
void SymbolsResolver::AddSymbol(const char* nid, const char* library, const char* module, SymbolType,
                                u64 address) {
    g_symbols.push_back({nid, library, module, address});
}
} // namespace Core::Loader

namespace Core {
void MemoryManager::SetRasterizer(Vulkan::Rasterizer* rasterizer_) {
    rasterizer = rasterizer_;
    // Existing guest mappings are replayed by the runtime when hooks are installed.
    runtime_memory_set_gpu_hooks(
        [](uintptr_t address, uint64_t size) {
            Memory::Instance()->GetRasterizer()->MapMemory(address, size);
        },
        [](uintptr_t address, uint64_t size) { Memory::Instance()->GetRasterizer()->UnmapMemory(address, size); },
        [](uintptr_t address, uint64_t size) {
            Memory::Instance()->GetRasterizer()->InvalidateMemory(address, size);
        });
    // bbport: data written into GPU memory by a path the GPU side hears of (file reads, the game's
    // resource loaders): invalidated, and an asset (it may keep a VRAM copy).
    runtime_memory_set_note_write_hook([](uintptr_t address, uint64_t size) {
        Memory::Instance()->GetRasterizer()->NoteAssetWrite(address, size);
    });
    // bbport: the game's libc copies (memcpy, memset, memmove) over pages the caches watch.
    runtime_memory_set_cpu_write_hook([](uintptr_t address, uint64_t size) {
        Memory::Instance()->GetRasterizer()->OnCpuWrite(address, size);
    });
}
void MemoryManager::InvalidateMemory(VAddr address, u64 size) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::InvalidateMemory(address, size); // bb-gpu has the caches
        return;
    }
    if (rasterizer) rasterizer->InvalidateMemory(address, size);
}
namespace {
// bbport: the mapped region of the last lookup, per thread, valid for the mapping table generation
// it was read at. The GPU command thread clamps and copies thousands of constant ranges a frame,
// nearly all within the region of the one before: two calls into the runtime each (~5% of it).
struct RegionCache {
    u64 generation = 1; // odd: invalid
    uintptr_t start = 0, end = 0;
};
thread_local RegionCache region_cache;

bool CachedMapped(VAddr address, u64 size) {
    static const uint64_t* const generation_ptr = runtime_memory_generation();
    const u64 generation = __atomic_load_n(generation_ptr, __ATOMIC_ACQUIRE);
    if (generation & 1) {
        return false; // being changed
    }
    auto& cache = region_cache;
    if (cache.generation == generation && address >= cache.start && address + size <= cache.end) {
        return true;
    }
    uintptr_t start = 0, end = 0;
    int mapped = 0;
    if (!runtime_memory_region(address, &start, &end, &mapped) || !mapped) {
        return false;
    }
    cache = {generation, start, end};
    return address + size <= end;
}
} // namespace

u64 MemoryManager::ClampRangeSize(VAddr virtual_addr, u64 size) {
    if (size && CachedMapped(virtual_addr, size)) {
        return size;
    }
    return runtime_memory_clamp(virtual_addr, size);
}
static void CopySparseSerial(VAddr source, u8* dest, u64 size) {
    // BB_READBACKS=2 protects GPU-written pages against reads: the copy threads and recorders
    // read through the backing view, as a read fault on them would wait for the GPU thread,
    // which waits for them.
    static const bool precise =
        EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise;
    if (precise) {
        runtime_memory_read_backing(source, dest, size);
        return;
    }
    if (size && CachedMapped(source, size)) {
        std::memcpy(dest, reinterpret_cast<const void*>(source), size);
        return;
    }
    while (size) {
        uintptr_t start = 0, end = 0;
        int mapped = 0;
        if (!runtime_memory_region(source, &start, &end, &mapped)) {
            end = source + size;
        }
        const u64 n = std::min<u64>(size, end - source);
        if (mapped) std::memcpy(dest, reinterpret_cast<const void*>(source), n);
        else std::memset(dest, 0, n);
        source += n; dest += n; size -= n;
    }
}
void MemoryManager::CopySparseMemory(VAddr source, u8* dest, u64 size) {
    // bbport: large uploads (streaming) are split across the copy threads.
    constexpr u64 Chunk = 512 * 1024;
    if (size < 4 * Chunk || !BbCopy::Enabled()) {
        return CopySparseSerial(source, dest, size);
    }
    BbCopy::ParallelFor((size + Chunk - 1) / Chunk, [&](std::size_t i) {
        const u64 offset = i * Chunk;
        CopySparseSerial(source + offset, dest + offset, std::min(Chunk, size - offset));
    });
}
void MemoryManager::ReadBacking(VAddr address, void* data, u64 size) {
    runtime_memory_read_backing(address, data, size);
}
bool MemoryManager::TryWriteBacking(void* address, const void* data, u64 size) {
    BbWriteLog::Note(reinterpret_cast<uintptr_t>(address), data, size, BbWriteLog::Backing);
    return runtime_memory_write_backing(reinterpret_cast<uintptr_t>(address), data, size) != 0;
}
void AddressSpace::Protect(VAddr virtual_addr, u64 size, MemoryPermission perms) {
    BbStats::Timer timer{BbStats::t_protect};
    const u64 pages = (size + 4095) / 4096;
    BbStats::protect_calls.fetch_add(1, std::memory_order_relaxed);
    BbStats::protect_pages.fetch_add(pages, std::memory_order_relaxed);
    if (!True(perms & MemoryPermission::Write)) {
        BbStats::protect_revoke_calls.fetch_add(1, std::memory_order_relaxed);
        BbStats::protect_revoke_pages.fetch_add(pages, std::memory_order_relaxed);
    }
    runtime_memory_gpu_protect(virtual_addr, size, True(perms & MemoryPermission::Read),
                               True(perms & MemoryPermission::Write));
}
boost::icl::interval_set<VAddr> AddressSpace::GetUsableRegions() {
    boost::icl::interval_set<VAddr> set;
    set += boost::icl::interval<VAddr>::right_open(0x10'0000'0000ULL, 0xfc'0000'0000ULL);
    return set;
}
} // namespace Core

// Kernel services the vendored libraries call directly.
namespace Libraries::Kernel {
u64 PS4_SYSV_ABI sceKernelGetTscFrequency() { return runtime_tsc_frequency(); }
u64 PS4_SYSV_ABI sceKernelReadTsc() {
#if defined(__aarch64__)
    return BbRemote::Back::ReadTsc(); // bb-gpu: the game process's TSC (its timestamps)
#else
    return Common::FencedRDTSC();
#endif
}
u64 PS4_SYSV_ABI sceKernelGetProcessTime() { return runtime_process_time_us(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounter() { return runtime_process_time_counter(); }
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounterFrequency() { return 1'000'000'000; }
s32 PS4_SYSV_ABI sceKernelIsNeoMode() { return 0; } // base PS4
s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* ver) {
    if (!ver) return BbErr::KernelInvalid;
    *ver = s32(g_sdk_version);
    return BbErr::Ok;
}
s32 PS4_SYSV_ABI sceKernelUsleep(u32 microseconds) {
    std::this_thread::sleep_for(std::chrono::microseconds(microseconds));
    return BbErr::Ok;
}
} // namespace Libraries::Kernel

namespace {
// SDL video must be driven from one thread: the window lives on its own host thread.
std::thread g_window_thread;
std::mutex g_window_mutex;
std::condition_variable g_window_cv;
bool g_window_ready;
} // namespace

#if defined(BB_PGO_GENERATE) && defined(BB_PGO_CLANG)
extern "C" int __llvm_profile_write_file(void);
// Clang's instrumented build (Windows): the counts so far rewritten every 30 s (one file, not
// summed: no reset), as the game may end through _exit.
static void StartProfileWriter() {
    std::thread([] {
        Common::SetCurrentThreadName("bb:pgo");
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            __llvm_profile_write_file();
            std::printf("PGO: profile written\n");
        }
    }).detach();
}
#elif defined(BB_PGO_GENERATE)
extern "C" void __gcov_dump(void);
extern "C" void __gcov_reset(void);
// Instrumented build (BB_PGO=generate): the game often ends through _exit (watchdog, guest
// exit), which skips the profile write at exit. Write every 30 s and reset: the files sum the
// intervals.
static void StartProfileWriter() {
    std::thread([] {
        Common::SetCurrentThreadName("bb:pgo");
        for (;;) {
            std::this_thread::sleep_for(std::chrono::seconds(30));
            __gcov_dump();
            __gcov_reset();
            std::printf("PGO: profile written\n");
        }
    }).detach();
}
#endif

#ifdef __APPLE__
namespace {
std::mutex g_main_mutex;
std::condition_variable g_main_cv;
std::function<void()> g_main_task;
} // namespace

extern "C" void bbgpu_main_thread_loop(void) {
    for (;;) {
        std::function<void()> task;
        {
            std::unique_lock lock{g_main_mutex};
            g_main_cv.wait(lock, [] { return static_cast<bool>(g_main_task); });
            task = std::move(g_main_task);
            g_main_task = nullptr;
        }
        task();
    }
}

/// Runs `task` on the process's main thread (parked in bbgpu_main_thread_loop).
static void RunOnMainThread(std::function<void()> task) {
    {
        std::scoped_lock lock{g_main_mutex};
        g_main_task = std::move(task);
    }
    g_main_cv.notify_all();
}
#endif

extern "C" int bbgpu_init(const BbGpuConfig* config) {
    BbThreads::guest_main_tid.store(BbThreads::HostTid(), std::memory_order_relaxed);
    BbSettings::Load();
    BbAffinity::Start();
    // What the translator knows about this game (games/), before anything asks for it.
    Game::Select(config->serial);
    // Its direct memory allocations the GPU never uses stay host memory (BB_HOST_ONLY_DEFAULT=0: none).
    if (const Game::Profile* game = Game::Active()) {
        if (!bb_env_off("BB_HOST_ONLY_DEFAULT") && game->host_only_allocations) {
            runtime_memory_set_host_only_allocations(game->host_only_allocations);
        }
    }
#if !defined(_WIN32) && !defined(__APPLE__) // bbport (macOS): Linux's signal context
    bbgpu_read_sample_start(); // BB_READ_SAMPLE (diagnostics)
#endif
#ifdef BB_PGO_GENERATE
    StartProfileWriter();
#endif
    g_sdk_version = config->sdk_version;
#ifndef ARCH_X86_64
    // bbport (the macOS native GPU process): the bytecode SRT walker checks the guest pointers
    // it follows: there, guest memory and the game image's data only (remote_back.cpp). Elsewhere
    // tables can also sit in the game image's own data (not a runtime mapping): probed then.
    Shader::srt_guest_readable = BbRemote::BackActive() ? &BbRemote::Back::GuestReadable
                                                        : +[](u64 address, u64 size) {
        if (Core::CachedMapped(address, size)) {
            return true;
        }
        u8 probe[8];
        return size <= sizeof(probe) &&
               BbPortable::ReadSelf(probe, reinterpret_cast<const void*>(address), size);
    };
#endif
    if (config->user_dir) setenv("BB_GPU_USER_DIR", config->user_dir, 0);
    Common::ElfInfo::Instance().Set(config->serial ? config->serial : "UNKNOWN",
                                    config->title ? config->title : "", config->sdk_version,
                                    config->psf_attributes);
    // bbport (native GPU process, docs/macos-native-gpu.md): the window, the Vulkan device and the
    // GPU core start in bb-gpu; here the HLE libraries register and forward to it.
    if (BbRemote::Requested()) {
        if (!BbRemote::Front::Init(*config)) {
            std::fprintf(stderr, "GPU process: could not start; BB_NATIVE_GPU=0 (the app's "
                                 "\"In-process GPU\") runs the game without it\n");
            return 1;
        }
        // The game's System menu pages (this process) offer MetalFX: bb-gpu checks the device
        // and falls back as it does for its own menu.
        BbSettings::Get().metalfx_supported = true;
        // The game's System menu saves only its own keys; bb-gpu's overlay owns the rest.
        BbSettings::save_game_menu_keys_only = true;
        // The hooks in the game's own code (fixes, live effects, aspect ratio): the game's code
        // is in this process; in one process the buffer cache installs them, here bb-gpu doesn't.
        // Their GPU ranges (fresh, dead, live) go to bb-gpu's buffer cache.
        if (const Game::Profile* game = Game::Active(); game && game->install_hooks) {
            if (bb_env_off("BB_GUEST_HOOKS")) {
                std::printf("Guest hooks: off (BB_GUEST_HOOKS=0): nothing in the game's own code is hooked\n");
            } else {
                game->install_hooks(Game::CoreServices{
                    .fresh_range = &BbRemote::Front::FreshRange,
                    .dead_range = &BbRemote::Front::DeadRange,
                    .live_range = &BbRemote::Front::LiveRange,
                });
            }
        }
#ifdef __APPLE__
        // Gamepads stay here (runtime_pad.c reads them through SDL): this process's main thread,
        // idle without a window, keeps their events and run loop going.
        RunOnMainThread([] {
            Common::SetCurrentThreadName("bb:input");
            SDL_InitSubSystem(SDL_INIT_GAMEPAD);
            for (;;) {
                SDL_PumpEvents();
                SDL_FlushEvents(SDL_EVENT_FIRST, SDL_EVENT_LAST); // nobody reads them here
                CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.004, false);
            }
        });
#endif
        Core::Loader::SymbolsResolver resolver;
        BbGnm::RegisterLib(&resolver);
        BbVideoOut::RegisterLib(&resolver);
        return 0;
    }
    const std::string title = config->title ? config->title : "Bloodborne";
    const s32 width = config->width, height = config->height;
    auto window_main = [title, width, height] {
        Common::SetCurrentThreadName("bb:window");
        auto* window = new Frontend::WindowSDL(width, height, title.c_str());
        {
            std::scoped_lock lock{g_window_mutex};
            g_window = window;
            g_window_ready = true;
        }
        g_window_cv.notify_all();
        while (window->PollEvents()) {
            if (BbRemote::BackActive()) {
                BbRemote::Back::PublishWindowState();
            }
            SDL_Delay(2);
        }
        LOG_INFO(Frontend, "Window closed by user");
        std::fflush(stdout);
        std::_Exit(0);
    };
#ifdef __APPLE__
    RunOnMainThread(window_main); // Cocoa: the window and its events belong to the main thread
#else
    g_window_thread = std::thread(window_main);
    g_window_thread.detach();
#endif
    {
        std::unique_lock lock{g_window_mutex};
        g_window_cv.wait(lock, [] { return g_window_ready; });
    }
    Core::Loader::SymbolsResolver resolver;
    // GnmDriver creates the presenter that the VideoOut present thread uses.
    BbGnm::RegisterLib(&resolver);
    BbVideoOut::RegisterLib(&resolver);
    return 0;
}

extern "C" void bbgpu_register_kernel(void) {
    Core::Loader::SymbolsResolver resolver;
    BbKernel::RegisterEventQueue(&resolver);
    BbAvPlayer::RegisterLib(&resolver);
}

extern "C" uintptr_t bbgpu_resolve(const char* scoped_nid) {
    const char* hash = std::strchr(scoped_nid, '#');
    const size_t length = hash ? size_t(hash - scoped_nid) : std::strlen(scoped_nid);
    for (const auto& symbol : g_symbols) {
        if (symbol.nid.size() == length && !std::memcmp(symbol.nid.data(), scoped_nid, length)) {
            return uintptr_t(symbol.address);
        }
    }
    return 0;
}

extern "C" int bbgpu_handle_fault(void* ucontext, void* address) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::HandleFault(ucontext, address); // bb-gpu tracks the pages
    }
    // BB_LABEL_TRAP (diagnostic): its read-only label pages first. Not passed on to GPU page
    // tracking (a write fault drains the draw pipe: thousands a second would change the timing
    // under test); label pages are not expected to be GPU-tracked.
    if (BbFreeCheck::OnTrapFault(ucontext, reinterpret_cast<std::uint64_t>(address))) {
        return 1;
    }
    if (Core::Signals::Instance()->DispatchAccessViolation(ucontext, address)) {
        return 1;
    }
    return BbFreeCheck::OnStaleTrapFault(reinterpret_cast<std::uint64_t>(address)) ? 1 : 0;
}

extern "C" void bbgpu_patch_image(unsigned char* image, uint64_t size) {
    BbGnmHooks::PatchImage(image, size);
    BbGameMenu::PatchImage(image, size);
    BbFrameRate::PatchImage(image, size);
}

extern "C" int bbgpu_share_range(void* address, uint64_t size, int prot) {
    return BbRemote::FrontActive() ? BbRemote::Front::ShareRange(address, size, prot) : 0;
}

extern "C" unsigned bbgpu_symbol_count(void) {
    return unsigned(g_symbols.size());
}

namespace Libraries::Kernel {
int* PS4_SYSV_ABI __Error() {
    return runtime_errno();
}
} // namespace Libraries::Kernel

extern "C" int bbgpu_overlay_captures_input(void) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::OverlayCapturesInput();
    }
    return BbOverlay::CapturesInput() ? 1 : 0;
}

extern "C" void bbgpu_mouse_take(double* dx, double* dy, int* wheel_up, int* wheel_down) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::MouseTake(dx, dy, wheel_up, wheel_down); // bb-gpu's window
        return;
    }
    double x = 0, y = 0;
    int up = 0, down = 0;
    if (g_window) {
        g_window->TakeMouse(x, y, up, down);
    }
    *dx = x;
    *dy = y;
    *wheel_up = up;
    *wheel_down = down;
}

/// bbport (macOS port): the keyboard through the GPU library, which owns the window (the native
/// GPU process runs the window in another process).
extern "C" const bool* bbgpu_keyboard_state(void) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::KeyboardState(); // bb-gpu's window
    }
    return SDL_WasInit(SDL_INIT_VIDEO) ? SDL_GetKeyboardState(nullptr) : nullptr;
}

extern "C" void bbgpu_keyboard_take_latched(volatile uint8_t* latch, int count) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::TakeKeyLatch(latch, count); // in-process: runtime_pad.c's own event watch
    }
}

extern "C" int bbgpu_mouse_captured(void) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::MouseCaptured();
    }
    return g_window && g_window->MouseCaptured() ? 1 : 0;
}

extern "C" uint32_t bbgpu_mouse_buttons(void) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::MouseButtons();
    }
    return bbgpu_mouse_captured() && SDL_WasInit(SDL_INIT_VIDEO) ? SDL_GetMouseState(nullptr, nullptr)
                                                                 : 0;
}

extern "C" void bbgpu_mouse_capture(void) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::MouseCapture(); // bb-gpu's window
        return;
    }
    if (g_window) {
        g_window->RequestMouseCapture();
    }
}

extern "C" uint64_t bbgpu_camera_state(void) {
    return BbGuestHooks::CameraState();
}

extern "C" int bbgpu_free_camera(void) {
    return BbGuestHooks::FreeCameraActive() ? 1 : 0;
}

extern "C" int bbgpu_camera_look(float yaw, float pitch) {
    if (!BbGuestHooks::CameraLookAvailable()) {
        return 0;
    }
    return BbGuestHooks::CameraLookAdd(pitch, yaw) ? 1 : 0;
}

extern "C" void bbgpu_mouse_look_enable(int enabled) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::MouseLookEnable(enabled);
        return;
    }
    Frontend::WindowSDL::EnableMouseLook(enabled != 0);
}

extern "C" int bbgpu_text_input_begin(const char* initial, const char* prompt) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::TextInputBegin(initial, prompt);
    }
    if (!g_window) return 0;
    g_window->BeginTextInput(initial ? initial : "", prompt ? prompt : "Text");
    return 1;
}

extern "C" int bbgpu_text_input_poll(char* out, uint64_t size) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::TextInputPoll(out, size);
    }
    if (!g_window) return 2;
    std::string text;
    const int state = g_window->PollTextInput(text);
    if (size) {
        const size_t n = std::min<size_t>(text.size(), size - 1);
        std::memcpy(out, text.data(), n);
        out[n] = 0;
    }
    return state;
}
