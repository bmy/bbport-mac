// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's frame rate set free by hooks in its code (bbport_frame_rate.h).
//
// The game's main loop (0x2018d20) runs its frame pacer (0x2034770: it waits for the frame time
// its mode asks, 1/30 s, and measures the frame at [pacer + 0x264]) and then steps the game by a
// fixed 1/30 s (0x2018e36). The hooks keep the pacer's mode at one vblank and no frame time to
// wait for beyond the shortest frame, and step the game by the measured frame time (1/480 to
// 1/30 s: below 30 FPS the game slows down, as it always did).
//
// What the game ties to a 30 FPS frame besides (found by the frame time each function is called
// with, BB_RE_ENTRY, and by what its code does with the constant): subsystems stepped by a fixed
// 1/30 s, per-frame steps and approaches, per-frame damping. Their instructions read the constant
// from a page next to the image instead (the instruction's own address operand moved; the game's
// code is otherwise as it was), where FrameStep writes the value for this frame's time: a step of
// 1/30 s becomes the frame's time, a per-frame factor f becomes f^(frames) for the frame's length
// in 30 FPS frames. Durations the game converts from 30 FPS frames into seconds stay as they are:
// what counts them down uses the real time.
#include "bbport_map_near.h"
#include "shim/bb_env.h"
#include "common/types.h"
#include "bbport_frame_rate.h"
#include "render/vk_low_latency.h"
#include "bbport_guest_code.h"
#include "remote/bb_remote.h"
#include "bbport_toggles.h"

#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

namespace BbFrameRate {
namespace {
using BbGuestCode::Emitter;
using u64 = std::uint64_t;
using u8 = std::uint8_t;

std::atomic<bool> unlocked{false};
u64 image_base = 0;

constexpr float Frame30 = 1.0f / 30.0f;
/// The highest frame rate the pacer allows ("uncap", and the most BB_FPS_LIMIT can ask for). The
/// game itself breaks above it: on loading screens at 880-1000 FPS its main loop outran its worker
/// threads and a pool allocator popped a freed node (guest fault at +0x263b8e7, 2 of 2 loads);
/// at 480 all loads went through (4 of 4). The cap was 240 before.
constexpr int MaxRate = 480;
constexpr float ShortestFrame = 1.0f / MaxRate;

/// What the redirected instructions read, rewritten by FrameStep for each frame.
struct alignas(64) Live {
    float dt = Frame30;               ///< the frame's time: the game steps by it
    float shortest = ShortestFrame;   ///< the pacer's frame time target
    float pad[2] = {};
    float decay95[4] = {0.95f, 0.95f, 0.95f, 0.95f}; ///< 0.95 per 30 FPS frame (read as four)
    float approach30 = Frame30;       ///< a 1/30 approach per 30 FPS frame
    float approach10 = 0.1f;          ///< a 0.1 approach per 30 FPS frame
    float step_down = -0.1f;          ///< -0.1 per 30 FPS frame
    float step_up = 0.1f;             ///< +0.1 per 30 FPS frame
    float decay08 = 0.8f;             ///< 0.8 per 30 FPS frame
    float cloth = 1.0f;               ///< cloth constraints' stiffness
    float model_turn = 0.0f;          ///< the model preview's turn per 30 FPS frame (radians per stick unit)
};
Live* live = nullptr;
float model_turn_base = 0.0f;         ///< the game's own model_turn (its constant, read at start)
int base_rate = 0;               /// the launched mode's rate (with BB_FPS_LIMIT)
std::atomic<int> live_limit{0};

/// The pacer at the main loop's step (rdi: it): this frame's values.
void PS4_SYSV_ABI FrameStep(u64 pacer) {
    // Low latency (NVIDIA Reflex, AMD Anti-Lag 2): the driver's sleep before the frame's
    // simulation.
    Vulkan::LowLatency::Get().GameFrameStart();
    const float measured = *reinterpret_cast<const float*>(pacer + 0x264);
    float dt = std::clamp(std::isfinite(measured) ? measured : Frame30, live->shortest, Frame30);
    // BB_FRAME_DT_SMOOTH=a (diagnostics, 0 < a < 1): the step follows the measured frame time by
    // that fraction a frame, so frame-to-frame jitter of the measurement does not reach the
    // simulation (time stays right on average).
    static const float smooth = [] {
        const char* env = bb_env("BB_FRAME_DT_SMOOTH");
        const float a = env && *env ? std::strtof(env, nullptr) : 0.0f;
        return a > 0.0f && a < 1.0f ? a : 0.0f;
    }();
    if (smooth > 0.0f) {
        static float smoothed = 0.0f;
        smoothed = smoothed > 0.0f ? smoothed + smooth * (dt - smoothed) : dt;
        dt = std::clamp(smoothed, live->shortest, Frame30);
    }
    const float frames = dt * 30.0f; // the frame's length in 30 FPS frames
    live->dt = dt;
    const float decay95 = std::pow(0.95f, frames);
    for (float& d : live->decay95) d = decay95;
    live->approach30 = 1.0f - std::pow(1.0f - Frame30, frames);
    live->approach10 = 1.0f - std::pow(0.9f, frames);
    live->step_down = -0.1f * frames;
    live->step_up = 0.1f * frames;
    live->decay08 = std::pow(0.8f, frames);
    live->model_turn = model_turn_base * frames;
    // The cloth setups' range checks of Stiffness, Rigid Factor and Allowed Compression (0x80d800
    // only warns) and a few uses of 1.0 next to them: the game's 1.0. The cloth follows the frame
    // rate by its constraints' stiffness factor instead (ClothStiffnessFactor).
    // BB_CLOTH_STIFFNESS=value (diagnostics): another value there.
    static const float cloth_override = [] {
        const char* env = bb_env("BB_CLOTH_STIFFNESS");
        return env && *env ? std::strtof(env, nullptr) : -1.0f;
    }();
    live->cloth = cloth_override > 0.0f ? cloth_override : 1.0f;
}

// ---- Cloth: the constraints' stiffness factor per substep ----
/// The game's exponent of Havok's stiffness adaptation (0x48d7bbc: -1.725), read at start.
float cloth_exponent = -1.725f;
std::atomic<u64> cloth_calls{0}, cloth_modes{0}, cloth_types{0};

/// Havok's stiffness factor of a constraint set for one substep (0x1167a80, replaced): `mode` 2
/// adapts the constraints' stiffness to how often they are solved, (substeps x a x b)^-1.725
/// (a, b the instance's scales at +0x1d0/+0x1d4), for a set of type 8 the iteration's share
/// instead and for types 5 and 10 only the last substep. The substeps are per frame: at 191 FPS
/// the cloth is solved 6.4 times as often as at 30 FPS by the same factor, and stiffer by as much
/// (it flutters). The factor here is the game's x frames (the frame's length in 30 FPS frames):
/// the cloth is pulled as far per second as at 30 FPS, whether or not it adapts to its own
/// substeps. At 30 FPS it is the game's.
/// Experiment bit 65536: the game's factor (A/B).
float PS4_SYSV_ABI ClothStiffnessFactor(const u8* set, int mode, int iteration, int substeps, float a,
                                        float b) {
    const std::uint32_t type = *reinterpret_cast<const std::uint32_t*>(set + 0x18);
    const u64 n = cloth_calls.fetch_add(1, std::memory_order_relaxed);
    cloth_modes.fetch_or(u64(1) << (unsigned(mode) & 63), std::memory_order_relaxed);
    cloth_types.fetch_or(u64(1) << (type & 63), std::memory_order_relaxed);
    if (n == 20000) {
        std::printf("Frame rate: cloth stiffness factor: modes %#llx, constraint types %#llx, "
                    "%.0f FPS: factor x%.3f\n",
                    (unsigned long long)cloth_modes.load(), (unsigned long long)cloth_types.load(),
                    1.0f / live->dt, live->dt * 30.0f);
    }
    const bool adapts = mode == 2 && !(substeps == 1 && a == 1.0f && b == 1.0f);
    if (adapts && type == 8) {
        return float(iteration + 1) / float(substeps);
    }
    if (adapts && (type == 5 || type == 10)) {
        return substeps - 1 == iteration ? 1.0f : 0.0f;
    }
    if (type == 5 || type == 8 || type == 10 || BbToggle::Experiment(65536)) {
        return adapts ? std::pow(float(substeps) * a * b, cloth_exponent) : 1.0f;
    }
    const float frames = live->dt * 30.0f;
    const float game = adapts ? std::pow(float(substeps) * a * b, cloth_exponent) : 1.0f;
    // Experiment bit 131072: the frame rate counts as more substeps by Havok's law too.
    if (BbToggle::Experiment(131072)) {
        return game * std::pow(frames, -cloth_exponent);
    }
    // A constraint solved with stiffness k pulls its particles k of the way per solve: at 30 FPS
    // k a frame, at any frame rate 1 - (1 - k)^frames a frame, ~k x frames (the per-frame factors'
    // rule above). x0.5 at 60 FPS, the community's 60 FPS value.
    return game * frames;
}

// ---- Redirected constants: instruction (its RIP-relative operand), the constant, the value ----
struct Redirect {
    u64 site;
    u64 constant;
    std::size_t value; ///< offsetof(Live, ..)
};
#define LIVE(field) offsetof(Live, field)
constexpr Redirect Redirects[] = {
    // A fixed 1/30 s step or "one frame" (the patches' lists, checked one by one):
    {0xab9f0d, 0x47efdcc, LIVE(dt)},   // a time step, one frame when unset
    {0xdca3ed, 0x4888688, LIVE(dt)},   // a per-frame budget
    {0xee596a, 0x48ad65c, LIVE(dt)},   // effects: a fade of at least one frame (three releases)
    {0xee5a8a, 0x48ad660, LIVE(dt)},
    {0xee5baa, 0x48ad664, LIVE(dt)},
    {0xef06f1, 0x48adc08, LIVE(dt)},   // at least one frame
    {0xef6992, 0x48add28, LIVE(dt)},
    {0x1044943, 0x48cc3f4, LIVE(dt)},  // an update by one frame
    {0x12a4416, 0x492577c, LIVE(dt)},  // a timer advanced by one frame each frame
    {0x15dc1f7, 0x4926c9c, LIVE(dt)},
    {0x173456e, 0x49270e4, LIVE(dt)},  // an update by one frame
    {0x17f9d26, 0x4927554, LIVE(dt)},
    {0x180f647, 0x49275f4, LIVE(dt)},  // a timer advanced by one frame each frame
    {0x18bc4f7, 0x4927c4c, LIVE(dt)},  // an update by one frame (twice)
    {0x18bc50d, 0x4927c4c, LIVE(dt)},
    {0x1dbc308, 0x4928e98, LIVE(dt)},  // frames counted at one frame's time
    {0x1f77d4a, 0x4929170, LIVE(dt)},  // an update by one frame (0 after a reset)
    {0x227784b, 0x4929b80, LIVE(dt)},  // timers advanced by one frame (0xebbda0)
    {0x2277892, 0x4929b80, LIVE(dt)},
    {0x2313dad, 0x4929e70, LIVE(dt)},  // a simulation step: 1/60 in the game's 60 Hz mode,
    {0x2313db7, 0x4929e74, LIVE(dt)},  // 1/30 otherwise
    {0x2315d71, 0x4929e74, LIVE(dt)},
    // Body parts the physics moves (0x1515ec0, each part every frame): their timers advanced by the
    // step they are given (0x180df90 at +0xf0, 0x1811650 at +0xd0), 1/30 s; not in the patches.
    {0x1515fab, 0x4926364, LIVE(dt)},
    {0x1516030, 0x4926364, LIVE(dt)},
    // Steps and approaches per frame:
    {0x19084f5, 0x4927fd4, LIVE(step_down)}, // a value ramped by 0.1 a frame
    {0x1908522, 0x4927fd4, LIVE(step_down)},
    {0x1908518, 0x4927fe0, LIVE(step_up)},
    {0x190854c, 0x4927fe0, LIVE(step_up)},
    {0x143e2b2, 0x4925e94, LIVE(approach30)}, // the follow camera's approaches
    {0x143af8c, 0x4925e68, LIVE(approach10)},
    // Damping per frame:
    {0x1a4abca, 0x490b5b0, LIVE(decay95)}, // a character's motion (0x1a4a730, x0.95 a frame)
    {0x1514c4d, 0x4926334, LIVE(decay08)}, // the run's wall check (x0.8 a frame while blocked)
    // Cloth constraints' stiffness:
    {0x47c6dc, 0x47693f0, LIVE(cloth)},
    {0x7e45be, 0x47b63e4, LIVE(cloth)},
    {0x7e4642, 0x47b63e4, LIVE(cloth)},
    {0x7e6239, 0x47b63e4, LIVE(cloth)},
    {0x7f706d, 0x47b8474, LIVE(cloth)},
    {0x7f70ad, 0x47b8474, LIVE(cloth)},
    {0x7f79e5, 0x47b8474, LIVE(cloth)},
    {0x7f7be8, 0x47b8474, LIVE(cloth)},
    {0x7f7dbd, 0x47b8474, LIVE(cloth)},
    {0x805898, 0x47b91c0, LIVE(cloth)},
    {0x805d18, 0x47b91c0, LIVE(cloth)},
    {0x806269, 0x47b91c0, LIVE(cloth)},
    {0x8062f0, 0x47b91c0, LIVE(cloth)},
    {0x809d62, 0x47b98b8, LIVE(cloth)},
    {0x80a4c9, 0x47b98b8, LIVE(cloth)},
    {0x80a876, 0x47b98b8, LIVE(cloth)},
    // The model preview (0x1db2780: the character creator, the stick turns the model; xmm0/xmm1 the
    // stick): yaw and pitch += stick x the constant, per frame. 7x as fast at 200 FPS (2026-10-27,
    // found with memscan + a watchpoint on the yaw at +0x888).
    {0x1db27b0, 0x4928dfc, LIVE(model_turn)},
};
#undef LIVE

/// [address, address + size) writable (the loader sets the image's protection after
/// bbgpu_patch_image; hooks installed earlier, BbGuestHooks, may have set a page's already).
void Writable(u64 address, std::size_t size) {
    const u64 page = u64(sysconf(_SC_PAGESIZE));
    const u64 start = address & ~(page - 1);
    mprotect(reinterpret_cast<void*>(start), address + size - start, PROT_READ | PROT_WRITE | PROT_EXEC);
}

/// The instruction at `r.site` reads `r.constant` RIP-relative (a VEX instruction, no immediate
/// after the operand): it reads the live value instead.
bool RedirectOperand(const Redirect& r) {
    u8* code = reinterpret_cast<u8*>(image_base + r.site);
    std::size_t disp_at = 0, length = 0;
    if (code[0] == 0xc5 && (code[3] & 0xc7) == 0x05) {
        disp_at = 4, length = 8;
    } else if (code[0] == 0xc4 && (code[4] & 0xc7) == 0x05) {
        disp_at = 5, length = 9;
    } else {
        return false;
    }
    std::int32_t disp = 0;
    std::memcpy(&disp, code + disp_at, 4);
    const u64 next = image_base + r.site + length;
    if (next + std::int64_t(disp) != image_base + r.constant) return false;
    const std::int64_t to = std::int64_t(reinterpret_cast<u64>(live) + r.value) - std::int64_t(next);
    if (to != std::int32_t(to)) return false;
    const std::int32_t to32 = std::int32_t(to);
    Writable(reinterpret_cast<u64>(code + disp_at), 4);
    std::memcpy(code + disp_at, &to32, 4);
    return true;
}

// ---- Detours: the instructions at a site replaced by a jump to a stub ----
u8* stub_page = nullptr;
std::size_t stub_used = 0;
constexpr std::size_t StubPageSize = 4096;

/// `e` into the stub page; a jump to it at `site` over `length` bytes (the rest int3) if the site
/// holds `expected`.
bool Detour(u64 site, const u8* expected, std::size_t length, const Emitter& e) {
    u8* code = reinterpret_cast<u8*>(image_base + site);
    if (length < 5 || std::memcmp(code, expected, length) != 0 || e.overflow ||
        stub_used + e.size > StubPageSize) {
        return false;
    }
    u8* stub = stub_page + stub_used;
    const std::int64_t rel = std::int64_t(reinterpret_cast<u64>(stub)) - std::int64_t(image_base + site + 5);
    if (rel != std::int32_t(rel)) return false;
    std::memcpy(stub, e.code.data(), e.size);
    stub_used += (e.size + 15) & ~std::size_t(15);
    const std::int32_t rel32 = std::int32_t(rel);
    Writable(reinterpret_cast<u64>(code), length);
    code[0] = 0xe9;
    std::memcpy(code + 1, &rel32, 4);
    for (std::size_t i = 5; i < length; ++i) code[i] = 0xcc;
    return true;
}

/// The pacer's mode, after the game chose it (0x203493d, `lea r15, [r12 + 0x268]`; r12 the
/// pacer): one vblank (+0x10), no flip skipping (+0x14), the shortest frame as the frame time
/// target (+0x18), no vblank counts to wait for (+0x268, +0x26c).
bool HookPacerMode() {
    constexpr u8 code[] = {0x4d, 0x8d, 0xbc, 0x24, 0x68, 0x02, 0x00, 0x00};
    Emitter e;
    e.Bytes({0x41, 0xc7, 0x44, 0x24, 0x10, 0x01, 0x00, 0x00, 0x00}); // mov dword [r12 + 0x10], 1
    e.Bytes({0x41, 0xc6, 0x44, 0x24, 0x14, 0x00});                   // mov byte [r12 + 0x14], 0
    e.Bytes({0x50, 0x48, 0xb8});                                       // push rax; movabs rax,
    e.U64(reinterpret_cast<u64>(&live->shortest));                     //   &shortest
    e.Bytes({0x8b, 0x00});                                             // mov eax, [rax]
    e.Bytes({0x41, 0x89, 0x44, 0x24, 0x18, 0x58});                     // mov [r12 + 0x18], eax; pop rax
    e.Bytes({0x49, 0xc7, 0x84, 0x24, 0x68, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}); // mov qword [r12 + 0x268], 0
    for (const u8 b : code) e.Bytes({b});
    e.JmpAbs(image_base + 0x203493d + sizeof(code));
    return Detour(0x203493d, code, sizeof(code), e);
}

/// The pacer keeps its one vblank when a frame came late (0x2034d96: `test dl, dl; je; mov eax,
/// [r12 + 0x10]`, then eax is stored back: 0 when late).
bool HookPacerLate() {
    constexpr u8 code[] = {0x84, 0xd2, 0x74, 0x05, 0x41, 0x8b, 0x44, 0x24, 0x10};
    Emitter e;
    e.Bytes({0x84, 0xd2});                   // test dl, dl
    e.Bytes({0x41, 0x8b, 0x44, 0x24, 0x10}); // mov eax, [r12 + 0x10]
    e.JmpAbs(image_base + 0x2034d9f);
    return Detour(0x2034d96, code, sizeof(code), e);
}

/// The main loop's step (0x2018e36: `mov dword [rbp - 0x38], 1/30`, its time; rbx the pacer):
/// FrameStep, then the frame's time.
bool HookStep() {
    constexpr u8 code[] = {0xc7, 0x45, 0xc8, 0x89, 0x88, 0x08, 0x3d};
    Emitter e;
    e.SaveVolatile();
    e.Bytes({0x48, 0x89, 0xdf}); // mov rdi, rbx
    e.CallAbs(reinterpret_cast<const void*>(&FrameStep));
    e.RestoreVolatile();
    e.StoreToFrame(&live->dt, -0x38);
    e.JmpAbs(image_base + 0x2018e36 + sizeof(code));
    return Detour(0x2018e36, code, sizeof(code), e);
}

/// A subsystem stepped by a fixed 1/30 s (`mov dword [rbp + offset], 1/30` at `site`): by the
/// frame's time.
bool HookFixedStep(u64 site, std::int8_t offset) {
    const u8 code[] = {0xc7, 0x45, u8(offset), 0x89, 0x88, 0x08, 0x3d};
    Emitter e;
    e.StoreToFrame(&live->dt, offset);
    e.JmpAbs(image_base + site + sizeof(code));
    return Detour(site, code, sizeof(code), e);
}

/// The run's wall check (0x1514c24: the distance moved this frame x 30 = a speed at 30 FPS): the
/// distance divided by the frame's time ([rbp - 0x74], the character update's), a speed at any
/// frame rate. Above ~90 FPS the per-frame distance fell under the check's threshold and the run
/// dropped to half speed.
/// 1: done by the hook, 2: the community's Sprint Fix patch already did it, 0: other code.
int HookRunSpeed() {
    constexpr u64 site = 0x1514c24;
    const u8* code = reinterpret_cast<const u8*>(image_base + site);
    constexpr u8 patched[] = {0xc5, 0xfa, 0x5e, 0x45, 0x8c, 0x0f, 0x1f, 0x00}; // vdivss; nop
    if (std::memcmp(code, patched, sizeof(patched)) == 0) {
        return 2;
    }
    std::int32_t disp = 0;
    std::memcpy(&disp, code + 4, 4);
    if (code[0] != 0xc5 || code[1] != 0xfa || code[2] != 0x59 || code[3] != 0x05 ||
        image_base + site + 8 + std::int64_t(disp) != image_base + 0x492632c) {
        return 0;
    }
    u8 expected[8];
    std::memcpy(expected, code, 8);
    Emitter e;
    e.Bytes({0xc5, 0xfa, 0x5e, 0x45, 0x8c}); // vdivss xmm0, xmm0, [rbp - 0x74]
    e.JmpAbs(image_base + site + 8);
    return Detour(site, expected, 8, e) ? 1 : 0;
}

/// Havok's stiffness factor (0x1167a80: `push rbp; mov rbp, rsp; push r15`) by
/// ClothStiffnessFactor (its arguments as they are, a tail call).
bool HookClothStiffness() {
    constexpr u64 site = 0x1167a80;
    constexpr u8 code[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
    std::memcpy(&cloth_exponent, reinterpret_cast<const void*>(image_base + 0x48d7bbc), sizeof(float));
    if (!(cloth_exponent < 0.0f && cloth_exponent > -4.0f)) return false;
    Emitter e;
    e.JmpAbs(reinterpret_cast<u64>(&ClothStiffnessFactor));
    return Detour(site, code, sizeof(code), e);
}

/// A mapping within +-2 GiB of the image (below it), for rel32 jumps and RIP-relative operands.
u8* MapNear(int prot) {
    // The game's code and data: its first 96 MiB (bbport_map_near.h).
    return static_cast<u8*>(BbMapNear::Map(image_base, image_base + (96ull << 20), StubPageSize,
                                           prot, 8ull << 20, "Frame rate"));
}
} // namespace

void PatchImage(unsigned char* image, std::uint64_t) {
    // BB_FPS (run.sh, the launcher's frame rate): uncap (default), 60 or 90 (the same, at most that
    // many frames a second), 30 (the game's own). BB_FRAME_HOOKS=0: the community's frame rate
    // patches instead (scripts/patches.py; comparisons).
    const char* fps = bb_env("BB_FPS");
    const char* hooks = bb_env("BB_FRAME_HOOKS");
    const int mode = !fps || std::strcmp(fps, "uncap") == 0 ? MaxRate : std::atoi(fps);
    if (mode <= 30 || (hooks && hooks[0] == '0')) {
        return;
    }
    image_base = reinterpret_cast<u64>(image);
    u8* data = MapNear(PROT_READ | PROT_WRITE);
    stub_page = MapNear(PROT_READ | PROT_WRITE | PROT_EXEC);
    if (!data || !stub_page) {
        std::printf("Frame rate: no memory next to the game's code: 30 FPS\n");
        return;
    }
    live = new (data) Live{};
    std::memcpy(&model_turn_base, image + 0x4928dfc, sizeof(model_turn_base));
    live->model_turn = model_turn_base;
    // The mode's rate and BB_FPS_LIMIT (the launcher's frame rate limit): the game's pacer waits for
    // it itself, before the frame's simulation (lower latency than holding finished frames back at
    // the present).
    int rate = std::clamp(mode, 30, MaxRate);
    if (const char* limit = bb_env("BB_FPS_LIMIT"); limit && std::atoi(limit) > 0) {
        rate = std::min(rate, std::clamp(std::atoi(limit), 30, MaxRate));
    }
    live->shortest = 1.0f / float(rate);
    base_rate = rate;
    // The core first: without all three the game keeps its own frame rate.
    if (!(HookPacerMode() && HookPacerLate() && HookStep())) {
        std::printf("Frame rate: the game's frame pacer is not the expected code: 30 FPS\n");
        mprotect(stub_page, StubPageSize, PROT_READ | PROT_EXEC);
        return;
    }
    unlocked = true;
    int steps = HookFixedStep(0x17f9b71, -0x48) + HookFixedStep(0x1572a7d, -0x30);
    const int run = HookRunSpeed();
    const bool cloth = HookClothStiffness();
    int redirected = 0;
    for (const Redirect& r : Redirects) {
        if (RedirectOperand(r)) {
            ++redirected;
        } else {
            std::printf("Frame rate: %#llx is not the expected code, left as it is\n", (unsigned long long)r.site);
        }
    }
    mprotect(stub_page, StubPageSize, PROT_READ | PROT_EXEC);
    std::printf("Frame rate: the model preview turns %.5f rad per stick unit per 30 FPS frame\n",
                model_turn_base);
    std::printf("Frame rate: free (hooks, no patches): the game steps by each frame's time "
                "(1/%.0f..1/30 s); %d fixed steps, %d of %zu per-frame values follow the frame's "
                "time, the run's wall check %s, the cloth's stiffness %s\n",
                1.0f / live->shortest, steps, redirected, std::size(Redirects),
                run == 1 ? "by speed" : run == 2 ? "by speed (a Sprint Fix patch did it already)"
                                                 : "unchanged (not found)",
                cloth ? "by solves per second" : "per frame (not found)");
}

bool Unlocked() {
    return unlocked.load(std::memory_order_relaxed);
}
void SetLimit(int fps) {
    live_limit.store(fps, std::memory_order_relaxed);
    // bbport (macOS native GPU process): F3 and the settings menu are bb-gpu's, the frame pacer's
    // hooks the game process's (MsgFrameLimit).
    if (BbRemote::BackActive()) {
        BbRemote::Back::FrameLimit(fps);
        return;
    }
    if (!unlocked.load(std::memory_order_relaxed) || !live) {
        return;
    }
    const int rate = fps > 0 ? std::clamp(fps, 30, MaxRate) : base_rate;
    live->shortest = 1.0f / float(rate);
    std::printf("Frame rate: limit %d FPS (live)\n", rate);
}
int Limit() {
    return live_limit.load(std::memory_order_relaxed);
}
} // namespace BbFrameRate
