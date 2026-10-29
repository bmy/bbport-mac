// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: fixes to the game's own code (CUSA03173 1.09), from the SrShadowy fork
// (extra/core/engine_fixes), put on the port's guest code hooks (Linux and Windows) and made
// safe for several threads.

#include "bbport_engine_fixes.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "bbport_guest_code_hooks.h"
#include "bbport_settings.h"

namespace BbEngineFixes {
namespace {

constexpr std::uint64_t ImageBase = 0x800000000ull;
/// vcvttss2si rcx, xmm0: the frame's play time in ms (xmm0 = dt * 1000), added to the save's
/// play time (GameDataMan + 0x94) right after.
constexpr std::uint64_t PlayTimeConvert = 0x19419de;
constexpr std::array<std::uint8_t, 5> PlayTimeConvertCode = {0xc4, 0xe1, 0xfa, 0x2c, 0xc8};
/// Havok's 96-byte hit collector push_back (rdi = the array, rsi = the hit).
constexpr std::uint64_t HitPushBack = 0xb84ce0;
constexpr std::array<std::uint8_t, 6> HitPushBackEntry = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x56};
/// The step before a save serializes the player's data.
constexpr std::uint64_t CharaStep = 0x1c43ee0;
constexpr std::array<std::uint8_t, 6> CharaStepEntry = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
constexpr std::uint64_t GameDataMan = 0x553b130;
/// The renderer's per-view setup (its entry), once a frame.
constexpr std::uint64_t RenderView = 0x229e990;
constexpr std::array<std::uint8_t, 6> RenderViewEntry = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
/// The light grid: its two subdivisions (3 or 1), the graphics manager (its tile size at
/// +0x78a8), the frame size the game renders.
constexpr std::uint64_t LightGridDivisions = 0x513ae80;
constexpr std::uint64_t GraphicsManager = 0x5540dd8;
constexpr std::uint64_t RenderSize = 0x51289f8;

std::atomic<std::uint64_t> hits_dropped{0}, chara_ids_cleared{0};

bool Enabled(const char* name) {
    return !bb_env_off(name);
}

// The game truncates each frame's milliseconds: at 120 FPS (8.33 ms) a frame counted 8 ms and the
// play time ran 4 % slow, more the higher the frame rate. The fraction is carried to the next
// frame (one thread: the game's main loop).
void __attribute__((sysv_abi)) PlayTimeHook(std::uint64_t, const Core::GuestRegisterSnapshot* r) {
    static double carry_ms = 0.0;
    auto* regs = const_cast<Core::GuestRegisterSnapshot*>(r);
    float ms = 0.0f;
    std::memcpy(&ms, regs->ymm[0].data(), sizeof(ms));
    const double total = double(ms) + carry_ms;
    if (!(total > 0.0) || total > 60000.0) {
        carry_ms = 0.0; // a pause or a hitch: as the game counted it
        return;
    }
    const double whole = double(std::uint64_t(total));
    carry_ms = total - whole;
    const float out = float(whole);
    std::memcpy(regs->ymm[0].data(), &out, sizeof(out));
}

// A degenerate query (a zero-length ray while the world is paused) filled the collector without
// end until an allocation failed (SIGSEGV at 0xf00010, SrShadowy fork). Past
// BB_HIT_COLLECTOR_CAP (65536) hits the call writes into this thread's scratch array instead: the
// game's function runs as it is, nothing is skipped.
struct alignas(16) ScratchArray {
    std::uint8_t head[0x20];
    void* data;
    std::int32_t size;
    std::int32_t capacity;
};
void __attribute__((sysv_abi)) HitPushBackHook(std::uint64_t, const Core::GuestRegisterSnapshot* r) {
    static const std::int32_t cap = [] {
        const char* env = bb_env("BB_HIT_COLLECTOR_CAP");
        const long v = env ? std::strtol(env, nullptr, 10) : 65536;
        return std::int32_t(v > 0 ? v : 65536);
    }();
    const auto* array = reinterpret_cast<const std::uint8_t*>(r->rdi);
    if (!array) {
        return;
    }
    std::int32_t size = 0;
    void* data = nullptr;
    std::memcpy(&size, array + 0x28, sizeof(size));
    std::memcpy(&data, array + 0x20, sizeof(data));
    if (size < cap && (data || size == 0)) {
        return;
    }
    alignas(16) thread_local std::array<std::uint8_t, 96> hit{};
    thread_local ScratchArray scratch{};
    scratch.data = hit.data();
    scratch.size = 0;
    scratch.capacity = 1;
    const_cast<Core::GuestRegisterSnapshot*>(r)->rdi = reinterpret_cast<std::uint64_t>(&scratch);
    if (hits_dropped.fetch_add(1, std::memory_order_relaxed) == 0) {
        std::printf("Engine fixes: a hit collector reached %d hits: further hits dropped\n",
                    int(size));
    }
}

// An invalid character id (0x8000000000000000) in the player's data (GameDataMan + 8 -> +0x690)
// cleared before the save serializes it (SrShadowy fork: corrupted saves).
void __attribute__((sysv_abi)) CharaStepHook(std::uint64_t, const Core::GuestRegisterSnapshot*) {
    std::uint64_t manager = 0, player = 0, id = 0;
    std::memcpy(&manager, reinterpret_cast<const void*>(ImageBase + GameDataMan), sizeof(manager));
    if (manager < 0x100000000ull) {
        return;
    }
    std::memcpy(&player, reinterpret_cast<const void*>(manager + 0x8), sizeof(player));
    if (player < 0x100000000ull) {
        return;
    }
    auto* field = reinterpret_cast<std::uint8_t*>(player + 0x690);
    std::memcpy(&id, field, sizeof(id));
    if (id == 0x8000000000000000ull) {
        const std::uint64_t none = 0;
        std::memcpy(field, &none, sizeof(none));
        if (chara_ids_cleared.fetch_add(1, std::memory_order_relaxed) == 0) {
            std::printf("Engine fixes: an invalid character id cleared before a save\n");
        }
    }
}

// light_grid_opt (SrShadowy fork): the light grid in tiles of a quarter of the frame and one
// subdivision instead of 128-pixel tiles and three; back to the game's when switched off.
// BB_LIGHT_GRID_OPT=1/0 overrides the setting (tests).
void __attribute__((sysv_abi)) RenderViewHook(std::uint64_t, const Core::GuestRegisterSnapshot*) {
    static const int forced = [] {
        const char* env = bb_env("BB_LIGHT_GRID_OPT");
        return env && *env ? (env[0] == '1' ? 1 : 0) : -1;
    }();
    static bool applied = false;
    const bool want = forced >= 0 ? forced == 1
                                  : BbSettings::Get().light_grid_opt.load(std::memory_order_relaxed);
    if (!want && !applied) {
        return;
    }
    auto* divisions = reinterpret_cast<std::uint32_t*>(ImageBase + LightGridDivisions);
    if ((divisions[0] != 3 && divisions[0] != 1) || (divisions[1] != 3 && divisions[1] != 1)) {
        return; // not the values this knows
    }
    std::uint64_t manager = 0;
    std::memcpy(&manager, reinterpret_cast<const void*>(ImageBase + GraphicsManager), sizeof(manager));
    if (manager < 0x100000000ull) {
        return;
    }
    const auto* size = reinterpret_cast<const std::uint32_t*>(ImageBase + RenderSize);
    const std::uint32_t width = size[0] >= 256 && size[0] <= 7680 ? size[0] : 1920;
    const std::uint32_t height = size[1] >= 144 && size[1] <= 4320 ? size[1] : 1080;
    auto* tile = reinterpret_cast<std::uint32_t*>(manager + 0x78a8);
    divisions[0] = divisions[1] = want ? 1 : 3;
    tile[0] = want ? std::clamp((width + 3) / 4, 128u, 16384u) : 128u;
    tile[1] = want ? std::clamp((height + 3) / 4, 128u, 16384u) : 128u;
    if (want != applied) {
        std::printf("Engine fixes: light grid %s\n", want ? "coarse (light_grid_opt)" : "as the game's");
    }
    applied = want;
}

} // namespace

void Install() {
    const auto install = [](const char* env, std::uint64_t offset, auto code, auto hook) {
        if (!Enabled(env)) {
            return "off (" + std::string(env) + "=0)";
        }
        return std::string(Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + offset),
                                                      code, offset, hook)
                               ? "on"
                               : "not installed (other code there)");
    };
    const std::string play = install("BB_FIX_PLAY_TIME", PlayTimeConvert, PlayTimeConvertCode,
                                     PlayTimeHook);
    const std::string hits = install("BB_FIX_HIT_COLLECTOR", HitPushBack, HitPushBackEntry,
                                     HitPushBackHook);
    const std::string chara = install("BB_FIX_CHARA_ID", CharaStep, CharaStepEntry, CharaStepHook);
    const std::string grid = install("BB_LIGHT_GRID_HOOK", RenderView, RenderViewEntry, RenderViewHook);
    std::printf("Engine fixes: high frame rate play time %s, hit collector cap %s, save character "
                "id guard %s, light grid control %s\n",
                play.c_str(), hits.c_str(), chara.c_str(), grid.c_str());
}

} // namespace BbEngineFixes
