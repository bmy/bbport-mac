// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne hooks (gpu/games/bloodborne/hooks*.cpp): what their files share. The detours'
// stub page and its helpers, the sites Install reports, the flags OnTrap and Install read, and
// the Install* functions each file defines. Only those files include it. Shared state is
// `inline`: one instance in the whole program.
#pragma once

#include "common/types.h"
#include "bbport_guest_hooks.h"
#include "bbport_settings.h"
#include "bbport_guest_code.h"
#include "bbport_heap_sites.h"
#include "bbport_toggles.h"
#include "bbport_fx_init_clear.h"
#include "bbport_effect_flags.h"
#include "platform/gnm/gnm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include "bbport_engine_fixes.h"
#include "bbport_guest_code_hooks.h"
#include <cmath>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>
#include <utility>

extern "C" void runtime_memory_note_write(uintptr_t address, uint64_t size);
extern "C" void runtime_re_watch(uint64_t address);

namespace BbGuestHooks {

using u64 = std::uint64_t;
using u8 = std::uint8_t;
constexpr u64 ImageBase = 0x800000000ull;

inline RangeCallback on_range_dead = nullptr;
inline RangeCallback on_range_live = nullptr;

/// `call memcpy` (0x261a1b0: rdi destination, rsi source, rdx size) in the resource loaders:
/// 0x216d420 and 0x2171d60 copy texture and buffer data into GPU memory (most of the write
/// faults while areas stream in).
struct CallSite {
    u64 offset;
    unsigned char code[5];
};

inline u8* stub_page = nullptr;
inline std::size_t stub_used = 0;
constexpr std::size_t StubPageSize = 65536;

using BbGuestCode::Emitter;

/// Copies `e` into the stub page; returns its address there, or 0.
inline u64 PlaceStub(const Emitter& e) {
    if (!stub_page || e.overflow || stub_used + e.size > StubPageSize) {
        return 0;
    }
    u8* at = stub_page + stub_used;
    std::memcpy(at, e.code.data(), e.size);
    stub_used += (e.size + 15) & ~std::size_t(15);
    return reinterpret_cast<u64>(at);
}

inline bool Rel32To(u64 from_next, u64 to, std::uint32_t& rel) {
    const std::int64_t d = std::int64_t(to) - std::int64_t(from_next);
    if (d < INT32_MIN || d > INT32_MAX) {
        return false;
    }
    rel = std::uint32_t(std::int32_t(d));
    return true;
}

/// Replaces bytes [1, 5) at `address` (byte 0 is the trap already there), then byte 0.
inline void WriteJumpOverTrap(u64 address, u8 opcode, std::uint32_t rel) {
    const long page_size = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
    const std::size_t span = (address + 5) - reinterpret_cast<u64>(page);
    mprotect(page, span, PROT_READ | PROT_WRITE | PROT_EXEC);
    auto* p = reinterpret_cast<u8*>(address);
    for (int i = 0; i < 4; ++i) {
        __atomic_store_n(p + 1 + i, u8(rel >> (8 * i)), __ATOMIC_SEQ_CST);
    }
    __atomic_store_n(p, opcode, __ATOMIC_SEQ_CST);
    mprotect(page, span, Core::GuestCodeProtection());
}

void HeapFind();

// (counted below)
extern "C" int runtime_memory_is_mapped(uintptr_t address, uint64_t size);

inline u64 JobProbeWord(u64 address) {
    u64 value = 0;
    if (address && runtime_memory_is_mapped(address, sizeof(value))) {
        std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(value));
    } else if (address) {
        // Stack locals and image vtables are outside the direct-memory map. Read them safely
        // through the kernel; returning zero for those hid the failed request's size/function.
        iovec local{&value, sizeof(value)}, remote{reinterpret_cast<void*>(address), sizeof(value)};
        if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != ssize_t(sizeof(value))) {
            value = 0;
        }
    }
    return value;
}

struct ParamStore {
    u64 offset;
    std::array<u8, 5> code; ///< the store the jump covers (no RIP-relative operand)
};
constexpr std::array<ParamStore, 5> CameraParamStores = {{
    {0x1436c2c, {0xc5, 0xfa, 0x11, 0x43, 0x5c}}, // vmovss %xmm0, 0x5c(%rbx): the follow camera
    {0x1436d8a, {0xc5, 0xfa, 0x11, 0x43, 0x5c}},
    {0x1436e68, {0xc5, 0xf8, 0x11, 0x4b, 0x50}}, // vmovups %xmm1, 0x50(%rbx): a blend of two
    {0x1436eec, {0xc5, 0xfa, 0x11, 0x43, 0x5c}},
    {0x1436f4b, {0xc5, 0xfa, 0x11, 0x43, 0x5c}},
}};

/// The HUD's markers over characters (lock-on, enemies' health bars, damage numbers): projected
/// with the current camera, then placed in the game's 1920x1080 UI by `* (0.5, -0.5, ..) +
/// (0.5, 0.5, ..)` (three places), and drawn with the UI. The port puts the UI in the middle of a
/// wider (taller) output at its height (width): there a marker's 0.5 becomes 0.5 * output aspect /
/// 16:9 (16:9 / aspect for y), so it lands where its character is. The detour multiplies by the
/// port's vector instead of the game's constant; a marker past the UI's edges is hidden by the
/// game itself, as off screen.
struct MarkerSite {
    u64 offset;
    u64 scale, bias; ///< the constants the two instructions read
};
constexpr std::array<MarkerSite, 3> MarkerSites = {{
    {0x1644379, 0x48f9a00, 0x48f9a10},
    {0x1644c45, 0x48f9a60, 0x48f9a70},
    {0x16452b7, 0x48f9ad0, 0x48f9ae0},
}};

inline std::atomic<bool> hud_anchor_on{true};

/// The GX swap-chain's per-frame render command (#162): its 16-byte node comes from a per-frame
/// bump allocator that returns null once the frame's heap is used up; the game writes through it
/// unchecked (`mov [r12 + 8], r13` at 0x2199310, then posts it to the render thread): a guest
/// fault at address 8. Without a node the post is skipped and the pending flag (this + 0x3ab)
/// stays set, so the command goes out with the next frame.
constexpr u64 GxNodeSite = 0x2199310;
constexpr u64 GxNodeSkip = 0x219934e; // past the post and the flag's clear

inline std::atomic<u64> gx_node_skip_trap{0}; ///< the stub's int3 for a null node (OnTrap)

inline bool gx_query_observer = false;

inline std::atomic<bool> look_installed{false};

// Defined in the hooks_*.cpp files; Install (hooks.cpp) calls them.
bool InstallHotEffects();
bool InstallKernelSampleClamp();
bool InstallFxInitClear();
bool InstallJobAllocationProbe();
bool InstallResourceCacheSize();
bool InstallJobSlotFix();
bool InstallHeapGrow();
bool InstallHeapTrace();
bool InstallFreeCameraProbe();
int InstallCameraAspectHooks();
int InstallMarkerHooks();
bool InstallViewportHook();
bool InstallHudHook();
bool InstallLookHook();
bool InstallGxNodeGuard();
bool InstallGxQueryGuard();
bool InstallAnalogProbe();
bool InstallStickProbe();
bool InstallCameraProbe();
bool InstallEntryProbe(u64 offset);

} // namespace BbGuestHooks
