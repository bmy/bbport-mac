// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne: the game's effects switched live (SSAO, motion blur, AA, shadows, chromatic
// aberration, depth of field, bloom, FXAA2_GS), the FX initialization clear and the
// blur-kernel sample clamp. Split out of gpu/shim/bbport_guest_hooks.cpp (2026-10-26).

#include "common/types.h"
#include "shim/bb_env.h"
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
#include "games/bloodborne/hooks_internal.h"

namespace BbGuestHooks {
namespace {

/// Game effects switched while the game runs (no restart). The community patches only set them
/// at start; here the same switches are written live:
///  - Ambient occlusion (SSAO), motion blur, the game's own anti-aliasing and the shadows of
///    dynamic lights are bytes of the renderer's options (+0x2968, +0x2969, +0x296a, +0x296e;
///    the patches change the immediates that set them in the constructor at 0x22c2542, 0x22c2532).
///    The renderer tests them while it builds each frame. The constructor's tail (0x22c254c)
///    reports the options' address; written there and on every change.
///  - Chromatic aberration (0x229faa8) and depth of field (0x21d7bbc) are code the patches
///    replace: written back and forth here; the game's own bytes are kept as they were found.
/// BB_HOT_EFFECTS=0: the patches at start as before (patches.py).
/// The options objects built so far (the game builds about ten at a load, one per render context).
constexpr std::size_t MaxEffectObjects = 64;
std::array<std::atomic<u64>, MaxEffectObjects> effects_objects{};
std::atomic<std::size_t> effects_object_count{0};
/// The code switches' state; the slot after the effects: the game's FXAA forced.
constexpr int FxaaSlot = BbSettings::EffectCount;
/// The AA step's `testb $4, 0x49(%r12)` (6 bytes) and where the step goes on (all text offsets).
constexpr u64 FxaaSite = 0x22a07a6;
constexpr u64 FxaaDone = 0x22a081a;
constexpr u64 AaPrepare = 0xe596f0; ///< (AA object, options): enabled, technique - 1, constants
constexpr u64 AaPass = 0xe58c50;    ///< (AA object, target, 0x34, 9, r8): the pass
std::atomic<int> effects_applied[BbSettings::EffectCount + 1];
struct HotCode {
    int effect;              ///< BbSettings::Effects index
    u64 offset;
    std::array<u8, 12> patch; /// the community patch's bytes
    std::size_t size;
    bool patch_when_on = false; /// the patch enables the effect (SSR), else disables it
    bool data = false;          /// a constant in the game's data, not code
    std::array<u8, 12> original{};
    bool ready = false;
};
/// SSR requires the complete startup patch, including its resource dimensions.
/// Partial live writes leave a half-size target behind full-size shader accesses.
std::array<HotCode, 4> hot_code = {{
    {0, 0x229faa8, {0xc7, 0x83, 0xac, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x90, 0x90}, 12},
    {1, 0x21d7bbc, {0xe9, 0xdf, 0x00, 0x00, 0x00, 0x90}, 6},
    {7, 0x21d72f0, {0x31, 0xf6, 0x0f, 0x1f, 0x40, 0x00}, 6}, // "Disable Bloom"
    // FXAA2_GS (after Kyo's "FXAA2_GS + DLAA_GS", one pass): the frame's AA step (behind the AA
    // switch, [r12+0x296a]) runs its pass with the technique the options give, or none. The
    // `testb` deciding it becomes a jump to a stub (InstallFxaaStub) running the pass with
    // technique 0, FXAA2_GS. Read every frame: live. Its bytes are made there.
    {FxaaSlot, FxaaSite, {}, 6, true},
}};
struct HotFlag {
    int effect;
    std::uint32_t offset;
};
/// These switches take effect when the game builds its render contexts (a load: teleport, death,
/// a save loaded): the frame's passes follow what the contexts were built with (checked
/// 2026-10-19 by the SSAO passes, ps 01ae5388 / cbbca5dc: unchanged by live writes, gone after a
/// load). Written live anyway, so that the next load builds them as set.
constexpr std::array<HotFlag, 4> hot_flags = {{{2, 0x2969}, {3, 0x2968}, {4, 0x296a}, {5, 0x296e}}};

/// Writes `size` bytes of code from the last to the first (the first byte, an opcode, is the
/// one that makes the new code reachable).
void WriteHotCode(u64 address, const u8* bytes, std::size_t size, bool data) {
    if (data) {
        // A constant: its page keeps the protection it has (read-only data) after the write.
        int prot = PROT_READ;
        if (FILE* maps = std::fopen("/proc/self/maps", "r")) {
            char line[512];
            while (std::fgets(line, sizeof(line), maps)) {
                unsigned long long lo = 0, hi = 0;
                char perms[8] = {};
                if (std::sscanf(line, "%llx-%llx %7s", &lo, &hi, perms) == 3 && address >= lo && address < hi) {
                    prot = (perms[0] == 'r' ? PROT_READ : 0) | (perms[1] == 'w' ? PROT_WRITE : 0) |
                           (perms[2] == 'x' ? PROT_EXEC : 0);
                    break;
                }
            }
            std::fclose(maps);
        }
        const long page_size = sysconf(_SC_PAGESIZE);
        void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
        const std::size_t span = (address + size) - reinterpret_cast<u64>(page);
        mprotect(page, span, prot | PROT_WRITE);
        std::memcpy(reinterpret_cast<void*>(address), bytes, size);
        mprotect(page, span, prot);
        return;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
    const std::size_t span = (address + size) - reinterpret_cast<u64>(page);
    mprotect(page, span, PROT_READ | PROT_WRITE | PROT_EXEC);
    auto* p = reinterpret_cast<u8*>(address);
    for (std::size_t i = size; i-- > 1;) {
        __atomic_store_n(p + i, bytes[i], __ATOMIC_SEQ_CST);
    }
    __atomic_store_n(p, bytes[0], __ATOMIC_SEQ_CST);
    mprotect(page, span, Core::GuestCodeProtection());
}

void ApplyHotEffects(bool force, u64 fresh = 0) {
    auto& settings = BbSettings::Get();
    BbSettings::SyncAntiAliasing(); // the game's AA and FXAA from anti_aliasing and the upscaler
    for (auto& code : hot_code) {
        if (!code.ready) {
            continue;
        }
        const bool on = code.effect == FxaaSlot
                            ? settings.fxaa_forced.load(std::memory_order_relaxed)
                            : settings.effects[code.effect].load(std::memory_order_relaxed);
        const int want = on ? 1 : 0;
        if (!force && effects_applied[code.effect].load() == want) {
            continue;
        }
        effects_applied[code.effect].store(want);
        const bool use_patch = (want != 0) == code.patch_when_on;
        WriteHotCode(ImageBase + code.offset, use_patch ? code.patch.data() : code.original.data(), code.size,
                     code.data);
    }
    const std::size_t count = std::min(effects_object_count.load(std::memory_order_acquire), MaxEffectObjects);
    std::size_t live_objects = 0;
    // BB_EFFECT_DUMP=1 (diagnostics): each options object's switches, capabilities and SSAO
    // target, every 10 s.
    static const bool dump = bb_env("BB_EFFECT_DUMP") != nullptr;
    static std::atomic<u64> last_dump{0};
    const u64 now_s = dump ? u64(std::time(nullptr)) : 0;
    const bool dump_now = dump && now_s >= last_dump.load() + 10 && (last_dump.store(now_s), true);
    for (std::size_t i = 0; i < count; ++i) {
        const u64 object = effects_objects[i].load(std::memory_order_relaxed);
        const bool is_fresh = object == fresh; // just built: certainly an options object
        // An object the game has freed since may hold anything: it is written only while its
        // switches still look like switches (every byte 0 or 1, +0x2968..+0x2970).
        // Still the options object: its switches look like switches, and the block of constants the
        // constructor copies from the game's data (32 bytes at +0x2980) is still whole. The game frees
        // these objects; the memory is reused, and a block of zeros has passed for switches: a write
        // there damaged the game's heap (Havok's job queue stopped, a black screen).
        u8 block[0x40]{};
        // process_vm_readv fails on the game's memory (its pages are protected for the kernel's copy);
        // the runtime's own map says whether the range is still mapped.
        if (!runtime_memory_is_mapped(object + 0x2960, sizeof(block))) {
            continue; // not mapped any more
        }
        std::memcpy(block, reinterpret_cast<const void*>(object + 0x2960), sizeof(block));
        bool plausible = std::memcmp(block + 0x20, reinterpret_cast<const void*>(ImageBase + 0x5128a40), 32) == 0;
        for (int b = 8; b <= 0x10; ++b) {
            plausible &= block[b] <= 1;
        }
        if (!plausible && !is_fresh) {
            continue;
        }
        ++live_objects;
        if (dump_now) {
            u64 caps = 0, target_a = 0, target_b = 0;
            std::memcpy(&caps, reinterpret_cast<const void*>(object + 0x48), 8);
            std::memcpy(&target_a, reinterpret_cast<const void*>(object + 0x10d0 + 0x68), 8);
            std::memcpy(&target_b, reinterpret_cast<const void*>(object + 0x10d0 + 0xc0), 8);
            std::printf("Effects dump: %#llx switches %02x %02x %02x .. %02x caps %#llx ssao target "
                        "%#llx/%#llx\n",
                        (unsigned long long)object, block[8], block[9], block[10], block[14],
                        (unsigned long long)caps, (unsigned long long)target_a,
                        (unsigned long long)target_b);
        }
        for (const auto& flag : hot_flags) {
            int want = settings.effects[flag.effect].load(std::memory_order_relaxed) ? 1 : 0;
            if (flag.effect == 3) {
                // Menus and auxiliary contexts can deliberately have no SSAO target.
                // Preserve that restriction instead of overriding the game's safety flag.
                want = BbEffectFlags::SsaoEnabled(
                    {reinterpret_cast<const u8*>(object), 0x29a0}, want != 0);
            }
            auto* byte = reinterpret_cast<volatile u8*>(object + flag.offset);
            if (force || *byte != want) {
                __atomic_store_n(const_cast<u8*>(byte), u8(want), __ATOMIC_RELAXED);
            }
        }
    }
    static std::atomic<int> told{0};
    if (force && told.fetch_add(1) < 8) {
        std::printf("Guest hooks: game effects written to %zu of %zu options objects (the rest are gone)\n",
                    live_objects, count);
    }
}

/// The renderer options object is built (rbx at the constructor's tail): its switches follow the
/// settings from the start.
void PS4_SYSV_ABI EffectsConstructed(u64 object) {
    // Built again at the same place: not counted twice.
    const std::size_t known = std::min(effects_object_count.load(std::memory_order_acquire), MaxEffectObjects);
    bool seen = false;
    for (std::size_t i = 0; i < known; ++i) {
        seen |= effects_objects[i].load(std::memory_order_relaxed) == object;
    }
    if (!seen) {
        const std::size_t at = effects_object_count.fetch_add(1);
        if (at < MaxEffectObjects) {
            effects_objects[at].store(object, std::memory_order_release);
        } else {
            effects_object_count.store(MaxEffectObjects);
        }
        static std::atomic<int> told{0};
        if (told.fetch_add(1) < 3) {
            std::printf("Guest hooks: renderer options at %#llx (the game builds several; game effects "
                        "follow the settings in each)\n", static_cast<unsigned long long>(object));
        }
    }
    ApplyHotEffects(true, object);
}

/// The FXAA2_GS stub; its jump goes into hot_code's FxaaSlot entry. Kyo's patch bytes in order,
/// with the second (DLAA_GS) pass left out:
///   lea rbx, [r12 + 0x1bc8]; options -> AA object; enabled = 1; technique = 0;
///   pass (rdi rbx, rsi target, 0x34, 9, r8 [rbp - 0xe88]); r15b = 1 (AA ran); jmp FxaaDone.
void InstallFxaaStub() {
    static constexpr std::array<u8, 6> test = {0x41, 0xf6, 0x44, 0x24, 0x49, 0x04};
    if (!stub_page || std::memcmp(reinterpret_cast<const void*>(ImageBase + FxaaSite), test.data(),
                                  test.size())) {
        return;
    }
    const u64 base = reinterpret_cast<u64>(stub_page + stub_used);
    Emitter e;
    bool reachable = true;
    const auto call = [&](u64 target) {
        std::uint32_t rel = 0;
        reachable = Rel32To(base + e.size + 5, ImageBase + target, rel) && reachable;
        e.Bytes({0xe8});
        e.U32(rel);
    };
    e.Bytes({0x49, 0x8d, 0x9c, 0x24, 0xc8, 0x1b, 0x00, 0x00}); // lea rbx, [r12 + 0x1bc8]
    e.Bytes({0x48, 0x89, 0xdf});                               // mov rdi, rbx
    e.Bytes({0x48, 0x8d, 0xb5, 0x20, 0xf5, 0xff, 0xff});       // lea rsi, [rbp - 0xae0]
    call(AaPrepare);
    e.Bytes({0xc6, 0x43, 0x09, 0x01});                         // mov byte [rbx + 9], 1
    e.Bytes({0x83, 0x63, 0x0c, 0x00});                         // and dword [rbx + 0xc], 0
    e.Bytes({0x48, 0x89, 0xdf});                               // mov rdi, rbx
    e.Bytes({0x48, 0x8d, 0xb5, 0x60, 0xf7, 0xff, 0xff});       // lea rsi, [rbp - 0x8a0]
    e.Bytes({0xba, 0x34, 0x00, 0x00, 0x00});                   // mov edx, 0x34
    e.Bytes({0xb9, 0x09, 0x00, 0x00, 0x00});                   // mov ecx, 9
    e.Bytes({0x4c, 0x8b, 0x85, 0x78, 0xf1, 0xff, 0xff});       // mov r8, [rbp - 0xe88]
    call(AaPass);
    e.Bytes({0x41, 0xb7, 0x01});                               // mov r15b, 1
    std::uint32_t rel_done = 0;
    if (!reachable || !Rel32To(base + e.size + 5, ImageBase + FxaaDone, rel_done)) {
        return;
    }
    e.Bytes({0xe9});
    e.U32(rel_done);
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || stub != base || !Rel32To(ImageBase + FxaaSite + 5, stub, rel)) {
        return;
    }
    for (auto& code : hot_code) {
        if (code.effect == FxaaSlot) {
            code.patch = {0xe9, u8(rel), u8(rel >> 8), u8(rel >> 16), u8(rel >> 24), 0x90};
        }
    }
}

void PS4_SYSV_ABI FxInitClear(void* context, void* state, void* texture, const float* color) {
    if (gettid() == getpid() && texture && color &&
        runtime_memory_is_mapped(reinterpret_cast<u64>(texture) + 0x10, 11 * sizeof(u32))) {
        std::array<u32, 11> target;
        std::memcpy(target.data(), static_cast<const u8*>(texture) + 0x10, sizeof(target));
        const auto clear = BbFxInit::Plan(target, std::span{color, 4});
        if (clear && runtime_memory_is_mapped(clear->address, clear->bytes) &&
            BbGnm::SubmitFxInitZero(clear->address, clear->bytes)) {
            static u64 count = 0; // main thread only
            if (++count <= 3 || (count & (count - 1)) == 0)
                std::printf("FX initialization: %llu targets cleared through private commands "
                            "(%#llx +%u bytes)\n", (unsigned long long)count,
                            (unsigned long long)clear->address, clear->bytes);
            return;
        }
    }
    using NativeClear = void(PS4_SYSV_ABI *)(void*, void*, void*, const float*);
    reinterpret_cast<NativeClear>(ImageBase + 0x2168630)(context, state, texture, color);
}

} // namespace

bool InstallHotEffects() {
    if (bb_env_off("BB_HOT_EFFECTS")) {
        return false;
    }
    constexpr u64 offset = 0x22c254c;
    static constexpr std::array<u8, 11> tail = {0x48, 0xc7, 0x83, 0x77, 0x29, 0x00,
                                                0x00, 0x00, 0x00, 0x00, 0x00};
    const u64 address = ImageBase + offset;
    if (std::memcmp(reinterpret_cast<const void*>(address), tail.data(), tail.size())) {
        return false; // another game version, or the patches changed it
    }
    InstallFxaaStub();
    // The code effects: the game's own bytes, taken before anything is written.
    for (auto& code : hot_code) {
        if (code.effect == FxaaSlot && code.patch[0] != 0xe9) {
            continue; // no FXAA2_GS stub
        }
        const u8* at = reinterpret_cast<const u8*>(ImageBase + code.offset);
        if (std::memcmp(at, code.patch.data(), code.size) == 0) {
            continue; // the startup patch is in (or the bytes are the same): left as it is
        }
        std::memcpy(code.original.data(), at, code.size);
        code.ready = true;
    }
    Emitter e;
    e.SaveVolatile();
    e.Bytes({0x48, 0x89, 0xdf});                           // mov rdi, rbx
    e.CallAbs(reinterpret_cast<const void*>(&EffectsConstructed));
    e.RestoreVolatile();
    for (const u8 b : tail) {
        e.Bytes({b});                                      // the instruction the jump covers
    }
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});         // jmp [rip]
    e.U64(address + tail.size());
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    for (auto& code : hot_code) {
        if (code.ready) {
            effects_applied[code.effect].store(-1);
        }
    }
    ApplyHotEffects(true); // the code effects before the game runs
    return true;
}

/// Yebis's blur-kernel sample parameters (GPUTexUtil_GetRecursiveSampleParameters, 0xbbc3e0): its
/// first argument, the base sample count, must be at least 3 (an assertion the retail build has
/// compiled out). Game data now and then gives 2; the function then runs on in a state that does
/// not end (the guest stops submitting: a black screen), so the count is raised to 3 at the entry.
/// BB_KERNEL_CLAMP=0: not installed.
bool InstallKernelSampleClamp() {
    if (const char* env = bb_env("BB_KERNEL_CLAMP"); env && env[0] == '0') {
        return false;
    }
    constexpr u64 offset = 0xbbc3e0;
    static constexpr std::array<u8, 6> entry = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57}; // push rbp; mov rbp, rsp; push r15
    const u64 address = ImageBase + offset;
    if (std::memcmp(reinterpret_cast<const void*>(address), entry.data(), entry.size())) {
        return false;
    }
    Emitter e;
    e.Bytes({0x83, 0xff, 0x03});                   // cmp edi, 3
    e.Bytes({0x7d, 0x05});                         // jge over
    e.Bytes({0xbf, 0x03, 0x00, 0x00, 0x00});       // mov edi, 3
    for (const u8 b : entry) {
        e.Bytes({b});                              // the instructions the jump covers
    }
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(address + entry.size());
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

bool InstallFxInitClear() {
    if (bb_env_off("BB_FX_INIT_CLEAR")) return false;
    // Only the three initialization calls in 0xebb960; regular clears retain the game's path.
    constexpr std::array<CallSite, 3> sites{{
        {0xebbc86, {0xe8, 0xa5, 0xc9, 0x2a, 0x01}},
        {0xebbca4, {0xe8, 0x87, 0xc9, 0x2a, 0x01}},
        {0xebbcc2, {0xe8, 0x69, 0xc9, 0x2a, 0x01}},
    }};
    for (const auto& site : sites)
        if (std::memcmp(reinterpret_cast<const void*>(ImageBase + site.offset), site.code, 5))
            return false;
    Emitter e;
    e.JmpAbs(reinterpret_cast<u64>(&FxInitClear));
    const u64 stub = PlaceStub(e);
    if (!stub) return false;
    for (const auto& site : sites) {
        std::uint32_t rel = 0;
        if (!Rel32To(ImageBase + site.offset + 5, stub, rel)) return false;
        WriteJumpOverTrap(ImageBase + site.offset, 0xe8, rel);
    }
    return true;
}

void ApplyGameEffects() {
    ApplyHotEffects(false);
}

} // namespace BbGuestHooks
