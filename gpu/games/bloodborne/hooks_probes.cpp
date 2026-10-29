// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne: reverse engineering probes (BB_RE_ANALOG, BB_RE_CAMSTICK, BB_RE_CAMERA,
// BB_RE_ENTRY, BB_RE_ADVANCE): who calls what, with which values. Diagnostics only. Split
// out of gpu/shim/bbport_guest_hooks.cpp (2026-10-26).

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
#ifdef __APPLE__
#include <sys/ucontext.h> // bbport: <ucontext.h> requires _XOPEN_SOURCE on macOS
#else
#include <ucontext.h>
#endif
#include <unistd.h>
#include <utility>
#include "games/bloodborne/hooks_internal.h"

namespace BbGuestHooks {
namespace {

/// BB_RE_ANALOG=1 (reverse engineering): the pad device's analog getter (0x26917d0: rdi the
/// device, esi the axis, the value at [rdi + 0x1e4 + 4 * axis]) goes through AnalogProbe, which
/// counts who asks for the right stick (axes 2 and 3) by return address, every 5 s.
constexpr u64 AnalogGetter = 0x26917d0;
constexpr unsigned char AnalogGetterCode[6] = {0xc5, 0xf8, 0x57, 0xc0, 0x83, 0xfe};
struct ProbeSlot {
    std::atomic<u64> key{0}, count{0};
};
std::array<ProbeSlot, 256> probe_slots;
std::atomic<long long> probe_printed{0};

std::atomic<u64> probe_virtual[4]{}; // axis 2/3 at 0x2698deb: r13 (the virtual index), the array
std::atomic<u64> probe_calls{0};

float PS4_SYSV_ABI AnalogProbe(const u8* device, int axis, u64 ret, u64 caller_r13) {
    static std::atomic<u64> targets[2]{}; // bit set of the virtual indices (< 64 + 64) per axis
    static std::atomic<u64> targets_high[2]{};
    if (ret == ImageBase + 0x2698deb && (axis == 2 || axis == 3)) {
        const u64 index = caller_r13 & 0xffffffff;
        if (index < 64) {
            targets[axis - 2].fetch_or(u64(1) << index, std::memory_order_relaxed);
        } else if (index < 128) {
            targets_high[axis - 2].fetch_or(u64(1) << (index - 64), std::memory_order_relaxed);
        }
        probe_virtual[(axis - 2) * 2].store(caller_r13 & 0xffffffff, std::memory_order_relaxed);
        probe_virtual[(axis - 2) * 2 + 1].store(*reinterpret_cast<const u64*>(device + 0x100),
                                                std::memory_order_relaxed);
        // BB_RE_WATCH_VIRTUAL=1: the runtime's watchpoint on the virtual right stick X.
        static const bool watch_virtual = bb_env_on("BB_RE_WATCH_VIRTUAL");
        static const u64 watch_index = [] {
            const char* env = bb_env("BB_RE_WATCH_INDEX");
            return env ? std::strtoull(env, nullptr, 0) : 62ull;
        }();
        if (watch_virtual && axis == 2 && (caller_r13 & 0xffffffff) == watch_index &&
            probe_calls.fetch_add(1) == 3000) {
            runtime_re_watch(*reinterpret_cast<const u64*>(device + 0x100) + 4 * (caller_r13 & 0xffffffff));
        }
    }
    const float value = unsigned(axis) <= 0x15 ? *reinterpret_cast<const float*>(device + 0x1e4 + 4 * axis)
                                                : 0.0f;
    // Both sticks (axes 0-1 the left, 2-3 the right), counted only while one is pushed: who turns
    // something with them right now (the character creator's model, 2026-10-27).
    if (unsigned(axis) <= 3 && std::fabs(value) > 0.2f) {
        const u64 key = (ret << 2) | u64(axis);
        for (std::size_t i = 0; i < probe_slots.size(); ++i) {
            auto& slot = probe_slots[((key * 0x9E3779B97F4A7C15ull >> 56) + i) & 255];
            u64 expected = 0;
            if (slot.key.load(std::memory_order_relaxed) == key ||
                slot.key.compare_exchange_strong(expected, key)) {
                slot.count.fetch_add(1, std::memory_order_relaxed);
                break;
            }
        }
    }
    const long long now = time(nullptr);
    long long printed = probe_printed.load(std::memory_order_relaxed);
    if (now - printed >= 5 && probe_printed.compare_exchange_strong(printed, now)) {
        std::printf("RE analog: stick readers while pushed (return - image, axis: count):");
        for (auto& slot : probe_slots) {
            if (const u64 key = slot.key.load(); key != 0) {
                std::printf(" %#llx/%llu:%llu", (unsigned long long)((key >> 2) - ImageBase),
                            (unsigned long long)(key & 3),
                            (unsigned long long)slot.count.exchange(0));
            }
        }
        std::printf("; targets RX %#llx:%#llx RY %#llx:%#llx",
                    (unsigned long long)targets_high[0].load(), (unsigned long long)targets[0].load(),
                    (unsigned long long)targets_high[1].load(), (unsigned long long)targets[1].load());
        std::printf("; virtual: RX index %llu in %#llx, RY index %llu in %#llx\n",
                    (unsigned long long)probe_virtual[0].load(), (unsigned long long)probe_virtual[1].load(),
                    (unsigned long long)probe_virtual[2].load(), (unsigned long long)probe_virtual[3].load());
    }
    // BB_RE_ZERO_VIRTUAL=1: the virtual inputs fed from the right stick's X at 0x2698deb get 0.
    static const bool zero_virtual = bb_env_on("BB_RE_ZERO_VIRTUAL");
    if (zero_virtual && axis == 2 && ret == ImageBase + 0x2698deb) {
        return 0.0f;
    }
    return value;
}

/// BB_RE_CAMSTICK=1 (reverse engineering): who calls the shared stick getter 0x1fdcc90 (vtable
/// slot 0x38 of the input manipulators: two dead zones from 0x49292f0, then 0xf92690), by return
/// address, every 5 s.
constexpr u64 StickThunk = 0x1fdcc90;
constexpr unsigned char StickThunkCode[4] = {0xc5, 0xfa, 0x10, 0x05};
std::array<ProbeSlot, 64> stick_slots;
std::atomic<long long> stick_printed{0};
thread_local u64 stick_ids = 0; // the call's action pair (esi | edx << 32)
std::atomic<u64> stick_seen[4]{}; // pair, last result, pair, last result (the first two pairs)
void PS4_SYSV_ABI StickCaller(u64 ret, u64 first, u64 second, u64 caller_r12) {
    stick_ids = (first & 0xffffffff) | (second << 32);
    // BB_RE_WATCH_CAMIN=1: the runtime's watchpoint on the camera input the manipulator keeps
    // (0x1526750 stores the camera stick at r12 + 0x40 after the call returning to 0x15283e4).
    static const bool watch_camera_input = bb_env_on("BB_RE_WATCH_CAMIN");
    static std::atomic<u64> camera_calls{0};
    if (watch_camera_input && ret == ImageBase + 0x15283e4 && camera_calls.fetch_add(1) == 1000) {
        runtime_re_watch(caller_r12 + 0x40);
    }
    for (std::size_t i = 0; i < stick_slots.size(); ++i) {
        auto& slot = stick_slots[((ret * 0x9E3779B97F4A7C15ull >> 58) + i) & 63];
        u64 expected = 0;
        if (slot.key.load(std::memory_order_relaxed) == ret || slot.key.compare_exchange_strong(expected, ret)) {
            slot.count.fetch_add(1, std::memory_order_relaxed);
            break;
        }
    }
    const long long now = time(nullptr);
    long long printed = stick_printed.load(std::memory_order_relaxed);
    if (now - printed >= 5 && stick_printed.compare_exchange_strong(printed, now)) {
        for (int i = 0; i < 4; i += 2) {
            const u64 ids = stick_seen[i].load();
            const u64 v = stick_seen[i + 1].load();
            float x, y;
            const std::uint32_t lo = std::uint32_t(v), hi = std::uint32_t(v >> 32);
            std::memcpy(&x, &lo, 4);
            std::memcpy(&y, &hi, 4);
            std::printf("RE stick (%#llx, %#llx): %.3f %.3f\n", (unsigned long long)(ids & 0xffffffff),
                        (unsigned long long)(ids >> 32), x, y);
        }
        std::printf("RE stick getter callers (return - image: count):");
        for (auto& slot : stick_slots) {
            if (const u64 key = slot.key.load(); key != 0) {
                std::printf(" %#llx:%llu", (unsigned long long)(key - ImageBase),
                            (unsigned long long)slot.count.exchange(0));
            }
        }
        std::printf("\n");
    }
}

/// The getter's result (x, y in rax): noted per action pair, returned (to be changed later).
u64 PS4_SYSV_ABI StickResult(u64 result) {
    const u64 ids = stick_ids;
    for (int i = 0; i < 4; i += 2) {
        u64 expected = 0;
        if (stick_seen[i].load() == ids || stick_seen[i].compare_exchange_strong(expected, ids)) {
            if (result != 0 || stick_seen[i + 1].load() == 0) {
                stick_seen[i + 1].store(result);
            }
            break;
        }
    }
    return result;
}

/// The follow camera's update (0x143ac60: rdi the camera; its angles at +0x140 and +0x144, from
/// the camera stick's x and y, in radians within [-pi, pi]). BB_RE_CAMERA=1 hooks its entry:
/// BB_RE_WATCH_CAMANGLE=1 watches +0x140 (who writes it), BB_RE_CAM_ADD=<radians> turns the camera
/// by that much each update (does writing the angle turn it?).
constexpr u64 CameraUpdate = 0x143ac60;
constexpr unsigned char CameraUpdateCode[6] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
void PS4_SYSV_ABI CameraEntry(u8* camera) {
    static const bool watch_angle = bb_env_on("BB_RE_WATCH_CAMANGLE");
    static const float add = [] {
        const char* env = bb_env("BB_RE_CAM_ADD");
        return env ? std::strtof(env, nullptr) : 0.0f;
    }();
    static std::atomic<u64> calls{0};
    const u64 n = calls.fetch_add(1, std::memory_order_relaxed);
    if (watch_angle && n == 1000) {
        runtime_re_watch(reinterpret_cast<u64>(camera) + 0x140);
    }
    if (add != 0.0f) {
        float& yaw = *reinterpret_cast<float*>(camera + 0x140);
        yaw += add;
        if (yaw > 3.14159265f) {
            yaw -= 6.28318531f;
        }
    }
    if (n % 2000 == 0) {
        const auto f = [camera](int o) { return *reinterpret_cast<float*>(camera + o); };
        std::printf("RE camera %p: angles %.4f %.4f speed %.3f/%.3f %.3f/%.3f ramp %.3f %.3f\n",
                    static_cast<void*>(camera), f(0x140), f(0x144), f(0x204), f(0x21c), f(0x208), f(0x220),
                    f(0x138), f(0x1e8));
    }
}

/// BB_RE_ENTRY=0x20f00,0x12ab4c0,... (reverse engineering): functions' entries (`push rbp; mov rbp,
/// rsp` then `push r1x` or `push rbx`) counted, with the float argument xmm0 (min, max, mean) and
/// rsi, every 2 s: how often the game calls them and with which frame time.
/// BB_RE_ADVANCE=1: the same for Scaleform's MovieImpl::Advance (0x20f00, xmm0: the time).
struct EntrySlot {
    u64 offset = 0;
    std::atomic<u64> calls{0};
    std::atomic<std::uint32_t> min_bits{0x7f800000}, max_bits{0};
    std::atomic<u64> last_rsi{0};
    double sum = 0;
};
std::array<EntrySlot, 128> entry_slots;
std::size_t entry_count = 0;
std::atomic<long long> entry_printed{0};

template <int I>
void PS4_SYSV_ABI EntryProbe(u64, u64 rsi, float x) {
    EntrySlot& slot = entry_slots[I];
    slot.calls.fetch_add(1, std::memory_order_relaxed);
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
    if (x < std::bit_cast<float>(slot.min_bits.load())) slot.min_bits.store(bits);
    if (x > std::bit_cast<float>(slot.max_bits.load()) || slot.max_bits.load() == 0) slot.max_bits.store(bits);
    slot.sum += x;
    slot.last_rsi.store(rsi, std::memory_order_relaxed);
    std::atomic<long long>& printed = entry_printed;
    const long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
    long long last = printed.load();
    if (now - last < 2000 || !printed.compare_exchange_strong(last, now)) return;
    for (std::size_t k = 0; k < entry_count; ++k) {
        EntrySlot& e = entry_slots[k];
        const u64 n = e.calls.exchange(0);
        if (n == 0) continue;
        std::printf("RE entry %#llx: %llu calls in 2 s, xmm0 %.5f..%.5f mean %.5f, rsi %#llx\n",
                    (unsigned long long)e.offset, (unsigned long long)n,
                    n ? std::bit_cast<float>(e.min_bits.load()) : 0.0f, std::bit_cast<float>(e.max_bits.load()),
                    n ? e.sum / double(n) : 0.0, (unsigned long long)e.last_rsi.load());
        e.sum = 0;
        e.min_bits.store(0x7f800000);
        e.max_bits.store(0);
    }
}
template <std::size_t... I>
constexpr std::array<void (PS4_SYSV_ABI*)(u64, u64, float), sizeof...(I)> EntryProbes(std::index_sequence<I...>) {
    return {&EntryProbe<int(I)>...};
}
constexpr auto entry_probes = EntryProbes(std::make_index_sequence<128>{});

} // namespace

/// The getter's first bytes become a jump to `mov rdx, [rsp]; mov rcx, r13; jmp AnalogProbe`
/// (a tail call).
bool InstallAnalogProbe() {
    const u64 address = ImageBase + AnalogGetter;
    if (std::memcmp(reinterpret_cast<const void*>(address), AnalogGetterCode, sizeof(AnalogGetterCode))) {
        return false;
    }
    Emitter e;
    e.Bytes({0x48, 0x8b, 0x14, 0x24});             // mov rdx, [rsp]
    e.Bytes({0x4c, 0x89, 0xe9});                   // mov rcx, r13 (the caller's)
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(reinterpret_cast<u64>(&AnalogProbe));
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

bool InstallStickProbe() {
    const u64 address = ImageBase + StickThunk;
    if (std::memcmp(reinterpret_cast<const void*>(address), StickThunkCode, sizeof(StickThunkCode))) {
        return false;
    }
    std::uint32_t inner, outer;
    std::memcpy(&inner, reinterpret_cast<const void*>(ImageBase + 0x49292f0), 4);
    std::memcpy(&outer, reinterpret_cast<const void*>(ImageBase + 0x49292f4), 4);
    Emitter e;
    e.SaveVolatile();
    e.Bytes({0x49, 0x8b, 0x7c, 0x24, 0x50}); // mov rdi, [r12 + 0x50]: the return address (rsi, rdx: the pair)
    e.Bytes({0x49, 0x8b, 0x0c, 0x24});       // mov rcx, [r12]: the caller's r12
    e.CallAbs(reinterpret_cast<const void*>(&StickCaller));
    e.RestoreVolatile();
    e.Bytes({0xb8}); e.U32(inner);              // mov eax, inner
    e.Bytes({0xc5, 0xf9, 0x6e, 0xc0});          // vmovd xmm0, eax
    e.Bytes({0xb8}); e.U32(outer);              // mov eax, outer
    e.Bytes({0xc5, 0xf9, 0x6e, 0xc8});          // vmovd xmm1, eax
    e.Bytes({0x48, 0x83, 0xec, 0x08});          // sub rsp, 8 (the callee's alignment)
    e.CallAbs(reinterpret_cast<const void*>(ImageBase + 0xf92690));
    e.Bytes({0x48, 0x83, 0xc4, 0x08});          // add rsp, 8
    e.SaveVolatile();
    e.Bytes({0x48, 0x89, 0xc7});                // mov rdi, rax
    e.CallAbs(reinterpret_cast<const void*>(&StickResult));
    e.Bytes({0x49, 0x89, 0x44, 0x24, 0x48});    // mov [r12 + 0x48], rax: the saved rax
    e.RestoreVolatile();
    e.Bytes({0xc3});                            // ret
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

bool InstallCameraProbe() {
    const u64 address = ImageBase + CameraUpdate;
    if (std::memcmp(reinterpret_cast<const void*>(address), CameraUpdateCode, sizeof(CameraUpdateCode))) {
        return false;
    }
    Emitter e;
    e.SaveVolatile();
    e.CallAbs(reinterpret_cast<const void*>(&CameraEntry)); // rdi: the camera, unchanged
    e.RestoreVolatile();
    e.Bytes({0x55, 0x48, 0x89, 0xe5, 0x41, 0x57});         // the instructions the jump covers
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});         // jmp [rip]
    e.U64(address + sizeof(CameraUpdateCode));
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

bool InstallEntryProbe(u64 offset) {
    if (entry_count == entry_slots.size()) return false;
    const u64 address = ImageBase + offset;
    const u8* code = reinterpret_cast<const u8*>(address);
    // push rbp; mov rbp, rsp; then push r12..r15 (2 bytes) or push rbx (1 byte)
    if (code[0] != 0x55 || code[1] != 0x48 || code[2] != 0x89 || code[3] != 0xe5) return false;
    std::size_t length = 0;
    if (code[4] == 0x41 && code[5] >= 0x54 && code[5] <= 0x57) length = 6;
    else if (code[4] == 0x53) length = 5;
    else return false;
    Emitter e;
    e.SaveVolatile();
    e.SaveXmm(); // xmm0 reaches the probe as its float argument
    e.CallAbs(reinterpret_cast<const void*>(entry_probes[entry_count]));
    e.RestoreXmm();
    e.RestoreVolatile();
    for (std::size_t i = 0; i < length; ++i) e.Bytes({code[i]});
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(address + length);
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) return false;
    entry_slots[entry_count++].offset = offset;
    const long page_size = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
    mprotect(page, (address + length) - reinterpret_cast<u64>(page), PROT_READ | PROT_WRITE | PROT_EXEC);
    reinterpret_cast<u8*>(address)[0] = 0xcc;
    for (std::size_t i = 5; i < length; ++i) reinterpret_cast<u8*>(address)[i] = 0xcc;
    mprotect(page, (address + length) - reinterpret_cast<u64>(page), Core::GuestCodeProtection());
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

} // namespace BbGuestHooks
