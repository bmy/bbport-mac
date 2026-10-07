// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the few CPU instructions the GPU side uses directly, for x86-64 (Linux, macOS under
// Rosetta) and arm64 (the native macOS GPU process, docs/macos-native-gpu.md): a cheap tick
// counter for timing statistics, and the spin-wait hint.
#pragma once

#include <cstdint>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

namespace BbCpu {

/// A monotonic counter for short intervals and ratios (TSC on x86-64, the virtual counter on
/// arm64: 24 MHz on Apple Silicon). Not comparable across machines; see TicksFromUs.
inline std::uint64_t Ticks() {
#if defined(__x86_64__)
    return __rdtsc();
#elif defined(__aarch64__)
    std::uint64_t value;
    asm volatile("mrs %0, cntvct_el0" : "=r"(value));
    return value;
#else
#error "BbCpu::Ticks: unsupported architecture"
#endif
}

/// Hint inside a spin-wait loop.
inline void Pause() {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    __builtin_arm_yield();
#endif
}

/// About `us` microseconds in Ticks(). x86-64 keeps the ~3.2 GHz the thresholds were tuned with
/// (the TSC rate is not read); arm64 uses the counter's architectural frequency.
inline std::uint64_t TicksFromUs(std::uint64_t us) {
#if defined(__aarch64__)
    std::uint64_t frequency;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(frequency));
    return us * frequency / 1000000;
#else
    return us * 3200;
#endif
}

} // namespace BbCpu
