// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the CPU's time stamp counter, read in order with the code around it.
#pragma once

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#include "common/types.h"

namespace Common {

/// bbport (the macOS native GPU process, arm64): the virtual counter there. Not the guest's TSC:
/// what the guest sees comes from the game process (BbRemote::Back::ReadTsc).
inline u64 FencedRDTSC() {
#if defined(__x86_64__)
    _mm_lfence();
    const u64 tsc = __rdtsc();
    _mm_lfence();
    return tsc;
#elif defined(__aarch64__)
    u64 value;
    asm volatile("isb\n\tmrs %0, cntvct_el0" : "=r"(value)::"memory");
    return value;
#else
#error "FencedRDTSC: unsupported architecture"
#endif
}

} // namespace Common
