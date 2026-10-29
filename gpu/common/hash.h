// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: combining hashes. The arithmetic is the usual golden-ratio mix (as boost's
// hash_combine); it must not change: pipeline cache keys on disk are made with it.
#pragma once

#include "common/types.h"

[[nodiscard]] inline u64 HashCombine(const u64 seed, const u64 hash) {
    return seed ^ (hash + 0x9e3779b9 + (seed << 12) + (seed >> 4));
}

[[nodiscard]] inline u32 HashCombine(const u32 seed, const u32 hash) {
    return seed ^ (hash + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
