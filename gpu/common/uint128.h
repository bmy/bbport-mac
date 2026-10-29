// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: 64-bit arithmetic through a 128-bit intermediate.
#pragma once

#include "common/types.h"

namespace Common {

/// a * b / d without overflowing in between (the quotient must fit 64 bits).
[[nodiscard]] inline u64 MultiplyAndDivide64(u64 a, u64 b, u64 d) {
    return u64(static_cast<unsigned __int128>(a) * b / d);
}

/// The high 64 bits of a * b.
[[nodiscard]] inline u64 MultiplyHigh(u64 a, u64 b) {
    return u64((static_cast<unsigned __int128>(a) * b) >> 64);
}

} // namespace Common
