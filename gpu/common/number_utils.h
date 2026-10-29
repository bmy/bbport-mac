// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: small float and normalized-integer formats to float (border colors and the like).
#pragma once

#include "common/types.h"

namespace NumberUtils {

/// Unsigned 11-bit float (5-bit exponent, 6-bit mantissa), as in R11G11B10.
float Uf11ToF32(u16 value);
/// Unsigned 10-bit float (5-bit exponent, 5-bit mantissa).
float Uf10ToF32(u16 value);
/// IEEE half float.
float Uf16ToF32(u16 value);

/// n-bit unsigned normalized: value / (2^n - 1). Signed: value / (2^(n-1) - 1).
float U2ToUnorm(u8 value);
float S2ToSnorm(s8 value);
float U4ToUnorm(u8 value);
float S4ToSnorm(s8 value);
float U5ToUnorm(u8 value);
float S5ToSnorm(s8 value);
float U6ToUnorm(u8 value);
float S6ToSnorm(s8 value);
float U8ToUnorm(u8 value);
float S8ToSnorm(s8 value);
float U10ToUnorm(u16 value);
float S10ToSnorm(s16 value);
float U16ToUnorm(u16 value);
float S16ToSnorm(s16 value);

} // namespace NumberUtils
