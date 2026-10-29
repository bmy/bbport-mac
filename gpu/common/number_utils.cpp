// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <bit>
#include <cmath>

#include "common/number_utils.h"

namespace NumberUtils {
namespace {

/// An unsigned small float: 5-bit exponent (bias 15) over a `mantissa_bits` mantissa.
float SmallUnsignedFloat(u32 value, u32 mantissa_bits) {
    const u32 mantissa = value & ((1u << mantissa_bits) - 1);
    const u32 exponent = (value >> mantissa_bits) & 0x1f;
    if (exponent == 0) {
        // Denormal: mantissa * 2^(1 - 15 - mantissa_bits).
        return std::ldexp(float(mantissa), 1 - 15 - int(mantissa_bits));
    }
    if (exponent == 31) {
        return std::bit_cast<float>(0x7f800000u | mantissa); // infinity, or NaN with a payload
    }
    const float fraction = 1.0f + float(mantissa) / float(1u << mantissa_bits);
    return std::ldexp(fraction, int(exponent) - 15);
}

/// value times 1 / (2^bits - 1) (the reciprocal in single precision, as the GPU tables are).
template <u32 Max, typename T>
float Normalized(T value) {
    constexpr float scale = 1.0f / float(Max);
    return float(value * scale);
}

} // namespace

float Uf11ToF32(u16 value) {
    return SmallUnsignedFloat(value, 6);
}

float Uf10ToF32(u16 value) {
    return SmallUnsignedFloat(value, 5);
}

float Uf16ToF32(u16 value) {
    // IEEE binary16 -> binary32.
    const u32 sign = u32(value & 0x8000) << 16;
    const u32 exponent = (value >> 10) & 0x1f;
    const u32 mantissa = value & 0x3ff;
    if (exponent == 0) {
        const float magnitude = std::ldexp(float(mantissa), -24);
        return sign ? -magnitude : magnitude;
    }
    if (exponent == 31) {
        return std::bit_cast<float>(sign | 0x7f800000u | (mantissa << 13));
    }
    return std::bit_cast<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

float U2ToUnorm(u8 value) {
    return Normalized<3>(value);
}
float S2ToSnorm(s8 value) {
    return Normalized<1>(value);
}
float U4ToUnorm(u8 value) {
    return Normalized<15>(value);
}
float S4ToSnorm(s8 value) {
    return Normalized<7>(value);
}
float U5ToUnorm(u8 value) {
    return Normalized<31>(value);
}
float S5ToSnorm(s8 value) {
    return Normalized<15>(value);
}
float U6ToUnorm(u8 value) {
    return Normalized<63>(value);
}
float S6ToSnorm(s8 value) {
    return Normalized<31>(value);
}
float U8ToUnorm(u8 value) {
    return Normalized<255>(value);
}
float S8ToSnorm(s8 value) {
    return Normalized<127>(value);
}
float U10ToUnorm(u16 value) {
    return Normalized<1023>(value);
}
float S10ToSnorm(s16 value) {
    return Normalized<511>(value);
}
float U16ToUnorm(u16 value) {
    return Normalized<65535>(value);
}
float S16ToSnorm(s16 value) {
    return Normalized<32767>(value);
}

} // namespace NumberUtils
