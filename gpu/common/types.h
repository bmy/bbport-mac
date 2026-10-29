// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: fixed-width integer names and size literals used across the GPU library.
#pragma once

#include <array>
#include <cstdint>

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using s8 = std::int8_t;
using s16 = std::int16_t;
using s32 = std::int32_t;
using s64 = std::int64_t;
using f32 = float;
using f64 = double;
/// Two 64-bit halves, low first.
using u128 = std::array<u64, 2>;

/// Guest addresses: the game's memory is mapped at the same addresses in this process.
using VAddr = std::uintptr_t;
using PAddr = std::uintptr_t;

/// Calling convention of the game's code (System V on the PS4 as on Linux x86-64).
#define PS4_SYSV_ABI __attribute__((sysv_abi))

constexpr unsigned long long operator""_KB(unsigned long long n) {
    return n << 10;
}
constexpr unsigned long long operator""_MB(unsigned long long n) {
    return n << 20;
}
constexpr unsigned long long operator""_GB(unsigned long long n) {
    return n << 30;
}
