// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: rounding to multiples.
#pragma once

#include <cstddef>
#include <type_traits>

namespace Common {

/// `value` rounded up to a multiple of `size` (any size).
template <typename T>
[[nodiscard]] constexpr T AlignUp(T value, std::size_t size) {
    static_assert(std::is_unsigned_v<T>, "AlignUp needs an unsigned type");
    const T rest = static_cast<T>(value % size);
    return rest == 0 ? value : static_cast<T>(value - rest + size);
}

/// `value` rounded down to a multiple of `size` (any size).
template <typename T>
[[nodiscard]] constexpr T AlignDown(T value, std::size_t size) {
    static_assert(std::is_unsigned_v<T>, "AlignDown needs an unsigned type");
    return static_cast<T>(value - value % size);
}

/// As AlignUp, `size` a power of two.
template <typename T>
[[nodiscard]] constexpr T AlignUpPow2(T value, T size) {
    return static_cast<T>((value + (size - 1)) & ~(size - 1));
}

/// As AlignDown, `size` a power of two.
template <typename T>
[[nodiscard]] constexpr T AlignDownPow2(T value, T size) {
    return static_cast<T>(value & ~(size - 1));
}

/// Whether `value` is a multiple of `alignment` (a power of two).
template <typename T>
    requires std::is_integral_v<T>
[[nodiscard]] constexpr bool IsAligned(T value, std::size_t alignment) {
    return (value & static_cast<T>(alignment - 1)) == 0;
}

template <typename T>
    requires std::is_integral_v<T>
[[nodiscard]] constexpr bool Is16KBAligned(T value) {
    return IsAligned(value, 16u << 10);
}

template <typename T>
    requires std::is_integral_v<T>
[[nodiscard]] constexpr bool Is64KBAligned(T value) {
    return IsAligned(value, 64u << 10);
}

template <typename T>
    requires std::is_integral_v<T>
[[nodiscard]] constexpr bool Is2MBAligned(T value) {
    return IsAligned(value, 2u << 20);
}

} // namespace Common
