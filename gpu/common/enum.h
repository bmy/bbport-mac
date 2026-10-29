// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: bit-flag operators for enums, and a small flag set.
#pragma once

#include <type_traits>
#include <utility>

#include "common/types.h"

/// Bitwise operators for an enum class used as flags, and True/False tests.
#define DECLARE_ENUM_FLAG_OPERATORS(E)                                                             \
    [[nodiscard]] constexpr E operator|(E a, E b) noexcept {                                       \
        return static_cast<E>(std::to_underlying(a) | std::to_underlying(b));                      \
    }                                                                                              \
    [[nodiscard]] constexpr E operator&(E a, E b) noexcept {                                       \
        return static_cast<E>(std::to_underlying(a) & std::to_underlying(b));                      \
    }                                                                                              \
    [[nodiscard]] constexpr E operator^(E a, E b) noexcept {                                       \
        return static_cast<E>(std::to_underlying(a) ^ std::to_underlying(b));                      \
    }                                                                                              \
    [[nodiscard]] constexpr E operator<<(E a, E b) noexcept {                                      \
        return static_cast<E>(std::to_underlying(a) << std::to_underlying(b));                     \
    }                                                                                              \
    [[nodiscard]] constexpr E operator>>(E a, E b) noexcept {                                      \
        return static_cast<E>(std::to_underlying(a) >> std::to_underlying(b));                     \
    }                                                                                              \
    [[nodiscard]] constexpr E operator~(E a) noexcept {                                            \
        return static_cast<E>(~std::to_underlying(a));                                             \
    }                                                                                              \
    constexpr E& operator|=(E& a, E b) noexcept {                                                  \
        return a = a | b;                                                                          \
    }                                                                                              \
    constexpr E& operator&=(E& a, E b) noexcept {                                                  \
        return a = a & b;                                                                          \
    }                                                                                              \
    constexpr E& operator^=(E& a, E b) noexcept {                                                  \
        return a = a ^ b;                                                                          \
    }                                                                                              \
    constexpr E& operator<<=(E& a, E b) noexcept {                                                 \
        return a = a << b;                                                                         \
    }                                                                                              \
    constexpr E& operator>>=(E& a, E b) noexcept {                                                 \
        return a = a >> b;                                                                         \
    }                                                                                              \
    [[nodiscard]] constexpr bool True(E a) noexcept {                                              \
        return std::to_underlying(a) != 0;                                                         \
    }                                                                                              \
    [[nodiscard]] constexpr bool False(E a) noexcept {                                             \
        return std::to_underlying(a) == 0;                                                         \
    }

namespace Common {

/// A set of flags, each enum value a bit position.
template <typename T>
class Flags {
public:
    using IntType = std::underlying_type_t<T>;

    Flags() = default;
    Flags(IntType raw) : bits_{raw} {}
    template <typename... More>
    Flags(T first, More... more) : bits_{Mask(first, more...)} {}

    template <typename... All>
    void set(All... flags) {
        bits_ |= Mask(flags...);
    }
    void set(Flags other) {
        bits_ |= other.bits_;
    }
    template <typename... All>
    void clr(All... flags) {
        bits_ &= ~Mask(flags...);
    }
    void clr(Flags other) {
        bits_ &= ~other.bits_;
    }
    template <typename... All>
    bool any(All... flags) const {
        return (bits_ & Mask(flags...)) != 0;
    }
    template <typename... All>
    bool all(All... flags) const {
        const IntType mask = Mask(flags...);
        return (bits_ & mask) == mask;
    }
    bool test(T flag) const {
        return any(flag);
    }
    bool isClear() const {
        return bits_ == 0;
    }
    void clrAll() {
        bits_ = 0;
    }
    u32 raw() const {
        return bits_;
    }
    Flags operator&(const Flags& o) const {
        return Flags(IntType(bits_ & o.bits_));
    }
    Flags operator|(const Flags& o) const {
        return Flags(IntType(bits_ | o.bits_));
    }
    Flags operator^(const Flags& o) const {
        return Flags(IntType(bits_ ^ o.bits_));
    }
    bool operator==(const Flags&) const = default;

private:
    template <typename... All>
    static IntType Mask(All... flags) {
        return IntType((IntType{0} | ... | IntType(IntType{1} << static_cast<IntType>(flags))));
    }
    IntType bits_ = 0;
};

} // namespace Common
