// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: fmt support for enums (printed as their numbers) and UTF-8 strings.
#pragma once

#include <string>
#include <string_view>
#include <type_traits>

#include <fmt/format.h>

template <typename E>
struct fmt::formatter<E, std::enable_if_t<std::is_enum_v<E>, char>>
    : fmt::formatter<std::underlying_type_t<E>> {
    template <typename Context>
    auto format(const E& value, Context& ctx) const -> decltype(ctx.out()) {
        return fmt::formatter<std::underlying_type_t<E>>::format(
            static_cast<std::underlying_type_t<E>>(value), ctx);
    }
};

namespace fmt {
/// A u8 string printed as a char string.
template <typename T = std::string_view>
struct UTF {
    T data;
    explicit UTF(std::u8string_view text)
        : data{reinterpret_cast<const char*>(text.data()), text.size()} {}
    explicit UTF(const std::u8string& text) : UTF(std::u8string_view{text}) {}
};
} // namespace fmt

template <>
struct fmt::formatter<fmt::UTF<std::string_view>, char> : fmt::formatter<std::string_view> {
    template <typename Context>
    auto format(const fmt::UTF<std::string_view>& text, Context& ctx) const {
        return fmt::formatter<std::string_view>::format(text.data, ctx);
    }
};
