// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: operations with their result and argument types (opcodes.inc).
#pragma once

#include <array>
#include <string_view>

#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

#include "common/types.h"
#include "shader/ir/type.h"

namespace Shader::IR {

enum class Opcode {
#define OPCODE(name, ...) name,
#include "shader/ir/opcodes.inc"
#undef OPCODE
};

constexpr size_t MaxArgs = 6;

namespace Detail {

struct OpcodeMeta {
    std::string_view name;
    Type type;
    std::array<Type, MaxArgs> args;
    u8 num_args;
};

/// A row of the table: the arguments are the types given.
template <typename... Args>
constexpr OpcodeMeta Meta(std::string_view name, Type type, Args... arg_types) {
    static_assert(sizeof...(Args) <= MaxArgs);
    return OpcodeMeta{name, type, {arg_types...}, u8(sizeof...(Args))};
}

constexpr auto MakeTable() {
    using enum Type;
    return std::array{
#define OPCODE(name, type, ...) Meta(#name, type __VA_OPT__(, ) __VA_ARGS__),
#include "shader/ir/opcodes.inc"
#undef OPCODE
    };
}

inline constexpr auto Table = MakeTable();

} // namespace Detail

/// The type of an operation's result.
[[nodiscard]] constexpr Type TypeOf(Opcode op) noexcept {
    return Detail::Table[static_cast<size_t>(op)].type;
}

/// The number of arguments (phis have any number: see Inst::NumArgs).
[[nodiscard]] constexpr size_t NumArgsOf(Opcode op) noexcept {
    return Detail::Table[static_cast<size_t>(op)].num_args;
}

/// The type an argument must have.
[[nodiscard]] constexpr Type ArgTypeOf(Opcode op, size_t arg_index) noexcept {
    return Detail::Table[static_cast<size_t>(op)].args[arg_index];
}

[[nodiscard]] constexpr std::string_view NameOf(Opcode op) noexcept {
    return Detail::Table[static_cast<size_t>(op)].name;
}

} // namespace Shader::IR

template <>
struct fmt::formatter<Shader::IR::Opcode> : fmt::formatter<std::string_view> {
    auto format(Shader::IR::Opcode op, format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(Shader::IR::NameOf(op), ctx);
    }
};

template <>
struct magic_enum::customize::enum_range<Shader::IR::Opcode> {
    static constexpr int min = 0;
    static constexpr int max = static_cast<int>(Shader::IR::Detail::Table.size()) - 1;
};
