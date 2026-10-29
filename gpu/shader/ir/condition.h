// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: branch conditions of the GCN program (scalar condition code, VCC
// and EXEC tested for zero).
#pragma once

#include <string_view>

#include <fmt/format.h>

#include "common/types.h"

namespace Shader::IR {

enum class Condition : u32 {
    False,
    True,
    Scc0,
    Scc1,
    Vccz,
    Vccnz,
    Execz,
    Execnz,
};

constexpr std::string_view NameOf(Condition condition) {
    constexpr std::string_view names[]{"False", "True",  "Scc0",  "Scc1",
                                       "Vccz",  "Vccnz", "Execz", "Execnz"};
    return u32(condition) < std::size(names) ? names[u32(condition)] : "?";
}

} // namespace Shader::IR

template <>
struct fmt::formatter<Shader::IR::Condition> : fmt::formatter<std::string_view> {
    auto format(Shader::IR::Condition condition, format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(NameOf(condition), ctx);
    }
};
