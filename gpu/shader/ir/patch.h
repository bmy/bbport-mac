// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: per-patch tessellation values: the tessellation factors, then 120
// generic components (30 vec4s).
#pragma once

#include <string>

#include <fmt/format.h>

#include "common/types.h"

namespace Shader::IR {

enum class Patch : u64 {
    TessellationLodLeft,
    TessellationLodTop,
    TessellationLodRight,
    TessellationLodBottom,
    TessellationLodInteriorU,
    TessellationLodInteriorV,
    Component0,
    Component119 = Component0 + 119,
};
static_assert(static_cast<u64>(Patch::Component119) == 125);

constexpr bool IsGeneric(Patch patch) noexcept {
    return patch >= Patch::Component0 && patch <= Patch::Component119;
}
constexpr Patch PatchFactor(u32 index) {
    return static_cast<Patch>(index);
}
constexpr Patch PatchGeneric(u32 index) {
    return static_cast<Patch>(static_cast<u32>(Patch::Component0) + index);
}
/// The vec4 of a generic component, and its element.
constexpr u32 GenericPatchIndex(Patch patch) {
    return (static_cast<u32>(patch) - static_cast<u32>(Patch::Component0)) / 4;
}
constexpr u32 GenericPatchElement(Patch patch) {
    return (static_cast<u32>(patch) - static_cast<u32>(Patch::Component0)) % 4;
}

[[nodiscard]] std::string NameOf(Patch patch);

} // namespace Shader::IR

template <>
struct fmt::formatter<Shader::IR::Patch> : fmt::formatter<std::string> {
    auto format(Shader::IR::Patch patch, format_context& ctx) const {
        return fmt::formatter<std::string>::format(Shader::IR::NameOf(patch), ctx);
    }
};
