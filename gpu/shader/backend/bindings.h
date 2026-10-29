// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader backend: the next free binding slots while the stages of a pipeline are laid
// out one after the other.
#pragma once

#include "common/types.h"

namespace Shader::Backend {

struct Bindings {
    u32 unified{};   ///< descriptor binding index (all kinds)
    u32 buffer{};    ///< buffer index (push data offsets)
    u32 user_data{}; ///< user data register slot in the push data

    auto operator<=>(const Bindings&) const = default;
};

} // namespace Shader::Backend
