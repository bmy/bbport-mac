// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/patch.h"

namespace Shader::IR {

std::string NameOf(Patch patch) {
    static constexpr const char* factors[]{"TessellationLodLeft",     "TessellationLodTop",
                                           "TessellationLodRight",    "TessellationLodBottom",
                                           "TessellationLodInteriorU", "TessellationLodInteriorV"};
    if (patch < Patch::Component0) {
        return factors[u32(patch)];
    }
    return fmt::format("Component{}", u32(patch) - u32(Patch::Component0));
}

} // namespace Shader::IR
