// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the copy shader (the hardware VS run after a geometry shader) reads
// the GS output from the GS-VS ring and exports it. Parsing it gives the ring layout: which
// attribute and component each ring offset holds, the vertex count and component count.
#pragma once

#include <map>
#include <span>
#include <utility>

#include "common/types.h"
#include "shader/ir/attribute.h"

namespace Shader {

struct CopyShaderData {
    /// Ring dword offset (of vertex 0) -> (attribute, component).
    std::map<u32, std::pair<Shader::IR::Attribute, u32>> attr_map;
    u32 num_attrs{0};       ///< parameter exports
    u32 output_vertices{0}; ///< GS max output vertices (the ring stride per component / 64)
    u32 num_comps{0};       ///< components read
};

CopyShaderData ParseCopyShader(std::span<const u32> code);

} // namespace Shader
