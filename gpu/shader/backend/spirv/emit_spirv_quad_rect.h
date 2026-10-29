// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader backend: tessellation shaders that draw GCN's rect and quad lists (Vulkan has
// no such primitives): rects are triangles with the fourth vertex interpolated, quads pass
// four control points to a passthrough evaluation shader.
#pragma once

#include <vector>

#include "common/types.h"

namespace Shader {
struct HwFragmentRuntimeInfo;
}

namespace Shader::Backend::SPIRV {

enum class AuxShaderType : u32 {
    RectListTCS,
    QuadListTCS,
    PassthroughTES,
};

/// The shader for the fragment shader's inputs (passed through or interpolated).
[[nodiscard]] std::vector<u32> EmitAuxilaryTessShader(AuxShaderType type,
                                                      const HwFragmentRuntimeInfo& fs_info);

} // namespace Shader::Backend::SPIRV
