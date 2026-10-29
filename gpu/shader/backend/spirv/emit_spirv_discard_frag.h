// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader backend: a fragment shader that discards where an interpolated clip distance is
// negative (for hosts whose clip distance clipping is broken).
#pragma once

#include <array>
#include <vector>

#include "common/types.h"
#include "shader/runtime_info.h"

namespace Shader::Backend::SPIRV {

/// For the vertex shader's misc outputs (which components carry clip distances).
[[nodiscard]] std::vector<u32> EmitDiscardFragmentShader(std::array<Shader::OutputMap, 3> outputs);

} // namespace Shader::Backend::SPIRV
