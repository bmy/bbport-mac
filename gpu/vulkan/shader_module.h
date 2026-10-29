// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport vulkan: shader modules from SPIR-V.
#pragma once

#include <span>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

/// A shader module of `code` (SPIR-V words); the caller destroys it.
vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device);

} // namespace Vulkan
