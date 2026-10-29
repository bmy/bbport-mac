// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: whether a view of an image may use another format: both in the same format
// compatibility class of the Vulkan specification (same texel block size, or the same
// compressed or depth/stencil layout).
#pragma once

#include "vulkan/vk_common.h"

namespace VideoCore {

bool IsVulkanFormatCompatible(vk::Format base, vk::Format view);

} // namespace VideoCore
