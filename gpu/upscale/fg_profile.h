// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport upscale: BB_FG_PROFILE=1 (diagnostics): GPU time of each pass of FSR 3.1's frame
// generation (optical flow and interpolation), from timestamps written before each of its compute
// jobs (the FSR bridge's job hook), averaged and printed every 300 frames.
#pragma once

#include "vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace FgProfile {

/// Whether BB_FG_PROFILE is set.
bool Enabled();
/// Recording thread, around the frame generation's commands in `cmd`.
void Begin(const Vulkan::Instance& instance, vk::CommandBuffer cmd);
void End(vk::CommandBuffer cmd);

} // namespace FgProfile
