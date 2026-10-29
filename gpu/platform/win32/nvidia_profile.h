// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: the NVIDIA driver's application profile of this executable (nvidia_profile.cpp).

#pragma once

namespace BbNvidiaProfile {

/// Before the Vulkan driver loads: the driver profile of this executable gets "Power management
/// mode: Prefer maximum performance" unless the user chose a mode for it. A no-op without an
/// NVIDIA driver; BB_NV_POWER=0 leaves the profile alone.
void Apply();

} // namespace BbNvidiaProfile
