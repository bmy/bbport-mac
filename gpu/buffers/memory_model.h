// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport buffers: how the GPU side reaches the game's memory. The GPU uses the game's direct
// memory where it is (its chunks are exported to or imported by Vulkan) and keeps what it reads
// often in VRAM copies; writers announce their writes (no page protection). A driver that cannot
// do this stops the game with the reason (GuestMemoryUnusable).
#pragma once

#include "common/types.h"

namespace VideoCore {

/// The driver cannot give the game's memory to the GPU in place: the game stops with `why`.
[[noreturn]] void GuestMemoryUnusable(const char* why);
/// BB_WRITE_VERIFY=1 (diagnostics): watched pages are protected as well, so writes nothing
/// announced show up as write faults.
bool WriteVerify();
/// BB_CONSTANTS_IN_PLACE=1: small read-only buffers (constants) are read where the game wrote
/// them, when the draw executes, instead of being copied when the draw is decoded or recorded.
/// Experiment bit 1 of BB_TOGGLE_FILE inverts it (A/B while the game runs).
bool ConstantsInPlace();

} // namespace VideoCore
