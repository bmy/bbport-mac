// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport memory: where the guest's faults on watched pages come from (frame statistics): which
// game functions write GPU memory and which textures they write, to replace page watching by
// explicit hooks. Lock-free tables, safe in the fault handler.
#pragma once

#include "common/types.h"

namespace BbMem::FaultSites {

/// A fault at `address` (the handler's signal context): counts its instruction and caller, and
/// keeps them as this thread's current fault.
void Note(void* context, VAddr address);
/// This thread's fault has been handled.
void Done();
/// The current fault of this thread hit an image (diagnostics).
void NoteImage(VAddr image_address, u64 image_size, u32 width, u32 height, u32 format,
               u32 tiling);
/// Prints and clears the sites of the last window.
void Report();

} // namespace BbMem::FaultSites
