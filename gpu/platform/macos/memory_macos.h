// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Mac's memory as a whole, for the GPU budget, the log and the collectors.
// The Mac's memory is shared by the CPU and the GPU: bb-gpu (this process), the game process
// (bb-probe, its parent, under Rosetta) and macOS draw on the same RAM, and Metal's limit
// (recommendedMaxWorkingSetSize) knows nothing of the other two (issues #3, #6: 16 GB Macs swap).
#pragma once
#ifdef __APPLE__

#include "common/types.h"

namespace BbMacMemory {

/// The Mac's RAM (hw.memsize).
u64 PhysicalBytes();

/// BB_MEM_BUDGET=metal: the GPU budget is Metal's limit less this process's memory outside
/// graphics memory (as before). Default ("machine"): also at most the Mac's RAM less a reserve
/// for macOS (OSReserveBytes) and the game process's footprint.
bool MachineBudget();
/// What macOS keeps: at least 2.5 GiB, 20 % of the RAM. BB_MEM_RESERVE_MB=N: N MiB (tests).
u64 OSReserveBytes();

/// This process's physical footprint (what Activity Monitor shows) and its graphics part.
bool OwnFootprint(u64& footprint, u64& graphics);
/// The game process's footprint (bb-gpu's parent, when this is bb-gpu): 0 otherwise, or when
/// macOS does not say.
u64 GameProcessFootprint();

struct System {
    u64 free = 0;              ///< free pages
    u64 compressed = 0;        ///< RAM the compressor holds
    u64 compressed_data = 0;   ///< what that holds, uncompressed
    u64 swap_used = 0, swap_total = 0;
    int pressure_level = -1;   ///< kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
    int memorystatus_level = -1; ///< kern.memorystatus_level: percent of memory available
};
System ReadSystem();

/// BB_MEM_PRESSURE=0: off. Watches macOS's memory pressure notifications (warning, critical) on
/// a dispatch queue; nothing there touches the renderer: TakePressure hands what came to the
/// thread that polls it (the texture collector).
void WatchPressure();
/// The pressure events (DISPATCH_MEMORYPRESSURE_WARN / _CRITICAL bits) since the last call.
unsigned long TakePressure();

} // namespace BbMacMemory

#endif // __APPLE__
