// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport BB_PM4_SELFTEST=1: tests of the command processor packets Bloodborne never sends
// (occlusion queries, COPY_DATA, COND_EXEC, SET_PREDICATION, MEM_SEMAPHORE, GPU clock timestamps,
// a compute queue waiting for the graphics queue).
// Some 40 s into the game, packets are decoded before and after the game's own submissions, into
// memory of the test's own; a few seconds later the results are checked and printed
// ("PM4 self-test: ... PASS/FAIL").
#pragma once

#include <functional>
#include <span>

#include "common/types.h"

namespace BbCp::Pm4SelfTest {

bool Enabled();

struct Injection {
    std::span<const u32> before; ///< decoded before the submission
    std::span<const u32> after;  ///< decoded after it
};

struct Hooks {
    /// The GPU's writes before this point visible in the game's memory there.
    std::function<void(VAddr, u64)> sync_for_cpu_read;
    /// Draws and dispatches skipped by predication so far.
    std::function<u64()> predicated_skips;
    /// Occlusion events translated so far.
    std::function<u64()> occlusion_events;
    /// Hands `words` (in the test's compute ring) to a compute queue of the test's own.
    std::function<void(VAddr ring, u32 ring_dwords, VAddr read_pointer, std::span<const u32>)>
        submit_compute;
};

/// A top-level submission of `num_dwords` is about to be decoded (GPU command thread).
Injection Next(u32 num_dwords, const Hooks& hooks);

} // namespace BbCp::Pm4SelfTest
