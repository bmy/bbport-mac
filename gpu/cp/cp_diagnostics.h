// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport cp: diagnostics of the command stream. BB_DCB_STATS=1: the structure of the graphics
// command buffers every 5 s (input for decoding them on several threads); bad packets reported
// with their surroundings; BB_FREE_CHECK: submissions the guest changed before they were decoded.
#pragma once

#include <chrono>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "common/types.h"

namespace BbCp::Diagnostics {

struct SubmittedCopy;
/// BB_GNM_WRITES_TRACE: retain bounded recent snapshots to identify reused command storage.
void RememberSubmittedCopy(std::shared_ptr<const SubmittedCopy> copy);

/// A type-3 opcode's name (its number in hex if unknown).
std::string OpName(u32 op);

bool DcbStatsEnabled();
/// A graphics buffer about to be decoded (`depth` > 0: an indirect buffer).
void ScanDcb(std::span<const u32> dcb, int depth);
/// WAIT_REG_MEM outcomes: 0 display buffer label, 1 pending fence, 2 met at once, 3 waited.
void NoteWait(int outcome, double ms = 0);

/// A packet that cannot be decoded: where, in which buffer, what surrounds it.
void ReportBadPacket(uintptr_t base, std::size_t dwords, const u32* at, u64 seq, int depth,
                     const SubmittedCopy* copy = nullptr);

/// Where a label for the guest is (a fence, an end-of-pipe or release event's value).
enum class LabelStep : u8 {
    ComputeDecoded, ///< RELEASE_MEM decoded (compute)
    ComputeOnGpu,   ///< RELEASE_MEM recorded as a GPU write
    ComputeWritten, ///< RELEASE_MEM written by a CPU thread
    GraphicsOnGpu,  ///< EOP/EOS recorded as a GPU write
    GraphicsWritten ///< EOP/EOS written by a CPU thread
};
/// Kept in a ring of the last 256 (any thread; a few nanoseconds).
void NoteLabel(LabelStep step, u64 address, u64 value, u32 bytes);
/// The last `count` labels (oldest first), each with its age and what its memory holds now.
std::string RecentLabels(size_t count);
/// Labels among the last `count` whose memory holds another value than their newest write, not
/// zero (the guest's own reset): the emulator's recorded writes into each (BB_POOL_WATCH).
void DumpRolledBackLabels(size_t count);
/// Guest memory read through the backing view (no fault, any protection); false when unmapped.
bool ReadGuest(const void* address, void* out, u32 bytes);

/// A submission's command buffers as copied when submitted (BB_COPY_GPU_BUFFERS).
struct SubmittedCopy {
    std::vector<u32> data; ///< dcb, then ccb
    const u32* guest_dcb{};
    std::size_t dcb_dwords{};
    const u32* guest_ccb{};
    std::size_t ccb_dwords{};
    std::chrono::steady_clock::time_point submitted;
    std::size_t bad_dcb_at_submit = ~std::size_t(0);
    /// RepairSubmittedCopy changed it: its waits give up after BB_REPAIRED_WAIT_MS (500).
    bool repaired = false;
};
/// Record whether the submitted snapshot already contains a malformed top-level packet.
void InspectSubmittedCopy(SubmittedCopy& copy);
/// A copy malformed at submit with one packet short of its values: that packet made a NOP (true).
bool RepairSubmittedCopy(SubmittedCopy& copy, u64 number);
/// After it was decoded: whether the guest changed its command buffers since.
void CheckSubmittedCopy(const SubmittedCopy& copy, u64 seq);

} // namespace BbCp::Diagnostics
