// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: indirect dispatches run from checked group counts.
//
// The game's indirect dispatches take their group counts from memory a shader wrote. Twice
// (issues #34 and #32, the PC memory model) those counts were garbage — 1.2 and 3.8 billion
// groups of the particle simulations — and the GPU, still running them after the kernel's
// timeout, was reset: device lost, the game gone. Bloodborne's real counts are at most a few
// hundred. Before each indirect dispatch a one-thread shader (indirect_guard.comp) copies the
// counts to a slot of our own, zero if they are implausible (past the device's limits or
// BB_INDIRECT_GUARD_MAX groups in all, default 4M), and the dispatch reads that slot. Skipped
// dispatches are counted and reported. BB_INDIRECT_GUARD=0: the game's counts as they are.
//
// Indirect draws the same way (indirect_draw_guard.comp): vertices x instances past
// BB_INDIRECT_GUARD_MAX_VERTICES (default 64M) run nothing (another game lost the device on draws of
// a billion indices). The check is a dispatch, so it ends the render pass the draw is in.
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <utility>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace VideoCore {
struct Buffer;
}

namespace Vulkan {

class Instance;
class Scheduler;

class IndirectGuard {
public:
    IndirectGuard(const Instance& instance, Scheduler& scheduler);
    ~IndirectGuard();

    static bool Enabled();

    /// Records the check of the arguments at `offset` in `args` (outside a render pass); returns
    /// the buffer and offset to dispatch from.
    std::pair<vk::Buffer, u64> CheckArgs(const VideoCore::Buffer& args, u64 offset);
    /// The same for `num_draws` indirect draws (`stride` bytes apart; indexed: 5 dwords each,
    /// else 4) outside a render pass; returns where the checked copies are, `words` * 4 bytes
    /// apart.
    std::pair<vk::Buffer, u64> CheckDrawArgs(const VideoCore::Buffer& args, u64 offset, u32 stride,
                                             u32 num_draws, bool indexed);

private:
    static constexpr u32 NumSlots = 1024;

    void Report();

    const Instance& instance;
    Scheduler& scheduler;
    std::unique_ptr<VideoCore::Buffer> checked; ///< NumSlots x 16 bytes of group counts
    std::unique_ptr<VideoCore::Buffer> skipped; ///< count and the last skipped counts (host)
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
    u32 max_count[3]{};
    u32 max_total = 0;
    std::atomic<u32> next_slot{0};
    static constexpr u32 NumDrawSlots = 16384; ///< draws' checked arguments, 5 dwords each
    std::unique_ptr<VideoCore::Buffer> checked_draws;
    vk::UniquePipeline draw_pipeline;
    u32 max_vertices = 0;
    u32 next_draw_slot = 0;
    std::atomic<u64> draws_checked{0};
    std::atomic<u32> reported{0};
    std::atomic<s64> last_report_ns{0};
};

} // namespace Vulkan
