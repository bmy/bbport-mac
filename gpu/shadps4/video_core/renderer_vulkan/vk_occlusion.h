// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's occlusion queries (ZPASS_DONE events) as Vulkan occlusion queries.
//
// On the PS4 every depth block keeps a running count of the samples that pass the depth test; a
// ZPASS_DONE event has each block write its count, 16 bytes apart, with the valid bit (63). The
// game issues one event before the draws it measures and one after, and takes the difference
// (Bloodborne: ~60 events a frame).
//
// Here the samples are counted by a chain of Vulkan occlusion query segments: one is active at
// all times once the game has used them, and a segment ends at every event and at every boundary
// a query may not span (a render pass beginning or ending, a command buffer: Scheduler::
// CarriedScope). An event remembers how many segments ended before it. Where no render pass is
// open (the next boundary), the segments' results are copied and one dispatch
// (occlusion_resolve.comp) adds them up in order, writing each pending event's running total into
// the game's memory through its device address. Results land a little after the PS4 would write
// them; until then their valid bits are clear, which the game checks as on the PS4.
#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace VideoCore {
class BufferCache;
struct Buffer;
} // namespace VideoCore

namespace Vulkan {

class Instance;

class OcclusionQueries final : public Scheduler::CarriedScope {
public:
    OcclusionQueries(const Instance& instance, Scheduler& scheduler,
                     VideoCore::BufferCache& buffer_cache);
    ~OcclusionQueries();

    /// A ZPASS_DONE event writing the counters of `pairs` depth blocks at `address` (recording
    /// thread, in stream order).
    void Event(VAddr address, u32 pairs);

    /// Events so far (BB_PM4_SELFTEST and the frame stats).
    u64 Events() const noexcept {
        return events;
    }

    void Suspend(bool inside_render_pass) override;
    void Resume(bool inside_render_pass) override;

private:
    static constexpr u32 NumSlots = 4096;
    static constexpr u32 ResetBatch = 512;

    struct Pending {
        u64 address; ///< device address of the first counter
        u32 pairs;
        u64 upto; ///< segments ended before the event
    };

    void BeginSegment(bool inside_render_pass);
    void EndSegment();
    /// Outside a render pass: the ended segments added up, the pending events written.
    void Resolve();
    /// Outside a render pass: query slots reset ahead of the segments that will use them.
    void ResetAhead();

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::BufferCache& buffer_cache;
    std::recursive_mutex mutex;
    vk::UniqueQueryPool pool;
    std::unique_ptr<VideoCore::Buffer> results; ///< segment results copied for the resolve
    std::unique_ptr<VideoCore::Buffer> total;   ///< the running sample count (64-bit)
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
    std::vector<Pending> pending;
    u64 begun = 0;       ///< segments begun (the next one uses slot begun % NumSlots)
    u64 resolved = 0;    ///< segments added to the total
    u64 reset_until = 0; ///< slots below it (absolute) are reset and unused
    bool active = false;
    bool started = false; ///< the game has used occlusion queries
    bool total_cleared = false;
    u64 events = 0;
    u64 starved = 0; ///< segments not begun in a render pass for want of a reset slot
};

} // namespace Vulkan
