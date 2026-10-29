// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: object motion vectors for temporal upscaling (docs/ROADMAP.md, step 1).
//
// G-buffer shaders preserve clip positions for the exact indexed range of each draw.
// Matching includes vertex streams, index contents and base offsets. A missing match falls
// back to camera motion. See motion_history.h for CPU bookkeeping and its regression test.

#pragma once

#include <array>
#include "upscale/motion_history.h"

#include "common/types.h"
#include "vulkan/vk_common.h"
#include "textures/image.h"

namespace Vulkan {

class Instance;
class Scheduler;
struct RenderState;

class ObjectMotion {
public:
    ObjectMotion(const Instance& instance, Scheduler& scheduler);
    ~ObjectMotion();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }

    using DrawInfo = Motion::Draw;
    u32 PrepareDraw(const DrawInfo& draw, Motion::Anchor anchor = {});
    /// Small skeletons: whether the bone palette changed (Motion::History::Moving).
    bool Moving(const Motion::History::GateKey& key, u64 palette, Motion::Anchor anchor = {}) {
        return history.Moving(key, palette, anchor);
    }
    /// Index range and topology hash of an indexed draw, reused across frames.
    template <class Scan>
    Motion::IndexRangeCache::Result IndexRange(const Motion::IndexRangeCache::Key& key,
                                               Scan&& scan) {
        return index_ranges.Get(key, frame, scan);
    }

    /// Rendering of a motion pipeline: the motion attachment (render-target size).
    void Attach(RenderState& state, u32 width, u32 height);

    /// Start of a frame (display pass).
    void OnFrameStart();
    /// BB_MOTION_TRACE=<file> (diagnostics): the frame after the file appears prints its motion draws.
    [[nodiscard]] bool Tracing() const noexcept { return tracing; }
    void InvalidateHistory() { history.NextFrame(); history.NextFrame(); }

    /// The motion image for the compose pass (layout General after this call) and whether it
    /// holds this frame's vectors at `width` x `height`.
    vk::ImageView PrepareRead(u32 width, u32 height, bool& valid);
    /// The image of the last PrepareRead (layout General), or null.
    [[nodiscard]] vk::ImageView View() const noexcept {
        return read_target && read_target->view ? *read_target->view : vk::ImageView{};
    }
    [[nodiscard]] vk::Image Image(u32 width, u32 height) const noexcept {
        const Target* target = Find(width, height);
        return target && target->written ? vk::Image(target->image) : vk::Image{};
    }

private:
    /// The params (host) and positions (device) buffers; at last session's addresses when the
    /// device can replay them.
    bool CreateBuffers(vk::DeviceSize params_size, vk::DeviceSize positions_size, void*& mapped,
                       u64& params_address, u64& positions_address);

    /// bbport: a motion image per render-target size. Scenes with G-buffer passes at two sizes
    /// (a reduced proxy and the full target) shared one image: every switch recreated it after
    /// a full GPU wait (Scheduler::Finish), twice a frame — ~70% of the draw recording thread's
    /// time and an idle GPU — and threw this frame's vectors away.
    struct Target {
        VideoCore::HostImage image;
        vk::UniqueImageView view;
        u32 width = 0, height = 0;
        bool written = false; ///< this frame's vectors are in the image
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
        u64 used_frame = 0;
    };
    static constexpr u32 MaxTargets = 4;

    [[nodiscard]] const Target* Find(u32 width, u32 height) const noexcept {
        for (const auto& target : targets) {
            if (target.image && target.width == width && target.height == height) {
                return &target;
            }
        }
        return nullptr;
    }
    /// The image of this size, created when missing (replacing the one used longest ago; it is
    /// destroyed once the GPU is done with it, without waiting).
    Target& GetTarget(u32 width, u32 height);

    const Instance& instance;
    Scheduler& scheduler;
    bool enabled = false;

    // Parameter ring (host visible): FrameSlots frames of ParamsPerFrame entries (two u32x4),
    // element 0 all zero (motion off).
    static constexpr u32 FrameSlots = 4;
    static constexpr u32 ParamsPerFrame = 8192;
    // Positions (device local): two halves (current/previous frame) of vec4; element 0 is reserved.
    // 4 Mi; 1 Mi on cards with 4 GB or less (128 MiB of VRAM there, 32: a scene draws ~1.3 Mi, the
    // rest falls back to camera motion). BB_MOTION_VERTICES_MI=N: N Mi.
    const u32 PositionsPerFrame;
    Motion::History history{PositionsPerFrame};
    Motion::IndexRangeCache index_ranges;
    std::array<u64, FrameSlots> params_ticks{};
    vk::Buffer params_buffer{};
    VmaAllocation params_allocation{};
    u32* params_mapped{};
    vk::Buffer positions_buffer{};
    VmaAllocation positions_allocation{};
    /// With buffer address replay (Instance::HasBufferAddressReplay) the two buffers have their
    /// own memory instead of VMA allocations.
    VkDeviceMemory params_memory{}, positions_memory{};

    u64 frame = 0;
    bool tracing = false;
    u32 params_used = 0;
    std::array<Target, MaxTargets> targets;
    const Target* read_target = nullptr; ///< of the last PrepareRead
};

} // namespace Vulkan
