// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: the pipeline state set by commands rather than baked into pipelines
// (viewports, depth and stencil tests, rasterizer switches, blend constants...). Setters mark
// what changed; CommitWith hands each change to the recorder as a small command holding only
// its own values. A test's parameters wait while the test is off (they are committed when it is
// turned on).
#pragma once

#include <algorithm>
#include <array>

#include <boost/container/static_vector.hpp>

#include "common/types.h"
#include "gcn/regs_raster.h"
#include "gcn/regs_targets.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

using Viewports = boost::container::static_vector<vk::Viewport, Gcn::NUM_VIEWPORTS>;
using Scissors = boost::container::static_vector<vk::Rect2D, Gcn::NUM_VIEWPORTS>;
using ColorWriteMasks = std::array<vk::ColorComponentFlags, Gcn::NUM_COLOR_BUFFERS>;

/// BB_DYNSTATE_CHECK=1 (diagnostics): the scissor each command buffer really has, set by whom;
/// every draw compares it with what the state cache believed when the draw was decoded.
namespace DynCheck {
bool Enabled();
void Begin(vk::CommandBuffer cmd);
void Set(vk::CommandBuffer cmd, const Scissors& scissors, const char* who);
void SetViewports(vk::CommandBuffer cmd, const Viewports& viewports, const char* who);
void Draw(vk::CommandBuffer cmd, const Viewports& viewports, const Scissors& scissors, u64 vs, u64 ps);
} // namespace DynCheck

struct StencilOps {
    vk::StencilOp fail_op{};
    vk::StencilOp pass_op{};
    vk::StencilOp depth_fail_op{};
    vk::CompareOp compare_op{};

    bool operator==(const StencilOps&) const = default;
};

class DynamicState {
public:
    /// Marks everything changed (a new command buffer, or commands that set state themselves).
    void Invalidate() {
        dirty = AllBits;
    }

    /// The viewports and scissors only: set again at the start of every render pass.
    void InvalidateViewports() {
        dirty |= ViewportsBit | ScissorsBit;
    }

    /// Whether CommitWith would emit anything.
    [[nodiscard]] bool AnyDirty() const noexcept {
        return (dirty & ~Deferred()) != 0;
    }

    /// Passes each change to `emit` as a closure `void(vk::CommandBuffer)` and clears what it
    /// emitted. Unsupported states are dropped; a disabled test's parameters stay pending.
    template <typename Emit>
    void CommitWith(bool depth_bounds_supported, bool color_write_mask_supported,
                    bool feedback_loop_supported, Emit&& emit) {
        using Face = vk::StencilFaceFlagBits;
        const u32 pending = dirty & ~Deferred();
        if (pending == 0) {
            return;
        }
        const auto changed = [&](Bit bit) { return (pending & bit) != 0; };
        if (changed(ViewportsBit)) {
            emit([v = viewports](vk::CommandBuffer c) {
                c.setViewportWithCount(v);
                if (DynCheck::Enabled()) {
                    DynCheck::SetViewports(c, v, "cache");
                }
            });
        }
        if (changed(ScissorsBit)) {
            emit([v = scissors](vk::CommandBuffer c) {
                c.setScissorWithCount(v);
                if (DynCheck::Enabled()) {
                    DynCheck::Set(c, v, "cache");
                }
            });
        }
        if (changed(DepthTestBit)) {
            emit([v = depth_test](vk::CommandBuffer c) { c.setDepthTestEnable(v); });
        }
        if (changed(DepthWriteBit)) {
            // Set even with the test off: a command buffer must have it.
            emit([v = depth_write](vk::CommandBuffer c) { c.setDepthWriteEnable(v); });
        }
        if (changed(DepthCompareBit)) {
            emit([v = depth_compare](vk::CommandBuffer c) { c.setDepthCompareOp(v); });
        }
        if (changed(DepthBoundsTestBit) && depth_bounds_supported) {
            emit([v = depth_bounds_test](vk::CommandBuffer c) { c.setDepthBoundsTestEnable(v); });
        }
        if (changed(DepthBoundsBit) && depth_bounds_supported) {
            emit([lo = depth_bounds[0], hi = depth_bounds[1]](vk::CommandBuffer c) {
                c.setDepthBounds(lo, hi);
            });
        }
        if (changed(DepthBiasEnableBit)) {
            emit([v = depth_bias_enable](vk::CommandBuffer c) { c.setDepthBiasEnable(v); });
        }
        if (changed(DepthBiasBit)) {
            emit([b = depth_bias](vk::CommandBuffer c) { c.setDepthBias(b[0], b[1], b[2]); });
        }
        if (changed(StencilTestBit)) {
            emit([v = stencil_test](vk::CommandBuffer c) { c.setStencilTestEnable(v); });
        }
        // Per face, both faces in one command when both changed to the same value.
        const auto per_face = [&](Bit front_bit, Bit back_bit, const auto& front,
                                  const auto& back, auto set) {
            const bool f = changed(front_bit);
            const bool b = changed(back_bit);
            if (f && b && front == back) {
                emit([v = front, set](vk::CommandBuffer c) { set(c, Face::eFrontAndBack, v); });
                return;
            }
            if (f) {
                emit([v = front, set](vk::CommandBuffer c) { set(c, Face::eFront, v); });
            }
            if (b) {
                emit([v = back, set](vk::CommandBuffer c) { set(c, Face::eBack, v); });
            }
        };
        per_face(StencilFrontOpsBit, StencilBackOpsBit, stencil_ops[0], stencil_ops[1],
                 [](vk::CommandBuffer c, Face face, const StencilOps& o) {
                     c.setStencilOp(face, o.fail_op, o.pass_op, o.depth_fail_op, o.compare_op);
                 });
        per_face(StencilFrontRefBit, StencilBackRefBit, stencil_reference[0],
                 stencil_reference[1],
                 [](vk::CommandBuffer c, Face face, u32 v) { c.setStencilReference(face, v); });
        per_face(StencilFrontWriteBit, StencilBackWriteBit, stencil_write_mask[0],
                 stencil_write_mask[1],
                 [](vk::CommandBuffer c, Face face, u32 v) { c.setStencilWriteMask(face, v); });
        per_face(StencilFrontCompareBit, StencilBackCompareBit, stencil_compare_mask[0],
                 stencil_compare_mask[1],
                 [](vk::CommandBuffer c, Face face, u32 v) { c.setStencilCompareMask(face, v); });
        if (changed(PrimitiveRestartBit)) {
            emit([v = primitive_restart](vk::CommandBuffer c) { c.setPrimitiveRestartEnable(v); });
        }
        if (changed(RasterizerDiscardBit)) {
            emit([v = rasterizer_discard](vk::CommandBuffer c) {
                c.setRasterizerDiscardEnable(v);
            });
        }
        if (changed(CullModeBit)) {
            emit([v = cull_mode](vk::CommandBuffer c) { c.setCullMode(v); });
        }
        if (changed(FrontFaceBit)) {
            emit([v = front_face](vk::CommandBuffer c) { c.setFrontFace(v); });
        }
        if (changed(BlendConstantsBit)) {
            emit([v = blend_constants](vk::CommandBuffer c) { c.setBlendConstants(v.data()); });
        }
        if (changed(ColorWriteMasksBit) && color_write_mask_supported) {
            emit([v = color_write_masks](vk::CommandBuffer c) { c.setColorWriteMaskEXT(0, v); });
        }
        if (changed(LineWidthBit)) {
            emit([v = line_width](vk::CommandBuffer c) { c.setLineWidth(v); });
        }
        if (changed(FeedbackLoopBit) && feedback_loop_supported) {
            emit([v = feedback_loop](vk::CommandBuffer c) {
                c.setAttachmentFeedbackLoopEnableEXT(v ? vk::ImageAspectFlagBits::eColor
                                                       : vk::ImageAspectFlagBits::eNone);
            });
        }
        dirty &= ~pending;
    }

    void SetViewports(const Viewports& v) {
        if (!std::ranges::equal(viewports, v)) {
            viewports = v;
            dirty |= ViewportsBit;
        }
    }
    void SetScissors(const Scissors& v) {
        if (!std::ranges::equal(scissors, v)) {
            scissors = v;
            dirty |= ScissorsBit;
        }
    }
    void SetDepthTestEnabled(bool v) {
        Update(depth_test, v, DepthTestBit);
    }
    void SetDepthWriteEnabled(bool v) {
        Update(depth_write, v, DepthWriteBit);
    }
    void SetDepthCompareOp(vk::CompareOp v) {
        Update(depth_compare, v, DepthCompareBit);
    }
    void SetDepthBoundsTestEnabled(bool v) {
        Update(depth_bounds_test, v, DepthBoundsTestBit);
    }
    void SetDepthBounds(float min, float max) {
        Update(depth_bounds, std::array{min, max}, DepthBoundsBit);
    }
    void SetDepthBiasEnabled(bool v) {
        Update(depth_bias_enable, v, DepthBiasEnableBit);
    }
    void SetDepthBias(float constant, float clamp, float slope) {
        Update(depth_bias, std::array{constant, clamp, slope}, DepthBiasBit);
    }
    void SetStencilTestEnabled(bool v) {
        Update(stencil_test, v, StencilTestBit);
    }
    void SetStencilOps(const StencilOps& front, const StencilOps& back) {
        Update(stencil_ops[0], front, StencilFrontOpsBit);
        Update(stencil_ops[1], back, StencilBackOpsBit);
    }
    void SetStencilReferences(u32 front, u32 back) {
        Update(stencil_reference[0], front, StencilFrontRefBit);
        Update(stencil_reference[1], back, StencilBackRefBit);
    }
    void SetStencilWriteMasks(u32 front, u32 back) {
        Update(stencil_write_mask[0], front, StencilFrontWriteBit);
        Update(stencil_write_mask[1], back, StencilBackWriteBit);
    }
    void SetStencilCompareMasks(u32 front, u32 back) {
        Update(stencil_compare_mask[0], front, StencilFrontCompareBit);
        Update(stencil_compare_mask[1], back, StencilBackCompareBit);
    }
    void SetPrimitiveRestartEnabled(bool v) {
        Update(primitive_restart, v, PrimitiveRestartBit);
    }
    void SetRasterizerDiscardEnabled(bool v) {
        Update(rasterizer_discard, v, RasterizerDiscardBit);
    }
    void SetCullMode(vk::CullModeFlags v) {
        Update(cull_mode, v, CullModeBit);
    }
    void SetFrontFace(vk::FrontFace v) {
        Update(front_face, v, FrontFaceBit);
    }
    void SetBlendConstants(const std::array<float, 4>& v) {
        Update(blend_constants, v, BlendConstantsBit);
    }
    void SetColorWriteMasks(const ColorWriteMasks& v) {
        Update(color_write_masks, v, ColorWriteMasksBit);
    }
    void SetLineWidth(float v) {
        Update(line_width, v, LineWidthBit);
    }
    void SetAttachmentFeedbackLoopEnabled(bool v) {
        Update(feedback_loop, v, FeedbackLoopBit);
    }

private:
    enum Bit : u32 {
        ViewportsBit = 1u << 0,
        ScissorsBit = 1u << 1,
        DepthTestBit = 1u << 2,
        DepthWriteBit = 1u << 3,
        DepthCompareBit = 1u << 4,
        DepthBoundsTestBit = 1u << 5,
        DepthBoundsBit = 1u << 6,
        DepthBiasEnableBit = 1u << 7,
        DepthBiasBit = 1u << 8,
        StencilTestBit = 1u << 9,
        StencilFrontOpsBit = 1u << 10,
        StencilBackOpsBit = 1u << 11,
        StencilFrontRefBit = 1u << 12,
        StencilBackRefBit = 1u << 13,
        StencilFrontWriteBit = 1u << 14,
        StencilBackWriteBit = 1u << 15,
        StencilFrontCompareBit = 1u << 16,
        StencilBackCompareBit = 1u << 17,
        PrimitiveRestartBit = 1u << 18,
        RasterizerDiscardBit = 1u << 19,
        CullModeBit = 1u << 20,
        FrontFaceBit = 1u << 21,
        BlendConstantsBit = 1u << 22,
        ColorWriteMasksBit = 1u << 23,
        LineWidthBit = 1u << 24,
        FeedbackLoopBit = 1u << 25,
    };
    static constexpr u32 AllBits = (1u << 26) - 1;
    static constexpr u32 StencilParameters = StencilFrontOpsBit | StencilBackOpsBit |
                                             StencilFrontRefBit | StencilBackRefBit |
                                             StencilFrontWriteBit | StencilBackWriteBit |
                                             StencilFrontCompareBit | StencilBackCompareBit;

    /// Parameters of tests that are off: kept pending until the test is on.
    u32 Deferred() const noexcept {
        return (depth_test ? 0 : DepthCompareBit) | (depth_bounds_test ? 0 : DepthBoundsBit) |
               (depth_bias_enable ? 0 : DepthBiasBit) | (stencil_test ? 0 : StencilParameters);
    }

    template <typename T>
    void Update(T& field, const T& value, Bit bit) {
        if (!(field == value)) {
            field = value;
            dirty |= bit;
        }
    }

public:
    const Scissors& CurrentScissors() const {
        return scissors;
    }
    const Viewports& CurrentViewports() const {
        return viewports;
    }

private:
    u32 dirty = 0;
    Viewports viewports{};
    Scissors scissors{};
    bool depth_test = false;
    bool depth_write = false;
    vk::CompareOp depth_compare{};
    bool depth_bounds_test = false;
    std::array<float, 2> depth_bounds{};
    bool depth_bias_enable = false;
    std::array<float, 3> depth_bias{}; ///< constant, clamp, slope
    bool stencil_test = false;
    std::array<StencilOps, 2> stencil_ops{}; ///< front, back
    std::array<u32, 2> stencil_reference{};
    std::array<u32, 2> stencil_write_mask{};
    std::array<u32, 2> stencil_compare_mask{};
    bool primitive_restart = false;
    bool rasterizer_discard = false;
    vk::CullModeFlags cull_mode{};
    vk::FrontFace front_face{};
    std::array<float, 4> blend_constants{};
    ColorWriteMasks color_write_masks{};
    float line_width = 0.0f;
    bool feedback_loop = false;
};

} // namespace Vulkan
