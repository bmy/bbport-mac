// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, the dynamic state of a draw (viewports and scissors, depth and
// stencil, primitive assembly, rasterization, blending), set on the scheduler's DynamicState
// (only what changed is recorded), and debug markers.

#include <mutex>
#include "shim/bb_env.h"
#include <unordered_map>

#include "render/raster_internal.h"

namespace Vulkan {

namespace {

/// PA_SC_SCREEN/WINDOW/GENERIC_SCISSOR intersected; the window offset moves the latter two.
Gcn::Scissor CombinedScissor(const Gcn::Regs& regs) {
    const bool offset = !regs.window_scissor.window_offset_disable;
    const s16 offset_x = offset ? regs.window_offset.window_x_offset : 0;
    const s16 offset_y = offset ? regs.window_offset.window_y_offset : 0;
    const auto top_left = [](s16 screen, s16 window, s16 generic, s16 window_offset) {
        return std::max({screen, s16(window + window_offset), s16(generic + window_offset)});
    };
    const auto bottom_right = [](s16 screen, s16 window, s16 generic, s16 window_offset) {
        return std::min({screen, s16(window + window_offset), s16(generic + window_offset)});
    };
    Gcn::Scissor scissor{};
    scissor.top_left_x =
        top_left(regs.screen_scissor.top_left_x, s16(regs.window_scissor.top_left_x),
                 s16(regs.generic_scissor.top_left_x), offset_x);
    scissor.top_left_y =
        top_left(regs.screen_scissor.top_left_y, s16(regs.window_scissor.top_left_y),
                 s16(regs.generic_scissor.top_left_y), offset_y);
    scissor.bottom_right_x =
        bottom_right(regs.screen_scissor.bottom_right_x, regs.window_scissor.bottom_right_x,
                     regs.generic_scissor.bottom_right_x, offset_x);
    scissor.bottom_right_y =
        bottom_right(regs.screen_scissor.bottom_right_y, regs.window_scissor.bottom_right_y,
                     regs.generic_scissor.bottom_right_y, offset_y);
    return scissor;
}

/// The depth range of a viewport. With [-W, W] clip space the hardware centers it on zoffset
/// (as RADV programs it: radv_pipeline_graphics.c, radv_cmd_buffer.c).
std::pair<float, float> DepthRange(const Gcn::Regs& regs, const Gcn::ViewportBounds& vp,
                                   bool unrestricted) {
    const auto& ctl = regs.viewport_control;
    const float zoffset = ctl.zoffset_enable ? vp.zoffset : 0.f;
    const float zscale = ctl.zscale_enable ? vp.zscale : 1.f;
    float min_depth = regs.clipper_control.clip_space == Gcn::ClipSpace::MinusWToW
                          ? zoffset - zscale
                          : zoffset;
    float max_depth = zoffset + zscale;
    if (!unrestricted) {
        min_depth = std::max(min_depth, 0.f);
        max_depth = std::min(max_depth, 1.f);
    }
    return {min_depth, max_depth};
}

bool IsListTopology(Gcn::PrimitiveType type) {
    switch (GcnToVk::PrimitiveType(type)) {
    case vk::PrimitiveTopology::ePointList:
    case vk::PrimitiveTopology::eLineList:
    case vk::PrimitiveTopology::eTriangleList:
    case vk::PrimitiveTopology::eLineListWithAdjacency:
    case vk::PrimitiveTopology::eTriangleListWithAdjacency:
        return true;
    default:
        return false;
    }
}

/// Patch lists, and the quad and rect lists drawn as patches (tessellation).
bool IsPatchListTopology(Gcn::PrimitiveType type) {
    return type == Gcn::PrimitiveType::PatchPrimitive || type == Gcn::PrimitiveType::QuadList ||
           type == Gcn::PrimitiveType::RectList;
}

/// A stencil face uses DB_STENCILREFMASK.STENCILOPVAL when one of its ops is REPLACE_OP.
bool UsesOpValue(Gcn::StencilFunc fail, Gcn::StencilFunc zpass, Gcn::StencilFunc zfail) {
    return fail == Gcn::StencilFunc::ReplaceOp || zpass == Gcn::StencilFunc::ReplaceOp ||
           zfail == Gcn::StencilFunc::ReplaceOp;
}

} // namespace

void Rasterizer::UpdateDynamicState(const GraphicsPipeline* pipeline, const bool is_indexed) const {
    BB_SECTION(UpdateDynamicState);
    UpdateViewportScissorState();
    UpdateDepthStencilState();
    UpdatePrimitiveState(is_indexed);
    UpdateRasterizationState();
    UpdateColorBlendingState(pipeline);
    scheduler.GetDynamicState().CommitWith(
        instance.IsDepthBoundsSupported(), instance.IsDynamicColorWriteMaskSupported(),
        instance.IsAttachmentFeedbackLoopLayoutSupported(),
        [&](auto&& command) { scheduler.Record(std::move(command)); });
}

void Rasterizer::UpdateViewportScissorState() const {
    const auto& regs = Regs();
    if (regs.polygon_control.enable_window_offset &&
        (regs.window_offset.window_x_offset != 0 || regs.window_offset.window_y_offset != 0)) {
        LOG_ERROR(Render_Vulkan,
                  "PA_SU_SC_MODE_CNTL.VTX_WINDOW_OFFSET_ENABLE support is not yet implemented.");
    }
    const Gcn::Scissor combined = CombinedScissor(regs);
    const auto& ctl = regs.viewport_control;
    const bool scaled = target_scale[0] != 1.0f || target_scale[1] != 1.0f;

    boost::container::static_vector<vk::Viewport, Gcn::NUM_VIEWPORTS> viewports;
    boost::container::static_vector<vk::Rect2D, Gcn::NUM_VIEWPORTS> scissors;
    for (u32 i = 0; i < Gcn::NUM_VIEWPORTS; i++) {
        const auto& vp = regs.viewports[i];
        if (vp.xscale == 0) {
            continue;
        }
        vk::Viewport viewport{};
        std::tie(viewport.minDepth, viewport.maxDepth) =
            DepthRange(regs, vp, instance.IsDepthRangeUnrestrictedSupported());
        if (regs.IsClipDisabled()) {
            // Screen-space positions: the translated vertex shader maps the hardware's whole
            // window range [0, 16383] to NDC, the viewport covers it.
            viewport.width = float(std::min<u32>(instance.GetMaxViewportWidth(), 16_KB));
            viewport.height = float(std::min<u32>(instance.GetMaxViewportHeight(), 16_KB));
        } else {
            const float xoffset = ctl.xoffset_enable ? vp.xoffset : 0.f;
            const float xscale = ctl.xscale_enable ? vp.xscale : 1.f;
            const float yoffset = ctl.yoffset_enable ? vp.yoffset : 0.f;
            const float yscale = ctl.yscale_enable ? vp.yscale : 1.f;
            // Scaled to the target drawn into (the upscaler's output-size images); the sub-pixel
            // jitter of scene geometry is the same shift as jittering the projection.
            viewport.x = (xoffset - xscale) * target_scale[0] + draw_jitter[0];
            viewport.y = (yoffset - yscale) * target_scale[1] + draw_jitter[1];
            viewport.width = xscale * 2.0f * target_scale[0];
            viewport.height = yscale * 2.0f * target_scale[1];
        }
        viewports.push_back(viewport);

        Gcn::Scissor scissor = combined;
        if (regs.mode_control.vport_scissor_enable) {
            const auto& own = regs.viewport_scissors[i];
            scissor.top_left_x = std::max(scissor.top_left_x, s16(own.top_left_x));
            scissor.top_left_y = std::max(scissor.top_left_y, s16(own.top_left_y));
            scissor.bottom_right_x =
                std::min(Gcn::Scissor::Clamp(scissor.bottom_right_x), own.bottom_right_x);
            scissor.bottom_right_y =
                std::min(Gcn::Scissor::Clamp(scissor.bottom_right_y), own.bottom_right_y);
        }
        if (scaled) {
            const auto scale = [](s32 v, float f) { return s32(std::lround(float(v) * f)); };
            const s32 x0 = scale(scissor.top_left_x, target_scale[0]);
            const s32 y0 = scale(scissor.top_left_y, target_scale[1]);
            const s32 x1 = scale(scissor.top_left_x + s32(scissor.GetWidth()), target_scale[0]);
            const s32 y1 = scale(scissor.top_left_y + s32(scissor.GetHeight()), target_scale[1]);
            scissors.push_back({
                .offset = {x0, y0},
                .extent = {u32(std::max(x1 - x0, 0)), u32(std::max(y1 - y0, 0))},
            });
        } else {
            scissors.push_back({
                .offset = {scissor.top_left_x, scissor.top_left_y},
                .extent = {scissor.GetWidth(), scissor.GetHeight()},
            });
        }
    }
    if (viewports.empty()) {
        // Vulkan needs one: an empty one outside the target.
        viewports.push_back({.x = -1.0f, .y = -1.0f, .width = 1.0f, .height = 1.0f,
                             .minDepth = 0.0f, .maxDepth = 1.0f});
        scissors.push_back({.offset = {0, 0}, .extent = {1, 1}});
    }
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetViewports(viewports);
    dynamic_state.SetScissors(scissors);
}

vk::Rect2D Rasterizer::ClearDrawRect(u32 width, u32 height) const {
    const auto& regs = Regs();
    const Gcn::Scissor combined = CombinedScissor(regs);
    s32 x0 = Gcn::Scissor::Clamp(combined.top_left_x);
    s32 y0 = Gcn::Scissor::Clamp(combined.top_left_y);
    s32 x1 = Gcn::Scissor::Clamp(combined.bottom_right_x);
    s32 y1 = Gcn::Scissor::Clamp(combined.bottom_right_y);
    if (regs.mode_control.vport_scissor_enable) {
        const auto& own = regs.viewport_scissors[0];
        x0 = std::max<s32>(x0, own.top_left_x);
        y0 = std::max<s32>(y0, own.top_left_y);
        x1 = std::min<s32>(x1, own.bottom_right_x);
        y1 = std::min<s32>(y1, own.bottom_right_y);
    }
    const auto& vp = regs.viewports[0];
    const auto& ctl = regs.viewport_control;
    if (!regs.IsClipDisabled() && vp.xscale != 0) {
        const float xoffset = ctl.xoffset_enable ? vp.xoffset : 0.f;
        const float xscale = std::abs(ctl.xscale_enable ? vp.xscale : 1.f);
        const float yoffset = ctl.yoffset_enable ? vp.yoffset : 0.f;
        const float yscale = std::abs(ctl.yscale_enable ? vp.yscale : 1.f);
        x0 = std::max(x0, s32(std::floor(xoffset - xscale)));
        y0 = std::max(y0, s32(std::floor(yoffset - yscale)));
        x1 = std::min(x1, s32(std::ceil(xoffset + xscale)));
        y1 = std::min(y1, s32(std::ceil(yoffset + yscale)));
    }
    x0 = std::clamp<s32>(x0, 0, s32(width));
    y0 = std::clamp<s32>(y0, 0, s32(height));
    x1 = std::clamp<s32>(x1, x0, s32(width));
    y1 = std::clamp<s32>(y1, y0, s32(height));
    return {.offset = {x0, y0}, .extent = {u32(x1 - x0), u32(y1 - y0)}};
}

void Rasterizer::UpdateDepthStencilState() const {
    const auto& regs = Regs();
    const auto& depth = regs.depth_control;
    auto& dynamic_state = scheduler.GetDynamicState();

    const bool depth_test = depth.depth_enable && regs.depth_buffer.DepthValid();
    dynamic_state.SetDepthTestEnabled(depth_test);
    if (depth_test) {
        // A depth clear pass writes the clear value, not the fragments' depth.
        dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable &&
                                           !regs.depth_render_control.depth_clear_enable);
        dynamic_state.SetDepthCompareOp(GcnToVk::CompareOp(depth.depth_func));
    }
    dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_enable);
    if (depth.depth_bounds_enable) {
        dynamic_state.SetDepthBounds(regs.depth_bounds_min, regs.depth_bounds_max);
    }
    const bool depth_bias = regs.polygon_control.NeedsBias();
    dynamic_state.SetDepthBiasEnabled(depth_bias);
    if (depth_bias) {
        const auto& offset = regs.poly_offset;
        const bool front = regs.polygon_control.enable_polygon_offset_front;
        dynamic_state.SetDepthBias(front ? offset.front_offset : offset.back_offset,
                                   offset.depth_bias,
                                   (front ? offset.front_scale : offset.back_scale) / 16.f);
    }

    const bool stencil_test = depth.stencil_enable && regs.depth_buffer.StencilValid();
    dynamic_state.SetStencilTestEnabled(stencil_test);
    if (!stencil_test) {
        return;
    }
    const auto& sc = regs.stencil_control;
    const bool two_sided = depth.backface_enable;
    const StencilOps front_ops{
        .fail_op = GcnToVk::StencilOp(sc.stencil_fail_front),
        .pass_op = GcnToVk::StencilOp(sc.stencil_zpass_front),
        .depth_fail_op = GcnToVk::StencilOp(sc.stencil_zfail_front),
        .compare_op = GcnToVk::CompareOp(depth.stencil_ref_func),
    };
    const StencilOps back_ops = two_sided ? StencilOps{
        .fail_op = GcnToVk::StencilOp(sc.stencil_fail_back),
        .pass_op = GcnToVk::StencilOp(sc.stencil_zpass_back),
        .depth_fail_op = GcnToVk::StencilOp(sc.stencil_zfail_back),
        .compare_op = GcnToVk::CompareOp(depth.stencil_bf_func),
    } : front_ops;
    dynamic_state.SetStencilOps(front_ops, back_ops);

    const auto front = regs.stencil_ref_front;
    const auto back = two_sided ? regs.stencil_ref_back : regs.stencil_ref_front;
    // Vulkan has one reference per face: a face replacing with STENCILOPVAL tests against it
    // too (a conflict only when the test values differ).
    const bool front_op =
        UsesOpValue(sc.stencil_fail_front, sc.stencil_zpass_front, sc.stencil_zfail_front);
    const bool back_op =
        two_sided ? UsesOpValue(sc.stencil_fail_back, sc.stencil_zpass_back, sc.stencil_zfail_back)
                  : front_op;
    const auto ref_conflict = [](Gcn::CompareFunc func, const Gcn::StencilRefMask& ref) {
        return func != Gcn::CompareFunc::Always && func != Gcn::CompareFunc::Never &&
               ref.stencil_test_val != ref.stencil_op_val;
    };
    if ((front_op && ref_conflict(depth.stencil_ref_func, front)) ||
        (back_op && two_sided && ref_conflict(depth.stencil_bf_func, back))) {
        LOG_WARNING(Render_Vulkan, "Stencil test requires test_val while ReplaceOp requires "
                                   "op_val; the stencil test will use op_val");
    }
    dynamic_state.SetStencilReferences(front_op ? front.stencil_op_val : front.stencil_test_val,
                                       back_op ? back.stencil_op_val : back.stencil_test_val);
    // A stencil clear pass writes the clear value only.
    const bool stencil_clear = regs.depth_render_control.stencil_clear_enable;
    dynamic_state.SetStencilWriteMasks(stencil_clear ? 0U : front.stencil_write_mask,
                                       stencil_clear ? 0U : back.stencil_write_mask);
    dynamic_state.SetStencilCompareMasks(front.stencil_mask, back.stencil_mask);
}

void Rasterizer::UpdatePrimitiveState(const bool is_indexed) const {
    const auto& regs = Regs();
    const auto type = regs.primitive_type;
    // Restart where the host supports it for the topology.
    const bool restart = (regs.enable_primitive_restart & 1) != 0 &&
                         (instance.IsListRestartSupported() || !IsListTopology(type)) &&
                         (instance.IsPatchListRestartSupported() || !IsPatchListTopology(type));
    ASSERT_MSG(!is_indexed || !restart || regs.primitive_restart_index == 0xFFFF ||
                   regs.primitive_restart_index == 0xFFFFFFFF,
               "Primitive restart index other than -1 is not supported yet");
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetPrimitiveRestartEnabled(restart);
    dynamic_state.SetRasterizerDiscardEnabled(regs.clipper_control.dx_rasterization_kill);
    dynamic_state.SetCullMode(GcnToVk::IsPrimitiveCulled(type)
                                  ? GcnToVk::CullMode(regs.polygon_control.CullingMode())
                                  : vk::CullModeFlagBits::eNone);
    dynamic_state.SetFrontFace(GcnToVk::FrontFace(regs.polygon_control.front_face));
}

void Rasterizer::UpdateRasterizationState() const {
    scheduler.GetDynamicState().SetLineWidth(Regs().line_control.Width());
}

void Rasterizer::UpdateColorBlendingState(const GraphicsPipeline* pipeline) const {
    auto& dynamic_state = scheduler.GetDynamicState();
    dynamic_state.SetBlendConstants(Regs().blend_constants);
    dynamic_state.SetColorWriteMasks(pipeline->GetGraphicsKey().write_masks);
    dynamic_state.SetAttachmentFeedbackLoopEnabled(attachment_feedback_loop);
}

namespace {
enum class MarkerOp { Begin, End, Insert };

/// Records a debug label (BB_VK_MARKERS=1). The guest's labels are not NUL-terminated.
void RecordMarker(Scheduler& scheduler, MarkerOp op, std::string_view text,
                  std::optional<u32> color = {}) {
    const std::string label{text};
    vk::DebugUtilsLabelEXT info{.pLabelName = label.c_str()};
    if (color) {
        const u32 c = *color;
        info.color = std::array<f32, 4>{f32((c >> 16) & 0xff) / 255.0f,
                                        f32((c >> 8) & 0xff) / 255.0f, f32(c & 0xff) / 255.0f,
                                        f32((c >> 24) & 0xff) / 255.0f};
    }
    const auto cmdbuf = scheduler.CommandBuffer();
    switch (op) {
    case MarkerOp::Begin:
        cmdbuf.beginDebugUtilsLabelEXT(info);
        break;
    case MarkerOp::End:
        cmdbuf.endDebugUtilsLabelEXT();
        break;
    case MarkerOp::Insert:
        cmdbuf.insertDebugUtilsLabelEXT(info);
        break;
    }
}

bool MarkersEnabled(bool from_guest) {
    return from_guest ? EmulatorSettings.IsVkGuestMarkersEnabled()
                      : EmulatorSettings.IsVkHostMarkersEnabled();
}
} // namespace

// The command buffer is the recording thread's while it has draws: drained first.
void Rasterizer::ScopeMarkerBegin(const std::string_view& str, bool from_guest) {
    if (MarkersEnabled(from_guest)) {
        DrainDrawPipe();
        RecordMarker(scheduler, MarkerOp::Begin, str);
    }
}

void Rasterizer::ScopeMarkerEnd(bool from_guest) {
    if (MarkersEnabled(from_guest)) {
        DrainDrawPipe();
        RecordMarker(scheduler, MarkerOp::End, {});
    }
}

void Rasterizer::ScopedMarkerInsert(const std::string_view& str, bool from_guest) {
    if (MarkersEnabled(from_guest)) {
        DrainDrawPipe();
        RecordMarker(scheduler, MarkerOp::Insert, str);
    }
}

void Rasterizer::ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                         bool from_guest) {
    if (MarkersEnabled(from_guest)) {
        DrainDrawPipe();
        RecordMarker(scheduler, MarkerOp::Insert, str, color);
    }
}

std::thread::id Rasterizer::GetGpuCommandProcessorThread() {
    return liverpool->GetGpuCommandProcessorThread();
}


} // namespace Vulkan

namespace Vulkan::DynCheck {
namespace {
/// Per command buffer, without locks (a global mutex hid the race): open addressing on the
/// handle; a buffer is recorded by one thread at a time, which alone touches its entry.
struct Shadow {
    std::atomic<VkCommandBuffer> key{VK_NULL_HANDLE};
    Viewports viewports;
    Scissors scissors;
    const char* viewports_by = "nothing";
    const char* scissors_by = "nothing";
};
constexpr u32 Slots = 4096;
Shadow shadows[Slots];
std::atomic<u64> draws{0}, reported{0};
Shadow* Find(vk::CommandBuffer cmd) {
    const VkCommandBuffer key = VkCommandBuffer(cmd);
    u32 at = u32((reinterpret_cast<uintptr_t>(key) * 0x9E3779B97F4A7C15ull) >> 52) & (Slots - 1);
    for (u32 i = 0; i < Slots; ++i, at = (at + 1) & (Slots - 1)) {
        VkCommandBuffer seen = shadows[at].key.load(std::memory_order_acquire);
        if (seen == key) {
            return &shadows[at];
        }
        if (seen == VK_NULL_HANDLE &&
            shadows[at].key.compare_exchange_strong(seen, key, std::memory_order_acq_rel)) {
            return &shadows[at];
        }
        if (seen == key) {
            return &shadows[at];
        }
    }
    return nullptr;
}
std::string Text(const Scissors& s) {
    std::string out;
    for (const auto& r : s) {
        out += fmt::format("({},{} {}x{})", r.offset.x, r.offset.y, r.extent.width, r.extent.height);
    }
    return out.empty() ? std::string{"none"} : out;
}
std::string Text(const Viewports& v) {
    std::string out;
    for (const auto& r : v) {
        out += fmt::format("({:g},{:g} {:g}x{:g})", r.x, r.y, r.width, r.height);
    }
    return out.empty() ? std::string{"none"} : out;
}
} // namespace

bool Enabled() {
    static const bool on = bb_env_on("BB_DYNSTATE_CHECK");
    return on;
}

void Begin(vk::CommandBuffer cmd) {
    if (Shadow* s = Find(cmd)) {
        s->viewports.clear();
        s->scissors.clear();
        s->viewports_by = s->scissors_by = "begin";
    }
}

void Set(vk::CommandBuffer cmd, const Scissors& scissors, const char* who) {
    if (Shadow* s = Find(cmd)) {
        s->scissors = scissors;
        s->scissors_by = who;
    }
}

void SetViewports(vk::CommandBuffer cmd, const Viewports& viewports, const char* who) {
    if (Shadow* s = Find(cmd)) {
        s->viewports = viewports;
        s->viewports_by = who;
    }
}

void Draw(vk::CommandBuffer cmd, const Viewports& viewports, const Scissors& scissors, u64 vs,
          u64 ps) {
    const u64 n = draws.fetch_add(1, std::memory_order_relaxed) + 1;
    const Shadow* s = Find(cmd);
    if (s && (!std::ranges::equal(s->viewports, viewports) ||
              !std::ranges::equal(s->scissors, scissors))) {
        if (reported.fetch_add(1, std::memory_order_relaxed) < 400) {
            std::printf("Dynstate check: frame %llu draw vs %016llx ps %016llx expects vp %s sc %s, "
                        "command buffer has vp %s (by %s) sc %s (by %s)\n",
                        (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed),
                        (unsigned long long)vs, (unsigned long long)ps, Text(viewports).c_str(),
                        Text(scissors).c_str(), Text(s->viewports).c_str(), s->viewports_by,
                        Text(s->scissors).c_str(), s->scissors_by);
        }
    }
    if (n % 1000000 == 0) {
        std::printf("Dynstate check: %llu draws, %llu mismatches\n", (unsigned long long)n,
                    (unsigned long long)reported.load(std::memory_order_relaxed));
    }
}
} // namespace Vulkan::DynCheck
