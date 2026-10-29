// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: what a draw's pipeline is made of: the runtime info of its stages, the graphics
// and compute pipeline keys from the registers, and the stages bound. Split out of
// pipeline_cache.cpp (2026-10-26).

#include <algorithm>
#include "shim/bb_env.h"
#include <atomic>
#include <bit>
#include <chrono>
#include <xxhash.h>
#include <condition_variable>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <fstream>
#include <thread>
#include <unordered_set>

#ifdef _WIN32
#include <windows.h>
#else
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#include "bbport_sections.h"
#include "bbport_toggles.h"
#include "buffers/buffer_cache.h"
#include "common/hash.h"
#include "common/scope_exit.h"
#include "render/compile_log.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/serdes.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "cp/command_processor.h"
#include "game_profile.h"
#include "gcn/decoder.h"
#include "bbport_free_check.h"
#include "render/gcn_to_vk.h"
#include "render/blob_store.h"
#include "render/pipeline_cache.h"
#include "bbport_settings.h"
#include "render/pipeline_store.h"
#include "render/scheduler.h"
#include "render/shader_capture.h"
#include "render/vk_draw_prep.h"
#include "shader/backend/spirv/emit_spirv.h"
#include "shader/info.h"
#include "shader/runtime_info.h"
#include "upscale/motion_history.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"
#include "render/pipeline_cache_internal.h"

namespace Vulkan {

namespace {
/// The vertex outputs as the hardware lays them out (VS_OUT_CNTL): the misc vector, then the
/// clip/cull distance vectors that are enabled.
u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const Gcn::VsOutputControl& ctl) {
    u32 num_outputs = 0;
    if (ctl.vs_out_misc_enable) {
        auto& misc = outputs[num_outputs++];
        misc[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc[1] = ctl.use_vtx_edge_flag
                      ? Output::EdgeFlag
                      : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc[2] = ctl.use_vtx_kill_flag
                      ? Output::KillFlag
                      : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }
    const auto distance = [&](u32 index) {
        if (ctl.IsClipDistEnabled(index)) {
            return Output(u32(Output::ClipDist0) + index);
        }
        return ctl.IsCullDistEnabled(index) ? Output(u32(Output::CullDist0) + index)
                                            : Output::None;
    };
    const std::array<bool, 2> distance_vectors = {bool(ctl.vs_out_ccdist0_enable),
                                                  bool(ctl.vs_out_ccdist1_enable)};
    for (u32 vector = 0; vector < distance_vectors.size(); ++vector) {
        if (distance_vectors[vector]) {
            auto& map = outputs[num_outputs++];
            for (u32 component = 0; component < map.size(); ++component) {
                map[component] = distance(vector * 4 + component);
            }
        }
    }
    return num_outputs;
}

/// Clip distances (guest-exported, or user clip planes the vertex stage lowers) need the
/// fragment stage to discard where the host has no clip distances (NVIDIA): it then reads them
/// at input locations after the vertex outputs.
bool NeedsClipDistanceEmulation(const Gcn::Regs& regs, const Shader::Profile& profile) {
    const bool lowers_user_clip_planes =
        regs.clipper_control.user_clip_plane_enable &&
        !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry));
    return ((regs.vs_output_control.clip_distance_enable &&
             !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Local))) ||
            lowers_user_clip_planes) &&
           profile.needs_clip_distance_emulation;
}

bool IsDualSourceBlend(const Gcn::BlendControl& blend) {
    if (!blend.enable) {
        return false;
    }
    bool dual = GcnToVk::IsDualSourceBlendFactor(blend.color_dst_factor) ||
                GcnToVk::IsDualSourceBlendFactor(blend.color_src_factor);
    if (blend.separate_alpha_blend) {
        dual |= GcnToVk::IsDualSourceBlendFactor(blend.alpha_dst_factor) ||
                GcnToVk::IsDualSourceBlendFactor(blend.alpha_src_factor);
    }
    return dual;
}

Gcn::PrimitiveType GsInputPrimitive(const Gcn::Regs& regs) {
    if (regs.stage_enable.raw != Gcn::ShaderStageEnable::LsHsEsGs) {
        return regs.primitive_type;
    }
    switch (regs.tess_config.topology) {
    case Gcn::TessellationTopology::Point:
        return Gcn::PrimitiveType::PointList;
    case Gcn::TessellationTopology::Line:
        return Gcn::PrimitiveType::LineList;
    case Gcn::TessellationTopology::TriangleCw:
    case Gcn::TessellationTopology::TriangleCcw:
        return Gcn::PrimitiveType::TriangleList;
    default:
        UNREACHABLE();
    }
}
} // namespace

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(PipelineSelection& sel, HwStage stage,
                                                           SwStage l_stage) {
    auto& info = sel.runtime_infos[u32(l_stage)];
    const auto& regs = *sel.regs;
    const auto common = [&](const auto& program) {
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local: {
        common(regs.ls_program);
        // The hull stage is selected first: its constant buffer has the LS output stride.
        Shader::TessellationDataConstantBuffer tess_constants{};
        sel.infos[u32(SwStage::TessellationControl)]->ReadTessConstantBuffer(tess_constants);
        info.hw.ls.ls_stride = tess_constants.ls_stride;
        break;
    }
    case HwStage::Hull:
        common(regs.hs_program);
        break;
    case HwStage::Export:
        common(regs.es_program);
        info.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        break;
    case HwStage::Geometry: {
        common(regs.gs_program);
        auto& gs = info.hw.gs;
        gs.num_outputs = MapOutputs(gs.outputs, regs.vs_output_control);
        gs.output_vertices = regs.vgt_gs_max_vert_out;
        gs.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        gs.in_primitive = GsInputPrimitive(regs);
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            gs.out_primitive[stream_id] = regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        gs.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        gs.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        gs.mode = regs.vgt_gs_mode.mode;
        // The copy shader (the VS program) moves the GS ring's vertices to the outputs.
        const auto copy = Gcn::GetParams(regs.vs_program);
        gs.vs_copy = copy.code;
        gs.vs_copy_hash = copy.hash;
        DumpShader(gs.vs_copy, gs.vs_copy_hash, HwStage::Vertex, 0, "copy.bin");
        break;
    }
    case HwStage::Vertex: {
        common(regs.vs_program);
        auto& vs = info.hw.vs;
        vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        vs.num_outputs = MapOutputs(vs.outputs, regs.vs_output_control);
        vs.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == Gcn::ClipSpace::MinusWToW;
        vs.clip_disable = regs.IsClipDisabled();
        vs.motion_vectors = sel.motion;
        break;
    }
    case HwStage::Fragment: {
        common(regs.ps_program);
        auto& fs = info.hw.fs;
        fs.en_flags = regs.ps_input_ena;
        fs.addr_flags = regs.ps_input_addr;
        fs.num_inputs = regs.num_interp;
        fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
        fs.num_samples =
            regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
                ? regs.aa_config.NumSamples()
                : 1;
        fs.z_export_format = regs.z_export_format;
        fs.motion_vectors = sel.motion;
        const auto& depth = regs.depth_shader_control;
        const u8 stencil_export =
            depth.stencil_op_val_export_enable | depth.stencil_test_val_export_enable;
        fs.mrtz_mask = depth.z_export_enable | (stencil_export << 1) |
                       (depth.mask_export_enable << 2) | (depth.coverage_to_mask_enable << 3);
        fs.dual_source_blending = IsDualSourceBlend(regs.blend_control[0]);
        for (u32 i = 0; i < regs.num_interp; i++) {
            const auto& input = regs.ps_inputs[i];
            fs.inputs[i] = {
                .param_index = u8(input.input_offset),
                .is_default = bool(input.use_default),
                .is_flat = bool(input.flat_shade),
                .default_value = u8(input.default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            fs.color_buffers[i] = sel.graphics_key.color_buffers[i];
        }
        fs.clip_distance_emulation = NeedsClipDistanceEmulation(regs, profile);
        break;
    }
    case HwStage::Compute: {
        const auto& cs = liverpool->GetCsRegs();
        info.props.num_user_data = cs.settings.num_user_regs;
        info.props.num_allocated_vgprs = cs.settings.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs.settings.fp_round_mode64;
        info.hw.cs.workgroup_size = {cs.num_thread_x.full, cs.num_thread_y.full,
                                     cs.num_thread_z.full};
        info.hw.cs.tgid_enable = {cs.IsTgidEnabled(0), cs.IsTgidEnabled(1), cs.IsTgidEnabled(2)};
        info.hw.cs.shared_memory_size = cs.SharedMemSize();
        break;
    }
    default:
        break;
    }
    switch (l_stage) {
    case SwStage::Vertex:
        info.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.sw.vs.vertex_sgpr_offset = sel.draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = sel.draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == Gcn::PrimitiveType::RectList ||
            regs.primitive_type == Gcn::PrimitiveType::QuadList;
        break;
    case SwStage::TessellationControl:
        info.sw.tcs.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.sw.tcs.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.sw.tcs.tess_type = regs.tess_config.type;
        info.sw.tcs.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        break;
    case SwStage::TessellationEval:
        info.sw.tes.tess_type = regs.tess_config.type;
        info.sw.tes.tess_topology = regs.tess_config.topology;
        info.sw.tes.tess_partitioning = regs.tess_config.partitioning;
        break;
    default:
        break;
    }
    return info;
}

bool PipelineCache::RefreshGraphicsKey(PipelineSelection& sel) {
    std::memset(&sel.graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = *sel.regs;
    auto& key = sel.graphics_key;
    const auto& db = regs.depth_buffer;
    const bool db_enabled = db.DepthValid() || db.StencilValid();

    key.z_format = db.DepthValid() ? db.z_info.format : Gcn::DepthBuffer::ZFormat::Invalid;
    key.stencil_format =
        db.StencilValid() ? db.stencil_info.format : Gcn::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type = regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? db.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool color_disabled =
        regs.color_control.mode == Gcn::ColorControl::OperationMode::Disable;
    const auto bound = [&](u32 cb) {
        return !color_disabled && regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb);
    };

    // The color targets the fragment stage is translated for.
    u32 num_bound = 0;
    for (u32 cb = 0; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
        if (!bound(cb)) {
            continue;
        }
        ++num_bound;
        const auto& col_buf = regs.color_buffers[cb];
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = col_buf.GetDataFmt();
        color_buffer.num_format = col_buf.GetNumberFmt();
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();
        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == Gcn::BlendControl::BlendFunc::Min ||
             bc.color_func == Gcn::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == Gcn::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == Gcn::BlendControl::BlendFactor::DstColor;
    }

#ifdef __APPLE__
    // bbport (macOS): no target bound at or past the motion slot (none in a gap below it either:
    // with five bound, they are 0-4).
    const bool motion_slot_free = [&] {
        for (u32 cb = Shader::MotionVectors::Output; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
            if (bound(cb)) {
                return false;
            }
        }
        return true;
    }();
    if (num_bound >= 5 && !motion_slot_free) {
        if (static std::atomic_flag told; !told.test_and_set(std::memory_order_relaxed)) {
            std::printf("Object motion: a pass with %u targets binds one at or past slot %u: no "
                        "motion vectors for it\n", num_bound, Shader::MotionVectors::Output);
        }
    }
#endif
    // Only potentially animated G-buffer draws may use the extra motion attachment. The
    // ordinary stage variant is needed first to inspect the vertex shader's resources.
    // Not for FSR 1 (spatial: no motion vectors).
    const bool motion_possible =
        Shader::MotionVectors::positions_address != 0 && num_bound >= 5 &&
        !BbSettings::IsSpatial(BbSettings::Get().upscaler.load(std::memory_order_relaxed)) &&
#ifdef __APPLE__
        motion_slot_free &&
#else
        !regs.color_buffers[Shader::MotionVectors::Output] &&
#endif
        db.DepthValid() &&
        db.NumSamples() == 1 && !regs.IsClipDisabled() &&
        regs.stage_enable.raw == Gcn::ShaderStageEnable::VgtStages::Vs;
    sel.motion = false;

    if (!RefreshGraphicsStages(sel)) {
        return false;
    }
    if (motion_possible) {
        const auto* vs = sel.infos[static_cast<u32>(SwStage::Vertex)];
        if (vs) {
            // Shaders with a bone palette (motion_history.h). Small skeletons (weapons, props)
            // also include static world pieces: the rasterizer gives those history only while
            // their constants change.
            for (const auto& resource : vs->buffers) {
                if (resource.IsSpecial()) {
                    continue;
                }
                const auto buffer = resource.GetSharp(*vs);
                if (buffer.Valid() && buffer.GetStride() == 16 &&
                    Motion::ClassifyBuffer(buffer.GetSize()) != Motion::BufferRole::Other) {
                    sel.motion = true;
                    break;
                }
            }
        }
        // BB_MOTION_SELECT_LOG=1: each G-buffer vertex shader once, with its buffer sizes and
        // the selection, to find animated models that the size rule leaves out.
        static const bool select_log = bb_env_on("BB_MOTION_SELECT_LOG");
        if (select_log && vs) {
            std::string sizes;
            for (const auto& resource : vs->buffers) {
                if (resource.IsSpecial()) {
                    continue;
                }
                const auto buffer = resource.GetSharp(*vs);
                sizes += fmt::format(" {}/{}", buffer.GetSize(), buffer.GetStride());
            }
            static std::mutex log_mutex;
            static std::unordered_set<u64> logged;
            std::scoped_lock lock{log_mutex};
            if (logged.insert(vs->pgm_hash ^ std::hash<std::string>{}(sizes)).second) {
                std::printf("Motion select: vs %016llx %s, buffers (size/stride):%s\n",
                            static_cast<unsigned long long>(vs->pgm_hash),
                            sel.motion ? "ON " : "off", sizes.c_str());
            }
        }
        // BB_OBJECT_MOTION_ALL=1: the old broad selection, for visual A/B tests.
        static const bool all_motion = bb_env_on("BB_OBJECT_MOTION_ALL");
        if (all_motion) {
            sel.motion = true;
        }
        if (sel.motion && !RefreshGraphicsStages(sel)) {
            return false;
        }
    }

    // Targets the fragment stage does not write are left out; the others get their blending,
    // write mask and sample count.
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (u32 cb = 0; cb < key.num_color_attachments; ++cb) {
        if (!bound(cb)) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }
        const auto& col_buf = regs.color_buffers[cb];
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }
        key.write_masks[cb] = vk::ColorComponentFlags{
            key.color_buffers[cb].swizzle.ApplyMask(regs.color_target_mask.GetMask(cb))};
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    if (sel.motion) {
        constexpr u32 mv = Shader::MotionVectors::Output;
        key.motion_vectors = 1;
        key.mrt_mask |= 1u << mv;
        key.num_color_attachments = mv + 1;
        auto& color_buffer = key.color_buffers[mv];
        // R32G32_UINT: the vector as two halves, the depth with the "written" flag in its sign
        // (8 bytes a pixel of the G-buffer instead of 16: EmitFragmentMotion).
        color_buffer.data_format = Gcn::DataFormat::Format32_32;
        color_buffer.num_format = Gcn::NumberFormat::Uint;
        color_buffer.swizzle = Gcn::IdentityMapping;
#ifdef __APPLE__
        // bbport (macOS): all channels (a two-channel format): KosmicKrisp applies a partial mask
        // in the shader through its framebuffer-fetch path, the one thing only this target had.
        key.write_masks[mv] = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
#else
        key.write_masks[mv] = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG;
#endif
        key.color_samples[mv] = 1;
    }

    // Color sample counts the host cannot mix with each other or with the depth buffer's all
    // take the depth buffer's.
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if ((!all_color_samples_same && !instance.IsMixedAnySamplesSupported()) ||
            (all_color_samples_same && depth_mismatch &&
             !instance.IsMixedDepthSamplesSupported())) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }
    return true;
}

bool PipelineCache::RefreshGraphicsStages(PipelineSelection& sel) {
    bool not_a_program = false;
    const auto& regs = *sel.regs;
    auto& key = sel.graphics_key;
    sel.fetch_shader = std::nullopt;
    sel.infos.fill(nullptr);
    sel.modules.fill(nullptr);

    Shader::Backend::Bindings binding{};
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto in = static_cast<u32>(stage_in);
        const auto out = static_cast<u32>(stage_out);
        const auto* pgm = regs.stage_enable.IsStageEnabled(in) ? regs.ProgramForStage(in) : nullptr;
        // A geometry stage also runs the copy shader (the VS program).
        const auto* copy = stage_in == HwStage::Geometry && pgm ? &regs.vs_program : nullptr;
        if (!pgm || !pgm->Address<u32*>() ||
            !LooksLikeProgram(pgm->Address<u32*>(), stage_in, !sel.worker) ||
            (copy && (!copy->Address<u32*>() ||
                      !LooksLikeProgram(copy->Address<u32*>(), HwStage::Vertex, !sel.worker)))) {
            not_a_program |= pgm && pgm->Address<u32*>();
            key.stage_hashes[out] = 0;
            sel.infos[out] = nullptr;
            return false;
        }
        const auto params = Gcn::GetParams(*pgm);
        if (!sel.worker && !program_cache.contains(params.hash) && !ProgramDecodes(params, stage_in)) {
            not_a_program = true;
            key.stage_hashes[out] = 0;
            sel.infos[out] = nullptr;
            return false;
        }
        std::optional<Shader::Gcn::FetchShaderData> fetch_shader;
        std::tie(sel.infos[out], sel.modules[out], fetch_shader, key.stage_hashes[out]) =
            GetProgram(sel, stage_in, stage_out, params, binding);
        if (!sel.infos[out]) {
            not_a_program |= !sel.worker; // its fetch shader does not decode (GetProgram)
            key.stage_hashes[out] = 0;
            return false;
        }
        if (fetch_shader) {
            sel.fetch_shader = fetch_shader;
        }
        return true;
    };

    if (!bind_stage(HwStage::Fragment, SwStage::Fragment)) {
        // No pixel shader: the clip distance discard shader may still be needed, decided from
        // this draw's registers (not a previous draw's fragment stage).
        auto& fs = sel.runtime_infos[u32(SwStage::Fragment)];
        fs.Initialize(HwStage::Fragment, SwStage::Fragment);
        fs.hw.fs.clip_distance_emulation = NeedsClipDistanceEmulation(regs, profile);
    }
    const auto* fs_info = sel.infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);

    // The geometry stages of each hardware configuration, in selection order (the hull stage
    // before the local one: the LS output stride is in the hull stage's constants).
    // Static arrays: an initializer_list assigned here would point at a temporary array that ends
    // with its statement (clang reuses that stack: stages read as garbage).
    using Stage = std::pair<HwStage, SwStage>;
    static constexpr Stage EsGsStages[] = {{HwStage::Export, SwStage::Vertex},
                                           {HwStage::Geometry, SwStage::Geometry}};
    static constexpr Stage LsHsStages[] = {{HwStage::Hull, SwStage::TessellationControl},
                                           {HwStage::Vertex, SwStage::TessellationEval},
                                           {HwStage::Local, SwStage::Vertex}};
    static constexpr Stage LsHsEsGsStages[] = {{HwStage::Hull, SwStage::TessellationControl},
                                               {HwStage::Export, SwStage::TessellationEval},
                                               {HwStage::Local, SwStage::Vertex},
                                               {HwStage::Geometry, SwStage::Geometry}};
    using Vgt = Gcn::ShaderStageEnable::VgtStages;
    const auto mode = regs.stage_enable.raw;
    bool tessellation = false, geometry = false;
    std::span<const Stage> stages;
    switch (mode) {
    case Vgt::Vs:
        // A missing vertex program is drawn without one.
        bind_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    case Vgt::EsGs:
        geometry = true;
        stages = EsGsStages;
        break;
    case Vgt::LsHs:
        tessellation = true;
        stages = LsHsStages;
        break;
    case Vgt::LsHsEsGs:
        tessellation = geometry = true;
        stages = LsHsEsGsStages;
        break;
    default:
        LOG_WARNING(Render_Vulkan, "unimplemented shader stage {}", u32(mode));
        return false;
    }
    if (tessellation && !instance.IsTessellationSupported()) {
        return false;
    }
    if (geometry) {
        if (!instance.IsGeometryStageSupported()) {
#ifdef __APPLE__
            // bbport (macOS port): once per run (Metal drivers have no geometry stage; logging
            // every such draw cost frame time).
            if (static std::atomic_flag warned; !warned.test_and_set()) {
                LOG_WARNING(Render_Vulkan,
                            "Geometry shader stage unsupported, skipping draws that need it");
            }
#else
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
#endif
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
    }
    for (const auto& [hw, sw] : stages) {
        if (!bind_stage(hw, sw)) {
            return false;
        }
    }
    if (not_a_program) {
        return false;
    }

    // Without dynamic vertex input the pipeline is specialized on the attribute formats
    // (strides stay dynamic state).
    const auto* vs_info = sel.infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && sel.fetch_shader && !instance.IsVertexInputDynamicState()) {
        u32 vertex_binding = 0;
        for (const auto& attrib : sel.fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
#ifdef __APPLE__
            // bbport (macOS): an impossible format is a V# read while the game rewrote it (data
            // format 9 with number format 7, code 23 on KosmicKrisp, which has no dynamic vertex
            // input so the format is part of the key). As GetVertexInputs (pipeline.cpp) does at
            // draw time: 32-bit floats for this draw, so key and pipeline agree; ahead of the
            // draw, the GPU thread is left to prepare it (as PrepareResources does).
            vk::Format format =
                GcnToVk::TrySurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
            // FormatInvalid (a missing attribute, a zeroed V#) stays undefined, as before.
            if (format == vk::Format::eUndefined &&
                buffer.GetDataFmt() != Gcn::DataFormat::FormatInvalid) {
                if (sel.worker) {
                    return false;
                }
                static std::atomic<u32> told{0};
                if (told.fetch_add(1, std::memory_order_relaxed) < 4) {
                    LOG_WARNING(Render_Vulkan, "vertex attribute {} with data format {} and number "
                                "format {} in a pipeline key: read as 32-bit floats",
                                attrib.semantic, u32(buffer.GetDataFmt()),
                                u32(buffer.GetNumberFmt()));
                }
                format = vk::Format::eR32G32B32A32Sfloat;
            }
            key.vertex_buffer_formats[vertex_binding++] = format;
#else
            key.vertex_buffer_formats[vertex_binding++] =
                GcnToVk::SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
#endif
        }
    }
    return true;
}

bool PipelineCache::RefreshComputeKey() {
    Shader::Backend::Bindings binding{};
    const auto& cs = liverpool->GetCsRegs();
    if (!cs.Address<u32*>() || !LooksLikeProgram(cs.Address<u32*>(), HwStage::Compute, true)) {
        return false;
    }
    const auto cs_params = Gcn::GetParams(cs);
    if (!program_cache.contains(cs_params.hash) && !ProgramDecodes(cs_params, HwStage::Compute)) {
        return false;
    }
    std::tie(sel.infos[0], sel.modules[0], sel.fetch_shader, compute_key.value) =
        GetProgram(sel, HwStage::Compute, SwStage::Compute, cs_params, binding);
    return true;
}

} // namespace Vulkan
