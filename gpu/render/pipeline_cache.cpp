// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: program and pipeline selection (see pipeline_cache.h).

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <thread>
#include <unordered_set>

#include "bbport_sections.h"
#include "bbport_toggles.h"
#include "buffers/buffer_cache.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "core/emulator_settings.h"
#include "cp/command_processor.h"
#include "game_profile.h"
#include "gcn/decoder.h"
#include "render/gcn_to_vk.h"
#include "render/pipeline_cache.h"
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

namespace Vulkan {

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

// Shader/pipeline compile time on the GPU thread, reported by BB_FRAME_STATS.
std::atomic<u64> g_bb_compile_ns;
std::atomic<u32> g_bb_compiles;

namespace {

constexpr u32 SpirvVersion1_6 = 0x00010600U;

constexpr std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

void AddCompileTime(std::chrono::steady_clock::time_point start) {
    g_bb_compile_ns += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count());
}

/// bbport (macOS): how long a draw that needs its new pipeline at once (a full-screen pass, a
/// one-time render) waits for it, BB_SYNC_PIPELINE_WAIT_MS; past that it goes without it until
/// the pipeline is done (compiled in the background meanwhile). Metal's compiler service
/// sometimes takes many seconds for one pipeline (a 23 s freeze for two). -1: waits as long as
/// it takes (the default off macOS).
int SyncWaitMs() {
    static const int value = [] {
        if (const char* env = std::getenv("BB_SYNC_PIPELINE_WAIT_MS"); env && *env) {
            return std::max(-1, std::atoi(env));
        }
#ifdef __APPLE__
        return 250;
#else
        return -1;
#endif
    }();
    return value;
}

/// bbport: a compile of 250 ms or more, named in the log (the same pipelines run after run would
/// mean the driver's caches do not keep them).
void NoteSlowCompile(const char* what, const GraphicsPipelineKey* key, u64 hash,
                     std::chrono::steady_clock::time_point start) {
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();
    if (ms < 250.0) {
        return;
    }
    if (key) {
        std::printf("GPU: slow pipeline compile, %.0f ms: %s %#llx (stages %#llx %#llx %#llx "
                    "%#llx %#llx)\n",
                    ms, what, (unsigned long long)hash,
                    (unsigned long long)key->stage_hashes[0], (unsigned long long)key->stage_hashes[1],
                    (unsigned long long)key->stage_hashes[2], (unsigned long long)key->stage_hashes[3],
                    (unsigned long long)key->stage_hashes[4]);
    } else {
        std::printf("GPU: slow pipeline compile, %.0f ms: %s %#llx\n", ms, what,
                    (unsigned long long)hash);
    }
}

struct CompileTimer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~CompileTimer() {
        AddCompileTime(start);
        ++g_bb_compiles;
    }
};

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

/// The render target a draw goes to: its first color buffer written, else its depth buffer,
/// else 0.
u64 DrawTarget(const Gcn::Regs& regs) {
    if (regs.color_control.mode != Gcn::ColorControl::OperationMode::Disable) {
        for (u32 cb = 0; cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
            if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb)) {
                return regs.color_buffers[cb].Address();
            }
        }
    }
    return regs.depth_buffer.DepthValid() ? regs.depth_buffer.DepthAddress() : 0;
}

Shader::Profile MakeProfile(const Instance& instance, u32 sparse_page_shift) {
    const auto& vk12 = instance.GetVk12Properties();
    const auto driver = instance.GetDriverID();
    const bool nvidia = driver == vk::DriverId::eNvidiaProprietary;
    return Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12.shaderDenormPreserveFloat32),
        .support_fp32_denorm_flush = bool(vk12.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance.IsShaderSubgroupClockSupported(),
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() && nvidia,
        // Older NVIDIA (Pascal) has no barycentrics; BB_INTERP_INT_FIX=0/1 overrides.
        .needs_integer_interpolation_fix = [&] {
            if (const char* env = std::getenv("BB_INTERP_INT_FIX")) {
                return env[0] == '1';
            }
            return !instance.IsFragmentShaderBarycentricSupported() && nvidia;
        }(),
        .needs_lds_barriers = nvidia || driver == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = driver == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = nvidia,
        .supports_shader_stencil_export = instance.IsShaderStencilExportSupported(),
        // BB_LAYER_MEMORY: buffers over nearly all memory go through the page table.
        .paged_buffers = VideoCore::BufferCache::LayerPagedActive(),
        // The game's buffer copy shader, from its profile (games/).
        .buffer_copy_shader_hash = Game::BufferCopyShader(),
    };
}

} // namespace

/// BB_ASYNC_PIPELINES: worker threads creating graphics pipelines (the driver's compile,
/// vkCreateGraphicsPipelines, with its own VkPipelineCache synchronization). Jobs start in order.
class PipelineCompiler {
public:
    explicit PipelineCompiler(u32 threads) {
        for (u32 i = 0; i < threads; ++i) {
            workers.emplace_back([this] { Run(); });
        }
    }
    ~PipelineCompiler() {
        {
            std::scoped_lock lk{mutex};
            stop = true;
        }
        cv.notify_all();
        for (auto& worker : workers) {
            worker.join();
        }
    }
    /// `urgent`: a pipeline a draw waits for (bbport, macOS: SyncWaitMs), ahead of the queue.
    void Submit(std::function<void()> job, bool urgent = false) {
        {
            std::scoped_lock lk{mutex};
            if (urgent) {
                jobs.push_front(std::move(job));
            } else {
                jobs.push_back(std::move(job));
            }
        }
        cv.notify_one();
    }

private:
    void Run() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, [&] { return stop || !jobs.empty(); });
                if (jobs.empty()) {
                    return; // stopping, every job done
                }
                job = std::move(jobs.front());
                jobs.pop_front();
            }
            job();
        }
    }
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::function<void()>> jobs;
    std::vector<std::thread> workers;
    bool stop = false;
};

struct PipelineCache::PendingPipeline {
    std::mutex mutex;
    std::condition_variable cv;
    bool done = false;
    /// bbport (macOS): a draw that needed it at once gave up waiting (SyncWaitMs): the draws
    /// go without it until it is done.
    bool gave_up = false;
    std::unique_ptr<GraphicsPipeline> pipeline;
    GraphicsPipeline::SerializationSupport sdata{};
};

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             BbCp::CommandProcessor* liverpool_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes},
      profile{MakeProfile(instance_, sparse_page_shift)} {
    sel.regs = &liverpool->regs;
    auto [cache_result, cache] = instance.GetDevice().createPipelineCacheUnique({});
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
    WarmUp();
}

PipelineCache::~PipelineCache() = default;

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

bool PipelineCache::PrepareGraphicsPipeline(PipelineSelection& worker_sel) {
    // Tessellation stages read constant buffers from memory at selection time: not prepared.
    if (worker_sel.regs->stage_enable.hs_en) {
        return false;
    }
    return RefreshGraphicsKey(worker_sel) && !worker_sel.worker->failed;
}

namespace {
/// The fetch shader a vertex program calls (through a pointer in its user data) decodes up to
/// its s_setpc_b64: the parse asserts on anything else. Memory the game is still filling, or
/// reused, holds garbage there (another game: "illegal instruction" while loading).
bool FetchCodeDecodes(const u32* code) {
    const u32* pc = code;
    for (u32 i = 0; i < 512; ++i) {
        const auto encoding = GcnIsa::EncodingOf(*pc);
        if (encoding == GcnIsa::InstEncoding::ILLEGAL) {
            return false;
        }
        const bool vector = encoding == GcnIsa::InstEncoding::VOP1 ||
                            encoding == GcnIsa::InstEncoding::VOP2 ||
                            encoding == GcnIsa::InstEncoding::VOPC;
        if (vector && ((*pc & 0x1FF) == 0xF9 || (*pc & 0x1FF) == 0xFA)) {
            return false; // SDWA/DPP: not in a PS4 program
        }
        if (GcnIsa::Decode(pc).opcode == GcnIsa::Opcode::S_SETPC_B64) {
            return true;
        }
    }
    return false;
}
} // namespace

/// The GPU's writes before this draw, for a program that is not whole in memory (a copy into
/// place still on the GPU): at most once per address every 100 ms (thousands of draws a frame
/// use code that is simply not there; each wait is a whole GPU round trip). False: not tried.
bool PipelineCache::SyncProgramMemory(VAddr address, u64 size) {
    if (!program_memory_sync) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    if (const auto tried = program_sync_tried.find(address);
        tried != program_sync_tried.end() && now - tried->second < std::chrono::milliseconds(100)) {
        return false;
    }
    if (program_sync_tried.size() >= 4096) {
        program_sync_tried.clear();
    }
    program_sync_tried[address] = now;
    program_memory_sync(address, size);
    return true;
}

/// Every program the games' compilers emit starts with "s_mov_b32 vcc_hi, length" and ends in
/// an "OrbShdr" header. Anything else at a program address is not a program yet: the game copied
/// it into place with the GPU (DMA) and the CPU's view does not have it (or only its start), or
/// it freed the memory. The header search would find the next program's header (or none: an
/// assertion) and the translator would decode garbage. On the GPU command thread (`may_sync`)
/// what the GPU wrote there is read back first; still nothing: the draw is skipped. A draw
/// preparation worker gives such a draw to the GPU command thread.
bool PipelineCache::LooksLikeProgram(const u32* code, HwStage stage, bool may_sync) {
    if (may_sync && program_write_sync) {
        program_write_sync(reinterpret_cast<VAddr>(code), 256);
    }
    // Its header where its first instruction says: a header found further on (the search) may
    // be the next program's while this one is still being copied in.
    const auto complete = [code] { return Gcn::DirectBinaryInfo(code) != nullptr; };
    if (complete()) {
        return true;
    }
    const u32 first = code[0];
    if (may_sync && SyncProgramMemory(reinterpret_cast<VAddr>(code), 0x10000)) {
        if (complete()) {
            static std::atomic<u32> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_INFO(Render_Vulkan, "stage {} program at {} arrived from the GPU's writes",
                         u32(stage), fmt::ptr(code));
            }
            return true;
        }
        // Read back and still no header in place: a program laid out differently (the header
        // searched for, as before), else not a program.
        if (first == 0xBEEB03FF && code[0] == 0xBEEB03FF && Gcn::FindBinaryInfo(code)) {
            static std::atomic<u32> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Render_Vulkan,
                            "stage {} program at {}: its header is not where its first "
                            "instruction says; the next header found is used",
                            u32(stage), fmt::ptr(code));
            }
            return true;
        }
    }
    static std::atomic<u32> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
        LOG_WARNING(Render_Vulkan,
                    "stage {} program at {} is not a whole program ({:#010x} {:#010x} {:#010x} "
                    "{:#010x}, {:#010x} before a read back, header {}): draw skipped",
                    u32(stage), fmt::ptr(code), code[0], code[1], code[2], code[3], first,
                    Gcn::FindBinaryInfo(code) ? "found" : "missing");
    }
    return false;
}

/// A new program decodes whole before it is translated. A program the GPU copied into place
/// (DMA_DATA, CP DMA) may not be in the CPU's view of the memory yet: then what the GPU wrote is
/// read back once and the program decoded again. Still not decoding: the draw is skipped.
bool PipelineCache::ProgramDecodes(const Shader::ShaderParams& params, HwStage stage) {
    const auto first_illegal = [&]() -> s64 {
        const u32* pc = params.code.data();
        const u32* const end = pc + params.code.size();
        while (pc < end) {
            const auto encoding = GcnIsa::EncodingOf(*pc);
            if (encoding == GcnIsa::InstEncoding::ILLEGAL) {
                return pc - params.code.data();
            }
            // SDWA (0xF9) and DPP (0xFA) operands came with later GCN generations: in a PS4
            // program such a word is garbage, not an instruction.
            const bool vector = encoding == GcnIsa::InstEncoding::VOP1 ||
                                encoding == GcnIsa::InstEncoding::VOP2 ||
                                encoding == GcnIsa::InstEncoding::VOPC;
            if (vector && ((*pc & 0x1FF) == 0xF9 || (*pc & 0x1FF) == 0xFA)) {
                return pc - params.code.data();
            }
            GcnIsa::Decode(pc);
        }
        return -1;
    };
    s64 bad = first_illegal();
    if (bad < 0) {
        return true;
    }
    const VAddr address = reinterpret_cast<VAddr>(params.code.data());
    const u64 bytes = params.code.size_bytes();
    const u32 word = params.code[bad];
    const s64 at = bad;
    if (SyncProgramMemory(address, bytes)) {
        bad = first_illegal();
    }
    static std::atomic<u32> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
        LOG_WARNING(Render_Vulkan,
                    "stage {} program {:#x} at {:#x} ({} bytes): word {:#010x} at dword {} does "
                    "not decode; {}",
                    u32(stage), params.hash, address, bytes, word, at,
                    bad < 0 ? "decodes after reading back what the GPU wrote"
                            : "still not after a read back: draw skipped");
    }
    return bad < 0;
}

const GraphicsPipeline* PipelineCache::TryPreparedPipeline(const PreparedDraw& prepared) {
    BB_SECTION(APrepared);
    if (prepared.state.load(std::memory_order_acquire) != PreparedDraw::Ready ||
        prepared.reg_checksum != liverpool->gfx_reg_checksum) {
        return nullptr;
    }
    // Same registers; the stage programs and their flattened user data (which also covers
    // everything read from guest memory for the specialization) must match as well. This is
    // the per-stage work GetProgram does on the regular path.
    const auto& regs = liverpool->regs;
    for (u32 i = 0; i < prepared.num_stages; ++i) {
        const auto& stage = prepared.stages[i];
        const auto* pgm = regs.ProgramForStage(static_cast<u32>(stage.hw_stage));
        if (pgm && pgm->Address<u32*>() && program_write_sync) {
            program_write_sync(pgm->template Address<VAddr>(), 256);
        }
        if (!pgm || !pgm->Address<u32*>() || !Gcn::DirectBinaryInfo(pgm->Address<u32*>())) {
            return nullptr; // not a program: the regular path skips the draw
        }
        const auto params = Gcn::GetParams(*pgm);
        if (params.hash != stage.hash) {
            return nullptr;
        }
        auto& info = const_cast<Program*>(stage.program)->info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        info.RefreshFlatBuf();
        if (info.pgm_base != stage.pgm_base || info.flattened_ud_buf.size() != stage.flat_size ||
            std::memcmp(info.flattened_ud_buf.data(), stage.flat,
                        stage.flat_size * sizeof(u32)) != 0) {
            return nullptr;
        }
    }
    const auto it = graphics_pipelines.find(prepared.key);
    return it != graphics_pipelines.end() ? it->second.get() : nullptr;
}

void PipelineCache::NoteDrawTarget(const Gcn::Regs& regs) {
    const u32 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    const u64 target = DrawTarget(regs);
    if (target == 0 || (target == last_target && frame == last_target_frame)) {
        return;
    }
    last_target = target;
    last_target_frame = frame;
    auto& streak = target_streaks[target];
    if (streak.last_frame != frame) {
        streak.frames = streak.last_frame + 1 == frame ? streak.frames + 1 : 1;
        streak.last_frame = frame;
    }
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params,
                                                           const PreparedDraw* prepared,
                                                           bool indirect) {
    BB_SECTION(APipeline);
    used_prepared = nullptr;
    NoteDrawTarget(liverpool->regs);
    if (prepared) {
        if (const auto* pipeline = TryPreparedPipeline(*prepared)) {
            used_prepared = prepared;
            return pipeline;
        }
    }
    sel.draw_indirect_params = params;
    if (!RefreshGraphicsKey(sel)) {
        return nullptr;
    }
    if (const auto found = graphics_pipelines.find(sel.graphics_key);
        found != graphics_pipelines.end()) {
        return found->second.get();
    }
    const bool skippable = AsyncSkippable(sel, indirect);
    if (const auto pending = pending_graphics.find(sel.graphics_key);
        pending != pending_graphics.end()) {
        const std::shared_ptr<PendingPipeline> job = pending->second;
        {
            std::unique_lock lk{job->mutex};
            if (!job->done) {
                // bbport (macOS): a draw that gave up before and is redrawn every frame goes
                // without it; one-time renders and loading screens still wait for it.
                const int wait_ms = SyncWaitMs();
                const bool bounded = wait_ms >= 0 && redrawn_every_frame;
                if (skippable || (job->gave_up && bounded)) {
                    ++async_skipped;
                    return nullptr; // drawn from the frame its pipeline is ready
                }
                // A pass that is not drawn every frame needs it now: the wait is compile time.
                const auto start = std::chrono::steady_clock::now();
                if (bounded) {
                    if (!job->cv.wait_for(lk, std::chrono::milliseconds(wait_ms),
                                          [&] { return job->done; })) {
                        job->gave_up = true;
                        AddCompileTime(start);
                        ++async_gave_up;
                        return nullptr;
                    }
                } else {
                    job->cv.wait(lk, [&] { return job->done; });
                }
                AddCompileTime(start);
                ++async_waited;
            }
        }
        pending_graphics.erase(sel.graphics_key);
        return FinishPipeline(sel.graphics_key, *job);
    }
    if (skippable) {
        return CompileInBackground(sel);
    }
    if (const int wait_ms = SyncWaitMs(); wait_ms >= 0 && redrawn_every_frame) {
        // bbport (macOS): a full-screen or indirect draw to a target drawn every frame: compiled
        // in the background, waited for at most wait_ms (a frame without it is drawn again next
        // frame). One-time renders and loading screens still wait as long as it takes.
        const auto key = sel.graphics_key;
        CompileInBackground(sel, true);
        const std::shared_ptr<PendingPipeline> job = pending_graphics.find(key)->second;
        const auto start = std::chrono::steady_clock::now();
        {
            std::unique_lock lk{job->mutex};
            if (!job->cv.wait_for(lk, std::chrono::milliseconds(wait_ms),
                                  [&] { return job->done; })) {
                job->gave_up = true;
                AddCompileTime(start);
                ++async_gave_up;
                std::printf("GPU: pipeline %#llx needed at once not compiled after %d ms: drawn "
                            "without it until it is (BB_SYNC_PIPELINE_WAIT_MS)\n",
                            (unsigned long long)std::hash<GraphicsPipelineKey>{}(key), wait_ms);
                return nullptr;
            }
        }
        AddCompileTime(start);
        ++async_waited;
        pending_graphics.erase(key);
        return FinishPipeline(key, *job);
    }

    const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(sel.graphics_key);
    LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
    CompileTimer timer;
    GraphicsPipeline::SerializationSupport sdata{};
    auto& slot = graphics_pipelines[sel.graphics_key];
    slot = std::make_unique<GraphicsPipeline>(instance, scheduler, desc_heap, profile,
                                              sel.graphics_key, *pipeline_cache, sel.infos,
                                              sel.runtime_infos, sel.fetch_shader, sel.modules,
                                              sdata, false);
    NoteSlowCompile("graphics", &sel.graphics_key, pipeline_hash, timer.start);
    PipelineStore::SaveGraphics(sel.graphics_key, pipeline_hash, sdata);
    sel.fetch_shader.reset();
    return slot.get();
}

const GraphicsPipeline* PipelineCache::CompileInBackground(const PipelineSelection& selection,
                                                           bool urgent) {
    if (!compiler) {
        const char* env = std::getenv("BB_ASYNC_PIPELINE_THREADS");
        const u32 threads =
            std::max(env && *env ? u32(std::strtoul(env, nullptr, 10))
                                 : std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u),
                     1u);
        compiler = std::make_unique<PipelineCompiler>(threads);
        std::printf("GPU: new pipelines of passes drawn every frame compile on %u threads "
                    "(BB_ASYNC_PIPELINES=0: on the GPU thread, the game waits)\n",
                    threads);
    }
    auto job = std::make_shared<PendingPipeline>();
    pending_graphics.emplace(selection.graphics_key, job);
    ++async_started;
    ++g_bb_compiles;
    compiler->Submit([this, job, key = selection.graphics_key, infos = selection.infos,
                      runtime_infos = selection.runtime_infos, fetch = selection.fetch_shader,
                      modules = selection.modules]() mutable {
        GraphicsPipeline::SerializationSupport sdata{};
        const auto compile_start = std::chrono::steady_clock::now();
        auto pipeline = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, key, *pipeline_cache, infos, runtime_infos,
            fetch, modules, sdata, false);
        NoteSlowCompile("graphics (background)", &key, std::hash<GraphicsPipelineKey>{}(key),
                        compile_start);
        std::scoped_lock lk{job->mutex};
        job->pipeline = std::move(pipeline);
        job->sdata = std::move(sdata);
        job->done = true;
        job->cv.notify_all();
    }, urgent);
    sel.fetch_shader.reset();
    return nullptr;
}

const GraphicsPipeline* PipelineCache::FinishPipeline(const GraphicsPipelineKey& key,
                                                      PendingPipeline& job) {
    PipelineStore::SaveGraphics(key, std::hash<GraphicsPipelineKey>{}(key), job.sdata);
    auto& slot = graphics_pipelines[key];
    slot = std::move(job.pipeline);
    return slot.get();
}

bool PipelineCache::AsyncSkippable(const PipelineSelection& selection, bool indirect) {
    // Only a pass drawn every frame may go without a draw for a frame or two (its render target
    // drawn to in each of the last frames, below). A one-time render compiles at once, as during
    // loading screens (a frame of fewer than 300 draws before this one). The pipeline must not
    // depend on the draw's vertex buffers (VK_EXT_vertex_input_dynamic_state). A size rule (the
    // scene's, the output's) missed most passes: many set a 16384 scissor.
    static const bool enabled = [] {
        const char* env = std::getenv("BB_ASYNC_PIPELINES");
        return !(env && env[0] == '0');
    }();
    redrawn_every_frame = false;
    if (!enabled || !instance.IsVertexInputDynamicState() || !selection.regs) {
        return false;
    }
    const u32 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    const u64 draws = BbStats::draws.load(std::memory_order_relaxed);
    if (frame != stats_frame) {
        last_frame_draws = draws - frame_start_draws;
        frame_start_draws = draws;
        stats_frame = frame;
        const u32 second = BbStats::coarse_second.load(std::memory_order_relaxed);
        if (async_started != 0 && second - stats_printed >= 5) {
            stats_printed = second;
            std::printf("GPU: %llu pipelines compiled in the background, %llu draws went without "
                        "theirs meanwhile, %llu waited for one, %llu gave up waiting\n",
                        (unsigned long long)async_started, (unsigned long long)async_skipped,
                        (unsigned long long)async_waited, (unsigned long long)async_gave_up);
            async_started = async_skipped = async_waited = async_gave_up = 0;
        }
    }
    if (last_frame_draws < 300) {
        return false;
    }
    constexpr u32 PerFrameStreak = 8;
    const u64 target = DrawTarget(*selection.regs);
    const auto streak = target_streaks.find(target);
    const bool per_frame =
        target != 0 && streak != target_streaks.end() && streak->second.frames >= PerFrameStreak;
    redrawn_every_frame = per_frame;
    // A full-screen pass (one triangle or quad: lighting, fog, post-processing) waits for its
    // pipeline: going without one for a frame left whole frames grey. So does an indirect draw
    // (its size unknown here). What may go without is geometry, which shows a frame later.
    constexpr u32 FullScreenIndices = 12;
    if (indirect || selection.regs->num_indices <= FullScreenIndices) {
        return false;
    }
    // A pass drawn every frame: its render target drawn to in each of the last 8 frames (the
    // scene, the output, shadow maps, half-size and depth-only passes alike). A one-time render
    // (a face baked for a character, a texture made once) goes to a target new or used now and
    // then: compiled at once.
    return per_frame;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}",
                 std::hash<ComputePipelineKey>{}(compute_key));
        CompileTimer timer;
        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *sel.infos[0],
                                                       sel.modules[0], sdata, false);
        NoteSlowCompile("compute", nullptr, std::hash<ComputePipelineKey>{}(compute_key),
                        timer.start);
        PipelineStore::SaveCompute(compute_key, sdata);
    }
    return it->second.get();
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

    // Only potentially animated G-buffer draws may use the extra motion attachment. The
    // ordinary stage variant is needed first to inspect the vertex shader's resources.
    const bool motion_possible =
        Shader::MotionVectors::positions_address != 0 && num_bound >= 5 &&
        !regs.color_buffers[Shader::MotionVectors::Output] && db.DepthValid() &&
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
        static const bool select_log = [] {
            const char* value = std::getenv("BB_MOTION_SELECT_LOG");
            return value && value[0] == '1';
        }();
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
        static const bool all_motion = [] {
            const char* value = std::getenv("BB_OBJECT_MOTION_ALL");
            return value && value[0] == '1';
        }();
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
        color_buffer.data_format = Gcn::DataFormat::Format32_32_32_32;
        color_buffer.num_format = Gcn::NumberFormat::Float;
        color_buffer.swizzle = Gcn::IdentityMapping;
        key.write_masks[mv] = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
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
            // bbport (macOS port): once per run (Metal drivers have no geometry stage; logging
            // every such draw cost frame time).
            if (static std::atomic_flag warned; !warned.test_and_set()) {
                LOG_WARNING(Render_Vulkan,
                            "Geometry shader stage unsupported, skipping draws that need it");
            }
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

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              std::span<const u32> code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");
    CompileTimer timer;

    // BB_SHADER_CAPTURE: the inputs as they were, for the record.
    const bool capture = BbShaderCapture::Enabled();
    std::optional<Shader::Info> in_info;
    const auto in_runtime_info = runtime_info;
    const auto in_binding = binding;
    if (capture) {
        in_info.emplace(info);
        BbShaderCapture::Begin();
    }
    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    if (capture) {
        BbShaderCapture::Write(*in_info, in_runtime_info, profile, in_binding, code, perm_idx, spv,
                               info, binding);
    }
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");

    const vk::ShaderModule module = CompileSPV(spv, instance.GetDevice());
    PipelineStore::SaveShaderBinary(std::move(spv), info.pgm_hash, perm_idx);
    SetObjectName(instance.GetDevice(), module, GetShaderName(info.hw_stage, info.pgm_hash, perm_idx));
    return module;
}

PipelineCache::Result PipelineCache::LookUpProgram(PipelineSelection& sel,
                                                   const Shader::RuntimeInfo& runtime_info,
                                                   const Shader::ShaderParams& params,
                                                   Shader::Backend::Bindings& binding,
                                                   HwStage hw_stage) {
    // A draw-preparation worker only looks up, with its own copy of the program's Info.
    auto& worker = *sel.worker;
    std::shared_lock lk{programs_mutex};
    const auto found = program_cache.find(params.hash);
    if (found == program_cache.end() || !found->second->info_template) {
        worker.failed = true;
        return {};
    }
    const Program* program = found->second.get();
    auto& info = worker.infos.try_emplace(program, *program->info_template).first->second;
    info.pgm_base = params.Base();
    info.user_data = params.user_data;
    // The walk of resource tables and the fetch shader parse read guest memory through pointers
    // in the registers; ahead of the GPU thread that memory may already be reused. A fault
    // returns here (runtime_fault_recover) and the draw is left to the GPU thread. A jump out of
    // the specialization leaks its partial allocations (rare).
    sigjmp_buf recover;
    if (sigsetjmp(recover, 0)) {
        worker.failed = true;
        return {};
    }
    runtime_fault_recover = &recover;
    info.RefreshFlatBuf();
    if (info.has_fetch_shader &&
        !FetchCodeDecodes(Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base))) {
        runtime_fault_recover = nullptr;
        worker.failed = true; // the GPU thread reads back what the GPU wrote, or skips the draw
        return {};
    }
    const auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
    runtime_fault_recover = nullptr;
    const auto it = std::ranges::find(program->modules, spec, &Program::Module::spec);
    if (it == program->modules.end()) {
        worker.failed = true;
        return {};
    }
    info.AddBindings(binding);
    const size_t perm_idx = std::distance(program->modules.begin(), it);
    worker.stages.push_back({program, params.hash, hw_stage, info.pgm_base, &info.flattened_ud_buf});
    return {&info, it->module, it->spec.fetch_shader_data, HashCombine(params.hash, perm_idx)};
}

PipelineCache::Result PipelineCache::GetProgram(PipelineSelection& sel, HwStage hw_stage,
                                                SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    auto runtime_info = BuildRuntimeInfo(sel, hw_stage, sw_stage);
    if (sel.worker) {
        return LookUpProgram(sel, runtime_info, params, binding, hw_stage);
    }

    const auto found = program_cache.find(params.hash); // this thread is the only writer
    if (found == program_cache.end()) {
        auto new_program = std::make_unique<Program>(hw_stage, sw_stage, params);
        const auto start = binding;
        const auto module = CompileModule(new_program->info, runtime_info, params.code, 0, binding);
        auto spec = Shader::StageSpecialization(new_program->info, runtime_info, profile, start);
        const u64 perm_hash = HashCombine(params.hash, 0);
        PipelineStore::SaveShaderMeta(new_program->info, spec, perm_hash, 0);
        new_program->AddPermut(module, std::move(spec));
        new_program->info_template = std::make_unique<Shader::Info>(new_program->info);
        Program* program = new_program.get();
        {
            std::unique_lock lk{programs_mutex};
            program_cache.emplace(params.hash, std::move(new_program));
        }
        return {&program->info, module, program->modules[0].spec.fetch_shader_data, perm_hash};
    }

    Program& program = *found.value();
    if (!program.info_template) {
        // Programs preloaded from the pipeline cache get their template on first use.
        std::unique_lock lk{programs_mutex};
        program.info_template = std::make_unique<Shader::Info>(program.info);
    }
    auto& info = program.info;
    info.pgm_base = params.Base(); // the inline constant buffer addresses depend on it
    info.user_data = params.user_data;
    info.RefreshFlatBuf();
    if (info.has_fetch_shader) {
        const u32* fetch = Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base);
        if (!sel.worker && fetch && program_write_sync) {
            program_write_sync(reinterpret_cast<VAddr>(fetch), 256);
        }
        if (!FetchCodeDecodes(fetch)) {
            SyncProgramMemory(reinterpret_cast<VAddr>(fetch), 4096);
            if (!FetchCodeDecodes(fetch)) {
                static std::atomic<u32> reported{0};
                if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
                    LOG_WARNING(Render_Vulkan,
                                "program {:#x}: its fetch shader at {} does not decode "
                                "({:#010x} {:#010x}): draw skipped",
                                params.hash, fmt::ptr(fetch), fetch[0], fetch[1]);
                }
                return {};
            }
        }
    }
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    // Consecutive draws of a program almost always use the same permutation.
    auto& modules = program.modules;
    auto it = program.last_used < modules.size() && modules[program.last_used].spec == spec
                  ? modules.begin() + program.last_used
                  : std::ranges::find(modules, spec, &Program::Module::spec);
    size_t perm_idx;
    if (it != modules.end()) {
        perm_idx = std::distance(modules.begin(), it);
        program.last_used = perm_idx;
        info.AddBindings(binding);
    } else {
        perm_idx = modules.size();
        auto new_info = Shader::Info(hw_stage, sw_stage, params);
        const auto module = CompileModule(new_info, runtime_info, params.code, perm_idx, binding);
        PipelineStore::SaveShaderMeta(info, spec, HashCombine(params.hash, perm_idx), perm_idx);
        std::unique_lock lk{programs_mutex};
        program.AddPermut(module, std::move(spec));
    }
    return {&info, modules[perm_idx].module, modules[perm_idx].spec.fetch_shader_data,
            HashCombine(params.hash, perm_idx)};
}

std::string PipelineCache::GetShaderName(HwStage stage, u64 hash, std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }
    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

} // namespace Vulkan
