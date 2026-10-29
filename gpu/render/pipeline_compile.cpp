// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: getting a draw's pipeline: prepared by the workers, found in the cache, or
// compiled (on this thread or in the background, with the compile time measured), and the shader
// modules. Split out of pipeline_cache.cpp (2026-10-26).

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
void AddCompileTime(std::chrono::steady_clock::time_point start) {
    g_bb_compile_ns += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - start)
                               .count());
}

/// BB_COMPILE_TRACE=1: every shader and pipeline compiled during the session is printed.
bool CompileTrace() {
    static const bool on = bb_env_on("BB_COMPILE_TRACE");
    return on;
}

struct CompileTimer {
    BbCompileLog::Kind kind;
    u64 hash;
    u8 stage = 0;
    u16 permutation = 0;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~CompileTimer() {
        AddCompileTime(start);
        BbCompileLog::Record(kind, hash, start, stage, permutation);
        ++g_bb_compiles;
        g_bb_compiles_total.fetch_add(1, std::memory_order_relaxed);
    }
};

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
} // namespace

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
    PollShaderImport();
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
        // An imported pipeline still in the compiler's queue: created here, now.
        if (job->make && !job->started.exchange(true)) {
            const auto start = std::chrono::steady_clock::now();
            std::unique_ptr<GraphicsPipeline> made;
            try {
                made = job->make();
            } catch (const std::exception&) {
                // compiled the usual way below
            }
            AddCompileTime(start);
            std::scoped_lock lk{job->mutex};
            job->pipeline = std::move(made);
            job->done = true;
        }
        {
            std::unique_lock lk{job->mutex};
            if (!job->done) {
                if (skippable) {
                    ++async_skipped;
                    return nullptr; // drawn from the frame its pipeline is ready
                }
                // A pass that is not drawn every frame needs it now: the wait is compile time.
                const auto start = std::chrono::steady_clock::now();
                job->cv.wait(lk, [&] { return job->done; });
                AddCompileTime(start);
                ++async_waited;
            }
        }
        pending_graphics.erase(sel.graphics_key);
        // An imported pipeline the driver rejected is compiled the usual way below.
        if (job->pipeline || !job->saved) {
            return FinishPipeline(sel.graphics_key, *job);
        }
    }
    if (skippable) {
        return CompileInBackground(sel);
    }

    const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(sel.graphics_key);
    LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
    if (CompileTrace()) std::printf("Compile: graphics pipeline %#llx\n", (unsigned long long)pipeline_hash);
    CompileTimer timer{BbCompileLog::Kind::Graphics, pipeline_hash};
    GraphicsPipeline::SerializationSupport sdata{};
    auto& slot = graphics_pipelines[sel.graphics_key];
    slot = std::make_unique<GraphicsPipeline>(instance, scheduler, desc_heap, profile,
                                              sel.graphics_key, *pipeline_cache, sel.infos,
                                              sel.runtime_infos, sel.fetch_shader, sel.modules,
                                              sdata, false);
    PipelineStore::SaveGraphics(sel.graphics_key, pipeline_hash, sdata);
    sel.fetch_shader.reset();
    return slot.get();
}

void PipelineCache::QueuePreloaded(std::vector<PreloadJob>& jobs) {
    if (!preload_compiler) {
        // One thread is left to the game.
        const u32 threads = std::clamp(std::thread::hardware_concurrency(), 2u, 9u) - 1;
        preload_compiler = std::make_unique<PipelineCompiler>(threads);
    }
    preload_start = std::chrono::steady_clock::now();
    for (auto& [key, make] : jobs) {
        if (graphics_pipelines.contains(key) || pending_graphics.contains(key)) {
            continue;
        }
        auto job = std::make_shared<PendingPipeline>();
        job->saved = true;
        job->make = std::move(make);
        pending_graphics.emplace(key, job);
        ++preload_total;
        g_bb_background_pipelines_total.fetch_add(1, std::memory_order_relaxed);
        preload_compiler->Submit([this, job] {
            SCOPE_EXIT {
                g_bb_background_pipelines_done.fetch_add(1, std::memory_order_relaxed);
            };
            if (!job->started.exchange(true)) {
                std::unique_ptr<GraphicsPipeline> pipeline;
                try {
                    pipeline = job->make();
                } catch (const std::exception&) {
                    // compiled again when the game asks for it
                }
                std::scoped_lock lk{job->mutex};
                job->pipeline = std::move(pipeline);
                job->done = true;
                job->cv.notify_all();
            }
            if (preload_done.fetch_add(1) + 1 == preload_total) {
                std::printf("Pipeline cache: %u preloaded pipelines created in the background in "
                            "%.0f ms\n",
                            preload_total,
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - preload_start)
                                .count());
            }
        });
    }
}

const GraphicsPipeline* PipelineCache::CompileInBackground(const PipelineSelection& selection) {
    auto job = std::make_shared<PendingPipeline>();
    pending_graphics.emplace(selection.graphics_key, job);
    ++async_started;
    ++g_bb_compiles;
    g_bb_compiles_total.fetch_add(1, std::memory_order_relaxed);
    Compiler().Submit([this, job, key = selection.graphics_key, infos = selection.infos,
                      runtime_infos = selection.runtime_infos, fetch = selection.fetch_shader,
                      modules = selection.modules]() mutable {
        const auto start = std::chrono::steady_clock::now();
        GraphicsPipeline::SerializationSupport sdata{};
        auto pipeline = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, key, *pipeline_cache, infos, runtime_infos,
            fetch, modules, sdata, false);
        BbCompileLog::Record(BbCompileLog::Kind::Background, std::hash<GraphicsPipelineKey>{}(key),
                             start);
        std::scoped_lock lk{job->mutex};
        job->pipeline = std::move(pipeline);
        job->sdata = std::move(sdata);
        job->done = true;
        job->cv.notify_all();
    });
    sel.fetch_shader.reset();
    return nullptr;
}

const GraphicsPipeline* PipelineCache::FinishPipeline(const GraphicsPipelineKey& key,
                                                      PendingPipeline& job) {
    if (!job.saved) {
        PipelineStore::SaveGraphics(key, std::hash<GraphicsPipelineKey>{}(key), job.sdata);
    }
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
    static const bool enabled = !bb_env_off("BB_ASYNC_PIPELINES");
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
                        "theirs meanwhile, %llu waited for one\n",
                        (unsigned long long)async_started, (unsigned long long)async_skipped,
                        (unsigned long long)async_waited);
            async_started = async_skipped = async_waited = 0;
        }
    }
    if (last_frame_draws < 300) {
        return false;
    }
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
    constexpr u32 PerFrameStreak = 8;
    const u64 target = DrawTarget(*selection.regs);
    const auto streak = target_streaks.find(target);
    return target != 0 && streak != target_streaks.end() && streak->second.frames >= PerFrameStreak;
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    if (!RefreshComputeKey()) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}",
                 std::hash<ComputePipelineKey>{}(compute_key));
        if (CompileTrace()) std::printf("Compile: compute pipeline %#llx\n", (unsigned long long)std::hash<ComputePipelineKey>{}(compute_key));
        CompileTimer timer{BbCompileLog::Kind::Compute, std::hash<ComputePipelineKey>{}(compute_key)};
        ComputePipeline::SerializationSupport sdata{};
        it.value() = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                       *pipeline_cache, compute_key, *sel.infos[0],
                                                       sel.modules[0], sdata, false);
        PipelineStore::SaveCompute(compute_key, sdata);
    }
    return it->second.get();
}

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              std::span<const u32> code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    if (CompileTrace()) std::printf("Compile: shader %#llx stage %u permutation %zu%s\n", (unsigned long long)info.pgm_hash, u32(info.hw_stage), perm_idx, runtime_info.hw_stage == Shader::HwStage::Vertex && runtime_info.hw.vs.motion_vectors ? " (motion)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");
    CompileTimer timer{BbCompileLog::Kind::Shader, info.pgm_hash, u8(info.hw_stage), u16(perm_idx)};

    // BB_SHADER_CAPTURE: the inputs as they were, for the record.
    const bool capture = BbShaderCapture::Enabled();
    std::optional<Shader::Info> in_info;
    const auto in_runtime_info = runtime_info;
    const auto in_binding = binding;
    // The inputs are kept in the pipeline cache too: a later build translates them again at
    // start instead of compiling while the game runs (shader_rebuild.h).
    const bool keep_input = EmulatorSettings.IsPipelineCacheEnabled();
    if (capture || keep_input) {
        in_info.emplace(info);
        BbShaderCapture::BeginInput();
    }
    const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
    auto spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    if (keep_input) {
        PipelineStore::SaveShaderInput(
            BbShaderCapture::TakeInput(*in_info, in_runtime_info, in_binding, code, perm_idx,
                                       info.flattened_ud_buf),
            info.pgm_hash, perm_idx);
    }
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

} // namespace Vulkan
