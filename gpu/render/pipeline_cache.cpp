// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: program and pipeline selection (see pipeline_cache.h).

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

// Shader/pipeline compile time on the GPU thread, reported by BB_FRAME_STATS.
std::atomic<u64> g_bb_compile_ns;
std::atomic<u32> g_bb_compiles;
/// For the overlay's "Compiling shaders" line: pipelines made in the background (the cache's
/// preload and imported recipes), queued and done, and whether recipes are being imported.
std::atomic<u32> g_bb_background_pipelines_total, g_bb_background_pipelines_done;
/// Every compile so far (g_bb_compiles is taken by the frame statistics every few seconds).
std::atomic<u32> g_bb_compiles_total;
std::atomic<bool> g_bb_shader_import_active;

namespace {

constexpr u32 SpirvVersion1_6 = 0x00010600U;

constexpr std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

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
            if (const char* env = bb_env("BB_INTERP_INT_FIX")) {
                return env[0] == '1';
            }
            return !instance.IsFragmentShaderBarycentricSupported() && nvidia;
        }(),
        .needs_lds_barriers = nvidia || driver == vk::DriverId::eMesaKosmickrisp,
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = driver == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = nvidia,
        .supports_shader_stencil_export = instance.IsShaderStencilExportSupported(),
#ifdef __APPLE__
        // BB_SPIRV_PRECISE=vertex: exact float math in the position stages only (macOS);
        // unset or anything else: everywhere, as upstream. Part of the profile, so switching
        // rebuilds the shader cache.
        .precise_vertex_stages_only = [] {
            const char* env = bb_env("BB_SPIRV_PRECISE");
            const bool vertex_only = env && std::strcmp(env, "vertex") == 0;
            if (vertex_only) {
                LOG_WARNING(Render, "Shaders: exact float math in vertex, tessellation and "
                                 "geometry stages only (BB_SPIRV_PRECISE=vertex)");
            }
            return vertex_only;
        }(),
        .tess_level_clamp = [] {
            const bool clamp = bb_env_on("BB_TESS_CLAMP");
            if (clamp) {
                LOG_WARNING(Render, "Shaders: tessellation factors clamped to the game's maximum "
                                    "tess level, as the PS4 does (BB_TESS_CLAMP=1)");
            }
            return clamp;
        }(),
        .tess_factor_cap = [] {
            const char* env = bb_env("BB_TESS_MAX");
            const u32 cap = env ? u32(std::strtoul(env, nullptr, 10)) : 0;
            if (cap) {
                LOG_WARNING(Render, "Shaders: tessellation factors capped at {} (BB_TESS_MAX, an "
                                    "experiment: less detail than the game asks for)", cap);
            }
            return cap;
        }(),
#endif
        // BB_LAYER_MEMORY: buffers over nearly all memory go through the page table.
        .paged_buffers = VideoCore::BufferCache::LayerPagedActive(),
        // The game's buffer copy shader, from its profile (games/).
        .buffer_copy_shader_hash = (Shader::paged_crossing_exclude =
                                        bb_env_off("BB_PAGED_COPY_CROSSING")
                                            ? Game::BufferCopyShader()
                                            : 0,
                                    Game::BufferCopyShader()),
    };
}

} // namespace

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
}

PipelineCache::~PipelineCache() {
    if (preload_compiler) {
        preload_compiler->Discard(); // a closing game does not wait for the rest
    }
}

bool PipelineCache::PrepareGraphicsPipeline(PipelineSelection& worker_sel) {
    // Tessellation stages read constant buffers from memory at selection time: not prepared.
    if (worker_sel.regs->stage_enable.hs_en) {
        return false;
    }
    return RefreshGraphicsKey(worker_sel) && !worker_sel.worker->failed;
}

PipelineCompiler& PipelineCache::Compiler() {
    if (!compiler) {
        const char* env = bb_env("BB_ASYNC_PIPELINE_THREADS");
        const u32 threads =
            std::max(env && *env ? u32(std::strtoul(env, nullptr, 10))
                                 : std::clamp(std::thread::hardware_concurrency() / 4, 1u, 4u),
                     1u);
        compiler = std::make_unique<PipelineCompiler>(threads);
        std::printf("GPU: new pipelines of passes drawn every frame compile on %u threads "
                    "(BB_ASYNC_PIPELINES=0: on the GPU thread, the game waits)\n",
                    threads);
    }
    return *compiler;
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
