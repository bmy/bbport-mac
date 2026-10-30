// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: the game's shader programs and the pipelines built from them. A program is a
// guest shader (by the hash of its code) with its translated permutations; a pipeline is keyed by
// the permutations of its stages and the fixed-function state. The GPU thread selects (and, when
// missing, creates) them per draw; draw-preparation workers select ahead of it, read-only. New
// graphics pipelines of passes drawn every frame are compiled on worker threads
// (BB_ASYNC_PIPELINES). Everything created is written to the pipeline cache on disk and preloaded
// at the next start (pipeline_store.cpp).
#pragma once

#include <chrono>

#include <functional>

#include <array>
#include <atomic>
#include <memory>
#include <optional>
#include <shared_mutex>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>
#include <tsl/robin_map.h>

#include "render/pipeline.h"
#include "shader/profile.h"
#include "shader/recompiler.h"
#include "shader/specialization.h"
#include "vulkan/descriptor_heap.h"

namespace BbCp {
class CommandProcessor;
}

namespace Gcn {
union Regs;
}

namespace Serialization {
struct Archive;
}

namespace Vulkan {

class Instance;
class Scheduler;

/// A guest shader program and its translated permutations (one per specialization).
struct Program {
    struct Module {
        vk::ShaderModule module;
        Shader::StageSpecialization spec;
    };
    static constexpr size_t MaxPermutations = 8;
    using ModuleList = boost::container::small_vector<Module, MaxPermutations>;

    Shader::Info info;
    ModuleList modules{};
    size_t last_used = 0; ///< permutation of the previous lookup, compared first
    /// `info` as translated, for draw-preparation workers (they must not read `info`, whose user
    /// data the GPU thread rewrites every draw). Guarded by PipelineCache::programs_mutex.
    std::unique_ptr<Shader::Info> info_template;

    Program() = default;
    Program(Shader::HwStage stage, Shader::SwStage l_stage, Shader::ShaderParams params)
        : info{stage, l_stage, params} {}

    void AddPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec) {
        modules.emplace_back(module, std::move(spec));
    }

    /// At its index as cached: permutation hashes include the index.
    void InsertPermut(vk::ShaderModule module, Shader::StageSpecialization&& spec,
                      size_t perm_idx) {
        modules.resize(std::max(modules.size(), perm_idx + 1));
        modules[perm_idx] = {module, std::move(spec)};
    }
};

/// Indirect draws: the user data registers the vertex and instance offsets go to.
struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
};

/// The state of one pipeline selection. The GPU thread owns one (PipelineCache::sel);
/// draw-preparation workers use their own with their register copies.
struct PipelineSelection {
    const Gcn::Regs* regs{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    std::optional<Shader::Gcn::FetchShaderData> fetch_shader{};
    GraphicsPipelineKey graphics_key{};
    bool motion = false;
    DrawIndirectParams draw_indirect_params{};
    struct PrepWorker* worker{}; ///< set: read-only selection for a draw-preparation worker
};

/// A draw-preparation worker's own program state (see vk_draw_prep.h).
struct PrepWorker {
    struct Stage {
        const Program* program;
        u64 hash;
        Shader::HwStage hw_stage;
        VAddr pgm_base;
        const std::vector<u32>* flat;
    };
    // Node-based: stages keep pointers to these Infos while later stages are inserted.
    std::unordered_map<const Program*, Shader::Info> infos;
    boost::container::static_vector<Stage, MaxShaderStages> stages;
    bool failed = false;
};

struct PreparedDraw;
class PipelineCompiler;

class PipelineCache {
public:
    [[nodiscard]] const Instance& GetInstance() const noexcept {
        return instance;
    }
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           BbCp::CommandProcessor* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    /// Preloads the pipeline cache on disk: after the object motion buffers exist (their
    /// addresses decide whether cached motion shaders are valid).
    void Preload() {
        WarmUp();
        StartShaderImport();
    }

    /// Closes the pipeline cache on disk (writes what is pending).
    void Sync();

    /// Makes what the GPU wrote over a range readable by the CPU (the rasterizer's ReadMemory):
    /// for programs the game copies into place with the GPU.
    void SetProgramMemorySync(std::function<void(VAddr, u64)> sync) {
        program_memory_sync = std::move(sync);
    }
    /// GPU command thread: before reading program code, the GPU's writes there (a copy into place
    /// queued before the draw) are waited for.
    void SetProgramWriteSync(std::function<void(VAddr, u64)> sync) {
        program_write_sync = std::move(sync);
    }

    /// `indirect`: the draw's arguments are in memory (never skipped while compiling).
    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {},
                                                const PreparedDraw* prepared = nullptr,
                                                bool indirect = false);

    /// Worker side of draw preparation: selects the pipeline key for `sel.regs` without
    /// creating anything. False when a program or permutation does not exist yet.
    bool PrepareGraphicsPipeline(PipelineSelection& sel);

    /// GPU-thread side: the pipeline for a prepared draw after checking that registers and
    /// flattened user data match; null to take the regular path.
    const GraphicsPipeline* TryPreparedPipeline(const PreparedDraw& prepared);

    /// The prepared draw the last GetGraphicsPipeline used, or null (regular path).
    [[nodiscard]] const PreparedDraw* UsedPrepared() const noexcept {
        return used_prepared;
    }

    const ComputePipeline* GetComputePipeline();

    /// A stage's translation: its Info, module, fetch shader and permutation hash.
    using Result = std::tuple<const Shader::Info*, vk::ShaderModule,
                              std::optional<Shader::Gcn::FetchShaderData>, u64>;
    Result GetProgram(PipelineSelection& sel, Shader::HwStage stage, Shader::SwStage l_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

private:
    bool RefreshGraphicsKey(PipelineSelection& sel);
    bool RefreshGraphicsStages(PipelineSelection& sel);
    bool RefreshComputeKey();
    Result LookUpProgram(PipelineSelection& sel, const Shader::RuntimeInfo& runtime_info,
                         const Shader::ShaderParams& params, Shader::Backend::Bindings& binding,
                         Shader::HwStage hw_stage);

    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   std::span<const u32> code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(PipelineSelection& sel, Shader::HwStage stage,
                                                Shader::SwStage l_stage);

    // The pipeline cache on disk (pipeline_store.cpp).
    void WarmUp();
    void SaveProfile();
    void DropPreloaded();
    bool LoadComputePipeline(Serialization::Archive& ar);
    /// A preloaded graphics pipeline still to create (WarmUp creates them on several threads).
    using PreloadJob = std::pair<GraphicsPipelineKey, std::function<std::unique_ptr<GraphicsPipeline>()>>;
    bool LoadGraphicsPipeline(Serialization::Archive& ar, std::vector<PreloadJob>& jobs);
    bool LoadPipelineStage(Serialization::Archive& ar, size_t stage);

    const Instance& instance;
    Scheduler& scheduler;
    BbCp::CommandProcessor* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    std::function<void(VAddr, u64)> program_memory_sync;
    std::function<void(VAddr, u64)> program_write_sync;
    /// Program addresses synchronized without making them whole, and when (GPU command thread).
    std::unordered_map<VAddr, std::chrono::steady_clock::time_point> program_sync_tried;
    bool ProgramDecodes(const Shader::ShaderParams& params, Shader::HwStage stage);
    bool LooksLikeProgram(const u32* code, Shader::HwStage stage, bool may_sync);
    bool SyncProgramMemory(VAddr address, u64 size);
    /// Exclusive for program/permutation insertions, shared for worker lookups.
    std::shared_mutex programs_mutex;
    const PreparedDraw* used_prepared = nullptr;
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    PipelineSelection sel{}; ///< GPU thread selection state
    ComputePipelineKey compute_key{};

    /// BB_ASYNC_PIPELINES: a new graphics pipeline of a pass drawn every frame is compiled on
    /// worker threads (PipelineCompiler) while its draws go without it for a frame or two,
    /// instead of the game standing still for the driver (18 ms a pipeline on a GTX 1060, 1-4 s
    /// hitches on a first session). Other passes (one-time renders) compile at once.
    struct PendingPipeline;
    /// Per render target (its address): the frames in a row draws went to it.
    struct TargetStreak {
        u32 last_frame = ~0u;
        u32 frames = 0;
    };
    void NoteDrawTarget(const Gcn::Regs& regs);
    bool AsyncSkippable(const PipelineSelection& sel, bool indirect);
    const GraphicsPipeline* CompileInBackground(const PipelineSelection& sel, bool urgent = false);
    const GraphicsPipeline* FinishPipeline(const GraphicsPipelineKey& key, PendingPipeline& job);
    tsl::robin_map<GraphicsPipelineKey, std::shared_ptr<PendingPipeline>> pending_graphics;
    std::unordered_map<u64, TargetStreak> target_streaks;
    u64 last_target = 0;
    u32 last_target_frame = ~0u;
    u32 stats_frame = ~0u;
    u64 frame_start_draws = 0, last_frame_draws = 0;
    u32 stats_printed = 0;
    u64 async_started = 0, async_skipped = 0, async_waited = 0; ///< statistics
    u64 async_gave_up = 0; ///< bbport (macOS): draws that stopped waiting (SyncWaitMs)
    /// bbport (macOS): AsyncSkippable's draw goes to a target drawn every frame (so a frame
    /// without it is redrawn next frame): only those stop waiting for a slow pipeline.
    bool redrawn_every_frame = false;
    PipelineCompiler& Compiler();
    /// The preloaded graphics pipelines, created in the background while the game starts (a cold
    /// driver cache: about a minute on a 4-thread laptop with NVIDIA); a draw that needs one first
    /// creates it itself (pending_graphics). BB_PRELOAD_BACKGROUND=0: before the game starts.
    void QueuePreloaded(std::vector<PreloadJob>& jobs);
    std::atomic<u32> preload_done{0};
    u32 preload_total = 0;
    std::chrono::steady_clock::time_point preload_start{};

    /// Shader recipes (shader_rebuild.h, pipeline_store.cpp): the programs of the recorded game
    /// translated for this GPU by a background process (bb-probe --import-shaders) at start,
    /// then merged into this cache a few programs a frame, their pipelines created on the
    /// compiler's threads.
    struct ShaderImport;
    void StartShaderImport();
#ifdef _WIN32
    static bool LaunchShaderImport(ShaderImport& state);
#endif
    /// GPU thread: merges what the import process wrote once it has finished.
    void PollShaderImport() {
        if (import_ready.load(std::memory_order_relaxed)) [[unlikely]] {
            MergeShaderImport(false);
        }
    }
    void MergeShaderImport(bool at_start);
    // Its steps (pipeline_import.cpp): one program's files, the permutations of one the game
    // translated meanwhile, one pipeline key and the pipeline it names.
    void MergeImportedProgram(ShaderImport& state, size_t index);
    void AdoptImportedProgram(ShaderImport& state, u64 hash, u64 perms);
    void MergeImportedKey(ShaderImport& state, std::vector<u8>&& bytes);
    void QueueImportedCompute(ShaderImport& state, Serialization::Archive& ar);
    void QueueImportedGraphics(ShaderImport& state, Serialization::Archive& ar);
    /// An imported permutation of a program the game translated meanwhile (programs_mutex held):
    /// its index there, the one with its specialization or a new one appended (saved under it).
    std::optional<size_t> AdoptImportedPermutation(std::vector<u8>&& meta, std::vector<u32>&& spv,
                                                   std::vector<u8>&& input);
    std::atomic<bool> import_ready{false};
    std::unique_ptr<ShaderImport> import; ///< destroyed after the compiler (its jobs done)

    std::unique_ptr<PipelineCompiler> preload_compiler; ///< QueuePreloaded's threads
    std::unique_ptr<PipelineCompiler> compiler; ///< last: its threads are joined first
};

} // namespace Vulkan
