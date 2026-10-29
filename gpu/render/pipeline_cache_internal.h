// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: the pipeline cache's own types its parts share (pipeline_cache.cpp and the
// pipeline_*.cpp files split from it): the background compiler, a pipeline being made, a recipe
// import. Only those files include it.
#pragma once

// bbport render: program and pipeline selection (see pipeline_cache.h).

#include <algorithm>
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

namespace Vulkan {

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

// Compile statistics, defined in pipeline_cache.cpp (read by the frame stats and the overlay).
extern std::atomic<u64> g_bb_compile_ns;
extern std::atomic<u32> g_bb_compiles;
extern std::atomic<u32> g_bb_background_pipelines_total, g_bb_background_pipelines_done;
extern std::atomic<u32> g_bb_compiles_total;
extern std::atomic<bool> g_bb_shader_import_active;

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
    void Submit(std::function<void()> job) {
        {
            std::scoped_lock lk{mutex};
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
    }
    /// Drops the jobs not started (the game is closing).
    void Discard() {
        std::scoped_lock lk{mutex};
        jobs.clear();
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
    std::unique_ptr<GraphicsPipeline> pipeline;
    GraphicsPipeline::SerializationSupport sdata{};
    bool saved = false; ///< its key is in the cache on disk already (an imported pipeline)
    /// An imported pipeline: created by whichever comes first, a compiler thread or the GPU
    /// thread that needs it (the compiler's queue is long at the first start).
    std::function<std::unique_ptr<GraphicsPipeline>()> make;
    std::atomic<bool> started{false};
};

/// Shader recipes being imported (StartShaderImport): the import process and the merge of what
/// it wrote into the cache directory's "import" folder.
struct PipelineCache::ShaderImport {
#ifdef _WIN32
    std::atomic<void*> process{nullptr}; ///< HANDLE of bb-probe --import-shaders
    /// Its command line, to start it again after a crash (it goes on from where it stopped).
    std::wstring exe, command;
    std::atomic<bool> stopping{false};
#else
    pid_t pid = -1;
#endif
    std::atomic<bool> running{false};
    std::thread watcher;
    bool listed = false, done = false;
    std::vector<std::pair<u64, std::vector<std::filesystem::path>>> programs; ///< hash, files
    std::vector<std::filesystem::path> keys;
    size_t next_program = 0, next_key = 0;
    u32 frame = ~0u;
    u32 merged = 0, skipped = 0, queued = 0, compute = 0, adopted = 0;
    /// Permutations of programs the game translated meanwhile, renumbered after its own
    /// (AdoptImportedPermutation): the import's permutation hash -> the cache's, for its keys.
    std::unordered_map<u64, u64> remap;
    std::chrono::steady_clock::time_point start;
    /// Compute pipelines the compiler's threads created, for the GPU thread to insert.
    std::mutex compute_mutex;
    std::vector<std::pair<ComputePipelineKey, std::unique_ptr<ComputePipeline>>> compute_made;
    std::atomic<u32> compute_queued{0};

    ~ShaderImport() {
        // The game ends first: the next start translates what is left.
#ifdef _WIN32
        stopping = true;
        if (void* handle = process.load(); running.load() && handle) {
            TerminateProcess(handle, 1);
        }
#else
        if (running.load()) {
            kill(pid, SIGTERM);
        }
#endif
        if (watcher.joinable()) {
            watcher.join();
        }
    }
};

} // namespace Vulkan
