// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: shader recipes imported while the game runs (a bb-probe --import-shaders
// process translating them) and merged into the cache as they come. Split out of
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
#ifdef __APPLE__
#include <mach-o/dyld.h> // bbport (macOS): _NSGetExecutablePath (no /proc/self/exe)
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

#ifdef _WIN32
/// bb-probe.exe --import-shaders started (state.exe, state.command), below normal priority: the
/// game runs meanwhile. Its report goes to the game's log.
bool PipelineCache::LaunchShaderImport(ShaderImport& state) {
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION created{};
    std::wstring command = state.command;
    if (!CreateProcessW(state.exe.c_str(), command.data(), nullptr, nullptr, TRUE,
                        BELOW_NORMAL_PRIORITY_CLASS, nullptr, nullptr, &startup, &created)) {
        return false;
    }
    CloseHandle(created.hThread);
    state.process = created.hProcess;
    return true;
}
#endif

void PipelineCache::StartShaderImport() {
    auto& store = BbCache::BlobStore::Instance();
    if (!store.IsOpened()) {
        return;
    }
    std::error_code ec;
    // Translated after the last session had ended: merged now, before the game draws (its compute
    // pipelines are collected later, PollShaderImport).
    if (std::filesystem::exists(store.Directory() / "import", ec)) {
        MergeShaderImport(true);
        import_ready.store(true, std::memory_order_relaxed);
    }
    // run.sh: the pack shipped with the port and the game's files the program code comes from.
    const char* pack = bb_env("BB_SHADER_RECIPES");
    const char* game = bb_env("BB_SHADER_GAME_DIR");
    if (!pack || !*pack || !game || !*game || !std::filesystem::exists(pack, ec)) {
        return;
    }
#if defined(__APPLE__) && !defined(__x86_64__)
    // bbport (the native GPU process): the importer is bb-probe's x86-64 translator; its programs
    // are not for this process's cache (SRT walkers as bytecode here).
    std::printf("Shader recipes: no import in the native GPU process\n");
    return;
#endif
    const std::string dir = store.Directory().string();
    const std::string params = fmt::format("{:x}", Shader::MotionVectors::params_address);
    const std::string positions = fmt::format("{:x}", Shader::MotionVectors::positions_address);
#ifndef _WIN32
    const char* argv[] = {"bb-probe", "--import-shaders", dir.c_str(), pack, game,
                          params.c_str(), positions.c_str(), nullptr};
    // Steam's overlay (run.sh preloads it for the game) has nothing to do in it.
    std::vector<char*> env;
    for (char** e = environ; *e; ++e) {
        if (std::strncmp(*e, "LD_PRELOAD=", 11) != 0) {
            env.push_back(*e);
        }
    }
    env.push_back(nullptr);
#endif
    if (!import) {
        import = std::make_unique<ShaderImport>();
    } else {
        // Kept (compute pipelines of the merge above may still be compiling): a new merge.
        import->listed = import->done = false;
        import->programs.clear();
        import->keys.clear();
        import->next_program = import->next_key = 0;
    }
#ifdef _WIN32
    {
        // bb-probe.exe again, below normal priority: the game runs meanwhile.
        wchar_t exe[MAX_PATH];
        std::wstring command;
        if (GetModuleFileNameW(nullptr, exe, MAX_PATH)) {
            command = L"\"" + std::wstring{exe} + L"\" --import-shaders";
            for (const std::string& arg : {dir, std::string{pack}, std::string{game}, params, positions}) {
                command += L" \"" + std::filesystem::path{arg}.wstring() + L"\"";
            }
        }
        import->exe = exe;
        import->command = command;
        if (command.empty() || !LaunchShaderImport(*import)) {
            import->done = true;
            return;
        }
    }
#else
#ifdef __APPLE__
    // bbport (macOS): this executable (bb-probe) by its path.
    char self[4096];
    uint32_t self_size = sizeof(self);
    const char* exe_path = _NSGetExecutablePath(self, &self_size) == 0 ? self : nullptr;
    if (!exe_path) {
        std::printf("Shader recipes: no import (the executable's path is too long)\n");
        import->done = true;
        return;
    }
#else
    const char* exe_path = "/proc/self/exe";
#endif
    if (posix_spawn(&import->pid, exe_path, nullptr, nullptr, const_cast<char**>(argv),
                    env.data()) != 0) {
        import->done = true;
        return;
    }
#endif
    import->running = true;
    g_bb_shader_import_active.store(true, std::memory_order_relaxed);
    import->watcher = std::thread([this, state = import.get()] {
        Common::SetCurrentThreadName("bb:ShaderImport");
#ifdef _WIN32
        // A crash (a translator fault: no fork here) ends the import process; started again, it
        // goes on after the program it was on (import.current, previous.work). A few times only.
        for (int restarts = 0;; ++restarts) {
            void* handle = state->process.load();
            WaitForSingleObject(handle, INFINITE);
            DWORD code = 0;
            GetExitCodeProcess(handle, &code);
            CloseHandle(handle);
            state->process = nullptr;
            if (code == 0 || state->stopping || restarts == 16) {
                break;
            }
            std::printf("Shader recipes: the import process ended (code %lu), started again\n",
                        static_cast<unsigned long>(code));
            if (!LaunchShaderImport(*state)) {
                break;
            }
        }
#else
        int status = 0;
        while (waitpid(state->pid, &status, 0) < 0 && errno == EINTR) {
        }
#endif
        state->running = false;
        std::error_code ec;
        if (std::filesystem::exists(BbCache::BlobStore::Instance().Directory() / "import", ec)) {
            import_ready.store(true, std::memory_order_relaxed);
        } else {
            g_bb_shader_import_active.store(false, std::memory_order_relaxed); // nothing new
        }
    });
}

namespace {

namespace fs = std::filesystem;

std::string Hex(u64 value) {
    return fmt::format("{:#018x}", value);
}

std::vector<u8> ReadImportFile(const fs::path& path) {
    std::vector<u8> bytes;
    const Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read};
    if (file.IsOpen()) {
        bytes.resize(file.GetSize());
        file.Read(bytes);
    }
    return bytes;
}

/// A program's imported files and the metadata of its `perms` permutations, deleted.
void RemoveImportedProgram(const fs::path& from, const std::vector<fs::path>& files, u64 hash,
                           u64 perms) {
    std::error_code ec;
    for (const auto& file : files) {
        fs::remove(file, ec);
    }
    for (u64 perm = 0; perm < perms; ++perm) {
        fs::remove(from / (Hex(HashCombine(hash, perm)) + ".meta"), ec);
    }
}

/// Stages of renumbered permutations (AdoptImportedPermutation): a pipeline key rewritten to the
/// cache's numbers, in `bytes` and its file. False: the key names a permutation left out (or is
/// damaged) and is dropped.
bool RemapImportedKey(const std::unordered_map<u64, u64>& remap, std::vector<u8>& bytes,
                      const fs::path& path) {
    bool drop = false, changed = false;
    try {
        Serialization::Archive in{std::vector<u8>{bytes}};
        Serialization::Reader reader{in};
        u32 version{}, is_compute{};
        reader.Read(version);
        reader.Read(is_compute);
        const auto map = [&](auto& value) {
            if (const auto it = remap.find(u64(value)); it != remap.end()) {
                drop |= it->second == 0;
                changed |= it->second != u64(value);
                value = static_cast<std::remove_reference_t<decltype(value)>>(it->second);
            }
        };
        Serialization::Archive out{};
        Serialization::Writer writer{out};
        writer.Write(version);
        writer.Write(is_compute);
        if (version != PipelineStore::PipelineKeyVersion) {
            // another build's: left as it is
        } else if (is_compute) {
            ComputePipelineKey key{};
            key.Deserialize(in);
            ComputePipeline::SerializationSupport sdata{};
            sdata.Deserialize(in);
            map(key.value);
            key.Serialize(out);
            sdata.Serialize(out);
        } else {
            GraphicsPipelineKey key{};
            key.Deserialize(in);
            GraphicsPipeline::SerializationSupport sdata{};
            sdata.Deserialize(in);
            for (auto& stage : key.stage_hashes) {
                map(stage);
            }
            key.Serialize(out);
            sdata.Serialize(out);
        }
        if (changed && !drop) {
            bytes = std::move(out.TakeOff());
            Common::FS::IOFile file{path, Common::FS::FileAccessMode::Write};
            if (file.IsOpen()) {
                file.Write(bytes);
            }
        }
    } catch (const std::exception&) {
        drop = true;
    }
    return !drop;
}

} // namespace

void PipelineCache::MergeShaderImport(bool at_start) {
    auto& store = BbCache::BlobStore::Instance();
    if (!import) {
        import = std::make_unique<ShaderImport>();
    }
    auto& state = *import;
    // Compute pipelines the compiler's threads created (unless the game made one meanwhile).
    if (state.compute_queued.load(std::memory_order_relaxed) != 0) {
        std::scoped_lock lk{state.compute_mutex};
        for (auto& [key, pipeline] : state.compute_made) {
            compute_pipelines.try_emplace(key, std::move(pipeline));
            state.compute_queued.fetch_sub(1, std::memory_order_relaxed);
        }
        state.compute_made.clear();
    }
    if (!at_start && state.running.load(std::memory_order_relaxed)) {
        return; // the import process is still translating (its folder appears when it is done)
    }
    if (state.done || !store.IsOpened()) {
        if (state.compute_queued.load(std::memory_order_relaxed) == 0) {
            import_ready.store(false, std::memory_order_relaxed);
        }
        return;
    }
    const fs::path dir = store.Directory(), from = dir / "import";
    std::error_code ec;
    if (!state.listed) {
        state.listed = true;
        state.start = std::chrono::steady_clock::now();
        std::map<u64, std::vector<fs::path>> by_program;
        for (const auto& entry : fs::directory_iterator{from, ec}) {
            const auto ext = entry.path().extension();
            if (ext == ".spv" || ext == ".in") {
                const u64 hash = std::strtoull(entry.path().stem().string().c_str(), nullptr, 16);
                by_program[hash].push_back(entry.path());
            } else if (ext == ".key") {
                state.keys.push_back(entry.path());
            }
        }
        state.programs.assign(by_program.begin(), by_program.end());
    }
    // A few milliseconds a frame on the GPU thread (files, shader modules), not one long hitch.
    const u32 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    if (!at_start && frame == state.frame) {
        return;
    }
    state.frame = frame;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
    const auto in_budget = [&] {
        return at_start || std::chrono::steady_clock::now() < deadline;
    };
    // The programs first, then the pipeline keys that name them.
    while (state.next_program < state.programs.size() && in_budget()) {
        MergeImportedProgram(state, state.next_program++);
    }
    while (state.next_program == state.programs.size() && state.next_key < state.keys.size() &&
           in_budget()) {
        const fs::path& path = state.keys[state.next_key++];
        std::vector<u8> bytes;
        {
            const Common::FS::IOFile file{path, Common::FS::FileAccessMode::Read};
            if (!file.IsOpen()) {
                continue;
            }
            bytes.resize(file.GetSize());
            file.Read(bytes);
        }
        if (!state.remap.empty() && !RemapImportedKey(state.remap, bytes, path)) {
            fs::remove(path, ec);
            continue;
        }
        fs::rename(path, dir / path.filename(), ec);
        MergeImportedKey(state, std::move(bytes));
    }
    if (state.next_program == state.programs.size() && state.next_key == state.keys.size()) {
        fs::remove_all(from, ec);
        state.done = true;
        g_bb_shader_import_active.store(false, std::memory_order_relaxed);
        std::printf("Shader recipes: %u programs merged into the pipeline cache (%u translated by "
                    "the game meanwhile, %u of their permutations joined theirs), %u graphics and "
                    "%u compute pipelines compiling in the background (%.0f ms)\n",
                    state.merged, state.skipped, state.adopted, state.queued, state.compute,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                              state.start)
                        .count());
    }
}

void PipelineCache::MergeImportedProgram(ShaderImport& state, size_t index) {
    auto& store = BbCache::BlobStore::Instance();
    const fs::path dir = store.Directory(), from = dir / "import";
    std::error_code ec;
    const auto& [hash, files] = state.programs[index];
    u64 perms = 0;
    for (const auto& file : files) {
        perms += file.extension() == ".spv" ? 1 : 0;
    }
    if (program_cache.contains(hash)) {
        AdoptImportedProgram(state, hash, perms);
        RemoveImportedProgram(from, files, hash, perms);
        ++state.skipped;
        return;
    }
    if (fs::exists(dir / (Hex(hash) + "_0.spv"), ec)) {
        RemoveImportedProgram(from, files, hash, perms);
        ++state.skipped;
        return;
    }
    for (const auto& file : files) {
        fs::rename(file, dir / file.filename(), ec);
    }
    const PipelineSelection saved = sel;
    for (u64 perm = 0; perm < perms; ++perm) {
        const std::string meta_name = Hex(HashCombine(hash, perm));
        fs::rename(from / (meta_name + ".meta"), dir / (meta_name + ".meta"), ec);
        std::vector<u8> meta;
        store.Load(BbCache::BlobType::ShaderMeta, meta_name, meta);
        try {
            Serialization::Archive ar{std::move(meta)};
            std::unique_lock lk{programs_mutex};
            LoadPipelineStage(ar, 0);
        } catch (const std::exception&) {
            // compiled in the game if it is needed
        }
    }
    sel = saved;
    ++state.merged;
}

void PipelineCache::AdoptImportedProgram(ShaderImport& state, u64 hash, u64 perms) {
    // Translated by the game meanwhile (its own permutation numbers): the import's permutations
    // join the game's, renumbered after them, and its pipelines with them (state.remap). They
    // used to be dropped: every permutation the game had not reached yet compiled when it did
    // (2026-10-27: copy shaders and post passes, ~30 compiles entering the level).
    const fs::path from = BbCache::BlobStore::Instance().Directory() / "import";
    std::unique_lock lk{programs_mutex};
    for (u64 perm = 0; perm < perms; ++perm) {
        const u64 perm_hash = HashCombine(hash, perm);
        const std::string name = fmt::format("{}_{}", Hex(hash), perm);
        auto meta = ReadImportFile(from / (Hex(perm_hash) + ".meta"));
        const auto bytes = ReadImportFile(from / (name + ".spv"));
        std::vector<u32> spv(bytes.size() / 4);
        std::memcpy(spv.data(), bytes.data(), spv.size() * 4);
        if (const auto idx = AdoptImportedPermutation(std::move(meta), std::move(spv),
                                                      ReadImportFile(from / (name + ".in")))) {
            state.remap[perm_hash] = HashCombine(hash, u64(*idx));
            ++state.adopted;
        } else {
            state.remap[perm_hash] = 0; // its pipelines are left out
        }
    }
}

void PipelineCache::MergeImportedKey(ShaderImport& state, std::vector<u8>&& bytes) {
    const PipelineSelection saved = sel;
    try {
        Serialization::Archive ar{std::move(bytes)};
        Serialization::Reader reader{ar};
        u32 version{}, is_compute{};
        reader.Read(version);
        reader.Read(is_compute);
        if (version != PipelineStore::PipelineKeyVersion) {
            // another build's
        } else if (is_compute) {
            QueueImportedCompute(state, ar);
        } else {
            QueueImportedGraphics(state, ar);
        }
    } catch (const std::exception&) {
        // a damaged key: the game builds the pipeline if it needs it
    }
    sel = saved;
}

void PipelineCache::QueueImportedCompute(ShaderImport& state, Serialization::Archive& ar) {
    ComputePipelineKey key{};
    key.Deserialize(ar);
    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);
    std::vector<u8> meta;
    BbCache::BlobStore::Instance().Load(BbCache::BlobType::ShaderMeta, Hex(key.value), meta);
    bool loaded = false;
    if (!meta.empty() && !compute_pipelines.contains(key)) {
        Serialization::Archive meta_ar{std::move(meta)};
        std::unique_lock lk{programs_mutex};
        loaded = LoadPipelineStage(meta_ar, 0);
    }
    if (!loaded) {
        return;
    }
    const Shader::Info* info = sel.infos[0];
    const vk::ShaderModule module = sel.modules[0];
    ++state.compute;
    state.compute_queued.fetch_add(1, std::memory_order_relaxed);
    Compiler().Submit([this, &state, key, info, module, sdata]() mutable {
        std::unique_ptr<ComputePipeline> pipeline;
        try {
            pipeline = std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile,
                                                         *pipeline_cache, key, *info, module,
                                                         sdata, true);
        } catch (const std::exception&) {
            // created at its first use instead
        }
        std::scoped_lock lk{state.compute_mutex};
        if (pipeline) {
            state.compute_made.emplace_back(key, std::move(pipeline));
        } else {
            state.compute_queued.fetch_sub(1, std::memory_order_relaxed);
        }
    });
}

void PipelineCache::QueueImportedGraphics(ShaderImport& state, Serialization::Archive& ar) {
    std::vector<PreloadJob> jobs;
    {
        std::unique_lock lk{programs_mutex};
        LoadGraphicsPipeline(ar, jobs);
    }
    for (auto& [key, make] : jobs) {
        if (graphics_pipelines.contains(key) || pending_graphics.contains(key)) {
            continue;
        }
        auto job = std::make_shared<PendingPipeline>();
        job->saved = true;
        job->make = std::move(make);
        pending_graphics.emplace(key, job);
        ++state.queued;
        g_bb_background_pipelines_total.fetch_add(1, std::memory_order_relaxed);
        Compiler().Submit([job] {
            SCOPE_EXIT {
                g_bb_background_pipelines_done.fetch_add(1, std::memory_order_relaxed);
            };
            if (job->started.exchange(true)) {
                return; // the GPU thread needed it first
            }
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
        });
    }
}

} // namespace Vulkan
