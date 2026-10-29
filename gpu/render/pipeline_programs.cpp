// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: the game's shader programs: whether memory holds a program, whether it decodes,
// the translated program and permutation a draw needs (GetProgram, and LookUpProgram for the
// draw-preparation workers). Split out of pipeline_cache.cpp (2026-10-26).

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
/// The fetch shader a vertex program calls (through a pointer in its user data) decodes up to
/// its s_setpc_b64: the parse asserts on anything else. Memory the game is still filling, or
/// reused, holds garbage there (another game: "illegal instruction" while loading).
/// Its length in dwords through the s_setpc_b64 (0: it does not decode).
u32 FetchCodeDwords(const u32* code) {
    const u32* pc = code;
    for (u32 i = 0; i < 512; ++i) {
        const auto encoding = GcnIsa::EncodingOf(*pc);
        if (encoding == GcnIsa::InstEncoding::ILLEGAL) {
            return 0;
        }
        const bool vector = encoding == GcnIsa::InstEncoding::VOP1 ||
                            encoding == GcnIsa::InstEncoding::VOP2 ||
                            encoding == GcnIsa::InstEncoding::VOPC;
        if (vector && ((*pc & 0x1FF) == 0xF9 || (*pc & 0x1FF) == 0xFA)) {
            return 0; // SDWA/DPP: not in a PS4 program
        }
        if (GcnIsa::Decode(pc).opcode == GcnIsa::Opcode::S_SETPC_B64) {
            return u32(pc - code);
        }
    }
    return 0;
}

/// Checked on every draw of a vertex program: decoded once per code (cached per thread by address
/// and compared with a copy, as ParseFetchShader does); the decode was ~5 % of the GPU command
/// thread on the Steam Deck.
bool FetchCodeDecodes(const u32* code) {
    thread_local std::unordered_map<const u32*, std::vector<u32>> decoded;
    if (const auto it = decoded.find(code);
        it != decoded.end() &&
        std::memcmp(code, it->second.data(), it->second.size() * sizeof(u32)) == 0) {
        return true;
    }
    const u32 dwords = FetchCodeDwords(code);
    if (dwords == 0) {
        return false;
    }
    if (decoded.size() >= 4096) {
        decoded.clear();
    }
    decoded[code].assign(code, code + dwords);
    return true;
}

/// The program being decoded on this thread (GcnIsa::decode_failure).
struct DecodingProgram {
    std::span<const u32> code;
    u64 hash = 0;
    VAddr Base() const { return reinterpret_cast<VAddr>(code.data()); }
};
thread_local DecodingProgram decoding_program;

/// Before an instruction the decoder cannot read stops the game: which program, where in it,
/// its words there, and the writes recorded into it (BB_POOL_WATCH: a GPU-side write over code).
void ReportDecodeFailure(const u32* code, const char* why) {
    const auto& p = decoding_program;
    const VAddr base = p.Base();
    const u64 size = p.code.size_bytes();
    std::fprintf(stderr, "Shader decode: %s in program %#llx (hash %#llx, %llu bytes) at +%#llx\n",
                 why, (unsigned long long)base, (unsigned long long)p.hash,
                 (unsigned long long)size,
                 (unsigned long long)(reinterpret_cast<VAddr>(code) - base));
    const u32* at = code - std::min<std::ptrdiff_t>(4, code - p.code.data());
    for (int i = 0; i < 12 && at + i < p.code.data() + p.code.size(); ++i) {
        std::fprintf(stderr, "%s%08x", i ? " " : "Shader decode: words ", at[i]);
    }
    std::fprintf(stderr, "\n");
    BbFreeCheck::DumpWrites(base, size, "shader program");
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
    decoding_program = {params.code, params.hash};
    GcnIsa::decode_failure = &ReportDecodeFailure;
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
    const u32* fetch = info.has_fetch_shader
                           ? Shader::Gcn::GetFetchShaderCode(info, info.fetch_shader_sgpr_base)
                           : nullptr;
    if (fetch && !sel.worker && program_write_sync) {
        program_write_sync(reinterpret_cast<VAddr>(fetch), 256);
    }
    if (fetch) {
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
    auto& modules = program.modules;
    // A known permutation, compared without building the specialization (Matches; the
    // tessellation stages build theirs: their runtime info comes from a constant buffer).
    // BB_SPEC_MATCH=0: always built and compared.
    static const bool match_in_place = !bb_env_off("BB_SPEC_MATCH");
    if (match_in_place && sw_stage != SwStage::TessellationControl &&
        sw_stage != SwStage::TessellationEval) {
        const auto* fetch_data = Shader::Gcn::ParseFetchShaderRef(info);
        const auto matches = [&](size_t i) {
            return modules[i].spec.Matches(info, runtime_info, profile, binding, fetch_data);
        };
        size_t found = modules.size();
        if (program.last_used < modules.size() && matches(program.last_used)) {
            found = program.last_used;
        } else {
            for (size_t i = 0; i < modules.size(); ++i) {
                if (i != program.last_used && matches(i)) {
                    found = i;
                    break;
                }
            }
        }
        // BB_SPEC_MATCH_CHECK=1 (diagnostics): the built specialization must find the same one.
        static const bool check = bb_env("BB_SPEC_MATCH_CHECK") != nullptr;
        if (check) {
            const auto built = Shader::StageSpecialization(info, runtime_info, profile, binding);
            size_t expected = modules.size();
            for (size_t i = 0; i < modules.size(); ++i) {
                if (modules[i].spec == built) {
                    expected = i;
                    break;
                }
            }
            static std::atomic<u64> checked{0}, differed{0};
            const bool same = expected == found ||
                              (expected < modules.size() && found < modules.size() &&
                               modules[expected].spec == modules[found].spec);
            differed += same ? 0 : 1;
            if (++checked % 100000 == 0 || (!same && differed.load() <= 8)) {
                std::printf("Spec match check: %llu lookups, %llu differed (program %#llx stage "
                            "%u: built %zu, matched %zu)\n",
                            (unsigned long long)checked.load(), (unsigned long long)differed.load(),
                            (unsigned long long)params.hash, u32(hw_stage), expected, found);
            }
        }
        if (found < modules.size()) {
            program.last_used = found;
            info.AddBindings(binding);
            return {&info, modules[found].module, modules[found].spec.fetch_shader_data,
                    HashCombine(params.hash, found)};
        }
    }
    auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);

    // Consecutive draws of a program almost always use the same permutation.
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

} // namespace Vulkan
