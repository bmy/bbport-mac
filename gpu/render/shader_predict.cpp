// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: shader recipe experiments and pack tools (Linux only, they fork their workers,
// except MergeMetas, which runs on Windows one target after another):
// shader input prediction (PredictTest), another player's cache metadata merged into a recipe
// pack (MergeMetas), packs merged (MergePacks) and the metadata reconstruction test. Split out of
// shader_rebuild.cpp (2026-10-26); the helpers it shares with it are in shader_rebuild_internal.h.
#include "render/shader_rebuild.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <miniz.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <csetjmp>
#include <dlfcn.h>
#include <ucontext.h>
#include <csignal>
#include <unistd.h>
#endif

#include "common/hash.h"
#include "common/path_util.h"
#include "common/serdes.h"
#include "render/pipeline.h"
#include "render/pipeline_store.h"
#include "render/shader_capture.h"
#include "shader/backend/spirv/emit_spirv.h"
#include "shader/info.h"
#include "shader/profile.h"
#include "shader/recompiler.h"
#include "shader/specialization.h"
#include "render/shader_rebuild_internal.h"

namespace BbShaderRebuild {

// Shader input prediction (experiment): an unrecorded program translated with the input another
// program was recorded with (its user data, descriptors, fetch shader, runtime info). Measured by
// leaving each recorded program out: does a donor's input give a specialization the program was
// really drawn with (the cache would then hold the permutation the game asks for)?
namespace {

#ifndef _WIN32
sigjmp_buf* predict_recover = nullptr;
u64 predict_fault_addr = 0, predict_fault_rip = 0;
void PredictSigaction(int sig, siginfo_t* info, void* context) {
    predict_fault_addr = reinterpret_cast<u64>(info->si_addr);
    predict_fault_rip = u64(static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_RIP]);
    if (predict_recover) {
        siglongjmp(*predict_recover, sig);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}
void PredictSignal(int sig) {
    if (predict_recover) {
        siglongjmp(*predict_recover, sig);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}
#endif
/// Why SpecWith failed: 1 memory not mappable, 2 a fault, 3 an exception, 4 another flat user data size.
int predict_fail = 0;

/// A recipe permutation's input with its program code and the programs its memory held.
std::optional<BbShaderCapture::Input> RecipeInput(const RecipeProgram& program, size_t perm_idx,
                                                  const GameShaders& game) {
    BbShaderCapture::Input in;
    if (!BbShaderCapture::ReadInput(program.perms[perm_idx].input, in, true)) {
        return std::nullopt;
    }
    const auto* code = game.Code(program.hash);
    if (!code) {
        return std::nullopt;
    }
    in.code = *code;
    for (const auto& ref : program.perms[perm_idx].refs) {
        const auto* ref_code = game.Code(ref.hash);
        if (!ref_code || ref.size > ref_code->size() * sizeof(u32)) {
            return std::nullopt;
        }
        if (ref.address == ~u64{0}) {
            in.vs_copy = *ref_code;
            continue;
        }
        const auto* bytes = reinterpret_cast<const u8*>(ref_code->data());
        in.regions.push_back({ref.address, std::vector<u8>(bytes, bytes + ref.size)});
    }
    return in;
}

struct PredictedSpec {
    std::unique_ptr<Shader::Info> info;
    Shader::StageSpecialization spec;
};

/// The specialization program `hash` (`code`) gets with `donor`'s input; nullopt on a failure.
std::optional<PredictedSpec> SpecWith(const BbShaderCapture::Input& donor, u64 hash,
                                      const std::vector<u32>& code,
                                      const Shader::Profile& profile) {
    BbShaderCapture::Input in = donor;
    in.code = code;
    in.pgm_hash = hash;
    Shader::MotionVectors::params_address = in.params_address;
    Shader::MotionVectors::positions_address = in.positions_address;
    auto regions = in.regions;
    regions.push_back({in.pgm_base, std::vector<u8>(reinterpret_cast<const u8*>(in.code.data()),
                                                    reinterpret_cast<const u8*>(in.code.data()) +
                                                        in.code.size() * sizeof(u32))});
    GuestMemory memory;
    if (!memory.Map(regions)) {
        predict_fail = 1;
        return std::nullopt;
    }
    auto runtime_info = in.runtime_info;
    if (in.hw_stage == Shader::HwStage::Geometry) {
        runtime_info.hw.gs.vs_copy = in.vs_copy;
    }
    const std::span<const u32> guest_code{reinterpret_cast<const u32*>(in.pgm_base),
                                          in.code.size()};
    PredictedSpec out;
    out.info = std::make_unique<Shader::Info>(
        in.hw_stage, in.sw_stage,
        Shader::ShaderParams{std::span<const u32, 16>{in.user_data}, guest_code, in.pgm_hash});
#ifndef _WIN32
    sigjmp_buf recover;
    if (sigsetjmp(recover, 1)) {
        predict_recover = nullptr;
        Shader::flat_buf_replay = nullptr;
        predict_fail = 2;
        return std::nullopt;
    }
    predict_recover = &recover;
#endif
    // Windows: a fault ends the process (MergeMetas resumes after the program it was on).
    Shader::flat_buf_replay = &in.flat;
    bool ok = true;
    try {
        Shader::Pools pools;
        auto info_runtime = runtime_info;
        (void)Shader::TranslateProgram(guest_code, pools, *out.info, info_runtime, profile);
        ok = out.info->srt_info.flattened_bufsize_dw == in.flat.size() || in.flat.empty();
        predict_fail = ok ? 0 : 4;
        if (ok) {
            out.info->RefreshFlatBuf();
            out.spec = Shader::StageSpecialization(*out.info, runtime_info, profile, in.bindings);
        }
    } catch (const std::exception&) {
        ok = false;
        predict_fail = 3;
    }
#ifndef _WIN32
    predict_recover = nullptr;
#endif
    Shader::flat_buf_replay = nullptr;
    if (!ok) {
        return std::nullopt;
    }
    return out;
}

} // namespace
#ifndef _WIN32
/// How a predicted specialization differs from the real one (bits, 0: the same):
/// 1 runtime info (pass state), 2 fetch shader (vertex attribute set), 4 vertex attribute count,
/// 8 vertex attribute types, 16 resource set (bound resources, counts), 32 buffer formats/strides,
/// 64 image types/formats, 128 sampler flags, 256 fmasks, 512 start bindings.
u32 SpecDiff(const Shader::StageSpecialization& a, const Shader::StageSpecialization& b) {
    u32 mask = 0;
    if (a.runtime_info != b.runtime_info) {
        mask |= 1;
        const auto& x = a.runtime_info;
        const auto& y = b.runtime_info;
        if (!(x.props == y.props)) mask |= 1024;
        if (!x.HasSameSwInfo(y.sw)) mask |= 2048;
        if (!x.HasSameHwInfo(y.hw)) {
            mask |= 4096;
            if (x.hw_stage == Shader::HwStage::Fragment) {
                const auto& p = x.hw.fs;
                const auto& q = y.hw.fs;
                if (!std::ranges::equal(p.color_buffers, q.color_buffers)) mask |= 8192;
                if (!(p.en_flags == q.en_flags) || !(p.addr_flags == q.addr_flags) ||
                    p.num_inputs != q.num_inputs ||
                    !std::ranges::equal(p.inputs.begin(), p.inputs.begin() + p.num_inputs,
                                        q.inputs.begin(), q.inputs.begin() + std::min(p.num_inputs, q.num_inputs))) mask |= 16384;
            } else if (x.hw_stage == Shader::HwStage::Vertex) {
                if (x.hw.vs.num_outputs != y.hw.vs.num_outputs || x.hw.vs.outputs != y.hw.vs.outputs) mask |= 16384;
                else mask |= 32768;
            }
        }
    }
    if (a.fetch_shader_data != b.fetch_shader_data) mask |= 2;
    if (a.vs_attribs.size() != b.vs_attribs.size()) {
        mask |= 4;
    } else if (a.vs_attribs != b.vs_attribs) {
        mask |= 8;
    }
    if (a.bitset != b.bitset || a.buffers.size() != b.buffers.size() ||
        a.images.size() != b.images.size() || a.samplers.size() != b.samplers.size()) {
        mask |= 16;
    } else {
        u32 binding = 0;
        for (size_t i = 0; i < a.buffers.size(); ++i) {
            if (b.bitset[binding++] && !(a.buffers[i] == b.buffers[i])) mask |= 32;
        }
        for (size_t i = 0; i < a.images.size(); ++i) {
            if (b.bitset[binding++] && !(a.images[i] == b.images[i])) mask |= 64;
        }
        for (size_t i = 0; i < a.samplers.size(); ++i) {
            if (!(a.samplers[i] == b.samplers[i])) mask |= 128;
        }
    }
    if (a.fmasks != b.fmasks) mask |= 256;
    if (b.bitset.any() && !(a.start == b.start)) mask |= 512;
    return mask;
}


int PredictTest(const char* cache_dir, const char* pack, const char* game_dir, const char* out_dir,
                u32 jobs) {
    const auto header = ReadFile(fs::path{cache_dir} / "profile.bin");
    if (header.size() != HeaderWords * 4 + sizeof(Shader::Profile)) {
        std::printf("predict: no profile.bin in %s\n", cache_dir);
        return 1;
    }
    Shader::Profile profile{};
    std::memcpy(&profile, header.data() + HeaderWords * 4, sizeof(profile));
    std::vector<RecipeProgram> programs;
    std::vector<RecipeKey> keys;
    if (!ReadPack(pack, programs, keys)) {
        std::printf("predict: %s is not for this build\n", pack);
        return 1;
    }
    GameShaders game;
    game.Load(game_dir, nullptr);
    // Donor classes: identical inputs apart from the program (its code, hash, index).
    struct Class {
        BbShaderCapture::Input in;
        std::unordered_set<u64> programs; ///< recorded with this input
    };
    std::vector<Class> classes;
    std::map<std::vector<u8>, size_t> class_of;
    for (const auto& program : programs) {
        for (size_t i = 0; i < program.perms.size(); ++i) {
            auto in = RecipeInput(program, i, game);
            if (!in) {
                continue;
            }
            auto key = *in;
            key.code.clear();
            key.pgm_hash = key.perm_idx = 0;
            auto [it, added] = class_of.emplace(BbShaderCapture::WriteInput(key), classes.size());
            if (added) {
                classes.push_back({std::move(*in), {}});
            }
            classes[it->second].programs.insert(program.hash);
        }
    }
    fs::create_directories(out_dir);
    {
        std::ofstream where{fs::path{out_dir} / "locations.txt"};
        for (const auto& [hash, at] : game.locations) {
            where << HexName(hash) << ' ' << at.first << ' ' << at.second << '\n';
        }
    }
    std::printf("predict: %zu programs, %zu donor classes, %zu programs in the game's files\n",
                programs.size(), classes.size(), game.Count());
    std::fflush(stdout);
    fs::create_directories(out_dir);
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
        signal(sig, PredictSignal);
    }
    // One child per program (a compiler assertion ends only it; it is started again after the
    // donor it was on). Lines: "true PERM ok|fail", "hit PERM CLASS", "fail CLASS".
    size_t next = 0;
    u32 running = 0;
    std::map<pid_t, std::pair<size_t, size_t>> children; // pid -> program, first class
    const auto start_child = [&](size_t p, size_t first) {
        const pid_t child = fork();
        if (child != 0) {
            if (child > 0) {
                children[child] = {p, first};
                ++running;
            }
            return;
        }
        const auto& program = programs[p];
        const fs::path out = fs::path{out_dir} / (HexName(program.hash) + ".txt");
        FILE* file = std::fopen(out.c_str(), first == 0 ? "w" : "a");
        const auto* code = game.Code(program.hash);
        std::vector<std::optional<PredictedSpec>> truth(program.perms.size());
        for (size_t i = 0; i < program.perms.size() && code; ++i) {
            if (auto in = RecipeInput(program, i, game)) {
                truth[i] = SpecWith(*in, program.hash, *code, profile);
            }
            if (first == 0) {
                std::fprintf(file, "true %zu %s\n", i, truth[i] ? "ok" : "fail");
            }
        }
        std::fflush(file);
        BbShaderCapture::Input own;
        if (program.perms.empty() || !BbShaderCapture::ReadInput(program.perms[0].input, own, true)) {
            std::fprintf(file, "done\n");
            std::fclose(file);
            _exit(0);
        }
        for (size_t c = first; c < classes.size() && code; ++c) {
            const auto& cls = classes[c];
            if (cls.in.hw_stage != own.hw_stage || cls.in.sw_stage != own.sw_stage) {
                continue;
            }
            std::fprintf(file, "try %zu\n", c);
            std::fflush(file);
            const auto spec = SpecWith(cls.in, program.hash, *code, profile);
            if (!spec) {
                std::fprintf(file, "fail %zu %d\n", c, predict_fail);
                continue;
            }
            for (size_t i = 0; i < truth.size(); ++i) {
                if (truth[i] && truth[i]->spec == spec->spec) {
                    std::fprintf(file, "hit %zu %zu\n", i, c);
                }
                if (truth[i]) {
                    std::fprintf(file, "diff %zu %zu %u\n", i, c, SpecDiff(spec->spec, truth[i]->spec));
                }
            }
        }
        std::fprintf(file, "done\n");
        std::fclose(file);
        _exit(0);
    };
    while (next < programs.size() || running) {
        while (running < jobs && next < programs.size()) {
            start_child(next++, 0);
        }
        int status = 0;
        const pid_t done = wait(&status);
        if (done <= 0) {
            break;
        }
        --running;
        const auto [p, first] = children[done];
        children.erase(done);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            continue;
        }
        // Crashed: the class it was trying is a failure, go on after it.
        const fs::path out = fs::path{out_dir} / (HexName(programs[p].hash) + ".txt");
        size_t last = ~size_t{0};
        std::ifstream log{out};
        for (std::string line; std::getline(log, line);) {
            if (line.rfind("try ", 0) == 0) {
                if (const size_t c = std::strtoull(line.c_str() + 4, nullptr, 10); c >= first) {
                    last = c;
                }
            }
        }
        if (last == ~size_t{0}) {
            std::ofstream{out, std::ios::app} << "crash-truth\n";
            continue;
        }
        std::ofstream{out, std::ios::app} << "crash " << last << '\n';
        start_child(p, last + 1);
    }
    // Class membership for the analysis.
    std::ofstream list{fs::path{out_dir} / "classes.txt"};
    for (size_t c = 0; c < classes.size(); ++c) {
        list << c << ' ' << u32(classes[c].in.hw_stage) << ' ' << u32(classes[c].in.sw_stage);
        for (const u64 hash : classes[c].programs) {
            list << ' ' << HexName(hash);
        }
        list << '\n';
    }
    return 0;
}
#endif

namespace {

/// A permutation another player's cache holds: its metadata only (no input record).
struct MetaTarget {
    u64 program = 0;
    u64 perm_idx = 0;
    u32 binary_version = 0;
    std::vector<u8> body; ///< the metadata after its header: specialization, then info
    std::unique_ptr<Shader::Info> info;
    Shader::StageSpecialization spec;
};

/// Metadata of version 7 (before the object motion buffer addresses) or 8.
std::optional<MetaTarget> ReadMetaTarget(const std::vector<u8>& bytes) {
    if (bytes.size() < 64) {
        return std::nullopt;
    }
    u32 version = 0;
    MetaTarget t;
    u64 perm_hash = 0;
    std::memcpy(&version, bytes.data(), 4);
    std::memcpy(&t.binary_version, bytes.data() + 4, 4);
    std::memcpy(&perm_hash, bytes.data() + 8, 8);
    std::memcpy(&t.perm_idx, bytes.data() + 16, 8);
    if (version != 7 && version != Vulkan::PipelineStore::ShaderMetaVersion) {
        return std::nullopt;
    }
    const size_t header = version == 7 ? 24 : 40;
    t.body.assign(bytes.begin() + header, bytes.end());
    t.info = std::make_unique<Shader::Info>();
    try {
        Serialization::Archive ar{std::vector<u8>(t.body)};
        t.spec.info = t.info.get();
        t.spec.Deserialize(ar);
        t.info->Deserialize(ar);
    } catch (const std::exception&) {
        return std::nullopt;
    }
    t.program = t.info->pgm_hash;
    if (HashCombine(t.program, t.perm_idx) != perm_hash) {
        return std::nullopt;
    }
    return t;
}

/// The metadata of `t` as permutation `perm_idx` of its program, in this build's version.
std::vector<u8> MetaAs(const MetaTarget& t, u64 perm_idx) {
    Serialization::Archive out;
    Serialization::Writer writer{out};
    writer.Write(Vulkan::PipelineStore::ShaderMetaVersion);
    writer.Write(t.binary_version);
    writer.Write(HashCombine(t.program, perm_idx));
    writer.Write(perm_idx);
    writer.Write(u64{0}); // object motion buffer addresses: the import puts its own
    writer.Write(u64{0});
    auto bytes = out.TakeOff();
    bytes.insert(bytes.end(), t.body.begin(), t.body.end());
    return bytes;
}

/// A recipe input record as permutation `perm_idx` of `program`.
std::vector<u8> InputAs(const std::vector<u8>& input, u64 program, u64 perm_idx) {
    BbShaderCapture::Input in;
    if (!BbShaderCapture::ReadInput(input, in, true)) {
        return {};
    }
    in.pgm_hash = program;
    in.perm_idx = perm_idx;
    in.code.clear();
    return BbShaderCapture::WriteInput(in);
}

bool SameSpec(const Shader::StageSpecialization& a, const Shader::StageSpecialization& b) {
    return a == b && b == a;
}

} // namespace


namespace {
struct KeyMerge {
    u32 added = 0, known = 0, dropped = 0;
};

/// Another cache's pipeline keys added to `keys`, their stages renamed through `perm_map` (their
/// permutation hash -> the merged pack's); one with a stage left out is dropped.
KeyMerge AddRenamedKeys(const std::vector<RecipeKey>& more_keys,
                        const std::unordered_map<u64, u64>& perm_map,
                        std::vector<RecipeKey>& keys) {
    KeyMerge result;
    std::unordered_set<std::string> names;
    for (const auto& key : keys) {
        names.insert(key.name);
    }
    for (const auto& key : more_keys) {
        if (key.bytes.size() < 8) {
            continue;
        }
        u32 is_compute = 0;
        std::memcpy(&is_compute, key.bytes.data() + 4, 4);
        std::vector<u8> bytes = key.bytes;
        std::string name;
        bool ok = true;
        if (is_compute) {
            Vulkan::ComputePipelineKey ck{};
            std::memcpy(&ck.value, bytes.data() + 8, sizeof(ck.value));
            const auto found = perm_map.find(ck.value);
            ok = found != perm_map.end();
            if (ok) {
                ck.value = found->second;
                std::memcpy(bytes.data() + 8, &ck.value, sizeof(ck.value));
                name = fmt::format("c_{:#018x}", ck.value);
            }
        } else {
            Vulkan::GraphicsPipelineKey gk{};
            if (bytes.size() < 8 + sizeof(gk)) {
                continue;
            }
            std::memcpy(&gk, bytes.data() + 8, sizeof(gk));
            for (auto& hash : gk.stage_hashes) {
                if (!hash) {
                    continue;
                }
                const auto found = perm_map.find(hash);
                if (found == perm_map.end()) {
                    ok = false;
                    break;
                }
                hash = found->second;
            }
            if (ok) {
                std::memcpy(bytes.data() + 8, &gk, sizeof(gk));
                name = fmt::format("g_{:#018x}", u64(std::hash<Vulkan::GraphicsPipelineKey>{}(gk)));
            }
        }
        if (!ok) {
            ++result.dropped;
        } else if (!names.insert(name).second) {
            ++result.known;
        } else {
            keys.push_back({name, std::move(bytes)});
            ++result.added;
        }
    }
    return result;
}
} // namespace
namespace {
/// A translation input rebuilt from a permutation's metadata alone: the user data and the flat
/// user data the shader read (its descriptors), the runtime info and start bindings of its
/// specialization; `regions` (fetch shaders) given by the caller.
BbShaderCapture::Input InputFromMeta(const MetaTarget& t, u64 pgm_base,
                                     std::vector<BbShaderCapture::Input::Region> regions) {
    BbShaderCapture::Input in;
    in.hw_stage = t.info->hw_stage;
    in.sw_stage = t.info->sw_stage;
    in.pgm_hash = t.program;
    in.perm_idx = t.perm_idx;
    in.pgm_base = pgm_base;
    const auto& flat = t.info->flattened_ud_buf;
    for (size_t i = 0; i < in.user_data.size() && i < flat.size(); ++i) {
        in.user_data[i] = flat[i];
    }
    in.flat = flat;
    in.runtime_info = t.spec.runtime_info;
    in.bindings = t.spec.start;
    in.regions = std::move(regions);
    return in;
}
} // namespace


namespace {
/// The vertex buffer descriptors a fetch shader reads in place (not in the flat user data): made
/// up from the specialization (each attribute's number class and component order), at the
/// addresses the user data pointers give. What the translator takes from them is what the
/// specialization records (the runtime reuses a module for any V# that matches it).
void SynthesizeVertexTables(const MetaTarget& t,
                            std::vector<BbShaderCapture::Input::Region>& regions) {
    if (!t.spec.fetch_shader_data) {
        return;
    }
    const auto& flat = t.info->flattened_ud_buf;
    const auto& attributes = t.spec.fetch_shader_data->attributes;
    for (size_t k = 0; k < attributes.size() && k < t.spec.vs_attribs.size(); ++k) {
        const auto& attribute = attributes[k];
        const u32 sgpr = attribute.sgpr_base;
        if (sgpr + 1 >= flat.size() || sgpr + 1 >= 16) {
            continue;
        }
        const u64 table = (u64(flat[sgpr]) | (u64(flat[sgpr + 1]) << 32)) & 0xFFFFFFFFFFFFull;
        if (table == 0) {
            continue;
        }
        const auto& spec = t.spec.vs_attribs[k];
        Gcn::Buffer buffer{};
        buffer.base_address = 0x10000;
        buffer.stride = 16;
        buffer.num_records = 0x100000;
        buffer.dst_sel_x = u32(spec.dst_select.r);
        buffer.dst_sel_y = u32(spec.dst_select.g);
        buffer.dst_sel_z = u32(spec.dst_select.b);
        buffer.dst_sel_w = u32(spec.dst_select.a);
        buffer.data_format = u32(Gcn::DataFormat::Format32_32_32_32);
        buffer.num_format = spec.num_class == Gcn::NumberClass::Sint   ? u32(Gcn::NumberFormat::Sint)
                            : spec.num_class == Gcn::NumberClass::Uint ? u32(Gcn::NumberFormat::Uint)
                                                                       : u32(Gcn::NumberFormat::Float);
        const auto* bytes = reinterpret_cast<const u8*>(&buffer);
        regions.push_back({table + u64(attribute.dword_offset) * 4,
                           std::vector<u8>(bytes, bytes + sizeof(buffer))});
    }
}
} // namespace

namespace {
/// A fetch shader that parses (Shader::Gcn::ParseFetchShader) into `data`: the game makes its
/// fetch shaders at run time from the vertex formats (Gnmx), so none is in its files; the
/// specialization keeps their parse. GCN (Sea Islands) encodings: SMRD s_load_dwordx4 per
/// attribute's V#, v_add_i32 of the base vertex/instance SGPRs, MUBUF buffer_load_format_* (MTBUF
/// tbuffer_load_format_* for typed loads), s_setpc_b64.
std::vector<u32> GenerateFetchShader(const Shader::Gcn::FetchShaderData& data) {
    std::vector<u32> code;
    const auto add_offset = [&](s8 sgpr, u32 vgpr) {
        if (sgpr >= 0) {
            code.push_back((37u << 25) | (vgpr << 17) | (vgpr << 9) | u32(sgpr)); // v_add_i32
        }
    };
    add_offset(data.vertex_offset_sgpr, 0);
    add_offset(data.instance_offset_sgpr, 3);
    constexpr u32 Quad = 2; // s[8:11]
    for (const auto& attribute : data.attributes) {
        code.push_back((0b11000u << 27) | (2u << 22) | ((Quad * 4) << 15) |
                       ((attribute.sgpr_base / 2) << 9) | (1u << 8) | (attribute.dword_offset & 0xff));
        const u32 op = std::clamp<u32>(attribute.num_elements, 1, 4) - 1;
        const bool typed = attribute.data_format != 0 || attribute.num_format != 0;
        const u32 word0 = typed ? (0b111010u << 26) | (u32(attribute.num_format & 7) << 23) |
                                      (u32(attribute.data_format & 15) << 19) | (op << 16) |
                                      (1u << 13) | (attribute.inst_offset & 0xfff)
                                : (0b111000u << 26) | (op << 18) | (1u << 13) |
                                      (attribute.inst_offset & 0xfff);
        code.push_back(word0);
        code.push_back(u32(attribute.instance_data) | (u32(attribute.dest_vgpr) << 8) |
                       (Quad << 16) | (0x80u << 24));
    }
    code.push_back((0b101111101u << 23) | (32u << 8)); // s_setpc_b64 s[0:1]
    return code;
}
} // namespace

namespace {
/// A permutation's input from its metadata alone (MetaReconstructTest: 96.9 % of the pack's give
/// their exact specialization): the flat user data, made-up vertex V# tables and a fetch shader
/// made from the specialization, at the addresses its user data holds.
BbShaderCapture::Input BuildMetaInput(const MetaTarget& t) {
    std::vector<BbShaderCapture::Input::Region> regions;
    SynthesizeVertexTables(t, regions);
    const auto& flat = t.info->flattened_ud_buf;
    const u32 fs_sgpr = t.info->fetch_shader_sgpr_base;
    const u64 fs_address = t.info->has_fetch_shader && fs_sgpr + 1 < flat.size()
                               ? (u64(flat[fs_sgpr]) | (u64(flat[fs_sgpr + 1]) << 32)) & 0xFFFFFFFFFFFFull
                               : 0;
    if (fs_address && t.spec.fetch_shader_data) {
        const auto fetch = GenerateFetchShader(*t.spec.fetch_shader_data);
        const auto* bytes = reinterpret_cast<const u8*>(fetch.data());
        regions.push_back({fs_address, std::vector<u8>(bytes, bytes + fetch.size() * 4)});
    }
    return InputFromMeta(t, 0x7f00000000ull, std::move(regions));
}
} // namespace
int MergeMetas(const char* cache_dir, const char* pack, const char* game_dir, const char* meta_dir,
               const char* out_pack, const char* work_dir, u32 jobs) {
    const auto header = ReadFile(fs::path{cache_dir} / "profile.bin");
    if (header.size() != HeaderWords * 4 + sizeof(Shader::Profile)) {
        std::printf("merge: no profile.bin in %s\n", cache_dir);
        return 1;
    }
    Shader::Profile profile{};
    std::memcpy(&profile, header.data() + HeaderWords * 4, sizeof(profile));
    std::vector<RecipeProgram> programs;
    std::vector<RecipeKey> keys;
    if (!ReadPack(pack, programs, keys)) {
        std::printf("merge: %s is not for this build\n", pack);
        return 1;
    }
    GameShaders game;
    game.Load(game_dir, nullptr);
    std::unordered_map<u64, size_t> program_index;
    for (size_t p = 0; p < programs.size(); ++p) {
        program_index[programs[p].hash] = p;
    }
    // The targets, by program.
    std::vector<MetaTarget> targets;
    u32 unreadable = 0, not_in_game = 0;
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator{meta_dir, ec}) {
        if (entry.path().extension() != ".meta") {
            continue;
        }
        auto t = ReadMetaTarget(ReadFile(entry.path()));
        if (!t) {
            ++unreadable;
        } else if (!game.Code(t->program)) {
            ++not_in_game;
        } else {
            targets.push_back(std::move(*t));
        }
    }
    std::ranges::sort(targets, [](const auto& a, const auto& b) {
        return std::tie(a.program, a.perm_idx) < std::tie(b.program, b.perm_idx);
    });
    // Donors: every permutation of the pack, with its stage and where its program lies.
    struct Donor {
        size_t program, perm;
        Shader::HwStage hw;
        Shader::SwStage sw;
    };
    std::vector<Donor> donors;
    for (size_t p = 0; p < programs.size(); ++p) {
        for (size_t i = 0; i < programs[p].perms.size(); ++i) {
            BbShaderCapture::Input in;
            if (BbShaderCapture::ReadInput(programs[p].perms[i].input, in, true)) {
                donors.push_back({p, i, in.hw_stage, in.sw_stage});
            }
        }
    }
    std::printf("merge: %zu permutations readable (%u unreadable, %u of programs not in the game's "
                "files), %zu donors in %zu programs\n",
                targets.size(), unreadable, not_in_game, donors.size(), programs.size());
    std::fflush(stdout);
    fs::create_directories(work_dir);
#ifndef _WIN32
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
        signal(sig, PredictSignal);
    }
#endif
    // One child per target (a compiler assertion ends only it; it goes on after that donor).
    // Lines: "have", "try D", "found D", "none".
    const auto order_for = [&](const MetaTarget& t) {
        std::vector<size_t> order;
        const auto here = game.locations.find(t.program);
        std::vector<std::pair<u64, size_t>> ranked;
        for (size_t d = 0; d < donors.size(); ++d) {
            const auto& donor = donors[d];
            if (donor.hw != t.info->hw_stage || donor.sw != t.info->sw_stage) {
                continue;
            }
            u64 rank = ~u64{0} >> 1;
            if (programs[donor.program].hash == t.program) {
                rank = 0;
            } else if (here != game.locations.end()) {
                const auto there = game.locations.find(programs[donor.program].hash);
                if (there != game.locations.end()) {
                    const u64 distance = there->second.second > here->second.second
                                             ? there->second.second - here->second.second
                                             : here->second.second - there->second.second;
                    rank = (there->second.first == here->second.first ? 1 : (u64{1} << 40)) +
                           distance;
                }
            }
            ranked.emplace_back(rank, d);
        }
        std::ranges::sort(ranked);
        for (const auto& [rank, d] : ranked) {
            order.push_back(d);
        }
        return order;
    };
    // One target's search, its log in work_dir/j ("try k" before each donor, then the result): a
    // translator crash ends the process there, and the search goes on from the next donor.
    const auto run_target = [&](size_t j, size_t first) {
        const auto& t = targets[j];
        FILE* file = std::fopen((fs::path{work_dir} / std::to_string(j)).string().c_str(), first ? "a" : "w");
        const auto* code = game.Code(t.program);
        if (first == 0) {
            // A crash before any donor leaves this line alone: the target is then given up.
            std::fprintf(file, "start\n");
            std::fflush(file);
            // Already in the pack: one of its permutations of this program has this specialization.
            if (const auto it = program_index.find(t.program); it != program_index.end()) {
                for (size_t i = 0; i < programs[it->second].perms.size(); ++i) {
                    const auto in = RecipeInput(programs[it->second], i, game);
                    const auto spec = in ? SpecWith(*in, t.program, *code, profile) : std::nullopt;
                    if (spec && SameSpec(spec->spec, t.spec)) {
                        std::fprintf(file, "have %zu\n", i);
                        std::fclose(file);
                        return;
                    }
                }
            }
            // Its own metadata (BuildMetaInput), when that gives its very specialization.
            if (code) {
                const auto in = BuildMetaInput(t);
                const auto spec = SpecWith(in, t.program, *code, profile);
                if (spec && SameSpec(spec->spec, t.spec)) {
                    std::fprintf(file, "meta\n");
                    std::fclose(file);
                    return;
                }
            }
        }
        const auto order = order_for(t);
        for (size_t k = first; k < order.size(); ++k) {
            const auto& donor = donors[order[k]];
            std::fprintf(file, "try %zu\n", k);
            std::fflush(file);
            const auto in = RecipeInput(programs[donor.program], donor.perm, game);
            const auto spec = in ? SpecWith(*in, t.program, *code, profile) : std::nullopt;
            if (spec && SameSpec(spec->spec, t.spec)) {
                std::fprintf(file, "found %zu\n", order[k]);
                std::fclose(file);
                return;
            }
        }
        std::fprintf(file, "none\n");
        std::fclose(file);
        return;
    };
#ifndef _WIN32
    const auto run_child = [&](size_t j, size_t first) -> pid_t {
        std::fflush(stdout);
        const pid_t child = fork();
        if (child != 0) {
            return child;
        }
        run_target(j, first);
        _exit(0);
    };
    std::map<pid_t, size_t> children;
    size_t next = 0;
    while (next < targets.size() || !children.empty()) {
        while (children.size() < jobs && next < targets.size()) {
            const size_t j = next++;
            if (const pid_t pid = run_child(j, 0); pid > 0) {
                children[pid] = j;
            }
        }
        int status = 0;
        const pid_t done = wait(&status);
        if (done <= 0) {
            break;
        }
        const size_t j = children[done];
        children.erase(done);
        if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            continue;
        }
        size_t last = ~size_t{0};
        std::ifstream log{fs::path{work_dir} / std::to_string(j)};
        for (std::string line; std::getline(log, line);) {
            if (line.rfind("try ", 0) == 0) {
                last = std::strtoull(line.c_str() + 4, nullptr, 10);
            }
        }
        if (last == ~size_t{0}) {
            std::ofstream{fs::path{work_dir} / std::to_string(j), std::ios::app} << "none\n";
            continue;
        }
        if (const pid_t pid = run_child(j, last + 1); pid > 0) {
            children[pid] = j;
        }
    }
#else
    // No fork (Windows): the targets one after another in this process (the shader recipe import,
    // bb-probe --import-shaders, its own process), resumed from the work folder after a crash: a
    // target with a result is done, one cut short goes on after the donor it was trying.
    (void)jobs;
    for (size_t j = 0; j < targets.size(); ++j) {
        const fs::path log_path = fs::path{work_dir} / std::to_string(j);
        size_t first = 0;
        bool started = false, finished = false;
        {
            std::ifstream log{log_path};
            for (std::string line; std::getline(log, line);) {
                started = true;
                if (line.rfind("try ", 0) == 0) {
                    first = std::strtoull(line.c_str() + 4, nullptr, 10) + 1;
                } else if (!line.empty() && line != "start") {
                    finished = true;
                }
            }
        }
        if (finished) {
            continue;
        }
        if (started && first == 0) {
            std::ofstream{log_path, std::ios::app} << "none\n"; // crashed before any donor
            continue;
        }
        run_target(j, first);
    }
#endif
    // Their permutation hashes -> the merged pack's (for their pipeline keys).
    std::unordered_map<u64, u64> perm_map;
    // The new pack: the old one, each program's new permutations after its own.
    u32 found_meta = 0;
    u32 have = 0, found_own = 0, found_other = 0, none = 0;
    std::unordered_set<u64> none_programs, none_new_programs;
    for (size_t j = 0; j < targets.size(); ++j) {
        const auto& t = targets[j];
        std::ifstream log{fs::path{work_dir} / std::to_string(j)};
        std::string result;
        for (std::string line; std::getline(log, line);) {
            if (line.rfind("try ", 0) != 0) {
                result = line;
            }
        }
        if (result.rfind("have ", 0) == 0) {
            perm_map[HashCombine(t.program, t.perm_idx)] =
                HashCombine(t.program, u64(std::strtoull(result.c_str() + 5, nullptr, 10)));
            ++have;
            continue;
        }
        if (result == "meta") {
            ++found_meta;
            auto it = program_index.find(t.program);
            if (it == program_index.end()) {
                it = program_index.emplace(t.program, programs.size()).first;
                programs.push_back({.hash = t.program});
            }
            auto& program = programs[it->second];
            const u64 perm_idx = program.perms.size();
            auto in = BuildMetaInput(t);
            in.pgm_hash = t.program;
            in.perm_idx = perm_idx;
            in.code.clear();
            RecipePerm perm;
            perm.input = BbShaderCapture::WriteInput(in);
            perm.meta = MetaAs(t, perm_idx);
            program.perms.push_back(std::move(perm));
            perm_map[HashCombine(t.program, t.perm_idx)] = HashCombine(t.program, perm_idx);
            continue;
        }
        if (result.rfind("found ", 0) != 0) {
            ++none;
            none_programs.insert(t.program);
            if (!program_index.contains(t.program)) {
                none_new_programs.insert(t.program);
            }
            continue;
        }
        const auto& donor = donors[std::strtoull(result.c_str() + 6, nullptr, 10)];
        const auto& donor_perm = programs[donor.program].perms[donor.perm];
        (programs[donor.program].hash == t.program ? found_own : found_other)++;
        auto it = program_index.find(t.program);
        if (it == program_index.end()) {
            it = program_index.emplace(t.program, programs.size()).first;
            programs.push_back({.hash = t.program});
        }
        auto& program = programs[it->second];
        const u64 perm_idx = program.perms.size();
        RecipePerm perm;
        perm.input = InputAs(donor_perm.input, t.program, perm_idx);
        perm.meta = MetaAs(t, perm_idx);
        perm.refs = donor_perm.refs;
        if (!perm.input.empty()) {
            program.perms.push_back(std::move(perm));
            perm_map[HashCombine(t.program, t.perm_idx)] = HashCombine(t.program, perm_idx);
        }
    }
    // Their pipelines (*.key in META_DIR or beside it, e.g. cache/key next to cache/meta).
    std::vector<RecipeKey> more_keys;
    for (const fs::path& root : {fs::path{meta_dir}, fs::path{meta_dir}.parent_path()}) {
        for (auto it = fs::recursive_directory_iterator{root, ec}; it != fs::recursive_directory_iterator{}; it.increment(ec)) {
            if (it->path().extension() == ".key") {
                auto bytes = ReadFile(it->path());
                u32 version = 0;
                if (bytes.size() >= 8) {
                    std::memcpy(&version, bytes.data(), 4);
                }
                if (version == Vulkan::PipelineStore::PipelineKeyVersion) {
                    more_keys.push_back({it->path().stem().string(), std::move(bytes)});
                }
            }
        }
        if (!more_keys.empty()) {
            break;
        }
    }
    const auto key_merge = AddRenamedKeys(more_keys, perm_map, keys);
    std::printf("merge: without an input: %zu programs (%zu of them not in the pack at all)\n",
                none_programs.size(), none_new_programs.size());
    size_t perm_count = 0;
    for (const auto& program : programs) {
        perm_count += program.perms.size();
    }
    const size_t written = WritePack(programs, keys, out_pack);
    std::printf("merge: %u already in the pack, %u new from their own metadata, %u new from the program's own inputs, %u new from "
                "other programs' inputs, %u without an input that matches; pipelines: %zu found, %u "
                "added, %u known, %u dropped\n"
                "merge: %zu programs, %zu permutations, %zu pipelines -> %s (%zu KiB)\n",
                have, found_meta, found_own, found_other, none, more_keys.size(), key_merge.added,
                key_merge.known, key_merge.dropped, programs.size(), perm_count, keys.size(),
                out_pack, written / 1024);
    return written ? 0 : 1;
}

#ifndef _WIN32
int MetaReconstructTest(const char* cache_dir, const char* pack, const char* game_dir, u32 jobs) {
    const auto header = ReadFile(fs::path{cache_dir} / "profile.bin");
    if (header.size() != HeaderWords * 4 + sizeof(Shader::Profile)) {
        return 1;
    }
    Shader::Profile profile{};
    std::memcpy(&profile, header.data() + HeaderWords * 4, sizeof(profile));
    std::vector<RecipeProgram> programs;
    std::vector<RecipeKey> keys;
    if (!ReadPack(pack, programs, keys)) {
        return 1;
    }
    GameShaders game;
    game.Load(game_dir, nullptr);
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
        struct sigaction action{};
        action.sa_sigaction = PredictSigaction;
        action.sa_flags = SA_SIGINFO | SA_NODEFER;
        sigaction(sig, &action, nullptr);
    }
    u32 total = 0, hits = 0, failed = 0, unreadable = 0;
    std::map<std::string, u32> reasons;
    for (const auto& program : programs) {
        const auto* code = game.Code(program.hash);
        for (size_t i = 0; i < program.perms.size() && code; ++i) {
            const auto t = ReadMetaTarget(program.perms[i].meta);
            if (!t) {
                ++unreadable;
                continue;
            }
            ++total;
            // The recipe's own memory only for its programs (fetch shaders): the data regions are
            // what the metadata has to replace.
            const auto full = RecipeInput(program, i, game);
            std::vector<BbShaderCapture::Input::Region> code_regions;
            // The fetch shader goes where the metadata's own user data points.
            const auto& flat = t->info->flattened_ud_buf;
            const u32 fs_sgpr = t->info->fetch_shader_sgpr_base;
            const u64 fs_address =
                t->info->has_fetch_shader && fs_sgpr + 1 < flat.size()
                    ? (u64(flat[fs_sgpr]) | (u64(flat[fs_sgpr + 1]) << 32)) & 0xFFFFFFFFFFFFull
                    : 0;
            for (const auto& ref : program.perms[i].refs) {
                if (ref.address == ~u64{0}) continue;
                const auto* ref_code = game.Code(ref.hash);
                const auto* bytes = reinterpret_cast<const u8*>(ref_code->data());
                code_regions.push_back({fs_address ? fs_address : ref.address,
                                        std::vector<u8>(bytes, bytes + ref.size)});
            }
            static const bool synth = !bb_env("BB_META_NO_SYNTH");
            if (synth) {
                SynthesizeVertexTables(*t, code_regions);
            }
            static const bool gen_fetch = !bb_env("BB_META_RECIPE_FETCH");
            if (gen_fetch && fs_address && t->spec.fetch_shader_data) {
                // The fetch shader made from the specialization, not the recipe's.
                std::erase_if(code_regions, [&](const auto& r) { return r.address == fs_address; });
                const auto fetch = GenerateFetchShader(*t->spec.fetch_shader_data);
                const auto* bytes = reinterpret_cast<const u8*>(fetch.data());
                code_regions.push_back({fs_address, std::vector<u8>(bytes, bytes + fetch.size() * 4)});
            }
            static const bool fixed_base = bb_env("BB_META_FIXED_BASE") != nullptr;
            auto in = InputFromMeta(*t, full && !fixed_base ? full->pgm_base : 0x7f00000000ull, code_regions);
            if (full) {
                in.vs_copy = full->vs_copy;
            }
            int fds[2];
            if (pipe(fds) != 0) return 1;
            std::fflush(stdout);
            const pid_t child = fork();
            if (child == 0) {
                close(fds[0]);
                const auto spec = SpecWith(in, program.hash, *code, profile);
                char r = !spec ? char('0' + predict_fail) : SameSpec(spec->spec, t->spec) ? 'H' : 'M';
                if (!spec && predict_fail == 2) {
                    Dl_info dl{};
                    dladdr(reinterpret_cast<void*>(predict_fault_rip), &dl);
                    const u64 a = predict_fault_addr;
                    u64 nearest = ~u64{0};
                    for (u32 ud = 0; ud + 1 < 16 && ud + 1 < in.flat.size(); ++ud) {
                        const u64 p = (u64(in.flat[ud]) | (u64(in.flat[ud + 1]) << 32)) & 0xFFFFFFFFFFFFull;
                        if (p && a >= p && a - p < nearest) nearest = a - p;
                    }
                    std::fprintf(stderr, "FAULT %s %#llx addr %#llx from-ud+%#llx pgm_base %#llx\n",
                                 dl.dli_sname ? dl.dli_sname : "?", (unsigned long long)(predict_fault_rip - (u64)dl.dli_fbase),
                                 (unsigned long long)a, (unsigned long long)nearest, (unsigned long long)in.pgm_base);
                }
                if (r == 'M') {
                    r = char('a' + std::min<u32>(25, std::countr_zero(SpecDiff(spec->spec, t->spec) | (1u << 25))));
                }
                [[maybe_unused]] auto w = write(fds[1], &r, 1);
                _exit(0);
            }
            close(fds[1]);
            char r = 'X';
            if (read(fds[0], &r, 1) != 1) r = 'X';
            close(fds[0]);
            int status = 0;
            waitpid(child, &status, 0);
            if (r == 'H') {
                ++hits;
            } else {
                ++failed;
                ++reasons[std::string(1, r)];
            }
        }
    }
    std::printf("meta reconstruct: %u permutations, %u exact (%.1f%%), %u not (%u unreadable)\n", total,
                hits, total ? 100.0 * hits / total : 0.0, failed, unreadable);
    for (const auto& [r, n] : reasons) {
        std::printf("  reason %s: %u\n", r.c_str(), n);
    }
    (void)jobs;
    return 0;
}
int MergePacks(const char* cache_dir, const char* pack_a, const char* pack_b, const char* game_dir,
               const char* out_pack, const char* work_dir, u32 jobs) {
    const auto header = ReadFile(fs::path{cache_dir} / "profile.bin");
    if (header.size() != HeaderWords * 4 + sizeof(Shader::Profile)) {
        std::printf("merge: no profile.bin in %s\n", cache_dir);
        return 1;
    }
    Shader::Profile profile{};
    std::memcpy(&profile, header.data() + HeaderWords * 4, sizeof(profile));
    std::vector<RecipeProgram> programs, more;
    std::vector<RecipeKey> keys, more_keys;
    if (!ReadPack(pack_a, programs, keys) || !ReadPack(pack_b, more, more_keys)) {
        std::printf("merge: a pack is not for this build\n");
        return 1;
    }
    GameShaders game;
    game.Load(game_dir, nullptr);
    std::unordered_map<u64, size_t> program_index;
    for (size_t p = 0; p < programs.size(); ++p) {
        program_index[programs[p].hash] = p;
    }
    fs::create_directories(work_dir);
    for (int sig : {SIGSEGV, SIGBUS, SIGILL, SIGFPE}) {
        signal(sig, PredictSignal);
    }
    // One child per program of the second pack: for each of its permutations, the first pack's
    // permutation with the same specialization ("same I J"), none ("new I"), or its own input does
    // not translate ("bad I"). A compiler assertion ends only that child (its program is skipped).
    const auto run_child = [&](size_t q) -> pid_t {
        std::fflush(stdout);
        const pid_t child = fork();
        if (child != 0) {
            return child;
        }
        const auto& program = more[q];
        FILE* file = std::fopen((fs::path{work_dir} / std::to_string(q)).c_str(), "w");
        const auto* code = game.Code(program.hash);
        std::vector<std::optional<PredictedSpec>> own;
        if (const auto it = program_index.find(program.hash); it != program_index.end() && code) {
            for (size_t i = 0; i < programs[it->second].perms.size(); ++i) {
                const auto in = RecipeInput(programs[it->second], i, game);
                own.push_back(in ? SpecWith(*in, program.hash, *code, profile) : std::nullopt);
            }
        }
        for (size_t i = 0; i < program.perms.size(); ++i) {
            const auto in = code ? RecipeInput(program, i, game) : std::nullopt;
            const auto spec = in ? SpecWith(*in, program.hash, *code, profile) : std::nullopt;
            if (!spec) {
                std::fprintf(file, "bad %zu\n", i);
                continue;
            }
            size_t same = ~size_t{0};
            for (size_t j = 0; j < own.size() && same == ~size_t{0}; ++j) {
                if (own[j] && SameSpec(own[j]->spec, spec->spec)) {
                    same = j;
                }
            }
            if (same != ~size_t{0}) {
                std::fprintf(file, "same %zu %zu\n", i, same);
            } else {
                std::fprintf(file, "new %zu\n", i);
            }
        }
        std::fprintf(file, "done\n");
        std::fclose(file);
        _exit(0);
    };
    std::map<pid_t, size_t> children;
    for (size_t next = 0; next < more.size() || !children.empty();) {
        while (children.size() < jobs && next < more.size()) {
            if (const pid_t pid = run_child(next); pid > 0) {
                children[pid] = next;
            }
            ++next;
        }
        int status = 0;
        const pid_t done = wait(&status);
        if (done <= 0) {
            break;
        }
        children.erase(done);
    }
    // Permutation hashes of the second pack -> the merged pack's.
    std::unordered_map<u64, u64> perm_map;
    u32 same = 0, added = 0, bad = 0, skipped_programs = 0;
    for (size_t q = 0; q < more.size(); ++q) {
        const auto& program = more[q];
        std::ifstream log{fs::path{work_dir} / std::to_string(q)};
        std::vector<std::string> lines;
        for (std::string line; std::getline(log, line);) {
            lines.push_back(line);
        }
        if (lines.empty() || lines.back() != "done") {
            ++skipped_programs;
            continue;
        }
        auto it = program_index.find(program.hash);
        for (const auto& line : lines) {
            size_t i = 0, j = 0;
            if (std::sscanf(line.c_str(), "same %zu %zu", &i, &j) == 2) {
                perm_map[HashCombine(program.hash, u64(i))] = HashCombine(program.hash, u64(j));
                ++same;
            } else if (std::sscanf(line.c_str(), "new %zu", &i) == 1) {
                if (it == program_index.end()) {
                    it = program_index.emplace(program.hash, programs.size()).first;
                    programs.push_back({.hash = program.hash});
                }
                auto& target = programs[it->second];
                const u64 perm_idx = target.perms.size();
                auto meta = ReadMetaTarget(program.perms[i].meta);
                RecipePerm perm;
                perm.input = InputAs(program.perms[i].input, program.hash, perm_idx);
                perm.refs = program.perms[i].refs;
                if (!meta || perm.input.empty()) {
                    ++bad;
                    continue;
                }
                perm.meta = MetaAs(*meta, perm_idx);
                target.perms.push_back(std::move(perm));
                perm_map[HashCombine(program.hash, u64(i))] = HashCombine(program.hash, perm_idx);
                ++added;
            } else if (line.rfind("bad ", 0) == 0) {
                ++bad;
            }
        }
    }
    // The second pack's pipelines, their stages renamed.
    const auto [keys_added, keys_known, keys_dropped] = AddRenamedKeys(more_keys, perm_map, keys);
    size_t perm_count = 0;
    for (const auto& program : programs) {
        perm_count += program.perms.size();
    }
    const size_t written = WritePack(programs, keys, out_pack);
    std::printf("merge: %u permutations already in the first pack, %u added, %u not translatable "
                "(%u programs skipped after a compiler failure); pipelines: %u added, %u known, "
                "%u dropped\n"
                "merge: %zu programs, %zu permutations, %zu pipelines -> %s (%zu KiB)\n",
                same, added, bad, skipped_programs, keys_added, keys_known, keys_dropped,
                programs.size(), perm_count, keys.size(), out_pack, written / 1024);
    return written ? 0 : 1;
}
#endif
} // namespace BbShaderRebuild

#ifndef _WIN32
/// bb-probe --predict-shaders-test CACHE_DIR PACK GAME_DIR OUT_DIR JOBS (shader input prediction).
extern "C" int bbgpu_predict_shaders_test(const char* cache_dir, const char* pack,
                                          const char* game_dir, const char* out_dir, int jobs) {
    return BbShaderRebuild::PredictTest(cache_dir, pack, game_dir, out_dir, u32(jobs));
}
#endif

/// bb-probe --merge-shader-metas CACHE_DIR PACK GAME_DIR META_DIR OUT_PACK WORK_DIR JOBS: another
/// player's cache (metadata only) added to a recipe pack where an input gives its specialization.
extern "C" int bbgpu_merge_shader_metas(const char* cache_dir, const char* pack, const char* game_dir,
                                        const char* meta_dir, const char* out_pack,
                                        const char* work_dir, int jobs) {
    return BbShaderRebuild::MergeMetas(cache_dir, pack, game_dir, meta_dir, out_pack, work_dir,
                                       u32(jobs));
}

#ifndef _WIN32
/// bb-probe --merge-shader-packs CACHE_DIR PACK_A PACK_B GAME_DIR OUT_PACK WORK_DIR JOBS.
extern "C" int bbgpu_merge_shader_packs(const char* cache_dir, const char* pack_a,
                                        const char* pack_b, const char* game_dir,
                                        const char* out_pack, const char* work_dir, int jobs) {
    return BbShaderRebuild::MergePacks(cache_dir, pack_a, pack_b, game_dir, out_pack, work_dir,
                                       u32(jobs));
}
#endif

#ifndef _WIN32
/// bb-probe --meta-reconstruct-test CACHE_DIR PACK GAME_DIR (diagnostics).
extern "C" int bbgpu_meta_reconstruct_test(const char* cache_dir, const char* pack, const char* game_dir) {
    return BbShaderRebuild::MetaReconstructTest(cache_dir, pack, game_dir, 1);
}
#endif
