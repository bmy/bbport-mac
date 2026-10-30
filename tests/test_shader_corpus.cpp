// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Shader corpus replay: compiles every compilation recorded with BB_SHADER_CAPTURE
// (render/shader_capture.h) again and compares the output with the recorded one: SPIR-V word
// for word, the shader info's serialized form and the bindings. The guest memory the compiler
// read is mapped back at its addresses. Usage: shader-corpus-test <dir|file>... [-v]
#include <signal.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "common/serdes.h"
#include "shader/backend/bindings.h"
#include "shader/backend/spirv/emit_spirv.h"
#include "shader/backend/spirv/emit_spirv_discard_frag.h"
#include "shader/backend/spirv/emit_spirv_quad_rect.h"
#include "shader/info.h"
#include "shader/profile.h"
#include "shader/recompiler.h"
#include "shader/runtime_info.h"

namespace {

struct Reader {
    const std::vector<u8>& bytes;
    size_t at = 0;
    bool ok = true;
    void Raw(void* out, size_t size) {
        if (at + size > bytes.size()) {
            ok = false;
            return;
        }
        std::memcpy(out, bytes.data() + at, size);
        at += size;
    }
    template <typename T>
    T Value() {
        T value{};
        Raw(&value, sizeof(T));
        return value;
    }
    std::vector<u32> Words() {
        std::vector<u32> words(Value<u32>());
        Raw(words.data(), words.size() * sizeof(u32));
        return words;
    }
};

struct Region {
    u64 address;
    std::vector<u8> bytes;
};

/// Guest memory of one record mapped at its addresses (whole pages, unmapped afterwards).
class GuestMemory {
public:
    bool Map(const std::vector<Region>& regions) {
        std::map<u64, u64> pages; // first page -> end
        for (const auto& region : regions) {
            const u64 first = region.address & ~u64{0xFFF};
            const u64 end = (region.address + region.bytes.size() + 0xFFF) & ~u64{0xFFF};
            pages[first] = std::max(pages[first], end);
        }
        // Merge overlapping page spans.
        std::vector<std::pair<u64, u64>> spans;
        for (const auto& [first, end] : pages) {
            if (!spans.empty() && first <= spans.back().second) {
                spans.back().second = std::max(spans.back().second, end);
            } else {
                spans.emplace_back(first, end);
            }
        }
        for (const auto& [first, end] : spans) {
            void* at = mmap(reinterpret_cast<void*>(first), end - first, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (at != reinterpret_cast<void*>(first)) {
                std::fprintf(stderr, "cannot map guest memory at %#llx\n",
                             (unsigned long long)first);
                return false;
            }
            mapped.emplace_back(first, end - first);
        }
        for (const auto& region : regions) {
            std::memcpy(reinterpret_cast<void*>(region.address), region.bytes.data(),
                        region.bytes.size());
        }
        return true;
    }
    ~GuestMemory() {
        for (const auto& [address, size] : mapped) {
            munmap(reinterpret_cast<void*>(address), size);
        }
    }

private:
    std::vector<std::pair<u64, u64>> mapped;
};


/// The serialized shader info's parts (Info::Serialize): the persistent fields as bytes, the
/// flattened user data, the SRT walker.
struct SerializedInfo {
    Shader::InfoPersistent persistent;
    std::vector<u32> flat;
    Shader::PersistentSrtInfo srt;
    std::vector<u8> walker;

    bool Parse(const std::vector<u8>& bytes) {
        Reader in{bytes};
        in.Raw(&persistent, sizeof(persistent));
        flat.resize(in.Value<size_t>());
        in.Raw(flat.data(), flat.size() * sizeof(u32));
        in.Raw(&srt, sizeof(srt));
        walker.resize(srt.walker_func_size);
        in.Raw(walker.data(), walker.size());
        return in.ok;
    }
};

/// What differs between two serialized infos, field by field (padding bytes ignored); empty
/// when nothing does.
std::string InfoDifference(const std::vector<u8>& recorded, const std::vector<u8>& replayed) {
    SerializedInfo a, b;
    if (!a.Parse(recorded) || !b.Parse(replayed)) {
        return "unreadable";
    }
    const auto& x = a.persistent;
    const auto& y = b.persistent;
#define SAME(field)                                                                            \
    if (!(x.field == y.field)) {                                                               \
        return #field;                                                                         \
    }
    SAME(buffers.size());
    for (size_t i = 0; i < x.buffers.size(); ++i) {
        const auto& p = x.buffers[i];
        const auto& q = y.buffers[i];
        if (!(p.sharp_fetch == q.sharp_fetch) || p.used_types != q.used_types ||
            p.buffer_type != q.buffer_type || p.is_written != q.is_written ||
            p.is_formatted != q.is_formatted || p.post_op != q.post_op ||
            p.post_op_dw1_mask != q.post_op_dw1_mask) {
            return "buffer " + std::to_string(i);
        }
    }
    SAME(images.size());
    for (size_t i = 0; i < x.images.size(); ++i) {
        const auto& p = x.images[i];
        const auto& q = y.images[i];
        if (!(p.sharp_fetch == q.sharp_fetch) || p.is_depth != q.is_depth ||
            p.is_atomic != q.is_atomic || p.is_array != q.is_array ||
            p.is_written != q.is_written || p.is_r128 != q.is_r128 ||
            p.needs_native != q.needs_native || p.constant_mip_index != q.constant_mip_index ||
            p.mip_fallback_mode != q.mip_fallback_mode || p.post_op != q.post_op) {
            return "image " + std::to_string(i);
        }
    }
    SAME(samplers.size());
    for (size_t i = 0; i < x.samplers.size(); ++i) {
        const auto& p = x.samplers[i];
        const auto& q = y.samplers[i];
        if (!(p.sharp_fetch == q.sharp_fetch) || p.post_op != q.post_op ||
            p.post_op_tsharp_dw3_off != q.post_op_tsharp_dw3_off || p.is_depth != q.is_depth) {
            return "sampler " + std::to_string(i);
        }
    }
    SAME(fmasks.size());
    for (size_t i = 0; i < x.fmasks.size(); ++i) {
        if (x.fmasks[i].sharp_idx != y.fmasks[i].sharp_idx) {
            return "fmask " + std::to_string(i);
        }
    }
    SAME(ud_mask.mask);
    SAME(fetch_shader_sgpr_base);
    SAME(shared_memory_scratch_size);
    SAME(pgm_hash);
    SAME(tess_consts_dword_offset);
    SAME(tess_consts_ptr_base);
    SAME(hw_stage);
    SAME(sw_stage);
    SAME(mrt_mask);
    SAME(has_fetch_shader);
    SAME(has_bitwise_xor);
    SAME(uses_dma);
#undef SAME
    if (a.flat != b.flat) {
        return "flattened user data";
    }
    if (a.srt.flattened_bufsize_dw != b.srt.flattened_bufsize_dw || a.walker != b.walker) {
        return "SRT walker";
    }
    return {};
}


/// Profiles of other GPUs, to record what the compiler makes for them (paths the game's own GPU
/// never takes): `--make-variants DIR` writes one record per variant, outputs from this build.
struct Variant {
    const char* name;
    void (*apply)(Shader::Profile&, Shader::RuntimeInfo&);
};
void NvidiaCommon(Shader::Profile& p, Shader::RuntimeInfo& rt) {
    p.subgroup_size = 32;
    p.supports_amd_shader_explicit_vertex_parameter = false;
    p.supports_native_cube_calc = false;
    p.supports_trinary_minmax = false;
    p.supports_image_load_store_lod = false;
    p.needs_lds_barriers = true;
    p.needs_buffer_offsets = true;
    p.needs_clip_distance_emulation = true;
    if (rt.hw_stage == Shader::HwStage::Fragment) {
        rt.hw.fs.clip_distance_emulation = true;
    }
}
const Variant variants[] = {
    {"turing",
     [](Shader::Profile& p, Shader::RuntimeInfo& rt) {
         NvidiaCommon(p, rt);
         p.supports_fragment_shader_barycentric = true;
         p.needs_manual_interpolation = true;
     }},
    {"pascal",
     [](Shader::Profile& p, Shader::RuntimeInfo& rt) {
         NvidiaCommon(p, rt);
         p.supports_fragment_shader_barycentric = false;
         p.needs_manual_interpolation = false;
         p.needs_integer_interpolation_fix = true;
     }},
    {"minimal",
     [](Shader::Profile& p, Shader::RuntimeInfo&) {
         p.support_float64 = false;
         p.support_int64 = false;
         p.support_int16 = false;
         p.support_int8 = false;
         p.support_float16 = false;
         p.supports_buffer_int64_atomics = false;
         p.supports_shared_int64_atomics = false;
         p.supports_workgroup_explicit_memory_layout = false;
         p.supports_buffer_fp32_atomic_min_max = false;
         p.supports_image_fp32_atomic_min_max = false;
         p.supports_shader_stencil_export = false;
         p.supports_shader_subgroup_clock = false;
     }},
    {"intel",
     [](Shader::Profile& p, Shader::RuntimeInfo&) {
         p.subgroup_size = 16;
         p.supports_amd_shader_explicit_vertex_parameter = false;
         p.supports_fragment_shader_barycentric = true;
         p.supports_native_cube_calc = false;
         p.supports_trinary_minmax = false;
         p.supports_image_load_store_lod = false;
         p.has_broken_spirv_clamp = true;
         p.needs_buffer_offsets = true;
     }},
};
const char* variant_dir = nullptr;

struct Out {
    std::vector<u8> bytes;
    void Raw(const void* data, size_t size) {
        const auto* p = static_cast<const u8*>(data);
        bytes.insert(bytes.end(), p, p + size);
    }
    template <typename T>
    void Value(const T& value) {
        Raw(&value, sizeof(T));
    }
    void Words(std::span<const u32> words) {
        Value(u32(words.size()));
        Raw(words.data(), words.size_bytes());
    }
};

enum class Outcome { Same, SpirvDiffers, InfoDiffers, BindingsDiffer, Unreadable, Nondeterministic };

/// The capture format (shader_capture.cpp): bbport, macOS 5 (Profile's tess flags).
#ifdef __APPLE__
constexpr u32 CaptureVersion = 5;
#else
constexpr u32 CaptureVersion = 2;
#endif

Outcome Replay(const std::string& path, bool verbose, u32 repeat) {
    std::ifstream file{path, std::ios::binary};
    const std::vector<u8> bytes{std::istreambuf_iterator<char>(file), {}};
    Reader in{bytes};
    char magic[4];
    in.Raw(magic, 4);
    if (std::memcmp(magic, "BBSC", 4) != 0 || in.Value<u32>() != CaptureVersion) {
        return Outcome::Unreadable;
    }
    Shader::MotionVectors::params_address = in.Value<u64>();
    Shader::MotionVectors::positions_address = in.Value<u64>();
    const auto hw_stage = Shader::HwStage(in.Value<u32>());
    const auto sw_stage = Shader::SwStage(in.Value<u32>());
    const u64 pgm_hash = in.Value<u64>();
    in.Value<u64>(); // permutation
    const u64 pgm_base = in.Value<u64>();
    alignas(16) auto user_data = in.Value<std::array<u32, Shader::ShaderParams::NumShaderUserData>>();
    const auto code = in.Words();
    auto runtime_info = in.Value<Shader::RuntimeInfo>();
    const auto vs_copy = in.Words();
    const auto profile = in.Value<Shader::Profile>();
    auto bindings = in.Value<Shader::Backend::Bindings>();
    std::vector<Region> regions(in.Value<u32>());
    for (auto& region : regions) {
        region.address = in.Value<u64>();
        region.bytes.resize(in.Value<u32>());
        in.Raw(region.bytes.data(), region.bytes.size());
    }
    const auto spv = in.Words();
    std::vector<u8> info_bytes(in.Value<u64>());
    in.Raw(info_bytes.data(), info_bytes.size());
    const auto out_bindings = in.Value<Shader::Backend::Bindings>();
    if (!in.ok) {
        return Outcome::Unreadable;
    }

    // BB_CORPUS_DUMP=<dir>: the recorded SPIR-V and the GCN program of each record, by name.
    if (const char* dump = std::getenv("BB_CORPUS_DUMP")) {
        const std::string base = std::string{dump} + "/" +
                                 std::filesystem::path{path}.filename().string();
        const auto put = [](const std::string& name, std::span<const u32> words) {
            std::ofstream{name, std::ios::binary}.write(reinterpret_cast<const char*>(words.data()),
                                                        words.size_bytes());
        };
        put(base + ".spv", spv);
        put(base + ".gcn", code);
    }
    const std::vector<Region> recorded_regions = regions;
    // The program at its address, as the compiler saw it.
    regions.push_back({pgm_base, std::vector<u8>(reinterpret_cast<const u8*>(code.data()),
                                                 reinterpret_cast<const u8*>(code.data()) +
                                                     code.size() * sizeof(u32))});
    GuestMemory memory;
    if (!memory.Map(regions)) {
        return Outcome::Unreadable;
    }
    if (hw_stage == Shader::HwStage::Geometry) {
        runtime_info.hw.gs.vs_copy = vs_copy;
    }
    const std::span<const u32> guest_code{reinterpret_cast<const u32*>(pgm_base), code.size()};
    Shader::Info info{hw_stage, sw_stage,
                      Shader::ShaderParams{std::span<const u32, 16>{user_data}, guest_code,
                                           pgm_hash}};
    // The SRT walker's result as it was in the game (from the recorded info).
    std::vector<u32> flat;
    if (info_bytes.size() >= sizeof(Shader::InfoPersistent) + sizeof(size_t)) {
        size_t words;
        std::memcpy(&words, info_bytes.data() + sizeof(Shader::InfoPersistent), sizeof(size_t));
        flat.resize(words);
        std::memcpy(flat.data(), info_bytes.data() + sizeof(Shader::InfoPersistent) + sizeof(size_t),
                    words * sizeof(u32));
    }
    Shader::flat_buf_replay = &flat;
    static Shader::Pools pools;
    const auto pristine_runtime_info = runtime_info;
    const auto pristine_bindings = bindings;
    const auto compile = [&](Shader::Info& target, Shader::Backend::Bindings& target_bindings) {
        auto rt = pristine_runtime_info;
        target_bindings = pristine_bindings;
        const auto program = Shader::TranslateProgram(guest_code, pools, target, rt, profile);
        return Shader::Backend::SPIRV::EmitSPIRV(profile, rt, program, target_bindings);
    };
    if (variant_dir) {
        // Each variant in a child process: an assertion of the compiler ends only that one.
        for (const auto& variant : variants) {
            const pid_t child = fork();
            if (child == 0) {
                Shader::Profile vprofile = profile;
                Shader::RuntimeInfo vrt = pristine_runtime_info;
                variant.apply(vprofile, vrt);
                Shader::Info vinfo{hw_stage, sw_stage,
                                   Shader::ShaderParams{std::span<const u32, 16>{user_data},
                                                        guest_code, pgm_hash}};
                Shader::Backend::Bindings vbindings = pristine_bindings;
                auto rt = vrt;
                const auto program =
                    Shader::TranslateProgram(guest_code, pools, vinfo, rt, vprofile);
                const auto vspv = Shader::Backend::SPIRV::EmitSPIRV(vprofile, rt, program, vbindings);
                Serialization::Archive archive;
                vinfo.Serialize(archive);
                const auto vinfo_bytes = archive.TakeOff();
                Out out;
                out.Raw("BBSC", 4);
                out.Value(CaptureVersion);
                out.Value(Shader::MotionVectors::params_address);
                out.Value(Shader::MotionVectors::positions_address);
                out.Value(u32(hw_stage));
                out.Value(u32(sw_stage));
                out.Value(pgm_hash);
                out.Value(u64(0));
                out.Value(pgm_base);
                out.Value(user_data);
                out.Words(code);
                auto stored_rt = vrt;
                if (hw_stage == Shader::HwStage::Geometry) {
                    stored_rt.hw.gs.vs_copy = {};
                }
                out.Value(stored_rt);
                out.Words(vs_copy);
                out.Value(vprofile);
                out.Value(pristine_bindings);
                out.Value(u32(recorded_regions.size()));
                for (const auto& region : recorded_regions) {
                    out.Value(region.address);
                    out.Value(u32(region.bytes.size()));
                    out.Raw(region.bytes.data(), region.bytes.size());
                }
                out.Words(vspv);
                out.Value(u64(vinfo_bytes.size()));
                out.Raw(vinfo_bytes.data(), vinfo_bytes.size());
                out.Value(vbindings);
                const std::string name = std::string{variant_dir} + "/" +
                                         std::filesystem::path(path).stem().string() + "_" +
                                         variant.name + ".bbsc";
                std::ofstream file{name, std::ios::binary};
                file.write(reinterpret_cast<const char*>(out.bytes.data()), out.bytes.size());
                _exit(file ? 0 : 1);
            }
            int status = 0;
            waitpid(child, &status, 0);
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
                std::printf("%s: variant %s failed to compile\n", path.c_str(), variant.name);
            }
        }
        return Outcome::Same;
    }
    const auto out_spv = compile(info, bindings);
    // Determinism: the same compilation again, with the heap laid out differently.
    for (u32 i = 0; i < repeat; ++i) {
        std::vector<std::unique_ptr<u8[]>> junk;
        for (u32 j = 0; j < 1 + i * 7 % 13; ++j) {
            junk.emplace_back(new u8[16 + (i * 131 + j * 977) % 4096]);
        }
        Shader::Info again{hw_stage, sw_stage,
                           Shader::ShaderParams{std::span<const u32, 16>{user_data}, guest_code,
                                                pgm_hash}};
        Shader::Backend::Bindings again_bindings;
        if (compile(again, again_bindings) != out_spv) {
            if (verbose) {
                std::printf("%s: NONDETERMINISTIC (run %u differs from the first)\n",
                            path.c_str(), i + 1);
            }
            return Outcome::Nondeterministic;
        }
    }

    if (out_spv != spv) {
        if (verbose) {
            size_t first = 0;
            while (first < std::min(spv.size(), out_spv.size()) && spv[first] == out_spv[first]) {
                ++first;
            }
            std::printf("%s: SPIR-V differs at word %zu (%zu vs %zu words)\n", path.c_str(),
                        first, out_spv.size(), spv.size());
            if (const char* dump = std::getenv("CORPUS_DUMP")) {
                const auto write = [&](const char* suffix, const std::vector<u32>& words) {
                    std::ofstream out{std::string{dump} + "/" +
                                          std::filesystem::path(path).stem().string() + suffix,
                                      std::ios::binary};
                    out.write(reinterpret_cast<const char*>(words.data()),
                              words.size() * sizeof(u32));
                };
                write(".recorded.spv", spv);
                write(".replayed.spv", out_spv);
            }
        }
        return Outcome::SpirvDiffers;
    }
    Serialization::Archive archive;
    info.Serialize(archive);
    const auto out_info = archive.TakeOff();
    if (const std::string differs = InfoDifference(info_bytes, out_info); !differs.empty()) {
        if (verbose) {
            std::printf("%s: shader info differs: %s\n", path.c_str(), differs.c_str());
        }
        return Outcome::InfoDiffers;
    }
    if (!(bindings == out_bindings)) {
        return Outcome::BindingsDiffer;
    }
    return Outcome::Same;
}

} // namespace

/// The record being replayed, for the fault report.
const char* current_record = "";

void OnFault(int, siginfo_t* info, void*) {
    char text[512];
    const int n = std::snprintf(text, sizeof(text), "%s: fault reading %p (memory not recorded)\n",
                                current_record, info->si_addr);
    write(2, text, n);
    _exit(3);
}

/// --aux: hashes of the generated auxiliary shaders (rect/quad tessellation, clip distance
/// discard) over a set of inputs, to compare implementations.
static int AuxHashes() {
    using namespace Shader::Backend::SPIRV;
    const auto fnv = [](const std::vector<u32>& code) {
        u64 hash = 0xcbf29ce484222325ULL;
        for (const u32 word : code) {
            hash = (hash ^ word) * 0x100000001b3ULL;
        }
        return hash;
    };
    u64 total = 0;
    u32 cases = 0;
    for (u32 num_inputs = 0; num_inputs <= 5; ++num_inputs) {
        for (u32 pattern = 0; pattern < 4; ++pattern) {
            Shader::HwFragmentRuntimeInfo fs{};
            fs.num_inputs = num_inputs;
            for (u32 i = 0; i < num_inputs; ++i) {
                auto& input = fs.inputs[i];
                input.param_index = u8((i * 3 + pattern) % 32);
                input.is_default = ((i + pattern) % 3) == 1;
                input.is_flat = ((i + pattern) % 4) == 2;
            }
            for (const auto type : {AuxShaderType::RectListTCS, AuxShaderType::QuadListTCS,
                                    AuxShaderType::PassthroughTES}) {
                const u64 hash = fnv(EmitAuxilaryTessShader(type, fs));
                std::printf("aux %u %u %u %016llx\n", num_inputs, pattern, u32(type),
                            (unsigned long long)hash);
                total ^= hash + 0x9e3779b97f4a7c15ULL + (total << 6) + (total >> 2);
                ++cases;
            }
        }
    }
    for (u32 mask = 0; mask < 64; ++mask) {
        std::array<Shader::OutputMap, 3> outputs{};
        u32 bit = 0;
        for (u32 loc = 0; loc < 3; ++loc) {
            for (u32 elem = 0; elem < 4; ++elem, ++bit) {
                const bool clip = bit < 6 && ((mask >> bit) & 1);
                outputs[loc][elem] =
                    clip ? Shader::Output(u32(Shader::Output::ClipDist0) + bit)
                         : (elem == 3 ? Shader::Output::PointSize : Shader::Output::None);
            }
        }
        const u64 hash = fnv(EmitDiscardFragmentShader(outputs));
        std::printf("discard %u %016llx\n", mask, (unsigned long long)hash);
        total ^= hash + 0x9e3779b97f4a7c15ULL + (total << 6) + (total >> 2);
        ++cases;
    }
    std::printf("%u aux shaders, total %016llx\n", cases, (unsigned long long)total);
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--aux") {
        return AuxHashes();
    }
    struct sigaction action{};
    action.sa_sigaction = OnFault;
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, nullptr);
    bool verbose = false;
    u32 repeat = 0;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-v") {
            verbose = true;
        } else if (arg == "--make-variants" && i + 1 < argc) {
            variant_dir = argv[++i];
        } else if (arg == "-r" && i + 1 < argc) {
            repeat = u32(std::atoi(argv[++i]));
        } else if (std::filesystem::is_directory(arg)) {
            for (const auto& entry : std::filesystem::directory_iterator(arg)) {
                if (entry.path().extension() == ".bbsc") {
                    files.push_back(entry.path());
                }
            }
        } else {
            files.push_back(arg);
        }
    }
    std::ranges::sort(files);
    std::array<u32, 6> counts{};
    for (const auto& path : files) {
        current_record = path.c_str();
        ++counts[u32(Replay(path, verbose, repeat))];
    }
    std::printf("%zu compilations: %u same, %u SPIR-V differs, %u info differs, %u bindings "
                "differ, %u unreadable, %u nondeterministic\n",
                files.size(), counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]);
    return counts[0] == files.size() ? 0 : 1;
}
