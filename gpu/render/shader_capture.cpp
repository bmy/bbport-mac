// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/shader_capture.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common/serdes.h"
#include "shader/backend/bindings.h"
#include "gcn/decoder.h"
#include "shader/info.h"
#include "shader/profile.h"
#include "shader/runtime_info.h"

namespace BbShaderCapture {

namespace {

/// Guest memory read while compiling: a size of 0 is a program (its length found by decoding).
struct Read {
    const u8* address;
    size_t size;
};
thread_local std::vector<Read> reads;

void RecordRead(const void* address, size_t size) {
    reads.push_back({static_cast<const u8*>(address), size});
}

const char* Directory() {
    static const char* dir = std::getenv("BB_SHADER_CAPTURE");
    return dir && *dir ? dir : nullptr;
}

/// Bytes of a program at `code` up to and including its s_setpc_b64 (fetch shaders return so).
size_t ProgramBytes(const u8* code) {
    const auto* start = reinterpret_cast<const u32*>(code);
    const u32* pc = start;
    for (u32 i = 0; i < 1024; ++i) {
        if (GcnIsa::Decode(pc).opcode == GcnIsa::Opcode::S_SETPC_B64) {
            break;
        }
    }
    return size_t(pc - start) * sizeof(u32);
}

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

} // namespace

bool Enabled() {
    return Directory() != nullptr;
}

void Begin() {
    reads.clear();
    Shader::guest_read_hook = RecordRead;
}

void Write(const Shader::Info& in_info, const Shader::RuntimeInfo& runtime_info,
           const Shader::Profile& profile, const Shader::Backend::Bindings& in_bindings,
           std::span<const u32> code, size_t perm_idx, std::span<const u32> spv,
           const Shader::Info& out_info, const Shader::Backend::Bindings& out_bindings) {
    Shader::guest_read_hook = nullptr;
    Out out;
    out.Raw("BBSC", 4);
    out.Value(u32{5}); // format version (5: RuntimeInfo tcs max_tess_level, Profile tess flags)
    // Globals the backend reads (object motion vectors).
    out.Value(Shader::MotionVectors::params_address);
    out.Value(Shader::MotionVectors::positions_address);
    out.Value(u32(in_info.hw_stage));
    out.Value(u32(in_info.sw_stage));
    out.Value(in_info.pgm_hash);
    out.Value(u64(perm_idx));
    out.Value(u64(in_info.pgm_base));
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    std::copy_n(in_info.user_data.begin(),
                std::min(in_info.user_data.size(), user_data.size()), user_data.begin());
    out.Value(user_data);
    out.Words(code);
    out.Value(runtime_info);
    out.Words(runtime_info.hw_stage == Shader::HwStage::Geometry ? runtime_info.hw.gs.vs_copy
                                                                 : std::span<const u32>{});
    out.Value(profile);
    out.Value(in_bindings);

    // The memory read, merged into ranges.
    std::vector<std::pair<const u8*, size_t>> ranges;
    for (const auto& read : reads) {
        ranges.emplace_back(read.address, read.size ? read.size : ProgramBytes(read.address));
    }
    std::ranges::sort(ranges);
    std::vector<std::pair<const u8*, size_t>> merged;
    for (const auto& [address, size] : ranges) {
        if (!merged.empty() && address <= merged.back().first + merged.back().second) {
            auto& last = merged.back();
            last.second = std::max<size_t>(last.second, size_t(address + size - last.first));
        } else {
            merged.emplace_back(address, size);
        }
    }
    out.Value(u32(merged.size()));
    for (const auto& [address, size] : merged) {
        out.Value(u64(reinterpret_cast<uintptr_t>(address)));
        out.Value(u32(size));
        out.Raw(address, size);
    }

    out.Words(spv);
    Serialization::Archive archive;
    out_info.Serialize(archive);
    const auto info_bytes = archive.TakeOff();
    out.Value(u64(info_bytes.size()));
    out.Raw(info_bytes.data(), info_bytes.size());
    out.Value(out_bindings);

    static std::atomic<u32> sequence{0};
    char name[512];
    std::snprintf(name, sizeof(name), "%s/%016llx_%u_%zu_%u.bbsc", Directory(),
                  (unsigned long long)in_info.pgm_hash, u32(in_info.hw_stage), perm_idx,
                  sequence.fetch_add(1));
    if (FILE* file = std::fopen(name, "wb")) {
        std::fwrite(out.bytes.data(), 1, out.bytes.size(), file);
        std::fclose(file);
    }
}

} // namespace BbShaderCapture
