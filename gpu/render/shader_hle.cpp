// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The game's buffer copy compute shader done as buffer copies: its control buffer lists
// (destination, source, count) records of one stride; copies close together are grouped so
// that each group needs one source and one destination buffer, and each group is one dispatch
// of buffer_multi_copy.comp (or vkCmdCopyBuffer for a few copies).
#include <algorithm>
#include "shim/bb_env.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include <xxhash.h>

#include "bbgpu/shaders/buffer_multi_copy_comp.h"
#include "bbport_toggles.h"
#include "game_profile.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/shader_hle.h"
#include "render/vk_gpu_profiler.h"
#include "shader/info.h"
#include "render/rasterizer.h"
#include "vulkan/instance.h"
#include "vulkan/shader_module.h"

extern std::unique_ptr<BbCp::CommandProcessor> liverpool;

namespace Vulkan {

// The copy shader runs ~57 times per frame with ~1024 small ranges each. As one
// vkCmdCopyBuffer with that many regions it cost ~37 ns of GPU time per range (~2.1 ms per
// frame) plus the recording; buffer_multi_copy.comp copies all ranges in one dispatch.
static bool MultiCopy(Rasterizer& rasterizer, const VideoCore::Buffer* src,
                      const VideoCore::Buffer* dst, std::span<const vk::BufferCopy> copies) {
    if (copies.size() < 8 || BbToggle::Disabled(BbToggle::MultiCopyShader)) {
        return false;
    }
    u64 src_min = ~0ull, src_max = 0, dst_min = ~0ull, dst_max = 0;
    for (const auto& copy : copies) {
        if ((copy.srcOffset | copy.dstOffset | copy.size) & 3) {
            return false; // dword copies only
        }
        src_min = std::min(src_min, copy.srcOffset);
        src_max = std::max(src_max, copy.srcOffset + copy.size);
        dst_min = std::min(dst_min, copy.dstOffset);
        dst_max = std::max(dst_max, copy.dstOffset + copy.size);
    }
    auto& runtime = rasterizer.GetRuntime();
    auto& scheduler = runtime.GetScheduler();
    const auto& instance = runtime.GetInstance();
    const u64 align = instance.StorageMinAlignment();
    src_min = Common::AlignDown(src_min, align);
    dst_min = Common::AlignDown(dst_min, align);

    struct Pipeline {
        vk::UniqueDescriptorSetLayout set_layout;
        vk::UniquePipelineLayout layout;
        vk::UniquePipeline pipeline;
    };
    static Pipeline pipe = [&] {
        const auto device = instance.GetDevice();
        Pipeline p;
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
        for (u32 i = 0; i < 3; ++i) {
            bindings[i] = {.binding = i,
                           .descriptorType = vk::DescriptorType::eStorageBuffer,
                           .descriptorCount = 1,
                           .stageFlags = vk::ShaderStageFlagBits::eCompute};
        }
        p.set_layout = Check(device.createDescriptorSetLayoutUnique({
            .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
            .bindingCount = static_cast<u32>(bindings.size()),
            .pBindings = bindings.data(),
        }));
        const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                          .offset = 0,
                                          .size = sizeof(u32)};
        p.layout = Check(device.createPipelineLayoutUnique({
            .setLayoutCount = 1,
            .pSetLayouts = &*p.set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        }));
        const auto module = CompileSPV(BUFFER_MULTI_COPY_COMP, device);
        p.pipeline = Check(device.createComputePipelineUnique(
            {}, vk::ComputePipelineCreateInfo{
                    .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                              .module = module,
                              .pName = "main"},
                    .layout = *p.layout,
                }));
        device.destroyShaderModule(module);
        return p;
    }();

    // The region table goes into the stream buffer (host writes, visible at submission).
    thread_local std::vector<u32> regions;
    regions.resize(copies.size() * 4);
    for (size_t i = 0; i < copies.size(); ++i) {
        regions[i * 4 + 0] = static_cast<u32>((copies[i].srcOffset - src_min) / 4);
        regions[i * 4 + 1] = static_cast<u32>((copies[i].dstOffset - dst_min) / 4);
        regions[i * 4 + 2] = static_cast<u32>(copies[i].size / 4);
        regions[i * 4 + 3] = 0;
    }
    auto& stream = rasterizer.GetBufferCache().GetStreamBuffer();
    const u64 table_size = regions.size() * sizeof(u32);
    const u64 table_offset = stream.Copy(regions.data(), table_size, align);

    if (auto* profiler = GpuProfiler::Get()) {
        profiler->Mark(0xC0B1ull, [] { return std::string{"copy shader HLE: barrier + dispatch"}; });
    }
    scheduler.EndRendering();
    const u64 src_size = src_max - src_min, dst_size = dst_max - dst_min;
    if (runtime.IsBufferAccessed(src, src_min, src_size) ||
        runtime.IsBufferAccessed(dst, dst_min, dst_size, true)) {
        runtime.FlushBarriers();
    }
    const u32 count = static_cast<u32>(copies.size());
    scheduler.RecordCrumb({.name = "HLE copy shader"}, [src = src->Handle(), dst = dst->Handle(), table = stream.Handle(), src_min,
                      src_size, dst_min, dst_size, table_offset, table_size,
                      count](vk::CommandBuffer cmdbuf) {
        const std::array<vk::DescriptorBufferInfo, 3> infos{{
            {src, src_min, src_size},
            {dst, dst_min, dst_size},
            {table, table_offset, table_size},
        }};
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (u32 i = 0; i < 3; ++i) {
            writes[i] = {.dstBinding = i,
                         .descriptorCount = 1,
                         .descriptorType = vk::DescriptorType::eStorageBuffer,
                         .pBufferInfo = &infos[i]};
        }
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *pipe.pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pipe.layout, 0, writes);
        cmdbuf.pushConstants(*pipe.layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(count),
                             &count);
        cmdbuf.dispatch(count, 1, 1);
    });
    runtime.AccessBuffer(src, src_min, src_size, vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead);
    runtime.AccessBuffer(dst, dst_min, dst_size, vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);
    return true;
}

/// The copy shader's three formatted buffers: control records, source and destination.
static bool ExecuteCopyShaderHLE(const Shader::Info& info, const Gcn::ComputeProgram& cs_program,
                                 Rasterizer& rasterizer) {
    auto& runtime = rasterizer.GetRuntime();
    auto& buffer_cache = rasterizer.GetBufferCache();

    const auto ctl_buf_sharp = info.buffers[0].GetSharp(info);
    const auto src_buf_sharp = info.buffers[1].GetSharp(info);
    const auto dst_buf_sharp = info.buffers[2].GetSharp(info);
    const auto buf_stride = src_buf_sharp.GetStride();
    ASSERT(buf_stride == dst_buf_sharp.GetStride());

    struct CopyShaderControl {
        u32 dst_idx;
        u32 src_idx;
        u32 end;
    };
    static_assert(sizeof(CopyShaderControl) == 12);
    ASSERT(ctl_buf_sharp.GetStride() == sizeof(CopyShaderControl));
    const auto ctl_buf = reinterpret_cast<const CopyShaderControl*>(ctl_buf_sharp.base_address);

    static std::vector<vk::BufferCopy> copies;
    copies.clear();
    copies.reserve(cs_program.dim_x);
    // BB_COPY_SHADER_TRACE=1 (diagnostics): the copy shader's buffers, every 2 s.
    static const bool trace = bb_env("BB_COPY_SHADER_TRACE") != nullptr;
    if (trace) {
        static auto last = std::chrono::steady_clock::now();
        static u32 calls = 0;
        ++calls;
        if (std::chrono::steady_clock::now() - last > std::chrono::seconds(2)) {
            last = std::chrono::steady_clock::now();
            u64 bytes = 0, lo = ~0ull, hi = 0;
            for (u32 i = 0; i < cs_program.dim_x; ++i) {
                bytes += u64(ctl_buf[i].end + 1) * buf_stride;
                lo = std::min<u64>(lo, u64(ctl_buf[i].dst_idx) * buf_stride);
                hi = std::max<u64>(hi, u64(ctl_buf[i].dst_idx + ctl_buf[i].end + 1) * buf_stride);
            }
            std::printf("Copy shader: %u calls; this one %u ranges %llu bytes; ctl %#llx size %u; "
                        "src %#llx size %u stride %u fmt %u/%u; dst %#llx size %u stride %u fmt "
                        "%u/%u; dst touched %#llx..%#llx; threads %u x %u, user data %u regs\n",
                        calls, cs_program.dim_x, (unsigned long long)bytes,
                        (unsigned long long)ctl_buf_sharp.base_address, u32(ctl_buf_sharp.GetSize()),
                        (unsigned long long)src_buf_sharp.base_address, u32(src_buf_sharp.GetSize()),
                        u32(buf_stride), u32(src_buf_sharp.GetDataFmt()),
                        u32(src_buf_sharp.GetNumberFmt()),
                        (unsigned long long)dst_buf_sharp.base_address, u32(dst_buf_sharp.GetSize()),
                        u32(dst_buf_sharp.GetStride()), u32(dst_buf_sharp.GetDataFmt()),
                        u32(dst_buf_sharp.GetNumberFmt()), (unsigned long long)lo,
                        (unsigned long long)hi, cs_program.num_thread_x.full,
                        cs_program.num_thread_y.full, u32(info.UserData().size()));
            calls = 0;
        }
    }

    for (u32 i = 0; i < cs_program.dim_x; i++) {
        const auto& [dst_idx, src_idx, end] = ctl_buf[i];
        const u32 local_dst_offset = dst_idx * buf_stride;
        const u32 local_src_offset = src_idx * buf_stride;
        const u32 local_size = (end + 1) * buf_stride;
        copies.emplace_back(local_src_offset, local_dst_offset, local_size);
    }

    // Grouping distance: 64 KiB. The copies are a few KiB spread over up to 57 MiB, and
    // each batch synchronizes (uploads, marks GPU-modified) its whole range: GPU time of the copy
    // shader 1.5 -> 0.6 ms/frame, GPU busy 85% -> 77%, frame rate no lower.
    // BB_COPY_MERGE_KB overrides it.
    static const vk::DeviceSize MaxDistanceForMerge = [] {
        const char* env = bb_env("BB_COPY_MERGE_KB");
        return env ? vk::DeviceSize(std::strtoull(env, nullptr, 10)) * 1024 : vk::DeviceSize(64_KB);
    }();
    // Groups: copies are added while the group's source and destination spans stay within the
    // merge distance.
    for (u32 batch_start = 0, batch_end = 0; batch_end < copies.size(); batch_start = batch_end) {
        const auto& first = copies[batch_start];
        auto src_min = first.srcOffset;
        auto src_max = first.srcOffset + first.size;
        auto dst_min = first.dstOffset;
        auto dst_max = first.dstOffset + first.size;
        for (++batch_end; batch_end < copies.size(); batch_end++) {
            const auto& [src_offset, dst_offset, size] = copies[batch_end];
            const auto new_src_min = std::min(src_min, src_offset);
            const auto new_src_max = std::max(src_max, src_offset + size);
            const auto new_dst_min = std::min(dst_min, dst_offset);
            const auto new_dst_max = std::max(dst_max, dst_offset + size);
            if (new_src_max - new_src_min > MaxDistanceForMerge ||
                new_dst_max - new_dst_min > MaxDistanceForMerge) {
                break;
            }
            src_min = new_src_min;
            src_max = new_src_max;
            dst_min = new_dst_min;
            dst_max = new_dst_max;
        }
        auto [src_buf, src_buf_offset] = buffer_cache.ObtainBuffer(
            src_buf_sharp.base_address + src_min, src_max - src_min, false);
        // BB_EXP_COPY_SRC_SNAPSHOT=1 (experiment): the source bytes as they are now, in the stream
        // buffer (what the old memory model did), not read in place when the GPU gets there.
        static const bool snapshot = bb_env("BB_EXP_COPY_SRC_SNAPSHOT") != nullptr;
        if (snapshot && src_max - src_min <= 256_KB &&
            !buffer_cache.IsRegionGpuModified(src_buf_sharp.base_address + src_min, src_max - src_min)) {
            auto& stream = buffer_cache.GetStreamBuffer();
            src_buf_offset = stream.Copy(
                reinterpret_cast<const void*>(src_buf_sharp.base_address + src_min), src_max - src_min,
                runtime.GetInstance().StorageMinAlignment());
            src_buf = &stream;
        }
        const auto [dst_buf, dst_buf_offset] = buffer_cache.ObtainBuffer(
            dst_buf_sharp.base_address + dst_min, dst_max - dst_min, true);
        const auto group = std::span{copies}.subspan(batch_start, batch_end - batch_start);
        for (auto& copy : group) {
            copy.srcOffset = copy.srcOffset - src_min + src_buf_offset;
            copy.dstOffset = copy.dstOffset - dst_min + dst_buf_offset;
        }
        LOG_TRACE(Render_Vulkan, "HLE buffer copy: src_size = {}, dst_size = {}", src_max - src_min,
                  dst_max - dst_min);
        if (!MultiCopy(rasterizer, src_buf, dst_buf, group)) {
            runtime.CopyBuffer(src_buf, dst_buf, group);
        }
    }

    return true;
}

bool ExecuteShaderHLE(const Shader::Info& info, const Gcn::Regs& regs,
                      const Gcn::ComputeProgram& cs_program, Rasterizer& rasterizer) {
    // The game's buffer copy shader, from its profile (games/); nothing else is replaced.
    const u64 copy_shader = Game::BufferCopyShader();
    if (copy_shader != 0 && info.pgm_hash == copy_shader) {
        // With the game's memory in place the game's shader runs (translated), reading its
        // copy list when the GPU executes it. BB_COPY_SHADER_NATIVE=0: the list read on the CPU
        // when the dispatch is recorded, copies of our own (before). Experiment bit 2 inverts it
        // while the game runs. The layer's paged stores now write through to guest memory while
        // keeping the current mirrors in sync, so the original shader is the default here too.
        static const bool native_env = [copy_shader] {
            const char* env = bb_env("BB_COPY_SHADER_NATIVE");
            const bool native = !env || env[0] != '0';
            const bool translated = native != BbToggle::Experiment(2);
            std::printf("GPU: buffer copy shader %llx: %s%s (BB_COPY_SHADER_NATIVE=0: HLE)\n",
                        (unsigned long long)copy_shader,
                        translated ? "original translated shader" : "HLE",
                        translated && VideoCore::BufferCache::LayerPagedActive()
                            ? ", paged writes through to guest memory" : "");
            return native;
        }();
        if (native_env != BbToggle::Experiment(2)) {
            return false;
        }
        // Its destinations stay in the game's memory, as when it runs as itself (see
        // Rasterizer::DispatchRecord): our translator reads what it copies there on the CPU.
        auto& buffer_cache = rasterizer.GetBufferCache();
        buffer_cache.force_writes_in_place = true;
        const bool done = ExecuteCopyShaderHLE(info, cs_program, rasterizer);
        buffer_cache.force_writes_in_place = false;
        return done;
    }
    return false;
}

} // namespace Vulkan

namespace Vulkan {

namespace {
/// Merges out[first..] ([begin, end) pairs) where one begins within `gap` of the end of what
/// precedes it in address order. The copy shader's entries come in a few ascending runs (~170
/// entries in ~6 runs): the runs are merged rather than the whole sorted (the sort was 4-6 % of the
/// recording thread).
void MergeRanges(std::vector<std::pair<unsigned long long, unsigned long long>>& out, size_t first,
                 u64 gap) {
    using Range = std::pair<unsigned long long, unsigned long long>;
    if (out.size() - first <= 1) {
        return;
    }
    thread_local std::vector<size_t> runs;
    runs.clear();
    runs.push_back(first);
    for (size_t i = first + 1; i < out.size(); ++i) {
        if (out[i].first < out[i - 1].first) {
            runs.push_back(i);
        }
    }
    runs.push_back(out.size());
    if (runs.size() > 34) {
        std::sort(out.begin() + ptrdiff_t(first), out.end());
    } else if (runs.size() > 2) {
        // Pairwise merges of neighbouring runs through a scratch copy until one run is left.
        thread_local std::vector<Range> scratch;
        thread_local std::vector<size_t> next;
        while (runs.size() > 2) {
            scratch.resize(out.size() - first);
            next.clear();
            for (size_t r = 0; r + 1 < runs.size(); r += 2) {
                const size_t lo = runs[r], mid = runs[r + 1];
                const size_t hi = r + 2 < runs.size() ? runs[r + 2] : mid;
                std::merge(out.begin() + ptrdiff_t(lo), out.begin() + ptrdiff_t(mid),
                           out.begin() + ptrdiff_t(mid), out.begin() + ptrdiff_t(hi),
                           scratch.begin() + ptrdiff_t(lo - first));
                next.push_back(lo);
            }
            next.push_back(out.size());
            std::copy(scratch.begin(), scratch.end(), out.begin() + ptrdiff_t(first));
            runs.swap(next);
        }
    }
    size_t merged = first;
    for (size_t i = first + 1; i < out.size(); ++i) {
        if (out[i].first <= out[merged].second + gap) {
            out[merged].second = std::max(out[merged].second, out[i].second);
        } else {
            out[++merged] = out[i];
        }
    }
    out.resize(merged + 1);
}

/// The copy shader's control buffer: one entry per workgroup.
struct CopyControl {
    u32 dst_idx;
    u32 src_idx;
    u32 end;
};

/// Appends the merged [begin, end) ranges the control entries name in one buffer (destinations or
/// sources). The game reissues the same copies frame after frame (~130 calls a frame of ~160
/// entries): the merge is kept by a hash of the entries and what turns them into addresses.
void BuildRanges(std::vector<std::pair<unsigned long long, unsigned long long>>& out,
                 const CopyControl* ctl, u32 groups, bool destinations, u64 base, u64 stride, u64 gap) {
    struct Key {
        u64 hash, base, stride, gap;
        u32 groups, destinations;
        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& key) const noexcept {
            return key.hash ^ (key.base * 0x9E3779B97F4A7C15ull) ^ (key.stride << 7) ^ key.groups ^
                   (u64(key.destinations) << 63);
        }
    };
    thread_local std::unordered_map<Key, std::vector<std::pair<unsigned long long, unsigned long long>>,
                                    KeyHash> memo;
    const Key key{XXH3_64bits(ctl, size_t(groups) * sizeof(CopyControl)), base, stride, gap, groups,
                  destinations};
    if (const auto it = memo.find(key); it != memo.end()) {
        out.insert(out.end(), it->second.begin(), it->second.end());
        return;
    }
    const size_t first = out.size();
    for (u32 i = 0; i < groups; ++i) {
        const u64 begin = base + u64(destinations ? ctl[i].dst_idx : ctl[i].src_idx) * stride;
        out.emplace_back(begin, begin + (u64(ctl[i].end) + 1) * stride);
    }
    MergeRanges(out, first, gap);
    if (memo.size() >= 4096) {
        memo.clear();
    }
    memo.emplace(key, std::vector<std::pair<unsigned long long, unsigned long long>>(
                          out.begin() + ptrdiff_t(first), out.end()));
}
} // namespace

bool CopyShaderRanges(const Shader::Info& info, u32 groups, int which, u64 gap, u64& base,
                      std::vector<std::pair<unsigned long long, unsigned long long>>& out) {
    const u64 copy_shader = Game::BufferCopyShader();
    if (copy_shader == 0 || info.pgm_hash != copy_shader || info.buffers.size() < 3 || groups == 0 ||
        (which != 1 && which != 2)) {
        return false;
    }
    const auto ctl_sharp = info.buffers[0].GetSharp(info);
    const auto sharp = info.buffers[which].GetSharp(info);
    const u64 stride = sharp.GetStride();
    if (ctl_sharp.GetStride() != 12 || stride == 0 || ctl_sharp.base_address == 0 ||
        u64(groups) * 12 > ctl_sharp.GetSize()) {
        return false;
    }
    const auto* ctl = reinterpret_cast<const CopyControl*>(ctl_sharp.base_address);
    base = sharp.base_address;
    const size_t first = out.size();
    BuildRanges(out, ctl, groups, which == 2, sharp.base_address, stride, gap);
    for (size_t i = first; i < out.size(); ++i) {
        out[i].second -= out[i].first;
    }
    return true;
}

bool CopyShaderEntries(const Shader::Info& info, u32 groups, std::vector<CopyEntry>& out) {
    const u64 copy_shader = Game::BufferCopyShader();
    if (copy_shader == 0 || info.pgm_hash != copy_shader || info.buffers.size() < 3 || groups == 0) {
        return false;
    }
    const auto ctl_sharp = info.buffers[0].GetSharp(info);
    const auto src_sharp = info.buffers[1].GetSharp(info);
    const auto dst_sharp = info.buffers[2].GetSharp(info);
    const u64 src_stride = src_sharp.GetStride(), dst_stride = dst_sharp.GetStride();
    if (ctl_sharp.GetStride() != 12 || src_stride == 0 || src_stride != dst_stride ||
        ctl_sharp.base_address == 0 || u64(groups) * 12 > ctl_sharp.GetSize()) {
        return false;
    }
    const auto* ctl = reinterpret_cast<const CopyControl*>(ctl_sharp.base_address);
    for (u32 i = 0; i < groups; ++i) {
        out.push_back({src_sharp.base_address + u64(ctl[i].src_idx) * src_stride,
                       dst_sharp.base_address + u64(ctl[i].dst_idx) * dst_stride,
                       (u64(ctl[i].end) + 1) * dst_stride,
                       ctl_sharp.base_address + u64(i) * sizeof(CopyControl)});
    }
    return true;
}

bool CopyShaderDestinations(const Shader::Info& info, u32 groups, u64 gap,
                            std::vector<std::pair<unsigned long long, unsigned long long>>& out) {
    const u64 copy_shader = Game::BufferCopyShader();
    if (copy_shader == 0 || info.pgm_hash != copy_shader || info.buffers.size() < 3 || groups == 0) {
        return false;
    }
    const auto ctl_sharp = info.buffers[0].GetSharp(info);
    const auto dst_sharp = info.buffers[2].GetSharp(info);
    const u64 stride = dst_sharp.GetStride();
    if (ctl_sharp.GetStride() != 12 || stride == 0 || ctl_sharp.base_address == 0 ||
        u64(groups) * 12 > ctl_sharp.GetSize()) {
        return false;
    }
    const auto* ctl = reinterpret_cast<const CopyControl*>(ctl_sharp.base_address);
    const size_t first = out.size();
    BuildRanges(out, ctl, groups, true, dst_sharp.base_address, stride, gap);
    // As sizes from here on.
    for (size_t i = first; i < out.size(); ++i) {
        out[i].second -= out[i].first;
    }
    return true;
}

} // namespace Vulkan

namespace Vulkan {

bool CopyDestinationsPrecise() {
    static const bool on = !bb_env_off("BB_COPY_DEST_PRECISE");
    return on;
}

unsigned long long CopyDestinationGap() {
    static const unsigned long long gap = [] {
        const char* env = bb_env("BB_COPY_DEST_GAP");
        return env ? std::strtoull(env, nullptr, 0) : 256ull;
    }();
    return gap;
}

} // namespace Vulkan
