// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "textures/detiler.h"

#include <array>
#include <bit>

#include "common/assert.h"
#include "common/logging/log.h"
#include "textures/image.h"
#include "textures/image_info.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "render/vk_gpu_profiler.h"
#include "vulkan/shader_module.h"

#include "bbgpu/shaders/detile_macro_128_comp.h"
#include "bbgpu/shaders/detile_macro_16_comp.h"
#include "bbgpu/shaders/detile_macro_32_comp.h"
#include "bbgpu/shaders/detile_macro_64_comp.h"
#include "bbgpu/shaders/detile_macro_8_comp.h"
#include "bbgpu/shaders/detile_macro_96_comp.h"
#include "bbgpu/shaders/detile_micro_128_comp.h"
#include "bbgpu/shaders/detile_micro_16_comp.h"
#include "bbgpu/shaders/detile_micro_32_comp.h"
#include "bbgpu/shaders/detile_micro_64_comp.h"
#include "bbgpu/shaders/detile_micro_8_comp.h"
#include "bbgpu/shaders/detile_micro_96_comp.h"

namespace VideoCore {

namespace {

/// The shader's level table (std430 scalar layout).
struct LevelTableData {
    u32 bank_swizzle;
    u32 micro_tiled_levels;
    u32 num_levels;
    std::array<ImageInfo::MipInfo, 16> levels;
};

std::span<const u32> ShaderCode(u32 bits, bool macro) {
    switch (bits) {
    case 8:
        return macro ? std::span<const u32>{DETILE_MACRO_8_COMP} : DETILE_MICRO_8_COMP;
    case 16:
        return macro ? std::span<const u32>{DETILE_MACRO_16_COMP} : DETILE_MICRO_16_COMP;
    case 32:
        return macro ? std::span<const u32>{DETILE_MACRO_32_COMP} : DETILE_MICRO_32_COMP;
    case 64:
        return macro ? std::span<const u32>{DETILE_MACRO_64_COMP} : DETILE_MICRO_64_COMP;
    case 96:
        return macro ? std::span<const u32>{DETILE_MACRO_96_COMP} : DETILE_MICRO_96_COMP;
    case 128:
        return macro ? std::span<const u32>{DETILE_MACRO_128_COMP} : DETILE_MICRO_128_COMP;
    default:
        UNREACHABLE_MSG("tiling of {}-bit texels", bits);
    }
}

} // namespace

Detiler::Detiler(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                 Vulkan::Runtime& runtime_, StreamBuffer& stream_buffer_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, stream_buffer{stream_buffer_} {
    const auto device = instance.GetDevice();
    const auto binding = [](u32 index, vk::DescriptorType type) {
        return vk::DescriptorSetLayoutBinding{
            .binding = index,
            .descriptorType = type,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        };
    };
    const std::array bindings = {
        binding(0, vk::DescriptorType::eStorageBuffer), // tiled
        binding(1, vk::DescriptorType::eStorageBuffer), // linear
        binding(2, vk::DescriptorType::eUniformBuffer), // level table
    };
    auto layout = device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    });
    ASSERT_MSG(layout.result == vk::Result::eSuccess, "detiler set layout: {}",
               vk::to_string(layout.result));
    set_layout = std::move(layout.value);
    const vk::DescriptorSetLayout set = *set_layout;
    auto pl_layout = device.createPipelineLayoutUnique({.setLayoutCount = 1, .pSetLayouts = &set});
    ASSERT_MSG(pl_layout.result == vk::Result::eSuccess, "detiler pipeline layout: {}",
               vk::to_string(pl_layout.result));
    pipeline_layout = std::move(pl_layout.value);
}

Detiler::~Detiler() = default;

vk::Pipeline Detiler::Pipeline(const ImageInfo& info, bool tiler) {
    const Key key{info.tile_mode, info.num_bits, info.num_samples, tiler};
    if (const auto it = pipelines.find(key); it != pipelines.end()) {
        return *it->second;
    }
    const auto micro_mode = Gcn::GetMicroTileMode(info.tile_mode);
    const bool macro = Gcn::IsMacroTiled(info.array_mode);
    std::array<u32, 12> constants{
        info.num_samples,
        u32(micro_mode),
        Gcn::GetMicroTileThickness(info.array_mode),
        u32(tiler),
    };
    if (macro) {
        const auto macro_mode =
            Gcn::CalculateMacrotileMode(info.tile_mode, info.num_bits, info.num_samples);
        const u32 banks = Gcn::GetNumBanks(macro_mode);
        constants[4] = u32(info.array_mode);
        constants[5] = u32(Gcn::GetPipeConfig(info.tile_mode));
        constants[6] = Gcn::GetBankWidth(macro_mode);
        constants[7] = Gcn::GetBankHeight(macro_mode);
        constants[8] = banks;
        constants[9] = std::bit_width(banks) - 1;
        constants[10] =
            Gcn::CalculateTileSplit(info.tile_mode, info.array_mode, micro_mode, info.num_bits);
        constants[11] = Gcn::GetMacrotileAspect(macro_mode);
    }
    std::array<vk::SpecializationMapEntry, 12> entries;
    for (u32 i = 0; i < entries.size(); ++i) {
        entries[i] = vk::SpecializationMapEntry{i, i * u32(sizeof(u32)), sizeof(u32)};
    }
    const u32 count = macro ? 12 : 4;
    const vk::SpecializationInfo specialization{
        .mapEntryCount = count,
        .pMapEntries = entries.data(),
        .dataSize = count * sizeof(u32),
        .pData = constants.data(),
    };

    const auto device = instance.GetDevice();
    const auto module = Vulkan::CompileSPV(ShaderCode(info.num_bits, macro), device);
    const auto name = fmt::format("{} {}-bit {}", Gcn::NameOf(info.tile_mode), info.num_bits,
                                  tiler ? "tiler" : "detiler");
    LOG_INFO(Render_Vulkan, "Creating tiling pipeline {}", name);
    Vulkan::SetObjectName(device, module, name);
    auto [result, pipeline] = device.createComputePipelineUnique(
        VK_NULL_HANDLE, vk::ComputePipelineCreateInfo{
                            .stage{
                                .stage = vk::ShaderStageFlagBits::eCompute,
                                .module = module,
                                .pName = "main",
                                .pSpecializationInfo = &specialization,
                            },
                            .layout = *pipeline_layout,
                        });
    device.destroyShaderModule(module);
    ASSERT_MSG(result == vk::Result::eSuccess, "tiling pipeline: {}", vk::to_string(result));
    const vk::Pipeline handle = *pipeline;
    pipelines.emplace(key, std::move(pipeline));
    return handle;
}

vk::DescriptorBufferInfo Detiler::LevelTable(const ImageInfo& info, u32 num_levels,
                                             u32 first_level) {
    LevelTableData table{
        .bank_swizzle = info.bank_swizzle,
        .micro_tiled_levels = info.micro_tiled_mips >> first_level,
        .num_levels = num_levels,
    };
    for (u32 level = 0; level < num_levels; ++level) {
        auto& entry = table.levels[level];
        entry = info.mips_layout[level + first_level];
        if (info.props.is_block) { // the shader counts 4x4 blocks
            entry.pitch = std::max((entry.pitch + 3) / 4, 1u);
            entry.height = std::max((entry.height + 3) / 4, 1u);
        }
    }
    return {
        .buffer = stream_buffer.Handle(),
        .offset = stream_buffer.Copy(&table, sizeof(table), instance.UniformMinAlignment()),
        .range = sizeof(table),
    };
}

void Detiler::Dispatch(vk::Pipeline pipeline, const vk::DescriptorBufferInfo& tiled,
                       const vk::DescriptorBufferInfo& linear,
                       const vk::DescriptorBufferInfo& levels, u32 texels) {
    // Recorded with copies of the buffer infos (the recording may run on another thread).
    scheduler.RecordCrumb({.name = "tiling"}, [pipeline, layout = *pipeline_layout, tiled, linear,
                                               levels, groups = texels / 64](vk::CommandBuffer cmd) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
        const auto write = [](u32 binding, vk::DescriptorType type,
                              const vk::DescriptorBufferInfo* info) {
            return vk::WriteDescriptorSet{
                .dstBinding = binding,
                .descriptorCount = 1,
                .descriptorType = type,
                .pBufferInfo = info,
            };
        };
        const std::array writes = {
            write(0, vk::DescriptorType::eStorageBuffer, &tiled),
            write(1, vk::DescriptorType::eStorageBuffer, &linear),
            write(2, vk::DescriptorType::eUniformBuffer, &levels),
        };
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, writes);
        cmd.dispatch(groups, 1, 1);
    });
}

std::pair<const Buffer*, u64> Detiler::DetileImage(const Buffer* in_buffer, u64 in_offset,
                                                   const ImageInfo& info, u32 first_level) {
    if (!info.props.is_tiled) {
        return {in_buffer, in_offset};
    }
    first_level = std::min(first_level, info.resources.levels - 1);
    const u64 skipped = first_level != 0 ? info.mips_layout[first_level].offset : 0;
    const u64 bytes = info.guest_size - skipped;
    const auto levels = LevelTable(info, info.resources.levels - first_level, first_level);
    const auto staging = runtime.GetStagingPool().Request(bytes, MemoryType::DeviceLocal);
    scheduler.EndRendering();
    runtime.FlushBarriers();
    Dispatch(Pipeline(info, false), {in_buffer->Handle(), in_offset, info.guest_size},
             {staging.buffer->Handle(), staging.offset, bytes}, levels,
             u32(bytes / (info.num_bits / 8)));
    // The source may be the arena (pre-uploads), which later uploads write: they wait for this.
    runtime.AccessBuffer(in_buffer, in_offset, info.guest_size,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderRead);
    runtime.AccessBuffer(staging.buffer, staging.offset, bytes,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);
    return {staging.buffer, staging.offset};
}

std::pair<const Buffer*, u64> Detiler::TileImage(Image& image,
                                                 std::span<vk::BufferImageCopy> buffer_copies,
                                                 const Buffer* out_buffer, u64 out_offset,
                                                 u64 vram_size) {
    const auto& info = image.info;
    if (!info.props.is_tiled) {
        for (auto& copy : buffer_copies) {
            copy.bufferOffset += out_offset;
        }
        runtime.DownloadImage(&image, out_buffer, buffer_copies);
        return {nullptr, 0};
    }
    const auto levels = LevelTable(info, u32(buffer_copies.size()));
    const auto staging = runtime.GetStagingPool().Request(info.guest_size, MemoryType::DeviceLocal);
    for (auto& copy : buffer_copies) {
        copy.bufferOffset += staging.offset;
    }
    runtime.DownloadImage(&image, staging.buffer, buffer_copies);
    runtime.FlushBarriers();
    // BB_GPU_PROFILE: the tiler's dispatch gets a mark of its own (without it its time went to
    // the pass or dispatch before: the light shafts' passes looked slow in #204, #210).
    auto* profiler = Vulkan::GpuProfiler::Get();
    const u64 resume = profiler ? profiler->Current() : 0;
    if (profiler) {
        profiler->Mark(0x711E'0000ull ^ info.guest_size, [&] {
            return fmt::format("tile image {}x{} fmt {} ({} bytes)", info.size.width,
                               info.size.height, u32(info.pixel_format), info.guest_size);
        });
    }
    struct ResumeMark {
        Vulkan::GpuProfiler* profiler;
        u64 key;
        ~ResumeMark() {
            if (profiler && key) {
                profiler->Resume(key);
            }
        }
    } resume_mark{profiler, resume};
    const u32 texels = info.guest_size / (info.num_bits / 8);
    // bbport: the tiler writes a texel at a time at swizzled addresses. Into host memory (the
    // game's memory in place, BB_LAYER_MEMORY) each of those writes crossed the bus on its own:
    // ~4.8 ms for a 1280x720 RGBA8 render target that a compute shader reads as a texel buffer,
    // four times a frame at 1440p, on an RTX 5090 (PCIe 5; the dispatches themselves took
    // ~0.1 ms). Tiled into VRAM and copied over in one transfer instead. Only where the tiler
    // writes every byte of the copy: all levels back to back over guest_size and whole
    // workgroups (else bytes it leaves alone would be overwritten).
    const bool host_destination = out_buffer->mem_type == MemoryType::HostCached ||
                                  out_buffer->mem_type == MemoryType::HostUncached;
    u64 covered = 0;
    for (u32 level = 0; level < buffer_copies.size(); ++level) {
        const auto& mip = info.mips_layout[level];
        covered = mip.offset == covered ? covered + mip.size : ~0ULL;
    }
    if (host_destination && covered == info.guest_size && texels % 64 == 0) {
        const auto tiled =
            runtime.GetStagingPool().Request(std::max<u64>(info.guest_size, vram_size),
                                             MemoryType::DeviceLocal);
        Dispatch(Pipeline(info, true), {tiled.buffer->Handle(), tiled.offset, info.guest_size},
                 {staging.buffer->Handle(), staging.offset, info.guest_size}, levels, texels);
        runtime.AccessBuffer(tiled.buffer, tiled.offset, info.guest_size,
                             vk::PipelineStageFlagBits2::eComputeShader,
                             vk::AccessFlagBits2::eShaderWrite);
        const vk::BufferCopy copy{
            .srcOffset = tiled.offset, .dstOffset = out_offset, .size = info.guest_size};
        runtime.CopyBuffer(tiled.buffer, out_buffer, {&copy, 1});
        return {tiled.buffer, tiled.offset};
    }
    Dispatch(Pipeline(info, true), {out_buffer->Handle(), out_offset, info.guest_size},
             {staging.buffer->Handle(), staging.offset, info.guest_size}, levels, texels);
    runtime.AccessBuffer(out_buffer, out_offset, info.guest_size,
                         vk::PipelineStageFlagBits2::eComputeShader,
                         vk::AccessFlagBits2::eShaderWrite);
    return {nullptr, 0};
}

} // namespace VideoCore
