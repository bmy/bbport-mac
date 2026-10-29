// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "buffers/fault_buffer.h"

#include <cstring>
#include <limits>

#include "common/div_ceil.h"
#include "common/logging/log.h"
#include "buffers/buffer_cache.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "render/scheduler.h"
#include "vulkan/shader_module.h"

#include "bbgpu/shaders/fault_scan_comp.h"

namespace VideoCore {

namespace {

constexpr u32 MaxFaults = 1024;
constexpr u64 AreaSize = MaxFaults * sizeof(u64);

} // namespace

FaultBuffer::FaultBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler_,
                         BufferCache& buffer_cache_, u32 page_bits, u64 num_pages_)
    : scheduler{scheduler_}, buffer_cache{buffer_cache_}, page_size{1ull << page_bits},
      num_pages{num_pages_}, bits_size{num_pages_ / 8},
      bits{instance, 0, bits_size, MemoryType::DeviceLocal},
      readback{instance, 0, MaxPendingReads * AreaSize, MemoryType::HostCached} {
    const auto device = instance.GetDevice();
    Vulkan::SetObjectName(device, bits.Handle(), "Fault Buffer");
    const auto binding = [](u32 index) {
        return vk::DescriptorSetLayoutBinding{
            .binding = index,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        };
    };
    const std::array bindings = {binding(0), binding(1)};
    set_layout = Vulkan::Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::DescriptorSetLayout set = *set_layout;
    pipeline_layout =
        Vulkan::Check(device.createPipelineLayoutUnique({.setLayoutCount = 1, .pSetLayouts = &set}));

    const std::array<u32, 2> constants{page_bits, MaxFaults};
    const std::array<vk::SpecializationMapEntry, 2> entries{{{0, 0, 4}, {1, 4, 4}}};
    const vk::SpecializationInfo specialization{
        .mapEntryCount = u32(entries.size()),
        .pMapEntries = entries.data(),
        .dataSize = sizeof(constants),
        .pData = constants.data(),
    };
    const auto module = Vulkan::CompileSPV(FAULT_SCAN_COMP, device);
    Vulkan::SetObjectName(device, module, "Fault Buffer Parser");
    pipeline = Vulkan::Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage{
                    .stage = vk::ShaderStageFlagBits::eCompute,
                    .module = module,
                    .pName = "main",
                    .pSpecializationInfo = &specialization,
                },
                .layout = *pipeline_layout,
            }));
    Vulkan::SetObjectName(device, *pipeline, "Fault Buffer Parser Pipeline");
    device.destroyShaderModule(module);
}

void FaultBuffer::ProcessFaultBuffer() {
    // The area is free once the GPU is done with its last scan and the results were read.
    if (const u64 tick = area_ticks[next_area]) {
        scheduler.Wait(tick);
        scheduler.PopPendingOperations();
    }
    const u32 area = next_area;
    const u64 offset = area * AreaSize;
    u8* results = readback.mapped_data.data() + offset;
    std::memset(results, 0, AreaSize);

    const auto barrier = [&](vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                             vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
        return vk::BufferMemoryBarrier2{
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .buffer = bits.Handle(),
            .offset = 0,
            .size = bits_size,
        };
    };
    const auto before = barrier(vk::PipelineStageFlagBits2::eAllCommands,
                                vk::AccessFlagBits2::eShaderWrite,
                                vk::PipelineStageFlagBits2::eComputeShader,
                                vk::AccessFlagBits2::eShaderRead);
    const auto after = barrier(vk::PipelineStageFlagBits2::eComputeShader,
                               vk::AccessFlagBits2::eShaderWrite,
                               vk::PipelineStageFlagBits2::eAllCommands,
                               vk::AccessFlagBits2::eShaderWrite);
    const vk::DescriptorBufferInfo bits_info{bits.Handle(), 0, bits_size};
    const vk::DescriptorBufferInfo results_info{readback.Handle(), offset, AreaSize};
    const std::array writes = {
        vk::WriteDescriptorSet{
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &bits_info,
        },
        vk::WriteDescriptorSet{
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &results_info,
        },
    };
    scheduler.EndRendering();
    const auto cmd = scheduler.CommandBuffer();
    const Vulkan::Breadcrumbs::Scope crumb{cmd, scheduler.CrumbStream(), "fault buffer"};
    cmd.pipelineBarrier2({
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &before,
    });
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, writes);
    // A word (32 pages) per invocation, 64 per workgroup.
    cmd.dispatch(Common::DivCeil(u32(num_pages / 32), 64u), 1, 1);
    cmd.pipelineBarrier2({
        .dependencyFlags = vk::DependencyFlagBits::eByRegion,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers = &after,
    });

    scheduler.DeferOperation([this, results, area] {
        faulted.Clear();
        const auto* entries = reinterpret_cast<const u64*>(results);
        const u32 count = u32(entries[0]);
        for (u32 i = 1; i <= count && i < MaxFaults; ++i) {
            faulted.Add(entries[i], page_size);
            LOG_INFO(Render_Vulkan, "Accessed non-GPU cached memory at {:#x}", entries[i]);
        }
        faulted.ForEach([&](VAddr start, VAddr end) {
            ASSERT_MSG(end - start <= std::numeric_limits<u32>::max(), "fault range too large");
            (void)buffer_cache.ObtainBuffer(start, u32(end - start), false);
        });
        area_ticks[area] = 0;
    });
    area_ticks[next_area] = scheduler.CurrentTick();
    next_area = (next_area + 1) % MaxPendingReads;
}

} // namespace VideoCore
