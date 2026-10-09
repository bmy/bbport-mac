// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/renderer_vulkan/vk_occlusion.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "video_core/buffer_cache/buffer.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/host_shaders/occlusion_resolve_comp.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {
struct Push {
    u32 count;
    u32 events;
};

struct EventEntry {
    u32 address_lo;
    u32 address_hi;
    u32 pairs;
    u32 upto;
};

/// Everything before it before everything after it (resolves are a few per frame).
void FullBarrier(vk::CommandBuffer cmdbuf) {
    const vk::MemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &barrier,
    });
}
} // namespace

OcclusionQueries::OcclusionQueries(const Instance& instance_, Scheduler& scheduler_,
                                   VideoCore::BufferCache& buffer_cache_)
    : instance{instance_}, scheduler{scheduler_}, buffer_cache{buffer_cache_} {
    const auto device = instance.GetDevice();
    pool = Check(device.createQueryPoolUnique({
        .queryType = vk::QueryType::eOcclusion,
        .queryCount = NumSlots,
    }));
    results = std::make_unique<VideoCore::Buffer>(instance, 0, NumSlots * sizeof(u64),
                                                  VideoCore::MemoryType::DeviceLocal,
                                                  "Occlusion results");
    total = std::make_unique<VideoCore::Buffer>(instance, 0, 256, VideoCore::MemoryType::DeviceLocal,
                                                "Occlusion total");
    std::array<vk::DescriptorSetLayoutBinding, 3> bindings{};
    for (u32 i = 0; i < bindings.size(); ++i) {
        bindings[i] = {.binding = i,
                       .descriptorType = vk::DescriptorType::eStorageBuffer,
                       .descriptorCount = 1,
                       .stageFlags = vk::ShaderStageFlagBits::eCompute};
    }
    set_layout = Check(device.createDescriptorSetLayoutUnique({
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    }));
    const vk::PushConstantRange range{.stageFlags = vk::ShaderStageFlagBits::eCompute,
                                      .offset = 0,
                                      .size = sizeof(Push)};
    pipeline_layout = Check(device.createPipelineLayoutUnique({
        .setLayoutCount = 1,
        .pSetLayouts = &*set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &range,
    }));
    const auto module = CompileSPV(OCCLUSION_RESOLVE_COMP, device);
    pipeline = Check(device.createComputePipelineUnique(
        {}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = module,
                          .pName = "main"},
                .layout = *pipeline_layout,
            }));
    device.destroyShaderModule(module);
    std::printf("GPU: the game's occlusion queries are Vulkan occlusion queries (%s counts)\n",
                instance.IsOcclusionQueryPrecise() ? "exact" : "approximate");
}

OcclusionQueries::~OcclusionQueries() {
    scheduler.SetCarriedScope(nullptr);
}

void OcclusionQueries::BeginSegment(bool inside_render_pass) {
    if (begun == reset_until) {
        if (inside_render_pass) {
            ++starved; // no reset can be recorded in a render pass: these samples go uncounted
            return;
        }
        ResetAhead();
    }
    const u32 slot = static_cast<u32>(begun % NumSlots);
    const bool precise = instance.IsOcclusionQueryPrecise();
    scheduler.Record([pool = *pool, slot, precise](vk::CommandBuffer cmdbuf) {
        cmdbuf.beginQuery(pool, slot,
                          precise ? vk::QueryControlFlagBits::ePrecise : vk::QueryControlFlags{});
    });
    ++begun;
    active = true;
}

void OcclusionQueries::EndSegment() {
    const u32 slot = static_cast<u32>((begun - 1) % NumSlots);
    scheduler.Record([pool = *pool, slot](vk::CommandBuffer cmdbuf) { cmdbuf.endQuery(pool, slot); });
    active = false;
}

void OcclusionQueries::ResetAhead() {
    // Slots from `resolved` on hold results not copied yet; those below `begun` were used.
    reset_until = std::max(reset_until, begun);
    const u64 target = std::min<u64>(begun + ResetBatch, resolved + NumSlots);
    if (target <= reset_until || reset_until - begun >= ResetBatch / 2) {
        return;
    }
    const u32 first = static_cast<u32>(reset_until % NumSlots);
    const u32 count = static_cast<u32>(target - reset_until);
    const u32 head = std::min(count, NumSlots - first);
    scheduler.Record([pool = *pool, first, count, head](vk::CommandBuffer cmdbuf) {
        cmdbuf.resetQueryPool(pool, first, head);
        if (count > head) {
            cmdbuf.resetQueryPool(pool, 0, count - head);
        }
    });
    reset_until = target;
}

void OcclusionQueries::Resolve() {
    const u64 ended = active ? begun - 1 : begun;
    if (pending.empty() && ended - resolved < NumSlots / 2) {
        return; // nothing to write, and room left
    }
    const u32 count = static_cast<u32>(ended - resolved);
    const u32 first = static_cast<u32>(resolved % NumSlots);
    std::vector<EventEntry> entries;
    entries.reserve(std::max<size_t>(pending.size(), 1));
    for (const auto& event : pending) {
        entries.push_back({u32(event.address), u32(event.address >> 32), event.pairs,
                           u32(event.upto - resolved)});
    }
    const u32 num_events = static_cast<u32>(entries.size());
    if (entries.empty()) {
        entries.push_back({}); // a binding to bind
    }
    auto& stream = buffer_cache.GetStreamBuffer();
    const u64 entries_size = entries.size() * sizeof(EventEntry);
    const u64 entries_offset =
        stream.Copy(entries.data(), entries_size, instance.StorageMinAlignment());
    const bool clear_total = !total_cleared;
    total_cleared = true;
    scheduler.Record([this, first, count, num_events, clear_total, stream = stream.Handle(),
                      entries_offset, entries_size](vk::CommandBuffer cmdbuf) {
        if (clear_total) {
            cmdbuf.fillBuffer(total->Handle(), 0, 8, 0);
        }
        const u32 head = std::min(count, NumSlots - first);
        if (head != 0) {
            cmdbuf.copyQueryPoolResults(*pool, first, head, results->Handle(), 0, sizeof(u64),
                                        vk::QueryResultFlagBits::e64 |
                                            vk::QueryResultFlagBits::eWait);
        }
        if (count > head) {
            cmdbuf.copyQueryPoolResults(*pool, 0, count - head, results->Handle(),
                                        head * sizeof(u64), sizeof(u64),
                                        vk::QueryResultFlagBits::e64 |
                                            vk::QueryResultFlagBits::eWait);
        }
        FullBarrier(cmdbuf);
        const std::array<vk::DescriptorBufferInfo, 3> infos{{
            {results->Handle(), 0, NumSlots * sizeof(u64)},
            {total->Handle(), 0, 8},
            {stream, entries_offset, entries_size},
        }};
        std::array<vk::WriteDescriptorSet, 3> writes{};
        for (u32 i = 0; i < writes.size(); ++i) {
            writes[i] = {.dstBinding = i,
                         .descriptorCount = 1,
                         .descriptorType = vk::DescriptorType::eStorageBuffer,
                         .pBufferInfo = &infos[i]};
        }
        const Push push{count, num_events};
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, writes);
        cmdbuf.pushConstants(*pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(push),
                             &push);
        cmdbuf.dispatch(1, 1, 1);
        FullBarrier(cmdbuf);
    });
    resolved = ended;
    pending.clear();
}

void OcclusionQueries::Suspend(bool inside_render_pass) {
    std::scoped_lock lk{mutex};
    if (!started) {
        return;
    }
    if (active) {
        EndSegment();
    }
    if (!inside_render_pass) {
        Resolve();
        ResetAhead();
    }
}

void OcclusionQueries::Resume(bool inside_render_pass) {
    std::scoped_lock lk{mutex};
    if (started && !active) {
        BeginSegment(inside_render_pass);
    }
}

void OcclusionQueries::Event(VAddr address, u32 pairs) {
    std::scoped_lock lk{mutex};
    if (!started) {
        // The first event: from here on a segment is always active (slots reset outside a
        // render pass first).
        scheduler.EndRendering();
        ResetAhead();
        started = true;
        scheduler.SetCarriedScope(this);
    }
    // The counters, written by the GPU through their device address; in place (the CPU reads
    // them, a VRAM copy would show it stale ones).
    const u32 size = (pairs - 1) * 16 + 8;
    buffer_cache.force_writes_in_place = true;
    const auto [buffer, offset] = buffer_cache.ObtainBuffer(address, size, true);
    buffer_cache.force_writes_in_place = false;
    if (active) {
        EndSegment();
    }
    pending.push_back({buffer->BufferDeviceAddress() + offset, pairs, begun});
    BeginSegment(scheduler.IsRendering());
    ++events;
    if (!scheduler.IsRendering() && pending.size() >= 64) {
        if (active) {
            EndSegment();
        }
        Resolve();
        ResetAhead();
        BeginSegment(false);
    }
    // BB_FRAME_STATS: how often the game measures (every 5 s).
    static auto last_report = std::chrono::steady_clock::now();
    static u64 reported = 0;
    if (const auto now = std::chrono::steady_clock::now();
        now - last_report >= std::chrono::seconds(5)) {
        std::printf("Occlusion queries: %llu events in %.0f s, %llu in all; %llu segments, "
                    "%llu not counted\n",
                    (unsigned long long)(events - reported),
                    std::chrono::duration<double>(now - last_report).count(),
                    (unsigned long long)events, (unsigned long long)begun,
                    (unsigned long long)starved);
        last_report = now;
        reported = events;
    }
}

} // namespace Vulkan
