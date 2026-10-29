// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "vulkan/command_pool.h"

#include <array>

#include "common/assert.h"
#include "vulkan/instance.h"
#include "vulkan/timeline.h"

namespace Vulkan {

namespace {
constexpr u32 GrowBy = 4;
}

CommandPool::CommandPool(const Instance& instance_, Semaphore* timeline_)
    : instance{instance_}, timeline{timeline_} {
    const vk::Device device = instance.GetDevice();
    auto [result, created] = device.createCommandPoolUnique({
        .flags = vk::CommandPoolCreateFlagBits::eTransient |
                 vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex(),
    });
    ASSERT_MSG(result == vk::Result::eSuccess, "command pool: {}", vk::to_string(result));
    pool = std::move(created);
    SetObjectName(device, *pool, "CommandPool");
}

CommandPool::~CommandPool() = default;

void CommandPool::Grow() {
    std::array<vk::CommandBuffer, GrowBy> buffers;
    const vk::CommandBufferAllocateInfo info{
        .commandPool = *pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = GrowBy,
    };
    const vk::Device device = instance.GetDevice();
    ASSERT(device.allocateCommandBuffers(&info, buffers.data()) == vk::Result::eSuccess);
    for (const vk::CommandBuffer buffer : buffers) {
        SetObjectName(device, buffer, "CommandPool: Command Buffer {}", entries.size());
        entries.push_back({buffer, 0});
    }
}

vk::CommandBuffer CommandPool::Commit() {
    const auto take = [&](size_t index) {
        entries[index].tick = timeline->CurrentTick();
        next = (index + 1) % entries.size();
        return entries[index].buffer;
    };
    // From where the last one was taken to the end, again after asking the GPU, then from the
    // start; else a new one.
    for (int attempt = 0; attempt < 2 && !entries.empty(); ++attempt) {
        if (attempt == 1) {
            timeline->Refresh();
        }
        const u64 done = timeline->KnownGpuTick();
        for (size_t i = next; i < entries.size(); ++i) {
            if (entries[i].tick <= done) {
                return take(i);
            }
        }
    }
    const u64 done = timeline->KnownGpuTick();
    for (size_t i = 0; i < next && i < entries.size(); ++i) {
        if (entries[i].tick <= done) {
            return take(i);
        }
    }
    const size_t first_new = entries.size();
    Grow();
    return take(first_new);
}

} // namespace Vulkan
