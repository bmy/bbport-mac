// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "vulkan/descriptor_heap.h"

#include <array>
#include <bit>

#include "common/assert.h"
#include "vulkan/instance.h"
#include "vulkan/timeline.h"

namespace Vulkan {

DescriptorHeap::DescriptorHeap(const Instance& instance, Semaphore* timeline_,
                               std::span<const vk::DescriptorPoolSize> pool_sizes_,
                               u32 sets_per_pool_)
    : device{instance.GetDevice()}, timeline{timeline_}, sets_per_pool{sets_per_pool_},
      pool_sizes{pool_sizes_} {
    NewPool();
}

DescriptorHeap::~DescriptorHeap() {
    device.destroyDescriptorPool(pool);
    for (const auto& [full, tick] : full_pools) {
        timeline->Wait(tick);
        device.destroyDescriptorPool(full);
    }
}

void DescriptorHeap::NewPool() {
    const auto [result, created] = device.createDescriptorPool({
        .flags = vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
        .maxSets = sets_per_pool,
        .poolSizeCount = u32(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });
    ASSERT_MSG(result == vk::Result::eSuccess, "descriptor pool: {}", vk::to_string(result));
    pool = created;
}

bool DescriptorHeap::AllocateBatch(vk::DescriptorSetLayout layout, Sets& sets) {
    std::array<vk::DescriptorSetLayout, Batch> layouts;
    layouts.fill(layout);
    sets.resize(Batch);
    const vk::DescriptorSetAllocateInfo info{
        .descriptorPool = pool,
        .descriptorSetCount = Batch,
        .pSetLayouts = layouts.data(),
    };
    const vk::Result result = device.allocateDescriptorSets(&info, sets.data());
    if (result == vk::Result::eSuccess) {
        return true;
    }
    ASSERT_MSG(result == vk::Result::eErrorOutOfPoolMemory ||
                   result == vk::Result::eErrorFragmentedPool,
               "descriptor set allocation: {}", vk::to_string(result));
    sets.clear();
    return false;
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout set_layout) {
    const u64 key = std::bit_cast<u64>(set_layout);
    auto it = free_sets.try_emplace(key).first;
    if (!it->second.empty()) {
        const vk::DescriptorSet set = it->second.back();
        it.value().pop_back();
        return set;
    }
    Sets sets;
    if (!AllocateBatch(set_layout, sets)) {
        // The pool is full: retired with the current tick; the oldest retired one is reset and
        // reused if the GPU is done with it, else a new pool. Sets of the old pool are dropped.
        full_pools.emplace_back(pool, timeline->CurrentTick());
        if (const auto [oldest, tick] = full_pools.front(); timeline->IsFree(tick)) {
            pool = oldest;
            full_pools.pop_front();
            const vk::Result reset = device.resetDescriptorPool(pool);
            ASSERT_MSG(reset == vk::Result::eSuccess, "descriptor pool reset: {}",
                       vk::to_string(reset));
        } else {
            NewPool();
        }
        free_sets.clear();
        ASSERT_MSG(AllocateBatch(set_layout, sets), "descriptor sets from a fresh pool");
        it = free_sets.try_emplace(key).first;
    }
    const vk::DescriptorSet set = sets.back();
    sets.pop_back();
    it.value() = std::move(sets);
    return set;
}

} // namespace Vulkan
