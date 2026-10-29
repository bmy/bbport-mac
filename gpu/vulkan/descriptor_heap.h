// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport vulkan: descriptor sets for pipelines without push descriptors, allocated in batches
// per set layout from pools; a full pool is reset for reuse once the GPU is done with it.
#pragma once

#include <deque>
#include <span>

#include <boost/container/static_vector.hpp>
#include <tsl/robin_map.h>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Semaphore;

class DescriptorHeap {
public:
    DescriptorHeap(const Instance& instance, Semaphore* timeline,
                   std::span<const vk::DescriptorPoolSize> pool_sizes, u32 sets_per_pool = 1024);
    ~DescriptorHeap();

    vk::DescriptorSet Commit(vk::DescriptorSetLayout set_layout);

private:
    static constexpr u32 Batch = 32;
    using Sets = boost::container::static_vector<vk::DescriptorSet, Batch>;

    /// A batch of sets of the layout from the current pool; false when it is full.
    bool AllocateBatch(vk::DescriptorSetLayout layout, Sets& sets);
    void NewPool();

    vk::Device device;
    Semaphore* timeline;
    u32 sets_per_pool;
    std::span<const vk::DescriptorPoolSize> pool_sizes;
    vk::DescriptorPool pool;
    std::deque<std::pair<vk::DescriptorPool, u64>> full_pools; ///< with the tick they were in use
    tsl::robin_map<u64, Sets> free_sets;                      ///< by layout
};

} // namespace Vulkan
