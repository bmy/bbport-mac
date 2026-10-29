// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport vulkan: command buffers for submissions, reused once the GPU is done with the
// submission that last used them (grown four at a time).
#pragma once

#include <vector>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Semaphore;

class CommandPool {
public:
    CommandPool(const Instance& instance, Semaphore* timeline);
    ~CommandPool();

    /// A command buffer free for the current submission.
    vk::CommandBuffer Commit();

private:
    struct Entry {
        vk::CommandBuffer buffer;
        u64 tick; ///< the submission that used it last
    };
    void Grow();

    const Instance& instance;
    Semaphore* timeline;
    vk::UniqueCommandPool pool;
    std::vector<Entry> entries;
    size_t next = 0; ///< where the search starts: buffers are reused in turn
};

} // namespace Vulkan
