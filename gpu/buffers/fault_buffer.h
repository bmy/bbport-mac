// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport buffers: shaders that reach memory through device addresses mark the pages the buffer
// cache does not hold in a bit buffer; once per submission the bits are collected on the GPU
// (buffers/shaders/fault_scan.comp) into a host buffer, and the pages are brought in when the GPU
// is done with it.
#pragma once

#include <array>

#include "buffers/buffer.h"
#include "buffers/range_set.h"

namespace VideoCore {

class BufferCache;

class FaultBuffer {
public:
    FaultBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                BufferCache& buffer_cache, u32 page_bits, u64 num_pages);

    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return &bits;
    }
    /// Collects the marked pages (results read back when the GPU finishes them).
    void ProcessFaultBuffer();

private:
    static constexpr size_t MaxPendingReads = 8;

    Vulkan::Scheduler& scheduler;
    BufferCache& buffer_cache;
    u64 page_size;
    u64 num_pages;
    u64 bits_size;
    Buffer bits;     ///< one bit per page
    Buffer readback; ///< MaxPendingReads areas of a count and page addresses
    std::array<u64, MaxPendingReads> area_ticks{};
    u32 next_area = 0;
    RangeSet faulted;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;
};

} // namespace VideoCore
