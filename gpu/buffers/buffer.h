// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport buffers: Vulkan buffers with their memory: VMA allocations (device local, upload,
// readback, stream), sparse arenas (blocks bound on demand), and buffers over guest memory chunks
// owned elsewhere. StreamBuffer: a ring of host-visible memory for per-draw data.
#pragma once

#include <cstring>
#include <array>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include "buffers/memory_model.h"
#include "common/assert.h"
#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

struct VmaAllocationInfo;

namespace VideoCore {

enum class MemoryType : u8 {
    DeviceLocal,  ///< VRAM
    HostUncached, ///< host visible, for uploads (write-combined)
    HostCached,   ///< host visible and cached, for readbacks
    Stream,       ///< device local and host visible if there is such memory, else host
    Sparse,       ///< no memory: blocks bound later (arenas)
};

constexpr vk::BufferUsageFlags ReadFlags =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

constexpr vk::BufferUsageFlags AllFlags =
    ReadFlags | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;

/// A buffer handle with its VMA allocation (none for sparse and external buffers).
struct BufferAllocation {
    BufferAllocation(vk::Device device, VmaAllocator allocator);
    ~BufferAllocation();
    BufferAllocation(const BufferAllocation&) = delete;
    BufferAllocation& operator=(const BufferAllocation&) = delete;
    BufferAllocation(BufferAllocation&& other) noexcept;
    BufferAllocation& operator=(BufferAllocation&& other) noexcept;

    /// Creates the buffer (sparse ones bound to nothing yet); fills `out_info` for allocations.
    void Create(vk::BufferCreateInfo& create_info, MemoryType type, VmaAllocationInfo* out_info);
    void Destroy();

    vk::Device device;
    vk::Buffer buffer{};
    vk::DeviceAddress bda_addr{};
    VmaAllocator allocator;
    VmaAllocation allocation{};
};

/// While it lives, DeviceLocal buffers made on this thread are device-local memory or none: a failed
/// allocation leaves the Buffer !Valid() instead of falling back to system memory (mirrors that
/// landed in system memory when VRAM was full: copies over the bus for nothing, 2026-10-26).
struct VramOnlyScope {
    VramOnlyScope();
    ~VramOnlyScope();
    VramOnlyScope(const VramOnlyScope&) = delete;
    VramOnlyScope& operator=(const VramOnlyScope&) = delete;
    bool previous;
};

struct Buffer {
    Buffer(const Vulkan::Instance& instance, VAddr cpu_addr, u64 size_bytes, MemoryType type,
           std::string_view debug_name = "");
    /// Over memory owned elsewhere (a guest memory chunk, exported): transfers and texel and
    /// storage reads; with the layer's memory module vertex, index, indirect data and device
    /// addresses too.
    Buffer(const Vulkan::Instance& instance, u64 size_bytes, vk::DeviceMemory memory,
           std::string_view debug_name);
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    Buffer(Buffer&&) = default;
    Buffer& operator=(Buffer&&) = default;

    [[nodiscard]] bool IsInBounds(VAddr addr, u64 size) const noexcept {
        return addr >= cpu_addr && addr + size <= cpu_addr + size_bytes;
    }
    /// False when made in a VramOnlyScope without room in VRAM (no handle, no memory).
    [[nodiscard]] bool Valid() const noexcept {
        return buffer.buffer != VK_NULL_HANDLE;
    }
    [[nodiscard]] VAddr CpuAddr() const noexcept {
        return cpu_addr;
    }
    [[nodiscard]] u64 Offset(VAddr addr) const noexcept {
        return addr - cpu_addr;
    }
    size_t SizeBytes() const {
        return size_bytes;
    }
    vk::Buffer Handle() const noexcept {
        return buffer.buffer;
    }
    vk::DeviceAddress BufferDeviceAddress() const noexcept {
        ASSERT_MSG(buffer.bda_addr != 0, "buffer has no device address");
        return buffer.bda_addr;
    }

    /// Host writes in [offset, offset + size) made visible to the device (non-coherent memory).
    void Flush(u64 offset, u64 size);
    /// Device writes made visible to the host.
    void Invalidate(u64 offset, u64 size);

    VAddr cpu_addr = 0;
    size_t size_bytes = 0;
    std::span<u8> mapped_data{};
    bool is_coherent{};
    MemoryType mem_type{MemoryType::DeviceLocal};
    /// Bytes the allocation holds when more than size_bytes (a pooled mirror, BufferCache::LayerNewMirror).
    u64 alloc_bytes = 0;
    BufferAllocation buffer;
};

/// A ring of memory handed out front to back. The ring is cut into Sections; each remembers the
/// newest submission (scheduler tick) that used it. Space within the section in use is a bump of
/// the cursor; entering the next section is the only point where the GPU must be done with what
/// that section held one lap earlier. Nothing is kept per request.
struct StreamBuffer : public Buffer {
    StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                 MemoryType mem_type, u64 size_bytes);

    /// Space for `size` bytes (to be followed by Commit): its pointer (null for memory not
    /// mapped) and offset; {null, 0} when it would have to wait and `allow_wait` is false.
    std::pair<u8*, u64> Map(u64 size, u64 alignment = 0, bool allow_wait = true);
    /// The mapped space is written: visible to the GPU, used by the current submission.
    void Commit();
    /// Map and Commit without a write.
    std::optional<u64> Reserve(u64 size, u64 alignment = 0, bool allow_wait = true);
    /// The submission tick of the last commit (0: never used).
    [[nodiscard]] u64 LastTick() const noexcept {
        return last_tick;
    }
    /// Copies `size` bytes from `src` into new space; its offset.
    u64 Copy(auto src, size_t size, size_t alignment = 0) {
        const auto [data, offset] = Map(size, alignment);
        ASSERT_MSG(data, "stream buffer: no room for {:#x} bytes", size);
        std::memcpy(data, reinterpret_cast<const void*>(src), size);
        Commit();
        return offset;
    }

private:
    static constexpr u32 Sections = 64;
    /// Finds room for the next `size` bytes (pending_offset, pending_size); false when it would
    /// have to wait for the GPU and may not.
    bool Prepare(u64 size, u64 alignment, bool allow_wait);
    /// The pending space is used by the current submission.
    void Advance();

    Vulkan::Scheduler& scheduler;
    vk::DeviceSize atom_size{};
    u64 section_bytes = 1;
    std::array<u64, Sections> section_ticks{}; ///< newest submission in each (0: none)
    u32 claimed = 0;      ///< sections [0, claimed) belong to the current lap
    u64 cursor = 0;       ///< the next free byte of the current lap
    u64 pending_offset = 0;
    u64 pending_size = 0; ///< the mapped space
    u64 last_tick = 0;
};

} // namespace VideoCore
