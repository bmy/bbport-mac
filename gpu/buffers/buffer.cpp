// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "buffers/buffer.h"
#include "shim/bb_env.h"

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <numeric>

#include <vk_mem_alloc.h>

#include "bbport_guest_memory.h"
#include "bbport_toggles.h"
#include "common/alignment.h"
#include "common/assert.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "render/scheduler.h"

namespace VideoCore {

namespace {

const char* TypeName(MemoryType type) {
    constexpr const char* names[] = {"DeviceLocal", "HostUncached", "HostCached", "Stream",
                                     "Sparse"};
    return u32(type) < std::size(names) ? names[u32(type)] : "Invalid";
}

/// VramOnlyScope (BufferCache mirrors): device-local memory or nothing.
thread_local bool vram_only = false;

/// VMA's view of a memory type: where it should live and how the host touches it.
VmaAllocationCreateInfo AllocationInfo(MemoryType type, bool device_address, u64 size) {
    VmaAllocationCreateInfo info{
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_UNKNOWN,
        .preferredFlags = type != MemoryType::DeviceLocal ? VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                                                          : VkMemoryPropertyFlags{},
    };
    // Buffers with device addresses: their own allocation (BB_BDA_DEDICATED_MB=N: only from N MiB
    // up, smaller ones in VMA's blocks). Mirrors in blocks fragmented them on the 2 GB laptop: the
    // blocks held 1.4-1.7 GB with 0.55-1 GB used, VRAM past the budget, the GPU paging (2026-10-25).
    static const u64 dedicated_from = [] {
        const char* env = bb_env("BB_BDA_DEDICATED_MB");
        return u64(env && *env ? std::strtoull(env, nullptr, 10) : 0) << 20;
    }();
    if (device_address && size >= dedicated_from) {
        info.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    }
    switch (type) {
    case MemoryType::DeviceLocal:
        info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (vram_only) {
            info.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        }
        break;
    case MemoryType::Stream:
        info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        break;
    case MemoryType::HostUncached:
        info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        info.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT |
                      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        break;
    case MemoryType::HostCached:
        info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        info.flags |=
            VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        break;
    case MemoryType::Sparse:
        break;
    }
    return info;
}

/// Sparse arenas take exported guest memory chunks (in place) as well as VRAM blocks: created
/// for external memory; a driver that cannot do that, or not with VRAM too, stops the game.
vk::Buffer CreateArena(vk::Device device, VmaAllocator allocator,
                       vk::BufferCreateInfo& create_info) {
    create_info.flags |=
        vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency;
    const vk::ExternalMemoryBufferCreateInfo external{
        .handleTypes = BbGuestMemory::HandleType(),
    };
    create_info.pNext = &external;
    const auto [result, buffer] = device.createBuffer(create_info);
    create_info.pNext = nullptr;
    if (result != vk::Result::eSuccess) {
        std::printf("Guest memory: the arena cannot take exported memory (%s)\n",
                    vk::to_string(result).c_str());
        GuestMemoryUnusable("the sparse arena cannot take exported memory (BB_LAYER_MEMORY=0)");
    }
    const auto requirements = device.getBufferMemoryRequirements(buffer);
    const VkPhysicalDeviceMemoryProperties* properties = nullptr;
    vmaGetMemoryProperties(allocator, &properties);
    for (u32 i = 0; i < properties->memoryTypeCount; ++i) {
        if ((requirements.memoryTypeBits >> i & 1) &&
            (properties->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            return buffer;
        }
    }
    device.destroyBuffer(buffer);
    GuestMemoryUnusable("the sparse arena for guest memory cannot take VRAM (BB_LAYER_MEMORY=0)");
}

} // namespace

VramOnlyScope::VramOnlyScope() : previous{vram_only} {
    vram_only = true;
}

VramOnlyScope::~VramOnlyScope() {
    vram_only = previous;
}

BufferAllocation::BufferAllocation(vk::Device device_, VmaAllocator allocator_)
    : device{device_}, allocator{allocator_} {}

BufferAllocation::~BufferAllocation() {
    Destroy();
}

BufferAllocation::BufferAllocation(BufferAllocation&& other) noexcept
    : device{other.device}, buffer{std::exchange(other.buffer, VK_NULL_HANDLE)},
      bda_addr{std::exchange(other.bda_addr, 0)},
      allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
      allocation{std::exchange(other.allocation, VK_NULL_HANDLE)} {}

BufferAllocation& BufferAllocation::operator=(BufferAllocation&& other) noexcept {
    if (this != &other) {
        Destroy();
        device = other.device;
        buffer = std::exchange(other.buffer, VK_NULL_HANDLE);
        bda_addr = std::exchange(other.bda_addr, 0);
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
    }
    return *this;
}

void BufferAllocation::Destroy() {
    if (allocation) {
        BbStats::VkMemTimer timer{BbStats::t_vk_free, BbStats::n_vk_free};
        vmaDestroyBuffer(allocator, buffer, allocation);
    } else if (buffer) {
        device.destroyBuffer(buffer);
    }
    buffer = VK_NULL_HANDLE;
    allocation = VK_NULL_HANDLE;
    bda_addr = 0;
}

void BufferAllocation::Create(vk::BufferCreateInfo& create_info, MemoryType type,
                              VmaAllocationInfo* out_info) {
    const bool device_address =
        bool(create_info.usage & vk::BufferUsageFlagBits::eShaderDeviceAddress);
    if (type == MemoryType::Sparse) {
        buffer = CreateArena(device, allocator, create_info);
    } else {
        const VmaAllocationCreateInfo alloc_info = AllocationInfo(type, device_address, create_info.size);
        const VkBufferCreateInfo vk_info = create_info;
        VkBuffer handle{};
        BbStats::VkMemTimer timer{BbStats::t_vk_alloc, BbStats::n_vk_alloc};
        const VkResult result =
            vmaCreateBuffer(allocator, &vk_info, &alloc_info, &handle, &allocation, out_info);
        if (result != VK_SUCCESS && vram_only) {
            allocation = VK_NULL_HANDLE; // no room in VRAM: the caller does without (Buffer::Valid)
            return;
        }
        ASSERT_MSG(result == VK_SUCCESS, "buffer allocation: {}",
                   vk::to_string(vk::Result{result}));
        buffer = handle;
    }
    if (device_address) {
        bda_addr = device.getBufferAddress({.buffer = buffer});
        ASSERT_MSG(bda_addr != 0, "no device address");
    }
}

Buffer::Buffer(const Vulkan::Instance& instance, VAddr cpu_addr_, u64 size_bytes_,
               MemoryType type, std::string_view debug_name)
    : cpu_addr{cpu_addr_}, size_bytes{size_bytes_}, mem_type{type},
      buffer{instance.GetDevice(), instance.GetAllocator()} {
    vk::BufferCreateInfo create_info{
        .size = size_bytes,
        .usage = AllFlags,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    // Arenas are read by the readback queue too.
    const std::array<u32, 2> families = {instance.GetGraphicsQueueFamilyIndex(),
                                         instance.GetReadbackQueueFamilyIndex()};
    if (type == MemoryType::Sparse && instance.GetReadbackQueue()) {
        create_info.sharingMode = vk::SharingMode::eConcurrent;
        create_info.queueFamilyIndexCount = u32(families.size());
        create_info.pQueueFamilyIndices = families.data();
    }
    VmaAllocationInfo alloc_info{};
    buffer.Create(create_info, type, &alloc_info);
    if (!buffer.buffer) {
        return; // VramOnlyScope and no room: !Valid()
    }

    const auto device = instance.GetDevice();
    if (debug_name.empty()) {
        Vulkan::SetObjectName(device, Handle(), "Buffer {:#x}:{:#x}", cpu_addr, size_bytes);
    } else {
        Vulkan::SetObjectName(device, Handle(), debug_name);
    }
    if (type != MemoryType::Sparse) {
        VkMemoryPropertyFlags flags{};
        vmaGetAllocationMemoryProperties(instance.GetAllocator(), buffer.allocation, &flags);
        if (alloc_info.pMappedData) {
            mapped_data = {static_cast<u8*>(alloc_info.pMappedData), size_bytes};
        }
        is_coherent = flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        // Big ones by name (what the port keeps in VRAM: 2 GB cards run short).
        if (size_bytes >= (u64{16} << 20)) {
            std::printf("Buffer: %.*s %llu MiB, %s\n", int(debug_name.size()), debug_name.data(),
                        (unsigned long long)(size_bytes >> 20),
                        (flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "VRAM" : "system memory");
        }
    }
}

Buffer::Buffer(const Vulkan::Instance& instance, u64 size_bytes_, vk::DeviceMemory memory,
               std::string_view debug_name)
    : size_bytes{size_bytes_}, mem_type{MemoryType::HostCached},
      buffer{instance.GetDevice(), instance.GetAllocator()} {
    const vk::ExternalMemoryBufferCreateInfo external{
        .handleTypes = BbGuestMemory::HandleType(),
    };
    const bool layer = BbGuestMemory::LayerMemory();
    vk::BufferUsageFlags usage =
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformBuffer |
        vk::BufferUsageFlagBits::eUniformTexelBuffer | vk::BufferUsageFlagBits::eStorageTexelBuffer;
    if (layer) {
        usage |= vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
                 vk::BufferUsageFlagBits::eIndirectBuffer |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    const auto device = instance.GetDevice();
    buffer.buffer = Vulkan::Check(device.createBuffer({
        .pNext = &external,
        .size = size_bytes,
        .usage = usage,
        .sharingMode = vk::SharingMode::eExclusive,
    }));
    const auto result = device.bindBufferMemory(buffer.buffer, memory, 0);
    ASSERT_MSG(result == vk::Result::eSuccess, "binding a guest memory chunk: {}",
               vk::to_string(result));
    if (layer) {
        buffer.bda_addr = device.getBufferAddress({.buffer = buffer.buffer});
    }
    Vulkan::SetObjectName(device, Handle(), debug_name);
    is_coherent = true;
}

void Buffer::Flush(u64 offset, u64 size) {
    if (!mapped_data.empty() && !is_coherent) {
        vmaFlushAllocation(buffer.allocator, buffer.allocation, offset, size);
    }
}

void Buffer::Invalidate(u64 offset, u64 size) {
    if (!mapped_data.empty() && !is_coherent) {
        vmaInvalidateAllocation(buffer.allocator, buffer.allocation, offset, size);
    }
}

StreamBuffer::StreamBuffer(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler_,
                           MemoryType type, u64 size)
    : Buffer{instance, 0, size, type}, scheduler{scheduler_},
      atom_size{instance.NonCoherentAtomSize()}, section_bytes{std::max<u64>(
                                                     (size + Sections - 1) / Sections, 1)} {
    Vulkan::SetObjectName(instance.GetDevice(), Handle(), "StreamBuffer({}):{:#x}", TypeName(type),
                          size);
}

bool StreamBuffer::Prepare(u64 size, u64 alignment, bool allow_wait) {
    // Host writes to non-coherent memory are flushed in whole atoms: pieces never share one.
    if (!mapped_data.empty() && !is_coherent) {
        size = Common::AlignUp(size, atom_size);
        alignment = alignment ? std::lcm(alignment, atom_size) : atom_size;
    }
    if (size > size_bytes) {
        return false;
    }
    u64 offset = alignment ? Common::AlignUp(cursor, alignment) : cursor;
    const bool wraps = offset + size > size_bytes;
    if (wraps) {
        offset = 0; // what does not fit before the end starts the next lap
    }
    // The sections this piece enters that the lap has not claimed yet: what they held one lap
    // earlier must be done. Checked before anything changes (a refusal leaves the ring as it was).
    const u32 first = wraps ? 0 : claimed;
    const u32 last = size ? u32((offset + size - 1) / section_bytes) : first;
    for (u32 section = first; section <= last && section < Sections; ++section) {
        const u64 tick = section_ticks[section];
        if (tick != 0 && !scheduler.IsFree(tick)) {
            if (!allow_wait) {
                return false;
            }
            scheduler.Wait(tick);
        }
    }
    if (wraps) {
        claimed = 0;
        // BB_STREAM_STATS=1 (diagnostics): what a frame streams, for the ring's size.
        static const bool stats = bb_env_on("BB_STREAM_STATS");
        if (stats) {
            static u64 laps = 0, frame0 = 0;
            static auto since = std::chrono::steady_clock::now();
            ++laps;
            const auto now = std::chrono::steady_clock::now();
            if (now - since >= std::chrono::seconds(10)) {
                const u64 frames = BbStats::frame_number.load(std::memory_order_relaxed) - frame0;
                std::printf("Stream buffer %s (%llu MiB): %llu laps in 10 s, %.1f MiB a frame\n",
                            TypeName(mem_type), (unsigned long long)(size_bytes >> 20),
                            (unsigned long long)laps,
                            frames ? double(laps * size_bytes) / frames / 1048576.0 : 0.0);
                laps = 0;
                frame0 = BbStats::frame_number.load(std::memory_order_relaxed);
                since = now;
            }
        }
    }
    claimed = std::max<u32>(claimed, std::min<u32>(last + 1, Sections));
    pending_offset = offset;
    pending_size = size;
    return true;
}

void StreamBuffer::Advance() {
    last_tick = scheduler.CurrentTick();
    if (pending_size != 0) {
        const u32 first = u32(pending_offset / section_bytes);
        const u32 last = u32((pending_offset + pending_size - 1) / section_bytes);
        for (u32 section = first; section <= last && section < Sections; ++section) {
            section_ticks[section] = last_tick;
        }
    }
    cursor = pending_offset + pending_size;
}

std::pair<u8*, u64> StreamBuffer::Map(u64 size, u64 alignment, bool allow_wait) {
    if (!Prepare(size, alignment, allow_wait)) {
        return {nullptr, 0};
    }
    return {mapped_data.empty() ? nullptr : mapped_data.data() + pending_offset, pending_offset};
}

void StreamBuffer::Commit() {
    if (mem_type == MemoryType::HostCached) {
        Invalidate(pending_offset, pending_size);
    } else {
        Flush(pending_offset, pending_size);
    }
    Advance();
}

std::optional<u64> StreamBuffer::Reserve(u64 size, u64 alignment, bool allow_wait) {
    if (!Prepare(size, alignment, allow_wait)) {
        return std::nullopt;
    }
    const u64 offset = pending_offset;
    Advance();
    return offset;
}

} // namespace VideoCore
