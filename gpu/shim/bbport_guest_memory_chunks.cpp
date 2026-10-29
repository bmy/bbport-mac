// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport guest memory: chunks the GPU gets whole (dma-buf exports, host imports, private regions).
#include "bbport_guest_memory_internal.h"

namespace BbGuestMemory::detail {

/// The runtime's chunk allocator: a dma-buf fd of `size` bytes of GPU-visible memory, or -1.
int AllocChunk(u64 phys, u64 size) {
    {
        std::scoped_lock lk{mutex};
        if (chunk_count == MaxChunks) {
            return -1;
        }
    }
    static std::atomic<int> failures{0};
    const auto fail = [&](const char* what, vk::Result result) {
        if (failures.fetch_add(1) < 4) {
            std::fprintf(stderr, "Guest memory: %s failed (%s); chunk %#llx stays in the memfd\n",
                         what, vk::to_string(result).c_str(), (unsigned long long)phys);
        }
        return -1;
    };
    const vk::ExternalMemoryBufferCreateInfo external{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    const vk::BufferCreateInfo buffer_ci{
        .pNext = &external,
        .size = size,
        .usage = ChunkBufferUsage(),
        .sharingMode = vk::SharingMode::eExclusive,
    };
    const auto [buffer_result, buffer] = device.createBuffer(buffer_ci);
    if (buffer_result != vk::Result::eSuccess) {
        return fail("buffer creation", buffer_result);
    }
    const auto requirements = device.getBufferMemoryRequirements(buffer);
    const bool in_vram = ChunkInVram(phys, size);
    std::uint32_t type = FindType(requirements.memoryTypeBits, in_vram);
    if (type == ~0u && in_vram) {
        type = FindType(requirements.memoryTypeBits, false); // no host-visible VRAM (no ReBAR)
    }
    if (type == ~0u) {
        device.destroyBuffer(buffer);
        return fail("finding a cached system memory type", vk::Result::eErrorFeatureNotPresent);
    }
    const vk::MemoryAllocateFlagsInfo address_flags{
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::ExportMemoryAllocateInfo export_info{
        .pNext = LayerMemory() ? &address_flags : nullptr,
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    };
    const auto [memory_result, memory] = device.allocateMemory({
        .pNext = &export_info,
        .allocationSize = requirements.size,
        .memoryTypeIndex = type,
    });
    if (memory_result != vk::Result::eSuccess) {
        device.destroyBuffer(buffer);
        return fail("allocation", memory_result);
    }
    if (const auto result = device.bindBufferMemory(buffer, memory, 0);
        result != vk::Result::eSuccess) {
        device.freeMemory(memory);
        device.destroyBuffer(buffer);
        return fail("binding", result);
    }
    const auto [fd_result, fd] = device.getMemoryFdKHR({
        .memory = memory,
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
    });
    if (fd_result != vk::Result::eSuccess) {
        device.freeMemory(memory);
        device.destroyBuffer(buffer);
        return fail("dma-buf export", fd_result);
    }
    {
        std::scoped_lock lk{mutex};
        std::size_t i = chunk_count++;
        for (; i > 0 && chunks[i - 1]->phys > phys; --i) {
            chunks[i] = chunks[i - 1];
        }
        chunks[i] = new Chunk{phys, size, buffer, memory, next_index++};
        RegisterChunk(chunks[i]);
    }
    const u64 total = chunk_bytes.fetch_add(size) + size;
    const bool vram_type = bool(memory_properties.memoryTypes[type].propertyFlags &
                                vk::MemoryPropertyFlagBits::eDeviceLocal);
    std::printf("Guest memory: direct memory %#llx+%llu MiB in GPU-visible memory (dma-buf%s), "
                "%llu MiB so far\n",
                (unsigned long long)phys, (unsigned long long)(size >> 20),
                vram_type ? ", VRAM through the BAR" : "", (unsigned long long)(total >> 20));
    return fd;
}


/// The runtime's chunk release (runtime_memory_set_guest_chunk_free): [phys, phys + size).
void FreeChunk(u64 phys, u64 size) {
    Chunk* chunk = nullptr;
    {
        std::scoped_lock lk{mutex};
        for (std::size_t i = 0; i < chunk_count; ++i) {
            if (chunks[i]->phys == phys && chunks[i]->size == size) {
                chunk = chunks[i];
                for (std::size_t j = i; j + 1 < chunk_count; ++j) {
                    chunks[j] = chunks[j + 1];
                }
                --chunk_count;
                break;
            }
        }
        if (!chunk) {
            return;
        }
        retired_chunks.push_back({chunk, std::chrono::steady_clock::now()});
        retired_count.store(retired_chunks.size(), std::memory_order_release);
    }
    BbLayer::GpuMemory::Get().RemoveGuestSource(chunk);
    chunk_bytes.fetch_sub(size);
    std::printf("Guest memory: direct memory %#llx+%llu MiB given back (its GPU memory freed in 3 s)\n",
                (unsigned long long)phys, (unsigned long long)(size >> 20));
}


std::uint32_t FindHostType(std::uint32_t bits) {
#ifdef __APPLE__
    // bbport (macOS): unified memory: KosmicKrisp's one type is device-local too.
    constexpr vk::MemoryPropertyFlags not_wanted{};
#else
    constexpr vk::MemoryPropertyFlags not_wanted = vk::MemoryPropertyFlagBits::eDeviceLocal;
#endif
    // Cached host memory first, then any host-visible type the import allows.
    for (const bool cached : {true, false}) {
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            const auto flags = memory_properties.memoryTypes[i].propertyFlags;
            if (!(bits & (1u << i)) || !(flags & vk::MemoryPropertyFlagBits::eHostVisible) ||
                (flags & not_wanted) || !TypeAllowed(flags) ||
                (cached &&!(flags & vk::MemoryPropertyFlagBits::eHostCached))) {
                continue;
            }
            return i;
        }
    }
    return ~0u;
}


/// Imports [pointer, pointer + size) as device memory with a buffer over it; false on failure.
bool ImportHost(void* pointer, u64 size, vk::Buffer& buffer, vk::DeviceMemory& memory,
                vk::Result& error) {
    const vk::ExternalMemoryBufferCreateInfo external{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
    };
    const auto [buffer_result, created] = device.createBuffer({
        .pNext = &external,
        .size = size,
        .usage = ChunkBufferUsage(),
        .sharingMode = vk::SharingMode::eExclusive,
    });
    if (buffer_result != vk::Result::eSuccess) {
        error = buffer_result;
        return false;
    }
    const auto [props_result, props] = device.getMemoryHostPointerPropertiesEXT(
        vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, pointer);
    const auto requirements = device.getBufferMemoryRequirements(created);
    // AMD's driver gives buffers made for external memory no memory type bits (RPCS3's
    // "AMD driver bug" workaround): the pointer's own types then.
    std::uint32_t bits = props.memoryTypeBits & requirements.memoryTypeBits;
    if (bits == 0 && props_result == vk::Result::eSuccess) {
        bits = props.memoryTypeBits;
    }
    const std::uint32_t type = props_result == vk::Result::eSuccess ? FindHostType(bits) : ~0u;
    if (type == ~0u) {
        device.destroyBuffer(created);
        error = props_result != vk::Result::eSuccess ? props_result
                                                     : vk::Result::eErrorFeatureNotPresent;
        return false;
    }
    const vk::MemoryAllocateFlagsInfo address_flags{
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::ImportMemoryHostPointerInfoEXT import_info{
        .pNext = LayerMemory() ? &address_flags : nullptr,
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        .pHostPointer = pointer,
    };
    const auto [memory_result, allocated] = device.allocateMemory({
        .pNext = &import_info,
        .allocationSize = size,
        .memoryTypeIndex = type,
    });
    if (memory_result != vk::Result::eSuccess) {
        device.destroyBuffer(created);
        error = memory_result;
        return false;
    }
    if (const auto result = device.bindBufferMemory(created, allocated, 0);
        result != vk::Result::eSuccess) {
        device.freeMemory(allocated);
        device.destroyBuffer(created);
        error = result;
        return false;
    }
#ifdef __APPLE__
    // bbport (macOS): KosmicKrisp reports success when Metal refuses the pointer
    // (newBufferWithBytesNoCopy nil): the buffer then has no GPU address.
    if (LayerMemory() && !device.getBufferAddress({.buffer = created})) {
        device.freeMemory(allocated);
        device.destroyBuffer(created);
        error = vk::Result::eErrorInvalidExternalHandle;
        return false;
    }
#endif
    buffer = created;
    memory = allocated;
    return true;
}


/// The runtime's chunk allocator in host import mode: CHUNK_IMPORTED (-3, the chunk stays in the
/// memfd) once its backing view is imported, or -1.
int AllocImported(u64 phys, u64 size) {
    return RegisterImported(phys, size, runtime_memory_backing_pointer(phys)) ? -3 : -1;
}

/// The runtime's importer for private regions: 0 once the GPU imported [pointer, +size).
int ImportPrivate(u64 phys, u64 size, void* pointer) {
    return RegisterImported(phys, size, pointer) ? 0 : -1;
}

bool RegisterImported(u64 phys, u64 size, void* pointer) {
    {
        std::scoped_lock lk{mutex};
        if (chunk_count == MaxChunks) {
            return false;
        }
    }
    if (!pointer || reinterpret_cast<std::uintptr_t>(pointer) % import_alignment != 0 ||
        size % import_alignment != 0) {
#ifdef __APPLE__
        // bbport: said, not silent (an unimported chunk reads as zeros on the GPU).
        static std::atomic<int> told{0};
        if (pointer && told.fetch_add(1, std::memory_order_relaxed) < 4) {
            std::printf("Guest memory: %p+%#llx not aligned to %llu bytes: not imported\n", pointer,
                        (unsigned long long)size, (unsigned long long)import_alignment);
        }
#endif
        return false;
    }
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::Result error{};
    bool imported = false;
    if (host_sim && !private_regions) {
        // The pages the backing view shows, as NVIDIA's import of it would take them.
        const int fd = MakeUdmabuf(runtime_memory_pool_fd(), phys, size);
        imported = fd >= 0 && ImportDmaBuf(fd, size, buffer, memory, error);
        if (fd < 0) {
            error = vk::Result::eErrorInvalidExternalHandle;
        } else if (!imported) {
            close(fd);
        }
    } else {
        const auto t0 = std::chrono::steady_clock::now();
        imported = ImportHost(pointer, size, buffer, memory, error);
        if (size >= (u64{64} << 20)) {
            std::printf("Guest memory: host import of %llu MiB took %.1f ms\n", (unsigned long long)(size >> 20),
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
    }
    if (!imported) {
        static std::atomic<int> failures{0};
        if (failures.fetch_add(1) < 4) {
            std::fprintf(stderr, "Guest memory: importing chunk %#llx failed (%s); it stays in the "
                                 "memfd\n",
                         (unsigned long long)phys, vk::to_string(error).c_str());
        }
        return false;
    }
    {
        std::scoped_lock lk{mutex};
        std::size_t i = chunk_count++;
        for (; i > 0 && chunks[i - 1]->phys > phys; --i) {
            chunks[i] = chunks[i - 1];
        }
        chunks[i] = new Chunk{phys, size, buffer, memory, next_index++};
        RegisterChunk(chunks[i]);
    }
    const u64 total = chunk_bytes.fetch_add(size) + size;
    std::printf("Guest memory: direct memory %#llx+%llu MiB imported by the GPU (host memory), "
                "%llu MiB so far\n",
                (unsigned long long)phys, (unsigned long long)(size >> 20),
                (unsigned long long)(total >> 20));
    return true;
}

} // namespace BbGuestMemory::detail
