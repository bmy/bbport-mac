// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport guest memory: startup checks of what the driver can do with the game's memory.
#include "bbport_guest_memory_internal.h"
#include "shim/bb_env.h"

namespace BbGuestMemory::detail {
/// Runs `record` on the graphics queue and waits; false on any failure.
template <typename Record>
bool RunOnce(const Vulkan::Instance& instance, Record&& record) {
    const vk::Device dev = instance.GetDevice();
    const auto [pool_result, pool] = dev.createCommandPool(
        {.queueFamilyIndex = instance.GetGraphicsQueueFamilyIndex()});
    if (pool_result != vk::Result::eSuccess) {
        return false;
    }
    bool ok = false;
    const auto [alloc_result, cmdbufs] = dev.allocateCommandBuffers(
        {.commandPool = pool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1});
    const auto [fence_result, fence] = dev.createFence({});
    if (alloc_result == vk::Result::eSuccess && fence_result == vk::Result::eSuccess) {
        const vk::CommandBuffer cmd = cmdbufs[0];
        if (cmd.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}) ==
            vk::Result::eSuccess) {
            record(cmd);
            const vk::MemoryBarrier to_host{.srcAccessMask = vk::AccessFlagBits::eTransferWrite,
                                            .dstAccessMask = vk::AccessFlagBits::eHostRead};
            cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                vk::PipelineStageFlagBits::eHost, {}, to_host, {}, {});
            if (cmd.end() == vk::Result::eSuccess) {
                const vk::SubmitInfo submit{.commandBufferCount = 1, .pCommandBuffers = &cmd};
                ok = instance.GetGraphicsQueue().submit(submit, fence) == vk::Result::eSuccess &&
                     dev.waitForFences(fence, true, 5'000'000'000ull) == vk::Result::eSuccess;
            }
        }
    }
    if (fence_result == vk::Result::eSuccess) {
        dev.destroyFence(fence);
    }
    dev.destroyCommandPool(pool);
    return ok;
}


/// udmabuf imports as lazy guest memory uses them: a sealed memfd, two overlapping udmabufs
/// imported, the GPU filling through one what the CPU then reads in the memfd, and copying what
/// the CPU wrote there through the other.
bool UdmabufWorks(const Vulkan::Instance& instance, const char*& why) {
#ifdef _WIN32
    (void)instance;
    why = "no udmabuf on Windows";
    return false;
#else
    why = "/dev/udmabuf cannot be opened";
    if (access("/dev/udmabuf", R_OK | W_OK) != 0) {
        return false;
    }
    if (!instance.IsGuestMemoryExportSupported()) {
        why = "no dma-buf import (VK_EXT_external_memory_dma_buf)";
        return false;
    }
    if (FILE* limit = std::fopen("/sys/module/udmabuf/parameters/size_limit_mb", "r")) {
        unsigned mb = 0;
        if (std::fscanf(limit, "%u", &mb) == 1 && mb >= 32) {
            import_limit = u64{mb} << 20;
        }
        std::fclose(limit);
    }
    constexpr u64 Size = u64{8} << 20, Half = Size / 2;
    const int memfd = memfd_create("bb-guest-memory-probe", MFD_CLOEXEC | MFD_ALLOW_SEALING);
    if (memfd < 0 || ftruncate(memfd, Size) != 0 ||
        fcntl(memfd, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW) != 0) {
        why = "a sealed memfd";
        if (memfd >= 0) {
            close(memfd);
        }
        return false;
    }
    auto* cpu = static_cast<std::uint8_t*>(
        mmap(nullptr, Size, PROT_READ | PROT_WRITE, MAP_SHARED, memfd, 0));
    const int fd_a = MakeUdmabuf(memfd, 0, Size), fd_b = MakeUdmabuf(memfd, Half, Half);
    close(memfd);
    vk::Buffer a{}, b{};
    vk::DeviceMemory ma{}, mb{};
    vk::Result error{};
    bool ok = false;
    if (cpu == MAP_FAILED || fd_a < 0 || fd_b < 0) {
        why = "udmabuf creation";
    } else if (!ImportDmaBuf(fd_a, Size, a, ma, error) || !ImportDmaBuf(fd_b, Half, b, mb, error)) {
        static std::string text;
        text = "the import of a udmabuf failed (" + vk::to_string(error) + ")";
        why = text.c_str();
    } else {
        const bool filled = RunOnce(instance, [&](vk::CommandBuffer cmd) {
            cmd.fillBuffer(a, Half + 4096, 4096, 0x5ca1ab1eu);
        });
        std::uint32_t seen = 0;
        std::memcpy(&seen, cpu + Half + 4096, sizeof(seen));
        std::memset(cpu + 8192, 0x5a, 4096);
        const bool copied = filled && seen == 0x5ca1ab1eu && RunOnce(instance, [&](vk::CommandBuffer cmd) {
            const vk::BufferCopy region{.srcOffset = 8192, .dstOffset = 4096 * 4, .size = 4096};
            cmd.copyBuffer(a, b, region);
        });
        std::uint32_t copy = 0;
        std::memcpy(&copy, cpu + Half + 4096 * 4, sizeof(copy));
        ok = copied && copy == 0x5a5a5a5au;
        why = ok ? "" : "the GPU's and the CPU's views of a udmabuf disagree";
    }
    if (a) {
        device.destroyBuffer(a);
    }
    if (b) {
        device.destroyBuffer(b);
    }
    if (ma) {
        device.freeMemory(ma);
    } else if (fd_a >= 0) {
        close(fd_a);
    }
    if (mb) {
        device.freeMemory(mb);
    } else if (fd_b >= 0) {
        close(fd_b);
    }
    if (cpu != MAP_FAILED) {
        munmap(cpu, Size);
    }
    return ok;
#endif
}

/// Host import as the PC model uses it, on a small memfd mapped twice (the guest's view and the
/// backing view): imported through one view, bound into a sparse buffer like the arena, written by
/// the GPU and read by the CPU through the other view, and the other way round.
bool HostImportWorks(const Vulkan::Instance& instance, const char*& why) {
    why = "VK_EXT_external_memory_host unavailable";
    if (!instance.IsHostMemoryImportSupported()) {
        return false;
    }
    const vk::Device dev = instance.GetDevice();
    vk::PhysicalDeviceExternalMemoryHostPropertiesEXT host_props{};
    vk::PhysicalDeviceProperties2 props2{.pNext = &host_props};
    instance.GetPhysicalDevice().getProperties2(&props2);
    import_alignment = std::max<vk::DeviceSize>(host_props.minImportedHostPointerAlignment, 4096);
    constexpr u64 Size = 4 << 20;
    if (Size % import_alignment != 0) {
        why = "import alignment above 4 MiB";
        return false;
    }
#ifdef _WIN32
    // A pagefile-backed section mapped twice (as the runtime's guest memory is on Windows).
    void* views[2] = {};
    if (!bb_win32_shared_views(Size, &views[0], &views[1])) {
        why = "a shared memory section";
        return false;
    }
    auto* view = static_cast<std::uint32_t*>(views[0]);
    auto* guest = static_cast<std::uint32_t*>(views[1]);
#else
    const int fd = memfd_create("bb-guest-memory-probe", MFD_CLOEXEC);
    if (fd < 0 || ftruncate(fd, Size) != 0) {
        why = "memfd";
        if (fd >= 0) {
            close(fd);
        }
        return false;
    }
    auto* view = static_cast<std::uint32_t*>(
        mmap(nullptr, Size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    auto* guest = static_cast<std::uint32_t*>(
        mmap(nullptr, Size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    close(fd);
#endif
    bool ok = false;
    vk::Buffer buffer, sparse;
    vk::DeviceMemory memory;
    vk::Result error{};
    if (view == MAP_FAILED || guest == MAP_FAILED) {
        why = "mmap of the memfd";
    } else if (!ImportHost(view, Size, buffer, memory, error)) {
        static std::string text;
        text = "the import of a memfd mapping failed (" + vk::to_string(error) + ")";
        why = text.c_str();
    } else if (LayerMemory()) {
        // The layer's memory module binds the imported buffers themselves (no sparse arena): the
        // same GPU and CPU writes through the imported buffer.
        guest[1024] = 0x5ca1ab1e;
        const bool ran = RunOnce(instance, [&](vk::CommandBuffer cmd) {
            cmd.fillBuffer(buffer, 0, 4096, 0x600dcafe);
            cmd.copyBuffer(buffer, buffer, vk::BufferCopy{4096, 8192, 4});
        });
        why = "GPU work on imported memory";
        if (ran) {
            const bool gpu_to_cpu = guest[0] == 0x600dcafe && guest[1023] == 0x600dcafe;
            const bool cpu_to_gpu = guest[2048] == 0x5ca1ab1e;
            why = !gpu_to_cpu ? "GPU writes not seen by the CPU" : "CPU writes not seen by the GPU";
            ok = gpu_to_cpu && cpu_to_gpu;
        }
    } else {
        const vk::ExternalMemoryBufferCreateInfo external{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT,
        };
        const auto [sparse_result, created] = dev.createBuffer({
            .pNext = &external,
            .flags = vk::BufferCreateFlagBits::eSparseBinding |
                     vk::BufferCreateFlagBits::eSparseResidency,
            .size = Size,
            .usage = ChunkUsage,
            .sharingMode = vk::SharingMode::eExclusive,
        });
        why = "a sparse buffer for imported memory";
        // The arena also binds VRAM blocks: the buffer must take device-local memory too.
        bool vram_ok = false;
        if (sparse_result == vk::Result::eSuccess) {
            const auto reqs = dev.getBufferMemoryRequirements(created);
            for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
                vram_ok |= (reqs.memoryTypeBits & (1u << i)) &&
                           (memory_properties.memoryTypes[i].propertyFlags &
                            vk::MemoryPropertyFlagBits::eDeviceLocal);
            }
            if (!vram_ok) {
                why = "a sparse buffer for imported memory cannot take VRAM";
                dev.destroyBuffer(created);
            }
        }
        if (sparse_result == vk::Result::eSuccess && vram_ok) {
            sparse = created;
            const vk::SparseMemoryBind bind{.resourceOffset = 0, .size = Size, .memory = memory,
                                            .memoryOffset = 0};
            const vk::SparseBufferMemoryBindInfo buffer_bind{.buffer = sparse, .bindCount = 1,
                                                             .pBinds = &bind};
            const auto [fence_result, fence] = dev.createFence({});
            why = "binding imported memory into a sparse buffer";
            if (fence_result == vk::Result::eSuccess &&
                instance.GetGraphicsQueue().bindSparse(
                    vk::BindSparseInfo{.bufferBindCount = 1, .pBufferBinds = &buffer_bind},
                    fence) == vk::Result::eSuccess &&
                dev.waitForFences(fence, true, 5'000'000'000ull) == vk::Result::eSuccess) {
                // GPU writes through the sparse buffer, CPU reads through the guest view; CPU
                // writes through the guest view, the GPU copies them through the sparse buffer.
                guest[1024] = 0x5ca1ab1e;
                const bool ran = RunOnce(instance, [&](vk::CommandBuffer cmd) {
                    cmd.fillBuffer(sparse, 0, 4096, 0x600dcafe);
                    cmd.copyBuffer(sparse, buffer, vk::BufferCopy{4096, 8192, 4});
                });
                why = "GPU work on imported memory";
                if (ran) {
                    const bool gpu_to_cpu = guest[0] == 0x600dcafe && guest[1023] == 0x600dcafe;
                    const bool cpu_to_gpu = guest[2048] == 0x5ca1ab1e;
                    why = !gpu_to_cpu ? "GPU writes not seen by the CPU"
                                      : "CPU writes not seen by the GPU";
                    ok = gpu_to_cpu && cpu_to_gpu;
                }
            }
            if (fence_result == vk::Result::eSuccess) {
                dev.destroyFence(fence);
            }
        }
    }
    if (sparse) {
        dev.destroyBuffer(sparse);
    }
    if (buffer) {
        dev.destroyBuffer(buffer);
    }
    if (memory) {
        dev.freeMemory(memory);
    }
    if (view != MAP_FAILED) {
        munmap(view, Size);
    }
    if (guest != MAP_FAILED) {
        munmap(guest, Size);
    }
    return ok;
}

/// Host import of private memory (VirtualAlloc'd on Windows, anonymous on Linux), as the runtime's
/// private regions give it: the GPU's and the CPU's writes through the imported buffer.
bool PrivateImportWorks(const Vulkan::Instance& instance, const char*& why) {
    why = "VK_EXT_external_memory_host unavailable";
    if (!instance.IsHostMemoryImportSupported() || !LayerMemory()) {
        if (!LayerMemory()) {
            why = "private regions need the layer's memory module";
        }
        return false;
    }
    vk::PhysicalDeviceExternalMemoryHostPropertiesEXT host_props{};
    vk::PhysicalDeviceProperties2 props2{.pNext = &host_props};
    instance.GetPhysicalDevice().getProperties2(&props2);
    import_alignment = std::max<vk::DeviceSize>(host_props.minImportedHostPointerAlignment, 4096);
    constexpr u64 Size = 4 << 20;
    auto* guest = static_cast<std::uint32_t*>(
        mmap(nullptr, Size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (guest == MAP_FAILED) {
        why = "private memory";
        return false;
    }
    bool ok = false;
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::Result error{};
    if (!ImportHost(guest, Size, buffer, memory, error)) {
        static std::string text;
        text = "the import of private memory failed (" + vk::to_string(error) + ")";
        why = text.c_str();
    } else {
        guest[1024] = 0x5ca1ab1e;
        const bool ran = RunOnce(instance, [&](vk::CommandBuffer cmd) {
            cmd.fillBuffer(buffer, 0, 4096, 0x600dcafe);
            cmd.copyBuffer(buffer, buffer, vk::BufferCopy{4096, 8192, 4});
        });
        why = "GPU work on imported private memory";
        if (ran) {
            const bool gpu_to_cpu = guest[0] == 0x600dcafe && guest[1023] == 0x600dcafe;
            const bool cpu_to_gpu = guest[2048] == 0x5ca1ab1e;
            why = !gpu_to_cpu ? "GPU writes not seen by the CPU" : "CPU writes not seen by the GPU";
            ok = gpu_to_cpu && cpu_to_gpu;
        }
    }
    if (buffer) {
        device.destroyBuffer(buffer);
    }
    if (memory) {
        device.freeMemory(memory);
    }
    munmap(guest, Size);
    return ok;
}

/// CPU read speed through a dma-buf mapping, MB/s (the game's code runs on this memory: an
/// uncached mapping would slow it to a crawl).
double MappedReadSpeed(const volatile std::uint64_t* p, u64 size) {
    std::uint64_t sum = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int pass = 0; pass < 4; ++pass) {
        for (u64 i = 0; i < size / 8; i += 8) {
            sum += p[i];
        }
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    static volatile std::uint64_t sink;
    sink = sum;
    return s > 0 ? 4.0 * size / s / 1e6 : 1e9;
}
/// dma-buf chunks: exported cached system memory the runtime maps at the game's addresses.
bool DmaBufWorks(const Vulkan::Instance& instance) {
#ifdef _WIN32
    (void)instance;
    std::printf("Guest memory: no dma-buf on Windows\n");
    return false;
#else
    {
        if (!instance.IsGuestMemoryExportSupported()) {
            std::printf("Guest memory: the driver cannot export system memory as a dma-buf\n");
            return false;
        }
        const vk::Device dev = instance.GetDevice();
        const vk::PhysicalDeviceMemoryProperties props = instance.GetPhysicalDevice().getMemoryProperties();
        constexpr u64 Size = 2 << 20, Offset = 1 << 20;
        const vk::ExternalMemoryBufferCreateInfo external{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
        };
        const auto [buffer_result, buffer] = dev.createBuffer({
            .pNext = &external,
            .size = Size,
            .usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
            .sharingMode = vk::SharingMode::eExclusive,
        });
        if (buffer_result != vk::Result::eSuccess) {
            return false;
        }
        const auto requirements = dev.getBufferMemoryRequirements(buffer);
        std::uint32_t type = ~0u;
        const auto want = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached;
        for (std::uint32_t i = 0; i < props.memoryTypeCount && type == ~0u; ++i) {
            const auto flags = props.memoryTypes[i].propertyFlags;
            if ((requirements.memoryTypeBits & (1u << i)) && (flags & want) == want &&
                !(flags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
                type = i;
            }
        }
        bool ok = false;
        vk::DeviceMemory memory{};
        int fd = -1;
        const vk::ExportMemoryAllocateInfo export_info{
            .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
        };
        if (type != ~0u) {
            const auto [memory_result, allocated] = dev.allocateMemory(
                {.pNext = &export_info, .allocationSize = requirements.size, .memoryTypeIndex = type});
            if (memory_result == vk::Result::eSuccess) {
                memory = allocated;
                const auto [fd_result, exported] = dev.getMemoryFdKHR(
                    {.memory = memory, .handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT});
                fd = fd_result == vk::Result::eSuccess ? exported : -1;
            }
        }
        if (fd >= 0) {
            // The whole buffer and one page inside it, writing through one, reading through the other.
            void* whole = mmap(nullptr, Size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            void* page = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, Offset);
            // BB_GUEST_WHOLE_CHUNKS=1: as if the offset mapping failed (tests the NVIDIA path).
            if (bb_env_on("BB_GUEST_WHOLE_CHUNKS") && page != MAP_FAILED) {
                munmap(page, 4096);
                page = MAP_FAILED;
            }
            if (whole != MAP_FAILED && page != MAP_FAILED) {
                *static_cast<volatile std::uint32_t*>(page) = 0x5ca1ab1e;
                ok = static_cast<volatile std::uint32_t*>(whole)[Offset / 4] == 0x5ca1ab1e;
            } else if (whole != MAP_FAILED) {
                // Offset 0 only (NVIDIA): a second mapping of the start must alias the first.
                void* again = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
                if (again != MAP_FAILED) {
                    *static_cast<volatile std::uint32_t*>(again) = 0x5ca1ab1e;
                    ok = whole_only = static_cast<volatile std::uint32_t*>(whole)[0] == 0x5ca1ab1e;
                    munmap(again, 4096);
                }
            }
            // The game's code works on this memory: an uncached CPU mapping would crawl.
            if (ok) {
                const double speed =
                    MappedReadSpeed(static_cast<const volatile std::uint64_t*>(whole), Size);
                std::printf("Guest memory: CPU reads through the driver's dma-buf: %.0f MB/s\n",
                            speed);
                if (speed < 1000.0) {
                    std::printf("Guest memory: CPU reads of the driver's dma-buf run at %.0f MB/s "
                                "(uncached?): not used\n",
                                speed);
                    ok = false;
                    whole_only = false;
                }
            }
            if (whole != MAP_FAILED) {
                munmap(whole, Size);
            }
            if (page != MAP_FAILED) {
                munmap(page, 4096);
            }
            close(fd);
        }
        if (memory) {
            dev.freeMemory(memory);
        }
        dev.destroyBuffer(buffer);
        if (!ok) {
            std::printf("Guest memory: the driver's dma-buf cannot be mapped (%s)\n",
                        fd >= 0 ? "mmap failed" : "no exportable cached system memory");
        } else if (whole_only) {
            std::printf("Guest memory: the driver's dma-buf maps at offset 0 only: one chunk per "
                        "direct memory allocation\n");
        }
        return ok;
    }
#endif
}
} // namespace BbGuestMemory::detail
