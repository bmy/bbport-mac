// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport guest memory: the state and functions its parts share (bbport_guest_memory.cpp: the
// choice of model, Install and the lookups; _chunks.cpp: chunks made by Vulkan or imported whole;
// _lazy.cpp: lazy imports of the memfd (udmabuf / host import); _probes.cpp: the startup checks of
// what the driver can do). Only those files include it.
#pragma once

#include "bbport_guest_memory.h"
#include "shim/bb_env.h"

#include <array>
#include <chrono>
#include <string>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <tuple>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#ifndef _WIN32
#include <execinfo.h>
#endif
#include <sys/stat.h>
#include <unistd.h>
#ifndef _WIN32
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#endif
#include "vulkan/instance.h"
#include "bblayer_gpu_memory.h"
#include "buffers/memory_model.h"
#include "bbport_toggles.h"
#include "shader/resource.h"

extern "C" void runtime_memory_set_guest_chunk_allocator(int (*alloc)(uint64_t phys,
                                                                      uint64_t size));
extern "C" void runtime_memory_set_guest_chunk_whole(int whole);
extern "C" void runtime_memory_set_guest_chunk_private(int (*import)(uint64_t phys, uint64_t size,
                                                                     void* pointer));
extern "C" void* runtime_memory_backing_pointer(uint64_t phys);
extern "C" int runtime_memory_pool_fd(void);
extern "C" void runtime_memory_note_pinned(uint64_t phys, uint64_t size);
extern "C" void runtime_memory_set_guest_chunk_free(void (*free_chunk)(uint64_t phys, uint64_t size));
extern "C" void runtime_memory_discard_direct(uint64_t phys, uint64_t size);
extern "C" int runtime_memory_direct_phys(uintptr_t address, uint64_t* phys, uintptr_t* end);
extern "C" void runtime_memory_set_guest_lazy(int lazy);


namespace BbGuestMemory::detail {
using u64 = std::uint64_t;
inline constexpr std::size_t MaxChunks = 4096;

inline vk::Device device;
inline vk::PhysicalDeviceMemoryProperties memory_properties;
inline std::mutex mutex;
inline std::array<Chunk*, MaxChunks> chunks{}; // sorted by phys
inline std::size_t chunk_count = 0;
/// Chunk::index of the next chunk, whichever way it is made (the buffer cache keys its chunk
/// buffers by it: two chunks with one index bound one through the other).
inline std::uint32_t next_index = 0;
inline bool whole_only = false; // the driver's dma-buf maps at offset 0 only: a chunk per allocation
enum class Mode { DmaBuf, HostImport };
inline Mode mode = Mode::DmaBuf;
/// Host import of private memory, one view per direct allocation (runtime private regions): for
/// drivers that import VirtualAlloc'd memory but not views of sections (Windows).
inline bool private_regions = false;
// BB_GUEST_MEMORY=hostsim (Linux, diagnostics): host memory import as on NVIDIA, the memfd's pages
// there given to the GPU through udmabuf instead (RADV refuses host imports of memfd mappings;
// udmabuf's size_limit_mb must hold the largest direct allocation, 2287 MiB).
inline bool host_sim = false;
inline std::atomic<u64> chunk_bytes{0};
// BB_GUEST_UNCACHED=1 (Instance::IsGuestUncached): only memory types the GPU reads past its caches.
inline bool uncached = false;


/// Chunks a heap gave back (runtime_memory_decommit; BB_HEAP_GROW): out of the lookups at once,
/// their buffers and memory freed 3 s later (RetiredChunks, DestroyRetiredChunks) when nothing
/// the GPU still runs can use them.
struct RetiredChunk {
    Chunk* chunk;
    std::chrono::steady_clock::time_point at;
};
inline std::vector<RetiredChunk> retired_chunks;
inline std::vector<Chunk*> dropped_chunks;
inline std::atomic<std::size_t> retired_count{0};

/// Host memory import (VK_EXT_external_memory_host): the chunk stays in the runtime's memfd, the
/// GPU imports its backing view. CPU access is ordinary cached memory; the GPU reads and writes it
/// over the bus, as a PC game's upload heap. For drivers whose dma-buf does not fit (NVIDIA: maps
/// at offset 0 only, CPU access through it may be uncached).
inline vk::DeviceSize import_alignment = 4096;

inline constexpr vk::BufferUsageFlags ChunkUsage =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

// --- Lazy guest memory (BB_GUEST_LAZY, default on with the layer's memory module) -------------
// The game's direct memory stays in the runtime's memfd: pages the game never touches cost no
// RAM, as with the old memory model. The GPU gets a range when it first uses it: the memfd's
// pages there imported (udmabuf on Linux, a dma-buf over the memfd; host memory import where
// that is the GPU's way, NVIDIA). Imports are made per granule, and for a range a buffer binds
// whole across granules (WholeLimit) the union again: imports of the same pages may overlap. With
// all of the game's memory GPU-visible from the start, 2.2 GiB of never written pages were
// resident in a heavy area (all of a 5 GiB pool; 8 GiB with the scene above 1080p).
/// BB_IMPORT_GRANULE_MB (power of two, 2..64; default 32): smaller ones leave fewer never written
/// pages imported (allocated) next to used ones, but make more imports (per-submission cost).
inline const u64 Granule = [] {
    const char* env = bb_env("BB_IMPORT_GRANULE_MB");
    const u64 mb = env && *env ? std::strtoull(env, nullptr, 10) : 32;
    return (mb >= 2 && mb <= 64 && (mb & (mb - 1)) == 0 ? mb : 32) << 20;
}();
inline bool lazy = false;
inline bool lazy_udmabuf = false; // else host memory import (NVIDIA)
inline int udmabuf_device = -1;
inline u64 import_limit = u64{64} << 20; // udmabuf.size_limit_mb; host imports: 256 MiB
inline std::mutex populate_mutex;
inline std::vector<std::pair<u64, u64>> failed_imports;
inline std::atomic<u64> imported_bytes{0}, imports_made{0};
/// Bytes of the imports that may hold pages of others (not a granule of their own: OverlapFits).
inline std::atomic<u64> overlap_bytes{0};
/// This thread resolves a read-only paged binding (SetPagedRead, Populate).
inline thread_local bool paged_read = false;
/// This thread resolves a binding the GPU writes (WrittenBinding): no overlap budget for it.
inline thread_local bool written_binding = false;

// Functions the parts share (defined in the file named before each group).
// bbport_guest_memory.cpp
bool TypeAllowed(vk::MemoryPropertyFlags flags);
bool ChunkInVram(u64 phys, u64 size);
std::uint32_t FindType(std::uint32_t bits, bool in_vram);
vk::BufferUsageFlags ChunkBufferUsage();
void RegisterChunk(Chunk* chunk);
// bbport_guest_memory_chunks.cpp
int AllocChunk(u64 phys, u64 size);
void FreeChunk(u64 phys, u64 size);
std::uint32_t FindHostType(std::uint32_t bits);
bool ImportHost(void* pointer, u64 size, vk::Buffer& buffer, vk::DeviceMemory& memory,
                vk::Result& error);
int AllocImported(u64 phys, u64 size);
int ImportPrivate(u64 phys, u64 size, void* pointer);
bool RegisterImported(u64 phys, u64 size, void* pointer);
// bbport_guest_memory_lazy.cpp
int MakeUdmabuf(int memfd, u64 offset, u64 size);
bool ImportDmaBuf(int fd, u64 size, vk::Buffer& buffer, vk::DeviceMemory& memory,
                  vk::Result& error);
bool ImportRange(u64 start, u64 size);
bool CoveredLocked(u64 start, u64 end);
u64 PoolBytes();
bool OverlapFits(u64 size);
bool ImportOnce(u64 start, u64 end, bool overlapping = false);
u64 HoleEnd(u64 phys);
bool CrossesImports(std::uint64_t va, std::uint64_t size);
bool Populate(u64 phys, u64 want);
// bbport_guest_memory_probes.cpp
bool UdmabufWorks(const Vulkan::Instance& instance, const char*& why);
bool HostImportWorks(const Vulkan::Instance& instance, const char*& why);
bool PrivateImportWorks(const Vulkan::Instance& instance, const char*& why);
bool DmaBufWorks(const Vulkan::Instance& instance);

} // namespace BbGuestMemory::detail

namespace BbGuestMemory {
using namespace detail;
} // namespace BbGuestMemory
