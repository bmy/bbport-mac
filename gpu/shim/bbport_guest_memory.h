// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: guest direct memory in GPU-visible system memory (the PC memory model).
//
// The runtime asks for direct memory in chunks (runtime_memory.c, CHUNK); each chunk is a
// Vulkan allocation of cached system memory, exported as a dma-buf that the runtime maps at the
// game's addresses and in its backing view. The GPU can then read the game's data in place: no
// CPU copy into staging buffers before an upload. A chunk that cannot be made stays in the
// runtime's memfd, as before.
#pragma once

#include <cstdint>
#include <vector>
#include "vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace BbGuestMemory {
struct Chunk {
    std::uint64_t phys = 0; ///< first byte of direct memory it holds
    std::uint64_t size = 0;
    vk::Buffer buffer;      ///< the whole chunk, for transfers
    vk::DeviceMemory memory;
    std::uint32_t index = 0; ///< creation order (stable key)
};

/// Whether the runtime can use such chunks: dma-buf export and mmap of the dma-buf. Checked once
/// with a small chunk. A driver that maps at page offsets gets 256 MiB chunks (the game's mappings
/// start inside them); NVIDIA's dma-buf maps at offset 0 only (the grid made every mapping after
/// the first fail and the game panic at startup), so there each direct allocation is its own chunk:
/// the game maps its allocations whole, at offset 0.
bool Usable(const Vulkan::Instance& instance);
/// Whether the GPU side's way to the game's memory fits this GPU: the layer's memory module on
/// any GPU, the sparse arena (BB_LAYER_MEMORY=0) on AMD only. Asked first: it picks the module.
bool MemoryModuleFits(const Vulkan::Instance& instance);
/// Whether the chunks are imported host memory (NVIDIA, Windows). Valid after
/// Usable(). The arena then keeps the first binding of every block (BufferCache FixedArena).
bool HostImported();
/// bbport BB_LAYER_MEMORY=1: the layer's memory module (gpu/layer/bblayer_gpu_memory.h) gives the
/// GPU the game's memory through the chunk buffers instead of a sparse arena. Chunks are then one
/// per direct memory allocation (a mapping lies whole in one chunk), their buffers take every use
/// (vertex, index, indirect, texel, device address) and are registered with the module.
bool LayerMemory();
/// The external memory handle type of the chunks (dma-buf or imported host memory): buffers bound
/// to them (the arena, chunk buffers) are created with it. Valid after Usable().
vk::ExternalMemoryHandleTypeFlagBits HandleType();
/// Hands the runtime direct memory chunks from now on (when enabled and supported).
void Install(const Vulkan::Instance& instance);
/// The chunk holding direct memory address `phys`, or null.
const Chunk* Find(std::uint64_t phys);
/// Lazy guest memory: imports the longest range an import may be from phys (up to size), for a
/// binding that must be one buffer range; false when nothing new was imported.
bool Widen(std::uint64_t phys, std::uint64_t size);
/// Lazy guest memory: this thread resolves a read-only paged binding (whole-memory buffers):
/// never written granules are not imported for it (its page table leaves them to the trash).
void SetPagedRead(bool on);

/// Chunk indices (Chunk::index) given back at least 3 s ago (a heap's piece, BB_HEAP_GROW): the
/// buffer cache drops its buffers over them, then calls DestroyRetiredChunks.
std::vector<std::uint32_t> RetiredChunks();
void DestroyRetiredChunks();

/// Lazy guest memory: the game's memory imported where the GPU uses it.
bool Lazy();
/// While a binding the GPU writes resolves (this thread): imports for it are not refused by the
/// overlap budget (BB_IMPORT_OVERLAP_MB).
struct WrittenBinding {
    explicit WrittenBinding(bool written);
    ~WrittenBinding();
    WrittenBinding(const WrittenBinding&) = delete;
    WrittenBinding& operator=(const WrittenBinding&) = delete;
    bool previous;
};
} // namespace BbGuestMemory
