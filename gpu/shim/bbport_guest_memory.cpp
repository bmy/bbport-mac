// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport guest memory: the model chosen at startup, Install, the chunk lookups (see
// bbport_guest_memory_internal.h for the parts).
#include "bbport_guest_memory_internal.h"
#include "shim/bb_env.h"

namespace BbGuestMemory::detail {
/// A type the guest memory may use: AMD's device-uncached types exactly when asked for.
bool TypeAllowed(vk::MemoryPropertyFlags flags) {
    return bool(flags & vk::MemoryPropertyFlagBits::eDeviceUncachedAMD) == uncached;
}

/// bbport BB_GUEST_VRAM (experiment): direct memory in VRAM the CPU maps through the PCI BAR
/// (Resizable BAR), as GDDR is the PS4's one memory: no VRAM copies, the GPU reads it at VRAM
/// speed. "1": all of it (the CPU reads much of it: 16 FPS); "graphics": the ranges of
/// BB_GUEST_VRAM_RANGES (physical lo-hi in hex, comma-separated), by default Bloodborne's graphics
/// heaps GraphicsPrivate and GraphicsPrivateB, which the GPU fills (its copy shader) and the CPU
/// does not touch (BB_READ_SAMPLE: 99.3-100 % of 64 KiB windows untouched for 200 ms).
struct VramRanges {
    bool all = false;
    std::vector<std::pair<u64, u64>> ranges;
};
const VramRanges& GuestVram() {
    static const VramRanges vram = [] {
        VramRanges v;
        const char* env = bb_env("BB_GUEST_VRAM");
        if (!env || !*env || env[0] == '0') {
            return v;
        }
        if (env[0] == '1') {
            v.all = true;
            return v;
        }
        const char* list = bb_env("BB_GUEST_VRAM_RANGES");
        std::string text = list && *list ? list : "42c00000-d1b00000,d1b00000-109100000";
        for (size_t at = 0; at < text.size();) {
            const size_t comma = std::min(text.find(',', at), text.size());
            const std::string item = text.substr(at, comma - at);
            const size_t dash = item.find('-');
            if (dash != std::string::npos) {
                v.ranges.emplace_back(std::strtoull(item.c_str(), nullptr, 16),
                                      std::strtoull(item.c_str() + dash + 1, nullptr, 16));
            }
            at = comma + 1;
        }
        return v;
    }();
    return vram;
}

/// Whether the direct memory chunk [phys, phys + size) goes to VRAM (BB_GUEST_VRAM).
bool ChunkInVram(u64 phys, u64 size) {
    const auto& vram = GuestVram();
    if (vram.all) {
        return true;
    }
    for (const auto& [lo, hi] : vram.ranges) {
        if (phys >= lo && phys + size <= hi) {
            return true;
        }
    }
    return false;
}

std::uint32_t FindType(std::uint32_t bits, bool in_vram) {
    if (in_vram) {
        const auto want = vk::MemoryPropertyFlagBits::eHostVisible |
                          vk::MemoryPropertyFlagBits::eDeviceLocal;
        for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
            if ((bits & (1u << i)) && (memory_properties.memoryTypes[i].propertyFlags & want) == want) {
                return i;
            }
        }
    }
    const auto want = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached;
    for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const auto flags = memory_properties.memoryTypes[i].propertyFlags;
        if ((bits & (1u << i)) && (flags & want) == want &&
            !(flags & vk::MemoryPropertyFlagBits::eDeviceLocal) && TypeAllowed(flags)) {
            return i;
        }
    }
    return ~0u;
}

/// Uses of a chunk's buffer. With the layer's memory module the chunk buffers are what the GPU
/// binds (no sparse arena): every buffer use, texel buffers and a device address too.
vk::BufferUsageFlags ChunkBufferUsage() {
    vk::BufferUsageFlags usage =
        vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eUniformBuffer |
        vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer |
        vk::BufferUsageFlagBits::eIndirectBuffer;
    if (LayerMemory()) {
        usage |= vk::BufferUsageFlagBits::eUniformTexelBuffer |
                 vk::BufferUsageFlagBits::eStorageTexelBuffer |
                 vk::BufferUsageFlagBits::eShaderDeviceAddress;
    }
    return usage;
}

/// A new chunk becomes a source of the layer's memory module.
void RegisterChunk(Chunk* chunk) {
    if (!LayerMemory()) {
        return;
    }
    const vk::DeviceAddress address = device.getBufferAddress({.buffer = chunk->buffer});
    BbLayer::GpuMemory::Get().AddGuestSource(chunk->phys, chunk->size, chunk->buffer, address,
                                             chunk);
}
} // namespace BbGuestMemory::detail

namespace BbGuestMemory {

namespace {
/// The GPU's choice of LayerMemory without BB_LAYER_MEMORY: -1 not known yet, 0 or 1.
std::atomic<int> layer_default{-1};
} // namespace

bool MemoryModuleFits(const Vulkan::Instance& instance) {
    static const bool ok = [&] {
        // The layer's memory module on every GPU, AMD's too (no sparse binding of the game's
        // memory, see LayerMemory); BB_LAYER_MEMORY=0: the sparse arena, AMD only. On an RX 7800
        // XT after 0.51 the module is ahead of the arena: route in Yharnam 4.30 against 4.60 ms a
        // frame (p99 5.8-6.0 against 6.6), Hunter's Dream with the camera turning 4.8-5.0 against
        // 5.2-5.3, Oedon Chapel 234 against 213 FPS. And it alone is right without the game's
        // profile: with the arena a game without one (the US release before 0.51) had the
        // allocator's bookkeeping overwritten by a readback of a stale VRAM copy (issue #119).
        // Not the arena on NVIDIA: its host memory import passes the startup checks, but each
        // rebinding of the game's memory (vkQueueBindSparse, blocks moved to and from VRAM) held
        // the GPU for 10 s and more, the desktop frozen with it (GTX 1660 Ti, driver 615.71). The
        // Linux driver's sparse binding has slowed down since 555: the time grows with the pages
        // already bound (NVIDIA developer forum, "Sparse texture binding is painfully slow").
        constexpr std::uint32_t AmdVendor = 0x1002;
        const std::uint32_t vendor = instance.GetVendorID();
        layer_default.store(1, std::memory_order_relaxed);
        if (LayerMemory()) {
            std::printf("Guest memory: the layer's memory module on this GPU (vendor 0x%04x): no "
                        "sparse binding of the game's memory\n",
                        vendor);
            return true;
        }
        if (vendor == AmdVendor) {
            return true;
        }
        std::printf("Guest memory: the sparse arena (BB_LAYER_MEMORY=0) is for AMD GPUs; this GPU "
                    "is vendor 0x%04x\n",
                    vendor);
        return false;
    }();
    return ok;
}

bool HostImported() {
    return mode == Mode::HostImport;
}

bool LayerMemory() {
    // BB_LAYER_MEMORY=1/0 chooses; else MemoryModuleFits (which asks first) sets the module for every GPU
    // (BB_LAYER_MEMORY=0: the sparse arena; NVIDIA's sparse binding stalls for seconds).
    static const bool on = [] {
        const char* env = bb_env("BB_LAYER_MEMORY");
        if (env && *env) {
            return env[0] == '1';
        }
        return layer_default.load(std::memory_order_relaxed) == 1;
    }();
    return on;
}


bool Usable(const Vulkan::Instance& instance) {
    static const bool usable = [&] {
        device = instance.GetDevice();
        memory_properties = instance.GetPhysicalDevice().getMemoryProperties();
        uncached = instance.IsGuestUncached();
        if (uncached) {
            std::printf("Guest memory: the GPU reads it uncached (BB_GUEST_UNCACHED=1, "
                        "VK_AMD_device_coherent_memory): a diagnostic, as on NVIDIA's imported "
                        "host memory\n");
        }
        // BB_GUEST_MEMORY=dmabuf|host chooses; else AMD uses dma-buf chunks (tested the most) and
        // other GPUs host memory import first (NVIDIA: its dma-buf maps at offset 0 only, and
        // CPU access through it may be slow).
        const char* forced = bb_env("BB_GUEST_MEMORY");
        const bool force_host = forced && std::string_view{forced} == "host";
        const bool force_dmabuf = forced && std::string_view{forced} == "dmabuf";
        const bool force_private = forced && std::string_view{forced} == "private";
#ifndef _WIN32
        // Lazy guest memory with the layer's memory module (BB_GUEST_LAZY=1; off by default since
        // 2026-10-23: buffers across its imports kept breaking — explosions, black surfaces, GPU
        // faults; the game's heaps are to grow on demand instead): udmabuf
        // imports where the GPU takes them (AMD; BB_GUEST_MEMORY=udmabuf on any GPU), host
        // memory imports elsewhere (NVIDIA).
        if (LayerMemory() && (bb_env_on("BB_GUEST_LAZY")) && !force_dmabuf &&
            !force_private) {
            const bool udmabuf_first =
                (forced && std::string_view{forced} == "udmabuf") ||
                (!force_host && instance.GetVendorID() == 0x1002);
            const char* why = "";
            if (udmabuf_first && UdmabufWorks(instance, why)) {
                mode = Mode::DmaBuf;
                lazy = lazy_udmabuf = true;
                std::printf("Guest memory: lazy, the GPU imports what it uses (udmabuf over the "
                            "game's memory, up to %llu MiB each)\n",
                            (unsigned long long)(import_limit >> 20));
                return true;
            }
            if (udmabuf_first) {
                std::printf("Guest memory: udmabuf does not work here: %s\n", why);
            }
            if (HostImportWorks(instance, why)) {
                mode = Mode::HostImport;
                lazy = true;
                import_limit = u64{256} << 20;
                std::printf("Guest memory: lazy, the GPU imports what it uses (host memory import, "
                            "alignment %llu)\n",
                            (unsigned long long)import_alignment);
                return true;
            }
            std::printf("Guest memory: host memory import does not work here: %s\n", why);
        }
#endif
        // BB_GUEST_MEMORY=private, or host import of section/memfd views failing (Windows drivers
        // import VirtualAlloc'd memory only, as D3D12's OpenExistingHeapFromAddress): private
        // regions, one view of each direct allocation.
        const auto try_private = [&] {
            const char* private_why = "";
            if (PrivateImportWorks(instance, private_why)) {
                mode = Mode::HostImport;
                private_regions = true;
                std::printf("Guest memory: private memory imported by the GPU, one view per direct "
                            "allocation (VK_EXT_external_memory_host, alignment %llu)\n",
                            (unsigned long long)import_alignment);
                return true;
            }
            std::printf("Guest memory: private memory import does not work here: %s\n", private_why);
            return false;
        };
        if (force_private) {
            return try_private();
        }
#ifndef _WIN32
        if (forced && std::string_view{forced} == "hostsim") {
            const char* why_not = "";
            if (!UdmabufWorks(instance, why_not)) {
                std::printf("Guest memory: hostsim: udmabuf does not work here: %s\n", why_not);
                return false;
            }
            mode = Mode::HostImport;
            host_sim = true;
            std::printf("Guest memory: hostsim: NVIDIA's host memory import imitated (udmabuf over "
                        "the memfd)\n");
            return true;
        }
#endif
#ifdef _WIN32
        const bool host_first = !force_dmabuf; // no dma-buf on Windows
#else
        const bool host_first =
            force_host || (!force_dmabuf && instance.GetVendorID() != 0x1002);
#endif
        const char* why = "";
        if (host_first && HostImportWorks(instance, why)) {
            mode = Mode::HostImport;
            std::printf("Guest memory: host memory imported by the GPU (VK_EXT_external_memory_host, "
                        "alignment %llu)\n",
                        (unsigned long long)import_alignment);
            return true;
        }
        if (host_first) {
            std::printf("Guest memory: host memory import does not work here: %s\n", why);
            if (try_private()) {
                return true;
            }
            if (force_host) {
                return false;
            }
        }
        mode = Mode::DmaBuf;
        return DmaBufWorks(instance);
    }();
    return usable;
}

vk::ExternalMemoryHandleTypeFlagBits HandleType() {
    return mode == Mode::HostImport ? vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT
                                    : vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
}

void Install(const Vulkan::Instance& instance) {
    // The GPU uses the game's memory in place: its direct memory must come from Vulkan.
    if (!MemoryModuleFits(instance) || !Usable(instance)) {
        VideoCore::GuestMemoryUnusable("its direct memory cannot be given to Vulkan (see the "
                                       "\"Guest memory:\" lines above)");
    }
    if (lazy) {
        // The game's memory stays in the runtime's memfd; the layer's module asks for imports.
        // The port's own direct memory past the game's (the render resource cache: bound in
        // ~64 MiB windows at any offset, more than an import may hold) is one chunk as before.
        runtime_memory_set_guest_lazy(1);
        runtime_memory_set_guest_chunk_whole(1);
        runtime_memory_set_guest_chunk_allocator(lazy_udmabuf ? &AllocChunk : &AllocImported);
        BbLayer::GpuMemory::Get().SetPopulate(&Populate);
        BbLayer::GpuMemory::Get().SetHoleEnd(&HoleEnd);
        if (!bb_env_off("BB_PAGED_CROSSING")) {
            Shader::paged_range_hook = &CrossesImports;
        }
        std::printf("Guest memory: the game's memory stays in its memfd until the GPU uses it%s\n",
                    Shader::paged_range_hook ? " (shader buffers across imports: page table)" : "");
        return;
    }
    const bool host = mode == Mode::HostImport;
    // A heap's pieces given back (BB_HEAP_GROW): their chunk freed, whichever way it was made.
    runtime_memory_set_guest_chunk_free(&FreeChunk);
    if (private_regions) {
        runtime_memory_set_guest_chunk_private(&ImportPrivate);
        std::printf("Guest memory: direct memory as private regions imported by the GPU\n");
        return;
    }
    // One chunk per direct allocation where the dma-buf maps at offset 0 only, and with the
    // layer's memory module (a mapping then lies whole in one chunk buffer).
    runtime_memory_set_guest_chunk_whole(LayerMemory() || (!host && whole_only) ? 1 : 0);
    runtime_memory_set_guest_chunk_allocator(host ? &AllocImported : &AllocChunk);

    std::printf("Guest memory: direct memory chunks come from Vulkan\n");
}

std::vector<std::uint32_t> RetiredChunks() {
    std::vector<std::uint32_t> indices;
    if (retired_count.load(std::memory_order_acquire) == 0) {
        return indices;
    }
    std::scoped_lock lk{mutex};
    const auto now = std::chrono::steady_clock::now();
    for (auto it = retired_chunks.begin(); it != retired_chunks.end();) {
        if (now - it->at >= std::chrono::seconds(3)) {
            indices.push_back(it->chunk->index);
            dropped_chunks.push_back(it->chunk);
            it = retired_chunks.erase(it);
        } else {
            ++it;
        }
    }
    retired_count.store(retired_chunks.size(), std::memory_order_release);
    return indices;
}

void DestroyRetiredChunks() {
    std::vector<Chunk*> chunks_now;
    {
        std::scoped_lock lk{mutex};
        chunks_now.swap(dropped_chunks);
    }
    for (Chunk* chunk : chunks_now) {
        device.destroyBuffer(chunk->buffer);
        device.freeMemory(chunk->memory);
        // Windows: the pages the import held go back to the system now that it let them go.
        runtime_memory_discard_direct(chunk->phys, chunk->size);
        delete chunk;
    }
}

WrittenBinding::WrittenBinding(bool written) : previous{written_binding} {
    written_binding = written;
}

WrittenBinding::~WrittenBinding() {
    written_binding = previous;
}

bool Lazy() {
    return lazy;
}

void SetPagedRead(bool on) {
    paged_read = on;
}

bool Widen(std::uint64_t phys, std::uint64_t size) {
    if (!lazy || size == 0) {
        return false;
    }
    // From its page: a binding of the import limit's size fits whole.
    constexpr u64 Page = 4096;
    const u64 start = phys / Page * Page;
    const u64 end =
        std::min({start + import_limit, (phys + size + Page - 1) / Page * Page, PoolBytes()});
    if (start >= end) {
        return false;
    }
    std::scoped_lock lk{populate_mutex};
    {
        std::scoped_lock chunks_lk{mutex};
        if (CoveredLocked(start, end)) {
            return false;
        }
    }
    return ImportOnce(start, end, true);
}

const Chunk* Find(std::uint64_t phys) {
    const auto find = [phys]() -> const Chunk* {
        std::scoped_lock lk{mutex};
        // The last chunk starting at or below phys; with lazy imports (which may overlap) the
        // one holding it that reaches furthest.
        std::size_t lo = 0, hi = chunk_count;
        while (lo < hi) {
            const std::size_t mid = (lo + hi) / 2;
            if (chunks[mid]->phys <= phys) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        const Chunk* best = nullptr;
        for (std::size_t i = lo; i > 0; --i) {
            const Chunk* c = chunks[i - 1];
            if (phys < c->phys + c->size && (!best || c->phys + c->size > best->phys + best->size)) {
                best = c;
            }
            if (!lazy || c->phys + std::max<u64>(import_limit, Granule) <= phys) {
                break;
            }
        }
        return best;
    };
    const Chunk* chunk = find();
    if (!chunk && lazy && Populate(phys, 1)) {
        chunk = find();
    }
    return chunk;
}
} // namespace BbGuestMemory
