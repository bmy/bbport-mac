// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport guest memory: lazy imports of the runtime's memfd (BB_GUEST_LAZY).
#include "bbport_guest_memory_internal.h"
#include "shim/bb_env.h"

namespace BbGuestMemory::detail {

/// A dma-buf of [offset, offset + size) of the memfd (udmabuf), or -1.
int MakeUdmabuf(int memfd, u64 offset, u64 size) {
#ifdef _WIN32
    (void)memfd, (void)offset, (void)size;
    return -1;
#else
    if (udmabuf_device < 0) {
        udmabuf_device = open("/dev/udmabuf", O_RDWR | O_CLOEXEC);
        if (udmabuf_device < 0) {
            return -1;
        }
    }
    udmabuf_create create{};
    create.memfd = static_cast<std::uint32_t>(memfd);
    create.flags = UDMABUF_FLAGS_CLOEXEC;
    create.offset = offset;
    create.size = size;
    return ioctl(udmabuf_device, UDMABUF_CREATE, &create);
#endif
}

/// Imports dma-buf `fd` (owned by Vulkan on success) as memory with a buffer of `size` over it.
bool ImportDmaBuf(int fd, u64 size, vk::Buffer& buffer, vk::DeviceMemory& memory,
                  vk::Result& error) {
    const auto [props_result, props] =
        device.getMemoryFdPropertiesKHR(vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT, fd);
    if (props_result != vk::Result::eSuccess) {
        error = props_result;
        return false;
    }
    const vk::ExternalMemoryBufferCreateInfo external{
        .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
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
    const auto requirements = device.getBufferMemoryRequirements(created);
    const std::uint32_t bits = props.memoryTypeBits & requirements.memoryTypeBits;
    std::uint32_t type = FindType(bits, false);
    if (type == ~0u) {
        type = FindHostType(bits);
    }
    if (type == ~0u) {
        device.destroyBuffer(created);
        error = vk::Result::eErrorFeatureNotPresent;
        return false;
    }
    const vk::MemoryAllocateFlagsInfo address_flags{
        .flags = vk::MemoryAllocateFlagBits::eDeviceAddress,
    };
    const vk::ImportMemoryFdInfoKHR import_info{
        .pNext = LayerMemory() ? &address_flags : nullptr,
        .handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT,
        .fd = fd,
    };
    const auto [memory_result, allocated] = device.allocateMemory({
        .pNext = &import_info,
        .allocationSize = std::max<u64>(size, requirements.size),
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
    buffer = created;
    memory = allocated;
    return true;
}

/// The GPU's import of [start, start + size) of direct memory, a new chunk; false on failure.
bool ImportRange(u64 start, u64 size) {
    const auto began = std::chrono::steady_clock::now();
    {
        std::scoped_lock lk{mutex};
        if (chunk_count == MaxChunks) {
            return false;
        }
    }
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::Result error{};
    bool ok = false;
    // Before its pages are taken: a release meanwhile clears them instead of punching them out of
    // the memfd (the import would keep the old pages, the game the new ones).
    runtime_memory_note_pinned(start, size);
    if (lazy_udmabuf) {
        // Its pages allocated through the backing view first, which the runtime advised as huge
        // pages: udmabuf alone would take 4 KiB ones (no mapping to advise), and the GPU reading
        // the game's memory in place lost 0.8 ms a frame to them on a Steam Deck.
#ifdef MADV_POPULATE_WRITE
        if (void* view = runtime_memory_backing_pointer(start)) {
            (void)madvise(view, size, MADV_POPULATE_WRITE);
        }
#endif
        const int fd = MakeUdmabuf(runtime_memory_pool_fd(), start, size);
        ok = fd >= 0 && ImportDmaBuf(fd, size, buffer, memory, error);
        if (fd >= 0 && !ok) {
            close(fd);
        }
    } else {
        void* pointer = runtime_memory_backing_pointer(start);
        ok = pointer && ImportHost(pointer, size, buffer, memory, error);
    }
    if (!ok) {
        static std::atomic<int> failures{0};
        if (failures.fetch_add(1) < 8) {
            std::fprintf(stderr, "Guest memory: importing %#llx+%llu MiB for the GPU failed (%s)\n",
                         (unsigned long long)start, (unsigned long long)(size >> 20),
                         lazy_udmabuf && error == vk::Result{} ? std::strerror(errno)
                                                              : vk::to_string(error).c_str());
        }
        return false;
    }
    {
        std::scoped_lock lk{mutex};
        std::size_t i = chunk_count++;
        for (; i > 0 && chunks[i - 1]->phys > start; --i) {
            chunks[i] = chunks[i - 1];
        }
        chunks[i] = new Chunk{start, size, buffer, memory, next_index++};
        RegisterChunk(chunks[i]);
    }
    imports_made.fetch_add(1, std::memory_order_relaxed);
#ifndef _WIN32
    // BB_IMPORT_TRACE=<lo MiB>-<hi MiB> (diagnostics): who asked for imports starting there.
    static const std::pair<u64, u64> trace = [] {
        unsigned long long lo = 0, hi = 0;
        const char* env = bb_env("BB_IMPORT_TRACE");
        if (!env || std::sscanf(env, "%llu-%llu", &lo, &hi) != 2) return std::pair<u64, u64>{1, 0};
        return std::pair<u64, u64>{lo << 20, hi << 20};
    }();
    if (start >= trace.first && start < trace.second) {
        static std::atomic<int> traced{0};
        if (traced.fetch_add(1) < 12) {
            const off_t data = lseek(runtime_memory_pool_fd(), off_t(start), SEEK_DATA);
            std::fprintf(stderr, "Guest memory: import %#llx+%llu MiB (paged read %d, first data at "
                         "%#llx) asked by:\n",
                         (unsigned long long)start, (unsigned long long)(size >> 20), int(paged_read),
                         (unsigned long long)data);
            void* frames[24];
            backtrace_symbols_fd(frames, backtrace(frames, 24), 2);
        }
    }
#endif
    static const bool log = bb_env("BB_IMPORT_LOG") != nullptr;
    if (log) {
        std::printf("Guest memory: import %#llx+%#llx in %.1f ms (frame %llu)\n",
                    (unsigned long long)start, (unsigned long long)size,
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - began)
                        .count(),
                    (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed));
    }
    const u64 before = imported_bytes.fetch_add(size, std::memory_order_relaxed);
    if ((before >> 28) != ((before + size) >> 28)) {
        std::printf("Guest memory: %llu MiB of the game's memory imported by the GPU so far (%llu "
                    "imports)\n",
                    (unsigned long long)((before + size) >> 20),
                    (unsigned long long)imports_made.load(std::memory_order_relaxed));
    }
    return true;
}

/// Whether a chunk holds [start, end) whole (mutex held).
bool CoveredLocked(u64 start, u64 end) {
    for (std::size_t i = 0; i < chunk_count && chunks[i]->phys <= start; ++i) {
        if (chunks[i]->phys + chunks[i]->size >= end) {
            return true;
        }
    }
    return false;
}

/// The size of the runtime's memfd (direct and flexible memory).
u64 PoolBytes() {
    static const u64 bytes = [] {
        struct stat st {};
        return fstat(runtime_memory_pool_fd(), &st) == 0 ? u64(st.st_size) : ~u64{0};
    }();
    return bytes;
}

/// Whether another import over pages an import may hold already fits the budget. The kernel
/// counts every import whole in GTT, the same pages again in each: a Steam Deck run imported
/// 6.9 GiB of a 4 GiB pool in 80 minutes (granule pairs, widened bindings), GTT reached 8.1 GiB
/// and command submission failed for memory (device lost). Granule imports never overlap one
/// another (the pool's size at most); the others get BB_IMPORT_OVERLAP_MB (1024) in all. Past
/// that a binding not held whole by one import takes the module's other path (a copy).
bool OverlapFits(u64 size) {
    // 1 GiB where GTT is small (a Steam Deck's 8 GiB: "not enough memory for command submission"
    // over it), a quarter of it where it is large (a desktop's half of RAM): there reads past the
    // budget went to VRAM copies (~500 bindings a second).
    static const u64 slack = [] {
        const char* env = bb_env("BB_IMPORT_OVERLAP_MB");
        if (env && *env) {
            return u64{std::strtoull(env, nullptr, 10)} << 20;
        }
        u64 gtt = 0;
        for (int card = 0; card < 8; ++card) {
            const std::string path =
                "/sys/class/drm/card" + std::to_string(card) + "/device/mem_info_gtt_total";
            if (FILE* f = std::fopen(path.c_str(), "r")) {
                unsigned long long bytes = 0;
                if (std::fscanf(f, "%llu", &bytes) == 1 && (gtt == 0 || bytes < gtt)) {
                    gtt = bytes;
                }
                std::fclose(f);
            }
        }
        return gtt >= (u64{12} << 30) ? gtt / 4 : u64{1024} << 20;
    }();
    // Against the overlapping imports themselves: the memfd's size is the whole pool (a Deck's
    // 5+ GiB), far above what the game uses (4 GiB), and a budget over it allowed 6.4 GiB of
    // imports (GTT 8.5 GiB) in the next session.
    const u64 overlapping = overlap_bytes.load(std::memory_order_relaxed);
    // BB_IMPORT_WRITTEN_BUDGET=1 (tests): written bindings under the budget too (as before).
    static const bool written_budget = bb_env_on("BB_IMPORT_WRITTEN_BUDGET");
    if ((written_binding && !written_budget) || overlapping + size <= slack) {
        return true;
    }
    static std::atomic<bool> told{false};
    if (!told.exchange(true)) {
        std::printf("Guest memory: %llu MiB of overlapping imports (%llu MiB in all): no more "
                    "(BB_IMPORT_OVERLAP_MB)\n",
                    (unsigned long long)(overlapping >> 20),
                    (unsigned long long)(imported_bytes.load(std::memory_order_relaxed) >> 20));
    }
    return false;
}

/// Imports [start, end) unless a chunk holds it or it failed before (populate_mutex held).
/// `overlapping`: it may hold pages of other imports (not a granule of its own).
bool ImportOnce(u64 start, u64 end, bool overlapping) {
    {
        std::scoped_lock chunks_lk{mutex};
        if (CoveredLocked(start, end)) {
            return true;
        }
    }
    if (std::ranges::find(failed_imports, std::pair{start, end}) != failed_imports.end()) {
        return false;
    }
    if (overlapping && !OverlapFits(end - start)) {
        return false;
    }
    if (!ImportRange(start, end - start)) {
        failed_imports.emplace_back(start, end);
        return false;
    }
    if (overlapping) {
        overlap_bytes.fetch_add(end - start, std::memory_order_relaxed);
    }
    return true;
}

/// Read-only paged bindings (SetPagedRead): a granule the game never wrote (a hole of the memfd)
/// is not imported for them: its pages would be allocated (udmabuf, MADV_POPULATE_WRITE) only
/// to hold zeros. A shader reading one whole-memory buffer published every block of it, and
/// 0.7-1.0 GiB of a 3.9 GiB pool were all-zero granules (Steam Deck and PC, 2026-10-23). Checked
/// again every frame: data the game writes there before a dispatch is seen at its binding.
std::array<std::atomic<u64>, 1024> hole_seen{}; // per granule: frame + 1 when found a hole then

u64 FrameNow() {
    return BbStats::frame_number.load(std::memory_order_relaxed);
}

/// Whether granule [start, end) of the memfd has no page yet (never written), this frame.
bool GranuleHole(u64 start, u64 end) {
    const u64 g = start / Granule;
    if (g >= hole_seen.size()) {
        return false;
    }
    const u64 frame = FrameNow() + 1;
    if (hole_seen[g].load(std::memory_order_relaxed) == frame) {
        return true;
    }
#ifdef _WIN32
    return false;
#else
    const off_t data = lseek(runtime_memory_pool_fd(), off_t(start), SEEK_DATA);
    const bool hole = (data < 0 && errno == ENXIO) || (data >= 0 && u64(data) >= end);
    if (hole) {
        hole_seen[g].store(frame, std::memory_order_relaxed);
    }
    return hole;
#endif
}

u64 HoleEnd(u64 phys) {
    const u64 g = phys / Granule;
    return g < hole_seen.size() && hole_seen[g].load(std::memory_order_relaxed) == FrameNow() + 1
               ? (g + 1) * Granule
               : 0;
}

/// Shader::paged_range_hook: [va, va + size) across a granule boundary of the address: through
/// the page table. A function of the buffer alone: decided again when it is bound, after the
/// shader's permutation was chosen, an answer from the runtime's current mappings could differ
/// (the game remaps memory while areas load) and the descriptor did not match the shader (GPUVM
/// faults, 2026-10-23). The game's heaps are mapped at 0x1000000000 + their physical address (a
/// multiple of a granule apart), so this is the imports' boundary there; elsewhere a range it
/// misses takes the import windows as before. BB_PAGED_CROSSING=0: windows only.
bool CrossesImports(std::uint64_t va, std::uint64_t size) {
    return va / Granule != (va + size - 1) / Granule;
}

/// The layer's populate callback (BbLayer::GuestPopulate): [phys, phys + want) imported whole:
/// its granule; a range across granules in 2 MiB steps (it overlaps the granules' imports: as
/// little as may be), where an import may be that large and the overlap budget allows; else the
/// granule holding phys.
bool Populate(u64 phys, u64 want) {
    const auto window = [](u64 a, u64 b, u64 align) {
        return std::pair<u64, u64>{a / align * align, (b + align - 1) / align * align};
    };
    if (paged_read) {
        const u64 start = phys / Granule * Granule, end = std::min(start + Granule, PoolBytes());
        if (GranuleHole(start, end)) {
            return false;
        }
    }
    auto [start, end] = window(phys, phys + std::max<u64>(want, 1), Granule);
    bool overlapping = false;
    if (end - start > Granule) {
        // Across one granule boundary: a window centred on it (half a granule each side, else a
        // granule), the same for every range there: ranges in 2 MiB steps made a new import for
        // nearly every binding, used up the overlap budget, and later ones went to VRAM copies.
        const u64 boundary = (phys / Granule + 1) * Granule, half = Granule / 2;
        if (phys >= boundary - half && phys + want <= boundary + half) {
            std::tie(start, end) = std::pair<u64, u64>{boundary - half, boundary + half};
        } else if (phys + want <= boundary + Granule) {
            std::tie(start, end) = std::pair<u64, u64>{boundary - Granule, boundary + Granule};
        } else {
            std::tie(start, end) = window(phys, phys + want, u64{2} << 20);
        }
        overlapping = end - start <= import_limit && OverlapFits(end - start);
        if (!overlapping) {
            std::tie(start, end) = window(phys, phys + 1, Granule);
        }
    }
    // Not past the memfd (the last granule of the pool may be shorter).
    end = std::min(end, PoolBytes());
    if (start >= end) {
        return false;
    }
    // Asked again for every resolution of a range no import holds whole (an import may not be
    // that large): the answer for a window is kept, without the lock (its key: start and length
    // in pages; bit 63 the answer).
    static std::array<std::atomic<u64>, 4096> answers{};
    const u64 key = (start >> 12) | (((end - start) >> 12) << 36);
    auto& slot = answers[((key * 0x9E3779B97F4A7C15ull) >> 52) & (answers.size() - 1)];
    static std::atomic<u64> calls{0}, fast{0}, made_before{0};
    calls.fetch_add(1, std::memory_order_relaxed);
    if (const u64 known = slot.load(std::memory_order_acquire); (known & ~(u64{1} << 63)) == key) {
        fast.fetch_add(1, std::memory_order_relaxed);
        return known >> 63;
    }
    bool ok;
    {
        std::scoped_lock lk{populate_mutex};
        const u64 before = imports_made.load(std::memory_order_relaxed);
        ok = ImportOnce(start, end, overlapping);
        made_before.fetch_add(imports_made.load(std::memory_order_relaxed) - before,
                              std::memory_order_relaxed);
    }
    slot.store(key | (u64{ok} << 63), std::memory_order_release);
    // Per 10 s: how often the module asked (BB_IMPORT_LOG=1).
    static const bool log = bb_env("BB_IMPORT_LOG") != nullptr;
    static std::atomic<long long> printed{0};
    const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    if (log && now - printed.load(std::memory_order_relaxed) >= 10) {
        printed.store(now, std::memory_order_relaxed);
        std::printf("Guest memory: populate asked %llu times in 10 s (%llu answered without the "
                    "lock), %llu imports made\n",
                    (unsigned long long)calls.exchange(0), (unsigned long long)fast.exchange(0),
                    (unsigned long long)made_before.exchange(0));
    }
    return ok;
}

} // namespace BbGuestMemory::detail
