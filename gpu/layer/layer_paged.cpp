// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport BB_LAYER_MEMORY: the page table huge bindings read through (entries, republishing,
// verification), CPU write marks, GPU write modes and the upload flush. Split out of
// buffer_cache_layer.cpp (2026-10-26); shared helpers in buffer_cache_layer_internal.h.
#include <algorithm>
#include "shim/bb_env.h"
#include <bit>
#include <array>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <sys/mman.h>
#include <ucontext.h>
#include <map>
#include <tuple>
#include <string>
#include <cstdio>
#include <fmt/format.h>
#include <unordered_map>
#include <xxhash.h>
#include "shader/profile.h"
#include "bbport_free_check.h"
#include "bbport_guest_memory.h"
#include "bbport_write_log.h"
#include "bbport_sections.h"
#include "bbport_toggles.h"
#include "buffers/buffer_cache.h"
#include "memory/dirty_tracker.h"
#include "common/signal_context.h"
#include "core/signals.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"
// After the Vulkan headers of the video core (their configuration).
#include "bblayer_gpu_memory.h"
#include "bblayer_write_traps.h"
#include "layer/buffer_cache_layer_internal.h"

namespace VideoCore {

u64 BufferCache::LayerEntry(u64 block) {
    // The device address of the block's current source: its mirror where valid, else the game's
    // memory (a block mapped whole, in a chunk), else 0 (reads give zero, writes are dropped).
    const VAddr va = block << block_shift;
    if (Layer().AnyValid(block, block + 1) && !layer_volatile.Contains(block, block + 1)) {
        if (const auto mirror = Layer().MirrorAt(va)) {
            return mirror->device_address + (va - mirror->base);
        }
    }
    const auto span = Layer().ResolveInPlace(va, block_size);
    if (!span || span->source->device_address == 0) {
        return 0;
    }
    return span->source->device_address + span->offset;
}

void BufferCache::LayerWriteEntries(u64 first, u64 end) {
    if (first >= end) {
        return;
    }
    // The table covers its window only (Shader::PageTableBase): blocks outside it read as unmapped.
    const u64 window_first = Shader::PageTableBase >> block_shift;
    const u64 window_end = window_first + (Shader::PageTableSpan >> block_shift);
    if (first < window_first || end > window_end) {
        static std::atomic<u32> told{0};
        if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
            std::printf("Layer memory: page table entries %#llx..%#llx outside its window "
                        "[%#llx, %#llx): left unmapped\n",
                        (unsigned long long)(first << block_shift), (unsigned long long)(end << block_shift),
                        (unsigned long long)Shader::PageTableBase,
                        (unsigned long long)(Shader::PageTableBase + Shader::PageTableSpan));
        }
        first = std::max(first, window_first);
        end = std::min(end, window_end);
        if (first >= end) {
            return;
        }
    }
    // Diagnostics (BB_FRAME_STATS): where the time goes (t_layer 8-11).
    auto mark = std::chrono::steady_clock::now();
    const auto lap = [&](size_t kind) {
        if (!BbStats::enabled) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        BbStats::t_layer[kind].fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now - mark).count(),
            std::memory_order_relaxed);
        mark = now;
    };
    BbStats::layer_entries_written.fetch_add(end - first, std::memory_order_relaxed);
    const auto staging =
        staging_pool.Request((end - first) * 2 * sizeof(vk::DeviceAddress), MemoryType::HostUncached);
    lap(8);
    // Read/current view, then CPU-visible guest view. A write-through store updates both: the CPU
    // sees the result without discarding all mirrors covered by a huge descriptor.
    auto* entries = reinterpret_cast<vk::DeviceAddress*>(staging.mapped);
    auto* guest_entries = entries + (end - first);
    // The guest view (as LayerEntry in place: a block mapped whole, in a chunk), one translation
    // per mapping, not per block: each takes the runtime's memory lock, which the game's threads
    // hold while they map memory; a level streaming in kept the GPU thread waiting 30-70 ms per
    // frame on the thousands a large promotion made.
    for (u64 block = first; block < end;) {
        const auto span = Layer().ResolvePrefix(block << block_shift, (end - block) << block_shift);
        const u64 whole = span ? span->size >> block_shift : 0;
        if (whole == 0) {
            guest_entries[block - first] = 0;
            // Paged reads of memory with no GPU source read zero: none seen in play; the game
            // profile's host-only allocations (Profile::host_only_allocations) would show up here.
            {
                static std::atomic<u64> sourceless{0};
                const u64 n = sourceless.fetch_add(1, std::memory_order_relaxed);
                if (n < 32 || (n & (n - 1)) == 0) {
                    std::printf("Layer memory: page table entry without a GPU source at %#llx (%llu so far)\n",
                                (unsigned long long)(block << block_shift), (unsigned long long)(n + 1));
                }
            }
            ++block;
            continue;
        }
        const vk::DeviceAddress base =
            span->source->device_address ? span->source->device_address + span->offset : 0;
        for (u64 i = 0; i < whole; ++i) {
            guest_entries[block - first + i] = base ? base + (i << block_shift) : 0;
        }
        block += whole;
    }
    lap(9);
    std::copy_n(guest_entries, end - first, entries);
    // BB_EXP_PAGED_IN_PLACE=1 (experiment, missing map geometry after area loads): paged
    // bindings read the game's memory itself, never a VRAM copy.
    static const bool paged_in_place = [] {
        const bool on = bb_env_on("BB_EXP_PAGED_IN_PLACE");
        if (on) {
            std::printf("Layer memory: paged bindings read in place (BB_EXP_PAGED_IN_PLACE=1)\n");
        }
        return on;
    }();
    // Valid blocks (not volatile) are read from their mirror.
    std::vector<std::pair<u64, u64>> valid;
    if (!paged_in_place) {
        Layer().ForEachValid(first, end, [&](u64 a, u64 b) { valid.emplace_back(a, b); });
    }
    for (const auto& [a, b] : valid) {
        std::optional<BbLayer::MemorySource> mirror;
        const bool any_volatile = layer_volatile.Overlaps(a, b);
        for (u64 block = a; block < b; ++block) {
            const VAddr va = block << block_shift;
            if (any_volatile && layer_volatile.Contains(block, block + 1)) {
                continue;
            }
            if (!mirror || va < mirror->base || va >= mirror->base + mirror->size) {
                mirror = Layer().MirrorAt(va);
            }
            if (mirror) {
                entries[block - first] = mirror->device_address + (va - mirror->base);
            }
        }
    }
    lap(10);
    if (PagedVerifyOn()) {
        for (u64 block = first; block < end; ++block) {
            layer_shadow[block] = entries[block - first];
        }
    }
    staging.Flush();
    // Keep the original current-view table layout for DMA/ReadConst users too; append the
    // guest-view table in the second half rather than interleaving its entries.
    const u64 bytes = (end - first) * sizeof(vk::DeviceAddress);
    const std::array<vk::BufferCopy, 2> copies{{
        {staging.offset, (first - window_first) * sizeof(vk::DeviceAddress), bytes},
        {staging.offset + bytes, bda_pagetable_buffer->SizeBytes() / 2 +
                                    (first - window_first) * sizeof(vk::DeviceAddress), bytes},
    }};
    runtime.CopyBuffer(staging.buffer, bda_pagetable_buffer.get(), copies);
    lap(11);
}

void BufferCache::LayerRepublish(u64 first, u64 end) {
    BbStats::Timer timer{BbStats::t_layer[6]};
    std::vector<std::pair<u64, u64>> runs;
    layer_published.ForEachInRange(first, end, [&](const Interval& iv) {
        runs.emplace_back(std::max(first, iv.start), std::min(end, iv.end));
    });
    for (const auto& [a, b] : runs) {
        LayerWriteEntries(a, b);
    }
}

void BufferCache::LayerNoteCpuWrite(VAddr address, u64 size) {
    if (size == 0) { // guest threads too: no layer_mirrors here
        return;
    }
    {
        // The CPU's bytes are newer than what the GPU wrote there: never copied back over them,
        // and the CPU uses these blocks (layer_cpu_blocks).
        std::scoped_lock lk{layer_uploads_mutex};
        layer_cpu_written.emplace_back(address, size);
        layer_cpu_written_pending.store(true, std::memory_order_release);
    }
    if (Layer().ValidBytes() == 0) {
        return;
    }
    // Announced: the write must not fault (a file read into it would fail): unprotected now, and
    // the block goes back in place.
    LayerDisarmTraps(address >> block_shift, ((address + size - 1) >> block_shift) + 1, true);
    std::scoped_lock lk{layer_uploads_mutex};
    layer_pending_uploads.emplace_back(address, size);
    layer_uploads_pending.store(true, std::memory_order_release);
}

void BufferCache::LayerForgetGpuData() {
    if (!layer_cpu_written_pending.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<std::pair<VAddr, u64>> ranges;
    {
        std::scoped_lock lk{layer_uploads_mutex};
        ranges.swap(layer_cpu_written);
        layer_cpu_written_pending.store(false, std::memory_order_release);
    }
    for (const auto& [address, size] : ranges) {
        gpu_modified_ranges.Subtract(address, size);
        memory_tracker->UnmarkRegionAsGpuModified(address, size);
        LayerMarkCpuBlocks(address >> block_shift, ((address + size - 1) >> block_shift) + 1);
        if (MarkLogged(address, size)) {
            std::printf("Layer mark: announced CPU write %#llx+%#llx\n", (unsigned long long)address,
                        (unsigned long long)size);
        }
    }
}

void BufferCache::LayerFlushUploads() {
    BbStats::Timer timer{BbStats::t_layer[7]};
    // What the CPU wrote into blocks with a VRAM copy: uploaded now, before a paged binding
    // (it reads mirrors through the page table, without the bound path's own upload).
    if (!layer_uploads_pending.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<std::pair<VAddr, u64>> ranges;
    {
        std::scoped_lock lk{layer_uploads_mutex};
        ranges.swap(layer_pending_uploads);
        layer_uploads_pending.store(false, std::memory_order_release);
    }
    for (const auto& [address, size] : ranges) {
        const u64 first = address >> block_shift;
        const u64 end = ((address + size - 1) >> block_shift) + 1;
        ForEachValidPiece(Layer(), first, end, block_shift,
                          [&](const BbLayer::MemorySource& mirror, u64 a, u64 b) {
                              const VAddr from = std::max<VAddr>(address, a << block_shift);
                              const VAddr to = std::min<VAddr>(address + size, b << block_shift);
                              if (from < to) {
                                  SynchronizeMemory(static_cast<Buffer*>(mirror.owner), from,
                                                    static_cast<u32>(to - from), false, false);
                              }
                          });
    }
}

/// BB_PAGED_VERIFY: every block a paged binding (under 64 MiB, or the copy shader's listed
/// ranges) reaches has the entry the mappings give now; mismatches are printed.
void BufferCache::LayerVerifyPaged(VAddr address, u64 size, bool is_written) {
    std::vector<std::pair<u64, u64>> ranges;
    for (const auto& limit : paged_ranges) {
        if (limit.ranges && limit.base == address) {
            for (const auto& [start, length] : *limit.ranges) {
                const u64 a = std::max<u64>(start, address);
                const u64 b = std::min<u64>(start + length, address + size);
                if (a < b) {
                    ranges.emplace_back(a, b);
                }
            }
        }
    }
    if (ranges.empty()) {
        if (size > 64_MB) {
            return;
        }
        ranges.emplace_back(address, address + size);
    }
    static std::atomic<u32> printed{0};
    static std::atomic<u64> checked{0}, wrong{0};
    for (const auto& [a, b] : ranges) {
        for (u64 block = a >> block_shift; block <= (b - 1) >> block_shift; ++block) {
            checked.fetch_add(1, std::memory_order_relaxed);
            if (Layer().AnyValid(block, block + 1)) {
                continue; // read from its mirror
            }
            u64 expected = 0, expected_phys = ~u64{0};
            if (const auto span = Layer().ResolvePrefix(block << block_shift, block_size);
                span && span->size >= block_size && span->source->device_address) {
                expected = span->source->device_address + span->offset;
                expected_phys = span->source->base + span->offset;
            }
            const auto it = layer_shadow.find(block);
            const u64 have = it == layer_shadow.end() ? 0 : it->second;
            // Another import of the same pages is as good (overlapping imports).
            if (have == expected ||
                (have != 0 && expected != 0 && Layer().PhysOfDevice(have) == expected_phys)) {
                continue;
            }
            wrong.fetch_add(1, std::memory_order_relaxed);
            if (printed.fetch_add(1, std::memory_order_relaxed) < 200) {
                std::printf("Paged verify: binding %#llx+%#llx%s block %#llx: entry %#llx (phys "
                            "%#llx), the mapping gives %#llx (phys %#llx; %s; %llu wrong of %llu "
                            "checked)\n",
                            (unsigned long long)address, (unsigned long long)size,
                            is_written ? " (written)" : "",
                            (unsigned long long)(block << block_shift), (unsigned long long)have,
                            (unsigned long long)(have ? Layer().PhysOfDevice(have) : 0),
                            (unsigned long long)expected, (unsigned long long)expected_phys,
                            layer_published.Contains(block, block + 1) ? "published" : "not published",
                            (unsigned long long)wrong.load(), (unsigned long long)checked.load());
            }
        }
    }
}

std::pair<const Buffer*, u64> BufferCache::LayerPagedRecord(VAddr address, u64 size,
                                                            bool is_written) {
    if (maintained_epoch != packet_epoch) {
        Maintain();
    }
    LayerProcessTraps();
    LayerFlushUploads();
    const u64 first = address >> block_shift;
    const u64 end = ((address + size - 1) >> block_shift) + 1;
    NoteUse(address, size);
    // BB_LAYER_VERIFY: the mirrors a paged binding reads too (every 16th frame: it may span GiB).
    if (!is_written && LayerVerifyOn() &&
        BbStats::frame_number.load(std::memory_order_relaxed) % 16 == 0) {
        LayerVerifyMirror(address, size, false);
    }
    // Page table entries for the blocks not published yet (those mapped now: an unmapped block
    // is tried again next time).
    std::vector<std::pair<u64, u64>> gaps;
    // The copy shader: only the blocks its control list names (paged_ranges).
    const std::vector<std::pair<unsigned long long, unsigned long long>>* only = nullptr;
    for (const auto& limit : paged_ranges) {
        if (limit.ranges && limit.base == address) {
            only = limit.ranges;
        }
    }
    if (only) {
        for (const auto& [range_start, range_size] : *only) {
            // Clipped to the binding, not dropped: a range reaching past its end (its last
            // element, a small source) was left unpublished, and the shader read whatever an
            // earlier binding had published for those blocks (garbage copied into vertex and
            // index buffers, 2026-10-23).
            const u64 range = std::max<u64>(range_start, address);
            const u64 range_end = std::min<u64>(range_start + range_size, address + size);
            if (range >= range_end) {
                continue;
            }
            const u64 rfirst = range >> block_shift, rend = ((range_end - 1) >> block_shift) + 1;
            NoteBlocksBound(rfirst, rend);
            layer_published.ForEachGap(rfirst, rend, [&](u64 a, u64 b) { gaps.emplace_back(a, b); });
        }
    } else {
        NoteBlocksBound(first, end);
        layer_published.ForEachGap(first, end, [&](u64 a, u64 b) { gaps.emplace_back(a, b); });
    }
    // Read only: memory the game never wrote is not imported for it (BbGuestMemory::SetPagedRead),
    // its blocks stay on the trash and are tried again at the next binding; such a run is
    // skipped whole (UnbackedEnd), not block by block.
    // BB_PAGED_HOLES=1 (experimental, off): measured 2026-10-23, the whole-memory bindings that
    // import the all-zero granules are written ones, which this does not cover.
    static const bool holes = bb_env_on("BB_PAGED_HOLES");
    const bool skip_holes = holes && !is_written;
    BbGuestMemory::SetPagedRead(skip_holes);
    for (const auto& [a, b] : gaps) {
        u64 run = a;
        for (u64 block = a; block <= b; ++block) {
            const bool mapped = block < b && Layer().ResolvePrefix(block << block_shift, 1);
            if (!mapped) {
                if (run < block) {
                    LayerWriteEntries(run, block);
                    layer_published.Add({run, block});
                }
                if (block < b && skip_holes) {
                    const u64 unbacked = Layer().UnbackedEnd(block << block_shift) >> block_shift;
                    if (unbacked > block + 1) {
                        block = std::min(unbacked, b) - 1;
                    }
                }
                run = block + 1;
            }
        }
    }
    BbGuestMemory::SetPagedRead(false);
    if (PagedVerifyOn()) {
        LayerVerifyPaged(address, size, is_written);
    }
    if (is_written) {
        BbFreeCheck::NoteGpuWrite(address, size, 0, BbFreeCheck::InPlaceWrite);
        NoteBindingWriteTick(address, size);
        // Where it writes is unknown: what is in VRAM there may now differ from the game's
        // memory; it is copied back before it goes in place (LayerDemote).
        if (!force_writes_in_place) {
            Layer().ForEachValid(first, end, [&](u64 a, u64 b) {
                gpu_modified_ranges.Add(a << block_shift, (b - a) << block_shift);
            });
        }
    }
    if (!layer_trash) {
        layer_trash = std::make_unique<Buffer>(instance, 0, 64_KB, MemoryType::DeviceLocal,
                                               "bbport paged trash");
        if (bb_env("BB_BDA_LOG")) {
            std::printf("BDA: paged trash at GPU %#llx, page table at %#llx+%#llx\n",
                        (unsigned long long)layer_trash->BufferDeviceAddress(),
                        (unsigned long long)bda_pagetable_buffer->BufferDeviceAddress(),
                        (unsigned long long)bda_pagetable_buffer->SizeBytes());
        }
    }
    if (skip_holes && !layer_zeros) {
        layer_zeros = std::make_unique<Buffer>(instance, 0, 64_KB, MemoryType::DeviceLocal,
                                               "bbport paged zeros");
        runtime.FillBuffer(layer_zeros.get(), 0, 64_KB, 0);
    }
    // A read-only binding reads zeros where nothing is mapped (memory the game never wrote).
    const vk::DeviceAddress trash =
        (skip_holes ? layer_zeros : layer_trash)->BufferDeviceAddress();
    const u32 record[8] = {static_cast<u32>(address),
                           static_cast<u32>(address >> 32),
                           static_cast<u32>(std::min<u64>(size, 0xFFFFFFFFull)),
                           u32(is_written && force_writes_in_place),
                           static_cast<u32>(trash),
                           static_cast<u32>(trash >> 32),
                           static_cast<u32>(bda_pagetable_buffer->SizeBytes() /
                                            (2 * sizeof(vk::DeviceAddress))),
                           0};
    const u64 offset = stream_buffer.Copy(record, sizeof(record), instance.StorageMinAlignment());
    BbStats::bound_in_place_bytes.fetch_add(size, std::memory_order_relaxed);
    return {&stream_buffer, offset};
}

bool BufferCache::LayerActive() {
    return BbGuestMemory::LayerMemory();
}

bool BufferCache::LayerPagedActive() {
    // BB_LAYER_PAGED=0 (tests): huge bindings cut in place instead of the page table.
    static const bool paged = !bb_env_off("BB_LAYER_PAGED");
    return paged && LayerActive();
}

int BufferCache::LayerGpuWritesMode() {
    // Anywhere (2) gave exploding vertices and garbage textures: GPU data in a VRAM copy of memory
    // the CPU also uses (the game's buffers, its heap's headers, reused allocations) went stale or
    // was copied back over newer CPU data. Images (render targets) are the GPU's own.
    static const int mode = [] {
        const char* env = bb_env("BB_LAYER_GPU_WRITES");
        return env && (env[0] == '0' || env[0] == '2') ? env[0] - '0' : 1;
    }();
    return mode;
}

bool BufferCache::LayerInImage(VAddr address, u64 size) {
    bool inside = false;
    texture_cache.ForEachImageInRegion(address, size, [&](ImageId, Image& image) {
        inside = inside || (image.info.guest_address <= address &&
                            address + size <= image.info.guest_address + image.info.guest_size);
    });
    return inside;
}

void BufferCache::LayerMarkCpuBlocks(u64 first, u64 end) {
    layer_cpu_blocks.Add({first, end});
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    for (u64 block = first; block < end; ++block) {
        layer_cpu_mark_second[block] = now;
    }
}

void BufferCache::LayerExpireCpuMarks(VAddr address, u64 size) {
    // BB_LAYER_CPU_MARK_SECONDS=n (default 5; 0: marks stay until the memory is handed out
    // again, as before). After a save was switched, the GPU wrote 8-16 MiB ranges of the next
    // area's data over blocks the last one's loaders had written: all of it in place, over the
    // bus (RX 7800 XT imitating NVIDIA: 140 -> 60 FPS; GTX 1060: 16 FPS). Blocks the CPU still
    // uses (hot, dynamic, watched) keep their marks; a write after that is caught by the trap
    // on the block's VRAM copy and sends it back in place, as for any block.
    static const u32 seconds = [] {
        const char* env = bb_env("BB_LAYER_CPU_MARK_SECONDS");
        return env && *env ? u32(std::strtoul(env, nullptr, 10)) : 5u;
    }();
    if (seconds == 0) {
        return;
    }
    const u64 first = (address + block_size - 1) >> block_shift; // whole blocks only
    const u64 end = (address + size) >> block_shift;
    static std::array<std::atomic<u64>, 5> expire_kept{}; // recent, hot, dynamic, watched, partial
    static std::atomic<u32> told_second{0};
    static const bool mark_log = bb_env("BB_LAYER_MARK_LOG") != nullptr;
    if (const u32 second = BbStats::coarse_second.load(std::memory_order_relaxed);
        mark_log && second - told_second.load() >= 10) {
        told_second = second;
        std::printf("Layer marks kept (block checks in 10 s): recent %llu, CPU-hot %llu, dynamic %llu, "
                    "watched %llu, partial %llu\n",
                    (unsigned long long)expire_kept[0].exchange(0), (unsigned long long)expire_kept[1].exchange(0),
                    (unsigned long long)expire_kept[2].exchange(0), (unsigned long long)expire_kept[3].exchange(0),
                    (unsigned long long)expire_kept[4].exchange(0));
    }
    if (first >= end) {
        ++expire_kept[4];
        return;
    }
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    std::vector<u64> stale;
    layer_cpu_blocks.ForEachInRange(first, end, [&](const auto& run) {
        for (u64 block = std::max<u64>(run.start, first); block < std::min<u64>(run.end, end);
             ++block) {
            const auto it = layer_cpu_mark_second.find(block);
            const u32 marked = it == layer_cpu_mark_second.end() ? 0 : it->second;
            // Dynamic by the GPU's mixed writes (not the CPU's: layer_dynamic_second) expires too.
            const auto dyn = layer_dynamic_second.find(block);
            const bool gpu_dynamic = dyn != layer_dynamic_second.end() && now - dyn->second >= seconds;
            const int why = now - marked < seconds                             ? 0
                            : layer_cpu_hot.Contains(block, block + 1)           ? 1
                            : dynamic_blocks.Contains(block) && !gpu_dynamic     ? 2
                            : layer_watched.Contains(block, block + 1)           ? 3
                                                                                 : -1;
            if (why >= 0) {
                ++expire_kept[why];
                continue;
            }
            stale.push_back(block);
        }
    });
    for (const u64 block : stale) {
        layer_cpu_blocks.Subtract(block, block + 1);
        layer_cpu_mark_second.erase(block);
        if (layer_dynamic_second.erase(block)) {
            dynamic_blocks.Subtract(block, block + 1);
            layer_write_bounces.Subtract(block, block + 1);
        }
    }
    if (!stale.empty()) {
        static std::atomic<u64> expired{0};
        const u64 total = expired.fetch_add(stale.size()) + stale.size();
        if ((total << block_shift >> 26) != ((total - stale.size()) << block_shift >> 26)) {
            std::printf("Layer memory: %llu MiB of stale CPU marks dropped under whole GPU writes\n",
                        (unsigned long long)((total << block_shift) >> 20));
        }
    }
}

bool BufferCache::LayerGpuWriteToVram(VAddr address, u64 size) {
    const int mode = LayerGpuWritesMode();
    if (mode == 1 && layer_cpu_blocks.Overlaps(address >> block_shift,
                                               ((address + size - 1) >> block_shift) + 1)) {
        LayerExpireCpuMarks(address, size);
    }
    const bool cpu_blocks = layer_cpu_blocks.Overlaps(address >> block_shift,
                                                      ((address + size - 1) >> block_shift) + 1);
    const bool vram = mode == 2 || (mode == 1 && !cpu_blocks);
    if (mode == 1 && cpu_blocks) {
        BbStats::gpu_writes_kept_by_cpu_bytes.fetch_add(size, std::memory_order_relaxed);
        LayerArmProbe(address, size);
    }
    // BB_LAYER_WRITE_LOG=1 (diagnostics): large GPU writes kept in place, and the images near them.
    static const bool log = bb_env("BB_LAYER_WRITE_LOG") != nullptr;
    if (log && !vram && size >= (u64{1} << 20)) {
        static std::map<std::pair<VAddr, u64>, u32> seen;
        if (seen.size() < 64 && seen[{address, size}]++ == 0) {
            std::string near;
            texture_cache.ForEachImageInRegion(address, size, [&](ImageId, Image& image) {
                near += fmt::format(" [{:#x}+{:#x} {}x{} fmt {}]", image.info.guest_address,
                                    image.info.guest_size, image.info.size.width,
                                    image.info.size.height, u32(image.info.pixel_format));
            });
            std::printf("Layer write in place: %#llx+%#llx; images:%s\n",
                        (unsigned long long)address, (unsigned long long)size,
                        near.empty() ? " none" : near.c_str());
        }
    }
    return vram;
}

} // namespace VideoCore
