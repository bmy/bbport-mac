// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport: BufferCache, who wrote what: CPU writes the game's hooks announce (assets, fresh
// ranges), late and large GPU writes, precise uploads, shadow copies and the copy-backs they
// queue. Split out of buffer_cache.cpp (2026-10-26); shared state in buffer_cache_internal.h.


#include <chrono>
#include "shim/bb_env.h"
#include <array>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <magic_enum/magic_enum.hpp>
#include "shader/profile.h"
#include "bbport_copy.h"
#include "bbport_toggles.h"
#include "bbport_sections.h"
#include "bbport_free_check.h"
#include "bbport_guest_memory.h"
#include "game_profile.h"
#include "bblayer_gpu_memory.h"
#include "bblayer_write_traps.h"
#include "common/alignment.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "cp/command_processor.h"
#include "buffers/buffer.h"
#include "buffers/buffer_cache.h"
#include "memory/dirty_tracker.h"
#include "render/pipeline.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"
#include "buffers/buffer_cache_internal.h"

#include <mutex>
#include <pthread.h>
#include "bbport_cpu.h" // bbport: not <x86intrin.h> (arm64: the native GPU process)
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <fmt/format.h>
#include <sys/mman.h>
#ifdef __APPLE__
#include <sys/ucontext.h> // bbport: <ucontext.h> requires _XOPEN_SOURCE on macOS
#else
#include <ucontext.h>
#endif
#include "common/signal_context.h"
#include "core/signals.h"
#include <vk_mem_alloc.h>
#include "buffers/buffer_cache_internal.h"

namespace VideoCore {

void BufferCache::NoteAssetWrite(VAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    std::scoped_lock lk{pending_assets_mutex};
    pending_assets.push_back({addr, size, false});
    assets_pending.store(true, std::memory_order_release);
}

void BufferCache::NoteFreshRange(VAddr addr, u64 size) {
    BbStats::gpu_range_allocs.fetch_add(1, std::memory_order_relaxed);
    BbStats::gpu_range_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
    if (static const char* mark_log = bb_env("BB_LAYER_MARK_LOG"); mark_log && size >= (1u << 20)) {
        std::printf("Layer mark: handed out %#llx+%#llx\n", (unsigned long long)addr,
                    (unsigned long long)size);
    }
    // The GPU's old data there is dead: whole pages inside stop being GPU-modified now, so the
    // writes that follow (announced after the fact) read none of it back over themselves.
    const VAddr pages_start = Common::AlignUp(addr, u64(4096));
    const VAddr pages_end = Common::AlignDown(addr + size, u64(4096));
    if (pages_start < pages_end) {
        memory_tracker->UnmarkRegionAsGpuModified(pages_start, pages_end - pages_start);
    }
    std::scoped_lock lk{pending_assets_mutex};
    pending_assets.push_back({addr, size, true});
    assets_pending.store(true, std::memory_order_release);
}

void BufferCache::NoteCpuWriteRange(VAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    for (VAddr block = addr >> block_shift; block <= (addr + size - 1) >> block_shift; ++block) {
        NoteCpuWrite(block << block_shift, 0);
    }
}

void BufferCache::ProcessPendingAssets() {
    if (!assets_pending.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<PendingRange> ranges;
    {
        std::scoped_lock lk{pending_assets_mutex};
        ranges.swap(pending_assets);
        assets_pending.store(false, std::memory_order_release);
    }
    for (const auto& [addr, size, fresh] : ranges) {
        const u64 first = addr >> block_shift, end = (addr + size + block_size - 1) >> block_shift;
        if (fresh) {
            // Its old contents are dead and the next writer may be code we do not hear of: in
            // place until a loader writes it. Until the move, uploads keep the VRAM copy current.
            for (auto& [key, write] : large_writes) {
                if (addr < write.addr + write.size && write.addr < addr + size) {
                    ++write.fresh;
                }
            }
            asset_bytes.Subtract(addr, addr + size);
            gpu_written_bytes.Subtract(addr, addr + size);
            promote_candidates.Subtract(first, end);
            // What the GPU wrote there before is dead too: never copied back over what the new
            // owner writes (only the GPU's data beside it is, QueueCopyBacks/LayerDemote).
            gpu_modified_ranges.Subtract(addr, size);
            memory_tracker->UnmarkRegionAsGpuModified(addr, size);
            if (LayerMode()) {
                // Blocks whole in it start over: the GPU may keep its data there in VRAM again (CPU
                // writes still queued mark them again, LayerForgetGpuData).
                const u64 whole_first = (addr + block_size - 1) >> block_shift;
                const u64 whole_end = (addr + size) >> block_shift;
                if (whole_first < whole_end) {
                    layer_cpu_blocks.Subtract(whole_first, whole_end);
                }
            }
            for (u64 block = first; block < end; ++block) {
                const VAddr block_addr = block << block_shift;
                const VAddr from = std::max<VAddr>(addr, block_addr);
                const VAddr to = std::min<VAddr>(addr + size, block_addr + block_size);
                const bool in_vram =
                    LayerMode() ? BbLayer::GpuMemory::Get().AnyValid(block, block + 1)
                                : resident_ranges.Contains(block) && !in_place_blocks.Contains(block);
                if (in_vram) {
                    if (memory_tracker->IsRegionGpuModified(block_addr, block_size) ||
                        gpu_modified_ranges.Intersects(block_addr, block_size)) {
                        // The GPU's data beside it is only in VRAM: it goes back into the game's
                        // memory as the submission ends, then the block moves (QueueCopyBacks).
                        // An upload now would put stale bytes over that data (whole pages).
                        copy_back_blocks.Add({block, block + 1});
                        continue;
                    }
                    demotions_waiting.push_back({block, scheduler.CurrentTick(), false});
                }
                memory_tracker->MarkRegionAsCpuModified(from, to - from);
            }
            continue;
        }
        asset_bytes.Add({addr, addr + size});
        // bbport: a block the GPU once wrote in place (dynamic for good) that the loaders have now
        // written whole holds the CPU's data only: it may have a VRAM copy again. Kept, such
        // blocks piled up as areas were loaded over memory the last one's GPU buffers had used
        // (164 -> 1117 MiB of dynamic blocks after a few teleports), and one of them in a range
        // kept the whole range in place (a 7 MiB binding read over the bus for one such block of
        // 110): the same spot went from 76.6 to 55.3 FPS. The GPU's writes there stay in place
        // from now on (layer_cpu_blocks), so the CPU's view (the translator reads shader code and
        // descriptors there) is never left behind a VRAM copy. Blocks the CPU writes often
        // (layer_cpu_hot) stay dynamic. BB_LAYER_RELOAD_RESET=0: as before.
        static const bool reload_reset = !bb_env_off("BB_LAYER_RELOAD_RESET");
        if (LayerMode() && reload_reset) {
            for (u64 block = first; block < end; ++block) {
                if (!dynamic_blocks.Contains(block) || layer_cpu_hot.Contains(block, block + 1)) {
                    continue;
                }
                bool whole = true;
                asset_bytes.ForEachGap(block << block_shift, (block + 1) << block_shift,
                                       [&](u64, u64) { whole = false; });
                if (!whole) {
                    continue;
                }
                LayerMarkCpuBlocks(block, block + 1);
                dynamic_blocks.Subtract(block, block + 1);
                layer_write_bounces.Subtract(block, block + 1);
                BbStats::dynamic_reloaded_bytes.fetch_add(block_size, std::memory_order_relaxed);
            }
        }
        for (u64 block = first; block < end; ++block) {
            const bool in_place =
                LayerMode() ? !BbLayer::GpuMemory::Get().AnyValid(block, block + 1)
                            : in_place_blocks.Contains(block) && resident_ranges.Contains(block);
            if (in_place && VramEligible(block)) {
                promote_candidates.Add({block, block + 1});
            }
        }
    }
}

void BufferCache::NoteLateCommandWrite(VAddr addr, const void* data, u64 size) {
    if (size == 0) {
        return;
    }
    const auto* bytes = static_cast<const u8*>(data);
    std::scoped_lock lk{late_writes_mutex};
    late_writes.emplace_back(addr, std::vector<u8>(bytes, bytes + size));
    late_writes_pending.store(true, std::memory_order_release);
}

void BufferCache::TraceBinding(VAddr addr, u64 size, bool is_written, int kind) {
    static const u64 trace_size = [] {
        const char* env = bb_env("BB_TRACE_SIZE");
        return env ? std::strtoull(env, nullptr, 0) : u64(0);
    }();
    // BB_TRACE_ADDR=addr,size: that range is traced from the start (with BB_TRACE_SIZE set).
    static const bool fixed_range = [this] {
        const char* env = bb_env("BB_TRACE_ADDR");
        if (!env) {
            return false;
        }
        char* end = nullptr;
        const u64 start = std::strtoull(env, &end, 0);
        traced_ranges.emplace_back(start, end && *end == ',' ? std::strtoull(end + 1, nullptr, 0) : 4096);
        return true;
    }();
    if (trace_size == 0) {
        return;
    }
    if (is_written && size == trace_size &&
        std::ranges::find(traced_ranges, std::pair<VAddr, u64>{addr, size}) == traced_ranges.end()) {
        if (traced_ranges.size() >= 16) {
            traced_ranges.erase(traced_ranges.begin() + (fixed_range ? 1 : 0));
        }
        traced_ranges.emplace_back(addr, size);
    }
    const bool traced = std::ranges::any_of(traced_ranges, [&](const auto& range) {
        return addr < range.first + range.second && range.first < addr + size;
    });
    if (traced) {
        ++trace_counts[kind];
        if (!is_written) {
            trace_min = std::min(trace_min, size);
            trace_max = std::max(trace_max, size);
        }
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - trace_report < std::chrono::seconds(5)) {
        return;
    }
    trace_report = now;
    std::printf("Trace %#llx-byte GPU buffers (%zu, 5 s): write in place %llu, write VRAM %llu; read "
                "in place %llu, snapshot %llu, VRAM %llu (sizes %llu-%llu); image source: in place "
                "%llu, chunk %llu, GPU data %llu, staging %llu\n",
                (unsigned long long)trace_size, traced_ranges.size(),
                (unsigned long long)trace_counts[0], (unsigned long long)trace_counts[1],
                (unsigned long long)trace_counts[2], (unsigned long long)trace_counts[3],
                (unsigned long long)trace_counts[4], (unsigned long long)(trace_max ? trace_min : 0),
                (unsigned long long)trace_max, (unsigned long long)trace_counts[5],
                (unsigned long long)trace_counts[6], (unsigned long long)trace_counts[7],
                (unsigned long long)trace_counts[9]);
    trace_counts = {};
    trace_min = ~0ull;
    trace_max = 0;
}

void BufferCache::NoteLargeGpuWrite(VAddr addr, u64 size) {
    auto& write = large_writes[addr ^ (size << 40)];
    if (write.count++ == 0) {
        write.addr = addr;
        write.size = size;
        write.images = texture_cache.DescribeImagesIn(addr, size);
    }
    write.in_place = IsInPlace(addr, size);
    const auto now = std::chrono::steady_clock::now();
    if (now - large_writes_report < std::chrono::seconds(5)) {
        return;
    }
    large_writes_report = now;
    std::vector<const LargeWrite*> rows;
    for (const auto& [key, entry] : large_writes) {
        rows.push_back(&entry);
    }
    std::ranges::sort(rows, [](const auto* a, const auto* b) { return a->count > b->count; });
    std::printf("GPU-written over 64 KiB (5 s):");
    for (std::size_t i = 0; i < std::min<std::size_t>(rows.size(), 12); ++i) {
        std::printf("%s %#llx+%#llx x%llu %s fresh %llu%s", i ? ";" : "",
                    (unsigned long long)rows[i]->addr, (unsigned long long)rows[i]->size,
                    (unsigned long long)rows[i]->count, rows[i]->in_place ? "in-place" : "VRAM",
                    (unsigned long long)rows[i]->fresh, rows[i]->images.c_str());
    }
    std::printf("\n");
    large_writes.clear();
}

void BufferCache::NotePreciseUpload(VAddr addr, u64 size) {
    if (size == 0) {
        return;
    }
    if (LayerMode()) {
        LayerNoteCpuWrite(addr, size);
    }
    std::scoped_lock lk{late_writes_mutex};
    precise_uploads.emplace_back(addr, size);
    late_writes_pending.store(true, std::memory_order_release);
}

void BufferCache::ProcessLateWrites() {
    if (!late_writes_pending.load(std::memory_order_acquire)) {
        return;
    }
    std::vector<std::pair<VAddr, u64>> uploads;
    std::vector<std::pair<VAddr, std::vector<u8>>> writes;
    {
        std::scoped_lock lk{late_writes_mutex};
        writes.swap(late_writes);
        uploads.swap(precise_uploads);
        late_writes_pending.store(false, std::memory_order_release);
    }
    for (const auto& [addr, bytes] : writes) {
        const u64 size = bytes.size();
        const u64 first = addr >> block_shift, last = (addr + size - 1) >> block_shift;
        if (((addr | size) & 3) != 0 || !resident_ranges.Contains(first, last + 1) ||
            in_place_blocks.Overlaps(first, last + 1)) {
            continue; // the GPU reads the game's memory there (or nothing yet)
        }
        const auto* arena = GetArena(first, last);
        runtime.UpdateBuffer(arena, arena->Offset(addr), bytes);
        BbStats::late_vram_writes.fetch_add(1, std::memory_order_relaxed);
    }
    // CPU bytes next to GPU data in VRAM: copied by the GPU from the game's memory, block by
    // block (a VRAM block's guest memory is one contiguous run of a chunk).
    for (const auto& [addr, size] : uploads) {
        const VAddr end = addr + size;
        for (VAddr at = addr; at < end;) {
            const u64 block = at >> block_shift;
            const VAddr to = std::min<VAddr>(end, (block + 1) << block_shift);
            u64 phys = 0;
            uintptr_t mapping_end = 0;
            const BbGuestMemory::Chunk* chunk = nullptr;
            if (resident_ranges.Contains(block) && !in_place_blocks.Contains(block) &&
                runtime_memory_direct_phys(at, &phys, &mapping_end) && to <= mapping_end &&
                (chunk = BbGuestMemory::Find(phys)) != nullptr &&
                phys + (to - at) <= chunk->phys + chunk->size) {
                const auto* arena = GetArena(block, block);
                const vk::BufferCopy copy{phys - chunk->phys, at - arena->cpu_addr, to - at};
                runtime.CopyFromGuestChunk(chunk->buffer, arena, std::span{&copy, 1});
                BbStats::late_vram_writes.fetch_add(1, std::memory_order_relaxed);
            }
            at = to;
        }
    }
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::ShadowCopy(VAddr addr, u64 size) {
    static const bool enabled = bb_env_on("BB_SHADOWS");
    constexpr u64 MaxShadow = 4_MB, ShadowRing = 128_MB;
    // A GPU write there not done yet: the copy (through the chunk's own buffer) would not wait
    // for it; the arena is read instead.
    if (!enabled || !GarlicInVram() || size > MaxShadow ||
        IsRegionGpuModified(addr, size)) {
        return std::nullopt;
    }
    if (const u64 tick = scheduler.CurrentTick(); tick != shadow_tick) {
        shadows.clear();
        shadow_tick = tick;
    }
    for (const auto& shadow : shadows) {
        if (shadow.addr == addr && shadow.size == size) {
            return std::pair<const Buffer*, u64>{shadow_buffer.get(), shadow.offset};
        }
    }
    const auto source = GuestChunkSource(addr, size);
    if (!source) {
        return std::nullopt;
    }
    if (!shadow_buffer) {
        shadow_buffer = std::make_unique<StreamBuffer>(instance, scheduler, MemoryType::DeviceLocal,
                                                       ShadowRing);
    }
    const auto offset = shadow_buffer->Reserve(size, instance.StorageMinAlignment());
    if (!offset) {
        return std::nullopt;
    }
    const vk::BufferCopy copy{source->second, *offset, size};
    runtime.CopyBuffer(source->first, shadow_buffer.get(), std::span{&copy, 1});
    shadows.push_back({addr, size, *offset});
    BbStats::shadow_copies.fetch_add(1, std::memory_order_relaxed);
    BbStats::shadow_bytes.fetch_add(size, std::memory_order_relaxed);
    return std::pair<const Buffer*, u64>{shadow_buffer.get(), *offset};
}

bool BufferCache::HotVertexCopyOn() {
    static const bool env_on = bb_env_on("BB_HOT_VERTEX_COPY");
    return env_on || BbToggle::Experiment(1024);
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::HotVertexCopy(VAddr addr, u64 size) {
    constexpr u64 MaxCopy = 8_MB, Piece = 1_MB;
    // A GPU write there not done yet: the CPU would copy what was there before it.
    if (!HotVertexCopyOn() || size > MaxCopy || size == 0 || stream_buffer.mapped_data.empty() ||
        IsRegionGpuModified(addr, size)) {
        return std::nullopt;
    }
    // The game writes such data before it submits the commands that read it: one copy serves
    // every binding of the same guest submission (cleared at the next: NewGuestWorkInOrder).
    for (const auto& copy : hot_copies) {
        if (copy.addr <= addr && addr + size <= copy.addr + copy.size) {
            return std::pair<const Buffer*, u64>{&stream_buffer, copy.offset + (addr - copy.addr)};
        }
    }
    // Not waiting for the ring: the range is read in place instead.
    const auto offset = stream_buffer.Reserve(size, 256, false);
    if (!offset) {
        return std::nullopt;
    }
    for (u64 done = 0; done < size; done += Piece) {
        const u64 n = std::min(Piece, size - done);
        SmallGuestCopy({
            .run = &RunGuestCopy,
            .context = this,
            .source = addr + done,
            .destination = reinterpret_cast<u64>(stream_buffer.mapped_data.data() + *offset + done),
            .size = n,
            .extra = reinterpret_cast<u64>(static_cast<Buffer*>(&stream_buffer)),
        });
    }
    hot_copies.push_back({addr, size, *offset});
    BbStats::shadow_copies.fetch_add(1, std::memory_order_relaxed);
    BbStats::shadow_bytes.fetch_add(size, std::memory_order_relaxed);
    return std::pair<const Buffer*, u64>{&stream_buffer, *offset};
}

bool BufferCache::VertexReadCopyOn() {
    static const bool env_on = bb_env_on("BB_VERTEX_READ_COPY");
    return env_on != BbToggle::Experiment(2048);
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::HotVertexCopySpan(VAddr addr, u64 size,
                                                                             u64 lo, u64 hi) {
    constexpr u64 MaxCopy = 4_MB, Piece = 1_MB;
    const u64 n = hi - lo;
    const u64 capacity = stream_buffer.SizeBytes();
    if (lo >= hi || hi > size || n > MaxCopy || stream_buffer.mapped_data.empty() ||
        IsRegionGpuModified(addr + lo, n)) {
        return std::nullopt;
    }
    // Bound so that addr sits at the returned offset: it must not fall before the buffer's start
    // nor the range past its end (the bytes outside [lo, hi) there are never read).
    const auto at = [&](u64 copied) -> std::optional<std::pair<const Buffer*, u64>> {
        if (copied < lo || copied - lo + size > capacity) {
            return std::nullopt;
        }
        return std::pair<const Buffer*, u64>{&stream_buffer, copied - lo};
    };
    // One copy serves every draw of the same guest submission that reads within it.
    for (const auto& copy : hot_copies) {
        if (copy.addr <= addr + lo && addr + hi <= copy.addr + copy.size) {
            if (auto bound = at(copy.offset + (addr + lo - copy.addr))) {
                return bound;
            }
        }
    }
    const auto offset = stream_buffer.Reserve(n, 256, false);
    if (!offset || !at(*offset)) {
        return std::nullopt;
    }
    for (u64 done = 0; done < n; done += Piece) {
        const u64 piece = std::min(Piece, n - done);
        SmallGuestCopy({
            .run = &RunGuestCopy,
            .context = this,
            .source = addr + lo + done,
            .destination = reinterpret_cast<u64>(stream_buffer.mapped_data.data() + *offset + done),
            .size = piece,
            .extra = reinterpret_cast<u64>(static_cast<Buffer*>(&stream_buffer)),
        });
    }
    hot_copies.push_back({addr + lo, n, *offset});
    BbStats::shadow_copies.fetch_add(1, std::memory_order_relaxed);
    BbStats::shadow_bytes.fetch_add(n, std::memory_order_relaxed);
    return at(*offset);
}

void BufferCache::DropShadows(VAddr addr, u64 size, std::source_location caller) {
    std::erase_if(shadows, [&](const Shadow& shadow) {
        return addr < shadow.addr + shadow.size && shadow.addr < addr + size;
    });
    if (write_shadows.empty()) {
        return;
    }
    // Entries do not overlap: the one starting before addr, then those starting inside.
    auto it = write_shadows.lower_bound(addr);
    if (it != write_shadows.begin() && std::prev(it)->first + std::prev(it)->second.size > addr) {
        --it;
    }
    static const bool trace = bb_env_on("BB_WRITE_SHADOW_TRACE");
    static u64 drops = 0;
    while (it != write_shadows.end() && it->first < addr + size) {
        if (trace && std::string_view{caller.function_name()}.find("ShadowGpuWrites") ==
                         std::string_view::npos &&
            drops++ % 2000 == 0) {
            std::printf("Write shadow dropped %#llx+%#llx by a write %#llx+%#llx (%s:%u)\n",
                        (unsigned long long)it->first, (unsigned long long)it->second.size,
                        (unsigned long long)addr, (unsigned long long)size, caller.file_name(),
                        unsigned(caller.line()));
        }
        it = write_shadows.erase(it);
    }
}

void BufferCache::DropWriteShadows() {
    static const bool trace = bb_env_on("BB_WRITE_SHADOW_TRACE");
    static u64 clears = 0;
    if (trace && !write_shadows.empty() && clears++ % 500 == 0) {
        std::printf("Write shadows cleared by a paged GPU write (%llu times)\n",
                    (unsigned long long)clears);
    }
    write_shadows.clear();
}

void BufferCache::ShadowGpuWrites(
    std::span<const std::pair<unsigned long long, unsigned long long>> ranges) {
    // Off by default: RX 7800 XT imitating NVIDIA, 148 copy shader dispatches a frame, 534
    // bindings/frame served from the copies: 120 -> 110 FPS (copies 0.22 ms, GPU idle +0.4 ms),
    // the G-buffer pass no faster (2026-10-28). BB_WRITE_SHADOWS=1 or experiment bit 512: on.
    static const bool enabled = bb_env_on("BB_WRITE_SHADOWS");
    constexpr u64 MaxRange = 64_KB, MaxDispatch = 4_MB, ShadowRing = 128_MB;
    if (!(enabled || BbToggle::Experiment(512)) || !GarlicInVram() || ranges.empty()) {
        return;
    }
    // Kept across submissions (the copy shader and the draws reading its constants are often in
    // different ones; cleared per submission, 5 of ~1500 bindings a frame found theirs): writes
    // after it drop them (DropShadows: CPU, command processor, bound GPU writes; paged GPU writes
    // of unknown range all of them), and the ring's next lap ends them (GpuWriteShadow).
    if (!shadow_buffer) {
        shadow_buffer = std::make_unique<StreamBuffer>(instance, scheduler, MemoryType::DeviceLocal,
                                                       ShadowRing);
    }
    // One copy command per source buffer (the ranges usually lie in one chunk of the game's memory).
    thread_local std::vector<std::pair<const Buffer*, vk::BufferCopy>> copies;
    copies.clear();
    u64 total = 0;
    static const bool trace = bb_env_on("BB_WRITE_SHADOW_TRACE");
    static u64 skipped = 0;
    const auto skip = [&](const char* why, u64 addr, u64 size) {
        if (trace && skipped++ % 2000 == 0) {
            std::printf("Write shadow skipped (%s): %#llx+%#llx of %zu ranges, %llu skipped\n", why,
                        (unsigned long long)addr, (unsigned long long)size, ranges.size(),
                        (unsigned long long)skipped);
        }
    };
    for (const auto& [addr, size] : ranges) {
        DropShadows(addr, size);
        if (size == 0 || size > MaxRange || total + size > MaxDispatch) {
            skip(size > MaxRange ? "large" : "total", addr, size);
            continue;
        }
        const auto source = GuestChunkSource(addr, size);
        if (!source) {
            skip("no chunk", addr, size);
            continue;
        }
        const auto offset = shadow_buffer->Reserve(size, instance.StorageMinAlignment());
        if (!offset) {
            break;
        }
        copies.emplace_back(source->first, vk::BufferCopy{source->second, *offset, size});
        write_shadows[addr] = {size, *offset, write_shadow_reserved};
        write_shadow_reserved += size + instance.StorageMinAlignment();
        total += size;
    }
    std::stable_sort(copies.begin(), copies.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    thread_local std::vector<vk::BufferCopy> group;
    for (size_t i = 0; i < copies.size();) {
        group.clear();
        const Buffer* source = copies[i].first;
        for (; i < copies.size() && copies[i].first == source; ++i) {
            group.push_back(copies[i].second);
        }
        // After the dispatch's paged write: CopyBuffer waits for it (Runtime::global_write).
        runtime.CopyBuffer(source, shadow_buffer.get(), group);
    }
    BbStats::write_shadow_copies.fetch_add(copies.size(), std::memory_order_relaxed);
    BbStats::write_shadow_bytes.fetch_add(total, std::memory_order_relaxed);
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::GpuWriteShadow(VAddr addr, u64 size) {
    // BB_WRITE_SHADOW_TRACE=1 (diagnostics): why lookups miss, printed now and then.
    static const bool trace = bb_env_on("BB_WRITE_SHADOW_TRACE");
    static u64 counts[4]{}; // before all, past an entry, cut short, hit
    const auto note = [&](int kind, u64 start, u64 len) {
        if (!trace) {
            return;
        }
        ++counts[kind];
        if (kind == 2 && counts[2] % 5000 == 1) {
            std::printf("Write shadow cut short: read %#llx+%#llx, copy %#llx+%#llx\n",
                        (unsigned long long)addr, (unsigned long long)size,
                        (unsigned long long)start, (unsigned long long)len);
        }
        if (kind == 1 && counts[1] % 5000 == 1) {
            std::printf("Write shadow none: read %#llx+%#llx, nearest below %#llx+%#llx, %zu copies\n",
                        (unsigned long long)addr, (unsigned long long)size,
                        (unsigned long long)start, (unsigned long long)len, write_shadows.size());
        }
        if ((counts[0] + counts[1] + counts[2] + counts[3]) % 100000 == 0) {
            std::printf("Write shadow lookups: %llu before all, %llu past, %llu cut short, %llu hits\n",
                        (unsigned long long)counts[0], (unsigned long long)counts[1],
                        (unsigned long long)counts[2], (unsigned long long)counts[3]);
        }
    };
    auto it = write_shadows.upper_bound(addr);
    if (it == write_shadows.begin()) {
        note(0, 0, 0);
        return std::nullopt;
    }
    --it;
    if (addr + size > it->first + it->second.size) {
        note(addr < it->first + it->second.size ? 2 : 1, it->first, it->second.size);
        return std::nullopt;
    }
    note(3, it->first, it->second.size);
    // Half a lap of the ring old: its bytes may be reused soon (ShadowCopy shares the ring).
    if (write_shadow_reserved - it->second.stamp > shadow_buffer->SizeBytes() / 2) {
        write_shadows.erase(it);
        return std::nullopt;
    }
    return std::pair<const Buffer*, u64>{shadow_buffer.get(), it->second.offset + (addr - it->first)};
}

bool BufferCache::UpdateGpuWritten(VAddr addr, std::span<const u8> data) {
    const u64 size = data.size();
    if (size == 0 || ((addr | size) & 3) != 0 ||
        IsAnyInPlace(addr, size) || !memory_tracker->IsRegionGpuModified(addr, size)) {
        return false;
    }
    const u64 first = addr >> block_shift, last = (addr + size - 1) >> block_shift;
    if (!resident_ranges.Contains(first, last + 1)) {
        return false;
    }
    const auto* arena = GetArena(first, last);
    runtime.UpdateBuffer(arena, arena->Offset(addr), data);
    BbStats::gpu_data_updates.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void BufferCache::NoteGpuWrite(VAddr addr, u64 size) {
    if (size == 0 || gpu_written_bytes.Contains(addr, addr + size)) {
        return;
    }
    // Only what the GPU writes regularly (in 3 frames of 60) is its data, like a default heap: a
    // buffer the GPU writes only now and then stays in place, where writes by the CPU are seen
    // (0x104ff30000, 192 KiB, written by the GPU every few seconds: in VRAM the picture smeared
    // while moving).
    constexpr u32 Frames = 3, Window = 60;
    const u32 frame = static_cast<u32>(BbStats::frame_number.load(std::memory_order_relaxed));
    auto& [last, frames] = gpu_write_frames[addr ^ (size << 40)];
    if (last != frame || frames == 0) {
        frames = frame - last > Window ? 1 : frames + 1;
        last = frame;
    }
    if (frames < Frames) {
        if (gpu_write_frames.size() > 65536) {
            std::erase_if(gpu_write_frames,
                          [frame](const auto& entry) { return frame - entry.second.first > Window; });
        }
        return;
    }
    gpu_written_bytes.Add({addr, addr + size});
    // Blocks the GPU already writes in place move to VRAM once they are all known data.
    for (u64 block = addr >> block_shift; block <= (addr + size - 1) >> block_shift; ++block) {
        const bool in_place =
            LayerMode() ? !BbLayer::GpuMemory::Get().AnyValid(block, block + 1)
                        : in_place_blocks.Contains(block) && resident_ranges.Contains(block);
        if (in_place && VramEligible(block)) {
            promote_candidates.Add({block, block + 1});
        }
    }
}

void BufferCache::QueueCopyBacks(u64 submitted) {
    if (copy_back_blocks.Empty()) {
        return;
    }
    if (LayerMode()) {
        // GPU data in mirrors back into the game's memory, then in place.
        for (const auto& range : copy_back_blocks) {
            LayerDemote(range.start, range.end, true, 4);
        }
        copy_back_blocks.Clear();
        return;
    }
    std::vector<u64> blocks;
    for (const auto& range : copy_back_blocks) {
        for (u64 block = range.start; block < range.end; ++block) {
            blocks.push_back(block);
        }
    }
    copy_back_blocks.Clear();
    bool moved = false;
    for (const u64 block : blocks) {
        const VAddr address = block << block_shift;
        u64 phys = 0;
        uintptr_t mapping_end = 0;
        const BbGuestMemory::Chunk* chunk = nullptr;
        if (!resident_ranges.Contains(block) || in_place_blocks.Contains(block) ||
            !runtime_memory_direct_phys(address, &phys, &mapping_end) ||
            (phys & (block_size - 1)) != 0 || mapping_end - address < block_size ||
            (chunk = BbGuestMemory::Find(phys)) == nullptr ||
            !address_space[block >> blocks_per_arena_page_shift]) {
            continue;
        }
        const auto source = GuestChunkSource(address, block_size);
        if (!source) {
            continue;
        }
        // What the GPU wrote there since the block was handed out (not the bytes handed out
        // again: the CPU may be writing them now), copied at the end of this submission.
        const auto* arena = GetArena(block, block);
        boost::container::small_vector<vk::BufferCopy, 8> copies;
        gpu_written_bytes.ForEachInRange(address, address + block_size, [&](const Interval& iv) {
            const VAddr from = std::max<VAddr>(iv.start, address);
            const VAddr to = std::min<VAddr>(iv.end, address + block_size);
            if (from < to) {
                copies.push_back({arena->Offset(from), source->second + (from - address), to - from});
            }
        });
        if (!copies.empty()) {
            runtime.CopyBuffer(arena, source->first, copies);
        }
        // Then in place from the next submission on (as ProcessDemotions).
        if (const auto old = resident_ranges.Find(block); old != resident_ranges.end()) {
            ReleaseResidencySlot(old->memory, (old->offset + (block - old->start)) << block_shift);
        }
        Backing backing;
        backing.start = block;
        backing.end = block + 1;
        backing.memory = chunk->memory;
        backing.offset = (phys - chunk->phys) >> block_shift;
        resident_ranges.Subtract(block, block + 1);
        resident_ranges.Add(backing);
        in_place_blocks.Add({block, block + 1});
        for (const auto& covering : arenas) {
            if (covering.cpu_addr <= address && address - covering.cpu_addr < covering.size_bytes) {
                BindsForArena(&covering)->binds.emplace_back(vk::SparseMemoryBind{
                    .resourceOffset = address - covering.cpu_addr,
                    .size = block_size,
                    .memory = chunk->memory,
                    .memoryOffset = backing.offset << block_shift,
                });
            }
        }
        memory_tracker->UnmarkRegionAsGpuModified(address, block_size);
        gpu_modified_ranges.Subtract(address, block_size);
        memory_tracker->MarkRegionAsCpuModified(address, block_size);
        BbStats::copied_back_blocks.fetch_add(1, std::memory_order_relaxed);
        moved = true;
    }
    if (moved) {
        bind_wait_tick = std::max(bind_wait_tick, submitted);
    }
}

} // namespace VideoCore
