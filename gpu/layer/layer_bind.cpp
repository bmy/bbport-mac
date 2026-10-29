// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

// bbport BB_LAYER_MEMORY: binding the game's memory for a draw (LayerBind: a mirror, the chunk
// in place, or the page table), its per-thread memo, and which blocks are watched for promotion.
// Split out of buffer_cache_layer.cpp (2026-10-26); shared helpers in buffer_cache_layer_internal.h.
#include <algorithm>
#include "shim/bb_env.h"
#include <bit>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <sys/mman.h>
#ifdef __APPLE__
#include <sys/ucontext.h> // bbport: <ucontext.h> requires _XOPEN_SOURCE on macOS
#else
#include <ucontext.h>
#endif
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
#include "core/memory.h"
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

std::optional<std::pair<const Buffer*, u64>> BufferCache::LayerBind(VAddr address, u64 size,
                                                                    bool is_written,
                                                                    bool is_texel_buffer,
                                                                    bool may_promote) {
    BB_SECTION(LayerBind);
    // A range the GPU writes is always imported whole (over the overlap budget): served by the
    // arena instead, its GPU-written data sat in a VRAM copy the copy shader and the CPU do not
    // see (exploding vertices, shader code copied from stale memory: 2026-10-23).
    const BbGuestMemory::WrittenBinding written_scope{is_written};
    // Residency changes queued since the last packet: loads, demotions, unmaps, idle blocks.
    if (maintained_epoch != packet_epoch) {
        Maintain();
    }
    LayerProcessTraps(); // blocks the CPU wrote go back in place before anything binds them
    const u64 first = address >> block_shift;
    const u64 end = ((address + size - 1) >> block_shift) + 1;
    NoteUse(address, size); // idle VRAM blocks go back in place (LayerProcessIdle)
    using Kind = BbLayer::Resolution::Kind;
    // The module's answer for this range while nothing moved (layer_generation: promotions,
    // demotions, unmaps); ranges it does not know yet are asked again.
    struct ResolveMemo {
        VAddr address = 0;
        u64 size = 0;
        u64 generation = 0;
        bool written = false; // resolved for a written binding (imported over the budget)
        BbLayer::Resolution resolution;
    };
    // On the heap: static TLS is reserved on every thread's stack, the game's small ones too.
    thread_local auto resolve_memo = std::make_unique<std::array<ResolveMemo, 2048>>();
    auto& memo =
        (*resolve_memo)[((address * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full)) >> 53];
    BbLayer::Resolution resolution;
    // A read's answer does not serve a written binding: the read may have been refused an import
    // (the overlap budget) the written one is given.
    if (memo.address == address && memo.size == size && memo.generation == layer_generation + 1 &&
        (memo.written || !is_written || memo.resolution.kind != Kind::Mixed)) {
        resolution = memo.resolution;
    } else {
        resolution = Layer().Resolve(address, size);
        if (resolution.kind != Kind::None) {
            memo = {address, size, layer_generation + 1, is_written, resolution};
        }
    }
    // Domains (BB_LAYER_DOMAINS=1) keep VRAM copies per game allocation; a read over several of
    // them, all in VRAM, merges them into one (as mirrors were before domains) instead of reading
    // in place over the bus (84 FPS for 245 on the PC without this).
    if (resolution.kind == Kind::Mixed && !is_written && LayerMergeAcross(first, end)) {
        resolution = Layer().Resolve(address, size);
        memo = {address, size, layer_generation + 1, is_written, resolution};
    }
    // BB_RESIDENCY_MIX=1: bindings per kind (none, in place, mirror, mixed) and written, per 2 s.
    static const bool mix_stats = [] {
        const char* env = bb_env("BB_RESIDENCY_MIX");
        return env && (env[0] == '1' || env[0] == '2');
    }();
    if (mix_stats) {
        static std::array<std::atomic<u64>, 8> count{}, bytes{};
        static std::atomic<u32> printed{0};
        const size_t k = size_t(resolution.kind) * 2 + (is_written ? 1 : 0);
        // In place: whether all its blocks may go to VRAM (else: why not, by block).
        static std::array<std::atomic<u64>, 4> in_place_why{}; // eligible, asset gap, dynamic, chunk
        // Unknown bytes, by why not watched (BB_LAYER_WATCH): watchable, the GPU's dynamic,
        // volatile, failed before, mapping smaller than a block, written binding.
        static std::array<std::atomic<u64>, 6> unwatched_why{};
        if (resolution.kind == Kind::InPlace) {
            int why = 0;
            for (u64 block = first; block < end && why == 0; ++block) {
                const VAddr a = block << block_shift;
                bool known = true;
                asset_bytes.ForEachGap(a, a + block_size, [&](u64 s, u64 e) {
                    known = known && gpu_written_bytes.Contains(s, e);
                });
                why = !known ? 1 : dynamic_blocks.Contains(block) ? 2
                      : !Layer().ResolveInPlace(a, block_size) ? 3 : 0;
                if (why == 1) {
                    int prot = 0, type = -1;
                    uintptr_t vma_end = 0;
                    const int w = is_written ? 5
                                  : dynamic_blocks.Contains(block) &&
                                            !layer_cpu_hot.Contains(block, block + 1) ? 1
                                  : layer_volatile.Contains(block, block + 1) ? 2
                                  : layer_watch_failed.Contains(block, block + 1) ? 3
                                  : (runtime_memory_vma_info(a, &prot, &type, &vma_end) &&
                                     (prot & 0x2) && vma_end < a + block_size) ? 4 : 0;
                    unwatched_why[w].fetch_add(size, std::memory_order_relaxed);
                }
            }
            in_place_why[why].fetch_add(size, std::memory_order_relaxed);
        }
        static std::atomic<u32> why_printed{0};
        if (BbStats::coarse_second.load() - why_printed.load() >= 2) {
            why_printed.store(BbStats::coarse_second.load());
            std::printf("Layer in place (MiB/2 s): eligible %llu, unknown bytes %llu, dynamic %llu, "
                        "not a whole chunk block %llu; unknown bytes not watched: watchable %llu, "
                        "GPU-dynamic %llu, volatile %llu, failed %llu, mapping %llu, written %llu\n",
                        (unsigned long long)(in_place_why[0].exchange(0) >> 20),
                        (unsigned long long)(in_place_why[1].exchange(0) >> 20),
                        (unsigned long long)(in_place_why[2].exchange(0) >> 20),
                        (unsigned long long)(in_place_why[3].exchange(0) >> 20),
                        (unsigned long long)(unwatched_why[0].exchange(0) >> 20),
                        (unsigned long long)(unwatched_why[1].exchange(0) >> 20),
                        (unsigned long long)(unwatched_why[2].exchange(0) >> 20),
                        (unsigned long long)(unwatched_why[3].exchange(0) >> 20),
                        (unsigned long long)(unwatched_why[4].exchange(0) >> 20),
                        (unsigned long long)(unwatched_why[5].exchange(0) >> 20));
        }
        // BB_RESIDENCY_MIX=2: also the bindings in place or mixed with the most bytes, and per
        // range its blocks in VRAM, eligible, dynamic and with unannounced bytes.
        static const bool mix_top = bb_env("BB_RESIDENCY_MIX")[0] == '2';
        if (mix_top && (resolution.kind == Kind::InPlace || resolution.kind == Kind::Mixed)) {
            static std::map<std::tuple<VAddr, u64, bool, int>, u64> top;
            static u32 top_printed = 0;
            top[{address, size, is_written, int(resolution.kind)}] += size;
            const u32 second = BbStats::coarse_second.load(std::memory_order_relaxed);
            if (second - top_printed >= 4) {
                top_printed = second;
                std::vector<std::pair<u64, std::tuple<VAddr, u64, bool, int>>> sorted;
                for (const auto& [key, total] : top) {
                    sorted.emplace_back(total, key);
                }
                std::ranges::sort(sorted, std::greater{});
                for (size_t i = 0; i < std::min<size_t>(sorted.size(), 12); ++i) {
                    const auto& [total, key] = sorted[i];
                    const auto& [a, s, w, kind] = key;
                    u64 vram = 0, eligible = 0, dynamic = 0, unknown = 0;
                    for (u64 b = a >> block_shift; b <= (a + s - 1) >> block_shift; ++b) {
                        if (Layer().AnyValid(b, b + 1)) {
                            ++vram;
                        } else if (dynamic_blocks.Contains(b)) {
                            ++dynamic;
                        } else if (LayerEligible(b)) {
                            ++eligible;
                        } else {
                            ++unknown;
                        }
                    }
                    std::printf("Layer top: %#llx+%#llx %s %s, %llu MiB in 4 s; blocks: VRAM %llu, "
                                "eligible %llu, dynamic %llu, unannounced %llu\n",
                                (unsigned long long)a, (unsigned long long)s,
                                kind == int(Kind::Mixed) ? "mixed" : "in place",
                                w ? "written" : "read", (unsigned long long)(total >> 20),
                                (unsigned long long)vram, (unsigned long long)eligible,
                                (unsigned long long)dynamic, (unsigned long long)unknown);
                }
                top.clear();
            }
        }
        count[k].fetch_add(1, std::memory_order_relaxed);
        bytes[k].fetch_add(size, std::memory_order_relaxed);
        const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
        if (now - printed.load(std::memory_order_relaxed) >= 2) {
            printed.store(now, std::memory_order_relaxed);
            std::string text;
            constexpr std::array<const char*, 4> names = {"none", "in place", "mirror", "mixed"};
            for (size_t i = 0; i < count.size(); ++i) {
                text += fmt::format(" {}{} {}/{}MiB", names[i / 2], (i & 1) ? " w" : "",
                                    count[i].exchange(0), bytes[i].exchange(0) >> 20);
            }
            std::printf("Layer bindings (2 s):%s\n", text.c_str());
        }
    }
    // As the arena gave VRAM to blocks of known data when first bound: a range all of known data
    // (or in VRAM already) is made one mirror now (the missing blocks copied from the game's
    // memory first, in stream order), then bound there. A few blocks with unannounced bytes
    // among many known ones (where the arena mixed VRAM and the game's memory) go into the
    // mirror as volatile blocks, refreshed for every binding that reads them. Not for a range the
    // GPU writes: copying its bytes back after the draw would also put back stale copies of what
    // the CPU (a frame ahead) wrote there meanwhile (exploding vertices).
    // Experiment bit 8 of BB_TOGGLE_FILE (diagnostics, while the game runs): reads bound in place
    // even where a VRAM copy is (a stale copy then shows by going away; nothing is moved).
    if (BbToggle::Experiment(8) && !is_written && resolution.kind == Kind::Mirror) {
        if (const auto in_place = LayerInPlace(address, size, true)) {
            return in_place;
        }
    }
    const bool vram_allowed =
        !(is_written && (force_writes_in_place || !LayerGpuWriteToVram(address, size))) &&
                              !BbToggle::Experiment(8);
    // Not for image uploads (may_promote false): the image is the GPU's copy of that data, read
    // once per upload, and the arena did not move it either. (Loaded data mostly gets its mirror
    // when loaded, ProcessPendingAssets: this alone did not shrink the mirrors.)
    if (resolution.kind != Kind::Mirror && vram_allowed && may_promote && GarlicInVramForLayer() &&
        !VramPromotionsPaused() && !LayerRecentlyDemoted(first, end)) {
        constexpr u64 VolatileMinBlocks = 4, VolatileMaxBlocks = 2;
        // A range found not movable is looked at again next frame or when something moved
        // (what makes blocks known data changes between frames: loads, GPU writes).
        struct TriedMemo {
            u64 first = 0, end = 0, generation = 0;
            u32 frame = 0;
            bool written = false;
        };
        thread_local std::array<TriedMemo, 256> tried{};
        auto& t = tried[(first ^ (end << 7)) & (tried.size() - 1)];
        const u32 frame = static_cast<u32>(BbStats::frame_number.load(std::memory_order_relaxed));
        const bool skip = t.first == first && t.end == end && t.generation == layer_generation &&
                          t.frame == frame && t.written == is_written;
        // Blocks with unannounced bytes are watched copies for a reading binding (BB_LAYER_WATCH);
        // for a writing one only its first and last block when it covers them in part (the bytes
        // beside it: layer_edge_blocks).
        std::vector<u64> blocks, unknown, watch, edges;
        for (u64 block = first; !skip && block < end && unknown.size() <= VolatileMaxBlocks;
             ++block) {
            if (!Layer().AnyValid(block, block + 1)) {
                const bool partial = (block << block_shift) < address ||
                                     ((block + 1) << block_shift) > address + size;
                if (LayerEligible(block)) {
                    blocks.push_back(block);
                } else if (!is_written && LayerWatchable(block)) {
                    blocks.push_back(block);
                    watch.push_back(block);
                } else if (is_written && partial && LayerWatchable(block)) {
                    blocks.push_back(block);
                    edges.push_back(block);
                } else {
                    unknown.push_back(block);
                }
            }
        }
        if (!skip) {
            t = {first, end, layer_generation, frame, is_written};
        }
        if (!skip && (unknown.empty() || (!is_written && end - first >= VolatileMinBlocks &&
                                unknown.size() <= VolatileMaxBlocks && LayerVolatileAllowed() &&
                                !BbToggle::Experiment(16)))) {
            for (const u64 block : watch) {
                layer_watched.Add({block, block + 1});
            }
            for (const u64 block : edges) {
                layer_edge_blocks.Add({block, block + 1});
            }
            LayerPromote(blocks);
            for (const u64 block : edges) {
                if (!Layer().AnyValid(block, block + 1)) {
                    layer_edge_blocks.Subtract(block, block + 1); // not moved (VRAM short)
                }
            }
            for (const u64 block : watch) {
                if (Layer().AnyValid(block, block + 1)) {
                    LayerMarkCpuBlocks(block, block + 1); // the GPU writes it in place
                } else {
                    layer_watched.Subtract(block, block + 1); // not moved (VRAM short)
                }
            }
            LayerMakeVolatile(unknown);
            resolution = Layer().Resolve(address, size);
        }
    }
    if (is_written && resolution.kind == Kind::Mirror && layer_volatile.Overlaps(first, end)) {
        // Volatile blocks go back in place (nothing to copy back: the GPU never writes them in
        // a mirror); the range is mixed now.
        std::vector<std::pair<u64, u64>> runs;
        layer_volatile.ForEachInRange(first, end, [&](const Interval& iv) {
            runs.emplace_back(std::max(first, iv.start), std::min(end, iv.end));
        });
        for (const auto& [a, b] : runs) {
            LayerDemote(a, b, false, 1);
        }
        resolution = Layer().Resolve(address, size);
    }
    // A read over volatile blocks while a render pass is open: in place. Their refresh is a copy,
    // and a copy ends the pass: the same pass was stored and loaded again up to ~100 times a frame
    // (light volumes, one draw each), the GPU 1.3-1.5x slower for the whole launch in the launches
    // that made such blocks. Outside a pass (dispatches, the first draw) the mirror is refreshed.
    // BB_LAYER_VOLATILE_IN_PASS=0: refreshed in passes too (as before).
    static const bool volatile_in_place = !bb_env_off("BB_LAYER_VOLATILE_IN_PASS");
    if (resolution.kind == Kind::Mirror && vram_allowed && !is_written && volatile_in_place &&
        scheduler.IsRendering() && layer_volatile.Overlaps(first, end)) {
        if (const auto in_place = LayerInPlace(address, size, true)) {
            BbStats::bound_in_place_bytes.fetch_add(size, std::memory_order_relaxed);
            layer_volatile_in_pass_bytes += size;
            std::pair<const Buffer*, u64> vram_copy{};
            if (is_texel_buffer) {
                SynchronizeMemoryFromImage(in_place->first, in_place->second, address,
                                           static_cast<u32>(size), &vram_copy);
            }
            TraceBinding(address, size, false, 2);
            if (vram_copy.first) {
                return vram_copy; // the image tiled in VRAM (SynchronizeMemoryFromImage)
            }
            return in_place;
        }
    }
    if (resolution.kind == Kind::Mirror && vram_allowed) {
        auto* mirror = static_cast<Buffer*>(resolution.span.source->owner);
        BbStats::bound_vram_bytes.fetch_add(size, std::memory_order_relaxed);
        NoteBlocksBound(first, end);
        NoteDirectUse(address, size); // stale copies yield first (LayerYieldToImages)
        if (layer_volatile.Overlaps(first, end)) {
            // Experiment bit 16 (diagnostics, while the game runs): no refreshes, the range read
            // in place instead.
            if (BbToggle::Experiment(16) && !is_written) {
                if (const auto in_place = LayerInPlace(address, size, true)) {
                    return in_place;
                }
            }
            LayerRefreshVolatile(address, size);
        }
        if (!is_written && LayerVerifyOn()) {
            LayerVerifyMirror(address, size, is_texel_buffer);
        }
        // Read before the upload check: a CPU change during it makes the memo stale at once.
        const u64 epoch = BbMem::cpu_change_epoch.load(std::memory_order_acquire);
        // Uploads what the CPU changed there since (announced writes), images aliasing it.
        SynchronizeMemory(mirror, address, static_cast<u32>(size), is_written, is_texel_buffer);
        if (!is_written && !is_texel_buffer && may_promote && !layer_volatile.Overlaps(first, end)) {
            StoreBindMemo(address, size, epoch, mirror, mirror->Offset(address), true);
        }
        if (is_written) {
            // The GPU's data is in VRAM only: copied back before the blocks go in place.
            gpu_modified_ranges.Add(address, size);
            NoteBindingWriteTick(address, size);
            LayerArmReadTraps(address, size); // and before the CPU reads it
        }
        TraceBinding(address, size, is_written, is_written ? 1 : 4);
        return std::pair<const Buffer*, u64>{mirror, mirror->Offset(address)};
    }
    if (resolution.kind == Kind::Mirror || resolution.kind == Kind::Mixed) {
        // One buffer range must hold it: in place. Ranges over nearly all memory in shaders are
        // paged instead (LayerPagedRecord). Written: its blocks in VRAM go back in place (the
        // GPU's data there copied back first). Read: only the GPU's data is copied back; the
        // blocks stay in VRAM for the bindings that lie whole in their mirror.
        if (is_written) {
            LayerDemote(first, end, true, resolution.kind == Kind::Mirror ? 0 : 1);
            // The GPU writes this range in place: its blocks stay in place (moving them back
            // would only bring them here again; the arena mixed them). All in one mirror: from
            // the second time (a reading binding brought it back every frame meanwhile: ~200 MiB/s
            // each way and traps set and cleared, the recording thread saturated at 42 FPS).
            if (resolution.kind == Kind::Mixed || layer_write_bounces.Overlaps(first, end)) {
                dynamic_blocks.Add({first, end});
                const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
                for (u64 block = first; block < end; ++block) {
                    layer_dynamic_second[block] = now;
                }
            } else {
                layer_write_bounces.Add({first, end});
            }

        } else {
            LayerCopyBack(first, end);
        }
    }
    const bool pure_in_place = resolution.kind == Kind::InPlace;
    const u64 epoch = BbMem::cpu_change_epoch.load(std::memory_order_acquire);
    // In place (huge bindings over several chunks are cut at the end of the first).
    const auto in_place = LayerInPlace(address, size, true);
    if (!in_place) {
        return std::nullopt;
    }
    // Its blocks of known data go to VRAM at the next submission (as the arena gave VRAM to such
    // blocks when first bound): once per range while the residency is unchanged.
    if (may_promote) {
        LayerQueuePromotion(first, end);
    }
    const auto [buffer, offset] = *in_place;
    if (is_written) {
        BbFreeCheck::NoteGpuWrite(address, size, 0, BbFreeCheck::InPlaceWrite);
        NoteBindingWriteTick(address, size);
        BbStats::bound_in_place_written_bytes.fetch_add(size, std::memory_order_relaxed);
        // What the GPU writes often moves to VRAM once its blocks are all known data
        // (NoteGpuWrite: written in 3 of 60 frames). Some writers stay in place (the copy
        // shader's destinations: force_writes_in_place).
        if (!force_writes_in_place) {
            NoteGpuWrite(address, size);
        }
    }
    BbStats::bound_in_place_bytes.fetch_add(size, std::memory_order_relaxed);
    std::pair<const Buffer*, u64> vram_copy{};
    if (is_texel_buffer && !is_written) {
        SynchronizeMemoryFromImage(buffer, offset, address, static_cast<u32>(size), &vram_copy);
    }
    // Small constants stay out (ObtainBuffer copies them when not GPU-written: decode order).
    if (!is_written && !is_texel_buffer && may_promote && pure_in_place &&
        (size > STREAM_THRESHOLD || ConstantsInPlace())) {
        StoreBindMemo(address, size, epoch, buffer, offset, false);
    }
    TraceBinding(address, size, is_written, is_written ? 0 : 2);
    if (vram_copy.first) {
        return vram_copy; // the image tiled in VRAM (SynchronizeMemoryFromImage)
    }
    return in_place;
}

bool BufferCache::BindMemoOn() {
    static const bool on = !bb_env_off("BB_BIND_MEMO");
    return on;
}

BufferCache::BindMemo& BufferCache::BindMemoSlot(VAddr address, u64 size) {
    // On the heap: static TLS is reserved on every thread's stack, the game's small ones too.
    thread_local auto memo = std::make_unique<std::array<BindMemo, 4096>>();
    return (*memo)[((address * 0x9E3779B97F4A7C15ull) ^ (size * 0xC2B2AE3D27D4EB4Full)) >> 52];
}

void BufferCache::StoreBindMemo(VAddr address, u64 size, u64 epoch, const Buffer* buffer,
                                u64 offset, bool vram) {
    if (!BindMemoOn()) {
        return;
    }
    BindMemoSlot(address, size) = {
        .address = address,
        .size = size,
        .generation = layer_generation,
        .epoch = epoch,
        .offset = offset,
        .buffer = buffer,
        .second = BbStats::coarse_second.load(std::memory_order_relaxed),
        .vram = vram,
    };
}

bool BufferCache::LayerEligible(u64 block) {
    // Every byte last written by a loader or by the GPU (writes we hear of), in a guest chunk.
    const VAddr address = block << block_shift;
    if (dynamic_blocks.Contains(block)) {
        return false;
    }
    bool known = true;
    asset_bytes.ForEachGap(address, address + block_size, [&](u64 start, u64 end) {
        known = known && gpu_written_bytes.Contains(start, end) &&
                LayerGpuWriteToVram(start, end - start);
    });
    return known && Layer().ResolveInPlace(address, block_size).has_value();
}

namespace {
std::array<u64, 5> copy_gaps{}; // copy shader calls by work since the previous one: 0, 1, 2-4, 5-16, 17+
}

void BufferCache::NoteCopyGap(u64 gap) {
    ++copy_gaps[gap == 0 ? 0 : gap == 1 ? 1 : gap <= 4 ? 2 : gap <= 16 ? 3 : 4];
}

void BufferCache::NoteCopyWhere(const std::vector<std::pair<unsigned long long, unsigned long long>>& src,
                                const std::vector<std::pair<unsigned long long, unsigned long long>>& dst,
                                u32 groups) {
    static const bool on = bb_env_on("BB_COPY_WHERE");
    if (!on) {
        return;
    }
    // Per second: the copy shader's source and destination bytes (its list's ranges, 64 KiB
    // merged) by where the GPU finds them, and how its destinations repeat.
    static std::array<u64, 8> bytes{}; // src: VRAM, host, GPU-modified, dst-of-earlier; dst: VRAM, host, rewritten, -
    static u64 calls = 0, total_groups = 0, frames_start = 0, last_frame = 0;
    static std::vector<std::pair<u64, u64>> frame_dst; // destinations this frame
    static u32 printed = 0;
    const u64 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    if (frames_start == 0) {
        frames_start = frame;
    }
    if (frame != last_frame) {
        frame_dst.clear();
        last_frame = frame;
    }
    const auto overlaps_frame_dst = [&](u64 a, u64 s) {
        for (const auto& [b, t] : frame_dst) {
            if (a < b + t && b < a + s) {
                return true;
            }
        }
        return false;
    };
    ++calls;
    total_groups += groups;
    for (const auto& [a, s] : src) {
        const u64 first = a >> block_shift, end = ((a + s - 1) >> block_shift) + 1;
        bytes[Layer().AllValid(first, end) ? 0 : 1] += s;
        bytes[2] += IsRegionGpuModified(a, s) ? s : 0;
        bytes[3] += overlaps_frame_dst(a, s) ? s : 0;
    }
    for (const auto& [a, s] : dst) {
        const u64 first = a >> block_shift, end = ((a + s - 1) >> block_shift) + 1;
        bytes[Layer().AnyValid(first, end) ? 4 : 5] += s;
        bytes[6] += overlaps_frame_dst(a, s) ? s : 0;
    }
    for (const auto& r : dst) {
        frame_dst.emplace_back(r.first, r.second);
    }
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed < 2) {
        return;
    }
    printed = now;
    const double f = double(std::max<u64>(frame - frames_start, 1));
    std::printf("Copy where (KiB/frame): src VRAM %.0f host %.0f GPU-modified %.0f from-dst %.0f; "
                "dst VRAM %.0f host %.0f rewritten-in-frame %.0f; %.1f calls/frame, %.1f groups/call\n",
                bytes[0] / 1024.0 / f, bytes[1] / 1024.0 / f, bytes[2] / 1024.0 / f, bytes[3] / 1024.0 / f,
                bytes[4] / 1024.0 / f, bytes[5] / 1024.0 / f, bytes[6] / 1024.0 / f, calls / f,
                calls ? double(total_groups) / double(calls) : 0.0);
    std::printf("Copy where: calls by draws/dispatches since the previous call (per frame): "
                "0 %.1f, 1 %.1f, 2-4 %.1f, 5-16 %.1f, 17+ %.1f\n",
                copy_gaps[0] / f, copy_gaps[1] / f, copy_gaps[2] / f, copy_gaps[3] / f,
                copy_gaps[4] / f);
    copy_gaps = {};
    bytes = {};
    calls = total_groups = 0;
    frames_start = frame;
}

void BufferCache::NoteVertexInPlace(VAddr address, u64 size) {
    static const bool on = bb_env("BB_VERTEX_IN_PLACE") &&
                           std::strtoull(bb_env("BB_VERTEX_IN_PLACE"), nullptr, 0) != 0;
    if (!on || size == 0) {
        return;
    }
    // Per frame: each block once, by the first reason that applies.
    static std::unordered_map<u64, u64> seen; // block -> frame
    static std::array<u64, 8> blocks{};       // reasons summed over the window
    static std::map<u64, u64> groups;         // 2 MiB group -> blocks over the window
    static u64 frames_start = 0;
    static u32 printed = 0;
    const u64 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    if (frames_start == 0) {
        frames_start = frame;
    }
    static std::map<std::pair<VAddr, u64>, u64> ranges; // bound ranges over the window
    ranges[{address, size}] += 1;
    for (u64 block = address >> block_shift; block <= (address + size - 1) >> block_shift; ++block) {
        auto [it, inserted] = seen.try_emplace(block, frame);
        if (!inserted && it->second == frame) {
            continue;
        }
        it->second = frame;
        const int why = Layer().AnyValid(block, block + 1)          ? 0 // mixed: VRAM copy elsewhere
                        : layer_cpu_hot.Contains(block, block + 1)    ? 1
                        : dynamic_blocks.Contains(block)              ? 2
                        : layer_watch_failed.Contains(block, block + 1) ? 3
                        : layer_cpu_blocks.Contains(block, block + 1) ? 4
                        : LayerEligible(block)                        ? 5
                        : LayerWatchable(block)                       ? 6
                                                                      : 7;
        ++blocks[why];
        ++groups[(block << block_shift) >> 21];
    }
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed < 2) {
        return;
    }
    printed = now;
    const double f = double(std::max<u64>(frame - frames_start, 1));
    const double kib = double(block_size >> 10) / f;
    std::printf("Vertex in place (KiB/frame, unique blocks): VRAM-mixed %.0f, CPU-hot %.0f, "
                "dynamic %.0f, watch failed %.0f, CPU-written %.0f, eligible %.0f, watchable %.0f, "
                "other %.0f\n",
                blocks[0] * kib, blocks[1] * kib, blocks[2] * kib, blocks[3] * kib, blocks[4] * kib,
                blocks[5] * kib, blocks[6] * kib, blocks[7] * kib);
    std::vector<std::pair<u64, u64>> top;
    for (const auto& [g, n] : groups) {
        top.emplace_back(n, g);
    }
    std::ranges::sort(top, std::greater{});
    std::string text;
    for (size_t i = 0; i < std::min<size_t>(top.size(), 10); ++i) {
        text += fmt::format(" {:#x}:{:.0f}", top[i].second << 21, top[i].first * kib);
    }
    std::printf("Vertex in place by 2 MiB (KiB/frame):%s\n", text.c_str());
    // The biggest ranges, and what their blocks are: why LayerBind does not move them.
    {
        std::vector<std::pair<u64, std::pair<VAddr, u64>>> by_bytes;
        // BB_VERTEX_IN_PLACE=0x...: only ranges at or above that address (one region).
        static const u64 region = std::strtoull(bb_env("BB_VERTEX_IN_PLACE"), nullptr, 0);
        for (const auto& [key, n] : ranges) {
            if (region > 1 && (key.first < region || key.first >= region + (256ull << 20))) {
                continue;
            }
            by_bytes.emplace_back(key.second * n, key);
        }
        std::ranges::sort(by_bytes, std::greater{});
        for (size_t i = 0; i < std::min<size_t>(by_bytes.size(), 6); ++i) {
            const auto [a, s] = by_bytes[i].second;
            u32 n = 0, watchable = 0, failed = 0, hot = 0, dyn = 0, vol = 0, watched = 0, vram = 0,
                strikes = 0;
            for (u64 b = a >> block_shift; b <= (a + s - 1) >> block_shift; ++b, ++n) {
                vram += Layer().AnyValid(b, b + 1);
                watchable += LayerWatchable(b);
                failed += layer_watch_failed.Contains(b, b + 1);
                hot += layer_cpu_hot.Contains(b, b + 1);
                dyn += dynamic_blocks.Contains(b);
                vol += layer_volatile.Contains(b, b + 1);
                watched += layer_watched.Contains(b, b + 1);
                if (const auto it = layer_watch_strikes.find(b); it != layer_watch_strikes.end()) {
                    strikes += it->second;
                }
            }
            int prot = 0, type = -1;
            uintptr_t vma_end = 0;
            const bool vma = runtime_memory_vma_info(a & ~(block_size - 1), &prot, &type, &vma_end);
            std::printf("Vertex in place range %#llx+%#llx x%llu: %u blocks: VRAM %u, watchable %u, "
                        "failed %u, CPU-hot %u, dynamic %u, volatile %u, watched %u, strikes %u; "
                        "recently demoted %d; vma %d prot %#x type %d end %#llx; in place %d\n",
                        (unsigned long long)a, (unsigned long long)s,
                        (unsigned long long)ranges[{a, s}], n, vram, watchable, failed, hot, dyn, vol,
                        watched, strikes,
                        LayerRecentlyDemoted(a >> block_shift, ((a + s - 1) >> block_shift) + 1) ? 1 : 0,
                        vma ? 1 : 0, prot, type, (unsigned long long)vma_end,
                        Layer().ResolveInPlace(a & ~(block_size - 1), block_size).has_value() ? 1 : 0);
        }
        ranges.clear();
    }
    // What the CPU really changes in those blocks: 4 KiB pages hashed now, again ~a frame later,
    // and compared with the hashes of 2 s ago.
    static std::unordered_map<u64, u64> page_hash; // page -> hash at the last print
    std::vector<u64> recent;
    for (const auto& [block, seen_frame] : seen) {
        if (frame - seen_frame <= 2) {
            recent.push_back(block);
        }
    }
    std::vector<u8> data(block_size);
    std::unordered_map<u64, u64> now_hash;
    const auto hash_all = [&](std::unordered_map<u64, u64>& out) {
        for (const u64 block : recent) {
            memory->CopySparseMemory(block << block_shift, data.data(), block_size);
            for (u64 p = 0; p < block_size; p += 4096) {
                out[((block << block_shift) + p) >> 12] = XXH3_64bits(data.data() + p, 4096);
            }
        }
    };
    const auto t0 = std::chrono::steady_clock::now();
    hash_all(now_hash);
    const auto t1 = std::chrono::steady_clock::now();
    u64 changed_2s = 0, known = 0;
    for (const auto& [page, h] : now_hash) {
        if (const auto it = page_hash.find(page); it != page_hash.end()) {
            ++known;
            changed_2s += it->second != h;
        }
    }
    // Again after ~one frame (7-20 ms).
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    std::unordered_map<u64, u64> later_hash;
    hash_all(later_hash);
    u64 changed_frame = 0;
    for (const auto& [page, h] : later_hash) {
        changed_frame += now_hash[page] != h;
    }
    std::printf("Vertex in place: %zu blocks (%llu pages); pages changed in ~10 ms+hash: %llu "
                "(%.0f KiB), in 2 s: %llu of %llu; hashing took %.1f ms\n",
                recent.size(), (unsigned long long)now_hash.size(),
                (unsigned long long)changed_frame, changed_frame * 4.0,
                (unsigned long long)changed_2s, (unsigned long long)known,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
    page_hash = std::move(now_hash);
    blocks = {};
    groups.clear();
    frames_start = frame;
    if (seen.size() > (1u << 20)) {
        seen.clear();
    }
}

bool BufferCache::LayerWatchEnabled() {
    // BB_LAYER_WATCH=1/0 (default 1). The reads over the bus cost far more on NVIDIA's imported
    // host memory than on AMD's dma-buf (GTX 1660 Ti: 30-35 FPS with the module, 60-70 with the
    // model of 0.3; RX 7800 XT: on par). It also replaces most volatile blocks, refreshed for
    // every binding (RX 7800 XT: 1.3 GB/s of copies -> 36 MB/s). Needs the write traps
    // (BB_LAYER_WRITE_TRAPS).
    static const bool on = [] {
        const bool enabled = !bb_env_off("BB_LAYER_WATCH");
        if (enabled && LayerTrapsOn()) {
            std::printf("Guest memory: blocks the CPU writes unannounced get VRAM copies too, "
                        "uploaded again after its next write (BB_LAYER_WATCH=1)\n");
        }
        return enabled;
    }();
    return on && LayerTrapsOn();
}

bool BufferCache::LayerWatchHot() {
    // BB_LAYER_WATCH_HOT=1 (experiment): blocks the CPU writes often (NoteCpuWrite: in 3 of 60
    // frames) are watched too and uploaded again after every write, as the model of 0.3 uploads
    // them. Off: they go back in place. Uploading them before bindings ended render passes
    // (RX 7800 XT: the GPU's frame 4.8 -> 7.2 ms).
    static const bool on = bb_env_on("BB_LAYER_WATCH_HOT");
    return on;
}

bool BufferCache::LayerWatchable(u64 block) {
    // Dynamic for the GPU's writes (in place in a mixed range): never; for the CPU's (often
    // written) only with BB_LAYER_WATCH_HOT=1.
    const bool dynamic = dynamic_blocks.Contains(block) &&
                         !(LayerWatchHot() && layer_cpu_hot.Contains(block, block + 1));
    if (!LayerWatchEnabled() || dynamic || layer_volatile.Contains(block, block + 1) ||
        layer_watch_failed.Contains(block, block + 1)) {
        return false;
    }
    const VAddr address = block << block_shift;
    if (!Layer().ResolveInPlace(address, block_size)) {
        return false; // no source to upload from
    }
    // A trap must be able to catch every CPU write: the CPU cannot write it, or one mapping
    // holds the whole block (LayerArmTraps).
    int prot = 0, type = -1;
    uintptr_t vma_end = 0;
    if (!runtime_memory_vma_info(address, &prot, &type, &vma_end)) {
        return false;
    }
    return !(prot & 0x2) || vma_end >= address + block_size;
}

void BufferCache::LayerRefreshWatched(u64 block) {
    // The CPU wrote it (a trap hit): uploaded again before the next binding over it (the bound
    // path's SynchronizeMemory; paged bindings: LayerFlushUploads), the trap set again for its
    // next write. The GPU never writes it in VRAM: nothing of the GPU's to keep.
    const VAddr from = block << block_shift;
    memory_tracker->UnmarkRegionAsGpuModified(from, block_size);
    gpu_modified_ranges.Subtract(from, block_size);
    memory_tracker->MarkRegionAsCpuModified(from, block_size);
    {
        std::scoped_lock lk{layer_uploads_mutex};
        layer_pending_uploads.emplace_back(from, block_size);
        layer_uploads_pending.store(true, std::memory_order_release);
    }
    LayerArmTraps(block, block + 1);
    ++layer_watch_refreshes;
}

} // namespace VideoCore
