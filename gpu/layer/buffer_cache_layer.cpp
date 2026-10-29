// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport BB_LAYER_MEMORY: the BufferCache side of the layer's memory module
// (gpu/layer/bblayer_gpu_memory.h, docs/MEMORY_MODULE_PLAN.ru.md). The game's memory is bound in
// place through its chunk buffers; blocks of known data (what loaders wrote) get VRAM copies in
// mirrors: one buffer per run of such blocks, runs that meet merged into a new mirror. No sparse
// binding: a move is a copy plus the module's bookkeeping, and bindings pick the source.
//
// Step 3a: only loaded data is copied to VRAM, and a range the GPU writes goes back in place
// first, so the game's memory always holds the current data and readers in place (the huge
// bindings over all memory) stay right.

#include <algorithm>
#include "shim/bb_env.h"
#include <bit>
#include <array>
#include <atomic>
#include <mutex>
#include <unordered_set>
#include <sys/mman.h>
#include "common/signal_context.h"
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

void BufferCache::LayerPromote(const std::vector<u64>& blocks) {
    BbStats::Timer timer{BbStats::t_layer[1]};
    // A budget of copies a frame (BB_PROMOTE_MB_PER_FRAME; 0: none; default 32 on cards with a budget
    // up to 4 GB, 256 above): after a load or a walk into a new area the GPU of the GTX 1050 laptop
    // spent 10-60 ms a frame on copies into VRAM for ~20 s (the frame counter it was counted by was
    // the flips until 2026-10-26, so 256 never limited anything). A level's first frame
    // copied hundreds of MiB of loaded data into VRAM at once (300 ms of "layer bind" on the GTX
    // 1050 laptop, the bus full). Past it, runs are read in place this frame and stay candidates
    // for the next ones (the first run of a frame always goes).
    static const u64 frame_budget = [this] {
        const char* env = bb_env("BB_PROMOTE_MB_PER_FRAME");
        const u64 def = instance.GetTotalMemoryBudget() <= (u64{4} << 30) ? 32 : 256;
        return u64(env && *env ? std::strtoull(env, nullptr, 10) : def) << 20;
    }();
    static u64 budget_frame = ~0ULL, spent = 0, deferred_bytes = 0;
    const u64 frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    if (frame != budget_frame) {
        budget_frame = frame;
        spent = 0;
    }
    const auto promote_run = [&](u64 first, u64 end) -> u64 {
        const u64 bytes = (end - first) << block_shift;
        if (frame_budget != 0 && spent != 0 && spent + bytes > frame_budget) {
            promote_candidates.Add({first, end});
            deferred_bytes += bytes;
            return 0;
        }
        spent += bytes;
        return LayerPromoteRun(first, end);
    };
    u64 moved = 0;
    for (size_t i = 0; i < blocks.size();) {
        // A run of consecutive candidate blocks.
        u64 a = blocks[i], b = a + 1;
        for (++i; i < blocks.size() && blocks[i] == b; ++i) {
            ++b;
        }
        // Only blocks still eligible and not in VRAM yet; then their run.
        u64 run_first = ~0ULL, run_end = 0;
        for (u64 block = a; block < b; ++block) {
            if (Layer().AnyValid(block, block + 1) || LayerRecentlyDemoted(block, block + 1) ||
                !(LayerEligible(block) || layer_watched.Contains(block, block + 1) ||
                  layer_edge_blocks.Contains(block, block + 1))) {
                if (run_first != ~0ULL) {
                    moved += promote_run(run_first, run_end);
                    run_first = ~0ULL;
                }
                continue;
            }
            if (run_first == ~0ULL) {
                run_first = block;
            }
            run_end = block + 1;
        }
        if (run_first != ~0ULL) {
            moved += promote_run(run_first, run_end);
        }
    }
    if (moved != 0) {
        static u64 promoted = 0;
        if (((promoted + moved) >> 26) != (promoted >> 26)) {
            std::printf("Layer memory: %llu MiB of loaded blocks copied to VRAM, %llu MiB valid "
                        "there in %zu mirrors\n",
                        (unsigned long long)((promoted + moved) >> 20),
                        (unsigned long long)(Layer().ValidBytes() >> 20), layer_mirrors.size());
        }
        promoted += moved;
    }
    static u64 reported_deferred = 0;
    if ((deferred_bytes >> 28) != (reported_deferred >> 28)) {
        reported_deferred = deferred_bytes;
        std::printf("Layer memory: %llu MiB of copies to VRAM left for later frames (budget %llu "
                    "MiB a frame)\n",
                    (unsigned long long)(deferred_bytes >> 20),
                    (unsigned long long)(frame_budget >> 20));
    }
}

/// Promoted runs: in a domain, in a domain but over a neighbour's mirror (merged as before), in no
/// domain (Layer memory lines, every 10 s).
std::array<std::atomic<u64>, 3> layer_domain_runs{};
std::array<std::atomic<u64>, 4096> layer_none_buckets{}; ///< blocks promoted in no domain, by 64 MiB

void BufferCache::LayerDomainAdd(VAddr addr, u64 size) {
    std::scoped_lock lk{layer_domains_mutex};
    // Domains the game freed without our hearing of it give way.
    for (auto it = layer_domains.lower_bound(addr); it != layer_domains.begin();) {
        --it;
        if (it->second <= addr) {
            break;
        }
        it = layer_domains.erase(it);
    }
    for (auto it = layer_domains.lower_bound(addr); it != layer_domains.end() && it->first < addr + size;) {
        it = layer_domains.erase(it);
    }
    layer_domains.emplace(addr, addr + size);
}

void BufferCache::LayerDomainDead(VAddr addr, u64 size) {
    // Its images too (BB_DEAD_IMAGES=0: left to the collector), on the GPU thread (ProcessDeadImages).
    static const bool dead_images = !bb_env_off("BB_DEAD_IMAGES");
    if (dead_images) {
        std::scoped_lock lk{pending_unmaps_mutex};
        dead_image_ranges.push_back(
            {addr, size, BbStats::guest_submit_stamp.load(std::memory_order_acquire)});
        dead_images_pending.store(true, std::memory_order_release);
    }
    {
        std::scoped_lock lk{layer_domains_mutex};
        if (const auto it = layer_domains.find(addr); it != layer_domains.end()) {
            layer_domains.erase(it);
        }
    }
    // Whole blocks only: an edge block may hold a live neighbour's data (in VRAM only, perhaps).
    const VAddr a = Common::AlignUp(addr, block_size), b = Common::AlignDown(addr + size, block_size);
    if (a < b) {
        UnmapInPlace(a, b - a);
    }
}

std::optional<std::pair<u64, u64>> BufferCache::LayerDomainOf(u64 first, u64 end) {
    std::scoped_lock lk{layer_domains_mutex};
    auto it = layer_domains.upper_bound(first << block_shift);
    if (it == layer_domains.begin()) {
        return std::nullopt;
    }
    --it;
    const u64 lo = it->first >> block_shift, hi = (it->second + block_size - 1) >> block_shift;
    if (first < lo || end > hi) {
        return std::nullopt;
    }
    return std::pair<u64, u64>{lo, hi};
}

u64 BufferCache::LayerPromoteRun(u64 first, u64 end) {
    // BB_LAYER_DOMAINS=0: off. Mirrors per game allocation (their GPU faults of 2026-10-23 were
    // copies past a mirror's end, ForEachValidPiece).
    static const bool domains_on = !bb_env_off("BB_LAYER_DOMAINS");
    // A run over several domains (a loader wrote neighbouring allocations one after another) is
    // cut at their edges: each part goes to its own domain's mirror. An edge block two domains
    // share goes with the lower one.
    if (domains_on) {
        std::vector<u64> cuts;
        {
            std::scoped_lock lk{layer_domains_mutex};
            auto it = layer_domains.upper_bound(first << block_shift);
            if (it != layer_domains.begin()) {
                --it;
            }
            for (; it != layer_domains.end() && (it->first >> block_shift) < end; ++it) {
                const u64 lo = it->first >> block_shift, hi = (it->second + block_size - 1) >> block_shift;
                if (lo > first && lo < end) cuts.push_back(lo + (cuts.empty() || cuts.back() < lo ? 0 : 0));
                if (hi > first && hi < end) cuts.push_back(hi);
            }
        }
        std::sort(cuts.begin(), cuts.end());
        cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
        if (!cuts.empty()) {
            u64 moved = 0, at = first;
            for (const u64 cut : cuts) {
                if (cut > at) {
                    moved += LayerPromoteOne(at, cut, domains_on);
                    at = cut;
                }
            }
            if (at < end) {
                moved += LayerPromoteOne(at, end, domains_on);
            }
            return moved;
        }
    }
    return LayerPromoteOne(first, end, domains_on);
}

u64 BufferCache::LayerPromoteOne(u64 first, u64 end, bool domains_on) {
    // A run inside one mirror's range (blocks it holds no valid data for): copied into that mirror,
    // nothing made.
    const auto overlapping = Layer().MirrorsIn(first << block_shift, end << block_shift);
    for (const auto& mirror : overlapping) {
        const u64 m_lo = mirror.base >> block_shift, m_hi = (mirror.base + mirror.size) >> block_shift;
        if (m_lo <= first && end <= m_hi) {
            return LayerPromoteInto(first, end, m_lo, m_hi, {mirror});
        }
    }
    // In a domain (a game allocation its profile told of): the mirror stays within it, the whole
    // of it when it is small (a file loaded in pieces then needs no merge), and nothing outside is
    // merged with it; freed, the allocation's mirror goes whole, nothing to copy.
    if (const auto domain = domains_on ? LayerDomainOf(first, end) : std::nullopt) {
        auto [lo, hi] = *domain;
        if (hi - lo > (64_MB >> block_shift)) {
            lo = first;
            hi = end;
        }
        // Mirrors over the edges (a neighbour's, sharing an edge block) are left out.
        std::vector<BbLayer::MemorySource> inside;
        for (const auto& mirror : Layer().MirrorsIn(lo << block_shift, hi << block_shift)) {
            const u64 m_lo = mirror.base >> block_shift, m_hi = (mirror.base + mirror.size) >> block_shift;
            if (m_lo >= domain->first && m_hi <= domain->second) {
                inside.push_back(mirror);
                lo = std::min(lo, m_lo);
                hi = std::max(hi, m_hi);
            } else if (m_hi <= first) {
                lo = std::max(lo, m_hi);
            } else if (m_lo >= end) {
                hi = std::min(hi, m_lo);
            } else {
                inside.clear();
                lo = hi = 0; // the run lies in someone else's mirror: as before
                break;
            }
        }
        if (lo <= first && end <= hi) {
            layer_domain_runs[0].fetch_add(1, std::memory_order_relaxed);
            return LayerPromoteInto(first, end, lo, hi, inside);
        }
        layer_domain_runs[1].fetch_add(1, std::memory_order_relaxed);
    } else if (domains_on) {
        layer_domain_runs[2].fetch_add(1, std::memory_order_relaxed);
        // Where runs in no domain are (64 MiB buckets of the address), for the frame stats line.
        const u64 bucket = (first << block_shift) >> 26;
        layer_none_buckets[bucket % layer_none_buckets.size()].fetch_add(end - first, std::memory_order_relaxed);
    }
    // The mirror range: the run and every mirror touching it (or touching what that adds).
    u64 lo = first, hi = end;
    std::vector<BbLayer::MemorySource> touching;
    for (bool grew = true; grew;) {
        grew = false;
        touching = Layer().MirrorsIn((lo - 1) << block_shift, (hi + 1) << block_shift);
        for (const auto& mirror : touching) {
            const u64 m_lo = mirror.base >> block_shift;
            const u64 m_hi = (mirror.base + mirror.size) >> block_shift;
            if (m_lo < lo || m_hi > hi) {
                lo = std::min(lo, m_lo);
                hi = std::max(hi, m_hi);
                grew = true;
            }
        }
    }
    // Room to grow when a new mirror is made over neighbours (BB_MIRROR_MERGE_GROWTH=0: at the top
    // only, as before): a loader filling an area piece by piece in any order had the mirror made
    // again at every piece (20, 30, 40, 44, 60 MiB within a frame on the 2 GB laptop, 2026-10-26);
    // with room the next pieces land inside it (copied in, nothing made).
    static const bool merge_growth = !bb_env_off("BB_MIRROR_MERGE_GROWTH");
    if (!(touching.size() == 1 && touching[0].base == (lo << block_shift) &&
          touching[0].size == ((hi - lo) << block_shift))) {
        u64 top = 0;
        for (const auto& mirror : touching) {
            top = std::max<u64>(top, (mirror.base + mirror.size) >> block_shift);
        }
        if (!touching.empty() && (end >= top || merge_growth)) {
            hi = LayerGrowthEnd(lo, hi);
        }
    }
    return LayerPromoteInto(first, end, lo, hi, touching);
}

bool BufferCache::LayerMergeAcross(u64 first, u64 end) {
    static const bool domains_on = !bb_env_off("BB_LAYER_DOMAINS");
    if (!domains_on || !Layer().AllValid(first, end)) {
        return false;
    }
    const auto mirrors = Layer().MirrorsIn(first << block_shift, end << block_shift);
    if (mirrors.size() < 2) {
        return false;
    }
    u64 lo = ~u64{0}, hi = 0;
    for (const auto& mirror : mirrors) {
        lo = std::min<u64>(lo, mirror.base >> block_shift);
        hi = std::max<u64>(hi, (mirror.base + mirror.size) >> block_shift);
    }
    if (hi - lo > (256_MB >> block_shift)) {
        return false; // read in place, as before domains for ranges over a mirror's reach
    }
    const VAddr start = lo << block_shift;
    const u64 bytes = (hi - lo) << block_shift;
    BbStats::mirror_new_reason = 1;
    auto merged = LayerNewMirror(start, bytes);
    BbStats::mirror_new_reason = 0;
    if (!merged) {
        return false;
    }
    for (const auto& old : mirrors) {
        auto* old_buffer = static_cast<Buffer*>(old.owner);
        boost::container::small_vector<vk::BufferCopy, 8> copies;
        Layer().ForEachValid(old.base >> block_shift, (old.base + old.size) >> block_shift,
                             [&](u64 a, u64 b) {
                                 copies.push_back({old_buffer->Offset(a << block_shift),
                                                   merged->Offset(a << block_shift),
                                                   (b - a) << block_shift});
                             });
        if (!copies.empty()) {
            runtime.CopyBuffer(old_buffer, merged.get(), copies);
        }
        Layer().RemoveMirror(old.base);
        auto node = layer_mirrors.extract(old.base);
        if (!node.empty()) {
            BbStats::residency_alloc_bytes.fetch_sub(old.size, std::memory_order_relaxed);
            layer_retired.push_back({std::move(node.mapped()), scheduler.CurrentTick()});
        }
    }
    Layer().AddMirror(start, bytes, merged->Handle(), merged->BufferDeviceAddress(), merged.get());
    layer_mirrors.emplace(start, std::move(merged));
    layer_mirror_born[start] = BbStats::coarse_second.load(std::memory_order_relaxed);
    BbStats::residency_alloc_bytes.fetch_add(bytes, std::memory_order_relaxed);
    LayerRepublish(lo, hi); // paged bindings find the merged mirror
    ++layer_generation;
    return true;
}

u64 BufferCache::LayerPromoteInto(u64 first, u64 end, u64 lo, u64 hi,
                                  const std::vector<BbLayer::MemorySource>& touching) {
    // Page table entries that change: the run, and the blocks of mirrors replaced by a new one.
    std::vector<std::pair<u64, u64>> moved;
    Buffer* target = nullptr;
    if (touching.size() == 1 && touching[0].base == (lo << block_shift) &&
        touching[0].size == ((hi - lo) << block_shift)) {
        target = static_cast<Buffer*>(touching[0].owner);
    } else {
        if (!LayerRoomFor((hi - lo) << block_shift)) {
            promote_candidates.Add({first, end});
            return 0;
        }
        // A new mirror over all of it; the old ones' valid blocks are copied into it and they
        // go once the GPU is past the commands recorded so far. One that grows (it takes in a
        // mirror it meets) gets room above it for as much again, up to LayerGrowthBlocks: a
        // loader writing a file in pieces made a new mirror, and a copy of the old one, for each
        // piece (15-27 GiB allocated for 1.3 GiB of data on loading a level, 7.4 GiB of VRAM
        // at once; streaming areas did the same while playing).
        const VAddr start = lo << block_shift;
        const u64 bytes = (hi - lo) << block_shift;
        if (static const bool trace = bb_env("BB_MIRROR_TRACE") != nullptr; trace) {
            u64 old_bytes = 0;
            for (const auto& t : touching) {
                old_bytes += t.size;
            }
            std::printf("Mirror trace: new %#llx+%llu MiB for run %#llx+%llu KiB, replacing %zu (%llu MiB)\n",
                        (unsigned long long)start, (unsigned long long)(bytes >> 20),
                        (unsigned long long)(first << block_shift),
                        (unsigned long long)(((end - first) << block_shift) >> 10), touching.size(),
                        (unsigned long long)(old_bytes >> 20));
        }
        auto mirror = LayerNewMirror(start, bytes);
        if (!mirror) {
            promote_candidates.Add({first, end});
            return 0;
        }
        target = mirror.get();
        for (const auto& old : touching) {
            auto* old_buffer = static_cast<Buffer*>(old.owner);
            moved.emplace_back(old.base >> block_shift, (old.base + old.size) >> block_shift);
            boost::container::small_vector<vk::BufferCopy, 8> copies;
            Layer().ForEachValid(old.base >> block_shift, (old.base + old.size) >> block_shift,
                                 [&](u64 v_first, u64 v_end) {
                                     const VAddr from = v_first << block_shift;
                                     copies.push_back({old_buffer->Offset(from),
                                                       target->Offset(from),
                                                       (v_end - v_first) << block_shift});
                                 });
            if (!copies.empty()) {
                runtime.CopyBuffer(old_buffer, target, copies);
            }
            Layer().RemoveMirror(old.base);
            auto node = layer_mirrors.extract(old.base);
            if (!node.empty()) {
                // Counted off as it goes (the statistic only grew with every merge).
                BbStats::residency_alloc_bytes.fetch_sub(old.size, std::memory_order_relaxed);
                layer_retired.push_back({std::move(node.mapped()), scheduler.CurrentTick()});
            }
        }
        Layer().AddMirror(start, bytes, target->Handle(), target->BufferDeviceAddress(), target);
        layer_mirrors.emplace(start, std::move(mirror));
        layer_mirror_born[start] = BbStats::coarse_second.load(std::memory_order_relaxed);
        BbStats::residency_alloc_bytes.fetch_add(bytes, std::memory_order_relaxed);
    }
    // The run's data: copied from the game's memory now (SynchronizeMemory, which also arms the
    // tracking of later CPU writes there).
    Layer().MarkValid(first, end);
    NoteUse(first << block_shift, (end - first) << block_shift);
    NoteBlocksBound(first, end); // promoted: not unbound at once
    NoteDirectUse(first << block_shift, (end - first) << block_shift); // not stale at once
    const VAddr from = first << block_shift;
    const u64 bytes = (end - first) << block_shift;
    memory_tracker->UnmarkRegionAsGpuModified(from, bytes);
    gpu_modified_ranges.Subtract(from, bytes);
    memory_tracker->MarkRegionAsCpuModified(from, bytes);
    SynchronizeMemory(target, from, static_cast<u32>(bytes), false, false);
    if (LayerVerifyOn()) {
        LayerVerifyForget(first, end); // a fresh copy: compared from its next binding
    }
    LayerArmTraps(first, end); // writes nobody announces send it back in place
    // Paged bindings find the mirror (a merge moved its other blocks too). Not all of it: a
    // mirror of a GiB rewrote 16k entries for every run moved into it (160k in a frame).
    LayerRepublish(first, end);
    for (const auto& [a, b] : moved) {
        LayerRepublish(a, b);
    }
    ++layer_generation;
    return bytes;
}

u64 BufferCache::LayerGrowthEnd(u64 lo, u64 hi) {
    // BB_LAYER_MIRROR_GROWTH_MB (64; 0: none): the most room a growing mirror gets above it.
    static const u64 growth_bytes = [] {
        const char* env = bb_env("BB_LAYER_MIRROR_GROWTH_MB");
        return (env && *env ? std::strtoull(env, nullptr, 10) : 64ull) << 20;
    }();
    const u64 extra = std::min(hi - lo, growth_bytes >> block_shift);
    if (extra == 0) {
        return hi;
    }
    u64 end = hi + extra;
    // Not over the next mirror, and only blocks that can be valid here: the game's memory whole
    // in a chunk, contiguous with the blocks below.
    for (const auto& mirror : Layer().MirrorsIn(hi << block_shift, end << block_shift)) {
        end = std::min(end, mirror.base >> block_shift);
    }
    if (end <= hi) {
        return hi;
    }
    const auto prefix = Layer().ResolvePrefix(hi << block_shift, (end - hi) << block_shift);
    if (!prefix) {
        return hi;
    }
    return std::min(end, hi + (prefix->size >> block_shift));
}

static std::array<std::atomic<u64>, 8> layer_demoted_bytes{};

void BufferCache::LayerDemote(u64 first, u64 end, bool copy_back, int reason) {
    BbStats::Timer timer{BbStats::t_layer[2]};
    LayerForgetGpuData();
    std::vector<std::pair<u64, u64>> runs;
    Layer().ForEachValid(first, end, [&](u64 a, u64 b) { runs.emplace_back(a, b); });
    if (runs.empty()) {
        return;
    }
    for (const auto& [a, b] : runs) {
        const VAddr from = a << block_shift;
        const u64 bytes = (b - a) << block_shift;
        if (copy_back) {
            // What the GPU wrote in VRAM goes back into the game's memory first (each part from
            // the mirror holding it).
            ForEachValidPiece(Layer(), a, b, block_shift,
                              [&](const BbLayer::MemorySource& mirror, u64 pa, u64 pb) {
                auto* buffer = static_cast<Buffer*>(mirror.owner);
                gpu_modified_ranges.ForEachInRange(pa << block_shift, (pb - pa) << block_shift,
                                                   [&](VAddr w_from, VAddr w_to) {
                    if (const auto target = LayerInPlace(w_from, w_to - w_from)) {
                        const vk::BufferCopy copy{buffer->Offset(w_from), target->second,
                                                  w_to - w_from};
                        BbFreeCheck::NoteGpuWrite(w_from, w_to - w_from, 0,
                                                  BbFreeCheck::MirrorCopyBack);
                        runtime.CopyBuffer(buffer, target->first, std::span{&copy, 1});
                    }
                });
            });
        }
        (void)from;
        (void)bytes;
        layer_volatile.Subtract(a, b);
        layer_edge_blocks.Subtract(a, b);
        // A watched block sent back for a GPU write (binding or command) is not watched again (it
        // came back for every binding, ~40 MiB/s while playing; memory handed out anew does not
        // clear it either: the game's allocator recycles some every frame); one sent back for an
        // announced write or a request, after the third time (loading a level sends many back
        // once). Idle or unmapped ones may be again.
        if (reason <= 3) {
            layer_watched.ForEachInRange(a, b, [&](const Interval& iv) {
                for (u64 block = std::max(a, iv.start); block < std::min(b, iv.end); ++block) {
                    if (reason <= 2 || ++layer_watch_strikes[block] >= 3) {
                        layer_watch_failed.Add({block, block + 1});
                    }
                }
            });
        }
        layer_watched.Subtract(a, b);
        LayerDisarmTraps(a, b, false);
        Layer().MarkInvalid(a, b);
        // In place from now on: never uploaded or watched again (writes go where the GPU reads).
        memory_tracker->MarkRegionAsCpuModified(from, bytes);
        layer_demoted_bytes[reason].fetch_add(bytes, std::memory_order_relaxed);
        if (reason >= 6) {
            const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
            if (group_demoted_second.empty()) {
                group_demoted_second.assign(group_use.size(), 0);
            }
            const u64 g_last = std::min<u64>(((b << block_shift) - 1) >> USE_GROUP_BITS,
                                             group_demoted_second.size() - 1);
            for (u64 g = (a << block_shift) >> USE_GROUP_BITS; g <= g_last; ++g) {
                group_demoted_second[g] = now;
            }
        }
        LayerRepublish(a, b); // paged bindings find them in place
        memory_tracker->UnmarkRegionAsGpuModified(from, bytes);
        gpu_modified_ranges.Subtract(from, bytes);
    }
    // Mirrors left without valid blocks go once the GPU is past the commands recorded so far.
    for (const auto& mirror : Layer().MirrorsIn(first << block_shift, end << block_shift)) {
        if (Layer().AnyValid(mirror.base >> block_shift, (mirror.base + mirror.size) >> block_shift)) {
            continue;
        }
        Layer().RemoveMirror(mirror.base);
        auto node = layer_mirrors.extract(mirror.base);
        if (!node.empty()) {
            BbStats::residency_alloc_bytes.fetch_sub(mirror.size, std::memory_order_relaxed);
            layer_retired.push_back({std::move(node.mapped()), scheduler.CurrentTick()});
        }
    }
    ++layer_generation;
}

namespace {
/// A mirror's allocation size: rounded up to an eighth of its power of two (at most 12.5 % more),
/// so mirrors of nearby sizes share pooled allocations.
u64 MirrorSizeClass(u64 bytes) {
    const u64 step = std::max<u64>(std::bit_floor(std::max<u64>(bytes, 1)) / 8, 64_KB);
    return (bytes + step - 1) / step * step;
}

u64 MirrorPoolCap(u64 budget) {
    // BB_MIRROR_POOL_MB: retired mirrors kept for new ones (each one a dedicated allocation:
    // the 2 GB laptop allocated and freed 9-13 GB a minute, frames with 300-570 MB of it took
    // 60-100 ms, 2026-10-25). Default an eighth of the budget, at most 256 MiB; 0: off.
    static const s64 forced = [] {
        const char* v = bb_env("BB_MIRROR_POOL_MB");
        return v && *v ? s64(std::strtoull(v, nullptr, 10)) : s64(-1);
    }();
    return forced >= 0 ? u64(forced) << 20 : std::min<u64>(256_MB, budget / 8);
}
} // namespace

bool BufferCache::LayerRoomFor(u64 bytes) {
    // BB_MIRROR_ALLOC_PERCENT (0 = no check, the default): a mirror that would take our device
    // memory past this share of the driver's current budget waits. OFF: measured on the 2 GB laptop
    // the memory calls were ~5 % of the long frames, and with 92 the working set (live 1.6 GB of
    // 1.75) sat at the limit, thousands of promotions waited and frames got longer (20.8 % over
    // 40 ms against 7-8 %, 2026-10-26). Allocating past the budget made WDDM page
    // other memory out inside the allocation call: 300-900 MiB allocated in a frame took the frame
    // 150-300 ms on the GTX 1050 laptop (2026-10-25).
    static const u64 percent = [] {
        const char* env = bb_env("BB_MIRROR_ALLOC_PERCENT");
        return env && *env ? std::strtoull(env, nullptr, 10) : 0ull;
    }();
    if (percent == 0 || !instance.CanReportMemoryUsage() || instance.IsIntegrated()) {
        return true;
    }
    const u64 size_class = MirrorSizeClass(bytes);
    if (const auto it = mirror_pool.lower_bound(size_class);
        it != mirror_pool.end() && it->first <= size_class + size_class / 4) {
        return true; // made again from a pooled one: no new memory
    }
    const u64 budget = instance.GetDeviceMemoryBudgetNow();
    const u64 live = BbStats::device_alloc_bytes.load(std::memory_order_relaxed) -
                     BbStats::device_free_bytes.load(std::memory_order_relaxed);
    if (budget == 0 || live + size_class <= budget / 100 * percent) {
        return true;
    }
    static std::atomic<u64> deferred{0};
    static std::atomic<u32> printed{0};
    deferred.fetch_add(1, std::memory_order_relaxed);
    BbStats::mirror_copies_waiting.store(true, std::memory_order_relaxed);
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed.load(std::memory_order_relaxed) >= 10) {
        printed.store(now, std::memory_order_relaxed);
        std::printf("Layer memory: %llu mirrors of copies to VRAM waited for room (live %llu MiB + %llu "
                    "MiB over %u%% of %llu MiB)\n",
                    (unsigned long long)deferred.exchange(0), (unsigned long long)(live >> 20),
                    (unsigned long long)(size_class >> 20), unsigned(percent),
                    (unsigned long long)(budget >> 20));
    }
    return false;
}

std::unique_ptr<Buffer> BufferCache::LayerNewMirror(VAddr start, u64 bytes) {
    const u64 size_class = MirrorSizeClass(bytes);
    if (MirrorPoolCap(instance.GetTotalMemoryBudget()) != 0) {
        // The smallest pooled allocation that holds it, up to a quarter bigger.
        const auto it = mirror_pool.lower_bound(size_class);
        if (it != mirror_pool.end() && it->first <= size_class + size_class / 4) {
            auto mirror = std::move(it->second.first);
            mirror_pool_bytes -= it->first;
            mirror_pool.erase(it);
            mirror->cpu_addr = start;
            mirror->size_bytes = bytes;
            ++mirror_pool_reuses;
            return mirror;
        }
        ++mirror_pool_misses;
    }
    // BB_MIRROR_VRAM_ONLY=1: VRAM or none (VMA puts "mirrors" in system memory when VRAM is full).
    // Off by default: VMA's budget check refused them at 1.5-1.65 GB of a 1.77-1.83 GB budget on
    // the 2 GB laptop, the geometry was read over the bus and the game fell to 8-29 FPS (2026-10-26).
    static const bool system_ok = !bb_env_on("BB_MIRROR_VRAM_ONLY");
    std::optional<VramOnlyScope> vram_only;
    if (!system_ok) {
        vram_only.emplace();
    }
    auto mirror = std::make_unique<Buffer>(instance, start, size_class, MemoryType::DeviceLocal,
                                           fmt::format("bbport mirror {:#x}+{:#x}", start, bytes));
    if (!mirror->Valid()) {
        static std::atomic<u64> refused{0};
        static std::atomic<u32> printed{0};
        refused.fetch_add(1, std::memory_order_relaxed);
        const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
        if (now - printed.load(std::memory_order_relaxed) >= 10) {
            printed.store(now, std::memory_order_relaxed);
            std::printf("Layer memory: %llu mirrors not made: no room in VRAM (read in place)\n",
                        (unsigned long long)refused.exchange(0));
        }
        return nullptr;
    }
    mirror->alloc_bytes = size_class;
    BbStats::mirror_new_bytes.fetch_add(size_class, std::memory_order_relaxed);
    BbStats::mirror_new_by[BbStats::mirror_new_reason].fetch_add(size_class, std::memory_order_relaxed);
    mirror->size_bytes = bytes;
    return mirror;
}

void BufferCache::LayerReleaseMirror(std::unique_ptr<Buffer> mirror) {
    const u64 alloc = mirror->alloc_bytes ? mirror->alloc_bytes : mirror->SizeBytes();
    const u64 cap = MirrorPoolCap(instance.GetTotalMemoryBudget());
    if (cap == 0 || alloc > cap || VramPromotionsPaused()) {
        mirror_free_queue.push_back(std::move(mirror)); // freed a few a frame
        LayerDrainMirrorFrees();
        return;
    }
    LayerTrimMirrorPool(cap - alloc);
    mirror_pool_bytes += alloc;
    mirror_pool.emplace(alloc, std::pair{std::move(mirror),
                                         BbStats::coarse_second.load(std::memory_order_relaxed)});
}

void BufferCache::LayerDrainMirrorFrees() {
    // BB_MIRROR_FREE_MB_PER_FRAME (64): freeing hundreds of MiB of dedicated allocations in one
    // frame held it 80-150 ms on the GTX 1050 laptop (NVIDIA, Windows; 2026-10-25). Four times
    // that while VRAM is past the point where copies wait.
    static const u64 per_frame = [] {
        const char* env = bb_env("BB_MIRROR_FREE_MB_PER_FRAME");
        return u64(env && *env ? std::strtoull(env, nullptr, 10) : 64) << 20;
    }();
    static u64 frame = ~0ULL, freed = 0;
    const u64 now = BbStats::gpu_frames.load(std::memory_order_relaxed);
    if (now != frame) {
        frame = now;
        freed = 0;
    }
    // At most BB_MIRROR_FREE_BACKLOG_MB (128) wait: loading an area retired hundreds of MiB a frame,
    // the queue grew past a GB and VRAM to twice the budget (the driver paging, 60-90 ms frames).
    static const u64 backlog = [] {
        const char* env = bb_env("BB_MIRROR_FREE_BACKLOG_MB");
        return u64(env && *env ? std::strtoull(env, nullptr, 10) : 128) << 20;
    }();
    const auto bytes_of = [](const std::unique_ptr<Buffer>& b) {
        return u64(b->alloc_bytes ? b->alloc_bytes : b->SizeBytes());
    };
    u64 queued = 0;
    for (const auto& b : mirror_free_queue) {
        queued += bytes_of(b);
    }
    const u64 limit = per_frame == 0 ? ~0ULL : VramPromotionsPaused() ? per_frame * 4 : per_frame;
    while (!mirror_free_queue.empty() &&
           (freed == 0 || freed < limit || queued > backlog)) {
        const u64 bytes = bytes_of(mirror_free_queue.front());
        freed += bytes;
        queued -= bytes;
        mirror_free_queue.pop_front();
    }
}

void BufferCache::LayerTrimMirrorPool(u64 keep) {
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    while (!mirror_pool.empty()) {
        auto oldest = mirror_pool.begin();
        for (auto it = mirror_pool.begin(); it != mirror_pool.end(); ++it) {
            if (it->second.second < oldest->second.second) {
                oldest = it;
            }
        }
        if (mirror_pool_bytes <= keep && now - oldest->second.second < 60) {
            break;
        }
        mirror_pool_bytes -= oldest->first;
        mirror_free_queue.push_back(std::move(oldest->second.first)); // freed a few a frame
        mirror_pool.erase(oldest);
    }
}

void BufferCache::LayerMaintain() {
    LayerProcessTraps();
    // Pooled mirrors: all freed while VRAM is short, else those past the cap or a minute old.
    if (static u32 pool_second = 0;
        BbStats::coarse_second.load(std::memory_order_relaxed) != pool_second) {
        pool_second = BbStats::coarse_second.load(std::memory_order_relaxed);
        LayerTrimMirrorPool(VramPromotionsPaused() ? 0 : MirrorPoolCap(instance.GetTotalMemoryBudget()));
    }
    LayerDrainMirrorFrees();
    // Every BB_MIRROR_COMPACT_SECONDS (1): mirror memory without valid blocks (data the game freed
    // or moved back in place) over BB_MIRROR_COMPACT_MB (64) is given back, not only when VRAM
    // runs short. 0 MiB: off.
    static const u64 compact_at = [] {
        const char* env = bb_env("BB_MIRROR_COMPACT_MB");
        return u64(env && *env ? std::strtoull(env, nullptr, 10) : 64) << 20;
    }();
    static const u32 compact_every = [] {
        const char* env = bb_env("BB_MIRROR_COMPACT_SECONDS");
        return u32(std::max<long>(env && *env ? std::strtol(env, nullptr, 10) : 1, 1));
    }();
    static u32 compacted_second = 0;
    // Waste no compaction could free (valid blocks closer than its Bridge): not tried again until
    // more of it comes (the mirrors were made again every second, 0 MiB freed).
    static u64 stuck_waste = 0;
    if (const u32 second = BbStats::coarse_second.load(std::memory_order_relaxed);
        compact_at != 0 && second - compacted_second >= compact_every) {
        compacted_second = second;
        u64 waste = 0;
        for (const auto& [start, mirror] : layer_mirrors) {
            u64 valid = 0;
            Layer().ForEachValid(start >> block_shift, (start + mirror->SizeBytes()) >> block_shift,
                                 [&](u64 a, u64 b) { valid += b - a; });
            waste += mirror->SizeBytes() - (valid << block_shift);
        }
        if (waste < stuck_waste) {
            stuck_waste = waste;
        }
        if (waste >= compact_at && waste >= stuck_waste + compact_at / 2) {
            stuck_waste = LayerCompactMirrors(waste) == 0 ? waste : 0;
        }
    }
    // Retired mirrors the GPU is done with. BB_LAYER_CHECK=1: no page table entry may still point
    // into one (its memory is freed now): such an entry stops the game.
    std::erase_if(layer_retired, [&](auto& retired) {
        if (!scheduler.GetWorkSemaphore()->IsFree(retired.second)) {
            return false;
        }
        static const bool bda_log = bb_env("BB_BDA_LOG") != nullptr;
        if (bda_log) {
            std::printf("BDA: mirror %#llx+%#llx at GPU %#llx freed (retired at tick %llu)\n",
                        (unsigned long long)retired.first->CpuAddr(),
                        (unsigned long long)retired.first->SizeBytes(),
                        (unsigned long long)retired.first->BufferDeviceAddress(),
                        (unsigned long long)retired.second);
        }
        if (PagedVerifyOn()) {
            const Buffer* gone = retired.first.get();
            const u64 dev = gone->BufferDeviceAddress(), bytes = gone->SizeBytes();
            for (u64 block = gone->CpuAddr() >> block_shift;
                 block < (gone->CpuAddr() + bytes) >> block_shift; ++block) {
                const auto it = layer_shadow.find(block);
                if (it != layer_shadow.end() && it->second - dev < bytes) {
                    std::fprintf(stderr, "Layer check: block %#llx's page table entry %#llx points into "
                                 "the retired mirror %#llx+%#llx\n",
                                 (unsigned long long)(block << block_shift),
                                 (unsigned long long)it->second, (unsigned long long)gone->CpuAddr(),
                                 (unsigned long long)bytes);
                    if (bb_env_on("BB_LAYER_CHECK")) {
                        std::fprintf(stderr, "STOP: stale page table entry (BB_LAYER_CHECK=1)\n");
                        std::fflush(stderr);
                        std::abort();
                    }
                    break;
                }
            }
        }
        LayerReleaseMirror(std::move(retired.first));
        return true;
    });
    static u32 printed = 0;
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed >= 10) {
        printed = now;
        u64 mirror_bytes = 0, asset_total = 0;
        for (const auto& range : asset_bytes) {
            asset_total += range.end - range.start;
        }
        for (const auto& [start, mirror] : layer_mirrors) {
            mirror_bytes += mirror->SizeBytes();
        }
        std::printf("Layer memory: promoted runs in domains %llu, over a neighbour %llu, in none %llu;"
                    " in none by 64 MiB of address (MiB):",
                    (unsigned long long)layer_domain_runs[0].load(),
                    (unsigned long long)layer_domain_runs[1].load(),
                    (unsigned long long)layer_domain_runs[2].load());
        for (size_t b = 0; b < layer_none_buckets.size(); ++b) {
            if (const u64 n = layer_none_buckets[b].load(std::memory_order_relaxed); n >> (20 - block_shift)) {
                std::printf(" %#llx:%llu", (unsigned long long)((0x1000000000ull >> 26 & ~u64{4095}) + b) << 26,
                            (unsigned long long)((n << block_shift) >> 20));
            }
        }
        std::printf("\n");
        std::printf("Layer memory: %llu MiB valid in VRAM, %zu mirrors of %llu MiB (%zu retired); "
                    "loaded data %llu MiB; vkQueueBindSparse: %llu calls, %llu ranges; write traps hit %llu\n",
                    (unsigned long long)(Layer().ValidBytes() >> 20), layer_mirrors.size(),
                    (unsigned long long)(mirror_bytes >> 20), layer_retired.size(),
                    (unsigned long long)(asset_total >> 20),
                    (unsigned long long)arena_bind_calls.load(),
                    (unsigned long long)arena_bind_ranges.load(), (unsigned long long)LayerTrapFaults());
        std::printf("Layer memory: mirror pool %llu MiB in %zu, %llu new mirrors from it, %llu allocated\n",
                    (unsigned long long)(mirror_pool_bytes >> 20), mirror_pool.size(),
                    (unsigned long long)mirror_pool_reuses, (unsigned long long)mirror_pool_misses);
        {
            // Valid mirror data by when a binding last used its 2 MiB group: what a shorter
            // idle time (BB_VRAM_IDLE_SECONDS) would give back.
            constexpr std::array<u32, 5> Ages{2, 5, 15, 30, 60};
            std::array<u64, Ages.size() + 1> bytes{};
            const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
            Layer().ForEachValid(0, ~0ULL, [&](u64 a, u64 b) {
                for (u64 block = a; block < b;) {
                    const u64 group = (block << block_shift) >> USE_GROUP_BITS;
                    const u64 group_end =
                        std::min(b, ((group + 1) << USE_GROUP_BITS) >> block_shift);
                    const u32 age = group < group_use.size() ? now - group_use[group] : ~0u;
                    size_t bucket = 0;
                    while (bucket < Ages.size() && age >= Ages[bucket]) {
                        ++bucket;
                    }
                    bytes[bucket] += (group_end - block) << block_shift;
                    block = group_end;
                }
            });
            std::printf("Layer memory: valid VRAM data by last use (MiB): <2 s %llu, <5 s %llu, "
                        "<15 s %llu, <30 s %llu, <60 s %llu, older %llu\n",
                        (unsigned long long)(bytes[0] >> 20), (unsigned long long)(bytes[1] >> 20),
                        (unsigned long long)(bytes[2] >> 20), (unsigned long long)(bytes[3] >> 20),
                        (unsigned long long)(bytes[4] >> 20), (unsigned long long)(bytes[5] >> 20));
            {
                // The same by the blocks the bindings named (paged ones: their whole range or
                // their control list's; a promotion counts).
                std::array<u64, Ages.size() + 2> by_block{}; // the last: never stamped
                Layer().ForEachValid(0, ~0ULL, [&](u64 a, u64 b) {
                    for (u64 block = a; block < b; ++block) {
                        const u32 bound = BlockBoundSecond(block);
                        size_t bucket = Ages.size() + 1;
                        if (bound != 0) {
                            const u32 age = now - bound;
                            bucket = 0;
                            while (bucket < Ages.size() && age >= Ages[bucket]) {
                                ++bucket;
                            }
                        }
                        by_block[bucket] += u64{1} << block_shift;
                    }
                });
                std::printf("Layer memory: valid VRAM data by block bound (MiB): <2 s %llu, <5 s %llu, "
                            "<15 s %llu, <30 s %llu, <60 s %llu, older %llu, never %llu\n",
                            (unsigned long long)(by_block[0] >> 20),
                            (unsigned long long)(by_block[1] >> 20),
                            (unsigned long long)(by_block[2] >> 20),
                            (unsigned long long)(by_block[3] >> 20),
                            (unsigned long long)(by_block[4] >> 20),
                            (unsigned long long)(by_block[5] >> 20),
                            (unsigned long long)(by_block[6] >> 20));
            }
        }
        u64 volatile_blocks = 0;
        for (const auto& range : layer_volatile) {
            volatile_blocks += range.end - range.start;
        }
        u64 watched_blocks = 0;
        for (const auto& range : layer_watched) {
            watched_blocks += range.end - range.start;
        }
        std::printf("Layer memory: %llu volatile blocks, %llu MiB refreshed in 10 s (%llu MiB read in "
                    "place in render passes); %llu watched blocks, %llu uploaded again after CPU writes; GPU writes in VRAM barred in "
                    "%llu MiB the CPU wrote\n",
                    (unsigned long long)volatile_blocks,
                    (unsigned long long)(layer_volatile_refresh_bytes >> 20),
                    (unsigned long long)(layer_volatile_in_pass_bytes >> 20),
                    (unsigned long long)watched_blocks, (unsigned long long)layer_watch_refreshes,
                    (unsigned long long)([&] {
                        u64 blocks = 0;
                        for (const auto& range : layer_cpu_blocks) {
                            blocks += range.end - range.start;
                        }
                        return blocks;
                    }() << block_shift >> 20));
        layer_volatile_refresh_bytes = 0;
        layer_volatile_in_pass_bytes = 0;
        layer_watch_refreshes = 0;
        std::printf("Layer memory: write traps in 10 s: announced %llu; the game's code on watched "
                    "%llu, on other copies %llu; other code on watched %llu, on other copies %llu\n",
                    (unsigned long long)layer_trap_kinds[0], (unsigned long long)layer_trap_kinds[1],
                    (unsigned long long)layer_trap_kinds[2], (unsigned long long)layer_trap_kinds[3],
                    (unsigned long long)layer_trap_kinds[4]);
        layer_trap_kinds = {};
        if (LayerReadTrapsOn()) {
            u64 trapped = 0;
            {
                std::scoped_lock lk{layer_read_trap_mutex};
                layer_read_trapped.ForEach([&](VAddr a, VAddr b) { trapped += b - a; });
            }
            std::printf("Layer memory: read traps (GPU data in VRAM only): %llu KiB trapped; %llu CPU "
                        "accesses in 10 s, %llu KiB copied back for them, %llu by the recording "
                        "thread before the GPU's writes went out (read as they were)\n",
                        (unsigned long long)(trapped >> 10),
                        (unsigned long long)layer_read_trap_hits.exchange(0),
                        (unsigned long long)(layer_read_trap_bytes.exchange(0) >> 10),
                        (unsigned long long)layer_read_trap_stale.exchange(0));
        }
        std::printf("Layer memory: back in place (MiB): written %llu, mixed %llu, command %llu, "
                    "request %llu, copy back %llu, unmap %llu, idle %llu, unbound %llu; GPU data copied back %llu MiB\n",
                    (unsigned long long)(layer_demoted_bytes[0].load() >> 20),
                    (unsigned long long)(layer_demoted_bytes[1].load() >> 20),
                    (unsigned long long)(layer_demoted_bytes[2].load() >> 20),
                    (unsigned long long)(layer_demoted_bytes[3].load() >> 20),
                    (unsigned long long)(layer_demoted_bytes[4].load() >> 20),
                    (unsigned long long)(layer_demoted_bytes[5].load() >> 20),
                    (unsigned long long)(layer_demoted_bytes[6].load() >> 20),
                    (unsigned long long)(layer_demoted_bytes[7].load() >> 20),
                    (unsigned long long)(layer_copied_back_bytes >> 20));
        // What keeps blocks in place for good (until handed out again), and the loaded data the
        // module has no copy of: over a long session the copies' share fell (84 % -> 50 %).
        const auto mib = [&](const auto& list) {
            u64 blocks = 0;
            for (const auto& iv : list) {
                blocks += iv.end - iv.start;
            }
            return (unsigned long long)((blocks << block_shift) >> 20);
        };
        u64 loaded_out = 0;
        asset_bytes.ForEachInRange(0, ~0ULL, [&](const Interval& iv) {
            for (u64 b = iv.start >> block_shift; b <= (iv.end - 1) >> block_shift; ++b) {
                loaded_out += Layer().AnyValid(b, b + 1) ? 0 : 1;
            }
        });
        std::printf("Layer memory: held in place (MiB): dynamic %llu (CPU-hot %llu), watch failed %llu, "
                    "CPU-written %llu (GPU writes kept in place by them: %llu MiB in 10 s), write bounces "
                    "%llu; dynamic blocks reloaded whole %llu; blocks with loaded data and no copy %llu MiB\n",
                    mib(dynamic_blocks), mib(layer_cpu_hot), mib(layer_watch_failed),
                    mib(layer_cpu_blocks),
                    (unsigned long long)(BbStats::gpu_writes_kept_by_cpu_bytes.exchange(0) >> 20),
                    mib(layer_write_bounces),
                    (unsigned long long)(BbStats::dynamic_reloaded_bytes.load() >> 20),
                    (unsigned long long)((loaded_out << block_shift) >> 20));
        LayerPrintProbe();
    }
}

void BufferCache::LayerProcessIdle() {
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (VramIdleSeconds() == 0) {
        return;
    }
    LayerYieldToImages(now);
    std::vector<std::pair<u64, u64>> idle;
    Layer().ForEachValid(0, ~0ULL, [&](u64 a, u64 b) {
        for (u64 block = a; block < b;) {
            const u64 group = (block << block_shift) >> USE_GROUP_BITS;
            const u64 group_end = std::min(b, ((group + 1) << USE_GROUP_BITS) >> block_shift);
            if (now - group_use[group] >= VramIdleSeconds()) {
                idle.emplace_back(block, group_end);
            }
            block = group_end;
        }
    });
    for (const auto& [a, b] : idle) {
        LayerDemote(a, b, true, 6);
        BbStats::vram_idle_bytes.fetch_add((b - a) << block_shift, std::memory_order_relaxed);
    }
    // Blocks no binding named for BB_MIRROR_UNBOUND_SECONDS (0 = never, the default: on the PC
    // the dark Yahar'gul streets fell to ~110 FPS with it, their data read over PCIe until it was
    // promoted again; 2026-10-25): promoted with their
    // neighbours (domains, runs), yet only the bound ones are read in VRAM (2026-10-24, PC standing:
    // 425 of 883 MiB never bound). Their mirrors give the room back when compacted.
    static const u32 unbound_seconds = [] {
        const char* env = bb_env("BB_MIRROR_UNBOUND_SECONDS");
        return env ? u32(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    if (unbound_seconds != 0) {
        std::vector<std::pair<u64, u64>> unbound;
        std::vector<std::pair<u64, u64>> unstamped;
        Layer().ForEachValid(0, ~0ULL, [&](u64 a, u64 b) {
            for (u64 block = a; block < b; ++block) {
                const u32 bound = BlockBoundSecond(block);
                if (bound == 0) {
                    unstamped.emplace_back(block, block + 1); // valid before stamps: from now
                } else if (now - bound >= unbound_seconds) {
                    if (!unbound.empty() && unbound.back().second == block) {
                        unbound.back().second = block + 1;
                    } else {
                        unbound.emplace_back(block, block + 1);
                    }
                }
            }
        });
        for (const auto& [a, b] : unstamped) {
            NoteBlocksBound(a, b);
        }
        for (const auto& [a, b] : unbound) {
            LayerDemote(a, b, true, 7);
        }
    }
}

bool BufferCache::LayerRecentlyDemoted(u64 first, u64 end) const {
    // BB_MIRROR_REPROMOTE_SECONDS (0): blocks sent back are not copied again that soon. 10 s on the
    // 2 GB laptop left an area's geometry read over PCIe for 10 s after entering it: 8 FPS
    // (2026-10-25).
    static const u32 seconds = [] {
        const char* env = bb_env("BB_MIRROR_REPROMOTE_SECONDS");
        return env ? u32(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    if (seconds == 0 || group_demoted_second.empty()) {
        return false;
    }
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    const u64 g_last = std::min<u64>(((end << block_shift) - 1) >> USE_GROUP_BITS,
                                     group_demoted_second.size() - 1);
    for (u64 g = (first << block_shift) >> USE_GROUP_BITS; g <= g_last; ++g) {
        const u32 at = group_demoted_second[g];
        if (at != 0 && now - at < seconds) {
            return true;
        }
    }
    return false;
}
void BufferCache::NoteBlocksBound(u64 first, u64 end) {
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    const u64 per_group = u64{1} << (USE_GROUP_BITS - block_shift);
    for (u64 block = first; block < end;) {
        const u64 group = block / per_group;
        const u64 group_end = std::min(end, (group + 1) * per_group);
        if (group >= block_bound.size()) {
            block_bound.resize(std::min<u64>(u64{1} << (ADDRESS_SPACE_BITS - USE_GROUP_BITS),
                                             std::max<u64>(group + 1, block_bound.size() * 2)));
            if (group >= block_bound.size()) {
                return;
            }
        }
        auto& stamps = block_bound[group];
        if (!stamps) {
            stamps = std::make_unique<u32[]>(per_group);
        }
        for (u64 b = block; b < group_end; ++b) {
            stamps[b - group * per_group] = now;
        }
        block = group_end;
    }
}

u32 BufferCache::BlockBoundSecond(u64 block) const {
    const u64 per_group = u64{1} << (USE_GROUP_BITS - block_shift);
    const u64 group = block / per_group;
    return group < block_bound.size() && block_bound[group]
               ? block_bound[group][block - group * per_group]
               : 0;
}
void BufferCache::NoteDirectUse(VAddr address, u64 size) {
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    const u64 last = std::min<u64>((address + size - 1) >> USE_GROUP_BITS, layer_group_use.size() - 1);
    for (u64 group = address >> USE_GROUP_BITS; group <= last; ++group) {
        if (layer_group_use[group] != now) {
            layer_group_use[group] = now;
        }
    }
}

u64 BufferCache::LayerCompactMirrors(u64 wanted) {
    // Mirror memory without valid blocks (sent back in place, unmapped, handed out anew, the room
    // to grow) stays allocated as long as its mirror lives. Mirrors with at least MinWaste of it,
    // the most first, are made again over their valid blocks only: one new mirror per run of
    // valid blocks (runs closer than Bridge stay in one), their data copied over, the old mirror
    // retired as in a merge. No copy is lost.
    // Mirrors younger than BB_MIRROR_COMPACT_AGE seconds (30) are left alone unless VRAM is past the
    // texture collector's critical mark: on the 2 GB laptop compaction cut the room a mirror had
    // just grown, the next run grew it again (80, 52, 112, 64 MiB within seconds), 411 mirrors of
    // 22.9 GB made in 8 minutes for 0.4 GB of data (2026-10-25).
    static const u32 min_age = [] {
        const char* env = bb_env("BB_MIRROR_COMPACT_AGE");
        return env && *env ? u32(std::strtoul(env, nullptr, 10)) : 30u;
    }();
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    const u64 critical = BbStats::gc_critical_bytes.load(std::memory_order_relaxed);
    const bool emergency =
        critical != 0 && BbStats::gc_used_bytes.load(std::memory_order_relaxed) >= critical;
    constexpr u64 MinWaste = 8_MB, Bridge = 4_MB;
    std::vector<std::pair<u64, VAddr>> wasteful; // bytes without valid blocks, start
    for (const auto& [start, mirror] : layer_mirrors) {
        if (const auto born = layer_mirror_born.find(start);
            !emergency && born != layer_mirror_born.end() && now - born->second < min_age) {
            continue;
        }
        const u64 lo = start >> block_shift, hi = (start + mirror->SizeBytes()) >> block_shift;
        u64 valid = 0;
        Layer().ForEachValid(lo, hi, [&](u64 a, u64 b) { valid += b - a; });
        const u64 waste = mirror->SizeBytes() - (valid << block_shift);
        if (waste >= MinWaste) {
            wasteful.emplace_back(waste, start);
        }
    }
    std::ranges::sort(wasteful, std::greater{});
    u64 freed = 0;
    size_t compacted = 0;
    for (const auto& [waste, start] : wasteful) {
        if (freed >= wanted) {
            break;
        }
        const u64 old_size = layer_mirrors.at(start)->SizeBytes();
        const u64 lo = start >> block_shift, hi = (start + old_size) >> block_shift;
        std::vector<std::pair<u64, u64>> segments;
        u64 remade = 0;
        Layer().ForEachValid(lo, hi, [&](u64 a, u64 b) {
            if (!segments.empty() && ((a - segments.back().second) << block_shift) < Bridge) {
                segments.back().second = b;
            } else {
                segments.emplace_back(a, b);
            }
        });
        for (const auto& [a, b] : segments) {
            remade += (b - a) << block_shift;
        }
        if (old_size - remade < MinWaste) {
            continue; // its waste lies between valid blocks: copies for nothing
        }
        // The new mirrors first: without room in VRAM for all of them this one stays as it is.
        std::vector<std::unique_ptr<Buffer>> made;
        for (const auto& [a, b] : segments) {
            BbStats::mirror_new_reason = 2;
            auto mirror = LayerNewMirror(a << block_shift, (b - a) << block_shift);
            BbStats::mirror_new_reason = 0;
            if (!mirror) {
                break;
            }
            made.push_back(std::move(mirror));
        }
        if (made.size() != segments.size()) {
            continue;
        }
        auto node = layer_mirrors.extract(start);
        Buffer* old_buffer = node.mapped().get();
        Layer().RemoveMirror(start);
        BbStats::residency_alloc_bytes.fetch_sub(old_size, std::memory_order_relaxed); // replaced below
        u64 kept = 0;
        for (size_t s = 0; s < segments.size(); ++s) {
            const auto [a, b] = segments[s];
            const VAddr base = a << block_shift;
            const u64 size = (b - a) << block_shift;
            auto mirror = std::move(made[s]);
            boost::container::small_vector<vk::BufferCopy, 8> copies;
            Layer().ForEachValid(a, b, [&](u64 v_first, u64 v_end) {
                const VAddr from = v_first << block_shift;
                copies.push_back({old_buffer->Offset(from), mirror->Offset(from),
                                  (v_end - v_first) << block_shift});
            });
            if (!copies.empty()) {
                runtime.CopyBuffer(old_buffer, mirror.get(), copies);
            }
            Layer().AddMirror(base, size, mirror->Handle(), mirror->BufferDeviceAddress(), mirror.get());
            layer_mirrors.emplace(base, std::move(mirror));
            layer_mirror_born[base] = BbStats::coarse_second.load(std::memory_order_relaxed);
            BbStats::residency_alloc_bytes.fetch_add(size, std::memory_order_relaxed);
            kept += size;
        }
        if (std::ranges::none_of(segments, [&](const auto& s) { return (s.first << block_shift) == start; })) {
            layer_mirror_born.erase(start);
        }
        layer_retired.push_back({std::move(node.mapped()), scheduler.CurrentTick()});
        LayerRepublish(lo, hi); // paged bindings find the new buffers
        ++layer_generation;
        freed += old_size - kept;
        ++compacted;
    }
    if (freed != 0) {
        std::printf("Layer memory: %zu mirrors made again over their valid blocks, "
                    "%llu MiB freed\n",
                    compacted, (unsigned long long)(freed >> 20));
    }
    return freed;
}

void BufferCache::LayerYieldToImages(u32 now) {
    BbStats::Timer timer{BbStats::t_layer[4]};
    // VRAM short: the usage where promotions wait (PromoteCriticalMargin short of the texture
    // collector's critical mark). Past that mark the collector evicts images used a moment ago,
    // re-uploaded at once (stutters), and mirrors are what it cannot free. Meanwhile the copies of
    // where the player is now wait for their room, read over the bus (10 FPS less after a lamp
    // travel on a 6 GB card), so room is made for them.
    if (!instance.CanReportMemoryUsage() || layer_mirrors.empty()) {
        return;
    }
    const u64 critical = BbStats::gc_critical_bytes.load(std::memory_order_relaxed);
    const u64 usage = BbStats::gc_used_bytes.load(std::memory_order_relaxed);
    if (critical == 0 || usage + PromoteCriticalMargin() < critical) {
        return;
    }
    const u64 wanted = usage + PromoteCriticalMargin() + StaleHeadroom() - critical;
    static const u32 idle_seconds = [] {
        const char* env = bb_env("BB_VRAM_PRESSURE_IDLE_SECONDS");
        return env && *env ? u32(std::strtoul(env, nullptr, 10)) : 10u;
    }();
    // 1. Valid blocks in 2 MiB groups no binding took directly for BB_VRAM_PRESSURE_IDLE_SECONDS
    // (10; 3 sent geometry in use through paged bindings back on the 2 GB laptop, read over PCIe):
    // an area left behind, a lamp travelled from. They go back in place, the longest unused
    // first, whatever the mirrors' floor, as far as the mirrors' memory without valid blocks does
    // not already cover the need. Only direct bindings count (layer_group_use): paged bindings
    // over nearly all memory mark every group as used (group_use), and nothing was ever idle
    // while playing.
    u64 waste = 0;
    for (const auto& [start, mirror] : layer_mirrors) {
        u64 valid = 0;
        Layer().ForEachValid(start >> block_shift, (start + mirror->SizeBytes()) >> block_shift,
                             [&](u64 a, u64 b) { valid += b - a; });
        waste += mirror->SizeBytes() - (valid << block_shift);
    }
    u64 stale_bytes = 0;
    if (idle_seconds != 0 && waste < wanted) {
        std::vector<std::tuple<u32, u64, u64>> stale; // last direct use, first block, end block
        // Under their floor (LayerMirrorFloor) the mirrors give way only for what was unused
        // BB_VRAM_FLOOR_IDLE_SECONDS (30): the texture collector makes room for them there, and
        // both yielding to each other had the 2 GB laptop send 5.5 GB of copies back in 8 minutes
        // and copy them again, and evict 30 images a second (2026-10-25).
        static const u32 floor_idle_seconds = [] {
            const char* env = bb_env("BB_VRAM_FLOOR_IDLE_SECONDS");
            return env && *env ? u32(std::strtoul(env, nullptr, 10)) : 30u;
        }();
        const bool under_floor = BbStats::residency_alloc_bytes.load(std::memory_order_relaxed) <
                                 LayerMirrorFloor(instance.GetTotalMemoryBudget());
        const u32 stale_after = under_floor ? std::max(idle_seconds, floor_idle_seconds) : idle_seconds;
        Layer().ForEachValid(0, ~0ULL, [&](u64 a, u64 b) {
            for (u64 block = a; block < b;) {
                const u64 group = (block << block_shift) >> USE_GROUP_BITS;
                const u64 group_end = std::min(b, ((group + 1) << USE_GROUP_BITS) >> block_shift);
                const u32 last = layer_group_use[group];
                if (now - last >= stale_after) {
                    stale.emplace_back(last, block, group_end);
                }
                block = group_end;
            }
        });
        std::ranges::sort(stale);
        for (const auto& [last, a, b] : stale) {
            if (waste + stale_bytes >= wanted) {
                break;
            }
            LayerDemote(a, b, true, 6);
            stale_bytes += (b - a) << block_shift;
        }
        if (stale_bytes != 0) {
            BbStats::vram_idle_bytes.fetch_add(stale_bytes, std::memory_order_relaxed);
            static u32 printed = 0;
            if (now - printed >= 5) {
                printed = now;
                std::printf("Layer memory: VRAM %llu MiB, texture collector critical at %llu MiB: "
                            "%llu MiB of VRAM copies not bound for %u s back in place\n",
                            (unsigned long long)(usage >> 20), (unsigned long long)(critical >> 20),
                            (unsigned long long)(stale_bytes >> 20), idle_seconds);
            }
        }
    }
    // 2. The memory without valid blocks is given back (mirrors made again, smaller).
    LayerCompactMirrors(wanted);
}

} // namespace VideoCore
