// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne: the game's GraphicsPrivate heap growing (and shrinking) on demand, and the heap
// traces (BB_HEAP_TRACE). Split out of gpu/shim/bbport_guest_hooks.cpp (2026-10-26).

#include "common/types.h"
#include "shim/bb_env.h"
#include "bbport_guest_hooks.h"
#include "bbport_settings.h"
#include "bbport_guest_code.h"
#include "bbport_heap_sites.h"
#include "bbport_toggles.h"
#include "bbport_fx_init_clear.h"
#include "bbport_effect_flags.h"
#include "platform/gnm/gnm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include "bbport_engine_fixes.h"
#include "bbport_guest_code_hooks.h"
#include <cmath>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>
#include <utility>
#include "bbport_guest_memory.h"
#include "games/bloodborne/hooks_internal.h"

namespace BbGuestHooks {
namespace {

// BB_HEAP_TRACE=1 (diagnostics, the game's heaps growing on demand): the game's in-band heap
// allocator (init 0x1fbf110(allocator, base, size), allocate 0x1fbf300) serves its direct memory
// heaps (sizes from the table at 0x4736d40). Per heap: the lowest and highest bytes its
// allocations ever reached, every 30 s.
struct HeapTrace {
    std::atomic<u64> allocator{0}, base{0}, size{0}, low{~u64{0}}, high{0}, count{0};
};
std::array<HeapTrace, 64> heap_traces;

HeapTrace* HeapTraceFor(u64 allocator) {
    for (auto& t : heap_traces) {
        const u64 have = t.allocator.load(std::memory_order_acquire);
        if (have == allocator) {
            return &t;
        }
        if (have == 0) {
            u64 expected = 0;
            if (t.allocator.compare_exchange_strong(expected, allocator) || expected == allocator) {
                t.base.store(JobProbeWord(allocator), std::memory_order_relaxed);
                t.size.store(JobProbeWord(allocator + 8), std::memory_order_relaxed);
                return &t;
            }
        }
    }
    return nullptr;
}

void PS4_SYSV_ABI HeapInitTrace(u64, const Core::GuestRegisterSnapshot* r) {
    std::printf("Heap trace: heap allocator %#llx over %#llx+%llu MiB\n",
                (unsigned long long)r->r15, (unsigned long long)r->r14,
                (unsigned long long)(r->r13 >> 20));
}

void PS4_SYSV_ABI HeapAllocTrace(u64, const Core::GuestRegisterSnapshot* r) {
    const u64 block = r->rax;
    const u64 allocator = JobProbeWord(r->rbp - 0xe0);
    const u64 bytes = JobProbeWord(r->rbp - 0xe8);
    static std::atomic<std::uint32_t> told{0};
    if (told.fetch_add(1, std::memory_order_relaxed) < 12) {
        std::printf("Heap trace: allocate -> %#llx, allocator %#llx (base %#llx size %#llx), %llu "
                    "bytes\n", (unsigned long long)block, (unsigned long long)allocator,
                    (unsigned long long)JobProbeWord(allocator),
                    (unsigned long long)JobProbeWord(allocator + 8), (unsigned long long)bytes);
    }
    if (!block || !allocator) {
        return;
    }
    HeapTrace* t = HeapTraceFor(allocator);
    if (!t) {
        return;
    }
    t->count.fetch_add(1, std::memory_order_relaxed);
    u64 low = t->low.load(std::memory_order_relaxed);
    while (block < low && !t->low.compare_exchange_weak(low, block)) {
    }
    u64 high = t->high.load(std::memory_order_relaxed);
    while (block + bytes > high && !t->high.compare_exchange_weak(high, block + bytes)) {
    }
    static std::atomic<long long> printed{0};
    const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
    long long last = printed.load(std::memory_order_relaxed);
    if (now - last < 30 || !printed.compare_exchange_strong(last, now)) {
        return;
    }
    for (const auto& h : heap_traces) {
        const u64 base = h.base.load(), size = h.size.load(), lo = h.low.load(), hi = h.high.load();
        if (!h.allocator.load() || size < (64ull << 20) || hi == 0) {
            continue;
        }
        std::printf("Heap trace: heap %#llx+%llu MiB: allocations reached [+%llu MiB, +%llu MiB) "
                    "(%llu allocations)\n",
                    (unsigned long long)base, (unsigned long long)(size >> 20),
                    (unsigned long long)((lo - base) >> 20), (unsigned long long)((hi - base) >> 20),
                    (unsigned long long)h.count.load());
    }
}

/// The free nodes of a game range allocator core (0x2088030; see BB_HEAP_WALK) as {start, end}.
std::vector<std::pair<u64, u64>> CoreFreeRanges(u64 core) {
    std::vector<std::pair<u64, u64>> out;
    for (u64 n = JobProbeWord(core + 0x88); n && n != core + 0x68 && out.size() < 4096;
         n = JobProbeWord(n + 0x20)) {
        out.emplace_back(JobProbeWord(n), JobProbeWord(JobProbeWord(n + 0x10)));
    }
    return out;
}

/// BB_HEAP_TRACE: GraphicsPrivate's allocations (FastBiHeap's large-block allocator A, its core
/// 0x2088030 called at 0x1fc5472 under the heap's lock): sizes, failures, the heap at the first.
void PS4_SYSV_ABI HeapGrowObserve(u64, const Core::GuestRegisterSnapshot* r) {
    const u64 core = r->rbx, size = r->r14, result = r->rax;
    static std::atomic<u64> calls{0}, failed{0}, largest{0}, bytes{0};
    if (calls.fetch_add(1, std::memory_order_relaxed) == 0) {
        const auto free_ranges = CoreFreeRanges(core);
        std::printf("Heap grow: first allocation in core %#llx (heap [%#llx, %#llx)), %zu free ranges:",
                    (unsigned long long)core, (unsigned long long)JobProbeWord(core + 0xa0),
                    (unsigned long long)JobProbeWord(core + 0xa8), free_ranges.size());
        for (const auto& [a, b] : free_ranges) {
            std::printf(" [%#llx, %#llx)", (unsigned long long)a, (unsigned long long)b);
        }
        std::printf("\n");
    }
    if (!result) failed.fetch_add(1, std::memory_order_relaxed);
    bytes.fetch_add(size, std::memory_order_relaxed);
    // The lowest the free bytes counter ([core + 0xb8]) ever went: the heap's peak use.
    static std::atomic<u64> least_free{~u64{0}};
    const u64 free_now = JobProbeWord(core + 0xb8);
    u64 least = least_free.load(std::memory_order_relaxed);
    if (free_now < least) {
        least_free.store(free_now, std::memory_order_relaxed);
        least = free_now;
    }
    u64 big = largest.load(std::memory_order_relaxed);
    while (size > big && !largest.compare_exchange_weak(big, size)) {
    }
    static std::atomic<long long> printed{0};
    const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
    long long last = printed.load(std::memory_order_relaxed);
    if (now - last >= 20 && printed.compare_exchange_strong(last, now)) {
        const auto free_ranges = CoreFreeRanges(core);
        u64 free_bytes = 0, top_free = 0;
        for (const auto& [a, b] : free_ranges) {
            free_bytes += b - a;
            top_free = std::max(top_free, b - a);
        }
        std::printf("Heap grow: core %#llx: least free ever %llu MiB; %llu allocations (%llu MiB asked, largest %llu KiB, %llu "
                    "failed); now %zu free ranges, %llu MiB free, largest %llu MiB, last free range "
                    "[%#llx, %#llx)\n",
                    (unsigned long long)core, (unsigned long long)(least >> 20), (unsigned long long)calls.load(),
                    (unsigned long long)(bytes.load() >> 20), (unsigned long long)(largest.load() >> 10),
                    (unsigned long long)failed.load(), free_ranges.size(),
                    (unsigned long long)(free_bytes >> 20), (unsigned long long)(top_free >> 20),
                    (unsigned long long)(free_ranges.empty() ? 0 : free_ranges.back().first),
                    (unsigned long long)(free_ranges.empty() ? 0 : free_ranges.back().second));
    }
}

// BB_HEAP_GROW=1: GraphicsPrivate (2287 MiB, the game's largest heap; FastBiHeap's allocator
// for blocks of 1 MiB and more, a range allocator with its nodes outside the heap: allocate
// 0x2088030(core, size, alignment), free 0x2088900(core, address); first fit in address order,
// a block cut from the start of a free range, merged on free with free neighbours only) has GPU
// memory for its first BB_HEAP_GROW_INITIAL_MB (512; 1024 before 2026-10-23: a standing run held
// 832 MiB with 512, the other 190 MiB empty) only (runtime_memory_set_grow). The rest is
// held by the allocator itself: at its first allocation (the heap still empty past its first few
// MiB) a filler up to there, then per piece of BB_HEAP_GROW_PIECE_MB (320; the game asks for up
// to 256 MiB at once) a reserve block and a 64 KiB fence at its end, then the filler freed. When an
// allocation finds no room, the next piece gets its GPU memory (runtime_memory_commit) and its
// reserve is freed: fences keep every block within one piece, so every buffer the GPU binds lies
// whole in one Vulkan chunk. The heap's counters ([core + 0xb0] blocks, [core + 0xb8] free bytes)
// are kept as if the reserves were free: the game budgets its streaming by them.
extern "C" void runtime_memory_set_grow(const std::uint64_t* size_at, std::uint64_t initial);
extern "C" int runtime_memory_deferred(std::uint64_t* phys, std::uint64_t* end);
extern "C" int runtime_memory_commit(std::uint64_t phys, std::uint64_t size);
extern "C" int runtime_memory_decommit(std::uint64_t phys, std::uint64_t size);
extern "C" int runtime_memory_direct_phys(uintptr_t address, std::uint64_t* phys, uintptr_t* end);
extern "C" void runtime_memory_commit_ahead(std::uint64_t phys, std::uint64_t size);
using CoreAlloc = u64(PS4_SYSV_ABI*)(u64 core, u64 size, u64 alignment);
using CoreFree = void(PS4_SYSV_ABI*)(u64 core, u64 address);
constexpr u64 CoreAllocate = 0x2088030, CoreRelease = 0x2088900;
constexpr u64 GrowFence = 64ull << 10;
struct GrowPiece {
    u64 start = 0, size = 0, reserve = 0;
    bool committed = false;
    long long empty_since = 0; ///< seconds (steady clock) since it is all free again, 0: in use
};
struct HeapGrowState {
    bool tried = false, active = false, shrink_off = false;
    u64 core = 0, low = 0, low_phys = 0;
    std::vector<GrowPiece> pieces;
};
HeapGrowState heap_grow;

/// Growth off after the runtime deferred the heap's tail: all of it gets its GPU memory now (the
/// game may hand out blocks there; the GPU must see them).
void HeapGrowCommitAll() {
    u64 dphys = 0, dend = 0;
    if (runtime_memory_deferred(&dphys, &dend) && runtime_memory_commit(dphys, dend - dphys) == 0) {
        std::printf("Heap grow: the deferred %llu MiB got their GPU memory (as without growth)\n",
                    (unsigned long long)((dend - dphys) >> 20));
    }
}

/// The first allocation: the reserves and fences, in place (nothing changed when anything differs).
void HeapGrowAhead();

void HeapGrowSetup(u64 core) {
    heap_grow.tried = true;
    struct CommitIfOff {
        ~CommitIfOff() {
            if (!heap_grow.active) {
                HeapGrowCommitAll();
            }
        }
    } commit_if_off;
    const auto alloc = reinterpret_cast<CoreAlloc>(ImageBase + CoreAllocate);
    const auto release = reinterpret_cast<CoreFree>(ImageBase + CoreRelease);
    const u64 low = JobProbeWord(core + 0xa0), high = JobProbeWord(core + 0xa8);
    u64 dphys = 0, dend = 0, low_phys = 0;
    uintptr_t map_end = 0;
    if (!runtime_memory_deferred(&dphys, &dend) ||
        !runtime_memory_direct_phys(low, &low_phys, &map_end) || dend - low_phys != high - low) {
        std::printf("Heap grow: off (core %#llx [%#llx, %#llx) is not the deferred allocation)\n",
                    (unsigned long long)core, (unsigned long long)low, (unsigned long long)high);
        return;
    }
    const u64 boundary = low + (dphys - low_phys);
    const auto free_ranges = CoreFreeRanges(core);
    if (free_ranges.size() != 1 || free_ranges[0].second != high ||
        free_ranges[0].first + GrowFence >= boundary) {
        std::printf("Heap grow: off (the heap is not as expected: %zu free ranges, first at %#llx)\n",
                    free_ranges.size(), (unsigned long long)(free_ranges.empty() ? 0 : free_ranges[0].first));
        return;
    }
    const u64 counted_blocks = JobProbeWord(core + 0xb0), counted_free = JobProbeWord(core + 0xb8);
    std::vector<u64> made; // in order, for an undo
    const auto undo = [&](const char* why) {
        for (auto it = made.rbegin(); it != made.rend(); ++it) {
            release(core, *it);
        }
        *reinterpret_cast<u64*>(core + 0xb0) = counted_blocks;
        *reinterpret_cast<u64*>(core + 0xb8) = counted_free;
        std::printf("Heap grow: off (%s)\n", why);
    };
    const auto take = [&](u64 size, u64 expected) {
        const u64 block = alloc(core, size, 16);
        if (block) made.push_back(block);
        return block == expected;
    };
    const u64 filler_at = free_ranges[0].first;
    if (!take(boundary - GrowFence - filler_at, filler_at) || !take(GrowFence, boundary - GrowFence)) {
        return undo("the filler or the first fence landed elsewhere");
    }
    const u64 piece = std::max<u64>(bb_env_int("BB_HEAP_GROW_PIECE_MB", 320), 258) << 20;
    for (u64 at = boundary; at < high;) {
        const u64 size = std::min(piece, high - at);
        const bool last = at + size >= high;
        if (!take(last ? size : size - GrowFence, at) ||
            (!last && !take(GrowFence, at + size - GrowFence))) {
            heap_grow.pieces.clear();
            return undo("a reserve landed elsewhere");
        }
        heap_grow.pieces.push_back({at, size, at, false});
        at += size;
    }
    release(core, made[0]); // the filler: the heap's first piece is the game's again
    *reinterpret_cast<u64*>(core + 0xb0) = counted_blocks;
    *reinterpret_cast<u64*>(core + 0xb8) = counted_free;
    heap_grow.core = core;
    heap_grow.low = low;
    heap_grow.low_phys = low_phys;
    heap_grow.active = true;
    std::printf("Heap grow: on: heap [%#llx, %#llx), GPU memory for its first %llu MiB, %zu pieces "
                "of up to %llu MiB on demand (counters kept: %llu blocks, %llu MiB free)\n",
                (unsigned long long)low, (unsigned long long)high,
                (unsigned long long)((boundary - low) >> 20), heap_grow.pieces.size(),
                (unsigned long long)(piece >> 20), (unsigned long long)counted_blocks,
                (unsigned long long)(counted_free >> 20));
    HeapGrowAhead();
}

/// The next piece's GPU memory made ahead, on a thread of its own (runtime_memory_commit_ahead;
/// host memory imports: Windows, NVIDIA): the import that gives a piece its GPU memory ran in the
/// game's allocating thread with its heap locked, 120-170 ms for 320 MiB while areas streamed in
/// (RX 7800 XT, Windows, 2026-10-27), every one a frame of that length. The piece's pages are the
/// game's reserve, unused until it is given; host memory imports never give a piece back (grow
/// only), so one made ahead is only made earlier. BB_HEAP_GROW_AHEAD=N pieces ahead (2; 0: off,
/// at most 4): one was not enough, the game asked for pieces 0.1-0.2 s apart while loading.
void HeapGrowAhead() {
    static const int depth = [] {
        const char* env = bb_env("BB_HEAP_GROW_AHEAD");
        const int n = env && *env ? std::atoi(env) : 2;
        return BbGuestMemory::HostImported() && !BbGuestMemory::Lazy() ? std::clamp(n, 0, 4) : 0;
    }();
    int left = depth;
    for (const auto& p : heap_grow.pieces) {
        if (left == 0) {
            return;
        }
        if (!p.committed) {
            runtime_memory_commit_ahead(heap_grow.low_phys + (p.start - heap_grow.low), p.size);
            --left;
        }
    }
}

/// An allocation found no room: the next piece gets its GPU memory and its reserve is freed.
bool HeapGrowOne(u64 core, u64 wanted) {
    const auto release = reinterpret_cast<CoreFree>(ImageBase + CoreRelease);
    for (auto& p : heap_grow.pieces) {
        if (p.committed) {
            continue;
        }
        if (wanted + GrowFence > p.size) {
            std::printf("Heap grow: %llu MiB asked, more than a piece holds\n",
                        (unsigned long long)(wanted >> 20));
            return false;
        }
        const u64 phys = heap_grow.low_phys + (p.start - heap_grow.low);
        if (runtime_memory_commit(phys, p.size) != 0) {
            std::printf("Heap grow: no GPU memory for the piece at %#llx\n", (unsigned long long)p.start);
            return false;
        }
        // The game's free adds the reserve to the counters, which counted it as free already.
        const u64 reserve = (p.start + p.size >= JobProbeWord(core + 0xa8)) ? p.size : p.size - GrowFence;
        *reinterpret_cast<u64*>(core + 0xb0) += 1;
        *reinterpret_cast<u64*>(core + 0xb8) -= reserve;
        release(core, p.reserve);
        p.committed = true;
        std::printf("Heap grow: piece [%#llx, +%llu MiB) given to the game (%llu MiB asked)\n",
                    (unsigned long long)p.start, (unsigned long long)(p.size >> 20),
                    (unsigned long long)(wanted >> 20));
        HeapGrowAhead();
        return true;
    }
    return false;
}

long long SteadySecondsNow() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/// A committed piece all free again for BB_HEAP_SHRINK_DELAY s (60; the game asked for nothing in
/// it meanwhile): reserved again, its GPU memory given back (freed 3 s later, when the GPU is done
/// with anything there). At most once a second.
/// The free list's head is pointed at the piece's free node for the allocation (first fit would
/// take a lower free range that fits), then put back.
void HeapGrowShrink(u64 core) {
    if (heap_grow.shrink_off) {
        return;
    }
    static long long checked = 0;
    const long long now = SteadySecondsNow();
    if (now == checked) {
        return;
    }
    checked = now;
    std::vector<std::pair<u64, u64>> free_nodes; // node, start
    for (u64 n = JobProbeWord(core + 0x88); n && n != core + 0x68 && free_nodes.size() < 4096;
         n = JobProbeWord(n + 0x20)) {
        free_nodes.emplace_back(n, JobProbeWord(n));
    }
    const u64 high = JobProbeWord(core + 0xa8);
    const auto alloc = reinterpret_cast<CoreAlloc>(ImageBase + CoreAllocate);
    const auto release = reinterpret_cast<CoreFree>(ImageBase + CoreRelease);
    // Every 20 s: how full each piece is (the first one: the heap below the deferred part).
    static long long reported = 0;
    if (now - reported >= 20) {
        reported = now;
        const u64 low = heap_grow.low;
        std::vector<std::pair<u64, u64>> spans{{low, heap_grow.pieces.empty() ? high : heap_grow.pieces[0].start}};
        for (const auto& q : heap_grow.pieces) spans.emplace_back(q.start, q.start + q.size);
        std::printf("Heap grow: pieces used (MiB):");
        for (std::size_t i = 0; i < spans.size(); ++i) {
            u64 free_bytes = 0;
            for (const auto& [n, start] : free_nodes) {
                const u64 end = JobProbeWord(JobProbeWord(n + 0x10));
                const u64 a = std::max(start, spans[i].first), b = std::min(end, spans[i].second);
                if (a < b) free_bytes += b - a;
            }
            const u64 span = spans[i].second - spans[i].first;
            std::printf(" %s%llu/%llu", i == 0 ? "" : (heap_grow.pieces[i - 1].committed ? "+" : "-"),
                        (unsigned long long)((span - free_bytes) >> 20), (unsigned long long)(span >> 20));
        }
        std::printf("\n");
    }
    for (auto& p : heap_grow.pieces) {
        if (!p.committed) {
            continue;
        }
        const u64 payload = p.start + p.size >= high ? p.size : p.size - GrowFence;
        u64 node = 0;
        for (const auto& [n, start] : free_nodes) {
            if (start == p.start && JobProbeWord(JobProbeWord(n + 0x10)) == p.start + payload) {
                node = n;
            }
        }
        if (!node) {
            p.empty_since = 0;
            continue;
        }
        if (!p.empty_since) {
            p.empty_since = now;
            continue;
        }
        // A minute free (BB_HEAP_SHRINK_DELAY, seconds): after 1 s the GTX 1050 laptop gave a piece
        // back and took it again moments later on entering an area, 75 ms under the memory map
        // lock each way (2026-10-25).
        static const long long delay = (long long)bb_env_int("BB_HEAP_SHRINK_DELAY", 60);
        if (now - p.empty_since < delay) {
            continue;
        }
        auto* head = reinterpret_cast<u64*>(core + 0x88);
        const u64 saved_head = *head;
        *head = node;
        const u64 block = alloc(core, payload, 16);
        if (saved_head != node) {
            *head = saved_head;
        }
        if (block != p.start) {
            if (block) {
                release(core, block);
            }
            p.empty_since = 0;
            std::printf("Heap grow: the piece at %#llx could not be reserved again\n",
                        (unsigned long long)p.start);
            continue;
        }
        // Invisible to the game's counters, as the reserves were.
        *reinterpret_cast<u64*>(core + 0xb0) -= 1;
        *reinterpret_cast<u64*>(core + 0xb8) += payload;
        const u64 phys = heap_grow.low_phys + (p.start - heap_grow.low);
        if (runtime_memory_decommit(phys, p.size) != 0) {
            *reinterpret_cast<u64*>(core + 0xb0) += 1;
            *reinterpret_cast<u64*>(core + 0xb8) -= payload;
            release(core, block);
            p.empty_since = 0;
            // Host memory imports (NVIDIA) cannot give a piece back: grow only from now on.
            heap_grow.shrink_off = true;
            std::printf("Heap grow: pieces are not given back here (grow only)\n");
            continue;
        }
        p.reserve = block;
        p.committed = false;
        p.empty_since = 0;
        std::printf("Heap grow: piece [%#llx, +%llu MiB) taken back (all free)\n",
                    (unsigned long long)p.start, (unsigned long long)(p.size >> 20));
    }
}

/// A's blocks handed out (address -> size), for their range when freed (under the heap's lock).
std::unordered_map<u64, u64> heap_blocks;

/// After a block of A was freed (0x1fc560a, the lock held): r15 the core, r14 the block. Its VRAM
/// copies go (the GPU side hears of it as of an unmapped range), then whether a piece is all free.
void PS4_SYSV_ABI HeapShrinkHook(u64, const Core::GuestRegisterSnapshot* r) {
    if (!heap_grow.active || r->r15 != heap_grow.core) {
        return;
    }
    if (const auto it = heap_blocks.find(r->r14); it != heap_blocks.end()) {
        if (on_range_dead) {
            on_range_dead(it->first, it->second);
        }
        heap_blocks.erase(it);
    }
    HeapGrowShrink(r->r15);
}

void PS4_SYSV_ABI HeapGrowHook(u64, const Core::GuestRegisterSnapshot* r) {
    const u64 core = r->rbx;
    if (!heap_grow.tried) {
        HeapGrowSetup(core);
    }
    if (!heap_grow.active || core != heap_grow.core) {
        return;
    }
    if (r->rax) {
        heap_blocks[r->rax] = r->r14;
        if (on_range_live) {
            on_range_live(r->rax, r->r14);
        }
        HeapGrowShrink(core);
        return;
    }
    const auto alloc = reinterpret_cast<CoreAlloc>(ImageBase + CoreAllocate);
    const u64 alignment = r->r15 > 16 ? r->r15 : 16;
    while (HeapGrowOne(core, r->r14)) {
        if (const u64 block = alloc(core, r->r14, alignment)) {
            heap_blocks[block] = r->r14;
            if (on_range_live) {
                on_range_live(block, r->r14);
            }
            const_cast<Core::GuestRegisterSnapshot*>(r)->rax = block;
            return;
        }
    }
}

} // namespace

/// BB_HEAP_FIND=<value> (reverse engineering): once, 60 s into the run, where the game's data,
/// bss and MAIN heap hold that qword, with the object around it and its vtable's first entries.
void HeapFind() {
    static const u64 value = [] {
        const char* env = bb_env("BB_HEAP_FIND");
        return env ? std::strtoull(env, nullptr, 16) : 0ull;
    }();
    static std::atomic<bool> done{false};
    if (!value || done.exchange(true)) {
        return;
    }
    const std::array<std::pair<u64, u64>, 2> ranges{{{ImageBase + 0x50dc000, 0x5f70f4},
                                                     {0x1007400000ull, 549453824ull}}};
    unsigned found = 0;
    for (const auto& [start, size] : ranges) {
        for (u64 a = start; a + 8 <= start + size && found < 40; a += 8) {
            if (!runtime_memory_is_mapped(a & ~u64{0xfff}, 0x1000) && a >= 0x1000000000ull) {
                a = (a | 0xfff) - 7;
                continue;
            }
            if (JobProbeWord(a) != value) {
                continue;
            }
            ++found;
            std::printf("Heap find: %#llx at %#llx:", (unsigned long long)value, (unsigned long long)a);
            for (int i = -6; i <= 6; ++i) {
                std::printf(" %+d:%#llx", i * 8, (unsigned long long)JobProbeWord(a + i * 8));
            }
            std::printf("\n");
            for (int i = -6; i <= 0; ++i) {
                const u64 vt = JobProbeWord(a + i * 8);
                if (vt >= ImageBase && vt < ImageBase + 0x6000000) {
                    std::printf("Heap find:   vtable %+d = %#llx:", i * 8, (unsigned long long)(vt - ImageBase));
                    for (int k = 0; k < 16; ++k) {
                        const u64 fn = JobProbeWord(vt + k * 8);
                        std::printf(" %#llx", (unsigned long long)(fn >= ImageBase ? fn - ImageBase : fn));
                    }
                    std::printf("\n");
                }
            }
        }
    }
    std::printf("Heap find: %#llx: %u places\n", (unsigned long long)value, found);
    // BB_HEAP_WALK=core[,core...]: the game's range allocator cores (0x2088030: nodes {start,
    // prev, next in address order, free-list prev/next}; free list head at core + 0x88, its
    // sentinel core + 0x68): every node in address order, free ones marked, as a 16 MiB map.
    if (const char* env = bb_env("BB_HEAP_WALK")) {
        for (const char* at = env; *at;) {
            char* end = nullptr;
            const u64 core = std::strtoull(at, &end, 16);
            at = *end ? end + 1 : end;
            std::vector<u64> free_nodes;
            for (u64 n = JobProbeWord(core + 0x88); n && n != core + 0x68 && free_nodes.size() < 100000;
                 n = JobProbeWord(n + 0x20)) {
                free_nodes.push_back(n);
            }
            if (free_nodes.empty()) {
                std::printf("Heap walk: core %#llx: no free nodes\n", (unsigned long long)core);
                continue;
            }
            // The address list is a ring: its first node is the one with the lowest start.
            u64 first = free_nodes[0];
            for (u64 n = JobProbeWord(first + 0x10), guard = 0; n && n != free_nodes[0] && guard < 200000;
                 n = JobProbeWord(n + 0x10), ++guard) {
                if (JobProbeWord(n) < JobProbeWord(first)) first = n;
            }
            std::sort(free_nodes.begin(), free_nodes.end());
            const auto is_free = [&](u64 n) {
                return std::binary_search(free_nodes.begin(), free_nodes.end(), n);
            };
            u64 nodes = 0, free_bytes = 0, used_bytes = 0, largest = 0, lo = ~u64{0}, hi = 0;
            std::array<u64, 256> used_map{}, free_map{};
            u64 map_base = 0;
            for (u64 n = first; n && nodes < 200000; ++nodes) {
                const u64 next = JobProbeWord(n + 0x10);
                const u64 start = JobProbeWord(n);
                const u64 stop = next ? JobProbeWord(next) : 0;
                if (!next || next == first || stop < start) {
                    hi = start; // the end marker
                    ++nodes;
                    break;
                }
                if (!map_base) map_base = start;
                lo = std::min(lo, start);
                const u64 bytes = stop > start ? stop - start : 0;
                const bool f = is_free(n);
                (f ? free_bytes : used_bytes) += bytes;
                if (f) largest = std::max(largest, bytes);
                for (u64 a = start; a < stop; a += 1 << 20) {
                    const u64 b = (a - map_base) >> 24;
                    if (b < 256) (f ? free_map : used_map)[b] += 1;
                }
                n = next;
            }
            std::string map;
            for (int b = 0; b < 256 && map_base + (u64(b) << 24) < hi; ++b) {
                map += used_map[b] == 0 ? '.' : free_map[b] == 0 ? '#' : '+';
            }
            std::printf("Heap walk: core %#llx: [%#llx, %#llx) %llu nodes, %zu free; used %llu MiB, "
                        "free %llu MiB, largest free %llu MiB |%s|\n",
                        (unsigned long long)core, (unsigned long long)lo, (unsigned long long)hi,
                        (unsigned long long)nodes, free_nodes.size(),
                        (unsigned long long)(used_bytes >> 20), (unsigned long long)(free_bytes >> 20),
                        (unsigned long long)(largest >> 20), map.c_str());
        }
    }
    // BB_HEAP_DUMP=addr[,addr...]: those objects, and the objects their first 8 fields point to
    // (one level): 24 fields each, vtable entries as image offsets.
    if (const char* env = bb_env("BB_HEAP_DUMP")) {
        const auto dump = [](const char* what, u64 object) {
            std::printf("Heap dump: %s %#llx:", what, (unsigned long long)object);
            for (int i = 0; i < 24; ++i) {
                std::printf(" +%x:%#llx", i * 8, (unsigned long long)JobProbeWord(object + i * 8));
            }
            const u64 vt = JobProbeWord(object);
            if (vt >= ImageBase && vt < ImageBase + 0x6000000) {
                std::printf(" | vtable %#llx:", (unsigned long long)(vt - ImageBase));
                for (int k = 0; k < 24; ++k) {
                    const u64 fn = JobProbeWord(vt + k * 8);
                    std::printf(" %#llx", (unsigned long long)(fn >= ImageBase ? fn - ImageBase : fn));
                }
            }
            std::printf("\n");
        };
        for (const char* at = env; *at;) {
            char* end = nullptr;
            const u64 object = std::strtoull(at, &end, 16);
            dump("object", object);
            for (int i = 1; i < 8; ++i) {
                const u64 sub = JobProbeWord(object + i * 8);
                if ((sub >= ImageBase && sub < ImageBase + 0x6000000) || (sub >= 0x1000000000ull && sub < 0x1200000000ull)) {
                    if (const u64 vt = JobProbeWord(sub); vt >= ImageBase && vt < ImageBase + 0x6000000) {
                        char what[32];
                        std::snprintf(what, sizeof(what), "  field +%x ->", i * 8);
                        dump(what, sub);
                    }
                }
            }
            at = *end ? end + 1 : end;
        }
    }
}

bool InstallHeapGrow() {
    // On by default (2026-10-23); BB_HEAP_GROW=0: the heap whole from the start.
    if (bb_env_off("BB_HEAP_GROW")) {
        return false;
    }
    // GFX_GraphicsPrivate's size in the game's heap table: read by the runtime when the game
    // allocates it (the startup patches may have grown it by then, for large outputs).
    const auto* size_at = reinterpret_cast<const std::uint64_t*>(ImageBase + 0x4736dd8);
    const u64 initial = bb_env_int("BB_HEAP_GROW_INITIAL_MB", 512) << 20;
    constexpr std::array<u8, 7> grown{0x49, 0x89, 0xc5, 0x49, 0x8b, 0x04, 0x24};
    if (!Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x1fc5477), grown, 0,
                                    HeapGrowHook)) {
        return false;
    }
    // The same after A's allocation from the heap's top (0x20882f0, vtable slot 0x88: FastBiHeap
    // when its small-block allocator is full).
    // BB_HEAP_TOP_HOOK=0: off (the GPU faults of 2026-10-23 were the mirror copies' bug, fixed).
    if (const char* top = bb_env("BB_HEAP_TOP_HOOK"); !(top && top[0] == '0') &&
        !Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x1fc56f7), grown, 1,
                                    HeapGrowHook)) {
        std::printf("Heap grow: the top allocation hook is not installed\n");
    }
    // mov rax, [rbx + 0xe8]: right after A's core free returns (lock held): freed blocks' images
    // and VRAM copies go, free pieces go back. BB_HEAP_SHRINK=0: grow only. (The black blotches and
    // the lost reflection of 2026-10-25 were the freed blocks' images dropped at the frame's end,
    // after the block was handed out again and drawn into: Rasterizer::ProcessDeadImagesInOrder.)
    constexpr std::array<u8, 7> freed{0x48, 0x8b, 0x83, 0xe8, 0x00, 0x00, 0x00};
    if (!bb_env_off("BB_HEAP_SHRINK") &&
        !Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x1fc560a), freed, 0,
                                    HeapShrinkHook)) {
        std::printf("Heap grow: the free hook is not installed (grow only)\n");
    }
    runtime_memory_set_grow(size_at, initial);
    return true;
}

bool InstallHeapTrace() {
    if (!bb_env_on("BB_HEAP_TRACE")) {
        return false;
    }
    // mov rdi, r15; mov rsi, r14; mov rdx, r13: the call of the allocator's init for a heap of
    // the table.
    constexpr std::array<u8, 9> init{0x4c, 0x89, 0xff, 0x4c, 0x89, 0xf6, 0x4c, 0x89, 0xea};
    // add rsp, 0xc8; pop rbx: the allocate function's epilogue, rax the block.
    constexpr std::array<u8, 8> done{0x48, 0x81, 0xc4, 0xc8, 0x00, 0x00, 0x00, 0x5b};
    bool ok = Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x1fc1194), init, 0,
                                         HeapInitTrace);
    // mov r13, rax; mov rax, [r12]: right after A's core allocation returns (lock held).
    constexpr std::array<u8, 7> grown{0x49, 0x89, 0xc5, 0x49, 0x8b, 0x04, 0x24};
    ok &= Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x1fc5477), grown, 0,
                                     HeapGrowObserve);
    ok &= Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x1fbf9f3), done, 0,
                                     HeapAllocTrace);
    return ok;
}

} // namespace BbGuestHooks
