// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: two-stage draw pipeline (docs/parallel_gpu.md, "Two-stage draw pipeline").
//
// The GPU command thread (stage A) decodes PM4, keeps the register file and selects pipelines;
// for a direct draw it writes a packet (the register blocks changed since the previous packet,
// the stages' user data, the draw parameters) into this ring and goes on decoding. The draw
// recording thread (stage B) owns the caches, the barriers and the command recording while
// packets are in flight: it applies each packet to its own copy of the registers and runs the
// rest of the draw. Anything else stage A does on that state waits for stage B to run dry
// (Drain) and then runs on stage A as before.

#pragma once

#include <cstdlib>
#include "bbport_threads.h"
#include "bbport_sections.h"
#include "bbport_timeline.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cstring>
#include <memory>
#include <cstdio>
#include <thread>
#include <vector>
#include "bbport_cpu.h"

#include "common/assert.h"
#include "common/thread.h"
#include "common/types.h"

namespace Vulkan {

class DrawPipe {
public:
    /// Runs a packet on stage B.
    using Handler = void (*)(void* context, const u8* packet, u32 size);

    DrawPipe(Handler handler_, void* context_) : handler{handler_}, context{context_} {
        ring = std::make_unique<u8[]>(Capacity);
        thread = std::jthread([this](std::stop_token stop) { Run(stop); });
    }

    ~DrawPipe() {
        Drain();
        for (auto& worker : resolvers) {
            worker.request_stop();
        }
        resolve_wake.fetch_add(1, std::memory_order_seq_cst);
        resolve_wake.notify_all();
        resolvers.clear(); // joined before the ring goes
        thread.request_stop();
        published.fetch_add(0, std::memory_order_seq_cst);
        wake.fetch_add(1, std::memory_order_seq_cst);
        wake.notify_one();
    }

    /// True on the draw recording thread.
    [[nodiscard]] static bool OnStageB() noexcept {
        return on_stage_b;
    }

    [[nodiscard]] u32 StageBThreadId() const noexcept {
        return stage_b_tid.load(std::memory_order_acquire);
    }

    /// Stage A: space for a packet of `size` bytes (8-byte aligned), valid until Commit().
    u8* Begin(u32 size) {
        size = Align(size + sizeof(Header));
        ASSERT(size <= Capacity / 4);
        u64 at = head;
        const u64 offset = at % Capacity;
        if (offset + size > Capacity) {
            // Wrap: the rest of the ring is skipped (a header with zero size marks it).
            WaitForSpace(at, Capacity - offset);
            reinterpret_cast<Header*>(ring.get() + offset)->size = 0;
            at += Capacity - offset;
            head = at;
            published.store(at, std::memory_order_release);
        }
        WaitForSpace(at, size);
        pending_size = size;
        auto* header = reinterpret_cast<Header*>(ring.get() + at % Capacity);
        header->size = size;
        return reinterpret_cast<u8*>(header + 1);
    }

    /// Stage A: hands the packet from Begin() to stage B. `resolve`: a resolve worker completes
    /// it first (StartResolvers; without workers the flag is ignored).
    void Commit(u32 payload_size, bool resolve = false) {
        reinterpret_cast<Header*>(ring.get() + head % Capacity)->payload = payload_size;
        if (!resolvers.empty()) {
            auto& slot = slots[(packets + 1) & SlotMask];
            slot.at = head;
            slot.state.store(resolve ? SlotPending : SlotDone, std::memory_order_release);
        }
        head += pending_size;
        published.store(head, std::memory_order_seq_cst);
        ++packets;
        if (!resolvers.empty()) {
            published_packets.store(packets, std::memory_order_seq_cst);
            if (resolvers_sleeping.load(std::memory_order_seq_cst)) {
                resolve_wake.fetch_add(1, std::memory_order_seq_cst);
                resolve_wake.notify_all();
            }
        }
        if (sleeping.load(std::memory_order_seq_cst)) {
            wake.fetch_add(1, std::memory_order_seq_cst);
            wake.notify_one();
        }
    }

    /// Stage 2 (docs/pipeline_4stage.ru.md): `count` threads complete the packets committed with
    /// `resolve` in packet order of claim (each worker sees its own packets in increasing order),
    /// before stage B runs them. Called once, before any packet.
    using Resolver = void (*)(void* context, u8* packet, u32 size, u64 number, u32 worker);
    void StartResolvers(Resolver resolver_, u32 count) {
        resolver = resolver_;
        slots = std::make_unique<Slot[]>(SlotMask + 1);
        for (u32 i = 0; i < count; ++i) {
            resolvers.emplace_back([this, i](std::stop_token stop) { ResolverRun(stop, i); });
        }
    }
    [[nodiscard]] u32 ResolverCount() const noexcept {
        return static_cast<u32>(resolvers.size());
    }
    /// Stage A: waits until every packet up to `number` is resolved.
    void WaitResolved(u64 number) {
        if (resolvers.empty()) {
            return;
        }
        number = std::min(number, packets);
        while (resolved_prefix < number) {
            if (slots[(resolved_prefix + 1) & SlotMask].state.load(std::memory_order_acquire) ==
                SlotDone) {
                ++resolved_prefix;
            } else {
                ++resolve_waits;
                BbCpu::Pause();
            }
        }
    }
    u64 resolve_waits = 0; ///< statistics (stage A): spins waiting for resolve workers

    /// Why stage A drained (statistics): a PM4 opcode, or one of these.
    enum Reason : u32 {
        ReasonConstRam = 256,
        ReasonCommands,
        ReasonCompute,
        ReasonSubmissionEnd,
        ReasonDraw,
        ReasonRasterizer,
        NumReasons,
    };
    std::array<u64, NumReasons> drains_by_reason{};
    std::array<u64, NumReasons> cycles_by_reason{};

    /// Stage A: waits until stage B has run every committed packet.
    void Drain(u32 reason = ReasonRasterizer) {
        if (consumed.load(std::memory_order_acquire) == head) {
            return;
        }
        // consumed never passes head: reaching it is being equal to it.
        WaitForStageB(consumed, head, WaitPosition, reason);
    }

    /// Stage A: the number the next committed packet gets (packets are numbered from 1), and
    /// whether stage B has run the packet with that number.
    [[nodiscard]] u64 NextPacket() const noexcept {
        return packets + 1;
    }
    [[nodiscard]] bool ReachedPacket(u64 number) const noexcept {
        return consumed_packets.load(std::memory_order_acquire) >= number;
    }
    /// Stage B: the number of the packet it runs.
    [[nodiscard]] u64 RunningPacket() const noexcept {
        return consumed_packets.load(std::memory_order_relaxed) + 1;
    }
    /// Stage A: waits until stage B has run the packet `number` (Drain: all of them).
    void WaitForPacket(u64 number, u32 reason) {
        if (ReachedPacket(number)) {
            return;
        }
        WaitForStageB(consumed_packets, number, WaitPacket, reason);
    }

    /// Stage A: position after the last committed packet; Reached(position) once B ran it.
    [[nodiscard]] u64 Head() const noexcept {
        return head;
    }
    [[nodiscard]] bool Reached(u64 position) const noexcept {
        return consumed.load(std::memory_order_acquire) >= position;
    }

    [[nodiscard]] bool Idle() const noexcept {
        return consumed.load(std::memory_order_acquire) == head;
    }
    /// Stage B: the position of the packet it runs (packets before it have run).
    [[nodiscard]] u64 Consumed() const noexcept {
        return consumed.load(std::memory_order_acquire);
    }
    /// Stage B: bytes of packets committed and not run yet (the work queued for this thread).
    [[nodiscard]] u64 Backlog() const noexcept {
        return published.load(std::memory_order_acquire) - consumed.load(std::memory_order_relaxed);
    }

    /// Statistics (stage A): packets, drains that had to wait, cycles waited, cycles stage B
    /// spent running packets.
    u64 packets = 0, drains = 0, drain_cycles = 0;
    [[nodiscard]] u64 BusyCycles() const noexcept {
        return busy_cycles.load(std::memory_order_relaxed);
    }

private:
#if defined(__APPLE__) && defined(__aarch64__)
    // bbport (macOS): Apple Silicon's cache lines are 128 bytes; 64-byte alignment left fields
    // of the two stages (published and consumed) on one line.
    static constexpr std::size_t CacheLine = 128;
#else
    static constexpr std::size_t CacheLine = 64;
#endif

    /// What stage A sleeps on (stage_a_waiting): consumed reaching a position, or
    /// consumed_packets reaching a packet number.
    enum WaitKind : u64 { WaitPosition = 0, WaitPacket = 1 };

    /// bbport (macOS): BB_DRAIN_SPIN_US, how long stage A spins in Drain and WaitForPacket
    /// before it sleeps. -1: spins, then yields, until stage B is there (the original wait,
    /// the default off macOS). On macOS 50: there the spin's Pause is the arm64 `yield` (about
    /// nothing), the old wait ended in sched_yield calls, and the GPU command thread spent most
    /// of a GPU-bound frame burning a core in them (a core the game, under Rosetta, wants, and
    /// on fanless Macs power).
    static int DrainSpinUs() {
        static const int value = [] {
            if (const char* env = std::getenv("BB_DRAIN_SPIN_US")) {
                return std::max(-1, std::atoi(env));
            }
#ifdef __APPLE__
            return 50;
#else
            return -1;
#endif
        }();
        return value;
    }

    /// Stage A (Drain, WaitForPacket; one thread): waits until stage B has moved `word`
    /// (consumed or consumed_packets, which only grow) to `target` or past it.
    ///
    /// Sleeping without losing the wakeup. Stage A publishes what it waits for
    /// (stage_a_waiting) and then reads `word`; stage B moves `word` and then reads
    /// stage_a_waiting (WakeStageA), notifying `word` when the value it stored reaches the
    /// target (under wait_mutex, which A holds from its check to its wait). All four are
    /// seq_cst, so they fall in one total order. Take B's store X that
    /// first reaches the target:
    /// - X before A's store of stage_a_waiting: A's later load of `word` (seq_cst, after
    ///   its own store) sees X or a later value, the target is reached, A does not sleep.
    /// - X after it: B's load of stage_a_waiting (after X) sees the target (only A writes
    ///   it, and A clears it only once it has seen the target reached), so B notifies.
    ///   A checks `word` and waits with wait_mutex held from the check into the wait, and B
    ///   notifies after taking that mutex: either A's check sees X, or A is already waiting
    ///   when B notifies.
    /// Earlier stores of B that do not reach the target need no notify: the one that does
    /// comes later. Wakeups without the target reached (spurious, or a stale target seen by
    /// B after A cleared it) only loop. The waits are 1:1 with stage A's single thread, so
    /// one stage_a_waiting word serves both kinds.
    void WaitForStageB(const std::atomic<u64>& word, u64 target, WaitKind kind, u32 reason) {
        ++drains;
        const u32 slot = reason < NumReasons ? reason : ReasonRasterizer;
        ++drains_by_reason[slot];
        const u64 start = BbCpu::Ticks();
        if (const int spin_us = DrainSpinUs(); spin_us < 0) {
            for (u32 spins = 0; word.load(std::memory_order_acquire) < target; ++spins) {
                if (spins < 4096) {
                    BbCpu::Pause();
                } else {
                    std::this_thread::yield();
                }
            }
        } else {
            // bbport (macOS): spin for a fixed time (the drains within a frame are short),
            // then sleep until stage B gets there (the once-a-frame drain waits for the GPU).
            const u64 spin_until = start + BbCpu::TicksFromUs(u64(spin_us));
            while (word.load(std::memory_order_acquire) < target) {
                if (BbCpu::Ticks() >= spin_until) {
                    stage_a_waiting.store((target << 1) | kind, std::memory_order_seq_cst);
                    {
                        // A condition variable, not word.wait(): libc++ waits on a u64
                        // through a shared table (its missed-notify fix of 2025 depends on
                        // the macOS version), and a lost wakeup here would be a hang. B
                        // notifies under the mutex, after A checked under it; the timeout
                        // bounds whatever else could go wrong to a millisecond.
                        std::unique_lock lock{wait_mutex};
                        while (word.load(std::memory_order_seq_cst) < target) {
                            wait_cv.wait_for(lock, std::chrono::milliseconds(1));
                        }
                    }
                    stage_a_waiting.store(0, std::memory_order_relaxed);
                    break;
                }
                BbCpu::Pause();
            }
        }
        const u64 waited = BbCpu::Ticks() - start;
        drain_cycles += waited;
        cycles_by_reason[slot] += waited;
    }

    /// Stage B, right after it moved consumed to `at` and consumed_packets to `ran` (both
    /// seq_cst stores): wakes stage A if it sleeps until one of them got there
    /// (WaitForStageB). Otherwise one load.
    void WakeStageA(u64 at, u64 ran) {
        const u64 waiting = stage_a_waiting.load(std::memory_order_seq_cst);
        if (waiting == 0) [[likely]] {
            return;
        }
        const u64 target = waiting >> 1;
        if (((waiting & 1) == WaitPacket ? ran : at) >= target) {
            { std::scoped_lock lock{wait_mutex}; }
            wait_cv.notify_one();
        }
    }

    struct Header {
        u32 size;    ///< bytes to the next packet (0: wrap to the ring start)
        u32 payload; ///< bytes written by the producer
    };
    static constexpr u64 Capacity = 16ull << 20;

    static constexpr u32 Align(u32 size) {
        return (size + 63) & ~63u;
    }

    void WaitForSpace(u64 at, u64 size) {
        while (at + size - consumed.load(std::memory_order_acquire) > Capacity) {
            BbCpu::Pause();
        }
    }

    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("bb:DrawRec");
        on_stage_b = true;
        BbSections::recording_thread = true;
        stage_b_tid.store(static_cast<u32>(BbThreads::HostTid()), std::memory_order_release);
        u64 at = 0;
        while (true) {
            // Packets follow each other within microseconds while a frame is decoded: spin,
            // and sleep only after a longer pause (between frames).
            u64 available = published.load(std::memory_order_acquire);
            if (available == at) {
                BbTimeline::Note(BbTimeline::RecorderIdleBegin);
                // With few hardware threads (Steam Deck: 8) a long spin takes time from the
                // game's own threads. BB_PIPE_SPIN_US overrides.
                static const auto spin_time = std::chrono::microseconds([] {
                    if (const char* env = std::getenv("BB_PIPE_SPIN_US")) {
                        return std::max(0, std::atoi(env));
                    }
                    return BbThreads::Available() >= 12 ? 200 : 50;
                }());
                const auto spin_until = std::chrono::steady_clock::now() + spin_time;
                for (u32 spins = 1; available == at; ++spins) {
                    if (stop.stop_requested()) {
                        return;
                    }
                    BbCpu::Pause();
                    if (!(spins & 255) && std::chrono::steady_clock::now() >= spin_until) {
                        const u32 seen = wake.load(std::memory_order_seq_cst);
                        sleeping.store(true, std::memory_order_seq_cst);
                        if (published.load(std::memory_order_seq_cst) == at &&
                            !stop.stop_requested()) {
                            wake.wait(seen, std::memory_order_seq_cst);
                        }
                        sleeping.store(false, std::memory_order_relaxed);
                    }
                    available = published.load(std::memory_order_acquire);
                }
                BbTimeline::Note(BbTimeline::RecorderIdleEnd);
            }
            const auto* header = reinterpret_cast<const Header*>(ring.get() + at % Capacity);
            if (header->size == 0) {
                at += Capacity - at % Capacity;
                // bbport (macOS): seq_cst, then WakeStageA (WaitForStageB's ordering), only
                // when stage A may sleep (else release, as before).
                if (DrainSpinUs() >= 0) {
                    consumed.store(at, std::memory_order_seq_cst);
                    WakeStageA(at, consumed_packets.load(std::memory_order_relaxed));
                } else {
                    consumed.store(at, std::memory_order_release);
                }
                continue;
            }
            if (!resolvers.empty()) {
                // Its resolve worker may still be completing it.
                const auto& slot = slots[(consumed_packets.load(std::memory_order_relaxed) + 1) &
                                         SlotMask];
                for (u32 spins = 0; slot.state.load(std::memory_order_acquire) != SlotDone;
                     ++spins) {
                    if (spins < 4096) {
                        BbCpu::Pause();
                    } else {
                        std::this_thread::yield();
                    }
                }
            }
            const u64 start = BbCpu::Ticks();
            handler(context, reinterpret_cast<const u8*>(header + 1), header->payload);
            busy_cycles.fetch_add(BbCpu::Ticks() - start, std::memory_order_relaxed);
            at += header->size;
            // bbport (macOS): seq_cst, then WakeStageA (WaitForStageB's ordering), only when
            // stage A may sleep (else release, as before).
            if (DrainSpinUs() >= 0) {
                const u64 ran = consumed_packets.fetch_add(1, std::memory_order_seq_cst) + 1;
                consumed.store(at, std::memory_order_seq_cst);
                WakeStageA(at, ran);
            } else {
                consumed_packets.fetch_add(1, std::memory_order_release);
                consumed.store(at, std::memory_order_release);
            }
        }
    }

    void ResolverRun(std::stop_token stop, u32 worker) {
        char name[16];
        std::snprintf(name, sizeof(name), "bb:Resolve%u", worker);
        Common::SetCurrentThreadName(name);
        static const auto spin_time = std::chrono::microseconds([] {
            if (const char* env = std::getenv("BB_PIPE_SPIN_US")) {
                return std::max(0, std::atoi(env));
            }
            return BbThreads::Available() >= 12 ? 200 : 50;
        }());
        while (!stop.stop_requested()) {
            u64 number = next_claim.load(std::memory_order_relaxed);
            if (number > published_packets.load(std::memory_order_acquire)) {
                const auto spin_until = std::chrono::steady_clock::now() + spin_time;
                for (u32 spins = 1;
                     number > published_packets.load(std::memory_order_acquire); ++spins) {
                    if (stop.stop_requested()) {
                        return;
                    }
                    BbCpu::Pause();
                    if (!(spins & 255) && std::chrono::steady_clock::now() >= spin_until) {
                        const u32 seen = resolve_wake.load(std::memory_order_seq_cst);
                        resolvers_sleeping.fetch_add(1, std::memory_order_seq_cst);
                        if (number > published_packets.load(std::memory_order_seq_cst) &&
                            !stop.stop_requested()) {
                            resolve_wake.wait(seen, std::memory_order_seq_cst);
                        }
                        resolvers_sleeping.fetch_sub(1, std::memory_order_relaxed);
                    }
                    number = next_claim.load(std::memory_order_relaxed);
                }
                continue;
            }
            if (!next_claim.compare_exchange_weak(number, number + 1, std::memory_order_acq_rel)) {
                continue;
            }
            auto& slot = slots[number & SlotMask];
            if (slot.state.load(std::memory_order_acquire) == SlotPending) {
                auto* header = reinterpret_cast<Header*>(ring.get() + slot.at % Capacity);
                resolver(context, reinterpret_cast<u8*>(header + 1), header->payload, number,
                         worker);
                slot.state.store(SlotDone, std::memory_order_release);
            }
        }
    }

    // Packet slots of the resolve stage: as many as packets fit in the ring (64-byte minimum).
    static constexpr u64 SlotMask = (Capacity / 64) - 1;
    enum : u32 { SlotPending = 1, SlotDone = 2 };
    struct Slot {
        u64 at = 0; ///< the packet's header position
        std::atomic<u32> state{0};
    };
    std::unique_ptr<Slot[]> slots;
    Resolver resolver = nullptr;
    u64 resolved_prefix = 0; ///< stage A
    alignas(CacheLine) std::atomic<u64> published_packets{0};
    alignas(CacheLine) std::atomic<u64> next_claim{1};
    alignas(CacheLine) std::atomic<u32> resolve_wake{0};
    std::atomic<u32> resolvers_sleeping{0};
    std::vector<std::jthread> resolvers;

    static inline thread_local bool on_stage_b = false;

    Handler handler;
    void* context;
    std::unique_ptr<u8[]> ring;
    u64 head = 0; ///< stage A's write position
    u32 pending_size = 0;
    alignas(CacheLine) std::atomic<u64> published{0};
    // Written by stage B for every packet.
    alignas(CacheLine) std::atomic<u64> consumed{0};
    std::atomic<u64> consumed_packets{0}; ///< packets stage B has run (NextPacket numbers)
    // bbport (macOS): statistics; here (with what stage B writes anyway), it was next to stage
    // A's own fields (false sharing).
    std::atomic<u64> busy_cycles{0};
    // Rarely written; read for every packet (stage_a_waiting, by stage B) or every commit
    // (sleeping, by stage A).
    alignas(CacheLine) std::atomic<u32> wake{0};
    std::atomic<bool> sleeping{false};
    /// bbport (macOS): what stage A sleeps until (WaitForStageB): 0 nothing, else the target
    /// shifted left by one, ORed with its WaitKind.
    std::atomic<u64> stage_a_waiting{0};
    std::mutex wait_mutex;           ///< WaitForStageB and WakeStageA
    std::condition_variable wait_cv; ///< stage A sleeps on it (WaitForStageB)
    std::atomic<u32> stage_b_tid{0};
    std::jthread thread;
};

} // namespace Vulkan
