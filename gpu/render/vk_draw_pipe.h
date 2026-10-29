// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
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
#include "shim/bb_env.h"
#include "bbport_threads.h"
#include "bbport_sections.h"
#include "bbport_timeline.h"

#include <array>
#include <atomic>
#include <chrono>
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
        ++drains;
        const u32 slot = reason < NumReasons ? reason : ReasonRasterizer;
        ++drains_by_reason[slot];
        const u64 start = BbCpu::Ticks();
        for (u32 spins = 0; consumed.load(std::memory_order_acquire) != head; ++spins) {
            if (spins < 4096) {
                BbCpu::Pause();
            } else {
                std::this_thread::yield();
            }
        }
        const u64 waited = BbCpu::Ticks() - start;
        drain_cycles += waited;
        cycles_by_reason[slot] += waited;
    }

    /// Stage A: the number the next committed packet gets (packets are numbered from 1), and
    /// whether stage B has run the packet with that number.
    [[nodiscard]] u64 NextPacket() const noexcept {
        return packets + 1;
    }
    [[nodiscard]] u64 ConsumedPackets() const noexcept {
        return consumed_packets.load(std::memory_order_acquire);
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
        ++drains;
        const u32 slot = reason < NumReasons ? reason : ReasonRasterizer;
        ++drains_by_reason[slot];
        const u64 start = BbCpu::Ticks();
        for (u32 spins = 0; !ReachedPacket(number); ++spins) {
            if (spins < 4096) {
                BbCpu::Pause();
            } else {
                std::this_thread::yield();
            }
        }
        const u64 waited = BbCpu::Ticks() - start;
        drain_cycles += waited;
        cycles_by_reason[slot] += waited;
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
    std::atomic<u64> busy_cycles{0};

private:
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
        BbThreads::MakeCritical();
        on_stage_b = true;
        BbSections::recording_thread = true;
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
                    if (const char* env = bb_env("BB_PIPE_SPIN_US")) {
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
                consumed.store(at, std::memory_order_release);
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
            consumed_packets.fetch_add(1, std::memory_order_release);
            consumed.store(at, std::memory_order_release);
        }
    }

    void ResolverRun(std::stop_token stop, u32 worker) {
        char name[16];
        std::snprintf(name, sizeof(name), "bb:Resolve%u", worker);
        Common::SetCurrentThreadName(name);
        static const auto spin_time = std::chrono::microseconds([] {
            if (const char* env = bb_env("BB_PIPE_SPIN_US")) {
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
    alignas(64) std::atomic<u64> published_packets{0};
    alignas(64) std::atomic<u64> next_claim{1};
    alignas(64) std::atomic<u32> resolve_wake{0};
    std::atomic<u32> resolvers_sleeping{0};
    std::vector<std::jthread> resolvers;

    static inline thread_local bool on_stage_b = false;
    Handler handler;
    void* context;
    std::unique_ptr<u8[]> ring;
    u64 head = 0; ///< stage A's write position
    u32 pending_size = 0;
    alignas(64) std::atomic<u64> published{0};
    alignas(64) std::atomic<u64> consumed{0};
    std::atomic<u64> consumed_packets{0}; ///< packets stage B has run (NextPacket numbers)
    alignas(64) std::atomic<u32> wake{0};
    std::atomic<bool> sleeping{false};
    std::jthread thread;
};

} // namespace Vulkan
