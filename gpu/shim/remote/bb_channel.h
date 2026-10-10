// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the channel between the game process (bb-probe, x86-64 under Rosetta on macOS) and the
// native GPU process (bb-gpu, arm64), docs/macos-native-gpu.md. Header-only, no dependencies.
//
// One shared memory region holds two rings, one per direction, and a set of reply slots:
// - Messages go through a ring in order (one process-local writer lock per ring, one reader
//   thread per process). A message is a header and up to MaxPayload bytes.
// - A synchronous call is a ring message that names a reply slot; the reader runs the handler and
//   writes the result into that slot. Calls stay in order with the asynchronous messages before
//   them (a write fault after the mapping it touches).
// - Waiting sleeps on a 32-bit word in the shared region (os_sync_wait_on_address with the shared
//   flag on macOS 14.4+, a shared futex on Linux) after a short spin.
// Both sides only use std::atomic with acquire/release, so x86-64 (TSO) and arm64 agree.
#pragma once

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#if defined(__APPLE__)
#include <os/os_sync_wait_on_address.h>
#elif defined(__linux__)
#include <climits>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace BbRemote {

inline constexpr std::uint32_t ChannelMagic = 0x62624348; // "bbCH"
inline constexpr std::uint32_t ChannelVersion = 1;
inline constexpr std::uint32_t RingBytes = 1u << 20;
inline constexpr std::uint32_t MaxPayload = 4096;
inline constexpr std::uint32_t NumReplySlots = 64;
inline constexpr std::uint32_t NoReply = ~0u;

// ---- waiting on a shared word ------------------------------------------------------------------

/// Sleeps while *word == expected (spurious returns are fine: callers loop).
inline void WaitWord(std::atomic<std::uint32_t>& word, std::uint32_t expected) {
    static_assert(sizeof(std::atomic<std::uint32_t>) == 4);
#if defined(__APPLE__)
    // macOS 14.4+ (KosmicKrisp needs 26 anyway); older systems poll.
    if (__builtin_available(macOS 14.4, *)) {
        os_sync_wait_on_address(&word, expected, 4, OS_SYNC_WAIT_ON_ADDRESS_SHARED);
    } else {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
#elif defined(__linux__)
    syscall(SYS_futex, &word, FUTEX_WAIT, expected, nullptr, nullptr, 0);
#else
    std::this_thread::yield();
#endif
}

inline void WakeWord(std::atomic<std::uint32_t>& word) {
#if defined(__APPLE__)
    if (__builtin_available(macOS 14.4, *)) {
        os_sync_wake_by_address_all(&word, 4, OS_SYNC_WAKE_BY_ADDRESS_SHARED);
    }
#elif defined(__linux__)
    syscall(SYS_futex, &word, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
#endif
}

/// Spins briefly, then sleeps, until ready() (re-checked after every wake of `word`).
template <typename Ready>
void WaitUntil(std::atomic<std::uint32_t>& word, std::atomic<std::uint32_t>& sleepers,
               Ready&& ready, std::chrono::microseconds spin = std::chrono::microseconds(30)) {
    const auto spin_until = std::chrono::steady_clock::now() + spin;
    for (std::uint32_t i = 0; !ready(); ++i) {
        if ((i & 63) != 63 || std::chrono::steady_clock::now() < spin_until) {
#if defined(__x86_64__)
            __builtin_ia32_pause();
#elif defined(__aarch64__)
            __builtin_arm_yield();
#endif
            continue;
        }
        const std::uint32_t seen = word.load(std::memory_order_acquire);
        sleepers.fetch_add(1, std::memory_order_seq_cst);
        if (!ready()) {
            WaitWord(word, seen);
        }
        sleepers.fetch_sub(1, std::memory_order_relaxed);
    }
}

/// Bumps `word` and wakes its sleepers, if any.
inline void Signal(std::atomic<std::uint32_t>& word, std::atomic<std::uint32_t>& sleepers) {
    word.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers.load(std::memory_order_seq_cst) != 0) {
        WakeWord(word);
    }
}

// ---- shared layout -----------------------------------------------------------------------------

struct MessageHeader {
    std::uint32_t size;  ///< header + payload, rounded up to 8 bytes; 0 = skip to the ring start
    std::uint32_t type;
    std::uint32_t reply; ///< reply slot of a synchronous call, or NoReply
    std::uint32_t payload_size;
};
static_assert(sizeof(MessageHeader) == 16);

struct alignas(64) RingControl {
    alignas(64) std::atomic<std::uint64_t> head{0}; ///< bytes written (writer)
    alignas(64) std::atomic<std::uint64_t> tail{0}; ///< bytes consumed (reader)
    alignas(64) std::atomic<std::uint32_t> data_word{0};   ///< bumped when a message is published
    std::atomic<std::uint32_t> data_sleepers{0};
    alignas(64) std::atomic<std::uint32_t> space_word{0};  ///< bumped when space is freed
    std::atomic<std::uint32_t> space_sleepers{0};
};

struct alignas(64) ReplySlot {
    std::atomic<std::uint32_t> state{0}; ///< Free, Waiting, Done
    std::atomic<std::uint32_t> sleepers{0};
    std::uint32_t size = 0;
    std::uint32_t pad = 0;
    std::uint8_t data[MaxPayload];
};
enum : std::uint32_t { SlotFree = 0, SlotWaiting = 1, SlotDone = 2 };

struct ChannelLayout {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint64_t total_size;
    RingControl control[2];        ///< [0] frontend -> backend, [1] backend -> frontend
    ReplySlot slots[2][NumReplySlots]; ///< [direction of the call]
    alignas(64) std::uint8_t rings[2][RingBytes];
};

inline constexpr std::size_t ChannelBytes = sizeof(ChannelLayout);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free &&
              std::atomic<std::uint32_t>::is_always_lock_free);

enum class Side : std::uint32_t { Frontend = 0, Backend = 1 };

// ---- one side's view ---------------------------------------------------------------------------

/// A received message: valid until the handler returns.
struct Message {
    std::uint32_t type;
    const void* data;
    std::uint32_t size;
};

class Channel {
public:
    /// `memory` is ChannelBytes of shared memory, zeroed by the creator (Init) before the peer
    /// attaches.
    Channel(void* memory, Side side_) : layout{static_cast<ChannelLayout*>(memory)}, side{side_} {}

    static void Init(void* memory) {
        auto* layout = new (memory) ChannelLayout{};
        layout->magic = ChannelMagic;
        layout->version = ChannelVersion;
        layout->total_size = sizeof(ChannelLayout);
    }

    [[nodiscard]] bool Valid() const {
        return layout->magic == ChannelMagic && layout->version == ChannelVersion &&
               layout->total_size == sizeof(ChannelLayout);
    }

    /// Queues a message to the peer (blocks while its ring is full).
    void Send(std::uint32_t type, const void* data, std::uint32_t size) {
        Write(type, data, size, NoReply);
    }

    /// Sends a message and waits for the peer's handler to answer it; returns the answer's size.
    std::uint32_t Call(std::uint32_t type, const void* data, std::uint32_t size, void* reply,
                       std::uint32_t reply_capacity) {
        ReplySlot* slots = layout->slots[Out()];
        std::uint32_t index = 0;
        for (;; index = (index + 1) % NumReplySlots) {
            std::uint32_t expected = SlotFree;
            if (slots[index].state.compare_exchange_strong(expected, SlotWaiting,
                                                            std::memory_order_acq_rel)) {
                break;
            }
            if (index == NumReplySlots - 1) {
                std::this_thread::yield(); // all busy: more callers than slots
            }
        }
        ReplySlot& slot = slots[index];
        Write(type, data, size, index);
        WaitUntil(slot.state, slot.sleepers,
                  [&] { return slot.state.load(std::memory_order_acquire) == SlotDone; });
        // The peer wrote the size: read it once and keep it within the slot.
        std::uint32_t full = slot.size;
        if (full > MaxPayload) {
            full = MaxPayload;
        }
        const std::uint32_t n = full < reply_capacity ? full : reply_capacity;
        if (n) {
            std::memcpy(reply, slot.data, n);
        }
        slot.state.store(SlotFree, std::memory_order_release);
        return full;
    }

    /// Reader thread: waits for the next message from the peer and runs `handler(message, reply)`
    /// with `reply(data, size)` for synchronous calls. Returns false once `stop()` is true.
    template <typename Handler, typename Stop>
    bool ReceiveOne(Handler&& handler, Stop&& stop) {
        RingControl& c = layout->control[In()];
        WaitUntil(c.data_word, c.data_sleepers, [&] {
            return stop() || c.head.load(std::memory_order_acquire) !=
                                 c.tail.load(std::memory_order_relaxed);
        });
        if (stop()) {
            return false;
        }
        // The other process wrote the ring, so nothing in it is trusted: the header is copied
        // once and checked against the ring's bounds, and the payload is copied out before the
        // handler sees it (the writer can't change it while it's being handled).
        std::uint64_t tail = c.tail.load(std::memory_order_relaxed);
        const std::uint64_t head = c.head.load(std::memory_order_acquire);
        const std::uint64_t available = head - tail;
        const std::uint32_t offset = static_cast<std::uint32_t>(tail % RingBytes);
        const std::uint8_t* ring = layout->rings[In()];
        std::uint32_t first_word;
        std::memcpy(&first_word, ring + offset, sizeof(first_word));
        if (first_word == 0) { // the writer skipped the end of the ring
            if (available < RingBytes - offset) {
                Corrupt("skip past the written data");
            }
            tail += RingBytes - offset;
            c.tail.store(tail, std::memory_order_release);
            Signal(c.space_word, c.space_sleepers);
            return true;
        }
        if (RingBytes - offset < sizeof(MessageHeader)) {
            Corrupt("header past the end of the ring");
        }
        MessageHeader header;
        std::memcpy(&header, ring + offset, sizeof(header));
        if (header.size != Align8(sizeof(MessageHeader) + header.payload_size) ||
            header.payload_size > MaxPayload || header.size > RingBytes - offset ||
            header.size > available ||
            (header.reply != NoReply && header.reply >= NumReplySlots)) {
            Corrupt("bad message header");
        }
        alignas(16) std::uint8_t payload[MaxPayload];
        if (header.payload_size) {
            std::memcpy(payload, ring + offset + sizeof(MessageHeader), header.payload_size);
        }
        const Message message{header.type, payload, header.payload_size};
        const std::uint32_t reply_index = header.reply;
        bool replied = false;
        auto reply = [&](const void* data, std::uint32_t size) {
            if (reply_index == NoReply || replied) {
                return;
            }
            ReplySlot& slot = layout->slots[In()][reply_index];
            slot.size = size < MaxPayload ? size : MaxPayload;
            if (slot.size) {
                std::memcpy(slot.data, data, slot.size);
            }
            replied = true;
            slot.state.store(SlotDone, std::memory_order_release);
            if (slot.sleepers.load(std::memory_order_seq_cst) != 0) {
                WakeWord(slot.state);
            }
        };
        handler(message, reply);
        reply(nullptr, 0); // a call the handler did not answer still returns
        c.tail.store(tail + header.size, std::memory_order_release);
        Signal(c.space_word, c.space_sleepers);
        return true;
    }

    /// Wakes this side's reader (after its stop condition became true).
    void WakeReader() {
        RingControl& c = layout->control[In()];
        Signal(c.data_word, c.data_sleepers);
    }

private:
    [[nodiscard]] std::uint32_t Out() const { return static_cast<std::uint32_t>(side); }
    [[nodiscard]] std::uint32_t In() const { return 1 - static_cast<std::uint32_t>(side); }

    static constexpr std::uint32_t Align8(std::uint32_t n) { return (n + 7) & ~7u; }

    /// The other process broke the protocol: stop rather than read or write out of bounds.
    [[noreturn]] static void Corrupt(const char* what) {
        std::fprintf(stderr, "bbport remote channel: %s; stopping\n", what);
        std::abort();
    }

    void Write(std::uint32_t type, const void* data, std::uint32_t size, std::uint32_t reply) {
        if (size > MaxPayload) {
            size = MaxPayload; // callers split larger data; never overruns the ring
        }
        const std::uint32_t total = Align8(sizeof(MessageHeader) + size);
        RingControl& c = layout->control[Out()];
        std::uint8_t* ring = layout->rings[Out()];
        std::scoped_lock lock{write_mutex};
        std::uint64_t head = c.head.load(std::memory_order_relaxed);
        const std::uint32_t offset = static_cast<std::uint32_t>(head % RingBytes);
        // A message never wraps: the rest of the ring is skipped (a zero size says so).
        const std::uint32_t skip = offset + total > RingBytes ? RingBytes - offset : 0;
        WaitUntil(c.space_word, c.space_sleepers, [&] {
            return head + skip + total - c.tail.load(std::memory_order_acquire) <= RingBytes;
        });
        if (skip) {
            reinterpret_cast<MessageHeader*>(ring + offset)->size = 0;
            head += skip;
        }
        auto* header = reinterpret_cast<MessageHeader*>(ring + head % RingBytes);
        header->size = total;
        header->type = type;
        header->reply = reply;
        header->payload_size = size;
        if (size) {
            std::memcpy(header + 1, data, size);
        }
        c.head.store(head + total, std::memory_order_release);
        Signal(c.data_word, c.data_sleepers);
    }

    ChannelLayout* layout;
    Side side;
    std::mutex write_mutex;
};

} // namespace BbRemote
