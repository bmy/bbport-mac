// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: bb-gpu's copy of the game's write trap reasons (runtime_memory_trap, bbport 0.5), one
// byte per 4 KiB page below Limit, with the order of its changes (protocol 12). The game process
// keeps the page protections and its own table, which is the one that counts; bb-gpu sets this
// one with its own trap calls and clears it where the game's mappings change, as runtime_memory.c
// trap_forget (remote_back.cpp). Header-only (tests/test_remote_protect_order.cpp uses it too).
//
// Every change gets a sequence number under the lock, the game process records the newest it
// applied per page (runtime_memory_trap_seq), and a lift of reasons the game holds and nobody
// here does (Stale) carries the number current when it was checked: the game drops it on a page a
// newer change already reached. So a trap an owner sets after the check is never undone by the
// lift, whichever arrives first, and nothing waits for the game under the lock.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sys/mman.h>

namespace BbRemote {

class TrapCopy {
public:
    static constexpr std::uint64_t Limit = 1ull << 40;

    struct Order {
        std::uint64_t address = 0, size = 0; ///< clipped to Limit; size 0: nothing to send
        std::uint64_t seq = 0;
    };

    /// `reason` on (or off) for the pages of [address, address + size); what to send.
    Order Change(std::uint64_t address, std::uint64_t size, unsigned reason, bool on) {
        if (!size || address >= Limit || address + size < address) {
            return {};
        }
        std::uint8_t* table = Table();
        if (!table) {
            return {};
        }
        const std::uint64_t end = std::min(address + size, Limit);
        std::scoped_lock lock{mutex};
        for (std::uint64_t page = address >> 12; page <= (end - 1) >> 12; ++page) {
            const std::uint8_t old = table[page];
            __atomic_store_n(&table[page], on ? std::uint8_t(old | reason)
                                              : std::uint8_t(old & ~reason),
                             __ATOMIC_RELEASE);
        }
        return {address, end - address, ++seq};
    }

    struct Lift {
        unsigned reasons = 0; ///< the game's reasons for the page that nobody here holds
        std::uint64_t seq = 0; ///< the newest change made before the check
    };

    /// Of the game's `game_reasons` for the page at `address`, the ones nobody here holds.
    Lift Stale(std::uint64_t address, unsigned game_reasons) {
        game_reasons &= 0xffu;
        if (!game_reasons || address >= Limit) {
            return {};
        }
        std::scoped_lock lock{mutex};
        const std::uint8_t* t = table_.load(std::memory_order_acquire);
        return {game_reasons & ~unsigned(t ? t[address >> 12] : 0), seq};
    }

    /// No reasons in [start, end) any more (the game's mappings there changed).
    void Forget(std::uint64_t start, std::uint64_t end) {
        std::uint8_t* t = table_.load(std::memory_order_acquire);
        if (!t || start >= Limit) {
            return;
        }
        end = std::min(end, Limit);
        std::scoped_lock lock{mutex};
        std::memset(t + (start >> 12), 0, ((end + 4095) >> 12) - (start >> 12));
    }

    unsigned Reasons(std::uint64_t address) const {
        const std::uint8_t* t = table_.load(std::memory_order_acquire);
        return t && address < Limit ? __atomic_load_n(&t[address >> 12], __ATOMIC_ACQUIRE) : 0;
    }

private:
    std::uint8_t* Table() {
        if (std::uint8_t* t = table_.load(std::memory_order_acquire)) {
            return t;
        }
        void* bytes = mmap(nullptr, Limit >> 12, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (bytes == MAP_FAILED) {
            std::perror("GPU process: write trap table");
            return nullptr;
        }
        std::uint8_t* expected = nullptr;
        if (!table_.compare_exchange_strong(expected, static_cast<std::uint8_t*>(bytes),
                                            std::memory_order_acq_rel)) {
            munmap(bytes, Limit >> 12);
            return expected;
        }
        return static_cast<std::uint8_t*>(bytes);
    }

    std::atomic<std::uint8_t*> table_{nullptr};
    std::mutex mutex; ///< never held while waiting for anything
    std::uint64_t seq = 0;
};

} // namespace BbRemote
