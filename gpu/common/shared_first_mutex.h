// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a reader-writer lock that never makes a reader wait for a writer that is only waiting:
// readers get in whenever no writer holds it. A thread that holds it shared may therefore take it
// shared again. One atomic word: the writer bit and the number of readers; waits are futexes.
#pragma once

#include <atomic>

#include "common/types.h"

namespace Common {

class SharedFirstMutex {
    static constexpr u32 Writer = 1u << 31;

public:
    void lock() {
        u32 expected = 0;
        while (!state.compare_exchange_weak(expected, Writer, std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
            if (expected != 0) {
                state.wait(expected, std::memory_order_relaxed);
            }
            expected = 0;
        }
    }

    [[nodiscard]] bool try_lock() {
        u32 expected = 0;
        return state.compare_exchange_strong(expected, Writer, std::memory_order_acquire,
                                             std::memory_order_relaxed);
    }

    void unlock() {
        state.store(0, std::memory_order_release);
        state.notify_all();
    }

    void lock_shared() {
        u32 current = state.load(std::memory_order_relaxed);
        for (;;) {
            if (current & Writer) {
                state.wait(current, std::memory_order_relaxed);
                current = state.load(std::memory_order_relaxed);
            } else if (state.compare_exchange_weak(current, current + 1, std::memory_order_acquire,
                                                   std::memory_order_relaxed)) {
                return;
            }
        }
    }

    [[nodiscard]] bool try_lock_shared() {
        u32 current = state.load(std::memory_order_relaxed);
        while (!(current & Writer)) {
            if (state.compare_exchange_weak(current, current + 1, std::memory_order_acquire,
                                            std::memory_order_relaxed)) {
                return true;
            }
        }
        return false;
    }

    void unlock_shared() {
        if (state.fetch_sub(1, std::memory_order_release) == 1) {
            state.notify_all(); // a writer may wait for the last reader
        }
    }

private:
    std::atomic<u32> state{0};
};

} // namespace Common
