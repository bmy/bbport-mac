// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport memory: the lock of a tracked region. Four bytes, unlocked when zero (tables of them
// live in zero-filled memory committed on first touch); spins briefly, as holders keep it for a
// few page-bit updates, then sleeps on a futex. Usable from the fault handler.
#pragma once

#include <atomic>

#include "bbport_cpu.h"
#include "common/types.h"

namespace BbMem {

class PageLock {
public:
    void lock() noexcept {
        u32 expected = 0;
        if (state.compare_exchange_strong(expected, 1, std::memory_order_acquire)) [[likely]] {
            return;
        }
        LockSlow();
    }
    bool try_lock() noexcept {
        u32 expected = 0;
        return state.compare_exchange_strong(expected, 1, std::memory_order_acquire);
    }
    void unlock() noexcept {
        if (state.exchange(0, std::memory_order_release) == 2) [[unlikely]] {
            state.notify_one();
        }
    }

private:
    void LockSlow() noexcept {
        for (int spin = 0; spin < 128; ++spin) {
            BbCpu::Pause();
            u32 expected = 0;
            if (state.load(std::memory_order_relaxed) == 0 &&
                state.compare_exchange_weak(expected, 1, std::memory_order_acquire)) {
                return;
            }
        }
        // 2: locked with sleepers (the unlocker wakes one).
        while (state.exchange(2, std::memory_order_acquire) != 0) {
            state.wait(2, std::memory_order_relaxed);
        }
    }

    std::atomic<u32> state{0};
};

} // namespace BbMem
