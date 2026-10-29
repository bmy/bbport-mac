// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport vulkan: a timeline semaphore counting submissions. Each submission signals its tick;
// work tagged with a tick is done on the GPU once the semaphore reaches it.
#pragma once

#include <atomic>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;

class Semaphore {
public:
    explicit Semaphore(const Instance& instance);
    ~Semaphore();

    /// The tick the next submission signals.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return current_tick.load(std::memory_order_acquire);
    }
    /// The newest tick known to be reached (Refresh asks the GPU).
    [[nodiscard]] u64 KnownGpuTick() const noexcept {
        return gpu_tick.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool IsFree(u64 tick) const noexcept {
        return KnownGpuTick() >= tick;
    }
    /// Takes the current tick for a submission (returned) and moves on.
    [[nodiscard]] u64 NextTick() noexcept {
        return current_tick.fetch_add(1, std::memory_order_release);
    }
    [[nodiscard]] vk::Semaphore Handle() const noexcept {
        return *semaphore;
    }

    /// Reads the GPU's progress. At most once per BB_TIMELINE_POLL_US (50): each read is a call into the
    /// kernel driver (D3DKMT on Windows: half of the recording thread's time on a GTX 1050 laptop;
    /// an ioctl on RADV: 8.6 % on the Steam Deck); a read sooner keeps the known tick.
    void Refresh();
    /// Returns once the GPU reached `tick` (a lost device stops the port with a report).
    void Wait(u64 tick);

    /// Diagnostics: called when a Wait has gone on for a second (`done` false, once) and when
    /// that wait ends (`done` true), on the waiting thread.
    using SlowWaitHook = void (*)(const Semaphore& semaphore, u64 tick, u64 waited_ms, bool done);
    static inline std::atomic<SlowWaitHook> slow_wait_hook{nullptr};

private:
    const Instance& instance;
    vk::UniqueSemaphore semaphore;
    std::atomic<u64> gpu_tick{0};
    std::atomic<u64> current_tick{1};
    std::atomic<s64> last_read_tsc{0}; ///< Refresh: __rdtsc() of the last read
};

} // namespace Vulkan
