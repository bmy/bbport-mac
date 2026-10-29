// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: thread names, priorities and precise sleeps.
#pragma once

#include <algorithm>
#include <chrono>
#include <string>

#include "common/types.h"

namespace Common {

enum class ThreadPriority : u32 { Low, Normal, High, VeryHigh, Critical };

/// Names the calling thread (as top and debuggers show it; 15 characters at most).
void SetCurrentThreadName(const char* name);
/// Names a thread by its pthread handle.
void SetThreadName(void* thread, const char* name);
[[nodiscard]] std::string GetCurrentThreadName();

void SetCurrentThreadPriority(ThreadPriority priority);
/// Real-time scheduling for periodic threads: not used on Linux (no-op).
void SetCurrentThreadRealtime(std::chrono::nanoseconds period);

/// Sleeps `duration`; with `interruptible` a signal ends it early. `remaining` gets what is left.
/// Sleeps until `deadline` (std::this_thread::sleep_until spun on Windows below 1 ms: MinGW's
/// nanosleep is Sleep(whole ms), Sleep(0) under one).
void SleepUntil(std::chrono::steady_clock::time_point deadline);

bool AccurateSleep(std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   bool interruptible);

/// Paces a loop to a fixed period: Start() at the top waits for the period's deadline, End() at
/// the bottom moves it on by one period. Deadlines are absolute, so wake-up latency does not
/// accumulate; a loop late by more than a period starts over instead of catching up.
class AccurateTimer {
public:
    explicit AccurateTimer(std::chrono::nanoseconds period);
    void Start();
    void End();
    [[nodiscard]] std::chrono::nanoseconds GetTotalWait() const {
        return total_wait;
    }

private:
    std::chrono::nanoseconds period;
    std::chrono::nanoseconds total_wait{};
    std::chrono::steady_clock::time_point deadline{}; ///< when the current period ends
};

/// Waits on `cv` until `pred()` or `deadline`, as cv.wait_until(lock, deadline, pred). On Windows
/// MinGW's libc++ waits in whole milliseconds, truncated, and retries: the last one of every timed
/// wait span a core (a 1 kHz vblank loop: 84 % of one). There the wait leaves 2 ms to short sleeps
/// (250 us, a high-resolution timer) between checks of `pred`, unlocked.
template <class CV, class Lock, class Pred>
bool WaitUntil(CV& cv, Lock& lock, std::chrono::steady_clock::time_point deadline, Pred pred) {
#ifdef _WIN32
    using namespace std::chrono;
    while (!pred()) {
        const auto left = deadline - steady_clock::now();
        if (left <= nanoseconds{0}) {
            return pred();
        }
        if (left > milliseconds{2}) {
            cv.wait_for(lock, floor<milliseconds>(left) - milliseconds{1});
        } else {
            lock.unlock();
            AccurateSleep(std::min<nanoseconds>(left, microseconds{250}), nullptr, false);
            lock.lock();
        }
    }
    return true;
#else
    return cv.wait_until(lock, deadline, pred);
#endif
}

template <class CV, class Lock, class Rep, class Period, class Pred>
bool WaitFor(CV& cv, Lock& lock, std::chrono::duration<Rep, Period> timeout, Pred pred) {
    return WaitUntil(cv, lock, std::chrono::steady_clock::now() +
                                   std::chrono::duration_cast<std::chrono::nanoseconds>(timeout),
                     pred);
}

} // namespace Common
