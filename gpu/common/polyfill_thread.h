// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: waits that end when a std::stop_token is triggered.
#pragma once

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stop_token>
#include <thread>
#include <utility>

namespace Common {

/// cv.wait(lk, pred), ending early once `token` asks to stop (cv: condition_variable_any).
template <typename Condvar, typename Lock, typename Pred>
void CondvarWait(Condvar& cv, std::unique_lock<Lock>& lk, std::stop_token token, Pred&& pred) {
    cv.wait(lk, token, std::forward<Pred>(pred));
}

/// Sleeps `duration` unless stopped first; true if the whole time passed.
template <typename Rep, typename Period>
bool StoppableTimedWait(std::stop_token token, const std::chrono::duration<Rep, Period>& duration) {
    std::mutex mutex;
    std::condition_variable_any cv;
    std::unique_lock lock{mutex};
    return !cv.wait_for(lock, token, duration, [&] { return token.stop_requested(); });
}

} // namespace Common
