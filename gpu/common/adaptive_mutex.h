// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: glibc's adaptive mutex: spins a little before sleeping, for locks held for a moment.
#pragma once

#include <pthread.h>

namespace Common {

#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
class AdaptiveMutex {
public:
    AdaptiveMutex() = default;
    ~AdaptiveMutex() {
        pthread_mutex_destroy(&mutex);
    }
    AdaptiveMutex(const AdaptiveMutex&) = delete;
    AdaptiveMutex& operator=(const AdaptiveMutex&) = delete;

    void lock() {
        pthread_mutex_lock(&mutex);
    }
    void unlock() {
        pthread_mutex_unlock(&mutex);
    }
    [[nodiscard]] bool try_lock() {
        return pthread_mutex_trylock(&mutex) == 0;
    }

private:
    pthread_mutex_t mutex = PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP;
};
#endif

} // namespace Common
