// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a shared hold of a mutex that the same thread may already hold shared. With
// SharedFirstMutex that is safe as it is (readers never wait for a waiting writer), so this is a
// plain shared lock, with no per-thread bookkeeping.
#pragma once

#include <shared_mutex>

namespace Common {

template <typename MutexType>
class RecursiveSharedLock {
public:
    explicit RecursiveSharedLock(MutexType& mutex_) : mutex{mutex_} {
        mutex.lock_shared();
    }
    ~RecursiveSharedLock() {
        mutex.unlock_shared();
    }
    RecursiveSharedLock(const RecursiveSharedLock&) = delete;
    RecursiveSharedLock& operator=(const RecursiveSharedLock&) = delete;

private:
    MutexType& mutex;
};

} // namespace Common
