// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: holds a run of locks from an array. Every holder takes its run in ascending order, so
// two overlapping runs cannot deadlock (no thread waits for a lock below one it holds).
#pragma once

namespace Common {

template <typename Iterator>
class RangeLockGuard {
public:
    RangeLockGuard(Iterator begin_, Iterator end_) : begin{begin_}, end{end_} {
        for (Iterator it = begin; it != end; ++it) {
            it->lock();
        }
    }
    ~RangeLockGuard() {
        for (Iterator it = begin; it != end; ++it) {
            it->unlock();
        }
    }
    RangeLockGuard(const RangeLockGuard&) = delete;
    RangeLockGuard& operator=(const RangeLockGuard&) = delete;

private:
    Iterator begin;
    Iterator end;
};

} // namespace Common
