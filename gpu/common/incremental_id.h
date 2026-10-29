// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: thread-safe increasing ids, starting at 1.
#pragma once

#include <atomic>

namespace Common {

template <typename Counter>
class IncrementalIdProvider {
public:
    Counter Next() {
        return next.fetch_add(1, std::memory_order_relaxed);
    }

private:
    std::atomic<Counter> next{1};
};

} // namespace Common
