// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport buffers: a set of address ranges (half-open, touching ones joined), by address and size.
#pragma once

#include <algorithm>

#include "common/interval_set.h"
#include "common/types.h"

namespace VideoCore {

class RangeSet {
public:
    void Add(VAddr address, u64 size) {
        ranges.Add({address, address + size});
    }
    void Subtract(VAddr address, u64 size) {
        ranges.Subtract(address, address + size);
    }
    void Clear() {
        ranges.Clear();
    }
    /// All of [address, address + size) is in the set (an empty range always is).
    bool Contains(VAddr address, u64 size) const {
        return size == 0 || ranges.Contains(address, address + size);
    }
    bool Intersects(VAddr address, u64 size) const {
        return ranges.Overlaps(address, address + size);
    }
    /// func(start, end) for each range.
    void ForEach(auto&& func) const {
        for (const Interval& range : ranges) {
            func(range.start, range.end);
        }
    }
    /// func(start, end) for each part of [address, address + size) in the set.
    void ForEachInRange(VAddr address, u64 size, auto&& func) const {
        const VAddr end = address + size;
        ranges.ForEachInRange(address, end, [&](const Interval& range) {
            func(std::max<VAddr>(range.start, address), std::min<VAddr>(range.end, end));
        });
    }
    /// func(start, size) for each part of [address, address + size) not in the set.
    void ForEachNotInRange(VAddr address, u64 size, auto&& func) const {
        ranges.ForEachGap(address, address + size,
                          [&](u64 start, u64 end) { func(start, end - start); });
    }

private:
    IntervalList<> ranges;
};

} // namespace VideoCore
