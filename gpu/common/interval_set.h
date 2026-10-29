// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a set of disjoint [start, end) ranges kept sorted in one array. An interval type may
// carry data (Backing: memory and offset); it says which neighbours may merge into one interval
// (CanMergeWith) and how to cut a piece out of itself (SubRange). Adding a range over existing
// ones replaces what it covers; touching ranges that may merge are merged.
#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <vector>

#include "common/types.h"

struct Interval {
    u64 start;
    u64 end;

    constexpr bool CanMergeWith(const Interval&) const noexcept {
        return true;
    }
    constexpr Interval SubRange(u64 a, u64 b) const noexcept {
        return {a, b};
    }
};

template <class IV>
concept IsInterval = std::derived_from<IV, Interval> && std::is_trivially_copyable_v<IV> &&
                     requires(const IV& a, const IV& b, u64 x) {
                         { a.CanMergeWith(b) } -> std::same_as<bool>;
                         { a.SubRange(x, x) } -> std::same_as<IV>;
                     };

template <class IV = Interval>
    requires IsInterval<IV>
class IntervalList {
public:
    using List = std::vector<IV>;
    using const_iterator = typename List::const_iterator;
    using iterator = typename List::iterator;

    const_iterator begin() const {
        return intervals.begin();
    }
    const_iterator end() const {
        return intervals.end();
    }
    void Clear() {
        intervals.clear();
    }
    bool Empty() const {
        return intervals.empty();
    }
    std::size_t Size() const {
        return intervals.size();
    }
    void Reserve(std::size_t n) {
        intervals.reserve(n);
    }

    /// Puts `value` in, over whatever it covers.
    void Add(IV value) {
        if (value.start >= value.end) [[unlikely]] {
            return;
        }
        // Intervals that overlap it, and the one right after it if it touches.
        const auto first = FirstEndingAfter(value.start);
        const auto last = std::upper_bound(first, intervals.end(), value.end,
                                           [](u64 at, const IV& iv) { return at < iv.start; });
        IV parts[3];
        std::size_t count = 0;
        if (first != last && first->start < value.start) {
            parts[count++] = first->SubRange(first->start, value.start);
        }
        parts[count++] = value;
        if (first != last) {
            const IV& tail = *std::prev(last);
            if (tail.end > value.end) {
                parts[count++] = tail.SubRange(value.end, tail.end);
            }
        }
        const std::size_t at = std::size_t(first - intervals.begin());
        Replace(first, last, parts, count);
        MergeAround(at, count);
    }

    /// Takes [start, end) out (the parts of intervals outside it stay).
    void Subtract(u64 start, u64 end) {
        if (start >= end) [[unlikely]] {
            return;
        }
        const auto first = FirstEndingAfter(start);
        const auto last = std::lower_bound(first, intervals.end(), end,
                                           [](const IV& iv, u64 at) { return iv.start < at; });
        if (first == last) {
            return;
        }
        IV parts[2];
        std::size_t count = 0;
        if (first->start < start) {
            parts[count++] = first->SubRange(first->start, start);
        }
        const IV& tail = *std::prev(last);
        if (tail.end > end) {
            parts[count++] = tail.SubRange(end, tail.end);
        }
        Replace(first, last, parts, count);
    }

    /// The interval holding `addr`, or end().
    const_iterator Find(u64 addr) const {
        const auto it = FirstEndingAfter(addr);
        return it != intervals.end() && it->start <= addr ? it : intervals.end();
    }

    bool Contains(u64 addr) const {
        return Find(addr) != intervals.end();
    }

    /// [start, end) is covered completely (by one interval or touching ones).
    bool Contains(u64 start, u64 end) const {
        if (start >= end) [[unlikely]] {
            return false;
        }
        u64 covered = start;
        for (auto it = FirstEndingAfter(start); it != intervals.end() && it->start <= covered;
             ++it) {
            covered = it->end;
            if (covered >= end) {
                return true;
            }
        }
        return false;
    }

    /// Some of [start, end) is in the set.
    bool Overlaps(u64 start, u64 end) const {
        if (start >= end) [[unlikely]] {
            return false;
        }
        const auto it = FirstEndingAfter(start);
        return it != intervals.end() && it->start < end;
    }

    /// Calls func(gap_start, gap_end) for each part of [start, end) not in the set.
    void ForEachGap(u64 start, u64 end, auto&& func) const {
        if (start >= end) [[unlikely]] {
            return;
        }
        u64 at = start;
        for (auto it = FirstEndingAfter(start); it != intervals.end() && it->start < end; ++it) {
            if (it->start > at) {
                func(at, it->start);
            }
            at = std::max(at, it->end);
        }
        if (at < end) {
            func(at, end);
        }
    }

    /// Calls func(interval) for each interval overlapping [start, end).
    void ForEachInRange(u64 start, u64 end, auto&& func) const {
        if (start >= end) [[unlikely]] {
            return;
        }
        for (auto it = FirstEndingAfter(start); it != intervals.end() && it->start < end; ++it) {
            func(*it);
        }
    }

protected:
    iterator FirstEndingAfter(u64 addr) {
        return std::upper_bound(intervals.begin(), intervals.end(), addr,
                                [](u64 at, const IV& iv) { return at < iv.end; });
    }
    const_iterator FirstEndingAfter(u64 addr) const {
        return std::upper_bound(intervals.begin(), intervals.end(), addr,
                                [](u64 at, const IV& iv) { return at < iv.end; });
    }

    /// [first, last) becomes parts[0, count).
    void Replace(iterator first, iterator last, const IV* parts, std::size_t count) {
        const std::size_t slots = std::size_t(last - first);
        const std::size_t reused = std::min(slots, count);
        std::copy_n(parts, reused, first);
        if (count < slots) {
            intervals.erase(first + count, last);
        } else if (count > slots) {
            intervals.insert(first + slots, parts + slots, parts + count);
        }
    }

    /// Merges touching, mergeable intervals among [at - 1, at + count].
    void MergeAround(std::size_t at, std::size_t count) {
        std::size_t i = at ? at - 1 : 0;
        std::size_t stop = std::min(at + count + 1, intervals.size());
        while (i + 1 < stop) {
            IV& a = intervals[i];
            const IV& b = intervals[i + 1];
            if (a.end == b.start && a.CanMergeWith(b)) {
                a.end = b.end;
                intervals.erase(intervals.begin() + std::ptrdiff_t(i + 1));
                --stop;
            } else {
                ++i;
            }
        }
    }

    List intervals;
};
