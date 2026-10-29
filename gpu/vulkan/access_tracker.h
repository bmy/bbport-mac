// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport vulkan: buffer ranges read and written by commands recorded since the last barrier.
// A transfer touching a range written before (or writing one read before) needs a barrier first.
// Per buffer, per kind: sorted disjoint ranges, merged on insert. Draws read constants from
// rings at a new offset each time; the ranges join into one that grows, so lookups stay a short
// binary search however many draws went by.
#pragma once

#include <algorithm>
#include <vector>

#include "common/types.h"

namespace Vulkan {

class AccessTracker {
public:
    enum class Kind : u32 { Read, Write };

    /// Whether [first, last] of `resource` overlaps a recorded range of `kind`.
    bool Overlaps(u64 resource, u64 first, u64 last, Kind kind) const {
        const Resource* entry = Find(resource);
        return entry && Overlaps(entry->ranges[u32(kind)], first, last);
    }

    /// Records [first, last] of `resource` as accessed for `kind`.
    void Insert(u64 resource, u64 first, u64 last, Kind kind) {
        Insert(Get(resource).ranges[u32(kind)], first, last);
        empty = false;
    }

    /// Forgets everything (a barrier was recorded); keeps the memory.
    void Clear() {
        if (empty) {
            return;
        }
        for (u32 i = 0; i < used; ++i) {
            resources[i].ranges[0].clear();
            resources[i].ranges[1].clear();
        }
        used = 0;
        last = 0;
        empty = true;
    }

    bool Empty() const {
        return empty;
    }

private:
    struct Range {
        u64 first;
        u64 last; ///< inclusive
    };
    using Ranges = std::vector<Range>;
    struct Resource {
        u64 id;
        Ranges ranges[2];
    };

    static bool Overlaps(const Ranges& ranges, u64 first, u64 last) {
        // The first range ending at or after `first`: the only one that can overlap first.
        const auto it = std::ranges::lower_bound(ranges, first, {}, &Range::last);
        return it != ranges.end() && it->first <= last;
    }

    static void Insert(Ranges& ranges, u64 first, u64 last) {
        // Ranges touching [first, last] (overlapping or adjacent) merge into it.
        const u64 reach = first == 0 ? 0 : first - 1;
        auto begin = std::ranges::lower_bound(ranges, reach, {}, &Range::last);
        if (begin != ranges.end() && begin->first <= first && begin->last >= last) {
            return; // already covered: the common case for draws re-binding a range
        }
        const u64 after = last == ~u64{0} ? last : last + 1;
        auto end = begin;
        while (end != ranges.end() && end->first <= after) {
            first = std::min(first, end->first);
            last = std::max(last, end->last);
            ++end;
        }
        if (begin == end) {
            ranges.insert(begin, Range{first, last});
            return;
        }
        *begin = Range{first, last};
        ranges.erase(begin + 1, end);
    }

    const Resource* Find(u64 id) const {
        if (used && resources[last].id == id) {
            return &resources[last];
        }
        for (u32 i = 0; i < used; ++i) {
            if (resources[i].id == id) {
                last = i;
                return &resources[i];
            }
        }
        return nullptr;
    }

    Resource& Get(u64 id) {
        if (const Resource* found = Find(id)) {
            return const_cast<Resource&>(*found);
        }
        if (used == resources.size()) {
            resources.emplace_back();
        }
        last = used++;
        resources[last].id = id;
        return resources[last];
    }

    std::vector<Resource> resources; ///< the first `used` hold this period's buffers
    u32 used = 0;
    mutable u32 last = 0; ///< the entry found last
    bool empty = true;
};

} // namespace Vulkan
