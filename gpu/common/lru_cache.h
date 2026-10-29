// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: objects ordered by the tick they were last used at (oldest first), for the texture
// cache's garbage collection. Items live in one array and link to each other by index; freed
// slots are reused.
#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include "common/types.h"

namespace Common {

template <typename ObjectType, typename TickType>
class LeastRecentlyUsedCache {
    static constexpr u32 None = ~0u;

    struct Item {
        ObjectType obj{};
        TickType tick{};
        u32 prev = None;
        u32 next = None;
    };

public:
    /// Adds `obj` as the most recently used; returns its id.
    std::size_t Insert(ObjectType obj, TickType tick) {
        u32 id;
        if (free_ids.empty()) {
            id = u32(items.size());
            items.emplace_back();
        } else {
            id = free_ids.back();
            free_ids.pop_back();
        }
        Item& item = items[id];
        item.obj = obj;
        item.tick = tick;
        Append(id);
        return id;
    }

    /// Marks `id` used at `tick` (an older tick changes nothing).
    void Touch(std::size_t id, TickType tick) {
        Item& item = items[id];
        if (item.tick >= tick) {
            return;
        }
        item.tick = tick;
        if (u32(id) != last) {
            Unlink(u32(id));
            Append(u32(id));
        }
    }

    void Free(std::size_t id) {
        Unlink(u32(id));
        free_ids.push_back(u32(id));
    }

    /// Calls `func(obj)` for the items last used at `tick` or before, oldest first. `func` may
    /// free the item it is given; one that returns bool ends the walk by returning true.
    template <typename Func>
    void ForEachItemBelow(TickType tick, Func&& func) {
        for (u32 id = first; id != None;) {
            const Item& item = items[id];
            if (static_cast<s64>(tick) - static_cast<s64>(item.tick) < 0) {
                return;
            }
            const u32 next = item.next;
            if constexpr (std::is_same_v<std::invoke_result_t<Func, ObjectType>, bool>) {
                if (func(item.obj)) {
                    return;
                }
            } else {
                func(item.obj);
            }
            id = next;
        }
    }

private:
    void Append(u32 id) {
        Item& item = items[id];
        item.prev = last;
        item.next = None;
        if (last != None) {
            items[last].next = id;
        } else {
            first = id;
        }
        last = id;
    }

    void Unlink(u32 id) {
        Item& item = items[id];
        (item.prev != None ? items[item.prev].next : first) = item.next;
        (item.next != None ? items[item.next].prev : last) = item.prev;
        item.prev = item.next = None;
    }

    std::vector<Item> items;
    std::vector<u32> free_ids;
    u32 first = None;
    u32 last = None;
};

} // namespace Common
