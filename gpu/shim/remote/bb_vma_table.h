// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the guest's mapping table inside the native GPU process (docs/macos-native-gpu.md):
// the game process reports every change (runtime_memory.c, its mirror hook) and this answers the
// GPU library's queries as runtime_memory.c does there (runtime_memory_region, _clamp, _vma_info,
// _direct_phys, _is_mapped). Header-only; tests/test_remote_table.cpp compares the two.
#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include <vector>

#include "remote/bb_protocol.h"

namespace BbRemote {

struct VmaEntry {
    std::uint64_t end;
    std::uint32_t kind; ///< MapDirect, MapFlexible
    std::int32_t prot, type;
    std::uint64_t phys;
};

/// What runtime_memory.c answers in the game process, for what is mirrored here.
class VmaTable {
public:
    void Map(std::uint64_t start, std::uint64_t end, const VmaEntry& entry) {
        Writer writer{*this};
        Carve(start, end);
        map.emplace(start, entry);
    }
    void Unmap(std::uint64_t start, std::uint64_t end) {
        Writer writer{*this};
        Carve(start, end);
    }
    void Protect(std::uint64_t start, std::uint64_t end, std::int32_t prot, std::int32_t type) {
        Writer writer{*this};
        Split(start);
        Split(end);
        for (auto it = map.lower_bound(start); it != map.end() && it->first < end; ++it) {
            it->second.prot = prot;
            if (type >= 0) {
                it->second.type = type;
            }
        }
    }
    /// The mapped pieces of [start, end).
    std::vector<std::pair<std::uint64_t, std::uint64_t>> Pieces(std::uint64_t start,
                                                                std::uint64_t end) const {
        std::shared_lock lock{mutex};
        std::vector<std::pair<std::uint64_t, std::uint64_t>> pieces;
        auto it = map.upper_bound(start);
        if (it != map.begin()) {
            --it;
        }
        for (; it != map.end() && it->first < end; ++it) {
            const std::uint64_t a = std::max(start, it->first), b = std::min(end, it->second.end);
            if (a < b) {
                pieces.emplace_back(a, b - a);
            }
        }
        return pieces;
    }
    std::vector<std::pair<std::uint64_t, std::uint64_t>> All() const {
        std::shared_lock lock{mutex};
        std::vector<std::pair<std::uint64_t, std::uint64_t>> all;
        for (const auto& [start, entry] : map) {
            all.emplace_back(start, entry.end - start);
        }
        return all;
    }

    int Region(std::uintptr_t address, std::uintptr_t* start, std::uintptr_t* end,
               int* mapped) const {
        std::shared_lock lock{mutex};
        auto next = map.upper_bound(address);
        if (next != map.begin()) {
            const auto prev = std::prev(next);
            if (address < prev->second.end) {
                *start = prev->first;
                *end = prev->second.end;
                *mapped = 1;
                return 1;
            }
        }
        if (next == map.end()) {
            return 0; // past the last mapping
        }
        *start = next == map.begin() ? 0 : std::prev(next)->second.end;
        *end = next->first;
        *mapped = 0;
        return 1;
    }
    std::uint64_t Clamp(std::uintptr_t address, std::uint64_t size) const {
        std::shared_lock lock{mutex};
        auto it = Containing(address);
        std::uint64_t length = 0;
        for (; it != map.end() && length < size; ++it) {
            if (it->first > address + length) {
                break;
            }
            length = it->second.end - address;
        }
        return std::min(length, size);
    }
    int VmaInfo(std::uintptr_t address, int* prot, int* type, std::uintptr_t* end) const {
        std::shared_lock lock{mutex};
        const auto it = Containing(address);
        if (it == map.end()) {
            return 0;
        }
        *prot = it->second.prot;
        *type = it->second.kind == MapDirect ? it->second.type : -1;
        *end = it->second.end;
        return 1;
    }
    int DirectPhys(std::uintptr_t address, std::uint64_t* phys, std::uintptr_t* end) const {
        std::shared_lock lock{mutex};
        const auto it = Containing(address);
        if (it == map.end() || it->second.kind != MapDirect) {
            return 0;
        }
        *phys = it->second.phys + (address - it->first);
        *end = it->second.end;
        return 1;
    }
    int Covered(std::uintptr_t address, std::uint64_t size) const {
        std::shared_lock lock{mutex};
        std::uint64_t at = address;
        for (auto it = Containing(address); at < address + size; ++it) {
            if (it == map.end() || it->first > at) {
                return 0;
            }
            at = it->second.end;
        }
        return 1;
    }
    const std::uint64_t* Generation() const {
        return &generation;
    }

private:
    /// Odd while changing: the GPU library's region caches hold for one even value.
    struct Writer {
        explicit Writer(VmaTable& table_) : table{table_}, lock{table_.mutex} {
            __atomic_add_fetch(&table.generation, 1, __ATOMIC_ACQ_REL);
        }
        ~Writer() {
            __atomic_add_fetch(&table.generation, 1, __ATOMIC_RELEASE);
        }
        VmaTable& table;
        std::unique_lock<std::shared_mutex> lock;
    };

    std::map<std::uint64_t, VmaEntry>::const_iterator Containing(std::uint64_t address) const {
        auto it = map.upper_bound(address);
        if (it == map.begin()) {
            return map.end();
        }
        --it;
        return address < it->second.end ? it : map.end();
    }
    void Split(std::uint64_t at) {
        auto it = map.upper_bound(at);
        if (it == map.begin()) {
            return;
        }
        --it;
        if (it->first < at && at < it->second.end) {
            VmaEntry right = it->second;
            right.phys += at - it->first;
            it->second.end = at;
            map.emplace(at, right);
        }
    }
    void Carve(std::uint64_t start, std::uint64_t end) {
        Split(start);
        Split(end);
        map.erase(map.lower_bound(start), map.lower_bound(end));
    }

    mutable std::shared_mutex mutex;
    std::map<std::uint64_t, VmaEntry> map;
    alignas(8) std::uint64_t generation = 0;
};

} // namespace BbRemote
