// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: which images lie over each 1 MiB page of the guest's address space, as a
// two-level table (1024 x 1024 pages over 40 bits; second-level blocks made on first use).
#pragma once

#include <array>
#include <memory>

#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "textures/types.h"

namespace VideoCore {

class ImageRegistry {
public:
    static constexpr u32 PageBits = 20;
    using Ids = boost::container::small_vector<ImageId, 16>;

    /// The images over `page`, or null when no image ever was.
    Ids* Find(u64 page) {
        Block* block = blocks[(page >> BlockBits) & (NumBlocks - 1)].get();
        return block ? &(*block)[page & (PagesPerBlock - 1)] : nullptr;
    }
    /// The images over `page` (its block made if needed).
    Ids& At(u64 page) {
        auto& block = blocks[(page >> BlockBits) & (NumBlocks - 1)];
        if (!block) {
            block = std::make_unique<Block>();
        }
        return (*block)[page & (PagesPerBlock - 1)];
    }

    /// Calls `func(ids)` for the pages of [address, address + size) that hold images, until it
    /// returns true.
    template <typename Func>
    void ForEachPage(VAddr address, u64 size, Func&& func) {
        const u64 last = (address + size - 1) >> PageBits;
        for (u64 page = address >> PageBits; page <= last; ++page) {
            if (Ids* ids = Find(page); ids && !ids->empty() && func(*ids)) {
                return;
            }
        }
    }
    /// Calls `func(page)` for every page of [address, address + size).
    template <typename Func>
    static void ForEachPageIndex(VAddr address, u64 size, Func&& func) {
        const u64 last = (address + size - 1) >> PageBits;
        for (u64 page = address >> PageBits; page <= last; ++page) {
            func(page);
        }
    }

private:
    static constexpr u32 BlockBits = 10;
    static constexpr u64 PagesPerBlock = 1ull << BlockBits;
    static constexpr u64 NumBlocks = 1ull << (40 - PageBits - BlockBits);
    using Block = std::array<Ids, PagesPerBlock>;

    std::array<std::unique_ptr<Block>, NumBlocks> blocks{};
};

} // namespace VideoCore
