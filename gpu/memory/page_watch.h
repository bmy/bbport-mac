// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport memory: which guest pages the GPU side watches, and the protections that make the
// guest's accesses to them known. A page is watched for writes by each image over it and by the
// buffer tracker while its CPU copy is clean; for reads while only the GPU has its newest data
// (readbacks). Faults on watched pages go to the rasterizer.
//
// A watched page gets the runtime's write watch (the writers that check it report; with
// BB_WRITE_VERIFY=1 page protection too). Images also set write traps (BB_IMAGE_TRAPS), so that
// writes nobody announced still reach the texture cache.
#pragma once

#include <memory>

#include "common/types.h"
#include "memory/page_bits.h"

namespace Vulkan {
class Rasterizer;
}

namespace BbMem {

inline constexpr u32 PageShift = 12;
inline constexpr u64 PageSize = 1ull << PageShift;
/// Pages are tracked in regions of 4 MiB, each with its lock and its bits.
inline constexpr u32 RegionShift = 22;
inline constexpr u64 RegionSize = 1ull << RegionShift;
inline constexpr size_t PagesPerRegion = RegionSize / PageSize;
/// The guest's address space.
inline constexpr u32 AddressBits = 40;
inline constexpr size_t NumPages = 1ull << (AddressBits - PageShift);
inline constexpr size_t NumRegions = 1ull << (AddressBits - RegionShift);

using RegionPages = PageBits<PagesPerRegion>;

constexpr VAddr PageStart(VAddr address) {
    return address & ~(PageSize - 1);
}
/// The start of the page after the one holding `address`.
constexpr VAddr NextPage(VAddr address) {
    return PageStart(address) + PageSize;
}

class PageWatch {
public:
    explicit PageWatch(Vulkan::Rasterizer* rasterizer);
    ~PageWatch();
    PageWatch(const PageWatch&) = delete;
    PageWatch& operator=(const PageWatch&) = delete;

    /// An image over [address, address + size) is watched for writes (`watch`), or no longer.
    template <bool watch>
    void WatchImage(VAddr address, u64 size);
    /// The pages of `mask` in the region at `region`: one watcher more (`watch`) or less, for
    /// writes or (`reads`) for reads.
    template <bool watch, bool reads>
    void WatchRegion(VAddr region, const RegionPages& mask);

    struct Impl;

private:
    std::unique_ptr<Impl> impl;
};

} // namespace BbMem
