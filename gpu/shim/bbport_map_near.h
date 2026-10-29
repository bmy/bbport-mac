// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: memory within rel32 reach (+-2 GiB) of the game's code, for hook trampolines and the
// stubs and data the port's patches jump to and address RIP-relative. One search for all of them
// (guest code hooks, frame rate, game menu, Gnm census).
//
// Hints below and above [first, last) in `step` strides, aligned to 64 KiB (Windows places
// allocations at its allocation granularity: an address between two was rounded and the mapping
// refused as not where asked, so the search above the image never succeeded there), then the
// runtime's low region allocator (runtime_low_map, above the image: where the runtime's own
// mappings land). On a Windows PC with the space under the image taken (RTX 3080, 2026-10-27) every
// hook failed: no frame rate patch (30 FPS), no port pages in the game's menu, no aspect ratio.
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <sys/mman.h>

extern "C" void* runtime_low_map(size_t size, int prot);

namespace BbMapNear {

inline constexpr std::int64_t Reach = 0x7000'0000; // inside a rel32 jump's 2 GiB, with room

inline bool InReach(std::uintptr_t p, std::size_t size, std::uintptr_t first, std::uintptr_t last) {
    const auto lo = std::int64_t(p), hi = std::int64_t(p + size);
    return hi - std::int64_t(first) < Reach && std::int64_t(last) - lo < Reach;
}

/// `size` bytes with `prot` within reach of [first, last), or nullptr. `what` names the user in
/// the log line of a failure.
inline void* Map(std::uintptr_t first, std::uintptr_t last, std::size_t size, int prot,
                 std::uint64_t step, const char* what) {
    constexpr std::uint64_t Granule = 64 << 10;
    int first_error = 0;
    for (std::uint64_t k = 1; k * step < std::uint64_t(Reach); ++k) {
        const std::uint64_t below = first > k * step ? (first - k * step) & ~(Granule - 1) : 0;
        const std::uint64_t above = (last + k * step + Granule - 1) & ~(Granule - 1);
        for (const std::uint64_t hint : {below, above}) {
            if (!hint || !InReach(std::uintptr_t(hint), size, first, last)) {
                continue;
            }
            void* p = mmap(reinterpret_cast<void*>(hint), size, prot,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
            if (p != MAP_FAILED) {
                return p;
            }
            if (!first_error) {
                first_error = errno;
            }
        }
    }
    if (void* p = runtime_low_map(size, prot);
        p && InReach(reinterpret_cast<std::uintptr_t>(p), size, first, last)) {
        std::printf("%s: memory near the game's code from the runtime's low region at %p (the "
                    "space around the image was taken: %s)\n",
                    what, p, std::strerror(first_error));
        return p;
    }
    std::printf("%s: no memory within 2 GiB of the game's code (%s)\n", what,
                std::strerror(first_error));
    return nullptr;
}

} // namespace BbMapNear
