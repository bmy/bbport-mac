// SPDX-FileCopyrightText: Copyright 2026 CLS_AMG
// SPDX-License-Identifier: GPL-2.0-or-later

#include "core/bbsc_memscan.h"

#ifdef _WIN32
#include <windows.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#ifndef _WIN32
#include <sys/uio.h>
#include <unistd.h>
#include "core/bbport_bbsc_kit.h" // Teefpyre (SrShadowy fork)
#endif

namespace Core::Bbsc {

namespace {

#ifdef _WIN32
// One region. The emulator changes page protection all the time (GPU tracking), so a page that was
// readable a moment ago may fault now. bbport: clang/mingw has no __try, so the region is read in
// chunks with ReadProcessMemory (a chunk that no longer reads ends the region, no fault); chunks
// overlap by length - 1 bytes so a match across a boundary is found once.
void SearchRegion(const std::uint8_t* start, std::size_t size, const std::uint8_t* needle,
                  std::size_t length, std::vector<std::uint64_t>& hits, std::size_t max_hits) {
    constexpr std::size_t Chunk = 1u << 20;
    if (size < length) {
        return;
    }
    std::vector<std::uint8_t> buffer(std::min(size, Chunk + length - 1));
    std::size_t offset = 0;
    while (hits.size() < max_hits && size - offset >= length) {
        const std::size_t want = std::min(buffer.size(), size - offset);
        SIZE_T read = 0;
        if (!ReadProcessMemory(GetCurrentProcess(), start + offset, buffer.data(), want, &read) ||
            read != want) {
            return;
        }
        const std::uint8_t* end = buffer.data() + want;
        const std::uint8_t* at = buffer.data();
        while (hits.size() < max_hits && end - at >= static_cast<std::ptrdiff_t>(length)) {
            at = static_cast<const std::uint8_t*>(
                std::memchr(at, needle[0], static_cast<std::size_t>(end - at) - length + 1));
            if (at == nullptr) {
                break;
            }
            if (std::memcmp(at, needle, length) == 0) {
                hits.push_back(reinterpret_cast<std::uint64_t>(start + offset + (at - buffer.data())));
            }
            ++at;
        }
        if (want < buffer.size()) {
            break;
        }
        offset += want - (length - 1);
    }
}
#endif

} // namespace

std::vector<std::uint64_t> FindInMemory(std::span<const std::uint8_t> needle, std::uint64_t lo,
                                        std::uint64_t hi, std::size_t max_hits) {
    std::vector<std::uint64_t> hits;
#ifdef _WIN32
    if (needle.empty()) {
        return hits;
    }
    constexpr DWORD Readable = PAGE_READONLY | PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READ |
                               PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    std::uint64_t at = lo;
    while (at < hi && hits.size() < max_hits) {
        MEMORY_BASIC_INFORMATION info{};
        if (VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) == 0) {
            break;
        }
        const auto start = reinterpret_cast<std::uint64_t>(info.BaseAddress);
        const std::uint64_t end = start + info.RegionSize;
        if (info.State == MEM_COMMIT && (info.Protect & Readable) != 0 &&
            (info.Protect & (PAGE_GUARD | PAGE_NOACCESS)) == 0) {
            SearchRegion(reinterpret_cast<const std::uint8_t*>(start), info.RegionSize,
                         needle.data(), needle.size(), hits, max_hits);
        }
        if (end <= at) {
            break;
        }
        at = end;
    }
#else
    extra_bbsc_find_in_memory(needle.data(), needle.size(), lo, hi, max_hits, &hits); // [extra: Teefpyre]
    return hits; // [extra: Teefpyre]
#endif
    return hits;
}

} // namespace Core::Bbsc
