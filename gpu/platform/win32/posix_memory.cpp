// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: mmap, munmap, mprotect and fallocate (sys/mman.h, bbport_posix_c.h) with the
// Linux semantics the runtime's guest memory relies on (src/runtime_memory.c).
//
// - A section fd (bb_section_create) plays the memfd's part: a pagefile-backed section; mmap of
//   it maps a view, fallocate(PUNCH_HOLE) zeroes a range of it.
// - The guest's address range is one placeholder reservation (bb_reserve_arena). A MAP_FIXED
//   mapping inside it replaces exactly its range, as on Linux: whatever was there (views,
//   reservations) is cut out first, placeholders are coalesced and split, and the parts of views
//   outside the range are mapped again with the page protections they had. An anonymous
//   PROT_NONE mapping there (a reservation) stays a placeholder; munmap leaves one.
// - Outside the arena: VirtualAlloc, or a view anywhere.
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <sys/mman.h>

namespace {

DWORD PageProtection(int prot) {
    if (prot & PROT_EXEC) {
        return (prot & PROT_WRITE) ? PAGE_EXECUTE_READWRITE
               : (prot & PROT_READ) ? PAGE_EXECUTE_READ
                                    : PAGE_EXECUTE;
    }
    if (prot & PROT_WRITE) return PAGE_READWRITE;
    if (prot & PROT_READ) return PAGE_READONLY;
    return PAGE_NOACCESS;
}

[[noreturn]] void Fail(const char* what, uintptr_t address, uint64_t size) {
    std::fprintf(stderr, "STOP: %s failed at %#llx (+%#llx): Windows error %lu\n", what,
                 (unsigned long long)address, (unsigned long long)size, GetLastError());
    std::fflush(stderr);
    std::exit(21);
}

// Sections (fds from bb_section_create).
constexpr int SectionFdBase = 0x50000000;
struct Section {
    HANDLE handle = nullptr;
    uint64_t size = 0;
    unsigned char* whole = nullptr; ///< a view of all of it (zeroing)
};
std::unordered_map<int, Section> sections;
int next_section = SectionFdBase;

// The arena: [arena_begin, arena_end) as pieces, each a placeholder or a view.
struct Piece {
    uintptr_t end;
    bool view;
    int fd;          ///< the view's section
    uint64_t offset; ///< the view's offset in it
    bool owned = false; ///< private memory of its own (bb_private_at): no view, no section
};
std::map<uintptr_t, Piece> pieces; ///< by start
std::atomic<unsigned long long> bb_win32_virtual_protects{0};
uintptr_t arena_begin = 0, arena_end = 0;
std::recursive_mutex memory_mutex;

bool InArena(uintptr_t a, uintptr_t b) {
    return arena_begin && a >= arena_begin && b <= arena_end;
}

void MapView(uintptr_t start, uintptr_t end, int fd, uint64_t offset, DWORD protect) {
    const Section& section = sections.at(fd);
    // A view cannot start without access; protections change after.
    const DWORD initial = protect == PAGE_NOACCESS ? PAGE_READWRITE : protect;
    if (!MapViewOfFile3(section.handle, GetCurrentProcess(), reinterpret_cast<void*>(start), offset,
                        end - start, MEM_REPLACE_PLACEHOLDER, initial, nullptr, 0)) {
        std::fprintf(stderr, "Memory: view of section %d offset %#llx (of %#llx) at %#llx failed\n", fd,
                     (unsigned long long)offset, (unsigned long long)section.size,
                     (unsigned long long)start);
        Fail("MapViewOfFile3", start, end - start);
    }
    DWORD old = 0;
    if (initial != protect &&
        !VirtualProtect(reinterpret_cast<void*>(start), end - start, protect, &old)) {
        Fail("VirtualProtect", start, end - start);
    }
}

struct ProtectionRun {
    uintptr_t start, end;
    DWORD protect;
};

void SaveProtections(uintptr_t start, uintptr_t end, std::vector<ProtectionRun>& runs) {
    for (uintptr_t at = start; at < end;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info))) {
            Fail("VirtualQuery", at, 0);
        }
        const uintptr_t next =
            std::min<uintptr_t>(reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize, end);
        runs.push_back({at, next, info.Protect});
        at = next;
    }
}

/// Leaves [a, b) one placeholder piece (what was mapped there is gone; views around it stay).
std::map<uintptr_t, Piece>::iterator Carve(uintptr_t a, uintptr_t b) {
    auto first = pieces.upper_bound(a);
    --first; // the piece holding a
    auto last = pieces.lower_bound(b); // first piece starting at or after b
    // Already exactly one placeholder?
    if (std::next(first) == last && !first->second.view && first->first == a &&
        first->second.end == b) {
        return first;
    }
    const uintptr_t lo = first->first;
    const uintptr_t hi = std::prev(last)->second.end;
    const Piece left = first->second, right = std::prev(last)->second;
    const uintptr_t right_start = std::prev(last)->first;
    // Private memory (bb_private_at) goes whole: the GPU imported it, part of it cannot stay.
    if ((left.owned && lo < a) || (right.owned && b < hi)) {
        Fail("cutting private guest memory", a, b - a);
    }
    const bool keep_left = left.view && lo < a;
    const bool keep_right = right.view && b < hi;
    std::vector<ProtectionRun> runs;
    if (keep_left) SaveProtections(lo, a, runs);
    if (keep_right) SaveProtections(b, hi, runs);
    size_t count = 0;
    for (auto it = first; it != last; ++it, ++count) {
        if (it->second.owned &&
            !VirtualFree(reinterpret_cast<void*>(it->first), 0, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            Fail("releasing private guest memory", it->first, it->second.end - it->first);
        }
        if (it->second.view &&
            !UnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(it->first),
                              MEM_PRESERVE_PLACEHOLDER)) {
            Fail("UnmapViewOfFile2", it->first, it->second.end - it->first);
        }
    }
    if (count > 1 && !VirtualFree(reinterpret_cast<void*>(lo), hi - lo,
                                  MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)) {
        Fail("coalescing placeholders", lo, hi - lo);
    }
    pieces.erase(first, last);
    if (lo < a) {
        if (!VirtualFree(reinterpret_cast<void*>(lo), a - lo, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            Fail("splitting a placeholder", lo, a - lo);
        }
        pieces[lo] = Piece{a, false, -1, 0};
        if (keep_left) {
            pieces[lo] = Piece{a, true, left.fd, left.offset};
            MapView(lo, a, left.fd, left.offset, PAGE_READWRITE);
        }
    }
    if (b < hi) {
        if (!VirtualFree(reinterpret_cast<void*>(a), b - a, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            Fail("splitting a placeholder", a, b - a);
        }
        pieces[b] = Piece{hi, false, -1, 0};
        if (keep_right) {
            const uint64_t offset = right.offset + (b - right_start);
            pieces[b] = Piece{hi, true, right.fd, offset};
            MapView(b, hi, right.fd, offset, PAGE_READWRITE);
        }
    }
    for (const ProtectionRun& run : runs) {
        DWORD old = 0;
        if (run.protect != PAGE_READWRITE &&
            !VirtualProtect(reinterpret_cast<void*>(run.start), run.end - run.start, run.protect, &old)) {
            Fail("restoring page protections", run.start, run.end - run.start);
        }
    }
    return pieces.emplace(a, Piece{b, false, -1, 0}).first;
}

} // namespace

extern "C" {

int bb_section_create(unsigned long long size) {
    std::scoped_lock lock{memory_mutex};
    Section section;
    section.size = size;
    section.handle = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_EXECUTE_READWRITE | SEC_COMMIT,
                                        DWORD(size >> 32), DWORD(size), nullptr);
    if (!section.handle) {
        std::fprintf(stderr,
                     "Runtime: cannot create %llu MiB of guest memory (Windows error %lu): is the "
                     "page file large enough?\n",
                     (unsigned long long)(size >> 20), GetLastError());
        errno = ENOMEM;
        return -1;
    }
    section.whole = static_cast<unsigned char*>(MapViewOfFile3(
        section.handle, GetCurrentProcess(), nullptr, 0, size, 0, PAGE_READWRITE, nullptr, 0));
    if (!section.whole) {
        CloseHandle(section.handle);
        errno = ENOMEM;
        return -1;
    }
    const int fd = next_section++;
    sections.emplace(fd, section);
    return fd;
}

int bb_reserve_arena(unsigned long long begin, unsigned long long end) {
    std::scoped_lock lock{memory_mutex};
    if (arena_begin) return 0;
    if (!VirtualAlloc2(nullptr, reinterpret_cast<void*>(begin), end - begin,
                       MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, nullptr, 0)) {
        std::fprintf(stderr, "STOP: cannot reserve the guest address range %#llx-%#llx (Windows "
                             "error %lu)\n",
                     (unsigned long long)begin, (unsigned long long)end, GetLastError());
        return -1;
    }
    arena_begin = begin;
    arena_end = end;
    pieces.emplace(begin, Piece{end, false, -1, 0});
    return 0;
}

void* bb_private_at(void* address, size_t size) {
    // Committed private memory exactly at [address, address + size) of the arena, replacing what
    // was there: the one view of a direct allocation whose memory the GPU imports (drivers import
    // VirtualAlloc'd memory, not views of sections; BbGuestMemory's private mode).
    std::scoped_lock lock{memory_mutex};
    const auto start = reinterpret_cast<uintptr_t>(address);
    if (!InArena(start, start + size)) {
        errno = EINVAL;
        return MAP_FAILED;
    }
    auto piece = Carve(start, start + size);
    if (!VirtualAlloc2(GetCurrentProcess(), address, size, MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                       PAGE_READWRITE, nullptr, 0)) {
        std::fprintf(stderr, "Memory: private memory at %#llx (+%#llx) failed: Windows error %lu\n",
                     (unsigned long long)start, (unsigned long long)size, GetLastError());
        errno = ENOMEM;
        return MAP_FAILED;
    }
    piece->second = Piece{start + size, false, -1, 0, true};
    return address;
}

void bb_discard(void* address, size_t size) {
    if (!size) return;
    VirtualAlloc(address, size, MEM_RESET, PAGE_READWRITE);
    VirtualUnlock(address, size); // not locked: this only trims the pages from the working set
}

int bb_section_discard(int fd, unsigned long long offset, unsigned long long length) {
    std::scoped_lock lock{memory_mutex};
    const auto it = sections.find(fd);
    if (it == sections.end() || offset + length > it->second.size) {
        errno = EINVAL;
        return -1;
    }
    bb_discard(it->second.whole + offset, size_t(length));
    return 0;
}

int fallocate(int fd, int mode, off_t offset, off_t length) {
    (void)mode; // FALLOC_FL_PUNCH_HOLE | FALLOC_FL_KEEP_SIZE: the range reads as zero again
    std::scoped_lock lock{memory_mutex};
    const auto it = sections.find(fd);
    if (it == sections.end() || uint64_t(offset) + uint64_t(length) > it->second.size) {
        errno = EINVAL;
        return -1;
    }
    std::memset(it->second.whole + offset, 0, size_t(length));
    return 0;
}

void* mmap(void* address, size_t size, int prot, int flags, int fd, off_t offset) {
    std::scoped_lock lock{memory_mutex};
    const auto start = reinterpret_cast<uintptr_t>(address);
    const bool fixed = (flags & (MAP_FIXED | MAP_FIXED_NOREPLACE)) != 0;
    const bool section = fd >= SectionFdBase && sections.count(fd);
    if (fd != -1 && !section) {
        errno = ENODEV; // files are read, not mapped
        return MAP_FAILED;
    }
    if (fixed && InArena(start, start + size)) {
        if ((flags & MAP_FIXED_NOREPLACE)) {
            auto it = pieces.upper_bound(start);
            --it;
            if (it->second.view || it->second.end < start + size) {
                errno = EEXIST;
                return MAP_FAILED;
            }
        }
        auto piece = Carve(start, start + size);
        if (section) {
            piece->second = Piece{start + size, true, fd, uint64_t(offset)};
            MapView(start, start + size, fd, uint64_t(offset), PageProtection(prot));
        } else if (prot != PROT_NONE) {
            // Private anonymous memory in the arena: a view of a section of its own.
            const int own = bb_section_create(size);
            if (own < 0) return MAP_FAILED;
            piece->second = Piece{start + size, true, own, 0};
            MapView(start, start + size, own, 0, PageProtection(prot));
        }
        return address;
    }
    if (section) {
        void* view = MapViewOfFile3(sections.at(fd).handle, GetCurrentProcess(), fixed ? address : nullptr,
                                    uint64_t(offset), size, 0, PageProtection(prot | PROT_READ | PROT_WRITE),
                                    nullptr, 0);
        if (!view) {
            errno = ENOMEM;
            return MAP_FAILED;
        }
        DWORD old = 0;
        if (prot != (PROT_READ | PROT_WRITE)) VirtualProtect(view, size, PageProtection(prot), &old);
        return view;
    }
    void* p = VirtualAlloc(address, size, prot == PROT_NONE && (flags & MAP_NORESERVE)
                                              ? MEM_RESERVE
                                              : MEM_RESERVE | MEM_COMMIT,
                           PageProtection(prot));
    if (!p || (fixed && p != address)) {
        if (p) VirtualFree(p, 0, MEM_RELEASE);
        errno = fixed ? EEXIST : ENOMEM;
        return MAP_FAILED;
    }
    return p;
}

int munmap(void* address, size_t size) {
    std::scoped_lock lock{memory_mutex};
    const auto start = reinterpret_cast<uintptr_t>(address);
    if (InArena(start, start + size)) {
        Carve(start, start + size);
        return 0;
    }
    MEMORY_BASIC_INFORMATION info{};
    if (VirtualQuery(address, &info, sizeof(info)) && info.Type == MEM_MAPPED) {
        return UnmapViewOfFile(address) ? 0 : -1;
    }
    return VirtualFree(address, 0, MEM_RELEASE) ? 0 : -1;
}

/// VirtualProtect calls made by mprotect (the frame statistics' stall lines).
unsigned long long bb_win32_protect_calls(void) {
    return bb_win32_virtual_protects.load(std::memory_order_relaxed);
}

int mprotect(void* address, size_t size, int prot) {
    // VirtualProtect cannot span allocations (views) nor uncommitted pages. In the arena the
    // pieces say where they are: a view or private memory is committed whole, a placeholder has
    // nothing. VirtualQuery is not asked there: its RegionSize is the run of pages with the same
    // attributes from the address on, which the kernel finds by walking them, 5-16 ms in a GiB
    // of touched memory and 50 ms in 3 GiB (vq_bench) — the game's heap is one view of 2.3 GiB,
    // and every write trap paid that (stalls of 50-850 ms while areas streamed in, RX 7800 XT and
    // RTX 5060 alike, 2026-10-27).
    const DWORD protect = PageProtection(prot);
    const auto end = reinterpret_cast<uintptr_t>(address) + size;
    {
        std::scoped_lock lock{memory_mutex};
        const auto start = reinterpret_cast<uintptr_t>(address);
        if (InArena(start, end)) {
            auto it = pieces.upper_bound(start);
            --it; // the piece holding start
            for (; it != pieces.end() && it->first < end; ++it) {
                if (!it->second.view && !it->second.owned) {
                    continue; // a placeholder: nothing mapped
                }
                const uintptr_t a = std::max(it->first, start), b = std::min(it->second.end, end);
                DWORD old = 0;
                bb_win32_virtual_protects.fetch_add(1, std::memory_order_relaxed);
                if (!VirtualProtect(reinterpret_cast<void*>(a), b - a, protect, &old)) {
                    errno = EACCES;
                    return -1;
                }
            }
            return 0;
        }
    }
    // Outside the arena (the runtime's own mappings): regions of one allocation in one call.
    const auto query = [](uintptr_t at, MEMORY_BASIC_INFORMATION& info) {
        return VirtualQuery(reinterpret_cast<void*>(at), &info, sizeof(info)) != 0;
    };
    const auto region_end = [end](const MEMORY_BASIC_INFORMATION& info) {
        return std::min<uintptr_t>(reinterpret_cast<uintptr_t>(info.BaseAddress) + info.RegionSize, end);
    };
    for (auto at = reinterpret_cast<uintptr_t>(address); at < end;) {
        MEMORY_BASIC_INFORMATION info{};
        if (!query(at, info)) {
            errno = ENOMEM;
            return -1;
        }
        uintptr_t next = region_end(info);
        if (info.State == MEM_COMMIT) {
            const void* allocation = info.AllocationBase;
            for (MEMORY_BASIC_INFORMATION more{};
                 next < end && query(next, more) && more.State == MEM_COMMIT && more.AllocationBase == allocation;) {
                next = region_end(more);
            }
            DWORD old = 0;
            bb_win32_virtual_protects.fetch_add(1, std::memory_order_relaxed);
            if (!VirtualProtect(reinterpret_cast<void*>(at), next - at, protect, &old)) {
                errno = EACCES;
                return -1;
            }
        }
        at = next;
    }
    return 0;
}

int bb_win32_shared_views(size_t size, void** first, void** second) {
    const HANDLE section = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                              DWORD(static_cast<unsigned long long>(size) >> 32),
                                              DWORD(size), nullptr);
    if (!section) {
        return 0;
    }
    *first = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size);
    *second = MapViewOfFile(section, FILE_MAP_ALL_ACCESS, 0, 0, size);
    CloseHandle(section); // the views keep it
    if (!*first || !*second) {
        if (*first) UnmapViewOfFile(*first);
        if (*second) UnmapViewOfFile(*second);
        *first = *second = MAP_FAILED;
        return 0;
    }
    return 1;
}

} // extern "C"
