// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the guest's memory inside the native GPU process (docs/macos-native-gpu.md). The game
// process backs direct and flexible memory with one shared memory object (runtime_memory.c,
// pool()); bb-gpu maps the same object at the same addresses, so every guest address libbbgpu
// uses stays a valid pointer. Header-only.
//
// The guest range is reserved first (no access, no backing) so that nothing else in bb-gpu (malloc,
// Metal, the loader) lands in it; mappings then replace parts of the reservation and unmapping
// puts the reservation back. bb-gpu never changes protection: the game process write-protects its
// own view to track CPU writes, and bb-gpu's writes (labels, readbacks) go through unprotected,
// as the runtime's "write backing" path does today.
#pragma once

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include <vector>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#endif

#ifndef MAP_NORESERVE
#define MAP_NORESERVE 0
#endif

namespace BbRemote {

class MemoryMirror {
public:
    MemoryMirror() = default;
    MemoryMirror(const MemoryMirror&) = delete;
    MemoryMirror& operator=(const MemoryMirror&) = delete;

    ~MemoryMirror() {
        if (end > begin) {
            munmap(reinterpret_cast<void*>(begin), end - begin);
        }
    }

    /// Reserves [begin_, end_) at exactly those addresses. False (with `error`) when any of it is
    /// taken already: the process must reserve before anything else could land there.
    bool Reserve(std::uint64_t begin_, std::uint64_t end_, std::string_view& error) {
        page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
        if (begin_ % page || end_ % page || end_ <= begin_) {
            error = "guest range not aligned to the host page size";
            return false;
        }
        // No MAP_FIXED: the hint is only taken when the whole range is free (checked below).
        void* at = mmap(reinterpret_cast<void*>(begin_), end_ - begin_, PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (at == MAP_FAILED) {
            error = "cannot reserve the guest range";
            return false;
        }
        if (reinterpret_cast<std::uint64_t>(at) != begin_) {
            munmap(at, end_ - begin_);
            error = "part of the guest range is already in use in this process";
            return false;
        }
        begin = begin_;
        end = end_;
        return true;
    }

    /// Like Reserve, but around what already occupies parts of [begin_, end_) in this process
    /// (on macOS the system allocator sets aside tens of GiB at a random address before main):
    /// those parts are Taken() and never mapped over. False only when reserving a free part fails.
    bool ReserveAround(std::uint64_t begin_, std::uint64_t end_, std::string_view& error) {
        page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
        if (begin_ % page || end_ % page || end_ <= begin_) {
            error = "guest range not aligned to the host page size";
            return false;
        }
        taken = Occupied(begin_, end_);
        std::uint64_t at = begin_;
        const auto reserve = [&](std::uint64_t from, std::uint64_t to) {
            if (from >= to) {
                return true;
            }
            void* got = mmap(reinterpret_cast<void*>(from), to - from, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (got == MAP_FAILED) {
                return false;
            }
            if (reinterpret_cast<std::uint64_t>(got) != from) {
                munmap(got, to - from);
                return false;
            }
            return true;
        };
        for (const auto& [taken_begin, taken_end] : taken) {
            if (!reserve(at, taken_begin)) {
                error = "cannot reserve a free part of the guest range";
                return false;
            }
            at = taken_end;
        }
        if (!reserve(at, end_)) {
            error = "cannot reserve a free part of the guest range";
            return false;
        }
        begin = begin_;
        end = end_;
        return true;
    }

    /// Parts of the range this process used before it could reserve them ([begin, end), sorted).
    [[nodiscard]] const std::vector<std::pair<std::uint64_t, std::uint64_t>>& Taken() const {
        return taken;
    }
    [[nodiscard]] bool Overlaps(std::uint64_t address, std::uint64_t size) const {
        for (const auto& [taken_begin, taken_end] : taken) {
            if (address < taken_end && taken_begin < address + size) {
                return true;
            }
        }
        return false;
    }

    /// Maps `size` bytes of `fd` at `offset` to `address`, readable and writable.
    bool Map(std::uint64_t address, std::uint64_t size, int fd, std::uint64_t offset) {
        if (!Inside(address, size) || offset % page || Overlaps(address, size)) {
            Report("map", address, size, EINVAL);
            return false;
        }
        std::scoped_lock lock{mutex};
        void* at = mmap(reinterpret_cast<void*>(address), size, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED, fd, static_cast<off_t>(offset));
        if (at == MAP_FAILED) {
            Report("map", address, size, errno);
            return false;
        }
        mapped_bytes += size;
        return true;
    }

    /// Returns [address, address + size) to the reservation (never the Taken() parts).
    bool Unmap(std::uint64_t address, std::uint64_t size) {
        if (!Inside(address, size)) {
            Report("unmap", address, size, EINVAL);
            return false;
        }
        std::scoped_lock lock{mutex};
        bool ok = true;
        std::uint64_t at = address;
        const std::uint64_t stop = address + size;
        const auto unmap = [&](std::uint64_t from, std::uint64_t to) {
            if (from >= to) {
                return;
            }
            void* got = mmap(reinterpret_cast<void*>(from), to - from, PROT_NONE,
                             MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
            if (got == MAP_FAILED) {
                Report("unmap", from, to - from, errno);
                ok = false;
            }
        };
        for (const auto& [taken_begin, taken_end] : taken) {
            if (taken_end <= at || taken_begin >= stop) {
                continue;
            }
            unmap(at, std::max(at, taken_begin));
            at = std::max(at, taken_end);
        }
        unmap(at, stop);
        mapped_bytes -= size < mapped_bytes ? size : mapped_bytes;
        return ok;
    }

    [[nodiscard]] std::uint64_t PageSize() const { return page; }
    [[nodiscard]] std::uint64_t MappedBytes() const { return mapped_bytes; }

private:
    [[nodiscard]] bool Inside(std::uint64_t address, std::uint64_t size) const {
        return size && address % page == 0 && size % page == 0 && address >= begin &&
               address + size <= end && address + size > address;
    }

    /// What this process already has in [from, to), page-aligned and merged.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> Occupied(std::uint64_t from,
                                                                    std::uint64_t to) const {
        std::vector<std::pair<std::uint64_t, std::uint64_t>> found;
        const auto add = [&](std::uint64_t a, std::uint64_t b) {
            a = std::max(a, from) & ~(page - 1);
            b = std::min(b, to);
            b = (b + page - 1) & ~(page - 1);
            if (a >= b) {
                return;
            }
            if (!found.empty() && a <= found.back().second) {
                found.back().second = std::max(found.back().second, b);
            } else {
                found.emplace_back(a, b);
            }
        };
#ifdef __APPLE__
        mach_vm_address_t address = from;
        while (address < to) {
            mach_vm_size_t size = 0;
            vm_region_basic_info_data_64_t info{};
            mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
            mach_port_t object = MACH_PORT_NULL;
            if (mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
                               reinterpret_cast<vm_region_info_t>(&info), &count,
                               &object) != KERN_SUCCESS ||
                address >= to) {
                break;
            }
            add(address, address + size);
            address += size;
        }
#else
        if (FILE* maps = std::fopen("/proc/self/maps", "r")) {
            unsigned long long a = 0, b = 0;
            char line[512];
            while (std::fgets(line, sizeof(line), maps)) {
                if (std::sscanf(line, "%llx-%llx", &a, &b) == 2 && a < to && b > from) {
                    add(a, b);
                }
            }
            std::fclose(maps);
        }
#endif
        return found;
    }

    void Report(const char* what, std::uint64_t address, std::uint64_t size, int error) {
        if (reports < 8) {
            ++reports;
            std::fprintf(stderr, "GPU process: guest %s %#llx+%#llx failed: %s\n", what,
                         static_cast<unsigned long long>(address),
                         static_cast<unsigned long long>(size), std::strerror(error));
        }
    }

    std::mutex mutex;
    std::uint64_t begin = 0, end = 0, page = 4096, mapped_bytes = 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> taken;
    int reports = 0;
};

} // namespace BbRemote
