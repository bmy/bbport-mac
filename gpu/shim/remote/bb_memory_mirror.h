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

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>

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

    /// Maps `size` bytes of `fd` at `offset` to `address`, readable and writable.
    bool Map(std::uint64_t address, std::uint64_t size, int fd, std::uint64_t offset) {
        if (!Inside(address, size) || offset % page) {
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

    /// Returns [address, address + size) to the reservation.
    bool Unmap(std::uint64_t address, std::uint64_t size) {
        if (!Inside(address, size)) {
            Report("unmap", address, size, EINVAL);
            return false;
        }
        std::scoped_lock lock{mutex};
        void* at = mmap(reinterpret_cast<void*>(address), size, PROT_NONE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_FIXED, -1, 0);
        if (at == MAP_FAILED) {
            Report("unmap", address, size, errno);
            return false;
        }
        mapped_bytes -= size < mapped_bytes ? size : mapped_bytes;
        return true;
    }

    [[nodiscard]] std::uint64_t PageSize() const { return page; }
    [[nodiscard]] std::uint64_t MappedBytes() const { return mapped_bytes; }

private:
    [[nodiscard]] bool Inside(std::uint64_t address, std::uint64_t size) const {
        return size && address % page == 0 && size % page == 0 && address >= begin &&
               address + size <= end && address + size > address;
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
    int reports = 0;
};

} // namespace BbRemote
