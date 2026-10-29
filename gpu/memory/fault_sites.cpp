// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "memory/fault_sites.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <string>

#include <dlfcn.h>
#include <fmt/format.h>
#include <unistd.h>

#include "bbport_portable.h"
#include "common/signal_context.h"

namespace BbMem::FaultSites {

namespace {

constexpr u64 GuestImage = 0x800000000ull, GuestImageEnd = 0x810000000ull;
constexpr u64 HostSite = 1ull << 63; ///< the key of host code: its address with this bit
constexpr u64 ReadSite = 1ull << 62; ///< a read fault (readbacks protect GPU data from reads)

bool InGuestImage(u64 address) {
    return address >= GuestImage && address < GuestImageEnd;
}

u64 Slot(u64 key, u32 shift) {
    return (key * 0x9E3779B97F4A7C15ull) >> shift;
}

/// Open addressing, keyed by the faulting instruction.
struct Site {
    std::atomic<u64> key{0};
    std::atomic<u64> caller{0};
    std::atomic<u64> count{0};
    std::atomic<u64> last_address{0};
};
std::array<Site, 512> sites;
std::atomic<u64> dropped{0};

/// Keyed by instruction and image.
struct ImageSite {
    std::atomic<u64> key{0};
    std::atomic<u64> site{0}, caller{0}, count{0}, image{0}, size{0};
    std::atomic<u32> width{0}, height{0}, format{0}, tiling{0};
};
std::array<ImageSite, 256> image_sites;

/// The fault this thread handles (0: none).
thread_local u64 current_key = 0, current_caller = 0;

/// Claims the slot of `key` in `table` (nullptr: full).
template <typename Table>
auto* Claim(Table& table, u64 key, u32 shift) {
    for (u64 i = 0, slot = Slot(key, shift); i < table.size(); ++i) {
        auto& entry = table[(slot + i) % table.size()];
        u64 expected = 0;
        if (entry.key.load(std::memory_order_relaxed) == key ||
            entry.key.compare_exchange_strong(expected, key)) {
            return &entry;
        }
    }
    return static_cast<decltype(&table[0])>(nullptr);
}

/// "+offset" for guest code, the symbol for host code.
std::string Name(u64 key) {
    const char* access = key & ReadSite ? "read:" : "";
    key &= ~ReadSite;
    if (!(key & HostSite)) {
        return fmt::format("{}+{:#x}", access, key);
    }
    const u64 address = key & ~HostSite;
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(address), &info) && info.dli_sname) {
        return fmt::format("{}host:{}+{:#x}", access, info.dli_sname,
                           address - u64(info.dli_saddr));
    }
    if (info.dli_fname) {
        return fmt::format("{}host:{}+{:#x}", access, info.dli_fname,
                           address - u64(info.dli_fbase));
    }
    return fmt::format("{}host:{:#x}", access, address);
}

/// Copies from this process's memory, failing softly on unmapped addresses.
bool ReadOwn(void* to, u64 from, size_t bytes) {
    // bbport: portable (process_vm_readv on Linux, Mach on macOS).
    return BbPortable::ReadSelf(to, reinterpret_cast<const void*>(from), bytes);
}

void ReportImages() {
    bool any = false;
    for (auto& site : image_sites) {
        const u64 count = site.count.exchange(0, std::memory_order_relaxed);
        if (count == 0) {
            continue;
        }
        std::printf("%s %s (from +%#llx) %llu into image %#llx size %llu %ux%u fmt %u tile %u",
                    any ? ";" : "Image write sites (guest code -> texture, this window):",
                    Name(site.site.load()).c_str(), (unsigned long long)site.caller.load(),
                    (unsigned long long)count, (unsigned long long)site.image.load(),
                    (unsigned long long)site.size.load(), site.width.load(), site.height.load(),
                    site.format.load(), site.tiling.load());
        any = true;
    }
    if (any) {
        std::printf("\n");
    }
}

} // namespace

void Note(void* context, VAddr address) {
    const greg_t* regs = Common::Gregs(context); // bbport: portable (macOS)
    current_key = 0;
    const u64 rip = u64(regs[REG_RIP]);
    const bool guest = InGuestImage(rip);
    u64 caller = 0;
    if (guest) {
        // [rbp + 8] when the guest code keeps frames (its memcpy-like leaves do not).
        u64 frame[2] = {};
        if (ReadOwn(frame, u64(regs[REG_RBP]), sizeof(frame)) && InGuestImage(frame[1])) {
            caller = frame[1];
        }
    } else {
        // Host code (a libc import run natively, the runtime): the first guest return address on
        // the stack is its guest caller.
        std::array<u64, 64> stack{};
        if (ReadOwn(stack.data(), u64(regs[REG_RSP]), sizeof(stack))) {
            const auto it = std::find_if(stack.begin(), stack.end(), InGuestImage);
            caller = it != stack.end() ? *it : 0;
        }
    }
    const u64 key =
        (guest ? rip - GuestImage : rip | HostSite) | (Common::IsWriteError(context) ? 0 : ReadSite);
    current_key = key;
    current_caller = caller ? caller - GuestImage : 0;
    if (auto* site = Claim(sites, key, 55)) {
        site->caller.store(current_caller, std::memory_order_relaxed);
        site->last_address.store(address, std::memory_order_relaxed);
        site->count.fetch_add(1, std::memory_order_relaxed);
    } else {
        dropped.fetch_add(1, std::memory_order_relaxed);
    }
}

void Done() {
    current_key = 0;
}

void NoteImage(VAddr image_address, u64 image_size, u32 width, u32 height, u32 format,
               u32 tiling) {
    if (current_key == 0) {
        return;
    }
    const u64 key = (current_key * 0x9E3779B97F4A7C15ull) ^ image_address;
    if (auto* site = Claim(image_sites, key, 56)) {
        site->site.store(current_key, std::memory_order_relaxed);
        site->caller.store(current_caller, std::memory_order_relaxed);
        site->image.store(image_address, std::memory_order_relaxed);
        site->size.store(image_size, std::memory_order_relaxed);
        site->width.store(width, std::memory_order_relaxed);
        site->height.store(height, std::memory_order_relaxed);
        site->format.store(format, std::memory_order_relaxed);
        site->tiling.store(tiling, std::memory_order_relaxed);
        site->count.fetch_add(1, std::memory_order_relaxed);
    }
}

void Report() {
    struct Row {
        u64 key, caller, count, address;
    };
    std::array<Row, sites.size()> rows{};
    size_t n = 0;
    u64 total = 0;
    for (auto& site : sites) {
        if (const u64 count = site.count.exchange(0, std::memory_order_relaxed)) {
            rows[n++] = {site.key.load(), site.caller.load(), count, site.last_address.load()};
            total += count;
        }
    }
    if (total == 0) {
        return;
    }
    ReportImages();
    std::sort(rows.begin(), rows.begin() + n,
              [](const Row& a, const Row& b) { return a.count > b.count; });
    std::printf("Write fault sites (guest code, this window): %llu faults",
                (unsigned long long)total);
    for (size_t i = 0; i < std::min<size_t>(n, 12); ++i) {
        std::printf("%s %s (from +%#llx) %llu at %#llx", i ? ";" : ":", Name(rows[i].key).c_str(),
                    (unsigned long long)rows[i].caller, (unsigned long long)rows[i].count,
                    (unsigned long long)rows[i].address);
    }
    std::printf("\n");
}

} // namespace BbMem::FaultSites
