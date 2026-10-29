// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "memory/page_watch.h"
#include "shim/bb_env.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>

#include <sys/mman.h>
#include <unistd.h>

#include "bblayer_write_traps.h"
#include "bbport_toggles.h"
#include "games/game_profile.h"
#include "common/assert.h"
#include "common/error.h"
#include "common/logging/log.h"
#include "common/range_lock.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/memory.h"
#include "core/signals.h"
#include "memory/fault_sites.h"
#include "memory/page_lock.h"
#include "buffers/buffer.h"
#include "render/rasterizer.h"

extern "C" void runtime_memory_set_write_watch(uintptr_t address, uint64_t size, int watch);
extern "C" int runtime_memory_vma_info(uintptr_t address, int* prot, int* type, uintptr_t* end);

namespace BbMem {

namespace {

using Core::MemoryPermission;

/// Zero-filled memory for a table over the address space, committed where it is touched.
template <typename T>
T* ReserveTable(size_t count, const char* name) {
    void* mem = mmap(nullptr, count * sizeof(T), PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    ASSERT_MSG(mem != MAP_FAILED, "page watch table {}", name);
    return static_cast<T*>(mem);
}

/// Watchers of a page: writes in the low 7 bits, a read watcher in the top bit (only the buffer
/// tracker watches reads, and buffers do not overlap).
constexpr u8 ReadWatcher = 0x80;
constexpr u8 WriteWatchers = 0x7f;

constexpr MemoryPermission Perms(u8 watchers) {
    MemoryPermission perms = MemoryPermission::None;
    if (!(watchers & ReadWatcher)) {
        perms |= MemoryPermission::Read;
    }
    if (!(watchers & WriteWatchers)) {
        perms |= MemoryPermission::Write;
    }
    return perms;
}

/// The pages under images are trapped for
/// writes (WriteTraps::Image): a write nobody announced (the game's own copy loops, its heap over
/// an old texture) still reaches the texture cache, on any game and GPU, without hooks in the
/// game's code; the hooks and the libc wrappers only make it sooner. BB_IMAGE_TRAPS=0: announced
/// writes only. Latched at the first use: the counts must match their traps for the whole run.
bool ImageTraps() {
    static const bool on = [] {
        const bool enabled = !bb_env_off("BB_IMAGE_TRAPS") && !VideoCore::WriteVerify();
        if (enabled) {
            std::printf("Guest memory: writes under images are trapped (BB_IMAGE_TRAPS=0: "
                        "announced writes only)\n");
        }
        return enabled;
    }();
    return on;
}

/// Coalesces page protection changes into runs of one permission.
class Runs {
public:
    explicit Runs(PageWatch::Impl& impl_, MemoryPermission perms_) : impl{impl_}, perms{perms_} {}
    ~Runs() {
        Flush();
    }
    /// The next page (in ascending order), its permission now and whether it changed.
    void Page(size_t page, MemoryPermission now, bool changed);

private:
    void Flush();

    PageWatch::Impl& impl;
    MemoryPermission perms;
    size_t begin = 0;
    size_t pages = 0;     ///< the run to apply
    size_t potential = 0; ///< the run with the unchanged pages of the same permission after it
};

} // namespace

struct PageWatch::Impl {
    static inline Vulkan::Rasterizer* rasterizer = nullptr;

    u8* watchers = ReserveTable<u8>(NumPages, "watchers");
    PageLock* locks = ReserveTable<PageLock>(NumRegions, "locks");
    /// ImageTraps: the images over each page; a page is trapped while one is there.
    u16* image_watchers = nullptr;
    std::once_flag image_watchers_once;

    explicit Impl(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }
    virtual ~Impl() {
        munmap(watchers, NumPages * sizeof(u8));
        munmap(locks, NumRegions * sizeof(PageLock));
        if (image_watchers) {
            munmap(image_watchers, NumPages * sizeof(u16));
        }
    }

    virtual void Protect(VAddr address, u64 size, MemoryPermission perms) = 0;

    static bool OnFault(void* context, void* fault_address);

    template <bool watch, bool reads>
    static u8 Change(u8& count) {
        if constexpr (reads && watch) {
            count |= ReadWatcher;
        } else if constexpr (reads) {
            ASSERT_MSG(count & ReadWatcher, "no read watcher");
            count &= WriteWatchers;
        } else if constexpr (watch) {
            ASSERT_MSG((count & WriteWatchers) != WriteWatchers, "too many write watchers");
            ++count;
        } else {
            ASSERT_MSG(count & WriteWatchers, "not enough write watchers");
            --count;
        }
        return count;
    }
    /// Whether the change of a page's watchers changes its permission.
    template <bool watch, bool reads>
    static bool Toggles(u8 count) {
        const u8 n = reads ? (count >> 7) : (count & WriteWatchers);
        return watch ? n == 1 : n == 0;
    }

    template <bool watch>
    void WatchWrites(VAddr address, u64 size) {
        size_t page = address >> PageShift;
        const size_t end = (address + size + PageSize - 1) >> PageShift;
        Common::RangeLockGuard lk(locks + page / PagesPerRegion,
                                  locks + (end + PagesPerRegion - 1) / PagesPerRegion);
        Runs runs{*this, Perms(watchers[page])};
        for (; page != end; ++page) {
            const u8 count = Change<watch, false>(watchers[page]);
            runs.Page(page, Perms(count), Toggles<watch, false>(count));
        }
    }

    template <bool watch, bool reads>
    void WatchRegion(VAddr region, const RegionPages& mask) {
        const auto first = mask.FirstRange();
        const auto last = mask.LastRange();
        if (first.first >= first.second) {
            return;
        }
        if (first == last) {
            // One run of pages.
            if constexpr (!reads) {
                WatchWrites<watch>(region + (first.first << PageShift),
                                   (first.second - first.first) << PageShift);
                return;
            }
        }
        const size_t base = region >> PageShift;
        ASSERT(base % PagesPerRegion == 0);
        std::scoped_lock lk{locks[base / PagesPerRegion]};
        Runs runs{*this, Perms(watchers[base + first.first])};
        for (size_t page = first.first; page < last.second; ++page) {
            u8& count = watchers[base + page];
            if (!mask.Get(page)) {
                runs.Page(base + page, Perms(count), false);
                continue;
            }
            Change<watch, reads>(count);
            runs.Page(base + page, Perms(count), Toggles<watch, reads>(count));
        }
    }

    template <bool watch>
    void WatchImageTraps(VAddr address, u64 size) {
        std::call_once(image_watchers_once,
                       [this] { image_watchers = ReserveTable<u16>(NumPages, "images"); });
        size_t page = address >> PageShift;
        const size_t end = (address + size + PageSize - 1) >> PageShift;
        Common::RangeLockGuard lk(locks + page / PagesPerRegion,
                                  locks + (end + PagesPerRegion - 1) / PagesPerRegion);
        // Runs of pages whose first image came or last one went: one trap change each.
        size_t run_begin = 0, run_pages = 0;
        const auto flush = [&] {
            if (run_pages != 0) {
                BbLayer::WriteTraps::Set(run_begin << PageShift, run_pages << PageShift,
                                         BbLayer::WriteTraps::Image, watch);
                run_pages = 0;
            }
        };
        for (; page != end; ++page) {
            u16& count = image_watchers[page];
            if constexpr (watch) {
                if (count++ != 0) {
                    continue;
                }
            } else {
                ASSERT_MSG(count > 0, "image watchers below zero");
                if (--count != 0) {
                    continue;
                }
            }
            if (run_pages != 0 && run_begin + run_pages == page) {
                ++run_pages;
            } else {
                flush();
                run_begin = page;
                run_pages = 1;
            }
        }
        flush();
    }
};

namespace {

void Runs::Page(size_t page, MemoryPermission now, bool changed) {
    if (now != perms) {
        Flush();
        perms = now;
    } else if (pages != 0) {
        // Same permission: the run may cover it if a changed page follows.
        ++potential;
    }
    if (changed) {
        if (pages == 0) {
            begin = page;
            potential = 1;
        }
        pages = potential;
    }
}

void Runs::Flush() {
    if (pages != 0) {
        impl.Protect(begin << PageShift, pages << PageShift, perms);
        pages = potential = 0;
    }
}

/// The runtime's write watch (BB_WRITE_VERIFY=1: page protection too, to see writes nothing
/// announced).
struct SignalWatch final : PageWatch::Impl {
    explicit SignalWatch(Vulkan::Rasterizer* rasterizer_) : Impl{rasterizer_} {
        // After BB_VRAM_ACCESS_TRAP (priority 0, diagnostics).
        Core::Signals::Instance()->RegisterAccessViolationHandler(OnFault, 1);
    }

    void Protect(VAddr address, u64 size, MemoryPermission perms) override {
        runtime_memory_set_write_watch(address, size, !True(perms & MemoryPermission::Write));
        if (!VideoCore::WriteVerify()) {
            return;
        }
        ASSERT_MSG(perms != MemoryPermission::Write, "write-only pages");
        Core::Memory::Instance()->GetAddressSpace().Protect(address, size, perms);
    }
};

} // namespace

bool PageWatch::Impl::OnFault(void* context, void* fault_address) {
    const auto address = reinterpret_cast<VAddr>(fault_address);
    const u64 rip = u64(Common::Gregs(context)[REG_RIP]); // bbport: portable (macOS)
    FaultSites::Note(context, address);
    // The draw recording thread handles its faults inline too.
    const bool gpu_thread = rasterizer->IsGpuSideThread();
    if (gpu_thread) {
        BbStats::gpu_signal_faults.fetch_add(1, std::memory_order_relaxed);
    }
    const bool write = Common::IsWriteError(context);
    // Traps on reads too (no access): their owner first.
    const unsigned reasons = BbLayer::WriteTraps::Reasons(address);
    if (reasons & BbLayer::WriteTraps::VramData) {
        return rasterizer->OnVramDataAccess(address, gpu_thread);
    }
    if (reasons & BbLayer::WriteTraps::Probe) {
        return rasterizer->OnProbeAccess(address, rip, write, gpu_thread);
    }
    if (reasons & BbLayer::WriteTraps::QueryReads) {
        return rasterizer->OnOcclusionPageAccess(address, rip, write, gpu_thread);
    }
    if (!write) {
        BbStats::read_faults.fetch_add(1, std::memory_order_relaxed);
        BbStats::Timer timer{BbStats::t_read_faults};
        return rasterizer->ReadMemory(address, 8, gpu_thread);
    }
    if (ImageTraps() && !(reasons & BbLayer::WriteTraps::Image)) {
        // Not an image's trap: another owner's being lifted by another thread, or one lifted
        // meanwhile (the write runs again), or a write the game may not do.
        int prot = 0, type = -1;
        uintptr_t end = 0;
        return runtime_memory_vma_info(address, &prot, &type, &end) && (prot & 0x2);
    }
    // The game's own bookkeeping (Profile::dynamic_writers: its heap's headers among GPU data) as
    // an announced write: the traps of the whole 64 KiB lifted at once and the GPU side told, without
    // waiting for the draw pipeline, as for libc copies. Streaming made thousands of these a
    // second, each held up the game's thread until the draws queued were recorded (hitches of
    // 60-900 ms on a 4-thread laptop). A second fault at the same address takes the full path.
    static const bool fast_writers = [] {
        // Off by default: a device loss on the GTX 1050 laptop came 6 minutes after it went in,
        // and it took only ~10 of the thousands of faults a second there (most are image traps).
        return bb_env_on("BB_FAST_WRITER_FAULTS");
    }();
    thread_local VAddr last_fast = 0;
    if (fast_writers && !gpu_thread && last_fast != address && Game::IsDynamicWriter(rip)) {
        last_fast = address;
        BbStats::fast_writer_faults.fetch_add(1, std::memory_order_relaxed);
        rasterizer->OnCpuWrite(address & ~VAddr{0xffff}, 0x10000);
        FaultSites::Done();
        return true;
    }
    last_fast = 0;
    BbStats::Timer timer{BbStats::t_write_faults};
    const bool handled = rasterizer->OnWriteFault(address, gpu_thread, rip);
    FaultSites::Done();
    return handled;
}

PageWatch::PageWatch(Vulkan::Rasterizer* rasterizer)
    : impl{std::make_unique<SignalWatch>(rasterizer)} {}

PageWatch::~PageWatch() = default;

template <bool watch>
void PageWatch::WatchImage(VAddr address, u64 size) {
    impl->WatchWrites<watch>(address, size);
    if (ImageTraps()) {
        impl->WatchImageTraps<watch>(address, size);
    }
}

template <bool watch, bool reads>
void PageWatch::WatchRegion(VAddr region, const RegionPages& mask) {
    impl->WatchRegion<watch, reads>(region, mask);
}

template void PageWatch::WatchImage<true>(VAddr, u64);
template void PageWatch::WatchImage<false>(VAddr, u64);
template void PageWatch::WatchRegion<true, true>(VAddr, const RegionPages&);
template void PageWatch::WatchRegion<true, false>(VAddr, const RegionPages&);
template void PageWatch::WatchRegion<false, true>(VAddr, const RegionPages&);
template void PageWatch::WatchRegion<false, false>(VAddr, const RegionPages&);

} // namespace BbMem
