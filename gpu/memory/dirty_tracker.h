// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport memory: which pages of GPU-used memory the CPU changed (to upload before the GPU reads
// them) and which the GPU changed (to copy back before the CPU reads them, readbacks). Kept per
// region of 4 MiB as bits, each region with its lock; pages the CPU has not changed are watched
// (PageWatch), so the first write marks them again.
#pragma once

#include <array>
#include "shim/bb_env.h"
#include <atomic>
#include <chrono>
#include <deque>
#include <mutex>
#include <type_traits>
#include <vector>

#include "bbport_toggles.h"
#include "core/emulator_settings.h"
#include "memory/page_lock.h"
#include "memory/page_watch.h"

namespace BbMem {

enum class Side { Cpu, Gpu };

/// Counts every change that marks pages CPU-modified anywhere (bumped before the bits are set,
/// as a guest write's fault comes before its store): a reading binding found settled at one
/// value needs no upload check while it holds (BufferCache's binding memo).
inline std::atomic<u64> cpu_change_epoch{0};

/// The page state of one region.
class Region {
public:
    void Reset(PageWatch* watch_, VAddr base_) {
        watch = watch_;
        base = base_;
        cpu_change_epoch.fetch_add(1, std::memory_order_acq_rel);
        cpu.Fill(); // never uploaded: all of it is the CPU's
        gpu.Clear();
        writable.Fill();
        readable.Fill();
    }

    VAddr Base() const {
        return base;
    }

    /// Pages [first, last) of the part of the region at `offset`, `size` bytes (offsets clamp at
    /// the region's start); false when that is empty.
    static bool Pages(s64 offset, s64 size, size_t& first, size_t& last) {
        first = size_t(std::max<s64>(offset, 0)) / PageSize;
        last = (size_t(std::max<s64>(offset + size, 0)) + PageSize - 1) / PageSize;
        last = std::min(last, PagesPerRegion);
        return first < last;
    }

    template <Side side>
    bool Modified(u64 offset, u64 size) const {
        size_t first, last;
        return Pages(s64(offset), s64(size), first, last) && Bits<side>().AnyInRange(first, last);
    }
    template <Side side>
    bool FullyModified(u64 offset, u64 size) const {
        size_t first, last;
        return !Pages(s64(offset), s64(size), first, last) || Bits<side>().AllInRange(first, last);
    }

    /// [address, address + size) changed by `side` (`set`), or brought in line (cleared).
    template <Side side, bool set>
    void Change(VAddr address, u64 size) {
        size_t first, last;
        if (!Pages(s64(address - base), s64(size), first, last)) {
            return;
        }
        if constexpr (side == Side::Cpu && set) {
            cpu_change_epoch.fetch_add(1, std::memory_order_acq_rel);
            CountWriteFaults(first, last);
        }
        auto& bits = Bits<side>();
        if constexpr (set) {
            bits.SetRange(first, last);
        } else {
            bits.UnsetRange(first, last);
        }
        if constexpr (side == Side::Cpu) {
            UpdateWatch<!set, false>();
        } else if (EmulatorSettings.GetReadbacksMode() == GpuReadbacksMode::Precise) {
            UpdateWatch<set, true>();
        }
    }

    /// After a guest write fault, the other pages of the window that are watched for CPU writes
    /// and hold no GPU-modified data are given up too (one fault instead of one per page).
    void ExtendWriteFault(VAddr window, u64 size) {
        size_t first, last;
        if (!Pages(s64(window - base), s64(size), first, last)) {
            return;
        }
        const RegionPages add(~(gpu | writable), first, last);
        if (add.None()) {
            return;
        }
        cpu_change_epoch.fetch_add(1, std::memory_order_acq_rel);
        cpu |= add;
        UpdateWatch<false, false>();
    }

    /// Calls `func(address, size)` for each run of pages of [address, address + size) modified
    /// by `side`; `clear`: they are no longer.
    template <Side side, bool clear>
    void ForEachModified(VAddr address, s64 size, auto&& func) {
        size_t first, last;
        if (!Pages(s64(address - base), size, first, last)) {
            return;
        }
        auto& bits = Bits<side>();
        // Every buffer binding asks: nothing CPU-modified is nothing to clear or watch again
        // (the CPU bits always match `writable` after a change).
        if (side == Side::Cpu && !bits.AnyInRange(first, last) &&
            !BbToggle::Disabled(BbToggle::PageTrackingEarlyExit)) {
            return;
        }
        const RegionPages mask(bits, first, last);
        if constexpr (clear) {
            bits.UnsetRange(first, last);
            if constexpr (side == Side::Cpu) {
                KeepHotPagesModified(first, last);
                UpdateWatch<true, false>();
            } else if (EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled) {
                UpdateWatch<false, true>();
            }
        }
        for (const auto& [start, end] : mask) {
            func(base + start * PageSize, (end - start) * PageSize);
        }
    }


    PageLock lock;

private:
    template <Side side>
    RegionPages& Bits() {
        return side == Side::Cpu ? cpu : gpu;
    }
    template <Side side>
    const RegionPages& Bits() const {
        return side == Side::Cpu ? cpu : gpu;
    }

    /// Pages whose watch no longer matches their bits: watched again (`watch`) or given up.
    template <bool on, bool reads>
    void UpdateWatch() {
        const RegionPages mask = reads ? (~gpu ^ readable) : (cpu ^ writable);
        if (mask.None()) {
            return;
        }
        if constexpr (reads) {
            readable = ~gpu;
        } else {
            writable = cpu;
        }
        watch->template WatchRegion<on, reads>(base, mask);
    }

    // Hot pages (BB_HOT_PAGES=1): pages the guest writes again and again (per-frame constants,
    // skinning output) cost a fault in the writer plus a re-watch on every upload; after
    // HotFaults faults a page stays unwatched and counts as always CPU-modified, uploaded on
    // every use instead. The set is rebuilt every HotPeriod. Opt-in: in Hunter's Nightmare it grew
    // to ~14k pages (56 MB), each re-uploaded on every binding, which cost far more than the
    // faults. A short period re-watches thousands of pages at once, a fault storm.
    static bool HotPagesEnabled() {
        static const bool enabled = bb_env_on("BB_HOT_PAGES");
        return enabled && !BbToggle::Disabled(BbToggle::HotPages);
    }
    static constexpr u8 HotFaults = 2;
    static constexpr auto HotPeriod = std::chrono::seconds(20);

    void CountWriteFaults(size_t first, size_t last) {
        const bool hot_enabled = HotPagesEnabled();
        for (size_t page = first; page < last; ++page) {
            if (writable.Get(page)) {
                continue; // not watched: no fault
            }
            BbStats::tracker_faults.fetch_add(1, std::memory_order_relaxed);
            if (!hot_enabled) {
                continue;
            }
            if (write_faults[page] < HotFaults) {
                ++write_faults[page];
                continue;
            }
            if (!hot.Get(page)) {
                if (num_hot == 0) {
                    hot_since = std::chrono::steady_clock::now();
                }
                hot.Set(page);
                ++num_hot;
                BbStats::hot_pages.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    void KeepHotPagesModified(size_t first, size_t last) {
        if (num_hot == 0) {
            return;
        }
        if (!HotPagesEnabled() || std::chrono::steady_clock::now() - hot_since > HotPeriod) {
            // Watched again (on this upload); counting starts over.
            BbStats::hot_pages.fetch_sub(num_hot, std::memory_order_relaxed);
            hot.Clear();
            write_faults.fill(0);
            num_hot = 0;
            return;
        }
        cpu_change_epoch.fetch_add(1, std::memory_order_acq_rel);
        cpu |= RegionPages(hot, first, last);
    }

    PageWatch* watch = nullptr;
    VAddr base = 0;
    RegionPages cpu;      ///< modified by the CPU since the GPU side's copy
    RegionPages gpu;      ///< modified by the GPU since the CPU's copy
    RegionPages writable; ///< not watched for writes (the CPU's already)
    RegionPages readable; ///< not watched for reads
    std::array<u8, PagesPerRegion> write_faults{};
    RegionPages hot;
    u32 num_hot = 0;
    std::chrono::steady_clock::time_point hot_since{};
};

class DirtyTracker {
public:
    explicit DirtyTracker(PageWatch& watch_) : watch{&watch_} {}

    bool IsRegionCpuModified(VAddr address, u64 size) {
        return ForEachRegion<true>(address, size, [](Region& r, u64 offset, u64 bytes) {
            return r.Modified<Side::Cpu>(offset, bytes);
        });
    }
    bool IsRegionGpuModified(VAddr address, u64 size) {
        return ForEachRegion<false>(address, size, [](Region& r, u64 offset, u64 bytes) {
            return r.Modified<Side::Gpu>(offset, bytes);
        });
    }

    void MarkRegionAsCpuModified(VAddr address, u64 size) {
        Locked(address, size, [](Region& r, VAddr at, u64 bytes) {
            r.Change<Side::Cpu, true>(at, bytes);
        });
    }
    void UnmarkRegionAsGpuModified(VAddr address, u64 size) {
        Locked(address, size, [](Region& r, VAddr at, u64 bytes) {
            r.Change<Side::Gpu, false>(at, bytes);
        });
    }
    /// See Region::ExtendWriteFault.
    void ExtendWriteFault(VAddr window, u64 size) {
        Locked(window, size,
               [](Region& r, VAddr at, u64 bytes) { r.ExtendWriteFault(at, bytes); });
    }

    /// A CPU write to [address, address + size): GPU-modified data there is copied back first
    /// (`on_flush`, readbacks), else the pages become CPU-modified.
    void InvalidateRegion(VAddr address, u64 size, auto&& on_flush) {
        ForEachRegion<false>(address, size, [&on_flush](Region& r, u64 offset, u64 bytes) {
            bool flush;
            {
                // Under the lock: the GPU thread may be marking the pages GPU-modified.
                std::scoped_lock lk{r.lock};
                flush = EmulatorSettings.GetReadbacksMode() != GpuReadbacksMode::Disabled &&
                        r.Modified<Side::Gpu>(offset, bytes);
                if (!flush) {
                    r.Change<Side::Cpu, true>(r.Base() + offset, bytes);
                }
            }
            if (flush) {
                on_flush();
            }
        });
    }

    /// Calls `func(address, size)` for each CPU-modified run of [address, address + size) and
    /// clears them, then `on_upload()`. `is_written` (a GPU write binding): the pages become
    /// GPU-modified, under the regions' locks until after `on_upload`.
    void ForEachUploadRange(VAddr address, u64 size, bool is_written, auto&& func,
                            auto&& on_upload) {
        const bool lock_free = !BbToggle::Disabled(BbToggle::LockFreeUploadCheck);
        // A write binding over pages the CPU did not change and the GPU already owns needs
        // neither an upload nor a state change.
        if (is_written && lock_free) {
            const bool unsettled =
                ForEachRegion<true>(address, size, [](Region& r, u64 offset, u64 bytes) {
                    return r.Modified<Side::Cpu>(offset, bytes) ||
                           !r.FullyModified<Side::Gpu>(offset, bytes);
                });
            if (!unsettled) {
                on_upload();
                return;
            }
        }
        ForEachRegion<true>(address, size, [&](Region& r, u64 offset, u64 bytes) {
            // Read bindings skip the lock when no page is CPU-modified: a guest write sets its
            // bit in the fault handler before the store, and the draw that uses the data
            // reaches this thread through the submission queue's lock.
            if (!is_written && lock_free && !r.Modified<Side::Cpu>(offset, bytes)) {
                return;
            }
            r.lock.lock();
            r.ForEachModified<Side::Cpu, true>(r.Base() + offset, s64(bytes), func);
            if (!is_written) {
                r.lock.unlock();
            }
        });
        on_upload();
        if (!is_written) {
            return;
        }
        ForEachRegion<false>(address, size, [](Region& r, u64 offset, u64 bytes) {
            r.Change<Side::Gpu, true>(r.Base() + offset, bytes);
            r.lock.unlock();
        });
    }

    /// Calls `func(address, size)` for each GPU-modified run of [address, address + size);
    /// `clear`: they are no longer.
    template <bool clear>
    void ForEachDownloadRange(VAddr address, u64 size, auto&& func) {
        Locked(address, size, [&func](Region& r, VAddr at, u64 bytes) {
            r.ForEachModified<Side::Gpu, clear>(at, s64(bytes), func);
        });
    }

private:
    Region* Find(VAddr address) const {
        return regions[address >> RegionShift].load(std::memory_order_acquire);
    }

    /// Calls `func(region, offset, bytes)` for the parts of [address, address + size) in each
    /// region (`create`: regions not used yet are created; else skipped). A `func` returning
    /// bool stops at the first true, which is returned.
    template <bool create, typename Func>
    bool ForEachRegion(VAddr address, u64 size, Func&& func) {
        constexpr bool stops = std::is_same_v<std::invoke_result_t<Func, Region&, u64, u64>, bool>;
        size_t index = address >> RegionShift;
        u64 offset = address & (RegionSize - 1);
        for (u64 remaining = size; remaining > 0; ++index, offset = 0) {
            const u64 bytes = std::min(RegionSize - offset, remaining);
            remaining -= bytes;
            Region* r = regions[index].load(std::memory_order_acquire);
            if (!r) {
                if constexpr (!create) {
                    continue;
                }
                r = Create(index);
            }
            if constexpr (stops) {
                if (func(*r, offset, bytes)) {
                    return true;
                }
            } else {
                func(*r, offset, bytes);
            }
        }
        return false;
    }

    /// `func(region, address, bytes)` under each existing region's lock.
    template <typename Func>
    void Locked(VAddr address, u64 size, Func&& func) {
        ForEachRegion<false>(address, size, [&func](Region& r, u64 offset, u64 bytes) {
            std::scoped_lock lk{r.lock};
            func(r, r.Base() + offset, bytes);
        });
    }

    Region* Create(size_t index) {
        std::scoped_lock lk{create_mutex};
        if (Region* r = regions[index].load(std::memory_order_relaxed)) {
            return r;
        }
        if (pool_used == RegionsPerBlock || blocks.empty()) {
            blocks.emplace_back(std::make_unique<std::array<Region, RegionsPerBlock>>());
            pool_used = 0;
        }
        Region* r = &(*blocks.back())[pool_used++];
        r->Reset(watch, VAddr(index) << RegionShift);
        regions[index].store(r, std::memory_order_release);
        return r;
    }

    static constexpr size_t RegionsPerBlock = 32;

    PageWatch* watch;
    std::mutex create_mutex;
    std::vector<std::unique_ptr<std::array<Region, RegionsPerBlock>>> blocks;
    size_t pool_used = 0;
    std::array<std::atomic<Region*>, NumRegions> regions{};
};

} // namespace BbMem
