// SPDX-License-Identifier: GPL-2.0-or-later
// The texture cache: write-backs of GPU-written images and the collector that evicts images
// and samplers when memory runs short.
#include "textures/texture_cache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "bbport_free_check.h"
#include "common/scope_exit.h"
#include "core/memory.h"
#include "buffers/buffer_cache.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#ifdef __APPLE__
#include <dispatch/dispatch.h>

#include "platform/macos/memory_macos.h"
#endif

namespace VideoCore {

namespace {

constexpr u64 MiB = 1ull << 20;
constexpr u64 GiB = 1ull << 30;
/// Without a budget from the driver.
constexpr u64 DefaultPressureMemory = GiB + 512 * MiB;
constexpr u64 DefaultCriticalMemory = 3 * GiB;
/// The usage the marks are placed under (cards with more memory keep the rest for it).
constexpr s64 TargetThreshold = 8 * GiB;

u64 SteadySeconds() {
    return u64(std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

u64 EnvU64(const char* name, u64 fallback) {
    const char* env = std::getenv(name);
    return env ? std::strtoull(env, nullptr, 10) : fallback;
}

#ifdef __APPLE__
/// bbport (macOS): macOS's memory pressure (BbMacMemory::WatchPressure), polled by the collector:
/// textures and idle guest copies go sooner for 30 s (NoteVramShort), and the marks are lower
/// meanwhile (percent of the budget: 85 after a warning, 70 after a critical notice).
u64 pressure_until = 0; ///< steady seconds
u64 pressure_percent = 100;

void PollMemoryPressure() {
    BbMacMemory::WatchPressure();
    const unsigned long events = BbMacMemory::TakePressure();
    if (events == 0) {
        return;
    }
    const bool critical = (events & DISPATCH_MEMORYPRESSURE_CRITICAL) != 0;
    const u64 now = SteadySeconds();
    // A critical notice is not undone by a warning within the 30 s.
    pressure_percent = now < pressure_until ? std::min<u64>(pressure_percent, critical ? 70 : 85)
                                            : (critical ? 70 : 85);
    pressure_until = now + 30;
    TextureCache::NoteVramShort();
    std::printf("Memory: macOS reports %s memory pressure: textures and idle guest copies go "
                "sooner for 30 s, texture collector marks at %llu%% (BB_MEM_PRESSURE=0: off)\n",
                critical ? "critical" : "warning", (unsigned long long)pressure_percent);
}
#endif

} // namespace

void TextureCache::UpdateMemoryBudget() {
    if (!instance.CanReportMemoryUsage()) {
        trigger_gc_memory = 0;
        pressure_gc_memory = DefaultPressureMemory;
        critical_gc_memory = DefaultCriticalMemory;
        return;
    }
    const s64 local = s64(instance.GetTotalMemoryBudget());
    const s64 threshold = std::min<s64>(local, TargetThreshold);
    // Pressure: 60 % of the threshold free, but on cards under 16 GB 30 % of the budget (at least
    // 2 GB): 60 % of an 8 GB card's budget had the collector evicting hundreds of images every
    // 5 s at 3 GB used (#204). Critical: 20 % of the threshold (at least 512 MB).
    const s64 pressure_free = std::min(threshold * 6 / 10, std::max(local * 3 / 10, s64(2 * GiB)));
    pressure_gc_memory = std::max<u64>(
        std::min(local - pressure_free, local - s64(GiB)), DefaultPressureMemory);
    critical_gc_memory = std::max<u64>(
        std::min(local - threshold * 2 / 10, local - s64(512 * MiB)), DefaultCriticalMemory);
    trigger_gc_memory = u64((local - threshold) / 2);
}

void TextureCache::ProcessDownloadImages() {
    // Every fence comes here (~450 a frame): the lock only when something is queued.
    if (!downloads_queued.load(std::memory_order_acquire)) {
        return;
    }
    std::scoped_lock lk{download_mutex};
    downloads_queued.store(false, std::memory_order_relaxed);
    for (const ImageId image_id : download_images) {
        DownloadImageMemory(image_id, true);
    }
    download_images.clear();
}

void TextureCache::DownloadImageMemory(ImageId image_id, bool sync) {
    Image& image = slot_images[image_id];
    if (False(image.flags & ImageFlagBits::GpuModified)) {
        return;
    }
    const ImageInfo& info = image.info;
    const u32 bytes =
        info.pitch * info.size.height * info.size.depth * info.resources.layers * (info.num_bits / 8);
    ASSERT(bytes <= info.guest_size);
    const auto copy_to = [&](vk::DeviceSize offset) {
        return vk::BufferImageCopy{
            .bufferOffset = offset,
            .bufferRowLength = info.pitch,
            .bufferImageHeight = info.size.height,
            .imageSubresource{
                .aspectMask = info.props.is_depth ? vk::ImageAspectFlagBits::eDepth
                                                  : vk::ImageAspectFlagBits::eColor,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = info.resources.layers,
            },
            .imageOffset = {0, 0, 0},
            .imageExtent = {info.size.width, info.size.height, info.size.depth},
        };
    };

    // The game's memory in place: the GPU copies the image there itself, in stream order (before
    // the fences that follow). Where the memory has a VRAM copy it copies into that too, so
    // buffers bound over it read the image's data in order (issue #122: motion blur read a render
    // target as a buffer from a copy refreshed only once the CPU write-back landed, frames later).
    if (GuestInPlace() && info.guest_address != 0) {
        const auto [arena, offset] = buffer_cache.ObtainBuffer(info.guest_address, bytes, true);
        auto copy = copy_to(offset);
        runtime.DownloadImage(&image, arena, std::span{&copy, 1});
        if (buffer_cache.IsInPlace(info.guest_address, bytes)) {
            return;
        }
    }
    const auto staging = runtime.GetStagingPool().Request(bytes, MemoryType::HostCached, 16, !sync);
    auto copy = copy_to(staging.offset);
    runtime.DownloadImage(&image, staging.buffer, std::span{&copy, 1});
    const auto write_back = [this, address = info.guest_address, bytes](const auto& download) {
        download.Invalidate();
        BbFreeCheck::Check(address, bytes, download.mapped, BbFreeCheck::ImageDownload);
        Core::Memory::Instance()->TryWriteBacking(std::bit_cast<u8*>(address), download.mapped,
                                                  bytes);
        if (!WriteTracking()) {
            buffer_cache.InvalidateMemory(address, bytes);
        }
    };
    if (sync) {
        scheduler.Finish();
        write_back(staging);
    } else {
        scheduler.DeferPriorityOperation([this, staging, write_back] {
            write_back(staging);
            runtime.GetStagingPool().FreeDeferred(staging);
        });
    }
}

void TextureCache::NoteVramShort() {
    vram_short_until.store(SteadySeconds() + 30, std::memory_order_relaxed);
}

bool TextureCache::VramShort() {
    return SteadySeconds() < vram_short_until.load(std::memory_order_relaxed);
}

void TextureCache::RunGarbageCollector() {
    SCOPE_EXIT {
        ++gc_tick;
    };
    GarbageCollectImages();
    GarbageCollectSamplers();
}

void TextureCache::GarbageCollectImages() {
    if (instance.CanReportMemoryUsage()) {
        total_used_memory = instance.GetDeviceMemoryUsage();
        // Integrated GPUs (Steam Deck): the usage covers system memory heaps holding much more
        // than images, over the startup budget: the marks follow the driver's current budget.
        // BB_GC_BUDGET_MB=N: so on any GPU with a fixed budget (tests).
        static const u64 forced_budget = EnvU64("BB_GC_BUDGET_MB", 0) << 20;
        if (instance.IsIntegrated() || forced_budget) {
            u64 budget = forced_budget ? forced_budget : instance.GetDeviceMemoryBudgetNow();
#ifdef __APPLE__
            // bbport (macOS): lower for 30 s after macOS's memory pressure notifications.
            if (instance.IsIntegrated()) {
                PollMemoryPressure();
                if (SteadySeconds() < pressure_until) {
                    budget = budget / 100 * pressure_percent;
                }
            }
#endif
            if (budget != 0) {
                trigger_gc_memory = budget / 10 * 7;
                pressure_gc_memory = budget / 100 * 85;
                critical_gc_memory = budget / 100 * 95;
            }
        }
    }
    BbStats::gc_used_bytes.store(total_used_memory, std::memory_order_relaxed);
    BbStats::gc_trigger_bytes.store(trigger_gc_memory, std::memory_order_relaxed);
    BbStats::gc_critical_bytes.store(critical_gc_memory, std::memory_order_relaxed);

    // Ages in seconds: gc_tick (a guest submission each, hundreds a second) at each of the last
    // 64 seconds. Once a second: near the driver's budget the collector evicts by submissions.
    const u64 second = SteadySeconds();
    if (second != gc_second) {
        gc_second = second;
        if (instance.CanReportMemoryUsage() && !instance.IsIntegrated()) {
            const u64 budget = instance.GetDeviceMemoryBudgetNow();
            if (budget != 0 && total_used_memory >= budget / 100 * 92 &&
                vram_short_until.load(std::memory_order_relaxed) < second + 2) {
                vram_short_until.store(second + 2, std::memory_order_relaxed);
            }
        }
        BbStats::coarse_second.store(u32(second), std::memory_order_relaxed);
        gc_tick_at_second[second % gc_tick_at_second.size()] = gc_tick;
    }
    if (BbToggle::Disabled(BbToggle::TextureCollector) || total_used_memory < trigger_gc_memory) {
        return;
    }

    // Below the pressure mark an image goes once unused for BB_GC_IDLE_SECONDS (20): the
    // textures of an area left behind. Under pressure after BB_GC_PRESSURE_IDLE_SECONDS (5), over
    // the critical mark after 1 s, and only then by submissions (80/160 ticks): textures off
    // screen for a moment were evicted and uploaded again when the camera turned back, and the
    // frame rate fell the longer one played.
    static const u64 idle_seconds = std::clamp<u64>(EnvU64("BB_GC_IDLE_SECONDS", 20), 1, 63);
    static const u64 pressure_idle_seconds =
        std::clamp<u64>(EnvU64("BB_GC_PRESSURE_IDLE_SECONDS", 5), 0, 63);
    const auto tick_seconds_ago = [&](u64 seconds) {
        return gc_tick_at_second[(second - seconds) % gc_tick_at_second.size()];
    };
    const u64 idle_tick = tick_seconds_ago(idle_seconds);

    std::scoped_lock lock{mutex};
    bool pressured = false;
    bool aggressive = false;
    u64 below_tick = 0;
    size_t deletions = 0;
    u32 visited = 0;
    // `emergency`: over the critical mark after what was unused for a second went.
    const auto configure = [&](bool allow_aggressive, bool emergency = false) {
        pressured = total_used_memory >= pressure_gc_memory;
        aggressive = allow_aggressive && total_used_memory >= critical_gc_memory;
        below_tick = gc_tick - std::min<u64>(aggressive ? 160 : pressured ? 80 : 16, gc_tick);
        if (!pressured && !aggressive) {
            below_tick = std::min(below_tick, idle_tick);
        } else if (!emergency && second >= vram_short_until.load(std::memory_order_relaxed)) {
            below_tick =
                std::min(below_tick, tick_seconds_ago(aggressive ? 1 : pressure_idle_seconds));
        }
        deletions = aggressive ? 40 : pressured ? 20 : 10;
        visited = 0;
    };
    const auto evict = [&](ImageId image_id) {
        if (deletions == 0 || ++visited > 256) {
            return true;
        }
        Image& image = slot_images[image_id];
        const bool write_back = image.SafeToDownload();
        // Images that cannot go now (tiled GPU-written ones cannot be written back; GPU-written
        // ones only under pressure) neither use up the deletions nor stay first in line: the
        // oldest ten being such targets stopped the collector for good.
        if ((write_back && image.info.IsTiled()) || (write_back && !pressured)) {
            lru_cache.Touch(image.lru_id, gc_tick);
            BbStats::gc_kept_images.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        --deletions;
        if (write_back) {
            // Synchronously, while the image still watches its pages: a deferred write-back
            // landed after they were given up, over whatever the game stored there meanwhile.
            DownloadImageMemory(image_id, true);
            ++gc_downloads;
        }
        ++gc_evictions;
        BbStats::gc_freed_images.fetch_add(1, std::memory_order_relaxed);
        FreeImage(image_id);
        if (total_used_memory < critical_gc_memory) {
            if (aggressive) {
                deletions >>= 2;
                aggressive = false;
                return false;
            }
            if (pressured && total_used_memory < pressure_gc_memory) {
                deletions >>= 1;
                pressured = false;
            }
        }
        return false;
    };

    configure(false);
    lru_cache.ForEachItemBelow(below_tick, evict);
    if (total_used_memory >= critical_gc_memory) {
        configure(true);
        lru_cache.ForEachItemBelow(below_tick, evict);
    }
    // bbport (macOS): over the critical mark even without any cached image (the rest of the
    // usage: guest copies, buffers, the driver's), evicting the images in use cannot get under it:
    // they were only uploaded again, every tick. The images are taken at the larger of their
    // guest size and all Vulkan images' memory (scaled targets are larger than the guest's).
    // BB_GC_THRASH_GUARD=0: as on Linux.
    bool hopeless = false;
#ifdef __APPLE__
    static const bool thrash_guard = EnvU64("BB_GC_THRASH_GUARD", 1) != 0;
    const u64 cached_images = std::max(BbStats::live_image_bytes.load(std::memory_order_relaxed),
                                       BbStats::vk_image_bytes.load(std::memory_order_relaxed));
    hopeless = thrash_guard && instance.IsIntegrated() &&
               total_used_memory - std::min(cached_images, total_used_memory) >= critical_gc_memory;
    if (hopeless) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::printf("Texture cache: %llu MiB in use without its %llu MiB of images, over the "
                        "critical mark (%llu MiB): images in use are not evicted for it "
                        "(BB_GC_THRASH_GUARD=0: they are)\n",
                        (unsigned long long)((total_used_memory - cached_images) >> 20),
                        (unsigned long long)(cached_images >> 20),
                        (unsigned long long)(critical_gc_memory >> 20));
        }
    }
#endif
    if (total_used_memory >= critical_gc_memory && !hopeless) {
        // Still over it: what was used a moment ago goes too (out of memory otherwise).
        configure(true, true);
        lru_cache.ForEachItemBelow(below_tick, evict);
    }
    // Evictions under pressure, at most every 5 s.
    if (pressured || gc_downloads != 0) {
        const auto now = std::chrono::steady_clock::now();
        if (now - gc_report_time >= std::chrono::seconds(5)) {
            std::printf("Texture cache: memory pressure, %llu of %llu MiB (critical %llu): "
                        "%llu images evicted, %llu written back since the last report\n",
                        (unsigned long long)(total_used_memory >> 20),
                        (unsigned long long)(pressure_gc_memory >> 20),
                        (unsigned long long)(critical_gc_memory >> 20),
                        (unsigned long long)gc_evictions, (unsigned long long)gc_downloads);
            gc_report_time = now;
            gc_evictions = gc_downloads = 0;
        }
    }
}

void TextureCache::GarbageCollectSamplers() {
    const u64 used = samplers.size();
    if (used < trigger_gc_samplers) {
        return;
    }
    std::scoped_lock lock{samplers_mutex};
    size_t deletions = 0;
    u64 below_tick = 0;
    const auto configure = [&](bool allow_aggressive) {
        const bool pressured = used >= pressure_gc_samplers;
        const bool aggressive = allow_aggressive && used >= critical_gc_samplers;
        below_tick = gc_tick - std::min<u64>(aggressive ? 160 : pressured ? 80 : 16, gc_tick);
        deletions = aggressive ? 40 : pressured ? 20 : 10;
    };
    const auto evict = [&](u64 key) {
        if (deletions == 0) {
            return true;
        }
        --deletions;
        sampler_lru.Free(samplers.at(key).lru_id);
        samplers.erase(key);
        return false;
    };
    configure(false);
    sampler_lru.ForEachItemBelow(below_tick, evict);
    if (used >= critical_gc_samplers) {
        configure(true);
        sampler_lru.ForEachItemBelow(below_tick, evict);
    }
}

} // namespace VideoCore
