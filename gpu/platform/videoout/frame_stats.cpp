// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: BB_FRAME_STATS=1: every 5 seconds, the flip rate, frame time spread and what the GPU
// side spent its time on; long frames (over 40 ms) get a line of their own. BB_FRAME_LOG=path:
// one CSV line per flip.

#include <algorithm>
#include "shim/bb_env.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <dirent.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

#include "bbport_heap_sites.h"
#include "bbport_threads.h"
#include "bbport_toggles.h"
#include "bbport_wait_trace.h"
#include "core/emulator_settings.h"
#include "platform/kernel/event_queue.h"
#include "platform/videoout/frame_stats.h"
#include "bbport_sections.h"

extern "C" uint64_t runtime_tsc_frequency(void);

extern "C" int runtime_memory_pool_fd(void);
#include "render/vk_breadcrumbs.h"
#include "memory/fault_sites.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"
#include "remote/bb_remote.h"

extern "C" void runtime_sleep_stats(uint64_t* calls, uint64_t* ns);
#ifdef _WIN32
extern "C" unsigned long long bb_win32_protect_calls(void); // posix_memory.cpp
#endif
extern "C" void runtime_memory_trap_stats(uint64_t* wait_ns, uint64_t* apply_ns, uint64_t* calls,
                                          uint64_t* bytes);
extern "C" void runtime_wait_report(double frames);

namespace Vulkan {
extern std::atomic<u64> g_bb_compile_ns;
extern std::atomic<u32> g_bb_compiles;
} // namespace Vulkan

namespace BbVideoOut {

bool FrameStatsEnabled() {
    static const bool enabled = EmulatorSettingsImpl::Flag("BB_FRAME_STATS", false);
    return enabled;
}

/// bbport: memory per statistics window: the kernel's count for the game (VRAM and GTT of its
/// DRM clients, RSS) and the parts we know of. GTT holds the game's direct memory (fixed).
static void PrintMemory() {
    u64 vram_kib = 0, gtt_kib = 0;
    std::vector<u64> clients;
    if (DIR* dir = opendir("/proc/self/fdinfo")) {
        while (const dirent* entry = readdir(dir)) {
            char path[64];
            std::snprintf(path, sizeof(path), "/proc/self/fdinfo/%s", entry->d_name);
            FILE* file = std::fopen(path, "r");
            if (!file) {
                continue;
            }
            char line[160];
            bool drm = false;
            unsigned long long client = 0, vram = 0, gtt = 0;
            while (std::fgets(line, sizeof(line), file)) {
                drm |= std::strncmp(line, "drm-driver:", 11) == 0;
                std::sscanf(line, "drm-client-id: %llu", &client);
                std::sscanf(line, "drm-memory-vram: %llu", &vram);
                std::sscanf(line, "drm-memory-gtt: %llu", &gtt);
            }
            std::fclose(file);
            // Duplicated descriptors share a client.
            if (drm && std::find(clients.begin(), clients.end(), client) == clients.end()) {
                clients.push_back(client);
                vram_kib += vram;
                gtt_kib += gtt;
            }
        }
        closedir(dir);
    }
    unsigned long long size_pages = 0, rss_pages = 0;
    if (FILE* statm = std::fopen("/proc/self/statm", "r")) {
        if (std::fscanf(statm, "%llu %llu", &size_pages, &rss_pages) != 2) {
            rss_pages = 0;
        }
        std::fclose(statm);
    }
    u64 rss = u64(rss_pages) * u64(sysconf(_SC_PAGESIZE));
#ifdef __APPLE__
    // bbport (macOS): no /proc; resident size from Mach (VRAM/GTT stay 0: unified memory).
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        rss = info.resident_size;
    }
#endif
// bbport (macOS): not counted; the pool is a POSIX shared memory object, not a memfd.
#if !defined(_WIN32) && !defined(__APPLE__)
    // The game's memory in the runtime's memfd: allocated pages (lazy guest memory leaves the
    // ones nobody touched out; the GPU's imports pin theirs).
    if (const int fd = runtime_memory_pool_fd(); fd >= 0) {
        u64 resident = 0;
        for (off_t at = 0;;) {
            const off_t data = lseek(fd, at, SEEK_DATA);
            if (data < 0) {
                break;
            }
            const off_t hole = lseek(fd, data, SEEK_HOLE);
            if (hole <= data) {
                break;
            }
            resident += u64(hole - data);
            at = hole;
        }
        std::printf("Memory: the game's memory %llu MiB allocated in its memfd\n",
                    (unsigned long long)(resident >> 20));
    }
#endif
    const u64 device = BbStats::device_alloc_bytes.load() - BbStats::device_free_bytes.load();
    u64 vma_blocks = 0, vma_used = 0;
    Vulkan::VmaDeviceUsage(vma_blocks, vma_used);
    std::printf("Memory: VRAM %llu MiB, GTT %llu MiB, RSS %llu MiB; Vulkan memory %llu MiB "
                "allocated (VMA blocks %llu MiB, %llu MiB of them used); Vulkan images %llu MiB; "
                "cached images %llu MiB in %llu; guest blocks in VRAM %llu MiB allocated, "
                "%llu MiB of it unused (%llu MiB moved back idle, %llu MiB unmapped, %llu MiB freed; %llu chunks, %llu emptied, blocks kept: %llu GPU-written, %llu not direct memory); texture collector: usage %llu MiB, starts at %llu MiB, %llu images freed, %llu GPU-written kept\n",
                (unsigned long long)(vram_kib >> 10), (unsigned long long)(gtt_kib >> 10),
                (unsigned long long)(rss >> 20), (unsigned long long)(device >> 20),
                (unsigned long long)(vma_blocks >> 20), (unsigned long long)(vma_used >> 20),
                (unsigned long long)(BbStats::vk_image_bytes.load() >> 20),
                (unsigned long long)(BbStats::live_image_bytes.load() >> 20),
                (unsigned long long)BbStats::live_images.load(),
                (unsigned long long)(BbStats::residency_alloc_bytes.load() >> 20),
                (unsigned long long)(BbStats::residency_unused_bytes.load() >> 20),
                (unsigned long long)(BbStats::vram_idle_bytes.exchange(0) >> 20),
                (unsigned long long)(BbStats::vram_unmapped_bytes.exchange(0) >> 20),
                (unsigned long long)(BbStats::vram_chunks_freed_bytes.exchange(0) >> 20),
                (unsigned long long)BbStats::residency_chunk_count.load(),
                (unsigned long long)BbStats::evacuations.exchange(0),
                (unsigned long long)BbStats::evac_kept_gpu.exchange(0),
                (unsigned long long)BbStats::evac_kept_other.exchange(0),
                (unsigned long long)(BbStats::gc_used_bytes.load() >> 20),
                (unsigned long long)(BbStats::gc_trigger_bytes.load() >> 20),
                (unsigned long long)BbStats::gc_freed_images.exchange(0),
                (unsigned long long)BbStats::gc_kept_images.exchange(0));
}

namespace {

using Clock = std::chrono::steady_clock;

/// The counters read at every flip: a long frame and the frame log print their change since the
/// last flip.
struct FlipCounters {
    u64 gpu_ns = 0; // the GPU thread's CPU time (Windows: wall time)
    u64 images = 0, image_bytes = 0, buffer_bytes = 0;
    u64 t[6] = {}; // ns in: resident, protect, image create, image refresh, staging, host copy waits
    u64 protect_calls = 0, protect_pages = 0, revoke_calls = 0, revoke_pages = 0;
    u64 minor_faults = 0, signal_faults = 0, process_faults = 0;
    u64 copy_ns = 0, copy_bytes = 0, copy_cpu = 0, copy_sys = 0, copy_faults = 0;
    u64 read_faults = 0, read_fault_ns = 0, write_fault_ns = 0;
    u64 draws = 0, dispatches = 0, submissions = 0;
    u64 sys_us = 0, user_us = 0, preempted = 0, waits = 0;
};

FlipCounters ReadFlipCounters() {
    FlipCounters c;
    c.protect_calls = BbStats::protect_calls.load();
    c.protect_pages = BbStats::protect_pages.load();
    c.revoke_calls = BbStats::protect_revoke_calls.load();
    c.revoke_pages = BbStats::protect_revoke_pages.load();
    c.minor_faults = BbStats::gpu_minor_faults.load();
    c.signal_faults = BbStats::gpu_signal_faults.load();
    c.copy_cpu = BbStats::t_copy_cpu.load();
    c.copy_sys = BbStats::copy_sys_us.load();
    c.copy_faults = BbStats::copy_minflt.load();
    c.read_faults = BbStats::read_faults.load();
    c.read_fault_ns = BbStats::t_read_faults.load();
    c.write_fault_ns = BbStats::t_write_faults.load();
    c.copy_ns = BbStats::t_copy.load();
    c.copy_bytes = BbStats::copy_bytes.load();
    if (rusage usage{}; getrusage(RUSAGE_SELF, &usage) == 0) {
        c.process_faults = usage.ru_minflt;
    }
    const u64 t[6] = {BbStats::t_resident.load(), BbStats::t_protect.load(),
                      BbStats::t_image_create.load(), BbStats::t_refresh.load(),
                      BbStats::t_staging.load(), BbStats::t_host_wait.load()};
    std::copy(std::begin(t), std::end(t), std::begin(c.t));
    c.draws = BbStats::draws.load();
    c.dispatches = BbStats::dispatches.load();
    c.submissions = BbStats::submissions.load();
    c.sys_us = BbStats::gpu_sys_us.load();
    c.user_us = BbStats::gpu_user_us.load();
    c.preempted = BbStats::gpu_invol_switches.load();
    c.waits = BbStats::gpu_vol_switches.load();
    if (const int clock = BbStats::gpu_thread_clock.load(); clock != -1) {
        timespec ts{};
        clock_gettime(static_cast<clockid_t>(clock), &ts);
        c.gpu_ns = u64(ts.tv_sec) * 1000000000ull + u64(ts.tv_nsec);
    } else {
#ifdef __APPLE__
        // bbport (macOS): no CPU clock for another thread; the GPU thread records its own CPU
        // time after each submission (getrusage RUSAGE_THREAD, platform/macos), so use that.
        c.gpu_ns = (BbStats::gpu_user_us.load() + BbStats::gpu_sys_us.load()) * 1000ull;
#else
        // No thread CPU clock (Windows): wall time, so the long-frame breakdowns below still print
        // (their "GPU thread" time is then the frame's).
        c.gpu_ns = u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                           Clock::now().time_since_epoch())
                           .count());
#endif
    }
    c.images = BbStats::images_registered.load();
    c.image_bytes = BbStats::image_upload_bytes.load();
    c.buffer_bytes = BbStats::buffer_upload_bytes.load();
    return c;
}

/// A long frame's Vulkan calls: image creation steps, VRAM, queue submissions, device memory.
/// Called at every flip (the counts since the last one).
void PrintVulkanStall(bool long_frame) {
    static u64 stall_va, stall_vf, stall_van, stall_vfn, stall_vs, stall_vsn;
    const u64 vs = BbStats::t_vk_submit.load(), vsn = BbStats::n_vk_submit.load();
    if (long_frame && vs - stall_vs > 5000000) {
        static u64 ic, ia, ib;
        const u64 c = BbStats::t_img_create.load(), a = BbStats::t_img_alloc.load(), b = BbStats::t_img_bind.load();
        std::printf("       image steps: create %.1f ms, memory %.1f ms, bind %.1f ms\n", (c - ic) / 1e6, (a - ia) / 1e6, (b - ib) / 1e6);
        u64 vu = 0, vb = 0;
        Vulkan::VmaDeviceBudget(vu, vb);
        std::printf("       VRAM: %llu of %llu MiB budget\n", (unsigned long long)(vu >> 20), (unsigned long long)(vb >> 20));
        ic = c; ia = a; ib = b;
        std::printf("       vkQueueSubmit: %.1f ms in %llu calls\n", (vs - stall_vs) / 1e6,
                    (unsigned long long)(vsn - stall_vsn));
    }
    stall_vs = vs;
    stall_vsn = vsn;
    const u64 va = BbStats::t_vk_alloc.load(), vf = BbStats::t_vk_free.load(),
              van = BbStats::n_vk_alloc.load(), vfn = BbStats::n_vk_free.load();
    if (long_frame && (va - stall_va) + (vf - stall_vf) > 5000000) {
        std::printf("       device memory calls: %.1f ms in %llu allocations, %.1f ms in %llu frees\n",
                    (va - stall_va) / 1e6, (unsigned long long)(van - stall_van),
                    (vf - stall_vf) / 1e6, (unsigned long long)(vfn - stall_vfn));
    }
    stall_va = va;
    stall_vf = vf;
    stall_van = van;
    stall_vfn = vfn;
}

/// A long frame's layer memory work and the runtime's write traps. Called at every flip.
void PrintLayerStall(bool long_frame) {
    static std::array<u64, 12> last_layer{};
    static u64 last_mprotects = 0, last_entries = 0;
    std::array<u64, 12> layer_now{};
    for (size_t i = 0; i < layer_now.size(); ++i) {
        layer_now[i] = BbStats::t_layer[i].load(std::memory_order_relaxed);
    }
    const u64 mprotects = BbStats::layer_mprotects.load(std::memory_order_relaxed);
    const u64 entries = BbStats::layer_entries_written.load(std::memory_order_relaxed);
    if (long_frame && layer_now != last_layer) {
        const auto ms = [&](size_t i) { return (layer_now[i] - last_layer[i]) / 1e6; };
        std::printf("       layer memory ms: traps set/cleared %.1f (%llu mprotect), "
                    "to VRAM %.1f, back in place %.1f, trap hits %.1f, yield %.1f, "
                    "volatile %.1f, republish %.1f (%llu entries: staging %.1f, guest view %.1f, "
                    "mirrors %.1f, copy %.1f), paged uploads %.1f\n",
                    ms(0), static_cast<unsigned long long>(mprotects - last_mprotects), ms(1),
                    ms(2), ms(3), ms(4), ms(5), ms(6),
                    static_cast<unsigned long long>(entries - last_entries), ms(8), ms(9),
                    ms(10), ms(11), ms(7));
    }
    // The runtime's write traps (every caller: this thread's mirrors and images, the game's
    // threads' fault handlers): time waiting for its locks against time protecting, and how much.
    static std::array<uint64_t, 4> last_trap{};
    std::array<uint64_t, 4> trap_now{};
    runtime_memory_trap_stats(&trap_now[0], &trap_now[1], &trap_now[2], &trap_now[3]);
    if (long_frame && trap_now[2] != last_trap[2]) {
        std::printf("       write traps (all threads): %llu calls, lock wait %.1f ms, protecting %.1f ms, %.1f MiB changed\n",
                    static_cast<unsigned long long>(trap_now[2] - last_trap[2]),
                    (trap_now[0] - last_trap[0]) / 1e6, (trap_now[1] - last_trap[1]) / 1e6,
                    (trap_now[3] - last_trap[3]) / 1048576.0);
    }
    last_trap = trap_now;
#ifdef _WIN32
    // How many VirtualProtect calls the mprotects above became (regions of one allocation join).
    static unsigned long long last_win_protects = 0;
    const unsigned long long win_protects = bb_win32_protect_calls();
    if (long_frame && win_protects != last_win_protects) {
        std::printf("       Windows: %llu VirtualProtect calls\n", win_protects - last_win_protects);
    }
    last_win_protects = win_protects;
#endif
    last_layer = layer_now;
    last_mprotects = mprotects;
    last_entries = entries;
}

/// What happened during a long frame (over 40 ms), from the counters and the work tallies.
void PrintStall(double frame_ms, const FlipCounters& c, const FlipCounters& last) {
    const bool long_frame = frame_ms > 40.0 && last.gpu_ns != 0;
    if (long_frame) {
        std::printf("       fault handlers: %llu read faults %.1f ms, write faults %.1f ms\n",
                    static_cast<unsigned long long>(c.read_faults - last.read_faults),
                    (c.read_fault_ns - last.read_fault_ns) / 1e6,
                    (c.write_fault_ns - last.write_fault_ns) / 1e6);
    }
    PrintVulkanStall(long_frame);
    if (long_frame) {
        std::printf("Stall: %.1f ms frame; GPU thread on CPU %.1f ms; %llu images registered, "
                    "%.1f MB image uploads, %.1f MB buffer uploads\n",
                    frame_ms, (c.gpu_ns - last.gpu_ns) / 1e6,
                    static_cast<unsigned long long>(c.images - last.images),
                    (c.image_bytes - last.image_bytes) / 1e6,
                    (c.buffer_bytes - last.buffer_bytes) / 1e6);
        std::printf("       %llu draws, %llu dispatches, %llu submissions; GPU thread user %.1f ms, "
                    "kernel %.1f ms, %llu preempted, %llu waits\n",
                    static_cast<unsigned long long>(c.draws - last.draws),
                    static_cast<unsigned long long>(c.dispatches - last.dispatches),
                    static_cast<unsigned long long>(c.submissions - last.submissions),
                    (c.user_us - last.user_us) / 1e3, (c.sys_us - last.sys_us) / 1e3,
                    static_cast<unsigned long long>(c.preempted - last.preempted),
                    static_cast<unsigned long long>(c.waits - last.waits));
        std::printf("       ms in: resident %.1f, protect %.1f, image create %.1f, "
                    "image refresh %.1f, staging %.1f, waiting for host copies %.1f\n",
                    (c.t[0] - last.t[0]) / 1e6, (c.t[1] - last.t[1]) / 1e6,
                    (c.t[2] - last.t[2]) / 1e6, (c.t[3] - last.t[3]) / 1e6,
                    (c.t[4] - last.t[4]) / 1e6, (c.t[5] - last.t[5]) / 1e6);
    }
    PrintLayerStall(long_frame);
    if (long_frame) {
        std::printf("       GPU thread page faults %llu (process %llu), protection faults %llu; "
                    "protect calls %llu (%llu pages), of which write-revoking %llu (%llu pages)\n",
                    static_cast<unsigned long long>(c.minor_faults - last.minor_faults),
                    static_cast<unsigned long long>(c.process_faults - last.process_faults),
                    static_cast<unsigned long long>(c.signal_faults - last.signal_faults),
                    static_cast<unsigned long long>(c.protect_calls - last.protect_calls),
                    static_cast<unsigned long long>(c.protect_pages - last.protect_pages),
                    static_cast<unsigned long long>(c.revoke_calls - last.revoke_calls),
                    static_cast<unsigned long long>(c.revoke_pages - last.revoke_pages));
    }
    if (long_frame && c.copy_ns > last.copy_ns) {
        std::printf("       guest copies %.1f MB in %.1f thread-ms, %.1f ms on CPU (kernel %.1f ms, "
                    "%llu page faults in large copies) (%.2f GB/s per thread)\n",
                    (c.copy_bytes - last.copy_bytes) / 1e6, (c.copy_ns - last.copy_ns) / 1e6,
                    (c.copy_cpu - last.copy_cpu) / 1e6, (c.copy_sys - last.copy_sys) / 1e3,
                    static_cast<unsigned long long>(c.copy_faults - last.copy_faults),
                    double(c.copy_bytes - last.copy_bytes) / double(c.copy_ns - last.copy_ns));
    }
}

/// bbport: BB_FRAME_LOG=path (with BB_FRAME_STATS): one line per flip, to find what the long
/// frames (stutter) have in common.
void WriteFrameLog(Clock::time_point now, double frame_ms, const FlipCounters& c,
                   const FlipCounters& last) {
    static FILE* const frame_log = [] () -> FILE* {
        const char* path = bb_env("BB_FRAME_LOG");
        FILE* file = path && *path ? std::fopen(path, "w") : nullptr;
        if (file) {
            std::fprintf(file, "t_s,frame_ms,present_wait_ms,tick_wait_ms,stage_a_cpu_ms,"
                               "images,image_mb,buffer_mb,protect_ms,resident_ms,create_ms,"
                               "refresh_ms,write_fault_ms,read_faults,guest_copy_mb,draws,"
                               "dispatches,submissions,compiles,kernel_ms,alloc_mb,"
                               "free_mb,stage_a_ms,stage_b_ms,vk_alloc_ms,vk_alloc_n,vk_free_ms,vk_free_n\n");
        }
        return file;
    }();
    if (!frame_log) {
        return;
    }
    static const auto log_start = now;
    static u64 log_twf, log_tick, log_present, log_compiles, log_rf;
    static u64 log_alloc, log_free;
    static u64 log_stage_a, log_stage_b;
    static u64 log_va, log_vf, log_van, log_vfn;
    // bbport: the stages count BbCpu::Ticks (arm64's counter in bb-gpu, not the game's TSC).
    static const double stage_ms =
        1e3 / double(std::max<u64>(BbCpu::CounterFrequency() ? BbCpu::CounterFrequency() : runtime_tsc_frequency(), 1));
    const u64 tick = BbStats::tick_wait_ns.load(), present = BbStats::present_wait_ns.load();
    const u64 compiles_now = Vulkan::g_bb_compiles.load();
    // tick_wait_ns and the compile count are reset every stats window.
    const auto delta = [](u64 now_value, u64& last_value) {
        const u64 d = now_value >= last_value ? now_value - last_value : now_value;
        last_value = now_value;
        return d;
    };
    std::fprintf(frame_log,
                 "%.4f,%.2f,%.2f,%.2f,%.2f,%llu,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%llu,"
                 "%.2f,%llu,%llu,%llu,%llu,%.2f,%.1f,%.1f,%.2f,%.2f,%.2f,%llu,%.2f,%llu\n",
                 std::chrono::duration<double>(now - log_start).count(), frame_ms,
                 delta(present, log_present) / 1e6, delta(tick, log_tick) / 1e6,
                 last.gpu_ns ? (c.gpu_ns - last.gpu_ns) / 1e6 : 0.0,
                 static_cast<unsigned long long>(c.images - last.images),
                 (c.image_bytes - last.image_bytes) / 1e6,
                 (c.buffer_bytes - last.buffer_bytes) / 1e6, (c.t[1] - last.t[1]) / 1e6,
                 (c.t[0] - last.t[0]) / 1e6, (c.t[2] - last.t[2]) / 1e6,
                 (c.t[3] - last.t[3]) / 1e6, delta(c.write_fault_ns, log_twf) / 1e6,
                 static_cast<unsigned long long>(delta(c.read_faults, log_rf)),
                 (c.copy_bytes - last.copy_bytes) / 1e6,
                 static_cast<unsigned long long>(c.draws - last.draws),
                 static_cast<unsigned long long>(c.dispatches - last.dispatches),
                 static_cast<unsigned long long>(c.submissions - last.submissions),
                 static_cast<unsigned long long>(delta(compiles_now, log_compiles)),
                 last.gpu_ns ? (c.sys_us - last.sys_us) / 1e3 : 0.0,
                 delta(BbStats::device_alloc_bytes.load(), log_alloc) / 1e6,
                 delta(BbStats::device_free_bytes.load(), log_free) / 1e6,
                 // BB_SECTIONS=1: the decoding (stage A) and recording (stage B) time of the
                 // frame's work, from the section timers (0 without them).
                 delta(BbSections::cycles[BbSections::ADecode].load(), log_stage_a) * stage_ms,
                 delta(BbSections::cycles[BbSections::Packet].load(), log_stage_b) * stage_ms,
                 delta(BbStats::t_vk_alloc.load(), log_va) / 1e6,
                 static_cast<unsigned long long>(delta(BbStats::n_vk_alloc.load(), log_van)),
                 delta(BbStats::t_vk_free.load(), log_vf) / 1e6,
                 static_cast<unsigned long long>(delta(BbStats::n_vk_free.load(), log_vfn)));
    static u32 unflushed = 0;
    if (++unflushed >= 120) {
        std::fflush(frame_log);
        unflushed = 0;
    }
}

/// The window's first line: frame rate, compiles, the GPU thread's cost and what it waited for.
void PrintFrameLine(double window, u32 frames, double worst_ms) {
    const u32 compiles = Vulkan::g_bb_compiles.exchange(0);
    const u64 compile_ns = Vulkan::g_bb_compile_ns.exchange(0);
    const u64 direct = Vulkan::Scheduler::direct_recordings.exchange(0);
    const u64 faults = BbStats::tracker_faults.exchange(0);
    // GPU command thread CPU time per draw: comparable between builds even when the scene
    // (and so the frame rate) differs a little.
    static u64 window_draws = 0, window_gpu_us = 0;
    const u64 all_draws = BbStats::draws.load();
    const u64 gpu_us = BbStats::gpu_user_us.load() + BbStats::gpu_sys_us.load();
    const double us_per_draw = all_draws > window_draws
                                   ? double(gpu_us - window_gpu_us) / (all_draws - window_draws)
                                   : 0.0;
    const double draws_per_frame = frames ? double(all_draws - window_draws) / frames : 0.0;
    window_draws = all_draws;
    window_gpu_us = gpu_us;
    std::printf("Frame stats: %.1f FPS, worst frame %.1f ms (vblank %u Hz); "
                "%u shader/pipeline compiles, %.1f ms; %llu recorder syncs; "
                "%.0f write faults/s, %lld hot pages; GPU thread %.2f us/draw, "
                "%.0f draws/frame, idle %.1f%%; blocked: recorder %.1f%%, host copies "
                "%.1f%% (%.0f/frame, %.0f/frame not needed), copy threads %.1f%%, GPU ticks %.1f%%, "
                "presenter frames %.1f%%, frames ahead %.1f%%, present calls %.1f%%; "
                "reduced-size draws %.0f/frame of %.0f in the scene\n",
                frames / window, worst_ms, EmulatorSettings.GetVblankFrequency(), compiles,
                compile_ns / 1e6, static_cast<unsigned long long>(direct),
                faults / window, static_cast<long long>(BbStats::hot_pages.load()),
                us_per_draw, draws_per_frame,
                BbStats::gpu_idle_ns.exchange(0) / (window * 1e7),
                BbStats::sync_recording_ns.exchange(0) / (window * 1e7),
                BbStats::host_copies_wait_ns.exchange(0) / (window * 1e7),
                frames ? double(BbStats::host_copy_waits.exchange(0)) / frames : 0.0,
                frames ? double(BbStats::host_copy_waits_skipped.exchange(0)) / frames : 0.0,
                BbStats::copy_threads_wait_ns.exchange(0) / (window * 1e7),
                BbStats::tick_wait_ns.exchange(0) / (window * 1e7),
                BbStats::frame_pool_wait_ns.exchange(0) / (window * 1e7),
                [&] {
                    static u64 last = 0;
                    const u64 total = BbStats::present_wait_ns.load();
                    const u64 d = total - last;
                    last = total;
                    return d / (window * 1e7);
                }(),
                BbStats::present_call_ns.exchange(0) / (window * 1e7),
                frames ? double(BbStats::reduced_draws.exchange(0)) / frames : 0.0,
                frames ? double(BbStats::scene_draws.exchange(0)) / frames : 0.0);
}

/// The window's memory lines: CPU writes over watched pages, memory use, what the draws bound
/// and how GPU data moved.
void PrintMemoryTraffic(double window, u32 frames) {
    BbMem::FaultSites::Report();
    if (const u64 notes = BbStats::cpu_write_notes.exchange(0)) {
        std::printf("Write notices: %.0f/s, %.2f MB/s (libc copies over pages the GPU side "
                    "watches)\n",
                    notes / window, BbStats::cpu_write_note_bytes.exchange(0) / (window * 1e6));
    }
    if (const u64 fast = BbStats::fast_writer_faults.exchange(0)) {
        std::printf("Write faults of the game's bookkeeping without a pipeline wait: %.0f/s\n",
                    fast / window);
    }
    Vulkan::Breadcrumbs::PrintProgress();
    PrintMemory();
    if (const u64 draws_n = BbStats::geometry_draws.exchange(0)) {
        const double f = 1e6 * double(std::max<u64>(frames, 1));
        const u64 v0 = BbStats::vertex_bytes[0].exchange(0), v1 = BbStats::vertex_bytes[1].exchange(0);
        const u64 i0 = BbStats::index_bytes[0].exchange(0), i1 = BbStats::index_bytes[1].exchange(0);
        std::printf("Geometry bound per frame (%llu indexed draws): vertex %.2f MB in VRAM, %.2f MB "
                    "in place (%.0f %%); index %.2f MB in VRAM, %.2f MB in place (%.0f %%)\n",
                    (unsigned long long)(draws_n / std::max<u64>(frames, 1)), v1 / f, v0 / f,
                    100.0 * double(v0) / double(std::max<u64>(v0 + v1, 1)), i1 / f, i0 / f,
                    100.0 * double(i0) / double(std::max<u64>(i0 + i1, 1)));
    }
    if (const u64 in_place = BbStats::bound_in_place_bytes.exchange(0)) {
        std::printf("Buffer bindings: %.1f MB/frame in place (%.1f GPU-written), %.1f MB/frame "
                    "VRAM copies\n",
                    in_place / (frames * 1e6),
                    BbStats::bound_in_place_written_bytes.exchange(0) / (frames * 1e6),
                    BbStats::bound_vram_bytes.exchange(0) / (frames * 1e6));
    }
    const u64 updates = BbStats::gpu_data_updates.exchange(0);
    const u64 copied_back = BbStats::copied_back_blocks.exchange(0);
    const u64 readbacks = BbStats::readbacks.exchange(0);
    const u64 late = BbStats::late_vram_writes.exchange(0);
    if (updates + copied_back + readbacks + late != 0) {
        std::printf("GPU data in VRAM: %.1f command writes/s put into it, %.1f exact writes/s "
                    "(labels, CPU bytes beside GPU data), %.1f blocks/s copied back, "
                    "%.1f readbacks/s\n",
                    updates / window, late / window, copied_back / window,
                    readbacks / window);
    }
}

/// The window's lines on the game's side: its waits, heaps, Gnm submissions and fences, sleeps
/// and the guest memory the GPU side hands out.
void PrintGuestSide(double window, u32 frames) {
    BbKernel::ReportEqueueWaits(frames);
    if (static int heap_reports = 0; ++heap_reports % 6 == 1) {
        BbHeapSites::Report();
    }
    runtime_wait_report(frames);
    for (std::size_t i = 0; i < BbStats::range_allocators.size(); ++i) {
        if (const u64 a = BbStats::range_allocators[i].load()) {
            std::printf("%s %#llx: %llu live, %.1f MB", i ? ";" : "GPU range allocators:",
                        (unsigned long long)a, (unsigned long long)BbStats::range_live[i].load(),
                        BbStats::range_bytes[i].load() / 1e6);
        }
    }
    if (BbStats::range_allocators[0].load()) {
        std::printf("\n");
    }
    // bbport (native GPU process): the guest's Gnm calls are counted in the game process
    // (MsgGameStats prints them there).
    if (!BbRemote::BackActive()) {
        std::printf("Gnm: %.1f sceGnmSubmitDone/frame, %.1f sceGnmAreSubmitsAllowed/frame, %.1f refused/frame; "
                    "guest waited for the previous frame %.1fx %.2f ms/frame; %.1f compute queue submissions/frame\n",
                    BbStats::submit_done_calls.exchange(0) / double(frames),
                    BbStats::submits_allowed_queries.exchange(0) / double(frames),
                    BbStats::submits_refused.exchange(0) / double(frames),
                    BbStats::gnm_frame_waits.exchange(0) / double(frames),
                    BbStats::gnm_frame_wait_ns.exchange(0) / (1e6 * double(frames)),
                    BbStats::asc_submits.exchange(0) / double(frames));
    } else {
        std::printf("Gnm: %.1f compute queue submissions/frame\n",
                    BbStats::asc_submits.exchange(0) / double(frames));
    }
    std::printf("EOP fences: %llu decoded, %llu labels written, %lld pending; %.1f labels/frame "
                "written by the GPU\n",
                (unsigned long long)BbStats::eop_decoded.load(),
                (unsigned long long)BbStats::eop_written.load(),
                (long long)(BbStats::eop_decoded.load() - BbStats::eop_written.load()),
                BbStats::gpu_labels.exchange(0) / double(frames));
    const u64 early = BbStats::idle_flushes.exchange(0);
    const u64 draining = BbStats::drain_flushes.exchange(0);
    if (early || draining) {
        std::printf("Honest labels: %.1f submissions/frame sent early (GPU idle), %.1f as the "
                    "GPU reached the last one\n",
                    early / double(frames), draining / double(frames));
    }
    uint64_t sleeps = 0, sleep_ns = 0;
    runtime_sleep_stats(&sleeps, &sleep_ns);
    if (sleeps != 0 && frames != 0) {
        std::printf("Guest sleeps per frame: %.1f, %.2f ms\n", sleeps / double(frames),
                    sleep_ns / (frames * 1e6));
    }
    if (const u64 copies = BbStats::shadow_copies.exchange(0)) {
        std::printf("Shadow copies: %.0f/frame, %.1f MB/frame (in-place data the CPU writes, "
                    "copied into VRAM for reading)\n",
                    copies / double(frames), BbStats::shadow_bytes.exchange(0) / (frames * 1e6));
    }
    if (const u64 copies = BbStats::write_shadow_copies.exchange(0)) {
        std::printf("Write shadows: %.0f/frame, %.1f KB/frame, %.0f bindings/frame read them, %.0f "
                    "small in-place reads without one (in-place GPU writes copied into VRAM after "
                    "their dispatch)\n",
                    copies / double(frames), BbStats::write_shadow_bytes.exchange(0) / (frames * 1e3),
                    BbStats::write_shadow_hits.exchange(0) / double(frames),
                    BbStats::write_shadow_misses.exchange(0) / double(frames));
    }
    if (const u64 allocs = BbStats::gpu_range_allocs.exchange(0)) {
        std::printf("Guest GPU memory: %.0f ranges/s handed out, %.2f MB/s\n", allocs / window,
                    BbStats::gpu_range_alloc_bytes.exchange(0) / (window * 1e6));
    }
}

/// Frame pacing: spread of the guest flip intervals (judder that the mean hides).
void PrintPacing(const std::vector<double>& intervals) {
    if (intervals.size() <= 2) {
        return;
    }
    std::vector<double> sorted = intervals;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    double sum = 0, sq = 0;
    u32 spikes = 0;
    for (const double ms : intervals) {
        sum += ms;
        sq += ms * ms;
        spikes += ms > 1.5 * median;
    }
    const double mean = sum / intervals.size();
    std::printf("Frame pacing: median %.2f ms, stddev %.2f ms, p99 %.2f ms, "
                "%u frames over 1.5x median\n",
                median, std::sqrt(std::max(0.0, sq / intervals.size() - mean * mean)),
                sorted[std::min(sorted.size() - 1, sorted.size() * 99 / 100)], spikes);
}

} // namespace

void OnFlipStats() {
    static Clock::time_point window_start = Clock::now(), last_flip = window_start;
    static u32 frames;
    static double worst_ms;
    static std::vector<double> intervals;
    static FlipCounters last;
    const auto now = Clock::now();
    const double frame_ms = std::chrono::duration<double, std::milli>(now - last_flip).count();
    worst_ms = std::max(worst_ms, frame_ms);
    intervals.push_back(frame_ms);
    last_flip = now;

    const FlipCounters counters = ReadFlipCounters();
    PrintStall(frame_ms, counters, last);
    WriteFrameLog(now, frame_ms, counters, last);
    last = counters;

    ++frames;
    const double window = std::chrono::duration<double>(now - window_start).count();
    if (window < 5.0) {
        return;
    }
    PrintFrameLine(window, frames, worst_ms);
    PrintMemoryTraffic(window, frames);
    PrintGuestSide(window, frames);
    // bbport: parallel Vulkan recording (vk_scheduler.h): command buffer segments.
    if (const u64 submits = Vulkan::Scheduler::recorded_submissions.exchange(0)) {
        const u64 segments = Vulkan::Scheduler::recorded_segments.exchange(0);
        std::printf("Recording: %.1f submissions/frame, %.2f command buffers each\n",
                    frames ? double(submits) / frames : 0.0, double(segments) / submits);
    }
    BbWaitTrace::Report(window);
    PrintPacing(intervals);
#ifdef __APPLE__
    // bbport (macOS): CPU time of every thread, by name and by owner (bbport_thread_cpu_mac).
    if (const std::string threads = BbThreads::ReportProcessThreads(window); !threads.empty()) {
        std::printf("%s%s\n", BbRemote::BackActive() ? "GPU process: " : "", threads.c_str());
    }
#endif
    if (BbRemote::BackActive()) {
        BbRemote::Back::GameStats(window, frames);
    }
    intervals.clear();
    window_start = now;
    frames = 0;
    worst_ms = 0;
}

} // namespace BbVideoOut
