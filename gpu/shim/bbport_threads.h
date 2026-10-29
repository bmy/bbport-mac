// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: helper thread sizing. Counts follow the hardware threads this process may run on
// (the affinity mask, so `taskset` can emulate a Steam Deck), and speculative helpers run as
// SCHED_IDLE: they use cores the game leaves idle and never take time from its threads.

#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include "shim/bb_env.h"
#include <cstdlib>
#include <thread>
#ifndef _WIN32
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#endif
#ifdef __APPLE__
#include <pthread.h>
#include <pthread/qos.h>
#endif

namespace BbThreads {

/// Kernel thread id of the calling thread (diagnostics, write log, draw pipe).
inline unsigned long HostTid() {
#ifdef __APPLE__
    std::uint64_t id = 0;
    pthread_threadid_np(nullptr, &id);
    return static_cast<unsigned long>(id);
#else
    return static_cast<unsigned long>(gettid());
#endif
}

/// The game's main thread: the process's first thread on Linux. bbport (macOS): the loader's
/// secondary thread (Cocoa keeps the real main thread), recorded by bbgpu_init, which runs on it.
inline std::atomic<unsigned long> guest_main_tid{0};
inline bool IsGuestMainThread() {
#ifdef __APPLE__
    return HostTid() == guest_main_tid.load(std::memory_order_relaxed);
#else
    return gettid() == getpid();
#endif
}

/// Hardware threads available to the process.
inline unsigned Available() {
#ifdef _WIN32
    return bb_win32_available_cpus();
#elif defined(__APPLE__)
    // bbport (macOS): no affinity masks; every core is available.
    return std::max(1u, std::thread::hardware_concurrency());
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        return std::max(1, CPU_COUNT(&set));
    }
    return std::max(1u, std::thread::hardware_concurrency());
#endif
}

/// The calling thread only runs on otherwise idle cores (falls back to the lowest nice level).
inline void MakeBackground() {
#ifdef _WIN32
    bb_win32_background_thread();
#elif defined(__APPLE__)
    // bbport (macOS): no SCHED_IDLE; the utility QoS class keeps helpers below the game's threads.
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#else
    sched_param param{};
    if (sched_setscheduler(0, SCHED_IDLE, &param) != 0) {
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 19);
    }
#endif
}

/// The draw pipeline's critical threads (GPU command thread, draw recording, Vulkan recording):
/// ahead of the game's own threads for the CPU. All threads ran at one priority (the guest's are
/// not mapped), so on 4 cores the stages took turns with a dozen game threads and some frames
/// recorded at half speed. BB_STAGE_PRIORITY=0 off (default), 1 above normal, 2 highest
/// (Windows; on Linux a lower nice value where the process may set one). GTX 1050 laptop, 4 cores,
/// 1080p: 31.6-31.9 -> 32.5-32.8 FPS, frames over 50 ms 16-25 -> 10-15 a minute; 2 was worse.
/// Off by default: with it the game sent command buffers it had not finished writing (a packet
/// cut short at load, "Bad PM4 packet", 2 of 10 runs vs 0 of ~25): its threads, preempted by
/// ours, race in ways the console's strict priorities never let happen (2026-10-25).
inline void MakeCritical() {
    static const int level = [] {
        const char* env = bb_env("BB_STAGE_PRIORITY");
        return env ? std::atoi(env) : 0;
    }();
    if (level <= 0) {
        return;
    }
#ifdef _WIN32
    bb_win32_critical_thread(level);
#elif defined(__APPLE__)
    // bbport (macOS): no per-thread nice value; the user-interactive QoS class instead.
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#else
    setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), level >= 2 ? -10 : -5);
#endif
}

#ifdef __APPLE__
/// CPU time of the calling thread in microseconds (macOS has no RUSAGE_THREAD).
/// Defined in bbport_thread_cpu_mac.cpp so Mach headers stay out of shared headers.
bool CurrentThreadCpuUs(std::uint64_t& user_us, std::uint64_t& sys_us);
/// BB_FRAME_STATS: CPU time of every thread of the process since the last call, by thread name
/// and by owner (this port, the game, the system): where a native GPU process would help.
std::string ReportProcessThreads(double window_s);
#endif

/// Helper threads of the GPU side that the GPU command thread's progress depends on (Vulkan
/// recording threads, guest memory copy threads): a fault there must never wait for that thread
/// (a read trap's copy back goes through it for the game's threads). Set once per thread.
inline thread_local bool gpu_side_helper = false;

} // namespace BbThreads
