// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: helper thread sizing. Counts follow the hardware threads this process may run on
// (the affinity mask, so `taskset` can emulate a Steam Deck), and speculative helpers run as
// SCHED_IDLE: they use cores the game leaves idle and never take time from its threads.

#pragma once

#include <algorithm>
#include <cstdint>
#include <sched.h>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
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

/// Hardware threads available to the process.
inline unsigned Available() {
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        return std::max(1, CPU_COUNT(&set));
    }
#endif
    return std::max(1u, std::thread::hardware_concurrency());
}

/// The calling thread only runs on otherwise idle cores (falls back to the lowest nice level).
inline void MakeBackground() {
#ifdef __APPLE__
    // No SCHED_IDLE on macOS: the utility QoS class keeps helpers below the game's threads.
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#else
    sched_param param{};
    if (sched_setscheduler(0, SCHED_IDLE, &param) != 0) {
        setpriority(PRIO_PROCESS, static_cast<id_t>(gettid()), 19);
    }
#endif
}

} // namespace BbThreads
