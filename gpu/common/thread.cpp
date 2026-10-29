// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <cerrno>
#include <cstring>
#include <ctime>

#include <pthread.h>
#include <sched.h>
#ifdef __APPLE__
#include <cstdlib>
#include <pthread/qos.h>
#endif

#include "common/logging/log.h"
#include "common/thread.h"
#ifdef _WIN32
extern "C" void bb_win32_set_thread_name(const char* name); // posix_win32.cpp
#endif

namespace Common {

#ifdef __APPLE__
namespace {
/// bbport BB_GPU_QOS=1 (macOS experiment): the threads every frame waits on run as
/// user-interactive work, so macOS keeps them on performance cores and ahead of other work.
void MaybeInteractive(const char* name) {
    static const bool interactive = [] {
        const char* env = std::getenv("BB_GPU_QOS");
        return env && env[0] == '1';
    }();
    if (!interactive) {
        return;
    }
    static constexpr const char* critical[] = {
        "bb:GpuComm", "bb:VkRecorder", "bb:DrawRec", "bb:DrawPrep",
        "bb:DrawScan",    "bb:Present",    "bb:remote",
    };
    for (const char* prefix : critical) {
        if (std::strncmp(name, prefix, std::strlen(prefix)) == 0) {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            return;
        }
    }
}
} // namespace
#endif

void SetCurrentThreadName(const char* name) {
#ifdef __APPLE__
    // bbport (macOS): a thread can only name itself.
    pthread_setname_np(name);
    MaybeInteractive(name);
#else
#ifdef _WIN32
    bb_win32_set_thread_name(name);
#endif
    SetThreadName(reinterpret_cast<void*>(pthread_self()), name);
#endif
}

void SetThreadName(void* thread, const char* name) {
#ifdef __APPLE__
    // bbport (macOS): only the calling thread can be named.
    if (reinterpret_cast<pthread_t>(thread) == pthread_self()) {
        SetCurrentThreadName(name);
    }
#else
    // Linux takes at most 15 characters (and refuses longer names outright).
    char truncated[16];
    std::strncpy(truncated, name, sizeof(truncated) - 1);
    truncated[sizeof(truncated) - 1] = '\0';
    pthread_setname_np(reinterpret_cast<pthread_t>(thread), truncated);
#endif
}

std::string GetCurrentThreadName() {
    char name[64];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return name;
}

void SetCurrentThreadPriority(ThreadPriority priority) {
    // SCHED_OTHER has one static priority; the niceness of a thread is what can change, and
    // only downwards without privileges. Low priority work gets a higher nice value.
    if (priority == ThreadPriority::Low) {
        sched_param param{};
        pthread_setschedparam(pthread_self(), SCHED_OTHER, &param);
    }
}

void SetCurrentThreadRealtime(std::chrono::nanoseconds) {}

bool AccurateSleep(std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   bool interruptible) {
    timespec request{.tv_sec = time_t(duration.count() / 1'000'000'000),
                     .tv_nsec = long(duration.count() % 1'000'000'000)};
    timespec left{};
    int result;
#ifdef _WIN32
    // MinGW's nanosleep is Sleep(whole ms): Sleep(0), a spin, under one (bbport_win32.h has the
    // high-resolution clock_nanosleep).
    (void)interruptible;
    result = clock_nanosleep(CLOCK_MONOTONIC, 0, &request, &left);
#else
    while ((result = nanosleep(&request, &left)) != 0 && errno == EINTR && !interruptible) {
        request = left;
    }
#endif
    if (remaining) {
        *remaining = result == 0 ? std::chrono::nanoseconds{0}
                                 : std::chrono::seconds{left.tv_sec} +
                                       std::chrono::nanoseconds{left.tv_nsec};
    }
    return result == 0;
}

void SleepUntil(std::chrono::steady_clock::time_point deadline) {
    const auto left = deadline - std::chrono::steady_clock::now();
    if (left.count() > 0) {
        AccurateSleep(std::chrono::duration_cast<std::chrono::nanoseconds>(left), nullptr, false);
    }
}

AccurateTimer::AccurateTimer(std::chrono::nanoseconds period_) : period{period_} {}

void AccurateTimer::Start() {
    const auto now = std::chrono::steady_clock::now();
    if (deadline.time_since_epoch().count() == 0) {
        deadline = now;
    }
    if (deadline > now) {
#ifdef _WIN32
        // steady_clock and CLOCK_MONOTONIC need not share an epoch here: a relative sleep.
        SleepUntil(deadline);
        return;
#endif
        const auto since_epoch = deadline.time_since_epoch();
        const timespec at{
            .tv_sec = time_t(std::chrono::duration_cast<std::chrono::seconds>(since_epoch).count()),
            .tv_nsec = long((since_epoch % std::chrono::seconds{1}).count())};
#ifdef __APPLE__
        // bbport (macOS): no clock_nanosleep; the steady clock is monotonic, so sleep the rest.
        (void)at;
        AccurateSleep(std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now), nullptr,
                      false);
#else
        while (clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &at, nullptr) == EINTR) {
        }
#endif
    }
}

void AccurateTimer::End() {
    const auto now = std::chrono::steady_clock::now();
    deadline += period;
    if (deadline + period < now) {
        deadline = now; // far behind: start over
    }
    total_wait = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline - now);
}

} // namespace Common
