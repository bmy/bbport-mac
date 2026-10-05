// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): per-thread CPU time for BB_FRAME_STATS, from Mach thread_info.
#ifdef __APPLE__
#include <mach/mach.h>
#include <pthread.h>
#include "bbport_threads.h"

namespace BbThreads {
bool CurrentThreadCpuUs(std::uint64_t& user_us, std::uint64_t& sys_us) {
    thread_basic_info_data_t info{};
    mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
    // pthread_mach_thread_np does not add a port reference (mach_thread_self would).
    if (thread_info(pthread_mach_thread_np(pthread_self()), THREAD_BASIC_INFO,
                    reinterpret_cast<thread_info_t>(&info), &count) != KERN_SUCCESS) {
        return false;
    }
    user_us = std::uint64_t(info.user_time.seconds) * 1000000 + std::uint64_t(info.user_time.microseconds);
    sys_us = std::uint64_t(info.system_time.seconds) * 1000000 + std::uint64_t(info.system_time.microseconds);
    return true;
}
} // namespace BbThreads
#endif
