// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Mac's memory as a whole (platform/macos/memory_macos.h).
#include "platform/macos/memory_macos.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <dispatch/dispatch.h>
#include <libproc.h>
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "remote/bb_remote.h"

namespace BbMacMemory {

namespace {
constexpr u64 GiB = u64{1} << 30;
std::atomic<unsigned long> pressure_events{0};

void OnPressure(void* context) {
    // A dispatch queue's thread: an atomic only (the renderer polls it, TakePressure).
    const auto source = static_cast<dispatch_source_t>(context);
    pressure_events.fetch_or(dispatch_source_get_data(source), std::memory_order_relaxed);
}

template <typename T>
bool Sysctl(const char* name, T& value) {
    size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && size == sizeof(value);
}
} // namespace

u64 PhysicalBytes() {
    static const u64 bytes = [] {
        u64 value = 0;
        return Sysctl("hw.memsize", value) ? value : 0;
    }();
    return bytes;
}

bool SmallMac() {
    const u64 ram = PhysicalBytes();
    return ram != 0 && ram <= 16 * GiB;
}

bool MachineBudget() {
    static const bool on = [] {
        const char* env = std::getenv("BB_MEM_BUDGET");
        return !(env && std::strcmp(env, "metal") == 0);
    }();
    return on;
}

u64 OSReserveBytes() {
    static const u64 reserve = [] {
        const char* env = std::getenv("BB_MEM_RESERVE_MB");
        if (env && *env) {
            // At most the RAM (a negative or huge value would wrap).
            const long long mb = std::strtoll(env, nullptr, 10);
            return std::min<u64>(u64(std::max(0LL, mb)), PhysicalBytes() >> 20) << 20;
        }
        return std::max<u64>(GiB * 5 / 2, PhysicalBytes() / 5);
    }();
    return reserve;
}

bool OwnFootprint(u64& footprint, u64& graphics) {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) !=
            KERN_SUCCESS ||
        count < TASK_VM_INFO_REV3_COUNT) {
        return false;
    }
    footprint = info.phys_footprint;
    const s64 graphics_signed =
        info.ledger_tag_graphics_footprint + info.ledger_tag_graphics_footprint_compressed;
    graphics = graphics_signed > 0 ? u64(graphics_signed) : 0;
    return true;
}

u64 GameProcessFootprint() {
    if (!BbRemote::BackActive()) {
        return 0; // one process: its own footprint holds the game
    }
    const pid_t parent = getppid();
    if (parent <= 1) {
        return 0; // the game process is gone
    }
    rusage_info_v4 info{};
    if (proc_pid_rusage(parent, RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t*>(&info)) != 0) {
        return 0;
    }
    return info.ri_phys_footprint;
}

System ReadSystem() {
    System system;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) {
        page = vm_page_size;
    }
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm),
                          &count) == KERN_SUCCESS) {
        system.free = u64(vm.free_count) * page;
        system.compressed = u64(vm.compressor_page_count) * page;
        system.compressed_data = u64(vm.total_uncompressed_pages_in_compressor) * page;
    }
    xsw_usage swap{};
    if (Sysctl("vm.swapusage", swap)) {
        system.swap_used = swap.xsu_used;
        system.swap_total = swap.xsu_total;
    }
    int level = 0;
    if (Sysctl("kern.memorystatus_vm_pressure_level", level)) {
        system.pressure_level = level;
    }
    if (Sysctl("kern.memorystatus_level", level)) {
        system.memorystatus_level = level;
    }
    return system;
}

void WatchPressure() {
    static const bool started = [] {
        const char* env = std::getenv("BB_MEM_PRESSURE");
        if (env && env[0] == '0') {
            std::printf("GPU (macOS): memory pressure notifications ignored (BB_MEM_PRESSURE=0)\n");
            return false;
        }
        dispatch_source_t source = dispatch_source_create(
            DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
            DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
            dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
        if (!source) {
            return false;
        }
        // Kept for the life of the process.
        dispatch_set_context(source, source);
        dispatch_source_set_event_handler_f(source, OnPressure);
        dispatch_resume(source);
        return true;
    }();
    (void)started;
}

unsigned long TakePressure() {
    if (pressure_events.load(std::memory_order_relaxed) == 0) {
        return 0;
    }
    return pressure_events.exchange(0, std::memory_order_relaxed);
}

} // namespace BbMacMemory
