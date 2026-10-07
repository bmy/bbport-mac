// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): per-thread CPU time for BB_FRAME_STATS, from Mach thread_info.
#ifdef __APPLE__
#include <mach/mach.h>
#include <pthread.h>
#include "bbport_threads.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

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

namespace {
struct ThreadSample {
    std::uint64_t cpu_ns = 0;
    std::string name;
};
std::unordered_map<std::uint64_t, ThreadSample> g_previous;

/// Groups numbered threads (bb:Copy0..3, GXWorker:1..4) under one name.
std::string GroupName(const char* raw) {
    std::string name = raw && raw[0] ? raw : "(unnamed)";
    while (!name.empty() && (std::isdigit(static_cast<unsigned char>(name.back())) ||
                             name.back() == ':' || name.back() == '_')) {
        name.pop_back();
    }
    return name.empty() ? std::string{raw} : name;
}

/// Who runs a thread: this port's GPU side, the system (Metal, Rosetta, AppKit), or the game.
const char* Owner(const std::string& name) {
    if (name.rfind("bb:", 0) == 0 || name.rfind("shadPS4", 0) == 0 || name.rfind("bb-prob", 0) == 0) {
        return "port";
    }
    if (name.rfind("com.apple", 0) == 0 || name.rfind("(unnamed)", 0) == 0 ||
        name.find("Metal") != std::string::npos || name.rfind("Dispatch", 0) == 0) {
        return "system";
    }
    return "game";
}
} // namespace

std::string ReportProcessThreads(double window_s) {
    thread_act_array_t threads = nullptr;
    mach_msg_type_number_t count = 0;
    if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS || window_s <= 0) {
        return {};
    }
    std::unordered_map<std::uint64_t, ThreadSample> current;
    std::map<std::string, std::pair<double, int>> groups; // CPU % of one core, threads
    std::map<std::string, double> owners;
    for (mach_msg_type_number_t i = 0; i < count; ++i) {
        thread_extended_info_data_t info{};
        mach_msg_type_number_t info_count = THREAD_EXTENDED_INFO_COUNT;
        thread_identifier_info_data_t id{};
        mach_msg_type_number_t id_count = THREAD_IDENTIFIER_INFO_COUNT;
        if (thread_info(threads[i], THREAD_EXTENDED_INFO, reinterpret_cast<thread_info_t>(&info),
                        &info_count) == KERN_SUCCESS &&
            thread_info(threads[i], THREAD_IDENTIFIER_INFO, reinterpret_cast<thread_info_t>(&id),
                        &id_count) == KERN_SUCCESS) {
            ThreadSample sample{info.pth_user_time + info.pth_system_time, GroupName(info.pth_name)};
            const auto previous = g_previous.find(id.thread_id);
            const std::uint64_t before = previous != g_previous.end() ? previous->second.cpu_ns : 0;
            const double percent = double(sample.cpu_ns - std::min(before, sample.cpu_ns)) /
                                   (window_s * 1e7);
            auto& group = groups[sample.name];
            group.first += percent;
            ++group.second;
            owners[Owner(sample.name)] += percent;
            current.emplace(id.thread_id, std::move(sample));
        }
        mach_port_deallocate(mach_task_self(), threads[i]);
    }
    vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads),
                  count * sizeof(thread_act_t));
    const bool first = g_previous.empty();
    g_previous = std::move(current);
    if (first) {
        return {};
    }
    std::vector<std::pair<std::string, std::pair<double, int>>> sorted(groups.begin(), groups.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
    char text[256];
    std::snprintf(text, sizeof(text), "CPU by thread (%% of one core): port %.0f, game %.0f, system %.0f;",
                  owners["port"], owners["game"], owners["system"]);
    std::string out = text;
    for (std::size_t i = 0; i < sorted.size() && i < 16; ++i) {
        if (sorted[i].second.first < 1.0) {
            break;
        }
        std::snprintf(text, sizeof(text), " %s%s %.0f", sorted[i].first.c_str(),
                      sorted[i].second.second > 1
                          ? ("x" + std::to_string(sorted[i].second.second)).c_str()
                          : "",
                      sorted[i].second.first);
        out += text;
        out += i + 1 < sorted.size() && i < 15 ? "," : "";
    }
    return out;
}
} // namespace BbThreads
#endif
