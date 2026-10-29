// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport BB_READ_SAMPLE=lo-hi[,lo-hi...] (diagnostics, Linux): how much the CPU reads guest memory
// the GPU reads (whether a heap could live in VRAM through ReBAR, where CPU reads are slow). Every
// 200 ms, 16 random 64 KiB windows of the ranges that are plain read-write (no trap of another
// module on them) lose all access for 200 ms; the first access to each is counted as a read or a
// write, with the code that made it (game offset or host). Each 10 s: the counts and top readers.
#include "shim/bb_env.h"
#ifndef _WIN32

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <dlfcn.h>
#include <sys/mman.h>
#include <ucontext.h>

#include "common/types.h"
#include "core/signals.h"

namespace BbReadSample {
namespace {

constexpr u64 Window = 64 * 1024;
constexpr size_t Slots = 16;

struct Slot {
    std::atomic<u64> base{0}; ///< 0: free
    std::atomic<u32> result{0}; ///< 0 untouched, 1 read, 2 write
    std::atomic<u64> rip{0};
};
std::array<Slot, Slots> slots;

bool OnFault(void* context, void* fault_address) {
    const u64 address = reinterpret_cast<u64>(fault_address);
    for (auto& slot : slots) {
        const u64 base = slot.base.load(std::memory_order_acquire);
        if (base == 0 || address < base || address >= base + Window) {
            continue;
        }
        const auto* uc = static_cast<const ucontext_t*>(context);
        const bool write = (uc->uc_mcontext.gregs[REG_ERR] & 2) != 0;
        u32 expected = 0;
        if (slot.result.compare_exchange_strong(expected, write ? 2 : 1)) {
            slot.rip.store(u64(uc->uc_mcontext.gregs[REG_RIP]), std::memory_order_relaxed);
        }
        mprotect(reinterpret_cast<void*>(base), Window, PROT_READ | PROT_WRITE);
        return true;
    }
    return false;
}

/// Read-write windows of [lo, hi) per /proc/self/maps (a trap of another module makes a page
/// read-only or inaccessible: such windows are left alone).
std::vector<std::pair<u64, u64>> PlainRanges(const std::vector<std::pair<u64, u64>>& wanted) {
    std::vector<std::pair<u64, u64>> out;
    FILE* maps = std::fopen("/proc/self/maps", "r");
    char line[512];
    while (maps && std::fgets(line, sizeof(line), maps)) {
        unsigned long lo = 0, hi = 0;
        char perms[8] = {};
        if (std::sscanf(line, "%lx-%lx %7s", &lo, &hi, perms) != 3 || perms[0] != 'r' ||
            perms[1] != 'w') {
            continue;
        }
        for (const auto& [a, b] : wanted) {
            const u64 x = std::max<u64>(a, lo), y = std::min<u64>(b, hi);
            if (x + Window <= y) {
                out.emplace_back(x, y);
            }
        }
    }
    if (maps) {
        std::fclose(maps);
    }
    return out;
}

void Run(std::vector<std::pair<u64, u64>> wanted) {
    std::mt19937_64 rng{12345};
    u64 sampled = 0, reads = 0, writes = 0;
    std::map<u64, u64> readers; // rip -> reads
    auto report = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    for (;;) {
        const auto plain = PlainRanges(wanted);
        u64 total = 0;
        for (const auto& [a, b] : plain) {
            total += (b - a) / Window;
        }
        if (total > 0) {
            for (auto& slot : slots) {
                u64 pick = rng() % total;
                for (const auto& [a, b] : plain) {
                    const u64 n = (b - a) / Window;
                    if (pick < n) {
                        const u64 base = (a + pick * Window + Window - 1) & ~(Window - 1);
                        if (base + Window <= b) {
                            slot.result.store(0);
                            slot.base.store(base, std::memory_order_release);
                            mprotect(reinterpret_cast<void*>(base), Window, PROT_NONE);
                        }
                        break;
                    }
                    pick -= n;
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        for (auto& slot : slots) {
            const u64 base = slot.base.load();
            if (base == 0) {
                continue;
            }
            if (slot.result.load() == 0) {
                mprotect(reinterpret_cast<void*>(base), Window, PROT_READ | PROT_WRITE);
            }
            slot.base.store(0, std::memory_order_release);
            ++sampled;
            const u32 result = slot.result.load();
            if (result == 1) {
                ++reads;
                ++readers[slot.rip.load()];
            } else if (result == 2) {
                ++writes;
            }
        }
        if (std::chrono::steady_clock::now() >= report && sampled > 0) {
            report += std::chrono::seconds(10);
            std::printf("Read sample (10 s): %llu windows of 64 KiB for 200 ms: %.1f%% read first, "
                        "%.1f%% written first, %.1f%% untouched\n",
                        (unsigned long long)sampled, 100.0 * reads / sampled,
                        100.0 * writes / sampled, 100.0 * (sampled - reads - writes) / sampled);
            std::vector<std::pair<u64, u64>> top;
            for (const auto& [rip, n] : readers) {
                top.emplace_back(n, rip);
            }
            std::sort(top.rbegin(), top.rend());
            std::string text;
            for (size_t i = 0; i < top.size() && i < 8; ++i) {
                const u64 rip = top[i].second;
                if (rip >= 0x800000000ull && rip < 0x800000000ull + 0x50d96dcull) {
                    char buf[64];
                    std::snprintf(buf, sizeof(buf), " game+%#llx x%llu;",
                                  (unsigned long long)(rip - 0x800000000ull),
                                  (unsigned long long)top[i].first);
                    text += buf;
                } else {
                    Dl_info info{};
                    dladdr(reinterpret_cast<void*>(rip), &info);
                    char buf[256];
                    std::snprintf(buf, sizeof(buf), " %s:%s x%llu;",
                                  info.dli_fname ? std::strrchr(info.dli_fname, '/') + 1 : "?",
                                  info.dli_sname ? info.dli_sname : "?",
                                  (unsigned long long)top[i].first);
                    text += buf;
                }
            }
            std::printf("Read sample: readers%s\n", text.c_str());
            std::fflush(stdout);
            sampled = reads = writes = 0;
            readers.clear();
        }
    }
}

} // namespace

/// Starts the sampler when BB_READ_SAMPLE is set.
void Start() {
    const char* env = bb_env("BB_READ_SAMPLE");
    if (!env || !*env) {
        return;
    }
    std::vector<std::pair<u64, u64>> wanted;
    for (const char* at = env; *at;) {
        char* end = nullptr;
        const u64 lo = std::strtoull(at, &end, 16);
        if (!end || *end != '-') {
            break;
        }
        const u64 hi = std::strtoull(end + 1, &end, 16);
        wanted.emplace_back(lo, hi);
        at = (*end == ',') ? end + 1 : end;
        if (*end != ',') {
            break;
        }
    }
    if (wanted.empty()) {
        return;
    }
    Core::Signals::Instance()->RegisterAccessViolationHandler(OnFault, 0);
    std::printf("Read sample: %zu ranges\n", wanted.size());
    std::thread{Run, std::move(wanted)}.detach();
}

} // namespace BbReadSample

#endif

#ifndef _WIN32
extern "C" void bbgpu_read_sample_start() {
    BbReadSample::Start();
}
#endif
