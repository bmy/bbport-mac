// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: test_remote_table.cpp with several guest threads changing mappings at once, their
// operations overlapping. The mirror events reach the GPU process's copy in the order the
// runtime made the changes (queued under its lock, sent by the queuing thread): once every
// thread is done, the copy must answer as the runtime does. Built with -fsanitize=thread it
// also checks the runtime's mirror queue for data races.
//   cc -c -std=c11 -D_GNU_SOURCE -w -fsanitize=thread -g -Isrc -I. src/runtime_memory.c -o /tmp/rm.o
//   c++ -std=c++20 -O1 -g -fsanitize=thread -Igpu/shim tests/test_remote_concurrent.cpp /tmp/rm.o \
//       -lpthread -o /tmp/concurrent-test && /tmp/concurrent-test
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "remote/bb_vma_table.h"

extern "C" {
typedef struct {
    const char* name;
    void* function;
} RuntimeExport;
uintptr_t runtime_lookup(const RuntimeExport* table, size_t count, const char* nid) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::strcmp(table[i].name, nid)) {
            return reinterpret_cast<uintptr_t>(table[i].function);
        }
    }
    return 0;
}
int32_t* runtime_errno(void) {
    static int32_t value;
    return &value;
}
void runtime_guest_call_sites(uint64_t sites[3]) {
    sites[0] = sites[1] = sites[2] = 0;
}
uintptr_t runtime_memory_resolve(const char* name);
int runtime_memory_host_pool(uint64_t host_bytes, int* fd, uint64_t* host_offset, uint64_t* total);
typedef void (*MirrorHook)(int op, uintptr_t address, uint64_t size, uint64_t phys, int kind,
                           int prot, int type, int shared);
void runtime_memory_set_mirror_hook(MirrorHook hook);
uint64_t runtime_memory_clamp(uintptr_t address, uint64_t size);
int runtime_memory_region(uintptr_t address, uintptr_t* start, uintptr_t* end, int* mapped);
int runtime_memory_vma_info(uintptr_t address, int* prot, int* type, uintptr_t* end);
int runtime_memory_direct_phys(uintptr_t address, uint64_t* phys, uintptr_t* end);
int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
}

namespace {
using namespace BbRemote;

#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed (step %d, address %#" PRIxPTR ")\n",      \
                         __FILE__, __LINE__, #x, step, probe);                                     \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

VmaTable table;
int step = 0;
uintptr_t probe = 0;
unsigned events = 0;

// As remote_front.cpp (MirrorHook) and remote_back.cpp (MapMemory, UnmapMemory, ProtectMemory).
void Mirror(int op, uintptr_t address, uint64_t size, uint64_t phys, int kind, int prot, int type,
            int) {
    ++events;
    switch (op) {
    case 0:
        table.Map(address, address + size,
                  VmaEntry{address + size, kind == 2 ? std::uint32_t(MapDirect)
                                                     : std::uint32_t(MapFlexible),
                           prot, kind == 2 ? type : -1, phys});
        break;
    case 1:
        table.Unmap(address, address + size);
        break;
    case 2:
        table.Protect(address, address + size, prot, type);
        break;
    }
}

template <typename F>
F Fn(const char* name) {
    const uintptr_t at = runtime_memory_resolve(name);
    if (!at) {
        std::fprintf(stderr, "missing %s\n", name);
        std::exit(1);
    }
    return reinterpret_cast<F>(at);
}

using AllocateFn = int32_t (*)(int64_t, int64_t, uint64_t, uint64_t, int, int64_t*);
using MapDirectFn = int32_t (*)(void**, uint64_t, int, int, int64_t, uint64_t);
using ReleaseFn = int32_t (*)(uint64_t, uint64_t);
using UnmapFn = int32_t (*)(void*, uint64_t);
using MapFlexibleFn = int32_t (*)(void**, uint64_t, int, int);
using ReserveFn = int32_t (*)(void**, uint64_t, int, uint64_t);
using ProtectFn = int32_t (*)(const void*, uint64_t, int);

constexpr uint64_t Page = 16384;
constexpr int MapFixed = 0x10;

void Compare(uintptr_t address) {
    probe = address;
    uintptr_t s1 = 0, e1 = 0, s2 = 0, e2 = 0;
    int m1 = 0, m2 = 0;
    const int r1 = runtime_memory_region(address, &s1, &e1, &m1);
    const int r2 = table.Region(address, &s2, &e2, &m2);
    // The runtime also lists reservations: a gap here, maybe split by them there.
    if (r1 && m1) {
        CHECK(r2 && m2 && s1 == s2 && e1 == e2);
    } else {
        CHECK(!(r2 && m2));
    }
    for (const uint64_t size : {uint64_t(1), Page, 5 * Page, uint64_t(1) << 26}) {
        CHECK(runtime_memory_clamp(address, size) == table.Clamp(address, size));
        CHECK(runtime_memory_is_mapped(address, size) == table.Covered(address, size));
    }
    int p1 = 0, t1 = 0, p2 = 0, t2 = 0;
    uintptr_t v1 = 0, v2 = 0;
    const int i1 = runtime_memory_vma_info(address, &p1, &t1, &v1);
    const int i2 = table.VmaInfo(address, &p2, &t2, &v2);
    CHECK(i1 == i2);
    if (i1) {
        CHECK(p1 == p2 && t1 == t2 && v1 == v2);
    }
    uint64_t f1 = 0, f2 = 0;
    const int d1 = runtime_memory_direct_phys(address, &f1, &v1);
    const int d2 = table.DirectPhys(address, &f2, &v2);
    CHECK(d1 == d2);
    if (d1) {
        CHECK(f1 == f2 && v1 == v2);
    }
}

} // namespace

int main() {
    int fd = -1;
    uint64_t host_offset = 0, total = 0;
    if (runtime_memory_host_pool(64ull << 20, &fd, &host_offset, &total) != 0) {
        std::fprintf(stderr, "host pool failed\n");
        return 1;
    }
    runtime_memory_set_mirror_hook(Mirror); // called with the runtime's send mutex held
    const auto allocate = Fn<AllocateFn>("sceKernelAllocateDirectMemory");
    const auto map_direct = Fn<MapDirectFn>("sceKernelMapDirectMemory");
    const auto unmap = Fn<UnmapFn>("sceKernelMunmap");
    const auto map_flexible = Fn<MapFlexibleFn>("sceKernelMapFlexibleMemory");
    const auto protect = Fn<ProtectFn>("sceKernelMprotect");

    std::mutex starts_mutex;
    std::vector<uintptr_t> starts; // mappings made by any thread (some are gone)
    constexpr int Threads = 4, Steps = 600;
    std::vector<std::thread> threads;
    for (int t = 0; t < Threads; ++t) {
        threads.emplace_back([&, t] {
            std::mt19937_64 random{1000u + unsigned(t)};
            const auto pick = [&](uint64_t n) { return n ? random() % n : 0; };
            const auto any_start = [&]() -> uintptr_t {
                std::scoped_lock lock{starts_mutex};
                return starts.empty() ? 0 : starts[pick(starts.size())];
            };
            const auto note = [&](void* at) {
                std::scoped_lock lock{starts_mutex};
                starts.push_back(reinterpret_cast<uintptr_t>(at));
            };
            for (int i = 0; i < Steps; ++i) {
                const int what = int(pick(6));
                if (what == 0) { // allocate and map direct memory
                    const uint64_t size = (1 + pick(8)) * Page;
                    int64_t phys = 0;
                    if (allocate(0, int64_t(1) << 32, size, Page, int(pick(4)), &phys) == 0) {
                        void* at = nullptr;
                        if (map_direct(&at, size, 0x33, 0, phys, Page) == 0) {
                            note(at);
                            if (pick(2)) { // and over another thread's mapping, fixed
                                if (const uintptr_t other = any_start()) {
                                    void* fixed = reinterpret_cast<void*>(other + pick(2) * Page);
                                    if (map_direct(&fixed, Page, 0x11, MapFixed, phys, Page) == 0) {
                                        note(fixed);
                                    }
                                }
                            }
                        }
                    }
                } else if (what == 1) { // flexible memory
                    void* at = nullptr;
                    if (map_flexible(&at, (1 + pick(4)) * Page, 0x3, 0) == 0) {
                        note(at);
                    }
                } else if (what == 2 || what == 3) { // unmap a piece of anyone's mapping
                    if (const uintptr_t at = any_start()) {
                        unmap(reinterpret_cast<void*>(at + pick(2) * Page), (1 + pick(3)) * Page);
                    }
                } else if (const uintptr_t at = any_start()) { // change protection
                    protect(reinterpret_cast<void*>(at + pick(2) * Page), (1 + pick(3)) * Page,
                            int(pick(2) ? 0x3 : 0x1));
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    step = Threads * Steps;
    for (const auto& [start, size] : table.All()) {
        for (const uintptr_t a : {start - 1, start, start + 1, start + size - 1, start + size}) {
            Compare(a);
        }
    }
    for (const uintptr_t start : starts) {
        Compare(start);
        Compare(start + Page);
    }
    std::printf("mapping table: %d operations on %d threads (%u mirror events), the GPU "
                "process's copy answers as the runtime does\n",
                step, Threads, events);
    return 0;
}
