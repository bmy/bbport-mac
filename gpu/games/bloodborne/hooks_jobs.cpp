// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne: the render job fixes (the job slot leak, the job allocation failure probe) and
// the size of the game's render resource cache. Split out of gpu/shim/bbport_guest_hooks.cpp
// (2026-10-26).

#include "common/types.h"
#include "shim/bb_env.h"
#include "bbport_guest_hooks.h"
#include "bbport_settings.h"
#include "bbport_guest_code.h"
#include "bbport_heap_sites.h"
#include "bbport_toggles.h"
#include "bbport_fx_init_clear.h"
#include "bbport_effect_flags.h"
#include "platform/gnm/gnm.h"

#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include "bbport_engine_fixes.h"
#include "bbport_guest_code_hooks.h"
#include <cmath>
#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>
#include <utility>
#include "games/bloodborne/hooks_internal.h"

namespace BbGuestHooks {
namespace {

/// The render job pool (found in a hang of Laurence's cinematic, 2026-10-16). Six slots, a queue at
/// 0x1056920 (take a free slot, sleeping until one is free) and 0x1056990 (give it back); the one
/// caller of both is the job runner's step at 0x1067f60: take a slot, check the job's resource
/// (0x2165150), run it, give the slot back. Its two error exits (the resource is not in the state
/// asked for; the run fails) leave through 0x1068092 returning 1 with the slot still held in
/// [runner + 0x120], and the next call takes another one: each failed job loses a slot for good.
/// After six, every thread that needs a slot sleeps for ever (the main thread and Havok's five
/// workers polled the same empty pool in the dump; six runners held its six slots). The exit now
/// gives the held slot back before it returns. BB_JOB_SLOT_FIX=0: as the game has it.
std::atomic<u64> job_slots_released{0};

void JobProbeObject(const char* name, u64 address) {
    std::printf("Job allocation: %s=%#llx", name, static_cast<unsigned long long>(address));
    if (address) {
        for (u64 off = 0; off < 0xe0; off += 8) {
            std::printf(" +%02llx=%#llx", static_cast<unsigned long long>(off),
                        static_cast<unsigned long long>(JobProbeWord(address + off)));
        }
    }
    std::printf("\n");
}

void JobProbeRangeHeap(u64 heap) {
    if (!heap || !runtime_memory_is_mapped(heap, 0xe0)) return;
    const u64 sentinel = heap + 0x68;
    u64 node = JobProbeWord(heap + 0x88), free_bytes = 0, largest = 0;
    unsigned blocks = 0;
    for (; node && node != sentinel && blocks < 8192; ++blocks) {
        if (!runtime_memory_is_mapped(node, 0x30)) return;
        const u64 begin = JobProbeWord(node + 8);
        const u64 end = JobProbeWord(JobProbeWord(node + 0x18) + 8);
        if (end < begin || end - begin > (1ull << 36)) return;
        free_bytes += end - begin;
        largest = std::max(largest, end - begin);
        node = JobProbeWord(node + 0x20);
    }
    if (node == sentinel) {
        std::printf("Job allocation: range heap=%#llx live=%llu reported_free=%llu free=%llu "
                    "largest=%llu free_blocks=%u\n", static_cast<unsigned long long>(heap),
                    static_cast<unsigned long long>(JobProbeWord(heap + 0xc8)),
                    static_cast<unsigned long long>(JobProbeWord(heap + 0xd0)),
                    static_cast<unsigned long long>(free_bytes),
                    static_cast<unsigned long long>(largest), blocks);
    }
}

// Error 9 in 0x2169c30 comes from allocation, not its preceding state check (error 8).
// Observe only failed retries/growth, preserving every guest register and flag.
void PS4_SYSV_ABI JobAllocationFailure(u64 tag, const Core::GuestRegisterSnapshot* r) {
    if (tag == 0 ? r->rcx != 0 : r->r14 != 0) return;
    static std::atomic<u64> failures{0};
    const u64 n = failures.fetch_add(1) + 1;
    if (n > 3 && (n & (n - 1)) != 0) return;
    const u64 allocator = tag == 0 ? r->r12 : JobProbeWord(r->rbx);
    const u64 bytes = tag == 0 ? (JobProbeWord(r->rbp - 0x54) & 0xffffffff)
                              : ((tag == 1 ? r->r15 : JobProbeWord(r->rbx + 0x18) & 0xffffffff) * 24 + 8);
    std::printf("Job allocation: %llu failures, %s bytes=%llu allocator=%#llx "
                "allocate_fn=+%#llx\n", static_cast<unsigned long long>(n),
                tag == 0 ? "resource buffer" : "tracking block",
                static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(allocator),
                static_cast<unsigned long long>(JobProbeWord(JobProbeWord(allocator) +
                                                             (tag == 0 ? 0x18 : 0x58)) - ImageBase));
    JobProbeObject("allocator", allocator);
    JobProbeObject("allocator+8 target", JobProbeWord(allocator + 8));
    JobProbeRangeHeap(JobProbeWord(allocator + 8));
    JobProbeObject("tracking", r->rbx);
}

// The resource cache used by 0x26d0d30 has its own backing allocation, independent of the
// graphics-heap budget patches. At high frame rates its 160 MiB range heap fills before the
// cinematic's geometry is prepared; failed 20-40 KiB allocations then leave missing SRTs.
// 320 MiB still failed on 4 MiB requests in the same load; 512 MiB survived that peak and
// returned to a few hundred live ranges with most of the cache free after streaming finished.
// Taken from the game's own allocator (2026-10-17) the larger backing starved the memory areas
// load into: after a trip to the Hunter's Dream and back, map pieces (walls, the city below,
// distant buildings) were missing. Now the game allocates its 160 MiB as always and the cache
// gets the port's direct memory past the game's size instead (runtime_extra_direct_map,
// BB_DMEM_EXTRA_MB): right after the allocation, its result and the size the range heap is
// set up with are replaced. BB_RENDER_RESOURCE_CACHE_MB=160: the game's cache as it is.
extern "C" int runtime_extra_direct_map(uint64_t size, uint64_t alignment, void** out);
void PS4_SYSV_ABI ResourceCacheBacking(u64, const Core::GuestRegisterSnapshot* r) {
    constexpr u64 native_bytes = 160ull << 20;
    if ((JobProbeWord(r->r14 + 0x54) & 0xffffffffu) != native_bytes || !r->rax) return;
    static const std::uint32_t bytes = [] {
        const char* env = bb_env("BB_RENDER_RESOURCE_CACHE_MB");
        const long mb = env ? std::strtol(env, nullptr, 10) : 512;
        return std::uint32_t(std::clamp(mb, 160l, 512l)) << 20;
    }();
    if (bytes == native_bytes) return;
    void* block = nullptr;
    if (const int error = runtime_extra_direct_map(bytes, 64ull << 10, &block); error || !block) {
        std::printf("Render resource cache: no port memory for %u MiB (error %#x): the game's "
                    "160 MiB\n", bytes >> 20, unsigned(error));
        return;
    }
    // The creation record can be a stack object: not necessarily in the runtime's map.
    std::memcpy(reinterpret_cast<void*>(r->r14 + 0x54), &bytes, sizeof(bytes));
    const_cast<Core::GuestRegisterSnapshot*>(r)->rax = reinterpret_cast<u64>(block);
    std::printf("Render resource cache: %u MiB in the port's memory at %p (the game's 160 MiB "
                "stays allocated)\n", bytes >> 20, block);
}

void PS4_SYSV_ABI JobSlotFix(u64 runner, u64 queue, std::uint32_t result,
                            std::uint32_t error) {
    if (result != 1) {
        return;
    }
    u64& held = *reinterpret_cast<u64*>(runner + 0x120);
    const u64 item = held;
    if (!item) {
        return;
    }
    using GiveBack = void(PS4_SYSV_ABI*)(u64 queue, u64 item);
    reinterpret_cast<GiveBack>(ImageBase + 0x1056990)(queue, item);
    held = 0;
    const u64 n = job_slots_released.fetch_add(1) + 1;
    if (n <= 5 || (n & (n - 1)) == 0) {
        const u64 job = *reinterpret_cast<const u64*>(runner + 0x118);
        const u64 resource = job ? *reinterpret_cast<const u64*>(job + 0x40) : 0;
        std::uint32_t state = 0;
        if (resource && runtime_memory_is_mapped(resource + 0x38, sizeof(state))) {
            std::memcpy(&state, reinterpret_cast<const void*>(resource + 0x38), sizeof(state));
        }
        std::printf("Guest hooks: a failed render job gave its slot back (%llu so far); "
                    "error=%u runner=%#llx resource=%#llx state[38..3b]=%08x\n",
                    static_cast<unsigned long long>(n), error,
                    static_cast<unsigned long long>(runner),
                    static_cast<unsigned long long>(resource), state);
    }
}

/// The runner is destroyed or reset (0x106798a, 0x1067a5c: `mov qword [rbx + 0x120], 0`) with a slot
/// still held (a job cancelled in the middle): the slot goes back first. The pool is the global the
/// game's own 0x1057350 returns.
void PS4_SYSV_ABI JobSlotDrop(u64 runner) {
    u64& held = *reinterpret_cast<u64*>(runner + 0x120);
    const u64 item = held;
    if (!item) {
        return;
    }
    const u64 queue = *reinterpret_cast<const u64*>(ImageBase + 0x54b8788);
    if (!queue) {
        return;
    }
    using GiveBack = void(PS4_SYSV_ABI*)(u64 queue, u64 item);
    reinterpret_cast<GiveBack>(ImageBase + 0x1056990)(queue, item);
    held = 0;
    const u64 n = job_slots_released.fetch_add(1) + 1;
    if (n <= 5 || (n & (n - 1)) == 0) {
        std::printf("Guest hooks: a render job destroyed with its slot gave it back (%llu so far)\n",
                    static_cast<unsigned long long>(n));
    }
}

bool InstallJobSlotDropHook(u64 offset) {
    // mov qword [rbx + 0x120], 0
    static constexpr std::array<u8, 11> code = {0x48, 0xc7, 0x83, 0x20, 0x01, 0x00,
                                                0x00, 0x00, 0x00, 0x00, 0x00};
    const u64 address = ImageBase + offset;
    if (std::memcmp(reinterpret_cast<const void*>(address), code.data(), code.size())) {
        return false;
    }
    Emitter e;
    e.SaveVolatile();
    e.Bytes({0x48, 0x89, 0xdf});                   // mov rdi, rbx (the runner)
    e.CallAbs(reinterpret_cast<const void*>(&JobSlotDrop));
    e.RestoreVolatile();
    for (const u8 b : code) {
        e.Bytes({b});
    }
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(address + code.size());
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

} // namespace

bool InstallJobAllocationProbe() {
    constexpr std::array<u8, 8> buffer_fail{0xb8, 9, 0, 0, 0, 0x48, 0x85, 0xc9};
    constexpr std::array<u8, 8> tracking_fail{0xb8, 9, 0, 0, 0, 0x4d, 0x85, 0xf6};
    bool ok = Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x26d0dac),
                                         buffer_fail, 0, JobAllocationFailure);
    ok &= Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x26d0e82),
                                     tracking_fail, 1, JobAllocationFailure);
    ok &= Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x26d0ebb),
                                     tracking_fail, 2, JobAllocationFailure);
    return ok;
}

bool InstallResourceCacheSize() {
    // mov rcx, rbx; mov [rcx + 0x10], rax: right after the allocator's Allocate(size, 16) returns.
    constexpr std::array<u8, 7> code{0x48, 0x89, 0xd9, 0x48, 0x89, 0x41, 0x10};
    return Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + 0x216b21d), code,
                                      0, ResourceCacheBacking);
}

bool InstallJobSlotFix() {
    if (bb_env_off("BB_JOB_SLOT_FIX")) {
        return false;
    }
    constexpr u64 offset = 0x1068092;
    // mov rax, [rbx]; cmp rax, [rbp-0x30]: the stack check every exit of that function ends in.
    static constexpr std::array<u8, 7> code = {0x48, 0x8b, 0x03, 0x48, 0x3b, 0x45, 0xd0};
    const u64 address = ImageBase + offset;
    if (std::memcmp(reinterpret_cast<const void*>(address), code.data(), code.size())) {
        return false;
    }
    Emitter e;
    e.SaveVolatile();
    e.Bytes({0x4c, 0x89, 0xf7});                   // mov rdi, r14 (the runner)
    // SaveVolatile uses r12 for its stack anchor; the game's r12 (the pool) is
    // saved at [r12]. Passing the anchor itself silently loses the slot:
    // GiveBack finds no matching item, then JobSlotFix clears the held pointer.
    e.Bytes({0x49, 0x8b, 0x34, 0x24});             // mov rsi, [r12] (the saved pool)
    e.Bytes({0x44, 0x89, 0xfa});                   // mov edx, r15d (the result)
    e.Bytes({0x89, 0xc1});                         // mov ecx, eax (the original error)
    e.CallAbs(reinterpret_cast<const void*>(&JobSlotFix));
    e.RestoreVolatile();
    for (const u8 b : code) {
        e.Bytes({b});                              // the two instructions the jump covers
    }
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(address + code.size());
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe9, rel);
    return InstallJobSlotDropHook(0x106798a) & InstallJobSlotDropHook(0x1067a5c) && true;
}

} // namespace BbGuestHooks
