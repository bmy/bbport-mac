// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne: guards in the game's GX layer: the per-frame command heap running full (#162)
// and occlusion queries without a handle. Split out of gpu/shim/bbport_guest_hooks.cpp
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

constexpr unsigned char GxNodeCode[9] = {0x4d, 0x89, 0x6c, 0x24, 0x08,  // mov [r12 + 8], r13
                                         0x49, 0x8b, 0x87, 0xb0};        // mov rax, [r15 + 0xb0]

/// The GX occlusion queries (crash 2026-10-20 in the Hunter's Dream after 1 h 40 min: a guest
/// fault at 0x2169edc on GXRenderThread, address 0). A query (0x30 bytes, made by its pool,
/// 0xec6820 and 0xec68d0, with 0x2288d00) gets its handle, its first field, from 0xe2f080(this, 1) ->
/// 0x21674f0, which allocates a 0x18-byte handle (the device's allocator at +0xdc10) and its data
/// (+0xdc18). On a failure it returns 8 or 9 and the handle stays null; the pool does not look.
/// The render command 0x2285e10 then begins the query (0xe2f130: inner state [this + 8] 0 -> 2,
/// then 0x2169ec0(device, device + 0xb730, handle), which reads the handle: the fault) and ends it
/// (0xe2f160: state 1, 0x216a070, which reads it too). The guard: at the begin, a query whose
/// creation failed (GxQueryCreated keeps their addresses) is created again; one still without a
/// handle is not begun, and its end is skipped. Its state stays 0, so the result getter (0xe2f190)
/// gives its stored result, 0 (nothing visible). A null handle the creation did not leave (a query
/// released while the render thread holds it) is only skipped: creating one would write into a
/// freed object. The pool creates on the main thread, the retry runs on GXRenderThread: both
/// allocators (0x1fc5f30, 0x1fc5ed0) hold their lock (+0x468) around the game's TLSF heap (0x2085a60),
/// and the rest of the creation touches only the query. The cause (the device's heap full late in a
/// long session) is a guess; the observer logs each failed creation once a minute.
/// BB_GX_QUERY_GUARD=0: not installed.
/// BB_RE_GX_QUERY=1: the first creations and guard events are logged with their thread.
/// BB_GX_QUERY_FAIL=N (diagnostics): every N-th creation (retries too) fails as with a full heap
/// (the game's release 0xe2f020 takes the new handle back), so the guard's paths run in any
/// session. The pool makes 8192 at startup and hands out the last made first: N = 2 reaches a scene.
constexpr u64 GxQueryBeginSite = 0xe2f134; // cmp dword [rdi + 8], 0; jne 0xe2f157 (begun already)
constexpr u64 GxQueryReturn = 0xe2f157;    // mov al, 1; pop rbp; ret
constexpr unsigned char GxQueryBeginCode[6] = {0x83, 0x7f, 0x08, 0x00, 0x75, 0x1d};
constexpr u64 GxQueryEndSite = 0xe2f164;   // mov dword [rdi + 8], 1
constexpr u64 GxQueryEndReturn = 0xe2f181; // mov al, 1; pop rbp; ret
constexpr unsigned char GxQueryEndCode[7] = {0xc7, 0x47, 0x08, 0x01, 0x00, 0x00, 0x00};
constexpr u64 GxQueryCreate = 0xe2f080;
constexpr u64 GxQueryRelease = 0xe2f020;
/// The creation's common exit (`mov rcx, [r15]; cmp rcx, [rbp - 0x20]`, the stack check): eax the
/// result, r14 where the handle goes (the query), rbx null when the handle's allocation failed.
constexpr u64 GxQueryCreateExit = 0x21675b0;
constexpr std::array<u8, 7> GxQueryCreateExitCode = {0x49, 0x8b, 0x0f, 0x48, 0x3b, 0x4d, 0xe0};
std::array<std::atomic<u64>, 256> gx_query_failed{}; ///< queries whose creation failed (0: free)
std::atomic<u64> gx_query_failures{0}, gx_query_recreated{0}, gx_query_skipped{0};
std::atomic<long long> gx_query_failure_told{0}, gx_query_guard_told{0};

/// True the first time and then at most once a minute (`last`: steady clock, ms).
bool OncePerMinute(std::atomic<long long>& last) {
    const long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
    long long seen = last.load(std::memory_order_relaxed);
    return (seen == 0 || now - seen >= 60000) && last.compare_exchange_strong(seen, now);
}

std::array<char, 32> ThreadName() {
    std::array<char, 32> name{};
    // Windows: the main thread has no name (an empty one).
    if (pthread_getname_np(pthread_self(), name.data(), name.size()) != 0 || !name[0]) name = {"?"};
    return name;
}

/// BB_RE_GX_QUERY=1: true for the first 16 events counted in `traced`.
bool GxQueryTrace(std::atomic<int>& traced) {
    static const bool trace = bb_env_on("BB_RE_GX_QUERY");
    return trace && traced.fetch_add(1) < 16;
}

void PS4_SYSV_ABI GxQueryCreated(u64, const Core::GuestRegisterSnapshot* r) {
    auto status = std::uint32_t(r->rax);
    const u64 query = r->r14;
    static const long fail_every = [] {
        const char* env = bb_env("BB_GX_QUERY_FAIL");
        return env ? std::strtol(env, nullptr, 10) : 0l;
    }();
    static std::atomic<u64> creations{0};
    const bool simulated = status == 0 && fail_every > 0 &&
                           (creations.fetch_add(1) + 1) % u64(fail_every) == 0;
    if (simulated) {
        using Release = void(PS4_SYSV_ABI*)(u64 query);
        reinterpret_cast<Release>(ImageBase + GxQueryRelease)(query);
        const_cast<Core::GuestRegisterSnapshot*>(r)->rax = status = 9;
    }
    static std::atomic<int> traced{0};
    if (GxQueryTrace(traced)) {
        std::printf("GX query: %#llx created, result %u%s (thread %s)\n", static_cast<unsigned long long>(query),
                    status, simulated ? " (BB_GX_QUERY_FAIL)" : "", ThreadName().data());
        // The device's two allocators (the pool's creation passes none of its own): their
        // functions, to check they still lock (another thread may call them).
        static std::atomic<bool> told{false};
        const u64 device = JobProbeWord(JobProbeWord(ImageBase + 0x55406c8));
        const u64 handles = JobProbeWord(device + 0xdc10), data = JobProbeWord(device + 0xdc18);
        const u64 handles_vt = JobProbeWord(handles), data_vt = JobProbeWord(data);
        if (handles_vt && data_vt && !told.exchange(true)) {
            std::printf("GX query: device %#llx; handle allocator %#llx: allocate +%#llx free +%#llx; "
                        "data allocator %#llx: allocate +%#llx free +%#llx\n",
                        static_cast<unsigned long long>(device), static_cast<unsigned long long>(handles),
                        static_cast<unsigned long long>(JobProbeWord(handles_vt + 0x58) - ImageBase),
                        static_cast<unsigned long long>(JobProbeWord(handles_vt + 0x70) - ImageBase),
                        static_cast<unsigned long long>(data),
                        static_cast<unsigned long long>(JobProbeWord(data_vt + 0x50) - ImageBase),
                        static_cast<unsigned long long>(JobProbeWord(data_vt + 0x70) - ImageBase));
        }
    }
    if (status == 0) {
        for (auto& slot : gx_query_failed) {
            u64 expected = query;
            if (slot.load(std::memory_order_relaxed) == query) slot.compare_exchange_strong(expected, 0);
        }
        return;
    }
    bool known = false;
    for (const auto& slot : gx_query_failed) known = known || slot.load(std::memory_order_relaxed) == query;
    for (auto& slot : gx_query_failed) {
        u64 expected = 0;
        if (known || slot.compare_exchange_strong(expected, query)) break;
    }
    const u64 n = gx_query_failures.fetch_add(1) + 1;
    if (!OncePerMinute(gx_query_failure_told)) return;
    std::printf("Guest hooks: a GX occlusion query could not be created (error %u: %s), %llu so far "
                "(thread %s); the query guard creates it again before it begins\n", status,
                simulated     ? "BB_GX_QUERY_FAIL"
                : status == 9 ? (r->rbx ? "no memory for its data" : "no memory for its handle")
                              : "the device or its data not ready",
                static_cast<unsigned long long>(n), ThreadName().data());
}

/// The begin's stub calls this for a query without a handle (rdi). A query whose creation failed
/// is created again (the game's own 0xe2f080, as the pool does); the stub then looks again.
void PS4_SYSV_ABI GxQueryWithoutHandle(u64 query) {
    bool failed = false;
    for (const auto& slot : gx_query_failed) failed = failed || slot.load(std::memory_order_relaxed) == query;
    if (failed) {
        using Create = bool(PS4_SYSV_ABI*)(u64 query, std::uint32_t kind);
        reinterpret_cast<Create>(ImageBase + GxQueryCreate)(query, 1); // GxQueryCreated keeps or forgets it
    }
    const bool created = *reinterpret_cast<const volatile u64*>(query) != 0;
    const u64 recreated = gx_query_recreated.fetch_add(created) + created;
    const u64 skipped = gx_query_skipped.fetch_add(!created) + !created;
    static std::atomic<int> traced{0};
    if (!OncePerMinute(gx_query_guard_told) && !GxQueryTrace(traced)) return;
    std::printf("Guest hooks: GX occlusion query %#llx had no handle at its begin (%s): %llu created "
                "again, %llu not begun (no fault; nothing visible) so far (thread %s)\n",
                static_cast<unsigned long long>(query),
                failed ? "its creation had failed" : "no failed creation seen: released?",
                static_cast<unsigned long long>(recreated), static_cast<unsigned long long>(skipped),
                ThreadName().data());
}

} // namespace

bool InstallGxNodeGuard() {
    const u64 address = ImageBase + GxNodeSite;
    if (std::memcmp(reinterpret_cast<const void*>(address), GxNodeCode, sizeof(GxNodeCode))) {
        return false;
    }
    Emitter e;
    e.Bytes({0x4d, 0x85, 0xe4});                   // test r12, r12
    e.Bytes({0x74, 0x13});                         // jz null (past the next 19 bytes)
    e.Bytes({0x4d, 0x89, 0x6c, 0x24, 0x08});       // mov [r12 + 8], r13
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(address + 5);
    const std::size_t trap_at = e.size;
    e.Bytes({0xcc});                               // null: OnTrap counts it and skips the post
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) return false;
    gx_node_skip_trap.store(stub + trap_at, std::memory_order_release);
    const long page_size = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
    mprotect(page, (address + 5) - reinterpret_cast<u64>(page), PROT_READ | PROT_WRITE | PROT_EXEC);
    reinterpret_cast<u8*>(address)[0] = 0xcc; // OnTrap runs the instruction meanwhile
    mprotect(page, (address + 5) - reinterpret_cast<u64>(page), Core::GuestCodeProtection());
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

bool InstallGxQueryGuard() {
    if (bb_env_off("BB_GX_QUERY_GUARD")) {
        return false;
    }
    const u64 begin = ImageBase + GxQueryBeginSite;
    const u64 end = ImageBase + GxQueryEndSite;
    if (std::memcmp(reinterpret_cast<const void*>(begin), GxQueryBeginCode, sizeof(GxQueryBeginCode)) ||
        std::memcmp(reinterpret_cast<const void*>(end), GxQueryEndCode, sizeof(GxQueryEndCode))) {
        return false;
    }
    // The begin: the game's state check, then the handle's.
    Emitter b;
    b.Bytes({0x83, 0x7f, 0x08, 0x00});             // cmp dword [rdi + 8], 0
    b.Bytes({0x75, 0x00});                         // jne done (begun or ended already)
    const std::size_t to_done = b.size;
    b.Bytes({0x48, 0x83, 0x3f, 0x00});             // cmp qword [rdi], 0
    b.Bytes({0x74, 0x00});                         // je none
    const std::size_t to_none = b.size;
    const std::size_t body = b.size;
    b.JmpAbs(begin + sizeof(GxQueryBeginCode));    // body: state 2, the begin
    const std::size_t done = b.size;
    b.JmpAbs(ImageBase + GxQueryReturn);
    b.code[to_done - 1] = u8(done - to_done);
    b.code[to_none - 1] = u8(b.size - to_none);
    b.SaveVolatile();                              // none: rdi (the query) and rsi kept
    b.CallAbs(reinterpret_cast<const void*>(&GxQueryWithoutHandle));
    b.RestoreVolatile();
    b.Bytes({0x48, 0x83, 0x3f, 0x00});             // cmp qword [rdi], 0: created again?
    b.Bytes({0x75, u8(body - (b.size + 2))});      // jne body
    b.Bytes({0xeb, u8(done - (b.size + 2))});      // jmp done: not begun, the state stays 0
    // The end: only a query with a handle (begun) is ended.
    Emitter e;
    e.Bytes({0x48, 0x83, 0x3f, 0x00});             // cmp qword [rdi], 0
    e.Bytes({0x74, 0x00});                         // je none
    const std::size_t to_skip = e.size;
    for (const u8 c : GxQueryEndCode) e.Bytes({c}); // mov dword [rdi + 8], 1
    e.JmpAbs(end + sizeof(GxQueryEndCode));
    e.code[to_skip - 1] = u8(e.size - to_skip);
    e.JmpAbs(ImageBase + GxQueryEndReturn);        // none: not ended, the state stays 0
    const u64 begin_stub = PlaceStub(b);
    const u64 end_stub = PlaceStub(e);
    std::uint32_t begin_rel = 0, end_rel = 0;
    if (!begin_stub || !end_stub || !Rel32To(begin + 5, begin_stub, begin_rel) ||
        !Rel32To(end + 5, end_stub, end_rel)) {
        return false;
    }
    // Without the creation observer a query without a handle is only skipped.
    gx_query_observer = Core::InstallGuestCodeHook(reinterpret_cast<void*>(ImageBase + GxQueryCreateExit),
                                                   GxQueryCreateExitCode, 0, GxQueryCreated);
    WriteJumpOverTrap(begin, 0xe9, begin_rel);
    WriteJumpOverTrap(end, 0xe9, end_rel);
    return true;
}

} // namespace BbGuestHooks
