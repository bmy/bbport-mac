// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Bloodborne: installing the hooks in the game's code (Install), the int3 trap handler, and
// the hooks that tell the GPU side about the game's GPU memory: its range allocator, its
// heap and the resource loaders' copies. Split out of gpu/shim/bbport_guest_hooks.cpp
// (2026-10-26); the detour helpers and shared state are in hooks_internal.h.

#include "bbport_map_near.h"
#include "shim/bb_env.h"
#include "common/types.h"
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
#ifdef __APPLE__
#include <sys/ucontext.h> // bbport: <ucontext.h> requires _XOPEN_SOURCE on macOS
#else
#include <ucontext.h>
#endif
#include <unistd.h>
#include <utility>
#include "games/bloodborne/hooks_internal.h"
#include "bbport_portable.h"
#include "common/signal_context.h"

namespace BbGuestHooks {
namespace {

/// The return of the game's GPU memory range allocator 0x26aa070 (rdi allocator, rsi size,
/// rdx alignment; it returns a node: [node + 8] the start, [[node + 0x18] + 8] the end, the next
/// node's start): `add rsp, 0x18` in its epilogue, reached by its own branches only.
constexpr u64 AllocReturn = 0x26aa255;
constexpr unsigned char AllocReturnCode[4] = {0x48, 0x83, 0xc4, 0x18};
RangeCallback on_range_allocated = nullptr;

/// The game's other GPU memory heap (TLSF-like, headers in the memory: free at 0x20858a0, alloc at
/// 0x2085a60), reached through a pointer at 0x54b9f38 set at startup. Its only three callers
/// (wrappers that record {address, size} in a map) pass r8 = rbp - 0x50 for the size handed out and
/// follow the call with `mov rbx, rax` (rax: the address).
constexpr std::array<u64, 3> HeapAllocReturns = {0x11e6629, 0x11e6736, 0x11e6846};
constexpr unsigned char HeapAllocReturnCode[3] = {0x48, 0x89, 0xc3};

/// BB_FRAME_STATS: the range allocator's collector (0x26aa860, `push rbp`; rdi the allocator): its
/// live range count [+0xc8] and byte count [+0xd0] are noted for the frame stats.
constexpr u64 CollectorEntry = 0x26aa860;
constexpr unsigned char CollectorEntryCode[1] = {0x55};

/// BB_HEAP_SITES=1: the game's deferred object release (0xbb5f10) skips the destruction when the
/// registry check (0xbf0790) fails: `js 0xbb5fc7` there, its result counted (eax).
constexpr u64 ReleaseCheck = 0xbb5fb0;
constexpr unsigned char ReleaseCheckCode[2] = {0x78, 0x15};

constexpr std::array<CallSite, 2> MemcpySites = {{
    {0x216d5ea, {0xe8, 0xc1, 0xcb, 0x4a, 0x00}},
    {0x2171fea, {0xe8, 0xc1, 0x81, 0x4a, 0x00}},
}};
std::array<std::atomic<u64>, MemcpySites.size()> hits{};

/// The game's memcpy at those sites: the GPU side is told before (its write traps lift: no fault
/// per page) and after the copy (the data now there), as for file reads.
void* PS4_SYSV_ABI LoaderCopy(void* dst, const void* src, std::size_t size) {
    runtime_memory_note_write(reinterpret_cast<u64>(dst), size);
    std::memcpy(dst, src, size);
    runtime_memory_note_write(reinterpret_cast<u64>(dst), size);
    return dst;
}

/// bbport: native detours (default; BB_GUEST_HOOK_TRAPS=1: int3 traps). Each hooked site jumps to
/// a stub next to the image that calls the handler and runs the instructions the jump covers;
/// the copy call sites call LoaderCopy through a trampoline there. An int3 trap costs a signal
/// (a few microseconds in the kernel) per call; a detour costs a call. The int3 written first
/// stays the fallback while the jump's bytes go in: OnTrap sends a thread that meets it to the stub.
struct NativeSite {
    u64 offset;
    std::atomic<u64> stub{0};
};
std::array<NativeSite, 4> native_sites{{{AllocReturn}, {HeapAllocReturns[0]}, {HeapAllocReturns[1]},
                                        {HeapAllocReturns[2]}}};

/// A page within +-2 GiB of the image (below its base: the image grows upwards).
u8* MapStubPage() {
    // Within reach of the game's code and data (its first 96 MiB): below the image, else above it,
    // else the runtime's low region (bbport_map_near.h). Below only, a PC with that space taken
    // (RTX 3080, Windows, 2026-10-28) went without every hook of the block below: the output's
    // aspect ratio (16:9 stretched over 21:9), the HUD's markers, native mouse look, the fixes.
    return static_cast<u8*>(BbMapNear::Map(ImageBase, ImageBase + (96ull << 20), StubPageSize,
                                           PROT_READ | PROT_WRITE | PROT_EXEC, 16ull << 20,
                                           "Guest hooks"));
}

// BB_HEAP_TRACE=1: which 16 MiB of the game's direct memory addresses its two GPU allocators
// ever handed out (0: the range allocator, 1: the in-band heap), printed every 60 s as runs.
constexpr u64 TraceBase = 0x1000000000ull, TraceBucket = 16ull << 20;
std::array<std::array<std::atomic<std::uint32_t>, 4096>, 2> heap_buckets{};

void HeapBucketsNote(int kind, u64 start, u64 size) {
    static const bool on = bb_env_on("BB_HEAP_TRACE");
    if (!on || start < TraceBase || size == 0) {
        return;
    }
    const u64 first = (start - TraceBase) / TraceBucket, last = (start + size - 1 - TraceBase) / TraceBucket;
    for (u64 b = first; b <= last && b < 4096; ++b) {
        heap_buckets[kind][b].fetch_add(1, std::memory_order_relaxed);
    }
    static std::atomic<long long> printed{0};
    const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::steady_clock::now().time_since_epoch()).count();
    long long last_print = printed.load(std::memory_order_relaxed);
    if (now - last_print < 60 || !printed.compare_exchange_strong(last_print, now)) {
        return;
    }
    static long long started = now;
    if (now - started >= 50) {
        HeapFind();
    }
    for (int k = 0; k < 2; ++k) {
        std::string runs;
        for (u64 b = 0; b < 4096;) {
            if (!heap_buckets[k][b].load(std::memory_order_relaxed)) {
                ++b;
                continue;
            }
            u64 e = b;
            u64 n = 0;
            while (e < 4096 && heap_buckets[k][e].load(std::memory_order_relaxed)) {
                n += heap_buckets[k][e].load(std::memory_order_relaxed);
                ++e;
            }
            char item[96];
            std::snprintf(item, sizeof(item), " %#llx-%#llx(%llu)",
                          (unsigned long long)(TraceBase + b * TraceBucket),
                          (unsigned long long)(TraceBase + e * TraceBucket), (unsigned long long)n);
            runs += item;
            b = e;
        }
        std::printf("Heap trace: %s ever handed out in:%s\n",
                    k == 0 ? "range allocator" : "in-band heap", runs.c_str());
    }
}

void PS4_SYSV_ABI AllocHook(u64 node) {
    if (!node) {
        return;
    }
    const u64 start = *reinterpret_cast<const u64*>(node + 8);
    const u64 next = *reinterpret_cast<const u64*>(node + 0x18);
    const u64 end = next ? *reinterpret_cast<const u64*>(next + 8) : start;
    if (end > start && end - start < (u64(1) << 32)) {
        HeapBucketsNote(0, start, end - start);
        on_range_allocated(start, end - start);
    }
}

void PS4_SYSV_ABI HeapHook(u64 address, u64 size) {
    if (address != 0 && size != 0 && size < (u64(1) << 32)) {
        HeapBucketsNote(1, address, size);
        on_range_allocated(address, size);
    }
}

/// The detour of a site whose int3 is in place (Patch). False: left as a trap.
bool InstallDetour(NativeSite& site) {
    const u64 address = ImageBase + site.offset;
    Emitter e;
    u64 back = 0;
    if (site.offset == AllocReturn) {
        // `add rsp, 0x18; pop rbx` (5 bytes) in the allocator's epilogue; rax: the node.
        if (reinterpret_cast<const u8*>(address)[4] != 0x5b) {
            return false;
        }
        e.SaveVolatile();
        e.Bytes({0x48, 0x89, 0xc7}); // mov rdi, rax (unchanged by the saving)
        e.CallAbs(reinterpret_cast<const void*>(&AllocHook));
        e.RestoreVolatile();
        e.Bytes({0x48, 0x83, 0xc4, 0x18}); // add rsp, 0x18
        e.Bytes({0x5b});                   // pop rbx
        back = address + 5;
    } else {
        // `mov rbx, rax; test rbx, rbx` (6 bytes) after the heap call; [rbp - 0x50]: the size.
        const auto* code = reinterpret_cast<const u8*>(address);
        if (code[3] != 0x48 || code[4] != 0x85 || code[5] != 0xdb) {
            return false;
        }
        e.Bytes({0x48, 0x89, 0xc3});                               // mov rbx, rax
        e.Bytes({0x48, 0x8d, 0xa4, 0x24, 0x80, 0xff, 0xff, 0xff}); // lea rsp, [rsp - 128]
        e.SaveVolatile();
        e.Bytes({0x48, 0x89, 0xdf});       // mov rdi, rbx
        e.Bytes({0x48, 0x8b, 0x75, 0xb0}); // mov rsi, [rbp - 0x50]
        e.CallAbs(reinterpret_cast<const void*>(&HeapHook));
        e.RestoreVolatile();
        e.Bytes({0x48, 0x8d, 0xa4, 0x24, 0x80, 0x00, 0x00, 0x00}); // lea rsp, [rsp + 128]
        e.Bytes({0x48, 0x85, 0xdb});                               // test rbx, rbx
        back = address + 6;
    }
    // jmp back
    const u64 jmp_at = reinterpret_cast<u64>(stub_page + stub_used) + e.size;
    std::uint32_t rel_back = 0;
    if (!Rel32To(jmp_at + 5, back, rel_back)) {
        return false;
    }
    e.Bytes({0xe9});
    e.U32(rel_back);
    const u64 stub = PlaceStub(e);
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    site.stub.store(stub, std::memory_order_release); // a thread at the trap now goes there
    WriteJumpOverTrap(address, 0xe9, rel);
    return true;
}

/// The copy call sites: the call's target becomes a trampoline to LoaderCopy.
bool InstallCallDetour(u64 offset) {
    Emitter e;
    e.Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00}); // jmp [rip]
    e.U64(reinterpret_cast<u64>(&LoaderCopy));
    const u64 stub = PlaceStub(e);
    const u64 address = ImageBase + offset;
    std::uint32_t rel = 0;
    if (!stub || !Rel32To(address + 5, stub, rel)) {
        return false;
    }
    WriteJumpOverTrap(address, 0xe8, rel);
    return true;
}

struct sigaction previous_action {};

void OnTrap(int sig, siginfo_t* info, void* context) {
    greg_t* g = Common::Gregs(context); // bbport: portable (macOS)
    const u64 rip = u64(g[REG_RIP]) - 1; // past the int3
    // The GX node guard (#162): its site while the detour goes in, and its null case.
    const bool gx_site = rip == ImageBase + GxNodeSite;
    if (gx_site || (rip != 0 && rip == gx_node_skip_trap.load(std::memory_order_acquire))) {
        if (gx_site && g[REG_R12] != 0) {
            *reinterpret_cast<u64*>(u64(g[REG_R12]) + 8) = u64(g[REG_R13]);
            g[REG_RIP] = greg_t(ImageBase + GxNodeSite + 5);
            return;
        }
        static std::atomic<bool> told{false};
        if (!told.exchange(true)) {
            static const char text[] = "Guest hooks: the GX per-frame command heap was full: a render "
                                       "command waits for the next frame (#162)\n";
            [[maybe_unused]] const auto written = write(STDOUT_FILENO, text, sizeof(text) - 1);
        }
        g[REG_RIP] = greg_t(ImageBase + GxNodeSkip);
        return;
    }
    for (const auto& site : native_sites) {
        if (rip == ImageBase + site.offset) {
            if (const u64 stub = site.stub.load(std::memory_order_acquire)) {
                g[REG_RIP] = greg_t(stub); // the detour is going in: its stub runs the site
                return;
            }
        }
    }
    if (rip == ImageBase + ReleaseCheck) {
        const auto result = static_cast<std::int32_t>(g[REG_RAX] & 0xffffffff);
        BbHeapSites::NoteReleaseCheck(std::uint32_t(result));
        g[REG_RIP] = greg_t(ImageBase + (result < 0 ? 0xbb5fc7 : ReleaseCheck + 2));
        return;
    }
    if (rip == ImageBase + CollectorEntry) {
        const u64 allocator = u64(g[REG_RDI]);
        for (std::size_t i = 0; i < BbStats::range_allocators.size(); ++i) {
            u64 expected = 0;
            if (BbStats::range_allocators[i].load(std::memory_order_relaxed) == allocator ||
                BbStats::range_allocators[i].compare_exchange_strong(expected, allocator)) {
                BbStats::range_live[i].store(*reinterpret_cast<const u64*>(allocator + 0xc8),
                                             std::memory_order_relaxed);
                BbStats::range_bytes[i].store(*reinterpret_cast<const u64*>(allocator + 0xd0),
                                              std::memory_order_relaxed);
                break;
            }
        }
        g[REG_RSP] -= 8; // push rbp
        *reinterpret_cast<u64*>(g[REG_RSP]) = u64(g[REG_RBP]);
        g[REG_RIP] = greg_t(ImageBase + CollectorEntry + 1);
        return;
    }
    for (const u64 site : HeapAllocReturns) {
        if (rip != ImageBase + site) {
            continue;
        }
        const u64 address = u64(g[REG_RAX]);
        const u64 size = *reinterpret_cast<const u64*>(u64(g[REG_RBP]) - 0x50);
        if (address != 0 && size != 0 && size < (u64(1) << 32)) {
            on_range_allocated(address, size);
        }
        g[REG_RBX] = g[REG_RAX];
        g[REG_RIP] = greg_t(ImageBase + site + sizeof(HeapAllocReturnCode));
        return;
    }
    if (rip == ImageBase + AllocReturn) {
        if (const u64 node = u64(g[REG_RAX])) {
            const u64 start = *reinterpret_cast<const u64*>(node + 8);
            const u64 next = *reinterpret_cast<const u64*>(node + 0x18);
            const u64 end = next ? *reinterpret_cast<const u64*>(next + 8) : start;
            if (end > start && end - start < (u64(1) << 32)) {
                on_range_allocated(start, end - start);
            }
        }
        g[REG_RSP] += 0x18;
        g[REG_RIP] = greg_t(ImageBase + AllocReturn + sizeof(AllocReturnCode));
        return;
    }
    for (std::size_t i = 0; i < MemcpySites.size(); ++i) {
        const auto& site = MemcpySites[i];
        if (rip != ImageBase + site.offset) {
            continue;
        }
        hits[i].fetch_add(1, std::memory_order_relaxed);
        // The call, to LoaderCopy instead of the game's memcpy: push the return address, jump.
        const u64 ret = ImageBase + site.offset + 5;
        g[REG_RSP] -= 8;
        *reinterpret_cast<u64*>(g[REG_RSP]) = ret;
        g[REG_RIP] = greg_t(reinterpret_cast<u64>(&LoaderCopy));
        return;
    }
    if (previous_action.sa_flags & SA_SIGINFO) {
        if (previous_action.sa_sigaction) {
            previous_action.sa_sigaction(sig, info, context);
            return;
        }
    } else if (previous_action.sa_handler != SIG_DFL && previous_action.sa_handler != SIG_IGN &&
               previous_action.sa_handler) {
        previous_action.sa_handler(sig);
        return;
    }
    signal(SIGTRAP, SIG_DFL);
    raise(SIGTRAP);
}

bool Patch(u64 address, const unsigned char* expected, std::size_t size) {
    unsigned char bytes[16]{};
    if (!BbPortable::ReadSelf(bytes, reinterpret_cast<const void*>(address), size) ||
        std::memcmp(bytes, expected, size) != 0) {
        return false;
    }
    const long page_size = sysconf(_SC_PAGESIZE);
    void* page = reinterpret_cast<void*>(address & ~u64(page_size - 1));
    const std::size_t span = (address + size) - reinterpret_cast<u64>(page);
    if (mprotect(page, span, PROT_READ | PROT_WRITE | PROT_EXEC) != 0) {
        return false;
    }
    __atomic_store_n(reinterpret_cast<unsigned char*>(address), static_cast<unsigned char>(0xcc),
                     __ATOMIC_SEQ_CST);
    mprotect(page, span, Core::GuestCodeProtection());
    return true;
}

} // namespace

void Install(RangeCallback on_gpu_range_allocated, RangeCallback on_range_freed,
             RangeCallback on_range_allocated_block) {
    on_range_dead = on_range_freed;
    on_range_live = on_range_allocated_block;
    static std::atomic<bool> installed{false};
    if (installed.exchange(true)) {
        return;
    }
    struct sigaction action {};
    action.sa_sigaction = OnTrap;
    action.sa_flags = SA_SIGINFO | SA_NODEFER;
    sigemptyset(&action.sa_mask);
    sigaction(SIGTRAP, &action, &previous_action);
    // Only the PC memory model needs notifications about GPU allocations and resource copies.
    // The remaining hooks (including the loading/cutscene fixes) are shared by both models.
    const bool memory_hooks =
        on_gpu_range_allocated != nullptr && !bb_env("BB_EXP_NO_MEMORY_HOOKS");
    int patched = 0;
    for (const auto& site : MemcpySites) {
        patched += memory_hooks && Patch(ImageBase + site.offset, site.code, sizeof(site.code));
    }
    on_range_allocated = on_gpu_range_allocated;
    const bool allocator =
        memory_hooks && Patch(ImageBase + AllocReturn, AllocReturnCode, sizeof(AllocReturnCode));
    // BB_HEAP_HOOKS=0 (bisecting): the heap's ranges are not reported.
    const char* heap_env = bb_env("BB_HEAP_HOOKS");
    const bool heap_hooks = memory_hooks && (!heap_env || heap_env[0] != '0');
    if (memory_hooks && BbStats::enabled) {
        Patch(ImageBase + CollectorEntry, CollectorEntryCode, sizeof(CollectorEntryCode));
    }
    int heap_sites = 0;
    for (const u64 site : HeapAllocReturns) {
        if (!heap_hooks) {
            break;
        }
        heap_sites += Patch(ImageBase + site, HeapAllocReturnCode, sizeof(HeapAllocReturnCode));
    }
    if (memory_hooks) {
        std::printf("Guest hooks: %d of %zu resource copy sites tell the GPU side of their writes\n",
                    patched, MemcpySites.size());
        std::printf("Guest hooks: GPU heap allocations seen at %d of %zu call sites\n", heap_sites,
                    HeapAllocReturns.size());
        std::printf("Guest hooks: GPU memory allocator %s\n",
                    allocator ? "tells the GPU side of new ranges" : "not found");
    } else {
        std::printf("Guest hooks: none for the GPU memory (this memory model keeps copies)\n");
    }
    // Native detours over the traps just placed (BB_GUEST_HOOK_TRAPS=1 keeps the traps).
    const char* traps_env = bb_env("BB_GUEST_HOOK_TRAPS");
    if (!(traps_env && traps_env[0] == '1') && (stub_page = MapStubPage()) == nullptr) {
        std::printf("Guest hooks: no memory near the game's code: the aspect ratio, HUD, mouse look "
                    "and render fixes are off\n");
    }
    if (!(traps_env && traps_env[0] == '1') && stub_page != nullptr) {
        int detours = 0;
        for (const auto& site : MemcpySites) {
            if (reinterpret_cast<const u8*>(ImageBase + site.offset)[0] == 0xcc) {
                detours += InstallCallDetour(site.offset);
            }
        }
        for (auto& site : native_sites) {
            if (reinterpret_cast<const u8*>(ImageBase + site.offset)[0] == 0xcc) {
                detours += InstallDetour(site);
            }
        }
        const int aspect_sites = InstallCameraAspectHooks();
        std::printf("Guest hooks: render job slot leak fix %s\n",
                    InstallJobSlotFix() ? "on" : "not installed");
        std::printf("Guest hooks: job allocation failure probe %s\n",
                    InstallJobAllocationProbe() ? "on" : "not installed");
        std::printf("Guest hooks: render resource cache size fix %s\n",
                    InstallResourceCacheSize() ? "on" : "not installed");
        std::printf("Guest hooks: GraphicsPrivate heap growing on demand %s\n",
                    InstallHeapGrow() ? "on" : "off");
        if (bb_env("BB_HEAP_TRACE")) {
            std::printf("Guest hooks: heap trace %s\n", InstallHeapTrace() ? "on" : "not installed");
        }
        std::printf("Guest hooks: FX initialization context race fix %s\n",
                    InstallFxInitClear() ? "on" : "off (BB_FX_INIT_CLEAR=0 or unmatched code)");
        std::printf("Guest hooks: blur-kernel sample count clamp %s\n",
                    InstallKernelSampleClamp() ? "on" : "not installed");
        std::printf("Guest hooks: game effects switched live %s\n",
                    InstallHotEffects() ? "on (no restart)" : "off (startup patches)");
        if (InstallFreeCameraProbe()) {
            std::printf("Guest hooks: free camera probe on (the mouse drives it)\n");
        }
        const int marker_places = InstallMarkerHooks();
        if (const char* env = bb_env("BB_HUD_ANCHOR"); env && env[0] == '0') hud_anchor_on = false;
        const bool hud_hooks = InstallViewportHook() && InstallHudHook();
        std::printf("Guest hooks: GX command heap guard %s\n",
                    InstallGxNodeGuard() ? "in (#162)" : "not installed (other code there)");
        std::printf("Guest hooks: GX occlusion query guard %s\n",
                    !InstallGxQueryGuard() ? "not installed (BB_GX_QUERY_GUARD=0 or other code there)"
                    : gx_query_observer    ? "in (a query without a handle is created again or not begun)"
                                           : "in (a query without a handle is not begun; no creation observer)");
        std::printf("Guest hooks: the output's aspect ratio on the fly: the camera at %d of %zu places, "
                    "the HUD's markers at %d of %zu, its groups %s\n", aspect_sites, CameraParamStores.size(),
                    marker_places, MarkerSites.size(), hud_hooks ? "at the edges" : "in the middle (no hook)");
        // BB_NATIVE_MOUSE=0: the mouse drives the right stick instead (runtime_pad.c); the detour
        // still scales the stick's turn (camera speeds).
        const bool look_hook = InstallLookHook();
        if (const char* env = bb_env("BB_NATIVE_MOUSE"); !(env && env[0] == '0')) {
            look_installed.store(look_hook, std::memory_order_release);
            std::printf("Guest hooks: native mouse look %s\n",
                        look_installed.load() ? "on (the mouse turns the camera itself)"
                                              : "not available (the mouse drives the right stick)");
        }
        if (const char* env = bb_env("BB_RE_ADVANCE"); env && env[0] == '1') {
            std::printf("Guest hooks: UI advance probe %s\n", InstallEntryProbe(0x20f00) ? "on" : "not installed");
        }
        if (const char* env = bb_env("BB_RE_ENTRY"); env && env[0]) {
            for (const char* p = env; *p;) {
                char* end = nullptr;
                const u64 offset = std::strtoull(p, &end, 16);
                if (end == p) break;
                std::printf("Guest hooks: entry probe at %#llx %s\n", (unsigned long long)offset,
                            InstallEntryProbe(offset) ? "on" : "not installed (other code there)");
                p = *end ? end + 1 : end;
            }
        }
        if (const char* env = bb_env("BB_RE_CAMERA"); env && env[0] == '1') {
            std::printf("Guest hooks: camera probe %s\n", InstallCameraProbe() ? "on" : "not installed");
        }
        if (const char* env = bb_env("BB_RE_CAMSTICK"); env && env[0] == '1') {
            std::printf("Guest hooks: stick getter probe %s\n", InstallStickProbe() ? "on" : "not installed");
        }
        if (const char* env = bb_env("BB_RE_ANALOG"); env && env[0] == '1') {
            std::printf("Guest hooks: analog getter probe %s\n",
                        InstallAnalogProbe() ? "on" : "not installed (other code there)");
        }
        mprotect(stub_page, StubPageSize, PROT_READ | PROT_EXEC);
        std::printf("Guest hooks: %d native detours (no traps; BB_GUEST_HOOK_TRAPS=1: int3), %zu of %zu "
                    "stub bytes\n", detours, stub_used, StubPageSize);
    }
    BbHeapSites::Install();
    BbEngineFixes::Install();
    if (bb_env_on("BB_HEAP_SITES")) {
        Patch(ImageBase + ReleaseCheck, ReleaseCheckCode, sizeof(ReleaseCheckCode));
    }
}

} // namespace BbGuestHooks
