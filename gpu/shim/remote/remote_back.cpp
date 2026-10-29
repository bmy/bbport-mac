// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the native GPU process, bb-gpu (docs/macos-native-gpu.md). It reserves the guest's
// address ranges, maps the game process's memory pool at the same addresses, and runs the GPU
// core (Liverpool, caches, shader recompiler, presenter, window) for the game process, which
// forwards to it (remote_front.cpp). The runtime's GPU interface (runtime_memory_*, clocks) is
// served here from bb-gpu's own copy of the guest's mapping table (bb_gpu_main.cpp exports it).
//
// bbport 0.7's memory model (the GPU uses the game's memory in place): BbGuestMemory::Install
// runs here and sets its chunk allocator in this file (SetChunkAllocator); the game process's
// runtime, which places the game's direct memory, calls it over ControlBlock::protect
// (MsgChunkAlloc, remote_front.cpp). A chunk is a host memory import of this process's own view
// of the shared pool (BackingPointer), so the game's mappings stay the pool's and the mirror
// stays right; dma-buf chunks and private regions (memory only one process has) are refused.
#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <cerrno>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <execinfo.h>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <sys/mman.h>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <vector>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/event.h>
#else
#include <sys/prctl.h>
#endif

#include "../bbgpu.h"
#include <SDL3/SDL.h>
#include "bblayer_write_traps.h"
#include "bbport_guest_memory.h"
#include "bbport_overlay.h"
#include "bbport_settings.h"
#include "bbport_portable.h"
#include "bbport_toggles.h"
#include "common/signal_context.h"
#include "common/slot_vector.h"
#include "common/thread.h"
#include "core/memory.h"
#include "core/signals.h"
#include "cp/command_processor.h"
#include "platform/interrupts.h"
#include "platform/videoout/videoout.h"
#include "remote/bb_control.h"
#include "remote/bb_memory_mirror.h"
#include "remote/bb_vma_table.h"
#include "shader/ir/passes/srt.h"
#include "remote/bb_protocol.h"
#include "remote/bb_remote.h"
#include "remote/bb_trap_copy.h"
#include "render/presenter.h"
#include "render/rasterizer.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<BbCp::CommandProcessor> liverpool;

namespace BbRemote::Back {

namespace VO = BbVideoOut;
static_assert(sizeof(VO::FlipStatus) == FlipStatusBytes);
static_assert(sizeof(VO::VblankStatus) == VblankStatusBytes);
static_assert(sizeof(VO::ResolutionStatus) == ResolutionBytes);
static_assert(sizeof(VO::BufferAttribute) == BufferAttributeBytes);
static_assert(sizeof(VO::DisplayBuffer) == VideoOutBufferBytes);
static_assert(sizeof(VO::BufferAttributeGroup) == AttributeGroupBytes);
static_assert(sizeof(ProtectArgs) <= MaxAskPayload);

void ForwardProtect(const ProtectArgs& args);

namespace {

// ---- state ------------------------------------------------------------------------------------

struct State {
    int pool_fd = -1;
    u64 pool_bytes = 0;
    ControlBlock* control = nullptr;
    std::unique_ptr<Channel> channel, protect, fresh, locked;
    MemoryMirror guest, low;
    VmaTable table;
    std::mutex hook_mutex;
    GpuRange hook_map = nullptr, hook_unmap = nullptr, hook_invalidate = nullptr;
    GpuRange hook_note_write = nullptr, hook_cpu_write = nullptr;
    // Clocks of the game process (HelloArgs).
    u64 start_ns = 0, tsc_sample = 0, tsc_sample_ns = 0, tsc_hz = 1'000'000'000;
    std::atomic<u64> flip_version{0};
    std::mutex inline_mutex;
    std::set<std::vector<u32>> inline_commands; ///< host command words (init sequences), kept
    u32 published_overlay = 2;
    u8* backing = nullptr; ///< the whole pool (runtime_memory_backing_pointer)
    // BbGuestMemory::Install's settings (runtime_memory_set_guest_chunk_*, _lazy).
    std::atomic<ChunkAlloc> chunk_alloc{nullptr};
    std::atomic<ChunkFree> chunk_free{nullptr};
    std::atomic<ChunkImport> chunk_private{nullptr};
    std::atomic<int> chunk_whole{0}, guest_lazy{0};
    std::string user_dir = "user";
};
State g;

/// bbport: the write fault this thread is answering (WriteFault): every protection decided
/// meanwhile is asked of the faulting thread, which applies it while it waits (ForwardProtect).
thread_local Channel::Reply* fault_reply = nullptr;

/// Sets fault_reply for a scope.
struct FaultScope {
    explicit FaultScope(Channel::Reply* reply) { fault_reply = reply; }
    ~FaultScope() { fault_reply = nullptr; }
    FaultScope(const FaultScope&) = delete;
    FaultScope& operator=(const FaultScope&) = delete;
};

/// Write traps (runtime_memory_trap, bbport 0.5): this process's copy of the game's reasons
/// (bb_trap_copy.h), for its GPU code's questions, set by its own trap calls and cleared where the
/// game's mappings change. Every owner of a trap (the layer's VRAM copies, images, occlusion and
/// VRAM-data reads) is GPU code, so it is this process's: the copy differs from the game's only
/// when a trap crosses a remap on its way (set here for the old mapping, applied there to the new
/// one). A fault brings the game's reasons for its page (FaultArgs::reasons): any nobody here
/// holds is lifted, and the fault goes through the same handlers as in one process (WriteFault).
constexpr u64 TrapLimit = TrapCopy::Limit;
TrapCopy traps;
/// runtime_memory_trap_stats: changes of the copy here, and the game process applying them.
std::atomic<u64> trap_wait_ns{0}, trap_apply_ns{0}, trap_calls{0}, trap_bytes{0};

void TrapForget(u64 start, u64 end) {
    traps.Forget(start, end);
}

template <typename T>
T Payload(const Message& message) {
    T value{};
    std::memcpy(&value, message.data, std::min<std::size_t>(sizeof(T), message.size));
    return value;
}

u64 MonotonicNs() {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return u64(now.tv_sec) * 1'000'000'000ull + u64(now.tv_nsec);
}

/// A range of the guest's own address space (messages naming GPU ranges).
bool GuestRange(u64 address, u64 size) {
    return size && address >= GuestBegin && address < GuestEnd && size <= GuestEnd - address;
}

MemoryMirror& MirrorFor(u64 address) {
    return address >= GuestBegin ? g.guest : g.low;
}

/// Private (executable) memory of the game process: readable zeros here (only inside the
/// mirrored range, so a bad message can't map over this process's own memory).
bool MapZeros(u64 address, u64 size) {
    return MirrorFor(address).MapZeros(address, size);
}

/// Stores every byte once (aligned pieces as single release stores), as runtime_memory.c does
/// for guest-visible writes: labels must never be seen half written or written twice.
void StoreOnce(u8* dst, const u8* src, u64 n) {
    if (n > 64) {
        // Data: after what came before it, and before what comes after (a label).
        std::atomic_thread_fence(std::memory_order_release);
        std::memcpy(dst, src, n);
        std::atomic_thread_fence(std::memory_order_release);
        return;
    }
    while (n) {
        const auto at = reinterpret_cast<uintptr_t>(dst);
        if (n >= 8 && !(at & 7)) {
            u64 v;
            std::memcpy(&v, src, 8);
            __atomic_store_n(reinterpret_cast<u64*>(dst), v, __ATOMIC_RELEASE);
            dst += 8, src += 8, n -= 8;
        } else if (n >= 4 && !(at & 3)) {
            u32 v;
            std::memcpy(&v, src, 4);
            __atomic_store_n(reinterpret_cast<u32*>(dst), v, __ATOMIC_RELEASE);
            dst += 4, src += 4, n -= 4;
        } else if (n >= 2 && !(at & 1)) {
            u16 v;
            std::memcpy(&v, src, 2);
            __atomic_store_n(reinterpret_cast<u16*>(dst), v, __ATOMIC_RELEASE);
            dst += 2, src += 2, n -= 2;
        } else {
            __atomic_store_n(dst, *src, __ATOMIC_RELEASE);
            ++dst, ++src, --n;
        }
    }
}

// ---- the VideoOut port --------------------------------------------------------------------------

VoState Snapshot(s32 handle) {
    VoState state{};
    VO::Port* port = VO::Remote::GetPort(handle);
    if (!port) {
        return state;
    }
    std::scoped_lock lock{port->status_mutex};
    state.version = g.flip_version.load(std::memory_order_acquire);
    std::memcpy(state.flip_status, &port->flip_status, FlipStatusBytes);
    std::memcpy(state.slots, port->buffers.data(), sizeof(state.slots));
    std::memcpy(state.groups, port->groups.data(), sizeof(state.groups));
    std::memcpy(state.resolution, &port->resolution, ResolutionBytes);
    state.is_open = port->is_open ? 1 : 0;
    return state;
}

void ReplyVo(auto&& reply, s32 result, s32 handle) {
    VoReply out{};
    out.result = result;
    out.state = Snapshot(handle);
    reply(&out, sizeof(out));
}

// ---- messages from the game process -----------------------------------------------------------

void Hello(const Message& message, auto&& reply) {
    const auto args = Payload<HelloArgs>(message);
    HelloReply out{ProtocolVersion, u32(sysconf(_SC_PAGESIZE)), 0, 0};
    if (args.protocol != ProtocolVersion) {
        out.error = 1;
    } else if (args.guest_begin != GuestBegin || args.guest_end != GuestEnd ||
               args.low_begin != LowBegin || args.low_end != LowEnd ||
               args.pool_size != g.pool_bytes) {
        out.error = 2; // built from different sources
    } else {
        for (const auto& [begin, end] : g.guest.Taken()) {
            if (out.taken_count < MaxTakenRanges && begin < GuestEnd) {
                out.taken[out.taken_count][0] = begin;
                out.taken[out.taken_count][1] = std::min(end, GuestEnd);
                ++out.taken_count;
            }
        }
        g.start_ns = args.start_monotonic_ns;
        g.tsc_sample = args.tsc_sample;
        g.tsc_sample_ns = args.tsc_sample_monotonic_ns;
        g.tsc_hz = args.tsc_frequency ? args.tsc_frequency : 1'000'000'000;
    }
    reply(&out, sizeof(out));
}

void Init(const Message& message, auto&& reply) {
    const auto args = Payload<InitArgs>(message);
    std::string title{args.title, strnlen(args.title, sizeof(args.title))};
    std::string serial{args.serial, strnlen(args.serial, sizeof(args.serial))};
    std::string user_dir{args.user_dir, strnlen(args.user_dir, sizeof(args.user_dir))};
    const BbGpuConfig config{title.c_str(), serial.c_str(), user_dir.c_str(), args.sdk_version,
                             args.psf_attributes, args.width, args.height};
    g.user_dir = user_dir.empty() ? "user" : user_dir;
    const int result = bbgpu_init(&config);
    if (result == 0) {
        g.control->state.gpu_ready.store(1, std::memory_order_release);
        std::printf("GPU process: native GPU core ready\n");
    }
    const ResultReply out{result, 0};
    reply(&out, sizeof(out));
}

// The game image's data: in a native process macOS keeps its shared library region over the
// low region (0x2f0000000-0xfc0000000 on macOS 27), so it is mapped elsewhere and the code that
// reads it directly (the SRT walker) is pointed there.
struct Alias {
    u64 begin, end, at;
};
constexpr std::size_t MaxAliases = 16;
std::array<Alias, MaxAliases> aliases{};
std::atomic<std::size_t> alias_count{0};

/// The game image's data where it is mapped at its own address here (MapShared outside the
/// table): the walker reads tables there, compute rings may sit there.
std::array<std::pair<u64, u64>, MaxAliases> shared_in_place{};
std::atomic<std::size_t> shared_in_place_count{0};

bool InSharedInPlace(u64 address, u64 size) {
    const std::size_t n = shared_in_place_count.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
        const auto [begin, end] = shared_in_place[i];
        if (address >= begin && address <= end && size <= end - address) {
            return true;
        }
    }
    return false;
}

/// Guest memory as this process sees it: its copy of the table and the game image's data.
bool GuestCovered(u64 address, u64 size) {
    return g.table.Covered(address, size) || InSharedInPlace(address, size);
}

/// Where the bytecode SRT walker may read in this process: guest memory (GuestCovered) and the
/// game image data's second mapping. Nothing else of this process.
bool InAlias(u64 address, u64 size) {
    const std::size_t n = alias_count.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
        const u64 begin = aliases[i].at, end = begin + (aliases[i].end - aliases[i].begin);
        if (address >= begin && address <= end && size <= end - address) {
            return true;
        }
    }
    return false;
}

u64 TranslateGuest(u64 address) {
    const std::size_t n = alias_count.load(std::memory_order_acquire);
    for (std::size_t i = 0; i < n; ++i) {
        if (address >= aliases[i].begin && address < aliases[i].end) {
            return aliases[i].at + (address - aliases[i].begin);
        }
    }
    return address;
}

bool MapAlias(const MapArgs& args) {
    const std::size_t n = alias_count.load(std::memory_order_relaxed);
    if (n == MaxAliases) {
        return false;
    }
    void* at = mmap(nullptr, args.size, PROT_READ | PROT_WRITE, MAP_SHARED, g.pool_fd,
                    static_cast<off_t>(args.offset));
    if (at == MAP_FAILED) {
        return false;
    }
    aliases[n] = {args.address, args.address + args.size, reinterpret_cast<u64>(at)};
    alias_count.store(n + 1, std::memory_order_release);
#ifndef ARCH_X86_64
    Shader::srt_guest_translate = TranslateGuest; // the bytecode SRT walker reads it directly
#endif
    std::printf("GPU process: the game's data at %#llx+%#llx is read at %p here\n",
                static_cast<unsigned long long>(args.address),
                static_cast<unsigned long long>(args.size), at);
    return true;
}

/// The control block (the channel itself) sits right after the guest range, inside the guest
/// mirror's reservation: no mapping message may touch it.
bool TouchesControl(u64 address, u64 size) {
    return address + size < address ||
           (address < ControlAddress + ControlBytes && ControlAddress < address + size);
}

void MapMemory(const MapArgs& args) {
    if (TouchesControl(args.address, args.size)) {
        std::fprintf(stderr, "GPU process: refused a mapping over the control block\n");
        return;
    }
    if (!(args.flags & MapUnshared) &&
        (args.offset > g.pool_bytes || args.size > g.pool_bytes - args.offset)) {
        std::fprintf(stderr, "GPU process: refused a mapping past the end of the shared pool\n");
        return;
    }
    TrapForget(args.address, args.address + args.size); // a new mapping starts without traps
    if (args.kind == MapShared && args.address < GuestBegin &&
        MirrorFor(args.address).Overlaps(args.address, args.size)) {
        if (!MapAlias(args)) {
            std::fprintf(stderr, "GPU process: the game's data at %#llx not shared\n",
                         static_cast<unsigned long long>(args.address));
        }
        return;
    }
    const bool ok = (args.flags & MapUnshared)
                        ? MapZeros(args.address, args.size)
                        : MirrorFor(args.address).Map(args.address, args.size, g.pool_fd,
                                                      args.offset);
    if (!ok) {
        static int reported = 0;
        if (reported++ < 8) {
            std::fprintf(stderr, "GPU process: guest mapping %#llx+%#llx not mirrored\n",
                         static_cast<unsigned long long>(args.address),
                         static_cast<unsigned long long>(args.size));
        }
        return;
    }
    if (args.kind == MapShared) {
        // The game's own data: not in the guest's mapping table.
        const std::size_t n = shared_in_place_count.load(std::memory_order_relaxed);
        if (n < shared_in_place.size()) {
            shared_in_place[n] = {args.address, args.address + args.size};
            shared_in_place_count.store(n + 1, std::memory_order_release);
        }
        return;
    }
    g.table.Map(args.address, args.address + args.size,
                VmaEntry{args.address + args.size, args.kind, args.prot, args.type, args.offset});
    GpuRange hook;
    {
        std::scoped_lock lock{g.hook_mutex};
        hook = g.hook_map;
    }
    if (hook) {
        hook(args.address, args.size);
    }
}

void UnmapMemory(const RangeArgs& args) {
    if (TouchesControl(args.address, args.size)) {
        std::fprintf(stderr, "GPU process: refused an unmapping of the control block\n");
        return;
    }
    GpuRange hook;
    {
        std::scoped_lock lock{g.hook_mutex};
        hook = g.hook_unmap;
    }
    // The GPU forgets the range first (runtime_memory.c direct_unmap), then it disappears.
    if (hook) {
        for (const auto& [address, size] : g.table.Pieces(args.address, args.address + args.size)) {
            hook(address, size);
        }
    }
    g.table.Unmap(args.address, args.address + args.size);
    MirrorFor(args.address).Unmap(args.address, args.size);
    TrapForget(args.address, args.address + args.size);
}

/// A range whose GPU memory changed under the same mappings (a heap piece's chunk made or given
/// back, runtime_memory_commit and _decommit): the GPU forgets and maps it again, as the runtime's
/// HOOK_UNMAP and HOOK_MAP there. Only what is mapped now.
void GpuRemap(const RangeArgs& args) {
    if (!GuestRange(args.address, args.size)) {
        return;
    }
    GpuRange map, unmap;
    {
        std::scoped_lock lock{g.hook_mutex};
        map = g.hook_map;
        unmap = g.hook_unmap;
    }
    for (const auto& [address, size] : g.table.Pieces(args.address, args.address + args.size)) {
        if (unmap) {
            unmap(address, size);
        }
        if (map) {
            map(address, size);
        }
    }
}

/// The game's hooks' ranges (Game::CoreServices: fresh, dead, live), for the buffer cache here.
void GuestHookRange(u32 type, const RangeArgs& args) {
    auto* rasterizer = Core::Memory::Instance()->GetRasterizer();
    if (!rasterizer || !GuestRange(args.address, args.size)) {
        return;
    }
    auto& cache = rasterizer->GetBufferCache();
    if (type == MsgFreshRange) {
        cache.NoteFreshRange(args.address, args.size);
    } else if (type == MsgDeadRange) {
        cache.LayerDomainDead(args.address, args.size);
    } else {
        cache.LayerDomainAdd(args.address, args.size);
    }
}

/// A trophy unlocked in the game process: its pop-up in this process's overlay. The icon only as
/// a trophy package's own image file (TROPnnn.PNG).
void ShowTrophy(const TrophyArgs& in) {
    TrophyArgs args = in;
    args.name[sizeof(args.name) - 1] = 0;
    args.icon[sizeof(args.icon) - 1] = 0;
    std::filesystem::path icon{args.icon};
    const std::string file = icon.filename().string();
    std::error_code error;
    if (file.size() != 11 || !file.starts_with("TROP") || !file.ends_with(".PNG") ||
        !std::all_of(file.begin() + 4, file.begin() + 7, [](char c) { return c >= '0' && c <= '9'; }) ||
        !std::filesystem::is_regular_file(icon, error)) {
        icon.clear();
    }
    const float delay = std::isfinite(args.delay) ? std::clamp(args.delay, 0.0f, 120.0f) : 0.0f;
    BbOverlay::ShowTrophy(args.grade, args.name, icon, delay);
}

/// What BbGuestMemory::Install set here, for the game process's runtime (MsgGuestMemory).
GuestMemoryReply GuestMemory() {
    GuestMemoryReply out{};
    out.allocator = g.chunk_alloc.load(std::memory_order_acquire) ? 1 : 0;
    out.host_import = BbGuestMemory::HostImported() ? 1 : 0;
    out.whole = g.chunk_whole.load(std::memory_order_relaxed) ? 1 : 0;
    out.lazy = g.guest_lazy.load(std::memory_order_relaxed) ? 1 : 0;
    out.free_chunks = g.chunk_free.load(std::memory_order_acquire) ? 1 : 0;
    out.private_regions = g.chunk_private.load(std::memory_order_acquire) ? 1 : 0;
    return out;
}

void CallHook(GpuRange State::*which, const RangeArgs& args) {
    GpuRange hook;
    {
        std::scoped_lock lock{g.hook_mutex};
        hook = g.*which;
    }
    if (hook) {
        hook(args.address, args.size);
    }
}

/// A ucontext for the in-process fault handlers (Core::Signals): the guest's instruction and
/// whether it wrote; no stack (the guest's is the game process's).
struct FaultContext {
    ucontext_t uc{};
#ifdef __APPLE__
    std::remove_pointer_t<decltype(ucontext_t{}.uc_mcontext)> mc{};
#endif
    FaultContext(u64 rip, bool write) {
#ifdef __APPLE__
        uc.uc_mcontext = &mc;
#if defined(__aarch64__)
        mc.__es.__esr = write ? 1u << 6 : 0; // WnR (Common::IsWriteError)
#else
        mc.__es.__err = write ? 2 : 0;
#endif
#else
        uc.uc_mcontext.gregs[REG_ERR] = write ? 2 : 0;
#endif
        Common::Gregs(&uc)[REG_RIP] = greg_t(rip);
    }
};

/// Reasons the game's table holds for the page that nobody here does (see TrapLimit): lifted
/// there by the faulting thread (fault_reply). bbport (protocol 12): the lift carries the copy's
/// order when it was checked (TrapCopy::Stale), and the game drops it on a page a newer change
/// already reached (runtime_memory_trap_seq): a trap an owner sets meanwhile stays, whichever
/// arrives first. Protocol 11 held the copy's lock across the faulting thread's apply instead,
/// which can wait for the runtime's lock: every MemoryTrap here waited with it. Dropped, the
/// stale reasons fault again and are lifted then.
bool LiftStaleTraps(u64 address, unsigned game_reasons) {
    if (!fault_reply) {
        return false;
    }
    const TrapCopy::Lift lift = traps.Stale(address, game_reasons);
    if (!lift.reasons) {
        return false;
    }
    const u64 t0 = MonotonicNs();
    ForwardProtect(
        ProtectArgs{address & ~u64(4095), 4096, ProtectTrap | ProtectLift | lift.reasons, 0, lift.seq});
    trap_apply_ns.fetch_add(MonotonicNs() - t0, std::memory_order_relaxed);
    trap_calls.fetch_add(1, std::memory_order_relaxed);
    trap_bytes.fetch_add(4096, std::memory_order_relaxed);
    return true;
}

void WriteFault(const Message& message, Channel::Reply& reply) {
    const auto args = Payload<FaultArgs>(message);
    // bbport: what the handlers decide is applied by the faulting thread while it waits (asked
    // over this call's slot), each before the handler goes on, as in one process; with the
    // runtime's lock held there too (args.lock_held: then on FaultThread). See FaultReply.
    const FaultScope scope{reply.CanAsk() ? &reply : nullptr};
    const bool lifted = LiftStaleTraps(args.address, args.reasons);
    int handled = 0;
    if (Core::Memory::Instance()->GetRasterizer()) {
        // As bbgpu_handle_fault in one process: the registered handlers in their order (the
        // layer's VRAM copy traps, then page tracking: page_watch.cpp's OnFault).
        FaultContext context{args.rip, args.is_write != 0};
        handled = Core::Signals::Instance()->DispatchAccessViolation(
                      &context.uc, reinterpret_cast<void*>(args.address))
                      ? 1
                      : 0;
    }
    // Only stale reasons held the page (nobody here claims it): the write is retried.
    const FaultReply out{handled || lifted ? 1 : 0, 0};
    reply(&out, sizeof(out));
}

std::span<const u32> InlineCommands(const u32* words, u32 count) {
    std::scoped_lock lock{g.inline_mutex};
    const auto [it, inserted] = g.inline_commands.emplace(words, words + count);
    if (inserted && g.inline_commands.size() == 1024) {
        std::fprintf(stderr, "GPU process: 1024 distinct host command buffers kept\n");
    }
    return {it->data(), it->size()};
}

void SubmitGfx(const Message& message) {
    SubmitGfxArgs args{};
    if (message.size < sizeof(args)) {
        return;
    }
    std::memcpy(&args, message.data, sizeof(args));
    const auto* words = reinterpret_cast<const u32*>(static_cast<const u8*>(message.data) +
                                                     sizeof(SubmitGfxArgs));
    const u32 available = (message.size - sizeof(SubmitGfxArgs)) / 4;
    u32 used = 0;
    // force_copy (0.6's FX initialization clears): the command processor copies the words at
    // submit, so inline ones are used from the message and not kept.
    const bool copy = args.force_copy != 0;
    if (copy && ((args.dcb_dwords && !args.inline_dcb) || (args.ccb_dwords && !args.inline_ccb))) {
        std::fprintf(stderr, "GPU process: a copied submission must travel inline; refused\n");
        return;
    }
    const auto span = [&](u64 address, u32 dwords, u32 is_inline) -> std::span<const u32> {
        if (!dwords) {
            return {};
        }
        if (is_inline) {
            if (dwords > available - used) { // (no wrap-around: used <= available)
                return {};
            }
            const std::span<const u32> kept =
                copy ? std::span<const u32>{words + used, dwords} : InlineCommands(words + used, dwords);
            used += dwords;
            return kept;
        }
        return {reinterpret_cast<const u32*>(address), dwords};
    };
    const auto dcb = span(args.dcb, args.dcb_dwords, args.inline_dcb);
    const auto ccb = span(args.ccb, args.ccb_dwords, args.inline_ccb);
    liverpool->SubmitGfx(dcb, ccb, copy);
}

void Handle(const Message& message, auto&& reply) {
    switch (message.type) {
    case MsgHello:
        Hello(message, reply);
        break;
    case MsgInit:
        Init(message, reply);
        break;
    case MsgShutdown: {
        const ResultReply out{0, 0};
        reply(&out, sizeof(out));
        std::fflush(stdout);
        std::_Exit(0);
    }
    case MsgMapMemory:
        MapMemory(Payload<MapArgs>(message));
        break;
    case MsgUnmapMemory:
        UnmapMemory(Payload<RangeArgs>(message));
        break;
    case MsgProtectMemory: {
        const auto args = Payload<MirrorProtectArgs>(message);
        g.table.Protect(args.address, args.address + args.size, args.prot, args.type);
        break;
    }
    case MsgInvalidate: {
        const auto args = Payload<RangeArgs>(message);
        Core::Memory::Instance()->InvalidateMemory(args.address, args.size);
        break;
    }
    case MsgNoteWrite:
        CallHook(&State::hook_note_write, Payload<RangeArgs>(message));
        break;
    case MsgCpuWrite:
        CallHook(&State::hook_cpu_write, Payload<RangeArgs>(message));
        break;
    case MsgSettingsChanged:
        BbSettings::ReloadLive();
        std::printf("GPU process: settings from the game's menu applied\n");
        break;
    case MsgGuestMemory: {
        const GuestMemoryReply out = GuestMemory();
        reply(&out, sizeof(out));
        break;
    }
    case MsgFreshRange:
    case MsgDeadRange:
    case MsgLiveRange:
        GuestHookRange(message.type, Payload<RangeArgs>(message));
        break;
    case MsgGpuRemap:
        GpuRemap(Payload<RangeArgs>(message));
        break;
    case MsgShowTrophy:
        ShowTrophy(Payload<TrophyArgs>(message));
        break;
    case MsgWriteFault:
        WriteFault(message, reply);
        break;

    case MsgSubmitGfx:
        SubmitGfx(message);
        break;
    case MsgSubmitAsc: {
        const auto args = Payload<SubmitAscArgs>(message);
        // Queues are numbered from 1 (DingDong); only mapped ones.
        if (args.vqid == 0 || args.vqid >= BbCp::CommandProcessor::NumTotalQueues ||
            !liverpool->asc_queues.is_allocated(Common::SlotId{args.vqid - 1})) {
            std::fprintf(stderr, "GPU process: submission to compute queue %u refused\n",
                         args.vqid);
            break;
        }
        liverpool->SubmitAsc(args.vqid, {reinterpret_cast<const u32*>(args.address), args.dwords});
        break;
    }
    case MsgSubmitDone:
        liverpool->SubmitDone(Payload<ValueArgs>(message).value);
        break;
    case MsgIsGpuIdle: {
        const ResultReply out{liverpool->IsGpuIdle() ? 1 : 0, 0};
        reply(&out, sizeof(out));
        break;
    }
    case MsgMapComputeQueue: {
        const auto args = Payload<ComputeQueueArgs>(message);
        // As sceGnmMapComputeQueue checks in the game process, and both in guest memory: the
        // command processor writes the read pointer and reads the ring.
        const u64 ring_bytes = u64(args.ring_size_dw) * 4;
        const bool valid = args.ring_size_dw != 0 && std::has_single_bit(args.ring_size_dw) &&
                           args.pipe_id < BbCp::CommandProcessor::NumComputePipes &&
                           args.read_ptr % 4 == 0 && GuestCovered(args.read_ptr, 4) &&
                           GuestCovered(args.ring_base, ring_bytes);
        s32 index = -1;
        if (valid) {
            const auto id = liverpool->asc_queues.insert(
                VAddr(args.ring_base), reinterpret_cast<u32*>(args.read_ptr), args.ring_size_dw,
                args.pipe_id);
            if (id.index + 1 < BbCp::CommandProcessor::NumTotalQueues) {
                *liverpool->asc_queues[id].read_addr = 0;
                index = s32(id.index);
            } else {
                liverpool->asc_queues.erase(id); // queue numbers past the command processor's
            }
        }
        const ResultReply out{index, 0};
        reply(&out, sizeof(out));
        break;
    }
    case MsgUnmapComputeQueue: {
        const Common::SlotId slot{u32(Payload<ValueArgs>(message).value)};
        if (liverpool->asc_queues.is_allocated(slot)) {
            liverpool->asc_queues.erase(slot);
        }
        break;
    }

    case MsgVoOpen: {
        const s32 handle = VO::Remote::Open();
        ReplyVo(reply, handle, handle > 0 ? handle : 1);
        break;
    }
    case MsgVoClose: {
        const auto args = Payload<VoArgs>(message);
        VO::Remote::Close(args.handle);
        const ResultReply out{0, 0};
        reply(&out, sizeof(out));
        break;
    }
    case MsgVoRegisterBuffers: {
        const auto args = Payload<VoRegisterArgs>(message);
        VO::BufferAttribute attribute{};
        std::memcpy(&attribute, args.attribute, sizeof(attribute));
        void* addresses[MaxDisplayBuffers] = {};
        for (u32 i = 0; i < MaxDisplayBuffers; ++i) {
            addresses[i] = reinterpret_cast<void*>(args.addresses[i]);
        }
        // (Display::RegisterBuffers checks the range against the slots too.)
        constexpr s32 ErrorInvalidValue = s32(0x80290001);
        const s32 result =
            args.count < 0 || args.count > s32(MaxDisplayBuffers)
                ? ErrorInvalidValue
                : VO::Remote::RegisterBuffers(args.handle, args.start, addresses, args.count,
                                              &attribute);
        ReplyVo(reply, result, args.handle);
        break;
    }
    case MsgVoUnregisterBuffers: {
        const auto args = Payload<VoArgs>(message);
        ReplyVo(reply, VO::Remote::UnregisterBuffers(args.handle, args.value), args.handle);
        break;
    }
    case MsgVoChangeAttribute: {
        const auto args = Payload<VoAttributeArgs>(message);
        VO::BufferAttribute attribute{};
        std::memcpy(&attribute, args.attribute, sizeof(attribute));
        ReplyVo(reply,
                VO::Remote::ChangeBufferAttribute(args.handle, args.index, &attribute),
                args.handle);
        break;
    }
    case MsgVoSubmitFlip: {
        const auto args = Payload<VoFlipArgs>(message);
        const s32 result =
            VO::Remote::SubmitFlip(args.handle, args.index, s32(args.mode), args.flip_arg);
        ReplyVo(reply, result == 0 ? 1 : 0, args.handle);
        break;
    }
    case MsgVoSubmitEopFlip: {
        const auto args = Payload<VoFlipArgs>(message);
        const s32 result = VO::sceVideoOutSubmitEopFlip(args.handle, u32(args.index), args.mode,
                                                        args.flip_arg, nullptr);
        ReplyVo(reply, result, args.handle);
        break;
    }
    case MsgVoSetFlipRate: {
        const auto args = Payload<VoArgs>(message);
        // The game process applied the frame rate settings (BbFrameRate) already.
        if (auto* port = VO::Remote::GetPort(args.handle)) {
            port->flip_rate = std::clamp(args.value, 0, 15);
        }
        break;
    }
    case MsgVoSetHdr: {
        const auto args = Payload<VoArgs>(message);
        if (auto* port = VO::Remote::GetPort(args.handle)) {
            port->is_hdr = args.value != 0;
        }
        break;
    }
    case MsgVoIsHdrSupported: {
        const ResultReply out{presenter && presenter->IsHDRSupported() ? 1 : 0, 0};
        reply(&out, sizeof(out));
        break;
    }
    case MsgVoSetGamma:
        if (presenter) {
            presenter->SetGamma(std::clamp(Payload<GammaArgs>(message).gamma, 0.1f, 2.0f));
        }
        break;

    case MsgTextInputBegin: {
        const auto args = Payload<TextInputArgs>(message);
        std::string initial{args.initial, strnlen(args.initial, sizeof(args.initial))};
        std::string prompt{args.prompt, strnlen(args.prompt, sizeof(args.prompt))};
        const ResultReply out{bbgpu_text_input_begin(initial.c_str(), prompt.c_str()), 0};
        reply(&out, sizeof(out));
        break;
    }
    case MsgTextInputPoll: {
        TextInputReply out{};
        out.state = bbgpu_text_input_poll(out.text, sizeof(out.text));
        reply(&out, sizeof(out));
        break;
    }
    default:
        std::fprintf(stderr, "GPU process: unknown message %u\n", message.type);
        break;
    }
}

void ReaderThread() {
    Common::SetCurrentThreadName("bb:remote");
    for (;;) {
        g.channel->ReceiveOne([](const Message& message, auto&& reply) { Handle(message, reply); },
                              [] { return false; });
    }
}

constexpr int ChunkImported = -3; ///< runtime_memory.c CHUNK_IMPORTED: the pool's pages imported

/// The runtime's chunk calls (ControlBlock::protect, game process -> here). Nothing here waits for
/// the game process: the runtime may hold its lock.
void HandleChunk(const Message& message, auto&& reply) {
    const auto args = Payload<ChunkArgs>(message);
    ResultReply out{-1, 0};
    const bool valid = args.size && args.phys < g.pool_bytes && args.size <= g.pool_bytes - args.phys;
    if (message.type == MsgChunkAlloc) {
        const ChunkAlloc alloc = g.chunk_alloc.load(std::memory_order_acquire);
        if (valid && alloc && g.backing) {
            const int result = alloc(args.phys, args.size);
            if (result >= 0) {
                // A dma-buf chunk: memory of this process only (refused at MsgGuestMemory).
                close(result);
                if (const ChunkFree free_chunk = g.chunk_free.load(std::memory_order_acquire)) {
                    free_chunk(args.phys, args.size);
                }
            }
            out.result = result == ChunkImported ? ChunkImported : -1;
        }
    } else if (message.type == MsgChunkFree) {
        if (const ChunkFree free_chunk = g.chunk_free.load(std::memory_order_acquire); valid && free_chunk) {
            free_chunk(args.phys, args.size);
        }
        out.result = 0;
    }
    reply(&out, sizeof(out));
}

void ChunkThread() {
    Common::SetCurrentThreadName("bb:remote-chunks");
    for (;;) {
        g.protect->ReceiveOne(
            [](const Message& message, auto&& reply) { HandleChunk(message, reply); },
            [] { return false; });
    }
}

/// bbport: the guest hooks' fresh ranges as calls (BB_REMOTE_FRESH_SYNC=1, ControlBlock::fresh):
/// NoteFreshRange here before the hook's thread writes the new block, as it does itself in one
/// process, and (protocol 12) after the main reader has handled what the game sent before it
/// (maps, remaps, dead and live ranges: FreshArgs::after). Deadlock-free on a thread of its own:
/// it waits for the main reader, holding nothing, then for NoteFreshRange's locks (the tracker's
/// regions, pending_assets_mutex). The main reader's waits (faulting threads' applies, the
/// runtime's lock, GPU threads) and those locks' holders' (the game process's protection thread
/// and reader) never wait for the hook's thread (game code: no runtime or HLE lock held) or for
/// this one. Not ChunkThread (must never wait: the runtime may hold its lock).
void FreshThread() {
    Common::SetCurrentThreadName("bb:remote-fresh");
    for (;;) {
        g.fresh->ReceiveOne(
            [](const Message& message, auto&& reply) {
                if (message.type == MsgFreshRange) {
                    const auto args = Payload<FreshArgs>(message);
                    g.channel->WaitHandled(args.after);
                    GuestHookRange(MsgFreshRange, RangeArgs{args.address, args.size});
                }
                const ResultReply out{0, 0};
                reply(&out, sizeof(out));
            },
            [] { return false; });
    }
}

/// bbport (protocol 12): write faults from threads that hold the runtime's lock
/// (ControlBlock::locked). On the main reader they could queue behind another fault whose
/// protections its faulting thread cannot apply until this one's thread leaves the lock: a cycle
/// one process does not have (each fault there is handled on its own thread). Here a lock-holding
/// thread's fault waits for nothing but its handlers, and their waits (owner locks, the faulting
/// thread's own apply, which needs only the lock it holds and the game's trap_mutex) are the ones
/// its own handler has in one process. They are served one at a time: two can be pending only
/// with both holding the lock shared, and one's handling needs the other to leave it only where
/// one process stops too (a nested read lock behind a waiting writer).
void FaultThread() {
    Common::SetCurrentThreadName("bb:remote-faults");
    for (;;) {
        g.locked->ReceiveOne(
            [](const Message& message, auto&& reply) {
                if (message.type == MsgWriteFault) {
                    WriteFault(message, reply);
                }
            },
            [] { return false; });
    }
}

/// bb-gpu ends with the game process (its exit or exec: a restart through run.sh).
void WatchParent() {
#ifdef __APPLE__
    const pid_t parent = getppid();
    std::thread([parent] {
        Common::SetCurrentThreadName("bb:remote-watch");
        const int queue = kqueue();
        struct kevent change{};
        EV_SET(&change, parent, EVFILT_PROC, EV_ADD | EV_ONESHOT, NOTE_EXIT | NOTE_EXEC, 0,
               nullptr);
        struct kevent event{};
        if (queue < 0 || kevent(queue, &change, 1, &event, 1, nullptr) < 0 ||
            getppid() != parent) {
            std::_Exit(0);
        }
        std::fflush(stdout);
        std::_Exit(0);
    }).detach();
#else
    prctl(PR_SET_PDEATHSIG, SIGKILL);
#endif
}

void FaultHandler(int sig, siginfo_t* info, void* context) {
    // The draw preparation workers read guest memory speculatively (runtime_fault_recover).
    if (runtime_fault_recover) {
        // The recovery points don't save the signal mask (no system call per walk): the signal
        // is unblocked here, or the thread's next fault would end the process.
        sigset_t faults;
        sigemptyset(&faults);
        sigaddset(&faults, SIGSEGV);
        sigaddset(&faults, SIGBUS);
        pthread_sigmask(SIG_UNBLOCK, &faults, nullptr);
        siglongjmp(*runtime_fault_recover, 1);
    }
    char line[160];
    const int n = std::snprintf(line, sizeof(line),
                                "GPU process: signal %d at %p (pc %#llx)\n", sig,
                                info ? info->si_addr : nullptr,
                                static_cast<unsigned long long>(BbPortable::Rip(context)));
    if (n > 0) {
        [[maybe_unused]] const auto written = write(2, line, std::size_t(n));
    }
    void* frames[64];
    backtrace_symbols_fd(frames, backtrace(frames, 64), 2);
    signal(sig, SIG_DFL);
    raise(sig);
}

void InstallFaultHandler() {
    struct sigaction action{};
    action.sa_sigaction = FaultHandler;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
    sigaction(SIGILL, &action, nullptr);
}

/// The game image's part of the low region (it is loaded at its start).
constexpr u64 LowReserveBytes = 4ull << 30;

/// What already occupies [begin, end) in this process (a failed reservation).
void ReportMappings(u64 begin, u64 end) {
#ifdef __APPLE__
    mach_vm_address_t address = begin;
    for (int shown = 0; address < end && shown < 12; ++shown) {
        mach_vm_size_t size = 0;
        vm_region_extended_info_data_t info{};
        mach_msg_type_number_t count = VM_REGION_EXTENDED_INFO_COUNT;
        mach_port_t object = MACH_PORT_NULL;
        if (mach_vm_region(mach_task_self(), &address, &size, VM_REGION_EXTENDED_INFO,
                           reinterpret_cast<vm_region_info_t>(&info), &count,
                           &object) != KERN_SUCCESS ||
            address >= end) {
            break;
        }
        std::fprintf(stderr,
                     "GPU process:   in use %#llx-%#llx (%llu MiB), protection %d, tag %u\n",
                     static_cast<unsigned long long>(address),
                     static_cast<unsigned long long>(address + size),
                     static_cast<unsigned long long>(size >> 20), info.protection, info.user_tag);
        address += size;
    }
#endif
}

u64 ArgValue(int argc, char** argv, const char* name, bool& found) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            found = true;
            return std::strtoull(argv[i + 1], nullptr, 10);
        }
    }
    found = false;
    return 0;
}

} // namespace

// ---- entry ------------------------------------------------------------------------------------

int Main(int argc, char** argv) {
    back_active = true;
    setvbuf(stdout, nullptr, _IOLBF, 0);
    bool has_fd = false, has_bytes = false, has_offset = false;
    g.pool_fd = int(ArgValue(argc, argv, "--pool-fd", has_fd));
    g.pool_bytes = ArgValue(argc, argv, "--pool-bytes", has_bytes);
    const u64 control_offset = ArgValue(argc, argv, "--control-offset", has_offset);
    if (!has_fd || !has_bytes || !has_offset) {
        std::fprintf(stderr, "bb-gpu: the game process starts this\n");
        return 2;
    }
    // The guest's ranges first, before anything else in this process can land in them. Of the
    // low region only the part the game image is loaded into (its shared data segments).
    const auto reserve = [](MemoryMirror& mirror, u64 begin, u64 end, const char* what) {
        std::string_view error;
        if (mirror.ReserveAround(begin, end, error)) {
            return true;
        }
        std::fprintf(stderr, "GPU process: cannot reserve the %s %#llx-%#llx: %.*s\n", what,
                     static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end),
                     int(error.size()), error.data());
        ReportMappings(begin, end);
        return false;
    };
    if (!reserve(g.guest, GuestBegin, GuestEnd + ControlBytes, "guest range") ||
        !reserve(g.low, LowBegin, LowBegin + LowReserveBytes, "game image range")) {
        return 3;
    }
    // This process's allocator took parts of them before main: the game's mappings avoid those
    // in the guest range (HelloReply); the control block and the game image cannot move.
    for (const auto& [begin, end] : g.guest.Taken()) {
        std::printf("GPU process: its own memory at %#llx-%#llx (%llu MiB)\n",
                    static_cast<unsigned long long>(begin), static_cast<unsigned long long>(end),
                    static_cast<unsigned long long>((end - begin) >> 20));
    }
    // The game image's range is macOS's shared library region in a native process: its data
    // is mapped elsewhere when it arrives (MapAlias).
    if (!g.low.Taken().empty()) {
        std::printf("GPU process: the game image's range is the system's here; its data will be "
                    "read from a second mapping\n");
    }
    if (g.guest.Overlaps(ControlAddress, ControlBytes)) {
        std::fprintf(stderr, "GPU process: the control block's address is in use here\n");
        return 3;
    }
    if (g.guest.Taken().size() > MaxTakenRanges) {
        std::fprintf(stderr, "GPU process: its own memory is in too many parts of the guest "
                             "range\n");
        return 3;
    }
    void* at = mmap(reinterpret_cast<void*>(ControlAddress), ControlBytes, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_FIXED, g.pool_fd, static_cast<off_t>(control_offset));
    if (at == MAP_FAILED) {
        std::perror("GPU process: control block");
        return 4;
    }
    g.control = static_cast<ControlBlock*>(at);
    // The whole pool once more, outside the guest's ranges: the memory model's chunks import it
    // (runtime_memory_backing_pointer), as the backing view in the game process.
    void* view = mmap(nullptr, g.pool_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, g.pool_fd, 0);
    if (view == MAP_FAILED) {
        std::perror("GPU process: view of the game's memory");
        return 4;
    }
    g.backing = static_cast<u8*>(view);
    g.channel = std::make_unique<Channel>(&g.control->channel, Side::Backend);
    g.protect = std::make_unique<Channel>(&g.control->protect, Side::Backend);
    g.fresh = std::make_unique<Channel>(&g.control->fresh, Side::Backend);
    g.locked = std::make_unique<Channel>(&g.control->locked, Side::Backend);
    if (!g.channel->Valid() || !g.protect->Valid() || !g.fresh->Valid() || !g.locked->Valid()) {
        std::fprintf(stderr, "GPU process: the control block is not initialized\n");
        return 5;
    }
    InstallFaultHandler();
    WatchParent();
    if (const char* toggles = std::getenv("BB_TOGGLES")) {
        runtime_disabled_optimizations = std::strtoull(toggles, nullptr, 0);
    }
    // BB_TOGGLE_FILE: the switches this process's GPU code reads are its own copies; it watches
    // the file as the game process does (runtime_memory.c), every 250 ms.
    if (const char* path = std::getenv("BB_TOGGLE_FILE")) {
        std::thread([file = std::string(path)] {
            const char* fixed = std::getenv("BB_TOGGLES");
            const unsigned long long always = fixed ? std::strtoull(fixed, nullptr, 0) : 0;
            for (unsigned long long last = ~0ull;;) {
                unsigned long long value = 0, experiment = 0;
                if (FILE* f = std::fopen(file.c_str(), "r")) {
                    if (std::fscanf(f, "%llu %llu", &value, &experiment) < 1) {
                        value = 0;
                    }
                    std::fclose(f);
                }
                __atomic_store_n(&runtime_experiment_bits, u64(experiment), __ATOMIC_RELEASE);
                if (value != last) {
                    __atomic_store_n(&runtime_disabled_optimizations, u64(value | always),
                                     __ATOMIC_RELEASE);
                    std::printf("GPU process: disabled optimizations mask=%llu\n", value);
                    last = value;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        }).detach();
    }
    BbIrq::forward = [](BbIrq::Line line) { Irq(u32(line)); };
    std::printf("GPU process: native (page size %ld), guest memory reserved\n",
                sysconf(_SC_PAGESIZE));
    std::thread(ReaderThread).detach();
    std::thread(ChunkThread).detach();
    std::thread(FreshThread).detach();
    std::thread(FaultThread).detach();
#ifdef __APPLE__
    bbgpu_main_thread_loop(); // Cocoa: the window and its events belong to the main thread
#else
    for (;;) {
        pause();
    }
#endif
    return 0;
}

// ---- calls from the GPU core ------------------------------------------------------------------

void FramesRetired(u64 frames) {
    const ValueArgs args{frames};
    g.channel->Send(MsgFramesRetired, &args, sizeof(args));
}

void ReleaseSubmissionLock() {
    g.channel->Send(MsgReleaseSubmissionLock, nullptr, 0);
}

void Irq(u32 irq) {
    const ValueArgs args{irq};
    g.channel->Send(MsgIrq, &args, sizeof(args));
}

void VoFlipStatus(const VO::FlipStatus& status, bool flip_event, s64 flip_arg) {
    VoStatusArgs args{};
    args.version = g.flip_version.fetch_add(1, std::memory_order_acq_rel) + 1;
    std::memcpy(args.flip_status, &status, FlipStatusBytes);
    args.flip_arg = flip_arg;
    args.flip_event = flip_event ? 1 : 0;
    g.channel->Send(MsgVoFlipStatus, &args, sizeof(args));
}

void VoVblank(const VO::VblankStatus& status, u64 event_count) {
    VoVblankArgs args{};
    std::memcpy(args.status, &status, VblankStatusBytes);
    args.event_count = event_count;
    g.channel->Send(MsgVoVblank, &args, sizeof(args));
}

u64* VideoOutLabels() {
    return g.control->state.vo_labels;
}

// bbport 0.6's short taps (runtime_pad.c key_latch, in the game process): every key press in this
// process's window, kept until the game process's pad takes it.
static bool LatchKeyPress(void*, SDL_Event* event) {
    if (event->type == SDL_EVENT_KEY_DOWN && !event->key.repeat && g.control) {
        const u32 code = u32(event->key.scancode);
        if (code < std::size(g.control->state.key_latch) * 64) {
            g.control->state.key_latch[code / 64].fetch_or(u64(1) << (code % 64),
                                                            std::memory_order_relaxed);
        }
    }
    return true;
}

void PublishWindowState() {
    static bool latch_watch = false;
    if (!latch_watch && SDL_WasInit(SDL_INIT_VIDEO)) {
        latch_watch = SDL_AddEventWatch(LatchKeyPress, nullptr);
    }
    // The keyboard: the game process's pad reads it (runtime_pad.c through bbgpu_keyboard_state).
    int count = 0;
    if (const bool* keys = SDL_GetKeyboardState(&count)) {
        const int n = std::min<int>(count, int(sizeof(g.control->state.keyboard)));
        for (int i = 0; i < n; ++i) {
            __atomic_store_n(&g.control->state.keyboard[i], u8(keys[i] ? 1 : 0),
                             __ATOMIC_RELAXED);
        }
    }
    // The mouse (0.51's mouse look): this process's window holds it; the game process's pad
    // takes the motion, wheel and buttons (bbgpu_mouse_* there).
    auto& s = g.control->state;
    if (const u32 look = s.mouse_look.load(std::memory_order_relaxed)) {
        bbgpu_mouse_look_enable(look == 2 ? 1 : 0);
    }
    if (s.mouse_capture_requests.exchange(0, std::memory_order_relaxed) != 0) {
        bbgpu_mouse_capture();
    }
    double dx = 0, dy = 0;
    int up = 0, down = 0;
    bbgpu_mouse_take(&dx, &dy, &up, &down);
    if (dx != 0 || dy != 0) {
        s.mouse_dx.fetch_add(std::int64_t(dx * 1024.0), std::memory_order_relaxed);
        s.mouse_dy.fetch_add(std::int64_t(dy * 1024.0), std::memory_order_relaxed);
    }
    if (up) {
        s.wheel_up.fetch_add(u32(up), std::memory_order_relaxed);
    }
    if (down) {
        s.wheel_down.fetch_add(u32(down), std::memory_order_relaxed);
    }
    s.mouse_captured.store(bbgpu_mouse_captured() ? 1 : 0, std::memory_order_relaxed);
    s.mouse_buttons.store(bbgpu_mouse_buttons(), std::memory_order_relaxed);
    const u32 captures = BbOverlay::CapturesInput() ? 1 : 0;
    if (captures != g.published_overlay) {
        g.published_overlay = captures;
        g.control->state.overlay_captures_input.store(captures, std::memory_order_release);
    }
    // The pad recording clock: Display::Flip runs here, the game process's pad reads it.
    s.world_us.store(bbgpu_world_us(), std::memory_order_relaxed);
    s.world_active.store(bbgpu_world_active() ? 1 : 0, std::memory_order_relaxed);
    // Settings the settings menu changes here and the game's code reads there.
    const auto& settings = BbSettings::Get();
    GameSettingsArgs now{};
    for (int i = 0; i < 4; ++i) {
        now.camera_speed[i] = settings.camera_speed[i].load(std::memory_order_relaxed);
    }
    now.light_grid_opt = settings.light_grid_opt.load(std::memory_order_relaxed) ? 1 : 0;
    static GameSettingsArgs sent{};
    static bool sent_once = false;
    if (!sent_once || std::memcmp(&now, &sent, sizeof(now)) != 0) {
        sent_once = true;
        sent = now;
        g.channel->Send(MsgGameSettings, &now, sizeof(now));
    }
}

void GameStats(double window_s, double frames) {
    const StatsArgs args{window_s, frames};
    g.channel->Send(MsgGameStats, &args, sizeof(args));
}

int PadPrompt(const char* name, int prefer_mouse, char* label, int size) {
    if (!name || !label || size < 2) {
        return 0;
    }
    label[0] = 0;
    PadPromptArgs args{};
    std::snprintf(args.name, sizeof(args.name), "%s", name);
    args.prefer_mouse = prefer_mouse;
    PadPromptReply out{};
    g.channel->Call(MsgPadPrompt, &args, sizeof(args), &out, sizeof(out));
    out.label[sizeof(out.label) - 1] = 0;
    std::snprintf(label, std::size_t(size), "%s", out.label);
    return out.kind;
}

unsigned PadPromptGeneration() {
    return g.control->state.prompt_generation.load(std::memory_order_acquire);
}

int PadPromptDevice() {
    return g.control->state.prompt_device.load(std::memory_order_relaxed);
}

int PadMouseLook() {
    return g.control->state.pad_mouse_look.load(std::memory_order_relaxed);
}

int FileHostPath(const char* guest, char* out, u64 size) {
    if (!guest || !out || size == 0) {
        return -1;
    }
    PathArgs args{};
    std::snprintf(args.path, sizeof(args.path), "%s", guest);
    PathReply reply{-1, 0, {}};
    g.channel->Call(MsgFileHostPath, &args, sizeof(args), &reply, sizeof(reply));
    reply.path[sizeof(reply.path) - 1] = 0;
    if (reply.result != 0) {
        return reply.result;
    }
    std::snprintf(out, std::size_t(size), "%s", reply.path);
    return 0;
}

void GameEffects(u64 mask) {
    const ValueArgs args{mask};
    g.channel->Send(MsgGameEffects, &args, sizeof(args));
}

void FrameLimit(int fps) {
    const ValueArgs args{u64(std::max(fps, 0))};
    g.channel->Send(MsgFrameLimit, &args, sizeof(args));
}

void DisplayAspect(float aspect) {
    u32 bits = 0;
    std::memcpy(&bits, &aspect, sizeof(bits));
    const ValueArgs args{bits};
    g.channel->Send(MsgDisplayAspect, &args, sizeof(args));
}

void Monitor(int width, int height) {
    const ValueArgs args{BbSettings::PackSize({std::clamp(width, 0, 0xffff), std::clamp(height, 0, 0xffff)})};
    g.channel->Send(MsgMonitor, &args, sizeof(args));
}

void RequestRestart() {
    // The game process restarts through run.sh; this process ends when it does (WatchParent).
    g.channel->Send(MsgRestart, nullptr, 0);
}

// ---- the runtime's GPU interface ----------------------------------------------------------------

u64 MemoryClamp(uintptr_t address, u64 size) {
    return g.table.Clamp(address, size);
}
int MemoryRegion(uintptr_t address, uintptr_t* start, uintptr_t* end, int* mapped) {
    return g.table.Region(address, start, end, mapped);
}
const u64* MemoryGeneration() {
    return g.table.Generation();
}
int MemoryVmaInfo(uintptr_t address, int* prot, int* type, uintptr_t* end) {
    return g.table.VmaInfo(address, prot, type, end);
}
int MemoryDirectPhys(uintptr_t address, u64* phys, uintptr_t* end) {
    return g.table.DirectPhys(address, phys, end);
}
int MemoryIsMapped(uintptr_t address, u64 size) {
    return g.table.Covered(address, size);
}

int MemoryWriteBacking(uintptr_t address, const void* data, u64 size) {
    // This process's view has no page protection: the address itself is the backing.
    if (!size || !g.table.Covered(address, size)) {
        return 0;
    }
    StoreOnce(reinterpret_cast<u8*>(address), static_cast<const u8*>(data), size);
    return 1;
}

void MemoryReadBacking(uintptr_t address, void* data, u64 size) {
    auto* out = static_cast<u8*>(data);
    std::memset(out, 0, size);
    for (const auto& [piece, bytes] : g.table.Pieces(address, address + size)) {
        std::memcpy(out + (piece - address), reinterpret_cast<const void*>(piece), bytes);
    }
}

/// A protection or trap for the game process to apply, applied when this returns: by the thread
/// whose write fault this thread is answering (fault_reply), else by the protection thread.
void ForwardProtect(const ProtectArgs& args) {
    ResultReply result{};
    if (fault_reply) {
        fault_reply->Ask(MsgProtect, &args, sizeof(args), &result, sizeof(result));
        return;
    }
    g.protect->Call(MsgProtect, &args, sizeof(args), &result, sizeof(result));
}

void MemoryGpuProtect(uintptr_t address, u64 size, int read, int write) {
    ForwardProtect(ProtectArgs{address, size, u32(read != 0), u32(write != 0), 0});
}

void MemoryTrap(uintptr_t address, u64 size, unsigned reason, int on) {
    if (reason & (ProtectTrap | ProtectLift)) {
        return;
    }
    const u64 t0 = MonotonicNs();
    const TrapCopy::Order order = traps.Change(address, size, reason, on != 0);
    if (!order.size) {
        return;
    }
    const u64 t1 = MonotonicNs();
    ForwardProtect(
        ProtectArgs{order.address, order.size, ProtectTrap | reason, on ? 1u : 0u, order.seq});
    trap_wait_ns.fetch_add(t1 - t0, std::memory_order_relaxed);
    trap_apply_ns.fetch_add(MonotonicNs() - t1, std::memory_order_relaxed);
    trap_calls.fetch_add(1, std::memory_order_relaxed);
    trap_bytes.fetch_add(order.size, std::memory_order_relaxed);
}

void TrapStats(u64* wait_ns, u64* apply_ns, u64* calls, u64* bytes) {
    *wait_ns = trap_wait_ns.load(std::memory_order_relaxed);
    *apply_ns = trap_apply_ns.load(std::memory_order_relaxed); // the game process's part included
    *calls = trap_calls.load(std::memory_order_relaxed);
    *bytes = trap_bytes.load(std::memory_order_relaxed);
}

unsigned MemoryTrapReasons(uintptr_t address) {
    return traps.Reasons(address);
}

void SetGpuHooks(GpuRange map, GpuRange unmap, GpuRange invalidate) {
    {
        std::scoped_lock lock{g.hook_mutex};
        g.hook_map = map;
        g.hook_unmap = unmap;
        g.hook_invalidate = invalidate;
    }
    // Mappings that exist already are replayed, as runtime_memory_set_gpu_hooks does.
    if (map) {
        for (const auto& [address, size] : g.table.All()) {
            map(address, size);
        }
    }
}

void SetNoteWriteHook(GpuRange hook) {
    std::scoped_lock lock{g.hook_mutex};
    g.hook_note_write = hook;
}

void SetCpuWriteHook(GpuRange hook) {
    std::scoped_lock lock{g.hook_mutex};
    g.hook_cpu_write = hook;
}

void SetChunkAllocator(ChunkAlloc alloc) {
    g.chunk_alloc.store(alloc, std::memory_order_release);
}
void SetChunkFree(ChunkFree free_chunk) {
    g.chunk_free.store(free_chunk, std::memory_order_release);
}
void SetChunkPrivate(ChunkImport import) {
    g.chunk_private.store(import, std::memory_order_release);
}
void SetChunkWhole(int whole) {
    g.chunk_whole.store(whole, std::memory_order_relaxed);
}
void SetGuestLazy(int lazy) {
    g.guest_lazy.store(lazy, std::memory_order_relaxed);
}

void* BackingPointer(u64 phys) {
    return g.backing && phys < g.pool_bytes ? g.backing + phys : nullptr;
}

int PoolFd() {
    return g.pool_fd;
}

void NotePinned(u64 phys, u64 size) {
    // Before the import takes the pages: the game process's runtime keeps them on release.
    const ChunkArgs args{phys, size};
    ResultReply result{};
    g.protect->Call(MsgNotePinned, &args, sizeof(args), &result, sizeof(result));
}

int PadSteamOnly() {
    return g.control ? g.control->state.steam_only.load(std::memory_order_relaxed) : 0;
}

const char* UserDir() {
    return g.user_dir.c_str();
}

bool GuestReadable(u64 address, u64 size) {
    return GuestCovered(address, size) || InAlias(address, size);
}

u64 ProcessTimeCounter() {
    return MonotonicNs() - g.start_ns;
}
u64 ProcessTimeUs() {
    return ProcessTimeCounter() / 1000;
}
u64 TscFrequency() {
    return g.tsc_hz;
}
u64 ReadTsc() {
    const u64 elapsed = MonotonicNs() - g.tsc_sample_ns;
    return g.tsc_sample + u64((unsigned __int128)elapsed * g.tsc_hz / 1'000'000'000u);
}

} // namespace BbRemote::Back
