// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the native GPU process, bb-gpu (docs/macos-native-gpu.md). It reserves the guest's
// address ranges, maps the game process's memory pool at the same addresses, and runs the GPU
// core (Liverpool, caches, shader recompiler, presenter, window) for the game process, which
// forwards to it (remote_front.cpp). The runtime's GPU interface (runtime_memory_*, clocks) is
// served here from bb-gpu's own copy of the guest's mapping table (bb_gpu_main.cpp exports it).
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <csetjmp>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <execinfo.h>
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
#include "bbport_overlay.h"
#include "bbport_settings.h"
#include "bbport_portable.h"
#include "bbport_toggles.h"
#include "common/slot_vector.h"
#include "common/thread.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/video_out.h"
#include "core/libraries/videoout/videoout_error.h"
#include "core/memory.h"
#include "core/platform.h"
#include "remote/bb_control.h"
#include "remote/bb_memory_mirror.h"
#include "remote/bb_vma_table.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "remote/bb_protocol.h"
#include "remote/bb_remote.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Libraries::VideoOut {
VideoOutPort* GetPortForRemote(s32 handle);
s32 PS4_SYSV_ABI sceVideoOutSubmitChangeBufferAttribute(s32 handle, s32 attributeIndex,
                                                        const BufferAttribute* attribute);
s32 PS4_SYSV_ABI sceVideoOutUnregisterBuffers(s32 handle, s32 attributeIndex);
} // namespace Libraries::VideoOut

namespace BbRemote::Back {

namespace VO = Libraries::VideoOut;
static_assert(sizeof(VO::FlipStatus) == FlipStatusBytes);
static_assert(sizeof(VO::SceVideoOutVblankStatus) == VblankStatusBytes);
static_assert(sizeof(VO::SceVideoOutResolutionStatus) == ResolutionBytes);
static_assert(sizeof(VO::BufferAttribute) == BufferAttributeBytes);
static_assert(sizeof(VO::VideoOutBuffer) == VideoOutBufferBytes);
static_assert(sizeof(VO::BufferAttributeGroup) == AttributeGroupBytes);

namespace {

// ---- state ------------------------------------------------------------------------------------

struct State {
    int pool_fd = -1;
    u64 pool_bytes = 0;
    ControlBlock* control = nullptr;
    std::unique_ptr<Channel> channel, protect;
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
};
State g;

/// A write fault from a thread holding the runtime's lock: its protections go back with the reply.
thread_local std::vector<ProtectArgs>* collected_protects = nullptr;

/// Write traps (runtime_memory_trap, bbport 0.5): the game process keeps the page protections
/// (and its own table of reasons, which is the one that counts); this process keeps a copy of the
/// reasons for its GPU code's questions, one byte per 4 KiB page below TrapLimit, set by its own
/// trap calls and cleared where the game's mappings change, as runtime_memory.c's trap_forget.
constexpr u64 TrapLimit = 1ull << 40;
std::atomic<u8*> trap_reasons{nullptr};
std::mutex trap_mutex;

u8* TrapTable() {
    if (u8* table = trap_reasons.load(std::memory_order_acquire)) {
        return table;
    }
    void* bytes = mmap(nullptr, TrapLimit >> 12, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (bytes == MAP_FAILED) {
        std::perror("GPU process: write trap table");
        return nullptr;
    }
    u8* expected = nullptr;
    if (!trap_reasons.compare_exchange_strong(expected, static_cast<u8*>(bytes),
                                              std::memory_order_acq_rel)) {
        munmap(bytes, TrapLimit >> 12);
        return expected;
    }
    return static_cast<u8*>(bytes);
}

void TrapForget(u64 start, u64 end) {
    u8* table = trap_reasons.load(std::memory_order_acquire);
    if (!table || start >= TrapLimit) {
        return;
    }
    end = std::min(end, TrapLimit);
    std::scoped_lock lock{trap_mutex};
    std::memset(table + (start >> 12), 0, ((end + 4095) >> 12) - (start >> 12));
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
    VO::VideoOutPort* port = VO::GetPortForRemote(handle);
    if (!port) {
        return state;
    }
    std::scoped_lock lock{port->port_mutex};
    state.version = g.flip_version.load(std::memory_order_acquire);
    std::memcpy(state.flip_status, &port->flip_status, FlipStatusBytes);
    std::memcpy(state.slots, port->buffer_slots.data(), sizeof(state.slots));
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
    Shader::srt_guest_translate = TranslateGuest;
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
        return; // the game's own data: not in the guest's mapping table
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

void WriteFault(const Message& message, auto&& reply) {
    const auto args = Payload<FaultArgs>(message);
    std::vector<ProtectArgs> protects;
    collected_protects = args.lock_held ? &protects : nullptr;
    int handled = 0;
    if (auto* rasterizer = Core::Memory::Instance()->GetRasterizer()) {
        // As page_manager.cpp's GuestFaultSignalHandler: pages trapped for reads too go to the
        // trap's owner first.
        const unsigned reasons = MemoryTrapReasons(args.address);
        if (reasons & BbLayer::WriteTraps::VramData) {
            handled = rasterizer->OnVramDataAccess(args.address, false);
        } else if (reasons & BbLayer::WriteTraps::QueryReads) {
            handled = rasterizer->OnOcclusionPageAccess(args.address, args.rip, args.is_write != 0,
                                                        false);
        } else {
            handled = args.is_write ? rasterizer->OnWriteFault(args.address, false, args.rip)
                                    : rasterizer->ReadMemory(args.address, 8, false);
        }
    }
    collected_protects = nullptr;
    FaultReply out{};
    out.handled = handled;
    out.count = u32(std::min<std::size_t>(protects.size(), MaxFaultProtects));
    std::copy_n(protects.begin(), out.count, out.protects);
    // More than fit (not expected): the rest as calls.
    for (std::size_t i = out.count; i < protects.size(); ++i) {
        ResultReply result{};
        g.protect->Call(MsgProtect, &protects[i], sizeof(ProtectArgs), &result, sizeof(result));
    }
    reply(&out, u32(offsetof(FaultReply, protects) + out.count * sizeof(ProtectArgs)));
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
    const auto span = [&](u64 address, u32 dwords, u32 is_inline) -> std::span<const u32> {
        if (!dwords) {
            return {};
        }
        if (is_inline) {
            if (used + dwords > available) {
                return {};
            }
            const auto kept = InlineCommands(words + used, dwords);
            used += dwords;
            return kept;
        }
        return {reinterpret_cast<const u32*>(address), dwords};
    };
    const auto dcb = span(args.dcb, args.dcb_dwords, args.inline_dcb);
    const auto ccb = span(args.ccb, args.ccb_dwords, args.inline_ccb);
    liverpool->SubmitGfx(dcb, ccb);
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
    case MsgWriteFault:
        WriteFault(message, reply);
        break;

    case MsgSubmitGfx:
        SubmitGfx(message);
        break;
    case MsgSubmitAsc: {
        const auto args = Payload<SubmitAscArgs>(message);
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
        const auto id = liverpool->asc_queues.insert(
            VAddr(args.ring_base), reinterpret_cast<u32*>(args.read_ptr), args.ring_size_dw,
            args.pipe_id);
        const ResultReply out{s32(id.index), 0};
        reply(&out, sizeof(out));
        break;
    }
    case MsgUnmapComputeQueue:
        liverpool->asc_queues.erase(Common::SlotId{u32(Payload<ValueArgs>(message).value)});
        break;

    case MsgVoOpen: {
        const s32 handle = VO::sceVideoOutOpen(0, VO::SCE_VIDEO_OUT_BUS_TYPE_MAIN, 0, nullptr);
        ReplyVo(reply, handle, handle > 0 ? handle : 1);
        break;
    }
    case MsgVoClose: {
        const auto args = Payload<VoArgs>(message);
        VO::sceVideoOutClose(args.handle);
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
        const s32 result = VO::sceVideoOutRegisterBuffers(args.handle, args.start, addresses,
                                                          args.count, &attribute);
        ReplyVo(reply, result, args.handle);
        break;
    }
    case MsgVoUnregisterBuffers: {
        const auto args = Payload<VoArgs>(message);
        ReplyVo(reply, VO::sceVideoOutUnregisterBuffers(args.handle, args.value), args.handle);
        break;
    }
    case MsgVoChangeAttribute: {
        const auto args = Payload<VoAttributeArgs>(message);
        VO::BufferAttribute attribute{};
        std::memcpy(&attribute, args.attribute, sizeof(attribute));
        ReplyVo(reply,
                VO::sceVideoOutSubmitChangeBufferAttribute(args.handle, args.index, &attribute),
                args.handle);
        break;
    }
    case MsgVoSubmitFlip: {
        const auto args = Payload<VoFlipArgs>(message);
        const s32 result =
            VO::sceVideoOutSubmitFlip(args.handle, args.index, s32(args.mode), args.flip_arg);
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
        VO::sceVideoOutSetFlipRate(args.handle, args.value);
        break;
    }
    case MsgVoSetHdr: {
        const auto args = Payload<VoArgs>(message);
        if (auto* port = VO::GetPortForRemote(args.handle)) {
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
            presenter->GetPPSettingsRef().gamma = Payload<GammaArgs>(message).gamma;
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
        std::fprintf(stderr, "bb-gpu: the game process starts this (BB_NATIVE_GPU=1)\n");
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
    g.channel = std::make_unique<Channel>(&g.control->channel, Side::Backend);
    g.protect = std::make_unique<Channel>(&g.control->protect, Side::Backend);
    if (!g.channel->Valid() || !g.protect->Valid()) {
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
    Platform::IrqController::forward = [](Platform::InterruptId irq) { Irq(u32(irq)); };
    std::printf("GPU process: native (page size %ld), guest memory reserved\n",
                sysconf(_SC_PAGESIZE));
    std::thread(ReaderThread).detach();
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

void VoVblank(const VO::SceVideoOutVblankStatus& status, u64 event_count) {
    VoVblankArgs args{};
    std::memcpy(args.status, &status, VblankStatusBytes);
    args.event_count = event_count;
    g.channel->Send(MsgVoVblank, &args, sizeof(args));
}

u64* VideoOutLabels() {
    return g.control->state.vo_labels;
}

void PublishWindowState() {
    // The keyboard: the game process's pad reads it (runtime_pad.c through bbgpu_keyboard_state).
    int count = 0;
    if (const bool* keys = SDL_GetKeyboardState(&count)) {
        const int n = std::min<int>(count, int(sizeof(g.control->state.keyboard)));
        for (int i = 0; i < n; ++i) {
            __atomic_store_n(&g.control->state.keyboard[i], u8(keys[i] ? 1 : 0),
                             __ATOMIC_RELAXED);
        }
    }
    const u32 captures = BbOverlay::CapturesInput() ? 1 : 0;
    if (captures != g.published_overlay) {
        g.published_overlay = captures;
        g.control->state.overlay_captures_input.store(captures, std::memory_order_release);
    }
}

void GameStats(double window_s, double frames) {
    const StatsArgs args{window_s, frames};
    g.channel->Send(MsgGameStats, &args, sizeof(args));
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

/// A protection or trap for the game process to apply: with the reply of the fault being handled
/// on this thread, or now.
void ForwardProtect(const ProtectArgs& args) {
    if (collected_protects) {
        collected_protects->push_back(args);
        return;
    }
    ResultReply result{};
    g.protect->Call(MsgProtect, &args, sizeof(args), &result, sizeof(result));
}

void MemoryGpuProtect(uintptr_t address, u64 size, int read, int write) {
    ForwardProtect(ProtectArgs{address, size, u32(read != 0), u32(write != 0)});
}

void MemoryTrap(uintptr_t address, u64 size, unsigned reason, int on) {
    if (!size || address >= TrapLimit || (reason & ProtectTrap)) {
        return;
    }
    u8* table = TrapTable();
    if (!table) {
        return;
    }
    const u64 end = std::min<u64>(address + size, TrapLimit);
    {
        std::scoped_lock lock{trap_mutex};
        for (u64 page = address >> 12; page <= (end - 1) >> 12; ++page) {
            const u8 old = table[page];
            __atomic_store_n(&table[page], on ? u8(old | reason) : u8(old & ~reason),
                             __ATOMIC_RELEASE);
        }
    }
    ForwardProtect(ProtectArgs{address, end - address, ProtectTrap | reason, on ? 1u : 0u});
}

unsigned MemoryTrapReasons(uintptr_t address) {
    const u8* table = trap_reasons.load(std::memory_order_acquire);
    return table && address < TrapLimit ? __atomic_load_n(&table[address >> 12], __ATOMIC_ACQUIRE)
                                        : 0;
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
