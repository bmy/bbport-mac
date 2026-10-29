// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Gnm driver's submission side: graphics submissions and flips, the end-of-frame pacing
// the game relies on, compute queues (DingDong), end-of-pipe events, and the library's table.
//
// Pacing (BB_SUBMIT_LOCK): the console blocks submissions after sceGnmSubmitDone until the GPU has
// finished the frame; the game frees that frame's memory then, so some lock is needed (without one
// its heap ran out after ~7 minutes).
//   decode (default, honest labels): a frame's first BB_SUBMIT_AHEAD (2) submissions go straight
//     to the command processor, later ones wait for the previous frame. The game builds the next
//     frame in the context of the frame before the previous one, so those first submissions are
//     translated while the GPU finishes the previous frame (Yahar'gul 1080p: 101 -> 154 FPS).
//   gpu: submissions after sceGnmSubmitDone wait until the GPU is idle (the console's lock).
//   ahead: sceGnmSubmitDone(F) waits for frame F-1 only (an experiment).

#include <algorithm>
#include "shim/bb_env.h"
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <span>
#include <string_view>
#include <utility>

#include "bbport_gnm_hooks.h"
#include "bbport_fx_init_clear.h"
#include "bbport_heap_sites.h"
#include "bbport_timeline.h"
#include "bbport_toggles.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "platform/kernel.h"
#include "platform/kernel/event_queue.h"
#include "core/libraries/libs.h"
#include "platform/videoout/videoout.h"
#include "platform/errors.h"
#include "game_profile.h"
#include "platform/gnm/gnm.h"
#include "platform/gnm/gnm_init_state.h"
#include "platform/gnm/gnm_pm4.h"
#include "platform/interrupts.h"
#include "cp/command_processor.h"
#include "render/presenter.h"
#include "render/rasterizer.h"
#include "remote/bb_remote.h"

namespace Frontend {
class WindowSDL;
}
extern Frontend::WindowSDL* g_window;
std::unique_ptr<Vulkan::Presenter> presenter;
std::unique_ptr<BbCp::CommandProcessor> liverpool;

extern "C" std::uint64_t runtime_heap_growths(void);


namespace BbGnm {

bool SubmitFxInitZero(u64 address, u32 bytes) {
    // bbport (macOS native GPU process): the queue is bb-gpu's; the commands travel inline, in
    // order with the game's submissions, and are copied there at submit as here.
    if ((!liverpool && !BbRemote::FrontActive()) || !bytes || bytes > 32768 || (bytes & 3) ||
        (address & 3))
        return false;
    const auto commands = BbFxInit::Commands({address, bytes});
    // This enters the graphics queue in order, with no register changes. The queue owns its
    // copy independently of BB_COPY_GPU_BUFFERS; the initializer may return immediately.
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::SubmitGfx(commands, {}, true); // false: the game's own clear runs
    }
    liverpool->SubmitGfx(commands, {}, true);
    return true;
}

using namespace Pm4;
using OrbisKernelEqueue = BbKernel::EqueueHandle;
using OrbisKernelEvent = BbKernel::Event;
using BbCp::CommandProcessor;

// bbport (native GPU process, docs/macos-native-gpu.md): the GPU core may run in bb-gpu, where
// these calls are forwarded; the guest's command buffers are in memory both processes share.
namespace Gpu {
static void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::SubmitGfx(dcb, ccb);
        return;
    }
    liverpool->SubmitGfx(dcb, ccb);
}
static void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::SubmitAsc(gnm_vqid, acb);
        return;
    }
    liverpool->SubmitAsc(gnm_vqid, acb);
}
static void SubmitDone(u64 frame) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::SubmitDone(frame);
        return;
    }
    liverpool->SubmitDone(frame);
}
static bool IsGpuIdle() {
    return BbRemote::FrontActive() ? BbRemote::Front::IsGpuIdle() : liverpool->IsGpuIdle();
}
struct AscRing {
    VAddr map_addr;
    u32 ring_size_dw;
};
static AscRing AscQueue(u32 vqid) {
    if (BbRemote::FrontActive()) {
        const auto queue = BbRemote::Front::GetAscQueue(vqid);
        return {VAddr(queue.map_addr), queue.ring_size_dw};
    }
    const auto& queue = liverpool->asc_queues[Common::SlotId{vqid}];
    return {queue.map_addr, queue.ring_size_dw};
}
} // namespace Gpu

namespace {

u32 sdk_version = 0;

// ---- Options ---------------------------------------------------------------------------------

enum class LockMode { Decode, Gpu, Ahead };

LockMode SubmitLock() {
    static const LockMode mode = [] {
        const char* env = bb_env("BB_SUBMIT_LOCK");
        const std::string_view value = env ? env : "";
        return value == "gpu" ? LockMode::Gpu : value == "ahead" ? LockMode::Ahead : LockMode::Decode;
    }();
    return mode;
}

u32 SubmitsAheadOption() {
    static const u32 ahead = [] {
        const char* env = bb_env("BB_SUBMIT_AHEAD");
        return env ? u32(std::max(0, std::atoi(env))) : 2u;
    }();
    return ahead;
}

// ---- Frame pacing ----------------------------------------------------------------------------

struct Pacing {
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> locked{false}; ///< BB_SUBMIT_LOCK=gpu: submissions wait for the GPU to idle
    u64 frames_submitted = 0; ///< sceGnmSubmitDone calls
    u64 frames_retired = 0;   ///< frames the GPU finished
    u32 submits_this_frame = 0;
    bool overlap_off = false; ///< the game's heap grew: no submissions ahead any more
};
Pacing pacing;
/// One submission at a time, in the guest's order.
std::mutex submit_mutex;
/// The hardware state goes before the first submission of each frame.
bool send_init_state = true;

u32 SubmitsAhead() {
    return pacing.overlap_off ? 0u : SubmitsAheadOption();
}

void AddWait(std::chrono::steady_clock::time_point start) {
    BbStats::gnm_frame_waits.fetch_add(1, std::memory_order_relaxed);
    BbStats::gnm_frame_wait_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
            .count(),
        std::memory_order_relaxed);
}

/// Until the GPU has finished every frame submitted so far.
void WaitPreviousFrame() {
    std::unique_lock lock{pacing.mutex};
    if (pacing.frames_retired >= pacing.frames_submitted) {
        return;
    }
    BbTimeline::Note(BbTimeline::GuestWaitBegin, pacing.frames_submitted);
    const auto start = std::chrono::steady_clock::now();
    pacing.cv.wait(lock, [] { return pacing.frames_retired >= pacing.frames_submitted; });
    AddWait(start);
    BbTimeline::Note(BbTimeline::GuestWaitEnd, pacing.frames_submitted);
}

/// BB_SUBMIT_LOCK=gpu: until the GPU went idle after the last sceGnmSubmitDone.
void WaitGpuIdle() {
    std::unique_lock lock{pacing.mutex};
    if (!pacing.locked) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    pacing.cv.wait(lock, [] { return !pacing.locked; });
    AddWait(start);
}

/// After a submission was handed over: the frame's later submissions wait for the previous frame.
void WaitAfterSubmit() {
    if (SubmitLock() == LockMode::Ahead || ++pacing.submits_this_frame <= SubmitsAhead()) {
        return;
    }
    WaitPreviousFrame();
}

/// Once a frame: the game's live heap allocations (wrappers on its malloc count them). The leak
/// that frames overlapping too far caused added ~1800 a second, so 100000 over the lowest count
/// seen (or the heap asking the system for more memory) stops the overlap for the session.
void WatchOverlapLeak() {
    // The live allocations are counted in Bloodborne's libc (BbHeapSites): its profile only.
    if (SubmitsAhead() == 0 || !Game::Active()) {
        return;
    }
    if (!BbHeapSites::Counting()) {
        BbHeapSites::InstallCounters();
        return;
    }
    static u64 frames = 0;
    static long long lowest = 0;
    const long long live = BbHeapSites::LiveAllocations();
    if (++frames < 600 || live < lowest) {
        lowest = live; // warming up: area loads
        return;
    }
    if (live - lowest > 100000 || runtime_heap_growths() != 0) {
        pacing.overlap_off = true;
        std::printf("Gnm: the game's heap grows (%lld allocations over %lld): frames no longer "
                    "overlap (BB_SUBMIT_AHEAD 0)\n",
                    live - lowest, lowest);
    }
}

/// Before a submission is handed over.
void WaitBeforeSubmit() {
    if (!SubmitLockOnDecode()) {
        WaitGpuIdle();
    }
}

// ---- Submissions -----------------------------------------------------------------------------

/// The hardware state the console driver gives the GPU before a frame's first submission (the
/// generation follows the game's SDK: before 2.00, 2.00 to 4.00, later), with a CLEAR_STATE.
std::span<const u32> FrameInitState() {
    static constexpr auto Clear = Packet(Op::ClearState, 0u);
    static constexpr auto Base = Join(Clear, InitState::HwBase);
    static constexpr auto Sdk200 = Join(Clear, InitState::Hw200);
    static constexpr auto Sdk350 = Join(Clear, InitState::Hw350);
    if (sdk_version < 0x2000000) {
        return Base;
    }
    return sdk_version < 0x4000000 ? std::span<const u32>{Sdk200} : std::span<const u32>{Sdk350};
}

s32 CheckBuffers(u32 count, const void* dcbs, const u32* dcb_sizes, const void* ccbs,
                 const u32* ccb_sizes) {
    if (count != 0 && (!dcbs || !dcb_sizes)) {
        LOG_ERROR(Lib_GnmDriver, "no draw command buffers");
        return BbErr::GnmInvalid;
    }
    if (count != 0 && ccb_sizes && !ccbs) {
        LOG_ERROR(Lib_GnmDriver, "constant buffer sizes without buffers");
        return BbErr::GnmInvalid;
    }
    for (u32 i = 0; i < count; i++) {
        // At most 2^20 - 1 dwords each.
        if (dcb_sizes[i] == 0 || dcb_sizes[i] > 0x3FFFFC ||
            (ccb_sizes && ccb_sizes[i] > 0x3FFFFC)) {
            LOG_ERROR(Lib_GnmDriver, "bad command buffer size at {}", i);
            return BbErr::GnmInvalid;
        }
    }
    return BbErr::Ok;
}

void Submit(u32 count, const u32* const* dcbs, const u32* dcb_sizes, const u32* const* ccbs,
            const u32* ccb_sizes) {
    if (send_init_state) {
        Gpu::SubmitGfx(FrameInitState(), {});
        send_init_state = false;
    }
    for (u32 i = 0; i < count; ++i) {
        const std::span dcb{dcbs[i], dcb_sizes[i] / 4};
        const std::span ccb{ccbs ? ccbs[i] : nullptr, ccb_sizes ? ccb_sizes[i] / 4 : 0};
        BbGnmHooks::CheckSubmission(dcb.data(), dcb.size());
        BbGnmHooks::CheckSubmission(ccb.data(), ccb.size());
        Gpu::SubmitGfx(dcb, ccb);
    }
}

/// The last 64 dwords of a flipping submission are a NOP the game reserved (prepareFlip, with a
/// payload saying what to do once the flip is queued). It becomes: write 1 to the display
/// buffer's label (the buffer is in use until the flip), then the payload's own action (write a
/// label, or an end-of-pipe event with or without an interrupt), padded with a NOP.
s32 PatchFlip(u32* cmdbuf, u32 size, u32 vo_handle, u32 buf_idx, u32 flip_mode, s64 flip_arg) {
    u32* const packet = cmdbuf + size - 64;
    ASSERT_MSG(packet[0] == Header(Op::Nop, 63), "no prepareFlip packet");
    u32 saved[7];
    std::memcpy(saved, packet, sizeof(saved));
    const u32 kind = saved[1];
    ASSERT_MSG((saved[2] & 3) == 0 || kind != Payload::PrepareFlipLabel, "bad flip packet");
    ASSERT_MSG(buf_idx != 0xFFFFFFFFu, "bad display buffer index");

    const s32 flip = BbVideoOut::sceVideoOutSubmitEopFlip(s32(vo_handle), buf_idx,
                                                                   flip_mode, flip_arg, nullptr);
    if (flip != BbErr::Ok) {
        LOG_ERROR(Lib_GnmDriver, "flip request failed: {:#x}", u32(flip));
        return u32(flip) == 0x80290012u ? BbErr::GnmFlipQueueFull : flip; // flip queue full
    }
    uintptr_t labels = 0;
    const s32 count = BbVideoOut::sceVideoOutGetBufferLabelAddress(s32(vo_handle), &labels);
    ASSERT_MSG(count == 16, "sceVideoOutGetBufferLabelAddress failed");
    const u64 label = (labels + buf_idx * sizeof(u64)) & ~u64{3};
    Emit(packet, Op::WriteData, false, false, 0x500u, u32(label), u32(label >> 32), 1u);

    // The flip marker the command processor acts on, sized to leave room for the action.
    const auto marker = [&](u32 body) {
        Emit(packet + 5, Op::Nop, false, false, Payload::PatchedFlip);
        packet[5] = Header(Op::Nop, body);
    };
    const u32 eop_event = (saved[5] & 0x3F) + 0x500 + (saved[6] & 0x3F) * 0x1000;
    switch (kind) {
    case Payload::PrepareFlip:
        marker(0x3A);
        break;
    case Payload::PrepareFlipLabel:
        marker(0x35);
        Emit(packet + 0x3B, Op::WriteData, false, false, 0x500u, saved[2] & 0xFFFFFFFCu, saved[3],
             saved[4]);
        break;
    case Payload::PrepareFlipInterruptLabel:
        marker(0x34);
        Emit(packet + 0x3A, Op::EventWriteEop, false, false, eop_event, saved[2] & 0xFFFFFFFCu,
             (saved[3] & 0xFFFF) | 0x22000000u, saved[4], 0u); // 32-bit data, interrupt
        break;
    case Payload::PrepareFlipInterrupt:
        marker(0x34);
        Emit(packet + 0x3A, Op::EventWriteEop, false, false, eop_event, 0u, 0x1000000u, 0u, 0u);
        break;
    default:
        break; // the label written, nothing else
    }
    return BbErr::Ok;
}

// ---- Compute queues --------------------------------------------------------------------------

std::array<u32, BbCp::CommandProcessor::NumComputeRings> queue_read_dw{};

/// End-of-pipe and compute release interrupts are the events games add (sceGnmAddEqEvent).
bool IsEventLine(BbIrq::Line line) {
    return u32(line) <= u32(BbIrq::Line::Compute6RelMem) || line == BbIrq::Line::GfxEop;
}

} // namespace

u32 SdkVersion() {
    return sdk_version;
}

bool SubmitLockOnDecode() {
    // Only with honest labels: without them the GPU idle interrupt comes at decode anyway, and
    // the idle check keeps waiting for deferred fences (BB_WORK_RETIRED, Steam Deck crashes).
    return SubmitLock() != LockMode::Gpu;
}

void ReleaseSubmissionLock() {
    if (BbRemote::BackActive()) {
        BbRemote::Back::ReleaseSubmissionLock(); // the guest waits in the game process
    }
    std::scoped_lock lock{pacing.mutex};
    pacing.locked = false;
    pacing.cv.notify_all();
}

void NoteFramesRetired(u64 frames) {
    if (BbRemote::BackActive()) {
        BbRemote::Back::FramesRetired(frames); // the guest waits in the game process
    }
    BbTimeline::Note(BbTimeline::FrameRetired, frames);
    std::scoped_lock lock{pacing.mutex};
    pacing.frames_retired = std::max(pacing.frames_retired, frames);
    pacing.cv.notify_all();
}

// ---- Exported functions ----------------------------------------------------------------------

s32 PS4_SYSV_ABI sceGnmAddEqEvent(OrbisKernelEqueue eq, u64 id, void* udata) {
    auto* queue = BbKernel::FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    OrbisKernelEvent event{};
    event.ident = id;
    event.filter = OrbisKernelEvent::GraphicsCore;
    event.flags = OrbisKernelEvent::Add;
    event.udata = udata;
    queue->Add(event);
    BbIrq::Listen(BbIrq::Line(id), queue, [queue, id](BbIrq::Line line) {
        if (IsEventLine(line)) {
            // The data word is the event type, as sceGnmGetEqEventType reads it.
            queue->Trigger(id, OrbisKernelEvent::GraphicsCore, reinterpret_cast<void*>(id));
        }
    });
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDeleteEqEvent(OrbisKernelEqueue eq, u64 id) {
    auto* queue = BbKernel::FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    queue->Remove(id, OrbisKernelEvent::GraphicsCore);
    BbIrq::Forget(BbIrq::Line(id), queue);
    return BbErr::Ok;
}

u64 PS4_SYSV_ABI sceGnmGetEqEventType(const OrbisKernelEvent* event) {
    return BbKernel::sceKernelGetEventData(event);
}

s32 PS4_SYSV_ABI sceGnmAreSubmitsAllowed() {
    const bool allowed = !pacing.locked;
    BbStats::submits_allowed_queries.fetch_add(1, std::memory_order_relaxed);
    if (!allowed) {
        BbStats::submits_refused.fetch_add(1, std::memory_order_relaxed);
    }
    return allowed;
}

// Workloads (the 5.00 SDK's way of grouping submissions): streams 0 to 15, one workload each.
s32 PS4_SYSV_ABI sceGnmCreateWorkloadStream(u64 name, u32* stream) {
    if (name == 0 || !stream) {
        return 3;
    }
    *stream = 1;
    return BbErr::Ok;
}
s32 PS4_SYSV_ABI sceGnmBeginWorkload(u32 stream, u64* workload) {
    if (!workload) {
        return 3;
    }
    *workload = stream < 0x10 ? 1 : 0;
    return stream > 0xF;
}
s32 PS4_SYSV_ABI sceGnmEndWorkload(u64 workload) {
    return workload == 0 ? 2 : ((workload >> 56) > 0xF) * 2;
}

u32 PS4_SYSV_ABI sceGnmGetGpuCoreClockFrequency() {
    return u32(GpuCoreClock);
}

u32 PS4_SYSV_ABI sceGnmGetOffChipTessellationBufferSize() {
    return 0x800000;
}

/// The tessellation factor ring: the last 256 MiB of the system-reserved range (0x7'0000'0000,
/// 4 GiB), where the console driver puts it.
u64 PS4_SYSV_ABI sceGnmGetTheTessellationFactorRingBufferBaseAddress() {
    return 0x7'F000'0000ull;
}

s32 PS4_SYSV_ABI sceGnmLogicalCuMaskToPhysicalCuMask(s64, s32 logical_cu_mask) {
    return logical_cu_mask;
}

s32 PS4_SYSV_ABI sceGnmMapComputeQueue(u32 pipe_id, u32 queue_id, u64 ring_base, u32 ring_size_dw,
                                       u32* read_ptr) {
    if (pipe_id >= BbCp::CommandProcessor::NumComputePipes) {
        return BbErr::GnmQueueBadPipe;
    }
    if (queue_id >= BbCp::CommandProcessor::NumQueuesPerPipe) {
        return BbErr::GnmQueueBadQueue;
    }
    if (ring_base % 256 != 0) {
        return BbErr::GnmQueueBadRingBase;
    }
    if (!std::has_single_bit(ring_size_dw)) {
        return BbErr::GnmQueueBadRingSize;
    }
    if (reinterpret_cast<uintptr_t>(read_ptr) % 4 != 0) {
        return BbErr::GnmQueueBadReadPtr;
    }
    u32 index = 0;
    if (BbRemote::FrontActive()) {
        // bbport (native GPU process): the queue lives in bb-gpu's command processor.
        const s32 remote = BbRemote::Front::MapComputeQueue(ring_base, read_ptr, ring_size_dw, pipe_id);
        if (remote < 0) {
            return BbErr::GnmQueueBadRingBase;
        }
        index = u32(remote);
        *read_ptr = 0;
    } else {
        const auto slot = liverpool->asc_queues.insert(ring_base, read_ptr, ring_size_dw, pipe_id);
        *liverpool->asc_queues[slot].read_addr = 0;
        index = slot.index;
    }
    const u32 vqid = index + 1; // DingDong numbers queues from 1
    LOG_INFO(Lib_GnmDriver, "compute pipe {} queue {} mapped to vqueue {}", pipe_id, queue_id, vqid);
    return s32(vqid);
}

s32 PS4_SYSV_ABI sceGnmMapComputeQueueWithPriority(u32 pipe_id, u32 queue_id, u64 ring_base,
                                                   u32 ring_size_dw, u32* read_ptr, u32) {
    return sceGnmMapComputeQueue(pipe_id, queue_id, ring_base, ring_size_dw, read_ptr);
}

s32 PS4_SYSV_ABI sceGnmUnmapComputeQueue(u32 vqid) {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::UnmapComputeQueue(vqid - 1);
        return BbErr::Ok;
    }
    liverpool->asc_queues.erase(Common::SlotId{vqid - 1});
    return BbErr::Ok;
}

/// The game wrote up to `next_dw` into compute queue `vqid`'s ring: hands the new part over (in
/// two pieces when it wrapped around the end).
void PS4_SYSV_ABI sceGnmDingDong(u32 vqid, u32 next_dw) {
    if (vqid == 0) {
        return;
    }
    std::scoped_lock lock{submit_mutex};
    WaitBeforeSubmit();
    const auto queue = Gpu::AscQueue(vqid - 1);
    u32& read_dw = queue_read_dw[vqid - 1];
    const auto* ring = reinterpret_cast<const u32*>(queue.map_addr);
    if (next_dw != read_dw) {
        if (next_dw < read_dw && next_dw != 0) {
            Gpu::SubmitAsc(vqid, {ring + read_dw, queue.ring_size_dw - read_dw});
            read_dw = 0;
        }
        const u32 end = next_dw ? next_dw : queue.ring_size_dw;
        const std::span commands{ring + read_dw, end - read_dw};
        read_dw = next_dw;
        Gpu::SubmitAsc(vqid, commands);
    }
    if (SubmitLockOnDecode()) {
        WaitAfterSubmit(); // handed over first, as the graphics submissions
    }
}

void PS4_SYSV_ABI sceGnmDingDongForWorkload(u32 vqid, u32 next_dw, u64) {
    sceGnmDingDong(vqid, next_dw);
}

s32 PS4_SYSV_ABI sceGnmSubmitCommandBuffers(u32 count, const u32* dcbs[], u32* dcb_sizes,
                                            const u32* ccbs[], u32* ccb_sizes) {
    if (!dcbs || !dcb_sizes) {
        LOG_ERROR(Lib_GnmDriver, "no draw command buffers");
        return BbErr::GnmInvalid;
    }
    if (const s32 error = CheckBuffers(count, dcbs, dcb_sizes, ccbs, ccb_sizes)) {
        return error;
    }
    std::scoped_lock lock{submit_mutex};
    WaitBeforeSubmit();
    Submit(count, dcbs, dcb_sizes, ccbs, ccb_sizes);
    if (SubmitLockOnDecode()) {
        WaitAfterSubmit();
    }
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmSubmitCommandBuffersForWorkload(u32, u32 count, const u32* dcbs[],
                                                       u32* dcb_sizes, const u32* ccbs[],
                                                       u32* ccb_sizes) {
    return sceGnmSubmitCommandBuffers(count, dcbs, dcb_sizes, ccbs, ccb_sizes);
}

s32 PS4_SYSV_ABI sceGnmSubmitAndFlipCommandBuffers(u32 count, u32* dcbs[], u32* dcb_sizes,
                                                   u32* ccbs[], u32* ccb_sizes, u32 vo_handle,
                                                   u32 buf_idx, u32 flip_mode, s64 flip_arg) {
    if (const s32 error = CheckBuffers(count, dcbs, dcb_sizes, ccbs, ccb_sizes)) {
        return error;
    }
    std::scoped_lock lock{submit_mutex};
    WaitBeforeSubmit();
    const s32 patched = PatchFlip(dcbs[count - 1], dcb_sizes[count - 1] / 4, vo_handle, buf_idx,
                                  flip_mode, flip_arg);
    if (patched != BbErr::Ok) {
        return patched;
    }
    Submit(count, const_cast<const u32* const*>(dcbs), dcb_sizes,
           const_cast<const u32* const*>(ccbs), ccb_sizes);
    if (SubmitLockOnDecode()) {
        WaitAfterSubmit();
    }
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmSubmitAndFlipCommandBuffersForWorkload(u32, u32 count, u32* dcbs[],
                                                              u32* dcb_sizes, u32* ccbs[],
                                                              u32* ccb_sizes, u32 vo_handle,
                                                              u32 buf_idx, u32 flip_mode,
                                                              s64 flip_arg) {
    return sceGnmSubmitAndFlipCommandBuffers(count, dcbs, dcb_sizes, ccbs, ccb_sizes, vo_handle,
                                             buf_idx, flip_mode, flip_arg);
}

s32 PS4_SYSV_ABI sceGnmSubmitDone() {
    BbStats::submit_done_calls.fetch_add(1, std::memory_order_relaxed);
    BbTimeline::Note(BbTimeline::GuestSubmitDone, pacing.frames_submitted + 1);
    std::scoped_lock lock{submit_mutex};
    if (SubmitLockOnDecode()) {
        // A frame without submissions waits here, as the lock made sceGnmSubmitDone wait.
        WaitPreviousFrame();
        pacing.submits_this_frame = 0;
        WatchOverlapLeak();
        u64 frame;
        {
            std::scoped_lock pacing_lock{pacing.mutex};
            frame = ++pacing.frames_submitted;
        }
        Gpu::SubmitDone(frame);
    } else {
        WaitGpuIdle();
        if (!Gpu::IsGpuIdle()) {
            std::scoped_lock pacing_lock{pacing.mutex};
            pacing.locked = true;
        }
        {
            std::scoped_lock pacing_lock{pacing.mutex};
            ++pacing.frames_submitted;
        }
        Gpu::SubmitDone(0);
    }
    send_init_state = true;
    return BbErr::Ok;
}

// ---- Registration ----------------------------------------------------------------------------

namespace {

enum class Lib { Driver, Compat, CoreClock, DebugModuleReset, DebugReset, ResourceRegistration,
                 WaitFreeSubmit };

constexpr const char* LibName(Lib lib) {
    switch (lib) {
    case Lib::Driver:
        return "libSceGnmDriver";
    case Lib::Compat:
        return "libSceGnmDriverCompat";
    case Lib::CoreClock:
        return "libSceGnmGetGpuCoreClockFrequency";
    case Lib::DebugModuleReset:
        return "libSceGnmDebugModuleReset";
    case Lib::DebugReset:
        return "libSceGnmDebugReset";
    case Lib::ResourceRegistration:
        return "libSceGnmDriverResourceRegistration";
    case Lib::WaitFreeSubmit:
        return "libSceGnmWaitFreeSubmit";
    }
    return "libSceGnmDriver";
}

struct Export {
    const char* nid;
    Lib lib;
    u64 address;
};

template <typename F>
u64 Fn(F* function) {
    return reinterpret_cast<u64>(function);
}

/// What the functions that do nothing here return: developer and debugging facilities a retail
/// console does not have (their documented failure codes), or calls the port has no use for.
enum class Stub {
    Silent,         ///< 0
    Missing,        ///< 0, reported once: a function the port should implement if a game needs it
    Failure,        ///< GnmFailure
    ValidationOff,  ///< GnmValidationOff
    CaptureFailed,  ///< GnmCaptureFailed
    RazorNotLoaded, ///< GnmRazorNotLoaded
    NoInterface,    ///< 0x80000000
    MinusOne,       ///< -1
    Unreachable,    ///< stops: the port cannot do what was asked
};

struct StubEntry {
    const char* nid;
    const char* name;
    Lib lib;
    Stub kind;
};

constexpr StubEntry Stubs[] = {
    {"EJapNl2+pgU", "sceGnmComputeWaitSemaphore", Lib::Driver, Stub::Missing},
    {"jwCEzr7uEP4", "sceGnmDebuggerGetAddressWatch", Lib::Driver, Stub::Failure},
    {"PNf0G7gvFHQ", "sceGnmDebuggerHaltWavefront", Lib::Driver, Stub::Failure},
    {"nO-tMnaxJiE", "sceGnmDebuggerReadGds", Lib::Driver, Stub::Failure},
    {"t0HIQWnvK9E", "sceGnmDebuggerReadSqIndirectRegister", Lib::Driver, Stub::Failure},
    {"HsLtF4jKe48", "sceGnmDebuggerResumeWavefront", Lib::Driver, Stub::Failure},
    {"JRKSSV0YzwA", "sceGnmDebuggerResumeWavefrontCreation", Lib::Driver, Stub::Failure},
    {"jpTMyYB8UBI", "sceGnmDebuggerSetAddressWatch", Lib::Driver, Stub::Failure},
    {"MJG69Q7ti+s", "sceGnmDebuggerWriteGds", Lib::Driver, Stub::Failure},
    {"PaFw9w6f808", "sceGnmDebuggerWriteSqIndirectRegister", Lib::Driver, Stub::Failure},
    {"qpGITzPE+Zc", "sceGnmDebugHardwareStatus", Lib::Driver, Stub::Missing},
    {"UtObDRQiGbs", "sceGnmDestroyWorkloadStream", Lib::Driver, Stub::Missing},
    {"HHo1BAljZO8", "sceGnmDisableMipStatsReport", Lib::Driver, Stub::Missing},
    {"jHdPvIzlpKc", "sceGnmDrawIndexMultiInstanced", Lib::Driver, Stub::Unreachable},
    {"cUCo8OvArrw", "sceGnmDrawIndirectCountMulti", Lib::Driver, Stub::Unreachable},
    {"stDSYW2SBVM", "sceGnmDrawOpaqueAuto", Lib::Driver, Stub::Missing},
    {"TLV4mswiZ4A", "sceGnmDriverCaptureInProgress", Lib::Driver, Stub::Silent},
    {"ODEeJ1GfDtE", "sceGnmDriverInternalRetrieveGnmInterface", Lib::Driver, Stub::NoInterface},
    {"4LSXsEKPTsE", "sceGnmDriverInternalRetrieveGnmInterfaceForGpuDebugger", Lib::Driver, Stub::NoInterface},
    {"MpncRjHNYRE", "sceGnmDriverInternalRetrieveGnmInterfaceForGpuException", Lib::Driver, Stub::NoInterface},
    {"EwjWGcIOgeM", "sceGnmDriverInternalRetrieveGnmInterfaceForHDRScopes", Lib::Driver, Stub::NoInterface},
    {"3EXdrVC7WFk", "sceGnmDriverInternalRetrieveGnmInterfaceForReplay", Lib::Driver, Stub::NoInterface},
    {"P9iKqxAGeck", "sceGnmDriverInternalRetrieveGnmInterfaceForResourceRegistration", Lib::Driver, Stub::NoInterface},
    {"t-vIc5cTEzg", "sceGnmDriverInternalRetrieveGnmInterfaceForValidation", Lib::Driver, Stub::NoInterface},
    {"BvvO8Up88Zc", "sceGnmDriverInternalVirtualQuery", Lib::Driver, Stub::Silent},
    {"R6z1xM3pW-w", "sceGnmDriverTraceInProgress", Lib::Driver, Stub::Silent},
    {"d88anrgNoKY", "sceGnmDriverTriggerCapture", Lib::Driver, Stub::RazorNotLoaded},
    {"4Mv9OXypBG8", "sceGnmFindResourcesPublic", Lib::Driver, Stub::Failure},
    {"iBt3Oe00Kvc", "sceGnmFlushGarlic", Lib::Driver, Stub::Missing},
    {"GviyYfFQIkc", "sceGnmGetCoredumpAddress", Lib::Driver, Stub::Missing},
    {"meiO-5ZCVIE", "sceGnmGetCoredumpMode", Lib::Driver, Stub::Missing},
    {"O-7nHKgcNSQ", "sceGnmGetCoredumpProtectionFaultTimestamp", Lib::Driver, Stub::Missing},
    {"bSJFzejYrJI", "sceGnmGetDbgGcHandle", Lib::Driver, Stub::MinusOne},
    {"pd4C7da6sEg", "sceGnmGetDebugTimestamp", Lib::Driver, Stub::Missing},
    {"H7-fgvEutM0", "sceGnmGetEqTimeStamp", Lib::Driver, Stub::Missing},
    {"oL4hGI1PMpw", "sceGnmGetGpuBlockStatus", Lib::Driver, Stub::Silent},
    {"tZCSL5ulnB4", "sceGnmGetGpuInfoStatus", Lib::Driver, Stub::Silent},
    {"iFirFzgYsvw", "sceGnmGetLastWaitedAddress", Lib::Driver, Stub::Silent},
    {"KnldROUkWJY", "sceGnmGetNumTcaUnits", Lib::Driver, Stub::Missing},
    {"QJjPjlmPAL0", "sceGnmGetOwnerName", Lib::Driver, Stub::Failure},
    {"dewXw5roLs0", "sceGnmGetPhysicalCounterFromVirtualized", Lib::Driver, Stub::Failure},
    {"fzJdEihTFV4", "sceGnmGetProtectionFaultTimeStamp", Lib::Driver, Stub::Silent},
    {"4PKnYXOhcx4", "sceGnmGetResourceBaseAddressAndSizeInBytes", Lib::Driver, Stub::Failure},
    {"O0S96YnD04U", "sceGnmGetResourceName", Lib::Driver, Stub::Failure},
    {"UBv7FkVfzcQ", "sceGnmGetResourceShaderGuid", Lib::Driver, Stub::Failure},
    {"bdqdvIkLPIU", "sceGnmGetResourceType", Lib::Driver, Stub::Failure},
    {"UoBuWAhKk7U", "sceGnmGetResourceUserData", Lib::Driver, Stub::Failure},
    {"nEyFbYUloIM", "sceGnmGetShaderProgramBaseAddress", Lib::Driver, Stub::Silent},
    {"k7iGTvDQPLQ", "sceGnmGetShaderStatus", Lib::Driver, Stub::Silent},
    {"QLdG7G-PBZo", "sceGnmGpuPaDebugEnter", Lib::Driver, Stub::Silent},
    {"tVEdZe3wlbY", "sceGnmGpuPaDebugLeave", Lib::Driver, Stub::Silent},
    {"aj3L-iaFmyk", "sceGnmInsertSetColorMarker", Lib::Driver, Stub::Missing},
    {"URDgJcXhQOs", "sceGnmInsertThreadTraceMarker", Lib::Driver, Stub::Silent},
    {"HRyNHoAjb6E", "sceGnmIsCoredumpValid", Lib::Driver, Stub::Missing},
    {"jg33rEKLfVs", "sceGnmIsUserPaEnabled", Lib::Driver, Stub::Silent},
    {"26PM5Mzl8zc", "sceGnmLogicalCuIndexToPhysicalCuIndex", Lib::Driver, Stub::Silent},
    {"Kl0Z3LH07QI", "sceGnmLogicalTcaUnitToPhysical", Lib::Driver, Stub::Missing},
    {"+N+wrSYBLIw", "sceGnmPaDisableFlipCallbacks", Lib::Driver, Stub::Silent},
    {"8WDA9RiXLaw", "sceGnmPaEnableFlipCallbacks", Lib::Driver, Stub::Silent},
    {"tNuT48mApTc", "sceGnmPaHeartbeat", Lib::Driver, Stub::Silent},
    {"6IMbpR7nTzA", "sceGnmQueryResourceRegistrationUserMemoryRequirements", Lib::Driver, Stub::Failure},
    {"+rJnw2e9O+0", "sceGnmRaiseUserExceptionEvent", Lib::Driver, Stub::Silent},
    {"9Mv61HaMhfA", "sceGnmRegisterGdsResource", Lib::Driver, Stub::Failure},
    {"t7-VbMosbR4", "sceGnmRegisterGnmLiveCallbackConfig", Lib::Driver, Stub::Silent},
    {"ZFqKFl23aMc", "sceGnmRegisterOwner", Lib::Driver, Stub::Failure},
    {"nvEwfYAImTs", "sceGnmRegisterResource", Lib::Driver, Stub::Failure},
    {"gObODli-OH8", "sceGnmRequestFlipAndSubmitDone", Lib::Driver, Stub::Missing},
    {"6YRHhh5mHCs", "sceGnmRequestFlipAndSubmitDoneForWorkload", Lib::Driver, Stub::Missing},
    {"f85orjx7qts", "sceGnmRequestMipStatsReportAndReset", Lib::Driver, Stub::Missing},
    {"hS0MKPRdNr0", "sceGnmSdmaClose", Lib::Driver, Stub::Failure},
    {"31G6PB2oRYQ", "sceGnmSdmaConstFill", Lib::Driver, Stub::Failure},
    {"Lg2isla2XeQ", "sceGnmSdmaCopyLinear", Lib::Driver, Stub::Failure},
    {"-Se2FY+UTsI", "sceGnmSdmaCopyTiled", Lib::Driver, Stub::Failure},
    {"OlFgKnBsALE", "sceGnmSdmaCopyWindow", Lib::Driver, Stub::Failure},
    {"LQQN0SwQv8c", "sceGnmSdmaFlush", Lib::Driver, Stub::Failure},
    {"suUlSjWr7CE", "sceGnmSdmaGetMinCmdSize", Lib::Driver, Stub::Failure},
    {"5AtqyMgO7fM", "sceGnmSdmaOpen", Lib::Driver, Stub::Failure},
    {"jtkqXpAOY6w", "sceGnmSetGsRingSizes", Lib::Driver, Stub::Missing},
    {"q-qhDxP67Hg", "sceGnmSetResourceRegistrationUserMemory", Lib::Driver, Stub::Failure},
    {"K3BKBBYKUSE", "sceGnmSetResourceUserData", Lib::Driver, Stub::Failure},
    {"0O3xxFaiObw", "sceGnmSetSpiEnableSqCounters", Lib::Driver, Stub::Failure},
    {"lN7Gk-p9u78", "sceGnmSetSpiEnableSqCountersForUnitInstance", Lib::Driver, Stub::Failure},
    {"+xuDhxlWRPg", "sceGnmSetupMipStatsReport", Lib::Driver, Stub::Missing},
    {"y+iI2lkX+qI", "sceGnmSetWaveLimitMultiplier", Lib::Driver, Stub::Missing},
    {"XiyzNZ9J4nQ", "sceGnmSetWaveLimitMultipliers", Lib::Driver, Stub::Missing},
    {"kkn+iy-mhyg", "sceGnmSpmEndSpm", Lib::Driver, Stub::Failure},
    {"aqhuK2Mj4X4", "sceGnmSpmInit", Lib::Driver, Stub::Failure},
    {"KHpZ9hJo1c0", "sceGnmSpmInit2", Lib::Driver, Stub::Failure},
    {"QEsMC+M3yjE", "sceGnmSpmSetDelay", Lib::Driver, Stub::Failure},
    {"hljMAxTLNF0", "sceGnmSpmSetMuxRam", Lib::Driver, Stub::Failure},
    {"bioGsp74SLM", "sceGnmSpmSetMuxRam2", Lib::Driver, Stub::Failure},
    {"cMWWYeqQQlM", "sceGnmSpmSetSelectCounter", Lib::Driver, Stub::Failure},
    {"-zJi8Vb4Du4", "sceGnmSpmSetSpmSelects", Lib::Driver, Stub::Failure},
    {"xTsOqp-1bE4", "sceGnmSpmSetSpmSelects2", Lib::Driver, Stub::Failure},
    {"AmmYLcJGTl0", "sceGnmSpmStartSpm", Lib::Driver, Stub::Failure},
    {"UHDiSFDxNao", "sceGnmSqttFini", Lib::Driver, Stub::Failure},
    {"a3tLC56vwug", "sceGnmSqttFinishTrace", Lib::Driver, Stub::Failure},
    {"L-owl1dSKKg", "sceGnmSqttGetBcInfo", Lib::Driver, Stub::Failure},
    {"LQtzqghKQm4", "sceGnmSqttGetGpuClocks", Lib::Driver, Stub::Failure},
    {"wYN5mmv6Ya8", "sceGnmSqttGetHiWater", Lib::Driver, Stub::Failure},
    {"9X4SkENMS0M", "sceGnmSqttGetStatus", Lib::Driver, Stub::Failure},
    {"lbMccQM2iqc", "sceGnmSqttGetTraceCounter", Lib::Driver, Stub::Failure},
    {"DYAC6JUeZvM", "sceGnmSqttGetTraceWptr", Lib::Driver, Stub::Failure},
    {"pS2tjBxzJr4", "sceGnmSqttGetWrapCounts", Lib::Driver, Stub::Failure},
    {"rXV8az6X+fM", "sceGnmSqttGetWrapCounts2", Lib::Driver, Stub::Failure},
    {"ARS+TNLopyk", "sceGnmSqttGetWritebackLabels", Lib::Driver, Stub::Failure},
    {"X6yCBYPP7HA", "sceGnmSqttInit", Lib::Driver, Stub::Failure},
    {"2IJhUyK8moE", "sceGnmSqttSelectMode", Lib::Driver, Stub::Failure},
    {"QA5h6Gh3r60", "sceGnmSqttSelectTarget", Lib::Driver, Stub::Failure},
    {"F5XJY1XHa3Y", "sceGnmSqttSelectTokens", Lib::Driver, Stub::Failure},
    {"wJtaTpNZfH4", "sceGnmSqttSetCuPerfMask", Lib::Driver, Stub::Failure},
    {"kY4dsQh+SH4", "sceGnmSqttSetDceEventWrite", Lib::Driver, Stub::Failure},
    {"7XRH1CIfNpI", "sceGnmSqttSetHiWater", Lib::Driver, Stub::Failure},
    {"05YzC2r3hHo", "sceGnmSqttSetTraceBuffer2", Lib::Driver, Stub::Failure},
    {"ASUric-2EnI", "sceGnmSqttSetTraceBuffers", Lib::Driver, Stub::Failure},
    {"gPxYzPp2wlo", "sceGnmSqttSetUserData", Lib::Driver, Stub::Failure},
    {"d-YcZX7SIQA", "sceGnmSqttSetUserdataTimer", Lib::Driver, Stub::Failure},
    {"ru8cb4he6O8", "sceGnmSqttStartTrace", Lib::Driver, Stub::Failure},
    {"gVuGo1nBnG8", "sceGnmSqttStopTrace", Lib::Driver, Stub::Failure},
    {"OpyolX6RwS0", "sceGnmSqttSwitchTraceBuffer", Lib::Driver, Stub::Failure},
    {"dl5u5eGBgNk", "sceGnmSqttSwitchTraceBuffer2", Lib::Driver, Stub::Failure},
    {"QLzOwOF0t+A", "sceGnmSqttWaitForEvent", Lib::Driver, Stub::Failure},
    {"yhFCnaz5daw", "sceGnmUnregisterAllResourcesForOwner", Lib::Driver, Stub::Failure},
    {"fhKwCVVj9nk", "sceGnmUnregisterOwnerAndResources", Lib::Driver, Stub::Failure},
    {"k8EXkhIP+lM", "sceGnmUnregisterResource", Lib::Driver, Stub::Failure},
    {"iCO804ZgzdA", "sceGnmValidateCommandBuffers", Lib::Driver, Stub::ValidationOff},
    {"SXw4dZEkgpA", "sceGnmValidateDisableDiagnostics", Lib::Driver, Stub::ValidationOff},
    {"BgM3t3LvcNk", "sceGnmValidateDisableDiagnostics2", Lib::Driver, Stub::ValidationOff},
    {"qGP74T5OWJc", "sceGnmValidateDispatchCommandBuffers", Lib::Driver, Stub::ValidationOff},
    {"hsZPf1lON7E", "sceGnmValidateDrawCommandBuffers", Lib::Driver, Stub::ValidationOff},
    {"RX7XCNSaL6I", "sceGnmValidateGetDiagnosticInfo", Lib::Driver, Stub::ValidationOff},
    {"5SHGNwLXBV4", "sceGnmValidateGetDiagnostics", Lib::Driver, Stub::ValidationOff},
    {"HzMN7ANqYEc", "sceGnmValidateGetVersion", Lib::Driver, Stub::Silent},
    {"rTIV11nMQuM", "sceGnmValidateOnSubmitEnabled", Lib::Driver, Stub::Silent},
    {"MBMa6EFu4Ko", "sceGnmValidateResetState", Lib::Driver, Stub::ValidationOff},
    {"Q7t4VEYLafI", "sceGnmValidationRegisterMemoryCheckCallback", Lib::Driver, Stub::ValidationOff},
    {"xeTLfxVIQO4", "sceRazorCaptureCommandBuffersOnlyImmediate", Lib::Driver, Stub::CaptureFailed},
    {"9thMn+uB1is", "sceRazorCaptureCommandBuffersOnlySinceLastFlip", Lib::Driver, Stub::CaptureFailed},
    {"u9YKpRRHe-M", "sceRazorCaptureImmediate", Lib::Driver, Stub::CaptureFailed},
    {"4UFagYlfuAM", "sceRazorCaptureSinceLastFlip", Lib::Driver, Stub::CaptureFailed},
    {"f33OrruQYbM", "sceRazorIsLoaded", Lib::Driver, Stub::Silent},
    {"Bj0GWi1jWcM", "Func_063D065A2D6359C3", Lib::Driver, Stub::Missing},
    {"DKusr7JYQp0", "Func_0CABACAFB258429D", Lib::Driver, Stub::Missing},
    {"FQzzNvwumaM", "Func_150CF336FC2E99A3", Lib::Driver, Stub::Missing},
    {"F8pof57lLUk", "Func_17CA687F9EE52D49", Lib::Driver, Stub::Missing},
    {"GHC4n3Wca0U", "Func_1870B89F759C6B45", Lib::Driver, Stub::Missing},
    {"JvkCnvaKlV4", "Func_26F9029EF68A955E", Lib::Driver, Stub::Missing},
    {"MB49u6sJLbA", "Func_301E3DBBAB092DB0", Lib::Driver, Stub::Missing},
    {"MLr+Fyrxf+8", "Func_30BAFE172AF17FEF", Lib::Driver, Stub::Missing},
    {"Pmo+ggPZUxc", "Func_3E6A3E8203D95317", Lib::Driver, Stub::Missing},
    {"QP7vDGU0xDQ", "Func_40FEEF0C6534C434", Lib::Driver, Stub::Missing},
    {"QWuQed5Mus4", "Func_416B9079DE4CBACE", Lib::Driver, Stub::Missing},
    {"R3TYO7Tdv5o", "Func_4774D83BB4DDBF9A", Lib::Driver, Stub::Missing},
    {"UGePHM7rmgA", "Func_50678F1CCEEB9A00", Lib::Driver, Stub::Missing},
    {"VKLsX6TGJBM", "Func_54A2EC5FA4C62413", Lib::Driver, Stub::Missing},
    {"WpxSyDE4rms", "Func_5A9C52C83138AE6B", Lib::Driver, Stub::Missing},
    {"XSIZOjHqEUI", "Func_5D22193A31EA1142", Lib::Driver, Stub::Missing},
    {"clo23rtglI0", "Func_725A36DEBB60948D", Lib::Driver, Stub::Missing},
    {"gCGlAvphubs", "Func_8021A502FA61B9BB", Lib::Driver, Stub::Missing},
    {"nQAv4PpA8OY", "Func_9D002FE0FA40F0E6", Lib::Driver, Stub::Missing},
    {"nSl-NqcCi3E", "Func_9D297F36A7028B71", Lib::Driver, Stub::Missing},
    {"otfsenvPebM", "Func_A2D7EC7A7BCF79B3", Lib::Driver, Stub::Missing},
    {"qhKjy4mQhUo", "Func_AA12A3CB8990854A", Lib::Driver, Stub::Missing},
    {"rcjdwAUCC8Y", "Func_ADC8DDC005020BC6", Lib::Driver, Stub::Missing},
    {"sKhoi2ectC0", "Func_B0A8688B679CB42D", Lib::Driver, Stub::Missing},
    {"tIkCC1FXpf8", "Func_B489020B5157A5FF", Lib::Driver, Stub::Missing},
    {"ut57TBmRQN0", "Func_BADE7B4C199140DD", Lib::Driver, Stub::Missing},
    {"0VEbnc-7Pdk", "Func_D1511B9DCFFB3DD9", Lib::Driver, Stub::Missing},
    {"1TRGZJsC5Y4", "Func_D53446649B02E58E", Lib::Driver, Stub::Missing},
    {"2Lbo4o4e8KM", "Func_D8B6E8E28E1EF0A3", Lib::Driver, Stub::Missing},
    {"2T1zOhnddFQ", "Func_D93D733A19DD7454", Lib::Driver, Stub::Missing},
    {"3plUQ7wqgxc", "Func_DE995443BC2A8317", Lib::Driver, Stub::Missing},
    {"326VKBUMI-8", "Func_DF6E9528150C23FF", Lib::Driver, Stub::Missing},
    {"7LTGukH+M1A", "Func_ECB4C6BA41FE3350", Lib::Driver, Stub::Missing},
    {"dqPBvjFVpTA", "sceGnmDebugModuleReset", Lib::DebugModuleReset, Stub::Failure},
    {"RNPAItiMLIg", "sceGnmDebugReset", Lib::DebugReset, Stub::Failure},
    {"xMMot887QXE", "Func_C4C328B7CF3B4171", Lib::DebugReset, Stub::Missing},
    {"vbcR4Ken6AA", "sceGnmFindResources", Lib::ResourceRegistration, Stub::Failure},
    {"eLQbNsKeTkU", "sceGnmGetResourceRegistrationBuffers", Lib::ResourceRegistration, Stub::Failure},
    {"j6mSQs3UgaY", "sceGnmRegisterOwnerForSystem", Lib::ResourceRegistration, Stub::Failure},
    {"HEOIaxbuVTA", "Func_1C43886B16EE5530", Lib::ResourceRegistration, Stub::Missing},
    {"gQNwGezNDgE", "Func_81037019ECCD0E01", Lib::ResourceRegistration, Stub::Missing},
    {"v7QcBXR48L8", "Func_BFB41C057478F0BF", Lib::ResourceRegistration, Stub::Missing},
    {"5R1E24FRI4w", "Func_E51D44DB8151238C", Lib::ResourceRegistration, Stub::Missing},
    {"+RaJBCVJZVM", "Func_F916890425496553", Lib::ResourceRegistration, Stub::Missing},
    {"R3TYO7Tdv5o", "Func_4774D83BB4DDBF9A", Lib::WaitFreeSubmit, Stub::Missing},
    {"ut57TBmRQN0", "Func_BADE7B4C199140DD", Lib::WaitFreeSubmit, Stub::Missing},
};

s64 StubResult(std::size_t index) {
    const StubEntry& entry = Stubs[index];
    switch (entry.kind) {
    case Stub::Silent:
        return 0;
    case Stub::Missing: {
        static std::array<std::atomic<bool>, std::size(Stubs)> reported{};
        if (!reported[index].exchange(true, std::memory_order_relaxed)) {
            LOG_WARNING(Lib_GnmDriver, "{} is not implemented (returns 0)", entry.name);
        }
        return 0;
    }
    case Stub::Failure:
        return BbErr::GnmFailure;
    case Stub::ValidationOff:
        return BbErr::GnmValidationOff;
    case Stub::CaptureFailed:
        return BbErr::GnmCaptureFailed;
    case Stub::RazorNotLoaded:
        return BbErr::GnmRazorNotLoaded;
    case Stub::NoInterface:
        return s64(0x80000000u);
    case Stub::MinusOne:
        return -1;
    case Stub::Unreachable:
        UNREACHABLE_MSG("{} is not supported", entry.name);
    }
    return 0;
}

/// One entry point per stub (the index tells it which one it is).
template <std::size_t Index>
s64 PS4_SYSV_ABI StubFunction() {
    return StubResult(Index);
}

using StubFn = s64(PS4_SYSV_ABI*)();

template <std::size_t... Index>
constexpr std::array<StubFn, sizeof...(Index)> MakeStubFunctions(
    std::index_sequence<Index...>) {
    return {&StubFunction<Index>...};
}

} // namespace

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    // bbport (native GPU process): in the game process the GPU core is bb-gpu's.
    if (!BbRemote::FrontActive()) {
        LOG_INFO(Lib_GnmDriver, "Initializing presenter");
        liverpool = std::make_unique<BbCp::CommandProcessor>();
        presenter = std::make_unique<Vulkan::Presenter>(*g_window, liverpool.get());
    }

    s32 version = 0;
    sdk_version =
        Libraries::Kernel::sceKernelGetCompiledSdkVersion(&version) == BbErr::Ok ? u32(version) : 0;

    // BB_SUBMIT_LOCK=gpu: the command processor going idle releases the lock.
    BbIrq::Listen(BbIrq::Line::GpuIdle, &pacing, [](BbIrq::Line) { ReleaseSubmissionLock(); });

    const Export exports[] = {
        {"b0xyllnVY-I", Lib::Driver, Fn(sceGnmAddEqEvent)},
        {"b08AgtPlHPg", Lib::Driver, Fn(sceGnmAreSubmitsAllowed)},
        {"ihxrbsoSKWc", Lib::Driver, Fn(sceGnmBeginWorkload)},
        {"ffrNQOshows", Lib::Driver, Fn(sceGnmComputeWaitOnAddress)},
        {"5udAm+6boVg", Lib::Driver, Fn(sceGnmCreateWorkloadStream)},
        {"PVT+fuoS9gU", Lib::Driver, Fn(sceGnmDeleteEqEvent)},
        {"bX5IbRvECXk", Lib::Driver, Fn(sceGnmDingDong)},
        {"byXlqupd8cE", Lib::Driver, Fn(sceGnmDingDongForWorkload)},
        {"0BzLGljcwBo", Lib::Driver, Fn(sceGnmDispatchDirect)},
        {"Z43vKp5k7r0", Lib::Driver, Fn(sceGnmDispatchIndirect)},
        {"wED4ZXCFJT0", Lib::Driver, Fn(sceGnmDispatchIndirectOnMec)},
        {"nF6bFRUBRAU", Lib::Driver, Fn(sceGnmDispatchInitDefaultHardwareState)},
        {"HlTPoZ-oY7Y", Lib::Driver, Fn(sceGnmDrawIndex)},
        {"GGsn7jMTxw4", Lib::Driver, Fn(sceGnmDrawIndexAuto)},
        {"ED9-Fjr8Ta4", Lib::Driver, Fn(sceGnmDrawIndexIndirect)},
        {"thbPcG7E7qk", Lib::Driver, Fn(sceGnmDrawIndexIndirectCountMulti)},
        {"5q95ravnueg", Lib::Driver, Fn(sceGnmDrawIndexIndirectMulti)},
        {"oYM+YzfCm2Y", Lib::Driver, Fn(sceGnmDrawIndexOffset)},
        {"4v+otIIdjqg", Lib::Driver, Fn(sceGnmDrawIndirect)},
        {"f5QQLp9rzGk", Lib::Driver, Fn(sceGnmDrawIndirectMulti)},
        {"Idffwf3yh8s", Lib::Driver, Fn(sceGnmDrawInitDefaultHardwareState)},
        {"QhnyReteJ1M", Lib::Driver, Fn(sceGnmDrawInitDefaultHardwareState175)},
        {"0H2vBYbTLHI", Lib::Driver, Fn(sceGnmDrawInitDefaultHardwareState200)},
        {"yb2cRhagD1I", Lib::Driver, Fn(sceGnmDrawInitDefaultHardwareState350)},
        {"8lH54sfjfmU", Lib::Driver, Fn(sceGnmDrawInitToDefaultContextState)},
        {"im2ZuItabu4", Lib::Driver, Fn(sceGnmDrawInitToDefaultContextState400)},
        {"Fa3x75OOLRA", Lib::Driver, Fn(sceGnmEndWorkload)},
        {"UoYY0DWMC0U", Lib::Driver, Fn(sceGnmGetEqEventType)},
        {"Fwvh++m9IQI", Lib::Driver, Fn(sceGnmGetGpuCoreClockFrequency)},
        {"FFVZcCu3zWU", Lib::Driver, Fn(sceGnmGetOffChipTessellationBufferSize)},
        {"ln33zjBrfjk", Lib::Driver, Fn(sceGnmGetTheTessellationFactorRingBufferBaseAddress)},
        {"NfvOrNzy6sk", Lib::Driver, Fn(sceGnmInsertDingDongMarker)},
        {"7qZVNgEu+SY", Lib::Driver, Fn(sceGnmInsertPopMarker)},
        {"aPIZJTXC+cU", Lib::Driver, Fn(sceGnmInsertPushColorMarker)},
        {"W1Etj-jlW7Y", Lib::Driver, Fn(sceGnmInsertPushMarker)},
        {"jiItzS6+22g", Lib::Driver, Fn(sceGnmInsertSetMarker)},
        {"1qXLHIpROPE", Lib::Driver, Fn(sceGnmInsertWaitFlipDone)},
        {"RU74kek-N0c", Lib::Driver, Fn(sceGnmLogicalCuMaskToPhysicalCuMask)},
        {"29oKvKXzEZo", Lib::Driver, Fn(sceGnmMapComputeQueue)},
        {"A+uGq+3KFtQ", Lib::Driver, Fn(sceGnmMapComputeQueueWithPriority)},
        {"MYRtYhojKdA", Lib::Driver, Fn(sceGnmResetVgtControl)},
        {"KXltnCwEJHQ", Lib::Driver, Fn(sceGnmSetCsShader)},
        {"Kx-h-nWQJ8A", Lib::Driver, Fn(sceGnmSetCsShaderWithModifier)},
        {"X9Omw9dwv5M", Lib::Driver, Fn(sceGnmSetEmbeddedPsShader)},
        {"+AFvOEXrKJk", Lib::Driver, Fn(sceGnmSetEmbeddedVsShader)},
        {"FUHG8sQ3R58", Lib::Driver, Fn(sceGnmSetEsShader)},
        {"UJwNuMBcUAk", Lib::Driver, Fn(sceGnmSetGsShader)},
        {"VJNjFtqiF5w", Lib::Driver, Fn(sceGnmSetHsShader)},
        {"vckdzbQ46SI", Lib::Driver, Fn(sceGnmSetLsShader)},
        {"bQVd5YzCal0", Lib::Driver, Fn(sceGnmSetPsShader)},
        {"5uFKckiJYRM", Lib::Driver, Fn(sceGnmSetPsShader350)},
        {"cFCp0NX8wf0", Lib::Driver, Fn(sceGnmSetVgtControl)},
        {"gAhCn6UiU4Y", Lib::Driver, Fn(sceGnmSetVsShader)},
        {"xbxNatawohc", Lib::Driver, Fn(sceGnmSubmitAndFlipCommandBuffers)},
        {"Ga6r7H6Y0RI", Lib::Driver, Fn(sceGnmSubmitAndFlipCommandBuffersForWorkload)},
        {"zwY0YV91TTI", Lib::Driver, Fn(sceGnmSubmitCommandBuffers)},
        {"jRcI8VcgTz4", Lib::Driver, Fn(sceGnmSubmitCommandBuffersForWorkload)},
        {"yvZ73uQUqrk", Lib::Driver, Fn(sceGnmSubmitDone)},
        {"ArSg-TGinhk", Lib::Driver, Fn(sceGnmUnmapComputeQueue)},
        {"nLM2i2+65hA", Lib::Driver, Fn(sceGnmUpdateGsShader)},
        {"GNlx+y7xPdE", Lib::Driver, Fn(sceGnmUpdateHsShader)},
        {"4MgRw-bVNQU", Lib::Driver, Fn(sceGnmUpdatePsShader)},
        {"mLVL7N7BVBg", Lib::Driver, Fn(sceGnmUpdatePsShader350)},
        {"V31V01UiScY", Lib::Driver, Fn(sceGnmUpdateVsShader)},
        {"pF1HQjbmQJ0", Lib::Compat, Fn(sceGnmDrawInitToDefaultContextStateInternalCommand)},
        {"jajhf-Gi3AI", Lib::Compat, Fn(sceGnmDrawInitToDefaultContextStateInternalSize)},
        {"Fwvh++m9IQI", Lib::CoreClock, Fn(sceGnmGetGpuCoreClockFrequency)},
    };
    for (const Export& entry : exports) {
        sym->AddSymbol(entry.nid, LibName(entry.lib), "libSceGnmDriver",
                       Core::Loader::SymbolType::Function, entry.address);
    }
    static constexpr auto stub_functions =
        MakeStubFunctions(std::make_index_sequence<std::size(Stubs)>{});
    for (std::size_t i = 0; i < std::size(Stubs); ++i) {
        sym->AddSymbol(Stubs[i].nid, LibName(Stubs[i].lib), "libSceGnmDriver",
                       Core::Loader::SymbolType::Function, reinterpret_cast<u64>(stub_functions[i]));
    }
}

} // namespace BbGnm
