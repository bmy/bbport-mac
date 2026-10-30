// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include "shim/bb_env.h"
#include <chrono>
#include <filesystem>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <pthread.h>
#include <sys/resource.h>
#ifndef _WIN32
#include <sys/syscall.h>
#endif
#include <csignal>
#include <dlfcn.h>
#include <sys/uio.h>
#include <unistd.h>

#include <fmt/format.h>

#include "bbport_ce_stats.h"
#include "bbport_cpu.h"
#include "bbport_free_check.h"
#include "bbport_threads.h"
#include "bbport_timeline.h"
#include "bbport_toggles.h"
#include "bbport_write_log.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/polyfill_thread.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "bbport_sections.h"
#include "cp/command_processor.h"
#include "cp/cp_diagnostics.h"
#include "cp/cp_packets.h"
#include "cp/cp_tasks.h"
#include "cp/pm4_selftest.h"
#include "platform/interrupts.h"
#include "platform/videoout/videoout.h"
#include "render/vk_draw_pipe.h"
#include "render/vk_draw_prep.h"
#include "buffers/buffer_cache.h"
#include "render/rasterizer.h"

namespace BbCp {

std::array<u8, 48 * 1024> CommandProcessor::constant_ram;


namespace {

/// bbport (macOS, native GPU process): while every queue waits (a WAIT_REG_MEM, a label the GPU
/// has not written yet) and none moves, the command thread spins this long, then sleeps between
/// passes over the queues: 50 us at a time until 2 ms, 200 us until 20 ms, then 1 ms (a new
/// submission or command ends a sleep early). With the GPU stuck, the spin took a whole core for
/// as long (16 s in run C, 2026-10-24), a quarter of a Mac with 4 performance cores.
/// BB_CP_WAIT_SPIN_US (default 100 on arm64 macOS); -1, or anything not a number (the default
/// elsewhere, Rosetta too, whose tick rate the thresholds do not know): it spins until the wait
/// ends, as upstream.
long WaitSpinUs() {
    static const long us = [] {
        if (const char* env = bb_env("BB_CP_WAIT_SPIN_US"); env && *env) {
            char* end = nullptr;
            const long value = std::strtol(env, &end, 10);
            return end != env && *end == '\0' && value >= 0 ? value : -1L;
        }
#if defined(__APPLE__) && defined(__aarch64__)
        return 100L;
#else
        return -1L;
#endif
    }();
    return us;
}

/// A pass over the queues in which every one with work waited and none moved: nothing until the
/// spin time is up, then how long to sleep (it grows with the time waited). `wait_start`
/// (BbCpu::Ticks) is 0 at the first such pass of a wait.
std::chrono::microseconds WaitBackoff(u64& wait_start) {
    using namespace std::chrono_literals;
    const u64 now = BbCpu::Ticks();
    if (wait_start == 0) {
        wait_start = now;
        return 0us;
    }
    static const u64 spin = BbCpu::TicksFromUs(u64(std::max(0L, WaitSpinUs())));
    static const u64 two_ms = BbCpu::TicksFromUs(2'000);
    static const u64 twenty_ms = BbCpu::TicksFromUs(20'000);
    const u64 waited = now - wait_start;
    if (waited < spin) {
        return 0us;
    }
    return waited < two_ms ? 50us : waited < twenty_ms ? 200us : 1000us;
}

/// Where a queue's front work waits: another one (a packet further on, other work) is a new wait.
struct WaitPoint {
    const void* work = nullptr;
    const u32* packet = nullptr;
    const u32* level_at = nullptr;
    int level = -1;
    bool operator==(const WaitPoint&) const = default;
};

using Vulkan::DrawPipe;

constexpr u32 ComputeShOffset = 0x200; ///< COMPUTE_* in the SH bank
constexpr u32 ComputeWords = sizeof(ComputeProgram) / 4;

bool InComputeRange(u32 sh_offset) {
    return sh_offset >= ComputeShOffset && sh_offset <= ComputeShOffset + ComputeWords;
}

u32 BodyWords(u32 header) {
    return ((header >> 16) & 0x3FFF) + 1;
}

} // namespace

// ---- Registers -------------------------------------------------------------------------------

u64 CommandProcessor::HashRegisterPacket(u64 checksum, const u32* words, u32 count) {
    for (u32 i = 0; i < count; ++i) {
        checksum = (checksum ^ words[i]) * 0x9E3779B97F4A7C15ull;
        checksum ^= checksum >> 29;
    }
    return checksum;
}

void CommandProcessor::ApplyGraphicsRegisterPacket(Regs& regs, const u32* packet, u64& checksum,
                                                   RegDirty* dirty, Gcn::RegTouched* touched,
                                                   bool hash) {
    const u32 header = packet[0];
    const u32 body_words = BodyWords(header);
    const u32* body = packet + 1;
    const auto mark = [&](u32 index, u32 count) {
        if (dirty) {
            dirty->Mark(index, count);
        }
        if (touched) {
            touched->Mark(index, count);
        }
    };
    const auto set = [&](u32 bank) {
        const u32 index = bank + (body[0] & 0xFFFF);
        const u32 count = body_words - 1;
        std::memcpy(&regs.reg_array[index], body + 1, count * sizeof(u32));
        mark(index, count);
    };
    const auto write = [&](u32 index, u32 value) {
        regs.reg_array[index] = value;
        mark(index, 1);
    };
    switch (Header{header}.Opcode()) {
    case Op::ClearState:
        if (touched) {
            Gcn::ResetRegs(regs, *touched);
        } else {
            regs.SetDefaults();
        }
        if (dirty) {
            dirty->Clear();
            dirty->reset = true;
        }
        break;
    case Op::SetConfigReg:
        set(Bank::Config);
        break;
    case Op::SetContextReg:
        set(Bank::Context);
        break;
    case Op::SetShReg:
        // The compute program goes to the queue's compute state, not the file.
        if (InComputeRange(body[0] & 0xFFFF)) {
            return;
        }
        set(Bank::Sh);
        break;
    case Op::SetUconfigReg:
        set(Bank::Uconfig);
        break;
    case Op::IndexType:
        write(Reg::VgtDmaIndexType, body[0]);
        break;
    case Op::DrawIndex2:
        // max size, base low, base high, index count, draw initiator
        write(Reg::VgtDmaMaxSize, body[0]);
        write(Reg::VgtDmaBase, body[1]);
        write(Reg::VgtDmaBaseHi, (regs.reg_array[Reg::VgtDmaBaseHi] & ~0xFFu) | (body[2] & 0xFF));
        write(Reg::VgtNumIndices, body[3]);
        write(Reg::VgtDrawInitiator, body[4]);
        break;
    case Op::DrawIndexOffset2:
        // max size, index offset, index count, draw initiator
        write(Reg::VgtDmaMaxSize, body[0]);
        write(Reg::VgtNumIndices, body[2]);
        write(Reg::VgtDrawInitiator, body[3]);
        break;
    case Op::DrawIndexAuto:
        write(Reg::VgtNumIndices, body[0]);
        write(Reg::VgtDrawInitiator, body[1]);
        break;
    case Op::NumInstances:
        write(Reg::VgtNumInstances, body[0]);
        break;
    case Op::IndexBase:
        write(Reg::VgtDmaBase, body[0]);
        write(Reg::VgtDmaBaseHi, (regs.reg_array[Reg::VgtDmaBaseHi] & ~0xFFu) | (body[1] & 0xFF));
        break;
    case Op::IndexBufferSize:
        write(Reg::VgtNumIndices, body[0]);
        break;
    case Op::EventWrite:
        // A streamout flush: the offsets are updated at once (no streamout here).
        if ((body[0] & 0x3F) != EventWrite::SoVgtStreamoutFlush) {
            return;
        }
        write(Reg::CpStrmoutCntl, regs.reg_array[Reg::CpStrmoutCntl] | 1);
        break;
    default:
        return;
    }
    if (hash) {
        checksum = HashRegisterPacket(checksum, packet, body_words + 1);
    }
}

// ---- Thread ----------------------------------------------------------------------------------

CommandProcessor::CommandProcessor()
    : guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    thread = std::jthread{[this](std::stop_token stop) { Run(stop); }};
    if (!bb_env_off("BB_STALL_REPORT")) {
        stall_watch = std::jthread{[this](std::stop_token stop) { StallWatch(stop); }};
    }
}

CommandProcessor::~CommandProcessor() {
    if (stall_watch.joinable()) {
        stall_watch.request_stop();
        stall_watch.join();
    }
    thread.request_stop();
    {
        std::scoped_lock lock{submit_mutex};
        submit_cv.notify_all();
    }
    thread.join();
}

void CommandProcessor::SubmitDone(u64 frame) noexcept {
    std::scoped_lock lock{submit_mutex};
    if (frame != 0) {
        frame_ends.emplace_back(submissions_total, frame);
    }
    submit_done = true;
    submit_cv.notify_one();
}

void CommandProcessor::WaitGpuIdle() noexcept {
    std::unique_lock lock{submit_mutex};
    submit_cv.wait(lock, [this] { return num_submits == 0; });
}

bool CommandProcessor::IsGpuIdle() const {
    static const bool use_retired = !bb_env_off("BB_WORK_RETIRED");
    // BB_SUBMIT_LOCK=decode with honest labels: decoded is enough.
    return num_submits == 0 && (work_retired || !use_retired || BbGnm::SubmitLockOnDecode());
}

void CommandProcessor::RunCommands() {
    while (num_commands) {
        // Commands touch the caches and record into the scheduler: the draw recording thread
        // must be idle, for each one (one arriving between two checks ran beside it before).
        if (rasterizer) {
            rasterizer->DrainDrawPipe(DrawPipe::ReasonCommands);
        }
        Common::UniqueFunction<void> command;
        {
            std::scoped_lock lock{submit_mutex};
            command = std::move(commands.front());
            commands.pop_front();
            --num_commands;
        }
        command();
    }
}

void CommandProcessor::WaitForWork(std::stop_token stop) {
    const auto has_work = [this] { return num_commands || num_submits || submit_done; };
    // Fences (and GPU idle) wait for work not submitted yet; it goes out once
    // this thread has had nothing to do for BB_HONEST_IDLE_US (0.2 ms): the guest may be waiting.
    if (rasterizer && (rasterizer->HasUnsubmittedSignals() || !rasterizer->DrawPipeIdle())) {
        static const auto idle_wait = std::chrono::microseconds([] {
            const char* env = bb_env("BB_HONEST_IDLE_US");
            return env ? std::strtoll(env, nullptr, 10) : 200ll;
        }());
        std::unique_lock lock{submit_mutex};
        if (!Common::WaitFor(submit_cv, lock, idle_wait, has_work)) {
            lock.unlock();
            if (rasterizer->PipeSignals()) {
                // The recording thread submits once it gets there (waiting for it to run dry,
                // tens of ms in heavy scenes, held up the guest's next submission).
                rasterizer->RequestSignalFlush();
            } else {
                rasterizer->Flush();
            }
        }
    }
    const auto idle_start = std::chrono::steady_clock::now();
    BbTimeline::Note(BbTimeline::DecoderIdleBegin);
    // A frame comes in ~20 submissions: waking from the condition variable for each took tens of
    // microseconds. Spin first, with cores to spare (BB_GPU_SPIN_US, default 200 with 12+
    // hardware threads, else 0).
    static const auto spin_time = std::chrono::microseconds([] {
        if (const char* env = bb_env("BB_GPU_SPIN_US")) {
            return std::max(0, std::atoi(env));
        }
        return BbThreads::Available() >= 12 ? 200 : 0;
    }());
    if (spin_time.count() > 0) {
        const auto spin_until = idle_start + spin_time;
        for (u32 spins = 1; !has_work() && !stop.stop_requested(); ++spins) {
            BbCpu::Pause();
            if (!(spins & 255) && std::chrono::steady_clock::now() >= spin_until) {
                break;
            }
        }
    }
    {
        std::unique_lock lock{submit_mutex};
        Common::CondvarWait(submit_cv, lock, stop, has_work);
    }
    BbStats::gpu_idle_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                           std::chrono::steady_clock::now() - idle_start)
                                           .count()),
                                   std::memory_order_relaxed);
    BbTimeline::Note(BbTimeline::DecoderIdleEnd);
}

void CommandProcessor::StallWatch(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:StallWatch");
    using Clock = std::chrono::steady_clock;
    constexpr auto Limit = std::chrono::seconds(5);
    u64 last_submitted = ~0ull, last_decoded = ~0ull;
    auto submitted_at = Clock::now(), decoded_at = Clock::now();
    bool reported = false;
    while (!stop.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        u64 submitted, decoded, pending;
        {
            std::scoped_lock lock{submit_mutex};
            submitted = submissions_total;
            decoded = decoded_total;
            pending = num_submits;
        }
        const auto now = Clock::now();
        if (submitted != last_submitted) {
            last_submitted = submitted;
            submitted_at = now;
            reported = false;
        }
        if (decoded != last_decoded || pending == 0) {
            last_decoded = decoded;
            decoded_at = now;
        }
        if (reported || submitted == 0) {
            continue;
        }
        if (now - decoded_at >= Limit) {
            reported = true;
            ReportStall("the queues have made no progress for 5 s");
        } else if (now - submitted_at >= Limit) {
            reported = true;
            ReportStall("the guest has submitted nothing for 5 s");
        }
    }
}

void CommandProcessor::ReportStall(const char* why) {
    std::string out;
    {
        std::scoped_lock lock{submit_mutex};
        out = fmt::format("GPU stall report: {} (submitted {}, decoded {}, waiting {}, commands "
                          "{})\n",
                          why, submissions_total, decoded_total, num_submits.load(),
                          num_commands.load());
    }
    // What each queue is at (read beside the command thread: a snapshot, maybe torn).
    for (u32 q = 0; q < num_mapped_queues && q < NumTotalQueues; ++q) {
        Queue& queue = queues[q];
        std::scoped_lock lock{queue.mutex};
        if (q != GfxQueueId && asc_queues.is_allocated(Common::SlotId{q - 1})) {
            const auto& ring = asc_queues[Common::SlotId{q - 1}];
            u32 read = 0;
            const bool readable = Diagnostics::ReadGuest(ring.read_addr, &read, sizeof(read));
            out += fmt::format("  compute queue {} (pipe {}): ring {:#x}, {} dwords, read pointer "
                               "{}{}, {} submissions queued\n",
                               q, ring.pipe_id, ring.map_addr, ring.ring_size_dw,
                               readable ? fmt::format("{}", read) : std::string("?"),
                               ring.tmp_dwords ? fmt::format(" ({} dwords of a split packet)",
                                                             ring.tmp_dwords)
                                               : std::string{},
                               queue.works.size());
        } else if (!queue.works.empty()) {
            out += fmt::format("  queue {}: {} submissions queued\n", q, queue.works.size());
        }
        if (queue.works.empty()) {
            continue;
        }
        const Work& work = *queue.works.front();
        if (!work.started || work.walk.top < 0) {
            out += "    first one not started\n";
            continue;
        }
        const Level& level = work.walk.levels[work.walk.top];
        u32 header = 0;
        const bool readable =
            level.at < level.end && Diagnostics::ReadGuest(level.at, &header, sizeof(header));
        out += fmt::format("    first one at depth {}: {:#x}, dword {} of {}, packet {}{}\n",
                           work.walk.top, reinterpret_cast<uintptr_t>(level.at),
                           level.at - level.begin, level.end - level.begin,
                           readable && (header >> 30) == 3
                               ? Diagnostics::OpName(u32(Header{header}.Opcode()))
                               : fmt::format("{:#x}", header),
                           work.waiting_at ? " (waiting)" : "");
    }
    if (rasterizer) {
        out += rasterizer->DescribeSignals();
    }
    out += "  last labels written for the guest (oldest first):\n";
    out += Diagnostics::RecentLabels(40);
    // The command thread's frames (the runtime's SIGUSR2 handler prints its frame-pointer chain
    // to stderr; addresses minus this library's base give the source lines).
#ifndef __APPLE__ // bbport (macOS): no SIGUSR2 is sent there (no tgkill), so no stack follows
    Dl_info self{};
    if (dladdr(reinterpret_cast<void*>(&BodyWords), &self)) {
        out += fmt::format("  command thread {} stack follows ({} at {:#x}):\n", thread_tid,
                           self.dli_fname ? self.dli_fname : "?",
                           reinterpret_cast<uintptr_t>(self.dli_fbase));
    }
#endif
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
    Diagnostics::DumpRolledBackLabels(40);
    std::fflush(stderr);
#if !defined(_WIN32) && !defined(__APPLE__) // bbport (macOS): no tgkill
    if (thread_tid != 0) {
        syscall(SYS_tgkill, getpid(), thread_tid, SIGUSR2);
    }
#endif
#if !defined(_WIN32) && !defined(__APPLE__) // bbport (macOS): no tgkill or /proc/self/task
    // Every thread of the process prints its frames and registers (the runtime's SIGUSR2 handler):
    // when the guest itself stops (its job system, a lock), they say what each one waits for.
    // BB_STALL_THREADS=0: the command thread only.
    static const bool all_threads = !bb_env_off("BB_STALL_THREADS");
    if (all_threads && !bb_env("BB_STALL_THREADS_DONE")) {
        setenv("BB_STALL_THREADS_DONE", "1", 1); // once per run
        std::fputs("  all threads follow (frames: image-relative, then registers and stack):\n", stdout);
        std::fflush(stdout);
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator("/proc/self/task", ec)) {
            const long tid = std::strtol(entry.path().filename().c_str(), nullptr, 10);
            if (tid > 0 && tid != long(thread_tid)) {
                syscall(SYS_tgkill, getpid(), tid, SIGUSR2);
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
        }
    }
#endif
}

void CommandProcessor::Run(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:GpuComm");
    BbThreads::MakeCritical();
    BbSections::command_thread = true;
#ifndef __APPLE__ // bbport (macOS): no clock for another thread's CPU time
    if (clockid_t clock; pthread_getcpuclockid(pthread_self(), &clock) == 0) {
        BbStats::gpu_thread_clock.store(int(clock));
    }
#endif
    thread_id = std::this_thread::get_id();
    thread_tid = u32(gettid());
    const bool wait_backoff = WaitSpinUs() >= 0;
    if (wait_backoff) {
        std::printf("Command processor: while every queue waits, it spins %ld us, then sleeps "
                    "between checks (BB_CP_WAIT_SPIN_US; -1: it spins, as upstream)\n",
                    WaitSpinUs());
    }
    u64 wait_start = 0; ///< BbCpu::Ticks at the first check of the current wait (0: none)
    std::array<WaitPoint, NumTotalQueues> wait_points{}; ///< where each queue waited last

    while (!stop.stop_requested()) {
        WaitForWork(stop);
        if (stop.stop_requested()) {
            break;
        }
        // The queues in turn: each one's front submission runs until it is done or waits.
        current_queue = -1;
        bool pass_waited = false; // this pass over the queues: one waited
        bool pass_moved = false;  // one made progress, or waits somewhere new
        while (num_submits || num_commands) {
            RunCommands();
            current_queue = int((current_queue + 1) % num_mapped_queues);
            if (wait_backoff && current_queue == 0) {
                if (pass_moved) {
                    wait_start = 0;
                } else if (pass_waited) {
                    if (const auto sleep = WaitBackoff(wait_start); sleep.count() > 0) {
                        std::unique_lock lock{submit_mutex};
                        submit_cv.wait_for(lock, sleep); // a submission or command wakes it
                    }
                }
                pass_waited = pass_moved = false;
            }
            Queue& queue = queues[current_queue];
            Work* work;
            {
                std::scoped_lock lock{queue.mutex};
                if (queue.works.empty()) {
                    continue;
                }
                work = queue.works.front().get();
            }
            if (!work->started) {
                if (rasterizer) {
                    rasterizer->ProcessDeadImagesInOrder(); // freed before this was submitted
                    rasterizer->NewGuestWorkInOrder();
                }
                if (work->stamp > BbStats::started_submit_stamp.load(std::memory_order_relaxed)) {
                    BbStats::started_submit_stamp.store(work->stamp, std::memory_order_release);
                }
            }
            if (work != last_decoded) {
                last_decoded = work;
                BbTimeline::Note(BbTimeline::DecodeBegin, u64(current_queue), decoded_total);
            }
            Status status;
            {
                BB_SECTION(ADecode);
                status = current_queue == GfxQueueId ? RunGraphics(*work) : RunCompute(*work);
            }
            if (status == Status::Wait) {
                BbCpu::Pause(); // the other queues' turn
                if (wait_backoff) {
                    const auto& walk = work->walk;
                    const WaitPoint point{work, work->waiting_at,
                                          walk.top >= 0 ? walk.levels[walk.top].at : nullptr,
                                          walk.top};
                    if (point != wait_points[current_queue]) {
                        wait_points[current_queue] = point;
                        pass_moved = true; // decoded up to another wait: that one starts afresh
                    }
                    pass_waited = true;
                }
                continue;
            }
            pass_moved = true;
            BbTimeline::Note(BbTimeline::DecodeEnd, u64(current_queue), decoded_total + 1);
            bool frame_end;
            {
                std::scoped_lock lock{queue.mutex};
                queue.works.pop_front();
                last_decoded = nullptr;
                --num_submits;
                std::scoped_lock submit_lock{submit_mutex};
                ++decoded_total;
                frame_end = !frame_ends.empty() && frame_ends.front().first <= decoded_total;
                submit_cv.notify_all();
            }
            if (frame_end) {
                SignalDecodedFrames(); // BB_SUBMIT_LOCK=frame: a frame's last submission decoded
            }
        }
        wait_start = 0;
        wait_points = {};
        AfterBatch();
    }
}

void CommandProcessor::AfterBatch() {
    if (submit_done) {
        if (rasterizer && rasterizer->PipeSignals()) {
            // BB_PIPE_SIGNALS: the frame's cache work runs on the recording thread after the
            // frame (this thread decodes the next one meanwhile).
            const u64 started = BbStats::started_submit_stamp.load(std::memory_order_acquire);
            rasterizer->RunInOrder(
                [](Vulkan::Rasterizer& r, const u8* data) {
                    u64 started_stamp;
                    std::memcpy(&started_stamp, data, sizeof(started_stamp));
                    r.OnSubmit(started_stamp);
                    r.Flush();
                    // BB_EXP_FINISH_FRAMES=1 (experiment): the GPU finishes each frame before the next.
                    static const bool finish = bb_env("BB_EXP_FINISH_FRAMES") != nullptr;
                    if (finish) {
                        r.Finish();
                    }
                },
                &started, sizeof(started));
        } else if (rasterizer) {
            rasterizer->OnSubmit(BbStats::started_submit_stamp.load(std::memory_order_acquire));
            rasterizer->Flush();
        }
        submit_done = false;
    }
    SignalDecodedFrames();

    // The guest takes GPU idle (sceGnmSubmitDone) as its work being done and frees the objects
    // that hold its fence labels: they must be written before, by the draw recording thread and
    // the fences it deferred. (BB_PIPE_SIGNALS: that thread does both in order, below.)
    const bool pipe_signals = rasterizer && rasterizer->PipeSignals();
    if (rasterizer && !pipe_signals) {
        rasterizer->DrainDrawPipe(DrawPipe::ReasonSubmissionEnd);
    }
    // GPU idle only if nothing was submitted meanwhile: the drain and the deferred fences take
    // milliseconds, and a submission plus sceGnmSubmitDone in that time would release the lock for
    // work not even decoded (late fences in the guest's heap, command buffers reused as garbage).
    if (rasterizer) {
        u64 generation;
        bool decoded;
        {
            std::scoped_lock lock{submit_mutex};
            generation = submissions_total;
            decoded = num_submits == 0;
        }
        // BB_SUBMIT_LOCK=decode: everything is decoded; the next submission need not wait for
        // the GPU (the idle interrupt still does).
        if (BbGnm::SubmitLockOnDecode() && decoded) {
            BbGnm::ReleaseSubmissionLock();
        }
        if (pipe_signals) {
            struct IdleSignal {
                CommandProcessor* self;
                u64 generation;
            } const signal{this, generation};
            rasterizer->RunInOrder(
                [](Vulkan::Rasterizer& r, const u8* data) {
                    IdleSignal idle;
                    std::memcpy(&idle, data, sizeof(idle));
                    r.SignalAfterGpu([idle] { idle.self->SignalGpuIdle(idle.generation); });
                    r.SubmitForSignals(); // at once if the GPU has run dry
                },
                &signal, sizeof(signal), BbToggle::PipelinedTasks, false);
            return;
        }
        rasterizer->SignalAfterGpu([this, generation] { SignalGpuIdle(generation); });
        return;
    }
    bool idle;
    {
        std::scoped_lock lock{submit_mutex};
        idle = num_submits == 0;
        if (idle) {
            work_retired = true;
        }
    }
    if (idle) {
        BbIrq::Raise(BbIrq::Line::GpuIdle);
    }
}

void CommandProcessor::SignalDecodedFrames() {
    u64 frame = 0;
    {
        std::scoped_lock lock{submit_mutex};
        while (!frame_ends.empty() && frame_ends.front().first <= decoded_total) {
            frame = frame_ends.front().second;
            frame_ends.pop_front();
        }
    }
    if (frame == 0) {
        return;
    }
    if (!rasterizer) {
        BbGnm::NoteFramesRetired(frame);
        return;
    }
    // Everything of the frame recorded, then submitted: finished once the GPU has executed it.
    if (rasterizer->PipeSignals()) {
        rasterizer->RunInOrder(
            [](Vulkan::Rasterizer& r, const u8* data) {
                u64 retired;
                std::memcpy(&retired, data, sizeof(retired));
                BbTimeline::Note(BbTimeline::PipeTask, 7, r.CurrentTick());
                r.NoteFrameEndTask(retired);
                r.SignalAfterGpu([retired] { BbGnm::NoteFramesRetired(retired); });
                r.Flush();
            },
            &frame, sizeof(frame), BbToggle::PipelinedTasks, false);
        return;
    }
    rasterizer->DrainDrawPipe(DrawPipe::ReasonSubmissionEnd);
    rasterizer->NoteFrameEndTask(frame);
    rasterizer->SignalAfterGpu([frame] { BbGnm::NoteFramesRetired(frame); });
    rasterizer->Flush();
}

void CommandProcessor::SignalGpuIdle(u64 generation) {
    bool idle;
    {
        std::scoped_lock lock{submit_mutex};
        idle = num_submits == 0 && submissions_total == generation;
        if (idle) {
            work_retired = true;
        }
    }
    if (idle) {
        BbIrq::Raise(BbIrq::Line::GpuIdle);
    }
}

// ---- Submissions -----------------------------------------------------------------------------

void CommandProcessor::SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb, bool force_copy) {
    Queue& queue = queues[GfxQueueId];
    auto work = std::make_unique<Work>();
    work->queue = GfxQueueId;
    // BB_COPY_GPU_BUFFERS: one copy per submission, alive until it is decoded (the guest reused
    // the memory before this thread got to it: "PM4 type 0" while an area loads).
    if (force_copy || EmulatorSettings.IsCopyGpuBuffers()) {
        auto copy = std::make_shared<SubmittedCopy>();
        copy->data.resize(dcb.size() + ccb.size());
        std::memcpy(copy->data.data(), dcb.data(), dcb.size_bytes());
        std::memcpy(copy->data.data() + dcb.size(), ccb.data(), ccb.size_bytes());
        copy->guest_dcb = force_copy ? copy->data.data() : dcb.data();
        copy->dcb_dwords = dcb.size();
        copy->guest_ccb = force_copy ? copy->data.data() + dcb.size() : ccb.data();
        copy->ccb_dwords = ccb.size();
        copy->submitted = std::chrono::steady_clock::now();
        Diagnostics::InspectSubmittedCopy(*copy);
        Diagnostics::RepairSubmittedCopy(
            *copy, BbStats::guest_submit_stamp.load(std::memory_order_relaxed) + 1);
        Diagnostics::RememberSubmittedCopy(copy);
        dcb = {copy->data.data(), copy->dcb_dwords};
        ccb = {copy->data.data() + copy->dcb_dwords, copy->ccb_dwords};
        work->copy = std::move(copy);
    }
    work->dcb = dcb;
    work->ccb = ccb;
    // The draw preparation workers' copy is made before the queue lock, which this thread needs.
    auto* draw_prep = rasterizer ? &rasterizer->GetDrawPreparation() : nullptr;
    auto prepared = draw_prep ? draw_prep->Build(dcb) : nullptr;
    {
        // Numbering, enqueueing and queueing under one lock keep the three orders the same.
        std::scoped_lock lock{queue.mutex};
        work->stamp = BbStats::guest_submit_stamp.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (draw_prep) {
            work->seq = gfx_submit_seq++;
            draw_prep->Enqueue(work->seq, std::move(prepared));
        }
        BbFreeCheck::NoteSubmit(work->seq, work->copy ? work->copy->guest_dcb : dcb.data(),
                                dcb.size_bytes());
        queue.works.push_back(std::move(work));
    }
    std::scoped_lock lock{submit_mutex};
    ++num_submits;
    ++submissions_total;
    BbTimeline::Note(BbTimeline::GuestSubmit, 0, submissions_total);
    work_retired = false;
    submit_cv.notify_one();
}

void CommandProcessor::SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    ASSERT_MSG(gnm_vqid > 0 && gnm_vqid < NumTotalQueues, "bad compute queue {}", gnm_vqid);
    BbStats::asc_submits.fetch_add(1, std::memory_order_relaxed);
    Queue& queue = queues[gnm_vqid];
    auto work = std::make_unique<Work>();
    work->queue = gnm_vqid;
    work->dcb = acb;
    {
        std::scoped_lock lock{queue.mutex};
        work->stamp = BbStats::guest_submit_stamp.fetch_add(1, std::memory_order_acq_rel) + 1;
        queue.works.push_back(std::move(work));
    }
    std::scoped_lock lock{submit_mutex};
    num_mapped_queues = std::max(num_mapped_queues, gnm_vqid + 1);
    ++num_submits;
    ++submissions_total;
    work_retired = false;
    submit_cv.notify_one();
}

} // namespace BbCp
