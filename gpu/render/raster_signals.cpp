// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, fences, labels, timestamps and occlusion results the guest observes, written
// when the GPU has done the work before them, and the GPU watchdog.

#include <fmt/format.h>
#include "shim/bb_env.h"

#include "bbport_free_check.h"
#include "render/raster_internal.h"
#include "bbport_copy.h"
#include "bbport_threads.h"
#include "bbport_toggles.h"
#include <pthread.h>

namespace Vulkan {

void Rasterizer::PublishSignals() {
    // Only the recording thread, or the GPU command thread while it is idle (drained), touches
    // them; other threads (a guest thread reading back) leave them to those.
    const bool on_b = DrawPipe::OnStageB();
    if (!(on_b || (OnStageA() && DrawPipeIdle()))) {
        return;
    }
    // Fences of the packets before this one (stage B), or of all (stage A, drained).
    const u64 upto = draw_pipe ? (on_b ? draw_pipe->Consumed() : draw_pipe->Head()) : 0;
    if (local_signals.empty()) {
        signals_published_upto.store(upto, std::memory_order_release);
        return;
    }
    bool was_empty;
    {
        std::scoped_lock lk{gpu_signals_mutex};
        if (!gpu_signal_started) {
            gpu_signal_thread =
                std::jthread([this](std::stop_token token) { GpuSignalLoop(token); });
            gpu_signal_started = true;
        }
        was_empty = gpu_signals.empty();
        for (auto& pending : local_signals) {
            gpu_signals.push_back(std::move(pending));
        }
    }
    local_signals.clear();
    local_signal_count.store(0, std::memory_order_release);
    signals_published_upto.store(upto, std::memory_order_release);
    if (was_empty) {
        gpu_signals_cv.notify_one();
    }
}

void Rasterizer::SignalAfterGpu(std::function<void()> signal, VAddr label, u64 value) {
    if (DrawPipe::OnStageB()) {
        // Kept here until the end of the guest submission or the next flush (in tick order).
        local_signals.push_back({scheduler.CurrentTick(), std::move(signal), label, value});
        local_signal_count.store(u32(local_signals.size()), std::memory_order_release);
        return;
    }
    if (OnStageA() && DrawPipeIdle()) {
        PublishSignals(); // the recording thread's come first (it is idle)
    }
    bool was_empty;
    {
        std::scoped_lock lk{gpu_signals_mutex};
        if (!gpu_signal_started) {
            gpu_signal_thread =
                std::jthread([this](std::stop_token token) { GpuSignalLoop(token); });
            gpu_signal_started = true;
        }
        was_empty = gpu_signals.empty();
        gpu_signals.push_back({scheduler.CurrentTick(), std::move(signal), label, value});
    }
    // The signal thread waits on the condition only with nothing queued (else it waits for the
    // GPU and takes the rest when it gets back): no notify per fence (~60 a frame).
    if (was_empty) {
        gpu_signals_cv.notify_one();
    }
}

void Rasterizer::RequestSignalFlush() {
    if (signal_flush_requested.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    static constexpr u8 none = 0;
    RunInOrder(
        [](Rasterizer& self, const u8*) {
            self.signal_flush_requested.store(false, std::memory_order_release);
            self.PublishSignals();
            if (self.HasUnsubmittedSignals()) {
                self.Flush();
            }
        },
        &none, 0, BbToggle::PipelinedTasks, false);
}

bool Rasterizer::SignalsPublished(u64 position) const {
    return !draw_pipe || signals_published_upto.load(std::memory_order_acquire) >= position;
}

bool Rasterizer::HasUnsubmittedSignals() {
    if (local_signal_count.load(std::memory_order_acquire) != 0) {
        return true; // for the current tick, held by the recording thread
    }
    std::scoped_lock lk{gpu_signals_mutex};
    return !gpu_signals.empty() && gpu_signals.back().tick >= scheduler.CurrentTick();
}

std::string Rasterizer::DescribeSignals() {
    auto* semaphore = scheduler.GetWorkSemaphore();
    semaphore->Refresh();
    std::string out = fmt::format(
        "  GPU finished tick {}, recording {}, handed to the queue {}; draw pipe {} (backlog {} "
        "bytes); {} signals held by the recording thread\n",
        semaphore->KnownGpuTick(), scheduler.CurrentTick(), scheduler.SubmittedTick(),
        DrawPipeIdle() ? "idle" : "busy", draw_pipe ? draw_pipe->Backlog() : 0,
        local_signal_count.load(std::memory_order_acquire));
    std::scoped_lock lk{gpu_signals_mutex};
    out += fmt::format("  {} signals wait for the GPU", gpu_signals.size());
    if (!gpu_signals.empty()) {
        const auto& front = gpu_signals.front();
        const auto& back = gpu_signals.back();
        out += fmt::format(": first tick {} (label {:#x} = {:#x}), last tick {} (label {:#x} = "
                           "{:#x})",
                           front.tick, front.label, front.value, back.tick, back.label, back.value);
    }
    out += '\n';
    return out;
}

bool Rasterizer::WriteGuestMemory(VAddr address, const void* data, u32 size) {
    if (size == 0 || (address | size) % 4 != 0) {
        return false;
    }
    const auto [buffer, offset] = buffer_cache.ObtainBuffer(address, size, true);
    if (!buffer_cache.IsInPlace(address, size)) {
        return false;
    }
    runtime.UpdateBuffer(buffer, offset, {static_cast<const u8*>(data), size});
    return true;
}

bool Rasterizer::WriteDataOnGpu(VAddr address, const void* data, u32 size) {
    static const bool enabled = [] {
        const char* env = bb_env("BB_GPU_COMMAND_WRITES");
        return !env || env[0] != '0';
    }();
    if (!enabled || size == 0 || (address | size) % 4 != 0) {
        return false;
    }
    // bbport BB_LAYER_MEMORY: a range whole in a mirror is written there too (asked first: the
    // target demotes it otherwise).
    const auto mirror = buffer_cache.CommandWriteMirror(address, size);
    const auto target = buffer_cache.CommandWriteTarget(address, size);
    if (!target) {
        return false;
    }
    if (mirror) {
        runtime.UpdateBuffer(mirror->first, mirror->second, {static_cast<const u8*>(data), size});
    }
    runtime.UpdateBuffer(target->first, target->second, {static_cast<const u8*>(data), size});
    // Bulk WRITE_DATA can initialize texture memory (including the private FX-history
    // clears). A cached Vulkan image at this address must be reuploaded from the new
    // buffer contents if fully overwritten. Otherwise a reload can keep the previous owner's image despite
    // the clear. Scalar fence/label writes retain their fast path.
    if (size > sizeof(u64)) {
        texture_cache.InvalidateMemoryFromGPU(address, size);
    }
    if (size <= sizeof(u64)) {
        u64 value = 0;
        std::memcpy(&value, data, size);
        SignalAfterGpu([] {}, address, value);
    }
    return true;
}

bool Rasterizer::OcclusionTranslated() {
    // Vulkan occlusion queries (BB_OCCLUSION_QUERIES=0: the fixed sequence written when decoded,
    // everything visible, as in the 0.3 model). A pass drawn at another size than the game's target
    // (the reduced scene with an upscaler, the output-size UI) has its samples scaled to the game's
    // resolution (OcclusionQueries::SetPassScale): the game read 1/4 or 1/9 of the PS4's counts
    // before (a lamp in Oedon Chapel: 84 286 and 37 455 samples instead of 337 280). With them its
    // glow was once seen to flash grey; not reproduced since (8 directions, the camera still, and
    // turning; 640x360 and 3840x2160 output).
    static const bool enabled = !bb_env_off("BB_OCCLUSION_QUERIES");
    // Experiment bit 4 of BB_TOGGLE_FILE: the fixed sequence while the game runs (A/B).
    return enabled && !BbToggle::Experiment(4);
}

bool Rasterizer::OcclusionEvent(VAddr address, u32 pairs) {
    if (!OcclusionTranslated() || pairs == 0) {
        return false;
    }
    if (!occlusion) {
        occlusion = std::make_unique<OcclusionQueries>(instance, scheduler, buffer_cache);
    }
    occlusion->Event(address, pairs);
    return true;
}

u64 Rasterizer::OcclusionEvents() const {
    return occlusion ? occlusion->Events() : 0;
}

bool Rasterizer::OnOcclusionPageAccess(VAddr addr, u64 rip, bool write, bool gpu_thread) {
    return occlusion && occlusion->OnAccess(addr, rip, write, gpu_thread);
}

bool Rasterizer::OnVramDataAccess(VAddr addr, bool gpu_thread) {
    using Caller = VideoCore::BufferCache::TrapCaller;
    // Who reads GPU data there (the first few, by thread name).
    static std::atomic<u32> told{0};
    if (told.fetch_add(1, std::memory_order_relaxed) < 16) {
        char name[32] = "?";
        pthread_getname_np(pthread_self(), name, sizeof(name));
        std::printf("Layer read trap: %#llx read by thread %s (%s)\n", (unsigned long long)addr,
                    name,
                    gpu_thread ? (OnStageA() ? "GPU command thread" : "recording thread")
                    : runtime_fault_recover ? "speculative reader"
                    : BbCopy::OnCopyThread() || BbThreads::gpu_side_helper ? "GPU-side helper"
                                             : "guest or other");
    }
    if (!gpu_thread) {
        if (runtime_fault_recover) {
            // A speculative reader (draw preparation) gives up at its recovery point; the draw
            // is prepared by the GPU command thread. It must not wait for that thread.
            return false;
        }
        if (BbCopy::OnCopyThread() || BbThreads::gpu_side_helper) {
            // The submissions wait for the copy threads: never wait for the command thread here
            // (as the recording thread: submitted GPU data is waited for, unsubmitted read as is).
            return buffer_cache.LayerReadTrapHit(addr, Caller::RecordingThread);
        }
        return buffer_cache.LayerReadTrapHit(addr, Caller::Guest);
    }
    if (OnStageA()) {
        // The copy back may flush the scheduler and record copies: the recording thread idle.
        DrainDrawPipe();
        return buffer_cache.LayerReadTrapHit(addr, Caller::CommandThread);
    }
    return buffer_cache.LayerReadTrapHit(addr, Caller::RecordingThread);
}

bool Rasterizer::WriteTimestampOnGpu(VAddr address, bool end_of_pipe) {
    // BB_GPU_TIMESTAMPS=0: written by the CPU (EOP: once the GPU is done; COPY_DATA: when decoded).
    static const bool enabled = !bb_env_off("BB_GPU_TIMESTAMPS");
    if (!enabled || address % 8 != 0) {
        return false;
    }
    if (!timestamps) {
        timestamps = std::make_unique<GpuTimestamps>(instance, scheduler, buffer_cache);
    }
    return timestamps->Write(address, end_of_pipe);
}

bool Rasterizer::WriteLabelOnGpu(VAddr address, u64 value, u32 num_bytes) {
    // Off by default on GFX8 (Polaris, RX 4xx/5xx): the guest stalls there with labels written by
    // the GPU (#203, RX 580: nothing submitted for 5 s after the first cutscene; BB_FREE_CHECK=1,
    // which also turns them off, fixes it). BB_GPU_LABELS=1 forces them on.
    static const bool enabled = [this] {
        const char* env = bb_env("BB_GPU_LABELS");
        if (!env && instance.IsAmdGfx8()) {
            std::printf("GPU: end-of-pipe labels are written by a CPU thread after the GPU "
                        "(GFX8; BB_GPU_LABELS=1: by the GPU)\n");
            return false;
        }
        return !env || env[0] != '0';
    }();
    if (!enabled || BbFreeCheck::Enabled() || (num_bytes != 4 && num_bytes != 8) ||
        address % num_bytes != 0) {
        return false;
    }
    const auto target = buffer_cache.GuestChunkSource(address, num_bytes);
    if (!target) {
        return false;
    }
    static const bool force_portable = [] {
        const char* env = bb_env("BB_GPU_LABELS_PORTABLE");
        return env && env[0] != '0';
    }();
    const bool marker = instance.IsBufferMarkerSupported() && !force_portable;
    static const bool announced = [&] {
        std::printf("GPU: end-of-pipe labels are written by the GPU (%s; "
                    "BB_GPU_LABELS=0: by a CPU thread after the GPU)\n",
                    marker ? "AMD buffer markers" : "portable Vulkan updates");
        return true;
    }();
    (void)announced;
    if (occlusion) {
        occlusion->WriteBeforeLabel(); // the occlusion results before it (stream order)
    }
    if (!marker) {
        // Core Vulkan transfer commands are forbidden inside a render pass. Preserve command
        // order by ending it here, then restart rendering normally at the next draw.
        scheduler.EndRendering();
        runtime.FlushBarriers();
    }
    // The 64-bit label as two 32-bit end-of-pipe writes, the high half first: whoever sees the
    // new low half (the guest polls that) sees the new high half too.
    scheduler.Record([buffer = target->first->Handle(), offset = target->second, value,
                      num_bytes, marker](vk::CommandBuffer cmdbuf) {
        if (!marker) {
            RecordPortableGpuLabel(cmdbuf, buffer, offset, value, num_bytes);
            return;
        }
        if (num_bytes == 8) {
            cmdbuf.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, buffer,
                                        offset + 4, static_cast<u32>(value >> 32));
        }
        cmdbuf.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, buffer, offset,
                                    static_cast<u32>(value));
    });
    if (!marker) {
        runtime.AccessBuffer(target->first, target->second, num_bytes,
                             vk::PipelineStageFlagBits2::eTransfer,
                             vk::AccessFlagBits2::eTransferWrite);
    }
    BbFreeCheck::NoteGpuWrite(address, num_bytes, value, BbFreeCheck::GpuLabel);
    // A VRAM copy of these bytes gets them too, at its next binding (as for CPU-written labels).
    buffer_cache.NoteLateCommandWrite(address, &value, num_bytes);
    BbStats::gpu_labels.fetch_add(1, std::memory_order_relaxed);
    // In stream order for WAIT_REG_MEM, and the submission policy (HasUnsubmittedSignals).
    SignalAfterGpu([] {}, address, value);
    return true;
}

bool Rasterizer::PipeSignals() const {
    static const bool enabled = [] {
        const char* env = bb_env("BB_PIPE_SIGNALS");
        return !env || env[0] != '0';
    }();
    return enabled && UseDrawPipe() && OnStageA();
}

void Rasterizer::SubmitForSignals() {
    if (PipeSignals()) {
        // Decided (and submitted) on the recording thread after this submission's draws and
        // fences: here they may not even be recorded yet.
        static constexpr u8 none = 0;
        RunInOrder([](Rasterizer& self, const u8*) { self.SubmitForSignalsNow(); }, &none, 0,
                   BbToggle::PipelinedTasks, false);
        return;
    }
    SubmitForSignalsNow();
}

void Rasterizer::SubmitForSignalsNow() {
    PublishSignals(); // the end of a guest submission
    static const auto interval = std::chrono::microseconds([] {
        const char* env = bb_env("BB_HONEST_FLUSH_US");
        return env ? std::strtoll(env, nullptr, 10) : 2000ll;
    }());
    if (!HasUnsubmittedSignals()) {
        return;
    }
    // Batched (one submission per BB_HONEST_FLUSH_US) while the GPU still has work; at once when it
    // has finished everything submitted: holding the work back then only leaves it idle. Not at
    // once while draws are queued for this thread (BB_HONEST_IDLE_BACKLOG bytes, default 64 KiB):
    // it is then what the GPU waits for, and a submission costs it ~65 us (heavy scenes: 13 a
    // frame; 95 -> 100 FPS with half as many).
    static const bool idle_flush = [] {
        const char* env = bb_env("BB_HONEST_IDLE_FLUSH");
        return !env || env[0] != '0';
    }();
    static const u64 idle_backlog = [] {
        const char* env = bb_env("BB_HONEST_IDLE_BACKLOG");
        return env ? std::strtoull(env, nullptr, 10) : u64(64 * 1024);
    }();
    const bool gpu_idle =
        idle_flush &&
        (!draw_pipe || !DrawPipe::OnStageB() || draw_pipe->Backlog() < idle_backlog) &&
        scheduler.IsFree(scheduler.CurrentTick() - 1);
    // BB_HONEST_DRAIN_FLUSH (default on): also at once when the GPU is on the last submission
    // (everything before it finished): it would run dry before the next batch, and the frame's
    // work recorded meanwhile waited for the interval (~1 ms of GPU idle a frame).
    static const bool drain_flush = [] {
        const char* env = bb_env("BB_HONEST_DRAIN_FLUSH");
        return !env || env[0] != '0';
    }();
    const auto since_flush = std::chrono::nanoseconds(
        std::chrono::steady_clock::now().time_since_epoch().count() -
        last_flush_ns.load(std::memory_order_relaxed));
    const bool gpu_draining = drain_flush && !gpu_idle && scheduler.CurrentTick() >= 2 &&
                              scheduler.IsFree(scheduler.CurrentTick() - 2);
    if (!gpu_idle && !gpu_draining && since_flush < interval) {
        return;
    }
    if (gpu_draining) {
        BbStats::drain_flushes.fetch_add(1, std::memory_order_relaxed);
    }
    if (gpu_idle) {
        BbStats::idle_flushes.fetch_add(1, std::memory_order_relaxed);
    }
    Flush();
}

bool Rasterizer::PendingSignalValue(VAddr address, u64& value) {
    std::scoped_lock lk{gpu_signals_mutex};
    for (auto it = gpu_signals.rbegin(); it != gpu_signals.rend(); ++it) {
        if (it->label == address) {
            value = it->value;
            return true;
        }
    }
    return false;
}

void Rasterizer::GpuSignalLoop(std::stop_token token) {
    Common::SetCurrentThreadName("bb:GpuSignals");
    auto* semaphore = scheduler.GetWorkSemaphore();
    const vk::Semaphore handle = semaphore->Handle();
    while (!token.stop_requested()) {
        u64 tick = 0;
        {
            std::unique_lock lk{gpu_signals_mutex};
            if (!gpu_signals_cv.wait(lk, token, [this] { return !gpu_signals.empty(); })) {
                return;
            }
            tick = gpu_signals.front().tick;
        }
        // Bounded waits: the tick may not be submitted yet (the GPU command thread submits it
        // when it runs out of work), and a stop request must be seen.
        const vk::SemaphoreWaitInfo wait_info = {
            .semaphoreCount = 1,
            .pSemaphores = &handle,
            .pValues = &tick,
        };
        // bbport: a lost device ends the wait with a report (it spun here forever: the game froze
        // with its sound playing, Steam Deck); 10 s without progress prints what is stuck once.
        const auto wait_start = std::chrono::steady_clock::now();
        bool watched = false;
        auto asked_at = wait_start - std::chrono::seconds(1);
        while (!token.stop_requested()) {
            // Short waits first: a tick not handed to the GPU yet is asked for (below).
            const vk::Result result = instance.GetDevice().waitSemaphores(&wait_info, 2'000'000);
            if (result == vk::Result::eSuccess) {
                break;
            }
            // The fence waits for work still being recorded, and the GPU command thread may be
            // idle already (it submits for fences once when it runs out of work; a fence queued
            // after that waited forever: a black screen, the guest polling the label). Ask it to
            // submit, every 50 ms while this lasts.
            if (result == vk::Result::eTimeout && !scheduler.IsSubmitted(tick) &&
                std::chrono::steady_clock::now() - asked_at >= std::chrono::milliseconds(50)) {
                asked_at = std::chrono::steady_clock::now();
                liverpool->SendCommand([this] {
                    if (PipeSignals()) {
                        RequestSignalFlush();
                    } else {
                        Flush();
                    }
                });
            }
            if (result == vk::Result::eErrorDeviceLost) {
                Breadcrumbs::ReportDeviceLost("waiting for the GPU");
                ASSERT_MSG(false, "Device lost while waiting for the GPU");
            }
            if (!watched && std::chrono::steady_clock::now() - wait_start > std::chrono::seconds(10)) {
                watched = true;
                GpuWatchdog(tick);
            }
        }
        semaphore->Refresh();
        BbTimeline::Note(BbTimeline::GpuDone, semaphore->KnownGpuTick());
        while (true) {
            std::function<void()> signal;
            {
                std::scoped_lock lk{gpu_signals_mutex};
                if (gpu_signals.empty() || !semaphore->IsFree(gpu_signals.front().tick)) {
                    break;
                }
                signal = std::move(gpu_signals.front().signal);
                // Kept queued while it runs: PendingSignalValue still sees the fence.
            }
            signal();
            std::scoped_lock lk{gpu_signals_mutex};
            gpu_signals.pop_front();
        }
    }
}

void Rasterizer::GpuWatchdog(u64 tick) {
    auto* semaphore = scheduler.GetWorkSemaphore();
    semaphore->Refresh();
    const u64 done = semaphore->KnownGpuTick(), recording = scheduler.CurrentTick();
    if (tick < recording && !scheduler.IsSubmitted(tick)) {
        // Handed over but not given to the queue yet: its ordered submit task waits (copies,
        // earlier tasks). The GPU is idle; the port's threads wait for each other.
        std::printf("GPU watchdog: submission %llu recorded but not handed to the queue after 10 s "
                    "(the GPU finished %llu, recording %llu, handed out %llu): the port's threads "
                    "are waiting for each other\n",
                    (unsigned long long)tick, (unsigned long long)done,
                    (unsigned long long)recording, (unsigned long long)scheduler.SubmittedTick());
    } else if (tick < recording) {
        std::printf("GPU watchdog: submission %llu not finished after 10 s (the GPU finished %llu, "
                    "recording %llu): the GPU is stuck\n",
                    (unsigned long long)tick, (unsigned long long)done, (unsigned long long)recording);
        buffer_cache.ReportArenaBinds();
        Breadcrumbs::ReportStuck("submitted work not finished in 10 s");
    } else {
        std::printf("GPU watchdog: waited 10 s for submission %llu, not submitted yet (the GPU finished "
                    "%llu, recording %llu, draw pipe backlog %llu bytes): the port's threads are "
                    "waiting for each other\n",
                    (unsigned long long)tick, (unsigned long long)done, (unsigned long long)recording,
                    (unsigned long long)(draw_pipe ? draw_pipe->Backlog() : 0));
    }
    std::fflush(stdout);
}

} // namespace Vulkan

namespace Vulkan {
void Rasterizer::WaitPreviousFrame() {
    static u64 previous = 0;
    auto* semaphore = scheduler.GetWorkSemaphore();
    const u64 now = scheduler.CurrentTick();
    if (previous != 0 && previous < now) {
        semaphore->Wait(previous);
    }
    scheduler.Flush(); // this frame's commands go to the GPU, their tick is the next wait
    previous = now;
}
} // namespace Vulkan
