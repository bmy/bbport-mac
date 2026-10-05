// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <mutex>
#include <string_view>
#include <string>
#include <thread>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <dlfcn.h>
#include <functional>

#include "bbport_copy.h"
#include "video_core/renderer_vulkan/vk_gpu_profiler.h"
#include "bbport_toggles.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/thread.h"
#include "imgui/renderer/texture_manager.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "bbport_threads.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

/// bbport: a parallel recording worker (see CutPoint).
struct Scheduler::RecordWorker {
    u32 index = 0;
    std::unique_ptr<CommandPool> pool; ///< used by this worker, or by the producer while idle
    std::mutex mutex;
    std::condition_variable_any cv;      ///< work queued
    std::condition_variable_any idle_cv; ///< queue empty and not busy
    std::deque<std::unique_ptr<RecordChunk>> queue;
    std::atomic<size_t> queued{0}; ///< queue.size() for lock-free polling
    bool busy = false;
    bool sleeping = false;
    std::jthread thread; ///< last: stopped before the rest goes
};

Scheduler::Scheduler(const Instance& instance, bool threaded_recording)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore} {
    // bbport: BB_VK_RECORD_THREAD=0 records on the calling thread.
    const char* env = std::getenv("BB_VK_RECORD_THREAD");
    if (threaded_recording && !(env && env[0] == '0')) {
        record_chunk = AcquireChunk();
        recorder_thread = std::jthread(std::bind_front(&Scheduler::RecorderThread, this));
        CreateRecordWorkers();
    }
    LatchRecordingMode();
#if TRACY_GPU_ENABLED
    profiler_scope = reinterpret_cast<tracy::VkCtxScope*>(std::malloc(sizeof(tracy::VkCtxScope)));
#endif
    if (segmented) {
        StartSegments();
    } else {
        AllocateWorkerCommandBuffers();
    }
    priority_pending_ops_thread =
        std::jthread(std::bind_front(&Scheduler::PriorityPendingOpsThread, this));
}

Scheduler::~Scheduler() {
    if (recorder_thread.joinable()) {
        SyncRecording();
        recorder_thread.request_stop();
        recorder_cv.notify_all();
        recorder_thread.join();
    }
    for (auto& worker : workers) {
        worker->thread.request_stop();
        worker->cv.notify_all();
        worker->thread.join();
    }
#if TRACY_GPU_ENABLED
    std::free(profiler_scope);
#endif
}

namespace {
/// bbport (BB_FRAME_STATS): render passes per frame, passes reopened with the attachments the
/// previous pass just closed, and where passes end. Each pass costs a Metal encoder on
/// KosmicKrisp, so these show how much merging could save.
struct PassStats {
    std::mutex mutex;
    std::unordered_map<std::string, u64> end_sites;
    u64 passes = 0, reopened = 0, frame_base = 0;
    std::chrono::steady_clock::time_point last_print = std::chrono::steady_clock::now();

    void Print() {
        const auto now = std::chrono::steady_clock::now();
        if (now - last_print < std::chrono::seconds(5)) {
            return;
        }
        last_print = now;
        const u64 frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
        const double frames = frame > frame_base ? double(frame - frame_base) : 1.0;
        frame_base = frame;
        std::vector<std::pair<u64, std::string>> top;
        for (auto& [site, count] : end_sites) {
            top.emplace_back(count, site);
        }
        std::sort(top.rbegin(), top.rend());
        std::printf("Render passes: %.0f/frame, %.0f/frame reopened with identical attachments; "
                    "ended by:",
                    passes / frames, reopened / frames);
        for (size_t i = 0; i < top.size() && i < 8; ++i) {
            std::printf(" %s %.0f%s", top[i].second.c_str(), top[i].first / frames,
                        i + 1 < top.size() && i < 7 ? "," : "");
        }
        std::printf("\n");
        end_sites.clear();
        passes = reopened = 0;
    }
};
PassStats& GetPassStats() {
    static PassStats stats;
    return stats;
}
} // namespace

void Scheduler::BeginRendering(const RenderState& new_state) {
    if (is_rendering && render_state == new_state) {
        return;
    }
    EndRendering();
    if (BbStats::enabled) {
        auto& stats = GetPassStats();
        std::scoped_lock lock{stats.mutex};
        ++stats.passes;
        if (last_ended_valid && last_ended_state == new_state) {
            ++stats.reopened;
        }
        stats.Print();
    }
    is_rendering = true;
    render_state = new_state;
    if (segmented) {
        // A pass costs about as much recording as 8 draws (a Metal encoder on KosmicKrisp).
        segment_units += 8;
    }

    std::array<vk::RenderingAttachmentInfo, 8> color_attachments;
    for (u32 i = 0; i < render_state.num_color_attachments; ++i) {
        const auto& cb = render_state.color_attachments[i];
        color_attachments[i] = vk::RenderingAttachmentInfo{
            .imageView = cb.image_view,
            .imageLayout = cb.image_layout,
            .loadOp = cb.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = vk::ClearValue{.color = vk::ClearColorValue{.uint32 = cb.clear_value}},
        };
    }

    const auto& db = render_state.depth_stencil_attachment;
    const vk::RenderingAttachmentInfo depth_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue =
            vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{.depth = std::bit_cast<float>(
                                                                          db.clear_value[0])}},
    };
    const vk::RenderingAttachmentInfo stencil_attachment = {
        .imageView = db.image_view,
        .imageLayout = db.image_layout,
        .loadOp = db.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = vk::ClearValue{.depthStencil =
                                         vk::ClearDepthStencilValue{.stencil = db.clear_value[1]}},
    };

    const vk::RenderingInfo rendering_info = {
        .renderArea =
            {
                .offset = {0, 0},
                .extent = {render_state.width, render_state.height},
            },
        .layerCount = render_state.num_layers,
        .colorAttachmentCount = render_state.num_color_attachments,
        .pColorAttachments = color_attachments.data(),
        .pDepthAttachment = db.has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = db.has_stencil ? &stencil_attachment : nullptr,
    };

    if (!recorder_thread.joinable()) {
        current_cmdbuf.beginRendering(rendering_info);
        return;
    }
    // The attachment infos live on this stack frame: the recorded closure keeps copies.
    Record([info = rendering_info, color_attachments, depth_attachment,
            stencil_attachment](vk::CommandBuffer cmdbuf) mutable {
        info.pColorAttachments = color_attachments.data();
        if (info.pDepthAttachment) {
            info.pDepthAttachment = &depth_attachment;
        }
        if (info.pStencilAttachment) {
            info.pStencilAttachment = &stencil_attachment;
        }
        cmdbuf.beginRendering(info);
    });
}

void Scheduler::EndRendering(std::source_location where) {
    if (!is_rendering) {
        return;
    }
    if (BbStats::enabled) {
        last_ended_state = render_state;
        last_ended_valid = true;
        std::string_view file = where.file_name();
        if (const auto slash = file.find_last_of('/'); slash != std::string_view::npos) {
            file.remove_prefix(slash + 1);
        }
        auto& stats = GetPassStats();
        std::scoped_lock lock{stats.mutex};
        ++stats.end_sites[std::string(file) + ":" + std::to_string(where.line())];
    }
    is_rendering = false;
    Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
}

void Scheduler::TraceDirectRecording(void* caller) {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_RECORDER_TRACE");
        return env && env[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    static std::unordered_map<void*, u64> callers;
    static u64 calls;
    std::scoped_lock lk{mutex};
    ++callers[caller];
    if (++calls % 2000) {
        return;
    }
    std::vector<std::pair<u64, void*>> top;
    for (const auto& [address, count] : callers) {
        top.emplace_back(count, address);
    }
    std::ranges::sort(top, std::greater{});
    for (size_t i = 0; i < std::min<size_t>(top.size(), 8); ++i) {
        Dl_info info{};
        dladdr(top[i].second, &info);
        std::printf("Recorder sync caller: %llu x %s+0x%lx\n",
                    static_cast<unsigned long long>(top[i].first),
                    info.dli_fname ? info.dli_fname : "?",
                    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(top[i].second) -
                                               reinterpret_cast<uintptr_t>(info.dli_fbase)));
    }
    callers.clear();
}

std::unique_ptr<RecordChunk> Scheduler::AcquireChunk() {
    std::scoped_lock lk{recorder_mutex};
    if (free_chunks.empty()) {
        return std::make_unique<RecordChunk>();
    }
    auto chunk = std::move(free_chunks.back());
    free_chunks.pop_back();
    return chunk;
}

namespace {
/// BB_COPIES_OFF_RECORDER=0/1; default on macOS, where the recording thread is the bottleneck
/// (each Vulkan call is ~20 us through KosmicKrisp and Metal under Rosetta) and copies or
/// fences queued behind its backlog made the draw recording thread wait at EOS/WriteData.
bool CopiesOffRecorderWanted() {
    static const bool wanted = [] {
        if (const char* env = std::getenv("BB_COPIES_OFF_RECORDER"); env && env[0]) {
            return env[0] == '1';
        }
#ifdef __APPLE__
        return true;
#else
        return false;
#endif
    }();
    return wanted;
}
} // namespace

namespace {
constexpr u32 MaxRecordWorkers = 8;

/// Parallel recording statistics (BB_FRAME_STATS "Recording:" line).
struct RecordStats {
    std::array<std::atomic<u64>, MaxRecordWorkers> worker_busy_ns{};
    std::atomic<u64> segments{0}, cuts{0}, skipped_direct{0}, skipped_label{0};
    std::atomic<u64> begin_end_ns{0}, begin_ends{0}, submit_wait_ns{0};
    /// Mode of the threaded scheduler: 0 single recorder, else the number of workers.
    std::atomic<u32> mode_workers{0};
    std::atomic<u32> configured_workers{0};
};
RecordStats& GetRecordStats() {
    static RecordStats stats;
    return stats;
}

u64 NanosecondsSince(std::chrono::steady_clock::time_point start) {
    return static_cast<u64>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count());
}

/// BB_VK_RECORD_WORKERS=N (1-8): parallel recording workers. Default 2 on macOS with
/// KosmicKrisp, where each Vulkan call costs ~20 us of driver time and one recording thread
/// limits the frame rate; 1 elsewhere (Linux: the recorder is mostly idle; MoltenVK encodes
/// Metal at submission, so recording threads would not help).
u32 WantedRecordWorkers([[maybe_unused]] const Instance& instance) {
    if (const char* env = std::getenv("BB_VK_RECORD_WORKERS"); env && env[0]) {
        return static_cast<u32>(std::clamp(std::atoi(env), 1, static_cast<int>(MaxRecordWorkers)));
    }
#ifdef __APPLE__
    if (instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp) {
        return 2;
    }
#endif
    return 1;
}

/// BB_VK_SEGMENTS=0/1: record each submission as several command buffers (segments). Default:
/// on with more than one worker. BB_VK_SEGMENTS=1 with one worker is the cut-only test mode
/// (segments and their state re-established, no threads in parallel).
bool SegmentsWanted(u32 workers) {
    if (const char* env = std::getenv("BB_VK_SEGMENTS"); env && env[0]) {
        return env[0] == '1';
    }
    return workers > 1;
}
} // namespace

void Scheduler::CreateRecordWorkers() {
    const u32 count = WantedRecordWorkers(instance);
    const bool wanted = SegmentsWanted(count);
    if (!wanted) {
        if (count > 1) {
            std::printf("Vulkan recording: BB_VK_RECORD_WORKERS=%u ignored with BB_VK_SEGMENTS=0\n",
                        count);
        }
        return;
    }
    cut_units = 250;
    if (const char* env = std::getenv("BB_VK_SEGMENT_UNITS"); env && env[0]) {
        cut_units = static_cast<u32>(std::clamp(std::atoi(env), 8, 1000000));
    }
    for (u32 i = 0; i < count; ++i) {
        auto worker = std::make_unique<RecordWorker>();
        worker->index = i;
        worker->pool = std::make_unique<CommandPool>(instance, &work_semaphore);
        workers.push_back(std::move(worker));
    }
    for (auto& worker : workers) {
        RecordWorker& w = *worker;
        w.thread = std::jthread([this, &w](std::stop_token stoken) { RecordWorkerThread(stoken, w); });
    }
    GetRecordStats().configured_workers.store(count, std::memory_order_relaxed);
    if (!CopiesOffRecorderWanted()) {
        std::printf("Vulkan recording: segments need BB_COPIES_OFF_RECORDER=1; one command "
                    "buffer per submission\n");
    }
    std::printf("Vulkan recording: %u worker%s, a new command buffer every ~%u units "
                "(BB_VK_RECORD_WORKERS, BB_VK_SEGMENTS, BB_VK_SEGMENT_UNITS)\n",
                count, count == 1 ? " (cut-only test mode)" : "s", cut_units);
}

void Scheduler::LatchRecordingMode() {
    // Called with no stream copy or stream signal pending (constructor; SubmitExecution after
    // SyncRecording and WaitHostCopies), so a switch cannot reorder copies or fences: every
    // stream signal has already been handed to BbCopy::AfterCopies, and signals handed over
    // from now on are queued behind it.
    copies_off_recorder = recorder_thread.joinable() && CopiesOffRecorderWanted() &&
                          !BbToggle::Disabled(BbToggle::RecorderHostCopies);
    // Segments need every recorded closure to be a pure Vulkan call: copies and fence signals
    // recorded in the stream would run on whichever worker gets their segment, out of order.
    segmented = copies_off_recorder && !workers.empty() &&
                !BbToggle::Disabled(BbToggle::ParallelRecording);
#if TRACY_GPU_ENABLED
    segmented = false; // the Tracy GPU scope lives in current_cmdbuf
#endif
    if (recorder_thread.joinable()) {
        GetRecordStats().mode_workers.store(segmented ? static_cast<u32>(workers.size()) : 0,
                                            std::memory_order_relaxed);
    }
}

void Scheduler::NewSegment() {
    cur_segment = &segments.emplace_back();
    cur_segment->worker = next_worker;
    next_worker = (next_worker + 1) % static_cast<u32>(workers.size());
    segment_units = 0;
    if (BbStats::enabled) {
        GetRecordStats().segments.fetch_add(1, std::memory_order_relaxed);
    }
}

void Scheduler::StartSegments() {
    // All workers are idle (SyncRecording) and every buffer of the last submission was
    // submitted: the old segments go.
    segments.clear();
    cur_segment = nullptr;
    NewSegment();
    current_cmdbuf = vk::CommandBuffer{}; // set by EnterDirectMode()
    dynamic_state.Invalidate();
}

void Scheduler::BeginSegment(RecordSegment& segment, CommandPool& pool) {
    // The pool stamps the buffer with the current tick: reused once the GPU is past the
    // submission (a tick taken during SubmitExecution is one later, which only delays reuse).
    segment.cmdbuf = pool.Commit();
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    Check(segment.cmdbuf.begin(begin_info));
    segment.begun = true;
}

void Scheduler::WaitWorkerIdle(RecordWorker& worker) {
    std::unique_lock lk{worker.mutex};
    worker.idle_cv.wait(lk, [&worker] { return worker.queue.empty() && !worker.busy; });
}

void Scheduler::HandOff(bool end_segment) {
    RecordSegment* const segment = cur_segment;
    const bool pending = !full_chunks.empty() || !record_chunk->Empty();
    if (!pending && (!end_segment || (!segment->sent && !segment->begun))) {
        return; // nothing to record, or nothing recorded that needs an end
    }
    RecordWorker& worker = *workers[segment->worker];
    bool wake;
    {
        std::scoped_lock lk{worker.mutex};
        for (auto& chunk : full_chunks) {
            chunk->segment = segment;
            worker.queue.push_back(std::move(chunk));
        }
        if (!record_chunk->Empty() || end_segment) {
            record_chunk->segment = segment;
            worker.queue.push_back(std::move(record_chunk));
        }
        worker.queue.back()->ends_segment = end_segment;
        wake = worker.sleeping;
        worker.queued.store(worker.queue.size(), std::memory_order_release);
    }
    full_chunks.clear();
    segment->sent = true;
    if (wake) {
        worker.cv.notify_one();
    }
    if (!record_chunk) {
        record_chunk = AcquireChunk();
    }
}

void Scheduler::CutSegment() {
    HandOff(true);
    if (!cur_segment->sent && !cur_segment->begun) {
        return; // nothing recorded in it yet: keep it
    }
    NewSegment();
    // Dynamic state is the only state the rasterizer carries between draws instead of
    // recording it per draw: the new buffer starts without it.
    dynamic_state.Invalidate();
}

void Scheduler::CutPoint(bool pass_ends, std::source_location where) {
    if (!segmented) {
        return;
    }
    ++segment_units;
    if (segment_units < cut_units) {
        return;
    }
    if (direct_mode) {
        if (BbStats::enabled) {
            GetRecordStats().skipped_direct.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    if (label_depth != 0) {
        if (BbStats::enabled) {
            GetRecordStats().skipped_label.fetch_add(1, std::memory_order_relaxed);
        }
        return;
    }
    if (is_rendering) {
        if (!pass_ends) {
            return; // never inside a render pass: wait for a draw that ends it anyway
        }
        // Where BeginRendering would have ended it: the rest of this draw only sets state.
        EndRendering(where);
    }
    CutSegment();
    if (BbStats::enabled) {
        GetRecordStats().cuts.fetch_add(1, std::memory_order_relaxed);
    }
}

void Scheduler::RecordWorkerThread(std::stop_token stoken, RecordWorker& worker) {
    Common::SetCurrentThreadName(("bb:VkRec" + std::to_string(worker.index)).c_str());
    auto& stats = GetRecordStats();
    while (true) {
        // As RecorderThread: spin briefly before sleeping; later workers spin less (with few
        // hardware threads the spin takes time from guest threads).
        static const auto spin_time = std::chrono::microseconds(
            BbThreads::Available() >= 12 ? 200 : 20);
        const auto my_spin = worker.index == 0 ? spin_time : spin_time / 4;
        const auto spin_until = std::chrono::steady_clock::now() + my_spin;
        for (u32 spins = 1; worker.queued.load(std::memory_order_acquire) == 0; ++spins) {
            __builtin_ia32_pause();
            if (!(spins & 255) &&
                (stoken.stop_requested() || std::chrono::steady_clock::now() >= spin_until)) {
                break;
            }
        }
        std::unique_ptr<RecordChunk> chunk;
        {
            std::unique_lock lk{worker.mutex};
            worker.sleeping = true;
            worker.cv.wait(lk, stoken, [&worker] { return !worker.queue.empty(); });
            worker.sleeping = false;
            if (worker.queue.empty()) {
                return; // stop requested
            }
            chunk = std::move(worker.queue.front());
            worker.queue.pop_front();
            worker.queued.store(worker.queue.size(), std::memory_order_release);
            worker.busy = true;
        }
        RecordSegment& segment = *chunk->segment;
        const auto start = BbStats::enabled ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
        if (!segment.begun) {
            const auto begin_start = std::chrono::steady_clock::now();
            BeginSegment(segment, *worker.pool);
            if (BbStats::enabled) {
                stats.begin_end_ns.fetch_add(NanosecondsSince(begin_start),
                                             std::memory_order_relaxed);
            }
        }
        chunk->Execute(segment.cmdbuf);
        if (chunk->ends_segment) {
            const auto end_start = std::chrono::steady_clock::now();
            Check(segment.cmdbuf.end());
            segment.ended = true;
            if (BbStats::enabled) {
                stats.begin_end_ns.fetch_add(NanosecondsSince(end_start),
                                             std::memory_order_relaxed);
                stats.begin_ends.fetch_add(1, std::memory_order_relaxed);
            }
        }
        if (BbStats::enabled) {
            stats.worker_busy_ns[worker.index].fetch_add(NanosecondsSince(start),
                                                         std::memory_order_relaxed);
        }
        chunk->segment = nullptr;
        chunk->ends_segment = false;
        {
            std::scoped_lock lk{recorder_mutex};
            free_chunks.push_back(std::move(chunk));
        }
        {
            std::scoped_lock lk{worker.mutex};
            worker.busy = false;
            if (worker.queue.empty()) {
                worker.idle_cv.notify_all();
            }
        }
    }
}

std::vector<BbCopy::Item> Scheduler::TakeHostCopies() {
    std::scoped_lock lk{host_batch_mutex};
    std::vector<BbCopy::Item> items;
    items.swap(host_batch);
    host_batch_bytes = 0;
    return items;
}

void Scheduler::QueueHostCopy(const BbCopy::Item& item) {
    if (!BbCopy::Enabled()) {
        item.run(item); // no copy threads: copied now, as BbCopy::QueueCopy does
        return;
    }
    if (BbStats::enabled) {
        BbStats::host_batch_copies.fetch_add(1, std::memory_order_relaxed);
    }
    {
        std::scoped_lock lk{host_batch_mutex};
        host_batch.push_back(item);
        host_batch_bytes += item.size;
        // One wakeup per ~512 KiB or 256 copies, as BbCopy::QueueCopy.
        if (host_batch_bytes < 512 * 1024 && host_batch.size() < 256) {
            return;
        }
    }
    FlushHostCopies();
}

void Scheduler::FlushHostCopies() {
    auto items = TakeHostCopies();
    if (items.empty()) {
        return;
    }
    if (BbStats::enabled) {
        BbStats::host_batches.fetch_add(1, std::memory_order_relaxed);
    }
    auto shared = std::make_shared<std::vector<BbCopy::Item>>(std::move(items));
    BbCopy::Async([shared] {
        for (const auto& item : *shared) {
            item.run(item);
        }
    });
}

void Scheduler::SignalAfterHostCopies(std::function<void()> signal) {
    if (!IsRecordingDeferred()) {
        WaitHostCopies();
        signal();
        return;
    }
    if (copies_off_recorder) {
        // Invariants (the guest frees the memory a fence guards once it sees the value, and
        // frees objects holding fence labels once the GPU is idle):
        // 1. A fence is written only after every guest copy issued before it has finished:
        //    the batch is started (Async) before AfterCopies, which waits for every task
        //    started so far.
        // 2. Fences are written in issue order: AfterCopies runs callbacks in registration
        //    order, and stream signals from before a mode switch were registered earlier
        //    (LatchRecordingMode).
        // 3. WaitDeferredSignals (GPU idle, WriteData, EOS, flip) waits for these signals:
        //    deferred_signals_issued/done count them like the stream ones.
        // Nothing waits for the recording thread: the guest observes fences, not Vulkan
        // recording, and recorded commands read only host buffers filled by those copies.
        FlushHostCopies();
        BbCopy::FlushBatch();
        deferred_signals_issued.fetch_add(1, std::memory_order_relaxed);
        if (BbStats::enabled) {
            BbStats::copy_thread_fences.fetch_add(1, std::memory_order_relaxed);
        }
        BbCopy::AfterCopies([signal = std::move(signal), done = deferred_signals_done]() mutable {
            signal();
            done->fetch_add(1, std::memory_order_release);
        });
        return;
    }
    BbCopy::FlushBatch();
    deferred_signals_issued.fetch_add(1, std::memory_order_relaxed);
    Record([signal = std::move(signal), done = deferred_signals_done](vk::CommandBuffer) mutable {
        BbCopy::AfterCopies([signal = std::move(signal), done = std::move(done)] {
            signal();
            done->fetch_add(1, std::memory_order_release);
        });
    });
    KickRecording(true);
}

void Scheduler::WaitDeferredSignals() {
    const u64 issued = deferred_signals_issued.load(std::memory_order_relaxed);
    if (deferred_signals_done->load(std::memory_order_acquire) >= issued ||
        BbToggle::Disabled(BbToggle::OrderedGuestWrites)) {
        return;
    }
    BbStats::WaitTimer timer{BbStats::host_copies_wait_ns};
    KickRecording(true);
    while (deferred_signals_done->load(std::memory_order_acquire) < issued) {
        // Helps the copy threads the signals wait for.
        BbCopy::WaitAsync();
        std::this_thread::yield();
    }
}

void Scheduler::WaitHostCopies() {
    if (host_copies_done.load(std::memory_order_acquire) < host_copies_issued) {
        BbStats::WaitTimer timer{BbStats::host_copies_wait_ns};
        BbStats::host_copy_waits.fetch_add(1, std::memory_order_relaxed);
        KickRecording(true);
        while (host_copies_done.load(std::memory_order_acquire) < host_copies_issued) {
            std::this_thread::yield();
        }
    }
    BbStats::WaitTimer timer{BbStats::copy_threads_wait_ns};
    // Copies batched off the recorder (QueueHostCopy) run here: handing them over only to
    // wait for them costs a wakeup.
    for (const auto& item : TakeHostCopies()) {
        item.run(item);
    }
    BbCopy::WaitAsync();
}

vk::CommandBuffer Scheduler::EnterDirectMode() {
    if (!segmented) {
        SyncRecording();
        direct_mode = true;
        return current_cmdbuf;
    }
    // Only the current segment's worker has to be done: the caller records into that segment's
    // buffer, after everything recorded into it so far. The other workers go on with earlier
    // segments; no cut happens until direct mode ends (KickRecording).
    KickRecording(true);
    BbStats::WaitTimer timer{BbStats::sync_recording_ns};
    RecordWorker& worker = *workers[cur_segment->worker];
    WaitWorkerIdle(worker);
    if (!cur_segment->begun) {
        // The worker is idle and gets nothing until direct mode ends: its pool is free.
        BeginSegment(*cur_segment, *worker.pool);
    }
    current_cmdbuf = cur_segment->cmdbuf;
    direct_mode = true;
    return current_cmdbuf;
}

void Scheduler::KickRecording(bool force) {
    if (!recorder_thread.joinable()) {
        return;
    }
    // Callers kick where nobody holds the raw command buffer: deferral resumes.
    direct_mode = false;
    // Batches of tens of KiB keep the queue handoff cheap relative to the work it carries.
    if (!force && full_chunks.empty() && record_chunk->Size() < 32 * 1024) {
        return;
    }
    if (segmented) {
        HandOff(false);
        return;
    }
    if (full_chunks.empty() && record_chunk->Empty()) {
        return;
    }
    bool wake;
    {
        std::scoped_lock lk{recorder_mutex};
        for (auto& chunk : full_chunks) {
            recorder_queue.push_back(std::move(chunk));
        }
        if (!record_chunk->Empty()) {
            recorder_queue.push_back(std::move(record_chunk));
        }
        wake = recorder_sleeping;
        queued_chunks.store(recorder_queue.size(), std::memory_order_release);
    }
    full_chunks.clear();
    // A busy recorder picks the new chunks up by itself: waking it is a syscall per draw.
    if (wake) {
        recorder_cv.notify_one();
    }
    if (!record_chunk) {
        record_chunk = AcquireChunk();
    }
}

void Scheduler::SyncRecording() {
    if (!recorder_thread.joinable()) {
        return;
    }
    KickRecording(true);
    BbStats::WaitTimer timer{BbStats::sync_recording_ns};
    {
        std::unique_lock lk{recorder_mutex};
        recorder_idle_cv.wait(lk, [this] { return recorder_queue.empty() && !recorder_busy; });
    }
    for (auto& worker : workers) {
        WaitWorkerIdle(*worker);
    }
}

void Scheduler::RecorderThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("bb:VkRecorder");
    while (true) {
        // Spin briefly before sleeping: the next chunk usually follows within microseconds,
        // and a sleeping recorder costs the GPU thread a wake-up syscall per kick. With few
        // hardware threads (Steam Deck: 8) the spin would take time from guest threads.
        static const auto spin_time = std::chrono::microseconds(
            BbThreads::Available() >= 12 ? 200 : 20);
        const auto spin_until = std::chrono::steady_clock::now() + spin_time;
        for (u32 spins = 1; queued_chunks.load(std::memory_order_acquire) == 0; ++spins) {
            __builtin_ia32_pause();
            // The clock is read every 256 pauses, not per iteration.
            if (!(spins & 255) &&
                (stoken.stop_requested() || std::chrono::steady_clock::now() >= spin_until)) {
                break;
            }
        }
        std::unique_ptr<RecordChunk> chunk;
        {
            std::unique_lock lk{recorder_mutex};
            recorder_sleeping = true;
            recorder_cv.wait(lk, stoken, [this] { return !recorder_queue.empty(); });
            recorder_sleeping = false;
            if (recorder_queue.empty()) {
                return; // stop requested
            }
            chunk = std::move(recorder_queue.front());
            recorder_queue.pop_front();
            queued_chunks.store(recorder_queue.size(), std::memory_order_release);
            recorder_busy = true;
        }
        // current_cmdbuf only changes after SyncRecording(), which waits for this thread.
        if (BbStats::enabled) {
            const auto start = std::chrono::steady_clock::now();
            chunk->Execute(current_cmdbuf);
            BbStats::recorder_busy_ns.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count(),
                std::memory_order_relaxed);
        } else {
            chunk->Execute(current_cmdbuf);
        }
        {
            std::scoped_lock lk{recorder_mutex};
            free_chunks.push_back(std::move(chunk));
            recorder_busy = false;
            if (recorder_queue.empty()) {
                recorder_idle_cv.notify_all();
            }
        }
    }
}

void Scheduler::Flush(SubmitInfo& info) {
    // When flushing, we only send data to the driver; no waiting is necessary.
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    // When finishing, we need to wait for the submission to have executed on the device.
    const u64 presubmit_tick = CurrentTick();
    SubmitInfo info{};
    SubmitExecution(info);
    Wait(presubmit_tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick >= work_semaphore.CurrentTick()) {
        // Make sure we are not waiting for the current tick without signalling
        SubmitInfo info{};
        Flush(info);
    }
    BbStats::WaitTimer timer{BbStats::tick_wait_ns};
    work_semaphore.Wait(tick);
}

void Scheduler::PopPendingOperations() {
    if (num_pending_ops.load(std::memory_order_acquire) == 0) {
        return; // every draw comes here
    }
    std::unique_lock lk(pending_ops_mutex);
    // bbport: this runs on every draw and dispatch. Querying the timeline semaphore is an
    // ioctl, so it is skipped when nothing waits and done once per 32 calls (~0.3 ms; reading
    // the clock per draw instead was itself a hot spot).
    if (pending_ops.empty()) {
        return;
    }
    if (!work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        if ((++pending_polls & 31) != 0 && !BbToggle::Disabled(BbToggle::PendingPollLimit)) {
            return;
        }
        work_semaphore.Refresh();
    }
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        pending_ops.front().callback();
        pending_ops.pop();
        num_pending_ops.fetch_sub(1, std::memory_order_release);
    }
}

void Scheduler::AllocateWorkerCommandBuffers() {
    const vk::CommandBufferBeginInfo begin_info = {
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };

    current_cmdbuf = command_pool.Commit();
    Check(current_cmdbuf.begin(begin_info));

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        static const auto scope_loc =
            GPU_SCOPE_LOCATION("Guest Frame", MarkersPalette::GpuMarkerColor);
        new (profiler_scope) tracy::VkCtxScope{profiler_ctx, &scope_loc, current_cmdbuf, true};
    }
#endif
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    std::scoped_lock lk{submit_mutex};
    const u64 signal_value = work_semaphore.NextTick();

#if TRACY_GPU_ENABLED
    auto* profiler_ctx = instance.GetProfilerContext();
    if (profiler_ctx) {
        profiler_scope->~VkCtxScope();
        TracyVkCollect(profiler_ctx, current_cmdbuf);
    }
#endif

    if (on_submit) {
        on_submit(info);
    }

    EndRendering();
    if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(this)) {
        // Until the next submission's first timestamp: mostly the GPU waiting for it.
        profiler->Mark(0x5B317ull, [] { return std::string{"(between submissions: GPU idle)"}; });
    }
    if (segmented) {
        HandOff(true); // the last segment's buffer ends on its worker
    }
    {
        const auto wait_start = std::chrono::steady_clock::now();
        SyncRecording();
        if (segmented && BbStats::enabled) {
            GetRecordStats().submit_wait_ns.fetch_add(NanosecondsSince(wait_start),
                                                      std::memory_order_relaxed);
        }
    }
    // Guest memory copies into staging read by this submission (copy threads).
    WaitHostCopies();
    submit_cmdbufs.clear();
    if (segmented) {
        // In stream order. Barriers and layout transitions keep working across the buffers:
        // their scopes follow submission order, which spans the buffers of one submit.
        for (const auto& segment : segments) {
            if (segment.begun) {
                ASSERT_MSG(segment.ended, "recording segment submitted before it ended");
                submit_cmdbufs.push_back(segment.cmdbuf);
            }
        }
    } else {
        Check(current_cmdbuf.end());
        submit_cmdbufs.push_back(current_cmdbuf);
    }

    const vk::Semaphore timeline = work_semaphore.Handle();
    info.AddSignal(timeline, signal_value);

    static constexpr std::array<vk::PipelineStageFlags, 2> wait_stage_masks = {
        vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
    };

    const vk::TimelineSemaphoreSubmitInfo timeline_si = {
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };

    const vk::SubmitInfo submit_info = {
        .pNext = &timeline_si,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stage_masks.data(),
        .commandBufferCount = static_cast<u32>(submit_cmdbufs.size()),
        .pCommandBuffers = submit_cmdbufs.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };

    ImGui::Core::TextureManager::Submit();
    auto submit_result = instance.GetGraphicsQueue().submit(submit_info, info.fence);
    ASSERT_MSG(submit_result != vk::Result::eErrorDeviceLost, "Device lost during submit");

    work_semaphore.Refresh();
    // Every stream copy and stream signal of this submission has run (SyncRecording,
    // WaitHostCopies): the mode may change here.
    LatchRecordingMode();
    if (segmented) {
        StartSegments();
    } else {
        AllocateWorkerCommandBuffers();
    }

    // Apply pending operations
    PopPendingOperations();
}

void Scheduler::PriorityPendingOpsThread(std::stop_token stoken) {
    Common::SetCurrentThreadName("shadPS4:GpuSchedPriorityPendingOpsRunner");

    while (!stoken.stop_requested()) {
        PendingOp op;
        {
            std::unique_lock lk(priority_pending_ops_mutex);
            priority_pending_ops_cv.wait(lk, stoken,
                                         [this] { return !priority_pending_ops.empty(); });
            if (stoken.stop_requested()) {
                break;
            }

            op = std::move(priority_pending_ops.front());
            priority_pending_ops.pop();
        }

        work_semaphore.Wait(op.gpu_tick);
        if (stoken.stop_requested()) {
            break;
        }

        op.callback();
    }
}

void Scheduler::PrintRecordingStats(double seconds, double frames) {
    const auto take = [](std::atomic<u64>& counter) {
        return counter.exchange(0, std::memory_order_relaxed);
    };
    auto& stats = GetRecordStats();
    const double per_frame = frames > 0 ? 1.0 / frames : 0.0;
    const double percent = seconds > 0 ? 100.0 / (seconds * 1e9) : 0.0;
    const u32 mode = stats.mode_workers.load(std::memory_order_relaxed);
    const u32 configured = stats.configured_workers.load(std::memory_order_relaxed);
    char workers_busy[128] = "";
    size_t at = 0;
    for (u32 i = 0; i < configured && i < MaxRecordWorkers; ++i) {
        const int n = std::snprintf(workers_busy + at, sizeof(workers_busy) - at, " %.1f%%",
                                    take(stats.worker_busy_ns[i]) * percent);
        if (n < 0 || static_cast<size_t>(n) >= sizeof(workers_busy) - at) {
            break;
        }
        at += static_cast<size_t>(n);
    }
    const u64 begin_ends = take(stats.begin_ends);
    const u64 begin_end_ns = take(stats.begin_end_ns);
    std::printf("Recording: %s; host copies %s the recorder, %.0f/frame in %.1f batches, "
                "%.1f fences/frame via copy threads; recorder busy %.1f%%, workers busy%s; "
                "%.1f segments/frame (%.1f cuts, skipped: %.1f direct mode, %.1f debug label), "
                "begin+end %.0f us/segment, submit wait %.2f ms/frame\n",
                mode ? (mode == 1 ? "segments on 1 worker (cut-only)" : "segments on workers")
                     : "one recorder, one command buffer per submission",
                CopiesOffRecorderWanted() && !BbToggle::Disabled(BbToggle::RecorderHostCopies)
                    ? "off"
                    : "on",
                take(BbStats::host_batch_copies) * per_frame,
                take(BbStats::host_batches) * per_frame,
                take(BbStats::copy_thread_fences) * per_frame,
                take(BbStats::recorder_busy_ns) * percent, configured ? workers_busy : " -",
                take(stats.segments) * per_frame, take(stats.cuts) * per_frame,
                take(stats.skipped_direct) * per_frame, take(stats.skipped_label) * per_frame,
                begin_ends ? begin_end_ns / 1e3 / double(begin_ends) : 0.0,
                take(stats.submit_wait_ns) * per_frame / 1e6);
}

void DynamicState::Commit(const Instance& instance, const vk::CommandBuffer& cmdbuf) {
    // A null command buffer only updates the dirty flags, exactly as recording would.
    CommitWith(instance.IsDepthBoundsSupported(), instance.IsDynamicColorWriteMaskSupported(),
               instance.IsAttachmentFeedbackLoopLayoutSupported(), [&](auto&& command) {
                   if (cmdbuf) {
                       command(cmdbuf);
                   }
               });
}

} // namespace Vulkan
