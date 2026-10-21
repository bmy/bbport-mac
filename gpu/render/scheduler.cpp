// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_timeline.h"
#include "render/scheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <tuple>
#include <unordered_map>
#include <array>
#include <mutex>

#include <pthread.h>

#include <dlfcn.h>
#include <execinfo.h>
#include <unistd.h>

#include "bbport_copy.h"
#include "bbport_cpu.h"
#include "bbport_threads.h"
#include "bbport_toggles.h"
#include "bbport_wait_trace.h"
#include "common/thread.h"
#include "render/vk_gpu_profiler.h"
#include "vulkan/instance.h"

namespace Vulkan {

std::mutex Scheduler::submit_mutex;

namespace {
// ProducerScope state of this thread: the producer function it is in (any scheduler).
thread_local const char* producer_scope_where = nullptr;
std::atomic<u32> producer_reports{0};

/// Names a thread of this process (its comm).
std::string ThreadName(u32 tid) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/self/task/%u/comm", tid);
    std::string name = "?";
    if (FILE* file = std::fopen(path, "r")) {
        char buffer[32]{};
        if (std::fgets(buffer, sizeof(buffer), file)) {
            name = buffer;
            if (!name.empty() && name.back() == '\n') {
                name.pop_back();
            }
        }
        std::fclose(file);
    }
    return name;
}

void ReportProducer(const char* what, u32 other, const char* other_where, const char* where) {
    if (producer_reports.fetch_add(1, std::memory_order_relaxed) >= 8) {
        return;
    }
    const u32 self = u32(gettid());
    std::fprintf(stderr,
                 "Scheduler: %s: thread %u (%s) entering %s while thread %u (%s) is in %s\n",
                 what, self, ThreadName(self).c_str(), where, other, ThreadName(other).c_str(),
                 other_where ? other_where : "?");
    void* frames[24];
    const int depth = backtrace(frames, 24);
    backtrace_symbols_fd(frames, depth, 2);
}
} // namespace

void Scheduler::ProducerScope::Enter(const char* where) noexcept {
    previous_where = producer_scope_where;
    static thread_local const u32 tid = u32(gettid());
    producer_scope_where = where;
    u32 expected = 0;
    if (scheduler.producer_tid.compare_exchange_strong(expected, tid,
                                                       std::memory_order_acq_rel)) {
        outer = true;
        scheduler.producer_where.store(where, std::memory_order_relaxed);
        return;
    }
    if (expected != tid) {
        ReportProducer("CONCURRENT recording", expected,
                       scheduler.producer_where.load(std::memory_order_relaxed), where);
    }
}

void Scheduler::ProducerScope::Leave() noexcept {
    producer_scope_where = previous_where;
    if (outer) {
        scheduler.producer_tid.store(0, std::memory_order_release);
    }
}

namespace {

/// Vulkan recording threads: BB_VK_RECORD_THREADS, else 2 from 8 hardware threads and 3
/// from 12. One thread records ~75% of a core at 200 FPS (4 cores / 8 threads); the segments
/// let the stream use more than one.
u32 RecordingThreadCount() {
    if (const char* env = std::getenv("BB_VK_RECORD_THREADS")) {
        return std::clamp(std::atoi(env), 1, 8);
    }
    const unsigned available = BbThreads::Available();
    return available >= 12 ? 3 : available >= 8 ? 2 : 1;
}

vk::CommandBuffer BeginCommandBuffer(CommandPool& pool) {
    const vk::CommandBuffer cmdbuf = pool.Commit();
    Check(cmdbuf.begin({.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit}));
    if (DynCheck::Enabled()) {
        DynCheck::Begin(cmdbuf);
    }
    return cmdbuf;
}

/// The queue's recent submissions (every scheduler), for the report of a GPU wait over a second.
struct QueueRecord {
    const Scheduler* scheduler = nullptr;
    const Semaphore* semaphore = nullptr; ///< its work timeline
    u64 tick = 0;                         ///< the tick it signals there
    u32 num_waits = 0;
    std::array<vk::Semaphore, SubmitInfo::MaxSemaphores> waits{};
    std::array<u64, SubmitInfo::MaxSemaphores> values{};
    std::array<const char*, SubmitInfo::MaxSemaphores> labels{};
    std::array<bool, SubmitInfo::MaxSemaphores> timeline{};
    u32 cmdbufs = 0;
    u32 queue_ops = 0; ///< before_submit operations (memory binds) before it
    u64 enter_ns = 0;  ///< came to QueueSubmit
    u64 ops_ns = 0;    ///< its queue operations were done
    u64 exit_ns = 0;   ///< vkQueueSubmit returned (0: not yet)
};
constexpr std::size_t QueueRecords = 256;
std::mutex queue_log_mutex;
std::array<QueueRecord, QueueRecords> queue_log;
u64 queue_log_count = 0;

u64 SteadyNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

} // namespace

void Scheduler::ReportSlowWait(const Semaphore& semaphore, u64 tick, u64 waited_ms, bool done) {
    char thread[32] = "?";
    pthread_getname_np(pthread_self(), thread, sizeof(thread));
    if (done) {
        std::printf("GPU wait: %s waited %llu ms in all for tick %llu\n", thread,
                    (unsigned long long)waited_ms, (unsigned long long)tick);
        std::fflush(stdout);
        return;
    }
    // Gathered under the lock (records of a scheduler being destroyed are cleared under it),
    // printed after it: QueueSubmit takes the lock too.
    std::string out;
    const auto add = [&out](const char* format, auto... args) {
        if constexpr (sizeof...(args) == 0) {
            out += format;
        } else {
            char line[512];
            std::snprintf(line, sizeof(line), format, args...);
            out += line;
        }
    };
    std::unique_lock lock{queue_log_mutex};
    const char* owner = "another timeline";
    const Scheduler* scheduler = nullptr;
    for (std::size_t i = 0; i < QueueRecords && i < queue_log_count; ++i) {
        if (queue_log[i].semaphore == &semaphore && queue_log[i].scheduler) {
            scheduler = queue_log[i].scheduler;
            owner = scheduler->name;
            break;
        }
    }
    const_cast<Semaphore&>(semaphore).Refresh();
    add("GPU wait: %s has waited %llu ms for tick %llu of %s; the GPU finished %llu",
                thread, (unsigned long long)waited_ms, (unsigned long long)tick, owner,
                (unsigned long long)semaphore.KnownGpuTick());
    if (scheduler) {
        const u64 queued = scheduler->SubmittedTick();
        add(", the queue has up to %llu, recording %llu", (unsigned long long)queued,
                    (unsigned long long)scheduler->CurrentTick());
        if (queued < tick) {
            add(" (tick %llu has not reached the queue: still recorded or submitted "
                        "here, not the GPU)",
                        (unsigned long long)tick);
        }
    }
    add("\n  the queue's unfinished submissions, oldest first:\n");
    const u64 now = SteadyNs();
    const u64 first = queue_log_count > QueueRecords ? queue_log_count - QueueRecords : 0;
    u32 shown = 0;
    for (u64 n = first; n < queue_log_count && shown < 24; ++n) {
        QueueRecord& r = queue_log[n % QueueRecords];
        if (!r.scheduler) {
            continue; // its scheduler is gone
        }
        const_cast<Semaphore*>(r.semaphore)->Refresh();
        if (r.exit_ns && r.semaphore->KnownGpuTick() >= r.tick) {
            continue;
        }
        ++shown;
        add("    %s tick %llu: %u command buffer%s", r.scheduler->name,
                    (unsigned long long)r.tick, r.cmdbufs, r.cmdbufs == 1 ? "" : "s");
        if (r.queue_ops) {
            add(", %u queue operation%s first (%.1f ms)", r.queue_ops,
                        r.queue_ops == 1 ? "" : "s", ((r.ops_ns ? r.ops_ns : now) - r.enter_ns) / 1e6);
        }
        if (r.exit_ns) {
            add("; submitted %.0f ms ago (vkQueueSubmit %.1f ms)", (now - r.exit_ns) / 1e6,
                        (r.exit_ns - r.ops_ns) / 1e6);
        } else {
            add("; IN SUBMISSION for %.0f ms (%s)", (now - r.enter_ns) / 1e6,
                        r.ops_ns ? "vkQueueSubmit" : "its queue operations");
        }
        for (u32 i = 0; i < r.num_waits; ++i) {
            add("%s %s", i == 0 ? "; waits for" : ",", r.labels[i] ? r.labels[i] : "?");
            if (r.timeline[i]) {
                const auto [result, value] =
                    r.scheduler->instance.GetDevice().getSemaphoreCounterValue(r.waits[i]);
                add(" %llu (at %llu%s)", (unsigned long long)r.values[i],
                            result == vk::Result::eSuccess ? (unsigned long long)value : 0ull,
                            result == vk::Result::eSuccess && value >= r.values[i] ? ": reached"
                                                                                   : ": NOT YET");
            }
        }
        add("\n");
    }
    if (shown == 0) {
        add("    none (the GPU finished all of them)\n");
    }
    lock.unlock();
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
    if (Breadcrumbs::Enabled()) {
        Breadcrumbs::ReportStuck("a GPU wait over 1 s");
    }
}

Scheduler::Scheduler(const Instance& instance, bool threaded_recording)
    : instance{instance}, work_semaphore{instance}, command_pool{instance, &work_semaphore} {
    name = threaded_recording ? "the draws" : "presentation";
    Semaphore::slow_wait_hook.store(&Scheduler::ReportSlowWait, std::memory_order_release);
    crumb_stream = Breadcrumbs::NewStream(
        instance, threaded_recording ? "draws: game and presenter" : "presentation");
    // BB_VK_RECORD_THREAD=0 records on the calling thread.
    const char* env = std::getenv("BB_VK_RECORD_THREAD");
    if (threaded_recording && !(env && env[0] == '0')) {
        record_chunk = AcquireChunk();
        ordered_chunk = AcquireChunk();
        // A segment of ~64 KiB of closures is ~60 draws: a frame of ~700 draws gives each
        // thread several segments, and the render passes a cut reopens stay few.
        const char* segment_kb = std::getenv("BB_VK_SEGMENT_KB");
        split_bytes = size_t(segment_kb ? std::max(1, std::atoi(segment_kb)) : 64) * 1024;
        const u32 count = RecordingThreadCount();
        for (u32 i = 0; i < count; ++i) {
            auto worker = std::make_unique<Worker>();
            if (i == 0) {
                worker->pool = &command_pool;
            } else {
                worker->own_pool = std::make_unique<CommandPool>(instance, &work_semaphore);
                worker->pool = worker->own_pool.get();
            }
            workers.push_back(std::move(worker));
        }
        for (u32 i = 0; i < count; ++i) {
            workers[i]->thread =
                std::jthread(std::bind_front(&Scheduler::RecorderThread, this), i);
        }
        std::printf("GPU: %u Vulkan recording thread%s, segments of %zu KiB\n", count,
                    count == 1 ? "" : "s", split_bytes / 1024);
    }
    // bbport BB_ASYNC_SUBMIT (default on; 0: off): submissions go out from the recording threads
    // once their segments are recorded, instead of this thread waiting for them (~65 us each).
    if (!workers.empty()) {
        const char* async = std::getenv("BB_ASYNC_SUBMIT");
        async_submit = !async || async[0] != '0';
        if (async_submit) {
            std::printf("GPU: submissions go out from the recording threads (BB_ASYNC_SUBMIT)\n");
        }
    }
    AllocateWorkerCommandBuffers();
    priority_ops_thread = std::jthread(std::bind_front(&Scheduler::PriorityOpsThread, this));
}

Scheduler::~Scheduler() {
    if (!workers.empty()) {
        SyncRecording();
        for (auto& worker : workers) {
            worker->thread.request_stop(); // wakes its wait on worker->cv
        }
        for (auto& worker : workers) {
            worker->thread.join();
        }
    }
    // After the last submission (the recording threads submit until they are joined).
    std::scoped_lock lock{queue_log_mutex};
    for (auto& record : queue_log) {
        if (record.scheduler == this) {
            record.scheduler = nullptr;
            record.semaphore = nullptr;
        }
    }
}

void Scheduler::BeginRendering(const RenderState& state) {
    if (is_rendering && render_state == state) {
        return;
    }
    // A cut of the command stream (MaybeSplit) closed this render pass: it goes on in the next
    // command buffer, its attachments not cleared again.
    const bool resume = resume_rendering && !is_rendering && render_state == state;
    // Not a cut of the command stream resuming (MaybeSplit ends no pass of its own).
    if (!resume && !is_rendering && render_state == state && pass_end_stack[0]) {
        TracePassBreak(pass_end_stack);
        pass_end_stack = {};
    }
    resume_rendering = false;
    EndRendering();
    CarrySuspend(false);
    is_rendering = true;
    render_state = state;

    // Owned by the recorded closure (it may run later, on a recording thread).
    struct Pass {
        std::array<vk::RenderingAttachmentInfo, 8> colors;
        vk::RenderingAttachmentInfo depth;
        vk::RenderingAttachmentInfo stencil;
        vk::RenderingInfo info;
        void Begin(vk::CommandBuffer cmdbuf) {
            info.pColorAttachments = colors.data();
            info.pDepthAttachment = info.pDepthAttachment ? &depth : nullptr;
            info.pStencilAttachment = info.pStencilAttachment ? &stencil : nullptr;
            cmdbuf.beginRendering(info);
        }
    } pass{};
    const auto load = [resume](bool clear) {
        return clear && !resume ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
    };
    for (u32 i = 0; i < state.num_color_attachments; ++i) {
        const auto& color = state.color_attachments[i];
        pass.colors[i] = {
            .imageView = color.image_view,
            .imageLayout = color.image_layout,
            .loadOp = load(color.is_clear),
            .storeOp = vk::AttachmentStoreOp::eStore,
            .clearValue = {.color = {.uint32 = color.clear_value}},
        };
    }
    const auto& ds = state.depth_stencil_attachment;
    pass.depth = {
        .imageView = ds.image_view,
        .imageLayout = ds.image_layout,
        .loadOp = load(ds.depth_clear),
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = {.depthStencil = {.depth = std::bit_cast<float>(ds.clear_value[0])}},
    };
    pass.stencil = pass.depth;
    pass.stencil.loadOp = load(ds.stencil_clear);
    pass.stencil.clearValue = vk::ClearValue{.depthStencil = {.stencil = ds.clear_value[1]}};
    pass.info = {
        .renderArea = {{0, 0}, {state.width, state.height}},
        .layerCount = state.num_layers,
        .colorAttachmentCount = state.num_color_attachments,
        .pDepthAttachment = ds.has_depth ? &pass.depth : nullptr,
        .pStencilAttachment = ds.has_stencil ? &pass.stencil : nullptr,
    };
    if (workers.empty()) {
        pass.Begin(current_cmdbuf);
    } else {
        Record([pass](vk::CommandBuffer cmdbuf) mutable { pass.Begin(cmdbuf); });
    }
    CarryResume(true);
}

void Scheduler::EndRendering() {
    if (!is_rendering) {
        return;
    }
    static const bool trace = [] {
        const char* env = std::getenv("BB_PASS_BREAK_TRACE");
        return env && env[0] == '1';
    }();
    if (trace) {
        // This function, then its caller and theirs.
        std::array<void*, std::tuple_size_v<PassEndStack> + 1> frames{};
        const int depth = backtrace(frames.data(), int(frames.size()));
        pass_end_stack = {};
        for (int i = 1; i < depth; ++i) {
            pass_end_stack[i - 1] = frames[i];
        }
    }
    CarrySuspend(true);
    is_rendering = false;
    Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
    CarryResume(false);
}

void Scheduler::TracePassBreak(const PassEndStack& stack) {
    static std::mutex mutex;
    static std::map<PassEndStack, u64> callers;
    static u64 breaks = 0;
    static auto last = std::chrono::steady_clock::now();
    std::scoped_lock lk{mutex};
    ++callers[stack];
    ++breaks;
    const auto now = std::chrono::steady_clock::now();
    if (now - last < std::chrono::seconds(5)) {
        return;
    }
    std::vector<std::pair<u64, PassEndStack>> top;
    for (const auto& [frames, count] : callers) {
        top.emplace_back(count, frames);
    }
    std::ranges::sort(top, std::greater{});
    std::printf("Render pass breaks (the same pass begun again): %llu in %.0f s\n",
                static_cast<unsigned long long>(breaks),
                std::chrono::duration<double>(now - last).count());
    for (size_t i = 0; i < std::min<size_t>(top.size(), 12); ++i) {
        std::string line;
        for (void* frame : top[i].second) {
            if (!frame) {
                break;
            }
            Dl_info info{};
            char text[256];
            if (dladdr(frame, &info) && info.dli_sname) {
                std::snprintf(text, sizeof(text), "%s+0x%lx", info.dli_sname,
                              static_cast<unsigned long>(reinterpret_cast<uintptr_t>(frame) -
                                                         reinterpret_cast<uintptr_t>(info.dli_saddr)));
            } else {
                std::snprintf(text, sizeof(text), "%s+0x%lx",
                              info.dli_fname ? info.dli_fname : "?",
                              static_cast<unsigned long>(reinterpret_cast<uintptr_t>(frame) -
                                                         reinterpret_cast<uintptr_t>(info.dli_fbase)));
            }
            line += line.empty() ? "" : " <- ";
            line += text;
        }
        std::printf("Render pass break caller: %llu x %s\n",
                    static_cast<unsigned long long>(top[i].first), line.c_str());
    }
    callers.clear();
    breaks = 0;
    last = now;
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

void Scheduler::SignalAfterHostCopies(std::function<void()> signal) {
    // A direct segment queues it too: signals queued before it must not be overtaken.
    if (!IsRecordingDeferred() && !direct_segment) {
        WaitHostCopies();
        signal();
        return;
    }
    BbCopy::FlushBatch();
    deferred_signals_issued.fetch_add(1, std::memory_order_relaxed);
    RecordOrdered([signal = std::move(signal), done = deferred_signals_done]() mutable {
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

namespace {
constexpr u32 HostCopyGranuleShift = 16;
constexpr u64 HostCopyMaxGranules = 64; // larger notes and checks are treated as everywhere
u32 HostCopySlotOf(u64 granule, u32 slots) {
    return u32((granule * 0x9e3779b97f4a7c15ull) >> 40) % slots;
}
} // namespace

void Scheduler::NoteHostCopySource(u64 address, u64 size) {
    if (size == 0) {
        return;
    }
    const u64 first = address >> HostCopyGranuleShift;
    const u64 last = (address + size - 1) >> HostCopyGranuleShift;
    std::scoped_lock lk{host_copy_sources_mutex};
    const u64 seq = ++host_copy_source_seq;
    if (last - first >= HostCopyMaxGranules) {
        host_copy_big_seq = seq;
        return;
    }
    for (u64 g = first; g <= last; ++g) {
        auto& slot = host_copy_slots[HostCopySlotOf(g, HostCopySlots)];
        if (slot.seq > host_copy_sources_done && slot.granule != g) {
            slot.granule = HostCopyWildcard; // two pending granules share it
        } else if (slot.granule != HostCopyWildcard || slot.seq <= host_copy_sources_done) {
            slot.granule = g;
        }
        slot.seq = seq;
    }
}

void Scheduler::WaitHostCopiesFor(u64 address, u64 size) {
    {
        std::scoped_lock lk{host_copy_sources_mutex};
        bool overlap = host_copy_big_seq > host_copy_sources_done;
        const u64 first = address >> HostCopyGranuleShift;
        const u64 last = (address + std::max<u64>(size, 1) - 1) >> HostCopyGranuleShift;
        if (last - first >= HostCopyMaxGranules) {
            overlap = true;
        }
        for (u64 g = first; !overlap && g <= last; ++g) {
            const auto& slot = host_copy_slots[HostCopySlotOf(g, HostCopySlots)];
            overlap = slot.seq > host_copy_sources_done &&
                      (slot.granule == g || slot.granule == HostCopyWildcard);
        }
        if (!overlap) {
            BbStats::host_copy_waits_skipped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    WaitHostCopies();
}

void Scheduler::WaitHostCopies() {
    u64 sources_seen;
    {
        std::scoped_lock lk{host_copy_sources_mutex};
        sources_seen = host_copy_source_seq;
    }
    // Copies noted after this point may still run when the waits below return.
    struct Forget {
        Scheduler& s;
        u64 seen;
        ~Forget() {
            std::scoped_lock lk{s.host_copy_sources_mutex};
            s.host_copy_sources_done = std::max(s.host_copy_sources_done, seen);
        }
    } forget{*this, sources_seen};
    if (host_copies_done.load(std::memory_order_acquire) < host_copies_issued) {
        BbStats::WaitTimer timer{BbStats::host_copies_wait_ns};
        BbStats::host_copy_waits.fetch_add(1, std::memory_order_relaxed);
        KickRecording(true);
        while (host_copies_done.load(std::memory_order_acquire) < host_copies_issued) {
            std::this_thread::yield();
        }
    }
    BbStats::WaitTimer timer{BbStats::copy_threads_wait_ns};
    BbCopy::WaitAsync();
}

void Scheduler::KickRecording(bool force) {
    ProducerScope producer{*this, "KickRecording"};
    if (workers.empty()) {
        return;
    }
    // Callers kick where nobody holds the raw command buffer: deferral resumes.
    if (direct_segment) {
        LeaveDirectSegment();
    }
    direct_mode = false;
    if (!force) {
        MaybeSplit();
    }
    // Batches of tens of KiB keep the queue handoff cheap relative to the work it carries.
    if (!force && full_chunks.empty() && CurrentChunk().Size() < 32 * 1024) {
        return;
    }
    HandOver();
}

void Scheduler::MaybeSplit() {
    if (workers.size() < 2 || current_segment + 1 >= MaxSegments || !IsRecordingDeferred() ||
        BbToggle::Disabled(BbToggle::ParallelRecording)) {
        return;
    }
    // Cutting inside a render pass closes and reopens it: only after a longer stretch.
    const size_t bytes = segment_bytes + CurrentChunk().Size();
    if (bytes < (is_rendering ? 2 * split_bytes : split_bytes)) {
        return;
    }
    if (is_rendering) {
        CarrySuspend(true);
        Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
        is_rendering = false;
        resume_rendering = true;
    } else {
        CarrySuspend(false);
    }
    if (async_submit) {
        // Its recording thread ends the segment's command buffer (the pool is that thread's).
        Record([](vk::CommandBuffer cmdbuf) { Check(cmdbuf.end()); });
    }
    HandOver();
    ++current_segment;
    batch->segments[current_segment] = vk::CommandBuffer{};
    segment_bytes = 0;
    active_worker.store(current_segment % workers.size(), std::memory_order_relaxed);
    // The next command buffer starts without dynamic state: the next draw sets all of it.
    // Everything else (pipeline, vertex and index buffers, descriptors, push constants) every
    // draw and dispatch records itself.
    dynamic_state.Invalidate();
    CarryResume(false);
}

void Scheduler::HandOver() {
    // Entered from inside Record*() on this thread: a fault handler recording mid-command.
    if (const char* inside = producer_scope_where;
        inside && (std::strcmp(inside, "Record") == 0 || std::strcmp(inside, "RecordData") == 0 ||
                   std::strcmp(inside, "RecordOrdered") == 0 ||
                   std::strcmp(inside, "RetireChunk") == 0)) {
        ReportProducer("REENTERED recording", u32(gettid()), inside, "HandOver");
    }
    ProducerScope producer{*this, "HandOver"};
    if (!record_chunk || !ordered_chunk) {
        // Left behind by another HandOver interrupted between handing a chunk over and
        // taking a new one (see ProducerScope). Recover instead of dereferencing null.
        ReportProducer("HandOver without a current chunk", producer_tid.load(), "?", "HandOver");
        if (!record_chunk) {
            record_chunk = AcquireChunk();
        }
        if (!ordered_chunk) {
            ordered_chunk = AcquireChunk();
        }
    }
    const bool commands = !full_chunks.empty() || !record_chunk->Empty();
    const bool ordered = !ordered_full.empty() || !ordered_chunk->Empty();
    if (!commands && !ordered) {
        return;
    }
    // The chunks that take the current ones' places, taken first (AcquireChunk locks
    // recorder_mutex) and swapped in below: record_chunk and ordered_chunk are never null, not
    // even for a fault handler recording on this thread meanwhile (issue #100).
    std::unique_ptr<RecordChunk> next_record = record_chunk->Empty() ? nullptr : AcquireChunk();
    std::unique_ptr<RecordChunk> next_ordered = ordered_chunk->Empty() ? nullptr : AcquireChunk();
    Worker& worker = *workers[current_segment % workers.size()];
    bool wake = false;
    {
        std::scoped_lock lk{recorder_mutex};
        if (commands) {
            for (auto& chunk : full_chunks) {
                chunk->segment = current_segment;
                chunk->batch = batch;
                worker.queue.push_back(std::move(chunk));
                ++batch->handed[current_segment];
            }
            if (next_record) {
                record_chunk.swap(next_record); // next_record: the chunk handed over
                segment_bytes += next_record->Size();
                next_record->segment = current_segment;
                next_record->batch = batch;
                worker.queue.push_back(std::move(next_record));
                ++batch->handed[current_segment];
            }
            worker.queued.store(worker.queue.size(), std::memory_order_release);
            wake = worker.sleeping;
        }
        if (ordered) {
            // They run after the commands handed over with them (recorded before or among them).
            const auto queue = [&](std::unique_ptr<RecordChunk> chunk) {
                chunk->segment = current_segment;
                chunk->batch = batch;
                chunk->after = batch->handed[current_segment];
                ordered_queue.push_back(std::move(chunk));
            };
            for (auto& chunk : ordered_full) {
                queue(std::move(chunk));
            }
            if (next_ordered) {
                ordered_chunk.swap(next_ordered); // next_ordered: the chunk handed over
                queue(std::move(next_ordered));
            }
            ordered_queued.store(ordered_queue.size(), std::memory_order_release);
            // Any thread may run them; the current segment's thread is woken for them (when it
            // sleeps): it spins for work anyway, and waking another thread per small copy cost
            // a few percent of the frame rate on 4 cores / 8 threads. A thread that is awake
            // picks them up by itself.
            if (!ordered_running) {
                wake |= worker.sleeping;
            }
        }
    }
    full_chunks.clear();
    ordered_full.clear();
    // A busy thread picks the new chunks up by itself: waking it is a syscall per draw.
    if (wake) {
        worker.cv.notify_one();
    }
}

bool Scheduler::RecordingIdle() const {
    if (!ordered_queue.empty() || ordered_running) {
        return false;
    }
    for (const auto& worker : workers) {
        if (!worker->queue.empty() || worker->busy) {
            return false;
        }
    }
    return true;
}

bool Scheduler::OrderedReady() const {
    if (ordered_queue.empty() || ordered_running) {
        return false;
    }
    const RecordChunk& next = *ordered_queue.front();
    for (u32 i = 0; i < next.segment; ++i) {
        if (next.batch->recorded[i] != next.batch->handed[i]) {
            return false;
        }
    }
    return next.batch->recorded[next.segment] >= next.after;
}

void Scheduler::SyncRecording() {
    if (workers.empty()) {
        return;
    }
    KickRecording(true);
    BbStats::WaitTimer timer{BbStats::sync_recording_ns};
    std::unique_lock lk{recorder_mutex};
    recorder_idle_cv.wait(lk, [this] { return RecordingIdle(); });
}

void Scheduler::EnterDirectMode() {
    // With submissions going out from the recording threads, the commands recorded
    // here get a segment of their own, from this thread's pool, after the segments handed over:
    // the recording threads go on with those instead of this thread waiting for them (~90 us,
    // the upscaler's passes every frame). BB_DIRECT_SEGMENT=0: the wait.
    static const bool own_segment = [] {
        const char* env = std::getenv("BB_DIRECT_SEGMENT");
        return !(env && env[0] == '0');
    }();
    if (own_segment && async_submit && current_segment + 2 < MaxSegments &&
        !BbToggle::Disabled(BbToggle::ThreadedRecording)) {
        if (is_rendering) {
            CarrySuspend(true);
            Record([](vk::CommandBuffer cmdbuf) { cmdbuf.endRendering(); });
            is_rendering = false;
            resume_rendering = true;
        } else {
            CarrySuspend(false);
        }
        // A segment nothing was handed to yet is not begun: it becomes the direct one.
        const bool untouched =
            batch->handed[current_segment] == 0 && full_chunks.empty() && CurrentChunk().Empty();
        if (!untouched) {
            // Its recording thread ends the segment's command buffer (the pool is that thread's).
            Record([](vk::CommandBuffer cmdbuf) { Check(cmdbuf.end()); });
            HandOver();
            ++current_segment;
        }
        if (!direct_pool) {
            direct_pool = std::make_unique<CommandPool>(instance, &work_semaphore);
        }
        current_cmdbuf = BeginCommandBuffer(*direct_pool);
        batch->segments[current_segment] = current_cmdbuf;
        segment_bytes = 0;
        direct_mode = true;
        direct_segment = true;
        dynamic_state.Invalidate();
        CarryResume(false);
        return;
    }
    SyncRecording();
    direct_mode = true;
    // The recording threads are idle: this thread may use the segment's command pool.
    auto& cmdbuf = batch->segments[current_segment];
    if (!cmdbuf) {
        cmdbuf = BeginCommandBuffer(*workers[current_segment % workers.size()]->pool);
    }
    current_cmdbuf = cmdbuf;
}

void Scheduler::LeaveDirectSegment() {
    if (is_rendering) {
        CarrySuspend(true);
        current_cmdbuf.endRendering();
        is_rendering = false;
        resume_rendering = true;
    } else {
        CarrySuspend(false);
    }
    Check(current_cmdbuf.end());
    current_cmdbuf = vk::CommandBuffer{};
    direct_mode = false;
    direct_segment = false;
    ++current_segment;
    batch->segments[current_segment] = vk::CommandBuffer{};
    segment_bytes = 0;
    active_worker.store(current_segment % workers.size(), std::memory_order_relaxed);
    dynamic_state.Invalidate();
    CarryResume(false);
}

void Scheduler::RecorderThread(std::stop_token stoken, u32 index) {
    BbThreads::gpu_side_helper = true; // the submissions go out from here (ordered tasks)
    if (index == 0) {
        Common::SetCurrentThreadName("bb:VkRecorder");
    } else {
        char name[16];
        std::snprintf(name, sizeof(name), "bb:VkRecorder%u", index);
        Common::SetCurrentThreadName(name);
    }
    Worker& worker = *workers[index];
    std::vector<std::unique_ptr<RecordChunk>> ordered;
    const auto has_work = [&] {
        return !worker.queue.empty() || OrderedReady();
    };
    while (true) {
        // The thread of the segment being recorded spins briefly before sleeping: its next
        // chunk usually follows within microseconds, and a sleeping thread costs the producer
        // a wake-up syscall per kick. With few hardware threads (Steam Deck: 8) the spin would
        // take time from guest threads. The others sleep until a new segment reaches them.
        if (active_worker.load(std::memory_order_relaxed) == index) {
            static const auto spin_time =
                std::chrono::microseconds(BbThreads::Available() >= 12 ? 200 : 20);
            const auto spin_until = std::chrono::steady_clock::now() + spin_time;
            for (u32 spins = 1; worker.queued.load(std::memory_order_acquire) == 0 &&
                                ordered_queued.load(std::memory_order_acquire) == 0;
                 ++spins) {
                BbCpu::Pause();
                // The clock is read every 256 pauses, not per iteration.
                if (!(spins & 255) &&
                    (stoken.stop_requested() || std::chrono::steady_clock::now() >= spin_until)) {
                    break;
                }
            }
        }
        std::unique_ptr<RecordChunk> chunk;
        {
            std::unique_lock lk{recorder_mutex};
            if (!has_work()) {
                worker.sleeping = true;
                worker.cv.wait(lk, stoken, has_work);
                worker.sleeping = false;
                if (!has_work()) {
                    return; // stop requested
                }
            }
            if (OrderedReady()) {
                // Every chunk whose commands are recorded, in order.
                do {
                    ordered.push_back(std::move(ordered_queue.front()));
                    ordered_queue.pop_front();
                } while (OrderedReady());
                ordered_running = true;
                ordered_queued.store(ordered_queue.size(), std::memory_order_release);
            } else {
                chunk = std::move(worker.queue.front());
                worker.queue.pop_front();
                worker.queued.store(worker.queue.size(), std::memory_order_release);
                worker.busy = true;
            }
        }
        if (!ordered.empty()) {
            for (auto& task : ordered) {
                task->Execute({});
            }
            std::scoped_lock lk{recorder_mutex};
            for (auto& task : ordered) {
                free_chunks.push_back(std::move(task));
            }
            ordered.clear();
            ordered_running = false;
            recorder_idle_cv.notify_all();
            continue;
        }
        // The segment's command buffer is begun by the first chunk; it is ended at the
        // submission, after SyncRecording(). Only this thread uses the pool meanwhile.
        auto& cmdbuf = chunk->batch->segments[chunk->segment];
        if (!cmdbuf) {
            cmdbuf = BeginCommandBuffer(*worker.pool);
        }
        chunk->Execute(cmdbuf);
        {
            std::scoped_lock lk{recorder_mutex};
            ++chunk->batch->recorded[chunk->segment];
            free_chunks.push_back(std::move(chunk));
            worker.busy = false;
            if (worker.queue.empty()) {
                recorder_idle_cv.notify_all();
            }
        }
    }
}

void Scheduler::Flush(SubmitInfo& info) {
    SubmitExecution(info);
}

void Scheduler::Flush() {
    SubmitInfo info{};
    Flush(info);
}

void Scheduler::Finish() {
    const u64 tick = CurrentTick();
    Flush();
    Wait(tick);
}

void Scheduler::Wait(u64 tick) {
    if (tick >= CurrentTick()) {
        Flush(); // the tick is the one being recorded: nothing would ever signal it
    }
    BbWaitTrace::Scope trace;
    BbStats::WaitTimer timer{BbStats::tick_wait_ns};
    work_semaphore.Wait(tick);
}

void Scheduler::PopPendingOperations() {
    if (num_pending_ops.load(std::memory_order_acquire) == 0) {
        return; // every draw comes here
    }
    std::scoped_lock lk{pending_ops_mutex};
    if (pending_ops.empty()) {
        return;
    }
    // Asking the driver is an ioctl: when the oldest operation is not free yet, only every
    // 32nd call asks (~0.3 ms apart at the draw rate).
    if (!work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        if ((++pending_polls & 31) != 0 && !BbToggle::Disabled(BbToggle::PendingPollLimit)) {
            return;
        }
        work_semaphore.Refresh();
    }
    while (!pending_ops.empty() && work_semaphore.IsFree(pending_ops.front().gpu_tick)) {
        auto op = std::move(pending_ops.front());
        pending_ops.pop_front();
        num_pending_ops.fetch_sub(1, std::memory_order_release);
        op.callback();
    }
}

SubmissionBatch* Scheduler::AcquireBatch() {
    std::scoped_lock lk{recorder_mutex};
    SubmissionBatch* next;
    if (free_batches.empty()) {
        next = all_batches.emplace_back(std::make_unique<SubmissionBatch>()).get();
    } else {
        next = free_batches.back();
        free_batches.pop_back();
    }
    next->Reset();
    return next;
}

void Scheduler::AllocateWorkerCommandBuffers() {
    if (async_submit) {
        // BB_ASYNC_SUBMIT: the earlier submissions may still be recorded (and their pools
        // used) on the recording threads: segment 0 is begun there too, on its first chunk,
        // or by EnterDirectMode once they are idle.
        batch = AcquireBatch();
        current_cmdbuf = vk::CommandBuffer{};
        direct_mode = false;
    } else {
        const vk::CommandBufferBeginInfo begin_info = {
            .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
        };
        current_cmdbuf = command_pool.Commit();
        Check(current_cmdbuf.begin(begin_info));
        if (DynCheck::Enabled()) {
            DynCheck::Begin(current_cmdbuf);
        }
        // The first segment (worker 0, whose pool this is) of the next submission.
        if (!batch) {
            batch = AcquireBatch();
        }
        batch->Reset();
        batch->segments[0] = current_cmdbuf;
    }
    current_segment = 0;
    segment_bytes = 0;
    active_worker.store(0, std::memory_order_relaxed);

    // Invalidate dynamic state so it gets applied to the new command buffer.
    dynamic_state.Invalidate();

}

void Scheduler::QueueSubmit(std::span<const vk::CommandBuffer> cmdbufs, SubmitInfo& info) {
    // A prepared frame is first read by the present blit (Transfer), and diagnostic
    // dumps also read it before ColorAttachmentOutput. Waiting only at color output
    // lets those reads race the encode that produces the frame. All submission waits
    // cover their earliest consumers, regardless of their index in SubmitInfo.
    static constexpr std::array<vk::PipelineStageFlags, SubmitInfo::MaxSemaphores> wait_stages{
        vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
        vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands};
    const vk::TimelineSemaphoreSubmitInfo timeline{
        .waitSemaphoreValueCount = info.num_wait_semas,
        .pWaitSemaphoreValues = info.wait_ticks.data(),
        .signalSemaphoreValueCount = info.num_signal_semas,
        .pSignalSemaphoreValues = info.signal_ticks.data(),
    };
    const vk::SubmitInfo submit{
        .pNext = &timeline,
        .waitSemaphoreCount = info.num_wait_semas,
        .pWaitSemaphores = info.wait_semas.data(),
        .pWaitDstStageMask = wait_stages.data(),
        .commandBufferCount = u32(cmdbufs.size()),
        .pCommandBuffers = cmdbufs.data(),
        .signalSemaphoreCount = info.num_signal_semas,
        .pSignalSemaphores = info.signal_semas.data(),
    };
    QueueRecord* record = nullptr;
    {
        std::scoped_lock lock{queue_log_mutex};
        record = &queue_log[queue_log_count++ % QueueRecords];
        *record = {.scheduler = this,
                   .semaphore = &work_semaphore,
                   .num_waits = info.num_wait_semas,
                   .waits = info.wait_semas,
                   .values = info.wait_ticks,
                   .labels = info.wait_labels,
                   .timeline = info.wait_timeline,
                   .cmdbufs = u32(cmdbufs.size()),
                   .queue_ops = u32(info.before_submit.size()),
                   .enter_ns = SteadyNs()};
        for (u32 i = 0; i < info.num_signal_semas; ++i) {
            if (info.signal_semas[i] == work_semaphore.Handle()) {
                record->tick = info.signal_ticks[i];
            }
        }
    }
    // The record stays this submission's: 256 later ones would have to come first (they wait
    // for submit_mutex, which this one holds).
    for (auto& operation : info.before_submit) {
        operation();
    }
    {
        std::scoped_lock lock{queue_log_mutex};
        record->ops_ns = SteadyNs();
    }
    const vk::Result result = instance.GetGraphicsQueue().submit(submit, info.fence);
    {
        std::scoped_lock lock{queue_log_mutex};
        record->exit_ns = SteadyNs();
    }
    if (result == vk::Result::eErrorDeviceLost) {
        Breadcrumbs::ReportDeviceLost("submit");
    }
    Check<"submit commands">(result);
    recorded_segments.fetch_add(cmdbufs.size(), std::memory_order_relaxed);
    recorded_submissions.fetch_add(1, std::memory_order_relaxed);
}

void Scheduler::SubmitExecution(SubmitInfo& info) {
    ProducerScope producer{*this, "SubmitExecution"};
    if (async_submit) {
        SubmitAsync(info);
        return;
    }
    std::scoped_lock lk{submit_mutex};
    const u64 tick = work_semaphore.NextTick();
    if (on_submit) {
        on_submit(info);
    }
    EndRendering();
    CarrySuspend(false);
    if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(this)) {
        // Until the next submission's first timestamp: mostly the GPU waiting for it.
        if (std::exchange(frame_end_submission, false)) {
            submission_in_frame = 0;
            profiler->Mark(0x5B319ull, [] {
                return std::string{"(after a frame's last submission: presenter, GPU idle)"};
            });
        } else {
            // Numbered within the frame: which of the frame's submissions the GPU waits after.
            const u32 n = std::min(submission_in_frame++, 15u);
            profiler->Mark(0x5B320ull + n, [n] {
                return fmt::format("(between submissions: GPU idle, after submission {})", n);
            });
        }
    }
    SyncRecording();
    // Guest memory copies into staging this submission reads (copy threads).
    WaitHostCopies();
    resume_rendering = false;
    // The segments' command buffers in stream order (the recording threads are idle).
    std::array<vk::CommandBuffer, MaxSegments> cmdbufs;
    u32 count = 0;
    if (workers.empty()) {
        cmdbufs[count++] = current_cmdbuf;
    } else {
        for (u32 i = 0; i <= current_segment; ++i) {
            if (batch->segments[i]) {
                cmdbufs[count++] = batch->segments[i];
            }
        }
    }
    for (u32 i = 0; i < count; ++i) {
        Check(cmdbufs[i].end());
    }
    info.AddSignal(work_semaphore.Handle(), tick);
    QueueSubmit({cmdbufs.data(), count}, info);
    submitted_tick.store(tick, std::memory_order_release);
    submitted_tick.notify_all();
    // No semaphore query here (an ioctl per submission): PopPendingOperations asks when an
    // operation waits, the GPU signal thread on every fence.
    AllocateWorkerCommandBuffers();
    if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(this)) {
        profiler->Mark(0x5B318ull, [] { // as in SubmitAsync
            return std::string{"(submission start: copies and barriers before its first mark)"};
        });
    }
    CarryResume(false);
    PopPendingOperations();
}

void Scheduler::SubmitAsync(SubmitInfo& info) {
    const u64 tick = work_semaphore.NextTick();
    if (on_submit) {
        on_submit(info);
    }
    EndRendering();
    CarrySuspend(false);
    if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(this)) {
        if (std::exchange(frame_end_submission, false)) {
            submission_in_frame = 0;
            profiler->Mark(0x5B319ull, [] {
                return std::string{"(after a frame's last submission: presenter, GPU idle)"};
            });
        } else {
            // Numbered within the frame: which of the frame's submissions the GPU waits after.
            const u32 n = std::min(submission_in_frame++, 15u);
            profiler->Mark(0x5B320ull + n, [n] {
                return fmt::format("(between submissions: GPU idle, after submission {})", n);
            });
        }
    }
    resume_rendering = false;
    // Every segment's command buffer is ended by its recording thread (the pool is its own);
    // MaybeSplit ended the earlier ones. A direct segment is this thread's.
    if (direct_segment) {
        Check(current_cmdbuf.end());
        current_cmdbuf = vk::CommandBuffer{};
        direct_mode = false;
        direct_segment = false;
    } else {
        Record([](vk::CommandBuffer cmdbuf) { Check(cmdbuf.end()); });
    }
    // Guest memory copies into staging this submission reads: on the copy threads by then.
    BbCopy::FlushBatch();
    info.AddSignal(work_semaphore.Handle(), tick);
    // Goes out once the commands handed over before it are recorded (an ordered task: in order
    // with the other submissions and after the copies and signals queued before it).
    RecordOrdered([this, submitted = batch, info = std::move(info), tick]() mutable {
        BbCopy::WaitAsync();
        std::array<vk::CommandBuffer, MaxSegments> cmdbufs;
        u32 count = 0;
        for (const auto cmdbuf : submitted->segments) {
            if (cmdbuf) {
                cmdbufs[count++] = cmdbuf;
            }
        }
        {
            std::scoped_lock lk{submit_mutex};
            QueueSubmit({cmdbufs.data(), count}, info);
        }
        BbTimeline::Note(BbTimeline::VulkanSubmit, tick, 2); // the actual vkQueueSubmit
        submitted_tick.store(tick, std::memory_order_release);
        submitted_tick.notify_all();
        std::scoped_lock lk{recorder_mutex};
        free_batches.push_back(submitted);
    });
    KickRecording(true);
    AllocateWorkerCommandBuffers();
    // BB_GPU_PROFILE: the next submission's first command, so that the gap above holds only the
    // GPU's time between submissions; what comes before its first mark is the work recorded
    // outside passes and dispatches (copies, uploads, barriers).
    if (auto* profiler = GpuProfiler::Get(); profiler && profiler->Records(this)) {
        profiler->Mark(0x5B318ull, [] {
            return std::string{"(submission start: copies and barriers before its first mark)"};
        });
    }
    CarryResume(false);
    PopPendingOperations();
}

void Scheduler::PriorityOpsThread(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:SchedOps");
    while (true) {
        PendingOp op;
        {
            std::unique_lock lk{priority_ops_mutex};
            if (!priority_ops_cv.wait(lk, stop, [this] { return !priority_ops.empty(); })) {
                return;
            }
            op = std::move(priority_ops.front());
            priority_ops.pop_front();
        }
        work_semaphore.Wait(op.gpu_tick);
        if (stop.stop_requested()) {
            return;
        }
        op.callback();
    }
}

} // namespace Vulkan
