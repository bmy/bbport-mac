// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: the command stream of a queue. Commands are recorded in order, by default on
// recording threads (closures handed over in chunks, segments recorded in parallel into their
// own command buffers); render passes open and close as the state asks; submissions signal a
// timeline semaphore whose ticks free deferred operations. Guest memory copies and the fences
// after them run in order with the commands (RecordOrdered).
#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <span>
#include <thread>
#include <vector>

#include "bbport_toggles.h"
#include "common/assert.h"
#include "common/unique_function.h"
#include "render/dynamic_state.h"
#include "render/record_chunk.h"
#include "render/vk_breadcrumbs.h"
#include "vulkan/command_pool.h"
#include "vulkan/timeline.h"

namespace Vulkan {

class Instance;

/// An attachment of a render pass: its view and layout, and whether (and to what) it is
/// cleared when the pass begins.
struct RenderAttachment {
    vk::ImageView image_view;
    vk::ImageLayout image_layout;
    std::array<u32, 4> clear_value; ///< color bits; depth: [0] float bits, [1] stencil
    union {
        u32 is_clear; ///< color: cleared
        struct {
            bool has_depth;
            bool depth_clear;
            bool has_stencil;
            bool stencil_clear;
        };
    };
};
static_assert(std::has_unique_object_representations_v<RenderAttachment>);

/// A render pass: compared bytewise to tell whether an open one can go on.
struct RenderState {
    std::array<RenderAttachment, 8> color_attachments;
    RenderAttachment depth_stencil_attachment;
    u16 width;
    u16 height;
    u16 num_layers;
    u16 num_color_attachments;

    bool operator==(const RenderState& other) const noexcept {
        return std::memcmp(this, &other, sizeof(RenderState)) == 0;
    }
};
static_assert(std::has_unique_object_representations_v<RenderState>);

/// What a submission waits for and signals besides the scheduler's own timeline tick.
struct SubmitInfo {
    static constexpr u32 MaxSemaphores = 4;
    std::array<vk::Semaphore, MaxSemaphores> wait_semas{};
    std::array<u64, MaxSemaphores> wait_ticks{};
    /// What each wait is, for the report of a long GPU wait; binary ones cannot be queried.
    std::array<const char*, MaxSemaphores> wait_labels{};
    std::array<bool, MaxSemaphores> wait_timeline{};
    std::array<vk::Semaphore, MaxSemaphores> signal_semas{};
    std::array<u64, MaxSemaphores> signal_ticks{};
    vk::Fence fence{};
    u32 num_wait_semas = 0;
    u32 num_signal_semas = 0;
    /// Queue operations that go out right before the submission, in queue order (sparse
    /// binds): with BB_ASYNC_SUBMIT the submission itself goes out later, on a recording thread.
    std::vector<std::function<void()>> before_submit;

    /// A binary semaphore.
    void AddWait(vk::Semaphore semaphore, const char* label = "a binary semaphore") {
        AddWaitImpl(semaphore, 1, label, false);
    }
    /// `tick` of a timeline semaphore.
    void AddWait(vk::Semaphore semaphore, u64 tick, const char* label = "a timeline") {
        AddWaitImpl(semaphore, tick, label, true);
    }
    void AddWaitImpl(vk::Semaphore semaphore, u64 tick, const char* label, bool timeline) {
        ASSERT(num_wait_semas < MaxSemaphores);
        wait_semas[num_wait_semas] = semaphore;
        wait_labels[num_wait_semas] = label;
        wait_timeline[num_wait_semas] = timeline;
        wait_ticks[num_wait_semas++] = tick;
    }
    void AddSignal(vk::Semaphore semaphore, u64 tick = 1) {
        ASSERT(num_signal_semas < MaxSemaphores);
        signal_semas[num_signal_semas] = semaphore;
        signal_ticks[num_signal_semas++] = tick;
    }
    void AddSignal(vk::Fence fence_) {
        fence = fence_;
    }
};

using SubmitFunc = Common::UniqueFunction<void, SubmitInfo&>;

class Scheduler {
public:
    /// `threaded_recording`: commands passed to Record() are recorded by recording threads.
    explicit Scheduler(const Instance& instance, bool threaded_recording = false);
    ~Scheduler();

    /// Submits what was recorded (it signals the current tick, with `info`'s waits and
    /// signals); the next commands belong to the next tick.
    void Flush(SubmitInfo& info);
    void Flush();

    /// Submits and waits until the GPU has run it.
    void Finish();

    /// Waits for the GPU to complete the submissions before the current one, which is
    /// not sent: for callers that keep recording into it (the upscaler between CommandBuffer()
    /// and the end of its pass) and free resources only earlier submissions used.
    void WaitSubmittedWork() {
        const u64 tick = CurrentTick();
        if (tick > 1) {
            Wait(tick - 1);
        }
    }

    /// Waits until the GPU has finished `tick` (submitting it first if it is the current one).
    void Wait(u64 tick);

    /// Runs the deferred operations whose tick the GPU has finished.
    void PopPendingOperations();

    /// Opens a render pass with `state`, unless one with exactly it is open.
    void BeginRendering(const RenderState& state);

    /// Closes the open render pass, if any.
    [[gnu::noinline]] void EndRendering();
    /// BB_PASS_BREAK_TRACE=1 (diagnostics): who ended a render pass that began again with the same
    /// state (a break: its attachments stored and loaded again), every 5 s. bbport: the caller and
    /// the callers above it (what ended it: a buffer copy alone does not say why it was made).
    using PassEndStack = std::array<void*, 4>;
    static void TracePassBreak(const PassEndStack& callers);

    /// Called at every submission, before it is prepared (it may add waits and signals).
    void SetSubmitCallback(SubmitFunc&& callback) {
        on_submit = std::move(callback);
    }

    /// The last render pass begun (open or not).
    const RenderState& GetRenderState() const {
        return render_state;
    }

    DynamicState& GetDynamicState() {
        return dynamic_state;
    }

    /// Returns the current command buffer for recording on the calling thread. With threaded
    /// recording this first waits until all commands passed to Record() are recorded, then
    /// records directly (Record() included, so callers may mix both) until the next
    /// KickRecording() or submission.
    [[gnu::noinline]] vk::CommandBuffer CommandBuffer() {
        ProducerScope producer{*this, "CommandBuffer"};
        if (!workers.empty() && !direct_mode) {
            EnterDirectMode();
            direct_recordings.fetch_add(1, std::memory_order_relaxed);
            TraceDirectRecording(__builtin_return_address(0));
        }
        return current_cmdbuf;
    }

    /// BB_RECORDER_TRACE=1: prints the most frequent CommandBuffer() callers every 2000 calls.
    static void TraceDirectRecording(void* caller);

    /// The recording side (current chunks, segments) is used by one thread at a time.
    /// A second thread inside it at once — or HandOver() entered from inside Record*() on the
    /// same thread (a fault handler) — is reported with both threads' names and this thread's
    /// stack: a null record_chunk/ordered_chunk crashed HandOver and SmallGuestCopy.
    class ProducerScope {
    public:
        ProducerScope(Scheduler& scheduler_, const char* where) noexcept : scheduler{scheduler_} {
            if (ProducerCheck()) {
                Enter(where);
            }
        }
        ~ProducerScope() noexcept {
            if (ProducerCheck()) {
                Leave();
            }
        }
        /// BB_PRODUCER_CHECK=1 (the launcher's crash diagnostics); off otherwise, the check
        /// costs an atomic exchange per recorded command.
        static bool ProducerCheck() noexcept {
            static const bool enabled = [] {
                const char* env = std::getenv("BB_PRODUCER_CHECK");
                return env && env[0] == '1';
            }();
            return enabled;
        }
        ProducerScope(const ProducerScope&) = delete;
        ProducerScope& operator=(const ProducerScope&) = delete;

    private:
        void Enter(const char* where) noexcept;
        void Leave() noexcept;
        Scheduler& scheduler;
        const char* previous_where = nullptr;
        bool outer = false;
    };

    /// Records `func(vk::CommandBuffer)` in order with other commands. The closure must own
    /// everything it uses (capture by value): it may run later on the recording thread.
    template <typename Func>
    void Record(Func&& func) {
        ProducerScope producer{*this, "Record"};
        if (workers.empty() || direct_mode) {
            func(current_cmdbuf);
            return;
        }
        if (BbToggle::Disabled(BbToggle::ThreadedRecording)) {
            EnterDirectMode();
            func(current_cmdbuf);
            return;
        }
        if (!CurrentChunk().Push(std::forward<Func>(func))) {
            RetireChunk();
            const bool pushed = CurrentChunk().Push(std::forward<Func>(func));
            ASSERT(pushed);
        }
    }

    /// Record() for a draw, dispatch or pass that runs shaders, between GPU breadcrumbs
    /// (vk_breadcrumbs.h): on a GPU hang they say which one never finished.
    template <typename Func>
    void RecordCrumb(const Breadcrumbs::Crumb& crumb, Func&& func) {
        if (!Breadcrumbs::Enabled()) {
            Record(std::forward<Func>(func));
            return;
        }
        const u32 id = Breadcrumbs::Note(crumb_stream, crumb);
        Record([func = std::forward<Func>(func), stream = crumb_stream,
                id](vk::CommandBuffer cmdbuf) mutable {
            Breadcrumbs::Mark(cmdbuf, stream, id, false);
            func(cmdbuf);
            Breadcrumbs::Mark(cmdbuf, stream, id, true);
        });
    }

    /// Waits until the submission signalling `tick` is in the queue. Another thread's
    /// submission waiting for the tick on the same queue must not go first (BB_ASYNC_SUBMIT
    /// submits from the recording threads): the queue would wait for a later submission.
    void WaitSubmitted(u64 tick) {
        u64 submitted = submitted_tick.load(std::memory_order_acquire);
        while (submitted < tick) {
            submitted_tick.wait(submitted, std::memory_order_acquire);
            submitted = submitted_tick.load(std::memory_order_acquire);
        }
    }

    /// The submission signalling `tick` is in the queue already (no wait).
    [[nodiscard]] u64 SubmittedTick() const noexcept {
        return submitted_tick.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool IsSubmitted(u64 tick) const noexcept {
        return submitted_tick.load(std::memory_order_acquire) >= tick;
    }

    /// The breadcrumb stream of this scheduler, for passes recorded on CommandBuffer().
    [[nodiscard]] u32 CrumbStream() const noexcept {
        return crumb_stream;
    }

    /// True when Record() defers commands (and RecordData() copies into chunks).
    [[nodiscard]] bool IsRecordingDeferred() const noexcept {
        return !workers.empty() && !direct_mode &&
               !BbToggle::Disabled(BbToggle::ThreadedRecording);
    }

    /// Makes room for `bytes` of RecordData() plus the command that uses them in the current
    /// chunk: data and command must share a chunk, which is recycled once executed.
    void ReserveRecordData(size_t bytes) {
        if (!IsRecordingDeferred()) {
            return;
        }
        ASSERT(bytes + 1024 <= RecordChunk::Capacity);
        if (RecordChunk::Capacity - CurrentChunk().Size() < bytes + 1024) {
            RetireChunk();
        }
    }

    /// Copies `data` into recording storage that lives until the command that uses it has
    /// been recorded (the same chunk). With threaded recording off, returns `data` itself.
    template <typename T>
    std::span<const T> RecordData(std::span<const T> data) {
        ProducerScope producer{*this, "RecordData"};
        if (!IsRecordingDeferred() || data.empty()) {
            return data;
        }
        // Room for the data and the command that follows it, so both stay in one chunk.
        const size_t bytes = data.size_bytes();
        ReserveRecordData(bytes + alignof(T));
        auto* dst = static_cast<T*>(CurrentChunk().Allocate(bytes, alignof(T)));
        std::memcpy(dst, data.data(), bytes);
        return {dst, data.size()};
    }

    /// Hands recorded commands to the recording threads. Called at points where no caller holds
    /// the raw command buffer; small batches are kept. Without `force` (the end of a draw or
    /// dispatch: nothing but the dynamic state and the render pass carries over to the next
    /// command) a long enough segment is cut there, and the next one goes to another thread.
    void KickRecording(bool force = false);

    /// Work that may not span command buffers or render pass instances (an occlusion
    /// query, vk_occlusion.h). While one is set, Suspend is called before every boundary (a
    /// render pass beginning or ending, a segment cut, a direct segment, a submission) and Resume
    /// right after it; `inside_render_pass` tells where the commands they record go. Called on
    /// the recording side, as Record.
    struct CarriedScope {
        virtual void Suspend(bool inside_render_pass) = 0;
        virtual void Resume(bool inside_render_pass) = 0;
    protected:
        ~CarriedScope() = default;
    };
    void SetCarriedScope(CarriedScope* scope) {
        carried_scope = scope;
    }

    /// Waits until every recorded command is in the command buffer.
    void SyncRecording();

    /// Guest memory copies on the recording thread (small ones, RecordHostCopy) or the
    /// copy threads (BbCopy::Async) must be done before the guest learns the GPU is past them
    /// (it may then rewrite the memory, e.g. UI vertices: flickering) and before the
    /// submission that reads them. Waits for all.
    void WaitHostCopies();

    /// Guest memory [address, address + size) a host copy issued now will read (the
    /// source of a deferred upload or rename). WaitHostCopiesFor waits only for these.
    void NoteHostCopySource(u64 address, u64 size);
    /// Waits for the host copies when one of them may still read [address, address + size): a
    /// write there (the command processor's WRITE_DATA/DMA) must not change what they copy.
    /// Writes elsewhere do not wait (the hardware's command processor writes do not wait for
    /// earlier draws either).
    void WaitHostCopiesFor(u64 address, u64 size);

    /// Runs `copy` on a recording thread in order with the commands, the other host copies and
    /// the signals of SignalAfterHostCopies (the threads spin for work anyway, so small copies cost no wakeup);
    /// WaitHostCopies() covers it.
    template <typename Func>
    void RecordHostCopy(Func&& copy) {
        RecordOrdered([copy = std::forward<Func>(copy), this, seq = ++host_copies_issued] {
            copy();
            host_copies_done.store(seq, std::memory_order_release);
        });
    }

    /// Runs `signal` once the guest memory copies issued so far are done, without
    /// waiting here: the recording thread reaches it after the copies queued before it and hands
    /// it to the copy threads' completion (BbCopy::AfterCopies).
    void SignalAfterHostCopies(std::function<void()> signal);

    /// Before a guest-visible write that is not deferred (WriteData, end-of-shader
    /// fences, flip): waits until every signal handed to SignalAfterHostCopies ran, so that
    /// the write cannot overtake an earlier fence. Otherwise the guest, seeing the later value,
    /// may free memory the earlier fence then writes into (corrupted heap, guest fault).
    void WaitDeferredSignals();

    /// Whether a render pass with exactly this state is open.
    /// A render pass instance is open in the command stream.
    [[nodiscard]] bool IsRendering() const noexcept {
        return is_rendering;
    }

    [[nodiscard]] bool IsRenderingWith(const RenderState& state) const {
        return is_rendering && render_state == state;
    }

    /// Whether BeginRendering(state) would start a new render pass.
    [[nodiscard]] bool WillBeginRendering(const RenderState& state) const {
        return !(is_rendering && render_state == state);
    }

    /// CommandBuffer() calls that waited for a recording thread (BB_FRAME_STATS).
    static inline std::atomic<u64> direct_recordings{0};
    /// Command buffer segments submitted (BB_FRAME_STATS: segments per submission).
    static inline std::atomic<u64> recorded_segments{0};
    static inline std::atomic<u64> recorded_submissions{0};

    /// The tick the commands recorded now will signal.
    [[nodiscard]] u64 CurrentTick() const noexcept {
        return work_semaphore.CurrentTick();
    }

    /// Whether the GPU has finished `tick` (asks the driver when the last answer was no).
    /// BB_GPU_PROFILE: the next submission ends a frame (the gap after it holds the presenter's
    /// own submission).
    void NoteFrameEndSubmission() noexcept {
        frame_end_submission = true;
    }

    [[nodiscard]] bool IsFree(u64 tick) noexcept {
        if (work_semaphore.IsFree(tick)) {
            return true;
        }
        work_semaphore.Refresh();
        return work_semaphore.IsFree(tick);
    }

    /// The timeline semaphore the submissions signal.
    [[nodiscard]] Semaphore* GetWorkSemaphore() noexcept {
        return &work_semaphore;
    }

    /// Runs `func` once the GPU has finished the current tick: at a later submission or
    /// PopPendingOperations, on the recording side.
    void DeferOperation(Common::UniqueFunction<void>&& func) {
        std::scoped_lock lk{pending_ops_mutex};
        pending_ops.push_back({std::move(func), CurrentTick()});
        num_pending_ops.fetch_add(1, std::memory_order_release);
    }

    /// Runs `func` as soon as the GPU has finished the current tick, on a thread of its own.
    void DeferPriorityOperation(Common::UniqueFunction<void>&& func) {
        {
            std::scoped_lock lk{priority_ops_mutex};
            priority_ops.push_back({std::move(func), CurrentTick()});
        }
        priority_ops_cv.notify_one();
    }

    /// Held around every queue submission and present (the queue is shared).
    static std::mutex submit_mutex;

private:
    static constexpr u32 MaxSegments = SubmissionBatch::MaxSegments;

    /// A recording thread. It owns a command pool (worker 0: the scheduler's) and records the
    /// segments i with i % workers.size() == its index.
    struct Worker {
        CommandPool* pool{};
        std::unique_ptr<CommandPool> own_pool;
        std::deque<std::unique_ptr<RecordChunk>> queue; ///< guarded by recorder_mutex
        std::atomic<size_t> queued{0};                  ///< queue.size() for lock-free polling
        bool busy = false;     ///< running a chunk (guarded by recorder_mutex)
        bool sleeping = false; ///< waiting on cv (guarded by recorder_mutex)
        std::condition_variable_any cv;
        std::jthread thread;
    };

    void AllocateWorkerCommandBuffers();
    SubmissionBatch* AcquireBatch();
    /// BB_ASYNC_SUBMIT: hands the submission to the recording threads (it goes out once its
    /// segments are recorded) instead of waiting for them here.
    void SubmitAsync(SubmitInfo& info);

    std::unique_ptr<RecordChunk> AcquireChunk();

    /// Moves the full current chunk aside; a new one takes its place (swapped in: never null).
    void RetireChunk() {
        ProducerScope producer{*this, "RetireChunk"};
        auto full = AcquireChunk();
        record_chunk.swap(full);
        if (full) {
            segment_bytes += full->Size();
            full_chunks.push_back(std::move(full));
        }
    }

    /// The current chunks, never null: replacements are swapped in, so a fault handler that
    /// records on this thread while one is replaced still finds one; a null one (a build without
    /// that, issue #100: a read of RecordChunk::Capacity at address 0x20000) is replaced here.
    RecordChunk& CurrentChunk() {
        if (!record_chunk) [[unlikely]] {
            record_chunk = AcquireChunk();
        }
        return *record_chunk;
    }
    RecordChunk& OrderedChunk() {
        if (!ordered_chunk) [[unlikely]] {
            ordered_chunk = AcquireChunk();
        }
        return *ordered_chunk;
    }

    /// Waits for the recording threads, then records on this thread into the current segment's
    /// command buffer until the next KickRecording(). With BB_ASYNC_SUBMIT, into a
    /// segment of its own instead, without waiting (direct_segment).
    void EnterDirectMode();

    /// Ends the direct segment (this thread's command buffer); the next commands go to a new
    /// segment of the recording threads.
    void LeaveDirectSegment();

    /// Whether ordered tasks run on the calling thread right away: no recording threads, or
    /// direct mode after waiting for them (not a direct segment: earlier tasks may be queued).
    [[nodiscard]] bool OrderedInline() const noexcept {
        return workers.empty() || (direct_mode && !direct_segment);
    }

    /// Cuts the command stream after the current segment when it is long enough.
    void MaybeSplit();

    /// Hands the current segment's chunks and the ordered chunks to the recording threads.
    void HandOver();

    /// Runs `func()` on a recording thread, in order with the other ordered tasks (guest memory
    /// copies and the fence signals after them) and after every command recorded before it, in
    /// all segments: the guest sees its fences no earlier than in 0.2. Fences signalled ahead of
    /// the queued recording coincided with frequent guest heap corruption (guest offset
    /// 0x263b8e7: 3 of 4 soak runs instead of none).
    template <typename Func>
    void RecordOrdered(Func&& func) {
        ProducerScope producer{*this, "RecordOrdered"};
        if (OrderedInline()) {
            func();
            return;
        }
        if (BbToggle::Disabled(BbToggle::ThreadedRecording)) {
            EnterDirectMode();
            func();
            return;
        }
        auto command = [func = std::forward<Func>(func)](vk::CommandBuffer) mutable { func(); };
        if (!OrderedChunk().Push(std::move(command))) {
            auto full = AcquireChunk();
            ordered_chunk.swap(full);
            ordered_full.push_back(std::move(full));
            const bool pushed = OrderedChunk().Push(std::move(command));
            ASSERT(pushed);
        }
    }

    [[nodiscard]] bool RecordingIdle() const; ///< with recorder_mutex held

    /// Whether the oldest ordered chunk may run: the commands handed over before it are
    /// recorded (with recorder_mutex held).
    [[nodiscard]] bool OrderedReady() const;

    void RecorderThread(std::stop_token stoken, u32 index);

    void SubmitExecution(SubmitInfo& info);

    void PriorityOpsThread(std::stop_token stop);
    /// Submits `cmdbufs` with `info`'s waits and signals (submit_mutex held).
    void QueueSubmit(std::span<const vk::CommandBuffer> cmdbufs, SubmitInfo& info);
    /// Semaphore::slow_wait_hook: what the queue holds while a wait takes over a second.
    static void ReportSlowWait(const Semaphore& semaphore, u64 tick, u64 waited_ms, bool done);
    const char* name = "";

private:
    const Instance& instance;
    u32 crumb_stream = 0; ///< GPU breadcrumbs (vk_breadcrumbs.h)
    Semaphore work_semaphore;
    CommandPool command_pool;
    DynamicState dynamic_state;
    SubmitFunc on_submit{};
    vk::CommandBuffer current_cmdbuf;
    struct PendingOp {
        Common::UniqueFunction<void> callback;
        u64 gpu_tick;
    };
    /// Operations may defer more operations while they run: a recursive mutex.
    std::deque<PendingOp> pending_ops;
    std::recursive_mutex pending_ops_mutex;
    u32 pending_polls = 0;
    std::atomic<u32> num_pending_ops{0}; ///< pending_ops.size(), checked without the lock
    std::deque<PendingOp> priority_ops;
    std::mutex priority_ops_mutex;
    std::condition_variable_any priority_ops_cv;
    std::jthread priority_ops_thread;
    RenderState render_state;
    bool is_rendering = false;
    // Threaded recording. Commands go into chunks of the current segment; segments are
    // recorded in parallel, each into its own command buffer, and a submission runs their
    // command buffers in order. Guest memory copies and fence signals go into ordered chunks,
    // which one recording thread at a time runs in order, each once the commands handed over
    // before it are recorded.
    std::vector<std::unique_ptr<Worker>> workers; ///< empty: recording on the calling thread
    /// The current submission's segments and the chunks of each handed to the recording threads
    /// and recorded (guarded by recorder_mutex); with BB_ASYNC_SUBMIT earlier ones may still be
    /// in flight (free_batches: those submitted).
    SubmissionBatch* batch = nullptr;
    std::vector<std::unique_ptr<SubmissionBatch>> all_batches;
    std::vector<SubmissionBatch*> free_batches; ///< guarded by recorder_mutex
    bool frame_end_submission = false;
    u32 submission_in_frame = 0; ///< BB_GPU_PROFILE: submissions since the frame's last one
    bool async_submit = false; ///< BB_ASYNC_SUBMIT (default on)
    std::atomic<u64> submitted_tick{0}; ///< the last tick handed to the queue (WaitSubmitted)
    u32 current_segment = 0;
    std::atomic<u32> active_worker{0}; ///< the worker of current_segment (it spins for work)
    size_t segment_bytes = 0;          ///< closures of the current segment handed over or retired
    size_t split_bytes = 0;            ///< segment length at which the stream is cut
    bool resume_rendering = false;     ///< a cut closed the render pass with render_state
    PassEndStack pass_end_stack{};     ///< BB_PASS_BREAK_TRACE: who ended the last render pass
    std::unique_ptr<RecordChunk> record_chunk;
    std::vector<std::unique_ptr<RecordChunk>> full_chunks;
    std::unique_ptr<RecordChunk> ordered_chunk;
    std::vector<std::unique_ptr<RecordChunk>> ordered_full;
    std::mutex recorder_mutex;
    std::condition_variable_any recorder_idle_cv;
    std::deque<std::unique_ptr<RecordChunk>> ordered_queue; ///< guarded by recorder_mutex
    std::atomic<size_t> ordered_queued{0};
    bool ordered_running = false; ///< a thread runs ordered chunks (guarded by recorder_mutex)
    std::vector<std::unique_ptr<RecordChunk>> free_chunks;
    bool direct_mode = false; ///< the command buffer is recorded on the caller's thread
    /// Direct_mode in a segment of its own (direct_pool), the recording threads going on
    /// with the segments before it.
    bool direct_segment = false;
    CarriedScope* carried_scope = nullptr;
    void CarrySuspend(bool inside_render_pass) {
        if (carried_scope) {
            carried_scope->Suspend(inside_render_pass);
        }
    }
    void CarryResume(bool inside_render_pass) {
        if (carried_scope) {
            carried_scope->Resume(inside_render_pass);
        }
    }
    std::unique_ptr<CommandPool> direct_pool; ///< used by the producer thread only
    u64 host_copies_issued = 0;
    std::atomic<u64> host_copies_done{0};
    /// Host copy sources by 64 KiB granule in a direct-mapped table: a slot is pending while its
    /// note is newer than the last completed WaitHostCopies; a slot two granules share is a
    /// wildcard (any granule mapping there counts as pending).
    struct HostCopySlot {
        u64 granule = 0;
        u64 seq = 0;
    };
    static constexpr u32 HostCopySlots = 4096;
    static constexpr u64 HostCopyWildcard = ~u64(0);
    std::mutex host_copy_sources_mutex;
    std::array<HostCopySlot, HostCopySlots> host_copy_slots{};
    u64 host_copy_source_seq = 0;
    u64 host_copy_sources_done = 0;  ///< notes up to this one are copied
    u64 host_copy_big_seq = 0;       ///< last note too large for the table (counts as everywhere)
    std::atomic<u64> deferred_signals_issued{0}; ///< by the thread recording (A or B)
    std::atomic<u32> producer_tid{0};                 ///< ProducerScope: the thread inside
    std::atomic<const char*> producer_where{nullptr}; ///< and where
    std::shared_ptr<std::atomic<u64>> deferred_signals_done =
        std::make_shared<std::atomic<u64>>(0);
};

} // namespace Vulkan
