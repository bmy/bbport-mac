// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: MetalFX temporal upscaler (BB_UPSCALER=metalfx; macOS with KosmicKrisp only). The
// scene's color, depth and motion vectors are copied into buffers on exportable memory whose
// MTLHeaps (VK_EXT_external_memory_metal) MetalFX reads on KosmicKrisp's own MTLDevice
// (gpu/shim/bbport_metalfx.mm); its output comes back the same way. The GPU thread submits the
// copies and records the copy back. By default (BB_METALFX_EVENTS=2, needs this port's
// KosmicKrisp patch 0002) a thread of its own waits for the copies on the CPU, then hands
// MetalFX the work; the copy back waits for MetalFX on the GPU. BB_METALFX_EVENTS=1: GPU-side
// waits both ways through shared events (Metal's wait can miss Vulkan's signal: 4 s stalls).
// BB_METALFX_EVENTS=3 (a test; patch 0007, macOS 26): MetalFX runs on KosmicKrisp's own Metal 4
// queue, committed in submission order between the input copies and the copy back (no second
// queue, no event waits; but a MetalFX stall then holds up the whole queue, with no cutoff).
// BB_METALFX_EVENTS=0, or a driver without shared events: MetalFX runs on the submission thread
// right before the copy back is submitted (it waits for the copies, then for MetalFX).
// BB_METALFX_SYNC=1: the GPU thread finishes the frame so far and waits for MetalFX itself.
// Worker mode's stall cutoff (BB_METALFX_STALL_MS, default 1000, 0: off): MetalFX work not done
// that long after it was handed over is given up, the frames waiting for it go on, and FSR 3.1
// takes over (the temporal upscaler may try MetalFX again later). BB_METALFX_TEST_STALL=N: a test
// that makes MetalFX's GPU work stall for 10 s in each of its first N runs.
// Elsewhere it reports itself unsupported.

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace BbMetalFx {
class Scaler;
struct Frame;
}

namespace Vulkan {

class Instance;
class Scheduler;

class MetalFxUpscaler {
public:
    MetalFxUpscaler(const Instance& instance, Scheduler& scheduler);
    ~MetalFxUpscaler();

    /// MetalFX can run: macOS, KosmicKrisp's MTLHeap export, MetalFX on that MTLDevice.
    [[nodiscard]] bool Supported() const noexcept {
        return supported;
    }
    /// Why it cannot run (now or since a failure), or null.
    [[nodiscard]] const char* Problem() const noexcept {
        return problem.empty() ? nullptr : problem.c_str();
    }
    /// The last failure will repeat (unsupported size, scaler creation): stop selecting it.
    [[nodiscard]] bool Fatal() const noexcept {
        return fatal;
    }
    /// After a menu change: a supported MetalFX may be tried again.
    void Retry() noexcept {
        if (supported) {
            fatal = false;
            transient = false;
            problem.clear();
            test_frames = 0;
        }
    }
    /// The last failure happened at run time (its GPU work failed or stalled) and need not
    /// repeat, unlike an unsupported size or a scaler that cannot be created.
    [[nodiscard]] bool Transient() const noexcept {
        return transient;
    }
    /// No stalled or failed frame is still on MetalFX's queue: a retry can start now.
    [[nodiscard]] bool Ready() const;

    struct Input {
        vk::Image image; ///< in General layout, TransferSrc usage
        vk::Format format;
        vk::ImageAspectFlagBits aspect;
    };
    struct Frame {
        Input color; ///< RGBA16F (HDR) or RGBA8
        Input depth; ///< D32 or D32S8 (depth aspect)
        Input motion; ///< RG16F, pixels, previous minus current position
        u32 width, height; ///< render size (the inputs' top-left)
        vk::Image output; ///< RGBA16F, output size; in General layout afterwards
        u32 out_width, out_height;
        bool hdr;
        std::array<float, 2> jitter; ///< render pixels, the FSR 3 convention
        bool reset;
    };
    /// Submits the frame so far (and waits for it when synchronous), upscales, and records the
    /// copy of the result into `output` in a new command buffer, submitted when asynchronous:
    /// callers must fetch CommandBuffer() again.
    bool Run(const Frame& frame);

private:
    struct Staging {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
        vk::DeviceSize size = 0;
        void* heap = nullptr; ///< id<MTLHeap>, owned by the memory
    };
    bool CreateStaging(Staging& staging, vk::DeviceSize size);
    void DestroyStaging(Staging& staging);
    bool EnsureResources(const Frame& frame);
    void Fail(std::string reason, bool permanent, bool at_run_time = false);

    const Instance& instance;
    Scheduler& scheduler;
    bool supported = false;
    bool fatal = false;
    bool transient = false;
    /// A failure at run time: the next run gets a new scaler and with it a new command queue
    /// (macOS ignores a queue's later work after one of its command buffers caused a GPU error:
    /// "Ignored (for causing prior/excessive GPU errors)", run D, 2026-10-24).
    bool rebuild_after_failure = false;
    std::string problem;
    void* get_metal_handle = nullptr; ///< PFN_vkGetMemoryMetalHandleEXT
    void SetUpEvents();
    /// Sync: the GPU thread waits for the GPU and MetalFX. SubmitThread (BB_METALFX_EVENTS=0, or
    /// no shared events): MetalFX runs right before the copy back is submitted, on the submission
    /// thread. Events (BB_METALFX_EVENTS=1): GPU-side waits through shared events. Worker
    /// (default): a thread of its own waits for the input copies on the CPU and hands MetalFX its
    /// work then; the copy back waits for MetalFX on the GPU.
    /// Queue (BB_METALFX_EVENTS=3, needs this port's KosmicKrisp patch 0007, macOS 26): MetalFX
    /// is committed to KosmicKrisp's own Metal 4 queue right before the copy back is submitted,
    /// so the queue's order and barriers sequence it; no event waits, no second queue.
    enum class Mode { Sync, SubmitThread, Events, Worker, Queue };
    struct Job; ///< a frame for the worker (vk_metalfx.cpp)
    struct Pending {
        u64 value;       ///< the done value its copy back waits for
        u64 copies_tick; ///< its input copies (work timeline)
        u64 issued_ns;
        u64 handed_ns = 0; ///< Worker: when it went to Metal (0: not yet)
    };
    void WorkerLoop(std::stop_token stop);
    /// Until the worker has handed every queued frame to MetalFX.
    void DrainWorker();
    void WatchLoop(std::stop_token stop);
    std::mutex job_mutex;
    std::condition_variable_any job_cv;
    std::deque<std::unique_ptr<Job>> jobs; ///< Job is defined in the .cpp (Mac-only types)
    bool job_running = false;
    std::jthread worker;
    /// BB_METALFX_EVENTS: a frame MetalFX has not finished after 0.5 s is reported (where each
    /// side got to), once per frame.
    std::jthread watch;
    std::mutex pending_mutex;
    std::deque<Pending> pending; ///< frames whose copy back waits, oldest first
    std::atomic<u64> encoded_value{0}; ///< Worker: the last frame handed to Metal
    /// The stall cutoff (BB_METALFX_STALL_MS, worker mode): a frame MetalFX has not finished that
    /// long after the worker handed it over is given up. Its copy back, and every later frame's,
    /// is let go from the CPU (`released_value`); the worker hands Metal nothing more, and FSR 3.1
    /// takes over. Cleared once MetalFX's queue is empty again (Run, after a retry).
    std::atomic<bool> stalled{false};
    std::atomic<u64> released_value{0};
    /// Let go of every frame issued so far (raises `done_event`, never lowers it). With
    /// `only_if_stalled`, nothing once `stalled` is clear (checked under pending_mutex, where Run
    /// clears it), so a frame issued after a retry is never let go.
    void ReleaseIssued(bool only_if_stalled);
    /// The first frame of a new scaler: its cutoff is longer (MetalFX's first frame has been slow).
    std::atomic<u64> first_value{0};
    /// BB_METALFX_TEST_STALL=N: the 600th frame of each of the first N MetalFX runs stalls.
    u32 test_stalls_left = 0;
    u32 test_frames = 0;
    Mode mode = Mode::Sync;
    std::mutex deferred_mutex;
    std::string deferred_error; ///< a submission-thread MetalFX failure, reported next frame
    /// Events: the scheduler's timeline and MetalFX's own, as id<MTLSharedEvent>.
    bool async = false;
    vk::Semaphore done_semaphore{}; ///< timeline signalled by Metal (via `done_event`)
    void* done_event = nullptr;
    void* work_event = nullptr;
    void* driver_queue = nullptr; ///< Queue mode: KosmicKrisp's id<MTL4CommandQueue>
    u64 done_value = 0;
    std::array<Staging, 4> staging{}; ///< color, depth, motion, output
    std::unique_ptr<BbMetalFx::Scaler> scaler;
    /// scaler->Encode, timed for a new scaler's first frame (the log says how long it took).
    bool Encode(const BbMetalFx::Frame& metal, std::string& error);
    std::atomic<bool> first_encode{false};
    u32 width = 0, height = 0, out_width = 0, out_height = 0;
    vk::Format color_format = vk::Format::eUndefined;
    bool hdr = false;
};

} // namespace Vulkan
