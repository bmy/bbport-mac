// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: optimizations that can be switched off while the game runs (BB_TOGGLE_FILE,
// see runtime_memory.c), to find which one changes rendering without restarting.
#pragma once
#include <array>
#include "shim/bb_env.h"
#include <atomic>
#include <csetjmp>
#include <chrono>
#include <cstdint>
#include <cstdlib>

extern "C" std::uint64_t runtime_disabled_optimizations;
/// Temporary experiment bits: the second number in BB_TOGGLE_FILE (runtime_memory.c).
extern "C" std::uint64_t runtime_experiment_bits;
/// Recovery point for speculative guest memory reads on this thread (runtime_memory.c).
extern "C" __thread sigjmp_buf* runtime_fault_recover;

namespace BbToggle {
enum : std::uint64_t {
    RegionCache = 1,
    FetchShaderCache = 2,
    PageTrackingEarlyExit = 4,
    PendingPollLimit = 8,
    ThreadedRecording = 16,
    ImageDescCache = 32,
    LockFreeUploadCheck = 64,
    FindImageCache = 128,
    DeferredUploads = 256,
    AccessMemo = 512, ///< retired (the access tracker merges ranges itself)
    TextureBindingMemo = 1024,
    CoarseReadTracking = 2048,
    PreparedResources = 4096,
    DrawPreparation = 8192,
    DeferredStreamCopies = 16384,
    HotPages = 32768,
    FaultWindow = 65536,
    ParallelCopies = 131072,
    PoolSmallCopies = 524288,
    RecordPrefetch = 1ull << 32,
    TextureViewMemo = 1ull << 33,
    TextureBindHelper = 1ull << 34,
    EarlyDrawInputs = 1ull << 35,
    ConstantRing = 1ull << 36,
    DrawPipeline = 1ull << 37,
    PipelinedTasks = 1ull << 38,
    PendingFenceWaits = 1ull << 39,
    PipelinedDispatch = 1ull << 40,
    PipelinedMemoryWrites = 1ull << 42,
    MultiCopyShader = 1ull << 43,
    RenderStateMemo = 1ull << 44,
    TextureSetMemo = 1ull << 45,
    PipelinedIndirectDraws = 1ull << 46,
    SceneAttachmentsOnly = 1ull << 47,
    SampleSceneProxies = 1ull << 48,
    SceneHalfRes = 1ull << 50, ///< live scaling also reduces the 960x540 post targets
    /// A copy shader (IsComputeImageCopy) into memory that is no image yet makes the image there
    /// instead of running through memory; bit set: the shader runs.
    ComputeCopyNewImage = 1ull << 49,
    /// Such a copy of a reduced scene target copies its proxy (SceneTargets::CopyProxy); bit
    /// set: the native images, after a resolve.
    SceneProxyCopies = 1ull << 41,
    /// Clears of proxied scene images clear the proxy too instead of resolving it first.
    SceneClearProxies = 1ull << 20,
    UpdateImageFastPath = 1u << 30,
    /// Bit set: the texture collector frees nothing (A/B of its evictions while the game runs).
    TextureCollector = 1ull << 31,
    // TAA A/B in one run: optional techniques, off by default (no measured gain, 2026-10-02).
    TaaTonemapBlend = 1ull << 51,
    TaaClip = 1ull << 52,
    TaaVariance = 1ull << 53,
    TaaFilter = 1ull << 54,
    // On by default (bit set: off): history of a thin feature this jitter phase missed is kept
    // when nothing moves. Static-camera flicker of railings/window bars p99.9 -45% (2026-10-03).
    TaaKeepNearerHistory = 1ull << 55,
    /// FSR 4 (4.0 and 4.1.1) gets the motion vectors with y negated (motionVectorScale.y = -1); FSR 3
    /// and TAA keep them. A/B of the vertical convention FSR 4 expects.
    Fsr4MotionYFlip = 1ull << 56,
    SceneMipBias = 1ull << 57, ///< negative LOD bias of G-buffer samplers at reduced scene sizes
    /// The command stream is cut into segments recorded in parallel (BB_VK_RECORD_THREADS);
    /// bit set: no new cuts, one recording thread at a time.
    ParallelRecording = 1ull << 58,
    /// Finished GPU writes are read back on the readback queue (BufferCache::DownloadMemory);
    /// bit set: on the graphics queue after everything queued, as before.
    ReadbackQueue = 1ull << 59,
    /// Camera vectors from the inverse of the view matrix itself instead of the game's stored
    /// inverse; bit set: the other one of BB_CAMERA_INVERSE (A/B while the game runs).
    CameraOwnInverse = 1ull << 60,
    /// Scene textures use BB_ANISO (16) times anisotropic filtering instead of the game's
    /// ratio; bit set: the game's own samplers (A/B while the game runs).
    ForcedAniso = 1ull << 61,
    /// Camera motion vectors: the screen y sign derived from the G-buffer viewport, inverted
    /// (A/B of the y convention while the game runs).
    CameraYFlip = 1ull << 62,
    /// FSR 4 on the scaled presets gets the tonemapped frame decoded to linear light (its output
    /// encoded again); bit set: the encoded frame as before (A/B while the game runs).
    Fsr4EncodedInput = 1ull << 63,
    // Bits 20-29 are used as raw debug toggles by the camera/object motion and the upscaler.
};
inline bool Disabled(std::uint64_t bit) {
    return (__atomic_load_n(&runtime_disabled_optimizations, __ATOMIC_RELAXED) & bit) != 0;
}
/// A temporary experiment switched on by bit `bit` of the second number in BB_TOGGLE_FILE.
inline bool Experiment(std::uint64_t bit) {
    return (__atomic_load_n(&runtime_experiment_bits, __ATOMIC_RELAXED) & bit) != 0;
}
} // namespace BbToggle

namespace BbStats {
/// Guest writes caught by page protection, and pages currently left unprotected as hot.
inline std::atomic<std::uint64_t> tracker_faults{0};
/// Writes the GPU side heard of from the writers (libc copies over watched pages, no write tracking).
inline std::atomic<std::uint64_t> cpu_write_notes{0}, cpu_write_note_bytes{0};
inline std::atomic<std::uint64_t> fast_writer_faults{0}; ///< write faults of the game's bookkeeping, no pipeline wait
/// Ranges the game's GPU memory allocator handed out (guest hook).
inline std::atomic<std::uint64_t> gpu_range_allocs{0}, gpu_range_alloc_bytes{0};
/// Buffer bindings: bytes bound where the game's memory is (read over PCIe on a
/// discrete GPU) and bytes bound to VRAM copies.
inline std::atomic<std::uint64_t> bound_in_place_bytes{0}, bound_vram_bytes{0};
inline std::atomic<std::uint64_t> bound_in_place_written_bytes{0};
/// Vertex and index data of the draws, by where the GPU reads it: [0] the game's memory in place
/// (over the bus on a discrete GPU), [1] a VRAM copy or the stream buffer; vertex streams and
/// index buffers as bound (sizes of the draw's ranges, not unique bytes).
inline std::atomic<std::uint64_t> vertex_bytes[2]{}, index_bytes[2]{}, geometry_draws{0};
/// Command processor writes put into the GPU's VRAM data (no readback); blocks whose GPU data went
/// back to the game's memory before moving in place; downloads of GPU data (readbacks).
inline std::atomic<std::uint64_t> gpu_data_updates{0}, copied_back_blocks{0}, readbacks{0};
inline std::atomic<std::uint64_t> late_vram_writes{0}; ///< labels and the like put into VRAM copies
/// Copies of in-place data the CPU writes into VRAM for reading (ShadowCopy).
inline std::atomic<std::uint64_t> shadow_copies{0}, shadow_bytes{0};
/// Ranges a dispatch wrote in place copied into VRAM after it (ShadowGpuWrites), and bindings
/// served from those copies.
inline std::atomic<std::uint64_t> write_shadow_copies{0}, write_shadow_bytes{0}, write_shadow_hits{0},
    write_shadow_misses{0}; ///< small in-place reads with none to read
/// Submissions sent early because the GPU had finished everything before them.
inline std::atomic<std::uint64_t> idle_flushes{0};
inline std::atomic<std::uint64_t> drain_flushes{0}; ///< submissions sent as the GPU reached the last one
/// EOP fences with data: decoded, and their labels written (a growing gap: lost fences, guest leaks).
inline std::atomic<std::uint64_t> eop_decoded{0}, eop_written{0}, gpu_labels{0};
/// sceGnmAreSubmitsAllowed calls, and those that answered no (submission lock held).
inline std::atomic<std::uint64_t> submits_allowed_queries{0}, submits_refused{0}, submit_done_calls{0};
/// Guest time blocked in Gnm submissions on the previous frame (submission lock or BB_SUBMIT_LOCK=frame).
inline std::atomic<std::uint64_t> gnm_frame_waits{0}, gnm_frame_wait_ns{0};
/// Submissions to the compute queues (sceGnmDingDong).
inline std::atomic<std::uint64_t> asc_submits{0};
/// The game's submissions to any queue, numbered from 1 when made (Work::stamp), and the highest
/// one stage A started to decode: what the game did on its CPU in between (a heap block freed)
/// comes before every submission numbered higher.
inline std::atomic<std::uint64_t> guest_submit_stamp{0}, started_submit_stamp{0};
/// The game's GPU range allocators (seen at their collector, 0x26aa860): object, live ranges, bytes.
inline std::array<std::atomic<std::uint64_t>, 4> range_allocators{}, range_live{}, range_bytes{};
inline std::atomic<std::int64_t> hot_pages{0};
/// Stall diagnostics (BB_FRAME_STATS): per-frame deltas printed for frames over 40 ms.
inline std::atomic<std::uint64_t> images_registered{0};
inline std::atomic<std::uint64_t> image_upload_bytes{0};
inline std::atomic<std::uint64_t> buffer_upload_bytes{0};
inline std::atomic<int> gpu_thread_clock{-1}; ///< clockid_t of the GPU command thread
inline std::atomic<std::uint64_t> draws{0}, dispatches{0}, submissions{0};
/// Frames the GPU command thread has started (display pass), for per-frame diagnostics.
inline std::atomic<std::uint64_t> gpu_frames{0};
/// Wall time spent in operations suspected of stalls (ns, all threads).
inline std::atomic<std::uint64_t> t_resident{0}, t_protect{0}, t_image_create{0}, t_refresh{0},
    t_staging{0}, t_host_wait{0}, t_copy{0}, copy_bytes{0}, t_read_faults{0}, read_faults{0},
    t_write_faults{0}, t_copy_cpu{0}, copy_sys_us{0}, copy_minflt{0};
/// Layer memory module (BB_LAYER_MEMORY), wall time per kind (ns, inclusive): 0 mprotect for write
/// traps, 1 promotions, 2 demotions, 3 trap processing, 4 yielding VRAM, 5 volatile refreshes,
/// 6 republishing paged entries, 7 paged uploads; and the mprotect calls.
inline std::array<std::atomic<std::uint64_t>, 12> t_layer{};
/// Page table entries written (blocks); 8-11 of t_layer: their staging request, guest view, mirrors,
/// copy command.
inline std::atomic<std::uint64_t> layer_entries_written{0};
inline std::atomic<std::uint64_t> layer_mprotects{0};
/// Diagnostics are collected only with BB_FRAME_STATS=1.
inline const bool enabled = [] {
    return bb_env_on("BB_FRAME_STATS");
}();
struct Timer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start =
        enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    ~Timer() {
        if (!enabled) {
            return;
        }
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count(),
                        std::memory_order_relaxed);
    }
};
/// Wall time the GPU command thread waited for guest submissions (ns).
inline std::atomic<std::uint64_t> gpu_idle_ns{0};
/// Draws recorded into the reduced scene targets, and draws after the scene started.
inline std::atomic<std::uint64_t> reduced_draws{0}, scene_draws{0};
/// Wall time spent blocked in the scheduler (ns): waiting for the recording thread to drain,
/// for host copies before a submission or fence, and for GPU ticks.
inline std::atomic<std::uint64_t> sync_recording_ns{0}, host_copies_wait_ns{0}, tick_wait_ns{0},
    copy_threads_wait_ns{0}, host_copy_waits{0}, host_copy_waits_skipped{0},
    deferred_signals_wait_ns{0}, deferred_signal_waits{0};
/// Wall time the frame preparation waited for the GPU to finish an earlier frame (BB_FRAMES_AHEAD):
/// how GPU-bound the frames are (BB_FRAME_LOG).
inline std::atomic<std::uint64_t> present_wait_ns{0};
/// Wall time the frame preparation waited for a free presenter frame (all of them still being
/// presented: with frame generation the presenter keeps each for the generated frames too).
inline std::atomic<std::uint64_t> frame_pool_wait_ns{0};
/// Wall time in vkQueuePresentKHR (with the submission mutex held).
inline std::atomic<std::uint64_t> present_call_ns{0};
/// Frames shown (Presenter::PrepareFrame).
inline std::atomic<std::uint32_t> frame_number{0};
/// Device memory allocated (VMA blocks, arena residency) and freed, bytes.
inline std::atomic<std::uint64_t> device_alloc_bytes{0}, device_free_bytes{0};
/// Memory statistics: registered images (their guest size) and how many; the buffer cache's VRAM
/// for guest blocks (allocated, never returned to the driver), and the part of it unused (its
/// free list and the rest of the current 64 MiB block).
inline std::atomic<std::uint64_t> live_image_bytes{0}, live_images{0};
/// Every Vulkan image we allocated (their memory as allocated: host sizes, mips, scene targets).
inline std::atomic<std::uint64_t> vk_image_bytes{0};
/// Time inside vmaCreate*/vmaDestroy* of buffers and images (ns) and their calls, always on: a
/// device memory allocation or free on Windows (WDDM) can take tens of ms near the VRAM budget.
inline std::atomic<std::uint64_t> t_vk_alloc{0}, t_vk_free{0}, n_vk_alloc{0}, n_vk_free{0};
/// Time inside vkQueueSubmit2 (ns): WDDM pages evicted memory back in there.
inline std::atomic<std::uint64_t> t_vk_submit{0}, n_vk_submit{0};
/// Image creation split (ns): vkCreateImage, the memory, vkBindImageMemory.
inline std::atomic<std::uint64_t> t_img_create{0}, t_img_alloc{0}, t_img_bind{0};
/// Device memory made by kind (the Device memory lines): new mirrors (not from the pool), images.
inline std::atomic<std::uint64_t> mirror_new_bytes{0}, image_new_bytes{0}, staging_new_bytes{0};
/// New mirror bytes by where they were made: [0] promotion, [1] merge across, [2] compaction.
inline std::atomic<std::uint64_t> mirror_new_by[3]{};
inline thread_local int mirror_new_reason = 0;
struct VkMemTimer {
    std::atomic<std::uint64_t>& total;
    std::atomic<std::uint64_t>& count;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~VkMemTimer() {
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start)
                            .count(),
                        std::memory_order_relaxed);
        count.fetch_add(1, std::memory_order_relaxed);
    }
};
inline std::atomic<std::uint64_t> residency_alloc_bytes{0}, residency_unused_bytes{0};
/// The texture cache collector: the usage it compares, the mark it starts at, images it freed.
inline std::atomic<std::uint64_t> gc_used_bytes{0}, gc_trigger_bytes{0}, gc_freed_images{0};
/// Its critical mark (images used a moment ago evicted past it): VRAM copies of the game's
/// memory stop short of it and yield first (BufferCache::VramPromotionsPaused, LayerProcessIdle).
inline std::atomic<std::uint64_t> gc_critical_bytes{0};
/// Copies of loaded blocks wait for VRAM (BufferCache::VramPromotionsPaused with candidates): only
/// then the texture collector makes room for the mirrors under their floor.
inline std::atomic<bool> mirror_copies_waiting{false};
/// MiB the driver's VRAM budget for the game is below the most seen this session (other programs
/// took video memory: browsers with hardware acceleration, video players, overlays); the FPS
/// readout shows it from 150 MiB on.
inline std::atomic<std::uint32_t> vram_budget_lost_mb{0};
/// Seconds of a steady clock, updated at every guest submission (cheap ages for caches).
inline std::atomic<std::uint32_t> coarse_second{0};
/// VRAM blocks moved back in place when idle, unbound with the memory the game unmapped, and
/// residency chunks given back to the driver.
inline std::atomic<std::uint64_t> vram_idle_bytes{0}, vram_unmapped_bytes{0}, vram_chunks_freed_bytes{0};
/// Dynamic blocks the loaders wrote whole since: they may have a VRAM copy again (ProcessPendingAssets).
inline std::atomic<std::uint64_t> dynamic_reloaded_bytes{0};
/// GPU writes (binding sizes) kept in place because the CPU wrote their blocks (layer_cpu_blocks).
inline std::atomic<std::uint64_t> gpu_writes_kept_by_cpu_bytes{0};
/// Residency chunks, chunks sent away (ProcessIdleBlocks), and their blocks that had to stay.
inline std::atomic<std::uint64_t> residency_chunk_count{0}, evacuations{0}, evac_kept_gpu{0}, evac_kept_other{0};
/// Images the collector passed over (GPU-written, not evictable without memory pressure).
inline std::atomic<std::uint64_t> gc_kept_images{0};
struct WaitTimer {
    std::atomic<std::uint64_t>& total;
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    ~WaitTimer() {
        total.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - start).count(),
                        std::memory_order_relaxed);
    }
};
/// GPU thread rusage, refreshed after each graphics submission.
inline std::atomic<std::uint64_t> gpu_sys_us{0}, gpu_user_us{0}, gpu_invol_switches{0},
    gpu_vol_switches{0}, gpu_minor_faults{0};
/// Protection faults (signals) taken by the GPU thread itself.
inline std::atomic<std::uint64_t> gpu_signal_faults{0};
/// Protection changes: calls and pages, those removing write access (TLB shootdowns) apart.
inline std::atomic<std::uint64_t> protect_calls{0}, protect_pages{0}, protect_revoke_calls{0},
    protect_revoke_pages{0};
} // namespace BbStats
