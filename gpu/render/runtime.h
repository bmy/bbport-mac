// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: transfers between buffers and images recorded on the draw scheduler, with the
// barriers they need. Buffer accesses since the last barrier are tracked by range (a transfer
// over a range written before, or writing one read before, flushes a barrier first); image
// layout changes are collected from the images' own state and flushed together.
#pragma once

#include <source_location>

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "buffers/buffer.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "textures/image.h"
#include "textures/types.h"
#include "vulkan/access_tracker.h"
#include "vulkan/staging_pool.h"

namespace VideoCore {
class ImageBlitter;
}

namespace Vulkan {

class GpuProfiler;
class Instance;
class Scheduler;
class SceneTargets;

class Runtime {
public:
    Runtime(const Instance& instance, Scheduler& scheduler);
    ~Runtime();

    Scheduler& GetScheduler() {
        return scheduler;
    }
    const Instance& GetInstance() const {
        return instance;
    }
    StagingBufferPool& GetStagingPool() {
        return staging_pool;
    }
    void TickFrame();

    /// The upscaler's scene targets: told about native accesses of the images they stand for.
    SceneTargets* scene_targets = nullptr;

    void CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                    std::span<const vk::BufferCopy> copies);
    void FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value);
    /// Copies from the game's memory itself (a guest memory chunk's buffer, written by the CPU)
    /// into `dst`: uploads without a CPU copy (BB_GUEST_IN_PLACE).
    void CopyFromGuestChunk(vk::Buffer src, const VideoCore::Buffer* dst,
                            std::span<const vk::BufferCopy> copies);
    /// `data` written into `dst` by the GPU in stream order (vkCmdUpdateBuffer pieces).
    void UpdateBuffer(const VideoCore::Buffer* dst, u64 offset, std::span<const u8> data);

    /// Queues `image`'s transition for an access (true if a barrier was added).
    bool Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                 vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                 std::optional<VideoCore::SubresourceRange> range = {},
                 std::source_location caller = std::source_location::current());

    /// BB_GPU_PROFILE: a profiler segment for a transfer of `image` while it lives; the
    /// segment it interrupted continues afterwards.
    class TransferMark {
    public:
        TransferMark(Runtime& runtime, const char* what, const VideoCore::Image& image);
        ~TransferMark();
        TransferMark(const TransferMark&) = delete;
        TransferMark& operator=(const TransferMark&) = delete;

    private:
        GpuProfiler* profiler = nullptr;
        u64 resume = 0;
    };

    void UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                     std::span<const vk::BufferImageCopy> copies);
    void DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                       std::span<const vk::BufferImageCopy> copies);
    /// All common levels (and layers or depth slices) of `src` into `dst`.
    void CopyImage(VideoCore::Image* src, VideoCore::Image* dst);
    /// `src` (one level) into level `mip`, layer `slice` of `dst`.
    void CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice);
    /// Between color and depth images (one of them may be multisampled depth).
    void CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst);
    void CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                          const VideoCore::SubresourceRange& range);
    void ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                      const VideoCore::SubresourceRange& src_range,
                      const VideoCore::SubresourceRange& dst_range);
    void ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                    const vk::ClearValue& value);

    /// Switches `image` to its backing with `num_samples` (made on first use), copying the
    /// contents over when `copy_backing`.
    void SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing = true);

    /// bbport (macOS): replaces `image`'s compressible backing (Image::NeedsFullUsage) with one
    /// of its full usage, the contents copied over; the old one and its views go once the GPU
    /// is done with them. `why` is for the log. Nothing to do for other backings.
    /// `feedback`: for a feedback loop, with the feedback loop usage (else without it, unless
    /// the backing had it); also upgrades a full backing that lacks it.
    void UpgradeUsage(VideoCore::Image* image, const char* why, bool feedback = false);

    /// bbport: `release` frees a replaced backing or proxy (UpgradeUsage, UpgradeProxy). The
    /// draw or dispatch being bound may still hold its views: it runs once the GPU is done
    /// with the commands recorded up to the next ReleaseRetired (after that draw, from
    /// Rasterizer::ResetBindings, and at each frame).
    void Retire(Common::UniqueFunction<void>&& release) {
        retired.push_back(std::move(release));
    }
    void ReleaseRetired();

    /// Records an access of [offset, offset + size) of `buffer` by commands just recorded.
    void AccessBuffer(const VideoCore::Buffer* buffer, u64 offset, u64 size,
                      vk::PipelineStageFlags2 stage, vk::AccessFlags2 access);
    /// Whether a barrier must come before reading (or with `check_read`, writing) the range.
    bool IsBufferAccessed(const VideoCore::Buffer* buffer, u64 offset, u64 size,
                          bool check_read = false);
    /// BB_LAYER_MEMORY: a shader reached memory through the page table (any buffer of the
    /// game's memory). A write there makes every later access wait for a barrier.
    void AccessGlobal(vk::PipelineStageFlags2 stage, vk::AccessFlags2 access);

    /// Records the queued barriers (memory and image layouts).
    void FlushBarriers();

    /// Runs before this thread changes image state (layouts, pending image barriers): the
    /// rasterizer joins its texture binding helper there (BindHelper).
    void SetImageAccessHook(void (*hook)(void*), void* context) {
        image_access_hook = hook;
        image_access_context = context;
    }
    void BeforeImageAccess() {
        if (image_access_hook) {
            image_access_hook(image_access_context);
        }
    }

private:
    /// Records the copies between two images in transfer layouts (after their transitions).
    void CopyImageRegions(VideoCore::Image* src, VideoCore::Image* dst,
                          std::span<const vk::ImageCopy> regions);
    /// Through a buffer, for depth <-> color copies without maintenance8.
    void CopyImageThroughBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                                const VideoCore::Buffer* buffer, u64 offset);
    /// The transitions for a transfer from `src` into `dst`, flushed if any.
    void PrepareTransfer(VideoCore::Image* src, VideoCore::Image* dst,
                         vk::PipelineStageFlags2 stage, bool extra_flush = false);

    void (*image_access_hook)(void*) = nullptr;
    void* image_access_context = nullptr;
    const Instance& instance;
    Scheduler& scheduler;
    std::unique_ptr<VideoCore::ImageBlitter> blitter;
    StagingBufferPool staging_pool;
    AccessTracker accesses;
    VideoCore::Image::Barriers image_barriers;
    vk::PipelineStageFlags2 pending_stages{};  ///< of accesses since the last barrier
    vk::AccessFlags2 pending_writes{};         ///< their write accesses
    bool global_write = false;                 ///< AccessGlobal wrote since the last barrier
    std::vector<Common::UniqueFunction<void>> retired; ///< Retire, until ReleaseRetired
};

} // namespace Vulkan
