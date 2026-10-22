// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/runtime.h"

#include <bit>
#include <algorithm>

#include <boost/container/small_vector.hpp>
#include <vulkan/vulkan_format_traits.hpp>

#include "bbport_toggles.h"
#include "render/gcn_to_vk.h"
#include "render/vk_gpu_profiler.h"
#include "textures/blitter.h"
#include "upscale/vk_scene_resolution.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"

namespace Vulkan {

namespace {
/// bbport: buffer copy regions per recorded command: their data and the command share a 128 KiB
/// recording chunk (Scheduler::ReserveRecordData keeps 1 KiB for the command).
constexpr size_t MaxRecordedBufferCopies = 4096;
static_assert(MaxRecordedBufferCopies * sizeof(vk::BufferCopy) + 1024 <= RecordChunk::Capacity);
} // namespace

namespace {

using Stage = vk::PipelineStageFlagBits2;
using AccessBit = vk::AccessFlagBits2;
using Layout = vk::ImageLayout;

constexpr vk::AccessFlags2 ReadAccesses =
    AccessBit::eIndexRead | AccessBit::eVertexAttributeRead | AccessBit::eUniformRead |
    AccessBit::eShaderRead | AccessBit::eColorAttachmentRead |
    AccessBit::eDepthStencilAttachmentRead | AccessBit::eTransferRead | AccessBit::eMemoryRead;
constexpr vk::AccessFlags2 WriteAccesses =
    AccessBit::eShaderWrite | AccessBit::eColorAttachmentWrite |
    AccessBit::eDepthStencilAttachmentWrite | AccessBit::eTransferWrite | AccessBit::eMemoryWrite |
    AccessBit::eTransformFeedbackWriteEXT;

/// The Vulkan image type of a GCN one (1D, 2D or 3D).
vk::ImageType DimensionOf(Gcn::ImageType type) noexcept {
    switch (type) {
    case Gcn::ImageType::Color1D:
    case Gcn::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case Gcn::ImageType::Color2D:
    case Gcn::ImageType::Color2DMsaa:
    case Gcn::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case Gcn::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE_MSG("image type {}", u32(type));
    }
}

/// Layer counts for a copy that Vulkan accepts: 3D images have one; same types copy the same
/// number; between 2D and 3D the 2D layers match the 3D depth.
std::pair<u32, u32> CopyLayers(const VideoCore::ImageInfo& src, const VideoCore::ImageInfo& dst,
                               u32 depth) {
    const auto src_type = DimensionOf(src.type);
    const auto dst_type = DimensionOf(dst.type);
    u32 src_layers = src_type == vk::ImageType::e3D ? 1 : src.resources.layers;
    u32 dst_layers = dst_type == vk::ImageType::e3D ? 1 : dst.resources.layers;
    if (src_type == dst_type) {
        src_layers = dst_layers = std::min(src_layers, dst_layers);
    } else if (src_type == vk::ImageType::e2D && dst_type == vk::ImageType::e3D) {
        src_layers = depth;
    } else if (src_type == vk::ImageType::e3D && dst_type == vk::ImageType::e2D) {
        dst_layers = depth;
    }
    return {src_layers, dst_layers};
}

/// The bytes of the buffer a buffer <-> image copy reaches (from its offset).
u32 BufferImageCopySize(const vk::BufferImageCopy& copy, vk::Format format) {
    const u32 row_length = copy.bufferRowLength ? copy.bufferRowLength : copy.imageExtent.width;
    const u32 height = copy.bufferImageHeight ? copy.bufferImageHeight : copy.imageExtent.height;
    const auto block = vk::blockExtent(format);
    const u32 block_size = vk::blockSize(format);
    const u32 row_pitch = (row_length / block[0]) * block_size;
    const u32 slice_pitch = (height / block[1]) * row_pitch;
    const u32 width_blocks = (copy.imageExtent.width + block[0] - 1) / block[0];
    const u32 height_blocks = (copy.imageExtent.height + block[1] - 1) / block[1];
    const u32 slices = copy.imageExtent.depth * copy.imageSubresource.layerCount;
    return (slices - 1) * slice_pitch + (height_blocks - 1) * row_pitch + width_blocks * block_size;
}

/// The written image now holds what the source holds (or new GPU data) and no stale upload.
void MarkCopied(VideoCore::Image* dst, VideoCore::ImageFlagBits modified) {
    dst->flags |= modified;
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

} // namespace

Runtime::Runtime(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_},
      blitter{std::make_unique<VideoCore::ImageBlitter>(instance_, scheduler_)},
      staging_pool{instance_, scheduler_} {}

Runtime::~Runtime() = default;

void Runtime::TickFrame() {
    staging_pool.TickFrame();
}

bool Runtime::IsBufferAccessed(const VideoCore::Buffer* buffer, u64 offset, u64 size,
                               bool check_read) {
    if (global_write) {
        return true; // a paged write may have touched it (AccessGlobal)
    }
    const u64 id = reinterpret_cast<u64>(buffer);
    const u64 last = offset + size - 1;
    return accesses.Overlaps(id, offset, last, AccessTracker::Kind::Write) ||
           (check_read && accesses.Overlaps(id, offset, last, AccessTracker::Kind::Read));
}

void Runtime::AccessBuffer(const VideoCore::Buffer* buffer, u64 offset, u64 size,
                           vk::PipelineStageFlags2 stage, vk::AccessFlags2 access) {
    const u64 id = reinterpret_cast<u64>(buffer);
    u64 first = offset;
    u64 last = offset + size - 1;
    if (access & WriteAccesses) {
        accesses.Insert(id, first, last, AccessTracker::Kind::Write);
    }
    if (access & ReadAccesses) {
        // Reads at 4 KiB granularity: ring allocations of neighbouring draws join into one
        // range. Wider read ranges only add barriers, never drop one.
        if (!BbToggle::Disabled(BbToggle::CoarseReadTracking)) {
            first &= ~u64{0xFFF};
            last |= 0xFFF;
        }
        accesses.Insert(id, first, last, AccessTracker::Kind::Read);
    }
    pending_stages |= stage;
    pending_writes |= access & WriteAccesses;
}

void Runtime::AccessGlobal(vk::PipelineStageFlags2 stage, vk::AccessFlags2 access) {
    pending_stages |= stage;
    if (access & AccessBit::eShaderWrite) {
        pending_writes |= AccessBit::eShaderWrite;
        global_write = true;
    }
}

void Runtime::FlushBarriers() {
    BeforeImageAccess();
    const bool memory = bool(pending_stages);
    if (!memory && image_barriers.empty()) {
        return;
    }
    scheduler.EndRendering();
    scheduler.Record([barrier = vk::MemoryBarrier2{
                          .srcStageMask = pending_stages,
                          .srcAccessMask = pending_writes,
                          .dstStageMask = Stage::eAllCommands,
                          .dstAccessMask = AccessBit::eMemoryRead | AccessBit::eMemoryWrite,
                      },
                      memory,
                      images = scheduler.RecordData(std::span<const vk::ImageMemoryBarrier2>(
                          image_barriers.data(), image_barriers.size()))](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2({
            .memoryBarrierCount = memory ? 1u : 0u,
            .pMemoryBarriers = &barrier,
            .imageMemoryBarrierCount = u32(images.size()),
            .pImageMemoryBarriers = images.data(),
        });
    });
    pending_stages = {};
    pending_writes = {};
    image_barriers.clear();
    accesses.Clear();
    global_write = false;
}

bool Runtime::Transit(VideoCore::Image* image, vk::ImageLayout dst_layout,
                      vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access,
                      std::optional<VideoCore::SubresourceRange> range,
                      std::source_location caller) {
    BeforeImageAccess();
    if (scene_targets) {
        scene_targets->NativeAccess(*image, dst_access, caller);
    }
    const size_t before = image_barriers.size();
    image->GetBarriers(image_barriers, dst_layout, dst_access, dst_stage, range);
    return image_barriers.size() != before;
}

Runtime::TransferMark::TransferMark(Runtime& runtime, const char* what,
                                    const VideoCore::Image& image) {
    profiler = GpuProfiler::Get();
    if (!profiler || !profiler->Records(&runtime.scheduler)) {
        profiler = nullptr;
        return;
    }
    resume = profiler->Current();
    const auto& info = image.info;
    const u64 key = (u64(reinterpret_cast<uintptr_t>(what)) << 20) ^
                    (u64(info.pixel_format) << 40) ^ (u64(info.size.width) << 16) ^
                    info.size.height ^ 0x7A5Full;
    profiler->Mark(key, [&] {
        return fmt::format("{} {} {}x{}", what, vk::to_string(info.pixel_format),
                           info.size.width, info.size.height);
    });
}

Runtime::TransferMark::~TransferMark() {
    if (profiler && resume) {
        profiler->Resume(resume);
    }
}

void Runtime::CopyBuffer(const VideoCore::Buffer* src, const VideoCore::Buffer* dst,
                         std::span<const vk::BufferCopy> copies) {
    scheduler.EndRendering();
    // Many regions (HLE copy shaders) are tracked as one bounding range per buffer:
    // conservative for barriers, and two lookups instead of two per region.
    const bool bounded = copies.size() > 4;
    u64 src_min = ~0ULL, src_max = 0, dst_min = ~0ULL, dst_max = 0;
    bool needs_flush = false;
    for (const auto& copy : copies) {
        if (bounded) {
            src_min = std::min<u64>(src_min, copy.srcOffset);
            src_max = std::max<u64>(src_max, copy.srcOffset + copy.size);
            dst_min = std::min<u64>(dst_min, copy.dstOffset);
            dst_max = std::max<u64>(dst_max, copy.dstOffset + copy.size);
        } else {
            needs_flush = needs_flush || IsBufferAccessed(src, copy.srcOffset, copy.size) ||
                          IsBufferAccessed(dst, copy.dstOffset, copy.size, true);
        }
    }
    if (bounded) {
        needs_flush = IsBufferAccessed(src, src_min, src_max - src_min) ||
                      IsBufferAccessed(dst, dst_min, dst_max - dst_min, true);
    }
    // BB_GPU_PROFILE: buffer copies get a mark of their own (by size class), as image transfers.
    GpuProfiler* profiler = GpuProfiler::Get();
    u64 resume = 0;
    if (profiler && profiler->Records(&scheduler)) {
        u64 total = 0;
        for (const auto& copy : copies) {
            total += copy.size;
        }
        const u32 log2 = total ? u32(std::bit_width(total) - 1) : 0;
        resume = profiler->Current();
        profiler->Mark(0xB0FC'0000ull + log2, [log2] {
            return fmt::format("buffer copy, {} KiB class", (1ull << log2) >> 10);
        });
    }
    if (needs_flush) {
        FlushBarriers();
    }
    // bbport: in groups that fit a recording chunk with their command (ReserveRecordData): a 4 MiB
    // pre-upload whose changed bytes lay in thousands of runs asserted there (code 23).
    for (size_t at = 0; at < copies.size(); at += MaxRecordedBufferCopies) {
        const auto part = copies.subspan(at, std::min(MaxRecordedBufferCopies, copies.size() - at));
        scheduler.Record([src = src->Handle(), dst = dst->Handle(),
                          regions = scheduler.RecordData(part)](vk::CommandBuffer cmdbuf) {
            cmdbuf.copyBuffer(src, dst, u32(regions.size()), regions.data());
        });
    }
    if (resume) {
        profiler->Resume(resume);
    }
    if (bounded) {
        AccessBuffer(src, src_min, src_max - src_min, Stage::eCopy, AccessBit::eTransferRead);
        AccessBuffer(dst, dst_min, dst_max - dst_min, Stage::eCopy, AccessBit::eTransferWrite);
        return;
    }
    for (const auto& copy : copies) {
        AccessBuffer(src, copy.srcOffset, copy.size, Stage::eCopy, AccessBit::eTransferRead);
        AccessBuffer(dst, copy.dstOffset, copy.size, Stage::eCopy, AccessBit::eTransferWrite);
    }
}

void Runtime::FillBuffer(const VideoCore::Buffer* dst, u64 offset, u64 size, u32 value) {
    scheduler.EndRendering();
    if (IsBufferAccessed(dst, offset, size, true)) {
        FlushBarriers();
    }
    scheduler.Record([dst = dst->Handle(), offset, size, value](vk::CommandBuffer cmdbuf) {
        cmdbuf.fillBuffer(dst, offset, size, value);
    });
    AccessBuffer(dst, offset, size, Stage::eClear, AccessBit::eTransferWrite);
}

void Runtime::CopyFromGuestChunk(vk::Buffer src, const VideoCore::Buffer* dst,
                                 std::span<const vk::BufferCopy> copies) {
    if (copies.empty()) {
        return;
    }
    scheduler.EndRendering();
    const bool needs_flush = std::ranges::any_of(copies, [&](const vk::BufferCopy& copy) {
        return IsBufferAccessed(dst, copy.dstOffset, copy.size, true);
    });
    if (needs_flush) {
        FlushBarriers();
    }
    for (size_t at = 0; at < copies.size(); at += MaxRecordedBufferCopies) {
        const auto part = copies.subspan(at, std::min(MaxRecordedBufferCopies, copies.size() - at));
        scheduler.Record([src, dst = dst->Handle(),
                          regions = scheduler.RecordData(part)](vk::CommandBuffer cmdbuf) {
            cmdbuf.copyBuffer(src, dst, u32(regions.size()), regions.data());
        });
    }
    for (const auto& copy : copies) {
        AccessBuffer(dst, copy.dstOffset, copy.size, Stage::eCopy, AccessBit::eTransferWrite);
    }
}

void Runtime::UpdateBuffer(const VideoCore::Buffer* dst, u64 offset, std::span<const u8> data) {
    if (data.empty()) {
        return;
    }
    scheduler.EndRendering();
    if (IsBufferAccessed(dst, offset, data.size(), true)) {
        FlushBarriers();
    }
    constexpr size_t Piece = 32 * 1024; // vkCmdUpdateBuffer takes 64 KiB at most
    for (size_t at = 0; at < data.size(); at += Piece) {
        const auto piece = data.subspan(at, std::min(Piece, data.size() - at));
        scheduler.Record([dst = dst->Handle(), to = offset + at,
                          bytes = scheduler.RecordData(piece)](vk::CommandBuffer cmdbuf) {
            cmdbuf.updateBuffer(dst, to, bytes.size_bytes(), bytes.data());
        });
    }
    AccessBuffer(dst, offset, data.size(), Stage::eClear, AccessBit::eTransferWrite);
}

void Runtime::UploadImage(VideoCore::Image* dst, const VideoCore::Buffer* src,
                          std::span<const vk::BufferImageCopy> copies) {
    SetBackingSamples(dst, dst->info.num_samples, false);
    scheduler.EndRendering();
    const TransferMark mark{*this, "image upload", *dst};
    bool needs_flush =
        Transit(dst, Layout::eTransferDstOptimal, Stage::eCopy, AccessBit::eTransferWrite);
    for (const auto& copy : copies) {
        needs_flush = needs_flush ||
                      IsBufferAccessed(src, copy.bufferOffset,
                                       BufferImageCopySize(copy, dst->info.pixel_format));
    }
    if (needs_flush) {
        FlushBarriers();
    }
    scheduler.Record([src = src->Handle(), image = dst->GetImage(),
                      regions = scheduler.RecordData(copies)](vk::CommandBuffer cmdbuf) {
        cmdbuf.copyBufferToImage(src, image, Layout::eTransferDstOptimal, u32(regions.size()),
                                 regions.data());
    });
    for (const auto& copy : copies) {
        AccessBuffer(src, copy.bufferOffset, BufferImageCopySize(copy, dst->info.pixel_format),
                     Stage::eCopy, AccessBit::eTransferRead);
    }
    dst->flags &= ~VideoCore::ImageFlagBits::Dirty;
}

void Runtime::DownloadImage(VideoCore::Image* src, const VideoCore::Buffer* dst,
                            std::span<const vk::BufferImageCopy> copies) {
    SetBackingSamples(src, src->info.num_samples);
    scheduler.EndRendering();
    const TransferMark mark{*this, "image download", *src};
    bool needs_flush =
        Transit(src, Layout::eTransferSrcOptimal, Stage::eCopy, AccessBit::eTransferRead);
    for (const auto& copy : copies) {
        needs_flush = needs_flush ||
                      IsBufferAccessed(dst, copy.bufferOffset,
                                       BufferImageCopySize(copy, src->info.pixel_format), true);
    }
    if (needs_flush) {
        FlushBarriers();
    }
    scheduler.Record([image = src->GetImage(), dst = dst->Handle(),
                      regions = scheduler.RecordData(copies)](vk::CommandBuffer cmdbuf) {
        cmdbuf.copyImageToBuffer(image, Layout::eTransferSrcOptimal, dst, u32(regions.size()),
                                 regions.data());
    });
    for (const auto& copy : copies) {
        AccessBuffer(dst, copy.bufferOffset, BufferImageCopySize(copy, src->info.pixel_format),
                     Stage::eCopy, AccessBit::eTransferWrite);
    }
}

void Runtime::PrepareTransfer(VideoCore::Image* src, VideoCore::Image* dst,
                              vk::PipelineStageFlags2 stage, bool extra_flush) {
    scheduler.EndRendering();
    bool needs_flush = Transit(src, Layout::eTransferSrcOptimal, stage, AccessBit::eTransferRead);
    needs_flush |= Transit(dst, Layout::eTransferDstOptimal, stage, AccessBit::eTransferWrite);
    if (needs_flush || extra_flush) {
        FlushBarriers();
    }
}

void Runtime::CopyImageRegions(VideoCore::Image* src, VideoCore::Image* dst,
                               std::span<const vk::ImageCopy> regions) {
    scheduler.Record([src = src->GetImage(), dst = dst->GetImage(),
                      regions = scheduler.RecordData(regions)](vk::CommandBuffer cmdbuf) {
        cmdbuf.copyImage(src, Layout::eTransferSrcOptimal, dst, Layout::eTransferDstOptimal,
                         u32(regions.size()), regions.data());
    });
}

void Runtime::CopyImage(VideoCore::Image* src, VideoCore::Image* dst) {
    if (src->info.pixel_format != dst->info.pixel_format) {
        LOG_DEBUG(Render_Vulkan, "image copy between formats {} and {}",
                  vk::to_string(src->info.pixel_format), vk::to_string(dst->info.pixel_format));
    }
    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);

    const auto src_type = DimensionOf(src->info.type);
    const auto dst_type = DimensionOf(dst->info.type);
    const vk::ImageAspectFlags src_aspect = src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    const vk::ImageAspectFlags dst_aspect = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    const u32 depth =
        dst->info.type == Gcn::ImageType::Color3D ? dst->info.size.depth : src->info.size.depth;
    const u32 levels = std::min(src->info.resources.levels, dst->info.resources.levels);
    boost::container::small_vector<vk::ImageCopy, 16> regions;
    for (u32 mip = 0; mip < levels; ++mip) {
        const u32 width = std::max(src->info.size.width >> mip, 1u);
        const u32 height = std::max(src->info.size.height >> mip, 1u);
        const u32 mip_depth = std::max(depth >> mip, 1u);
        auto [src_layers, dst_layers] = CopyLayers(src->info, dst->info, mip_depth);
        u32 extent_depth = 1;
        if (src_type == vk::ImageType::e2D && dst_type == vk::ImageType::e3D) {
            extent_depth = src_layers; // 2D layers into depth slices
        } else if (src_type == vk::ImageType::e3D && dst_type == vk::ImageType::e2D) {
            extent_depth = dst_layers;
        } else if (src_type == vk::ImageType::e3D) {
            src_layers = dst_layers = 1;
            extent_depth = mip_depth;
        } else {
            src_layers = dst_layers = std::min(src_layers, dst_layers);
        }
        regions.push_back({
            .srcSubresource = {src_aspect, mip, 0, src_layers},
            .dstSubresource = {dst_aspect, mip, 0, dst_layers},
            .extent = {width, height, extent_depth},
        });
    }
    PrepareTransfer(src, dst, Stage::eCopy);
    CopyImageRegions(src, dst, regions);
    MarkCopied(dst, src->flags & VideoCore::ImageFlagBits::GpuModified);
}

void Runtime::CopyImageThroughBuffer(VideoCore::Image* src, VideoCore::Image* dst,
                                     const VideoCore::Buffer* buffer, u64 offset) {
    ASSERT(src->info.resources.layers == dst->info.resources.layers &&
           std::min(src->info.resources.levels, dst->info.resources.levels) == 1);
    SetBackingSamples(dst, dst->info.num_samples, false);
    SetBackingSamples(src, src->info.num_samples);
    vk::BufferImageCopy download{
        .bufferOffset = offset,
        .imageSubresource = {src->aspect_mask & ~vk::ImageAspectFlagBits::eStencil, 0, 0,
                             src->info.resources.layers},
        .imageExtent = {src->info.size.width, src->info.size.height, src->info.size.depth},
    };
    const u32 size = BufferImageCopySize(download, src->info.pixel_format);
    PrepareTransfer(src, dst, Stage::eCopy, IsBufferAccessed(buffer, offset, size, true));
    vk::BufferImageCopy upload = download;
    upload.imageSubresource.aspectMask = dst->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
    scheduler.Record([src = src->GetImage(), dst = dst->GetImage(), buffer = buffer->Handle(),
                      download, upload](vk::CommandBuffer cmdbuf) {
        cmdbuf.copyImageToBuffer(src, Layout::eTransferSrcOptimal, buffer, download);
        const vk::MemoryBarrier2 written{
            .srcStageMask = Stage::eCopy,
            .srcAccessMask = AccessBit::eTransferWrite,
            .dstStageMask = Stage::eCopy,
            .dstAccessMask = AccessBit::eTransferRead,
        };
        cmdbuf.pipelineBarrier2({
            .dependencyFlags = vk::DependencyFlagBits::eByRegion,
            .memoryBarrierCount = 1,
            .pMemoryBarriers = &written,
        });
        cmdbuf.copyBufferToImage(buffer, dst, Layout::eTransferDstOptimal, upload);
    });
    AccessBuffer(buffer, offset, size, Stage::eCopy,
                 AccessBit::eTransferWrite | AccessBit::eTransferRead);
    MarkCopied(dst, src->flags & VideoCore::ImageFlagBits::GpuModified);
}

void Runtime::CopyMip(VideoCore::Image* src, VideoCore::Image* dst, u32 mip, u32 slice) {
    const u32 dst_shift = dst->info.props.is_block ? 2 : 0;
    const u32 src_shift = src->info.props.is_block ? 2 : 0;
    ASSERT(std::max(dst->info.size.width >> (mip + dst_shift), 1u) ==
           (src->info.size.width >> src_shift));
    ASSERT(std::max(dst->info.size.height >> (mip + dst_shift), 1u) ==
           (src->info.size.height >> src_shift));
    ASSERT(std::max(dst->info.mips_layout[mip].pitch >> dst_shift, 1u) ==
           (src->info.pitch >> src_shift));
    const auto [src_layers, dst_layers] = CopyLayers(src->info, dst->info, src->info.size.depth);
    const vk::ImageCopy region{
        .srcSubresource = {src->aspect_mask, 0, 0, src_layers},
        .dstSubresource = {src->aspect_mask, mip, slice, dst_layers},
        .extent = {src->info.size.width, src->info.size.height, src->info.size.depth},
    };
    SetBackingSamples(dst, dst->info.num_samples);
    SetBackingSamples(src, src->info.num_samples);
    PrepareTransfer(src, dst, Stage::eCopy);
    CopyImageRegions(src, dst, {&region, 1});
    MarkCopied(dst, src->flags & VideoCore::ImageFlagBits::GpuModified);
}

void Runtime::CopyColorAndDepth(VideoCore::Image* src, VideoCore::Image* dst) {
    if (src->info.num_samples == 1 && dst->info.num_samples == 1) {
        if (instance.IsMaintenance8Supported() ||
            src->info.props.is_depth == dst->info.props.is_depth) {
            CopyImage(src, dst);
            return;
        }
        // Depth and color images cannot be copied into each other without maintenance8.
        constexpr size_t CopyBufferSize = 128_MB;
        const auto staging = staging_pool.Request(CopyBufferSize, VideoCore::MemoryType::DeviceLocal);
        CopyImageThroughBuffer(src, dst, staging.buffer, staging.offset);
        return;
    }
    if (src->info.num_samples == 1 && dst->info.props.is_depth) {
        // A draw puts the source's channels into the samples of the destination.
        bool needs_flush = Transit(src, Layout::eShaderReadOnlyOptimal, Stage::eFragmentShader,
                                   AccessBit::eShaderRead);
        needs_flush |= Transit(dst, Layout::eDepthStencilAttachmentOptimal,
                               Stage::eEarlyFragmentTests | Stage::eLateFragmentTests,
                               AccessBit::eDepthStencilAttachmentWrite);
        if (needs_flush) {
            FlushBarriers();
        }
        blitter->ColorAsMsDepth(dst->info.size.width, dst->info.size.height,
                                dst->info.num_samples, src->info.pixel_format,
                                dst->info.pixel_format, src->GetImage(), dst->GetImage());
        return;
    }
    LOG_WARNING(Render_Vulkan, "copy between multisampled color and depth images: not done");
}

void Runtime::CopyDepthStencil(VideoCore::Image* src, VideoCore::Image* dst,
                               const VideoCore::SubresourceRange& range) {
    PrepareTransfer(src, dst, Stage::eCopy);
    const vk::ImageSubresourceLayers layers{src->aspect_mask & dst->aspect_mask, 0,
                                            range.base.layer, range.extent.layers};
    const vk::ImageCopy region{
        .srcSubresource = layers,
        .dstSubresource = layers,
        .extent = {dst->info.size.width, dst->info.size.height, 1},
    };
    CopyImageRegions(src, dst, {&region, 1});
    MarkCopied(dst, VideoCore::ImageFlagBits::GpuModified);
}

void Runtime::ResolveImage(VideoCore::Image* src, VideoCore::Image* dst,
                           const VideoCore::SubresourceRange& src_range,
                           const VideoCore::SubresourceRange& dst_range) {
    SetBackingSamples(dst, 1, false);
    const bool resolve = src->backing->num_samples != 1;
    PrepareTransfer(src, dst, resolve ? Stage::eResolve : Stage::eCopy);
    const auto [src_layers, dst_layers] = CopyLayers(src->info, dst->info, 1);
    const vk::ImageSubresourceLayers src_layers_info{vk::ImageAspectFlagBits::eColor, 0,
                                                     src_range.base.layer, src_layers};
    const vk::ImageSubresourceLayers dst_layers_info{vk::ImageAspectFlagBits::eColor, 0,
                                                     dst_range.base.layer, dst_layers};
    const vk::Extent3D extent{src->info.size.width, src->info.size.height, 1};
    if (resolve) {
        scheduler.Record([src = src->GetImage(), dst = dst->GetImage(),
                          region = vk::ImageResolve{.srcSubresource = src_layers_info,
                                                    .dstSubresource = dst_layers_info,
                                                    .extent = extent}](vk::CommandBuffer cmdbuf) {
            cmdbuf.resolveImage(src, Layout::eTransferSrcOptimal, dst, Layout::eTransferDstOptimal,
                                region);
        });
    } else {
        const vk::ImageCopy region{
            .srcSubresource = src_layers_info,
            .dstSubresource = dst_layers_info,
            .extent = extent,
        };
        CopyImageRegions(src, dst, {&region, 1});
    }
    MarkCopied(dst, VideoCore::ImageFlagBits::GpuModified);
}

void Runtime::ClearImage(VideoCore::Image* dst, const VideoCore::SubresourceRange& range,
                         const vk::ClearValue& value) {
    scheduler.EndRendering();
    if (scene_targets) {
        scene_targets->BeginClear(*dst, range);
    }
    if (Transit(dst, Layout::eTransferDstOptimal, Stage::eClear, AccessBit::eTransferWrite,
                range)) {
        FlushBarriers();
    }
    const vk::ImageSubresourceRange vk_range{vk::ImageAspectFlagBits::eColor, range.base.level,
                                             range.extent.levels, range.base.layer,
                                             range.extent.layers};
    scheduler.Record([image = dst->GetImage(), color = value.color,
                      vk_range](vk::CommandBuffer cmdbuf) {
        cmdbuf.clearColorImage(image, Layout::eTransferDstOptimal, color, vk_range);
    });
    if (scene_targets) {
        scene_targets->EndClear(*dst, range, value.color);
    }
    MarkCopied(dst, VideoCore::ImageFlagBits::GpuModified);
}

void Runtime::SetBackingSamples(VideoCore::Image* image, u32 num_samples, bool copy_backing) {
    BeforeImageAccess();
    auto& backing = image->backing;
    if (!backing || backing->num_samples == num_samples) {
        return;
    }
    const auto& info = image->info;
    ASSERT_MSG(!info.props.is_depth, "only color images change sample counts");
    VideoCore::Image::BackingImage* target;
    const auto it = std::ranges::find(image->backing_images, num_samples,
                                      &VideoCore::Image::BackingImage::num_samples);
    if (it != image->backing_images.end()) {
        target = std::addressof(*it);
    } else {
        auto create_info = backing->image.image_ci;
        create_info.samples = GcnToVk::NumSamples(num_samples, image->supported_samples);
        // bbport (macOS): multisampled backings in full usage (UpgradeUsage copies single-sample
        // backings only).
        const bool compressible = backing->compressible && num_samples <= 1;
        if (backing->compressible && !compressible) {
            create_info.flags |= image->full_create_flags;
            create_info.usage = image->full_usage;
        }
        target = &image->backing_images.emplace_back();
        target->num_samples = num_samples;
        target->compressible = compressible;
        target->image = VideoCore::HostImage{instance.GetDevice(), instance.GetAllocator()};
        target->image.Create(create_info);
        SetObjectName(instance.GetDevice(), target->image.image,
                      "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{} (backing)",
                      info.size.width, info.size.height, info.size.depth,
                      Gcn::NameOf(info.tile_mode), vk::to_string(info.pixel_format),
                      info.guest_address, info.guest_size, info.resources.layers,
                      info.resources.levels, num_samples);
    }

    if (copy_backing) {
        scheduler.EndRendering();
        ASSERT(info.resources.levels == 1 && info.resources.layers == 1);
        Transit(image, Layout::eShaderReadOnlyOptimal, Stage::eFragmentShader,
                AccessBit::eShaderRead);
        // The new backing's old contents do not matter.
        constexpr auto Stages = Stage::eColorAttachmentOutput;
        constexpr auto Accesses = AccessBit::eColorAttachmentRead | AccessBit::eColorAttachmentWrite;
        constexpr auto TargetLayout = Layout::eColorAttachmentOptimal;
        image_barriers.push_back({
            .srcStageMask = Stage::eAllCommands,
            .srcAccessMask = AccessBit::eMemoryRead | AccessBit::eMemoryWrite,
            .dstStageMask = Stages,
            .dstAccessMask = Accesses,
            .oldLayout = Layout::eUndefined,
            .newLayout = TargetLayout,
            .image = target->image,
            .subresourceRange = {image->aspect_mask, 0, 1, 0, info.resources.layers},
        });
        FlushBarriers();
        blitter->CopyMsImage(info.size.width, info.size.height, target->num_samples,
                             info.pixel_format, backing->num_samples > 1, backing->image,
                             target->image);
        target->state.layout = TargetLayout;
        target->state.access_mask = Accesses;
        target->state.pl_stage = Stages;
    }
    backing = target;
    image->usage_flags = target->image.image_ci.usage;
}

void Runtime::UpgradeUsage(VideoCore::Image* image, const char* why) {
    BeforeImageAccess();
    VideoCore::Image::BackingImage* old = image->backing;
    if (!old || !old->compressible) {
        return;
    }
    const auto& info = image->info;
    auto create_info = old->image.image_ci;
    create_info.flags |= image->full_create_flags;
    create_info.usage = image->full_usage;
    auto& next = image->backing_images.emplace_back();
    next.num_samples = old->num_samples;
    next.image = VideoCore::HostImage{instance.GetDevice(), instance.GetAllocator()};
    next.image.Create(create_info);
    SetObjectName(instance.GetDevice(), next.image.image,
                  "Image {}x{}x{} {} {} {:#x}:{:#x} L:{} M:{} S:{} (full usage)", info.size.width,
                  info.size.height, info.size.depth, Gcn::NameOf(info.tile_mode),
                  vk::to_string(info.pixel_format), info.guest_address, info.guest_size,
                  info.resources.layers, info.resources.levels, next.num_samples);

    // The contents, unless the old backing was never used (an image made for this binding).
    const bool used =
        old->state.layout != Layout::eUndefined || !old->subresource_states.empty();
    if (used) {
        scheduler.EndRendering();
        const TransferMark mark{*this, "image usage upgrade", *image};
        Transit(image, Layout::eTransferSrcOptimal, Stage::eCopy, AccessBit::eTransferRead);
        image_barriers.push_back({
            .srcStageMask = Stage::eNone,
            .srcAccessMask = AccessBit::eNone,
            .dstStageMask = Stage::eCopy,
            .dstAccessMask = AccessBit::eTransferWrite,
            .oldLayout = Layout::eUndefined,
            .newLayout = Layout::eTransferDstOptimal,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = next.image,
            .subresourceRange = {image->aspect_mask, 0, VK_REMAINING_MIP_LEVELS, 0,
                                 VK_REMAINING_ARRAY_LAYERS},
        });
        FlushBarriers();
        const auto& extent = create_info.extent;
        boost::container::small_vector<vk::ImageCopy, 16> regions;
        for (u32 level = 0; level < create_info.mipLevels; ++level) {
            const vk::ImageSubresourceLayers layers{image->aspect_mask, level, 0,
                                                    create_info.arrayLayers};
            regions.push_back({
                .srcSubresource = layers,
                .dstSubresource = layers,
                .extent = {std::max(extent.width >> level, 1u),
                           std::max(extent.height >> level, 1u),
                           std::max(extent.depth >> level, 1u)},
            });
        }
        scheduler.Record([src = old->image.image, dst = next.image.image,
                          regions = scheduler.RecordData(std::span<const vk::ImageCopy>(
                              regions.data(), regions.size()))](vk::CommandBuffer cmdbuf) {
            cmdbuf.copyImage(src, Layout::eTransferSrcOptimal, dst, Layout::eTransferDstOptimal,
                             u32(regions.size()), regions.data());
        });
        next.state = {Stage::eCopy, AccessBit::eTransferWrite, Layout::eTransferDstOptimal};
    }

    // The old backing stays in the list (pointers to it tell memos it changed) but is never
    // found again; its image and views go once the commands recorded so far are done.
    old->num_samples = 0;
    old->compressible = false;
    old->image_view_infos.clear();
    scheduler.DeferOperation([views = image->slot_image_views, ids = std::move(old->image_view_ids),
                              host = std::move(old->image)]() mutable {
        for (const VideoCore::ImageViewId id : ids) {
            views->erase(id);
        }
    });
    old->image_view_ids.clear();
    old->last_view = 0;
    image->backing = &next;
    image->usage_flags = create_info.usage;

    static std::atomic<u64> upgrades{0}, bytes{0};
    const u64 count = upgrades.fetch_add(1, std::memory_order_relaxed) + 1;
    const u64 total = bytes.fetch_add(used ? next.image.size_bytes : 0,
                                      std::memory_order_relaxed) +
                      (used ? next.image.size_bytes : 0);
    if (count <= 32 || count % 256 == 0) {
        std::printf("Texture usage upgrade %llu (%s): %ux%ux%u %s, %u levels, %u layers, %u "
                    "samples%s; %.1f MB copied in all\n",
                    (unsigned long long)count, why, info.size.width, info.size.height,
                    info.size.depth, vk::to_string(info.pixel_format).c_str(),
                    info.resources.levels, info.resources.layers, next.num_samples,
                    used ? "" : " (unused, not copied)", double(total) / 1e6);
    }
}

} // namespace Vulkan
