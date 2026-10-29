// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, DMA (fills, copies, GDS reads) and the guest memory events the
// caches follow: CPU writes, write faults, readbacks, GPU mappings.
//
// DMA into memory the GPU uses in place, or whose newest data is in a VRAM copy, is done by the
// GPU in stream order: on the CPU now it would land before earlier draws read
// the old data. Elsewhere (unbound memory: shader code, descriptors the command processor reads)
// the CPU does it in decode order.

#include "render/raster_internal.h"

namespace Vulkan {

namespace {
using BufferSlice = std::pair<const VideoCore::Buffer*, u64>;
} // namespace

/// Whether the GPU's view of [address, address + size) is not the game's memory alone: used in
/// place, written by the GPU, or mirrored in VRAM (BB_LAYER_MEMORY).
static bool GpuHolds(VideoCore::BufferCache& cache, VAddr address, u64 size) {
    return cache.IsAnyInPlace(address, size) || cache.IsRegionGpuModified(address, size) ||
           VideoCore::BufferCache::LayerMirrored(address, size);
}

void Rasterizer::FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds) {
    DrainDrawPipe();
    ASSERT_MSG(address % 4 == 0 && num_bytes % 4 == 0,
               "FillBuffer address and size must be a multiple of 4 bytes");
    if (!is_gds) {
        const bool meta = texture_cache.ClearMeta(address);
        if (FrameCapture::Active()) {
            char note[128];
            std::snprintf(note, sizeof(note), "\n  fill %#llx size %u value %#x meta %u",
                          (unsigned long long)address, num_bytes, value, u32(meta));
            FrameCapture::Note(note);
        }
        if (!GpuHolds(buffer_cache, address, num_bytes)) {
            BbFreeCheck::Check(address, num_bytes, &value, BbFreeCheck::DmaFill);
            u32* words = std::bit_cast<u32*>(address);
            std::fill(words, words + num_bytes / sizeof(u32), value);
            BbWriteLog::Note(address, words, num_bytes, BbWriteLog::Dma);
            NoteAssetWrite(address, num_bytes);
            return;
        }
    }
    const auto [buffer, offset] = is_gds ? BufferSlice{buffer_cache.GetGdsBuffer(), address}
                                         : buffer_cache.ObtainBuffer(address, num_bytes, true);
    runtime.FillBuffer(buffer, offset, num_bytes, value);
}

bool Rasterizer::DmaMayWriteOnCpu(VAddr dst, u32 num_bytes) {
    // FillBuffer's and CopyBuffer's test.
    return !GpuHolds(buffer_cache, dst, num_bytes);
}

void Rasterizer::CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds) {
    DrainDrawPipe();
    // On the CPU when neither side is the GPU's: the source not newer in VRAM (BB_LAYER_MEMORY)
    // nor an image's memory.
    if (!dst_gds && !src_gds && !GpuHolds(buffer_cache, dst, num_bytes) &&
        !buffer_cache.IsRegionGpuModified(src, num_bytes) &&
        !VideoCore::BufferCache::LayerMirrored(src, num_bytes) &&
        !texture_cache.FindImageFromRange(src, num_bytes)) {
        BbFreeCheck::Check(dst, num_bytes, std::bit_cast<const void*>(src), BbFreeCheck::DmaCopy);
        std::memcpy(std::bit_cast<void*>(dst), std::bit_cast<void*>(src), num_bytes);
        BbWriteLog::Note(dst, std::bit_cast<const void*>(dst), num_bytes, BbWriteLog::Dma);
        NoteAssetWrite(dst, num_bytes);
        return;
    }
    texture_cache.InvalidateMemoryFromGPU(dst, num_bytes);
    const auto* gds = buffer_cache.GetGdsBuffer();
    const auto [src_buffer, src_offset] =
        src_gds ? BufferSlice{gds, src} : buffer_cache.ObtainBuffer(src, num_bytes, false, true);
    const auto [dst_buffer, dst_offset] =
        dst_gds ? BufferSlice{gds, dst} : buffer_cache.ObtainBuffer(dst, num_bytes, true, true);
    const vk::BufferCopy copy{.srcOffset = src_offset, .dstOffset = dst_offset, .size = num_bytes};
    runtime.CopyBuffer(src_buffer, dst_buffer, std::span{&copy, 1});
}

bool Rasterizer::InvalidateMemory(VAddr addr, u64 size, bool assume_locks) {
    DrainDrawPipe();
    if (!IsMapped(addr, size)) {
        return false; // the GPU does not use it
    }
    buffer_cache.InvalidateMemory(addr, size, assume_locks);
    texture_cache.InvalidateMemory(addr, size);
    return true;
}

void Rasterizer::NoteAssetWrite(VAddr addr, u64 size) {
    buffer_cache.NoteAssetWrite(addr, size);
    InvalidateAfterWrite(addr, size);
}

void Rasterizer::InvalidateAfterWrite(VAddr addr, u64 size) {
    if (!buffer_cache.IsRegionGpuModified(addr, size)) {
        InvalidateMemory(addr, size);
        return;
    }
    // GPU data beside the bytes the CPU wrote lives in VRAM: an invalidation would read it back
    // (a 512 KiB window, whole pages) and upload pages over what the GPU writes meanwhile.
    // Exactly the CPU's bytes go into VRAM instead; images over them are refreshed.
    DrainDrawPipe();
    buffer_cache.NotePreciseUpload(addr, size);
    texture_cache.InvalidateMemory(addr, size);
}

void Rasterizer::NoteCommandWrite(VAddr addr, const void* data, u64 size, bool recording_thread) {
    if (!IsMapped(addr, size)) {
        return; // the GPU does not use it
    }
    if (!recording_thread) {
        DrainDrawPipe();
    }
    buffer_cache.DropShadows(addr, size);
    // An invalidation would read the GPU's data back over these bytes first (a readback).
    if (!buffer_cache.UpdateGpuWritten(addr, std::span{static_cast<const u8*>(data), size})) {
        buffer_cache.InvalidateMemory(addr, size, recording_thread);
    }
    texture_cache.InvalidateMemory(addr, size);
}

void Rasterizer::OnCpuWrite(VAddr addr, u64 size) {
    BbStats::cpu_write_notes.fetch_add(1, std::memory_order_relaxed);
    BbStats::cpu_write_note_bytes.fetch_add(size, std::memory_order_relaxed);
    if (IsMapped(addr, size)) {
        InvalidateAfterWrite(addr, size);
        buffer_cache.NoteCpuWriteRange(addr, size);
    }
}

bool Rasterizer::OnWriteFault(VAddr addr, bool assume_locks, u64 guest_rip) {
    DrainDrawPipe();
    if (!InvalidateMemory(addr, 8, assume_locks)) {
        return false;
    }
    buffer_cache.ExtendWriteFault(addr, guest_rip);
    return true;
}

bool Rasterizer::ReadMemory(VAddr addr, u64 size, bool assume_locks) {
    DrainDrawPipe();
    if (!IsMapped(addr, size)) {
        return false; // the GPU does not use it
    }
    buffer_cache.ReadMemory(addr, size, false, assume_locks);
    return true;
}

void Rasterizer::ProcessDownloadImages() {
    DrainDrawPipe();
    texture_cache.ProcessDownloadImages();
}

bool Rasterizer::IsMapped(VAddr addr, u64 size) {
    // Empty ranges and ranges wrapping the address space are never mapped.
    if (size == 0 || addr > std::numeric_limits<u64>::max() - size) {
        return false;
    }
    const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    Common::RecursiveSharedLock lock{mapped_ranges_mutex};
    return boost::icl::contains(mapped_ranges, range);
}

void Rasterizer::MapMemory(VAddr addr, u64 size) {
    {
        std::scoped_lock lock{mapped_ranges_mutex};
        mapped_ranges += decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
    }
}

void Rasterizer::UnmapMemory(VAddr addr, u64 size) {
    buffer_cache.UnmapInPlace(addr, size);
    buffer_cache.DropTwins(addr, size);
    DrainDrawPipe();
    buffer_cache.InvalidateMemory(addr, size);
    texture_cache.UnmapMemory(addr, size);
    std::scoped_lock lock{mapped_ranges_mutex};
    mapped_ranges -= decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
}

} // namespace Vulkan
