// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport vulkan: staging memory for uploads (HostUncached), readbacks (HostCached) and device-local
// scratch, per memory type:
//   - segments of 16 MiB handed out front to back. A segment remembers only the last submission
//     that used it; once the GPU is done with that one the whole segment is empty again. Nothing
//     is tracked per request.
//   - buffers of their own for requests over 16 MiB and for deferred ones (used past the
//     submission, until FreeDeferred), reused for requests of up to 1.5x less.
// Memory idle for ~30 s is given back; the upload segments keep BB_STAGING_KEEP_MB (512).
#pragma once

#include <array>
#include <memory>
#include <vector>

#include "buffers/buffer.h"
#include "common/types.h"

namespace Vulkan {

class Instance;
class Scheduler;

/// A piece of staging memory.
struct StagingBufferRef {
    VideoCore::Buffer* buffer{};
    u64 offset{};
    u64 size{};
    u8* mapped{}; ///< null for memory the host cannot see
    VideoCore::MemoryType type{};

    void Flush() const noexcept {
        buffer->Flush(offset, size);
    }
    void Invalidate() const noexcept {
        buffer->Invalidate(offset, size);
    }
};

class StagingBufferPool {
public:
    StagingBufferPool(const Instance& instance, Scheduler& scheduler);
    ~StagingBufferPool();
    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    /// `size` bytes of `type` memory, free for the current submission. `deferred`: a buffer of its
    /// own, held until FreeDeferred (its use outlives the submission).
    [[nodiscard]] StagingBufferRef Request(u64 size, VideoCore::MemoryType type, u64 alignment = 0,
                                           bool deferred = false);
    /// A deferred piece is done with: reusable once the current submission is.
    void FreeDeferred(const StagingBufferRef& ref);
    /// Once a frame: idle memory given back.
    void TickFrame();

private:
    static constexpr size_t NumTypes = 3; ///< DeviceLocal, HostUncached, HostCached

    struct Segment {
        std::unique_ptr<VideoCore::Buffer> buffer;
        u64 used = 0;      ///< bytes handed out since it was last empty
        u64 last_tick = 0; ///< the submission that used it last (0: none)
        u64 last_frame = 0;
        u64 last_ms = 0;   ///< wall time of the last use (VRAM ones go by it: frames are slow while loading)
    };
    struct Segments {
        std::vector<Segment> list;
        size_t current = 0; ///< the one requests are carved from
    };
    struct Whole {
        std::unique_ptr<VideoCore::Buffer> buffer;
        u64 last_tick = 0;
        u64 last_frame = 0;
        u64 last_ms = 0;
        bool held = false; ///< deferred, not freed yet
    };

    StagingBufferRef FromSegments(Segments& segments, u64 size, u64 alignment,
                                  VideoCore::MemoryType type);
    StagingBufferRef FromWhole(u64 size, VideoCore::MemoryType type, bool deferred);
    Segment& NewSegment(Segments& segments, VideoCore::MemoryType type);
    void TrimIdle(VideoCore::MemoryType type);
    void PublishStats() const;

    const Instance& instance;
    Scheduler& scheduler;
    std::array<Segments, NumTypes> segments;
    std::array<std::vector<Whole>, NumTypes> wholes;
    u64 keep_segments = 0; ///< upload segments never given back
    u64 frame = 0;
};

} // namespace Vulkan
