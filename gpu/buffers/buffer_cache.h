// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport buffers: the buffer cache: guest memory bound for the GPU. Ranges lie in sparse arenas
// (one per 4 GiB of the address space, blocks bound on demand: VRAM copies uploaded from the
// guest, or the game's own memory in place), in the layer's memory module (chunk buffers and VRAM
// mirrors), or are copied per draw into the stream buffer (small constants). See
// buffer_cache.cpp and layer/buffer_cache_layer.cpp.

#pragma once

namespace BbCp {
class CommandProcessor;
}
namespace BbMem {
class PageWatch;
class DirtyTracker;
} // namespace BbMem

#include "bbport_copy.h"

#include <atomic>
#include <deque>
#include <future>
#include <map>
#include <mutex>
#include <array>
#include <memory>
#include <optional>
#include <source_location>
#include <unordered_map>
#include <boost/container/small_vector.hpp>

#include "common/interval_set.h"
#include "common/types.h"
#include "buffers/buffer.h"
#include "buffers/fault_buffer.h"
#include "buffers/range_set.h"
#include "vulkan/timeline.h"


namespace Core {
class MemoryManager;
}

namespace BbLayer {
struct MemorySource;
}

namespace Vulkan {
class GraphicsPipeline;
struct SubmitInfo;
class Runtime;
class StagingBufferPool;
} // namespace Vulkan

namespace VideoCore {

class TextureCache;



/// bbport: an IntervalList whose every change advances `generation`: lookups of the buffer
/// cache's residency remember their answers for as long as nothing changed (ResidencyMemo).
template <class IV = Interval>
class TrackedIntervalList : public IntervalList<IV> {
public:
    void Add(IV value) {
        ++generation;
        IntervalList<IV>::Add(value);
    }
    void Subtract(u64 start, u64 end) {
        ++generation;
        IntervalList<IV>::Subtract(start, end);
    }
    void Clear() {
        ++generation;
        IntervalList<IV>::Clear();
    }
    static inline u64 generation = 1;
};

class BufferCache {
    static constexpr u64 ADDRESS_SPACE_BITS = 40;
    static constexpr u64 ARENA_PAGE_BITS = 32;
    static constexpr u64 ARENA_PAGE_SIZE = u64{1} << ARENA_PAGE_BITS;
    static constexpr u64 NUM_ARENA_PAGES = u64{1} << (ADDRESS_SPACE_BITS - ARENA_PAGE_BITS);
    static constexpr u64 MIN_BLOCK_SIZE = 16_KB;

public:
    /// Read-only bindings up to this size are copied into a stream buffer (bbport: public for
    /// the draw pipeline's constant ring).
    static constexpr u64 STREAM_THRESHOLD = 16_KB;

    explicit BufferCache(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
                         Vulkan::Runtime& runtime, BbCp::CommandProcessor* liverpool,
                         TextureCache& texture_cache, BbMem::PageWatch& tracker);
    ~BufferCache();

    /// Returns a pointer to GDS device local buffer.
    [[nodiscard]] const Buffer* GetGdsBuffer() const noexcept {
        return &gds_buffer;
    }

    /// Retrieves the device local DBA page table buffer.
    /// The mirrors' share of the driver's VRAM budget the texture collector's mark does not take
    /// (the collector lowers its marks while they hold less).
    static u64 LayerMirrorFloor(u64 budget);
    /// At most this much of the floor is kept free for the mirrors (LayerMirrorFloor).
    static u64 LayerMirrorHeadroom(u64 budget);
    [[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept {
        return bda_pagetable_buffer.get();
    }

    /// Retrieves the fault buffer.
    [[nodiscard]] Buffer* GetFaultBuffer() noexcept {
        return fault_manager->GetFaultBuffer();
    }

    /// Retrieves the stream buffer.
    StreamBuffer& GetStreamBuffer() noexcept {
        return stream_buffer;
    }

    /// Returns minimum granularity of a sparse memory bind.
    u32 GetSparsePageShift() const noexcept {
        return block_shift;
    }

    /// Invalidates any buffer in the logical page range.
    void InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks = false);

    /// bbport: a guest write fault at `device_addr` was handled; unprotect its neighbourhood.
    void ExtendWriteFault(VAddr device_addr, u64 guest_rip = 0);

    /// Flushes any GPU modified buffer in the logical page range back to CPU memory.
    void ReadMemory(VAddr device_addr, u64 size, bool is_write = false, bool assume_locks = false);

    /// BB_LAYER_PROBE (diagnostics): a CPU access to a probed page, counted; the trap goes.
    bool LayerProbeHit(VAddr address, u64 rip, bool write, bool gpu_thread);
    /// Who hit a read trap: a guest (or runtime) thread, the GPU command thread (stage A, the
    /// draw pipe drained) or the draw recording thread (stage B: it may not record or flush).
    enum class TrapCaller { Guest, CommandThread, RecordingThread };
    /// bbport BB_LAYER_READ_TRAPS: a CPU access hit a page whose newest data the GPU wrote into a
    /// VRAM copy only: the block's GPU data is copied back into the game's memory first (a guest's
    /// access on the GPU command thread). False: not such a page.
    bool LayerReadTrapHit(VAddr address, TrapCaller caller);

    /// bbport: a new draw or dispatch is being recorded: the per-binding housekeeping of
    /// EnsureResident (unmaps, demotions, assets, late writes) runs once for it, not per binding.
    void NewPacket() noexcept {
        ++packet_epoch;
    }
    /// bbport: GPU writes of the bindings resolved now stay in the game's memory (no VRAM copy).
    bool force_writes_in_place = false;
    /// bbport: with force_writes_in_place, the bytes the bindings now resolved really write (the copy
    /// shader's control list, CopyShaderDestinations: address, size); null: all of each binding.
    const std::vector<std::pair<unsigned long long, unsigned long long>>* precise_writes = nullptr;
    /// The game's copy shader (its control list, CopyShaderRanges): for a paged binding at one of
    /// these bases only the blocks of its ranges get page table entries (and imports), not the
    /// GiB it binds. Set around its binding, like precise_writes.
    struct PagedRanges {
        u64 base = 0;
        const std::vector<std::pair<unsigned long long, unsigned long long>>* ranges = nullptr;
    };
    std::array<PagedRanges, 2> paged_ranges{};

    /// Finds a buffer for the specified region.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBuffer(VAddr device_addr, u32 size,
                                                             bool is_written,
                                                             bool is_texel_buffer = false) {
        const auto result = ObtainBufferImpl(device_addr, size, is_written, is_texel_buffer);
        if (in_place_read_hook && result.first != &stream_buffer) {
            in_place_read_hook(in_place_read_context, device_addr, size);
        }
        return result;
    }
    /// Called for every binding the GPU will read where it is (not a stream copy): the rasterizer
    /// notes them so that a constant RAM dump over them waits for the GPU, and only those.
    void (*in_place_read_hook)(void*, VAddr, u64) = nullptr;
    void* in_place_read_context = nullptr;
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferImpl(VAddr device_addr, u32 size,
                                                                 bool is_written,
                                                                 bool is_texel_buffer);

    /// Attempts to obtain a buffer without modifying the cache contents.
    [[nodiscard]] std::pair<const Buffer*, u64> ObtainBufferForImage(VAddr device_addr, u32 size);

    /// bbport: where a command processor write (WRITE_DATA) the GPU performs in
    /// stream order lands: the arena over the game's memory, if the range is bound there in place.
    /// Not counted as a GPU write of the block (those move blocks to VRAM): it is CPU data, as a PC
    /// game's constant updates. The range counts as GPU-modified until the GPU has done it, so no
    /// copy of it is taken before (IsRegionGpuModified).
    [[nodiscard]] std::optional<std::pair<const Buffer*, u64>> CommandWriteTarget(VAddr device_addr,
                                                                                u32 size);
    /// bbport BB_LAYER_MEMORY: the mirror holding all of a command write's range, written along
    /// with the game's memory (the range keeps its VRAM copy); empty: none, it goes in place.
    [[nodiscard]] std::optional<std::pair<const Buffer*, u64>> CommandWriteMirror(VAddr device_addr,
                                                                                u32 size);

    /// Return true when a region is modified from the CPU
    [[nodiscard]] bool IsRegionCpuModified(VAddr addr, size_t size);

    /// Return true when a region is modified from the GPU
    [[nodiscard]] bool IsRegionGpuModified(VAddr addr, size_t size);
    /// Diagnostics (BB_TRACE_CB): which check of IsRegionGpuModified holds: 1 the page tracker,
    /// 2 bytes the GPU wrote into a VRAM copy, 3 a GPU write not done yet, 0 none.
    [[nodiscard]] u32 GpuModifiedWhy(VAddr addr, size_t size);

    /// Processes the fault buffer.
    void ProcessFaultBuffer();

    /// Synchronizes all buffers needed for DMA.
    void SynchronizeDmaBuffers();

    /// bbport (BB_GPU_WRITE_TWINS): forgets the twins of unmapped guest memory.
    void DropTwins(VAddr addr, u64 size);

    /// Commits pending sparse buffer memory binds. Must be called before every scheduler submit.
    void SubmitPendingArenaBinds(Vulkan::SubmitInfo& info);
    /// bbport (GPU watchdog): prints how far the GPU got with the arena binds (vkQueueBindSparse).
    void ReportArenaBinds();

    /// bbport: the game unmapped [addr, addr + size) (a guest thread): arena
    /// blocks bound to its memory there are unbound before the next binding (GPU side).
    void UnmapInPlace(VAddr addr, u64 size);
    /// A game allocation [addr, addr + size) the game's profile knows (a heap block): VRAM copies
    /// of data in it stay within it (one mirror per such domain, no merge across its edges).
    void LayerDomainAdd(VAddr addr, u64 size);
    /// The game freed it: its domain goes, and its whole blocks' VRAM copies with it (queued as an
    /// unmap; the blocks it shares with neighbours stay).
    void LayerDomainDead(VAddr addr, u64 size);
    /// bbport: drops the images of ranges LayerDomainDead queued. Not under the texture cache's
    /// mutex (Rasterizer::OnSubmit; Maintain runs inside texture lookups).
    /// `started_stamp`: the highest submission stage A had started when this was queued; a range
    /// freed before it (an image there may be the new allocation's already) counts as late.
    void ProcessDeadImages(u64 started_stamp);
    [[nodiscard]] bool DeadImagesPending() const noexcept {
        return dead_images_pending.load(std::memory_order_acquire);
    }
    /// Whether the range is bound to the game's own memory (no uploads, tracking or readbacks).
    [[nodiscard]] bool IsInPlace(VAddr addr, u64 size) const;
    /// bbport: [addr, addr + size) gets data through a path
    /// the GPU side hears of (file reads, the game's resource loaders, DMA): its blocks may then
    /// keep a VRAM copy (any thread).
    void NoteAssetWrite(VAddr addr, u64 size);
    /// bbport: the game's GPU memory allocator handed out [addr, addr + size) again (any thread): its
    /// blocks lose the asset mark and are bound in place, since what writes them next may be code
    /// the GPU side does not hear of.
    void NoteFreshRange(VAddr addr, u64 size);
    /// bbport, without write tracking: the command processor wrote `data` at addr in decode order.
    /// Where the GPU's data there lives in VRAM (GPU-modified, not in place) the same bytes are
    /// written into it in stream order and true is returned; else the caller invalidates.
    bool UpdateGpuWritten(VAddr addr, std::span<const u8> data);
    /// bbport: copies of [addr, addr + size) in VRAM (ShadowCopy) are stale from here on in stream
    /// order: the GPU or the command processor wrote there (GPU side thread).
    void DropShadows(VAddr addr, u64 size,
                     std::source_location caller = std::source_location::current());
    /// bbport BB_LAYER_MEMORY (GPU side thread, right after a dispatch, outside any render pass):
    /// the small ranges it wrote in the game's memory are copied by the GPU into VRAM, and later
    /// small read-only bindings of them in this submission read that copy instead of the game's
    /// memory over PCIe (constants the game's copy shader writes: every wave of a G-buffer pass
    /// read them across the bus, the GPU busy at low power). One copy after the writer, no render
    /// pass cut. Off by default (no gain measured on AMD); BB_WRITE_SHADOWS=1 or experiment bit
    /// 512: on.
    void ShadowGpuWrites(std::span<const std::pair<unsigned long long, unsigned long long>> ranges);
    /// A GPU write of unknown range (a paged binding): every ShadowGpuWrites copy may be stale.
    void DropWriteShadows();
    /// bbport, without write tracking (any thread): the command processor wrote these bytes into the
    /// game's memory outside the command stream (labels, timestamps, occlusion results; through
    /// the backing mapping, which nothing sees). Where VRAM holds a copy of them it gets them too, at
    /// the next binding.
    void NoteLateCommandWrite(VAddr addr, const void* data, u64 size);
    /// bbport, without write tracking (any thread): the CPU wrote [addr, addr + size) next to data
    /// the GPU keeps in VRAM. Exactly those bytes are copied into VRAM from the game's memory at the
    /// next binding: no readback of the GPU's data, nothing uploaded over it.
    void NotePreciseUpload(VAddr addr, u64 size);
    /// bbport: the CPU wrote [addr, addr + size) where the GPU side asked to hear of it (no write
    /// tracking): counted per block like a write fault, for moving per-frame data in place.
    void NoteCpuWriteRange(VAddr addr, u64 size);
    /// BB_CHUNK_TEXTURES: texture data read by the GPU straight from the guest memory chunk it is
    /// in (no CPU copy, no VRAM buffer copy), when the range is contiguous there. Also the target
    /// of labels the GPU writes (Rasterizer::WriteLabelOnGpu). GPU side threads.
    std::optional<std::pair<const Buffer*, u64>> GuestChunkSource(VAddr address, u64 size);
    /// bbport BB_LAYER_MEMORY: whether the layer's memory module binds the game's memory.
    static bool LayerActive();
    /// bbport BB_LAYER_MEMORY: [addr, addr + size) touches a block with a VRAM copy (it may hold the
    /// GPU's newest data): bound through the module, never copied from the game's memory.
    [[nodiscard]] static bool LayerMirrored(VAddr addr, u64 size);
    /// Paged bindings in use (BB_LAYER_PAGED=0: off); GPU-written data in VRAM (BB_LAYER_GPU_WRITES).
    static bool LayerPagedActive();
    /// BB_LAYER_GPU_WRITES: where GPU writes may land in VRAM copies: 0 nowhere (in place), 1 (the
    /// default) in blocks the CPU has not written since they were handed out (layer_cpu_blocks:
    /// render targets and other GPU-only data), 2 anywhere.
    static int LayerGpuWritesMode();
    bool LayerGpuWriteToVram(VAddr address, u64 size);
    bool LayerInImage(VAddr address, u64 size);
    /// A paged binding (Shader::IsPagedBuffer): the record its shader reads ({address, size,
    /// trash address}, see EmitContext::PagedPointer), with page table entries for the range.
    std::pair<const Buffer*, u64> LayerPagedRecord(VAddr address, u64 size, bool is_written);
    /// The CPU wrote [address, address + size): VRAM copies there are refreshed before paged
    /// bindings read them.
    void LayerNoteCpuWrite(VAddr address, u64 size);
    /// bbport BB_LAYER_MEMORY: [address, address + size) in place as a range of its chunk's buffer,
    /// when the layer's memory module finds it whole in one chunk.
    /// clamp_huge: a binding of 64 MiB or more that runs on past its chunk is cut at the end of
    /// the part in place (the descriptor range is clamped to the buffer).
    std::optional<std::pair<const Buffer*, u64>> LayerInPlace(VAddr address, u64 size,
                                                              bool clamp_huge = false);
    std::optional<std::pair<const Buffer*, u64>> LayerInPlaceUncached(VAddr address, u64 size,
                                                                      bool clamp_huge);
    /// BB_TRACE_SHADER: what a binding got: the game's memory (a chunk buffer), a copy in the
    /// stream buffer, or another buffer (VRAM mirrors, the arena).
    [[nodiscard]] const char* BoundPlace(const Buffer* buffer) const {
        if (buffer == &stream_buffer) {
            return "stream copy";
        }
        for (const auto& chunk : chunk_buffers) {
            if (chunk.get() == buffer) {
                return "game memory";
            }
        }
        return "other (VRAM)";
    }
    /// Whether any part of the range is.
    [[nodiscard]] bool IsAnyInPlace(VAddr addr, u64 size) const {
        if (LayerActive() && size != 0) {
            return !IsInVramLayer(addr, size); // bbport BB_LAYER_MEMORY: not all in one mirror
        }
        return size != 0 &&
               in_place_blocks.Overlaps(addr >> block_shift, ((addr + size - 1) >> block_shift) + 1);
    }
    /// bbport BB_LAYER_MEMORY: [addr, addr + size) whole in one VRAM mirror.
    [[nodiscard]] bool IsInVramLayer(VAddr addr, u64 size) const;

private:
    struct ArenaBinds {
        const Buffer* arena;
        boost::container::small_vector<vk::SparseMemoryBind, 32> binds;
    };

    ArenaBinds* BindsForArena(const Buffer* arena) {
        auto it = std::ranges::find(pending_binds, arena, &ArenaBinds::arena);
        if (it != pending_binds.end()) {
            return std::addressof(*it);
        }
        return &pending_binds.emplace_back(arena);
    }

    const Buffer* GetArena(u64 first_block, u64 last_block);

    /// `in_place`: bind what can be in place even when Garlic (texture data, read once into an
    /// image: a VRAM copy of it would only cost an upload).
    void EnsureResident(const Buffer* arena, u64 first_block, u64 last_block,
                        bool in_place = false);
    /// Binds the blocks of [start, end) that are direct memory in a guest memory
    /// chunk to that memory; the others are added to `rest` (VRAM, as before).
    struct ResidentBind {
        u64 start, end; ///< blocks
        vk::DeviceMemory memory;
        u64 offset;
    };
    void BindInPlace(u64 start, u64 end, std::vector<ResidentBind>& out, IntervalList<>& rest,
                     bool any_type = false);
    /// bbport: the housekeeping EnsureResident did before every binding (Process*), once a packet.
    void Maintain();
    u64 packet_epoch = 1, maintained_epoch = 0;
    /// bbport: answers of EnsureResident ("all resident") and IsInPlace for block ranges, valid
    /// while resident_ranges and in_place_blocks keep their generation.
    struct ResidencyMemo {
        u64 first = ~0ull, last = 0, generation = 0;
        bool resident = false, in_place = false;
    };
    std::array<ResidencyMemo, 256> residency_memo{};
    static u64 ResidencyGeneration() noexcept {
        return TrackedIntervalList<Backing>::generation + TrackedIntervalList<>::generation;
    }
    void ProcessPendingUnmaps();
    /// bbport: a write fault (guest thread). Blocks the CPU keeps writing (faults in
    /// 3 frames of 60) move from VRAM to the game's memory in place: dynamic data, like an upload
    /// heap; static data stays in VRAM, like a default heap.
    void NoteCpuWrite(VAddr address, u64 guest_rip);
    /// GPU side: moves the requested blocks once the submission that may still upload into their
    /// VRAM copy was submitted; the rebind waits for it on the GPU (bind_wait_tick).
    void ProcessDemotions();
    u64 demotions_scanned_tick = ~0ULL;
    void ProcessPendingAssets();
    void ProcessLateWrites();
    /// BB_FRAME_STATS: GPU-written bindings over 64 KiB, reported every 5 s (where they are bound,
    /// which images lie over them).
    void NoteLargeGpuWrite(VAddr addr, u64 size);
    /// bbport on a discrete GPU: BB_SHADOWS=1 (off by default: 85 copies, 20 MB a
    /// frame in Yahar'gul cost more than reading in place, 48 -> 44 FPS): a read-only binding of data the
    /// CPU writes in place is copied once per submission by the GPU into VRAM and bound there, as a
    /// PC game copies an upload heap into a default heap: one sequential copy over PCIe instead of
    /// every shader reading it across the bus.
    std::optional<std::pair<const Buffer*, u64>> ShadowCopy(VAddr addr, u64 size);
    struct Shadow {
        VAddr addr;
        u64 size, offset;
    };
public:
    /// Vertex data bound in place (the CPU writes it often: ~26 MB a frame in heavy views, read by
    /// the depth, G-buffer and shadow passes over the bus each): copied by the CPU into the stream
    /// buffer once per guest submission and bound there (no copy command: no pass break).
    /// BB_HOT_VERTEX_COPY=1 or experiment bit 1024.
    std::optional<std::pair<const Buffer*, u64>> HotVertexCopy(VAddr addr, u64 size);
    static bool HotVertexCopyOn();
    /// BB_VERTEX_READ_COPY=1 (experiment bit 2048 inverts it): of such a range [addr, addr+size)
    /// only [lo, hi) is copied, the bytes the draw's vertex and instance range reads (~1/10 of
    /// the bound bytes); the result is where addr would be (the rest there is never read).
    std::optional<std::pair<const Buffer*, u64>> HotVertexCopySpan(VAddr addr, u64 size, u64 lo,
                                                                   u64 hi);
    static bool VertexReadCopyOn();
    void ClearHotCopies() {
        hot_copies.clear();
    }
private:
    struct HotCopy {
        VAddr addr;
        u64 size, offset;
    };
    std::vector<HotCopy> hot_copies; ///< this guest submission's copies
    std::unique_ptr<StreamBuffer> shadow_buffer;
    std::vector<Shadow> shadows; ///< this submission's copies
    u64 shadow_tick = 0;
    /// ShadowGpuWrites: the VRAM copy (in shadow_buffer) holding [addr, addr + size).
    std::optional<std::pair<const Buffer*, u64>> GpuWriteShadow(VAddr addr, u64 size);
    struct WriteShadow {
        u64 size, offset;
        u64 stamp; ///< write_shadow_reserved when made: valid until the ring comes round
    };
    std::map<VAddr, WriteShadow> write_shadows; ///< by address, never overlapping
    u64 write_shadow_reserved = 0;              ///< bytes ShadowGpuWrites took from the ring
    /// BB_TRACE_SIZE=bytes (diagnostics): bindings over GPU-written buffers of that size, counted by
    /// the path they take, reported every 5 s.
    void TraceBinding(VAddr addr, u64 size, bool is_written, int kind);
    std::vector<std::pair<VAddr, u64>> traced_ranges;
    std::array<u64, 10> trace_counts{};
    u64 trace_texel = 0, trace_min = ~0ull, trace_max = 0;
    std::chrono::steady_clock::time_point trace_report{};
    struct LargeWrite {
        VAddr addr = 0;
        u64 size = 0, count = 0;
        u64 fresh = 0; ///< ranges handed out again over it in the window
        bool in_place = false;
        std::string images;
    };
    std::unordered_map<u64, LargeWrite> large_writes;
    std::chrono::steady_clock::time_point large_writes_report{};
    /// Without write tracking: [addr, addr + size) is bound for GPU writes (GPU side thread).
    void NoteGpuWrite(VAddr addr, u64 size);
    /// Whether a block may keep a VRAM copy without write tracking: Garlic data only loaders wrote.
    bool VramEligible(u64 block);
    /// Called as a submission goes out (its tick `submitted`): candidate blocks get VRAM for the next
    /// submission, bound once `submitted` is done on the GPU, filled from the game's memory there.
    /// Called as a submission goes out: blocks partly handed out again whose GPU data is in VRAM get
    /// that data copied back into the game's memory at its end, and are bound in place for the
    /// next submission (once `submitted` is done).
    void QueueCopyBacks(u64 submitted);
    void QueuePromotions(u64 submitted);
    void ReportInPlaceBlocks(); ///< bbport BB_RESIDENCY_REPORT (diagnostics)
    /// Arena residency memory from the 64 MiB blocks: the memory and the byte offset in it.
    std::pair<vk::DeviceMemory, u64> AllocateResidency(u64 bytes);
    /// bbport: BbStats::residency_unused_bytes after the free list or the current block changed.
    void NoteResidencyUnused();
    /// bbport: idle VRAM blocks (see group_use).
    void NoteUse(VAddr address, u64 size);
    /// bbport BB_RESIDENCY_MIX=1 (diagnostics): bindings over blocks in VRAM and in place at once.
    void NoteResidencyMix(u64 first_block, u64 last_block, u64 size, bool is_written);
    void ProcessIdleBlocks();
    /// A VRAM slot of a block that left it: reused by the next move to VRAM.
    void ReleaseResidencySlot(vk::DeviceMemory memory, u64 offset);
    /// Unbinds a VRAM block of memory the game unmapped (uploaded again if mapped and used).
    void EvictVramBlock(u64 block);
    /// bbport: chunks without blocks for 5 s, their rebinds long done: back to the driver.
    void FreeEmptyChunks(u32 now);
    [[nodiscard]] static u32 VramIdleSeconds();
    void DownloadMemory(const Buffer* arena, VAddr device_addr, u64 size);

    bool SynchronizeMemory(const Buffer* arena, VAddr device_addr, u32 size, bool is_written,
                           bool is_texel_buffer);

    /// bbport: batched small copies on the copy threads (BbCopy::QueueCopy).
    static void RunGuestCopy(const BbCopy::Item& item);
    void SmallGuestCopy(const BbCopy::Item& item);

    const Buffer* UploadCopies(const Buffer* arena, std::span<vk::BufferCopy> copies,
                               size_t total_size_bytes);

    /// The image at device_addr, if any, tiled into `buffer` at `offset` (where the range is).
    /// `vram_copy`: where the shader may read the whole range instead, when the image was tiled
    /// in VRAM on its way to host memory (BB_TILE_BIND_VRAM; else left alone).
    bool SynchronizeMemoryFromImage(const Buffer* buffer, u64 offset, VAddr device_addr, u32 size,
                                    std::pair<const Buffer*, u64>* vram_copy = nullptr);
    /// bbport BB_LAYER_MEMORY (statistics): a binding not whole in one chunk.
    void NoteLayerMiss(VAddr address, u64 size, bool is_written);
    /// bbport BB_LAYER_MEMORY (buffer_cache_layer.cpp): a binding through the layer's memory
    /// module: a mirror in VRAM, or in place (empty: neither; the arena takes it). may_promote
    /// false (image uploads): a mirror only where the range is in VRAM already, none made for it.
    /// bbport: reading bindings LayerBind resolved to a whole mirror or in place, reused while
    /// nothing moved (layer_generation), the CPU marked nothing modified anywhere
    /// (BbMem::cpu_change_epoch: no upload can be due) and within the same second (the idle
    /// tracking's use marks are per second). BB_BIND_MEMO=0: every binding takes the full path.
    struct BindMemo {
        VAddr address = 0;
        u64 size = 0, generation = 0, epoch = 0, offset = 0;
        const Buffer* buffer = nullptr;
        u32 second = 0;
        bool vram = false;
    };
    static BindMemo& BindMemoSlot(VAddr address, u64 size);
    static bool BindMemoOn();
    void StoreBindMemo(VAddr address, u64 size, u64 epoch, const Buffer* buffer, u64 offset,
                       bool vram);
    std::optional<std::pair<const Buffer*, u64>> LayerBind(VAddr address, u64 size, bool is_written,
                                                           bool is_texel_buffer,
                                                           bool may_promote = true);
    /// Whether a block may get a VRAM copy (all of it loaded data, in a guest chunk).
    bool LayerEligible(u64 block);
public:
    /// BB_VERTEX_IN_PLACE=1 (diagnostics): vertex ranges bound in place, unique blocks per frame
    /// by why they have no VRAM copy, every 2 s.
    void NoteVertexInPlace(VAddr address, u64 size);
    /// BB_COPY_WHERE=1 (diagnostics): where the copy shader's sources and destinations are.
    void NoteCopyWhere(const std::vector<std::pair<unsigned long long, unsigned long long>>& src,
                       const std::vector<std::pair<unsigned long long, unsigned long long>>& dst,
                       u32 groups);
    void NoteCopyGap(u64 gap);
private:
    /// BB_LAYER_WATCH: whether a block with unannounced bytes may get a watched VRAM copy.
    bool LayerWatchable(u64 block);
    static bool LayerWatchEnabled();
    static bool LayerWatchHot();
    /// A watched block the CPU wrote: uploaded again before the next binding, trap set again.
    void LayerRefreshWatched(u64 block);
    /// Every block of [first_block, end_block) in VRAM already or eligible (LayerEligible).
    bool LayerAllKnown(u64 first_block, u64 end_block);
    /// Candidate blocks (sorted) copied to VRAM: mirrors created or merged.
    void LayerPromote(const std::vector<u64>& blocks);
    u64 LayerPromoteRun(u64 first_block, u64 end_block);
    /// The end block of a mirror over [lo, hi) that grows: room above it to grow into.
    u64 LayerGrowthEnd(u64 lo, u64 hi);
    /// Blocks [first, end) back in place (copy_back: what the GPU wrote in VRAM first).
    /// reason (statistics): 0 written binding, 1 mixed binding, 2 command write, 3 demotion
    /// request, 4 copy back, 5 unmap, 6 idle.
    void LayerDemote(u64 first_block, u64 end_block, bool copy_back, int reason);
    /// Frees retired mirrors the GPU is done with.
    void LayerMaintain();
    /// Mirrored blocks not bound for BB_VRAM_IDLE_SECONDS go back in place.
    void LayerProcessIdle();
    /// VRAM near the texture collector's critical mark: idle mirrors go back in place.
    void LayerYieldToImages(u32 now);
    /// Makes mirrors with memory left without valid blocks again over those blocks only.
    u64 LayerCompactMirrors(u64 wanted);
    /// The GPU writes [address, address + size) in the submission being recorded (write ticks).
    void NoteWriteTick(VAddr address, u64 size);
    /// NoteWriteTick for a written binding: only the parts precise_writes names, when it is set.
    void NoteBindingWriteTick(VAddr address, u64 size);
    /// Page table entry of a block (LayerPagedRecord), its writes, and rewrites after changes.
    u64 LayerEntry(u64 block);
    void LayerWriteEntries(u64 first_block, u64 end_block);
    void LayerRepublish(u64 first_block, u64 end_block);
    /// CPU writes into blocks with VRAM copies, uploaded before paged bindings.
    void LayerFlushUploads();
    void LayerForgetGpuData();
    /// The GPU's data in mirrors over [first, end) copied into the game's memory (blocks stay).
    void LayerCopyBack(u64 first_block, u64 end_block);
    /// Blocks of [first, end) that may go to VRAM become promotion candidates.
    void LayerQueuePromotion(u64 first_block, u64 end_block);
    /// Unknown blocks (LayerEligible false) made volatile in a mirror (layer_volatile).
    void LayerMakeVolatile(const std::vector<u64>& blocks);
    /// Write traps on blocks with VRAM copies (buffer_cache_layer.cpp): set when a block gets its
    /// copy, removed when it goes back in place or the CPU announces a write there (then it goes
    /// back in place too); hits are handled before the next binding.
    void LayerArmTraps(u64 first_block, u64 end_block);
    void LayerDisarmTraps(u64 first_block, u64 end_block, bool demote);
    void LayerProcessTraps();
    static u64 LayerTrapFaults();
    /// Before a binding of a mirror over volatile blocks: their bytes in it refreshed.
    void LayerRefreshVolatile(VAddr address, u64 size);
    static bool GarlicInVramForLayer();
    /// VRAM past BB_VRAM_PROMOTE_PERCENT of the budget, or within PromoteCriticalMargin of the
    /// texture collector's critical mark: no new copies of loaded blocks for now.
    bool VramPromotionsPaused();
    /// Headroom below the texture collector's critical mark: copies stop at it, and idle mirrors
    /// go back in place past it (LayerYieldToImages). 384 MiB, at most 1/12 of the VRAM budget:
    /// on a 2 GB card (budget 1646 MiB) 384 MiB held the usage at ~1 GB (GTX 1050 laptop).
    u64 PromoteCriticalMargin() const;
    /// Freed beyond the pause mark when stale mirrors go (LayerYieldToImages): room for the
    /// copies waiting. 256 MiB, at most 1/16 of the budget.
    u64 StaleHeadroom() const;

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    // bbport: texel buffer ranges known not to alias an image, per image registry generation.
    struct ImageMiss {
        VAddr address = 0;
        u32 size = 0;
        u64 generation = ~0ULL;
    };
    std::array<ImageMiss, 256> image_miss_cache{};
    Vulkan::Runtime& runtime;
    Vulkan::StagingBufferPool& staging_pool;
    BbCp::CommandProcessor* liverpool;
    Core::MemoryManager* memory;
    TextureCache& texture_cache;
    std::unique_ptr<BbMem::DirtyTracker> memory_tracker;

    StreamBuffer stream_buffer;
    Buffer gds_buffer;
    RangeSet gpu_modified_ranges;

    std::unique_ptr<FaultBuffer> fault_manager;
    std::unique_ptr<Buffer> bda_pagetable_buffer;

    std::array<const Buffer*, NUM_ARENA_PAGES> address_space{};
    std::deque<Buffer> arenas;
    std::vector<ArenaBinds> pending_binds;
    Vulkan::Semaphore memory_semaphore;
    /// bbport BB_LAYER_MEMORY: mirrors by start address, retired ones with the tick to wait for.
    std::map<VAddr, std::unique_ptr<Buffer>> layer_mirrors;
    /// LayerDomainAdd's domains: start -> end (bytes), under layer_domains_mutex.
    std::map<VAddr, VAddr> layer_domains;
    std::mutex layer_domains_mutex;
    /// The domain holding blocks [first, end) whole: its blocks [lo, hi) (edges rounded in).
    std::optional<std::pair<u64, u64>> LayerDomainOf(u64 first, u64 end);
    /// BB_LAYER_DOMAINS: mirrors over [first, end), all of it valid in VRAM, merged into one.
    bool LayerMergeAcross(u64 first, u64 end);
    /// LayerPromoteRun for a run within one domain at most.
    u64 LayerPromoteOne(u64 first, u64 end, bool domains_on);
    /// LayerPromoteRun's second half: the run [first, end) into a mirror over [lo, hi) made of
    /// `touching` (the one there when it is exactly that, else a new one they are copied into).
    u64 LayerPromoteInto(u64 first, u64 end, u64 lo, u64 hi,
                         const std::vector<BbLayer::MemorySource>& touching);
    std::vector<std::pair<std::unique_ptr<Buffer>, u64>> layer_retired;
    /// A mirror over [start, start + bytes): a pooled one of its size class, else a new one.
    std::unique_ptr<Buffer> LayerNewMirror(VAddr start, u64 bytes);
    /// Whether a new mirror of `bytes` fits in what the GPU has left (our device memory live now,
    /// not the driver's once-a-second reading): else the run waits, read in place meanwhile.
    bool LayerRoomFor(u64 bytes);
    /// A retired mirror the GPU is done with: kept for LayerNewMirror (BB_MIRROR_POOL_MB) or freed.
    void LayerReleaseMirror(std::unique_ptr<Buffer> mirror);
    /// Pool memory over `keep` bytes, and entries older than a minute, freed (oldest first).
    void LayerTrimMirrorPool(u64 keep);
    /// When each mirror was made (by its start): young ones are not compacted (LayerCompactMirrors).
    std::unordered_map<VAddr, u32> layer_mirror_born;
    std::multimap<u64, std::pair<std::unique_ptr<Buffer>, u32>> mirror_pool; ///< by allocation size; second it came
    u64 mirror_pool_bytes = 0;
    u64 mirror_pool_reuses = 0, mirror_pool_misses = 0;
    /// Retired mirrors to free, a few a frame (LayerDrainMirrorFrees).
    std::deque<std::unique_ptr<Buffer>> mirror_free_queue;
    void LayerDrainMirrorFrees();
    u64 layer_generation = 0;
    u64 layer_copied_back_bytes = 0; ///< statistics
    IntervalList<> layer_published; ///< blocks whose page table entries are written
    /// BB_PAGED_VERIFY=1 (diagnostics): the current-view entries as written (block -> address).
    std::unordered_map<u64, u64> layer_shadow;
    void LayerVerifyPaged(VAddr address, u64 size, bool is_written);
    /// A read binding not whole in one import of lazy guest memory: its pieces copied by the GPU,
    /// in stream order, into the stream buffer (fresh at every binding); empty when a piece is in
    /// no import or it is too large.
    std::optional<std::pair<const Buffer*, u64>> LayerStitch(VAddr address, u64 size);
    std::mutex layer_uploads_mutex;
    std::vector<std::pair<VAddr, u64>> layer_pending_uploads;
    /// CPU writes the GPU side heard of: their bytes lose their GPU-modified marks (LayerForgetGpuData).
    std::vector<std::pair<VAddr, u64>> layer_cpu_written;
    std::atomic<bool> layer_cpu_written_pending{false};
    std::atomic<bool> layer_uploads_pending{false};
    std::unique_ptr<Buffer> layer_trash; ///< where paged accesses out of range go
    /// What read-only paged bindings read where nothing is mapped (memory never written: zeros).
    std::unique_ptr<Buffer> layer_zeros;
    /// Blocks with bytes nobody announced (an edge shared with other data, padding between
    /// images) in a mirror only so that the rest of a reading binding's range can be there:
    /// refreshed from the game's memory for every binding over them. Paged bindings and the
    /// module's readers see them in place.
    IntervalList<> layer_volatile;
    /// Blocks the CPU wrote since they were handed out (traps, announced writes): GPU writes there
    /// stay in place (LayerGpuWriteToVram).
    IntervalList<> layer_cpu_blocks;
    /// When each block of layer_cpu_blocks was last marked (BbStats::coarse_second). A mark the
    /// CPU has not renewed for BB_LAYER_CPU_MARK_SECONDS goes once the GPU writes the whole block
    /// (LayerExpireCpuMarks): the game reused the memory without an allocation we hear of.
    std::unordered_map<u64, u32> layer_cpu_mark_second;
    /// When GPU writes over a mixed range made blocks dynamic (layer_write_bounces, not the CPU's):
    /// such marks expire as the CPU's do (LayerExpireCpuMarks).
    std::unordered_map<u64, u32> layer_dynamic_second;
    void LayerMarkCpuBlocks(u64 first, u64 end);
    void LayerExpireCpuMarks(VAddr address, u64 size);
    /// Blocks with bytes the CPU wrote unannounced, in a mirror for reading bindings anyway
    /// (BB_LAYER_WATCH, LayerWatchable): a write trap catches the CPU's next write, the block is
    /// uploaded again before the next binding and the trap set again. The GPU never writes them
    /// in VRAM (layer_cpu_blocks). On NVIDIA shaders reading the game's memory over the bus were
    /// slow where the model of 0.3 had copies in VRAM.
    IntervalList<> layer_watched;
    /// Blocks a range the GPU writes covers only in part (its first and last, the rest of them
    /// unannounced bytes), moved to VRAM with it all the same (LayerBind): a trap catches the CPU
    /// writing beside it, and the GPU's bytes there are copied back before the block goes in place.
    /// A frame-sized buffer of 305 blocks went in place for two such blocks (#149).
    IntervalList<> layer_edge_blocks;
    /// Watched blocks that went back in place for writes (LayerDemote): not watched again.
    IntervalList<> layer_watch_failed;
    std::unordered_map<u64, u8> layer_watch_strikes; ///< watched blocks sent back for requests
    /// Ranges a GPU-writing binding sent back in place once (LayerBind): the second time they
    /// stay there (dynamic_blocks).
    IntervalList<> layer_write_bounces;
    /// Per 2 MiB of guest addresses (as group_use): the second a binding last took a mirror there
    /// directly (LayerBind), or blocks there moved to VRAM. Paged bindings do not count. Stale
    /// ones go back in place first when VRAM is short (LayerYieldToImages).
    std::vector<u32> layer_group_use;
    void NoteDirectUse(VAddr address, u64 size);
    /// Dynamic blocks made so by the CPU's writes (ProcessDemotions), not by the GPU's.
    IntervalList<> layer_cpu_hot;
    u64 layer_volatile_refresh_bytes = 0; ///< statistics
    u64 layer_volatile_in_pass_bytes = 0; ///< statistics: volatile reads bound in place in passes
    /// BB_LAYER_READ_TRAPS: pages trapped for reads since the GPU wrote them in a VRAM copy only
    /// (their mutex: the draw recording thread arms, the GPU command thread lifts), the newest
    /// copy back recorded (a page without GPU data left waits for it), statistics.
    RangeSet layer_read_trapped;
    std::mutex layer_read_trap_mutex;
    u64 layer_copy_back_tick = 0;
    std::atomic<u64> layer_read_trap_hits{0}, layer_read_trap_bytes{0};
    static bool LayerReadTrapsOn();
    void LayerArmReadTraps(VAddr address, u64 size);
    /// BB_LAYER_PROBE=1 (diagnostics): which CPU code touches blocks the GPU writes in place for
    /// the CPU's sake (layer_cpu_blocks), and how soon. Nothing moves; see LayerArmProbe.
    static bool LayerProbeOn();
    void LayerArmProbe(VAddr address, u64 size);
    void LayerPrintProbe();
    /// BB_LAYER_VERIFY=1 (diagnostics): a VRAM copy read while the game's bytes changed and no
    /// trap or announced write said so (a stale copy) is reported with what is known of the block.
    static bool LayerVerifyOn();
    void LayerVerifyMirror(VAddr address, u64 size, bool is_texel_buffer);
    void LayerVerifyForget(u64 first, u64 end);
    /// `may_record`: the scheduler may be flushed and copies recorded (not on the recording
    /// thread, which may be inside a render pass with its own commands half recorded).
    void LayerCopyBackForCpu(VAddr address, bool may_record);
    std::atomic<u64> layer_read_trap_stale{0}; ///< recording thread hits on unsubmitted GPU data
    u64 layer_watch_refreshes = 0;        ///< statistics: watched blocks uploaded again
    /// Statistics: trap hits announced; by the game's code on watched / other blocks; by other
    /// code on watched / other blocks.
    std::array<u64, 5> layer_trap_kinds{};
    std::atomic<u64> arena_bind_calls{0}, arena_bind_ranges{0}; ///< vkQueueBindSparse calls, ranges
    std::atomic<u64> arena_bind_signaled{0}; ///< memory_semaphore value the last call signals

    struct Backing : public Interval {
        vk::DeviceMemory memory;
        u64 offset; ///< in blocks, like start and end (SubRange, CanMergeWith)
        constexpr bool CanMergeWith(const Backing& other) const noexcept {
            return memory == other.memory && offset + (end - start) == other.offset;
        }
        constexpr Backing SubRange(u64 a, u64 b) const noexcept {
            return {{a, b}, memory, offset + (a - start)};
        }
    };
    TrackedIntervalList<Backing> resident_ranges;
    /// bbport: blocks bound to the game's own memory, and guest unmaps to apply.
    TrackedIntervalList<> in_place_blocks;
    std::mutex pending_unmaps_mutex;
    std::vector<std::pair<VAddr, u64>> pending_unmaps;
    std::atomic<bool> unmaps_pending{false};
    std::atomic<u64> unmaps_queued{0}; ///< every UnmapInPlace (LayerInPlace's memo)
    /// bbport: ranges the game's allocator freed (LayerDomainDead): their images go (ProcessDeadImages).
    struct DeadRange {
        VAddr addr;
        u64 size;
        u64 stamp; ///< BbStats::guest_submit_stamp when freed
    };
    std::vector<DeadRange> dead_image_ranges;
    std::atomic<bool> dead_images_pending{false};
    std::mutex dynamic_mutex;
    std::unordered_map<u64, std::pair<u32, u32>> cpu_write_frames; ///< block -> last frame, frames
    std::unordered_map<u64, std::pair<u32, u32>> gpu_write_frames; ///< written range -> last frame, frames
    std::vector<u64> demote_requests;
    std::atomic<bool> demotes_requested{false};
    struct Demotion {
        u64 block;
        u64 tick;         ///< requested at
        bool cpu_written; ///< written by the CPU often: in place from now on (else: handed out again)
        bool idle = false;  ///< bbport: not bound for BB_VRAM_IDLE_SECONDS (ProcessIdleBlocks)
        bool force = false; ///< bbport: its chunk is emptied, idle or not (back once bound)
    };
    std::vector<Demotion> demotions_waiting;
    IntervalList<> dynamic_blocks;                     ///< bound in place even when Garlic
    /// Bytes last written through a path the GPU side hears of (file reads, the game's resource
    /// loaders, DMA) and not handed out again by the game's GPU memory allocator since. Only blocks
    /// made of such bytes keep a VRAM copy: anything else in a block may be written by code we do
    /// not hear of.
    IntervalList<> asset_bytes;
    IntervalList<> promote_candidates; ///< in-place blocks that may get a VRAM copy again
    IntervalList<> copy_back_blocks;   ///< VRAM blocks with GPU data, handed out again in part
    IntervalList<> gpu_written_bytes;  ///< bound for GPU writes since handed out (known data too)
    u64 promoted_bytes = 0;
    /// VRAM slots of blocks that left VRAM, for the next blocks moved there (in their chunks).
    u64 free_slot_count = 0;
    std::mutex pending_assets_mutex;
    struct PendingRange {
        VAddr addr;
        u64 size;
        bool fresh; ///< handed out again by the game's GPU memory allocator (else: an asset)
    };
    std::vector<PendingRange> pending_assets;
    std::atomic<bool> assets_pending{false};
    std::mutex late_writes_mutex;
    std::vector<std::pair<VAddr, std::vector<u8>>> late_writes;
    std::vector<std::pair<VAddr, u64>> precise_uploads;
    std::atomic<bool> late_writes_pending{false};
    u64 bind_wait_tick = 0;                            ///< the next arena binds wait for it
    std::array<std::unique_ptr<Buffer>, 4096> chunk_buffers{}; ///< per guest memory chunk (Chunk::index)

    u32 arena_memory_type_index{};
    /// bbport: system memory for residency when VRAM is full (4 GB cards), if sparse binding takes it
    std::optional<u32> arena_fallback_type_index;
    u32 arena_memory_type_bits = 0; ///< memory types the arena's blocks can be bound to
    vk::DeviceMemory residency_memory{}; ///< bbport: the 64 MiB block arena residency comes from
    u64 residency_size = 0, residency_used = 0;
    std::future<vk::DeviceMemory> spare_residency; ///< the next block, allocated in the background
    /// bbport: VRAM blocks no longer needed. Per 2 MiB of guest addresses, the second
    /// (BbStats::coarse_second) it was last bound: blocks in VRAM unused for BB_VRAM_IDLE_SECONDS
    /// (60, 0: never) go back to the game's memory, and residency chunks left without blocks
    /// go back to the driver (ProcessIdleBlocks). Blocks of unmapped memory leave VRAM too.
    static constexpr u64 USE_GROUP_BITS = 21;
    std::vector<u32> group_use;
    /// Per block of mirrored data, by 2 MiB group (allocated when first used): the second a binding
    /// last named it (its promotion counts). Blocks promoted with their neighbours and not bound
    /// go back in place (LayerProcessIdle, BB_MIRROR_UNBOUND_SECONDS).
    std::vector<std::unique_ptr<u32[]>> block_bound;
    /// Per 2 MiB group: the second blocks there last left VRAM for being idle, yielding to images
    /// or unbound; they are not promoted again for BB_MIRROR_REPROMOTE_SECONDS (15): on a 2 GB card
    /// the same data went out and back in, gigabytes a session over PCIe (2026-10-24).
    std::vector<u32> group_demoted_second;
    bool LayerRecentlyDemoted(u64 first, u64 end) const;
    void NoteBlocksBound(u64 first, u64 end);
    u32 BlockBoundSecond(u64 block) const;
    /// Groups with blocks moved in place for idleness, and those blocks: bound again, they get a
    /// VRAM copy again (QueuePromotions). Else the GPU kept reading them over PCIe (FPS 90 -> 55).
    std::vector<u8> group_demoted;
    IntervalList<> idle_demoted;
    struct ResidencyChunk {
        u64 size;
        u64 used;         ///< bytes handed out and not returned
        u32 empty_second; ///< when `used` dropped to 0
        u64 empty_tick;   ///< the submission then being recorded (it carries the rebinds)
        std::vector<u64> free_offsets; ///< slots of blocks that left (byte offsets)
        u32 evacuated_second = 0;      ///< when its blocks were last sent away (ProcessIdleBlocks)
    };
    std::unordered_map<VkDeviceMemory, ResidencyChunk> residency_chunks;
    u32 idle_scan_second = 0;
    u32 last_evacuation = 0; ///< ProcessIdleBlocks: when a chunk was last emptied
    u32 block_size{};
    u32 block_shift{};
    u32 blocks_per_arena_page{};
    u32 blocks_per_arena_page_shift{};
};

} // namespace VideoCore
