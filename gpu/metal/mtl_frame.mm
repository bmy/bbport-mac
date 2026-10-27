// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal command path's frame encoder (mtl_frame.h). Built with ARC.
// After Wooloomooloo2/bloodborne_mac_port's mtl_frame.mm (78b59f5, GPL-2.0-or-later): the same
// draw state, binding and pass handling, on KosmicKrisp's objects, with a KosmicKrisp fallback for
// what is not encoded. Classic Metal on a queue of our own (Metal 4 has no tessellator): direct
// binding, command buffers that retain nothing, encoders ordered by one fence (as KosmicKrisp's
// barrier after every encoder), and shared events between our queue and KosmicKrisp's.

#import <Metal/Metal.h>

// macOS 26 throughout (the depth bounds test, residency sets): nothing here runs unless Init found
// it (its @available check), and the layer calls none of it before Init succeeded.
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <xxhash.h>
#include <mach/mach_time.h>

#include "metal/kk_exports.h"
#include "metal/mtl_frame.h"
#include "metal/mtl_geometry.h"
#include "metal/mtl_pso_cache.h"
#include "metal/mtl_raster_util.h"
#include "metal/mtl_shader.h"
#include "metal/mtl_shadow.h"
#include "metal/mtl_tessellation.h"
#include "metal/vk_to_mtl.h"

namespace MetalFrame {
namespace {

// ---- Logging and counters ----

__attribute__((format(printf, 1, 2))) void Print(const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
    std::fflush(stdout);
}

/// Once per text.
void Note(const std::string& what) {
    static std::mutex mutex;
    static std::vector<std::string> seen;
    std::scoped_lock lock{mutex};
    if (std::find(seen.begin(), seen.end(), what) == seen.end()) {
        seen.push_back(what);
        Print("Metal frame: %s\n", what.c_str());
    }
}

struct Counters {
    std::atomic<unsigned long long> metal_draws{}, metal_dispatches{}, metal_transfers{},
        fallback_draws{},
        fallback_dispatches{}, fallback_other{}, metal_buffers{}, kk_segments{}, switches{},
        metal_passes{}, pass_breaks{}, kk_reopens{}, gpu_errors{}, tess_draws{}, geometry_draws{},
        encoders{}, epochs{}, ///< bbport: Metal encoders; barrier groups (BB_METAL_ORDER=barriers)
        residency_adds{}, residency_removes{}, residency_slow_commits{}; ///< bbport: our set
};
Counters counters;
std::atomic<unsigned long long> transfers_total{0}; ///< TransfersTotal

/// bbport: a Metal command buffer failed on the GPU (a fault or a timeout): what is recorded
/// from then on goes to KosmicKrisp (KkOnly), so that one hang is not followed by more (each GPU
/// reset can take the WindowServer down). BB_METAL_KEEP_AFTER_ERROR=1: Metal goes on.
std::atomic<bool> gpu_failed{false};

/// bbport, BB_METAL_GPU_DIAG=1: command buffers report each encoder's execution status on an
/// error; encoders are labelled and every draw and dispatch leaves a signpost (the guest shader
/// hashes as the renderer logs them), so a failure names the encoder and the work in it.
bool GpuDiag() {
    static const bool on = [] {
        const char* env = std::getenv("BB_METAL_GPU_DIAG");
        const bool yes = env && env[0] == '1';
        if (yes) {
            Print("Metal frame: GPU diagnostics: encoder status on errors, labels and a signpost "
                  "per draw and dispatch (BB_METAL_GPU_DIAG=1)\n");
        }
        return yes;
    }();
    return on;
}

__attribute__((format(printf, 2, 3))) void Signpost(id<MTLCommandEncoder> enc, const char* format,
                                                    ...) {
    if (!enc) {
        return;
    }
    char text[256];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof text, format, args);
    va_end(args);
    [enc insertDebugSignpost:[NSString stringWithUTF8String:text]];
}

/// bbport: seconds on the clock of MTLCommandBuffer's GPUStartTime / GPUEndTime (mach absolute
/// time, as CACurrentMediaTime).
double NowSeconds() {
    static const double scale = [] {
        mach_timebase_info_data_t info{};
        mach_timebase_info(&info);
        return double(info.numer) / double(info.denom) * 1e-9;
    }();
    return double(mach_absolute_time()) * scale;
}

/// bbport: a command buffer of ours that took 300 ms or more from its commit to its end on the
/// GPU: how long it waited to start (earlier work, its event waits) and how long it ran. `what`
/// says which (the wait for KosmicKrisp before a run, a run's command buffer).
void NoteSlow(id<MTLCommandBuffer> done, double committed, const char* what, u32 index, u32 count) {
    const double start = done.GPUStartTime, end = done.GPUEndTime;
    if (end <= 0.0 || end - committed < 0.3) {
        return;
    }
    static std::atomic<u32> logged{0};
    if (logged++ >= 200) {
        return;
    }
    Print("Metal frame: slow on the GPU: %s %u of %u: %.0f ms from commit to end (started %.0f ms "
          "after its commit, ran %.0f ms)\n",
          what, index + 1, count, (end - committed) * 1e3,
          start > 0.0 ? (start - committed) * 1e3 : -1.0, start > 0.0 ? (end - start) * 1e3 : -1.0);
}

/// The failed command buffer's encoders that did not complete, with their last signposts.
void PrintEncoderStatus(id<MTLCommandBuffer> done) {
    NSArray* infos = done.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    if (![infos isKindOfClass:[NSArray class]]) {
        Print("Metal frame:   (no encoder status: BB_METAL_GPU_DIAG=1 gives it)\n");
        return;
    }
    u32 completed = 0, printed = 0;
    for (id<MTLCommandBufferEncoderInfo> info in infos) {
        if (info.errorState == MTLCommandEncoderErrorStateCompleted) {
            ++completed;
            continue;
        }
        if (printed++ >= 8) {
            continue;
        }
        const char* state = info.errorState == MTLCommandEncoderErrorStateFaulted ? "FAULTED"
                            : info.errorState == MTLCommandEncoderErrorStateAffected ? "affected"
                            : info.errorState == MTLCommandEncoderErrorStatePending  ? "pending"
                                                                                     : "unknown";
        NSArray<NSString*>* posts = info.debugSignposts;
        Print("Metal frame:   encoder \"%s\": %s; %lu signposts%s\n",
              info.label ? info.label.UTF8String : "?", state, (unsigned long)posts.count,
              posts.count ? ", the last ones:" : "");
        const NSUInteger first = posts.count > 12 ? posts.count - 12 : 0;
        for (NSUInteger i = first; i < posts.count; ++i) {
            Print("Metal frame:     %s\n", posts[i].UTF8String);
        }
    }
    Print("Metal frame:   %u encoders completed, %u did not\n", completed, printed);
}

/// Where a slow submission's time went (Submit): logged when one takes 100 ms or more.
struct SubmitTimes {
    using Clock = std::chrono::steady_clock;
    Clock::time_point start = Clock::now();
    double kk_ms{}, kk_max_ms{}, empty_ms{}, empty_max_ms{}, metal_ms{}, metal_max_ms{};
    u32 kk{}, empty{}, metal{};

    static double Since(Clock::time_point t) {
        return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
    }
    void Add(double& sum, double& max, u32& n, double ms) {
        sum += ms;
        max = std::max(max, ms);
        ++n;
    }
};
std::atomic<u32> slow_submissions_logged{0};

/// Why draws and dispatches went to KosmicKrisp, by reason (the report's top few).
std::mutex reasons_mutex;
std::unordered_map<std::string, unsigned long long> reasons;
void Reason(const char* why) {
    std::scoped_lock lock{reasons_mutex};
    ++reasons[why];
}

// ---- The device ----

VkDevice vk_device{};
id<MTLDevice> device;
PFN_vkExportMetalObjectsEXT export_objects{};
/// Our own queue (classic Metal: the hardware tessellator); KosmicKrisp's residency set and ours
/// are on it.
id<MTLCommandQueue> queue;
/// Every Metal encoder we open waits for this fence when it starts and updates it when it ends:
/// each one runs after the one before it in commit order, as KosmicKrisp puts a full barrier after
/// each of its encoders. (Fences are evaluated in commit order, across command buffers.)
id<MTLFence> chain;
id<MTLResidencySet> kk_residency; ///< KosmicKrisp's (patch 0009), on our queue until Shutdown
/// Between the two queues (KosmicKrisp's timeline semaphores, their MTLSharedEvents): KosmicKrisp
/// signals `to_metal` after its work and our command buffers wait for it; ours signal `to_kk` and
/// KosmicKrisp's submissions wait for it. Values only grow (submissions are serialized).
VkSemaphore to_metal_sem{}, to_kk_sem{};
id<MTLSharedEvent> to_metal_event, to_kk_event;
u64 to_metal_value = 0, to_kk_value = 0;
/// Work on one queue the other has not waited for yet.
bool kk_ahead = false, metal_ahead = false;

/// KosmicKrisp's own entry points (the driver's lookup: not the layer's).
struct Driver {
#define BB_DRIVER_FUNCTIONS(X)                                                                     \
    X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkResetCommandBuffer)                     \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkQueueSubmit2) X(vkCreateSemaphore)           \
    X(vkGetSemaphoreCounterValue) X(vkCmdBeginRendering) X(vkCmdEndRendering)                      \
    X(vkDestroySemaphore)                      \
    X(vkCmdBindPipeline) X(vkCmdPushDescriptorSetKHR) X(vkCmdBindDescriptorSets)                   \
    X(vkCmdPushConstants) X(vkCmdBindVertexBuffers2) X(vkCmdBindIndexBuffer)                       \
    X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetViewportWithCount)                           \
    X(vkCmdSetScissorWithCount) X(vkCmdSetCullMode) X(vkCmdPipelineBarrier)                        \
    X(vkCmdSetFrontFace) X(vkCmdSetDepthTestEnable) X(vkCmdSetDepthWriteEnable)                    \
    X(vkCmdSetDepthCompareOp) X(vkCmdSetDepthBoundsTestEnable) X(vkCmdSetDepthBounds)              \
    X(vkCmdSetDepthBiasEnable) X(vkCmdSetDepthBias) X(vkCmdSetStencilTestEnable)                   \
    X(vkCmdSetStencilOp) X(vkCmdSetStencilCompareMask) X(vkCmdSetStencilWriteMask)                 \
    X(vkCmdSetStencilReference) X(vkCmdSetPrimitiveRestartEnable)                                  \
    X(vkCmdSetRasterizerDiscardEnable) X(vkCmdSetBlendConstants) X(vkCmdSetColorWriteMaskEXT)      \
    X(vkCmdSetLineWidth) X(vkCmdSetAttachmentFeedbackLoopEnableEXT) X(vkCmdPipelineBarrier2)       \
    X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkCmdDrawIndirect) X(vkCmdDrawIndexedIndirect)              \
    X(vkCmdDrawIndirectCount) X(vkCmdDrawIndexedIndirectCount) X(vkCmdDispatch)                    \
    X(vkCmdDispatchIndirect) X(vkCmdCopyBuffer) X(vkCmdFillBuffer) X(vkCmdUpdateBuffer)            \
    X(vkCmdCopyImage) X(vkCmdCopyBufferToImage) X(vkCmdCopyImageToBuffer)                          \
    X(vkCmdClearColorImage) X(vkCmdClearDepthStencilImage)
#define BB_MEMBER(name) PFN_##name name{};
    BB_DRIVER_FUNCTIONS(BB_MEMBER)
#undef BB_MEMBER
};
Driver driver;

/// Zero-filled stand-ins for unbound slots (the shaders read every slot they declare).
id<MTLBuffer> null_buffer;
id<MTLTexture> null_texture;
id<MTLSamplerState> null_sampler;

/// Our own allocations the queue needs resident (scratch memory, the stand-ins).
id<MTLResidencySet> residency;
std::mutex residency_mutex;

/// bbport: a commit of our residency set, timed (a slow one is counted and the first logged).
void CommitResidency() {
    const auto start = std::chrono::steady_clock::now();
    [residency commit];
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (ms >= 20.0 && counters.residency_slow_commits++ < 20) {
        Print("Metal frame: committing our residency set took %.0f ms\n", ms);
    }
}

void MakeResident(id<MTLAllocation> allocation) {
    std::scoped_lock lock{residency_mutex};
    [residency addAllocation:allocation];
    CommitResidency();
    ++counters.residency_adds;
}

/// bbport: scratch buffers stay resident and pooled (BB_METAL_SCRATCH_TRIM=1: the old caps, the
/// rest removed from the residency set when given back): adding and removing allocations of a
/// queue's residency set while its work is in flight is a suspect in the stalls where none of
/// our command buffers started for 4 s.
bool TrimScratch() {
    static const bool trim = [] {
        const char* env = std::getenv("BB_METAL_SCRATCH_TRIM");
        return env && env[0] == '1';
    }();
    return trim;
}

void Unresident(id<MTLAllocation> allocation) {
    std::scoped_lock residency_lock{residency_mutex};
    [residency removeAllocation:allocation];
    CommitResidency();
    ++counters.residency_removes;
}

// ---- KosmicKrisp's objects ----

struct BufferRef {
    id<MTLBuffer> buffer;
    u64 offset{}; ///< where the VkBuffer starts in `buffer`
    u64 size{};   ///< the VkBuffer's own size (its MTLBuffer can be bigger)
};

std::mutex objects_mutex;
std::unordered_map<VkBuffer, BufferRef> buffers;
std::unordered_map<VkBuffer, u64> buffer_sizes; ///< from vkCreateBuffer
struct ViewKey {
    VkImageView view;
    u32 use;
    bool operator==(const ViewKey&) const = default;
};
struct ViewKeyHash {
    size_t operator()(const ViewKey& k) const {
        return std::hash<const void*>{}(k.view) ^ (size_t(k.use) * 0x9e3779b97f4a7c15ull);
    }
};
struct ViewTexture {
    id<MTLTexture> texture;
    u32 level{}, slice{};
};
std::unordered_map<ViewKey, ViewTexture, ViewKeyHash> view_textures;
struct Sampler {
    id<MTLSamplerState> state; ///< nil: no Metal equivalent (draws with it fall back)
};
std::unordered_map<VkSampler, Sampler> samplers;
/// What an image was made as (vkCreateImage): copies need the Vulkan format.
struct ImageInfo {
    VkFormat format{};
    VkImageType type{};
};
std::unordered_map<VkImage, ImageInfo> images;
std::unordered_map<VkImageView, VkFormat> view_formats; ///< from vkCreateImageView

void Export(VkBaseOutStructure* what) {
    VkExportMetalObjectsInfoEXT info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
        .pNext = what,
    };
    export_objects(vk_device, &info);
}

/// The VkBuffer's MTLBuffer and its start (patch 0008), kept until vkDestroyBuffer.
BufferRef ResolveBuffer(VkBuffer buffer) {
    if (!buffer) {
        return {};
    }
    {
        std::scoped_lock lock{objects_mutex};
        if (auto it = buffers.find(buffer); it != buffers.end()) {
            return it->second;
        }
    }
    BbMetal::ExportBufferObject out{};
    out.buffer = buffer;
    Export(reinterpret_cast<VkBaseOutStructure*>(&out));
    BufferRef ref{out.mtlBuffer, out.offset};
    if (ref.buffer) {
        std::scoped_lock lock{objects_mutex};
        const auto size = buffer_sizes.find(buffer);
        ref.size = size != buffer_sizes.end() ? size->second : ref.buffer.length - ref.offset;
        buffers[buffer] = ref;
    }
    return ref;
}

ViewTexture ResolveView(VkImageView view, u32 use) {
    if (!view) {
        return {};
    }
    const ViewKey key{view, use};
    {
        std::scoped_lock lock{objects_mutex};
        if (auto it = view_textures.find(key); it != view_textures.end()) {
            return it->second;
        }
    }
    BbMetal::ExportViewTexture out{};
    out.imageView = view;
    out.use = use;
    Export(reinterpret_cast<VkBaseOutStructure*>(&out));
    ViewTexture t{out.mtlTexture, out.level, out.slice};
    // Not kept for attachments: a swapchain image's texture is the drawable it was acquired
    // with, a new one each frame (an attachment's export is once per render pass).
    if (t.texture && use != BbMetal::ViewTextureRender) {
        std::scoped_lock lock{objects_mutex};
        view_textures[key] = t;
    }
    return t;
}

id<MTLTexture> ResolveBufferView(VkBufferView view) {
    if (!view) {
        return nil;
    }
    VkExportMetalTextureInfoEXT out{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
        .bufferView = view,
        .plane = VK_IMAGE_ASPECT_PLANE_0_BIT,
    };
    Export(reinterpret_cast<VkBaseOutStructure*>(&out));
    return out.mtlTexture;
}

// ---- Scratch memory (push constants, buffer sizes): shared, bump-allocated ----

constexpr u64 ScratchChunk = 4ull << 20;
std::mutex scratch_mutex;
std::vector<id<MTLBuffer>> scratch_pool;

id<MTLBuffer> TakeScratch() {
    {
        std::scoped_lock lock{scratch_mutex};
        if (!scratch_pool.empty()) {
            id<MTLBuffer> b = scratch_pool.back();
            scratch_pool.pop_back();
            return b;
        }
    }
    id<MTLBuffer> b = [device newBufferWithLength:ScratchChunk
                                          options:MTLResourceStorageModeShared |
                                                  MTLResourceCPUCacheModeWriteCombined |
                                                  MTLResourceHazardTrackingModeUntracked];
    b.label = @"bbport Metal frame scratch";
    MakeResident(b);
    return b;
}

void GiveScratch(std::vector<id<MTLBuffer>>& chunks) {
    std::scoped_lock lock{scratch_mutex};
    for (id<MTLBuffer> b : chunks) {
        if (scratch_pool.size() < (TrimScratch() ? 64u : 4096u)) {
            scratch_pool.push_back(b);
        } else {
            Unresident(b); // freed: out of the set too
        }
    }
    chunks.clear();
}

// ---- GPU scratch (between a tessellated draw's kernels and its patch draw): private ----

constexpr u64 GpuScratchChunk = 64ull << 20;
std::mutex gpu_scratch_mutex;
std::vector<id<MTLBuffer>> gpu_scratch_pool;
u64 gpu_scratch_pooled = 0; ///< bytes in the pool (bbport: at most GpuScratchPoolBytes)
constexpr u64 GpuScratchPoolBytes = 1ull << 30; ///< as the old 16 chunks of 64 MiB

/// A chunk of at least `size` bytes: the smallest pooled one that fits, else a new one of a
/// multiple of GpuScratchChunk (bbport: larger ones are pooled too, up to GpuScratchPoolBytes in
/// all, so that heavy tessellation does not add and remove allocations of the residency set
/// every frame).
id<MTLBuffer> TakeGpuScratch(u64 size) {
    {
        std::scoped_lock lock{gpu_scratch_mutex};
        auto best = gpu_scratch_pool.end();
        for (auto it = gpu_scratch_pool.begin(); it != gpu_scratch_pool.end(); ++it) {
            if ((*it).length >= size && (best == gpu_scratch_pool.end() || (*it).length < (*best).length)) {
                best = it;
            }
        }
        if (best != gpu_scratch_pool.end()) {
            id<MTLBuffer> b = *best;
            gpu_scratch_pool.erase(best);
            gpu_scratch_pooled -= b.length;
            return b;
        }
    }
    const u64 rounded = TrimScratch() ? std::max(size, GpuScratchChunk)
                                      : (size + GpuScratchChunk - 1) / GpuScratchChunk * GpuScratchChunk;
    id<MTLBuffer> b = [device newBufferWithLength:std::max(rounded, GpuScratchChunk)
                                          options:MTLResourceStorageModePrivate |
                                                  MTLResourceHazardTrackingModeUntracked];
    b.label = @"bbport Metal frame GPU scratch";
    MakeResident(b);
    return b;
}

void GiveGpuScratch(std::vector<id<MTLBuffer>>& chunks) {
    std::scoped_lock lock{gpu_scratch_mutex};
    for (id<MTLBuffer> b : chunks) {
        const bool keep = TrimScratch()
                              ? b.length == GpuScratchChunk && gpu_scratch_pool.size() < 16
                              : gpu_scratch_pooled + b.length <= GpuScratchPoolBytes;
        if (keep) {
            gpu_scratch_pool.push_back(b);
            gpu_scratch_pooled += b.length;
        } else {
            Unresident(b);
        }
    }
    chunks.clear();
}

// ---- KosmicKrisp segments: command buffers of our own, recycled by a timeline ----

VkSemaphore kk_timeline{};
u64 kk_timeline_next = 1; ///< the last value given out (submissions are serialized)

struct KkBuffer {
    VkCommandBuffer cb{};
    std::atomic<u64> done_at{0}; ///< the timeline value its submission signals; 0: in use
};

/// Each command buffer has a command pool of its own: pools are externally synchronized, and a
/// segment is recorded by whichever thread records its owner (that can change between commands),
/// so no two segments ever share one. Never freed (segments may still be on the GPU at exit).
std::mutex kk_mutex;
std::vector<KkBuffer*> kk_buffers;

KkBuffer* TakeKk() {
    u64 completed = 0;
    driver.vkGetSemaphoreCounterValue(vk_device, kk_timeline, &completed);
    {
        std::scoped_lock lock{kk_mutex};
        for (KkBuffer* b : kk_buffers) {
            const u64 done = b->done_at.load(std::memory_order_acquire);
            if (done && done <= completed) {
                b->done_at.store(0, std::memory_order_release); // ours now
                driver.vkResetCommandBuffer(b->cb, 0);
                return b;
            }
        }
    }
    auto* b = new KkBuffer;
    VkCommandPool pool{};
    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = 0,
    };
    const VkCommandBufferAllocateInfo info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandBufferCount = 1,
    };
    // Called only through the driver's own entry points (never the loader), which ignore the
    // loader's word in a dispatchable handle.
    if (driver.vkCreateCommandPool(vk_device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
        delete b;
        return nullptr;
    }
    VkCommandBufferAllocateInfo alloc = info;
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    if (driver.vkAllocateCommandBuffers(vk_device, &alloc, &b->cb) != VK_SUCCESS) {
        delete b;
        return nullptr;
    }
    std::scoped_lock lock{kk_mutex};
    kk_buffers.push_back(b);
    return b;
}

// ---- Metal command buffers ----

/// What a Metal segment's completion hands back.
// ---- Fences (BB_METAL_ORDER=barriers: one per encoder) ----

std::mutex fence_mutex;
std::vector<id<MTLFence>> fence_pool;

id<MTLFence> TakeFence() {
    {
        std::scoped_lock lock{fence_mutex};
        if (!fence_pool.empty()) {
            id<MTLFence> f = fence_pool.back();
            fence_pool.pop_back();
            return f;
        }
    }
    return [device newFence];
}

void GiveFences(std::vector<id<MTLFence>>& fences) {
    std::scoped_lock lock{fence_mutex};
    for (id<MTLFence> f : fences) {
        if (fence_pool.size() < 4096) {
            fence_pool.push_back(f);
        }
    }
    fences.clear();
}

struct Retire {
    std::vector<id<MTLBuffer>> scratch;
    std::vector<id<MTLBuffer>> gpu_scratch;
    /// bbport: the encoders' fences (BB_METAL_ORDER=barriers), reused once the GPU is done (the
    /// command buffers retain nothing).
    std::vector<id<MTLFence>> fences;
    /// Holds: the owner until its submission is done, and each committed Metal command buffer
    /// of it until the GPU is done with it.
    std::atomic<u32> pending{0};
};

void Release(const std::shared_ptr<Retire>& r) {
    if (r->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        GiveScratch(r->scratch);
        GiveGpuScratch(r->gpu_scratch);
        GiveFences(r->fences);
    }
}

/// Every encoder: after the one before it (the chain fence), and before the next.
void WaitChain(id<MTLRenderCommandEncoder> e) {
    // Fragment too: a pass's loads, clears and stores (a pass without draws has nothing else).
    [e waitForFence:chain beforeStages:MTLRenderStageVertex | MTLRenderStageFragment];
}
void WaitChain(id<MTLComputeCommandEncoder> e) {
    [e waitForFence:chain];
}
void WaitChain(id<MTLBlitCommandEncoder> e) {
    [e waitForFence:chain];
}
void UpdateChain(id<MTLRenderCommandEncoder> e) {
    [e updateFence:chain afterStages:MTLRenderStageVertex | MTLRenderStageFragment];
}
void UpdateChain(id<MTLComputeCommandEncoder> e) {
    [e updateFence:chain];
}
void UpdateChain(id<MTLBlitCommandEncoder> e) {
    [e updateFence:chain];
}

} // namespace

// ---- The state of one owned command buffer ----

/// One descriptor, decoded for Metal.
struct Bound {
    id<MTLBuffer> buffer; ///< buffers: where the range starts (`offset`), `size` bytes of it
    u64 offset{};
    u64 size{};
    id<MTLTexture> texture;
    id<MTLSamplerState> sampler;
    bool ok{}; ///< has a Metal object (a missing one makes its draws fall back)
    const char* why{"a descriptor type the Metal path does not bind"}; ///< when not ok
};

/// A push descriptor write as recorded (for the fallback's replay) and decoded (for Metal).
struct PushWrite {
    VkWriteDescriptorSet write{};
    std::vector<VkDescriptorBufferInfo> buffer_infos;
    std::vector<VkDescriptorImageInfo> image_infos;
    std::vector<VkBufferView> texel_views;
    std::vector<Bound> bound;
};

struct PushSet {
    VkPipelineLayout layout{};
    std::vector<PushWrite> writes; ///< one per (binding, first array element)
};

/// bbport: a descriptor set's contents as last updated (vkUpdateDescriptorSets), decoded as push
/// writes are; a snapshot (an update makes a new one), so a bind keeps what it bound.
struct SetContents {
    std::vector<PushWrite> writes; ///< one per (binding, first array element)
};

struct BoundSets {
    VkPipelineLayout layout{};
    u32 first{};
    std::vector<VkDescriptorSet> sets;
    std::vector<u32> dynamic_offsets;
};

struct PushCall {
    VkPipelineLayout layout{};
    VkShaderStageFlags stages{};
    u32 offset{}, size{};
    bool operator==(const PushCall&) const = default;
};

struct Segment {
    id<MTLCommandBuffer> metal; ///< or:
    KkBuffer* kk{};
};

struct Owned {
    VkCommandBuffer vk{};
    std::vector<Segment> segments;

    // The open segment: a Metal command buffer, or a KosmicKrisp one.
    id<MTLCommandBuffer> cmd;
    KkBuffer* kk{};
    bool kk_in_pass{}; ///< the KosmicKrisp segment has the render pass begun
    bool kk_graphics{}, kk_compute{}; ///< it has the graphics / compute state (replayed)

    std::shared_ptr<Retire> retire;
    u32 metal_segments{}; ///< Metal command buffers made so far (KkOnly)
    id<MTLRenderCommandEncoder> render;
    id<MTLComputeCommandEncoder> compute;
    id<MTLBlitCommandEncoder> blit;
    id<MTLBuffer> scratch;
    u64 scratch_used{};
    id<MTLBuffer> gpu_scratch;
    u64 gpu_scratch_used{};

    // ---- The Vulkan state as recorded (Metal encodes from it; the fallback replays it) ----
    VkPipeline graphics_pipeline{}, compute_pipeline{};
    const MetalShadow::GraphicsEntry* graphics{};
    /// bbport: the draw's render pipeline (DrawBlocker): the entry's, or its variant for the
    /// pass's depth and stencil formats (`variant_building`: that variant is not ready yet).
    const Metal::RenderPipeline* draw_render{};
    bool variant_building{};
    const Metal::ComputePipeline* compute_metal{};
    std::unordered_map<u32, PushSet> push_sets; ///< by bind point << 16 | set
    std::unordered_map<u32, BoundSets> bound_sets; ///< by bind point << 16 | set
    /// bbport: the contents of the bound sets whose contents are known (DescriptorSetsUpdated),
    /// by bind point << 16 | set; a bound set without them makes its draws fall back.
    std::unordered_map<u32, std::shared_ptr<const SetContents>> set_contents;
    /// Push constants per bind point, as KosmicKrisp keeps them (graphics, compute).
    std::array<std::array<u8, 256>, 2> push{};
    std::vector<PushCall> push_calls;
    struct VertexBuffer {
        VkBuffer buffer{};
        VkDeviceSize offset{}, size{VK_WHOLE_SIZE}, stride{};
        bool set{};
    };
    std::array<VertexBuffer, 32> vertex_buffers{};
    VkBuffer index_buffer{};
    VkDeviceSize index_offset{};
    VkIndexType index_type{VK_INDEX_TYPE_UINT16};

    std::array<bool, size_t(State::Count)> state_set{};
    std::vector<VkViewport> viewports;
    std::vector<VkRect2D> scissors;
    bool viewport_with_count{}, scissor_with_count{}; ///< which call set them (the replay's)
    VkCullModeFlags cull_mode{};
    VkFrontFace front_face{VK_FRONT_FACE_COUNTER_CLOCKWISE};
    bool depth_test{}, depth_write{}, depth_bounds_test{}, depth_bias_enable{}, stencil_test{},
        primitive_restart{}, rasterizer_discard{};
    u32 feedback_loop{};
    u32 active_queries{}; ///< occlusion queries begun and not ended (QueryBegun)
    VkCompareOp depth_compare{VK_COMPARE_OP_ALWAYS};
    float depth_bounds[2]{0.0f, 1.0f};
    float depth_bias[3]{}; ///< constant, clamp, slope
    struct Face {
        VkStencilOp fail{}, pass{}, depth_fail{};
        VkCompareOp compare{VK_COMPARE_OP_ALWAYS};
        u32 compare_mask{0xff}, write_mask{0xff}, reference{};
    } front, back;
    float blend_constants[4]{};
    std::array<VkColorComponentFlags, 8> write_masks{};
    u32 write_mask_count{};
    float line_width{1.0f};

    // ---- The render pass ----
    bool in_pass{};
    bool pass_begun{};   ///< opened once (on either side): later opens load what is there
    bool pass_metal{};   ///< Metal can render into it (its attachments resolved)
    VkRenderingInfo pass_info{};
    std::vector<VkRenderingAttachmentInfo> pass_colors;
    VkRenderingAttachmentInfo pass_depth{}, pass_stencil{};
    bool has_depth{}, has_stencil{};
    /// bbport, BB_METAL_ORDER=barriers: the encoders of this Metal segment are ordered by the
    /// renderer's barriers instead of each after the one before (the chain). `waits`: the fences
    /// of the encoders before the last barrier (each encoder waits for them; the chain instead
    /// while `wait_chain`); `epoch`: those since (they may overlap); `last`: the last encoder's;
    /// `pass_part`: the open pass's last render encoder's (a reopened pass loads what it stored).
    std::vector<id<MTLFence>> waits, epoch;
    bool wait_chain{true};
    bool compute_spans_barrier{}; ///< the open compute encoder took a barrier as memoryBarrier
    bool draw_kernels{}; ///< a draw's kernels were just encoded: its render encoder waits for them
    id<MTLFence> last, pass_part;
    MTLRenderPassDescriptor* pass_first;  ///< as begun (its clears)
    MTLRenderPassDescriptor* pass_reopen; ///< loading
    u32 target_width{}, target_height{};
    std::array<u32, 8> pass_color_formats{};
    u32 pass_depth_format{}, pass_stencil_format{}, pass_samples{1};
};

namespace {

// ---- Segments ----

// ---- Encoder order ----

/// BB_METAL_ORDER=barriers: encoders between two barriers of the renderer may overlap on the GPU
/// (as Vulkan allows); the default (chain) runs each after the one before, as KosmicKrisp does.
bool OrderByBarriers() {
    static const bool on = [] {
        const char* env = std::getenv("BB_METAL_ORDER");
        const bool barriers = env && std::strcmp(env, "barriers") == 0;
        Print("Metal frame: encoders ordered %s (BB_METAL_ORDER=%s)\n",
              barriers ? "by the renderer's barriers: those between two may overlap"
                       : "one after another",
              barriers ? "barriers" : "chain");
        return barriers;
    }();
    return on;
}

/// What a new encoder waits for. `reopen`: it continues the open render pass (after its last
/// part and whatever ran in between, a tessellated draw's kernels).
template <typename Encoder>
void OrderStart(Owned& o, Encoder e, bool reopen) {
    if (!OrderByBarriers()) {
        WaitChain(e);
        return;
    }
    const auto wait = [&](id<MTLFence> f) {
        if constexpr (std::is_same_v<Encoder, id<MTLRenderCommandEncoder>>) {
            [e waitForFence:f beforeStages:MTLRenderStageVertex | MTLRenderStageFragment];
        } else {
            [e waitForFence:f];
        }
    };
    if (o.wait_chain) {
        wait(chain);
    }
    for (id<MTLFence> f : o.waits) {
        wait(f);
    }
    if (reopen) { // (also a draw's render encoder after its kernels)
        if (o.last) {
            wait(o.last);
        }
        if (o.pass_part && o.pass_part != o.last) {
            wait(o.pass_part);
        }
    }
}

/// An encoder's end: its fence (for the encoders after the next barrier).
template <typename Encoder>
void OrderEnd(Owned& o, Encoder e) {
    ++counters.encoders;
    if (!OrderByBarriers()) {
        UpdateChain(e);
        return;
    }
    id<MTLFence> f = TakeFence();
    if (!o.retire) {
        o.retire = std::make_shared<Retire>();
    }
    o.retire->fences.push_back(f);
    if constexpr (std::is_same_v<Encoder, id<MTLRenderCommandEncoder>>) {
        [e updateFence:f afterStages:MTLRenderStageVertex | MTLRenderStageFragment];
        o.pass_part = f;
    } else {
        [e updateFence:f];
    }
    if constexpr (std::is_same_v<Encoder, id<MTLComputeCommandEncoder>>) {
        if (o.compute_spans_barrier) {
            // Its dispatches before the barrier are before every encoder after it.
            o.waits.push_back(f);
            o.compute_spans_barrier = false;
        }
    }
    o.epoch.push_back(f);
    o.last = f;
}

/// A barrier: the encoders after it wait for those before it (transitively for the earlier
/// ones: each of those waited for the ones before its own barrier).
void OrderBarrier(Owned& o) {
    if (o.epoch.empty()) {
        return;
    }
    o.waits = std::move(o.epoch);
    o.epoch.clear();
    o.wait_chain = false;
    ++counters.epochs;
}

/// The segment's end: the chain passed on after all of its encoders (the next segment, and the
/// submission's end for KosmicKrisp, wait for the chain).
void OrderSegmentEnd(Owned& o, id<MTLCommandBuffer> cmd) {
    if (!OrderByBarriers()) {
        return;
    }
    if (!o.epoch.empty() || !o.waits.empty()) {
        id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
        if (GpuDiag()) {
            blit.label = @"segment end";
        }
        for (id<MTLFence> f : o.waits) {
            [blit waitForFence:f];
        }
        for (id<MTLFence> f : o.epoch) {
            [blit waitForFence:f];
        }
        [blit updateFence:chain];
        [blit endEncoding];
    }
    o.waits.clear();
    o.epoch.clear();
    o.wait_chain = true;
    o.compute_spans_barrier = false;
    o.draw_kernels = false;
    o.last = nil;
    o.pass_part = nil;
}

/// Ends the render encoder (every one renders a Vulkan pass, its store actions left to its end):
/// `pass_end`, at vkCmdEndRendering, stores what Vulkan's store operations keep; any other end
/// splits the pass (it reopens, loading), so everything is stored.
void CloseRender(Owned& o, bool pass_end) {
    if (!o.render) {
        return;
    }
    MTLRenderPassDescriptor* rp = o.pass_first;
    const auto store = [&](VkAttachmentStoreOp op) {
        return pass_end && op == VK_ATTACHMENT_STORE_OP_DONT_CARE ? MTLStoreActionDontCare
                                                                 : MTLStoreActionStore;
    };
    for (u32 i = 0; rp && i < o.pass_colors.size(); ++i) {
        if (rp.colorAttachments[i].texture) {
            [o.render setColorStoreAction:store(o.pass_colors[i].storeOp) atIndex:i];
        }
    }
    if (rp && rp.depthAttachment.texture) {
        [o.render setDepthStoreAction:store(o.pass_depth.storeOp)];
    }
    if (rp && rp.stencilAttachment.texture) {
        [o.render setStencilStoreAction:store(o.pass_stencil.storeOp)];
    }
    if (pass_end) {
        ++counters.metal_passes;
    } else if (o.in_pass) {
        ++counters.pass_breaks;
    }
    OrderEnd(o, o.render);
    [o.render endEncoding];
    o.render = nil;
}

void EndCompute(Owned& o) {
    if (o.compute) {
        OrderEnd(o, o.compute);
        [o.compute endEncoding];
        o.compute = nil;
    }
}

void EndBlit(Owned& o) {
    if (o.blit) {
        OrderEnd(o, o.blit);
        [o.blit endEncoding];
        o.blit = nil;
    }
}

void EndEncoders(Owned& o) {
    CloseRender(o, false);
    EndCompute(o);
    EndBlit(o);
}

void EndMetal(Owned& o) {
    if (!o.cmd) {
        return;
    }
    EndEncoders(o);
    OrderSegmentEnd(o, o.cmd);
    o.segments.push_back({.metal = o.cmd});
    o.cmd = nil;
}

void EndKk(Owned& o) {
    if (!o.kk) {
        return;
    }
    if (o.kk_in_pass) {
        driver.vkCmdEndRendering(o.kk->cb);
        o.kk_in_pass = false;
    }
    driver.vkEndCommandBuffer(o.kk->cb);
    o.segments.push_back({.kk = o.kk});
    o.kk = nullptr;
}

id<MTLCommandBuffer> Metal(Owned& o) {
    if (o.cmd) {
        return o.cmd;
    }
    if (o.kk) {
        EndKk(o);
        ++counters.switches;
    }
    if (!o.retire) {
        o.retire = std::make_shared<Retire>();
    }
    // Retains nothing: KosmicKrisp's objects live as their Vulkan objects (in use until the
    // submission's fence), ours (scratch) until the Retire is released.
    if (GpuDiag()) {
        MTLCommandBufferDescriptor* d = [MTLCommandBufferDescriptor new];
        d.retainedReferences = NO;
        d.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
        o.cmd = [queue commandBufferWithDescriptor:d];
    } else {
        o.cmd = [queue commandBufferWithUnretainedReferences];
    }
    o.cmd.label = @"bbport Metal frame";
    ++o.metal_segments;
    return o.cmd;
}

/// Metal command buffers per owned one, at most: each is made when its segment begins and counts
/// against the queue's limit until it completes, so one that switches between the two without
/// end must not exhaust the queue (the rest of it then stays on KosmicKrisp).
constexpr u32 MaxMetalSegments = 64;

/// Why this command must stay in KosmicKrisp (null: it may go to Metal): an occlusion query is
/// active, or the owned command buffer has had its Metal segments.
const char* KkOnly(const Owned& o) {
    static const bool keep = [] {
        const char* env = std::getenv("BB_METAL_KEEP_AFTER_ERROR");
        return env && env[0] == '1';
    }();
    if (gpu_failed.load(std::memory_order_relaxed) && !keep) {
        return "a GPU error earlier (Metal is off from then on)";
    }
    if (o.active_queries) {
        return "an occlusion query";
    }
    if (!o.cmd && o.metal_segments >= MaxMetalSegments) {
        return "too many Metal segments in one command buffer";
    }
    return nullptr;
}

// ---- Replay into a KosmicKrisp segment ----

void ReplayPass(Owned& o, VkCommandBuffer cb) {
    VkRenderingInfo info = o.pass_info;
    std::vector<VkRenderingAttachmentInfo> colors = o.pass_colors;
    VkRenderingAttachmentInfo depth = o.pass_depth, stencil = o.pass_stencil;
    // bbport: this part may not be the pass's last (a later draw can go to Metal, or a new
    // segment begin): what it draws is kept for the next part to load (as CloseRender's split
    // stores do on the Metal side). The renderer's passes store anyway.
    const auto keep = [](VkRenderingAttachmentInfo& a) {
        if (a.storeOp == VK_ATTACHMENT_STORE_OP_DONT_CARE ||
            a.storeOp == VK_ATTACHMENT_STORE_OP_NONE) {
            a.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        }
    };
    for (auto& c : colors) {
        keep(c);
    }
    keep(depth);
    keep(stencil);
    if (o.pass_begun) {
        ++counters.kk_reopens;
        for (auto& c : colors) {
            c.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        }
        depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        stencil.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    }
    info.pColorAttachments = colors.data();
    info.colorAttachmentCount = u32(colors.size());
    info.pDepthAttachment = o.has_depth ? &depth : nullptr;
    info.pStencilAttachment = o.has_stencil ? &stencil : nullptr;
    driver.vkCmdBeginRendering(cb, &info);
    o.pass_begun = true;
}

void ReplayState(Owned& o, VkCommandBuffer cb) {
    const auto set = [&](State s) { return o.state_set[size_t(s)]; };
    if (set(State::Viewport) && !o.viewports.empty()) {
        if (o.viewport_with_count) {
            driver.vkCmdSetViewportWithCount(cb, u32(o.viewports.size()), o.viewports.data());
        } else {
            driver.vkCmdSetViewport(cb, 0, u32(o.viewports.size()), o.viewports.data());
        }
    }
    if (set(State::Scissor) && !o.scissors.empty()) {
        if (o.scissor_with_count) {
            driver.vkCmdSetScissorWithCount(cb, u32(o.scissors.size()), o.scissors.data());
        } else {
            driver.vkCmdSetScissor(cb, 0, u32(o.scissors.size()), o.scissors.data());
        }
    }
    if (set(State::CullMode)) {
        driver.vkCmdSetCullMode(cb, o.cull_mode);
    }
    if (set(State::FrontFace)) {
        driver.vkCmdSetFrontFace(cb, o.front_face);
    }
    if (set(State::DepthTest)) {
        driver.vkCmdSetDepthTestEnable(cb, o.depth_test);
    }
    if (set(State::DepthWrite)) {
        driver.vkCmdSetDepthWriteEnable(cb, o.depth_write);
    }
    if (set(State::DepthCompare)) {
        driver.vkCmdSetDepthCompareOp(cb, o.depth_compare);
    }
    if (set(State::DepthBoundsTest) && driver.vkCmdSetDepthBoundsTestEnable) {
        driver.vkCmdSetDepthBoundsTestEnable(cb, o.depth_bounds_test);
    }
    if (set(State::DepthBounds) && driver.vkCmdSetDepthBounds) {
        driver.vkCmdSetDepthBounds(cb, o.depth_bounds[0], o.depth_bounds[1]);
    }
    if (set(State::DepthBiasEnable)) {
        driver.vkCmdSetDepthBiasEnable(cb, o.depth_bias_enable);
    }
    if (set(State::DepthBias)) {
        driver.vkCmdSetDepthBias(cb, o.depth_bias[0], o.depth_bias[1], o.depth_bias[2]);
    }
    if (set(State::StencilTest)) {
        driver.vkCmdSetStencilTestEnable(cb, o.stencil_test);
    }
    if (set(State::StencilOp)) {
        driver.vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, o.front.fail, o.front.pass,
                                 o.front.depth_fail, o.front.compare);
        driver.vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, o.back.fail, o.back.pass,
                                 o.back.depth_fail, o.back.compare);
    }
    if (set(State::StencilCompareMask)) {
        driver.vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_BIT, o.front.compare_mask);
        driver.vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_BACK_BIT, o.back.compare_mask);
    }
    if (set(State::StencilWriteMask)) {
        driver.vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_BIT, o.front.write_mask);
        driver.vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_BACK_BIT, o.back.write_mask);
    }
    if (set(State::StencilReference)) {
        driver.vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_BIT, o.front.reference);
        driver.vkCmdSetStencilReference(cb, VK_STENCIL_FACE_BACK_BIT, o.back.reference);
    }
    if (set(State::PrimitiveRestart)) {
        driver.vkCmdSetPrimitiveRestartEnable(cb, o.primitive_restart);
    }
    if (set(State::RasterizerDiscard)) {
        driver.vkCmdSetRasterizerDiscardEnable(cb, o.rasterizer_discard);
    }
    if (set(State::BlendConstants)) {
        driver.vkCmdSetBlendConstants(cb, o.blend_constants);
    }
    if (set(State::ColorWriteMask) && driver.vkCmdSetColorWriteMaskEXT && o.write_mask_count) {
        driver.vkCmdSetColorWriteMaskEXT(cb, 0, o.write_mask_count, o.write_masks.data());
    }
    if (set(State::LineWidth)) {
        driver.vkCmdSetLineWidth(cb, o.line_width);
    }
    if (set(State::FeedbackLoop) && driver.vkCmdSetAttachmentFeedbackLoopEnableEXT) {
        driver.vkCmdSetAttachmentFeedbackLoopEnableEXT(cb, o.feedback_loop);
    }
}

void ReplayBindings(Owned& o, VkCommandBuffer cb, VkPipelineBindPoint point) {
    std::vector<u32> keys;
    for (const auto& [key, s] : o.bound_sets) {
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    for (const u32 key : keys) {
        const BoundSets& s = o.bound_sets[key];
        if ((key >> 16) != u32(point) || s.sets.empty()) {
            continue; // a marker for a set bound with the one before (CmdBindDescriptorSets)
        }
        driver.vkCmdBindDescriptorSets(cb, point, s.layout, s.first, u32(s.sets.size()),
                                       s.sets.data(), u32(s.dynamic_offsets.size()),
                                       s.dynamic_offsets.data());
    }
    for (auto& [key, s] : o.push_sets) {
        if ((key >> 16) != u32(point) || s.writes.empty()) {
            continue;
        }
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(s.writes.size());
        for (PushWrite& w : s.writes) {
            VkWriteDescriptorSet write = w.write;
            write.pBufferInfo = w.buffer_infos.empty() ? nullptr : w.buffer_infos.data();
            write.pImageInfo = w.image_infos.empty() ? nullptr : w.image_infos.data();
            write.pTexelBufferView = w.texel_views.empty() ? nullptr : w.texel_views.data();
            writes.push_back(write);
        }
        driver.vkCmdPushDescriptorSetKHR(cb, point, s.layout, key & 0xffff, u32(writes.size()),
                                         writes.data());
    }
    const bool compute = point == VK_PIPELINE_BIND_POINT_COMPUTE;
    for (const PushCall& p : o.push_calls) {
        if (bool(p.stages & VK_SHADER_STAGE_COMPUTE_BIT) == compute) {
            driver.vkCmdPushConstants(cb, p.layout, p.stages, p.offset, p.size,
                                      o.push[compute].data() + p.offset);
        }
    }
}

/// The open KosmicKrisp segment's command buffer, opened (from Metal) as needed; `in_pass`:
/// with the render pass begun on it, else outside any pass.
VkCommandBuffer Kk(Owned& o, bool in_pass) {
    if (!o.kk) {
        if (o.cmd) {
            EndMetal(o);
            ++counters.switches;
        }
        o.kk = TakeKk();
        if (!o.kk) {
            Note("no command buffer for a KosmicKrisp segment: commands are lost");
            return VK_NULL_HANDLE;
        }
        const VkCommandBufferBeginInfo begin{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        driver.vkBeginCommandBuffer(o.kk->cb, &begin);
        o.kk_graphics = o.kk_compute = false;
        ++counters.kk_segments;
    }
    if (in_pass && o.in_pass && !o.kk_in_pass) {
        ReplayPass(o, o.kk->cb);
        o.kk_in_pass = true;
    } else if (!in_pass && o.kk_in_pass) {
        driver.vkCmdEndRendering(o.kk->cb);
        o.kk_in_pass = false;
    }
    return o.kk->cb;
}

/// A KosmicKrisp segment ready for a draw or dispatch with the whole state.
VkCommandBuffer KkWithState(Owned& o, VkPipelineBindPoint point) {
    const bool graphics = point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    VkCommandBuffer cb = Kk(o, graphics);
    if (!cb) {
        return cb;
    }
    // A segment gets the bind point's whole state once (whoever opened it); after that the state
    // commands go to it as they come (Mirror), and dynamic state outlives render passes.
    if (graphics && !o.kk_graphics) {
        ReplayState(o, cb);
        for (u32 i = 0; i < o.vertex_buffers.size(); ++i) {
            const auto& vb = o.vertex_buffers[i];
            if (vb.set) {
                driver.vkCmdBindVertexBuffers2(cb, i, 1, &vb.buffer, &vb.offset, &vb.size,
                                               vb.stride ? &vb.stride : nullptr);
            }
        }
        if (o.index_buffer) {
            driver.vkCmdBindIndexBuffer(cb, o.index_buffer, o.index_offset, o.index_type);
        }
        if (o.graphics_pipeline) {
            driver.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, o.graphics_pipeline);
        }
        ReplayBindings(o, cb, VK_PIPELINE_BIND_POINT_GRAPHICS);
        o.kk_graphics = true;
    }
    if (!graphics && !o.kk_compute) {
        if (o.compute_pipeline) {
            driver.vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, o.compute_pipeline);
        }
        ReplayBindings(o, cb, VK_PIPELINE_BIND_POINT_COMPUTE);
        o.kk_compute = true;
    }
    return cb;
}

/// While a KosmicKrisp segment is open, state commands go to it too (it then needs no replay).
template <typename F>
void Mirror(Owned& o, F&& f) {
    if (o.kk) {
        f(o.kk->cb);
    }
}

// ---- Metal encoders ----

id<MTLRenderCommandEncoder> RenderEncoder(Owned& o) {
    if (o.render) {
        return o.render;
    }
    id<MTLCommandBuffer> cmd = Metal(o);
    EndCompute(o);
    EndBlit(o);
    // A reopened pass, or a draw's render encoder after its kernels (EncodeTessellated,
    // EncodeGeometry), even as the pass's first.
    const bool reopen = o.pass_begun || o.draw_kernels;
    o.draw_kernels = false;
    o.render = [cmd renderCommandEncoderWithDescriptor:o.pass_begun ? o.pass_reopen : o.pass_first];
    o.pass_begun = true;
    if (o.render) {
        OrderStart(o, o.render, reopen);
        if (GpuDiag()) {
            o.render.label = reopen ? @"render pass (continued)" : @"render pass";
        }
    }
    return o.render;
}

id<MTLComputeCommandEncoder> ComputeEncoder(Owned& o) {
    if (o.compute) {
        return o.compute;
    }
    id<MTLCommandBuffer> cmd = Metal(o);
    CloseRender(o, false);
    EndBlit(o);
    // Concurrent: Vulkan orders dispatches only through barriers (CmdPipelineBarrier).
    o.compute = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    OrderStart(o, o.compute, false);
    if (GpuDiag()) {
        o.compute.label = @"compute";
    }
    return o.compute;
}

id<MTLBlitCommandEncoder> BlitEncoder(Owned& o) {
    if (o.blit) {
        return o.blit;
    }
    id<MTLCommandBuffer> cmd = Metal(o);
    CloseRender(o, false);
    EndCompute(o);
    o.blit = [cmd blitCommandEncoder];
    OrderStart(o, o.blit, false);
    if (GpuDiag()) {
        o.blit.label = @"transfers";
    }
    return o.blit;
}

/// `bytes` of `data` in scratch memory (shared, kept until the segment is done): `buffer`/
/// `offset` say where.
void Scratch(Owned& o, const void* data, u64 bytes, id<MTLBuffer>* buffer, u64* offset) {
    const u64 size = std::max<u64>((bytes + 255) & ~u64(255), 256);
    if (!o.scratch || o.scratch_used + size > ScratchChunk) {
        o.scratch = TakeScratch();
        o.scratch_used = 0;
        if (!o.retire) {
            o.retire = std::make_shared<Retire>();
        }
        o.retire->scratch.push_back(o.scratch);
    }
    const u64 at = o.scratch_used;
    o.scratch_used += size;
    std::memcpy(static_cast<u8*>(o.scratch.contents) + at, data, bytes);
    *buffer = o.scratch;
    *offset = at;
}

/// `bytes` of GPU scratch (private, kept until the segment is done): `buffer`/`offset` say where.
void GpuScratch(Owned& o, u64 bytes, id<MTLBuffer>* buffer, u64* offset) {
    const u64 size = std::max<u64>((bytes + 255) & ~u64(255), 256);
    if (!o.gpu_scratch || o.gpu_scratch_used + size > o.gpu_scratch.length) {
        o.gpu_scratch = TakeGpuScratch(size);
        o.gpu_scratch_used = 0;
        if (!o.retire) {
            o.retire = std::make_shared<Retire>();
        }
        o.retire->gpu_scratch.push_back(o.gpu_scratch);
    }
    *buffer = o.gpu_scratch;
    *offset = o.gpu_scratch_used;
    o.gpu_scratch_used += size;
}

// ---- Descriptors ----

void Decode(PushWrite& w) {
    const VkWriteDescriptorSet& write = w.write;
    w.bound.assign(write.descriptorCount, Bound{});
    for (u32 i = 0; i < write.descriptorCount; ++i) {
        Bound& b = w.bound[i];
        switch (write.descriptorType) {
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: {
            const VkDescriptorBufferInfo& info = w.buffer_infos[i];
            if (!info.buffer) {
                // A null descriptor (nullDescriptor): an empty buffer. Every access of the Metal
                // shaders is checked against the bound size (mtl_spirv_robust.h): with size 0
                // reads give zero and writes are dropped; the zero buffer is only addressed.
                b.buffer = null_buffer;
                b.offset = 0;
                b.size = 0;
                b.ok = true;
                break;
            }
            const BufferRef ref = ResolveBuffer(info.buffer);
            b.why = "a buffer descriptor without a Metal buffer";
            if (ref.buffer) {
                b.buffer = ref.buffer;
                b.offset = ref.offset + info.offset;
                b.size = info.range == VK_WHOLE_SIZE
                             ? (info.offset < ref.size ? ref.size - info.offset : 0)
                             : info.range;
                b.ok = true;
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: {
            const VkDescriptorImageInfo& info = w.image_infos[i];
            const bool storage = write.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            b.why = !info.imageView ? "a null image descriptor (no stand-in of its type yet)"
                    : storage       ? "a storage image without a Metal texture"
                                    : "a sampled image without a Metal texture";
            b.texture = ResolveView(info.imageView, storage ? BbMetal::ViewTextureStorage
                                                            : BbMetal::ViewTextureSampled)
                            .texture;
            b.ok = b.texture != nil;
            if (storage && b.ok) {
                // Writes to a format stored swizzled would land in the wrong channels.
                b.why = "a storage image of a format stored swizzled";
                std::scoped_lock lock{objects_mutex};
                const auto f = view_formats.find(info.imageView);
                b.ok = f != view_formats.end() &&
                       Metal::VkToMtl::FormatSwizzle(vk::Format(f->second)) ==
                           Metal::VkToMtl::IdentitySwizzle;
            }
            if (write.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
                std::scoped_lock lock{objects_mutex};
                auto it = samplers.find(info.sampler);
                b.sampler = it == samplers.end() ? nil : it->second.state;
                if (b.ok && !b.sampler) {
                    b.why = it == samplers.end()
                                ? "an unknown or immutable sampler"
                                : "a sampler Metal lacks (border color or extension)";
                }
                b.ok = b.ok && b.sampler;
            }
            break;
        }
        case VK_DESCRIPTOR_TYPE_SAMPLER: {
            std::scoped_lock lock{objects_mutex};
            auto it = samplers.find(w.image_infos[i].sampler);
            b.sampler = it == samplers.end() ? nil : it->second.state;
            b.ok = b.sampler != nil;
            b.why = it == samplers.end() ? "an unknown or immutable sampler"
                                         : "a sampler Metal lacks (border color or extension)";
            break;
        }
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            b.texture = ResolveBufferView(w.texel_views[i]);
            b.ok = b.texture != nil;
            b.why = "a texel buffer without a Metal texture";
            break;
        default:
            break;
        }
    }
}

/// bbport: descriptor sets' contents by handle (DescriptorSetsAllocated, DescriptorSetsUpdated);
/// null: not known (written by a copy or a type not decoded). Never destroyed: recording threads
/// may still look sets up while the process exits.
std::mutex& set_contents_mutex = *new std::mutex;
auto& known_sets = *new std::unordered_map<VkDescriptorSet, std::shared_ptr<const SetContents>>;
/// The sets allocated from each pool (forgotten at its reset or destruction).
auto& pool_sets = *new std::unordered_map<VkDescriptorPool, std::vector<VkDescriptorSet>>;

std::shared_ptr<const SetContents> SetContentsOf(VkDescriptorSet set) {
    std::scoped_lock lock{set_contents_mutex};
    const auto it = known_sets.find(set);
    return it == known_sets.end() ? nullptr : it->second;
}

/// The element of a pushed binding (or of a bound set's), null when nothing was written there.
const Bound* FindBound(const Owned& o, VkPipelineBindPoint point, u32 set, u32 binding, u32 e) {
    const std::vector<PushWrite>* writes = nullptr;
    if (const auto it = o.push_sets.find((u32(point) << 16) | set); it != o.push_sets.end()) {
        writes = &it->second.writes;
    } else if (const auto c = o.set_contents.find((u32(point) << 16) | set);
               c != o.set_contents.end()) {
        writes = &c->second->writes; // bbport: a bound descriptor set's
    }
    if (!writes) {
        return nullptr;
    }
    for (const PushWrite& w : *writes) {
        if (w.write.dstBinding == binding && e >= w.write.dstArrayElement &&
            e < w.write.dstArrayElement + w.write.descriptorCount) {
            return &w.bound[e - w.write.dstArrayElement];
        }
    }
    return nullptr;
}

/// Whether every resource a stage reads has a Metal object (else `why`).
bool StageResolvable(const Owned& o, VkPipelineBindPoint point, const Metal::CompiledShader* shader,
                     const char*& why) {
    if (!shader) {
        return true;
    }
    for (const auto& b : shader->bindings) {
        if (b.kind == Metal::ShaderBinding::Kind::PushConstants) {
            continue;
        }
        if (o.bound_sets.contains((u32(point) << 16) | b.set) &&
            !o.set_contents.contains((u32(point) << 16) | b.set)) {
            why = "a descriptor set (not pushed) whose contents are not known";
            return false;
        }
        for (u32 e = 0; e < b.count; ++e) {
            const Bound* bound = FindBound(o, point, b.set, b.binding, e);
            if (bound && !bound->ok) {
                why = bound->why;
                return false;
            }
        }
    }
    return true;
}

/// The Metal stage a shader binds to.
enum class Stage { Vertex, Fragment, Compute };

/// Sets one slot of `stage` on the open encoder (render for Vertex/Fragment, compute otherwise).
void SetBuffer(Owned& o, Stage stage, id<MTLBuffer> buffer, u64 offset, u32 slot) {
    switch (stage) {
    case Stage::Vertex:
        [o.render setVertexBuffer:buffer offset:offset atIndex:slot];
        break;
    case Stage::Fragment:
        [o.render setFragmentBuffer:buffer offset:offset atIndex:slot];
        break;
    case Stage::Compute:
        [o.compute setBuffer:buffer offset:offset atIndex:slot];
        break;
    }
}
void SetBytes(Owned& o, Stage stage, const void* data, u64 size, u32 slot) {
    switch (stage) {
    case Stage::Vertex:
        [o.render setVertexBytes:data length:size atIndex:slot];
        break;
    case Stage::Fragment:
        [o.render setFragmentBytes:data length:size atIndex:slot];
        break;
    case Stage::Compute:
        [o.compute setBytes:data length:size atIndex:slot];
        break;
    }
}
void SetTexture(Owned& o, Stage stage, id<MTLTexture> texture, u32 slot) {
    switch (stage) {
    case Stage::Vertex:
        [o.render setVertexTexture:texture atIndex:slot];
        break;
    case Stage::Fragment:
        [o.render setFragmentTexture:texture atIndex:slot];
        break;
    case Stage::Compute:
        [o.compute setTexture:texture atIndex:slot];
        break;
    }
}
void SetSampler(Owned& o, Stage stage, id<MTLSamplerState> sampler, u32 slot) {
    switch (stage) {
    case Stage::Vertex:
        [o.render setVertexSamplerState:sampler atIndex:slot];
        break;
    case Stage::Fragment:
        [o.render setFragmentSamplerState:sampler atIndex:slot];
        break;
    case Stage::Compute:
        [o.compute setSamplerState:sampler atIndex:slot];
        break;
    }
}

/// Binds a stage's resources on the open encoder (StageResolvable first); false with `why` when
/// one has no Metal object or comes from a bound descriptor set.
bool BindStage(Owned& o, Stage stage, VkPipelineBindPoint point, const Metal::CompiledShader* shader,
               const char*& why) {
    if (!shader) {
        return true;
    }
    using Kind = Metal::ShaderBinding::Kind;
    std::array<u32, Metal::kMaxBuffers> sizes{};
    for (const auto& b : shader->bindings) {
        if (b.kind == Kind::PushConstants) {
            continue;
        }
        if (o.bound_sets.contains((u32(point) << 16) | b.set) &&
            !o.set_contents.contains((u32(point) << 16) | b.set)) {
            why = "a descriptor set (not pushed) whose contents are not known";
            return false;
        }
        for (u32 e = 0; e < b.count; ++e) {
            const Bound* bound = FindBound(o, point, b.set, b.binding, e);
            if (bound && !bound->ok) {
                why = bound->why;
                return false;
            }
            const u32 slot = b.msl_index + e;
            switch (b.kind) {
            case Kind::Buffer:
                if (bound) {
                    SetBuffer(o, stage, bound->buffer, bound->offset, slot);
                } else {
                    SetBuffer(o, stage, null_buffer, 0, slot);
                }
                if (slot < sizes.size()) {
                    // An unbound slot is an empty buffer: its accesses are dropped.
                    sizes[slot] = u32(std::min<u64>(bound ? bound->size : 0, 0xffffffffu));
                }
                break;
            case Kind::Texture:
            case Kind::StorageTexture:
            case Kind::TexelBuffer:
                SetTexture(o, stage, bound && bound->texture ? bound->texture : null_texture, slot);
                break;
            case Kind::Sampler:
                SetSampler(o, stage, bound && bound->sampler ? bound->sampler : null_sampler, slot);
                break;
            default:
                break;
            }
        }
    }
    if (shader->uses_push_constants) {
        const auto& push = o.push[point == VK_PIPELINE_BIND_POINT_COMPUTE];
        SetBytes(o, stage, push.data(), push.size(), shader->push_constant_buffer);
    }
    if (shader->buffer_size_buffer != ~0u) {
        SetBytes(o, stage, sizes.data(), sizeof(u32) * std::max(1u, shader->num_buffers),
                 shader->buffer_size_buffer);
    }
    return true;
}

// ---- Draws ----

/// The draw's state where a generic pipeline (the renderer's own passes) fixes it instead of
/// taking it from the dynamic state (MetalShadow::FixedState); the game's pipelines take it all
/// from the dynamic state.
struct Effective {
    VkCullModeFlags cull_mode;
    VkFrontFace front_face;
    bool depth_test, depth_write;
    VkCompareOp depth_compare;
    bool stencil_test, depth_bias_enable;
    Owned::Face front, back;
    bool depth_bounds_test;
    float depth_bounds[2];
    bool dynamic_write_masks; ///< the dynamic color write masks apply (not a pipeline's own)
    float blend_constants[4];
};
Effective EffectiveState(const Owned& o) {
    Effective e{o.cull_mode,
                o.front_face,
                o.depth_test,
                o.depth_write,
                o.depth_compare,
                o.stencil_test,
                o.depth_bias_enable,
                o.front,
                o.back,
                o.depth_bounds_test,
                {o.depth_bounds[0], o.depth_bounds[1]},
                o.state_set[size_t(State::ColorWriteMask)],
                {o.blend_constants[0], o.blend_constants[1], o.blend_constants[2],
                 o.blend_constants[3]}};
    if (o.graphics && o.graphics->generic) {
        const MetalShadow::FixedState& f = o.graphics->fixed;
        for (auto [face, fixed] : {std::pair{&e.front, &f.stencil_front},
                                   std::pair{&e.back, &f.stencil_back}}) {
            if (f.stencil_op) {
                face->fail = fixed->fail;
                face->pass = fixed->pass;
                face->depth_fail = fixed->depth_fail;
                face->compare = fixed->compare;
            }
            if (f.stencil_compare_mask) {
                face->compare_mask = fixed->compare_mask;
            }
            if (f.stencil_write_mask) {
                face->write_mask = fixed->write_mask;
            }
            if (f.stencil_reference) {
                face->reference = fixed->reference;
            }
        }
        if (f.depth_bounds) {
            e.depth_bounds_test = f.depth_bounds_test;
        }
        if (f.depth_bounds_values) {
            e.depth_bounds[0] = f.min_depth_bounds;
            e.depth_bounds[1] = f.max_depth_bounds;
        }
        if (f.write_masks) {
            e.dynamic_write_masks = false;
        }
        if (f.cull) {
            e.cull_mode = f.cull_mode;
        }
        if (f.front_face) {
            e.front_face = f.front;
        }
        if (f.depth) {
            e.depth_test = f.depth_test;
        }
        if (f.depth_write_fixed) {
            e.depth_write = f.depth_write;
        }
        if (f.depth_compare_fixed) {
            e.depth_compare = f.depth_compare;
        }
        if (f.blend_constants_fixed) {
            std::copy_n(f.blend_constants, 4, e.blend_constants);
        }
        if (f.stencil) {
            e.stencil_test = f.stencil_test;
        }
        if (f.depth_bias) {
            e.depth_bias_enable = f.depth_bias_enable;
        }
    }
    return e;
}

/// A vertex buffer's stride: a generic pipeline's own unless its strides are dynamic (then as
/// bound, or its own when bound without one); as bound for the game's.
u64 VertexStride(const Owned& o, const Owned::VertexBuffer& vb, u32 location) {
    if (!o.graphics || !o.graphics->generic || location >= o.graphics->strides.size() ||
        (vb.stride && !o.graphics->fixed.strides)) {
        return vb.stride;
    }
    return o.graphics->strides[location];
}

/// Why a draw cannot be encoded in Metal (null: it can).
/// bbport: the render pipeline for the draw (Owned::draw_render): the entry's when its depth and
/// stencil formats are the pass's, else (a plain game pipeline) its variant for the pass's
/// (MetalShadow::FormatVariant; built on first use, the draw falls back meanwhile).
/// BB_METAL_FORMAT_VARIANTS=0 keeps the entry's (for bisecting).
void SelectRender(Owned& o) {
    static const bool variants = [] {
        const char* env = std::getenv("BB_METAL_FORMAT_VARIANTS");
        const bool on = !(env && env[0] == '0');
        if (!on) {
            Print("Metal frame: no depth/stencil format variants (BB_METAL_FORMAT_VARIANTS=0)\n");
        }
        return on;
    }();
    const auto* entry = o.graphics;
    o.draw_render = entry ? entry->render : nullptr;
    o.variant_building = false;
    const Metal::RenderPipeline* p = o.draw_render;
    if (!variants || !p || !entry->source || entry->tess || entry->geometry ||
        (p->depth_pixel_format == o.pass_depth_format &&
         p->stencil_pixel_format == o.pass_stencil_format)) {
        return;
    }
    bool building = false;
    if (const Metal::RenderPipeline* variant = MetalShadow::FormatVariant(
            *entry, o.pass_depth_format, o.pass_stencil_format, building)) {
        o.draw_render = variant;
    }
    o.variant_building = building;
}

const char* DrawBlocker(Owned& o) {
    if (!o.in_pass) {
        return "a draw outside a render pass";
    }
    if (!o.pass_metal) {
        return "a render pass Metal cannot render into";
    }
    const auto* entry = o.graphics;
    if (!entry || !entry->render) {
        return "no Metal pipeline";
    }
    const Metal::RenderPipeline& p = *entry->render; // (a format variant differs in formats only)
    if (p.primitive_emulated && !entry->rect_list) {
        return "a topology Metal lacks";
    }
    if (p.fill_point) {
        return "point polygon mode";
    }
    static const bool flat_last_in_metal = [] {
        const char* env = std::getenv("BB_METAL_FLAT_LAST");
        const bool on = env && env[0] == '1';
        if (on) {
            Print("Metal frame: draws with flat inputs from the last vertex stay in Metal (which "
                  "takes them from the first; BB_METAL_FLAT_LAST=1)\n");
        }
        return on;
    }();
    if (entry->flat_from_last && !flat_last_in_metal) {
        return "flat inputs from the last vertex"; // Metal takes them from the first
    }
    if (p.clip_minus_w_to_w && !p.clip_fixed && !entry->geometry) {
        // (Converted by the vertex function when built so, or by a geometry draw's pass-through.)
        return "clip space z from -w";
    }
    if (o.rasterizer_discard) {
        return "rasterizer discard";
    }
    const Effective e = EffectiveState(o);
    if (e.depth_bounds_test &&
        !(e.depth_bounds[0] >= 0.0f && e.depth_bounds[0] <= e.depth_bounds[1] &&
          e.depth_bounds[1] <= 1.0f)) {
        return "depth bounds outside [0, 1]"; // Metal takes them inside only
    }
    if (o.feedback_loop) {
        return "an attachment feedback loop";
    }
    if (e.dynamic_write_masks) {
        for (u32 i = 0; i < o.write_mask_count; ++i) {
            if (o.write_masks[i] != VkColorComponentFlags(p.key.write_masks[i])) {
                return "a color write mask other than the pipeline's";
            }
        }
    }
    for (u32 i = 0; i < o.pass_color_formats.size(); ++i) {
        const u32 pipeline_format = i < p.num_color_attachments ? p.color_pixel_formats[i] : 0;
        if (pipeline_format != o.pass_color_formats[i]) {
            return "attachment formats other than the pipeline's";
        }
    }
    // Last before the formats: a variant is only built for a draw nothing else keeps out of Metal.
    SelectRender(o);
    const Metal::RenderPipeline& chosen = *o.draw_render;
    if (chosen.depth_pixel_format != o.pass_depth_format ||
        chosen.stencil_pixel_format != o.pass_stencil_format) {
        return o.variant_building
                   ? "depth or stencil formats other than the pipeline's (its variant is being built)"
                   : "depth or stencil formats other than the pipeline's";
    }
    if (chosen.sample_count != o.pass_samples) {
        return "a sample count other than the pipeline's";
    }
    if (o.index_type == VK_INDEX_TYPE_UINT8_EXT) {
        return "8-bit indices";
    }
    return nullptr;
}

/// Indexed strips without primitive restart: Metal always restarts at the all-ones index, which
/// only matters for a draw that really uses vertex 65535 (or 2^32 - 1; lists have never been
/// checked here either). They are drawn in Metal;
/// BB_METAL_STRIP_RESTART=0 sends them to KosmicKrisp (as before; for bisecting).
bool RestartDiffers(const Owned& o) {
    static const bool strict = [] {
        const char* env = std::getenv("BB_METAL_STRIP_RESTART");
        const bool on = env && env[0] == '0';
        if (on) {
            Print("Metal frame: strips without primitive restart go to KosmicKrisp "
                  "(BB_METAL_STRIP_RESTART=0)\n");
        }
        return on;
    }();
    const u32 type = o.graphics->render->primitive_type; // (a format variant's is the same)
    return strict && !o.primitive_restart &&
           (type == MTLPrimitiveTypeTriangleStrip || type == MTLPrimitiveTypeLineStrip);
}

/// The depth/stencil state of the draw.
Metal::DepthStencilKey DepthStencilKeyOf(const Owned& o, const Metal::RenderPipeline& p) {
    const Effective e = EffectiveState(o);
    Metal::DepthStencilKey ds;
    // An aspect the pipeline does not use (a format variant's ignored one) is never tested.
    ds.depth_test = e.depth_test && p.depth_pixel_format && !p.depth_ignored;
    ds.depth_write = e.depth_write && p.depth_pixel_format && !p.depth_ignored;
    ds.depth_compare = vk::CompareOp(e.depth_compare);
    ds.stencil_test = e.stencil_test && p.stencil_pixel_format && !p.stencil_ignored;
    const auto face = [](const Owned::Face& f) {
        Metal::DepthStencilKey::Face out;
        out.fail_op = vk::StencilOp(f.fail);
        out.pass_op = vk::StencilOp(f.pass);
        out.depth_fail_op = vk::StencilOp(f.depth_fail);
        out.compare_op = vk::CompareOp(f.compare);
        out.compare_mask = u8(f.compare_mask);
        out.write_mask = u8(f.write_mask);
        return out;
    };
    ds.front = face(e.front);
    ds.back = face(e.back);
    return ds;
}

/// Encodes the draw state; false with `why` when the draw goes to KosmicKrisp after all
/// (checked before the encoder opens, so a fallback does not end a KosmicKrisp segment first).
bool PrepareDraw(Owned& o, const char*& why) {
    const Metal::RenderPipeline& p = *o.draw_render; // DrawBlocker's choice
    if (!StageResolvable(o, VK_PIPELINE_BIND_POINT_GRAPHICS, p.vs, why) ||
        !StageResolvable(o, VK_PIPELINE_BIND_POINT_GRAPHICS, p.fs, why)) {
        return false;
    }
    for (u32 i = 0; i < p.num_vertex_slots; ++i) {
        const auto& slot = p.vertex_slots[i];
        if (slot.location >= o.vertex_buffers.size()) {
            continue;
        }
        const auto& vb = o.vertex_buffers[slot.location];
        if (vb.buffer && !ResolveBuffer(vb.buffer).buffer) {
            why = "a vertex buffer without a Metal buffer";
            return false;
        }
        if (vb.buffer && !VertexStride(o, vb, slot.location) && !slot.constant) {
            why = "a vertex stride of 0"; // every vertex reads one element: no Metal layout
            return false;
        }
    }
    const Effective effective = EffectiveState(o);
    const auto cull = Metal::VkToMtl::CullMode(vk::CullModeFlags(effective.cull_mode));
    // Scissors first: an empty one draws nothing (and stays out of Metal, which rejects it).
    std::array<MTLScissorRect, 16> scissors;
    const u32 ns = u32(std::min<size_t>(o.scissors.size(), scissors.size()));
    for (u32 i = 0; i < ns; ++i) {
        const VkRect2D& s = o.scissors[i];
        Metal::RasterUtil::ScissorRect r;
        if (!Metal::RasterUtil::ClampScissor(s.offset.x, s.offset.y, s.extent.width,
                                             s.extent.height, o.target_width, o.target_height,
                                             r)) {
            why = "an empty scissor";
            return false;
        }
        scissors[i] = {r.x, r.y, r.width, r.height};
    }
    id<MTLRenderCommandEncoder> enc = RenderEncoder(o);
    if (!enc) {
        why = "no render encoder";
        return false;
    }
    if (!BindStage(o, Stage::Vertex, VK_PIPELINE_BIND_POINT_GRAPHICS, p.vs, why) ||
        !BindStage(o, Stage::Fragment, VK_PIPELINE_BIND_POINT_GRAPHICS, p.fs, why)) {
        return false;
    }
    [enc setRenderPipelineState:(__bridge id<MTLRenderPipelineState>)p.state];
    [enc setTriangleFillMode:MTLTriangleFillMode(p.fill_mode)];
    [enc setDepthClipMode:p.depth_clip ? MTLDepthClipModeClip : MTLDepthClipModeClamp];
    [enc setCullMode:cull.cull_all ? MTLCullModeNone : MTLCullMode(cull.mode)];
    [enc setFrontFacingWinding:MTLWinding(
                                   Metal::RasterUtil::FrontWinding(vk::FrontFace(effective.front_face)))];
    std::array<MTLViewport, 16> viewports;
    const u32 nv = u32(std::min<size_t>(o.viewports.size(), viewports.size()));
    for (u32 i = 0; i < nv; ++i) {
        const VkViewport& v = o.viewports[i];
        const auto m = Metal::RasterUtil::ToMetalViewport(v.x, v.y, v.width, v.height, v.minDepth,
                                                          v.maxDepth);
        viewports[i] = {m.origin_x, m.origin_y, m.width, m.height, m.znear, m.zfar};
    }
    if (nv) {
        [enc setViewports:viewports.data() count:nv];
    }
    if (ns) {
        [enc setScissorRects:scissors.data() count:ns];
    }
    const Metal::DepthStencilKey ds = DepthStencilKeyOf(o, p);
    void* state = MetalShadow::DepthStencil(ds);
    if (!state) {
        why = "no Metal depth/stencil state";
        return false;
    }
    [enc setDepthStencilState:(__bridge id<MTLDepthStencilState>)state];
    if (ds.stencil_test) {
        [enc setStencilFrontReferenceValue:effective.front.reference
                        backReferenceValue:effective.back.reference];
    }
    // Metal's depth bounds test (macOS 26; on the stored depth, as Vulkan's); [0, 1] turns it off.
    if (effective.depth_bounds_test && p.depth_pixel_format &&
        !p.depth_ignored) { // the test needs a depth attachment the pipeline uses
        [enc setDepthTestMinBound:effective.depth_bounds[0] maxBound:effective.depth_bounds[1]];
    } else {
        [enc setDepthTestMinBound:0.0f maxBound:1.0f];
    }
    if (effective.depth_bias_enable) {
        [enc setDepthBias:o.depth_bias[0] slopeScale:o.depth_bias[2] clamp:o.depth_bias[1]];
    } else {
        [enc setDepthBias:0.0f slopeScale:0.0f clamp:0.0f];
    }
    [enc setBlendColorRed:effective.blend_constants[0]
                    green:effective.blend_constants[1]
                     blue:effective.blend_constants[2]
                    alpha:effective.blend_constants[3]];
    for (u32 i = 0; i < p.num_vertex_slots; ++i) {
        const auto& slot = p.vertex_slots[i];
        const auto& vb = slot.location < o.vertex_buffers.size() ? o.vertex_buffers[slot.location]
                                                                 : Owned::VertexBuffer{};
        id<MTLBuffer> buffer = null_buffer;
        u64 offset = 0;
        if (vb.buffer) {
            const BufferRef ref = ResolveBuffer(vb.buffer);
            if (!ref.buffer) {
                why = "a vertex buffer without a Metal buffer";
                return false;
            }
            buffer = ref.buffer;
            offset = ref.offset + vb.offset;
        }
        u64 stride = VertexStride(o, vb, slot.location);
        if (slot.constant || !stride) {
            stride = 16;
        }
        [enc setVertexBuffer:buffer offset:offset attributeStride:stride atIndex:slot.slot];
    }
    return true;
}

MTLPrimitiveType PrimitiveOf(const Owned& o) {
    return o.graphics->rect_list ? MTLPrimitiveTypeTriangle
                                 : MTLPrimitiveType(o.draw_render->primitive_type);
}

/// The draw in Metal if it can be, else in a KosmicKrisp segment (`kk` records it there).
template <typename Encode, typename KkRecord>
void Draw(Owned& o, const char* extra_blocker, Encode&& encode, KkRecord&& kk) {
    const char* why = KkOnly(o)       ? KkOnly(o)
                      : extra_blocker ? extra_blocker
                                      : DrawBlocker(o);
    if (!why && EffectiveState(o).cull_mode == VK_CULL_MODE_FRONT_AND_BACK) {
        why = "culling front and back"; // Metal has no such mode (points and lines still draw)
    }
    if (!why && PrepareDraw(o, why)) {
        if (GpuDiag()) {
            const auto& h = o.graphics->hashes;
            Signpost(o.render, "draw: vertex %#llx pixel %#llx%s", (unsigned long long)h[0],
                     (unsigned long long)h[4], o.graphics->generic ? " (the renderer's own)" : "");
        }
        encode(o.render);
        ++counters.metal_draws;
        return;
    }
    Reason(why);
    ++counters.fallback_draws;
    if (VkCommandBuffer cb = KkWithState(o, VK_PIPELINE_BIND_POINT_GRAPHICS)) {
        kk(cb);
    }
}

template <typename Encode, typename KkRecord>
void Dispatch(Owned& o, Encode&& encode, KkRecord&& kk) {
    const char* why = KkOnly(o)         ? KkOnly(o)
                      : o.compute_metal ? nullptr
                                        : "no Metal pipeline";
    if (!why) {
        StageResolvable(o, VK_PIPELINE_BIND_POINT_COMPUTE, o.compute_metal->shader, why);
    }
    if (!why) {
        id<MTLComputeCommandEncoder> enc = ComputeEncoder(o);
        if (BindStage(o, Stage::Compute, VK_PIPELINE_BIND_POINT_COMPUTE, o.compute_metal->shader,
                      why)) {
            [enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)
                                             o.compute_metal->state];
            const auto& t = o.compute_metal->threads_per_group;
            if (GpuDiag()) {
                Signpost(enc, "dispatch: compute %#llx (%ux%ux%u threads a group)",
                         (unsigned long long)o.compute_metal->key_value, t[0], t[1], t[2]);
            }
            encode(enc, MTLSizeMake(t[0], t[1], t[2]));
            ++counters.metal_dispatches;
            return;
        }
    }
    Reason(why);
    ++counters.fallback_dispatches;
    if (VkCommandBuffer cb = KkWithState(o, VK_PIPELINE_BIND_POINT_COMPUTE)) {
        kk(cb);
    }
}

// ---- Tessellated draws (mtl_tessellation.h) ----

/// The vertex buffers of a vertex (export) kernel's stage input, as the draw has them bound.
struct KernelStreams {
    std::array<Metal::Tessellation::VertexBinding, Vulkan::MaxVertexBufferCount> bindings;
    u32 count{};

    bool Resolve(const Owned& o, std::span<const Metal::VertexBufferSlot> slots, const char*& why) {
        count = 0;
        for (const auto& slot : slots) {
            const auto& vb = slot.location < o.vertex_buffers.size()
                                 ? o.vertex_buffers[slot.location]
                                 : Owned::VertexBuffer{};
            id<MTLBuffer> buffer = null_buffer;
            u64 offset = 0;
            if (vb.buffer) {
                const BufferRef ref = ResolveBuffer(vb.buffer);
                if (!ref.buffer) {
                    why = "a vertex buffer without a Metal buffer";
                    return false;
                }
                if (!vb.stride && !slot.constant) {
                    why = "a vertex stride of 0";
                    return false;
                }
                buffer = ref.buffer;
                offset = ref.offset + vb.offset;
            }
            const u64 stride = slot.constant || !vb.stride ? 16 : vb.stride;
            bindings[count++] = {(__bridge void*)buffer, offset, stride, slot.slot};
        }
        return true;
    }
    std::span<const Metal::Tessellation::VertexBinding> Span() const {
        return {bindings.data(), count};
    }
};

struct TessArgs {
    u32 count{};          ///< vertices (indices) of the draw
    u32 instances{};
    u32 first{};          ///< the first vertex, or the first index
    s32 vertex_offset{};  ///< indexed: the base vertex
    u32 first_instance{};
    bool indexed{};
};

/// Whether the draw's scissors leave nothing to draw (as PrepareDraw finds them).
bool EmptyScissor(const Owned& o) {
    for (const VkRect2D& s : o.scissors) {
        Metal::RasterUtil::ScissorRect r;
        if (!Metal::RasterUtil::ClampScissor(s.offset.x, s.offset.y, s.extent.width,
                                             s.extent.height, o.target_width, o.target_height,
                                             r)) {
            return true;
        }
    }
    return false;
}

/// The vertex and control shaders as compute kernels writing GPU scratch, then the patch draw of
/// the evaluation and fragment shaders in the pass (reopened, loading). False with `why` when the
/// draw goes to KosmicKrisp: checked before the kernels are encoded, except that a render encoder
/// may still fail to open (the kernels' work is then lost, the draw falls back).
bool EncodeTessellated(Owned& o, const TessArgs& a, const char*& why) {
    const Metal::TessellationPipeline& t = *o.graphics->tess;
    const Metal::RenderPipeline& p = *t.render;
    const u32 patches = t.input_control_points ? a.count / t.input_control_points : 0;
    if (!patches || !a.instances) {
        return true; // nothing to draw
    }
    if (a.instances > 1) {
        why = "an instanced tessellated draw";
        return false;
    }
    if (a.indexed && a.vertex_offset < 0) {
        why = "a tessellated draw with a negative vertex offset"; // the stage-in region's origin
        return false;
    }
    Metal::Tessellation::DrawParams params;
    params.num_vertices = patches * t.input_control_points;
    params.first_vertex = a.indexed ? u32(a.vertex_offset) : a.first;
    params.first_instance = a.first_instance;
    if (a.indexed) {
        const BufferRef index = ResolveBuffer(o.index_buffer);
        if (!index.buffer) {
            why = "no Metal index buffer";
            return false;
        }
        const u32 size = o.index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2;
        params.index_type = size == 4 ? 2 : 1;
        params.index_buffer = (__bridge void*)index.buffer;
        params.index_offset = index.offset + o.index_offset + u64(a.first) * size;
    }
    const Metal::TessKernel& kernel = t.vertex[params.index_type];
    if (!kernel.state || !kernel.shader || !t.control.state || !t.control.shader) {
        why = "a tessellation kernel not built";
        return false;
    }
    constexpr auto graphics = VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (!StageResolvable(o, graphics, kernel.shader, why) ||
        !StageResolvable(o, graphics, t.control.shader, why) ||
        !StageResolvable(o, graphics, p.vs, why) || !StageResolvable(o, graphics, p.fs, why)) {
        return false;
    }
    if (EmptyScissor(o)) {
        why = "an empty scissor";
        return false;
    }
    if (!MetalShadow::DepthStencil(DepthStencilKeyOf(o, p))) {
        why = "no Metal depth/stencil state";
        return false;
    }
    KernelStreams streams;
    if (!streams.Resolve(o, std::span{t.vertex_slots.data(), t.num_vertex_slots}, why)) {
        return false;
    }

    // Steps 1 and 2: the kernels (after the encoder before them, the chain).
    const auto plan = Metal::Tessellation::PlanScratch(t, patches);
    id<MTLBuffer> scratch;
    u64 at = 0;
    GpuScratch(o, plan.total, &scratch, &at);
    id<MTLComputeCommandEncoder> enc = ComputeEncoder(o);
    o.draw_kernels = true;
    if (GpuDiag()) {
        const auto& h = o.graphics->hashes;
        Signpost(enc, "tessellation kernels: vertex %#llx hull %#llx, %u patches x %u instances, "
                 "%llu scratch bytes", (unsigned long long)h[0], (unsigned long long)h[1],
                 patches, a.instances, (unsigned long long)plan.total);
    }
    // Every resource was found above (StageResolvable), so the binding succeeds.
    const auto bind = [&](const Metal::CompiledShader& shader) {
        const char* bind_why = nullptr;
        BindStage(o, Stage::Compute, graphics, &shader, bind_why);
    };
    Metal::Tessellation::EncodeKernels((__bridge void*)enc, t, params, streams.Span(),
                                       (__bridge void*)scratch, at, plan, bind, bind);
    // Step 3: the patch draw (the render encoder after the compute one, the chain).
    if (!PrepareDraw(o, why)) {
        return false; // the kernels' output goes unused
    }
    id<MTLRenderCommandEncoder> r = o.render;
    if (GpuDiag()) {
        const auto& h = o.graphics->hashes;
        Signpost(r, "tessellated draw: domain %#llx pixel %#llx, %u patches", (unsigned long long)h[2],
                 (unsigned long long)h[4], patches);
    }
    [r setTessellationFactorBuffer:scratch offset:at + plan.factor_offset instanceStride:0];
    [r setVertexBuffer:scratch offset:at + plan.control_offset atIndex:Metal::kTessInputBuffer];
    [r setVertexBuffer:scratch offset:at + plan.patch_offset atIndex:Metal::kTessPatchBuffer];
    [r setVertexBuffer:scratch offset:at + plan.factor_offset atIndex:Metal::kTessFactorBuffer];
    [r drawPatches:t.output_control_points
                   patchStart:0
                   patchCount:patches
             patchIndexBuffer:nil
       patchIndexBufferOffset:0
                instanceCount:1
                 baseInstance:a.first_instance];
    ++counters.tess_draws;
    return true;
}

/// A geometry-shader draw (mtl_geometry.h): per chunk of primitives the export and geometry
/// shaders as compute kernels writing vertex records and strip indices into GPU scratch, then the
/// strip draws of the pass-through vertex function in the pass (reopened, loading). False with
/// `why` when the draw goes to KosmicKrisp (as EncodeTessellated).
bool EncodeGeometry(Owned& o, const TessArgs& a, const char*& why) {
    const Metal::GeometryPipeline& g = *o.graphics->geometry;
    const Metal::RenderPipeline& p = *g.render;
    const u32 input_vertices = g.info.input_vertices;
    const u32 prims = input_vertices ? a.count / input_vertices : 0;
    if (!prims || !a.instances) {
        return true; // nothing to draw
    }
    if (a.instances > 1) {
        why = "an instanced geometry-shader draw";
        return false;
    }
    if (a.indexed && a.vertex_offset < 0) {
        why = "a geometry-shader draw with a negative vertex offset";
        return false;
    }
    Metal::Tessellation::DrawParams params;
    params.num_vertices = prims * input_vertices;
    params.first_vertex = a.indexed ? u32(a.vertex_offset) : a.first;
    params.first_instance = a.first_instance;
    u64 index_size = 0;
    if (a.indexed) {
        const BufferRef index = ResolveBuffer(o.index_buffer);
        if (!index.buffer) {
            why = "no Metal index buffer";
            return false;
        }
        index_size = o.index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2;
        params.index_type = index_size == 4 ? 2 : 1;
        params.index_buffer = (__bridge void*)index.buffer;
        params.index_offset = index.offset + o.index_offset + u64(a.first) * index_size;
    }
    const Metal::TessKernel& es = g.es[params.index_type];
    if (!es.state || !es.shader || !g.gs.state || !g.gs.shader) {
        why = "a geometry kernel not built";
        return false;
    }
    constexpr auto graphics = VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (!StageResolvable(o, graphics, es.shader, why) ||
        !StageResolvable(o, graphics, g.gs.shader, why) ||
        !StageResolvable(o, graphics, p.vs, why) || !StageResolvable(o, graphics, p.fs, why)) {
        return false;
    }
    if (EmptyScissor(o)) {
        why = "an empty scissor";
        return false;
    }
    if (!MetalShadow::DepthStencil(DepthStencilKeyOf(o, p))) {
        why = "no Metal depth/stencil state";
        return false;
    }
    KernelStreams streams;
    if (!streams.Resolve(o, std::span{g.vertex_slots.data(), g.num_vertex_slots}, why)) {
        return false;
    }
    // Steps 1 and 2 per chunk (the kernels of all chunks in one compute encoder).
    struct Chunk {
        Metal::Geometry::Plan plan;
        id<MTLBuffer> buffer;
        u64 offset{};
    };
    std::vector<Chunk> chunks;
    const auto bind = [&](const Metal::CompiledShader& shader) {
        const char* bind_why = nullptr;
        BindStage(o, Stage::Compute, graphics, &shader, bind_why);
    };
    const u32 chunk_prims = std::max(1u, Metal::Geometry::MaxPrimitivesPerChunk(g));
    for (u32 at = 0; at < prims; at += chunk_prims) {
        const u32 n = std::min(chunk_prims, prims - at);
        Chunk c;
        c.plan = Metal::Geometry::PlanScratch(g, n);
        id<MTLBuffer> buffer;
        GpuScratch(o, c.plan.total, &buffer, &c.offset);
        c.buffer = buffer;
        Metal::Tessellation::DrawParams draw = params;
        draw.num_vertices = n * input_vertices;
        const u64 first_vertex = u64(at) * input_vertices;
        if (a.indexed) {
            draw.index_offset = params.index_offset + first_vertex * index_size;
        } else {
            draw.first_vertex = params.first_vertex + u32(first_vertex);
        }
        id<MTLComputeCommandEncoder> enc = ComputeEncoder(o);
        o.draw_kernels = true;
        if (GpuDiag()) {
            const auto& h = o.graphics->hashes;
            Signpost(enc, "geometry kernels: vertex %#llx geometry %#llx", (unsigned long long)h[0],
                     (unsigned long long)h[3]);
        }
        Metal::Geometry::EncodeKernels((__bridge void*)enc, g, draw, at, streams.Span(),
                                       (__bridge void*)c.buffer, c.offset, c.plan, bind, bind);
        chunks.push_back(c);
    }
    // Step 3: the strip draws (the render encoder after the compute one, the chain). 0xFFFFFFFF
    // restarts a strip (Metal always restarts at it).
    if (!PrepareDraw(o, why)) {
        return false; // the kernels' output goes unused
    }
    id<MTLRenderCommandEncoder> r = o.render;
    for (const Chunk& c : chunks) {
        // The pass-through function reads the records at its buffer 0.
        [r setVertexBuffer:c.buffer offset:c.offset + c.plan.record_offset atIndex:0];
        [r drawIndexedPrimitives:MTLPrimitiveType(p.primitive_type)
                      indexCount:u64(c.plan.prims) * g.info.IndexSlots()
                       indexType:MTLIndexTypeUInt32
                     indexBuffer:c.buffer
               indexBufferOffset:c.offset + c.plan.index_offset
                   instanceCount:1
                      baseVertex:0
                    baseInstance:a.first_instance];
    }
    ++counters.geometry_draws;
    return true;
}

/// BB_METAL_TESS=0 / BB_METAL_GEOMETRY=0: those draws stay on KosmicKrisp (for bisecting a broken
/// picture). Read once, logged once.
bool KindInMetal(const char* variable, const char* what) {
    const char* env = std::getenv(variable);
    const bool off = env && env[0] == '0';
    if (off) {
        Print("Metal frame: %s draws stay on KosmicKrisp (%s=0)\n", what, variable);
    }
    return !off;
}
bool TessInMetal() {
    static const bool on = KindInMetal("BB_METAL_TESS", "tessellated");
    return on;
}
/// bbport: how long a draw's recording waits for a tessellated or geometry-shader pipeline's
/// Metal build still pending (MetalShadow::FindGraphicsWaiting) rather than sending its draws to
/// KosmicKrisp's emulation of those stages, which a heavy scene can hang the GPU with (its
/// generated geometry outgrows the driver's heap; patch 0010). BB_METAL_EMULATED_WAIT_MS (0: no
/// wait), 1000 by default; none when both kinds stay on KosmicKrisp anyway.
u32 EmulatedWaitMs();

bool GeometryInMetal() {
    static const bool on = KindInMetal("BB_METAL_GEOMETRY", "geometry-shader");
    return on;
}

u32 EmulatedWaitMs() {
    static const u32 ms = [] {
        if (!TessInMetal() && !GeometryInMetal()) {
            return 0u;
        }
        const char* env = std::getenv("BB_METAL_EMULATED_WAIT_MS");
        // 1 s by default: with KosmicKrisp patch 0010 its emulation no longer hangs the GPU
        // when its heap runs out, and several new pipelines at a load used to add up to tens
        // of seconds while their Metal builds were compiled from source.
        const u32 value = env ? u32(std::strtoul(env, nullptr, 10)) : 1000u;
        Print("Metal frame: a tessellated or geometry-shader draw waits up to %u ms for its Metal "
              "pipeline (BB_METAL_EMULATED_WAIT_MS)\n", value);
        return value;
    }();
    return ms;
}

/// A tessellated or geometry-shader draw in Metal if it can be, else in a KosmicKrisp segment.
template <typename KkRecord>
void DrawTessellated(Owned& o, const TessArgs& a, KkRecord&& kk) {
    const char* why = KkOnly(o)                              ? KkOnly(o)
                      : o.graphics->tess && !TessInMetal()   ? "tessellation (BB_METAL_TESS=0)"
                      : !o.graphics->tess && !GeometryInMetal() ? "a geometry shader (BB_METAL_GEOMETRY=0)"
                                                             : DrawBlocker(o);
    if (!why && o.cull_mode == VK_CULL_MODE_FRONT_AND_BACK) {
        why = "culling front and back";
    }
    if (!why && (o.graphics->tess ? EncodeTessellated(o, a, why) : EncodeGeometry(o, a, why))) {
        ++counters.metal_draws;
        return;
    }
    Reason(why);
    ++counters.fallback_draws;
    if (VkCommandBuffer cb = KkWithState(o, VK_PIPELINE_BIND_POINT_GRAPHICS)) {
        kk(cb);
    }
}

// ---- Render passes ----

MTLLoadAction Load(VkAttachmentLoadOp op) {
    return op == VK_ATTACHMENT_LOAD_OP_CLEAR       ? MTLLoadActionClear
           : op == VK_ATTACHMENT_LOAD_OP_DONT_CARE ? MTLLoadActionDontCare
                                                   : MTLLoadActionLoad;
}
/// The clear color as Metal reads it for the pixel format (integer formats: the integer values).
MTLClearColor ClearColor(const VkClearColorValue& c, MTLPixelFormat format) {
    switch (format) {
    case MTLPixelFormatR8Uint: case MTLPixelFormatR16Uint: case MTLPixelFormatR32Uint:
    case MTLPixelFormatRG8Uint: case MTLPixelFormatRG16Uint: case MTLPixelFormatRG32Uint:
    case MTLPixelFormatRGBA8Uint: case MTLPixelFormatRGBA16Uint: case MTLPixelFormatRGBA32Uint:
    case MTLPixelFormatRGB10A2Uint:
        return MTLClearColorMake(c.uint32[0], c.uint32[1], c.uint32[2], c.uint32[3]);
    case MTLPixelFormatR8Sint: case MTLPixelFormatR16Sint: case MTLPixelFormatR32Sint:
    case MTLPixelFormatRG8Sint: case MTLPixelFormatRG16Sint: case MTLPixelFormatRG32Sint:
    case MTLPixelFormatRGBA8Sint: case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA32Sint:
        return MTLClearColorMake(c.int32[0], c.int32[1], c.int32[2], c.int32[3]);
    default:
        return MTLClearColorMake(c.float32[0], c.float32[1], c.float32[2], c.float32[3]);
    }
}

/// The Metal descriptors of the begun pass; false when Metal cannot render into it.
bool BuildPass(Owned& o) {
    const VkRenderingInfo& info = o.pass_info;
    // Suspended / resumed passes skip their store / load operations: not modelled here.
    if (info.viewMask || info.layerCount > 1 || o.pass_colors.size() > 8 ||
        info.renderArea.offset.x || info.renderArea.offset.y ||
        (info.flags & (VK_RENDERING_SUSPENDING_BIT | VK_RENDERING_RESUMING_BIT))) {
        return false;
    }
    MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
    u32 width = info.renderArea.offset.x + info.renderArea.extent.width;
    u32 height = info.renderArea.offset.y + info.renderArea.extent.height;
    u32 samples = 0;
    const auto note_texture = [&](id<MTLTexture> t, u32 level) {
        width = u32(std::max<NSUInteger>(1, t.width >> level));
        height = u32(std::max<NSUInteger>(1, t.height >> level));
        if (samples && samples != t.sampleCount) {
            return false;
        }
        samples = u32(t.sampleCount);
        return true;
    };
    o.pass_color_formats.fill(0);
    o.pass_depth_format = o.pass_stencil_format = 0;
    for (u32 i = 0; i < o.pass_colors.size(); ++i) {
        const VkRenderingAttachmentInfo& a = o.pass_colors[i];
        if (!a.imageView) {
            continue;
        }
        if (a.resolveMode != VK_RESOLVE_MODE_NONE) {
            return false;
        }
        {
            // A format stored swizzled: Metal's pipelines would write its channels in the wrong
            // places (KosmicKrisp remaps them).
            std::scoped_lock lock{objects_mutex};
            const auto f = view_formats.find(a.imageView);
            if (f == view_formats.end() ||
                !(Metal::VkToMtl::FormatSwizzle(vk::Format(f->second)) ==
                  Metal::VkToMtl::IdentitySwizzle)) {
                return false;
            }
        }
        const ViewTexture t = ResolveView(a.imageView, BbMetal::ViewTextureRender);
        if (!t.texture || !note_texture(t.texture, t.level)) {
            return false;
        }
        rp.colorAttachments[i].texture = t.texture;
        rp.colorAttachments[i].level = t.level;
        rp.colorAttachments[i].slice = t.slice;
        rp.colorAttachments[i].loadAction = Load(a.loadOp);
        // Decided at the encoder's end (CloseRender): Vulkan's store operation at the pass's
        // end, a store where the pass is split (reopened) before it.
        rp.colorAttachments[i].storeAction = MTLStoreActionUnknown;
        rp.colorAttachments[i].clearColor = ClearColor(a.clearValue.color, t.texture.pixelFormat);
        o.pass_color_formats[i] = u32(t.texture.pixelFormat);
    }
    if (o.has_depth && o.pass_depth.imageView) {
        const VkRenderingAttachmentInfo& a = o.pass_depth;
        if (a.resolveMode != VK_RESOLVE_MODE_NONE) {
            return false;
        }
        const ViewTexture t = ResolveView(a.imageView, BbMetal::ViewTextureRender);
        if (!t.texture || !note_texture(t.texture, t.level)) {
            return false;
        }
        rp.depthAttachment.texture = t.texture;
        rp.depthAttachment.level = t.level;
        rp.depthAttachment.slice = t.slice;
        rp.depthAttachment.loadAction = Load(a.loadOp);
        rp.depthAttachment.storeAction = MTLStoreActionUnknown;
        rp.depthAttachment.clearDepth = a.clearValue.depthStencil.depth;
        o.pass_depth_format = u32(t.texture.pixelFormat);
    }
    if (o.has_stencil && o.pass_stencil.imageView) {
        const VkRenderingAttachmentInfo& a = o.pass_stencil;
        if (a.resolveMode != VK_RESOLVE_MODE_NONE) {
            return false;
        }
        const ViewTexture t = ResolveView(a.imageView, BbMetal::ViewTextureRender);
        if (!t.texture || !note_texture(t.texture, t.level)) {
            return false;
        }
        if (!Metal::VkToMtl::HasStencilPixelFormat(u32(t.texture.pixelFormat))) {
            return false;
        }
        rp.stencilAttachment.texture = t.texture;
        rp.stencilAttachment.level = t.level;
        rp.stencilAttachment.slice = t.slice;
        rp.stencilAttachment.loadAction = Load(a.loadOp);
        rp.stencilAttachment.storeAction = MTLStoreActionUnknown;
        rp.stencilAttachment.clearStencil = a.clearValue.depthStencil.stencil;
        o.pass_stencil_format = u32(t.texture.pixelFormat);
    }
    if (!samples) {
        rp.renderTargetWidth = std::max(1u, width);
        rp.renderTargetHeight = std::max(1u, height);
        rp.defaultRasterSampleCount = 1;
        samples = 1;
    }
    if (info.renderArea.extent.width < width || info.renderArea.extent.height < height) {
        return false; // Metal would clear (and store) the whole attachment
    }
    o.target_width = width;
    o.target_height = height;
    o.pass_samples = samples;
    o.pass_first = rp;
    MTLRenderPassDescriptor* again = [rp copy];
    for (u32 i = 0; i < o.pass_colors.size(); ++i) {
        if (again.colorAttachments[i].texture) {
            again.colorAttachments[i].loadAction = MTLLoadActionLoad;
        }
    }
    if (again.depthAttachment.texture) {
        again.depthAttachment.loadAction = MTLLoadActionLoad;
    }
    if (again.stencilAttachment.texture) {
        again.stencilAttachment.loadAction = MTLLoadActionLoad;
    }
    o.pass_reopen = again;
    return true;
}

// ---- The queue watchdog: what KosmicKrisp's queue waits for when it is behind ----

/// A KosmicKrisp submission of ours, or a present (KosmicKrisp's own submission, which the
/// layer sees only as the present), for the watchdog: the `to_metal` value reached once it is
/// done (a present: that of the next submission, which runs after it) and what it waits for
/// (each wait's shared event, kept so it can be read later).
struct KkSubmitRecord {
    struct Wait {
        VkSemaphore semaphore{};
        u64 value{};
        id<MTLSharedEvent> event; ///< nil: not exported (a binary semaphore)
    };
    u64 seq{};
    u64 to_metal{};
    double submitted{}; ///< NowSeconds()
    u32 command_buffers{};
    bool present = false;
    u64 to_kk_wait{}; ///< the value of our queue's signal it waits for; 0: none
    std::vector<Wait> waits;
};

/// Never destroyed: the watchdog thread may still run at exit.
std::mutex& watch_mutex = *new std::mutex;
std::deque<KkSubmitRecord>& watch_records = *new std::deque<KkSubmitRecord>; ///< oldest first
u64 watch_seq = 0;                  ///< under watch_mutex
std::atomic<bool> watch_enabled{false};

/// BB_METAL_WATCHDOG=0: no watchdog.
bool WatchdogWanted() {
    const char* v = std::getenv("BB_METAL_WATCHDOG");
    return !(v && v[0] == '0');
}

/// Keeps a record (the queue's submissions are serialized; the lock is for the watchdog).
void WatchRecord(KkSubmitRecord&& rec, const VkSemaphore* semaphores, const u64* values,
                 u32 count) {
    for (u32 i = 0; i < count; ++i) {
        VkExportMetalSharedEventInfoEXT e{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT,
            .semaphore = semaphores[i],
        };
        Export(reinterpret_cast<VkBaseOutStructure*>(&e));
        rec.waits.push_back({semaphores[i], values ? values[i] : 0, e.mtlSharedEvent});
    }
    rec.submitted = NowSeconds();
    std::scoped_lock lock{watch_mutex};
    rec.seq = ++watch_seq;
    if (watch_records.size() >= 4096) {
        watch_records.pop_front(); // (the watchdog pops what is done: only if it stopped)
    }
    watch_records.push_back(std::move(rec));
}

/// Every 250 ms: when the oldest KosmicKrisp submission (or present) not done is 1 s old, and
/// again at 3 s, what it and the five after it wait for, read from their shared events then;
/// and how long it took once it is done.
void WatchQueues(id<MTLSharedEvent> metal_event, id<MTLSharedEvent> kk_event) {
    u64 reported = 0;      ///< the seq of the record reported, while it is not done
    double reported_at = 0.0; ///< its submission
    bool reported_present = false;
    u32 stage = 0, stalls = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        std::vector<KkSubmitRecord> behind;
        u64 latest = 0;
        double age = 0.0;
        bool done_reported = false;
        {
            std::scoped_lock lock{watch_mutex};
            const u64 reached = metal_event.signaledValue;
            while (!watch_records.empty() && watch_records.front().to_metal <= reached) {
                done_reported = done_reported || watch_records.front().seq == reported;
                watch_records.pop_front();
            }
            // A present counts from the submission after it (it is done with that one): without
            // one (the game paused, minimised) it is not behind.
            if (!done_reported && !watch_records.empty() &&
                !(watch_records.front().present && watch_records.size() < 2)) {
                const KkSubmitRecord& front = watch_records.front();
                const double since =
                    front.present ? watch_records[1].submitted : front.submitted;
                age = NowSeconds() - since;
                if (front.seq != reported) {
                    stage = 0;
                }
                const u32 due = age >= 3.0 ? 2 : age >= 1.0 ? 1 : 0;
                if (due > stage && (stage > 0 || stalls < 100)) {
                    if (stage == 0) {
                        ++stalls;
                    }
                    stage = due;
                    reported = front.seq;
                    reported_at = since;
                    reported_present = front.present;
                    latest = watch_records.back().to_metal;
                    for (const KkSubmitRecord& r : watch_records) {
                        if (behind.size() >= 6) {
                            break;
                        }
                        behind.push_back(r);
                    }
                }
            }
        }
        if (done_reported) {
            Print("Metal frame: KosmicKrisp's %s reported behind was done %.0f ms after it was "
                  "submitted (a present: the submission after it; to within 250 ms)\n",
                  reported_present ? "present" : "submission", (NowSeconds() - reported_at) * 1e3);
            reported = 0;
            stage = 0;
            continue;
        }
        if (behind.empty()) {
            continue;
        }
        @autoreleasepool {
            const double now = NowSeconds();
            Print("Metal frame: KosmicKrisp's queue is behind: its oldest %s not done was "
                  "submitted (a present: the submission after it) %.0f ms ago (done at to_metal %llu, the latest %llu); to_metal at "
                  "%llu, to_kk at %llu\n",
                  behind.front().present ? "present" : "submission", age * 1e3,
                  (unsigned long long)behind.front().to_metal, (unsigned long long)latest,
                  (unsigned long long)metal_event.signaledValue,
                  (unsigned long long)kk_event.signaledValue);
            bool oldest_all_reached = true, oldest_unknown = false;
            for (size_t n = 0; n < behind.size(); ++n) {
                const KkSubmitRecord& r = behind[n];
                std::string waits;
                char line[192];
                if (r.to_kk_wait) {
                    const u64 at = kk_event.signaledValue;
                    std::snprintf(line, sizeof(line), " our queue %llu (at %llu: %s);",
                                  (unsigned long long)r.to_kk_wait, (unsigned long long)at,
                                  at >= r.to_kk_wait ? "reached" : "NOT YET");
                    waits += line;
                    oldest_all_reached = oldest_all_reached && (n > 0 || at >= r.to_kk_wait);
                }
                for (const KkSubmitRecord::Wait& w : r.waits) {
                    if (!w.event) {
                        std::snprintf(line, sizeof(line), " semaphore %p (binary: unknown);",
                                      (void*)w.semaphore);
                        oldest_unknown = oldest_unknown || n == 0;
                    } else {
                        const u64 at = w.event.signaledValue;
                        std::snprintf(line, sizeof(line),
                                      " semaphore %p value %llu (its event at %llu: %s);",
                                      (void*)w.semaphore, (unsigned long long)w.value,
                                      (unsigned long long)at,
                                      at >= w.value ? "reached" : "NOT YET");
                        oldest_all_reached = oldest_all_reached && (n > 0 || at >= w.value);
                    }
                    waits += line;
                }
                if (r.present) {
                    Print("    present, submitted %.0f ms ago, after to_metal %llu; waits for:%s\n",
                          (now - r.submitted) * 1e3, (unsigned long long)(r.to_metal - 1),
                          waits.empty() ? " nothing" : waits.c_str());
                } else {
                    Print("    submission to_metal %llu: %u command buffers, submitted %.0f ms "
                          "ago; waits for:%s\n",
                          (unsigned long long)r.to_metal, r.command_buffers,
                          (now - r.submitted) * 1e3, waits.empty() ? " nothing" : waits.c_str());
                }
            }
            if (oldest_all_reached) {
                Print("    (every wait of the oldest that can be read is reached%s: what holds it "
                      "is not one of those semaphores; a drawable a present waits for, or the "
                      "GPU)\n",
                      oldest_unknown ? ", binary ones unknown" : "");
            }
        }
    }
}

void StartWatchdog() {
    static bool started = false;
    if (started) {
        // A second device: the thread keeps the first one's events.
        watch_enabled.store(false, std::memory_order_relaxed);
        return;
    }
    if (WatchdogWanted()) {
        started = true;
        watch_enabled.store(true, std::memory_order_relaxed);
        std::thread(WatchQueues, to_metal_event, to_kk_event).detach();
    }
}

} // namespace

// ---- The interface ----

void Shutdown() {
    @autoreleasepool {
        if (!queue || !kk_residency) {
            return;
        }
        // (No wait for the queue: a gate command buffer waiting for KosmicKrisp could hang the
        // exit; the queue's residency sets apply to the command buffers committed after.)
        [queue removeResidencySet:kk_residency];
        kk_residency = nil;
    }
}

bool Init(VkDevice vkdev, PFN_vkGetDeviceProcAddr driver_proc) {
    vk_device = vkdev;
#define BB_RESOLVE(name)                                                                           \
    driver.name = reinterpret_cast<PFN_##name>(driver_proc(vkdev, #name));
    BB_DRIVER_FUNCTIONS(BB_RESOLVE)
#undef BB_RESOLVE
    export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        driver_proc(vkdev, "vkExportMetalObjectsEXT"));
    if (!export_objects || !driver.vkQueueSubmit2 || !driver.vkCmdBeginRendering ||
        !driver.vkCmdPushDescriptorSetKHR || !driver.vkCmdBindVertexBuffers2 ||
        !driver.vkCmdSetViewportWithCount || !driver.vkGetSemaphoreCounterValue) {
        Print("Metal frame: the driver lacks what the frame encoder needs; it stays off\n");
        return false;
    }
    if (@available(macOS 26.0, *)) {
        @autoreleasepool {
            VkExportMetalDeviceInfoEXT dev{.sType = VK_STRUCTURE_TYPE_EXPORT_METAL_DEVICE_INFO_EXT};
            Export(reinterpret_cast<VkBaseOutStructure*>(&dev));
            device = dev.mtlDevice;
            if (!device) {
                Print("Metal frame: KosmicKrisp exports no device; the frame encoder stays off\n");
                return false;
            }
            if (MTLCreateSystemDefaultDevice() != device) {
                Print("Metal frame: KosmicKrisp's device is not the system default one (the "
                      "shadow build's); the frame encoder stays off\n");
                return false;
            }
            BbMetal::ExportResidencySet kk_set{};
            Export(reinterpret_cast<VkBaseOutStructure*>(&kk_set));
            if (!kk_set.mtlResidencySet) {
                Print("Metal frame: KosmicKrisp exports no residency set (patch 0009); the frame "
                      "encoder stays off\n");
                return false;
            }
            MTLResidencySetDescriptor* rd = [MTLResidencySetDescriptor new];
            rd.label = @"bbport Metal frame";
            NSError* error = nil;
            residency = [device newResidencySetWithDescriptor:rd error:&error];
            if (!residency) {
                Print("Metal frame: no residency set (%s); the frame encoder stays off\n",
                      error ? error.localizedDescription.UTF8String : "unknown");
                return false;
            }
            // Command buffers are made when a Metal segment begins and committed at the
            // submission; uncommitted ones count against the queue's limit, and a recording
            // thread that reaches it waits: room for many frames' worth of segments.
            queue = [device newCommandQueueWithMaxCommandBufferCount:8192];
            if (!queue) {
                Print("Metal frame: no command queue; the frame encoder stays off\n");
                return false;
            }
            queue.label = @"bbport Metal frame";
            kk_residency = (__bridge id<MTLResidencySet>)kk_set.mtlResidencySet;
            [queue addResidencySet:kk_residency];
            [queue addResidencySet:residency];
            // Zeros: what a null descriptor reads.
            null_buffer = [device newBufferWithLength:65536 options:MTLResourceStorageModeShared];
            std::memset(null_buffer.contents, 0, null_buffer.length);
            MTLTextureDescriptor* td =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                   width:1
                                                                  height:1
                                                               mipmapped:NO];
            td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            td.storageMode = MTLStorageModePrivate;
            null_texture = [device newTextureWithDescriptor:td];
            MTLSamplerDescriptor* sd = [MTLSamplerDescriptor new];
            null_sampler = [device newSamplerStateWithDescriptor:sd];
            MakeResident(null_buffer);
            MakeResident(null_texture);
            // The chain fence, updated once so that no wait depends on whether it was used yet.
            chain = [device newFence];
            {
                id<MTLCommandBuffer> prime = [queue commandBuffer];
                id<MTLBlitCommandEncoder> blit = [prime blitCommandEncoder];
                [blit updateFence:chain];
                [blit endEncoding];
                [prime commit];
                [prime waitUntilCompleted];
            }
            const VkSemaphoreTypeCreateInfo type{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = 1, // a discarded segment's value: reusable at once
            };
            const VkSemaphoreCreateInfo sem{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                            .pNext = &type};
            if (driver.vkCreateSemaphore(vkdev, &sem, nullptr, &kk_timeline) != VK_SUCCESS) {
                Print("Metal frame: no timeline semaphore; the frame encoder stays off\n");
                return false;
            }
            // The two directions between the queues: timelines from 0 (nothing to wait for).
            const VkSemaphoreTypeCreateInfo zero{
                .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
                .initialValue = 0,
            };
            const VkSemaphoreCreateInfo zero_sem{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                                                 .pNext = &zero};
            const auto event_of = [&](VkSemaphore semaphore) -> id<MTLSharedEvent> {
                VkExportMetalSharedEventInfoEXT e{
                    .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT,
                    .semaphore = semaphore,
                };
                Export(reinterpret_cast<VkBaseOutStructure*>(&e));
                return e.mtlSharedEvent;
            };
            if (driver.vkCreateSemaphore(vkdev, &zero_sem, nullptr, &to_metal_sem) != VK_SUCCESS ||
                driver.vkCreateSemaphore(vkdev, &zero_sem, nullptr, &to_kk_sem) != VK_SUCCESS ||
                !(to_metal_event = event_of(to_metal_sem)) ||
                !(to_kk_event = event_of(to_kk_sem))) {
                Print("Metal frame: no shared events between the queues; the frame encoder stays "
                      "off\n");
                return false;
            }
            StartWatchdog();
            Print("Metal frame: encoding owned command buffers in Metal on %s, on a queue of its "
                  "own; what it does not encode goes to KosmicKrisp segments (BB_METAL_FRAME=1)\n",
                  device.name.UTF8String);
            return true;
        }
    }
    Print("Metal frame: needs macOS 26; the frame encoder stays off\n");
    return false;
}

Owned* Begin(VkCommandBuffer command_buffer) {
    auto* o = new Owned;
    o->vk = command_buffer;
    return o;
}

void End(Owned& o) {
    @autoreleasepool {
        EndMetal(o);
        EndKk(o);
    }
}

void Discard(Owned* o) {
    if (!o) {
        return;
    }
    @autoreleasepool {
        End(*o);
        for (Segment& s : o->segments) {
            if (s.kk) {
                s.kk->done_at.store(1, std::memory_order_release); // never submitted
            }
        }
        if (o->retire) {
            o->retire->pending.fetch_add(1, std::memory_order_relaxed);
            Release(o->retire); // never committed: back at once
        }
    }
    delete o;
}

void Presenting(const VkPresentInfoKHR* info) {
    if (!info || !watch_enabled.load(std::memory_order_relaxed)) {
        return;
    }
    // KosmicKrisp's present submission runs before the next submission of ours: done when that
    // one's `to_metal` value is reached.
    WatchRecord({.to_metal = to_metal_value + 1, .present = true}, info->pWaitSemaphores, nullptr,
                info->waitSemaphoreCount);
}

VkResult Submit(VkQueue vkqueue, const Batch* batches, uint32_t count, VkFence fence) {
    @autoreleasepool {
        // Each KosmicKrisp submission makes new allocations resident (its residency set is on
        // our queue too), signals `to_metal` for our command buffers after it, and waits for
        // `to_kk` when ours ran since the last one; a run of our command buffers waits for
        // `to_metal` first when KosmicKrisp ran since, and signals `to_kk` at its end.
        std::vector<VkCommandBufferSubmitInfo> kk_run;
        std::vector<KkBuffer*> kk_run_buffers;
        std::vector<id<MTLCommandBuffer>> metal_run;
        std::vector<std::shared_ptr<Retire>> metal_retires;
        std::vector<std::shared_ptr<Retire>> held; ///< released when the submission is done
        VkResult result = VK_SUCCESS;
        SubmitTimes times;

        const auto submit = [&](const VkSemaphoreSubmitInfo* waits, u32 wait_count,
                                const VkSemaphoreSubmitInfo* signals, u32 signal_count,
                                VkFence f) {
            std::vector<VkSemaphoreSubmitInfo> all_waits(waits, waits + wait_count);
            const bool waited_metal = metal_ahead;
            if (metal_ahead) {
                all_waits.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                     .semaphore = to_kk_sem,
                                     .value = to_kk_value,
                                     .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
                metal_ahead = false;
            }
            std::vector<VkSemaphoreSubmitInfo> all_signals(signals, signals + signal_count);
            if (!kk_run_buffers.empty()) {
                const u64 value = ++kk_timeline_next;
                for (KkBuffer* b : kk_run_buffers) {
                    b->done_at.store(value, std::memory_order_release);
                }
                all_signals.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                       .semaphore = kk_timeline,
                                       .value = value,
                                       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
            }
            all_signals.push_back({.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                                   .semaphore = to_metal_sem,
                                   .value = to_metal_value + 1,
                                   .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT});
            const VkSubmitInfo2 info{
                .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                .waitSemaphoreInfoCount = u32(all_waits.size()),
                .pWaitSemaphoreInfos = all_waits.data(),
                .commandBufferInfoCount = u32(kk_run.size()),
                .pCommandBufferInfos = kk_run.data(),
                .signalSemaphoreInfoCount = u32(all_signals.size()),
                .pSignalSemaphoreInfos = all_signals.data(),
            };
            const auto t = SubmitTimes::Clock::now();
            const VkResult r = driver.vkQueueSubmit2(vkqueue, 1, &info, f);
            // Without command buffers KosmicKrisp's submission is its waits, signals and the
            // residency of every allocation (kk_device_make_resources_resident).
            if (kk_run.empty()) {
                times.Add(times.empty_ms, times.empty_max_ms, times.empty, SubmitTimes::Since(t));
            } else {
                times.Add(times.kk_ms, times.kk_max_ms, times.kk, SubmitTimes::Since(t));
            }
            if (r == VK_SUCCESS) {
                ++to_metal_value;
                kk_ahead = true;
                if (watch_enabled.load(std::memory_order_relaxed)) {
                    std::vector<VkSemaphore> sems(wait_count);
                    std::vector<u64> values(wait_count);
                    for (u32 i = 0; i < wait_count; ++i) {
                        sems[i] = waits[i].semaphore;
                        values[i] = waits[i].value;
                    }
                    WatchRecord({.to_metal = to_metal_value,
                                 .command_buffers = u32(kk_run.size()),
                                 .to_kk_wait = waited_metal ? to_kk_value : 0},
                                sems.data(), values.data(), wait_count);
                }
            } else {
                // Nothing of it reached the queue: no value to wait for, our work still ahead.
                result = r;
                metal_ahead = metal_ahead || waited_metal;
                for (KkBuffer* b : kk_run_buffers) {
                    b->done_at.store(1, std::memory_order_release); // reusable
                }
            }
            kk_run.clear();
            kk_run_buffers.clear();
        };
        const auto flush_kk = [&] {
            if (!kk_run.empty()) {
                submit(nullptr, 0, nullptr, 0, VK_NULL_HANDLE);
            }
        };
        const auto flush_metal = [&] {
            if (metal_run.empty()) {
                return;
            }
            const auto t = SubmitTimes::Clock::now();
            if (kk_ahead) {
                // After KosmicKrisp's work: a command buffer that waits for it and passes the
                // chain on (every encoder after it waits for the chain).
                id<MTLCommandBuffer> gate = [queue commandBufferWithUnretainedReferences];
                gate.label = @"bbport Metal frame: after KosmicKrisp";
                [gate encodeWaitForEvent:to_metal_event value:to_metal_value];
                id<MTLBlitCommandEncoder> blit = [gate blitCommandEncoder];
                [blit waitForFence:chain];
                [blit updateFence:chain];
                [blit endEncoding];
                const double gate_committed = NowSeconds();
                [gate addCompletedHandler:^(id<MTLCommandBuffer> done) {
                  NoteSlow(done, gate_committed, "the wait for KosmicKrisp's work before a run", 0,
                           1);
                }];
                [gate commit];
                kk_ahead = false;
            }
            // The run's end: after its last encoder (the chain), for KosmicKrisp.
            id<MTLCommandBuffer> last = metal_run.back();
            {
                id<MTLBlitCommandEncoder> blit = [last blitCommandEncoder];
                [blit waitForFence:chain];
                [blit updateFence:chain];
                [blit endEncoding];
            }
            [last encodeSignalEvent:to_kk_event value:++to_kk_value];
            metal_ahead = true;
            for (auto& r : metal_retires) {
                r->pending.fetch_add(1, std::memory_order_relaxed);
            }
            std::vector<std::shared_ptr<Retire>> retires = std::move(metal_retires);
            const u32 run_size = u32(metal_run.size());
            u32 run_index = 0;
            for (id<MTLCommandBuffer> cb : metal_run) {
                const bool is_last = cb == last;
                const double committed = NowSeconds();
                const u32 index = run_index++;
                [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                  NoteSlow(done, committed, "a run's command buffer", index, run_size);
                }];
                // Held by the handler: the queue keeps an uncommitted command buffer, not a
                // committed one, alive for its handlers (they run once it completed).
                [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                  if (done.status == MTLCommandBufferStatusError) {
                      if (!gpu_failed.exchange(true)) {
                          Print("Metal frame: after this GPU error the rest goes to KosmicKrisp "
                                "(BB_METAL_KEEP_AFTER_ERROR=1: Metal goes on)\n");
                      }
                      if (counters.gpu_errors++ < 4) {
                          Print("Metal frame: a Metal command buffer failed on the GPU: %s\n",
                                done.error ? done.error.localizedDescription.UTF8String
                                           : "unknown error");
                          PrintEncoderStatus(done);
                      }
                  }
                }];
                if (is_last) {
                    // The chain orders every encoder of the run before the last one's end.
                    const u64 signal = to_kk_value;
                    [cb addCompletedHandler:^(id<MTLCommandBuffer> done) {
                      if (done.status == MTLCommandBufferStatusError &&
                          to_kk_event.signaledValue < signal) {
                          // Its commands stopped at the fault, the signal with them: KosmicKrisp
                          // would wait for it for ever.
                          to_kk_event.signaledValue = signal;
                      }
                      for (auto& r : retires) {
                          Release(r);
                      }
                    }];
                }
                [cb commit];
            }
            times.Add(times.metal_ms, times.metal_max_ms, times.metal, SubmitTimes::Since(t));
            counters.metal_buffers += metal_run.size();
            metal_run.clear();
            metal_retires.clear();
        };

        for (u32 b = 0; b < count; ++b) {
            const Batch& batch = batches[b];
            // Its waits (and, for the first, KosmicKrisp's residency made current for our
            // command buffers): a later batch without waits needs no submission of its own.
            if (b == 0 || batch.wait_count) {
                submit(batch.waits, batch.wait_count, nullptr, 0, VK_NULL_HANDLE);
            }
            for (u32 i = 0; i < batch.command_buffer_count; ++i) {
                Owned* o = batch.owned[i];
                if (!o) {
                    flush_metal();
                    kk_run.push_back({.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                      .commandBuffer = batch.command_buffers[i]});
                    continue;
                }
                End(*o); // vkEndCommandBuffer did; a no-op then
                if (o->retire) {
                    o->retire->pending.fetch_add(1, std::memory_order_relaxed);
                    held.push_back(o->retire);
                }
                for (Segment& s : o->segments) {
                    if (s.metal) {
                        flush_kk();
                        metal_run.push_back(s.metal);
                        // Counted once per run its Metal buffers are in: released when the
                        // last such run is done.
                        if (metal_retires.empty() || metal_retires.back() != o->retire) {
                            metal_retires.push_back(o->retire);
                        }
                    } else {
                        flush_metal();
                        kk_run.push_back({.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                          .commandBuffer = s.kk->cb});
                        kk_run_buffers.push_back(s.kk);
                    }
                }
                delete o;
            }
            flush_metal();
            const bool last = b + 1 == count;
            if (!kk_run.empty() || batch.signal_count || (last && fence)) {
                submit(nullptr, 0, batch.signals, batch.signal_count,
                       last ? fence : VK_NULL_HANDLE);
            }
        }
        if (count == 0 && fence) {
            submit(nullptr, 0, nullptr, 0, fence);
        }
        if (metal_ahead) {
            // KosmicKrisp's queue is not idle before our work is done (vkQueueWaitIdle, and the
            // next submission's order).
            submit(nullptr, 0, nullptr, 0, VK_NULL_HANDLE);
        }
        for (auto& r : held) {
            Release(r);
        }
        if (const double ms = SubmitTimes::Since(times.start);
            ms >= 100.0 && slow_submissions_logged.fetch_add(1) < 50) {
            Print("Metal frame: a submission took %.1f ms: %u KosmicKrisp submissions with command "
                  "buffers %.1f ms (longest %.1f), %u without (waits, signals, residency) %.1f ms "
                  "(longest %.1f), %u Metal "
                  "commits %.1f ms (longest %.1f)\n",
                  ms, times.kk, times.kk_ms, times.kk_max_ms, times.empty, times.empty_ms,
                  times.empty_max_ms, times.metal, times.metal_ms, times.metal_max_ms);
        }
        return result;
    }
}

void DescriptorSetsAllocated(VkDescriptorPool pool, uint32_t count, const VkDescriptorSet* sets) {
    std::scoped_lock lock{set_contents_mutex};
    const auto empty = std::make_shared<const SetContents>();
    auto& owned = pool_sets[pool];
    for (u32 i = 0; i < count; ++i) {
        if (sets[i]) {
            known_sets[sets[i]] = empty;
            owned.push_back(sets[i]);
        }
    }
}

void DescriptorSetsFreed(uint32_t count, const VkDescriptorSet* sets) {
    std::scoped_lock lock{set_contents_mutex};
    for (u32 i = 0; sets && i < count; ++i) {
        known_sets.erase(sets[i]); // (the pool's list keeps the handle until its reset)
    }
}

void DescriptorPoolReset(VkDescriptorPool pool) {
    std::scoped_lock lock{set_contents_mutex};
    const auto it = pool_sets.find(pool);
    if (it == pool_sets.end()) {
        return;
    }
    for (const VkDescriptorSet set : it->second) {
        known_sets.erase(set);
    }
    pool_sets.erase(it);
}

void DescriptorSetsUpdated(uint32_t write_count, const VkWriteDescriptorSet* writes,
                           uint32_t copy_count, const VkCopyDescriptorSet* copies) {
    if (!write_count && !copy_count) {
        return;
    }
    @autoreleasepool {
        // The writes decoded outside the lock, by set, in order.
        std::vector<std::pair<VkDescriptorSet, PushWrite>> decoded;
        std::vector<VkDescriptorSet> unknown;
        decoded.reserve(write_count);
        for (u32 i = 0; i < write_count; ++i) {
            const VkWriteDescriptorSet& in = writes[i];
            const u32 n = in.descriptorCount;
            PushWrite w;
            w.write = in;
            w.write.pNext = nullptr;
            bool ok = true;
            switch (in.descriptorType) {
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
                ok = in.pBufferInfo || !n;
                if (ok) {
                    w.buffer_infos.assign(in.pBufferInfo, in.pBufferInfo + n);
                }
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                ok = in.pTexelBufferView || !n;
                if (ok) {
                    w.texel_views.assign(in.pTexelBufferView, in.pTexelBufferView + n);
                }
                break;
            case VK_DESCRIPTOR_TYPE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
            case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
                ok = in.pImageInfo || !n;
                if (ok) {
                    w.image_infos.assign(in.pImageInfo, in.pImageInfo + n);
                }
                break;
            default:
                ok = false; // dynamic buffers (offsets at bind), input attachments, the rest
                break;
            }
            if (!ok) {
                unknown.push_back(in.dstSet);
                continue;
            }
            Decode(w);
            decoded.emplace_back(in.dstSet, std::move(w));
        }
        for (u32 i = 0; i < copy_count; ++i) {
            unknown.push_back(copies[i].dstSet); // (not decoded: their draws fall back)
        }
        std::scoped_lock lock{set_contents_mutex};
        std::unordered_map<VkDescriptorSet, std::shared_ptr<SetContents>> updated;
        for (auto& [set, w] : decoded) {
            auto& next = updated[set];
            if (!next) {
                const auto it = known_sets.find(set);
                if (it == known_sets.end() || !it->second) {
                    continue; // allocated before the layer saw it, or unknown already
                }
                next = std::make_shared<SetContents>(*it->second);
            }
            // An earlier write of the binding is replaced when it covers the same elements;
            // one that overlaps otherwise would shadow the new elements (FindBound takes the
            // first match): such a set is not decoded (its draws fall back).
            const u32 first = w.write.dstArrayElement, end = first + w.write.descriptorCount;
            bool overlap = false;
            auto at = next->writes.end();
            for (auto x = next->writes.begin(); x != next->writes.end(); ++x) {
                if (x->write.dstBinding != w.write.dstBinding) {
                    continue;
                }
                const u32 xf = x->write.dstArrayElement, xe = xf + x->write.descriptorCount;
                if (xf == first && xe == end) {
                    at = x;
                } else if (xf < end && first < xe) {
                    overlap = true;
                }
            }
            if (overlap) {
                unknown.push_back(set);
            } else if (at != next->writes.end()) {
                *at = std::move(w);
            } else {
                next->writes.push_back(std::move(w));
            }
        }
        for (auto& [set, next] : updated) {
            if (next) {
                known_sets[set] = std::move(next);
            }
        }
        for (const VkDescriptorSet set : unknown) {
            if (const auto it = known_sets.find(set); it != known_sets.end()) {
                it->second = nullptr; // until it is allocated again
            }
        }
    }
}

void SamplerCreated(VkSampler sampler, const VkSamplerCreateInfo* info) {
    using namespace Metal::VkToMtl;
    @autoreleasepool {
        Sampler s{};
        bool supported = true;
        for (auto* n = static_cast<const VkBaseInStructure*>(info->pNext); n; n = n->pNext) {
            if (n->sType != VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO) {
                supported = false; // custom border colors and the like
            }
        }
        const auto border = BorderColor(vk::BorderColor(info->borderColor));
        supported = supported && border.supported;
        if (supported) {
            MTLSamplerDescriptor* d = [MTLSamplerDescriptor new];
            d.minFilter = MTLSamplerMinMagFilter(SamplerFilter(vk::Filter(info->minFilter)));
            d.magFilter = MTLSamplerMinMagFilter(SamplerFilter(vk::Filter(info->magFilter)));
            d.mipFilter = MTLSamplerMipFilter(MipFilter(vk::SamplerMipmapMode(info->mipmapMode)));
            d.sAddressMode =
                MTLSamplerAddressMode(AddressMode(vk::SamplerAddressMode(info->addressModeU)));
            d.tAddressMode =
                MTLSamplerAddressMode(AddressMode(vk::SamplerAddressMode(info->addressModeV)));
            d.rAddressMode =
                MTLSamplerAddressMode(AddressMode(vk::SamplerAddressMode(info->addressModeW)));
            d.lodMinClamp = info->minLod;
            d.lodMaxClamp = info->maxLod;
            d.lodBias = info->mipLodBias;
            d.maxAnisotropy =
                info->anisotropyEnable ? NSUInteger(std::clamp(info->maxAnisotropy, 1.0f, 16.0f))
                                       : 1;
            if (info->compareEnable) {
                d.compareFunction =
                    MTLCompareFunction(SamplerCompare(vk::CompareOp(info->compareOp)));
            }
            d.borderColor = MTLSamplerBorderColor(border.color);
            d.normalizedCoordinates = !info->unnormalizedCoordinates;
            for (auto* n = static_cast<const VkBaseInStructure*>(info->pNext); n; n = n->pNext) {
                if (n->sType == VK_STRUCTURE_TYPE_SAMPLER_REDUCTION_MODE_CREATE_INFO) {
                    const auto* r = reinterpret_cast<const VkSamplerReductionModeCreateInfo*>(n);
                    d.reductionMode = r->reductionMode == VK_SAMPLER_REDUCTION_MODE_MIN
                                          ? MTLSamplerReductionModeMinimum
                                      : r->reductionMode == VK_SAMPLER_REDUCTION_MODE_MAX
                                          ? MTLSamplerReductionModeMaximum
                                          : MTLSamplerReductionModeWeightedAverage;
                }
            }
            s.state = [device newSamplerStateWithDescriptor:d];
        }
        std::scoped_lock lock{objects_mutex};
        samplers[sampler] = s;
    }
}

void SamplerDestroyed(VkSampler sampler) {
    std::scoped_lock lock{objects_mutex};
    samplers.erase(sampler);
}

void ImageViewCreated(VkImageView view, const VkImageViewCreateInfo* info) {
    std::scoped_lock lock{objects_mutex};
    view_formats[view] = info->format;
    for (u32 use = 0; use < 4; ++use) {
        view_textures.erase({view, use});
    }
}

void ImageViewDestroyed(VkImageView view) {
    std::scoped_lock lock{objects_mutex};
    view_formats.erase(view);
    for (u32 use = 0; use < 4; ++use) {
        view_textures.erase({view, use});
    }
}

void BufferCreated(VkBuffer buffer, VkDeviceSize size) {
    std::scoped_lock lock{objects_mutex};
    buffer_sizes[buffer] = size;
    buffers.erase(buffer);
}

void BufferDestroyed(VkBuffer buffer) {
    std::scoped_lock lock{objects_mutex};
    buffers.erase(buffer);
    buffer_sizes.erase(buffer);
}

// ---- Commands ----

namespace {
/// Ends a Metal render encoder for something that must come between its draws in Vulkan order
/// (a barrier, a fallback outside the pass): the pass reopens, loading, at the next draw.
/// `pass_end`: at vkCmdEndRendering (Vulkan's store operations apply).
void BreakPass(Owned& o, bool pass_end = false) {
    CloseRender(o, pass_end);
}
} // namespace

void CmdPipelineBarrier(Owned& o, VkCommandBuffer, const VkDependencyInfo* info) {
    @autoreleasepool {
        if (o.kk) {
            driver.vkCmdPipelineBarrier2(o.kk->cb, info);
            return;
        }
        if (OrderByBarriers()) {
            if (o.render) {
                BreakPass(o);
            }
            if (o.compute && o.epoch.empty()) {
                // The epoch's only encoder: its dispatches after the barrier wait for those
                // before it, and it already waited for everything earlier.
                [o.compute memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
                o.compute_spans_barrier = true;
                return;
            }
            EndEncoders(o);
            OrderBarrier(o);
            return;
        }
        if (o.render) {
            // Inside a pass (a self-dependency): Metal orders by ending the encoder.
            BreakPass(o);
        }
        if (o.compute) {
            // Dispatches of a concurrent encoder: after everything before the barrier.
            [o.compute memoryBarrierWithScope:MTLBarrierScopeBuffers | MTLBarrierScopeTextures];
        }
        // A blit encoder runs its commands in order; the next encoder waits for it (the chain).
        EndBlit(o);
    }
}

void CmdPipelineBarrier1(Owned& o, VkCommandBuffer cb, VkPipelineStageFlags src,
                         VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t memory_count,
                         const VkMemoryBarrier* memory, uint32_t buffer_count,
                         const VkBufferMemoryBarrier* buffer_barriers, uint32_t image_count,
                         const VkImageMemoryBarrier* images) {
    if (o.kk) {
        // As recorded (also valid inside a render pass if it was).
        driver.vkCmdPipelineBarrier(o.kk->cb, src, dst, flags, memory_count, memory, buffer_count,
                                    buffer_barriers, image_count, images);
        return;
    }
    // Metal: one full barrier stands for it (CmdPipelineBarrier, with no segment open).
    const VkMemoryBarrier2 full{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
    };
    std::vector<VkImageMemoryBarrier2> image_barriers;
    for (u32 i = 0; i < image_count; ++i) {
        // Layout transitions are no-ops on KosmicKrisp; kept for the validation of a segment.
        const VkImageMemoryBarrier& b = images[i];
        image_barriers.push_back({
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .srcAccessMask = b.srcAccessMask,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            .dstAccessMask = b.dstAccessMask,
            .oldLayout = b.oldLayout,
            .newLayout = b.newLayout,
            .srcQueueFamilyIndex = b.srcQueueFamilyIndex,
            .dstQueueFamilyIndex = b.dstQueueFamilyIndex,
            .image = b.image,
            .subresourceRange = b.subresourceRange,
        });
    }
    const VkDependencyInfo info{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &full,
        .imageMemoryBarrierCount = u32(image_barriers.size()),
        .pImageMemoryBarriers = image_barriers.data(),
    };
    CmdPipelineBarrier(o, cb, &info);
}

void CmdBeginRendering(Owned& o, const VkRenderingInfo* info) {
    @autoreleasepool {
        o.in_pass = true;
        o.pass_begun = false;
        o.pass_part = nil;
        o.pass_info = *info;
        o.pass_info.pNext = nullptr;
        o.pass_colors.assign(info->pColorAttachments,
                             info->pColorAttachments + info->colorAttachmentCount);
        for (auto& c : o.pass_colors) {
            c.pNext = nullptr;
        }
        o.has_depth = info->pDepthAttachment != nullptr;
        o.has_stencil = info->pStencilAttachment != nullptr;
        o.pass_depth = o.has_depth ? *info->pDepthAttachment : VkRenderingAttachmentInfo{};
        o.pass_stencil = o.has_stencil ? *info->pStencilAttachment : VkRenderingAttachmentInfo{};
        o.pass_depth.pNext = o.pass_stencil.pNext = nullptr;
        o.pass_metal = BuildPass(o);
        if (!o.pass_metal) {
            Note("a render pass Metal cannot render into (resolve, layers, view mask or "
                 "attachments without Metal textures): its draws go to KosmicKrisp");
        }
        if (o.kk) {
            // The segment is open: it takes the pass now (a following Metal draw ends it).
            Kk(o, true);
        }
    }
}

void CmdEndRendering(Owned& o) {
    @autoreleasepool {
        if (!o.pass_begun) {
            // Never opened (no draws): its clears are still due.
            if (o.kk || !o.pass_metal || KkOnly(o)) {
                Kk(o, true);
            } else {
                RenderEncoder(o);
            }
        }
        if (o.kk && o.kk_in_pass) {
            driver.vkCmdEndRendering(o.kk->cb);
            o.kk_in_pass = false;
        }
        BreakPass(o, true);
        o.in_pass = false;
        o.pass_first = nil;
        o.pass_reopen = nil;
    }
}

void CmdBindPipeline(Owned& o, VkPipelineBindPoint point, VkPipeline pipeline) {
    if (point == VK_PIPELINE_BIND_POINT_COMPUTE) {
        o.compute_pipeline = pipeline;
        o.compute_metal = MetalShadow::FindCompute(pipeline);
    } else {
        o.graphics_pipeline = pipeline;
        // (No wait when the draws go to KosmicKrisp anyway: after a GPU error.)
        o.graphics = MetalShadow::FindGraphicsWaiting(
            pipeline, gpu_failed.load(std::memory_order_relaxed) ? 0 : EmulatedWaitMs(),
            TessInMetal(), GeometryInMetal());
    }
    Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdBindPipeline(cb, point, pipeline); });
}

void CmdPushDescriptorSet(Owned& o, VkPipelineBindPoint point, VkPipelineLayout layout,
                          uint32_t set, uint32_t count, const VkWriteDescriptorSet* writes) {
    @autoreleasepool {
        PushSet& s = o.push_sets[(u32(point) << 16) | set];
        if (s.layout != layout) {
            s.writes.clear(); // another layout's bindings mean nothing in this one
        }
        s.layout = layout;
        o.bound_sets.erase((u32(point) << 16) | set);
        o.set_contents.erase((u32(point) << 16) | set);
        for (u32 i = 0; i < count; ++i) {
            const VkWriteDescriptorSet& in = writes[i];
            PushWrite w;
            w.write = in;
            w.write.pNext = nullptr;
            const u32 n = in.descriptorCount;
            switch (in.descriptorType) {
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
            case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
                w.buffer_infos.assign(in.pBufferInfo, in.pBufferInfo + n);
                break;
            case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
            case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
                w.texel_views.assign(in.pTexelBufferView, in.pTexelBufferView + n);
                break;
            default:
                w.image_infos.assign(in.pImageInfo, in.pImageInfo + n);
                break;
            }
            Decode(w);
            auto it = std::find_if(s.writes.begin(), s.writes.end(), [&](const PushWrite& x) {
                return x.write.dstBinding == in.dstBinding &&
                       x.write.dstArrayElement == in.dstArrayElement;
            });
            if (it != s.writes.end()) {
                *it = std::move(w);
            } else {
                s.writes.push_back(std::move(w));
            }
        }
    }
    Mirror(o, [&](VkCommandBuffer cb) {
        driver.vkCmdPushDescriptorSetKHR(cb, point, layout, set, count, writes);
    });
}

void CmdBindDescriptorSets(Owned& o, VkPipelineBindPoint point, VkPipelineLayout layout,
                           uint32_t first, uint32_t count, const VkDescriptorSet* sets,
                           uint32_t dynamic_count, const uint32_t* dynamic_offsets) {
    // Kept per call (for the replay); a draw that reads one of these sets falls back. Earlier
    // binds of these sets go (a bind that covered more is dropped whole: the replay binds what
    // is current, in set order).
    for (u32 i = 0; i < count; ++i) {
        o.push_sets.erase((u32(point) << 16) | (first + i));
        // bbport: the set's contents for the Metal draws (none with dynamic offsets: the
        // offsets are not applied to them).
        static const bool sets_in_metal = [] {
            const char* env = std::getenv("BB_METAL_DESCRIPTOR_SETS");
            const bool on = !(env && env[0] == '0');
            if (!on) {
                Print("Metal frame: draws reading descriptor sets go to KosmicKrisp "
                      "(BB_METAL_DESCRIPTOR_SETS=0)\n");
            }
            return on;
        }();
        std::shared_ptr<const SetContents> contents =
            dynamic_count || !sets_in_metal ? nullptr : SetContentsOf(sets[i]);
        if (contents) {
            o.set_contents[(u32(point) << 16) | (first + i)] = std::move(contents);
        } else {
            o.set_contents.erase((u32(point) << 16) | (first + i));
        }
    }
    // What is left of a partly covered bind stays (without dynamic offsets: they cannot be
    // told apart per set; such a bind is dropped whole).
    std::vector<std::pair<u32, BoundSets>> kept;
    std::erase_if(o.bound_sets, [&](const auto& entry) {
        const auto& [key, b] = entry;
        if ((key >> 16) != u32(point) || b.sets.empty()) {
            return (key >> 16) == u32(point) && (key & 0xffff) >= first &&
                   (key & 0xffff) < first + count; // markers inside the new range
        }
        const u32 begin = key & 0xffff;
        const u32 end = begin + u32(b.sets.size());
        if (!(begin < first + count && first < end)) {
            return false;
        }
        if (b.dynamic_offsets.empty()) {
            if (begin < first) {
                kept.push_back({(u32(point) << 16) | begin,
                                {b.layout, begin,
                                 std::vector<VkDescriptorSet>(b.sets.begin(),
                                                              b.sets.begin() + (first - begin)),
                                 {}}});
            }
            if (end > first + count) {
                const u32 from = first + count;
                kept.push_back({(u32(point) << 16) | from,
                                {b.layout, from,
                                 std::vector<VkDescriptorSet>(b.sets.begin() + (from - begin),
                                                              b.sets.end()),
                                 {}}});
            }
        }
        return true;
    });
    for (auto& [key, b] : kept) {
        o.bound_sets[key] = std::move(b);
    }
    BoundSets b{layout, first, std::vector<VkDescriptorSet>(sets, sets + count),
                std::vector<u32>(dynamic_offsets, dynamic_offsets + dynamic_count)};
    o.bound_sets[(u32(point) << 16) | first] = std::move(b);
    for (u32 i = 1; i < count; ++i) {
        // The sets after the first are in the first's entry; a marker for the draw check.
        o.bound_sets[(u32(point) << 16) | (first + i)] = {};
    }
    Mirror(o, [&](VkCommandBuffer cb) {
        driver.vkCmdBindDescriptorSets(cb, point, layout, first, count, sets, dynamic_count,
                                       dynamic_offsets);
    });
}

void CmdPushConstants(Owned& o, VkPipelineLayout layout, VkShaderStageFlags stages,
                      uint32_t offset, uint32_t size, const void* values) {
    if (u64(offset) + size <= o.push[0].size()) {
        // Into the bind point(s) the stages belong to.
        if (stages & VK_SHADER_STAGE_COMPUTE_BIT) {
            std::memcpy(o.push[1].data() + offset, values, size);
        }
        if (stages & ~VkShaderStageFlags(VK_SHADER_STAGE_COMPUTE_BIT)) {
            std::memcpy(o.push[0].data() + offset, values, size);
        }
        const PushCall call{layout, stages, offset, size};
        if (std::find(o.push_calls.begin(), o.push_calls.end(), call) == o.push_calls.end()) {
            o.push_calls.push_back(call);
            if (o.push_calls.size() > 32) {
                o.push_calls.erase(o.push_calls.begin());
            }
        }
    }
    Mirror(o, [&](VkCommandBuffer cb) {
        driver.vkCmdPushConstants(cb, layout, stages, offset, size, values);
    });
}

void CmdBindVertexBuffers(Owned& o, uint32_t first, uint32_t count, const VkBuffer* bufs,
                          const VkDeviceSize* offsets, const VkDeviceSize* sizes,
                          const VkDeviceSize* strides) {
    for (u32 i = 0; i < count && first + i < o.vertex_buffers.size(); ++i) {
        auto& vb = o.vertex_buffers[first + i];
        vb.buffer = bufs[i];
        vb.offset = offsets[i];
        vb.size = sizes ? sizes[i] : VK_WHOLE_SIZE;
        if (strides) {
            vb.stride = strides[i];
        }
        vb.set = true;
    }
    Mirror(o, [&](VkCommandBuffer cb) {
        driver.vkCmdBindVertexBuffers2(cb, first, count, bufs, offsets, sizes, strides);
    });
}

void CmdBindIndexBuffer(Owned& o, VkBuffer buffer, VkDeviceSize offset, VkIndexType type) {
    o.index_buffer = buffer;
    o.index_offset = offset;
    o.index_type = type;
    Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdBindIndexBuffer(cb, buffer, offset, type); });
}

void CmdSetViewports(Owned& o, uint32_t first, uint32_t count, const VkViewport* viewports,
                     bool with_count) {
    if (with_count) {
        o.viewports.assign(viewports, viewports + count);
    } else {
        if (o.viewports.size() < first + count) {
            o.viewports.resize(first + count);
        }
        std::copy_n(viewports, count, o.viewports.begin() + first);
    }
    o.state_set[size_t(State::Viewport)] = true;
    o.viewport_with_count = with_count;
    Mirror(o, [&](VkCommandBuffer cb) {
        if (with_count) {
            driver.vkCmdSetViewportWithCount(cb, count, viewports);
        } else {
            driver.vkCmdSetViewport(cb, first, count, viewports);
        }
    });
}

void CmdSetScissors(Owned& o, uint32_t first, uint32_t count, const VkRect2D* scissors,
                    bool with_count) {
    if (with_count) {
        o.scissors.assign(scissors, scissors + count);
    } else {
        if (o.scissors.size() < first + count) {
            o.scissors.resize(first + count);
        }
        std::copy_n(scissors, count, o.scissors.begin() + first);
    }
    o.state_set[size_t(State::Scissor)] = true;
    o.scissor_with_count = with_count;
    Mirror(o, [&](VkCommandBuffer cb) {
        if (with_count) {
            driver.vkCmdSetScissorWithCount(cb, count, scissors);
        } else {
            driver.vkCmdSetScissor(cb, first, count, scissors);
        }
    });
}

void CmdSetFlag(Owned& o, State state, uint32_t value) {
    o.state_set[size_t(state)] = true;
    switch (state) {
    case State::CullMode:
        o.cull_mode = value;
        Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetCullMode(cb, value); });
        break;
    case State::FrontFace:
        o.front_face = VkFrontFace(value);
        Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetFrontFace(cb, VkFrontFace(value)); });
        break;
    case State::DepthTest:
        o.depth_test = value;
        Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetDepthTestEnable(cb, value); });
        break;
    case State::DepthWrite:
        o.depth_write = value;
        Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetDepthWriteEnable(cb, value); });
        break;
    case State::DepthCompare:
        o.depth_compare = VkCompareOp(value);
        Mirror(o, [&](VkCommandBuffer cb) {
            driver.vkCmdSetDepthCompareOp(cb, VkCompareOp(value));
        });
        break;
    case State::DepthBoundsTest:
        o.depth_bounds_test = value;
        Mirror(o, [&](VkCommandBuffer cb) {
            if (driver.vkCmdSetDepthBoundsTestEnable) {
                driver.vkCmdSetDepthBoundsTestEnable(cb, value);
            }
        });
        break;
    case State::DepthBiasEnable:
        o.depth_bias_enable = value;
        Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetDepthBiasEnable(cb, value); });
        break;
    case State::StencilTest:
        o.stencil_test = value;
        Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetStencilTestEnable(cb, value); });
        break;
    case State::PrimitiveRestart:
        o.primitive_restart = value;
        Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetPrimitiveRestartEnable(cb, value); });
        break;
    case State::RasterizerDiscard:
        o.rasterizer_discard = value;
        Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetRasterizerDiscardEnable(cb, value); });
        break;
    case State::FeedbackLoop:
        o.feedback_loop = value;
        Mirror(o, [&](VkCommandBuffer cb) {
            if (driver.vkCmdSetAttachmentFeedbackLoopEnableEXT) {
                driver.vkCmdSetAttachmentFeedbackLoopEnableEXT(cb, value);
            }
        });
        break;
    default:
        break;
    }
}

void CmdSetDepthBounds(Owned& o, float min, float max) {
    o.depth_bounds[0] = min;
    o.depth_bounds[1] = max;
    o.state_set[size_t(State::DepthBounds)] = true;
    Mirror(o, [&](VkCommandBuffer cb) {
        if (driver.vkCmdSetDepthBounds) {
            driver.vkCmdSetDepthBounds(cb, min, max);
        }
    });
}

void CmdSetDepthBias(Owned& o, float constant, float clamp, float slope) {
    o.depth_bias[0] = constant;
    o.depth_bias[1] = clamp;
    o.depth_bias[2] = slope;
    o.state_set[size_t(State::DepthBias)] = true;
    Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetDepthBias(cb, constant, clamp, slope); });
}

void CmdSetStencilOp(Owned& o, VkStencilFaceFlags faces, VkStencilOp fail, VkStencilOp pass,
                     VkStencilOp depth_fail, VkCompareOp compare) {
    for (auto* f : {&o.front, &o.back}) {
        if (faces & (f == &o.front ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT)) {
            f->fail = fail;
            f->pass = pass;
            f->depth_fail = depth_fail;
            f->compare = compare;
        }
    }
    o.state_set[size_t(State::StencilOp)] = true;
    Mirror(o, [&](VkCommandBuffer cb) {
        driver.vkCmdSetStencilOp(cb, faces, fail, pass, depth_fail, compare);
    });
}

void CmdSetStencilValue(Owned& o, State state, VkStencilFaceFlags faces, uint32_t value) {
    for (auto* f : {&o.front, &o.back}) {
        if (faces & (f == &o.front ? VK_STENCIL_FACE_FRONT_BIT : VK_STENCIL_FACE_BACK_BIT)) {
            (state == State::StencilReference     ? f->reference
             : state == State::StencilWriteMask ? f->write_mask
                                                : f->compare_mask) = value;
        }
    }
    o.state_set[size_t(state)] = true;
    Mirror(o, [&](VkCommandBuffer cb) {
        if (state == State::StencilReference) {
            driver.vkCmdSetStencilReference(cb, faces, value);
        } else if (state == State::StencilWriteMask) {
            driver.vkCmdSetStencilWriteMask(cb, faces, value);
        } else {
            driver.vkCmdSetStencilCompareMask(cb, faces, value);
        }
    });
}

void CmdSetBlendConstants(Owned& o, const float constants[4]) {
    std::copy_n(constants, 4, o.blend_constants);
    o.state_set[size_t(State::BlendConstants)] = true;
    Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetBlendConstants(cb, constants); });
}

void CmdSetColorWriteMask(Owned& o, uint32_t first, uint32_t count,
                          const VkColorComponentFlags* masks) {
    for (u32 i = 0; i < count && first + i < o.write_masks.size(); ++i) {
        o.write_masks[first + i] = masks[i];
    }
    o.write_mask_count = std::max<u32>(o.write_mask_count,
                                       std::min<u32>(first + count, u32(o.write_masks.size())));
    o.state_set[size_t(State::ColorWriteMask)] = true;
    Mirror(o, [&](VkCommandBuffer cb) {
        if (driver.vkCmdSetColorWriteMaskEXT) {
            driver.vkCmdSetColorWriteMaskEXT(cb, first, count, masks);
        }
    });
}

void CmdSetLineWidth(Owned& o, float width) {
    o.line_width = width;
    o.state_set[size_t(State::LineWidth)] = true;
    Mirror(o, [&](VkCommandBuffer cb) { driver.vkCmdSetLineWidth(cb, width); });
}

void CmdDraw(Owned& o, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
             uint32_t first_instance) {
    if (!vertex_count || !instance_count) {
        return; // nothing in Vulkan; Metal's validation rejects empty draws
    }
    @autoreleasepool {
        if (o.graphics && (o.graphics->tess || o.graphics->geometry)) {
            DrawTessellated(o, {vertex_count, instance_count, first_vertex, 0, first_instance, false},
                            [&](VkCommandBuffer cb) {
                                driver.vkCmdDraw(cb, vertex_count, instance_count, first_vertex,
                                                 first_instance);
                            });
            return;
        }
        // The rewritten rect list shader builds rectangle k from guest vertices 3k..3k+2
        // (mtl_spirv_rectlist.h): a draw starting inside a rectangle would shift every one.
        const char* blocker = o.graphics && o.graphics->rect_list && first_vertex % 3
                                  ? "a rect list draw starting inside a rectangle"
                                  : nullptr;
        Draw(
            o, blocker,
            [&](id<MTLRenderCommandEncoder> enc) {
                u32 start = first_vertex, n = vertex_count;
                if (o.graphics->rect_list) {
                    // Two triangles (6 vertices) per rectangle of 3 (mtl_spirv_rectlist.h).
                    start = first_vertex / 3 * 6;
                    n = vertex_count / 3 * 6;
                }
                [enc drawPrimitives:PrimitiveOf(o)
                        vertexStart:start
                        vertexCount:n
                      instanceCount:instance_count
                       baseInstance:first_instance];
            },
            [&](VkCommandBuffer cb) {
                driver.vkCmdDraw(cb, vertex_count, instance_count, first_vertex, first_instance);
            });
    }
}

void CmdDrawIndexed(Owned& o, uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                    int32_t vertex_offset, uint32_t first_instance) {
    if (!index_count || !instance_count) {
        return; // nothing in Vulkan; Metal's validation rejects empty draws
    }
    @autoreleasepool {
        if (o.graphics && (o.graphics->tess || o.graphics->geometry)) {
            DrawTessellated(o,
                            {index_count, instance_count, first_index, vertex_offset,
                             first_instance, true},
                            [&](VkCommandBuffer cb) {
                                driver.vkCmdDrawIndexed(cb, index_count, instance_count,
                                                        first_index, vertex_offset,
                                                        first_instance);
                            });
            return;
        }
        const BufferRef index = ResolveBuffer(o.index_buffer);
        const char* blocker = !index.buffer ? "no Metal index buffer"
                              : o.graphics && o.graphics->rect_list ? "an indexed rect list"
                              : o.graphics && o.graphics->render && RestartDiffers(o)
                                  ? "strips without primitive restart"
                                  : nullptr;
        Draw(
            o, blocker,
            [&](id<MTLRenderCommandEncoder> enc) {
                const u32 size = o.index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2;
                const u64 start = index.offset + o.index_offset + u64(first_index) * size;
                [enc drawIndexedPrimitives:PrimitiveOf(o)
                                indexCount:index_count
                                 indexType:size == 4 ? MTLIndexTypeUInt32 : MTLIndexTypeUInt16
                               indexBuffer:index.buffer
                         indexBufferOffset:start
                             instanceCount:instance_count
                                baseVertex:vertex_offset
                              baseInstance:first_instance];
            },
            [&](VkCommandBuffer cb) {
                driver.vkCmdDrawIndexed(cb, index_count, instance_count, first_index,
                                        vertex_offset, first_instance);
            });
    }
}

void CmdDrawIndirect(Owned& o, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                     uint32_t stride, bool indexed) {
    @autoreleasepool {
        const BufferRef args = ResolveBuffer(buffer);
        const BufferRef index = indexed ? ResolveBuffer(o.index_buffer) : BufferRef{};
        const char* blocker = o.graphics && o.graphics->tess       ? "an indirect tessellated draw"
                              : o.graphics && o.graphics->geometry ? "an indirect geometry-shader draw"
                              : !args.buffer                ? "no Metal indirect buffer"
                              : indexed && !index.buffer    ? "no Metal index buffer"
                              : o.graphics && o.graphics->rect_list ? "an indirect rect list"
                              : indexed && o.graphics && o.graphics->render && RestartDiffers(o)
                                  ? "strips without primitive restart"
                                  : nullptr;
        Draw(
            o, blocker,
            [&](id<MTLRenderCommandEncoder> enc) {
                for (u32 i = 0; i < count; ++i) {
                    const u64 at = args.offset + offset + u64(i) * stride;
                    if (indexed) {
                        const u32 size = o.index_type == VK_INDEX_TYPE_UINT32 ? 4 : 2;
                        const u64 start = index.offset + o.index_offset;
                        [enc drawIndexedPrimitives:PrimitiveOf(o)
                                         indexType:size == 4 ? MTLIndexTypeUInt32
                                                             : MTLIndexTypeUInt16
                                       indexBuffer:index.buffer
                                 indexBufferOffset:start
                                    indirectBuffer:args.buffer
                              indirectBufferOffset:at];
                    } else {
                        [enc drawPrimitives:PrimitiveOf(o)
                                  indirectBuffer:args.buffer
                            indirectBufferOffset:at];
                    }
                }
            },
            [&](VkCommandBuffer cb) {
                (indexed ? driver.vkCmdDrawIndexedIndirect : driver.vkCmdDrawIndirect)(
                    cb, buffer, offset, count, stride);
            });
    }
}

void CmdDrawIndirectCount(Owned& o, VkBuffer buffer, VkDeviceSize offset, VkBuffer count_buffer,
                          VkDeviceSize count_offset, uint32_t max_count, uint32_t stride,
                          bool indexed) {
    @autoreleasepool {
        Draw(
            o, "a draw count from the GPU", [&](id<MTLRenderCommandEncoder>) {},
            [&](VkCommandBuffer cb) {
                (indexed ? driver.vkCmdDrawIndexedIndirectCount : driver.vkCmdDrawIndirectCount)(
                    cb, buffer, offset, count_buffer, count_offset, max_count, stride);
            });
    }
}

void CmdDispatch(Owned& o, uint32_t x, uint32_t y, uint32_t z) {
    if (!x || !y || !z) {
        return; // nothing in Vulkan; Metal's validation rejects empty dispatches
    }
    @autoreleasepool {
        Dispatch(
            o,
            [&](id<MTLComputeCommandEncoder> enc, MTLSize threads) {
                [enc dispatchThreadgroups:MTLSizeMake(x, y, z) threadsPerThreadgroup:threads];
            },
            [&](VkCommandBuffer cb) { driver.vkCmdDispatch(cb, x, y, z); });
    }
}

void CmdDispatchIndirect(Owned& o, VkBuffer buffer, VkDeviceSize offset) {
    @autoreleasepool {
        const BufferRef args = ResolveBuffer(buffer);
        if (!args.buffer) {
            Reason("no Metal indirect buffer");
            ++counters.fallback_dispatches;
            if (VkCommandBuffer cb = KkWithState(o, VK_PIPELINE_BIND_POINT_COMPUTE)) {
                driver.vkCmdDispatchIndirect(cb, buffer, offset);
            }
            return;
        }
        Dispatch(
            o,
            [&](id<MTLComputeCommandEncoder> enc, MTLSize threads) {
                [enc dispatchThreadgroupsWithIndirectBuffer:args.buffer
                                       indirectBufferOffset:args.offset + offset
                                      threadsPerThreadgroup:threads];
            },
            [&](VkCommandBuffer cb) { driver.vkCmdDispatchIndirect(cb, buffer, offset); });
    }
}

// ---- Transfers (outside render passes) ----

namespace {

enum class Transfer : u32 { Copy, Fill, Update, BufferToImage, ImageToBuffer, ImageCopy, Clear };

/// Which transfers are encoded in Metal (for bisecting a broken picture), BB_METAL_TRANSFERS:
/// unset or 1: all; 0: none; a comma-separated list of kinds (copy, fill, update, b2i, i2b,
/// image, clear): only those; kinds with a leading '-': all but those.
bool TransfersInMetal(Transfer kind) {
    static const u32 mask = [] {
        constexpr u32 all = (1u << 7) - 1;
        const char* env = std::getenv("BB_METAL_TRANSFERS");
        if (!env || !*env || std::strcmp(env, "1") == 0) {
            return all;
        }
        if (std::strcmp(env, "0") == 0) {
            Print("Metal frame: every transfer to KosmicKrisp (BB_METAL_TRANSFERS=0)\n");
            return 0u;
        }
        static constexpr std::array<const char*, 7> names{"copy", "fill",  "update", "b2i",
                                                         "i2b",  "image", "clear"};
        u32 only = 0, without = 0;
        std::string list{env};
        for (size_t start = 0; start <= list.size();) {
            size_t end = list.find(',', start);
            if (end == std::string::npos) {
                end = list.size();
            }
            std::string token = list.substr(start, end - start);
            start = end + 1;
            const bool minus = !token.empty() && token[0] == '-';
            if (minus) {
                token.erase(0, 1);
            }
            const auto it = std::find_if(names.begin(), names.end(),
                                         [&](const char* n) { return token == n; });
            if (it == names.end()) {
                if (!token.empty()) {
                    Print("Metal frame: BB_METAL_TRANSFERS: unknown kind '%s'\n", token.c_str());
                }
                continue;
            }
            (minus ? without : only) |= 1u << u32(it - names.begin());
        }
        const u32 result = (only ? only : all) & ~without;
        std::string in_metal;
        for (u32 i = 0; i < names.size(); ++i) {
            if (result & (1u << i)) {
                in_metal += in_metal.empty() ? "" : ", ";
                in_metal += names[i];
            }
        }
        Print("Metal frame: transfers in Metal: %s (BB_METAL_TRANSFERS=%s)\n",
              in_metal.empty() ? "none" : in_metal.c_str(), env);
        return result;
    }();
    return (mask >> u32(kind)) & 1u;
}

/// A transfer for KosmicKrisp: why, counted.
VkCommandBuffer KkTransfer(Owned& o, const char* why) {
    Reason(why);
    ++counters.fallback_other;
    return Kk(o, false);
}

/// The image's texture (exported at each use: a swapchain image's changes per frame) and its
/// Vulkan format; nil when its Metal pixel format is not exactly the Vulkan one (the bytes of a
/// buffer copy would not line up).
id<MTLTexture> ImageTexture(VkImage image, VkFormat& format) {
    {
        std::scoped_lock lock{objects_mutex};
        const auto it = images.find(image);
        if (it == images.end()) {
            return nil;
        }
        format = it->second.format;
    }
    VkExportMetalTextureInfoEXT out{
        .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_TEXTURE_INFO_EXT,
        .image = image,
        .plane = VK_IMAGE_ASPECT_PLANE_0_BIT,
    };
    Export(reinterpret_cast<VkBaseOutStructure*>(&out));
    id<MTLTexture> texture = out.mtlTexture;
    if (!texture || Metal::VkToMtl::PixelFormat(vk::Format(format)) != u32(texture.pixelFormat) ||
        !Metal::VkToMtl::IsExactPixelFormat(vk::Format(format))) {
        return nil;
    }
    return texture;
}

/// Bytes and block size of a pixel format, for buffer <-> texture copies.
struct Block {
    u32 bytes{}, width{1}, height{1};
};
Block BlockOf(MTLPixelFormat f) {
    switch (f) {
    case MTLPixelFormatA8Unorm: case MTLPixelFormatR8Unorm: case MTLPixelFormatR8Snorm:
    case MTLPixelFormatR8Uint: case MTLPixelFormatR8Sint: case MTLPixelFormatStencil8:
        return {1};
    case MTLPixelFormatR16Unorm: case MTLPixelFormatR16Snorm: case MTLPixelFormatR16Uint:
    case MTLPixelFormatR16Sint: case MTLPixelFormatR16Float: case MTLPixelFormatRG8Unorm:
    case MTLPixelFormatRG8Snorm: case MTLPixelFormatRG8Uint: case MTLPixelFormatRG8Sint:
    case MTLPixelFormatB5G6R5Unorm: case MTLPixelFormatA1BGR5Unorm: case MTLPixelFormatABGR4Unorm:
    case MTLPixelFormatBGR5A1Unorm: case MTLPixelFormatDepth16Unorm:
        return {2};
    case MTLPixelFormatR32Uint: case MTLPixelFormatR32Sint: case MTLPixelFormatR32Float:
    case MTLPixelFormatRG16Unorm: case MTLPixelFormatRG16Snorm: case MTLPixelFormatRG16Uint:
    case MTLPixelFormatRG16Sint: case MTLPixelFormatRG16Float: case MTLPixelFormatRGBA8Unorm:
    case MTLPixelFormatRGBA8Unorm_sRGB: case MTLPixelFormatRGBA8Snorm: case MTLPixelFormatRGBA8Uint:
    case MTLPixelFormatRGBA8Sint: case MTLPixelFormatBGRA8Unorm: case MTLPixelFormatBGRA8Unorm_sRGB:
    case MTLPixelFormatRGB10A2Unorm: case MTLPixelFormatRGB10A2Uint:
    case MTLPixelFormatRG11B10Float: case MTLPixelFormatRGB9E5Float:
    case MTLPixelFormatBGR10A2Unorm: case MTLPixelFormatDepth32Float:
        return {4};
    case MTLPixelFormatRG32Uint: case MTLPixelFormatRG32Sint: case MTLPixelFormatRG32Float:
    case MTLPixelFormatRGBA16Unorm: case MTLPixelFormatRGBA16Snorm: case MTLPixelFormatRGBA16Uint:
    case MTLPixelFormatRGBA16Sint: case MTLPixelFormatRGBA16Float:
        return {8};
    case MTLPixelFormatRGBA32Uint: case MTLPixelFormatRGBA32Sint: case MTLPixelFormatRGBA32Float:
        return {16};
    case MTLPixelFormatBC1_RGBA: case MTLPixelFormatBC1_RGBA_sRGB: case MTLPixelFormatBC4_RUnorm:
    case MTLPixelFormatBC4_RSnorm:
        return {8, 4, 4};
    case MTLPixelFormatBC2_RGBA: case MTLPixelFormatBC2_RGBA_sRGB: case MTLPixelFormatBC3_RGBA:
    case MTLPixelFormatBC3_RGBA_sRGB: case MTLPixelFormatBC5_RGUnorm:
    case MTLPixelFormatBC5_RGSnorm: case MTLPixelFormatBC6H_RGBFloat:
    case MTLPixelFormatBC6H_RGBUfloat: case MTLPixelFormatBC7_RGBAUnorm:
    case MTLPixelFormatBC7_RGBAUnorm_sRGB:
        return {16, 4, 4};
    default:
        return {}; // Depth32Float_Stencil8 and the rest: per aspect below, or not in Metal
    }
}

/// The format and blit option of one aspect of `texture` for a buffer copy; false if Metal has
/// no such copy.
bool AspectCopy(id<MTLTexture> texture, VkImageAspectFlags aspect, MTLPixelFormat& format,
                MTLBlitOption& option) {
    format = texture.pixelFormat;
    option = MTLBlitOptionNone;
    const bool combined = format == MTLPixelFormatDepth32Float_Stencil8;
    if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
        if (!combined && format != MTLPixelFormatStencil8) {
            return false;
        }
        option = combined ? MTLBlitOptionStencilFromDepthStencil : MTLBlitOptionNone;
        format = MTLPixelFormatStencil8;
    } else if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
        if (combined) {
            option = MTLBlitOptionDepthFromDepthStencil;
            format = MTLPixelFormatDepth32Float;
        } else if (format != MTLPixelFormatDepth32Float && format != MTLPixelFormatDepth16Unorm) {
            return false;
        }
    } else if (aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
        return false;
    }
    return BlockOf(format).bytes != 0;
}

/// A copy region's layer count (VK_REMAINING_ARRAY_LAYERS is legal there with maintenance5).
u32 LayerCount(id<MTLTexture> t, const VkImageSubresourceLayers& layers) {
    if (layers.layerCount != VK_REMAINING_ARRAY_LAYERS) {
        return layers.layerCount;
    }
    const u32 faces = t.textureType == MTLTextureTypeCube ||
                              t.textureType == MTLTextureTypeCubeArray
                          ? 6
                          : 1;
    const u32 all = u32(t.arrayLength) * faces;
    return all > layers.baseArrayLayer ? all - layers.baseArrayLayer : 0;
}

/// Buffer <-> image copies in Metal; false (nothing encoded) when one region cannot be.
bool BufferImageCopies(Owned& o, VkBuffer buffer, VkImage image, u32 count,
                       const VkBufferImageCopy* regions, bool to_image) {
    const BufferRef ref = ResolveBuffer(buffer);
    VkFormat format{};
    id<MTLTexture> texture = ImageTexture(image, format);
    if (!ref.buffer || !texture) {
        return false;
    }
    struct Plan {
        MTLBlitOption option;
        u64 bytes_per_row, bytes_per_image;
    };
    std::vector<Plan> plans(count);
    for (u32 i = 0; i < count; ++i) {
        const VkBufferImageCopy& r = regions[i];
        MTLPixelFormat aspect_format;
        if (!AspectCopy(texture, r.imageSubresource.aspectMask, aspect_format, plans[i].option)) {
            return false;
        }
        const Block block = BlockOf(aspect_format);
        const u32 row_texels = r.bufferRowLength ? r.bufferRowLength : r.imageExtent.width;
        const u32 image_rows = r.bufferImageHeight ? r.bufferImageHeight : r.imageExtent.height;
        plans[i].bytes_per_row = u64((row_texels + block.width - 1) / block.width) * block.bytes;
        plans[i].bytes_per_image =
            plans[i].bytes_per_row * ((image_rows + block.height - 1) / block.height);
        if (texture.textureType == MTLTextureType3D &&
            LayerCount(texture, r.imageSubresource) != 1) {
            return false;
        }
    }
    id<MTLBlitCommandEncoder> enc = BlitEncoder(o);
    for (u32 i = 0; i < count; ++i) {
        const VkBufferImageCopy& r = regions[i];
        const MTLSize size = MTLSizeMake(r.imageExtent.width, r.imageExtent.height,
                                         r.imageExtent.depth);
        const MTLOrigin origin = MTLOriginMake(r.imageOffset.x, r.imageOffset.y, r.imageOffset.z);
        const u32 layers = LayerCount(texture, r.imageSubresource);
        for (u32 layer = 0; layer < layers; ++layer) {
            const u64 offset =
                ref.offset + r.bufferOffset + u64(layer) * plans[i].bytes_per_image;
            const u32 slice = r.imageSubresource.baseArrayLayer + layer;
            // Metal wants bytesPerImage 0 for a 2D copy of one image.
            const u64 per_image = texture.textureType == MTLTextureType3D ? plans[i].bytes_per_image
                                                                          : 0;
            if (to_image) {
                [enc copyFromBuffer:ref.buffer
                           sourceOffset:offset
                      sourceBytesPerRow:plans[i].bytes_per_row
                    sourceBytesPerImage:per_image
                             sourceSize:size
                              toTexture:texture
                       destinationSlice:slice
                       destinationLevel:r.imageSubresource.mipLevel
                      destinationOrigin:origin
                                options:plans[i].option];
            } else {
                [enc copyFromTexture:texture
                                 sourceSlice:slice
                                 sourceLevel:r.imageSubresource.mipLevel
                                sourceOrigin:origin
                                  sourceSize:size
                                    toBuffer:ref.buffer
                           destinationOffset:offset
                      destinationBytesPerRow:plans[i].bytes_per_row
                    destinationBytesPerImage:per_image
                                     options:plans[i].option];
            }
        }
    }
    return true;
}

/// Each (level, layer) of the ranges, `each(aspects, level, layer)`.
template <typename Each>
void ForEachSubresource(id<MTLTexture> texture, u32 count, const VkImageSubresourceRange* ranges,
                        Each&& each) {
    for (u32 r = 0; r < count; ++r) {
        const auto& range = ranges[r];
        const u32 levels = range.levelCount == VK_REMAINING_MIP_LEVELS
                               ? u32(texture.mipmapLevelCount) - range.baseMipLevel
                               : range.levelCount;
        const u32 slices = texture.textureType == MTLTextureTypeCube ||
                                   texture.textureType == MTLTextureTypeCubeArray
                               ? 6
                               : 1;
        const u32 all_layers = u32(texture.arrayLength) * slices;
        const u32 layers = range.layerCount == VK_REMAINING_ARRAY_LAYERS
                               ? all_layers - range.baseArrayLayer
                               : range.layerCount;
        for (u32 level = 0; level < levels; ++level) {
            for (u32 layer = 0; layer < layers; ++layer) {
                each(range.aspectMask, range.baseMipLevel + level, range.baseArrayLayer + layer);
            }
        }
    }
}

/// A texture a clear can render into: 2D kinds with render target usage.
bool ClearableTexture(id<MTLTexture> t) {
    const MTLTextureType type = t.textureType;
    return (t.usage & MTLTextureUsageRenderTarget) &&
           (type == MTLTextureType2D || type == MTLTextureType2DArray ||
            type == MTLTextureTypeCube || type == MTLTextureTypeCubeArray ||
            type == MTLTextureType2DMultisample);
}

/// A render pass of its own that only clears (ending what is open).
void ClearPass(Owned& o, MTLRenderPassDescriptor* rp) {
    id<MTLCommandBuffer> cmd = Metal(o);
    EndEncoders(o);
    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rp];
    if (GpuDiag()) {
        enc.label = @"color clear";
    }
    OrderStart(o, enc, false);
    OrderEnd(o, enc);
    o.pass_part = nil; // (not the open pass's)
    [enc endEncoding];
}

} // namespace

void CmdCopyBuffer(Owned& o, VkBuffer src, VkBuffer dst, uint32_t count,
                   const VkBufferCopy* regions) {
    @autoreleasepool {
        const BufferRef s = ResolveBuffer(src), d = ResolveBuffer(dst);
        if (!TransfersInMetal(Transfer::Copy) || KkOnly(o) || !s.buffer || !d.buffer) {
            if (VkCommandBuffer cb = KkTransfer(o, "a buffer copy KosmicKrisp makes")) {
                driver.vkCmdCopyBuffer(cb, src, dst, count, regions);
            }
            return;
        }
        id<MTLBlitCommandEncoder> enc = BlitEncoder(o);
        for (u32 i = 0; i < count; ++i) {
            [enc copyFromBuffer:s.buffer
                     sourceOffset:s.offset + regions[i].srcOffset
                         toBuffer:d.buffer
                destinationOffset:d.offset + regions[i].dstOffset
                             size:regions[i].size];
        }
        ++counters.metal_transfers;
        transfers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void CmdFillBuffer(Owned& o, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, uint32_t data) {
    @autoreleasepool {
        const BufferRef d = ResolveBuffer(dst);
        const u8 byte = u8(data);
        if (!TransfersInMetal(Transfer::Fill) || KkOnly(o) || !d.buffer ||
            data != u32(byte) * 0x01010101u) {
            if (VkCommandBuffer cb = KkTransfer(o, "a fill KosmicKrisp makes")) {
                driver.vkCmdFillBuffer(cb, dst, offset, size, data);
            }
            return;
        }
        // VK_WHOLE_SIZE: to the end of the buffer, in whole words.
        const u64 rest = d.size > offset ? (d.size - offset) & ~u64(3) : 0;
        const u64 length = size == VK_WHOLE_SIZE ? rest : size;
        if (length) {
            [BlitEncoder(o) fillBuffer:d.buffer range:NSMakeRange(d.offset + offset, length)
                                 value:byte];
        }
        ++counters.metal_transfers;
        transfers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void CmdUpdateBuffer(Owned& o, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                     const void* data) {
    @autoreleasepool {
        const BufferRef d = ResolveBuffer(dst);
        if (!TransfersInMetal(Transfer::Update) || KkOnly(o) || !d.buffer) {
            if (VkCommandBuffer cb = KkTransfer(o, "an update KosmicKrisp makes")) {
                driver.vkCmdUpdateBuffer(cb, dst, offset, size, data);
            }
            return;
        }
        id<MTLBuffer> staging;
        u64 at = 0;
        Scratch(o, data, size, &staging, &at);
        [BlitEncoder(o) copyFromBuffer:staging
                          sourceOffset:at
                              toBuffer:d.buffer
                     destinationOffset:d.offset + offset
                                  size:size];
        ++counters.metal_transfers;
        transfers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void CmdCopyBufferToImage(Owned& o, VkBuffer src, VkImage dst, VkImageLayout layout,
                          uint32_t count, const VkBufferImageCopy* regions) {
    @autoreleasepool {
        if (!TransfersInMetal(Transfer::BufferToImage) || KkOnly(o) ||
            !BufferImageCopies(o, src, dst, count, regions, true)) {
            if (VkCommandBuffer cb = KkTransfer(o, "a buffer to image copy KosmicKrisp makes")) {
                driver.vkCmdCopyBufferToImage(cb, src, dst, layout, count, regions);
            }
            return;
        }
        ++counters.metal_transfers;
        transfers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void CmdCopyImageToBuffer(Owned& o, VkImage src, VkImageLayout layout, VkBuffer dst,
                          uint32_t count, const VkBufferImageCopy* regions) {
    @autoreleasepool {
        if (!TransfersInMetal(Transfer::ImageToBuffer) || KkOnly(o) ||
            !BufferImageCopies(o, dst, src, count, regions, false)) {
            if (VkCommandBuffer cb = KkTransfer(o, "an image to buffer copy KosmicKrisp makes")) {
                driver.vkCmdCopyImageToBuffer(cb, src, layout, dst, count, regions);
            }
            return;
        }
        ++counters.metal_transfers;
        transfers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void CmdCopyImage(Owned& o, VkImage src, VkImageLayout src_layout, VkImage dst,
                  VkImageLayout dst_layout, uint32_t count, const VkImageCopy* regions) {
    @autoreleasepool {
        VkFormat src_format{}, dst_format{};
        id<MTLTexture> s = TransfersInMetal(Transfer::ImageCopy) && !KkOnly(o)
                               ? ImageTexture(src, src_format)
                               : nil;
        id<MTLTexture> d = s ? ImageTexture(dst, dst_format) : nil;
        // Same texture kind both ways (3D <-> 2D array copies turn layers into depth: left to
        // KosmicKrisp).
        bool metal = s && d && s.pixelFormat == d.pixelFormat && s.sampleCount == d.sampleCount &&
                     s.textureType == d.textureType && BlockOf(s.pixelFormat).bytes != 0;
        for (u32 i = 0; metal && i < count; ++i) {
            const u32 layers = LayerCount(s, regions[i].srcSubresource);
            metal = regions[i].srcSubresource.aspectMask == VK_IMAGE_ASPECT_COLOR_BIT &&
                    layers == LayerCount(d, regions[i].dstSubresource) &&
                    (s.textureType != MTLTextureType3D || layers == 1);
        }
        if (!metal) {
            if (VkCommandBuffer cb = KkTransfer(o, "an image copy KosmicKrisp makes")) {
                driver.vkCmdCopyImage(cb, src, src_layout, dst, dst_layout, count, regions);
            }
            return;
        }
        id<MTLBlitCommandEncoder> enc = BlitEncoder(o);
        for (u32 i = 0; i < count; ++i) {
            const VkImageCopy& r = regions[i];
            const u32 layers = LayerCount(s, r.srcSubresource);
            for (u32 layer = 0; layer < layers; ++layer) {
                [enc copyFromTexture:s
                          sourceSlice:r.srcSubresource.baseArrayLayer + layer
                          sourceLevel:r.srcSubresource.mipLevel
                         sourceOrigin:MTLOriginMake(r.srcOffset.x, r.srcOffset.y, r.srcOffset.z)
                           sourceSize:MTLSizeMake(r.extent.width, r.extent.height, r.extent.depth)
                            toTexture:d
                     destinationSlice:r.dstSubresource.baseArrayLayer + layer
                     destinationLevel:r.dstSubresource.mipLevel
                    destinationOrigin:MTLOriginMake(r.dstOffset.x, r.dstOffset.y, r.dstOffset.z)];
            }
        }
        ++counters.metal_transfers;
        transfers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void CmdClearColorImage(Owned& o, VkImage image, VkImageLayout layout,
                        const VkClearColorValue* color, uint32_t count,
                        const VkImageSubresourceRange* ranges) {
    @autoreleasepool {
        VkFormat format{};
        id<MTLTexture> texture =
            TransfersInMetal(Transfer::Clear) && !KkOnly(o) ? ImageTexture(image, format)
                                                                   : nil;
        // bbport: a format stored swizzled (BGRA twins, 4-4-4-4) clears with its channels moved
        // (each stored channel gets the value of the channel that reads it).
        const auto swizzle = Metal::VkToMtl::FormatSwizzle(vk::Format(format));
        VkClearColorValue stored{};
        bool permutation = true;
        {
            const u8 reads[4] = {swizzle.r, swizzle.g, swizzle.b, swizzle.a};
            for (u32 c = 0; c < 4; ++c) {
                if (reads[c] >= Metal::VkToMtl::SwizzleRed &&
                    reads[c] <= Metal::VkToMtl::SwizzleAlpha) {
                    stored.uint32[reads[c] - Metal::VkToMtl::SwizzleRed] = color->uint32[c];
                } else if (reads[c] != Metal::VkToMtl::SwizzleOne) {
                    permutation = false; // (a channel read as one: BC1, not a clear target)
                }
            }
        }
        const char* why = !texture                       ? "a color clear KosmicKrisp makes"
                          : !ClearableTexture(texture)   ? "a color clear of an image Metal cannot render into"
                          : Metal::VkToMtl::IsDepthPixelFormat(u32(texture.pixelFormat))
                              ? "a color clear of a depth format"
                          : !permutation ? "a color clear of a format stored swizzled"
                                         : nullptr;
        if (why) {
            if (VkCommandBuffer cb = KkTransfer(o, why)) {
                driver.vkCmdClearColorImage(cb, image, layout, color, count, ranges);
            }
            return;
        }
        ForEachSubresource(texture, count, ranges, [&](u32, u32 level, u32 layer) {
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            rp.colorAttachments[0].texture = texture;
            rp.colorAttachments[0].level = level;
            rp.colorAttachments[0].slice = layer;
            rp.colorAttachments[0].loadAction = MTLLoadActionClear;
            rp.colorAttachments[0].storeAction = MTLStoreActionStore;
            rp.colorAttachments[0].clearColor = ClearColor(stored, texture.pixelFormat);
            ClearPass(o, rp);
        });
        ++counters.metal_transfers;
        transfers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void CmdClearDepthStencilImage(Owned& o, VkImage image, VkImageLayout layout,
                               const VkClearDepthStencilValue* value, uint32_t count,
                               const VkImageSubresourceRange* ranges) {
    @autoreleasepool {
        VkFormat format{};
        id<MTLTexture> texture =
            TransfersInMetal(Transfer::Clear) && !KkOnly(o) ? ImageTexture(image, format)
                                                                   : nil;
        if (!texture || !ClearableTexture(texture) ||
            !Metal::VkToMtl::IsDepthPixelFormat(u32(texture.pixelFormat))) {
            if (VkCommandBuffer cb = KkTransfer(o, "a depth clear KosmicKrisp makes")) {
                driver.vkCmdClearDepthStencilImage(cb, image, layout, value, count, ranges);
            }
            return;
        }
        const bool has_stencil = Metal::VkToMtl::HasStencilPixelFormat(u32(texture.pixelFormat));
        ForEachSubresource(texture, count, ranges, [&](u32 aspects, u32 level, u32 layer) {
            MTLRenderPassDescriptor* rp = [MTLRenderPassDescriptor renderPassDescriptor];
            // The aspect not cleared is loaded and stored as it is.
            rp.depthAttachment.texture = texture;
            rp.depthAttachment.level = level;
            rp.depthAttachment.slice = layer;
            rp.depthAttachment.loadAction = (aspects & VK_IMAGE_ASPECT_DEPTH_BIT)
                                                ? MTLLoadActionClear
                                                : MTLLoadActionLoad;
            rp.depthAttachment.storeAction = MTLStoreActionStore;
            rp.depthAttachment.clearDepth = value->depth;
            if (has_stencil) {
                rp.stencilAttachment.texture = texture;
                rp.stencilAttachment.level = level;
                rp.stencilAttachment.slice = layer;
                rp.stencilAttachment.loadAction = (aspects & VK_IMAGE_ASPECT_STENCIL_BIT)
                                                      ? MTLLoadActionClear
                                                      : MTLLoadActionLoad;
                rp.stencilAttachment.storeAction = MTLStoreActionStore;
                rp.stencilAttachment.clearStencil = value->stencil;
            }
            ClearPass(o, rp);
        });
        ++counters.metal_transfers;
        transfers_total.fetch_add(1, std::memory_order_relaxed);
    }
}

void ImageCreated(VkImage image, const VkImageCreateInfo* info) {
    std::scoped_lock lock{objects_mutex};
    images[image] = {info->format, info->imageType};
}

void ImageDestroyed(VkImage image) {
    std::scoped_lock lock{objects_mutex};
    images.erase(image);
}

VkCommandBuffer FallbackBuffer(Owned& o, const char* name, bool in_pass) {
    @autoreleasepool {
        (void)name;
        ++counters.fallback_other;
        if (!in_pass) {
            BreakPass(o);
        }
        return Kk(o, in_pass && o.in_pass);
    }
}

void QueryBegun(Owned& o) {
    ++o.active_queries;
}

void QueryEnded(Owned& o) {
    if (o.active_queries) {
        --o.active_queries;
    }
}

unsigned long long TransfersTotal() {
    return transfers_total.load(std::memory_order_relaxed);
}

void Report(double seconds) {
    const auto take = [&](std::atomic<unsigned long long>& c) {
        return double(c.exchange(0, std::memory_order_relaxed)) / seconds;
    };
    std::string top;
    {
        std::scoped_lock lock{reasons_mutex};
        std::vector<std::pair<unsigned long long, std::string>> sorted;
        for (auto& [why, n] : reasons) {
            sorted.emplace_back(n, why);
        }
        reasons.clear();
        std::sort(sorted.rbegin(), sorted.rend());
        for (size_t i = 0; i < sorted.size() && i < 6; ++i) {
            char item[160];
            std::snprintf(item, sizeof item, "%s%s %.0f", top.empty() ? "" : ", ",
                          sorted[i].second.c_str(), double(sorted[i].first) / seconds);
            top += item;
        }
    }
    Print("Metal frame: per second %.0f draws, %.0f dispatches and %.0f transfers in Metal, %.0f "
          "draws and %.0f dispatches and %.0f other commands to KosmicKrisp (%.0f segments, %.0f "
          "switches); %.0f Metal command buffers; render passes ended in Metal %.0f, split %.0f "
          "times in Metal, reopened %.0f times on KosmicKrisp; %.0f tessellated and %.0f "
          "geometry-shader draws in Metal; %.0f Metal encoders, %.0f barrier groups; residency "
          "set: %.0f added, %.0f removed; GPU errors so far %llu; to KosmicKrisp because: %s\n",
          take(counters.metal_draws), take(counters.metal_dispatches),
          take(counters.metal_transfers),
          take(counters.fallback_draws), take(counters.fallback_dispatches),
          take(counters.fallback_other), take(counters.kk_segments), take(counters.switches),
          take(counters.metal_buffers), take(counters.metal_passes), take(counters.pass_breaks),
          take(counters.kk_reopens), take(counters.tess_draws), take(counters.geometry_draws),
          take(counters.encoders), take(counters.epochs), take(counters.residency_adds),
          take(counters.residency_removes), counters.gpu_errors.load(),
          top.empty() ? "-" : top.c_str());
}

} // namespace MetalFrame
