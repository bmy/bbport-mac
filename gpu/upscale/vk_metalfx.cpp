// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_metalfx.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>

#include "common/thread.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"

#ifdef __APPLE__
// Only the C types of VK_EXT_external_memory_metal: vulkan.hpp is included without the Metal
// platform define everywhere, and defining it here would change its inline classes (ODR).
#include <vulkan/vulkan_metal.h>
#include "bbport_metalfx.h"
#else
namespace BbMetalFx {
class Scaler {};
} // namespace BbMetalFx
#endif

namespace Vulkan {

#if defined(__APPLE__) && defined(VK_EXT_external_memory_metal)

namespace {

constexpr VkExternalMemoryHandleTypeFlagBits HeapHandle =
    VK_EXTERNAL_MEMORY_HANDLE_TYPE_MTLHEAP_BIT_EXT;

bool EnvOn(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1';
}

} // namespace

struct MetalFxUpscaler::Job {
    BbMetalFx::Frame frame;
    u64 copies_tick;
};

namespace {

u64 NowNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

} // namespace

MetalFxUpscaler::MetalFxUpscaler(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
    if (!instance.IsExternalMemoryMetalEnabled()) {
        Fail("VK_EXT_external_memory_metal is missing (MoltenVK?); MetalFX needs KosmicKrisp",
             true);
        return;
    }
    const vk::PhysicalDeviceExternalBufferInfo info{
        .usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        .handleType = static_cast<vk::ExternalMemoryHandleTypeFlagBits>(HeapHandle),
    };
    const auto properties = instance.GetPhysicalDevice().getExternalBufferProperties(info);
    if (!(properties.externalMemoryProperties.externalMemoryFeatures &
          vk::ExternalMemoryFeatureFlagBits::eExportable)) {
        Fail("the driver cannot export buffer memory as an MTLHeap", true);
        return;
    }
    get_metal_handle =
        reinterpret_cast<void*>(instance.GetDevice().getProcAddr("vkGetMemoryMetalHandleEXT"));
    if (!get_metal_handle) {
        Fail("vkGetMemoryMetalHandleEXT is missing", true);
        return;
    }
    // A small exported buffer gives KosmicKrisp's MTLDevice, the one MetalFX must run on.
    Staging probe;
    if (!CreateStaging(probe, 64 * 1024)) {
        Fail("cannot allocate exportable memory", true);
        return;
    }
    std::string reason;
    supported = BbMetalFx::Supported(probe.heap, reason);
    DestroyStaging(probe);
    if (!supported) {
        Fail(reason, true);
        return;
    }
    std::printf("Upscaler: MetalFX available (temporal scaler on KosmicKrisp's MTLDevice)\n");
    SetUpEvents();
}

void MetalFxUpscaler::SetUpEvents() {
    if (EnvOn("BB_METALFX_SYNC")) {
        mode = Mode::Sync;
        std::printf("Upscaler: MetalFX synchronous (BB_METALFX_SYNC=1)\n");
        return;
    }
    // Default: MetalFX on the submission thread (no waits on the GPU thread, none across APIs).
    // BB_METALFX_EVENTS=1: Metal and Vulkan wait for each other on the GPU through shared events
    // (timed out on KosmicKrisp: Metal never saw the Vulkan value).
    mode = Mode::SubmitThread;
    const char* events_env = std::getenv("BB_METALFX_EVENTS");
    const bool worker_mode = events_env && events_env[0] == '2';
    if (!EnvOn("BB_METALFX_EVENTS") && !worker_mode) {
        std::printf("Upscaler: MetalFX on the submission thread (no GPU thread waits)\n");
        return;
    }
    if (!instance.IsMetalObjectsEnabled()) {
        std::printf("Upscaler: MetalFX synchronous (the driver lacks VK_EXT_metal_objects: "
                    "rebuild KosmicKrisp with the port's patches)\n");
        return;
    }
    const auto device = instance.GetDevice();
    const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        device.getProcAddr("vkExportMetalObjectsEXT"));
    if (!export_objects) {
        std::printf("Upscaler: MetalFX synchronous (vkExportMetalObjectsEXT is missing)\n");
        return;
    }
    const vk::SemaphoreTypeCreateInfo timeline{
        .semaphoreType = vk::SemaphoreType::eTimeline,
        .initialValue = 0,
    };
    const auto created = device.createSemaphore({.pNext = &timeline});
    if (created.result != vk::Result::eSuccess) {
        return;
    }
    done_semaphore = created.value;
    const auto event_of = [&](vk::Semaphore semaphore) -> void* {
        VkExportMetalSharedEventInfoEXT event_info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_SHARED_EVENT_INFO_EXT,
            .semaphore = static_cast<VkSemaphore>(semaphore),
        };
        VkExportMetalObjectsInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
            .pNext = &event_info,
        };
        export_objects(static_cast<VkDevice>(device), &info);
        return reinterpret_cast<void*>(event_info.mtlSharedEvent);
    };
    done_event = event_of(done_semaphore);
    work_event = event_of(scheduler.GetWorkSemaphore()->Handle());
    if (!done_event || !work_event) {
        std::printf("Upscaler: MetalFX synchronous (no shared events from the driver)\n");
        device.destroySemaphore(done_semaphore);
        done_semaphore = vk::Semaphore{};
        done_event = work_event = nullptr;
        return;
    }
    async = true;
    watch = std::jthread{[this](std::stop_token stop) { WatchLoop(stop); }};
    if (worker_mode) {
        mode = Mode::Worker;
        worker = std::jthread{[this](std::stop_token stop) { WorkerLoop(stop); }};
        std::printf("Upscaler: MetalFX from a thread of its own (it waits for the input copies; "
                    "the copy back waits for MetalFX on the GPU; BB_METALFX_EVENTS=2)\n");
        return;
    }
    mode = Mode::Events;
    std::printf("Upscaler: MetalFX without CPU waits (Vulkan timelines as Metal shared events)\n");
}

void MetalFxUpscaler::WorkerLoop(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:MetalFX");
    for (;;) {
        Job job;
        {
            std::unique_lock lock{job_mutex};
            if (!job_cv.wait(lock, stop, [this] { return !jobs.empty(); })) {
                return;
            }
            job = *jobs.front();
            jobs.pop_front();
            job_running = true;
        }
        // The input copies: their submission's value on the scheduler's timeline.
        const u64 start = NowNs();
        scheduler.GetWorkSemaphore()->Wait(job.copies_tick);
        const u64 waited = NowNs() - start;
        if (waited >= 500'000'000) {
            std::printf("Upscaler: MetalFX worker waited %.0f ms for the input copies (tick %llu)\n",
                        waited / 1e6, (unsigned long long)job.copies_tick);
            std::fflush(stdout);
        }
        std::string error;
        if (!scaler->Encode(job.frame, error)) {
            // The copy back must not wait forever.
            BbMetalFx::SignalEvent(job.frame.signal_event, job.frame.signal_value);
            std::scoped_lock lock{deferred_mutex};
            deferred_error = error;
        }
        {
            std::scoped_lock lock{job_mutex};
            job_running = false;
        }
        job_cv.notify_all();
    }
}

void MetalFxUpscaler::DrainWorker() {
    if (mode != Mode::Worker) {
        return;
    }
    std::unique_lock lock{job_mutex};
    job_cv.wait(lock, [this] { return jobs.empty() && !job_running; });
}

void MetalFxUpscaler::WatchLoop(std::stop_token stop) {
    u64 reported = 0;
    while (!stop.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const u64 value = issued_value.load(std::memory_order_acquire);
        if (!value || value == reported) {
            continue;
        }
        const u64 done = BbMetalFx::EventValue(done_event);
        const u64 waited_ns = NowNs() - issued_at_ns.load(std::memory_order_relaxed);
        if (done >= value || waited_ns < 500'000'000) {
            continue;
        }
        reported = value;
        auto* work = scheduler.GetWorkSemaphore();
        work->Refresh();
        std::printf("Upscaler: MetalFX frame %llu not done after %.0f ms (mode %s): its input "
                    "copies are tick %llu, the work timeline is at %llu (Metal sees %llu); "
                    "MetalFX's own timeline at %llu\n",
                    (unsigned long long)value, waited_ns / 1e6,
                    mode == Mode::Worker ? "worker" : "events",
                    (unsigned long long)issued_copies.load(), (unsigned long long)work->KnownGpuTick(),
                    (unsigned long long)BbMetalFx::EventValue(work_event),
                    (unsigned long long)done);
        std::fflush(stdout);
    }
}

MetalFxUpscaler::~MetalFxUpscaler() {
    // Both threads use the scaler and the events: stopped before those go.
    DrainWorker();
    worker = {};
    watch = {};
    if (scaler || staging[0].buffer) {
        scheduler.Finish();
    }
    scaler.reset();
    for (auto& s : staging) {
        DestroyStaging(s);
    }
    if (done_semaphore) {
        instance.GetDevice().destroySemaphore(done_semaphore);
    }
}

void MetalFxUpscaler::Fail(std::string reason, bool permanent) {
    if (problem != reason) {
        std::printf("Upscaler: MetalFX unavailable: %s\n", reason.c_str());
    }
    problem = std::move(reason);
    fatal |= permanent;
}

bool MetalFxUpscaler::CreateStaging(Staging& s, vk::DeviceSize size) {
    const auto device = instance.GetDevice();
    const VkExternalMemoryBufferCreateInfo external{
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = HeapHandle,
    };
    const auto buffer = device.createBuffer(vk::BufferCreateInfo{
        .pNext = &external,
        .size = size,
        .usage = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst,
        .sharingMode = vk::SharingMode::eExclusive,
    });
    if (buffer.result != vk::Result::eSuccess) {
        return false;
    }
    s.buffer = buffer.value;
    s.size = size;
    const auto requirements = device.getBufferMemoryRequirements(s.buffer);
    const auto memory_properties = instance.GetPhysicalDevice().getMemoryProperties();
    u32 type = ~0u;
    for (u32 i = 0; i < memory_properties.memoryTypeCount; ++i) {
        if (!(requirements.memoryTypeBits & (1u << i))) {
            continue;
        }
        if (type == ~0u || (memory_properties.memoryTypes[i].propertyFlags &
                            vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            type = i;
            if (memory_properties.memoryTypes[i].propertyFlags &
                vk::MemoryPropertyFlagBits::eDeviceLocal) {
                break;
            }
        }
    }
    // One allocation per buffer at offset 0: the MTLHeap's buffer view starts there.
    const VkMemoryDedicatedAllocateInfo dedicated{
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .buffer = static_cast<VkBuffer>(s.buffer),
    };
    const VkExportMemoryAllocateInfo export_info{
        .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .handleTypes = HeapHandle,
    };
    if (type == ~0u) {
        DestroyStaging(s);
        return false;
    }
    const auto memory = device.allocateMemory(vk::MemoryAllocateInfo{
        .pNext = &export_info,
        .allocationSize = requirements.size,
        .memoryTypeIndex = type,
    });
    if (memory.result != vk::Result::eSuccess) {
        DestroyStaging(s);
        return false;
    }
    s.memory = memory.value;
    if (device.bindBufferMemory(s.buffer, s.memory, 0) != vk::Result::eSuccess) {
        DestroyStaging(s);
        return false;
    }
    const VkMemoryGetMetalHandleInfoEXT get_info{
        .sType = VK_STRUCTURE_TYPE_MEMORY_GET_METAL_HANDLE_INFO_EXT,
        .memory = static_cast<VkDeviceMemory>(s.memory),
        .handleType = HeapHandle,
    };
    const auto get = reinterpret_cast<PFN_vkGetMemoryMetalHandleEXT>(get_metal_handle);
    if (get(static_cast<VkDevice>(device), &get_info, &s.heap) != VK_SUCCESS || !s.heap) {
        DestroyStaging(s);
        return false;
    }
    return true;
}

void MetalFxUpscaler::DestroyStaging(Staging& s) {
    const auto device = instance.GetDevice();
    if (s.buffer) {
        device.destroyBuffer(s.buffer);
    }
    if (s.memory) {
        device.freeMemory(s.memory);
    }
    s = {};
}

bool MetalFxUpscaler::EnsureResources(const Frame& frame) {
    if (scaler && frame.width == width && frame.height == height &&
        frame.out_width == out_width && frame.out_height == out_height &&
        frame.color.format == color_format && frame.hdr == hdr) {
        return true;
    }
    // The old buffers may still be in a submitted copy (or a frame queued for MetalFX).
    DrainWorker();
    scheduler.Finish();
    scaler.reset();
    for (auto& s : staging) {
        DestroyStaging(s);
    }
    width = frame.width;
    height = frame.height;
    out_width = frame.out_width;
    out_height = frame.out_height;
    color_format = frame.color.format;
    hdr = frame.hdr;
    const bool rgba16 = color_format == vk::Format::eR16G16B16A16Sfloat;
    const vk::DeviceSize pixels = vk::DeviceSize(width) * height;
    const std::array<vk::DeviceSize, 4> sizes{pixels * (rgba16 ? 8 : 4), pixels * 4, pixels * 4,
                                              vk::DeviceSize(out_width) * out_height * 8};
    for (u32 i = 0; i < staging.size(); ++i) {
        if (!CreateStaging(staging[i], sizes[i])) {
            Fail("cannot allocate exportable staging buffers", true);
            return false;
        }
    }
    const BbMetalFx::Config config{
        .in_width = width,
        .in_height = height,
        .out_width = out_width,
        .out_height = out_height,
        .color = rgba16 ? BbMetalFx::ColorFormat::Rgba16Float : BbMetalFx::ColorFormat::Rgba8Unorm,
        .hdr = hdr,
    };
    std::string error;
    scaler = BbMetalFx::Scaler::Create(
        config, {staging[0].heap, staging[1].heap, staging[2].heap, staging[3].heap}, error);
    if (!scaler) {
        Fail(error, true);
        return false;
    }
    std::printf("Upscaler: MetalFX %ux%u -> %ux%u (%s)\n", width, height, out_width, out_height,
                hdr ? "HDR scene color" : "tonemapped frame");
    return true;
}

bool MetalFxUpscaler::Run(const Frame& frame) {
    if (!supported || fatal) {
        return false;
    }
    const auto color = frame.color.format;
    const auto depth = frame.depth.format;
    if ((color != vk::Format::eR16G16B16A16Sfloat && color != vk::Format::eR8G8B8A8Unorm &&
         color != vk::Format::eR8G8B8A8Srgb) ||
        (depth != vk::Format::eD32Sfloat && depth != vk::Format::eD32SfloatS8Uint) ||
        frame.motion.format != vk::Format::eR16G16Sfloat) {
        Fail("unsupported input formats " + vk::to_string(color) + " / " + vk::to_string(depth),
             true);
        return false;
    }
    if (!EnsureResources(frame)) {
        return false;
    }
    auto cmd = scheduler.CommandBuffer();
    const vk::MemoryBarrier2 before_copy{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .dstAccessMask = vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite,
    };
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &before_copy});
    const auto download = [&](const Input& input, const Staging& target) {
        const vk::BufferImageCopy region{
            .imageSubresource = {input.aspect, 0, 0, 1},
            .imageExtent = {frame.width, frame.height, 1},
        };
        cmd.copyImageToBuffer(input.image, vk::ImageLayout::eGeneral, target.buffer, region);
    };
    download(frame.color, staging[0]);
    download(frame.depth, staging[1]);
    download(frame.motion, staging[2]);
    const vk::MemoryBarrier2 to_metal{
        .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
        .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eHost,
        .dstAccessMask = vk::AccessFlagBits2::eHostRead,
    };
    cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &to_metal});
    // An earlier frame that failed after this thread moved on (submission thread, events).
    {
        std::string error;
        if (async && scaler->TakeError(error)) {
            Fail(error, true);
            return false;
        }
        std::scoped_lock lock{deferred_mutex};
        if (!deferred_error.empty()) {
            Fail(std::exchange(deferred_error, {}), true);
            return false;
        }
    }

    // Diagnostics for the conventions (A/B on the Mac): jitter sign, motion sign, reversed Z.
    static const bool invert_jitter = EnvOn("BB_METALFX_INVERT_JITTER");
    static const bool invert_motion = EnvOn("BB_METALFX_INVERT_MOTION");
    static const bool depth_reversed = EnvOn("BB_METALFX_DEPTH_REVERSED");
    const float jitter_sign = invert_jitter ? -1.0f : 1.0f;
    const float motion_sign = invert_motion ? -1.0f : 1.0f;
    BbMetalFx::Frame metal{.jitter_x = jitter_sign * frame.jitter[0],
                           .jitter_y = jitter_sign * frame.jitter[1],
                           .motion_scale_x = motion_sign,
                           .motion_scale_y = motion_sign,
                           .reset = frame.reset,
                           .depth_reversed = depth_reversed};

    // This submission's timeline value: the input copies are done once the GPU reaches it.
    const u64 copies_tick = scheduler.CurrentTick();
    if (mode == Mode::Sync) {
        // The GPU finishes everything first; MetalFX runs here and is waited for.
        scheduler.Finish();
        std::string error;
        if (!scaler->Encode(metal, error)) {
            Fail(error, true);
            return false;
        }
    } else if (mode == Mode::Worker) {
        // The input copies go to the queue; the worker waits for them, then hands MetalFX the
        // work (queued below, with the copy back).
        scheduler.Flush();
    } else if (mode == Mode::Events) {
        scheduler.Flush();
        metal.wait_event = work_event;
        metal.wait_value = copies_tick;
        metal.signal_event = done_event;
        metal.signal_value = done_value + 1;
        std::string error;
        if (!scaler->Encode(metal, error)) {
            Fail(error, true);
            return false;
        }
    } else {
        // Mode::SubmitThread: MetalFX runs right before the copy back is submitted (below).
        scheduler.Flush();
    }

    cmd = scheduler.CommandBuffer();
    const auto output_barrier = [&](vk::ImageLayout old_layout, vk::PipelineStageFlags2 src,
                                    vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst,
                                    vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = src,
            .srcAccessMask = src_access,
            .dstStageMask = dst,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = vk::ImageLayout::eGeneral,
            .image = frame.output,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    };
    constexpr auto rw = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
    // Every pixel is overwritten: the previous contents are not needed.
    output_barrier(vk::ImageLayout::eUndefined, vk::PipelineStageFlagBits2::eAllCommands, rw,
                   vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    const vk::BufferImageCopy region{
        .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .imageExtent = {frame.out_width, frame.out_height, 1},
    };
    cmd.copyBufferToImage(staging[3].buffer, frame.output, vk::ImageLayout::eGeneral, region);
    output_barrier(vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eTransfer,
                   vk::AccessFlagBits2::eTransferWrite, vk::PipelineStageFlagBits2::eAllCommands,
                   rw);
    if (mode == Mode::Events || mode == Mode::Worker) {
        // The copy back (and, in queue order, everything after it) waits for MetalFX on the GPU.
        ++done_value;
        if (mode == Mode::Worker) {
            metal.signal_event = done_event;
            metal.signal_value = done_value;
            {
                std::scoped_lock lock{job_mutex};
                jobs.push_back(std::make_unique<Job>(Job{metal, copies_tick}));
            }
            job_cv.notify_all();
        }
        issued_copies.store(copies_tick, std::memory_order_relaxed);
        issued_at_ns.store(NowNs(), std::memory_order_relaxed);
        issued_value.store(done_value, std::memory_order_release);
        SubmitInfo info{};
        info.AddWait(done_semaphore, done_value);
        scheduler.Flush(info);
    } else if (mode == Mode::SubmitThread) {
        // Right before the copy back goes to the queue (BB_ASYNC_SUBMIT: on a recording thread,
        // after the input copies were submitted): wait for them, run MetalFX and wait for it. The
        // GPU thread goes on with the next frame meanwhile.
        SubmitInfo info{};
        info.before_submit.push_back(
            [this, metal, copies_tick, work = scheduler.GetWorkSemaphore()] {
                const u64 start = NowNs();
                work->Wait(copies_tick);
                const u64 copied = NowNs();
                std::string error;
                if (!scaler->Encode(metal, error)) {
                    std::scoped_lock lock{deferred_mutex};
                    deferred_error = error;
                }
                const u64 end = NowNs();
                if (end - start >= 500'000'000) {
                    std::printf("Upscaler: MetalFX frame took %.0f ms on the submission thread "
                                "(input copies, tick %llu: %.0f ms; MetalFX: %.0f ms)\n",
                                (end - start) / 1e6, (unsigned long long)copies_tick,
                                (copied - start) / 1e6, (end - copied) / 1e6);
                    std::fflush(stdout);
                }
            });
        scheduler.Flush(info);
    }
    return true;
}

#else // not macOS (or Vulkan headers without VK_EXT_external_memory_metal)

MetalFxUpscaler::MetalFxUpscaler(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_} {
#ifdef __APPLE__
    problem = "Vulkan headers without VK_EXT_external_memory_metal (rerun setup_deps.sh)";
    std::printf("Upscaler: MetalFX unavailable: %s\n", problem.c_str());
#else
    problem = "MetalFX is macOS-only";
#endif
    fatal = true;
}

struct MetalFxUpscaler::Job {};

MetalFxUpscaler::~MetalFxUpscaler() = default;

bool MetalFxUpscaler::Run(const Frame&) {
    return false;
}

#endif

} // namespace Vulkan
