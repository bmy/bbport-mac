// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_metalfx.h"

#include <algorithm>
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
#include <sys/sysctl.h>
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

/// The stall cutoff, worker mode (BB_METALFX_STALL_MS, default 1000; 0: off, frames wait until
/// macOS stops the command buffer, which took a minute on 2026-10-24).
u64 StallCutoffNs() {
    static const u64 cutoff = [] {
        const char* env = std::getenv("BB_METALFX_STALL_MS");
        const long ms = env && *env ? std::strtol(env, nullptr, 10) : 1000;
        return ms > 0 ? u64(ms) * 1'000'000ull : 0ull;
    }();
    return cutoff;
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
    // Under Rosetta (the x86-64 game process) MetalFX's Neural Engine part fails to set up (an
    // appleneuralengine "doCompileModel" error), and its work then hangs now and then until
    // macOS stops it after about 4 s. In the arm64 GPU process neither happens.
    int translated = 0;
    std::size_t size = sizeof(translated);
    if (sysctlbyname("sysctl.proc_translated", &translated, &size, nullptr, 0) == 0 &&
        translated == 1) {
        std::printf("Upscaler: MetalFX under Rosetta can stall for seconds and then fail (FSR 3.1 "
                    "takes over); the native GPU process runs it properly (the default where "
                    "it is built; this is BB_NATIVE_GPU=0 or a build without it)\n");
    }
    SetUpEvents();
}

void MetalFxUpscaler::SetUpEvents() {
    if (EnvOn("BB_METALFX_SYNC")) {
        mode = Mode::Sync;
        std::printf("Upscaler: MetalFX synchronous (BB_METALFX_SYNC=1)\n");
        return;
    }
    // Default (BB_METALFX_EVENTS=2): the worker. Its own thread waits for the input copies on
    // the CPU and gives MetalFX the work with no GPU-side wait; the copy back waits for MetalFX
    // on the GPU. With BB_METALFX_EVENTS=1 Metal also waits for Vulkan on the GPU, and that wait
    // now and then never woke (2026-10-21: Vulkan had finished the copies, MetalFX's command
    // buffer waited about 4 s until macOS stopped it, native GPU process too); the worker ran a
    // full session without one. BB_METALFX_EVENTS=0: on the submission thread, CPU waits on both
    // sides (slower). Without shared events from the driver, the submission thread too.
    mode = Mode::SubmitThread;
    const char* events_env = std::getenv("BB_METALFX_EVENTS");
    const char events = events_env && *events_env ? events_env[0] : '2';
    const bool queue_mode = events == '3';
    const bool worker_mode = events == '2' || queue_mode; // the worker is queue mode's fallback
    if (events != '1' && !worker_mode) {
        std::printf("Upscaler: MetalFX on the submission thread (BB_METALFX_EVENTS=%s)\n",
                    events_env);
        return;
    }
    if (!instance.IsMetalObjectsEnabled()) {
        std::printf("Upscaler: MetalFX on the submission thread (the driver lacks VK_EXT_metal_objects: "
                    "rebuild KosmicKrisp with the port's patches)\n");
        return;
    }
    const auto device = instance.GetDevice();
    const auto export_objects = reinterpret_cast<PFN_vkExportMetalObjectsEXT>(
        device.getProcAddr("vkExportMetalObjectsEXT"));
    if (!export_objects) {
        std::printf("Upscaler: MetalFX on the submission thread (vkExportMetalObjectsEXT is missing)\n");
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
        std::printf("Upscaler: MetalFX on the submission thread (no shared events from the driver)\n");
        device.destroySemaphore(done_semaphore);
        done_semaphore = vk::Semaphore{};
        done_event = work_event = nullptr;
        return;
    }
    async = true;
    mode = worker_mode ? Mode::Worker : Mode::Events;
    if (queue_mode) {
        VkExportMetalCommandQueueInfoEXT queue_info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_COMMAND_QUEUE_INFO_EXT,
            .queue = static_cast<VkQueue>(instance.GetGraphicsQueue()),
        };
        VkExportMetalObjectsInfoEXT info{
            .sType = VK_STRUCTURE_TYPE_EXPORT_METAL_OBJECTS_INFO_EXT,
            .pNext = &queue_info,
        };
        export_objects(static_cast<VkDevice>(device), &info);
        driver_queue = reinterpret_cast<void*>(queue_info.mtlCommandQueue);
        if (driver_queue) {
            mode = Mode::Queue;
            watch = std::jthread{[this](std::stop_token stop) { WatchLoop(stop); }};
            std::printf("Upscaler: MetalFX on KosmicKrisp's own queue, in its order "
                        "(BB_METALFX_EVENTS=3, a test; no stall cutoff: a stall there holds up "
                        "the whole queue, and a MetalFX GPU error there may lose the device; "
                        "after any failure the worker hand-over takes over)\n");
            return;
        }
        std::printf("Upscaler: MetalFX cannot use KosmicKrisp's queue (the driver lacks the "
                    "port's patch 0007): the worker hand-over instead\n");
    }
    watch = std::jthread{[this](std::stop_token stop) { WatchLoop(stop); }};
    if (worker_mode) {
        if (const char* env = std::getenv("BB_METALFX_TEST_STALL"); env && *env) {
            test_stalls_left = u32(std::max(0l, std::strtol(env, nullptr, 10)));
            std::printf("Upscaler: MetalFX test (BB_METALFX_TEST_STALL=%u): the 600th frame of "
                        "each of the first %u MetalFX runs waits 10 s on the GPU (macOS may stop "
                        "it first)\n",
                        test_stalls_left, test_stalls_left);
        }
        worker = std::jthread{[this](std::stop_token stop) { WorkerLoop(stop); }};
        std::printf("Upscaler: MetalFX from a thread of its own (it waits for the input copies; "
                    "the copy back waits for MetalFX on the GPU; BB_METALFX_EVENTS=0 or 1: the "
                    "other hand-overs)\n");
        if (const u64 cutoff = StallCutoffNs()) {
            std::printf("Upscaler: MetalFX work not done %llu ms after the worker hands it over is "
                        "given up, and FSR 3.1 takes over (BB_METALFX_STALL_MS; 0: wait for macOS "
                        "to stop it)\n",
                        (unsigned long long)(cutoff / 1'000'000));
        } else {
            std::printf("Upscaler: MetalFX stall cutoff off (BB_METALFX_STALL_MS=0)\n");
        }
        return;
    }
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
        if (stalled.load(std::memory_order_acquire)) {
            // An earlier frame stalled on MetalFX's queue (the cutoff): this one would queue
            // behind it. Its copy back goes on without it, from the CPU.
            u64 released = released_value.load(std::memory_order_relaxed);
            while (released < job.frame.signal_value &&
                   !released_value.compare_exchange_weak(released, job.frame.signal_value)) {
            }
            BbMetalFx::SignalEvent(job.frame.signal_event, job.frame.signal_value);
        } else if (!Encode(job.frame, error)) {
            // The copy back must not wait forever; earlier frames still on the GPU signal lower
            // values (a timeline never goes down), so they finish first.
            scaler->WaitIdle();
            BbMetalFx::SignalEvent(job.frame.signal_event, job.frame.signal_value);
            std::scoped_lock lock{deferred_mutex};
            deferred_error = error;
        } else {
            std::scoped_lock lock{pending_mutex};
            for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
                if (it->value == job.frame.signal_value) {
                    it->handed_ns = NowNs();
                    break;
                }
            }
        }
        encoded_value.store(job.frame.signal_value, std::memory_order_relaxed);
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

void MetalFxUpscaler::ReleaseIssued(bool only_if_stalled) {
    std::scoped_lock lock{pending_mutex};
    if (only_if_stalled && !stalled.load(std::memory_order_acquire)) {
        return;
    }
    u64 top = released_value.load(std::memory_order_relaxed);
    if (!pending.empty()) {
        top = std::max(top, pending.back().value);
    }
    u64 released = released_value.load(std::memory_order_relaxed);
    while (released < top && !released_value.compare_exchange_weak(released, top)) {
    }
    // Raised only: the stalled command buffer, once it ends, may set a lower value (a later
    // pass here puts it back up).
    BbMetalFx::SignalEvent(done_event, top);
}

void MetalFxUpscaler::WatchLoop(std::stop_token stop) {
    u64 reported = 0;
    while (!stop.stop_requested()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // Frames issued since the stall (the GPU thread had not seen it yet) go on too.
        ReleaseIssued(true);
        const u64 done = BbMetalFx::EventValue(done_event);
        Pending oldest{};
        {
            std::scoped_lock lock{pending_mutex};
            while (!pending.empty() && pending.front().value <= done) {
                pending.pop_front();
            }
            if (pending.empty()) {
                continue;
            }
            oldest = pending.front();
        }
        const u64 now_ns = NowNs();
        u64 cutoff = mode == Mode::Worker ? StallCutoffNs() : 0;
        if (cutoff && oldest.value == first_value.load(std::memory_order_relaxed)) {
            cutoff = std::max<u64>(cutoff, 15'000'000'000ull); // a new scaler's first frame
        }
        if (cutoff && oldest.handed_ns && now_ns - oldest.handed_ns >= cutoff &&
            !stalled.load(std::memory_order_acquire)) {
            // MetalFX's own GPU work has not finished. The copy back of this frame, and of every
            // later one, would wait for it until macOS stops the command buffer (a minute, with
            // the game frozen, on 2026-10-24): they go on without it, and FSR 3.1 takes over.
            stalled.store(true, std::memory_order_release);
            ReleaseIssued(false);
            char text[160];
            std::snprintf(text, sizeof(text),
                          "MetalFX's GPU work stalled (frame %llu not done %.0f ms after it was "
                          "handed over)",
                          (unsigned long long)oldest.value, (now_ns - oldest.handed_ns) / 1e6);
            {
                std::scoped_lock lock{deferred_mutex};
                deferred_error = text;
            }
            std::printf("Upscaler: %s; the frames waiting for it go on without it "
                        "(BB_METALFX_STALL_MS=0: wait for macOS instead)\n",
                        text);
            std::fflush(stdout);
            continue;
        }
        const u64 waited_ns = now_ns - oldest.issued_ns;
        if (oldest.value == reported || waited_ns < 500'000'000) {
            continue;
        }
        reported = oldest.value;
        auto* work = scheduler.GetWorkSemaphore();
        work->Refresh();
        char worker_note[96] = "";
        if (mode == Mode::Worker) {
            std::snprintf(worker_note, sizeof(worker_note),
                          "; the worker handed Metal frames up to %llu",
                          (unsigned long long)encoded_value.load(std::memory_order_relaxed));
        }
        std::printf("Upscaler: MetalFX frame %llu: %s after %.0f ms "
                    "(%s mode). Its input copies are tick %llu; the work timeline is at %llu "
                    "(Metal sees %llu); MetalFX's timeline is at %llu%s\n",
                    (unsigned long long)oldest.value,
                    mode == Mode::Queue ? "not done" : "its copy back still waits",
                    waited_ns / 1e6,
                    mode == Mode::Worker ? "worker" : mode == Mode::Queue ? "queue" : "events",
                    (unsigned long long)oldest.copies_tick,
                    (unsigned long long)work->KnownGpuTick(),
                    (unsigned long long)BbMetalFx::EventValue(work_event),
                    (unsigned long long)done, worker_note);
        std::fflush(stdout);
    }
}

MetalFxUpscaler::~MetalFxUpscaler() {
    // Both threads use the scaler and the events: stopped before those go.
    DrainWorker();
    worker = {};
    watch = {};
    const bool was_stalled = stalled.load();
    if (was_stalled) {
        ReleaseIssued(false); // the copy backs still queued must not wait (Finish below)
    }
    bool abandoned = false;
    if (was_stalled && scaler && scaler->Busy()) {
        // A stalled frame is still on MetalFX's queue (the cutoff let its copy back go): waiting
        // for it could take until macOS stops it. Its buffers stay allocated.
        std::printf("Upscaler: MetalFX's stalled frame is still on the GPU; its buffers are left "
                    "allocated\n");
        scaler->Abandon();
        scaler.reset();
        abandoned = true;
    }
    if (scaler || staging[0].buffer || abandoned) {
        scheduler.Finish();
    }
    scaler.reset();
    for (auto& s : staging) {
        if (abandoned) {
            s = {}; // left allocated: the stalled command buffer may still use them
        } else {
            DestroyStaging(s);
        }
    }
    if (done_semaphore) {
        instance.GetDevice().destroySemaphore(done_semaphore);
    }
}

void MetalFxUpscaler::Fail(std::string reason, bool permanent, bool at_run_time) {
    if (problem != reason) {
        std::printf("Upscaler: MetalFX unavailable: %s\n", reason.c_str());
    }
    problem = std::move(reason);
    fatal |= permanent;
    transient = at_run_time;
    rebuild_after_failure |= at_run_time;
}

void MetalFxUpscaler::LeaveQueueMode(const std::string& why) {
    if (mode != Mode::Queue) {
        return;
    }
    std::printf("Upscaler: MetalFX leaves KosmicKrisp's queue for the session (%s): the worker "
                "hand-over instead\n",
                why.c_str());
    mode = Mode::Worker;
    if (!worker.joinable()) {
        worker = std::jthread{[this](std::stop_token stop) { WorkerLoop(stop); }};
    }
}

bool MetalFxUpscaler::Ready() const {
    return !(stalled.load(std::memory_order_acquire) || rebuild_after_failure) || !scaler ||
           !scaler->Busy();
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
    // The first MetalFX frame of a session has taken about 14 s (2026-10-21, arm64, the GPU thread
    // busy throughout): creation and the first encode are timed to tell which part it is.
    const auto start = std::chrono::steady_clock::now();
    scaler = BbMetalFx::Scaler::Create(
        config, {staging[0].heap, staging[1].heap, staging[2].heap, staging[3].heap}, error,
        mode == Mode::Queue ? driver_queue : nullptr);
    if (!scaler && mode == Mode::Queue) {
        // No Metal 4 MetalFX here (the GPU, the OS, the queue): the worker hand-over instead.
        LeaveQueueMode(error);
        error.clear();
        scaler = BbMetalFx::Scaler::Create(
            config, {staging[0].heap, staging[1].heap, staging[2].heap, staging[3].heap}, error);
    }
    if (!scaler) {
        Fail(error, true);
        return false;
    }
    std::printf("Upscaler: MetalFX %ux%u -> %ux%u (%s), created in %.0f ms\n", width, height,
                out_width, out_height, hdr ? "HDR scene color" : "tonemapped frame",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count());
    first_encode.store(true, std::memory_order_relaxed);
    first_value.store(done_value + 1, std::memory_order_relaxed);
    return true;
}

bool MetalFxUpscaler::Encode(const BbMetalFx::Frame& metal, std::string& error) {
    if (!first_encode.exchange(false, std::memory_order_relaxed)) {
        return scaler->Encode(metal, error);
    }
    const auto start = std::chrono::steady_clock::now();
    const bool ok = scaler->Encode(metal, error);
    std::printf("Upscaler: MetalFX's first frame encoded in %.0f ms\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                    .count());
    std::fflush(stdout);
    return ok;
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
    if (stalled.load(std::memory_order_acquire)) {
        // After the cutoff. The stall itself is reported first (FSR 3.1 takes over, even if the
        // stalled frame ends at once); MetalFX runs again (after a retry) only once that frame
        // has left MetalFX's queue, and its buffers are not replaced before.
        std::string reason;
        {
            std::scoped_lock lock{deferred_mutex};
            reason = std::exchange(deferred_error, {});
        }
        if (!reason.empty() || (scaler && scaler->Busy())) {
            Fail(reason.empty() ? "MetalFX's stalled frame is still on the GPU" : reason, true,
                 true);
            return false;
        }
        DrainWorker(); // frames queued during the stall: let go by the worker, not encoded
        {
            std::scoped_lock lock{pending_mutex};
            // Whatever the stalled frame set at its end, the timeline goes back up to what was
            // let go (nothing is in flight, so nothing lowers it after).
            BbMetalFx::SignalEvent(done_event, released_value.load(std::memory_order_relaxed));
            stalled.store(false, std::memory_order_release);
        }
        if (std::string ended; scaler && scaler->TakeError(ended)) {
            std::printf("Upscaler: MetalFX's stalled frame ended: %s\n", ended.c_str());
        }
    }
    if (rebuild_after_failure) {
        // After a failure at run time: a new scaler, and with it a new command queue (macOS
        // ignores the old queue's work once it caused a GPU error). Not while that work is still
        // on the GPU: its buffers would go.
        DrainWorker(); // frames queued for the old scaler are handed over first
        if (scaler && scaler->Busy()) {
            Fail("MetalFX's failed frame is still on the GPU", true, true);
            return false;
        }
        if (mode == Mode::Queue) {
            // Its own queue is the driver's here: a new scaler would not get a new one. The
            // worker hand-over (a queue of its own) from now on, if the device lives on.
            LeaveQueueMode("its work failed on the driver's queue");
        }
        if (scaler) {
            std::printf("Upscaler: MetalFX gets a new scaler and command queue after its failure "
                        "(macOS ignores a queue's work after it caused a GPU error)\n");
            scheduler.Finish(); // the submission thread's MetalFX callbacks (SubmitThread mode)
            scaler.reset();     // EnsureResources makes the new one
        }
        rebuild_after_failure = false;
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
            Fail(error, true, true);
            return false;
        }
        std::scoped_lock lock{deferred_mutex};
        if (!deferred_error.empty()) {
            Fail(std::exchange(deferred_error, {}), true, true);
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
        if (!Encode(metal, error)) {
            Fail(error, true, true); // the GPU's failure (Encode waited for it)
            return false;
        }
    } else if (mode == Mode::Worker || mode == Mode::Queue) {
        // The input copies go to the queue. Worker: the worker waits for them, then hands
        // MetalFX the work (queued below, with the copy back). Queue: MetalFX follows them on the
        // same queue (committed right before the copy back, below).
        scheduler.Flush();
    } else if (mode == Mode::Events) {
        scheduler.Flush();
        metal.wait_event = work_event;
        metal.wait_value = copies_tick;
        metal.signal_event = done_event;
        metal.signal_value = done_value + 1;
        std::string error;
        if (!Encode(metal, error)) {
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
            if (test_stalls_left && ++test_frames == 600) {
                metal.test_stall = true;
                --test_stalls_left;
                std::printf("Upscaler: MetalFX test: frame %llu waits 10 s on the GPU "
                            "(BB_METALFX_TEST_STALL, %u more)\n",
                            (unsigned long long)done_value, test_stalls_left);
            }
            {
                std::scoped_lock lock{job_mutex};
                jobs.push_back(std::make_unique<Job>(Job{metal, copies_tick}));
            }
            job_cv.notify_all();
        }
        {
            std::scoped_lock lock{pending_mutex};
            if (pending.size() >= 256) {
                pending.pop_front();
            }
            pending.push_back({done_value, copies_tick, NowNs()});
        }
        SubmitInfo info{};
        info.AddWait(done_semaphore, done_value, "MetalFX");
        scheduler.Flush(info);
    } else if (mode == Mode::Queue) {
        // Committed to KosmicKrisp's queue right before the copy back is submitted (under the
        // submission lock, after the input copies' submission): the queue's order and barriers
        // put MetalFX between them. Its timeline is signalled on that queue for the reports.
        ++done_value;
        metal.signal_event = done_event;
        metal.signal_value = done_value;
        {
            std::scoped_lock lock{pending_mutex};
            if (pending.size() >= 256) {
                pending.pop_front();
            }
            pending.push_back({done_value, copies_tick, NowNs()});
        }
        SubmitInfo info{};
        info.before_submit.push_back([this, metal] {
            // Under the submission lock: a slow first encode (pipelines MetalFX builds) holds up
            // every submission meanwhile, so the log times it.
            const bool first = first_encode.exchange(false, std::memory_order_relaxed);
            const auto start = std::chrono::steady_clock::now();
            std::string error;
            if (!scaler->EncodeOnQueue(metal, error)) { // it still signals, on the queue
                std::scoped_lock lock{deferred_mutex};
                deferred_error = error;
            }
            if (first) {
                std::printf("Upscaler: MetalFX's first frame encoded in %.0f ms (on the driver's "
                            "queue, under the submission lock)\n",
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - start)
                                .count());
                std::fflush(stdout);
            }
        });
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
                if (!Encode(metal, error)) {
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

bool MetalFxUpscaler::Ready() const {
    return true;
}

#endif

} // namespace Vulkan
