// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include "shim/bb_env.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bbport_toggles.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "platform/errors.h"
#include "platform/kernel.h"
#include "platform/kernel/event_queue.h"
#include "platform/videoout/display.h"
#include "render/rasterizer.h"
#include "platform/videoout/frame_stats.h"
#include "render/vk_frame_capture.h"
#include "cp/command_processor.h"
#include "render/presenter.h"
#include "remote/bb_remote.h"
#include "bbport_frame_rate.h"
#include "bbport_timeline.h"
#include "bbport_sections.h"

extern "C" uint64_t runtime_tsc_frequency(void);

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<BbCp::CommandProcessor> liverpool;

namespace BbVideoOut {

namespace {
/// The game shows its world now (Display::Flip): the pad recording clock runs (bbgpu_world_active).
std::atomic<bool> world_active{false};
/// Microseconds of world frames (each flip's interval while the world is shown): the pad clock.
std::atomic<u64> world_us{0};

constexpr s32 ErrorInvalidValue = s32(0x80290001);
constexpr s32 ErrorInvalidPitch = s32(0x80290004);
constexpr s32 ErrorInvalidTilingMode = s32(0x80290007);
constexpr s32 ErrorInvalidAspectRatio = s32(0x80290008);
constexpr s32 ErrorResourceBusy = s32(0x80290009);
constexpr s32 ErrorNoEmptySlot = s32(0x8029000F);
constexpr s32 ErrorSlotOccupied = s32(0x80290010);

/// The checks both registering and changing an attribute make.
s32 CheckAttribute(const BufferAttribute& attribute) {
    if (attribute.reserved0 != 0 || attribute.reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "buffer attribute: reserved fields set");
        return ErrorInvalidValue;
    }
    if (attribute.aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "buffer attribute: aspect ratio {}", attribute.aspect_ratio);
        return ErrorInvalidAspectRatio;
    }
    if (attribute.width > attribute.pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "buffer attribute: width {} over pitch {}", attribute.width,
                  attribute.pitch_in_pixel);
        return ErrorInvalidPitch;
    }
    if (attribute.tiling_mode != TilingMode::Tile && attribute.tiling_mode != TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "buffer attribute: tiling mode {}", s32(attribute.tiling_mode));
        return ErrorInvalidTilingMode;
    }
    return BbErr::Ok;
}

void TriggerEvents(const std::vector<s64>& queues, InternalEvent event, u64 payload) {
    for (const s64 handle : queues) {
        if (auto* queue = BbKernel::FindQueue(handle)) {
            // The hint: event id in the low byte, the flip argument or vblank count from bit 16.
            queue->Trigger(u64(event), BbKernel::Event::VideoOut,
                           reinterpret_cast<void*>(u64(event) | (payload << 16)));
        }
    }
}

void RemoveEvents(std::vector<s64>& queues, InternalEvent event) {
    for (const s64 handle : queues) {
        if (auto* queue = BbKernel::FindQueue(handle)) {
            queue->Remove(u64(event), BbKernel::Event::VideoOut);
        }
    }
    queues.clear();
}

} // namespace

Display::Display(u32 width, u32 height) {
    main_port.resolution.full_width = s32(width);
    main_port.resolution.full_height = s32(height);
    main_port.resolution.pane_width = s32(width);
    main_port.resolution.pane_height = s32(height);
    // bbport (native GPU process, docs/macos-native-gpu.md): the labels live in memory both
    // processes share. In the game process this display only forwards: bb-gpu presents.
    if (BbRemote::FrontActive()) {
        main_port.buffer_labels = BbRemote::Front::VideoOutLabels();
        BbRemote::Front::SetVideoOutPort(&main_port);
        return;
    }
    if (BbRemote::BackActive()) {
        main_port.buffer_labels = BbRemote::Back::VideoOutLabels();
    }
    separate_present = !bb_env_off("BB_PRESENT_THREAD");
    if (separate_present) {
        present_thread = std::jthread([this](std::stop_token stop) { PresentThread(stop); });
    }
    vblank_thread = std::jthread([this](std::stop_token stop) { VblankThread(stop); });
}

Display::~Display() = default;

s32 Display::Open() {
    if (main_port.is_open) {
        return ErrorResourceBusy;
    }
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::VoOpen();
    }
    main_port.is_open = true;
    liverpool->SetVoPort(&main_port);
    return 1;
}

void Display::Close() {
    if (BbRemote::FrontActive()) {
        BbRemote::Front::VoClose(1); // then this copy of the port is cleared as below
    }
    std::scoped_lock lock{request_mutex};
    Port& port = main_port;
    port.is_open = false;
    port.flip_rate = 0;
    port.prev_index = -1;
    for (std::size_t i = 0; i < MaxDisplayBuffers; ++i) {
        port.StoreLabel(i, 0);
    }
    port.groups = {};
    port.vblank_status = {};
    {
        std::scoped_lock status_lock{port.status_mutex};
        port.flip_status = {};
        if (BbRemote::BackActive()) {
            BbRemote::Back::VoFlipStatus(port.flip_status, false, 0);
        }
    }
    port.buffers.fill({});
    RemoveEvents(port.flip_queues, InternalEvent::Flip);
    RemoveEvents(port.vblank_queues, InternalEvent::Vblank);
}

Port* Display::GetPort(s32 handle) {
    return handle == 1 ? &main_port : nullptr;
}

s32 Display::RegisterBuffers(Port* port, s32 start, void* const* addresses, s32 count,
                             const BufferAttribute* attribute) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::VoRegisterBuffers(start, addresses, count, attribute);
    }
    const auto free_group = std::ranges::find_if(
        port->groups, [](const BufferAttributeGroup& group) { return !group.is_occupied; });
    if (free_group == port->groups.end()) {
        return ErrorNoEmptySlot;
    }
    const s32 group_index = s32(free_group - port->groups.begin());
    if (start < 0 || count < 0 || std::size_t(start + count) > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut, "too many display buffers: from {}, {}", start, count);
        return ErrorInvalidValue;
    }
    const auto first = port->buffers.begin() + start;
    if (std::any_of(first, first + count,
                    [](const DisplayBuffer& buffer) { return buffer.group_index != -1; })) {
        return ErrorSlotOccupied;
    }
    if (const s32 error = CheckAttribute(*attribute)) {
        return error;
    }
    LOG_INFO(Lib_VideoOut,
             "display buffers {} to {}: format {:#x}, tiling {}, {}x{}, pitch {}, option {:#x}",
             start, start + count - 1, u32(attribute->pixel_format), s32(attribute->tiling_mode),
             attribute->width, attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    group.attrib = *attribute;
    group.is_occupied = true;
    for (s32 i = 0; i < count; i++) {
        const auto address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffers[start + i] = {group_index, address, 0};
        Vulkan::FrameCapture::AddDisplayBuffer(address);
        // bbport: a release store: the guest polls the labels (in the game process, with the
        // native GPU process), and what came before must be visible first.
        port->StoreLabel(start + i, 0);
        port->SignalVoLabel();
        presenter->RegisterVideoOutSurface(group, address);
    }
    return group_index;
}

s32 Display::UnregisterBuffers(Port* port, s32 group) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::VoUnregisterBuffers(group);
    }
    if (group < 0 || std::size_t(group) >= MaxDisplayBufferGroups ||
        !port->groups[group].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "no display buffer group {}", group);
        return ErrorInvalidValue;
    }
    port->groups[group].is_occupied = false;
    for (auto& buffer : port->buffers) {
        if (buffer.group_index == group) {
            buffer.group_index = -1;
        }
    }
    return BbErr::Ok;
}

s32 Display::ChangeBufferAttribute(Port* port, s32 group, const BufferAttribute* attribute) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::VoChangeBufferAttribute(group, attribute);
    }
    if (group < 0 || std::size_t(group) >= MaxDisplayBufferGroups ||
        !port->groups[group].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "no display buffer group {}", group);
        return ErrorInvalidValue;
    }
    if (const s32 error = CheckAttribute(*attribute)) {
        return error;
    }
    std::scoped_lock lock{port->status_mutex};
    port->groups[group].attrib = *attribute;
    return BbErr::Ok;
}

bool Display::SubmitFlip(Port* port, s32 index, s64 flip_arg, bool at_eop) {
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::VoSubmitFlip(index, flip_arg);
    }
    {
        std::scoped_lock lock{port->status_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "flip queue full");
            return false;
        }
        if (at_eop) {
            ++port->flip_status.gc_queue_num;
        }
        ++port->flip_status.flip_pending_num; // queued and GPU-side flips together
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
        if (BbRemote::BackActive()) {
            BbRemote::Back::VoFlipStatus(port->flip_status, false, 0);
        }
    }
    if (at_eop) {
        QueueFlip(port, index, flip_arg, true);
    } else {
        // The frame is prepared on the command processor's thread, in order with its work.
        liverpool->SendCommand([=, this] { QueueFlip(port, index, flip_arg, false); });
    }
    return true;
}

void Display::QueueFlip(Port* port, s32 index, s64 flip_arg, bool at_eop) {
    Vulkan::Frame* frame;
    if (index == -1) {
        frame = presenter->PrepareBlankFrame(false);
    } else {
        const DisplayBuffer& buffer = port->buffers[index];
        ASSERT_MSG(buffer.group_index >= 0, "flip to an unregistered display buffer");
        frame = presenter->PrepareFrame(port->groups[buffer.group_index], buffer.address_left);
    }
    // BB_FLIP_AHEAD (default 1, 0 off): the buffer shown before this flip goes back to the game
    // when this flip is queued instead of when it is shown at the next vblank. A game with two
    // display buffers otherwise waits for a vblank whenever a frame takes a little longer than
    // one (45-55 FPS shown as 30/60 alternating). End-of-pipe flips only: they are queued once
    // the GPU has finished the frame (honest labels), so as on the console the game starts a
    // frame only after the GPU finished the one before. (Giving back this flip's own buffer let
    // it run two frames ahead: another game rewrote per-frame data the GPU had still to read,
    // garbage indirect draws lost the device, shader code read zero.) The frame has its own copy
    // of the buffer, recorded before any later drawing into it. At most BB_FLIP_AHEAD flips wait
    // to be shown.
    static const u32 ahead_frames = [] {
        const char* env = bb_env("BB_FLIP_AHEAD");
        return env ? u32(std::max(0, std::atoi(env))) : 1u;
    }();
    bool ahead;
    {
        std::scoped_lock lock{request_mutex};
        ahead = index >= 0 && at_eop && requests.size() < ahead_frames;
        requests.push({frame, port, flip_arg, index, at_eop});
    }
    request_cv.notify_one();
    int released = -1;
    {
        std::scoped_lock lock{port->vblank_mutex};
        if (ahead && port->last_queued >= 0 && port->last_queued != index) {
            released = port->last_queued;
            ++port->released_ahead[released];
            port->StoreLabel(released, 0);
        }
        port->last_queued = index;
    }
    if (released >= 0) {
        port->label_cv.notify_all();
    }
}

bool Display::TakeRequest(Request& request) {
    std::scoped_lock lock{request_mutex};
    if (requests.empty()) {
        return false;
    }
    request = requests.front();
    requests.pop();
    return true;
}

void Display::RunPresenter(std::function<void()> work, bool only_if_idle) {
    if (!separate_present) {
        work();
        return;
    }
    {
        std::scoped_lock lock{present_mutex};
        if (only_if_idle && (present_busy || !present_work.empty())) {
            return; // frames are queued: a blank one adds nothing
        }
        present_work.push_back(std::move(work));
    }
    present_cv.notify_one();
}

void Display::PresentThread(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:Present");
    while (true) {
        std::function<void()> work;
        {
            std::unique_lock lock{present_mutex};
            present_busy = false;
            if (!present_cv.wait(lock, stop, [this] { return !present_work.empty(); })) {
                return;
            }
            work = std::move(present_work.front());
            present_work.pop_front();
            present_busy = true;
        }
        work();
    }
}

void Display::Flip(const Request& request) {
    static u64 flips = 0;
    // The game shows its world (a frame of BB_PAD_WORLD_DRAWS draws or more, 150): the clock of pad
    // recordings runs (bbgpu_world_active), loading screens and the title do not count.
    {
        static const u64 world_draws = [] {
            const char* env = bb_env("BB_PAD_WORLD_DRAWS");
            return env && *env ? std::strtoull(env, nullptr, 10) : 150ull;
        }();
        static u64 last_draws = 0;
        static auto last_flip = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        const u64 draws = BbStats::draws.load(std::memory_order_relaxed);
        const bool world = draws - last_draws >= world_draws;
        if (world) {
            world_us.fetch_add(u64(std::chrono::duration_cast<std::chrono::microseconds>(now - last_flip).count()),
                               std::memory_order_relaxed);
        }
        last_flip = now;
        world_active.store(world, std::memory_order_relaxed);
        last_draws = draws;
    }
    BbTimeline::Note(BbTimeline::FlipProcessed, ++flips);
    BbSections::FrameEnd(runtime_tsc_frequency());
    RunPresenter([frame = request.frame, hdr = request.port->is_hdr] {
        presenter->SetHDR(hdr);
        presenter->Present(frame);
    });
    Port* port = request.port;
    Vulkan::FrameCapture::OnFlip(request.index >= 0 ? port->buffers[request.index].address_left
                                                    : 0);
    if (FrameStatsEnabled()) {
        OnFlipStats();
    }
    // BB_HONEST_FLIPS=1 (an experiment): the flip completes once the GPU has finished the frame,
    // as on the console (a vblank after the frame's commands end), instead of when it is queued.
    static const bool honest_flips = bb_env_on("BB_HONEST_FLIPS");
    if (honest_flips) {
        presenter->WaitRendered(request.frame);
    }
    {
        std::scoped_lock lock{port->status_mutex};
        FlipStatus& status = port->flip_status;
        status.count++;
        status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        status.tsc = Libraries::Kernel::sceKernelReadTsc();
        status.flip_arg = request.flip_arg;
        status.current_buffer = request.index;
        if (request.eop) {
            --status.gc_queue_num;
        }
        --status.flip_pending_num;
        if (BbRemote::BackActive()) {
            // The game process triggers its flip events (the queues below are its own).
            BbRemote::Back::VoFlipStatus(status, true, request.flip_arg);
        }
    }
    TriggerEvents(port->flip_queues, InternalEvent::Flip, u64(request.flip_arg));
    // The buffer shown before is free again.
    if (port->prev_index != -1) {
        {
            std::scoped_lock lock{port->vblank_mutex};
            u32& ahead = port->released_ahead[port->prev_index];
            if (ahead != 0) {
                --ahead; // given back when its flip was queued
            } else {
                port->StoreLabel(port->prev_index, 0);
            }
        }
        port->label_cv.notify_all();
    }
    port->prev_index = request.index;
}

void Display::DrawBlankFrame() {
    RunPresenter(
        [] {
            Vulkan::Frame* frame = presenter->PrepareBlankFrame(true);
            presenter->Present(frame, false, false);
        },
        true);
}

void Display::VblankThread(std::stop_token stop) {
    const std::chrono::nanoseconds vblank_period(1'000'000'000 / EmulatorSettings.GetVblankFrequency());
    Common::SetCurrentThreadName("bb:Vblank");
    Common::AccurateTimer timer{vblank_period};

    // A frame limit (BB_FPS_LIMIT, or the one set while the game runs: BbFrameRate::SetLimit): a
    // flip waits for its slot, one period after the slot before. A late flip moves the slots
    // (BB_PACE_CATCHUP=1: they advance by one period and may lag one behind, so a late frame was
    // followed by an early one: 48 ms then 18 ms at 30 FPS, a saw in the frame time graph).
    const u32 frame_limit = EmulatorSettings.GetFrameLimit();
    static const bool catch_up = bb_env_on("BB_PACE_CATCHUP");
    const auto period_now = [&] {
        const int live = BbFrameRate::Limit();
        const u32 fps = live > 0 ? u32(std::clamp(live, 30, 480)) : frame_limit;
        return fps ? std::chrono::nanoseconds(1'000'000'000 / fps) : std::chrono::nanoseconds(0);
    };
    auto next_flip = std::chrono::steady_clock::now();
    std::printf("VideoOut: vblank %u Hz, frame limit %u FPS\n",
                EmulatorSettings.GetVblankFrequency(), frame_limit);
    // With a frame limit or the uncapped presets a flip is shown as soon as it arrives (and its
    // slot allows), between vblanks, instead of at the next vblank.
    const auto immediate_flips = [&] {
        return frame_limit != 0 || BbFrameRate::Limit() > 0 || EmulatorSettings.IsUncappedVblank();
    };
    const auto flip = [&](const Request& request, std::chrono::steady_clock::time_point now) {
        if (const auto frame_period = period_now(); frame_period.count()) {
            next_flip = catch_up ? std::max(next_flip + frame_period, now - frame_period)
                                 : std::max(next_flip, now) + frame_period;
        }
        Flip(request);
    };

    Port& port = main_port;
    while (!stop.stop_requested()) {
        timer.Start();
        const auto tick_deadline = std::chrono::steady_clock::now() + vblank_period;
        const auto now = std::chrono::steady_clock::now();
        if ((!period_now().count() || now >= next_flip) &&
            port.vblank_status.count % u64(port.flip_rate + 1) == 0) {
            Request request;
            if (TakeRequest(request)) {
                flip(request, now);
            } else if (timer.GetTotalWait().count() < 0 && !port.is_open) {
                DrawBlankFrame(); // nothing opened the display yet
            }
        }
        {
            // sceVideoOutGetVblankStatus reads it concurrently.
            std::scoped_lock lock{port.vblank_mutex};
            const u64 event_count = port.vblank_status.count; // bbport: native GPU process
            TriggerEvents(port.vblank_queues, InternalEvent::Vblank, event_count);
            port.vblank_status.count++;
            port.vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            port.vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            port.vblank_cv.notify_all();
            if (BbRemote::BackActive()) {
                BbRemote::Back::VoVblank(port.vblank_status, event_count);
            }
        }
        if (!immediate_flips() || port.flip_rate != 0) {
            timer.End();
            continue;
        }
        // Until the next vblank: flips as they come.
        while (!stop.stop_requested()) {
            {
                std::unique_lock lock{request_mutex};
                if (!Common::WaitUntil(request_cv, lock, tick_deadline,
                                           [this] { return !requests.empty(); })) {
                    break;
                }
            }
            if (std::chrono::steady_clock::now() < next_flip) {
                if (next_flip >= tick_deadline) {
                    break;
                }
                Common::SleepUntil(next_flip);
            }
            Request request;
            if (TakeRequest(request)) {
                flip(request, std::chrono::steady_clock::now());
            }
        }
        Common::SleepUntil(tick_deadline);
    }
}

} // namespace BbVideoOut

/// The game shows its world (a frame of BB_PAD_WORLD_DRAWS draws or more): runtime_pad.c times
/// recordings and replays by it, so loading screens of any length do not shift them.
extern "C" int bbgpu_world_active(void) {
    return BbVideoOut::world_active.load(std::memory_order_relaxed) ? 1 : 0;
}

extern "C" unsigned long long bbgpu_world_us(void) {
    return BbVideoOut::world_us.load(std::memory_order_relaxed);
}
