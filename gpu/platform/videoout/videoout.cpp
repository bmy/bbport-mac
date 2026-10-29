// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <cstring>
#include "shim/bb_env.h"
#include <memory>
#include <utility>

#include "common/thread.h"
#include "bbport_frame_rate.h"
#include "common/assert.h"
#include "common/elf_info.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "core/libraries/libs.h"
#include "platform/errors.h"
#include "platform/interrupts.h"
#include "platform/kernel/event_queue.h"
#include "platform/videoout/display.h"
#include "render/vk_low_latency.h"
#include "platform/videoout/videoout.h"
#include "render/presenter.h"
#include "remote/bb_remote.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;

namespace BbVideoOut {

namespace {

std::unique_ptr<Display> display;

constexpr s32 ErrorInvalidValue = s32(0x80290001);
constexpr s32 ErrorInvalidAddress = s32(0x80290002);
constexpr s32 ErrorResourceBusy = s32(0x80290009);
constexpr s32 ErrorInvalidIndex = s32(0x8029000A);
constexpr s32 ErrorInvalidHandle = s32(0x8029000B);
constexpr s32 ErrorInvalidEventQueue = s32(0x8029000C);
constexpr s32 ErrorInvalidEvent = s32(0x8029000D);
constexpr s32 ErrorFlipQueueFull = s32(0x80290012);

constexpr int BusTypeMain = 0;
constexpr u64 CapabilityBt2020Pq = 0x80;
constexpr u8 ColorimetryBt2020Pq = 12;
constexpr u8 ColorimetryAny = 0xFF;

/// The public event ids sceVideoOutGetEventId reports.
enum class EventId : s32 { Flip = 0, Vblank = 1, PreVblankStart = 2, SetMode = 8, Position = 12 };

using BbKernel::Event;

Port* OpenPort(s32 handle) {
    Port* port = display->GetPort(handle);
    return port && port->is_open ? port : nullptr;
}

s32 AddEvent(s64 eq, s32 handle, void* udata, InternalEvent ident, bool vblank) {
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    auto* queue = BbKernel::FindQueue(eq);
    if (!queue) {
        return ErrorInvalidEventQueue;
    }
    Event event{};
    event.ident = u64(ident);
    event.filter = Event::VideoOut;
    event.flags = Event::Add;
    event.udata = udata;
    queue->Add(event);
    (vblank ? port->vblank_queues : port->flip_queues).push_back(eq);
    return BbErr::Ok;
}

s32 DeleteEvent(s64 eq, s32 handle, InternalEvent ident, bool vblank) {
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    auto* queue = BbKernel::FindQueue(eq);
    if (!queue) {
        return ErrorInvalidEventQueue;
    }
    // (shadPS4 removed the event with the port handle as its ident: the event stayed.)
    queue->Remove(u64(ident), Event::VideoOut);
    auto& queues = vblank ? port->vblank_queues : port->flip_queues;
    std::erase(queues, eq);
    return BbErr::Ok;
}

void PS4_SYSV_ABI sceVideoOutSetBufferAttribute(BufferAttribute* attribute, PixelFormat format,
                                                u32 tiling_mode, u32 aspect_ratio, u32 width,
                                                u32 height, u32 pitch_in_pixel) {
    *attribute = {};
    attribute->pixel_format = format;
    attribute->tiling_mode = TilingMode(tiling_mode);
    attribute->aspect_ratio = s32(aspect_ratio);
    attribute->width = width;
    attribute->height = height;
    attribute->pitch_in_pixel = pitch_in_pixel;
}

s32 PS4_SYSV_ABI sceVideoOutAddFlipEvent(s64 eq, s32 handle, void* udata) {
    return AddEvent(eq, handle, udata, InternalEvent::Flip, false);
}
s32 PS4_SYSV_ABI sceVideoOutDeleteFlipEvent(s64 eq, s32 handle) {
    return DeleteEvent(eq, handle, InternalEvent::Flip, false);
}
s32 PS4_SYSV_ABI sceVideoOutAddVblankEvent(s64 eq, s32 handle, void* udata) {
    return AddEvent(eq, handle, udata, InternalEvent::Vblank, true);
}
s32 PS4_SYSV_ABI sceVideoOutDeleteVblankEvent(s64 eq, s32 handle) {
    return DeleteEvent(eq, handle, InternalEvent::Vblank, true);
}

s32 PS4_SYSV_ABI sceVideoOutRegisterBuffers(s32 handle, s32 start, void* const* addresses,
                                            s32 count, const BufferAttribute* attribute) {
    if (!addresses || !attribute) {
        return ErrorInvalidAddress;
    }
    Port* port = OpenPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    return display->RegisterBuffers(port, start, addresses, count, attribute);
}

s32 PS4_SYSV_ABI sceVideoOutUnregisterBuffers(s32 handle, s32 group) {
    Port* port = OpenPort(handle);
    return port ? display->UnregisterBuffers(port, group) : ErrorInvalidHandle;
}

s32 PS4_SYSV_ABI sceVideoOutSubmitChangeBufferAttribute(s32 handle, s32 group,
                                                        const BufferAttribute* attribute) {
    Port* port = OpenPort(handle);
    return port ? display->ChangeBufferAttribute(port, group, attribute) : ErrorInvalidHandle;
}

s32 PS4_SYSV_ABI sceVideoOutSetFlipRate(s32 handle, s32 rate) {
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    // With the frame rate free (bbport_frame_rate) a frame is shown as soon as it comes; the
    // game asks for every second vblank (30 FPS).
    port->flip_rate = BbFrameRate::Unlocked() ? 0 : rate;
    // BB_FLIP_RATE=n: the rate asked for is replaced (0: every vblank; games without frame rate
    // hooks, e.g. 60 FPS where a game asks for 30).
    if (const char* forced = bb_env("BB_FLIP_RATE"); forced && *forced) {
        port->flip_rate = std::atoi(forced);
        static s32 reported = -1;
        if (std::exchange(reported, rate) != rate) {
            std::printf("VideoOut: flip rate %d asked, %d forced (BB_FLIP_RATE)\n", rate, port->flip_rate);
        }
    }
    if (BbRemote::FrontActive()) {
        BbRemote::Front::VoSetFlipRate(handle, port->flip_rate); // bb-gpu's vblank thread uses it
    }
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceVideoOutIsFlipPending(s32 handle) {
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    std::scoped_lock lock{port->status_mutex};
    return port->flip_status.flip_pending_num;
}

s32 PS4_SYSV_ABI sceVideoOutSubmitFlip(s32 handle, s32 index, s32 flip_mode, s64 flip_arg) {
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    if (index < -1 || index >= s32(MaxDisplayBuffers) ||
        (index != -1 && port->buffers[index].group_index < 0)) {
        LOG_ERROR(Lib_VideoOut, "flip to display buffer {}: not registered", index);
        return ErrorInvalidIndex;
    }
    if (flip_mode != 1) {
        LOG_WARNING(Lib_VideoOut, "flip mode {}", flip_mode);
    }
    Vulkan::LowLatency::Get().GameFrameEnd(); // the game's thread: its frame is handed over
    return display->SubmitFlip(port, index, flip_arg) ? BbErr::Ok : ErrorFlipQueueFull;
}

s32 PS4_SYSV_ABI sceVideoOutGetEventId(const Event* event) {
    if (!event) {
        return ErrorInvalidAddress;
    }
    if (event->filter != Event::VideoOut) {
        return ErrorInvalidEvent;
    }
    switch (InternalEvent(event->ident)) {
    case InternalEvent::Flip:
        return s32(EventId::Flip);
    case InternalEvent::Vblank:
    case InternalEvent::SysVblank:
        return s32(EventId::Vblank);
    case InternalEvent::PreVblankStart:
        return s32(EventId::PreVblankStart);
    case InternalEvent::SetMode:
        return s32(EventId::SetMode);
    case InternalEvent::Position:
        return s32(EventId::Position);
    default:
        return ErrorInvalidEvent;
    }
}

/// The flip argument (sign-extended from 48 bits for flips) or vblank count.
s32 PS4_SYSV_ABI sceVideoOutGetEventData(const Event* event, s64* data) {
    if (!event || !data) {
        return ErrorInvalidAddress;
    }
    if (event->filter != Event::VideoOut) {
        return ErrorInvalidEvent;
    }
    const u64 value = event->data >> 16;
    const bool negative_flip_arg =
        event->ident == u64(InternalEvent::Flip) && s64(event->data) < 0;
    *data = s64(negative_flip_arg ? value | 0xFFFF'0000'0000'0000ull : value);
    return BbErr::Ok;
}

/// Events of this kind since the last delivery (bits 12-15 of the data).
s32 PS4_SYSV_ABI sceVideoOutGetEventCount(const Event* event) {
    if (!event) {
        return ErrorInvalidAddress;
    }
    if (event->filter != Event::VideoOut) {
        return ErrorInvalidEvent;
    }
    return s32((event->data >> 12) & 0xF);
}

s32 PS4_SYSV_ABI sceVideoOutGetFlipStatus(s32 handle, FlipStatus* status) {
    if (!status) {
        return ErrorInvalidAddress;
    }
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    std::scoped_lock lock{port->status_mutex};
    *status = port->flip_status;
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceVideoOutGetVblankStatus(s32 handle, VblankStatus* status) {
    if (!status) {
        return ErrorInvalidAddress;
    }
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    std::scoped_lock lock{port->vblank_mutex};
    *status = port->vblank_status;
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceVideoOutGetResolutionStatus(s32 handle, ResolutionStatus* status) {
    Port* port = OpenPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    *status = port->resolution;
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceVideoOutOpen(s32 user_id, s32 bus_type, s32 index, const void* param) {
    ASSERT_MSG(bus_type == BusTypeMain, "video out bus {}", bus_type);
    if (index != 0) {
        return ErrorInvalidValue;
    }
    const s32 handle = display->Open();
    if (handle < 0) {
        LOG_ERROR(Lib_VideoOut, "the display is open already");
        return ErrorResourceBusy;
    }
    return handle;
}

s32 PS4_SYSV_ABI sceVideoOutClose(s32) {
    display->Close();
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceVideoOutWaitVblank(s32 handle) {
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    std::unique_lock lock{port->vblank_mutex};
    const u64 count = port->vblank_status.count;
    port->vblank_cv.wait(lock, [&] { return port->vblank_status.count != count; });
    return BbErr::Ok;
}

struct DeviceCapabilityInfo {
    u64 capability;
};

s32 PS4_SYSV_ABI sceVideoOutGetDeviceCapabilityInfo(s32, DeviceCapabilityInfo* info) {
    // HDR when both the display and the game support it.
    const bool display_hdr =
        BbRemote::FrontActive() ? BbRemote::Front::VoIsHdrSupported() : presenter->IsHDRSupported();
    info->capability =
        display_hdr && Common::ElfInfo::Instance().GetPSFAttributes().support_hdr
            ? CapabilityBt2020Pq
            : 0;
    return BbErr::Ok;
}

struct ColorSettings {
    float gamma;
    u32 reserved[3];
};

s32 PS4_SYSV_ABI sceVideoOutColorSettingsSetGamma(ColorSettings* settings, float gamma) {
    if (gamma < 0.1f || gamma > 2.0f) {
        return ErrorInvalidValue;
    }
    settings->gamma = gamma;
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceVideoOutAdjustColor(s32 handle, const ColorSettings* settings) {
    if (!settings) {
        return ErrorInvalidAddress;
    }
    if (!display->GetPort(handle)) {
        return ErrorInvalidHandle;
    }
    if (BbRemote::FrontActive()) {
        BbRemote::Front::VoSetGamma(settings->gamma);
        return BbErr::Ok;
    }
    presenter->SetGamma(settings->gamma);
    return BbErr::Ok;
}

/// SceVideoOutMode.
struct Mode {
    u32 size;
    u8 encoding;
    u8 range;
    u8 colorimetry;
    u8 depth;
    u64 refresh_rate;
    u64 resolution;
    u8 reserved[8];
};

void PS4_SYSV_ABI sceVideoOutModeSetAny_(Mode* mode, u32 size) {
    std::memset(mode, 0xFF, size);
    mode->size = size;
}

s32 PS4_SYSV_ABI sceVideoOutConfigureOutputMode_(s32 handle, u32 reserved, const Mode* mode,
                                                 const void*, u32, u32) {
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    if (reserved != 0) {
        return ErrorInvalidValue;
    }
    switch (mode->colorimetry) {
    case ColorimetryAny:
        port->is_hdr = false;
        break;
    case ColorimetryBt2020Pq:
        if (Common::ElfInfo::Instance().GetPSFAttributes().support_hdr) {
            port->is_hdr = true;
        }
        break;
    default:
        return ErrorInvalidValue;
    }
    if (BbRemote::FrontActive()) {
        BbRemote::Front::VoSetHdr(handle, port->is_hdr); // bb-gpu presents
    }
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceVideoOutSetWindowModeMargins(s32, s32, s32) {
    return BbErr::Ok; // no window margins on the PC
}

} // namespace

s32 PS4_SYSV_ABI sceVideoOutGetBufferLabelAddress(s32 handle, uintptr_t* label_addr) {
    if (!label_addr) {
        return ErrorInvalidAddress;
    }
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    *label_addr = reinterpret_cast<uintptr_t>(port->buffer_labels);
    return s32(MaxDisplayBuffers);
}

s32 sceVideoOutSubmitEopFlip(s32 handle, u32 buf_id, u32 mode, s64 flip_arg, void**) {
    Port* port = display->GetPort(handle);
    if (!port) {
        return ErrorInvalidHandle;
    }
    // bbport (native GPU process): the flip interrupt is raised in bb-gpu's command processor,
    // which flips there, in its command stream order.
    if (BbRemote::FrontActive()) {
        return BbRemote::Front::VoSubmitEopFlip(handle, buf_id, mode, flip_arg);
    }
    if (buf_id >= MaxDisplayBuffers) {
        return ErrorInvalidIndex;
    }
    BbIrq::Once(BbIrq::Line::GfxFlip, [port, buf_id, flip_arg](BbIrq::Line) {
        // The display buffer's label is written by the GPU (end-of-pipe labels): a slow GPU may not
        // have written it yet when the CPU reaches the flip (GTX 1050: "flip interrupt out of
        // order", the game stopped). Waited for a while, then the flip goes on regardless.
        if (port->buffer_labels[buf_id] != 1) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
            while (__atomic_load_n(&port->buffer_labels[buf_id], __ATOMIC_ACQUIRE) != 1 &&
                   std::chrono::steady_clock::now() < until) {
                Common::AccurateSleep(std::chrono::microseconds(100), nullptr, false);
            }
            if (port->buffer_labels[buf_id] != 1) {
                static std::atomic<u32> told{0};
                if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                    std::printf("VideoOut: flip of buffer %u before its label (%llu): shown anyway\n",
                                buf_id, (unsigned long long)port->buffer_labels[buf_id]);
                }
            }
        }
        const bool queued = display->SubmitFlip(port, s32(buf_id), flip_arg, true);
        ASSERT_MSG(queued, "end-of-pipe flip refused");
    });
    Vulkan::LowLatency::Get().GameFrameEnd(); // the game's thread: its frame is handed over
    return BbErr::Ok;
}

namespace Remote {
Port* GetPort(s32 handle) {
    return display ? display->GetPort(handle) : nullptr;
}
s32 Open() {
    return sceVideoOutOpen(0, BusTypeMain, 0, nullptr);
}
s32 Close(s32 handle) {
    return sceVideoOutClose(handle);
}
s32 RegisterBuffers(s32 handle, s32 start, void* const* addresses, s32 count,
                    const BufferAttribute* attribute) {
    return sceVideoOutRegisterBuffers(handle, start, addresses, count, attribute);
}
s32 UnregisterBuffers(s32 handle, s32 group) {
    return sceVideoOutUnregisterBuffers(handle, group);
}
s32 ChangeBufferAttribute(s32 handle, s32 group, const BufferAttribute* attribute) {
    return sceVideoOutSubmitChangeBufferAttribute(handle, group, attribute);
}
s32 SubmitFlip(s32 handle, s32 index, s32 flip_mode, s64 flip_arg) {
    return sceVideoOutSubmitFlip(handle, index, flip_mode, flip_arg);
}
} // namespace Remote

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    display = std::make_unique<Display>(EmulatorSettings.GetInternalScreenWidth(),
                                        EmulatorSettings.GetInternalScreenHeight());
    const auto fn = [](auto* function) { return reinterpret_cast<u64>(function); };
    const std::pair<const char*, u64> exports[] = {
        {"SbU3dwp80lQ", fn(sceVideoOutGetFlipStatus)},
        {"U46NwOiJpys", fn(sceVideoOutSubmitFlip)},
        {"w3BY+tAEiQY", fn(sceVideoOutRegisterBuffers)},
        {"HXzjK9yI30k", fn(sceVideoOutAddFlipEvent)},
        {"Xru92wHJRmg", fn(sceVideoOutAddVblankEvent)},
        {"CBiu4mCE1DA", fn(sceVideoOutSetFlipRate)},
        {"i6-sR91Wt-4", fn(sceVideoOutSetBufferAttribute)},
        {"6kPnj51T62Y", fn(sceVideoOutGetResolutionStatus)},
        {"8XGijEoThE0", fn(sceVideoOutGetResolutionStatus)}, // sceVideoOutSysGetResolutionStatus
        {"Ek+VR4lcJQI", fn(sceVideoOutAddVblankEvent)},      // sceVideoOutSysAddVblankEvent
        {"Up36PTk687E", fn(sceVideoOutOpen)},
        {"zgXifHT9ErY", fn(sceVideoOutIsFlipPending)},
        {"N5KDtkIjjJ4", fn(sceVideoOutUnregisterBuffers)},
        {"OcQybQejHEY", fn(sceVideoOutGetBufferLabelAddress)},
        {"uquVH4-Du78", fn(sceVideoOutClose)},
        {"1FZBKy8HeNU", fn(sceVideoOutGetVblankStatus)},
        {"d1AjT2uZJn0", fn(sceVideoOutGetVblankStatus)},
        {"kGVLc3htQE8", fn(sceVideoOutGetDeviceCapabilityInfo)},
        {"j6RaAUlaLv0", fn(sceVideoOutWaitVblank)},
        {"U2JJtSqNKZI", fn(sceVideoOutGetEventId)},
        {"rWUTcKdkUzQ", fn(sceVideoOutGetEventData)},
        {"Mt4QHHkxkOc", fn(sceVideoOutGetEventCount)},
        {"DYhhWbJSeRg", fn(sceVideoOutColorSettingsSetGamma)},
        {"pv9CI5VC+R0", fn(sceVideoOutAdjustColor)},
        {"-Ozn0F1AFRg", fn(sceVideoOutDeleteFlipEvent)},
        {"oNOQn3knW6s", fn(sceVideoOutDeleteVblankEvent)},
        {"pjkDsgxli6c", fn(sceVideoOutModeSetAny_)},
        {"N1bEoJ4SRw4", fn(sceVideoOutConfigureOutputMode_)},
        {"MTxxrOCeSig", fn(sceVideoOutSetWindowModeMargins)},
        {"IOdgHlCGU-k", fn(sceVideoOutSubmitChangeBufferAttribute)},
    };
    for (const auto& [nid, address] : exports) {
        sym->AddSymbol(nid, "libSceVideoOut", "libSceVideoOut", Core::Loader::SymbolType::Function,
                       address);
    }
}

} // namespace BbVideoOut
