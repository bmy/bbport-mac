// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: libSceVideoOut, the display: display buffers the game registers, flips (now, or at the
// end of a GPU frame), vblank and flip events, and the statuses games poll. One port (the main
// display, handle 1).
#pragma once

#include <array>
#include <condition_variable>
#include <mutex>
#include <vector>

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace BbVideoOut {

inline constexpr std::size_t MaxDisplayBuffers = 16;
inline constexpr std::size_t MaxDisplayBufferGroups = 4;

enum class PixelFormat : u32 {
    Unknown,
    A8R8G8B8Srgb = 0x80000000,
    A8B8G8R8Srgb = 0x80002200,
    A2R10G10B10 = 0x88060000,
    A2R10G10B10Srgb = 0x88000000,
    A2R10G10B10Bt2020Pq = 0x88740000,
    A16R16G16B16Float = 0xC1060000,
};

enum class TilingMode : s32 {
    Tile = 0,
    Linear = 1,
};

/// SceVideoOutBufferAttribute.
struct BufferAttribute {
    PixelFormat pixel_format;
    TilingMode tiling_mode;
    s32 aspect_ratio;
    u32 width;
    u32 height;
    u32 pitch_in_pixel;
    u32 option;
    u32 reserved0;
    u64 reserved1;
};

/// Display buffers registered together share an attribute.
struct BufferAttributeGroup {
    bool is_occupied;
    BufferAttribute attrib;
};

struct DisplayBuffer {
    s32 group_index = -1;
    uintptr_t address_left = 0;
    uintptr_t address_right = 0;
};

/// SceVideoOutFlipStatus.
struct FlipStatus {
    u64 count = 0;
    u64 process_time = 0;
    u64 tsc = 0;
    s64 flip_arg = -1;
    u64 submit_tsc = 0;
    u64 reserved0 = 0;
    s32 gc_queue_num = 0;     ///< flips waiting for the end of a GPU frame
    s32 flip_pending_num = 0; ///< flips not shown yet
    s32 current_buffer = -1;
    u32 reserved1 = 0;
};

/// SceVideoOutResolutionStatus.
struct ResolutionStatus {
    s32 full_width = 1280;
    s32 full_height = 720;
    s32 pane_width = 1280;
    s32 pane_height = 720;
    u64 refresh_rate = 3; // 59.94 Hz
    float screen_size_in_inch = 50;
    u16 flags = 0;
    u16 reserved0 = 0;
    u32 reserved1[3] = {};
};

/// SceVideoOutVblankStatus.
struct VblankStatus {
    u64 count = 0;
    u64 process_time = 0;
    u64 tsc = 0;
    u64 reserved = 0;
    u8 flags = 0;
    u8 pad[7] = {};
};

/// Event idents in the kernel queues (sceVideoOutGetEventId maps them to the public ones).
enum class InternalEvent : u64 {
    Flip = 0x6,
    Vblank = 0x7,
    SetMode = 0x51,
    Position = 0x58,
    PreVblankStart = 0x59,
    SysVblank = 0x63,
};

/// The main display.
struct Port {
    ResolutionStatus resolution;
    std::array<DisplayBuffer, MaxDisplayBuffers> buffers;
    std::array<u64, MaxDisplayBuffers> own_labels{};
    /// One label per buffer: 1 while a flip to it is queued (the GPU waits on it to draw again).
    /// Contiguous. bbport: own_labels, or with the native GPU process the memory both processes
    /// share (docs/macos-native-gpu.md): the guest's GPU commands write them. Stores to them are
    /// release stores (StoreLabel): the guest polls them in the game process.
    u64* buffer_labels = own_labels.data();
    std::array<BufferAttributeGroup, MaxDisplayBufferGroups> groups{};
    FlipStatus flip_status;
    VblankStatus vblank_status;
    std::vector<s64> flip_queues;   ///< event queues with a flip event
    std::vector<s64> vblank_queues; ///< event queues with a vblank event
    std::mutex vblank_mutex;        ///< vblank status, labels' waiters
    std::mutex status_mutex;        ///< flip status, attributes
    std::condition_variable label_cv;
    std::condition_variable vblank_cv;
    int flip_rate = 0; ///< show a flip every (flip_rate + 1) vblanks
    int prev_index = -1;
    /// Buffers given back to the game when the flip after theirs was queued (Display::QueueFlip,
    /// BB_FLIP_AHEAD), per buffer: the release when that flip is shown is skipped.
    std::array<u32, MaxDisplayBuffers> released_ahead{};
    int last_queued = -1; ///< the buffer of the last flip queued (-1: a blank one or none)
    bool is_open = false;
    bool is_hdr = false;

    /// `address` is one of the buffer labels (the command processor sleeps on those).
    bool IsVoLabel(const u64* address) const {
        return address >= &buffer_labels[0] && address <= &buffer_labels[MaxDisplayBuffers - 1];
    }
    void StoreLabel(std::size_t index, u64 value) {
        __atomic_store_n(&buffer_labels[index], value, __ATOMIC_RELEASE);
    }
    void WaitVoLabel(auto&& ready) {
        std::unique_lock lock{vblank_mutex};
        label_cv.wait(lock, ready);
    }
    void SignalVoLabel() {
        std::scoped_lock lock{vblank_mutex};
        label_cv.notify_one();
    }
};

/// The address of the 16 buffer labels (returns 16, the count, as the console does).
s32 PS4_SYSV_ABI sceVideoOutGetBufferLabelAddress(s32 handle, uintptr_t* label_addr);
/// Queues a flip that happens at the end of the GPU frame (the flip interrupt); Gnm calls it
/// for sceGnmSubmitAndFlipCommandBuffers.
s32 sceVideoOutSubmitEopFlip(s32 handle, u32 buf_id, u32 mode, s64 flip_arg, void** unk);

void RegisterLib(Core::Loader::SymbolsResolver* sym);

/// bbport (native GPU process, docs/macos-native-gpu.md): bb-gpu's message handlers call the
/// library as the game does in-process.
namespace Remote {
Port* GetPort(s32 handle);
s32 Open();
s32 Close(s32 handle);
s32 RegisterBuffers(s32 handle, s32 start, void* const* addresses, s32 count,
                    const BufferAttribute* attribute);
s32 UnregisterBuffers(s32 handle, s32 group);
s32 ChangeBufferAttribute(s32 handle, s32 group, const BufferAttribute* attribute);
s32 SubmitFlip(s32 handle, s32 index, s32 flip_mode, s64 flip_arg);
} // namespace Remote

} // namespace BbVideoOut
