// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the control block shared by the game process and the native GPU process
// (docs/macos-native-gpu.md). It is part of the guest memory pool (its host span) and mapped at
// the same fixed address in both, just above the guest's range, so pointers into it are valid on
// both sides: the VideoOut buffer labels live here, and the guest's GPU commands write them.
#pragma once

#include <atomic>
#include <cstdint>

#include "remote/bb_channel.h"

namespace BbRemote {

inline constexpr std::uint64_t GuestBegin = 0x7000000000ull; ///< runtime_memory.c USER_MIN (macOS)
inline constexpr std::uint64_t GuestEnd = 0xfc00000000ull;   ///< USER_MAX
inline constexpr std::uint64_t LowBegin = 0x0800000000ull;   ///< the low region (game image)
inline constexpr std::uint64_t LowEnd = 0x0fc0000000ull;     ///< platform.h BB_LOW_MAX (macOS)

inline constexpr std::uint64_t ControlAddress = GuestEnd;
inline constexpr std::uint64_t ControlBytes = 8ull << 20;
/// Pool bytes after direct and flexible memory for the control block and shared image data.
inline constexpr std::uint64_t HostSpanBytes = 512ull << 20;

/// State both processes read and write directly (no messages).
struct SharedState {
    alignas(64) std::uint64_t vo_labels[16]; ///< BbVideoOut::Port::buffer_labels (guest-visible)
    alignas(64) std::atomic<std::uint32_t> overlay_captures_input{0}; ///< written by bb-gpu
    std::atomic<std::uint32_t> gpu_ready{0};
    /// The game window's keyboard (SDL scancodes, 1 while held), written by bb-gpu's window loop.
    alignas(64) std::uint8_t keyboard[512];
    /// Mouse look (bbport 0.51): written by bb-gpu's window loop, taken by the game process's pad
    /// (bbgpu_mouse_*). Motion in 1/1024 pixels and wheel notches pile up until taken.
    alignas(64) std::atomic<std::int64_t> mouse_dx{0}, mouse_dy{0};
    std::atomic<std::uint32_t> wheel_up{0}, wheel_down{0};
    std::atomic<std::uint32_t> mouse_captured{0}, mouse_buttons{0};
    /// The game's mouse_look setting for bb-gpu's window: 0 not said yet, 1 off, 2 on.
    std::atomic<std::uint32_t> mouse_look{0};
    /// Requests from the game process's pad to capture the mouse (bbgpu_mouse_capture), counted:
    /// bb-gpu's window loop takes them.
    std::atomic<std::uint32_t> mouse_capture_requests{0};
    /// The button prompts (bbport 0.6): runtime_pad.c's prompt state, published by the game
    /// process once a frame for bb-gpu's texture uploads (bbport_button_prompts.cpp).
    alignas(64) std::atomic<std::uint32_t> prompt_generation{0};
    std::atomic<std::int32_t> prompt_device{0};
    std::atomic<std::int32_t> pad_mouse_look{0};
    /// runtime_pad_steam_only, for bb-gpu's window (keys and mouse a Steam pad echoes).
    std::atomic<std::int32_t> steam_only{0};
    /// Keys pressed since the game process's pad last read them (bbport 0.6's short taps,
    /// runtime_pad.c key_latch): bit n = SDL scancode n, set by bb-gpu's window, taken (cleared)
    /// by the game process.
    alignas(64) std::atomic<std::uint64_t> key_latch[8]{};
    /// The pad recording clock (bbgpu_world_active, bbgpu_world_us: Display::Flip in bb-gpu),
    /// published by bb-gpu's window loop for the game process's pad.
    alignas(64) std::atomic<std::uint64_t> world_us{0};
    std::atomic<std::uint32_t> world_active{0};
};

struct ControlBlock {
    ChannelLayout channel;
    /// Page protection (MsgProtect, bb-gpu -> game process), served by a thread of its own:
    /// it needs nothing but the runtime's lock, so it is never stuck behind a VideoOut or event
    /// queue lock, and protections apply in the order bb-gpu decides them. The other way the
    /// runtime's chunk allocator (MsgChunkAlloc, MsgChunkFree), served by a bb-gpu thread that
    /// never waits for the game process: the runtime may call it with its lock held.
    ChannelLayout protect;
    /// bbport: the guest hooks' fresh ranges as calls (BB_REMOTE_FRESH_SYNC=1, game process ->
    /// bb-gpu), served by a bb-gpu thread that does nothing else (remote_back.cpp FreshThread).
    ChannelLayout fresh;
    alignas(4096) SharedState state;
};
static_assert(sizeof(ControlBlock) <= ControlBytes);
static_assert(ControlBytes % 16384 == 0 && HostSpanBytes % 16384 == 0);

} // namespace BbRemote
