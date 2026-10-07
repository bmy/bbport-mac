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
inline constexpr std::uint64_t ControlBytes = 4ull << 20;
/// Pool bytes after direct and flexible memory for the control block and shared image data.
inline constexpr std::uint64_t HostSpanBytes = 512ull << 20;

/// State both processes read and write directly (no messages).
struct SharedState {
    alignas(64) std::uint64_t vo_labels[16]; ///< VideoOutPort::buffer_labels (guest-visible)
    alignas(64) std::atomic<std::uint32_t> overlay_captures_input{0}; ///< written by bb-gpu
    std::atomic<std::uint32_t> gpu_ready{0};
};

struct ControlBlock {
    ChannelLayout channel;
    alignas(4096) SharedState state;
};
static_assert(sizeof(ControlBlock) <= ControlBytes);
static_assert(ControlBytes % 16384 == 0 && HostSpanBytes % 16384 == 0);

} // namespace BbRemote
