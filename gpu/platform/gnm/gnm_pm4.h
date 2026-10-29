// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: writing PM4 packets the way the PS4's Gnm driver lays them out in a game's command
// buffers. Two forms: Emit* write into a buffer at run time; the constexpr builders (Sh, Ctx, ...)
// make whole register sequences at compile time (gnm_init_state.h).
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>

#include "common/types.h"
#include "cp/cp_pm4.h"

namespace BbGnm::Pm4 {

using BbCp::Op;

/// Type-3 header: `body` dwords follow it.
constexpr u32 Header(Op op, u32 body, bool compute = false, bool predicated = false) {
    return (3u << 30) | (((body - 1) & 0x3FFFu) << 16) | (u32(op) << 8) | (u32(compute) << 1) |
           u32(predicated);
}

/// NOP payloads the driver and the translator agree on (markers, flips).
namespace Payload {
inline constexpr u32 MarkerPush = 0x68750001u;
inline constexpr u32 MarkerPop = 0x68750002u;
inline constexpr u32 MarkerSet = 0x68750003u;
inline constexpr u32 ColorMarkerPush = 0x6875000Eu;
inline constexpr u32 PatchedFlip = 0x68750776u;
inline constexpr u32 PrepareFlip = 0x68750777u;
inline constexpr u32 PrepareFlipLabel = 0x68750778u;
inline constexpr u32 PrepareFlipInterrupt = 0x68750780u;
inline constexpr u32 PrepareFlipInterruptLabel = 0x68750781u;
} // namespace Payload

// ---- Run time --------------------------------------------------------------------------------

/// A packet: header and body. Returns the dword after it.
template <typename... Body>
inline u32* Emit(u32* at, Op op, bool compute, bool predicated, Body... body) {
    *at++ = Header(op, u32(sizeof...(Body)), compute, predicated);
    ((*at++ = u32(body)), ...);
    return at;
}

/// SET_SH_REG / SET_CONTEXT_REG / SET_UCONFIG_REG: `reg` (offset in its bank), then values.
template <typename... Values>
inline u32* EmitSh(u32* at, u32 reg, Values... values) {
    return Emit(at, Op::SetShReg, false, false, reg, values...);
}
template <typename... Values>
inline u32* EmitShCompute(u32* at, u32 reg, Values... values) {
    return Emit(at, Op::SetShReg, true, false, reg, values...);
}
template <typename... Values>
inline u32* EmitCtx(u32* at, u32 reg, Values... values) {
    return Emit(at, Op::SetContextReg, false, false, reg, values...);
}
template <typename... Values>
inline u32* EmitUconfig(u32* at, u32 reg, Values... values) {
    return Emit(at, Op::SetUconfigReg, false, false, reg, values...);
}

/// A NOP whose body is `body` dwords long, the first of them 0, the others left as they are:
/// the driver pads its fixed-size packets this way. Returns the dword after it.
inline u32* EmitPadding(u32* at, u32 body) {
    at[0] = Header(Op::Nop, body);
    at[1] = 0;
    return at + 1 + body;
}

/// Pads [at, end) with one NOP.
inline void PadTo(u32* at, const u32* end) {
    EmitPadding(at, u32(end - at - 1));
}

// ---- Compile time ----------------------------------------------------------------------------

template <std::size_t N>
using Words = std::array<u32, N>;

template <typename... Body>
constexpr Words<1 + sizeof...(Body)> Packet(Op op, Body... body) {
    return {Header(op, u32(sizeof...(Body))), u32(body)...};
}
template <typename... Values>
constexpr auto Sh(u32 reg, Values... values) {
    return Packet(Op::SetShReg, reg, values...);
}
template <typename... Values>
constexpr auto Ctx(u32 reg, Values... values) {
    return Packet(Op::SetContextReg, reg, values...);
}
template <typename... Values>
constexpr auto Uconfig(u32 reg, Values... values) {
    return Packet(Op::SetUconfigReg, reg, values...);
}

/// Packets one after another.
template <std::size_t... N>
constexpr Words<(N + ...)> Join(const Words<N>&... parts) {
    Words<(N + ...)> out{};
    std::size_t at = 0;
    ((std::copy(parts.begin(), parts.end(), out.begin() + at), at += N), ...);
    return out;
}

} // namespace BbGnm::Pm4
