// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the GPU's interrupt lines as the system libraries see them. The command processor
// raises a line (end of pipe, flip, compute release, idle); libraries listen: Gnm turns end-of-pipe
// interrupts into the game's event queue events, VideoOut does a flip on each flip interrupt.
#pragma once

#include <functional>

#include "common/types.h"

namespace BbIrq {

enum class Line : u32 {
    Compute0RelMem = 0x00, ///< RELEASE_MEM on compute pipe 0 (to 6)
    Compute6RelMem = 0x06,
    GfxFlip = 0x08,
    GpuIdle = 0x09,
    GfxEop = 0x40, ///< end of pipe on the graphics ring
};
inline constexpr u32 LineCount = 0x41;

using Handler = std::function<void(Line)>;

/// Calls `handler` on every interrupt of `line` until Forget(line, owner).
void Listen(Line line, const void* owner, Handler handler);
void Forget(Line line, const void* owner);
/// Calls `handler` on one interrupt of `line` (handlers queue up, one per interrupt).
void Once(Line line, Handler handler);
/// Runs the line's handlers (on the raising thread).
void Raise(Line line);

/// bbport (native GPU process, docs/macos-native-gpu.md): in bb-gpu, every interrupt also goes to
/// the game process, whose listeners (Gnm's event queues) run there. Set once, before any raise.
inline void (*forward)(Line) = nullptr;

} // namespace BbIrq
