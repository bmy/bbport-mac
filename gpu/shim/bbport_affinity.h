// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: physical cores of their own for the two threads every frame waits for (the draw
// pipeline's stages: bb:GpuComm and bb:DrawRec). With SMT the scheduler put busy threads on their
// sibling hardware threads: a heavy scene (1500 draws a frame) ran 16% slower on 6 cores / 12
// threads than on the same 6 cores without SMT, 12% on 8 / 16 (bbport_affinity.cpp).
#pragma once

namespace BbAffinity {

/// Starts the thread that keeps the placement (Linux, CPUs with SMT and 6+ cores; BB_CORE_RESERVE=1
/// or 0 forces it on or off).
void Start();

} // namespace BbAffinity
