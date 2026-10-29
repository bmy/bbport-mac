// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's frame rate set free by hooks in its code (BB_FPS=uncap, the default), not by
// patches: its frame pacer no longer waits for a 30 FPS frame, its step gets the frame's real time,
// and what it ties to a frame (a 30 FPS frame's time, a per-frame step, smoothing and damping) reads
// values the hooks recompute from that time every frame (bbport_frame_rate.cpp).
#pragma once

#include <cstdint>

namespace BbFrameRate {
/// Installs the hooks (at load, before the game runs: bbgpu_patch_image).
void PatchImage(unsigned char* image, std::uint64_t size);
/// Whether the hooks run the game's frame rate: the video out then shows a frame as soon as the
/// game submits it (the game's own flip rate asks for 30 FPS).
bool Unlocked();
/// A frame rate limit while the game runs (30..480; 0: the launched mode and limit). The game's
/// frame pacer waits for it itself. Lowers a running game to 30 FPS for what the unlocked rate
/// breaks (the cinematic of Laurence, the optional boss: a hang above 30 FPS, as with the
/// community 60 FPS patch).
void SetLimit(int fps);
int Limit();
} // namespace BbFrameRate
