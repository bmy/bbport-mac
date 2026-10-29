// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: hooks in the game's code that write GPU memory. Where the game
// copies resource data into memory the GPU uses, the GPU side is told once the data is there, as a
// driver is told of a resource update; the write needs no page protection to be seen. Each site
// is checked byte for byte against the expected code (another game version: not installed).
#pragma once

#include <cstdint>

namespace BbGuestHooks {
/// [address, address + size) of GPU memory the game has just handed out again: its old contents
/// are dead and whatever writes it next may be code the GPU side does not hear of.
using RangeCallback = void (*)(std::uint64_t address, std::uint64_t size);

/// Installs the hooks once (GPU side, after the game is loaded).
/// `on_range_freed`: a GraphicsPrivate heap block the game freed (BB_HEAP_GROW): its VRAM copies go.
/// `on_range_allocated_block`: a GraphicsPrivate block the game allocated (its VRAM copies stay in it).
void Install(RangeCallback on_gpu_range_allocated, RangeCallback on_range_freed = nullptr,
             RangeCallback on_range_allocated_block = nullptr);

/// The output's aspect ratio (the upscaler, when its size changes): the camera's from its next
/// update (wider: more to the sides, taller: more above and below), and the HUD markers' places
/// in the UI the port draws in the middle of the output at its height (width). 16:9: the game's.
void SetDisplayAspect(float aspect);
/// The current camera's state the hooks above write (its projection at +0x50), 0 until known.
std::uint64_t CameraState();
/// The free camera (community patch) is on: the mouse drives it, not the follow camera.
bool FreeCameraActive();
/// The game effects chosen in the settings take effect now (a no-op when nothing changed, or
/// when the effects are only set by the startup patches, BB_HOT_EFFECTS=0).
void ApplyGameEffects();

/// Native mouse look (BB_NATIVE_MOUSE, on by default): whether the camera hook is in, and a turn
/// (radians: pitch, yaw) for the camera's next update; false while the camera does not take the
/// turns (lock-on, menus: the caller drives the stick instead).
bool CameraLookAvailable();
bool CameraLookAdd(float pitch, float yaw);
} // namespace BbGuestHooks
