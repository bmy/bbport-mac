// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: button prompts for the device in use. The game draws its key guide ("<?conclusion?>:
// OK", item pick-ups, tutorials) with 32x32 icons of PlayStation buttons (KG_OK, KG_Cancel,
// KG_L1, ... in menu/common.tpf.dcx), which every menu movie shows as external images. Here those
// icons are known by their bytes (hashed from the game's own file, so any release and any mod of
// it) and, while the keyboard and mouse or another gamepad was used last, sampled from icons drawn
// by the port: the bound key, mouse button or that pad's button (runtime_pad_prompt).
// BB_BUTTON_PROMPTS=0: the game's icons always.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace BbPrompts {

/// The port's icons: square, RGBA8 premultiplied, this many pixels a side.
constexpr std::uint32_t IconSize = 128;

/// Starts reading the game's icons (a thread; IconOf waits for it once).
void Start();

/// Whether the prompts may change at all (BB_BUTTON_PROMPTS, the game's icons found).
bool Enabled();

/// The icon a texture of the game holds: its index + 1, 0 when it is none of them. Only 32x32
/// single-level textures are asked (TextureCache::RefreshImage).
std::uint32_t IconOf(const void* data, std::size_t size);
/// The game's icon `icon` (IconOf's numbering) as in its file, 1024 bytes; null when unknown.
const std::uint8_t* IconBytes(std::uint32_t icon);

/// The game's icons as large as they are in its files (32x32).
constexpr std::uint32_t GameIconSize = 32;

/// Where an atlas of the game's (a texture its HUD packs several key guide icons into, e.g. the
/// cross of "Examine" at a lamp) holds one: the 32x32 cell at (x, y) shows icon `icon` (IconOf's
/// numbering) as the icon's own texture does.
struct AtlasCell {
    std::uint32_t x, y, icon;
};
/// The atlas a texture of the game is: its index + 1, 0 when none. Known by its bytes (hashed
/// from the game's file); only textures of an atlas's size are hashed.
std::uint32_t AtlasOf(const void* data, std::size_t size, std::uint32_t width, std::uint32_t height);
/// The cells of atlas `atlas` (AtlasOf's numbering) whose icons were found.
std::span<const AtlasCell> AtlasCells(std::uint32_t atlas);

/// The icons for the device in use. `generation` changes whenever they do; `drawn[i]` false: the
/// game's own icon i stays (a PlayStation pad, the d-pad on any pad, an unbound input).
struct Set {
    std::uint32_t generation = 0;
    std::uint32_t count = 0;
    std::vector<bool> drawn;
    std::vector<std::uint32_t> pixels; ///< count * IconSize * IconSize
};

/// The set when it changed since `generation` (cheap otherwise: one atomic load).
bool Update(std::uint32_t generation, Set& out);

} // namespace BbPrompts
