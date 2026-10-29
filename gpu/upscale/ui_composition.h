// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace Vulkan::UiComposition {

enum class Background { None, Copy, Temporal };

// UI composition must not require scene depth/camera or an active FSR context.
constexpr Background Choose(bool scaled, bool ui_draw, bool scene_ready, bool fsr_active) {
    if (!scaled || !ui_draw) {
        return Background::None;
    }
    return scene_ready && fsr_active ? Background::Temporal : Background::Copy;
}

inline bool NativeViewport(float width, float height) {
    return std::abs(std::abs(width) - 1920.0f) < 0.5f &&
           std::abs(std::abs(height) - 1080.0f) < 0.5f;
}

// Scaleform draws, including the first stencil/movie pass. A native-size viewport
// alone also matches every fullscreen post pass when guest targets stay at 1080p.
constexpr bool MovieShader(uint64_t hash) {
    return hash == 0x34e8a281 || hash == 0x81d336ce || hash == 0x09957251 ||
           hash == 0x24042a9b || hash == 0xa400228b ||
           // The title menu while it animates (its highlight): without these its frames were
           // shown at the game's 1080p between output-size ones, a flicker (BB_UI_TRACE=1).
           hash == 0xbb4d5f8c || hash == 0xe94b5b06 || hash == 0x9951d199 ||
           hash == 0x44072322 || hash == 0xa334a845;
}

// Where a pass drawn into the output-size image goes: it fills the output. The game's UI is drawn
// for the output's aspect ratio already: its Scaleform movies show their wider (taller) view
// squeezed into the 1920x1080 frame (BbGuestHooks: their viewport has the output's pixel aspect
// ratio), which the output stretches back.
inline std::array<float, 2> Scale(uint32_t guest_width, uint32_t guest_height, uint32_t output_width,
                                  uint32_t output_height, bool native_coordinates) {
    return {float(output_width) / float(native_coordinates ? 1920 : guest_width),
            float(output_height) / float(native_coordinates ? 1080 : guest_height)};
}

// Where the game's 16:9 screen ends in such a frame (normalized, at most 1): x on a wider output,
// y on a taller one.
inline std::array<float, 2> ScreenEdges(uint32_t output_width, uint32_t output_height) {
    const float k = float(output_width) * 1080.0f / (float(output_height) * 1920.0f);
    return {std::min(1.0f, 1.0f / k), std::min(1.0f, k)};
}

// The UI's full-screen fills and backgrounds (the menus' dimming, fades) reach the output's
// edges: their vertices in the corners of the game's screen, or past them, are moved to the
// output's corners by the vertex shader (push constant motion_param: the screen's edges, x in the
// high 16 bits, y in bits 1-15, bit 0 set when clip space +y is the bottom of the screen (the
// draw's viewport y scale positive); EmitUiEdgeStretch). 0: none.
inline uint32_t EdgeParam(float edge_x, float edge_y, bool y_down) {
    const auto q = [](float f, float max) { return uint32_t(std::lround(std::clamp(f, 0.0f, 1.0f) * max)); };
    return std::max(2u, q(edge_x, 65535.0f) << 16 | q(edge_y, 32767.0f) << 1) | (y_down ? 1u : 0u);
}

} // namespace Vulkan::UiComposition
