// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_cursor.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <SDL3/SDL.h>

namespace BbCursor {

namespace {

struct Pt {
    float x, y;
};

// A worn silver lancet: a long point, hooked guard, bound grip and small dark garnet.
// Keep the upper-left tip at the existing hotspot; the silhouette must still read at 32 px.
constexpr std::array<Pt, 13> Shape = {{{1.0f, 1.0f},
                                     {2.8f, 21.0f},
                                     {6.8f, 17.8f},
                                     {11.0f, 26.2f},
                                     {10.5f, 28.1f},
                                     {13.5f, 30.0f},
                                     {16.1f, 28.0f},
                                     {14.0f, 26.0f},
                                     {10.0f, 17.0f},
                                     {17.8f, 16.2f},
                                     {15.8f, 13.8f},
                                     {11.5f, 14.4f},
                                     {1.0f, 1.0f}}};
constexpr int ShapeCount = static_cast<int>(Shape.size()) - 1;

bool Inside(float x, float y) {
    bool in = false;
    for (int i = 0, j = ShapeCount - 1; i < ShapeCount; j = i++) {
        const Pt a = Shape[i], b = Shape[j];
        if (((a.y > y) != (b.y > y)) && (x < (b.x - a.x) * (y - a.y) / (b.y - a.y) + a.x)) {
            in = !in;
        }
    }
    return in;
}

float EdgeDistance(float x, float y) {
    float best = 1e9f;
    for (int i = 0, j = ShapeCount - 1; i < ShapeCount; j = i++) {
        const Pt a = Shape[j], b = Shape[i];
        const float dx = b.x - a.x, dy = b.y - a.y;
        const float t = std::clamp(((x - a.x) * dx + (y - a.y) * dy) / (dx * dx + dy * dy), 0.0f, 1.0f);
        const float px = a.x + t * dx - x, py = a.y + t * dy - y;
        best = std::min(best, std::sqrt(px * px + py * py));
    }
    return best;
}

struct Rgba {
    float r, g, b, a;
};

Rgba Mix(Rgba a, Rgba b, float t) {
    t = std::clamp(t, 0.0f, 1.0f);
    return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t, a.a + (b.a - a.a) * t};
}

// One sub-sample of the cursor at (x, y) in the 32 box.
Rgba Sample(float x, float y) {
    if (!Inside(x, y)) {
        // A soft shadow below and right, so the pointer reads over bright scenes too.
        if (Inside(x - 0.9f, y - 1.1f)) {
            return {0.0f, 0.0f, 0.0f, 0.35f};
        }
        return {0, 0, 0, 0};
    }
    const float edge = EdgeDistance(x, y);
    constexpr Rgba Outline = {0.055f, 0.060f, 0.065f, 1.0f};
    constexpr Rgba Silver = {0.74f, 0.75f, 0.72f, 1.0f};
    constexpr Rgba Steel = {0.27f, 0.30f, 0.31f, 1.0f};
    constexpr Rgba Highlight = {0.91f, 0.90f, 0.83f, 1.0f};

    // Two bevels, divided by a fine ridge. Low-contrast etched marks break up the
    // metal without turning the small pointer into a noisy texture.
    const float ridge = 1.0f + (y - 1.0f) * 0.44f;
    Rgba body = x < ridge ? Silver : Steel;
    body = Mix(body, Highlight, std::max(0.0f, 1.0f - std::abs(x - ridge) / 0.45f) * 0.7f);
    const float etching = std::abs(x - ridge - (std::fmod(y, 4.0f) - 2.0f) * 0.45f);
    if (y > 6.0f && y < 14.0f && etching < 0.18f) {
        body = Mix(body, Outline, 0.35f);
    }
    // The hooked crossguard catches light; the handle is almost black, with silver wire.
    if (y > 14.0f && y < 21.0f && std::abs(y + x * 0.34f - 20.0f) < 0.65f) {
        body = Silver;
    }
    if (y > 19.0f && y < 26.5f && x > 6.0f) {
        body = {0.13f, 0.14f, 0.15f, 1.0f};
        if (std::fmod(y + x * 0.5f, 2.7f) < 0.55f) {
            body = Mix(body, Silver, 0.5f);
        }
    }
    if (y >= 26.5f) {
        // A restrained oxblood inset in the pommel, framed by tarnished silver.
        body = {0.24f, 0.075f, 0.085f, 1.0f};
        if (edge < 1.0f) {
            body = Silver;
        }
    }
    // A narrow pale rim against dark menus, outside a charcoal contour against snow/light.
    if (edge < 1.15f) {
        body = Mix(body, Highlight, 0.7f);
    }
    return Mix(body, Outline, std::clamp((0.7f - edge) / 0.45f, 0.0f, 1.0f));
}

SDL_Surface* Render(int size) {
    SDL_Surface* surface = SDL_CreateSurface(size, size, SDL_PIXELFORMAT_RGBA32);
    if (!surface) {
        return nullptr;
    }
    const float scale = 32.0f / static_cast<float>(size);
    constexpr int Sub = 4;
    for (int py = 0; py < size; ++py) {
        auto* row = reinterpret_cast<std::uint8_t*>(surface->pixels) + py * surface->pitch;
        for (int px = 0; px < size; ++px) {
            float r = 0, g = 0, b = 0, a = 0;
            for (int sy = 0; sy < Sub; ++sy) {
                for (int sx = 0; sx < Sub; ++sx) {
                    const Rgba c = Sample((static_cast<float>(px) + (sx + 0.5f) / Sub) * scale,
                                          (static_cast<float>(py) + (sy + 0.5f) / Sub) * scale);
                    r += c.r * c.a;
                    g += c.g * c.a;
                    b += c.b * c.a;
                    a += c.a;
                }
            }
            const float n = static_cast<float>(Sub * Sub);
            const float alpha = a / n;
            const float inv = a > 0 ? 1.0f / a : 0.0f;
            row[px * 4 + 0] = static_cast<std::uint8_t>(std::clamp(r * inv, 0.0f, 1.0f) * 255.0f + 0.5f);
            row[px * 4 + 1] = static_cast<std::uint8_t>(std::clamp(g * inv, 0.0f, 1.0f) * 255.0f + 0.5f);
            row[px * 4 + 2] = static_cast<std::uint8_t>(std::clamp(b * inv, 0.0f, 1.0f) * 255.0f + 0.5f);
            row[px * 4 + 3] = static_cast<std::uint8_t>(std::clamp(alpha, 0.0f, 1.0f) * 255.0f + 0.5f);
        }
    }
    return surface;
}

} // namespace

SDL_Cursor* Create() {
    if (bb_env_off("BB_CURSOR")) {
        return nullptr;
    }
    SDL_Surface* base = Render(32);
    if (!base) {
        return nullptr;
    }
    if (SDL_Surface* large = Render(64)) {
        SDL_AddSurfaceAlternateImage(base, large); // high-DPI displays
        SDL_DestroySurface(large);
    }
    SDL_Cursor* cursor = SDL_CreateColorCursor(base, 1, 1);
    SDL_DestroySurface(base);
    return cursor;
}

} // namespace BbCursor
