// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what the overlay's files share (bbport_overlay.cpp: the window, input and drawing;
// bbport_overlay_sections.cpp: the menu's sections).
#pragma once

#include <atomic>

#include "bbport_settings.h"
#include "common/types.h"

// render/pipeline_cache.cpp: the shader compiles (the "Compiling shaders" line, the Shaders section).
namespace Vulkan {
extern std::atomic<u32> g_bb_compiles_total, g_bb_background_pipelines_total,
    g_bb_background_pipelines_done;
extern std::atomic<bool> g_bb_shader_import_active;
} // namespace Vulkan
using Vulkan::g_bb_background_pipelines_done;
using Vulkan::g_bb_background_pipelines_total;
using Vulkan::g_bb_compiles_total;
using Vulkan::g_bb_shader_import_active;

namespace BbOverlay::detail {

inline bool dirty = false; // settings changed while open: saved on close
inline float base_scale = 1.0f;

bool OverlayTrace();

// Marks the settings dirty when a widget changed them.
template <typename T>
void Store(std::atomic<T>& target, T value, bool changed) {
    if (changed) {
        target = value;
        dirty = true;
    }
}

void Checkbox(const char* label, std::atomic<bool>& value);
void Slider(const char* label, std::atomic<float>& value, float lo, float hi);
/// A "(?)" after the last item, the text in its tooltip.
void Hint(const char* text);
/// Text about a problem (an upscaler that cannot run, a missing file), in orange.
void ProblemText(const char* text);

// The menu's sections (bbport_overlay_sections.cpp).
void ShadersSection();
bool UpscalerSection(BbSettings::Values& s);
void ReactiveMaskSection(BbSettings::Values& s, bool taa);
void OutputSection(BbSettings::Values& s);
void GameEffectsSection(BbSettings::Values& s);
void FrameRateLimitSection(BbSettings::Values& s);

} // namespace BbOverlay::detail
