// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: in-game settings menu (Dear ImGui), drawn by the presenter into the swapchain image
// after the game frame, at display resolution. Insert (keyboard) or L3+R3 (gamepad) opens it;
// while it is open the game gets no pad/keyboard input. Settings live in bbport_settings.h.

#pragma once

#include <filesystem>
#include <string>
#include "common/types.h"
#include "vulkan/vk_common.h"

union SDL_Event;
struct SDL_Window;

namespace Vulkan {
class Instance;
}

namespace BbOverlay {

/// Present thread, once: the ImGui context and its Vulkan backend.
void Init(const Vulkan::Instance& instance, vk::Format format, u32 image_count);

/// Window thread, for every SDL event: true when the menu consumed it.
bool HandleEvent(const SDL_Event& event);
/// Turns SDL text input on while the menu edits a value (window thread, once per poll).
void UpdateTextInput(SDL_Window* window);

/// Whether anything is drawn this frame (menu open or FPS counter on).
bool Visible();
/// A line of text over the game until called with null (bbgpu_overlay_banner).
void SetBanner(const char* text);
/// A trophy unlocked (bbport_trophies.cpp): announced at the top right for a few seconds, one
/// after another. grade: P G S B, '?' unknown; icon: the trophy's PNG (empty: the grade's badge).
/// delay: seconds of drawn frames before it shows (BB_TROPHY_TEST: past the game's start).
void ShowTrophy(char grade, const std::string& name, const std::filesystem::path& icon, float delay = 0.0f);

/// Present thread: draws into `view` (layout ColorAttachmentOptimal).
void Render(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent);

/// The menu or the text dialog is open: the game's input is held neutral.
bool CapturesInput();

/// The settings menu is open: the window shows the system cursor over it.
bool MenuOpen();

/// Window thread: the game's text dialog (ImeDialog) state, drawn as a box over the frame.
void SetTextPrompt(bool active, const std::string& prompt, const std::string& text);

} // namespace BbOverlay
