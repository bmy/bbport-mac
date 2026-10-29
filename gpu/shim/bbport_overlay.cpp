// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_overlay.h"
#include "bbport_overlay_internal.h"
#include "shim/bb_env.h"

#include "bbport_toggles.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <unordered_map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <SDL3/SDL.h>
#include <fmt/format.h>
#include "bbport_guest_hooks.h"
#include "bbport_frame_rate.h"
#include "bbport_settings.h"
#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "imgui_internal.h" // RegisterUserTexture: the trophy icons
#include "vulkan/instance.h"
#include "render/compile_log.h"
#include "render/scheduler.h"

// DejaVu Sans (Cyrillic), embedded (third_party/fonts, Bitstream Vera license).
#ifdef __APPLE__
// bbport (macOS): Mach-O: read-only data section, C symbols carry a leading underscore,
// .private_extern hides.
asm(".section __TEXT,__const\n"
    ".p2align 4\n"
    ".private_extern _bb_font_ttf\n"
    ".globl _bb_font_ttf\n"
    "_bb_font_ttf:\n"
    ".incbin \"" BB_FONT_PATH "\"\n"
    ".private_extern _bb_font_ttf_end\n"
    ".globl _bb_font_ttf_end\n"
    "_bb_font_ttf_end:\n"
    ".text\n");
#else
#ifdef _WIN32
#define BB_FONT_SECTION ".section .rdata,\"dr\"\n"
#define BB_FONT_HIDDEN(name) ""
#define BB_FONT_END ".text\n"
#else
#define BB_FONT_SECTION ".section .rodata\n"
#define BB_FONT_HIDDEN(name) ".hidden " name "\n"
#define BB_FONT_END ".previous\n"
#endif
asm(BB_FONT_SECTION
    ".balign 16\n"
    BB_FONT_HIDDEN("bb_font_ttf")
    ".global bb_font_ttf\n"
    "bb_font_ttf:\n"
    ".incbin \"" BB_FONT_PATH "\"\n"
    BB_FONT_HIDDEN("bb_font_ttf_end")
    ".global bb_font_ttf_end\n"
    "bb_font_ttf_end:\n"
    BB_FONT_END);
#endif
extern "C" const unsigned char bb_font_ttf[];
extern "C" const unsigned char bb_font_ttf_end[];

extern "C" void runtime_restart(void); // bb-probe (probe.c)


namespace BbOverlay {

namespace {

using namespace detail;

std::mutex imgui_mutex; // the ImGui context: window thread (input) and present thread
bool initialized = false;
std::atomic<bool> menu_open{false};
bool l3_down = false, r3_down = false;
// Issue #304: L3+R3 opens the menu only when held (OpenHoldMs), not when both sticks are pressed in
// a fight (lock-on and camera reset); closing is immediate. The hold starts at `l3r3_since`
// (steady clock ns, 0: not held) and is checked at each present (Visible); `open_requested` opens
// it in Render, under the ImGui lock.
constexpr long long OpenHoldMs = 800;
std::atomic<long long> l3r3_since{0};
std::atomic<bool> open_requested{false};
long long NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
// The game's text dialog (ImeDialog, the character name), typed on the keyboard: drawn while it
// is open. In fullscreen the window title that showed it is not visible (issues #17, #19).
std::mutex prompt_mutex;
std::atomic<bool> prompt_active{false};
/// A module's line over the game (bbgpu_overlay_banner: BBSC's "joining the host" during its
/// loads, where the game itself shows a black screen).
std::mutex banner_mutex;
std::string banner_text;
std::atomic<bool> banner_active{false};
std::string prompt_title, prompt_text;
/// Trophies to announce, oldest first; the front one is on screen (as shadPS4's trophy_ui.cpp:
/// slides in from the right, stays, fades out over the last second).
struct TrophyToast {
    char grade;
    std::string name;
    float elapsed = 0.0f;
    std::filesystem::path icon;
};
// The trophy icons (TROPnnn.PNG, 240x240), decoded once and handed to the Vulkan backend as user
// textures (it uploads them with the font atlas); kept for the session (40 at most). null: none.
std::unordered_map<std::string, ImTextureData*> trophy_icons;
std::mutex trophy_mutex;
std::deque<TrophyToast> trophy_queue;
std::atomic<bool> trophy_active{false};
constexpr float TrophySeconds = 6.0f, TrophySlide = 0.4f;

// Present rate for the FPS counter.
std::chrono::steady_clock::time_point last_present{};
float frame_ms_avg = 0.0f;

float PixelDensity(SDL_WindowID id);

/// The size the menu is drawn at (the swapchain's, Draw), for the mouse's position in it.
std::atomic<u32> display_width{0}, display_height{0};

/// A mouse position in the window to the menu's pixels. The swapchain may be another size than the
/// window, which shows it scaled: on Windows a window of 1920x1080 on a 2560x1080 monitor kept the
/// monitor's size, the mouse's x was a quarter short of the drawn menu and clicks missed the
/// checkboxes while rows still lit up (2026-10-27). Without a size yet: the pixel density.
ImVec2 ToDisplay(SDL_WindowID id, float x, float y) {
    SDL_Window* window = SDL_GetWindowFromID(id);
    int w = 0, h = 0;
    const u32 dw = display_width.load(std::memory_order_relaxed), dh = display_height.load(std::memory_order_relaxed);
    if (window && dw && dh && SDL_GetWindowSize(window, &w, &h) && w > 0 && h > 0) {
        return {x * float(dw) / float(w), y * float(dh) / float(h)};
    }
    const float density = PixelDensity(id);
    return {x * density, y * density};
}

void SetOpen(bool value) {
    if (menu_open.exchange(value) == value) {
        return;
    }
    // The system cursor shows over the menu (window.cpp); ImGui learns where it is now, not at
    // the next motion: mouse motion is not passed on while the menu is closed.
    if (value) {
        if (SDL_Window* window = SDL_GetMouseFocus()) {
            float x = 0.0f, y = 0.0f;
            SDL_GetMouseState(&x, &y);
            const ImVec2 at = ToDisplay(SDL_GetWindowID(window), x, y);
            ImGui::GetIO().AddMousePosEvent(at.x, at.y);
        }
    }
    if (!value && dirty) {
        dirty = false;
        BbSettings::Save();
    }
}

ImGuiKey KeyFromSdl(SDL_Keycode key) {
    switch (key) {
    case SDLK_TAB: return ImGuiKey_Tab;
    case SDLK_LEFT: return ImGuiKey_LeftArrow;
    case SDLK_RIGHT: return ImGuiKey_RightArrow;
    case SDLK_UP: return ImGuiKey_UpArrow;
    case SDLK_DOWN: return ImGuiKey_DownArrow;
    case SDLK_PAGEUP: return ImGuiKey_PageUp;
    case SDLK_PAGEDOWN: return ImGuiKey_PageDown;
    case SDLK_HOME: return ImGuiKey_Home;
    case SDLK_END: return ImGuiKey_End;
    case SDLK_DELETE: return ImGuiKey_Delete;
    case SDLK_BACKSPACE: return ImGuiKey_Backspace;
    case SDLK_SPACE: return ImGuiKey_Space;
    case SDLK_RETURN: return ImGuiKey_Enter;
    case SDLK_KP_ENTER: return ImGuiKey_KeypadEnter;
    case SDLK_ESCAPE: return ImGuiKey_Escape;
    case SDLK_LCTRL: return ImGuiKey_LeftCtrl;
    case SDLK_RCTRL: return ImGuiKey_RightCtrl;
    case SDLK_LSHIFT: return ImGuiKey_LeftShift;
    case SDLK_RSHIFT: return ImGuiKey_RightShift;
    case SDLK_LALT: return ImGuiKey_LeftAlt;
    case SDLK_RALT: return ImGuiKey_RightAlt;
    default: return ImGuiKey_None;
    }
}

ImGuiKey KeyFromGamepad(u8 button) {
    switch (button) {
    case SDL_GAMEPAD_BUTTON_SOUTH: return ImGuiKey_GamepadFaceDown;
    case SDL_GAMEPAD_BUTTON_EAST: return ImGuiKey_GamepadFaceRight;
    case SDL_GAMEPAD_BUTTON_WEST: return ImGuiKey_GamepadFaceLeft;
    case SDL_GAMEPAD_BUTTON_NORTH: return ImGuiKey_GamepadFaceUp;
    case SDL_GAMEPAD_BUTTON_DPAD_UP: return ImGuiKey_GamepadDpadUp;
    case SDL_GAMEPAD_BUTTON_DPAD_DOWN: return ImGuiKey_GamepadDpadDown;
    case SDL_GAMEPAD_BUTTON_DPAD_LEFT: return ImGuiKey_GamepadDpadLeft;
    case SDL_GAMEPAD_BUTTON_DPAD_RIGHT: return ImGuiKey_GamepadDpadRight;
    case SDL_GAMEPAD_BUTTON_LEFT_SHOULDER: return ImGuiKey_GamepadL1;
    case SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER: return ImGuiKey_GamepadR1;
    case SDL_GAMEPAD_BUTTON_START: return ImGuiKey_GamepadStart;
    case SDL_GAMEPAD_BUTTON_BACK: return ImGuiKey_GamepadBack;
    default: return ImGuiKey_None;
    }
}

float PixelDensity(SDL_WindowID id) {
    SDL_Window* window = SDL_GetWindowFromID(id);
    const float density = window ? SDL_GetWindowPixelDensity(window) : 1.0f;
    return density > 0.0f ? density : 1.0f;
}

/// The FPS readout: the frames shown and their interval; with frame generation the multiplier
/// in between ("240 FPS  FG 4x  4.2 ms"), from the generated frames shown per game frame.
std::string FpsText() {
    const auto& s = BbSettings::Get();
    const float shown = frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f;
    const int generated = s.frame_generation.load(std::memory_order_relaxed) != BbSettings::FrameGenOff
                              ? s.frame_gen_now.load(std::memory_order_relaxed)
                              : 0;
    char text[192];
    if (generated > 0) {
        std::snprintf(text, sizeof(text),
                      BbSettings::MenuText("%.0f FPS  FG %dx  %.1f ms", "%.0f FPS  FG %dx  %.1f мс"),
                      shown, generated + 1, frame_ms_avg);
    } else {
        std::snprintf(text, sizeof(text), BbSettings::MenuText("%.0f FPS  %.1f ms", "%.0f FPS  %.1f мс"),
                      shown, frame_ms_avg);
    }
    // Other programs took video memory from the game (BufferCache::VramPromotionsPaused).
    if (const u32 lost = BbStats::vram_budget_lost_mb.load(std::memory_order_relaxed); lost >= 150) {
        const size_t used = std::strlen(text);
        std::snprintf(text + used, sizeof(text) - used,
                      BbSettings::MenuText("  VRAM -%u MB (other apps)", "  VRAM −%u МБ (другие программы)"),
                      lost);
    }
    return text;
}

} // namespace

namespace detail {

/// BB_OVERLAY_TRACE=1 (diagnostics): the menu's mouse buttons as SDL gives them and as ImGui
/// takes them (Windows: items highlight but clicks do not change them).
bool OverlayTrace() {
    static const bool on = bb_env_on("BB_OVERLAY_TRACE");
    return on;
}

void Checkbox(const char* label, std::atomic<bool>& value) {
    bool v = value;
    const bool changed = ImGui::Checkbox(label, &v);
    if (changed && OverlayTrace()) {
        std::printf("Overlay trace: checkbox \"%s\" -> %d (id %#x)\n", label, int(v), unsigned(ImGui::GetItemID()));
    }
    Store(value, v, changed);
}

void Slider(const char* label, std::atomic<float>& value, float lo, float hi) {
    float v = value;
    Store(value, v, ImGui::SliderFloat(label, &v, lo, hi, "%.2f"));
}

void Hint(const char* text) {
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::BeginItemTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 30.0f);
        ImGui::TextUnformatted(text);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

} // namespace detail

namespace {

// The settings menu as in 0.3 and 0.4: one window (moved with the mouse, its place kept in
// bbport.ini), sections one under the other, ImGui's own widgets and keyboard/gamepad navigation.
void Menu() {
    auto& s = BbSettings::Get();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    // Where it was moved last (bbport.ini menu_pos, a fraction of the screen), kept on screen.
    ImVec2 pos(viewport->WorkPos.x + 40.0f * base_scale, viewport->WorkPos.y + 40.0f * base_scale);
    if (s.menu_x >= 0.0f && s.menu_y >= 0.0f) {
        const float margin = 80.0f * base_scale;
        pos.x = viewport->WorkPos.x +
                std::clamp(s.menu_x * viewport->WorkSize.x, 0.0f, std::max(viewport->WorkSize.x - margin, 0.0f));
        pos.y = viewport->WorkPos.y +
                std::clamp(s.menu_y * viewport->WorkSize.y, 0.0f, std::max(viewport->WorkSize.y - margin, 0.0f));
    }
    ImGui::SetNextWindowPos(pos, ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(620.0f * base_scale, 0.0f), ImGuiCond_Appearing);
    bool keep_open = true;
#ifdef __APPLE__ // bbport: Mac keyboards have no Insert key (HandleEvent)
    const char* menu_title =
        BbSettings::MenuText("Bloodborne - Graphics  (F1 / ` / hold L3+R3)###bbport_settings",
                             "Bloodborne — настройки  (F1 / ` / удерживать L3+R3)###bbport_settings");
#else
    const char* menu_title =
        BbSettings::MenuText("Bloodborne - Graphics  (Insert / hold L3+R3)###bbport_settings",
                             "Bloodborne — настройки  (Insert / удерживать L3+R3)###bbport_settings");
#endif
    if (!ImGui::Begin(menu_title, &keep_open, ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }
    // Moved: remembered (saved with the settings when the menu closes).
    if (!ImGui::IsWindowAppearing() && viewport->WorkSize.x > 0.0f && viewport->WorkSize.y > 0.0f) {
        const ImVec2 at = ImGui::GetWindowPos();
        const float fx = (at.x - viewport->WorkPos.x) / viewport->WorkSize.x;
        const float fy = (at.y - viewport->WorkPos.y) / viewport->WorkSize.y;
        if (std::abs(at.x - pos.x) >= 1.0f || std::abs(at.y - pos.y) >= 1.0f) {
            s.menu_x = fx;
            s.menu_y = fy;
            dirty = true;
        }
    }
    // Always readable, even when the rest of the menu is in Russian.
    static const char* languages[] = {"English", "Русский"};
    int language = s.menu_language == BbSettings::MenuLanguage::Russian ? 1 : 0;
    if (ImGui::Combo("Language", &language, languages, 2)) {
        s.menu_language =
            language == 1 ? BbSettings::MenuLanguage::Russian : BbSettings::MenuLanguage::English;
        s.menu_language_chosen = true;
        BbSettings::Save();
    }
    ImGui::TextUnformatted(FpsText().c_str());

    const bool taa = UpscalerSection(s);
    ReactiveMaskSection(s, taa);
    OutputSection(s);
    GameEffectsSection(s);
    FrameRateLimitSection(s);
    ImGui::SeparatorText(BbSettings::MenuText("Camera (right stick)", "Камера (правый стик)"));
    Slider(BbSettings::MenuText("Horizontal speed", "Скорость по горизонтали"), s.camera_speed[1], 0.25f, 2.0f);
    Slider(BbSettings::MenuText("Vertical speed", "Скорость по вертикали"), s.camera_speed[0], 0.25f, 2.0f);

    ImGui::SeparatorText(BbSettings::MenuText("Other", "Прочее"));
    Checkbox(BbSettings::MenuText("FPS counter in corner", "Счётчик FPS в углу"), s.show_fps);
    ShadersSection();

    ImGui::Spacing();
    if (ImGui::Button(BbSettings::MenuText("Close", "Закрыть"))) {
        keep_open = false;
    }
    // Issues #74, #116: leaving the game from the pad (this menu opens with L3+R3). A second
    // press confirms; the window thread then closes like the window's own close button.
    static bool quit_armed;
    ImGui::SameLine();
    if (ImGui::Button(quit_armed ? BbSettings::MenuText("Press again to quit", "Нажмите ещё раз для выхода")
                                 : BbSettings::MenuText("Quit game", "Выйти из игры"))) {
        if (quit_armed) {
            SDL_Event quit{};
            quit.type = SDL_EVENT_QUIT;
            SDL_PushEvent(&quit);
        }
        quit_armed = !quit_armed;
    }
    if (quit_armed && !ImGui::IsItemFocused() && !ImGui::IsItemHovered()) {
        quit_armed = false;
    }
    ImGui::TextDisabled("%s", BbSettings::MenuText("Settings are saved to bbport.ini",
                                             "Настройки сохраняются в bbport.ini"));
    ImGui::End();
    if (!keep_open) {
        SetOpen(false);
    }
}

void FpsCounter() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float pad = 12.0f * base_scale;
    ImGui::SetNextWindowPos(
        ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - pad, viewport->WorkPos.y + pad),
        ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("##fps", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    const auto& s = BbSettings::Get();
    ImGui::Text("%s  %s", FpsText().c_str(),
                s.upscaler == BbSettings::UpscalerFsr3     ? "FSR 3.1"
                : s.upscaler == BbSettings::UpscalerFsr4   ? "FSR 4"
                : s.upscaler == BbSettings::UpscalerFsr411 ? "FSR 4.1.1"
                : s.upscaler == BbSettings::UpscalerTaa    ? "TAA"
                : s.upscaler == BbSettings::UpscalerDlss   ? "DLSS"
                : s.upscaler == BbSettings::UpscalerFsr1   ? "FSR 1"
                : s.upscaler == BbSettings::UpscalerFsrAmd ? "FSR 4 (AMD)"
                : s.upscaler == BbSettings::UpscalerMetalFx ? "MetalFX"
                                                           : "");
    ImGui::End();
}

/// "Compiling shaders": pipelines made in the background (the cache's preload, imported recipes)
/// as done / queued, the recipe import, and the game's own compiles of the last 1.5 s (idea from
/// the SrShadowy fork's precompile splash; here a line in the corner while it lasts).
/// BB_SHADER_PROGRESS=0 hides it.
struct ShaderProgress {
    u32 done = 0, total = 0, recent = 0;
    bool importing = false;
};
ShaderProgress CurrentShaderProgress() {
    static const bool enabled = !bb_env_off("BB_SHADER_PROGRESS");
    ShaderProgress p;
    if (!enabled) {
        return p;
    }
    p.total = g_bb_background_pipelines_total.load(std::memory_order_relaxed);
    p.done = g_bb_background_pipelines_done.load(std::memory_order_relaxed);
    // An import longer than 3 minutes is not shown (a stuck child must not keep the line up).
    static const long long start = NowNs();
    p.importing = g_bb_shader_import_active.load(std::memory_order_relaxed) &&
                  NowNs() - start < 180'000'000'000ll;
    static u32 last_compiles = 0;
    static long long last_change = 0;
    static u32 window_start_compiles = 0;
    const u32 compiles = g_bb_compiles_total.load(std::memory_order_relaxed);
    const long long now = NowNs();
    if (compiles != last_compiles) {
        if (now - last_change > 1'500'000'000ll) {
            window_start_compiles = last_compiles;
        }
        last_compiles = compiles;
        last_change = now;
    }
    if (last_change != 0 && now - last_change <= 1'500'000'000ll) {
        p.recent = compiles - window_start_compiles;
    }
    return p;
}
bool ShaderProgressShown() {
    const ShaderProgress p = CurrentShaderProgress();
    const bool shown = p.done < p.total || p.importing || p.recent != 0;
    static bool was_shown = false;
    if (shown != was_shown) {
        was_shown = shown;
        std::printf("Overlay: \"Compiling shaders\" %s (%u of %u in the background%s, %u just now)\n",
                    shown ? "shown" : "hidden", p.done, p.total,
                    p.importing ? ", recipes importing" : "", p.recent);
    }
    return shown;
}
void ShaderProgressLine() {
    const ShaderProgress p = CurrentShaderProgress();
    std::string text;
    if (p.done < p.total) {
        text = fmt::format("{} {} / {}",
                           BbSettings::MenuText("Compiling shaders", "Компиляция шейдеров"), p.done,
                           p.total);
    } else if (p.importing) {
        text = BbSettings::MenuText("Preparing shaders...", "Подготовка шейдеров...");
    } else if (p.recent != 0) {
        text = fmt::format("{} ({})",
                           BbSettings::MenuText("Compiling shaders", "Компиляция шейдеров"),
                           p.recent);
    } else {
        return;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float pad = 12.0f * base_scale;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + pad,
                                   viewport->WorkPos.y + viewport->WorkSize.y - pad),
                            ImGuiCond_Always, ImVec2(0.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("##shader_progress", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::TextUnformatted(text.c_str());
    ImGui::End();
}

// A trophy icon as an ImGui texture (present thread, under imgui_mutex); null when it cannot be
// read or SDL is older than 3.4 (no PNG loader).
ImTextureData* TrophyIcon(const std::filesystem::path& path) {
    if (path.empty()) {
        return nullptr;
    }
    const auto [it, added] = trophy_icons.try_emplace(path.string(), nullptr);
    if (!added) {
        return it->second;
    }
#if SDL_VERSION_ATLEAST(3, 4, 0)
    // UTF-8 for SDL (on Windows path::c_str() is wide, string() the ANSI code page).
    const std::u8string utf8 = path.u8string();
    const char* name = reinterpret_cast<const char*>(utf8.c_str());
    SDL_Surface* loaded = SDL_LoadPNG(name);
    SDL_Surface* rgba = loaded ? SDL_ConvertSurface(loaded, SDL_PIXELFORMAT_RGBA32) : nullptr;
    SDL_DestroySurface(loaded);
    if (!rgba) {
        std::printf("Overlay: trophy icon %s not loaded: %s\n", name, SDL_GetError());
        return nullptr;
    }
    auto* tex = IM_NEW(ImTextureData)();
    tex->Create(ImTextureFormat_RGBA32, rgba->w, rgba->h);
    for (int y = 0; y < rgba->h; ++y) {
        std::memcpy(tex->GetPixelsAt(0, y), static_cast<const unsigned char*>(rgba->pixels) + y * rgba->pitch,
                    std::size_t(rgba->w) * 4);
    }
    SDL_DestroySurface(rgba);
    ImGui::RegisterUserTexture(tex);
    it->second = tex;
#endif
    return it->second;
}

// The front trophy of the queue: a dark panel at the top centre (the game's HUD has its health at
// the top left, its echoes and insight at the top right), sliding down, with a badge in the
// grade's colour, "Trophy earned" and the name.
void TrophyWindow(float dt) {
    TrophyToast toast;
    {
        std::scoped_lock lock{trophy_mutex};
        if (trophy_queue.empty()) {
            trophy_active = false;
            return;
        }
        auto& front = trophy_queue.front();
        if (front.elapsed <= 0.0f && front.elapsed + dt > 0.0f) {
            std::printf("Overlay: trophy shown: %s\n", front.name.c_str());
        }
        front.elapsed += dt;
        if (front.elapsed > TrophySeconds) {
            trophy_queue.pop_front();
            trophy_active = !trophy_queue.empty();
            return;
        }
        toast = front;
    }
    if (toast.elapsed < 0.0f) {
        return; // a delayed one (BB_TROPHY_TEST)
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float s = base_scale;
    const ImVec2 size{380.0f * s, 76.0f * s};
    const float slide = std::min(toast.elapsed / TrophySlide, 1.0f);
    const float eased = 1.0f - (1.0f - slide) * (1.0f - slide);
    const float x = viewport->WorkPos.x + (viewport->WorkSize.x - size.x) * 0.5f;
    const float y = viewport->WorkPos.y - size.y + (size.y + 24.0f * s) * eased;
    const float alpha = std::clamp(std::min(slide, TrophySeconds - toast.elapsed), 0.0f, 1.0f);
    auto* draw = ImGui::GetForegroundDrawList();
    const auto a = [alpha](int r, int g, int b, int base) { return IM_COL32(r, g, b, int(base * alpha)); };
    draw->AddRectFilled({x, y}, {x + size.x, y + size.y}, a(12, 10, 8, 220), 8.0f * s);
    draw->AddRect({x, y}, {x + size.x, y + size.y}, a(150, 130, 100, 200), 8.0f * s, 0, 1.5f * s);
    // The badge: the grade's colour (platinum, gold, silver, bronze; grey unknown) and its letter.
    const ImU32 badge = toast.grade == 'P'   ? a(200, 215, 235, 255)
                        : toast.grade == 'G' ? a(222, 180, 60, 255)
                        : toast.grade == 'S' ? a(185, 190, 196, 255)
                        : toast.grade == 'B' ? a(176, 112, 64, 255)
                                             : a(120, 120, 120, 255);
    const ImVec2 centre{x + 38.0f * s, y + size.y * 0.5f};
    const char letter[2] = {toast.grade == '?' ? 'T' : toast.grade, 0};
    const ImVec2 letter_size = ImGui::CalcTextSize(letter);
    if (ImTextureData* icon = TrophyIcon(toast.icon)) {
        // The game's icon, the grade as a small badge on its corner.
        const float half = 26.0f * s;
        draw->AddImageRounded(icon->GetTexRef(), {centre.x - half, centre.y - half}, {centre.x + half, centre.y + half},
                              {0, 0}, {1, 1}, a(255, 255, 255, 255), 4.0f * s);
        const ImVec2 dot{centre.x + half - 2.0f * s, centre.y + half - 2.0f * s};
        draw->AddCircleFilled(dot, 9.0f * s, badge);
        draw->AddCircle(dot, 9.0f * s, a(30, 25, 20, 255), 0, 1.5f * s);
    } else {
        draw->AddCircleFilled(centre, 24.0f * s, badge);
        draw->AddCircle(centre, 24.0f * s, a(30, 25, 20, 255), 0, 2.0f * s);
        draw->AddText({centre.x - letter_size.x * 0.5f, centre.y - letter_size.y * 0.5f}, a(30, 25, 20, 255), letter);
    }
    const float text_x = x + 76.0f * s, wrap = size.x - 88.0f * s;
    // In the game's language, as the name (BB_LANGUAGE, the PS4 number: 8 Russian).
    static const bool russian = [] {
        const char* env = bb_env("BB_LANGUAGE");
        return env && std::atoi(env) == 8;
    }();
    draw->AddText({text_x, y + 12.0f * s}, a(200, 185, 150, 255), russian ? "Получен трофей" : "Trophy earned");
    draw->AddText(nullptr, 0.0f, {text_x, y + 12.0f * s + ImGui::GetTextLineHeightWithSpacing()},
                  a(240, 232, 215, 255), toast.name.c_str(), nullptr, wrap);
}

void TextPrompt() {
    std::string title, text;
    {
        std::scoped_lock lock{prompt_mutex};
        title = prompt_title;
        text = prompt_text;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x * 0.5f,
                                   viewport->WorkPos.y + viewport->WorkSize.y * 0.5f),
                            ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowBgAlpha(0.9f);
    ImGui::Begin("##textprompt", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    ImGui::TextUnformatted(title.c_str());
    ImGui::Separator();
    ImGui::Text("%s_", text.c_str());
    ImGui::Separator();
    ImGui::TextUnformatted("Keyboard: type, Backspace = delete, Enter = OK, Esc = cancel");
    ImGui::End();
}

/// BB_MENU_KEYS_FILE=<file> (scripted tests): tokens toggle up down left right enter back l1 r1,
/// consumed when the file appears (it is removed), one key press per frame.
void ScriptedKeys() {
    static const char* path = bb_env("BB_MENU_KEYS_FILE");
    static std::chrono::steady_clock::time_point last_check{};
    static std::vector<std::string> queue;
    static bool release = false, mouse_release = false;
    static ImGuiKey held = ImGuiKey_None;
    if (!path) {
        return;
    }
    ImGuiIO& io = ImGui::GetIO();
    if (release) {
        io.AddKeyEvent(held, false);
        release = false;
        return;
    }
    if (mouse_release) {
        io.AddMouseButtonEvent(0, false);
        mouse_release = false;
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (queue.empty() && now - last_check > std::chrono::milliseconds(100)) {
        last_check = now;
        if (FILE* f = std::fopen(path, "r")) {
            char token[32];
            while (std::fscanf(f, "%31s", token) == 1) {
                queue.emplace_back(token);
            }
            std::fclose(f);
            std::remove(path);
        }
    }
    if (queue.empty()) {
        return;
    }
    const std::string token = queue.front();
    queue.erase(queue.begin());
    if (token == "toggle") {
        SetOpen(!menu_open);
        return;
    }
    // "click X Y": the left button pressed at display pixel X, Y and released the frame after.
    if (token == "click" && queue.size() >= 2) {
        const float x = std::strtof(queue[0].c_str(), nullptr), y = std::strtof(queue[1].c_str(), nullptr);
        queue.erase(queue.begin(), queue.begin() + 2);
        io.AddMousePosEvent(x, y);
        io.AddMouseButtonEvent(0, true);
        mouse_release = true;
        return;
    }
    held = token == "up" ? ImGuiKey_GamepadDpadUp : token == "down" ? ImGuiKey_GamepadDpadDown
         : token == "left" ? ImGuiKey_GamepadDpadLeft : token == "right" ? ImGuiKey_GamepadDpadRight
         : token == "enter" ? ImGuiKey_GamepadFaceDown : token == "l1" ? ImGuiKey_GamepadL1
         : token == "r1" ? ImGuiKey_GamepadR1 : token == "kdown" ? ImGuiKey_DownArrow
         : token == "kup" ? ImGuiKey_UpArrow : ImGuiKey_None;
    if (token == "back") {
        SetOpen(false);
        return;
    }
    if (held != ImGuiKey_None) {
        io.AddKeyEvent(held, true);
        release = true;
    }
}

} // namespace

void SetTextPrompt(bool active, const std::string& prompt, const std::string& text) {
    {
        std::scoped_lock lock{prompt_mutex};
        prompt_title = prompt;
        prompt_text = text;
    }
    prompt_active = active;
}

void Init(const Vulkan::Instance& instance, vk::Format format, u32 image_count) {
    std::scoped_lock lock{imgui_mutex};
    if (initialized) {
        return;
    }
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr; // window positions are not kept
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_NavEnableGamepad;
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    io.BackendPlatformName = "bbport";

    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;
    style.Colors[ImGuiCol_WindowBg].w = 0.92f;

    ImFontConfig font_config;
    font_config.FontDataOwnedByAtlas = false;
    io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(bb_font_ttf),
                                   int(bb_font_ttf_end - bb_font_ttf), 18.0f, &font_config);

    const vk::Instance vk_instance = instance.GetInstance();
    ImGui_ImplVulkan_LoadFunctions(
        instance.ApiVersion(),
        [](const char* name, void* user) {
            return VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr(
                *static_cast<const vk::Instance*>(user), name);
        },
        const_cast<vk::Instance*>(&vk_instance));

    const VkFormat color_format = static_cast<VkFormat>(format);
    ImGui_ImplVulkan_InitInfo info{};
    info.ApiVersion = instance.ApiVersion();
    info.Instance = vk_instance;
    info.PhysicalDevice = instance.GetPhysicalDevice();
    info.Device = instance.GetDevice();
    info.QueueFamily = instance.GetGraphicsQueueFamilyIndex();
    info.Queue = instance.GetGraphicsQueue();
    info.DescriptorPoolSize = 16;
    info.MinImageCount = std::max(image_count, 2u);
    info.ImageCount = std::max(image_count, 2u);
    info.UseDynamicRendering = true;
    info.PipelineInfoMain.PipelineRenderingCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &color_format,
    };
    if (!ImGui_ImplVulkan_Init(&info)) {
        std::printf("Overlay: ImGui Vulkan backend init failed\n");
        ImGui::DestroyContext();
        return;
    }
    initialized = true;
#ifdef __APPLE__
    std::printf("Overlay: menu ready (F1, ` or §, or hold L3+R3)\n");
#else
    std::printf("Overlay: menu ready (Insert, or hold L3+R3)\n");
#endif
}

void UpdateTextInput(SDL_Window* window) {
    bool want = false;
    {
        std::scoped_lock lock{imgui_mutex};
        want = initialized && menu_open && ImGui::GetIO().WantTextInput;
    }
    if (want != SDL_TextInputActive(window)) {
        if (want) {
            SDL_StartTextInput(window);
        } else {
            SDL_StopTextInput(window);
        }
    }
}

bool HandleEvent(const SDL_Event& event) {
    std::scoped_lock lock{imgui_mutex};
    if (!initialized) {
        return false;
    }
    ImGuiIO& io = ImGui::GetIO();
    const bool is_open = menu_open;
    switch (event.type) {
    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
        const bool down = event.type == SDL_EVENT_KEY_DOWN;
        // bbport: Mac keyboards have no Insert key: F1 (fn+F1 unless F-keys are standard), ` or §.
#ifdef __APPLE__
        const bool menu_key = event.key.key == SDLK_INSERT || event.key.key == SDLK_F1 ||
                              event.key.key == 0x60 /* ` */ || event.key.key == 0xA7 /* § */;
#else
        const bool menu_key = event.key.key == SDLK_INSERT;
#endif
        if (down && !event.key.repeat && (menu_key || (is_open && event.key.key == SDLK_ESCAPE))) {
            SetOpen(menu_key ? !is_open : false);
            return true;
        }
        if (!is_open) {
            return false;
        }
        io.AddKeyEvent(ImGuiMod_Ctrl, (event.key.mod & SDL_KMOD_CTRL) != 0);
        io.AddKeyEvent(ImGuiMod_Shift, (event.key.mod & SDL_KMOD_SHIFT) != 0);
        io.AddKeyEvent(ImGuiMod_Alt, (event.key.mod & SDL_KMOD_ALT) != 0);
#ifdef __APPLE__
        // ImGui swaps Ctrl and Cmd on macOS: Cmd is its Ctrl (Cmd+click types a slider value).
        io.AddKeyEvent(ImGuiMod_Super, (event.key.mod & SDL_KMOD_GUI) != 0);
#endif
        if (const ImGuiKey key = KeyFromSdl(event.key.key); key != ImGuiKey_None) {
            io.AddKeyEvent(key, down);
        }
        return true;
    }
    case SDL_EVENT_GAMEPAD_BUTTON_DOWN:
    case SDL_EVENT_GAMEPAD_BUTTON_UP: {
        const bool down = event.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN;
        const u8 button = event.gbutton.button;
        if (button == SDL_GAMEPAD_BUTTON_LEFT_STICK) {
            l3_down = down;
        } else if (button == SDL_GAMEPAD_BUTTON_RIGHT_STICK) {
            r3_down = down;
        }
        if (l3_down && r3_down) {
            if (down && is_open) {
                SetOpen(false);
                return true;
            }
            if (down && l3r3_since.load() == 0) {
                l3r3_since = NowNs(); // the game sees the press too (a fight goes on)
            }
        } else {
            l3r3_since = 0;
        }
        if (!is_open) {
            return false;
        }
        if (const ImGuiKey key = KeyFromGamepad(button); key != ImGuiKey_None) {
            io.AddKeyEvent(key, down);
        }
        return true;
    }
    case SDL_EVENT_TEXT_INPUT: {
        // Typed characters (Ctrl+click on a slider, a text field): key events alone erase but
        // do not type. SDL sends them while text input is on (UpdateTextInput).
        if (!is_open) {
            return false;
        }
        io.AddInputCharactersUTF8(event.text.text);
        return true;
    }
    case SDL_EVENT_MOUSE_MOTION: {
        if (!is_open) {
            return false;
        }
        const ImVec2 at = ToDisplay(event.motion.windowID, event.motion.x, event.motion.y);
        io.AddMousePosEvent(at.x, at.y);
        return true;
    }
    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
        if (!is_open) {
            return false;
        }
        const int button = event.button.button == SDL_BUTTON_LEFT    ? 0
                           : event.button.button == SDL_BUTTON_RIGHT  ? 1
                           : event.button.button == SDL_BUTTON_MIDDLE ? 2
                                                                      : -1;
        if (button >= 0) {
            io.AddMouseButtonEvent(button, event.type == SDL_EVENT_MOUSE_BUTTON_DOWN);
        }
        if (OverlayTrace()) {
            std::printf("Overlay trace: SDL button %d %s at %.0f,%.0f (mouse id %u, clicks %d)\n",
                        int(event.button.button), event.type == SDL_EVENT_MOUSE_BUTTON_DOWN ? "down" : "up",
                        event.button.x, event.button.y, unsigned(event.button.which), int(event.button.clicks));
        }
        return true;
    }
    case SDL_EVENT_MOUSE_WHEEL:
        if (!is_open) {
            return false;
        }
        io.AddMouseWheelEvent(event.wheel.x, event.wheel.y);
        return true;
    default:
        return false;
    }
}

void SetBanner(const char* text) {
    {
        std::scoped_lock lock{banner_mutex};
        banner_text = text ? text : "";
    }
    banner_active = text != nullptr && text[0] != 0;
}

void ShowTrophy(char grade, const std::string& name, const std::filesystem::path& icon, float delay) {
    std::scoped_lock lock{trophy_mutex};
    trophy_queue.push_back({grade, name, -delay, icon});
    trophy_active = true;
}

bool Visible() {
    // BB_OVERLAY_OPEN=1 (tests): the menu open from the start, its sections unfolded.
    static bool open_at_start = bb_env_on("BB_OVERLAY_OPEN");
    if (open_at_start && initialized) {
        open_at_start = false;
        open_requested = true;
    }
    if (const long long since = l3r3_since.load();
        since != 0 && !menu_open && NowNs() - since >= OpenHoldMs * 1'000'000) {
        l3r3_since = 0;
        open_requested = true;
    }
    const bool shader_progress = ShaderProgressShown();
    return initialized && (menu_open || open_requested || prompt_active || banner_active ||
                           trophy_active || BbSettings::Get().show_fps || shader_progress);
}

bool MenuOpen() {
    return menu_open;
}

bool CapturesInput() {
    // The text dialog too: keys typed into it (Backspace is the touchpad) stay out of the game.
    return menu_open || prompt_active;
}

void Render(vk::CommandBuffer cmdbuf, vk::ImageView view, vk::Extent2D extent) {

    // Present interval for the FPS readout (measured also while nothing is drawn).
    const auto now = std::chrono::steady_clock::now();
    const float ms = std::chrono::duration<float, std::milli>(now - last_present).count();
    last_present = now;
    if (ms > 0.0f && ms < 1000.0f) {
        frame_ms_avg = frame_ms_avg == 0.0f ? ms : frame_ms_avg * 0.95f + ms * 0.05f;
    }
    // BB_FPS_PRINT=1: the present rate every 5 s on stdout, without the cost of BB_FRAME_STATS
    // (its timers are in the draw path): to measure as players run the game.
    static const bool fps_print = bb_env_on("BB_FPS_PRINT");
    if (fps_print) {
        static auto window_start = now;
        static std::uint64_t presents = 0;
        ++presents;
        if (const double s = std::chrono::duration<double>(now - window_start).count(); s >= 5.0) {
            std::printf("Present rate: %.1f FPS over %.1f s\n", presents / s, s);
            std::fflush(stdout);
            window_start = now;
            presents = 0;
        }
    }
    if (bb_env("BB_MENU_KEYS_FILE") && initialized) {
        std::scoped_lock lock{imgui_mutex};
        ScriptedKeys();
    }
    if (!Visible()) {
        return;
    }
    std::scoped_lock lock{imgui_mutex};
    if (open_requested.exchange(false)) {
        SetOpen(true);
    }
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize =ImVec2(float(extent.width), float(extent.height));
    display_width.store(extent.width, std::memory_order_relaxed);
    display_height.store(extent.height, std::memory_order_relaxed);
    io.DeltaTime = ms > 0.0f && ms < 1000.0f ? ms / 1000.0f : 1.0f / 60.0f;
    // UI scale follows the display height (1080p = 1).
    const float scale = std::max(float(extent.height) / 1080.0f, 0.75f);
    if (std::abs(scale - base_scale) > 0.01f) {
        ImGuiStyle& style = ImGui::GetStyle();
        style.ScaleAllSizes(scale / base_scale);
        style.FontScaleMain = scale;
        base_scale = scale;
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();
    if (OverlayTrace() && menu_open && (io.MouseClicked[0] || io.MouseReleased[0])) {
        const ImGuiContext& g = *ImGui::GetCurrentContext();
        std::printf("Overlay trace: ImGui left %s at %.0f,%.0f, display %.0fx%.0f, hovered id %#x, active id %#x\n",
                    io.MouseClicked[0] ? "clicked" : "released", io.MousePos.x, io.MousePos.y,
                    io.DisplaySize.x, io.DisplaySize.y, unsigned(g.HoveredIdPreviousFrame), unsigned(g.ActiveId));
    }
    if (menu_open) {
        Menu();
    }
    if (BbSettings::Get().show_fps && !menu_open) {
        FpsCounter();
    }
    if (prompt_active && !menu_open) {
        TextPrompt();
    }
    if (!menu_open) {
        ShaderProgressLine();
    }
    if (trophy_active) {
        TrophyWindow(io.DeltaTime);
    }
    if (banner_active) {
        std::string text;
        {
            std::scoped_lock banner_lock{banner_mutex};
            text = banner_text;
        }
        const ImVec2 size = ImGui::CalcTextSize(text.c_str());
        auto* draw = ImGui::GetForegroundDrawList();
        const ImVec2 at{(io.DisplaySize.x - size.x) * 0.5f, io.DisplaySize.y * 0.82f};
        draw->AddRectFilled({at.x - 24.0f, at.y - 12.0f}, {at.x + size.x + 24.0f, at.y + size.y + 12.0f},
                            IM_COL32(0, 0, 0, 170), 6.0f);
        draw->AddText(at, IM_COL32(230, 220, 200, 255), text.c_str());
    }
    ImGui::Render();

    const vk::RenderingAttachmentInfo attachment{
        .imageView = view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
    };
    cmdbuf.beginRendering(vk::RenderingInfo{
        .renderArea = {{0, 0}, extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &attachment,
    });
    {
        // Font atlas uploads submit to the graphics queue themselves.
        std::scoped_lock submit_lock{Vulkan::Scheduler::submit_mutex};
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdbuf);
    }
    cmdbuf.endRendering();
}

} // namespace BbOverlay

extern "C" void bbgpu_overlay_banner(const char* text) {
    BbOverlay::SetBanner(text);
}
