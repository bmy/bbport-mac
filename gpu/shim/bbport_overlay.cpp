// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_overlay.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include <SDL3/SDL.h>
#include "bbport_settings.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_vulkan.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

// DejaVu Sans (Cyrillic), embedded (third_party/fonts, Bitstream Vera license).
#ifdef __APPLE__
// Mach-O: read-only data section, C symbols carry a leading underscore, .private_extern hides.
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
asm(".section .rodata\n"
    ".balign 16\n"
    ".hidden bb_font_ttf\n"
    ".global bb_font_ttf\n"
    "bb_font_ttf:\n"
    ".incbin \"" BB_FONT_PATH "\"\n"
    ".hidden bb_font_ttf_end\n"
    ".global bb_font_ttf_end\n"
    "bb_font_ttf_end:\n"
    ".previous\n");
#endif
extern "C" const unsigned char bb_font_ttf[];
extern "C" const unsigned char bb_font_ttf_end[];
// DejaVu Serif (same license): the settings menu, styled like the game's own menus.
#ifdef __APPLE__
asm(".section __TEXT,__const\n"
    ".p2align 4\n"
    ".private_extern _bb_serif_ttf\n"
    ".globl _bb_serif_ttf\n"
    "_bb_serif_ttf:\n"
    ".incbin \"" BB_SERIF_FONT_PATH "\"\n"
    ".private_extern _bb_serif_ttf_end\n"
    ".globl _bb_serif_ttf_end\n"
    "_bb_serif_ttf_end:\n"
    ".text\n");
#else
asm(".section .rodata\n"
    ".balign 16\n"
    ".hidden bb_serif_ttf\n"
    ".global bb_serif_ttf\n"
    "bb_serif_ttf:\n"
    ".incbin \"" BB_SERIF_FONT_PATH "\"\n"
    ".hidden bb_serif_ttf_end\n"
    ".global bb_serif_ttf_end\n"
    "bb_serif_ttf_end:\n"
    ".previous\n");
#endif
extern "C" const unsigned char bb_serif_ttf[];
extern "C" const unsigned char bb_serif_ttf_end[];

extern "C" void runtime_restart(void); // bb-probe (probe.c)

namespace BbOverlay {

namespace {

std::mutex imgui_mutex; // the ImGui context: window thread (input) and present thread
bool initialized = false;
std::atomic<bool> menu_open{false};
bool l3_down = false, r3_down = false;
bool dirty = false; // settings changed while open: saved on close
float base_scale = 1.0f;
ImFont* sans_font = nullptr;
ImFont* serif_font = nullptr;
bool focus_request = true; // the menu was opened: its first row takes the focus

// The game's text dialog (ImeDialog, the character name), typed on the keyboard: drawn while it
// is open. In fullscreen the window title that showed it is not visible (issues #17, #19).
std::mutex prompt_mutex;
std::atomic<bool> prompt_active{false};
std::string prompt_title, prompt_text;

// Present rate for the FPS counter.
std::chrono::steady_clock::time_point last_present{};
float frame_ms_avg = 0.0f;

float PixelDensity(SDL_WindowID id);

void SetOpen(bool value) {
    if (menu_open.exchange(value) == value) {
        return;
    }
    // The system cursor shows over the menu (window.cpp); ImGui learns where it is now, not at
    // the next motion: mouse motion is not passed on while the menu is closed.
    if (value) {
        focus_request = true;
        if (SDL_Window* window = SDL_GetMouseFocus()) {
            float x = 0.0f, y = 0.0f;
            SDL_GetMouseState(&x, &y);
            const float density = PixelDensity(SDL_GetWindowID(window));
            ImGui::GetIO().AddMousePosEvent(x * density, y * density);
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
    case SDLK_Q: return ImGuiKey_Q;
    case SDLK_E: return ImGuiKey_E;
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

// Marks the settings dirty when a widget changed them.
template <typename T>
void Store(std::atomic<T>& target, T value, bool changed) {
    if (changed) {
        target = value;
        dirty = true;
    }
}

void Checkbox(const char* label, std::atomic<bool>& value) {
    bool v = value;
    Store(value, v, ImGui::Checkbox(label, &v));
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

// The settings menu, laid out like a PC game's options screen in the game's own style: the game
// dimmed behind, a centred panel framed by ornament lines, tabs (L1/R1, Q/E, mouse), one setting
// per row ("name  < value >": left/right change it, Cross/Enter steps it, the mouse clicks either
// side), the focused row's description at the bottom and the controls in the footer.
namespace Ui {

float S(float v) {
    return v * base_scale;
}
const char* T(const char* english, const char* russian) {
    return BbSettings::MenuText(english, russian);
}
ImU32 Gold(float alpha = 1.0f) {
    return ImGui::GetColorU32(ImVec4(0.80f, 0.68f, 0.46f, alpha));
}
ImU32 Bright(float alpha = 1.0f) {
    return ImGui::GetColorU32(ImVec4(0.96f, 0.88f, 0.70f, alpha));
}
ImU32 Plain(float alpha = 1.0f) {
    return ImGui::GetColorU32(ImVec4(0.80f, 0.77f, 0.70f, alpha));
}
ImU32 Dim(float alpha = 1.0f) {
    return ImGui::GetColorU32(ImVec4(0.48f, 0.46f, 0.42f, alpha));
}

/// The game's dialog border: a line fading out to both ends, a small diamond in the middle.
void Ornament(ImDrawList* draw, float x0, float x1, float y) {
    const float mid = (x0 + x1) * 0.5f, t = S(1.2f), d = S(5.0f);
    draw->AddRectFilledMultiColor(ImVec2(x0, y - t * 0.5f), ImVec2(mid, y + t * 0.5f), Gold(0.0f),
                                  Gold(0.85f), Gold(0.85f), Gold(0.0f));
    draw->AddRectFilledMultiColor(ImVec2(mid, y - t * 0.5f), ImVec2(x1, y + t * 0.5f), Gold(0.85f),
                                  Gold(0.0f), Gold(0.0f), Gold(0.85f));
    draw->AddQuadFilled(ImVec2(mid, y - d), ImVec2(mid + d, y), ImVec2(mid, y + d),
                        ImVec2(mid - d, y), Gold(0.95f));
    draw->AddQuadFilled(ImVec2(mid - d * 3.0f, y - d * 0.5f), ImVec2(mid - d * 2.5f, y),
                        ImVec2(mid - d * 3.0f, y + d * 0.5f), ImVec2(mid - d * 3.5f, y), Gold(0.7f));
    draw->AddQuadFilled(ImVec2(mid + d * 3.0f, y - d * 0.5f), ImVec2(mid + d * 3.5f, y),
                        ImVec2(mid + d * 3.0f, y + d * 0.5f), ImVec2(mid + d * 2.5f, y), Gold(0.7f));
}

// Focus is the menu's own, as in a game: a row index moved by up/down to the next enabled row
// (ImGui's navigation did not reach these rows), set by the mouse, scrolled into view.
struct State {
    int tab = 0;
    bool focus_first = true;
    int focus = 0;                   ///< the focused row of the current tab
    int move = 0;                    ///< -1 / +1: up / down pressed this frame
    bool activate = false;           ///< Cross / Enter pressed this frame
    bool scroll = false;             ///< bring the focused row into view
    int rows = 0;                    ///< rows drawn this frame
    std::vector<bool> enabled;       ///< per row drawn this frame
    const char* hint = nullptr;      ///< the focused row's description
    const char* next_hint = nullptr; ///< collected while drawing this frame
};
State state;

/// One setting row. Returns -1 / +1 when the value is to step back / forward (keys, Cross, a
/// click on either side of the value), 0 otherwise. A slider row also takes a drag: `drag`
/// receives the fraction under the mouse.
int Row(const char* id, const char* label, const char* value, const char* hint, bool enabled,
        float slider = -1.0f, float* drag = nullptr) {
    ImGui::PushID(id);
    const int index = state.rows++;
    state.enabled.push_back(enabled);
    const float width = ImGui::GetContentRegionAvail().x, height = S(44.0f);
    const ImVec2 at = ImGui::GetCursorScreenPos();
    ImGui::BeginDisabled(!enabled);
    const bool pressed = ImGui::InvisibleButton("##row", ImVec2(width, height));
    const bool hovered = ImGui::IsItemHovered();
    const bool active = ImGui::IsItemActive();
    ImGui::EndDisabled();
    if (enabled && hovered && (ImGui::GetIO().MouseDelta.x != 0.0f || ImGui::GetIO().MouseDelta.y != 0.0f ||
                               pressed)) {
        state.focus = index;
    }
    const bool focused = enabled && index == state.focus;
    if (focused && state.scroll) {
        ImGui::SetScrollHereY(0.5f);
        state.scroll = false;
    }
    if (focused && hint) {
        state.next_hint = hint;
    }
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const float mid_y = at.y + height * 0.5f;
    if (focused || hovered) {
        // The game's highlight: a glow strongest under the text, fading to the sides.
        const float a = focused ? 0.22f : 0.10f;
        const float x0 = at.x, x1 = at.x + width, xm = at.x + width * 0.35f;
        draw->AddRectFilledMultiColor(ImVec2(x0, at.y), ImVec2(xm, at.y + height), Gold(0.0f),
                                      Gold(a), Gold(a), Gold(0.0f));
        draw->AddRectFilledMultiColor(ImVec2(xm, at.y), ImVec2(x1, at.y + height), Gold(a),
                                      Gold(0.0f), Gold(0.0f), Gold(a));
        if (focused) {
            draw->AddLine(ImVec2(x0 + width * 0.1f, at.y + height - S(1.0f)),
                          ImVec2(x1 - width * 0.1f, at.y + height - S(1.0f)), Gold(0.45f), S(1.0f));
        }
    }
    const ImU32 label_color = !enabled ? Dim() : focused ? Bright() : Plain();
    const ImVec2 label_size = ImGui::CalcTextSize(label);
    draw->AddText(ImVec2(at.x + S(28.0f), mid_y - label_size.y * 0.5f), label_color, label);

    // The value column: "<  value  >" centred, or a bar for sliders.
    const float col0 = at.x + width * 0.56f, col1 = at.x + width - S(28.0f);
    const float col_mid = (col0 + col1) * 0.5f;
    int step = 0;
    if (enabled && value) {
        const ImU32 arrow = focused ? Gold(1.0f) : Gold(0.55f);
        const float h = S(7.0f);
        draw->AddTriangleFilled(ImVec2(col0, mid_y), ImVec2(col0 + h, mid_y - h),
                                ImVec2(col0 + h, mid_y + h), arrow);
        draw->AddTriangleFilled(ImVec2(col1, mid_y), ImVec2(col1 - h, mid_y - h),
                                ImVec2(col1 - h, mid_y + h), arrow);
    }
    if (slider >= 0.0f) {
        const float b0 = col0 + S(24.0f), b1 = col1 - S(24.0f) - S(60.0f);
        const float bh = S(3.0f);
        draw->AddRectFilled(ImVec2(b0, mid_y - bh), ImVec2(b1, mid_y + bh), Dim(0.6f), bh);
        draw->AddRectFilled(ImVec2(b0, mid_y - bh),
                            ImVec2(b0 + (b1 - b0) * std::clamp(slider, 0.0f, 1.0f), mid_y + bh),
                            enabled ? Gold(0.9f) : Dim(), bh);
        const float knob = b0 + (b1 - b0) * std::clamp(slider, 0.0f, 1.0f);
        draw->AddCircleFilled(ImVec2(knob, mid_y), S(7.0f), enabled ? Bright() : Dim());
        if (value) {
            const ImVec2 size = ImGui::CalcTextSize(value);
            draw->AddText(ImVec2(col1 - S(24.0f) - size.x, mid_y - size.y * 0.5f),
                          enabled ? Plain() : Dim(), value);
        }
        if (enabled && drag && active && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            const float x = ImGui::GetIO().MousePos.x;
            if (x >= b0 - S(12.0f) && x <= b1 + S(12.0f)) {
                *drag = std::clamp((x - b0) / (b1 - b0), 0.0f, 1.0f);
            }
        }
    } else if (value) {
        const ImVec2 size = ImGui::CalcTextSize(value);
        draw->AddText(ImVec2(col_mid - size.x * 0.5f, mid_y - size.y * 0.5f),
                      !enabled ? Dim() : focused ? Bright() : Plain(), value);
    }
    if (enabled && focused) {
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow) || ImGui::IsKeyPressed(ImGuiKey_GamepadDpadLeft) ||
            ImGui::IsKeyPressed(ImGuiKey_GamepadLStickLeft)) {
            step = -1;
        } else if (ImGui::IsKeyPressed(ImGuiKey_RightArrow) ||
                   ImGui::IsKeyPressed(ImGuiKey_GamepadDpadRight) ||
                   ImGui::IsKeyPressed(ImGuiKey_GamepadLStickRight)) {
            step = 1;
        }
    }
    if (enabled && pressed && slider < 0.0f) {
        step = ImGui::GetIO().MousePos.x < col_mid && ImGui::GetIO().MousePos.x >= col0 - S(12.0f) ? -1 : 1;
    } else if (focused && state.activate) {
        step = 1;
        state.activate = false;
    }
    ImGui::PopID();
    return step;
}

/// A non-interactive line under the rows (state, warnings).
void Note(const char* text, ImU32 color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::Indent(S(28.0f));
    ImGui::PushTextWrapPos(ImGui::GetContentRegionAvail().x - S(28.0f));
    ImGui::TextUnformatted(text);
    ImGui::PopTextWrapPos();
    ImGui::Unindent(S(28.0f));
    ImGui::PopStyleColor();
}

const char* OnOff(bool on) {
    return on ? T("On", "Вкл") : T("Off", "Выкл");
}

void Toggle(const char* id, const char* label, std::atomic<bool>& value, const char* hint,
            bool enabled = true) {
    if (Row(id, label, OnOff(value), hint, enabled) != 0) {
        Store(value, !value.load(), true);
    }
}

/// A choice among `count` values (wrapping); `allowed(i)` false skips a value.
template <typename Label, typename Allowed>
void Choice(const char* id, const char* label, int current, int count, Label&& name,
            Allowed&& allowed, const char* hint, bool enabled, const std::function<void(int)>& set) {
    const int step = Row(id, label, name(current), hint, enabled);
    if (step == 0) {
        return;
    }
    for (int n = 1; n <= count; ++n) {
        const int next = ((current + step * n) % count + count) % count;
        if (allowed(next)) {
            if (next != current) {
                set(next);
            }
            return;
        }
    }
}

void SliderRow(const char* id, const char* label, std::atomic<float>& value, float lo, float hi,
               float step_size, const char* hint, bool enabled = true) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.2f", value.load());
    float drag = -1.0f;
    const int step = Row(id, label, text, hint, enabled, (value - lo) / (hi - lo), &drag);
    if (step != 0) {
        const float v = std::clamp(std::round((value + step * step_size) / step_size) * step_size, lo, hi);
        Store(value, v, true);
    } else if (drag >= 0.0f) {
        const float v = std::round((lo + drag * (hi - lo)) / 0.01f) * 0.01f;
        Store(value, v, v != value);
    }
}

bool RestartNeeded() {
    auto& s = BbSettings::Get();
    bool restart = s.object_motion != s.startup_object_motion || s.model_lod != s.startup_model_lod ||
                   s.live_resolution != s.startup_live_resolution ||
                   BbSettings::ResolutionNeedsRestart();
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        restart |= s.effects[e] != s.startup_effects[e];
    }
    return restart;
}

void DisplayTab() {
    auto& s = BbSettings::Get();
    static const char* outputs[] = {"1280 x 720", "1920 x 1080", "2560 x 1440", "3840 x 2160"};
    Choice("output", T("Output resolution", "Разрешение вывода"), s.output_res, BbSettings::OutputCount,
           [](int i) { return outputs[i]; }, [](int) { return true; },
           BbSettings::FixedRenderSession()
               ? T("Size of the final frame and UI. The preset sets the scene size relative to "
                   "the output: 4K Performance = 1920x1080. Applies after restarting the game.",
                   "Размер готового кадра и интерфейса. Пресет задаёт размер сцены относительно "
                   "вывода: 4K Performance = 1920x1080. Применяется после перезапуска игры.")
               : T("The final frame and UI size changes at the next frame. The preset sets the "
                   "scene size relative to the output: 4K Performance = 1920x1080. Changing it "
                   "resets the upscaler's history (a brief pause).",
                   "Размер готового кадра и интерфейса меняется со следующего кадра. Пресет задаёт "
                   "размер сцены относительно вывода: 4K Performance = 1920x1080. Смена сбрасывает "
                   "историю апскейлера (короткая пауза)."),
           true, [&](int i) { Store(s.output_res, i, true); });
    const char* live_modes[] = {T("Auto (by GPU)", "Авто (по видеокарте)"), T("Off (faster)", "Выкл (быстрее)"),
                                T("On", "Вкл")};
    Choice("live", T("Live resolution changes", "Смена разрешения на лету"), s.live_resolution + 1, 3,
           [&](int i) { return live_modes[i]; }, [](int) { return true; },
           T("On: output resolution and preset change without restarting, but the game's "
             "post-processing stays at 1080p (slower on the Steam Deck and older GPUs). Off: "
             "everything renders at the preset's size, changes need a restart. Auto turns it on "
             "for strong discrete GPUs. Applies after restarting the game.",
             "Вкл: разрешение вывода и пресет меняются без перезапуска, но постобработка игры "
             "остаётся в 1080p (медленнее на Steam Deck и старых видеокартах). Выкл: всё рисуется "
             "в размере пресета, смена — через перезапуск. Авто включает её на мощных дискретных "
             "видеокартах. Применяется после перезапуска игры."),
           true, [&](int i) { Store(s.live_resolution, i - 1, true); });
    Toggle("fps", T("FPS counter", "Счётчик FPS"), s.show_fps,
           T("Frames per second and frame time in the top right corner.",
             "Кадры в секунду и время кадра в правом верхнем углу."));
    static const char* languages[] = {"English", "Русский"};
    Choice("language", "Language", s.menu_language == BbSettings::MenuLanguage::Russian ? 1 : 0, 2,
           [](int i) { return languages[i]; }, [](int) { return true; },
           T("Language of this menu.", "Язык этого меню."), true, [&](int i) {
               s.menu_language = i == 1 ? BbSettings::MenuLanguage::Russian
                                        : BbSettings::MenuLanguage::English;
               BbSettings::Save();
           });
}

void UpscalerTab() {
    auto& s = BbSettings::Get();
    const char* upscalers[] = {T("Off", "Выкл"), "FSR 3.1", "FSR 4 (INT8)", "FSR 4.1.1",
                               T("TAA (native)", "TAA (нативное)"), "DLSS (NVIDIA RTX)",
                               "MetalFX (Apple)"};
    static_assert(std::size(upscalers) == BbSettings::UpscalerCount);
    // Unsupported values are skipped: FSR 4 and DLSS on the Mac, MetalFX elsewhere.
    const auto supported = [&](int i) {
        return i == BbSettings::UpscalerFsr4      ? s.fsr4_supported.load()
               : i == BbSettings::UpscalerFsr411  ? s.fsr411_supported.load()
               : i == BbSettings::UpscalerDlss    ? s.dlss_supported.load()
               : i == BbSettings::UpscalerMetalFx ? s.metalfx_supported.load()
                                                  : true;
    };
    const char* upscaler_hint =
        s.upscaler == BbSettings::UpscalerFsr411
            ? T("FSR 4.1.1: the model from AMD's 4.1.1 DLL, reproduced in Vulkan (output matches "
                "the DLL). Assets are built from your DLL in the launcher.",
                "FSR 4.1.1: модель из DLL AMD 4.1.1, воспроизведённая в Vulkan (результат совпадает "
                "с DLL). Ассеты собираются из вашей DLL в лаунчере.")
        : s.upscaler == BbSettings::UpscalerFsr4
            ? T("FSR 4 in INT8 mode (v07 model from AMD's FidelityFX SDK sources): higher quality "
                "than FSR 3.1, heavier.",
                "FSR 4 в режиме INT8 (модель v07 из исходников AMD FidelityFX SDK): качество выше, "
                "чем у FSR 3.1, проход тяжелее.")
        : s.upscaler == BbSettings::UpscalerTaa
            ? T("TAA anti-aliases the scene at the output resolution, without an FSR model.",
                "TAA сглаживает сцену в разрешении вывода, без модели FSR.")
        : s.upscaler == BbSettings::UpscalerMetalFx
            ? T("MetalFX: Apple's temporal upscaler, on the same GPU as KosmicKrisp. Slower than "
                "FSR 3.1 here (each frame goes from Vulkan to Metal and back). Native AA works only "
                "if the GPU supports a scale of 1.",
                "MetalFX: временной апскейлер Apple на том же GPU, что и KosmicKrisp. Здесь медленнее "
                "FSR 3.1 (кадр переходит из Vulkan в Metal и обратно). Native AA работает, только "
                "если GPU поддерживает масштаб 1.")
        : s.upscaler == BbSettings::UpscalerDlss
            ? T("NVIDIA DLSS Super Resolution (RTX GPUs; Native AA is DLAA). Needs the DLSS bridge "
                "and NVIDIA's library next to the game; otherwise FSR 3.1 is used.",
                "NVIDIA DLSS Super Resolution (видеокарты RTX; Native AA — это DLAA). Нужны мост DLSS "
                "и библиотека NVIDIA рядом с игрой; без них используется FSR 3.1.")
            : T("Temporal upscaler: anti-aliasing and rendering below the output resolution. FSR 4 "
                "needs a GPU with the required features (unavailable values are skipped).",
                "Временной апскейлер: сглаживание и отрисовка ниже разрешения вывода. FSR 4 нужна "
                "видеокарта с нужными возможностями (недоступные значения пропускаются).");
    Choice("upscaler", T("Upscaler", "Апскейлер"), s.upscaler, BbSettings::UpscalerCount,
           [&](int i) { return upscalers[i]; }, supported, upscaler_hint, true,
           [&](int i) { Store(s.upscaler, i, true); });
    const bool upscaler_on = s.upscaler != BbSettings::UpscalerOff;
    const bool taa = s.upscaler == BbSettings::UpscalerTaa;
    static char preset_names[BbSettings::PresetCount][64];
    for (int i = 0; i < BbSettings::PresetCount; ++i) {
        const float scale = BbSettings::PresetScale(i);
        std::snprintf(preset_names[i], sizeof(preset_names[i]), "%s  %dx%d", BbSettings::PresetName(i),
                      int(std::lround(BbSettings::OutputWidths[s.output_res] / scale / 2) * 2),
                      int(std::lround(BbSettings::OutputHeights[s.output_res] / scale / 2) * 2));
    }
    Choice("preset", T("Quality preset", "Пресет качества"), taa ? BbSettings::NativeAA : s.preset.load(),
           BbSettings::PresetCount, [](int i) { return preset_names[i]; }, [](int) { return true; },
           T("Native AA: the upscaler only anti-aliases. The other presets render the scene below "
             "the output resolution (the size shown); the UI stays at the output resolution.",
             "Native AA: апскейлер только сглаживает. Остальные пресеты рисуют сцену ниже "
             "разрешения вывода (показан размер); интерфейс — в разрешении вывода."),
           upscaler_on && !taa, [&](int i) { Store(s.preset, i, true); });
    Toggle("sharpen", T("Sharpening", "Резкость"), s.sharpen,
           T("RCAS sharpening after the upscaler.", "Повышение резкости RCAS после апскейлера."),
           upscaler_on);
    SliderRow("sharpness", T("Sharpness", "Сила резкости"), s.sharpness, 0.0f, 2.0f, 0.05f,
              T("Up to 1: the upscaler's own RCAS. Above 1 another RCAS pass is added.",
                "До 1 — RCAS самого апскейлера. Выше 1 добавляется ещё один проход RCAS."),
              upscaler_on && s.sharpen);
    Toggle("jitter", T("Subpixel jitter", "Субпиксельный сдвиг"), s.jitter,
           T("Each frame shifts the scene by a fraction of a pixel so the upscaler reconstructs "
             "more detail from several frames.",
             "Каждый кадр сцена сдвигается на долю пикселя, и апскейлер собирает больше деталей из "
             "нескольких кадров."),
           upscaler_on);
    Toggle("motion", T("Character motion vectors", "Векторы движения персонажей"), s.object_motion,
           T("Accurate motion for animated objects: clothes and weapons break up less in motion. "
             "Applies after restarting the game.",
             "Точное движение анимированных объектов: одежда и оружие меньше рассыпаются в "
             "движении. Применяется после перезапуска игры."),
           upscaler_on);
    if (BbSettings::IsFsr4(s.upscaler)) {
        Toggle("fsr4_exposure", T("FSR 4: auto exposure", "FSR 4: автоэкспозиция"), s.fsr4_auto_exposure,
               T("The FSR 4 network normalises colour by exposure and uses it to decide when to "
                 "drop older frames.",
                 "Сеть FSR 4 нормирует цвет по экспозиции и по ней решает, когда отбросить прошлые "
                 "кадры."));
        Toggle("fsr4_jitter", T("FSR 4: invert jitter sign", "FSR 4: обратный знак сдвига"),
               s.fsr4_invert_jitter, T("For diagnosing ghosting.", "Для проверки шлейфов."));
    }
    char line[160];
    std::snprintf(line, sizeof(line), T("Scene renders at %d x %d", "Сцена рисуется в %d x %d"),
                  s.active_render_width.load(), s.active_render_height.load());
    ImGui::Spacing();
    Note(line, Dim());
    if (BbSettings::FixedRenderSession()) {
        std::snprintf(line, sizeof(line), T("Preset at startup: %s (changes after a restart)",
                                            "Пресет при запуске: %s (смена — после перезапуска)"),
                      BbSettings::PresetName(s.startup_preset));
        Note(line, Dim());
    }
    if (const char* problem = s.metalfx_problem.load()) {
        std::snprintf(line, sizeof(line), T("MetalFX unavailable: %s", "MetalFX недоступен: %s"),
                      problem);
        Note(line, ImGui::GetColorU32(ImVec4(1.0f, 0.55f, 0.35f, 1.0f)));
    }
    if (const char* problem = s.dlss_problem.load(); problem && s.upscaler == BbSettings::UpscalerDlss) {
        std::snprintf(line, sizeof(line), "DLSS: %s", problem);
        Note(line, ImGui::GetColorU32(ImVec4(1.0f, 0.55f, 0.35f, 1.0f)));
    }
    if (const char* problem = s.fsr4_problem.load()) {
        std::snprintf(line, sizeof(line), T("FSR 4 unavailable: %s", "FSR 4 недоступен: %s"), problem);
        Note(line, ImGui::GetColorU32(ImVec4(1.0f, 0.55f, 0.35f, 1.0f)));
    }
}

void EffectsTab() {
    auto& s = BbSettings::Get();
    const char* lods[] = {T("Highest (-2)", "Максимальная (-2)"), T("Game default", "Как в игре"),
                          T("Lower (1)", "Ниже (1)"), T("Lowest (2)", "Минимальная (2)")};
    static constexpr int lod_values[] = {-2, 0, 1, 2};
    int lod_index = 1;
    for (int i = 0; i < 4; ++i) {
        if (lod_values[i] == s.model_lod) lod_index = i;
    }
    Choice("lod", T("Model detail", "Детализация моделей"), lod_index, 4, [&](int i) { return lods[i]; },
           [](int) { return true; },
           T("Level of detail of models (game patch). Applies after restarting the game.",
             "Детализация моделей (патч игры). Применяется после перезапуска игры."),
           true, [&](int i) { Store(s.model_lod, lod_values[i], true); });
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        const auto& effect = BbSettings::Effects[e];
        const char* hint =
            std::string_view(effect.key) == "debug_camera"
                ? T("Hold Cross and press L3 (keyboard: Space + Z). Applies after restarting.",
                    "Удерживайте Cross и нажмите L3 (клавиатура: Space + Z). После перезапуска.")
            : std::string_view(effect.key) == "debug_menu"
                ? T("Left side of the touchpad (Tab). Needs the adhoc folder from Nexus mod #253. "
                    "Applies after restarting.",
                    "Левая сторона тачпада (Tab). Нужна папка adhoc из мода Nexus #253. После "
                    "перезапуска.")
                : T("A game patch, applied when the game starts: after a restart. Motion blur and "
                    "shadows from dynamic lights cost the most GPU time.",
                    "Патч игры, применяется при запуске: после перезапуска. Размытие в движении и тени "
                    "от динамических источников нагружают видеочип сильнее всего.");
        Toggle(effect.key, T(effect.label, effect.label_ru), s.effects[e], hint);
    }
}

void AdvancedTab() {
    auto& s = BbSettings::Get();
    const bool upscaler_on = s.upscaler != BbSettings::UpscalerOff;
    const bool taa = s.upscaler == BbSettings::UpscalerTaa;
    Toggle("reactive", T("Reactive mask", "Маска реактивности"), s.reactive,
           BbSettings::IsFsr4(s.upscaler)
               ? T("Marks transparent effects (haze, light, water, particles): there the FSR 4 "
                   "output is blended with the current frame so effects do not drag older frames.",
                   "Помечает прозрачные эффекты (дымку, свет, воду, частицы): там результат FSR 4 "
                   "смешивается с текущим кадром, и эффекты не тянут за собой прошлые кадры.")
               : T("Marks transparent effects so the upscaler relies less on older frames: fewer "
                   "trails, more shimmer under them.",
                   "Помечает прозрачные эффекты, чтобы апскейлер меньше опирался на прошлые кадры: "
                   "меньше шлейфов, больше дрожания под ними."),
           upscaler_on && !taa);
    const bool mask = upscaler_on && !taa && s.reactive;
    SliderRow("reactive_scale", T("Mask scale", "Масштаб маски"), s.reactive_scale, 0.0f, 4.0f, 0.1f,
              nullptr, mask);
    SliderRow("reactive_threshold", T("Mask threshold", "Порог маски"), s.reactive_threshold, 0.0f,
              1.0f, 0.05f, nullptr, mask);
    SliderRow("reactive_max", T("Mask maximum", "Максимум маски"), s.reactive_max, 0.0f, 1.0f, 0.05f,
              nullptr, mask);
    const char* views[] = {T("Off", "Выкл"), T("Reactive mask", "Маска реактивности"),
                           T("Motion vectors", "Векторы движения")};
    Choice("debug_view", T("Debug view", "Отладочный вид"), s.debug_view, BbSettings::DebugViewCount,
           [&](int i) { return views[i]; }, [](int) { return true; },
           T("Motion vectors: red/green = horizontal/vertical motion (8 px = full), blue = the pixel "
             "has an accurate object vector. A moving object with neither is a trail.",
             "Векторы движения: красный/зелёный — движение по горизонтали/вертикали (8 пикселей = "
             "полная яркость), синий — у пикселя точный вектор объекта. Движущийся предмет без того "
             "и другого даёт шлейф."),
           upscaler_on, [&](int i) { s.debug_view = i; });
}

} // namespace Ui

void Menu() {
    using namespace Ui;
    ImGuiIO& io = ImGui::GetIO();
    const ImVec2 display = io.DisplaySize;
    ImGui::GetBackgroundDrawList()->AddRectFilled(ImVec2(0, 0), display, IM_COL32(0, 0, 0, 165));

    static const char* tab_en[] = {"Display", "Upscaler", "Game effects", "Advanced"};
    static const char* tab_ru[] = {"Изображение", "Апскейлер", "Эффекты игры", "Дополнительно"};
    constexpr int tabs = 4;
    if (focus_request) {
        focus_request = false;
        state.focus_first = true;
    }
    const bool prev_tab = ImGui::IsKeyPressed(ImGuiKey_GamepadL1, false) ||
                          ImGui::IsKeyPressed(ImGuiKey_Q, false) ||
                          ImGui::IsKeyPressed(ImGuiKey_PageUp, false);
    const bool next_tab = ImGui::IsKeyPressed(ImGuiKey_GamepadR1, false) ||
                          ImGui::IsKeyPressed(ImGuiKey_E, false) ||
                          ImGui::IsKeyPressed(ImGuiKey_PageDown, false);
    if (prev_tab || next_tab) {
        state.tab = (state.tab + (next_tab ? 1 : tabs - 1)) % tabs;
        state.focus_first = true;
    }
    if (state.focus_first) {
        state.focus_first = false;
        state.focus = 0;
        state.scroll = true;
    }
    const auto pressed_any = [](std::initializer_list<ImGuiKey> keys) {
        for (const ImGuiKey key : keys) {
            if (ImGui::IsKeyPressed(key, true)) {
                return true;
            }
        }
        return false;
    };
    state.move = pressed_any({ImGuiKey_UpArrow, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadLStickUp})       ? -1
                 : pressed_any({ImGuiKey_DownArrow, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadLStickDown}) ? 1
                                                                                                          : 0;
    state.activate = ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false) ||
                     ImGui::IsKeyPressed(ImGuiKey_Space, false) || ImGui::IsKeyPressed(ImGuiKey_GamepadFaceDown, false);
    state.rows = 0;
    state.enabled.clear();

    const ImVec2 size(std::min(S(1180.0f), display.x - S(40.0f)), std::min(S(800.0f), display.y - S(40.0f)));
    ImGui::SetNextWindowPos(ImVec2(display.x * 0.5f, display.y * 0.5f), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(size, ImGuiCond_Always);
    ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.035f, 0.032f, 0.028f, 0.94f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(S(36.0f), S(24.0f)));
    ImGui::PushFont(serif_font, 21.0f);
    ImGui::Begin("##bbport_settings", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoScrollWithMouse);
    ImDrawList* draw = ImGui::GetWindowDrawList();
    const ImVec2 win = ImGui::GetWindowPos();
    const float x0 = win.x + S(24.0f), x1 = win.x + size.x - S(24.0f);
    Ornament(draw, x0, x1, win.y + S(10.0f));
    Ornament(draw, x0, x1, win.y + size.y - S(10.0f));

    // Title.
    ImGui::PushFont(serif_font, 34.0f);
    const char* title = T("Settings", "Настройки");
    const float title_w = ImGui::CalcTextSize(title).x;
    ImGui::SetCursorPosX((size.x - title_w) * 0.5f);
    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(Bright()));
    ImGui::TextUnformatted(title);
    ImGui::PopStyleColor();
    ImGui::PopFont();
    // Close (mouse), top right.
    {
        const char* close = "×";
        ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
        ImGui::SetCursorPos(ImVec2(size.x - S(64.0f), S(22.0f)));
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(Gold()));
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::ColorConvertU32ToFloat4(Gold(0.2f)));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::ColorConvertU32ToFloat4(Gold(0.35f)));
        if (ImGui::Button(close, ImVec2(S(34.0f), S(34.0f)))) {
            SetOpen(false);
        }
        ImGui::PopStyleColor(4);
        ImGui::PopItemFlag();
    }

    // Tabs: L1 ... R1, the current one gold and underlined.
    ImGui::SetCursorPosY(S(76.0f));
    {
        const float row_y = ImGui::GetCursorScreenPos().y;
        float widths[tabs], total = 0.0f;
        const float gap = S(46.0f);
        for (int i = 0; i < tabs; ++i) {
            widths[i] = ImGui::CalcTextSize(T(tab_en[i], tab_ru[i])).x;
            total += widths[i] + (i ? gap : 0.0f);
        }
        float x = win.x + (size.x - total) * 0.5f;
        ImGui::PushFont(sans_font, 15.0f);
        const ImVec2 l1 = ImGui::CalcTextSize("L1 / Q");
        draw->AddText(ImVec2(x - gap - l1.x, row_y + S(4.0f)), Gold(0.6f), "L1 / Q");
        draw->AddText(ImVec2(x + total + gap, row_y + S(4.0f)), Gold(0.6f), "E / R1");
        ImGui::PopFont();
        ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
        for (int i = 0; i < tabs; ++i) {
            const char* name = T(tab_en[i], tab_ru[i]);
            ImGui::SetCursorScreenPos(ImVec2(x, row_y));
            ImGui::PushID(i);
            if (ImGui::InvisibleButton("##tab", ImVec2(widths[i], S(32.0f)))) {
                state.tab = i;
                state.focus_first = true;
            }
            const bool hovered = ImGui::IsItemHovered();
            ImGui::PopID();
            const bool current = i == state.tab;
            draw->AddText(ImVec2(x, row_y), current ? Bright() : hovered ? Plain() : Dim(), name);
            if (current) {
                draw->AddLine(ImVec2(x - S(6.0f), row_y + S(31.0f)), ImVec2(x + widths[i] + S(6.0f), row_y + S(31.0f)),
                              Gold(0.9f), S(1.5f));
            }
            x += widths[i] + gap;
        }
        ImGui::PopItemFlag();
        ImGui::SetCursorScreenPos(ImVec2(win.x + S(36.0f), row_y + S(44.0f)));
    }
    draw->AddLine(ImVec2(x0 + S(20.0f), ImGui::GetCursorScreenPos().y),
                  ImVec2(x1 - S(20.0f), ImGui::GetCursorScreenPos().y), Gold(0.25f), S(1.0f));
    ImGui::Dummy(ImVec2(0, S(6.0f)));

    // Rows (scrolling when they do not fit), then the description and the controls.
    const float footer = S(150.0f);
    state.next_hint = nullptr;
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, S(2.0f)));
    ImGui::PushStyleVar(ImGuiStyleVar_ScrollbarSize, S(6.0f));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrab, ImGui::ColorConvertU32ToFloat4(Gold(0.30f)));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabHovered, ImGui::ColorConvertU32ToFloat4(Gold(0.55f)));
    ImGui::PushStyleColor(ImGuiCol_ScrollbarGrabActive, ImGui::ColorConvertU32ToFloat4(Gold(0.75f)));
    ImGui::BeginChild("##rows", ImVec2(0.0f, ImGui::GetContentRegionAvail().y - footer), ImGuiChildFlags_None,
                      ImGuiWindowFlags_NoBackground);
    switch (state.tab) {
    case 0: DisplayTab(); break;
    case 1: UpscalerTab(); break;
    case 2: EffectsTab(); break;
    default: AdvancedTab(); break;
    }
    if (RestartNeeded()) {
        ImGui::Spacing();
        if (Row("restart", T("Apply and restart the game", "Применить и перезапустить игру"), nullptr,
                T("Some changes apply only when the game starts (game patches, resolution at "
                  "startup, motion vectors). The game restarts from the title screen; your "
                  "progress is in the save.",
                  "Часть изменений применяется только при запуске игры (патчи игры, разрешение при "
                  "запуске, векторы движения). Игра перезапустится с титульного экрана; прогресс — в "
                  "сохранении."),
                true) != 0) {
            BbSettings::Save();
            runtime_restart();
        }
    }
    // Up / down: the next enabled row (also off a row that became disabled).
    if (state.rows != 0) {
        int focus = std::clamp(state.focus, 0, state.rows - 1);
        const int dir = state.move != 0 ? state.move : 1;
        if (state.move != 0 || !state.enabled[focus]) {
            for (int n = state.move != 0 ? 1 : 0; n < state.rows; ++n) {
                const int i = ((focus + dir * n) % state.rows + state.rows) % state.rows;
                if (state.enabled[i]) {
                    if (i != state.focus) {
                        state.scroll = true;
                    }
                    focus = i;
                    break;
                }
            }
        }
        state.focus = focus;
    }
    ImGui::EndChild();
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
    state.hint = state.next_hint;

    // Description of the focused setting.
    const float desc_y = win.y + size.y - footer;
    draw->AddLine(ImVec2(x0 + S(20.0f), desc_y + S(6.0f)), ImVec2(x1 - S(20.0f), desc_y + S(6.0f)), Gold(0.25f),
                  S(1.0f));
    ImGui::SetCursorScreenPos(ImVec2(win.x + S(64.0f), desc_y + S(18.0f)));
    ImGui::PushFont(serif_font, 17.0f);
    ImGui::BeginGroup(); // the lines start where the first one does
    ImGui::PushTextWrapPos(size.x - S(64.0f));
    if (RestartNeeded()) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.95f, 0.72f, 0.35f, 1.0f));
        ImGui::TextUnformatted(T("Some changes apply after restarting the game.",
                                 "Часть изменений применится после перезапуска игры."));
        ImGui::PopStyleColor();
    }
    if (state.hint) {
        ImGui::PushStyleColor(ImGuiCol_Text, ImGui::ColorConvertU32ToFloat4(Plain(0.85f)));
        ImGui::TextUnformatted(state.hint);
        ImGui::PopStyleColor();
    }
    ImGui::PopTextWrapPos();
    ImGui::EndGroup();
    ImGui::PopFont();

    // Controls.
    ImGui::PushFont(sans_font, 15.0f);
#ifdef __APPLE__ // bbport: Mac keyboards have no Insert key (HandleEvent)
    const char* controls = T("Enter / Cross: change     Left / Right: value     Q / E, L1 / R1: tabs     "
                             "Esc / Circle: close     F1, `, L3 + R3: menu",
                             "Enter / Крест: изменить     Влево / Вправо: значение     Q / E, L1 / R1: "
                             "вкладки     Esc / Круг: закрыть     F1, `, L3 + R3: меню");
#else
    const char* controls = T("Enter / Cross: change     Left / Right: value     Q / E, L1 / R1: tabs     "
                             "Esc / Circle: close     Insert, L3 + R3: menu",
                             "Enter / Крест: изменить     Влево / Вправо: значение     Q / E, L1 / R1: "
                             "вкладки     Esc / Круг: закрыть     Insert, L3 + R3: меню");
#endif
    const ImVec2 cs = ImGui::CalcTextSize(controls);
    draw->AddText(ImVec2(win.x + (size.x - cs.x) * 0.5f, win.y + size.y - S(40.0f)), Gold(0.65f), controls);
    ImGui::PopFont();

    ImGui::End();
    ImGui::PopFont();
    ImGui::PopStyleVar(2);
    ImGui::PopStyleColor();
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
    ImGui::Text(BbSettings::MenuText("%.0f FPS  %.1f ms  %s", "%.0f FPS  %.1f мс  %s"),
                frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f, frame_ms_avg,
                s.upscaler == BbSettings::UpscalerFsr3     ? "FSR 3.1"
                : s.upscaler == BbSettings::UpscalerFsr4   ? "FSR 4"
                : s.upscaler == BbSettings::UpscalerFsr411 ? "FSR 4.1.1"
                : s.upscaler == BbSettings::UpscalerTaa    ? "TAA"
                : s.upscaler == BbSettings::UpscalerMetalFx ? "MetalFX"
                : s.upscaler == BbSettings::UpscalerDlss   ? "DLSS"
                                                           : "");
    ImGui::End();
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
    static const char* path = std::getenv("BB_MENU_KEYS_FILE");
    static std::chrono::steady_clock::time_point last_check{};
    static std::vector<std::string> queue;
    static bool release = false;
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
    // The menu moves its own focus (Ui::State); ImGui only gets the keys.
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
    sans_font = io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(bb_font_ttf),
                                               int(bb_font_ttf_end - bb_font_ttf), 18.0f, &font_config);
    ImFontConfig serif_config;
    serif_config.FontDataOwnedByAtlas = false;
    serif_font = io.Fonts->AddFontFromMemoryTTF(const_cast<unsigned char*>(bb_serif_ttf),
                                                int(bb_serif_ttf_end - bb_serif_ttf), 21.0f, &serif_config);

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
    std::printf("Overlay: menu ready (F1, ` or §, or L3+R3)\n");
#else
    std::printf("Overlay: menu ready (Insert or L3+R3)\n");
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
        if (down && l3_down && r3_down) {
            SetOpen(!is_open);
            return true;
        }
        if (!is_open) {
            return false;
        }
        if (down && button == SDL_GAMEPAD_BUTTON_EAST) { // Circle: back to the game
            SetOpen(false);
            return true;
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
        const float density = PixelDensity(event.motion.windowID);
        io.AddMousePosEvent(event.motion.x * density, event.motion.y * density);
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

bool Visible() {
    return initialized && (menu_open || prompt_active || BbSettings::Get().show_fps);
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
    if (std::getenv("BB_MENU_KEYS_FILE") && initialized) {
        std::scoped_lock lock{imgui_mutex};
        ScriptedKeys();
    }
    if (!Visible()) {
        return;
    }
    std::scoped_lock lock{imgui_mutex};
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(float(extent.width), float(extent.height));
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
    if (menu_open) {
        Menu();
    }
    if (BbSettings::Get().show_fps && !menu_open) {
        FpsCounter();
    }
    if (prompt_active && !menu_open) {
        TextPrompt();
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
