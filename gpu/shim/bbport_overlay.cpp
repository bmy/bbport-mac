// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_overlay.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <string>

#include <SDL3/SDL.h>
#include "bbport_lang.h"
#include "bbport_settings.h"
#include "imgui.h"
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

extern "C" void runtime_restart(void); // bb-probe (probe.c)

namespace BbOverlay {

namespace {

using BbLang::Tr;

std::mutex imgui_mutex; // the ImGui context: window thread (input) and present thread
bool initialized = false;
std::atomic<bool> menu_open{false};
bool l3_down = false, r3_down = false;
bool dirty = false; // settings changed while open: saved on close
float base_scale = 1.0f;

// The game's text dialog (ImeDialog, the character name), typed on the keyboard: drawn while it
// is open. In fullscreen the window title that showed it is not visible (issues #17, #19).
std::mutex prompt_mutex;
std::atomic<bool> prompt_active{false};
std::string prompt_title, prompt_text;

// Present rate for the FPS counter.
std::chrono::steady_clock::time_point last_present{};
float frame_ms_avg = 0.0f;

void SetOpen(bool value) {
    if (menu_open.exchange(value) == value) {
        return;
    }
    ImGui::GetIO().MouseDrawCursor = value;
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

void Menu() {
    auto& s = BbSettings::Get();
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + 40.0f * base_scale,
                                   viewport->WorkPos.y + 40.0f * base_scale),
                            ImGuiCond_Appearing);
    ImGui::SetNextWindowSize(ImVec2(620.0f * base_scale, 0.0f), ImGuiCond_Appearing);
    bool keep_open = true;
    // ###: the window ID does not depend on the language.
#ifdef __APPLE__
    const char* menu_title = Tr("Bloodborne — настройки  (F1 / ` / L3+R3)###bbport_menu",
                                "Bloodborne — settings  (F1 / ` / L3+R3)###bbport_menu");
#else
    const char* menu_title = Tr("Bloodborne — настройки  (Insert / L3+R3)###bbport_menu",
                                "Bloodborne — settings  (Insert / L3+R3)###bbport_menu");
#endif
    if (!ImGui::Begin(menu_title, &keep_open,
                      ImGuiWindowFlags_NoCollapse)) {
        ImGui::End();
        return;
    }
    ImGui::Text(Tr("%.0f FPS  (%.1f мс)", "%.0f FPS  (%.1f ms)"),
                frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f, frame_ms_avg);

    ImGui::SeparatorText(Tr("Временной апскейлер", "Temporal upscaler"));
    // Filled once: the language does not change while running.
    static const char* const upscalers[] = {
        Tr("Выкл", "Off"), "FSR 3.1", "FSR 4 (INT8)", "FSR 4.1.1 (INT8)",
        Tr("TAA (нативное сглаживание)", "TAA (native anti-aliasing)"), "MetalFX (Apple)"};
    static_assert(std::size(upscalers) == BbSettings::UpscalerCount);
    int upscaler = s.upscaler;
    if (ImGui::BeginCombo(Tr("Апскейлер", "Upscaler"), upscalers[upscaler])) {
        for (int i = 0; i < BbSettings::UpscalerCount; ++i) {
            const bool supported = i == BbSettings::UpscalerFsr4 ? s.fsr4_supported.load()
                : i == BbSettings::UpscalerFsr411 ? s.fsr411_supported.load()
                : i == BbSettings::UpscalerMetalFx ? s.metalfx_supported.load() : true;
#ifdef __APPLE__
            // bbport: FSR 4 needs INT8 dot products and compute derivatives that KosmicKrisp lacks.
            if (BbSettings::IsFsr4(i) && !supported) {
                continue;
            }
#else
            if (i == BbSettings::UpscalerMetalFx) {
                continue; // macOS only
            }
#endif
            ImGui::BeginDisabled(!supported);
            if (ImGui::Selectable(upscalers[i], i == upscaler)) {
                Store(s.upscaler, i, true);
            }
            ImGui::EndDisabled();
            if (!supported) {
                ImGui::SameLine();
                ImGui::TextDisabled("%s", Tr("— не поддерживается видеокартой",
                                             "— unsupported by this GPU"));
            }
        }
#ifndef __APPLE__ // bbport: DLSS needs an NVIDIA GPU and driver, XeSS is not ported to macOS.
        static const char* later[] = {"DLSS", "XeSS"};
        for (const char* name : later) {
            ImGui::BeginDisabled();
            ImGui::Selectable(name, false);
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("%s", Tr("— в работе", "— in progress"));
        }
#endif
        ImGui::EndCombo();
    }
#ifdef __APPLE__
    if (const char* problem = s.metalfx_problem.load()) {
        ImGui::PushTextWrapPos();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f),
                           Tr("MetalFX недоступен: %s", "MetalFX unavailable: %s"), problem);
        ImGui::PopTextWrapPos();
    }
    if (s.upscaler == BbSettings::UpscalerMetalFx) {
        Hint(Tr("MetalFX: временной апскейлер Apple на том же GPU, что и KosmicKrisp. Первая "
                "версия ждёт GPU на каждом кадре, поэтому медленнее, чем могла бы. Native AA "
                "(x1.0) работает, только если GPU поддерживает масштаб 1: иначе выберите другой "
                "пресет.",
                "MetalFX: Apple's temporal upscaler, on the same GPU as KosmicKrisp. This first "
                "version waits for the GPU every frame, so it is slower than it could be. "
                "Native AA (x1.0) works only if the GPU supports a scale of 1; otherwise choose "
                "another preset."));
    }
#endif
    if (const char* problem = s.fsr4_problem.load()) {
        ImGui::PushTextWrapPos();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f),
                           Tr("FSR 4 недоступен: %s", "FSR 4 unavailable: %s"), problem);
        if (!BbSettings::IsFsr4(s.upscaler))
            ImGui::TextUnformatted(Tr("Активен режим, выбранный выше. FSR 4 можно выбрать снова.",
                                      "The mode selected above is active. You can select FSR 4 "
                                      "again."));
        ImGui::PopTextWrapPos();
    }
    if (BbSettings::IsFsr4(s.upscaler)) {
        if (s.upscaler == BbSettings::UpscalerFsr411) {
            Hint(Tr("FSR 4.1.1 в режиме INT8: модель из DLL AMD 4.1.1, воспроизведённая в Vulkan "
                    "(результат совпадает с DLL). Одна модель для Native..Performance и отдельная "
                    "для Ultra Performance. Ассеты: tools/fsr4cap/build_assets.sh (нужны DLL и "
                    "Proton).",
                    "FSR 4.1.1 in INT8 mode: the model from AMD's 4.1.1 DLL, reproduced in "
                    "Vulkan (output matches the DLL). One model for Native..Performance and a "
                    "separate one for Ultra Performance. Assets: tools/fsr4cap/build_assets.sh "
                    "(needs the DLLs and Proton)."));
        } else {
            Hint(Tr("FSR 4 в режиме INT8 (модель v07 из исходников AMD FidelityFX SDK). Качество "
                    "выше, чем у FSR 3.1, но проход тяжелее. Смена пресета пересобирает модель "
                    "(короткая пауза). Ассеты: tools/fetch_fsr4_assets.sh.",
                    "FSR 4 in INT8 mode (model v07 from the AMD FidelityFX SDK sources). Higher "
                    "quality than FSR 3.1, but a heavier pass. Changing the preset rebuilds the "
                    "model (a short pause). Assets: tools/fetch_fsr4_assets.sh."));
        }
        Checkbox(Tr("FSR 4: авто-экспозиция", "FSR 4: auto exposure"), s.fsr4_auto_exposure);
        Checkbox(Tr("FSR 4: обратный знак jitter",
                    "FSR 4: invert jitter sign"), s.fsr4_invert_jitter);
        Hint(Tr("Проверка при гостинге: сеть FSR 4 нормирует цвет по экспозиции и по ней решает, "
                "когда отбросить прошлые кадры. Меняются сразу, без перезапуска.",
                "For checking ghosting: the FSR 4 network normalises colour by exposure and "
                "uses it to decide when to discard past frames. Both apply immediately, no "
                "restart."));
    }
    const bool upscaler_on = s.upscaler != BbSettings::UpscalerOff;
    const bool taa = s.upscaler == BbSettings::UpscalerTaa;
    ImGui::BeginDisabled(!upscaler_on);
    ImGui::BeginDisabled(taa);
    int preset = taa ? BbSettings::NativeAA : s.preset.load();
    char preset_label[64];
    std::snprintf(preset_label, sizeof(preset_label), "%s (x%.3g)", BbSettings::PresetName(preset),
                  BbSettings::PresetScale(preset));
    if (ImGui::BeginCombo(Tr("Пресет", "Preset"), preset_label)) {
        for (const int i : BbSettings::PresetOrder) {
            char label[64];
            const float scale = BbSettings::PresetScale(i);
            const int output = s.output_res;
            std::snprintf(label, sizeof(label), Tr("%s (x%.3g, рендер %dx%d)",
                                                   "%s (x%.3g, render %dx%d)"),
                          BbSettings::PresetName(i), scale,
                          int(std::lround(BbSettings::OutputWidths[output] / scale / 2) * 2),
                          int(std::lround(BbSettings::OutputHeights[output] / scale / 2) * 2));
            if (ImGui::Selectable(label, i == preset)) {
                Store(s.preset, i, true);
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    if (taa) {
        ImGui::TextWrapped("%s", Tr("TAA сглаживает сцену в разрешении вывода, без модели FSR "
                                    "и апскейлинга. Сохранённый пресет FSR восстановится при "
                                    "выборе FSR.",
                                    "TAA anti-aliases the scene at the output resolution, with no "
                                    "FSR model or upscaling. Your saved FSR preset comes back "
                                    "when you select FSR."));
    }
    ImGui::Text(Tr("Активный рендер сцены: %d x %d",
                   "Scene render resolution: %d x %d"), s.active_render_width.load(),
                s.active_render_height.load());
    if (BbSettings::FixedRenderSession()) {
        ImGui::Text(Tr("Пресет при запуске: %s",
                       "Preset at start: %s"), BbSettings::PresetName(s.startup_preset));
        if (const char* automatic = std::getenv("BB_AUTO_RENDER_RES");
            automatic && automatic[0] == '1') {
            Hint(Tr("При выводе не 1080p вся игра рисуется в разрешении пресета (патч при "
                    "запуске): это быстрее всего на Steam Deck и слабых GPU. Смена пресета или "
                    "разрешения вывода — после перезапуска. Пункт «Смена разрешения на лету» ниже "
                    "включает смену без перезапуска (постобработка тогда остаётся в 1080p, "
                    "медленнее).",
                    "When the output is not 1080p, the whole game renders at the preset's "
                    "resolution (patched at start): fastest on the Steam Deck and weaker GPUs. "
                    "Preset or output resolution changes apply after a restart. “Live "
                    "resolution changes” below allows changes without a restart "
                    "(post-processing then stays at 1080p, which is slower)."));
        } else {
            Hint(Tr("BB_RENDER_RES фиксирует размер сцены при запуске. Уберите эту явную "
                    "переменную для смены разрешения и пресетов без перезапуска игры.",
                    "BB_RENDER_RES fixes the scene size at start. Remove this explicit variable "
                    "to change resolution and presets without restarting the game."));
        }
    } else {
        Hint(Tr("Native AA: FSR работает как сглаживание. Остальные пресеты уменьшают разрешение "
                "отрисовки сцены относительно вывода. Интерфейс рисуется в разрешении вывода. "
                "Пресет применяется со следующего кадра без перезапуска игры.",
                "Native AA: FSR works as anti-aliasing. The other presets lower the scene's "
                "render resolution relative to the output. The UI is drawn at the output "
                "resolution. The preset applies from the next frame, without restarting the "
                "game."));
    }
    Checkbox(Tr("Резкость (RCAS)", "Sharpening (RCAS)"), s.sharpen);
    ImGui::BeginDisabled(!s.sharpen);
    Slider(Tr("Сила резкости", "Sharpness"), s.sharpness, 0.0f, 2.0f);
    Hint(Tr("До 1 — резкость самого апскейлера (RCAS). Выше 1 добавляется ещё один проход RCAS. "
            "Ctrl+клик по ползунку — ввести точное значение.",
            "Up to 1: the upscaler's own sharpening (RCAS). Above 1 adds another RCAS pass. "
            "Ctrl+click the slider (Cmd+click on a Mac) to type an exact value."));
    ImGui::EndDisabled();
    Checkbox(Tr("Субпиксельный сдвиг (jitter)", "Sub-pixel jitter"), s.jitter);
    Hint(Tr("Каждый кадр сцена сдвигается на долю пикселя, и апскейлер собирает из нескольких "
            "кадров больше деталей. Без него получается только сглаживание по истории.",
            "Each frame the scene shifts by a fraction of a pixel, so the upscaler gathers more "
            "detail over several frames. Without it you only get anti-aliasing from the "
            "history."));

    ImGui::SeparatorText(Tr("Маска реактивности", "Reactive mask"));
    ImGui::BeginDisabled(taa);
    Checkbox(Tr("Включить маску", "Enable mask"), s.reactive);
    Hint(Tr("Помечает прозрачные эффекты (частицы, дымку), чтобы апскейлер меньше опирался на "
            "прошлые кадры. Меньше шлейфов за эффектами, но под ними возвращается дрожание.",
            "Marks transparent effects (particles, haze) so the upscaler relies less on past "
            "frames. Fewer trails behind effects, but shimmering returns underneath them."));
    ImGui::BeginDisabled(!s.reactive);
    Slider(Tr("Масштаб", "Scale"), s.reactive_scale, 0.0f, 4.0f);
    Slider(Tr("Порог", "Threshold"), s.reactive_threshold, 0.0f, 1.0f);
    Slider(Tr("Максимум", "Maximum"), s.reactive_max, 0.0f, 1.0f);
    bool show_mask = s.debug_view == BbSettings::DebugReactive;
    if (ImGui::Checkbox(Tr("Показать маску (отладка)", "Show mask (debug)"), &show_mask)) {
        s.debug_view = show_mask ? BbSettings::DebugReactive : BbSettings::DebugNone;
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    Checkbox(Tr("Векторы движения персонажей", "Character motion vectors"), s.object_motion);
    Hint(Tr("Точные векторы для анимированных объектов: одежда и оружие меньше рассыпаются "
            "при движении. Статичная сцена не получает дополнительный проход. "
            "Изменение применяется после перезапуска игры.",
            "Exact vectors for animated objects: clothing and weapons break up less in motion. "
            "A static scene gets no extra pass. Applies after restarting the game."));
    bool show_motion = s.debug_view == BbSettings::DebugMotion;
    if (ImGui::Checkbox(Tr("Показать векторы движения (отладка)",
                           "Show motion vectors (debug)"), &show_motion)) {
        s.debug_view = show_motion ? BbSettings::DebugMotion : BbSettings::DebugNone;
    }
    Hint(Tr("Красный/зелёный: движение по горизонтали/вертикали (8 пикселей = полная яркость). "
            "Синий: пиксель получил точный вектор объекта, а не только движение камеры. "
            "Движущийся предмет без синего и без красного/зелёного апскейлер считает "
            "неподвижным, отсюда шлейф.",
            "Red/green: horizontal/vertical motion (8 pixels = full brightness). Blue: the "
            "pixel got an exact object vector, not just camera motion. The upscaler treats a "
            "moving object with neither blue nor red/green as static, hence the trail."));
    ImGui::EndDisabled(); // upscaler off

    ImGui::SeparatorText(Tr("Разрешение вывода", "Output resolution"));
    static const char* outputs[] = {"1280 x 720", "1920 x 1080", "2560 x 1440", "3840 x 2160"};
    int output = s.output_res;
    if (ImGui::BeginCombo(Tr("Разрешение вывода", "Output resolution"), outputs[output])) {
        for (int i = 0; i < BbSettings::OutputCount; ++i) {
            if (ImGui::Selectable(outputs[i], i == output)) {
                Store(s.output_res, i, true);
            }
        }
        ImGui::EndCombo();
    }
    if (BbSettings::FixedRenderSession()) {
        Hint(Tr("Размер готового кадра и интерфейса. Пресет задаёт размер сцены относительно "
                "вывода: 4K Performance = 1920x1080. Применяется после перезапуска игры.",
                "Size of the final frame and the UI. The preset sets the scene size relative to "
                "the output: 4K Performance = 1920x1080. Applies after restarting the game."));
    } else {
        Hint(Tr("Размер готового кадра и интерфейса меняется на границе следующего кадра. "
                "Пресет задаёт размер сцены относительно вывода: 4K Performance = 1920x1080. "
                "Смена размера сбрасывает историю FSR и может вызвать короткую паузу.",
                "The final frame and the UI resize at the next frame boundary. The preset sets "
                "the scene size relative to the output: 4K Performance = 1920x1080. Resizing "
                "resets the FSR history and may cause a short pause."));
    }
    static const char* const live_modes[] = {Tr("Авто (по видеокарте)", "Auto (by GPU)"),
                                             Tr("Выключена (быстрее)", "Off (faster)"),
                                             Tr("Включена", "On")};
    int live = s.live_resolution + 1;
    if (ImGui::BeginCombo(Tr("Смена разрешения на лету",
                             "Live resolution changes"), live_modes[live])) {
        for (int i = 0; i < 3; ++i) {
            if (ImGui::Selectable(live_modes[i], i == live)) {
                Store(s.live_resolution, i - 1, true);
            }
        }
        ImGui::EndCombo();
    }
    Hint(Tr("Включена: разрешение вывода и пресет меняются без перезапуска, но постобработка игры "
            "остаётся в 1080p — на Steam Deck и старых видеокартах это заметно медленнее. "
            "Выключена: всё рисуется в разрешении пресета, смена — через перезапуск. Авто включает "
            "её на мощных дискретных видеокартах. Применяется после перезапуска игры.",
            "On: the output resolution and preset change without a restart, but the game's "
            "post-processing stays at 1080p, which is noticeably slower on the Steam Deck and "
            "older GPUs. Off: everything renders at the preset's resolution; changes need a "
            "restart. Auto turns it on for powerful discrete GPUs. Applies after restarting the "
            "game."));
    ImGui::SeparatorText(Tr("Эффекты игры (после перезапуска)", "Game effects (after restart)"));
    static const char* const lods[] = {Tr("Максимальная (-2)", "Highest (-2)"),
                                       Tr("Как в игре", "As in the game"),
                                       Tr("Ниже (1)", "Lower (1)"),
                                       Tr("Минимальная (2)", "Lowest (2)")};
    static constexpr int lod_values[] = {-2, 0, 1, 2};
    int lod_index = 1;
    for (int i = 0; i < 4; ++i) {
        if (lod_values[i] == s.model_lod) lod_index = i;
    }
    if (ImGui::BeginCombo(Tr("Детализация моделей", "Model detail"), lods[lod_index])) {
        for (int i = 0; i < 4; ++i) {
            if (ImGui::Selectable(lods[i], i == lod_index)) {
                Store(s.model_lod, lod_values[i], true);
            }
        }
        ImGui::EndCombo();
    }
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        Checkbox(Tr(BbSettings::Effects[e].label, BbSettings::Effects[e].label_en), s.effects[e]);
    }
    Hint(Tr("Эффекты включаются и выключаются патчами игры при запуске (patches/Bloodborne.xml). "
            "Размытие в движении и тени от динамических источников заметно нагружают GPU.",
            "Effects are switched on and off by game patches at start (patches/Bloodborne.xml). "
            "Motion blur and dynamic light shadows add a noticeable GPU load."));
    Hint(Tr("Свободная камера: удерживайте Cross и нажимайте L3 (клавиатура: Space + Z). "
            "Debug menu: левый touchpad / Tab. Нужны DbgFont14h.ccm и DbgFont14h.tpf "
            "в dvdroot_ps4/font из мода Nexus #253. Правый touchpad: Backspace.",
            "Free camera: hold Cross and press L3 (keyboard: Space + Z). Debug menu: left "
            "touchpad / Tab. Needs DbgFont14h.ccm and DbgFont14h.tpf in dvdroot_ps4/font from "
            "Nexus mod #253. Right touchpad: Backspace."));

    bool restart = s.object_motion != s.startup_object_motion ||
                   s.model_lod != s.startup_model_lod ||
                   s.live_resolution != s.startup_live_resolution ||
                   BbSettings::ResolutionNeedsRestart();
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        restart |= s.effects[e] != s.startup_effects[e];
    }
    if (restart) {
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "%s",
                           Tr("Изменения применятся после перезапуска игры",
                              "Changes apply after restarting the game"));
        if (ImGui::Button(Tr("Применить и перезапустить игру", "Apply and restart game"))) {
            BbSettings::Save();
            runtime_restart();
        }
    }

    ImGui::SeparatorText(Tr("Прочее", "Other"));
    Checkbox(Tr("Счётчик FPS в углу", "FPS counter in the corner"), s.show_fps);

    ImGui::Spacing();
    if (ImGui::Button(Tr("Закрыть", "Close"))) {
        keep_open = false;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("%s", Tr("Настройки сохраняются в bbport.ini",
                                 "Settings are saved to bbport.ini"));
    ImGui::End();
    if (!keep_open) {
        SetOpen(false);
    }
}

void FpsCounter() {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    const float pad = 12.0f * base_scale;
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkPos.x + viewport->WorkSize.x - pad,
                                   viewport->WorkPos.y + pad),
                            ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.5f);
    ImGui::Begin("##fps", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNav |
                     ImGuiWindowFlags_NoFocusOnAppearing);
    const auto& s = BbSettings::Get();
    ImGui::Text(Tr("%.0f FPS  %.1f мс  %s",
                   "%.0f FPS  %.1f ms  %s"), frame_ms_avg > 0.0f ? 1000.0f / frame_ms_avg : 0.0f,
                frame_ms_avg,
                s.upscaler == BbSettings::UpscalerFsr3   ? "FSR 3.1"
                : s.upscaler == BbSettings::UpscalerFsr4 ? "FSR 4"
                : s.upscaler == BbSettings::UpscalerFsr411 ? "FSR 4.1.1"
                : s.upscaler == BbSettings::UpscalerTaa ? "TAA"
                : s.upscaler == BbSettings::UpscalerMetalFx ? "MetalFX"
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
        // Mac keyboards have no Insert key: F1 (fn+F1 unless F-keys are standard), ` or §.
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
