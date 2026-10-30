// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the overlay menu's sections (upscaler, reactive mask, output, game effects, frame rate
// limit, shaders); Menu() in bbport_overlay.cpp puts them one under the other.
#include "bbport_overlay_internal.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>

#include "bbport_frame_rate.h"
#include "bbport_guest_hooks.h"
#include "imgui.h"
#include "render/compile_log.h"

extern "C" void runtime_restart(void); // bb-probe (probe.c)

namespace BbOverlay::detail {

/// The shaders and pipelines compiled in this session, newest first (BbCompileLog): what the
/// background did and what the game had to wait for. A row copies its hash.
void ShadersSection() {
    static const bool unfolded = bb_env("BB_OVERLAY_OPEN") != nullptr;
    if (!ImGui::CollapsingHeader(
            BbSettings::MenuText("Shaders (diagnostics)", "Шейдеры (диагностика)"),
            unfolded ? ImGuiTreeNodeFlags_DefaultOpen : ImGuiTreeNodeFlags_None)) {
        return;
    }
    std::uint64_t total = 0;
    const auto events = BbCompileLog::Newest(40, total);
    const u32 bg_total = g_bb_background_pipelines_total.load(std::memory_order_relaxed);
    const u32 bg_done = g_bb_background_pipelines_done.load(std::memory_order_relaxed);
    ImGui::Text("%s: %llu", BbSettings::MenuText("Compiled this session", "Скомпилировано за сессию"),
                static_cast<unsigned long long>(total));
    ImGui::Text("%s: %u / %u%s", BbSettings::MenuText("Made in the background", "Собрано в фоне"),
                bg_done, bg_total,
                g_bb_shader_import_active.load(std::memory_order_relaxed)
                    ? BbSettings::MenuText(", recipes importing", ", идёт импорт рецептов")
                    : "");
    if (events.empty()) {
        ImGui::TextDisabled("%s", BbSettings::MenuText("Nothing compiled yet", "Пока ничего"));
        return;
    }
    static constexpr const char* stages[] = {"FS", "VS", "GS", "ES", "HS", "LS", "CS"};
    const auto now = std::chrono::steady_clock::now();
    if (ImGui::BeginTable("##compiles", 4,
                          ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit |
                              ImGuiTableFlags_ScrollY,
                          ImVec2(0.0f, 220.0f * base_scale))) {
        ImGui::TableSetupColumn(BbSettings::MenuText("Ago", "Назад"));
        ImGui::TableSetupColumn(BbSettings::MenuText("What", "Что"));
        ImGui::TableSetupColumn(BbSettings::MenuText("Hash", "Хеш"));
        ImGui::TableSetupColumn("ms");
        ImGui::TableHeadersRow();
        for (const auto& e : events) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%.0f s", std::chrono::duration<double>(now - e.when).count());
            ImGui::TableNextColumn();
            switch (e.kind) {
            case BbCompileLog::Kind::Shader:
                ImGui::Text("%s %s #%u", BbSettings::MenuText("shader", "шейдер"),
                            e.stage < 7 ? stages[e.stage] : "?", unsigned(e.permutation));
                break;
            case BbCompileLog::Kind::Graphics:
                ImGui::TextUnformatted(BbSettings::MenuText("pipeline (waited)", "конвейер (ожидание)"));
                break;
            case BbCompileLog::Kind::Compute:
                ImGui::TextUnformatted(BbSettings::MenuText("compute pipeline", "конвейер compute"));
                break;
            case BbCompileLog::Kind::Background:
                ImGui::TextUnformatted(BbSettings::MenuText("pipeline (background)", "конвейер (фон)"));
                break;
            }
            ImGui::TableNextColumn();
            char hash[24];
            std::snprintf(hash, sizeof(hash), "%#llx", static_cast<unsigned long long>(e.hash));
            ImGui::PushID(&e);
            if (ImGui::Selectable(hash, false)) {
                ImGui::SetClipboardText(hash);
            }
            ImGui::PopID();
            ImGui::TableNextColumn();
            ImGui::Text("%.1f", e.ms);
        }
        ImGui::EndTable();
    }
}

/// Text about a problem (an upscaler that cannot run, a missing file), in orange.
void ProblemText(const char* text) {
        ImGui::PushTextWrapPos();
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.3f, 1.0f), "%s", text);
        ImGui::PopTextWrapPos();
}

/// The menu's "Temporal upscaler" section: the upscaler, its preset, sharpness and model; true
/// when the upscaler is TAA (the reactive mask has nothing to do then).
bool UpscalerSection(BbSettings::Values& s) {
    ImGui::SeparatorText(BbSettings::MenuText("Temporal upscaler", "Временной апскейлер"));
    const char* upscalers[] = {
        BbSettings::MenuText("Off", "Выкл"), "FSR 3.1", "FSR 4 (INT8)", "FSR 4.1.x",
        BbSettings::MenuText("TAA (native anti-aliasing)", "TAA (нативное сглаживание)"),
        "DLSS (NVIDIA RTX)", BbSettings::MenuText("FSR 1 (spatial, weak GPUs)", "FSR 1 (пространственный, слабые GPU)"),
        BbSettings::MenuText("FSR 4 (AMD's DLL)", "FSR 4 (DLL AMD)"), "MetalFX (Apple)"};
    static_assert(sizeof(upscalers) / sizeof(upscalers[0]) == BbSettings::UpscalerCount);
    int upscaler = s.upscaler;
    if (ImGui::BeginCombo(BbSettings::MenuText("Upscaler", "Апскейлер"), upscalers[upscaler])) {
        for (int i = 0; i < BbSettings::UpscalerCount; ++i) {
            const bool supported = i == BbSettings::UpscalerFsr4     ? s.fsr4_supported.load()
                                   : i == BbSettings::UpscalerFsr411 ? s.fsr411_supported.load()
                                   : i == BbSettings::UpscalerDlss   ? s.dlss_supported.load()
                                   : i == BbSettings::UpscalerFsrAmd ? s.fsr_amd_supported.load()
                                   : i == BbSettings::UpscalerMetalFx ? s.metalfx_supported.load()
                                                                     : true;
#ifdef __APPLE__
            // bbport: FSR 4 needs INT8 dot products and compute derivatives that KosmicKrisp
            // lacks (AMD's DLLs are Windows'); DLSS needs an NVIDIA GPU.
            if ((BbSettings::IsFsr4(i) || i == BbSettings::UpscalerDlss) && !supported) {
                continue;
            }
#else
            if (i == BbSettings::UpscalerMetalFx) {
                continue; // bbport: macOS only
            }
#endif
            ImGui::BeginDisabled(!supported);
            if (ImGui::Selectable(upscalers[i], i == upscaler)) {
                s.upscaler_fallback_from = -1; // a choice (FSR 3.1 too): no stand-in to keep
                ++s.upscaler_chosen;
                Store(s.upscaler, i, true);
            }
            ImGui::EndDisabled();
            if (!supported && i == BbSettings::UpscalerFsrAmd) {
                ImGui::SameLine();
                ImGui::TextDisabled("%s", BbSettings::MenuText("— AMD's DLLs not found",
                                                         "— нет DLL AMD"));
            } else if (!supported) {
                ImGui::SameLine();
                ImGui::TextDisabled("%s", BbSettings::MenuText("— not supported by this GPU",
                                                         "— не поддерживается видеокартой"));
            }
        }
#ifndef __APPLE__ // bbport: XeSS is not ported to macOS.
        static const char* later[] = {"XeSS"};
        for (const char* name : later) {
            ImGui::BeginDisabled();
            ImGui::Selectable(name, false);
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::TextDisabled("%s", BbSettings::MenuText("— in development", "— в работе"));
        }
#endif
        ImGui::EndCombo();
    }
    if (s.upscaler == BbSettings::UpscalerFsr411 || s.upscaler == BbSettings::UpscalerFsrAmd) {
        // The FSR 4.1 model: those whose asset sets are built (tools/fsr4cap: the 4.1.1 DLL, AMD's
        // driver DLL for the 4.1.2 ones) and the GPU runs, or those AMD's DLLs hold (fsr_amd: FP8
        // needs FP8 matrices); switched on the fly (FSR 3.1 while the new one is built).
        const bool amd = s.upscaler == BbSettings::UpscalerFsrAmd;
        const char* models[BbSettings::Fsr4ModelCount] = {
            amd ? BbSettings::MenuText("Auto (the DLL's choice)", "Авто (выбор DLL)")
                : BbSettings::MenuText("Auto (by the GPU)", "Авто (по видеокарте)"),
            "FP8", "INT8", "INT8 RDNA2", "INT8 Lite"};
        const unsigned available = amd ? s.fsr_amd_models.load() : s.fsr4_models_available.load();
        const int model = std::clamp(s.fsr4_model.load(), 0, BbSettings::Fsr4ModelCount - 1);
        if (ImGui::BeginCombo(BbSettings::MenuText("FSR 4 model", "Модель FSR 4"), models[model])) {
            for (int i = 0; i < BbSettings::Fsr4ModelCount; ++i) {
                const bool usable = i == BbSettings::Fsr4Auto || (available >> i & 1u);
                ImGui::BeginDisabled(!usable);
                if (ImGui::Selectable(models[i], i == model)) {
                    Store(s.fsr4_model, i, true);
                }
                ImGui::EndDisabled();
                if (!usable) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", amd ? BbSettings::MenuText("— not on this GPU",
                                                                         "— не на этой видеокарте")
                                                  : BbSettings::MenuText("— not built (launcher)",
                                                                         "— не собрана (лаунчер)"));
                }
            }
            ImGui::EndCombo();
        }
        if (const char* active = s.fsr4_model_active.load(); active && !amd) {
            ImGui::SameLine();
            ImGui::TextDisabled("%s", active);
        }
        if (amd) {
            Hint(BbSettings::MenuText(
                "Which of the four FSR 4 models in AMD's DLLs runs: Auto is the one the DLL picks "
                "for this GPU. FP8: RDNA4 only. INT8: RDNA3's. INT8 RDNA2: made for RX 6000. INT8 "
                "Lite: the light one (Steam Deck, APUs), faster and softer.",
                "Какая из четырёх моделей FSR 4 в DLL AMD работает: «Авто» — та, что DLL выбирает "
                "для этой видеокарты. FP8: только RDNA4. INT8: модель RDNA3. INT8 RDNA2: для "
                "RX 6000. INT8 Lite: лёгкая (Steam Deck, APU), быстрее и мягче."));
        } else Hint(BbSettings::MenuText(
            "FP8: RDNA4's model (emulated, slower, on RDNA3). INT8: FSR 4.1 on any GPU. INT8 RDNA2: "
            "FSR 4.1.2's model made for RX 6000. INT8 Lite: FSR 4.1.2's light model (Steam Deck, "
            "APUs): faster, softer. The 4.1.2 models are built from AMD's driver DLL in the "
            "launcher.",
            "FP8: модель RDNA4 (на RDNA3 эмулируется, медленнее). INT8: FSR 4.1 на любой видеокарте. "
            "INT8 RDNA2: модель FSR 4.1.2 для RX 6000. INT8 Lite: лёгкая модель FSR 4.1.2 (Steam "
            "Deck, APU): быстрее, мягче. Модели 4.1.2 собираются в лаунчере из DLL драйвера AMD."));
    }
    if (s.upscaler == BbSettings::UpscalerDlss) {
        Hint(BbSettings::MenuText(
            "NVIDIA DLSS Super Resolution (RTX GPUs; Native AA is DLAA). Needs the DLSS bridge "
            "and NVIDIA's library next to the game; otherwise FSR 3.1 is used.",
            "NVIDIA DLSS Super Resolution (видеокарты RTX; Native AA — это DLAA). Нужны мост DLSS "
            "и библиотека NVIDIA рядом с игрой; без них используется FSR 3.1."));
    }
#ifdef __APPLE__
    if (const char* problem = s.metalfx_problem.load()) {
        char line[192];
        std::snprintf(line, sizeof(line),
                      BbSettings::MenuText("MetalFX unavailable: %s", "MetalFX недоступен: %s"),
                      problem);
        ProblemText(line);
    }
    if (s.upscaler == BbSettings::UpscalerMetalFx) {
        Hint(BbSettings::MenuText(
            "MetalFX: Apple's temporal upscaler, on the same GPU as KosmicKrisp. Slower than "
            "FSR 3.1 here (each frame goes from Vulkan to Metal and back). Native AA (x1.0) works "
            "only if the GPU supports a scale of 1; otherwise choose another preset.",
            "MetalFX: временной апскейлер Apple на том же GPU, что и KosmicKrisp. Здесь медленнее "
            "FSR 3.1 (кадр переходит из Vulkan в Metal и обратно). Native AA (x1.0) работает, "
            "только если GPU поддерживает масштаб 1: иначе выберите другой пресет."));
    }
#endif
    if (const char* problem = s.dlss_problem.load(); problem && s.upscaler == BbSettings::UpscalerDlss) {
        char line[192];
        std::snprintf(line, sizeof(line), "DLSS: %s", problem);
        ProblemText(line);
    }
    if (const char* problem = s.fsr4_problem.load()) {
        char line[192];
        std::snprintf(line, sizeof(line),
                      BbSettings::MenuText("FSR 4 unavailable: %s", "FSR 4 недоступен: %s"), problem);
        ProblemText(line);
        if (!BbSettings::IsFsr4(s.upscaler)) {
            ImGui::PushTextWrapPos();
            ImGui::TextUnformatted(BbSettings::MenuText(
                "The mode selected above is active. You can select FSR 4 again.",
                "Активен режим, выбранный выше. FSR 4 можно выбрать снова."));
            ImGui::PopTextWrapPos();
        }
    }
    if (BbSettings::IsFsr4(s.upscaler)) {
        if (s.upscaler == BbSettings::UpscalerFsrAmd) {
            if (const char* provider = s.fsr_amd_provider.load()) {
                ImGui::TextDisabled("%s", provider);
            }
            Hint(BbSettings::MenuText(
                "AMD's FSR SDK 2.4 DLLs for Vulkan (amd_fidelityfx_loader_vk.dll and "
                "amd_fidelityfx_upscaler_vk.dll next to the game or in BB_FFX_DIR). They choose the "
                "model themselves (from the AMD driver too); on other GPUs it may be FSR 3.1.",
                "DLL AMD FSR SDK 2.4 для Vulkan (amd_fidelityfx_loader_vk.dll и "
                "amd_fidelityfx_upscaler_vk.dll рядом с игрой или в BB_FFX_DIR). Модель они "
                "выбирают сами (в том числе из драйвера AMD); на других видеокартах это может "
                "быть FSR 3.1."));
        } else if (s.upscaler == BbSettings::UpscalerFsr411) {
            Hint(BbSettings::MenuText(
                "FSR 4.1.1: the model from AMD's 4.1.1 DLL, reproduced in Vulkan (output matches "
                "the DLL). Assets are built from your DLL in the launcher.",
                "FSR 4.1.1: модель из DLL AMD 4.1.1, воспроизведённая в Vulkan (результат совпадает "
                "с DLL). Ассеты собираются из вашей DLL в лаунчере."));
        } else {
            Hint(BbSettings::MenuText(
                "FSR 4 in INT8 mode (v07 model from AMD FidelityFX SDK sources). Higher quality "
                "than FSR 3.1, but the pass is more demanding. Changing the preset rebuilds the "
                "model (a brief pause).",
                "FSR 4 в режиме INT8 (модель v07 из исходников AMD FidelityFX SDK). Качество выше, "
                "чем у FSR 3.1, но проход тяжелее. Смена пресета пересобирает модель (короткая "
                "пауза)."));
        }
        Checkbox(BbSettings::MenuText("FSR 4: auto exposure", "FSR 4: авто-экспозиция"),
                 s.fsr4_auto_exposure);
        Checkbox(BbSettings::MenuText("FSR 4: invert jitter sign", "FSR 4: обратный знак jitter"),
                 s.fsr4_invert_jitter);
        Hint(BbSettings::MenuText(
            "For diagnosing ghosting: the FSR 4 network normalizes color by exposure and uses "
            "it to decide when to discard previous frames. Changes apply immediately, without a "
            "restart.",
            "Проверка при гостинге: сеть FSR 4 нормирует цвет по экспозиции и по ней решает, "
            "когда отбросить прошлые кадры. Меняются сразу, без перезапуска."));
    }
    const bool upscaler_on = s.upscaler != BbSettings::UpscalerOff;
    const bool taa = s.upscaler == BbSettings::UpscalerTaa;
    ImGui::BeginDisabled(!upscaler_on);
    ImGui::BeginDisabled(taa);
    int preset = taa ? BbSettings::NativeAA : s.preset.load();
    char preset_label[64];
    std::snprintf(preset_label, sizeof(preset_label), "%s (x%.3g)", BbSettings::PresetName(preset),
                  BbSettings::PresetScale(preset));
    if (ImGui::BeginCombo(BbSettings::MenuText("Preset", "Пресет"), preset_label)) {
        for (const int i : BbSettings::PresetOrder) {
            char label[64];
            const float scale = BbSettings::PresetScale(i);
            const auto output = BbSettings::CurrentOutput();
            std::snprintf(
                label, sizeof(label),
                // bbport (macOS port): %.3g shows Ultra Quality's x1.25 (x%.1f made it x1.2).
                BbSettings::MenuText("%s (x%.3g, render %dx%d)", "%s (x%.3g, рендер %dx%d)"),
                BbSettings::PresetName(i), scale, int(std::lround(output.width / scale / 2) * 2),
                int(std::lround(output.height / scale / 2) * 2));
            if (ImGui::Selectable(label, i == preset)) {
                Store(s.preset, i, true);
            }
        }
        ImGui::EndCombo();
    }
    ImGui::EndDisabled();
    if (taa) {
        ImGui::TextWrapped("%s", BbSettings::MenuText(
            "TAA anti-aliases the scene at output resolution, without an FSR model or upscaling. "
            "Your saved FSR preset is restored when you select FSR.",
            "TAA сглаживает сцену в разрешении вывода, без модели FSR и апскейлинга. "
            "Сохранённый пресет FSR восстановится при выборе FSR."));
    }
    ImGui::Text(
        BbSettings::MenuText("Active scene render: %d x %d", "Активный рендер сцены: %d x %d"),
        s.active_render_width.load(), s.active_render_height.load());
    if (BbSettings::HybridSession()) {
        ImGui::Text(BbSettings::MenuText("Startup preset: %s", "Пресет при запуске: %s"),
                    BbSettings::PresetName(s.startup_preset));
        Hint(BbSettings::MenuText(
            "The game renders at the startup preset's size as if it were its own resolution: "
            "nothing extra a frame. Another preset applies at once (drawn in stand-in targets of "
            "its size). From the next start the chosen preset is the game's own size.",
            "Игра рисует в размере пресета запуска как в собственном разрешении: ничего лишнего "
            "за кадр. Другой пресет применяется сразу (рисуется в подменных целях своего "
            "размера). Со следующего запуска выбранный пресет станет собственным размером игры."));
    } else if (BbSettings::FixedRenderSession()) {
        ImGui::Text(BbSettings::MenuText("Startup preset: %s", "Пресет при запуске: %s"),
                    BbSettings::PresetName(s.startup_preset));
        if (bb_env_on("BB_AUTO_RENDER_RES")) {
            Hint(BbSettings::MenuText(
                "When output is not 1080p, the entire game renders at the preset resolution "
                "(startup patch): this is fastest on Steam Deck and weaker GPUs. Preset or output "
                "resolution changes require a restart. Enable Live resolution changes below to "
                "change them without restarting (post-processing then stays at 1080p, which is "
                "slower).",
                "При выводе не 1080p вся игра рисуется в разрешении пресета (патч при запуске): "
                "это быстрее всего на Steam Deck и слабых GPU. Смена пресета или разрешения "
                "вывода — после перезапуска. Пункт «Смена разрешения на лету» ниже включает "
                "смену без перезапуска (постобработка тогда остаётся в 1080p, медленнее)."));
        } else {
            Hint(BbSettings::MenuText(
                "BB_RENDER_RES fixes the scene size at startup. Remove this explicit environment "
                "variable to change resolution and presets without restarting the game.",
                "BB_RENDER_RES фиксирует размер сцены при запуске. Уберите эту явную переменную "
                "для смены разрешения и пресетов без перезапуска игры."));
        }
    } else {
        Hint(BbSettings::MenuText(
            "Native AA: the upscaler acts as anti-aliasing. Other presets reduce the scene render "
            "resolution relative to the output. The UI renders at output resolution. "
            "The preset applies from the next frame without restarting the game.",
            "Native AA: апскейлер работает как сглаживание. Остальные пресеты уменьшают разрешение "
            "отрисовки сцены относительно вывода. Интерфейс рисуется в разрешении вывода. "
            "Пресет применяется со следующего кадра без перезапуска игры."));
    }
    // Sharpening works without an upscaler too (over the game's AA or FXAA: the presenter's
    // encode pass), so it stays enabled when the upscaler is off.
    ImGui::EndDisabled(); // upscaler off
    Checkbox(BbSettings::MenuText("Sharpening (RCAS)", "Резкость (RCAS)"), s.sharpen);
    ImGui::BeginDisabled(!s.sharpen);
    Slider(BbSettings::MenuText("Sharpness", "Сила резкости"), s.sharpness, 0.0f, 2.0f);
    Hint(BbSettings::MenuText(
        "With an upscaler: up to 1 its own sharpening (RCAS), above 1 another RCAS pass. Without "
        "one: RCAS on the finished frame (over the game's AA or FXAA), up to 1 AMD's strongest. "
#ifdef __APPLE__ // bbport: ImGui makes Cmd its Ctrl on macOS (HandleEvent).
        "Cmd+click the slider to enter an exact value.",
#else
        "Ctrl+click the slider to enter an exact value.",
#endif
        "С апскейлером: до 1 — его собственная резкость (RCAS), выше 1 — ещё один проход RCAS. "
        "Без апскейлера: RCAS по готовому кадру (поверх сглаживания игры или FXAA), 1 — "
        "максимум AMD. Ctrl+клик по ползунку — ввести точное значение."));
    ImGui::EndDisabled();
    ImGui::BeginDisabled(!upscaler_on);
    // Frame generation: FSR 3.1 on any GPU (experimental); DLSS Frame Generation where it can run
    // (GeForce RTX 40 and newer with NVIDIA's library), and on any RTX GPU the entry says why not.
    {
        const int current = std::clamp(s.frame_generation.load(), 0, BbSettings::FrameGenCount - 1);
        const bool fsr3_shown = true;
        const bool dlss_shown = s.dlss_supported.load() || current == BbSettings::FrameGenDlss;
        const int dlss_max = s.dlssg_max_multiplier.load(std::memory_order_relaxed);
        const bool dlss_available = dlss_max > 1;
        const bool shown = fsr3_shown || dlss_available || s.dlss_supported.load() ||
                           current != BbSettings::FrameGenOff;
        if (shown) {
            const char* names[BbSettings::FrameGenCount] = {
                BbSettings::MenuText("Off", "Выкл"),
                BbSettings::MenuText("FSR 3.1 (experimental)", "FSR 3.1 (экспериментально)"),
                "DLSS (NVIDIA RTX 40/50)",
                BbSettings::MenuText("FSR 4 (AMD's DLL)", "FSR 4 (DLL AMD)")};
            const bool amd_available = s.fg_amd_supported.load(std::memory_order_relaxed);
            if (ImGui::BeginCombo(BbSettings::MenuText("Frame generation", "Генерация кадров"),
                                  names[current])) {
                for (int i = 0; i < BbSettings::FrameGenCount; ++i) {
                    if ((i == BbSettings::FrameGenFsr3 && !fsr3_shown) ||
                        (i == BbSettings::FrameGenDlss && !dlss_shown) ||
                        (i == BbSettings::FrameGenFsrAmd && !amd_available && i != current)) {
                        continue;
                    }
                    const bool supported =
                        i == BbSettings::FrameGenOff ||
                        (i == BbSettings::FrameGenFsr3
                             ? s.frame_generation_supported.load(std::memory_order_relaxed)
                         : i == BbSettings::FrameGenFsrAmd ? amd_available
                                                           : dlss_available);
                    ImGui::BeginDisabled(!supported);
                    if (ImGui::Selectable(names[i], i == current)) {
                        Store(s.frame_generation, i, true);
                    }
                    ImGui::EndDisabled();
                    if (!supported) {
                        ImGui::SameLine();
                        ImGui::TextDisabled("%s", BbSettings::MenuText("— not available here",
                                                                 "— здесь недоступно"));
                    }
                }
                ImGui::EndCombo();
            }
            Hint(BbSettings::MenuText(
                "Frames generated between each two of the game's: twice the frames on screen "
                "(DLSS on GeForce RTX 50: up to four times), a little more input lag. DLSS Frame "
                "Generation needs the DLSS bridge, NVIDIA's libnvidia-ngx-dlssg.so next to the "
                "game or in the launcher's DLSS folder and a GeForce RTX 40 or newer. Works with "
                "every upscaler preset except Off; menus and movies are not generated.",
                "Между каждыми двумя кадрами игры показываются сгенерированные: на экране вдвое "
                "больше кадров (DLSS на GeForce RTX 50 — до четырёх раз), задержка ввода чуть "
                "выше. DLSS Frame Generation нужны мост DLSS, libnvidia-ngx-dlssg.so от NVIDIA "
                "рядом с игрой или в папке DLSS лаунчера и GeForce RTX 40 или новее. Работает со "
                "всеми пресетами апскейлера, кроме «Выкл»; меню и ролики не генерируются."));
            if (current == BbSettings::FrameGenDlss && dlss_max > 2) {
                int multiplier = std::clamp(s.frame_generation_multiplier.load(), 2, dlss_max);
                const bool changed = ImGui::SliderInt(
                    BbSettings::MenuText("Multiplier", "Множитель"), &multiplier, 2, dlss_max, "%dx");
                Store(s.frame_generation_multiplier, multiplier, changed);
                Hint(BbSettings::MenuText(
                    "Frames on screen per game frame: 2x one generated, 4x three. The display "
                    "needs that refresh rate for them to be seen.",
                    "Кадров на экране на один кадр игры: 2x — один сгенерированный, 4x — три. "
                    "Чтобы их было видно, нужна такая частота обновления дисплея."));
            }
            const char* problem = current == BbSettings::FrameGenDlss
                                      ? s.dlssg_problem.load(std::memory_order_relaxed)
                                  : current == BbSettings::FrameGenFsr3
                                      ? s.frame_generation_problem.load(std::memory_order_relaxed)
                                  : current == BbSettings::FrameGenFsrAmd
                                      ? s.fg_amd_problem.load(std::memory_order_relaxed)
                                      : nullptr;
            if (current == BbSettings::FrameGenFsrAmd) {
                if (const char* provider = s.fg_amd_provider.load(std::memory_order_relaxed)) {
                    ImGui::TextDisabled("%s", provider);
                }
            }
            if (problem) {
                char line[256];
                std::snprintf(line, sizeof(line), BbSettings::MenuText("Frame generation: %s",
                                                                       "Генерация кадров: %s"),
                              problem);
                ProblemText(line);
            }
            // The generation runs on the upscaler's scaled path (TemporalUpscaler::RunScaled).
            if (current != BbSettings::FrameGenOff) {
                const int upscaler = s.upscaler.load(std::memory_order_relaxed);
                if (upscaler == BbSettings::UpscalerOff || upscaler == BbSettings::UpscalerTaa) {
                    ProblemText(BbSettings::MenuText(
                        "Frame generation needs an upscaler: FSR or DLSS (Native AA too)",
                        "Генерации кадров нужен апскейлер: FSR или DLSS (подойдёт и «Натив AA»)"));
                } else if (s.frame_generation_restart.load(std::memory_order_relaxed)) {
                    ProblemText(BbSettings::MenuText(
                        "Frame generation starts after a restart of the game",
                        "Генерация кадров включится после перезапуска игры"));
                }
            }
        }
    }
    Checkbox(BbSettings::MenuText("Subpixel jitter", "Субпиксельный сдвиг (jitter)"), s.jitter);
    Hint(BbSettings::MenuText(
        "Each frame shifts the scene by a fraction of a pixel, letting the upscaler reconstruct "
        "more detail from multiple frames. Without it, only history-based anti-aliasing remains.",
        "Каждый кадр сцена сдвигается на долю пикселя, и апскейлер собирает из нескольких "
        "кадров больше деталей. Без него получается только сглаживание по истории."));
    return taa;
}

/// The menu's "Reactive mask" section.
void ReactiveMaskSection(BbSettings::Values& s, bool taa) {
    ImGui::SeparatorText(BbSettings::MenuText("Reactive mask", "Маска реактивности"));
    ImGui::BeginDisabled(taa);
    Checkbox(BbSettings::MenuText("Enable mask", "Включить маску"), s.reactive);
    Hint(BbSettings::IsFsr4(s.upscaler)
             ? BbSettings::MenuText(
                   "Marks transparent effects (haze, light, water, particles). FSR 4 takes no "
                   "mask itself: its output is blended with the current frame there, so the "
                   "effects do not drag previous frames along.",
                   "Помечает прозрачные эффекты (дымку, свет, воду, частицы). FSR 4 сам маску не "
                   "принимает: там его результат смешивается с текущим кадром, и эффекты не "
                   "тянут за собой прошлые кадры.")
             : BbSettings::MenuText(
                   "Marks transparent effects (particles, haze) so the upscaler relies less on "
                   "previous frames. Reduces trails behind effects, but shimmering returns "
                   "underneath them.",
                   "Помечает прозрачные эффекты (частицы, дымку), чтобы апскейлер меньше "
                   "опирался на прошлые кадры. Меньше шлейфов за эффектами, но под ними "
                   "возвращается дрожание."));
    ImGui::BeginDisabled(!s.reactive);
    Slider(BbSettings::MenuText("Scale", "Масштаб"), s.reactive_scale, 0.0f, 4.0f);
    Slider(BbSettings::MenuText("Threshold", "Порог"), s.reactive_threshold, 0.0f, 1.0f);
    Slider(BbSettings::MenuText("Maximum", "Максимум"), s.reactive_max, 0.0f, 1.0f);
    bool show_mask = s.debug_view == BbSettings::DebugReactive;
    if (ImGui::Checkbox(BbSettings::MenuText("Show mask (debug)", "Показать маску (отладка)"),
                        &show_mask)) {
        s.debug_view = show_mask ? BbSettings::DebugReactive : BbSettings::DebugNone;
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    Checkbox(BbSettings::MenuText("Character motion vectors", "Векторы движения персонажей"),
             s.object_motion);
    Hint(BbSettings::MenuText(
        "Accurate vectors for animated objects: clothing and weapons break up less "
        "during movement. The static scene does not get an additional pass. "
        "Changes apply after restarting the game.",
        "Точные векторы для анимированных объектов: одежда и оружие меньше рассыпаются "
        "при движении. Статичная сцена не получает дополнительный проход. "
        "Изменение применяется после перезапуска игры."));
    bool show_motion = s.debug_view == BbSettings::DebugMotion;
    if (ImGui::Checkbox(BbSettings::MenuText("Show motion vectors (debug)",
                                             "Показать векторы движения (отладка)"),
                        &show_motion)) {
        s.debug_view = show_motion ? BbSettings::DebugMotion : BbSettings::DebugNone;
    }
    Hint(BbSettings::MenuText(
        "Red/green: horizontal/vertical motion (8 pixels = full brightness). "
        "Blue: the pixel received an accurate object vector, not just camera motion. "
        "The upscaler treats a moving object with no blue or red/green as "
        "stationary, causing trails.",
        "Красный/зелёный: движение по горизонтали/вертикали (8 пикселей = полная яркость). "
        "Синий: пиксель получил точный вектор объекта, а не только движение камеры. "
        "Движущийся предмет без синего и без красного/зелёного апскейлер считает "
        "неподвижным, отсюда шлейф."));
    ImGui::EndDisabled(); // upscaler off
}

/// The menu's "Output" section: aspect ratio, output resolution, display.
void OutputSection(BbSettings::Values& s) {
    ImGui::SeparatorText(BbSettings::MenuText("Output", "Вывод"));
    // The aspect ratio first; its resolutions follow (another one keeps the height).
    const auto aspect_label = [](int a) {
        if (a != BbSettings::AspectAuto) return std::string(BbSettings::AspectKey(a));
        return std::string(BbSettings::MenuText("Auto", "Авто")) + " (" +
               BbSettings::AspectKey(BbSettings::ResolvedAspect(a)) + ")";
    };
    const int aspect = s.aspect;
    if (ImGui::BeginCombo(BbSettings::MenuText("Aspect ratio", "Соотношение сторон"),
                          aspect_label(aspect).c_str())) {
        for (int a = 0; a < BbSettings::AspectCount; ++a) {
            if (ImGui::Selectable(aspect_label(a).c_str(), a == aspect)) {
                Store(s.output_res, BbSettings::PackSize(BbSettings::NearestOutput(a, BbSettings::CurrentOutput())),
                      true);
                Store(s.aspect, a, true);
            }
        }
        ImGui::EndCombo();
    }
    const auto size_label = [](BbSettings::OutputSize o) {
        return std::to_string(o.width) + " x " + std::to_string(o.height);
    };
    const auto current = BbSettings::CurrentOutput();
    const bool output_auto = s.output_auto.load();
    const std::string auto_label =
        std::string(BbSettings::MenuText("Auto (monitor)", "Авто (монитор)")) + " " + size_label(current);
    if (ImGui::BeginCombo(BbSettings::MenuText("Resolution", "Разрешение"),
                          output_auto ? auto_label.c_str() : size_label(current).c_str())) {
        if (ImGui::Selectable(auto_label.c_str(), output_auto)) {
            Store(s.output_auto, true, true);
        }
        for (const auto& o : BbSettings::Outputs(s.aspect)) {
            if (ImGui::Selectable(size_label(o).c_str(), !output_auto && o == current)) {
                Store(s.output_res, BbSettings::PackSize(o), true);
                Store(s.output_auto, false, true);
            }
        }
        ImGui::EndCombo();
    }
    if (BbSettings::HybridSession()) {
        Hint(BbSettings::MenuText(
            "Size of the final frame and UI; applies at once. The preset sets the scene size "
            "relative to the output: 4K Performance = 1920x1080. Another aspect ratio widens the "
            "view (the UI keeps to the screen edges).",
            "Размер готового кадра и интерфейса, меняется сразу. Пресет задаёт размер сцены "
            "относительно вывода: 4K Performance = 1920x1080. Другое соотношение сторон "
            "расширяет обзор (интерфейс у краёв экрана)."));
    } else if (BbSettings::FixedRenderSession()) {
        Hint(BbSettings::MenuText(
            "Size of the final frame and UI. The preset sets the scene size relative to "
            "the output: 4K Performance = 1920x1080. Another aspect ratio widens the view (the "
            "UI keeps to the screen edges). Applies after restarting the game.",
            "Размер готового кадра и интерфейса. Пресет задаёт размер сцены относительно "
            "вывода: 4K Performance = 1920x1080. Другое соотношение сторон расширяет обзор "
            "(интерфейс у краёв экрана). Применяется после перезапуска игры."));
    } else {
        Hint(BbSettings::MenuText(
            "The final frame and UI size changes at the next frame boundary. "
            "The preset sets the scene size relative to the output: 4K Performance = 1920x1080. "
            "Another aspect ratio widens the view (the UI keeps to the screen edges). "
            "Changing the size resets the upscaler's history and may cause a brief pause.",
            "Размер готового кадра и интерфейса меняется на границе следующего кадра. "
            "Пресет задаёт размер сцены относительно вывода: 4K Performance = 1920x1080. "
            "Другое соотношение сторон расширяет обзор (интерфейс у краёв экрана). "
            "Смена размера сбрасывает историю апскейлера и может вызвать короткую паузу."));
    }
    // Low latency (NVIDIA Reflex through VK_NV_low_latency2, AMD Anti-Lag 2 through
    // VK_AMD_anti_lag): shown where the driver has one, and on any NVIDIA GPU so that the entry
    // says when the driver is too old.
    if (s.low_latency_supported.load(std::memory_order_relaxed) || s.dlss_supported.load()) {
        const char* latency_modes[BbSettings::LowLatencyCount] = {
            BbSettings::MenuText("Off", "Выкл"), BbSettings::MenuText("On", "Вкл"),
            BbSettings::MenuText("On + Boost", "Вкл + Boost")};
        const int latency = std::clamp(s.low_latency.load(), 0, BbSettings::LowLatencyCount - 1);
        const bool supported = s.low_latency_supported.load(std::memory_order_relaxed);
        const bool anti_lag = s.low_latency_backend.load(std::memory_order_relaxed) == 2;
        // With DLSS Frame Generation at least On (BbSettings::EffectiveLowLatency): Off is shown
        // as what runs and cannot be chosen.
        const int effective = std::clamp(BbSettings::EffectiveLowLatency(), 0, BbSettings::LowLatencyCount - 1);
        const bool forced = effective != latency;
        ImGui::BeginDisabled(!supported);
        if (ImGui::BeginCombo(anti_lag ? BbSettings::MenuText("Low latency (AMD Anti-Lag 2)",
                                                              "Низкая задержка (AMD Anti-Lag 2)")
                                       : BbSettings::MenuText("Low latency (NVIDIA Reflex)",
                                                              "Низкая задержка (NVIDIA Reflex)"),
                              latency_modes[effective])) {
            for (int i = 0; i < BbSettings::LowLatencyCount; ++i) {
                const bool allowed = i != BbSettings::LowLatencyOff || !forced;
                ImGui::BeginDisabled(!allowed);
                if (ImGui::Selectable(latency_modes[i], i == effective)) {
                    Store(s.low_latency, i, true);
                }
                ImGui::EndDisabled();
                if (!allowed) {
                    ImGui::SameLine();
                    ImGui::TextDisabled("%s", BbSettings::MenuText("— on with DLSS frame generation",
                                                             "— вкл с генерацией кадров DLSS"));
                }
            }
            ImGui::EndCombo();
        }
        ImGui::EndDisabled();
        Hint(anti_lag ? BbSettings::MenuText(
            "AMD Anti-Lag 2 (VK_AMD_anti_lag; on Linux Mesa's anti-lag layer): the game's thread "
            "waits before each frame for as long as the GPU still has work queued, so that frames "
            "do not queue up in front of it; less input lag, a little less throughput when "
            "GPU-bound. Boost is Reflex's: here the same as On.",
            "AMD Anti-Lag 2 (VK_AMD_anti_lag; в Linux — слой anti-lag из Mesa): поток игры ждёт "
            "перед каждым кадром, пока у видеокарты ещё есть работа в очереди, чтобы кадры не "
            "копились перед ней; меньше задержка ввода, чуть меньше пропускная способность при "
            "упоре в GPU. Boost — режим Reflex: здесь то же, что «Вкл».")
                      : BbSettings::MenuText(
            "NVIDIA Reflex through the driver's Vulkan low-latency mode: the game's thread waits "
            "for the driver before each frame so that frames do not queue up in front of the GPU; "
            "less input lag, a little less throughput when GPU-bound. Boost keeps the GPU clocks "
            "up. Recommended with frame generation. Needs NVIDIA's driver 550 or newer; off "
            "elsewhere.",
            "NVIDIA Reflex через режим низкой задержки драйвера (Vulkan): поток игры ждёт "
            "драйвер перед каждым кадром, чтобы кадры не копились перед видеокартой; меньше "
            "задержка ввода, чуть меньше пропускная способность при упоре в GPU. Boost держит "
            "частоты GPU. Рекомендуется с генерацией кадров. Нужен драйвер NVIDIA 550 или новее; "
            "иначе выключено."));
        if (!supported) {
            ProblemText(BbSettings::MenuText(
                "Low latency: the driver has no VK_NV_low_latency2 (NVIDIA 550 or newer)",
                "Низкая задержка: в драйвере нет VK_NV_low_latency2 (NVIDIA 550 или новее)"));
        }
    }
}

/// The menu's "Game effects" section.
void GameEffectsSection(BbSettings::Values& s) {
    ImGui::SeparatorText(BbSettings::MenuText("Game effects",
                                              "Эффекты игры (после перезапуска)"));
    const char* lods[] = {BbSettings::MenuText("Highest (-2)", "Максимальная (-2)"),
                          BbSettings::MenuText("Game default", "Как в игре"),
                          BbSettings::MenuText("Lower (1)", "Ниже (1)"),
                          BbSettings::MenuText("Lowest (2)", "Минимальная (2)")};
    static constexpr int lod_values[] = {-2, 0, 1, 2};
    int lod_index = 1;
    for (int i = 0; i < 4; ++i) {
        if (lod_values[i] == s.model_lod) lod_index = i;
    }
    if (ImGui::BeginCombo(BbSettings::MenuText("Model detail", "Детализация моделей"),
                          lods[lod_index])) {
        for (int i = 0; i < 4; ++i) {
            if (ImGui::Selectable(lods[i], i == lod_index)) {
                Store(s.model_lod, lod_values[i], true);
            }
        }
        ImGui::EndCombo();
    }
    // Applied each frame (BbEngineFixes): no restart.
    Checkbox(BbSettings::MenuText("Coarse light grid", "Крупная сетка освещения"),
             s.light_grid_opt);
    // Anti-aliasing: one choice instead of the game's AA switch (BbSettings::SyncAntiAliasing).
    {
        const char* modes[BbSettings::AaCount] = {
            BbSettings::MenuText("Auto (the game's without an upscaler)", "Авто (игры без апскейлера)"),
            BbSettings::MenuText("The game's own", "Собственное игры"),
            BbSettings::MenuText("FXAA2_GS (the game's)", "FXAA2_GS (игры)"),
            BbSettings::MenuText("Off", "Выкл"),
            BbSettings::MenuText("SMAA (without an upscaler)", "SMAA (без апскейлера)")};
        // Shown in this order: Off last.
        constexpr int order[BbSettings::AaCount] = {BbSettings::AaAuto, BbSettings::AaGame,
                                                    BbSettings::AaFxaa, BbSettings::AaSmaa,
                                                    BbSettings::AaOff};
        const int aa = std::clamp(s.anti_aliasing.load(), 0, BbSettings::AaCount - 1);
        if (ImGui::BeginCombo(BbSettings::MenuText("Anti-aliasing", "Сглаживание"), modes[aa])) {
            for (const int i : order) {
                if (ImGui::Selectable(modes[i], i == aa)) {
                    Store(s.anti_aliasing, i, true);
                }
            }
            ImGui::EndCombo();
        }
        Hint(BbSettings::MenuText(
            "The game's anti-aliasing runs before the upscaler and only blurs its input: Auto "
            "switches it off while an upscaler is on. FXAA2_GS (the game's AA step with its "
            "FXAA2_GS technique, after Kyo's patch) applies at once when the game's AA was on, "
            "else at the next load, like the game's AA. SMAA (1x) runs on the finished scene "
            "before the UI, only without an upscaler, at once. Sharpening below works with any of "
            "them.",
            "Сглаживание игры работает до апскейлера и только мылит его вход: «Авто» выключает "
            "его, пока включён апскейлер. FXAA2_GS (шаг сглаживания игры с её техникой FXAA2_GS, "
            "по патчу Kyo) применяется сразу, если сглаживание игры было включено, иначе, как и "
            "оно, со следующей загрузки. SMAA (1x) идёт по готовой сцене до интерфейса, только без "
            "апскейлера, сразу. Резкость ниже работает с любым вариантом."));
    }
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        const auto& effect = BbSettings::Effects[e];
        if (e == BbSettings::GameAaEffect) {
            continue; // the anti-aliasing choice above
        }
        Checkbox(BbSettings::MenuText(effect.label, effect.label_ru), s.effects[e]);
        if (std::string_view(effect.key) == "debug_camera") {
            Hint(BbSettings::MenuText("Hold Cross and press L3 (keyboard: F + Left Ctrl).",
                                      "Удерживайте Cross и нажмите L3 (клавиатура: F + Left Ctrl)."));
        } else if (std::string_view(effect.key) == "debug_menu") {
            Hint(BbSettings::MenuText(
                "Left side of the touchpad (Tab); right side: Backspace. Needs the adhoc folder "
                "from Nexus mod #253 (the fonts in adhoc/font).",
                "Левая сторона тачпада (Tab), правая — Backspace. Нужна папка adhoc из мода "
                "Nexus #253 (шрифты в adhoc/font)."));
        }
    }
    BbGuestHooks::ApplyGameEffects();
    Hint(BbSettings::MenuText(
        "Chromatic aberration and depth of field switch immediately; ambient occlusion, motion blur, "
        "the game's AA and dynamic shadows at the next load (a lamp, a death, a save loaded). SSR "
        "reflections, intros, free camera and debug menu apply after a restart. Motion blur and "
        "shadows from dynamic lights place a significant load on the GPU.",
        "Хроматическая аберрация и глубина резкости переключаются сразу; SSAO, размытие в движении, "
        "сглаживание игры и динамические тени — со следующей загрузки (лампа, смерть, загрузка "
        "сохранения). Отражения SSR, заставки, свободная камера и debug menu — после перезапуска. "
        "Размытие в движении и тени от динамических источников заметно нагружают GPU."));
    Checkbox(BbSettings::MenuText("GT-VBAO ambient occlusion", "Затенение GT-VBAO"), s.vbao);
    Hint(BbSettings::MenuText(
        "Replaces the game's SSAO with GT-VBAO, at once (needs Ambient occlusion on). "
        "No dark halos around thin objects. Port by redditor69.",
        "Заменяет SSAO игры на GT-VBAO, сразу (нужно включённое затенение SSAO). "
        "Без тёмных ореолов вокруг тонких предметов. Автор порта — redditor69."));
    bool show_ao = s.debug_view == BbSettings::DebugAo;
    if (ImGui::Checkbox(BbSettings::MenuText("Show AO only (debug)", "Показать только затенение (отладка)"),
                        &show_ao)) {
        s.debug_view = show_ao ? BbSettings::DebugAo : BbSettings::DebugNone;
    }
    Hint(BbSettings::MenuText(
        "The frame shows only the AO the lighting uses (white = unoccluded): VBAO when on, "
        "else the game's SSAO.",
        "Кадр показывает только затенение, которое использует освещение (белое = без "
        "затенения): VBAO, если включено, иначе SSAO игры."));

    bool restart =
        s.object_motion != s.startup_object_motion || s.model_lod != s.startup_model_lod ||
        BbSettings::ResolutionNeedsRestart();
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        restart |= !BbSettings::IsHotEffect(e) && s.effects[e] != s.startup_effects[e];
    }
    if (restart) {
        ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "%s",
                           BbSettings::MenuText("Changes apply after restarting the game",
                                                "Изменения применятся после перезапуска игры"));
        if (ImGui::Button(
                BbSettings::MenuText("Apply and restart game", "Применить и перезапустить игру"))) {
            BbSettings::Save();
            runtime_restart();
        }
    }
}

/// The menu's "Frame rate limit (now)" section.
void FrameRateLimitSection(BbSettings::Values& s) {
    ImGui::SeparatorText(BbSettings::MenuText("Frame rate limit (now)", "Лимит кадров (сейчас)"));
    {
        static const int steps[] = {0, 30, 60, 90, 120, 144, 240};
        static const char* labels[] = {"As launched", "30", "60", "90", "120", "144", "240"};
        static const char* labels_ru[] = {"Как при запуске", "30", "60", "90", "120", "144", "240"};
        int index = 0;
        for (int i = 0; i < 7; ++i) {
            if (steps[i] == BbFrameRate::Limit()) index = i;
        }
        if (ImGui::BeginCombo(BbSettings::MenuText("Limit (F3: 30 FPS on and off)", "Лимит (F3: 30 FPS вкл/выкл)"),
                              BbSettings::MenuText(labels[index], labels_ru[index]))) {
            for (int i = 0; i < 7; ++i) {
                if (ImGui::Selectable(BbSettings::MenuText(labels[i], labels_ru[i]), i == index)) {
                    BbFrameRate::SetLimit(steps[i]);
                }
            }
            ImGui::EndCombo();
        }
        Hint(BbSettings::MenuText(
            "Not saved. At 30 FPS the cinematic of Laurence (optional boss) does not hang, as it can "
            "above 30 (the community 60 FPS patch has the same problem).",
            "Не сохраняется. На 30 FPS кинематика Лоуренса (необязательный босс) не зависает, "
            "как может выше 30 (у общего патча на 60 FPS та же проблема)."));
    }
}

} // namespace BbOverlay::detail
