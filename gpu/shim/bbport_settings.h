// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: user settings changed at run time from the in-game menu (bbport_overlay.h) and kept
// in bbport.ini (BB_CONFIG overrides the path). Environment variables override the file at
// start. Readers load the atomics every frame; writers are the menu and Load().

#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <string>

namespace BbSettings {

enum Upscaler : int { UpscalerOff = 0, UpscalerFsr3 = 1, UpscalerFsr4 = 2, UpscalerFsr411 = 3,
                      UpscalerTaa = 4, UpscalerDlss = 5, UpscalerFsr1 = 6, UpscalerFsrAmd = 7,
                      UpscalerMetalFx = 8, // bbport (macOS); bbport.ini stores the name ("metalfx")
                      UpscalerCount };
/// FSR 1 (EASU + RCAS): spatial, no history or motion vectors; the cheapest (weak GPUs). The game's
/// post runs at the scene size and the finished frame is upscaled.
inline bool IsSpatial(int upscaler) {
    return upscaler == UpscalerFsr1;
}
/// FSR 4 v07, FSR 4.1.1 or AMD's own FSR DLLs (upscale/vk_fsr_amd): the same inputs, settings and
/// placement in the frame.
inline bool IsFsr4(int upscaler) {
    return upscaler == UpscalerFsr4 || upscaler == UpscalerFsr411 || upscaler == UpscalerFsrAmd;
}
// bbport: UltraQuality (x1.25) comes last so the stored numbers of the others stay.
enum Preset : int {
    NativeAA = 0, Quality, Balanced, Performance, UltraPerformance, UltraQuality, PresetCount
};
/// The presets in menu order (finest first).
inline constexpr int PresetOrder[PresetCount] = {NativeAA,     UltraQuality, Quality,
                                                 Balanced,     Performance,  UltraPerformance};
/// Frame generation (upscale/vk_frame_generation): frames made between two of the game's.
/// FSR 3.1 frame interpolation (any GPU, 2x), DLSS Frame Generation (GeForce RTX 40: 2x, RTX 50:
/// up to 4x).
enum FrameGen : int {
    FrameGenOff = 0,
    FrameGenFsr3 = 1,
    FrameGenDlss = 2,
    FrameGenFsrAmd = 3, ///< AMD's frame generation DLL (Windows; FSR 4.0.2 ML on RDNA4)
    FrameGenCount
};
/// Low latency (render/vk_low_latency: NVIDIA Reflex through VK_NV_low_latency2, AMD Anti-Lag 2
/// through VK_AMD_anti_lag).
enum LowLatencyMode : int { LowLatencyOff = 0, LowLatencyOn = 1, LowLatencyBoost = 2, LowLatencyCount };
enum DebugView : int { DebugNone = 0, DebugReactive = 1, DebugMotion = 2, DebugAo = 3, DebugViewCount };
enum class MenuLanguage { English, Russian };

/// Game effects switched by the community patches at start (patches.py EFFECTS): ini key,
/// menu label, default (the game's own behaviour).
struct Effect {
    const char* key;
    const char* label;
    const char* label_ru;
    bool default_on;
};
inline constexpr Effect Effects[] = {
    {"effect_chromatic_aberration", "Chromatic aberration", "Хроматическая аберрация", true},
    {"effect_dof", "Depth of field (DoF)", "Глубина резкости (DoF)", true},
    {"effect_motion_blur", "Motion blur", "Размытие в движении", true},
    {"effect_ssao", "Ambient occlusion (SSAO)", "Затенение SSAO", true},
    {"effect_game_aa", "Game's own anti-aliasing", "Собственное сглаживание игры", true},
    {"effect_dynamic_shadows", "Shadows from dynamic lights", "Тени от динамических источников", true},
    {"effect_ssr", "Screen-space reflections (not in original game)", "Отражения SSR (не было в игре)", false},
    {"effect_bloom", "Bloom", "Свечение (Bloom)", true},
    {"skip_intro", "Skip startup intros", "Пропуск заставок при запуске", false},
    {"debug_camera", "Free camera (Cross + L3)", "Свободная камера (Cross + L3)", false},
    {"debug_menu", "Debug menu (requires font files)", "Debug menu (нужны файлы шрифтов)", false},
};
inline constexpr int EffectCount = int(sizeof(Effects) / sizeof(Effects[0]));
/// The game's own anti-aliasing effect (Effects index), derived from anti_aliasing.
inline constexpr int GameAaEffect = 4;
/// Anti-aliasing (anti_aliasing in bbport.ini): auto = the game's own without an upscaler, none
/// with one (the upscaler anti-aliases; the game's AA before it only blurs); game; fxaa = the
/// game's AA step running FXAA2_GS (after Kyo's patch); off; smaa = SMAA 1x on the finished scene
/// without an upscaler (upscale/vk_smaa.h), the game's AA off.
enum AntiAliasing : int { AaAuto = 0, AaGame = 1, AaFxaa = 2, AaOff = 3, AaSmaa = 4, AaCount };
const char* AntiAliasingName(int mode);
/// Texture quality (texture_quality in bbport.ini; BB_TEXTURE_QUALITY overrides): block-compressed
/// mipmapped textures of 512 texels and more drop their largest level (half) or two (quarter) in
/// VRAM; auto = half on GPUs with 3 GiB of VRAM or less, else full. For textures created after a
/// change (the next area).
enum TextureQuality : int { TqAuto = 0, TqFull = 1, TqHalf = 2, TqQuarter = 3, TqCount };
const char* TextureQualityName(int quality);
/// FSR 4.1's model (fsr4_model in bbport.ini), from the asset sets built (fsr4_411/: tools/fsr4cap,
/// the 4.1.1 DLL and AMD's driver DLL for 4.1.2): auto by the GPU (FP8 on RDNA4, INT8 Lite on an
/// RDNA2 APU such as the Steam Deck, INT8 RDNA2 on other GPUs without matrix cores, else INT8);
/// FP8 runs emulated (slower) without FP8 matrices.
enum Fsr4Model : int {
    Fsr4Auto = 0, Fsr4Fp8 = 1, Fsr4Int8 = 2, Fsr4Rdna2 = 3, Fsr4Lite = 4, Fsr4ModelCount
};
const char* Fsr4ModelName(int model);
/// SSR changes resource dimensions: the complete patch must precede construction.
inline constexpr bool IsHotEffect(int effect) {
    return effect >= 0 && effect < 8 && effect != 6;
}
/// Output aspect ratios, and the output resolutions of each: the upscaler's output and the UI's
/// targets, changed live. The game keeps its 1920x1080 frame; for another aspect ratio its camera,
/// HUD markers and UI placement follow the output at run time (bbport_guest_hooks, the UI
/// composition). Auto: the group of the monitor the game is on.
enum Aspect : int { AspectAuto = 0, Aspect4x3, Aspect16x10, Aspect16x9, Aspect21x9, Aspect32x9, AspectCount };
struct OutputSize {
    int width = 1920, height = 1080;
    bool operator==(const OutputSize&) const = default;
};
inline constexpr OutputSize OutputDefault{1920, 1080}; ///< the game's own size
constexpr std::uint32_t PackSize(OutputSize s) {
    return std::uint32_t(s.width) | std::uint32_t(s.height) << 16;
}
constexpr OutputSize UnpackSize(std::uint32_t v) {
    return {int(v & 0xffff), int(v >> 16)};
}
/// An aspect ratio's outputs, smallest first.
struct OutputList {
    std::array<OutputSize, 12> sizes{};
    int count = 0;
    const OutputSize* begin() const { return sizes.data(); }
    const OutputSize* end() const { return sizes.data() + count; }
    int IndexOf(OutputSize size) const; ///< -1: not in the list
};
/// bbport.ini value ("auto", "4:3", "16:10", "16:9", "21:9", "32:9").
const char* AspectKey(int aspect);
/// The monitor the game's window is on (window.cpp): Auto follows it. 0x0 until known.
void SetMonitor(int width, int height);
/// The same, only noted (the game process with the native GPU process, whose window reports it:
/// its System menu labels Auto by it; bb-gpu fits and saves the output).
void NoteMonitor(int width, int height);
OutputSize Monitor();
/// Auto resolved to the group nearest the monitor's aspect ratio (16:9 until it is known).
int ResolvedAspect(int aspect);
/// The output resolutions of an aspect ratio (Auto: the monitor's group and its own size).
OutputList Outputs(int aspect);
/// The output of `aspect` nearest `size` in height: another aspect ratio keeps the height.
OutputSize NearestOutput(int aspect, OutputSize size);
/// The output in use: the chosen resolution in the chosen aspect ratio.
OutputSize CurrentOutput();
/// "2560x1080".
std::string SizeText(OutputSize size);

struct Values {
    std::atomic<MenuLanguage> menu_language{MenuLanguage::Russian};
    /// Chosen in the menu (or already in bbport.ini): only then is it saved, so the language
    /// the menu merely started with keeps following the launcher's (#134).
    std::atomic<bool> menu_language_chosen{false};
    std::atomic<int> upscaler{UpscalerFsr3};
    /// bbport (macOS): MetalFX failed at run time and FSR 3.1 stands in until it is tried again
    /// (the temporal upscaler's automatic retry). Save() keeps this upscaler, the one chosen, in
    /// bbport.ini while `upscaler` holds FSR 3.1. -1: none.
    std::atomic<int> upscaler_fallback_from{-1};
    /// Bumped whenever an upscaler is chosen (the overlay, or another one in bbport.ini from the
    /// game's menu), even the one already set: a stand-in's retries end there.
    std::atomic<std::uint32_t> upscaler_chosen{0};
    std::atomic<int> preset{NativeAA};
    std::atomic<bool> sharpen{true};
    std::atomic<float> sharpness{0.3f};
    std::atomic<bool> jitter{true};
    std::atomic<bool> reactive{false};
    std::atomic<bool> object_motion{true};
    std::atomic<float> reactive_scale{1.0f};
    std::atomic<float> reactive_threshold{0.2f};
    std::atomic<float> reactive_max{0.9f};
    std::atomic<int> debug_view{DebugNone};
    std::atomic<bool> show_fps{false};
    // GT-VBAO ambient occlusion in place of the game's SSAO (needs effect_ssao on)
    std::atomic<bool> vbao{false};
    /// The right stick's camera turn (#208): [0] vertical (the pitch), [1] horizontal (the yaw), as
    /// multiples of the game's own speed (the same for both axes, in radians a second). The
    /// native look detour (games/bloodborne/hooks_camera.cpp) multiplies the turn by the four floats as
    /// they are; [2] and [3] stay 1.
    alignas(16) std::atomic<float> camera_speed[4]{1.0f, 1.0f, 1.0f, 1.0f};
    /// The settings menu's position (fraction of the screen), -1 until it is moved.
    std::atomic<float> menu_x{-1.0f}, menu_y{-1.0f};
    // FSR 4 checks (menu): the provider's auto exposure, the jitter sign it is given.
    std::atomic<bool> fsr4_auto_exposure{true};
    std::atomic<bool> fsr4_invert_jitter{false};
    std::atomic<int> active_render_width{1920}, active_render_height{1080};
    /// Applied at start (patches.py); the menu shows when a restart is needed.
    std::atomic<bool> effects[EffectCount]{};
    std::atomic<int> model_lod{0}; ///< -2 highest .. 2 lowest, 0 the game's
    /// The game's light grid in large tiles (a quarter of the frame, one depth slice) instead of
    /// 128-pixel ones: less light culling (SrShadowy fork). Applied each frame.
    std::atomic<bool> light_grid_opt{false};
    std::atomic<int> anti_aliasing{AaAuto};
    std::atomic<int> texture_quality{TqAuto};
    std::atomic<int> fsr4_model{Fsr4Auto};
    /// Which Fsr4Model the asset sets and the GPU allow (bit per model; the upscaler sets it).
    std::atomic<unsigned> fsr4_models_available{0};
    /// The model the replay runs, for the menu ("INT8 (RDNA2)"); null before its first frame.
    std::atomic<const char*> fsr4_model_active{nullptr};
    /// What anti_aliasing comes to (SyncAntiAliasing): the game's FXAA forced (effects[GameAaEffect]
    /// holds the game's own AA).
    std::atomic<bool> fxaa_forced{false};
#ifdef __APPLE__
    // bbport (macOS): new users start at the PS4's own 16:9 (and 1920x1080, OutputDefault): no
    // resolution patch, no more than the PS4 asks of the GPU. Every aspect ratio works here since
    // 0.6 final (the camera and HUD hooks install with either memory model), Auto included.
    std::atomic<int> aspect{Aspect16x9};
#else
    std::atomic<int> aspect{AspectAuto};
#endif
    /// The chosen output resolution (PackSize; CurrentOutput fits it to the aspect ratio).
    std::atomic<std::uint32_t> output_res{PackSize(OutputDefault)};
    /// output_res=auto (the default): the monitor's own resolution (CurrentOutput).
#ifdef __APPLE__
    // bbport (macOS): off for new users as well (1920x1080 at 16:9, above); Auto would be the
    // Retina panel's pixels.
    std::atomic<bool> output_auto{false};
#else
    std::atomic<bool> output_auto{true};
#endif
    /// Live preset changes (run.sh): 1 on by default (over the startup patch: the launch preset
    /// costs nothing extra, lower ones apply at once), 0 off, -1 auto. On restart.
    std::atomic<int> live_resolution{1};
    /// Frame generation (upscale/vk_frame_generation): frames generated between each two of
    /// the game's, shown between them (needs the upscaler's scaled path). Live. FrameGenFsr3 is
    /// hidden in the menu until its artifacts are fixed (only BB_FRAME_GEN turns it on).
    std::atomic<int> frame_generation{FrameGenOff};
    /// Frames on screen per game frame with DLSS: 2 (RTX 40), up to 4 (RTX 50). FSR 3.1: 2.
    std::atomic<int> frame_generation_multiplier{2};
    /// Whether the GPU can run FSR 3 frame generation (set by the renderer), or why not.
    std::atomic<bool> frame_generation_supported{false};
    std::atomic<const char*> frame_generation_problem{nullptr};
    /// Frame generation chosen in a session started without it at 1080p native (the unscaled path
    /// has none: TemporalUpscaler::RunScaled only): it starts after a restart.
    std::atomic<bool> frame_generation_restart{false};
    /// The scene can be reduced live (SceneTargets::DepthProxies; set by the upscaler). Without it
    /// presets and the output apply at the next start (ResolutionNeedsRestart).
    std::atomic<bool> scene_proxies{true};
    /// DLSS Frame Generation: the highest multiplier this GPU offers (1: unavailable), or why
    /// not (set by the renderer; null until the device exists, or when it is available).
    std::atomic<int> dlssg_max_multiplier{1};
    std::atomic<const char*> dlssg_problem{nullptr};
    /// Low latency (NVIDIA Reflex): off, on, on with boost. Live. On by default where the driver
    /// has it (low_latency_supported, set by the renderer); nothing changes elsewhere.
    std::atomic<int> low_latency{LowLatencyOn};
    std::atomic<bool> low_latency_supported{false};
    /// Which one runs (render/vk_low_latency Backend): 0 none, 1 NVIDIA Reflex, 2 AMD Anti-Lag 2.
    std::atomic<int> low_latency_backend{0};
    /// The presenter's measure for the FPS counter: the game's own frame rate (real frames),
    /// and the generated frames shown per game frame right now (0: none).
    std::atomic<float> game_fps{0.0f};
    std::atomic<int> frame_gen_now{0};
    /// Why FSR 4 cannot run (assets, device features), or null. Set by the renderer.
    std::atomic<const char*> fsr4_problem{nullptr};
    std::atomic<bool> fsr4_supported{false}, fsr411_supported{false};
    /// AMD's FSR DLLs (Windows) were found and loaded; the provider they run (for the menu).
    std::atomic<bool> fsr_amd_supported{false};
    std::atomic<const char*> fsr_amd_provider{nullptr};
    /// Which Fsr4Model AMD's DLLs can be made to run (bit per model; fsr4_model picks one).
    std::atomic<unsigned> fsr_amd_models{0};
    /// Frame generation from AMD's DLL (FrameGenFsrAmd): found, the provider it runs (for the
    /// menu), or why it stopped.
    std::atomic<bool> fg_amd_supported{false};
    std::atomic<const char*> fg_amd_provider{nullptr};
    std::atomic<const char*> fg_amd_problem{nullptr};
    /// MetalFX (macOS with KosmicKrisp only); why it cannot run, or null. Set by the renderer.
    std::atomic<bool> metalfx_supported{false};
    std::atomic<const char*> metalfx_problem{nullptr};
    /// DLSS (gpu/dlss_bridge, NVIDIA RTX) is ready, or why not (null before the device exists).
    std::atomic<bool> dlss_supported{false};
    std::atomic<const char*> dlss_problem{nullptr};

    /// Startup settings for the explicit BB_RENDER_RES compatibility patch only.
    int startup_preset = NativeAA;
    int startup_upscaler = UpscalerFsr3;
    bool startup_object_motion = true;
    bool startup_effects[EffectCount]{};
    int startup_model_lod = 0;
    /// The output the startup patch was made for (BB_OUTPUT_RES), else 0x0.
    OutputSize startup_output{0, 0};
    int startup_live_resolution = 0;
};

Values& Get();
/// Localized overlay text; the Language selector itself stays in English.
const char* MenuText(const char* english, const char* russian);

/// Reads the file, then the environment overrides. Called once at start.
void Load();
/// Checks the loaded choice before the first frame: unsupported FSR 4 uses FSR 3.1, MetalFX off.
void ConfigureUpscalerSupport(bool fsr4, bool fsr411, bool metalfx = false);
/// After device creation: whether AMD's FSR DLLs loaded; that choice falls back to FSR 3.1 without.
void ConfigureFsrAmdSupport(bool available);
/// After device creation: DLSS availability; a DLSS setting falls back to FSR 3.1 without it.
void ConfigureDlssSupport(bool available, const char* problem);
/// Startup-patched scene dimensions cannot change until run.sh prepares a new image.
bool FixedRenderSession();
/// effects[GameAaEffect] and fxaa_forced from anti_aliasing and the upscaler (live; the game's AA
/// takes effect at the next load, FXAA at once).
void SyncAntiAliasing();
/// The startup patch with live presets over it (BB_LIVE_HYBRID=1): the launch preset is the
/// game's own size; others are drawn through proxies until the next start.
bool HybridSession();
int RenderPreset();
bool ResolutionNeedsRestart();
/// Writes the file (menu changes).
void Save();
/// bbport (macOS native GPU process): the game process writes only the keys the game's System
/// menu pages change; the rest (the overlay's advanced settings, its language and position) are
/// bb-gpu's, whose copy is the current one, and stay as the file has them.
inline bool save_game_menu_keys_only = false;
/// bbport (macOS native GPU process): reads bbport.ini again into the values the game's System menu
/// pages change (output, upscaler, preset, sharpening, FPS counter, model detail, effects), so
/// that the process which did not make a change follows it. Startup values stay.
void ReloadLive();

/// Render resolution divisor of a preset (1.0 native, 1.5 quality, ...).
float PresetScale(int preset);
const char* PresetName(int preset);
/// "off", "fsr3", "dlss", "fsr_amd" (bbport.ini, BB_FRAME_GEN).
const char* FrameGenName(int frame_gen);
/// "off", "on", "boost" (bbport.ini, BB_LOW_LATENCY).
const char* LowLatencyName(int mode);
/// The low latency mode that runs: the setting, but at least On with DLSS Frame Generation (as
/// NVIDIA requires of games: generated frames add latency that Reflex takes back).
int EffectiveLowLatency();
const char* UpscalerName(int upscaler);

} // namespace BbSettings
