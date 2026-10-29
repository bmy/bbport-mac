// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_map_near.h"
#include "shim/bb_env.h"
#include "common/types.h"
#include "bbport_game_menu.h"

#include <new>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <SDL3/SDL_events.h>

#include "bbport_guest_hooks.h"
#include "bbport_settings.h"
#include <fmt/format.h>
#include "remote/bb_remote.h"

namespace BbGameMenu {
namespace {

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// The game's menu code (eboot 1.09 vaddrs; found from the System menu builder 0x1bb3ad0).
constexpr u64 MsgLookup = 0xf6f340;      ///< (repository, table, category, id) -> UTF-16 text
constexpr u64 AddPageItem = 0x1b4e2a0;   ///< (menu, name+help texts, page factory, out)
constexpr u64 MakeText = 0x1ae8cc0;      ///< (out 0x40 bytes, category, id): a text entry
constexpr u64 FactoryToFunction = 0x1b6ed90; ///< ({vtable, fn}, out std::function) -> its __f_
constexpr u64 CreatePage = 0x1bb4c70;    ///< (out, arg, content std::function): an options page
constexpr u64 OpenPage = 0x1b20900;      ///< (a, b, layout name, rows fn, 0, 0): the page's content
constexpr u64 ListPush = 0x1b1bbd0;      ///< (value list, {u32 value, text}): up to 32 values
constexpr u64 ListRow = 0x1b78ab0;   ///< (page, texts, int* value, value list, int* default): pop-up list
constexpr u64 PairPush = 0x1b2c200;  ///< (two-value list, {u8 value, text})
constexpr u64 PairRow = 0x1b2a100;   ///< (page, texts, u8* value, two-value list, u8* default): left/right
constexpr u64 SliderRow = 0x1b2ac00; ///< (page, texts, u8* value 0..10, u8* default)
/// (page, texts, int* value, value list, int* default): the value's text in the row, left/right
/// step through the list (the game's own multi-value rows, 0x1b20f74).
constexpr u64 SelectRow = 0x1b29370;
/// A SelectRow row (0xcf0 bytes, vtable 0x5339e90): `int* value` at +0x3c8, its values (0x48 bytes
/// each: u32 value, text) from +0x3d8 (8-aligned), their count at +0xce0, the list widget showing
/// them at +0x100.
constexpr std::size_t RowValue = 0x3c8, RowValues = 0x3d8, RowCount = 0xce0, RowList = 0x100;
/// (row, input, ..., xmm0 frame time): a SelectRow's input, each of the open page's rows every
/// frame (the game's menu thread).
constexpr u64 RowInput = 0x1b26b00;
/// (row + RowList): the row's list widget draws its values' texts (looked up again).
constexpr u64 ListDraw = 0x1ad06e0;
/// (row): the row shows its value's text (looked up again).
constexpr u64 RowShowValue = 0x1b2b7c0;
/// (row + RowList, count): the list widget's number of values.
constexpr u64 ListSetCount = 0x1acf2c0;
/// (row + RowList, index): the list widget moves to a value; (row + RowList): shows where it is.
constexpr u64 ListSelect = 0x1acf6b0, ListShow = 0x1acfb20;
/// The row's copy of its default value (the page's "Default").
constexpr std::size_t RowDefault = 0x3d4;
constexpr u64 FactoryVtable = 0x533d120; ///< a System item's {vtable, page factory fn}
constexpr u64 ContentVtable = 0x5343880; ///< an options page's {vtable, content fn}
/// The option page layout (optionsetting.gfx) the port's pages use, by name: its title's text
/// id and its rows. ControllSetting stacked its six rows upwards, and a row's pop-up list showed
/// the rows below through it: the runtime gives the game a copy with them stacked downwards, as
/// the game's own pop-up rows have them (src/runtime_file.c, menu_layout_file).
struct Layout {
    u64 name;
    u32 title, subtitle; ///< StaticText ids in the layout (0: none)
    std::size_t rows;
};
constexpr Layout ControlsLayout{0x4934065, 113020, 0, 6}; // "ControllSetting"
constexpr u32 TextName = 0xc8, TextHelp = 0xc9; // SP_menu text / SP_one-line help
constexpr u32 ScreenSoundItem = 110001;         // "Screen/Sound" in the System menu: ours follow it

const u8 MsgLookupPrologue[] = {0x89, 0xf0, 0x48, 0x8b, 0x77, 0x08, 0x48, 0x8b, 0x34, 0xc6};
const u8 AddPageItemPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};
const u8 RowInputPrologue[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57};

u64 image_base = 0;
template <typename F>
F Game(u64 va) {
    return reinterpret_cast<F>(image_base + va);
}

// ---- Texts: ids from IdBase up, looked up by the game like its own messages. ----
constexpr u32 IdBase = 0x7f0000;
struct Text {
    std::u16string english, russian;
};
std::vector<Text> texts;

std::u16string Utf16(const char* utf8) {
    std::u16string out;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(utf8); *p;) {
        u32 c = *p++;
        if (c >= 0xf0) {
            c = (c & 7) << 18 | (p[0] & 0x3f) << 12 | (p[1] & 0x3f) << 6 | (p[2] & 0x3f);
            p += 3;
        } else if (c >= 0xe0) {
            c = (c & 15) << 12 | (p[0] & 0x3f) << 6 | (p[1] & 0x3f);
            p += 2;
        } else if (c >= 0xc0) {
            c = (c & 31) << 6 | (p[0] & 0x3f);
            p += 1;
        }
        if (c >= 0x10000) {
            c -= 0x10000;
            out.push_back(char16_t(0xd800 + (c >> 10)));
            out.push_back(char16_t(0xdc00 + (c & 0x3ff)));
        } else {
            out.push_back(char16_t(c));
        }
    }
    return out;
}
u32 AddText(const char* english, const char* russian) {
    texts.push_back({Utf16(english), Utf16(russian)});
    return IdBase + u32(texts.size() - 1);
}
/// The game's own language decides (its "Screen/Sound" text in Cyrillic: Russian, else English);
/// the port's menu language until the game has looked that text up.
enum class GameLanguage { Unknown, English, Russian };
std::atomic<GameLanguage> game_language{GameLanguage::Unknown};

bool RussianTexts() {
    const GameLanguage game = game_language.load();
    return game == GameLanguage::Unknown
               ? BbSettings::Get().menu_language == BbSettings::MenuLanguage::Russian
               : game == GameLanguage::Russian;
}

// Texts that change while the game runs (the chosen aspect ratio's resolutions, "Auto (21:9)"):
// fixed buffers rewritten in place, so a text the game holds stays valid.
constexpr u32 DynamicBase = IdBase + 0x8000;
constexpr std::size_t DynamicLength = 40;
struct DynamicText {
    char16_t english[DynamicLength]{}, russian[DynamicLength]{};
};
constexpr u32 OutputSlots = 12; ///< BbSettings::OutputList
std::array<DynamicText, 1 + OutputSlots> dynamic_texts{};
constexpr u32 AutoText = DynamicBase;         ///< "Auto (21:9)"
constexpr u32 OutputTexts = DynamicBase + 1;  ///< the resolutions (OutputSlots)

void Store16(char16_t (&out)[DynamicLength], const char* utf8) {
    const std::u16string text = Utf16(utf8);
    const std::size_t n = std::min(text.size(), DynamicLength - 1);
    for (std::size_t i = 0; i < n; ++i) {
        __atomic_store_n(&out[i], text[i], __ATOMIC_RELAXED);
    }
    __atomic_store_n(&out[n], char16_t(0), __ATOMIC_RELAXED);
}
void SetDynamic(u32 id, const char* english, const char* russian) {
    DynamicText& t = dynamic_texts[id - DynamicBase];
    Store16(t.english, english);
    Store16(t.russian, russian);
}

bool OurText(u32 id) {
    return (id >= IdBase && id < IdBase + texts.size()) ||
           (id >= DynamicBase && id < DynamicBase + dynamic_texts.size());
}
const char16_t* TextOf(u32 id) {
    const bool russian = RussianTexts();
    if (id >= DynamicBase) {
        const DynamicText& t = dynamic_texts[id - DynamicBase];
        return russian ? t.russian : t.english;
    }
    const Text& t = texts[id - IdBase];
    return russian ? t.russian.c_str() : t.english.c_str();
}

// ---- The port's values the game's rows edit (ints: the list rows bind ints). ----
enum Field : int {
    Aspect, OutputRes, Upscaler, Preset, Sharpness, ModelLod, TextureQuality, FirstEffect, FieldCount = FirstEffect + BbSettings::EffectCount
};
int values[FieldCount];  // what the game's rows show and change (list rows bind ints)
u8 bytes[FieldCount];    // the same for toggle and slider rows (they bind bytes)
bool byte_field[FieldCount]; // fields shown by toggle or slider rows
int default_values[FieldCount]; // the port's defaults: what the page's "Default" sets
u8 default_bytes[FieldCount];
int applied[FieldCount]; // what Poll applied last
std::atomic<bool> values_valid{false};
constexpr float SharpnessStep = 0.2f; // the game's slider: 0..10 -> 0.0 .. 2.0
constexpr int SharpnessSteps = 11;
constexpr int LodValues[] = {-2, 0, 1, 2};
/// The resolution row's values: the outputs of the aspect ratio it shows (value i: entry i). Set
/// when the page is built and when another aspect ratio is chosen on it (RefreshOutputRow, the
/// game's menu thread; Poll reads it once the rows have rested).
BbSettings::OutputList menu_outputs;
int output_row_aspect = -1; ///< the aspect ratio the resolution row shows

BbSettings::OutputSize MenuOutput(int aspect, int index) {
    if (menu_outputs.count == 0) return BbSettings::NearestOutput(aspect, BbSettings::OutputDefault);
    return BbSettings::NearestOutput(aspect, menu_outputs.sizes[std::clamp(index, 0, menu_outputs.count - 1)]);
}

/// The resolution row's value of `output` (0 if none).
int MenuOutputIndex(int aspect, BbSettings::OutputSize output) {
    for (int i = 0; i < menu_outputs.count; ++i) {
        if (MenuOutput(aspect, i) == output) return i;
    }
    return 0;
}

/// The resolution row's texts for `aspect`, "Auto (..)" for the monitor's aspect ratio.
void RefreshOutputs(int aspect) {
    const std::string group = BbSettings::AspectKey(BbSettings::ResolvedAspect(BbSettings::AspectAuto));
    SetDynamic(AutoText, ("Auto (" + group + ")").c_str(), ("Авто (" + group + ")").c_str());
    for (int i = 0; i < int(OutputSlots); ++i) {
        const auto o = MenuOutput(aspect, i);
        const std::string text = std::to_string(o.width) + " x " + std::to_string(o.height);
        SetDynamic(OutputTexts + u32(i), text.c_str(), text.c_str());
    }
}

/// The rows' values of settings `s`.
void Fill(const BbSettings::Values& s, int* values) {
    values[Aspect] = s.aspect;
    values[OutputRes] = MenuOutputIndex(s.aspect, BbSettings::NearestOutput(
                                                      s.aspect, BbSettings::UnpackSize(s.output_res)));
    values[Upscaler] = s.upscaler;
    values[Preset] = 0;
    for (int i = 0; i < BbSettings::PresetCount; ++i) {
        if (BbSettings::PresetOrder[i] == s.preset) values[Preset] = i;
    }
    // One slider for both: 0 is sharpening off.
    values[Sharpness] =
        s.sharpen ? std::clamp(int(std::lround(s.sharpness / SharpnessStep)), 1, SharpnessSteps - 1) : 0;
    values[ModelLod] = 1;
    for (int i = 0; i < 4; ++i) {
        if (LodValues[i] == s.model_lod) values[ModelLod] = i;
    }
    values[TextureQuality] = std::clamp(int(s.texture_quality), 0, BbSettings::TqCount - 1);
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        values[FirstEffect + e] = s.effects[e];
    }
}

void LoadValues() {
    // bbport (macOS native GPU process): the settings menu runs in bb-gpu; its changes reach this
    // process through bbport.ini.
    if (BbRemote::FrontActive()) {
        BbSettings::ReloadLive();
    }
    const int aspect = BbSettings::Get().aspect;
    menu_outputs = BbSettings::Outputs(aspect);
    RefreshOutputs(aspect);
    Fill(BbSettings::Get(), values);
    output_row_aspect = values[Aspect];
    for (int f = 0; f < FieldCount; ++f) {
        bytes[f] = u8(values[f]);
    }
    // "Default" on the page: the game's resolution in this list.
    default_values[OutputRes] =
        MenuOutputIndex(aspect, BbSettings::NearestOutput(aspect, BbSettings::OutputDefault));
    std::memcpy(applied, values, sizeof(values));
    values_valid = true;
}

// ---- Pages and rows ----
enum class Kind { List, Toggle, Slider, Select };
struct Row {
    Field field;
    u32 name, help;
    std::vector<u32> choices; ///< value texts, value = index (a toggle: off, on)
    Kind kind = Kind::List;
};
struct Page {
    u32 name, help;
    const Layout* layout;
    std::vector<Row> rows; ///< at most the layout's
};
std::vector<Page> pages;
// "Exit to desktop" in the System menu (the title screen's too): a Quit Game to the desktop, as PC
// games have (Discord requests 2026-10-21; the Nexus mod "Just Exit Game" did it until now).
u32 exit_name = 0, exit_help = 0;

void DefineTexts() {
    const u32 off = AddText("Off", "Выкл"), on = AddText("On", "Вкл");
    const std::vector<u32> toggle = {off, on};
    const auto page = [](const char* en, const char* ru, const char* help_en, const char* help_ru,
                         const Layout& layout) {
        const u32 name = AddText(en, ru);
        pages.push_back({name, AddText(help_en, help_ru), &layout, {}});
    };
    const auto row = [](Field f, const char* en, const char* ru, const char* help_en,
                        const char* help_ru, std::vector<u32> choices, Kind kind = Kind::List) {
        if (kind == Kind::List && choices.size() == 2) {
            kind = Kind::Toggle; // left/right in place, as the game's own on/off rows
        }
        pages.back().rows.push_back(
            {f, AddText(en, ru), AddText(help_en, help_ru), std::move(choices), kind});
    };

    // Two pages of the ControllSetting layout's six rows (the game's other layouts are fixed
    // forms): the picture, then the game's effects. Choices are the game's pop-up rows
    // (SelectRow). The other patches (intros, the free camera, the debug menu), the game's own AA
    // and the FPS counter are in the overlay menu and the launcher.
    page("Display", "Изображение", "Aspect ratio, resolution, upscaler and detail (bbport)",
         "Соотношение сторон, разрешение, апскейлер и детализация (bbport)", ControlsLayout);
    row(Aspect, "Aspect ratio", "Соотношение сторон",
        "Of the final frame: wider sees more to the sides, the UI keeps to the screen edges",
        "Готового кадра: шире - больше видно по бокам, интерфейс у краёв экрана",
        [] {
            std::vector<u32> aspects = {AutoText};
            for (int a = BbSettings::AspectAuto + 1; a < BbSettings::AspectCount; ++a) {
                aspects.push_back(AddText(BbSettings::AspectKey(a), BbSettings::AspectKey(a)));
            }
            return aspects;
        }(),
        Kind::Select);
    {
        std::vector<u32> outputs;
        for (u32 i = 0; i < OutputSlots; ++i) outputs.push_back(OutputTexts + i);
        row(OutputRes, "Resolution", "Разрешение", "Size of the final frame and the UI",
            "Размер готового кадра и интерфейса", std::move(outputs), Kind::Select);
    }
    row(Upscaler, "Upscaler", "Апскейлер", "Temporal upscaler and anti-aliasing",
        "Временной апскейлер и сглаживание",
        {off, AddText("FSR 3.1", "FSR 3.1"), AddText("FSR 4", "FSR 4"), AddText("FSR 4.1.1", "FSR 4.1.1"),
         AddText("TAA", "TAA"), AddText("DLSS", "DLSS"), AddText("FSR 1", "FSR 1"),
         AddText("FSR 4 (AMD DLL)", "FSR 4 (DLL AMD)"), AddText("MetalFX", "MetalFX")},
        Kind::Select);
    // bbport (macOS port): six presets with Ultra Quality (x1.25), in BbSettings::PresetOrder
    // (sharpest first); the rows are that order, not the stored preset numbers.
    row(Preset, "Quality preset", "Пресет", "Scene resolution relative to the output",
        "Разрешение сцены относительно вывода",
        {AddText("Native AA", "Native AA"), AddText("Ultra Quality", "Ultra Quality"),
         AddText("Quality", "Quality"), AddText("Balanced", "Balanced"),
         AddText("Performance", "Performance"), AddText("Ultra Performance", "Ultra Performance")},
        Kind::Select);
    row(Sharpness, "Sharpness", "Резкость", "RCAS after the upscaler: 0 off, 10 strongest",
        "RCAS после апскейлера: 0 выкл, 10 сильнее всего", {}, Kind::Slider);
    row(ModelLod, "Model detail", "Детализация", "Level of detail of models (after a restart)",
        "Детализация моделей (после перезапуска)",
        {AddText("Highest", "Максимальная"), AddText("Game default", "Как в игре"),
         AddText("Lower", "Ниже"), AddText("Lowest", "Минимальная")},
        Kind::Select);
    row(TextureQuality, "Texture quality", "Качество текстур",
        "Large textures in VRAM: half or a quarter of their size (from the next area); auto: half "
        "with 3 GB of VRAM or less",
        "Крупные текстуры в видеопамяти: половина или четверть размера (со следующей локации); "
        "авто: половина при 3 ГБ видеопамяти и меньше",
        {AddText("Auto", "Авто"), AddText("Full", "Полное"), AddText("Half", "Половина"),
         AddText("Quarter", "Четверть")},
        Kind::Select);

    page("Effects", "Эффекты", "The game's effects (bbport): aberration and DoF at once, the rest at the next load",
         "Эффекты игры (bbport): аберрация и DoF сразу, остальные — со следующей загрузки", ControlsLayout);

    // Short names for the game's narrow name column (BbSettings::Effects order); the help line
    // has the details.
    struct Short {
        const char *key, *en, *ru, *help_en, *help_ru;
        bool graphics = true;
    };
    static const Short shorts[] = {
        {"effect_chromatic_aberration", "Chromatic aberration", "Хром. аберрация",
         "Colour fringes at the frame edges", "Цветные каймы по краям кадра"},
        {"effect_dof", "Depth of field", "Глубина резкости", "Blur of distant and near objects (DoF)",
         "Размытие дальних и близких объектов (DoF)"},
        {"effect_motion_blur", "Motion blur", "Размытие в движении", "Blur when the camera turns (from the next load)",
         "Размытие при повороте камеры (со следующей загрузки)"},
        {"effect_ssao", "Ambient occlusion", "Затенение SSAO", "Shading in corners and contacts (SSAO; from the next load)",
         "Затенение в углах и местах касания (SSAO; со следующей загрузки)"},
        {"effect_game_aa", "Game's own AA", "Сглаживание игры", "The game's own anti-aliasing",
         "Собственное сглаживание игры", false},
        {"effect_dynamic_shadows", "Dynamic shadows", "Динамические тени", "Shadows from dynamic lights (from the next load)",
         "Тени от динамических источников света (со следующей загрузки)"},
        {"effect_ssr", "SSR reflections", "Отражения SSR", "Screen-space reflections (not in the original)",
         "Экранные отражения (не было в оригинале)"},
        {"skip_intro", "Skip intros", "Пропуск заставок", "Skip the startup logos and intro",
         "Пропуск логотипов и заставки при запуске", false},
        {"debug_camera", "Free camera", "Свободная камера", "Toggled with Cross + L3",
         "Включается Cross + L3", false},
        {"debug_menu", "Debug menu", "Debug menu", "The game's debug menu (requires font files)",
         "Отладочное меню игры (нужны файлы шрифтов)", false},
    };
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        const auto& effect = BbSettings::Effects[e];
        const Short* name = nullptr;
        for (const Short& s : shorts) {
            if (std::strcmp(s.key, effect.key) == 0) name = &s;
        }
        if ((name && !name->graphics) || pages.back().rows.size() == pages.back().layout->rows) {
            continue; // the overlay menu and the launcher have it
        }
        // A new effect without a short name: its full label.
        const bool live = BbSettings::IsHotEffect(e);
        const std::string help_en = std::string(name ? name->help_en : effect.label) + (live ? "" : " (after a restart)");
        const std::string help_ru =
            std::string(name ? name->help_ru : effect.label_ru) + (live ? "" : " (после перезапуска)");
        row(Field(FirstEffect + e), name ? name->en : effect.label, name ? name->ru : effect.label_ru,
            help_en.c_str(), help_ru.c_str(), toggle);
    }
    exit_name = AddText("Exit to desktop", "Выйти на рабочий стол");
    exit_help = AddText("Closes the game, as its window's close button does",
                        "Закрывает игру, как кнопка закрытия окна");
}

// ---- The game's structures, as its own code builds them. ----
constexpr std::size_t TextSize = 0x40;     // text entry (MakeText): text pointer, string, flag
constexpr std::size_t ValueSize = 0x48;    // {u32 value; text}
constexpr std::size_t ListSize = 0x920;    // 32 values + count at +0x908 (+ alignment slack)
constexpr std::size_t PairListSize = 0xb0; // 2 values + count at +0x98 (+ alignment slack)
constexpr std::size_t ListCountOffset = 0x908, PairCountOffset = 0x98;
constexpr std::size_t FunctionSize = 0x28; // std::function: 32-byte buffer, __f_ at +0x20

/// Destroys a std::function the way the game's code does (its __f_ at +0x20).
void DestroyFunction(u8* f) {
    void* impl = *reinterpret_cast<void**>(f + 0x20);
    if (!impl) {
        return;
    }
    using Destroy = void (PS4_SYSV_ABI*)(void*, int);
    Destroy destroy = (*reinterpret_cast<Destroy**>(impl))[4];
    destroy(impl, impl != f);
    *reinterpret_cast<void**>(f + 0x20) = nullptr;
}

/// Frees a text entry's string the way the game's code does (heap buffer once capacity >= 8).
void DestroyText(u8* text) {
    if (*reinterpret_cast<u64*>(text + 0x28) >= 8) {
        void* allocator = *reinterpret_cast<void**>(text + 0x30);
        using Free = void (PS4_SYSV_ABI*)(void*, void*);
        (*reinterpret_cast<Free**>(allocator))[0x70 / 8](allocator, *reinterpret_cast<void**>(text + 0x10));
    }
    *reinterpret_cast<u64*>(text + 0x28) = 7;
    *reinterpret_cast<u64*>(text + 0x20) = 0;
    *reinterpret_cast<char16_t*>(text + 0x10) = 0;
}
void DestroyTexts(u8* pair) {
    DestroyText(pair);
    DestroyText(pair + TextSize);
}
/// A value list's entries (count at `count_offset`), after the row has copied them.
void DestroyValues(u8* list, std::size_t count_offset) {
    const u64 count = *reinterpret_cast<u64*>(list + count_offset);
    for (u64 i = 0; i < count; ++i) {
        DestroyText(list + i * ValueSize + 8);
    }
}

void MakeTexts(u8* pair, u32 name, u32 help) {
    using Make = void* (PS4_SYSV_ABI*)(void*, u32, u32);
    Game<Make>(MakeText)(pair, TextName, name);
    Game<Make>(MakeText)(pair + TextSize, TextHelp, help);
}

/// Upscalers this GPU and build can run (known once the device exists, before any menu).
bool UpscalerAvailable(int i) {
    const auto& s = BbSettings::Get();
    return i == BbSettings::UpscalerFsr4     ? s.fsr4_supported.load()
           : i == BbSettings::UpscalerFsr411 ? s.fsr411_supported.load()
           : i == BbSettings::UpscalerDlss   ? s.dlss_supported.load()
           : i == BbSettings::UpscalerFsrAmd ? s.fsr_amd_supported.load()
           : i == BbSettings::UpscalerMetalFx ? s.metalfx_supported.load()
                                             : true;
}

/// Another aspect ratio chosen on the page: the resolution row gets that aspect ratio's outputs
/// as its values (as many; the texts are looked up in the port's buffers, RefreshOutputs, when the
/// row shows them) and keeps the chosen height.
void RefreshOutputRow(u8* row) {
    const int aspect = std::clamp(values[Aspect], 0, BbSettings::AspectCount - 1);
    if (aspect == output_row_aspect) return;
    const BbSettings::OutputSize shown = MenuOutput(output_row_aspect, values[OutputRes]);
    output_row_aspect = aspect;
    menu_outputs = BbSettings::Outputs(aspect);
    RefreshOutputs(aspect);
    u8* items = row + RowValues;
    items += (0 - reinterpret_cast<std::uintptr_t>(items)) & 7;
    u64& count = *reinterpret_cast<u64*>(row + RowCount);
    // More values: copies of the first (its text an empty in-place string, see SelectRow) with
    // their own value and text.
    const u64 want = u64(menu_outputs.count);
    const bool in_place = *reinterpret_cast<u64*>(items + 0x30) < 8;
    if (count > 0 && want <= 32 && (want <= count || in_place)) {
        for (u64 i = count; i < want; ++i) {
            std::memcpy(items + i * ValueSize, items, ValueSize);
        }
        for (u64 i = 0; i < want; ++i) {
            *reinterpret_cast<u32*>(items + i * ValueSize) = u32(i);
            *reinterpret_cast<const char16_t**>(items + i * ValueSize + 8) = TextOf(OutputTexts + u32(i));
        }
        count = want;
        Game<void (PS4_SYSV_ABI*)(void*, u32)>(ListSetCount)(row + RowList, u32(want));
    }
    const int value = MenuOutputIndex(aspect, BbSettings::NearestOutput(aspect, shown));
    values[OutputRes] = value;
    default_values[OutputRes] = MenuOutputIndex(aspect, BbSettings::NearestOutput(aspect, BbSettings::OutputDefault));
    *reinterpret_cast<int*>(row + RowDefault) = default_values[OutputRes];
    Game<int (PS4_SYSV_ABI*)(void*, u32)>(ListSelect)(row + RowList, u32(value));
    Game<void (PS4_SYSV_ABI*)(void*)>(ListShow)(row + RowList);
    Game<void (PS4_SYSV_ABI*)(void*)>(RowShowValue)(row);
    Game<void (PS4_SYSV_ABI*)(void*)>(ListDraw)(row + RowList);
}

void AddRows(void* page, const std::vector<Row>& rows) {
    using Push = void (PS4_SYSV_ABI*)(void*, void*);
    using AddRow = void (PS4_SYSV_ABI*)(void*, void*, int*, void*, const int*);
    using Make = void* (PS4_SYSV_ABI*)(void*, u32, u32);
    for (const Row& row : rows) {
        alignas(16) u8 pair[2 * TextSize] = {};
        MakeTexts(pair, row.name, row.help);
        if (row.kind == Kind::Slider) {
            using Slider = void (PS4_SYSV_ABI*)(void*, void*, u8*, const u8*);
            Game<Slider>(SliderRow)(page, pair, &bytes[row.field], &default_bytes[row.field]);
            DestroyTexts(pair);
            continue;
        }
        if (row.kind == Kind::Toggle) {
            using PairRowFn = void (PS4_SYSV_ABI*)(void*, void*, u8*, void*, const u8*);
            u8* two = static_cast<u8*>(::operator new(PairListSize, std::align_val_t{16}));
            std::memset(two, 0, PairListSize);
            for (std::size_t i = 0; i < 2; ++i) {
                alignas(16) u8 value[ValueSize] = {};
                value[0] = u8(i);
                Game<Make>(MakeText)(value + 8, TextName, row.choices[i]);
                Game<Push>(PairPush)(two, value);
                DestroyText(value + 8);
            }
            Game<PairRowFn>(PairRow)(page, pair, &bytes[row.field], two, &default_bytes[row.field]);
            DestroyValues(two, PairCountOffset);
            ::operator delete(two, std::align_val_t{16});
            DestroyTexts(pair);
            continue;
        }
        u8* list = static_cast<u8*>(::operator new(ListSize, std::align_val_t{16}));
        std::memset(list, 0, ListSize);
        // The resolution row: as many as the aspect ratio has (its texts are rewritten).
        const std::size_t count =
            row.field == OutputRes ? std::size_t(menu_outputs.count) : row.choices.size();
        for (std::size_t i = 0; i < count; ++i) {
            if (row.field == Upscaler && !UpscalerAvailable(int(i))) {
                continue; // the list's values are the upscaler numbers, not positions
            }
            alignas(16) u8 value[ValueSize] = {};
            *reinterpret_cast<u32*>(value) = u32(i);
            Game<Make>(MakeText)(value + 8, TextName, row.choices[i]);
            Game<Push>(ListPush)(list, value);
            DestroyText(value + 8);
        }
        Game<AddRow>(row.kind == Kind::Select ? SelectRow : ListRow)(page, pair, &values[row.field], list,
                                                                    &default_values[row.field]);
        DestroyValues(list, ListCountOffset);
        ::operator delete(list, std::align_val_t{16});
        DestroyTexts(pair);
    }
}

std::atomic<int> building_page{-1}; // while the game builds one of our pages (its menu thread)
// Called by the game's menu code (guest threads, System V ABI like the game), one set per page.
template <int P>
void PS4_SYSV_ABI Rows(void* page, void*) {
    LoadValues();
    AddRows(page, pages[P].rows);
}
template <int P>
void* PS4_SYSV_ABI Content(void* a, void* b) {
    using Open = void* (PS4_SYSV_ABI*)(void*, void*, const char*, void*, u64, u64);
    void* page = Game<Open>(OpenPage)(a, b, Game<const char*>(pages[P].layout->name),
                                      reinterpret_cast<void*>(&Rows<P>), 0, 0);
    building_page = -1; // the layout's texts were looked up
    return page;
}
void* Factory(void* out, void* arg, void* content, int page) {
    using Create = void* (PS4_SYSV_ABI*)(void*, void*, void*);
    alignas(16) u8 function[FunctionSize] = {};
    *reinterpret_cast<u64*>(function) = image_base + ContentVtable;
    *reinterpret_cast<void**>(function + 8) = content;
    *reinterpret_cast<void**>(function + 0x20) = function;
    // The page is built later (the game's menu task): its layout title is looked up right before
    // the content function runs, which takes the override back.
    building_page = page;
    Game<Create>(CreatePage)(out, arg, function);
    DestroyFunction(function);
    return out;
}
template <int P>
void* PS4_SYSV_ABI PageFactory(void* out, void* arg) {
    return Factory(out, arg, reinterpret_cast<void*>(&Content<P>), P);
}
/// "Exit to desktop": the window's quit (gpu/shim/window.cpp ends the game), and a page for the
/// menu meanwhile (it expects one).
void* PS4_SYSV_ABI QuitFactory(void* out, void* arg) {
    std::printf("Game menu: exit to desktop\n");
    SDL_Event event{};
    event.type = SDL_EVENT_QUIT;
    SDL_PushEvent(&event);
    return Factory(out, arg, reinterpret_cast<void*>(&Content<0>), 0);
}
constexpr std::size_t MaxPages = 6;
void* const factories[MaxPages] = {
    reinterpret_cast<void*>(&PageFactory<0>), reinterpret_cast<void*>(&PageFactory<1>),
    reinterpret_cast<void*>(&PageFactory<2>), reinterpret_cast<void*>(&PageFactory<3>),
    reinterpret_cast<void*>(&PageFactory<4>), reinterpret_cast<void*>(&PageFactory<5>)};

// ---- Hooks ----
using RowInputFn = u64 (PS4_SYSV_ABI*)(u8*, u64, u64, u64, u64, u64, double);
RowInputFn row_input_original = nullptr;

u64 PS4_SYSV_ABI RowInputHook(u8* row, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5, double frame_time) {
    // The game asks the open page's rows: the resolution row is alive while it does.
    if (*reinterpret_cast<int**>(row + RowValue) == &values[OutputRes]) {
        RefreshOutputRow(row);
    }
    return row_input_original(row, a1, a2, a3, a4, a5, frame_time);
}

using LookupFn = const char16_t* (PS4_SYSV_ABI*)(void*, u32, u32, u32);
using AddFn = void* (PS4_SYSV_ABI*)(void*, void*, void*, void*);
LookupFn lookup_original = nullptr;
AddFn add_original = nullptr;
std::atomic<const void*> screen_sound_text{nullptr};

const char16_t* PS4_SYSV_ABI LookupHook(void* repository, u32 table, u32 category, u32 id) {
    if (OurText(id)) {
        // BB_GAME_MENU_TRACE=1: how often the game asks for our texts (per 2 s).
        static const bool count = bb_env_on("BB_GAME_MENU_TRACE");
        if (count) {
            static std::atomic<u64> lookups{0};
            static std::atomic<long long> printed{0};
            ++lookups;
            const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                                      std::chrono::steady_clock::now().time_since_epoch())
                                      .count();
            if (now - printed.load() >= 2) {
                printed = now;
                std::printf("Game menu: %llu lookups of the port's texts in 2 s (last id %u)\n",
                            (unsigned long long)lookups.exchange(0), id - IdBase);
            }
        }
        return TextOf(id);
    }
    if (const int p = building_page.load(); p >= 0 && category == TextName) {
        if (id == pages[p].layout->title) return TextOf(pages[p].name);
        if (pages[p].layout->subtitle && id == pages[p].layout->subtitle) return TextOf(pages[p].help);
    }
    const char16_t* text = lookup_original(repository, table, category, id);
    // BB_MSG_TRACE=1 (diagnostics): the game's text lookups per 2 s by category, with an example
    // id and caller and its text (which menu is on screen, issue #255).
    static const bool msg_trace = bb_env_on("BB_MSG_TRACE");
    if (msg_trace) {
        struct Seen {
            u64 count = 0;
            u32 id = 0;
            u64 caller = 0;
            std::string text;
        };
        static std::mutex mutex;
        static std::map<std::pair<u32, u32>, Seen> seen;
        static long long printed = 0;
        std::scoped_lock lock{mutex};
        auto& s = seen[{table, category}];
        if (s.count++ == 0) {
            s.id = id;
            s.caller = reinterpret_cast<u64>(__builtin_return_address(0)) - image_base;
            for (const char16_t* c = text; c && *c && s.text.size() < 40; ++c) {
                s.text += *c < 0x80 ? char(*c) : '?';
            }
        }
        const long long now = std::chrono::duration_cast<std::chrono::seconds>(
                                  std::chrono::steady_clock::now().time_since_epoch())
                                  .count();
        if (now - printed >= 2) {
            printed = now;
            std::string line;
            for (const auto& [key, v] : seen) {
                line += fmt::format(" [{}/{:#x}: {} e.g. {} from {:#x} \"{}\"]", key.first, key.second,
                                    v.count, v.id, v.caller, v.text);
            }
            std::printf("Msg trace (2 s):%s\n", line.c_str());
            seen.clear();
        }
    }
    static const bool trace = bb_env_on("BB_GAME_MENU_TRACE");
    if (trace && id >= 110000 && id < 120000) {
        std::printf("Game menu: text %#x/%u%s from %#llx\n", category, id, building_page >= 0 ? " (our page)" : "",
                    (unsigned long long)(reinterpret_cast<u64>(__builtin_return_address(0)) - image_base));
    }
    if (category == TextName && id == ScreenSoundItem && text) {
        screen_sound_text = text;
        bool cyrillic = false;
        for (const char16_t* c = text; *c; ++c) {
            cyrillic = cyrillic || (*c >= 0x400 && *c < 0x500);
        }
        game_language = cyrillic ? GameLanguage::Russian : GameLanguage::English;
    }
    return text;
}

void AddItem(void* menu, u32 name, u32 help, void* factory) {
    using ToFunction = void* (PS4_SYSV_ABI*)(void*, void*);
    alignas(16) u8 pair[2 * TextSize] = {};
    MakeTexts(pair, name, help);
    alignas(16) u8 source[FunctionSize] = {};
    *reinterpret_cast<u64*>(source) = image_base + FactoryVtable;
    *reinterpret_cast<void**>(source + 8) = factory;
    *reinterpret_cast<void**>(source + 0x20) = source;
    alignas(16) u8 function[FunctionSize] = {};
    *reinterpret_cast<void**>(function + 0x20) = Game<ToFunction>(FactoryToFunction)(source, function);
    alignas(16) u8 out[16] = {};
    add_original(menu, pair, function, out);
    DestroyFunction(function);
    DestroyTexts(pair);
}

void* PS4_SYSV_ABI AddHook(void* menu, void* pair, void* function, void* out) {
    void* result = add_original(menu, pair, function, out);
    const void* screen_sound = screen_sound_text.load();
    if (screen_sound && *reinterpret_cast<const void* const*>(pair) == screen_sound) {
        for (std::size_t i = 0; i < pages.size() && i < MaxPages; ++i) {
            AddItem(menu, pages[i].name, pages[i].help, factories[i]);
        }
        if (exit_name) {
            AddItem(menu, exit_name, exit_help, reinterpret_cast<void*>(&QuitFactory));
        }
    }
    return result;
}

/// `jmp hook` at the function (via a near thunk); its moved prologue + `jmp` back is the original.
bool Detour(unsigned char* image, u64 va, const u8* prologue, std::size_t length, void* hook,
            void** original, u8*& cursor) {
    u8* site = image + va;
    if (std::memcmp(site, prologue, length) != 0) {
        std::printf("Game menu: unexpected code at %#llx, not hooked\n", (unsigned long long)va);
        return false;
    }
    u8* trampoline = cursor;
    std::memcpy(trampoline, prologue, length);
    trampoline[length] = 0xe9;
    const std::int64_t back = std::int64_t(site + length) - std::int64_t(trampoline + length + 5);
    const std::int32_t back32 = std::int32_t(back);
    std::memcpy(trampoline + length + 1, &back32, 4);
    u8* thunk = trampoline + 32;
    thunk[0] = 0x48;
    thunk[1] = 0xb8; // movabs rax, hook
    std::memcpy(thunk + 2, &hook, 8);
    thunk[10] = 0xff;
    thunk[11] = 0xe0; // jmp rax
    const std::int64_t to = std::int64_t(thunk) - std::int64_t(site + 5);
    if (back != back32 || to != std::int32_t(to)) {
        std::printf("Game menu: stubs out of jump range, not hooked\n");
        return false;
    }
    const std::int32_t to32 = std::int32_t(to);
    site[0] = 0xe9;
    std::memcpy(site + 1, &to32, 4);
    for (std::size_t i = 5; i < length; ++i) {
        site[i] = 0xcc;
    }
    *original = trampoline;
    cursor += 64;
    return true;
}

/// A mapping within +-2 GiB of the image, for rel32 jumps both ways.
u8* MapNear(unsigned char* image, u64 image_size, std::size_t size) {
    const u64 base = reinterpret_cast<u64>(image);
    return static_cast<u8*>(BbMapNear::Map(base, base + image_size, size, PROT_READ | PROT_WRITE,
                                           24ull << 20, "Game menu"));
}

} // namespace

void PatchImage(unsigned char* image, std::uint64_t size) {
    if (bb_env_off("BB_GAME_MENU")) {
        return;
    }
    image_base = reinterpret_cast<u64>(image);
    const std::size_t page = std::size_t(sysconf(_SC_PAGESIZE));
    u8* stubs = MapNear(image, size, page);
    if (!stubs) {
        std::printf("Game menu: no memory near the image, the port's pages are off\n");
        return;
    }
    DefineTexts();
    for (const Page& p : pages) {
        for (const Row& r : p.rows) {
            byte_field[r.field] = r.kind == Kind::Toggle || r.kind == Kind::Slider;
        }
    }
    static const BbSettings::Values defaults;
    Fill(defaults, default_values);
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        default_values[FirstEffect + e] = BbSettings::Effects[e].default_on;
    }
    for (int f = 0; f < FieldCount; ++f) {
        default_bytes[f] = u8(default_values[f]);
    }
    u8* cursor = stubs;
    // The lookup first: the menu hook only acts once it has seen the Screen/Sound text.
    const bool ok =
        Detour(image, MsgLookup, MsgLookupPrologue, sizeof(MsgLookupPrologue),
               reinterpret_cast<void*>(&LookupHook), reinterpret_cast<void**>(&lookup_original), cursor) &&
        Detour(image, AddPageItem, AddPageItemPrologue, sizeof(AddPageItemPrologue),
               reinterpret_cast<void*>(&AddHook), reinterpret_cast<void**>(&add_original), cursor) &&
        Detour(image, RowInput, RowInputPrologue, sizeof(RowInputPrologue),
               reinterpret_cast<void*>(&RowInputHook), reinterpret_cast<void**>(&row_input_original), cursor);
    mprotect(stubs, page, PROT_READ | PROT_EXEC);
    std::printf("Game menu: %s (%zu pages in System)\n", ok ? "the port's pages are in" : "hooks failed, off",
                pages.size());
}


bool Poll() {
    if (!values_valid) {
        return false;
    }
    // A copy: the game's rows are bound to `values` (writing it would move them back).
    int now_values[FieldCount];
    std::memcpy(now_values, values, sizeof(now_values));
    for (int f = 0; f < FieldCount; ++f) {
        if (byte_field[f]) {
            now_values[f] = bytes[f];
        }
    }
    // The aspect ratio, the resolution and the preset re-create the scene targets and the
    // upscaler (a few grey frames): stepping through several did that at each one. Applied once
    // they rest for 0.4 s; the other rows at once.
    using Clock = std::chrono::steady_clock;
    static int resting[3] = {-1, -1, -1};
    static Clock::time_point rested_since{};
    const auto now = Clock::now();
    if (resting[0] != now_values[Aspect] || resting[1] != now_values[OutputRes] ||
        resting[2] != now_values[Preset]) {
        resting[0] = now_values[Aspect];
        resting[1] = now_values[OutputRes];
        resting[2] = now_values[Preset];
        rested_since = now;
    }
    if (now - rested_since < std::chrono::milliseconds(400)) {
        now_values[Aspect] = applied[Aspect];
        now_values[OutputRes] = applied[OutputRes];
        now_values[Preset] = applied[Preset];
    }
    if (std::memcmp(now_values, applied, sizeof(now_values)) == 0) {
        return false;
    }
    auto& s = BbSettings::Get();
    if (now_values[Aspect] != applied[Aspect]) {
        // Another aspect ratio keeps the resolution row's height (its texts show the new
        // aspect ratio's sizes, RefreshOutputRow).
        const int aspect = std::clamp(now_values[Aspect], 0, BbSettings::AspectCount - 1);
        s.output_res = BbSettings::PackSize(MenuOutput(aspect, now_values[OutputRes]));
        s.output_auto = false;
        s.aspect = aspect;
    } else if (now_values[OutputRes] != applied[OutputRes]) {
        s.output_res = BbSettings::PackSize(MenuOutput(s.aspect, now_values[OutputRes]));
        s.output_auto = false;
    }
    const int upscaler = std::clamp(now_values[Upscaler], 0, BbSettings::UpscalerCount - 1);
    // bbport (native GPU process): the current one stays whatever this process knows of it (the
    // native GPU process decides).
    s.upscaler = UpscalerAvailable(upscaler) || upscaler == s.upscaler ? upscaler
                                                                       : int(BbSettings::UpscalerFsr3);
    s.preset = BbSettings::PresetOrder[std::clamp(now_values[Preset], 0, BbSettings::PresetCount - 1)];
    s.sharpen = now_values[Sharpness] != 0;
    if (now_values[Sharpness] != 0) {
        s.sharpness = std::clamp(now_values[Sharpness], 0, SharpnessSteps - 1) * SharpnessStep;
    }
    s.model_lod = LodValues[std::clamp(now_values[ModelLod], 0, 3)];
    s.texture_quality = std::clamp(now_values[TextureQuality], 0, BbSettings::TqCount - 1);
    for (int e = 0; e < BbSettings::EffectCount; ++e) {
        s.effects[e] = now_values[FirstEffect + e] != 0;
    }
    // The game's AA switch here sets the anti-aliasing choice (the overlay has all of them).
    const int game_aa = FirstEffect + BbSettings::GameAaEffect;
    if (now_values[game_aa] != applied[game_aa]) {
        s.anti_aliasing = now_values[game_aa] ? int(BbSettings::AaGame) : int(BbSettings::AaOff);
    }
    BbGuestHooks::ApplyGameEffects();
    std::memcpy(applied, now_values, sizeof(now_values));
    BbSettings::Save();
    return true;
}

} // namespace BbGameMenu
