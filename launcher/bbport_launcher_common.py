# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher's settings, choices and helpers without a window (bbport_launcher.py and its page
modules import them all; tests use them without GTK)."""

import datetime
import json
import math
import shutil
import os
import re
import signal
import subprocess
import threading
import sys
import time
from pathlib import Path
from bbport_assets import fsr411_problem
from bbport_i18n import language, set_language, tr

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gio, GLib, Gtk  # noqa: E402

PORT_DIR = Path(__file__).resolve().parent.parent  # native_probe (or the package's copy)
sys.path.insert(0, str(PORT_DIR / 'scripts'))
from mods import discover as discover_mods  # noqa: E402
import game_check  # noqa: E402
import online  # noqa: E402  (the online module: addresses and the connection test)
import trophy_list  # noqa: E402  (the Trophies page: what the port keeps in the user folder)
from patches import external_patches  # noqa: E402
# Packaged (AppImage): generated files, saves and bbport.ini live in BB_DATA_DIR.
PACKAGED = bool(os.environ.get("BB_PREBUILT"))
DATA_DIR = Path(os.environ.get("BB_DATA_DIR", PORT_DIR))
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "bbport-launcher"
CONFIG_FILE = CONFIG_DIR / "settings.json"
MAX_LOG_LINES = 5000
# FSR 4.1.1 assets built from the user's AMD DLL (tools/fsr4cap, also in the package).
FSR4CAP_DIR = PORT_DIR / "tools" / "fsr4cap"
sys.path.insert(0, str(FSR4CAP_DIR))
# The package's own environment (its Mesa, libraries, GTK modules) must not reach Proton's
# container: Proton runs the DLL on the system's drivers.
PACKAGE_ONLY_ENV = (
    "LD_LIBRARY_PATH", "VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES",
    "VK_LAYER_PATH", "VK_ADD_LAYER_PATH", "__EGL_VENDOR_LIBRARY_DIRS", "GDK_PIXBUF_MODULE_FILE",
    "GIO_EXTRA_MODULES", "GI_TYPELIB_PATH", "FONTCONFIG_FILE", "GSETTINGS_SCHEMA_DIR", "PYTHONPATH",
    "PYTHONHOME",
)
# build_assets.sh exit statuses (see there).
FSR411_BUILD_ERRORS = {
    2: "DLL не подходит: подробности в журнале",
    3: ("Не найден подходящий Proton: установите GE-Proton 10 или новее (ProtonUp-Qt) или Proton "
        "Experimental / Proton-CachyOS в Steam. Если они есть — запустите в Steam любую игру с "
        "этим Proton, чтобы Steam поставил его рантайм"),
    4: "Не хватает программ для сборки: список в журнале",
    5: ("Ни один Proton не запустил FSR 4.1 из этой DLL. Либо они слишком старые (подходят "
        "GE-Proton 10+, Proton Experimental, Proton-CachyOS 11), либо DLL не включает FSR 4.1 на "
        "этой видеокарте (Steam Deck?): тогда соберите на ПК с Radeon RX 7000/9000 и скопируйте "
        "папку fsr4_411. Подробности в журнале"),
    6: ("Записанные проходы не совпали с тем, что повторяет bbport (другая версия DLL или "
        "новая видеокарта?): подробности в журнале"),
    7: "Не найден загрузчик FidelityFX 2.x",
    8: ("На NixOS запись идёт на самой системе (systemd --user) через umu-launcher из nix-shell, "
        "а здесь их не нашлось: установите umu-launcher или Nix. Подробности в журнале"),
}

# Choices: (label, value). The first entry is the default. Labels are translated when shown.
UI_LANGUAGES = [("Как в системе", ""), ("Русский", "ru"), ("English", "en"), ("Português (Brasil)", "pt_BR"),
                ("简体中文", "zh_CN"), ("Français", "fr_FR")]
UPSCALERS = [("FSR 4", "fsr4"), ("FSR 4.1.1", "fsr411"), ("FSR 3", "fsr3"),
             ("FSR 1 (для слабых видеокарт)", "fsr1"), ("DLSS (NVIDIA RTX)", "dlss"),
             ("TAA (нативное сглаживание)", "taa"), ("Выключен", "off")]
PRESETS = [("Native AA", 0), ("Quality (x1.5)", 1), ("Balanced (x1.7)", 2),
           ("Performance (x2)", 3), ("Ultra Performance (x3)", 4)]
# Output aspect ratios and their resolutions (gpu/shim/bbport_settings.cpp, the same lists): the
# game keeps its 1920x1080 frame; for another aspect ratio its camera sees more to the sides (above
# and below) and the UI keeps to the screen edges, changed live. Auto: the monitor's group.
ASPECTS = ["auto", "4:3", "16:10", "16:9", "21:9", "32:9"]
ASPECT_RATIOS = {"4:3": 4 / 3, "16:10": 16 / 10, "16:9": 16 / 9, "21:9": 64 / 27, "32:9": 32 / 9}
OUTPUTS = {
    "4:3": [(640, 480), (800, 600), (1024, 768), (1280, 960), (1440, 1080), (1600, 1200), (2048, 1536),
            (2880, 2160)],
    "16:10": [(640, 400), (960, 600), (1280, 800), (1440, 900), (1680, 1050), (1920, 1200), (2560, 1600),
              (3840, 2400)],
    "16:9": [(640, 360), (854, 480), (960, 540), (1280, 720), (1600, 900), (1920, 1080), (2560, 1440),
             (3200, 1800), (3840, 2160)],
    "21:9": [(860, 360), (1280, 540), (1720, 720), (2560, 1080), (3440, 1440), (3840, 1600), (5120, 2160)],
    "32:9": [(1280, 360), (1920, 540), (2560, 720), (3840, 1080), (5120, 1440), (7680, 2160)],
}


def resolved_aspect(aspect, monitor):
    """The aspect ratio group: Auto is the one nearest the monitor's (16:9 when unknown)."""
    if aspect in OUTPUTS:
        return aspect
    if not monitor:
        return "16:9"
    ratio = monitor[0] / monitor[1]
    return min(OUTPUTS, key=lambda a: abs(math.log(ratio / ASPECT_RATIOS[a])))


def output_sizes(aspect, monitor):
    """The resolutions of an aspect ratio; Auto adds the monitor's own size."""
    sizes = list(OUTPUTS[resolved_aspect(aspect, monitor)])
    if (aspect == "auto" and monitor and monitor not in sizes and 640 <= monitor[0] <= 7680
            and 360 <= monitor[1] <= 4320):
        sizes = sorted(sizes + [monitor], key=lambda size: (size[1], size[0]))
    return sizes


def nearest_output(sizes, size):
    """The size itself if listed, else the one nearest in height (another aspect ratio keeps it)."""
    return size if size in sizes else min(sizes, key=lambda s: abs(s[1] - size[1]))


def parse_size(text, default=(1920, 1080)):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        return (w, h) if w > 0 and h > 0 else default
    except ValueError:
        return default
# Game effects (patches applied at start): bbport.ini key, title, default.
EFFECTS = [
    ("effect_chromatic_aberration", "Хроматическая аберрация", True),
    ("effect_dof", "Глубина резкости (DoF)", True),
    ("effect_motion_blur", "Размытие в движении", True),
    ("effect_ssao", "Затенение SSAO", True),
    ("effect_dynamic_shadows", "Тени от динамических источников", True),
    ("effect_ssr", "Отражения SSR (не было в игре)", False),
    ("effect_bloom", "Свечение (Bloom)", True),
    ("skip_intro", "Пропуск заставок при запуске", False),
    ("debug_camera", "Свободная камера (Cross + L3 / F + Ctrl)", False),
    ("debug_menu", "Debug menu (левый touchpad / Tab; нужны шрифты)", False),
]
# Anti-aliasing (bbport.ini anti_aliasing; the game's own AA switch follows it): auto = the game's
# own without an upscaler, none with one; the game's AA step with FXAA2_GS (after Kyo's patch,
# live); SMAA 1x on the finished scene, without an upscaler.
ANTI_ALIASING = [("Авто (игры без апскейлера)", "auto", "С апскейлером сглаживание игры выключено: оно только мылит его вход"),
                 ("Собственное игры", "game"), ("FXAA2_GS (игры)", "fxaa", "Применяется сразу"),
                 ("SMAA (без апскейлера)", "smaa", "Применяется сразу; с апскейлером не работает"), ("Выкл", "off")]
# bbport.ini texture_quality: large textures in VRAM at full, half or a quarter size.
TEXTURE_QUALITY = [("Авто", "auto", "Половина при 3 ГБ видеопамяти и меньше"), ("Полное", "full", ""),
                   ("Половина", "half", "Вдвое меньше по каждой стороне"),
                   ("Четверть", "quarter", "Вчетверо меньше по каждой стороне")]
MODEL_LOD = [("Как в игре", "0"), ("Максимальная (-2)", "-2"), ("Ниже (1)", "1"),
             ("Минимальная (2)", "2")]
FPS_MODES = [("Без ограничения", "uncap"), ("60", "60"), ("90", "90"),
             ("30 (как на PS4)", "30")]
# Issue #299 (runtime_pad.c button_prompts): the button icons of a device whatever is used.
BUTTON_PROMPTS = [("Авто (последнее устройство)", "auto"), ("PlayStation", "playstation"),
                  ("Xbox", "xbox"), ("Клавиатура и мышь", "keyboard")]
PRESENT_MODES = [("Mailbox", "Mailbox"), ("FIFO (VSync)", "Fifo"),
                 ("FIFO Relaxed", "FifoRelaxed"), ("Immediate", "Immediate")]
# A third element is a note shown under the row's title while that choice is selected: the
# choice labels stay short, so the selected one is shown in full.
DRAW_PIPE = [("Авто", "", "Включён при 8 и более потоках процессора"), ("Включён", "1"),
             ("Выключен", "0", "Стабильнее, но медленнее")]
# The game's text languages (dvdroot_ps4/msg) by the PS4 system language code (#249).
LANGUAGES = [("Английский", "1"), ("Английский (Великобритания)", "18"), ("Русский", "8"),
             ("Японский", "0"), ("Французский", "2"), ("Испанский", "3"),
             ("Испанский (Латинская Америка)", "20"), ("Немецкий", "4"), ("Итальянский", "5"),
             ("Нидерландский", "6"), ("Португальский", "7"), ("Португальский (Бразилия)", "17"),
             ("Польский", "16"), ("Турецкий", "19"), ("Корейский", "9"),
             ("Китайский (упрощённый)", "11"), ("Китайский (традиционный)", "10"),
             ("Финский", "12"), ("Шведский", "13"), ("Датский", "14"), ("Норвежский", "15")]
# Issue #269 (scripts/patches.py language_writes): the game's own text and voice language, any
# the files have, whatever the system language, the copy's region and the save chose.
GAME_TEXT_LANGUAGES = [("Как выберет игра", "auto"), ("English (US)", "enus"), ("English (UK)", "engb"),
                       ("Русский", "ruru"), ("日本語", "jajp"), ("Français", "frfr"), ("Italiano", "itit"),
                       ("Deutsch", "dede"), ("Español", "eses"), ("Español (Latinoamérica)", "esar"),
                       ("Português (Brasil)", "ptbr"), ("Português", "ptpt"), ("Nederlands", "nlnl"),
                       ("Polski", "plpl"), ("Dansk", "dadk"), ("Norsk", "nono"), ("Suomi", "fifi"),
                       ("Svenska", "svse"), ("Türkçe", "trtr"), ("中文（简体）", "zhcn"),
                       ("中文（繁體）", "zhtw"), ("한국어", "kokr")]
GAME_VOICE_LANGUAGES = [("Как выберет игра", "auto"), ("English", "engb"), ("日本語", "jajp"),
                        ("Français", "frfr"), ("Italiano", "itit"), ("Deutsch", "dede"),
                        ("Español", "eses"), ("Español (Latinoamérica)", "esar"),
                        ("Português (Brasil)", "ptbr")]

DEFAULTS = {
    "ui_language": "",
    "game_dir": "" if PACKAGED else str(PORT_DIR.parent / "CUSA03173"),
    "user_dir": "",
    "mods_dir": "",
    "mods_enabled": True,
    "patches_dir": "",
    "language": "1",
    "fullscreen": False,
    "hdr": False,
    "present_mode": "Mailbox",
    "gamepad": "",
    "gamepad_name": "",
    "display": "",
    # The GPU by its Vulkan name ("" = the port's choice: a discrete one first).
    "gpu": "",
    "skip_network_choice": True,
    "fps_mode": "uncap",
    "fps_limit": 0,
    "draw_pipe": "",
    "mangohud": False,
    "frame_stats": False,
    "save_log": False,
    "crash_diag": False,
    "gpu_profile": False,
    "vk_validation": False,
    "extra_env": "",
    # The player's trophy key (32 hex digits, as shadPS4 asks for): the game decrypts the trophy
    # names with it (BB_TROPHY_KEY, gpu/shim/bbport_trophies.cpp). Never in the repository.
    "trophy_key": "",
    # Online play (Online page, the online module gpu/bbnet20): off by default; Seamless Hunt's
    # servers (BBSC's: shadNet, its WebAPI and the game server).
    "online": False,
    # BBSC, Bloodborne seamless co-op (seamlesshunt.com): needs a server running its
    # game server next to shadNet (Seamless Hunt's does).
    "online_bbsc": True,
    "online_community": online.COMMUNITY_SERVER,
    "online_server": online.SHADNET_SERVER,
    "online_webapi": online.WEBAPI_SERVER,
    "online_npid": "",
    "online_password": "",
    "online_upnp": True,
    # The pool, both in the online module gpu/bbnet20: "seamless" (shadNet with BBSC) or "huntersdream"
    # (The Hunter's Dream, bbhost's protocol; the account comes from the Discord sign-in).
    "online_pool": "seamless",
    "thd_server": "",
    "thd_online_id": "",
    "thd_token": "",
}

# bbport.ini keys the launcher edits; the rest of the file is kept.
INI_DEFAULTS = {
    "upscaler": "fsr4",
    "preset": "4",
    "sharpen": "1",
    "sharpness": "0.50",
    "object_motion": "1",
    "show_fps": "1",
    "vbao": "0",
    "output_res": "auto",
    "model_lod": "0",
    "frame_generation": "off",
    "frame_generation_multiplier": "2",
    "low_latency": "on",
    "mouse_look": "1",
    "mouse_sensitivity": "1.00",
    "mouse_invert_y": "0",
    "mouse_invert_x": "0",
    "camera_speed_x": "1.00",
    "camera_speed_y": "1.00",
    "button_prompts": "auto",
    "game_text_language": "auto",
    "game_voice_language": "auto",
    **{key: "1" if default else "0" for key, _, default in EFFECTS},
}


def load_settings():
    settings = dict(DEFAULTS)
    try:
        settings.update(json.loads(CONFIG_FILE.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        pass
    # Addresses saved as the old defaults (shadPS4's public shadNet, which accepts only shadPS4's
    # own builds since 2026-10-12, and The Hunter's Dream) become the new ones.
    for key, old_values in online.OLD_DEFAULTS.items():
        if str(settings.get(key, '')).strip() in old_values:
            settings[key] = DEFAULTS[key]
    return settings


def save_settings(settings):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.touch(mode=0o600)
    CONFIG_FILE.chmod(0o600)  # it holds the shadNet password
    CONFIG_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")


def ini_path():
    return Path(os.environ.get("BB_CONFIG", DATA_DIR / "bbport.ini"))


def load_ini():
    values = dict(INI_DEFAULTS)
    lines = []
    try:
        lines = ini_path().read_text(encoding='utf-8').splitlines()
    except OSError:
        pass
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values, lines


def save_ini(values, lines):
    """Rewrites the edited keys in place, appends missing ones, keeps comments and others; a key
    whose value is None is removed (a control binding back to its default)."""
    written = set()
    out = []
    for line in lines:
        if "=" in line and not line.lstrip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key in values:
                if values[key] is not None:
                    out.append(f"{key}={values[key]}")
                written.add(key)
                continue
        out.append(line)
    if not lines:
        out.append("# bbport settings (in-game menu: Insert / hold L3+R3)")
    for key, value in values.items():
        if key not in written and value is not None:
            out.append(f"{key}={value}")
    ini_path().write_text("\n".join(out) + "\n", encoding="utf-8")


def patches_dir(settings):
    return Path(settings.get("patches_dir") or DATA_DIR / "patches").expanduser()


# DLSS (vk_dlss.cpp): the bridge (libbbport_dlss.so) and NVIDIA's runtime library
# libnvidia-ngx-dlss.so.<version>, beside bb-probe (the package, a build with DLSS_SDK_ROOT) or the
# player's own in <user>/dlss, which the game prefers. Linux needs the .so from NVIDIA's DLSS SDK
# (github.com/NVIDIA/DLSS, lib/Linux_x86_64/rel); a game's nvngx_dlss.dll is Windows-only.
NGX_PREFIX = "libnvidia-ngx-dlss.so."
# DLSS Frame Generation (vk_frame_generation.cpp): NVIDIA's libnvidia-ngx-dlssg.so.<version> from
# the same SDK folder (310.7 or newer), beside the Super Resolution library; GeForce RTX 40+.
NGX_FG_PREFIX = "libnvidia-ngx-dlssg.so."
DLSS_SDK_URL = "https://github.com/NVIDIA/DLSS/tree/main/lib/Linux_x86_64/rel"
# Frame generation (bbport.ini frame_generation + frame_generation_multiplier): the launcher
# offers DLSS; FSR 3.1 stays hidden until its artifacts are fixed (BB_FRAME_GEN=1 in the game).
FRAME_GENS = [("Выключена", "off"), ("DLSS 2x (NVIDIA RTX 40/50)", "dlss2"),
              ("DLSS 3x (NVIDIA RTX 50)", "dlss3"), ("DLSS 4x (NVIDIA RTX 50)", "dlss4")]
# Low latency (bbport.ini low_latency): NVIDIA Reflex through VK_NV_low_latency2 (NVIDIA driver
# 550+), else AMD Anti-Lag 2 through VK_AMD_anti_lag (Mesa's anti-lag layer, AMD's driver); nothing
# changes elsewhere. With DLSS frame generation the game keeps it at least on.
LOW_LATENCIES = [("Включена", "on"), ("Включена + Boost", "boost"), ("Выключена", "off")]


def frame_gen_choice(ini):
    """The FRAME_GENS value for the ini's frame_generation and multiplier."""
    if ini.get("frame_generation") == "dlss":
        try:
            multiplier = min(max(int(ini.get("frame_generation_multiplier", "2")), 2), 4)
        except ValueError:
            multiplier = 2
        return f"dlss{multiplier}"
    return "off"


def frame_gen_ini(choice):
    """The ini values (frame_generation, frame_generation_multiplier) of a FRAME_GENS value."""
    if choice.startswith("dlss"):
        return "dlss", choice[len("dlss"):]
    return "off", "2"


def binaries_dir():
    return PORT_DIR / ("bin" if PACKAGED else "out")


def ngx_libraries(directory, prefix=NGX_PREFIX):
    """NVIDIA's DLSS runtime libraries (prefix: Super Resolution's, NGX_FG_PREFIX for Frame
    Generation's) in directory, newest version last."""
    def version(path):
        return tuple(int(p) if p.isdigit() else 0 for p in path.name[len(prefix):].split("."))
    try:
        return sorted((p for p in Path(directory).iterdir() if p.name.startswith(prefix)), key=version)
    except OSError:
        return []


def ngx_prefix_of(name):
    """Which of NVIDIA's libraries a file name is: NGX_PREFIX, NGX_FG_PREFIX or None."""
    for prefix in (NGX_FG_PREFIX, NGX_PREFIX):  # the longer first: "dlss.so." is not "dlssg.so."
        if name.startswith(prefix):
            return prefix
    return None


def fsr411_dir():
    """Where the FSR 4.1.1 assets are looked up, as run.sh does."""
    if os.environ.get("BB_FSR411_DIR"):
        return Path(os.environ["BB_FSR411_DIR"])
    return PORT_DIR / "fsr4_411" if (PORT_DIR / "fsr4_411").is_dir() else DATA_DIR / "fsr4_411"


def fsr411_build_command(upscaler, loader=None):
    """tools/fsr4cap/build_assets.sh for the user's DLL: command and environment. The package
    writes into the data directory (it is read-only itself) and removes the work folder after."""
    env = dict(os.environ)
    if PACKAGED:
        for key in PACKAGE_ONLY_ENV:
            env.pop(key, None)
        env["BB_FSR4CAP_WORK"] = str(DATA_DIR / "fsr4cap")
        env["BB_FSR4CAP_CLEAN"] = "1"
    tools = FSR4CAP_DIR / "bin"
    if (tools / "dxil-spirv").is_file() and (tools / "fsr4cap.exe").is_file():
        env["BB_FSR4CAP_TOOLS"] = str(tools)
    env["BB_FSR411_OUT"] = os.environ.get("BB_FSR411_DIR") or str(
        (DATA_DIR if PACKAGED else PORT_DIR) / "fsr4_411")
    command = ["bash", str(FSR4CAP_DIR / "build_assets.sh"), str(upscaler)]
    return command + ([str(loader)] if loader else []), env


def online_module(pool="seamless"):
    """The online module beside the GPU library (gpu/bbnet20, both pools; a build without its
    libraries has none), or the one BB_NET_MODULE names (the folder package's online/)."""
    forced = os.environ.get("BB_NET_MODULE", "")
    if forced:
        return Path(forced).is_file()
    name = "libbbnet20.so"
    return (PORT_DIR / ("bin" if PACKAGED else "out") / "gpu" / name).is_file()


def game_environment(s):
    """Environment for run.sh from the launcher settings."""
    env = dict(os.environ)
    env["BB_GAME_DIR"] = str(Path(s["game_dir"]).expanduser())
    if s["user_dir"]:
        env["BB_USER_DIR"] = s["user_dir"]
    env["BB_MODS_DIR"] = str(Path(s.get("mods_dir") or DATA_DIR / "mods").expanduser())
    env["BB_MODS_CONFIG"] = str(DATA_DIR / "mods.json")
    env["BB_MODS_ENABLED"] = "1" if s.get("mods_enabled", True) else "0"
    env["BB_PATCHES_DIR"] = str(patches_dir(s))
    env["BB_PATCHES_CONFIG"] = str(DATA_DIR / "patches.json")
    env["BB_LANGUAGE"] = s["language"]
    # The in-game menu speaks the launcher's language until one is chosen in it (menu_language).
    env["BB_MENU_LANGUAGE"] = language()
    env["BB_FULLSCREEN"] = "1" if s["fullscreen"] else "0"
    env["BB_PRESENT_MODE"] = s["present_mode"]
    if s.get("gamepad"):
        env["BB_GAMEPAD"] = s["gamepad"]
    if s.get("display"):
        env["BB_DISPLAY"] = s["display"]
    env["BB_SKIP_NETWORK_CHOICE"] = "1" if s.get("skip_network_choice", True) else "0"
    if s["hdr"]:
        env["BB_HDR"] = "1"
    env["BB_FPS"] = s["fps_mode"]
    if s["fps_limit"] > 0:
        env["BB_FPS_LIMIT"] = str(s["fps_limit"])
    if s["draw_pipe"]:
        env["BB_DRAW_PIPE"] = s["draw_pipe"]
    if s["mangohud"]:
        env["MANGOHUD"] = "1"
        # Its default lines and the game's RAM and VRAM (a MANGOHUD_CONFIG of the user's wins;
        # set, it takes the config file's place).
        env.setdefault("MANGOHUD_CONFIG", "fps,frametime,frame_timing,gpu_stats,cpu_stats,ram,vram")
    if s["frame_stats"]:
        env["BB_FRAME_STATS"] = "1"
    if s.get("save_log"):
        env["BB_SAVE_LOG"] = "1"
    # The GPU chosen by name: its index in Vulkan's order now (BB_GPU_ID). One not found (taken
    # out, renamed by a driver update) leaves the choice to the port. BB_GPU_ID from the shell
    # wins.
    if s.get("gpu") and "BB_GPU_ID" not in os.environ:
        index = next((index for index, _vendor, _kind, name, _best in connected_gpus()
                      if name == s["gpu"]), None)
        if index is not None:
            env["BB_GPU_ID"] = str(index)
    if s.get("crash_diag"):
        env["BB_FREE_CHECK"] = "1"
        env["BB_WRITE_LOG"] = "1"
        env["BB_PRODUCER_CHECK"] = "1"
    if s["gpu_profile"]:
        env["BB_GPU_PROFILE"] = "1"
    if s["vk_validation"]:
        env["BB_VK_VALIDATION"] = "1"
    pool = "huntersdream" if s.get("online_pool") == "huntersdream" else "seamless"
    if s.get("online") and online_module(pool):
        server = s["online_server"].strip() or online.SHADNET_SERVER
        community = online.community_address(s["online_community"])
        env["BB_ONLINE"] = "1"
        env["BB_SHADNET_SERVER"] = server
        env["BB_SHADNET_WEBAPI"] = online.webapi_setting(s["online_webapi"], server, community)
        env["BB_SHADNET_NPID"] = s["online_npid"].strip()
        env["BB_SHADNET_PASSWORD"] = s["online_password"]
        env["BB_UPNP"] = "1" if s["online_upnp"] else "0"
        env["BB_COMMUNITY_SERVER"] = community
        if pool == "seamless" and s.get("online_bbsc"):
            env["BB_BBSC"] = "1"
        if pool == "huntersdream":
            # run.sh turns these into the backend's addresses and the host overrides.
            env["BB_NET_BACKEND"] = "thd"
            env["BB_COMMUNITY_SERVER"] = online.thd_base(s.get("thd_server", ""))
            env["BB_THD_ONLINE_ID"] = s.get("thd_online_id", "")
            env["BB_THD_TOKEN"] = s.get("thd_token", "")
    if trophy_list.valid_key(s.get("trophy_key", "")):
        env["BB_TROPHY_KEY"] = s["trophy_key"].strip()
    for item in s["extra_env"].split():
        if "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    return env


def open_url(url):
    """Opens url in the user's browser: GIO, then Python's webbrowser (xdg-open and the like);
    True when one of them took it."""
    try:
        if Gio.AppInfo.launch_default_for_uri(url, None):
            return True
    except GLib.Error:
        pass
    try:
        import webbrowser
        return bool(webbrowser.open(url))
    except Exception:
        return False


def flat_button(icon, tooltip, handler):
    button = Gtk.Button(icon_name=icon, valign=Gtk.Align.CENTER, tooltip_text=tooltip)
    button.add_css_class("flat")
    button.connect("clicked", handler)
    return button


def create_shortcuts():
    """A menu entry and a desktop icon that start the game at once with the saved settings,
    patches and mods included (--play), without the launcher's window. Returns the files."""
    appimage = os.environ.get("APPIMAGE")
    command = [appimage, "--play"] if appimage else [sys.executable, str(Path(__file__).resolve().parent / "bbport_launcher.py"), "--play"]
    exec_line = " ".join(f'"{arg}"' if any(c in arg for c in ' "\\') else arg for arg in command)
    data = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    icon = "bbport"
    for source in (PORT_DIR.parent / "icons" / "hicolor" / "256x256" / "apps" / "bbport.png",
                   PORT_DIR / "packaging" / "bbport.png"):
        if source.is_file():
            target = data / "icons" / "hicolor" / "256x256" / "apps" / "bbport.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            icon = str(target)
            break
    entry = ("[Desktop Entry]\nType=Application\nName=Bloodborne\n"
             "Comment=Bloodborne (bbport) with the launcher's saved settings\n"
             f"Exec={exec_line}\nIcon={icon}\nTerminal=false\nCategories=Game;\n"
             "StartupWMClass=bbport\n")
    desktop = None
    try:
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=5)
        desktop = Path(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        pass
    if desktop is None or desktop == Path.home():
        desktop = Path.home() / "Desktop"
    written = []
    for folder in (data / "applications", desktop):
        if folder == desktop and not folder.is_dir():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "bloodborne-bbport.desktop"
        path.write_text(entry, encoding="utf-8")
        path.chmod(0o755)
        # GNOME runs a desktop file only once it is trusted.
        if shutil.which("gio"):
            subprocess.run(["gio", "set", str(path), "metadata::trusted", "true"],
                           capture_output=True, check=False)
        written.append(path)
    return written


def open_folder(window, path):
    """Opens `path` in the file manager (created first, so that a new saves folder opens)."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except OSError:
        return
    Gtk.FileLauncher.new(Gio.File.new_for_path(str(path))).launch(window, None, None)


SETTINGS_SCHEMA = json.loads((Path(__file__).resolve().parent / "settings_schema.json").read_text(encoding="utf-8"))
# The sections with items for Linux (the Windows launcher's Online section is its own here: the
# Online page).
LINUX_SECTIONS = [section for section in SETTINGS_SCHEMA["sections"]
                  if any("linux" in item.get("platforms", ["linux"]) for item in section["items"])]
SECTION_ICONS = {"game": "input-gaming-symbolic", "display": "video-display-symbolic",
                 "graphics": "applications-graphics-symbolic", "effects": "weather-clear-night-symbolic",
                 "performance": "power-profile-performance-symbolic", "controls": "input-keyboard-symbolic",
                 "mods": "folder-symbolic", "developer": "utilities-system-monitor-symbolic"}


def schema_text(texts):
    """A text of the schema in the launcher language: Russian, the translation of the Russian
    text where there is one, else English."""
    if not texts:
        return ""
    russian = texts.get("ru", "")
    if language() == "ru":
        return russian
    # The schema's own translation (pt_BR, zh_CN, fr_FR: shared with the Windows launcher) first.
    if texts.get(language()):
        return texts[language()]
    translated = tr(russian)
    return translated if translated != russian else texts.get("en", russian)


def whole_label_factory():
    """Choices shown whole: the default ellipsizes the selected one next to a long description."""
    factory = Gtk.SignalListItemFactory()
    factory.connect("setup", lambda _f, item: item.set_child(Gtk.Label(xalign=0)))
    factory.connect("bind", lambda _f, item: item.get_child().set_label(item.get_item().get_string()))
    return factory


def combo_row(title, subtitle, choices, current):
    model = Gtk.StringList.new([tr(choice[0]) for choice in choices])
    row = Adw.ComboRow(title=title, model=model, factory=whole_label_factory())
    values = [choice[1] for choice in choices]
    notes = [tr(choice[2]) if len(choice) > 2 else None for choice in choices]

    def show_note(*_):
        selected = row.get_selected()
        note = row.notes[selected] if selected < len(row.notes) else None
        lines = [line for line in (row.description, note, row.hint) if line]
        row.set_subtitle("\n".join(lines))

    row.description, row.notes, row.hint = subtitle, notes, None
    row.show_note = show_note
    row.set_selected(values.index(current) if current in values else 0)
    show_note()
    row.connect("notify::selected", show_note)
    row.values = values
    return row


def combo_value(row):
    return row.values[row.get_selected()]


# Controls (runtime_pad.c): input, label, default keyboard keys (Dark Souls III-like; a chord
# is "Left Shift+Mouse Left"), default gamepad buttons (SDL names).
# bbport.ini key.<input>= / pad.<input>= replace a default; no line keeps it. Mouse buttons and the
# wheel are keys ("Mouse Left", "Wheel Down"); they act while the game holds the mouse.
CONTROLS = [
    ("cross", "Крест (взаимодействие)", "F, Return", "a"),
    ("circle", "Круг (уклонение, бег; на клавиатуре — Space и Shift ниже)", "", "b"),
    ("dodge", "Кувырок (короткое нажатие Круга; Shift+Space — прыжок)", "Space", None),
    ("sprint", "Бег (удержание Круга)", "Left Shift", None),
    ("walk", "Медленная ходьба (удержание)", "Left Alt", None),
    ("square", "Квадрат (быстрый предмет)", "Tab, R", "x"),
    ("triangle", "Треугольник (флакон крови)", "Q", "y"),
    ("l1", "L1 (трансформация оружия)", "E, Mouse X2", "leftshoulder"),
    ("r1", "R1 (атака)", "Mouse Left", "rightshoulder"),
    ("l2", "L2 (выстрел)", "Mouse Right", "lefttrigger"),
    ("r2", "R2 (сильная атака)", "Left Shift+Mouse Left, Mouse X1", "righttrigger"),
    ("l3", "L3", "Left Ctrl", "leftstick"),
    ("r3", "R3 (захват цели)", "C, Mouse Middle", "rightstick"),
    ("options", "Options (меню; в меню игры Escape — «назад», Круг)", "Escape", "start"),
    ("touchpad", "Тачпад, левая половина (жесты)", "G", "back, touchpad"),
    ("touchpad_right", "Тачпад, правая половина (личные вещи)", "Backspace", ""),
    ("up", "Крестовина вверх", "3, Up, Wheel Up", "dpup"),
    ("down", "Крестовина вниз", "4, Down, Wheel Down", "dpdown"),
    ("left", "Крестовина влево (оружие левой руки)", "1, Left", "dpleft"),
    ("right", "Крестовина вправо (оружие правой руки)", "2, Right", "dpright"),
    ("move_up", "Движение вперёд", "W", None),
    ("move_down", "Движение назад", "S", None),
    ("move_left", "Движение влево", "A", None),
    ("move_right", "Движение вправо", "D", None),
    ("look_up", "Камера вверх", "I", None),
    ("look_down", "Камера вниз", "K", None),
    ("look_left", "Камера влево", "J", None),
    ("look_right", "Камера вправо", "L", None),
]


def connected_gamepads():
    """(GUID, name) of the connected gamepads (bb-gpu-capabilities --gamepads), [] if unknown."""
    tool = PORT_DIR / ("bin" if PACKAGED else "out") / "bb-gpu-capabilities"
    try:
        run = subprocess.run([str(tool), "--gamepads"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    return [tuple(line.split("\t", 1)) for line in run.stdout.splitlines() if "\t" in line]


def connected_displays():
    """(BB_DISPLAY value, label, (w, h) or None, primary) of the monitors (bb-gpu-capabilities
    --displays), [] if unknown. The value is the monitor's name, or its number when two monitors
    share a name."""
    tool = PORT_DIR / ("bin" if PACKAGED else "out") / "bb-gpu-capabilities"
    try:
        run = subprocess.run([str(tool), "--displays"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = [line.split("\t") for line in run.stdout.splitlines() if line.count("\t") == 2]
    names = [name for name, _, _ in rows]
    return [(name if names.count(name) == 1 else str(number),
             f"{number}: {name} ({size})" + (tr(", основной") if primary == "1" else ""),
             parse_size(size, None), primary == "1")
            for number, (name, size, primary) in enumerate(rows, 1)]



def connected_gpus():
    """(index, vendor ID, type, name, the port's own choice) of the GPUs in Vulkan's order
    (bb-gpu-capabilities --gpus; BB_GPU_ID is the index), [] if unknown."""
    tool = PORT_DIR / ("bin" if PACKAGED else "out") / "bb-gpu-capabilities"
    env = {key: value for key, value in os.environ.items() if key != "BB_GPU_ID"}
    try:
        run = subprocess.run([str(tool), "--gpus"], capture_output=True, text=True, timeout=10,
                             env=env)
    except (OSError, subprocess.SubprocessError):
        return []
    rows = [line.split("\t") for line in run.stdout.splitlines() if line.count("\t") == 4]
    return [(int(index), vendor, kind, name, best == "1")
            for index, vendor, kind, name, best in rows if index.isdigit()]


class FolderList:
    """A preferences group with a folder row and one switch row per item below it."""

    def __init__(self, group):
        self.group = group
        self.rows = []

    def clear(self):
        for _, row in self.rows:
            self.group.remove(row)
        self.rows = []

    def add(self, key, row):
        self.group.add(row)
        self.rows.append((key, row))
