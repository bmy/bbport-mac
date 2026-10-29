#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Bloodborne launcher for Windows (Tkinter; launcher/bbport_launcher.py is the Linux one).

The same settings as on Linux: the game's in bbport.ini next to the port (upscaler, preset,
output, effects, ...), the launcher's own (game folder, frame rate, window, ...) in
%APPDATA%\\bbport-launcher\\settings.json. PLAY prepares and starts the game through run.py in
this process and shows its output; every run is logged in logs\\.

Frozen with PyInstaller (packaging/windows/package.sh) into Bloodborne.exe; `Bloodborne.exe
--play` starts the game with the saved settings without the window. The layout of the package and
the idea of a frozen Tkinter launcher follow Supermedo's Windows port (github.com/Supermedo/
bloodborne_pc, GPL-2.0), written anew for this tree.
"""
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading

FROZEN = getattr(sys, 'frozen', False)
PACKAGE_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PACKAGE_DIR))
sys.path.insert(0, str(PACKAGE_DIR / 'scripts'))
DATA_DIR = Path(os.environ.get('BB_DATA_DIR', PACKAGE_DIR))
CONFIG_DIR = Path(os.environ.get('APPDATA', Path.home())) / 'bbport-launcher'
CONFIG_FILE = CONFIG_DIR / 'settings.json'
INI_FILE = DATA_DIR / 'bbport.ini'
MAX_LOG_LINES = 5000

# The lists of launcher/bbport_launcher.py (the Linux launcher; GTK is not used here).
UPSCALERS = [('FSR 4', 'fsr4'), ('FSR 3', 'fsr3'), ('TAA (нативное сглаживание)', 'taa'), ('Выключен', 'off')]
PRESETS = [('Native AA', '0'), ('Quality (x1.5)', '1'), ('Balanced (x1.7)', '2'),
           ('Performance (x2)', '3'), ('Ultra Performance (x3)', '4')]
OUTPUTS = ['1280x720', '1600x900', '1920x1080', '2560x1440', '3840x2160', '1920x1200', '2560x1600',
           '2560x1080', '3440x1440', '5120x1440']
FPS_MODES = [('Без ограничения', 'uncap'), ('60', '60'), ('90', '90'), ('30 (как на PS4)', '30')]
PRESENT_MODES = [('Mailbox', 'Mailbox'), ('FIFO (VSync)', 'Fifo'), ('FIFO Relaxed', 'FifoRelaxed'),
                 ('Immediate', 'Immediate')]
LANGUAGES = [('Английский', '1'), ('Английский (Великобритания)', '18'), ('Русский', '8'),
             ('Японский', '0'), ('Французский', '2'), ('Испанский', '3'),
             ('Испанский (Латинская Америка)', '20'), ('Немецкий', '4'), ('Итальянский', '5'),
             ('Нидерландский', '6'), ('Португальский', '7'), ('Португальский (Бразилия)', '17'),
             ('Польский', '16'), ('Турецкий', '19'), ('Корейский', '9'),
             ('Китайский (упрощённый)', '11'), ('Китайский (традиционный)', '10'),
             ('Финский', '12'), ('Шведский', '13'), ('Датский', '14'), ('Норвежский', '15')]
EFFECTS = [
    ('effect_chromatic_aberration', 'Хроматическая аберрация', True),
    ('effect_dof', 'Глубина резкости (DoF)', True),
    ('effect_motion_blur', 'Размытие в движении', True),
    ('effect_ssao', 'Затенение SSAO', True),
    ('effect_dynamic_shadows', 'Тени от динамических источников', True),
    ('effect_ssr', 'Отражения SSR (не было в игре)', False),
    ('effect_bloom', 'Свечение (Bloom)', True),
    ('skip_intro', 'Пропуск заставок при запуске', False),
]
ANTI_ALIASING = [('Авто (игры без апскейлера)', 'auto'), ('Собственное игры', 'game'),
                 ('FXAA2_GS (игры)', 'fxaa'), ('SMAA (без апскейлера)', 'smaa'), ('Выкл', 'off')]
# bbport.ini texture_quality: large textures in VRAM at full, half or a quarter size.
TEXTURE_QUALITY = [('Авто (половина при 3 ГБ видеопамяти и меньше)', 'auto'), ('Полное', 'full'),
                   ('Половина', 'half'), ('Четверть', 'quarter')]
# bbport.ini low_latency: NVIDIA Reflex, or AMD Anti-Lag 2 (AMD's driver); Boost is Reflex's.
LOW_LATENCIES = [('Включена', 'on'), ('Включена + Boost (NVIDIA)', 'boost'), ('Выключена', 'off')]
MODEL_LOD = [('Как в игре', '0'), ('Максимальная (-2)', '-2'), ('Ниже (1)', '1'), ('Минимальная (2)', '2')]

DEFAULTS = {
    'game_dir': '',
    'user_dir': '',
    'language': '8',
    'fullscreen': False,
    'present_mode': 'Mailbox',
    'skip_network_choice': True,
    'fps_mode': 'uncap',
    'fps_limit': 0,
    'gpu_id': '',
    'save_log': False,
    'extra_env': '',
    'trophy_key': '',  # the player's trophy key (Bloodborne.exe's Trophies section): BB_TROPHY_KEY
}
INI_DEFAULTS = {'upscaler': 'fsr4', 'preset': '3', 'output_res': 'auto', 'model_lod': '0',
                'menu_language': 'ru', 'low_latency': 'on', 'anti_aliasing': 'auto', 'texture_quality': 'auto'}


def load_settings():
    settings = dict(DEFAULTS)
    try:
        settings.update(json.loads(CONFIG_FILE.read_text(encoding='utf-8')))
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding='utf-8')


def read_ini():
    values = dict(INI_DEFAULTS)
    for key, _title, default in EFFECTS:
        values[key] = '1' if default else '0'
    try:
        for line in INI_FILE.read_text(encoding='utf-8', errors='replace').splitlines():
            if '=' in line and not line.lstrip().startswith('#'):
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip()
    except OSError:
        pass
    return values


def write_ini(changes):
    """Changes keys of bbport.ini; its other lines (the in-game menu's) stay as they are."""
    lines = []
    try:
        lines = INI_FILE.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        lines = ['# bbport settings (in-game menu: Insert / L3+R3)']
    left = dict(changes)
    for i, line in enumerate(lines):
        if '=' in line and not line.lstrip().startswith('#'):
            key = line.split('=', 1)[0].strip()
            if key in left:
                lines[i] = f'{key}={left.pop(key)}'
    lines += [f'{key}={value}' for key, value in left.items()]
    INI_FILE.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def game_environment(s):
    """Environment for run.py from the launcher settings (game_environment of the Linux one)."""
    env = dict(os.environ)
    env['BB_DATA_DIR'] = str(DATA_DIR)
    env['BB_GAME_DIR'] = s['game_dir']
    if s.get('user_dir'):
        env['BB_USER_DIR'] = s['user_dir']
    env['BB_LANGUAGE'] = s['language']
    # The new launcher's language (Bloodborne.exe, WPF) is the port menu's too.
    # The port's in-game menu has Russian and English: the other launcher languages get English.
    env['BB_MENU_LANGUAGE'] = 'ru' if (s.get('ui_language') or 'ru') == 'ru' else 'en'
    env['BB_FULLSCREEN'] = '1' if s['fullscreen'] else '0'
    env['BB_PRESENT_MODE'] = s['present_mode']
    # The monitor (its name, or number) and the controller (SDL GUID) the launcher chose.
    if s.get('display'):
        env['BB_DISPLAY'] = s['display']
    if s.get('gamepad'):
        env['BB_GAMEPAD'] = s['gamepad']
    env['BB_SKIP_NETWORK_CHOICE'] = '1' if s['skip_network_choice'] else '0'
    env['BB_FPS'] = s['fps_mode']
    if int(s.get('fps_limit') or 0) > 0:
        env['BB_FPS_LIMIT'] = str(int(s['fps_limit']))
    if str(s.get('gpu_id', '')).strip():
        env['BB_GPU_ID'] = str(s['gpu_id']).strip()
    if s.get('save_log'):
        env['BB_SAVE_LOG'] = '1'
    if s.get('frame_stats'):
        env['BB_FRAME_STATS'] = '1'
    if s.get('gpu_profile'):
        env['BB_GPU_PROFILE'] = '1'
    if s.get('vk_validation'):
        env['BB_VK_VALIDATION'] = '1'
    if s.get('crash_diag'):
        env['BB_FREE_CHECK'] = env['BB_WRITE_LOG'] = env['BB_PRODUCER_CHECK'] = '1'
    # Mods and third-party patches (Bloodborne.exe's Mods section; run.py hands them to
    # scripts/mods.py and patches.py): the folders and whether mods load.
    env['BB_MODS_DIR'] = str(s.get('mods_dir') or DATA_DIR / 'mods')
    env['BB_MODS_ENABLED'] = '0' if s.get('mods_enabled') in (False, '0', 0) else '1'
    env['BB_PATCHES_DIR'] = str(s.get('patches_dir') or DATA_DIR / 'patches')
    # Online play (Bloodborne.exe's Online section; game_environment of the Linux launcher): two
    # pools, Seamless Hunt's shadNet server (with BBSC, seamless co-op, when chosen) and The
    # Hunter's Dream (Discord sign-in). run.py adds the module and the servers' addresses.
    if s.get('online') in (True, '1', 1):
        import online
        server = str(s.get('online_server') or '').strip() or online.SHADNET_SERVER
        community = online.community_address(s.get('online_community'))
        env['BB_ONLINE'] = '1'
        env['BB_SKIP_NETWORK_CHOICE'] = '0'  # the game's online / offline choice is needed
        env['BB_SHADNET_SERVER'] = server
        env['BB_SHADNET_WEBAPI'] = online.webapi_setting(s.get('online_webapi'), server, community)
        env['BB_SHADNET_NPID'] = str(s.get('online_npid') or '').strip()
        env['BB_SHADNET_PASSWORD'] = str(s.get('online_password') or '')
        env['BB_UPNP'] = '0' if s.get('online_upnp') in (False, '0', 0) else '1'
        env['BB_COMMUNITY_SERVER'] = community
        if s.get('online_pool') == 'huntersdream':
            env['BB_NET_BACKEND'] = 'thd'
            env['BB_COMMUNITY_SERVER'] = online.thd_base(s.get('thd_server', ''))
            env['BB_THD_ONLINE_ID'] = str(s.get('thd_online_id') or '')
            env['BB_THD_TOKEN'] = str(s.get('thd_token') or '')
        elif s.get('online_bbsc', True) not in (False, '0', 0):
            env['BB_BBSC'] = '1'
    trophy_key = str(s.get('trophy_key') or '').strip()
    if re.fullmatch(r'[0-9A-Fa-f]{32}', trophy_key):
        env['BB_TROPHY_KEY'] = trophy_key
    for item in s.get('extra_env', '').split():
        if '=' in item:
            key, value = item.split('=', 1)
            env[key] = value
    return env


def play(settings, on_line=None):
    import run
    return run.run(env=game_environment(settings), on_line=on_line)


def main_window():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    settings = load_settings()
    ini = read_ini()
    root = tk.Tk()
    version_file = PACKAGE_DIR / 'VERSION'  # issue #307
    version = version_file.read_text().strip() if version_file.is_file() else ''
    root.title(f'Bloodborne (bbport {version})' if version else 'Bloodborne (bbport)')
    root.geometry('900x700')
    icon = PACKAGE_DIR / 'launcher' / 'bloodborne.ico'
    if icon.is_file():
        try:
            root.iconbitmap(str(icon))
        except tk.TclError:
            pass
    notebook = ttk.Notebook(root)
    notebook.pack(fill='x', padx=8, pady=8)
    variables = {}

    def row(parent, r, title, widget):
        ttk.Label(parent, text=title).grid(row=r, column=0, sticky='w', padx=6, pady=3)
        widget.grid(row=r, column=1, sticky='we', padx=6, pady=3)
        parent.columnconfigure(1, weight=1)

    def combo(parent, r, title, choices, value, key, store):
        labels = [label for label, _ in choices]
        current = next((label for label, v in choices if v == str(value)), labels[0])
        var = tk.StringVar(value=current)
        row(parent, r, title, ttk.Combobox(parent, textvariable=var, values=labels, state='readonly'))
        variables[key] = (store, lambda: next(v for label, v in choices if label == var.get()))

    def check(parent, r, title, value, key, store):
        var = tk.BooleanVar(value=bool(value))
        ttk.Checkbutton(parent, text=title, variable=var).grid(row=r, column=0, columnspan=2, sticky='w',
                                                                padx=6, pady=2)
        variables[key] = (store, var.get)

    def entry(parent, r, title, value, key, store):
        var = tk.StringVar(value=str(value))
        row(parent, r, title, ttk.Entry(parent, textvariable=var))
        variables[key] = (store, var.get)

    # Game
    game = ttk.Frame(notebook)
    notebook.add(game, text='Игра')
    game_var = tk.StringVar(value=settings['game_dir'])
    folder = ttk.Frame(game)
    ttk.Entry(folder, textvariable=game_var).pack(side='left', fill='x', expand=True)
    ttk.Button(folder, text='Выбрать…', command=lambda: game_var.set(
        filedialog.askdirectory(title='Папка игры (с eboot.bin)') or game_var.get())).pack(side='left', padx=4)
    row(game, 0, 'Папка игры (CUSA03173, с eboot.bin)', folder)
    variables['game_dir'] = ('s', game_var.get)
    combo(game, 1, 'Язык игры', LANGUAGES, settings['language'], 'language', 's')
    check(game, 2, 'Пропускать выбор «онлайн / оффлайн»', settings['skip_network_choice'], 'skip_network_choice', 's')
    entry(game, 3, 'Папка сохранений (пусто: user рядом с игрой)', settings['user_dir'], 'user_dir', 's')

    # Graphics
    gfx = ttk.Frame(notebook)
    notebook.add(gfx, text='Графика')
    combo(gfx, 0, 'Апскейлер', UPSCALERS, ini['upscaler'], 'upscaler', 'i')
    combo(gfx, 1, 'Пресет', PRESETS, ini['preset'], 'preset', 'i')
    combo(gfx, 2, 'Разрешение вывода', [('Авто (монитор)', 'auto')] + [(o, o) for o in OUTPUTS] +
          ([(ini['output_res'], ini['output_res'])] if ini['output_res'] not in OUTPUTS + ['auto'] else []),
          ini['output_res'], 'output_res', 'i')
    combo(gfx, 3, 'Частота кадров', FPS_MODES, settings['fps_mode'], 'fps_mode', 's')
    entry(gfx, 4, 'Ограничение FPS (0: нет)', settings['fps_limit'], 'fps_limit', 's')
    combo(gfx, 5, 'Режим показа кадров', PRESENT_MODES, settings['present_mode'], 'present_mode', 's')
    check(gfx, 6, 'Полный экран', settings['fullscreen'], 'fullscreen', 's')
    combo(gfx, 7, 'Детализация моделей', MODEL_LOD, ini['model_lod'], 'model_lod', 'i')
    combo(gfx, 8, 'Низкая задержка (NVIDIA Reflex / AMD Anti-Lag 2)', LOW_LATENCIES,
          ini['low_latency'], 'low_latency', 'i')
    combo(gfx, 9, 'Сглаживание', ANTI_ALIASING, ini['anti_aliasing'], 'anti_aliasing', 'i')
    combo(gfx, 10, 'Качество текстур', TEXTURE_QUALITY, ini['texture_quality'], 'texture_quality', 'i')

    # Effects
    fx = ttk.Frame(notebook)
    notebook.add(fx, text='Эффекты')
    for i, (key, title, _default) in enumerate(EFFECTS):
        check(fx, i, title, ini.get(key) == '1', key, 'ib')

    # Advanced
    adv = ttk.Frame(notebook)
    notebook.add(adv, text='Дополнительно')
    entry(adv, 0, 'Видеокарта (номер в Vulkan, пусто: авто)', settings['gpu_id'], 'gpu_id', 's')
    check(adv, 1, 'Статистика кадров в журнале (logs\\*.frames.csv)', settings['save_log'], 'save_log', 's')
    entry(adv, 2, 'Переменные окружения (KEY=VALUE через пробел)', settings['extra_env'], 'extra_env', 's')
    ttk.Label(adv, text=f'Данные порта: {DATA_DIR}\nНастройки лаунчера: {CONFIG_FILE}',
              foreground='gray').grid(row=3, column=0, columnspan=2, sticky='w', padx=6, pady=8)

    log = tk.Text(root, height=20, wrap='none', state='disabled', font=('Consolas', 9))
    log.pack(fill='both', expand=True, padx=8)
    buttons = ttk.Frame(root)
    buttons.pack(fill='x', padx=8, pady=8)
    lines = queue.Queue()

    def collect():
        for key, (store, get) in variables.items():
            value = get()
            if store == 's':
                settings[key] = value
            elif store == 'i':
                ini[key] = value
            else:
                ini[key] = '1' if value else '0'
        try:
            settings['fps_limit'] = max(0, int(settings.get('fps_limit') or 0))
        except ValueError:
            settings['fps_limit'] = 0
        save_settings(settings)
        write_ini({key: ini[key] for key in ['upscaler', 'preset', 'output_res', 'model_lod',
                                            'anti_aliasing', 'low_latency', 'texture_quality'] +
                   [key for key, _t, _d in EFFECTS]})

    def append(text):
        log.configure(state='normal')
        log.insert('end', text + '\n')
        excess = int(log.index('end-1c').split('.')[0]) - MAX_LOG_LINES
        if excess > 0:
            log.delete('1.0', f'{excess}.0')
        log.see('end')
        log.configure(state='disabled')

    def pump():
        try:
            while True:
                item = lines.get_nowait()
                if item is None:
                    play_button.configure(state='normal')
                else:
                    append(item)
        except queue.Empty:
            pass
        root.after(100, pump)

    def start():
        collect()
        if not (Path(settings['game_dir']) / 'eboot.bin').is_file():
            messagebox.showerror('Bloodborne', 'В выбранной папке нет eboot.bin: выберите папку игры (CUSA03173).')
            return
        play_button.configure(state='disabled')

        def worker():
            import run
            original = run.log

            def to_window(line):
                original(line)
                lines.put(line)
            run.log = to_window
            try:
                play(settings)
            except Exception as error:  # the window must learn about any failure
                lines.put(f'STOP: {error!r}')
            finally:
                run.log = original
                lines.put(None)
        threading.Thread(target=worker, daemon=True).start()

    def open_logs():
        (DATA_DIR / 'logs').mkdir(exist_ok=True)
        os.startfile(str(DATA_DIR / 'logs'))  # noqa: S606 (Windows)

    play_button = ttk.Button(buttons, text='ИГРАТЬ', command=start)
    play_button.pack(side='right')
    ttk.Button(buttons, text='Сохранить', command=collect).pack(side='right', padx=6)
    ttk.Button(buttons, text='Журналы', command=open_logs).pack(side='left')
    root.after(100, pump)
    root.protocol('WM_DELETE_WINDOW', lambda: (collect(), root.destroy()))
    root.mainloop()


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == '--run-script':
        # A Python script of the package (tools/fsr4cap) run by this frozen interpreter: players
        # have no Python of their own.
        import runpy
        script = sys.argv[2]
        sys.argv = sys.argv[2:]
        sys.path.insert(0, str(Path(script).resolve().parent))
        try:
            runpy.run_path(script, run_name='__main__')
        except SystemExit as e:
            if isinstance(e.code, str):
                print(e.code, file=sys.stderr)
                return 1
            return e.code or 0
        return 0
    if len(sys.argv) >= 3 and sys.argv[1] == '--build-fsr411':
        # The WPF launcher's "build from an AMD DLL" (its log reads stdout): FSR 4.1.1 assets, or
        # FSR 4.1.2's driver models (tools/fsr4cap/build_assets_win.py).
        if sys.stdout:
            sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        sys.path.insert(0, str(PACKAGE_DIR / 'tools' / 'fsr4cap'))
        os.environ.setdefault('BB_DATA_DIR', str(DATA_DIR))
        import build_assets_win
        return build_assets_win.main(sys.argv[2:])
    if '--play' in sys.argv[1:]:
        # Run by the WPF launcher (bbport-run.exe --play, a console build): its log reads stdout.
        if sys.stdout:
            sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
        return play(load_settings())
    main_window()
    return 0


if __name__ == '__main__':
    sys.exit(main())
