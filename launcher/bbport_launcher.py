#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Bloodborne launcher for the native port (GTK4 / libadwaita).

Picks the game folder, edits the port's settings (bbport.ini: upscaler, preset, ...) and the
start-up tweaks passed as environment variables to run.sh, starts and stops the game and shows
its output. Launcher settings live in ~/.config/bbport-launcher/settings.json.
Russian, English, Brazilian Portuguese and Simplified Chinese (bbport_i18n: the Russian
text is the key).
"""

from bbport_launcher_common import *  # noqa: F401,F403
from bbport_launcher_content import ContentPages
from bbport_launcher_log import LogPage
from bbport_launcher_online import OnlinePage
from bbport_launcher_trophies import TrophiesPage
from bbport_launcher_upscalers import UpscalerFiles


# The window's pages and lists live in the bbport_launcher_*.py modules (mixins).
class LauncherWindow(ContentPages, OnlinePage, TrophiesPage, LogPage, UpscalerFiles,
                     Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="Bloodborne")
        self.set_default_size(1000, 760)
        self.settings = load_settings()
        set_language(self.settings.get("ui_language", ""))
        self.ini, self.ini_lines = load_ini()
        self.process = None
        self.stream = None
        self.fsr411_build = None  # tools/fsr4cap/build_assets.sh while it runs
        self.build()
        self.connect("close-request", self.on_close)

    def build(self):
        """(Re)creates the window's content in the current launcher language."""
        selected = getattr(self, "section_name", os.environ.get("BB_LAUNCHER_SECTION") or "game")
        log_text = self.log_text() if hasattr(self, "log_view") else ""
        self.stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        self.section_titles = {}
        self.sidebar = Gtk.ListBox(css_classes=["navigation-sidebar"])
        self.sidebar_rows = {}
        pages = [(section["id"], schema_text(section["title"]), SECTION_ICONS[section["id"]], page,
                  schema_text(section.get("description", {})))
                 for section, page in zip(LINUX_SECTIONS, self.build_settings_page())]
        pages.append(("trophies", tr("Трофеи"), "starred-symbolic", self.build_trophies_page(), ""))
        pages.append(("online", tr("Онлайн"), "network-workgroup-symbolic", self.build_online_page(), ""))
        pages.append(("log", tr("Журнал"), "utilities-terminal-symbolic", self.build_log_page(), ""))
        for name, title, icon, page, description in pages:
            if description and isinstance(page, Adw.PreferencesPage):
                page.set_description(description)
            self.stack.add_named(page, name)
            self.section_titles[name] = title
            box = Gtk.Box(spacing=12, margin_top=6, margin_bottom=6, margin_start=6, margin_end=6)
            box.append(Gtk.Image(icon_name=icon))
            box.append(Gtk.Label(label=title, xalign=0, hexpand=True))
            row = Gtk.ListBoxRow(child=box)
            row.section = name
            self.sidebar.append(row)
            self.sidebar_rows[name] = row
            if name == "log":
                row.set_margin_top(0)
        self.sidebar.connect("row-selected", self.on_section_selected)

        # Sidebar: the sections; content: the section, the launch button in its header bar.
        side_header = Adw.HeaderBar()
        # The port's version (VERSION, issue #307).
        if (PORT_DIR / "VERSION").is_file():
            version = Gtk.Label(label="bbport " + (PORT_DIR / "VERSION").read_text().strip())
            version.add_css_class("dim-label")
            side_header.pack_start(version)
        side = Adw.ToolbarView()
        side.add_top_bar(side_header)
        side.set_content(Gtk.ScrolledWindow(child=self.sidebar,
                                            hscrollbar_policy=Gtk.PolicyType.NEVER))
        self.content_header = Adw.HeaderBar()
        self.launch_button = Gtk.Button()
        self.launch_button.connect("clicked", self.on_launch)
        self.content_header.pack_end(self.launch_button)
        content = Adw.ToolbarView()
        content.add_top_bar(self.content_header)
        self.toasts = Adw.ToastOverlay()
        self.toasts.set_child(self.stack)
        content.set_content(self.toasts)

        self.split = Adw.NavigationSplitView(min_sidebar_width=200, max_sidebar_width=260)
        self.split.set_sidebar(Adw.NavigationPage(title="Bloodborne", child=side))
        self.content_page = Adw.NavigationPage(title="", child=content)
        self.split.set_content(self.content_page)
        # Narrow windows: one pane at a time (the sidebar first, a section opens over it).
        breakpoint = Adw.Breakpoint.new(Adw.BreakpointCondition.parse("max-width: 640sp"))
        breakpoint.add_setter(self.split, "collapsed", True)
        for old in list(getattr(self, "breakpoints", [])):
            self.remove_breakpoint(old)
        self.breakpoints = [breakpoint]
        self.add_breakpoint(breakpoint)
        self.set_content(self.split)

        self.log_view.get_buffer().set_text(log_text)
        self.select_section(selected if selected in self.sidebar_rows else "game")
        self.update_launch_button()
        self.update_game_status()
        self.update_user_status()
        self.update_upscaler_status()

    def on_section_selected(self, _box, row):
        if row is None:
            return
        self.section_name = row.section
        if row.section == "trophies":
            self.refresh_trophies()  # the game may have unlocked some since
        self.stack.set_visible_child_name(row.section)
        self.content_page.set_title(self.section_titles[row.section])
        self.split.set_show_content(True)

    def select_section(self, name):
        self.sidebar.select_row(self.sidebar_rows[name])

    def update_launch_button(self):
        running = self.process is not None
        self.launch_button.set_label(tr("Остановить") if running else tr("Запустить"))
        self.launch_button.remove_css_class("destructive-action" if not running else "suggested-action")
        self.launch_button.add_css_class("destructive-action" if running else "suggested-action")
        # The game and an FSR 4.1.1 build do not run together (both want the GPU, the build
        # replaces the assets the game reads).
        self.launch_button.set_sensitive(self.fsr411_build is None)
        if hasattr(self, "fsr411_button"):
            self.fsr411_button.set_sensitive(not running)
        if hasattr(self, "fsr412_button"):
            self.fsr412_button.set_sensitive(not running)

    # --- settings page -------------------------------------------------------------------

    def build_settings_page(self):
        """The settings, one page per section of launcher/settings_schema.json (in its order)."""
        sec = {section["id"]: Adw.PreferencesPage() for section in LINUX_SECTIONS}

        launcher = Adw.PreferencesGroup()
        self.ui_language_row = combo_row(tr("Язык лаунчера") + " / Launcher language", None,
                                         UI_LANGUAGES, self.settings.get("ui_language", ""))
        self.ui_language_row.connect("notify::selected", self.on_ui_language)
        launcher.add(self.ui_language_row)
        sec["game"].add(launcher)

        game = Adw.PreferencesGroup(title=tr("Игра"))
        self.game_row = Adw.ActionRow(title=tr("Папка игры (CUSA03173)"))
        self.game_status = Gtk.Image()
        self.game_row.add_suffix(self.game_status)
        self.game_row.add_suffix(flat_button("folder-open-symbolic", tr("Выбрать папку с eboot.bin"),
                                             self.on_choose_game))
        self.game_row.add_suffix(flat_button("system-file-manager-symbolic",
                                             tr("Открыть в файловом менеджере"),
                                             lambda _b: open_folder(self, self.game_dir())))
        game.add(self.game_row)
        # Saves and the shader cache: user/ in the data directory unless chosen.
        self.user_row = Adw.ActionRow(title=tr("Папка сохранений"))
        self.user_status = Gtk.Image()
        self.user_row.add_suffix(self.user_status)
        self.user_row.add_suffix(flat_button("folder-open-symbolic", tr("Выбрать папку сохранений"),
                                             self.on_choose_user))
        self.user_row.add_suffix(flat_button("system-file-manager-symbolic",
                                             tr("Открыть в файловом менеджере"),
                                             lambda _b: open_folder(self, self.user_dir())))
        self.user_reset = flat_button("edit-undo-symbolic", tr("Вернуть папку по умолчанию"),
                                      self.on_reset_user)
        self.user_row.add_suffix(self.user_reset)
        game.add(self.user_row)
        self.language_row = combo_row(tr("Язык системы"), None, LANGUAGES, self.settings["language"])
        game.add(self.language_row)
        self.text_language_row = combo_row(tr("Язык текста игры"), None, GAME_TEXT_LANGUAGES,
                                           self.ini.get("game_text_language", "auto"))
        game.add(self.text_language_row)
        self.voice_language_row = combo_row(tr("Язык озвучки"), None, GAME_VOICE_LANGUAGES,
                                            self.ini.get("game_voice_language", "auto"))
        game.add(self.voice_language_row)
        # A desktop and menu shortcut that starts the game directly (--play).
        shortcut_row = Adw.ActionRow(title=tr("Ярлык на рабочем столе"),
                                     subtitle=tr("Запускает игру сразу, с сохранёнными настройками, патчами и модами"))
        shortcut_button = Gtk.Button(label=tr("Создать"), valign=Gtk.Align.CENTER)
        shortcut_button.connect("clicked", lambda _b: self.on_create_shortcut())
        shortcut_row.add_suffix(shortcut_button)
        game.add(shortcut_row)
        sec["game"].add(game)

        mode = Adw.PreferencesGroup(title=tr("Видеокарта"))
        # The GPU the game runs on, as in shadPS4 (laptops with two GPUs).
        self.gpu_row = Adw.ComboRow(title=tr("Видеокарта"))
        self.gpu_row.add_suffix(flat_button("view-refresh-symbolic", tr("Обновить список"),
                                            lambda _button: self.fill_gpus()))
        self.fill_gpus()
        mode.add(self.gpu_row)

        self.mods_group = Adw.PreferencesGroup(
            title=tr("Моды"), description=tr(
                "Распакуйте каждый мод в отдельную папку (с dvdroot_ps4 или сразу с chr/, parts/ и т. п.). "
                "При совпадении файлов побеждает мод ниже в списке. Применяется при запуске."))
        self.mods_enabled_row = Adw.SwitchRow(title=tr("Загружать моды"),
                                               active=self.settings["mods_enabled"])
        self.mods_group.add(self.mods_enabled_row)
        self.mods_folder_row = Adw.ActionRow(title=tr("Папка модов"))
        self.mods_folder_row.add_suffix(flat_button("folder-open-symbolic", tr("Выбрать папку модов"),
                                                    self.on_choose_mods))
        self.mods_folder_row.add_suffix(flat_button("system-file-manager-symbolic", tr("Открыть папку модов"),
                                                    lambda _b: open_folder(self, self.mods_dir())))
        self.mods_folder_row.add_suffix(flat_button("view-refresh-symbolic", tr("Обновить список"),
                                                    self.on_refresh_mods))
        self.mods_group.add(self.mods_folder_row)
        self.mod_list = FolderList(self.mods_group)
        self.refresh_mods()
        sec["mods"].add(self.mods_group)

        self.patches_group = Adw.PreferencesGroup(
            title=tr("Сторонние патчи"),
            description=tr("XML-патчи в формате shadPS4 для версии 01.09 из папки патчей. "
                           "Применяются при запуске."))
        self.patches_folder_row = Adw.ActionRow(title=tr("Папка патчей"))
        self.patches_folder_row.add_suffix(flat_button("folder-open-symbolic", tr("Выбрать папку патчей"),
                                                       self.on_choose_patches))
        self.patches_folder_row.add_suffix(flat_button(
            "system-file-manager-symbolic", tr("Открыть папку патчей"),
            lambda _b: open_folder(self, patches_dir(self.settings))))
        self.patches_folder_row.add_suffix(flat_button("view-refresh-symbolic", tr("Обновить список"),
                                                       self.on_refresh_patches))
        self.patches_group.add(self.patches_folder_row)
        self.patch_list = FolderList(self.patches_group)
        self.refresh_patches()
        sec["mods"].add(self.patches_group)

        screen = Adw.PreferencesGroup(title=tr("Экран"))
        # The aspect ratio, then its resolutions (rebuilt when it changes, the height kept).
        self.aspect_row = Adw.ComboRow(title=tr("Соотношение сторон"), subtitle=tr(
            "Шире 16:9 — больше обзор по бокам, выше — сверху и снизу; интерфейс у краёв экрана"))
        self.aspect_row.values = ASPECTS
        self.output_row = Adw.ComboRow(title=tr("Разрешение вывода"),
                                       subtitle=tr("Апскейлер дорисовывает кадр; Steam Deck — 1280×800 (16:10)"))
        self.output_row.values = []
        screen.add(self.aspect_row)
        screen.add(self.output_row)
        self.fullscreen_row = Adw.SwitchRow(title=tr("Полноэкранный режим"),
                                            active=self.settings["fullscreen"])
        screen.add(self.fullscreen_row)
        # Issue #69: the window (and fullscreen) went to the monitor SDL calls primary.
        self.display_row = Adw.ComboRow(title=tr("Монитор"))
        self.display_row.add_suffix(flat_button("view-refresh-symbolic", tr("Обновить список"),
                                                lambda _button: self.fill_displays()))
        self.fill_displays()
        screen.add(self.display_row)
        aspect = self.ini.get("aspect_ratio", "auto")
        self.fill_aspects(aspect if aspect in ASPECTS else "auto")
        # Auto (output_res=auto, and new settings): the monitor's own resolution.
        saved_output = self.ini.get("output_res", "auto")
        self.fill_outputs(parse_size(saved_output), saved_output == "auto")
        self.aspect_row.connect("notify::selected",
                                lambda *_: self.fill_outputs(*self.output_choice()))
        self.display_row.connect("notify::selected", lambda *_: self.on_display_changed())
        self.present_row = combo_row(tr("Режим показа кадров"), None, PRESENT_MODES,
                                     self.settings["present_mode"])
        screen.add(self.present_row)
        self.hdr_row = Adw.SwitchRow(title=tr("Разрешить HDR"), active=self.settings["hdr"])
        screen.add(self.hdr_row)
        sec["display"].add(screen)

        # Issue #15: the first gamepad SDL found was taken (wheels and other controllers too).
        controls = Adw.PreferencesGroup(title=tr("Управление"))
        self.gamepad_row = Adw.ComboRow(title=tr("Контроллер"))
        # Controller names are long: the selected one is shown in full under the title.
        self.gamepad_row.connect("notify::selected", lambda *_: self.show_gamepad())
        self.gamepad_row.add_suffix(flat_button("view-refresh-symbolic", tr("Обновить список"),
                                                lambda _button: self.fill_gamepads()))
        self.fill_gamepads()
        controls.add(self.gamepad_row)
        # Bindings: "Assign" waits for a key or button (bb-gpu-capabilities --read-input).
        self.control_rows = {}
        for kind, title, icon in (("key", tr("Клавиатура и мышь"), "input-keyboard-symbolic"),
                                  ("pad", tr("Геймпад"), "input-gaming-symbolic")):
            expander = Adw.ExpanderRow(title=title,
                                       subtitle=tr("Назначение кнопок; применяется при запуске игры"))
            for name, label, key_default, pad_default in CONTROLS:
                default = key_default if kind == "key" else pad_default
                if default is None:
                    continue
                row = Adw.ActionRow(title=tr(label))
                row.add_suffix(flat_button(icon, tr("Назначить"),
                                           lambda _b, k=kind, n=name: self.assign_control(k, n)))
                row.add_suffix(flat_button("list-add-symbolic", tr("Добавить ещё одну"),
                                           lambda _b, k=kind, n=name: self.assign_control(k, n, True)))
                row.add_suffix(flat_button("edit-undo-symbolic", tr("Сбросить"),
                                           lambda _b, k=kind, n=name: self.set_control(k, n, None)))
                expander.add_row(row)
                self.control_rows[(kind, name)] = (row, default)
                self.show_control(kind, name)
            controls.add(expander)
        # Mouse look (runtime_pad.c): a click in the game takes the mouse, F1 lets it go; the mouse
        # turns the game camera itself (gpu/games/bloodborne/hooks_camera.cpp, BB_NATIVE_MOUSE).
        self.mouse_look_row = Adw.SwitchRow(
            title=tr("Камера мышью"),
            subtitle=tr("Щелчок в окне игры захватывает мышь, F1 — отпускает"),
            active=self.ini.get("mouse_look", "1") == "1")
        controls.add(self.mouse_look_row)
        self.mouse_sensitivity_row = Adw.SpinRow.new_with_range(0.1, 10.0, 0.05)
        self.mouse_sensitivity_row.set_title(tr("Чувствительность мыши"))
        self.mouse_sensitivity_row.set_digits(2)
        self.mouse_sensitivity_row.set_value(float(self.ini.get("mouse_sensitivity", "1.0")))
        controls.add(self.mouse_sensitivity_row)
        self.mouse_invert_row = Adw.SwitchRow(title=tr("Инвертировать мышь по вертикали"),
                                              active=self.ini.get("mouse_invert_y") == "1")
        controls.add(self.mouse_invert_row)
        self.mouse_invert_x_row = Adw.SwitchRow(title=tr("Инвертировать мышь по горизонтали"),
                                                active=self.ini.get("mouse_invert_x") == "1")
        controls.add(self.mouse_invert_x_row)
        def mouse_rows_sensitive(*_):
            for row in (self.mouse_sensitivity_row, self.mouse_invert_row, self.mouse_invert_x_row):
                row.set_sensitive(self.mouse_look_row.get_active())
        self.mouse_look_row.connect("notify::active", mouse_rows_sensitive)
        mouse_rows_sensitive()
        # The right stick's camera speed per axis (#208): the game turns both at the same angular
        # speed, which looks faster vertically (the view is narrower than it is wide).
        self.camera_speed_rows = {}
        for key, title in (("camera_speed_x", "Скорость камеры (стик) по горизонтали"),
                           ("camera_speed_y", "Скорость камеры (стик) по вертикали")):
            row = Adw.SpinRow.new_with_range(0.25, 2.0, 0.05)
            row.set_title(tr(title))
            row.set_digits(2)
            row.set_value(float(self.ini.get(key, "1.0")))
            controls.add(row)
            self.camera_speed_rows[key] = row
        self.prompts_row = combo_row(tr("Значки кнопок"), None, BUTTON_PROMPTS,
                                     self.ini.get("button_prompts", "auto"))
        controls.add(self.prompts_row)
        sec["controls"].add(controls)

        upscaler = Adw.PreferencesGroup(title=tr("Апскейлер"))
        libraries = Adw.PreferencesGroup(title=tr("Библиотеки AMD и NVIDIA"), description=tr(
            "Свои файлы AMD и NVIDIA: порт их не распространяет"))
        quality = Adw.PreferencesGroup(title=tr("Качество изображения"))
        self.upscaler_row = combo_row(tr("Апскейлер"), None, UPSCALERS, self.ini["upscaler"])
        self.upscaler_row.connect("notify::selected", lambda *_: self.update_upscaler_status())
        upscaler.add(self.upscaler_row)
        # FSR 4.1.1 is built from the user's own AMD DLL (nothing of AMD's is shipped).
        self.fsr411_row = Adw.ActionRow(title=tr("FSR 4.1.1 из своей DLL AMD"))
        self.fsr411_button = Gtk.Button(valign=Gtk.Align.CENTER)
        self.fsr411_button.connect("clicked", self.on_fsr411_button)
        self.fsr411_row.add_suffix(self.fsr411_button)
        self.fsr411_row.set_visible((FSR4CAP_DIR / "build_assets.sh").is_file())
        libraries.add(self.fsr411_row)
        # FSR 4.1.2's models (INT8 RDNA2, INT8 Lite) from AMD's driver, also the user's own.
        self.fsr412_row = Adw.ActionRow(title=tr("FSR 4.1.2 из драйвера AMD"))
        self.fsr412_button = Gtk.Button(valign=Gtk.Align.CENTER, label=tr("Выбрать файл…"))
        self.fsr412_button.connect("clicked", self.on_fsr412_button)
        self.fsr412_row.add_suffix(self.fsr412_button)
        self.fsr412_row.set_subtitle(tr(
            "Модели INT8 RDNA2 и INT8 Lite: установщик AMD Adrenalin 26.10.1+ (.exe) или "
            "amdxcffx64.dll из него. Нужна и DLL FSR 4.1.x (как для FSR 4.1.1); ~4 минуты. "
            "Модель выбирается в меню игры"))
        self.fsr412_row.set_visible((FSR4CAP_DIR / "driver_model.py").is_file())
        libraries.add(self.fsr412_row)
        self.update_fsr411_row()
        # DLSS: NVIDIA's library, chosen like the FSR 4.1.1 DLL (the package may bring one).
        self.dlss_row = Adw.ActionRow(title=tr("DLSS: библиотека NVIDIA"))
        self.dlss_row.add_suffix(flat_button("document-open-symbolic", tr("Выбрать файл…"),
                                             lambda _b: self.choose_ngx()))
        self.dlss_reset_button = flat_button("edit-undo-symbolic", tr("Убрать выбранную"),
                                             lambda _b: self.reset_ngx())
        self.dlss_row.add_suffix(self.dlss_reset_button)
        libraries.add(self.dlss_row)
        self.update_dlss_row()
        # Frame generation (the game's menu has it too): FSR 3.1 on any GPU (experimental), DLSS
        # where an NVIDIA GPU is connected (or the GPUs are unknown).
        vendors = {vendor.lower() for _index, vendor, _kind, _name, _best in connected_gpus()}
        nvidia = not vendors or any("10de" in vendor for vendor in vendors)
        frame_gens = [FRAME_GENS[0], ("FSR 3.1 (экспериментально)", "fsr3")]
        frame_gens += [entry for entry in FRAME_GENS[1:] if nvidia or self.ini.get("frame_generation") == "dlss"]
        self.frame_gen_row = combo_row(
            tr("Генерация кадров"),
            tr("Кадры между кадрами игры. FSR 3.1 — на любой видеокарте, пока экспериментально "
               "(возможны артефакты по краям). DLSS: нужны мост DLSS, libnvidia-ngx-dlssg.so от NVIDIA "
               "(выбирается выше, как библиотека DLSS) и GeForce RTX 40 или новее; 3x и 4x — RTX 50"),
            frame_gens, "fsr3" if self.ini.get("frame_generation") == "fsr3" else frame_gen_choice(self.ini))
        upscaler.add(self.frame_gen_row)
        # Low latency (Reflex, Anti-Lag 2): with DLSS frame generation at least on (the game enforces
        # it; the row follows so that what is shown is what runs).
        low_latency = self.ini.get("low_latency", "on")
        self.low_latency_row = combo_row(
            tr("Низкая задержка (NVIDIA Reflex / AMD Anti-Lag 2)"),
            tr("Поток игры ждёт драйвер перед каждым кадром, чтобы кадры не копились перед "
               "видеокартой: меньше задержка ввода. NVIDIA: Reflex, драйвер 550 и новее; AMD и Intel: "
               "Anti-Lag 2 (Boost — то же, что «Включена»). С генерацией кадров DLSS всегда не ниже "
               "«Включена»"),
            LOW_LATENCIES, low_latency if low_latency in ("on", "boost", "off") else "on")

        def enforce_low_latency(*_):
            if combo_value(self.frame_gen_row).startswith("dlss") and combo_value(self.low_latency_row) == "off":
                self.low_latency_row.set_selected(0)  # "on"
        self.frame_gen_row.connect("notify::selected", enforce_low_latency)
        self.low_latency_row.connect("notify::selected", enforce_low_latency)
        enforce_low_latency()
        self.preset_row = combo_row(tr("Пресет"), None, PRESETS, int(self.ini.get("preset", "4")))
        self.preset_row.connect("notify::selected", lambda *_: self.update_upscaler_status())
        self.output_row.connect("notify::selected", lambda *_: self.update_upscaler_status())
        upscaler.add(self.preset_row)
        self.sharpen_row = Adw.SwitchRow(title=tr("Резкость (RCAS)"),
                                         subtitle=tr("И без апскейлера: поверх сглаживания игры или FXAA"),
                                         active=self.ini.get("sharpen") == "1")
        upscaler.add(self.sharpen_row)
        self.sharpness_row = Adw.SpinRow.new_with_range(0.0, 2.0, 0.05)
        self.sharpness_row.set_title(tr("Сила резкости"))
        self.sharpness_row.set_digits(2)
        self.sharpness_row.set_value(float(self.ini.get("sharpness", "0.5")))
        upscaler.add(self.sharpness_row)
        self.motion_row = Adw.SwitchRow(
            title=tr("Векторы движения объектов"),
            subtitle=tr("Меньше гостинга на персонажах; стоит около 10% FPS"),
            active=self.ini.get("object_motion") == "1")
        quality.add(self.motion_row)
        self.show_fps_row = Adw.SwitchRow(title=tr("Показывать FPS"),
                                          active=self.ini.get("show_fps") == "1")
        quality.add(self.show_fps_row)
        self.vbao_row = Adw.SwitchRow(title=tr("Затенение GT-VBAO"),
                                      subtitle=tr("Вместо SSAO игры (нужно затенение SSAO)"),
                                      active=self.ini.get("vbao") == "1")
        quality.add(self.vbao_row)
        saved_aa = self.ini.get("anti_aliasing") or ("off" if self.ini.get("effect_game_aa") == "0" else "auto")
        self.aa_row = combo_row(tr("Сглаживание"), None, ANTI_ALIASING, saved_aa)
        quality.add(self.aa_row)
        sec["graphics"].add(upscaler)
        sec["graphics"].add(quality)
        sec["graphics"].add(libraries)

        effects = Adw.PreferencesGroup(title=tr("Эффекты игры"),
                                       description=tr("Патчи игры, применяются при запуске"))
        self.lod_row = combo_row(tr("Детализация моделей"), None, MODEL_LOD,
                                 self.ini.get("model_lod", "0"))
        quality.add(self.lod_row)
        self.texture_row = combo_row(tr("Качество текстур"), tr("Крупные текстуры в видеопамяти"),
                                     TEXTURE_QUALITY, self.ini.get("texture_quality", "auto"))
        quality.add(self.texture_row)
        self.effect_rows = {}
        for key, title, default in EFFECTS:
            row = Adw.SwitchRow(title=tr(title),
                                active=self.ini.get(key, "1" if default else "0") == "1")
            if key == "debug_menu":
                row.set_subtitle(tr("Нужна папка adhoc из мода Nexus #253 (шрифты adhoc/font) — в "
                                    "dvdroot_ps4 игры или модом; без неё патч не применяется"))
            self.effect_rows[key] = row
            effects.add(row)
        # The title's "play online / play offline": the port has no PSN.
        self.skip_network_row = Adw.SwitchRow(
            title=tr("Пропускать выбор «по сети / вне сети»"),
            subtitle=tr("Игра сразу открывает главное меню в режиме вне сети (патч игры)"),
            active=self.settings.get("skip_network_choice", True))
        game.add(self.skip_network_row)
        sec["effects"].add(effects)

        frames = Adw.PreferencesGroup(title=tr("Частота кадров"))
        self.fps_row = combo_row(tr("Режим"), tr("Какой патч частоты кадров применить к игре"),
                                 FPS_MODES, self.settings["fps_mode"])
        frames.add(self.fps_row)
        self.limit_row = Adw.SpinRow.new_with_range(0, 480, 1)
        self.limit_row.set_title(tr("Ограничение FPS"))
        self.limit_row.set_subtitle(tr("0 — без ограничения; укажите число, чтобы ограничить FPS"))
        self.limit_row.set_value(self.settings["fps_limit"])
        frames.add(self.limit_row)
        sec["performance"].add(frames)

        perf = Adw.PreferencesGroup(title=tr("Задержка и конвейер"))
        self.pipe_row = combo_row(
            tr("Двухстадийный конвейер GPU"),
            tr("Быстрее на 20–30%; при нестабильности выключите"), DRAW_PIPE,
            self.settings["draw_pipe"])
        perf.add(self.low_latency_row)
        perf.add(self.pipe_row)
        sec["performance"].add(perf)
        sec["performance"].add(mode)

        dev = Adw.PreferencesGroup(title=tr("Для разработчика"))
        self.mangohud_row = Adw.SwitchRow(title="MangoHud", active=self.settings["mangohud"])
        dev.add(self.mangohud_row)
        self.save_log_row = Adw.SwitchRow(
            title=tr("Сохранять журнал и статистику в файл"),
            subtitle=tr("В папку logs в каталоге данных: для разбора рывков и вылетов"),
            active=self.settings.get("save_log", False))
        dev.add(self.save_log_row)
        self.crash_diag_row = Adw.SwitchRow(
            title=tr("Диагностика вылетов"),
            subtitle=tr("Проверяет кучу игры и записывает записи в её память; немного медленнее"),
            active=self.settings.get("crash_diag", False))
        dev.add(self.crash_diag_row)
        self.stats_row = Adw.SwitchRow(title=tr("Статистика кадров в журнале"),
                                       subtitle="BB_FRAME_STATS",
                                       active=self.settings["frame_stats"])
        dev.add(self.stats_row)
        self.profile_row = Adw.SwitchRow(title=tr("Профиль GPU в журнале"),
                                         subtitle="BB_GPU_PROFILE",
                                         active=self.settings["gpu_profile"])
        dev.add(self.profile_row)
        self.validation_row = Adw.SwitchRow(title=tr("Слои валидации Vulkan"),
                                            subtitle=tr("Сильно замедляет"),
                                            active=self.settings["vk_validation"])
        dev.add(self.validation_row)
        self.extra_row = Adw.EntryRow(title=tr("Доп. переменные (ИМЯ=значение через пробел)"),
                                      text=self.settings["extra_env"])
        dev.add(self.extra_row)
        sec["developer"].add(dev)
        self.describe_rows()
        return list(sec.values())

    def describe_rows(self):
        """Each row's description (and its choices' notes) from launcher/settings_schema.json."""
        rows = {
            "game_dir": self.game_row, "user_dir": self.user_row, "language": self.language_row,
            "game_text_language": self.text_language_row, "game_voice_language": self.voice_language_row,
            "skip_network_choice": self.skip_network_row, "display": self.display_row,
            "fullscreen": self.fullscreen_row, "aspect_ratio": self.aspect_row,
            "output_res": self.output_row, "present_mode": self.present_row, "hdr": self.hdr_row,
            "upscaler": self.upscaler_row, "preset": self.preset_row, "anti_aliasing": self.aa_row,
            "sharpen": self.sharpen_row, "sharpness": self.sharpness_row,
            "texture_quality": self.texture_row, "model_lod": self.lod_row,
            "object_motion": self.motion_row, "vbao": self.vbao_row,
            "frame_generation": self.frame_gen_row, "show_fps": self.show_fps_row,
            "fps_mode": self.fps_row, "fps_limit": self.limit_row, "low_latency": self.low_latency_row,
            "gpu": self.gpu_row, "draw_pipe": self.pipe_row, "gamepad": self.gamepad_row,
            "mouse_look": self.mouse_look_row, "mouse_sensitivity": self.mouse_sensitivity_row,
            "mouse_invert_y": self.mouse_invert_row, "mouse_invert_x": self.mouse_invert_x_row, "button_prompts": self.prompts_row,
            "mods_enabled": self.mods_enabled_row, "mangohud": self.mangohud_row,
            "save_log": self.save_log_row, "crash_diag": self.crash_diag_row,
            "frame_stats": self.stats_row, "gpu_profile": self.profile_row,
            "vk_validation": self.validation_row, **self.camera_speed_rows, **self.effect_rows,
        }
        # Rows whose subtitle shows a state (a path, a device): the description is a tooltip.
        stateful = {"game_dir", "user_dir", "gpu", "gamepad", "display"}
        for section in SETTINGS_SCHEMA["sections"]:
            for item in section["items"]:
                row = rows.get(item["key"])
                text = schema_text(item.get("description", {}))
                if row is None or not text:
                    continue
                if item["key"] in stateful:
                    row.set_tooltip_text(text)
                elif hasattr(row, "show_note"):
                    row.description = text
                    notes = {choice["value"]: schema_text(choice["description"])
                             for choice in item.get("choices", []) if "description" in choice}
                    if notes:
                        row.notes = [notes.get(value) for value in row.values]
                    row.show_note()
                elif item["key"] == "debug_menu" and row.get_subtitle():
                    continue  # its own note names the fonts it needs
                else:
                    row.set_subtitle(text)

    def on_ui_language(self, row, _param):
        choice = combo_value(row)
        if choice == self.settings.get("ui_language", ""):
            return
        self.store()
        self.settings["ui_language"] = choice
        save_settings(self.settings)
        if self.process:
            # The log keeps streaming into the current page; rebuild on the next start.
            set_language(choice)
            self.toasts.add_toast(Adw.Toast(
                title=tr("Применится после перезапуска лаунчера, пока игра запущена")))
            return
        set_language(choice)
        # After this handler returns: the row that emitted the signal is replaced.
        GLib.idle_add(lambda: self.build() and False)

    def game_dir(self):
        return Path(self.settings["game_dir"]).expanduser()

    def user_dir(self):
        return Path(self.settings["user_dir"]).expanduser() if self.settings["user_dir"] \
            else DATA_DIR / "user"

    def update_user_status(self):
        path = self.user_dir()
        custom = bool(self.settings["user_dir"])
        self.user_row.set_subtitle(str(path) if custom else tr("По умолчанию: {}").format(path))
        saves = path / "savedata"
        found = saves.is_dir() and any(saves.iterdir())
        self.user_status.set_from_icon_name("object-select-symbolic" if found else "document-new-symbolic")
        self.user_status.set_tooltip_text(tr("Найдены сохранения") if found
                                          else tr("Сохранений пока нет: игра создаст их здесь"))
        self.user_reset.set_sensitive(custom)

    def choose_folder(self, title, current, done):
        dialog = Gtk.FileDialog(title=title)
        if Path(current).is_dir():
            dialog.set_initial_folder(Gio.File.new_for_path(str(current)))

        def finish(dialog, result):
            try:
                folder = dialog.select_folder_finish(result)
            except GLib.Error:
                return
            if folder:
                done(folder.get_path())
        dialog.select_folder(self, None, finish)

    def on_choose_user(self, _button):
        def chosen(path):
            self.settings["user_dir"] = path
            self.update_user_status()
            self.store()
        self.choose_folder(tr("Папка сохранений"), self.user_dir(), chosen)

    def on_reset_user(self, _button):
        self.settings["user_dir"] = ""
        self.update_user_status()
        self.store()

    def update_upscaler_status(self):
        value = combo_value(self.upscaler_row)
        if value == "fsr4":
            ok = (PORT_DIR / "fsr4_shaders").is_dir()
            hint = tr("Ассеты найдены") if ok else tr("Нет ассетов: tools/fetch_fsr4_assets.sh")
        elif value == "fsr411":
            directory = fsr411_dir()
            problem = fsr411_problem(directory, self.output_size_now(),
                                     int(combo_value(self.preset_row)))
            hint = tr("Ассеты для выбранного режима найдены") if not problem else (
                tr("{}. Соберите FSR 4.1.1 из своей DLL кнопкой «Выбрать DLL…» ниже").format(problem))
        elif value == "dlss":
            # The game looks for the bridge and NVIDIA's library next to bb-probe (vk_dlss.cpp); the
            # AppImage has neither (NVIDIA's DLSS SDK is not bundled): DLSS then falls back to FSR 3.1.
            binaries = PORT_DIR / ("bin" if PACKAGED else "out")
            found = (binaries / "libbbport_dlss.so").is_file() and bool(
                ngx_libraries(self.user_dir() / "dlss") or ngx_libraries(binaries))
            hint = tr("DLSS найден (нужна видеокарта NVIDIA RTX)") if found else tr(
                "DLSS не найден: выберите библиотеку NVIDIA ниже (и нужна сборка с мостом DLSS). Игра включит FSR 3.1")
        else:
            hint = tr("Сглаживание в разрешении вывода без модели FSR") if value == "taa" else None
        self.preset_row.set_sensitive(value not in ("taa", "off"))
        # Sharpening works without an upscaler too (RCAS on the finished frame).
        self.sharpen_row.set_sensitive(True)
        self.sharpness_row.set_sensitive(True)
        self.upscaler_row.hint = hint
        self.upscaler_row.show_note()

    def game_problem(self, path):
        """game_check.problem for the folder, kept until its eboot.bin or param.sfo changes (the
        image hash reads ~90 MB)."""
        try:
            key = (str(path),) + tuple(
                (f.stat().st_size, f.stat().st_mtime_ns) for f in (path / "eboot.bin", path / "sce_sys/param.sfo"))
        except OSError:
            key = (str(path),)
        if getattr(self, "_game_problem_key", None) != key:
            self._game_problem_key = key
            self._game_problem = game_check.problem(path)
        return self._game_problem

    def update_game_status(self):
        path = self.game_dir()
        ok = bool(self.settings["game_dir"]) and (path / "eboot.bin").is_file()
        self.game_row.set_subtitle(str(path) if self.settings["game_dir"] else tr("не выбрана"))
        # Other versions of the game start and then crash in its code (issues #7, #13, #14).
        problem = self.game_problem(path) if ok else None
        if not ok:
            tooltip = tr("Нет eboot.bin в папке")
        elif problem:
            kind, title, version = problem
            tooltip = {
                "missing_update": tr("Нужно обновление 1.09: скопируйте файлы дампа обновления 1.09 в папку игры с заменой (найдена версия {})").format(version),
                "wrong_eboot": tr("eboot.bin не от версии 1.09: скопируйте eboot.bin из дампа обновления 1.09 в папку игры с заменой"),
                "patched_eboot": tr("eboot.bin с вшитым патчем 60 FPS от Lance McDonald: из-за него игра падает в меню жестов. Скопируйте чистый eboot.bin из дампа обновления 1.09 в папку игры с заменой"),
                "other_title": tr("Это не магазинное издание Bloodborne (найдено {}): нужен Bloodborne любого региона с обновлением 1.09").format(title),
                "unreadable": tr("eboot.bin не читается как расшифрованный исполняемый файл PS4: сделайте дамп заново"),
                "damaged_files": tr("Файлы игры повреждены при распаковке: шейдеры не распаковываются, игра зависнет на загрузке. Распакуйте игру и обновление 1.09 заново исправленным инструментом (issue #81)"),
            }[kind]
            self.game_row.set_subtitle(f"{path}\n{tooltip}")
        else:
            tooltip = tr("Bloodborne CUSA03173, версия 1.09")
        self.game_status.set_from_icon_name("object-select-symbolic" if ok and not problem else "dialog-warning-symbolic")
        self.game_status.set_tooltip_text(tooltip)
        self.launch_button.set_sensitive(ok or self.process is not None)

    def on_choose_game(self, _button):
        def chosen(path):
            self.settings["game_dir"] = path
            self.update_game_status()
            self.store()
        self.choose_folder(tr("Папка игры (с eboot.bin)"), self.game_dir(), chosen)

    def show_control(self, kind, name):
        row, default = self.control_rows[(kind, name)]
        value = self.ini.get(f"{kind}.{name}")
        if value is None:
            row.set_subtitle(tr("{} (по умолчанию)").format(default) if default else tr("не назначено"))
        else:
            row.set_subtitle(value or tr("не назначено"))

    def set_control(self, kind, name, value):
        """value: the binding, or None for the default."""
        self.ini[f"{kind}.{name}"] = value
        self.show_control(kind, name)
        self.store()

    def assign_control(self, kind, name, append=False):
        """The next key or button pressed: the binding, or (append) one more for the same action
        (runtime_pad.c takes up to four per action, chords like "Left Shift+Mouse Left" too)."""
        tool = PORT_DIR / ("bin" if PACKAGED else "out") / "bb-gpu-capabilities"
        try:
            process = Gio.Subprocess.new([str(tool), "--read-input", kind],
                                         Gio.SubprocessFlags.STDOUT_PIPE)
        except GLib.Error as error:
            self.toasts.add_toast(Adw.Toast(title=tr("Не удалось запустить: {}").format(error.message)))
            return
        row, _default = self.control_rows[(kind, name)]
        row.set_subtitle(tr("Нажмите клавишу или кнопку… (Esc — отмена)"))

        def done(proc, result):
            try:
                _ok, out, _err = proc.communicate_utf8_finish(result)
            except GLib.Error:
                out = ""
            words = (out or "").strip().split(" ", 1)
            if len(words) == 2 and words[0] == kind:
                value = words[1]
                if append:
                    current = self.ini.get(f"{kind}.{name}")
                    current = _default if current is None else current
                    keys = [key.strip() for key in (current or "").split(",") if key.strip()]
                    if value not in keys:
                        keys = (keys + [value])[-4:]
                    value = ", ".join(keys)
                self.set_control(kind, name, value)
            else:
                self.show_control(kind, name)
        process.communicate_utf8_async(None, None, done)

    def fill_gamepads(self):
        """The controller choices: the first connected one, the connected ones, and the saved
        choice while it is not connected."""
        current = self.settings.get("gamepad", "")
        choices = [(tr("Авто"), "", tr("Первый подключённый"))]
        choices += [(name, guid, name) for guid, name in connected_gamepads()]
        if current and current not in [guid for _, guid, _ in choices]:
            name = self.settings.get("gamepad_name") or current
            choices.append((tr("{} (не подключён)").format(name), current, name))
        self.gamepad_row.set_model(Gtk.StringList.new([label for label, _, _ in choices]))
        self.gamepad_row.values = [guid for _, guid, _ in choices]
        self.gamepad_row.names = [name for _, _, name in choices]
        self.gamepad_row.set_selected(self.gamepad_row.values.index(current) if current in self.gamepad_row.values else 0)
        self.show_gamepad()

    def monitor_size(self):
        """The size of the monitor chosen for the game (Auto's aspect ratio), None if unknown."""
        value = combo_value(self.display_row) if self.display_row.values else ""
        sizes = getattr(self, "display_sizes", {})
        return sizes.get(value) or sizes.get("")

    def fill_aspects(self, current):
        labels = [tr("Авто") + f" ({resolved_aspect('auto', self.monitor_size())})"] + ASPECTS[1:]
        self.aspect_row.set_model(Gtk.StringList.new(labels))
        self.aspect_row.set_selected(ASPECTS.index(current))

    def fill_outputs(self, keep, auto=False):
        """Auto (the monitor's resolution) and the chosen aspect ratio's resolutions; Auto or the
        one of `keep`'s height selected."""
        sizes = output_sizes(combo_value(self.aspect_row), self.monitor_size())
        chosen = nearest_output(sizes, keep)
        monitor = self.monitor_size()
        auto_label = tr("Авто (монитор)") + (f" {monitor[0]}×{monitor[1]}" if monitor else "")
        self.output_row.values = ["auto"] + [f"{w}x{h}" for w, h in sizes]
        self.output_row.set_model(Gtk.StringList.new(
            [auto_label] + [f"{w}×{h}" + (" (Steam Deck)" if (w, h) == (1280, 800) else "") for w, h in sizes]))
        self.output_row.set_selected(0 if auto else 1 + sizes.index(chosen))

    def output_choice(self):
        """(size, auto) of the output row: Auto's size is the monitor's."""
        value = combo_value(self.output_row)
        if value == "auto":
            return self.monitor_size() or (1920, 1080), True
        return parse_size(value), False

    def output_size_now(self):
        size, _auto = self.output_choice()
        return f"{size[0]}x{size[1]}"

    def on_display_changed(self):
        aspect = combo_value(self.aspect_row)
        output, auto = self.output_choice()
        self.fill_aspects(aspect)
        self.fill_outputs(output, auto)

    def fill_displays(self):
        """The monitor choices: the primary one, the connected ones, and the saved choice while it
        is not connected."""
        current = self.settings.get("display", "")
        displays = connected_displays()
        self.display_sizes = {value: size for value, _label, size, _primary in displays}
        primary = [size for _value, _label, size, is_primary in displays if is_primary]
        if primary:
            self.display_sizes[""] = primary[0]
        choices = [(tr("Основной"), "")] + [(label, value) for value, label, _size, _primary in displays]
        if current and current not in [value for _, value in choices]:
            choices.append((tr("{} (не подключён)").format(current), current))
        self.display_row.set_model(Gtk.StringList.new([label for label, _ in choices]))
        self.display_row.values = [value for _, value in choices]
        self.display_row.set_selected(self.display_row.values.index(current) if current in self.display_row.values else 0)

    def fill_gpus(self):
        """The GPU choices: the port's own (named), the GPUs found, and the saved one while it is
        not found."""
        current = self.settings.get("gpu", "")
        gpus = connected_gpus()
        best = [name for _index, _vendor, _kind, name, is_best in gpus if is_best]
        auto = tr("Авто") + (f" ({best[0]})" if best else "")
        choices = [(auto, "")]
        for _index, _vendor, kind, name, _best in gpus:
            note = {"integrated": tr("встроенная"), "cpu": tr("программная, очень медленно")}.get(kind)
            choices.append((f"{name} ({note})" if note else name, name))
        if current and current not in [value for _, value in choices]:
            choices.append((tr("{} (не найдена)").format(current), current))
        self.gpu_row.set_model(Gtk.StringList.new([label for label, _ in choices]))
        self.gpu_row.values = [value for _, value in choices]
        self.gpu_row.set_selected(self.gpu_row.values.index(current) if current in self.gpu_row.values else 0)
        self.gpu_row.set_subtitle(tr("На ноутбуке с двумя видеокартами — дискретная; применяется при запуске игры"))

    def show_gamepad(self):
        names = getattr(self.gamepad_row, "names", None)
        selected = names[self.gamepad_row.get_selected()] if names else ""
        self.gamepad_row.set_subtitle("\n".join(
            line for line in (selected, tr("Выбранный берётся, как только подключится")) if line))

    def store(self):
        s = self.settings
        s["gamepad"] = combo_value(self.gamepad_row)
        s["gamepad_name"] = self.gamepad_row.names[self.gamepad_row.get_selected()]
        s["display"] = combo_value(self.display_row)
        s["gpu"] = combo_value(self.gpu_row)
        s["skip_network_choice"] = self.skip_network_row.get_active()
        s["language"] = combo_value(self.language_row)
        s["fullscreen"] = self.fullscreen_row.get_active()
        s["present_mode"] = combo_value(self.present_row)
        s["hdr"] = self.hdr_row.get_active()
        s["fps_mode"] = combo_value(self.fps_row)
        s["fps_limit"] = int(self.limit_row.get_value())
        s["draw_pipe"] = combo_value(self.pipe_row)
        s["mangohud"] = self.mangohud_row.get_active()
        s["frame_stats"] = self.stats_row.get_active()
        s["save_log"] = self.save_log_row.get_active()
        s["crash_diag"] = self.crash_diag_row.get_active()
        s["gpu_profile"] = self.profile_row.get_active()
        s["vk_validation"] = self.validation_row.get_active()
        s["extra_env"] = self.extra_row.get_text().strip()
        s["trophy_key"] = self.trophy_key_row.get_text().strip()
        s["mods_enabled"] = self.mods_enabled_row.get_active()
        s["online"] = self.online_row.get_active()
        s["online_bbsc"] = self.bbsc_row.get_active()
        s["online_community"] = self.community_row.get_text().strip()
        s["online_server"] = self.server_row.get_text().strip()
        s["online_webapi"] = self.webapi_row.get_text().strip()
        s["online_npid"] = self.npid_row.get_text().strip()
        s["online_password"] = self.password_row.get_text()
        s["online_upnp"] = self.upnp_row.get_active()
        s["online_pool"] = self.online_pool()
        s["thd_server"] = self.thd_server_row.get_text().strip()
        s["thd_online_id"], s["thd_token"] = self.thd_id, self.thd_token
        self.save_mod_profile()
        self.save_patch_profile()
        save_settings(s)
        frame_gen_choice_now = combo_value(self.frame_gen_row)
        if frame_gen_choice_now == "fsr3":
            frame_generation, frame_generation_multiplier = "fsr3", self.ini.get("frame_generation_multiplier", "2")
        else:
            frame_generation, frame_generation_multiplier = frame_gen_ini(frame_gen_choice_now)
        self.ini.update({
            "upscaler": combo_value(self.upscaler_row),
            "frame_generation": frame_generation,
            "frame_generation_multiplier": frame_generation_multiplier,
            "low_latency": combo_value(self.low_latency_row),
            "preset": str(combo_value(self.preset_row)),
            "sharpen": "1" if self.sharpen_row.get_active() else "0",
            "sharpness": f"{self.sharpness_row.get_value():.2f}",
            "object_motion": "1" if self.motion_row.get_active() else "0",
            "show_fps": "1" if self.show_fps_row.get_active() else "0",
            "vbao": "1" if self.vbao_row.get_active() else "0",
            "aspect_ratio": combo_value(self.aspect_row),
            "output_res": combo_value(self.output_row),
            "model_lod": combo_value(self.lod_row),
            "texture_quality": combo_value(self.texture_row),
            "mouse_look": "1" if self.mouse_look_row.get_active() else "0",
            "mouse_sensitivity": f"{self.mouse_sensitivity_row.get_value():.2f}",
            "mouse_invert_y": "1" if self.mouse_invert_row.get_active() else "0",
            "mouse_invert_x": "1" if self.mouse_invert_x_row.get_active() else "0",
            **{key: f"{row.get_value():.2f}" for key, row in self.camera_speed_rows.items()},
            "button_prompts": combo_value(self.prompts_row),
            "game_text_language": combo_value(self.text_language_row),
            "game_voice_language": combo_value(self.voice_language_row),
            **{key: "1" if row.get_active() else "0" for key, row in self.effect_rows.items()},
            "anti_aliasing": combo_value(self.aa_row),
        })
        save_ini(self.ini, self.ini_lines)
        self.ini, self.ini_lines = load_ini()

    def environment(self):
        return game_environment(self.settings)

    # --- mods ----------------------------------------------------------------------------

    def on_create_shortcut(self):
        self.store()  # the shortcut plays with what the window shows now
        try:
            paths = create_shortcuts()
        except OSError as error:
            self.toasts.add_toast(Adw.Toast(title=tr("Не удалось создать ярлык: {}").format(error)))
            return
        self.toasts.add_toast(Adw.Toast(title=tr("Ярлык создан: {}").format(
            ", ".join(str(path.parent) for path in paths))))

    def on_launch(self, _button):
        if self.process:
            self.stop_game()
            return
        npid = self.npid_row.get_text().strip()
        if online_module() and not self.online_row.get_active() and npid and hasattr(Adw, "AlertDialog"):
            dialog = Adw.AlertDialog(heading="Bloodborne", body=tr(
                "Учётная запись для игры онлайн указана, но «Играть онлайн» выключено"))
            dialog.add_response("cancel", tr("Отмена"))
            dialog.add_response("offline", tr("Играть офлайн"))
            dialog.add_response("online", tr("Играть онлайн"))
            dialog.set_response_appearance("online", Adw.ResponseAppearance.SUGGESTED)

            def on_response(_dialog, response):
                if response != "cancel":
                    self.online_row.set_active(response == "online")
                    self.start_game()
            dialog.connect("response", on_response)
            dialog.present(self)
            return
        self.start_game()

    def start_game(self):
        if self.online_row.get_active() and "@" in self.npid_row.get_text():
            self.alert("Bloodborne", tr("Online ID — это имя учётной записи (NPID), "
                                        "зарегистрированное на сервере, а не адрес email"))
            self.select_section("online")
            return
        self.store()
        self.log_view.get_buffer().set_text("")
        launcher = Gio.SubprocessLauncher.new(
            Gio.SubprocessFlags.STDOUT_PIPE | Gio.SubprocessFlags.STDERR_MERGE)
        launcher.set_environ([f"{k}={v}" for k, v in self.environment().items()])
        launcher.set_cwd(str(PORT_DIR))
        try:
            # setsid: the game and its helpers form one process group, stopped together.
            self.process = launcher.spawnv(["setsid", "bash", str(PORT_DIR / "run.sh")])
        except GLib.Error as error:
            self.toasts.add_toast(Adw.Toast(title=tr("Не удалось запустить: {}").format(error.message)))
            return
        self.stream = Gio.DataInputStream.new(self.process.get_stdout_pipe())
        self.read_line()
        self.process.wait_async(None, self.on_exit)
        self.update_launch_button()
        self.select_section("log")

    def read_line(self):
        self.stream.read_line_async(GLib.PRIORITY_DEFAULT, None, self.on_line)

    def on_line(self, stream, result):
        try:
            line, _length = stream.read_line_finish_utf8(result)
        except GLib.Error:
            return
        if line is None:
            return
        self.append_log(line + "\n")
        self.read_line()

    def stop_game(self):
        if not self.process:
            return
        pid = int(self.process.get_identifier())
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            self.process.force_exit()
        GLib.timeout_add_seconds(3, self.kill_if_running, pid)

    def kill_if_running(self, pid):
        if self.process:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass
        return False

    def on_exit(self, process, result):
        try:
            process.wait_finish(result)
        except GLib.Error:
            pass
        status = process.get_exit_status() if process.get_if_exited() else -1
        self.process = None
        self.update_launch_button()
        self.update_game_status()
        self.refresh_trophies()  # what this session unlocked, and the names once decrypted
        self.append_log(tr("\n— игра завершилась (код {}) —\n").format(status))

    def on_close(self, _window):
        self.store()
        if self.process:
            self.stop_game()
        if self.fsr411_build:
            self.stop_fsr411_build()
        return False

    # --- FSR 4.1.1 from the user's DLL -----------------------------------------------------



class LauncherApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="io.github.bbport.Launcher",
                         flags=Gio.ApplicationFlags.DEFAULT_FLAGS)

    def do_activate(self):
        window = self.get_active_window() or LauncherWindow(self)
        window.present()


def play():
    """--play: the game with the saved settings, no window (Steam Deck game mode)."""
    settings = load_settings()
    # The launcher window does this in its constructor; without it the in-game menu's language
    # was the module default (Russian) whatever was chosen (#134).
    set_language(settings.get("ui_language", ""))
    if not (Path(settings["game_dir"]).expanduser() / "eboot.bin").is_file():
        print("bbport: choose the game folder in the launcher first", file=sys.stderr)
        return 1
    os.chdir(PORT_DIR)
    os.execvpe("bash", ["bash", str(PORT_DIR / "run.sh")], game_environment(settings))


def build_fsr411(arguments):
    """--build-fsr411 <upscaler DLL> [loader DLL]: the launcher's FSR 4.1.1 build, without a window."""
    if not 1 <= len(arguments) <= 2:
        print("usage: --build-fsr411 <amd_fidelityfx_upscaler_dx12.dll> [loader DLL]", file=sys.stderr)
        return 1
    command, env = fsr411_build_command(*arguments)
    return subprocess.call(command, env=env, cwd=PORT_DIR)


if __name__ == "__main__":
    if "--play" in sys.argv[1:]:
        sys.exit(play())
    if "--build-fsr411" in sys.argv[1:]:
        sys.exit(build_fsr411(sys.argv[sys.argv.index("--build-fsr411") + 1:]))
    sys.exit(LauncherApp().run(sys.argv))
