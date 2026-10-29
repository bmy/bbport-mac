# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher's online page: server pool, THD account, connection test."""

from bbport_launcher_common import *  # noqa: F401,F403


class OnlinePage:
    """Part of LauncherWindow (bbport_launcher.py)."""

    def build_online_page(self):
        s = self.settings
        page = Adw.PreferencesPage()
        play = Adw.PreferencesGroup(description=tr(
            "Сообщения, пятна крови и призраки приходят с игрового сервера; колокола и призывы "
            "идут через сервер shadNet. По умолчанию оба — Seamless Hunt (сервер BBSC). Игра по сети "
            "совместима с версией для Windows: нужны версия игры 1.09 и тот же сервер"))
        self.online_row = Adw.SwitchRow(title=tr("Играть онлайн"),
                                        subtitle=tr("Выключено: игра остаётся офлайн, как раньше"),
                                        active=s["online"])
        play.add(self.online_row)
        if not online_module():
            self.online_row.set_sensitive(False)
            self.online_row.set_subtitle(tr("Эта сборка без модуля онлайна (libbbnet20.so): игра только офлайн"))
        self.pool_row = combo_row(tr("Пул"), tr("С кем играть: у всех в сессии должен быть один пул"), [
            ("Seamless Hunt (seamlesshunt.com)", "seamless",
             "Сервер BBSC: бесшовный кооператив, колокола и призывы; аккаунт создаётся на сайте"),
            ("The Hunter's Dream (thehuntersdream.com)", "huntersdream",
             "Кооператив с игроками bbhost через The Hunter's Dream; вход через Discord"),
        ], "huntersdream" if s.get("online_pool") == "huntersdream" else "seamless")
        play.add(self.pool_row)
        page.add(play)

        game = Adw.PreferencesGroup(title=tr("Игровой сервер (сообщения, пятна крови, призраки)"))
        self.game_group = game
        self.community_row = Adw.EntryRow(title=tr("Игровой сервер"), text=s["online_community"])
        game.add(self.community_row)
        page.add(game)

        coop = None
        coop = Adw.PreferencesGroup(
            title=tr("Сервер кооператива (колокола и призывы)"),
            description=tr("play.seamlesshunt.com:31313 — сервер Seamless Hunt (BBSC). Публичный "
                           "сервер shadPS4 принимает только сборки shadPS4. Для частного сервера "
                           "укажите host:port от его владельца. WebAPI: оставьте пустым, тогда "
                           "берётся адрес сервера с портом 31315"))
        self.server_row = Adw.EntryRow(title=tr("Адрес сервера"), text=s["online_server"])
        self.webapi_row = Adw.EntryRow(title=tr("Адрес WebAPI"), text=s["online_webapi"])
        coop.add(self.server_row)
        coop.add(self.webapi_row)
        self.bbsc_row = Adw.SwitchRow(
            title=tr("Бесшовный кооператив (BBSC)"),
            subtitle=tr("Сессия переживает смерти, лампы, смену локаций и боссов; до 5 охотников. "
                        "Нужен сервер с игровым сервером BBSC, у всех игроков включено") + ". " +
                     tr("Пока включено, колокола игры заменены колоколами BBSC; после выключения "
                        "прежние вернутся при следующей загрузке персонажа (сохранение копируется "
                        "в save-backups)"),
            active=s.get("online_bbsc", False))
        coop.add(self.bbsc_row)
        self.coop_group = coop
        page.add(coop)

        account = Adw.PreferencesGroup(
            title=tr("Учётная запись"),
            description=tr("Имя учётной записи shadNet, не email. Зарегистрируйтесь на "
                           "seamlesshunt.com или спросите владельца частного сервера. Пароль "
                           "хранится на этом компьютере вместе с настройками лаунчера"))
        self.npid_row = Adw.EntryRow(title=tr("Online ID (NPID)"), text=s["online_npid"])
        self.password_row = Adw.PasswordEntryRow(title=tr("Пароль"), text=s["online_password"])
        account.add(self.npid_row)
        account.add(self.password_row)
        self.account_group = account
        page.add(account)

        thd = Adw.PreferencesGroup(
            title=tr("The Hunter's Dream"),
            description=tr("Вход через Discord, как в bbhost: на сайте нужно подтвердить код. Пароль не "
                           "нужен, токен хранится на этом компьютере вместе с настройками лаунчера"))
        self.thd_server_row = Adw.EntryRow(title=tr("Адрес сервера"), text=s.get("thd_server", ""))
        self.thd_server_row.set_tooltip_text(online.HUNTERS_DREAM_SERVER)
        thd.add(self.thd_server_row)
        self.thd_account_row = Adw.ActionRow()
        self.thd_sign_button = Gtk.Button(valign=Gtk.Align.CENTER)
        self.thd_sign_button.connect("clicked", self.on_thd_sign)
        self.thd_account_row.add_suffix(self.thd_sign_button)
        thd.add(self.thd_account_row)
        self.thd_group = thd
        self.thd_id, self.thd_token = s.get("thd_online_id", ""), s.get("thd_token", "")
        self.thd_signing = False
        self.thd_cancel = False
        self.show_thd_account()
        page.add(thd)

        connection = Adw.PreferencesGroup(
            title=tr("Соединение"),
            description=tr("У всех в сессии должны быть одна версия игры (1.09) и один сервер. "
                           "Читы меняют игру и для других игроков: выключайте их при совместной игре"))
        self.upnp_row = Adw.SwitchRow(title=tr("UPnP (открыть порт на роутере автоматически)"),
                                      subtitle=tr("Выключите при Tailscale или другом VPN"),
                                      active=s["online_upnp"])
        connection.add(self.upnp_row)
        test = Adw.ActionRow(title=tr("Проверить соединение"))
        self.test_button = Gtk.Button(label=tr("Проверить"), valign=Gtk.Align.CENTER)
        self.test_button.connect("clicked", self.on_test_connection)
        test.add_suffix(self.test_button)
        connection.add(test)
        self.test_rows = []
        self.connection_group = connection
        page.add(connection)
        self.pool_row.connect("notify::selected", self.on_pool_changed)
        self.on_pool_changed()
        return page

    def online_pool(self):
        return combo_value(self.pool_row)

    def on_pool_changed(self, *_):
        pool = self.online_pool()
        legacy, thd = pool == "seamless", pool == "huntersdream"
        self.game_group.set_visible(not thd)
        self.coop_group.set_visible(not thd)
        self.account_group.set_visible(not thd)
        self.bbsc_row.set_visible(legacy)
        self.thd_group.set_visible(thd)
        if not online_module(pool):
            self.pool_row.set_subtitle(tr("В этой сборке нет модуля для этого пула (libbbnet20.so)"))

    def show_thd_account(self):
        link = getattr(self, "thd_link", None)
        if link is not None:
            self.thd_account_row.remove(link)
            self.thd_link = None
        if self.thd_id and self.thd_token:
            self.thd_account_row.set_title(tr("Вы вошли как {}").format(GLib.markup_escape_text(self.thd_id)))
            self.thd_account_row.set_subtitle("")
            self.thd_sign_button.set_label(tr("Выйти"))
        else:
            self.thd_account_row.set_title(tr("Вход не выполнен"))
            self.thd_account_row.set_subtitle(tr("Нужен аккаунт Discord, привязанный к The Hunter's Dream"))
            self.thd_sign_button.set_label(tr("Войти через Discord"))
        if self.thd_signing:
            self.thd_sign_button.set_label(tr("Отмена"))
        self.thd_sign_button.set_sensitive(True)

    def on_thd_sign(self, _button):
        if self.thd_signing:
            self.thd_cancel = True  # the waiting thread stops; done() resets the row
            return
        self.thd_cancel = False
        if self.thd_id and self.thd_token:
            self.thd_id = self.thd_token = ""
            self.show_thd_account()
            return
        self.thd_signing = True
        self.show_thd_account()
        server = self.thd_server_row.get_text().strip()

        def show_code(page, code):
            def show():
                self.thd_account_row.set_title(tr("Код входа: {}").format(code))
                self.thd_account_row.set_subtitle(GLib.markup_escape_text(tr(
                    "На странице {}: 1) нажмите «Sign in with Discord to link» и войдите; "
                    "2) вернувшись на страницу, введите код в поле кода (не в «Sign in with a code from the game»). "
                    "Код скопирован в буфер").format(page)))
                # The code goes to the clipboard; the page opens by whatever works in this environment
                # (an AppImage may have no default handler for the GIO call), else a button does it.
                try:
                    self.thd_account_row.get_clipboard().set(code)
                except Exception:
                    pass
                if not open_url(page):
                    link = Gtk.LinkButton(uri=page, label=tr("Открыть страницу"), valign=Gtk.Align.CENTER)
                    self.thd_account_row.add_prefix(link)
                    self.thd_link = link
                return False
            GLib.idle_add(show)

        def done(name, token, error):
            self.thd_signing = False
            if error:
                self.show_thd_account()
                if error != "cancelled":
                    self.thd_account_row.set_subtitle(GLib.markup_escape_text(error))
            else:
                self.thd_id, self.thd_token = name, token
                self.settings["thd_online_id"], self.settings["thd_token"] = name, token
                save_settings(self.settings)
                self.show_thd_account()
            return False

        def work():
            try:
                name, token = online.thd_sign_in(server, show_code, device="bbport on Linux",
                                                  cancelled=lambda: self.thd_cancel)
            except ValueError as error:
                GLib.idle_add(done, "", "", str(error))
                return
            GLib.idle_add(done, name, token, "")
        threading.Thread(target=work, daemon=True).start()

    def online_addresses(self):
        server = self.server_row.get_text().strip() or online.SHADNET_SERVER
        community = online.community_address(self.community_row.get_text())
        webapi = online.webapi_setting(self.webapi_row.get_text(), server, community)
        return server, webapi, community

    def on_test_connection(self, _button):
        if self.online_pool() == "huntersdream":
            self.test_thd_connection()
            return
        server, webapi, community = self.online_addresses()
        names = {"game": tr("Игровой сервер"), "shadnet": tr("Сервер shadNet"), "webapi": "WebAPI"}
        for row in self.test_rows:
            self.connection_group.remove(row)
        self.test_rows = []
        self.test_button.set_sensitive(False)
        self.test_button.set_label(tr("Проверка…"))

        def show(results):
            for name, address, ok, detail in results:
                # Plain text: an error such as "<urlopen error ...>" is not Pango markup.
                row = Adw.ActionRow(title=GLib.markup_escape_text(f"{names[name]}: {address}"),
                                    subtitle=GLib.markup_escape_text(
                                        tr("доступен") if ok else f'{tr("недоступен")} ({detail})'))
                row.add_prefix(Gtk.Image(icon_name="object-select-symbolic" if ok else "dialog-warning-symbolic"))
                self.connection_group.add(row)
                self.test_rows.append(row)
            self.test_button.set_sensitive(True)
            self.test_button.set_label(tr("Проверить"))
            return False

        def work():
            results = online.check(server, webapi, community)
            GLib.idle_add(show, results)
        threading.Thread(target=work, daemon=True).start()

    def test_thd_connection(self):
        names = {"game": tr("Игровой сервер"), "service": tr("Сервис кооператива")}
        server = self.thd_server_row.get_text().strip()
        for row in self.test_rows:
            self.connection_group.remove(row)
        self.test_rows = []
        self.test_button.set_sensitive(False)
        self.test_button.set_label(tr("Проверка…"))

        def show(results):
            for name, address, ok, detail in results:
                row = Adw.ActionRow(title=GLib.markup_escape_text(f"{names[name]}: {address}"),
                                    subtitle=GLib.markup_escape_text(
                                        tr("доступен") if ok else f'{tr("недоступен")} ({detail})'))
                row.add_prefix(Gtk.Image(icon_name="object-select-symbolic" if ok else "dialog-warning-symbolic"))
                self.connection_group.add(row)
                self.test_rows.append(row)
            self.test_button.set_sensitive(True)
            self.test_button.set_label(tr("Проверить"))
            return False

        threading.Thread(target=lambda: GLib.idle_add(show, online.check_thd(server)), daemon=True).start()

    # --- trophies page ------------------------------------------------------------------

    TROPHY_GRADES = {"P": "Платиновый", "G": "Золотой", "S": "Серебряный", "B": "Бронзовый"}
