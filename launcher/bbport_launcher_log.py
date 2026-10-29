# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher's log page: the game's output, copy and export."""

from bbport_launcher_common import *  # noqa: F401,F403


class LogPage:
    """Part of LauncherWindow (bbport_launcher.py)."""

    def build_log_page(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.log_view = Gtk.TextView(editable=False, monospace=True, cursor_visible=False,
                                     wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.log_view.set_top_margin(8)
        self.log_view.set_left_margin(8)
        self.log_view.set_right_margin(8)
        scroller = Gtk.ScrolledWindow(vexpand=True, child=self.log_view)
        self.log_scroller = scroller
        box.append(scroller)
        bar = Gtk.ActionBar()
        self.export_log_button = Gtk.Button(label=tr("Экспорт журнала…"))
        self.export_log_button.connect("clicked", self.on_export_log)
        bar.pack_end(self.export_log_button)
        self.copy_log_button = Gtk.Button(label=tr("Копировать"))
        self.copy_log_button.connect("clicked", self.on_copy_log)
        bar.pack_end(self.copy_log_button)
        box.append(bar)
        buffer = self.log_view.get_buffer()
        buffer.connect("changed", self.on_log_changed)
        self.on_log_changed(buffer)
        return box

    def on_log_changed(self, buffer):
        has_text = buffer.get_char_count() > 0
        self.copy_log_button.set_sensitive(has_text)
        self.export_log_button.set_sensitive(has_text)

    def log_text(self):
        buffer = self.log_view.get_buffer()
        return buffer.get_text(*buffer.get_bounds(), False)

    def on_copy_log(self, _button):
        self.get_clipboard().set(self.log_text())
        self.toasts.add_toast(Adw.Toast(title=tr("Журнал скопирован")))

    def on_export_log(self, _button):
        text = self.log_text()
        dialog = Gtk.FileDialog(title=tr("Экспорт журнала"))
        dialog.set_initial_name(f"bbport-{time.strftime('%Y%m%d_%H%M%S')}.log")

        def finish(dialog, result):
            try:
                chosen = dialog.save_finish(result)
            except GLib.Error:
                return
            if not chosen or not chosen.get_path():
                return
            try:
                Path(chosen.get_path()).write_text(text, encoding="utf-8")
            except OSError as error:
                self.toasts.add_toast(Adw.Toast(
                    title=tr("Не удалось сохранить журнал: {}").format(error.strerror or error)))
                return
            self.toasts.add_toast(Adw.Toast(title=tr("Журнал сохранён: {}").format(chosen.get_path())))
        dialog.save(self, None, finish)

    def append_log(self, text):
        buffer = self.log_view.get_buffer()
        buffer.insert(buffer.get_end_iter(), text)
        extra = buffer.get_line_count() - MAX_LOG_LINES
        if extra > 0:
            buffer.delete(buffer.get_start_iter(), buffer.get_iter_at_line(extra)[1])
        adj = self.log_scroller.get_vadjustment()
        GLib.idle_add(lambda: adj.set_value(adj.get_upper()) and False)

    # --- process -------------------------------------------------------------------------
