// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The Online section's own cards, as the Linux launcher's Online page has them
// (launcher/bbport_launcher_online.py): The Hunter's Dream account (Discord sign-in with a code
// approved on the website) and the connection check. Both run scripts/online.py through
// bbport-run.exe --run-script (players have no Python of their own).
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;

namespace BbportLauncher;

public partial class MainWindow
{
    private static readonly Dictionary<string, (string Ru, string En)> OnlineTexts = new()
    {
        ["thd_signed_in"] = ("Вы вошли как {0}", "Signed in as {0}"),
        ["thd_signed_out"] = ("Вход не выполнен: нужен аккаунт Discord, привязанный к The Hunter's Dream",
                              "Not signed in: a Discord account linked to The Hunter's Dream is needed"),
        ["thd_sign_in"] = ("Войти через Discord", "Sign in with Discord"),
        ["thd_sign_out"] = ("Выйти", "Sign out"),
        ["thd_cancel"] = ("Отмена", "Cancel"),
        ["thd_waiting"] = ("Запрашиваю код входа…", "Asking for a sign-in code…"),
        ["thd_code"] = ("Код входа: {0}. На открывшейся странице ({1}): 1) нажмите «Sign in with Discord to link» и войдите; 2) вернувшись на страницу, введите код в поле кода (не в «Sign in with a code from the game»). Код скопирован в буфер.",
                        "Sign-in code: {0}. On the page that opened ({1}): 1) press \"Sign in with Discord to link\" and sign in; 2) back on the page, enter the code in the code field (not \"Sign in with a code from the game\"). The code is on the clipboard."),
        ["thd_failed"] = ("Вход не удался: {0}", "Sign-in failed: {0}"),
        ["check"] = ("Проверить", "Check"),
        ["checking"] = ("Проверка…", "Checking…"),
        ["reachable"] = ("доступен", "reachable"),
        ["not_reachable"] = ("недоступен", "not reachable"),
        ["check_game"] = ("Игровой сервер", "Game server"),
        ["check_shadnet"] = ("Сервер shadNet", "shadNet server"),
        ["check_webapi"] = ("WebAPI", "WebAPI"),
        ["check_service"] = ("Сервис кооператива", "Co-op service"),
    };

    private string O(string key) => Translated(key, OnlineTexts[key].Ru, OnlineTexts[key].En);

    private Process? onlineTool;

    /// bbport-run.exe --run-script scripts/online.py ARGS with the given environment; its standard
    /// output and error lines go to the handlers (on the window's thread), exited(code) at the end.
    private Process? RunOnlineScript(IEnumerable<string> args, Dictionary<string, string> environment,
                                     Action<string> output, Action<string> error, Action<int> exited)
    {
        var runner = Path.Combine(Store.PackageDir, "bbport-run.exe");
        if (!File.Exists(runner))
        {
            MessageBox.Show(this, T("no_runner"), "Bloodborne", MessageBoxButton.OK, MessageBoxImage.Error);
            return null;
        }
        var info = new ProcessStartInfo(runner)
        {
            WorkingDirectory = Store.PackageDir, UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
        };
        info.ArgumentList.Add("--run-script");
        info.ArgumentList.Add(Path.Combine(Store.PackageDir, "scripts", "online.py"));
        foreach (var arg in args)
            info.ArgumentList.Add(arg);
        info.Environment["PYTHONIOENCODING"] = "utf-8";
        info.Environment["PYTHONUNBUFFERED"] = "1";
        foreach (var (key, value) in environment)
            info.Environment[key] = value;
        var process = new Process { StartInfo = info, EnableRaisingEvents = true };
        process.OutputDataReceived += (_, line) =>
        {
            if (line.Data is { } text)
                Dispatcher.BeginInvoke(() => output(text));
        };
        process.ErrorDataReceived += (_, line) =>
        {
            if (line.Data is { } text)
                Dispatcher.BeginInvoke(() => error(text));
        };
        process.Exited += (_, _) =>
        {
            // The last lines first: WaitForExit drains the redirected streams.
            process.WaitForExit();
            var code = process.ExitCode;
            Dispatcher.BeginInvoke(() =>
            {
                process.Dispose();
                exited(code);
            });
        };
        try
        {
            process.Start();
        }
        catch (Exception failure)
        {
            MessageBox.Show(this, failure.Message, "Bloodborne", MessageBoxButton.OK, MessageBoxImage.Error);
            return null;
        }
        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        return process;
    }

    /// The Hunter's Dream account: signed in (its online id) or not, the sign-in button.
    private Border ThdAccountCard(JsonObject item)
    {
        var button = new Button { Padding = new Thickness(12, 4, 12, 4), MinWidth = 160 };
        var (card, note) = Card(item, button);

        void Show()
        {
            var id = store.Settings["thd_online_id"]?.ToString() ?? "";
            var token = store.Settings["thd_token"]?.ToString() ?? "";
            if (onlineTool is not null)
            {
                button.Content = O("thd_cancel");
                return;
            }
            var inside = id.Length > 0 && token.Length > 0;
            button.Content = inside ? O("thd_sign_out") : O("thd_sign_in");
            ShowNote(note, inside ? string.Format(O("thd_signed_in"), id) : O("thd_signed_out"));
        }

        button.Click += (_, _) =>
        {
            if (onlineTool is not null)
            {
                try
                {
                    onlineTool.Kill(entireProcessTree: true);
                }
                catch (Exception)
                {
                }
                return;
            }
            if ((store.Settings["thd_token"]?.ToString() ?? "").Length > 0)
            {
                store.Settings["thd_online_id"] = "";
                store.Settings["thd_token"] = "";
                store.Save();
                Show();
                return;
            }
            var lines = new List<string>();
            var problem = "";
            ShowNote(note, O("thd_waiting"));
            onlineTool = RunOnlineScript(
                new[] { "thd-signin" },
                new Dictionary<string, string> { ["BB_COMMUNITY_SERVER"] = store.Get("thd_server") },
                output: line => lines.Add(line.Trim()),
                error: line =>
                {
                    // "Open <page> and enter <code>" (online.py thd-signin): the page opens in the
                    // browser and the code goes to the clipboard.
                    var match = Regex.Match(line, @"^Open (\S+) and enter (\S+)");
                    if (!match.Success)
                    {
                        if (line.Trim().Length > 0)
                            problem = line.Trim();
                        return;
                    }
                    var (page, code) = (match.Groups[1].Value, match.Groups[2].Value);
                    ShowNote(note, string.Format(O("thd_code"), code, page));
                    try
                    {
                        Clipboard.SetText(code);
                    }
                    catch (Exception)
                    {
                    }
                    try
                    {
                        Process.Start(new ProcessStartInfo(page) { UseShellExecute = true });
                    }
                    catch (Exception)
                    {
                    }
                },
                exited: code =>
                {
                    onlineTool = null;
                    var done = lines.Where(line => line.Length > 0).ToList();
                    if (code == 0 && done.Count >= 2)
                    {
                        store.Settings["thd_online_id"] = done[^2];
                        store.Settings["thd_token"] = done[^1];
                        store.Save();
                        Show();
                    }
                    else
                    {
                        Show();
                        if (problem.Length > 0)
                            ShowNote(note, string.Format(O("thd_failed"), problem));
                    }
                });
            Show();
        };
        Show();
        return card;
    }

    /// The connection check: the pool's servers, each reachable or not (online.py check / thd-check).
    private Border OnlineTestCard(JsonObject item)
    {
        var button = new Button { Content = O("check"), Padding = new Thickness(12, 4, 12, 4), MinWidth = 120 };
        var (card, note) = Card(item, button);
        button.Click += (_, _) =>
        {
            var thd = store.Get("online_pool") == "huntersdream";
            var environment = thd
                ? new Dictionary<string, string> { ["BB_COMMUNITY_SERVER"] = store.Get("thd_server") }
                : new Dictionary<string, string>
                {
                    ["BB_SHADNET_SERVER"] = store.Get("online_server"),
                    ["BB_SHADNET_WEBAPI"] = store.Get("online_webapi"),
                    ["BB_COMMUNITY_SERVER"] = store.Get("online_community"),
                };
            var names = new Dictionary<string, string>
            {
                ["game"] = O("check_game"), ["shadnet"] = O("check_shadnet"), ["webapi"] = O("check_webapi"),
                ["service"] = O("check_service"),
            };
            var results = new List<string>();
            button.IsEnabled = false;
            button.Content = O("checking");
            ShowNote(note, "");
            RunOnlineScript(
                new[] { thd ? "thd-check" : "check" }, environment,
                output: line =>
                {
                    // "<name>: <address>: reachable|not reachable <detail>"
                    var match = Regex.Match(line, @"^(\w+): (\S+): (not reachable|reachable)\s*(.*)$");
                    if (!match.Success)
                        return;
                    var name = names.GetValueOrDefault(match.Groups[1].Value, match.Groups[1].Value);
                    var ok = match.Groups[3].Value == "reachable";
                    var detail = match.Groups[4].Value.Trim();
                    results.Add((ok ? "✔ " : "⚠ ") + $"{name}: {match.Groups[2].Value} — " +
                                (ok ? O("reachable") : O("not_reachable") + (detail.Length > 0 ? $" ({detail})" : "")));
                    ShowNote(note, string.Join("\n", results));
                },
                error: _ => { },
                exited: _ =>
                {
                    button.IsEnabled = true;
                    button.Content = O("check");
                });
        };
        return card;
    }
}
