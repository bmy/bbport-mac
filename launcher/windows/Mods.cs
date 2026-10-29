// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The Mods section's own cards, as the Linux launcher has them (launcher/bbport_launcher_content.py):
// the mods folder with its mods in load order (mods.json: order and the ones turned off; run.py
// gives them to scripts/mods.py, whose content_root this follows) and the third-party shadPS4 XML
// patches for 01.09 (patches.json: the switches that differ from each file's isEnabled;
// scripts/patches.py external_patches).
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows;
using System.Windows.Controls;
using System.Xml.Linq;
using Microsoft.Win32;

namespace BbportLauncher;

public partial class MainWindow
{
    private static readonly Dictionary<string, (string Ru, string En)> ModTexts = new()
    {
        ["open"] = ("Открыть", "Open"),
        ["refresh"] = ("Обновить", "Refresh"),
        ["no_mods"] = ("Модов нет: распакуйте каждый мод в отдельную папку здесь.",
                       "No mods: unpack each mod into its own folder here."),
        ["no_patches"] = ("Патчей нет: положите сюда XML-патчи shadPS4.", "No patches: put shadPS4 XML patches here."),
        ["earlier"] = ("Загрузить раньше", "Load earlier"),
        ["later"] = ("Загрузить позже", "Load later"),
        ["author"] = ("Автор: {0}", "Author: {0}"),
    };

    private string M(string key) => Translated(key, ModTexts[key].Ru, ModTexts[key].En);

    // The game's top-level folders and the retail serials (scripts/mods.py GAME_FOLDERS, SERIALS).
    private static readonly HashSet<string> GameFolders = new(StringComparer.OrdinalIgnoreCase)
    {
        "action", "adhoc", "chr", "event", "facegen", "font", "map", "menu", "movie", "msg", "mtd", "obj",
        "other", "param", "paramdef", "parts", "remo", "script", "sfx", "shader", "sound",
    };
    private static readonly string[] Serials =
        { "CUSA03173", "CUSA00900", "CUSA00207", "CUSA00208", "CUSA00299", "CUSA01363", "CUSA03179", "CUSA03014", "CUSA03023" };
    private static readonly HashSet<string> BloodborneIds = new(Serials);

    private string ModsDir => store.Get("mods_dir") is { Length: > 0 } dir ? dir : Path.Combine(Store.DataDir, "mods");
    private string PatchesDir => store.Get("patches_dir") is { Length: > 0 } dir ? dir : Path.Combine(Store.DataDir, "patches");
    private static string ModsConfig => Path.Combine(Store.DataDir, "mods.json");
    private static string PatchesConfig => Path.Combine(Store.DataDir, "patches.json");

    /// A folder with an accepted mod layout (scripts/mods.py content_root): dvdroot_ps4 in it, in
    /// app0 or a serial folder; the game's own folders without it; or one wrapper folder around these.
    private static bool IsMod(string folder)
    {
        if (!Directory.Exists(folder))
            return false;
        foreach (var wrapper in new[] { "", "app0" }.Concat(Serials))
        {
            var basePath = wrapper.Length > 0 ? Path.Combine(folder, wrapper) : folder;
            if (Directory.Exists(Path.Combine(basePath, "dvdroot_ps4")))
                return true;
        }
        var entries = new DirectoryInfo(folder).EnumerateFileSystemInfos().Where(e => !e.Name.StartsWith('.')).ToList();
        var folders = entries.OfType<DirectoryInfo>().ToList();
        if (folders.Count > 0 && folders.All(f => GameFolders.Contains(f.Name)))
            return true;
        var notes = new[] { ".txt", ".md", ".jpg", ".png", ".ini" };
        if (folders.Count == 1 && !entries.OfType<FileInfo>().Any(f => !notes.Contains(f.Extension.ToLowerInvariant())))
            return IsMod(folders[0].FullName);
        return false;
    }

    private static JsonObject ReadJson(string path)
    {
        try
        {
            return JsonNode.Parse(File.ReadAllText(path, Encoding.UTF8)) as JsonObject ?? new JsonObject();
        }
        catch (Exception)
        {
            return new JsonObject();
        }
    }

    private static List<string> Strings(JsonNode? node) =>
        node is JsonArray array ? array.Select(n => n?.ToString() ?? "").Where(s => s.Length > 0).ToList() : new();

    private static void WriteJson(string path, JsonObject value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, value.ToJsonString(new JsonSerializerOptions
        {
            WriteIndented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        }) + "\n", new UTF8Encoding(false));
    }

    /// The folder line of both cards: the path, Browse, Open, Refresh.
    private (DockPanel Panel, TextBlock Path) FolderLine(string settingKey, Func<string> current, Action changed)
    {
        var path = new TextBlock
        {
            TextWrapping = TextWrapping.Wrap, FontSize = 12, Opacity = 0.85, VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(0, 0, 12, 0),
        };
        var buttons = new StackPanel { Orientation = Orientation.Horizontal };
        var browse = new Button { Content = T("browse"), Padding = new Thickness(14, 5, 14, 5) };
        var open = new Button { Content = M("open"), Padding = new Thickness(14, 5, 14, 5), Margin = new Thickness(8, 0, 0, 0) };
        var refresh = new Button { Content = M("refresh"), Padding = new Thickness(14, 5, 14, 5), Margin = new Thickness(8, 0, 0, 0) };
        buttons.Children.Add(browse);
        buttons.Children.Add(open);
        buttons.Children.Add(refresh);
        var panel = new DockPanel();
        DockPanel.SetDock(buttons, Dock.Right);
        panel.Children.Add(buttons);
        panel.Children.Add(path);
        browse.Click += (_, _) =>
        {
            var dialog = new OpenFolderDialog();
            if (Directory.Exists(current()))
                dialog.InitialDirectory = current();
            if (dialog.ShowDialog(this) == true)
            {
                store.Set(settingKey, dialog.FolderName);
                store.Save();
                changed();
            }
        };
        open.Click += (_, _) =>
        {
            try
            {
                Directory.CreateDirectory(current());
                Process.Start(new ProcessStartInfo("explorer.exe", $"\"{current()}\"") { UseShellExecute = true });
            }
            catch (Exception)
            {
            }
        };
        refresh.Click += (_, _) => changed();
        return (panel, path);
    }

    private Border ModsCard(JsonObject item)
    {
        var list = new StackPanel { Margin = new Thickness(0, 8, 0, 0) };
        var holder = new StackPanel();
        Action? refill = null;
        var (line, path) = FolderLine("mods_dir", () => ModsDir, () => refill?.Invoke());
        holder.Children.Add(line);
        holder.Children.Add(list);
        var (card, _) = Card(item, holder, wide: true);
        var order = new List<string>();
        var disabled = new HashSet<string>();

        void Save()
        {
            WriteJson(ModsConfig, new JsonObject
            {
                ["order"] = new JsonArray(order.Select(n => (JsonNode?)JsonValue.Create(n)).ToArray()),
                ["disabled"] = new JsonArray(order.Where(disabled.Contains).Select(n => (JsonNode?)JsonValue.Create(n)).ToArray()),
            });
        }

        void Fill()
        {
            list.Children.Clear();
            var available = Directory.Exists(ModsDir)
                ? Directory.EnumerateDirectories(ModsDir).Where(IsMod).Select(Path.GetFileName).OfType<string>()
                    .OrderBy(n => n, StringComparer.OrdinalIgnoreCase).ToList()
                : new List<string>();
            var profile = ReadJson(ModsConfig);
            // Saved order first, new folders after it (alphabetically): scripts/mods.py selected().
            order = Strings(profile["order"]).Concat(available).Distinct().Where(available.Contains).ToList();
            disabled = Strings(profile["disabled"]).ToHashSet();
            path.Text = ModsDir + (order.Count == 0 ? " — " + M("no_mods") : "");
            for (var i = 0; i < order.Count; ++i)
            {
                var name = order[i];
                var index = i;
                var row = new DockPanel { Margin = new Thickness(0, 2, 0, 2) };
                var moves = new StackPanel { Orientation = Orientation.Horizontal };
                var up = new Button { Content = "↑", Width = 32, ToolTip = M("earlier"), IsEnabled = index > 0 };
                var down = new Button { Content = "↓", Width = 32, ToolTip = M("later"), Margin = new Thickness(6, 0, 0, 0),
                                        IsEnabled = index < order.Count - 1 };
                moves.Children.Add(up);
                moves.Children.Add(down);
                DockPanel.SetDock(moves, Dock.Right);
                row.Children.Add(moves);
                var box = new CheckBox { Content = name, IsChecked = !disabled.Contains(name), VerticalAlignment = VerticalAlignment.Center };
                row.Children.Add(box);
                box.Click += (_, _) =>
                {
                    if (box.IsChecked == true)
                        disabled.Remove(name);
                    else
                        disabled.Add(name);
                    Save();
                };
                up.Click += (_, _) => Move(index, -1);
                down.Click += (_, _) => Move(index, 1);
                list.Children.Add(row);
            }
        }

        void Move(int index, int direction)
        {
            var target = index + direction;
            if (target < 0 || target >= order.Count)
                return;
            (order[index], order[target]) = (order[target], order[index]);
            Save();
            Fill();
        }

        refill = Fill;
        Fill();
        return card;
    }

    /// shadPS4 XML patches for this version in the folder (scripts/patches.py external_patches): key
    /// "<file>/<patch name>", the file's isEnabled, author and note. The port's own
    /// patches\Bloodborne.xml is not one of them.
    private static List<(string Key, bool Default, string Name, string File, string Author, string Note)> ExternalPatches(string folder)
    {
        var found = new List<(string, bool, string, string, string, string)>();
        if (!Directory.Exists(folder))
            return found;
        var builtIn = Path.GetFullPath(Path.Combine(Store.PackageDir, "patches", "Bloodborne.xml"));
        foreach (var file in Directory.EnumerateFiles(folder, "*.xml").OrderBy(f => f, StringComparer.Ordinal))
        {
            if (string.Equals(Path.GetFullPath(file), builtIn, StringComparison.OrdinalIgnoreCase))
                continue;
            XDocument document;
            try
            {
                document = XDocument.Load(file);
            }
            catch (Exception)
            {
                continue;
            }
            var ids = document.Descendants("ID").Select(e => e.Value.Trim()).Where(v => v.Length > 0).ToHashSet();
            if (ids.Count > 0 && !ids.Overlaps(BloodborneIds))
                continue;
            foreach (var meta in document.Descendants("Metadata"))
            {
                if ((string?)meta.Attribute("AppVer") != "01.09" || ((string?)meta.Attribute("AppElf") ?? "eboot.bin") != "eboot.bin")
                    continue;
                var name = (string?)meta.Attribute("Name") ?? "";
                found.Add(($"{Path.GetFileName(file)}/{name}",
                           string.Equals((string?)meta.Attribute("isEnabled"), "true", StringComparison.OrdinalIgnoreCase),
                           name, Path.GetFileName(file), (string?)meta.Attribute("Author") ?? "",
                           ((string?)meta.Attribute("Note") ?? "").Replace("\\n", "\n")));
            }
        }
        return found;
    }

    private Border PatchesCard(JsonObject item)
    {
        var list = new StackPanel { Margin = new Thickness(0, 8, 0, 0) };
        var holder = new StackPanel();
        Action? refill = null;
        var (line, path) = FolderLine("patches_dir", () => PatchesDir, () => refill?.Invoke());
        holder.Children.Add(line);
        holder.Children.Add(list);
        var (card, _) = Card(item, holder, wide: true);

        void Fill()
        {
            list.Children.Clear();
            var found = ExternalPatches(PatchesDir);
            var profile = ReadJson(PatchesConfig);
            var enabled = Strings(profile["enabled"]).ToHashSet();
            var disabled = Strings(profile["disabled"]).ToHashSet();
            path.Text = PatchesDir + (found.Count == 0 ? " — " + M("no_patches") : "");
            foreach (var patch in found)
            {
                var active = enabled.Contains(patch.Key) || (patch.Default && !disabled.Contains(patch.Key));
                var text = new StackPanel();
                text.Children.Add(new TextBlock { Text = patch.Name.Length > 0 ? patch.Name : patch.Key, TextWrapping = TextWrapping.Wrap });
                var detail = patch.File + (patch.Author.Length > 0 ? " · " + string.Format(M("author"), patch.Author) : "");
                text.Children.Add(new TextBlock { Text = detail, FontSize = 11, Opacity = 0.7 });
                var box = new CheckBox { Content = text, IsChecked = active, Margin = new Thickness(0, 3, 0, 3) };
                if (patch.Note.Length > 0)
                    box.ToolTip = patch.Note;
                var key = patch.Key;
                var fallback = patch.Default;
                box.Click += (_, _) =>
                {
                    // Only the switches that differ from the file's isEnabled are kept (patches.json).
                    var now = ReadJson(PatchesConfig);
                    var on = Strings(now["enabled"]).ToHashSet();
                    var off = Strings(now["disabled"]).ToHashSet();
                    on.Remove(key);
                    off.Remove(key);
                    if (box.IsChecked == true && !fallback)
                        on.Add(key);
                    if (box.IsChecked != true && fallback)
                        off.Add(key);
                    WriteJson(PatchesConfig, new JsonObject
                    {
                        ["enabled"] = new JsonArray(on.Order(StringComparer.Ordinal).Select(k => (JsonNode?)JsonValue.Create(k)).ToArray()),
                        ["disabled"] = new JsonArray(off.Order(StringComparer.Ordinal).Select(k => (JsonNode?)JsonValue.Create(k)).ToArray()),
                    });
                };
                list.Children.Add(box);
            }
        }

        refill = Fill;
        Fill();
        return card;
    }
}
