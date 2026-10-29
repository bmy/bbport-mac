// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The Trophies section: what the port keeps of the game's trophies in the user folder (as
// launcher/trophy_list.py reads it for the Linux launcher) and the player's trophy key.
// gpu/shim/bbport_trophies.cpp keeps <user>/trophy/<NP communication ID>/unlocked.txt and, once
// the game has run with the key, the decrypted TROP.XML / TROP_XX.XML with the TROPnnn.PNG icons;
// src/runtime_trophy.c appends every unlock with its time to <user>/trophies.jsonl.
using System.Globalization;
using System.IO;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Xml.Linq;

namespace BbportLauncher;

public partial class MainWindow
{
    private const string TrophyNpId = "NPWR05818_00";

    private sealed record TrophyEntry(int Id, char Grade, bool Hidden, string Name, string Detail,
                                      bool Unlocked, string? Time, string? Icon);

    private static readonly Dictionary<string, (string Ru, string En)> TrophyTexts = new()
    {
        ["trophies"] = ("Трофеи", "Trophies"),
        ["trophies_description"] = (
            "Трофеи, полученные в этой папке сохранений. Названия, описания и значки игра берёт из своего " +
            "пакета трофеев: для этого нужен ключ трофеев — тот же, что просит shadPS4.",
            "The trophies earned in this saves folder. The game takes their names, descriptions and icons " +
            "from its trophy package: that needs the trophy key, the same one shadPS4 asks for."),
        ["key"] = ("Ключ трофеев", "Trophy key"),
        ["key_description"] = ("32 шестнадцатеричные цифры. Хранится только в настройках лаунчера.",
                               "32 hexadecimal digits. Kept in the launcher's settings only."),
        ["key_bad"] = ("Нужно ровно 32 шестнадцатеричные цифры (0–9, A–F).",
                       "Exactly 32 hexadecimal digits are needed (0-9, A-F)."),
        ["progress"] = ("Прогресс", "Progress"),
        ["none"] = ("Трофеев пока нет", "No trophies yet"),
        ["none_description"] = ("Они появятся здесь после первого полученного в игре трофея.",
                                "They show here once the game has awarded the first one."),
        ["got_of"] = ("Получено {0} из {1}", "{0} of {1} earned"),
        ["got"] = ("Получено: {0}", "Earned: {0}"),
        ["unnamed"] = ("Названия неизвестны", "Names unknown"),
        ["unnamed_description"] = ("Укажите ключ трофеев выше и запустите игру: она расшифрует названия и значки.",
                                   "Enter the trophy key above and start the game: it decrypts the names and icons."),
        ["number"] = ("Трофей №{0}", "Trophy #{0}"),
        ["hidden"] = ("Скрытый трофей", "Hidden trophy"),
        ["hidden_description"] = ("Откроется, когда будет получен", "Shown once earned"),
        ["earned"] = ("Получен", "Earned"),
        ["P"] = ("Платиновый", "Platinum"),
        ["G"] = ("Золотой", "Gold"),
        ["S"] = ("Серебряный", "Silver"),
        ["B"] = ("Бронзовый", "Bronze"),
    };

    private string TT(string key) => Translated(key, TrophyTexts[key].Ru, TrophyTexts[key].En);

    /// A card's title and description in the shape Card() takes (a schema item's texts).
    private JsonObject TrophyItem(string title, string description = "") => new()
    {
        ["title"] = new JsonObject { [language] = title },
        ["description"] = new JsonObject { [language] = description },
    };

    private string TrophyUserDir()
    {
        var chosen = store.Get("user_dir");
        return chosen.Length > 0 ? chosen : Path.Combine(Store.DataDir, "user");
    }

    /// The trophies kept for the game: every one when the names are known, else the unlocked ones.
    private (List<TrophyEntry> List, bool Named) LoadTrophies()
    {
        var user = TrophyUserDir();
        var times = new Dictionary<int, string>();
        try
        {
            foreach (var line in File.ReadAllLines(Path.Combine(user, "trophies.jsonl")))
            {
                try
                {
                    var node = JsonNode.Parse(line);
                    var id = (int?)node?["trophy_id"];
                    var stamp = (string?)node?["unlocked_at"];
                    if (id is int i && stamp is not null &&
                        (!times.TryGetValue(i, out var first) || string.CompareOrdinal(stamp, first) < 0))
                        times[i] = stamp;
                }
                catch (Exception)
                {
                }
            }
        }
        catch (IOException)
        {
        }

        string? folder = null;
        var root = Path.Combine(user, "trophy");
        if (Directory.Exists(root))
        {
            var folders = Directory.GetDirectories(root)
                .Where(d => File.Exists(Path.Combine(d, "unlocked.txt")) || File.Exists(Path.Combine(d, "TROP.XML")))
                .ToList();
            folder = folders.FirstOrDefault(d => Path.GetFileName(d) == TrophyNpId)
                     ?? folders.OrderByDescending(Directory.GetLastWriteTimeUtc).FirstOrDefault();
        }
        var got = new HashSet<int>(times.Keys);
        if (folder is not null)
        {
            try
            {
                foreach (var line in File.ReadAllLines(Path.Combine(folder, "unlocked.txt")))
                    if (int.TryParse(line.Trim(), out var id))
                        got.Add(id);
            }
            catch (IOException)
            {
            }
        }

        var known = new List<(int Id, char Grade, bool Hidden, string Name, string Detail)>();
        if (folder is not null)
        {
            var number = int.TryParse(store.Get("language"), out var n) ? n : 1;
            foreach (var name in new[] { $"TROP_{number:00}.XML", "TROP.XML" })
            {
                known = ParseTrophyXml(Path.Combine(folder, name));
                if (known.Count > 0)
                    break;
            }
        }
        var named = known.Count > 0;
        if (!named)
            known = got.OrderBy(id => id).Select(id => (id, '?', false, "", "")).ToList();
        var list = known.Select(t =>
        {
            var icon = folder is null ? null : Path.Combine(folder, $"TROP{t.Id:000}.PNG");
            return new TrophyEntry(t.Id, t.Grade, t.Hidden, t.Name, t.Detail, got.Contains(t.Id),
                                   times.GetValueOrDefault(t.Id), icon is not null && File.Exists(icon) ? icon : null);
        }).ToList();
        return (list, named);
    }

    /// The trophies of a TROP.XML (it may begin with a signature comment); empty when unreadable.
    private static List<(int Id, char Grade, bool Hidden, string Name, string Detail)> ParseTrophyXml(string path)
    {
        var list = new List<(int, char, bool, string, string)>();
        try
        {
            var text = File.ReadAllText(path);
            var start = text.IndexOf("<trophyconf", StringComparison.Ordinal);
            var end = text.LastIndexOf("</trophyconf>", StringComparison.Ordinal);
            if (start < 0 || end < start)
                return list;
            var root = XElement.Parse(text[start..(end + "</trophyconf>".Length)]);
            foreach (var trophy in root.Descendants("trophy"))
            {
                if (!int.TryParse((string?)trophy.Attribute("id"), out var id))
                    continue;
                var grade = ((string?)trophy.Attribute("ttype") ?? "?") is { Length: > 0 } g ? g[0] : '?';
                list.Add((id, grade, (string?)trophy.Attribute("hidden") == "yes",
                          ((string?)trophy.Element("name") ?? "").Trim(), ((string?)trophy.Element("detail") ?? "").Trim()));
            }
        }
        catch (Exception)
        {
            list.Clear();
        }
        return list;
    }

    /// The section: the key, the progress, the list.
    private void BuildTrophies()
    {
        Items.Children.Clear();
        switches.Clear();
        dependents.Clear();
        SectionTitle.Text = TT("trophies");
        SectionDescription.Text = TT("trophies_description");

        var key = new PasswordBox { Password = store.Get("trophy_key"), Width = 320 };
        var (keyCard, keyNote) = Card(TrophyItem(TT("key"), TT("key_description")), key);
        void CheckKey()
        {
            var text = key.Password.Trim();
            store.Set("trophy_key", text);
            ShowNote(keyNote, text.Length > 0 && !Regex.IsMatch(text, "^[0-9A-Fa-f]{32}$") ? TT("key_bad") : "");
        }
        key.PasswordChanged += (_, _) => CheckKey();
        CheckKey();
        Items.Children.Add(keyCard);

        var (list, named) = LoadTrophies();
        Items.Children.Add(Heading(TT("progress")));
        if (list.Count == 0)
        {
            Items.Children.Add(Card(TrophyItem(TT("none"), TT("none_description")), new Border()).Card);
            return;
        }
        var unlocked = list.Count(t => t.Unlocked);
        if (named)
        {
            var grades = string.Join(", ", "PGSB".Where(g => list.Any(t => t.Grade == g)).Select(g =>
                $"{TT(g.ToString())}: {list.Count(t => t.Grade == g && t.Unlocked)}/{list.Count(t => t.Grade == g)}"));
            var bar = new ProgressBar { Minimum = 0, Maximum = list.Count, Value = unlocked, Width = 200, Height = 6 };
            Items.Children.Add(Card(TrophyItem(string.Format(TT("got_of"), unlocked, list.Count), grades), bar).Card);
        }
        else
        {
            Items.Children.Add(Card(TrophyItem(string.Format(TT("got"), unlocked)), new Border()).Card);
            Items.Children.Add(Card(TrophyItem(TT("unnamed"), TT("unnamed_description")), new Border()).Card);
        }
        Items.Children.Add(Heading(TT("trophies")));
        foreach (var trophy in list)
            Items.Children.Add(TrophyCard(trophy, named));
    }

    private TextBlock Heading(string text) => new()
    {
        Text = text, FontSize = 14, FontWeight = FontWeights.SemiBold, Margin = new Thickness(2, 22, 0, 8),
    };

    private Border TrophyCard(TrophyEntry trophy, bool named)
    {
        var secret = trophy.Hidden && !trophy.Unlocked;
        var title = !named ? string.Format(TT("number"), trophy.Id)
                    : secret ? TT("hidden")
                    : trophy.Name.Length > 0 ? trophy.Name : string.Format(TT("number"), trophy.Id);
        var detail = secret ? TT("hidden_description") : trophy.Detail;

        FrameworkElement picture;
        if (trophy.Icon is not null && !secret)
        {
            var bitmap = new BitmapImage();
            bitmap.BeginInit();
            bitmap.UriSource = new Uri(trophy.Icon);
            bitmap.CacheOption = BitmapCacheOption.OnLoad; // the file stays free for the game
            bitmap.DecodePixelWidth = 96;
            bitmap.EndInit();
            picture = new Image { Source = bitmap, Width = 48, Height = 48 };
        }
        else
        {
            picture = new TextBlock
            {
                Text = "", FontFamily = new FontFamily("Segoe Fluent Icons, Segoe MDL2 Assets"), FontSize = 32,
                Width = 48, TextAlignment = TextAlignment.Center, VerticalAlignment = VerticalAlignment.Center,
            };
        }
        picture.Margin = new Thickness(0, 0, 14, 0);

        var text = new StackPanel { VerticalAlignment = VerticalAlignment.Center };
        text.Children.Add(new TextBlock { Text = title, FontSize = 14, TextWrapping = TextWrapping.Wrap });
        if (detail.Length > 0)
            text.Children.Add(new TextBlock
            {
                Text = detail, FontSize = 12, Opacity = 0.72, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 2, 0, 0),
            });

        var info = new List<string>();
        if (TrophyTexts.ContainsKey(trophy.Grade.ToString()))
            info.Add(TT(trophy.Grade.ToString()));
        if (trophy.Unlocked)
            info.Add(trophy.Time is { } time ? LocalTime(time) : TT("earned"));
        var side = new TextBlock
        {
            Text = string.Join("\n", info), FontSize = 12, Opacity = 0.72, TextAlignment = TextAlignment.Right,
            VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(12, 0, 0, 0),
        };

        var grid = new Grid();
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        Grid.SetColumn(text, 1);
        Grid.SetColumn(side, 2);
        grid.Children.Add(picture);
        grid.Children.Add(text);
        grid.Children.Add(side);
        var card = new Border
        {
            Child = grid, Padding = new Thickness(16, 10, 16, 10), Margin = new Thickness(0, 0, 0, 4),
            CornerRadius = new CornerRadius(6), BorderThickness = new Thickness(1),
            Opacity = trophy.Unlocked ? 1.0 : 0.55,
        };
        card.SetResourceReference(Border.BackgroundProperty, "CardBackgroundFillColorDefaultBrush");
        card.SetResourceReference(Border.BorderBrushProperty, "CardStrokeColorDefaultBrush");
        return card;
    }

    /// trophies.jsonl's time (UTC, "2026-10-26T21:05:00Z") in local time.
    private static string LocalTime(string stamp) =>
        DateTime.TryParse(stamp, CultureInfo.InvariantCulture,
                          DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var utc)
            ? utc.ToLocalTime().ToString("dd.MM.yyyy HH:mm", CultureInfo.InvariantCulture)
            : stamp;
}
