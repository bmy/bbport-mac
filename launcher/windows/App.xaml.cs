// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
using System.Diagnostics;
using System.IO;
using System.Windows;

namespace BbportLauncher;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        // `Bloodborne.exe --play` (shortcuts): the game with the saved settings, no window.
        if (e.Args.Contains("--play"))
        {
            var runner = Path.Combine(Store.PackageDir, "bbport-run.exe");
            if (File.Exists(runner))
                Process.Start(new ProcessStartInfo(runner, "--play")
                {
                    WorkingDirectory = Store.PackageDir, UseShellExecute = false, CreateNoWindow = true,
                });
            Shutdown();
            return;
        }
        base.OnStartup(e);
    }
}
