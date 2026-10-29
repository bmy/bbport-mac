# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The launcher's GPU choice: kept by name, passed to the game as its index (BB_GPU_ID)."""
from paths import ROOT
import importlib.util
import os
import subprocess
import sys
import types
import unittest
from unittest import mock

# The launcher's window needs GTK; these tests use its functions only.
gi = types.ModuleType("gi")
gi.require_version = lambda *_args: None
gi.repository = mock.MagicMock()
# Real classes for the window's bases: LauncherWindow also inherits the page mixins
# (bbport_launcher_*.py), and a mock base next to them is a metaclass conflict.
gi.repository.Adw.ApplicationWindow = type("ApplicationWindow", (), {})
gi.repository.Adw.Application = type("Application", (), {})
sys.modules.setdefault("gi", gi)
sys.modules.setdefault("gi.repository", gi.repository)
sys.path.insert(0, str(ROOT / "launcher"))
spec = importlib.util.spec_from_file_location("bbport_launcher", ROOT / "launcher/bbport_launcher.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)

GPUS = ("0\t0x8086\tintegrated\tIntel(R) UHD Graphics 630\t0\n"
        "1\t0x10de\tdiscrete\tNVIDIA GeForce RTX 4060 Laptop GPU\t1\n"
        "2\t0x10005\tcpu\tllvmpipe (LLVM 21.1.8, 256 bits)\t0\n")


class LauncherGpuTests(unittest.TestCase):
    def setUp(self):
        run = mock.patch.object(launcher.subprocess, "run", return_value=subprocess.CompletedProcess(
            [], 0, stdout=GPUS, stderr=""))
        self.run = run.start()
        self.addCleanup(run.stop)
        self.settings = dict(launcher.DEFAULTS, game_dir="/games/CUSA03173", user_dir="")

    def test_gpus_listed_in_vulkans_order(self):
        gpus = launcher.connected_gpus()
        self.assertEqual([(index, name, best) for index, _vendor, _kind, name, best in gpus],
                         [(0, "Intel(R) UHD Graphics 630", False),
                          (1, "NVIDIA GeForce RTX 4060 Laptop GPU", True),
                          (2, "llvmpipe (LLVM 21.1.8, 256 bits)", False)])
        self.assertEqual(self.run.call_args.args[0][1:], ["--gpus"])

    def test_chosen_gpu_passed_by_index(self):
        self.settings["gpu"] = "Intel(R) UHD Graphics 630"
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BB_GPU_ID", None)
            self.assertEqual(launcher.game_environment(self.settings).get("BB_GPU_ID"), "0")

    def test_auto_and_missing_gpu_leave_the_choice_to_the_port(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BB_GPU_ID", None)
            self.assertNotIn("BB_GPU_ID", launcher.game_environment(self.settings))
            self.settings["gpu"] = "AMD Radeon RX 580"
            self.assertNotIn("BB_GPU_ID", launcher.game_environment(self.settings))

    def test_shell_gpu_id_wins(self):
        self.settings["gpu"] = "Intel(R) UHD Graphics 630"
        with mock.patch.dict(os.environ, {"BB_GPU_ID": "1"}):
            self.assertEqual(launcher.game_environment(self.settings)["BB_GPU_ID"], "1")


if __name__ == "__main__":
    unittest.main()
