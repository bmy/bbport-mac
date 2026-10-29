# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The environment variable list (gpu/shim/bb_env_vars.def) against the code, and its document."""
from paths import ROOT
import contextlib
import io
import unittest

import env_vars


class EnvVarsTests(unittest.TestCase):
    def test_list_matches_the_code(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = env_vars.check()
        self.assertTrue(ok, out.getvalue())

    def test_document_is_current(self):
        self.assertEqual((ROOT / 'docs/ENVIRONMENT.md').read_text(encoding='utf-8'), env_vars.document(),
                         'run: python3 scripts/env_vars.py --doc')


if __name__ == '__main__':
    unittest.main()
