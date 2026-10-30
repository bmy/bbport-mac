# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The port's environment variables (gpu/shim/bb_env_vars.def): the list against the code, and
docs/ENVIRONMENT.md made from it.

    python3 scripts/env_vars.py --check   # names read but not listed, or listed and read nowhere
    python3 scripts/env_vars.py --doc     # writes docs/ENVIRONMENT.md
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEF = ROOT / 'gpu/shim/bb_env_vars.def'
DOC = ROOT / 'docs/ENVIRONMENT.md'

KINDS = {
    'SETTING': ('Settings', 'What players and the launchers set. The menu, bbport.ini and the launcher '
                'write most of them; set one by hand to override them.'),
    'TUNING': ('Tuning', 'Defaults to change on a machine where they misbehave, workarounds, and '
               'switches that turn a fix off for comparison.'),
    'DEBUG': ('Diagnostics', 'Logs, dumps, traces, probes and experiments, for finding bugs. Most cost '
              'frame time; none is meant for playing.'),
    'INTERNAL': ('Internal', 'Passed between the launchers, run.py / run.sh and the game. Not meant to '
                 'be set by hand.'),
    'BUILD': ('Build', 'Read by the build and packaging scripts only.'),
}

# The code: C, C++ and Objective-C++ string literals, and the scripts' own lines (not comments).
CODE_DIRS = ('src', 'gpu')
CODE_SKIP = ('third_party', 'vendor')
SCRIPT_FILES = ('run.py', 'run.sh', 'build.sh')
SCRIPT_DIRS = ('launcher', 'scripts', 'tools', 'packaging')
SCRIPT_SKIP = ('__pycache__', 'win-bench', 'ffx')
# BB_* names in the code that are not environment variables (macros, include guards).
NOT_VARIABLES = {'BB_ENV', 'BB_ENV_H'}

ENTRY = re.compile(r'^BB_ENV\((BB_[A-Z0-9_]+), ([A-Z]+), "((?:[^"\\]|\\.)*)"\)$')


def registry():
    entries = {}
    for number, line in enumerate(DEF.read_text(encoding='utf-8').splitlines(), 1):
        if not line.startswith('BB_ENV('):
            continue
        match = ENTRY.match(line)
        if not match or match[2] not in KINDS:
            raise SystemExit(f'{DEF.relative_to(ROOT)}:{number}: not BB_ENV(NAME, KIND, "text")')
        if match[1] in entries:
            raise SystemExit(f'{DEF.relative_to(ROOT)}:{number}: {match[1]} listed twice')
        entries[match[1]] = (match[2], match[3].replace('\\"', '"').replace('\\\\', '\\'))
    return entries


def files(dirs, suffixes, skip):
    for top in dirs:
        for path in sorted((ROOT / top).rglob('*')):
            if path.suffix in suffixes and path.is_file() and not set(path.parts) & set(skip):
                yield path


def used_names():
    """name -> first place the code or a script uses it."""
    used = {}
    literal = re.compile(r'"(BB_[A-Z0-9_]+)"')
    for path in files(CODE_DIRS, ('.c', '.cpp', '.h', '.inc', '.mm'), CODE_SKIP):
        if path == DEF:
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            for name in literal.findall(line):
                used.setdefault(name, f'{path.relative_to(ROOT)}:{number}')
    scripts = [ROOT / name for name in SCRIPT_FILES] + list(
        files(SCRIPT_DIRS, ('.py', '.sh', '.cs', '.json', '.nix', '.cmake', '.txt'), SCRIPT_SKIP))
    token = re.compile(r'(?:(?<=-D)|(?<![A-Za-z0-9_]))BB_[A-Z0-9_]+\b')  # -DBB_X: a CMake option
    for path in scripts:
        if not path.is_file() or path.name == 'CMakeLists.txt' and 'third_party' in path.parts:
            continue
        for number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            code = re.split(r'(?:^|\s)#', line, maxsplit=1)[0] if path.suffix in ('.py', '.sh', '.nix') else line
            code = code.split('//', 1)[0] if path.suffix == '.cs' else code
            for name in token.findall(code):
                used.setdefault(name, f'{path.relative_to(ROOT)}:{number}')
    for name in NOT_VARIABLES:
        used.pop(name, None)
    return used


def check():
    listed, used = registry(), used_names()
    unlisted = sorted(set(used) - set(listed))
    unused = sorted(set(listed) - set(used))
    for name in unlisted:
        print(f'{used[name]}: {name} is not in gpu/shim/bb_env_vars.def')
    for name in unused:
        print(f'gpu/shim/bb_env_vars.def: {name} is read nowhere (remove it)')
    return not unlisted and not unused


def document():
    listed = registry()
    lines = ['# Environment variables', '',
             'Every `BB_*` variable the port reads, by kind. Made by `scripts/env_vars.py --doc` from',
             '`gpu/shim/bb_env_vars.def`; edit that list, not this file. The game prints the variables set',
             'at its start (`Environment:` in the log) and warns about `BB_*` names it does not know.', '',
             '`=1: ...` / `=0: ...` / `=N: ...` say what that value does; `set: ...` means any value',
             'turns it on (even `0`).', '']
    for kind, (title, about) in KINDS.items():
        names = sorted(name for name, (k, _) in listed.items() if k == kind)
        lines += [f'## {title} ({len(names)})', '', about, '', '| Variable | What it does |', '|---|---|']
        lines += [f'| `{name}` | {listed[name][1].replace("|", "/")} |' for name in names]
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--check', action='store_true', help='the list against the code')
    parser.add_argument('--doc', action='store_true', help='write docs/ENVIRONMENT.md')
    args = parser.parse_args()
    ok = True
    if args.check or not args.doc:
        ok = check()
    if args.doc:
        DOC.write_text(document(), encoding='utf-8')
        print(f'Wrote {DOC.relative_to(ROOT)}')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
