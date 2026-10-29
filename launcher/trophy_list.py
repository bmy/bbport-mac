# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The game's trophies for the launcher's Trophies page: what the port keeps in the user folder.

gpu/shim/bbport_trophies.cpp keeps <user>/trophy/<NP communication ID>/: unlocked.txt (the ids
unlocked, one a line) and, once the game has run with the player's trophy key, the decrypted
TROP.XML / TROP_XX.XML (XX: the system language) with the TROPnnn.PNG icons. src/runtime_trophy.c
appends every unlock with its time to <user>/trophies.jsonl. Without the XML only the unlocked
trophies are known, by number. No GTK here: tests/test_trophy_list.py checks it."""

import json
import re
from pathlib import Path
from xml.etree import ElementTree

BLOODBORNE_NP_ID = "NPWR05818_00"


def trophy_folder(user_dir):
    """<user>/trophy/<NP communication ID> of the game: Bloodborne's when it is there, else the
    folder written last; None when the game has not kept any trophy yet."""
    root = Path(user_dir) / "trophy"
    if not root.is_dir():
        return None
    folders = [p for p in root.iterdir() if p.is_dir() and
               ((p / "unlocked.txt").is_file() or (p / "TROP.XML").is_file())]
    if not folders:
        return None
    for folder in folders:
        if folder.name == BLOODBORNE_NP_ID:
            return folder
    return max(folders, key=lambda p: p.stat().st_mtime)


def parse_xml(text):
    """The trophies of a TROP.XML text: [{id, grade, platinum, hidden, name, detail}] in its order.
    The file may begin with a signature comment or bytes before the XML."""
    start = text.find("<trophyconf")
    if start < 0:
        return []
    end = text.rfind("</trophyconf>")
    body = text[start:end + len("</trophyconf>")] if end > start else text[start:]
    try:
        root = ElementTree.fromstring(body)
    except ElementTree.ParseError:
        return []
    trophies = []
    for node in root.iter("trophy"):
        try:
            trophy_id = int(node.get("id", ""))
        except ValueError:
            continue
        try:
            platinum = int(node.get("pid", "-1"))
        except ValueError:
            platinum = -1
        trophies.append({
            "id": trophy_id,
            "grade": (node.get("ttype") or "?")[:1],
            "platinum": platinum,
            "hidden": node.get("hidden") == "yes",
            "name": (node.findtext("name") or "").strip(),
            "detail": (node.findtext("detail") or "").strip(),
        })
    return trophies


def unlock_times(user_dir):
    """The first unlock time of each trophy id in <user>/trophies.jsonl ("2026-10-26T21:05:00Z")."""
    times = {}
    path = Path(user_dir) / "trophies.jsonl"
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return times
    for line in lines:
        try:
            event = json.loads(line)
            trophy_id = int(event["trophy_id"])
            stamp = str(event["unlocked_at"])
        except (ValueError, KeyError, TypeError):
            continue
        if trophy_id not in times or stamp < times[trophy_id]:
            times[trophy_id] = stamp
    return times


def unlocked_ids(folder):
    ids = set()
    try:
        lines = (folder / "unlocked.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ids
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and re.fullmatch(r"\d+", line):
            ids.add(int(line))
    return ids


def load(user_dir, language="1"):
    """The Trophies page's content:
    {"folder": Path or None, "named": the names are known, "trophies": [{id, grade, name, detail,
    hidden, unlocked, time, icon}], "unlocked": n, "total": n, "grades": {"P": [got, of], ...}}.
    language: the system language the game sees (the PS4 number: 1 English, 8 Russian)."""
    folder = trophy_folder(user_dir)
    times = unlock_times(user_dir)
    got = (unlocked_ids(folder) if folder else set()) | set(times)
    known = []
    if folder:
        try:
            number = int(language)
        except (TypeError, ValueError):
            number = 1
        for name in (f"TROP_{number:02d}.XML", "TROP.XML"):
            path = folder / name
            if path.is_file():
                known = parse_xml(path.read_text(encoding="utf-8", errors="replace"))
                if known:
                    break
    named = bool(known)
    if not named:
        known = [{"id": i, "grade": "?", "platinum": -1, "hidden": False, "name": "", "detail": ""}
                 for i in sorted(got)]
    trophies = []
    grades = {}
    for trophy in known:
        icon = folder / f"TROP{trophy['id']:03d}.PNG" if folder else None
        entry = dict(trophy, unlocked=trophy["id"] in got, time=times.get(trophy["id"]),
                     icon=icon if icon and icon.is_file() else None)
        trophies.append(entry)
        count = grades.setdefault(entry["grade"], [0, 0])
        count[0] += entry["unlocked"]
        count[1] += 1
    return {"folder": folder, "named": named, "trophies": trophies,
            "unlocked": sum(t["unlocked"] for t in trophies), "total": len(trophies), "grades": grades}


def valid_key(text):
    """Whether text is a trophy key: 32 hexadecimal digits (what BB_TROPHY_KEY takes)."""
    return bool(re.fullmatch(r"[0-9A-Fa-f]{32}", text.strip()))
