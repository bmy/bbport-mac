#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The game's shader programs from its files, and how many of them a shader cache covers.

The idea is from the SrShadowy fork (scripts/extract_static_shaders.py, its "static catalog");
this one also reads the BND4 entry names (gxflvershader: GXFlver_<material>_<pass>[_Skin][_Ins]
.vpo/.ppo/...), which say the pass and the material of each program: the base for predicting the
shaders a player has not met yet.

  shader_catalog.py build GAME_DIR OUT.json
      GAME_DIR: the game's folder (CUSA03173, with dvdroot_ps4/shader). Each program: its hash
      (the "OrbShdr" block's shader_hash, the port's pgm_hash), stage, code size, entry names and
      bundles.
  shader_catalog.py coverage CATALOG.json CACHE_DIR|shader_recipes.bin [...]
      Which catalog programs the caches hold (their 0x<hash>_<perm>.spv/.meta/.in names), by
      bundle, stage and pass; the cache's programs not in the catalog (made at run time: fetch
      shaders, the port's own) are counted apart.
"""
from __future__ import annotations

import collections
import json
import re
import struct
import sys
import zlib
from pathlib import Path

STAGES = {0: "vs", 1: "ps", 2: "gs", 3: "hs", 4: "ds", 5: "cs", 6: "es", 7: "ls"}


def undcx(data: bytes) -> bytes:
    if data[:4] != b"DCX\0" or data[0x28:0x2C] != b"DFLT":
        raise ValueError("not a DCX (DFLT) file")
    size, packed = struct.unpack_from(">II", data, 0x1C)
    out = zlib.decompress(data[0x4C:0x4C + packed])
    if len(out) != size:
        raise ValueError("DCX size mismatch")
    return out


def bnd4_entries(data: bytes):
    """(name, bytes) of each entry. Bloodborne's shader BND4s: 0x24-byte entries (flags, -1,
    compressed and uncompressed sizes, u32 offset, id, name offset), UTF-16 names."""
    if data[:4] != b"BND4":
        raise ValueError("not a BND4")
    count = struct.unpack_from("<I", data, 0x0C)[0]
    entry_size = struct.unpack_from("<Q", data, 0x20)[0]
    unicode = data[0x30] != 0
    if entry_size != 0x24:
        raise ValueError(f"BND4 entries of {entry_size:#x} bytes: not handled")
    for i in range(count):
        at = 0x40 + i * entry_size
        size, _, offset, _, name_at = struct.unpack_from("<QQIiI", data, at + 8)
        if unicode:
            end = name_at
            while data[end:end + 2] != b"\0\0":
                end += 2
            name = data[name_at:end].decode("utf-16-le")
        else:
            name = data[name_at:data.index(b"\0", name_at)].decode("shift_jis", "replace")
        yield name, data[offset:offset + size]


def programs(blob: bytes):
    """(hash, stage, code bytes) of each "OrbShdr" block in a file whose program is found before
    it: its first instruction (s_mov_b32 vcc_hi, literal) gives the block's offset, as the port's
    GameShaders::Scan checks (the executable holds the word elsewhere too)."""
    at = blob.find(b"OrbShdr")
    while at != -1 and at + 28 <= len(blob):
        bits = struct.unpack_from("<I", blob, at + 8)[0]
        shader_hash = struct.unpack_from("<Q", blob, at + 16)[0]
        length = bits >> 8
        if length >= 8 and length % 4 == 0 and length <= at:
            for offset in range(length, min(at, length + 1024) + 1, 4):
                first, literal = struct.unpack_from("<II", blob, at - offset)
                if first == 0xBEEB03FF and (literal + 1) * 8 == offset:
                    yield shader_hash, STAGES.get((bits >> 2) & 0xF, str((bits >> 2) & 0xF)), length
                    break
        at = blob.find(b"OrbShdr", at + 7)


PASSES = ("Gbuf", "Fwd", "FwdSdw", "FwdSSdw", "Dpt", "DptA", "DptV", "DptVA", "DptVCube",
          "DptVACube", "Pnt", "Velo")


def pass_of(name: str) -> str:
    """GXFlver_<material, may have "_">_[Bloody_]<pass>[_Skin][_Ins][@[cl]]: the pass (one of
    PASSES), "Skin"/"Ins" appended; "-" for other bundles and names without a known pass."""
    stem = re.sub(r"\.[a-z]+$", "", name.rsplit("\\", 1)[-1])
    if not stem.startswith("GXFlver_"):
        return "-"
    tokens = [t.split("@")[0] for t in stem.split("_")[1:]]
    found = [t for t in tokens if t in PASSES]
    if not found:
        return "-"
    return "_".join([found[-1]] + [t for t in ("Skin", "Ins") if t in tokens])


def build(game: Path, out: Path) -> int:
    shader_dir = game / "dvdroot_ps4" / "shader"
    if not shader_dir.is_dir():
        shader_dir = game
    catalog: dict[str, dict] = {}
    bundles = {}
    for path in sorted(shader_dir.glob("*.shaderbnd.dcx")):
        data = bnd4_entries_data = undcx(path.read_bytes())
        entries = 0
        for name, blob in bnd4_entries(bnd4_entries_data):
            for shader_hash, stage, length in programs(blob):
                key = f"0x{shader_hash:016x}"
                item = catalog.setdefault(key, {"stage": stage, "code_bytes": length,
                                                "bundles": [], "names": []})
                if path.name not in item["bundles"]:
                    item["bundles"].append(path.name)
                if len(item["names"]) < 8:
                    item["names"].append(name.rsplit("\\", 1)[-1])
                entries += 1
        bundles[path.name] = entries
        del data
    # Programs in the executable itself (the renderer's own).
    eboot = game / "eboot.bin"
    if eboot.is_file():
        entries = 0
        for shader_hash, stage, length in programs(eboot.read_bytes()):
            catalog.setdefault(f"0x{shader_hash:016x}", {"stage": stage, "code_bytes": length,
                                                        "bundles": ["eboot.bin"], "names": []})
            entries += 1
        bundles["eboot.bin"] = entries
    out.write_text(json.dumps({"format": "bbport-shader-catalog-1", "bundles": bundles,
                               "programs": catalog}, indent=1), encoding="utf-8")
    stages = collections.Counter(p["stage"] for p in catalog.values())
    print(f"{len(catalog)} programs ({sum(bundles.values())} entries) in {len(bundles)} bundles: "
          + ", ".join(f"{s} {n}" for s, n in stages.most_common()))
    for name, n in bundles.items():
        print(f"  {name:36s} {n:6d}")
    return 0


def cache_hashes(directory: Path) -> set[int]:
    found = set()
    for path in directory.rglob("0x*_*.*"):
        match = re.match(r"0x([0-9a-f]{16})_\d+\.(spv|meta|in)$", path.name)
        if match:
            found.add(int(match.group(1), 16))
    return found


def recipe_hashes(path: Path) -> set[int]:
    """The programs of a recipe pack (shader_recipes.bin: "BBSR", six u32 versions, the raw size,
    a zlib stream: u32 count, then per program u64 hash, u32 perms, per perm two blobs and refs)."""
    data = path.read_bytes()
    if data[:4] != b"BBSR":
        raise ValueError(f"{path}: not a recipe pack")
    body = zlib.decompress(data[4 + 5 * 4 + 8:])
    at = 0

    def take(fmt):
        nonlocal at
        values = struct.unpack_from(fmt, body, at)
        at += struct.calcsize(fmt)
        return values

    found = set()
    for _ in range(take("<I")[0]):
        found.add(take("<Q")[0])
        for _ in range(take("<I")[0]):
            for _ in range(2):
                blob = take("<I")[0]
                at += blob
            refs = take("<I")[0]
            at += refs * 20
    return found


def coverage(catalog_path: Path, caches: list[Path]) -> int:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))["programs"]
    known = {int(k, 16): v for k, v in catalog.items()}
    seen = set()
    for source in caches:
        seen |= recipe_hashes(source) if source.is_file() else cache_hashes(source)
    matched = seen & known.keys()
    print(f"cache programs {len(seen)}: {len(matched)} in the game's files, "
          f"{len(seen) - len(matched)} made at run time; catalog covered {len(matched)} of "
          f"{len(known)} ({100.0 * len(matched) / max(1, len(known)):.1f} %)")
    for title, key in (("bundle", lambda p: p["bundles"][0]), ("stage", lambda p: p["stage"]),
                       ("pass", lambda p: pass_of(p["names"][0]) if p["names"] else "(eboot, no names)")):
        total, hit = collections.Counter(), collections.Counter()
        for h, program in known.items():
            k = key(program)
            total[k] += 1
            hit[k] += h in matched
        print(f"by {title}:")
        for k, n in total.most_common(14):
            print(f"  {k:36s} {hit[k]:5d} / {n:5d}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[0] == "build":
        return build(Path(argv[1]), Path(argv[2]))
    if len(argv) >= 3 and argv[0] == "coverage":
        return coverage(Path(argv[1]), [Path(p) for p in argv[2:]])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
