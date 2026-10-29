#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
"""The game files bbport runs: Bloodborne with the 1.09 update merged in. The retail releases (the
US CUSA00900, the European Game of the Year edition CUSA03173, the Asian The Old Hunters Edition
CUSA03023, ...) share the same 1.09 executable.

Other versions start and then fail inside the game's code (the base game 1.00 faults at guest
offset 0x20348b8): hooks and patches use the addresses of this one executable. The check compares
the loaded executable image, whose hash is the same whatever tool dumped it (the SELF headers
around it differ). The shaders must also unpack: an extraction tool that leaves
parts of compressed sectors stale (LibOrbisPkg PkgTool on .NET 6 or newer, issue #81) makes the
game hang while it loads them, with no error. BB_SKIP_GAME_CHECK=1 skips both.

Run as a script, it checks every .dcx file of a game folder: game_check.py GAME_DIR
(game_check.py --texture-pack GAME_DIR [CACHE]: whether the folder holds a texture pack, for run.sh)
"""
import hashlib
import json
import os
import struct
import sys
import zlib
from pathlib import Path

# The retail releases (the list of content IDs from PR #117): Bloodborne, then the editions with
# The Old Hunters. The executable decides in the end: SUPPORTED_IMAGE.
SUPPORTED_TITLES = ('CUSA00900', 'CUSA00207', 'CUSA00208', 'CUSA00299', 'CUSA01363',  # US EU UK JP AS
                    'CUSA03179', 'CUSA03173', 'CUSA03014', 'CUSA03023')  # US EU JP AS
SUPPORTED_VERSION = '01.09'
SUPPORTED_IMAGE = '071df19c8880086d97182dbc057bc8cb37badaca57d9112683836b24a0444c0a'
# The 1.09 executable with Lance McDonald's 60 fps patch applied to it, as some dumps ship it
# (seen on US CUSA00900): one of its edits inverts a branch in the menu code (+0xf6d90d) and the
# gestures menu crashes the game. bbport has its own 60 fps; the clean 1.09 eboot.bin is needed.
LANCE_60FPS_IMAGE = 'bbc91f4dff6bc3118b039464289e39878a8fdfc802a07fab9d5fe84d935a676f'
# The game loads these first; a broken extraction damages most files, these among them.
CHECKED_FOLDER = 'dvdroot_ps4/shader'


def image_sha256(game):
    """The SHA-256 of eboot.bin's loaded image (its loadable segments at their addresses)."""
    from prepare import parse_self, span  # same directory
    elf, header, ph, _segments, _missing = parse_self((Path(game) / 'eboot.bin').read_bytes())
    loads = [p for p in ph if p['type'] in (1, 0x61000010)]
    image = bytearray(max(p['vaddr'] + p['memsz'] for p in loads))
    for p in loads:
        image[p['vaddr']:p['vaddr'] + p['filesz']] = span(elf, p['offset'], p['filesz'])
    return hashlib.sha256(image).hexdigest()


def dcx_broken(path):
    """True when a DFLT .dcx file does not unpack to its stated size or fails zlib's checksum.
    Other compressions are not checked."""
    data = Path(path).read_bytes()
    if data[:4] != b'DCX\0' or data[0x18:0x1C] != b'DCS\0' or data[0x28:0x2C] != b'DFLT':
        return False
    size, packed = struct.unpack_from('>II', data, 0x1C)
    try:
        return len(zlib.decompress(data[0x4C:0x4C + packed])) != size
    except zlib.error:
        return True


def broken_files(game, folder=CHECKED_FOLDER):
    """The .dcx files under the game's folder that do not unpack (paths relative to the game)."""
    root = Path(game)
    return [str(f.relative_to(root)) for f in sorted((root / folder).rglob('*.dcx')) if dcx_broken(f)]


# Texture packs (#240): upscaled map textures, baked into the game folder or merged from the mods
# folder, fill the game's graphics heaps where the retail game fits. The game's own GPU allocations
# then fail: textures flicker, and the effects of the Forbidden Woods (loaded behind Iosefka's
# Clinic) got a missing buffer and crashed the game (139). run.sh gives such a folder the larger
# heaps of the resolution patches above 1080p. The retail 1.09 maps' textures are at most 4096
# pixels on a side; the pack of #240 has 8192x2048 ones.
RETAIL_TEXTURE_SIDE = 4096
# Only archive entries this large are opened: a texture with a side above 4096 packs to more (in
# the pack of #240, 4.2 MiB and up; 8192x256 BC1 with mips is 1.4 MiB unpacked). One entry in
# twelve of the retail maps.
TEXTURE_PACK_MIN_ENTRY = 256 << 10


def texture_files(game):
    """The map texture archives (.tpfbhd, read with their .tpfbdt) and loose .tpf.dcx files.
    Directory links are followed: mods.py's merged view links the folders no mod changes."""
    found = []
    for folder, _dirs, names in os.walk(Path(game) / 'dvdroot_ps4/map', followlinks=True):
        for name in names:
            if name.casefold().endswith(('.tpfbhd', '.tpf.dcx')):
                found.append(Path(folder) / name)
    return sorted(found)


def tpf_side(blob):
    """The larger side of the first texture in a DFLT .dcx PS4 TPF, 0 for anything else."""
    if blob[:4] != b'DCX\0' or blob[0x28:0x2C] != b'DFLT':
        return 0
    try:
        tpf = zlib.decompressobj().decompress(blob[0x4C:], 32)
    except zlib.error:
        return 0
    # TPF header (16 bytes, platform 4: PS4), then the first texture: offset, size, format, type,
    # mipmaps, flags, width, height.
    if tpf[:4] != b'TPF\0' or len(tpf) < 32 or tpf[12] != 4:
        return 0
    return max(struct.unpack_from('<HH', tpf, 28))


def archive_side(bhd_path):
    """The larger side of the first oversized texture in a .tpfbhd/.tpfbdt pair, else 0."""
    bdt_path = next((p for p in bhd_path.parent.iterdir() if p.name.casefold() ==
                     bhd_path.name[:-3].casefold() + 'bdt'), None)
    bhd = bhd_path.read_bytes()
    # BHF4 with the game's 36-byte entries: flags, -1, packed size, unpacked size, offset, id, name.
    if bdt_path is None or bhd[:4] != b'BHF4' or struct.unpack_from('<q', bhd, 0x20)[0] != 36:
        return 0
    count = struct.unpack_from('<i', bhd, 0x0C)[0]
    with open(bdt_path, 'rb') as bdt:
        for i in range(count):
            packed, = struct.unpack_from('<q', bhd, 0x40 + 36 * i + 8)
            offset, = struct.unpack_from('<I', bhd, 0x40 + 36 * i + 24)
            if packed < TEXTURE_PACK_MIN_ENTRY:
                continue
            bdt.seek(offset)
            side = tpf_side(bdt.read(0x4C + 512))
            if side > RETAIL_TEXTURE_SIDE:
                return side
    return 0


def find_texture_pack(game, files):
    """(file, side) for the first of `files` with a texture larger than the retail game's."""
    for path in files:
        try:
            if path.name.casefold().endswith('.tpfbhd'):
                side = archive_side(path)
            elif path.stat().st_size >= TEXTURE_PACK_MIN_ENTRY:
                with open(path, 'rb') as f:
                    side = tpf_side(f.read(0x4C + 512))
            else:
                side = 0
        except (OSError, struct.error, IndexError):
            continue
        if side > RETAIL_TEXTURE_SIDE:
            return str(path.relative_to(game)), side
    return None


def texture_pack(game, cache=None):
    """(file, side): a map texture of the game folder larger than the retail game's (its file,
    relative to the game, and its larger side), or None. `cache`: a JSON file that keeps the
    answer until a texture file's size or time changes (the first look reads ~8600 entries)."""
    game = Path(game)
    files = texture_files(game)
    stats = []
    for path in files:
        try:
            st = path.stat()
        except OSError:
            continue
        stats.append((str(path.relative_to(game)), st.st_size, st.st_mtime_ns))
    key = hashlib.sha256(repr(stats).encode()).hexdigest()
    if cache:
        try:
            saved = json.loads(Path(cache).read_text(encoding='utf-8'))
            if saved.get('key') == key:
                return tuple(saved['found']) if saved.get('found') else None
        except (OSError, ValueError, TypeError, KeyError):
            pass
        print(f'Texture check: looking for a texture pack in the map textures (once, kept in '
              f'{Path(cache).name})', file=sys.stderr)
    found = find_texture_pack(game, files)
    if cache:
        try:
            Path(cache).write_text(json.dumps({'key': key, 'found': found}), encoding='utf-8')
        except OSError:
            pass
    return found


def problem(game, image_hash=None):
    """None for the supported game, else (kind, title, version): kind is 'missing_update' (base
    game or an older update), 'wrong_eboot' (param.sfo says 1.09, eboot.bin is another version),
    'patched_eboot' (the 1.09 eboot.bin with a 60 fps patch baked in), 'other_title' (another
    edition), 'unreadable' or 'damaged_files' (the right game, but its shaders do not unpack: a
    broken extraction)."""
    if os.environ.get('BB_SKIP_GAME_CHECK') == '1':
        return None
    from prepare import sfo
    try:
        info = sfo((Path(game) / 'sce_sys/param.sfo').read_bytes())
    except (OSError, ValueError, IndexError):
        info = {}
    title, version = info.get('TITLE_ID', '?'), info.get('APP_VER', '?')
    try:
        image_hash = image_hash or image_sha256(game)
        if image_hash == SUPPORTED_IMAGE:
            return ('damaged_files', title, version) if broken_files(game) else None
    except (OSError, ValueError, IndexError, StopIteration, KeyError):
        return 'unreadable', title, version
    if image_hash == LANCE_60FPS_IMAGE:
        return 'patched_eboot', title, version
    if title not in SUPPORTED_TITLES:
        return 'other_title', title, version
    if version != SUPPORTED_VERSION:
        return 'missing_update', title, version
    return 'wrong_eboot', title, version


def explain(kind, title, version):
    """What is wrong and what to do, for the log (English)."""
    found = f'Found {title} version {version} (sce_sys/param.sfo).'
    return {
        'missing_update': f'{found} bbport needs the 1.09 update merged into the game folder: copy '
                          'everything from the dumped 1.09 update (e.g. CUSA03173-patch) into the '
                          'game folder, replacing files (eboot.bin and sce_sys too).',
        'wrong_eboot': f'{found} param.sfo is from 1.09 but eboot.bin is not: copy eboot.bin from '
                       'the dumped 1.09 update into the game folder, replacing the old one.',
        'patched_eboot': f'{found} This eboot.bin has the 60 fps patch by Lance McDonald applied '
                         'to it, which crashes the game when the gestures menu opens (bbport has its '
                         'own 60 fps): copy eboot.bin from a clean dump of the 1.09 update into the '
                         'game folder, replacing this one.',
        'other_title': f'{found} This is not a retail release of Bloodborne: bbport runs Bloodborne '
                       '(any region, The Old Hunters editions too) with update 1.09.',
        'unreadable': f'{found} eboot.bin could not be read as a decrypted PS4 executable: dump '
                      'the game and the 1.09 update again.',
        'damaged_files': f'{found} The game files are damaged: the shaders in {CHECKED_FOLDER} do '
                         'not unpack, and the game would hang while loading them. The extraction '
                         'tool left parts of compressed sectors stale (LibOrbisPkg PkgTool on '
                         '.NET 6 or newer, issue #81): extract the game and the 1.09 update again '
                         'with a fixed tool. scripts/game_check.py GAME_DIR checks every file.',
    }[kind]


def main(argv):
    """game_check.py GAME_DIR: every .dcx file of the game, those that do not unpack listed.
    game_check.py --texture-pack GAME_DIR [CACHE]: "1 <what was found>" for a texture pack, else "0"."""
    if len(argv) in (3, 4) and argv[1] == '--texture-pack':
        found = texture_pack(argv[2], argv[3] if len(argv) == 4 else None)
        print(f'1 a texture pack: {found[0]} has {found[1]}-pixel textures, the retail game '
              f'{RETAIL_TEXTURE_SIDE} at most' if found else '0')
        return 0
    if len(argv) != 2:
        print('Usage: game_check.py GAME_DIR | --texture-pack GAME_DIR [CACHE]', file=sys.stderr)
        return 2
    game = Path(argv[1])
    files = sorted((game / 'dvdroot_ps4').rglob('*.dcx'))
    broken = []
    for number, path in enumerate(files, 1):
        if dcx_broken(path):
            broken.append(path)
            print(f'damaged: {path.relative_to(game)}')
        if number % 1000 == 0:
            print(f'{number} of {len(files)} checked', file=sys.stderr)
    print(f'{len(broken)} of {len(files)} .dcx files do not unpack')
    return 1 if broken else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
