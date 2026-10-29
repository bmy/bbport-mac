# SPDX-FileCopyrightText: Copyright 2026 bbport contributors
# SPDX-License-Identifier: GPL-2.0-or-later
from paths import ROOT
import contextlib
import io
import os
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import game_check
from test_prepare import fixture


def dcx(payload, packed=None):
    """A DFLT .dcx file as the game's (sizes big-endian, zlib data at 0x4C)."""
    packed = zlib.compress(payload) if packed is None else packed
    header = bytearray(0x4C)
    header[0:4], header[0x18:0x1C], header[0x24:0x28], header[0x28:0x2C] = (
        b'DCX\0', b'DCS\0', b'DCP\0', b'DFLT')
    struct.pack_into('>II', header, 0x1C, len(payload), len(packed))
    header[0x44:0x48] = b'DCA\0'
    return bytes(header) + packed


def param_sfo(entries):
    """A minimal param.sfo with UTF-8 string entries."""
    keys = b''.join(k.encode() + b'\0' for k in entries)
    values = [v.encode() + b'\0' for v in entries.values()]
    count = len(entries)
    key_table = 20 + 16 * count
    data_table = key_table + len(keys)
    data = bytearray(struct.pack('<4sIIII', b'\0PSF', 0x101, key_table, data_table, count))
    key_off = data_off = 0
    for key, value in zip(entries, values):
        data += struct.pack('<HHIII', key_off, 0x204, len(value), len(value), data_off)
        key_off += len(key) + 1
        data_off += len(value)
    return bytes(data + keys + b''.join(values))


class GameCheckTests(unittest.TestCase):
    def game(self, title, version, eboot=None):
        path = Path(self.tmp.name)
        (path / 'sce_sys').mkdir(exist_ok=True)
        (path / 'sce_sys/param.sfo').write_bytes(param_sfo({'APP_VER': version, 'TITLE_ID': title}))
        (path / 'eboot.bin').write_bytes(fixture() if eboot is None else eboot)
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ.pop('BB_SKIP_GAME_CHECK', None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_base_game_without_the_update(self):
        self.assertEqual(game_check.problem(self.game('CUSA03173', '01.00')),
                         ('missing_update', 'CUSA03173', '01.00'))

    def test_update_metadata_with_an_old_executable(self):
        self.assertEqual(game_check.problem(self.game('CUSA03173', '01.09'))[0], 'wrong_eboot')

    def test_other_edition(self):
        # The EU alpha test: not a retail release.
        self.assertEqual(game_check.problem(self.game('CUSA01322', '01.09'))[0], 'other_title')

    def test_unreadable_executable(self):
        self.assertEqual(game_check.problem(self.game('CUSA03173', '01.09', b'junk'))[0], 'unreadable')

    def test_supported_image_passes(self):
        game = self.game('CUSA03173', '01.09')
        with mock.patch.object(game_check, 'SUPPORTED_IMAGE', game_check.image_sha256(game)):
            self.assertIsNone(game_check.problem(game))

    def shaders(self, game, data):
        folder = game / game_check.CHECKED_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'gxshader.shaderbnd.dcx').write_bytes(data)

    def test_intact_shaders_pass(self):
        game = self.game('CUSA03173', '01.09')
        self.shaders(game, dcx(bytes(range(256)) * 64))
        with mock.patch.object(game_check, 'SUPPORTED_IMAGE', game_check.image_sha256(game)):
            self.assertIsNone(game_check.problem(game))

    def test_damaged_shaders(self):
        # As LibOrbisPkg on .NET 6+ (issue #81): part of the data replaced by stale bytes.
        game = self.game('CUSA03173', '01.09')
        packed = bytearray(zlib.compress(os.urandom(4096)))
        packed[len(packed) // 2:] = packed[:len(packed) - len(packed) // 2]
        self.shaders(game, dcx(os.urandom(4096), bytes(packed)))
        with mock.patch.object(game_check, 'SUPPORTED_IMAGE', game_check.image_sha256(game)):
            self.assertEqual(game_check.problem(game)[0], 'damaged_files')
        self.assertEqual(game_check.broken_files(game),
                         [f'{game_check.CHECKED_FOLDER}/gxshader.shaderbnd.dcx'])

    def test_other_compressions_are_not_checked(self):
        game = self.game('CUSA03173', '01.09')
        data = bytearray(dcx(b'x' * 100))
        data[0x28:0x2C] = b'KRAK'
        self.shaders(game, bytes(data[:0x50]))
        self.assertEqual(game_check.broken_files(game), [])

    def test_every_retail_base_game_without_the_update(self):
        for title in game_check.SUPPORTED_TITLES:
            with self.subTest(title=title):
                self.assertEqual(game_check.problem(self.game(title, '01.00'))[0], 'missing_update')

    def test_eboot_with_the_60fps_patch(self):
        game = self.game('CUSA00900', '01.09')
        with mock.patch.object(game_check, 'LANCE_60FPS_IMAGE', game_check.image_sha256(game)):
            self.assertEqual(game_check.problem(game)[0], 'patched_eboot')

    def test_skip_switch(self):
        os.environ['BB_SKIP_GAME_CHECK'] = '1'
        self.assertIsNone(game_check.problem(self.game('CUSA03173', '01.00')))

    def test_every_problem_is_explained(self):
        for kind in ('missing_update', 'wrong_eboot', 'patched_eboot', 'other_title',
                     'unreadable', 'damaged_files'):
            self.assertIn('CUSA', game_check.explain(kind, 'CUSA03173', '01.00'))


def tpf(width, height, data_bytes=0):
    """A PS4 TPF with one texture; random data so its .dcx stays as large as the data."""
    entry = struct.pack('<IiBBBBHH', 0x34, data_bytes, 0, 0, 1, 0, width, height) + bytes(8)
    return (struct.pack('<4sIiBBBB', b'TPF\0', data_bytes, 1, 4, 3, 1, 0) + entry +
            os.urandom(data_bytes))


def tpf_archive(folder, name, textures):
    """name.tpfbhd/.tpfbdt as the game's maps have them (BHF4, 36-byte entries)."""
    folder.mkdir(parents=True, exist_ok=True)
    blobs = [dcx(t) for t in textures]
    bhd = bytearray(b'BHF4' + bytes(0x3C))
    struct.pack_into('<i', bhd, 0x0C, len(blobs))
    struct.pack_into('<q', bhd, 0x20, 36)
    offset = 0x30
    for blob in blobs:
        bhd += struct.pack('<BxxxiqqIiI', 0x40, -1, len(blob), len(blob), offset, 0, 0)
        offset += len(blob)
    (folder / f'{name}.tpfbhd').write_bytes(bytes(bhd))
    (folder / f'{name}.tpfbdt').write_bytes(bytes(0x30) + b''.join(blobs))


class TexturePackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.game = Path(self.tmp.name)
        self.maps = self.game / 'dvdroot_ps4/map'
        self.large = game_check.TEXTURE_PACK_MIN_ENTRY

    def tearDown(self):
        self.tmp.cleanup()

    def test_retail_sized_textures_are_no_pack(self):
        tpf_archive(self.maps / 'm24', 'm24_0000', [tpf(4096, 4096, self.large), tpf(512, 512)])
        self.assertIsNone(game_check.texture_pack(self.game))

    def test_texture_larger_than_the_retail_game(self):
        # #240: an 8192x2048 sky, the retail one 2048x512.
        tpf_archive(self.maps / 'm24', 'm24_0000', [tpf(512, 512), tpf(8192, 2048, self.large)])
        self.assertEqual(game_check.texture_pack(self.game),
                         ('dvdroot_ps4/map/m24/m24_0000.tpfbhd', 8192))

    def test_loose_texture_file(self):
        self.maps.mkdir(parents=True)
        (self.maps / 'm24_envmap.tpf.dcx').write_bytes(dcx(tpf(2048, 8192, self.large)))
        self.assertEqual(game_check.texture_pack(self.game),
                         ('dvdroot_ps4/map/m24_envmap.tpf.dcx', 8192))

    def test_linked_folders_of_the_merged_mod_view(self):
        # mods.py links the folders no mod changes: map/m24 is a link there.
        original = Path(self.tmp.name) / 'original'
        tpf_archive(original / 'm24', 'm24_0000', [tpf(8192, 2048, self.large)])
        self.maps.mkdir(parents=True)
        (self.maps / 'm24').symlink_to(original / 'm24', target_is_directory=True)
        self.assertEqual(game_check.texture_pack(self.game),
                         ('dvdroot_ps4/map/m24/m24_0000.tpfbhd', 8192))

    def test_small_entries_are_not_opened(self):
        # An oversized texture always packs larger: smaller entries are skipped unread.
        tpf_archive(self.maps / 'm24', 'm24_0000', [tpf(8192, 2048)])
        self.assertIsNone(game_check.texture_pack(self.game))

    def test_answer_kept_until_a_texture_file_changes(self):
        cache = self.game / 'texture-pack.json'
        tpf_archive(self.maps / 'm27', 'm27_0000', [tpf(8192, 8192, self.large)])
        with contextlib.redirect_stderr(io.StringIO()) as told:
            found = game_check.texture_pack(self.game, cache)
        self.assertIn('Texture check', told.getvalue())
        with mock.patch.object(game_check, 'find_texture_pack') as scan, \
                contextlib.redirect_stderr(io.StringIO()) as told:
            self.assertEqual(game_check.texture_pack(self.game, cache), found)
        scan.assert_not_called()
        self.assertEqual(told.getvalue(), '')
        tpf_archive(self.maps / 'm27', 'm27_0000', [tpf(2048, 2048, self.large + 1)])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertIsNone(game_check.texture_pack(self.game, cache))

    def test_command_line_for_run_sh(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(game_check.main(['game_check.py', '--texture-pack', str(self.game)]), 0)
        self.assertEqual(out.getvalue(), '0\n')
        tpf_archive(self.maps / 'm24', 'm24_0000', [tpf(8192, 2048, self.large)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            game_check.main(['game_check.py', '--texture-pack', str(self.game)])
        self.assertTrue(out.getvalue().startswith('1 a texture pack: dvdroot_ps4/map/m24/m24_0000'))


if __name__ == '__main__':
    unittest.main()
