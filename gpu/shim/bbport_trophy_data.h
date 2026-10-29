// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game's trophy data without the system around it (bbport_trophies.cpp; the layout
// and the key scheme as shadPS4's trp.cpp and np_trophy.cpp have them). trophy00.trp in
// sce_sys/trophy is a package of named entries (big-endian header, 64-byte entries from 0x60);
// its .ESFM entries are XML encrypted with AES-128-CBC: the first 16 bytes the IV, the key the
// trophy key's AES of the title's NP communication ID (npbind.dat, padded to 16 bytes). TROP.XML
// and TROP_XX.XML (XX: the system language) list each trophy: id, grade (ttype P G S B), the
// platinum it counts towards (pid), name, detail. Kept apart from Vulkan and OpenSSL: tested in
// tests/test_trophies.cpp.

#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace BbTrophy {

struct TrpEntry {
    std::string name;
    std::uint64_t offset = 0, size = 0;
};

/// The package's entries; empty when it is not a trophy package or an entry lies outside it.
std::vector<TrpEntry> ReadTrpEntries(std::span<const std::uint8_t> trp);

/// The NP communication ID in npbind.dat ("NPWR05818_00"); nullopt when there is none.
std::optional<std::string> NpCommId(std::span<const std::uint8_t> npbind);

/// A decrypted ESFM: the PKCS#7 padding removed (left as it is when it does not look like one).
std::string StripPadding(std::span<const std::uint8_t> plain);

struct Trophy {
    int id = -1;
    char grade = '?'; // P platinum, G gold, S silver, B bronze
    int platinum = -1; // the platinum trophy it counts towards (-1: none)
    bool hidden = false;
    std::string name, detail;
};

/// The trophies of a TROP.XML / TROP_XX.XML text (entities decoded).
std::vector<Trophy> ParseTrophies(std::string_view xml);

/// The trophies to announce when the game unlocks `id`, in order: `id` unless it was unlocked
/// before, then its platinum once every trophy counting towards it is unlocked. All of them
/// are added to `unlocked`.
std::vector<int> Unlock(const std::vector<Trophy>& trophies, std::set<int>& unlocked, int id);

/// The hex key ("21F4...", 32 digits) as 16 bytes; nullopt when it is not one.
std::optional<std::vector<std::uint8_t>> ParseKey(std::string_view hex);

} // namespace BbTrophy
