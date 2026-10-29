// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The trophy data without the game (gpu/shim/bbport_trophy_data.h): the package's entries, the NP
// communication ID, the padding, the XML, the platinum rule, the key's text. Synthetic data only:
// no game files and no key.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <set>
#include <string>
#include <vector>
#include "bbport_trophy_data.h"

using namespace BbTrophy;

static void Put(std::vector<std::uint8_t>& d, std::size_t at, std::uint64_t v, int bytes) {
    for (int i = bytes - 1; i >= 0; --i, v >>= 8) {
        d[at + i] = std::uint8_t(v);
    }
}

static void Package() {
    // Two entries of 64 bytes from 0x60, their data after.
    std::vector<std::uint8_t> trp(0x60 + 2 * 64 + 20);
    Put(trp, 0, 0xDCA24D00u, 4);
    Put(trp, 16, 2, 4);
    Put(trp, 20, 64, 4);
    const char* names[] = {"TROP.ESFM", "TROP000.PNG"};
    for (int i = 0; i < 2; ++i) {
        const std::size_t at = 0x60 + i * 64;
        for (std::size_t c = 0; names[i][c]; ++c) trp[at + c] = std::uint8_t(names[i][c]);
        Put(trp, at + 32, 0x60 + 2 * 64 + i * 10, 8);
        Put(trp, at + 40, 10, 8);
    }
    const auto entries = ReadTrpEntries(trp);
    assert(entries.size() == 2 && entries[0].name == "TROP.ESFM" && entries[1].name == "TROP000.PNG");
    assert(entries[1].offset == 0x60 + 128 + 10 && entries[1].size == 10);
    // An entry past the end, a wrong magic: nothing.
    auto broken = trp;
    Put(broken, 0x60 + 40, 1000, 8);
    assert(ReadTrpEntries(broken).empty());
    broken = trp;
    broken[0] = 0;
    assert(ReadTrpEntries(broken).empty());
}

static void Ids() {
    std::vector<std::uint8_t> npbind(0x84 + 12 + 20, 0);
    const std::string id = "NPWR00001_00";
    for (std::size_t i = 0; i < id.size(); ++i) npbind[0x84 + i] = std::uint8_t(id[i]);
    assert(NpCommId(npbind) == id);
    npbind[0x84] = 'X';
    assert(!NpCommId(npbind));
    assert(!NpCommId(std::vector<std::uint8_t>(10)));
}

static void Padding() {
    const std::string text = "<trophyconf/>";
    std::vector<std::uint8_t> padded(text.begin(), text.end());
    padded.insert(padded.end(), 3, 3);
    assert(StripPadding(padded) == text);
    std::vector<std::uint8_t> plain(text.begin(), text.end()); // ends in '>': not padding
    assert(StripPadding(plain) == text);
}

static void Xml() {
    const std::string xml = "<!--sig-->\n<trophyconf version=\"1.1\">\n"
                            " <title-name>Test</title-name>\n"
                            " <trophy id=\"000\" hidden=\"no\" ttype=\"P\" pid=\"-1\">\n"
                            "  <name>All of it</name>\n  <detail>Everything.</detail>\n </trophy>\n"
                            " <trophy id=\"001\" hidden=\"yes\" ttype=\"G\" pid=\"000\">\n"
                            "  <name>Fish &amp; Chips</name>\n  <detail>A &quot;test&quot;.</detail>\n </trophy>\n"
                            " <trophy id=\"002\" hidden=\"no\" ttype=\"B\" pid=\"000\"/>\n"
                            " <trophy id=\"003\" hidden=\"no\" ttype=\"S\" pid=\"-1\">\n  <name>Extra</name>\n </trophy>\n"
                            "</trophyconf>\n";
    const auto t = ParseTrophies(xml);
    assert(t.size() == 4);
    assert(t[0].id == 0 && t[0].grade == 'P' && t[0].platinum == -1 && t[0].name == "All of it");
    assert(t[1].id == 1 && t[1].grade == 'G' && t[1].platinum == 0 && t[1].hidden);
    assert(t[1].name == "Fish & Chips" && t[1].detail == "A \"test\".");
    assert(t[2].id == 2 && t[2].grade == 'B' && t[2].name.empty());
    assert(t[3].grade == 'S' && t[3].platinum == -1);

    // The platinum comes with the last trophy counting towards it, once; a second unlock of the
    // same trophy announces nothing; trophies outside it (003) never bring it.
    std::set<int> unlocked;
    assert((Unlock(t, unlocked, 1) == std::vector<int>{1}));
    assert(Unlock(t, unlocked, 1).empty());
    assert((Unlock(t, unlocked, 3) == std::vector<int>{3}));
    assert((Unlock(t, unlocked, 2) == std::vector<int>{2, 0}));
    assert(unlocked.contains(0) && Unlock(t, unlocked, 0).empty());
    // Unknown names (no key): ids alone, no platinum rule.
    std::set<int> bare;
    assert((Unlock({}, bare, 7) == std::vector<int>{7}) && Unlock({}, bare, 7).empty());
}

static void Keys() {
    const auto key = ParseKey("00112233445566778899AABBCCDDeeff\n");
    assert(key && key->size() == 16 && (*key)[0] == 0x00 && (*key)[1] == 0x11 && (*key)[15] == 0xFF);
    assert(ParseKey("\"00112233445566778899aabbccddeeff\""));
    assert(!ParseKey("0011"));
    assert(!ParseKey("zz112233445566778899aabbccddeeff"));
}

int main() {
    Package();
    Ids();
    Padding();
    Xml();
    Keys();
    std::printf("trophy-test: ok\n");
}
