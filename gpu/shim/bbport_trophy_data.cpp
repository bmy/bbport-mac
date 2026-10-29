// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "bbport_trophy_data.h"

#include <algorithm>
#include <cstdlib>

namespace BbTrophy {
namespace {

std::uint64_t Be(std::span<const std::uint8_t> d, std::size_t at, int bytes) {
    std::uint64_t v = 0;
    for (int i = 0; i < bytes; ++i) {
        v = v << 8 | d[at + i];
    }
    return v;
}

// An attribute's value in one start tag ("" when absent).
std::string_view Attribute(std::string_view tag, std::string_view name) {
    const std::string key = " " + std::string(name) + "=\"";
    const auto at = tag.find(key);
    if (at == std::string_view::npos) {
        return {};
    }
    const auto from = at + key.size();
    const auto to = tag.find('"', from);
    return to == std::string_view::npos ? std::string_view{} : tag.substr(from, to - from);
}

std::string Decode(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '&') {
            out += text[i];
            continue;
        }
        const auto end = text.find(';', i);
        if (end == std::string_view::npos) {
            out += text[i];
            continue;
        }
        const std::string_view entity = text.substr(i + 1, end - i - 1);
        if (entity == "amp") out += '&';
        else if (entity == "lt") out += '<';
        else if (entity == "gt") out += '>';
        else if (entity == "quot") out += '"';
        else if (entity == "apos") out += '\'';
        else {
            out.append(text.substr(i, end - i + 1)); // numeric and others as they are
        }
        i = end;
    }
    return out;
}

// The text of <element>...</element> inside `body` ("" when absent).
std::string Element(std::string_view body, std::string_view element) {
    const std::string open = "<" + std::string(element) + ">", close = "</" + std::string(element) + ">";
    const auto from = body.find(open);
    if (from == std::string_view::npos) {
        return {};
    }
    const auto to = body.find(close, from + open.size());
    return to == std::string_view::npos ? std::string{}
                                        : Decode(body.substr(from + open.size(), to - from - open.size()));
}

int Number(std::string_view text, int fallback) {
    if (text.empty()) {
        return fallback;
    }
    const std::string s(text);
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    return end == s.c_str() ? fallback : int(v);
}

} // namespace

std::vector<TrpEntry> ReadTrpEntries(std::span<const std::uint8_t> trp) {
    // Header: magic DCA24D00, version, file size (8), entry count, entry size; entries from 0x60.
    if (trp.size() < 0x60 || Be(trp, 0, 4) != 0xDCA24D00u) {
        return {};
    }
    const std::uint64_t count = Be(trp, 16, 4), entry_size = Be(trp, 20, 4);
    if (entry_size < 48 || 0x60 + count * entry_size > trp.size()) {
        return {};
    }
    std::vector<TrpEntry> entries;
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::size_t at = 0x60 + i * entry_size;
        TrpEntry e;
        for (std::size_t c = 0; c < 32 && trp[at + c]; ++c) {
            e.name += char(trp[at + c]);
        }
        e.offset = Be(trp, at + 32, 8);
        e.size = Be(trp, at + 40, 8);
        if (e.offset > trp.size() || e.size > trp.size() - e.offset) {
            return {};
        }
        entries.push_back(std::move(e));
    }
    return entries;
}

std::optional<std::string> NpCommId(std::span<const std::uint8_t> npbind) {
    constexpr std::size_t At = 0x84, Length = 12;
    if (npbind.size() < At + Length) {
        return std::nullopt;
    }
    std::string id(reinterpret_cast<const char*>(npbind.data() + At), Length);
    if (id.rfind("NPWR", 0) != 0) {
        return std::nullopt;
    }
    return id;
}

std::string StripPadding(std::span<const std::uint8_t> plain) {
    std::size_t size = plain.size();
    if (size > 0) {
        const std::uint8_t pad = plain[size - 1];
        if (pad >= 1 && pad <= 16 && pad <= size &&
            std::all_of(plain.end() - pad, plain.end(), [pad](std::uint8_t b) { return b == pad; })) {
            size -= pad;
        }
    }
    return std::string(reinterpret_cast<const char*>(plain.data()), size);
}

std::vector<Trophy> ParseTrophies(std::string_view xml) {
    std::vector<Trophy> trophies;
    std::size_t pos = 0;
    while ((pos = xml.find("<trophy ", pos)) != std::string_view::npos) {
        const auto tag_end = xml.find('>', pos);
        if (tag_end == std::string_view::npos) {
            break;
        }
        const std::string_view tag = xml.substr(pos, tag_end - pos);
        Trophy t;
        t.id = Number(Attribute(tag, "id"), -1);
        const auto grade = Attribute(tag, "ttype");
        t.grade = grade.empty() ? '?' : grade[0];
        t.platinum = Number(Attribute(tag, "pid"), -1);
        t.hidden = Attribute(tag, "hidden") == "yes";
        std::size_t next = tag_end + 1;
        if (tag.back() != '/') { // <trophy ...>...</trophy>: the name and detail inside
            const auto close = xml.find("</trophy>", tag_end);
            const std::string_view body =
                xml.substr(tag_end + 1, (close == std::string_view::npos ? xml.size() : close) - tag_end - 1);
            t.name = Element(body, "name");
            t.detail = Element(body, "detail");
            next = close == std::string_view::npos ? xml.size() : close;
        }
        if (t.id >= 0) {
            trophies.push_back(std::move(t));
        }
        pos = next;
    }
    return trophies;
}

std::vector<int> Unlock(const std::vector<Trophy>& trophies, std::set<int>& unlocked, int id) {
    std::vector<int> announce;
    if (!unlocked.insert(id).second) {
        return announce;
    }
    announce.push_back(id);
    const auto it = std::find_if(trophies.begin(), trophies.end(), [id](const Trophy& t) { return t.id == id; });
    if (it == trophies.end() || it->platinum < 0 || unlocked.contains(it->platinum)) {
        return announce;
    }
    const int platinum = it->platinum;
    const bool complete = std::all_of(trophies.begin(), trophies.end(), [&](const Trophy& t) {
        return t.platinum != platinum || unlocked.contains(t.id);
    });
    if (complete) {
        unlocked.insert(platinum);
        announce.push_back(platinum);
    }
    return announce;
}

std::optional<std::vector<std::uint8_t>> ParseKey(std::string_view hex) {
    while (!hex.empty() && (hex.back() == ' ' || hex.back() == '\r' || hex.back() == '\n' || hex.back() == '"')) {
        hex.remove_suffix(1);
    }
    while (!hex.empty() && (hex.front() == ' ' || hex.front() == '"')) {
        hex.remove_prefix(1);
    }
    if (hex.size() != 32) {
        return std::nullopt;
    }
    std::vector<std::uint8_t> key(16);
    for (std::size_t i = 0; i < 16; ++i) {
        int v = 0;
        for (int k = 0; k < 2; ++k) {
            const char c = hex[i * 2 + k];
            const int d = c >= '0' && c <= '9'   ? c - '0'
                          : c >= 'a' && c <= 'f' ? c - 'a' + 10
                          : c >= 'A' && c <= 'F' ? c - 'A' + 10
                                                 : -1;
            if (d < 0) {
                return std::nullopt;
            }
            v = v * 16 + d;
        }
        key[i] = std::uint8_t(v);
    }
    return key;
}

} // namespace BbTrophy
