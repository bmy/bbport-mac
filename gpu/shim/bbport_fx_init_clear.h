// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "cp/cp_pm4.h"

namespace BbFxInit {

struct ZeroClear {
    std::uint64_t address;
    std::uint32_t bytes;
};

// The three 2x2 RG16F history targets initialized by Bloodborne's 0xebb960.
// A uniform zero has the same representation in every tile; the requested alpha
// is not stored by this two-component target. Reject other clears and layouts.
inline std::optional<ZeroClear> Plan(std::span<const std::uint32_t> target,
                                    std::span<const float> color) {
    if (target.size() < 11 || color.size() < 4 || color[0] != 0 || color[1] != 0 ||
        color[2] != 0 || color[3] != 1 || target[0] == 0 || target[3] != 0 ||
        target[6] != 0x00020002 || target[7] || target[8] || target[9] || target[10]) {
        return std::nullopt;
    }
    const auto info = target[4], attrib = target[5];
    // RG16, FLOAT, standard component mapping, no compression/fast clear/Neo mode.
    if (((info >> 2) & 31) != 5 || ((info >> 8) & 7) != 7 ||
        (info & (3u | (1u << 7) | (3u << 11) | (3u << 13) | (1u << 28) | (1u << 31))) ||
        (attrib & 31) != 14 || (attrib & (31u << 12))) {
        return std::nullopt;
    }
    const std::uint64_t pitch = ((target[1] & 0x7ff) + 1) * 8;
    const std::uint64_t texels = ((target[2] & 0x3fffff) + 1) * 64;
    const std::uint64_t bytes = texels * 4;
    const std::uint64_t address = std::uint64_t(target[0]) << 8;
    if (pitch < 2 || texels / pitch < 2 || texels % pitch || bytes > 32768 ||
        address + bytes > (1ull << 40)) {
        return std::nullopt;
    }
    return ZeroClear{address, std::uint32_t(bytes)};
}

inline std::vector<std::uint32_t> Commands(ZeroClear clear) {
    const std::uint32_t words = clear.bytes / 4;
    std::vector<std::uint32_t> commands(words + 4, 0);
    commands[0] = (3u << 30) | ((words + 2) << 16) | (std::uint32_t(BbCp::Op::WriteData) << 8);
    commands[1] = 5u << 8; // consecutive memory destination, graphics engine
    commands[2] = std::uint32_t(clear.address);
    commands[3] = std::uint32_t(clear.address >> 32);
    return commands;
}

} // namespace BbFxInit
