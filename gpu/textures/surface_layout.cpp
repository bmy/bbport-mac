// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "textures/surface_layout.h"

#include <array>
#include <bit>

#include "common/assert.h"

namespace VideoCore {

namespace {

constexpr u32 MicroTileSize = 8;

u32 AlignUp(u32 value, u32 alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

/// [alt][tile mode][log2 bpp - 3][log2 samples]; zero for modes that are not macro tiled.
using ExtentTable = std::array<std::array<std::array<std::array<MacroTileExtent, 4>, 5>, 32>, 2>;

ExtentTable BuildExtents() {
    ExtentTable table{};
    for (u32 alt = 0; alt < 2; ++alt) {
        for (u32 mode = 0; mode < Gcn::NUM_TILE_MODES; ++mode) {
            const auto tile_mode = Gcn::TileMode(mode);
            if (Gcn::NameOf(tile_mode).empty() ||
                !Gcn::IsMacroTiled(Gcn::GetArrayMode(tile_mode))) {
                continue;
            }
            const u32 pipes = Gcn::GetPipeCount(alt ? Gcn::GetAltPipeConfig(tile_mode)
                                                    : Gcn::GetPipeConfig(tile_mode));
            for (u32 b = 0; b < 5; ++b) {
                for (u32 s = 0; s < 4; ++s) {
                    const auto macro = Gcn::CalculateMacrotileMode(tile_mode, 8u << b, 1u << s);
                    const u32 aspect = alt ? Gcn::GetAltMacrotileAspect(macro)
                                           : Gcn::GetMacrotileAspect(macro);
                    const u32 banks = alt ? Gcn::GetAltNumBanks(macro) : Gcn::GetNumBanks(macro);
                    const u32 bank_height =
                        alt ? Gcn::GetAltBankHeight(macro) : Gcn::GetBankHeight(macro);
                    table[alt][mode][b][s] = {
                        pipes * Gcn::GetBankWidth(macro) * MicroTileSize * aspect,
                        banks * bank_height * MicroTileSize / aspect,
                    };
                }
            }
        }
    }
    return table;
}

} // namespace

MacroTileExtent GetMacroTileExtent(Gcn::TileMode tile_mode, u32 bpp, u32 num_samples, bool alt) {
    static const ExtentTable extents = BuildExtents();
    ASSERT_MSG(num_samples >= 1 && num_samples <= 8, "{} samples", num_samples);
    ASSERT_MSG(bpp >= 8 && bpp <= 128, "{} bits per texel", bpp);
    const u32 b = std::bit_width(bpp) - 4; // 8 -> 0 ... 128 -> 4; 96 counts as 64
    const u32 s = std::bit_width(num_samples) - 1;
    return extents[alt][u32(tile_mode) % Gcn::NUM_TILE_MODES][b][s];
}

LevelSize LinearAlignedSize(u32 pitch, u32 height, u32 bpp, u32 num_samples) {
    const u32 bytes_per_texel = (bpp + 7) / 8;
    const u32 pitch_align = std::max(8u, 64u / bytes_per_texel);
    const u32 slice_align = std::max(64u, 256u / bytes_per_texel);
    u32 padded = AlignUp(pitch, pitch_align);
    while (u64(padded) * height * num_samples % slice_align != 0) {
        padded += pitch_align;
    }
    const u64 texels = u64(padded) * height * num_samples;
    return {padded, height, (texels * bpp + 7) / 8};
}

LevelSize MicroTiledSize(u32 pitch, u32 height, u32 thickness, u32 bpp, u32 num_samples) {
    u32 padded = AlignUp(pitch, MicroTileSize);
    const u32 rows = AlignUp(height, MicroTileSize);
    const auto bytes = [&] { return (u64(padded) * rows * bpp * num_samples + 7) / 8; };
    while (bytes() * thickness % 256 != 0) {
        padded += MicroTileSize;
    }
    return {padded, rows, bytes()};
}

LevelSize MacroTiledSize(u32 pitch, u32 height, u32 bpp, u32 num_samples, Gcn::TileMode tile_mode,
                         bool alt) {
    const auto tile = GetMacroTileExtent(tile_mode, bpp, num_samples, alt);
    ASSERT_MSG(tile.width != 0 && tile.height != 0, "tile mode {} is not macro tiled",
               u32(tile_mode));
    const u32 padded = AlignUp(pitch, tile.width);
    const u32 rows = AlignUp(height, tile.height);
    return {padded, rows, (u64(padded) * rows * num_samples * bpp + 7) / 8};
}

bool IsMacroTiledMip(u32 pitch, u32 height, u32 bpp, u32 num_samples, Gcn::TileMode tile_mode,
                     u32 level, bool alt) {
    if (level == 0) {
        return true;
    }
    const auto tile = GetMacroTileExtent(tile_mode, bpp, num_samples, alt);
    return pitch >= tile.width && height >= tile.height;
}

} // namespace VideoCore
