// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "gcn/tiling.h"

#include <algorithm>
#include <array>
#include <bit>

#include "common/assert.h"

namespace Gcn {

namespace {

constexpr u32 MicroTileSize = 8; ///< texels per side of a micro tile
constexpr u32 DramRowSize = 1024;

struct TileModeInfo {
    std::string_view name;
    ArrayMode array;
    MicroTileMode micro;
    PipeConfig pipes;
    PipeConfig alt_pipes;
    u8 sample_split;
    u16 tile_split; ///< depth surfaces: bytes
};

/// The PS4's GB_TILE_MODE table.
constexpr std::array<TileModeInfo, NUM_TILE_MODES> TileModes = [] {
    std::array<TileModeInfo, NUM_TILE_MODES> t{};
    using enum ArrayMode;
    using M = MicroTileMode;
    constexpr auto P8 = PipeConfig::P8_32x32_16x16, P8b = PipeConfig::P8_32x32_8x16,
                   P16 = PipeConfig::P16_32x32_8x16;
    const auto set = [&t](TileMode mode, std::string_view name, ArrayMode array, M micro,
                          PipeConfig pipes, u8 sample_split, u16 tile_split = 64) {
        t[u32(mode)] = {name, array, micro, pipes, P16, sample_split, tile_split};
    };
    set(TileMode::Depth2DThin64, "Depth2DThin64", Array2DTiledThin1, M::Depth, P8, 1, 64);
    set(TileMode::Depth2DThin128, "Depth2DThin128", Array2DTiledThin1, M::Depth, P8, 1, 128);
    set(TileMode::Depth2DThin256, "Depth2DThin256", Array2DTiledThin1, M::Depth, P8, 1, 256);
    set(TileMode::Depth2DThin512, "Depth2DThin512", Array2DTiledThin1, M::Depth, P8, 1, 512);
    set(TileMode::Depth2DThin1K, "Depth2DThin1K", Array2DTiledThin1, M::Depth, P8, 1, 1024);
    set(TileMode::Depth1DThin, "Depth1DThin", Array1DTiledThin1, M::Depth, P8, 1);
    set(TileMode::Depth2DThinPrt256, "Depth2DThinPrt256", ArrayPrt2DTiledThin1, M::Depth, P8, 1,
        256);
    set(TileMode::Depth2DThinPrt1K, "Depth2DThinPrt1K", ArrayPrt2DTiledThin1, M::Depth, P8, 1,
        1024);
    set(TileMode::DisplayLinearAligned, "DisplayLinearAligned", ArrayLinearAligned, M::Display,
        P8, 1);
    set(TileMode::Display1DThin, "Display1DThin", Array1DTiledThin1, M::Display, P8, 1);
    set(TileMode::Display2DThin, "Display2DThin", Array2DTiledThin1, M::Display, P8, 2);
    set(TileMode::DisplayThinPrt, "DisplayThinPrt", ArrayPrtTiledThin1, M::Display, P8b, 2);
    set(TileMode::Display2DThinPrt, "Display2DThinPrt", ArrayPrt2DTiledThin1, M::Display, P8, 2);
    set(TileMode::Thin1DThin, "Thin1DThin", Array1DTiledThin1, M::Thin, P8, 1);
    set(TileMode::Thin2DThin, "Thin2DThin", Array2DTiledThin1, M::Thin, P8, 2);
    set(TileMode::Thin3DThin, "Thin3DThin", Array3DTiledThin1, M::Thin, P8b, 2);
    set(TileMode::ThinThinPrt, "ThinThinPrt", ArrayPrtTiledThin1, M::Thin, P8b, 2);
    set(TileMode::Thin2DThinPrt, "Thin2DThinPrt", ArrayPrt2DTiledThin1, M::Thin, P8, 2);
    set(TileMode::Thin3DThinPrt, "Thin3DThinPrt", ArrayPrt3DTiledThin1, M::Thin, P8, 2);
    set(TileMode::Thick1DThick, "Thick1DThick", Array1DTiledThick, M::Thick, P8, 1);
    set(TileMode::Thick2DThick, "Thick2DThick", Array2DTiledThick, M::Thick, P8, 1);
    set(TileMode::Thick3DThick, "Thick3DThick", Array3DTiledThick, M::Thick, P8b, 1);
    set(TileMode::ThickThickPrt, "ThickThickPrt", ArrayPrtTiledThick, M::Thick, P8b, 1);
    set(TileMode::Thick2DThickPrt, "Thick2DThickPrt", ArrayPrt2DTiledThick, M::Thick, P8, 1);
    set(TileMode::Thick3DThickPrt, "Thick3DThickPrt", ArrayPrt3DTiledThick, M::Thick, P8b, 1);
    set(TileMode::Thick2DXThick, "Thick2DXThick", Array2DTiledXThick, M::Thick, P8, 1);
    set(TileMode::Thick3DXThick, "Thick3DXThick", Array3DTiledXThick, M::Thick, P8b, 1);
    set(TileMode::DisplayLinearGeneral, "DisplayLinearGeneral", ArrayLinearGeneral, M::Display,
        PipeConfig::P2, 1);
    t[u32(TileMode::DisplayLinearGeneral)].alt_pipes = PipeConfig::P2;
    return t;
}();

const TileModeInfo& Info(TileMode mode) {
    const u32 index = u32(mode);
    ASSERT_MSG(index < NUM_TILE_MODES && !TileModes[index].name.empty(), "tile mode {}", index);
    return TileModes[index];
}

struct MacroTileModeInfo {
    u8 bank_height;
    u8 banks;
    u8 aspect;
    u8 alt_bank_height;
    u8 alt_banks;
    u8 alt_aspect;
};

/// The PS4's GB_MACROTILE_MODE table (bank width is 1 throughout).
constexpr std::array<MacroTileModeInfo, 16> MacroTileModes = {{
    {4, 16, 4, 4, 8, 2},  // 1x4_16
    {2, 16, 2, 4, 8, 2},  // 1x2_16
    {1, 16, 2, 2, 8, 1},  // 1x1_16
    {1, 16, 2, 2, 8, 1},  // 1x1_16_Dup
    {1, 8, 1, 1, 8, 1},   // 1x1_8
    {1, 4, 1, 1, 8, 1},   // 1x1_4
    {1, 2, 1, 1, 4, 1},   // 1x1_2
    {1, 2, 1, 1, 2, 1},   // 1x1_2_Dup
    {8, 16, 4, 4, 16, 2}, // 1x8_16
    {4, 16, 4, 4, 8, 2},  // 1x4_16_Dup
    {2, 16, 2, 2, 8, 1},  // 1x2_16_Dup
    {1, 16, 2, 1, 8, 1},  // 1x1_16_Dup2
    {1, 8, 1, 1, 4, 1},   // 1x1_8_Dup
    {1, 4, 1, 1, 4, 1},   // 1x1_4_Dup
    {1, 2, 1, 1, 2, 1},   // 1x1_2_Dup2
    {1, 2, 1, 1, 2, 1},   // 1x1_2_Dup3
}};

const MacroTileModeInfo& Info(MacroTileMode mode) {
    ASSERT_MSG(u32(mode) < MacroTileModes.size(), "macro tile mode {}", u32(mode));
    return MacroTileModes[u32(mode)];
}

u32 CheckedArrayMode(ArrayMode mode) {
    ASSERT_MSG(u32(mode) < 16, "array mode {}", u32(mode));
    return u32(mode);
}

/// Array modes as bits.
constexpr u32 Bit(ArrayMode mode) {
    return 1u << u32(mode);
}
constexpr u32 MacroTiledModes = 0xFFF0; ///< all but linear and 1D
constexpr u32 PrtModes = Bit(ArrayMode::ArrayPrtTiledThin1) | Bit(ArrayMode::ArrayPrt2DTiledThin1) |
                         Bit(ArrayMode::ArrayPrtTiledThick) |
                         Bit(ArrayMode::ArrayPrt2DTiledThick) |
                         Bit(ArrayMode::ArrayPrt3DTiledThin1) |
                         Bit(ArrayMode::ArrayPrt3DTiledThick);
constexpr u32 ThickModes = Bit(ArrayMode::Array1DTiledThick) | Bit(ArrayMode::Array2DTiledThick) |
                           Bit(ArrayMode::Array3DTiledThick) | Bit(ArrayMode::ArrayPrtTiledThick) |
                           Bit(ArrayMode::ArrayPrt2DTiledThick) |
                           Bit(ArrayMode::ArrayPrt3DTiledThick);
constexpr u32 XThickModes = Bit(ArrayMode::Array2DTiledXThick) | Bit(ArrayMode::Array3DTiledXThick);

} // namespace

std::string_view NameOf(TileMode tile_mode) {
    const u32 index = u32(tile_mode);
    return index < NUM_TILE_MODES ? TileModes[index].name : std::string_view{};
}

ArrayMode GetArrayMode(TileMode tile_mode) {
    return Info(tile_mode).array;
}
MicroTileMode GetMicroTileMode(TileMode tile_mode) {
    return Info(tile_mode).micro;
}
PipeConfig GetPipeConfig(TileMode tile_mode) {
    return Info(tile_mode).pipes;
}
PipeConfig GetAltPipeConfig(TileMode tile_mode) {
    return Info(tile_mode).alt_pipes;
}
u32 GetSampleSplit(TileMode tile_mode) {
    return Info(tile_mode).sample_split;
}
u32 GetTileSplitHw(TileMode tile_mode) {
    return Info(tile_mode).tile_split;
}

u32 GetBankWidth(MacroTileMode mode) {
    Info(mode);
    return 1;
}
u32 GetBankHeight(MacroTileMode mode) {
    return Info(mode).bank_height;
}
u32 GetNumBanks(MacroTileMode mode) {
    return Info(mode).banks;
}
u32 GetMacrotileAspect(MacroTileMode mode) {
    return Info(mode).aspect;
}
u32 GetAltBankHeight(MacroTileMode mode) {
    return Info(mode).alt_bank_height;
}
u32 GetAltNumBanks(MacroTileMode mode) {
    return Info(mode).alt_banks;
}
u32 GetAltMacrotileAspect(MacroTileMode mode) {
    return Info(mode).alt_aspect;
}

bool IsMacroTiled(ArrayMode array_mode) {
    return (MacroTiledModes >> CheckedArrayMode(array_mode)) & 1;
}
bool IsPrt(ArrayMode array_mode) {
    return (PrtModes >> CheckedArrayMode(array_mode)) & 1;
}
u32 GetMicroTileThickness(ArrayMode array_mode) {
    const u32 bit = 1u << CheckedArrayMode(array_mode);
    return bit & XThickModes ? 8 : bit & ThickModes ? 4 : 1;
}

u32 GetPipeCount(PipeConfig pipe_cfg) {
    const u32 config = u32(pipe_cfg);
    ASSERT_MSG(config <= u32(PipeConfig::P16) && (config == 0 || config >= 4),
               "pipe config {}", config);
    return config == 0 ? 2 : config < 8 ? 4 : config < 16 ? 8 : 16;
}

u32 CalculateTileSplit(TileMode tile_mode, ArrayMode array_mode, MicroTileMode micro_tile_mode,
                       u32 bpp) {
    if (micro_tile_mode == MicroTileMode::Depth) {
        return std::min(DramRowSize, GetTileSplitHw(tile_mode));
    }
    const u32 thickness = GetMicroTileThickness(array_mode);
    const u32 tile_bytes = (bpp * MicroTileSize * MicroTileSize * thickness + 7) / 8;
    return std::min(DramRowSize, std::max(256u, GetSampleSplit(tile_mode) * tile_bytes));
}

MacroTileMode CalculateMacrotileMode(TileMode tile_mode, u32 bpp, u32 num_samples) {
    ASSERT_MSG(std::has_single_bit(num_samples) && num_samples <= 16, "{} samples", num_samples);
    ASSERT_MSG(bpp >= 1 && bpp <= 128, "{} bits per element", bpp);
    const ArrayMode array_mode = GetArrayMode(tile_mode);
    ASSERT_MSG(IsMacroTiled(array_mode), "tile mode {} is not macro tiled", u32(tile_mode));
    const u32 thickness = GetMicroTileThickness(array_mode);
    const u32 tile_bytes_1x = bpp * MicroTileSize * MicroTileSize * thickness / 8;
    const u32 split = CalculateTileSplit(tile_mode, array_mode, GetMicroTileMode(tile_mode), bpp);
    const u32 tile_bytes = std::min(split, num_samples * tile_bytes_1x);
    // 64 bytes: mode 0; each doubling the next (PRT modes from 8).
    const u32 index = std::bit_width(tile_bytes / 64) - 1;
    return MacroTileMode(IsPrt(array_mode) ? index + 8 : index);
}

} // namespace Gcn
