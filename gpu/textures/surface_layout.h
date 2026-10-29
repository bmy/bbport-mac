// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: how a surface level lies in memory: its padded pitch and height and its size,
// linear (aligned), micro tiled (8x8 texel tiles) or macro tiled (tiles spread over the GPU's
// pipes and banks). Sizes in bytes; pitch and height in texels (blocks for compressed formats).
#pragma once

#include "common/types.h"
#include "gcn/tiling.h"

namespace VideoCore {

struct LevelSize {
    u32 pitch;
    u32 height;
    u64 bytes;
};

/// Texels of one macro tile of a surface in `tile_mode`, `bpp` bits per texel (rounded down to a
/// power of two), `num_samples` up to 8; `alt`: the PS4 Pro's layout.
struct MacroTileExtent {
    u32 width;
    u32 height;
};
MacroTileExtent GetMacroTileExtent(Gcn::TileMode tile_mode, u32 bpp, u32 num_samples, bool alt);

/// Linear: rows padded to 64 bytes (at least 8 texels), the slice to 256 bytes (at least 64
/// texels), by widening the rows.
LevelSize LinearAlignedSize(u32 pitch, u32 height, u32 bpp, u32 num_samples);

/// Micro tiled: whole 8x8 tiles; `thickness` slices of a tile together a multiple of 256 bytes,
/// by widening the rows.
LevelSize MicroTiledSize(u32 pitch, u32 height, u32 thickness, u32 bpp, u32 num_samples);

/// Macro tiled: whole macro tiles.
LevelSize MacroTiledSize(u32 pitch, u32 height, u32 bpp, u32 num_samples, Gcn::TileMode tile_mode,
                         bool alt);

/// Whether mip `level` of a macro tiled surface stays macro tiled: the base level always, smaller
/// ones while they span a whole macro tile (else the hardware stores them micro tiled).
bool IsMacroTiledMip(u32 pitch, u32 height, u32 bpp, u32 num_samples, Gcn::TileMode tile_mode,
                     u32 level, bool alt);

} // namespace VideoCore
