// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the tessellation constants buffer the PS4 driver binds for hull
// and domain shaders (sizes and offsets of the control point and patch regions in LDS).
#pragma once

#include "common/types.h"

namespace Shader {

struct TessellationDataConstantBuffer {
    u32 ls_stride;         ///< input control point stride
    u32 hs_cp_stride;      ///< output control point stride (ls_stride if the hull passes through)
    u32 num_patches;       ///< patches in the thread group
    u32 hs_output_base;    ///< LDS offset of the output control points (0 for passthrough)
    u32 patch_const_size;  ///< 16 * per-patch attributes
    u32 patch_const_base;  ///< hs_output_base + patch_output_size
    u32 patch_output_size; ///< output control point stride * output control points
    f32 off_chip_tessellation_factor_threshold;
    u32 first_edge_tess_factor_index;
};

/// The buffer's dwords by name.
enum class TessConstantAttribute : u32 {
    LsStride,
    HsCpStride,
    HsNumPatch,
    HsOutputBase,
    PatchConstSize,
    PatchConstBase,
    PatchOutputSize,
    OffChipTessellationFactorThreshold,
    FirstEdgeTessFactorIndex,
};

} // namespace Shader
