// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: what the host device supports and needs, as the translator and
// backend see it (filled from the Vulkan instance).
#pragma once

#include "common/types.h"

namespace Shader {

/// bbport: the BDA page table covers guest addresses [PageTableBase, PageTableBase + PageTableSpan):
/// the low region (image, stacks) and the game's mappings, which the runtime places from 0x1000000000
/// up (about 5 GiB). All of the 40-bit space took 256 MiB of VRAM (two tables), 13 % of a 2 GB card;
/// this window 16 MiB. Addresses outside it read as unmapped. On macOS: two halves, below.
inline constexpr u64 PageTableBase = 0x800000000ull;
inline constexpr u64 PageTableSpan = u64{1} << 36;
#ifdef __APPLE__
/// bbport: macOS keeps a GPU carve-out at 0x1000000000-0x6FFFFFFFFF (src/platform.h): the game's
/// mappings start at 0x7000000000, above it. The window is two halves of one table: the low
/// region [PageTableBase, +PageTableLowSpan) (image, stacks) first, then the mappings
/// [PageTableUserBase, +PageTableSpan - PageTableLowSpan). Same table size as on Linux.
inline constexpr u64 PageTableLowSpan = u64{1} << 35;
inline constexpr u64 PageTableUserBase = 0x7000000000ull;
#endif

/// The offset in the table's window of a guest address: PageTableSpan or more outside it. The
/// shaders compute the same (EmitContext::PageTableOffset).
constexpr u64 PageTableOffset(u64 address) {
#ifdef __APPLE__
    if (address >= PageTableUserBase) {
        const u64 offset = address - PageTableUserBase;
        return offset < PageTableSpan - PageTableLowSpan ? offset + PageTableLowSpan : PageTableSpan;
    }
    const u64 offset = address - PageTableBase;
    return offset < PageTableLowSpan ? offset : PageTableSpan;
#else
    return address - PageTableBase;
#endif
}

/// Whether [address, address + size) lies in one piece of the window (size > 0).
constexpr bool InPageTableWindow(u64 address, u64 size) {
    const u64 first = PageTableOffset(address);
    return size != 0 && first < PageTableSpan && size <= PageTableSpan - first &&
           PageTableOffset(address + size - 1) == first + size - 1;
}

struct Profile {
    u32 max_viewport_width{};
    u32 max_viewport_height{};
    u32 max_shared_memory_size{};
    u32 supported_spirv{0x00010000};
    u32 subgroup_size{};
    u32 sparse_page_shift{};
    // Types.
    bool support_int8{};
    bool support_int16{};
    bool support_int64{};
    bool support_float16{};
    bool support_float64{};
    // Float controls.
    bool supports_denorm_behavior_independence{};
    bool supports_rounding_mode_independence{};
    bool support_fp16_denorm_preserve{};
    bool support_fp16_denorm_flush{};
    bool support_fp16_round_to_zero{};
    bool support_fp32_denorm_preserve{};
    bool support_fp32_denorm_flush{};
    bool support_fp32_round_to_zero{};
    bool support_fp64_denorm_preserve{};
    bool support_fp64_denorm_flush{};
    bool support_fp64_round_to_zero{};
    bool support_fp16_signed_zero_inf_nan_preserve{};
    bool support_fp32_signed_zero_inf_nan_preserve{};
    bool support_fp64_signed_zero_inf_nan_preserve{};
    // Instructions and features.
    bool supports_image_load_store_lod{};
    bool supports_native_cube_calc{};
    bool supports_trinary_minmax{};
    bool supports_buffer_fp32_atomic_min_max{};
    bool supports_image_fp32_atomic_min_max{};
    bool supports_buffer_int64_atomics{};
    bool supports_shared_int64_atomics{};
    bool supports_workgroup_explicit_memory_layout{};
    bool supports_amd_shader_explicit_vertex_parameter{};
    bool supports_fragment_shader_barycentric{};
    bool supports_shader_subgroup_clock{};
    // Workarounds.
    bool has_broken_spirv_clamp{};
    bool lower_left_origin_mode{};
    bool needs_manual_interpolation{};
    /// NVIDIA without fragment shader barycentrics (Pascal and older): integer inputs are
    /// interpolated, truncate them.
    bool needs_integer_interpolation_fix{};
    bool needs_lds_barriers{};
    bool needs_buffer_offsets{};
    bool needs_unorm_fixup{};
    bool needs_clip_distance_emulation{};
    bool supports_shader_stencil_export{};
    /// BB_LAYER_MEMORY: guest buffers of IsPagedBuffer size go through the page table.
    bool paged_buffers{};
    /// The game's buffer copy compute shader (its GCN program hash, from the game's profile;
    /// 0 = none): its paged stores go through to the game's memory as well.
    u64 buffer_copy_shader_hash{};

    bool operator==(const Profile&) const = default;
};

} // namespace Shader
