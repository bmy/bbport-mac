// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: what the host device supports and needs, as the translator and
// backend see it (filled from the Vulkan instance).
#pragma once

#include "common/types.h"

namespace Shader {

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
    /// bbport (macOS), BB_SPIRV_PRECISE=vertex: float adds, multiplies, divides and FMAs are
    /// NoContraction only in the stages that make positions (vertex, tessellation, geometry),
    /// not in fragment and compute shaders. KosmicKrisp compiles every NoContraction operation
    /// in Metal's safe math mode, which keeps the compiler from optimising it.
    bool precise_vertex_stages_only{};
    /// bbport (macOS) experiment, BB_TESS_MAX=n: the hull shaders' tessellation factors are
    /// capped at n (0: as the game writes them). Here so that changing it rebuilds the cache;
    /// the hull shader transform reads the variable itself.
    u32 tess_factor_cap{};
    /// BB_LAYER_MEMORY: guest buffers of IsPagedBuffer size go through the page table.
    bool paged_buffers{};
    /// The game's buffer copy compute shader (its GCN program hash, from the game's profile;
    /// 0 = none): its paged stores go through to the game's memory as well.
    u64 buffer_copy_shader_hash{};

    bool operator==(const Profile&) const = default;
};

} // namespace Shader
