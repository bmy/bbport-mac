// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport gcn: the render target registers: color buffers and blending (CB_*), depth and
// stencil (DB_*).
#pragma once

#include <array>

#include "common/assert.h"
#include "common/types.h"
#include "gcn/formats.h"
#include "gcn/tiling.h"

namespace Gcn {

inline constexpr u32 NUM_COLOR_BUFFERS = 8;

/// CB_BLEND_RED..ALPHA.
using BlendConstants = std::array<float, 4>;

/// CB_BLEND0_CONTROL..7.
struct BlendControl {
    enum class BlendFactor : u32 {
        Zero = 0,
        One = 1,
        SrcColor = 2,
        OneMinusSrcColor = 3,
        SrcAlpha = 4,
        OneMinusSrcAlpha = 5,
        DstAlpha = 6,
        OneMinusDstAlpha = 7,
        DstColor = 8,
        OneMinusDstColor = 9,
        SrcAlphaSaturate = 10,
        ConstantColor = 13,
        OneMinusConstantColor = 14,
        Src1Color = 15,
        InvSrc1Color = 16,
        Src1Alpha = 17,
        InvSrc1Alpha = 18,
        ConstantAlpha = 19,
        OneMinusConstantAlpha = 20,
    };
    enum class BlendFunc : u32 {
        Add = 0,
        Subtract = 1,
        Min = 2,
        Max = 3,
        ReverseSubtract = 4,
    };

    BlendFactor color_src_factor : 5;
    BlendFunc color_func : 3;
    BlendFactor color_dst_factor : 5;
    u32 : 3;
    BlendFactor alpha_src_factor : 5;
    BlendFunc alpha_func : 3;
    BlendFactor alpha_dst_factor : 5;
    u32 separate_alpha_blend : 1;
    u32 enable : 1;
    u32 disable_rop3 : 1;

    bool operator==(const BlendControl& other) const = default;
};

/// CB_COLOR_CONTROL.
struct ColorControl {
    enum class OperationMode : u32 {
        Disable = 0,
        Normal = 1,
        EliminateFastClear = 2,
        Resolve = 3,
        Err = 4,
        FmaskDecompress = 5,
    };
    /// ROP3 codes of the logic operations.
    enum class LogicOp : u32 {
        Clear = 0x00,
        Nor = 0x11,
        AndInverted = 0x22,
        CopyInverted = 0x33,
        AndReverse = 0x44,
        Invert = 0x55,
        BrushXor = 0x5a,
        Xor = 0x66,
        Nand = 0x77,
        And = 0x88,
        Equiv = 0x99,
        Noop = 0xAA,
        OrInverted = 0xBB,
        Copy = 0xCC,
        OrReverse = 0xDD,
        Or = 0xEE,
        Set = 0xFF,
    };

    u32 disable_dual_quad : 1;
    u32 : 2;
    u32 degamma_enable : 1;
    OperationMode mode : 3;
    u32 : 9;
    LogicOp rop3 : 8;
};

/// CB_TARGET_MASK / CB_SHADER_MASK: four component bits per target.
struct ColorBufferMask {
    enum ColorComponent : u32 {
        ComponentR = 1u << 0,
        ComponentG = 1u << 1,
        ComponentB = 1u << 2,
        ComponentA = 1u << 3,
    };

    u32 raw;

    u32 GetMask(u32 buf_id) const {
        return (raw >> (buf_id * 4)) & 0xf;
    }
    void SetMask(u32 buf_id, u32 mask) {
        raw = (raw & ~(0xfu << (buf_id * 4))) | (mask << (buf_id * 4));
    }
};

/// CB_COLOR0_BASE..CB_COLOR0_CLEAR_WORD1 (+ two unused): one color target, 15 registers apart.
struct ColorBuffer {
    enum class EndianSwap : u32 {
        None = 0,
        Swap8In16 = 1,
        Swap8In32 = 2,
        Swap8In64 = 3,
    };
    enum class SwapMode : u32 {
        Standard = 0,
        Alternate = 1,
        StandardReverse = 2,
        AlternateReverse = 3,
    };
    enum class RoundMode : u32 {
        ByHalf = 0,
        Truncate = 1,
    };

    u32 base_address; ///< 256-byte units
    struct {
        u32 tile_max : 11; ///< pitch / 8 - 1
        u32 : 9;
        u32 fmask_tile_max : 11;
    } pitch;
    struct {
        u32 tile_max : 22; ///< pitch * height / 64 - 1
    } slice;
    struct {
        u32 slice_start : 11;
        u32 : 2;
        u32 slice_max : 11;
    } view;
    union Color0Info {
        u32 raw;
        struct {
            EndianSwap endian : 2;
            u32 format : 5;
            u32 linear_general : 1;
            u32 number_type : 3;
            SwapMode comp_swap : 2;
            u32 fast_clear : 1;
            u32 compression : 1;
            u32 blend_clamp : 1;
            u32 blend_bypass : 1;
            u32 simple_float : 1;
            RoundMode round_mode : 1;
            u32 cmask_is_linear : 1;
            u32 blend_opt_dont_rd_dst : 3;
            u32 blend_opt_discard_pixel : 3;
            u32 fmask_compression_disable_ci : 1;
            u32 fmask_compress_1frag_only : 1;
            u32 dcc_enable : 1;
            u32 cmask_addr_type : 2;
            u32 alt_tile_mode : 1;
        };
    } info;
    union Color0Attrib {
        u32 raw;
        struct {
            TileMode tile_mode_index : 5;
            u32 fmask_tile_mode_index : 5;
            u32 fmask_bank_height : 2;
            u32 num_samples_log2 : 3;
            u32 num_fragments_log2 : 2;
            u32 force_dst_alpha_1 : 1;
        };
    } attrib;
    u32 pad0;
    u32 cmask_base_address;
    struct {
        u32 tile_max : 14;
    } cmask_slice;
    u32 fmask_base_address;
    struct {
        u32 tile_max : 14;
    } fmask_slice;
    u32 clear_word0;
    u32 clear_word1;
    std::array<u32, 2> pad1;

    /// Bound: an address and a format.
    explicit operator bool() const {
        return base_address && info.format;
    }

    u32 Pitch() const {
        return (pitch.tile_max + 1) * 8;
    }
    u32 Height() const {
        return (slice.tile_max + 1) * 64 / Pitch();
    }
    /// Linear general surfaces keep the low address byte in the view's first slice.
    u64 Address() const {
        return u64(base_address) << 8 | (info.linear_general ? (view.slice_start & 0xff) : 0);
    }
    VAddr CmaskAddress() const {
        return VAddr(cmask_base_address) << 8;
    }
    VAddr FmaskAddress() const {
        return VAddr(fmask_base_address) << 8;
    }
    u32 NumSamples() const {
        return 1u << attrib.num_fragments_log2;
    }
    u32 BaseSlice() const {
        return info.linear_general ? 0 : view.slice_start;
    }
    u32 NumSlices() const {
        return view.slice_max + 1;
    }
    u32 GetColorSliceSize() const {
        const u32 bytes_per_element = NumBitsPerBlock(DataFormat(info.format)) / 8;
        return bytes_per_element * (slice.tile_max + 1) * 64 * NumSamples();
    }

    TileMode GetTileMode() const {
        return info.linear_general ? TileMode::DisplayLinearGeneral : attrib.tile_mode_index;
    }
    bool IsTiled() const {
        const TileMode mode = GetTileMode();
        return mode != TileMode::DisplayLinearAligned && mode != TileMode::DisplayLinearGeneral;
    }

    DataFormat GetDataFmt() const {
        return RemapDataFormat(DataFormat(info.format));
    }
    /// Color buffers say SnormNz where textures say Srgb.
    NumberFormat GetFixedNumberFormat() const {
        const auto number_fmt = NumberFormat(info.number_type);
        return number_fmt == NumberFormat::SnormNz ? NumberFormat::Srgb : number_fmt;
    }
    NumberFormat GetNumberFmt() const {
        return RemapNumberFormat(GetFixedNumberFormat(), DataFormat(info.format));
    }
    NumberConversion GetNumberConversion() const {
        return MapNumberConversion(GetFixedNumberFormat(), DataFormat(info.format));
    }

    /// The components the target stores, by COMP_SWAP and component count.
    CompMapping Swizzle() const {
        using enum CompSwizzle;
        // [swap][components - 1]: which shader output component lands in r, g, b, a.
        static constexpr CompSwizzle Table[4][4][4] = {
            // Standard
            {{Red, Zero, Zero, Zero},
             {Red, Green, Zero, Zero},
             {Red, Green, Blue, Zero},
             {Red, Green, Blue, Alpha}},
            // Alternate
            {{Green, Zero, Zero, Zero},
             {Red, Alpha, Zero, Zero},
             {Red, Green, Alpha, Zero},
             {Blue, Green, Red, Alpha}},
            // StandardReverse
            {{Blue, Zero, Zero, Zero},
             {Green, Red, Zero, Zero},
             {Blue, Green, Red, Zero},
             {Alpha, Blue, Green, Red}},
            // AlternateReverse
            {{Alpha, Zero, Zero, Zero},
             {Alpha, Red, Zero, Zero},
             {Alpha, Green, Red, Zero},
             {Alpha, Red, Green, Blue}},
        };
        const auto format = DataFormat(info.format);
        const u32 components = NumComponents(format);
        if (components == 0) {
            return {.r = Zero, .g = Zero, .b = Zero, .a = Zero};
        }
        const auto& sel = Table[u32(info.comp_swap)][components - 1];
        return RemapSwizzle(format, {.r = sel[0], .g = sel[1], .b = sel[2], .a = sel[3]});
    }
};

enum class ZOrder : u32 {
    LateZ = 0,
    EarlyZLateZ = 1,
    ReZ = 2,
    EarlyZReZ = 3,
};

enum class ConservativeDepth : u32 {
    Any = 0,
    LessThanZ = 1,
    GreaterThanZ = 2,
};

/// DB_SHADER_CONTROL.
struct DepthShaderControl {
    u32 z_export_enable : 1;
    u32 stencil_test_val_export_enable : 1;
    u32 stencil_op_val_export_enable : 1;
    u32 : 1;
    ZOrder z_order : 2;
    u32 kill_enable : 1;
    u32 coverage_to_mask_enable : 1;
    u32 mask_export_enable : 1;
    u32 exec_on_hier_fail : 1;
    u32 exec_on_noop : 1;
    u32 alpha_to_mask_disable : 1;
    u32 depth_before_shader : 1;
    ConservativeDepth conservative_z_export : 2;
};

enum class CompareFunc : u32 {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
};

/// DB_DEPTH_CONTROL.
struct DepthControl {
    u32 stencil_enable : 1;
    u32 depth_enable : 1;
    u32 depth_write_enable : 1;
    u32 depth_bounds_enable : 1;
    CompareFunc depth_func : 3;
    u32 backface_enable : 1;
    CompareFunc stencil_ref_func : 3;
    u32 : 9;
    CompareFunc stencil_bf_func : 3;
    u32 : 7;
    u32 enable_color_writes_on_depth_fail : 1;
    u32 disable_color_writes_on_depth_pass : 1;
};

enum class StencilFunc : u32 {
    Keep = 0,
    Zero = 1,
    Ones = 2,
    ReplaceTest = 3,
    ReplaceOp = 4,
    AddClamp = 5,
    SubClamp = 6,
    Invert = 7,
    AddWrap = 8,
    SubWrap = 9,
    And = 10,
    Or = 11,
    Xor = 12,
    Nand = 13,
    Nor = 14,
    Xnor = 15,
};

/// DB_STENCIL_CONTROL.
struct StencilControl {
    StencilFunc stencil_fail_front : 4;
    StencilFunc stencil_zpass_front : 4;
    StencilFunc stencil_zfail_front : 4;
    StencilFunc stencil_fail_back : 4;
    StencilFunc stencil_zpass_back : 4;
    StencilFunc stencil_zfail_back : 4;
};

/// DB_STENCILREFMASK / _BF.
struct StencilRefMask {
    u8 stencil_test_val;
    u8 stencil_mask;
    u8 stencil_write_mask;
    u8 stencil_op_val;
};

/// DB_RENDER_CONTROL.
struct DepthRenderControl {
    u32 depth_clear_enable : 1;
    u32 stencil_clear_enable : 1;
    u32 depth_copy : 1;
    u32 stencil_copy : 1;
    u32 resummarize_enable : 1;
    u32 stencil_compress_disable : 1;
    u32 depth_compress_disable : 1;
    u32 copy_centroid : 1;
    u32 copy_sample : 1;
    u32 decompress_enable : 1;
};

/// DB_DEPTH_VIEW.
struct DepthView {
    u32 slice_start : 11;
    u32 : 2;
    u32 slice_max : 11;
    u32 z_read_only : 1;
    u32 stencil_read_only : 1;

    u32 NumSlices() const {
        return slice_max + 1;
    }
};

enum class ForceEnable : u32 {
    Off = 0,
    Enable = 1,
    Disable = 2,
};

enum class ForceSumm : u32 {
    Off = 0,
    MinZ = 1,
    MaxZ = 2,
    Both = 3,
};

/// DB_RENDER_OVERRIDE.
struct DepthRenderOverride {
    ForceEnable force_hiz_enable : 2;
    ForceEnable force_his_enable0 : 2;
    ForceEnable force_his_enable1 : 2;
    u32 force_shader_z_order : 1;
    u32 fast_z_disable : 1;
    u32 fast_stencil_disable : 1;
    u32 noop_cull_disable : 1;
    u32 force_color_kill : 1;
    u32 force_z_read : 1;
    u32 force_stencil_read : 1;
    ForceEnable force_full_z_range : 2;
    u32 force_qc_smask_conflict : 1;
    u32 disable_viewport_clamp : 1;
    u32 ignore_sc_zrange : 1;
    u32 disable_fully_covered : 1;
    ForceSumm force_z_limit_summ : 2;
    u32 max_tiles_in_dtt : 5;
    u32 disable_tile_rate_tiles : 1;
    u32 force_z_dirty : 1;
    u32 force_stencil_dirty : 1;
    u32 force_z_valid : 1;
    u32 force_stencil_valid : 1;
    u32 preserve_compression : 1;
};

/// DB_EQAA.
struct Eqaa {
    u32 max_anchor_samples : 1;
    u32 : 3;
    u32 ps_iter_samples : 3;
    u32 : 1;
    u32 mask_export_num_samples : 3;
    u32 : 1;
    u32 alpha_to_mask_num_samples : 3;
    u32 : 1;
    u32 high_quality_intersections : 1;
    u32 incoherent_eqaa_reads : 1;
    u32 interpolate_comp_z : 1;
    u32 interpolate_src_z : 1;
    u32 static_anchor_associations : 1;
    u32 alpha_to_mask_eqaa_disable : 1;
    u32 : 2;
    u32 overrasterization_amount : 3;
    u32 enable_postz_overrasterization : 1;
};

/// DB_Z_INFO..DB_DEPTH_SLICE: the depth and stencil target.
struct DepthBuffer {
    enum class ZFormat : u32 {
        Invalid = 0,
        Z16 = 1,
        Z32Float = 3,
    };
    enum class StencilFormat : u32 {
        Invalid = 0,
        Stencil8 = 1,
    };

    struct ZInfo {
        ZFormat format : 2;
        u32 num_samples : 2; ///< log2
        u32 : 9;
        u32 tile_split : 3;
        u32 : 4;
        u32 tile_mode_index : 3;
        u32 decompress_on_n_zplanes : 4;
        u32 allow_expclear : 1;
        u32 read_size : 1;
        u32 tile_surface_enable : 1;
        u32 clear_disallowed : 1;
        u32 zrange_precision : 1;
    } z_info;
    struct {
        StencilFormat format : 1;
    } stencil_info;
    u32 z_read_base; ///< 256-byte units, as the bases below
    u32 stencil_read_base;
    u32 z_write_base;
    u32 stencil_write_base;
    struct {
        u32 pitch_tile_max : 11;
        u32 height_tile_max : 11;
    } depth_size;
    struct {
        u32 tile_max : 22;
    } depth_slice;

    u64 DepthAddress() const {
        return u64(z_read_base) << 8;
    }
    u64 StencilAddress() const {
        return u64(stencil_read_base) << 8;
    }
    u64 DepthWriteAddress() const {
        return u64(z_write_base) << 8;
    }
    u64 StencilWriteAddress() const {
        return u64(stencil_write_base) << 8;
    }

    bool DepthValid() const {
        return DepthAddress() != 0 && z_info.format != ZFormat::Invalid;
    }
    bool StencilValid() const {
        return StencilAddress() != 0 && stencil_info.format != StencilFormat::Invalid;
    }
    bool DepthWriteValid() const {
        return DepthWriteAddress() != 0 && z_info.format != ZFormat::Invalid;
    }
    bool StencilWriteValid() const {
        return StencilWriteAddress() != 0 && stencil_info.format != StencilFormat::Invalid;
    }

    u32 Pitch() const {
        return (depth_size.pitch_tile_max + 1) * 8;
    }
    u32 Height() const {
        return (depth_size.height_tile_max + 1) * 8;
    }
    u32 NumSamples() const {
        return 1u << z_info.num_samples;
    }
    u32 NumBits() const {
        return z_info.format == ZFormat::Z32Float ? 32 : 16;
    }
    u32 GetDepthSliceSize() const {
        ASSERT(z_info.format != ZFormat::Invalid);
        return (depth_slice.tile_max + 1) * 64 * (NumBits() / 8) * NumSamples();
    }

    TileMode GetTileMode() const {
        return TileMode(z_info.tile_mode_index);
    }
    bool IsTiled() const {
        const TileMode mode = GetTileMode();
        return mode != TileMode::DisplayLinearAligned && mode != TileMode::DisplayLinearGeneral;
    }
};

/// TA_BC_BASE: the border colors of samplers that use custom ones.
struct BorderColorBuffer {
    u64 base_addr : 40; ///< 256-byte units

    template <typename T = VAddr>
    T Address() const {
        return std::bit_cast<T>(u64(base_addr) << 8);
    }
};

/// A render target's size the driver gives in a NOP hint after CB/DB setup.
union CbDbExtent {
    struct {
        u16 width;
        u16 height;
    };
    u32 raw;

    bool Valid() const {
        return raw != 0;
    }
};

} // namespace Gcn
