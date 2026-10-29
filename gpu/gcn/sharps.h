// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport gcn: the resource descriptors shaders read from their user data (Sea Islands ISA,
// chapter 8): buffers (V#, 128 bits), images (T#, 256 bits) and samplers (S#, 128 bits).
#pragma once

#include <algorithm>
#include <bit>
#include <cstring>
#include <string_view>

#include <fmt/format.h>

#include "common/alignment.h"
#include "common/assert.h"
#include "gcn/formats.h"
#include "gcn/tiling.h"

namespace Gcn {

/// V#: a buffer.
struct Buffer {
    u64 base_address : 40;
    u64 : 8;
    u64 stride : 14;
    u64 cache_swizzle : 1;
    u64 swizzle_enable : 1;
    u32 num_records;
    u32 dst_sel_x : 3;
    u32 dst_sel_y : 3;
    u32 dst_sel_z : 3;
    u32 dst_sel_w : 3;
    u32 num_format : 3;
    u32 data_format : 4;
    u32 element_size : 2;
    u32 index_stride : 2;
    u32 add_tid_enable : 1;
    u32 : 3;
    u32 mtype : 3;
    u32 type : 2; ///< 0: a buffer (the bits of a T#'s type)

    /// A descriptor of nothing (address 1: not a null pointer).
    static constexpr Buffer Null() {
        return Placeholder(0);
    }
    static constexpr Buffer Placeholder(u32 size) {
        Buffer buffer{};
        buffer.base_address = 1;
        buffer.num_records = size;
        return buffer;
    }

    bool Valid() const {
        return type == 0;
    }
    explicit operator bool() const noexcept {
        return num_records != 0;
    }
    bool operator==(const Buffer& other) const noexcept {
        return std::memcmp(this, &other, sizeof(Buffer)) == 0;
    }

    CompMapping DstSelect() const {
        return RemapSwizzle(DataFormat(data_format), {.r = CompSwizzle(dst_sel_x),
                                                      .g = CompSwizzle(dst_sel_y),
                                                      .b = CompSwizzle(dst_sel_z),
                                                      .a = CompSwizzle(dst_sel_w)});
    }
    NumberFormat GetNumberFmt() const noexcept {
        return RemapNumberFormat(NumberFormat(num_format), DataFormat(data_format));
    }
    DataFormat GetDataFmt() const noexcept {
        return RemapDataFormat(DataFormat(data_format));
    }
    NumberConversion GetNumberConversion() const noexcept {
        return MapNumberConversion(NumberFormat(num_format), DataFormat(data_format));
    }
    u32 GetStride() const noexcept {
        return stride;
    }
    /// Bytes: records of `stride` bytes, or bytes when the stride is zero.
    u32 GetSize() const noexcept {
        return stride == 0 ? num_records : stride * num_records;
    }
    u32 NumDwords() const noexcept {
        return Common::AlignUp(GetSize(), u32(sizeof(u32))) / sizeof(u32);
    }
    /// Swizzled buffers: 8, 16, 32 or 64 records per index.
    u32 GetIndexStride() const noexcept {
        return 8u << index_stride;
    }
    /// Swizzled buffers: 2, 4, 8 or 16 bytes per element.
    u32 GetElementSize() const noexcept {
        return 2u << element_size;
    }
};
static_assert(sizeof(Buffer) == 16);

enum class ImageType : u64 {
    Invalid = 0,
    Color1D = 8,
    Color2D = 9,
    Color3D = 10,
    Cube = 11,
    Color1DArray = 12,
    Color2DArray = 13,
    Color2DMsaa = 14,
    Color2DMsaaArray = 15,
};

constexpr std::string_view NameOf(ImageType type) {
    constexpr std::string_view names[] = {"Color1D",      "Color2D",      "Color3D",
                                          "Cube",         "Color1DArray", "Color2DArray",
                                          "Color2DMsaa",  "Color2DMsaaArray"};
    if (type == ImageType::Invalid) {
        return "Invalid";
    }
    const u64 index = u64(type) - u64(ImageType::Color1D);
    return index < std::size(names) ? names[index] : "Unknown";
}

/// T#: an image.
struct Image {
    u64 base_address : 38;
    u64 mtype_l2 : 2;
    u64 min_lod : 12;
    u64 data_format : 6;
    u64 num_format : 4;
    u64 mtype : 2;

    u64 width : 14;
    u64 height : 14;
    u64 perf_modulation : 3;
    u64 interlaced : 1;
    u64 dst_sel_x : 3;
    u64 dst_sel_y : 3;
    u64 dst_sel_z : 3;
    u64 dst_sel_w : 3;
    u64 base_level : 4;
    u64 last_level : 4;
    u64 tiling_index : 5;
    u64 pow2pad : 1;
    u64 mtype2 : 1;
    u64 atc : 1;
    u64 type : 4; ///< ImageType; a V# has zero here

    u64 depth : 13;
    u64 pitch : 14;
    u64 : 5;
    u64 base_array : 13;
    u64 last_array : 13;
    u64 : 6;

    u64 min_lod_warn : 12;
    u64 counter_bank_id : 8;
    u64 lod_hw_cnt_en : 1;
    // PS4 Pro only.
    u64 compression_en : 1;
    u64 alpha_is_on_msb : 1;
    u64 color_transform : 1;
    u64 alt_tile_mode : 1;
    u64 : 39;

    /// A 1x1 image of nothing, for unbound slots (depth: a 32-bit float format).
    static constexpr Image Null(bool is_depth) {
        Image image{};
        image.data_format = u64(is_depth ? DataFormat::Format32 : DataFormat::Format8_8_8_8);
        image.num_format = u64(is_depth ? NumberFormat::Float : NumberFormat::Unorm);
        image.dst_sel_x = u64(CompSwizzle::Red);
        image.dst_sel_y = u64(CompSwizzle::Green);
        image.dst_sel_z = u64(CompSwizzle::Blue);
        image.dst_sel_w = u64(CompSwizzle::Alpha);
        image.tiling_index = u64(TileMode::Thin1DThin);
        image.type = u64(ImageType::Color2D);
        return image;
    }

    bool Valid() const {
        return type & 8;
    }
    VAddr Address() const {
        return base_address << 8;
    }
    explicit operator bool() const noexcept {
        return base_address != 0;
    }

    CompMapping DstSelect() const {
        return RemapSwizzle(DataFormat(data_format), {.r = CompSwizzle(dst_sel_x),
                                                      .g = CompSwizzle(dst_sel_y),
                                                      .b = CompSwizzle(dst_sel_z),
                                                      .a = CompSwizzle(dst_sel_w)});
    }

    u32 Pitch() const {
        return pitch + 1;
    }

    bool IsCube() const noexcept {
        return ImageType(type) == ImageType::Cube;
    }
    bool IsMsaa() const noexcept {
        return ImageType(type) == ImageType::Color2DMsaa ||
               ImageType(type) == ImageType::Color2DMsaaArray;
    }
    /// The type, cubes as 2D arrays.
    ImageType GetType() const noexcept {
        return IsCube() ? ImageType::Color2DArray : ImageType(type);
    }

    /// Layers: array slices (cubes: six faces each, 3D images one), padded to a power of two.
    [[nodiscard]] u32 NumLayers() const noexcept {
        u32 layers = GetType() == ImageType::Color3D ? 1 : u32(depth) + 1;
        if (IsCube()) {
            layers *= 6;
        }
        return pow2pad ? std::bit_ceil(layers) : layers;
    }
    /// MSAA images keep their sample count where others keep the last level.
    u32 NumLevels() const {
        return IsMsaa() ? 1 : u32(last_level) + 1;
    }
    u32 NumSamples() const {
        return IsMsaa() ? 1u << last_level : 1;
    }

    DataFormat GetDataFmt() const noexcept {
        return RemapDataFormat(DataFormat(data_format));
    }
    NumberFormat GetNumberFmt() const noexcept {
        return RemapNumberFormat(NumberFormat(num_format), DataFormat(data_format));
    }
    NumberConversion GetNumberConversion() const noexcept {
        return MapNumberConversion(NumberFormat(num_format), DataFormat(data_format));
    }

    TileMode GetTileMode() const {
        return TileMode(tiling_index);
    }
    bool IsTiled() const {
        return GetTileMode() != TileMode::DisplayLinearAligned &&
               GetTileMode() != TileMode::DisplayLinearGeneral;
    }

    /// PS4 Pro alternate tiling: the bank swizzle kept in the low bits of the address.
    u8 GetBankSwizzle() const {
        const TileMode tile_mode = GetTileMode();
        const DataFormat dfmt = GetDataFmt();
        if (!alt_tile_mode || dfmt == DataFormat::FormatInvalid ||
            !IsMacroTiled(GetArrayMode(tile_mode))) {
            return 0;
        }
        const auto macro =
            CalculateMacrotileMode(tile_mode, NumBitsPerElement(dfmt), NumSamples());
        const u32 banks = GetAltNumBanks(macro);
        return u8((((banks - 1) << 4) & base_address) >> 4);
    }

    /// The type without arrays and samples.
    ImageType GetBaseType() const noexcept {
        switch (const ImageType t = GetType()) {
        case ImageType::Color1DArray:
            return ImageType::Color1D;
        case ImageType::Color2DArray:
        case ImageType::Color2DMsaa:
        case ImageType::Color2DMsaaArray:
            return ImageType::Color2D;
        default:
            return t;
        }
    }

    /// The type of a view an instruction makes (`is_array`: it addresses layers). Cubes stay
    /// arrays either way.
    ImageType GetViewType(bool is_array) const noexcept {
        const ImageType t = GetType();
        if (IsCube() || is_array) {
            return t;
        }
        switch (t) {
        case ImageType::Color1DArray:
            return ImageType::Color1D;
        case ImageType::Color2DArray:
            return ImageType::Color2D;
        case ImageType::Color2DMsaaArray:
            return ImageType::Color2DMsaa;
        default:
            return t;
        }
    }

    /// Levels of such a view (at least one).
    [[nodiscard]] u32 NumViewLevels(bool is_array) const noexcept {
        const ImageType t = GetViewType(is_array);
        if (t == ImageType::Color2DMsaa || t == ImageType::Color2DMsaaArray) {
            return 1;
        }
        const u32 end = std::min<u32>(u32(last_level) + 1, NumLevels());
        return end > base_level ? end - u32(base_level) : 1;
    }

    /// Layers of such a view (at least one).
    [[nodiscard]] u32 NumViewLayers(bool is_array) const noexcept {
        switch (GetViewType(is_array)) {
        case ImageType::Color1D:
        case ImageType::Color2D:
        case ImageType::Color2DMsaa:
        case ImageType::Color3D:
            return 1;
        default: {
            const u32 end = std::min<u32>(u32(last_array) + 1, NumLayers());
            return end > base_array ? end - u32(base_array) : 1;
        }
        }
    }
};
static_assert(sizeof(Image) == 32);

enum class ClampMode : u64 {
    Wrap = 0,
    Mirror = 1,
    ClampLastTexel = 2,
    MirrorOnceLastTexel = 3,
    ClampHalfBorder = 4,
    MirrorOnceHalfBorder = 5,
    ClampBorder = 6,
    MirrorOnceBorder = 7,
};

enum class AnisoRatio : u64 {
    One,
    Two,
    Four,
    Eight,
    Sixteen,
};

enum class DepthCompare : u64 {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterEqual = 6,
    Always = 7,
};

enum class FilterMode : u64 {
    Blend = 0,
    Min = 1,
    Max = 2,
};

enum class Filter : u64 {
    Point = 0,
    Bilinear = 1,
    AnisoPoint = 2,
    AnisoLinear = 3,
};

constexpr bool IsAnisoFilter(Filter filter) {
    return filter == Filter::AnisoPoint || filter == Filter::AnisoLinear;
}

enum class MipFilter : u64 {
    None = 0,
    Point = 1,
    Linear = 2,
};

enum class BorderColor : u64 {
    TransparentBlack = 0,
    OpaqueBlack = 1,
    White = 2,
    Custom = 3,
};

/// S#: a sampler.
struct Sampler {
    ClampMode clamp_x : 3;
    ClampMode clamp_y : 3;
    ClampMode clamp_z : 3;
    AnisoRatio max_aniso : 3;
    DepthCompare depth_compare_func : 3;
    u64 force_unnormalized : 1;
    u64 aniso_threshold : 3;
    u64 mc_coord_trunc : 1;
    u64 force_degamma : 1;
    u64 aniso_bias : 6;
    u64 trunc_coord : 1;
    u64 disable_cube_wrap : 1;
    FilterMode filter_mode : 2;
    u64 : 1;
    u64 min_lod : 12;
    u64 max_lod : 12;
    u64 perf_mip : 4;
    u64 perf_z : 4;

    u64 lod_bias : 14;
    u64 lod_bias_sec : 6;
    Filter xy_mag_filter : 2;
    Filter xy_min_filter : 2;
    u64 z_filter : 2;
    MipFilter mip_filter : 2;
    u64 mip_point_preclamp : 1;
    u64 disable_lsb_ceil : 1;
    u64 : 2;
    u64 border_color_ptr : 12;
    u64 : 18;
    BorderColor border_color_type : 2;

    explicit operator bool() const noexcept {
        u64 raw[2];
        std::memcpy(raw, this, sizeof(raw));
        return raw[0] != 0 || raw[1] != 0;
    }
    bool Valid() const {
        return true;
    }
    bool operator==(const Sampler& other) const noexcept {
        return std::memcmp(this, &other, sizeof(Sampler)) == 0;
    }

    /// Signed 6.8 fixed point.
    float LodBias() const noexcept {
        return float(s16(u16(lod_bias << 2)) >> 2) / 256.0f;
    }
    /// Unsigned 4.8 fixed point.
    float MinLod() const noexcept {
        return float(min_lod) / 256.0f;
    }
    float MaxLod() const noexcept {
        return float(max_lod) / 256.0f;
    }
    float MaxAniso() const {
        // 3 bits, 0..4 valid: a stale S# read ahead (draw preparation) may hold 5..7; clamped
        // like the hardware's 16x cap instead of ending the session.
        const u32 ratio = std::min(u32(max_aniso), u32(AnisoRatio::Sixteen));
        return float(1u << ratio);
    }
};
static_assert(sizeof(Sampler) == 16);

} // namespace Gcn

template <>
struct fmt::formatter<Gcn::ImageType> : fmt::formatter<std::string_view> {
    auto format(Gcn::ImageType type, format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(Gcn::NameOf(type), ctx);
    }
};
