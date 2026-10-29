// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport gcn: the data and number formats of buffers, images and render targets (Sea Islands
// ISA, "Data and Image Formats"), their component swizzles, and how the translator maps the ones
// Vulkan lacks (the remaps: a component order or number format Vulkan has, plus a conversion the
// shaders apply).
#pragma once

#include <array>
#include <string_view>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/types.h"

namespace Gcn {

enum class DataFormat : u32 {
    FormatInvalid = 0,
    Format8 = 1,
    Format16 = 2,
    Format8_8 = 3,
    Format32 = 4,
    Format16_16 = 5,
    Format10_11_11 = 6,
    Format11_11_10 = 7,
    Format10_10_10_2 = 8,
    Format2_10_10_10 = 9,
    Format8_8_8_8 = 10,
    Format32_32 = 11,
    Format16_16_16_16 = 12,
    Format32_32_32 = 13,
    Format32_32_32_32 = 14,
    Format5_6_5 = 16,
    Format1_5_5_5 = 17,
    Format5_5_5_1 = 18,
    Format4_4_4_4 = 19,
    Format8_24 = 20,
    Format24_8 = 21,
    FormatX24_8_32 = 22,
    FormatGB_GR = 32,
    FormatBG_RG = 33,
    Format5_9_9_9 = 34,
    FormatBc1 = 35,
    FormatBc2 = 36,
    FormatBc3 = 37,
    FormatBc4 = 38,
    FormatBc5 = 39,
    FormatBc6 = 40,
    FormatBc7 = 41,
    FormatFmask8_S2_F1 = 44,
    FormatFmask8_S4_F1 = 45,
    FormatFmask8_S8_F1 = 46,
    FormatFmask8_S2_F2 = 47,
    FormatFmask8_S4_F2 = 48,
    FormatFmask8_S4_F4 = 49,
    FormatFmask16_S16_F1 = 50,
    FormatFmask16_S8_F2 = 51,
    FormatFmask32_S16_F2 = 52,
    FormatFmask32_S8_F4 = 53,
    FormatFmask32_S8_F8 = 54,
    FormatFmask64_S16_F4 = 55,
    FormatFmask64_S16_F8 = 56,
    Format4_4 = 57,
    Format6_5_5 = 58,
    Format1 = 59,
    Format1_Reversed = 60,
    Format32_As_8 = 61,
    Format32_As_8_8 = 62,
    Format32_As_32_32_32_32 = 63,
};

enum class NumberFormat : u32 {
    Unorm = 0,
    Snorm = 1,
    Uscaled = 2,
    Sscaled = 3,
    Uint = 4,
    Sint = 5,
    SnormNz = 6,
    Float = 7,
    Srgb = 9,
    Ubnorm = 10,
    UbnormNz = 11,
    Ubint = 12,
    Ubscaled = 13,
};

/// How a shader sees a format's values.
enum class NumberClass : u8 {
    Float = 0,
    Sint = 1,
    Uint = 2,
};

/// A destination component's source (DST_SEL).
enum class CompSwizzle : u8 {
    Zero = 0,
    One = 1,
    Red = 4,
    Green = 5,
    Blue = 6,
    Alpha = 7,
};

/// What the shaders do to a value read or written in the Vulkan format chosen for a remapped one.
enum class NumberConversion : u32 {
    None = 0,
    UintToUscaled = 1,
    SintToSscaled = 2,
    UnormToUbnorm = 3,
    Sint8ToSnormNz = 4,
    Sint16ToSnormNz = 5,
    Uint32ToUnorm = 6,
    SrgbToNorm = 7,
};

/// The four destination selects of a resource.
union CompMapping {
    struct {
        CompSwizzle r;
        CompSwizzle g;
        CompSwizzle b;
        CompSwizzle a;
    };
    std::array<CompSwizzle, 4> array;

    bool operator==(const CompMapping& other) const {
        return array == other.array;
    }

    /// The four values a read of `data` (r, g, b, a) gives.
    template <typename T>
    [[nodiscard]] std::array<T, 4> Apply(const std::array<T, 4>& data) const {
        std::array<T, 4> out;
        for (u32 i = 0; i < 4; ++i) {
            const u32 sel = u32(array[i]);
            out[i] = sel >= u32(CompSwizzle::Red) ? data[sel - u32(CompSwizzle::Red)]
                                                  : T(sel == u32(CompSwizzle::One) ? 1 : 0);
        }
        return out;
    }

    /// A write mask of destination components as a mask of source components.
    [[nodiscard]] u32 ApplyMask(u32 mask) const {
        u32 out = 0;
        for (u32 i = 0; i < 4; ++i) {
            out |= ((mask >> i) & 1) << Map(i);
        }
        return out;
    }

    /// The selects that undo this one (components nothing selects read zero).
    [[nodiscard]] CompMapping Inverse() const {
        CompMapping out{};
        for (u32 source = 0; source < 4; ++source) {
            out.array[source] = CompSwizzle::Zero;
            for (u32 i = 0; i < 4; ++i) {
                if (u32(array[i]) == u32(CompSwizzle::Red) + source) {
                    out.array[source] = CompSwizzle(u32(CompSwizzle::Red) + i);
                    break;
                }
            }
        }
        return out;
    }

    /// The source component of destination `comp` (constants keep their own index).
    [[nodiscard]] u32 Map(u32 comp) const {
        const u32 sel = u32(array[comp]);
        return sel >= u32(CompSwizzle::Red) ? sel - u32(CompSwizzle::Red) : comp;
    }
};

inline constexpr CompMapping IdentityMapping = {
    .r = CompSwizzle::Red,
    .g = CompSwizzle::Green,
    .b = CompSwizzle::Blue,
    .a = CompSwizzle::Alpha,
};

constexpr bool IsBlockCoded(DataFormat format) {
    return format >= DataFormat::FormatBc1 && format <= DataFormat::FormatBc7;
}

constexpr bool IsFmask(DataFormat format) {
    return format >= DataFormat::FormatFmask8_S2_F1 && format <= DataFormat::FormatFmask64_S16_F8;
}

constexpr bool Is32BitPerComponent(DataFormat format) {
    return format == DataFormat::Format32 || format == DataFormat::Format32_32 ||
           format == DataFormat::Format32_32_32 || format == DataFormat::Format32_32_32_32;
}

/// Packed formats stored with the components in the reverse order of a Vulkan format.
constexpr DataFormat RemapDataFormat(DataFormat format) {
    switch (format) {
    case DataFormat::Format11_11_10:
        return DataFormat::Format10_11_11;
    case DataFormat::Format10_10_10_2:
        return DataFormat::Format2_10_10_10;
    case DataFormat::Format5_5_5_1:
        return DataFormat::Format1_5_5_5;
    default:
        return format;
    }
}

/// The number format of the Vulkan format used (the difference made up by NumberConversion).
constexpr NumberFormat RemapNumberFormat(NumberFormat format, DataFormat data_format) {
    switch (format) {
    case NumberFormat::Unorm:
        // 32-bit unorm has no Vulkan format: read as uint, scaled in the shader.
        return Is32BitPerComponent(data_format) ? NumberFormat::Uint : format;
    case NumberFormat::Srgb:
        // BC4/BC5 hold non-color data and BC6 HDR floats: none has an sRGB Vulkan format.
        return data_format == DataFormat::FormatBc4 || data_format == DataFormat::FormatBc5 ||
                       data_format == DataFormat::FormatBc6
                   ? NumberFormat::Unorm
                   : format;
    case NumberFormat::Uscaled:
        return NumberFormat::Uint;
    case NumberFormat::Sscaled:
    case NumberFormat::SnormNz:
        return NumberFormat::Sint;
    case NumberFormat::Ubnorm:
        return NumberFormat::Unorm;
    case NumberFormat::Float:
        // 8-bit float: games read the stencil of a depth-stencil image this way. As uint it
        // matches the stencil format and passes the bits through.
        return data_format == DataFormat::Format8 ? NumberFormat::Uint : format;
    default:
        return format;
    }
}

/// The component order of the Vulkan format used, as destination selects.
constexpr CompMapping RemapSwizzle(DataFormat format, CompMapping swizzle) {
    const CompSwizzle in[4] = {swizzle.r, swizzle.g, swizzle.b, swizzle.a};
    const auto order = [&in](u32 r, u32 g, u32 b, u32 a) {
        return CompMapping{.r = in[r], .g = in[g], .b = in[b], .a = in[a]};
    };
    // One component: the PS4 shows it as R or A as the selects say, Vulkan only as R.
    const auto alpha_to_red = [](CompSwizzle sel) {
        return sel == CompSwizzle::Alpha ? CompSwizzle::Red : sel;
    };
    switch (format) {
    case DataFormat::Format1_5_5_5:
    case DataFormat::Format11_11_10:
    case DataFormat::Format5_6_5:
        return order(2, 1, 0, 3);
    case DataFormat::Format10_10_10_2:
        return order(3, 2, 1, 0);
    case DataFormat::Format4_4_4_4:
        return order(1, 2, 3, 0);
    case DataFormat::Format8:
    case DataFormat::Format16:
    case DataFormat::Format32:
        return CompMapping{.r = alpha_to_red(in[0]),
                           .g = alpha_to_red(in[1]),
                           .b = alpha_to_red(in[2]),
                           .a = alpha_to_red(in[3])};
    default:
        return swizzle;
    }
}

constexpr NumberConversion MapNumberConversion(NumberFormat num_fmt, DataFormat data_fmt) {
    switch (num_fmt) {
    case NumberFormat::Unorm:
        return Is32BitPerComponent(data_fmt) ? NumberConversion::Uint32ToUnorm
                                             : NumberConversion::None;
    case NumberFormat::Srgb:
        return data_fmt == DataFormat::FormatBc6 ? NumberConversion::SrgbToNorm
                                                 : NumberConversion::None;
    case NumberFormat::Uscaled:
        return NumberConversion::UintToUscaled;
    case NumberFormat::Sscaled:
        return NumberConversion::SintToSscaled;
    case NumberFormat::Ubnorm:
        return NumberConversion::UnormToUbnorm;
    case NumberFormat::SnormNz:
        switch (data_fmt) {
        case DataFormat::Format8:
        case DataFormat::Format8_8:
        case DataFormat::Format8_8_8_8:
            return NumberConversion::Sint8ToSnormNz;
        case DataFormat::Format16:
        case DataFormat::Format16_16:
        case DataFormat::Format16_16_16_16:
            return NumberConversion::Sint16ToSnormNz;
        default:
            // No real descriptor has it; a half-written one read ahead by a draw-preparation
            // worker can (#166). Nothing to convert: a specialization keyed by it never matches.
            return NumberConversion::None;
        }
    default:
        return NumberConversion::None;
    }
}

constexpr NumberClass GetNumberClass(NumberFormat nfmt) {
    return nfmt == NumberFormat::Sint   ? NumberClass::Sint
           : nfmt == NumberFormat::Uint ? NumberClass::Uint
                                        : NumberClass::Float;
}

constexpr bool IsRgb(CompSwizzle swizzle) {
    return swizzle == CompSwizzle::Red || swizzle == CompSwizzle::Green ||
           swizzle == CompSwizzle::Blue;
}

constexpr bool IsInteger(NumberFormat nfmt) {
    return nfmt == NumberFormat::Sint || nfmt == NumberFormat::Uint;
}

std::string_view NameOf(DataFormat fmt);
std::string_view NameOf(NumberFormat fmt);

/// Components, bits per block (a texel, or a 4x4 block of compressed formats) and bits per
/// texel. Reserved formats: no components, ~0 bits.
u32 NumComponents(DataFormat format);
u32 NumBitsPerBlock(DataFormat format);
u32 NumBitsPerElement(DataFormat format);

} // namespace Gcn

template <>
struct fmt::formatter<Gcn::DataFormat> : fmt::formatter<std::string_view> {
    auto format(Gcn::DataFormat fmt, format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(Gcn::NameOf(fmt), ctx);
    }
};

template <>
struct fmt::formatter<Gcn::NumberFormat> : fmt::formatter<std::string_view> {
    auto format(Gcn::NumberFormat fmt, format_context& ctx) const {
        return fmt::formatter<std::string_view>::format(Gcn::NameOf(fmt), ctx);
    }
};
