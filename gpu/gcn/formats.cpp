// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "gcn/formats.h"

namespace Gcn {

namespace {

constexpr u32 Reserved = ~0u;

struct FormatInfo {
    std::string_view name;
    u32 components;
    u32 block_bits;   ///< a texel, or a 4x4 block of block-coded formats
    u32 element_bits; ///< a texel
};

/// By data format number. Fmask surfaces and the formats after them carry their name only.
constexpr std::array<FormatInfo, 64> Formats = [] {
    std::array<FormatInfo, 64> t{};
    for (auto& f : t) {
        f = {"", 0, Reserved, Reserved};
    }
    const auto set = [&t](DataFormat format, std::string_view name, u32 components, u32 bits,
                          u32 element_bits) {
        t[u32(format)] = {name, components, bits, element_bits};
    };
    using enum DataFormat;
    set(FormatInvalid, "FormatInvalid", 0, 0, 0);
    set(Format8, "Format8", 1, 8, 8);
    set(Format16, "Format16", 1, 16, 16);
    set(Format8_8, "Format8_8", 2, 16, 16);
    set(Format32, "Format32", 1, 32, 32);
    set(Format16_16, "Format16_16", 2, 32, 32);
    set(Format10_11_11, "Format10_11_11", 3, 32, 32);
    set(Format11_11_10, "Format11_11_10", 3, 32, 32);
    set(Format10_10_10_2, "Format10_10_10_2", 4, 32, 32);
    set(Format2_10_10_10, "Format2_10_10_10", 4, 32, 32);
    set(Format8_8_8_8, "Format8_8_8_8", 4, 32, 32);
    set(Format32_32, "Format32_32", 2, 64, 64);
    set(Format16_16_16_16, "Format16_16_16_16", 4, 64, 64);
    set(Format32_32_32, "Format32_32_32", 3, 96, 96);
    set(Format32_32_32_32, "Format32_32_32_32", 4, 128, 128);
    set(Format5_6_5, "Format5_6_5", 3, 16, 16);
    set(Format1_5_5_5, "Format1_5_5_5", 4, 16, 16);
    set(Format5_5_5_1, "Format5_5_5_1", 4, 16, 16);
    set(Format4_4_4_4, "Format4_4_4_4", 4, 16, 16);
    set(Format8_24, "Format8_24", 2, 32, 32);
    set(Format24_8, "Format24_8", 2, 32, 32);
    set(FormatX24_8_32, "FormatX24_8_32", 2, 64, 64);
    set(FormatGB_GR, "FormatGB_GR", 3, 16, 16);
    set(FormatBG_RG, "FormatBG_RG", 3, 16, 16);
    set(Format5_9_9_9, "Format5_9_9_9", 4, 32, 32);
    // Block-coded: 4x4 texels per block.
    set(FormatBc1, "FormatBc1", 4, 64, 4);
    set(FormatBc2, "FormatBc2", 4, 128, 8);
    set(FormatBc3, "FormatBc3", 4, 128, 8);
    set(FormatBc4, "FormatBc4", 1, 64, 4);
    set(FormatBc5, "FormatBc5", 2, 128, 8);
    set(FormatBc6, "FormatBc6", 3, 128, 8);
    set(FormatBc7, "FormatBc7", 4, 128, 8);
    t[44].name = "FormatFmask8_S2_F1";
    t[45].name = "FormatFmask8_S4_F1";
    t[46].name = "FormatFmask8_S8_F1";
    t[47].name = "FormatFmask8_S2_F2";
    t[48].name = "FormatFmask8_S4_F2";
    t[49].name = "FormatFmask8_S4_F4";
    t[50].name = "FormatFmask16_S16_F1";
    t[51].name = "FormatFmask16_S8_F2";
    t[52].name = "FormatFmask32_S16_F2";
    t[53].name = "FormatFmask32_S8_F4";
    t[54].name = "FormatFmask32_S8_F8";
    t[55].name = "FormatFmask64_S16_F4";
    t[56].name = "FormatFmask64_S16_F8";
    t[57].name = "Format4_4";
    t[58].name = "Format6_5_5";
    t[59].name = "Format1";
    t[60].name = "Format1_Reversed";
    t[61].name = "Format32_As_8";
    t[62].name = "Format32_As_8_8";
    t[63].name = "Format32_As_32_32_32_32";
    return t;
}();

/// Formats with sizes known (the translator handles nothing after BC7).
const FormatInfo& Info(DataFormat format) {
    const u32 index = u32(format);
    ASSERT_MSG(index <= u32(DataFormat::FormatBc7), "data format {} has no size", index);
    return Formats[index];
}

} // namespace

std::string_view NameOf(DataFormat fmt) {
    const u32 index = u32(fmt);
    return index < Formats.size() && !Formats[index].name.empty() ? Formats[index].name
                                                                  : "FormatReserved";
}

std::string_view NameOf(NumberFormat fmt) {
    static constexpr std::array<std::string_view, 14> Names = {
        "Unorm", "Snorm", "Uscaled", "Sscaled", "Uint",     "Sint",  "SnormNz",
        "Float", "",      "Srgb",    "Ubnorm",  "UbnormNz", "Ubint", "Ubscaled",
    };
    const u32 index = u32(fmt);
    return index < Names.size() && !Names[index].empty() ? Names[index] : "NumberReserved";
}

u32 NumComponents(DataFormat format) {
    return Info(format).components;
}

u32 NumBitsPerBlock(DataFormat format) {
    return Info(format).block_bits;
}

u32 NumBitsPerElement(DataFormat format) {
    return Info(format).element_bits;
}

} // namespace Gcn
