// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/gcn_to_vk.h"

#include <algorithm>
#include <array>
#include <bit>

#include "common/logging/log.h"
#include "common/number_utils.h"

namespace Vulkan::GcnToVk {

namespace {

/// A lookup table indexed by a GCN enum; `fallback` (with a warning naming `what`) where Vulkan
/// has nothing closer; out of range stops the port.
template <typename To, size_t N>
To Lookup(const std::array<To, N>& table, u32 index, const char* what) {
    ASSERT_MSG(index < N, "unknown {} {}", what, index);
    return table[index];
}

} // namespace

vk::StencilOp StencilOp(Gcn::StencilFunc op) {
    using enum vk::StencilOp;
    // Keep, Zero, Ones (none: replace), ReplaceTest, ReplaceOp, AddClamp, SubClamp, Invert,
    // AddWrap, SubWrap, then bitwise operations Vulkan lacks (keep).
    static constexpr std::array<vk::StencilOp, 16> table = {
        eKeep, eZero, eReplace, eReplace, eReplace, eIncrementAndClamp, eDecrementAndClamp,
        eInvert, eIncrementAndWrap, eDecrementAndWrap, eKeep, eKeep, eKeep, eKeep, eKeep, eKeep,
    };
    if (op == Gcn::StencilFunc::Ones) {
        LOG_WARNING(Render_Vulkan, "Unsupported stencil op {}, using Replace.", u32(op));
    } else if (u32(op) >= u32(Gcn::StencilFunc::And)) {
        LOG_WARNING(Render_Vulkan, "Unsupported bitwise stencil op {}, using Keep.", u32(op));
    }
    return Lookup(table, u32(op), "stencil op");
}

vk::CompareOp CompareOp(Gcn::CompareFunc func) {
    // The GCN order is Vulkan's.
    ASSERT_MSG(u32(func) <= u32(Gcn::CompareFunc::Always), "compare func {}", u32(func));
    return vk::CompareOp(u32(func));
}

bool IsPrimitiveCulled(Gcn::PrimitiveType type) {
    using enum Gcn::PrimitiveType;
    switch (type) {
    case TriangleList:
    case TriangleFan:
    case TriangleStrip:
    case PatchPrimitive:
    case AdjTriangleList:
    case AdjTriangleStrip:
    case QuadList:
    case QuadStrip:
    case Polygon:
        return true;
    case None:
    case PointList:
    case LineList:
    case LineStrip:
    case AdjLineList:
    case AdjLineStrip:
    case RectList: // screen-aligned, never culled
    case LineLoop:
        return false;
    default:
        UNREACHABLE_MSG("primitive type {}", u32(type));
    }
}

vk::PrimitiveTopology PrimitiveType(Gcn::PrimitiveType type) {
    using enum Gcn::PrimitiveType;
    using T = vk::PrimitiveTopology;
    switch (type) {
    case PointList:
        return T::ePointList;
    case LineList:
        return T::eLineList;
    case LineStrip:
        return T::eLineStrip;
    case TriangleList:
        return T::eTriangleList;
    case TriangleFan:
    case Polygon:
        return T::eTriangleFan;
    case TriangleStrip:
        return T::eTriangleStrip;
    case QuadStrip:
        LOG_WARNING(Render_Vulkan, "Unimplemented primitive type, using TriangleStrip.");
        return T::eTriangleStrip;
    case AdjLineList:
        return T::eLineListWithAdjacency;
    case AdjLineStrip:
        return T::eLineStripWithAdjacency;
    case AdjTriangleList:
        return T::eTriangleListWithAdjacency;
    case AdjTriangleStrip:
        return T::eTriangleStripWithAdjacency;
    // Quads and rectangles are expanded by a tessellation stage from patches.
    case PatchPrimitive:
    case QuadList:
    case RectList:
        return T::ePatchList;
    default:
        UNREACHABLE_MSG("Unimplemented primitive type: {}", u32(type));
    }
}

vk::PolygonMode PolygonMode(Gcn::PolygonMode mode) {
    static constexpr std::array table = {vk::PolygonMode::ePoint, vk::PolygonMode::eLine,
                                         vk::PolygonMode::eFill};
    return Lookup(table, u32(mode), "polygon mode");
}

vk::CullModeFlags CullMode(Gcn::CullMode mode) {
    ASSERT_MSG(u32(mode) <= 3, "cull mode {}", u32(mode));
    return vk::CullModeFlags(u32(mode)); // none, front, back, both: Vulkan's bits
}

vk::FrontFace FrontFace(Gcn::FrontFace face) {
    return face == Gcn::FrontFace::Clockwise ? vk::FrontFace::eClockwise
                                             : vk::FrontFace::eCounterClockwise;
}

vk::BlendFactor BlendFactor(Gcn::BlendControl::BlendFactor factor) {
    using enum vk::BlendFactor;
    // By GCN number; 11 and 12 do not exist.
    static constexpr std::array<vk::BlendFactor, 21> table = {
        eZero, eOne, eSrcColor, eOneMinusSrcColor, eSrcAlpha, eOneMinusSrcAlpha, eDstAlpha,
        eOneMinusDstAlpha, eDstColor, eOneMinusDstColor, eSrcAlphaSaturate, eZero, eZero,
        eConstantColor, eOneMinusConstantColor, eSrc1Color, eOneMinusSrc1Color, eSrc1Alpha,
        eOneMinusSrc1Alpha, eConstantAlpha, eOneMinusConstantAlpha,
    };
    const u32 index = u32(factor);
    ASSERT_MSG(index < table.size() && index != 11 && index != 12, "Unknown blend factor: {}",
               index);
    return table[index];
}

bool IsDualSourceBlendFactor(Gcn::BlendControl::BlendFactor factor) {
    using enum Gcn::BlendControl::BlendFactor;
    return factor == Src1Color || factor == Src1Alpha || factor == InvSrc1Color ||
           factor == InvSrc1Alpha;
}

vk::BlendOp BlendOp(Gcn::BlendControl::BlendFunc func) {
    using enum vk::BlendOp;
    static constexpr std::array table = {eAdd, eSubtract, eMin, eMax, eReverseSubtract};
    return Lookup(table, u32(func), "blend op");
}

vk::LogicOp LogicOp(Gcn::ColorControl::LogicOp op) {
    using G = Gcn::ColorControl::LogicOp;
    using enum vk::LogicOp;
    // ROP3 codes of the 16 two-operand operations, and BrushXor (pattern xor dest: as xor).
    struct Entry {
        G rop;
        vk::LogicOp op;
    };
    static constexpr std::array<Entry, 17> table = {{
        {G::Clear, eClear},
        {G::Nor, eNor},
        {G::AndInverted, eAndInverted},
        {G::CopyInverted, eCopyInverted},
        {G::AndReverse, eAndReverse},
        {G::Invert, eInvert},
        {G::BrushXor, eXor},
        {G::Xor, eXor},
        {G::Nand, eNand},
        {G::And, eAnd},
        {G::Equiv, eEquivalent},
        {G::Noop, eNoOp},
        {G::OrInverted, eOrInverted},
        {G::Copy, eCopy},
        {G::OrReverse, eOrReverse},
        {G::Or, eOr},
        {G::Set, eSet},
    }};
    if (op == G::BrushXor) {
        LOG_WARNING(Render_Vulkan, "Unimplemented logic op {:#x}, using closest equivalent",
                    u32(op));
    }
    for (const auto& entry : table) {
        if (entry.rop == op) {
            return entry.op;
        }
    }
    UNREACHABLE_MSG("Unknown logic op {}", u32(op));
}

vk::SamplerAddressMode ClampMode(Gcn::ClampMode mode) {
    using enum vk::SamplerAddressMode;
    // Wrap, Mirror, ClampLastTexel, MirrorOnceLastTexel, ClampHalfBorder, MirrorOnceHalfBorder,
    // ClampBorder, MirrorOnceBorder: half borders and mirrored borders as their closest.
    static constexpr std::array table = {eRepeat,           eMirroredRepeat,     eClampToEdge,
                                         eMirrorClampToEdge, eClampToBorder,     eMirrorClampToEdge,
                                         eClampToBorder,    eMirrorClampToEdge};
    if (mode == Gcn::ClampMode::ClampHalfBorder || mode == Gcn::ClampMode::MirrorOnceHalfBorder ||
        mode == Gcn::ClampMode::MirrorOnceBorder) {
        LOG_WARNING(Render_Vulkan, "Unimplemented clamp mode {}, using closest equivalent.",
                    u32(mode));
    }
    return Lookup(table, u32(mode), "clamp mode");
}

vk::CompareOp DepthCompare(Gcn::DepthCompare comp) {
    ASSERT_MSG(u32(comp) <= u32(Gcn::DepthCompare::Always), "depth compare {}", u32(comp));
    return vk::CompareOp(u32(comp));
}

vk::Filter Filter(Gcn::Filter filter) {
    static constexpr std::array table = {vk::Filter::eNearest, vk::Filter::eLinear,
                                         vk::Filter::eNearest, vk::Filter::eLinear};
    return Lookup(table, u32(filter), "filter"); // anisotropic ones by their base filter
}

vk::SamplerReductionMode FilterMode(Gcn::FilterMode mode) {
    using enum vk::SamplerReductionMode;
    static constexpr std::array table = {eWeightedAverage, eMin, eMax};
    return Lookup(table, u32(mode), "filter mode");
}

vk::SamplerMipmapMode MipFilter(Gcn::MipFilter filter) {
    using enum vk::SamplerMipmapMode;
    static constexpr std::array table = {eNearest, eNearest, eLinear}; // none: the nearest level
    return Lookup(table, u32(filter), "mip filter");
}

vk::BorderColor BorderColor(Gcn::BorderColor color) {
    using enum vk::BorderColor;
    static constexpr std::array table = {eFloatTransparentBlack, eFloatOpaqueBlack,
                                         eFloatOpaqueWhite, eFloatCustomEXT};
    return Lookup(table, u32(color), "border color");
}

vk::ComponentSwizzle ComponentSwizzle(Gcn::CompSwizzle swizzle) {
    using enum vk::ComponentSwizzle;
    switch (swizzle) {
    case Gcn::CompSwizzle::Zero:
        return eZero;
    case Gcn::CompSwizzle::One:
        return eOne;
    case Gcn::CompSwizzle::Red:
        return eR;
    case Gcn::CompSwizzle::Green:
        return eG;
    case Gcn::CompSwizzle::Blue:
        return eB;
    case Gcn::CompSwizzle::Alpha:
        return eA;
    default:
        UNREACHABLE_MSG("component swizzle {}", u32(swizzle));
    }
}

vk::ComponentMapping ComponentMapping(Gcn::CompMapping mapping) {
    return {
        .r = ComponentSwizzle(mapping.r),
        .g = ComponentSwizzle(mapping.g),
        .b = ComponentSwizzle(mapping.b),
        .a = ComponentSwizzle(mapping.a),
    };
}

namespace {

// What GCN allows each data and number format for (Sea Islands ISA, "Data and Image Formats"):
// buffer reads (vertex fetch), image reads, storage writes, render targets. Texel buffers need
// nothing: the shaders interpret their format.
constexpr vk::FormatFeatureFlags2 BufferRead = vk::FormatFeatureFlagBits2::eVertexBuffer;
constexpr vk::FormatFeatureFlags2 ImageRead = vk::FormatFeatureFlagBits2::eTransferSrc |
                                              vk::FormatFeatureFlagBits2::eTransferDst |
                                              vk::FormatFeatureFlagBits2::eSampledImage;
constexpr vk::FormatFeatureFlags2 ImageWrite =
    vk::FormatFeatureFlagBits2::eStorageImage |
    vk::FormatFeatureFlagBits2::eStorageReadWithoutFormat |
    vk::FormatFeatureFlagBits2::eStorageWriteWithoutFormat;
constexpr vk::FormatFeatureFlags2 RenderTarget = vk::FormatFeatureFlagBits2::eColorAttachment;
constexpr vk::FormatFeatureFlags2 AllUses = BufferRead | ImageRead | ImageWrite | RenderTarget;

constexpr vk::FormatFeatureFlags2 DataFormatUses(Gcn::DataFormat format) {
    using enum Gcn::DataFormat;
    switch (format) {
    case Format32_32_32:
        return BufferRead | ImageRead;
    case Format5_6_5:
    case Format1_5_5_5:
    case Format5_5_5_1:
    case Format4_4_4_4:
        return ImageRead | ImageWrite | RenderTarget;
    case Format8_24:
    case Format24_8:
    case FormatX24_8_32:
        return ImageRead | RenderTarget;
    default:
        if (format <= Format32_32_32_32) {
            return AllUses; // the plain formats up to 32_32_32_32
        }
        return Gcn::IsFmask(format) ? ImageRead | ImageWrite : ImageRead;
    }
}

constexpr vk::FormatFeatureFlags2 NumberFormatUses(Gcn::NumberFormat format) {
    using enum Gcn::NumberFormat;
    switch (format) {
    case Unorm:
    case Snorm:
    case Uint:
    case Sint:
    case Float:
        return AllUses;
    case Uscaled:
    case Sscaled:
    case SnormNz:
        return BufferRead | ImageRead;
    case Srgb:
        return ImageRead | RenderTarget;
    default:
        return ImageRead;
    }
}

constexpr SurfaceFormatInfo Surface(Gcn::DataFormat data, Gcn::NumberFormat number,
                                    vk::Format format) {
    return {data, number, format, DataFormatUses(data) & NumberFormatUses(number)};
}

/// Pairs with a Vulkan format. Uscaled, Sscaled and Ubnorm are remapped (and converted in the
/// shaders) before they get here; the invalid format has every number format, unformatted.
constexpr auto SurfaceFormatTable = [] {
    using D = Gcn::DataFormat;
    using N = Gcn::NumberFormat;
    using F = vk::Format;
    constexpr std::array<N, 13> numbers = {N::Unorm,   N::Snorm,  N::Uscaled, N::Sscaled, N::Uint,
                                           N::Sint,    N::SnormNz, N::Float,  N::Srgb,    N::Ubnorm,
                                           N::UbnormNz, N::Ubint, N::Ubscaled};
    std::array<SurfaceFormatInfo, 80> t{};
    size_t n = 0;
    for (const N number : numbers) {
        t[n++] = Surface(D::FormatInvalid, number, F::eUndefined);
    }
    const auto add = [&](D data, std::initializer_list<std::pair<N, F>> pairs) {
        for (const auto& [number, format] : pairs) {
            t[n++] = Surface(data, number, format);
        }
    };
    add(D::Format8, {{N::Unorm, F::eR8Unorm}, {N::Snorm, F::eR8Snorm}, {N::Uint, F::eR8Uint},
                     {N::Sint, F::eR8Sint}, {N::Srgb, F::eR8Srgb}});
    add(D::Format16, {{N::Unorm, F::eR16Unorm}, {N::Snorm, F::eR16Snorm}, {N::Uint, F::eR16Uint},
                      {N::Sint, F::eR16Sint}, {N::Float, F::eR16Sfloat}});
    add(D::Format8_8, {{N::Unorm, F::eR8G8Unorm}, {N::Snorm, F::eR8G8Snorm},
                       {N::Uint, F::eR8G8Uint}, {N::Sint, F::eR8G8Sint}, {N::Srgb, F::eR8G8Srgb}});
    add(D::Format32, {{N::Uint, F::eR32Uint}, {N::Sint, F::eR32Sint}, {N::Float, F::eR32Sfloat}});
    add(D::Format16_16, {{N::Unorm, F::eR16G16Unorm}, {N::Snorm, F::eR16G16Snorm},
                         {N::Uint, F::eR16G16Uint}, {N::Sint, F::eR16G16Sint},
                         {N::Float, F::eR16G16Sfloat}});
    add(D::Format10_11_11, {{N::Float, F::eB10G11R11UfloatPack32}});
    // 11_11_10 and 10_10_10_2 are remapped to 10_11_11 and 2_10_10_10.
    add(D::Format2_10_10_10, {{N::Unorm, F::eA2B10G10R10UnormPack32},
                              {N::Snorm, F::eA2B10G10R10SnormPack32},
                              {N::Uint, F::eA2B10G10R10UintPack32},
                              {N::Sint, F::eA2B10G10R10SintPack32}});
    add(D::Format8_8_8_8, {{N::Unorm, F::eR8G8B8A8Unorm}, {N::Snorm, F::eR8G8B8A8Snorm},
                           {N::Uint, F::eR8G8B8A8Uint}, {N::Sint, F::eR8G8B8A8Sint},
                           {N::Srgb, F::eR8G8B8A8Srgb}});
    add(D::Format32_32, {{N::Uint, F::eR32G32Uint}, {N::Sint, F::eR32G32Sint},
                         {N::Float, F::eR32G32Sfloat}});
    add(D::Format16_16_16_16,
        {{N::Unorm, F::eR16G16B16A16Unorm}, {N::Snorm, F::eR16G16B16A16Snorm},
         {N::Uint, F::eR16G16B16A16Uint}, {N::Sint, F::eR16G16B16A16Sint},
         {N::SnormNz, F::eR16G16B16A16Snorm}, {N::Float, F::eR16G16B16A16Sfloat}});
    add(D::Format32_32_32, {{N::Uint, F::eR32G32B32Uint}, {N::Sint, F::eR32G32B32Sint},
                            {N::Float, F::eR32G32B32Sfloat}});
    add(D::Format32_32_32_32, {{N::Uint, F::eR32G32B32A32Uint}, {N::Sint, F::eR32G32B32A32Sint},
                               {N::Float, F::eR32G32B32A32Sfloat}});
    add(D::Format5_6_5, {{N::Unorm, F::eR5G6B5UnormPack16}});
    add(D::Format1_5_5_5, {{N::Unorm, F::eA1R5G5B5UnormPack16}});
    add(D::Format5_5_5_1, {{N::Unorm, F::eR5G5B5A1UnormPack16}});
    add(D::Format4_4_4_4, {{N::Unorm, F::eB4G4R4A4UnormPack16}});
    add(D::Format5_9_9_9, {{N::Float, F::eE5B9G9R9UfloatPack32}});
    add(D::FormatBc1, {{N::Unorm, F::eBc1RgbaUnormBlock}, {N::Srgb, F::eBc1RgbaSrgbBlock}});
    add(D::FormatBc2, {{N::Unorm, F::eBc2UnormBlock}, {N::Srgb, F::eBc2SrgbBlock}});
    add(D::FormatBc3, {{N::Unorm, F::eBc3UnormBlock}, {N::Srgb, F::eBc3SrgbBlock}});
    add(D::FormatBc4, {{N::Unorm, F::eBc4UnormBlock}, {N::Snorm, F::eBc4SnormBlock}});
    add(D::FormatBc5, {{N::Unorm, F::eBc5UnormBlock}, {N::Snorm, F::eBc5SnormBlock}});
    add(D::FormatBc6, {{N::Unorm, F::eBc6HUfloatBlock}, {N::Snorm, F::eBc6HSfloatBlock}});
    add(D::FormatBc7, {{N::Unorm, F::eBc7UnormBlock}, {N::Srgb, F::eBc7SrgbBlock}});
    if (n != t.size()) {
        throw "surface format table size";
    }
    return t;
}();

/// [data format][number format]: the Vulkan format (eUndefined: none).
constexpr auto SurfaceFormatIndex = [] {
    std::array<std::array<vk::Format, 16>, 64> index{};
    for (auto& row : index) {
        row.fill(vk::Format::eUndefined);
    }
    for (const auto& entry : SurfaceFormatTable) {
        index[u32(entry.data_format)][u32(entry.number_format)] = entry.vk_format;
    }
    return index;
}();

} // namespace

std::span<const SurfaceFormatInfo> SurfaceFormats() {
    return SurfaceFormatTable;
}

vk::Format TrySurfaceFormat(Gcn::DataFormat data_format, Gcn::NumberFormat num_format) {
    return SurfaceFormatIndex[u32(data_format) & 63][u32(num_format) & 15];
}

vk::Format SurfaceFormat(Gcn::DataFormat data_format, Gcn::NumberFormat num_format) {
    const vk::Format format = TrySurfaceFormat(data_format, num_format);
    ASSERT_MSG(format != vk::Format::eUndefined || data_format == Gcn::DataFormat::FormatInvalid,
               "Unknown data_format={} and num_format={}", u32(data_format), u32(num_format));
    return format;
}

std::span<const DepthFormatInfo> DepthFormats() {
    using Z = Gcn::DepthBuffer::ZFormat;
    using S = Gcn::DepthBuffer::StencilFormat;
    constexpr auto attachment = vk::FormatFeatureFlagBits2::eDepthStencilAttachment;
    static constexpr std::array<DepthFormatInfo, 6> formats = {{
        {Z::Invalid, S::Invalid, vk::Format::eUndefined, attachment},
        {Z::Invalid, S::Stencil8, vk::Format::eD32SfloatS8Uint, attachment},
        {Z::Z16, S::Invalid, vk::Format::eD16Unorm, attachment},
        {Z::Z16, S::Stencil8, vk::Format::eD16UnormS8Uint, attachment},
        {Z::Z32Float, S::Invalid, vk::Format::eD32Sfloat, attachment},
        {Z::Z32Float, S::Stencil8, vk::Format::eD32SfloatS8Uint, attachment},
    }};
    return formats;
}

vk::Format DepthFormat(Gcn::DepthBuffer::ZFormat z_format,
                       Gcn::DepthBuffer::StencilFormat stencil_format) {
    for (const auto& entry : DepthFormats()) {
        if (entry.z_format == z_format && entry.stencil_format == stencil_format) {
            return entry.vk_format;
        }
    }
    UNREACHABLE_MSG("Unknown z_format={} and stencil_format={}", u32(z_format),
                    u32(stencil_format));
}

namespace {

/// A component of a clear value: `bits` at `shift` of the 64-bit clear value.
struct Field {
    u8 shift;
    u8 bits;
};

/// Clear value layouts: components in GCN order, number formats allowed.
struct ClearLayout {
    u8 count;
    Field fields[4];
    bool unorm;  ///< and sRGB when `srgb`
    bool srgb;
    bool snorm;  ///< and SnormNz
    bool integer;
    bool floats;
};

float ToUnorm(u32 value, u32 bits) {
    using namespace NumberUtils;
    switch (bits) {
    case 2:
        return U2ToUnorm(u8(value));
    case 4:
        return U4ToUnorm(u8(value));
    case 5:
        return U5ToUnorm(u8(value));
    case 6:
        return U6ToUnorm(u8(value));
    case 8:
        return U8ToUnorm(u8(value));
    case 10:
        return U10ToUnorm(u16(value));
    case 16:
        return U16ToUnorm(u16(value));
    default:
        return value ? 1.0f : 0.0f; // one bit
    }
}

float ToSnorm(u32 value, u32 bits) {
    using namespace NumberUtils;
    switch (bits) {
    case 2:
        return S2ToSnorm(s8(value));
    case 8:
        return S8ToSnorm(s8(value));
    case 10:
        return S10ToSnorm(s16(value));
    case 16:
        return S16ToSnorm(s16(value));
    default:
        UNREACHABLE_MSG("{}-bit snorm", bits);
    }
}

float ToFloat(u32 value, u32 bits) {
    switch (bits) {
    case 10:
        return NumberUtils::Uf10ToF32(u16(value));
    case 11:
        return NumberUtils::Uf11ToF32(u16(value));
    case 16:
        return NumberUtils::Uf16ToF32(u16(value));
    default:
        return std::bit_cast<float>(value);
    }
}

const ClearLayout* ClearLayoutOf(Gcn::DataFormat format) {
    using enum Gcn::DataFormat;
    // count, fields, unorm, srgb, snorm, integer, float
    static constexpr ClearLayout c8{1, {{0, 8}}, true, true, true, true, false};
    static constexpr ClearLayout c16{1, {{0, 16}}, true, false, true, true, true};
    static constexpr ClearLayout c8_8{2, {{0, 8}, {8, 8}}, true, true, true, true, false};
    static constexpr ClearLayout c32{1, {{0, 32}}, false, false, false, true, true};
    static constexpr ClearLayout c16_16{2, {{0, 16}, {16, 16}}, true, false, true, true, true};
    static constexpr ClearLayout c10_10_10_2{
        4, {{0, 2}, {2, 10}, {12, 10}, {22, 10}}, true, false, true, true, false};
    static constexpr ClearLayout c2_10_10_10{
        4, {{0, 10}, {10, 10}, {20, 10}, {30, 2}}, true, false, true, true, false};
    static constexpr ClearLayout c8_8_8_8{
        4, {{0, 8}, {8, 8}, {16, 8}, {24, 8}}, true, true, true, true, false};
    static constexpr ClearLayout c32_32{2, {{0, 32}, {32, 32}}, false, false, false, true, true};
    static constexpr ClearLayout c16_16_16_16{
        4, {{0, 16}, {16, 16}, {32, 16}, {48, 16}}, true, false, true, true, true};
    // Only 64 bits of clear value: the first word for x, y and z.
    static constexpr ClearLayout c32_32_32_32{
        4, {{0, 32}, {0, 32}, {0, 32}, {32, 32}}, false, false, false, true, true};
    switch (format) {
    case Format8:
        return &c8;
    case Format16:
        return &c16;
    case Format8_8:
        return &c8_8;
    case Format32:
        return &c32;
    case Format16_16:
        return &c16_16;
    case Format10_10_10_2:
        return &c10_10_10_2;
    case Format2_10_10_10:
        return &c2_10_10_10;
    case Format8_8_8_8:
        return &c8_8_8_8;
    case Format32_32:
        return &c32_32;
    case Format16_16_16_16:
        return &c16_16_16_16;
    case Format32_32_32_32:
        return &c32_32_32_32;
    default:
        return nullptr;
    }
}

/// Packed formats with one interpretation whatever the number format: components as
/// (shift, bits, float?) and unorm otherwise.
bool PackedClear(Gcn::DataFormat format, u64 value, vk::ClearColorValue& color) {
    using enum Gcn::DataFormat;
    const auto bits = [&](u32 shift, u32 count) { return u32(value >> shift) & ((1u << count) - 1); };
    switch (format) {
    case Format10_11_11:
        color.float32[0] = NumberUtils::Uf11ToF32(u16(bits(0, 11)));
        color.float32[1] = NumberUtils::Uf11ToF32(u16(bits(11, 11)));
        color.float32[2] = NumberUtils::Uf10ToF32(u16(bits(22, 10)));
        return true;
    case Format11_11_10:
        color.float32[0] = NumberUtils::Uf10ToF32(u16(bits(0, 10)));
        color.float32[1] = NumberUtils::Uf11ToF32(u16(bits(10, 11)));
        color.float32[2] = NumberUtils::Uf11ToF32(u16(bits(21, 11)));
        return true;
    case Format5_9_9_9: {
        // A shared exponent (5 bits, bias 15) over 9-bit mantissas.
        const float scale = std::bit_cast<float>(u32(int(bits(27, 5)) - 10 + 127) << 23);
        for (u32 i = 0; i < 3; ++i) {
            color.float32[i] = float(bits(i * 9, 9)) * scale;
        }
        return true;
    }
    case Format5_6_5:
        color.float32[0] = ToUnorm(bits(0, 5), 5);
        color.float32[1] = ToUnorm(bits(5, 6), 6);
        color.float32[2] = ToUnorm(bits(11, 5), 5);
        return true;
    case Format1_5_5_5:
        for (u32 i = 0; i < 3; ++i) {
            color.float32[i] = ToUnorm(bits(i * 5, 5), 5);
        }
        color.float32[3] = bits(15, 1) ? 1.0f : 0.0f;
        return true;
    case Format5_5_5_1:
        color.float32[0] = bits(0, 1) ? 1.0f : 0.0f;
        for (u32 i = 1; i < 4; ++i) {
            color.float32[i] = ToUnorm(bits(1 + (i - 1) * 5, 5), 5);
        }
        return true;
    case Format4_4_4_4:
        for (u32 i = 0; i < 4; ++i) {
            color.float32[i] = ToUnorm(bits(i * 4, 4), 4);
        }
        return true;
    default:
        return false;
    }
}

} // namespace

vk::ClearValue ColorBufferClearValue(const Gcn::ColorBuffer& color_buffer) {
    const auto format = Gcn::DataFormat(color_buffer.info.format);
    const auto number = color_buffer.GetFixedNumberFormat();
    const u64 value = u64(color_buffer.clear_word0) | u64(color_buffer.clear_word1) << 32;
    const bool integer = number == Gcn::NumberFormat::Uint || number == Gcn::NumberFormat::Sint;

    vk::ClearColorValue color{};
    if (integer) {
        color.uint32[3] = 1; // components a format lacks: alpha one
    } else {
        color.float32[3] = 1.0f;
    }

    if (!PackedClear(format, value, color)) {
        const ClearLayout* layout = ClearLayoutOf(format);
        const bool unorm = number == Gcn::NumberFormat::Unorm ||
                           (number == Gcn::NumberFormat::Srgb && layout && layout->srgb);
        const bool snorm =
            number == Gcn::NumberFormat::Snorm || number == Gcn::NumberFormat::SnormNz;
        const bool floats = number == Gcn::NumberFormat::Float;
        if (!layout) {
            LOG_ERROR(Render_Vulkan, "Unsupported color buffer format: {}", format);
        } else if (!(unorm && layout->unorm) && !(snorm && layout->snorm) &&
                   !(integer && layout->integer) && !(floats && layout->floats)) {
            LOG_ERROR(Render_Vulkan, "Unsupported number type {} for format {}", number, format);
        } else {
            for (u32 i = 0; i < layout->count; ++i) {
                const Field& field = layout->fields[i];
                const u64 mask = field.bits == 64 ? ~0ull : (1ull << field.bits) - 1;
                const u32 raw = u32((value >> field.shift) & mask);
                if (integer) {
                    color.uint32[i] = raw;
                } else if (floats) {
                    color.float32[i] = ToFloat(raw, field.bits);
                } else if (snorm) {
                    color.float32[i] = ToSnorm(raw, field.bits);
                } else {
                    color.float32[i] = ToUnorm(raw, field.bits);
                }
            }
        }
    }
    // In the target's component order (the bits move as they are, integer or float).
    color.float32 = color_buffer.Swizzle().Apply(color.float32);
    return {.color = color};
}

vk::SampleCountFlagBits NumSamples(u32 num_samples, vk::SampleCountFlags supported) {
    ASSERT_MSG(std::has_single_bit(num_samples) && num_samples <= 16, "{} samples",
               num_samples);
    while (num_samples > 1 && !(supported & vk::SampleCountFlagBits(num_samples))) {
        num_samples /= 2;
    }
    return vk::SampleCountFlagBits(num_samples);
}

} // namespace Vulkan::GcnToVk
