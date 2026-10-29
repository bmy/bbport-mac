// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "textures/image_info.h"

#include <algorithm>
#include <bit>

#include "common/assert.h"
#include "platform/kernel.h"
#include "platform/videoout/videoout.h"
#include "shader/resource.h"
#include "textures/surface_layout.h"
#include "render/gcn_to_vk.h"

namespace VideoCore {

namespace {

using Vulkan::GcnToVk::DepthFormat;
using Vulkan::GcnToVk::PromoteFormatToDepth;
using Vulkan::GcnToVk::SurfaceFormat;

/// Display buffers keep RGBA order here; the frame's view swaps to BGRA where needed.
vk::Format DisplayFormat(BbVideoOut::PixelFormat format) {
    using enum BbVideoOut::PixelFormat;
    switch (format) {
    case A8B8G8R8Srgb:
    case A8R8G8B8Srgb:
        return vk::Format::eR8G8B8A8Srgb;
    case A2R10G10B10:
    case A2R10G10B10Srgb:
    case A2R10G10B10Bt2020Pq:
        return vk::Format::eA2B10G10R10UnormPack32;
    default:
        UNREACHABLE_MSG("display buffer format {:#x}", u32(format));
    }
}

/// Render targets have one level: its layout, all slices.
void LayOutTarget(ImageInfo& info, u32 tiled_slice_bytes, u32 tiled_height) {
    if (info.props.is_tiled) {
        info.guest_size = tiled_slice_bytes * info.resources.layers;
        info.mips_layout[0] = {info.guest_size, info.pitch, tiled_height, 0};
        return;
    }
    const auto level =
        LinearAlignedSize(info.pitch, info.size.height, info.num_bits, info.num_samples);
    info.guest_size = u32(level.bytes) * info.resources.layers;
    info.mips_layout[0] = {info.guest_size, info.pitch, info.size.height, 0};
}

} // namespace

ImageInfo::ImageInfo(const BbVideoOut::BufferAttributeGroup& group, VAddr cpu_address) noexcept {
    const auto& attrib = group.attrib;
    props.is_tiled = attrib.tiling_mode == BbVideoOut::TilingMode::Tile;
    tile_mode = props.is_tiled ? Gcn::TileMode::Display2DThin : Gcn::TileMode::DisplayLinearAligned;
    array_mode = Gcn::GetArrayMode(tile_mode);
    pixel_format = DisplayFormat(attrib.pixel_format);
    type = Gcn::ImageType::Color2D;
    size.width = attrib.width;
    size.height = attrib.height;
    // Rows of non-linear display buffers are 128 texels wide multiples.
    pitch = attrib.tiling_mode == BbVideoOut::TilingMode::Linear ? size.width
                                                                 : (size.width + 127) & ~127u;
    num_bits = attrib.pixel_format == BbVideoOut::PixelFormat::A16R16G16B16Float ? 64 : 32;
    ASSERT_MSG(num_bits == 32, "64-bit display buffers");
    guest_address = cpu_address;
    UpdateSize();
}

ImageInfo::ImageInfo(const Gcn::ColorBuffer& buffer, Gcn::CbDbExtent hint) noexcept {
    props.is_tiled = buffer.IsTiled();
    tile_mode = buffer.GetTileMode();
    array_mode = Gcn::GetArrayMode(tile_mode);
    pixel_format = SurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
    num_samples = buffer.NumSamples();
    num_bits = Gcn::NumBitsPerBlock(buffer.GetDataFmt());
    type = Gcn::ImageType::Color2D;
    pitch = buffer.Pitch();
    size = {hint.Valid() ? hint.width : pitch, hint.Valid() ? hint.height : buffer.Height(), 1};
    resources.layers = buffer.NumSlices();
    meta_info.cmask_addr = buffer.info.fast_clear ? buffer.CmaskAddress() : 0;
    meta_info.fmask_addr = buffer.info.compression ? buffer.FmaskAddress() : 0;
    guest_address = buffer.Address();
    LayOutTarget(*this, buffer.GetColorSliceSize(), buffer.Height());
    alt_tile = Libraries::Kernel::sceKernelIsNeoMode() && buffer.info.alt_tile_mode;
}

ImageInfo::ImageInfo(const Gcn::DepthBuffer& buffer, u32 num_slices, VAddr htile_address,
                     Gcn::CbDbExtent hint, bool write_buffer) noexcept {
    tile_mode = buffer.GetTileMode();
    array_mode = Gcn::GetArrayMode(tile_mode);
    pixel_format = DepthFormat(buffer.z_info.format, buffer.stencil_info.format);
    type = Gcn::ImageType::Color2D;
    props.is_tiled = buffer.IsTiled();
    props.is_depth = true;
    props.has_stencil = buffer.stencil_info.format != Gcn::DepthBuffer::StencilFormat::Invalid;
    num_samples = buffer.NumSamples();
    num_bits = buffer.NumBits();
    pitch = buffer.Pitch();
    size = {hint.Valid() ? hint.width : pitch, hint.Valid() ? hint.height : buffer.Height(), 1};
    resources.layers = num_slices;
    meta_info.htile_addr = buffer.z_info.tile_surface_enable ? htile_address : 0;
    stencil_addr = write_buffer ? buffer.StencilWriteAddress() : buffer.StencilAddress();
    stencil_size = pitch * size.height; // one byte per texel
    guest_address = write_buffer ? buffer.DepthWriteAddress() : buffer.DepthAddress();
    LayOutTarget(*this, buffer.GetDepthSliceSize(), buffer.Height());
}

ImageInfo::ImageInfo(const Gcn::Image& image, const Shader::ImageResource& desc) noexcept {
    tile_mode = image.GetTileMode();
    array_mode = Gcn::GetArrayMode(tile_mode);
    pixel_format = SurfaceFormat(image.GetDataFmt(), image.GetNumberFmt());
    if (desc.is_depth) {
        pixel_format = PromoteFormatToDepth(pixel_format);
        props.is_depth = true;
    }
    type = image.GetBaseType();
    props.is_tiled = image.IsTiled();
    props.is_volume = type == Gcn::ImageType::Color3D;
    props.is_pow2 = image.pow2pad;
    props.is_block = Gcn::IsBlockCoded(image.GetDataFmt());
    size = {u32(image.width) + 1, u32(image.height) + 1,
            props.is_volume ? u32(image.depth) + 1 : 1};
    pitch = image.Pitch();
    resources = {image.NumLevels(), image.NumLayers()};
    num_samples = image.NumSamples();
    num_bits = Gcn::NumBitsPerBlock(image.GetDataFmt());
    bank_swizzle = image.GetBankSwizzle();
    guest_address = image.Address();
    alt_tile = Libraries::Kernel::sceKernelIsNeoMode() && image.alt_tile_mode;
    UpdateSize();
}

} // namespace VideoCore

namespace VideoCore {

ImageInfo ImageInfo::Reduced(u32 skip) const {
    ImageInfo out = *this;
    out.size.width = std::max(size.width >> skip, 1u);
    out.size.height = std::max(size.height >> skip, 1u);
    out.pitch = std::max(pitch >> skip, 1u);
    out.resources.levels = resources.levels - skip;
    for (u32 level = 0; level < out.resources.levels; ++level) {
        out.mips_layout[level] = mips_layout[level + skip];
    }
    out.micro_tiled_mips = micro_tiled_mips >> skip;
    out.skip_levels = u8(skip);
    return out;
}

} // namespace VideoCore
