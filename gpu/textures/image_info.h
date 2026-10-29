// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: what an image is in guest memory and on the host: its format and size, its
// mip levels and layers, and where each level lies in the guest's memory (tiling).
#pragma once

#include <array>

#include "common/types.h"
#include "gcn/regs_targets.h"
#include "gcn/sharps.h"
#include "gcn/tiling.h"
#include "textures/types.h"
#include "vulkan/vk_common.h"

namespace BbVideoOut {
struct BufferAttributeGroup;
}

namespace Shader {
struct ImageResource;
}

namespace VideoCore {

struct ImageProperties {
    u32 is_volume : 1;
    u32 is_tiled : 1;
    u32 is_pow2 : 1;   ///< levels and slices padded to powers of two
    u32 is_block : 1;  ///< block compressed (4x4 texels per block)
    u32 is_depth : 1;
    u32 has_stencil : 1;
};

struct ImageInfo {
    ImageInfo() = default;
    /// A display buffer.
    ImageInfo(const BbVideoOut::BufferAttributeGroup& group, VAddr cpu_address) noexcept;
    /// A color target; `hint`: its size the driver gave (else the pitch and height).
    ImageInfo(const Gcn::ColorBuffer& buffer, Gcn::CbDbExtent hint) noexcept;
    /// A depth target (`write_buffer`: its write addresses, else the read ones).
    ImageInfo(const Gcn::DepthBuffer& buffer, u32 num_slices, VAddr htile_address,
              Gcn::CbDbExtent hint, bool write_buffer = false) noexcept;
    /// A texture or storage image.
    ImageInfo(const Gcn::Image& image, const Shader::ImageResource& desc) noexcept;

    bool IsTiled() const {
        return tile_mode != Gcn::TileMode::DisplayLinearAligned;
    }
    /// Pitch and height in blocks.
    Extent2D BlockDim() const {
        const u32 shift = props.is_block ? 2 : 0;
        return {pitch >> shift, size.height >> shift};
    }

    /// Whether views of one can show the other: compatible formats, the same samples and bits.
    bool IsCompatible(const ImageInfo& info) const;
    /// The level of `info` this image is (one level here), or -1.
    s32 MipOf(const ImageInfo& info) const;
    /// The layer of `info`'s level `mip` this image is, or -1.
    s32 SliceOf(const ImageInfo& info, s32 mip) const;
    /// Lays out the levels (mips_layout) and computes guest_size.
    void UpdateSize();
    /// The image without its `skip` largest levels (texture_quality): level 0 is level `skip` of
    /// the guest's; the guest address and size stay (the detiler takes the full description).
    ImageInfo Reduced(u32 skip) const;

    struct {
        VAddr cmask_addr;
        VAddr fmask_addr;
        VAddr htile_addr;
        s32 htile_clear_mask = -1;
    } meta_info{};

    ImageProperties props{};
    vk::Format pixel_format = vk::Format::eUndefined;
    Gcn::ImageType type{};
    SubresourceExtent resources;
    Extent3D size{1, 1, 1};
    u32 num_bits{};
    u32 num_samples = 1;
    u32 pitch{};
    Gcn::TileMode tile_mode = Gcn::TileMode::DisplayLinearAligned;
    Gcn::ArrayMode array_mode = Gcn::ArrayMode::ArrayLinearAligned;
    /// Where a level lies: bytes of all its slices, padded pitch and height, offset in the image.
    struct MipInfo {
        u32 size;
        u32 pitch;
        u32 height;
        u32 offset;
    };
    std::array<MipInfo, 16> mips_layout{};
    u32 micro_tiled_mips{}; ///< macro tiled images: the levels stored micro tiled (bits)
    VAddr guest_address{};
    u32 guest_size{};
    u8 bank_swizzle{};
    bool alt_tile{}; ///< the PS4 Pro's layout
    u8 skip_levels{}; ///< Reduced: the guest's levels left out

    VAddr stencil_addr{};
    u32 stencil_size{};
};

} // namespace VideoCore
