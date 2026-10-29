// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: GCN state and formats as Vulkan sees them: blend, depth, stencil and raster
// state, sampler fields, surface and depth formats (with what each can be used for), clear colors
// of color targets, sample counts.
#pragma once

#include <span>

#include "common/assert.h"
#include "gcn/formats.h"
#include "gcn/regs_geometry.h"
#include "gcn/regs_raster.h"
#include "gcn/regs_targets.h"
#include "gcn/sharps.h"
#include "vulkan/vk_common.h"

namespace Vulkan::GcnToVk {

vk::StencilOp StencilOp(Gcn::StencilFunc op);
vk::CompareOp CompareOp(Gcn::CompareFunc func);
/// Whether primitives of the type have a face to cull.
bool IsPrimitiveCulled(Gcn::PrimitiveType type);
vk::PrimitiveTopology PrimitiveType(Gcn::PrimitiveType type);
vk::PolygonMode PolygonMode(Gcn::PolygonMode mode);
vk::CullModeFlags CullMode(Gcn::CullMode mode);
vk::FrontFace FrontFace(Gcn::FrontFace face);
vk::BlendFactor BlendFactor(Gcn::BlendControl::BlendFactor factor);
bool IsDualSourceBlendFactor(Gcn::BlendControl::BlendFactor factor);
vk::BlendOp BlendOp(Gcn::BlendControl::BlendFunc func);
vk::LogicOp LogicOp(Gcn::ColorControl::LogicOp op);
vk::SamplerAddressMode ClampMode(Gcn::ClampMode mode);
vk::CompareOp DepthCompare(Gcn::DepthCompare comp);
vk::Filter Filter(Gcn::Filter filter);
vk::SamplerReductionMode FilterMode(Gcn::FilterMode mode);
vk::SamplerMipmapMode MipFilter(Gcn::MipFilter filter);
vk::BorderColor BorderColor(Gcn::BorderColor color);
vk::ComponentSwizzle ComponentSwizzle(Gcn::CompSwizzle swizzle);
vk::ComponentMapping ComponentMapping(Gcn::CompMapping mapping);

/// A data and number format pair with its Vulkan format and the uses GCN allows it.
struct SurfaceFormatInfo {
    Gcn::DataFormat data_format;
    Gcn::NumberFormat number_format;
    vk::Format vk_format;
    vk::FormatFeatureFlags2 flags;
};
std::span<const SurfaceFormatInfo> SurfaceFormats();
/// The Vulkan format (pairs without one stop the port, but the invalid data format).
vk::Format SurfaceFormat(Gcn::DataFormat data_format, Gcn::NumberFormat num_format);
/// eUndefined for pairs without one (speculative readers: the draw preparation workers).
vk::Format TrySurfaceFormat(Gcn::DataFormat data_format, Gcn::NumberFormat num_format);

struct DepthFormatInfo {
    Gcn::DepthBuffer::ZFormat z_format;
    Gcn::DepthBuffer::StencilFormat stencil_format;
    vk::Format vk_format;
    vk::FormatFeatureFlags2 flags;
};
std::span<const DepthFormatInfo> DepthFormats();
vk::Format DepthFormat(Gcn::DepthBuffer::ZFormat z_format,
                       Gcn::DepthBuffer::StencilFormat stencil_format);

/// A color target's clear color (CB_COLOR*_CLEAR_WORD0/1), in its component order.
vk::ClearValue ColorBufferClearValue(const Gcn::ColorBuffer& color_buffer);

/// The sample count, halved until the device supports it (at least one).
vk::SampleCountFlagBits NumSamples(u32 num_samples, vk::SampleCountFlags supported);

/// Formats a T# names for depth images (R32, R16): sampling the depth aspect.
inline bool IsFormatDepthCompatible(vk::Format format) {
    return format == vk::Format::eD32Sfloat || format == vk::Format::eR32Sfloat ||
           format == vk::Format::eR32Uint || format == vk::Format::eD16Unorm ||
           format == vk::Format::eR16Unorm;
}
/// Formats a T# names for the stencil aspect (8-bit).
inline bool IsFormatStencilCompatible(vk::Format format) {
    return format == vk::Format::eS8Uint || format == vk::Format::eR8Uint ||
           format == vk::Format::eR8Unorm;
}
/// The depth format a color format of a depth texture stands for.
inline vk::Format PromoteFormatToDepth(vk::Format format) {
    switch (format) {
    case vk::Format::eR32Sfloat:
    case vk::Format::eR32Uint:
        return vk::Format::eD32Sfloat;
    case vk::Format::eR16Unorm:
        return vk::Format::eD16Unorm;
    default:
        UNREACHABLE_MSG("no depth format for {}", vk::to_string(format));
    }
}

} // namespace Vulkan::GcnToVk
