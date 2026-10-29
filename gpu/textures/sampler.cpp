// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "textures/sampler.h"

#include <algorithm>
#include <array>

#include "common/logging/log.h"
#include "render/gcn_to_vk.h"
#include "vulkan/instance.h"

namespace VideoCore {

Sampler::Sampler(const Vulkan::Instance& instance, const Gcn::Sampler& sampler,
                 Gcn::BorderColorBuffer border_color_base, bool is_depth, float extra_lod_bias) {
    namespace ToVk = Vulkan::GcnToVk;
    const bool anisotropic = instance.IsAnisotropicFilteringSupported() &&
                             (Gcn::IsAnisoFilter(sampler.xy_mag_filter) ||
                              Gcn::IsAnisoFilter(sampler.xy_min_filter));
    vk::BorderColor border = ToVk::BorderColor(sampler.border_color_type);
    if (border == vk::BorderColor::eFloatCustomEXT && !instance.IsCustomBorderColorSupported()) {
        LOG_WARNING(Render_Vulkan, "no custom border colors: opaque black instead");
        border = vk::BorderColor::eFloatOpaqueBlack;
    }
    // Custom border colors: four floats each in the table TA_BC_BASE points at.
    vk::SamplerCustomBorderColorCreateInfoEXT custom{};
    if (border == vk::BorderColor::eFloatCustomEXT) {
        const auto* table = border_color_base.Address<const std::array<float, 4>*>();
        custom.customBorderColor = vk::ClearColorValue{table[sampler.border_color_ptr]};
        custom.format = vk::Format::eR32G32B32A32Sfloat;
    }
    const float max_bias = instance.MaxSamplerLodBias();
    const vk::SamplerCreateInfo create_info = {
        .pNext = border == vk::BorderColor::eFloatCustomEXT ? &custom : nullptr,
        .magFilter = ToVk::Filter(sampler.xy_mag_filter),
        .minFilter = ToVk::Filter(sampler.xy_min_filter),
        .mipmapMode = ToVk::MipFilter(sampler.mip_filter),
        .addressModeU = ToVk::ClampMode(sampler.clamp_x),
        .addressModeV = ToVk::ClampMode(sampler.clamp_y),
        .addressModeW = ToVk::ClampMode(sampler.clamp_z),
        .mipLodBias = std::clamp(sampler.LodBias() + extra_lod_bias, -max_bias, max_bias),
        .anisotropyEnable = anisotropic,
        .maxAnisotropy =
            anisotropic ? std::clamp(sampler.MaxAniso(), 1.0f, instance.MaxSamplerAnisotropy())
                        : 1.0f,
        .compareEnable = is_depth,
        .compareOp = ToVk::DepthCompare(sampler.depth_compare_func),
        .minLod = sampler.MinLod(),
        .maxLod = sampler.MaxLod(),
        .borderColor = border,
        .unnormalizedCoordinates = false, // the shaders scale unnormalized coordinates
    };
    auto [result, created] = instance.GetDevice().createSamplerUnique(create_info);
    ASSERT_MSG(result == vk::Result::eSuccess, "sampler: {}", vk::to_string(result));
    handle = std::move(created);
}

Sampler::~Sampler() = default;

} // namespace VideoCore
