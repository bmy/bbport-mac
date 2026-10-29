// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: SMAA 1x (third_party/smaa, MIT) for anti_aliasing=smaa without an upscaler: run on the
// game's finished LDR scene (RGBA8, gamma space) before the UI draws over it. The game's own AA
// is off then (BbSettings::SyncAntiAliasing).

#pragma once

#include "common/types.h"
#include "vulkan/vk_common.h"
#include "textures/image.h"

namespace Vulkan {

class Instance;
class Scheduler;

class Smaa {
public:
    Smaa(const Instance& instance, Scheduler& scheduler);
    ~Smaa();

    /// Records the three passes over `color` (sampled in `color_layout` through `color_view`, a
    /// UNORM view) of `width` x `height`. The result is Output(), RGBA8 in the GENERAL layout,
    /// readable by any later command. False: nothing recorded.
    bool Record(vk::ImageView color_view, vk::ImageLayout color_layout, u32 width, u32 height);

    [[nodiscard]] vk::Image Output() const {
        return vk::Image(output_image);
    }

private:
    bool EnsureImages(u32 width, u32 height);
    void UploadLookupTextures();

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueSampler linear_sampler;
    vk::UniqueSampler point_sampler;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline edges_pipeline;
    vk::UniquePipeline weights_pipeline;
    vk::UniquePipeline blend_pipeline;
    VideoCore::HostImage area_image, search_image; ///< SMAA's precomputed lookup textures
    vk::UniqueImageView area_view, search_view;
    bool uploaded = false;
    VideoCore::HostImage edges_image, weights_image, output_image;
    vk::UniqueImageView edges_view, weights_view, output_view;
    u32 image_width = 0, image_height = 0;
};

} // namespace Vulkan
