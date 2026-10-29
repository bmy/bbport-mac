// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: copies between images that Vulkan cannot copy directly, drawn with a
// full-screen triangle: a color image holding one sample per channel into a multisampled depth
// image, and between images of different sample counts.
#pragma once

#include <vector>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Scheduler;
} // namespace Vulkan

namespace VideoCore {

class ImageBlitter {
public:
    ImageBlitter(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler);
    ~ImageBlitter();

    /// Level 0, layer 0 of `source` (color, `src_format`) into depth image `dest` with
    /// `num_samples`: sample N from channel N.
    void ColorAsMsDepth(u32 width, u32 height, u32 num_samples, vk::Format src_format,
                        vk::Format dst_format, vk::Image source, vk::Image dest);
    /// Level 0, layer 0 of `source` (multisampled when `src_msaa`) into `dest` with
    /// `num_samples`, texel for texel (and sample for sample).
    void CopyMsImage(u32 width, u32 height, u32 num_samples, vk::Format format, bool src_msaa,
                     vk::Image source, vk::Image dest);

private:
    enum class Kind : u32 { DepthFromColor, Copy, CopyMsaa };
    struct Pipeline {
        Kind kind;
        u32 num_samples;
        vk::Format format; ///< of the attachment
        vk::UniquePipeline pipeline;
    };

    vk::Pipeline GetPipeline(Kind kind, u32 num_samples, vk::Format format);
    vk::ImageView View(vk::Image image, vk::Format format, vk::ImageAspectFlags aspect,
                       vk::ImageUsageFlags usage);
    /// Draws the triangle sampling `source_view` into the rendering begun by the caller.
    void Draw(vk::Pipeline pipeline, vk::ImageView source_view, u32 width, u32 height);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueShaderModule vertex;
    vk::UniqueShaderModule depth_from_color;
    vk::UniqueShaderModule copy;
    vk::UniqueShaderModule copy_msaa;
    std::vector<Pipeline> pipelines;
};

} // namespace VideoCore
