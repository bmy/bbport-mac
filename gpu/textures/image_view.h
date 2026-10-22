// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: how a binding sees an image: view type, format, levels and layers, swizzle.
#pragma once

#include "gcn/regs_targets.h"
#include "gcn/sharps.h"
#include "textures/types.h"
#include "vulkan/vk_common.h"

namespace Shader {
struct ImageResource;
}

namespace Vulkan {
class Instance;
}

namespace VideoCore {

struct ImageViewInfo {
    ImageViewInfo() = default;
    /// A texture or storage image binding.
    ImageViewInfo(const Gcn::Image& image, const Shader::ImageResource& desc) noexcept;
    /// A color target.
    explicit ImageViewInfo(const Gcn::ColorBuffer& buffer) noexcept;
    /// A depth target.
    ImageViewInfo(const Gcn::DepthBuffer& buffer, Gcn::DepthView view, Gcn::DepthControl control);

    Gcn::ImageType type = Gcn::ImageType::Color2D;
    vk::Format format = vk::Format::eR8G8B8A8Unorm;
    SubresourceRange range;
    vk::ComponentMapping mapping{};
    u32 min_lod = 0; ///< 4.8 fixed point
    bool is_storage = false;

    auto operator<=>(const ImageViewInfo&) const = default;
};

struct Image;

/// Whether an image made in `image` format without mutable formats can have a view in `view`
/// format: the same format, or its sRGB or linear twin (KosmicKrisp: Metal needs no pixel
/// format view usage for those; Instance::IsCompressibleImages is only on there).
[[nodiscard]] bool ViewableWithoutMutableFormat(vk::Format view, vk::Format image);

struct ImageView {
    ImageView(const Vulkan::Instance& instance, const ImageViewInfo& info, const Image& image);
    ~ImageView();
    ImageView(const ImageView&) = delete;
    ImageView& operator=(const ImageView&) = delete;
    ImageView(ImageView&&) = default;
    ImageView& operator=(ImageView&&) = default;

    ImageViewInfo info;
    vk::UniqueImageView image_view;
};

} // namespace VideoCore
