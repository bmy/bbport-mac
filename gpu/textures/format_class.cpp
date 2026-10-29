// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "textures/format_class.h"

#include <cstring>

#include <vulkan/vulkan_format_traits.hpp>

namespace VideoCore {

bool IsVulkanFormatCompatible(vk::Format base, vk::Format view) {
    // An undefined view format takes any image; an undefined image has no views.
    if (base == view || view == vk::Format::eUndefined) {
        return true;
    }
    if (base == vk::Format::eUndefined) {
        return false;
    }
    const char* base_class = vk::compatibilityClass(base);
    const char* view_class = vk::compatibilityClass(view);
    if (base_class == view_class || std::strcmp(base_class, view_class) == 0) {
        return true;
    }
    // BC images also take uncompressed views of their block size (block-texel views).
    return std::strcmp(vk::compressionScheme(base), "BC") == 0 && !vk::isCompressed(view) &&
           vk::blockSize(base) == vk::blockSize(view);
}

} // namespace VideoCore
