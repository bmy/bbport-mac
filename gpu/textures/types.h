// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: the small types of the texture cache.
#pragma once

#include <compare>

#include "common/slot_vector.h"
#include "common/types.h"

namespace VideoCore {

using ImageId = Common::SlotId;
using ImageViewId = Common::SlotId;

struct Extent2D {
    u32 width;
    u32 height;

    bool operator==(const Extent2D&) const = default;
};

struct Extent3D {
    u32 width;
    u32 height;
    u32 depth;

    bool operator==(const Extent3D&) const = default;
};

/// The first mip level and array layer of a part of an image.
struct SubresourceBase {
    u32 level = 0;
    u32 layer = 0;

    auto operator<=>(const SubresourceBase&) const = default;
};

/// How many mip levels and array layers.
struct SubresourceExtent {
    u32 levels = 1;
    u32 layers = 1;

    auto operator<=>(const SubresourceExtent&) const = default;
};

struct SubresourceRange {
    SubresourceBase base;
    SubresourceExtent extent;

    auto operator<=>(const SubresourceRange&) const = default;
};

} // namespace VideoCore
