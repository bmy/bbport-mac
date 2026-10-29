// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: a Vulkan sampler for an S#.
#pragma once

#include "gcn/regs_targets.h"
#include "gcn/sharps.h"
#include "vulkan/vk_common.h"

namespace Vulkan {
class Instance;
}

namespace VideoCore {

class Sampler {
public:
    /// `is_depth`: for depth comparisons (a compare sampler read without one is undefined in
    /// Vulkan); `extra_lod_bias`: added to the S#'s (reduced-size scene rendering).
    Sampler(const Vulkan::Instance& instance, const Gcn::Sampler& sampler,
            Gcn::BorderColorBuffer border_color_base, bool is_depth, float extra_lod_bias = 0.0f);
    ~Sampler();
    Sampler(const Sampler&) = delete;
    Sampler& operator=(const Sampler&) = delete;
    Sampler(Sampler&&) = default;
    Sampler& operator=(Sampler&&) = default;

    vk::Sampler Handle() const noexcept {
        return *handle;
    }

    size_t lru_id{};

private:
    vk::UniqueSampler handle;
};

} // namespace VideoCore
