// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: FSR 4 (source-v07 INT8/DOT4 model, FSR-Vulkan provider) as an alternative to the
// FSR 3.1 context of TemporalUpscaler. It consumes the same inputs: HDR scene color, depth,
// motion vectors in render pixels (previous - current) and the jitter offset.
//
// The model is per preset (native, quality, balanced, performance, ultraperf) and per output
// tier; its SPIR-V passes and weights come from tools/fetch_fsr4_assets.sh (BB_FSR4_DIR overrides
// the fsr4_shaders directory). A preset change rebuilds the model graph.

#pragma once

#include <array>
#include <memory>
#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;
class Scheduler;

class Fsr4Upscaler {
public:
    Fsr4Upscaler(const Instance& instance, Scheduler& scheduler);
    ~Fsr4Upscaler();

    /// An image in layout General, written/read by compute shaders.
    struct Image {
        vk::Image image;
        vk::ImageView view;
        u32 width, height;
        /// The image's own format: AMD's DLLs make their views from it (upscale/vk_fsr_amd).
        vk::Format format = vk::Format::eUndefined;
    };
    struct Frame {
        vk::CommandBuffer cmdbuf;
        Image color, depth, motion, output;
        u32 render_width, render_height;
        int preset; ///< BbSettings::Preset
        std::array<float, 2> jitter;
        float frame_ms, near_plane, far_plane, vertical_fov;
        float sharpness;
        bool sharpen, reset, auto_exposure;
    };

    /// Records FSR 4 into `frame.cmdbuf`; the output stays in General. False when FSR 4
    /// cannot run (Problem() says why; missing assets or device features are permanent).
    bool Record(const Frame& frame);

    /// Whether the v07 model for `preset` and this output size is ready; if not, it is built
    /// on a background thread and this answers false until it is (the caller runs another
    /// upscaler meanwhile). True too when FSR 4 cannot run at all: Record() says why.
    bool Prepare(int preset, u32 output_width, u32 output_height);

    /// Drops the context and any permanent failure (a retry after a menu change); a model
    /// being built in the background is kept.
    void Reset();

    /// Why the last Record failed, or null.
    [[nodiscard]] const char* Problem() const noexcept;
    /// Assets or device features are missing: FSR 4 will not run in this session.
    [[nodiscard]] bool Fatal() const noexcept;
    /// AMD's FSR DLLs (upscaler fsr_amd) were found and loaded.
    [[nodiscard]] bool AmdAvailable() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
