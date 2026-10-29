// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: GT-VBAO (ground-truth AO with a visibility bitmask, Therrien et al. 2023) in place of the
// game's SSAO. The game keeps its own AO passes; after the last one the AO target is overwritten
// with VBAO computed from the scene depth, so the game's lighting uses it as its own. The game
// has two SSAO paths (docs/vbao.md): a temporal one ending in an upsample into an R8 target at
// the scene size (pixel shader c45c51cb), and a cheaper one whose half-size RGBA8 target is read
// directly (pixel shader cbbca5dc; the AO is its red channel).

#pragma once

#include "common/types.h"
#include "vulkan/vk_common.h"
#include "textures/image.h"

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class Instance;
class Scheduler;
class Runtime;
class CameraMotion;

class Vbao {
public:
    /// Pixel shader of the game's AO upsample pass: its R8 color target is the final AO.
    static constexpr u64 AoUpsampleShader = 0xc45c51cb;
    /// Pixel shader of the game's half-size SSAO without history: the AO is the red channel of
    /// its RGBA8 target.
    static constexpr u64 AoHalfShader = 0xcbbca5dc;

    Vbao(const Instance& instance, Scheduler& scheduler, VideoCore::TextureCache& texture_cache,
         Runtime& runtime, const CameraMotion& camera);
    ~Vbao();

    /// The setting is on and the camera of the scene is known.
    [[nodiscard]] bool Enabled() const;

    /// The last pass of one of the game's AO paths drew `target`: overwrites the AO with VBAO
    /// when Enabled.
    /// `temporal`: a temporal upscaler accumulates the frame, the sample pattern may change.
    /// `normals`: the G-buffer normals the game's SSAO reads (world space, unorm), or null.
    void OnAoPass(VideoCore::ImageId target, VideoCore::ImageId normals, bool temporal);

    /// The pass copying the finished frame to a display buffer: the AO-only debug view
    /// (debug_view = DebugAo) draws this frame's AO target over `frame`.
    void OnDisplayPass(VideoCore::ImageId frame);

    /// A scaled upscaler wrote `output` (RGBA8, shown instead of the game's frame): the AO
    /// view draws there.
    void OnUpscaled(vk::Image output, u32 width, u32 height);

    /// The pixel shader of a draw that ends one of the game's AO paths.
    [[nodiscard]] static bool AoShader(u64 ps_hash) noexcept {
        return ps_hash == AoUpsampleShader || ps_hash == AoHalfShader;
    }

private:
    /// Draws the AO target `ao_id` as grey into `output`; `guest`: output is that texture cache
    /// image (its barriers go through the runtime), else an image outside the cache.
    void DrawView(VideoCore::ImageId ao_id, vk::Image output, u32 width, u32 height,
                  VideoCore::Image* guest);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    const CameraMotion& camera;
    vk::UniqueDescriptorSetLayout desc_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline pipeline;           ///< the raw AO and view depth
    vk::UniquePipeline blur_pipeline;      ///< denoise into R8 at the scene size
    vk::UniquePipeline blur_rgba_pipeline; ///< into the red channel of the half-size RGBA8
    VideoCore::HostImage raw_image;      ///< RGBA16F: raw AO, view z, normal xy
    vk::UniqueImageView raw_view;
    u32 raw_width = 0, raw_height = 0;
    vk::UniquePipeline view_pipeline; ///< the AO-only debug view
    VideoCore::ImageId frame_ao;      ///< this frame's AO target, for the debug view
    u32 frame_ao_width = 0, frame_ao_height = 0;
    bool view_drawn = false; ///< the AO view went into the upscaler's output this frame
    u32 frame = 0;
};

} // namespace Vulkan
