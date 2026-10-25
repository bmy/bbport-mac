// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal pipelines the shadow build made (mtl_shadow.mm), by Vulkan pipeline
// handle, for the Metal command path (docs/MACOS_METAL_RENDERER.md, stage 3). From
// Wooloomooloo2/bloodborne_mac_port (78b59f5, GPL-2.0-or-later).

#pragma once

#include "vulkan/vk_common.h"

namespace Metal {
struct RenderPipeline;
struct ComputePipeline;
struct TessellationPipeline;
struct GeometryPipeline;
struct DepthStencilKey;
} // namespace Metal

namespace MetalShadow {

struct GraphicsEntry {
    const Metal::RenderPipeline* render{};     ///< null: no Metal pipeline (the draw is skipped)
    const Metal::TessellationPipeline* tess{}; ///< a tessellated draw's pipelines
    const Metal::GeometryPipeline* geometry{}; ///< a geometry-shader draw's pipelines
    bool rect_list{};                          ///< the vertex shader is the rect list rewrite
};

/// Null when the Vulkan pipeline has no Metal counterpart (yet).
const GraphicsEntry* FindGraphics(VkPipeline pipeline);
const Metal::ComputePipeline* FindCompute(VkPipeline pipeline);
/// id<MTLDepthStencilState> for the key (the pipeline cache's), null on failure.
void* DepthStencil(const Metal::DepthStencilKey& key);

} // namespace MetalShadow
