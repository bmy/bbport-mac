// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal pipelines the shadow build made (mtl_shadow.mm), by Vulkan pipeline
// handle, for the Metal command path (docs/MACOS_METAL_RENDERER.md, stage 3). From
// Wooloomooloo2/bloodborne_mac_port (78b59f5, GPL-2.0-or-later).

#pragma once

#include <array>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Metal {
struct RenderPipeline;
struct ComputePipeline;
struct TessellationPipeline;
struct GeometryPipeline;
struct DepthStencilKey;
} // namespace Metal

namespace MetalShadow {

/// State a generic (host pass) pipeline fixes at creation; the game's pipelines set it all
/// dynamically. A field applies where its flag (cull, front_face, depth, stencil, depth_bias) is
/// set: that state is not dynamic in the pipeline.
struct FixedState {
    bool cull{}, front_face{}, depth{}, stencil{}, depth_bias{};
    VkCullModeFlags cull_mode{};
    VkFrontFace front{VK_FRONT_FACE_COUNTER_CLOCKWISE};
    bool depth_test{}, depth_write{};
    VkCompareOp depth_compare{VK_COMPARE_OP_ALWAYS};
    bool stencil_test{};
    bool depth_bias_enable{};
};

struct GraphicsEntry {
    const Metal::RenderPipeline* render{};     ///< null: no Metal pipeline (the draw is skipped)
    const Metal::TessellationPipeline* tess{}; ///< a tessellated draw's pipelines
    const Metal::GeometryPipeline* geometry{}; ///< a geometry-shader draw's pipelines
    bool rect_list{};                          ///< the vertex shader is the rect list rewrite
    /// bbport: built from its Vulkan create info (the renderer's own passes, not the game's
    /// pipelines): its fixed state and vertex binding strides.
    bool generic{};
    FixedState fixed;
    std::array<u32, 32> strides{};
};

/// Null when the Vulkan pipeline has no Metal counterpart (yet).
const GraphicsEntry* FindGraphics(VkPipeline pipeline);
const Metal::ComputePipeline* FindCompute(VkPipeline pipeline);
/// id<MTLDepthStencilState> for the key (the pipeline cache's), null on failure.
void* DepthStencil(const Metal::DepthStencilKey& key);

/// bbport: the command layer's view of shader modules and pipelines (frame mode). Pipelines made
/// outside a GameBuildScope (the renderer's own passes: presentation, copies, upscaling, motion
/// vectors) are built in Metal from their Vulkan create infos, at creation, on the creating
/// thread; the game's come through Graphics/Compute with their keys.
void ModuleCreated(VkShaderModule module, const VkShaderModuleCreateInfo* info);
void GraphicsCreated(const VkGraphicsPipelineCreateInfo* info, VkPipeline pipeline);
void ComputeCreated(const VkComputePipelineCreateInfo* info, VkPipeline pipeline);
void PipelineDestroyed(VkPipeline pipeline);

} // namespace MetalShadow
