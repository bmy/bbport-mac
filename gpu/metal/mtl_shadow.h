// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal pipelines the shadow build made (mtl_shadow.mm), by Vulkan pipeline
// handle, for the Metal command path (docs/MACOS_METAL_RENDERER.md, stage 3). From
// Wooloomooloo2/bloodborne_mac_port (78b59f5, GPL-2.0-or-later).

#pragma once

#include <array>
#include <memory>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Metal {
struct RenderPipeline;
struct ComputePipeline;
struct TessellationPipeline;
struct GeometryPipeline;
struct DepthStencilKey;
namespace Geometry {
class IndirectKernels;
}
} // namespace Metal

namespace MetalShadow {

/// State a generic (host pass) pipeline fixes at creation; the game's pipelines set it all
/// dynamically. A field applies where its flag (cull, front_face, depth, stencil, depth_bias) is
/// set: that state is not dynamic in the pipeline.
struct FixedState {
    bool cull{}, front_face{}, depth{}, depth_write_fixed{}, depth_compare_fixed{}, stencil{},
        depth_bias{};
    VkCullModeFlags cull_mode{};
    VkFrontFace front{VK_FRONT_FACE_COUNTER_CLOCKWISE};
    bool depth_test{}, depth_write{};
    VkCompareOp depth_compare{VK_COMPARE_OP_ALWAYS};
    bool stencil_test{};
    bool depth_bias_enable{};
    /// The stencil faces, per part (VK_DYNAMIC_STATE_STENCIL_OP, _COMPARE_MASK, _WRITE_MASK,
    /// _REFERENCE).
    bool stencil_op{}, stencil_compare_mask{}, stencil_write_mask{}, stencil_reference{};
    struct Face {
        VkStencilOp fail{}, pass{}, depth_fail{};
        VkCompareOp compare{VK_COMPARE_OP_ALWAYS};
        u32 compare_mask{0xff}, write_mask{0xff}, reference{};
    } stencil_front, stencil_back;
    /// The depth bounds test and its bounds.
    bool depth_bounds{}, depth_bounds_test{}, depth_bounds_values{};
    float min_depth_bounds{0.0f}, max_depth_bounds{1.0f};
    /// The color write masks are the pipeline's (VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT absent).
    bool write_masks{};
    /// The blend constants (VK_DYNAMIC_STATE_BLEND_CONSTANTS absent).
    bool blend_constants_fixed{};
    float blend_constants[4]{};
    /// The vertex strides are the pipeline's (GraphicsEntry::strides), whatever is bound
    /// (VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE absent).
    bool strides{};
};

struct VariantSource; ///< what a plain game pipeline was built from (mtl_shadow.mm)

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
    /// bbport: the pipeline takes flat values from the last vertex (its key's provoking vertex)
    /// and its fragment shader has flat inputs: Metal takes them from the first, so its draws
    /// go to KosmicKrisp.
    bool flat_from_last{};
    /// bbport: the guest shaders' hashes (vertex, hull, domain, geometry, pixel; 0: none) as
    /// the renderer logs them (BB_METAL_GPU_DIAG's draw notes).
    std::array<u64, 5> hashes{};
    /// bbport: a plain game pipeline's build inputs, for its variants with a pass's depth and
    /// stencil formats (FormatVariant); null for the others.
    std::shared_ptr<const VariantSource> source;
};

/// bbport: the entry's render pipeline built for a pass's depth and stencil formats
/// (MTLPixelFormat, 0: none): ready, or null with `building` set while it is being built (posted
/// on first use) or with `building` clear when it cannot be built.
const Metal::RenderPipeline* FormatVariant(const GraphicsEntry& entry, u32 depth, u32 stencil,
                                           bool& building);

/// Null when the Vulkan pipeline has no Metal counterpart (yet).
const GraphicsEntry* FindGraphics(VkPipeline pipeline);
/// bbport: as FindGraphics, but a tessellated (`tess`: those draws go to Metal) or geometry-shader
/// (`geometry`) game pipeline whose Metal build is still pending is waited for: until
/// `timeout_ms` after the first wait for it (later waits share that deadline); its build is
/// queued first. Otherwise its draws would go to KosmicKrisp's emulation of those stages.
const GraphicsEntry* FindGraphicsWaiting(VkPipeline pipeline, u32 timeout_ms, bool tess,
                                         bool geometry);
const Metal::ComputePipeline* FindCompute(VkPipeline pipeline);
/// bbport: why a pipeline has no Metal counterpart now, for the fallback counts (a static
/// string): still being built, its build gave none, built after its bind, or never given.
const char* WhyMissing(VkPipeline pipeline, bool compute);
/// bbport: the prep and index-copy kernels of indirect geometry-shader draws, built on first
/// use; null when they could not be (logged once).
const Metal::Geometry::IndirectKernels* GeometryIndirectKernels();
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
