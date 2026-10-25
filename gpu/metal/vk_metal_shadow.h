// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal shadow build (docs/MACOS_METAL_RENDERER.md, stage 2). With
// BB_METAL_SHADOW=1 in the arm64 GPU host, every pipeline the Vulkan path creates is also built as
// a Metal pipeline state (metal/mtl_shadow.mm) and the results are counted in the log; nothing
// draws with them. Without BB_METAL_BACKEND (every other build) these are no-ops. From
// Wooloomooloo2/bloodborne_mac_port (78b59f5, GPL-2.0-or-later).

#pragma once

#include <span>
#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {
struct GraphicsPipelineKey;
struct ComputePipelineKey;
} // namespace Vulkan

namespace MetalShadow {

/// One vertex attribute as the Vulkan pipeline declares it (GraphicsPipeline::GetVertexInputs).
struct VertexInput {
    u32 location{};
    bool per_instance{};
    u32 divisor{1};
};

#ifdef BB_METAL_BACKEND
bool Enabled();
/// The SPIR-V a Vulkan shader module was made from (CompileSPV).
void RegisterModule(vk::ShaderModule module, std::span<const u32> spirv);
/// A graphics pipeline the Vulkan path built; `modules`/`has_stage` are indexed by SwStage,
/// `blend` holds its color attachments' blend states. `pipeline`: the Vulkan pipeline, so the
/// Metal command path finds the Metal one for it.
void Graphics(const Vulkan::GraphicsPipelineKey& key, std::span<const vk::ShaderModule> modules,
              std::span<const bool> has_stage, std::span<const VertexInput> inputs,
              std::span<const vk::PipelineColorBlendAttachmentState> blend, bool aux_fragment,
              vk::Pipeline pipeline);
void Compute(const Vulkan::ComputePipelineKey& key, vk::ShaderModule module, vk::Pipeline pipeline);
#else
inline bool Enabled() {
    return false;
}
inline void RegisterModule(vk::ShaderModule, std::span<const u32>) {}
inline void Graphics(const Vulkan::GraphicsPipelineKey&, std::span<const vk::ShaderModule>,
                     std::span<const bool>, std::span<const VertexInput>,
                     std::span<const vk::PipelineColorBlendAttachmentState>, bool, vk::Pipeline) {}
inline void Compute(const Vulkan::ComputePipelineKey&, vk::ShaderModule, vk::Pipeline) {}
#endif

} // namespace MetalShadow
