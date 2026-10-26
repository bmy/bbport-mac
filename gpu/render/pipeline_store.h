// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: the pipeline cache on disk (render/blob_store.h). Every translated permutation
// is stored as its SPIR-V ("<program hash>_<index>") and its metadata (the specialization and
// Info, by permutation hash); every pipeline as its key and what recreating it needs. At start
// the pipelines are preloaded (PipelineCache::WarmUp). Entries written by another build or for
// another GPU, or damaged, make the whole cache be rebuilt.
#pragma once

#include <vector>

#include "common/types.h"
#include "render/pipeline.h"

namespace Shader {
struct Info;
struct StageSpecialization;
} // namespace Shader

namespace Vulkan::PipelineStore {

/// Increment the one whose layout changes.
constexpr u32 ShaderBinaryVersion = 20u; // the UI edge snap: corners, side edges only low (subtitles)
constexpr u32 ShaderMetaVersion = 8u;    // SwTessControlRuntimeInfo::max_tess_level
constexpr u32 PipelineKeyVersion = 5u;   // Info layout (ImageResource::needs_native)

void SaveCompute(const ComputePipelineKey& key, const ComputePipeline::SerializationSupport& sdata);
void SaveGraphics(const GraphicsPipelineKey& key, u64 hash,
                  const GraphicsPipeline::SerializationSupport& sdata);
void SaveShaderMeta(const Shader::Info& info, const Shader::StageSpecialization& spec,
                    u64 perm_hash, size_t perm_idx);
void SaveShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx);

} // namespace Vulkan::PipelineStore
