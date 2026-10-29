// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
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
#ifdef __APPLE__
constexpr u32 ShaderBinaryVersion = 25u; // bbport: the page table's two halves on macOS
#else
constexpr u32 ShaderBinaryVersion = 24u; // paged records read once, a lookup per page (PagedDwords)
#endif
constexpr u32 ShaderMetaVersion = 8u;    // the object motion buffer addresses
constexpr u32 PipelineKeyVersion = 5u;   // Info layout (ImageResource::needs_native)

void SaveCompute(const ComputePipelineKey& key, const ComputePipeline::SerializationSupport& sdata);
void SaveGraphics(const GraphicsPipelineKey& key, u64 hash,
                  const GraphicsPipeline::SerializationSupport& sdata);
void SaveShaderMeta(const Shader::Info& info, const Shader::StageSpecialization& spec,
                    u64 perm_hash, size_t perm_idx);
void SaveShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx);
/// The inputs of a permutation's compilation (shader_capture.h TakeInput).
void SaveShaderInput(std::vector<u8>&& input, u64 pgm_hash, size_t perm_idx);

} // namespace Vulkan::PipelineStore
