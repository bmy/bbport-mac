// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: translating the pipeline cache's recorded shader inputs again after an update
// that changed the shader compiler's output (shader_rebuild.cpp), before the game maps its memory.
// The same inputs without the game's code are the shader recipes shipped with the port: a new
// cache gets the shaders of the whole recorded game translated in the background at the first
// start, the code taken from the player's own files (docs/shader_recipes.md).
#pragma once

#include <filesystem>

#include "common/types.h"

namespace BbShaderRebuild {

/// Rebuilds every game cache under the user's cache directory written by an earlier compiler
/// (same metadata and pipeline key layouts); the number rebuilt. Linux only (0 elsewhere).
int RebuildCaches();
/// One cache directory translated again (RebuildCaches' child): 0 when it all went through.
int RebuildOne(const std::filesystem::path& dir);

/// Writes the recipe pack of `cache_dir`'s compilations (no program code: references to the
/// programs in `game_dir`, the game's dump); 0 on success.
int ExportRecipes(const char* cache_dir, const char* game_dir, const char* out_path);

/// Translates the pack's programs `cache_dir` does not have for this GPU (the cache's profile)
/// into `cache_dir`/import, merged by the pipeline cache (PipelineCache::MergeImport). In its own
/// process (bb-probe --import-shaders): the translation maps guest addresses. Linux only.
/// Shader input prediction experiment (Linux): each recorded program translated with every other
/// program's input, the specializations compared with its own (results in `out_dir`).
int PredictTest(const char* cache_dir, const char* pack, const char* game_dir, const char* out_dir,
                u32 jobs);
/// Another player's cache (metadata only, META_DIR) merged into a recipe pack (Linux): each
/// permutation gets the input of one the pack has that gives its specialization exactly.
int MergeMetas(const char* cache_dir, const char* pack, const char* game_dir, const char* meta_dir,
               const char* out_pack, const char* work_dir, u32 jobs);
/// Two recipe packs merged (Linux): the second's permutations matched to the first's by their
/// specialization, new ones appended, its pipeline keys renamed to the merged permutations.
int MergePacks(const char* cache_dir, const char* pack_a, const char* pack_b, const char* game_dir,
               const char* out_pack, const char* work_dir, u32 jobs);
int MetaReconstructTest(const char* cache_dir, const char* pack, const char* game_dir, u32 jobs);
int ImportRecipes(const char* cache_dir, const char* pack, const char* game_dir, u64 motion_params,
                  u64 motion_positions);

} // namespace BbShaderRebuild
