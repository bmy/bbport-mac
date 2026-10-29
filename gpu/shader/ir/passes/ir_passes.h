// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the IR passes, in the order the translator runs them (recompiler.cpp).
#pragma once

#include "shader/ir/basic_block.h"
#include "shader/ir/program.h"
#include "shader/ir/passes/resource_pass.h"

namespace Shader {
struct Profile;
/// NVIDIA's clip distance interpolation is broken: fragment shaders discard instead.
void InjectClipDistanceAttributes(IR::Program& program, RuntimeInfo& runtime_info);
} // namespace Shader

namespace Shader::Optimization {

void SsaRewritePass(IR::Program& program);
void SsaRepairPass(IR::Program& program);
void PhiSimplificationPass(IR::Program& program);
void InverseBallotEliminationPass(IR::Program& program);
void LowerWave64BallotPass(IR::Program& program, const RuntimeInfo& runtime_info,
                           const Profile& profile);
void LowerHardwareIntrinsics(IR::Program& program);
void LowerPhisToRegsPass(IR::Program& program);
void DeadCodeEliminationPass(IR::Program& program);
/// Exact integers from interpolated float varyings on GPUs without manual interpolation.
void InterpolatedIntegerPass(IR::Program& program);
void ConstantPropagationPass(IR::BlockList& program);
void FlattenExtendedUserdataPass(IR::Program& program);
void ReadLaneEliminationPass(IR::Program& program);
ResourceDiscoveryList ResourceDiscoverPass(IR::Program& program, const Profile& profile);
void ResourcePatchingPass(Shader::Info& info, const ResourceDiscoveryList& sharp_usages,
                          const Profile& profile);
void CollectShaderInfoPass(IR::Program& program, const Profile& profile);
void LowerBufferFormatToRaw(IR::Program& program);
void LowerUserClipPlanes(IR::Program& program, const RuntimeInfo& runtime_info);
void LowerFp64ToFp32(IR::Program& program);
void RingAccessElimination(const IR::Program& program, const RuntimeInfo& runtime_info);
void TessellationPreprocess(IR::Program& program, RuntimeInfo& runtime_info);
void HullShaderTransform(IR::Program& program, const RuntimeInfo& runtime_info);
void DomainShaderTransform(const IR::Program& program, const RuntimeInfo& runtime_info);
void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile);
void SharedMemorySimplifyPass(IR::Program& program, const Profile& profile);
void SharedMemoryToStoragePass(IR::Program& program, const RuntimeInfo& runtime_info,
                               const Profile& profile);

} // namespace Shader::Optimization
