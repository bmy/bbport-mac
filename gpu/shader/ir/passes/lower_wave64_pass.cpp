// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Compute shaders written for 64-lane waves on hosts with smaller subgroups (32): 64-bit ballots,
// reads of a fixed lane and the lane-below mask go through shared memory scratch (one dword per
// host subgroup). Only in control flow every invocation of the workgroup runs.
#include <algorithm>
#include <optional>
#include <vector>

#include "common/alignment.h"
#include "common/logging/log.h"
#include "shader/ir/breadth_first_search.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/info.h"
#include "shader/profile.h"

namespace Shader::Optimization {

namespace {

bool IsDivergent(const IR::U1& condition) {
    if (condition.IsImmediate()) {
        return false;
    }
    return IR::BreadthFirstSearch(
               condition.Inst(),
               [](const IR::Inst* inst) -> std::optional<bool> {
                   if (inst->GetOpcode() == IR::Opcode::LaneId ||
                       (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
                        inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId)) {
                       return true;
                   }
                   return std::nullopt;
               })
        .value_or(false);
}

/// The blocks outside divergent conditionals and outside loops (none if the structure is not
/// understood).
std::vector<IR::Block*> UniformBlocks(const IR::Program& program) {
    using Type = IR::AbstractSyntaxNode::Type;
    struct Conditional {
        const IR::Block* merge;
        bool divergent;
    };
    std::vector<IR::Block*> blocks;
    std::vector<Conditional> conditionals;
    std::vector<const IR::Block*> loops;
    u32 divergent_depth = 0;
    for (const auto& node : program.syntax_list) {
        switch (node.type) {
        case Type::If: {
            const bool divergent = IsDivergent(node.data.if_node.cond);
            conditionals.push_back({node.data.if_node.merge, divergent});
            divergent_depth += u32(divergent);
            break;
        }
        case Type::EndIf:
            ASSERT(!conditionals.empty() && conditionals.back().merge == node.data.end_if.merge);
            divergent_depth -= u32(conditionals.back().divergent);
            conditionals.pop_back();
            break;
        case Type::Loop:
            loops.push_back(node.data.loop.merge);
            break;
        case Type::Repeat:
            if (loops.empty() || loops.back() != node.data.repeat.merge) {
                return {};
            }
            loops.pop_back();
            break;
        case Type::Block:
            if (divergent_depth == 0 && loops.empty()) {
                blocks.push_back(node.data.block);
            }
            break;
        default:
            break;
        }
    }
    if (!conditionals.empty() || !loops.empty()) {
        return {};
    }
    return blocks;
}

} // namespace

void LowerWave64BallotPass(IR::Program& program, const RuntimeInfo& runtime_info,
                           const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute || profile.subgroup_size == 64) {
        return;
    }
    const auto [size_x, size_y, size_z] = runtime_info.hw.cs.workgroup_size;
    const u32 num_threads = size_x * size_y * size_z;
    if (num_threads <= 32) {
        return;
    }
    std::vector<IR::Inst*> worklist;
    const auto uniform = UniformBlocks(program);
    for (IR::Block* block : program.blocks) {
        const bool is_uniform = std::ranges::contains(uniform, block);
        const auto take = [&](IR::Inst& inst) {
            if (is_uniform) {
                worklist.push_back(&inst);
            } else {
                LOG_WARNING(Render_Recompiler, "{} instruction in non uniform control flow",
                            inst.GetOpcode());
            }
        };
        for (IR::Inst& inst : block->Instructions()) {
            const auto op = inst.GetOpcode();
            if (op == IR::Opcode::ReadLane && inst.Arg(1).IsImmediate()) {
                take(inst);
            } else if (op == IR::Opcode::Ballot) {
                // Ballots whose high half is read.
                if (std::ranges::any_of(inst.Uses(), [](const IR::Use& use) {
                        return use.user->GetOpcode() == IR::Opcode::UnpackUint2x32;
                    })) {
                    take(inst);
                }
            } else if (op == IR::Opcode::GetAttributeU32 &&
                       inst.Arg(0).Attribute() == IR::Attribute::SubgroupLtMask) {
                worklist.push_back(&inst);
            }
        }
    }
    if (worklist.empty()) {
        return;
    }
    // The scratch after the shader's own shared memory.
    const auto& cs = runtime_info.hw.cs;
    const u32 scratch_base = Common::AlignUp(cs.shared_memory_size, sizeof(u64));
    const u32 scratch_size = (Common::AlignUp(num_threads, 64) / profile.subgroup_size) * sizeof(u32);
    program.info.shared_memory_scratch_size = scratch_base + scratch_size - cs.shared_memory_size;

    for (IR::Inst* inst : worklist) {
        LOG_INFO(Render_Recompiler, "Lowering {} instruction for wave64", inst->GetOpcode());
        IR::IREmitter ir{*inst->GetParent(), IR::Block::InstructionList::s_iterator_to(*inst)};
        const IR::U32 invocation_index = ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex);
        const IR::U32 subgroup_id = ir.ShiftRightLogical(invocation_index, ir.Imm32(5));
        if (inst->GetOpcode() == IR::Opcode::Ballot) {
            // Each 32-lane half writes its mask; the 64-bit mask is both, read back.
            const IR::U32 mask_low =
                IR::U32{ir.CompositeExtract(ir.UnpackUint2x32(ir.Ballot(IR::U1{inst->Arg(0)})), 0)};
            const IR::U32 offset =
                ir.IAdd(ir.Imm32(scratch_base), ir.ShiftLeftLogical(subgroup_id, ir.Imm32(2u)));
            ir.WriteShared(32, mask_low, offset);
            ir.Barrier();
            const IR::U64 mask = IR::U64{ir.LoadShared(64, false, offset)};
            ir.Barrier();
            inst->ReplaceUsesWithAndRemove(mask);
        } else if (inst->GetOpcode() == IR::Opcode::ReadLane) {
            // Each half publishes its lane's value; the one of the lane's half is read.
            const IR::U32 lane32 = ir.BitwiseAnd(IR::U32{inst->Arg(1)}, ir.Imm32(31));
            const IR::U32 half = ir.ShiftRightLogical(IR::U32{inst->Arg(1)}, ir.Imm32(5u));
            const IR::U32 offset =
                ir.IAdd(ir.Imm32(scratch_base), ir.ShiftLeftLogical(subgroup_id, ir.Imm32(2u)));
            ir.WriteShared(32, ir.ReadLane(IR::U32{inst->Arg(0)}, lane32), offset);
            ir.Barrier();
            const IR::U32 value =
                IR::U32{ir.LoadShared(32, false,
                                      ir.IAdd(ir.BitwiseAnd(offset, ir.Imm32(~7u)),
                                              ir.ShiftLeftLogical(half, ir.Imm32(2u))))};
            ir.Barrier();
            inst->ReplaceUsesWithAndRemove(value);
        } else {
            // SubgroupLtMask over 64 lanes: (1 << lane) - 1, low or high dword.
            const IR::U32 lane = ir.BitwiseAnd(invocation_index, ir.Imm32(63));
            const IR::U64 mask =
                ir.ISub(ir.ShiftLeftLogical(ir.Imm64(u64{1}), lane), ir.Imm64(u64{1}));
            inst->ReplaceUsesWithAndRemove(
                ir.CompositeExtract(ir.UnpackUint2x32(mask), inst->Arg(1).U32()));
        }
    }
}

} // namespace Shader::Optimization
