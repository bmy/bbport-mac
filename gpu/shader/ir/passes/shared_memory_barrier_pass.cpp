// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Compute shaders of exactly one GCN wave (64 invocations) synchronize shared memory implicitly
// on the console; where the host's subgroups are smaller (NVIDIA: 32) barriers are inserted:
// between a shared write and a later read (or a read and a later write) in a block, at the end of
// such blocks, and at the merge after divergent branches and loops (never inside them).
#include <optional>
#include <unordered_set>

#include "shader/ir/breadth_first_search.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/profile.h"

namespace Shader::Optimization {

namespace {

constexpr u32 GcnSubgroupSize = 64;

using BlockSet = std::unordered_set<const IR::Block*>; // membership only

bool IsSharedRead(const IR::Inst& inst) {
    const auto op = inst.GetOpcode();
    return op == IR::Opcode::LoadSharedU16 || op == IR::Opcode::LoadSharedU32 ||
           op == IR::Opcode::LoadSharedU64;
}

bool IsSharedWrite(const IR::Inst& inst) {
    const auto op = inst.GetOpcode();
    if (op >= IR::Opcode::SharedAtomicIAdd32 && op <= IR::Opcode::SharedAtomicCmpSwap64) {
        return !inst.Flags<bool>(); // GDS atomics are not shared memory
    }
    return op == IR::Opcode::WriteSharedU16 || op == IR::Opcode::WriteSharedU32 ||
           op == IR::Opcode::WriteSharedU64;
}

void BarriersInBlock(IR::Block* block) {
    enum class Next { None, BeforeWrite, BeforeRead } next = Next::None;
    for (IR::Inst& inst : block->Instructions()) {
        const bool read = IsSharedRead(inst);
        if (!read && !IsSharedWrite(inst)) {
            continue;
        }
        if (next == (read ? Next::BeforeRead : Next::BeforeWrite)) {
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            ir.Barrier();
        }
        next = read ? Next::BeforeWrite : Next::BeforeRead;
    }
    if (next != Next::None) {
        IR::IREmitter ir{*block, --block->end()};
        ir.Barrier();
    }
}

void BarrierAtStart(IR::Block* block) {
    IR::IREmitter ir{*block, std::ranges::find_if_not(block->Instructions(), IR::IsPhi)};
    ir.Barrier();
}

/// Whether a condition depends on the invocation (so lanes may go different ways).
bool IsDivergent(const IR::U1& cond) {
    return IR::BreadthFirstSearch(cond, [](IR::Inst* inst) -> std::optional<bool> {
               if (inst->GetOpcode() == IR::Opcode::GetAttributeU32 &&
                   inst->Arg(0).Attribute() == IR::Attribute::LocalInvocationId) {
                   return true;
               }
               return std::nullopt;
           }) == true;
}

/// Loops invocations may leave on different iterations (by their merge blocks).
BlockSet DivergentLoops(const IR::AbstractSyntaxList& syntax_list) {
    BlockSet loops;
    for (const auto& node : syntax_list) {
        if (node.type == IR::AbstractSyntaxNode::Type::Repeat &&
            IsDivergent(node.data.repeat.cond)) {
            loops.emplace(node.data.repeat.merge);
        } else if (node.type == IR::AbstractSyntaxNode::Type::Break &&
                   IsDivergent(node.data.break_node.cond)) {
            loops.emplace(node.data.break_node.merge);
        }
    }
    return loops;
}

} // namespace

void SharedMemoryBarrierPass(IR::Program& program, const RuntimeInfo& runtime_info,
                             const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const auto& cs = runtime_info.hw.cs;
    const u32 group_size = cs.workgroup_size[0] * cs.workgroup_size[1] * cs.workgroup_size[2];
    if (cs.shared_memory_size == 0 || group_size != GcnSubgroupSize ||
        !profile.needs_lds_barriers) {
        return;
    }
    using Type = IR::AbstractSyntaxNode::Type;
    u32 divergence_depth = 0;
    BlockSet divergence_end;
    const BlockSet divergent_loops = DivergentLoops(program.syntax_list);
    for (const auto& node : program.syntax_list) {
        switch (node.type) {
        case Type::EndIf:
            if (divergence_end.contains(node.data.end_if.merge)) {
                --divergence_depth;
            }
            break;
        case Type::If:
            if (IsDivergent(node.data.if_node.cond)) {
                if (divergence_depth == 0) {
                    BarrierAtStart(node.data.if_node.merge);
                }
                ++divergence_depth;
                divergence_end.emplace(node.data.if_node.merge);
            }
            break;
        case Type::Loop:
            if (divergent_loops.contains(node.data.loop.merge)) {
                ++divergence_depth;
            }
            break;
        case Type::Repeat:
            if (divergent_loops.contains(node.data.repeat.merge)) {
                ASSERT(divergence_depth > 0);
                if (--divergence_depth == 0) {
                    BarrierAtStart(node.data.repeat.merge);
                }
            }
            break;
        case Type::Block:
            if (divergence_depth == 0) {
                BarriersInBlock(node.data.block);
            }
            break;
        default:
            break;
        }
    }
}

} // namespace Shader::Optimization
