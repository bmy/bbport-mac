// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// v_readlane of a constant lane after v_writelane to that lane (scalar values spilled into VGPR
// lanes and read back) is the written value. Through control flow the read follows the phis:
// each phi on the way is duplicated with the lane's values, and phis left trivial are removed.
#include <queue>
#include <unordered_map>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/program.h"

namespace Shader::Optimization {

namespace {

/// Walks back a WriteLane chain: the write to `lane`, or the value under the chain.
IR::Inst* SearchChain(IR::Inst* inst, u32 lane) {
    while (inst->GetOpcode() == IR::Opcode::WriteLane) {
        if (inst->Arg(2).U32() == lane) {
            break;
        }
        inst = inst->Arg(0).Inst();
    }
    return inst;
}

/// Whether every path back from `inst` ends at a write to `lane` (or undef) through phis and
/// other reads only.
bool IsPossibleToEliminate(IR::Inst* start, u32 lane) {
    boost::container::small_vector<IR::Inst*, 16> visited;
    std::queue<IR::Inst*> queue;
    queue.push(start);
    while (!queue.empty()) {
        IR::Inst* const inst = SearchChain(queue.front(), lane);
        queue.pop();
        const IR::Opcode op = inst->GetOpcode();
        if (op == IR::Opcode::WriteLane) {
            continue;
        }
        if (op != IR::Opcode::ReadLane && op != IR::Opcode::Phi && op != IR::Opcode::UndefU32) {
            return false;
        }
        // Rightmost arguments first.
        for (size_t arg = inst->NumArgs(); arg--;) {
            const IR::Value value = inst->Arg(arg);
            if (value.IsImmediate()) {
                continue;
            }
            IR::Inst* const arg_inst = value.Inst();
            if (std::ranges::find(visited, arg_inst) == visited.end()) {
                visited.push_back(arg_inst);
                queue.push(arg_inst);
            }
        }
    }
    return true;
}

/// The lane's value at a phi network, made once per phi; `made` keeps creation order.
struct LanePhis {
    std::unordered_map<IR::Inst*, IR::Value> map;
    std::vector<IR::Inst*> made;
};

IR::Value GetRealValue(LanePhis& phis, IR::Inst* inst, u32 lane) {
    inst = SearchChain(inst, lane);
    switch (inst->GetOpcode()) {
    case IR::Opcode::WriteLane:
        return inst->Arg(1);
    case IR::Opcode::UndefU32:
        return IR::Value{inst};
    case IR::Opcode::Phi:
        break;
    default:
        UNREACHABLE();
    }
    const auto [it, is_new] = phis.map.try_emplace(inst);
    if (!is_new) {
        // A phi cycle: the duplicate is already there.
        return it->second;
    }
    IR::Block* const block = inst->GetParent();
    IR::Inst* const phi =
        &*block->PrependNewInst(IR::Block::InstructionList::s_iterator_to(*inst), IR::Opcode::Phi);
    phi->SetFlags(IR::Type::U32);
    it->second = IR::Value{phi};
    phis.made.push_back(phi);

    // All operands first (the recursion may add phis), then attached in order.
    boost::container::small_vector<IR::Value, 5> args;
    for (size_t i = 0; i < inst->NumArgs(); ++i) {
        const IR::Value arg = inst->Arg(i);
        args.push_back(arg.IsImmediate() ? arg : GetRealValue(phis, arg.Inst(), lane));
    }
    for (size_t i = 0; i < inst->NumArgs(); ++i) {
        phi->AddPhiOperand(inst->PhiBlock(i), args[i]);
    }
    return IR::Value{phi};
}

/// Removes phis that merge a single value (besides themselves), and the ones that become so.
void RemoveTrivialPhis(std::vector<IR::Inst*>& worklist) {
    while (!worklist.empty()) {
        IR::Inst* const phi = worklist.back();
        worklist.pop_back();
        if (phi->GetOpcode() == IR::Opcode::Void) {
            continue;
        }
        IR::Value same;
        bool trivial = true;
        for (size_t i = 0; i < phi->NumArgs(); ++i) {
            const IR::Value op = phi->Arg(i);
            if (op == same || op == IR::Value{phi}) {
                continue;
            }
            if (!same.IsEmpty()) {
                trivial = false;
                break;
            }
            same = op;
        }
        if (!trivial) {
            continue;
        }
        ASSERT(!same.IsEmpty());
        IR::Block* const block = phi->GetParent();
        for (const auto& [user, operand] : phi->Uses()) {
            if (user->GetOpcode() == IR::Opcode::Phi && user != phi) {
                worklist.push_back(user);
            }
        }
        phi->ReplaceUsesWithAndRemove(same);
        block->Instructions().erase(IR::Block::InstructionList::s_iterator_to(*phi));
    }
}

} // namespace

void ReadLaneEliminationPass(IR::Program& program) {
    LanePhis phis;
    std::vector<IR::Inst*> worklist;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != IR::Opcode::ReadLane || !inst.Arg(1).IsImmediate()) {
                continue;
            }
            const u32 lane = inst.Arg(1).U32();
            IR::Inst* const producer = SearchChain(inst.Arg(0).Inst(), lane);
            IR::Value replacement{};
            if (producer->GetOpcode() == IR::Opcode::WriteLane) {
                replacement = producer->Arg(1);
            } else if (producer->GetOpcode() == IR::Opcode::Phi &&
                       IsPossibleToEliminate(producer, lane)) {
                replacement = GetRealValue(phis, producer, lane);
                worklist.insert(worklist.end(), phis.made.begin(), phis.made.end());
                phis.map.clear();
                phis.made.clear();
            }
            if (replacement.IsEmpty()) {
                continue;
            }
            for (const auto& [user, operand] : inst.Uses()) {
                if (user->GetOpcode() == IR::Opcode::Phi) {
                    worklist.push_back(user);
                }
            }
            inst.ReplaceUsesWith(replacement);
        }
    }
    RemoveTrivialPhis(worklist);
}

} // namespace Shader::Optimization
