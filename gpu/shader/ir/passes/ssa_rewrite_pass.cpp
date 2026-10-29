// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Register reads and writes (GCN registers, goto variables, virtual registers) become SSA
// values, by the on-the-fly construction of Braun et al., "Simple and Efficient Construction of
// Static Single Assignment Form" (CC 2013): blocks are visited in reverse post order, a read
// looks back through the predecessors, a join block gets a phi (operands filled once all blocks
// are visited), and phis that merge one value are removed at the end.
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <magic_enum/magic_enum.hpp>

#include "common/assert.h"
#include "shader/ir/basic_block.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/program.h"

namespace Shader::Optimization {

namespace {

using IR::Opcode;
using IR::RegType;

/// The current value of each register per block: GCN registers in the block's SsaState slots,
/// the others in maps.
class DefTable {
public:
    IR::Value& Def(IR::Block* block, IR::RegTag tag) {
        if (tag.IsIntrusive()) {
            return block->ssa_state.values[tag.Index()];
        }
        switch (tag.type) {
        case RegType::GotoVariable:
            return goto_vars[tag.index][block];
        case RegType::VirtualReg:
            return virtual_regs[tag.reg.Key()][block];
        default:
            UNREACHABLE();
        }
    }

private:
    using PerBlock = std::unordered_map<IR::Block*, IR::Value>;
    std::unordered_map<u32, PerBlock> goto_vars;
    std::unordered_map<u64, PerBlock> virtual_regs;
};

IR::Type TypeOf(IR::RegTag tag) {
    switch (tag.type) {
    case RegType::ScalarReg:
    case RegType::VectorReg:
    case RegType::VccLo:
    case RegType::VccHi:
    case RegType::M0:
        return IR::Type::U32;
    case RegType::Scc:
    case RegType::Exec:
    case RegType::GotoVariable:
        return IR::Type::U1;
    case RegType::VirtualReg:
        return tag.reg.type;
    default:
        UNREACHABLE();
    }
}

Opcode UndefOpcode(IR::RegTag tag) {
    switch (TypeOf(tag)) {
    case IR::Type::U1:
        return Opcode::UndefU1;
    case IR::Type::U32:
        return Opcode::UndefU32;
    case IR::Type::U64:
        return Opcode::UndefU64;
    case IR::Type::U32x2:
        return Opcode::UndefU32x2;
    case IR::Type::U32x3:
        return Opcode::UndefU32x3;
    case IR::Type::U32x4:
        return Opcode::UndefU32x4;
    case IR::Type::F32:
        return Opcode::UndefF32;
    default:
        UNREACHABLE_MSG("No undefined value of {} for {}", TypeOf(tag),
                        magic_enum::enum_name(tag.type));
    }
}

/// A register access: which register, and whether it writes it.
struct RegAccess {
    IR::RegTag tag;
    bool write{};
};

RegAccess AccessOf(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case Opcode::SetScalarRegister:
        return {IR::RegTag{inst.Arg(0).ScalarReg()}, true};
    case Opcode::GetScalarRegister:
        return {IR::RegTag{inst.Arg(0).ScalarReg()}};
    case Opcode::SetVectorRegister:
        return {IR::RegTag{inst.Arg(0).VectorReg()}, true};
    case Opcode::GetVectorRegister:
        return {IR::RegTag{inst.Arg(0).VectorReg()}};
    case Opcode::SetVirtualRegister:
        return {IR::RegTag{inst.Arg(0).VirtualReg()}, true};
    case Opcode::GetVirtualRegister:
        return {IR::RegTag{inst.Arg(0).VirtualReg()}};
    case Opcode::SetGotoVariable:
        return {IR::RegTag{RegType::GotoVariable, inst.Arg(0).U32()}, true};
    case Opcode::GetGotoVariable:
        return {IR::RegTag{RegType::GotoVariable, inst.Arg(0).U32()}};
    case Opcode::SetExec:
        return {IR::RegTag{RegType::Exec}, true};
    case Opcode::GetExec:
        return {IR::RegTag{RegType::Exec}};
    case Opcode::SetScc:
        return {IR::RegTag{RegType::Scc}, true};
    case Opcode::GetScc:
        return {IR::RegTag{RegType::Scc}};
    case Opcode::SetVccLo:
        return {IR::RegTag{RegType::VccLo}, true};
    case Opcode::GetVccLo:
        return {IR::RegTag{RegType::VccLo}};
    case Opcode::SetVccHi:
        return {IR::RegTag{RegType::VccHi}, true};
    case Opcode::GetVccHi:
        return {IR::RegTag{RegType::VccHi}};
    case Opcode::SetM0:
        return {IR::RegTag{RegType::M0}, true};
    case Opcode::GetM0:
        return {IR::RegTag{RegType::M0}};
    default:
        return {};
    }
}

class SsaBuilder {
public:
    void Write(IR::RegTag tag, IR::Block* block, const IR::Value& value) {
        defs.Def(block, tag) = value;
    }

    IR::Value Read(IR::RegTag tag, IR::Block* block) {
        // Blocks with one predecessor passed on the way take the value found.
        boost::container::small_vector<IR::Block*, 16> chain;
        IR::Value result{};
        while (block) {
            if (const IR::Value& def = defs.Def(block, tag); !def.IsEmpty()) {
                result = def;
                break;
            }
            const auto preds = block->ImmPredecessors();
            if (preds.size() == 1) {
                chain.push_back(std::exchange(block, preds.front()));
                continue;
            }
            if (preds.empty()) {
                // Read before any write.
                result = IR::Value{&*block->PrependNewInst(block->begin(), UndefOpcode(tag))};
            } else {
                // A join: a phi, the block's value from now on (which ends cycles).
                IR::Inst* const phi = &*block->PrependNewInst(block->begin(), Opcode::Phi);
                phi->SetFlags(TypeOf(tag));
                phi->SetRegTag(tag);
                pending_phis.push_back(phi);
                result = IR::Value{phi};
            }
            Write(tag, block, result);
            break;
        }
        for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
            Write(tag, *it, result);
        }
        return result;
    }

    /// Fills the phis' operands (reads may add phis on the way).
    void FillPhis() {
        for (size_t i = 0; i < pending_phis.size(); i++) {
            IR::Inst* const phi = pending_phis[i];
            for (IR::Block* const pred : phi->GetParent()->ImmPredecessors()) {
                phi->AddPhiOperand(pred, Read(phi->GetRegTag(), pred));
            }
        }
    }

    /// Removes phis merging one value besides themselves, and the ones that become so.
    void RemoveTrivialPhis() {
        std::vector<IR::Inst*> worklist = std::move(pending_phis);
        while (!worklist.empty()) {
            IR::Inst* const phi = worklist.back();
            worklist.pop_back();
            if (phi->GetOpcode() == Opcode::Void) {
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
            IR::Block* const block = phi->GetParent();
            if (same.IsEmpty()) {
                // Only itself: undefined, placed after the block's phis.
                const auto after_phis = std::ranges::find_if_not(block->Instructions(), IR::IsPhi);
                same = IR::Value{
                    &*block->PrependNewInst(after_phis, UndefOpcode(phi->GetRegTag()))};
            }
            for (const auto& [user, operand] : phi->Uses()) {
                if (user->GetOpcode() == Opcode::Phi && user != phi) {
                    worklist.push_back(user);
                }
            }
            phi->ReplaceUsesWithAndRemove(same);
            block->Instructions().erase(IR::Block::InstructionList::s_iterator_to(*phi));
        }
    }

private:
    DefTable defs;
    std::vector<IR::Inst*> pending_phis;
};

} // namespace

void SsaRewritePass(IR::Program& program) {
    SsaBuilder builder;
    for (auto it = program.post_order_blocks.rbegin(); it != program.post_order_blocks.rend();
         ++it) {
        IR::Block* const block = *it;
        for (IR::Inst& inst : block->Instructions()) {
            const RegAccess access = AccessOf(inst);
            if (!access.tag) {
                continue;
            }
            if (access.write) {
                builder.Write(access.tag, block, inst.Arg(inst.NumArgs() - 1));
            } else {
                inst.ReplaceUsesWithAndRemove(builder.Read(access.tag, block));
            }
        }
    }
    builder.FillPhis();
    builder.RemoveTrivialPhis();
}

} // namespace Shader::Optimization
