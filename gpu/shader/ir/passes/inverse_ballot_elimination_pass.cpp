// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// GCN keeps per-lane booleans as 64-bit masks (VCC, EXEC, compare results): the translator
// builds them with Ballot and reads them back with InverseBallot. This pass turns the round
// trips back into booleans: through Ballot directly, and through the mask logic (and, or, xor,
// not, select, phis) by doing the logic on the booleans.
#include <limits>
#include <unordered_map>
#include <vector>

#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/expression_elimination.h"
#include "shader/ir/passes/ir_passes.h"

namespace Shader::Optimization {

namespace {

/// A composite rebuilt from the elements of one vector, in order, is that vector.
void FoldCompositeConstruct(IR::Inst& inst, IR::Opcode extract) {
    IR::Value source{};
    for (size_t i = 0; i < inst.NumArgs(); ++i) {
        const IR::Value value = inst.Arg(i);
        if (value.IsImmediate()) {
            return;
        }
        const IR::Inst* const element = value.Inst();
        if (element->GetOpcode() != extract || element->Arg(1).U32() != i) {
            return;
        }
        if (source.IsEmpty()) {
            source = element->Arg(0);
        } else if (source != element->Arg(0)) {
            return;
        }
    }
    inst.ReplaceUsesWithAndRemove(source);
}

/// f(g(x)) = x for an inverse pair.
void FoldInverse(IR::Inst& inst, IR::Opcode inverse) {
    const IR::Value value = inst.Arg(0);
    if (!value.IsImmediate() && value.Inst()->GetOpcode() == inverse) {
        inst.ReplaceUsesWithAndRemove(value.Inst()->Arg(0));
    }
}

void Simplify(IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::PackUint2x32:
        FoldInverse(inst, IR::Opcode::UnpackUint2x32);
        break;
    case IR::Opcode::CompositeConstructU32x2:
        FoldCompositeConstruct(inst, IR::Opcode::CompositeExtractU32x2);
        break;
    case IR::Opcode::CompositeConstructU32x3:
        FoldCompositeConstruct(inst, IR::Opcode::CompositeExtractU32x3);
        break;
    case IR::Opcode::CompositeConstructU32x4:
        FoldCompositeConstruct(inst, IR::Opcode::CompositeExtractU32x4);
        break;
    default:
        break;
    }
}

bool IsMaskLogic(IR::Opcode op) {
    switch (op) {
    case IR::Opcode::BitwiseAnd64:
    case IR::Opcode::BitwiseNot64:
    case IR::Opcode::BitwiseOr64:
    case IR::Opcode::BitwiseXor64:
    case IR::Opcode::SelectU64:
    case IR::Opcode::Phi:
        return true;
    default:
        return false;
    }
}

} // namespace

void InverseBallotEliminationPass(IR::Program& program) {
    std::vector<IR::Inst*> worklist;
    for (IR::Block* const block : program.blocks) {
        RunLocalCSE(block);
        for (IR::Inst& inst : block->Instructions()) {
            Simplify(inst);
            if (inst.GetOpcode() == IR::Opcode::InverseBallot) {
                worklist.push_back(&inst);
            }
        }
    }
    // The boolean form of each mask producer, made once (lookups only).
    std::unordered_map<IR::Inst*, IR::Value> booleans;
    while (!worklist.empty()) {
        IR::Inst* const inst = worklist.back();
        worklist.pop_back();
        if (inst->GetOpcode() == IR::Opcode::Void) {
            continue;
        }
        const IR::Value mask = inst->Arg(0);
        if (mask.IsImmediate()) {
            if (mask.U64() == 0ull) {
                inst->ReplaceUsesWithAndRemove(IR::Value{false});
            } else if (mask.U64() == std::numeric_limits<u64>::max()) {
                inst->ReplaceUsesWithAndRemove(IR::Value{true});
            } else {
                UNREACHABLE_MSG("InverseBallot of the constant {:#x}", mask.U64());
            }
            continue;
        }
        IR::Inst* const producer = mask.Inst();
        const IR::Opcode op = producer->GetOpcode();
        if (op == IR::Opcode::Ballot) {
            inst->ReplaceUsesWithAndRemove(producer->Arg(0));
            continue;
        }
        IR::Block* const block = producer->GetParent();
        const auto at = IR::Block::InstructionList::s_iterator_to(*producer);
        if (op == IR::Opcode::UndefU64) {
            const IR::Value undef{&*block->PrependNewInst(at, IR::Opcode::UndefU1)};
            inst->ReplaceUsesWithAndRemove(undef);
            continue;
        }
        if (!IsMaskLogic(op)) {
            continue;
        }
        auto [it, is_new] = booleans.try_emplace(producer);
        if (is_new) {
            IR::IREmitter ir{*block, at};
            // The operands' booleans are new InverseBallots, simplified in turn.
            const auto inverse = [&](IR::IREmitter& emitter, size_t arg) {
                const IR::U1 value = emitter.InverseBallot(IR::U64{producer->Arg(arg)});
                worklist.push_back(value.Inst());
                return value;
            };
            if (op == IR::Opcode::Phi) {
                IR::Inst* const phi = &*block->PrependNewInst(at, IR::Opcode::Phi);
                phi->SetFlags(IR::Type::U1);
                for (size_t i = 0; i < producer->NumArgs(); ++i) {
                    IR::Block* const pred = producer->PhiBlock(i);
                    IR::IREmitter pred_ir{*pred};
                    phi->AddPhiOperand(pred, inverse(pred_ir, i));
                }
                it->second = IR::Value{phi};
            } else if (op == IR::Opcode::SelectU64) {
                const IR::U1 a = inverse(ir, 1);
                const IR::U1 b = inverse(ir, 2);
                it->second = ir.Select(IR::U1{producer->Arg(0)}, a, b);
            } else {
                const IR::U1 a = inverse(ir, 0);
                if (op == IR::Opcode::BitwiseNot64) {
                    it->second = ir.LogicalNot(a);
                } else {
                    const IR::U1 b = inverse(ir, 1);
                    if (op == IR::Opcode::BitwiseAnd64) {
                        it->second = ir.LogicalAnd(a, b);
                    } else if (op == IR::Opcode::BitwiseOr64) {
                        it->second = ir.LogicalOr(a, b);
                    } else {
                        it->second = ir.LogicalXor(a, b);
                    }
                }
            }
        }
        // Every InverseBallot of the producer takes the boolean (a copy: replacing edits uses).
        const auto uses = producer->Uses();
        for (const auto& [user, operand] : uses) {
            if (user->GetOpcode() == IR::Opcode::InverseBallot) {
                user->ReplaceUsesWithAndRemove(it->second);
            }
        }
    }
}

} // namespace Shader::Optimization
