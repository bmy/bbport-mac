// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/frontend/cfg.h"

#include <algorithm>
#include <optional>

#include "common/assert.h"
#include "common/logging/log.h"

namespace Shader::Gcn {

using GcnIsa::InstCategory;
using GcnIsa::Opcode;
using GcnIsa::OperandField;

namespace {

/// The condition under which a block ending in `inst` takes its branch.
IR::Condition ConditionOf(const GcnInst& inst) {
    if (inst.IsCmpx()) {
        return IR::Condition::Execnz;
    }
    switch (inst.opcode) {
    case Opcode::S_CBRANCH_SCC0:
        return IR::Condition::Scc0;
    case Opcode::S_CBRANCH_SCC1:
        return IR::Condition::Scc1;
    case Opcode::S_CBRANCH_VCCZ:
        return IR::Condition::Vccz;
    case Opcode::S_CBRANCH_VCCNZ:
        return IR::Condition::Vccnz;
    case Opcode::S_CBRANCH_EXECZ:
        return IR::Condition::Execz;
    case Opcode::S_CBRANCH_EXECNZ:
        return IR::Condition::Execnz;
    default:
        return IR::Condition::True;
    }
}

/// Scalar instructions, branches and lane accesses run whatever the EXEC mask.
bool IgnoresExec(const GcnInst& inst) {
    switch (inst.category) {
    case InstCategory::ScalarALU:
    case InstCategory::ScalarMemory:
    case InstCategory::FlowControl:
        return true;
    default:
        break;
    }
    return inst.opcode == Opcode::V_READLANE_B32 || inst.opcode == Opcode::V_WRITELANE_B32 ||
           inst.opcode == Opcode::V_READFIRSTLANE_B32;
}

/// The target of an S_SETPC_B64 that jumps by an offset from S_GETPC_B64:
///   s_getpc_b64 s[n:n+1]; ...; s_add_u32/s_sub_u32 sn, sn, literal; ...; s_setpc_b64 s[n:n+1]
std::optional<u32> SetPcTarget(std::span<const GcnInst> list, u32 setpc_index,
                               std::span<const u32> pc_of) {
    if (setpc_index < 3) {
        return std::nullopt;
    }
    const auto& getpc = list[setpc_index - 3];
    const auto& arith = list[setpc_index - 2];
    const auto& setpc = list[setpc_index];
    const bool add = arith.opcode == Opcode::S_ADD_U32;
    if (getpc.opcode != Opcode::S_GETPC_B64 || (!add && arith.opcode != Opcode::S_SUB_U32) ||
        setpc.opcode != Opcode::S_SETPC_B64 || getpc.dst[0].code != setpc.src[0].code ||
        arith.dst[0].code != setpc.src[0].code || arith.src_count < 2 ||
        arith.src[1].field != OperandField::LiteralConst) {
        return std::nullopt;
    }
    const s32 offset = add ? s32(arith.src[1].code) : -s32(arith.src[1].code);
    const u32 base = pc_of[setpc_index - 3] + getpc.length;
    return u32(s32(base) + offset) & ~0x3u;
}

/// Where the branch ending at `index` (at byte address `pc`) goes.
u32 BranchTargetOf(std::span<const GcnInst> list, u32 index, std::span<const u32> pc_of,
                   u32 pc) {
    const auto& inst = list[index];
    if (inst.opcode != Opcode::S_SETPC_B64) {
        return inst.BranchTarget(pc);
    }
    const auto target = SetPcTarget(list, index, pc_of);
    ASSERT_MSG(target, "S_SETPC_B64 at {:#x} (instruction {}) without a recognized offset", pc,
               index);
    return *target;
}

} // namespace

CFG::CFG(Common::ObjectPool<Block>& block_pool_, std::span<const GcnInst> inst_list_)
    : block_pool{block_pool_}, inst_list{inst_list_} {
    FindLabels();
    MakeBlocks();
    LinkBlocks();
    SplitDivergenceScopes();
    RemoveUnreachableBlocks();
}

void CFG::FindLabels() {
    index_to_pc.resize(inst_list.size() + 1);
    labels.reserve(32);
    labels.push_back(0); // the entry
    u32 pc = 0;
    for (u32 i = 0; i < inst_list.size(); ++i) {
        index_to_pc[i] = pc;
        const auto& inst = inst_list[i];
        const u32 next = pc + inst.length;
        if (inst.IsUnconditionalBranch()) {
            labels.push_back(BranchTargetOf(inst_list, i, index_to_pc, pc));
            labels.push_back(next); // the branch ends its block
        } else if (inst.IsConditionalBranch()) {
            const u32 target = inst.BranchTarget(pc);
            if (target != next) {
                labels.push_back(target);
                labels.push_back(next);
            }
        } else if (inst.opcode == Opcode::S_ENDPGM) {
            labels.push_back(next);
        }
        pc = next;
    }
    index_to_pc[inst_list.size()] = pc;
    std::ranges::sort(labels);
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
}

u32 CFG::IndexOf(u32 pc) const {
    if (pc == 0) {
        return 0;
    }
    const auto it = std::ranges::lower_bound(index_to_pc, pc);
    ASSERT(it != index_to_pc.end() || pc > index_to_pc.back());
    return u32(it - index_to_pc.begin());
}

Block& CFG::BlockAt(u32 pc) {
    const auto it = std::ranges::lower_bound(blocks.blocks, pc, {}, &Block::begin);
    ASSERT_MSG(it != blocks.blocks.end() && (*it)->begin == pc, "no block at {:#x}", pc);
    return **it;
}

void CFG::MakeBlocks() {
    // A block from each label to the next (the last label is the program's end).
    for (size_t i = 0; i + 1 < labels.size(); ++i) {
        const u32 end_index = IndexOf(labels[i + 1]) - 1;
        Block* block = block_pool.Create();
        block->begin = labels[i];
        block->end = labels[i + 1];
        block->begin_index = IndexOf(labels[i]);
        block->end_index = end_index;
        block->end_inst = inst_list[end_index];
        block->cond = ConditionOf(block->end_inst);
        blocks.blocks.push_back(block);
    }
}

void CFG::LinkBlocks() {
    for (Block* block : blocks.blocks) {
        const auto& end_inst = block->end_inst;
        if (!end_inst.IsTerminateInstruction()) {
            // Falls through to the next block.
            Block& next = BlockAt(block->end);
            block->branch_true = &next;
            block->end_class = EndClass::Branch;
            ++next.num_predecessors;
            continue;
        }
        const u32 branch_pc = block->end - end_inst.length;
        if (end_inst.opcode == Opcode::S_ENDPGM) {
            block->end_class = EndClass::Exit;
            continue;
        }
        const u32 target = BranchTargetOf(inst_list, block->end_index, index_to_pc, branch_pc);
        if (end_inst.IsUnconditionalBranch()) {
            Block& taken = BlockAt(target);
            block->branch_true = &taken;
            block->end_class = EndClass::Branch;
            ++taken.num_predecessors;
        } else if (end_inst.IsConditionalBranch()) {
            Block& next = BlockAt(block->end);
            block->end_class = EndClass::Branch;
            if (target != block->end) {
                Block& taken = BlockAt(target);
                block->branch_true = &taken;
                block->branch_false = &next;
                ++taken.num_predecessors;
            } else {
                // A branch to the next instruction.
                block->branch_true = &next;
                block->cond = IR::Condition::True;
            }
            ++next.num_predecessors;
        } else {
            UNREACHABLE_MSG("branch {} at {:#x}", GcnIsa::NameOf(end_inst.opcode), branch_pc);
        }
    }
}

void CFG::SplitDivergenceScopes() {
    // A scope opens where EXEC is narrowed (saving it, or masking it out after a save) ...
    const auto opens = [](const GcnInst& inst) {
        return inst.opcode == Opcode::S_AND_SAVEEXEC_B64 ||
               (inst.opcode == Opcode::S_ANDN2_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               inst.IsCmpx();
    };
    // ... and closes where EXEC is restored, at the branch skipping it, or at the end.
    const auto closes = [](const GcnInst& inst) {
        return (inst.opcode == Opcode::S_MOV_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               inst.opcode == Opcode::S_CBRANCH_EXECZ || inst.opcode == Opcode::S_ENDPGM ||
               (inst.opcode == Opcode::S_ANDN2_B64 && inst.dst[0].field == OperandField::ExecLo);
    };
    auto& list = blocks.blocks;
    // The blocks inserted below are visited as well.
    for (size_t b = 0; b < list.size(); ++b) {
        Block* const parent = list[b];
        size_t insert_at = b + 1;
        s32 scope_begin = -1;
        for (u32 index = parent->begin_index; index <= parent->end_index; ++index) {
            const auto& inst = inst_list[index];
            const bool close = closes(inst);
            if ((close || index == parent->end_index) && scope_begin != -1) {
                // An empty scope, or one whose instructions all ignore EXEC, needs no block.
                const bool empty = index - scope_begin == 1 && close;
                const auto body_first = inst_list.begin() + scope_begin + 1;
                const auto body_end = inst_list.begin() + index + !close;
                if (!empty && !std::all_of(body_first, body_end, IgnoresExec)) {
                    // The instructions from the first to the last one EXEC affects.
                    s32 first = scope_begin;
                    do {
                        ++first;
                    } while (IgnoresExec(inst_list[first]));
                    s32 last = s32(index);
                    while (IgnoresExec(inst_list[last])) {
                        --last;
                    }
                    Block* scope = block_pool.Create();
                    scope->begin = index_to_pc[first];
                    scope->end = index_to_pc[last];
                    scope->begin_index = u32(first);
                    scope->end_index = u32(last);
                    scope->end_inst = inst_list[last];
                    scope->num_predecessors = 1;
                    list.insert(list.begin() + insert_at++, scope);
                    scope->cond = IR::Condition::True;
                    scope->branch_false = nullptr;
                    if (u32(last) != parent->end_index) {
                        // The rest of the parent goes to an epilogue block both reach.
                        Block* epilogue = block_pool.Create();
                        epilogue->begin = index_to_pc[last + 1];
                        epilogue->end = parent->end;
                        epilogue->begin_index = u32(last + 1);
                        epilogue->end_index = parent->end_index;
                        epilogue->end_inst = parent->end_inst;
                        epilogue->cond = parent->cond;
                        epilogue->end_class = parent->end_class;
                        epilogue->branch_true = parent->branch_true;
                        epilogue->branch_false = parent->branch_false;
                        epilogue->num_predecessors = 2;
                        list.insert(list.begin() + insert_at++, epilogue);
                        scope->branch_true = epilogue;
                        parent->branch_false = epilogue;
                    } else {
                        // The scope ends the parent: both go where the parent went.
                        ASSERT(parent->cond == IR::Condition::True && parent->branch_true);
                        scope->branch_true = parent->branch_true;
                        parent->branch_false = parent->branch_true;
                        ++parent->branch_true->num_predecessors;
                    }
                    // The parent ends right before the scope and enters it when a lane is on.
                    const u32 parent_last = u32(first - 1);
                    parent->end = index_to_pc[parent_last];
                    parent->end_index = parent_last;
                    parent->end_inst = inst_list[parent_last];
                    parent->cond = IR::Condition::Execnz;
                    parent->end_class = EndClass::Branch;
                    parent->branch_true = scope;
                }
                scope_begin = -1;
            }
            if (opens(inst)) {
                scope_begin = s32(index);
            }
        }
    }
}

void CFG::RemoveUnreachableBlocks() {
    auto& list = blocks.blocks;
    std::erase_if(list, [first = list.empty() ? nullptr : list.front()](const Block* block) {
        if (block == first || block->num_predecessors != 0) {
            return false;
        }
        LOG_WARNING(Render_Recompiler, "Removing unreachable block begin={:#x}", block->begin);
        return true;
    });
}

} // namespace Shader::Gcn
