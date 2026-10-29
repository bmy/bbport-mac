// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Removes instructions without side effects whose results nothing (transitively) with side
// effects uses.
#include <queue>
#include <unordered_set>

#include "shader/ir/passes/ir_passes.h"

namespace Shader::Optimization {

void DeadCodeEliminationPass(IR::Program& program) {
    // First the unused ones, last to first (removing one may leave an earlier one unused); the
    // instructions with side effects are the roots of what stays.
    std::unordered_set<IR::Inst*> live; // membership only
    std::queue<IR::Inst*> worklist;
    for (IR::Block* block : program.post_order_blocks) {
        for (auto it = block->end(); it != block->begin();) {
            --it;
            const bool side_effects = it->MayHaveSideEffects();
            if (!side_effects && !it->HasUses()) {
                it->Invalidate();
                it = block->Instructions().erase(it);
            } else if (side_effects) {
                live.insert(&*it);
                worklist.push(&*it);
            }
        }
    }
    // Everything the roots use.
    while (!worklist.empty()) {
        IR::Inst* const inst = worklist.front();
        worklist.pop();
        for (size_t i = 0; i < inst->NumArgs(); ++i) {
            IR::Inst* const arg = inst->Arg(i).TryInst();
            if (arg && live.insert(arg).second) {
                worklist.push(arg);
            }
        }
    }
    for (IR::Block* block : program.post_order_blocks) {
        for (auto it = block->begin(); it != block->end();) {
            if (live.contains(&*it)) {
                ++it;
            } else {
                it->Invalidate();
                it = block->Instructions().erase(it);
            }
        }
    }
}

} // namespace Shader::Optimization
