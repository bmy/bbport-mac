// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/dominance.h"

namespace Shader::IR {

void ComputeDominators(const BlockList& post_order) {
    Block* const entry = post_order.back();
    entry->immediate_dominator = entry;
    // The common dominator of two blocks: walk up from the one later in post order.
    const auto intersect = [](Block* a, Block* b) {
        while (a != b) {
            while (a->po_index < b->po_index) {
                a = a->immediate_dominator;
            }
            while (b->po_index < a->po_index) {
                b = b->immediate_dominator;
            }
        }
        return a;
    };
    bool changed = true;
    while (changed) {
        changed = false;
        // Reverse post order, without the entry.
        for (auto it = post_order.rbegin() + 1; it != post_order.rend(); ++it) {
            Block* const block = *it;
            Block* idom = nullptr;
            for (Block* const pred : block->ImmPredecessors()) {
                if (pred->immediate_dominator) {
                    idom = idom ? intersect(idom, pred) : pred;
                }
            }
            if (block->immediate_dominator != idom) {
                block->immediate_dominator = idom;
                changed = true;
            }
        }
    }
}

void ComputeDominanceFrontiers(const BlockList& post_order) {
    for (Block* const block : post_order) {
        if (block->ImmPredecessors().size() < 2) {
            continue;
        }
        for (Block* const pred : block->ImmPredecessors()) {
            for (Block* runner = pred; runner != block->immediate_dominator;
                 runner = runner->immediate_dominator) {
                runner->dominance_frontiers.push_back(block);
            }
        }
    }
}

} // namespace Shader::IR
