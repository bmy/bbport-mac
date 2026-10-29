// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: dominators and dominance frontiers (Cooper, Harvey and Kennedy,
// "A Simple, Fast Dominance Algorithm"), from a post order.
#pragma once

#include "shader/ir/basic_block.h"

namespace Shader::IR {

/// Sets immediate_dominator of every block (the entry dominates itself).
void ComputeDominators(const BlockList& post_order);
/// Fills dominance_frontiers (needs the dominators).
void ComputeDominanceFrontiers(const BlockList& post_order);

/// Whether `a` dominates `b`.
inline bool Dominates(Block* a, Block* b) {
    for (Block* current = b; current; current = current->immediate_dominator) {
        if (current == a) {
            return true;
        }
        if (current->immediate_dominator == current) {
            break;
        }
    }
    return false;
}

} // namespace Shader::IR
