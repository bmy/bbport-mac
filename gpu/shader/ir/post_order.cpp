// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/post_order.h"

#include <unordered_set>

#include <boost/container/small_vector.hpp>

namespace Shader::IR {

BlockList PostOrder(Block* first_block) {
    // Depth first: a block is emitted once none of its successors is unvisited. Successors
    // are taken in order; a block whose successor was just pushed is pushed again under it.
    BlockList order;
    std::unordered_set<Block*> visited{first_block}; // membership only: order-free
    boost::container::small_vector<Block*, 16> stack{first_block};
    const auto first_visit = [&visited](Block* block) {
        return visited.insert(block).second;
    };
    while (!stack.empty()) {
        Block* const block = stack.back();
        stack.pop_back();
        bool descended = false;
        for (Block* successor : block->ImmSuccessors()) {
            if (first_visit(successor)) {
                stack.push_back(block);
                stack.push_back(successor);
                descended = true;
                break;
            }
        }
        if (!descended) {
            block->po_index = u32(order.size());
            order.push_back(block);
        }
    }
    return order;
}

} // namespace Shader::IR
