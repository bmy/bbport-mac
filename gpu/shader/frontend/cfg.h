// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the control flow graph of a GCN program. Blocks are the linear
// ranges between branch targets and after branches; a block ends in a branch (taken when its
// condition holds, else falling to branch_false) or in the end of the program. Runs of
// instructions under an EXEC mask set up in the block (a divergence scope) get blocks of their
// own, entered when any lane is active.
#pragma once

#include <iterator>
#include <span>
#include <vector>

#include "common/object_pool.h"
#include "common/types.h"
#include "gcn/isa.h"
#include "shader/ir/condition.h"

namespace Shader::IR {
class Block;
}

namespace Shader::Gcn {

using GcnIsa::GcnInst;

enum class EndClass {
    Branch, ///< ends in a branch (or falls through to the next block)
    Exit,   ///< ends the program
};

struct Block {
    u32 begin;       ///< byte address of the first instruction
    u32 end;         ///< byte address after the last one
    u32 begin_index; ///< instruction indices, both inclusive
    u32 end_index;
    u32 num_predecessors{};
    IR::Condition cond{}; ///< of branch_true
    GcnInst end_inst{};
    EndClass end_class{};
    Block* branch_true{};
    Block* branch_false{};
    IR::Block* ir_block{};
    bool is_dummy{}; ///< a block of the structurizer, not of the program
};

/// The blocks in address order, as references.
class BlockList {
public:
    class iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Block;
        using difference_type = std::ptrdiff_t;
        using pointer = Block*;
        using reference = Block&;

        iterator() = default;
        explicit iterator(std::vector<Block*>::const_iterator it_) : it{it_} {}
        Block& operator*() const {
            return **it;
        }
        Block* operator->() const {
            return *it;
        }
        iterator& operator++() {
            ++it;
            return *this;
        }
        iterator operator++(int) {
            return iterator{it++};
        }
        bool operator==(const iterator&) const = default;

    private:
        std::vector<Block*>::const_iterator it;
    };

    iterator begin() const {
        return iterator{blocks.begin()};
    }
    iterator end() const {
        return iterator{blocks.end()};
    }
    size_t size() const {
        return blocks.size();
    }

private:
    std::vector<Block*> blocks;
    friend class CFG;
};

class CFG {
public:
    CFG(Common::ObjectPool<Block>& block_pool, std::span<const GcnInst> inst_list);

    BlockList::iterator begin() const {
        return blocks.begin();
    }
    BlockList::iterator end() const {
        return blocks.end();
    }

    BlockList blocks;

private:
    void FindLabels();
    void MakeBlocks();
    void LinkBlocks();
    void SplitDivergenceScopes();
    void RemoveUnreachableBlocks();

    /// The index of the instruction at byte address `pc` (or of the first one after it).
    u32 IndexOf(u32 pc) const;
    Block& BlockAt(u32 pc);

    Common::ObjectPool<Block>& block_pool;
    std::span<const GcnInst> inst_list;
    std::vector<u32> index_to_pc; ///< byte address of each instruction, and of the end
    std::vector<u32> labels;      ///< block starts, sorted
};

} // namespace Shader::Gcn
