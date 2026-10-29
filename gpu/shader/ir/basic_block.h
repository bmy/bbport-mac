// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: a basic block: its instructions, its edges, and what the passes
// keep per block (SSA values, dominators, post-order index).
#pragma once

#include <bit>
#include <initializer_list>
#include <map>
#include <span>
#include <string>
#include <vector>

#include <boost/intrusive/list.hpp>

#include "common/object_pool.h"
#include "common/types.h"
#include "shader/ir/microinstruction.h"
#include "shader/ir/reg.h"
#include "shader/ir/ssa.h"
#include "shader/ir/value.h"

namespace Shader::Gcn {
struct Block;
}

namespace Shader::IR {

class Block {
public:
    using InstructionList = boost::intrusive::list<Inst>;
    using size_type = InstructionList::size_type;
    using iterator = InstructionList::iterator;
    using const_iterator = InstructionList::const_iterator;
    using reverse_iterator = InstructionList::reverse_iterator;
    using const_reverse_iterator = InstructionList::const_reverse_iterator;

    explicit Block(Common::ObjectPool<Inst>& inst_pool_);
    ~Block();

    Block(const Block&) = delete;
    Block& operator=(const Block&) = delete;
    Block(Block&&) = default;
    Block& operator=(Block&&) = default;

    void AppendNewInst(Opcode op, std::initializer_list<Value> args);
    /// A new instruction before `insertion_point`.
    iterator PrependNewInst(iterator insertion_point, const Inst& base_inst);
    iterator PrependNewInst(iterator insertion_point, Opcode op,
                            std::initializer_list<Value> args = {}, u32 flags = 0);

    /// An edge to `block`.
    void AddBranch(Block* block);

    [[nodiscard]] InstructionList& Instructions() noexcept {
        return instructions;
    }
    [[nodiscard]] const InstructionList& Instructions() const noexcept {
        return instructions;
    }
    [[nodiscard]] std::span<Block* const> ImmPredecessors() const noexcept {
        return imm_predecessors;
    }
    [[nodiscard]] std::span<Block* const> ImmSuccessors() const noexcept {
        return imm_successors;
    }

    template <typename T>
    void SetDefinition(T def) {
        definition = std::bit_cast<u32>(def);
    }
    template <typename T>
    [[nodiscard]] T Definition() const noexcept {
        return std::bit_cast<T>(definition);
    }

    [[nodiscard]] bool empty() const {
        return instructions.empty();
    }
    [[nodiscard]] size_type size() const {
        return instructions.size();
    }
    [[nodiscard]] Inst& front() {
        return instructions.front();
    }
    [[nodiscard]] const Inst& front() const {
        return instructions.front();
    }
    [[nodiscard]] Inst& back() {
        return instructions.back();
    }
    [[nodiscard]] const Inst& back() const {
        return instructions.back();
    }
    [[nodiscard]] iterator begin() {
        return instructions.begin();
    }
    [[nodiscard]] const_iterator begin() const {
        return instructions.cbegin();
    }
    [[nodiscard]] iterator end() {
        return instructions.end();
    }
    [[nodiscard]] const_iterator end() const {
        return instructions.cend();
    }
    [[nodiscard]] reverse_iterator rbegin() {
        return instructions.rbegin();
    }
    [[nodiscard]] const_reverse_iterator rbegin() const {
        return instructions.rbegin();
    }
    [[nodiscard]] reverse_iterator rend() {
        return instructions.rend();
    }
    [[nodiscard]] const_reverse_iterator rend() const {
        return instructions.rend();
    }
    [[nodiscard]] const_iterator cbegin() const {
        return instructions.cbegin();
    }
    [[nodiscard]] const_iterator cend() const {
        return instructions.cend();
    }
    [[nodiscard]] const_reverse_iterator crbegin() const {
        return instructions.crbegin();
    }
    [[nodiscard]] const_reverse_iterator crend() const {
        return instructions.crend();
    }

    SsaState ssa_state;
    const Shader::Gcn::Block* cfg_block{}; ///< the program's block it came from (if any)
    U1 branch_cond{};

    Common::ObjectPool<Inst>* inst_pool;
    InstructionList instructions;

    Block* immediate_dominator{};
    std::vector<Block*> dominance_frontiers;
    u32 po_index{}; ///< position in the post order

    std::vector<Block*> imm_predecessors;
    std::vector<Block*> imm_successors;

    u32 definition{}; ///< the backend's label
};

using BlockList = std::vector<Block*>;

/// Text forms of a block for shader dumps.
[[nodiscard]] std::string DumpBlock(const Block& block);
[[nodiscard]] std::string DumpBlock(const Block& block,
                                    const std::map<const Block*, size_t>& block_to_index,
                                    std::map<const Inst*, size_t>& inst_to_index,
                                    size_t& inst_index);

} // namespace Shader::IR
