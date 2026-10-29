// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: the structured program as a list of nodes (blocks and the
// if/loop/break constructs around them), as the SPIR-V backend emits it.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "shader/ir/value.h"

namespace Shader::IR {

class Block;

struct AbstractSyntaxNode {
    enum class Type {
        Block,
        If,
        EndIf,
        Loop,
        Repeat,
        Break,
        Return,
        Unreachable,
    };
    union Data {
        Block* block;
        struct {
            U1 cond;
            Block* body;
            Block* merge;
        } if_node;
        struct {
            Block* merge;
        } end_if;
        struct {
            Block* body;
            Block* continue_block;
            Block* merge;
        } loop;
        struct {
            U1 cond;
            Block* loop_header;
            Block* merge;
        } repeat;
        struct {
            U1 cond;
            Block* merge;
            Block* skip;
        } break_node;
    };
    Data data{};
    Type type{};
};
using AbstractSyntaxList = std::vector<AbstractSyntaxNode>;

/// A node's text for shader dumps.
std::string DumpASLNode(const AbstractSyntaxNode& node,
                        const std::map<const Block*, size_t>& block_to_index,
                        const std::map<const Inst*, size_t>& inst_to_index);

} // namespace Shader::IR
