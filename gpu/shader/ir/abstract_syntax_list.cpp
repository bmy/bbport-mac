// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/abstract_syntax_list.h"

#include <fmt/format.h>

namespace Shader::IR {

std::string DumpASLNode(const AbstractSyntaxNode& node,
                        const std::map<const Block*, size_t>& block_to_index,
                        const std::map<const Inst*, size_t>& inst_to_index) {
    const auto block = [&](const Block* b) {
        const auto it = block_to_index.find(b);
        return it != block_to_index.end() ? fmt::format("${}", it->second) : std::string{"$?"};
    };
    const auto cond = [&](const U1& value) {
        if (value.IsImmediate()) {
            return std::string{value.U1() ? "true" : "false"};
        }
        const auto it = inst_to_index.find(value.Inst());
        return it != inst_to_index.end() ? fmt::format("%{}", it->second) : std::string{"%?"};
    };
    const auto& d = node.data;
    switch (node.type) {
    case AbstractSyntaxNode::Type::Block:
        return "Block: " + block(d.block);
    case AbstractSyntaxNode::Type::If:
        return fmt::format("If: cond = {}, body = {}, merge = {}", cond(d.if_node.cond),
                           block(d.if_node.body), block(d.if_node.merge));
    case AbstractSyntaxNode::Type::EndIf:
        return "EndIf: merge = " + block(d.end_if.merge);
    case AbstractSyntaxNode::Type::Loop:
        return fmt::format("Loop: body = {}, continue = {}, merge = {}", block(d.loop.body),
                           block(d.loop.continue_block), block(d.loop.merge));
    case AbstractSyntaxNode::Type::Repeat:
        return fmt::format("Repeat: cond = {}, header = {}, merge = {}", cond(d.repeat.cond),
                           block(d.repeat.loop_header), block(d.repeat.merge));
    case AbstractSyntaxNode::Type::Break:
        return fmt::format("Break: cond = {}, merge = {}, skip = {}", cond(d.break_node.cond),
                           block(d.break_node.merge), block(d.break_node.skip));
    case AbstractSyntaxNode::Type::Return:
        return "Return";
    case AbstractSyntaxNode::Type::Unreachable:
        return "Unreachable";
    }
    return "?";
}

} // namespace Shader::IR
