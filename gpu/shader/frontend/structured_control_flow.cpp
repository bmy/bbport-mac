// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Structurization: the CFG becomes a statement tree with gotos (each block a labelled statement,
// branches gotos to labels), the gotos are eliminated by the transformations of Erosa and
// Hendren, "Taming Control Flow: A Structured Approach to Eliminating Goto Statements" (ICCL
// 1994): outward and inward movement and lifting until goto and label are siblings, then a
// conditional (forward) or a loop (backward), with goto variables carrying the conditions. The
// tree is then emitted as the abstract syntax list (if/loop/break/return nodes over IR blocks)
// the SPIR-V backend writes as structured control flow.
#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/intrusive/list.hpp>
#include <fmt/format.h>

#include "shader/frontend/structured_control_flow.h"
#include "shader/ir/ir_emitter.h"
#include "shader/recompiler.h"

namespace Shader::Gcn {

namespace {

struct Statement;

// normal_link: the tree is not destroyed in order.
using ListBaseHook =
    boost::intrusive::list_base_hook<boost::intrusive::link_mode<boost::intrusive::normal_link>>;
using Tree = boost::intrusive::list<Statement, boost::intrusive::base_hook<ListBaseHook>,
                                    boost::intrusive::constant_time_size<false>>;
using Node = Tree::iterator;

enum class StatementType {
    Code,        ///< a CFG block
    Goto,        ///< if (cond) goto label
    Label,
    If,          ///< if (cond) { children }
    Loop,        ///< do { children } while (cond)
    Break,       ///< if (cond) break
    Return,
    Unreachable,
    Function,    ///< the root
    // Conditions:
    Identity,    ///< a branch condition of the program
    Not,
    Or,
    SetVariable, ///< goto_L<id> = op
    Variable,    ///< goto_L<id>
};

bool HasChildren(StatementType type) {
    return type == StatementType::If || type == StatementType::Loop ||
           type == StatementType::Function;
}

// Constructor tags.
struct Goto {};
struct Label {};
struct If {};
struct Loop {};
struct Break {};
struct Return {};
struct Unreachable {};
struct FunctionTag {};
struct Identity {};
struct Not {};
struct Or {};
struct SetVariable {};
struct Variable {};

struct Statement : ListBaseHook {
    Statement(const Block* block_, Statement* up_)
        : block{block_}, up{up_}, type{StatementType::Code} {}
    Statement(Goto, Statement* cond_, Node label_, Statement* up_)
        : label{label_}, cond{cond_}, up{up_}, type{StatementType::Goto} {}
    Statement(Label, u32 id_, Statement* up_) : id{id_}, up{up_}, type{StatementType::Label} {}
    Statement(If, Statement* cond_, Tree&& children_, Statement* up_)
        : children{std::move(children_)}, cond{cond_}, up{up_}, type{StatementType::If} {}
    Statement(Loop, Statement* cond_, Tree&& children_, Statement* up_)
        : children{std::move(children_)}, cond{cond_}, up{up_}, type{StatementType::Loop} {}
    Statement(Break, Statement* cond_, Statement* up_)
        : cond{cond_}, up{up_}, type{StatementType::Break} {}
    Statement(Return, Statement* up_) : up{up_}, type{StatementType::Return} {}
    Statement(Unreachable, Statement* up_) : up{up_}, type{StatementType::Unreachable} {}
    Statement(FunctionTag) : children{}, type{StatementType::Function} {}
    Statement(Identity, IR::Value cond_ref_, IR::Condition guest_cond_, Statement* up_)
        : cond_ref{cond_ref_}, guest_cond{guest_cond_}, up{up_}, type{StatementType::Identity} {}
    Statement(Not, Statement* op_, Statement* up_) : op{op_}, up{up_}, type{StatementType::Not} {}
    Statement(Or, Statement* op_a_, Statement* op_b_, Statement* up_)
        : op_a{op_a_}, op_b{op_b_}, up{up_}, type{StatementType::Or} {}
    Statement(SetVariable, u32 id_, Statement* op_, Statement* up_)
        : op{op_}, id{id_}, up{up_}, type{StatementType::SetVariable} {}
    Statement(Variable, u32 id_, Statement* up_)
        : id{id_}, up{up_}, type{StatementType::Variable} {}

    ~Statement() {
        if (HasChildren(type)) {
            std::destroy_at(&children);
        }
    }

    union {
        const Block* block;
        Node label;
        Tree children;
        IR::U1 cond_ref;
        Statement* op;
        Statement* op_a;
    };
    union {
        Statement* cond;
        Statement* op_b;
        u32 id;
        IR::Condition guest_cond;
    };
    Statement* up{};
    StatementType type;
};

/// Debug printing of a condition and of a tree.
[[maybe_unused]] std::string DumpExpr(const Statement* stmt) {
    switch (stmt->type) {
    case StatementType::Identity:
        return fmt::format("{}", stmt->guest_cond);
    case StatementType::Not:
        return fmt::format("!{}", DumpExpr(stmt->op));
    case StatementType::Or:
        return fmt::format("{} || {}", DumpExpr(stmt->op_a), DumpExpr(stmt->op_b));
    case StatementType::Variable:
        return fmt::format("goto_L{}", stmt->id);
    default:
        return "<invalid type>";
    }
}

[[maybe_unused]] std::string DumpTree(const Tree& tree, u32 indentation = 0) {
    std::string out;
    const std::string indent(indentation, ' ');
    for (const auto& stmt : tree) {
        switch (stmt.type) {
        case StatementType::Code:
            out += fmt::format("{}    Block {:04x} -> {:04x};\n", indent, stmt.block->begin,
                               stmt.block->end);
            break;
        case StatementType::Goto:
            out += fmt::format("{}    if ({}) goto L{};\n", indent, DumpExpr(stmt.cond),
                               stmt.label->id);
            break;
        case StatementType::Label:
            out += fmt::format("{}L{}:\n", indent, stmt.id);
            break;
        case StatementType::If:
            out += fmt::format("{}    if ({}) {{\n", indent, DumpExpr(stmt.cond));
            out += DumpTree(stmt.children, indentation + 4);
            out += fmt::format("{}    }}\n", indent);
            break;
        case StatementType::Loop:
            out += fmt::format("{}    do {{\n", indent);
            out += DumpTree(stmt.children, indentation + 4);
            out += fmt::format("{}    }} while ({});\n", indent, DumpExpr(stmt.cond));
            break;
        case StatementType::Break:
            out += fmt::format("{}    if ({}) break;\n", indent, DumpExpr(stmt.cond));
            break;
        case StatementType::Return:
            out += fmt::format("{}    return;\n", indent);
            break;
        case StatementType::Unreachable:
            out += fmt::format("{}    unreachable;\n", indent);
            break;
        case StatementType::SetVariable:
            out += fmt::format("{}    goto_L{} = {};\n", indent, stmt.id, DumpExpr(stmt.op));
            break;
        default:
            UNREACHABLE_MSG("Statement can't be printed");
        }
    }
    return out;
}

size_t Level(Node stmt) {
    size_t level{0};
    for (const Statement* node = stmt->up; node; node = node->up) {
        ++level;
    }
    return level;
}

/// One is in a statement that is a sibling of the other (their nesting paths meet one level
/// above the shallower).
bool IsDirectlyRelated(Node goto_stmt, Node label_stmt) {
    const size_t goto_level = Level(goto_stmt);
    const size_t label_level = Level(label_stmt);
    const bool label_shallower = label_level < goto_level;
    const size_t min_level = label_shallower ? label_level : goto_level;
    size_t max_level = label_shallower ? goto_level : label_level;
    const Node min = label_shallower ? label_stmt : goto_stmt;
    Node max = label_shallower ? goto_stmt : label_stmt;
    while (max_level > min_level) {
        --max_level;
        max = max->up;
    }
    return min->up == max->up;
}

bool IsIndirectlyRelated(Node goto_stmt, Node label_stmt) {
    return goto_stmt->up != label_stmt->up && !IsDirectlyRelated(goto_stmt, label_stmt);
}

/// The statement among `uncle`'s siblings that contains `nephew`.
Node SiblingFromNephew(Node uncle, Node nephew) noexcept {
    const Statement* const parent = uncle->up;
    Statement* it = &*nephew;
    while (it->up != parent) {
        it = it->up;
    }
    return Tree::s_iterator_to(*it);
}

/// left_sibling comes before right_sibling.
bool AreOrdered(Node left_sibling, Node right_sibling) noexcept {
    const Node end = right_sibling->up->children.end();
    for (auto it = right_sibling; it != end; ++it) {
        if (it == left_sibling) {
            return false;
        }
    }
    return true;
}

/// The statement holding the label comes before the goto (inward movement needs a lift first).
bool NeedsLift(Node goto_stmt, Node label_stmt) noexcept {
    return AreOrdered(SiblingFromNephew(goto_stmt, label_stmt), goto_stmt);
}

void CheckNoBreaks(const Tree& tree) {
    if (std::ranges::find(tree, StatementType::Break, &Statement::type) != tree.end()) {
        UNREACHABLE_MSG("Capturing statement with break nodes");
    }
}

/// Builds the goto tree of the CFG and eliminates its gotos (last first).
class GotoPass {
public:
    GotoPass(CFG& cfg, Common::ObjectPool<Statement>& pool_) : pool{pool_} {
        const std::vector<Node> gotos = BuildTree(cfg);
        for (auto it = gotos.rbegin(); it != gotos.rend(); ++it) {
            RemoveGoto(*it);
        }
    }

    Statement& RootStatement() noexcept {
        return root_stmt;
    }

private:
    void RemoveGoto(Node goto_stmt) {
        const Node label_stmt = goto_stmt->label;
        // Directly related first (outward movement)...
        if (IsIndirectlyRelated(goto_stmt, label_stmt)) {
            while (!IsDirectlyRelated(goto_stmt, label_stmt)) {
                goto_stmt = MoveOutward(goto_stmt);
            }
        }
        // ...then siblings.
        if (IsDirectlyRelated(goto_stmt, label_stmt)) {
            const size_t label_level = Level(label_stmt);
            size_t goto_level = Level(goto_stmt);
            if (goto_level > label_level) {
                for (; goto_level > label_level; --goto_level) {
                    goto_stmt = MoveOutward(goto_stmt);
                }
            } else {
                if (NeedsLift(goto_stmt, label_stmt)) {
                    goto_stmt = Lift(goto_stmt);
                }
                for (; goto_level < label_level; ++goto_level) {
                    goto_stmt = MoveInward(goto_stmt);
                }
            }
        }
        if (std::next(goto_stmt) == label_stmt) {
            goto_stmt->up->children.erase(goto_stmt);
        } else if (AreOrdered(goto_stmt, label_stmt)) {
            EliminateAsConditional(goto_stmt, label_stmt);
        } else {
            EliminateAsLoop(goto_stmt, label_stmt);
        }
    }

    std::vector<Node> BuildTree(CFG& cfg) {
        std::vector<Node> gotos;
        Statement* const false_stmt =
            pool.Create(Identity{}, IR::Value{false}, IR::Condition::False, &root_stmt);
        Tree& root = root_stmt.children;
        std::unordered_map<Block*, Node> labels;
        labels.reserve(cfg.blocks.size());
        u32 label_id{0};
        for (Block& block : cfg.blocks) {
            Statement* const label = pool.Create(Label{}, label_id++, &root_stmt);
            labels.emplace(&block, root.insert(root.end(), *label));
        }
        for (Block& block : cfg.blocks) {
            const Node label = labels.at(&block);
            const Node ip = std::next(label);
            // The block's goto variable starts false (at the top, and again after its label).
            const auto reset_variable = [&]() -> Statement& {
                return *pool.Create(SetVariable{}, label->id, false_stmt, &root_stmt);
            };
            root.push_front(reset_variable());
            root.insert(ip, reset_variable());
            root.insert(ip, *pool.Create(&block, &root_stmt));

            switch (block.end_class) {
            case EndClass::Branch: {
                Statement* const always =
                    pool.Create(Identity{}, IR::Value{true}, IR::Condition::True, &root_stmt);
                const auto add_goto = [&](Statement* cond, Block* target) {
                    gotos.push_back(
                        root.insert(ip, *pool.Create(Goto{}, cond, labels.at(target), &root_stmt)));
                };
                if (block.cond == IR::Condition::True) {
                    add_goto(always, block.branch_true);
                } else if (block.cond == IR::Condition::False) {
                    add_goto(always, block.branch_false);
                } else {
                    // The translator left the condition in a ConditionRef at the block's end.
                    const IR::Value value = block.ir_block->branch_cond;
                    ASSERT(!value.IsEmpty());
                    IR::Inst* const cond_ref = value.Inst();
                    Statement* const taken =
                        pool.Create(Identity{}, cond_ref->Arg(0), block.cond, &root_stmt);
                    cond_ref->Invalidate();
                    add_goto(taken, block.branch_true);
                    add_goto(always, block.branch_false);
                }
                break;
            }
            case EndClass::Exit:
                root.insert(ip, *pool.Create(Return{}, &root_stmt));
                break;
            }
        }
        return gotos;
    }

    void UpdateTreeUp(Statement* tree) {
        for (Statement& stmt : tree->children) {
            stmt.up = tree;
        }
    }

    /// goto L; S...; L:  ->  if (!cond) { S... } L:
    void EliminateAsConditional(Node goto_stmt, Node label_stmt) {
        Tree& body = goto_stmt->up->children;
        Tree if_body;
        if_body.splice(if_body.begin(), body, std::next(goto_stmt), label_stmt);
        Statement* const cond = pool.Create(Not{}, goto_stmt->cond, &root_stmt);
        Statement* const if_stmt = pool.Create(If{}, cond, std::move(if_body), goto_stmt->up);
        UpdateTreeUp(if_stmt);
        body.insert(goto_stmt, *if_stmt);
        body.erase(goto_stmt);
    }

    /// L: S...; goto L;  ->  do { L: S... } while (cond)
    void EliminateAsLoop(Node goto_stmt, Node label_stmt) {
        Tree& body = goto_stmt->up->children;
        Tree loop_body;
        loop_body.splice(loop_body.begin(), body, label_stmt, goto_stmt);
        Statement* const loop = pool.Create(Loop{}, goto_stmt->cond, std::move(loop_body),
                                            goto_stmt->up);
        UpdateTreeUp(loop);
        body.insert(goto_stmt, *loop);
        body.erase(goto_stmt);
    }

    [[nodiscard]] Node MoveOutward(Node goto_stmt) {
        switch (goto_stmt->up->type) {
        case StatementType::If:
            return MoveOutwardIf(goto_stmt);
        case StatementType::Loop:
            return MoveOutwardLoop(goto_stmt);
        default:
            UNREACHABLE_MSG("Invalid outward movement");
        }
    }

    /// Into the sibling statement holding the label: set the variable, skip the statements up
    /// to it if set, and goto on the variable at the start of its body (an if also enters on
    /// the variable).
    [[nodiscard]] Node MoveInward(Node goto_stmt) {
        Statement* const parent = goto_stmt->up;
        Tree& body = parent->children;
        const Node label = goto_stmt->label;
        const Node label_nested_stmt = SiblingFromNephew(goto_stmt, label);
        const u32 label_id = label->id;

        Statement* const set_var = pool.Create(SetVariable{}, label_id, goto_stmt->cond, parent);
        body.insert(goto_stmt, *set_var);

        Tree if_body;
        if_body.splice(if_body.begin(), body, std::next(goto_stmt), label_nested_stmt);
        Statement* const variable = pool.Create(Variable{}, label_id, &root_stmt);
        Statement* const neg_var = pool.Create(Not{}, variable, &root_stmt);
        if (!if_body.empty()) {
            Statement* const if_stmt = pool.Create(If{}, neg_var, std::move(if_body), parent);
            UpdateTreeUp(if_stmt);
            body.insert(goto_stmt, *if_stmt);
        }
        body.erase(goto_stmt);

        switch (label_nested_stmt->type) {
        case StatementType::If:
            label_nested_stmt->cond =
                pool.Create(Or{}, variable, label_nested_stmt->cond, &root_stmt);
            break;
        case StatementType::Loop:
            break;
        default:
            UNREACHABLE_MSG("Invalid inward movement");
        }
        Tree& nested_tree = label_nested_stmt->children;
        Statement* const new_goto = pool.Create(Goto{}, variable, label, &*label_nested_stmt);
        return nested_tree.insert(nested_tree.begin(), *new_goto);
    }

    /// A goto after the statement holding its label: a loop from that statement to the goto,
    /// entered again while the variable is set, with the goto at its start.
    [[nodiscard]] Node Lift(Node goto_stmt) {
        Statement* const parent = goto_stmt->up;
        Tree& body = parent->children;
        const Node label = goto_stmt->label;
        const u32 label_id = label->id;
        const Node label_nested_stmt = SiblingFromNephew(goto_stmt, label);

        Tree loop_body;
        loop_body.splice(loop_body.begin(), body, label_nested_stmt, goto_stmt);
        CheckNoBreaks(loop_body);
        Statement* const variable = pool.Create(Variable{}, label_id, &root_stmt);
        Statement* const loop_stmt = pool.Create(Loop{}, variable, std::move(loop_body), parent);
        UpdateTreeUp(loop_stmt);
        body.insert(goto_stmt, *loop_stmt);

        Statement* const new_goto = pool.Create(Goto{}, variable, label, loop_stmt);
        loop_stmt->children.push_front(*new_goto);
        const Node new_goto_node = loop_stmt->children.begin();

        Statement* const set_var = pool.Create(SetVariable{}, label_id, goto_stmt->cond, loop_stmt);
        loop_stmt->children.push_back(*set_var);

        body.erase(goto_stmt);
        return new_goto_node;
    }

    /// Out of an if: set the variable, skip the rest of the if body if set, goto on the
    /// variable after the if.
    Node MoveOutwardIf(Node goto_stmt) {
        const Node parent = Tree::s_iterator_to(*goto_stmt->up);
        Tree& body = parent->children;
        const u32 label_id = goto_stmt->label->id;
        Statement* const set_goto_var =
            pool.Create(SetVariable{}, label_id, goto_stmt->cond, &*parent);
        body.insert(goto_stmt, *set_goto_var);

        Tree if_body;
        if_body.splice(if_body.begin(), body, std::next(goto_stmt), body.end());
        if_body.pop_front();
        Statement* const cond = pool.Create(Variable{}, label_id, &root_stmt);
        Statement* const neg_cond = pool.Create(Not{}, cond, &root_stmt);
        Statement* const if_stmt = pool.Create(If{}, neg_cond, std::move(if_body), &*parent);
        UpdateTreeUp(if_stmt);
        body.insert(goto_stmt, *if_stmt);
        body.erase(goto_stmt);

        Statement* const new_cond = pool.Create(Variable{}, label_id, &root_stmt);
        Statement* const new_goto = pool.Create(Goto{}, new_cond, goto_stmt->label, parent->up);
        Tree& parent_tree = parent->up->children;
        return parent_tree.insert(std::next(parent), *new_goto);
    }

    /// Out of a loop: set the variable and break, goto on the variable after the loop.
    Node MoveOutwardLoop(Node goto_stmt) {
        Statement* const parent = goto_stmt->up;
        Tree& body = parent->children;
        const u32 label_id = goto_stmt->label->id;
        Statement* const set_goto_var =
            pool.Create(SetVariable{}, label_id, goto_stmt->cond, parent);
        Statement* const cond = pool.Create(Variable{}, label_id, &root_stmt);
        Statement* const break_stmt = pool.Create(Break{}, cond, parent);
        body.insert(goto_stmt, *set_goto_var);
        body.insert(goto_stmt, *break_stmt);
        body.erase(goto_stmt);

        const Node loop = Tree::s_iterator_to(*goto_stmt->up);
        Statement* const new_goto_cond = pool.Create(Variable{}, label_id, &root_stmt);
        Statement* const new_goto =
            pool.Create(Goto{}, new_goto_cond, goto_stmt->label, loop->up);
        Tree& parent_tree = loop->up->children;
        return parent_tree.insert(std::next(loop), *new_goto);
    }

    Common::ObjectPool<Statement>& pool;
    Statement root_stmt{FunctionTag{}};
};

/// The first code statement from `node` before any nested statement.
[[nodiscard]] Statement* TryFindForwardBlock(Node node, Node end) {
    for (; node != end && !HasChildren(node->type); ++node) {
        if (node->type == StatementType::Code) {
            return &*node;
        }
    }
    return nullptr;
}

[[nodiscard]] IR::U1 VisitExpr(IR::IREmitter& ir, const Statement& stmt) {
    switch (stmt.type) {
    case StatementType::Identity:
        return stmt.cond_ref;
    case StatementType::Not:
        return ir.LogicalNot(VisitExpr(ir, *stmt.op));
    case StatementType::Or:
        return ir.LogicalOr(VisitExpr(ir, *stmt.op_a), VisitExpr(ir, *stmt.op_b));
    case StatementType::Variable:
        return ir.GetGotoVariable(stmt.id);
    default:
        UNREACHABLE_MSG("Statement type {}", u32(stmt.type));
    }
}

/// Emits the structured tree as the abstract syntax list, making the blocks loops and ifs need
/// (headers, continue blocks, merges, returns) and the edges between blocks.
class TranslatePass {
public:
    TranslatePass(Pools& pools_, Common::ObjectPool<Statement>& stmt_pool_,
                  Common::ObjectPool<Block>& flow_blocks_, Statement& root_stmt,
                  IR::AbstractSyntaxList& syntax_list_)
        : pools{pools_}, stmt_pool{stmt_pool_}, flow_blocks{flow_blocks_},
          syntax_list{syntax_list_} {
        Visit(root_stmt, nullptr, nullptr);
    }

private:
    IR::Block* NewBlock() {
        return pools.block_pool.Create(pools.inst_pool);
    }

    void AddBlockNode(IR::Block* block) {
        auto& node = syntax_list.emplace_back();
        node.type = IR::AbstractSyntaxNode::Type::Block;
        node.data.block = block;
    }

    void Visit(Statement& parent, IR::Block* break_block, IR::Block* fallthrough_block) {
        IR::Block* current_block{};
        IR::Block::iterator insert_point{};
        Tree& tree = parent.children;
        // Code before the first statement goes to the first code block, or a new one.
        if (Statement* const code_stmt = TryFindForwardBlock(tree.begin(), tree.end())) {
            current_block = code_stmt->block->ir_block;
            insert_point = current_block->begin();
        } else {
            current_block = NewBlock();
            insert_point = current_block->end();
            AddBlockNode(current_block);
        }
        for (Statement& stmt : tree) {
            switch (stmt.type) {
            case StatementType::Label:
                break;
            case StatementType::Code: {
                current_block = stmt.block->ir_block;
                insert_point = current_block->end();
                for (const auto& node : syntax_list) {
                    if (node.type == IR::AbstractSyntaxNode::Type::Block) {
                        ASSERT(node.data.block != current_block);
                    }
                }
                AddBlockNode(current_block);
                break;
            }
            case StatementType::SetVariable: {
                IR::IREmitter ir{*current_block, insert_point};
                ir.SetGotoVariable(stmt.id, VisitExpr(ir, *stmt.op));
                break;
            }
            case StatementType::If: {
                IR::Block* const merge_block = MergeBlock(parent, stmt);
                IR::IREmitter ir{*current_block, insert_point};
                const IR::U1 cond{ir.ConditionRef(VisitExpr(ir, *stmt.cond))};
                const size_t if_node_index = syntax_list.size();
                syntax_list.emplace_back();

                const size_t then_block_index = syntax_list.size();
                Visit(stmt, break_block, merge_block);

                IR::Block* const then_block = syntax_list.at(then_block_index).data.block;
                current_block->AddBranch(then_block);
                current_block->AddBranch(merge_block);
                current_block = merge_block;
                insert_point = current_block->begin();

                auto& if_node = syntax_list[if_node_index];
                if_node.type = IR::AbstractSyntaxNode::Type::If;
                if_node.data.if_node.cond = cond;
                if_node.data.if_node.body = then_block;
                if_node.data.if_node.merge = merge_block;

                auto& endif_node = syntax_list.emplace_back();
                endif_node.type = IR::AbstractSyntaxNode::Type::EndIf;
                endif_node.data.end_if.merge = merge_block;
                break;
            }
            case StatementType::Loop: {
                IR::Block* const header = NewBlock();
                if (current_block) {
                    current_block->AddBranch(header);
                }
                AddBlockNode(header);
                IR::Block* const continue_block = NewBlock();
                IR::Block* const merge_block = MergeBlock(parent, stmt);
                const size_t loop_node_index = syntax_list.size();
                syntax_list.emplace_back();

                const size_t body_block_index = syntax_list.size();
                Visit(stmt, merge_block, continue_block);

                // The condition is evaluated in the continue block, at the loop's end.
                IR::IREmitter ir{*continue_block};
                const IR::U1 cond{ir.ConditionRef(VisitExpr(ir, *stmt.cond))};
                IR::Block* const body_block = syntax_list.at(body_block_index).data.block;
                header->AddBranch(body_block);
                continue_block->AddBranch(header);
                continue_block->AddBranch(merge_block);
                current_block = merge_block;
                insert_point = current_block->begin();

                auto& loop = syntax_list[loop_node_index];
                loop.type = IR::AbstractSyntaxNode::Type::Loop;
                loop.data.loop.body = body_block;
                loop.data.loop.continue_block = continue_block;
                loop.data.loop.merge = merge_block;

                AddBlockNode(continue_block);
                auto& repeat = syntax_list.emplace_back();
                repeat.type = IR::AbstractSyntaxNode::Type::Repeat;
                repeat.data.repeat.cond = cond;
                repeat.data.repeat.loop_header = header;
                repeat.data.repeat.merge = merge_block;
                break;
            }
            case StatementType::Break: {
                IR::Block* const skip_block = MergeBlock(parent, stmt);
                IR::IREmitter ir{*current_block, insert_point};
                const IR::U1 cond{ir.ConditionRef(VisitExpr(ir, *stmt.cond))};
                current_block->AddBranch(break_block);
                current_block->AddBranch(skip_block);
                current_block = skip_block;
                insert_point = current_block->end();

                auto& break_node = syntax_list.emplace_back();
                break_node.type = IR::AbstractSyntaxNode::Type::Break;
                break_node.data.break_node.cond = cond;
                break_node.data.break_node.merge = break_block;
                break_node.data.break_node.skip = skip_block;
                break;
            }
            case StatementType::Return: {
                IR::Block* const return_block = NewBlock();
                IR::IREmitter{*return_block}.Epilogue();
                current_block->AddBranch(return_block);
                AddBlockNode(return_block);
                current_block = nullptr;
                insert_point = {};
                syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Return;
                break;
            }
            case StatementType::Unreachable:
                current_block = nullptr;
                insert_point = {};
                syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Unreachable;
                break;
            default:
                UNREACHABLE_MSG("Statement type {}", u32(stmt.type));
            }
        }
        if (current_block) {
            if (fallthrough_block) {
                current_block->AddBranch(fallthrough_block);
            } else {
                syntax_list.emplace_back().type = IR::AbstractSyntaxNode::Type::Unreachable;
            }
        }
    }

    /// The block after `stmt`: the next code statement's, or a new empty block (a dummy flow
    /// block inserted as a code statement).
    IR::Block* MergeBlock(Statement& parent, Statement& stmt) {
        Tree& tree = stmt.up->children;
        Statement* merge_stmt = TryFindForwardBlock(std::next(Tree::s_iterator_to(stmt)), tree.end());
        if (!merge_stmt) {
            Block* const flow_block = flow_blocks.Create();
            flow_block->is_dummy = true;
            IR::Block* const ir_block = NewBlock();
            flow_block->ir_block = ir_block;
            ir_block->cfg_block = flow_block;
            merge_stmt = stmt_pool.Create(flow_block, &parent);
            parent.children.insert(std::next(Tree::s_iterator_to(stmt)), *merge_stmt);
        }
        return merge_stmt->block->ir_block;
    }

    Pools& pools;
    Common::ObjectPool<Statement>& stmt_pool;
    Common::ObjectPool<Block>& flow_blocks;
    IR::AbstractSyntaxList& syntax_list;
};

} // namespace

IR::AbstractSyntaxList BuildASL(Pools& pools, CFG& cfg, Common::ObjectPool<Block>& flow_blocks) {
    Common::ObjectPool<Statement> stmt_pool{64};
    GotoPass goto_pass{cfg, stmt_pool};
    IR::AbstractSyntaxList syntax_list;
    TranslatePass{pools, stmt_pool, flow_blocks, goto_pass.RootStatement(), syntax_list};
    return syntax_list;
}

} // namespace Shader::Gcn
