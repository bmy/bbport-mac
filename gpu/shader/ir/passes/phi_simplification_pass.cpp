// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Simplifies networks of phis, taken as strongly connected components (Tarjan):
//  - an SCC whose outside operands all come from the same one-operand instruction (a conversion,
//    a component extract) merges the operands instead, and the instruction is applied once after
//    the phis; this may make the SCCs that use it foldable in turn;
//  - SCCs of the same shape (phis in the same blocks, the same internal edges) with the same
//    outside operands are duplicates: the uses of one take the other, which may make the SCCs
//    using them duplicates in turn.
#include <algorithm>
#include <functional>
#include <limits>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/container/small_vector.hpp>

#include "common/assert.h"
#include "common/hash.h"
#include "shader/frontend/cfg.h"
#include "shader/ir/passes/expression_elimination.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/program.h"

namespace Shader::Optimization {

namespace {

using IR::Opcode;
using InstList = boost::container::small_vector<IR::Inst*, 4>;

bool IsCompositeExtract(Opcode op) {
    return op == Opcode::CompositeExtractU32x2 || op == Opcode::CompositeExtractU32x3 ||
           op == Opcode::CompositeExtractU32x4;
}

/// The instruction applied to every outside operand of an SCC.
struct CommonOp {
    Opcode op{Opcode::Void};
    u32 extract_comp{};
};

bool InSameScc(const IR::Inst* arg, const IR::Inst* phi) {
    return arg->GetOpcode() == Opcode::Phi && arg->scc_index == phi->scc_index;
}

/// The index of `phi` in `phis`.
std::ptrdiff_t PositionIn(const InstList& phis, const IR::Inst* phi) {
    return std::distance(phis.begin(), std::ranges::find(phis, phi));
}

struct PhiScc {
    bool queued{};
    bool dead{};
    u64 hash{};
    InstList phis;

    /// Whether every outside operand is the same one-operand instruction (or the same component
    /// extract); InverseBallot stays (the ballot passes need it at the operand).
    bool FindCommonOp(CommonOp& out) const {
        out = {};
        for (const IR::Inst* const phi : phis) {
            for (size_t i = 0; i < phi->NumArgs(); ++i) {
                if (phi->Arg(i).IsImmediate()) {
                    return false;
                }
                const IR::Inst* const arg = phi->Arg(i).Inst();
                if (arg->GetOpcode() == Opcode::Phi) {
                    if (InSameScc(arg, phi)) {
                        continue;
                    }
                    return false;
                }
                const Opcode op = arg->GetOpcode();
                if (out.op == Opcode::Void) {
                    out.op = op;
                    if (IsCompositeExtract(op)) {
                        out.extract_comp = arg->Arg(1).U32();
                    } else if (arg->NumArgs() != 1 || op == Opcode::InverseBallot) {
                        return false;
                    }
                } else if (op != out.op ||
                           (IsCompositeExtract(op) && arg->Arg(1).U32() != out.extract_comp)) {
                    return false;
                }
            }
        }
        ASSERT(out.op != Opcode::Void);
        return true;
    }

    /// The phis of other SCCs using this one.
    InstList OutsidePhiUsers() const {
        InstList users;
        for (const IR::Inst* const phi : phis) {
            for (const auto& [user, operand] : phi->Uses()) {
                if (user->GetOpcode() == Opcode::Phi && user->scc_index != phi->scc_index) {
                    users.push_back(user);
                }
            }
        }
        return users;
    }

    /// Phis merge the operands' operands; the outside uses take the common op applied after the
    /// phis.
    void FoldCommonOp(const CommonOp& common) {
        const IR::Type type = IR::ArgTypeOf(common.op, 0);
        for (IR::Inst* const phi : phis) {
            for (size_t i = 0; i < phi->NumArgs(); ++i) {
                const IR::Inst* const arg = phi->Arg(i).Inst();
                if (arg->GetOpcode() != Opcode::Phi) {
                    phi->SetArg(i, arg->Arg(0));
                }
            }
            phi->SetRegTag(IR::EMPTY_REG_TAG);
            phi->SetFlags(type);

            IR::Block* const block = phi->GetParent();
            IR::Inst* replacement{};
            // (New uses go to the front: the replacement's own use is not visited.)
            const auto& uses = phi->Uses();
            for (auto it = uses.begin(); it != uses.end();) {
                const auto [user, operand] = *it++;
                if (InSameScc(user, phi)) {
                    continue;
                }
                if (!replacement) {
                    const auto after_phis = std::ranges::find_if_not(block->Instructions(), IR::IsPhi);
                    if (IsCompositeExtract(common.op)) {
                        replacement = &*block->PrependNewInst(
                            after_phis, common.op,
                            {IR::Value{phi}, IR::Value{common.extract_comp}});
                    } else {
                        replacement =
                            &*block->PrependNewInst(after_phis, common.op, {IR::Value{phi}});
                    }
                }
                user->SetArg(operand, IR::Value{replacement});
            }
        }
    }

    /// A hash of the shape: phi count, blocks (by program address, deterministic), immediates,
    /// and the internal edges.
    void ComputeShapeHash() {
        hash = HashCombine(0ull, phis.size());
        for (const IR::Inst* const phi : phis) {
            const IR::Block* const block = phi->GetParent();
            hash = HashCombine(hash, block->cfg_block ? u64(block->cfg_block->begin)
                                                      : u64(block->po_index));
            for (size_t i = 0; i < phi->NumArgs(); ++i) {
                const IR::Value arg = phi->Arg(i);
                hash = HashCombine(hash, arg.IsImmediate());
                if (arg.IsImmediate()) {
                    hash = HashCombine(hash, arg.Hash());
                    continue;
                }
                const bool internal = InSameScc(arg.Inst(), phi);
                hash = HashCombine(hash, internal);
                if (internal) {
                    hash = HashCombine(hash, PositionIn(phis, arg.Inst()));
                }
            }
        }
    }

    /// The same shape with the same outside operands.
    bool IsSameAs(const PhiScc& other) const {
        if (phis.size() != other.phis.size()) {
            return false;
        }
        for (size_t i = 0; i < phis.size(); ++i) {
            const IR::Inst* const a = phis[i];
            const IR::Inst* const b = other.phis[i];
            if (a->GetParent() != b->GetParent()) {
                return false;
            }
            for (size_t j = 0; j < a->NumArgs(); ++j) {
                const IR::Value value_a = a->Arg(j);
                const IR::Value value_b = b->Arg(j);
                if (value_a.IsImmediate() != value_b.IsImmediate()) {
                    return false;
                }
                if (value_a.IsImmediate()) {
                    if (value_a.Hash() != value_b.Hash()) {
                        return false;
                    }
                    continue;
                }
                const IR::Inst* const arg_a = value_a.Inst();
                const IR::Inst* const arg_b = value_b.Inst();
                if (arg_a->GetOpcode() != arg_b->GetOpcode()) {
                    return false;
                }
                if (arg_a->GetOpcode() == Opcode::Phi) {
                    if (arg_a->Flags<IR::Type>() != arg_b->Flags<IR::Type>()) {
                        return false;
                    }
                    const bool internal = arg_a->scc_index == a->scc_index;
                    if (internal != (arg_b->scc_index == b->scc_index)) {
                        return false;
                    }
                    if (internal) {
                        if (PositionIn(phis, arg_a) != PositionIn(other.phis, arg_b)) {
                            return false;
                        }
                        continue;
                    }
                }
                if (arg_a != arg_b) {
                    return false;
                }
            }
        }
        return true;
    }
};

/// Tarjan's algorithm over the phi graph (iterative), numbering the phis' scc_index.
class SccFinder {
public:
    explicit SccFinder(std::span<IR::Inst* const> all_phis) {
        nodes.reserve(all_phis.size());
        for (IR::Inst* const phi : all_phis) {
            if (nodes[phi].index == Unvisited) {
                Visit(phi);
            }
        }
        for (u32 i = 0; i < sccs.size(); i++) {
            for (IR::Inst* const phi : sccs[i].phis) {
                phi->scc_index = i;
            }
        }
    }

    std::vector<PhiScc> sccs;

private:
    static constexpr u32 Unvisited = std::numeric_limits<u32>::max();

    struct Node {
        u32 index{Unvisited};
        u32 lowlink{Unvisited};
        bool on_stack{};
    };
    struct Frame {
        IR::Inst* phi;
        size_t next_arg;
    };

    void Push(IR::Inst* phi, Node& node) {
        node.index = counter;
        node.lowlink = counter++;
        node.on_stack = true;
        stack.push_back(phi);
        dfs.push_back({phi, 0});
    }

    void Visit(IR::Inst* root) {
        Push(root, nodes[root]);
        while (!dfs.empty()) {
            auto& [phi, next_arg] = dfs.back();
            Node& node = nodes[phi];
            bool descended = false;
            for (; next_arg < phi->NumArgs(); next_arg++) {
                const IR::Value value = phi->Arg(next_arg);
                if (value.IsImmediate() || value.Inst()->GetOpcode() != Opcode::Phi) {
                    continue;
                }
                IR::Inst* const arg = value.Inst();
                Node& arg_node = nodes[arg];
                if (arg_node.index == Unvisited) {
                    next_arg++;
                    Push(arg, arg_node);
                    descended = true;
                    break;
                }
                if (arg_node.on_stack) {
                    node.lowlink = std::min(node.lowlink, arg_node.index);
                }
            }
            if (descended) {
                continue;
            }
            if (node.lowlink == node.index) {
                PhiScc& scc = sccs.emplace_back();
                IR::Inst* member;
                do {
                    member = stack.back();
                    stack.pop_back();
                    nodes[member].on_stack = false;
                    scc.phis.push_back(member);
                } while (member != phi);
            }
            const u32 lowlink = node.lowlink;
            dfs.pop_back();
            if (!dfs.empty()) {
                Node& parent = nodes[dfs.back().phi];
                parent.lowlink = std::min(parent.lowlink, lowlink);
            }
        }
    }

    std::unordered_map<IR::Inst*, Node> nodes;
    InstList stack;
    boost::container::small_vector<Frame, 8> dfs;
    u32 counter{0};
};

void FoldCommonOps(std::vector<PhiScc>& sccs) {
    std::vector<u32> worklist;
    worklist.reserve(sccs.size());
    for (u32 i = 0; i < sccs.size(); i++) {
        worklist.push_back(i);
        sccs[i].queued = true;
    }
    while (!worklist.empty()) {
        PhiScc& scc = sccs[worklist.back()];
        worklist.pop_back();
        scc.queued = false;

        CommonOp op;
        if (!scc.FindCommonOp(op)) {
            continue;
        }
        const InstList users = scc.OutsidePhiUsers();
        do {
            scc.FoldCommonOp(op);
        } while (scc.FindCommonOp(op));
        for (IR::Inst* const user : users) {
            if (!std::exchange(sccs[user->scc_index].queued, true)) {
                worklist.push_back(user->scc_index);
            }
        }
    }
}

void Deduplicate(std::vector<PhiScc>& sccs) {
    struct Group {
        boost::container::small_vector<u32, 4> sccs;
        bool queued{};
    };
    // Keyed by the shape hash itself; the worklist follows this map's order.
    std::unordered_map<u64, Group, std::identity> groups;
    for (size_t i = 0; i < sccs.size(); ++i) {
        sccs[i].ComputeShapeHash();
        groups[sccs[i].hash].sccs.emplace_back(i);
    }
    std::vector<Group*> worklist;
    worklist.reserve(groups.size());
    for (auto& [hash, group] : groups) {
        if (group.sccs.size() > 1) {
            worklist.push_back(&group);
            group.queued = true;
        }
    }

    // The uses of `dup` take `keep`; SCCs using `dup` may have become duplicates.
    const auto replace = [&](const PhiScc& keep, const PhiScc& dup) {
        for (size_t k = 0; k < dup.phis.size(); k++) {
            IR::Inst* const old_phi = dup.phis[k];
            const auto uses = old_phi->Uses();
            for (const auto& [user, operand] : uses) {
                if (user->GetOpcode() == Opcode::Phi) {
                    if (user->scc_index == old_phi->scc_index) {
                        continue;
                    }
                    Group& user_group = groups[sccs[user->scc_index].hash];
                    if (!std::exchange(user_group.queued, true)) {
                        worklist.push_back(&user_group);
                    }
                }
                if (user->GetOpcode() != Opcode::Void) {
                    user->SetArg(operand, IR::Value{keep.phis[k]});
                    RunLocalCSE(user->GetParent());
                }
            }
        }
    };

    while (!worklist.empty()) {
        Group* const group = worklist.back();
        worklist.pop_back();
        group->queued = false;
        for (size_t i = 0; i < group->sccs.size(); ++i) {
            const PhiScc& keep = sccs[group->sccs[i]];
            if (keep.dead) {
                continue;
            }
            for (size_t j = i + 1; j < group->sccs.size(); ++j) {
                PhiScc& dup = sccs[group->sccs[j]];
                if (dup.dead || !keep.IsSameAs(dup)) {
                    continue;
                }
                replace(keep, dup);
                dup.dead = true;
            }
        }
    }
}

} // namespace

void PhiSimplificationPass(IR::Program& program) {
    std::vector<IR::Inst*> all_phis;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != Opcode::Phi) {
                break;
            }
            all_phis.push_back(&inst);
        }
    }
    SccFinder finder{all_phis};
    FoldCommonOps(finder.sccs);
    Deduplicate(finder.sccs);
}

} // namespace Shader::Optimization
