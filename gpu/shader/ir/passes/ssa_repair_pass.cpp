// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// After structurization a result may be used where its block no longer dominates the use: such
// a result goes through a virtual register (stored after it, loaded before each such use), which
// SSA rewriting turns back into values with phis.
#include <boost/container/small_vector.hpp>

#include "shader/ir/dominance.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"

namespace Shader::Optimization {

void SsaRepairPass(IR::Program& program) {
    auto& post_order = program.post_order_blocks;
    IR::ComputeDominators(post_order);
    IR::ComputeDominanceFrontiers(post_order);

    boost::container::small_vector<IR::Use, 2> undominated;
    for (IR::Block* block : post_order) {
        for (IR::Inst& inst : block->Instructions()) {
            for (const IR::Use& use : inst.Uses()) {
                if (!IR::Dominates(block, use.user->GetParent())) {
                    undominated.push_back(use);
                }
            }
            if (undominated.empty()) {
                continue;
            }
            const IR::VirtualReg reg{program.next_reg_index++, inst.Type()};
            IR::IREmitter store{*block, ++IR::Block::InstructionList::s_iterator_to(inst)};
            store.SetVirtualReg(reg, IR::Value{&inst});
            for (const auto& [user, operand] : undominated) {
                IR::IREmitter load{*user->GetParent(),
                                   IR::Block::InstructionList::s_iterator_to(*user)};
                user->SetArg(operand, load.GetVirtualReg(reg));
            }
            undominated.clear();
        }
    }
}

} // namespace Shader::Optimization
