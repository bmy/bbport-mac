// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Before structurization: phis become register accesses again. A phi standing for a GCN register
// reads that register; any other gets a virtual register each predecessor stores its value to.
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"

namespace Shader::Optimization {

namespace {

IR::Value Load(IR::IREmitter& ir, IR::RegTag tag) {
    switch (tag.type) {
    case IR::RegType::ScalarReg:
        return ir.GetScalarReg(tag.sreg);
    case IR::RegType::VectorReg:
        return ir.GetVectorReg(tag.vreg);
    case IR::RegType::Scc:
        return ir.GetScc();
    case IR::RegType::Exec:
        return ir.GetExec();
    case IR::RegType::VccLo:
        return ir.GetVccLo();
    case IR::RegType::VccHi:
        return ir.GetVccHi();
    case IR::RegType::M0:
        return ir.GetM0();
    case IR::RegType::VirtualReg:
        return ir.GetVirtualReg(tag.reg);
    default:
        UNREACHABLE_MSG("register kind {}", u32(tag.type));
    }
}

} // namespace

void LowerPhisToRegsPass(IR::Program& program) {
    // Reverse post order; phis lead their blocks (removed ones become voids).
    for (auto block_it = program.post_order_blocks.rbegin();
         block_it != program.post_order_blocks.rend(); ++block_it) {
        IR::Block* const block = *block_it;
        for (auto it = block->begin(); it != block->end(); ++it) {
            IR::Inst& phi = *it;
            if (phi.GetOpcode() == IR::Opcode::Void) {
                continue;
            }
            if (phi.GetOpcode() != IR::Opcode::Phi) {
                break;
            }
            IR::IREmitter ir{*block, it};
            if (const auto tag = phi.GetRegTag()) {
                phi.ReplaceUsesWithAndRemove(Load(ir, tag));
                continue;
            }
            const IR::VirtualReg reg{program.next_reg_index++, phi.Flags<IR::Type>()};
            for (size_t i = 0; i < phi.NumArgs(); ++i) {
                IR::IREmitter pred{*phi.PhiBlock(i)};
                pred.SetVirtualReg(reg, phi.Arg(i));
            }
            phi.ReplaceUsesWithAndRemove(ir.GetVirtualReg(reg));
        }
    }
}

} // namespace Shader::Optimization
