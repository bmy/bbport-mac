// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::builder::{Builder, InstrBuilder};
use crate::ir::*;
use crate::ops::*;

fn lower_blend_call(b: &mut impl Builder, op: OpBlendCall, flow: FlowCtrl) {
    let prolog_length = 2 * 8;

    // Don't even encode the offset if BLEND exits after execution.
    let blend_offset = if flow.get_end_shader() {
        0
    } else {
        prolog_length
    };
    let mut blend = Instr::from(OpBlend {
        color_type: op.color_type,
        descr: op.descr.clone(),
        coverage: op.coverage,
        color: op.color,
        offset: blend_offset,
        render_target_idx: op.render_target_idx,
    });
    blend.flow = flow;
    b.push_instr(blend);

    // Setup return addr
    let link = b.model().preload_reg(PreloadReg::BlendReturnAddr).unwrap();

    if flow.get_end_shader() {
        // Jumping to 0 terminates execution
        b.push_op(OpMov {
            dst: link.into(),
            dst_type: DataType::I32,
            src: 0_u32.into(),
        });
    } else {
        let pc = b.model().fau().special(SpecialFAU::Pc).unwrap().word(0);
        // RA = instr after OpJump
        // Too late for legalization, MUST be a legal IADD_IMM
        b.push_op(OpIAdd {
            dst: link.into(),
            dst_type: DataType::I32,
            saturate: false,
            srcs: [pc.into(), (prolog_length - 8).into()],
        });
    }

    // Indirect jump to blend function
    let blend_fn_addr = op.descr.word(1);
    b.push_op(OpJump {
        not: true,
        cond: 0u32.into(),
        combine_op: BranchCombineOp::None,
        address: blend_fn_addr,
    });
}

impl Shader<'_> {
    /// This pass lowers every BLEND_CALL virtual op in the equivalent BLEND +
    /// prologue/jump instructions that can call the appropriate blend shader
    /// when the descriptor requires them.  This pass must run after `opt_flow`.
    /// After this pass no other pass should add/remove or reoder instructions
    /// as it can mess up the crafted prologue.  This must be the last pass.
    pub fn lower_blend_call(&mut self) {
        if self.info.is_blend {
            // Blend shaders cannot have blend calls
            return;
        }
        let model = self.model;
        self.map_instrs(|instr, _| match instr.op {
            Op::BlendCall(op) => {
                let mut b = InstrBuilder::new(model);
                lower_blend_call(&mut b, *op, instr.flow);
                b.into_mapped()
            }
            _ => [instr].into(),
        })
    }
}
