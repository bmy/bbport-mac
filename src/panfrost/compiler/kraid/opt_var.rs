// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use compiler::dataflow::ForwardDataflow;

use crate::ir::{BasicBlock, Instr, Op, Shader, SrcRef};
use crate::ops::{SamplePosition, VaryingUpdateMode};

#[derive(PartialEq, Clone)]
enum HiddenRegister {
    /// Nothing seen yet
    Unknown,
    Known(SamplePosition, SrcRef),
    /// Clobbered or conflicting registers
    Undefined,
}

impl HiddenRegister {
    fn from_instr(i: &Instr) -> Option<HiddenRegister> {
        let (update_mode, sample_pos, src) = match &i.op {
            Op::LdVar(op) => (op.update, op.sample_position, &op.src),
            Op::LdVarBuf(op) => (op.update, op.sample_position, &op.src),
            Op::LdVarSpecial(op) => (op.update, op.sample_position, &op.src),
            _ => return None,
        };

        match update_mode {
            VaryingUpdateMode::Store => {
                assert!(src.swizzle.is_none());
                assert!(src.src_mod.is_none());
                Some(HiddenRegister::Known(sample_pos, src.src_ref.clone()))
            }
            VaryingUpdateMode::Clobber => Some(HiddenRegister::Undefined),
            VaryingUpdateMode::Retrieve | VaryingUpdateMode::None => None,
        }
    }

    fn join(&mut self, new: &HiddenRegister) {
        match self {
            HiddenRegister::Unknown => *self = new.clone(),
            HiddenRegister::Known(_, _) => {
                if self != new {
                    *self = HiddenRegister::Undefined;
                }
            }
            HiddenRegister::Undefined => {}
        }
    }
}

fn clear_update_mode(i: &mut Instr) {
    let (update, sample_pos, src) = match &mut i.op {
        Op::LdVar(op) => (&mut op.update, &mut op.sample_position, &mut op.src),
        Op::LdVarBuf(op) => {
            (&mut op.update, &mut op.sample_position, &mut op.src)
        }
        Op::LdVarSpecial(op) => {
            (&mut op.update, &mut op.sample_position, &mut op.src)
        }
        _ => panic!("Instr doesn't use any hidden register"),
    };
    *update = VaryingUpdateMode::Retrieve;
    *sample_pos = SamplePosition::None;
    *src = 0_u32.into();
}

fn last_reg_set(b: &BasicBlock) -> Option<HiddenRegister> {
    b.instrs
        .iter()
        .filter_map(HiddenRegister::from_instr)
        .next_back()
}

impl Shader<'_> {
    pub fn opt_var(&mut self) {
        let cfg = &self.blocks;
        // First pass: gather the last register set by each block
        let block_usage: Vec<_> = cfg.iter().map(|b| last_reg_set(b)).collect();

        // Second pass: forward dataflow to get the initial register state for
        //              each block
        let mut reg_in: Vec<HiddenRegister> =
            vec![HiddenRegister::Unknown; cfg.len()];
        let mut reg_out: Vec<HiddenRegister> =
            vec![HiddenRegister::Unknown; cfg.len()];
        reg_in[0] = HiddenRegister::Undefined;
        ForwardDataflow {
            cfg,
            block_in: &mut reg_in[..],
            block_out: &mut reg_out[..],
            transfer: |block_idx, _block, live_out, live_in| {
                let old = live_out.clone();
                if let Some(reg) = &block_usage[block_idx] {
                    *live_out = reg.clone();
                } else {
                    *live_out = live_in.clone();
                }
                *live_out != old
            },
            join: |live_in, pred_live_out| live_in.join(pred_live_out),
        }
        .solve();

        // Third pass: simulate each instruction with the (now known) initial
        //             state and remove the Update if the computed register is
        //             the same
        for (bi, b) in &mut self.blocks.iter_mut().enumerate() {
            let mut current_reg = reg_in[bi].clone();
            debug_assert!(current_reg != HiddenRegister::Unknown);

            for i in &mut b.instrs {
                let Some(reg_usage) = HiddenRegister::from_instr(i) else {
                    continue;
                };
                if reg_usage == current_reg
                    && reg_usage != HiddenRegister::Undefined
                {
                    clear_update_mode(i);
                }
                current_reg = reg_usage;
            }
        }
    }
}
