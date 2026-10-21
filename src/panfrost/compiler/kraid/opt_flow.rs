// Copyright © 2026 Arm Ltd.
// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use compiler::dataflow::ForwardDataflow;

use crate::flow::FlowWaitBit;
use crate::ir::*;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SlotState {
    Unknown,
    Signaled,
    Waited,
}

impl SlotState {
    pub fn then(&mut self, other: Self) {
        *self = match (*self, other) {
            (a, SlotState::Unknown) => a,
            (_, b) => b,
        };
    }

    pub fn join(&mut self, other: Self) {
        *self = match (*self, other) {
            (SlotState::Unknown, a) | (a, SlotState::Unknown) => a,
            (SlotState::Waited, SlotState::Waited) => SlotState::Waited,
            _ => SlotState::Signaled,
        };
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct WaitSlots {
    resource: SlotState,
    zs: SlotState,
    barrier: SlotState,
}

impl WaitSlots {
    pub fn unknown() -> Self {
        Self {
            barrier: SlotState::Unknown,
            resource: SlotState::Unknown,
            zs: SlotState::Unknown,
        }
    }

    pub fn signaled() -> Self {
        Self {
            barrier: SlotState::Signaled,
            resource: SlotState::Signaled,
            zs: SlotState::Signaled,
        }
    }

    pub fn issue(&mut self, instr: &Instr) {
        if matches!(&instr.op, Op::Barrier(_)) {
            self.barrier = SlotState::Signaled;
        }
    }

    pub fn wait_bit(&mut self, bit: FlowWaitBit) {
        match bit {
            FlowWaitBit::Resource => {
                self.resource = SlotState::Waited;
            }
            FlowWaitBit::ZS => {
                self.zs = SlotState::Waited;
            }
            FlowWaitBit::Barrier => {
                self.barrier = SlotState::Waited;
                // resource is used to wait on read-only resources
                // barrier is used to wait on both read-only and read-write resources
                // so it effectively waits for both
                self.resource = SlotState::Waited;
                // barrier also waits for dependency slot 6
                self.zs = SlotState::Waited;
            }
            _ => {}
        }
    }

    pub fn wait(&mut self, flow: FlowCtrl) {
        for bit in
            [FlowWaitBit::Resource, FlowWaitBit::ZS, FlowWaitBit::Barrier]
        {
            if flow.get_wait_bit(bit) {
                self.wait_bit(bit);
            }
        }
    }

    pub fn is_waited(&self, bit: FlowWaitBit) -> bool {
        let slot = match bit {
            FlowWaitBit::Resource => self.resource,
            FlowWaitBit::ZS => self.zs,
            FlowWaitBit::Barrier => self.barrier,
            _ => return false,
        };
        slot == SlotState::Waited
    }

    pub fn then(&mut self, other: Self) {
        self.resource.then(other.resource);
        self.zs.then(other.zs);
        self.barrier.then(other.barrier);
    }

    pub fn join(&mut self, other: Self) {
        self.resource.join(other.resource);
        self.zs.join(other.zs);
        self.barrier.join(other.barrier);
    }
}

fn dce_flow(shader: &mut Shader) {
    let cfg = &shader.blocks;
    // First pass: gather waits per-block
    let block_usage: Vec<_> = cfg
        .iter()
        .map(|b| {
            b.instrs.iter().fold(WaitSlots::unknown(), |mut s, instr| {
                s.issue(instr);
                s.wait(instr.flow);
                s
            })
        })
        .collect();

    // Second pass: forward dataflow to get the initial register state for each
    //              block
    let mut reg_in = vec![WaitSlots::unknown(); cfg.len()];
    let mut reg_out = vec![WaitSlots::unknown(); cfg.len()];
    reg_in[0] = WaitSlots::signaled();
    if shader.info.is_blend {
        // blend shaders come from a BLEND call, those require a zs + barrier
        // wait, so we can clear those slots
        reg_in[0].wait_bit(FlowWaitBit::Barrier);
    }
    ForwardDataflow {
        cfg,
        block_in: &mut reg_in[..],
        block_out: &mut reg_out[..],
        transfer: |block_idx, _block, live_out, live_in| {
            let old = *live_out;
            let mut state = *live_in;
            state.then(block_usage[block_idx]);
            *live_out = state;

            *live_out != old
        },
        join: |live_in, pred_live_out| live_in.join(*pred_live_out),
    }
    .solve();

    // Third pass: Remove useless waits
    for (bi, block) in shader.blocks.iter_mut().enumerate() {
        let mut state = reg_in[bi];
        for instr in &mut block.instrs {
            state.issue(instr);
            for bit in
                [FlowWaitBit::ZS, FlowWaitBit::Resource, FlowWaitBit::Barrier]
            {
                if state.is_waited(bit) {
                    instr.flow.take_wait_bit(bit);
                }
            }
            state.wait(instr.flow);
        }
    }
}

fn merge_waits(shader: &mut Shader) {
    for block in &mut shader.blocks {
        for i in (1..block.instrs.len()).rev() {
            if !matches!(block.instrs[i].op, Op::Nop(_)) {
                continue;
            }
            let nop_flow = block.instrs[i].flow;
            if nop_flow.get_end_shader()
                || nop_flow.get_discard()
                || nop_flow.get_reconverge()
                || nop_flow == FlowCtrl::NONE
            {
                continue;
            }

            let prev_flow = &mut block.instrs[i - 1].flow;

            // In theory we can promote a .wait_resource into a .barrier but that
            // is not useful in practice (as it also waits for r/w resources)
            if nop_flow.get_wait_bit(FlowWaitBit::Resource)
                && prev_flow.wait != 0
            {
                continue;
            }

            prev_flow.wait |= nop_flow.wait;
            block.instrs[i].flow.wait = 0;
        }

        block.instrs.retain(|i| {
            i.flow != FlowCtrl::NONE || !matches!(i.op, Op::Nop(_))
        });
    }
}

fn has_nop_end(block: &BasicBlock) -> bool {
    block
        .instrs
        .last()
        .is_some_and(|i| matches!(&i.op, Op::Nop(_)) && i.flow.get_end_shader())
}

/// This pass will remove NOP.end instructions if the .end modifier can
/// be encoded on the instruction before. Barriers at the end of shaders
/// are also safe to remove.
fn merge_end(shader: &mut Shader) {
    for block in &mut shader.blocks {
        if !has_nop_end(block) || block.instrs.len() == 0 {
            continue;
        }

        let nop_end = block.instrs.pop().unwrap();

        while block
            .instrs
            .last()
            .is_some_and(|i| matches!(i.op, Op::Barrier(_)))
        {
            block.instrs.pop();
        }

        if let Some(last) = block.instrs.last_mut() {
            let slot = last.flow.get_msg_slot_idx();
            last.flow = FlowCtrl::NONE;
            if let Some(slot) = slot {
                last.flow.set_msg_slot_idx(slot);
            }
            last.flow.set_end_shader();
        } else {
            block.instrs.push(nop_end);
        }
    }
}

impl Shader<'_> {
    /// This pass optimizes and merges flow wait bits.
    ///
    /// 1. Performs data-flow analysis to remove waits on already waited slots
    /// 2. Merges NOP waits into previous instructions where possible
    /// 3. Merges NOP.end with the previous instructions
    pub fn opt_flow(&mut self) {
        dce_flow(self);
        merge_waits(self);
        merge_end(self);
    }
}
