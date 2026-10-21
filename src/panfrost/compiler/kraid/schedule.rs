// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::liveness::*;
use crate::ops::{MemoryEffect, VaryingUpdateMode};
use crate::ssa_value::SSAValueAllocator;
use compiler::bitset::BitSet;
use rustc_hash::FxHashMap;
use std::ops::Range;

fn is_mem(op: &Op) -> bool {
    match op.memory_effect() {
        // We ignore ConstRead since the value of constant memory should never
        // be affected by other memory ops
        MemoryEffect::None | MemoryEffect::ConstRead => false,
        MemoryEffect::Read | MemoryEffect::Write | MemoryEffect::ReadWrite => {
            true
        }
    }
}

fn is_mem_write(op: &Op) -> bool {
    match op.memory_effect() {
        MemoryEffect::None | MemoryEffect::ConstRead | MemoryEffect::Read => {
            false
        }
        MemoryEffect::Write | MemoryEffect::ReadWrite => true,
    }
}

fn is_barrier(op: &Op) -> bool {
    matches!(op, Op::Barrier(_) | Op::ScheduleBarrier(_))
}

fn respects_barrier(op: &Op) -> bool {
    // ALU can freely slide past barriers
    is_barrier(op) || is_mem(op)
}

fn writes_var_hidden(op: &Op) -> bool {
    match op.var_update_mode() {
        VaryingUpdateMode::Store | VaryingUpdateMode::Clobber => true,
        VaryingUpdateMode::Retrieve | VaryingUpdateMode::None => false,
    }
}

fn uses_var_hidden(op: &Op) -> bool {
    op.var_update_mode() != VaryingUpdateMode::None
}

struct DepTracker {
    /// For each instruction, the bitset of instructions which depend on it
    deps: Vec<BitSet<usize>>,
    /// For each instruction, the number of instructions it depends on
    count: Vec<u32>,
    /// Bitset of instructions for which count == 0
    ready: BitSet<usize>,
}

impl DepTracker {
    fn add_ip(&mut self, ip: usize) {
        debug_assert!(self.count[ip] == 0);
        self.ready.insert(ip);
    }

    fn add_dep(&mut self, ip: usize, dep_ip: usize) {
        if self.deps[ip].insert(dep_ip) {
            self.count[dep_ip] += 1;
            self.ready.remove(dep_ip);
        }
    }

    fn ready(&self) -> impl Iterator<Item = usize> + use<'_> {
        self.ready.iter()
    }

    fn remove_ip(&mut self, ip: usize) {
        debug_assert!(self.count[ip] == 0);
        debug_assert!(self.ready.contains(ip));

        self.ready.remove(ip);
        for dep_ip in self.deps[ip].iter() {
            self.count[dep_ip] -= 1;
            if self.count[dep_ip] == 0 {
                self.ready.insert(dep_ip);
            }
        }
        self.deps[ip].clear();
    }

    fn for_block(block: &BasicBlock, body_range: Range<usize>) -> DepTracker {
        let instr_count = block.instrs.len();
        let mut deps = DepTracker {
            deps: (0..instr_count).map(|_| BitSet::new()).collect(),
            count: (0..instr_count).map(|_| 0_u32).collect(),
            ready: BitSet::new(),
        };

        // Map from an SSAValue to its definition in this block, if any.
        let mut def_ip: FxHashMap<SSAValue, usize> = Default::default();

        // IP of the last memory instruction
        let mut bar_ip = usize::MAX;

        // IP of the last barrier instruction
        let mut mem_ip = usize::MAX;

        // IP of the last instruction to access the varying hidden register
        let mut var_ip = usize::MAX;

        // IP of the last discard instruction
        let mut discard_ip = usize::MAX;

        for ip in body_range.clone() {
            let instr = &block.instrs[ip];
            deps.add_ip(ip);

            for ssa in instr.iter_ssa_uses() {
                if let Some(&def_ip) = def_ip.get(ssa) {
                    deps.add_dep(ip, def_ip);
                }
            }

            if bar_ip != usize::MAX && respects_barrier(&instr.op) {
                deps.add_dep(ip, bar_ip);
            }
            if is_barrier(&instr.op) {
                bar_ip = ip;
            }

            // Capture WaW and RaW hazards
            if mem_ip != usize::MAX && is_mem(&instr.op) {
                deps.add_dep(ip, mem_ip);
            }
            if is_mem_write(&instr.op) {
                mem_ip = ip;
            }

            // Capture WaW and RaW hazards
            if var_ip != usize::MAX && uses_var_hidden(&instr.op) {
                deps.add_dep(ip, var_ip);
            }
            if writes_var_hidden(&instr.op) {
                var_ip = ip;
            }

            if discard_ip != usize::MAX && instr.reads_discard() {
                deps.add_dep(ip, discard_ip);
            }
            if instr.writes_discard() {
                discard_ip = ip;
            }

            for ssa in instr.iter_ssa_defs() {
                def_ip.insert(*ssa, ip);
            }
        }

        bar_ip = usize::MAX;
        mem_ip = usize::MAX;
        var_ip = usize::MAX;
        discard_ip = usize::MAX;
        for ip in body_range.clone().rev() {
            let instr = &block.instrs[ip];

            if bar_ip != usize::MAX && respects_barrier(&instr.op) {
                deps.add_dep(bar_ip, ip);
            }
            if is_barrier(&instr.op) {
                bar_ip = ip;
            }

            // Capture WaR hazards
            if mem_ip != usize::MAX && is_mem(&instr.op) {
                deps.add_dep(mem_ip, ip);
            }
            if is_mem_write(&instr.op) {
                mem_ip = ip;
            }

            // Capture WaR hazards
            if var_ip != usize::MAX && uses_var_hidden(&instr.op) {
                deps.add_dep(var_ip, ip);
            }
            if writes_var_hidden(&instr.op) {
                var_ip = ip;
            }

            if discard_ip != usize::MAX && instr.reads_discard() {
                deps.add_dep(discard_ip, ip);
            }
            if instr.writes_discard() {
                discard_ip = ip;
            }
        }

        deps
    }
}

fn pressure_schedule_block(
    model: &dyn Model,
    ssa_alloc: &SSAValueAllocator,
    b: &mut BasicBlock,
    bl: &BlockLiveness,
) {
    let body_range = b.body_ip_range();

    let mut deps = DepTracker::for_block(b, body_range.clone());

    let mut live = LiveSet::new();
    for idx in bl.live_out_set().iter() {
        live.insert(ssa_alloc.lookup_by_idx(idx));
    }

    // We're not going to schedule the postlude but we need to account for it
    // in the live set or our estimates will be all out of whack.
    let mut max_live = LiveBytes::default();
    for ip in body_range.end..b.instrs.len() {
        let bytes = live.insert_instr_bottom_up(model, &b.instrs[ip]);
        max_live = max_live.max(bytes);
    }

    let mut end_ip = body_range.end;
    let mut schedule: Vec<_> = (0..b.instrs.len())
        .map(|ip| {
            if body_range.contains(&ip) {
                usize::MAX
            } else {
                ip
            }
        })
        .collect();

    loop {
        let mut best_ip = usize::MAX;
        let mut best_pressure = i32::MAX;
        for ip in deps.ready() {
            let mut rel_pressure = 0_i32;
            for ssa in b.instrs[ip].iter_ssa_defs() {
                if live.contains(ssa) {
                    rel_pressure -= i32::from(ssa.bytes());
                }
            }
            for ssa in b.instrs[ip].iter_ssa_uses() {
                if !live.contains(ssa) {
                    rel_pressure += i32::from(ssa.bytes());
                }
            }
            if rel_pressure <= best_pressure {
                best_ip = ip;
                best_pressure = rel_pressure;
            }
        }

        if best_ip == usize::MAX {
            break;
        }

        // Assert that no instruction gets placed twice
        assert!(schedule[best_ip] == usize::MAX);
        end_ip -= 1;
        schedule[best_ip] = end_ip;

        let bytes = live.insert_instr_bottom_up(model, &b.instrs[best_ip]);
        max_live = max_live.max(bytes);

        deps.remove_ip(best_ip);
    }

    // Assert we placed all of them
    assert!(end_ip == body_range.start);

    // Replace with the new scheduling if it's better.
    if max_live.reg < bl.max_live_bytes().reg {
        // SAFETY:
        //
        // We already asserted that we placed each instruction exactly once.
        unsafe {
            b.reorder_instrs(|ip| Some(schedule[ip]), b.instrs.len());
        }
    }
}

impl Shader<'_> {
    pub fn schedule_for_pressure(&mut self) {
        let live = Liveness::for_shader(self);
        for (bi, block) in self.blocks.iter_mut().enumerate() {
            pressure_schedule_block(
                self.model,
                &self.ssa_alloc,
                block,
                live.block(bi),
            );
        }
    }
}
