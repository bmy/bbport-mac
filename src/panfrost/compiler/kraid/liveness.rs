// Copyright © 2022-2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::ir::*;
use crate::model::RegByteSet;
use crate::ra;

use compiler::bitset::BitSet;
use compiler::dataflow::BackwardDataflow;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ord;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LiveBytes {
    pub reg: u32,
    pub mem: u32,
}

impl std::ops::Add<LiveBytes> for LiveBytes {
    type Output = LiveBytes;

    fn add(self, other: LiveBytes) -> LiveBytes {
        LiveBytes {
            reg: self.reg + other.reg,
            mem: self.mem + other.mem,
        }
    }
}

impl LiveBytes {
    fn get_mut(&mut self, is_mem: bool) -> &mut u32 {
        if is_mem { &mut self.mem } else { &mut self.reg }
    }

    pub fn max(self, other: LiveBytes) -> LiveBytes {
        LiveBytes {
            reg: self.reg.max(other.reg),
            mem: self.mem.max(other.mem),
        }
    }
}

/// Returns the number of extra bytes clobbered by this instruction that don't
/// appear in a fixed-reg source.
fn instr_extra_clobber_reg_bytes(model: &dyn Model, instr: &Instr) -> u8 {
    let clobber_regs = ra::instr_clobbered_regs(model, &instr.op);
    if clobber_regs.is_empty() {
        return 0;
    }

    // Call instructions don't have destinations
    assert!(instr.dsts().is_empty());

    let mut clobbered = RegByteSet::new();
    for reg in clobber_regs {
        clobbered.insert_range(reg.byte_range());
    }

    // Remove any fixed-reg sources
    for src in instr.srcs() {
        let SrcRef::SSA(vec) = &src.src_ref else {
            continue;
        };
        let Some(reg) = model.op_fixed_src_reg(&instr.op, src) else {
            continue;
        };

        let reg_bytes = reg.byte_range();
        let ssa_end = reg_bytes.start + u16::from(vec.bytes());
        debug_assert!(ssa_end <= reg_bytes.end);
        let ssa_bytes = reg_bytes.start..ssa_end;

        clobbered.remove_range(ssa_bytes);
    }

    clobbered.len().try_into().unwrap()
}

#[derive(Clone, Default)]
pub struct LiveSet {
    bytes: LiveBytes,
    set: FxHashSet<SSAValue>,
    bit_set: BitSet<u32>,
}

impl LiveSet {
    pub fn new() -> LiveSet {
        Default::default()
    }

    pub fn as_bit_set(&self) -> &BitSet<u32> {
        &self.bit_set
    }

    pub fn contains(&self, ssa: &SSAValue) -> bool {
        self.bit_set.contains(ssa.idx())
    }

    pub fn bytes(&self) -> LiveBytes {
        self.bytes
    }

    pub fn insert(&mut self, ssa: SSAValue) -> bool {
        if self.bit_set.insert(ssa.idx()) {
            let inserted = self.set.insert(ssa);
            debug_assert!(inserted);
            *self.bytes.get_mut(ssa.is_mem()) += u32::from(ssa.bytes());
            true
        } else {
            false
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &SSAValue> + use<'_> {
        self.set.iter()
    }

    pub fn remove(&mut self, ssa: &SSAValue) -> bool {
        if self.bit_set.remove(ssa.idx()) {
            let removed = self.set.remove(ssa);
            debug_assert!(removed);
            *self.bytes.get_mut(ssa.is_mem()) -= u32::from(ssa.bytes());
            true
        } else {
            false
        }
    }

    /// This method updates the live set for the given instruction, assuming a
    /// bottom-up walk of the instructions.  We assume that the live set
    /// represents the values live after the instruction and update it to
    /// represent the values live before the instruction.
    ///
    /// Since SSA values only ever have a single static definition, none of the
    /// SSA values written by this instruction are live before it and so can
    /// be removed from the live set.  All SSA values used by this instruction
    /// must be live before it and so they are added to the live set.
    /// Importantly, unlike updating the live set for a top-down walk, neither
    /// of those actions depends on global liveness analysis.
    ///
    /// The returned live bytes represent the maximum number of bytes live
    /// before, after, or during the execution of the given instruction. With
    /// the exception of OpCopy, the destinations of instructions are assumed
    /// to go live before sources are killed.  This can lead to slightly higher
    /// instantaneous pressure but gives the register allocator more freedom
    /// when making register choices.
    pub fn insert_instr_bottom_up(
        &mut self,
        model: &dyn Model,
        instr: &Instr,
    ) -> LiveBytes {
        if let Op::Copy(op) = &instr.op {
            // Copy is a special case and we always lower it to something
            // that has exact copy semantics and is able to fully handle
            // 8 and 16-bit destinations.  As such, we can treat it as
            // killing its sources before making its destinaion live.
            for ssa in op.iter_ssa_defs() {
                self.remove(ssa);
            }
            for ssa in op.iter_ssa_uses() {
                self.insert(*ssa);
            }
            self.bytes
        } else {
            for ssa in instr.iter_ssa_uses() {
                self.insert(*ssa);
            }

            let mut live = self.bytes;
            live.reg += u32::from(instr_extra_clobber_reg_bytes(model, instr));

            for ssa in instr.iter_ssa_defs() {
                if self.remove(ssa) {
                    *live.get_mut(ssa.is_mem()) +=
                        4_u32.saturating_sub(ssa.bytes().into());
                } else {
                    *live.get_mut(ssa.is_mem()) += 4;
                }
            }
            live
        }
    }

    /// This method updates the live set for the given instruction, assuming a
    /// top-down walk of the instructions.  We assume that the live set
    /// represents the values live before the instruction and update it to
    /// represent the values live after the instruction.
    ///
    /// Any SSA values written by this instruction are added to the set.  Any
    /// SSA values used (read or written) by this instruction are removed from
    /// the set if killed by this instruction.  (A value is killed if
    /// BlockLiveness::is_used_after_ip() returns false.)  Since this depends
    /// on liveness analysis, a BlockLiveness and the ip of the instruction
    /// are required.
    ///
    /// The returned live bytes represent the maximum number of bytes live
    /// before, after, or during the execution of the given instruction. With
    /// the exception of OpCopy, the destinations of instructions are assumed
    /// to go live before sources are killed.  This can lead to slightly higher
    /// instantaneous pressure but gives the register allocator more freedom
    /// when making register choices.
    pub fn insert_instr_top_down(
        &mut self,
        model: &dyn Model,
        ip: usize,
        instr: &Instr,
        bl: &BlockLiveness,
    ) -> LiveBytes {
        if let Op::Copy(op) = &instr.op {
            // Copy is a special case and we always lower it to something
            // that has exact copy semantics and is able to fully handle
            // 8 and 16-bit destinations.  As such, we can treat it as
            // killing its sources before making its destinaion live.
            for ssa in op.iter_ssa_uses() {
                if !bl.is_live_after_ip(ssa, ip) {
                    self.remove(ssa);
                }
            }
            let mut extra: LiveBytes = Default::default();
            for ssa in op.iter_ssa_defs() {
                if bl.is_live_after_ip(ssa, ip) {
                    self.insert(*ssa);
                } else {
                    *extra.get_mut(ssa.is_mem()) += u32::from(ssa.bytes());
                }
            }
            self.bytes + extra
        } else {
            let mut extra: LiveBytes = Default::default();
            for ssa in instr.iter_ssa_defs() {
                // For sub-register destinations, we assume that they
                // instantaneously use an entire register.  See also
                // BlockLiveness::get_instr_pressure().
                if bl.is_live_after_ip(ssa, ip) {
                    *extra.get_mut(ssa.is_mem()) +=
                        4_u32.saturating_sub(ssa.bytes().into());
                    self.insert(*ssa);
                } else {
                    *extra.get_mut(ssa.is_mem()) += 4;
                }
            }

            let mut live = self.bytes + extra;
            live.reg += u32::from(instr_extra_clobber_reg_bytes(model, instr));

            for ssa in instr.iter_ssa_uses() {
                if !bl.is_live_after_ip(ssa, ip) {
                    self.remove(ssa);
                }
            }

            live
        }
    }
}

impl std::fmt::Debug for LiveSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut vec: Vec<_> = self.set.iter().cloned().collect();
        vec.sort_by_key(|ssa| ssa.idx());
        write!(f, "{vec:?}")
    }
}

impl FromIterator<SSAValue> for LiveSet {
    fn from_iter<T: IntoIterator<Item = SSAValue>>(iter: T) -> Self {
        let mut set = LiveSet::new();
        let iter = iter.into_iter();
        set.set.reserve(iter.size_hint().0);
        for ssa in iter {
            set.insert(ssa);
        }
        set
    }
}

impl Extend<SSAValue> for LiveSet {
    fn extend<T: IntoIterator<Item = SSAValue>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.set.reserve(iter.size_hint().0);
        for ssa in iter {
            self.insert(ssa);
        }
    }
}

#[derive(Default)]
pub struct BlockLiveness {
    defs: BitSet<u32>,
    uses: BitSet<u32>,
    last_use: FxHashMap<SSAValue, usize>,
    live_in: BitSet<u32>,
    live_out: BitSet<u32>,
    max_live: LiveBytes,
}

impl BlockLiveness {
    fn new() -> Self {
        Default::default()
    }

    fn add_def(&mut self, ssa: SSAValue) {
        self.defs.insert(ssa.idx());
    }

    fn add_use(&mut self, ssa: SSAValue, ip: usize) {
        self.uses.insert(ssa.idx());
        self.last_use.insert(ssa, ip);
    }

    pub fn is_live_after_ip(&self, val: &SSAValue, ip: usize) -> bool {
        if self.live_out.contains(val.idx()) {
            true
        } else if let Some(last_use_ip) = self.last_use.get(val) {
            *last_use_ip > ip
        } else {
            false
        }
    }

    pub fn live_in_set(&self) -> &BitSet<u32> {
        &self.live_in
    }

    pub fn live_out_set(&self) -> &BitSet<u32> {
        &self.live_out
    }

    /// Returns true if @val is live-in to this block
    pub fn is_live_in(&self, val: &SSAValue) -> bool {
        self.live_in_set().contains(val.idx())
    }

    /// Returns true if @val is live-out of this block
    pub fn is_live_out(&self, val: &SSAValue) -> bool {
        self.live_out_set().contains(val.idx())
    }

    pub fn max_live_bytes(&self) -> LiveBytes {
        self.max_live
    }

    /// Returns the instantaneous pressure delta for the given instruction,
    /// assuming a top-down walk of the instructions.  This is the number of
    /// additional bytes which this instruction will instantaneously use.
    /// This includes any SSA values written by this instruction as well as
    /// padding bytes.  For OpBlend, this includes and any bytes clobbered.
    ///
    /// For OpCopy, the does not include any source SSA values which are killed
    /// by this instruction.  For all other instructions, however, we assume
    /// that the destinations go live before the sources are killed and so the
    /// instantaneous pressure includes both sources and destinations.
    pub fn get_instr_pressure_top_down(
        &self,
        model: &dyn Model,
        ip: usize,
        instr: &Instr,
    ) -> u8 {
        let mut bytes = 0_u8;
        if let Op::Copy(op) = &instr.op {
            // Copy is a special case and we always lower it to something
            // that has exact copy semantics and is able to fully handle
            // 8 and 16-bit destinations.  As such, we can treat it as
            // killing its sources before making its destinaion live.
            for ssa in op.iter_ssa_defs() {
                bytes += ssa.bytes();
            }
            for ssa in op.iter_ssa_uses() {
                if !self.is_live_after_ip(ssa, ip) {
                    bytes = bytes.saturating_sub(ssa.bytes());
                }
            }
        } else {
            // For sub-register destinations, we assume that they
            // instantaneously use an entire register.  This restriction is
            // only really needed for message instructions but the increase
            // in pressure of 1/2 or 3/4 of a register isn't important and
            // we have don't have access to the model here to be able to
            // tell the difference.
            for dst in instr.dsts() {
                if let DstRef::SSA(vec) = &dst.dst_ref {
                    bytes += vec.comps() * 4;
                }
            }
            bytes += instr_extra_clobber_reg_bytes(model, instr);
        }
        bytes
    }
}

pub struct Liveness {
    blocks: Vec<BlockLiveness>,
    max_live: LiveBytes,
}

impl Liveness {
    pub fn for_shader(s: &Shader) -> Liveness {
        let mut l = Liveness {
            blocks: Vec::new(),
            max_live: Default::default(),
        };

        for b in s.blocks.iter() {
            let mut bl = BlockLiveness::new();

            for (ip, instr) in b.instrs.iter().enumerate() {
                for ssa in instr.iter_ssa_uses() {
                    bl.add_use(*ssa, ip);
                }
                for ssa in instr.iter_ssa_defs() {
                    bl.add_def(*ssa);
                }
            }

            l.blocks.push(bl);
        }
        assert!(l.blocks.len() == s.blocks.len());

        let mut live_in: Vec<BitSet<u32>> =
            (0..s.blocks.len()).map(|_| Default::default()).collect();
        let mut live_out: Vec<BitSet<u32>> =
            (0..s.blocks.len()).map(|_| Default::default()).collect();
        BackwardDataflow {
            cfg: &s.blocks,
            block_in: &mut live_in[..],
            block_out: &mut live_out[..],
            transfer: |block_idx, _, live_in, live_out| {
                let bl = &l.blocks[block_idx];
                live_in.union_with(
                    (live_out.s(..) | bl.uses.s(..)) - bl.defs.s(..),
                )
            },
            join: |live_out, succ_live_in| {
                *live_out |= succ_live_in.s(..);
            },
        }
        .solve();

        for ((bl, b_live_in), b_live_out) in l
            .blocks
            .iter_mut()
            .zip(live_in.into_iter())
            .zip(live_out.into_iter())
        {
            bl.live_in = b_live_in;
            bl.live_out = b_live_out;
        }

        // Now that we have live sets, compute the max live per-block and
        // for the whold shader.
        let mut block_live_out: Vec<LiveSet> = Vec::new();
        for (bi, bb) in s.blocks.iter().enumerate() {
            let bl = &mut l.blocks[bi];
            let mut live = LiveSet::new();

            // Predecessors are added block order so we can just grab the first
            // one (if any) and it will be a block we've processed.
            if let Some(pred_idx) = s.blocks.pred_indices(bi).first() {
                let pred_out = &block_live_out[*pred_idx];
                live = pred_out
                    .iter()
                    .cloned()
                    .filter(|ssa| bl.is_live_in(ssa))
                    .collect();
            }

            for (ip, instr) in bb.instrs.iter().enumerate() {
                let live_at_instr =
                    live.insert_instr_top_down(s.model, ip, instr, bl);
                bl.max_live = bl.max_live.max(live_at_instr);
            }
            l.max_live = l.max_live.max(bl.max_live);

            assert!(block_live_out.len() == bi);
            block_live_out.push(live);
        }

        l
    }

    pub fn block(&self, idx: usize) -> &BlockLiveness {
        &self.blocks[idx]
    }

    pub fn max_live_bytes(&self) -> LiveBytes {
        self.max_live
    }
}
