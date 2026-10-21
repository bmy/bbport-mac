// Copyright © 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use std::cmp::Reverse;
use std::num::{NonZeroU32, NonZeroU64};

use crate::ir::*;
use kraid_bindings::*;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum ConstEntry {
    I32(NonZeroU32),
    I64(NonZeroU64),
}

impl ConstEntry {
    fn from_src_promotable(
        model: &dyn Model,
        op: &Op,
        src: &Src,
    ) -> Option<Self> {
        match &src.src_ref {
            SrcRef::Imm32(v) => {
                // Already encodable inline, so it needs no FAU slot
                if model.op_src_supports_imm32(op, src, v.get()) {
                    None
                } else {
                    Some(ConstEntry::I32(*v))
                }
            }
            SrcRef::Imm64(v) => Some(ConstEntry::I64(*v)),
            _ => None,
        }
    }

    fn words(self) -> u8 {
        match self {
            ConstEntry::I32(_) => 1,
            ConstEntry::I64(_) => 2,
        }
    }
}

#[derive(Default)]
struct ConstTable(FxHashMap<ConstEntry, u32>);

impl ConstTable {
    fn add_op_src(&mut self, model: &dyn Model, op: &Op, src: &Src) {
        let Some(key) = ConstEntry::from_src_promotable(model, op, src) else {
            return;
        };

        self.0.entry(key).and_modify(|w| *w += 1).or_insert(1);
    }

    fn into_sorted(self) -> Vec<(ConstEntry, u32)> {
        let mut weighted: Vec<_> = self.0.into_iter().collect();
        // Break ties using the entry itself
        weighted.sort_by_key(|(e, w)| Reverse((*w, *e)));
        weighted
    }
}

fn fau_available(fau: &pan_fau_layout) -> u32 {
    unsafe { pan_fau_available(fau) as u32 }
}

fn fau_emit_const(fau: &mut pan_fau_layout, imm: u32) -> u16 {
    unsafe { pan_fau_emit_const(fau, imm) }.try_into().unwrap()
}

fn promote_consts(s: &mut Shader, fau: &mut pan_fau_layout) {
    let mut const_table = ConstTable::default();

    // TODO: Take into account control flow and try to use source mods to unify
    // constants together.
    for block in s.blocks.iter() {
        for instr in block.instrs.iter() {
            for src in instr.srcs() {
                const_table.add_op_src(s.model, &instr.op, src);
            }
        }
    }

    let avail = fau_available(fau);
    // Carve a range aligned to 64-bits (2 words)
    let mut aligned = avail.saturating_sub(fau.count % 2) & !1;
    // But let's still fill the holes
    let mut holes = avail - aligned;

    let mut selected32 = Vec::new();
    let mut selected64 = Vec::new();
    for (entry, _) in const_table.into_sorted() {
        if aligned == 0 && holes == 0 {
            break;
        }
        let words = u32::from(entry.words());
        if words == 1 && holes > 0 {
            holes -= 1;
        } else if words <= aligned {
            aligned -= words;
        } else {
            continue;
        }

        match entry {
            ConstEntry::I32(imm32) => selected32.push(imm32),
            ConstEntry::I64(imm64) => selected64.push(imm64),
        }
    }

    let mut entry_to_idx = FxHashMap::default();
    let has_wide = !selected64.is_empty();
    let mut selected32 = selected32.into_iter();
    let selected64 = selected64.into_iter();

    // If we have an unaligned hole fill it with a 32-bit const
    if has_wide && (fau.count % 2) != 0 {
        if let Some(imm32) = selected32.next() {
            let fau_idx = fau_emit_const(fau, imm32.get());
            entry_to_idx.insert(ConstEntry::I32(imm32), fau_idx);
        } else {
            fau_emit_const(fau, 0);
        }
    }

    for imm64 in selected64 {
        let fau_idx = fau_emit_const(fau, imm64.get() as u32);
        fau_emit_const(fau, (imm64.get() >> 32) as u32);
        debug_assert_eq!(fau_idx % 2, 0);
        entry_to_idx.insert(ConstEntry::I64(imm64), fau_idx);
    }

    for imm32 in selected32 {
        let fau_idx = fau_emit_const(fau, imm32.get());
        entry_to_idx.insert(ConstEntry::I32(imm32), fau_idx);
    }

    for block in s.blocks.iter_mut() {
        for instr in block.instrs.iter_mut() {
            for src_idx in 0..instr.srcs().len() {
                let src = &instr.srcs()[src_idx];

                let Some(entry) =
                    ConstEntry::from_src_promotable(s.model, &instr.op, src)
                else {
                    continue;
                };
                let Some(idx) = entry_to_idx.get(&entry) else {
                    continue;
                };
                let fau_ref = match entry {
                    ConstEntry::I32(_) => FAURef::user_i32(*idx),
                    ConstEntry::I64(_) => FAURef::user_i64(*idx),
                };

                instr.srcs_mut()[src_idx].src_ref = SrcRef::FAU(fau_ref);
            }
        }
    }
}

impl Shader<'_> {
    pub fn opt_promote_consts(&mut self, fau: &mut pan_fau_layout) {
        promote_consts(self, fau);
    }
}
