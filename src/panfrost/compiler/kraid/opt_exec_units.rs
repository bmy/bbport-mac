// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::data_type::NumericType;
use crate::ir::*;
use crate::ops::{CmpOp, MuxOp, OpCSel, OpMux};

/// MUX executes in the SFU unit, while CSEL executes in the CVT unit.  CVT is
/// always wider than the SFU unit by at least 4x, this is always an optimization.
fn try_replace_mux_csel(op: &OpMux) -> Option<OpCSel> {
    if !matches!(op.dst_type, DataType::I32 | DataType::V2I16) {
        return None;
    };

    if op.src0.swizzle != Swizzle::NONE
        || op.src1.swizzle != Swizzle::NONE
        || op.sel.swizzle != Swizzle::NONE
    {
        return None;
    }

    let (cmp_op, num) = match op.mux_op {
        MuxOp::Neg => (CmpOp::Ge, NumericType::SignedInteger),
        MuxOp::IntZero => (CmpOp::Ne, NumericType::SignedInteger),
        MuxOp::FpZero => (CmpOp::Ne, NumericType::Float),
        MuxOp::Bit => return None,
    };

    Some(OpCSel {
        dst: op.dst.clone(),
        cmp_type: DataType::get(op.dst_type.comps(), num, op.dst_type.bits()),
        cmp_op,
        cmp_srcs: [op.sel.clone(), 0u32.into()],
        sel_srcs: [op.src1.clone(), op.src0.clone()],
    })
}

fn translate_instr(i: &mut Instr) {
    if let Op::Mux(op) = &mut i.op {
        if let Some(csel) = try_replace_mux_csel(op) {
            i.op = csel.into();
        }
    }
}

impl Shader<'_> {
    /// Tries to replace operations with cheaper, equivalent ones.  The only
    /// example right now is MUX being replaced with CSEL (SFU -> CVT), we might
    /// want this list to be expanded in the future.  Must run before
    /// legalization as we are inserting a 0 source that might conflict with
    /// other FAUs (in v9-v10).
    pub fn opt_exec_units(&mut self) {
        self.map_instrs(|mut i, _| {
            translate_instr(&mut i);
            [i].into()
        });
    }
}
