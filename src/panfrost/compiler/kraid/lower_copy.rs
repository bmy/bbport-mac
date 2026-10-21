// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::builder::*;
use crate::ir::*;
use crate::ops::*;

fn lower_copy(b: &mut impl Builder, copy: OpCopy) {
    debug_assert!(copy.src.src_mod.is_none());
    debug_assert!(copy.src_supports_swizzle(&copy.src, copy.src.swizzle));

    if let DstRef::Mem(mem) = &copy.dst.dst_ref {
        assert!(
            !matches!(&copy.src.src_ref, SrcRef::Mem(_)),
            "We can't handle mem <-> mem copies this late"
        );
        b.store_tls(mem.offset, DataType::i(mem.range * 8), copy.src);
        return;
    } else if let SrcRef::Mem(mem) = &copy.src.src_ref {
        b.load_tls_to(copy.dst, DataType::i(mem.range * 8), mem.offset);
        return;
    }

    if copy.dst_type == DataType::I8 {
        debug_assert!(copy.dst.lanes.is_byte());

        // We don't have an ALU ops which support an actual byte lane select
        // so we're going to have to carefully stomp the whole reg
        let mut dst_reg = *copy.dst.dst_ref.as_reg().unwrap();
        dst_reg.range = RegRange::Regs(1);

        // Handle all immediates, including zero with logic ops
        if let Ok(imm) = u32::try_from(&copy.src.src_ref) {
            let mask = copy.dst.lanes.u32_mask().unwrap();
            // For an immediate copy, we can just do and AND+ADD
            if b.model().arch() >= 11 {
                b.push_op(OpShiftLop {
                    dst: dst_reg.into(),
                    dst_type: DataType::U32,
                    shift_op: ShiftOp::None,
                    logic_op: LogicOp::And,
                    not_result: false,
                    src0: dst_reg.into(),
                    shift: 0_u8.into(),
                    src2: (!mask).into(),
                });
            } else {
                // AND_IMM is not implemented yet, we need to get creative
                // We should do: `~(0xFF << i) & reg` but since .bnot is not
                // available in src0, we apply De Morgan: `~((0xFF << i) | ~reg)`
                let const_table = &b.model().fau().small_constants;
                let offset = copy.dst.lanes.as_byte_range().unwrap().start;
                let kFF = const_table.find_imm8(|x| x == u8::MAX).unwrap();
                let kFF = kFF.swizzle(Swizzle::widen_u8(0));
                b.push_op(OpShiftLop {
                    dst: dst_reg.into(),
                    dst_type: DataType::U32,
                    shift_op: ShiftOp::LShift,
                    logic_op: LogicOp::Or,
                    not_result: true,
                    src0: kFF,
                    shift: const_table
                        .find_imm8(|i| i == offset * 8)
                        .expect("Cannot find small const offset"),
                    src2: Src::from(dst_reg).bnot(),
                });
            }

            // Add back the immediate now
            let imm = copy.src.swizzle.fold_u32(imm).unwrap();
            if imm != 0 {
                b.push_op(OpIAdd {
                    dst: dst_reg.into(),
                    dst_type: DataType::I32,
                    saturate: false,
                    srcs: [dst_reg.into(), (imm & mask).into()],
                });
            }
        } else {
            let dst_b = copy.dst.lanes.as_byte_range().unwrap().start;
            debug_assert!(dst_b < 4);

            // If we don't have an immediate, the best we have is MKVEC.i8 which
            // lets us construct a v4i8 from two i8s and an i16.  Unfortunately,
            // we don't really get to decide which i16.  It always swaps bytes
            // as part of the construction.  For the low half of the register,
            // this is fine.  We can use a write mask to ignore the top bits.
            // For the high half, though, we need two instructions: A MKVEC.i8
            // to do the byte select and a swap to swap back.

            if dst_b < 2 {
                // Only write the bottom half
                dst_reg.range = RegRange::Half0;

                let mut srcs =
                    [Src::from(dst_reg).byte(0), Src::from(dst_reg).byte(1)];
                srcs[usize::from(dst_b)] = copy.src;

                b.push_op(OpMkVecV2I8I16 {
                    dst: dst_reg.into(),
                    srcs,
                    accum: 0_u16.into(),
                });
            } else {
                let mut srcs =
                    [Src::from(dst_reg).byte(2), Src::from(dst_reg).byte(3)];
                srcs[usize::from(dst_b - 2)] = copy.src;

                b.push_op(OpMkVecV2I8I16 {
                    dst: dst_reg.into(),
                    srcs,
                    accum: Src::from(dst_reg).half(0),
                });

                // We now have byte 2 or 3 swapped out for the selected src
                // byte by the two halves have been flipped.
                b.push_op(OpMkVecV2I16 {
                    dst: dst_reg.into(),
                    srcs: [
                        Src::from(dst_reg).half(1),
                        Src::from(dst_reg).half(0),
                    ],
                });
            }
        }
    } else {
        debug_assert!(copy.dst_type.total_bits() >= 16);
        debug_assert!(!copy.dst.lanes.is_byte());

        if let SrcRef::Imm64(imm) = copy.src.src_ref {
            // Handle non-zero 64-bit immediates with two MOV.i32
            assert!(copy.dst_type == DataType::I64);
            let imm = copy.src.swizzle.fold_u64(imm.get()).unwrap();
            b.push_op(OpMov {
                dst: copy.dst.clone().word(0),
                dst_type: DataType::I32,
                src: (imm as u32).into(),
            });
            b.push_op(OpMov {
                dst: copy.dst.word(1),
                dst_type: DataType::I32,
                src: ((imm >> 32) as u32).into(),
            });
        } else if let SrcRef::Imm32(imm) = copy.src.src_ref {
            // Handle non-zero 32-bit immediates with MOV.i32
            assert!(copy.dst_type.total_bits() <= 32);
            let imm = copy.src.swizzle.fold_u32(imm.get()).unwrap();
            let mov_type = if copy.dst.lanes.is_half() {
                DataType::V2I16
            } else {
                DataType::I32
            };
            b.push_op(OpMov {
                dst: copy.dst,
                dst_type: mov_type,
                src: imm.into(),
            });
        } else if copy.dst_type.total_bits() <= 32
            && copy.src.swizzle == Swizzle::NONE
        {
            b.push_op(OpMov {
                dst: copy.dst,
                dst_type: DataType::I32,
                src: copy.src,
            });
        } else {
            // Upgrade to a 32-bit type.  The lane mask will take care of
            // masking off the unused components
            let bits = copy.dst_type.bits();
            assert!(bits <= 64);
            let comps = 32_u8.div_ceil(bits);
            let dst_type = DataType::v(comps, DataType::u(bits));

            // Try to use IAdd if we can to try and prioritize the CVT pipe
            // over the SFU pipe.
            let op = Op::from(OpIAdd {
                dst: copy.dst.clone(),
                dst_type,
                saturate: false,
                srcs: [copy.src.clone(), 0u32.into()],
            });
            let src = &op.srcs()[0];
            if b.model().op_src_supports_swizzle(&op, src, src.swizzle) {
                b.push_op(op);
            } else {
                b.push_op(OpShiftLop {
                    dst: copy.dst,
                    dst_type,
                    shift_op: ShiftOp::None,
                    logic_op: LogicOp::None,
                    not_result: false,
                    src0: copy.src,
                    shift: 0_u8.into(),
                    src2: 0_u32.into(),
                });
            }
        }
    }
}

impl Shader<'_> {
    pub fn lower_copy(&mut self) {
        let model = self.model;
        self.map_instrs(|instr, _| match instr.op {
            Op::Copy(op) => {
                let mut b = InstrBuilder::new(model);
                lower_copy(&mut b, *op);
                b.into_mapped()
            }
            _ => [instr].into(),
        })
    }
}
