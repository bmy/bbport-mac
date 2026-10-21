// Copyright © 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use crate::{ir::*, swizzle::AsmSwizzleWiden};
use compiler::{enum_as_u8::EnumAsU8, float16::F16};

fn get_other_variants(op: &Op) -> &[DataType] {
    use DataType::*;

    match op.variant() {
        Some(V4I8) => &[V4U8, V4S8],
        Some(V2I16) => &[V2U16, V2S16],
        Some(I32) => &[U32, S32],
        Some(I64) => &[U64, S64],
        _ => &[],
    }
}

#[derive(PartialEq)]
struct SupportedSwizzle {
    swz: Swizzle,
    variant: Option<DataType>,
}

impl SupportedSwizzle {
    const HF0: Self = SupportedSwizzle {
        swz: Swizzle::HF0,
        variant: Some(DataType::F32),
    };
}

impl std::fmt::Display for SupportedSwizzle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(variant) = self.variant {
            write!(f, "({},{})", self.swz, variant)
        } else {
            write!(f, "({},None)", self.swz)
        }
    }
}

fn supported_swizzles(
    model: &dyn Model,
    op: &mut Op,
    src_idx: usize,
    out: &mut Vec<SupportedSwizzle>,
) {
    out.clear();

    let base_variant = op.variant();
    let other_variants: Vec<_> = get_other_variants(op)
        .iter()
        .filter(|v| op.is_valid_variant(**v))
        .map(|v| Some(*v))
        .collect();
    let variants =
        std::iter::once(base_variant).chain(other_variants.into_iter());

    for variant in variants {
        if let Some(variant) = variant {
            op.set_variant(variant);
        }
        let src = &op.srcs()[src_idx];
        for swz in model.op_src_supported_swizzles(op, src).iter() {
            if swz == AsmSwizzleWiden::None {
                continue;
            }
            let Some(swz) = swz.to_swizzle(op.src_type(src)) else {
                continue;
            };

            out.push(SupportedSwizzle { swz, variant });
        }
    }

    if let Some(variant) = base_variant {
        op.set_variant(variant);
    }
}

fn supported_mods(
    model: &dyn Model,
    op: &Op,
    src_idx: usize,
    out: &mut Vec<SrcMod>,
) {
    out.clear();
    let src = &op.srcs()[src_idx];
    for m in SrcMod::VARIANTS.iter() {
        if model.op_src_supports_mod(op, src, m) {
            out.push(m);
        }
    }
}

fn get_imm_bits_read_mask(swz: &Swizzle, src_type: DataType) -> u32 {
    let bytes_read = swz.bytes_read(src_type.total_bytes());

    // Can't read any bytes from the upper word because we are reading from an imm32 here.
    assert!((bytes_read & 0b1111_0000) == 0);

    let mut mask = 0;
    for i in 0..4 {
        if (bytes_read & (1 << i)) != 0 {
            mask |= 0xff << (i * 8);
        }
    }

    mask
}

fn try_f32_as_f16(v: f32) -> Option<F16> {
    let half = F16::from_f32_rtne(v);
    let widened = Swizzle::HF0.fold_u32(half.to_bits() as u32).unwrap();

    if widened == v.to_bits() {
        Some(half)
    } else {
        None
    }
}

#[derive(Debug, PartialEq)]
struct SqueezedImm {
    imm: u32,
    swz: Swizzle,
    variant: Option<DataType>,
}

fn squeeze_imm(
    src: &Src,
    swizzles: &[SupportedSwizzle],
    src_type: DataType,
) -> Option<SqueezedImm> {
    let mut tmp_src = src.clone();
    tmp_src.src_mod = SrcMod::None;
    let Some(swizzled) = tmp_src.resolve_imm(src_type) else {
        return None;
    };

    if src_type.is_float_type() && swizzles.contains(&SupportedSwizzle::HF0) {
        assert!(src_type == DataType::F32);
        assert!((swizzled & 0xffff_ffff << 32) == 0);

        if let Some(half) = try_f32_as_f16(f32::from_bits(swizzled as u32)) {
            return Some(SqueezedImm {
                imm: half.to_bits() as u32,
                swz: Swizzle::HF0,
                variant: None, // we never want to change the variant of f32.
            });
        }
    } else if src_type.is_any_int_type() {
        // For replication, all components are the same, try the lowest one.
        // For a widen, the higher bytes just contain 0x00 or 0xff.
        let ok_bytes_read = [0b1, 0b11, 0b1111];
        let mut best_read = src.swizzle.bytes_read(src_type.total_bytes());
        let mut best_squeezed = None;

        for SupportedSwizzle { swz, variant } in swizzles {
            let bytes_read = swz.bytes_read(src_type.total_bytes());
            if bytes_read >= best_read || !ok_bytes_read.contains(&bytes_read) {
                continue;
            }

            let read_mask = get_imm_bits_read_mask(swz, src_type) as u64;
            let try_imm: u32 = (swizzled & read_mask).try_into().unwrap();

            let Some(resolved) =
                Src::from(try_imm).swizzle(*swz).resolve_imm(src_type)
            else {
                continue;
            };
            if resolved == swizzled {
                best_read = bytes_read;
                best_squeezed = Some(SqueezedImm {
                    imm: try_imm,
                    swz: *swz,
                    variant: *variant,
                });
            }

            // Stop if we can't do better.
            if best_read == 0b1 {
                break;
            }
        }

        return best_squeezed;
    }

    None
}

fn normalize_neg(src: &mut Src, mods: &[SrcMod], src_type: DataType) {
    let SrcRef::Imm32(imm32) = &src.src_ref else {
        return;
    };
    let imm32 = u32::from(*imm32);

    let Some(swizzled) =
        Src::from(imm32).swizzle(src.swizzle).resolve_imm(src_type)
    else {
        return;
    };

    // Check if the most significant bit read from the imm is set.
    // This means a negative value for signed int and float, and is a reasonable choice for
    // unsigned.
    let bytes_read = src.swizzle.bytes_read(src_type.total_bytes());
    if bytes_read == 0 {
        return;
    }
    let highest_byte_read = 7 - bytes_read.leading_zeros();
    let check_bit = 0x80u64 << (8 * highest_byte_read);

    if ((imm32 as u64) & check_bit) == 0 {
        return;
    }
    let try_mod = if src_type.is_float_type() {
        SrcMod::FNeg
    } else {
        SrcMod::BNot
    };
    if !mods.contains(&try_mod) {
        return;
    }

    let imm32_modified = match src.swizzle {
        // If we widen from the lower half to f32, folding with f32 flips the wrong bit.
        Swizzle::HF0 => try_mod.fold_u32(DataType::F16, imm32).unwrap(),
        _ => try_mod.fold_u32(src_type, imm32).unwrap(),
    };

    let resolved = Src::from(imm32_modified)
        .swizzle(src.swizzle)
        .modify(try_mod)
        .resolve_imm(src_type);
    if resolved == Some(swizzled) {
        // For FADD.v2f16 _ 0xbc00.h00            we do not want
        //     FADD.v2f16 _ 0x80003c00.h00.fneg   as the result but
        //     FADD.v2f16 _ 0x3c00.h00.fneg
        // so mask away the bits we don't read anyways.
        let mask = get_imm_bits_read_mask(&src.swizzle, src_type);
        let imm32_modified = imm32_modified & mask;

        src.src_ref = imm32_modified.into();
        src.src_mod = try_mod.modify(src.src_mod);
    }
}

fn normalize_consts_instr(
    instr: &mut Instr,
    swizzles: &mut Vec<SupportedSwizzle>,
    mods: &mut Vec<SrcMod>,
    model: &dyn Model,
) {
    for src_idx in 0..instr.srcs().len() {
        if !matches!(&instr.srcs()[src_idx].src_ref, SrcRef::Imm32(_)) {
            continue;
        }

        supported_swizzles(model, &mut instr.op, src_idx, swizzles);

        let (src, src_type) = instr.op.srcs_types().nth(src_idx).unwrap();
        if let Some(si) = squeeze_imm(src, &swizzles, src_type) {
            let src = &mut instr.op.srcs_mut()[src_idx];
            src.swizzle = si.swz;
            src.src_ref = si.imm.into();
            if let Some(variant) = si.variant {
                instr.set_variant(variant);
            }
        }

        supported_mods(model, &instr.op, src_idx, mods);
        let (src, src_type) = instr.op.srcs_types_mut().nth(src_idx).unwrap();
        normalize_neg(src, &mods, src_type);
    }
}

fn normalize_consts(s: &mut Shader) {
    let mut swizzles = Default::default();
    let mut mods = Default::default();

    let instrs = s.blocks.iter_mut().flat_map(|b| b.instrs.iter_mut());

    for instr in instrs {
        normalize_consts_instr(instr, &mut swizzles, &mut mods, s.model);
    }
}

impl Shader<'_> {
    pub fn opt_normalize_consts(&mut self) {
        normalize_consts(self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        data_type::DataType::*,
        model::model_for_gpu_id,
        ops::{LogicOp, OpFAdd, OpFLog16, OpIAdd, OpShiftLop},
        ssa_value::{AllocSSA, SSAValueAllocator},
    };
    use paste::paste;

    macro_rules! isrc {
        ($imm:expr, $swz:expr, $m:expr) => {
            paste! {
                Src::from($imm).swizzle(Swizzle::$swz).modify(SrcMod::$m)
            }
        };
    }

    fn and(dst_type: DataType, x: Src, y: Src) -> OpShiftLop {
        OpShiftLop {
            dst: DstRef::None.into(),
            dst_type,
            shift_op: Default::default(),
            logic_op: LogicOp::And,
            not_result: false,
            src0: x,
            shift: 0_u8.into(),
            src2: y,
        }
    }

    fn iadd(dst_type: DataType, x: Src, y: Src) -> OpIAdd {
        OpIAdd {
            dst: DstRef::None.into(),
            dst_type,
            saturate: false,
            srcs: [x, y],
        }
    }

    fn fadd(dst_type: DataType, x: Src, y: Src) -> OpFAdd {
        OpFAdd {
            dst: DstRef::None.into(),
            dst_type,
            round: Default::default(),
            clamp: Default::default(),
            srcs: [x, y],
        }
    }

    fn flog16(x: Src) -> OpFLog16 {
        OpFLog16 {
            dst: DstRef::None.into(),
            src: x,
        }
    }

    macro_rules! test_normalize {
        ($name:tt, $any32:ident, $input:expr, $expected:expr) => {
            paste! {
                #[test]
                fn [<test_normalize_ $name>]() {
                    let model = model_for_gpu_id(0xd8000000, 4).unwrap();
                    let mut ssa_alloc = SSAValueAllocator::default();
                    let $any32 = ssa_alloc.alloc_ref(32);

                    let mut instr: Instr = ($input).into();
                    let expected: Instr = ($expected).into();

                    normalize_consts_instr(
                        &mut instr,
                        &mut Default::default(),
                        &mut Default::default(),
                        model.as_ref(),
                    );

                    assert_eq!(instr.variant(), expected.variant());
                    assert_eq!(instr.srcs(), expected.srcs());
                }
            }
        };
        ($name:tt, $input:expr, $expected:expr) => {
            test_normalize!($name, _ignore, $input, $expected);
        };
    }

    fn f16b(f: f32) -> u16 {
        compiler::float16::F16::from_f32_rtne(f).to_bits()
    }

    fn v2f16b(h0: f32, h1: f32) -> u32 {
        let low = compiler::float16::F16::from_f32_rtne(h0).to_bits() as u32;
        let high = compiler::float16::F16::from_f32_rtne(h1).to_bits() as u32;
        high << 16 | low
    }

    test_normalize!(
        fadd_f32_m1,
        any32,
        fadd(F32, isrc!(-1.0, NONE, None), any32.clone().into()),
        fadd(F32, isrc!(f16b(1.0), HF0, FNeg), any32.clone().into())
    );

    test_normalize!(
        fadd_f32_m1_neg,
        any32,
        fadd(F32, isrc!(-1.0, NONE, FNeg), any32.clone().into()),
        fadd(F32, isrc!(f16b(1.0), HF0, None), any32.clone().into())
    );

    test_normalize!(
        fadd_f32_m1_abs,
        any32,
        fadd(F32, isrc!(-1.0, NONE, FAbs), any32.clone().into()),
        fadd(F32, isrc!(f16b(1.0), HF0, FAbs), any32.clone().into())
    );

    test_normalize!(
        fadd_v2f16_m1_1,
        fadd(V2F16, isrc!(v2f16b(-1.0, 1.0), NONE, None), 0u8.into()),
        fadd(V2F16, isrc!(v2f16b(-1.0, 1.0), NONE, None), 0u8.into())
    );

    test_normalize!(
        fadd_v2f16_1_m1,
        fadd(V2F16, isrc!(v2f16b(1.0, -1.0), NONE, None), 0u8.into()),
        fadd(V2F16, isrc!(v2f16b(-1.0, 1.0), NONE, FNeg), 0u8.into())
    );

    test_normalize!(
        fadd_v2f16_m1_1_swap,
        fadd(V2F16, isrc!(v2f16b(-1.0, 1.0), H10, None), 0u8.into()),
        fadd(V2F16, isrc!(v2f16b(-1.0, 1.0), H10, None), 0u8.into())
    );

    test_normalize!(
        fadd_v2f16_repl,
        fadd(V2F16, isrc!(f16b(-1.0), H00, None), 0u8.into()),
        fadd(V2F16, isrc!(f16b(1.0), H00, FNeg), 0u8.into())
    );

    test_normalize!(
        flog_f16_m1,
        flog16(isrc!(f16b(-1.0), NONE, None)),
        flog16(isrc!(f16b(1.0), NONE, FNeg))
    );

    test_normalize!(
        and_u32,
        any32,
        and(U32, 0xffu32.into(), any32.clone().into()),
        and(U32, isrc!(0xffu32, widen_u8(0), None), any32.clone().into())
    );

    test_normalize!(
        and_v4u8,
        and(V4U8, 0u8.into(), isrc!(0xfa_bc_12_34_u32, NONE, None)),
        and(V4U8, 0u8.into(), isrc!(0x05_43_ed_cb_u32, NONE, BNot))
    );

    test_normalize!(
        and_u32_bnot,
        and(U32, 0u8.into(), 0xfffffff0u32.into()),
        and(U32, 0u8.into(), isrc!(0x0000000fu32, NONE, BNot))
    );

    test_normalize!(
        iadd_i64_to_u64,
        iadd(I64, isrc!(0x1234u32, widen_u32(0), None), 0u8.into()),
        iadd(U64, isrc!(0x1234u32, widen_u16(0), None), 0u8.into())
    );

    test_normalize!(
        iadd_i32_to_s32,
        iadd(I32, isrc!(-5i32 as u32, NONE, None), 0u8.into()),
        iadd(S32, isrc!(-5i8 as u8, widen_s8(0), None), 0u8.into())
    );
}
