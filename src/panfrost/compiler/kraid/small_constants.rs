// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use compiler::enum_as_u8::EnumAsU8;
use compiler::smallvec::SmallVec;

use crate::ir::*;
use crate::model::SmallConstantTable;
use crate::ops::LogicOp;

fn supported_swizzles(
    model: &dyn Model,
    op: &Op,
    src: &Src,
    out: &mut Vec<Swizzle>,
) {
    out.clear();
    for swz in model.op_src_supported_swizzles(op, src).iter() {
        let Some(swz) = swz.to_swizzle(op.src_type(src)) else {
            continue;
        };
        debug_assert!(model.op_src_supports_swizzle(op, src, swz));
        out.push(swz);
    }
}

fn supported_mods(
    model: &dyn Model,
    op: &Op,
    src: &Src,
    out: &mut Vec<SrcMod>,
) {
    out.clear();
    for m in SrcMod::VARIANTS.iter() {
        if model.op_src_supports_mod(op, src, m) {
            out.push(m);
        }
    }
}

fn any_imm_srcs(instr: &Instr) -> bool {
    instr
        .srcs()
        .iter()
        .any(|src| matches!(src.src_ref, SrcRef::Imm32(_)))
}

#[derive(Default)]
struct SrcPermutation {
    from: u8,
    to: u8,
}

impl SrcPermutation {
    fn swap(from: u8, to: u8) -> Self {
        SrcPermutation { from, to }
    }

    fn is_empty(&self) -> bool {
        self.from == self.to
    }

    fn apply(&self, instr: &mut Instr) {
        if !self.is_empty() {
            instr.srcs_mut().swap(self.to.into(), self.from.into());
        }
    }

    fn unapply(&self, instr: &mut Instr) {
        // Same as apply but clearer to read where it's used.
        self.apply(instr);
    }

    fn is_legal(&self, model: &dyn Model, instr: &Instr) -> bool {
        if self.is_empty() {
            return true;
        }

        let src_f = instr.srcs().get(self.from as usize).unwrap();
        let src_t = instr.srcs().get(self.to as usize).unwrap();

        model.op_src_supports_swizzle(&instr.op, src_f, src_t.swizzle)
            && model.op_src_supports_swizzle(&instr.op, src_t, src_f.swizzle)
            && model.op_src_supports_mod(&instr.op, src_f, src_t.src_mod)
            && model.op_src_supports_mod(&instr.op, src_t, src_f.src_mod)
    }
}

fn get_commutative_srcs(instr: &Instr) -> SmallVec<SrcPermutation> {
    match &instr.op {
        // All logic ops are commutative in the logic op sources.
        Op::ShiftLop(op) if op.logic_op != LogicOp::None => {
            let no_shift = matches!(&op.shift.src_ref, SrcRef::Zero);
            if no_shift {
                return [SrcPermutation::swap(0, 2)].into();
            } else {
                Default::default()
            }
        }
        _ => Default::default(),
    }
}

struct ScRepl<'a> {
    sc: &'a SmallConstant,
    src_mod: SrcMod,
    swizzle: Swizzle,
}

fn try_as_small_const<'a>(
    src: &Src,
    src_type: DataType,
    sc_table: &'a SmallConstantTable,
    mods: &[SrcMod],
    swizzles: &[Swizzle],
) -> Option<ScRepl<'a>> {
    if src.src_ref == SrcRef::Zero {
        return None;
    }
    let Some(imm_read) = src.resolve_imm(src_type) else {
        return None;
    };
    for swz in swizzles {
        for m in mods {
            for sc in sc_table {
                if Src::from(sc.imm32)
                    .swizzle(*swz)
                    .modify(*m)
                    .resolve_imm(src_type)
                    .is_some_and(|v| v == imm_read)
                {
                    return Some(ScRepl {
                        sc,
                        src_mod: *m,
                        swizzle: *swz,
                    });
                }
            }
        }
    }

    None
}

fn find_lowerable_srcs<'a>(
    instr: &Instr,
    swizzles: &[Vec<Swizzle>],
    mods: &[Vec<SrcMod>],
    sc_table: &'a SmallConstantTable,
) -> SmallVec<(usize, ScRepl<'a>)> {
    let mut result: SmallVec<_> = Default::default();

    for (idx, (src, src_type)) in instr.srcs_types().enumerate() {
        if let Some(repl) = try_as_small_const(
            src,
            src_type,
            sc_table,
            &mods[idx],
            &swizzles[idx],
        ) {
            result.push((idx, repl));
        }
    }

    result
}

#[derive(Default)]
struct VecCtx {
    swizzles: Vec<Vec<Swizzle>>,
    mods: Vec<Vec<SrcMod>>,
}

fn lower_instr(instr: &mut Instr, model: &dyn Model, ctx: &mut VecCtx) {
    let sc_table = &model.fau().small_constants;
    let op = &instr.op;

    ctx.mods.resize(instr.srcs().len(), Default::default());
    ctx.swizzles.resize(instr.srcs().len(), Default::default());

    for (idx, src) in instr.srcs().iter().enumerate() {
        supported_mods(model, op, src, &mut ctx.mods[idx]);
        supported_swizzles(model, op, src, &mut ctx.swizzles[idx]);
    }

    let mut permutation = Default::default();
    let mut replacements: SmallVec<_> = Default::default();

    for p in
        std::iter::once(Default::default()).chain(get_commutative_srcs(instr))
    {
        if !p.is_legal(model, instr) {
            continue;
        }

        p.apply(instr);
        let r = find_lowerable_srcs(instr, &ctx.swizzles, &ctx.mods, sc_table);
        p.unapply(instr);

        if r.len() > replacements.len() {
            replacements = r;
            permutation = p;
        }
    }

    permutation.apply(instr);

    for (target_idx, repl) in replacements {
        let src = &mut instr.srcs_mut()[target_idx];
        src.swizzle = repl.swizzle;
        src.src_mod = repl.src_mod;
        src.src_ref = SrcRef::FAU(FAURef::from(repl.sc).into());

        let src = &instr.srcs()[target_idx];
        let op = &instr.op;
        debug_assert!(model.op_src_supports_swizzle(op, src, src.swizzle));
        debug_assert!(model.op_src_supports_mod(op, src, src.src_mod));
    }
}

impl Shader<'_> {
    pub fn lower_small_constants(&mut self) {
        let mut vec_ctx: VecCtx = Default::default();

        for b in self.blocks.iter_mut() {
            for instr in b.instrs.iter_mut() {
                if !any_imm_srcs(instr) {
                    continue;
                }
                lower_instr(instr, self.model, &mut vec_ctx);
            }
        }
    }
}
