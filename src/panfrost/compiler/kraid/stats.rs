// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use std::fmt;

use crate::debug::{DEBUG, DebugFlags};
use crate::ir::*;
use crate::isa::ExecUnit;
use crate::ops::*;

use compiler::bitset::BitSet;
use kraid_bindings::*;

#[derive(Default)]
struct VaStatCount {
    fma: f32,
    cvt: f32,
    sfu: f32,
    v: f32,
    t: f32,
    ls: f32,
    spills: u32,
    fills: u32,
    spill_cost: u32,
    fau_used: u16,
    regs: BitSet<usize>,
}

impl VaStatCount {
    fn mark_reg(&mut self, reg: &RegRef) {
        let idx = usize::from(reg.idx);
        let n = match reg.range {
            RegRange::Regs(n) => n,
            _ => 1,
        };
        self.regs.set_range(idx..(idx + usize::from(n)));
    }

    fn mark_fau(&mut self, fau: &FAURef) {
        if fau.page == FAUPage::User {
            self.fau_used = self.fau_used.max(fau.idx / 2 + 1);
        }
    }

    fn visit_instr(
        &mut self,
        instr: &Instr,
        per_spill_cost: u32,
        model: &dyn Model,
    ) {
        for src in instr.srcs() {
            match &src.src_ref {
                SrcRef::FAU(fau) => self.mark_fau(fau),
                SrcRef::Reg(reg) => self.mark_reg(reg),
                _ => (),
            }
        }

        let mut dst_bytes = 0;
        for dst in instr.dsts() {
            dst_bytes += dst.dst_ref.bytes_written();
            if let DstRef::Reg(reg) = &dst.dst_ref {
                self.mark_reg(reg);
            }
        }

        if matches!(&instr.op, Op::BlendCall(_)) {
            // This will get lowered later into a BLEND + prologue, both
            // should not be counted in the stats metrics
            return;
        }

        let cycles = match &instr.op {
            Op::MMulI32(_) | Op::MMulF32(_) | Op::MMulF16(_) => {
                // While not written in the ISA, MMUL* executes in 4 cycles
                4
            }
            _ => {
                let cycles = dst_bytes.div_ceil(4).max(1);
                cycles * model.op_exec_time(&instr.op).unwrap_or(1)
            }
        };

        match model.op_exec_unit(&instr.op).unwrap() {
            ExecUnit::Cvt => self.cvt += f32::from(cycles),
            ExecUnit::Fma => self.fma += f32::from(cycles),
            ExecUnit::Sfu => self.sfu += f32::from(cycles),
            ExecUnit::Msg => match &instr.op {
                Op::ACmpXchg(_)
                | Op::Atom(_)
                | Op::Atom1(_)
                | Op::LeaBuf(_)
                | Op::LeaPka(_)
                | Op::LeaTex(_)
                | Op::LdAttr(_)
                | Op::LdCvt(_)
                | Op::LdGClk(_)
                | Op::LdPka(_)
                | Op::LdTex(_)
                | Op::Load(_)
                | Op::StCvt(_)
                | Op::Store(_) => {
                    self.ls += 1.0;
                }
                Op::LdVarSpecial(op) if op.name == VarSpecialName::FragZ => {}
                Op::LdVar(_)
                | Op::LdVarBuf(_)
                | Op::LdVarBufFlat(_)
                | Op::LdVarFlat(_)
                | Op::LdVarSpecial(_) => {
                    let total_bytes = if let Op::LdVarBuf(op) = &instr.op {
                        (op.mem_type.bits() / 8) * op.dst_type.comps()
                    } else {
                        dst_bytes
                    };
                    self.v += f32::from(total_bytes.div_ceil(4));
                }
                Op::TexFetch(_)
                | Op::TexGather(_)
                | Op::TexGradient(_)
                | Op::TexSingle(_) => {
                    self.t += 1.0;
                }
                Op::ATest(_)
                | Op::Barrier(_)
                | Op::Blend(_)
                | Op::LdTile(_)
                | Op::StTile(_)
                | Op::ZSEmit(_) => {
                    // These aren't counted
                }
                _ => panic!("Unknown message instruction"),
            },
        }

        match &instr.op {
            Op::Load(op) if op.is_tls => {
                self.fills += 1;
                self.spill_cost =
                    self.spill_cost.saturating_add(per_spill_cost);
            }
            Op::Store(op) if op.is_tls => {
                self.spills += 1;
                self.spill_cost =
                    self.spill_cost.saturating_add(per_spill_cost);
            }
            _ => (),
        }
    }

    fn normalize(&mut self, model: &dyn Model) {
        // v14 doesn't have rates anymore
        if model.arch() >= 14 {
            return;
        }
        let Some(pm) = model.pan_model() else {
            return;
        };

        let rates = &pm.rates;
        assert_ne!(rates.cvt, 0);
        assert_ne!(rates.fma, 0);
        assert_ne!(rates.sfu, 0);
        assert_ne!(rates.varying, 0);
        assert_ne!(rates.texel, 0);
        self.cvt /= rates.cvt as f32;
        self.fma /= rates.fma as f32;
        self.sfu /= rates.sfu as f32;
        self.v /= rates.varying as f32;
        self.t /= rates.texel as f32;
    }

    fn fmt_instr_details(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let stats = [
            ("fma", self.fma),
            ("cvt", self.cvt),
            ("sfu", self.sfu),
            ("v", self.v),
            ("t", self.t),
            ("ls", self.ls),
            ("spill", self.spills as f32),
            ("fill", self.fills as f32),
            ("spill_cost", self.spill_cost as f32),
        ];
        let stats = stats.iter().filter(|(_n, c)| *c != 0.0);

        for (i, (name, unit)) in stats.enumerate() {
            if i != 0 {
                write!(f, ", ")?;
            }
            write!(f, "{name}={unit:.3}")?;
        }
        Ok(())
    }

    fn into_c_stats(
        self,
        s: &Shader,
        instrs: usize,
        loops: usize,
    ) -> valhall_stats {
        let Self { fma, cvt, sfu, .. } = self;

        let alu =
            unsafe { pan_va_compute_alu_bound(s.model.arch(), fma, cvt, sfu) };

        valhall_stats {
            instrs: instrs.try_into().unwrap(),
            cycles: alu.max(self.v).max(self.t).max(self.ls),
            fma,
            cvt,
            sfu,
            alu,
            v: self.v,
            t: self.t,
            ls: self.ls,
            code_size: 0, // Filled in after encoding
            constant_data_size: s
                .constant_pool
                .as_ref()
                .map_or(0, |p| p.data.len())
                .try_into()
                .unwrap(),
            threads: s.model.max_threads(s.info.registers_used).into(),
            loops: loops.try_into().unwrap(),
            spills: self.spills,
            fills: self.fills,
            spill_cost: self.spill_cost,
            registers_used: self.regs.len().try_into().unwrap(),
            uniforms_used: self.fau_used.into(),
        }
    }
}

fn per_spill_cost(s: &Shader, bi: usize) -> u32 {
    // The cost of a spill/fill is 10*depth for now.  This matches the old
    // Bifrost compiler
    u32::try_from(10 * (s.blocks.loop_depth(bi) + 1)).unwrap_or(u32::MAX)
}

/// Print the shader annotating each instruction with its cost
fn report_stats_per_instr(s: &Shader) {
    eprintln!("Kraid shader per-instruction stats details:");
    eprint!(
        "{}",
        Fmt(|f| s.fmt_annotate(f, |bi, _ii, i| {
            let per_spill_cost = per_spill_cost(s, bi);
            let mut stats = VaStatCount::default();

            stats.visit_instr(i, per_spill_cost, s.model);
            stats.normalize(s.model);
            format!("{}", Fmt(|f| stats.fmt_instr_details(f)))
        }))
    )
}

fn get_va_stats(s: &Shader) -> valhall_stats {
    let mut stats = VaStatCount::default();
    let mut instrs = 0usize;
    let mut loops = 0usize;

    for (i, block) in s.blocks.iter().enumerate() {
        if s.blocks.is_loop_header(i) {
            loops += 1;
        }
        instrs += block.instrs.len();

        let per_spill_cost = per_spill_cost(s, i);
        for instr in &block.instrs {
            stats.visit_instr(instr, per_spill_cost, s.model);
        }
    }

    stats.normalize(s.model);
    stats.into_c_stats(s, instrs, loops)
}

impl Shader<'_> {
    pub fn get_stats(&self) -> pan_stats {
        if DEBUG.contains(DebugFlags::PRINT) {
            report_stats_per_instr(self);
        }

        if self.model.arch() >= 9 {
            pan_stats {
                isa: PAN_STAT_VALHALL,
                __bindgen_anon_1: pan_stats__bindgen_ty_1 {
                    valhall: get_va_stats(self),
                },
            }
        } else {
            panic!("Kraid only supports Valhall (v9) and later GPUs");
        }
    }
}
