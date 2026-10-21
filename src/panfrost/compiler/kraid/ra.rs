// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

use crate::builder::{SSABuilder, SSAInstrBuilder};
use crate::debug::*;
use crate::ir::*;
use crate::liveness::*;
use crate::ops::{OpBranch, OpPhiSrc, OpRegOut};
use crate::parallel_copy::*;
use crate::phi::PhiMap;
use crate::ssa_value::*;
use compiler::bitset::*;
use compiler::cfg::CFG;
use compiler::union_find::UnionFind;
use rustc_hash::{FxBuildHasher, FxHashMap};
use std::collections::VecDeque;
use std::ops::{BitOrAssign, Range};

/// A structure that models an arena from which to allocate SSA values.  An
/// arena may be backed by registers or memory.  This struct mostly isn't
/// stateful (besides a count of how many bytes have been used) and exists to
/// tells the allocator about the arena (its size, etc.) and provides methods
/// for mapping byte ranges used by RA back into the RegRef etc. used by
/// instructions.
struct Arena {
    /// Limit on the number of bytes allocated
    limit: u16,

    /// Number of bytes actually used
    used: std::cell::Cell<u16>,

    /// Granularity (in bytes) to return from bytes_used
    granularity: u8,

    /// True if this is a memory register file
    is_mem: bool,

    /// True if this arena wants round-robin allocation
    round_robin: bool,

    /// For memory, the offset into the TLS where we can start allocating
    tls_offset: u16,

    /// True if we are on v9-14 and in 32-reg mode.  In this case, the middle
    /// 32 registers of the register arena are missing.  To deal with this, we
    /// assume a contiguous 32 register arena and place the high regs in 48..64
    /// as part of reg_for_bytes().
    is_v9_32reg: bool,

    /// True if we are on v9-14 and in 64-reg mode.
    is_v9_64reg: bool,
}

impl Arena {
    /// Creates a new register arena
    pub fn new_reg(model: &dyn Model, limit: u16) -> Arena {
        let granularity = if model.arch() < 15 { 32 * 4 } else { 16 * 4 };
        Arena {
            limit: limit.next_multiple_of(granularity.into()),
            used: 0.into(),
            granularity,
            is_mem: false,
            round_robin: true,
            tls_offset: 0,
            is_v9_32reg: model.arch() < 15 && limit <= 32 * 4,
            is_v9_64reg: model.arch() < 15 && limit > 32 * 4,
        }
    }

    /// Creates a new memory arena, starting at `tls_start`.  The newly created
    /// arena subsumes the entire TLS range and the total amount of TLS used by
    /// the shader after spill allocation is returned by bytes_used().  This
    /// allows us to deal with any alignments inside the arena.
    fn new_mem(_model: &dyn Model, tls_start: u16) -> Arena {
        let granularity = 8;
        Arena {
            limit: u16::MAX - tls_start,
            used: 0.into(),
            granularity,
            is_mem: true,
            round_robin: false,
            tls_offset: tls_start.next_multiple_of(granularity.into()),
            is_v9_32reg: false,
            is_v9_64reg: false,
        }
    }

    /// Creates a new register arena for blend shaders
    pub fn new_blend(model: &dyn Model) -> Arena {
        let limit = 16 * 4;
        Arena {
            limit: limit.into(),
            used: 0.into(),
            granularity: limit,
            is_mem: false,
            round_robin: true,
            tls_offset: 0,
            is_v9_32reg: model.arch() < 15,
            is_v9_64reg: false,
        }
    }

    /// Returns the number of bytes used from this arena.  This will be updated
    /// as we allocate and can be queried after RA is complete to know the
    /// final amount we need to report to the driver.
    pub fn bytes_used(&self) -> u16 {
        let used = self.used.get();
        if self.is_v9_64reg && used > 16 * 4 {
            // If we've run in 64-reg mode up until now and we've used more
            // than the first 16 then we've probably used some out of the
            // middle and it's not safe to report 32 registers.
            64 * 4
        } else {
            used.next_multiple_of(self.granularity.into())
        }
    }

    /// Returns the same as bytes_used but in units of 32-bit registers
    pub fn regs_used(&self) -> u8 {
        debug_assert!(!self.is_mem);
        debug_assert!(self.bytes_used() % 4 == 0);
        (self.bytes_used() / 4).try_into().unwrap()
    }

    /// Returns true if the given SSAValue is in this arena
    pub fn contains_ssa(&self, ssa: &SSAValue) -> bool {
        ssa.is_mem() == self.is_mem
    }

    /// Returns true if the given SSARef is in this arena
    pub fn contains_ref(&self, vec: &SSARef) -> bool {
        let contains = self.contains_ssa(&vec[0]);
        for i in 1..vec.len() {
            debug_assert_eq!(self.contains_ssa(&vec[i]), contains);
        }
        contains
    }

    /// Returns true if the given RegRef is entirely out-of-bounds
    pub fn is_reg_oob(&self, reg: &RegRef) -> bool {
        self.reg_to_bytes(reg).end >= self.limit()
    }

    /// Returns true if this arena is for registers.  This controls whether
    /// or not we handle OpRegIn and OpRegOut
    pub fn is_reg(&self) -> bool {
        !self.is_mem
    }

    /// Returns true if this arena is for memory.
    pub fn is_mem(&self) -> bool {
        self.is_mem
    }

    /// Returns the maximum number of bytes that can be allocated from this
    /// arena.
    pub fn limit(&self) -> u16 {
        self.limit
    }

    /// Returns true if the given range maps to a contiguous range in the arena
    pub fn is_contiguous(&self, bytes: Range<u16>) -> bool {
        if self.is_v9_32reg {
            bytes.end <= 16 * 4 || bytes.start >= 16 * 4
        } else {
            true
        }
    }

    fn mark_used(&self, bytes: Range<u16>) {
        debug_assert!(self.is_contiguous(bytes.clone()));
        self.used.set(self.used.get().max(bytes.end));
    }

    /// Maps a byte range to a [RegRef]
    pub fn reg_for_bytes(&self, mut bytes: Range<u16>) -> RegRef {
        debug_assert!(!self.is_mem);
        self.mark_used(bytes.clone());
        if self.is_v9_32reg {
            if bytes.start < (16 * 4) {
                assert!(bytes.end <= 16 * 4);
            } else {
                assert!(bytes.end <= 32 * 4);
                bytes.start += 32 * 4;
                bytes.end += 32 * 4;
            }
        }
        RegRef::from_byte_range(bytes.clone()).unwrap()
    }

    /// Maps a [RegRef] for a byte range
    pub fn reg_to_bytes(&self, reg: &RegRef) -> Range<u16> {
        debug_assert!(!self.is_mem);
        let mut bytes = reg.byte_range();
        if self.is_v9_32reg {
            if bytes.start < (16 * 4) {
                assert!(bytes.end <= 16 * 4);
            } else {
                assert!(bytes.start >= 48 * 4);
                assert!(bytes.end <= 64 * 4);
                bytes.start -= 32 * 4;
                bytes.end -= 32 * 4;
            }
        }
        self.mark_used(bytes.clone());
        bytes
    }

    /// Maps a byte range to a [MemRef]
    fn mem_for_bytes(&self, mut bytes: Range<u16>) -> MemRef {
        debug_assert!(self.is_mem);
        bytes.start += self.tls_offset;
        bytes.end += self.tls_offset;
        self.mark_used(bytes.clone());
        MemRef::from_byte_range(bytes).unwrap()
    }

    /// Maps a byte range to a [Src]
    pub fn src_for_bytes(&self, bytes: Range<u16>) -> Src {
        if self.is_mem {
            self.mem_for_bytes(bytes).into()
        } else {
            self.reg_for_bytes(bytes).into()
        }
    }

    /// Maps a byte range to a [Dst]
    pub fn dst_for_bytes(&self, bytes: Range<u16>) -> Dst {
        if self.is_mem {
            self.mem_for_bytes(bytes).into()
        } else {
            self.reg_for_bytes(bytes).into()
        }
    }
}

fn swizzle_byte_range(bytes: Range<u16>, swizzle: Swizzle) -> Range<u16> {
    let swz_bytes = match swizzle {
        Swizzle::B0000 => 0..1,
        Swizzle::B1111 => 1..2,
        Swizzle::B2222 => 2..3,
        Swizzle::B3333 => 3..4,
        Swizzle::H00 => 0..2,
        Swizzle::H11 => 2..4,
        Swizzle::NONE => return bytes,
        _ => panic!("Not a byte range select swizzle"),
    };
    let start = bytes.start + swz_bytes.start;
    let end = bytes.start + swz_bytes.end;
    debug_assert!(end <= bytes.end);
    start..end
}

fn widen_lanes(lanes: DstLanes) -> DstLanes {
    use DstLanes::*;
    match lanes {
        None => AnyB,
        All => panic!("Everything supports ALL"),
        AnyB => AnyH,
        AnyH | H0 | H1 => All,
        AnyHF | HF0 | HF1 => panic!("DstLanes::HF* cannot be widened"),
        B0 | B1 => H0,
        B2 | B3 => H1,
    }
}

fn fold_lanes(alloc_lanes: DstLanes, dst_lanes: DstLanes) -> DstLanes {
    use DstLanes::*;
    match alloc_lanes {
        All => {
            // F16 narrows have to map to map to a HF lanes
            assert!(!matches!(dst_lanes, AnyHF | HF0 | HF1));
            alloc_lanes
        }
        AnyB | B0 | B1 | B2 | B3 => {
            assert!(dst_lanes == AnyB || dst_lanes == alloc_lanes);
            alloc_lanes
        }
        AnyH => {
            assert!(dst_lanes == AnyH);
            AnyH
        }
        H0 => match dst_lanes {
            AnyHF | HF0 => HF0,
            AnyB | AnyH | B0 | B1 | H0 => H0,
            _ => panic!("Invalid dst_lanes: {dst_lanes}"),
        },
        H1 => match dst_lanes {
            AnyHF | HF1 => HF1,
            AnyB | AnyH | B2 | B3 | H1 => H1,
            _ => panic!("Invalid dst_lanes: {dst_lanes}"),
        },
        None | AnyHF | HF0 | HF1 => {
            panic!("Invalid alloc_lanes: {alloc_lanes}");
        }
    }
}

fn aligned_u16_range(start: u16, len: u16) -> Range<u16> {
    debug_assert!(len.is_power_of_two());
    let start = start & !(len - 1);
    start..(start + len)
}

pub fn instr_clobbered_regs(model: &dyn Model, op: &Op) -> Vec<RegRef> {
    match op {
        Op::BlendCall(_) => {
            let link = model.preload_reg(PreloadReg::BlendReturnAddr).unwrap();
            let lower16 = RegRef::new(0, RegRange::Regs(16));
            vec![lower16, link]
        }
        _ => Vec::new(),
    }
}

/// A register alignment constraint, specified as an 8-bit bitfield of possible
/// offsets from an even register.  For registers which do not need to be even-
/// aligned, they simply repeat the constraint in both halves of the u8.
#[derive(Clone, Copy, Default)]
struct RegAlignConstraint(u8);

impl RegAlignConstraint {
    fn new() -> Self {
        Default::default()
    }

    fn for_align(mul: u8, offset: u8) -> Self {
        let mask = match mul {
            1 => 0xff,
            2 => 0x55 << offset,
            4 => 0x11 << offset,
            8 => 0x01 << offset,
            _ => panic!("Invalid register alignment multiplier"),
        };
        RegAlignConstraint(mask)
    }

    fn for_dst_lanes(dst_lanes: DstLanes) -> Self {
        let (align_mul, align_off) = dst_lanes.align();
        RegAlignConstraint::for_align(align_mul, align_off)
    }

    fn for_dst_lanes_set(dst_lanes_set: DstLanesSet) -> Self {
        let mut align = RegAlignConstraint::new();
        for lanes in dst_lanes_set.iter() {
            align |= RegAlignConstraint::for_dst_lanes(lanes);
        }
        align
    }

    fn for_op_dst(model: &dyn Model, op: &Op, dst: &Dst) -> (u8, Self) {
        let supported_lanes = model.op_dst_supported_lanes(op);
        let b_lanes = DstLanesSet::from_array([
            DstLanes::B0,
            DstLanes::B1,
            DstLanes::B2,
            DstLanes::B3,
        ]);
        let h_lanes = DstLanesSet::from_array([DstLanes::H0, DstLanes::H1]);

        let bytes = dst.dst_ref.bytes_written();
        if bytes > 4 {
            // Valhall requires that 64-bit destinations and staging
            // registers writing more than a single register use an even
            // register.
            debug_assert_eq!(dst.lanes, DstLanes::All);
            debug_assert!(supported_lanes.contains(DstLanes::All));
            (bytes, RegAlignConstraint::for_align(8, 0))
        } else if model.op_dst_is_staging_reg(op) {
            // Staging register writes generally can't be widened.
            debug_assert!(supported_lanes.contains(dst.lanes));

            let align = if dst.lanes == DstLanes::AnyB {
                RegAlignConstraint::for_dst_lanes_set(supported_lanes & b_lanes)
            } else if dst.lanes == DstLanes::AnyH {
                RegAlignConstraint::for_dst_lanes_set(supported_lanes & h_lanes)
            } else {
                RegAlignConstraint::for_dst_lanes(dst.lanes)
            };

            // Staging register writes respect lanes in the sense that that's
            // where they put the data but they may not do partial writes
            // correctly.  We need to request a whole register, regardless of
            // destination size or alignment.
            (4, align)
        } else {
            // For ALU ops, 8 and 16-bit destinations might need to be widened
            let mut alloc_lanes = dst.lanes;
            while !supported_lanes.contains(alloc_lanes) {
                alloc_lanes = widen_lanes(alloc_lanes);
            }
            let alloc_bytes = alloc_lanes.bytes(4);
            debug_assert!(alloc_bytes >= bytes);

            // Take the align from the original destination and mask off any
            // alignments that the op disallows.  In the common case, nothing
            // is masked off because the ops we widen typically support both
            // DstLanes::H0 and DstLanes::H1.
            let mut align = RegAlignConstraint::for_dst_lanes(dst.lanes);
            if alloc_lanes == DstLanes::AnyB {
                for lanes in (b_lanes - supported_lanes).iter() {
                    let byte_range = lanes.as_byte_range().unwrap();
                    debug_assert!(byte_range.start < 4);
                    debug_assert_eq!(byte_range.len(), 1);
                    align.0 &= !(0b00010001 << byte_range.start);
                }
            } else if alloc_lanes == DstLanes::AnyH {
                for lanes in (h_lanes - supported_lanes).iter() {
                    let byte_range = lanes.as_byte_range().unwrap();
                    debug_assert!(byte_range.start < 4);
                    debug_assert_eq!(byte_range.len(), 2);
                    align.0 &= !(0b00110011 << byte_range.start);
                }
            }
            debug_assert!(!align.is_empty());
            (alloc_bytes, align)
        }
    }

    fn for_op_src(model: &dyn Model, op: &Op, src: &Src) -> Self {
        let src_type = op.src_type(src);
        let bytes = src.src_ref.bytes_read();

        if src_type == DataType::SR {
            assert!(src.swizzle.is_none());
            assert!(bytes % 4 == 0);
        }

        if bytes > 4 {
            // Valhall requires that 64-bit sources and staging registers
            // reading more than a single register use an even register.
            Self::for_align(8, 0)
        } else if src_type == DataType::SR {
            debug_assert!(bytes == 4);
            Self::for_align(4, 0)
        } else {
            let swizzles: &[(u8, Swizzle)] = match bytes {
                1 => &[
                    (0, Swizzle::B0000),
                    (1, Swizzle::B1111),
                    (2, Swizzle::B2222),
                    (3, Swizzle::B3333),
                ],
                2 => &[(0, Swizzle::H00), (2, Swizzle::H11)],
                4 => {
                    &[(0, Swizzle::NONE), (0, Swizzle::W00), (4, Swizzle::W11)]
                }
                _ => panic!("Invalid SSA value size"),
            };

            let align_mul = if model.op_src_is_64bit(op, src) { 8 } else { 4 };

            let mut align = Self::new();
            for &(b, s) in swizzles {
                let Some(s) = s.swizzle(src.swizzle) else {
                    continue;
                };
                if model.op_src_supports_swizzle(op, src, s) {
                    align |= Self::for_align(align_mul, b);
                }
            }
            align
        }
    }

    fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Return the maximal alignment pair (mul, offset) which every value which
    /// satisfies this constraint will also satisfy.
    fn max_align(&self) -> (u8, u8) {
        assert!(!self.is_empty());
        let first_log2 = self.0.trailing_zeros().try_into().unwrap();
        if self.0 == (1_u8 << first_log2) {
            (8, first_log2)
        } else if (self.0 & !(0x11 << first_log2)) == 0 {
            (4, first_log2 % 4)
        } else if (self.0 & !(0x55 << first_log2)) == 0 {
            (2, first_log2 % 2)
        } else {
            (1, 0)
        }
    }

    fn satisfied(&self, b: usize) -> bool {
        (self.0 & (1 << (b % 8))) != 0
    }
}

impl std::ops::BitAndAssign for RegAlignConstraint {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0
    }
}

impl std::ops::BitOrAssign for RegAlignConstraint {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0
    }
}

struct SSAAffinity {
    /// For each byte of a register pair, an estimate of how much it would cost
    /// to place the SSAValue starting at that byte.
    align_cost: [u8; 8],

    /// SSAValue of the vector representative, if any, or None if we are the
    /// representative.  This is the first SSAValue defined (in program order)
    /// out of the first vector SSARef that uses this SSAValue.
    vec_repr: Option<SSAValue>,

    /// If this SSAValue is part of a vector, the offset (in bytes) from the
    /// representative to this SSA value.  If we are the representative, this
    /// is the offset from the start of the vector.
    vec_offset: i8,

    /// The size of the vector use of this SSAValue, or zero
    vec_size: u8,

    /// Preferred register, in bytes, or u16::MAX
    reg_byte: u16,
}

impl Default for SSAAffinity {
    fn default() -> SSAAffinity {
        SSAAffinity::new()
    }
}

impl SSAAffinity {
    const fn new() -> SSAAffinity {
        SSAAffinity {
            align_cost: [0_u8; 8],
            vec_repr: None,
            vec_offset: 0,
            vec_size: 0,
            reg_byte: u16::MAX,
        }
    }

    fn add_align(&mut self, align: RegAlignConstraint) {
        // Add one for every byte not in the align mask
        for i in 0..8 {
            if (align.0 & (1 << i)) == 0 {
                let byte = &mut self.align_cost[i];
                *byte = (*byte).saturating_add(1);
            }
        }
    }

    #[inline]
    fn align_cost(&self, b: u16) -> u8 {
        self.align_cost[usize::from(b % 8)]
    }

    fn reg_byte(&self) -> Option<u16> {
        if self.reg_byte == u16::MAX {
            None
        } else {
            Some(self.reg_byte)
        }
    }
}

struct SSAVecComp {
    byte: u8,
    ssa: SSAValue,
    ord: u32,
}

struct SSAVecRepr {
    size: u8,
    comps: u8,
    repr: Option<SSAVecComp>,
}

impl SSAVecRepr {
    fn new(size: u8) -> SSAVecRepr {
        SSAVecRepr {
            size,
            comps: 0,
            repr: None,
        }
    }

    fn add_comp(&mut self, byte: u8, ssa: SSAValue, ord: u32) {
        debug_assert_ne!(ord, 0);
        let comp = SSAVecComp { byte, ssa, ord };
        if let Some(repr) = &mut self.repr {
            if comp.ord < repr.ord {
                *repr = comp;
            }
        } else {
            self.repr = Some(comp);
        }
        self.comps += 1;
    }

    fn for_srcs(
        size: u8,
        srcs: &[Src],
        src_bytes: u8,
        def_order: &SSAValueIndexedVec<u32>,
    ) -> SSAVecRepr {
        debug_assert_eq!(
            usize::from(size),
            srcs.len() * usize::from(src_bytes)
        );

        let mut repr = SSAVecRepr::new(size);
        for (i, src) in srcs.iter().enumerate() {
            if let Some(vec) = src.as_ssa() {
                debug_assert_eq!(vec.comps(), 1);
                debug_assert_eq!(vec.bytes(), src_bytes);
                let byte = u8::try_from(i).unwrap() * src_bytes;
                repr.add_comp(byte, vec[0], def_order[vec[0]]);
            }
        }
        repr
    }

    fn for_ssa_ref(
        vec: &SSARef,
        def_order: &SSAValueIndexedVec<u32>,
    ) -> SSAVecRepr {
        let mut repr = SSAVecRepr::new(vec.bytes());
        for (i, ssa) in vec.iter().enumerate() {
            let byte = u8::try_from(i).unwrap() * 4;
            repr.add_comp(byte, *ssa, def_order[ssa]);
        }
        repr
    }

    fn set_affinity(&self, byte: u8, ssa: &SSAValue, a: &mut SSAAffinity) {
        a.add_align(RegAlignConstraint::for_align(self.size.min(8), byte));

        let repr = self.repr.as_ref().unwrap();
        if self.comps > 1 && a.vec_size == 0 {
            if repr.ssa == *ssa {
                a.vec_repr = None;
                a.vec_offset = repr.byte as i8;
            } else {
                a.vec_repr = Some(repr.ssa);
                a.vec_offset = (byte as i8) - (repr.byte as i8);
            }
            a.vec_size = self.size;
        }
    }
}

struct AffinityMapBuilder<'a> {
    model: &'a dyn Model,
    arena: &'a Arena,
    phi_map: &'a PhiMap,
    ssa_affinities: SSAValueIndexedVec<SSAAffinity>,
    phi_webs: UnionFind<SSAValue, FxBuildHasher>,
    def_order: SSAValueIndexedVec<u32>,
}

impl AffinityMapBuilder<'_> {
    fn add_instr(&mut self, bl: &BlockLiveness, instr: &Instr) {
        let def_order = &mut self.def_order;
        match &instr.op {
            Op::MkVecV2I16(op) => {
                let repr = SSAVecRepr::for_srcs(4, &op.srcs, 2, def_order);
                for (i, src) in op.srcs.iter().enumerate() {
                    if let Some(vec) = src.as_ssa() {
                        debug_assert_eq!(vec.comps(), 1);
                        let a = &mut self.ssa_affinities[vec[0]];
                        let i = u8::try_from(i).unwrap();
                        repr.set_affinity(i * 2, &vec[0], a);
                    }
                }
            }
            Op::MkVecV2I8(op) => {
                let repr = SSAVecRepr::for_srcs(2, &op.srcs, 1, def_order);
                for (i, src) in op.srcs.iter().enumerate() {
                    if let Some(vec) = src.as_ssa() {
                        debug_assert_eq!(vec.comps(), 1);
                        let a = &mut self.ssa_affinities[vec[0]];
                        let i = u8::try_from(i).unwrap();
                        repr.set_affinity(i, &vec[0], a);
                    }
                }
            }
            Op::MkVecV4I8(op) => {
                let repr = SSAVecRepr::for_srcs(4, &op.srcs, 1, def_order);
                for (i, src) in op.srcs.iter().enumerate() {
                    if let Some(vec) = src.as_ssa() {
                        debug_assert_eq!(vec.comps(), 1);
                        let a = &mut self.ssa_affinities[vec[0]];
                        let i = u8::try_from(i).unwrap();
                        repr.set_affinity(i, &vec[0], a);
                    }
                }
            }
            Op::PhiSrc(op) => {
                // As a heuristic, we skip the union if the phi source
                // interferes with the phi destination (equivalently: the phi
                // source is live-out of the source block).  These phis could
                // never be coalesced, so the union can only hurt (and it does
                // in practice in complex web scenarios). Note this case is
                // only possible because we do not lower the input program to
                // conventional SSA (CSSA) form.
                if let SrcRef::SSA(src_vec) = &op.src.src_ref {
                    if src_vec.bytes() == op.phi.bytes() {
                        let dst_vec = self.phi_map.get_dst_ssa(&op.phi);
                        assert_eq!(src_vec.len(), dst_vec.len());
                        for (dst_ssa, src_ssa) in dst_vec.iter().zip(src_vec) {
                            if bl.is_live_out(&src_ssa) {
                                continue;
                            }
                            self.phi_webs.union(*dst_ssa, *src_ssa);
                        }
                    }
                }
            }
            Op::RegOut(op) => {
                if let SrcRef::SSA(vec) = &op.src.src_ref {
                    if self.arena.contains_ref(vec) {
                        debug_assert_eq!(op.reg.bytes(), vec.bytes());
                        let bytes = self.arena.reg_to_bytes(&op.reg);
                        for (ssa, bytes) in vec.iter_zip_bytes(bytes) {
                            self.ssa_affinities[ssa].reg_byte = bytes.start;
                        }
                    }
                }
            }
            _ => {
                for src in instr.srcs() {
                    let SrcRef::SSA(vec) = &src.src_ref else {
                        continue;
                    };

                    if self.arena.contains_ref(vec) {
                        if let Some(reg) =
                            self.model.op_fixed_src_reg(&instr.op, src)
                        {
                            let bytes = self.arena.reg_to_bytes(&reg);
                            for (ssa, bytes) in vec.iter_zip_bytes(bytes) {
                                self.ssa_affinities[ssa].reg_byte = bytes.start;
                            }
                        }
                    }

                    if vec.comps() > 1 {
                        let repr = SSAVecRepr::for_ssa_ref(vec, def_order);
                        for (i, ssa) in vec.iter().enumerate() {
                            let a = &mut self.ssa_affinities[ssa];
                            let i = u8::try_from(i).unwrap();
                            repr.set_affinity(i * 4, ssa, a);
                        }
                    } else {
                        debug_assert!(vec.comps() == 1);
                        let a = &mut self.ssa_affinities[vec[0]];

                        // Scalars have alignment based on supported
                        // swizzles
                        a.add_align(RegAlignConstraint::for_op_src(
                            self.model, &instr.op, src,
                        ));
                    }
                }
            }
        }
    }
}

struct AffinityMap {
    ssa_affinities: SSAValueIndexedVec<SSAAffinity>,
    phi_webs: SSAValueIndexedVec<Option<SSAValue>>,
}

impl AffinityMap {
    fn for_shader(
        s: &Shader,
        arena: &Arena,
        live: &Liveness,
        phi_map: &PhiMap,
    ) -> AffinityMap {
        let ssa_count = s.ssa_alloc.count();
        let mut b = AffinityMapBuilder {
            model: s.model,
            arena,
            phi_map,
            ssa_affinities: SSAValueIndexedVec::with_count(ssa_count),
            phi_webs: UnionFind::new(),
            def_order: SSAValueIndexedVec::with_count(ssa_count),
        };

        let mut def_idx = (1_u32..).into_iter();
        for (bi, block) in s.blocks.iter().enumerate() {
            let bl = live.block(bi);

            for instr in &block.instrs {
                b.add_instr(bl, instr);

                for ssa in instr.iter_ssa_defs() {
                    b.def_order[ssa] = def_idx.next().unwrap();
                }
            }
        }

        // Flatten the UnionFind to an SSAValueIndexedVec for efficiency
        let mut phi_webs: SSAValueIndexedVec<Option<SSAValue>> =
            SSAValueIndexedVec::with_count(s.ssa_alloc.count());
        for (k, v) in b.phi_webs.into_iter() {
            debug_assert_eq!(k.bits(), v.bits());
            phi_webs[k] = Some(v);
        }

        AffinityMap {
            ssa_affinities: b.ssa_affinities,
            phi_webs,
        }
    }

    fn get_ssa(&self, ssa: SSAValue) -> &SSAAffinity {
        &self.ssa_affinities[ssa]
    }

    fn align_cost(&self, vec: &SSARef, mut b: u16) -> u8 {
        let mut cost = 0_u8;
        for ssa in vec {
            debug_assert!(
                (b % u16::from(ssa.bytes())) == 0,
                "Misaligned byte offset: {b} for {ssa}"
            );
            let a = self.get_ssa(*ssa);
            cost = cost.saturating_add(a.align_cost(b));
            b += u16::from(ssa.bytes());
        }
        cost
    }

    fn reg_bytes(&self, vec: &SSARef) -> Option<Range<u16>> {
        let start = self.ssa_affinities[vec[0]].reg_byte()?;
        let mut end = start + u16::from(vec[0].bytes());
        for i in 1..vec.len() {
            let b = self.ssa_affinities[vec[i]].reg_byte()?;
            if b != end {
                return None;
            }
            end = b + u16::from(vec[i].bytes());
        }
        Some(start..end)
    }
}

struct WrapOnceCounter {
    start: usize,
    limit: usize,
    cur: usize,
}

impl WrapOnceCounter {
    fn new(mut start: usize, limit: usize) -> WrapOnceCounter {
        if start >= limit {
            assert!(start == limit);
            start = 0;
        }
        WrapOnceCounter {
            start,
            limit,
            cur: start,
        }
    }

    fn get(&self) -> Option<usize> {
        if self.limit == 0 {
            None
        } else {
            Some(self.cur)
        }
    }

    /// Advance the counter forwards, returns false if we hit the end
    fn advance(&mut self, val: usize) {
        debug_assert!(val > self.cur);
        if val < self.limit {
            self.cur = val;
        } else {
            // Otherwise, wrap around
            self.limit = self.start;
            self.start = 0;
            self.cur = 0;
        }
    }
}

#[derive(Default)]
struct PinnedByteSet(BitSet<usize>);

impl PinnedByteSet {
    fn bytes_are_unpinned(&self, bytes: Range<u16>) -> bool {
        let bytes = bytes.start.into()..bytes.end.into();
        self.0.all_unset_in_range(bytes)
    }

    fn clear(&mut self) {
        self.0.clear()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn pin_bytes(&mut self, bytes: Range<u16>) {
        debug_assert!(self.bytes_are_unpinned(bytes.clone()));
        self.pin_bytes_no_check(bytes);
    }

    fn pin_bytes_no_check(&mut self, bytes: Range<u16>) {
        let bytes = bytes.start.into()..bytes.end.into();
        self.0.set_range(bytes);
    }
}

impl BitOrAssign<&PinnedByteSet> for PinnedByteSet {
    fn bitor_assign(&mut self, other: &PinnedByteSet) {
        self.0 |= other.0.s(..);
    }
}

struct LocalRegAlloc<'a> {
    model: &'a dyn Model,

    /// Allocation arena
    arena: &'a Arena,

    /// Affinity map
    affinities: &'a AffinityMap,

    /// Map from phi web representatives to the most recent byte range assigned
    /// to some SSAValue in the that web.
    phi_web_bytes: FxHashMap<SSAValue, Range<u16>>,

    /// Bitset of bytes currently used
    used: BitSet<usize>,

    /// Local map of SSA indices to bytes.  Only defined for live SSA values
    idx_bytes: Vec<Range<u16>>,

    /// Map of bytes back to SSA value indices.  This is only defined for bytes
    /// marked used.
    byte_idx: Vec<u32>,

    /// When searching for a free byte range, the byte to start at.
    search_start: std::cell::Cell<u16>,

    /// PinnedByteSet for bytes that are live-in to the current instruction
    pinned_in: PinnedByteSet,

    /// PinnedByteSet for bytes that are live-out of the current instruction
    pinned_out: PinnedByteSet,

    /// PinnedByteSet for bytes that are live-in or live-out
    pinned_in_out: PinnedByteSet,
}

impl LocalRegAlloc<'_> {
    fn new<'a>(
        model: &'a dyn Model,
        arena: &'a Arena,
        affinities: &'a AffinityMap,
    ) -> LocalRegAlloc<'a> {
        let mut byte_idx = Vec::new();
        byte_idx.resize(usize::from(arena.limit()), u32::MAX);
        LocalRegAlloc {
            model,
            arena,
            affinities,
            phi_web_bytes: Default::default(),
            used: Default::default(),
            idx_bytes: Default::default(),
            byte_idx,
            search_start: 0.into(),
            pinned_in: Default::default(),
            pinned_out: Default::default(),
            pinned_in_out: Default::default(),
        }
    }

    /// Updates idx_bytes without actually assigning it
    fn set_idx_bytes(&mut self, idx: u32, bytes: Range<u16>) {
        let idx = usize::try_from(idx).unwrap();
        if self.idx_bytes.len() <= idx {
            self.idx_bytes.resize_with(idx + 1, || 0..0);
        }
        self.idx_bytes[idx] = bytes;
    }

    fn assign_idx_bytes(&mut self, idx: u32, bytes: Range<u16>) {
        let bytes_usize = bytes.start.into()..bytes.end.into();
        debug_assert!(self.used.all_unset_in_range(bytes_usize.clone()));

        if let Some(repr) = self.affinities.phi_webs[idx] {
            debug_assert!(bytes.len() == usize::from(repr.bytes()));
            self.phi_web_bytes.insert(repr, bytes.clone());
        }

        self.used.set_range(bytes_usize);
        for b in bytes.clone() {
            self.byte_idx[usize::from(b)] = idx;
        }

        self.set_idx_bytes(idx, bytes);
    }

    fn free_bytes(&mut self, bytes: Range<u16>) {
        let bytes_usize = bytes.start.into()..bytes.end.into();
        debug_assert!(self.used.all_set_in_range(bytes_usize.clone()));

        self.used.unset_range(bytes_usize);
    }

    /// Updates idx_bytes without actually assigning it
    fn set_ssa_bytes(&mut self, ssa: &SSAValue, bytes: Range<u16>) {
        assert!(
            bytes.len() == usize::from(ssa.bytes()),
            "The size of the byte range must match the SSA value",
        );
        assert!(
            bytes.start % u16::from(ssa.bytes()) == 0,
            "SSA values must always be aligned to their size",
        );
        self.set_idx_bytes(ssa.idx(), bytes);
    }

    fn assign_ssa_bytes(&mut self, ssa: &SSAValue, bytes: Range<u16>) {
        assert!(
            bytes.len() == usize::from(ssa.bytes()),
            "The size of the byte range must match the SSA value",
        );
        assert!(
            bytes.start % u16::from(ssa.bytes()) == 0,
            "SSA values must always be aligned to their size",
        );
        self.assign_idx_bytes(ssa.idx(), bytes);
    }

    fn idx_bytes(&self, idx: u32) -> Range<u16> {
        let bytes = &self.idx_bytes[usize::try_from(idx).unwrap()];
        debug_assert!(!bytes.is_empty());
        bytes.clone()
    }

    fn ssa_bytes(&self, ssa: &SSAValue) -> Range<u16> {
        self.idx_bytes(ssa.idx())
    }

    fn ssa_bytes_offset(
        &self,
        ssa: &SSAValue,
        offset: i16,
    ) -> Option<Range<u16>> {
        let ssa_bytes = self.ssa_bytes(ssa);
        let start = ssa_bytes.start.checked_add_signed(offset)?;
        let end = ssa_bytes.end.checked_add_signed(offset)?;
        if end > self.arena.limit() {
            return None;
        }
        Some(start..end)
    }

    fn byte_idx(&self, byte: u16) -> Option<u32> {
        if self.used.contains(byte.into()) {
            Some(self.byte_idx[usize::from(byte)])
        } else {
            None
        }
    }

    fn ssa_ref_bytes(&self, vec: &SSARef) -> Option<Range<u16>> {
        let mut vec_bytes = self.ssa_bytes(&vec[0]);
        for i in 1..vec.len() {
            let ssa_bytes = self.ssa_bytes(&vec[i]);
            if ssa_bytes.start == vec_bytes.end {
                vec_bytes.end = ssa_bytes.end;
            } else {
                return None;
            }
        }
        Some(vec_bytes)
    }

    fn assign_ssa_ref_bytes(&mut self, vec: &SSARef, bytes: Range<u16>) {
        debug_assert_eq!(bytes.len(), usize::from(vec.bytes()));
        for (ssa, bytes) in vec.iter_zip_bytes(bytes) {
            self.assign_ssa_bytes(ssa, bytes);
        }
    }

    fn assign_ssa_ref_reg(&mut self, vec: &SSARef, reg: &RegRef) {
        self.assign_ssa_ref_bytes(vec, self.arena.reg_to_bytes(reg));
    }

    fn idx_phi_bytes(&self, idx: u32) -> Option<Range<u16>> {
        let repr = self.affinities.phi_webs[idx]?;
        self.phi_web_bytes.get(&repr).cloned()
    }

    fn ssa_ref_phi_bytes(&self, vec: &SSARef) -> Option<Range<u16>> {
        let mut vec_bytes = self.idx_phi_bytes(vec[0].idx())?;
        for i in 1..vec.len() {
            let ssa_bytes = self.idx_phi_bytes(vec[i].idx())?;
            if ssa_bytes.start == vec_bytes.end {
                vec_bytes.end = ssa_bytes.end;
            } else {
                return None;
            }
        }
        Some(vec_bytes)
    }

    fn bytes_are_unused(&self, bytes: Range<u16>) -> bool {
        let bytes = bytes.start.into()..bytes.end.into();
        self.used.all_unset_in_range(bytes)
    }

    fn find_aligned_unused_unpinned_range(
        &self,
        pinned: &PinnedByteSet,
        start: usize,
        count: usize,
        align_mul: usize,
        align_offset: usize,
    ) -> usize {
        let mut unused = self.used.find_aligned_unset_range(
            start,
            count,
            align_mul,
            align_offset,
        );
        loop {
            let unpinned = pinned.0.find_aligned_unset_range(
                unused,
                count,
                align_mul,
                align_offset,
            );
            if unpinned == unused {
                break;
            }

            unused = self.used.find_aligned_unset_range(
                unpinned,
                count,
                align_mul,
                align_offset,
            );
            if unpinned == unused {
                break;
            }
        }
        unused
    }

    fn is_aligned_unpinned_range(
        &self,
        pinned: &PinnedByteSet,
        bytes: Range<u16>,
        align: RegAlignConstraint,
    ) -> bool {
        align.satisfied(bytes.start.into())
            && self.arena.is_contiguous(bytes.clone())
            && pinned.bytes_are_unpinned(bytes)
    }

    fn find_unpinned_bytes(
        &self,
        pinned: &PinnedByteSet,
        bytes: u8,
        align: RegAlignConstraint,
        cost_fn: impl Fn(u16) -> u8,
    ) -> Option<u16> {
        let mut best = (u16::MAX, u8::MAX);

        // First, loop through unused registers in the hopes that one of them
        // ends up having cost 0
        let (align_mul, align_offset) = align.max_align();
        let mut cur = WrapOnceCounter::new(
            usize::from(self.search_start.get()),
            usize::from(self.arena.limit()),
        );
        while let Some(start) = cur.get() {
            let b = self.find_aligned_unused_unpinned_range(
                pinned,
                start,
                usize::from(bytes),
                usize::from(align_mul),
                usize::from(align_offset),
            );
            cur.advance(b + usize::from(align_mul));

            if b + usize::from(bytes) > usize::from(self.arena.limit()) {
                continue;
            }

            if !align.satisfied(b) {
                continue;
            }

            let b = u16::try_from(b).unwrap();
            if !self.arena.is_contiguous(b..(b + u16::from(bytes))) {
                continue;
            }

            let c = cost_fn(b);
            if c == 0 {
                return Some(b);
            } else if c < best.1 {
                best = (b, c);
            }
        }

        // This is the bad case.  Loop through all unpinned bytes, ignoring
        // used bytes and check them all.
        let mut cur = WrapOnceCounter::new(
            usize::from(self.search_start.get()),
            usize::from(self.arena.limit()),
        );
        while let Some(start) = cur.get() {
            let b = pinned.0.find_aligned_unset_range(
                start,
                usize::from(bytes),
                usize::from(align_mul),
                usize::from(align_offset),
            );
            cur.advance(b + usize::from(align_mul));

            if b + usize::from(bytes) > usize::from(self.arena.limit()) {
                continue;
            }

            if !align.satisfied(b) {
                continue;
            }

            let b = u16::try_from(b).unwrap();
            if !self.arena.is_contiguous(b..(b + u16::from(bytes))) {
                continue;
            }

            let c = cost_fn(b);
            if c < best.1 {
                best = (b, c);
            }
        }

        if best.0 == u16::MAX {
            None
        } else {
            Some(best.0)
        }
    }

    fn choose_bytes(
        &self,
        p: &PinnedByteSet,
        bytes: u8,
        align: RegAlignConstraint,
        cost_fn: impl Fn(u16) -> u8,
    ) -> Range<u16> {
        let b = self
            .find_unpinned_bytes(p, bytes, align, cost_fn)
            .expect("Out of registers!");
        if self.arena.round_robin {
            self.search_start.set(b + u16::from(bytes));
        }
        b..(b + u16::from(bytes))
    }

    fn choose_aligned_bytes(&self, p: &PinnedByteSet, bytes: u8) -> Range<u16> {
        let align = RegAlignConstraint::for_align(bytes, 0);
        self.choose_bytes(p, bytes, align, |_| 0)
    }

    fn choose_ssa_ref_bytes(
        &self,
        p: &PinnedByteSet,
        vec: &SSARef,
        align: RegAlignConstraint,
        cost_fn: impl Fn(u16) -> u8,
    ) -> Range<u16> {
        // First check to see if we have a specific reg affinity and if those
        // bytes happen to be free.
        if let Some(bytes) = self.affinities.reg_bytes(vec) {
            if self.is_aligned_unpinned_range(p, bytes.clone(), align)
                && self.bytes_are_unused(bytes.clone())
            {
                return bytes;
            }
        }

        if vec.comps() == 1 {
            let ssa = vec[0];
            let a = &self.affinities.get_ssa(ssa);

            // Include the alignment cost in the cost function
            let cost_fn = |b: u16| cost_fn(b).saturating_add(a.align_cost(b));

            if a.vec_size > 0 {
                // We have a vector use. Try to find a "nice" allocation
                debug_assert!(a.vec_size >= ssa.bytes());
                if let Some(vec_repr) = &a.vec_repr {
                    // Try to choose from the vector based on our offset from
                    // the representative.
                    if let Some(bytes) =
                        self.ssa_bytes_offset(vec_repr, a.vec_offset.into())
                    {
                        if self.is_aligned_unpinned_range(
                            p,
                            bytes.clone(),
                            align,
                        ) && self.bytes_are_unused(bytes.clone())
                        {
                            return bytes;
                        }
                    }

                    // Fall back to the default allocation
                    self.choose_bytes(p, ssa.bytes(), align, cost_fn)
                } else {
                    // If we're the vector representative, we want to find
                    // enough space for the whole vector.
                    debug_assert!(a.vec_offset >= 0);
                    let vec_offset = u16::try_from(a.vec_offset).unwrap();
                    let vec_cost_fn = |b: u16| {
                        let Some(start) = b.checked_sub(vec_offset) else {
                            return a.vec_size;
                        };

                        let end = start + u16::from(a.vec_size);
                        if end > self.arena.limit() {
                            return a.vec_size;
                        }

                        self.used
                            .count_set_in_range(start.into()..end.into())
                            .try_into()
                            .unwrap_or(u8::MAX)
                    };

                    let bytes = self.choose_bytes(p, ssa.bytes(), align, |b| {
                        cost_fn(b).saturating_add(vec_cost_fn(b))
                    });

                    // Increment the round-robin past the whole vector
                    if self.arena.round_robin {
                        self.search_start.set(
                            bytes.start + u16::from(a.vec_size) - vec_offset,
                        );
                    }

                    bytes
                }
            } else {
                // No vector use
                self.choose_bytes(p, ssa.bytes(), align, cost_fn)
            }
        } else {
            // For vectors, we assume the default is the best we can do. Either
            // the vector use is just going to use this vector (the likely case)
            // or it'll get split up.  The chances of two vectors being combined
            // into one where everything still aligns is pretty low.
            self.choose_bytes(p, vec.bytes(), align, |b| {
                cost_fn(b).saturating_add(self.affinities.align_cost(vec, b))
            })
        }
    }

    fn choose_fixed_bytes(
        &self,
        reg: RegRef,
        bytes: u8,
        align: RegAlignConstraint,
    ) -> Range<u16> {
        assert!(reg.bytes() >= bytes);
        let b = self.arena.reg_to_bytes(&reg).start;
        assert!(align.satisfied(b.into()));
        b..(b + u16::from(bytes))
    }

    fn choose_src_bytes(
        &self,
        vec: &SSARef,
        is_killed: bool,
        align: RegAlignConstraint,
        src_bytes: &BitSet<usize>,
    ) -> Range<u16> {
        let p = if is_killed {
            &self.pinned_in
        } else {
            &self.pinned_in_out
        };

        // Common case: Try to re-choose the old value
        if let Some(vec_bytes) = self.ssa_ref_bytes(vec) {
            if self.is_aligned_unpinned_range(p, vec_bytes.clone(), align) {
                return vec_bytes;
            }
        }

        let bytes = vec.bytes();
        self.choose_ssa_ref_bytes(p, vec, align, |b| {
            let bytes = b..(b + u16::from(bytes));
            src_bytes
                .count_set_in_range(bytes.start.into()..bytes.end.into())
                .try_into()
                .unwrap_or(u8::MAX)
        })
    }

    fn choose_dst_bytes(
        &self,
        vec: &SSARef,
        bytes: u8,
        align: RegAlignConstraint,
    ) -> Range<u16> {
        let ssa_bytes = vec.bytes();
        let p = &self.pinned_out;

        if let Some(phi_bytes) = self.ssa_ref_phi_bytes(vec) {
            if self.is_aligned_unpinned_range(p, phi_bytes.clone(), align)
                && self.bytes_are_unused(phi_bytes.clone())
            {
                debug_assert_eq!(phi_bytes.len(), usize::from(ssa_bytes));
                return phi_bytes;
            }
        }

        if vec.comps() == 1 {
            debug_assert!(ssa_bytes <= bytes && bytes <= 4);
            self.choose_ssa_ref_bytes(p, vec, align, |b| {
                let bytes = aligned_u16_range(b, bytes.into());
                debug_assert!(b + u16::from(ssa_bytes) <= bytes.end);
                self.used
                    .count_set_in_range(bytes.start.into()..bytes.end.into())
                    .try_into()
                    .unwrap_or(u8::MAX)
            })
        } else {
            debug_assert_eq!(ssa_bytes, bytes);
            self.choose_ssa_ref_bytes(p, vec, align, |_| 0)
        }
    }

    fn alloc_regs_instr(
        &mut self,
        ip: usize,
        instr: &mut Instr,
        pcopy: &mut ParallelCopy,
        bl: &BlockLiveness,
    ) {
        // We use a bitmask for indices
        assert!(instr.srcs().len() <= 8);
        assert!(instr.dsts().len() <= 8);

        struct SrcDst {
            is_src: bool,
            is_killed: bool,
            mask: u8,
            bytes: u8,
            align: RegAlignConstraint,
            fixed_reg: Option<RegRef>,
            /// Byte range assigned to the SSARef
            ssa_bytes: Range<u16>,
            /// Byte range allocated
            ra_bytes: Range<u16>,
            vec: SSARef,
        }

        struct Evicted {
            is_src: bool,
            bytes: Range<u16>,
            idx: u32,
        }

        let mut src_bytes = BitSet::new();
        let mut evicted = VecDeque::new();
        let mut srcs_dsts: Vec<SrcDst> = Vec::new();
        for (i, src) in instr.srcs().iter().enumerate() {
            let SrcRef::SSA(vec) = &src.src_ref else {
                continue;
            };

            if !self.arena.contains_ref(vec) {
                continue;
            }

            let bytes = vec.bytes();
            let align =
                RegAlignConstraint::for_op_src(self.model, &instr.op, src);

            let fixed_reg = if self.arena.is_reg() {
                self.model.op_fixed_src_reg(&instr.op, src)
            } else {
                None
            };

            let mut first_seen = true;
            for src_dst in srcs_dsts.iter_mut() {
                if &src_dst.vec == vec {
                    first_seen = false;
                    debug_assert!(src_dst.is_src);
                    src_dst.mask |= 1 << i;
                    debug_assert_eq!(src_dst.bytes, bytes);
                    src_dst.align &= align;
                    if fixed_reg.is_some() {
                        assert!(src_dst.fixed_reg.is_none());
                        src_dst.fixed_reg = fixed_reg;
                    }
                    break;
                }
            }
            if first_seen {
                // This is the first time we've seen this SSA ref.  Evict it
                // and add it to the list.  The evict handling at the end will
                // ensure we copy it back into place.
                let mut is_killed = true;
                for ssa in vec {
                    if bl.is_live_after_ip(ssa, ip) {
                        is_killed = false;
                    }

                    let ssa_bytes = self.ssa_bytes(ssa);
                    src_bytes.set_range(
                        ssa_bytes.start.into()..ssa_bytes.end.into(),
                    );
                    evicted.push_back(Evicted {
                        is_src: true,
                        bytes: ssa_bytes.clone(),
                        idx: ssa.idx(),
                    });
                    self.free_bytes(ssa_bytes);
                }

                srcs_dsts.push(SrcDst {
                    is_src: true,
                    is_killed,
                    mask: 1 << i,
                    bytes,
                    align,
                    fixed_reg,
                    ssa_bytes: 0..0,
                    ra_bytes: 0..0,
                    vec: vec.clone(),
                });
            }
        }

        for (i, dst) in instr.dsts().iter().enumerate() {
            let DstRef::SSA(vec) = &dst.dst_ref else {
                continue;
            };

            if !self.arena.contains_ref(vec) {
                continue;
            }

            let (bytes, align) =
                RegAlignConstraint::for_op_dst(self.model, &instr.op, dst);

            let fixed_reg = if self.arena.is_reg() {
                self.model.op_fixed_dst_reg(&instr.op, dst)
            } else {
                None
            };

            srcs_dsts.push(SrcDst {
                is_src: false,
                is_killed: false,
                mask: 1 << i,
                bytes,
                align,
                fixed_reg,
                ssa_bytes: 0..0,
                ra_bytes: 0..0,
                vec: vec.clone(),
            });
        }

        // Evict everything in the clobber set.
        if self.arena.is_reg() {
            for reg in instr_clobbered_regs(self.model, &instr.op) {
                let bytes = self.arena.reg_to_bytes(&reg);
                for b in bytes.clone() {
                    if let Some(idx) = self.byte_idx(b) {
                        let idx_bytes = self.idx_bytes(idx);
                        evicted.push_back(Evicted {
                            is_src: false,
                            bytes: idx_bytes.clone(),
                            idx,
                        });
                        self.free_bytes(idx_bytes);
                    }
                }

                // We don't allow clobbers with destinations.  Mark them in
                // pinned_in_out to ensure that no sources we need preserved
                // get assigned to clobbered regs.
                debug_assert!(instr.dsts().is_empty());
                self.pinned_in_out.pin_bytes_no_check(bytes);
            }
        }

        // Sort by size in descending order.  sort_by_key() is guaranteed to be
        // stable so this also ensures that sources get processed first.
        // Constrained registers have higher priority (otherwise other sources
        // would steal their place)
        srcs_dsts.sort_by_key(|a| {
            std::cmp::Reverse((a.fixed_reg.is_some(), a.bytes))
        });

        for src_dst in &mut srcs_dsts {
            let ssa_bytes = if let Some(reg) = src_dst.fixed_reg {
                self.choose_fixed_bytes(reg, src_dst.bytes, src_dst.align)
            } else if src_dst.is_src {
                debug_assert_eq!(src_dst.bytes, src_dst.vec.bytes());
                self.choose_src_bytes(
                    &src_dst.vec,
                    src_dst.is_killed,
                    src_dst.align,
                    &src_bytes,
                )
            } else {
                self.choose_dst_bytes(
                    &src_dst.vec,
                    src_dst.bytes,
                    src_dst.align,
                )
            };
            debug_assert_eq!(ssa_bytes.len(), usize::from(src_dst.vec.bytes()));

            // Expand the byte range, if needed.  This can happen for ALU dsts
            // if we had to widen the ALU op.
            let bytes = if ssa_bytes.len() < usize::from(src_dst.bytes) {
                let aligned =
                    aligned_u16_range(ssa_bytes.start, src_dst.bytes.into());
                debug_assert!(ssa_bytes.end <= aligned.end);
                aligned
            } else {
                debug_assert_eq!(ssa_bytes.len(), usize::from(src_dst.bytes));
                ssa_bytes.clone()
            };

            for b in bytes.clone() {
                if let Some(idx) = self.byte_idx(b) {
                    let idx_bytes = self.idx_bytes(idx);
                    evicted.push_back(Evicted {
                        is_src: false,
                        bytes: idx_bytes.clone(),
                        idx,
                    });
                    self.free_bytes(idx_bytes);
                }
            }

            // Pin the range
            if src_dst.is_src {
                debug_assert_eq!(ssa_bytes, bytes);
                self.pinned_in.pin_bytes(bytes.clone());
                for (ssa, bytes) in src_dst.vec.iter_zip_bytes(bytes.clone()) {
                    if bl.is_live_after_ip(ssa, ip) {
                        self.pinned_out.pin_bytes(bytes.clone());
                    }
                }
            } else {
                self.pinned_out.pin_bytes(bytes.clone());
            }
            self.pinned_in_out.pin_bytes_no_check(bytes.clone());

            src_dst.ssa_bytes = ssa_bytes;
            src_dst.ra_bytes = bytes;
        }

        for src_dst in &srcs_dsts {
            let ssa_bytes = src_dst.ssa_bytes.clone();
            let ra_bytes = src_dst.ra_bytes.clone();

            // Assign bytes to SSA values but only if the value is live after
            // this instruction.  Only actually assign if the value is still
            // live after this instruction.
            for (ssa, bytes) in src_dst.vec.iter_zip_bytes(ssa_bytes.clone()) {
                if bl.is_live_after_ip(ssa, ip) {
                    self.assign_ssa_bytes(ssa, bytes.clone());
                } else {
                    // If it's not live, update idx_bytes without assigning.
                    // This way it represents the last place `ssa` ever lived.
                    // We need this for the eviction code below.
                    self.set_ssa_bytes(ssa, bytes.clone());
                }
            }

            if src_dst.is_src {
                let mut mask = src_dst.mask;
                while mask != 0 {
                    let i = mask.trailing_zeros();
                    mask &= !(1 << i);

                    let i = usize::try_from(i).unwrap();
                    let src = &instr.srcs()[i];

                    let ra_src = self.arena.src_for_bytes(ra_bytes.clone());
                    if let SrcRef::Reg(mut reg) = ra_src.src_ref {
                        let mut swz = ra_src.swizzle;
                        if self.model.op_src_is_64bit(&instr.op, src) {
                            let word = reg.idx & 1;
                            reg.idx &= !1;
                            if src.swizzle.is_byte_swizzle() {
                                assert!(word == 0);
                                assert!(reg.range.bytes() <= 4);
                            } else if reg.range == RegRange::Regs(1) {
                                swz = Swizzle::replicate_word(word);
                                if word == 1 {
                                    reg.range = RegRange::Regs(2);
                                }
                            } else {
                                assert!(word == 0);
                            }
                        }

                        let src = &mut instr.srcs_mut()[i];
                        src.src_ref = reg.into();
                        src.swizzle = swz
                            .swizzle(src.swizzle)
                            .expect("8 and 16-bit sources have to swizzle");

                        // If all the SSA values read by this source are killed,
                        // mark this as the last use.  Only do this if the SSA
                        // values in question consume whole registers because
                        // the hardware's last-use concept is per-register, not
                        // per-byte.
                        let whole_regs = src_dst.vec[0].bits() == 32;
                        src.last_use = src_dst.is_killed && whole_regs;
                    } else {
                        assert_eq!(src.swizzle, ra_src.swizzle);
                        instr.srcs_mut()[i].src_ref = ra_src.src_ref;
                    }
                }
            } else {
                debug_assert!(src_dst.mask.is_power_of_two());
                let i = usize::try_from(src_dst.mask.trailing_zeros()).unwrap();

                if self.model.op_dst_is_staging_reg(&instr.op) {
                    // Staging registers respect lanes.  We just widened the
                    // destination to avoid data races.
                    let dst = &mut instr.dsts_mut()[i];
                    *dst = self.arena.dst_for_bytes(ssa_bytes);
                } else {
                    // For ALU ops, we need to widen the destination as needed
                    let dst = &mut instr.dsts_mut()[i];
                    let ra_dst = self.arena.dst_for_bytes(ra_bytes);
                    dst.dst_ref = ra_dst.dst_ref;
                    dst.lanes = fold_lanes(ra_dst.lanes, dst.lanes);
                }
            }
        }

        while let Some(e) = evicted.pop_front() {
            let dst_bytes = if e.is_src {
                // If it's a source, it's already been placed.  Just look it up.
                let idx_bytes = self.idx_bytes(e.idx);
                debug_assert_eq!(idx_bytes.len(), e.bytes.len());
                idx_bytes
            } else {
                let nr_bytes = e.bytes.len().try_into().unwrap();
                let bytes =
                    self.choose_aligned_bytes(&self.pinned_in_out, nr_bytes);

                // Evict anything that might happen to be in dst_bytes
                for b in bytes.clone() {
                    if let Some(idx) = self.byte_idx(b) {
                        let idx_bytes = self.idx_bytes(idx);
                        evicted.push_back(Evicted {
                            is_src: false,
                            bytes: idx_bytes.clone(),
                            idx,
                        });
                        self.free_bytes(idx_bytes);
                    }
                }

                // Pin dst_bytes so we don't try to re-use it
                self.pinned_in_out.pin_bytes(bytes.clone());

                // Assign the evicted idx to the new location
                self.assign_idx_bytes(e.idx, bytes.clone());

                bytes
            };

            pcopy.add_copy(
                self.arena.dst_for_bytes(dst_bytes),
                self.arena.src_for_bytes(e.bytes),
            );
        }

        // Clean up by unpinning everything
        self.pinned_in.clear();
        self.pinned_out.clear();
        self.pinned_in_out.clear();
    }

    fn try_coalesce_mkvec(
        &mut self,
        ip: usize,
        bl: &BlockLiveness,
        instr: &Instr,
    ) -> bool {
        debug_assert!(instr.dsts().len() == 1);
        let dst = &instr.dsts()[0];

        let DstRef::SSA(dst_vec) = &dst.dst_ref else {
            return false;
        };
        assert!(dst_vec.comps() == 1);
        let dst_ssa = &dst_vec[0];

        if !self.arena.contains_ssa(dst_ssa) {
            return false;
        }

        let srcs = instr.srcs();
        let get_src_bytes = |src: &Src| {
            debug_assert!(src.src_mod.is_none());

            let vec = src.as_ssa()?;
            debug_assert!(vec.comps() == 1);
            let ssa = &vec[0];

            if !self.arena.contains_ssa(ssa) {
                return None;
            }

            // We can only coalesce if all the sources are killed
            if bl.is_live_after_ip(ssa, ip) {
                return None;
            }

            Some(self.ssa_bytes(ssa))
        };

        let Some(mut vec_bytes) = get_src_bytes(&srcs[0]) else {
            return false;
        };
        for i in 1..srcs.len() {
            let Some(src_bytes) = get_src_bytes(&srcs[i]) else {
                return false;
            };
            if src_bytes.start == vec_bytes.end {
                vec_bytes.end = src_bytes.end;
            } else {
                return false;
            }
        }

        let (dst_bytes, align) =
            RegAlignConstraint::for_op_dst(self.model, &instr.op, dst);
        debug_assert_eq!(dst_bytes, dst_vec.bytes());
        if !align.satisfied(vec_bytes.start.into()) {
            return false;
        }

        for src in srcs {
            let ssa = &src.src_ref.as_ssa().unwrap()[0];
            self.free_bytes(self.ssa_bytes(ssa));
        }
        self.assign_ssa_bytes(dst_ssa, vec_bytes.clone());

        // Free on the off chance the destination is immediately killed
        if !bl.is_live_after_ip(dst_ssa, ip) {
            self.free_bytes(vec_bytes.clone());
        }

        true
    }
}

/// A wrapper around AllocSSA that only allows registers
struct RegSSAAlloc<'a>(&'a mut SSAValueAllocator);

impl AllocSSA for RegSSAAlloc<'_> {
    fn alloc_ssa_value(&mut self, bits: u8, is_mem: bool) -> SSAValue {
        assert!(!is_mem);
        self.0.alloc_ssa_value(bits, false)
    }
}

struct GlobalRegAlloc<'a> {
    local: LocalRegAlloc<'a>,

    /// Map of blocks to live-out maps.  For live-out values, the index is the
    /// index of the SSA value.  For phis, the index is the index of the phi
    /// destination SSA value.
    live_out: Vec<Option<FxHashMap<u32, Range<u16>>>>,

    /// Pinned byte set for cross-block allocation
    pinned: PinnedByteSet,
}

impl GlobalRegAlloc<'_> {
    fn new<'a>(
        model: &'a dyn Model,
        arena: &'a Arena,
        affinities: &'a AffinityMap,
    ) -> GlobalRegAlloc<'a> {
        GlobalRegAlloc {
            local: LocalRegAlloc::new(model, arena, affinities),
            live_out: Default::default(),
            pinned: Default::default(),
        }
    }

    fn start_shader(&mut self, cfg: &CFG<BasicBlock>) {
        debug_assert!(self.local.used.is_empty());

        let bi = 0;
        assert!(cfg.pred_indices(bi).is_empty());

        if self.local.arena.is_reg() {
            let mut is_preamble = true;
            for instr in &cfg[bi].instrs {
                if let Op::RegIn(op) = &instr.op {
                    debug_assert!(is_preamble);
                    let dst_vec = op.dst.dst_ref.as_ssa().unwrap();
                    self.local.assign_ssa_ref_reg(dst_vec, &op.reg);
                } else if !matches!(&instr.op, Op::Nop(_)) {
                    if cfg!(debug_assertions) {
                        is_preamble = false;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    fn end_shader(
        &mut self,
        cfg: &CFG<BasicBlock>,
        bi: usize,
        reg_outs: &mut Vec<Box<OpRegOut>>,
        pcopy: &mut ParallelCopy,
    ) {
        debug_assert!(cfg.succ_indices(bi).is_empty());

        if self.local.arena.is_reg() {
            for op in std::mem::take(reg_outs) {
                if let RegRange::Regs(words) = op.reg.range {
                    for i in 0..words {
                        pcopy.add_copy(
                            op.reg.word(i).into(),
                            op.src.clone().word(i),
                        );
                    }
                } else {
                    pcopy.add_copy(op.reg.into(), op.src);
                }
            }
        }

        // After the block is done, nothing is used
        self.local.used.clear();
    }

    fn start_block(
        &mut self,
        cfg: &CFG<BasicBlock>,
        live: &Liveness,
        ssa_alloc: &SSAValueAllocator,
        bi: usize,
    ) {
        debug_assert!(self.local.used.is_empty());

        let preds = cfg.pred_indices(bi);
        debug_assert!(!preds.is_empty());

        // For any block, exactly one predecessor will have a populated
        // live-out set.  That set is canonical for all edges into this
        // block.  See start_block() for more details.
        let mut pred_live_out = None;
        for &pi in preds {
            if let Some(live_out) = &self.live_out[pi] {
                let old = pred_live_out.replace(live_out);
                assert!(old.is_none());
            }
        }
        let pred_live_out = pred_live_out.unwrap();

        let mut live_in = live.block(bi).live_in_set().clone();
        live_in.retain(|idx| {
            let ssa = ssa_alloc.lookup_by_idx(idx);
            self.local.arena.contains_ssa(&ssa)
        });

        for instr in &cfg[bi].instrs {
            if let Op::PhiDst(op) = &instr.op {
                let dst_vec = op.dst.dst_ref.as_ssa().unwrap();
                if self.local.arena.contains_ref(dst_vec) {
                    for ssa in dst_vec {
                        live_in.insert(ssa.idx());
                    }
                }
            } else if !matches!(&instr.op, Op::Nop(_)) {
                break;
            }
        }

        for idx in live_in.iter() {
            let bytes = pred_live_out.get(&idx).unwrap();
            self.local.assign_idx_bytes(idx, bytes.clone());
        }
    }

    fn choose_live_out_bytes(
        &self,
        bytes: u8,
        prefer: Option<Range<u16>>,
        src_bytes: &BitSet<usize>,
    ) -> Range<u16> {
        let p = &self.pinned;
        if let Some(prefer) = prefer {
            if p.bytes_are_unpinned(prefer.clone()) {
                return prefer;
            }
        }

        let align = RegAlignConstraint::for_align(bytes, 0);
        let b = self.local.find_unpinned_bytes(p, bytes, align, |b| {
            let bytes = b..(b + u16::from(bytes));
            let bytes = bytes.start.into()..bytes.end.into();
            debug_assert!(bytes.len() <= 8);
            src_bytes.count_set_in_range(bytes) as u8
        });
        let b = b.expect("Failed to allocate live-out");

        b..(b + u16::from(bytes))
    }

    fn end_block(
        &mut self,
        cfg: &CFG<BasicBlock>,
        live: &Liveness,
        ssa_alloc: &SSAValueAllocator,
        bi: usize,
        phi_srcs: &mut Vec<Box<OpPhiSrc>>,
        mut branch: Option<&mut Box<OpBranch>>,
        phi_map: &PhiMap,
        pcopy: &mut ParallelCopy,
    ) {
        debug_assert!(self.pinned.is_empty());

        let succ = cfg.succ_indices(bi);
        assert!(!succ.is_empty());

        let mut live_out = None;
        if succ.len() == 1 {
            // If we only have one successor, we can't have a branch condition
            if let Some(ref branch) = branch {
                assert!(branch.cond.is_zero());
            }

            for &pi in cfg.pred_indices(succ[0]) {
                if pi == bi {
                    continue;
                }

                // We have no critical edges
                assert_eq!(cfg.succ_indices(pi).len(), 1);

                // The live-out of a block is the union of the live-ins of its
                // successors.  If a block only has one successor, then its
                // live-in must equal that block's live-out.  As a corrolary,
                // if a block has one successor and we have no critical edges,
                // then all the predecessors must have the same live-out.
                debug_assert!(
                    live.block(bi).live_out_set()
                        == live.block(pi).live_out_set()
                );

                // If our peer has a live-out, use that.  We just asserted that
                // we have the same live-out set as our peer and, since our
                // live-out maps are relative to phi destinations, we can use
                // another block's live-out set just fine.
                //
                // This ensures that each block only has one predecessor with
                // a live-out map, allowing us to make that one map canonical
                // for all edges into that block.
                if let Some(p_live_out) = &self.live_out[pi] {
                    live_out = Some(p_live_out);
                }
            }
        } else {
            // We don't allow critical edges.  Since we are the sole predecessor
            // to all our successors, there is no peer with a live-in set.
            for &si in succ {
                assert_eq!(cfg.pred_indices(si).len(), 1);
            }

            // Since all our successors have a single predecessor, we can't be
            // involved in any phis.
            assert!(phi_srcs.is_empty());
        }

        let bl = live.block(bi);

        // Grab the live-out set and filter it down to just the values we care
        // about.  We'll use this a few places below.
        let mut live_out_set = bl.live_out_set().clone();
        live_out_set.retain(|idx| {
            let ssa = ssa_alloc.lookup_by_idx(idx);
            self.local.arena.contains_ssa(&ssa)
        });

        if let Some(live_out) = live_out {
            // In this case, someone already set up our live-out.  We just have
            // to emit copies to shuffle everything into place.
            for idx in live_out_set.iter() {
                let src_bytes = self.local.idx_bytes(idx);
                let dst_bytes = live_out.get(&idx).unwrap().clone();
                pcopy.add_copy(
                    self.local.arena.dst_for_bytes(dst_bytes),
                    self.local.arena.src_for_bytes(src_bytes),
                );
            }

            phi_srcs.retain(|op| {
                let dst_vec = phi_map.get_dst_ssa(&op.phi);
                if !self.local.arena.contains_ref(dst_vec) {
                    return true;
                }

                if let SrcRef::SSA(src_vec) = &op.src.src_ref {
                    debug_assert_eq!(dst_vec.len(), src_vec.len());
                    for (dst_ssa, src_ssa) in dst_vec.iter().zip(src_vec.iter())
                    {
                        let dst_bytes = live_out.get(&dst_ssa.idx()).unwrap();
                        let src_bytes = swizzle_byte_range(
                            self.local.idx_bytes(src_ssa.idx()),
                            op.src.swizzle,
                        );
                        pcopy.add_copy(
                            self.local.arena.dst_for_bytes(dst_bytes.clone()),
                            self.local.arena.src_for_bytes(src_bytes),
                        );
                    }
                } else {
                    for w in 0..dst_vec.comps() {
                        let idx = dst_vec[usize::from(w)].idx();
                        let dst_bytes = live_out.get(&idx).unwrap().clone();
                        let dst = self.local.arena.dst_for_bytes(dst_bytes);
                        pcopy.add_copy(dst, op.src.clone().word(w));
                    }
                }

                false
            });

            // After the block is done, nothing is used
            self.local.used.clear();

            return;
        }

        // If se got here, we're building the live-out.
        //
        // Start by accumulating the source bytes
        let mut all_src_bytes = BitSet::new();
        for idx in live_out_set.iter() {
            let bytes = self.local.idx_bytes(idx);
            let bytes = bytes.start.into()..bytes.end.into();
            all_src_bytes.set_range(bytes);
        }
        for op in phi_srcs.iter() {
            if let SrcRef::SSA(src_vec) = &op.src.src_ref {
                for src_ssa in src_vec {
                    if self.local.arena.contains_ssa(src_ssa) {
                        let bytes = self.local.idx_bytes(src_ssa.idx());
                        let bytes = bytes.start.into()..bytes.end.into();
                        all_src_bytes.set_range(bytes);
                    }
                }
            }
        }

        // Now, place everything.  Go largest to smallest to reduce so that
        // we can guarantee everything fits.
        let mut live_out: FxHashMap<u32, Range<u16>> = Default::default();
        for chunk_bytes in [8, 4, 2, 1] {
            // First place any chunk_bytes sized live-out values
            live_out_set.retain(|idx| {
                let idx_bytes = self.local.idx_bytes(idx);
                if idx_bytes.len() != usize::from(chunk_bytes) {
                    return true;
                }

                let dst_bytes = self.choose_live_out_bytes(
                    chunk_bytes,
                    Some(idx_bytes.clone()),
                    &all_src_bytes,
                );

                self.pinned.pin_bytes(dst_bytes.clone());
                pcopy.add_copy(
                    self.local.arena.dst_for_bytes(dst_bytes.clone()),
                    self.local.arena.src_for_bytes(idx_bytes.clone()),
                );
                let old = live_out.insert(idx, dst_bytes.clone());
                assert!(old.is_none());

                false
            });

            // Now place the chunk_bytes sized branch condition, if any
            for branch in branch.iter_mut() {
                let SrcRef::SSA(vec) = &branch.cond.src_ref else {
                    continue;
                };

                assert!(phi_srcs.is_empty());

                assert_eq!(vec.comps(), 1);
                let ssa = &vec[0];
                if !self.local.arena.contains_ssa(ssa) {
                    continue;
                }

                if ssa.bytes() != chunk_bytes {
                    continue;
                }
                let idx = ssa.idx();

                // If our branch condition is live-out, use the post-shuffle
                // value.
                if bl.is_live_out(ssa) {
                    let bytes = live_out.get(&idx).unwrap().clone();
                    branch.cond = self.local.arena.src_for_bytes(bytes);
                    continue;
                }

                // Otherwise, we need to RA the branch condition
                let idx_bytes = self.local.idx_bytes(idx);
                let dst_bytes = self.choose_live_out_bytes(
                    chunk_bytes,
                    Some(idx_bytes.clone()),
                    &all_src_bytes,
                );

                self.pinned.pin_bytes(dst_bytes.clone());
                pcopy.add_copy(
                    self.local.arena.dst_for_bytes(dst_bytes.clone()),
                    self.local.arena.src_for_bytes(idx_bytes.clone()),
                );

                branch.cond = self.local.arena.src_for_bytes(dst_bytes);
            }

            // Now place any chunk_bytes sized phis
            phi_srcs.retain(|op| {
                if op.phi.bytes() < chunk_bytes {
                    return true;
                }

                let dst_vec = phi_map.get_dst_ssa(&op.phi);
                if !self.local.arena.contains_ref(dst_vec) {
                    return true;
                }

                let src_vec = op.src.src_ref.as_ssa();
                let src_bytes = src_vec
                    .and_then(|vec| self.local.ssa_ref_bytes(vec))
                    .map(|bytes| swizzle_byte_range(bytes, op.src.swizzle));
                let dst_bytes = self.choose_live_out_bytes(
                    chunk_bytes,
                    src_bytes,
                    &all_src_bytes,
                );

                self.pinned.pin_bytes(dst_bytes.clone());

                for (i, (dst_ssa, dst_bytes)) in
                    dst_vec.iter_zip_bytes(dst_bytes).enumerate()
                {
                    if let Some(src_vec) = src_vec {
                        debug_assert_eq!(src_vec.len(), dst_vec.len());
                        let src_bytes = swizzle_byte_range(
                            self.local.idx_bytes(src_vec[i].idx()),
                            op.src.swizzle,
                        );
                        pcopy.add_copy(
                            self.local.arena.dst_for_bytes(dst_bytes.clone()),
                            self.local.arena.src_for_bytes(src_bytes),
                        );
                    } else {
                        pcopy.add_copy(
                            self.local.arena.dst_for_bytes(dst_bytes.clone()),
                            op.src.clone().word(i.try_into().unwrap()),
                        );
                    }
                    let old = live_out.insert(dst_ssa.idx(), dst_bytes);
                    assert!(old.is_none());
                }

                false
            });
        }

        debug_assert!(live_out_set.is_empty());
        if cfg!(debug_assertions) {
            for op in phi_srcs.iter() {
                let dst_vec = phi_map.get_dst_ssa(&op.phi);
                debug_assert!(!self.local.arena.contains_ref(dst_vec));
            }
        }

        // Clean up by unpinning everything
        self.pinned.clear();

        // After the block is done, nothing is used
        self.local.used.clear();

        let old = self.live_out[bi].replace(live_out);
        assert!(old.is_none());
    }

    fn pcopy_alloc<'a>(
        &self,
        ssa_alloc: &'a mut SSAValueAllocator,
    ) -> Option<RegSSAAlloc<'a>> {
        // We only want to provide an AllocSSA to ParallelCopy::into_instrs()
        // if we are copying memory values and we want to restrict it to only
        // being able to allocate registers, not memory.
        if self.local.arena.is_mem() {
            Some(RegSSAAlloc(ssa_alloc))
        } else {
            None
        }
    }

    fn alloc_regs_block(
        &mut self,
        cfg: &mut CFG<BasicBlock>,
        live: &Liveness,
        ssa_alloc: &mut SSAValueAllocator,
        bi: usize,
        phi_map: &PhiMap,
    ) {
        if bi == 0 {
            self.start_shader(cfg);
        } else {
            self.start_block(cfg, live, ssa_alloc, bi);
        }

        let bl = live.block(bi);
        let mut instrs = Vec::new();
        let mut phi_srcs = Vec::new();
        let mut reg_outs = Vec::new();
        let mut branch = None;
        for (ip, mut instr) in
            std::mem::take(&mut cfg[bi].instrs).into_iter().enumerate()
        {
            match instr.op {
                Op::Branch(op) => {
                    let old = branch.replace(op);
                    assert!(old.is_none());
                    continue;
                }
                Op::PhiDst(ref op) => {
                    debug_assert_ne!(bi, 0);
                    // These are handled by start_block if they are in the
                    // current arena.  If handled, start_block sets dst_ref
                    // to None and we can drop it.  If not, we need to leave
                    // it in place.
                    let dst_vec = op.dst.dst_ref.as_ssa().unwrap();
                    if !self.local.arena.contains_ref(dst_vec) {
                        instrs.push(instr);
                    }
                    continue;
                }
                Op::PhiSrc(op) => {
                    phi_srcs.push(op);
                    continue;
                }
                Op::MkVecV2I16(_) | Op::MkVecV2I8(_) | Op::MkVecV4I8(_) => {
                    if self.local.try_coalesce_mkvec(ip, bl, &instr) {
                        continue;
                    }
                }
                Op::RegIn(_) => {
                    // These were handled by start_shader if is_reg().  If not,
                    // then we're allocating memory and we need to leave them
                    // alone.
                    debug_assert_eq!(bi, 0);
                    if !self.local.arena.is_reg() {
                        instrs.push(instr);
                    }
                    continue;
                }
                Op::RegOut(op) => {
                    reg_outs.push(op);
                    continue;
                }
                _ => (),
            }

            let mut pcopy =
                ParallelCopy::new(self.local.model, self.local.arena.is_mem());
            self.local.alloc_regs_instr(ip, &mut instr, &mut pcopy, bl);
            let mut pcopy_alloc = self.pcopy_alloc(ssa_alloc);
            instrs.extend(pcopy.into_instrs(pcopy_alloc.as_mut()));
            instrs.push(instr);
        }

        if cfg.succ_indices(bi).is_empty() {
            assert!(phi_srcs.is_empty());
            assert!(branch.is_none());
            let mut pcopy =
                ParallelCopy::new(self.local.model, self.local.arena.is_mem());
            self.end_shader(cfg, bi, &mut reg_outs, &mut pcopy);
            let mut pcopy_alloc = self.pcopy_alloc(ssa_alloc);
            instrs.extend(pcopy.into_instrs(pcopy_alloc.as_mut()));
            instrs.extend(reg_outs.into_iter().map(Instr::from));
        } else {
            assert!(reg_outs.is_empty());
            let mut pcopy =
                ParallelCopy::new(self.local.model, self.local.arena.is_mem());
            self.end_block(
                cfg,
                live,
                ssa_alloc,
                bi,
                &mut phi_srcs,
                branch.as_mut(),
                phi_map,
                &mut pcopy,
            );
            let mut pcopy_alloc = self.pcopy_alloc(ssa_alloc);
            instrs.extend(pcopy.into_instrs(pcopy_alloc.as_mut()));
            instrs.extend(phi_srcs.into_iter().map(Instr::from));
            instrs.extend(branch.map(Instr::from));
        }

        cfg[bi].instrs = instrs;
    }
}

fn alloc_regs(s: &mut Shader, arena: &Arena, live: Liveness) {
    let phi_map = PhiMap::for_shader(s);
    let affinities = AffinityMap::for_shader(s, &arena, &live, &phi_map);

    let mut ra = GlobalRegAlloc::new(s.model, arena, &affinities);
    ra.live_out.resize_with(s.blocks.len(), Default::default);
    for bi in 0..s.blocks.len() {
        ra.alloc_regs_block(
            &mut s.blocks,
            &live,
            &mut s.ssa_alloc,
            bi,
            &phi_map,
        );
    }
}

#[derive(Default)]
struct SSARegMap {
    ssa_bytes: FxHashMap<SSAValue, Range<u16>>,
}

impl SSARegMap {
    pub fn new() -> SSARegMap {
        Default::default()
    }

    pub fn insert_ssa_ref(&mut self, vec: &SSARef, reg: RegRef) {
        for (ssa, bytes) in vec.iter_zip_bytes(reg.byte_range()) {
            self.ssa_bytes.insert(*ssa, bytes);
        }
    }

    pub fn get_ssa_ref_reg(&self, vec: &SSARef) -> Option<RegRef> {
        let mut vec_bytes = self.ssa_bytes.get(&vec[0])?.clone();
        for i in 1..vec.len() {
            let ssa_bytes = self.ssa_bytes.get(&vec[i])?;
            if ssa_bytes.start == vec_bytes.end {
                vec_bytes.end = ssa_bytes.end;
            } else {
                return None;
            }
        }
        RegRef::from_byte_range(vec_bytes).ok()
    }
}

impl Shader<'_> {
    /// If multiple phis in a block read the same source, RA will need to insert
    /// at least one copy regardless, but the messy resulting phi webs means RA
    /// will end up inserting many copies (or even swaps). Instead, we lower
    /// away repeated sources pre-RA by inserting that single copy preemptively,
    /// allowing RA's phi web heuristics to do their job.
    ///
    /// This is less heavyhanded than a full CSSA lowering.
    fn lower_repeated_phi_srcs(&mut self) {
        let mut seen = BitSet::new();
        for block in &mut self.blocks {
            let mut b = SSAInstrBuilder::new(self.model, &mut self.ssa_alloc);
            for op in block.iter_phi_srcs_mut() {
                let SrcRef::SSA(vec) = &mut op.src.src_ref else {
                    continue;
                };

                for ssa in vec {
                    if !seen.insert(*ssa) {
                        *ssa = b.copy_ssa(*ssa);
                    }
                }
            }
            block.insert_before_postlude(b.into_mapped());
            seen.clear();
        }
    }

    /// This pass propagates OpRegIn instructions with an OOB reg.  Since the
    /// reg is OOB, we know we will never allocate a register, and therefore
    /// never write to a register which conflicts with it.  So we can treat OOB
    /// registers as constants and safely copy-propagate the OpRegIn without
    /// worrying about their contents changing.  This is particularly useful
    /// for blend shaders where we want to keep everything inside the first 16
    /// registers but most of the preloads live in R48..64.
    fn prop_oob_reg_in(&mut self, arena: &Arena) {
        let model = self.model;
        let mut map = SSARegMap::new();
        self.map_instrs(|mut instr, _| {
            if let Op::RegIn(op) = &instr.op {
                let DstRef::SSA(vec) = &op.dst.dst_ref else {
                    panic!("We must have SSA destinations");
                };

                if arena.is_reg_oob(&op.reg) {
                    map.insert_ssa_ref(vec, op.reg);
                    [].into()
                } else {
                    [instr].into()
                }
            } else {
                for src_idx in 0..instr.srcs().len() {
                    let src = &instr.srcs()[src_idx];
                    let SrcRef::SSA(vec) = &src.src_ref else {
                        continue;
                    };

                    let Some(reg) = map.get_ssa_ref_reg(vec) else {
                        continue;
                    };

                    let swz = Swizzle::from(reg.range)
                        .swizzle(src.swizzle)
                        .expect("16-bit and smaller sources have to swizzle");
                    if !model.op_src_supports_swizzle(&instr.op, src, swz) {
                        continue;
                    }

                    let src = &mut instr.srcs_mut()[src_idx];
                    src.src_ref = reg.into();
                    src.swizzle = swz;
                }
                [instr].into()
            }
        });
    }

    pub fn assign_registers(&mut self) {
        pass!(self.lower_repeated_phi_srcs());

        if self.info.is_blend {
            let arena = Arena::new_blend(self.model);

            pass!(self.prop_oob_reg_in(&arena));

            let live = Liveness::for_shader(self);
            assert!(
                live.max_live_bytes().reg <= u32::from(arena.limit()),
                "Blend shaders cannot spill"
            );

            self.run_pass("allocating registers", |s| {
                alloc_regs(s, &arena, live);
            });
            self.info.registers_used = arena.regs_used();
            return;
        }

        let mut reg_limit: u16 = if DEBUG.contains(DebugFlags::SPILL) {
            16 * 4
        } else {
            u16::from(self.model.max_reg_count()) * 4
        };

        let mut live = Liveness::for_shader(self);
        let max_live = live.max_live_bytes();
        assert_eq!(max_live.mem, 0);
        if max_live.reg > u32::from(reg_limit) {
            // If we spill, we have to reduce registers by 2 to make room for
            // parallel copying memory
            reg_limit -= 8;

            self.spill_values(live, reg_limit.into());
            if DEBUG.contains(DebugFlags::PRINT) {
                eprintln!("Kraid shader after spill_values:\n{self}");
            }
            // We don't validate before SSA repair

            pass!(self.repair_ssa());
            pass!(self.opt_dce());

            live = Liveness::for_shader(self)
        }

        let max_live = live.max_live_bytes();
        assert!(
            max_live.reg <= u32::from(reg_limit),
            "Not enough registers: max_live = {} B, should be <= {reg_limit} B",
            max_live.reg
        );

        if max_live.mem > 0 {
            let mem_arena = Arena::new_mem(
                self.model,
                self.info.tls_size.try_into().unwrap(),
            );
            self.run_pass("allocating spills", |s| {
                alloc_regs(s, &mem_arena, live);
            });
            self.info.tls_size = mem_arena.bytes_used().into();

            live = Liveness::for_shader(self)
        }

        let live_reg_bytes = max_live.reg.try_into().unwrap();
        let reg_arena = Arena::new_reg(self.model, live_reg_bytes);
        self.run_pass("allocating registers", |s| {
            alloc_regs(s, &reg_arena, live);
        });
        self.info.registers_used = reg_arena.regs_used();
    }
}
