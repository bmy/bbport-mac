// Copyright © 2026 Collabora, Ltd.
// Copyright © 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

pub use crate::data_type::DataType;
use crate::data_type::{NumericType, PartialDataType};
use crate::debug::{DEBUG, DebugFlags};
pub use crate::flow::FlowCtrl;
pub use crate::model::Model;
pub use crate::ops::Op;
use crate::ops::{OpPhiDst, OpPhiSrc};
pub use crate::phi::Phi;
use crate::phi::PhiAllocator;
use crate::ssa_value::SSAValueAllocator;
pub use crate::ssa_value::{SSARef, SSAValue};
pub use crate::swizzle::Swizzle;
use crate::swizzle::*;
use compiler::as_slice::*;
use compiler::cfg::CFG;
use compiler::enum_as_u8::*;
use compiler::float16::F16;
use compiler::smallvec::*;
use kraid_proc_macros::EnumAsU8;
use mesa_util::bitview::BitViewable;

use std::fmt;
use std::num::{NonZeroU32, NonZeroU64};
use std::ops::{Deref, DerefMut, Range};

pub struct SmallConstant {
    pub idx: u8,
    pub imm32: u32,
    pub name: &'static str,
}

impl fmt::Display for SmallConstant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SpecialFAU {
    WarpId,
    FramebufferSize,
    ATestDatum,
    Sample,
    BlendDescriptor0,
    BlendDescriptor1,
    BlendDescriptor2,
    BlendDescriptor3,
    BlendDescriptor4,
    BlendDescriptor5,
    BlendDescriptor6,
    BlendDescriptor7,
    ThreadLocalPointer,
    WorkgroupLocalPointer,
    ResourceTablePointer,
    LaneId,
    CoreId,
    ShaderOutput,
    PrepassState,
    Pc,
}

impl SpecialFAU {
    pub fn blend_descriptor(n: u8) -> SpecialFAU {
        assert!(n < 8);
        // SAFETY:
        //
        // We use repr(u8) and Rust guarantees that implicit discriminants are
        // assigned by incrementing by one for each enum variant.
        unsafe { std::mem::transmute(SpecialFAU::BlendDescriptor0 as u8 + n) }
    }
}

impl fmt::Display for SpecialFAU {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use SpecialFAU::*;
        let name = match &self {
            WarpId => "warp_id",
            FramebufferSize => "framebuffer_size",
            ATestDatum => "atest_datum",
            Sample => "sample",
            BlendDescriptor0 => "blend_descriptor0",
            BlendDescriptor1 => "blend_descriptor1",
            BlendDescriptor2 => "blend_descriptor2",
            BlendDescriptor3 => "blend_descriptor3",
            BlendDescriptor4 => "blend_descriptor4",
            BlendDescriptor5 => "blend_descriptor5",
            BlendDescriptor6 => "blend_descriptor6",
            BlendDescriptor7 => "blend_descriptor7",
            ThreadLocalPointer => "thread_local_pointer",
            WorkgroupLocalPointer => "workgroup_local_pointer",
            ResourceTablePointer => "resource_table_pointer",
            LaneId => "lane_id",
            CoreId => "core_id",
            ShaderOutput => "shader_output",
            PrepassState => "prepass_state",
            Pc => "pc",
        };
        write!(f, "{name}")
    }
}

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum FAUPage {
    /// The user FAU table.  This assumes a single, flat table, unlike the
    /// hardware which splits the user FAU into 4 pages.  In Kraid, the page
    /// split for the user FAU is handled by the back-end as a per-generation
    /// constraint, not represented in the IR.
    User,

    /// FAU special page 0
    Special0,

    /// FAU special page 3
    Special1,

    /// FAU special page 3
    Special3,

    /// The small constant table
    SmallConst,
}

impl FAUPage {
    pub fn is_small_const(&self) -> bool {
        *self == FAUPage::SmallConst
    }

    pub fn is_special(&self) -> bool {
        matches!(
            self,
            FAUPage::Special0 | FAUPage::Special1 | FAUPage::Special3
        )
    }
}

#[derive(Clone, Copy)]
pub struct FAURef {
    pub page: FAUPage,

    /// The FAU index, in units of 32-bit words.  The hardware uses 64-bit
    /// words and has a word select bit in the source encoding.  In this
    /// representation, the bottom bit is the word select and the upper 15 bits
    /// are the 64-bit FAU index.  For 64-bit access, the bottom bit must be
    /// zero.
    pub idx: u16,

    /// If this FAU is a special FAU, this provides the semantic label which
    /// says what that FAU contains.  This is only for pretty printing and has
    /// no meaning beyond that.
    pub special: Option<SpecialFAU>,

    /// Load 64 bytes
    pub load64: bool,

    /// Optional metadata to pretty print small constants
    pub imm32: Option<u32>,
}

impl PartialEq for FAURef {
    fn eq(&self, other: &FAURef) -> bool {
        // special is intentionally missing
        self.page.eq(&other.page)
            && self.idx.eq(&other.idx)
            && self.load64.eq(&other.load64)
    }
}

impl FAURef {
    pub fn user_i32(idx: u16) -> Self {
        FAURef {
            page: FAUPage::User,
            idx,
            special: None,
            load64: false,
            imm32: None,
        }
    }

    pub fn user_i64(idx: u16) -> Self {
        assert!((idx % 2) == 0);
        FAURef {
            page: FAUPage::User,
            idx,
            special: None,
            load64: true,
            imm32: None,
        }
    }
}

impl fmt::Display for FAURef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.page == FAUPage::SmallConst {
            debug_assert!(!self.load64);
            return write!(f, "k{}", self.idx);
        }

        let idx = self.idx >> 1;
        let w = self.idx % 2;

        if let Some(special) = self.special {
            write!(f, "{special}")?;
        } else {
            match self.page {
                FAUPage::User => write!(f, "u{idx}")?,
                FAUPage::Special0 => write!(f, "s0:{idx}")?,
                FAUPage::Special1 => write!(f, "s1:{idx}")?,
                FAUPage::Special3 => write!(f, "s3:{idx}")?,
                FAUPage::SmallConst => panic!("Already handled"),
            }
        }

        if self.load64 {
            debug_assert_eq!(w, 0);
        } else {
            write!(f, ".w{w}")?;
        }
        Ok(())
    }
}

impl FAURef {
    pub fn word(mut self, word: u8) -> FAURef {
        assert!(word < 2);
        assert!(word == 0 || self.load64);
        self.idx += u16::from(word);
        self.load64 = false;
        self
    }
}

impl From<&SmallConstant> for FAURef {
    fn from(sc: &SmallConstant) -> FAURef {
        FAURef {
            page: FAUPage::SmallConst,
            idx: sc.idx.into(),
            special: None,
            load64: false,
            imm32: Some(sc.imm32),
        }
    }
}

#[repr(u8)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, EnumAsU8,
)]
pub enum PreloadReg {
    /* Compute */
    ///  0..16 -> local_id_0
    /// 16..32 -> local_id_1
    LocalId01,
    ///  0..16 -> local_id_2
    LocalId2,
    WorkgroupId0,
    WorkgroupId1,
    WorkgroupId2,
    GlobalId0,
    GlobalId1,
    GlobalId2,

    /* Vertex */
    InternalId,
    VertexId,
    InstanceId,
    DrawId,
    ViewId,

    /* Fragment */
    PrimitiveId,
    PrimitiveFlags,
    ///  0..16 -> position_x
    /// 16..32 -> position_y
    PositionXY,
    ///  0..16 -> cumulative_coverage
    /// 16..32 -> undefined
    CumulativeCoverage,
    ///  0..16 -> rasterizer_coverage
    /// 16..32 -> undefined
    RasterizerCoverage,
    /// 0 ..16 -> undefined
    /// 16..24 -> sample_id
    /// 24..32 -> centroid_id
    SampleCentroidId,
    FrameArg,

    /* Blend Shader ABI */
    /// Components of the first color (max 4x32 bits)
    BlendInputSrc0,
    /// Components of the second color (for double-source blending)
    BlendInputSrc1,
    /// Return address (where to jump when the blend shader finishes)
    BlendReturnAddr,
}

impl PreloadReg {
    pub fn reg_size(&self) -> u8 {
        match self {
            Self::FrameArg => 2,
            Self::BlendInputSrc0 | Self::BlendInputSrc1 => 4,
            _ => 1,
        }
    }
}

impl fmt::Display for PreloadReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use PreloadReg::*;
        let name = match &self {
            LocalId01 => "LOCAL_ID_01",
            LocalId2 => "LOCAL_ID_2",
            WorkgroupId0 => "WORKGROUP_ID_0",
            WorkgroupId1 => "WORKGROUP_ID_1",
            WorkgroupId2 => "WORKGROUP_ID_2",
            GlobalId0 => "GLOBAL_ID_0",
            GlobalId1 => "GLOBAL_ID_1",
            GlobalId2 => "GLOBAL_ID_2",
            InternalId => "INTERNAL_ID",
            VertexId => "VERTEX_ID",
            InstanceId => "INSTANCE_ID",
            DrawId => "DRAW_ID",
            ViewId => "VIEW_ID",
            PrimitiveId => "PRIMITIVE_ID",
            PrimitiveFlags => "PRIMITIVE_FLAGS",
            PositionXY => "POSIZTION_XY",
            CumulativeCoverage => "CUMULATIVE_COVERAGE",
            RasterizerCoverage => "RASTERIZER_COVERAGE",
            SampleCentroidId => "SAMPLE_CENTROID_ID",
            FrameArg => "FRAME_ARG",
            BlendInputSrc0 => "BLEND_IN_SRC0",
            BlendInputSrc1 => "BLEND_IN_SRC1",
            BlendReturnAddr => "BLEND_RETURN_ADDR",
        };
        write!(f, "{name}")
    }
}

pub type PreloadRegSet = U8EnumSet<PreloadReg, 1>;

/// Handle referencing an external resource (e.g. sampler, texture, attribute,
/// uniform buffer...).  It is just a pair of indices, one selecting a "table",
/// the other selecting a descriptor within the table.  Tables are either lists
/// of descriptors stored in memory or a "virtual" table for hardware-controlled
/// resources
pub struct ResHandle {
    /// The resource table index or special table enumerant specifying
    /// which table is being referenced
    pub table: u32,
    /// The index into the table at which to find the descriptor
    pub index: u32,
}

impl ResHandle {
    pub fn from_bits(imm: u32) -> Self {
        ResHandle {
            table: imm.get_bit_range_u64(24..32) as u32,
            index: imm.get_bit_range_u64(0..24) as u32,
        }
    }

    /// Returns true if the handle can be encoded in an immediate
    pub fn fits_imm_op(&self, index_bits: u8) -> bool {
        let table = self.table;
        let table_valid = table <= 11 || (table >= 60 && table <= 63);
        let index_valid = self.index < (1 << index_bits);

        table_valid && index_valid
    }
}

impl TryFrom<&Src> for ResHandle {
    type Error = &'static str;

    fn try_from(value: &Src) -> Result<Self, Self::Error> {
        u32::try_from(&value.src_ref)
            .map(ResHandle::from_bits)
            .map_err(|_| "Not an immediate")
    }
}

impl fmt::Display for ResHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.table, self.index)
    }
}

/// This struct describes the range of registers read or written by a RegRef.
/// The range provided here acts as a mask on the destination but is purely
/// informational for sources.  In all cases, the instruction operates relative
/// to the register itself.  If a source needs to read from a value in the top
/// half of a register, it is swizzled accordingly.  For 16-bit destinations,
/// the instruction itself continues to operate 32 bits wide and the register
/// write is simply masked.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegRange {
    Byte0,
    Byte1,
    Byte2,
    Byte3,
    Half0,
    Half1,
    Regs(u8),
}

impl RegRange {
    #[inline]
    fn byte_offset_count(&self) -> (u8, u8) {
        match self {
            RegRange::Byte0 => (0, 1),
            RegRange::Byte1 => (1, 1),
            RegRange::Byte2 => (2, 1),
            RegRange::Byte3 => (3, 1),
            RegRange::Half0 => (0, 2),
            RegRange::Half1 => (2, 2),
            RegRange::Regs(n) => (0, n * 4),
        }
    }

    fn from_byte_offset_count(
        offset: u8,
        count: u8,
    ) -> Result<RegRange, &'static str> {
        match (offset, count) {
            (0, 1) => Ok(RegRange::Byte0),
            (1, 1) => Ok(RegRange::Byte1),
            (2, 1) => Ok(RegRange::Byte2),
            (3, 1) => Ok(RegRange::Byte3),
            (0, 2) => Ok(RegRange::Half0),
            (2, 2) => Ok(RegRange::Half1),
            (0, _) => {
                if count % 4 == 0 {
                    Ok(RegRange::Regs(count / 4))
                } else {
                    Err("Misaligned register range")
                }
            }
            _ => Err("Misaligned register range"),
        }
    }

    pub fn byte_offset(&self) -> u8 {
        self.byte_offset_count().0
    }

    pub fn bytes(&self) -> u8 {
        self.byte_offset_count().1
    }
}

impl From<RegRange> for Swizzle {
    fn from(range: RegRange) -> Swizzle {
        match range {
            RegRange::Byte0 => Swizzle::B0000,
            RegRange::Byte1 => Swizzle::B1111,
            RegRange::Byte2 => Swizzle::B2222,
            RegRange::Byte3 => Swizzle::B3333,
            RegRange::Half0 => Swizzle::H00,
            RegRange::Half1 => Swizzle::H11,
            RegRange::Regs(_) => Swizzle::NONE,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct RegRef {
    pub idx: u8,
    pub range: RegRange,
}

impl fmt::Display for RegRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_base(f)?;

        match &self.range {
            RegRange::Byte0 => write!(f, ".b0"),
            RegRange::Byte1 => write!(f, ".b1"),
            RegRange::Byte2 => write!(f, ".b2"),
            RegRange::Byte3 => write!(f, ".b3"),
            RegRange::Half0 => write!(f, ".h0"),
            RegRange::Half1 => write!(f, ".h1"),
            RegRange::Regs(n) => {
                if *n > 1 {
                    write!(f, "..{}", self.idx + n)?;
                }
                Ok(())
            }
        }
    }
}

impl RegRef {
    fn fmt_base(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "r{}", self.idx)
    }

    pub fn bytes(&self) -> u8 {
        self.range.bytes()
    }

    pub fn byte_range(&self) -> Range<u16> {
        let (offset, bytes) = self.range.byte_offset_count();
        let b_start = u16::from(self.idx) * 4 + u16::from(offset);
        b_start..(b_start + u16::from(bytes))
    }

    pub fn new(idx: u8, range: RegRange) -> Self {
        Self { idx, range }
    }

    pub fn from_byte_range(range: Range<u16>) -> Result<RegRef, &'static str> {
        let idx = (range.start / 4)
            .try_into()
            .map_err(|_| "Register range too large")?;
        let range = RegRange::from_byte_offset_count(
            (range.start % 4).try_into().unwrap(),
            (range.end - range.start)
                .try_into()
                .map_err(|_| "Register range too large")?,
        )?;
        Ok(RegRef { idx, range })
    }

    pub fn intersect(&self, other: RegRef) -> Option<RegRef> {
        let a = self.byte_range();
        let b = other.byte_range();
        let start = a.start.max(b.start);
        let end = a.end.min(b.end);
        if start >= end {
            None
        } else {
            // Can't be too large, it must be smaller than both a and b
            Some(RegRef::from_byte_range(start..end).unwrap())
        }
    }

    pub fn word(self, word: u8) -> RegRef {
        let RegRange::Regs(nregs) = self.range else {
            assert!(word == 0);
            return self;
        };
        assert!(word < nregs, "RegRef::word() out of bounds");
        RegRef {
            idx: self.idx + word,
            range: RegRange::Regs(1),
        }
    }

    pub fn reg_range(&self) -> Range<u16> {
        let bytes = self.byte_range();
        u16::from(self.idx)..u16::from(bytes.end.div_ceil(4))
    }
}

#[derive(Clone, Copy, PartialEq)]
pub struct MemRef {
    pub offset: u16,
    pub range: u8,
}

impl fmt::Display for MemRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "m{}..{}",
            self.offset,
            self.offset + u16::from(self.range)
        )
    }
}

impl MemRef {
    pub fn bytes(&self) -> u8 {
        self.range
    }

    pub fn byte_range(&self) -> Range<u16> {
        self.offset..(self.offset + u16::from(self.range))
    }

    pub fn from_byte_range(range: Range<u16>) -> Result<MemRef, &'static str> {
        Ok(MemRef {
            offset: range.start,
            range: range
                .len()
                .try_into()
                .map_err(|_| "Memory range too large")?,
        })
    }

    pub fn word(self, word: u8) -> MemRef {
        let offset = u16::from(word) * 4;
        assert!(offset + 4 <= u16::from(self.range));
        MemRef {
            offset: self.offset + offset,
            range: 4,
        }
    }
}

#[derive(Clone, PartialEq)]
pub enum SrcRef {
    /// A zero value
    Zero,
    /// A 32-bit immediate
    Imm32(NonZeroU32),
    Imm64(NonZeroU64),
    FAU(FAURef),
    SSA(SSARef),
    Reg(RegRef),
    Mem(MemRef),
}

impl fmt::Display for SrcRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SrcRef::Zero => write!(f, "k0"),
            SrcRef::Imm32(u) => write!(f, "{u:#x}"),
            SrcRef::Imm64(u) => write!(f, "{u:#x}"),
            SrcRef::FAU(fau) => fau.fmt(f),
            SrcRef::SSA(ssa) => ssa.fmt(f),
            SrcRef::Reg(reg) => reg.fmt(f),
            SrcRef::Mem(mem) => mem.fmt(f),
        }
    }
}

impl fmt::Debug for SrcRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as fmt::Display>::fmt(self, f)
    }
}

impl SrcRef {
    pub fn as_ssa(&self) -> Option<&SSARef> {
        match self {
            SrcRef::SSA(ssa) => Some(ssa),
            _ => None,
        }
    }

    pub fn as_mut_ssa(&mut self) -> Option<&mut SSARef> {
        match self {
            SrcRef::SSA(ssa) => Some(ssa),
            _ => None,
        }
    }

    pub fn as_reg(&self) -> Option<&RegRef> {
        match self {
            SrcRef::Reg(reg) => Some(reg),
            _ => None,
        }
    }

    /// Returns the number of bytes read
    pub fn bytes_read(&self) -> u8 {
        match self {
            SrcRef::Zero => 4,
            SrcRef::Imm32(_) => 4,
            SrcRef::Imm64(_) => 8,
            SrcRef::FAU(fau) => {
                if fau.load64 {
                    8
                } else {
                    4
                }
            }
            SrcRef::SSA(vec) => vec.bytes(),
            SrcRef::Reg(reg) => reg.bytes(),
            SrcRef::Mem(mem) => mem.bytes(),
        }
    }

    pub fn is_small_const(&self) -> bool {
        matches!(
            self,
            SrcRef::FAU(FAURef {
                page: FAUPage::SmallConst,
                ..
            })
        )
    }

    pub fn word(self, word: u8) -> SrcRef {
        match self {
            SrcRef::Zero => SrcRef::Zero,
            SrcRef::Imm32(u) => {
                assert!(word == 0);
                SrcRef::Imm32(u)
            }
            SrcRef::Imm64(u) => {
                assert!(word < 2);
                SrcRef::from((u.get() >> (word * 32)) as u32)
            }
            SrcRef::FAU(fau) => fau.word(word).into(),
            SrcRef::SSA(ssa) => ssa[usize::from(word)].into(),
            SrcRef::Reg(reg) => reg.word(word).into(),
            SrcRef::Mem(mem) => mem.word(word).into(),
        }
    }
}

impl From<u32> for SrcRef {
    fn from(u: u32) -> SrcRef {
        if let Some(nz) = NonZeroU32::new(u) {
            SrcRef::Imm32(nz)
        } else {
            SrcRef::Zero
        }
    }
}

impl From<u64> for SrcRef {
    fn from(u: u64) -> SrcRef {
        if let Some(nz) = NonZeroU64::new(u) {
            SrcRef::Imm64(nz)
        } else {
            SrcRef::Zero
        }
    }
}

impl From<f32> for SrcRef {
    fn from(u: f32) -> SrcRef {
        SrcRef::from(u.to_bits())
    }
}

impl TryFrom<&SrcRef> for u64 {
    type Error = &'static str;

    fn try_from(src_ref: &SrcRef) -> Result<u64, Self::Error> {
        match src_ref {
            SrcRef::Zero => Ok(0),
            SrcRef::Imm32(nz) => Ok(nz.get().into()),
            SrcRef::Imm64(nz) => Ok(nz.get()),
            _ => Err("Value not known at compile time"),
        }
    }
}

impl TryFrom<&SrcRef> for u32 {
    type Error = &'static str;

    fn try_from(src_ref: &SrcRef) -> Result<u32, Self::Error> {
        u64::try_from(src_ref)?
            .try_into()
            .map_err(|_| "Value cannot fit in u32")
    }
}

impl From<FAURef> for SrcRef {
    fn from(fau: FAURef) -> SrcRef {
        SrcRef::FAU(fau)
    }
}

impl<T: Into<SSARef>> From<T> for SrcRef {
    fn from(ssa: T) -> SrcRef {
        SrcRef::SSA(ssa.into())
    }
}

impl From<RegRef> for SrcRef {
    fn from(reg: RegRef) -> SrcRef {
        SrcRef::Reg(reg)
    }
}

impl From<MemRef> for SrcRef {
    fn from(mem: MemRef) -> SrcRef {
        SrcRef::Mem(mem)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Default, Eq, Hash, PartialEq, EnumAsU8, Debug)]
pub enum SrcMod {
    #[default]
    None = 0,
    FAbs = 1,
    FNeg = 2,
    FNegAbs = 3,
    BNot = 4,
}

impl fmt::Display for SrcMod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SrcMod::None => Ok(()),
            SrcMod::FAbs => write!(f, ".fabs"),
            SrcMod::FNeg => write!(f, ".fneg"),
            SrcMod::FNegAbs => write!(f, ".fabs.fneg"),
            SrcMod::BNot => write!(f, ".bnot"),
        }
    }
}

fn float_sign_bits(data_type: DataType) -> Option<u32> {
    match data_type {
        DataType::F16 | DataType::V2F16 => Some(0x80008000),
        DataType::F32 => Some(0x80000000),
        _ => None,
    }
}

impl SrcMod {
    pub fn is_none(&self) -> bool {
        *self == SrcMod::None
    }

    pub fn bnot(self) -> SrcMod {
        use SrcMod::*;
        match self {
            None => BNot,
            FAbs | FNeg | FNegAbs => {
                panic!("Cannot compose float and bitwise modifiers");
            }
            BNot => None,
        }
    }

    pub fn fabs(self) -> SrcMod {
        use SrcMod::*;
        match self {
            None | FAbs | FNeg | FNegAbs => FAbs,
            BNot => panic!("Cannot compose float and bitwise modifiers"),
        }
    }

    pub fn fneg(self) -> SrcMod {
        use SrcMod::*;
        match self {
            None => FNeg,
            FAbs => FNegAbs,
            FNeg => None,
            FNegAbs => FAbs,
            BNot => panic!("Cannot compose float and bitwise modifiers"),
        }
    }

    pub fn modify(self, other: SrcMod) -> SrcMod {
        use SrcMod::*;
        match other {
            None => self,
            FAbs => self.fabs(),
            FNeg => self.fneg(),
            FNegAbs => self.fabs().fneg(),
            BNot => self.bnot(),
        }
    }

    pub fn fold_u32(self, data_type: DataType, u: u32) -> Option<u32> {
        match self {
            SrcMod::None => Some(u),
            SrcMod::FAbs => Some(u & !float_sign_bits(data_type)?),
            SrcMod::FNeg => Some(u ^ float_sign_bits(data_type)?),
            SrcMod::FNegAbs => Some(u | float_sign_bits(data_type)?),
            SrcMod::BNot => Some(!u),
        }
    }

    pub fn fold_u64(self, u: u64) -> Option<u64> {
        match self {
            SrcMod::None => Some(u),
            // No instruction uses F64 or V2F32
            SrcMod::FAbs | SrcMod::FNeg | SrcMod::FNegAbs => None,
            SrcMod::BNot => Some(!u),
        }
    }
}

/// An instruction source, consisting of a SrcRef referencing the actual data,
/// a Swizzle which may shuffle the SrcRef data around, and a SrcMod which
/// modifies the data before it is consumed by the instruction.
///
/// Logically, the swizzle is applied first and then the source modifier.  At
/// the ISA level, it doesn't matter which is applied first because the ISA
/// never allows incompatible source modifiers and swizzles so the source
/// modifier can always be applied either before or after the swizzle without
/// affecting everything.  Howver, because we represent a superset of the ISA,
/// we need the order to be well-defined.
#[derive(Clone, Debug, PartialEq)]
pub struct Src {
    pub src_ref: SrcRef,
    pub swizzle: Swizzle,
    pub src_mod: SrcMod,
    pub last_use: bool,
}

pub struct FmtSrc<'a> {
    src: &'a Src,
    src_type: DataType,
}

fn fmt_constant_scalar(
    value: u64,
    data_type: DataType,
    f: &mut fmt::Formatter,
) -> fmt::Result {
    let bits = usize::from(data_type.bits());

    debug_assert!((1..=64).contains(&bits));

    // Assume value is already masked.
    let hex_digits = bits.div_ceil(4);
    write!(f, "0x{value:0hex_digits$x}")?;

    match data_type.num_type() {
        NumericType::Float => {
            write!(f, " (")?;
            match bits {
                16 => write!(f, "{}", F16::from_bits(value as u16))?,
                32 => write!(f, "{}", f32::from_bits(value as u32))?,
                _ => panic!("unexpected bit size for float"),
            }
            write!(f, ")")
        }
        NumericType::SignedInteger => {
            let shift = 64 - bits;
            let signed = ((value << shift) as i64) >> shift;
            write!(f, " ({signed})")
        }
        NumericType::UnsignedInteger => {
            write!(f, " ({value})")
        }
        NumericType::Integer => {
            let shift = 64 - bits;
            let signed = ((value << shift) as i64) >> shift;

            if signed >= 0 {
                // Signed and unsigned interpretations are identical.
                write!(f, " ({value})")
            } else {
                write!(f, " ({signed}, {value})")
            }
        }
        NumericType::Auto => Ok(()),
    }
}

fn fmt_constant(
    value: u32,
    data_type: DataType,
    swizzle: Swizzle,
    src_mod: SrcMod,
    f: &mut fmt::Formatter,
) -> fmt::Result {
    let total_bits = data_type.total_bits();
    let value = match total_bits {
        0..=32 => {
            let value = swizzle
                .fold_u32(value)
                .expect("invalid 32-bit small-constant swizzle");

            u64::from(
                src_mod
                    .fold_u32(data_type, value)
                    .expect("invalid 32-bit small-constant modifier"),
            )
        }
        64 => {
            // Small constants are stored as 32-bit values, but 64-bit source
            // operands apply 64-bit swizzles and modifiers to the
            // zero-extended value.
            let value = swizzle
                .fold_u64(u64::from(value))
                .expect("invalid 64-bit small-constant swizzle");

            src_mod
                .fold_u64(value)
                .expect("invalid 64-bit small-constant modifier")
        }
        _ => panic!("unsupported small-constant source width"),
    };

    let component_bits = data_type.bits();
    let components = data_type.comps();
    let scalar_type = data_type.scalar_type();

    debug_assert!(component_bits > 0 && component_bits <= 64);
    debug_assert!(u16::from(component_bits) * u16::from(components) <= 64);

    for component in 0..components {
        if component == 0 && components > 1 {
            write!(f, "{{")?;
        } else if component > 0 {
            write!(f, ", ")?;
        }

        let component_bits = usize::from(component_bits);
        let component_start = usize::from(component) * component_bits;
        let component_value = value.get_bit_range_u64(
            component_start..(component_start + component_bits),
        );
        fmt_constant_scalar(component_value, scalar_type, f)?;

        if component == components - 1 && components > 1 {
            write!(f, "}}")?;
        }
    }
    Ok(())
}

impl fmt::Display for FmtSrc<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lu = if self.src.last_use { "^" } else { "" };
        let raw_const = DEBUG.contains(DebugFlags::PRINT_RAW_CONST);
        match &self.src.src_ref {
            SrcRef::Reg(reg) => reg.fmt_base(f)?,
            // Special handling for pretty-printing small constants.
            SrcRef::FAU(FAURef {
                page: FAUPage::SmallConst,
                imm32: Some(value),
                ..
            }) if !raw_const => {
                fmt_constant(
                    *value,
                    self.src_type,
                    self.src.swizzle,
                    self.src.src_mod,
                    f,
                )?;
                write!(f, "{lu}")?;
                return Ok(());
            }
            SrcRef::Imm32(value) if !raw_const => {
                fmt_constant(
                    (*value).into(),
                    self.src_type,
                    self.src.swizzle,
                    self.src.src_mod,
                    f,
                )?;
                write!(f, "{lu}")?;
                return Ok(());
            }
            SrcRef::Zero if !raw_const => {
                fmt_constant(
                    0,
                    self.src_type,
                    self.src.swizzle,
                    self.src.src_mod,
                    f,
                )?;
                write!(f, "{lu}")?;
                return Ok(());
            }
            src_ref => write!(f, "{src_ref}")?,
        }
        write!(f, "{lu}")?;
        if let Some(asm_swz) =
            AsmSwizzleWiden::from_swizzle(self.src_type, self.src.swizzle)
        {
            write!(f, "{asm_swz}")?;
        } else {
            write!(f, "{}", self.src.swizzle)?;
        }
        write!(f, "{}", self.src.src_mod)
    }
}

impl Src {
    /// Swizzles a `Src` by the given `swizzle`.  This helper is intended to be
    /// used in cases where you know what you're doing, such as when building
    /// some IR.  It's up to the caller to guarantee that the resulting swizzle
    /// is valid and non-zero.
    ///
    /// # Panics
    ///
    /// This function panics if the composed swizzle is invalid or zero.
    pub fn swizzle(mut self, swizzle: Swizzle) -> Src {
        self.swizzle = self.swizzle.swizzle(swizzle).unwrap();
        assert!(!self.swizzle.is_zero());

        // If the new swizzle only reads the first word, make sure we only
        // reference the first word.
        if self.swizzle.bytes_read(8) <= 0xf {
            self.src_ref = self.src_ref.word(0);
        }

        self
    }

    pub fn byte(self, byte: u8) -> Src {
        self.swizzle(Swizzle::replicate_byte(byte))
    }

    pub fn half(self, half: u8) -> Src {
        self.swizzle(Swizzle::replicate_half(half))
    }

    pub fn word(self, word: u8) -> Src {
        assert!(word < 2);
        assert!(self.src_mod.is_none());
        if let Some(swizzle_word) = self.swizzle.word(word) {
            use SwizzleWord::*;
            match swizzle_word {
                Zero => 0_u32.into(),
                Word0 => Src::from(self.src_ref.word(0)),
                Word1 => Src::from(self.src_ref.word(1)),
                Sign0 => Src::from(self.src_ref.word(0)).swizzle(Swizzle::S3),
                Sign1 => Src::from(self.src_ref.word(1)).swizzle(Swizzle::S3),
            }
        } else {
            // In this case, it's a byte swizzle that we sign-extend
            debug_assert!(self.swizzle.is_byte_swizzle());
            debug_assert!(self.src_ref.bytes_read() <= 4);
            if word == 0 {
                self
            } else {
                let swizzle = self.swizzle.swizzle(Swizzle::S3).unwrap();
                if swizzle.is_zero() {
                    0_u32.into()
                } else {
                    Src { swizzle, ..self }
                }
            }
        }
    }

    pub fn zero(bits: u8) -> Src {
        match bits {
            8 => Src::from(0_u8),
            16 => Src::from(0_u16),
            32 => Src::from(0_u32),
            _ => panic!("Invalid float bit size"),
        }
    }

    pub fn fneg_zero(bits: u8) -> Src {
        Src::zero(bits).fneg()
    }

    pub fn modify(mut self, src_mod: SrcMod) -> Src {
        self.src_mod = self.src_mod.modify(src_mod);
        self
    }

    pub fn bnot(self) -> Src {
        self.modify(SrcMod::BNot)
    }

    pub fn fabs(self) -> Src {
        self.modify(SrcMod::FAbs)
    }

    pub fn fneg(self) -> Src {
        self.modify(SrcMod::FNeg)
    }

    pub fn is_zero(&self) -> bool {
        if !matches!(self.src_mod, SrcMod::BNot | SrcMod::None) {
            return false;
        }
        // Always assume we are 64-bits, 32-bit swizzles sign-extend to 64-bits
        // before applying .bnot anyways.  u32::MAX.bnot might not be 0 if read
        // from a 64-bit source (unless specifically sign-extended)
        matches!(self.resolve_imm(DataType::U64), Some(0))
    }

    pub fn is_fneg_zero(&self, src_type: DataType) -> bool {
        match self.src_ref {
            SrcRef::Zero => {
                matches!(self.src_mod, SrcMod::FNeg | SrcMod::FNegAbs)
            }
            SrcRef::Imm32(imm) => {
                if let Some(imm) = self.src_mod.fold_u32(src_type, imm.into()) {
                    match src_type {
                        DataType::F16 => (imm as u16) == 0x8000,
                        DataType::V2F16 => imm == 0x80008000,
                        DataType::F32 => imm == 0x80000000,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            // We could possibly detect that an FAU is k0 but that requries
            // digging into the small constant table in the model and it's
            // generally not worth it.  We should just be using Zero when
            // that's what we want.
            _ => false,
        }
    }

    pub fn replicates_byte(&self) -> bool {
        match self.src_ref {
            SrcRef::Zero => true,
            SrcRef::Imm32(u) => {
                self.swizzle.fold_u32(u.into()).is_some_and(|u| {
                    let b = u.to_le_bytes();
                    b[0] == b[1] && b[0] == b[2] && b[0] == b[3]
                })
            }
            SrcRef::Imm64(u) => {
                self.swizzle.fold_u64(u.into()).is_some_and(|u| {
                    let b = u.to_le_bytes();
                    b[1..].iter().all(|e| *e == b[0])
                })
            }
            _ => self.swizzle.replicates_byte(),
        }
    }

    pub fn replicates_half(&self) -> bool {
        match self.src_ref {
            SrcRef::Zero => true,
            SrcRef::Imm32(u) => self
                .swizzle
                .fold_u32(u.into())
                .is_some_and(|u| (u & 0xffff) == (u >> 16)),
            SrcRef::Imm64(u) => {
                self.swizzle.fold_u64(u.into()).is_some_and(|u| {
                    (1..4).all(|i| ((u >> (16 * i)) & 0xffff) == (u & 0xffff))
                })
            }
            _ => self.swizzle.replicates_half(),
        }
    }

    pub fn as_ssa(&self) -> Option<&SSARef> {
        let vec = self.src_ref.as_ssa()?;
        let swz = match vec.bytes() {
            1 => Swizzle::B0000,
            2 => Swizzle::H00,
            _ => Swizzle::NONE,
        };
        if self.src_mod.is_none() && self.swizzle == swz {
            Some(vec)
        } else {
            None
        }
    }

    pub fn resolve_imm(&self, src_type: DataType) -> Option<u64> {
        let imm64 = u64::try_from(&self.src_ref).ok()?;

        match src_type.total_bits() {
            i if i <= 32 => self
                .swizzle
                .fold_u32(imm64 as u32)
                .and_then(|tmp| self.src_mod.fold_u32(src_type, tmp))
                .map(u64::from),
            64 => self
                .swizzle
                .fold_u64(imm64)
                .and_then(|tmp| self.src_mod.fold_u64(tmp)),
            _ => panic!("Invalid source width"),
        }
    }
}

impl<T: Into<SrcRef>> From<T> for Src {
    fn from(src_ref: T) -> Src {
        let src_ref = src_ref.into();
        let swizzle = match &src_ref {
            SrcRef::Zero
            | SrcRef::Imm32(_)
            | SrcRef::Imm64(_)
            | SrcRef::FAU(_) => Swizzle::NONE,
            SrcRef::SSA(vec) => match vec.bytes() {
                1 => Swizzle::B0000,
                2 => Swizzle::H00,
                _ => Swizzle::NONE,
            },
            SrcRef::Reg(reg) => reg.range.into(),
            SrcRef::Mem(mem) => match mem.range {
                1 => Swizzle::B0000,
                2 => Swizzle::H00,
                _ => Swizzle::NONE,
            },
        };
        Src {
            src_ref,
            swizzle,
            src_mod: Default::default(),
            last_use: false,
        }
    }
}

impl From<u16> for Src {
    fn from(u: u16) -> Src {
        Src::from(u32::from(u)).half(0)
    }
}

impl From<u8> for Src {
    fn from(u: u8) -> Src {
        Src::from(u32::from(u)).byte(0)
    }
}

#[derive(Clone)]
pub enum DstRef {
    None,
    SSA(SSARef),
    Reg(RegRef),
    Mem(MemRef),
}

impl fmt::Display for DstRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DstRef::None => write!(f, "null"),
            DstRef::SSA(ssa) => ssa.fmt(f),
            DstRef::Reg(reg) => reg.fmt(f),
            DstRef::Mem(mem) => mem.fmt(f),
        }
    }
}

impl DstRef {
    pub fn as_ssa(&self) -> Option<&SSARef> {
        match self {
            DstRef::SSA(ssa) => Some(ssa),
            _ => None,
        }
    }

    pub fn as_mut_ssa(&mut self) -> Option<&mut SSARef> {
        match self {
            DstRef::SSA(ssa) => Some(ssa),
            _ => None,
        }
    }

    pub fn as_reg(&self) -> Option<&RegRef> {
        match self {
            DstRef::Reg(reg) => Some(reg),
            _ => None,
        }
    }

    pub fn as_mem(&self) -> Option<&MemRef> {
        match self {
            DstRef::Mem(mem) => Some(mem),
            _ => None,
        }
    }

    pub fn is_none(&self) -> bool {
        matches!(self, DstRef::None)
    }

    pub fn bytes_written(&self) -> u8 {
        match self {
            DstRef::None => 0,
            DstRef::SSA(vec) => vec.bytes(),
            DstRef::Reg(reg) => reg.bytes(),
            DstRef::Mem(mem) => mem.bytes(),
        }
    }

    pub fn word(self, word: u8) -> DstRef {
        match self {
            DstRef::None => DstRef::None,
            DstRef::SSA(ssa) => ssa[usize::from(word)].into(),
            DstRef::Reg(reg) => reg.word(word).into(),
            DstRef::Mem(mem) => mem.word(word).into(),
        }
    }
}

impl<T: Into<SSARef>> From<T> for DstRef {
    fn from(ssa: T) -> DstRef {
        DstRef::SSA(ssa.into())
    }
}

impl From<RegRef> for DstRef {
    fn from(reg: RegRef) -> DstRef {
        DstRef::Reg(reg)
    }
}

impl From<MemRef> for DstRef {
    fn from(mem: MemRef) -> DstRef {
        DstRef::Mem(mem)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, EnumAsU8, Eq, Hash, PartialEq)]
pub enum DstLanes {
    /// The destination is never written
    None,

    /// All lanes of the destination are written
    All,

    /// It only writes one byte, but all bytes are the same so it doesn't
    /// matter which one you pick.  This gives the most freedom to register
    /// assignment.
    AnyB,

    /// It only writes one 16-bit half, but both halves are the same so it
    /// doesn't matter which one you pick.  This gives the most freedom to
    /// register assignment.
    AnyH,

    /// Narrow to F16, using either half.
    AnyHF,

    // Bytes
    B0,
    B1,
    B2,
    B3,

    // Halves
    H0,
    H1,

    // Narrow to F16, choosing H0 or H1
    HF0,
    HF1,
}

pub type DstLanesSet = <DstLanes as EnumAsU8>::VariantSet;
pub type AsmSwizzleWidenSet = <AsmSwizzleWiden as EnumAsU8>::VariantSet;

impl fmt::Display for DstLanes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DstLanes::None => Ok(()),
            DstLanes::All => Ok(()),
            DstLanes::AnyB => write!(f, ".any_b"),
            DstLanes::AnyH => write!(f, ".any_h"),
            DstLanes::AnyHF => write!(f, ".any_hf"),
            DstLanes::B0 => write!(f, ".b0"),
            DstLanes::B1 => write!(f, ".b1"),
            DstLanes::B2 => write!(f, ".b2"),
            DstLanes::B3 => write!(f, ".b3"),
            DstLanes::H0 => write!(f, ".h0"),
            DstLanes::H1 => write!(f, ".h1"),
            DstLanes::HF0 => write!(f, ".hf0"),
            DstLanes::HF1 => write!(f, ".hf1"),
        }
    }
}

impl From<RegRange> for DstLanes {
    fn from(range: RegRange) -> DstLanes {
        match range {
            RegRange::Byte0 => DstLanes::B0,
            RegRange::Byte1 => DstLanes::B1,
            RegRange::Byte2 => DstLanes::B2,
            RegRange::Byte3 => DstLanes::B3,
            RegRange::Half0 => DstLanes::H0,
            RegRange::Half1 => DstLanes::H1,
            RegRange::Regs(_) => DstLanes::All,
        }
    }
}

impl DstLanes {
    pub const ALL_B: DstLanesSet = unsafe {
        DstLanesSet::from_u8_array([
            DstLanes::AnyB as u8,
            DstLanes::B0 as u8,
            DstLanes::B1 as u8,
            DstLanes::B2 as u8,
            DstLanes::B3 as u8,
        ])
    };

    pub const ALL_H: DstLanesSet = unsafe {
        DstLanesSet::from_u8_array([
            DstLanes::AnyH as u8,
            DstLanes::H0 as u8,
            DstLanes::H1 as u8,
        ])
    };

    pub const ALL_HF: DstLanesSet = unsafe {
        DstLanesSet::from_u8_array([
            DstLanes::AnyHF as u8,
            DstLanes::HF0 as u8,
            DstLanes::HF1 as u8,
        ])
    };

    pub fn byte(byte: u8) -> DstLanes {
        match byte {
            0 => DstLanes::B0,
            1 => DstLanes::B1,
            2 => DstLanes::B2,
            3 => DstLanes::B3,
            _ => panic!("Invalid lane byte"),
        }
    }

    pub fn half(byte: u8) -> DstLanes {
        match byte {
            0 => DstLanes::H0,
            1 => DstLanes::H1,
            _ => panic!("Invalid lane half"),
        }
    }

    pub fn bytes(&self, dst_bytes: u8) -> u8 {
        use DstLanes::*;
        match self {
            None => 0,
            All => dst_bytes,
            AnyH | AnyHF | H0 | H1 | HF0 | HF1 => 2,
            AnyB | B0 | B1 | B2 | B3 => 1,
        }
    }

    pub fn align(&self) -> (u8, u8) {
        match self {
            DstLanes::None => (0, 0),
            DstLanes::All => (4, 0),
            DstLanes::AnyB => (1, 0),
            DstLanes::AnyH | DstLanes::AnyHF => (2, 0),
            DstLanes::B0 => (4, 0),
            DstLanes::B1 => (4, 1),
            DstLanes::B2 => (4, 2),
            DstLanes::B3 => (4, 3),
            DstLanes::H0 | DstLanes::HF0 => (4, 0),
            DstLanes::H1 | DstLanes::HF1 => (4, 2),
        }
    }

    pub fn is_byte(&self) -> bool {
        DstLanes::ALL_B.contains(*self)
    }

    pub fn is_half(&self) -> bool {
        DstLanes::ALL_H.contains(*self)
    }

    pub fn is_f16_narrow(&self) -> bool {
        DstLanes::ALL_HF.contains(*self)
    }

    pub fn u32_mask(&self) -> Option<u32> {
        match self {
            DstLanes::None => Some(0),
            DstLanes::All => Some(!0_u32),
            DstLanes::AnyB => None,
            DstLanes::AnyH | DstLanes::AnyHF => None,
            DstLanes::B0 => Some(0x000000ff),
            DstLanes::B1 => Some(0x0000ff00),
            DstLanes::B2 => Some(0x00ff0000),
            DstLanes::B3 => Some(0xff000000),
            DstLanes::H0 | DstLanes::HF0 => Some(0x0000ffff),
            DstLanes::H1 | DstLanes::HF1 => Some(0xffff0000),
        }
    }

    pub fn as_byte_range(&self) -> Option<Range<u8>> {
        match self {
            DstLanes::None => Some(0..0),
            DstLanes::All => Some(0..4),
            DstLanes::AnyB => None,
            DstLanes::AnyH | DstLanes::AnyHF => None,
            DstLanes::B0 => Some(0..1),
            DstLanes::B1 => Some(1..2),
            DstLanes::B2 => Some(2..3),
            DstLanes::B3 => Some(3..4),
            DstLanes::H0 | DstLanes::HF0 => Some(0..2),
            DstLanes::H1 | DstLanes::HF1 => Some(2..4),
        }
    }
}

#[derive(Clone)]
pub struct Dst {
    pub dst_ref: DstRef,
    pub lanes: DstLanes,
}

impl fmt::Display for Dst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.dst_ref {
            DstRef::Reg(reg) => reg.fmt_base(f)?,
            dst_ref => write!(f, "{dst_ref}")?,
        }
        write!(f, "{}", self.lanes)
    }
}

impl Dst {
    // NOTE:
    //
    // We only support word on Dst because SSAValue is at most 32 bits so we
    // can write to the different words of an SSSARef and still be in SSA form.
    pub fn word(mut self, word: u8) -> Dst {
        assert_eq!(self.lanes, DstLanes::All);
        self.dst_ref = self.dst_ref.word(word);
        self
    }
}

impl<T: Into<DstRef>> From<T> for Dst {
    fn from(dst_ref: T) -> Dst {
        let dst_ref: DstRef = dst_ref.into();
        let lanes = match &dst_ref {
            DstRef::None => DstLanes::None,
            DstRef::SSA(vec) => {
                let bits = vec[0].bits();
                if bits == 8 {
                    debug_assert_eq!(vec.comps(), 1);
                    DstLanes::AnyB
                } else if bits == 16 {
                    debug_assert_eq!(vec.comps(), 1);
                    DstLanes::AnyH
                } else {
                    debug_assert_eq!(bits, 32);
                    DstLanes::All
                }
            }
            DstRef::Reg(reg) => reg.range.into(),
            DstRef::Mem(mem) => match mem.range {
                1 => DstLanes::AnyB,
                2 => DstLanes::AnyH,
                _ => DstLanes::All,
            },
        };
        Dst { dst_ref, lanes }
    }
}

pub trait HasVariants {
    const VARIANTS: &'static [DataType];

    fn variant(&self) -> DataType;

    fn set_variant(&mut self, variant: DataType);

    fn is_valid_variant(variant: DataType) -> bool {
        Self::VARIANTS.contains(&variant)
    }
}

#[derive(Clone)]
pub struct DataTypeIter {
    variant: Option<DataType>,
    types: std::slice::Iter<'static, PartialDataType>,
}

impl Iterator for DataTypeIter {
    type Item = DataType;

    fn next(&mut self) -> Option<DataType> {
        let t = self.types.next()?;
        if let Some(v) = self.variant {
            Some(t.specialize(v))
        } else {
            Some(t.as_data_type())
        }
    }
}

pub trait DisplayOp: AsSlice<Dst, Attr = PartialDataType> {
    fn fmt_dsts(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dsts = self.as_slice();

        for (i, dst) in dsts.iter().enumerate() {
            if i != 0 {
                write!(f, " ")?;
            }
            write!(f, "{}", &dst)?;
        }
        Ok(())
    }

    fn fmt_name(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    fn fmt_body(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result;

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dst = format!("{}", Fmt(|f| self.fmt_dsts(f)));
        if !dst.is_empty() {
            write!(f, "{dst} = ")?;
        }
        self.fmt_name(f)?;
        self.fmt_body(f)?;
        Ok(())
    }
}

pub trait Opcode:
    AsSlice<Src, Attr = PartialDataType>
    + AsSlice<Dst, Attr = PartialDataType>
    + DisplayOp
{
    fn variant(&self) -> Option<DataType>;
    fn set_variant(&mut self, data_type: DataType);
    fn is_valid_variant(&self, data_type: DataType) -> bool;

    fn srcs(&self) -> &[Src] {
        self.as_slice()
    }

    fn srcs_mut(&mut self) -> &mut [Src] {
        self.as_mut_slice()
    }

    fn src_types(&self) -> DataTypeIter {
        DataTypeIter {
            variant: self.variant(),
            types: AsSlice::<Src>::attrs(self).iter(),
        }
    }

    fn raw_src_types(&self) -> &[PartialDataType] {
        AsSlice::<Src>::attrs(self)
    }

    fn srcs_types(&self) -> impl Iterator<Item = (&Src, DataType)> {
        let t = self.src_types();
        self.srcs().iter().zip(t)
    }

    fn srcs_raw_types(&self) -> impl Iterator<Item = (&Src, PartialDataType)> {
        let t = self.raw_src_types().iter().cloned();
        self.srcs().iter().zip(t)
    }

    fn srcs_types_mut(&mut self) -> impl Iterator<Item = (&mut Src, DataType)> {
        let t = self.src_types();
        self.srcs_mut().iter_mut().zip(t)
    }

    fn src_idx(&self, src: &Src) -> usize {
        let r = self.srcs().as_ptr_range();
        assert!(r.contains(&(src as *const Src)));
        unsafe { (src as *const Src).offset_from(r.start) as usize }
    }

    fn src_type(&self, src: &Src) -> DataType {
        let src_idx = self.src_idx(src);
        let src_type = AsSlice::<Src>::attrs(self)[src_idx];
        if let Some(v) = self.variant() {
            src_type.specialize(v)
        } else {
            src_type.as_data_type()
        }
    }

    fn iter_ssa_uses(&self) -> impl DoubleEndedIterator<Item = &SSAValue> {
        self.srcs().iter().flat_map(|src| {
            if let SrcRef::SSA(vec) = &src.src_ref {
                vec.iter()
            } else {
                (&[]).iter()
            }
        })
    }

    fn iter_ssa_uses_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut SSAValue> {
        self.srcs_mut().iter_mut().flat_map(|src| {
            if let SrcRef::SSA(vec) = &mut src.src_ref {
                vec.iter_mut()
            } else {
                (&mut []).iter_mut()
            }
        })
    }

    fn iter_ssa_defs(&self) -> impl DoubleEndedIterator<Item = &SSAValue> {
        self.dsts().iter().flat_map(|dst| {
            if let DstRef::SSA(vec) = &dst.dst_ref {
                vec.iter()
            } else {
                (&[]).iter()
            }
        })
    }

    fn iter_ssa_defs_mut(
        &mut self,
    ) -> impl DoubleEndedIterator<Item = &mut SSAValue> {
        self.dsts_mut().iter_mut().flat_map(|dst| {
            if let DstRef::SSA(vec) = &mut dst.dst_ref {
                vec.iter_mut()
            } else {
                (&mut []).iter_mut()
            }
        })
    }

    fn iter_reg_uses(&self) -> impl DoubleEndedIterator<Item = &RegRef> {
        self.srcs().iter().filter_map(|src| src.src_ref.as_reg())
    }

    fn iter_reg_defs(&self) -> impl DoubleEndedIterator<Item = &RegRef> {
        self.dsts().iter().filter_map(|dst| dst.dst_ref.as_reg())
    }

    fn fmt_src<'a>(&self, src: &'a Src) -> FmtSrc<'a> {
        FmtSrc {
            src,
            src_type: self.src_type(src),
        }
    }

    fn fmt_handle_src(&self, src: &Src) -> String {
        if let Ok(descr) = ResHandle::try_from(src) {
            descr.to_string()
        } else {
            self.fmt_src(src).to_string()
        }
    }

    fn dsts(&self) -> &[Dst] {
        self.as_slice()
    }

    fn dsts_mut(&mut self) -> &mut [Dst] {
        self.as_mut_slice()
    }

    fn dst_types(&self) -> DataTypeIter {
        DataTypeIter {
            variant: self.variant(),
            types: AsSlice::<Dst>::attrs(self).iter(),
        }
    }

    fn raw_dst_types(&self) -> &[PartialDataType] {
        AsSlice::<Dst>::attrs(self)
    }

    fn dsts_types(&self) -> impl Iterator<Item = (&Dst, DataType)> {
        let t = self.dst_types();
        self.dsts().iter().zip(t)
    }

    fn dsts_raw_types(&self) -> impl Iterator<Item = (&Dst, PartialDataType)> {
        let t = self.raw_dst_types().iter().cloned();
        self.dsts().iter().zip(t)
    }

    #[allow(dead_code)]
    fn dsts_types_mut(&mut self) -> impl Iterator<Item = (&mut Dst, DataType)> {
        let t = self.dst_types();
        self.dsts_mut().iter_mut().zip(t)
    }

    fn dst_idx(&self, dst: &Dst) -> usize {
        let r = self.dsts().as_ptr_range();
        assert!(r.contains(&(dst as *const Dst)));
        unsafe { (dst as *const Dst).offset_from(r.start) as usize }
    }

    fn dst_type(&self, dst: &Dst) -> DataType {
        let dst_idx = self.dst_idx(dst);
        let dst_type = AsSlice::<Dst>::attrs(self)[dst_idx];
        if let Some(v) = self.variant() {
            dst_type.specialize(v)
        } else {
            dst_type.as_data_type()
        }
    }
}

/// A trait that allows querying various properties of an opcode.  Virtual ops,
/// which must be lowered implement this trait directly while it may require
/// going through `Model` for other ops.
pub trait VirtualOpcode {
    fn is_message(&self) -> bool {
        false
    }

    fn src_is_staging_reg(&self, _src: &Src) -> bool {
        false
    }

    fn src_is_64bit(&self, _src: &Src) -> bool {
        false
    }

    fn src_supports_imm32(&self, _src: &Src, _imm: u32) -> bool {
        false
    }

    fn src_supports_swizzle(&self, _src: &Src, swizzle: Swizzle) -> bool {
        swizzle == Swizzle::NONE
    }

    fn src_supported_swizzles(
        &self,
        src: &Src,
        src_type: DataType,
    ) -> AsmSwizzleWidenSet {
        // This is for correctness, but should be implemented specifically for Ops where this is
        // called often.
        AsmSwizzleWiden::VARIANTS
            .iter()
            .filter(|asw| {
                let Some(swizzle) = asw.to_swizzle(src_type) else {
                    return false;
                };
                self.src_supports_swizzle(src, swizzle)
            })
            .collect()
    }

    fn src_supports_mod(&self, _src: &Src, src_mod: SrcMod) -> bool {
        src_mod.is_none()
    }

    fn dst_is_staging_reg(&self) -> bool {
        false
    }

    fn dst_supported_lanes(&self) -> DstLanesSet {
        DstLanesSet::from_array([DstLanes::All])
    }
}

// Hack struct so we can re-use Formatters.  Shamelessly stolen from
// https://users.rust-lang.org/t/reusing-an-fmt-formatter/8531/4
pub struct Fmt<F>(pub F)
where
    F: Fn(&mut fmt::Formatter) -> fmt::Result;

impl<F> fmt::Display for Fmt<F>
where
    F: Fn(&mut fmt::Formatter) -> fmt::Result,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        (self.0)(f)
    }
}

#[derive(Clone)]
pub struct Instr {
    pub op: Op,
    pub flow: FlowCtrl,
}

impl Instr {
    pub const MAX_SRC_COUNT: usize = 5;
}

impl Deref for Instr {
    type Target = Op;

    fn deref(&self) -> &Op {
        &self.op
    }
}

impl DerefMut for Instr {
    fn deref_mut(&mut self) -> &mut Op {
        &mut self.op
    }
}

impl<T: Into<Op>> From<T> for Instr {
    fn from(op: T) -> Instr {
        let op = op.into();
        if let Some(variant) = op.variant() {
            assert!(op.is_valid_variant(variant));
        }
        Instr {
            op,
            flow: Default::default(),
        }
    }
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dst = format!("{}", Fmt(|f| self.op.fmt_dsts(f)));
        if !dst.is_empty() {
            write!(f, "{dst} = ")?;
        }
        self.op.fmt_name(f)?;
        write!(f, "{}", self.flow)?;
        self.op.fmt_body(f)?;
        Ok(())
    }
}

pub type MappedInstrs = SmallVec<Instr>;

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub struct Label {
    idx: u32,
}

impl fmt::Display for Label {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "__label{}", self.idx)
    }
}

#[derive(Default)]
pub struct LabelAllocator {
    count: u32,
}

impl LabelAllocator {
    pub fn alloc(&mut self) -> Label {
        let idx = self.count;
        self.count += 1;
        Label { idx }
    }
}

pub struct BasicBlock {
    pub label: Label,
    pub instrs: Vec<Instr>,
}

impl BasicBlock {
    pub fn map_instrs(&mut self, map: impl FnMut(Instr) -> MappedInstrs) {
        let instrs = std::mem::take(&mut self.instrs);
        self.instrs = instrs.into_iter().flat_map(map).collect();
    }

    // SAFETY: The caller must guarantee that:
    //
    //  - For each `i in 0..instrs.len()`, `remap_idx(i)` returns `None` or
    //    `Some(j)` where `(0..count).contains(j)`
    //
    //  - For each `j in 0..count`, `Some(j)` is returned exactly once
    //
    pub unsafe fn reorder_instrs(
        &mut self,
        remap_ip: impl Fn(usize) -> Option<usize>,
        count: usize,
    ) {
        let instrs = std::mem::take(&mut self.instrs);

        self.instrs.reserve(count);
        let uninit = self.instrs.spare_capacity_mut();

        for (ip, instr) in instrs.into_iter().enumerate() {
            if let Some(r) = remap_ip(ip) {
                uninit[r].write(instr);
            }
        }

        unsafe { self.instrs.set_len(count) };
    }

    pub fn is_prelude_instr(instr: &Instr) -> bool {
        matches!(&instr.op, Op::PhiDst(_) | Op::RegIn(_))
    }

    pub fn is_postlude_instr(instr: &Instr) -> bool {
        match &instr.op {
            Op::Branch(_) | Op::PhiSrc(_) | Op::RegOut(_) => true,
            Op::Nop(_) => instr.flow.get_end_shader(),
            _ => false,
        }
    }

    pub fn is_branch_instr(instr: &Instr) -> bool {
        matches!(&instr.op, Op::Branch(_))
    }

    /// Returns the IP which marks the end of the prelude and the start of the
    /// body section.  This will be the IP of the first body instruction.
    fn prelude_end_ip(&self) -> usize {
        for (ip, instr) in self.instrs.iter().enumerate() {
            if !BasicBlock::is_prelude_instr(instr) {
                return ip;
            }
        }
        self.instrs.len()
    }

    /// Returns the IP which marks the end of the body section and the start of
    /// the postlude.  This will be the IP of the first postlude instruction.
    fn postlude_start_ip(&self) -> usize {
        for (ip, instr) in self.instrs.iter().enumerate().rev() {
            if !BasicBlock::is_postlude_instr(instr) {
                return ip + 1;
            }
        }
        0
    }

    /// Returns the IP range of the instructions that make up this block's body.
    /// These are all the instructions after the prelude but before the
    /// postlude.
    pub fn body_ip_range(&self) -> Range<usize> {
        self.prelude_end_ip()..self.postlude_start_ip()
    }

    /// Returns the ip of the OpBranch or the end of the block.
    fn branch_ip_or_end(&self) -> usize {
        if self.instrs.last().is_some_and(BasicBlock::is_branch_instr) {
            self.instrs.len() - 1
        } else {
            self.instrs.len()
        }
    }

    pub fn iter_prelude(&self) -> impl Iterator<Item = &Instr> {
        self.instrs
            .iter()
            .take_while(|instr| BasicBlock::is_prelude_instr(*instr))
    }

    pub fn iter_postlude_mut(&mut self) -> impl Iterator<Item = &mut Instr> {
        let start = self.postlude_start_ip();
        self.instrs[start..].iter_mut()
    }

    pub fn iter_phi_dsts(&self) -> impl Iterator<Item = &OpPhiDst> {
        self.iter_prelude().filter_map(|instr| {
            if let Op::PhiDst(op) = &instr.op {
                Some(op.as_ref())
            } else {
                None
            }
        })
    }

    pub fn iter_phi_srcs_mut(&mut self) -> impl Iterator<Item = &mut OpPhiSrc> {
        self.iter_postlude_mut().filter_map(|instr| {
            if let Op::PhiSrc(op) = &mut instr.op {
                Some(op.as_mut())
            } else {
                None
            }
        })
    }

    pub fn insert_after_prelude(
        &mut self,
        iter: impl IntoIterator<Item = Instr>,
    ) {
        let ip = self.prelude_end_ip();
        self.instrs.splice(ip..ip, iter);
    }

    pub fn insert_before_postlude(
        &mut self,
        iter: impl IntoIterator<Item = Instr>,
    ) {
        let ip = self.postlude_start_ip();
        self.instrs.splice(ip..ip, iter);
    }

    pub fn insert_phi_dsts(
        &mut self,
        iter: impl IntoIterator<Item = OpPhiDst>,
    ) {
        let iter = iter.into_iter().map(Instr::from);
        self.instrs.splice(0..0, iter);
    }

    pub fn insert_phi_srcs(
        &mut self,
        iter: impl IntoIterator<Item = OpPhiSrc>,
    ) {
        let iter = iter.into_iter().map(Instr::from);
        let ip = self.branch_ip_or_end();
        self.instrs.splice(ip..ip, iter);
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:", self.label)?;
        for i in &self.instrs {
            write!(f, "\n    {i}")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct ShaderInfo {
    /// Number of registers used
    pub registers_used: u8,
    /// Thread local storage size, in bytes
    pub tls_size: u32,
    /// Bitset of preloaded registers
    pub register_preload: u64,
    /// True if we have OpLdGclk
    pub has_ld_gclk: bool,
    /// True if we have any flat load
    pub uses_flat_shading: bool,

    /// Fragment shaders blend (and blend2) types
    pub blend_types: [Option<DataType>; 8],
    pub blend1_type: Option<DataType>,
    /// Is this a blend shader?
    pub is_blend: bool,
}

impl ShaderInfo {
    pub fn add_preload(&mut self, reg: &RegRef) {
        debug_assert!(reg.bytes() % 4 == 0);
        for i in 0..(reg.bytes() / 4) {
            self.register_preload |= 1 << (reg.idx + i);
        }
    }
}

/// Constant data from nir_opt_large_constants, appended to the shader
/// binary at encode time and addressed PC-relative through its label.
pub struct ConstantPool {
    pub label: Label,
    pub data: Vec<u8>,
}

pub struct Shader<'a> {
    pub model: &'a dyn Model,
    pub ssa_alloc: SSAValueAllocator,
    pub phi_alloc: PhiAllocator,
    pub blocks: CFG<BasicBlock>,
    pub info: ShaderInfo,
    pub constant_pool: Option<ConstantPool>,
}

impl Shader<'_> {
    pub fn map_instrs(
        &mut self,
        mut map: impl FnMut(Instr, &mut SSAValueAllocator) -> MappedInstrs,
    ) {
        let alloc = &mut self.ssa_alloc;
        for b in &mut self.blocks {
            b.map_instrs(|i| map(i, alloc));
        }
    }

    pub fn run_pass<R>(
        &mut self,
        name: &str,
        pass: impl FnOnce(&mut Self) -> R,
    ) -> R {
        let res = pass(self);
        if DEBUG.contains(DebugFlags::PRINT) {
            eprintln!("Kraid shader after {name}:\n{self}");
        }
        self.validate();
        res
    }

    pub fn is_empty(&self) -> bool {
        self.blocks.len() == 1
            && self.blocks[0].instrs.len() == 1
            && matches!(self.blocks[0].instrs[0].op, Op::Nop(_))
    }

    pub fn fmt_annotate(
        &self,
        f: &mut fmt::Formatter<'_>,
        mut annotator: impl FnMut(usize, usize, &Instr) -> String,
    ) -> fmt::Result {
        enum Line {
            Empty,
            Label(String),
            Instr {
                dst: String,
                body: String,
                annot: String,
            },
        }
        let mut max_dst_len = 0usize;
        let mut max_body_len = 0usize;
        let space_before_instr = 4;

        let mut rows = vec![];
        for (bi, block) in self.blocks.iter().enumerate() {
            rows.push(Line::Label(block.label.to_string()));

            for (ip, instr) in block.instrs.iter().enumerate() {
                let mut dst = format!("{}", Fmt(|f| instr.fmt_dsts(f)));
                if !dst.is_empty() {
                    dst = format!("{dst} = ");
                }

                let body = format!(
                    "{}{}{}",
                    Fmt(|f| instr.fmt_name(f)),
                    instr.flow,
                    Fmt(|f| instr.fmt_body(f)),
                );
                let annot = annotator(bi, ip, instr);

                max_dst_len = max_dst_len.max(dst.chars().count());
                max_body_len = max_body_len.max(body.chars().count());

                rows.push(Line::Instr { dst, body, annot });
            }
            rows.push(Line::Empty);
        }

        let dst_align = space_before_instr + max_dst_len;
        let body_align = max_body_len;

        for row in rows {
            match row {
                Line::Empty => writeln!(f)?,
                Line::Label(l) => writeln!(f, "{l}:")?,
                Line::Instr { dst, body, annot } => {
                    write!(f, "{dst:>dst_align$}")?;
                    if annot.is_empty() {
                        writeln!(f, "{body}")?;
                    } else {
                        writeln!(f, "{body:<body_align$} # {annot}")?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl fmt::Display for Shader<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_annotate(f, |_, _, _| String::new())
    }
}

macro_rules! pass {
    ($s:ident . $method:ident ( $($args:tt)* )) => {
        $s.run_pass(stringify!($method), |x| x.$method($($args)*))
    };
}

pub(crate) use pass;
