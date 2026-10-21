use core::fmt;
use std::f32::consts::{E, PI};
use std::ops::Range;
use std::sync::OnceLock;
use std::{io, iter, slice};

use crate::builder::*;
use crate::data_type::NumericType;
use crate::debug::{DEBUG, DebugFlags};
use crate::flow::FlowWaitBit;
use crate::foldable::{FoldData, Foldable};
use crate::ir::*;
use crate::model::{Model, model_for_gpu_id};
use crate::ops::*;
use crate::ssa_value::{AllocSSA, SSAValueAllocator};
use crate::swizzle::{AsmSwizzleWiden, SwizzleByte, SwizzleWord};
use acorn::Acorn;
use compiler::cfg::CFGBuilder;
use compiler::enum_as_u8::EnumAsU8;
use compiler::float16::F16;
use kraid_hw_runner::{HwError, InvocationInfo, TestRunner};
use mesa_util::bitview::BitViewable;
use rustc_hash::FxBuildHasher;

/// Enables libpanfrost_decode logs for debugging purposes.
const DEVICE_DEBUG: bool = false;

/// Even when the test does not use the FAU directly, it is still needed
/// to load CB0 args.
const FAU_ONLY_ARGS: &[u32] = &[0u32; 4];

struct RunSingleton {
    model: Box<dyn Model + Sync + Send>,
    runner: TestRunner,
}

static RUN_SINGLETON: OnceLock<RunSingleton> = OnceLock::new();

/// Interesting f32 inputs.  Every entry is also used negated,
static F32_CURATED: &[f32] = &[
    // Trivial values
    0.0,
    1.0,
    0.5,
    2.0,
    8.0,
    32.0,
    1.5,
    2.5,
    PI,
    E,
    // Specials
    f32::INFINITY,
    f32::NAN,
    f32::from_bits(f32::NAN.to_bits() | 0x1),
    f32::from_bits(f32::NAN.to_bits() | 0x535),
    f32::from_bits(0x7FFFFFFF), // NaN, all payload bits set
    f32::from_bits(0x7F800001), // signaling NaN
    f32::from_bits(0x7FBFFFFF), // signaling NaN, all payload bits set
    // ULP neighbours
    f32::from_bits(0x3EFF_FFFF), // 0.5.next_down
    f32::from_bits(0x3F7F_FFFF), // 1.0.next_down
    f32::from_bits(0x3F80_0001), // 1.0.next_up
    // Subnormals and the normal/subnormal boundary
    f32::from_bits(0x0000_0001), // 0.0.next_up
    f32::MIN_POSITIVE,
    f32::from_bits(0x0080_0001), // MIN_POSITIVE.next_up
    f32::from_bits(0x007F_FFFF), // MIN_POSITIVE.next_down
    f32::EPSILON,
    f32::MAX,
    f32::from_bits(0x7F7F_FFFE), // MAX.next_down
    // Mid-range exponents, to test product/sum subnormals
    f32::from_bits(0x0D80_0000), // 2^-100
    f32::from_bits(0x2000_0000), // 2^-63
    f32::from_bits(0x5F00_0000), // 2^63
    f32::from_bits(0x7180_0000), // 2^100
    // Integer boundaries.
    16777216.0,                  // 2^24
    f32::from_bits(0x4EFF_FFFF), // 2147483520.0, 2^31.next_down
    2147483648.0,                // 2^31
    f32::from_bits(0x4F7F_FFFF), // 4294967040.0, 2^32.next_down
    4294967296.0,                // 2^32
    // f16 range boundaries
    f32::from_bits(0x3300_0000), // 2^-25, rtne tie to +0
    f32::from_bits(0x3300_0001), // 2^-25.next_up, rounds to f16 subnormal
    f32::from_bits(0x3380_0000), // 2^-24, smallest f16 subnormal
    f32::from_bits(0x3880_0000), // 2^-14, smallest f16 normal
    f32::from_bits(0x3F80_1000), // 1.0 + 2^-11, rtne tie down to 1.0
    f32::from_bits(0x3F80_3000), // 1.0 + 3*2^-11, rtne tie up
    65504.0,                     // f16::MAX
    65519.0,                     // f16::MAX.next_up rtne tie, rounds down
    65520.0,                     // rounds to f16 infinity
];

/// Interesting f16 inputs.
static F16_CURATED: &[F16] = &[
    // Trivial values
    F16::ZERO,
    F16::from_bits(0x3C00), // 1.0
    F16::from_bits(0x3800), // 0.5
    F16::from_bits(0x4000), // 2.0
    F16::from_bits(0x4800), // 8.0
    F16::from_bits(0x5000), // 32.0
    F16::from_bits(0x3E00), // 1.5
    F16::from_bits(0x4100), // 2.5
    F16::from_bits(0x4248), // PI
    F16::from_bits(0x4170), // E
    // Specials
    F16::INFINITY,
    F16::NAN,
    F16::from_bits(0x7E01), // NaN with payload
    F16::from_bits(0x7E35), // NaN with payload
    F16::from_bits(0x7FFF), // NaN, all payload bits set
    F16::from_bits(0x7C01), // signaling NaN
    F16::from_bits(0x7DFF), // signaling NaN, all payload bits set
    // ULP neighbours
    F16::from_bits(0x37FF), // 0.5.next_down
    F16::from_bits(0x3BFF), // 1.0.next_down
    F16::from_bits(0x3C01), // 1.0.next_up
    // Subnormals and the normal/subnormal boundary
    F16::from_bits(0x0001), // smallest subnormal
    F16::from_bits(0x0400), // exp=1
    F16::from_bits(0x0401), // exp=1, man=1
    F16::from_bits(0x03FF), // largest subnormal
    F16::from_bits(0x1400), // f16::EPSILON
    F16::MAX,
    F16::from_bits(0x7bfe), // MAX.next_down
    // Mid-range exponents, to test product/sum subnormals
    F16::from_bits(0x1C00), // 2^-8
    F16::from_bits(0x5C00), // 2^8
    // Integer boundaries
    F16::from_bits(0x6400), // 1024, 2^10
    F16::from_bits(0x6800), // 2048, 2^11
    F16::from_bits(0x6801), // 2050, (2049 is not representable)
];

impl RunSingleton {
    fn create() -> Result<RunSingleton, HwError> {
        let runner = TestRunner::new(DEVICE_DEBUG)?;

        let gpu_id = runner.gpu_id();
        let gpu_variant = runner.gpu_variant();
        let model =
            model_for_gpu_id(gpu_id, gpu_variant).map_err(io::Error::other)?;

        Ok(RunSingleton { model, runner })
    }

    pub fn get() -> &'static RunSingleton {
        RUN_SINGLETON.get_or_init(|| {
            RunSingleton::create().expect("Failed to create test device")
        })
    }

    fn try_execute(&self, info: InvocationInfo) -> Result<(), HwError> {
        self.runner.run(info)
    }

    fn execute(&self, info: InvocationArgs) {
        self.try_execute(info.0).expect("Error on job submission");
    }
}

fn transmute_slice_to_u8<T: Sized>(data: &[T]) -> &[u8] {
    // SAFETY: we are just transmuting a [u32] to [u8] of same byte-length
    unsafe {
        slice::from_raw_parts(data.as_ptr() as *mut u8, size_of_val(data))
    }
}

fn transmute_mut_slice_to_u8<T: Sized>(data: &mut [T]) -> &mut [u8] {
    // SAFETY: we are just transmuting a [u32] to [u8] of same byte-length.
    // We are also creating a second mutable slice to the same memory, but that
    // one is connected to the lifetime of the original slice by the function
    // signature. While the u8 slice is live, nobody can use the u32 slice.
    unsafe {
        slice::from_raw_parts_mut(
            data.as_mut_ptr() as *mut u8,
            size_of_val(data),
        )
    }
}

#[derive(Clone, Copy)]
pub enum Precision {
    Exact,
    Ulp(u32),
    Abs(f32),
}

fn f32_ulp_dist(a: f32, b: f32) -> u32 {
    let ulp_key = |x: f32| {
        let sign_bit = 1 << 31;
        let bits = x.to_bits();
        if bits & sign_bit != 0 {
            !bits
        } else {
            bits ^ sign_bit
        }
    };

    ulp_key(a).abs_diff(ulp_key(b))
}

fn f16_ulp_dist(a: F16, b: F16) -> u16 {
    let ulp_key = |x: F16| {
        let sign_bit = 1 << 15;
        let bits = x.to_bits();
        if bits & sign_bit != 0 {
            !bits
        } else {
            bits ^ sign_bit
        }
    };

    ulp_key(a).abs_diff(ulp_key(b))
}

fn cmp_eq_f32(real: f32, expected: f32, prec: Precision) -> bool {
    if real.is_nan() || expected.is_nan() {
        return real.is_nan() && expected.is_nan();
    }
    match prec {
        Precision::Exact => real.to_bits() == expected.to_bits(),
        Precision::Ulp(ulps) => {
            if real.to_bits() == expected.to_bits() {
                return true; // Also catches infs
            }
            f32_ulp_dist(real, expected) <= ulps
        }
        Precision::Abs(prec) => (real - expected).abs() < prec,
    }
}

fn cmp_eq_f16(real: F16, expected: F16, prec: Precision) -> bool {
    if real.is_nan() || expected.is_nan() {
        return real.is_nan() && expected.is_nan();
    }
    match prec {
        Precision::Exact => real.to_bits() == expected.to_bits(),
        Precision::Ulp(ulps) => {
            if real.to_bits() == expected.to_bits() {
                return true; // Also catches infs
            }
            u32::from(f16_ulp_dist(real, expected)) <= ulps
        }
        Precision::Abs(prec) => f32::from((real - expected).abs()) < prec,
    }
}

// `assert_f32_eq!(expected, hardware, prec, "msg {..}")`
macro_rules! assert_f32_eq {
    ($expected:expr, $hardware:expr, $prec:expr, $($fmt:tt)+) => {{
        if !cmp_eq_f32($expected, $hardware, $prec) {
            panic!(
                "Test {} failed\nExpected: {}\nHardware: {}",
                format_args!($($fmt)+), $expected, $hardware
            );
        }
    }};
}

fn sample_f32_range(rng: &mut Acorn, range: Range<f32>) -> f32 {
    let t = (rng.get_u32() as f64 / u32::MAX as f64) as f32;
    t * (range.end - range.start) + range.start
}

fn sample_float_special(rng: &mut Acorn, bits: u8) -> u32 {
    let ctrl = rng.get_u32();
    // Throw random floats half of the time to check everything
    if ctrl & 0b01 != 0 {
        return rng.get_u32() & (u32::MAX >> (32 - bits));
    }
    let negate = ctrl & 0b10 != 0;

    let x = rng.get_u32() as usize;
    match bits {
        16 => {
            let v = F16_CURATED[x % F16_CURATED.len()];
            let v = if negate { -v } else { v };
            v.to_bits().into()
        }
        32 => {
            let v = F32_CURATED[x % F32_CURATED.len()];
            let v = if negate { -v } else { v };
            v.to_bits()
        }
        _ => panic!("Unsupported float width"),
    }
}

fn sample_datatype(rng: &mut Acorn, data_type: DataType) -> u32 {
    match data_type {
        DataType::F32 => sample_float_special(rng, 32),
        DataType::F16 => sample_float_special(rng, 16),
        DataType::V2F16 => {
            sample_float_special(rng, 16) << 16 | sample_float_special(rng, 16)
        }
        _ => rng.get_u32(),
    }
}

pub struct TestShaderBuilder<'a> {
    model: &'a dyn Model,
    b: InstrBuilder<'a>,
    info: ShaderInfo,
    ssa_alloc: SSAValueAllocator,
    start_block: BasicBlock,
    data_addr: SrcRef,
    max_data_offset: u16,
}

const WARP_SIZE: u32 = 16;

impl<'a> TestShaderBuilder<'a> {
    pub fn new(model: &'a dyn Model) -> Self {
        let mut label_alloc = LabelAllocator::default();
        let mut ssa_alloc = SSAValueAllocator::default();
        let mut b = SSAInstrBuilder::new(model, &mut ssa_alloc);
        let mut info = ShaderInfo::default();

        // ABI: struct hw_runner_shader_args
        let data_addr_base = FAURef::user_i64(0);
        let data_stride = FAURef::user_i32(2);

        let invoc_id: SSAValue = b.alloc_ssa(32);
        let global_id_reg = model.preload_reg(PreloadReg::GlobalId0).unwrap();
        info.add_preload(&global_id_reg);
        b.push_op(OpRegIn {
            dst: invoc_id.into(),
            dst_type: DataType::I32,
            reg: global_id_reg,
            preload: Some(PreloadInfo {
                set: PreloadRegSet::from_array([PreloadReg::GlobalId0]),
                comp: 0,
            }),
        });

        let data_offset = b.alloc_ref(64);
        b.push_op(OpIMul {
            dst: data_offset.clone().into(),
            dst_type: DataType::U64,
            saturate: false,
            srcs: [
                Src::from(data_stride).swizzle(Swizzle::widen_u32(0)),
                Src::from(invoc_id).swizzle(Swizzle::widen_u32(0)),
            ],
        });

        let data_addr = b.alloc_ref(64);
        b.push_op(OpIAdd {
            dst: data_addr.clone().into(),
            dst_type: DataType::U64,
            saturate: false,
            srcs: [data_addr_base.into(), data_offset.into()],
        });

        let start_block = BasicBlock {
            label: label_alloc.alloc(),
            instrs: b.into_vec(),
        };

        TestShaderBuilder {
            model,
            b: InstrBuilder::new(model),
            info,
            ssa_alloc,
            start_block,
            data_addr: data_addr.into(),
            max_data_offset: 0,
        }
    }

    pub fn ld_test_data(&mut self, offset: u16, bits: u8) -> SSARef {
        let dst = self.alloc_ref(bits.into());

        self.max_data_offset = self.max_data_offset.max(offset);

        self.push_op(OpLoad {
            dst: dst.clone().into(),
            dst_type: DataType::get(1, NumericType::Integer, bits),
            is_tls: false,
            access: MemAccess::None,
            addr: self.data_addr.clone().into(),
            offset: offset.try_into().unwrap(),
        });

        dst
    }

    pub fn st_test_data(&mut self, offset: u16, data: SSARef) {
        self.max_data_offset = self.max_data_offset.max(offset);

        self.push_op(OpStore {
            src_type: DataType::get(1, NumericType::Integer, data.bytes() * 8),
            is_tls: false,
            is_psiz: false,
            access: MemAccess::None,
            data: data.into(),
            addr: self.data_addr.clone().into(),
            offset: offset.try_into().unwrap(),
        });
    }

    fn compile(self) -> CompiledTestCase {
        let Self {
            model,
            mut b,
            info,
            ssa_alloc,
            mut start_block,
            max_data_offset,
            ..
        } = self;

        let exit = b.push_op(OpNop {});
        exit.flow.set_end_shader();

        start_block.instrs.extend(b.into_mapped());
        let mut cfg: CFGBuilder<Label, BasicBlock, FxBuildHasher> =
            CFGBuilder::new();
        cfg.add_node(start_block.label, start_block);

        let mut s = Shader {
            model,
            ssa_alloc,
            phi_alloc: Default::default(),
            blocks: cfg.as_cfg(false),
            info,
            constant_pool: Default::default(),
        };
        s.validate();

        pass!(s.remat_constants());
        pass!(s.widen_alu_ops());
        pass!(s.legalize_src_swizzles());
        pass!(s.opt_copy_prop());
        pass!(s.lower_mkvec_swz());
        pass!(s.opt_dce());
        pass!(s.lower_small_constants());
        pass!(s.legalize());
        pass!(s.assign_registers());
        pass!(s.lower_copy());
        pass!(s.assign_message_slots());

        let bin = model.encode_shader(&s);

        CompiledTestCase {
            code: bin,
            max_data_offset,
            // ABI: we always load the CB0 args at offset 0 for now
            fau_args_offset: 0,
            info: s.info,
        }
    }
}

impl Builder for TestShaderBuilder<'_> {
    fn arch(&self) -> u8 {
        self.b.arch()
    }

    fn model(&self) -> &dyn Model {
        self.b.model()
    }

    fn push_instr(&mut self, instr: Instr) -> &mut Instr {
        self.b.push_instr(instr)
    }
}

impl AllocSSA for TestShaderBuilder<'_> {
    fn alloc_ssa_value(&mut self, bits: u8, is_mem: bool) -> SSAValue {
        self.ssa_alloc.alloc_ssa_value(bits, is_mem)
    }
}

/// Similar to TestShaderBuilder, but useful for creating "raw" shaders, without
/// passing through the compiler, useful if you want to test the hardware rather
/// than the software.  Since the compiler no longer aids us, we need to specify
/// an ABI and do more things manually, for example, load/stores of test data
/// need a 64-bit register to always be alive, placement of this can be
/// configured.  You must also always specify legal non-virtual instructions,
/// and remember to set the correct messages and wait slots.
pub struct RawTestShaderBuilder<'a> {
    model: &'a dyn Model,
    b: InstrBuilder<'a>,
    info: ShaderInfo,
    start_block: BasicBlock,
    data_addr: RegRef,
    max_data_offset: u16,
}

impl<'a> RawTestShaderBuilder<'a> {
    pub fn new(model: &'a dyn Model) -> Self {
        let mut label_alloc = LabelAllocator::default();
        let mut b = InstrBuilder::new(model);
        let mut info = ShaderInfo::default();

        // ABI: struct hw_runner_shader_args
        let data_addr_base = FAURef::user_i64(0);
        let data_stride = FAURef::user_i32(2);

        let invoc_id =
            model.preload_reg(PreloadReg::GlobalId0).unwrap().word(0);
        info.register_preload |= 1 << invoc_id.idx;

        // Those don't intersect, we can use r0 for both
        let data_offset = RegRef::new(0, RegRange::Regs(2));
        let data_addr = RegRef::new(0, RegRange::Regs(2));

        b.push_op(OpIMul {
            dst: data_offset.into(),
            dst_type: DataType::U64,
            saturate: false,
            srcs: [
                Src::from(data_stride).swizzle(Swizzle::widen_u32(0)),
                Src::from(invoc_id).swizzle(Swizzle::widen_u32(0)),
            ],
        });

        b.push_op(OpIAdd {
            dst: data_addr.into(),
            dst_type: DataType::U64,
            saturate: false,
            srcs: [data_addr_base.into(), data_offset.into()],
        });

        let start_block = BasicBlock {
            label: label_alloc.alloc(),
            instrs: b.into_vec(),
        };

        // Some high defaults, shaders will override this if necessary
        info.registers_used = 64;

        RawTestShaderBuilder {
            model,
            b: InstrBuilder::new(model),
            info,
            start_block,
            data_addr,
            max_data_offset: 0,
        }
    }

    #[allow(dead_code)]
    pub fn ld_test_data_to(&mut self, dst: Dst, offset: u16, bits: u8) {
        self.max_data_offset = self.max_data_offset.max(offset);

        let instr = self.push_op(OpLoad {
            dst,
            dst_type: DataType::get(1, NumericType::Integer, bits),
            is_tls: false,
            access: MemAccess::None,
            addr: self.data_addr.into(),
            offset: offset.try_into().unwrap(),
        });
        instr.flow.set_msg_slot_idx(0);
        instr.flow.set_wait_bit(FlowWaitBit::Slot0);
    }

    pub fn st_test_data(&mut self, offset: u16, data: RegRef) {
        self.max_data_offset = self.max_data_offset.max(offset);

        let instr = self.push_op(OpStore {
            src_type: DataType::get(1, NumericType::Integer, data.bytes() * 8),
            is_tls: false,
            is_psiz: false,
            access: MemAccess::None,
            data: data.into(),
            addr: self.data_addr.into(),
            offset: offset.try_into().unwrap(),
        });
        instr.flow.set_msg_slot_idx(0);
        instr.flow.set_wait_bit(FlowWaitBit::Slot0);
    }

    fn compile_with(
        self,
        run_pass: impl FnOnce(&mut Shader),
    ) -> CompiledTestCase {
        let Self {
            model,
            mut b,
            info,
            mut start_block,
            max_data_offset,
            ..
        } = self;

        let exit = b.push_op(OpNop {});
        exit.flow.set_end_shader();

        start_block.instrs.extend(b.into_mapped());
        let mut cfg: CFGBuilder<Label, BasicBlock, FxBuildHasher> =
            CFGBuilder::new();
        cfg.add_node(start_block.label, start_block);

        let mut s = Shader {
            model,
            ssa_alloc: Default::default(),
            phi_alloc: Default::default(),
            blocks: cfg.as_cfg(false),
            info,
            constant_pool: Default::default(),
        };

        if DEBUG.contains(DebugFlags::PRINT) {
            eprintln!("Kraid raw shader before encoding:\n{s}");
        }

        run_pass(&mut s);

        let bin = model.encode_shader(&s);

        CompiledTestCase {
            code: bin,
            max_data_offset,
            // ABI: we always load the CB0 args at offset 0 for now
            fau_args_offset: 0,
            info: s.info,
        }
    }

    fn compile(self) -> CompiledTestCase {
        self.compile_with(|_| {})
    }
}

impl Builder for RawTestShaderBuilder<'_> {
    fn arch(&self) -> u8 {
        self.b.arch()
    }

    fn model(&self) -> &dyn Model {
        self.b.model()
    }

    fn push_instr(&mut self, instr: Instr) -> &mut Instr {
        self.b.push_instr(instr)
    }
}

/// Small helper struct around InvocationInfo to support builder arguments
struct InvocationArgs<'a>(InvocationInfo<'a>);

impl<'a> InvocationArgs<'a> {
    #[allow(dead_code)]
    pub fn with_fau(mut self, fau: &'a [u32]) -> Self {
        self.0.fau = fau;
        self
    }

    pub fn with_buffers(mut self, buffers: &'a mut [&'a mut [u8]]) -> Self {
        self.0.buffers = buffers;
        self
    }
}

struct CompiledTestCase {
    code: Vec<u32>,
    info: ShaderInfo,
    #[allow(dead_code)]
    max_data_offset: u16,
    fau_args_offset: usize,
}

impl CompiledTestCase {
    fn with_data_raw<'a>(
        &'a self,
        data: &'a mut [u8],
        data_stride: u32,
        invocations: u32,
    ) -> InvocationArgs<'a> {
        InvocationArgs(InvocationInfo {
            code: transmute_slice_to_u8(&self.code),
            fau: FAU_ONLY_ARGS,
            fau_args_offset: self.fau_args_offset,
            data,
            data_stride,
            buffers: &mut [],
            register_preload: self.info.register_preload,
            register_count: self.info.registers_used,
            invocations,
        })
    }

    fn with_data<'a, T>(&'a self, data: &'a mut [T]) -> InvocationArgs<'a> {
        let invocations = data.len().try_into().expect("Too many invocations");
        let data_stride = size_of::<T>().try_into().unwrap();
        let data_raw = transmute_mut_slice_to_u8(data);
        self.with_data_raw(data_raw, data_stride, invocations)
    }
}

#[test]
fn test_sanity() {
    let run = RunSingleton::get();
    let b = TestShaderBuilder::new(&*run.model);
    let bin = b.compile();
    let mut data = [0u32; WARP_SIZE as usize];
    let case = bin.with_data(&mut data);
    run.execute(case);
}

#[test]
fn test_copy_single() {
    let run = RunSingleton::get();
    let mut b = TestShaderBuilder::new(&*run.model);
    let data = b.ld_test_data(0, 32);
    b.st_test_data(4, data);

    let bin = b.compile();
    // First, do a small copy (32-bits)
    let mut data = [42, 67, 31, 41, 0, 0, 0, 0];
    let case = bin.with_data_raw(&mut data, 0, WARP_SIZE);

    run.execute(case);
    assert_eq!(&data[0..4], &data[4..8]);
}

#[test]
fn test_copy_8bit() {
    let run = RunSingleton::get();
    let mut b = TestShaderBuilder::new(&*run.model);
    let w = b.ld_test_data(3, 8);
    let z = b.ld_test_data(2, 8);
    let y = b.ld_test_data(1, 8);
    let x = b.ld_test_data(0, 8);
    let data = b.mkvec_v4i8(x.into(), y.into(), z.into(), w.into());
    b.st_test_data(4, data.into());

    let bin = b.compile();
    // First, do a small copy (32-bits)
    let mut data = [42, 67, 31, 41, 0, 0, 0, 0];
    let case = bin.with_data_raw(&mut data, 0, WARP_SIZE);

    run.execute(case);
    assert_eq!(&data[0..4], &data[4..8]);
}

#[test]
fn test_copy_warp() {
    let run = RunSingleton::get();
    let mut b = TestShaderBuilder::new(&*run.model);
    let data = b.ld_test_data(0, 32);
    b.st_test_data(4 * WARP_SIZE as u16, data);

    let bin = b.compile();

    const READ_SIZE: usize = 4 * WARP_SIZE as usize;
    let mut data = [0u8; READ_SIZE * 2];
    for i in 0..READ_SIZE {
        data[i] = (i as u8) * 4 + 1;
    }
    let case = bin.with_data_raw(&mut data, 4, WARP_SIZE);

    run.execute(case);
    assert_eq!(&data[..READ_SIZE], &data[READ_SIZE..]);

    // Now do the same with half the invocations and check that it
    // only copies the first half
    const HALF_READ_SIZE: usize = READ_SIZE / 2;
    for i in 0..data.len() {
        data[i] = (i as u8) * 4 + 1;
    }
    data[READ_SIZE..].fill(0);
    let case = bin.with_data_raw(&mut data, 4, WARP_SIZE / 2);

    run.execute(case);
    assert_eq!(
        &data[..HALF_READ_SIZE],
        &data[READ_SIZE..(READ_SIZE + HALF_READ_SIZE)]
    );
    assert_eq!(&data[(READ_SIZE + HALF_READ_SIZE)..], &[0; HALF_READ_SIZE])
}

#[test]
fn test_copy_large() {
    // Test with more than warp_size invocations
    let run = RunSingleton::get();
    let mut b = TestShaderBuilder::new(&*run.model);
    let data = b.ld_test_data(0, 32);
    b.st_test_data(4 * 2 * WARP_SIZE as u16, data);

    let bin = b.compile();

    const READ_SIZE: usize = 4 * (2 * WARP_SIZE) as usize;
    let mut data = [0u8; READ_SIZE * 2];
    for i in 0..READ_SIZE {
        data[i] = (i as u8) * 4 + 1;
    }
    let case = bin.with_data_raw(&mut data, 4, 2 * WARP_SIZE);

    run.execute(case);
    assert_eq!(&data[..READ_SIZE], &data[READ_SIZE..]);
}

/// Test all lanes for LD_PKA.8/16/32
/// Uncovers LD_PKA.b1 hardware bug
#[test]
fn test_ld_pka() {
    let run = RunSingleton::get();

    const PKA_VAL: [u8; 4] = [0x89, 0xAB, 0xCD, 0xEF];

    let mut failures = Vec::new();
    for range in [
        RegRange::Byte0,
        RegRange::Byte1,
        RegRange::Byte2,
        RegRange::Byte3,
        RegRange::Half0,
        RegRange::Half1,
        RegRange::Regs(1),
    ] {
        let bin = {
            let mut b = RawTestShaderBuilder::new(&*run.model);
            let dst = RegRef::new(2, RegRange::Regs(1));
            let sub_dst = RegRef::new(2, range);

            b.push_op(OpMov {
                dst: dst.into(),
                dst_type: DataType::I32,
                src: SrcRef::Zero.into(),
            });

            let instr = b.push_op(OpLdPka {
                dst: sub_dst.into(),
                dst_type: DataType::i(range.bytes() * 8),
                access: MemAccess::None,
                offset: 0u32.into(),
                handle: 0u32.into(),
            });
            instr.flow.set_msg_slot_idx(0);
            instr.flow.set_wait_bit(FlowWaitBit::Slot0);

            b.st_test_data(0, dst);
            b.compile()
        };

        let mut data = [0u32; 1];
        {
            let mut buf = PKA_VAL;
            let mut buffers: [&mut [u8]; 1] = [&mut buf];
            let case = bin.with_data(&mut data).with_buffers(&mut buffers);
            run.execute(case);
        }

        let off = usize::from(range.byte_offset());
        let len = usize::from(range.bytes());
        let mut expected = [0u8; 4];
        expected[off..(off + len)].copy_from_slice(&PKA_VAL[..len]);
        let got = data[0].to_le_bytes();

        let lane = DstLanes::from(range);
        println!(
            "lane {lane} expected {expected:02x?} got {got:02x?}: {}",
            if got != expected { "FAIL" } else { "PASS" }
        );
        if got != expected {
            failures.push(lane);
        }
    }

    // Expectations:
    // - v9-v15 has a hardware bug where .b1 is treated as .b0, and .b3 as .b2
    let expected = match run.model.arch() {
        9..=15 => failures == vec![DstLanes::B1, DstLanes::B3],
        _ => failures.is_empty(),
    };
    assert!(expected, "LD_PKA assumptions wrong for lanes: {failures:?}");
}

/// Test lower_copy.rs
#[test]
fn test_lower_copy() {
    let run = RunSingleton::get();

    const SOURCE: u32 = 0x89ABCDEF;
    const INIT_DST: u32 = 0x41424344;

    for test_imm in [false, true] {
        for range in [
            RegRange::Byte0,
            RegRange::Byte1,
            RegRange::Byte2,
            RegRange::Byte3,
            RegRange::Half0,
            RegRange::Half1,
            RegRange::Regs(1),
        ] {
            let lanes = DstLanes::from(range);
            let mask = lanes.u32_mask().unwrap();

            let bin = {
                let mut b = RawTestShaderBuilder::new(&*run.model);
                let copy_src = if test_imm {
                    let start = usize::from(range.byte_offset() * 8);
                    let end = start + usize::from(range.bytes() * 8);
                    let imm = SOURCE.get_bit_range_u64(start..end) as u32;
                    // The width picks the replicating swizzle lower_copy folds.
                    match range.bytes() {
                        1 => Src::from(imm as u8),
                        2 => Src::from(imm as u16),
                        _ => Src::from(imm),
                    }
                } else {
                    let src = RegRef::new(2, RegRange::Regs(1));
                    b.ld_test_data_to(src.into(), 0, 32);
                    Src::from(src).swizzle(range.into())
                };

                let copy_reg = RegRef::new(3, RegRange::Regs(1));
                b.ld_test_data_to(copy_reg.into(), 4, 32);

                b.push_op(OpCopy {
                    dst: RegRef::new(3, range).into(),
                    dst_type: DataType::i(range.bytes() * 8),
                    src: copy_src,
                });

                b.st_test_data(8, copy_reg);
                b.compile_with(|s| s.lower_copy())
            };

            let mut data = [SOURCE, INIT_DST, 0xDEFDEFDE];
            let case = bin.with_data(&mut data);
            run.execute(case);

            let expected = (INIT_DST & !mask) | (SOURCE & mask);
            let got = data[2];

            assert_eq!(
                expected, got,
                "lane {lanes} expected {expected:08x} got {got:08x}"
            );
        }
    }
}

fn parse_folded(folded: &mut [u64], words: &[u32], types: DataTypeIter) {
    let mut offset = 0;
    for (comp, dtype) in folded.iter_mut().zip(types) {
        match dtype.total_bits() {
            8 | 16 | 32 => {
                *comp = words[offset] as u64;
                offset += 1;
            }
            64 => {
                *comp = words[offset] as u64 | (words[offset + 1] as u64) << 32;
                offset += 2;
            }
            _ => panic!("Invalid data size"),
        }
    }
    assert_eq!(offset, words.len());
}

fn folded_eq(
    a: &[u64],
    b: &[u64],
    types: DataTypeIter,
    precision: Precision,
) -> bool {
    a.iter()
        .zip(b.iter())
        .zip(types)
        .all(|((a, b), dtype)| match dtype {
            DataType::F32 => {
                let a = f32::from_bits(*a as u32);
                let b = f32::from_bits(*b as u32);
                cmp_eq_f32(a, b, precision)
            }
            DataType::F16 => {
                let a = F16::from_bits(*a as u16);
                let b = F16::from_bits(*b as u16);
                cmp_eq_f16(a, b, precision)
            }
            DataType::V2F16 => {
                let a0 = F16::from_bits(*a as u16);
                let b0 = F16::from_bits(*b as u16);
                let a1 = F16::from_bits((*a >> 16) as u16);
                let b1 = F16::from_bits((*b >> 16) as u16);
                cmp_eq_f16(a0, b0, precision) && cmp_eq_f16(a1, b1, precision)
            }
            _ => a == b,
        })
}

fn format_folded(data: &[u64], types: DataTypeIter) -> String {
    use std::fmt::Write;
    let mut s = "[".to_string();
    for (i, (data, dtype)) in data.iter().zip(types).enumerate() {
        if i != 0 {
            write!(s, ", ").unwrap();
        }
        match dtype {
            DataType::F32 => {
                let f = f32::from_bits(*data as u32);
                write!(s, "{f:?} (0x{:08x})", f.to_bits()).unwrap();
            }
            DataType::F16 => {
                let f = F16::from_bits(*data as u16);
                write!(s, "{f:?} (0x{:04x})", f.to_bits()).unwrap();
            }
            DataType::V2F16 => {
                let x = F16::from_bits(*data as u16);
                let y = F16::from_bits((*data >> 16) as u16);
                write!(
                    s,
                    "[{x:?} (0x{:04x}), {y:?} (0x{:04x})]",
                    x.to_bits(),
                    y.to_bits()
                )
                .unwrap();
            }
            _ => {
                write!(s, "{data}").unwrap();
            }
        }
    }
    s += "]";
    s
}

pub fn test_foldable_op_with(
    mut op: impl Foldable + Clone + Into<Op> + fmt::Debug,
    precision: Precision,
    mut rand_u32: impl FnMut(usize, DataType) -> u32,
) {
    let run = RunSingleton::get();
    let mut b = TestShaderBuilder::new(&*run.model);

    let mut offset_words = 0u16;
    let mut word_types = Vec::new();

    for (src, src_type) in op.srcs_types_mut() {
        let read_bits = src_type.total_bits();
        let words = read_bits.div_ceil(32);
        let data = b.ld_test_data(offset_words * 4, words * 32);
        offset_words += u16::from(words);

        let swiz_src = if src.swizzle.is_none() {
            // We always load words.  If there's no swizzle, we need to at
            // least trim it down to fit inside the source type.
            let src_type_swizzle = match read_bits {
                8 => Swizzle::replicate_byte(0),
                16 => Swizzle::replicate_half(0),
                _ => Swizzle::NONE,
            };
            Src::from(data).swizzle(src_type_swizzle)
        } else {
            Src::from(data).swizzle(src.swizzle)
        };
        src.src_ref = swiz_src.src_ref;
        src.swizzle = swiz_src.swizzle;

        word_types.extend(iter::repeat_n(src_type, words as usize));
    }
    let src_words = usize::from(offset_words);

    let mut fold_src = vec![0u64; op.srcs().len()];
    let mut fold_dst = vec![0u64; op.dsts().len()];
    for (dst, dst_type) in op.dsts_types_mut() {
        let write_bits = dst_type.total_bits();
        dst.dst_ref = b.alloc_ref(write_bits.into()).into();
        dst.lanes = match (dst.lanes, write_bits) {
            (DstLanes::None | DstLanes::All, 8) => DstLanes::AnyB,
            (DstLanes::None | DstLanes::All, 16) => DstLanes::AnyH,
            (DstLanes::None | DstLanes::All, _) => DstLanes::All,
            (lanes, _) => lanes,
        };
    }

    b.push_op(op.clone());
    let op = op; // Remove mutability

    for dst in op.dsts() {
        let DstRef::SSA(ssa) = &dst.dst_ref else {
            unreachable!(); // We set it as SSA before
        };
        b.st_test_data(offset_words * 4, ssa.clone());
        offset_words += u16::from(ssa.bytes().div_ceil(4));
    }
    let total_words = usize::from(offset_words);
    let dst_words = total_words - src_words;

    let bin = b.compile();

    // We're throwing random data at it here so the idea is that the number
    // of test cases we need to get good coverage is relative to the square
    // of the number of components.  For a big op with 3 srcs, this is going
    // to give us 2500 iterations. (copied from NAK)
    let invocations = src_words * src_words * 100;
    let mut data = Vec::with_capacity(invocations * (src_words + dst_words));

    assert!(src_words == word_types.len());
    for _ in 0..invocations {
        data.extend(word_types.iter().enumerate().map(|(i, &wtype)| {
            let bits = wtype.total_bits().min(32) as u32;
            rand_u32(i, wtype) & (u32::MAX >> (u32::BITS - bits))
        }));
        data.extend(iter::repeat_n(0, dst_words));
    }
    assert!(data.len() == invocations * (src_words + dst_words));

    let data_bytes = transmute_mut_slice_to_u8(&mut data);
    let case = bin.with_data_raw(
        data_bytes,
        4 * total_words as u32,
        invocations.try_into().unwrap(),
    );

    run.execute(case);

    let mut hw_dst = fold_dst.clone();
    for invoc_id in 0..invocations {
        let data_off = invoc_id * total_words;
        let data = &data[data_off..(data_off + total_words)];

        parse_folded(&mut fold_src, &data[..src_words], op.src_types());
        parse_folded(&mut hw_dst, &data[src_words..], op.dst_types());

        fold_dst.fill(0);
        let mut fold = FoldData {
            dsts: &mut fold_dst,
            srcs: &fold_src,
            op: &op,
        };
        op.fold(&*run.model, &mut fold);

        if !folded_eq(&hw_dst, &fold_dst, op.dst_types(), precision) {
            let input_s = format_folded(&fold_src, op.src_types());
            let hw_out_s = format_folded(&hw_dst, op.dst_types());
            let fold_out_s = format_folded(&fold_dst, op.dst_types());
            eprintln!("Foldable test data mismatch for {op:?}:");
            eprintln!("| Input:    {input_s}");
            eprintln!("| Hardware: {hw_out_s}");
            eprintln!("| Folded:   {fold_out_s}");
            panic!("Folding test data mismatch");
        }
    }
}

pub fn test_foldable_op(
    op: impl Foldable + Clone + Into<Op> + fmt::Debug,
    precision: Precision,
) {
    let mut a = Acorn::new();
    let rand_gen = |_, dt| sample_datatype(&mut a, dt);
    test_foldable_op_with(op, precision, rand_gen);
}

#[test]
fn test_op_bitrev() {
    let op = OpBitRev {
        dst: DstRef::None.into(),
        src: 0_u32.into(),
    };

    test_foldable_op(op, Precision::Exact);
}

#[test]
fn test_op_clz() {
    const DATA_TYPES: &[DataType] =
        &[DataType::U32, DataType::V2U16, DataType::V4U8];

    // The .mask modifier only outputs all-bits if the input value is 0
    // Test some edge-cases then test random data
    let mut edge_cases: Vec<u32> = vec![
        0x0000_0000, // CLZ -> 32, .mask -> 0xffffffff
        0xffff_ffff,
        0x8000_0000,
        0x7fff_ffff,
        0x0000_0002,
        0x0000_0003,
    ];
    for n in 0..32 {
        edge_cases.push(1u32 << n);
        edge_cases.push(u32::MAX >> n);
    }

    for &src_type in DATA_TYPES {
        for mask in [false, true] {
            let op = OpClz {
                dst: DstRef::None.into(),
                src_type,
                mask,
                src: 0_u32.into(),
            };

            let mut a = Acorn::new();
            let mut idx = 0usize;
            test_foldable_op_with(op, Precision::Exact, |_, _| {
                let v = edge_cases
                    .get(idx)
                    .copied()
                    .unwrap_or_else(|| a.get_u32() >> (a.get_u32() % 32));
                idx += 1;
                v
            });
        }
    }
}

#[test]
fn test_op_csel() {
    const DATA_TYPES: &[DataType] = &[
        DataType::S32,
        DataType::U32,
        DataType::F32,
        DataType::V2S16,
        DataType::V2U16,
        DataType::V2F16,
    ];

    const CMP_OPS: &[CmpOp] = &[
        CmpOp::Eq,
        CmpOp::Gt,
        CmpOp::Ge,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::GtLt,
        CmpOp::Total,
    ];

    for &cmp_type in DATA_TYPES {
        for &cmp_op in CMP_OPS {
            if cmp_type.num_type() != NumericType::Float
                && matches!(cmp_op, CmpOp::GtLt | CmpOp::Total)
            {
                continue;
            }

            let op = OpCSel {
                dst: DstRef::None.into(),
                cmp_type,
                cmp_op,
                cmp_srcs: [0_u32.into(), 0_u32.into()],
                sel_srcs: [0_u32.into(), 0_u32.into()],
            };
            test_foldable_op(op, Precision::Exact);
        }
    }
}

#[test]
fn test_op_cubefaceidx() {
    let op = OpCubeFaceIdx {
        dst: DstRef::None.into(),
        coords: [0_u32.into(), 0_u32.into(), 0_u32.into()],
    };

    test_foldable_op(op, Precision::Exact);
}

#[test]
fn test_op_cubefacemax() {
    let op = OpCubeFaceMax {
        dst: DstRef::None.into(),
        coords: [0_u32.into(), 0_u32.into(), 0_u32.into()],
    };

    test_foldable_op(op, Precision::Ulp(1));
}

#[test]
fn test_op_f16_to_f32() {
    let op = OpF16ToF32 {
        dst: DstRef::None.into(),
        src: 0_u32.into(),
    };
    test_foldable_op(op, Precision::Ulp(0));
}

#[test]
fn test_op_f32_to_f16() {
    let run = RunSingleton::get();

    // F32_TO_F16 only available from v11
    if run.model.arch() < 11 {
        return;
    }

    const ROUND_MODES: &[FRound] = &[
        FRound::NearestEven,
        FRound::Up,
        FRound::Down,
        FRound::TowardsZero,
    ];

    const CLAMP_MODES: &[FClamp] = &[
        FClamp::None,
        FClamp::ZeroToInf,
        FClamp::NegOneToOne,
        FClamp::ZeroToOne,
    ];
    for &round in ROUND_MODES {
        for &clamp in CLAMP_MODES {
            let op = OpF32ToF16 {
                dst: DstRef::None.into(),
                src: 0_u32.into(),
                round,
                clamp,
            };
            test_foldable_op(op, Precision::Ulp(0));
        }
    }
}

#[test]
fn test_op_f32_to_i32() {
    const DATA_TYPES: &[DataType] = &[DataType::S32, DataType::U32];

    const ROUND_MODES: &[FRound] = &[
        FRound::NearestEven,
        FRound::Up,
        FRound::Down,
        FRound::TowardsZero,
    ];

    for &dst_type in DATA_TYPES {
        for &round in ROUND_MODES {
            let op = OpF32ToI32 {
                dst: DstRef::None.into(),
                src: 0_u32.into(),
                round,
                dst_type,
            };
            test_foldable_op(op, Precision::Ulp(0));
        }
    }
}

#[test]
fn test_op_fadd() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::V2F16];

    const ROUND_MODES: &[FRound] = &[
        FRound::NearestEven,
        FRound::Up,
        FRound::Down,
        FRound::TowardsZero,
    ];

    const CLAMP_MODES: &[FClamp] = &[
        FClamp::None,
        FClamp::ZeroToInf,
        FClamp::NegOneToOne,
        FClamp::ZeroToOne,
    ];

    for &dst_type in DATA_TYPES {
        for &round in ROUND_MODES {
            for &clamp in CLAMP_MODES {
                let op = OpFAdd {
                    dst: DstRef::None.into(),
                    dst_type,
                    round,
                    clamp,
                    srcs: [0_u32.into(), 0_u32.into()],
                };
                // FRound is not emulated correctly
                let ulps = if round == FRound::NearestEven { 0 } else { 1 };
                test_foldable_op(op, Precision::Ulp(ulps));
            }
        }
    }
}

#[test]
fn test_op_fadd_lscale() {
    const ROUND_MODES: &[FRound] = &[
        FRound::NearestEven,
        FRound::Up,
        FRound::Down,
        FRound::TowardsZero,
    ];

    const CLAMP_MODES: &[FClamp] = &[
        FClamp::None,
        FClamp::ZeroToInf,
        FClamp::NegOneToOne,
        FClamp::ZeroToOne,
    ];

    for &round in ROUND_MODES {
        for &clamp in CLAMP_MODES {
            let op = OpFAddLScale {
                dst: DstRef::None.into(),
                round,
                clamp,
                srcs: [0_u32.into(), 0_u32.into()],
            };
            // FRound is not emulated correctly
            let ulps = if round == FRound::NearestEven { 0 } else { 1 };
            test_foldable_op(op, Precision::Ulp(ulps));
        }
    }
}

#[test]
fn test_op_fcmp() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::V2F16];

    const CMP_OPS: &[CmpOp] = &[
        CmpOp::Eq,
        CmpOp::Gt,
        CmpOp::Ge,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
        CmpOp::GtLt,
        CmpOp::Total,
    ];

    const ACCUM_OPS: &[CmpAccumOp] =
        &[CmpAccumOp::None, CmpAccumOp::And, CmpAccumOp::Or];

    const RES_TYPES: &[CmpResultType] =
        &[CmpResultType::I1, CmpResultType::F1, CmpResultType::M1];

    let mut a = Acorn::new();
    for &src_type in DATA_TYPES {
        for &cmp_op in CMP_OPS {
            for &accum_op in ACCUM_OPS {
                for &res_type in RES_TYPES {
                    let op = OpFCmp {
                        dst: DstRef::None.into(),
                        src_type,
                        res_type,
                        cmp_op,
                        srcs: [0_u32.into(), 0_u32.into()],
                        accum: 0_u32.into(),
                        accum_op,
                    };
                    // Accum is always treated as a bool so let's use 0-1
                    // (otherwise it would always be true)
                    let rng = |i, dt| match i {
                        2 => a.get_u32() % 2,
                        _ => sample_datatype(&mut a, dt),
                    };
                    test_foldable_op_with(op, Precision::Exact, rng);
                }
            }
        }
    }
}

#[test]
fn test_op_flush() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::V2F16];

    const NAN_MODES: &[FlushNanMode] = &[
        FlushNanMode::None,
        FlushNanMode::FlushNan,
        FlushNanMode::QuietNan,
    ];

    for &src_type in DATA_TYPES {
        for ftz in [false, true] {
            for flush_inf in [false, true] {
                for &flush_nan in NAN_MODES {
                    let op = OpFlush {
                        dst: DstRef::None.into(),
                        src_type,
                        src: 0_u32.into(),
                        ftz,
                        flush_inf,
                        flush_nan,
                    };
                    test_foldable_op(op, Precision::Exact);
                }
            }
        }
    }
}

#[test]
fn test_op_fma() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::V2F16];

    const ROUND_MODES: &[FRound] = &[
        FRound::NearestEven,
        FRound::Up,
        FRound::Down,
        FRound::TowardsZero,
    ];

    const CLAMP_MODES: &[FClamp] = &[
        FClamp::None,
        FClamp::ZeroToInf,
        FClamp::NegOneToOne,
        FClamp::ZeroToOne,
    ];

    for &dst_type in DATA_TYPES {
        for &round in ROUND_MODES {
            for &clamp in CLAMP_MODES {
                let op = OpFma {
                    dst: DstRef::None.into(),
                    dst_type,
                    round,
                    clamp,
                    srcs: [0_u32.into(), 0_u32.into(), 0_u32.into()],
                };
                // FRound is not emulated correctly, F16 has double-rounding issues
                test_foldable_op(op, Precision::Ulp(1));
            }
        }
    }
}

#[test]
fn test_op_fmin() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::V2F16];

    const CLAMP_MODES: &[FClamp] = &[
        FClamp::None,
        FClamp::ZeroToInf,
        FClamp::NegOneToOne,
        FClamp::ZeroToOne,
    ];

    for &dst_type in DATA_TYPES {
        for propagate_nan in [false, true] {
            for &clamp in CLAMP_MODES {
                let op = OpFMin {
                    dst: DstRef::None.into(),
                    dst_type,
                    clamp,
                    propagate_nan,
                    srcs: [0_u32.into(), 0_u32.into()],
                };
                test_foldable_op(op, Precision::Ulp(0));
            }
        }
    }
}

#[test]
fn test_op_fmax() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::V2F16];

    const CLAMP_MODES: &[FClamp] = &[
        FClamp::None,
        FClamp::ZeroToInf,
        FClamp::NegOneToOne,
        FClamp::ZeroToOne,
    ];

    for &dst_type in DATA_TYPES {
        for propagate_nan in [false, true] {
            for &clamp in CLAMP_MODES {
                let op = OpFMax {
                    dst: DstRef::None.into(),
                    dst_type,
                    clamp,
                    propagate_nan,
                    srcs: [0_u32.into(), 0_u32.into()],
                };
                test_foldable_op(op, Precision::Ulp(0));
            }
        }
    }
}

#[test]
fn test_op_frcp() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::F16];

    for &dst_type in DATA_TYPES {
        let op = OpFRcp {
            dst: DstRef::None.into(),
            dst_type,
            src: 0_u32.into(),
        };
        // The docs only guarantee a maximum error strictly below 1 ulp
        test_foldable_op(op, Precision::Ulp(1));
    }
}

#[test]
fn test_op_fround() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::V2F16];

    const ROUND_MODES: &[FRound] = &[
        FRound::NearestEven,
        FRound::Up,
        FRound::Down,
        FRound::TowardsZero,
        FRound::NearestValue,
    ];

    let run = RunSingleton::get();
    for &src_type in DATA_TYPES {
        // 16-bits are only available in arch <= 10
        if src_type.bits() == 16 && run.model.arch() > 10 {
            continue;
        }
        for &round in ROUND_MODES {
            let op = OpFRound {
                dst: DstRef::None.into(),
                src_type,
                round,
                src: 0_u32.into(),
            };
            test_foldable_op(op, Precision::Ulp(0));
        }
    }
}

#[test]
fn test_op_frsq() {
    const DATA_TYPES: &[DataType] = &[DataType::F32, DataType::F16];

    for &dst_type in DATA_TYPES {
        let op = OpFRsq {
            dst: DstRef::None.into(),
            dst_type,
            src: 0_u32.into(),
        };
        // Hardware has 1 ULP precision, but we have double-rounding issues
        test_foldable_op(op, Precision::Ulp(2));
    }
}

#[test]
fn test_op_hadd() {
    let run = RunSingleton::get();

    // HADD was removed in v11
    if run.model.arch() > 10 {
        return;
    }

    const DATA_TYPES: &[DataType] = &[
        DataType::V4S8,
        DataType::V4U8,
        DataType::V2S16,
        DataType::V2U16,
        DataType::S32,
        DataType::U32,
    ];

    const WIDENS: &[AsmSwizzleWiden] = &[
        AsmSwizzleWiden::None,
        AsmSwizzleWiden::B00,
        AsmSwizzleWiden::B02,
        AsmSwizzleWiden::B20,
        AsmSwizzleWiden::H00,
        AsmSwizzleWiden::H10,
        AsmSwizzleWiden::H0,
        AsmSwizzleWiden::H1,
    ];

    for &dst_type in DATA_TYPES {
        for widen in WIDENS {
            let Some(src0_swizzle) = widen.to_swizzle(dst_type) else {
                continue;
            };
            for round_up in [false, true] {
                let op = OpHAdd {
                    dst: DstRef::None.into(),
                    dst_type,
                    round_up,
                    srcs: [
                        Src::from(0_u32).swizzle(src0_swizzle),
                        0_u32.into(),
                    ],
                };
                test_foldable_op(op, Precision::Exact);
            }
        }
    }
}

#[test]
fn test_op_iabs() {
    const DATA_TYPES: &[DataType] =
        &[DataType::V4S8, DataType::V2S16, DataType::S32];

    const WIDENS: &[AsmSwizzleWiden] = &[
        AsmSwizzleWiden::None,
        AsmSwizzleWiden::H0,
        AsmSwizzleWiden::B0,
        AsmSwizzleWiden::B2,
    ];

    let run = RunSingleton::get();
    for &dst_type in DATA_TYPES {
        // 8-bits are only supported in arch <= v10
        if dst_type.bits() == 8 && run.model.arch() > 10 {
            continue;
        }
        for widen in WIDENS {
            let Some(src0_swizzle) = widen.to_swizzle(dst_type) else {
                continue;
            };

            let op = OpIAbs {
                dst: DstRef::None.into(),
                src: Src::from(0_u32).swizzle(src0_swizzle),
                dst_type,
            };
            test_foldable_op(op, Precision::Exact);
        }
    }
}

#[test]
fn test_op_iadd() {
    const DATA_TYPES: &[DataType] = &[
        DataType::V4S8,
        DataType::V4U8,
        DataType::V2S16,
        DataType::V2U16,
        DataType::S32,
        DataType::U32,
        DataType::S64,
        DataType::U64,
    ];

    const WIDENS: &[AsmSwizzleWiden] = &[
        AsmSwizzleWiden::None,
        AsmSwizzleWiden::B00,
        AsmSwizzleWiden::B02,
        AsmSwizzleWiden::B20,
        AsmSwizzleWiden::H00,
        AsmSwizzleWiden::H10,
        AsmSwizzleWiden::H0,
        AsmSwizzleWiden::H1,
        AsmSwizzleWiden::W0,
        // w1 is not supported on src0
    ];

    let run = RunSingleton::get();
    for &dst_type in DATA_TYPES {
        // 8-bits are only supported in arch <= v10
        if dst_type.bits() == 8 && run.model.arch() > 10 {
            continue;
        }
        for &widen in WIDENS {
            let Some(src0_swizzle) = widen.to_swizzle(dst_type) else {
                continue;
            };
            for saturate in [false, true] {
                // Not supported by hw
                if saturate && dst_type.bits() == 64 {
                    continue;
                }

                let op = OpIAdd {
                    dst: DstRef::None.into(),
                    srcs: [
                        Src::from(0_u32).swizzle(src0_swizzle),
                        0_u32.into(),
                    ],
                    dst_type,
                    saturate,
                };
                test_foldable_op(op, Precision::Exact);
            }
        }
    }
}

#[test]
fn test_op_icmp() {
    const DATA_TYPES: &[DataType] = &[
        DataType::V4S8,
        DataType::V4U8,
        DataType::V2S16,
        DataType::V2U16,
        DataType::S32,
        DataType::U32,
    ];

    const CMP_OPS: &[CmpOp] = &[
        CmpOp::Eq,
        CmpOp::Gt,
        CmpOp::Ge,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
    ];

    const ACCUM_OPS: &[CmpAccumOp] =
        &[CmpAccumOp::None, CmpAccumOp::And, CmpAccumOp::Or];

    const RES_TYPES: &[CmpResultType] =
        &[CmpResultType::I1, CmpResultType::F1, CmpResultType::M1];

    let mut a = Acorn::new();
    let run = RunSingleton::get();
    for &src_type in DATA_TYPES {
        // 8-bits are only supported in arch <= v10
        if src_type.bits() == 8 && run.model.arch() > 10 {
            continue;
        }
        for &cmp_op in CMP_OPS {
            for &accum_op in ACCUM_OPS {
                for &res_type in RES_TYPES {
                    // No 8-bit floats
                    if src_type.bits() == 8 && res_type == CmpResultType::F1 {
                        continue;
                    }

                    let op = OpICmp {
                        dst: DstRef::None.into(),
                        src_type,
                        res_type,
                        cmp_op,
                        srcs: [0_u32.into(), 0_u32.into()],
                        accum: 0_u32.into(),
                        accum_op,
                    };
                    // Accum is always treated as a bool so let's use 0-1
                    // (otherwise it would always be true)
                    let rng = |i, dt| match i {
                        2 => a.get_u32() % 2,
                        _ => sample_datatype(&mut a, dt),
                    };
                    test_foldable_op_with(op, Precision::Exact, rng);
                }
            }
        }
    }
}

#[test]
fn test_op_icmp_multi() {
    const DATA_TYPES: &[DataType] = &[DataType::S32, DataType::U32];

    const CMP_OPS: &[CmpOp] = &[
        CmpOp::Eq,
        CmpOp::Gt,
        CmpOp::Ge,
        CmpOp::Ne,
        CmpOp::Lt,
        CmpOp::Le,
    ];

    const RES_TYPES: &[CmpResultType] = &[
        CmpResultType::I1,
        CmpResultType::F1,
        CmpResultType::M1,
        CmpResultType::C,
    ];

    let mut a = Acorn::new();
    for &src_type in DATA_TYPES {
        for &cmp_op in CMP_OPS {
            for &res_type in RES_TYPES {
                if res_type == CmpResultType::C && src_type == DataType::S32 {
                    continue;
                }

                let op = OpICmpMulti {
                    dst: DstRef::None.into(),
                    src_type,
                    res_type,
                    cmp_op,
                    srcs: [0_u32.into(), 0_u32.into()],
                    accum: 0_u32.into(),
                };
                // Accum should be 0, 1, or -1
                let rng = |i, dt| match i {
                    2 => (a.get_u32() % 3) - 1,
                    _ => sample_datatype(&mut a, dt),
                };
                test_foldable_op_with(op, Precision::Exact, rng);
            }
        }
    }
}

#[test]
fn test_op_idpadd() {
    let model = RunSingleton::get().model.as_ref();

    const SRC_TYPES: &[DataType] = &[DataType::V4S8, DataType::V4U8];

    for saturate in [false, true] {
        let op = OpIDpAdd {
            dst: DstRef::None.into(),
            dst_type: DataType::U32,
            saturate,
            src_types: [DataType::V4U8; 2],
            srcs: [0_u32.into(), 0_u32.into()],
            accum: 0_u32.into(),
        };
        test_foldable_op(op, Precision::Exact);
    }

    for &src0_type in SRC_TYPES {
        for &src1_type in SRC_TYPES {
            let src_types = [src0_type, src1_type];
            if model.arch() < 14 && src_types != [DataType::V4S8; 2] {
                continue;
            }

            for saturate in [false, true] {
                let op = OpIDpAdd {
                    dst: DstRef::None.into(),
                    dst_type: DataType::S32,
                    saturate,
                    src_types,
                    srcs: [0_u32.into(), 0_u32.into()],
                    accum: 0_u32.into(),
                };
                test_foldable_op(op, Precision::Exact);
            }
        }
    }
}

#[test]
fn test_op_imul() {
    const DATA_TYPES: &[DataType] = &[
        DataType::V2S16,
        DataType::V2U16,
        DataType::S32,
        DataType::U32,
        DataType::S64,
        DataType::U64,
    ];

    const WIDENS: &[AsmSwizzleWiden] = &[
        AsmSwizzleWiden::None,
        AsmSwizzleWiden::B00,
        AsmSwizzleWiden::B02,
        AsmSwizzleWiden::B20,
        AsmSwizzleWiden::H00,
        AsmSwizzleWiden::H10,
        AsmSwizzleWiden::H0,
        AsmSwizzleWiden::H1,
        AsmSwizzleWiden::W0,
        // w1 is not supported
    ];

    for &dst_type in DATA_TYPES {
        for &widen in WIDENS {
            let Some(src0_swizzle) = widen.to_swizzle(dst_type) else {
                continue;
            };
            for saturate in [false, true] {
                let mut src1 = Src::from(0_u32);
                // IMUL.64 is a 32x32->64 multiply, None swizzle is not supported
                if dst_type.bits() == 64 {
                    if saturate || widen == AsmSwizzleWiden::None {
                        continue;
                    }
                    src1 = src1.swizzle(Swizzle::widen_wx(dst_type, 0));
                }

                let op = OpIMul {
                    dst: DstRef::None.into(),
                    srcs: [Src::from(0_u32).swizzle(src0_swizzle), src1],
                    dst_type,
                    saturate,
                };
                test_foldable_op(op, Precision::Exact);
            }
        }
    }
}

#[test]
fn test_op_isub() {
    const DATA_TYPES: &[DataType] = &[
        DataType::V2S16,
        DataType::V2U16,
        DataType::S32,
        DataType::U32,
        DataType::S64,
        DataType::U64,
    ];

    const WIDENS: &[AsmSwizzleWiden] = &[
        AsmSwizzleWiden::None,
        AsmSwizzleWiden::B00,
        AsmSwizzleWiden::B02,
        AsmSwizzleWiden::B20,
        AsmSwizzleWiden::H00,
        AsmSwizzleWiden::H10,
        AsmSwizzleWiden::H0,
        AsmSwizzleWiden::H1,
        AsmSwizzleWiden::W0,
        // w1 is not supported on src0
    ];

    for &dst_type in DATA_TYPES {
        for &widen in WIDENS {
            let Some(src0_swizzle) = widen.to_swizzle(dst_type) else {
                continue;
            };
            for saturate in [false, true] {
                // Not supported by hw
                if saturate && dst_type.bits() == 64 {
                    continue;
                }

                let op = OpISub {
                    dst: DstRef::None.into(),
                    srcs: [
                        Src::from(0_u32).swizzle(src0_swizzle),
                        0_u32.into(),
                    ],
                    dst_type,
                    saturate,
                };
                test_foldable_op(op, Precision::Exact);
            }
        }
    }
}

#[test]
fn test_op_mux() {
    const DATA_TYPES: &[DataType] =
        &[DataType::V4I8, DataType::V2I16, DataType::I32];
    const MUX_OPS: &[MuxOp] =
        &[MuxOp::Neg, MuxOp::IntZero, MuxOp::FpZero, MuxOp::Bit];

    for &dst_type in DATA_TYPES {
        for &mux_op in MUX_OPS {
            if mux_op == MuxOp::FpZero && dst_type != DataType::I32 {
                continue;
            }
            let op = OpMux {
                dst: DstRef::None.into(),
                dst_type,
                mux_op,
                src0: 0_u32.into(),
                src1: 0_u32.into(),
                sel: 0_u32.into(),
            };
            test_foldable_op(op, Precision::Exact);
        }
    }
}

#[test]
fn test_op_popcount() {
    let op = OpPopCount {
        dst: DstRef::None.into(),
        src: 0_u32.into(),
    };

    test_foldable_op(op, Precision::Exact);
}

#[test]
fn test_op_shift_lop() {
    const DATA_TYPES: &[DataType] = &[
        DataType::V4U8,
        DataType::V2U16,
        DataType::U32,
        DataType::U64,
    ];

    const SHIFT_OPS: &[ShiftOp] = &[
        ShiftOp::None,
        ShiftOp::LShift,
        ShiftOp::RShift,
        ShiftOp::ARShift,
        ShiftOp::RRot,
        ShiftOp::LRot,
    ];

    const LOGIC_OPS: &[LogicOp] =
        &[LogicOp::None, LogicOp::Or, LogicOp::And, LogicOp::Xor];

    const WIDENS: &[AsmSwizzleWiden] = &[
        AsmSwizzleWiden::None,
        AsmSwizzleWiden::B0,
        AsmSwizzleWiden::H0,
    ];

    for &dst_type in DATA_TYPES {
        for widen in WIDENS {
            let Some(src0_swizzle) = widen.to_swizzle(dst_type) else {
                continue;
            };
            for &shift_op in SHIFT_OPS {
                for &logic_op in LOGIC_OPS {
                    for not_result in [false, true] {
                        let op = OpShiftLop {
                            dst: DstRef::None.into(),
                            dst_type,
                            shift_op,
                            logic_op,
                            not_result,
                            src0: Src::from(0_u32).swizzle(src0_swizzle),
                            shift: 0_u32.into(),
                            src2: 0_u32.into(),
                        };
                        test_foldable_op(op, Precision::Exact);
                    }
                }
            }
        }
    }
}

fn sample_swizzle(rng: &mut Acorn, src_type: DataType) -> Option<Swizzle> {
    let sample_swizzle_byte = |rng: &mut Acorn| match rng.get_u32() % 9 {
        0 => SwizzleByte::Zero,
        n @ 1..=4 => SwizzleByte::byte((n - 1) as u8),
        n => SwizzleByte::sign((n - 5) as u8),
    };

    let swizzle = match src_type.total_bits() {
        8 => Swizzle::from_swizzle_bytes([sample_swizzle_byte(rng); 4])?,
        16 => {
            let [b0, b1] = std::array::from_fn(|_| sample_swizzle_byte(rng));
            Swizzle::from_swizzle_bytes([b0, b1, b0, b1])?
        }
        32 => Swizzle::from_swizzle_bytes(std::array::from_fn(|_| {
            sample_swizzle_byte(rng)
        }))?,
        64 => {
            if rng.get_u32() % 2 == 0 {
                // Word swizzles
                Swizzle::from_swizzle_words(std::array::from_fn(|_| {
                    let i =
                        rng.get_u32() as usize % SwizzleWord::VARIANTS.len();
                    SwizzleWord::VARIANTS.iter().nth(i).unwrap()
                }))
            } else {
                // Byte swizzles
                Swizzle::from_swizzle_bytes(std::array::from_fn(|_| {
                    sample_swizzle_byte(rng)
                }))?
            }
        }
        _ => panic!("Invalid src_type"),
    };

    // Swizzle::ZERO is never legal
    Some(swizzle).filter(|x| *x != Swizzle::ZERO)
}

#[test]
fn test_op_swz() {
    const DATA_TYPES: &[DataType] = &[
        DataType::I8,
        DataType::V2I8,
        DataType::V4I8,
        DataType::F16,
        DataType::I16,
        DataType::V2F16,
        DataType::V2I16,
        DataType::F32,
        DataType::I32,
        DataType::I64,
        DataType::S64,
        DataType::U64,
    ];
    const SAMPLES: usize = 64;

    let mut rng = Acorn::new();
    for &src_type in DATA_TYPES {
        // OpSwz must be able to handle all hw variants
        let mut cases: Vec<Swizzle> = AsmSwizzleWiden::VARIANTS
            .iter()
            .filter_map(|widen| widen.to_swizzle(src_type))
            .collect();

        // It should also be able to encode all "weird" variants, let's sample
        // some at random
        cases.extend(
            std::iter::repeat_with(|| sample_swizzle(&mut rng, src_type))
                .flatten()
                .take(SAMPLES),
        );

        if src_type == DataType::F32 {
            cases.push(Swizzle::HF0);
            cases.push(Swizzle::HF1);
        }

        for swizzle in cases {
            let op = OpSwz {
                dst: DstRef::None.into(),
                src_type,
                src: Src::from(0_u32).swizzle(swizzle),
            };
            if !op.src_supports_swizzle(&op.src, swizzle) {
                continue;
            }
            test_foldable_op(op, Precision::Exact);
        }
    }
}

mod builder {
    use super::*;

    #[test]
    fn test_fexp() {
        // Vulkan Environment for SPIR-V requires an absolute precision of
        // 3 + 2*|x| ULP
        const BASE_RANGE: Range<f32> = 0.0..10.0;
        const RANGE: Range<f32> = -150.0..150.0;

        let run = RunSingleton::get();
        let shader = {
            let mut b = TestShaderBuilder::new(&*run.model);
            let log2_base = b.ld_test_data(0, 32);
            let input = b.ld_test_data(4, 32);

            let dst = b.alloc_ssa(32);
            b.fexp_32_to(dst.into(), input.into(), log2_base.into());
            b.st_test_data(8, dst.into());

            b.compile()
        };

        let mut rng = Acorn::new();
        let exp_case =
            |base: f32, num: f32| [base.log2().to_bits(), num.to_bits(), 0];
        // Notable cases
        let mut data = vec![
            exp_case(1.0, 0.0),
            exp_case(2.0, 0.0),
            exp_case(2.0, 1.0),
            exp_case(2.0, 2.0),
            exp_case(4.0, 2.0),
            exp_case(2.0, -1.0),
            exp_case(2.0, -2.0),
            exp_case(E, 2.0),
        ];

        for _ in 0..1000 {
            let a = sample_f32_range(&mut rng, BASE_RANGE);
            let b = sample_f32_range(&mut rng, RANGE);
            data.push([a.to_bits(), b.to_bits(), 0]);
        }

        let case = shader.with_data(&mut data);
        run.execute(case);
        for arr in data {
            let [base_log2, arg, res] = arr.map(f32::from_bits);
            let comp = (base_log2 * arg).exp2().flush_subnormals();

            let ulps = 3 + 2 * ((base_log2 * arg).abs() as u32);
            let prec = Precision::Ulp(ulps);
            assert_f32_eq!(comp, res, prec, "fexp({base_log2}, {arg})");
        }
    }

    #[test]
    fn test_flog2() {
        const RANGE: Range<f32> = -1000.0..1000.0;

        let run = RunSingleton::get();
        let shader = {
            let mut b = TestShaderBuilder::new(&*run.model);
            let input = b.ld_test_data(0, 32);

            let dst = b.flog2_32(input.into());
            b.st_test_data(4, dst.into());
            b.compile()
        };

        let mut rng = Acorn::new();
        let log_case = |arg: f32| [arg.to_bits(), 0];
        // Notable cases
        let mut data = vec![
            log_case(0.5),
            log_case(1.0),
            log_case(2.0),
            log_case(4.0),
            log_case(E),
            log_case(10.0),
            log_case(-1.0),
            log_case(f32::NAN),
        ];

        for _ in 0..1000 {
            let x = sample_f32_range(&mut rng, RANGE);
            data.push([x.to_bits(), 0]);
        }

        let case = shader.with_data(&mut data);
        run.execute(case);
        for arr in data {
            let [input, comp] = arr.map(f32::from_bits);
            let res = input.log2();

            let prec = if (0.5..=2.0).contains(&input) {
                Precision::Abs(2f32.powi(-21))
            } else {
                Precision::Ulp(3)
            };
            assert_f32_eq!(comp, res, prec, "flog2({input})");
        }
    }

    #[test]
    fn test_sin_cos() {
        // Vulkan Environment for SPIR-V requires an absolute precision of 2^-11
        // in the range [-PI, PI]
        const RANGE: Range<f32> = -PI..PI;
        let prec = Precision::Abs(2f32.powi(-11));

        let run = RunSingleton::get();
        let shader = {
            let mut b = TestShaderBuilder::new(&*run.model);
            let input = b.ld_test_data(0, 32);

            let sin = b.alloc_ssa(32);
            let cos = b.alloc_ssa(32);
            b.fsincos_32_to(sin.into(), input.clone().into(), false);
            b.fsincos_32_to(cos.into(), input.into(), true);

            b.st_test_data(4, sin.into());
            b.st_test_data(8, cos.into());

            b.compile()
        };

        let mut data = Vec::new();
        let mut rng = Acorn::new();
        // Notable cases
        for c in [0f32, PI, PI / 2.0, PI / 4.0, 2.0 * PI, f32::NAN] {
            data.push([c.to_bits(), 0, 0]);
        }
        for _ in 0..1000 {
            let x = sample_f32_range(&mut rng, RANGE);
            data.push([x.to_bits(), 0, 0]);
        }

        let case = shader.with_data(&mut data);
        run.execute(case);
        for arr in data {
            let [input, csin, ccos] = arr.map(f32::from_bits);
            // Rust doesn't specify a precision, but Kraid tests are almost surely
            // built with glibc, that should offer more precision what what we need
            let (esin, ecos) = input.sin_cos();

            assert_f32_eq!(esin, csin, prec, "sin({input})");
            assert_f32_eq!(ecos, ccos, prec, "cos({input})");
        }
    }
}
