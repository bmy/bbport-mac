// Copyright (c) 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use std::io::Write;

use crate::isa::v9;

pub use v9::decode::{Mnemonic, PrintCtx, Variant, print, try_decode};

use kraid_bindings::*;

struct CFileWrapper(*mut FILE);

impl Write for CFileWrapper {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }

        let size = buf.len().try_into().unwrap();
        let nwritten = unsafe {
            fwrite(buf.as_ptr() as *const std::ffi::c_void, 1, size, self.0)
        };

        if nwritten == 0 {
            Err(std::io::Error::last_os_error())
        } else {
            Ok(nwritten as usize)
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        let res = unsafe { fflush(self.0) };
        if res == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

fn trim_at_first_zero(instrs: &[u64]) -> &[u64] {
    for (idx, instr) in instrs.iter().enumerate() {
        if *instr == 0 {
            return &instrs[0..idx];
        }
    }
    instrs
}

#[unsafe(no_mangle)]
pub extern "C" fn kraid_disassemble(
    fp: *mut FILE,
    code: *const std::ffi::c_void,
    size: usize,
    verbose: bool,
    arch: std::ffi::c_uchar,
) {
    if code == std::ptr::null() || size == 0 {
        return;
    }

    assert!(size % align_of::<u64>() == 0);
    assert!((code as *const u64).is_aligned());

    let raw_slice =
        std::ptr::slice_from_raw_parts(code as *const u64, size / 8);
    // SAFETY: Code is not null and properly aligned.
    let instrs: &[u64] = unsafe { &*raw_slice };
    let instrs = trim_at_first_zero(instrs);

    let args = Args {
        arch: arch.try_into().unwrap(),
        print_hexdump: verbose,
        print_offset: verbose,
    };
    let mut fp_wrap = CFileWrapper(fp);

    // Ignore the result, nothing we can do here if it didn't work.
    let _ = disassemble(&mut fp_wrap, &args, instrs);
}

pub struct Args {
    pub arch: u8,
    pub print_hexdump: bool,
    pub print_offset: bool,
}

pub fn disassemble<W: Write>(
    f: &mut W,
    args: &Args,
    instrs: &[u64],
) -> std::io::Result<()> {
    let arch = args.arch;
    let mut print_ctx = v9::decode::PrintCtx {
        pc_base: 0,
        fau32: false,
    };

    for instr in instrs {
        if args.print_offset {
            write!(f, "{:04x}:    ", print_ctx.pc_base)?;
        }
        if args.print_hexdump {
            write!(f, "{:016x}    ", instr)?;
        }
        match v9::decode::try_decode(*instr, arch) {
            Ok((m, v)) => {
                v9::decode::print(*instr, m, v, arch, f, &print_ctx)?;
                writeln!(f, "")?;
            }
            Err(err) => {
                writeln!(f, "{}", err)?;
            }
        }
        print_ctx.pc_base += 8;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::isa::InvalidInstrError;

    use super::*;
    use paste::paste;

    fn decode_to_string(
        val: u64,
        arch: u8,
        ctx: PrintCtx,
    ) -> Result<String, InvalidInstrError> {
        let mut buffer = Vec::new();

        match try_decode(val, arch) {
            Ok((mn, var)) => {
                print(val, mn, var, arch, &mut buffer, &ctx).unwrap();
                let result_string = String::from_utf8(buffer).unwrap();
                Ok(result_string)
            }
            Err(err) => Err(err),
        }
    }

    macro_rules! print_ctx {
        ($( $field:ident = $value:expr ),*) => {{
            let mut ctx: PrintCtx = Default::default();

            $(ctx.$field = $value;);*

            ctx
        }};
    }

    macro_rules! disasm_case {
        ($val:tt, $result:literal, $ctx:expr, $arch:literal) => {
            paste! {
                #[test]
                fn [<decode_ $val>]() {
                    let res = decode_to_string($val, $arch, $ctx);
                    assert!(res.is_ok());
                    assert_eq!(res.unwrap().as_str(), $result);
                }
            }
        };
        ($val:tt, $result:literal, $arch:literal) => {
            disasm_case!($val, $result, Default::default(), $arch);
        };
        ($val:tt, $result:literal, $ctx:expr) => {
            disasm_case!($val, $result, $ctx, 13u8);
        };
        ($val:tt, $result:literal) => {
            disasm_case!($val, $result, Default::default(), 13u8);
        };
    }

    disasm_case!(0x00a4c8d00000c048, "FADD.f32 r8, r8^.abs.neg, k0.neg");

    disasm_case!(
        0x00728a46303b903c,
        "LD_CHECKED_IMM.i96.slot0 @r10:r11:r12, [r60:r61], #0x490, r59"
    );

    disasm_case!(
        0x001fc00000001842,
        "BRANCH r2^, 0xbfb7",
        print_ctx!(pc_base = 0xbeef)
    );

    disasm_case!(0x001fc007fffffe5b, "BRANCH r27^, 0xfffffffffffffff8");

    disasm_case!(
        0x007f40083230fe0a,
        "BLEND.slot0.v4.f32 @r0:r1:r2:r3, [r10:r11], 0x7f8, r48"
    );

    disasm_case!(
        0x007f40043330fe0a,
        "BLEND.slot0.v4.f16 @r0:r1, [r10:r11], 0x7f8, r48"
    );

    disasm_case!(
        0x0920c0041fc70048,
        "ACMPXCHG.i32.slot0.wait0 @r7, @r0:r1, [r8^:r9^], #0x0"
    );

    disasm_case!(0x00b4c9015091c014, "AND.i32.not r9, r20, u8.w1");
    disasm_case!(
        0x00b4c9015091c015,
        "AND.i32.not r9, r21, u17",
        print_ctx!(fau32 = true)
    );

    // This one is ambiguous, the bit pattern matches FMUL_RSCALE as well.
    // It should be decoded as FADD_RSCALE.
    disasm_case!(0x0160c00404c0d001, "FADD_RSCALE.f32 r0, r1, k0.neg, r4");
    // If src1 is not #1.0, then it should become an FMUL_RSCALE.
    disasm_case!(0x0160c00404c00201, "FMUL_RSCALE.f32 r0, r1, r2, r4");
    // And if src2 is not 0 either, it should become FMA_RSCALE.
    disasm_case!(0x0160c00404030201, "FMA_RSCALE.f32 r0, r1, r2, r3.neg, r4");

    disasm_case!(
        0x092540b49ff00085,
        "TEX_FETCH.slot2.wait0.skip.tex2d.dst32.rgba @r48:r49:r50:r51, @r0:r1, u2.zext",
        14u8
    );
}
