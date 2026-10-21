// Copyright (c) 2026 Arm Ltd.
// SPDX-License-Identifier: MIT

use kraid_rs::decode::{Args as DisasmArgs, disassemble};
use std::io::BufRead;

#[derive(Debug)]
enum ParseArgError {
    ParseInt(core::num::ParseIntError),
    Str(String),
}

impl From<core::num::ParseIntError> for ParseArgError {
    fn from(e: core::num::ParseIntError) -> ParseArgError {
        ParseArgError::ParseInt(e)
    }
}

impl From<String> for ParseArgError {
    fn from(e: String) -> ParseArgError {
        ParseArgError::Str(e)
    }
}

impl std::fmt::Display for ParseArgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseArgError::ParseInt(e) => write!(f, "{}", e),
            ParseArgError::Str(e) => write!(f, "{}", e),
        }
    }
}

struct Args {
    arch: u8,
    print_hexdump: bool,
    print_offset: bool,
}

impl Args {
    fn parse() -> Result<Args, ParseArgError> {
        let mut args = Args {
            print_hexdump: false,
            print_offset: false,
            arch: 13,
        };

        let mut args_iter = std::env::args().skip(1);

        while let Some(arg) = args_iter.next() {
            match arg.as_str() {
                "--arch" => {
                    let arch = u8::from_str_radix(
                        args_iter
                            .next()
                            .expect("Missing --arch value")
                            .as_str(),
                        10,
                    )?;
                    args.arch = arch;
                }
                "--print-offset" => args.print_offset = true,
                "--print-hexdump" => args.print_hexdump = true,
                s => return Err(format!("Unknown option '{}'", s).into()),
            }
        }

        Ok(args)
    }
}

impl From<Args> for DisasmArgs {
    fn from(value: Args) -> Self {
        Self {
            arch: value.arch,
            print_offset: value.print_offset,
            print_hexdump: value.print_hexdump,
        }
    }
}

/// Read binary to disassemble from stdin, one byte per line in hex.
fn from_stdin() -> Vec<u64> {
    let mut instrs_bin: Vec<u64> = Default::default();

    let mut buf = [0u8; 8];
    let mut count = 0;

    let stdin = std::io::stdin();
    for input in stdin.lock().lines() {
        let input = input.expect("failed to read line");

        let stripped = input.strip_prefix("0x").unwrap_or(&input);
        if let Ok(byte) = u8::from_str_radix(stripped, 16) {
            buf[count] = byte;
        } else {
            eprintln!("Failed to parse input byte from: {}", input);
        }
        count += 1;

        if count >= 8 {
            debug_assert!(count == 8);
            instrs_bin.push(u64::from_le_bytes(buf));
            count = 0;
            buf = [0; 8];
        }
    }

    // Delete zero padding.
    let end = instrs_bin
        .iter()
        .rev()
        .position(|i| *i != 0)
        .map(|idx| instrs_bin.len() - idx)
        .unwrap_or(instrs_bin.len());
    instrs_bin.truncate(end);

    instrs_bin
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if let Err(e) = args {
        println!("Invalid arguments: {e}");
        return Ok(());
    }
    let args = args.unwrap();

    let binary = from_stdin();

    let stdio = std::io::stdout();
    let mut lock = stdio.lock();
    disassemble(&mut lock, &args.into(), &binary)
}
