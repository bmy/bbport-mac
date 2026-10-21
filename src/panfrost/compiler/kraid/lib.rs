// Copyright © 2026 Collabora, Ltd.
// SPDX-License-Identifier: MIT

pub extern crate mesa_util;

mod builder;
mod compile;
mod data_type;
pub mod decode;
mod dst_mod_prop;
mod encode_v9;
mod flow;
mod foldable;
#[cfg(test)]
mod hw_tests;
mod ir;
mod isa;
mod jump_thread;
mod legalize;
mod legalize_src_swizzles;
mod liveness;
mod lower_blend_call;
mod lower_copy;
mod lower_mkvec_swz;
mod message_slots;
mod model;
mod nir;
mod ops;
mod opt_copy_prop;
mod opt_dce;
mod opt_exec_units;
mod opt_flow;
mod opt_normalize_consts;
mod opt_promote_consts;
mod opt_var;
mod parallel_copy;
mod phi;
mod ra;
mod reconvergence;
mod remat_constants;
mod repair_ssa;
mod required_waits;
mod schedule;
mod small_constants;
mod spill;
mod ssa_value;
mod stats;
mod swizzle;
mod validate;
mod widen_alu_ops;

mod debug {
    bitflags::bitflags! {
        pub struct DebugFlags: u32 {
            const PRINT = 1 << 0;
            const VALIDATE = 1 << 1;
            const SPILL = 1 << 2;
            const SERIAL = 1 << 3;
            const PRINT_RAW_CONST = 1 << 4;
            const STATS = 1 << 5;
        }
    }

    fn get_debug_flags() -> DebugFlags {
        let debug_var = "KRAID_DEBUG";
        let Ok(debug_str) = std::env::var(debug_var) else {
            return DebugFlags::empty();
        };

        let mut flags = DebugFlags::empty();
        for flag in debug_str.split(',') {
            match flag.trim() {
                "print" => flags |= DebugFlags::PRINT,
                "validate" => flags |= DebugFlags::VALIDATE,
                "spill" => flags |= DebugFlags::SPILL,
                "serial" => flags |= DebugFlags::SERIAL,
                "print-raw-constants" => flags |= DebugFlags::PRINT_RAW_CONST,
                "stats" => flags |= DebugFlags::STATS,
                unk => eprintln!("Unknown {debug_var} flag \"{}\"", unk),
            }
        }
        flags
    }

    pub struct Debug {
        flags: std::sync::OnceLock<DebugFlags>,
    }

    impl std::ops::Deref for Debug {
        type Target = DebugFlags;

        fn deref(&self) -> &DebugFlags {
            self.flags.get_or_init(get_debug_flags)
        }
    }

    pub static DEBUG: Debug = Debug {
        flags: std::sync::OnceLock::new(),
    };
}

#[unsafe(no_mangle)]
pub extern "C" fn kraid_get_compiler_flags() -> u32 {
    debug::DEBUG.bits().into()
}
