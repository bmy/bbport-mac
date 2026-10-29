// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what a SIGSEGV handler needs from its ucontext (x86-64 Linux; x86-64 macOS under
// Rosetta 2, where the registers live in the Darwin thread state; arm64 macOS, the native GPU
// process, where only the host's own faults reach a handler and the x86 names are stand-ins).
#pragma once

#include <cstddef>
#include <sys/ucontext.h>

#include "common/types.h"

#if defined(__APPLE__) && defined(__aarch64__)
// bbport (the native GPU process, docs/macos-native-gpu.md): the Darwin arm64 thread state as
// 64-bit fields (x0-x28, fp, lr, sp, pc, then cpsr and padding). The guest's handlers never run
// there (no guest code); RIP and RSP name the pc and sp so the shared code reads something real.
enum : int {
    REG_RAX = 0, REG_RBX, REG_RCX, REG_RDX, REG_RDI, REG_RSI, REG_RBP, REG_R8 = 8, REG_R9,
    REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15, REG_RSP = 31, REG_RIP = 32,
    REG_EFL = 33, REG_CSGSFS = 33
};
typedef long long greg_t;
static_assert(offsetof(__darwin_arm_thread_state64, __sp) == REG_RSP * sizeof(greg_t));
static_assert(offsetof(__darwin_arm_thread_state64, __pc) == REG_RIP * sizeof(greg_t));
static_assert(offsetof(__darwin_arm_thread_state64, __cpsr) == REG_EFL * sizeof(greg_t));
#elif defined(__APPLE__)
// bbport (macOS): Linux's register indices over the Darwin x86 thread state, which keeps the
// general-purpose registers as consecutive 64-bit fields in this order (struct
// __darwin_x86_thread_state64: rax, rbx, rcx, rdx, rdi, rsi, rbp, rsp, r8-r15, rip, rflags, cs).
// REG_ERR has no place there: Common::IsWriteError reads the exception state instead.
enum : int {
    REG_RAX = 0, REG_RBX, REG_RCX, REG_RDX, REG_RDI, REG_RSI, REG_RBP, REG_RSP, REG_R8, REG_R9,
    REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15, REG_RIP, REG_EFL, REG_CSGSFS
};
typedef long long greg_t;
static_assert(offsetof(__darwin_x86_thread_state64, __rip) == REG_RIP * sizeof(greg_t));
static_assert(offsetof(__darwin_x86_thread_state64, __rflags) == REG_EFL * sizeof(greg_t));
static_assert(offsetof(__darwin_x86_thread_state64, __r15) == REG_R15 * sizeof(greg_t));
#endif

namespace Common {

/// The interrupted general-purpose registers, indexed by REG_* (Linux's gregs); writable, as in a
/// Linux handler (RIP, the flags).
inline greg_t* Gregs(void* context) {
#ifdef __APPLE__
    return reinterpret_cast<greg_t*>(&static_cast<ucontext_t*>(context)->uc_mcontext->__ss);
#else
    return static_cast<ucontext_t*>(context)->uc_mcontext.gregs;
#endif
}
inline const greg_t* Gregs(const void* context) {
    return Gregs(const_cast<void*>(context));
}

/// The faulting instruction.
inline void* GetRip(void* context) {
    return reinterpret_cast<void*>(Gregs(context)[REG_RIP]);
}

/// The fault was a write (page fault error code bit 1).
inline bool IsWriteError(void* context) {
#if defined(__APPLE__) && defined(__aarch64__)
    // A data abort's ESR: bit 6 (WnR) is a write.
    return (static_cast<ucontext_t*>(context)->uc_mcontext->__es.__esr >> 6) & 1;
#elif defined(__APPLE__)
    return static_cast<ucontext_t*>(context)->uc_mcontext->__es.__err & 0x2;
#else
    return static_cast<ucontext_t*>(context)->uc_mcontext.gregs[REG_ERR] & 0x2;
#endif
}

} // namespace Common
