// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: the Linux x86-64 signal context the fault handlers read (gregs[REG_*]),
// filled from the vectored exception handler's CONTEXT by the runtime (BbWin32ToUcontext) and
// written back after the handler (it may change RIP or the flags).
#pragma once

#include "bbport_signal.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    REG_R8 = 0, REG_R9, REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15, REG_RDI, REG_RSI,
    REG_RBP, REG_RBX, REG_RDX, REG_RAX, REG_RCX, REG_RSP, REG_RIP, REG_EFL, REG_CSGSFS, REG_ERR,
    REG_TRAPNO, REG_OLDMASK, REG_CR2
};
#define NGREG 23
typedef long long greg_t;
typedef greg_t gregset_t[NGREG];
typedef struct {
    gregset_t gregs;
    void* fpregs;
    unsigned long long reserved[8];
} mcontext_t;
typedef struct ucontext_t {
    unsigned long uc_flags;
    struct ucontext_t* uc_link;
    mcontext_t uc_mcontext;
} ucontext_t;



#ifdef __cplusplus
}
#endif

#if defined(_WINNT_) || defined(_WINDOWS_)
/// The exception's registers as a signal context, and back.
static inline void BbWin32ToUcontext(const CONTEXT* c, unsigned long long fault_error, ucontext_t* u) {
    greg_t* g = u->uc_mcontext.gregs;
    g[REG_R8] = c->R8; g[REG_R9] = c->R9; g[REG_R10] = c->R10; g[REG_R11] = c->R11;
    g[REG_R12] = c->R12; g[REG_R13] = c->R13; g[REG_R14] = c->R14; g[REG_R15] = c->R15;
    g[REG_RDI] = c->Rdi; g[REG_RSI] = c->Rsi; g[REG_RBP] = c->Rbp; g[REG_RBX] = c->Rbx;
    g[REG_RDX] = c->Rdx; g[REG_RAX] = c->Rax; g[REG_RCX] = c->Rcx; g[REG_RSP] = c->Rsp;
    g[REG_RIP] = c->Rip; g[REG_EFL] = c->EFlags; g[REG_ERR] = (greg_t)fault_error;
}
static inline void BbUcontextToWin32(const ucontext_t* u, CONTEXT* c) {
    const greg_t* g = u->uc_mcontext.gregs;
    c->R8 = g[REG_R8]; c->R9 = g[REG_R9]; c->R10 = g[REG_R10]; c->R11 = g[REG_R11];
    c->R12 = g[REG_R12]; c->R13 = g[REG_R13]; c->R14 = g[REG_R14]; c->R15 = g[REG_R15];
    c->Rdi = g[REG_RDI]; c->Rsi = g[REG_RSI]; c->Rbp = g[REG_RBP]; c->Rbx = g[REG_RBX];
    c->Rdx = g[REG_RDX]; c->Rax = g[REG_RAX]; c->Rcx = g[REG_RCX]; c->Rsp = g[REG_RSP];
    c->Rip = g[REG_RIP]; c->EFlags = (unsigned long)g[REG_EFL];
}
#endif
