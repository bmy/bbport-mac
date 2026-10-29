// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: sigaction over a vectored exception handler (posix_win32.cpp). The handlers the
// port has for Linux (write traps' SIGSEGV, the int3 hooks' SIGTRAP) run as they are: an access
// violation is SIGSEGV (si_addr the address, REG_ERR bit 1 a write), a breakpoint SIGTRAP (RIP
// past the int3, as on Linux), an illegal instruction SIGILL. The context they change is resumed.
// A handler left at SIG_DFL passes the exception on (Windows' own handling, a crash report).
#pragma once

#include <signal.h>

#ifdef __cplusplus
extern "C" {
#endif

// Linux numbers for what the UCRT lacks (it has SIGINT 2, SIGILL 4, SIGFPE 8, SIGSEGV 11,
// SIGTERM 15, SIGABRT 22).
#ifndef SIGTRAP
#define SIGTRAP 5
#endif
#ifndef SIGBUS
#define SIGBUS 7
#endif
#ifndef SIGUSR1
#define SIGUSR1 10
#endif
#ifndef SIGUSR2
#define SIGUSR2 12
#endif
#ifndef SIGPROF
#define SIGPROF 27
#endif
#define BB_NSIG 65

#define SA_SIGINFO 0x4
#define SA_ONSTACK 0x08000000
#define SA_RESTART 0x10000000
#define SA_NODEFER 0x40000000

#define SEGV_MAPERR 1
#define SEGV_ACCERR 2

typedef unsigned long long sigset_t;

typedef struct {
    int si_signo;
    int si_errno;
    int si_code;
    void* si_addr;
} siginfo_t;

struct sigaction {
    union {
        void (*sa_handler)(int);
        void (*sa_sigaction)(int, siginfo_t*, void*);
    };
    sigset_t sa_mask;
    int sa_flags;
};

int sigaction(int sig, const struct sigaction* action, struct sigaction* previous);
static inline int sigemptyset(sigset_t* set) {
    *set = 0;
    return 0;
}
static inline int sigfillset(sigset_t* set) {
    *set = ~0ull;
    return 0;
}
static inline int sigaddset(sigset_t* set, int sig) {
    *set |= 1ull << (sig & 63);
    return 0;
}
static inline int sigdelset(sigset_t* set, int sig) {
    *set &= ~(1ull << (sig & 63));
    return 0;
}
static inline int sigismember(const sigset_t* set, int sig) {
    return (int)((*set >> (sig & 63)) & 1);
}
/// No signal masks on Windows: exceptions cannot be blocked.
static inline int sigprocmask(int how, const sigset_t* set, sigset_t* old) {
    (void)how;
    (void)set;
    if (old) *old = 0;
    return 0;
}
#define SIG_BLOCK 0
#define SIG_UNBLOCK 1
#define SIG_SETMASK 2

#ifdef __cplusplus
}
#endif
