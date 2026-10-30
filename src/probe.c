// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* Minimal x86-64 native loader experiment. Not a PS4 emulator or game port. */
#define _GNU_SOURCE
#include <stdint.h>
#include "../gpu/shim/bb_env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>
#include <cpuid.h>
#include "runtime.h"
#include "platform.h"
#include "runtime_trophy.h"
#include "gpu/bbgpu.h"
#if !defined(__x86_64__) || !defined(__GNUC__)
#error This prototype requires x86-64 GCC or Clang (including MinGW).
#endif
#include <sys/mman.h>
#include <unistd.h>
#include <signal.h>
#ifdef __APPLE__
#include <sys/ucontext.h>
#else
#include <ucontext.h>
#endif
#include <fcntl.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <pthread.h>
#include <sys/uio.h>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <process.h>
#include <windows.h>
#else
#include <malloc.h>
#include <sys/syscall.h>
#endif

typedef struct { uint64_t address, size, flags; } Segment;
typedef struct { uint64_t target, kind, value, addend; } Reloc;
static char (*names)[128];
static uint64_t import_count;
static unsigned char *image;
static size_t page_size;
typedef struct { uint64_t base, size, init, tls_address, tls_memsz, tls_filesz, tls_module; } LinkedModule;
static LinkedModule modules[16];
static uint64_t module_count;
static int entered_game;
static int gpu_enabled;
int vulkan_smoke(void);

static void fail(const char *message) { fprintf(stderr, "ERROR: %s\n", message); exit(1); }

/* The BB_* variables set (gpu/shim/bb_env_vars.def, docs/ENVIRONMENT.md), and any BB_* name the
 * port does not read: a typo leaves a setting off without a word. The internal ones (run.py and the
 * launchers set them every run) are left out, accounts and keys shown as set only. */
static int name_has(const char *entry, size_t length, const char *part) {
    const char *at = strstr(entry, part);
    return at && (size_t)(at - entry) < length;
}

enum { BB_ENV_SETTING, BB_ENV_TUNING, BB_ENV_DEBUG, BB_ENV_INTERNAL, BB_ENV_BUILD };
static void print_environment(void) {
    static const struct { const char *name; int kind; } known[] = {
#define BB_ENV(name, kind, text) {#name, BB_ENV_##kind},
#include "../gpu/shim/bb_env_vars.def"
#undef BB_ENV
    };
#ifdef _WIN32
    char **env = _environ;
#else
    extern char **environ;
    char **env = environ;
#endif
    int printed = 0;
    for (; env && *env; ++env) {
        const char *entry = *env, *eq = strchr(entry, '=');
        if (strncmp(entry, "BB_", 3) || !eq) continue;
        const size_t length = (size_t)(eq - entry);
        int kind = -1;
        for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); ++i) {
            if (strlen(known[i].name) == length && !strncmp(known[i].name, entry, length)) {
                kind = known[i].kind;
                break;
            }
        }
        if (kind < 0) {
            printf("Environment: %.*s is not a bbport variable (a typo? docs/ENVIRONMENT.md)\n",
                   (int)length, entry);
            continue;
        }
        if (kind == BB_ENV_INTERNAL || kind == BB_ENV_BUILD) continue;
        const int secret = name_has(entry, length, "PASSWORD") || name_has(entry, length, "TOKEN") ||
                           name_has(entry, length, "_KEY");
        printf("%s %.*s=%s", printed++ ? "" : "Environment:", (int)length, entry, secret ? "(set)" : eq + 1);
    }
    if (printed) printf("\n");
}

/* The game's code was compiled for the PS4's CPU (AMD Jaguar) and runs as it is: AVX, BMI1
 * (andn/bextr/blsr/tzcnt), MOVBE, LZCNT and POPCNT, thousands of each in eboot.bin. A CPU without
 * them stops at the first one with SIGILL (exit code 132, issue #26), and lzcnt/tzcnt even run as
 * bsr/bsf there, with other results. Said before the game starts; BB_SKIP_CPU_CHECK=1 skips it. */
static void check_cpu(void) {
    const char *skip = bb_env("BB_SKIP_CPU_CHECK");
    if (skip && !strcmp(skip, "1")) return;
#ifdef __APPLE__
    /* bbport (macOS): Rosetta 2 runs AVX, BMI1, MOVBE and LZCNT but does not report them in
     * CPUID unless asked to (ROSETTA_ADVERTISE_AVX), so this check would always refuse. */
    return;
#endif
    unsigned a, b, c, d, leaf1_c = 0, leaf7_b = 0, ext1_c = 0;
    if (__get_cpuid(1, &a, &b, &c, &d)) leaf1_c = c;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) leaf7_b = b;
    if (__get_cpuid(0x80000001u, &a, &b, &c, &d)) ext1_c = c;
    int avx = (leaf1_c & bit_AVX) && (leaf1_c & bit_OSXSAVE);
    if (avx) {
        unsigned lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        avx = (lo & 6) == 6; /* the OS saves SSE and AVX state */
    }
    const struct { int present; const char *name; } features[] = {
        {avx, "AVX"}, {(leaf7_b & bit_BMI) != 0, "BMI1"}, {(leaf1_c & bit_MOVBE) != 0, "MOVBE"},
        {(ext1_c & bit_LZCNT) != 0, "LZCNT"}, {(leaf1_c & bit_POPCNT) != 0, "POPCNT"},
        {(leaf1_c & bit_SSE4_2) != 0, "SSE4.2"},
    };
    char missing[64] = "";
    for (unsigned i = 0; i < sizeof(features) / sizeof(features[0]); ++i) {
        if (features[i].present) continue;
        if (missing[0]) strcat(missing, ", ");
        strcat(missing, features[i].name);
    }
    if (!missing[0]) return;
    fprintf(stderr,
            "ERROR: this CPU lacks %s. Bloodborne's code was compiled for the PS4's CPU and uses "
            "these instructions directly: it needs an Intel Haswell (4th generation Core, 2013) or "
            "newer, or an AMD Ryzen. BB_SKIP_CPU_CHECK=1 starts anyway.\n", missing);
    exit(1);
}
static uint64_t read64(FILE *f) {
    unsigned char b[8];
    if (fread(b, 1, 8, f) != 8) fail("truncated boot file");
    uint64_t n = 0;
    for (int i = 7; i >= 0; --i) n = (n << 8) | b[i];
    return n;
}
static size_t bb_round_page(size_t size) { return (size + page_size - 1) & ~(page_size - 1); }
static void *allocate(size_t size) {
    void *low=runtime_low_map(size,PROT_READ|PROT_WRITE);
    if (low) return low;
#ifdef _WIN32
    void *p = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (!p) fail("VirtualAlloc failed");
#else
    void *p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) fail("mmap failed");
#endif
    return p;
}
static void protect(void *p, size_t size, unsigned flags) {
#ifdef _WIN32
    DWORD old, mode = PAGE_NOACCESS;
    if (flags & 1) mode = (flags & 2) ? PAGE_EXECUTE_READWRITE : PAGE_EXECUTE_READ;
    else if (flags & 2) mode = PAGE_READWRITE;
    else if (flags & 4) mode = PAGE_READONLY;
    if (!VirtualProtect(p, size, mode, &old)) fail("VirtualProtect failed");
    FlushInstructionCache(GetCurrentProcess(), p, size);
    runtime_low_protect((uintptr_t)p, size, ((flags & 4) ? PROT_READ : 0) | ((flags & 2) ? PROT_WRITE : 0) |
                                            ((flags & 1) ? PROT_EXEC : 0));
#else
    int mode = ((flags & 4) ? PROT_READ : 0) | ((flags & 2) ? PROT_WRITE : 0) | ((flags & 1) ? PROT_EXEC : 0);
    if (mprotect(p, size, mode)) fail("mprotect failed");
    runtime_low_protect((uintptr_t)p, size, mode);
#endif
}
static ABI __attribute__((noreturn)) void unresolved(uint32_t id, uintptr_t argument) {
    if (id >= import_count) fail("bad import trap index");
    printf("STOP: first unsupported PS4 import: %s (index %u)\n", names[id], id);
    printf("API: %s\n", runtime_import_name(names[id]));
    printf("Caller return offset: 0x%" PRIxPTR "; first argument: 0x%" PRIxPTR "\n",
           (uintptr_t)__builtin_return_address(0) - (uintptr_t)image, argument);
    uintptr_t caller=(uintptr_t)__builtin_return_address(0)-(uintptr_t)image;
    for (uint64_t m=0;m<module_count;++m)
        if (caller>=modules[m].base && caller-modules[m].base<modules[m].size)
            printf("Caller in linked module %" PRIu64 " (%s): +0x%" PRIxPTR "\n",m,m==0 ? "libc.prx" : "system module",(uintptr_t)(caller-modules[m].base));
    runtime_report();
    puts(entered_game ? "Original guest entry instructions executed; game initialization is incomplete." :
                        "Native libc initialization is incomplete; game entry has not run.");
    fflush(NULL);
    _exit(20); /* no destructors: GPU, audio and guest threads are still running */
}
#ifndef _WIN32
/* enter_on_stack(entry, arg0, arg1, stack_top): call entry(arg0,arg1) on a new stack. */
void enter_on_stack(void *entry, void *arg0, void *arg1, void *top);
#ifdef __APPLE__
#define ASM_SYMBOL(name) "_" #name /* Mach-O: C symbols carry a leading underscore */
#else
#define ASM_SYMBOL(name) #name
#endif
__asm__(".text\n.globl " ASM_SYMBOL(enter_on_stack) "\n" ASM_SYMBOL(enter_on_stack) ":\n"
        " push %rbp\n mov %rsp,%rbp\n and $-16,%rcx\n mov %rcx,%rsp\n"
        " mov %rdi,%rax\n mov %rsi,%rdi\n mov %rdx,%rsi\n call *%rax\n"
        " mov %rbp,%rsp\n pop %rbp\n ret\n");
#endif
static ABI void guest_exit(void) { puts("Runtime: process finalizer callback reached"); }
#ifdef _WIN32
/* The game's entry on the main thread's low stack (bb_run_on_stack). */
static void enter_game(void *p) {
    struct { void *entry, *params; } *start=p;
    typedef void (ABI *Entry)(void *, void (ABI *)(void));
    ((Entry)start->entry)(start->params, guest_exit);
}
/* The linker rewrote the game's `mov r64, fs:[0]` (its thread pointer) to `mov r64, gs:[0]`
 * (link_modules.py). GS is the TEB on Windows: the load reads a TEB TLS slot instead
 * (gs:[0x1480 + 8 * slot], TlsSlots), where each guest thread keeps its TCB (runtime_thread.c). */
static void patch_guest_tls(const Segment *segments, uint64_t ns) {
    const uint32_t slot_offset=0x1480u+8u*(uint32_t)bb_guest_tls_slot();
    uint64_t patched=0;
    for (uint64_t i=0;i<ns;++i) {
        if (!(segments[i].flags&1) || segments[i].size<9) continue;
        unsigned char *p=image+segments[i].address, *end=p+segments[i].size-9;
        for (; p<=end; ++p) {
            if (p[0]!=0x65 || (p[1]!=0x48 && p[1]!=0x4c) || p[2]!=0x8b || (p[3]&0xc7)!=0x04 || p[4]!=0x25 ||
                p[5] || p[6] || p[7] || p[8]) continue;
            const uint64_t at=(uint64_t)(p-image), first=at&~(uint64_t)(page_size-1);
            protect(image+first,bb_round_page(at+9)-first,7); /* hooks may have left it read-only */
            memcpy(p+5,&slot_offset,4);
            ++patched;
        }
    }
    printf("Runtime: %" PRIu64 " guest thread pointer loads read TEB TLS slot %d\n",patched,bb_guest_tls_slot());
}
#endif
static void fault(int sig, siginfo_t *info, void *context) {
    /* GPU page tracking (write-protected guest pages) is resolved first. */
    /* macOS reports protection faults on shared (pool) mappings as SIGBUS, Linux as SIGSEGV. */
    if (gpu_enabled && (sig == SIGSEGV || sig == SIGBUS) && bbgpu_handle_fault(context, info->si_addr)) return;
    /* A speculative guest memory read (runtime_memory.c) failed: resume its recovery point. */
    if ((sig == SIGSEGV || sig == SIGBUS) && runtime_fault_recover) {
        sigjmp_buf *recover = runtime_fault_recover;
        runtime_fault_recover = NULL;
        sigset_t unblock;
        sigemptyset(&unblock);
        sigaddset(&unblock, sig);
        pthread_sigmask(SIG_UNBLOCK, &unblock, NULL);
        siglongjmp(*recover, 1);
    }
    /* The process is terminating: dladdr/snprintf are acceptable here. */
    ucontext_t *uc = context;
    uintptr_t rip = (uintptr_t)BB_CTX_RIP(uc);
    char line[512];
    Dl_info where;
    if (rip - (uintptr_t)image < 0x10000000)
        snprintf(line, sizeof(line), "Guest fault (signal %d) at guest offset 0x%lx, address %p\n",
                 sig, (unsigned long)(rip - (uintptr_t)image), info->si_addr);
    else if (dladdr((void *)rip, &where) && where.dli_fname)
        snprintf(line, sizeof(line), "Host fault (signal %d) in %s+0x%lx (%s), address %p\n", sig, where.dli_fname,
                 (unsigned long)(rip - (uintptr_t)where.dli_fbase), where.dli_sname ? where.dli_sname : "?", info->si_addr);
    else
        snprintf(line, sizeof(line), "Fault (signal %d) at RIP %p, address %p\n", sig, (void *)rip, info->si_addr);
    { ssize_t written_=write(2, line, strlen(line)); (void)written_; }
#ifdef _WIN32
    {
        char mem[256];
        bb_win32_describe_address(info->si_addr, mem, sizeof(mem));
        char out[320];
        const int n = snprintf(out, sizeof(out), "  the address: %s\n", mem);
        ssize_t written_=write(2, out, (size_t)n); (void)written_;
        const greg_t *g = uc->uc_mcontext.gregs;
        const int m = snprintf(out, sizeof(out), "  rax %llx rbx %llx rcx %llx rdx %llx rsi %llx rdi %llx r8 %llx r9 %llx\n",
                               (unsigned long long)g[REG_RAX], (unsigned long long)g[REG_RBX], (unsigned long long)g[REG_RCX],
                               (unsigned long long)g[REG_RDX], (unsigned long long)g[REG_RSI], (unsigned long long)g[REG_RDI],
                               (unsigned long long)g[REG_R8], (unsigned long long)g[REG_R9]);
        written_=write(2, out, (size_t)m); (void)written_;
    }
#endif
    /* A guest fault: its thread and the game's call chain (return addresses along rbp, read with
     * process_vm_readv so a broken chain cannot fault again). */
    if (rip - (uintptr_t)image < 0x10000000) {
        char thread[32] = "?";
        pthread_getname_np(pthread_self(), thread, sizeof(thread));
        int at = snprintf(line, sizeof(line), "  thread %s; calls:", thread);
        uintptr_t rbp = (uintptr_t)BB_CTX_RBP(uc); /* bbport: portable (platform.h) */
        for (int i = 0; i < 12 && rbp && at < (int)sizeof(line) - 24; ++i) {
            uintptr_t frame[2] = {0, 0};
            struct iovec local = {frame, sizeof(frame)}, remote = {(void *)rbp, sizeof(frame)};
            if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != (ssize_t)sizeof(frame)) break;
            if (frame[1] - (uintptr_t)image >= 0x10000000) break;
            at += snprintf(line + at, sizeof(line) - at, " +0x%lx", (unsigned long)(frame[1] - (uintptr_t)image));
            if (frame[0] <= rbp) break;
            rbp = frame[0];
        }
        snprintf(line + at, sizeof(line) - at, "\n");
        { ssize_t written_=write(2, line, strlen(line)); (void)written_; }
    }
    { void runtime_re_watch_report(void); runtime_re_watch_report(); }
    if (gpu_enabled) bbgpu_dump_guest_writes(context);
    /* Outside the image and any shared object (generated code, a freed mapping): the mapping
     * from /proc/self/maps, and the thread. */
    if (rip - (uintptr_t)image >= 0x10000000 && !(dladdr((void *)rip, &where) && where.dli_fname)) {
        char thread[32] = "?";
        pthread_getname_np(pthread_self(), thread, sizeof(thread));
        snprintf(line, sizeof(line), "  thread %s; mapping of RIP: ", thread);
        { ssize_t written_=write(2, line, strlen(line)); (void)written_; }
        FILE *maps = fopen("/proc/self/maps", "r");
        int found = 0;
        while (maps && fgets(line, sizeof(line), maps)) {
            unsigned long from, to;
            if (sscanf(line, "%lx-%lx", &from, &to) == 2 && rip >= from && rip < to) {
                ssize_t written_=write(2, line, strlen(line)); (void)written_;
                found = 1;
                break;
            }
        }
        if (maps) fclose(maps);
        if (!found) { ssize_t written_=write(2, "none\n", 5); (void)written_; }
    }
    /* Host call chain (frames with unwind info; guest frames end it). */
    void *frames[32];
    int depth = backtrace(frames, 32);
    for (int i = 2; i < depth; ++i) {
        if (dladdr(frames[i], &where) && where.dli_fname)
            snprintf(line, sizeof(line), "  #%d %s+0x%lx (%s)\n", i, where.dli_fname,
                     (unsigned long)((uintptr_t)frames[i] - (uintptr_t)where.dli_fbase), where.dli_sname ? where.dli_sname : "?");
        else
            snprintf(line, sizeof(line), "  #%d %p\n", i, frames[i]);
        ssize_t written_=write(2, line, strlen(line)); (void)written_;
    }
    _exit(128 + sig);
}
static uintptr_t exe_base;
#ifndef _WIN32
/* Watchdog: dump RIP and the rbp frame chain of every thread (guest offsets
 * when inside the image). Reads use process_vm_readv so bad frames cannot fault. */
static void write_hex(char *out, uint64_t v) {
    const char digits[] = "0123456789abcdef";
    for (int i = 0; i < 16; ++i) out[i] = digits[(v >> (60 - i * 4)) & 15];
}
#ifdef __APPLE__
#include <stddef.h>
_Static_assert(offsetof(struct __darwin_x86_thread_state64, __r15) == BB_REG_R15 * 8,
               "BB_REG_* follow Darwin's thread state");
#endif
static void dump_frames(ucontext_t *uc) {
    char line[] = "  tid=0000000000000000 rip=0000000000000000 image-relative=0000000000000000 host-relative=0000000000000000\n";
    uintptr_t rip=(uintptr_t)BB_CTX_RIP(uc), rbp=(uintptr_t)BB_CTX_RBP(uc);
    uint64_t tid=bb_thread_id();
    /* First argument register: the lock address when a thread waits on a futex. */
    char arg[]="  tid=0000000000000000 rdi=0000000000000000\n";
    write_hex(arg+6,tid); write_hex(arg+27,(uint64_t)BB_CTX_RDI(uc));
    { ssize_t written_=write(2,arg,sizeof(arg)-1); (void)written_; }
    /* All the registers and the top of the stack: what a thread waits for (a lock, a job, an event)
     * is in them; the hang of a game thread is read from here afterwards. */
    {
        static const struct { const char *name; int reg; } regs[]={
            {"rax",BB_REG_RAX},{"rbx",BB_REG_RBX},{"rcx",BB_REG_RCX},{"rdx",BB_REG_RDX},{"rsi",BB_REG_RSI},{"rdi",BB_REG_RDI},
            {"rbp",BB_REG_RBP},{"rsp",BB_REG_RSP},{"r8",BB_REG_R8},{"r9",BB_REG_R9},{"r10",BB_REG_R10},{"r11",BB_REG_R11},
            {"r12",BB_REG_R12},{"r13",BB_REG_R13},{"r14",BB_REG_R14},{"r15",BB_REG_R15},
        };
        for (size_t i=0;i<sizeof(regs)/sizeof(*regs);++i) {
            char r[]="  tid=0000000000000000 NNN=0000000000000000\n";
            write_hex(r+6,tid);
            r[23]=regs[i].name[0]; r[24]=regs[i].name[1]; r[25]=regs[i].name[2] ? regs[i].name[2] : ' ';
            write_hex(r+27,(uint64_t)BB_CTX_REG(uc,regs[i].reg));
            { ssize_t written_=write(2,r,sizeof(r)-1); (void)written_; }
        }
        /* What the pointer registers point at (guest addresses only): the 8 qwords there, so the
         * object a thread waits on can be read from the log. */
        static const int pointer_regs[]={BB_REG_RDI,BB_REG_RSI,BB_REG_RBX,BB_REG_R12,BB_REG_R13,BB_REG_R14,BB_REG_R15,BB_REG_RCX,BB_REG_RDX};
        for (size_t i=0;i<sizeof(pointer_regs)/sizeof(*pointer_regs);++i) {
            const uintptr_t p=(uintptr_t)BB_CTX_REG(uc,pointer_regs[i]);
            /* Guest memory: the low part below BB_LOW_MAX and the first 8 GiB of user memory
             * (on Linux the same 0x800000000-0x1200000000 as before; on the Mac user memory
             * starts at 0x7000000000, above Rosetta's carve-out). */
            if (!((p>=0x800000000ull && p<BB_LOW_MAX) || (p>=BB_USER_MIN && p<BB_USER_MIN+0x200000000ull))) continue;
            uint64_t words[8];
            struct iovec local={words,sizeof(words)}, remote={(void *)p,sizeof(words)};
            if (process_vm_readv(getpid(),&local,1,&remote,1,0)!=(ssize_t)sizeof(words)) continue;
            for (int k=0;k<8;++k) {
                char m[]="  tid=0000000000000000 at 0000000000000000+00=0000000000000000\n";
                write_hex(m+6,tid); write_hex(m+26,p);
                m[43]=(char)('0'+(k*8)/10%10); m[44]=(char)('0'+(k*8)%10);
                write_hex(m+46,words[k]);
                { ssize_t written_=write(2,m,sizeof(m)-1); (void)written_; }
            }
        }
        uintptr_t sp=(uintptr_t)BB_CTX_REG(uc,BB_REG_RSP);
        for (int i=0;i<24;++i) {
            uint64_t word=0;
            struct iovec local={&word,sizeof(word)}, remote={(void *)(sp+8*(uintptr_t)i),sizeof(word)};
            if (process_vm_readv(getpid(),&local,1,&remote,1,0)!=(ssize_t)sizeof(word)) break;
            char w[]="  tid=0000000000000000 stack+000=0000000000000000\n";
            write_hex(w+6,tid);
            w[29]=(char)('0'+(i*8)/100%10); w[30]=(char)('0'+(i*8)/10%10); w[31]=(char)('0'+(i*8)%10);
            write_hex(w+33,word);
            { ssize_t written_=write(2,w,sizeof(w)-1); (void)written_; }
        }
    }
    for (int depth=0; depth<24; ++depth) {
        write_hex(line+6,tid); write_hex(line+27,rip); write_hex(line+59,rip-(uintptr_t)image); write_hex(line+90,rip-exe_base);
        { ssize_t written_=write(2,line,sizeof(line)-1); (void)written_; }
        uintptr_t frame[2];
        struct iovec local={frame,sizeof(frame)}, remote={(void *)rbp,sizeof(frame)};
#ifdef __APPLE__
        (void)local; (void)remote;
        mach_vm_size_t got=0;
        if (!rbp || mach_vm_read_overwrite(mach_task_self(),(mach_vm_address_t)rbp,sizeof(frame),(mach_vm_address_t)(uintptr_t)frame,&got)!=KERN_SUCCESS || got!=sizeof(frame)) break;
#else
        if (!rbp || process_vm_readv(getpid(),&local,1,&remote,1,0)!=(ssize_t)sizeof(frame)) break;
#endif
        if (frame[0]<=rbp) break;
        rbp=frame[0]; rip=frame[1];
    }
}
static void thread_dump(int sig, siginfo_t *info, void *context) { (void)sig; (void)info; dump_frames(context); }
static void watchdog(int sig, siginfo_t *info, void *context) {
    (void)info;
    const char head[]="STOP: watchdog timeout; thread stacks:\n";
    { ssize_t written_=write(2,head,sizeof(head)-1); (void)written_; }
    dump_frames(context);
#ifdef __APPLE__
    /* macOS: no per-thread signal by thread id; only the watchdog thread's stack is dumped. */
#else
    int dir=open("/proc/self/task",O_RDONLY|O_DIRECTORY);
    char buffer[4096];
    long n;
    pid_t self=gettid();
    while (dir>=0 && (n=syscall(SYS_getdents64,dir,buffer,sizeof(buffer)))>0)
        for (long at=0; at<n;) {
            struct { uint64_t ino; int64_t off; unsigned short reclen; unsigned char type; char name[]; } *d=(void *)(buffer+at);
            pid_t tid=(pid_t)strtol(d->name,NULL,10);
            if (tid>0 && tid!=self) { syscall(SYS_tgkill,getpid(),tid,SIGUSR2); usleep(20000); }
            at+=d->reclen;
        }
#endif
    usleep(100000);
    _exit(128 + sig);
}
#endif
/* param.sfo lookup: string or integer value of key, 0 when absent. */
static int sfo_value(const char *path, const char *key, char *text, size_t text_size, uint32_t *number) {
    FILE *f=fopen(path,"rb");
    if (!f) return 0;
    unsigned char data[65536];
    size_t n=fread(data,1,sizeof(data),f); fclose(f);
    if (n<20 || memcmp(data,"\0PSF",4)) return 0;
    uint32_t keys, values, count;
    memcpy(&keys,data+8,4); memcpy(&values,data+12,4); memcpy(&count,data+16,4);
    for (uint32_t i=0;i<count && 20+i*16+16<=n;++i) {
        const unsigned char *e=data+20+i*16;
        uint16_t key_offset, format; uint32_t length, offset;
        memcpy(&key_offset,e,2); memcpy(&format,e+2,2); memcpy(&length,e+4,4); memcpy(&offset,e+12,4);
        if (keys+key_offset>=n || values+offset+length>n || strcmp((const char *)data+keys+key_offset,key)) continue;
        if (format==0x0404 && number && length>=4) { memcpy(number,data+values+offset,4); return 1; }
        if (text && text_size) {
            size_t copy=length<text_size-1 ? length : text_size-1;
            memcpy(text,data+values+offset,copy); text[copy]=0;
        }
        return 1;
    }
    return 0;
}
static int mapped(Segment *segments, uint64_t count, uint64_t address, uint64_t bytes) {
    for (uint64_t i = 0; i < count; ++i)
        if (address >= segments[i].address && bytes <= segments[i].size &&
            address - segments[i].address <= segments[i].size - bytes) return 1;
    return 0;
}
/* BBPATCH2 (patches.py): the patches' image base, then byte writes at image offsets, applied
 * after relocation. A write may replace a whole base-relative pointer slot (60/90 FPS++ swap
 * function pointers): the patch holds the address at the patches' base, rebased here. */
static void apply_patches(const char *path, Segment *segments, uint64_t ns, const Reloc *relocs, uint64_t nr) {
    FILE *f=fopen(path,"rb");
    char magic[8];
    if (!f || fread(magic,1,8,f)!=8 || memcmp(magic,"BBPATCH2",8)) fail("invalid patch file");
    uint64_t base=read64(f), count=read64(f), bytes=0, rebased=0;
    unsigned char data[4096];
    for (uint64_t i=0;i<count;++i) {
        uint64_t offset=read64(f), length=read64(f);
        if (!length || length>sizeof(data) || !mapped(segments,ns,offset,length) || fread(data,1,length,f)!=length)
            fail("bad patch entry");
        uint64_t slots[sizeof(data)/8]; size_t nslots=0;
        for (uint64_t r=0;r<nr;++r) {
            if (!(relocs[r].target<offset+length && offset<relocs[r].target+8)) continue;
            uint64_t target=relocs[r].target, value;
            if (relocs[r].kind || target<offset || target+8>offset+length) fail("patch overlaps a relocation");
            memcpy(&value,data+(target-offset),8);
            if (value<base || !mapped(segments,ns,value-base,1)) fail("patch writes a pointer outside the image");
            slots[nslots++]=target;
        }
        /* Hooks the GPU library installed before (at its start: bbport_guest_hooks) leave their
           pages read-only and executable; the image gets its final protection after this. */
        {
            const uint64_t first=offset & ~(uint64_t)(page_size-1);
            protect(image+first,(size_t)(bb_round_page(offset+length)-first),7);
        }
        memcpy(image+offset,data,length);
        for (size_t s=0;s<nslots;++s) {
            uint64_t value;
            memcpy(&value,image+slots[s],8);
            value=(uint64_t)(uintptr_t)image+(value-base);
            memcpy(image+slots[s],&value,8);
        }
        rebased+=nslots;
        bytes+=length;
    }
    if (fgetc(f)!=EOF) fail("trailing data in patch file");
    fclose(f);
    printf("Patches: %" PRIu64 " writes, %" PRIu64 " bytes applied, %" PRIu64 " pointers rebased\n",count,bytes,rebased);
}
/* Restarts the game through run.sh (the settings menu: a new render resolution is a patch
 * applied at start). Descriptors are closed first so the old GPU device and its memory are
 * released before the new process opens its own. */
volatile int runtime_restarting;
void runtime_restart(void) {
    fflush(NULL);
    puts("Runtime: restarting through run.sh");
    /* The GPU threads still run until exec: their Vulkan calls fail once the device fd is
     * closed below, and an assertion there must not end the process (exit 23) before exec. */
    runtime_restarting = 1;
    __sync_synchronize();
#ifdef _WIN32
    /* A new process with the same command line (the launcher's settings are in bbport.ini). */
    STARTUPINFOW startup = {.cb = sizeof(startup)};
    PROCESS_INFORMATION process;
    if (CreateProcessW(NULL, GetCommandLineW(), NULL, NULL, FALSE, 0, NULL, NULL, &startup, &process)) {
        CloseHandle(process.hThread); CloseHandle(process.hProcess);
    } else {
        fprintf(stderr, "runtime_restart: CreateProcess failed (%lu)\n", GetLastError());
    }
    _exit(0);
#else
#ifdef __APPLE__
    for (int fd=3, top=getdtablesize(); fd<top; ++fd) close(fd);
#else
    syscall(SYS_close_range, 3u, ~0u, 0u);
#endif
    execlp("bash", "bash", "run.sh", (char *)NULL);
    perror("runtime_restart: exec");
    _exit(1);
#endif
}

static int probe_main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
#ifndef _WIN32
    /* Keep host heap objects handed to the guest (thread handles, TLS) in the
       non-PIE brk heap, i.e. below 1 TiB: the guest packs pointers into 40 bits. */
#ifdef __APPLE__
    /* macOS: the runtime's allocations come from the low heap instead (src/low_heap.c). */
#else
    mallopt(M_ARENA_MAX,1);
    mallopt(M_MMAP_THRESHOLD,32*1024*1024);
#endif
#endif
    if (argc == 2 && !strcmp(argv[1], "--vulkan-only")) return vulkan_smoke();
    /* Shader recipes (gpu/render/shader_rebuild.h): no game here, the guest's addresses stay free. */
    if (argc == 5 && !strcmp(argv[1], "--export-shader-recipes"))
        return bbgpu_export_shader_recipes(argv[2], argv[3], argv[4]);
    if (argc == 7 && !strcmp(argv[1], "--import-shaders"))
        return bbgpu_import_shader_recipes(argv[2], argv[3], argv[4], strtoull(argv[5], NULL, 16),
                                           strtoull(argv[6], NULL, 16));
    if (argc == 3 && !strcmp(argv[1], "--rebuild-shader-cache"))
        return bbgpu_rebuild_shader_cache(argv[2]);
    if (argc == 9 && !strcmp(argv[1], "--merge-shader-metas"))
        return bbgpu_merge_shader_metas(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], atoi(argv[8]));
#ifndef _WIN32
    if (argc == 7 && !strcmp(argv[1], "--predict-shaders-test"))
        return bbgpu_predict_shaders_test(argv[2], argv[3], argv[4], argv[5], atoi(argv[6]));
    if (argc == 9 && !strcmp(argv[1], "--merge-shader-packs"))
        return bbgpu_merge_shader_packs(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], atoi(argv[8]));
    if (argc == 5 && !strcmp(argv[1], "--meta-reconstruct-test"))
        return bbgpu_meta_reconstruct_test(argv[2], argv[3], argv[4]);
#endif
    print_environment();
    check_cpu();
    /* Before the guest's memory exists: the recorded shader inputs are mapped back at their
     * addresses to translate them again for this build (render/shader_rebuild.h). */
    bbgpu_rebuild_shader_caches();
    runtime_memory_reserve();
    int cpu_only = 0, strict_imports = 0;
    unsigned timeout_seconds = 10;
    const char *content_profile=NULL, *app0=NULL, *user_dir=NULL, *patch_file=NULL;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--cpu-only")) cpu_only = 1;
        else if (!strcmp(argv[i], "--strict-imports")) strict_imports = 1;
        else if (!strcmp(argv[i], "--content-profile") && i+1<argc) content_profile=argv[++i];
        else if (!strcmp(argv[i], "--app0") && i+1<argc) app0=argv[++i];
        else if (!strcmp(argv[i], "--user") && i+1<argc) user_dir=argv[++i];
        else if (!strcmp(argv[i], "--patches") && i+1<argc) patch_file=argv[++i];
        else if (!strcmp(argv[i], "--timeout") && i+1<argc) timeout_seconds=(unsigned)strtoul(argv[++i],NULL,10);
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }
    if (argc < 2) {
        fprintf(stderr, "Usage: %s boot.bin [--cpu-only] [--strict-imports] [--content-profile file] [--app0 dir] [--user dir] [--patches file] [--timeout seconds] | --vulkan-only\n", argv[0]);
        return 1;
    }
    if (content_profile) {
        FILE *profile=fopen(content_profile,"rb");
        unsigned char data[28];
        if (!profile) fail("cannot open content profile");
        if (fread(data,1,sizeof(data),profile)!=sizeof(data) || fgetc(profile)!=EOF || memcmp(data,"BBCONT01",8)) fail("invalid content profile");
        fclose(profile);
        uint32_t values[5];
        for (unsigned i=0;i<5;++i) {
            unsigned char *p=data+8+i*4;
            values[i]=(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24;
        }
        runtime_content_configure(values);
    }
    if (app0) {
        runtime_file_configure(app0, user_dir ? user_dir : "user");
        char sfo[4096], id[16]="";
        snprintf(sfo,sizeof(sfo),"%s/sce_sys/param.sfo",app0);
        if (sfo_value(sfo,"INSTALL_DIR_SAVEDATA",id,sizeof(id),NULL) || sfo_value(sfo,"TITLE_ID",id,sizeof(id),NULL))
            runtime_savedata_configure(id);
        char serial[16]="", title[128]="", version[16]="";
        sfo_value(sfo,"TITLE_ID",serial,sizeof(serial),NULL);
        sfo_value(sfo,"TITLE",title,sizeof(title),NULL);
        sfo_value(sfo,"APP_VER",version,sizeof(version),NULL);
        runtime_net_configure(app0,serial,title,version);
        trophy_store_configure(serial);
    }
    bbgpu_register_kernel();
    page_size = (size_t)sysconf(_SC_PAGESIZE);
    struct sigaction sa = {0}; sa.sa_sigaction = fault; sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGSEGV, &sa, NULL); sigaction(SIGILL, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
    Dl_info self_info;
    if (dladdr((void *)probe_main,&self_info)) exe_base=(uintptr_t)self_info.dli_fbase;
#ifndef _WIN32
    struct sigaction dump = {0}; dump.sa_sigaction = thread_dump; dump.sa_flags = SA_SIGINFO|SA_RESTART;
    sigemptyset(&dump.sa_mask); sigaction(SIGUSR2, &dump, NULL);
    struct sigaction alarm_action = {0}; alarm_action.sa_sigaction = watchdog; alarm_action.sa_flags = SA_SIGINFO;
    sigemptyset(&alarm_action.sa_mask); sigaction(SIGALRM, &alarm_action, NULL);
    alarm(timeout_seconds); /* 0 disables the watchdog */
#endif
    FILE *f = fopen(argv[1], "rb");
    if (!f) fail("cannot open boot file; run prepare.py first");
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || (memcmp(magic, "BBPROBE1", 8) && memcmp(magic, "BBPROBE2", 8) && memcmp(magic,"BBPROBE3",8) && memcmp(magic,"BBPROBE4",8) && memcmp(magic,"BBPROBE5",8))) fail("bad boot file signature");
    uint64_t size = read64(f), entry = read64(f), ns = read64(f), nr = read64(f);
    import_count = read64(f);
    uint64_t capabilities = memcmp(magic, "BBPROBE1", 8) ? read64(f) : 0;
    if (capabilities & ~UINT64_C(1)) fail("unknown runtime capabilities");
    runtime_start(strict_imports ? 0 : capabilities);
    if (!size || size > 512*1024*1024 || entry >= size || !ns || ns > 64 || nr > 1000000 || import_count > 100000)
        fail("boot file limits exceeded");
    int multi=!memcmp(magic,"BBPROBE5",8);
    int linked=!memcmp(magic,"BBPROBE3",8) || !memcmp(magic,"BBPROBE4",8) || multi;
    int native_libc=linked && !strict_imports && (capabilities&1);
    uint64_t main_tls[4]={0};
    uint64_t nb=0,procparam=0;
    uint64_t *bindings=calloc(import_count ? import_count : 1,sizeof(*bindings));
    uint64_t *binding_kinds=calloc(import_count ? import_count : 1,sizeof(*binding_kinds));
    if (!bindings || !binding_kinds) fail("allocation failed");
    if (multi) {
        /* BBPROBE5: procparam, eboot TLS, module table, bindings (link_modules.py). */
        procparam=read64(f);
        for (int i=0;i<4;++i) main_tls[i]=read64(f);
        module_count=read64(f);
        if (!module_count || module_count>sizeof(modules)/sizeof(*modules)) fail("invalid module count");
        for (uint64_t m=0;m<module_count;++m) {
            LinkedModule *x=&modules[m];
            x->base=read64(f); x->size=read64(f); x->init=read64(f); x->tls_address=read64(f);
            x->tls_memsz=read64(f); x->tls_filesz=read64(f); x->tls_module=read64(f);
        }
        nb=read64(f);
    } else if (linked) {
        LinkedModule *x=&modules[0];
        module_count=1;
        x->base=read64(f); x->size=read64(f); x->init=read64(f);
        x->tls_address=read64(f); x->tls_memsz=read64(f); x->tls_filesz=read64(f); x->tls_module=2; nb=read64(f);
        procparam=read64(f);
        if (!memcmp(magic,"BBPROBE4",8)) for (int i=0;i<4;++i) main_tls[i]=read64(f);
    }
    if (linked) {
        if (main_tls[1]>main_tls[2] || main_tls[2]>1024*1024 || main_tls[0]>size ||
            main_tls[1]>size-main_tls[0] || (main_tls[3] & (main_tls[3]-1)) || main_tls[3]>4096)
            fail("invalid eboot TLS metadata");
        for (uint64_t m=0;m<module_count;++m) {
            LinkedModule *x=&modules[m];
            if (x->base>=size || !x->size || x->size>size-x->base || x->init<x->base || x->init-x->base>=x->size ||
                (x->tls_module && (x->tls_address>=size || x->tls_memsz>size-x->tls_address || x->tls_memsz>1024*1024 ||
                                   x->tls_filesz>x->tls_memsz || x->tls_module<2 || x->tls_module>7)))
                fail("invalid linked module metadata");
        }
        if (nb>import_count) fail("invalid binding count");
        for (uint64_t i=0;i<nb;++i) {
            uint64_t index=read64(f),address=read64(f),kind=read64(f);
            int inside=0;
            for (uint64_t m=0;m<module_count;++m)
                if (address>=modules[m].base && address-modules[m].base<modules[m].size) inside=1;
            if (index>=import_count || !inside || (kind!=1 && kind!=2) || bindings[index]) fail("invalid native binding");
            bindings[index]=address; binding_kinds[index]=kind;
        }
    }
    Segment *segments = calloc(ns, sizeof(*segments));
    Reloc *relocs = calloc(nr ? nr : 1, sizeof(*relocs));
    names = calloc(import_count ? import_count : 1, sizeof(*names));
    if (!segments || !relocs || !names) fail("allocation failed");
    for (uint64_t i = 0; i < ns; ++i) {
        segments[i].address = read64(f);
        segments[i].size = read64(f);
        segments[i].flags = read64(f);
        if (segments[i].address > size || segments[i].size > size - segments[i].address ||
            segments[i].address % page_size || segments[i].flags > 7) fail("bad segment");
    }
    if (fread(names, 128, import_count, f) != import_count) fail("truncated import names");
    for (uint64_t i = 0; i < import_count; ++i)
        if (!memchr(names[i], 0, 128)) fail("unterminated import name");
    for (uint64_t i = 0; i < nr; ++i) {
        relocs[i].target = read64(f);
        relocs[i].kind = read64(f);
        relocs[i].value = read64(f);
        relocs[i].addend = read64(f);
        if (!mapped(segments, ns, relocs[i].target, 8) || relocs[i].kind > 2 ||
            (relocs[i].kind && relocs[i].value >= import_count) ||
            (relocs[i].kind != 2 && relocs[i].addend) || relocs[i].addend >= page_size) fail("bad relocation");
    }
    for (uint64_t m=0;m<module_count;++m)
        if (!mapped(segments,ns,modules[m].init,1) || (modules[m].tls_module && !mapped(segments,ns,modules[m].tls_address,modules[m].tls_memsz)))
            fail("unmapped module metadata");
    if (module_count && !mapped(segments,ns,procparam,64)) fail("unmapped procparam");
    for (uint64_t i=0;i<import_count;++i) {
        if (!bindings[i]) continue;
        if (!mapped(segments,ns,bindings[i],1)) fail("unmapped native export");
        if (binding_kinds[i]==1) {
            int executable=0;
            for (uint64_t s=0;s<ns;++s)
                if ((segments[s].flags&1) && bindings[i]>=segments[s].address &&
                    bindings[i]-segments[s].address<segments[s].size) executable=1;
            if (!executable) fail("native function is not executable");
        }
    }
    image = allocate(bb_round_page(size));
    if (fread(image, 1, size, f) != size || fgetc(f) != EOF) fail("incorrect memory image size");
    fclose(f);
    if (!cpu_only) {
        char title[128]="Bloodborne", serial[16]="UNKNOWN", sfo[4096];
        uint32_t attributes=0;
        snprintf(sfo,sizeof(sfo),"%s/sce_sys/param.sfo",app0 ? app0 : ".");
        sfo_value(sfo,"TITLE",title,sizeof(title),NULL);
        sfo_value(sfo,"TITLE_ID",serial,sizeof(serial),NULL);
        sfo_value(sfo,"ATTRIBUTE",NULL,0,&attributes);
        uint64_t sdk=0;
        if (procparam) memcpy(&sdk,image+procparam+16,8); /* procparam: size, magic, count, sdk_version */
        BbGpuConfig gpu={title,serial,user_dir ? user_dir : "user",(uint32_t)sdk,attributes,1920,1080};
        gpu_enabled=1; /* page tracking starts while the rasterizer registers guest memory */
        if (bbgpu_init(&gpu)) fail("GPU initialization failed");
        printf("GPU: window and Vulkan presenter ready; SDK 0x%08x, %u HLE symbols\n",(unsigned)sdk,bbgpu_symbol_count());
    }
    unsigned char *traps = allocate(bb_round_page((import_count + 1) * 32));
    unsigned char *data_traps = allocate((import_count + 1) * page_size);
    protect(data_traps, (import_count + 1) * page_size, 0);
    for (uint64_t i = 0; i < import_count; ++i) {
        unsigned char *t = traps + i * 32;
        /* SysV: mov edi, index; movabs rax, handler; jmp rax. No fake return values. */
        uint32_t index = (uint32_t)i;
        uintptr_t handler = (uintptr_t)unresolved;
        t[0] = 0x48; t[1] = 0x89; t[2] = 0xfe; /* mov rsi,rdi: preserve arg0 */
        t[3] = 0xbf; memcpy(t + 4, &index, 4);
        t[8] = 0x48; t[9] = 0xb8; memcpy(t + 10, &handler, 8);
        t[18] = 0xff; t[19] = 0xe0;
    }
    /* BB_LIST_UNRESOLVED=1 (porting another game): every import neither the runtime nor a
     * native module provides, once each, before the game runs. */
    unsigned char *listed = bb_env_on("BB_LIST_UNRESOLVED") ? calloc(import_count + 1, 1) : NULL;
    uint64_t unresolved_count = 0;
    for (uint64_t i = 0; i < nr; ++i) {
        uintptr_t value = relocs[i].kind == 2 ? (uintptr_t)(data_traps + page_size * relocs[i].value + relocs[i].addend)
                        : relocs[i].kind == 1 ? (uintptr_t)(traps + 32 * relocs[i].value)
                        : (uintptr_t)image + relocs[i].value;
        if (relocs[i].kind) {
            uintptr_t resolved = runtime_resolve(names[relocs[i].value], relocs[i].kind == 2);
            if (resolved) value = resolved + relocs[i].addend;
            else if (native_libc && bindings[relocs[i].value]) {
                uint64_t address=bindings[relocs[i].value];
                if (binding_kinds[relocs[i].value]!=relocs[i].kind || !mapped(segments,ns,address,relocs[i].addend+1)) fail("native export kind/range mismatch");
                value=(uintptr_t)image+address+relocs[i].addend;
            } else if (listed && !listed[relocs[i].value]) {
                listed[relocs[i].value] = 1;
                ++unresolved_count;
                printf("Unresolved import: %s%s\n", names[relocs[i].value],
                       relocs[i].kind == 2 ? " (data)" : "");
            }
        }
        memcpy(image + relocs[i].target, &value, 8);
    }
    if (listed) {
        printf("Unresolved imports: %" PRIu64 "\n", unresolved_count);
        free(listed);
    }
    runtime_net_image_mapped(image, size); /* online modules that read the original code */
    if (patch_file) apply_patches(patch_file, segments, ns, relocs, nr);
#ifdef __APPLE__
    {
        /* bbport 0.6 installs the hooks in the game's code with either memory model, at the GPU's
           start (bbport_guest_hooks), and they leave their pages read-only and executable: as
           apply_patches does, the code is made writable for this; the final protection below. */
        size_t tls_sites=0;
        for (uint64_t i = 0; i < ns; ++i)
            if (segments[i].flags & 1) {
                const uint64_t first=segments[i].address & ~(uint64_t)(page_size-1);
                protect(image+first,(size_t)(bb_round_page(segments[i].address+segments[i].size)-first),7);
                tls_sites += runtime_tls_fixup(image + segments[i].address, segments[i].size);
            }
        printf("TLS: %zu guest TCB loads redirected to the pthread TSD slot\n", tls_sites);
    }
#endif
    if (!cpu_only) bbgpu_patch_image(image, size);
    /* bbport (native GPU process): the GPU reads some of the game's own data
     * (resource tables in its data segments), so those move into memory the GPU process shares.
     * Only segments whose 16 KiB pages hold nothing else; before the protection below (all of it
     * is still readable and writable here), which then applies to the shared pages. */
#ifndef _WIN32
    if (!cpu_only) {
        const uint64_t share_page = 16384;
        for (uint64_t i = 0; i < ns; ++i) {
            const uint64_t start = segments[i].address, end = (start + segments[i].size + share_page - 1) & ~(share_page - 1);
            if ((segments[i].flags & 1) || !segments[i].size || start % share_page) continue;
            int alone = 1;
            for (uint64_t j = 0; j < ns; ++j)
                if (j != i && segments[j].size && segments[j].address < end && start < segments[j].address + segments[j].size) alone = 0;
            if (!alone) continue;
            if (bbgpu_share_range(image + start, end - start, PROT_READ | PROT_WRITE))
                fail("cannot share the game's data with the GPU process");
        }
    }
#endif
#ifdef _WIN32
    patch_guest_tls(segments, ns);
#endif
    /* bbport (macOS): bb_round_page: <mach/mach.h> defines round_page. */
    protect(traps, bb_round_page((import_count + 1) * 32), 5);
    protect(image, bb_round_page(size), 0);
    int executable_entry = 0;
    for (uint64_t i = 0; i < ns; ++i) {
        protect(image + segments[i].address, bb_round_page(segments[i].size), (unsigned)segments[i].flags);
        if ((segments[i].flags & 1) && entry >= segments[i].address && entry - segments[i].address < segments[i].size)
            executable_entry = 1;
    }
    if (!executable_entry) fail("entry is not executable");
    printf("Mapped %" PRIu64 " bytes, %" PRIu64 " segments; applied %" PRIu64 " relocations\n", size, ns, nr);
    runtime_net_image_ready((uintptr_t)image, size); /* online game modules (BBSC) hook it now */
    if (native_libc) {
        for (uint64_t m=0;m<module_count;++m) {
            int init_executable=0;
            for (uint64_t i=0;i<ns;++i)
                if ((segments[i].flags&1) && modules[m].init>=segments[i].address && modules[m].init-segments[i].address<segments[i].size) init_executable=1;
            if (!init_executable) fail("module init is not executable");
            if (modules[m].tls_module)
                runtime_set_module_tls(modules[m].tls_module,image+modules[m].tls_address,modules[m].tls_filesz,modules[m].tls_memsz);
        }
        runtime_set_main_tls(image+main_tls[0],main_tls[1],main_tls[2],main_tls[3]);
        if (module_count) runtime_libc_base=(uintptr_t)image+modules[0].base;
        runtime_thread_attach_main();
        runtime_set_procparam(image+procparam);
        /* Dependencies start in link order (libc first), as the PS4 dynamic linker does. */
        for (uint64_t m=0;m<module_count;++m) {
            printf("Starting linked module %" PRIu64 " at image offset 0x%" PRIx64 "; native bindings=%" PRIu64 "\n",m,modules[m].init,nb);
            typedef int (ABI *ModuleInit)(uint64_t,void *,void *);
            int result=((ModuleInit)(image+modules[m].init))(0,NULL,NULL);
            printf("Module %" PRIu64 " initializer returned %d\n",m,result);
            if (result) fail("module initializer failed");
        }
    }
    runtime_net_start(); /* online sessions start with the game */
    if (bb_env("BB_LOW_INFO_TEST")) {
        /* What the online module's memory queries get for the game's statics (#282): the local
         * player's and the global state's pointers, code, and the end of the image. */
        const uint64_t offsets[]={0x553E878,0x5556678,0x553B130,0x5540300,0x1000,0x2000000};
        for (size_t i=0;i<sizeof(offsets)/sizeof(*offsets);++i) {
            int prot=0; uintptr_t end=0;
            const int ok=runtime_memory_low_info((uintptr_t)image+offsets[i],&prot,&end);
            printf("Low info: image+%#" PRIx64 ": %s prot %d end image+%#" PRIxPTR "\n",offsets[i],
                   ok ? "mapped" : "NOT mapped",prot,ok ? end-(uintptr_t)image : 0);
        }
    }
    printf("Entering original x86-64 code at guest offset 0x%" PRIx64 "\n", entry);
    entered_game=1;
    struct { uint64_t argc; const char *argv[2]; } params = {1, {"/app0/eboot.bin", NULL}};
#ifdef _WIN32
    enum { MAIN_STACK=8*1024*1024 };
    unsigned char *stack=runtime_low_map(MAIN_STACK,PROT_READ|PROT_WRITE);
    if (!stack) fail("cannot allocate guest main stack");
    static struct { void *entry, *params; } start;
    start.entry=image+entry; start.params=&params;
    bb_run_on_stack(enter_game,&start,stack,stack+MAIN_STACK-64);
#else
    /* The guest main thread runs on a stack below 1 TiB like PS4 stacks. */
    enum { MAIN_STACK=8*1024*1024 };
    unsigned char *stack=runtime_low_map(MAIN_STACK,PROT_READ|PROT_WRITE);
    if (!stack) fail("cannot allocate guest main stack");
    enter_on_stack(image+entry,&params,(void *)guest_exit,stack+MAIN_STACK-64);
#endif
    fail("entry unexpectedly returned");
}

#ifdef __APPLE__
/* macOS: Cocoa windows exist only on the main thread. The loader and the game's own main thread
 * run on a secondary thread with a main-sized stack; the real main thread serves the GPU
 * library's window (bbgpu_main_thread_loop). Whichever side ends the process calls exit. */
typedef struct { int argc; char **argv; } MainArgs;
static void *probe_thread(void *arg) {
    const MainArgs *args = arg;
    /* bbport (macOS): named for the CPU-by-thread report, where it counts as the game's (a
     * "bb-probe" prefix would count as the port's: bbport_thread_cpu_mac.cpp, Owner). */
    bb_set_thread_name("GameMain");
    exit(probe_main(args->argc, args->argv));
}
/* bbport (macOS): the performance and efficiency cores, for logs from small Macs (4P+4E): the
 * game under Rosetta and the GPU side's threads share them. */
static void log_core_counts(void) {
    int perf = 0, eff = 0, logical = 0;
    size_t size = sizeof(int);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &size, NULL, 0) != 0) perf = 0;
    size = sizeof(int);
    if (sysctlbyname("hw.perflevel1.physicalcpu", &eff, &size, NULL, 0) != 0) eff = 0;
    size = sizeof(int);
    if (sysctlbyname("hw.logicalcpu", &logical, &size, NULL, 0) != 0) logical = 0;
    if (perf > 0) printf("CPU: %d performance cores, %d efficiency cores (%d logical)\n", perf, eff, logical);
    else printf("CPU: %d logical cores (no performance levels)\n", logical);
}
int main(int argc, char **argv) {
    static MainArgs args;
    args.argc = argc; args.argv = argv;
    /* stdout is a pipe (run.sh tees it to out/last-run.log): line-buffer it so diagnostics
     * printed just before a crash reach the log. */
    setvbuf(stdout, NULL, _IOLBF, 0);
    log_core_counts();
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, (size_t)64 << 20);
    pthread_t thread;
    if (pthread_create(&thread, &attr, probe_thread, &args)) { perror("pthread_create"); return 1; }
    pthread_attr_destroy(&attr);
    bbgpu_main_thread_loop();
    return 0;
}
#else
int main(int argc, char **argv) { return probe_main(argc, argv); }
#endif
