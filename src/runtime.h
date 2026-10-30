// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#ifndef BB_RUNTIME_H
#define BB_RUNTIME_H
#include <string.h>
#include "../gpu/shim/bb_env.h" /* bbport: runtime_frame_stats */
#include <time.h>
#include <stdint.h>
#include <stddef.h>
#include "low_heap.h"
#include "platform.h" /* bbport (macOS): host-OS differences */
#include <setjmp.h>
/* Recovery point for speculative guest memory reads on this thread (probe.c fault handler). */
extern __thread sigjmp_buf *runtime_fault_recover;
/* Restarts the game (in-game settings menu, render resolution change). */
void runtime_restart(void);
#define ABI __attribute__((sysv_abi))
/* The clock of the runtime's condition variable deadlines: monotonic on Linux; winpthreads'
 * conditions take only CLOCK_REALTIME deadlines, and so do macOS's (no pthread_condattr_setclock;
 * platform.h accepts CLOCK_REALTIME there). */
#if defined(_WIN32) || defined(__APPLE__)
#define RUNTIME_COND_CLOCK CLOCK_REALTIME
#else
#define RUNTIME_COND_CLOCK CLOCK_MONOTONIC
#endif
/* A local time's offset east of UTC, seconds (Windows' struct tm has no tm_gmtoff). */
#ifdef _WIN32
#define RUNTIME_TM_GMTOFF(t) bb_gmtoff(&(t))
#else
#define RUNTIME_TM_GMTOFF(t) ((t).tm_gmtoff)
#endif
typedef void (ABI *GuestCallback)(void);
void runtime_start(uint64_t capabilities);
uintptr_t runtime_resolve(const char *name, int is_data);
void runtime_report(void);
void runtime_finalize(void *dso);
uintptr_t runtime_mutex_resolve(const char *name);
void runtime_mutex_report(void);
uintptr_t runtime_memory_resolve(const char *name);
void runtime_memory_report(void);
/* Reserves the guest's address range (Windows; probe.c, first thing). */
void runtime_memory_reserve(void);
int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
/* The CPU is about to write the range outside guest code (a file read): tells the GPU side. */
void runtime_memory_note_write(uintptr_t address, uint64_t size);
void runtime_memory_note_cpu_write(uintptr_t address, uint64_t size);
void runtime_memory_prepare_cpu_write(uintptr_t address, uint64_t size);
/* bbport (frame stats): a guest thread was blocked `ns` in the runtime (0 cond, 1 mutex, 2 sema, 3 sleep). */
void runtime_wait_note(int kind, uint64_t ns);
void runtime_wait_report(double frames);
/* bbport: BB_FRAME_STATS set (macOS: 1, y or t, the GPU side's rule there, BbStats::enabled and
 * FrameStatsEnabled; the readers here took any value, even 0, for on). */
static inline int runtime_frame_stats(void) {
    static int enabled=-1;
#ifdef __APPLE__
    if (enabled<0) { const char *e=bb_env("BB_FRAME_STATS"); enabled=e && (e[0]=='1' || e[0]=='y' || e[0]=='t'); }
#else
    if (enabled<0) enabled=bb_env("BB_FRAME_STATS")!=NULL;
#endif
    return enabled;
}
void runtime_guest_call_sites(uint64_t out[3]);
/* bbport: times the game's heap asked for more memory (posix_mmap): a sign it leaks. */
uint64_t runtime_heap_growths(void);
static inline uint64_t runtime_wait_clock(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (uint64_t)t.tv_sec*1000000000+(uint64_t)t.tv_nsec; }
const char *runtime_import_name(const char *name);
uintptr_t runtime_rwlock_resolve(const char *name);
void runtime_rwlock_report(void);
void runtime_set_libc_tls(const void *data, uint64_t filesz, uint64_t memsz);
void runtime_set_module_tls(uint64_t module, const void *data, uint64_t filesz, uint64_t memsz);
void runtime_set_procparam(void *param);
void **runtime_application_heap_api(void);
uintptr_t runtime_thread_resolve(const char *name);
void runtime_thread_report(void);
void runtime_set_main_tls(const void *data, uint64_t filesz, uint64_t memsz, uint64_t align);
#ifdef __APPLE__
size_t runtime_tls_fixup(unsigned char *code, size_t size);
#endif
void runtime_thread_attach_main(void);
int32_t *runtime_errno(void);
uintptr_t runtime_sema_resolve(const char *name);
void runtime_sema_report(void);
unsigned runtime_sema_waiters(uint32_t id);
uintptr_t runtime_time_resolve(const char *name);
void runtime_content_configure(const uint32_t values[5]);
uintptr_t runtime_content_resolve(const char *name);
void runtime_content_report(void);
/* An import names `nid`: the NID alone decides (the hash of the symbol's name), whatever library
 * letters the game's import carries (pthread_mutex_lock is libScePosix's in one game, libkernel's
 * in another). */
static inline int runtime_nid_is(const char *name, const char *nid) {
    const size_t length = strcspn(name, "#");
    return length == strlen(nid) && !strncmp(name, nid, length);
}
typedef struct { const char *name; void *function; } RuntimeExport;
#define RUNTIME_LOOKUP(table, nid) runtime_lookup(table, sizeof(table)/sizeof(*(table)), nid)
const char *runtime_symbol(const char *nid);
uintptr_t runtime_lookup(const RuntimeExport *table, size_t count, const char *nid);
uintptr_t runtime_kernel_resolve(const char *name);
uintptr_t runtime_file_resolve(const char *name);
uintptr_t runtime_services_resolve(const char *name);
/* Online play (runtime_net.c, BB_ONLINE=1): the online module answers the network and PSN imports. */
void runtime_net_configure(const char *app0, const char *serial, const char *title, const char *version);
int runtime_net_enabled(void);
int runtime_net_symbol(const char *symbol);
uintptr_t runtime_net_resolve(const char *name);
void runtime_net_image_ready(uintptr_t base, uint64_t size);
void runtime_net_image_mapped(uint8_t *image, uint64_t size);
void runtime_net_start(void);
__attribute__((noreturn)) void runtime_thread_exit(void *value);
/* Guest threads for host code (the online module's NP threads); 0 or a positive errno value. */
int32_t runtime_thread_spawn(void **thread, void *(ABI *entry)(void *), void *argument, uint64_t stack, const char *name);
int32_t runtime_thread_join_spawned(void *thread, void **result);
void runtime_file_report(void);
void runtime_file_configure(const char *app0, const char *user);
int runtime_file_mount(const char *guest, const char *host);
void runtime_file_unmount(const char *guest);
int runtime_file_translate(const char *guest, char *out, size_t size);
int64_t runtime_file_open(const char *path, int flags, int mode);
int64_t runtime_file_close(int fd);
int64_t runtime_file_read(int fd, void *buffer, uint64_t size);
int64_t runtime_file_pread(int fd, void *buffer, uint64_t size, int64_t offset);
int64_t runtime_file_write(int fd, const void *buffer, uint64_t size);
int64_t runtime_file_lseek(int fd, int64_t offset, int whence);
int64_t runtime_file_stat(const char *path, void *guest_stat);
int64_t runtime_file_fstat(int fd, void *guest_stat);
int64_t runtime_file_getdents(int fd, char *buffer, uint64_t size, int64_t *base);
void runtime_thread_keys_cleanup(void);
/* Guest-visible errno values are FreeBSD's. */
int32_t runtime_guest_errno(int host_errno);
void *runtime_low_map(size_t size, int prot);
void *runtime_low_map_high(size_t size, int prot); /* macOS low heap: top of the low region */
/* bbport: the loader changed the protection of part of the low region (mmap PROT_ bits). */
void runtime_low_protect(uintptr_t start, size_t size, int prot);
/* bbport: the low region's mapping holding address: 1 with its PS4 protection bits and end. */
int runtime_memory_low_info(uintptr_t address, int *prot, uintptr_t *end);
uintptr_t runtime_ajm_resolve(const char *name);
void runtime_ajm_report(void);
uintptr_t runtime_audio_resolve(const char *name);
void runtime_audio_report(void);
uintptr_t runtime_pad_resolve(const char *name);
void runtime_pad_report(void);
uintptr_t runtime_rtc_resolve(const char *name);
const char *runtime_file_user_dir(void);
/* A guest path ("/app0/...") as the host file it maps to: 0, or an errno. */
int runtime_file_host_path(const char *guest,char *out,size_t size);
/* runtime_remote.c: the game's files fetched from a phone over adb on first open (BB_APP0_ADB). */
void runtime_remote_configure(const char *app0);
int runtime_remote_fetch(const char *host);
int runtime_remote_size(const char *host,int64_t *size);
/* The character save the game uses: its slot (userdataNNNN, 0..9) and host folder; -1 before any. */
int runtime_savedata_active_slot(char *dir, size_t size);
void runtime_savedata_configure(const char *title);
uintptr_t runtime_savedata_resolve(const char *name);
void runtime_savedata_report(void);
void runtime_thread_attach_host(const char *name);
/* The game's libc.prx where it was linked (image + module base), 0 before or without it. */
extern uintptr_t runtime_libc_base;
/* Calls that need no work here (runtime_compat.c). */
uintptr_t runtime_compat_resolve(const char *name);
#endif
