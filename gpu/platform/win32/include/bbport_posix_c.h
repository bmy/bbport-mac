// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: the POSIX file and time calls of the runtime (src/, C) with Linux semantics
// (posix_win32.cpp): struct stat with st_mtim/st_blocks, dirent with d_type, directory fds
// (O_DIRECTORY, openat, fstatat, fdopendir, dirfd), pread/pwrite at an offset without moving
// the file position, rename over an existing file, binary mode always. Paths are UTF-8 (the
// executable's manifest sets the UTF-8 code page). Macros rename the calls: C only, after the
// UCRT headers that declare the originals.
#pragma once
#ifndef __cplusplus

#include <limits.h>
#include <stdlib.h>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <process.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>

#ifndef PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP
#define PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP PTHREAD_RECURSIVE_MUTEX_INITIALIZER
#endif

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif
#ifndef O_NONBLOCK
#define O_NONBLOCK 0
#endif
#ifndef O_SYNC
#define O_SYNC 0
#endif
#ifndef O_DIRECTORY
#define O_DIRECTORY 0x200000
#endif
#ifndef O_NOFOLLOW
#define O_NOFOLLOW 0
#endif
#define AT_FDCWD (-100)
#define AT_SYMLINK_NOFOLLOW 0x100
#define AT_REMOVEDIR 0x200

struct bb_stat {
    uint64_t st_dev, st_ino;
    uint32_t st_mode, st_nlink, st_uid, st_gid;
    uint64_t st_rdev;
    int64_t st_size, st_blksize, st_blocks;
    struct timespec st_atim, st_mtim, st_ctim;
};
int bb_stat(const char* path, struct bb_stat* out);
int bb_fstat(int fd, struct bb_stat* out);
int bb_fstatat(int dir, const char* path, struct bb_stat* out, int flags);
#define stat bb_stat
#define lstat bb_stat
#define fstat bb_fstat
#define fstatat bb_fstatat

int bb_open(const char* path, int flags, ...);
int bb_openat(int dir, const char* path, int flags, ...);
int bb_close(int fd);
ssize_t bb_pread(int fd, void* buffer, size_t size, int64_t offset);
ssize_t bb_pwrite(int fd, const void* buffer, size_t size, int64_t offset);
int bb_fsync(int fd);
int bb_ftruncate(int fd, int64_t size);
int bb_rename(const char* from, const char* to);
int bb_mkdir(const char* path, int mode);
int bb_unlinkat(int dir, const char* path, int flags);
int bb_mkdirat(int dir, const char* path, int mode);
int bb_renameat(int from_dir, const char* from, int to_dir, const char* to);
ssize_t bb_readlink(const char* path, char* out, size_t size);
#define open bb_open
#define openat bb_openat
#define close bb_close
#define pread bb_pread
#define pwrite bb_pwrite
#define fsync bb_fsync
#define ftruncate bb_ftruncate
#define rename bb_rename
#define mkdir(path, mode) bb_mkdir(path, mode)
#define unlinkat bb_unlinkat
#define mkdirat bb_mkdirat
#define renameat bb_renameat
#define readlink bb_readlink

#define DT_UNKNOWN 0
#define DT_DIR 4
#define DT_REG 8
#define DT_LNK 10
struct bb_dirent {
    uint64_t d_ino;
    unsigned char d_type;
    char d_name[1024];
};
typedef struct bb_DIR bb_DIR;
bb_DIR* bb_opendir(const char* path);
bb_DIR* bb_fdopendir(int fd);
struct bb_dirent* bb_readdir(bb_DIR* dir);
int bb_closedir(bb_DIR* dir);
int bb_dirfd(bb_DIR* dir);
void bb_rewinddir(bb_DIR* dir);
#define dirent bb_dirent
#define DIR bb_DIR
#define opendir bb_opendir
#define fdopendir bb_fdopendir
#define readdir bb_readdir
#define closedir bb_closedir
#define dirfd bb_dirfd
#define rewinddir bb_rewinddir

static inline struct tm* localtime_r(const time_t* t, struct tm* out) {
    return localtime_s(out, t) == 0 ? out : NULL;
}
static inline struct tm* gmtime_r(const time_t* t, struct tm* out) {
    return gmtime_s(out, t) == 0 ? out : NULL;
}
/// tm_gmtoff (east of UTC, seconds) of a local time.
long bb_gmtoff(const struct tm* local);

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif
static inline char* realpath(const char* path, char* out) {
    return _fullpath(out, path, PATH_MAX);
}

/// An anonymous file (a temporary one, deleted when closed).
#define MFD_CLOEXEC 1
int memfd_create(const char* name, unsigned flags);

/// The calling thread's stack [low, high) (pthread_getattr_np's use here).
void bb_thread_stack(uintptr_t* low, uintptr_t* high);

/// C11 aligned_alloc: the UCRT has none. Its blocks are released with _aligned_free (the
/// runtime's are never released: guest threads' TLS).
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)

/// Guest TLS (runtime_thread.c): the loader rewrote the game's `mov r64, fs:[0]` to read a TEB
/// TLS slot (gs:[0x1480 + 8 * slot]); each guest thread stores its TCB there.
int bb_guest_tls_slot(void);
/* bbport: the calling thread's name for Windows tools (SetThreadDescription; winpthreads keeps it to itself). */
void bb_win32_set_thread_name(const char* name);
/* bbport (diagnostics): VirtualQuery of an address as text: state, protection, type, allocation. */
void bb_win32_describe_address(const void* address, char* text, unsigned size);
void bb_guest_tls_set(void* tcb);

/// getrandom (sys/random.h): from the system's generator.
ssize_t getrandom(void* buffer, size_t size, unsigned flags);

#endif // !__cplusplus
