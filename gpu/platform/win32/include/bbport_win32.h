// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: included first in every GPU library source (CMake -include). The few libc
// functions glibc has and the UCRT does not, and sigsetjmp (bbport_jmp.h).
#pragma once

// 64-bit off_t (MinGW's is 32-bit otherwise): guest memory offsets pass 2 GiB.
#ifndef _FILE_OFFSET_BITS
#define _FILE_OFFSET_BITS 64
#endif
// Windows 10 1803 and later: placeholders (VirtualAlloc2, MapViewOfFile3) for guest memory.
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef NTDDI_VERSION
#define NTDDI_VERSION 0x0A000005
#endif
// rand_s (stdlib.h, before anything includes it).
#ifndef _CRT_RAND_S
#define _CRT_RAND_S
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <errno.h>
#include <wchar.h>
#ifdef __cplusplus
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <span>
#endif

#include "bbport_jmp.h"
#include "bbport_signal.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline char* strcasestr(const char* haystack, const char* needle) {
    const size_t n = strlen(needle);
    for (; *haystack; ++haystack) {
        if (_strnicmp(haystack, needle, n) == 0) return (char*)(size_t)haystack;
    }
    return n == 0 ? (char*)(size_t)haystack : NULL;
}

/// GNU memmem (the UCRT has none).
static inline void* memmem(const void* haystack, size_t size, const void* needle, size_t length) {
    if (length == 0) return (void*)(size_t)haystack;
    const unsigned char* h = (const unsigned char*)haystack;
    for (size_t i = 0; i + length <= size; ++i) {
        if (h[i] == *(const unsigned char*)needle && memcmp(h + i, needle, length) == 0) return (void*)(h + i);
    }
    return NULL;
}

/// The GNU variant: the message, not an error code.
static inline char* bb_strerror_r(int error, char* buffer, size_t size) {
    if (strerror_s(buffer, size, error) != 0 && size) buffer[0] = 0;
    return buffer;
}
#define strerror_r bb_strerror_r

static inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value) == 0 ? 0 : -1;
}
static inline int unsetenv(const char* name) {
    return _putenv_s(name, "") == 0 ? 0 : -1;
}
/// Waits for a signal: here, forever (the debugger's wait in common/assert.cpp).
int pause(void);

/// The calling thread's id (GetCurrentThreadId).
int gettid(void);

#define _SC_PAGESIZE 30
#define _SC_PAGE_SIZE _SC_PAGESIZE
#define _SC_NPROCESSORS_ONLN 84
/// Page size (4 KiB; VirtualAlloc reserves in 64 KiB granules) and the processor count.
long sysconf(int name);

/// Threads (bbport_threads.h): the processors this process may run on; the calling thread at
/// idle priority.
/// A pagefile-backed section of `size` bytes mapped twice (both views see the same memory);
/// munmap unmaps either.
int bb_win32_shared_views(size_t size, void** first, void** second);
unsigned bb_win32_available_cpus(void);
void bb_win32_background_thread(void);
/// BbThreads::MakeCritical: 1 above normal, 2 highest.
void bb_win32_critical_thread(int level);

#ifndef _countof
#define _countof(array) (sizeof(array) / sizeof((array)[0]))
#endif
/// MSVC's bounds-checked swprintf: the standard one takes the same arguments.
#ifndef swprintf_s
#define swprintf_s swprintf
#endif

#ifdef __cplusplus
}
#endif

// The runtime's (C) file and time calls with Linux semantics.
#include "bbport_posix_c.h"

// clock_nanosleep: winpthreads takes CLOCK_REALTIME only (EINVAL at once for CLOCK_MONOTONIC), so
// the audio pacing loop and the vblank timer spun a core each on Windows (GTX 1050 laptop: 72 %
// and 71 % of a core of 4). posix_win32.cpp: any clock, on a high-resolution waitable timer.
#include <time.h>
#include <pthread_time.h>
#ifdef __cplusplus
extern "C" {
#endif
int bb_clock_nanosleep(clockid_t clock, int flags, const struct timespec* request,
                       struct timespec* remain);
#ifdef __cplusplus
}
#endif
#define clock_nanosleep bb_clock_nanosleep
