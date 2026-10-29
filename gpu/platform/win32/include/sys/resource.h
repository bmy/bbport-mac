// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: getrusage for the process or the calling thread (posix_win32.cpp).
#pragma once

#include <sys/time.h>

#define RUSAGE_SELF 0
#define RUSAGE_THREAD 1

#ifdef __cplusplus
extern "C" {
#endif

struct rusage {
    struct timeval ru_utime;
    struct timeval ru_stime;
    long ru_maxrss, ru_minflt, ru_majflt, ru_nvcsw, ru_nivcsw;
};
int getrusage(int who, struct rusage* usage);

#ifdef __cplusplus
}
#endif
