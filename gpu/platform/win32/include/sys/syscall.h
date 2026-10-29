// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: the Linux syscalls the runtime makes directly; gettid only (others fail, ENOSYS).
#pragma once

#define SYS_gettid 186
#define SYS_tgkill 234
#define SYS_getdents64 217
#define SYS_arch_prctl 158

#ifdef __cplusplus
extern "C" {
#endif
long syscall(long number, ...);
#ifdef __cplusplus
}
#endif
