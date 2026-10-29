// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: process_vm_readv on this process (a read that fails instead of faulting:
// ReadProcessMemory), the only use the GPU library has for it.
#pragma once

#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct iovec {
    void* iov_base;
    size_t iov_len;
};
ssize_t process_vm_readv(int pid, const struct iovec* local, unsigned long local_count,
                         const struct iovec* remote, unsigned long remote_count, unsigned long flags);

#ifdef __cplusplus
}
#endif
