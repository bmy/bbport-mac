// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: backtrace over RtlCaptureStackBackTrace; symbols as module+offset.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

int backtrace(void** buffer, int size);
void backtrace_symbols_fd(void* const* buffer, int size, int fd);

#ifdef __cplusplus
}
#endif
