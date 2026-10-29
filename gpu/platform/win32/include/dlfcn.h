// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: dlopen and friends over LoadLibrary (posix_win32.cpp).
#pragma once

#define RTLD_LAZY 0x1
#define RTLD_NOW 0x2
#define RTLD_LOCAL 0x0
#define RTLD_GLOBAL 0x100
#define RTLD_NOLOAD 0x4
#define RTLD_DEFAULT ((void*)0)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char* dli_fname; ///< the module's path
    void* dli_fbase;       ///< its load address
    const char* dli_sname; ///< null: no symbols from the PE exports here
    void* dli_saddr;
} Dl_info;

void* dlopen(const char* path, int flags);
void* dlsym(void* module, const char* name);
int dlclose(void* module);
const char* dlerror(void);
int dladdr(const void* address, Dl_info* info);

#ifdef __cplusplus
}
#endif
