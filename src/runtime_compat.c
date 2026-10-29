// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
/* Calls of Bloodborne's that need no work here: memory locking and syncing hints, libc's
 * per-module finalizer hook (the runtime runs the game's exit handlers itself). The entries other
 * games needed (event flags, time zones, dialogs and PSN services Bloodborne does not use) are in
 * the full project copy of 2026-10-28. Unknown functions still stop. */
#include "runtime.h"

static ABI int32_t ok(void) { return 0; }

static const RuntimeExport exports[] = {
    {"sceKernelMlock", ok}, {"msync", ok}, {"__pthread_cxa_finalize", ok},
};

uintptr_t runtime_compat_resolve(const char *name) { return RUNTIME_LOOKUP(exports, name); }
