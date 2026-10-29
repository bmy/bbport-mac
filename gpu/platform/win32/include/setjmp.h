// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport Windows: MinGW's setjmp.h (re)defines setjmp/longjmp on every inclusion (outside its
// guard), with a longjmp that unwinds through SEH and stops at the game's frames. In C they are
// bb_setjmp/bb_longjmp (bbport_jmp.h) after each inclusion: no include guard here on purpose.
#include_next <setjmp.h>
#ifndef __cplusplus
#include "bbport_jmp.h"
#undef setjmp
#undef longjmp
#define setjmp(buffer) bb_setjmp((unsigned long long*)(buffer))
#define longjmp(buffer, value) bb_longjmp((unsigned long long*)(buffer), value)
#endif
