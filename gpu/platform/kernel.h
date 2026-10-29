// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: libkernel functions the GPU library implements (shim/bbgpu.cpp: clocks and process
// facts from the port's runtime) or needs. The online module links against these names.
#pragma once

#include "common/types.h"

namespace Libraries::Kernel {

u64 PS4_SYSV_ABI sceKernelGetTscFrequency();
u64 PS4_SYSV_ABI sceKernelReadTsc();
u64 PS4_SYSV_ABI sceKernelGetProcessTime();
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounter();
u64 PS4_SYSV_ABI sceKernelGetProcessTimeCounterFrequency();
/// 0: the port is a base PS4, never a PS4 Pro.
s32 PS4_SYSV_ABI sceKernelIsNeoMode();
s32 PS4_SYSV_ABI sceKernelGetCompiledSdkVersion(s32* ver);
s32 PS4_SYSV_ABI sceKernelUsleep(u32 microseconds);
/// The calling guest thread's errno.
int* PS4_SYSV_ABI __Error();

} // namespace Libraries::Kernel
