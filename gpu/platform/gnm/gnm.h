// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: libSceGnmDriver, the PS4's graphics driver library as games call it. It writes PM4
// packets into the game's command buffers (draws, dispatches, shader state, markers: the console
// driver's exact layouts, which the game's own code sizes buffers for) and takes the finished
// buffers to the command processor (submissions, compute queues, flips, end-of-frame pacing).
#pragma once

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace BbGnm {

/// Creates the command processor and presenter and registers the library's functions.
void RegisterLib(Core::Loader::SymbolsResolver* sym);
/// Initialize a small FX target without changing a guest graphics context's cursor or state.
bool SubmitFxInitZero(u64 address, u32 bytes);

/// The submission lock is released when the command processor has decoded a frame (instead of
/// when the GPU went idle): only with honest labels.
bool SubmitLockOnDecode();
void ReleaseSubmissionLock();
/// Frames (sceGnmSubmitDone calls) the GPU has finished.
void NoteFramesRetired(u64 frames);

/// The GPU's core clock (what sceGnmGetGpuCoreClockFrequency reports).
inline constexpr u64 GpuCoreClock = 800'000'000;

// ---- Command writers (gnm_commands.cpp) ------------------------------------------------------
// Each returns 0 (or the dwords written) and fills exactly `size` dwords; the checks on sizes and
// arguments are the console driver's.

s32 PS4_SYSV_ABI sceGnmComputeWaitOnAddress(u32* cmdbuf, u32 size, uintptr_t addr, u32 mask,
                                            u32 cmp_func, u32 ref);
s32 PS4_SYSV_ABI sceGnmDispatchDirect(u32* cmdbuf, u32 size, u32 threads_x, u32 threads_y,
                                      u32 threads_z, u32 flags);
s32 PS4_SYSV_ABI sceGnmDispatchIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 flags);
s32 PS4_SYSV_ABI sceGnmDispatchIndirectOnMec(u32* cmdbuf, u32 size, u64 args, u32 modifier);
u32 PS4_SYSV_ABI sceGnmDispatchInitDefaultHardwareState(u32* cmdbuf, u32 size);
s32 PS4_SYSV_ABI sceGnmDrawIndex(u32* cmdbuf, u32 size, u32 index_count, uintptr_t index_addr,
                                 u32 flags, u32 type);
s32 PS4_SYSV_ABI sceGnmDrawIndexAuto(u32* cmdbuf, u32 size, u32 index_count, u32 flags);
s32 PS4_SYSV_ABI sceGnmDrawIndexIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 shader_stage,
                                         u32 vertex_sgpr_offset, u32 instance_sgpr_offset,
                                         u32 flags);
s32 PS4_SYSV_ABI sceGnmDrawIndexIndirectCountMulti(u32* cmdbuf, u32 size, u32 data_offset,
                                                   u32 max_count, u64 count_addr, u32 shader_stage,
                                                   u32 vertex_sgpr_offset, u32 instance_sgpr_offset,
                                                   u32 flags);
s32 PS4_SYSV_ABI sceGnmDrawIndexIndirectMulti(u32* cmdbuf, u32 size, u32 data_offset, u32 max_count,
                                              u32 shader_stage, u32 vertex_sgpr_offset,
                                              u32 instance_sgpr_offset, u32 flags);
s32 PS4_SYSV_ABI sceGnmDrawIndexOffset(u32* cmdbuf, u32 size, u32 index_offset, u32 index_count,
                                       u32 flags);
s32 PS4_SYSV_ABI sceGnmDrawIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 shader_stage,
                                    u32 vertex_sgpr_offset, u32 instance_sgpr_offset, u32 flags);
s32 PS4_SYSV_ABI sceGnmDrawIndirectMulti(u32* cmdbuf, u32 size, u32 data_offset, u32 max_count,
                                         u32 shader_stage, u32 vertex_sgpr_offset,
                                         u32 instance_sgpr_offset, u32 flags);
u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState(u32* cmdbuf, u32 size);
u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState175(u32* cmdbuf, u32 size);
u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState200(u32* cmdbuf, u32 size);
u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState350(u32* cmdbuf, u32 size);
u32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextState(u32* cmdbuf, u32 size);
u32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextState400(u32* cmdbuf, u32 size);
s32 PS4_SYSV_ABI sceGnmInsertDingDongMarker(u32* cmdbuf, u32 size);
s32 PS4_SYSV_ABI sceGnmInsertPopMarker(u32* cmdbuf, u32 size);
s32 PS4_SYSV_ABI sceGnmInsertPushColorMarker(u32* cmdbuf, u32 size, const char* marker, u32 color);
s32 PS4_SYSV_ABI sceGnmInsertPushMarker(u32* cmdbuf, u32 size, const char* marker);
s32 PS4_SYSV_ABI sceGnmInsertSetMarker(u32* cmdbuf, u32 size, const char* marker);
s32 PS4_SYSV_ABI sceGnmInsertWaitFlipDone(u32* cmdbuf, u32 size, s32 vo_handle, u32 buf_idx);
s32 PS4_SYSV_ABI sceGnmResetVgtControl(u32* cmdbuf, u32 size);
s32 PS4_SYSV_ABI sceGnmSetCsShader(u32* cmdbuf, u32 size, const u32* cs_regs);
s32 PS4_SYSV_ABI sceGnmSetCsShaderWithModifier(u32* cmdbuf, u32 size, const u32* cs_regs,
                                               u32 modifier);
s32 PS4_SYSV_ABI sceGnmSetEmbeddedPsShader(u32* cmdbuf, u32 size, u32 shader_id, u32 modifier);
s32 PS4_SYSV_ABI sceGnmSetEmbeddedVsShader(u32* cmdbuf, u32 size, u32 shader_id, u32 modifier);
s32 PS4_SYSV_ABI sceGnmSetEsShader(u32* cmdbuf, u32 size, const u32* es_regs, u32 modifier);
s32 PS4_SYSV_ABI sceGnmSetGsShader(u32* cmdbuf, u32 size, const u32* gs_regs);
s32 PS4_SYSV_ABI sceGnmSetHsShader(u32* cmdbuf, u32 size, const u32* hs_regs, u32 ls_hs_config);
s32 PS4_SYSV_ABI sceGnmSetLsShader(u32* cmdbuf, u32 size, const u32* ls_regs, u32 modifier);
s32 PS4_SYSV_ABI sceGnmSetPsShader(u32* cmdbuf, u32 size, const u32* ps_regs);
s32 PS4_SYSV_ABI sceGnmSetPsShader350(u32* cmdbuf, u32 size, const u32* ps_regs);
s32 PS4_SYSV_ABI sceGnmSetVgtControl(u32* cmdbuf, u32 size, u32 prim_group_sz_minus_one,
                                     u32 partial_vs_wave_mode, u32 wd_switch_only_on_eop_mode);
s32 PS4_SYSV_ABI sceGnmSetVsShader(u32* cmdbuf, u32 size, const u32* vs_regs, u32 modifier);
s32 PS4_SYSV_ABI sceGnmUpdateGsShader(u32* cmdbuf, u32 size, const u32* gs_regs);
s32 PS4_SYSV_ABI sceGnmUpdateHsShader(u32* cmdbuf, u32 size, const u32* hs_regs, u32 ls_hs_config);
s32 PS4_SYSV_ABI sceGnmUpdatePsShader(u32* cmdbuf, u32 size, const u32* ps_regs);
s32 PS4_SYSV_ABI sceGnmUpdatePsShader350(u32* cmdbuf, u32 size, const u32* ps_regs);
s32 PS4_SYSV_ABI sceGnmUpdateVsShader(u32* cmdbuf, u32 size, const u32* vs_regs, u32 modifier);
/// The context reset of the game's SDK generation, and its size.
s32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextStateInternalCommand(u32* cmdbuf, u32 size);
s32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextStateInternalSize();

/// The game's compiled SDK version (sceKernelGetCompiledSdkVersion), read at registration.
u32 SdkVersion();

} // namespace BbGnm
