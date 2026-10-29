// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Gnm driver's command writers. The game reserves exactly the space the console
// driver uses for each call and checks the result, so sizes, argument checks and padding are the
// console's: a NOP fills what a packet leaves of its reservation (its first dword zeroed, the rest
// left as they are). The port is a base PS4: no PS4 Pro (Neo) layouts.

#include <span>
#include <cstring>

#include "bbport_gnm_hooks.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "platform/errors.h"
#include "platform/gnm/gnm.h"
#include "platform/gnm/gnm_init_state.h"
#include "platform/gnm/gnm_pm4.h"
#include "platform/videoout/videoout.h"


namespace BbGnm {

using namespace Pm4;
using Observe = BbGnmHooks::DriverWrite;

namespace {

/// Shader stages as the indirect draws name them, and the user data register their vertex and
/// instance offsets go to (SPI_SHADER_USER_DATA_{VS,ES,LS}_0; the others have none).
enum Stage : u32 { Cs, Ps, Vs, Gs, Es, Hs, Ls, StageCount };
constexpr u32 UserDataBase[StageCount] = {0, 0, 0x4C, 0, 0xCC, 0, 0x14C};

/// A user data slot (0 = none) as the indirect draw packets take it.
u32 UserDataSlot(u32 stage, u32 sgpr) {
    return sgpr == 0 ? 0 : (sgpr & 0xFFFF) + UserDataBase[stage];
}

/// The NOP the driver writes in place of a context register when only updating a shader: the
/// translator reads it as "this register changes" (marker 0xC01E0000 | register).
template <typename... Values>
u32* EmitUpdate(u32* at, u32 reg, Values... values) {
    return Emit(at, Op::Nop, false, false, 0xC01E0000u | reg, values...);
}

constexpr bool Predicated(u32 flags) {
    return flags & 1;
}

/// PM4 type-3 body sizes of the indirect argument records.
constexpr u32 DrawIndirectStride = 16;        // vertex count, instances, first vertex, instance
constexpr u32 DrawIndexIndirectStride = 20;   // + base vertex

/// Shader register blocks (the sce::Gnm shader structures' register halves). Index 1 is the
/// program address's high part, which must be 0: shaders live below 1 TiB.
bool CheckRegs(const u32* regs) {
    if (!regs) {
        LOG_ERROR(Lib_GnmDriver, "null shader registers");
        return false;
    }
    if (regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "shader address above 1 TiB");
        return false;
    }
    return true;
}

/// The PGM_RSRC1 a shader modifier changes (its VGPR/SGPR counts); `allowed` bits may differ.
u32 ApplyModifier(u32 rsrc1, u32 modifier, u32 allowed) {
    return modifier == 0 ? rsrc1 : (rsrc1 & allowed) | modifier;
}

u32 HwInit(u32* cmdbuf, u32 size, std::span<const u32> state) {
    Observe observe{cmdbuf, std::min(size, InitState::HwInitWords)};
    if (size < InitState::HwInitWords) {
        return 0;
    }
    u32* at = std::copy(InitState::ClearAndSync.begin(), InitState::ClearAndSync.end(), cmdbuf);
    at = std::copy(state.begin(), state.end(), at);
    PadTo(at, cmdbuf + InitState::HwInitWords);
    return InitState::HwInitWords;
}

/// A debug marker: payload type, the string with its terminator, padding to the driver's size.
/// Push and set markers take ((len + 8) / 4 + (len + 12) / 8 * 2) body dwords after the header.
s32 Marker(u32* cmdbuf, u32 size, const char* marker, u32 payload) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || !marker) {
        return -1;
    }
    const u32 len = u32(std::strlen(marker));
    const u32 words = ((len + 8) >> 2) + ((len + 12) >> 3) * 2;
    if (words + 2 != size) {
        return -1;
    }
    cmdbuf[0] = Header(Op::Nop, words + 1);
    cmdbuf[1] = payload;
    auto* text = reinterpret_cast<u8*>(cmdbuf + 2);
    std::memcpy(text, marker, len + 1);
    std::memset(text + len + 1, 0, words * 4 - (len + 1));
    return BbErr::Ok;
}

} // namespace

s32 PS4_SYSV_ABI sceGnmComputeWaitOnAddress(u32* cmdbuf, u32 size, uintptr_t addr, u32 mask,
                                            u32 cmp_func, u32 ref) {
    if (!cmdbuf || size != 14) {
        return -1;
    }
    // A 3-dword NOP, WAIT_REG_MEM (memory above 64 KiB, a register below), a 2-dword NOP.
    EmitPadding(cmdbuf, 3);
    const bool memory = addr > 0xFFFF;
    Emit(cmdbuf + 4, Op::WaitRegMem, false, false, (u32(memory) << 4) | (cmp_func & 7),
         u32(addr & (memory ? 0xFFFFFFFCu : 0xFFFFu)), u32(addr >> 32), ref, mask, 10u);
    EmitPadding(cmdbuf + 11, 2);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDispatchDirect(u32* cmdbuf, u32 size, u32 threads_x, u32 threads_y,
                                      u32 threads_z, u32 flags) {
    if (!cmdbuf || size != 9 || s32(threads_x | threads_y | threads_z) < 0) {
        return -1;
    }
    u32* at = Emit(cmdbuf, Op::DispatchDirect, true, Predicated(flags), threads_x, threads_y,
                   threads_z, (flags & 0x18) + 1); // + ordered append mode
    EmitPadding(at, 3);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDispatchIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 flags) {
    if (!cmdbuf || size != 7) {
        return -1;
    }
    u32* at = Emit(cmdbuf, Op::DispatchIndirect, true, Predicated(flags), data_offset,
                   (flags & 0x18) + 1);
    EmitPadding(at, 3);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDispatchIndirectOnMec(u32* cmdbuf, u32 size, u64 args, u32 modifier) {
    if (!cmdbuf || size != 8 || args == 0 || (args & 3) != 0) {
        return BbErr::Fail;
    }
    // DISPATCH_INDIRECT with the arguments' address in the packet (compute rings), then a NOP.
    Emit(cmdbuf, Op::DispatchIndirect, true, modifier & 1, u32(args), u32(args >> 32),
         (modifier & 0x18) | 1);
    EmitPadding(cmdbuf + 4, 3);
    return BbErr::Ok;
}

u32 PS4_SYSV_ABI sceGnmDispatchInitDefaultHardwareState(u32* cmdbuf, u32 size) {
    Observe observe{cmdbuf, std::min(size, InitState::HwInitWords)};
    if (size < InitState::HwInitWords) {
        return 0;
    }
    u32* at = EmitShCompute(cmdbuf, 0x216, 0xFFFFFFFFu); // COMPUTE_STATIC_THREAD_MGMT_SE0
    at = EmitShCompute(at, 0x217, 0xFFFFFFFFu);          // COMPUTE_STATIC_THREAD_MGMT_SE1
    at = EmitShCompute(at, 0x215, 0x170u);               // COMPUTE_RESOURCE_LIMITS
    at = Emit(at, Op::AcquireMem, false, false, 0x28000000u, 0u, 0u, 0u, 0u, 10u);
    PadTo(at, cmdbuf + InitState::HwInitWords);
    return InitState::HwInitWords;
}

s32 PS4_SYSV_ABI sceGnmDrawIndex(u32* cmdbuf, u32 size, u32 index_count, uintptr_t index_addr,
                                 u32 flags, u32 type) {
    if (!cmdbuf || size != 10 || index_addr == 0 || (index_addr & 1) != 0 ||
        (flags & 0x1FFFFFFE) != 0) {
        return -1;
    }
    // DRAW_INDEX_2: max size, base, count, initiator. Never predicated.
    u32* at = Emit(cmdbuf, Op::DrawIndex2, false, false, index_count, u32(index_addr),
                   u32(index_addr >> 32), index_count, 0u);
    EmitPadding(at, 3);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexAuto(u32* cmdbuf, u32 size, u32 index_count, u32 flags) {
    if (!cmdbuf || size != 7 || (flags & 0x1FFFFFFE) != 0) {
        return -1;
    }
    u32* at = Emit(cmdbuf, Op::DrawIndexAuto, false, false, index_count, 2u); // auto index
    EmitPadding(at, 3);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 shader_stage,
                                         u32 vertex_sgpr_offset, u32 instance_sgpr_offset,
                                         u32 flags) {
    if (!cmdbuf || size != 9 || shader_stage >= StageCount || vertex_sgpr_offset >= 16 ||
        instance_sgpr_offset >= 16) {
        return -1;
    }
    u32* at = Emit(cmdbuf, Op::DrawIndexIndirect, false, Predicated(flags), data_offset,
                   UserDataSlot(shader_stage, vertex_sgpr_offset),
                   UserDataSlot(shader_stage, instance_sgpr_offset), 0u);
    EmitPadding(at, 3);
    return BbErr::Ok;
}

static bool MultiDrawStage(u32 stage) {
    return stage == Vs || stage == Es || stage == Ls;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexIndirectCountMulti(u32* cmdbuf, u32 size, u32 data_offset,
                                                   u32 max_count, u64 count_addr, u32 shader_stage,
                                                   u32 vertex_sgpr_offset, u32 instance_sgpr_offset,
                                                   u32 flags) {
    if (!cmdbuf || size != 16 || vertex_sgpr_offset >= 16 || instance_sgpr_offset >= 16 ||
        !MultiDrawStage(shader_stage)) {
        return -1;
    }
    EmitPadding(cmdbuf, 2);
    u32* at = Emit(cmdbuf + 3, Op::DrawIndexIndirectCountMulti, false, Predicated(flags),
                   data_offset, UserDataSlot(shader_stage, vertex_sgpr_offset),
                   UserDataSlot(shader_stage, instance_sgpr_offset),
                   u32(count_addr != 0) << 30, // count from memory
                   max_count, u32(count_addr), u32(count_addr >> 32), DrawIndexIndirectStride, 0u);
    EmitPadding(at, 2);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexIndirectMulti(u32* cmdbuf, u32 size, u32 data_offset, u32 max_count,
                                              u32 shader_stage, u32 vertex_sgpr_offset,
                                              u32 instance_sgpr_offset, u32 flags) {
    if (!cmdbuf || size != 11 || vertex_sgpr_offset >= 16 || instance_sgpr_offset >= 16 ||
        !MultiDrawStage(shader_stage)) {
        return -1;
    }
    u32* at = Emit(cmdbuf, Op::DrawIndexIndirectMulti, false, Predicated(flags), data_offset,
                   UserDataSlot(shader_stage, vertex_sgpr_offset),
                   UserDataSlot(shader_stage, instance_sgpr_offset), max_count,
                   DrawIndexIndirectStride, 0u);
    EmitPadding(at, 3);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDrawIndexOffset(u32* cmdbuf, u32 size, u32 index_offset, u32 index_count,
                                       u32 flags) {
    if (!cmdbuf || size != 9) {
        return -1;
    }
    u32* at = Emit(cmdbuf, Op::DrawIndexOffset2, false, Predicated(flags), index_count,
                   index_offset, index_count, 0u);
    EmitPadding(at, 3);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDrawIndirect(u32* cmdbuf, u32 size, u32 data_offset, u32 shader_stage,
                                    u32 vertex_sgpr_offset, u32 instance_sgpr_offset, u32 flags) {
    if (!cmdbuf || size != 9 || shader_stage >= StageCount || vertex_sgpr_offset >= 16 ||
        instance_sgpr_offset >= 16) {
        return -1;
    }
    u32* at = Emit(cmdbuf, Op::DrawIndirect, false, Predicated(flags), data_offset,
                   UserDataSlot(shader_stage, vertex_sgpr_offset),
                   UserDataSlot(shader_stage, instance_sgpr_offset), 2u); // auto index
    EmitPadding(at, 3);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmDrawIndirectMulti(u32* cmdbuf, u32 size, u32 data_offset, u32 max_count,
                                         u32 shader_stage, u32 vertex_sgpr_offset,
                                         u32 instance_sgpr_offset, u32 flags) {
    if (!cmdbuf || size != 11 || shader_stage >= StageCount || vertex_sgpr_offset >= 16 ||
        instance_sgpr_offset >= 16) {
        return -1;
    }
    // Six body dwords, as DRAW_INDEX_INDIRECT_MULTI. (shadPS4's writer gave the header four,
    // which made the command processor read the stride as the next packet's header.)
    u32* at = Emit(cmdbuf, Op::DrawIndirectMulti, false, Predicated(flags), data_offset,
                   UserDataSlot(shader_stage, vertex_sgpr_offset),
                   UserDataSlot(shader_stage, instance_sgpr_offset), max_count, DrawIndirectStride,
                   2u);
    EmitPadding(at, 3);
    return BbErr::Ok;
}

u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState(u32* cmdbuf, u32 size) {
    return HwInit(cmdbuf, size, InitState::HwBase);
}
u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState175(u32* cmdbuf, u32 size) {
    return HwInit(cmdbuf, size, InitState::Hw175);
}
u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState200(u32* cmdbuf, u32 size) {
    return HwInit(cmdbuf, size, InitState::Hw200);
}
u32 PS4_SYSV_ABI sceGnmDrawInitDefaultHardwareState350(u32* cmdbuf, u32 size) {
    return HwInit(cmdbuf, size, InitState::Hw350);
}

u32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextState(u32* cmdbuf, u32 size) {
    if (size != InitState::ContextWords) {
        return 0;
    }
    u32* at = std::copy(InitState::Context.begin(), InitState::Context.end(), cmdbuf);
    PadTo(at, cmdbuf + InitState::ContextWords);
    return InitState::ContextWords;
}

u32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextState400(u32* cmdbuf, u32 size) {
    if (size != InitState::Context400Words) {
        return 0;
    }
    u32* at = std::copy(InitState::Context400.begin(), InitState::Context400.end(), cmdbuf);
    PadTo(at, cmdbuf + InitState::Context400Words);
    return InitState::Context400Words;
}

s32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextStateInternalCommand(u32* cmdbuf, u32 size) {
    return s32(SdkVersion() >= 0x4000000 ? sceGnmDrawInitToDefaultContextState400(cmdbuf, size)
                                         : sceGnmDrawInitToDefaultContextState(cmdbuf, size));
}

s32 PS4_SYSV_ABI sceGnmDrawInitToDefaultContextStateInternalSize() {
    return s32(SdkVersion() >= 0x4000000 ? InitState::Context400Words : InitState::ContextWords);
}

s32 PS4_SYSV_ABI sceGnmInsertDingDongMarker(u32* cmdbuf, u32 size) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size != 4) {
        return -1;
    }
    Emit(cmdbuf, Op::Nop, false, false, 0u, 0u, 0u);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmInsertPopMarker(u32* cmdbuf, u32 size) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size != 6) {
        return -1;
    }
    Emit(cmdbuf, Op::Nop, false, false, Payload::MarkerPop, 0u, 0u, 0u, 0u);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmInsertPushColorMarker(u32* cmdbuf, u32 size, const char* marker, u32 color) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || !marker) {
        return -1;
    }
    // The string, 8 bytes the driver leaves as they are, the color, zeros.
    const u32 len = u32(std::strlen(marker));
    const u32 words = ((len + 12) >> 2) + ((len + 16) >> 3) * 2;
    if (words + 2 != size) {
        return -1;
    }
    cmdbuf[0] = Header(Op::Nop, words + 1);
    cmdbuf[1] = Payload::ColorMarkerPush;
    auto* text = reinterpret_cast<u8*>(cmdbuf + 2);
    std::memcpy(text, marker, len + 1);
    std::memcpy(text + len + 1 + 8, &color, sizeof(color));
    std::memset(text + len + 1 + 12, 0, words * 4 - (len + 1 + 12));
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmInsertPushMarker(u32* cmdbuf, u32 size, const char* marker) {
    return Marker(cmdbuf, size, marker, Payload::MarkerPush);
}

s32 PS4_SYSV_ABI sceGnmInsertSetMarker(u32* cmdbuf, u32 size, const char* marker) {
    return Marker(cmdbuf, size, marker, Payload::MarkerSet);
}

s32 PS4_SYSV_ABI sceGnmInsertWaitFlipDone(u32* cmdbuf, u32 size, s32 vo_handle, u32 buf_idx) {
    Observe observe{cmdbuf, size};
    if (size != 7) {
        return -1;
    }
    // Waits until the display buffer's label (written 1 at the flip request) is back to 0.
    uintptr_t labels = 0;
    const s32 count = BbVideoOut::sceVideoOutGetBufferLabelAddress(vo_handle, &labels);
    ASSERT_MSG(count == 16, "sceVideoOutGetBufferLabelAddress failed");
    const u64 label = (labels + buf_idx * sizeof(u64)) & ~u64{3};
    Emit(cmdbuf, Op::WaitRegMem, false, false, 0x13u, u32(label), u32(label >> 32), 0u,
         0xFFFFFFFFu, 10u); // memory, equal
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmResetVgtControl(u32* cmdbuf, u32 size) {
    if (!cmdbuf || size != 3) {
        return -1;
    }
    EmitCtx(cmdbuf, 0x2AA, 0xFFu); // IA_MULTI_VGT_PARAM
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmSetVgtControl(u32* cmdbuf, u32 size, u32 prim_group_sz_minus_one,
                                     u32 partial_vs_wave_mode, u32 wd_switch_only_on_eop_mode) {
    if (!cmdbuf || size != 3 || prim_group_sz_minus_one >= 0x100 ||
        (wd_switch_only_on_eop_mode | partial_vs_wave_mode) >= 2) {
        return -1;
    }
    EmitCtx(cmdbuf, 0x2AA, ((partial_vs_wave_mode & 1) << 16) | (prim_group_sz_minus_one & 0xFFFF));
    return BbErr::Ok;
}

// ---- Shaders ---------------------------------------------------------------------------------
// The register blocks: [0] program address >> 8, [1] its high part, [2] PGM_RSRC1, [3]
// PGM_RSRC2, then the stage's context registers.

s32 PS4_SYSV_ABI sceGnmSetCsShader(u32* cmdbuf, u32 size, const u32* cs_regs) {
    return sceGnmSetCsShaderWithModifier(cmdbuf, size, cs_regs, 0);
}

s32 PS4_SYSV_ABI sceGnmSetCsShaderWithModifier(u32* cmdbuf, u32 size, const u32* cs_regs,
                                               u32 modifier) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size <= 0x18 || (modifier & 0xFFFFFC3F) != 0 || !CheckRegs(cs_regs)) {
        return -1;
    }
    u32* at = EmitSh(cmdbuf, 0x20C, cs_regs[0], 0u); // COMPUTE_PGM_LO/HI
    at = EmitSh(at, 0x212, ApplyModifier(cs_regs[2], modifier, 0xFFFFFC3F),
                cs_regs[3]);                               // COMPUTE_PGM_RSRC1/2
    at = EmitSh(at, 0x207, cs_regs[4], cs_regs[5], cs_regs[6]); // COMPUTE_NUM_THREAD_X/Y/Z
    EmitPadding(at, 11);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmSetEsShader(u32* cmdbuf, u32 size, const u32* es_regs, u32 modifier) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size < 0x14 || (es_regs && (modifier & 0xFCFFFC3F)) || !CheckRegs(es_regs)) {
        return -1;
    }
    u32* at = EmitSh(cmdbuf, 0xC8, es_regs[0], 0u); // SPI_SHADER_PGM_LO/HI_ES
    at = EmitSh(at, 0xCA, ApplyModifier(es_regs[2], modifier, 0xFCFFFC3F), es_regs[3]);
    EmitPadding(at, 11);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmSetGsShader(u32* cmdbuf, u32 size, const u32* gs_regs) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size < 0x1D || !CheckRegs(gs_regs)) {
        return -1;
    }
    u32* at = EmitSh(cmdbuf, 0x88, gs_regs[0], 0u); // SPI_SHADER_PGM_LO/HI_GS
    at = EmitSh(at, 0x8A, gs_regs[2], gs_regs[3]);
    at = EmitCtx(at, 0x2E5, gs_regs[4]); // VGT_STRMOUT_CONFIG
    at = EmitCtx(at, 0x29B, gs_regs[5]); // VGT_GS_OUT_PRIM_TYPE
    at = EmitCtx(at, 0x2E4, gs_regs[6]); // VGT_GS_INSTANCE_CNT
    EmitPadding(at, 11);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmUpdateGsShader(u32* cmdbuf, u32 size, const u32* gs_regs) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size < 0x1D || !CheckRegs(gs_regs)) {
        return -1;
    }
    u32* at = EmitSh(cmdbuf, 0x88, gs_regs[0], 0u);
    at = EmitSh(at, 0x8A, gs_regs[2], gs_regs[3]);
    at = EmitUpdate(at, 0x2E5, gs_regs[4]);
    at = EmitUpdate(at, 0x29B, gs_regs[5]);
    at = EmitUpdate(at, 0x2E4, gs_regs[6]);
    EmitPadding(at, 11);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmSetHsShader(u32* cmdbuf, u32 size, const u32* hs_regs, u32 ls_hs_config) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size < 0x1E || !CheckRegs(hs_regs)) {
        return -1;
    }
    u32* at = EmitSh(cmdbuf, 0x108, hs_regs[0], 0u); // SPI_SHADER_PGM_LO/HI_HS
    at = EmitSh(at, 0x10A, hs_regs[2], hs_regs[3]);
    at = EmitCtx(at, 0x286, hs_regs[5], hs_regs[6]); // VGT_HOS_MAX/MIN_TESS_LEVEL
    at = EmitCtx(at, 0x2DB, hs_regs[4]);             // VGT_TF_PARAM
    at = EmitCtx(at, 0x2D6, ls_hs_config);           // VGT_LS_HS_CONFIG
    EmitPadding(at, 11);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmUpdateHsShader(u32* cmdbuf, u32 size, const u32* hs_regs, u32 ls_hs_config) {
    if (!cmdbuf || size <= 0x1C || !CheckRegs(hs_regs)) {
        return -1;
    }
    u32* at = EmitSh(cmdbuf, 0x108, hs_regs[0], 0u);
    at = EmitSh(at, 0x10A, hs_regs[2], hs_regs[3]);
    at = EmitUpdate(at, 0x286, hs_regs[5], hs_regs[6]);
    at = EmitUpdate(at, 0x2DB, hs_regs[4]);
    at = EmitUpdate(at, 0x2D6, ls_hs_config);
    EmitPadding(at, 11);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmSetLsShader(u32* cmdbuf, u32 size, const u32* ls_regs, u32 modifier) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size < 0x17) {
        return -1;
    }
    // A modifier sets PGM_RSRC1.SGPRS (bits 6-9) and may set VGPR_COMP_CNT (24-25) too.
    const u32 allowed = (modifier & 0xFFFFFC3F) == 0 ? 0xFFFFFC3F : 0xFCFFFC3F;
    if ((ls_regs && (modifier & allowed) != 0) || !CheckRegs(ls_regs)) {
        return -1;
    }
    u32* at = EmitSh(cmdbuf, 0x148, ls_regs[0], 0u); // SPI_SHADER_PGM_LO/HI_LS
    at = EmitSh(at, 0x14B, ls_regs[3]);              // SPI_SHADER_PGM_RSRC2_LS
    at = EmitSh(at, 0x14A, ApplyModifier(ls_regs[2], modifier, allowed), ls_regs[3]);
    EmitPadding(at, 11);
    return BbErr::Ok;
}

namespace {

/// The pixel shader's registers; `update` writes the context ones as update markers.
u32* EmitPsRegs(u32* at, const u32* ps_regs, bool update, u32 pgm_hi = 0) {
    at = EmitSh(at, 0x8, ps_regs[0], pgm_hi);       // SPI_SHADER_PGM_LO/HI_PS
    at = EmitSh(at, 0xA, ps_regs[2], ps_regs[3]);   // SPI_SHADER_PGM_RSRC1/2_PS
    const auto ctx = [&](u32 reg, auto... values) {
        at = update ? EmitUpdate(at, reg, values...) : EmitCtx(at, reg, values...);
    };
    ctx(0x1C4, ps_regs[4], ps_regs[5]); // SPI_SHADER_Z_FORMAT, SPI_SHADER_COL_FORMAT
    ctx(0x1B3, ps_regs[6], ps_regs[7]); // SPI_PS_INPUT_ENA, SPI_PS_INPUT_ADDR
    ctx(0x1B6, ps_regs[8]);             // SPI_PS_IN_CONTROL
    ctx(0x1B8, ps_regs[9]);             // SPI_BARYC_CNTL
    ctx(0x203, ps_regs[10]);            // DB_SHADER_CONTROL
    ctx(0x8F, ps_regs[11]);             // CB_SHADER_MASK
    return at;
}

/// sceGnmSet/UpdatePsShader(350): 40 dwords; no registers = no pixel shader.
s32 PsShader(u32* cmdbuf, u32 size, const u32* ps_regs, bool update, bool sdk350) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size <= 0x27) {
        return -1;
    }
    if (!ps_regs) {
        u32* at = EmitSh(cmdbuf, 0x8, 0u, 0u);
        at = update ? EmitUpdate(at, 0x203, 0u) : EmitCtx(at, 0x203, 0u); // DB_SHADER_CONTROL
        if (sdk350) {
            at = EmitCtx(at, 0x8F, 0xFu); // CB_SHADER_MASK: all of target 0
        }
        PadTo(at, cmdbuf + 40);
        return BbErr::Ok;
    }
    if (ps_regs[1] != 0) {
        LOG_ERROR(Lib_GnmDriver, "shader address above 1 TiB");
        return -1;
    }
    EmitPadding(EmitPsRegs(cmdbuf, ps_regs, update), 11);
    return BbErr::Ok;
}

} // namespace

s32 PS4_SYSV_ABI sceGnmSetPsShader(u32* cmdbuf, u32 size, const u32* ps_regs) {
    return PsShader(cmdbuf, size, ps_regs, false, false);
}
s32 PS4_SYSV_ABI sceGnmSetPsShader350(u32* cmdbuf, u32 size, const u32* ps_regs) {
    return PsShader(cmdbuf, size, ps_regs, false, true);
}
s32 PS4_SYSV_ABI sceGnmUpdatePsShader(u32* cmdbuf, u32 size, const u32* ps_regs) {
    return PsShader(cmdbuf, size, ps_regs, true, false);
}
s32 PS4_SYSV_ABI sceGnmUpdatePsShader350(u32* cmdbuf, u32 size, const u32* ps_regs) {
    return PsShader(cmdbuf, size, ps_regs, true, true);
}

namespace {

s32 VsShader(u32* cmdbuf, u32 size, const u32* vs_regs, u32 modifier, bool update) {
    Observe observe{cmdbuf, size};
    if (!cmdbuf || size <= 0x1C || (vs_regs && (modifier & 0xFCFFFC3F)) || !CheckRegs(vs_regs)) {
        return -1;
    }
    u32* at = EmitSh(cmdbuf, 0x48, vs_regs[0], 0u); // SPI_SHADER_PGM_LO/HI_VS
    at = EmitSh(at, 0x4A, ApplyModifier(vs_regs[2], modifier, 0xFCFFFC3F), vs_regs[3]);
    const auto ctx = [&](u32 reg, u32 value) {
        at = update ? EmitUpdate(at, reg, value) : EmitCtx(at, reg, value);
    };
    ctx(0x207, vs_regs[6]); // PA_CL_VS_OUT_CNTL
    ctx(0x1B1, vs_regs[4]); // SPI_VS_OUT_CONFIG
    ctx(0x1C3, vs_regs[5]); // SPI_SHADER_POS_FORMAT
    EmitPadding(at, 11);
    return BbErr::Ok;
}

} // namespace

s32 PS4_SYSV_ABI sceGnmSetVsShader(u32* cmdbuf, u32 size, const u32* vs_regs, u32 modifier) {
    return VsShader(cmdbuf, size, vs_regs, modifier, false);
}
s32 PS4_SYSV_ABI sceGnmUpdateVsShader(u32* cmdbuf, u32 size, const u32* vs_regs, u32 modifier) {
    return VsShader(cmdbuf, size, vs_regs, modifier, true);
}

// ---- The driver's embedded shaders -----------------------------------------------------------
// Programs the console driver keeps for clears and copies (sce::Gnm embedded shaders), each with
// its binary header and register block. The driver places them at 0xFE000Fx0; here they live in
// the library, so their address is computed instead.

namespace {

// clang-format off
alignas(256) constexpr u32 EmbeddedPs0[] = {
    0xBEEB03FFu, 0x00000003u, // s_mov_b32 vcc_hi, 3
    0x7E000280u,              // v_mov_b32 v0, 0
    0x5E000100u,              // v_cvt_pkrtz_f16_f32 v0, v0, v0
    0xBF800000u,              // s_nop
    0xF8001C0Fu, 0x00000000u, // exp mrt0, v0, v0 compr vm done
    0xBF810000u,              // s_endpgm
    0x5362724Fu, 0x07726468u, 0x00002043u, 0u, 0xB0A45B2Bu, 0x1D39766Du, 0x72044B7Bu, 0x0000000Fu,
    0x0FE000F0u, 0u, 0xC0000u, 4u, 0u, 4u, 2u, 2u, 0u, 0u, 0x10u, 0xFu, 0xCu, 0u, 0u, 0u,
};
alignas(256) constexpr u32 EmbeddedPs1[] = {
    0xBEEB03FFu, 0x00000003u, // s_mov_b32 vcc_hi, 3
    0x7E040280u,              // v_mov_b32 v2, 0
    0xF8001803u, 0x02020202u, // exp mrt0, v2, v2, off, off vm done
    0xBF810000u,              // s_endpgm
    0x5362724Fu, 0x07726468u, 0x00001841u, 0x04080002u, 0x98B9CB94u, 0u, 0x6F130734u, 0x0000000Fu,
    0x0FE000F2u, 0u, 0x2000u, 0u, 0u, 2u, 2u, 2u, 0u, 0u, 0x10u, 3u, 0xCu,
};
/// A full-screen triangle with one UV set.
alignas(256) constexpr u32 EmbeddedVs0[] = {
    0xBEEB03FFu, 0x00000007u, // s_mov_b32 vcc_hi, 7
    0x36020081u,              // v_and_b32 v1, 1, v0
    0x34020281u,              // v_lshlrev_b32 v1, 1, v1
    0x360000C2u,              // v_and_b32 v0, -2, v0
    0x4A0202C1u,              // v_add_i32 v1, vcc, -1, v1
    0x4A0000C1u,              // v_add_i32 v0, vcc, -1, v0
    0x7E020B01u,              // v_cvt_f32_i32 v1, v1
    0x7E000B00u,              // v_cvt_f32_i32 v0, v0
    0x7E040280u,              // v_mov_b32 v2, 0
    0x7E0602F2u,              // v_mov_b32 v3, 1.0
    0xF80008CFu, 0x03020001u, // exp pos0, v1, v0, v2, v3 done
    0xF800020Fu, 0x03030303u, // exp param0, v3, v3, v3, v3
    0xBF810000u,              // s_endpgm
    0x5362724Fu, 0x07726468u, 0x00004047u, 0u, 0x47F8C29Fu, 0x9B2DA5CFu, 0xFF7C5B7Du, 0x00000017u,
    0x0FE000F1u, 0u, 0x000C0000u, 4u, 0u, 4u, 0u, 7u,
};
// clang-format on

} // namespace

s32 PS4_SYSV_ABI sceGnmSetEmbeddedPsShader(u32* cmdbuf, u32 size, u32 shader_id, u32 modifier) {
    if (shader_id > 1) {
        LOG_ERROR(Lib_GnmDriver, "unknown embedded pixel shader {}", shader_id);
        return BbErr::GnmFailure;
    }
    // 0: writes zero to target 0 (compressed); 1: zero to its first two channels.
    const auto address = reinterpret_cast<uintptr_t>(shader_id == 0 ? EmbeddedPs0 : EmbeddedPs1);
    const u32 ps0[12] = {u32(address >> 8), u32(address >> 40), 0xC0000u, 4u, 0u, 4u,
                         2u, 2u, 0u, 0u, 0x10u, 0xFu};
    const u32 ps1[12] = {u32(address >> 8), u32(address >> 40), 0x2000u, 0u, 0u, 2u,
                         2u, 2u, 0u, 0u, 0x10u, 3u};
    // The driver's own path (sceGnmSetPsShader350) refuses addresses above 1 TiB; written here.
    EmitPadding(EmitPsRegs(cmdbuf, shader_id == 0 ? ps0 : ps1, false, u32(address >> 40)), 11);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceGnmSetEmbeddedVsShader(u32* cmdbuf, u32 size, u32 shader_id, u32 modifier) {
    if (shader_id != 0) {
        LOG_ERROR(Lib_GnmDriver, "unknown embedded vertex shader {}", shader_id);
        return BbErr::GnmFailure;
    }
    const auto address = reinterpret_cast<uintptr_t>(EmbeddedVs0);
    u32* at = EmitSh(cmdbuf, 0x48, u32(address >> 8), u32(address >> 40));
    at = EmitSh(at, 0x4A, 0xC0000u, 4u);
    at = EmitCtx(at, 0x207, 0u); // PA_CL_VS_OUT_CNTL
    at = EmitCtx(at, 0x1B1, 0u); // SPI_VS_OUT_CONFIG: one parameter
    at = EmitCtx(at, 0x1C3, 4u); // SPI_SHADER_POS_FORMAT
    EmitPadding(at, 11);
    return BbErr::Ok;
}

} // namespace BbGnm
