/* Copyright © 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

/**
 * INTEL EMPLOYEES:
 *
 * This file should only contain the SrcIDs that could easily be found by any
 * 3rd party modifying the UMD to deliberately cause page faults via specific
 * hardware units.
 *
 * https://gfxspecs.intel.com/Predator/Home/Index/73545
 */
#define INTEL_SRCID_XEFI_DECLS                                                \
_SRCID_XEFI_DECL(GAM,               0x00)                                     \
_SRCID_XEFI_DECL(CS,                0x03)                                     \
_SRCID_XEFI_DECL(SOL,               0x04)                                     \
_SRCID_XEFI_DECL(VF,                0x05)                                     \
_SRCID_XEFI_DECL(CFEG,              0x06)                                     \
_SRCID_XEFI_DECL(SF,                0x0B)                                     \
_SRCID_XEFI_DECL(GAM_PD,            0x0C)                                     \
_SRCID_XEFI_DECL(MSC,               0x0F)                                     \
_SRCID_XEFI_DECL(WM,                0x10)                                     \
_SRCID_XEFI_DECL(DAPR,              0x11)                                     \
_SRCID_XEFI_DECL(L2NODE,            0x12)                                     \
_SRCID_XEFI_DECL(RCPBE,             0x14)                                     \
_SRCID_XEFI_DECL(IECP,              0x15)                                     \
_SRCID_XEFI_DECL(RCC,               0x17)                                     \
_SRCID_XEFI_DECL(VFH,               0x19)                                     \
_SRCID_XEFI_DECL(VFGR,              0x1A)                                     \
_SRCID_XEFI_DECL(LSC,               0x1B)                                     \
_SRCID_XEFI_DECL(HIZ,               0x1D)                                     \
_SRCID_XEFI_DECL(RCZ,               0x1E)                                     \
_SRCID_XEFI_DECL(IC,                0x20)                                     \
_SRCID_XEFI_DECL(DM_ASSEMBLY,       0x24)                                     \
_SRCID_XEFI_DECL(SVSM,              0x28)                                     \
_SRCID_XEFI_DECL(SARB,              0x2F)                                     \
_SRCID_XEFI_DECL(DM_NON_ASSEMBLY,   0x34)                                     \
_SRCID_XEFI_DECL(LBDM,              0x35)                                     \
_SRCID_XEFI_DECL(DC,                0x3C)                                     \
_SRCID_XEFI_DECL(PROF,              0x3D)                                     \
_SRCID_XEFI_DECL(GAM_VTD,           0x45)                                     \
_SRCID_XEFI_DECL(XETLB,             0x46)                                     \
_SRCID_XEFI_DECL(GAM_STLB,          0x4A)                                     \
_SRCID_XEFI_DECL(GAM_TR,            0x4D)                                     \
_SRCID_XEFI_DECL(RC_CLR,            0x55)                                     \
_SRCID_XEFI_DECL(CC,                0x56)                                     \
_SRCID_XEFI_DECL(RCC_1,             0x57)                                     \
_SRCID_XEFI_DECL(HIZ_CLR,           0x58)                                     \
_SRCID_XEFI_DECL(CLR_VP,            0x59)                                     \
_SRCID_XEFI_DECL(AMFS_MBB,          0x5A)                                     \
_SRCID_XEFI_DECL(AMFS_CC,           0x5B)                                     \
_SRCID_XEFI_DECL(IZFE,              0x5C)                                     \
_SRCID_XEFI_DECL(ZPBE,              0x5D)                                     \
_SRCID_XEFI_DECL(DUALQ_CS,          0x5E)                                     \
_SRCID_XEFI_DECL(COMP_CS,           0x5F)                                     \
_SRCID_XEFI_DECL(IME,               0x60)                                     \
_SRCID_XEFI_DECL(CRE,               0x61)                                     \
_SRCID_XEFI_DECL(OAR,               0x62)                                     \
_SRCID_XEFI_DECL(OAC,               0x63)                                     \
_SRCID_XEFI_DECL(VCS,               0x81)                                     \
_SRCID_XEFI_DECL(VMC,               0x82)                                     \
_SRCID_XEFI_DECL(VIP,               0x83)                                     \
_SRCID_XEFI_DECL(BSP,               0x86)                                     \
_SRCID_XEFI_DECL(VMX_RS,            0x88)                                     \
_SRCID_XEFI_DECL(VMX_BS,            0x89)                                     \
_SRCID_XEFI_DECL(VMX_RA,            0x8A)                                     \
_SRCID_XEFI_DECL(VLF_RS,            0x8C)                                     \
_SRCID_XEFI_DECL(VLF_FW,            0x8D)                                     \
_SRCID_XEFI_DECL(SFC_LB_VD,         0x8E)                                     \
_SRCID_XEFI_DECL(SFC_FW_VD,         0x8F)                                     \
_SRCID_XEFI_DECL(VD_ENC,            0x90)                                     \
_SRCID_XEFI_DECL(VD_ENC_RS,         0x91)                                     \
_SRCID_XEFI_DECL(VD_ENC_SO,         0x92)                                     \
_SRCID_XEFI_DECL(HEVC_ENC_HEVC_REF, 0x93)                                     \
_SRCID_XEFI_DECL(HEVC_ENC_HEVC_SRC, 0x94)                                     \
_SRCID_XEFI_DECL(MFL_VMC,           0x98)                                     \
_SRCID_XEFI_DECL(MFL_VMX_BS,        0x9A)                                     \
_SRCID_XEFI_DECL(MFL_VMX_RA,        0x9B)                                     \
_SRCID_XEFI_DECL(MFL_VLF_RS,        0x9C)                                     \
_SRCID_XEFI_DECL(MFL_VLF_FW,        0x9D)                                     \
_SRCID_XEFI_DECL(HRS,               0xA0)                                     \
_SRCID_XEFI_DECL(HPO,               0xA1)                                     \
_SRCID_XEFI_DECL(HLC_VNC_BS,        0xA2)                                     \
_SRCID_XEFI_DECL(HLC_SO,            0xA3)                                     \
_SRCID_XEFI_DECL(HLC_IH,            0xA4)                                     \
_SRCID_XEFI_DECL(VNC_BA,            0xA6)                                     \
_SRCID_XEFI_DECL(HMC,               0xA8)                                     \
_SRCID_XEFI_DECL(HED,               0xAA)                                     \
_SRCID_XEFI_DECL(HPP,               0xAB)                                     \
_SRCID_XEFI_DECL(HLF_RS,            0xAC)                                     \
_SRCID_XEFI_DECL(HLF_FW,            0xAD)                                     \
_SRCID_XEFI_DECL(MCMP,              0xB0)                                     \
_SRCID_XEFI_DECL(SFC_HIST_VD,       0xB8)                                     \
_SRCID_XEFI_DECL(AMC,               0xB9)                                     \
_SRCID_XEFI_DECL(AMX,               0xBA)                                     \
_SRCID_XEFI_DECL(ALC_BA,            0xBB)                                     \
_SRCID_XEFI_DECL(ALC_BS,            0xBC)                                     \
_SRCID_XEFI_DECL(VECS,              0xC0)                                     \
_SRCID_XEFI_DECL(GAV,               0xC1)                                     \
_SRCID_XEFI_DECL(VFW,               0xC2)                                     \
_SRCID_XEFI_DECL(VEO_P1,            0xC3)                                     \
_SRCID_XEFI_DECL(VEO_P2,            0xC4)                                     \
_SRCID_XEFI_DECL(SFC_LB_VE,         0xC5)                                     \
_SRCID_XEFI_DECL(SFC_FW_VE,         0xC6)                                     \
_SRCID_XEFI_DECL(IECP_RO,           0xC7)                                     \
_SRCID_XEFI_DECL(OAG,               0xF5)                                     \
_SRCID_XEFI_DECL(OAM,               0xF6)                                     \
_SRCID_XEFI_DECL(BCF_LINK,          0xF9)                                     \
_SRCID_XEFI_DECL(BCS_LINK,          0xFA)                                     \
_SRCID_XEFI_DECL(BLB2,              0xFB)                                     \
_SRCID_XEFI_DECL(BCS,               0xFC)                                     \
_SRCID_XEFI_DECL(BLB,               0xFD)                                     \
_SRCID_XEFI_DECL(BLB_LINK,          0xFE)                                     \
_SRCID_XEFI_DECL(BCPF,              0xFF)

/**
 * Maps the internal SrcIDs to the relevant API concept.
 *
 * These mappings were mostly reverse engineered from studying the simulator
 * and are therefore not guaranteed to be a 100% complete map of all SrcIDs.
 */
#define INTEL_SRCID_MESA_DECLS                                                \
_SRCID_MESA_DECL(AMFS_SCRATCH_ACCESS,   { AMFS_CC })                          \
_SRCID_MESA_DECL(AMFS_STATE,            { AMFS_MBB })                         \
_SRCID_MESA_DECL(BINDING_TABLE,         { SARB })                             \
_SRCID_MESA_DECL(BLITTER_ACCESS,        { BLB, BLB2, BLB_LINK })              \
_SRCID_MESA_DECL(CCS_SURFACE_ACCESS,    { CC })                               \
_SRCID_MESA_DECL(CS_INSTRUCTION_FETCH,  { CS, DUALQ_CS, COMP_CS, VCS,         \
                                          VECS, BCS, BCS_LINK })              \
_SRCID_MESA_DECL(CS_MEMORY_ACCESS,      { CFEG, VFH, VFGR, BCF_LINK,          \
                                          BCPF, PROF })                       \
_SRCID_MESA_DECL(DATAPORT_ACCESS,       { L2NODE, LSC, DC })                  \
_SRCID_MESA_DECL(DATAPORT_STATE,        { DAPR })                             \
_SRCID_MESA_DECL(DEPTH_ACCESS,          { HIZ, HIZ_CLR, RCZ })                \
_SRCID_MESA_DECL(DEPTH_STATE,           { IZFE, ZPBE })                       \
_SRCID_MESA_DECL(EU_INSTRUCTION_FETCH,  { IC })                               \
_SRCID_MESA_DECL(MEDIA_CODEC_BITSTREAM, { BSP, VMX_BS, VMX_RA, VD_ENC_SO,     \
                                          MFL_VMC, MFL_VMX_BS, MFL_VMX_RA,    \
                                          HPO, HLC_VNC_BS, HLC_SO, HLC_IH,    \
                                          VNC_BA, HED, ALC_BA, ALC_BS })      \
_SRCID_MESA_DECL(MEDIA_CODEC_FEEDBACK,  { SFC_HIST_VD })                      \
_SRCID_MESA_DECL(MEDIA_CODEC_SURFACE,   { IME, CRE, VMC, VIP, VMX_RS, VLF_RS, \
                                          VLF_FW, SFC_LB_VD, SFC_FW_VD,       \
                                          VD_ENC, VD_ENC_RS,                  \
                                          HEVC_ENC_HEVC_REF,                  \
                                          HEVC_ENC_HEVC_SRC, MFL_VLF_RS,      \
                                          MFL_VLF_FW, HRS, HMC, HPP, HLF_RS,  \
                                          HLF_FW, MCMP, AMC, AMX })           \
_SRCID_MESA_DECL(MEDIA_ENHANCE_ACCESS,  { IECP, IECP_RO, GAV, VFW, VEO_P1,    \
                                          VEO_P2, SFC_LB_VE, SFC_FW_VE, })    \
_SRCID_MESA_DECL(OA_FEEDBACK,           { OAG, OAM })                         \
_SRCID_MESA_DECL(PAGE_WALKER,           { GAM, GAM_PD, GAM_VTD, XETLB,        \
                                          GAM_STLB, GAM_TR })                 \
_SRCID_MESA_DECL(PIXELPORT_ACCESS,      { RCC, RC_CLR })                      \
_SRCID_MESA_DECL(PIXELPORT_STATE,       { MSC, RCPBE, RCC_1 })                \
_SRCID_MESA_DECL(RASTERIZER_STATE,      { SF, WM })                           \
_SRCID_MESA_DECL(SAMPLER_ACCESS,        { DM_ASSEMBLY, DM_NON_ASSEMBLY,       \
                                          LBDM })                             \
_SRCID_MESA_DECL(SAMPLER_STATE,         { SVSM })                             \
_SRCID_MESA_DECL(TRANSFORM_FEEDBACK,    { SOL })                              \
_SRCID_MESA_DECL(VERTEX_FETCH,          { VF })
