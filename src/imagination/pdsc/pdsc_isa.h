/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef PDSC_ISA_H
#define PDSC_ISA_H

/**
 * \file pdsc_isa.h
 *
 * \brief PDS compiler isa definitions and helpers.
 */

#include "pdsc.h"

#include <stdbool.h>
#include <stdint.h>

/* STM sources. */

enum pdsc_slc_mode_ld {
   PDSC_SLC_MODE_LD_BYPASS = 0x0,
   PDSC_SLC_MODE_LD_CACHED = 0x1,
   PDSC_SLC_MODE_LD_CACHED_RD_NA = 0x3,
};

enum pdsc_slc_mode_st {
   PDSC_SLC_MODE_ST_WRITE_THROUGH = 0x0,
   PDSC_SLC_MODE_ST_WRITE_BACK = 0x1,
};

enum pdsc_ccslc_ld {
   PDSC_CCSLC_LD_NORMAL = 0x0,
   PDSC_CCSLC_LD_FORCE_LINE_FILL = 0x1,
};

enum pdsc_ccslc_st {
   PDSC_CCSLC_ST_LAZY_WRITE_BACK = 0x0,
   PDSC_CCSLC_ST_WRITE_THROUGH = 0x1,
};

enum pdsc_ccslc_idf {
   PDSC_CCSLC_IDF_FENCE_IN_SLC = 0x0,
   PDSC_CCSLC_IDF_FENCE_TO_MEM = 0x1,
};

enum pdsc_ccmcu_ld {
   PDSC_CCMCU_LD_NORMAL = 0x0,
   PDSC_CCMCU_LD_FORCE_LINE_FILL = 0x1,
};

enum pdsc_ccmcu_st {
   PDSC_CCMCU_ST_LAZY_WRITE_BACK = 0x0,
   PDSC_CCMCU_ST_WRITE_THROUGH = 0x1,
};

enum pdsc_ccmcu_idf {
   PDSC_CCMCU_IDF_FENCE_IN_MCU = 0x0,
   PDSC_CCMCU_IDF_FENCE_TO_SLC = 0x1,
};

enum pdsc_cmode_ld {
   PDSC_CMODE_LD_CACHED = 0x0,
   PDSC_CMODE_LD_BYPASS = 0x1,
   PDSC_CMODE_LD_FORCE_LINE_FILL = 0x2,
};

enum pdsc_cmode_st {
   PDSC_CMODE_ST_WRITE_THROUGH = 0x0,
   PDSC_CMODE_ST_WRITE_BACK = 0x1,
   PDSC_CMODE_ST_LAZY_WRITE_BACK = 0x2,
};

enum pdsc_primtype {
   PDSC_PRIMTYPE_POINT = 0x1,
   PDSC_PRIMTYPE_LINE = 0x2,
   PDSC_PRIMTYPE_TRIANGLE = 0x3,
};

enum pdsc_aop {
   PDSC_AOP_ADD = 0x0,
   PDSC_AOP_SUB = 0x1,
   PDSC_AOP_XCHG = 0x2,
   PDSC_AOP_UMIN = 0x4,
   PDSC_AOP_IMIN = 0x5,
   PDSC_AOP_UMAX = 0x6,
   PDSC_AOP_IMAX = 0x7,
   PDSC_AOP_AND = 0x8,
   PDSC_AOP_OR = 0x9,
   PDSC_AOP_XOR = 0xa,
};

enum pdsc_store_dest {
   PDSC_STORE_DEST_UNIFIED = 0x0,
   PDSC_STORE_DEST_COMMON = 0x1,
};

enum pdsc_doutw_bsize {
   PDSC_DOUTW_BSIZE_LOWER = 0x0,
   PDSC_DOUTW_BSIZE_UPPER = 0x1,
   PDSC_DOUTW_BSIZE_ALL64 = 0x2,
   PDSC_DOUTW_BSIZE_NONE = 0x3,
   PDSC_DOUTW_BSIZE_ALL128 = 0x7,
};

enum pdsc_doutu_sample_rate {
   PDSC_DOUTU_SAMPLE_RATE_INSTANCE = 0x0,
   PDSC_DOUTU_SAMPLE_RATE_SELECTIVE = 0x1,
   PDSC_DOUTU_SAMPLE_RATE_FULL = 0x2,
};

enum pdsc_douti_shademodel {
   PDSC_DOUTI_SHADEMODEL_FLAT_VERTEX0 = 0x0,
   PDSC_DOUTI_SHADEMODEL_FLAT_VERTEX1 = 0x1,
   PDSC_DOUTI_SHADEMODEL_FLAT_VERTEX2 = 0x2,
   PDSC_DOUTI_SHADEMODEL_GOURAUD = 0x3,
};

enum pdsc_douti_size {
   PDSC_DOUTI_SIZE_1D = 0x0,
   PDSC_DOUTI_SIZE_2D = 0x1,
   PDSC_DOUTI_SIZE_3D = 0x2,
   PDSC_DOUTI_SIZE_4D = 0x3,
};

/* PDSC_TYPE_STM_SRC0 */
struct pdsc_stm_src0 {
   enum pdsc_slc_mode_st slcmode;
   pdsc_ref ptemp_prim_needed_written;
   pdsc_ref ptemp_address;
   uint64_t address;
};

/* PDSC_TYPE_STM_SRC1 */
struct pdsc_stm_src1 {
   uint32_t prim_needed;
   uint32_t prim_written;
};

/* PDSC_TYPE_STM_SRC2 */
struct pdsc_stm_src2 {
   uint8_t vioff;
   uint8_t dmasize;
   uint16_t vooff;
};

/* PDSC_TYPE_STM_SRC3 */
struct pdsc_stm_src3 {
   enum pdsc_primtype primtype;
   uint16_t vosize;
   bool eop;
   uint64_t limit;
};

/* LD source. */

/* PDSC_TYPE_LD64_SRC0 */
struct pdsc_ld64_src0 {
   enum pdsc_slc_mode_ld slcmode;
   uint8_t ccpri;
   enum pdsc_ccslc_ld ccslc;
   enum pdsc_ccmcu_ld ccmcu;

   pdsc_ref dest;
   enum pdsc_cmode_ld cmode;
   uint8_t count8;
   uint64_t srcadd;
};

/* ST source. */

/* PDSC_TYPE_ST32_SRC0 */
struct pdsc_st32_src0 {
   enum pdsc_slc_mode_st slcmode;
   uint8_t ccpri;
   enum pdsc_ccslc_st ccslc;
   enum pdsc_ccmcu_st ccmcu;

   pdsc_ref src;
   enum pdsc_cmode_st cmode;
   uint8_t count4;
   uint64_t dstadd;
};

/* STMP sources. */

/* PDSC_TYPE_STMP_SRC0 */
struct pdsc_stmp_src0 {
   enum pdsc_slc_mode_st slcmode;
   uint8_t ccpri;
   enum pdsc_ccslc_st ccslc;
   enum pdsc_ccmcu_st ccmcu;

   pdsc_ref ptemp_prim_written;
   uint64_t address;
};

/* PDSC_TYPE_STMP_SRC1 */
struct pdsc_stmp_src1 {
   uint32_t prim_id;
   uint8_t vioff;
   uint8_t dmasize;
   uint16_t vooff;
};

/* PDSC_TYPE_STMP_SRC2 */
struct pdsc_stmp_src2 {
   enum pdsc_primtype primtype;
   uint16_t vosize;
   bool eop;
   uint64_t limit;
};

/* AA source. */

/* PDSC_TYPE_AA_SRC0 */
struct pdsc_aa_src0 {
   enum pdsc_aop aop;
   uint8_t ccpri;
   enum pdsc_ccslc_st ccslc;
   enum pdsc_ccmcu_st ccmcu;
   uint64_t srcadd;
};

/* IDF source. */

/* PDSC_TYPE_IDF_SRC0 */
struct pdsc_idf_src0 {
   uint8_t ccpri;
   enum pdsc_ccslc_idf ccslc;
   enum pdsc_ccmcu_idf ccmcu;
   uint64_t srcadd;
};

/* POL source. */

/* PDSC_TYPE_POL_SRC0 */
struct pdsc_pol_src0 {
   uint8_t ccpri;
   enum pdsc_ccslc_ld ccslc;
   enum pdsc_ccmcu_ld ccmcu;

   pdsc_ref dest;
   uint64_t srcadd;
};

/* DDMAD source. */

/* PDSC_TYPE_DDMAD_SRC3 */
struct pdsc_ddmad_src3 {
   uint32_t msize; /* if (ddmadt) */
   bool test; /* if (ddmadt) */

   bool last_issue;
   enum pdsc_store_dest store_dest;
   enum pdsc_cmode_ld cmode;
   uint16_t ao; /* dest reg offset */
   uint16_t bsize; /* num dwords */
};

/* DOUTD sources. */

/* PDSC_TYPE_DOUTD_SRC0 */
struct pdsc_doutd_src0 {
   enum pdsc_slc_mode_ld slcmode;
   uint8_t ccpri;
   enum pdsc_ccslc_ld ccslc;
   enum pdsc_ccmcu_ld ccmcu;
   uint16_t doffset;
   uint64_t sbase; /* base_addr */
};

/* PDSC_TYPE_DOUTD_SRC1 */
struct pdsc_doutd_src1 {
   bool last_issue;
   enum pdsc_store_dest store_dest;
   enum pdsc_cmode_ld cmode;
   uint16_t ao; /* dest reg offset */
   uint16_t bsize; /* num dwords */
};

/* DOUTW source. */

/* PDSC_TYPE_DOUTW_SRC1 */
struct pdsc_doutw_src1 {
   bool last_issue;
   enum pdsc_store_dest store_dest;
   enum pdsc_cmode_ld cmode;
   uint16_t ao; /* dest reg offset */
   enum pdsc_doutw_bsize bsize;
};

/* DOUTU source. */
#define PDSC_DOUTU_TEMP_GRANULARITY 4U

/* PDSC_TYPE_DOUTU_SRC0 */
struct pdsc_doutu_src0 {
   bool dual_phase;
   uint8_t temps;
   enum pdsc_doutu_sample_rate sample_rate;
   uint32_t exe_off;
};

/* DOUTI source. */

/* PDSC_TYPE_DOUTI_SRC0 */
struct pdsc_douti_src0 {
   bool last_issue;
   uint8_t dest;

   bool depth_bias;
   bool primitive_id;
   enum pdsc_douti_shademodel shademodel;
   bool point_sprite;
   bool wrap_u;
   bool wrap_v;
   bool wrap_s;
   enum pdsc_douti_size size;
   bool f16;
   bool perspective;
   uint8_t f32_offset;
   uint8_t f16_offset;
};

/**/

enum pdsc_opcodea {
   PDSC_OPCODEA_MAD = 0x0,
};

enum pdsc_opcodeb {
   PDSC_OPCODEB_SFTLP32 = 0x2,
   PDSC_OPCODEB_STM = 0x3,
};

enum pdsc_opcodec {
   PDSC_OPCODEC_ADD64 = 0x8,
   PDSC_OPCODEC_ADD32 = 0x9,
   PDSC_OPCODEC_SFTLP64 = 0xa,
   PDSC_OPCODEC_CMP = 0xb,
   PDSC_OPCODEC_BRA = 0xc,
   PDSC_OPCODEC_SP = 0xd,
   PDSC_OPCODEC_DDMAD = 0xe,
   PDSC_OPCODEC_DOUT = 0xf,
};

enum pdsc_opcodesp {
   PDSC_OPCODESP_LD = 0x0,
   PDSC_OPCODESP_ST = 0x1,
   PDSC_OPCODESP_WDF = 0x2,
   PDSC_OPCODESP_LIMM = 0x3,
   PDSC_OPCODESP_LOCK = 0x4,
   PDSC_OPCODESP_RELEASE = 0x5,
   PDSC_OPCODESP_HALT = 0x6,
   PDSC_OPCODESP_STMC = 0x7,
   PDSC_OPCODESP_STMP = 0x8,
   PDSC_OPCODESP_IDIV = 0x9,
   PDSC_OPCODESP_AA = 0xa,
   PDSC_OPCODESP_IDF = 0xb,
   PDSC_OPCODESP_POL = 0xc,
   PDSC_OPCODESP_NOP = 0xf,
};

enum pdsc_lop {
   PDSC_LOP_NONE = 0x0,
   PDSC_LOP_NOT = 0x1,
   PDSC_LOP_AND = 0x2,
   PDSC_LOP_OR = 0x3,
   PDSC_LOP_XOR = 0x4,
   PDSC_LOP_XNOR = 0x5,
   PDSC_LOP_NAND = 0x6,
   PDSC_LOP_NOR = 0x7,
};

enum pdsc_cop {
   PDSC_COP_EQ = 0x0,
   PDSC_COP_GT = 0x1,
   PDSC_COP_LT = 0x2,
   PDSC_COP_NE = 0x3,
};

enum pdsc_pred {
   PDSC_PRED_P0 = 0x0,
   PDSC_PRED_IF0 = 0x1,
   PDSC_PRED_IF1 = 0x2,
   PDSC_PRED_SO_OVF_P0 = 0x3,
   PDSC_PRED_SO_OVF_P1 = 0x4,
   PDSC_PRED_SO_OVF_P2 = 0x5,
   PDSC_PRED_SO_OVF_P3 = 0x6,
   PDSC_PRED_SO_OVF_GLOBAL = 0x7,
   PDSC_PRED_KEEP = 0x8,
   PDSC_PRED_OOB = 0x9,
};

enum pdsc_somask {
   PDSC_SOMASK_0 = (1 << 0),
   PDSC_SOMASK_1 = (1 << 1),
   PDSC_SOMASK_2 = (1 << 2),
   PDSC_SOMASK_3 = (1 << 3),
   PDSC_SOMASK_GLOBAL = (1 << 4),
};

enum pdsc_dstdout {
   PDSC_DSTDOUT_D = 0x0,
   PDSC_DSTDOUT_W = 0x1,
   PDSC_DSTDOUT_U = 0x2,
   PDSC_DSTDOUT_V = 0x3,
   PDSC_DSTDOUT_I = 0x4,
   PDSC_DSTDOUT_C = 0x5,
};

#define PDSC_REGS32_CONST_BASE 0
#define PDSC_REGS32_TEMP_BASE (PDSC_MAX_CONSTS)
#define PDSC_REGS32_PTEMP_BASE (PDSC_MAX_CONSTS + (2 * PDSC_MAX_TEMPS))

#define PDSC_REGS32TP_TEMP_BASE 0
#define PDSC_REGS32TP_PTEMP_BASE (PDSC_MAX_TEMPS)

/* Preloaded temps for PDS vertex shader program. */
enum pdsc_pre_temp_vs {
   PDSC_PRE_TEMP_VS_VERTEX_ID = 0,
   PDSC_PRE_TEMP_VS_INSTANCE_ID = 1,
};

/* Preloaded temps for PDS pixel event program. */
enum pdsc_pre_temp_pe {
   PDSC_PRE_TEMP_PE_TASK_TYPE = 1,
};

/* Preloaded temps for PDS streamout shader program. */
enum pdsc_pre_temp_so {
   PDSC_PRE_TEMP_SO_STREAM_PRESENT = 0,
   PDSC_PRE_TEMP_SO_PRIM_ID = 1,
};

/* Preloaded temps for PDS compute coefficient sync program.
 */
enum pdsc_pre_temp_ccs {
   PDSC_PRE_TEMP_CCS_WGID_X = 0,
   PDSC_PRE_TEMP_CCS_WGID_Y = 1,
   PDSC_PRE_TEMP_CCS_LID_XYZ = 2,
   PDSC_PRE_TEMP_CCS_WGID_Z = 3,
};

/* Preloaded temps for PDS compute shader program. */
enum pdsc_pre_temp_cs {
   PDSC_PRE_TEMP_CS_LID_X = 0,
   PDSC_PRE_TEMP_CS_LID_YZ = 1,
};

#endif /* PDSC_ISA_H */
