/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef PDSC_OPS_H
#define PDSC_OPS_H

/**
 * \file pdsc_ops.h
 *
 * \brief PDS compiler op header.
 */

#include "pdsc.h"
#include "pdsc_isa.h"

#include "util/macros.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

enum pdsc_op {
   PDSC_OP_SFTLP64,
   PDSC_OP_SFTLP32,
   PDSC_OP_STM,
   PDSC_OP_MAD,
   PDSC_OP_ADD64,
   PDSC_OP_ADD32,
   PDSC_OP_CMP,
   PDSC_OP_BRA,
   PDSC_OP_LD64,
   PDSC_OP_ST32,
   PDSC_OP_IDIV,
   PDSC_OP_STMP,
   PDSC_OP_AA,
   PDSC_OP_WDF,
   PDSC_OP_IDF,
   PDSC_OP_POL,
   PDSC_OP_LIMM,
   PDSC_OP_LOCK,
   PDSC_OP_RELEASE,
   PDSC_OP_HALT,
   PDSC_OP_NOP,
   PDSC_OP_STMC,
   PDSC_OP_DDMAD,
   PDSC_OP_DOUT,

   _PDSC_OP_COUNT,
};

enum pdsc_mod {
   PDSC_MOD_CC, /* bool */

   PDSC_MOD_LOP, /* enum pdsc_lop */

   PDSC_MOD_CC_SO_OVF, /* bool */
   PDSC_MOD_CC_GLOBAL_OVF, /* bool */
   PDSC_MOD_TST, /* bool */
   PDSC_MOD_SO, /* uint2 */

   PDSC_MOD_SNA, /* bool */
   PDSC_MOD_ALUM, /* bool */

   PDSC_MOD_COP, /* enum pdsc_cop */

   PDSC_MOD_SRCC, /* enum pdsc_pred */
   PDSC_MOD_SETC, /* enum pdsc_pred */
   PDSC_MOD_NEG, /* bool */

   PDSC_MOD_SOMASK, /* enum pdsc_somask */

   PDSC_MOD_END, /* bool */

   PDSC_MOD_DSTDOUT, /* enum pdsc_dstdout */

   _PDSC_MOD_COUNT,
};

enum PACKED pdsc_valid_type {
   PDSC_NULL = BITFIELD_BIT(0),

   PDSC_VALID_TYPE_CONST32 = BITFIELD_BIT(1),
   PDSC_VALID_TYPE_TEMP32 = BITFIELD_BIT(2),
   PDSC_VALID_TYPE_PTEMP32 = BITFIELD_BIT(3),

   PDSC_VALID_TYPE_CONST64 = BITFIELD_BIT(4),
   PDSC_VALID_TYPE_TEMP64 = BITFIELD_BIT(5),
   PDSC_VALID_TYPE_PTEMP64 = BITFIELD_BIT(6),

   PDSC_GLOBAL = BITFIELD_BIT(7),

   PDSC_IMM = BITFIELD_BIT(8),
   PDSC_SIMM = BITFIELD_BIT(9),

   PDSC_REGS32 = PDSC_VALID_TYPE_CONST32 | PDSC_VALID_TYPE_TEMP32 |
                 PDSC_VALID_TYPE_PTEMP32,
   PDSC_REGS32TP = PDSC_VALID_TYPE_TEMP32 | PDSC_VALID_TYPE_PTEMP32,
   PDSC_REGS32T = PDSC_VALID_TYPE_TEMP32,

   PDSC_REGS64 = PDSC_VALID_TYPE_CONST64 | PDSC_VALID_TYPE_TEMP64 |
                 PDSC_VALID_TYPE_PTEMP64,
   PDSC_REGS64TP = PDSC_VALID_TYPE_TEMP64 | PDSC_VALID_TYPE_PTEMP64,
   PDSC_REGS64T = PDSC_VALID_TYPE_TEMP64,
   PDSC_REGS64C = PDSC_VALID_TYPE_CONST64,
};

static_assert(sizeof(enum pdsc_valid_type) == sizeof(uint16_t),
              "sizeof(enum pdsc_valid_type) != sizeof(uint16_t)");

struct pdsc_op_info {
   enum pdsc_valid_type valid_dst_types;
   enum pdsc_valid_type valid_src_types[PDSC_MAX_SRCS];
   uint8_t mod_map[_PDSC_MOD_COUNT];
};

extern const struct pdsc_op_info pdsc_op_info[_PDSC_OP_COUNT];

/**/

bool pdsc_has_mod(const pdsc_program *p, pdsc_instr_t i, enum pdsc_mod mod);
uint8_t pdsc_get_mod(const pdsc_program *p, pdsc_instr_t i, enum pdsc_mod mod);
void pdsc_set_mod(pdsc_program *p,
                  pdsc_instr_t i,
                  enum pdsc_mod mod,
                  uint8_t value);

#define DEF_MOD(mod, mod_type, mod_enum)                                    \
   static inline bool pdsc_has_##mod(const pdsc_program *p, pdsc_instr_t i) \
   {                                                                        \
      return pdsc_has_mod(p, i, mod_enum);                                  \
   }                                                                        \
   static inline mod_type pdsc_get_##mod(const pdsc_program *p,             \
                                         pdsc_instr_t i)                    \
   {                                                                        \
      return (mod_type)pdsc_get_mod(p, i, mod_enum);                        \
   }                                                                        \
   static inline void pdsc_set_##mod(pdsc_program *p,                       \
                                     pdsc_instr_t i,                        \
                                     mod_type value)                        \
   {                                                                        \
      pdsc_set_mod(p, i, mod_enum, value);                                  \
   }

DEF_MOD(cc, bool, PDSC_MOD_CC)
DEF_MOD(lop, enum pdsc_lop, PDSC_MOD_LOP)
DEF_MOD(cc_so_ovf, bool, PDSC_MOD_CC_SO_OVF)
DEF_MOD(cc_global_ovf, bool, PDSC_MOD_CC_GLOBAL_OVF)
DEF_MOD(tst, bool, PDSC_MOD_TST)
DEF_MOD(so, uint8_t, PDSC_MOD_SO)
DEF_MOD(sna, bool, PDSC_MOD_SNA)
DEF_MOD(alum, bool, PDSC_MOD_ALUM)
DEF_MOD(cop, enum pdsc_cop, PDSC_MOD_COP)
DEF_MOD(srcc, enum pdsc_pred, PDSC_MOD_SRCC)
DEF_MOD(setc, enum pdsc_pred, PDSC_MOD_SETC)
DEF_MOD(neg, bool, PDSC_MOD_NEG)
DEF_MOD(somask, enum pdsc_somask, PDSC_MOD_SOMASK)
DEF_MOD(end, bool, PDSC_MOD_END)
DEF_MOD(dstdout, enum pdsc_dstdout, PDSC_MOD_DSTDOUT)

#undef DEF_MOD

struct pdsc_sftlp_mods {
   bool cc;
   enum pdsc_lop lop;
};

struct pdsc_stm_mods {
   bool cc;
   bool cc_so_ovf;
   bool cc_global_ovf;
   bool tst;
   uint8_t so;
};

struct pdsc_add_mods {
   bool cc;
   bool sna;
   bool alum;
};

struct pdsc_cmp_mods {
   bool cc;
   enum pdsc_cop cop;
};

struct pdsc_bra_mods {
   enum pdsc_pred srcc;
   enum pdsc_pred setc;
   bool neg;
};

struct pdsc_cc_mods {
   bool cc;
};

struct pdsc_stmc_mods {
   bool cc;
   enum pdsc_somask somask;
};

struct pdsc_ddmad_mods {
   bool cc;
   bool end;
};

struct pdsc_dout_mods {
   bool cc;
   bool end;
   enum pdsc_dstdout dstdout;
};

pdsc_instr_t _pdsc_sftlp64(pdsc_program *p,
                           pdsc_ref dst,
                           pdsc_ref src0,
                           pdsc_ref src1,
                           pdsc_ref src2,
                           struct pdsc_sftlp_mods mods);

#define pdsc_sftlp64(p, dst, src0, src1, src2, ...) \
   _pdsc_sftlp64(p,                                 \
                 dst,                               \
                 src0,                              \
                 src1,                              \
                 src2,                              \
                 (struct pdsc_sftlp_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_sftlp32(pdsc_program *p,
                           pdsc_ref dst,
                           pdsc_ref src0,
                           pdsc_ref src1,
                           pdsc_ref src2,
                           struct pdsc_sftlp_mods mods);

#define pdsc_sftlp32(p, dst, src0, src1, src2, ...) \
   _pdsc_sftlp32(p,                                 \
                 dst,                               \
                 src0,                              \
                 src1,                              \
                 src2,                              \
                 (struct pdsc_sftlp_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_stm(pdsc_program *p,
                       pdsc_ref src0,
                       pdsc_ref src1,
                       pdsc_ref src2,
                       pdsc_ref src3,
                       struct pdsc_stm_mods mods);

#define pdsc_stm(p, src0, src1, src2, src3, ...) \
   _pdsc_stm(p,                                  \
             src0,                               \
             src1,                               \
             src2,                               \
             src3,                               \
             (struct pdsc_stm_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_mad(pdsc_program *p,
                       pdsc_ref dst,
                       pdsc_ref src0,
                       pdsc_ref src1,
                       pdsc_ref src2,
                       struct pdsc_add_mods mods);

#define pdsc_mad(p, dst, src0, src1, src2, ...) \
   _pdsc_mad(p,                                 \
             dst,                               \
             src0,                              \
             src1,                              \
             src2,                              \
             (struct pdsc_add_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_add64(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_add_mods mods);

#define pdsc_add64(p, dst, src0, src1, ...) \
   _pdsc_add64(p, dst, src0, src1, (struct pdsc_add_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_add32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_add_mods mods);

#define pdsc_add32(p, dst, src0, src1, ...) \
   _pdsc_add32(p, dst, src0, src1, (struct pdsc_add_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_cmp(pdsc_program *p,
                       pdsc_ref src0,
                       pdsc_ref src1,
                       struct pdsc_cmp_mods mods);

#define pdsc_cmp(p, src0, src1, ...) \
   _pdsc_cmp(p, src0, src1, (struct pdsc_cmp_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_bra(pdsc_program *p,
                       pdsc_instr_t target_instr,
                       struct pdsc_bra_mods mods);

#define pdsc_bra(p, target_instr, ...) \
   _pdsc_bra(p, target_instr, (struct pdsc_bra_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t
_pdsc_ld64(pdsc_program *p, pdsc_ref src0, struct pdsc_cc_mods mods);

#define pdsc_ld64(p, src0, ...) \
   _pdsc_ld64(p, src0, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t
_pdsc_st32(pdsc_program *p, pdsc_ref src0, struct pdsc_cc_mods mods);

#define pdsc_st32(p, src0, ...) \
   _pdsc_st32(p, src0, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_idiv(pdsc_program *p,
                        pdsc_ref dst,
                        pdsc_ref src0,
                        pdsc_ref src1,
                        struct pdsc_cc_mods mods);

#define pdsc_idiv(p, dst, src0, src1, ...) \
   _pdsc_idiv(p, dst, src0, src1, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_stmp(pdsc_program *p,
                        pdsc_ref src0,
                        pdsc_ref src1,
                        pdsc_ref src2,
                        struct pdsc_stm_mods mods);

#define pdsc_stmp(p, src0, src1, src2, ...) \
   _pdsc_stmp(p, src0, src1, src2, (struct pdsc_stm_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_aa(pdsc_program *p,
                      pdsc_ref src0,
                      pdsc_ref src1,
                      struct pdsc_cc_mods mods);

#define pdsc_aa(p, src0, src1, ...) \
   _pdsc_aa(p, src0, src1, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_wdf(pdsc_program *p, struct pdsc_cc_mods mods);

#define pdsc_wdf(p, ...) _pdsc_wdf(p, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t
_pdsc_idf(pdsc_program *p, pdsc_ref src0, struct pdsc_cc_mods mods);

#define pdsc_idf(p, src0, ...) \
   _pdsc_idf(p, src0, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_pol(pdsc_program *p,
                       pdsc_ref src0,
                       pdsc_ref src1,
                       struct pdsc_cc_mods mods);

#define pdsc_pol(p, src0, src1, ...) \
   _pdsc_pol(p, src0, src1, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_limm(pdsc_program *p,
                        pdsc_ref dst,
                        pdsc_ref src0,
                        struct pdsc_cc_mods mods);

#define pdsc_limm(p, dst, src0, ...) \
   _pdsc_limm(p, dst, src0, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_lock(pdsc_program *p, struct pdsc_cc_mods mods);

#define pdsc_lock(p, ...) \
   _pdsc_lock(p, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_release(pdsc_program *p, struct pdsc_cc_mods mods);

#define pdsc_release(p, ...) \
   _pdsc_release(p, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_halt(pdsc_program *p, struct pdsc_cc_mods mods);

#define pdsc_halt(p, ...) \
   _pdsc_halt(p, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_nop(pdsc_program *p, struct pdsc_cc_mods mods);

#define pdsc_nop(p, ...) _pdsc_nop(p, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_stmc(pdsc_program *p, struct pdsc_stmc_mods mods);

#define pdsc_stmc(p, ...) \
   _pdsc_stmc(p, (struct pdsc_stmc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_ddmad(pdsc_program *p,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         pdsc_ref src2,
                         pdsc_ref src3,
                         struct pdsc_ddmad_mods mods);

#define pdsc_ddmad(p, src0, src1, src2, src3, ...) \
   _pdsc_ddmad(p,                                  \
               src0,                               \
               src1,                               \
               src2,                               \
               src3,                               \
               (struct pdsc_ddmad_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_dout(pdsc_program *p,
                        pdsc_ref src0,
                        pdsc_ref src1,
                        pdsc_ref src2,
                        struct pdsc_dout_mods mods);

#define pdsc_dout(p, src0, src1, src2, ...) \
   _pdsc_dout(p, src0, src1, src2, (struct pdsc_dout_mods){ 0, ##__VA_ARGS__ })

/**/

struct pdsc_doutd_params {
   uint8_t _;
   struct pdsc_dout_mods mods;
   struct pdsc_doutd_src0 src0;
   struct pdsc_doutd_src1 src1;
};

pdsc_instr_t _pdsc_doutd(pdsc_program *p,
                         uint8_t src0_id,
                         uint8_t src1_id,
                         bool init_srcs,
                         struct pdsc_doutd_params params);

#define pdsc_doutd(p, src0_id, src1_id, init_srcs, ...) \
   _pdsc_doutd(p,                                       \
               src0_id,                                 \
               src1_id,                                 \
               init_srcs,                               \
               (struct pdsc_doutd_params){ 0, ##__VA_ARGS__ })

/**/

struct pdsc_doutw_params {
   uint8_t _;
   struct pdsc_dout_mods mods;
   struct pdsc_doutw_src1 src1;
};

pdsc_instr_t _pdsc_doutw(pdsc_program *p,
                         pdsc_ref src0,
                         uint8_t src1_id,
                         pdsc_ref src2,
                         bool init_srcs,
                         struct pdsc_doutw_params params);

#define pdsc_doutw(p, src0, src1_id, src2, init_srcs, ...) \
   _pdsc_doutw(p,                                          \
               src0,                                       \
               src1_id,                                    \
               src2,                                       \
               init_srcs,                                  \
               (struct pdsc_doutw_params){ 0, ##__VA_ARGS__ })

/**/

struct pdsc_doutu_params {
   uint8_t _;
   struct pdsc_dout_mods mods;
   struct pdsc_doutu_src0 src0;
};

pdsc_instr_t _pdsc_doutu(pdsc_program *p,
                         uint8_t src0_id,
                         bool init_srcs,
                         struct pdsc_doutu_params params);

#define pdsc_doutu(p, src0_id, init_srcs, ...) \
   _pdsc_doutu(p,                              \
               src0_id,                        \
               init_srcs,                      \
               (struct pdsc_doutu_params){ 0, ##__VA_ARGS__ })

/**/
pdsc_instr_t
_pdsc_doutv(pdsc_program *p, pdsc_ref src1, struct pdsc_dout_mods mods);

#define pdsc_doutv(p, src1, ...) \
   _pdsc_doutv(p, src1, (struct pdsc_dout_mods){ 0, ##__VA_ARGS__ })

/**/

struct pdsc_douti_params {
   uint8_t _;
   struct pdsc_dout_mods mods;
   struct pdsc_douti_src0 src0;
};

pdsc_instr_t _pdsc_douti(pdsc_program *p,
                         uint8_t src0_id,
                         bool init_srcs,
                         struct pdsc_douti_params params);

#define pdsc_douti(p, src0_id, init_srcs, ...) \
   _pdsc_douti(p,                              \
               src0_id,                        \
               init_srcs,                      \
               (struct pdsc_douti_params){ 0, ##__VA_ARGS__ })

/**/

pdsc_instr_t _pdsc_doutc(pdsc_program *p, struct pdsc_dout_mods mods);

#define pdsc_doutc(p, ...) \
   _pdsc_doutc(p, (struct pdsc_dout_mods){ 0, ##__VA_ARGS__ })

/**/

pdsc_instr_t _pdsc_mov32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src,
                         struct pdsc_cc_mods mods);

#define pdsc_mov32(p, dst, src, ...) \
   _pdsc_mov32(p, dst, src, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_mov64(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src,
                         struct pdsc_cc_mods mods);

#define pdsc_mov64(p, dst, src, ...) \
   _pdsc_mov64(p, dst, src, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t pdsc_setc(pdsc_program *p, enum pdsc_pred setc);
pdsc_instr_t pdsc_branch_push(pdsc_program *p, enum pdsc_pred srcc, bool neg);
void pdsc_branch_pop(pdsc_program *p, pdsc_instr_t i);

pdsc_instr_t _pdsc_shl32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         uint8_t shift,
                         struct pdsc_cc_mods mods);

#define pdsc_shl32(p, dst, src0, shift, ...) \
   _pdsc_shl32(p, dst, src0, shift, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_shr32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         uint8_t shift,
                         struct pdsc_cc_mods mods);

#define pdsc_shr32(p, dst, src0, shift, ...) \
   _pdsc_shr32(p, dst, src0, shift, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_shr64(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         uint8_t shift,
                         struct pdsc_cc_mods mods);

#define pdsc_shr64(p, dst, src0, shift, ...) \
   _pdsc_shr64(p, dst, src0, shift, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_xor32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_cc_mods mods);

#define pdsc_xor32(p, dst, src0, src1, ...) \
   _pdsc_xor32(p, dst, src0, src1, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_xor64(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_cc_mods mods);

#define pdsc_xor64(p, dst, src0, src1, ...) \
   _pdsc_xor64(p, dst, src0, src1, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

#define pdsc_zero64(p, dst, ...) pdsc_xor64(p, dst, dst, dst, ##__VA_ARGS__)

pdsc_instr_t _pdsc_and32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_cc_mods mods);

#define pdsc_and32(p, dst, src0, src1, ...) \
   _pdsc_and32(p, dst, src0, src1, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

pdsc_instr_t _pdsc_or32(pdsc_program *p,
                        pdsc_ref dst,
                        pdsc_ref src0,
                        pdsc_ref src1,
                        struct pdsc_cc_mods mods);

#define pdsc_or32(p, dst, src0, src1, ...) \
   _pdsc_or32(p, dst, src0, src1, (struct pdsc_cc_mods){ 0, ##__VA_ARGS__ })

struct pdsc_ld_params {
   uint8_t _;
   struct pdsc_cc_mods mods;
   struct pdsc_ld64_src0 src0;
};

pdsc_instr_t _pdsc_ld(pdsc_program *p,
                      pdsc_ref dest,
                      uint8_t dest_ref_id,
                      uint8_t src0_id,
                      bool init_srcs,
                      struct pdsc_ld_params params);

#define pdsc_ld(p, dest, dest_ref_id, src0_id, init_srcs, ...) \
   _pdsc_ld(p,                                                 \
            dest,                                              \
            dest_ref_id,                                       \
            src0_id,                                           \
            init_srcs,                                         \
            (struct pdsc_ld_params){ 0, ##__VA_ARGS__ })

struct pdsc_st_params {
   uint8_t _;
   struct pdsc_cc_mods mods;
   struct pdsc_st32_src0 src0;
};

pdsc_instr_t _pdsc_st(pdsc_program *p,
                      pdsc_ref src,
                      uint8_t src_ref_id,
                      uint8_t src0_id,
                      bool init_srcs,
                      struct pdsc_st_params params);

#define pdsc_st(p, src, src_ref_id, src0_id, init_srcs, ...) \
   _pdsc_st(p,                                               \
            src,                                             \
            src_ref_id,                                      \
            src0_id,                                         \
            init_srcs,                                       \
            (struct pdsc_st_params){ 0, ##__VA_ARGS__ })
#endif /* PDSC_OPS_H */
