/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pdsc_ops.c
 *
 * \brief PDS compiler op info and helpers.
 */

#include "pdsc.h"
#include "pdsc_internal.h"
#include "pdsc_isa.h"
#include "pdsc_isa_pack.h"

#include "util/compiler.h"

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

const struct pdsc_op_info pdsc_op_info[_PDSC_OP_COUNT] = {
   [PDSC_OP_SFTLP64] = {
      .valid_dst_types = PDSC_REGS64TP,
      .valid_src_types = {
         [0] = PDSC_REGS64TP,
         [1] = PDSC_REGS64TP | PDSC_NULL,
         [2] = PDSC_REGS32 | PDSC_SIMM | PDSC_NULL,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_LOP] = 2,
      },
   },
   [PDSC_OP_SFTLP32] = {
      .valid_dst_types = PDSC_REGS32T,
      .valid_src_types = {
         [0] = PDSC_REGS32T,
         [1] = PDSC_REGS32 | PDSC_NULL,
         [2] = PDSC_REGS32TP | PDSC_SIMM | PDSC_NULL,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_LOP] = 2,
      },
   },
   [PDSC_OP_STM] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64TP,
         [1] = PDSC_REGS64TP,
         [2] = PDSC_REGS32,
         [3] = PDSC_REGS64TP,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_CC_SO_OVF] = 2,
         [PDSC_MOD_CC_GLOBAL_OVF] = 3,
         [PDSC_MOD_TST] = 4,
         [PDSC_MOD_SO] = 5,
      },
   },
   [PDSC_OP_MAD] = {
      .valid_dst_types = PDSC_REGS64T,
      .valid_src_types = {
         [0] = PDSC_REGS32,
         [1] = PDSC_REGS32,
         [2] = PDSC_REGS64,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_SNA] = 2,
         [PDSC_MOD_ALUM] = 3,
      },
   },
   [PDSC_OP_ADD64] = {
      .valid_dst_types = PDSC_REGS64TP,
      .valid_src_types = {
         [0] = PDSC_REGS64,
         [1] = PDSC_REGS64,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_SNA] = 2,
         [PDSC_MOD_ALUM] = 3,
      },
   },
   [PDSC_OP_ADD32] = {
      .valid_dst_types = PDSC_REGS32TP,
      .valid_src_types = {
         [0] = PDSC_REGS32,
         [1] = PDSC_REGS32,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_SNA] = 2,
         [PDSC_MOD_ALUM] = 3,
      },
   },
   [PDSC_OP_CMP] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64TP,
         [1] = PDSC_REGS64 | PDSC_IMM,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_COP] = 2,
      },
   },
   [PDSC_OP_BRA] = {
      .valid_dst_types = 0,
      .valid_src_types = { 0 },
      .mod_map = {
         [PDSC_MOD_SRCC] = 1,
         [PDSC_MOD_SETC] = 2,
         [PDSC_MOD_NEG] = 3,
      },
   },
   [PDSC_OP_LD64] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_ST32] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_IDIV] = {
      .valid_dst_types = PDSC_REGS32TP,
      .valid_src_types = {
         [0] = PDSC_REGS32,
         [1] = PDSC_REGS32,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_STMP] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64,
         [1] = PDSC_REGS64T,
         [2] = PDSC_REGS64,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_CC_SO_OVF] = 2,
         [PDSC_MOD_CC_GLOBAL_OVF] = 3,
         [PDSC_MOD_TST] = 4,
         [PDSC_MOD_SO] = 5,
      },
   },
   [PDSC_OP_AA] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64,
         [0] = PDSC_REGS32,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_WDF] = {
      .valid_dst_types = 0,
      .valid_src_types = { 0 },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_IDF] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_POL] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64,
         [1] = PDSC_REGS32,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_LIMM] = {
      .valid_dst_types = PDSC_REGS32T,
      .valid_src_types = {
         [0] = PDSC_IMM | PDSC_GLOBAL,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_LOCK] = {
      .valid_dst_types = 0,
      .valid_src_types = { 0 },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_RELEASE] = {
      .valid_dst_types = 0,
      .valid_src_types = { 0 },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_HALT] = {
      .valid_dst_types = 0,
      .valid_src_types = { 0 },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_NOP] = {
      .valid_dst_types = 0,
      .valid_src_types = { 0 },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
      },
   },
   [PDSC_OP_STMC] = {
      .valid_dst_types = 0,
      .valid_src_types = { 0 },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_SOMASK] = 2,
      },
   },
   [PDSC_OP_DDMAD] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS32,
         [1] = PDSC_REGS32T,
         [2] = PDSC_REGS64,
         [3] = PDSC_REGS64C,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_END] = 2,
      },
   },
   [PDSC_OP_DOUT] = {
      .valid_dst_types = 0,
      .valid_src_types = {
         [0] = PDSC_REGS64 | PDSC_NULL,
         [1] = PDSC_REGS32 | PDSC_NULL,
         [2] = PDSC_REGS64 | PDSC_NULL,
      },
      .mod_map = {
         [PDSC_MOD_CC] = 1,
         [PDSC_MOD_END] = 2,
         [PDSC_MOD_DSTDOUT] = 3,
      },
   },
};

pdsc_ref pdsc_src_get(const pdsc_program *p, pdsc_instr_t i, uint8_t n)
{
   assert(i < util_dynarray_num_elements(&p->instrs, pdsc_instr));
   pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);

   assert(n < _i->num_srcs);

   return _i->src[n];
}

void pdsc_src_set(pdsc_program *p, pdsc_instr_t i, uint8_t n, pdsc_ref ref)
{
   assert(i < util_dynarray_num_elements(&p->instrs, pdsc_instr));
   pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);

   assert(n < _i->num_srcs);
   _i->src[n] = ref;
}

pdsc_ref pdsc_dst_get(const pdsc_program *p, pdsc_instr_t i)
{
   assert(i < util_dynarray_num_elements(&p->instrs, pdsc_instr));
   pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);

   assert(_i->op < ARRAY_SIZE(pdsc_op_info));
   ASSERTED const struct pdsc_op_info *info = &pdsc_op_info[_i->op];
   assert(info->valid_dst_types);

   return _i->dst;
}

void pdsc_dst_set(pdsc_program *p, pdsc_instr_t i, pdsc_ref ref)
{
   assert(i < util_dynarray_num_elements(&p->instrs, pdsc_instr));
   pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);

   assert(_i->op < ARRAY_SIZE(pdsc_op_info));
   ASSERTED const struct pdsc_op_info *info = &pdsc_op_info[_i->op];
   assert(info->valid_dst_types);

   _i->dst = ref;
}

static inline uint8_t _pdsc_op_mod_map(const pdsc_instr *_i, enum pdsc_mod mod)
{
   assert(_i->op < ARRAY_SIZE(pdsc_op_info));
   const struct pdsc_op_info *info = &pdsc_op_info[_i->op];

   assert(mod < ARRAY_SIZE(info->mod_map));
   return info->mod_map[mod];
}

static inline bool _pdsc_has_mod(const pdsc_instr *_i, enum pdsc_mod mod)
{
   return _pdsc_op_mod_map(_i, mod) > 0;
}

static inline void
_pdsc_set_mod(pdsc_instr *_i, enum pdsc_mod mod, uint8_t value)
{
   assert(_pdsc_has_mod(_i, mod));
   _i->mod[_pdsc_op_mod_map(_i, mod) - 1] = value;
}

static inline uint8_t
pdsc_op_mod_map(const pdsc_program *p, pdsc_instr_t i, enum pdsc_mod mod)
{
   assert(i < util_dynarray_num_elements(&p->instrs, pdsc_instr));
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);

   return _pdsc_op_mod_map(_i, mod);
}

bool pdsc_has_mod(const pdsc_program *p, pdsc_instr_t i, enum pdsc_mod mod)
{
   return pdsc_op_mod_map(p, i, mod) > 0;
}

uint8_t pdsc_get_mod(const pdsc_program *p, pdsc_instr_t i, enum pdsc_mod mod)
{
   assert(pdsc_has_mod(p, i, mod));

   pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   return _i->mod[pdsc_op_mod_map(p, i, mod) - 1];
}

void pdsc_set_mod(pdsc_program *p,
                  pdsc_instr_t i,
                  enum pdsc_mod mod,
                  uint8_t value)
{
   assert(pdsc_has_mod(p, i, mod));
   pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   _i->mod[pdsc_op_mod_map(p, i, mod) - 1] = value;
}

/**/

static pdsc_instr *pdsc_append_instr(pdsc_program *p,
                                     enum pdsc_op op,
                                     pdsc_ref dst,
                                     uint8_t num_srcs,
                                     pdsc_ref src[num_srcs])
{
   pdsc_instr _i = {
      .op = op,
      .dst = dst,
      .num_srcs = num_srcs,
   };

   if (num_srcs)
      memcpy(_i.src, src, sizeof(*src) * num_srcs);
   util_dynarray_append(&p->instrs, _i);
   return util_dynarray_top_ptr(&p->instrs, pdsc_instr);
}

/**/

pdsc_instr_t _pdsc_sftlp64(pdsc_program *p,
                           pdsc_ref dst,
                           pdsc_ref src0,
                           pdsc_ref src1,
                           pdsc_ref src2,
                           struct pdsc_sftlp_mods mods)

{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_SFTLP64,
                                      dst,
                                      3,
                                      (pdsc_ref[]){ src0, src1, src2 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_LOP, mods.lop);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_sftlp32(pdsc_program *p,
                           pdsc_ref dst,
                           pdsc_ref src0,
                           pdsc_ref src1,
                           pdsc_ref src2,
                           struct pdsc_sftlp_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_SFTLP32,
                                      dst,
                                      3,
                                      (pdsc_ref[]){ src0, src1, src2 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_LOP, mods.lop);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_stm(pdsc_program *p,
                       pdsc_ref src0,
                       pdsc_ref src1,
                       pdsc_ref src2,
                       pdsc_ref src3,
                       struct pdsc_stm_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_STM,
                                      pdsc_ref_null(),
                                      4,
                                      (pdsc_ref[]){ src0, src1, src2, src3 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_CC_SO_OVF, mods.cc_so_ovf);
   _pdsc_set_mod(_i, PDSC_MOD_CC_GLOBAL_OVF, mods.cc_global_ovf);
   _pdsc_set_mod(_i, PDSC_MOD_TST, mods.tst);
   _pdsc_set_mod(_i, PDSC_MOD_SO, mods.so);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_mad(pdsc_program *p,
                       pdsc_ref dst,
                       pdsc_ref src0,
                       pdsc_ref src1,
                       pdsc_ref src2,
                       struct pdsc_add_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_MAD,
                                      dst,
                                      3,
                                      (pdsc_ref[]){ src0, src1, src2 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_SNA, mods.sna);
   _pdsc_set_mod(_i, PDSC_MOD_ALUM, mods.alum);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_add64(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_add_mods mods)
{
   pdsc_instr *_i =
      pdsc_append_instr(p, PDSC_OP_ADD64, dst, 2, (pdsc_ref[]){ src0, src1 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_SNA, mods.sna);
   _pdsc_set_mod(_i, PDSC_MOD_ALUM, mods.alum);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_add32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_add_mods mods)
{
   pdsc_instr *_i =
      pdsc_append_instr(p, PDSC_OP_ADD32, dst, 2, (pdsc_ref[]){ src0, src1 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_SNA, mods.sna);
   _pdsc_set_mod(_i, PDSC_MOD_ALUM, mods.alum);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_cmp(pdsc_program *p,
                       pdsc_ref src0,
                       pdsc_ref src1,
                       struct pdsc_cmp_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_CMP,
                                      pdsc_ref_null(),
                                      2,
                                      (pdsc_ref[]){ src0, src1 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_COP, mods.cop);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t
_pdsc_bra(pdsc_program *p, pdsc_instr_t target_instr, struct pdsc_bra_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p, PDSC_OP_BRA, pdsc_ref_null(), 0, NULL);

   _pdsc_set_mod(_i, PDSC_MOD_SRCC, mods.srcc);
   _pdsc_set_mod(_i, PDSC_MOD_SETC, mods.setc);
   _pdsc_set_mod(_i, PDSC_MOD_NEG, mods.neg);

   _i->target_instr = target_instr;
   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t
_pdsc_ld64(pdsc_program *p, pdsc_ref src0, struct pdsc_cc_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_LD64,
                                      pdsc_ref_null(),
                                      1,
                                      (pdsc_ref[]){ src0 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t
_pdsc_st32(pdsc_program *p, pdsc_ref src0, struct pdsc_cc_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_ST32,
                                      pdsc_ref_null(),
                                      1,
                                      (pdsc_ref[]){ src0 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_idiv(pdsc_program *p,
                        pdsc_ref dst,
                        pdsc_ref src0,
                        pdsc_ref src1,
                        struct pdsc_cc_mods mods)
{
   pdsc_instr *_i =
      pdsc_append_instr(p, PDSC_OP_IDIV, dst, 2, (pdsc_ref[]){ src0, src1 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_stmp(pdsc_program *p,
                        pdsc_ref src0,
                        pdsc_ref src1,
                        pdsc_ref src2,
                        struct pdsc_stm_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_STMP,
                                      pdsc_ref_null(),
                                      3,
                                      (pdsc_ref[]){ src0, src1, src2 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_CC_SO_OVF, mods.cc_so_ovf);
   _pdsc_set_mod(_i, PDSC_MOD_CC_GLOBAL_OVF, mods.cc_global_ovf);
   _pdsc_set_mod(_i, PDSC_MOD_TST, mods.tst);
   _pdsc_set_mod(_i, PDSC_MOD_SO, mods.so);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_aa(pdsc_program *p,
                      pdsc_ref src0,
                      pdsc_ref src1,
                      struct pdsc_cc_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_AA,
                                      pdsc_ref_null(),
                                      2,
                                      (pdsc_ref[]){ src0, src1 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_wdf(pdsc_program *p, struct pdsc_cc_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p, PDSC_OP_WDF, pdsc_ref_null(), 0, NULL);

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_idf(pdsc_program *p, pdsc_ref src0, struct pdsc_cc_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_IDF,
                                      pdsc_ref_null(),
                                      1,
                                      (pdsc_ref[]){ src0 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_pol(pdsc_program *p,
                       pdsc_ref src0,
                       pdsc_ref src1,
                       struct pdsc_cc_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_IDF,
                                      pdsc_ref_null(),
                                      2,
                                      (pdsc_ref[]){ src0, src1 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_limm(pdsc_program *p,
                        pdsc_ref dst,
                        pdsc_ref src0,
                        struct pdsc_cc_mods mods)
{
   pdsc_instr *_i =
      pdsc_append_instr(p, PDSC_OP_LIMM, dst, 1, (pdsc_ref[]){ src0 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_lock(pdsc_program *p, struct pdsc_cc_mods mods)
{
   pdsc_instr *_i =
      pdsc_append_instr(p, PDSC_OP_LOCK, pdsc_ref_null(), 0, NULL);

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_release(pdsc_program *p, struct pdsc_cc_mods mods)
{
   pdsc_instr *_i =
      pdsc_append_instr(p, PDSC_OP_RELEASE, pdsc_ref_null(), 0, NULL);

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_halt(pdsc_program *p, struct pdsc_cc_mods mods)
{
   pdsc_instr *_i =
      pdsc_append_instr(p, PDSC_OP_HALT, pdsc_ref_null(), 0, NULL);

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_nop(pdsc_program *p, struct pdsc_cc_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p, PDSC_OP_NOP, pdsc_ref_null(), 0, NULL);

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_stmc(pdsc_program *p, struct pdsc_stmc_mods mods)
{
   pdsc_instr *_i =
      pdsc_append_instr(p, PDSC_OP_STMC, pdsc_ref_null(), 0, NULL);

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_SOMASK, mods.somask);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_ddmad(pdsc_program *p,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         pdsc_ref src2,
                         pdsc_ref src3,
                         struct pdsc_ddmad_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_DDMAD,
                                      pdsc_ref_null(),
                                      4,
                                      (pdsc_ref[]){ src0, src1, src2, src3 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_END, mods.end);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

pdsc_instr_t _pdsc_dout(pdsc_program *p,
                        pdsc_ref src0,
                        pdsc_ref src1,
                        pdsc_ref src2,
                        struct pdsc_dout_mods mods)
{
   pdsc_instr *_i = pdsc_append_instr(p,
                                      PDSC_OP_DOUT,
                                      pdsc_ref_null(),
                                      3,
                                      (pdsc_ref[]){ src0, src1, src2 });

   _pdsc_set_mod(_i, PDSC_MOD_CC, mods.cc);
   _pdsc_set_mod(_i, PDSC_MOD_END, mods.end);
   _pdsc_set_mod(_i, PDSC_MOD_DSTDOUT, mods.dstdout);

   return util_dynarray_num_elements(&p->instrs, pdsc_instr) - 1;
}

/**/

pdsc_instr_t _pdsc_doutd(pdsc_program *p,
                         uint8_t src0_id,
                         uint8_t src1_id,
                         bool init_srcs,
                         struct pdsc_doutd_params params)
{
   char name[16];

   params.mods.dstdout = PDSC_DSTDOUT_D;

   sprintf(name, "doutd%u_src0", p->num_doutds);
   pdsc_ref src0 = pdsc_find_or_alloc_const64(p, name, PDSC_TYPE_DOUTD_SRC0, src0_id);

   if (init_srcs) {
      uint64_t packed_src0 = pdsc_doutd_src0_pack(p, &params.src0);
      pdsc_assign_const64(p, src0, packed_src0);
   }

   sprintf(name, "doutd%u_src1", p->num_doutds);
   pdsc_ref src1 = pdsc_find_or_alloc_const32(p, name, PDSC_TYPE_DOUTD_SRC1, src1_id);

   if (init_srcs) {
      uint32_t packed_src1 = pdsc_doutd_src1_pack(p, &params.src1);
      pdsc_assign_const32(p, src1, packed_src1);
   }

   ++p->num_doutds;

   return _pdsc_dout(p, src0, src1, pdsc_ref_null(), params.mods);
}

/**/

pdsc_instr_t _pdsc_doutw(pdsc_program *p,
                         pdsc_ref src0,
                         uint8_t src1_id,
                         pdsc_ref src2,
                         bool init_srcs,
                         struct pdsc_doutw_params params)
{
   char name[16];

   params.mods.dstdout = PDSC_DSTDOUT_W;

   sprintf(name, "doutw%u_src1", p->num_doutws);
   pdsc_ref src1 = pdsc_alloc_const32(p, name, PDSC_TYPE_DOUTW_SRC1, src1_id);

   if (init_srcs) {
      uint32_t packed_src1 = pdsc_doutw_src1_pack(p, &params.src1);
      pdsc_assign_const32(p, src1, packed_src1);
   }

   ++p->num_doutws;

   return _pdsc_dout(p, src0, src1, src2, params.mods);
}

/**/

pdsc_instr_t _pdsc_doutu(pdsc_program *p,
                         uint8_t src0_id,
                         bool init_srcs,
                         struct pdsc_doutu_params params)
{
   char name[16];

   params.mods.dstdout = PDSC_DSTDOUT_U;

   sprintf(name, "doutu%u_src0", p->num_doutus);
   pdsc_ref src0 =
      pdsc_find_or_alloc_const64(p, name, PDSC_TYPE_DOUTU_SRC0, src0_id);

   if (init_srcs) {
      uint32_t packed_src0 = pdsc_doutu_src0_pack(p, &params.src0);
      pdsc_assign_const64(p, src0, packed_src0);
   }

   ++p->num_doutus;

   return _pdsc_dout(p, src0, pdsc_ref_null(), pdsc_ref_null(), params.mods);
}

/**/

pdsc_instr_t
_pdsc_doutv(pdsc_program *p, pdsc_ref src1, struct pdsc_dout_mods mods)
{
   mods.dstdout = PDSC_DSTDOUT_V;

   ++p->num_doutvs;

   return _pdsc_dout(p, pdsc_ref_null(), src1, pdsc_ref_null(), mods);
}

/**/

pdsc_instr_t _pdsc_douti(pdsc_program *p,
                         uint8_t src0_id,
                         bool init_srcs,

                         struct pdsc_douti_params params)
{
   char name[16];

   params.mods.dstdout = PDSC_DSTDOUT_I;

   sprintf(name, "douti%u_src0", p->num_doutis);
   pdsc_ref src0 = pdsc_alloc_const64(p, name, PDSC_TYPE_DOUTI_SRC0, src0_id);

   if (init_srcs) {
      if (!params.src0.f16)
         params.src0.f16_offset = params.src0.f32_offset;

      uint64_t packed_src0 = pdsc_douti_src0_pack(p, &params.src0);
      pdsc_assign_const64(p, src0, packed_src0);
   }

   ++p->num_doutis;

   return _pdsc_dout(p, src0, pdsc_ref_null(), pdsc_ref_null(), params.mods);
}

/**/

pdsc_instr_t _pdsc_doutc(pdsc_program *p, struct pdsc_dout_mods mods)
{
   mods.dstdout = PDSC_DSTDOUT_C;
   ++p->num_doutcs;
   return _pdsc_dout(p, pdsc_ref_null(), pdsc_ref_null(), pdsc_ref_null(), mods);
}

/**/

static pdsc_ref pdsc_find_existing_const_value(const pdsc_program *p,
                                               bool is_64bit,
                                               uint64_t value)
{
   uint8_t index;
   pdsc_foreach_assigned_const_index (p, index) {
      /* Make sure the bits match. */
      if (!!is_64bit != !!BITSET_TEST(p->consts_64, index))
         continue;

      /* Make sure the value matches. */
      uint64_t const_value = pdsc_get_assigned_const(p, index, is_64bit);
      if (value != const_value)
         continue;

      /* Don't risk re-using consts with assigned patch IDs,
       * as they could get changed.
       */
      if (pdsc_get_const_patch_id(p, index) != PDSC_CONST_PATCH_ID_NONE)
         continue;

      return is_64bit ? pdsc_ref_const64(index) : pdsc_ref_const32(index);
   }

   return pdsc_ref_null();
}

static pdsc_ref
pdsc_find_or_alloc_const_value(pdsc_program *p, bool is_64bit, uint64_t value)
{
   pdsc_ref const_src = pdsc_find_existing_const_value(p, is_64bit, value);
   if (pdsc_ref_is_null(const_src)) {
      char name[64];
      sprintf(name, "const%s_%" PRIu64, is_64bit ? "64" : "32", value);
      if (is_64bit) {
         const_src = pdsc_alloc_const64(p,
                                        name,
                                        PDSC_TYPE_IMM,
                                        PDSC_CONST_PATCH_ID_NONE);
         pdsc_assign_const64(p, const_src, value);
      } else {
         const_src = pdsc_alloc_const32(p,
                                        name,
                                        PDSC_TYPE_IMM,
                                        PDSC_CONST_PATCH_ID_NONE);
         pdsc_assign_const32(p, const_src, value);
      }
   }

   return const_src;
}

pdsc_instr_t _pdsc_mov32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src,
                         struct pdsc_cc_mods mods)
{
   pdsc_ref const_zero_src;
   assert(pdsc_ref_is_temp32(dst) || pdsc_ref_is_ptemp32(dst));

   switch (src.type) {
   case PDSC_REF_TYPE_IMM:
      if (src.val <= UINT16_MAX && pdsc_ref_is_temp32(dst))
         return pdsc_limm(p, dst, src, .cc = mods.cc);

      src = pdsc_find_or_alloc_const_value(p, false, src.val);
      FALLTHROUGH;

   case PDSC_REF_TYPE_CONST_REG:
   case PDSC_REF_TYPE_PTEMP_REG:
      const_zero_src = pdsc_find_or_alloc_const_value(p, false, 0);
      return pdsc_add32(p, dst, src, const_zero_src, .cc = mods.cc);

   case PDSC_REF_TYPE_TEMP_REG:
      if (pdsc_ref_is_temp32(dst))
         return pdsc_sftlp32(p,
                             dst,
                             src,
                             pdsc_ref_null(),
                             pdsc_ref_null(),
                             .cc = mods.cc);

      const_zero_src = pdsc_find_or_alloc_const_value(p, false, 0);
      return pdsc_add32(p, dst, src, const_zero_src, .cc = mods.cc);

   case PDSC_REF_TYPE_GLOBAL_REG:
      assert(pdsc_ref_is_temp32(dst));
      return pdsc_limm(p, dst, src, .cc = mods.cc);

   default:
      break;
   }

   UNREACHABLE("Invalid src type.");
}

pdsc_instr_t _pdsc_mov64(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src,
                         struct pdsc_cc_mods mods)
{
   pdsc_ref const_zero_src;
   assert(pdsc_ref_is_temp64(dst) || pdsc_ref_is_ptemp64(dst));

   switch (src.type) {
   case PDSC_REF_TYPE_IMM:
      src = pdsc_find_or_alloc_const_value(p, true, src.val);
      FALLTHROUGH;

   case PDSC_REF_TYPE_CONST_REG:
      const_zero_src = pdsc_find_or_alloc_const_value(p, true, 0);
      return pdsc_add64(p, dst, src, const_zero_src, .cc = mods.cc);

   case PDSC_REF_TYPE_PTEMP_REG:
   case PDSC_REF_TYPE_TEMP_REG:
      return pdsc_sftlp64(p,
                          dst,
                          src,
                          pdsc_ref_null(),
                          pdsc_ref_null(),
                          .cc = mods.cc);

   default:
      break;
   }

   UNREACHABLE("Invalid src type.");
}

pdsc_instr_t pdsc_setc(pdsc_program *p, enum pdsc_pred setc)
{
   pdsc_instr_t next_i = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   return pdsc_bra(p,
                   next_i + 1,
                   .srcc = PDSC_PRED_KEEP,
                   .setc = setc,
                   .neg = false);
}

pdsc_instr_t pdsc_branch_push(pdsc_program *p, enum pdsc_pred srcc, bool neg)
{
   return pdsc_bra(p,
                   PDSC_TARGET_INSTR_NONE,
                   .srcc = srcc,
                   .setc = PDSC_PRED_KEEP,
                   .neg = neg);
}

void pdsc_branch_pop(pdsc_program *p, pdsc_instr_t i)
{
   pdsc_instr_t next_i = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   assert(i < next_i);

   pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   assert(_i->op == PDSC_OP_BRA);
   assert(_i->target_instr == PDSC_TARGET_INSTR_NONE);

   _i->target_instr = next_i;

   /* TODO: validate that this ends up being a valid destination. */
}

pdsc_instr_t _pdsc_shl32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         uint8_t shift,
                         struct pdsc_cc_mods mods)
{
   assert(shift <= 32);
   return pdsc_sftlp32(p,
                       dst,
                       src0,
                       pdsc_ref_null(),
                       pdsc_ref_simm(shift),
                       .cc = mods.cc,
                       .lop = PDSC_LOP_NONE);
}

pdsc_instr_t _pdsc_shr32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         uint8_t shift,
                         struct pdsc_cc_mods mods)
{
   assert(shift <= 32);
   return pdsc_sftlp32(p,
                       dst,
                       src0,
                       pdsc_ref_null(),
                       pdsc_ref_simm(-shift),
                       .cc = mods.cc,
                       .lop = PDSC_LOP_NONE);
}

pdsc_instr_t _pdsc_shr64(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         uint8_t shift,
                         struct pdsc_cc_mods mods)
{
   assert(shift <= 64);
   return pdsc_sftlp64(p,
                       dst,
                       src0,
                       pdsc_ref_null(),
                       pdsc_ref_simm(-shift),
                       .cc = mods.cc,
                       .lop = PDSC_LOP_NONE);
}

pdsc_instr_t _pdsc_and32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_cc_mods mods)
{
   return pdsc_sftlp32(p,
                       dst,
                       src0,
                       src1,
                       pdsc_ref_null(),
                       .cc = mods.cc,
                       .lop = PDSC_LOP_AND);
}

pdsc_instr_t _pdsc_or32(pdsc_program *p,
                        pdsc_ref dst,
                        pdsc_ref src0,
                        pdsc_ref src1,
                        struct pdsc_cc_mods mods)
{
   return pdsc_sftlp32(p,
                       dst,
                       src0,
                       src1,
                       pdsc_ref_null(),
                       .cc = mods.cc,
                       .lop = PDSC_LOP_OR);
}

pdsc_instr_t _pdsc_xor32(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_cc_mods mods)
{
   return pdsc_sftlp32(p,
                       dst,
                       src0,
                       src1,
                       pdsc_ref_null(),
                       .cc = mods.cc,
                       .lop = PDSC_LOP_XOR);
}

pdsc_instr_t _pdsc_xor64(pdsc_program *p,
                         pdsc_ref dst,
                         pdsc_ref src0,
                         pdsc_ref src1,
                         struct pdsc_cc_mods mods)
{
   return pdsc_sftlp64(p,
                       dst,
                       src0,
                       src1,
                       pdsc_ref_null(),
                       .cc = mods.cc,
                       .lop = PDSC_LOP_XOR);
}

pdsc_instr_t _pdsc_ld(pdsc_program *p,
                      pdsc_ref dest,
                      uint8_t dest_ref_id,
                      uint8_t src0_id,
                      bool init_srcs,
                      struct pdsc_ld_params params)
{
   char name[16];

   pdsc_save_ref(p, dest_ref_id, dest);
   params.src0.dest = dest;

   sprintf(name, "ld%u_src0", p->num_lds);
   pdsc_ref src0 = pdsc_alloc_const64(p, name, PDSC_TYPE_LD64_SRC0, src0_id);

   if (init_srcs) {
      uint64_t packed_src0 = pdsc_ld64_src0_pack(p, &params.src0);
      pdsc_assign_const64(p, src0, packed_src0);
   }

   ++p->num_lds;

   return _pdsc_ld64(p, src0, params.mods);
}

pdsc_instr_t _pdsc_st(pdsc_program *p,
                      pdsc_ref src,
                      uint8_t src_ref_id,
                      uint8_t src0_id,
                      bool init_srcs,
                      struct pdsc_st_params params)
{
   char name[16];

   pdsc_save_ref(p, src_ref_id, src);
   params.src0.src = src;

   sprintf(name, "st%u_src0", p->num_sts);
   pdsc_ref src0 = pdsc_alloc_const64(p, name, PDSC_TYPE_ST32_SRC0, src0_id);

   if (init_srcs) {
      uint64_t packed_src0 = pdsc_st32_src0_pack(p, &params.src0);
      pdsc_assign_const64(p, src0, packed_src0);
   }

   ++p->num_sts;

   return _pdsc_st32(p, src0, params.mods);
}
