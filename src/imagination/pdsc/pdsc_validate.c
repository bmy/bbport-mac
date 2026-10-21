/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pdsc_validate.c
 *
 * \brief PDS compiler validation functions.
 */

#include "pdsc.h"
#include "pdsc_internal.h"

#include "util/bitset.h"

#include <assert.h>
#include <stdbool.h>
#include <string.h>

static void pdsc_validate_const_regs(const pdsc_program *p)
{
   BITSET_DECLARE(consts_result, PDSC_MAX_CONSTS);

   /* Ensure all assigned consts are used. */
   BITSET_ANDNOT(consts_result, p->consts_assigned, p->consts_used);
   assert(BITSET_IS_EMPTY(consts_result));

   /* Ensure 64-bit consts are correctly aligned. */
   BITSET_DECLARE(odd_bits, PDSC_MAX_CONSTS);
   memset(odd_bits, 0xaa, sizeof(odd_bits));

   BITSET_AND(consts_result, odd_bits, p->consts_64);
   assert(BITSET_IS_EMPTY(consts_result));
}

static void pdsc_validate_temp_regs(const pdsc_program *p)
{
   BITSET_DECLARE(temps_result, PDSC_MAX_TEMPS);

   /* Ensure 64-bit temps are correctly aligned. */
   BITSET_DECLARE(odd_bits, PDSC_MAX_TEMPS);
   memset(odd_bits, 0xaa, sizeof(odd_bits));

   BITSET_AND(temps_result, odd_bits, p->temps_64);
   assert(BITSET_IS_EMPTY(temps_result));

   /* TODO: validate p->temps_array_elem ? */
}

static void pdsc_validate_ptemp_regs(const pdsc_program *p)
{
   BITSET_DECLARE(ptemps_result, PDSC_MAX_PTEMPS);

   /* Ensure 64-bit ptemps are correctly aligned. */
   BITSET_DECLARE(odd_bits, PDSC_MAX_PTEMPS);
   memset(odd_bits, 0xaa, sizeof(odd_bits));

   BITSET_AND(ptemps_result, odd_bits, p->ptemps_64);
   assert(BITSET_IS_EMPTY(ptemps_result));
}

static bool pdsc_ref_type_is_valid(pdsc_ref ref,
                                   enum pdsc_valid_type valid_types)
{
   if (!valid_types)
      return pdsc_ref_is_null(ref);

   u_foreach_bit (b, valid_types) {
      switch (BITFIELD_BIT(b)) {
      case PDSC_NULL:
         if (pdsc_ref_is_null(ref))
            return true;
         break;

      case PDSC_VALID_TYPE_CONST32:
         if (pdsc_ref_is_const32(ref))
            return true;
         break;

      case PDSC_VALID_TYPE_TEMP32:
         if (pdsc_ref_is_temp32(ref))
            return true;
         break;

      case PDSC_VALID_TYPE_PTEMP32:
         if (pdsc_ref_is_ptemp32(ref))
            return true;
         break;

      case PDSC_VALID_TYPE_CONST64:
         if (pdsc_ref_is_const64(ref))
            return true;
         break;

      case PDSC_VALID_TYPE_TEMP64:
         if (pdsc_ref_is_temp64(ref))
            return true;
         break;

      case PDSC_VALID_TYPE_PTEMP64:
         if (pdsc_ref_is_ptemp64(ref))
            return true;
         break;

      case PDSC_GLOBAL:
         if (pdsc_ref_is_global(ref))
            return true;
         break;

      case PDSC_IMM:
         if (pdsc_ref_is_imm(ref))
            return true;
         break;

      case PDSC_SIMM:
         if (pdsc_ref_is_simm(ref))
            return true;
         break;

      default:
         UNREACHABLE("Invalid dst type in op info.");
      }
   }

   return false;
}

static unsigned pdsc_max_valid_srcs(const struct pdsc_op_info *info)
{
   unsigned u;
   for (u = 0; u < ARRAY_SIZE(info->valid_src_types); ++u)
      if (!info->valid_src_types[u])
         break;

   return u;
}

static void pdsc_validate_ref(const pdsc_program *p, pdsc_ref ref)
{
   if (pdsc_ref_is_null(ref))
      return;

   if (pdsc_ref_is_imm(ref) || pdsc_ref_is_simm(ref)) {
      assert(pdsc_ref_get_chans(ref) == 1);
      assert(!pdsc_ref_get_alias(ref));
      return;
   }

   assert(!pdsc_ref_is_64bit(ref) || !(pdsc_ref_get_val(ref) & 0b1));

   switch (pdsc_ref_get_type(ref)) {
   case PDSC_REF_TYPE_CONST_REG:
      assert(pdsc_ref_get_val(ref) < PDSC_MAX_CONSTS);
      break;

   case PDSC_REF_TYPE_TEMP_REG:
      assert(pdsc_ref_get_val(ref) < PDSC_MAX_TEMPS);
      break;

   case PDSC_REF_TYPE_PTEMP_REG:
      assert(pdsc_ref_get_val(ref) < PDSC_MAX_PTEMPS);
      break;

   case PDSC_REF_TYPE_GLOBAL_REG:
      assert(pdsc_ref_get_val(ref) < PDSC_MAX_GLOBALS);
      break;

   default:
      UNREACHABLE("");
   }
}

/* TODO: make sure null refs still point somewhere valid. */
/* TODO: feature-dependent validation. */

static void pdsc_validate_instr(const pdsc_program *p,
                                pdsc_instr_t i,
                                const pdsc_instr *_i,
                                const struct pdsc_op_info *info,
                                pdsc_instr_t num_instrs)
{
   assert(pdsc_ref_type_is_valid(_i->dst, info->valid_dst_types));
   pdsc_validate_ref(p, _i->dst);

   assert(_i->num_srcs <= pdsc_max_valid_srcs(info));
   for (unsigned u = 0; u < ARRAY_SIZE(_i->src); ++u) {
      assert(pdsc_ref_type_is_valid(_i->src[u], info->valid_src_types[u]));
      pdsc_validate_ref(p, _i->src[u]);
   }

   if (_i->op == PDSC_OP_BRA) {
      assert(_i->target_instr != PDSC_TARGET_INSTR_NONE);
      assert(_i->target_instr < num_instrs);
      assert(_i->target_instr != i);
   }
}

static void pdsc_validate_instrs(const pdsc_program *p)
{
   pdsc_instr_t num_instrs = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   for (pdsc_instr_t i = 0; i < num_instrs; ++i) {
      const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
      const struct pdsc_op_info *info = &pdsc_op_info[_i->op];
      pdsc_validate_instr(p, i, _i, info, num_instrs);
   }
}

/* TODO NEXT: add `const uint32_t *const_buffer` arg like print/patch */
void pdsc_validate_program(const pdsc_program *p)
{
   pdsc_validate_const_regs(p);
   pdsc_validate_temp_regs(p);
   pdsc_validate_ptemp_regs(p);

   pdsc_validate_instrs(p);
}
