
/*
 * Copyright (C) 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "bi_builder.h"
#include "va_compiler.h"
#include "valhall.h"

static unsigned
va_instr_exec_time(bi_instr *I, unsigned arch)
{
   switch (I->op) {
   /* CLPER and FROUND.f32 became two-word ops at v11 */
   case BI_OPCODE_CLPER_I32:
   case BI_OPCODE_FROUND_F32:
      return arch >= 11 ? 2 : 1;

   /* MMUL always takes 4 cycles */
   case BI_OPCODE_MMUL_F32:
   case BI_OPCODE_MMUL_V2F16:
   case BI_OPCODE_MMUL_V4S8:
   case BI_OPCODE_MMUL_V4U8:
      return 4;

   default:
      return 1;
   }
}

static enum va_unit
va_arch_adjusted_unit(bi_instr *I, unsigned arch)
{
   switch (I->op) {
   /* 32-bit shift ops moved from SFU to CVT at v11 */
   case BI_OPCODE_LSHIFT_AND_I32:
   case BI_OPCODE_LSHIFT_OR_I32:
   case BI_OPCODE_LSHIFT_XOR_I32:
   case BI_OPCODE_RSHIFT_AND_I32:
   case BI_OPCODE_RSHIFT_OR_I32:
   case BI_OPCODE_RSHIFT_XOR_I32:
   case BI_OPCODE_CLPER_I32:
      return arch >= 11 ? VA_UNIT_CVT : VA_UNIT_SFU;

   /* FROUND.f32 moved from CVT to FMA at v11, */
   case BI_OPCODE_FROUND_F32:
      return arch >= 11 ? VA_UNIT_FMA : VA_UNIT_CVT;

   default:
      return valhall_opcodes[I->op].unit;
   }
}

void
va_count_instr_stats(bi_instr *I, struct va_stats *stats, unsigned arch)
{
   bi_foreach_dest(I, d) {
      if (I->dest[d].type == BI_INDEX_REGISTER)
         stats->reg_mask |= (uint64_t)bi_writemask(I, d) << I->dest[d].value;
   }
   bi_foreach_src(I, s) {
      if (I->src[s].type == BI_INDEX_REGISTER) {
         unsigned pos = I->src[s].offset + I->src[s].value;
         unsigned count = bi_count_read_registers(I, s);
         stats->reg_mask |= ((uint64_t)BITFIELD_MASK(count)) << pos;
      }
      if (I->src[s].type == BI_INDEX_FAU) {
         bi_index index = I->src[s];
         unsigned val = index.value;
         if (val >= BIR_FAU_UNIFORM) {
            val = val & ~BIR_FAU_UNIFORM;
            if (val < BIR_FAU_UNIFORM) {
               stats->nr_fau_uniforms = MAX2(stats->nr_fau_uniforms, val+1);
            }
         }
      }
   }

   /* Adjusted for 64-bit arithmetic */
   unsigned words = bi_count_write_registers(I, 0);
   unsigned cycles = words * va_instr_exec_time(I, arch);

   switch (va_arch_adjusted_unit(I, arch)) {
   /* Arithmetic is 2x slower for 64-bit than 32-bit */
   case VA_UNIT_FMA:
      stats->fma += cycles;
      return;

   case VA_UNIT_CVT:
      stats->cvt += cycles;
      return;

   case VA_UNIT_SFU:
      stats->sfu += cycles;
      return;

   /* Varying is counted int loaded 32-bit components */
   case VA_UNIT_V: {
      /* LD_VAR_SPECIAL.frag_z is free */
      if (I->op == BI_OPCODE_LD_VAR_SPECIAL &&
          I->varying_name == BI_VARYING_NAME_FRAG_Z)
         return;
      bool src16;
      if (I->op == BI_OPCODE_LD_VAR_BUF_F16 ||
          I->op == BI_OPCODE_LD_VAR_BUF_F32 ||
          I->op == BI_OPCODE_LD_VAR_BUF_IMM_F16 ||
          I->op == BI_OPCODE_LD_VAR_BUF_IMM_F32) {
         src16 = I->source_format == BI_SOURCE_FORMAT_F16 ||
                 I->source_format == BI_SOURCE_FORMAT_FLAT16;
      } else {
         src16 = bi_is_regfmt_16(I->register_format);
      }
      stats->v += DIV_ROUND_UP((I->vecsize + 1) * (src16 ? 2 : 4), 4);
      return;
   }

   /* We just count load/store and texturing for now */
   case VA_UNIT_LS:
      stats->ls++;
      return;

   case VA_UNIT_T:
      stats->t++;
      return;

   /* Fused varying+texture loads 2 FP32 components of varying for texture
    * coordinates and then textures */
   case VA_UNIT_VT:
      stats->ls += (2 * 2);
      stats->t++;
      return;

   /* Nothing to do here */
   case VA_UNIT_NONE:
      return;
   }

   UNREACHABLE("Invalid unit");
}
