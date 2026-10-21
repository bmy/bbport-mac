/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pco_nir_alu.c
 *
 * \brief PCO NIR per-ALU instruction pass.
 */
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_builder_opcodes.h"
#include "compiler/nir/nir_opcodes.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "nir_defines.h"
#include "pco_internal.h"

/**
 * \brief Lowers ALU instructions with float_controls2 decorations.
 *
 * \param[in,out] b NIR builder.
 * \param[in,out] alu ALU instruction to lower.
 * \param[in,out] cb_data additional data.
 * \return True if the pass made progress.
 */
static bool
pco_nir_lower_alu_instr(nir_builder *b, nir_alu_instr *alu, void *cb_data)
{
   if (!nir_alu_instr_is_signed_zero_inf_nan_preserve(alu))
      return false;

   uint32_t old_fp_math_ctrl = b->fp_math_ctrl;
   switch (alu->op) {
   case nir_op_fmax:
   case nir_op_fmin:
      /* All hardware comparison instructions return false if any operand is NaN
       * so a NaN check before is needed.
       * Directly return the other operand if one of the operands is NaN,
       * otherwise return the operation result.
       */
      if (!nir_alu_instr_is_nan_preserve(alu))
         break;

      b->cursor = nir_after_instr(&alu->instr);
      b->fp_math_ctrl = alu->fp_math_ctrl;
      alu->fp_math_ctrl &= ~nir_fp_preserve_nan;

      nir_def *src0_def = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *src1_def = nir_ssa_for_alu_src(b, alu, 1);

      nir_def *fminmax_nan =
         nir_bcsel(b,
                   nir_fisnan(b, src0_def),
                   src1_def,
                   nir_bcsel(b, nir_fisnan(b, src1_def), src0_def, &alu->def));
      nir_def_rewrite_uses_after(&alu->def, fminmax_nan);

      b->fp_math_ctrl = old_fp_math_ctrl;
      return true;

   case nir_op_fsign:
      /* Explicitly check for -0.0 as fsign requires that to be the result in
       * case the input is also -0.0.
       */
      if (!nir_alu_instr_is_signed_zero_preserve(alu))
         break;

      b->fp_math_ctrl = alu->fp_math_ctrl;
      b->cursor = nir_after_instr(&alu->instr);
      nir_def *alu_src = nir_ssa_for_alu_src(b, alu, 0);
      nir_def *fsign_sz = nir_bcsel(b,
                                    nir_feq(b, alu_src, nir_imm_float(b, -0.0)),
                                    nir_imm_float(b, -0.0),
                                    &alu->def);
      alu->fp_math_ctrl &= ~nir_fp_preserve_signed_zero;
      nir_def_rewrite_uses_after(&alu->def, fsign_sz);
      b->fp_math_ctrl = old_fp_math_ctrl;
      return true;

   default:
      break;
   }

   return false;
}

/**
 * \brief Pass that lowers ALU instructions based on special conditions.
 *
 * \param[in,out] shader NIR shader.
 * \return True if the pass made progress.
 */
bool pco_nir_lower_alu(nir_shader *shader)
{
   return nir_shader_alu_pass(shader,
                              pco_nir_lower_alu_instr,
                              nir_metadata_control_flow,
                              NULL);
}
