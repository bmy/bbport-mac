/* Copyright © 2026 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

/* This pass replaces FS input load intrinsics with AMD-specific intrinsics where src0 specifies
 * the M0 register value, which must be subgroup-uniform and it's the prim_mask input SGPR.
 *
 * If the IO offset src is non-constant (indirect), it must be subgroup-uniform, and it's added
 * to M0.lds_offset as follows: M0.lds_offset += io_offset * attr_stride;
 */

#include "nir_builder.h"
#include "ac_nir.h"

static bool
lower_intr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input &&
       intr->intrinsic != nir_intrinsic_load_interpolated_input &&
       intr->intrinsic != nir_intrinsic_load_input_vertex &&
       intr->intrinsic != nir_intrinsic_load_per_primitive_input)
      return false;

   b->cursor = nir_before_instr(&intr->instr);

   const struct ac_shader_args *args = (const struct ac_shader_args *)data;
   nir_def *prim_mask = ac_nir_load_arg(b, args, args->prim_mask);
   nir_src *offset = nir_get_io_offset_src(intr);
   nir_scalar s = nir_scalar_resolved(offset->ssa, 0);
   unsigned slot = nir_intrinsic_base(intr);

   if (!nir_scalar_is_const(s)) {
      /* We expect offset to be convergent, but we don't check it because divergence metadata
       * isn't expected to be valid at this point.
       */

      /* From the RDNA 4 ISA PDF (paraphrased):
       *
       * Bits 16:30 have a bit for each quad (i.e. it's a quad mask) indicating that this quad
       * begins a new primitive. Zero indicates same primitive as the previous quad. There is
       * an implied "one" for the first quad in the wave (every wave begins a new primitive) and
       * so the first bit of the quad mask is omitted. Bit 31 is bc_optimize, which should be
       * ignored. Wave32 has only bits 7 bits set in the quad mask since it can have only 8 quads.
       */
      nir_def *actual_prim_mask = nir_ubfe_imm(b, prim_mask, 16, 15);
      nir_def *num_prims = nir_iadd_imm(b, nir_bit_count(b, actual_prim_mask), 1);

      /* Each attribute has 48 bytes (sizeof(vec4[3])). The HW stores attribute 0 for all wave
       * primitives consecutively in LDS first, then attribute 1, etc.
       */
      nir_def *attr_stride = nir_imul_imm(b, num_prims, 48);
      nir_def *rel_lds_offset = nir_imul(b, attr_stride, nir_mov_scalar(b, s));
      nir_def *lds_offset = nir_iadd(b, nir_iand_imm(b, prim_mask, 0xffff), rel_lds_offset);

      prim_mask = nir_ior(b, nir_iand_imm(b, prim_mask, 0xffff0000),
                          nir_iand_imm(b, lds_offset, 0xffff));
   } else {
      /* Fold the offset manually instead of requiring nir_opt_constant_folding. */
      slot += nir_scalar_as_uint(s);
   }

   assert(intr->def.num_components == 1);

   /* Change the intrinsics to AMD. */
   if (intr->intrinsic == nir_intrinsic_load_interpolated_input) {
      nir_def *value =
         nir_load_interpolated_input_amd(b, intr->def.bit_size, intr->src[0].ssa, prim_mask,
                                         .ps_input_info_amd.slot = slot,
                                         .ps_input_info_amd.component = nir_intrinsic_component(intr),
                                         .ps_input_info_amd.high_16bits =
                                            nir_intrinsic_io_semantics(intr).high_16bits);
      nir_def_replace(&intr->def, value);
   } else {
      unsigned vertex_index = intr->intrinsic == nir_intrinsic_load_input_vertex ?
                                 nir_src_as_uint(intr->src[0]) : 0;
      nir_def *value = nir_load_input_vertex_amd(b, prim_mask,
                                                 .ps_input_info_amd.slot = slot,
                                                 .ps_input_info_amd.component = nir_intrinsic_component(intr),
                                                 .ps_input_info_amd.vertex_index = vertex_index);
      if (intr->def.bit_size == 16) {
         value = nir_channel(b, nir_unpack_32_2x16(b, value),
                             nir_intrinsic_io_semantics(intr).high_16bits);
      }

      nir_def_replace(&intr->def, value);
   }

   return true;
}

bool
ac_nir_lower_fs_input_loads(nir_shader *nir, const struct ac_shader_args *args)
{
   return nir_shader_intrinsics_pass(nir, lower_intr, nir_metadata_control_flow, (void*)args);
}
