/*
 * Copyright © 2026 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include "etnaviv_nir.h"
#include "compiler/nir/nir_xfb_info.h"

/* the HW vertex id is first + i, not zero based */
static nir_def *
capture_index(nir_builder *b)
{
   nir_def *inst = nir_imul(b, nir_load_instance_id(b),
                            nir_load_num_vertices(b));
   return nir_isub(b, nir_iadd(b, inst, nir_load_vertex_id(b)),
                   nir_load_first_vertex(b));
}

static bool
lower_xfb_store(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   nir_xfb_info *xfb_info = data;

   if (intr->intrinsic != nir_intrinsic_store_deref)
      return false;

   nir_deref_instr *deref = nir_src_as_deref(intr->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);

   if (!var || var->data.mode != nir_var_shader_out)
      return false;

   assert(!var->data.compact);

   unsigned location = var->data.location;
   for (nir_deref_instr *d = deref; d->deref_type != nir_deref_type_var; d = nir_deref_instr_parent(d)) {
      assert(d->deref_type == nir_deref_type_array);
      assert(nir_src_is_const(d->arr.index));
      location += nir_src_as_uint(d->arr.index) *
                  glsl_count_attribute_slots(d->type, false);
   }

   b->cursor = nir_after_instr(&intr->instr);

   nir_def *index = NULL;

   for (unsigned i = 0; i < xfb_info->output_count; i++) {
      const nir_xfb_output_info *out = &xfb_info->outputs[i];

      if (out->location != location)
         continue;

      assert(!out->data_is_16bit);

      /* component_mask and the shifted write mask are relative to the
       * vec4 slot, the stored value starts at location_frac
       */
      const unsigned frac = var->data.location_frac;
      const nir_component_mask_t write_mask = nir_intrinsic_write_mask(intr);
      assert((unsigned)(write_mask << frac) <= 0xf);
      nir_component_mask_t captured = out->component_mask &
                                      (write_mask << frac);

      uint16_t stride = xfb_info->buffers[out->buffer].stride;

      unsigned remaining = captured;
      while (remaining) {
         int lo, count;
         u_bit_scan_consecutive_range(&remaining, &lo, &count);

         /* the stored value is component aligned with the variable, so
          * slot component c is value channel c - frac
          */
         const nir_component_mask_t channels = BITFIELD_RANGE(lo, count) >> frac;

         if (!index)
            index = capture_index(b);

         const unsigned offset = out->offset +
            util_bitcount(out->component_mask & BITFIELD_MASK(lo)) * 4;
         nir_def *base =
            nir_iadd(b, nir_load_xfb_address(b, 32, .base = out->buffer),
                     nir_imul_imm(b, index, stride));

         nir_def *value = nir_channels(b, intr->src[1].ssa, channels);
         nir_store_global_offset(b, value, base, nir_imm_int(b, offset));
      }
   }

   return index != NULL;
}

bool
etna_nir_lower_xfb(nir_shader *shader)
{
   nir_xfb_info *xfb_info = shader->xfb_info;

   if (!xfb_info)
      return false;

   return nir_shader_intrinsics_pass(shader, lower_xfb_store,
                                     nir_metadata_none, xfb_info);
}
