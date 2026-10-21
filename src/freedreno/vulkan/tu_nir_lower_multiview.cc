/*
 * Copyright © 2020 Valve Corporation
 * SPDX-License-Identifier: MIT
 */

#include "nir/nir_builder.h"

#include "ir3/ir3_nir.h"
#include "tu_device.h"
#include "tu_shader.h"

/* Some a6xx variants cannot support a non-contiguous multiview mask. Instead,
 * inside the shader something like this needs to be inserted:
 *
 * gl_Position = ((1ull << gl_ViewIndex) & view_mask) ? gl_Position : vec4(0.);
 *
 * Scan backwards until we find the gl_Position write (there should only be
 * one). This also needs to happen with multi-position, which doesn't respect
 * the view mask.
 */
static bool
lower_multiview_mask(nir_shader *nir, uint32_t *mask)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   if (util_is_power_of_two_or_zero(*mask + 1)) {
      return nir_no_progress(impl);
   }

   nir_builder b = nir_builder_create(impl);

   uint32_t old_mask = *mask;
   *mask = BIT(util_logbase2(old_mask) + 1) - 1;

   nir_foreach_block_reverse(block, impl) {
      nir_foreach_instr_reverse(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
         if (intrin->intrinsic != nir_intrinsic_store_deref)
            continue;

         nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
         if (!nir_deref_mode_is(deref, nir_var_shader_out))
            continue;

         nir_variable *var = nir_deref_instr_get_variable(deref);
         if (var->data.location != VARYING_SLOT_POS)
            continue;

         nir_def *orig_src = intrin->src[1].ssa;
         b.cursor = nir_before_instr(instr);

         /* ((1ull << gl_ViewIndex) & mask) != 0 */
         nir_def *cmp =
            nir_i2b(&b, nir_iand(&b, nir_imm_int(&b, old_mask),
                                  nir_ishl(&b, nir_imm_int(&b, 1),
                                           nir_load_view_index(&b))));

         nir_def *src = nir_bcsel(&b, cmp, orig_src, nir_imm_float(&b, 0.));
         nir_src_rewrite(&intrin->src[1], src);

         return nir_progress(true, impl, nir_metadata_control_flow);
      }
   }

   return nir_no_progress(impl);
}

bool
tu_nir_lower_multiview(nir_shader *nir, uint32_t mask, struct tu_device *dev,
                       bool last_stage)
{
   bool progress = false;
   nir_lower_multiview_options options = {
      .view_mask = mask,
      .allowed_per_view_outputs =
         last_stage ? VARYING_BIT_POS : ~0ull,
   };

   if (!dev->physical_device->info->props.supports_multiview_mask &&
       last_stage)
      NIR_PASS(progress, nir, lower_multiview_mask, &options.view_mask);

   unsigned num_views = util_logbase2(mask) + 1;

   /* Blob doesn't apply multipos optimization starting from 11 views
    * even on a650, however in practice, with the limit of 16 views,
    * tests pass on a640/a650 and fail on a630.
    */
   unsigned max_views_for_multipos =
      dev->physical_device->info->props.supports_multiview_mask ? 16 : 10;

   /* Speculatively assign output locations so that we know num_outputs. We
    * will assign output locations for real after this pass.
    */
   nir_assign_io_var_locations(nir, nir_var_shader_out);

   if (!last_stage) {
      /* We will store outputs per-view and loop over all active views in the
       * shader.
       */
      NIR_PASS(progress, nir, nir_lower_multiview, options);

   /* In addition to the generic checks done by NIR, check that we don't
    * overflow VPC with the extra copies of gl_Position.
    */
   } else if (!dev->physical_device->compiler_options.no_multi_pos &&
       num_views <= max_views_for_multipos && nir->num_outputs + (num_views - 1) <= 32 &&
       nir_can_lower_multiview(nir, options)) {
      /* It appears that the multiview mask is ignored when multi-position
       * output is enabled, so we have to write 0 to inactive views ourselves.
       */
      NIR_PASS(progress, nir, lower_multiview_mask, &options.view_mask);

      NIR_PASS(_, nir, nir_lower_multiview, options);

      /* nir_lower_multiview creates a loop that loops over the view mask and
       * uses indirect stores. Since we only support direct
       * store_per_view_output, we have to make sure these indirects are gone.
       * Lowering the IO vars to temporaries will replace them with indirects
       * on registers.
       */
      ir3_nir_lower_io_vars_to_temporaries(nir);

      progress = true;
   }

   return progress;
}

/*
 * Software multiview lowering for devices without HW multiview support.
 *
 * Unlike the HW multiview path which uses per-view position output and the
 * GPU's built-in multiview mode, the SW path works by duplicating draws on
 * the CPU side (one draw per view).  The view index is passed via a driver
 * param (view_index at dword 5, placed after the vec4 that
 * CP_DRAW_INDIRECT_MULTI overwrites), and the VS writes it to
 * VARYING_SLOT_LAYER so the rasterizer targets the correct layer.
 */

static bool
lower_view_index_to_layer_id_filter(const nir_instr *instr,
                                    UNUSED const void *data)
{
   if (instr->type != nir_instr_type_intrinsic)
      return false;
   return nir_instr_as_intrinsic(instr)->intrinsic ==
          nir_intrinsic_load_view_index;
}

static nir_def *
lower_view_index_to_layer_id(nir_builder *b, nir_instr *instr,
                             UNUSED void *data)
{
   return nir_load_layer_id(b);
}

bool
tu_nir_lower_multiview_sw_vs(nir_shader *nir)
{
   assert(nir->info.stage == MESA_SHADER_VERTEX);

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);

   /* If the shader already writes gl_Layer, demote the existing output to a
    * temp so our store takes precedence.
    */
   nir_variable *existing_layer =
      nir_find_variable_with_location(nir, nir_var_shader_out,
                                     VARYING_SLOT_LAYER);
   if (existing_layer) {
      existing_layer->data.mode = nir_var_shader_temp;
      existing_layer->data.location = 0;
      nir_fixup_deref_modes(nir);
   }

   nir_builder b = nir_builder_at(nir_after_impl(impl));

   /* Write gl_Layer = view_index so the rasterizer targets the correct
    * layer.  load_view_index survives to IR3 where it becomes a driver
    * param read.
    */
   nir_variable *layer_var =
      nir_variable_create(nir, nir_var_shader_out,
                          glsl_int_type(), "gl_Layer");
   layer_var->data.location = VARYING_SLOT_LAYER;
   nir_store_var(&b, layer_var, nir_load_view_index(&b), 0x1);
   nir->info.outputs_written |= BITFIELD64_BIT(VARYING_SLOT_LAYER);

   return true;
}

bool
tu_nir_lower_multiview_sw_fs(nir_shader *nir)
{
   assert(nir->info.stage == MESA_SHADER_FRAGMENT);

   /* In the fragment shader, replace load_view_index with load_layer_id.
    * The VS wrote the view index to gl_Layer, so the FS can read it back
    * from the rasterizer-provided layer ID.
    */
   return nir_shader_lower_instructions(nir,
                                        lower_view_index_to_layer_id_filter,
                                        lower_view_index_to_layer_id, NULL);
}
