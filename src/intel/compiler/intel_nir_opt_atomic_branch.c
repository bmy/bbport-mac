/*
 * Copyright 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "intel_nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"

/* This pass replaces selected atomics with a load and a branch around the
 * atomic when the loaded value shows that the atomic would not modify memory.
 * Skipping the atomic can avoid expensive atomic messages, while preserving
 * the atomic return value by using the loaded value on the skipped path.
 *
 * This is not a general atomic simplification.  Replacing an atomic RMW with a
 * load does not preserve the same ordering/linearization effects for arbitrary
 * racing memory.  The max/min cases are intended for locations that are only
 * updated in the direction of the operation: max atomics only raise the value,
 * and min atomics only lower it.  With that property, a pre-load can prove the
 * atomic would not change memory.  The zero-data iadd/umax cases only skip
 * atomics that cannot change the memory value.
 */

static nir_alu_type
image_atomic_type(nir_intrinsic_instr *intrin)
{
   enum pipe_format format = nir_intrinsic_format(intrin);

   assert(util_format_is_pure_integer(format));

   nir_alu_type base_type =
      util_format_is_pure_sint(format) ? nir_type_int : nir_type_uint;

   return base_type | intrin->def.bit_size;
}

static nir_def *
load_for_atomic(nir_builder *b, nir_intrinsic_instr *intrin)
{
   nir_intrinsic_instr *load = NULL;

   switch (intrin->intrinsic) {
   case nir_intrinsic_ssbo_atomic:
      load = nir_intrinsic_instr_create(b->shader, nir_intrinsic_load_ssbo);
      load->src[0] = nir_src_for_ssa(intrin->src[0].ssa);
      load->src[1] = nir_src_for_ssa(intrin->src[1].ssa);
      nir_intrinsic_set_offset_shift(load, nir_intrinsic_offset_shift(intrin));
      nir_intrinsic_set_align(load, intrin->def.bit_size / 8, 0);
      break;

   case nir_intrinsic_shared_atomic:
      load = nir_intrinsic_instr_create(b->shader, nir_intrinsic_load_shared);
      load->src[0] = nir_src_for_ssa(intrin->src[0].ssa);
      nir_intrinsic_set_base(load, nir_intrinsic_base(intrin));
      nir_intrinsic_set_align(load, intrin->def.bit_size / 8, 0);
      break;

   case nir_intrinsic_global_atomic:
      load = nir_intrinsic_instr_create(b->shader, nir_intrinsic_load_global);
      load->src[0] = nir_src_for_ssa(intrin->src[0].ssa);
      nir_intrinsic_set_align(load, intrin->def.bit_size / 8, 0);
      break;

   case nir_intrinsic_image_atomic:
   case nir_intrinsic_bindless_image_atomic:
      load = nir_intrinsic_instr_create(b->shader,
         intrin->intrinsic == nir_intrinsic_image_atomic ? nir_intrinsic_image_load
                                                         : nir_intrinsic_bindless_image_load);
      load->src[0] = nir_src_for_ssa(intrin->src[0].ssa);
      load->src[1] = nir_src_for_ssa(intrin->src[1].ssa);
      load->src[2] = nir_src_for_ssa(intrin->src[2].ssa);
      /* Image atomics have no LOD operand.  The typed image load path does
       * not use this source, so leave it undef to avoid issues with BRW
       * treating all-convergent sources as scalar.
       */
      load->src[3] = nir_src_for_ssa(nir_undef(b, 1, 32));
      nir_intrinsic_set_image_dim(load, nir_intrinsic_image_dim(intrin));
      nir_intrinsic_set_image_array(load, nir_intrinsic_image_array(intrin));
      nir_intrinsic_set_format(load, nir_intrinsic_format(intrin));
      nir_intrinsic_set_dest_type(load, image_atomic_type(intrin));
      break;

   default:
      UNREACHABLE("Unexpected atomic intrinsic");
   }

   unsigned access = nir_intrinsic_access(intrin) & ~ACCESS_COHERENT;
   nir_intrinsic_set_access(load, access);

   load->num_components = 1;
   nir_def_init(&load->instr, &load->def, 1, intrin->def.bit_size);

   nir_builder_instr_insert(b, &load->instr);

   return &load->def;
}

static nir_def *
clone_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin)
{
   nir_intrinsic_instr *cloned =
      nir_instr_as_intrinsic(nir_instr_clone(b->shader, &intrin->instr));
   nir_builder_instr_insert(b, &cloned->instr);
   return &cloned->def;
}

static bool
intel_nir_opt_atomic_branch_intrin(nir_builder *b, nir_intrinsic_instr *intrin,
                                    void *cb_data)
{
   const unsigned enabled_cases = *(unsigned *)cb_data;

   switch (intrin->intrinsic) {
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_shared_atomic:
   case nir_intrinsic_global_atomic:
      break;

   case nir_intrinsic_image_atomic:
   case nir_intrinsic_bindless_image_atomic: {
      /* This pass only synthesizes scalar replacement loads.  Skip image
       * atomics with multi-component results, such as lowered 64-bit formats
       * represented as R32G32.
       */
      if (intrin->def.num_components != 1)
         return false;

      /* There is no 64-bit typed read message.  Since this pass runs after
       * brw_nir_lower_storage_image(), a synthesized 64-bit image_load would
       * not get lowered to R32G32.
       */
      if (intrin->def.bit_size == 64)
         return false;

      enum pipe_format format = nir_intrinsic_format(intrin);
      if (format == PIPE_FORMAT_NONE ||
          !util_format_is_pure_integer(format))
         return false;

      break;
   }

   default:
      return false;
   }

   /* A volatile atomic must always be performed as written. */
   if (nir_intrinsic_has_access(intrin) &&
       (nir_intrinsic_access(intrin) & ACCESS_VOLATILE))
      return false;

   nir_def *data = nir_get_io_data_src(intrin)->ssa;
   if (data->num_components != 1)
      return false;

   b->cursor = nir_before_instr(&intrin->instr);

   nir_def *result = NULL;

   nir_atomic_op op = nir_intrinsic_atomic_op(intrin);
   switch (op) {
   case nir_atomic_op_imin:
   case nir_atomic_op_umin:
      if (enabled_cases & INTEL_ATOMIC_BRANCH_MIN) {
         nir_def *loaded_data = load_for_atomic(b, intrin);
         nir_def *should_update = op == nir_atomic_op_imin ?
            nir_ilt(b, data, loaded_data) :
            nir_ult(b, data, loaded_data);
         nir_def *atomic_data;

         nir_push_if(b, should_update);
         {
            atomic_data = clone_intrinsic(b, intrin);
         }
         nir_pop_if(b, NULL);

         result = nir_if_phi(b, atomic_data, loaded_data);
         break;
      }

      return false;

   case nir_atomic_op_imax:
   case nir_atomic_op_umax:
      if (enabled_cases & INTEL_ATOMIC_BRANCH_MAX) {
         nir_def *loaded_data = load_for_atomic(b, intrin);
         nir_def *should_update = op == nir_atomic_op_imax ?
            nir_ilt(b, loaded_data, data) :
            nir_ult(b, loaded_data, data);
         nir_def *atomic_data;

         nir_push_if(b, should_update);
         {
            atomic_data = clone_intrinsic(b, intrin);
         }
         nir_pop_if(b, NULL);

         result = nir_if_phi(b, atomic_data, loaded_data);
         break;
      }

      if (op != nir_atomic_op_umax ||
          (enabled_cases & INTEL_ATOMIC_BRANCH_SKIP_ON_ZERO) == 0)
         return false;
      FALLTHROUGH;

   case nir_atomic_op_iadd:
      if (enabled_cases & INTEL_ATOMIC_BRANCH_SKIP_ON_ZERO) {
         nir_def *zero = nir_imm_zero(b, data->num_components, data->bit_size);
         nir_def *loaded_data;
         nir_def *atomic_data;

         nir_push_if(b, nir_ieq(b, data, zero));
         {
            loaded_data = load_for_atomic(b, intrin);
         }
         nir_push_else(b, NULL);
         {
            atomic_data = clone_intrinsic(b, intrin);
         }
         nir_pop_if(b, NULL);

         result = nir_if_phi(b, loaded_data, atomic_data);

         break;
      }

      return false;

   default:
      return false;
   }

   assert(result != NULL);

   nir_def_rewrite_uses(&intrin->def, result);
   nir_instr_remove(&intrin->instr);
   return true;
}

bool
intel_nir_opt_atomic_branch(nir_shader *shader, unsigned enabled_cases)
{
   assert(shader->info.stage == MESA_SHADER_COMPUTE);

   return nir_shader_intrinsics_pass(shader,
                                     intel_nir_opt_atomic_branch_intrin,
                                     nir_metadata_none,
                                     &enabled_cases);
}
