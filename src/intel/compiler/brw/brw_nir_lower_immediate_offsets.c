/*
 * Copyright (c) 2025 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "compiler/nir/nir_builder.h"
#include "brw_eu.h"
#include "brw_nir.h"

struct state {
   const struct intel_device_info *devinfo;
   bool efficient_64bit;
};

static bool
lower_immediate_offsets(nir_builder *b, nir_intrinsic_instr *intrin, void *data)
{
   struct state *state = data;
   uint32_t max_bits = UINT32_MAX;
   enum lsc_addr_surface_type binding_type;

   switch (intrin->intrinsic) {
   case nir_intrinsic_load_ssbo_intel:
   case nir_intrinsic_load_ubo_uniform_block_intel:
   case nir_intrinsic_load_ssbo_uniform_block_intel:
   case nir_intrinsic_store_ssbo_intel:
   case nir_intrinsic_store_ssbo_block_intel: {
      nir_src *binding = nir_get_io_index_src(intrin);
      const bool has_resource =
         nir_def_is_intrinsic(binding->ssa) &&
         nir_def_as_intrinsic(binding->ssa)->intrinsic ==
         nir_intrinsic_resource_intel;
      bool bti_is_const;
      if (has_resource) {
         nir_intrinsic_instr *resource =
            nir_def_as_intrinsic(binding->ssa);
         bti_is_const = nir_src_is_const(resource->src[1]);
         binding_type =
            (nir_intrinsic_resource_access_intel(resource) &
             nir_resource_intel_bindless) != 0 ? LSC_ADDR_SURFTYPE_BSS :
            (nir_intrinsic_resource_access_intel(resource) &
             nir_resource_intel_internal) != 0 ? LSC_ADDR_SURFTYPE_SS :
            LSC_ADDR_SURFTYPE_BTI;
      } else {
         if (state->efficient_64bit) {
            binding_type = LSC_ADDR_SURFTYPE_BSS;
         } else {
            bti_is_const = nir_src_is_const(*nir_get_io_index_src(intrin));
            binding_type = LSC_ADDR_SURFTYPE_BTI;
         }
      }
      /* The BTI index and the base offset got into the extended descriptor
       * (see BSpec 63997 for the format).
       *
       * When the BTI index constant, the extended descriptor is encoded into
       * the SEND instruction (no need to use the address register, see BSpec
       * 56890). This is referred to as the extended descriptor immediate.
       *
       * When BTI is not a constant, the extended descriptor is put into the
       * address register but only the BTI index part of it. The base offset
       * needs to go in the SEND instruction (see programming note on BSpec
       * 63997).
       *
       * When the extended descriptor is coming from the address register,
       * some of the bits in the SEND instruction cannot be used for the
       * immediate extended descriptor part and that includes bits you would
       * want to use for the base offset... Slow clap to the HW design here.
       *
       * So put set max bits to 0 in that case and set the base offset to 0
       * since it's unusable.
       */
      if (binding_type == LSC_ADDR_SURFTYPE_BTI && !bti_is_const)
         max_bits = 0;
      break;
   }
   case nir_intrinsic_load_global_intel:
   case nir_intrinsic_store_global_intel:
   case nir_intrinsic_load_global_constant_uniform_block_intel:
      binding_type = LSC_ADDR_SURFTYPE_FLAT;
      break;
   default:
      if (nir_is_shared_access(intrin)) {
         binding_type = LSC_ADDR_SURFTYPE_FLAT;
         break;
      }

      return false;
   }

   assert(nir_intrinsic_has_base(intrin));

   if (nir_intrinsic_base(intrin) == 0)
      return false;

   max_bits = MIN2(
      brw_max_immediate_offset_bits(state->devinfo, binding_type,
                                    state->efficient_64bit,
                                    nir_is_shared_access(intrin)),
      max_bits);

   b->cursor = nir_before_instr(&intrin->instr);

   nir_src *offset_src = nir_get_io_offset_src(intrin);

   if (max_bits == 0) {
      nir_src_rewrite(
         offset_src,
         nir_iadd_imm(
            b, offset_src->ssa, nir_intrinsic_base(intrin)));
      nir_intrinsic_set_base(intrin, 0);
      return true;
   }

   const int32_t min = u_intN_min(max_bits);
   const int32_t max = u_intN_max(max_bits);

   const int32_t base = nir_intrinsic_base(intrin);

   /* Xe3P+ : BSpec 71885/72045: Global Offset
    *    "Specified the signed global offset (in number of data size
    *     elements) applied to all addresses in the message"
    *
    * We often convert 8/16bits to D8U32/D16U32, so don't go down lower than 4
    * bytes.
    *
    * TODO: better predict what types are going to be used?
    */
   const unsigned alignment =
      !state->efficient_64bit ? 4 :
      brw_nir_intrinsic_data_element_size(intrin);

   if ((base % alignment) == 0 && base >= min && base <= max)
      return false;

   int32_t addition = (base / (max + 1)) * (max + 1);
   int32_t new_base = base - addition;

   int32_t unaligned = new_base % alignment;
   addition += unaligned;
   new_base -= unaligned;

   assert(new_base >= min && new_base <= max);

   nir_src_rewrite(
      offset_src, nir_iadd_imm(
         b, offset_src->ssa, addition));
   nir_intrinsic_set_base(intrin, new_base);

   return true;
}

bool
brw_nir_lower_immediate_offsets(nir_shader *shader,
                                const struct intel_device_info *devinfo,
                                bool efficient_64bit)
{
   struct state state = {
      .devinfo = devinfo,
      .efficient_64bit = efficient_64bit,
   };
   return nir_shader_intrinsics_pass(shader, lower_immediate_offsets,
                                     nir_metadata_control_flow, &state);
}
