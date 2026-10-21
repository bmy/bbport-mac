/*
 * Copyright © 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "intel_nir.h"
#include "compiler/nir/nir_builder.h"

struct source_extract {
   nir_intrinsic_instr *resource_intel;
   nir_scalar upper_dword;
   nir_scalar lower_dword;
   unsigned const_index;

   nir_def *ret;
};

static struct source_extract
extract_handle_offset(nir_builder *b,
                      nir_def *vec2_handle,
                      uint32_t max_value,
                      uint32_t align_value)
{
   assert(vec2_handle->num_components == 2 &&
          vec2_handle->bit_size == 32);
   struct source_extract s = {
      .lower_dword = nir_scalar_chase_movs((nir_scalar){ vec2_handle, 0 }),
      .upper_dword = nir_scalar_chase_movs((nir_scalar){ vec2_handle, 1 }),
   };

   nir_instr *instr = nir_def_instr(vec2_handle);
   nir_intrinsic_instr *intrin = NULL;
   if (instr->type == nir_instr_type_intrinsic) {
      intrin = nir_instr_as_intrinsic(instr);
      if (intrin->intrinsic != nir_intrinsic_resource_intel) {
         s.ret = nir_pack_64_2x32(b, vec2_handle);
         return s;
      }

      s.resource_intel = intrin;
      s.lower_dword = nir_scalar_chase_movs((nir_scalar){ intrin->src[1].ssa, 0 });
      s.upper_dword = nir_scalar_chase_movs((nir_scalar){ intrin->src[1].ssa, 1 });
   }

   instr = nir_def_instr(s.lower_dword.def);
   if (instr->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      if (alu->op == nir_op_iadd) {

         nir_scalar src[2] = {
            { alu->src[0].src.ssa, alu->src[0].swizzle[0] },
            { alu->src[1].src.ssa, alu->src[1].swizzle[0] },
         };
         for (uint32_t i = 0; i < 2; i++) {
            src[i] = nir_scalar_chase_movs(src[i]);
            if (!nir_scalar_is_const(src[i]))
               continue;

            int64_t offset = nir_scalar_as_uint(src[i]);
            if ((offset % align_value) != 0)
               break;

            if (offset >= max_value)
               break;

            s.const_index = offset / align_value;
            s.lower_dword = src[i == 0 ? 1 : 0];
            break;
         }
      }
   }

   if (s.const_index) {
      s.ret = nir_pack_64_2x32_split(b,
                                       nir_channel(b, s.lower_dword.def, s.lower_dword.comp),
                                       nir_channel(b, s.upper_dword.def, s.upper_dword.comp));
   } else {
      s.ret = nir_pack_64_2x32(b, s.resource_intel ? s.resource_intel->src[1].ssa : vec2_handle);
   }

   if (s.resource_intel) {
      s.ret = nir_resource_intel(
         b, 1, 64,
         s.resource_intel->src[0].ssa,
         s.ret,
         s.resource_intel->src[2].ssa,
         s.resource_intel->src[3].ssa,
         .desc_set = nir_intrinsic_desc_set(s.resource_intel),
         .binding = nir_intrinsic_binding(s.resource_intel),
         .resource_block_intel = s.const_index,
         .resource_access_intel = nir_intrinsic_resource_access_intel(s.resource_intel));
   }

   return s;
}

static bool
lower_intrinsic(nir_builder *b, nir_intrinsic_instr *intrin, void *_)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ubo_uniform_block_intel:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_load_ssbo_intel:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_store_ssbo_intel:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
   case nir_intrinsic_load_ssbo_block_intel:
   case nir_intrinsic_store_ssbo_block_intel:
   case nir_intrinsic_load_ssbo_uniform_block_intel:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_store:
   case nir_intrinsic_bindless_image_atomic:
   case nir_intrinsic_bindless_image_atomic_swap: {
      b->cursor = nir_before_instr(&intrin->instr);

      nir_src *surface = nir_get_io_index_src(intrin);
      struct source_extract s = extract_handle_offset(
         b, surface->ssa, 64 * 31 /* 5 bits */, 64);
      nir_src_rewrite(surface, s.ret);
      return true;
   }

   default:
      return false;
   }
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex, void *_)
{

   bool progress = false;
   int index;

   index = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   if (index != -1) {
      b->cursor = nir_before_instr(&tex->instr);

      struct source_extract s =
         extract_handle_offset(b, tex->src[index].src.ssa,
                               64 * 31 /* 5 bits */, 64);
      nir_src_rewrite(&tex->src[index].src, s.ret);
      tex->texture_index = s.const_index;
      progress = true;
   }

   index = nir_tex_instr_src_index(tex, nir_tex_src_sampler_handle);
   if (index != -1) {
      b->cursor = nir_before_instr(&tex->instr);

      struct source_extract s =
         extract_handle_offset(b, tex->src[index].src.ssa,
                               32 * 7 /* 3 bits */, 32);
      nir_src_rewrite(&tex->src[index].src, s.ret);
      tex->sampler_index = s.const_index;
      progress = true;
   }

   return progress;
}

static bool
lower_instruction(nir_builder *b, nir_instr *instr, void *data)
{
   switch (instr->type) {
   case nir_instr_type_tex:
      return lower_tex(b, nir_instr_as_tex(instr), data);
   case nir_instr_type_intrinsic:
      return lower_intrinsic(b, nir_instr_as_intrinsic(instr), data);
   default:
      return false;
   }
}

bool
intel_nir_lower_vec2_surface_sampler(nir_shader *nir)
{
   return nir_shader_instructions_pass(nir, lower_instruction,
                                       nir_metadata_control_flow, NULL);
}
