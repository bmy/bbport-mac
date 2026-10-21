/*
 * Copyright © 2026 Valve Corporation
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "nir.h"
#include "nir_builder.h"
#include "nir_deref.h"

/* For vector arrays, DXC produces a scalar array with strided access.
 * Try to detect such arrays and reverse it, because arrays of vectors
 * can be optimized better by other passes like nir_opt_large_constants
 * or nir_opt_shared_vars_to_subgroup.
 */

struct array_info {
   bool has_indirect : 1;
   bool disallow_stride_2 : 1;
   bool disallow_stride_3 : 1;
   bool disallow_stride_4 : 1;
   uint8_t new_stride;
};

struct stride_desc {
   nir_scalar mul_src;
   nir_scalar mul;
   uint32_t stride;
   uint32_t offset;
};

static struct stride_desc
get_stride_desc(nir_scalar idx)
{
   struct stride_desc desc = { 0 };

   if (nir_scalar_is_const(idx)) {
      desc.offset = nir_scalar_as_uint(idx);
      return desc;
   }

   if (nir_scalar_is_alu(idx) &&
       nir_scalar_alu_op(idx) == nir_op_iadd) {
      nir_scalar src0 = nir_scalar_chase_alu_src(idx, 0);
      nir_scalar src1 = nir_scalar_chase_alu_src(idx, 1);
      if (nir_scalar_is_const(src1)) {
         SWAP(src0, src1);
      } else if (!nir_scalar_is_const(src0)) {
         desc.stride = 1;
         return desc;
      }

      desc.offset = nir_scalar_as_uint(src0);
      idx = src1;
   }

   if (!nir_scalar_is_alu(idx)) {
      desc.stride = 1;
      return desc;
   }

   if (nir_scalar_alu_op(idx) == nir_op_imul) {
      nir_scalar src0 = nir_scalar_chase_alu_src(idx, 0);
      nir_scalar src1 = nir_scalar_chase_alu_src(idx, 1);
      if (nir_scalar_is_const(src0)) {
         SWAP(src0, src1);
      } else if (!nir_scalar_is_const(src1)) {
         desc.stride = 1;
         return desc;
      }

      desc.stride = nir_scalar_as_uint(src1);
      desc.mul = idx;
      desc.mul_src = src0;
   } else if (nir_scalar_alu_op(idx) == nir_op_ishl) {
      nir_scalar src1 = nir_scalar_chase_alu_src(idx, 1);
      if (!nir_scalar_is_const(src1)) {
         desc.stride = 1;
         return desc;
      }

      desc.stride = 1u << (nir_scalar_as_uint(src1) & (idx.def->bit_size - 1));
      desc.mul = idx;
      desc.mul_src = nir_scalar_chase_alu_src(idx, 0);
   } else {
      desc.stride = 1;
   }

   return desc;
}

static struct array_info *
get_array_info(struct hash_table *var_stride_map, nir_variable *var)
{
   struct hash_entry *entry = _mesa_hash_table_search(var_stride_map, var);
   return entry ? entry->data : NULL;
}

static bool
mul_is_nuw(nir_shader *shader, const struct stride_desc *stride_desc, struct hash_table *range_ht)
{
   if (nir_scalar_as_alu(stride_desc->mul)->no_unsigned_wrap)
      return true;

   uint64_t uub = nir_unsigned_upper_bound(shader, range_ht, stride_desc->mul_src);
   return uub * stride_desc->stride <= UINT32_MAX;
}

static void
update_array_stride(struct array_info *info, nir_shader *shader,
                    const struct stride_desc *desc, struct hash_table *range_ht)
{
   info->has_indirect |= desc->stride != 0;
   info->disallow_stride_2 |= (desc->stride % 2) != 0;
   info->disallow_stride_3 |= (desc->stride % 3) != 0;
   info->disallow_stride_4 |= (desc->stride % 4) != 0;

   /* imul(udiv(imul(a, 3), 3), 3) is sadly not imul(a, 3)
    * so we can only allow this optimization when overflow
    * does not happen.
    */
   if (!info->disallow_stride_3 && desc->stride && !mul_is_nuw(shader, desc, range_ht))
      info->disallow_stride_3 = true;
}

static bool
init_array_vars_in_list(struct exec_list *vars,
                        nir_variable_mode modes,
                        struct hash_table *var_stride_map,
                        void *mem_ctx)
{
   if (!modes)
      return false;

   bool has_array = false;

   nir_foreach_variable_in_list(var, vars) {
      if (!(var->data.mode & modes))
         continue;

      if (var->data.aliased_shared_memory || !glsl_type_is_array(var->type))
         continue;

      const glsl_type *element_type = glsl_get_array_element(var->type);
      if (!glsl_type_is_scalar(element_type))
         continue;

      struct array_info *info = rzalloc(mem_ctx, struct array_info);

      _mesa_hash_table_insert(var_stride_map, var, info);
      has_array = true;
   }

   return has_array;
}

static bool
update_deref_array_stride(nir_shader *shader, nir_deref_instr *deref_var, struct array_info *info,
                          struct hash_table *range_ht)
{
   nir_foreach_use_including_if(use, &deref_var->def) {
      if (nir_src_is_if(use))
         return false;

      nir_instr *use_instr = nir_src_use_instr(use);
      if (use_instr->type != nir_instr_type_deref)
         return false;

      nir_deref_instr *deref_array = nir_instr_as_deref(use_instr);
      if (deref_array->deref_type != nir_deref_type_array)
         return false;

      if (deref_array->arr.index.ssa->bit_size != 32)
         return false;

      nir_foreach_use_including_if(array_use, &deref_array->def) {
         if (nir_src_is_if(array_use))
            return false;

         nir_instr *array_use_instr = nir_src_use_instr(array_use);
         if (array_use_instr->type != nir_instr_type_intrinsic)
            return false;

         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(array_use_instr);
         switch (intrin->intrinsic) {
         case nir_intrinsic_load_deref:
         case nir_intrinsic_store_deref:
         case nir_intrinsic_deref_atomic:
         case nir_intrinsic_deref_atomic_swap:
            if (array_use != &intrin->src[0])
               return false;
            break;
         default:
            return false;
         }
      }

      struct stride_desc stride_desc = get_stride_desc(nir_scalar_resolved(deref_array->arr.index.ssa, 0));
      update_array_stride(info, shader, &stride_desc, range_ht);
   }

   return true;
}

static void
find_array_stride(nir_function_impl *impl, struct hash_table *var_stride_map, nir_variable_mode modes,
                  struct hash_table *range_ht)
{
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var ||
             !(deref->modes & modes))
            continue;

         struct array_info *info = get_array_info(var_stride_map, deref->var);
         if (!info)
            continue;

         if (!update_deref_array_stride(impl->function->shader, deref, info, range_ht))
            _mesa_hash_table_remove_key(var_stride_map, deref->var);
      }
   }
}

static nir_constant *
create_null_constant(void *ctx, const struct glsl_type *type)
{
   nir_constant *c = rzalloc(ctx, nir_constant);
   c->is_null_constant = true;
   if (glsl_type_is_array_or_matrix(type)) {
      c->num_elements = glsl_get_length(type);
      c->elements = ralloc_array(ctx, nir_constant *, c->num_elements);

      c->elements[0] = create_null_constant(ctx, glsl_get_array_element(type));
      for (unsigned i = 1; i < c->num_elements; i++)
         c->elements[i] = c->elements[0];
   } else {
      assert(glsl_type_is_vector_or_scalar(type));
   }

   return c;
}

static bool
array_to_vec_list(nir_shader *shader, struct exec_list *vars,
                  nir_variable_mode modes, struct hash_table *var_stride_map)
{
   if (!modes)
      return false;

   bool progress = false;
   nir_foreach_variable_in_list(var, vars) {
      if (!(var->data.mode & modes))
         continue;

      struct array_info *info = get_array_info(var_stride_map, var);
      if (!info)
         continue;

      info->new_stride = 0;
      if (!info->has_indirect) {
         /* Without indirect access, don't change anything. */
      } else if (!info->disallow_stride_4) {
         info->new_stride = 4;
      } else if (!info->disallow_stride_3) {
         info->new_stride = 3;
      } else if (!info->disallow_stride_2) {
         info->new_stride = 2;
      }

      if (!info->new_stride) {
         _mesa_hash_table_remove_key(var_stride_map, var);
         continue;
      }

      const glsl_type *scalar_type = glsl_get_array_element(var->type);

      const struct glsl_type *vec_type =
         glsl_vector_type(glsl_get_base_type(scalar_type), info->new_stride);

      uint32_t array_length = DIV_ROUND_UP(glsl_get_length(var->type), info->new_stride);
      const struct glsl_type *new_type = glsl_array_type(vec_type, array_length, 0);

      var->type = new_type;
      if (var->constant_initializer) {
         assert(var->constant_initializer->is_null_constant);
         var->constant_initializer = create_null_constant(shader, new_type);
      }
      progress = true;
   }

   return progress;
}

static void
update_intrin(nir_builder *b, nir_intrinsic_instr *intrin, unsigned new_stride, unsigned vec_offset)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_load_deref: {
      intrin->num_components = new_stride;
      intrin->def.num_components = new_stride;

      b->cursor = nir_after_instr(&intrin->instr);
      nir_def *res = nir_channel(b, &intrin->def, vec_offset);
      nir_def_rewrite_uses_after(&intrin->def, res);
      break;
   }
   case nir_intrinsic_store_deref: {
      intrin->num_components = new_stride;
      b->cursor = nir_before_instr(&intrin->instr);

      nir_def *data[NIR_MAX_VEC_COMPONENTS];
      nir_def *undef = nir_undef(b, 1, intrin->src[1].ssa->bit_size);

      for (unsigned i = 0; i < new_stride; i++) {
         if (i == vec_offset)
            data[i] = intrin->src[1].ssa;
         else
            data[i] = undef;
      }

      nir_src_rewrite(&intrin->src[1], nir_vec(b, data, new_stride));
      nir_intrinsic_set_write_mask(intrin, nir_intrinsic_write_mask(intrin) << vec_offset);
      break;
   }
   case nir_intrinsic_deref_atomic:
   case nir_intrinsic_deref_atomic_swap: {
      b->cursor = nir_before_instr(&intrin->instr);

      nir_deref_instr *new_deref =
         nir_build_deref_array_imm(b, nir_src_as_deref(intrin->src[0]), vec_offset);

      nir_src_rewrite(&intrin->src[0], &new_deref->def);
      break;
   }
   default:
      UNREACHABLE("unexpected use intrinsic");
   }
}

static void
update_deref_array(nir_builder *b, nir_deref_instr *deref,
                   unsigned new_stride, struct hash_table *range_ht)
{
   assert(deref->deref_type == nir_deref_type_array);
   deref->type = glsl_vector_type(glsl_get_base_type(deref->type), new_stride);

   struct stride_desc stride_desc = get_stride_desc(nir_scalar_resolved(deref->arr.index.ssa, 0));
   assert(stride_desc.stride % new_stride == 0);

   b->cursor = nir_before_instr(&deref->instr);

   nir_def *new_idx = NULL;
   if (stride_desc.stride > 1) {
      if (mul_is_nuw(b->shader, &stride_desc, range_ht)) {
         unsigned remaining = stride_desc.stride / new_stride;
         new_idx = nir_imul_imm(b, nir_mov_scalar(b, stride_desc.mul_src), remaining);
      } else {
         new_idx = nir_udiv_imm(b, nir_mov_scalar(b, stride_desc.mul), new_stride);
      }

      new_idx = nir_iadd_imm(b, new_idx, stride_desc.offset / new_stride);
   } else {
      assert(stride_desc.stride == 0);
      new_idx = nir_imm_int(b, stride_desc.offset / new_stride);
   }

   nir_src_rewrite(&deref->arr.index, new_idx);

   unsigned vec_offset = stride_desc.offset % new_stride;
   nir_foreach_use_including_if_safe(use, &deref->def) {
      nir_instr *use_instr = nir_src_use_instr(use);
      if (use_instr->type == nir_instr_type_deref)
         continue; /* Updating atomics inserts new deref uses. */
      update_intrin(b, nir_instr_as_intrinsic(use_instr), new_stride, vec_offset);
   }
}

static void
update_derefs_to_vec(nir_function_impl *impl, struct hash_table *var_stride_map,
                     nir_variable_mode modes, struct hash_table *range_ht)
{
   nir_builder b = nir_builder_create(impl);

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_deref)
            continue;
         nir_deref_instr *deref = nir_instr_as_deref(instr);
         if (deref->deref_type != nir_deref_type_var ||
             !(deref->modes & modes))
            continue;

         struct array_info *info = get_array_info(var_stride_map, deref->var);
         if (!info)
            continue;

         deref->type = deref->var->type;

         nir_foreach_use_including_if(use, &deref->def) {
            update_deref_array(&b, nir_instr_as_deref(nir_src_use_instr(use)),
                               info->new_stride, range_ht);
         }
      }
   }
}

bool
nir_opt_scalar_array_vars_to_vec(nir_shader *shader, nir_variable_mode modes)
{
   assert((modes & (nir_var_shader_temp | nir_var_function_temp | nir_var_mem_shared)) == modes);

   if (shader->info.shared_memory_explicit_layout)
      modes &= ~nir_var_mem_shared;

   const nir_variable_mode global_modes = modes & ~nir_var_function_temp;

   void *mem_ctx = ralloc_context(NULL);

   struct hash_table var_stride_map;
   _mesa_pointer_hash_table_init(&var_stride_map, mem_ctx);

   struct hash_table range_ht;
   _mesa_pointer_hash_table_init(&range_ht, mem_ctx);

   bool has_global_array = init_array_vars_in_list(&shader->variables,
                                                   global_modes,
                                                   &var_stride_map,
                                                   mem_ctx);

   nir_foreach_function_impl(impl, shader) {
      bool has_local_array = init_array_vars_in_list(&impl->locals,
                                                     modes & nir_var_function_temp,
                                                     &var_stride_map,
                                                     mem_ctx);

      if (has_global_array || has_local_array) {
         _mesa_hash_table_clear(&range_ht, NULL);
         find_array_stride(impl, &var_stride_map, modes, &range_ht);
      }
   }

   if (_mesa_hash_table_num_entries(&var_stride_map) == 0) {
      ralloc_free(mem_ctx);
      nir_shader_preserve_all_metadata(shader);
      return false;
   }

   bool global_to_vec = has_global_array && array_to_vec_list(shader,
                                                              &shader->variables,
                                                              global_modes,
                                                              &var_stride_map);

   bool progress = false;
   nir_foreach_function_impl(impl, shader) {
      bool locals_to_vec = array_to_vec_list(shader,
                                             &impl->locals,
                                             nir_var_function_temp,
                                             &var_stride_map);

      if (global_to_vec || locals_to_vec) {
         _mesa_hash_table_clear(&range_ht, NULL);
         update_derefs_to_vec(impl, &var_stride_map, modes, &range_ht);

         progress = nir_progress(true, impl, nir_metadata_control_flow);
      } else {
         nir_no_progress(impl);
      }
   }

   ralloc_free(mem_ctx);

   return progress;
}
