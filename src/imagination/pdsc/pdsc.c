/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pdsc.c
 *
 * \brief Main PDS compiler implementation.
 */

#include "pdsc.h"
#include "pdsc_internal.h"

#include "util/bitset.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

void pdsc_init(void)
{
   pdsc_debug_init();
}

pdsc_program *
pdsc_program_create(void *mem_ctx, enum pdsc_feature features, const char *name)
{
   pdsc_program *p = rzalloc_size(mem_ctx, sizeof(*p));

   p->name = ralloc_strdup(p, name);
   p->features = features;

   util_dynarray_init(&p->saved_refs, p);
   util_dynarray_init(&p->const_patch_ids, p);
   util_dynarray_init(&p->const_types, p);
   util_dynarray_init(&p->const_names, p);
   util_dynarray_init(&p->temp_names, p);
   util_dynarray_init(&p->ptemp_names, p);
   util_dynarray_init(&p->instrs, p);
   util_dynarray_init(&p->const_data, p);
   util_dynarray_init(&p->instr_data, p);

   return p;
}

uint8_t pdsc_program_temps_used(const pdsc_program *p)
{
   return BITSET_LAST_BIT(p->temps_used);
}

#define PDSC_ALLOC_FAIL UINT8_MAX
static uint8_t pdsc_alloc_contig_regs(uint8_t contig,
                                      bool align64,
                                      BITSET_WORD *regset,
                                      uint8_t bitset_bits)
{
   assert(contig > 0 && contig <= bitset_bits);

   BITSET_DECLARE(regset_not, bitset_bits);
   BITSET_COPY(regset_not, regset);
   BITSET_NOT(regset_not);

   BITSET_DECLARE(scan, bitset_bits);
   BITSET_COPY(scan, regset_not);

   do {
      for (uint8_t u = 1; u < contig; ++u) {
         BITSET_DECLARE(regset_not_shr, bitset_bits);
         BITSET_COPY(regset_not_shr, regset_not);
         BITSET_SHR(regset_not_shr, u);

         __bitset_and(scan, scan, regset_not_shr, ARRAY_SIZE(scan));
      }

      if (!BITSET_IS_EMPTY(scan)) {
         uint8_t bit = BITSET_FFS(scan) - 1;

         if (!align64 || !(bit & 1))
            return bit;

         BITSET_CLEAR(regset_not, bit);
         BITSET_COPY(scan, regset_not);
      }
   } while (!BITSET_IS_EMPTY(scan));

   return PDSC_ALLOC_FAIL;
}

pdsc_ref pdsc_alloc_const32(pdsc_program *p,
                            const char *name,
                            enum pdsc_type type,
                            uint8_t patch_id)
{
   uint8_t index =
      pdsc_alloc_contig_regs(1, false, p->consts_used, PDSC_MAX_CONSTS);

   if (index == PDSC_ALLOC_FAIL)
      UNREACHABLE("Constants exhausted.");

   assert(!BITSET_TEST(p->consts_used, index));
   BITSET_SET(p->consts_used, index);

   ASSERTED uint8_t alias = pdsc_add_const_name(p, index, name);
   assert(!alias);
   pdsc_set_const_type(p, index, type);
   pdsc_set_const_patch_id(p, index, patch_id, false);

   if (util_dynarray_num_elements(&p->const_data, uint32_t) < index + 1)
      (void)!util_dynarray_resize_zero(&p->const_data, uint32_t, index + 1);

   return pdsc_ref_const32(index);
}

pdsc_ref pdsc_find_or_alloc_const32(pdsc_program *p,
                                    const char *name,
                                    enum pdsc_type type,
                                    uint8_t patch_id)
{
   uint8_t index = pdsc_get_const_index(p, patch_id);

   if (index != PDSC_CONST_INDEX_NOT_FOUND)
      return pdsc_ref_const32(index);

   return pdsc_alloc_const32(p, name, type, patch_id);
}

void pdsc_assign_const32(pdsc_program *p, pdsc_ref ref, uint32_t value)
{
   assert(pdsc_ref_is_const32(ref));
   uint16_t index = pdsc_ref_get_val(ref);

   assert(!BITSET_TEST(p->consts_assigned, index));
   BITSET_SET(p->consts_assigned, index);

   if (util_dynarray_num_elements(&p->const_data, uint32_t) < index + 1)
      (void)!util_dynarray_resize_zero(&p->const_data, uint32_t, index + 1);

   pdsc_set_assigned_const(p, index, false, value);
}

pdsc_ref pdsc_alloc_const64(pdsc_program *p,
                            const char *name,
                            enum pdsc_type type,
                            uint8_t patch_id)
{
   uint8_t index =
      pdsc_alloc_contig_regs(2, true, p->consts_used, PDSC_MAX_CONSTS);

   if (index == PDSC_ALLOC_FAIL)
      UNREACHABLE("Constants exhausted.");

   assert(!BITSET_TEST(p->consts_used, index));
   assert(!BITSET_TEST(p->consts_used, index + 1));

   BITSET_SET(p->consts_used, index);
   BITSET_SET(p->consts_used, index + 1);

   assert(!BITSET_TEST(p->consts_64, index));
   BITSET_SET(p->consts_64, index);

   ASSERTED uint8_t alias = pdsc_add_const_name(p, index, name);
   assert(!alias);
   pdsc_set_const_type(p, index, type);
   pdsc_set_const_patch_id(p, index, patch_id, true);

   if (util_dynarray_num_elements(&p->const_data, uint32_t) < index + 2)
      (void)!util_dynarray_resize_zero(&p->const_data, uint32_t, index + 2);

   return pdsc_ref_const64(index);
}

pdsc_ref pdsc_find_or_alloc_const64(pdsc_program *p,
                                    const char *name,
                                    enum pdsc_type type,
                                    uint8_t patch_id)
{
   uint8_t index = pdsc_get_const_index(p, patch_id);

   if (index != PDSC_CONST_INDEX_NOT_FOUND)
      return pdsc_ref_const64(index);

   return pdsc_alloc_const64(p, name, type, patch_id);
}

void pdsc_assign_const64(pdsc_program *p, pdsc_ref ref, uint64_t value)
{
   assert(pdsc_ref_is_const64(ref));
   uint16_t index = pdsc_ref_get_val(ref);

   assert(!BITSET_TEST(p->consts_assigned, index));
   BITSET_SET(p->consts_assigned, index);

   assert(!BITSET_TEST(p->consts_assigned, index + 1));
   BITSET_SET(p->consts_assigned, index + 1);

   if (util_dynarray_num_elements(&p->const_data, uint32_t) < index + 2)
      (void)!util_dynarray_resize_zero(&p->const_data, uint32_t, index + 2);

   pdsc_set_assigned_const(p, index, true, value);
}

pdsc_ref pdsc_decl_temp32(pdsc_program *p,
                          const char *name,
                          uint8_t index,
                          ASSERTED bool aliased)
{
   assert(!BITSET_TEST(p->temps_used, index) || aliased);
   BITSET_SET(p->temps_used, index);

   uint8_t alias = pdsc_add_temp_name(p, index, name);
   assert(!alias || (aliased && alias <= 0xf));
   pdsc_ref ref = pdsc_ref_temp32(index);
   return pdsc_ref_set_alias(ref, alias);
}

pdsc_ref pdsc_decl_temp64(pdsc_program *p,
                          const char *name,
                          uint8_t index,
                          ASSERTED bool aliased)
{
   assert(!BITSET_TEST(p->temps_used, index) || aliased);
   assert(!BITSET_TEST(p->temps_used, index + 1) || aliased);
   BITSET_SET(p->temps_used, index);
   BITSET_SET(p->temps_used, index + 1);

   assert(!BITSET_TEST(p->temps_64, index) || aliased);
   BITSET_SET(p->temps_64, index);

   uint8_t alias = pdsc_add_temp_name(p, index, name);
   assert(!alias || (aliased && alias <= 0xf));
   pdsc_ref ref = pdsc_ref_temp64(index);
   return pdsc_ref_set_alias(ref, alias);
}

static inline pdsc_ref pdsc_decl_temp32_array(pdsc_program *p,
                                              const char *name,
                                              uint8_t index,
                                              uint8_t elems)
{
   for (unsigned u = 0; u < elems; ++u) {
      assert(!BITSET_TEST(p->temps_used, index + u));
      BITSET_SET(p->temps_used, index + u);

      if (u > 0) {
         assert(!BITSET_TEST(p->temps_array_elem, index + u));
         BITSET_SET(p->temps_array_elem, index + u);
      }
   }

   uint8_t alias = pdsc_add_temp_name(p, index, name);
   assert(!alias);
   pdsc_ref ref = pdsc_ref_temp32_array(index, elems);
   return ref;
}

static inline pdsc_ref pdsc_decl_temp64_array(pdsc_program *p,
                                              const char *name,
                                              uint8_t index,
                                              uint8_t elems)
{
   for (unsigned u = 0; u < elems; ++u) {
      assert(!BITSET_TEST(p->temps_used, index + 2 * u));
      assert(!BITSET_TEST(p->temps_used, index + 2 * u + 1));
      BITSET_SET(p->temps_used, index + 2 * u);
      BITSET_SET(p->temps_used, index + 2 * u + 1);

      assert(!BITSET_TEST(p->temps_64, index + 2 * u));
      BITSET_SET(p->temps_64, index + 2 * u);

      if (u > 0) {
         assert(!BITSET_TEST(p->temps_array_elem, index + 2 * u));
         BITSET_SET(p->temps_array_elem, index + 2 * u);
      }

      if (elems > 1) {
         assert(!BITSET_TEST(p->temps_array_elem, index + 2 * u + 1));
         BITSET_SET(p->temps_array_elem, index + 2 * u + 1);
      }
   }

   uint8_t alias = pdsc_add_temp_name(p, index, name);
   assert(!alias);
   pdsc_ref ref = pdsc_ref_temp64_array(index, elems);
   return ref;
}

pdsc_ref
pdsc_alloc_temp32_array(pdsc_program *p, const char *name, uint8_t elems)
{
   uint8_t index =
      pdsc_alloc_contig_regs(elems, false, p->temps_used, PDSC_MAX_TEMPS);

   if (index == PDSC_ALLOC_FAIL)
      UNREACHABLE("Temps exhausted.");

   return pdsc_decl_temp32_array(p, name, index, elems);
}

pdsc_ref
pdsc_alloc_temp64_array(pdsc_program *p, const char *name, uint8_t elems)
{
   uint8_t index =
      pdsc_alloc_contig_regs(elems * 2, true, p->temps_used, PDSC_MAX_TEMPS);

   if (index == PDSC_ALLOC_FAIL)
      UNREACHABLE("Temps exhausted.");

   return pdsc_decl_temp64_array(p, name, index, elems);
}

pdsc_ref pdsc_decl_ptemp32(pdsc_program *p,
                           const char *name,
                           uint8_t index,
                           ASSERTED bool aliased)
{
   assert(!BITSET_TEST(p->ptemps_used, index) || aliased);
   BITSET_SET(p->ptemps_used, index);

   uint8_t alias = pdsc_add_ptemp_name(p, index, name);
   assert(!alias || (aliased && alias <= 0xf));
   pdsc_ref ref = pdsc_ref_ptemp32(index);
   return pdsc_ref_set_alias(ref, alias);
}

pdsc_ref pdsc_decl_ptemp64(pdsc_program *p,
                           const char *name,
                           uint8_t index,
                           ASSERTED bool aliased)
{
   assert(!BITSET_TEST(p->ptemps_used, index) || aliased);
   assert(!BITSET_TEST(p->ptemps_used, index + 1) || aliased);
   BITSET_SET(p->ptemps_used, index);
   BITSET_SET(p->ptemps_used, index + 1);

   assert(!BITSET_TEST(p->ptemps_64, index) || aliased);
   BITSET_SET(p->ptemps_64, index);

   uint8_t alias = pdsc_add_ptemp_name(p, index, name);
   assert(!alias || (aliased && alias <= 0xf));
   pdsc_ref ref = pdsc_ref_ptemp64(index);
   return pdsc_ref_set_alias(ref, alias);
}

void pdsc_patch_program(pdsc_program *p,
                        unsigned num_patches,
                        const pdsc_patch patches[num_patches],
                        uint32_t *const_buffer)
{
   for (unsigned u = 0; u < num_patches; ++u) {
      const pdsc_patch *patch = &patches[u];
      assert(patch->patch_id != PDSC_CONST_PATCH_ID_NONE);

      uint8_t const_index = pdsc_get_const_index(p, patch->patch_id);
      assert(const_index != PDSC_CONST_INDEX_NOT_FOUND);

      bool is_64bit = pdsc_const_patch_id_is_64bit(p, patch->patch_id);
      BITSET_SET(p->consts_assigned, const_index);
      if (is_64bit)
         BITSET_SET(p->consts_assigned, const_index + 1);

      if (!const_buffer) {
         pdsc_set_assigned_const(p, const_index, is_64bit, patch->value);
      } else {
         const_buffer[const_index] = patch->value & UINT32_MAX;
         if (is_64bit)
            const_buffer[const_index + 1] = patch->value >> 32;
      }
   }

   if (PDSC_DEBUG(PRINT)) {
      fputs("ir after pdsc_patch_program:\n", stdout);
      pdsc_print_program(stdout, p, const_buffer);
   }
}

void pdsc_save_ref(pdsc_program *p, uint8_t ref_id, pdsc_ref ref)
{
   if (ref_id == PDSC_REF_ID_NONE)
      return;

   if (util_dynarray_num_elements(&p->saved_refs, pdsc_ref) < ref_id + 1)
      (void)!util_dynarray_resize_zero(&p->saved_refs, pdsc_ref, ref_id + 1);

   *util_dynarray_element(&p->saved_refs, pdsc_ref, ref_id) = ref;
}

pdsc_ref pdsc_load_ref(const pdsc_program *p, uint8_t ref_id)
{
   if (ref_id == PDSC_REF_ID_NONE)
      return pdsc_ref_null();

   assert(ref_id < util_dynarray_num_elements(&p->saved_refs, pdsc_ref));
   return *util_dynarray_element(&p->saved_refs, pdsc_ref, ref_id);
}
