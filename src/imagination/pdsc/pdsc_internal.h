/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef PDSC_INTERNAL_H
#define PDSC_INTERNAL_H

/**
 * \file pdsc_internal.h
 *
 * \brief PDS compiler internal header.
 */

#include "pdsc.h"

#include "util/bitset.h"
#include "util/macros.h"
#include "util/u_dynarray.h"

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#define PDSC_MAX_SRCS 4
#define PDSC_MAX_MODS 5

#define PDSC_MAX_CONSTS 128
#define PDSC_MAX_TEMPS 32
#define PDSC_MAX_PTEMPS 32
#define PDSC_MAX_GLOBALS 2

typedef struct PACKED _pdsc_ref {
   union PACKED {
      uint16_t val;
      int16_t sval;
   };

   uint8_t chans : 4;
   uint8_t alias : 4;

   enum PACKED pdsc_ref_type {
      PDSC_REF_TYPE_NULL,

      PDSC_REF_TYPE_IMM,
      PDSC_REF_TYPE_SIMM,

      PDSC_REF_TYPE_CONST_REG,
      PDSC_REF_TYPE_TEMP_REG,
      PDSC_REF_TYPE_PTEMP_REG,
      PDSC_REF_TYPE_GLOBAL_REG,

      _PDSC_REF_TYPE_COUNT,
   } type : 3;

   bool is_64bit : 1;
   bool is_elem32 : 1;
   bool is_composite64 : 1;
   bool is_array_elem : 1;

   uint8_t _pad : 1;
} pdsc_ref;
static_assert(sizeof(pdsc_ref) == sizeof(uint32_t),
              "sizeof(pdsc_ref) != sizeof(uint32_t)");

static inline pdsc_ref pdsc_ref_null(void)
{
   return (pdsc_ref){
      .type = PDSC_REF_TYPE_NULL,
   };
}

static inline pdsc_ref pdsc_ref_imm(uint16_t val)
{
   return (pdsc_ref){
      .val = val,
      .type = PDSC_REF_TYPE_IMM,
   };
}

static inline pdsc_ref pdsc_ref_simm(int16_t sval)
{
   return (pdsc_ref){
      .sval = sval,
      .type = PDSC_REF_TYPE_SIMM,
   };
}

static inline pdsc_ref pdsc_ref_const32(uint16_t index)
{
   return (pdsc_ref){
      .val = index,
      .is_64bit = false,
      .type = PDSC_REF_TYPE_CONST_REG,
   };
}

static inline pdsc_ref pdsc_ref_const64(uint16_t index)
{
   assert(!(index & 0b1));

   return (pdsc_ref){
      .val = index,
      .type = PDSC_REF_TYPE_CONST_REG,
      .is_64bit = true,
   };
}

static inline pdsc_ref pdsc_ref_temp32(uint16_t index)
{
   return (pdsc_ref){
      .val = index,
      .type = PDSC_REF_TYPE_TEMP_REG,
      .is_64bit = false,
   };
}

static inline pdsc_ref pdsc_ref_temp64(uint16_t index)
{
   assert(!(index & 0b1));

   return (pdsc_ref){
      .val = index,
      .type = PDSC_REF_TYPE_TEMP_REG,
      .is_64bit = true,
   };
}

static inline pdsc_ref pdsc_ref_temp32_array(uint16_t index, uint8_t chans)
{
   return (pdsc_ref){
      .val = index,
      .type = PDSC_REF_TYPE_TEMP_REG,
      .is_64bit = false,
      .chans = chans - 1,
   };
}

static inline pdsc_ref pdsc_ref_temp64_array(uint16_t index, uint8_t chans)
{
   assert(!(index & 0b1));

   return (pdsc_ref){
      .val = index,
      .type = PDSC_REF_TYPE_TEMP_REG,
      .is_64bit = true,
      .chans = chans - 1,
   };
}

static inline pdsc_ref pdsc_ref_ptemp32(uint16_t index)
{
   return (pdsc_ref){
      .val = index,
      .type = PDSC_REF_TYPE_PTEMP_REG,
      .is_64bit = false,
   };
}

static inline pdsc_ref pdsc_ref_ptemp64(uint16_t index)
{
   assert(!(index & 0b1));

   return (pdsc_ref){
      .val = index,
      .type = PDSC_REF_TYPE_PTEMP_REG,
      .is_64bit = true,
   };
}

static inline pdsc_ref pdsc_ref_global(uint16_t index)
{
   return (pdsc_ref){
      .val = index,
      .type = PDSC_REF_TYPE_GLOBAL_REG,
   };
}

static inline pdsc_ref pdsc_ref_lo32(pdsc_ref ref)
{
   assert(ref.is_64bit);
   assert(!ref.is_composite64);
   ref.is_64bit = false;
   ref.is_elem32 = true;

   return ref;
}

static inline pdsc_ref pdsc_ref_hi32(pdsc_ref ref)
{
   assert(ref.is_64bit);
   assert(!ref.is_composite64);
   ref.is_64bit = false;
   ref.is_elem32 = true;
   ++ref.val;

   return ref;
}

static inline pdsc_ref pdsc_ref_64_2x32(pdsc_ref ref_lo, pdsc_ref ref_hi)
{
   assert(!ref_lo.is_64bit && !ref_hi.is_64bit);
   assert(ref_lo.type == ref_hi.type);
   assert(!ref_lo.chans && !ref_hi.chans);
   assert(!(ref_lo.val & 0b1));
   assert(ref_lo.val + 1 == ref_hi.val);

   ref_lo.is_64bit = true;
   ref_lo.is_composite64 = true;
   return ref_lo;
}

static inline pdsc_ref pdsc_ref_elem64(pdsc_ref ref, unsigned elem)
{
   assert(ref.is_64bit);
   assert(!ref.is_composite64);
   assert(!ref.is_array_elem);
   assert(elem <= ref.chans);

   ref.is_array_elem = true;
   ref.chans = elem * 2;
   ref.val += elem * 2;

   return ref;
}

static inline pdsc_ref pdsc_ref_elem32(pdsc_ref ref, unsigned elem)
{
   assert(!ref.is_64bit);
   assert(!ref.is_array_elem);
   assert(elem <= ref.chans);

   ref.is_array_elem = true;
   ref.chans = elem;
   ref.val += elem;

   return ref;
}

static inline bool pdsc_ref_is_null(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_NULL;
}

static inline bool pdsc_ref_is_imm(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_IMM;
}

static inline bool pdsc_ref_is_simm(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_SIMM;
}

static inline bool pdsc_ref_is_const32(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_CONST_REG && !ref.is_64bit;
}

static inline bool pdsc_ref_is_const64(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_CONST_REG && ref.is_64bit;
}

static inline bool pdsc_ref_is_temp32(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_TEMP_REG && !ref.is_64bit;
}

static inline bool pdsc_ref_is_temp64(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_TEMP_REG && ref.is_64bit;
}

static inline bool pdsc_ref_is_ptemp32(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_PTEMP_REG && !ref.is_64bit;
}

static inline bool pdsc_ref_is_ptemp64(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_PTEMP_REG && ref.is_64bit;
}

static inline bool pdsc_ref_is_global(pdsc_ref ref)
{
   return ref.type == PDSC_REF_TYPE_GLOBAL_REG;
}

static inline bool pdsc_ref_is_scalar(pdsc_ref ref)
{
   return !ref.chans;
}

static inline bool pdsc_ref_is_elem32(pdsc_ref ref)
{
   return ref.is_elem32;
}

static inline bool pdsc_ref_is_composite64(pdsc_ref ref)
{
   return ref.is_composite64;
}

static inline bool pdsc_ref_is_64bit(pdsc_ref ref)
{
   return ref.is_64bit;
}

static inline enum pdsc_ref_type pdsc_ref_get_type(pdsc_ref ref)
{
   return ref.type;
}

static inline uint8_t pdsc_ref_get_chans(pdsc_ref ref)
{
   return ref.chans + 1;
}

static inline uint8_t pdsc_ref_get_alias(pdsc_ref ref)
{
   return ref.alias;
}

static inline uint16_t pdsc_ref_get_val(pdsc_ref ref)
{
   return ref.val;
}

static inline int16_t pdsc_ref_get_sval(pdsc_ref ref)
{
   return ref.sval;
}

static inline pdsc_ref pdsc_ref_set_chans(pdsc_ref ref, uint8_t chans)
{
   ref.chans = chans - 1;

   return ref;
}

static inline pdsc_ref pdsc_ref_set_val(pdsc_ref ref, uint16_t val)
{
   ref.val = val;
   return ref;
}

static inline pdsc_ref pdsc_ref_set_sval(pdsc_ref ref, int16_t sval)
{
   ref.sval = sval;
   return ref;
}

static inline pdsc_ref pdsc_ref_set_alias(pdsc_ref ref, uint8_t alias)
{
   ref.alias = alias;
   return ref;
}

static inline pdsc_ref pdsc_ref_offset_val(pdsc_ref ref, int16_t offset)
{
   int32_t val = pdsc_ref_get_val(ref);
   val += offset;
   assert(val >= 0 && val < UINT16_MAX);

   return pdsc_ref_set_val(ref, val);
}

static inline pdsc_ref pdsc_ref_offset_sval(pdsc_ref ref, int16_t offset)
{
   return pdsc_ref_set_sval(ref, pdsc_ref_get_sval(ref) + offset);
}

enum ENUM_PACKED pdsc_type {
   PDSC_TYPE_IMM,
   PDSC_TYPE_STM_SRC0,
   PDSC_TYPE_STM_SRC1,
   PDSC_TYPE_STM_SRC2,
   PDSC_TYPE_STM_SRC3,
   PDSC_TYPE_LD64_SRC0,
   PDSC_TYPE_ST32_SRC0,
   PDSC_TYPE_STMP_SRC0,
   PDSC_TYPE_STMP_SRC1,
   PDSC_TYPE_STMP_SRC2,
   PDSC_TYPE_AA_SRC0,
   PDSC_TYPE_IDF_SRC0,
   PDSC_TYPE_POL_SRC0,
   PDSC_TYPE_DDMAD_SRC3,
   PDSC_TYPE_DOUTU_SRC0,
   PDSC_TYPE_DOUTD_SRC0,
   PDSC_TYPE_DOUTD_SRC1,
   PDSC_TYPE_DOUTW_SRC1,
   PDSC_TYPE_DOUTI_SRC0,

   _PDSC_TYPE_COUNT,
};
static_assert(sizeof(enum pdsc_type) == sizeof(uint8_t),
              "sizeof(enum pdsc_type) != sizeof(uint8_t)");

typedef struct _pdsc_program pdsc_program;

pdsc_ref pdsc_alloc_const32(pdsc_program *p,
                            const char *name,
                            enum pdsc_type type,
                            uint8_t patch_id);
pdsc_ref pdsc_find_or_alloc_const32(pdsc_program *p,
                                    const char *name,
                                    enum pdsc_type type,
                                    uint8_t patch_id);
void pdsc_assign_const32(pdsc_program *p, pdsc_ref ref, uint32_t value);
pdsc_ref pdsc_alloc_const64(pdsc_program *p,
                            const char *name,
                            enum pdsc_type type,
                            uint8_t patch_id);
pdsc_ref pdsc_find_or_alloc_const64(pdsc_program *p,
                                    const char *name,
                                    enum pdsc_type type,
                                    uint8_t patch_id);
void pdsc_assign_const64(pdsc_program *p, pdsc_ref ref, uint64_t value);

pdsc_ref pdsc_decl_temp32(pdsc_program *p,
                          const char *name,
                          uint8_t index,
                          bool aliased);
pdsc_ref pdsc_decl_temp64(pdsc_program *p,
                          const char *name,
                          uint8_t index,
                          bool aliased);

pdsc_ref
pdsc_alloc_temp32_array(pdsc_program *p, const char *name, uint8_t elems);
pdsc_ref
pdsc_alloc_temp64_array(pdsc_program *p, const char *name, uint8_t elems);

pdsc_ref pdsc_decl_ptemp32(pdsc_program *p,
                           const char *name,
                           uint8_t index,
                           bool aliased);
pdsc_ref pdsc_decl_ptemp64(pdsc_program *p,
                           const char *name,
                           uint8_t index,
                           bool aliased);

#define ALLOC_CONST32(p, name, patch_id) \
   pdsc_ref name = pdsc_alloc_const32((p), #name, PDSC_TYPE_IMM, (patch_id))

#define ALLOC_VAL_CONST32(p, name, value)                                     \
   pdsc_ref name = pdsc_alloc_const32((p), #name, PDSC_TYPE_IMM, (patch_id)); \
   pdsc_assign_const32((p), (name), (value))

#define ALLOC_CONST64(p, name, patch_id) \
   pdsc_ref name = pdsc_alloc_const64((p), #name, PDSC_TYPE_IMM, (patch_id))

#define ALLOC_VAL_CONST64(p, name, value)                                    \
   pdsc_ref name =                                                           \
      pdsc_alloc_const64(p, #name, PDSC_TYPE_IMM, PDSC_CONST_PATCH_ID_NONE); \
   pdsc_assign_const64((p), (name), (value))

#define ALLOC_TEMP32(p, name) \
   pdsc_ref name = pdsc_alloc_temp32_array((p), #name, 1)

#define ALLOC_TEMP64(p, name) \
   pdsc_ref name = pdsc_alloc_temp64_array((p), #name, 1)

#define ALLOC_TEMP32_ARRAY(p, name, elems) \
   pdsc_ref name = pdsc_alloc_temp32_array((p), #name, (elems))

#define ALLOC_TEMP64_ARRAY(p, name, elems) \
   pdsc_ref name = pdsc_alloc_temp64_array((p), #name, (elems))

#define DECL_TEMP32(p, name, index) \
   pdsc_ref name = pdsc_decl_temp32((p), #name, (index), false)

#define DECL_TEMP64(p, name, index) \
   pdsc_ref name = pdsc_decl_temp64((p), #name, (index), false)

#define DECL_PTEMP32(p, name, index) \
   pdsc_ref name = pdsc_decl_ptemp32((p), #name, (index), false)

#define DECL_PTEMP64(p, name, index) \
   pdsc_ref name = pdsc_decl_ptemp64((p), #name, (index), false)

#define DECL_TEMP32_ALIAS(p, name, index) \
   pdsc_ref name = pdsc_decl_temp32((p), #name, (index), true)

#define DECL_TEMP64_ALIAS(p, name, index) \
   pdsc_ref name = pdsc_decl_temp64((p), #name, (index), true)

#define DECL_PTEMP32_ALIAS(p, name, index) \
   pdsc_ref name = pdsc_decl_ptemp32((p), #name, (index), true)

#define DECL_PTEMP64_ALIAS(p, name, index) \
   pdsc_ref name = pdsc_decl_ptemp64((p), #name, (index), true)

typedef uint16_t pdsc_instr_t;
#define PDSC_TARGET_INSTR_NONE UINT16_MAX
typedef struct _pdsc_instr {
   uint8_t op;

   union {
      struct {
         pdsc_ref dst;
         pdsc_ref src[PDSC_MAX_SRCS];
         uint8_t num_srcs;
      };

      pdsc_instr_t target_instr;
   };

   uint8_t mod[PDSC_MAX_MODS];
} pdsc_instr;

typedef struct _pdsc_program {
   const char *name;
   enum pdsc_feature features;

   struct util_dynarray saved_refs;
   struct util_dynarray const_patch_ids;

   struct util_dynarray const_types;
   struct util_dynarray const_names;
   BITSET_DECLARE(consts_64, PDSC_MAX_CONSTS);
   BITSET_DECLARE(consts_used, PDSC_MAX_CONSTS);
   BITSET_DECLARE(consts_assigned, PDSC_MAX_CONSTS);

   struct util_dynarray temp_names;
   BITSET_DECLARE(temps_64, PDSC_MAX_TEMPS);
   BITSET_DECLARE(temps_array_elem, PDSC_MAX_TEMPS);
   BITSET_DECLARE(temps_used, PDSC_MAX_TEMPS);

   struct util_dynarray ptemp_names;
   BITSET_DECLARE(ptemps_64, PDSC_MAX_PTEMPS);
   BITSET_DECLARE(ptemps_used, PDSC_MAX_PTEMPS);

   uint16_t num_doutds;
   uint16_t num_doutws;
   uint16_t num_doutus;
   uint16_t num_doutvs;
   uint16_t num_doutis;
   uint16_t num_doutcs;
   uint16_t num_lds;
   uint16_t num_sts;

   struct util_dynarray instrs;

   /* Encoded data. */
   struct util_dynarray const_data;
   struct util_dynarray instr_data;
} pdsc_program;

#define PDSC_HAS(p, feature) unlikely((p)->features &(PDSC_FEATURE_##feature))

#define PDSC_CONST_PATCH_ID_NONE UINT8_MAX
#define PDSC_CONST_PATCH_ID_64BIT UINT8_MAX
static inline void pdsc_set_const_patch_id(pdsc_program *p,
                                           uint8_t index,
                                           uint8_t patch_id,
                                           bool is_64bit)
{
   assert(index < PDSC_MAX_CONSTS);
   assert(!(index & 1) || !is_64bit);
   if (index >= util_dynarray_num_elements(&p->const_patch_ids, uint8_t)) {
      (void)!util_dynarray_resize_zero(&p->const_patch_ids,
                                       uint8_t,
                                       index + (is_64bit ? 2 : 1));
   }
   assert(index < util_dynarray_num_elements(&p->const_patch_ids, uint8_t));

   if (patch_id == PDSC_CONST_PATCH_ID_NONE)
      return;

   assert(patch_id < PDSC_MAX_CONSTS);
   *util_dynarray_element(&p->const_patch_ids, uint8_t, index) = patch_id + 1;
   if (is_64bit)
      *util_dynarray_element(&p->const_patch_ids, uint8_t, index + 1) =
         PDSC_CONST_PATCH_ID_64BIT;
}

static inline uint8_t pdsc_get_const_patch_id(const pdsc_program *p,
                                              uint8_t index)
{
   assert(index < util_dynarray_num_elements(&p->const_patch_ids, uint8_t));
   uint8_t patch_id =
      *util_dynarray_element(&p->const_patch_ids, uint8_t, index);
   if (!patch_id || patch_id == PDSC_CONST_PATCH_ID_64BIT)
      return PDSC_CONST_PATCH_ID_NONE;

   assert((patch_id - 1) < PDSC_MAX_CONSTS);
   return patch_id - 1;
}

#define PDSC_CONST_INDEX_NOT_FOUND PDSC_MAX_CONSTS
static inline uint8_t pdsc_get_const_index(const pdsc_program *p,
                                           uint8_t patch_id)
{
   assert(patch_id < PDSC_MAX_CONSTS);
   util_dynarray_foreach (&p->const_patch_ids, uint8_t, elem) {
      if (*elem != (patch_id + 1))
         continue;

      uint8_t index =
         elem - util_dynarray_element(&p->const_patch_ids, uint8_t, 0);
      return index;
   }

   return PDSC_CONST_INDEX_NOT_FOUND;
}

static inline bool pdsc_const_patch_id_is_64bit(const pdsc_program *p,
                                                uint8_t patch_id)
{
   uint8_t index = pdsc_get_const_index(p, patch_id);

   if (index & 1)
      return false;

   if ((index + 1) >= util_dynarray_num_elements(&p->const_patch_ids, uint8_t))
      return false;

   return *util_dynarray_element(&p->const_patch_ids, uint8_t, index + 1) ==
          PDSC_CONST_PATCH_ID_64BIT;
}

static inline void
pdsc_set_const_type(pdsc_program *p, uint8_t index, enum pdsc_type type)
{
   assert(index < PDSC_MAX_CONSTS);
   if (util_dynarray_num_elements(&p->const_types, enum pdsc_type) < index + 1)
      (void)!util_dynarray_resize_zero(&p->const_types, enum pdsc_type, index + 1);

   *util_dynarray_element(&p->const_types, enum pdsc_type, index) = type;
}

static inline enum pdsc_type pdsc_get_const_type(const pdsc_program *p,
                                                 uint8_t index)
{
   assert(index < util_dynarray_num_elements(&p->const_types, enum pdsc_type));
   return *util_dynarray_element(&p->const_types, enum pdsc_type, index);
}

static inline uint8_t pdsc_add_name(pdsc_program *p,
                                    struct util_dynarray *buf,
                                    uint8_t index,
                                    const char *name)
{
   if (util_dynarray_num_elements(buf, const char **) < index + 1)
      (void)!util_dynarray_resize_zero(buf, const char **, index + 1);

   const char ***names = util_dynarray_element(buf, const char **, index);

   uint8_t alias = 0;
   if (*names)
      for (const char **aliases = *names; *aliases; ++aliases)
         ++alias;

   *names = reralloc_array_size(p, *names, sizeof(*names), alias + 2);
   (*names)[alias] = ralloc_strdup(p, name);
   (*names)[alias + 1] = NULL;

   return alias;
}

static inline uint8_t
pdsc_add_const_name(pdsc_program *p, uint8_t index, const char *name)
{
   assert(index < PDSC_MAX_CONSTS);
   return pdsc_add_name(p, &p->const_names, index, name);
}

static inline uint8_t
pdsc_add_temp_name(pdsc_program *p, uint8_t index, const char *name)
{
   assert(index < PDSC_MAX_TEMPS);
   return pdsc_add_name(p, &p->temp_names, index, name);
}

static inline uint8_t
pdsc_add_ptemp_name(pdsc_program *p, uint8_t index, const char *name)
{
   assert(index < PDSC_MAX_PTEMPS);
   return pdsc_add_name(p, &p->ptemp_names, index, name);
}

static inline const char **pdsc_get_names(const struct util_dynarray *buf,
                                          uint8_t index)
{
   if (index >= util_dynarray_num_elements(buf, const char **))
      return NULL;

   return *util_dynarray_element(buf, const char **, index);
}

static inline const char *
pdsc_get_name(const struct util_dynarray *buf, uint8_t index, uint8_t alias)
{
   if (index >= util_dynarray_num_elements(buf, const char **))
      return NULL;

   const char **names = pdsc_get_names(buf, index);
   if (!names)
      return NULL;

   uint8_t alias_count = 0;
   for (const char **aliases = names; *aliases; ++aliases)
      ++alias_count;

   if (alias >= alias_count)
      return NULL;

   return names[alias];
}

static inline const char *
pdsc_get_const_name(const pdsc_program *p, uint8_t index, uint8_t alias)
{
   return pdsc_get_name(&p->const_names, index, alias);
}

static inline const char *
pdsc_get_temp_name(const pdsc_program *p, uint8_t index, uint8_t alias)
{
   return pdsc_get_name(&p->temp_names, index, alias);
}

static inline const char *
pdsc_get_ptemp_name(const pdsc_program *p, uint8_t index, uint8_t alias)
{
   return pdsc_get_name(&p->ptemp_names, index, alias);
}

static inline uint64_t
pdsc_get_assigned_const(const pdsc_program *p, uint8_t index, bool is_64bit)
{
   assert(BITSET_TEST(p->consts_assigned, index));
   assert(!is_64bit || BITSET_TEST(p->consts_assigned, index + 1));

   assert(util_dynarray_num_elements(&p->const_data, uint32_t) >
          (index + (is_64bit ? 1 : 0)));

   uint64_t value = *util_dynarray_element(&p->const_data, uint32_t, index);

   if (is_64bit) {
      value |=
         ((uint64_t)*util_dynarray_element(&p->const_data, uint32_t, index + 1)
          << 32);
   }

   return value;
}

static inline void pdsc_set_assigned_const(pdsc_program *p,
                                           uint8_t index,
                                           bool is_64bit,
                                           uint64_t value)
{
   assert(BITSET_TEST(p->consts_assigned, index));
   assert(!is_64bit || BITSET_TEST(p->consts_assigned, index + 1));

   assert(util_dynarray_num_elements(&p->const_data, uint32_t) >
          (index + (is_64bit ? 1 : 0)));

   *util_dynarray_element(&p->const_data, uint32_t, index) = value & UINT32_MAX;

   if (is_64bit)
      *util_dynarray_element(&p->const_data, uint32_t, index + 1) = value >> 32;
}

#define pdsc_foreach_const_index(p, index)                         \
   BITSET_FOREACH_SET ((index), (p)->consts_used, PDSC_MAX_CONSTS) \
      if (!((index) & 1) || !BITSET_TEST((p)->consts_64, (index) - 1))

#define pdsc_foreach_assigned_const_index(p, index)                    \
   BITSET_FOREACH_SET ((index), (p)->consts_assigned, PDSC_MAX_CONSTS) \
      if (!((index) & 1) || !BITSET_TEST((p)->consts_64, (index) - 1))

#define pdsc_foreach_temp_index(p, index)                        \
   BITSET_FOREACH_SET ((index), (p)->temps_used, PDSC_MAX_TEMPS) \
      if (!BITSET_TEST((p)->temps_array_elem, (index)) &&        \
          (!((index) & 1) || !BITSET_TEST((p)->temps_64, (index) - 1)))

#define pdsc_foreach_ptemp_index(p, index)                         \
   BITSET_FOREACH_SET ((index), (p)->ptemps_used, PDSC_MAX_PTEMPS) \
      if (!((index) & 1) || !BITSET_TEST((p)->ptemps_64, (index) - 1))

pdsc_ref pdsc_src_get(const pdsc_program *p, pdsc_instr_t i, uint8_t n);
void pdsc_src_set(pdsc_program *p, pdsc_instr_t i, uint8_t n, pdsc_ref ref);

pdsc_ref pdsc_dst_get(const pdsc_program *p, pdsc_instr_t i);
void pdsc_dst_set(pdsc_program *p, pdsc_instr_t i, pdsc_ref ref);

#define PDSC_REF_ID_NONE UINT8_MAX
void pdsc_save_ref(pdsc_program *p, uint8_t ref_id, pdsc_ref ref);
pdsc_ref pdsc_load_ref(const pdsc_program *p, uint8_t ref_id);

/* Debug. */
enum pdsc_debug {
   PDSC_DEBUG_VAL_SKIP = BITFIELD64_BIT(0),
   PDSC_DEBUG_PRINT = BITFIELD64_BIT(1),
   PDSC_DEBUG_RAW_REGS = BITFIELD64_BIT(2),
   PDSC_DEBUG_PRINT_BINARY = BITFIELD64_BIT(3),
};

extern uint64_t pdsc_debug;

#ifndef NDEBUG
#   define PDSC_DEBUG(flag) unlikely(pdsc_debug &(PDSC_DEBUG_##flag))
#else
#   define PDSC_DEBUG(flag) false
#endif /* NDEBUG */

extern bool pdsc_color;

void pdsc_debug_init(void);

#include "pdsc_ops.h"

#endif /* PDSC_INTERNAL_H */
