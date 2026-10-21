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
#include "pdsc_isa_pack.h"

#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_dynarray.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

unsigned pdsc_program_code_size(const pdsc_program *p)
{
   return util_dynarray_num_elements(&p->instr_data, uint8_t);
}

unsigned pdsc_program_data_size(const pdsc_program *p)
{
   return util_dynarray_num_elements(&p->const_data, uint8_t);
}

const void *pdsc_program_code(const pdsc_program *p)
{
   return util_dynarray_begin(&p->instr_data);
}

const void *pdsc_program_data(const pdsc_program *p)
{
   return util_dynarray_begin(&p->const_data);
}

void *pdsc_program_write_linked(const pdsc_program *p,
                                void *target,
                                unsigned code_offset)
{
   memcpy((uint8_t *)target, pdsc_program_data(p), pdsc_program_data_size(p));
   memcpy((uint8_t *)target + code_offset,
          pdsc_program_code(p),
          pdsc_program_code_size(p));
   return target;
}

unsigned pdsc_program_linked_size(const pdsc_program *p, unsigned code_offset)
{
   return code_offset + pdsc_program_code_size(p);
}

static void pdsc_encode_instr(pdsc_program *p, pdsc_instr_t i, enum pdsc_op op)
{
   util_dynarray_append(&p->instr_data, pdsc_instr_pack(p, i, op));
}

void pdsc_encode_program(pdsc_program *p)
{
   if (PDSC_DEBUG(PRINT)) {
      fputs("ir before pdsc_encode_program:\n", stdout);
      pdsc_print_program(stdout, p, NULL);
   }

   assert(!util_dynarray_num_elements(&p->instr_data, uint32_t));
   pdsc_instr_t num_instrs = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   for (pdsc_instr_t i = 0; i < num_instrs; ++i) {
      pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
      pdsc_encode_instr(p, i, _i->op);
   }

   /* TODO: alignment? */

   if (PDSC_DEBUG(PRINT_BINARY)) {
      fputs("binary after pdsc_encode_program:\n", stdout);
      pdsc_print_binary(stdout, p);
   }
}

void pdsc_finalize_program(pdsc_program *p)
{
   /* Size code to dword alignment. */
   unsigned size = util_dynarray_num_elements(&p->instr_data, uint8_t);
   size = ALIGN_POT(size, sizeof(uint32_t));
   (void)!!util_dynarray_resize_zero(&p->instr_data, uint8_t, size);

   /* Size data to 4x dword alignment. */
   size = util_dynarray_num_elements(&p->const_data, uint8_t);
   size = ALIGN_POT(size, 4 * sizeof(uint32_t));
   (void)!!util_dynarray_resize_zero(&p->const_data, uint8_t, size);

   if (PDSC_DEBUG(PRINT_BINARY)) {
      fputs("binary after pdsc_finalize_program:\n", stdout);
      pdsc_print_binary(stdout, p);
   }
}

static void pdsc_decode_instr(pdsc_program *p, uint32_t packed)
{
   UNUSED pdsc_instr_t i = pdsc_instr_unpack(p, packed);

   /* TODO: set up used consts, temps, ptemps + decls, etc. based on srcs/dsts
    */
}

void pdsc_decode_program(pdsc_program *p)
{
   /* TODO: alignment? */

   assert(!util_dynarray_num_elements(&p->instrs, pdsc_instr));
   unsigned num_instrs = util_dynarray_num_elements(&p->instr_data, uint32_t);
   for (unsigned u = 0; u < num_instrs; ++u) {
      uint32_t packed = *util_dynarray_element(&p->instr_data, uint32_t, u);
      pdsc_decode_instr(p, packed);
   }

   /* TODO: validate */
   /* TODO: rebuild meta? */
}

/* TODO NEXT: maybe shrink some sizes down? */

static void pdsc_serialize_names(struct blob *blob,
                                 const struct util_dynarray *buf)
{
   uint8_t num_elems = util_dynarray_num_elements(buf, const char **);
   blob_write_uint8(blob, num_elems);

   util_dynarray_foreach (buf, const char **, names) {
      uint8_t num_names = 0;
      if (*names)
         for (const char **aliases = *names; *aliases; ++aliases)
            ++num_names;

      blob_write_uint8(blob, num_names);
      for (unsigned u = 0; u < num_names; ++u)
         blob_write_string(blob, (*names)[u]);
   }
}

void pdsc_serialize(struct blob *blob, const pdsc_program *p)
{
   /* Data segment size. */
   unsigned data_size = pdsc_program_data_size(p);
   blob_write_uint32(blob, data_size);

   /* Data segment. */
   blob_write_bytes(blob, pdsc_program_data(p), data_size);

   /* Code segment size. */
   unsigned code_size = pdsc_program_code_size(p);
   blob_write_uint32(blob, code_size);

   /* Code segment. */
   blob_write_bytes(blob, pdsc_program_code(p), code_size);

   /* Features. */
   blob_write_uint8(blob, p->features);

   blob_write_uint32(blob, p->saved_refs.size);
   blob_write_bytes(blob, p->saved_refs.data, p->saved_refs.size);

   blob_write_uint32(blob, p->const_patch_ids.size);
   blob_write_bytes(blob, p->const_patch_ids.data, p->const_patch_ids.size);

   blob_write_bytes(blob, p->consts_64, sizeof(p->consts_64));
   blob_write_bytes(blob, p->consts_used, sizeof(p->consts_used));
   blob_write_bytes(blob, p->consts_assigned, sizeof(p->consts_assigned));

   blob_write_bytes(blob, p->temps_64, sizeof(p->temps_64));
   blob_write_bytes(blob, p->temps_used, sizeof(p->temps_used));

   blob_write_bytes(blob, p->ptemps_64, sizeof(p->ptemps_64));
   blob_write_bytes(blob, p->ptemps_used, sizeof(p->ptemps_used));

   blob_write_uint32(blob, p->const_types.size);
   blob_write_bytes(blob, p->const_types.data, p->const_types.size);

#if MESA_DEBUG
   /* Name. */
   unsigned name_len = p->name ? strlen(p->name) : 0;
   blob_write_uint32(blob, name_len);
   if (name_len)
      blob_write_bytes(blob, p->name, name_len);

   blob_write_uint16(blob, p->num_doutds);
   blob_write_uint16(blob, p->num_doutws);
   blob_write_uint16(blob, p->num_doutus);
   blob_write_uint16(blob, p->num_doutvs);
   blob_write_uint16(blob, p->num_doutis);
   blob_write_uint16(blob, p->num_doutcs);
   blob_write_uint16(blob, p->num_lds);

   pdsc_serialize_names(blob, &p->const_names);
   pdsc_serialize_names(blob, &p->temp_names);
   pdsc_serialize_names(blob, &p->ptemp_names);
#endif /* MESA_DEBUG */

   /* Align the blob so that we can store it as an array of uint32_ts. */
   blob_align(blob, sizeof(uint32_t));
}

static void pdsc_deserialize_names(struct blob_reader *blob,
                                   pdsc_program *p,
                                   struct util_dynarray *buf)
{
   uint8_t num_elems = blob_read_uint8(blob);
   (void)!!util_dynarray_resize_zero(buf, const char **, num_elems);

   util_dynarray_foreach (buf, const char **, names) {
      uint8_t num_names = blob_read_uint8(blob);
      *names = ralloc_array_size(p, sizeof(*names), num_names + 1);

      for (unsigned u = 0; u < num_names; ++u)
         (*names)[u] = ralloc_strdup(p, blob_read_string(blob));

      (*names)[num_names] = NULL;
   }
}

pdsc_program *pdsc_deserialize(void *mem_ctx, struct blob_reader *blob)
{
   pdsc_program *p = pdsc_program_create(mem_ctx, 0, NULL);
   unsigned sz;

   /* Data segment. */
   sz = blob_read_uint32(blob);
   (void)!!util_dynarray_resize_zero(&p->const_data, uint8_t, sz);
   if (sz)
      memcpy(util_dynarray_begin(&p->const_data),
             blob_read_bytes(blob, sz),
             sz);

   /* Code segment. */
   sz = blob_read_uint32(blob);
   (void)!!util_dynarray_resize_zero(&p->instr_data, uint8_t, sz);
   if (sz)
      memcpy(util_dynarray_begin(&p->instr_data),
             blob_read_bytes(blob, sz),
             sz);

   p->features = blob_read_uint8(blob);

   sz = blob_read_uint32(blob);
   (void)!!util_dynarray_resize_zero(&p->saved_refs, uint8_t, sz);
   if (sz)
      memcpy(util_dynarray_begin(&p->saved_refs),
             blob_read_bytes(blob, sz),
             sz);

   sz = blob_read_uint32(blob);
   (void)!!util_dynarray_resize_zero(&p->const_patch_ids, uint8_t, sz);
   if (sz)
      memcpy(util_dynarray_begin(&p->const_patch_ids),
             blob_read_bytes(blob, sz),
             sz);

   memcpy(p->consts_64,
          blob_read_bytes(blob, sizeof(p->consts_64)),
          sizeof(p->consts_64));
   memcpy(p->consts_used,
          blob_read_bytes(blob, sizeof(p->consts_used)),
          sizeof(p->consts_used));
   memcpy(p->consts_assigned,
          blob_read_bytes(blob, sizeof(p->consts_assigned)),
          sizeof(p->consts_assigned));

   memcpy(p->temps_64,
          blob_read_bytes(blob, sizeof(p->temps_64)),
          sizeof(p->temps_64));
   memcpy(p->temps_used,
          blob_read_bytes(blob, sizeof(p->temps_used)),
          sizeof(p->temps_used));

   memcpy(p->ptemps_64,
          blob_read_bytes(blob, sizeof(p->ptemps_64)),
          sizeof(p->ptemps_64));
   memcpy(p->ptemps_used,
          blob_read_bytes(blob, sizeof(p->ptemps_used)),
          sizeof(p->ptemps_used));

   sz = blob_read_uint32(blob);
   (void)!!util_dynarray_resize_zero(&p->const_types, uint8_t, sz);
   if (sz)
      memcpy(util_dynarray_begin(&p->const_types),
             blob_read_bytes(blob, sz),
             sz);

#if MESA_DEBUG
   /* Name. */
   sz = blob_read_uint32(blob);
   if (sz)
      p->name = ralloc_strdup(p, blob_read_bytes(blob, sz));

   p->num_doutds = blob_read_uint16(blob);
   p->num_doutws = blob_read_uint16(blob);
   p->num_doutus = blob_read_uint16(blob);
   p->num_doutvs = blob_read_uint16(blob);
   p->num_doutis = blob_read_uint16(blob);
   p->num_doutcs = blob_read_uint16(blob);
   p->num_lds = blob_read_uint16(blob);

   pdsc_deserialize_names(blob, p, &p->const_names);
   pdsc_deserialize_names(blob, p, &p->temp_names);
   pdsc_deserialize_names(blob, p, &p->ptemp_names);
#endif /* MESA_DEBUG */

   return p;
}
