/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PDSC_ISA_PACK_H
#define PDSC_ISA_PACK_H

/**
 * \file pdsc_isa_pack.h
 *
 * \brief PDS compiler isa pack/unpack functions.
 */

#include "pdsc.h"
#include "pdsc_internal.h"
#include "pdsc_isa.h"

#include "util/bitpack_helpers.h"

/* TODO NEXT: validation (e.g. for doutu addr needing to be !(addr & 0b11)) */
/* TODO NEXT: packing here so that e.g. temps don't have to be supplied in 4-reg
 * increments.
 */

static inline uint8_t pdsc_regs32_pack(pdsc_ref ref)
{
   if (pdsc_ref_is_null(ref))
      return 0;
   else if (pdsc_ref_is_const32(ref))
      return PDSC_REGS32_CONST_BASE + pdsc_ref_get_val(ref);
   else if (pdsc_ref_is_temp32(ref))
      return PDSC_REGS32_TEMP_BASE + pdsc_ref_get_val(ref);
   else if (pdsc_ref_is_ptemp32(ref))
      return PDSC_REGS32_PTEMP_BASE + pdsc_ref_get_val(ref);

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs32_unpack(uint8_t packed)
{
   if (packed >= PDSC_REGS32_CONST_BASE &&
       packed < PDSC_REGS32_CONST_BASE + PDSC_MAX_CONSTS) {
      return pdsc_ref_const32(packed - PDSC_REGS32_CONST_BASE);
   } else if (packed >= PDSC_REGS32_TEMP_BASE &&
              packed < PDSC_REGS32_TEMP_BASE + PDSC_MAX_TEMPS) {
      return pdsc_ref_temp32(packed - PDSC_REGS32_TEMP_BASE);
   } else if (packed >= PDSC_REGS32_PTEMP_BASE &&
              packed < PDSC_REGS32_PTEMP_BASE + PDSC_MAX_PTEMPS) {
      return pdsc_ref_ptemp32(packed - PDSC_REGS32_PTEMP_BASE);
   }

   UNREACHABLE("Invalid packed data.");
}

static inline uint8_t pdsc_regs32tp_pack(pdsc_ref ref)
{
   if (pdsc_ref_is_null(ref))
      return 0;
   else if (pdsc_ref_is_temp32(ref))
      return PDSC_REGS32TP_TEMP_BASE + pdsc_ref_get_val(ref);
   else if (pdsc_ref_is_ptemp32(ref))
      return PDSC_REGS32TP_PTEMP_BASE + pdsc_ref_get_val(ref);

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs32tp_unpack(uint8_t packed)
{
   if (packed >= PDSC_REGS32TP_TEMP_BASE &&
       packed < PDSC_REGS32TP_TEMP_BASE + PDSC_MAX_TEMPS) {
      return pdsc_ref_temp32(packed - PDSC_REGS32TP_TEMP_BASE);
   } else if (packed >= PDSC_REGS32TP_PTEMP_BASE &&
              packed < PDSC_REGS32TP_PTEMP_BASE + PDSC_MAX_PTEMPS) {
      return pdsc_ref_ptemp32(packed - PDSC_REGS32TP_PTEMP_BASE);
   }

   UNREACHABLE("Invalid packed data.");
}

static inline uint8_t pdsc_regs32t_pack(pdsc_ref ref)
{
   if (pdsc_ref_is_null(ref))
      return 0;
   else if (pdsc_ref_is_temp32(ref))
      return pdsc_ref_get_val(ref);

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs32t_unpack(uint8_t packed)
{
   if (packed < PDSC_MAX_TEMPS)
      return pdsc_ref_temp32(packed);

   UNREACHABLE("Invalid packed data.");
}

static inline uint8_t pdsc_regs32p_pack(pdsc_ref ref)
{
   if (pdsc_ref_is_ptemp32(ref))
      return pdsc_ref_get_val(ref);

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs32p_unpack(uint8_t packed)
{
   if (packed < PDSC_MAX_PTEMPS)
      return pdsc_ref_ptemp32(packed);

   UNREACHABLE("Invalid packed data.");
}

static inline uint8_t pdsc_regs64_pack(pdsc_ref ref)
{
   assert(!(pdsc_ref_get_val(ref) & 0b1));

   if (pdsc_ref_is_null(ref))
      return 0;
   else if (pdsc_ref_is_const64(ref))
      return (PDSC_REGS32_CONST_BASE + pdsc_ref_get_val(ref)) >> 1;
   else if (pdsc_ref_is_temp64(ref))
      return (PDSC_REGS32_TEMP_BASE + pdsc_ref_get_val(ref)) >> 1;
   else if (pdsc_ref_is_ptemp64(ref))
      return (PDSC_REGS32_PTEMP_BASE + pdsc_ref_get_val(ref)) >> 1;

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs64_unpack(uint8_t packed)
{
   packed <<= 1;

   if (packed >= PDSC_REGS32_CONST_BASE &&
       packed < PDSC_REGS32_CONST_BASE + PDSC_MAX_CONSTS) {
      return pdsc_ref_const64(packed - PDSC_REGS32_CONST_BASE);
   } else if (packed >= PDSC_REGS32_TEMP_BASE &&
              packed < PDSC_REGS32_TEMP_BASE + PDSC_MAX_TEMPS) {
      return pdsc_ref_temp64(packed - PDSC_REGS32_TEMP_BASE);
   } else if (packed >= PDSC_REGS32_PTEMP_BASE &&
              packed < PDSC_REGS32_PTEMP_BASE + PDSC_MAX_PTEMPS) {
      return pdsc_ref_ptemp64(packed - PDSC_REGS32_PTEMP_BASE);
   }

   UNREACHABLE("Invalid packed data.");
}

static inline uint8_t pdsc_regs64tp_pack(pdsc_ref ref)
{
   assert(!(pdsc_ref_get_val(ref) & 0b1));

   if (pdsc_ref_is_null(ref))
      return 0;
   else if (pdsc_ref_is_temp64(ref))
      return (PDSC_REGS32TP_TEMP_BASE + pdsc_ref_get_val(ref)) >> 1;
   else if (pdsc_ref_is_ptemp64(ref))
      return (PDSC_REGS32TP_PTEMP_BASE + pdsc_ref_get_val(ref)) >> 1;

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs64tp_unpack(uint8_t packed)
{
   packed <<= 1;

   if (packed >= PDSC_REGS32TP_TEMP_BASE &&
       packed < PDSC_REGS32TP_TEMP_BASE + PDSC_MAX_TEMPS) {
      return pdsc_ref_temp64(packed - PDSC_REGS32TP_TEMP_BASE);
   } else if (packed >= PDSC_REGS32TP_PTEMP_BASE &&
              packed < PDSC_REGS32TP_PTEMP_BASE + PDSC_MAX_PTEMPS) {
      return pdsc_ref_ptemp64(packed - PDSC_REGS32TP_PTEMP_BASE);
   }

   UNREACHABLE("Invalid packed data.");
}

static inline uint8_t pdsc_regs64t_pack(pdsc_ref ref)
{
   assert(!(pdsc_ref_get_val(ref) & 0b1));

   if (pdsc_ref_is_null(ref))
      return 0;
   else if (pdsc_ref_is_temp64(ref))
      return pdsc_ref_get_val(ref) >> 1;

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs64t_unpack(uint8_t packed)
{
   packed <<= 1;

   if (packed < PDSC_MAX_TEMPS)
      return pdsc_ref_temp64(packed);

   UNREACHABLE("Invalid packed data.");
}

static inline uint8_t pdsc_regs64c_pack(pdsc_ref ref)
{
   assert(!(pdsc_ref_get_val(ref) & 0b1));

   if (pdsc_ref_is_null(ref))
      return 0;
   else if (pdsc_ref_is_const64(ref))
      return pdsc_ref_get_val(ref) >> 1;

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs64c_unpack(uint8_t packed)
{
   packed <<= 1;

   if (packed < PDSC_MAX_CONSTS)
      return pdsc_ref_const64(packed);

   UNREACHABLE("Invalid packed data.");
}

static inline uint8_t pdsc_regs64p_pack(pdsc_ref ref)
{
   assert(!(pdsc_ref_get_val(ref) & 0b1));

   if (pdsc_ref_is_ptemp64(ref))
      return pdsc_ref_get_val(ref) >> 1;

   UNREACHABLE("Invalid ref type.");
}

static inline pdsc_ref pdsc_regs64p_unpack(uint8_t packed)
{
   packed <<= 1;

   if (packed < PDSC_MAX_PTEMPS)
      return pdsc_ref_ptemp64(packed);

   UNREACHABLE("Invalid packed data.");
}

static inline uint64_t pdsc_stm_src0_pack(const pdsc_program *p,
                                          const struct pdsc_stm_src0 *src0)
{
   uint64_t packed = 0;

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      packed |= util_bitpack_uint(src0->slcmode, 62, 63);

   assert(pdsc_ref_is_ptemp64(src0->ptemp_prim_needed_written));
   packed |=
      util_bitpack_uint(pdsc_regs64p_pack(src0->ptemp_prim_needed_written),
                        45,
                        49);
   assert(pdsc_ref_is_ptemp64(src0->ptemp_address));
   packed |= util_bitpack_uint(pdsc_regs64p_pack(src0->ptemp_address), 40, 44);
   assert(!(src0->address & 0b11));
   packed |= util_bitpack_uint(src0->address, 0, 39);

   return packed;
}

static inline void pdsc_stm_src0_unpack(const pdsc_program *p,
                                        struct pdsc_stm_src0 *src0,
                                        uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      src0->slcmode = util_bitpack_uint_unpack(packed, 62, 63);

   uint8_t ptemp_prim_needed_written_packed =
      util_bitpack_uint_unpack(packed, 45, 49);
   src0->ptemp_prim_needed_written =
      pdsc_regs64p_unpack(ptemp_prim_needed_written_packed);
   uint8_t ptemp_address_packed = util_bitpack_uint_unpack(packed, 40, 44);
   src0->ptemp_address = pdsc_regs64p_unpack(ptemp_address_packed);
   src0->address = util_bitpack_uint_unpack(packed, 0, 39);
   assert(!(src0->address & 0b11));
}

static inline uint64_t pdsc_stm_src1_pack(const pdsc_program *p,
                                          const struct pdsc_stm_src1 *src1)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint(src1->prim_needed, 32, 63);
   packed |= util_bitpack_uint(src1->prim_written, 0, 31);

   return packed;
}

static inline void pdsc_stm_src1_unpack(const pdsc_program *p,
                                        struct pdsc_stm_src1 *src1,
                                        uint64_t packed)
{
   memset(src1, 0, sizeof(*src1));

   src1->prim_needed = util_bitpack_uint_unpack(packed, 32, 63);
   src1->prim_written = util_bitpack_uint_unpack(packed, 0, 31);
}

static inline uint32_t pdsc_stm_src2_pack(const pdsc_program *p,
                                          const struct pdsc_stm_src2 *src2)
{
   uint32_t packed = 0;

   packed |= util_bitpack_uint(src2->vioff, 24, 30);
   packed |= util_bitpack_uint(src2->dmasize, 9, 16);
   packed |= util_bitpack_uint(src2->vooff, 0, 8);

   return packed;
}

static inline void pdsc_stm_src2_unpack(const pdsc_program *p,
                                        struct pdsc_stm_src2 *src2,
                                        uint32_t packed)
{
   memset(src2, 0, sizeof(*src2));

   src2->vioff = util_bitpack_uint_unpack(packed, 24, 30);
   src2->dmasize = util_bitpack_uint_unpack(packed, 9, 16);
   src2->vooff = util_bitpack_uint_unpack(packed, 0, 8);
}

static inline uint64_t pdsc_stm_src3_pack(const pdsc_program *p,
                                          const struct pdsc_stm_src3 *src3)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint_nonzero(src3->primtype, 53, 54);
   packed |= util_bitpack_uint(src3->vosize, 44, 52);
   packed |= util_bitpack_uint(src3->eop, 43, 43);
   assert(!(src3->limit & 0b11));
   packed |= util_bitpack_uint(src3->limit, 0, 39);

   return packed;
}

static inline void pdsc_stm_src3_unpack(const pdsc_program *p,
                                        struct pdsc_stm_src3 *src3,
                                        uint64_t packed)
{
   memset(src3, 0, sizeof(*src3));

   src3->primtype = util_bitpack_uint_unpack_nonzero(packed, 53, 54);
   src3->vosize = util_bitpack_uint_unpack(packed, 44, 52);
   src3->eop = util_bitpack_uint_unpack(packed, 43, 43);
   src3->limit = util_bitpack_uint_unpack(packed, 0, 39);
   assert(!(src3->limit & 0b11));
}

static inline uint64_t pdsc_ld64_src0_pack(const pdsc_program *p,
                                           const struct pdsc_ld64_src0 *src0)
{
   uint64_t packed = 0;

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      packed |= util_bitpack_uint(src0->slcmode, 62, 63);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      packed |= util_bitpack_uint(src0->ccpri, 54, 55);
      packed |= util_bitpack_uint(src0->ccslc, 53, 53);
      packed |= util_bitpack_uint(src0->ccmcu, 52, 52);
   }

   packed |= util_bitpack_uint(pdsc_regs64tp_pack(src0->dest), 47, 51);
   packed |= util_bitpack_uint(src0->cmode, 44, 45);
   packed |= util_bitpack_uint(src0->count8, 41, 43);
   assert(!(src0->srcadd & 0b11));
   packed |= util_bitpack_uint(src0->srcadd, 0, 39);

   return packed;
}

static inline void pdsc_ld64_src0_unpack(const pdsc_program *p,
                                         struct pdsc_ld64_src0 *src0,
                                         uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      src0->slcmode = util_bitpack_uint_unpack(packed, 62, 63);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      src0->ccpri = util_bitpack_uint_unpack(packed, 54, 55);
      src0->ccslc = util_bitpack_uint_unpack(packed, 53, 53);
      src0->ccmcu = util_bitpack_uint_unpack(packed, 52, 52);
   }

   uint8_t dest_packed = util_bitpack_uint_unpack(packed, 47, 51);
   src0->dest = pdsc_regs64tp_unpack(dest_packed);
   src0->cmode = util_bitpack_uint_unpack(packed, 44, 45);
   src0->count8 = util_bitpack_uint_unpack(packed, 41, 43);
   src0->srcadd = util_bitpack_uint_unpack(packed, 0, 39);
   assert(!(src0->srcadd & 0b11));
}

static inline uint64_t pdsc_st32_src0_pack(const pdsc_program *p,
                                           const struct pdsc_st32_src0 *src0)
{
   uint64_t packed = 0;

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      packed |= util_bitpack_uint(src0->slcmode, 62, 63);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      packed |= util_bitpack_uint(src0->ccpri, 54, 55);
      packed |= util_bitpack_uint(src0->ccslc, 53, 53);
      packed |= util_bitpack_uint(src0->ccmcu, 52, 52);
   }

   packed |= util_bitpack_uint(pdsc_regs32tp_pack(src0->src), 46, 51);
   packed |= util_bitpack_uint(src0->cmode, 44, 45);
   packed |= util_bitpack_uint(src0->count4, 40, 43);
   assert(!(src0->dstadd & 0b11));
   packed |= util_bitpack_uint(src0->dstadd, 0, 39);

   return packed;
}

static inline void pdsc_st32_src0_unpack(const pdsc_program *p,
                                         struct pdsc_st32_src0 *src0,
                                         uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      src0->slcmode = util_bitpack_uint_unpack(packed, 62, 63);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      src0->ccpri = util_bitpack_uint_unpack(packed, 54, 55);
      src0->ccslc = util_bitpack_uint_unpack(packed, 53, 53);
      src0->ccmcu = util_bitpack_uint_unpack(packed, 52, 52);
   }

   uint8_t src_packed = util_bitpack_uint_unpack(packed, 46, 51);
   src0->src = pdsc_regs32tp_unpack(src_packed);
   src0->cmode = util_bitpack_uint_unpack(packed, 44, 45);
   src0->count4 = util_bitpack_uint_unpack(packed, 40, 43);
   src0->dstadd = util_bitpack_uint_unpack(packed, 0, 39);
   assert(!(src0->dstadd & 0b11));
}

static inline uint64_t pdsc_stmp_src0_pack(const pdsc_program *p,
                                           const struct pdsc_stmp_src0 *src0)
{
   uint64_t packed = 0;

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      packed |= util_bitpack_uint(src0->slcmode, 62, 63);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      packed |= util_bitpack_uint(src0->ccpri, 54, 55);
      packed |= util_bitpack_uint(src0->ccslc, 53, 53);
      packed |= util_bitpack_uint(src0->ccmcu, 52, 52);
   }

   packed |=
      util_bitpack_uint(pdsc_regs32p_pack(src0->ptemp_prim_written), 40, 45);
   assert(!(src0->address & 0b11));
   packed |= util_bitpack_uint(src0->address, 0, 39);

   return packed;
}

static inline void pdsc_stmp_src0_unpack(const pdsc_program *p,
                                         struct pdsc_stmp_src0 *src0,
                                         uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      src0->slcmode = util_bitpack_uint_unpack(packed, 62, 63);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      src0->ccpri = util_bitpack_uint_unpack(packed, 54, 55);
      src0->ccslc = util_bitpack_uint_unpack(packed, 53, 53);
      src0->ccmcu = util_bitpack_uint_unpack(packed, 52, 52);
   }

   uint8_t ptemp_prim_written_packed = util_bitpack_uint_unpack(packed, 40, 45);
   src0->ptemp_prim_written = pdsc_regs32p_unpack(ptemp_prim_written_packed);
   src0->address = util_bitpack_uint_unpack(packed, 0, 39);
   assert(!(src0->address & 0b11));
}

static inline uint64_t pdsc_stmp_src1_pack(const pdsc_program *p,
                                           const struct pdsc_stmp_src1 *src1)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint(src1->prim_id, 32, 63);
   packed |= util_bitpack_uint(src1->vioff, 24, 30);
   packed |= util_bitpack_uint(src1->dmasize, 9, 16);
   packed |= util_bitpack_uint(src1->vooff, 0, 8);

   return packed;
}

static inline void pdsc_stmp_src1_unpack(const pdsc_program *p,
                                         struct pdsc_stmp_src1 *src1,
                                         uint64_t packed)
{
   memset(src1, 0, sizeof(*src1));

   src1->prim_id = util_bitpack_uint_unpack(packed, 32, 63);
   src1->vioff = util_bitpack_uint_unpack(packed, 24, 30);
   src1->dmasize = util_bitpack_uint_unpack(packed, 9, 16);
   src1->vooff = util_bitpack_uint_unpack(packed, 0, 8);
}

static inline uint64_t pdsc_stmp_src2_pack(const pdsc_program *p,
                                           const struct pdsc_stmp_src2 *src2)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint_nonzero(src2->primtype, 53, 54);
   packed |= util_bitpack_uint(src2->vosize, 44, 52);
   packed |= util_bitpack_uint(src2->eop, 43, 43);
   assert(!(src2->limit & 0b11));
   packed |= util_bitpack_uint(src2->limit, 0, 39);

   return packed;
}

static inline void pdsc_stmp_src2_unpack(const pdsc_program *p,
                                         struct pdsc_stmp_src2 *src2,
                                         uint64_t packed)
{
   memset(src2, 0, sizeof(*src2));

   src2->primtype = util_bitpack_uint_unpack_nonzero(packed, 53, 54);
   src2->vosize = util_bitpack_uint_unpack(packed, 44, 52);
   src2->eop = util_bitpack_uint_unpack(packed, 43, 43);
   src2->limit = util_bitpack_uint_unpack(packed, 0, 39);
   assert(!(src2->limit & 0b11));
}

static inline uint64_t pdsc_aa_src0_pack(const pdsc_program *p,
                                         const struct pdsc_aa_src0 *src0)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint(src0->aop, 56, 59);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      packed |= util_bitpack_uint(src0->ccpri, 54, 55);
      packed |= util_bitpack_uint(src0->ccslc, 53, 53);
      packed |= util_bitpack_uint(src0->ccmcu, 52, 52);
   }

   assert(!(src0->srcadd & 0b11));
   packed |= util_bitpack_uint(src0->srcadd, 0, 39);

   return packed;
}

static inline void pdsc_aa_src0_unpack(const pdsc_program *p,
                                       struct pdsc_aa_src0 *src0,
                                       uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   src0->aop = util_bitpack_uint_unpack(packed, 56, 59);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      src0->ccpri = util_bitpack_uint_unpack(packed, 54, 55);
      src0->ccslc = util_bitpack_uint_unpack(packed, 53, 53);
      src0->ccmcu = util_bitpack_uint_unpack(packed, 52, 52);
   }

   src0->srcadd = util_bitpack_uint(packed, 0, 39);
   assert(!(src0->srcadd & 0b11));
}

static inline uint64_t pdsc_idf_src0_pack(const pdsc_program *p,
                                          const struct pdsc_idf_src0 *src0)
{
   uint64_t packed = 0;

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      packed |= util_bitpack_uint(src0->ccpri, 54, 55);
      packed |= util_bitpack_uint(src0->ccslc, 53, 53);
      packed |= util_bitpack_uint(src0->ccmcu, 52, 52);
   }

   assert(!(src0->srcadd & 0b11));
   packed |= util_bitpack_uint(src0->srcadd, 0, 39);

   return packed;
}

static inline void pdsc_idf_src0_unpack(const pdsc_program *p,
                                        struct pdsc_idf_src0 *src0,
                                        uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      src0->ccpri = util_bitpack_uint_unpack(packed, 54, 55);
      src0->ccslc = util_bitpack_uint_unpack(packed, 53, 53);
      src0->ccmcu = util_bitpack_uint_unpack(packed, 52, 52);
   }

   src0->srcadd = util_bitpack_uint(packed, 0, 39);
   assert(!(src0->srcadd & 0b11));
}

static inline uint64_t pdsc_pol_src0_pack(const pdsc_program *p,
                                          const struct pdsc_pol_src0 *src0)
{
   uint64_t packed = 0;

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      packed |= util_bitpack_uint(src0->ccpri, 54, 55);
      packed |= util_bitpack_uint(src0->ccslc, 53, 53);
      packed |= util_bitpack_uint(src0->ccmcu, 52, 52);
   }

   packed |= util_bitpack_uint(pdsc_regs32tp_pack(src0->dest), 46, 51);

   assert(!(src0->srcadd & 0b11));
   packed |= util_bitpack_uint(src0->srcadd, 0, 39);

   return packed;
}

static inline void pdsc_pol_src0_unpack(const pdsc_program *p,
                                        struct pdsc_pol_src0 *src0,
                                        uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      src0->ccpri = util_bitpack_uint_unpack(packed, 54, 55);
      src0->ccslc = util_bitpack_uint_unpack(packed, 53, 53);
      src0->ccmcu = util_bitpack_uint_unpack(packed, 52, 52);
   }

   uint8_t dest_packed = util_bitpack_uint_unpack(packed, 46, 51);
   src0->dest = pdsc_regs32tp_unpack(dest_packed);

   src0->srcadd = util_bitpack_uint(packed, 0, 39);
   assert(!(src0->srcadd & 0b11));
}

static inline uint64_t pdsc_ddmad_src3_pack(const pdsc_program *p,
                                            const struct pdsc_ddmad_src3 *src3)
{
   uint64_t packed = 0;

   if (PDSC_HAS(p, DDMADT)) {
      packed |= util_bitpack_uint(src3->msize, 33, 63);
      packed |= util_bitpack_uint(src3->test, 32, 32);
   }

   packed |= util_bitpack_uint(src3->last_issue, 31, 31);
   packed |= util_bitpack_uint(src3->store_dest, 28, 28);
   packed |= util_bitpack_uint(src3->cmode, 26, 27);
   packed |= util_bitpack_uint(src3->ao, 13, 25);
   packed |= util_bitpack_uint(src3->bsize, 0, 11);

   return packed;
}

static inline void pdsc_ddmad_src3_unpack(const pdsc_program *p,
                                          struct pdsc_ddmad_src3 *src3,
                                          uint64_t packed)
{
   memset(src3, 0, sizeof(*src3));

   if (PDSC_HAS(p, DDMADT)) {
      src3->msize = util_bitpack_uint_unpack(packed, 33, 63);
      src3->test = util_bitpack_uint_unpack(packed, 32, 32);
   }

   src3->last_issue = util_bitpack_uint_unpack(packed, 31, 31);
   src3->store_dest = util_bitpack_uint_unpack(packed, 28, 28);
   src3->cmode = util_bitpack_uint_unpack(packed, 26, 27);
   src3->ao = util_bitpack_uint_unpack(packed, 13, 25);
   src3->bsize = util_bitpack_uint_unpack(packed, 0, 11);
}

static inline uint64_t pdsc_doutd_src0_pack(const pdsc_program *p,
                                            const struct pdsc_doutd_src0 *src0)
{
   uint64_t packed = 0;

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      packed |= util_bitpack_uint(src0->slcmode, 62, 63);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      packed |= util_bitpack_uint(src0->ccpri, 60, 61);
      packed |= util_bitpack_uint(src0->ccslc, 59, 59);
      packed |= util_bitpack_uint(src0->ccmcu, 58, 58);
   }

   packed |= util_bitpack_uint(src0->doffset, 40, 52);
   packed |= util_bitpack_uint(src0->sbase, 0, 39);

   return packed;
}

static inline void pdsc_doutd_src0_unpack(const pdsc_program *p,
                                          struct pdsc_doutd_src0 *src0,
                                          uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      src0->slcmode = util_bitpack_uint_unpack(packed, 62, 63);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      src0->ccpri = util_bitpack_uint_unpack(packed, 60, 61);
      src0->ccslc = util_bitpack_uint_unpack(packed, 59, 59);
      src0->ccmcu = util_bitpack_uint_unpack(packed, 58, 58);
   }

   src0->doffset = util_bitpack_uint_unpack(packed, 40, 52);
   src0->sbase = util_bitpack_uint_unpack(packed, 0, 39);
}

static inline uint32_t pdsc_doutd_src1_pack(const pdsc_program *p,
                                            const struct pdsc_doutd_src1 *src1)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint(src1->last_issue, 31, 31);
   packed |= util_bitpack_uint(src1->store_dest, 28, 28);
   packed |= util_bitpack_uint(src1->cmode, 26, 27);
   packed |= util_bitpack_uint(src1->ao, 13, 25);
   packed |= util_bitpack_uint(src1->bsize, 0, 11);

   return (uint32_t)packed;
}

static inline void pdsc_doutd_src1_unpack(const pdsc_program *p,
                                          struct pdsc_doutd_src1 *src1,
                                          uint32_t packed)
{
   memset(src1, 0, sizeof(*src1));

   src1->last_issue = util_bitpack_uint_unpack(packed, 31, 31);
   src1->store_dest = util_bitpack_uint_unpack(packed, 28, 28);
   src1->cmode = util_bitpack_uint_unpack(packed, 26, 27);
   src1->ao = util_bitpack_uint_unpack(packed, 13, 25);
   src1->bsize = util_bitpack_uint_unpack(packed, 0, 11);
}

static inline uint32_t pdsc_doutw_src1_pack(const pdsc_program *p,
                                            const struct pdsc_doutw_src1 *src1)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint(src1->last_issue, 31, 31);
   packed |= util_bitpack_uint(src1->store_dest, 28, 28);

   if (!PDSC_HAS(p, DOUT_EXT))
      packed |= util_bitpack_uint(src1->cmode, 26, 27);

   packed |= util_bitpack_uint(src1->ao, 13, 25);
   packed |= util_bitpack_uint(src1->bsize, 0, 2);

   return (uint32_t)packed;
}

static inline void pdsc_doutw_src1_unpack(const pdsc_program *p,
                                          struct pdsc_doutw_src1 *src1,
                                          uint32_t packed)
{
   memset(src1, 0, sizeof(*src1));

   src1->last_issue = util_bitpack_uint_unpack(packed, 31, 31);
   src1->store_dest = util_bitpack_uint_unpack(packed, 28, 28);

   if (!PDSC_HAS(p, DOUT_EXT))
      src1->cmode = util_bitpack_uint_unpack(packed, 26, 27);

   src1->ao = util_bitpack_uint_unpack(packed, 13, 25);
   src1->bsize = util_bitpack_uint_unpack(packed, 0, 2);
}

static inline uint64_t pdsc_doutu_src0_pack(const pdsc_program *p,
                                            const struct pdsc_doutu_src0 *src0)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint(src0->dual_phase, 41, 41);
   packed |=
      util_bitpack_uint(DIV_ROUND_UP(src0->temps, PDSC_DOUTU_TEMP_GRANULARITY),
                        35,
                        40);
   packed |= util_bitpack_uint(src0->sample_rate, 33, 34);
   assert(!(src0->exe_off & 0b11));
   packed |= util_bitpack_uint(src0->exe_off, 0, 31);

   return packed;
}

static inline void pdsc_doutu_src0_unpack(const pdsc_program *p,
                                          struct pdsc_doutu_src0 *src0,
                                          uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   src0->dual_phase = util_bitpack_uint_unpack(packed, 41, 41);
   src0->temps =
      util_bitpack_uint_unpack(packed, 35, 40) * PDSC_DOUTU_TEMP_GRANULARITY;
   src0->sample_rate = util_bitpack_uint_unpack(packed, 33, 34);
   src0->exe_off = util_bitpack_uint_unpack(packed, 0, 31);
   assert(!(src0->exe_off & 0b11));
}

static inline uint64_t pdsc_douti_src0_pack(const pdsc_program *p,
                                            const struct pdsc_douti_src0 *src0)
{
   uint64_t packed = 0;

   packed |= util_bitpack_uint(src0->last_issue, 63, 63);
   packed |= util_bitpack_uint(src0->dest, 54, 61);

   packed |= util_bitpack_uint(src0->depth_bias, 27, 27);
   packed |= util_bitpack_uint(src0->primitive_id, 26, 26);
   packed |= util_bitpack_uint(src0->shademodel, 24, 25);
   packed |= util_bitpack_uint(src0->point_sprite, 23, 23);
   packed |= util_bitpack_uint(src0->wrap_u, 22, 22);
   packed |= util_bitpack_uint(src0->wrap_v, 21, 21);
   packed |= util_bitpack_uint(src0->wrap_s, 20, 20);
   packed |= util_bitpack_uint(src0->size, 18, 19);
   packed |= util_bitpack_uint(src0->f16, 17, 17);
   packed |= util_bitpack_uint(src0->perspective, 16, 16);
   packed |= util_bitpack_uint(src0->f32_offset, 8, 15);
   packed |= util_bitpack_uint(src0->f16_offset, 0, 7);

   return packed;
}

static inline void pdsc_douti_src0_unpack(const pdsc_program *p,
                                          struct pdsc_douti_src0 *src0,
                                          uint64_t packed)
{
   memset(src0, 0, sizeof(*src0));

   src0->last_issue = util_bitpack_uint_unpack(packed, 63, 63);
   src0->dest = util_bitpack_uint_unpack(packed, 54, 61);

   src0->depth_bias = util_bitpack_uint_unpack(packed, 27, 27);
   src0->primitive_id = util_bitpack_uint_unpack(packed, 26, 26);
   src0->shademodel = util_bitpack_uint_unpack(packed, 24, 25);
   src0->point_sprite = util_bitpack_uint_unpack(packed, 23, 23);
   src0->wrap_u = util_bitpack_uint_unpack(packed, 22, 22);
   src0->wrap_v = util_bitpack_uint_unpack(packed, 21, 21);
   src0->wrap_s = util_bitpack_uint_unpack(packed, 20, 20);
   src0->size = util_bitpack_uint_unpack(packed, 18, 19);
   src0->f16 = util_bitpack_uint_unpack(packed, 17, 17);
   src0->perspective = util_bitpack_uint_unpack(packed, 16, 16);
   src0->f32_offset = util_bitpack_uint_unpack(packed, 8, 15);
   src0->f16_offset = util_bitpack_uint_unpack(packed, 0, 7);
}

static inline uint32_t pdsc_sftlp64_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SFTLP64, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);

   enum pdsc_lop lop = pdsc_ref_is_null(_i->src[1]) ? PDSC_LOP_NONE
                                                    : pdsc_get_lop(p, i);
   packed |= util_bitpack_uint(lop, 24, 26);

   bool im = pdsc_ref_is_null(_i->src[2]) || pdsc_ref_is_simm(_i->src[2]);
   packed |= util_bitpack_uint(im, 23, 23);

   packed |= util_bitpack_uint(pdsc_regs64tp_pack(_i->src[0]), 18, 22);
   packed |= util_bitpack_uint(pdsc_regs64tp_pack(_i->src[1]), 13, 17);

   uint8_t src2;
   if (pdsc_ref_is_null(_i->src[2])) {
      src2 = 0;
   } else if (pdsc_ref_is_simm(_i->src[2])) {
      assert(_i->src[2].sval >= -64 && _i->src[2].sval <= 64);
      src2 = _i->src[2].val & 0xff;
   } else {
      src2 = pdsc_regs32_pack(_i->src[2]);
   }

   packed |= util_bitpack_uint(src2, 5, 12);
   packed |= util_bitpack_uint(pdsc_regs64tp_pack(_i->dst), 0, 4);

   return packed;
}

static inline pdsc_instr_t pdsc_sftlp64_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);
   bool cc = util_bitpack_uint_unpack(packed, 27, 27);
   enum pdsc_lop lop = util_bitpack_uint_unpack(packed, 24, 26);
   bool im = util_bitpack_uint_unpack(packed, 23, 23);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 18, 22);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 13, 17);
   uint8_t src2_packed = util_bitpack_uint_unpack(packed, 5, 12);
   int8_t src2_spacked = util_bitpack_sint_unpack(packed, 5, 12);
   uint8_t dst_packed = util_bitpack_uint_unpack(packed, 0, 4);

   pdsc_ref src0 = pdsc_regs64tp_unpack(src0_packed);
   pdsc_ref src1 = (lop == PDSC_LOP_NONE) ? pdsc_ref_null()
                                          : pdsc_regs64tp_unpack(src1_packed);
   pdsc_ref src2 =
      !im ? pdsc_regs32_unpack(src2_packed)
          : (!src2_packed ? pdsc_ref_null() : pdsc_ref_simm(src2_spacked));
   pdsc_ref dst = pdsc_regs64tp_unpack(dst_packed);

   assert(opcodec == PDSC_OPCODEC_SFTLP64);
   return pdsc_sftlp64(p, dst, src0, src1, src2, .cc = cc, .lop = lop);
}

static inline uint32_t pdsc_sftlp32_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEB_SFTLP32, 29, 31);

   bool im = pdsc_ref_is_null(_i->src[2]) || pdsc_ref_is_simm(_i->src[2]);
   packed |= util_bitpack_uint(im, 28, 28);

   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);

   enum pdsc_lop lop = pdsc_ref_is_null(_i->src[1]) ? PDSC_LOP_NONE
                                                    : pdsc_get_lop(p, i);
   packed |= util_bitpack_uint(lop, 24, 26);

   packed |= util_bitpack_uint(pdsc_regs32t_pack(_i->src[0]), 19, 23);
   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[1]), 11, 18);

   uint8_t src2;
   if (pdsc_ref_is_null(_i->src[2])) {
      src2 = 0;
   } else if (pdsc_ref_is_simm(_i->src[2])) {
      assert(_i->src[2].sval >= -32 && _i->src[2].sval <= 32);
      src2 = _i->src[2].val & 0x3f;
   } else {
      src2 = pdsc_regs32tp_pack(_i->src[2]);
   }

   packed |= util_bitpack_uint(src2, 5, 10);
   packed |= util_bitpack_uint(pdsc_regs32t_pack(_i->dst), 0, 4);

   return packed;
}

static inline pdsc_instr_t pdsc_sftlp32_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodeb opcodeb =
      util_bitpack_uint_unpack(packed, 29, 31);
   bool im = util_bitpack_uint_unpack(packed, 28, 28);
   bool cc = util_bitpack_uint_unpack(packed, 27, 27);
   enum pdsc_lop lop = util_bitpack_uint_unpack(packed, 24, 26);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 19, 23);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 11, 18);
   uint8_t src2_packed = util_bitpack_uint_unpack(packed, 5, 10);
   int8_t src2_spacked = util_bitpack_sint_unpack(packed, 5, 10);
   uint8_t dst_packed = util_bitpack_uint_unpack(packed, 0, 4);

   pdsc_ref src0 = pdsc_regs32t_unpack(src0_packed);
   pdsc_ref src1 = (lop == PDSC_LOP_NONE) ? pdsc_ref_null()
                                          : pdsc_regs32_unpack(src1_packed);
   pdsc_ref src2 =
      !im ? pdsc_regs32_unpack(src2_packed)
          : (!src2_packed ? pdsc_ref_null() : pdsc_ref_simm(src2_spacked));
   pdsc_ref dst = pdsc_regs32tp_unpack(dst_packed);

   assert(opcodeb == PDSC_OPCODEB_SFTLP32);
   return pdsc_sftlp32(p, dst, src0, src1, src2, .cc = cc, .lop = lop);
}

static inline uint32_t pdsc_stm_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEB_STM, 29, 31);

   packed |= util_bitpack_uint(!!pdsc_get_cc_global_ovf(p, i), 28, 28);
   packed |= util_bitpack_uint(!!pdsc_get_cc_so_ovf(p, i), 27, 27);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 26, 26);

   packed |= util_bitpack_uint(!!pdsc_get_tst(p, i), 25, 25);
   packed |= util_bitpack_uint(pdsc_get_so(p, i), 23, 24);

   packed |= util_bitpack_uint(pdsc_regs64tp_pack(_i->src[0]), 18, 22);
   packed |= util_bitpack_uint(pdsc_regs64tp_pack(_i->src[1]), 13, 17);
   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[2]), 5, 12);
   packed |= util_bitpack_uint(pdsc_regs64tp_pack(_i->src[3]), 0, 4);

   return packed;
}

static inline pdsc_instr_t pdsc_stm_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodeb opcodeb =
      util_bitpack_uint_unpack(packed, 29, 31);

   bool cc_global_ovf = util_bitpack_uint_unpack(packed, 28, 28);
   bool cc_so_ovf = util_bitpack_uint_unpack(packed, 27, 27);
   bool cc = util_bitpack_uint_unpack(packed, 26, 26);
   bool tst = util_bitpack_uint_unpack(packed, 25, 25);
   uint8_t so = util_bitpack_uint_unpack(packed, 23, 24);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 18, 22);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 13, 17);
   uint8_t src2_packed = util_bitpack_uint_unpack(packed, 5, 12);
   uint8_t src3_packed = util_bitpack_uint_unpack(packed, 0, 4);

   pdsc_ref src0 = pdsc_regs64tp_unpack(src0_packed);
   pdsc_ref src1 = pdsc_regs64tp_unpack(src1_packed);
   pdsc_ref src2 = pdsc_regs32_unpack(src2_packed);
   pdsc_ref src3 = pdsc_regs64tp_unpack(src3_packed);

   assert(opcodeb == PDSC_OPCODEB_STM);
   return pdsc_stm(p,
                   src0,
                   src1,
                   src2,
                   src3,
                   .cc = cc,
                   .cc_so_ovf = cc_so_ovf,
                   .cc_global_ovf = cc_global_ovf,
                   .tst = tst,
                   .so = so);
}

static inline uint32_t pdsc_mad_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEA_MAD, 30, 31);

   packed |= util_bitpack_uint(!!pdsc_get_sna(p, i), 29, 29);
   packed |= util_bitpack_uint(!!pdsc_get_alum(p, i), 28, 28);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);

   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[0]), 19, 26);
   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[1]), 11, 18);
   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[2]), 4, 10);
   packed |= util_bitpack_uint(pdsc_regs64t_pack(_i->dst), 0, 3);

   return packed;
}

static inline pdsc_instr_t pdsc_mad_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodea opcodea =
      util_bitpack_uint_unpack(packed, 30, 31);

   bool sna = util_bitpack_uint_unpack(packed, 29, 29);
   bool alum = util_bitpack_uint_unpack(packed, 28, 28);
   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 19, 26);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 11, 18);
   uint8_t src2_packed = util_bitpack_uint_unpack(packed, 4, 10);
   uint8_t dst_packed = util_bitpack_uint_unpack(packed, 0, 3);

   pdsc_ref src0 = pdsc_regs32_unpack(src0_packed);
   pdsc_ref src1 = pdsc_regs32_unpack(src1_packed);
   pdsc_ref src2 = pdsc_regs64_unpack(src2_packed);
   pdsc_ref dst = pdsc_regs64t_unpack(dst_packed);

   assert(opcodea == PDSC_OPCODEA_MAD);
   return pdsc_mad(p, dst, src0, src1, src2, .cc = cc, .sna = sna, .alum = alum);
}

static inline uint32_t pdsc_add64_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_ADD64, 28, 31);

   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(!!pdsc_get_alum(p, i), 26, 26);
   packed |= util_bitpack_uint(!!pdsc_get_sna(p, i), 24, 24);

   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 12, 18);
   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[1]), 5, 11);
   packed |= util_bitpack_uint(pdsc_regs64tp_pack(_i->dst), 0, 4);

   return packed;
}

static inline pdsc_instr_t pdsc_add64_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);
   bool alum = util_bitpack_uint_unpack(packed, 26, 26);
   bool sna = util_bitpack_uint_unpack(packed, 24, 24);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 12, 18);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 5, 11);
   uint8_t dst_packed = util_bitpack_uint_unpack(packed, 0, 4);

   pdsc_ref src0 = pdsc_regs64_unpack(src0_packed);
   pdsc_ref src1 = pdsc_regs64_unpack(src1_packed);
   pdsc_ref dst = pdsc_regs64tp_unpack(dst_packed);

   assert(opcodec == PDSC_OPCODEC_ADD64);
   return pdsc_add64(p, dst, src0, src1, .cc = cc, .sna = sna, .alum = alum);
}

static inline uint32_t pdsc_add32_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_ADD32, 28, 31);

   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(!!pdsc_get_alum(p, i), 26, 26);
   packed |= util_bitpack_uint(!!pdsc_get_sna(p, i), 24, 24);

   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[0]), 14, 21);
   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[1]), 6, 13);
   packed |= util_bitpack_uint(pdsc_regs32tp_pack(_i->dst), 0, 5);

   return packed;
}

static inline pdsc_instr_t pdsc_add32_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);
   bool alum = util_bitpack_uint_unpack(packed, 26, 26);
   bool sna = util_bitpack_uint_unpack(packed, 24, 24);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 14, 21);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 6, 13);
   uint8_t dst_packed = util_bitpack_uint_unpack(packed, 0, 5);

   pdsc_ref src0 = pdsc_regs32_unpack(src0_packed);
   pdsc_ref src1 = pdsc_regs32_unpack(src1_packed);
   pdsc_ref dst = pdsc_regs32tp_unpack(dst_packed);

   assert(opcodec == PDSC_OPCODEC_ADD32);
   return pdsc_add32(p, dst, src0, src1, .cc = cc, .sna = sna, .alum = alum);
}

static inline uint32_t pdsc_cmp_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_CMP, 28, 31);

   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(pdsc_get_cop(p, i), 25, 26);

   bool setcp = true;
   packed |= util_bitpack_uint(setcp, 24, 24);

   bool im = pdsc_ref_is_imm(_i->src[1]);
   packed |= util_bitpack_uint(im, 23, 23);

   packed |= util_bitpack_uint(pdsc_regs64tp_pack(_i->src[0]), 18, 22);

   if (im)
      packed |= util_bitpack_uint(_i->src[1].val, 2, 17);
   else
      packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[1]), 2, 8);

   return packed;
}

static inline pdsc_instr_t pdsc_cmp_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);
   enum pdsc_cop cop = util_bitpack_uint_unpack(packed, 25, 26);

   ASSERTED bool setcp = util_bitpack_uint_unpack(packed, 24, 24);
   assert(setcp);

   bool im = util_bitpack_uint_unpack(packed, 23, 23);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 18, 22);

   pdsc_ref src0 = pdsc_regs64tp_unpack(src0_packed);

   pdsc_ref src1;
   if (im) {
      uint8_t src1_packed = util_bitpack_uint_unpack(packed, 2, 17);
      src1 = pdsc_ref_imm(src1_packed);
   } else {
      uint8_t src1_packed = util_bitpack_uint_unpack(packed, 2, 8);
      src1 = pdsc_regs64_unpack(src1_packed);
   }

   assert(opcodec == PDSC_OPCODEC_CMP);
   return pdsc_cmp(p, src0, src1, .cc = cc, .cop = cop);
}

static inline uint32_t pdsc_bra_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_BRA, 28, 31);

   packed |= util_bitpack_uint(pdsc_get_srcc(p, i), 24, 27);
   packed |= util_bitpack_uint(pdsc_get_neg(p, i), 23, 23);
   packed |= util_bitpack_uint(pdsc_get_setc(p, i), 19, 22);

   int32_t addr = _i->target_instr - i;
   packed |= util_bitpack_sint(addr, 0, 18);

   return packed;
}

static inline pdsc_instr_t pdsc_bra_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   enum pdsc_pred srcc = util_bitpack_uint_unpack(packed, 24, 27);
   bool neg = util_bitpack_uint_unpack(packed, 23, 23);
   enum pdsc_pred setc = util_bitpack_uint_unpack(packed, 19, 22);

   int32_t addr = util_bitpack_sint_unpack(packed, 0, 18);
   addr += (int32_t)util_dynarray_num_elements(&p->instrs, pdsc_instr);
   assert(addr >= 0);
   uint16_t target_instr = addr;

   assert(opcodec == PDSC_OPCODEC_BRA);
   return pdsc_bra(p, target_instr, .srcc = srcc, .setc = setc, .neg = neg);
}

static inline uint32_t pdsc_ld64_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_LD, 23, 26);

   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 0, 6);

   return packed;
}

static inline pdsc_instr_t pdsc_ld64_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 0, 6);

   pdsc_ref src0 = pdsc_regs64_unpack(src0_packed);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_LD);
   return pdsc_ld64(p, src0, .cc = cc);
}

static inline uint32_t pdsc_st32_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_ST, 23, 26);

   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 0, 6);

   return packed;
}

static inline pdsc_instr_t pdsc_st32_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 0, 6);

   pdsc_ref src0 = pdsc_regs64_unpack(src0_packed);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_ST);
   return pdsc_st32(p, src0, .cc = cc);
}

static inline uint32_t pdsc_idiv_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_IDIV, 23, 26);

   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[0]), 14, 21);
   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[1]), 6, 13);
   packed |= util_bitpack_uint(pdsc_regs32tp_pack(_i->dst), 0, 5);

   return packed;
}

static inline pdsc_instr_t pdsc_idiv_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 14, 21);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 6, 13);
   uint8_t dst_packed = util_bitpack_uint_unpack(packed, 0, 5);

   pdsc_ref src0 = pdsc_regs32_unpack(src0_packed);
   pdsc_ref src1 = pdsc_regs32_unpack(src1_packed);
   pdsc_ref dst = pdsc_regs32tp_unpack(dst_packed);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_IDIV);
   return pdsc_idiv(p, dst, src0, src1, .cc = cc);
}

static inline uint32_t pdsc_stmp_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_tst(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_STMP, 23, 26);

   packed |= util_bitpack_uint(!!pdsc_get_cc_global_ovf(p, i), 22, 22);
   packed |= util_bitpack_uint(!!pdsc_get_cc_so_ovf(p, i), 21, 21);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 20, 20);

   packed |= util_bitpack_uint(pdsc_get_so(p, i), 18, 19);

   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 11, 17);
   packed |= util_bitpack_uint(pdsc_regs64t_pack(_i->src[1]), 7, 10);
   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[2]), 0, 6);

   return packed;
}

static inline pdsc_instr_t pdsc_stmp_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool tst = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   bool cc_global_ovf = util_bitpack_uint_unpack(packed, 22, 22);
   bool cc_so_ovf = util_bitpack_uint_unpack(packed, 21, 21);
   bool cc = util_bitpack_uint_unpack(packed, 20, 20);

   uint8_t so = util_bitpack_uint_unpack(packed, 18, 19);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 11, 17);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 7, 10);
   uint8_t src2_packed = util_bitpack_uint_unpack(packed, 0, 6);

   pdsc_ref src0 = pdsc_regs64_unpack(src0_packed);
   pdsc_ref src1 = pdsc_regs64t_unpack(src1_packed);
   pdsc_ref src2 = pdsc_regs64_unpack(src2_packed);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_STMP);
   return pdsc_stmp(p,
                    src0,
                    src1,
                    src2,
                    .cc = cc,
                    .cc_so_ovf = cc_so_ovf,
                    .cc_global_ovf = cc_global_ovf,
                    .tst = tst,
                    .so = so);
}

static inline uint32_t pdsc_aa_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_AA, 23, 26);

   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[1]), 7, 14);
   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 0, 6);

   return packed;
}

static inline pdsc_instr_t pdsc_aa_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 7, 14);
   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 0, 6);

   pdsc_ref src1 = pdsc_regs32_unpack(src1_packed);
   pdsc_ref src0 = pdsc_regs64_unpack(src0_packed);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_AA);
   return pdsc_aa(p, src0, src1, .cc = cc);
}

static inline uint32_t pdsc_wdf_pack(const pdsc_program *p, pdsc_instr_t i)
{
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_WDF, 23, 26);

   return packed;
}

static inline pdsc_instr_t pdsc_wdf_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_WDF);
   return pdsc_wdf(p, .cc = cc);
}

static inline uint32_t pdsc_idf_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_IDF, 23, 26);

   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 0, 6);

   return packed;
}

static inline pdsc_instr_t pdsc_idf_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 0, 6);

   pdsc_ref src0 = pdsc_regs64_unpack(src0_packed);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_IDF);
   return pdsc_idf(p, src0, .cc = cc);
}

static inline uint32_t pdsc_pol_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_POL, 23, 26);

   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[1]), 7, 14);
   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 0, 6);

   return packed;
}

static inline pdsc_instr_t pdsc_pol_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 7, 14);
   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 0, 6);

   pdsc_ref src1 = pdsc_regs32_unpack(src1_packed);
   pdsc_ref src0 = pdsc_regs64_unpack(src0_packed);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_POL);
   return pdsc_pol(p, src0, src1, .cc = cc);
}

static inline uint32_t pdsc_limm_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_LIMM, 23, 26);

   packed |= util_bitpack_uint(pdsc_regs32t_pack(_i->dst), 18, 22);

   packed |= util_bitpack_uint(_i->src[0].val, 2, 17);

   assert(pdsc_ref_is_imm(_i->src[0]) || pdsc_ref_is_global(_i->src[0]));

   bool gr = pdsc_ref_is_global(_i->src[0]);
   packed |= util_bitpack_uint(gr, 1, 1);

   return packed;
}

static inline pdsc_instr_t pdsc_limm_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   uint8_t dst_packed = util_bitpack_uint_unpack(packed, 18, 22);
   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 2, 17);

   bool gr = util_bitpack_uint_unpack(packed, 1, 1);

   pdsc_ref dst = pdsc_regs32t_unpack(dst_packed);
   pdsc_ref src0 = gr ? pdsc_ref_global(src0_packed)
                      : pdsc_ref_imm(src0_packed);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_LIMM);
   return pdsc_limm(p, dst, src0, .cc = cc);
}

static inline uint32_t pdsc_lock_pack(const pdsc_program *p, pdsc_instr_t i)
{
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_LOCK, 23, 26);

   return packed;
}

static inline pdsc_instr_t pdsc_lock_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_LOCK);
   return pdsc_lock(p, .cc = cc);
}

static inline uint32_t pdsc_release_pack(const pdsc_program *p, pdsc_instr_t i)
{
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_RELEASE, 23, 26);

   return packed;
}

static inline pdsc_instr_t pdsc_release_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_RELEASE);
   return pdsc_release(p, .cc = cc);
}

static inline uint32_t pdsc_halt_pack(const pdsc_program *p, pdsc_instr_t i)
{
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_HALT, 23, 26);

   return packed;
}

static inline pdsc_instr_t pdsc_halt_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_HALT);
   return pdsc_halt(p, .cc = cc);
}

static inline uint32_t pdsc_nop_pack(const pdsc_program *p, pdsc_instr_t i)
{
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_NOP, 23, 26);

   return packed;
}

static inline pdsc_instr_t pdsc_nop_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_NOP);
   return pdsc_nop(p, .cc = cc);
}

static inline uint32_t pdsc_stmc_pack(const pdsc_program *p, pdsc_instr_t i)
{
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_SP, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(PDSC_OPCODESP_STMC, 23, 26);

   packed |= util_bitpack_uint(pdsc_get_somask(p, i), 0, 4);

   return packed;
}

static inline pdsc_instr_t pdsc_stmc_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);

   ASSERTED enum pdsc_opcodesp opcodesp =
      util_bitpack_uint_unpack(packed, 23, 26);

   enum pdsc_somask somask = util_bitpack_uint_unpack(packed, 0, 4);

   assert(opcodec == PDSC_OPCODEC_SP);
   assert(opcodesp == PDSC_OPCODESP_STMC);
   return pdsc_stmc(p, .cc = cc, .somask = somask);
}

static inline uint32_t pdsc_ddmad_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_DDMAD, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(!!pdsc_get_end(p, i), 26, 26);

   packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[0]), 18, 25);
   packed |= util_bitpack_uint(pdsc_regs32t_pack(_i->src[1]), 13, 17);
   packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[2]), 6, 12);
   packed |= util_bitpack_uint(pdsc_regs64c_pack(_i->src[3]), 0, 5);

   return packed;
}

static inline pdsc_instr_t pdsc_ddmad_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);
   bool end = util_bitpack_uint_unpack(packed, 26, 26);

   uint8_t src0_packed = util_bitpack_uint_unpack(packed, 18, 25);
   uint8_t src1_packed = util_bitpack_uint_unpack(packed, 13, 17);
   uint8_t src2_packed = util_bitpack_uint_unpack(packed, 6, 12);
   uint8_t src3_packed = util_bitpack_uint_unpack(packed, 0, 5);

   pdsc_ref src0 = pdsc_regs32_unpack(src0_packed);
   pdsc_ref src1 = pdsc_regs32t_unpack(src1_packed);
   pdsc_ref src2 = pdsc_regs64_unpack(src2_packed);
   pdsc_ref src3 = pdsc_regs64_unpack(src3_packed);

   assert(opcodec == PDSC_OPCODEC_DDMAD);
   return pdsc_ddmad(p, src0, src1, src2, src3, .cc = cc, .end = end);
}

static inline uint32_t pdsc_dout_pack(const pdsc_program *p, pdsc_instr_t i)
{
   const pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);
   uint32_t packed = 0;

   packed |= util_bitpack_uint(PDSC_OPCODEC_DOUT, 28, 31);
   packed |= util_bitpack_uint(!!pdsc_get_cc(p, i), 27, 27);
   packed |= util_bitpack_uint(!!pdsc_get_end(p, i), 26, 26);

   if (PDSC_HAS(p, DOUT_EXT)) {
      packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[2]), 18, 24);
      packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[1]), 10, 17);
      packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 3, 9);
   } else {
      packed |= util_bitpack_uint(pdsc_regs32_pack(_i->src[1]), 16, 23);
      packed |= util_bitpack_uint(pdsc_regs64_pack(_i->src[0]), 8, 14);
   }

   packed |= util_bitpack_uint(pdsc_get_dstdout(p, i), 0, 2);

   return packed;
}

static inline pdsc_instr_t pdsc_dout_unpack(pdsc_program *p, uint32_t packed)
{
   ASSERTED enum pdsc_opcodec opcodec =
      util_bitpack_uint_unpack(packed, 28, 31);

   bool cc = util_bitpack_uint_unpack(packed, 27, 27);
   bool end = util_bitpack_uint_unpack(packed, 26, 26);

   uint8_t src2_packed;
   uint8_t src1_packed;
   uint8_t src0_packed;

   if (PDSC_HAS(p, DOUT_EXT)) {
      src2_packed = util_bitpack_uint_unpack(packed, 18, 24);
      src1_packed = util_bitpack_uint_unpack(packed, 10, 17);
      src0_packed = util_bitpack_uint_unpack(packed, 3, 9);
   } else {
      src1_packed = util_bitpack_uint_unpack(packed, 16, 23);
      src0_packed = util_bitpack_uint_unpack(packed, 8, 14);
   }

   enum pdsc_dstdout dstdout = util_bitpack_uint_unpack(packed, 0, 2);

   pdsc_ref src2 = PDSC_HAS(p, DOUT_EXT) ? pdsc_regs64_unpack(src2_packed)
                                         : pdsc_ref_null();
   pdsc_ref src1 = pdsc_regs32_unpack(src1_packed);
   pdsc_ref src0 = pdsc_regs64_unpack(src0_packed);

   assert(opcodec == PDSC_OPCODEC_DOUT);
   return pdsc_dout(p,
                    src0,
                    src1,
                    src2,
                    .cc = cc,
                    .end = end,
                    .dstdout = dstdout);
}

static inline uint32_t
pdsc_instr_pack(pdsc_program *p, pdsc_instr_t i, enum pdsc_op op)
{
   switch (op) {
   case PDSC_OP_SFTLP64:
      return pdsc_sftlp64_pack(p, i);

   case PDSC_OP_SFTLP32:
      return pdsc_sftlp32_pack(p, i);

   case PDSC_OP_STM:
      return pdsc_stm_pack(p, i);

   case PDSC_OP_MAD:
      return pdsc_mad_pack(p, i);

   case PDSC_OP_ADD64:
      return pdsc_add64_pack(p, i);

   case PDSC_OP_ADD32:
      return pdsc_add32_pack(p, i);

   case PDSC_OP_CMP:
      return pdsc_cmp_pack(p, i);

   case PDSC_OP_BRA:
      return pdsc_bra_pack(p, i);

   case PDSC_OP_LD64:
      return pdsc_ld64_pack(p, i);

   case PDSC_OP_ST32:
      return pdsc_st32_pack(p, i);

   case PDSC_OP_IDIV:
      return pdsc_idiv_pack(p, i);

   case PDSC_OP_STMP:
      return pdsc_stmp_pack(p, i);

   case PDSC_OP_AA:
      return pdsc_aa_pack(p, i);

   case PDSC_OP_WDF:
      return pdsc_wdf_pack(p, i);

   case PDSC_OP_IDF:
      return pdsc_idf_pack(p, i);

   case PDSC_OP_POL:
      return pdsc_pol_pack(p, i);

   case PDSC_OP_LIMM:
      return pdsc_limm_pack(p, i);

   case PDSC_OP_LOCK:
      return pdsc_lock_pack(p, i);

   case PDSC_OP_RELEASE:
      return pdsc_release_pack(p, i);

   case PDSC_OP_HALT:
      return pdsc_halt_pack(p, i);

   case PDSC_OP_NOP:
      return pdsc_nop_pack(p, i);

   case PDSC_OP_STMC:
      return pdsc_stmc_pack(p, i);

   case PDSC_OP_DDMAD:
      return pdsc_ddmad_pack(p, i);

   case PDSC_OP_DOUT:
      return pdsc_dout_pack(p, i);

   default:
      break;
   }

   UNREACHABLE("Invalid op.");
}

static inline enum pdsc_op pdsc_op_decode(uint32_t packed)
{
   enum pdsc_opcodec opcodec = util_bitpack_uint_unpack(packed, 28, 31);
   enum pdsc_opcodesp opcodesp = util_bitpack_uint_unpack(packed, 23, 26);

   switch (opcodec) {
   case PDSC_OPCODEC_ADD64:
      return PDSC_OP_ADD64;
   case PDSC_OPCODEC_ADD32:
      return PDSC_OP_ADD32;
   case PDSC_OPCODEC_SFTLP64:
      return PDSC_OP_SFTLP64;
   case PDSC_OPCODEC_CMP:
      return PDSC_OP_CMP;
   case PDSC_OPCODEC_BRA:
      return PDSC_OP_BRA;
   case PDSC_OPCODEC_SP:
      switch (opcodesp) {
      case PDSC_OPCODESP_LD:
         return PDSC_OP_LD64;
      case PDSC_OPCODESP_ST:
         return PDSC_OP_ST32;
      case PDSC_OPCODESP_WDF:
         return PDSC_OP_WDF;
      case PDSC_OPCODESP_LIMM:
         return PDSC_OP_LIMM;
      case PDSC_OPCODESP_LOCK:
         return PDSC_OP_LOCK;
      case PDSC_OPCODESP_RELEASE:
         return PDSC_OP_RELEASE;
      case PDSC_OPCODESP_HALT:
         return PDSC_OP_HALT;
      case PDSC_OPCODESP_STMC:
         return PDSC_OP_STMC;
      case PDSC_OPCODESP_STMP:
         return PDSC_OP_STMP;
      case PDSC_OPCODESP_IDIV:
         return PDSC_OP_IDIV;
      case PDSC_OPCODESP_AA:
         return PDSC_OP_AA;
      case PDSC_OPCODESP_IDF:
         return PDSC_OP_IDF;
      case PDSC_OPCODESP_POL:
         return PDSC_OP_POL;
      case PDSC_OPCODESP_NOP:
         return PDSC_OP_NOP;
      default:
         break;
      }
      UNREACHABLE("Invalid sp op.");
   case PDSC_OPCODEC_DDMAD:
      return PDSC_OP_DDMAD;
   case PDSC_OPCODEC_DOUT:
      return PDSC_OP_DOUT;
   default:
      break;
   }

   enum pdsc_opcodeb opcodeb = util_bitpack_uint_unpack(packed, 29, 31);
   switch (opcodeb) {
   case PDSC_OPCODEB_SFTLP32:
      return PDSC_OP_SFTLP32;
   case PDSC_OPCODEB_STM:
      return PDSC_OP_STM;
   default:
      break;
   }

   enum pdsc_opcodea opcodea = util_bitpack_uint_unpack(packed, 30, 31);
   switch (opcodea) {
   case PDSC_OPCODEA_MAD:
      return PDSC_OP_MAD;
   default:
      break;
   }

   UNREACHABLE("Invalid op.");
}

static inline pdsc_instr_t pdsc_instr_unpack(pdsc_program *p, uint32_t packed)
{
   enum pdsc_op op = pdsc_op_decode(packed);

   switch (op) {
   case PDSC_OP_SFTLP64:
      return pdsc_sftlp64_unpack(p, packed);

   case PDSC_OP_SFTLP32:
      return pdsc_sftlp32_unpack(p, packed);

   case PDSC_OP_STM:
      return pdsc_stm_unpack(p, packed);

   case PDSC_OP_MAD:
      return pdsc_mad_unpack(p, packed);

   case PDSC_OP_ADD64:
      return pdsc_add64_unpack(p, packed);

   case PDSC_OP_ADD32:
      return pdsc_add32_unpack(p, packed);

   case PDSC_OP_CMP:
      return pdsc_cmp_unpack(p, packed);

   case PDSC_OP_BRA:
      return pdsc_bra_unpack(p, packed);

   case PDSC_OP_LD64:
      return pdsc_ld64_unpack(p, packed);

   case PDSC_OP_ST32:
      return pdsc_st32_unpack(p, packed);

   case PDSC_OP_IDIV:
      return pdsc_idiv_unpack(p, packed);

   case PDSC_OP_STMP:
      return pdsc_stmp_unpack(p, packed);

   case PDSC_OP_AA:
      return pdsc_aa_unpack(p, packed);

   case PDSC_OP_WDF:
      return pdsc_wdf_unpack(p, packed);

   case PDSC_OP_IDF:
      return pdsc_idf_unpack(p, packed);

   case PDSC_OP_POL:
      return pdsc_pol_unpack(p, packed);

   case PDSC_OP_LIMM:
      return pdsc_limm_unpack(p, packed);

   case PDSC_OP_LOCK:
      return pdsc_lock_unpack(p, packed);

   case PDSC_OP_RELEASE:
      return pdsc_release_unpack(p, packed);

   case PDSC_OP_HALT:
      return pdsc_halt_unpack(p, packed);

   case PDSC_OP_NOP:
      return pdsc_nop_unpack(p, packed);

   case PDSC_OP_STMC:
      return pdsc_stmc_unpack(p, packed);

   case PDSC_OP_DDMAD:
      return pdsc_ddmad_unpack(p, packed);

   case PDSC_OP_DOUT:
      return pdsc_dout_unpack(p, packed);

   default:
      break;
   }

   UNREACHABLE("Invalid op.");
}

#endif /* PDSC_ISA_PACK_H */
