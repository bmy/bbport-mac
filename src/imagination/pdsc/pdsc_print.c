/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * \file pdsc_print.c
 *
 * \brief PDS compiler print functions.
 */

#include "pdsc.h"
#include "pdsc_internal.h"
#include "pdsc_isa_pack.h"

#include "util/bitset.h"
#include "util/compiler.h"
#include "util/macros.h"
#include "util/u_hexdump.h"

#include <assert.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

enum color_esc {
   ESC_RESET = 0,
   ESC_BLACK,
   ESC_RED,
   ESC_GREEN,
   ESC_YELLOW,
   ESC_BLUE,
   ESC_PURPLE,
   ESC_CYAN,
   ESC_WHITE,
   _ESC_COUNT,
};

static
const char *color_esc[2][_ESC_COUNT] = {
   [0] = {
      [ESC_RESET] = "",
      [ESC_BLACK] = "",
      [ESC_RED] = "",
      [ESC_GREEN] = "",
      [ESC_YELLOW] = "",
      [ESC_BLUE] = "",
      [ESC_PURPLE] = "",
      [ESC_CYAN] = "",
      [ESC_WHITE] = "",
   },
   [1] = {
      [ESC_RESET] = "\033[0m",
      [ESC_BLACK] = "\033[0;30m",
      [ESC_RED] = "\033[0;31m",
      [ESC_GREEN] = "\033[0;32m",
      [ESC_YELLOW] = "\033[0;33m",
      [ESC_BLUE] = "\033[0;34m",
      [ESC_PURPLE] = "\033[0;35m",
      [ESC_CYAN] = "\033[0;36m",
      [ESC_WHITE] = "\033[0;37m",
   },
};

static inline void RESET(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_RESET], fp);
}

static inline void BLACK(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_BLACK], fp);
}

static inline void RED(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_RED], fp);
}

static inline void GREEN(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_GREEN], fp);
}

static inline void YELLOW(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_YELLOW], fp);
}

static inline void BLUE(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_BLUE], fp);
}

static inline void PURPLE(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_PURPLE], fp);
}

static inline void CYAN(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_CYAN], fp);
}

static inline void WHITE(FILE *fp)
{
   fputs(color_esc[pdsc_color][ESC_WHITE], fp);
}

/* TODO NEXT: drop asserts, just have in validation. */
static void
pdsc_print_type(FILE *fp, enum pdsc_type type, ASSERTED bool is_64bit)
{
   CYAN(fp);
   switch (type) {
   case PDSC_TYPE_IMM:
      fputs("imm", fp);
      fputs(is_64bit ? "64" : "32", fp);
      break;
   case PDSC_TYPE_STM_SRC0:
      assert(is_64bit);
      fputs("stm_src0", fp);
      break;
   case PDSC_TYPE_STM_SRC1:
      assert(is_64bit);
      fputs("stm_src1", fp);
      break;
   case PDSC_TYPE_STM_SRC2:
      assert(!is_64bit);
      fputs("stm_src2", fp);
      break;
   case PDSC_TYPE_STM_SRC3:
      assert(is_64bit);
      fputs("stm_src3", fp);
      break;
   case PDSC_TYPE_LD64_SRC0:
      assert(is_64bit);
      fputs("ld64_src0", fp);
      break;
   case PDSC_TYPE_ST32_SRC0:
      assert(is_64bit);
      fputs("st32_src0", fp);
      break;
   case PDSC_TYPE_STMP_SRC0:
      assert(is_64bit);
      fputs("stmp_src0", fp);
      break;
   case PDSC_TYPE_STMP_SRC1:
      assert(is_64bit);
      fputs("stmp_src1", fp);
      break;
   case PDSC_TYPE_STMP_SRC2:
      assert(is_64bit);
      fputs("stmp_src2", fp);
      break;
   case PDSC_TYPE_AA_SRC0:
      assert(is_64bit);
      fputs("aa_src0", fp);
      break;
   case PDSC_TYPE_IDF_SRC0:
      assert(is_64bit);
      fputs("idf_src0", fp);
      break;
   case PDSC_TYPE_POL_SRC0:
      assert(is_64bit);
      fputs("pol_src0", fp);
      break;
   case PDSC_TYPE_DDMAD_SRC3:
      assert(is_64bit);
      fputs("ddmad_src3", fp);
      break;
   case PDSC_TYPE_DOUTU_SRC0:
      assert(is_64bit);
      fputs("doutu_src0", fp);
      break;
   case PDSC_TYPE_DOUTD_SRC0:
      assert(is_64bit);
      fputs("doutd_src0", fp);
      break;
   case PDSC_TYPE_DOUTD_SRC1:
      assert(!is_64bit);
      fputs("doutd_src1", fp);
      break;
   case PDSC_TYPE_DOUTW_SRC1:
      assert(!is_64bit);
      fputs("doutw_src1", fp);
      break;
   case PDSC_TYPE_DOUTI_SRC0:
      assert(is_64bit);
      fputs("douti_src0", fp);
      break;
   default:
      UNREACHABLE("Unexpected type.");
   }

   fputs("_t", fp);
   RESET(fp);
}

static void pdsc_print_ref_basic(FILE *fp, const pdsc_program *p, pdsc_ref ref)
{
   /* Don't need channel offset for elements. */
   if (ref.is_elem32 || !ref.is_array_elem)
      ref.chans = 0;

   switch (ref.type) {
   case PDSC_REF_TYPE_NULL:
      return;

   case PDSC_REF_TYPE_IMM:
      BLUE(fp);
      fprintf(fp, "%" PRIu16, pdsc_ref_get_val(ref));
      RESET(fp);
      return;

   case PDSC_REF_TYPE_SIMM:
      BLUE(fp);
      fprintf(fp, "%" PRId16, pdsc_ref_get_sval(ref));
      RESET(fp);
      return;

   case PDSC_REF_TYPE_CONST_REG:
      fputs("c", fp);
      break;

   case PDSC_REF_TYPE_TEMP_REG:
      fputs("t", fp);
      break;

   case PDSC_REF_TYPE_PTEMP_REG:
      fputs("pt", fp);
      break;

   case PDSC_REF_TYPE_GLOBAL_REG:
      fputs("g", fp);
      break;

   default:
      UNREACHABLE("Invalid ref type.");
   }

   fprintf(fp, "[%u", pdsc_ref_get_val(ref));

   if (!pdsc_ref_is_scalar(ref) || pdsc_ref_is_64bit(ref)) {
      uint8_t n = pdsc_ref_is_64bit(ref) ? 2 : 1;
      uint8_t end = pdsc_ref_get_val(ref) + n * pdsc_ref_get_chans(ref) - 1;
      fprintf(fp, "..%u", end);
   }

   fputs("]", fp);
}

static inline bool
pdsc_print_ref_named(FILE *fp, const pdsc_program *p, pdsc_ref ref)
{
   uint8_t elem = 0;
   bool hi32 = false;

   /* Lookup base reference if this is an element. */
   if (ref.is_elem32) {
      hi32 = !!(ref.val & 1);
      ref.val &= ~0b1;
   }

   if (ref.is_array_elem) {
      elem = ref.chans;
      if (ref.is_elem32 || pdsc_ref_is_64bit(ref)) {
         assert(!(elem & 1));
         elem /= 2;
      }

      ref.val -= ref.chans;
      ref.chans = 0;
   }

   const char *name = NULL;
   switch (ref.type) {
   case PDSC_REF_TYPE_NULL:
   case PDSC_REF_TYPE_IMM:
   case PDSC_REF_TYPE_SIMM:
   /* TODO: support global reg names?. */
   case PDSC_REF_TYPE_GLOBAL_REG:
      break;

   case PDSC_REF_TYPE_CONST_REG:
      name =
         pdsc_get_const_name(p, pdsc_ref_get_val(ref), pdsc_ref_get_alias(ref));
      break;

   case PDSC_REF_TYPE_TEMP_REG:
      name =
         pdsc_get_temp_name(p, pdsc_ref_get_val(ref), pdsc_ref_get_alias(ref));
      break;

   case PDSC_REF_TYPE_PTEMP_REG:
      name =
         pdsc_get_ptemp_name(p, pdsc_ref_get_val(ref), pdsc_ref_get_alias(ref));
      break;

   default:
      UNREACHABLE("Invalid ref type.");
   }

   if (!name)
      return false;

   YELLOW(fp);
   fprintf(fp, "%s", name);
   RESET(fp);

   if (ref.is_array_elem)
      fprintf(fp, "[%" PRIu8 "]", elem);

   if (ref.is_elem32)
      fprintf(fp, ".%s32", hi32 ? "hi" : "lo");

   return true;
}

static void pdsc_print_ref(FILE *fp, const pdsc_program *p, pdsc_ref ref)
{
   /* Special case: ref made up of two refs. */
   if (pdsc_ref_is_composite64(ref)) {
      fputs("{", fp);

      ref.is_composite64 = false;
      ref.is_64bit = false;
      pdsc_print_ref(fp, p, ref);
      fputs(",", fp);

      ++ref.val;
      pdsc_print_ref(fp, p, ref);

      fputs("}", fp);
      return;
   }

   bool has_named = pdsc_print_ref_named(fp, p, ref);

   if (has_named)
      fputs(" (", fp);

   pdsc_print_ref_basic(fp, p, ref);

   if (has_named)
      fputs(")", fp);
}

static const char *pdsc_dout_store_dest_str(enum pdsc_store_dest store_dest)
{
   switch (store_dest) {
   case PDSC_STORE_DEST_UNIFIED:
      return "vi";
   case PDSC_STORE_DEST_COMMON:
      return "cf/sh";
   default:
      break;
   }

   UNREACHABLE("Invalid dout store_dest.");
}

static const char *pdsc_slc_mode_ld_str(enum pdsc_slc_mode_ld slcmode_ld)
{
   switch (slcmode_ld) {
   case PDSC_SLC_MODE_LD_BYPASS:
      return "bypass";
   case PDSC_SLC_MODE_LD_CACHED:
      return "cached";
   case PDSC_SLC_MODE_LD_CACHED_RD_NA:
      return "cached_rd_na";
   default:
      break;
   }

   UNREACHABLE("Invalid slc_mode_ld.");
}

static const char *pdsc_slc_mode_st_str(enum pdsc_slc_mode_st slcmode_st)
{
   switch (slcmode_st) {
   case PDSC_SLC_MODE_ST_WRITE_THROUGH:
      return "write_through";
   case PDSC_SLC_MODE_ST_WRITE_BACK:
      return "write_back";
   default:
      break;
   }

   UNREACHABLE("Invalid slc_mode_st.");
}

static const char *pdsc_ccslc_ld_str(enum pdsc_ccslc_ld ccslc_ld)
{
   switch (ccslc_ld) {
   case PDSC_CCSLC_LD_NORMAL:
      return "normal";
   case PDSC_CCSLC_LD_FORCE_LINE_FILL:
      return "force_line_fill";
   default:
      break;
   }

   UNREACHABLE("Invalid ccslc_ld.");
}

static const char *pdsc_ccslc_st_str(enum pdsc_ccslc_st ccslc_st)
{
   switch (ccslc_st) {
   case PDSC_CCSLC_ST_LAZY_WRITE_BACK:
      return "lazy_write_back";
   case PDSC_CCSLC_ST_WRITE_THROUGH:
      return "write_through";
   default:
      break;
   }

   UNREACHABLE("Invalid ccslc_st.");
}

static const char *pdsc_ccslc_idf_str(enum pdsc_ccslc_idf ccslc_idf)
{
   switch (ccslc_idf) {
   case PDSC_CCSLC_IDF_FENCE_IN_SLC:
      return "fence_in_slc";
   case PDSC_CCSLC_IDF_FENCE_TO_MEM:
      return "fence_to_mem";
   default:
      break;
   }

   UNREACHABLE("Invalid ccslc_idf.");
}

static const char *pdsc_ccmcu_ld_str(enum pdsc_ccmcu_ld ccmcu_ld)
{
   switch (ccmcu_ld) {
   case PDSC_CCMCU_LD_NORMAL:
      return "normal";
   case PDSC_CCMCU_LD_FORCE_LINE_FILL:
      return "force_line_fill";
   default:
      break;
   }

   UNREACHABLE("Invalid ccmcu_ld.");
}

static const char *pdsc_ccmcu_st_str(enum pdsc_ccmcu_st ccmcu_st)
{
   switch (ccmcu_st) {
   case PDSC_CCMCU_ST_LAZY_WRITE_BACK:
      return "lazy_write_back";
   case PDSC_CCMCU_ST_WRITE_THROUGH:
      return "write_through";
   default:
      break;
   }

   UNREACHABLE("Invalid ccmcu_st.");
}

static const char *pdsc_ccmcu_idf_str(enum pdsc_ccmcu_idf ccmcu_idf)
{
   switch (ccmcu_idf) {
   case PDSC_CCMCU_IDF_FENCE_IN_MCU:
      return "fence_in_mcu";
   case PDSC_CCMCU_IDF_FENCE_TO_SLC:
      return "fence_to_slc";
   default:
      break;
   }

   UNREACHABLE("Invalid ccmcu_st.");
}

static const char *
pdsc_doutu_sample_rate_str(enum pdsc_doutu_sample_rate sample_rate)
{
   switch (sample_rate) {
   case PDSC_DOUTU_SAMPLE_RATE_INSTANCE:
      return "instance";
   case PDSC_DOUTU_SAMPLE_RATE_SELECTIVE:
      return "selective";
   case PDSC_DOUTU_SAMPLE_RATE_FULL:
      return "full";
   default:
      break;
   }

   UNREACHABLE("Invalid sample_rate.");
}

static const char *pdsc_cmode_ld_str(enum pdsc_cmode_ld cmode_ld)
{
   switch (cmode_ld) {
   case PDSC_CMODE_LD_CACHED:
      return "cached";
   case PDSC_CMODE_LD_BYPASS:
      return "bypass";
   case PDSC_CMODE_LD_FORCE_LINE_FILL:
      return "force_line_fill";
   default:
      break;
   }

   UNREACHABLE("Invalid cmode_ld.");
}

static const char *pdsc_cmode_st_str(enum pdsc_cmode_st cmode_st)
{
   switch (cmode_st) {
   case PDSC_CMODE_ST_WRITE_THROUGH:
      return "write_through";
   case PDSC_CMODE_ST_WRITE_BACK:
      return "write_back";
   case PDSC_CMODE_ST_LAZY_WRITE_BACK:
      return "lazy_write_back";
   default:
      break;
   }

   UNREACHABLE("Invalid cmode_st.");
}

static const char *pdsc_primtype_str(enum pdsc_primtype primtype)
{
   switch (primtype) {
   case PDSC_PRIMTYPE_POINT:
      return "point";
   case PDSC_PRIMTYPE_LINE:
      return "line";
   case PDSC_PRIMTYPE_TRIANGLE:
      return "triangle";
   default:
      break;
   }

   UNREACHABLE("Invalid primtype.");
}

static const char *pdsc_aop_str(enum pdsc_aop aop)
{
   switch (aop) {
   case PDSC_AOP_ADD:
      return "add";
   case PDSC_AOP_SUB:
      return "sub";
   case PDSC_AOP_XCHG:
      return "xchg";
   case PDSC_AOP_UMIN:
      return "umin";
   case PDSC_AOP_IMIN:
      return "min";
   case PDSC_AOP_UMAX:
      return "umax";
   case PDSC_AOP_IMAX:
      return "imax";
   case PDSC_AOP_AND:
      return "and";
   case PDSC_AOP_OR:
      return "or";
   case PDSC_AOP_XOR:
      return "xor";
   default:
      break;
   }

   UNREACHABLE("Invalid aop.");
}

static void pdsc_print_stm_src0(FILE *fp,
                                const pdsc_program *p,
                                const struct pdsc_stm_src0 *src0)
{
   fputs("{\n", fp);

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS)) {
      PURPLE(fp);
      fputs("\tslcmode", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_slc_mode_st_str(src0->slcmode));
   }

   PURPLE(fp);
   fputs("\tptemp_prim_needed_written", fp);
   RESET(fp);
   fprintf(fp, ": ");
   pdsc_print_ref(fp, p, src0->ptemp_prim_needed_written);
   fputs("\n", fp);

   PURPLE(fp);
   fputs("\tptemp_address", fp);
   RESET(fp);
   fprintf(fp, ": ");
   pdsc_print_ref(fp, p, src0->ptemp_address);
   fputs("\n", fp);

   PURPLE(fp);
   fputs("\taddress", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src0->address);

   fputs("}", fp);
}

static void pdsc_print_stm_src1(FILE *fp, const struct pdsc_stm_src1 *src1)
{
   fputs("{\n", fp);

   PURPLE(fp);
   fputs("\tprim_needed", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu32 "\n", src1->prim_needed);

   PURPLE(fp);
   fputs("\tprim_written", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu32 "\n", src1->prim_written);

   fputs("}", fp);
}

static void pdsc_print_stm_src2(FILE *fp, const struct pdsc_stm_src2 *src2)
{
   fputs("{\n", fp);

   PURPLE(fp);
   fputs("\tvioff", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu8 "\n", src2->vioff);

   PURPLE(fp);
   fputs("\tdmasize", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu8 "\n", src2->dmasize);

   PURPLE(fp);
   fputs("\tvooff", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu16 "\n", src2->vooff);

   fputs("}", fp);
}

static void pdsc_print_stm_src3(FILE *fp, const struct pdsc_stm_src3 *src3)
{
   fputs("{\n", fp);

   PURPLE(fp);
   fputs("\tprimtype", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_primtype_str(src3->primtype));

   PURPLE(fp);
   fputs("\tvosize", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu16 "\n", src3->vosize);

   if (src3->eop) {
      RED(fp);
      fputs("\teop\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tlimit", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src3->limit);

   fputs("}", fp);
}

static void pdsc_print_ld64_src0(FILE *fp,
                                 const pdsc_program *p,
                                 const struct pdsc_ld64_src0 *src0)
{
   fputs("{\n", fp);

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS)) {
      PURPLE(fp);
      fputs("\tslcmode", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_slc_mode_ld_str(src0->slcmode));
   }

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      PURPLE(fp);
      fputs("\tccpri", fp);
      RESET(fp);
      fprintf(fp, ": %" PRIu8 "\n", src0->ccpri);

      PURPLE(fp);
      fputs("\tccslc", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccslc_ld_str(src0->ccslc));

      PURPLE(fp);
      fputs("\tccmcu", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccmcu_ld_str(src0->ccmcu));
   }

   PURPLE(fp);
   fputs("\tdest", fp);
   RESET(fp);
   fprintf(fp, ": ");
   pdsc_print_ref(fp, p, src0->dest);
   fputs("\n", fp);

   PURPLE(fp);
   fputs("\tcmode", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_cmode_ld_str(src0->cmode));

   PURPLE(fp);
   fputs("\tcount8", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu8 "\n", src0->count8);

   PURPLE(fp);
   fputs("\tsrcadd", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src0->srcadd);

   fputs("}", fp);
}

static void pdsc_print_st32_src0(FILE *fp,
                                 const pdsc_program *p,
                                 const struct pdsc_st32_src0 *src0)
{
   fputs("{\n", fp);

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS)) {
      PURPLE(fp);
      fputs("\tslcmode", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_slc_mode_st_str(src0->slcmode));
   }

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      PURPLE(fp);
      fputs("\tccpri", fp);
      RESET(fp);
      fprintf(fp, ": %" PRIu8 "\n", src0->ccpri);

      PURPLE(fp);
      fputs("\tccslc", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccslc_st_str(src0->ccslc));

      PURPLE(fp);
      fputs("\tccmcu", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccmcu_st_str(src0->ccmcu));
   }

   PURPLE(fp);
   fputs("\tsrc", fp);
   RESET(fp);
   fprintf(fp, ": ");
   pdsc_print_ref(fp, p, src0->src);
   fputs("\n", fp);

   PURPLE(fp);
   fputs("\tcmode", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_cmode_st_str(src0->cmode));

   PURPLE(fp);
   fputs("\tcount4", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu8 "\n", src0->count4);

   PURPLE(fp);
   fputs("\tdstadd", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src0->dstadd);

   fputs("}", fp);
}

static void pdsc_print_stmp_src0(FILE *fp,
                                 const pdsc_program *p,
                                 const struct pdsc_stmp_src0 *src0)
{
   fputs("{\n", fp);

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS)) {
      PURPLE(fp);
      fputs("\tslcmode", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_slc_mode_st_str(src0->slcmode));
   }

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      PURPLE(fp);
      fputs("\tccpri", fp);
      RESET(fp);
      fprintf(fp, ": %" PRIu8 "\n", src0->ccpri);

      PURPLE(fp);
      fputs("\tccslc", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccslc_st_str(src0->ccslc));

      PURPLE(fp);
      fputs("\tccmcu", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccmcu_st_str(src0->ccmcu));
   }

   PURPLE(fp);
   fputs("\tptemp_prim_written", fp);
   RESET(fp);
   fprintf(fp, ": ");
   pdsc_print_ref(fp, p, src0->ptemp_prim_written);
   fputs("\n", fp);

   PURPLE(fp);
   fputs("\taddress", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src0->address);

   fputs("}", fp);
}

static void pdsc_print_stmp_src1(FILE *fp, const struct pdsc_stmp_src1 *src1)
{
   fputs("{\n", fp);

   PURPLE(fp);
   fputs("\tprim_id", fp);
   RESET(fp);
   fprintf(fp, ": 0x%08" PRIx32 "\n", src1->prim_id);

   PURPLE(fp);
   fputs("\tvioff", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu8 "\n", src1->vioff);

   PURPLE(fp);
   fputs("\tdmasize", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu8 "\n", src1->dmasize);

   PURPLE(fp);
   fputs("\tvooff", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu16 "\n", src1->vooff);

   fputs("}", fp);
}

static void pdsc_print_stmp_src2(FILE *fp, const struct pdsc_stmp_src2 *src2)
{
   fputs("{\n", fp);

   PURPLE(fp);
   fputs("\tprimtype", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_primtype_str(src2->primtype));

   PURPLE(fp);
   fputs("\tvosize", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu16 "\n", src2->vosize);

   if (src2->eop) {
      RED(fp);
      fputs("\teop\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tlimit", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src2->limit);

   fputs("}", fp);
}

static void pdsc_print_aa_src0(FILE *fp,
                               const pdsc_program *p,
                               const struct pdsc_aa_src0 *src0)
{
   fputs("{\n", fp);

   PURPLE(fp);
   fputs("\taop", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_aop_str(src0->aop));

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      PURPLE(fp);
      fputs("\tccpri", fp);
      RESET(fp);
      fprintf(fp, ": %" PRIu8 "\n", src0->ccpri);

      PURPLE(fp);
      fputs("\tccslc", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccslc_st_str(src0->ccslc));

      PURPLE(fp);
      fputs("\tccmcu", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccmcu_st_str(src0->ccmcu));
   }

   PURPLE(fp);
   fputs("\tsrcadd", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src0->srcadd);

   fputs("}", fp);
}

static void pdsc_print_idf_src0(FILE *fp,
                                const pdsc_program *p,
                                const struct pdsc_idf_src0 *src0)
{
   fputs("{\n", fp);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      PURPLE(fp);
      fputs("\tccpri", fp);
      RESET(fp);
      fprintf(fp, ": %" PRIu8 "\n", src0->ccpri);

      PURPLE(fp);
      fputs("\tccslc", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccslc_idf_str(src0->ccslc));

      PURPLE(fp);
      fputs("\tccmcu", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccmcu_idf_str(src0->ccmcu));
   }

   PURPLE(fp);
   fputs("\tsrcadd", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src0->srcadd);

   fputs("}", fp);
}

static void pdsc_print_pol_src0(FILE *fp,
                                const pdsc_program *p,
                                const struct pdsc_pol_src0 *src0)
{
   fputs("{\n", fp);

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      PURPLE(fp);
      fputs("\tccpri", fp);
      RESET(fp);
      fprintf(fp, ": %" PRIu8 "\n", src0->ccpri);

      PURPLE(fp);
      fputs("\tccslc", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccslc_ld_str(src0->ccslc));

      PURPLE(fp);
      fputs("\tccmcu", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccmcu_ld_str(src0->ccmcu));
   }

   PURPLE(fp);
   fputs("\tdest", fp);
   RESET(fp);
   fprintf(fp, ": ");
   pdsc_print_ref(fp, p, src0->dest);
   fputs("\n", fp);

   PURPLE(fp);
   fputs("\tsrcadd", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src0->srcadd);

   fputs("}", fp);
}

static void pdsc_print_ddmad_src3(FILE *fp,
                                  const pdsc_program *p,
                                  const struct pdsc_ddmad_src3 *src3)
{
   fputs("{\n", fp);

   if (PDSC_HAS(p, DDMADT)) {
      PURPLE(fp);
      fputs("\tmsize", fp);
      RESET(fp);
      fprintf(fp, ": %" PRIu32 "\n", src3->msize);

      if (src3->test) {
         RED(fp);
         fputs("\ttest\n", fp);
         RESET(fp);
      }
   }

   if (src3->last_issue) {
      RED(fp);
      fputs("\tlast_issue\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tbsize", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu16 "\n", src3->bsize);

   PURPLE(fp);
   fputs("\tdest", fp);
   RESET(fp);
   fputs(": ", fp);

   if (!src3->bsize)
      fputs("(none, !bsize) ", fp);

   if (src3->bsize > 1) {
      fprintf(fp,
              "%s%" PRIu16 "..%" PRIu16,
              pdsc_dout_store_dest_str(src3->store_dest),
              src3->ao,
              src3->ao + src3->bsize - 1);
   } else {
      fprintf(fp,
              "%s%" PRIu16,
              pdsc_dout_store_dest_str(src3->store_dest),
              src3->ao);
   }
   fputs("\n", fp);

   PURPLE(fp);
   fputs("\tcmode", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_cmode_ld_str(src3->cmode));

   fputs("}", fp);
}

static void pdsc_print_doutu_src0(FILE *fp, const struct pdsc_doutu_src0 *src0)
{
   fputs("{\n", fp);

   if (src0->dual_phase) {
      RED(fp);
      fputs("\tdual_phase\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\ttemps", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu8 "\n", src0->temps * 4);

   PURPLE(fp);
   fputs("\tsample_rate", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_doutu_sample_rate_str(src0->sample_rate));

   PURPLE(fp);
   fputs("\texe_off", fp);
   RESET(fp);
   fprintf(fp, ": 0x%08" PRIx32 "\n", src0->exe_off);

   fputs("}", fp);
}

static void pdsc_print_doutd_src0(FILE *fp,
                                  const pdsc_program *p,
                                  struct pdsc_doutd_src0 *src0)
{
   fputs("{\n", fp);

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS)) {
      PURPLE(fp);
      fputs("\tslcmode", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_slc_mode_ld_str(src0->slcmode));
   }

   if (PDSC_HAS(p, CACHE_HIERARCHY)) {
      PURPLE(fp);
      fputs("\tccpri", fp);
      RESET(fp);
      fprintf(fp, ": %" PRIu8 "\n", src0->ccpri);

      PURPLE(fp);
      fputs("\tccslc", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccslc_ld_str(src0->ccslc));

      PURPLE(fp);
      fputs("\tccmcu", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_ccmcu_ld_str(src0->ccmcu));
   }

   PURPLE(fp);
   fputs("\tdoffset", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu16 "\n", src0->doffset);

   PURPLE(fp);
   fputs("\tsbase", fp);
   RESET(fp);
   fprintf(fp, ": 0x%010" PRIx64 "\n", src0->sbase);

   fputs("}", fp);
}

static void pdsc_print_doutd_src1(FILE *fp, const struct pdsc_doutd_src1 *src1)
{
   fputs("{\n", fp);

   if (src1->last_issue) {
      RED(fp);
      fputs("\tlast_issue\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tbsize", fp);
   RESET(fp);
   fprintf(fp, ": %" PRIu16 "\n", src1->bsize);

   PURPLE(fp);
   fputs("\tdest", fp);
   RESET(fp);
   fputs(": ", fp);

   if (!src1->bsize)
      fputs("(none, !bsize) ", fp);

   if (src1->bsize > 1) {
      fprintf(fp,
              "%s%" PRIu16 "..%" PRIu16,
              pdsc_dout_store_dest_str(src1->store_dest),
              src1->ao,
              src1->ao + src1->bsize - 1);
   } else {
      fprintf(fp,
              "%s%" PRIu16,
              pdsc_dout_store_dest_str(src1->store_dest),
              src1->ao);
   }
   fputs("\n", fp);

   PURPLE(fp);
   fputs("\tcmode", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_cmode_ld_str(src1->cmode));

   fputs("}", fp);
}

static const char *pdsc_doutw_src1_bsize_str(enum pdsc_doutw_bsize bsize)
{
   switch (bsize) {
   case PDSC_DOUTW_BSIZE_LOWER:
      return "32 (lo)";
   case PDSC_DOUTW_BSIZE_UPPER:
      return "32 (hi)";
   case PDSC_DOUTW_BSIZE_ALL64:
      return "64";
   case PDSC_DOUTW_BSIZE_NONE:
      return "0";
   case PDSC_DOUTW_BSIZE_ALL128:
      return "128";
   default:
      break;
   }

   UNREACHABLE("Invalid doutw bsize.");
}

static void pdsc_print_doutw_src1(FILE *fp,
                                  const pdsc_program *p,
                                  struct pdsc_doutw_src1 *src1)
{
   fputs("{\n", fp);

   if (src1->last_issue) {
      RED(fp);
      fputs("\tlast_issue\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tbsize", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_doutw_src1_bsize_str(src1->bsize));

   PURPLE(fp);
   fputs("\tdest", fp);
   RESET(fp);
   fputs(": ", fp);

   switch (src1->bsize) {
   case PDSC_DOUTW_BSIZE_NONE:
      fputs("(none, !bsize) ", fp);
      FALLTHROUGH;

   case PDSC_DOUTW_BSIZE_LOWER:
   case PDSC_DOUTW_BSIZE_UPPER:
      fprintf(fp,
              "%s%" PRIu16,
              pdsc_dout_store_dest_str(src1->store_dest),
              src1->ao);
      break;

   case PDSC_DOUTW_BSIZE_ALL64:
      fprintf(fp,
              "%s%" PRIu16 "..%" PRIu16,
              pdsc_dout_store_dest_str(src1->store_dest),
              src1->ao,
              src1->ao + 1);
      break;

   case PDSC_DOUTW_BSIZE_ALL128:
      fprintf(fp,
              "%s%" PRIu16 "..%" PRIu16,
              pdsc_dout_store_dest_str(src1->store_dest),
              src1->ao,
              src1->ao + 3);
      break;

   default:
      UNREACHABLE("Invalid doutw bsize.");
   }
   fputs("\n", fp);

   if (!PDSC_HAS(p, DOUT_EXT)) {
      PURPLE(fp);
      fputs("\tcmode", fp);
      RESET(fp);
      fprintf(fp, ": %s\n", pdsc_cmode_ld_str(src1->cmode));
   }

   fputs("}", fp);
}

static const char *
pdsc_douti_src0_shademodel_str(enum pdsc_douti_shademodel shademodel)
{
   switch (shademodel) {
   case PDSC_DOUTI_SHADEMODEL_FLAT_VERTEX0:
      return "flat_vertex0";
   case PDSC_DOUTI_SHADEMODEL_FLAT_VERTEX1:
      return "flat_vertex1";
   case PDSC_DOUTI_SHADEMODEL_FLAT_VERTEX2:
      return "flat_vertex2";
   case PDSC_DOUTI_SHADEMODEL_GOURAUD:
      return "gouraud";
   default:
      break;
   }

   UNREACHABLE("Invalid douti shademodel.");
}

static const char *pdsc_douti_src0_size_str(enum pdsc_douti_size size)
{
   switch (size) {
   case PDSC_DOUTI_SIZE_1D:
      return "1d";
   case PDSC_DOUTI_SIZE_2D:
      return "2d";
   case PDSC_DOUTI_SIZE_3D:
      return "3d";
   case PDSC_DOUTI_SIZE_4D:
      return "4d";
   default:
      break;
   }

   UNREACHABLE("Invalid douti size.");
}

static void pdsc_print_douti_src0(FILE *fp, const struct pdsc_douti_src0 *src0)
{
   fputs("{\n", fp);

   if (src0->last_issue) {
      RED(fp);
      fputs("\tlast_issue\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tdest", fp);
   RESET(fp);
   fprintf(fp, ": %u\n", src0->dest);

   if (src0->depth_bias) {
      RED(fp);
      fputs("\tdepth_bias\n", fp);
      RESET(fp);
   }

   if (src0->primitive_id) {
      RED(fp);
      fputs("\tprimitive_id\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tshademodel", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_douti_src0_shademodel_str(src0->shademodel));

   if (src0->point_sprite) {
      RED(fp);
      fputs("\tpoint_sprite\n", fp);
      RESET(fp);
   }

   if (src0->wrap_u) {
      RED(fp);
      fputs("\twrap_u\n", fp);
      RESET(fp);
   }

   if (src0->wrap_v) {
      RED(fp);
      fputs("\twrap_v\n", fp);
      RESET(fp);
   }

   if (src0->wrap_s) {
      RED(fp);
      fputs("\twrap_s\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tsize", fp);
   RESET(fp);
   fprintf(fp, ": %s\n", pdsc_douti_src0_size_str(src0->size));

   if (src0->f16) {
      RED(fp);
      fputs("\tf16\n", fp);
      RESET(fp);
   }

   if (src0->perspective) {
      RED(fp);
      fputs("\tperspective\n", fp);
      RESET(fp);
   }

   PURPLE(fp);
   fputs("\tf32_offset", fp);
   RESET(fp);
   fprintf(fp, ": %u\n", src0->f32_offset);

   PURPLE(fp);
   fputs("\tf16_offset", fp);
   RESET(fp);
   fprintf(fp, ": %u\n", src0->f16_offset);

   fputs("}", fp);
}

/* TODO: implement commented out functions. */
static void pdsc_print_typed_value(FILE *fp,
                                   const pdsc_program *p,
                                   enum pdsc_type type,
                                   uint64_t value,
                                   bool is_64bit)
{
   switch (type) {
   case PDSC_TYPE_IMM:
      BLUE(fp);
      if (is_64bit)
         fprintf(fp, "0x%016" PRIx64, value);
      else
         fprintf(fp, "0x%08" PRIx32, (uint32_t)value);
      RESET(fp);
      break;

   case PDSC_TYPE_STM_SRC0: {
      struct pdsc_stm_src0 src0;
      pdsc_stm_src0_unpack(p, &src0, value);
      pdsc_print_stm_src0(fp, p, &src0);
      break;
   }

   case PDSC_TYPE_STM_SRC1: {
      struct pdsc_stm_src1 src1;
      pdsc_stm_src1_unpack(p, &src1, value);
      pdsc_print_stm_src1(fp, &src1);
      break;
   }

   case PDSC_TYPE_STM_SRC2: {
      struct pdsc_stm_src2 src2;
      pdsc_stm_src2_unpack(p, &src2, value);
      pdsc_print_stm_src2(fp, &src2);
      break;
   }

   case PDSC_TYPE_STM_SRC3: {
      struct pdsc_stm_src3 src3;
      pdsc_stm_src3_unpack(p, &src3, value);
      pdsc_print_stm_src3(fp, &src3);
      break;
   }

   case PDSC_TYPE_LD64_SRC0: {
      struct pdsc_ld64_src0 src0;
      pdsc_ld64_src0_unpack(p, &src0, value);
      pdsc_print_ld64_src0(fp, p, &src0);
      break;
   }

   case PDSC_TYPE_ST32_SRC0: {
      struct pdsc_st32_src0 src0;
      pdsc_st32_src0_unpack(p, &src0, value);
      pdsc_print_st32_src0(fp, p, &src0);
      break;
   }

   case PDSC_TYPE_STMP_SRC0: {
      struct pdsc_stmp_src0 src0;
      pdsc_stmp_src0_unpack(p, &src0, value);
      pdsc_print_stmp_src0(fp, p, &src0);
      break;
   }

   case PDSC_TYPE_STMP_SRC1: {
      struct pdsc_stmp_src1 src1;
      pdsc_stmp_src1_unpack(p, &src1, value);
      pdsc_print_stmp_src1(fp, &src1);
      break;
   }

   case PDSC_TYPE_STMP_SRC2: {
      struct pdsc_stmp_src2 src2;
      pdsc_stmp_src2_unpack(p, &src2, value);
      pdsc_print_stmp_src2(fp, &src2);
      break;
   }

   case PDSC_TYPE_AA_SRC0: {
      struct pdsc_aa_src0 src0;
      pdsc_aa_src0_unpack(p, &src0, value);
      pdsc_print_aa_src0(fp, p, &src0);
      break;
   }

   case PDSC_TYPE_IDF_SRC0: {
      struct pdsc_idf_src0 src0;
      pdsc_idf_src0_unpack(p, &src0, value);
      pdsc_print_idf_src0(fp, p, &src0);
      break;
   }

   case PDSC_TYPE_POL_SRC0: {
      struct pdsc_pol_src0 src0;
      pdsc_pol_src0_unpack(p, &src0, value);
      pdsc_print_pol_src0(fp, p, &src0);
      break;
   }

   case PDSC_TYPE_DDMAD_SRC3: {
      struct pdsc_ddmad_src3 src3;
      pdsc_ddmad_src3_unpack(p, &src3, value);
      pdsc_print_ddmad_src3(fp, p, &src3);
      break;
   }

   case PDSC_TYPE_DOUTU_SRC0: {
      struct pdsc_doutu_src0 src0;
      pdsc_doutu_src0_unpack(p, &src0, value);
      pdsc_print_doutu_src0(fp, &src0);
      break;
   }

   case PDSC_TYPE_DOUTD_SRC0: {
      struct pdsc_doutd_src0 src0;
      pdsc_doutd_src0_unpack(p, &src0, value);
      pdsc_print_doutd_src0(fp, p, &src0);
      break;
   }

   case PDSC_TYPE_DOUTD_SRC1: {
      struct pdsc_doutd_src1 src1;
      pdsc_doutd_src1_unpack(p, &src1, value);
      pdsc_print_doutd_src1(fp, &src1);
      break;
   }

   case PDSC_TYPE_DOUTW_SRC1: {
      struct pdsc_doutw_src1 src1;
      pdsc_doutw_src1_unpack(p, &src1, value);
      pdsc_print_doutw_src1(fp, p, &src1);
      break;
   }

   case PDSC_TYPE_DOUTI_SRC0: {
      struct pdsc_douti_src0 src0;
      pdsc_douti_src0_unpack(p, &src0, value);
      pdsc_print_douti_src0(fp, &src0);
      break;
   }

   default:
      UNREACHABLE("Unexpected type.");
   }
}

static void pdsc_print_names(FILE *fp, const char **names)
{
   if (!names)
      return;

   bool single_name = !names[1];

   fputs(" ", fp);

   if (!single_name)
      fputs("<", fp);

   for (const char **aliases = names; *aliases; ++aliases) {
      if (aliases != names)
         fputs(", ", fp);

      YELLOW(fp);
      fprintf(fp, "%s", *aliases);
      RESET(fp);
   }

   if (!single_name)
      fputs(">", fp);
}

static void pdsc_print_const(FILE *fp,
                             const pdsc_program *p,
                             uint8_t index,
                             const uint32_t *const_buffer)
{
   bool is_64bit = BITSET_TEST(p->consts_64, index);

   enum pdsc_type type = pdsc_get_const_type(p, index);
   pdsc_print_type(fp, type, is_64bit);

   fprintf(fp, " c[%u", index);

   if (is_64bit)
      fprintf(fp, "..%u", index + 1);

   fputs("]", fp);

   const char **names = pdsc_get_names(&p->const_names, index);
   pdsc_print_names(fp, names);

   fputs(" = ", fp);

   if (!BITSET_TEST(p->consts_assigned, index)) {
      fputs("(unassigned)", fp);
   } else {
      uint64_t value;
      if (!const_buffer) {
         value = pdsc_get_assigned_const(p, index, is_64bit);
      } else {
         value = const_buffer[index];
         if (is_64bit)
            value |= (uint64_t)const_buffer[index + 1] << 32;
      }
      pdsc_print_typed_value(fp, p, type, value, is_64bit);
   }

   fputs(";", fp);

   uint8_t patch_id = pdsc_get_const_patch_id(p, index);
   if (patch_id != PDSC_CONST_PATCH_ID_NONE)
      fprintf(fp, " (patch_id %u)", patch_id);
}

static void pdsc_print_temp(FILE *fp, const pdsc_program *p, uint8_t index)
{
   bool is_array = !BITSET_TEST(p->temps_array_elem, index) &&
                   BITSET_TEST(p->temps_array_elem, index + 1);
   uint8_t array_elems = 1;

   if (is_array) {
      BITSET_DECLARE(temps_array_elem, PDSC_MAX_TEMPS);
      BITSET_COPY(temps_array_elem, p->temps_array_elem);
      BITSET_SET_RANGE(temps_array_elem, 0, index);
      BITSET_NOT(temps_array_elem);
      unsigned array_end = BITSET_FFS(temps_array_elem);
      array_elems = array_end - index - 1;
   }

   bool is_64bit = BITSET_TEST(p->temps_64, index);

   pdsc_print_type(fp, PDSC_TYPE_IMM, is_64bit);

   fprintf(fp, " t[%u", index);

   if (is_array)
      fprintf(fp, "..%u", index + array_elems - 1);
   else if (is_64bit)
      fprintf(fp, "..%u", index + 1);

   fputs("]", fp);

   const char **names = pdsc_get_names(&p->temp_names, index);
   pdsc_print_names(fp, names);

   fputs(";", fp);
}

static void pdsc_print_ptemp(FILE *fp, const pdsc_program *p, uint8_t index)
{
   bool is_64bit = BITSET_TEST(p->ptemps_64, index);

   pdsc_print_type(fp, PDSC_TYPE_IMM, is_64bit);

   fprintf(fp, " pt[%u", index);

   if (is_64bit)
      fprintf(fp, "..%u", index + 1);

   fputs("]", fp);

   const char **names = pdsc_get_names(&p->ptemp_names, index);
   pdsc_print_names(fp, names);

   fputs(";", fp);
}

static const char *pdsc_op_str(enum pdsc_op op)
{
   switch (op) {
   case PDSC_OP_SFTLP64:
      return "sftlp64";
   case PDSC_OP_SFTLP32:
      return "sftlp32";
   case PDSC_OP_STM:
      return "stm";
   case PDSC_OP_MAD:
      return "mad";
   case PDSC_OP_ADD64:
      return "add64";
   case PDSC_OP_ADD32:
      return "add32";
   case PDSC_OP_CMP:
      return "cmp";
   case PDSC_OP_BRA:
      return "bra";
   case PDSC_OP_LD64:
      return "ld64";
   case PDSC_OP_ST32:
      return "st32";
   case PDSC_OP_IDIV:
      return "idiv";
   case PDSC_OP_STMP:
      return "stmp";
   case PDSC_OP_AA:
      return "aa";
   case PDSC_OP_WDF:
      return "wdf";
   case PDSC_OP_IDF:
      return "idf";
   case PDSC_OP_POL:
      return "pol";
   case PDSC_OP_LIMM:
      return "limm";
   case PDSC_OP_LOCK:
      return "lock";
   case PDSC_OP_RELEASE:
      return "release";
   case PDSC_OP_HALT:
      return "halt";
   case PDSC_OP_NOP:
      return "nop";
   case PDSC_OP_STMC:
      return "stmc";
   case PDSC_OP_DDMAD:
      return "ddmad";
   case PDSC_OP_DOUT:
      return "dout";
   default:
      break;
   }

   UNREACHABLE("Invalid op.");
}

static const char *pdsc_lop_str(enum pdsc_lop lop)
{
   switch (lop) {
   case PDSC_LOP_NONE:
      return "";
   case PDSC_LOP_NOT:
      return ".not";
   case PDSC_LOP_AND:
      return ".and";
   case PDSC_LOP_OR:
      return ".or";
   case PDSC_LOP_XOR:
      return ".xor";
   case PDSC_LOP_XNOR:
      return ".xnor";
   case PDSC_LOP_NAND:
      return ".nand";
   case PDSC_LOP_NOR:
      return ".nor";
   default:
      break;
   }

   UNREACHABLE("Invalid logic op.");
}

static const char *pdsc_cop_str(enum pdsc_cop cop)
{
   switch (cop) {
   case PDSC_COP_EQ:
      return ".eq";
   case PDSC_COP_GT:
      return ".gt";
   case PDSC_COP_LT:
      return ".lt";
   case PDSC_COP_NE:
      return ".ne";
   default:
      break;
   }

   UNREACHABLE("Invalid comparison op.");
}

static const char *pdsc_pred_str(enum pdsc_pred pred)
{
   switch (pred) {
   case PDSC_PRED_P0:
      return "p0";
   case PDSC_PRED_IF0:
      return "if0";
   case PDSC_PRED_IF1:
      return "if1";
   case PDSC_PRED_SO_OVF_P0:
      return "so_ovf_p0";
   case PDSC_PRED_SO_OVF_P1:
      return "so_ovf_p1";
   case PDSC_PRED_SO_OVF_P2:
      return "so_ovf_p2";
   case PDSC_PRED_SO_OVF_P3:
      return "so_ovf_p3";
   case PDSC_PRED_SO_OVF_GLOBAL:
      return "so_ovf_global";
   case PDSC_PRED_KEEP:
      return "";
   case PDSC_PRED_OOB:
      return "oob";
   default:
      break;
   }

   UNREACHABLE("Invalid predicate.");
}

static const char *pdsc_dstdout_str(enum pdsc_dstdout dstdout)
{
   switch (dstdout) {
   case PDSC_DSTDOUT_D:
      return "d";
   case PDSC_DSTDOUT_W:
      return "w";
   case PDSC_DSTDOUT_U:
      return "u";
   case PDSC_DSTDOUT_V:
      return "v";
   case PDSC_DSTDOUT_I:
      return "i";
   case PDSC_DSTDOUT_C:
      return "c";
   default:
      break;
   }

   UNREACHABLE("Invalid dout destination.");
};

/* No dst/srcs, only branch target. */
static void pdsc_print_instr_bra(FILE *fp,
                                 const pdsc_program *p,
                                 pdsc_instr_t i,
                                 const pdsc_instr *_i)
{
   pdsc_instr_t target_instr = _i->target_instr;
   assert(target_instr < util_dynarray_num_elements(&p->instrs, pdsc_instr));

   int32_t rel_target_instr = target_instr - i;

   enum pdsc_pred srcc = pdsc_get_srcc(p, i);
   bool neg = pdsc_get_neg(p, i);
   if (srcc != PDSC_PRED_KEEP)
      fprintf(fp, "%s%s ? ", neg ? "!" : "", pdsc_pred_str(srcc));

   GREEN(fp);
   fprintf(fp, "%s", pdsc_op_str(_i->op));
   RESET(fp);
   fprintf(fp, " %03" PRIu16 " (%+" PRId32 ")", target_instr, rel_target_instr);

   enum pdsc_pred setc = pdsc_get_setc(p, i);
   if (setc != PDSC_PRED_KEEP)
      fprintf(fp, "; setc %s", pdsc_pred_str(setc));

   fputs(";", fp);
}

static void pdsc_print_instr(FILE *fp,
                             const pdsc_program *p,
                             pdsc_instr_t i,
                             const pdsc_instr *_i)
{
   if (_i->op == PDSC_OP_BRA)
      return pdsc_print_instr_bra(fp, p, i, _i);

   bool printed_args = false;

   if (pdsc_has_cc(p, i) && pdsc_get_cc(p, i))
      fputs("cc ? ", fp);

   GREEN(fp);
   fprintf(fp, "%s", pdsc_op_str(_i->op));
   RESET(fp);

   /* Op mods. */
   if (pdsc_has_dstdout(p, i))
      fprintf(fp, "%s", pdsc_dstdout_str(pdsc_get_dstdout(p, i)));

   if (pdsc_has_tst(p, i) && pdsc_get_tst(p, i))
      fputs(".tst", fp);

   if (pdsc_has_lop(p, i))
      fprintf(fp, "%s", pdsc_lop_str(pdsc_get_lop(p, i)));

   if (pdsc_has_sna(p, i) && pdsc_get_sna(p, i))
      fputs(".minus", fp);

   if (pdsc_has_alum(p, i))
      fprintf(fp, ".%c", pdsc_get_alum(p, i) ? 's' : 'u');

   if (pdsc_has_cop(p, i))
      fprintf(fp, "%s", pdsc_cop_str(pdsc_get_cop(p, i)));

   if (pdsc_has_neg(p, i) && pdsc_get_neg(p, i))
      fputs(".neg", fp);

   if (pdsc_has_end(p, i) && pdsc_get_end(p, i))
      fputs(".end", fp);

   if (pdsc_has_so(p, i)) {
      assert(pdsc_has_cc_global_ovf(p, i) && pdsc_has_cc_so_ovf(p, i));

      uint8_t so = pdsc_get_so(p, i);

      if (pdsc_get_cc_global_ovf(p, i))
         fputs(".global_ovf", fp);

      if (pdsc_get_cc_so_ovf(p, i))
         fputs(".so_ovf", fp);

      fprintf(fp, " so%u", so);
      printed_args = true;
   }

   if (pdsc_has_somask(p, i)) {
      enum pdsc_somask somask = pdsc_get_somask(p, i);
      if (!somask) {
         fputs(".<none>", fp);
      } else {
         if (somask & PDSC_SOMASK_0)
            fputs(".so0", fp);

         if (somask & PDSC_SOMASK_1)
            fputs(".so1", fp);

         if (somask & PDSC_SOMASK_2)
            fputs(".so2", fp);

         if (somask & PDSC_SOMASK_3)
            fputs(".so3", fp);

         if (somask & PDSC_SOMASK_GLOBAL)
            fputs(".global", fp);
      }
   }

   if (!pdsc_ref_is_null(_i->dst)) {
      fprintf(fp, "%s ", printed_args ? "," : "");
      pdsc_print_ref(fp, p, _i->dst);
      printed_args = true;
   }

   for (uint8_t u = 0; u < _i->num_srcs; ++u) {
      if (printed_args)
         fputs(",", fp);

      fputs(" ", fp);

      if (!pdsc_ref_is_null(_i->src[u]))
         pdsc_print_ref(fp, p, _i->src[u]);
      else
         fputs("_", fp);

      printed_args = true;
   }

   fputs(";", fp);
}

static void pdsc_print_program_features(FILE *fp, const pdsc_program *p)
{
   if (!p->features)
      fputs(" (none)", fp);

   if (PDSC_HAS(p, DOUT_EXT))
      fputs(" dout_ext", fp);

   if (PDSC_HAS(p, IDIV))
      fputs(" idiv", fp);

   if (PDSC_HAS(p, STMP))
      fputs(" stmp", fp);

   if (PDSC_HAS(p, DDMADT))
      fputs(" ddmadt", fp);

   if (PDSC_HAS(p, CACHE_HIERARCHY))
      fputs(" cache_hierarchy", fp);

   if (PDSC_HAS(p, MCU_CACHE_CONTROLS))
      fputs(" mcu_cache_controls", fp);
}

void pdsc_print_program(FILE *fp,
                        const pdsc_program *p,
                        const uint32_t *const_buffer)
{
   if (p->name)
      fprintf(fp, "\"%s\" ", p->name);

   fputs("pds program:\n", fp);

   fputs("features:", fp);
   pdsc_print_program_features(fp, p);
   fputs("\n", fp);

   if (!BITSET_IS_EMPTY(p->consts_used)) {
      fprintf(fp,
              "const regs used: %u, required: %u\n",
              BITSET_COUNT(p->consts_used),
              BITSET_LAST_BIT(p->consts_used));

      uint8_t index;
      pdsc_foreach_const_index (p, index) {
         pdsc_print_const(fp, p, index, const_buffer);
         fputs("\n", fp);
      }

      fputs("\n", fp);
   }

   if (!BITSET_IS_EMPTY(p->temps_used)) {
      fprintf(fp, "temp regs referenced: %u\n", BITSET_COUNT(p->temps_used));

      uint8_t index;
      pdsc_foreach_temp_index (p, index) {
         pdsc_print_temp(fp, p, index);
         fputs("\n", fp);
      }

      fputs("\n", fp);
   }

   if (!BITSET_IS_EMPTY(p->ptemps_used)) {
      fprintf(fp, "ptemp regs referenced: %u\n", BITSET_COUNT(p->ptemps_used));

      uint8_t index;
      pdsc_foreach_ptemp_index (p, index) {
         pdsc_print_ptemp(fp, p, index);
         fputs("\n", fp);
      }

      fputs("\n", fp);
   }

   pdsc_instr_t num_instrs = util_dynarray_num_elements(&p->instrs, pdsc_instr);
   if (num_instrs > 0) {
      fprintf(fp, "%u instr%s\n", num_instrs, num_instrs == 1 ? "" : "s");

      for (pdsc_instr_t i = 0; i < num_instrs; ++i) {
         pdsc_instr *_i = util_dynarray_element(&p->instrs, pdsc_instr, i);

         fprintf(fp, "%03u: ", i);
         pdsc_print_instr(fp, p, i, _i);
         fputs("\n", fp);
      }

      fputs("\n", fp);
   }
}

void pdsc_print_binary(FILE *fp, const pdsc_program *p)
{
   if (p->name)
      fprintf(fp, "\"%s\" ", p->name);

   fputs("pds program binary:\n", fp);

   fprintf(fp, "data segment (%u bytes):\n", pdsc_program_data_size(p));
   u_hexdump(fp, pdsc_program_data(p), pdsc_program_data_size(p), false);
   fprintf(fp, "code segment (%u bytes):\n", pdsc_program_code_size(p));
   u_hexdump(fp, pdsc_program_code(p), pdsc_program_code_size(p), false);
}
