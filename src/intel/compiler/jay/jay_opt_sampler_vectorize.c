/*
 * Copyright 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "util/macros.h"
#include "jay_builder.h"
#include "jay_builder_opcodes.h"
#include "jay_ir.h"
#include "jay_opcodes.h"
#include "jay_private.h"

/**
 * Is this def a (single-ugpr, _, _, _, _, _, _, _, _) collect?
 */
static bool
is_single_index_ugpr_vector(jay_shader *s, jay_def d)
{
   if (jay_num_values(d) != jay_ugpr_per_gpr(s))
      return false;
   if (jay_channel(d, 0) == 0)
      return false;
   for (unsigned i = 1; i < jay_num_values(d); ++i) {
      if (jay_channel(d, i) != 0)
         return false;
   }
   return true;
}

/**
 * Candidate SENDs for txf combining must obey the following criteria:
 * - uses the sampler SFID
 * - has no predication
 * - writes to a single GPR's worth of UGPRs
 * - has no indirect descriptors
 * - has one source, which is 1 UGPR padded out to a GRF
 * - operates on 32-bit data
 * - does not skip helpers
 */
static bool
is_candidate(jay_shader *s, jay_inst *I)
{
   return I->op == JAY_OPCODE_SEND &&
          jay_send_sfid(I) == GEN_SFID_SAMPLER &&
          !I->predication &&
          jay_is_uniform(I->dst) &&
          jay_num_values(I->dst) == jay_ugpr_per_gpr(s) &&
          jay_is_imm(I->src[0]) &&
          jay_is_null(I->src[3]) &&
          is_single_index_ugpr_vector(s, I->src[2]) &&
          jay_type_size_bits(I->type) == 32 &&
          !jay_send_skip_helpers(I);
}

static void
merge_to_one(jay_function *f, jay_inst **sends, jay_def *srcs, unsigned count)
{
   jay_builder b = jay_init_builder(f, jay_before_inst(sends[0]));

   jay_def srcs_copy[32];

   for (unsigned i = 0; i < count; i++) {
      srcs_copy[i] = jay_MOV_u32(&b, jay_imm(jay_as_uint(srcs[i])));
   }

   unsigned count_roundup = util_next_power_of_two(count);

   for (unsigned i = count; i < count_roundup; i++) {
      srcs_copy[i] = jay_alloc_def(&b, UGPR, 1);
      jay_UNDEF(&b, srcs_copy[i]);
   }

   jay_def src = jay_collect_vectors(&b, srcs_copy, count_roundup);

   unsigned values_per_send = 1;

   jay_def dst = jay_alloc_def(&b, UGPR, count_roundup * values_per_send);

   jay_SEND(&b, .sfid = GEN_SFID_SAMPLER,
            .msg_desc = jay_as_uint(sends[0]->src[0]), .desc = jay_null(),
            .ex_desc = sends[0]->src[1], .header = jay_null(), .srcs = &src,
            .nr_srcs = 1, .type = sends[0]->type,
            .src_type = { jay_src_type(sends[0], 2) }, .dst = dst,
            .bindless = jay_send_bindless(sends[0]), .pure = true,
            .explicit_simd_width = count_roundup, .uniform = true);

   for (unsigned i = 0; i < count; i++) {
      jay_copy(&b, sends[i]->dst, jay_extract_range(dst, i, values_per_send));
      jay_remove_instruction(sends[i]);
   }
}

static void
merge_sends(jay_function *f, jay_inst **sends, jay_def *srcs, unsigned count)
{
   const unsigned max_simd = 16;

   for (size_t i = 0; i < count; i += max_simd) {
      unsigned lanes = MIN2(count - i, max_simd);
      merge_to_one(f, sends + i, srcs + i, lanes);
   }
}

static void
pass(jay_function *f)
{
   jay_inst **immediate_values = calloc(f->ssa_alloc, sizeof(jay_inst *));

   jay_foreach_block(f, block) {
      jay_inst *sends[32] = {};
      unsigned count = 0;

      jay_def srcs[32] = {};

      jay_foreach_inst_in_block_safe(block, I) {
         if (I->op == JAY_OPCODE_MOV &&
             jay_is_imm(I->src[0]) &&
             !I->predication) {
            jay_foreach_dst_index(I, dst, i) {
               immediate_values[i] = I;
            }
         }

         if (!is_candidate(f->shader, I))
            continue;

         /* only use sampler instructions that have immediate texcoords */
         jay_inst *src_imm = immediate_values[jay_channel(I->src[2], 0)];
         if (!src_imm)
            continue;

         /* Candidate messages always have these properties */
         assert(jay_send_pure(I));
         assert(!jay_send_eot(I));
         assert(!jay_send_check_tdr(I));
         assert(!jay_send_ex_desc_imm(I));

         if (count == 0) {
            srcs[count] = (src_imm)->src[0];
            sends[count++] = I;
            continue;
         } else if (count == 32) {
            break;
         }

         if (jay_num_values(sends[0]->dst) != jay_num_values(I->dst) ||
             jay_as_uint(sends[0]->src[0]) != jay_as_uint(I->src[0]) ||
             !jay_defs_equivalent(sends[0]->src[1], I->src[1]))
            continue;

         srcs[count] = ((jay_inst *) src_imm)->src[0];
         sends[count++] = I;
      }

      if (count > 1) {
         merge_sends(f, sends, srcs, count);
      }
   }

   free(immediate_values);
}

JAY_DEFINE_FUNCTION_PASS(jay_opt_sampler_vectorize, pass)
