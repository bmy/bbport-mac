/*
 * Copyright © 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "intel_nir.h"
#include "nir_builder.h"

struct frag_out_ctx {
   nir_scalar colour[FRAG_RESULT_MAX][4];
   nir_def *outputs[FRAG_RESULT_MAX];
   bool dual_blend;
   bool replicate_alpha;
};

static bool
collect_fragment_output(nir_builder *b, nir_intrinsic_instr *intr, void *ctx_)
{
   struct frag_out_ctx *ctx = ctx_;
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   const unsigned wrmask = nir_intrinsic_write_mask(intr);
   const unsigned c = nir_intrinsic_component(intr);

   gl_frag_result loc = nir_intrinsic_io_semantics(intr).location;
   if (loc == FRAG_RESULT_COLOR)
      loc = FRAG_RESULT_DATA0;
   else if (loc == FRAG_RESULT_DUAL_SRC_BLEND)
      ctx->dual_blend = true;

   if (loc < FRAG_RESULT_DATA0) {
      assert(c == 0 && wrmask == 1);
      assert(!ctx->outputs[loc] && "each non-colour output written only once");
      ctx->outputs[loc] = intr->src[0].ssa;

      /* Remove SampleMask writes that don't mask out any samples */
      const unsigned all_samples = BITFIELD_MASK(8);
      if (loc == FRAG_RESULT_SAMPLE_MASK &&
          nir_src_is_const(intr->src[0]) &&
          (nir_src_as_uint(intr->src[0]) & all_samples) == all_samples) {
         ctx->outputs[loc] = NULL;
         b->shader->info.outputs_written &= ~BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
      }
   } else {
      u_foreach_bit(i, wrmask) {
         assert(!ctx->colour[loc][c + i].def &&
                "each colour component written only once");
         ctx->colour[loc][c + i] = nir_get_scalar(intr->src[0].ssa, i);
      }
   }

   nir_instr_remove(&intr->instr);
   return true;
}

/* nir_vec_scalar colour components, filling any unwritten with undef */
static bool
gather_colour_components(nir_builder *b,
                         struct frag_out_ctx *ctx,
                         gl_frag_result loc,
                         nir_def *undef)
{
   bool written = false;

   /* If the alpha of a render target is missing, replicate the RT0 alpha. */
   for (unsigned c = 0; c < 4; c++) {
      if (!ctx->colour[loc][c].def) {
         ctx->colour[loc][c] = (c == 3 && loc > FRAG_RESULT_DATA0) ?
            ctx->colour[FRAG_RESULT_DATA0][3] :
            nir_get_scalar(undef, 0);
      } else {
         written = true;
      }
   }

   if (written)
      ctx->outputs[loc] = nir_vec_scalars(b, ctx->colour[loc], 4);

   return written;
}

static void
insert_rt_store(nir_builder *b, struct frag_out_ctx *ctx, signed target,
                intel_nir_rt_write_cb cb, void *cb_data)
{
   const unsigned src0_alpha_loc =
      FRAG_RESULT_DATA0 + (ctx->replicate_alpha ? 0 : MAX2(target, 0));

   nir_def *colour = ctx->outputs[FRAG_RESULT_DATA0 + MAX2(target, 0)];
   nir_def *dual_colour = ctx->outputs[FRAG_RESULT_DUAL_SRC_BLEND] ?: colour;
   nir_def *src0_alpha = nir_mov_scalar(b, ctx->colour[src0_alpha_loc][3]);

   /* OpenGL drivers get the jackpot of nonsense lowering, where a shader
    * starts with :
    *   - write to a variable FRAG_RESULT_DATA0, with color_is_dual_source=true
    *   - io lowering turns it into store_output on location FRAG_RESULT_DUAL_SRC_BLEND
    *   - nir_unlower_io_to_vars is run and now recreates a variable FRAG_RESULT_DUAL_SRC_BLEND
    *   - io lowering turns it into store_output on location FRAG_RESULT_DUAL_SRC_BLEND
    *
    * As a result with can end up here with colour==NULL and dual_colour!=NULL.
    */
   if (colour == NULL && dual_colour != NULL)
      colour = dual_colour;

   nir_store_render_target_intel(b,
                                 cb != NULL ? cb(b, target, cb_data) : nir_undef(b, 1, 64),
                                 colour, dual_colour, src0_alpha,
                                 ctx->outputs[FRAG_RESULT_SAMPLE_MASK],
                                 ctx->outputs[FRAG_RESULT_DEPTH],
                                 ctx->outputs[FRAG_RESULT_STENCIL],
                                 .target = target);
}

bool
intel_nir_lower_fragment_outputs(nir_shader *shader,
                                 unsigned nr_colour_regions,
                                 bool replicate_alpha,
                                 intel_nir_rt_write_cb cb,
                                 void *cb_data)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(shader);
   nir_builder b_ = nir_builder_at(nir_after_impl(impl));
   nir_builder *b = &b_;

   struct frag_out_ctx ctx = { .replicate_alpha = replicate_alpha };
   nir_function_intrinsics_pass(impl, collect_fragment_output,
                                nir_metadata_control_flow, &ctx);

   nir_def *undef = nir_undef(b, 1, 32);
   if (!ctx.outputs[FRAG_RESULT_DEPTH])
      ctx.outputs[FRAG_RESULT_DEPTH] = undef;
   if (!ctx.outputs[FRAG_RESULT_STENCIL])
      ctx.outputs[FRAG_RESULT_STENCIL] = undef;
   if (!ctx.outputs[FRAG_RESULT_SAMPLE_MASK])
      ctx.outputs[FRAG_RESULT_SAMPLE_MASK] = undef;

   if (ctx.dual_blend) {
      gather_colour_components(b, &ctx, FRAG_RESULT_DATA0, undef);
      gather_colour_components(b, &ctx, FRAG_RESULT_DUAL_SRC_BLEND, undef);
      insert_rt_store(b, &ctx, 0, cb, cb_data);
      return true;
   }
   ctx.outputs[FRAG_RESULT_DUAL_SRC_BLEND] = nir_undef(b, 4, 32);

   bool written = false;
   for (unsigned i = 0; i < nr_colour_regions; i++) {
      if (gather_colour_components(b, &ctx, FRAG_RESULT_DATA0 + i, undef)) {
         insert_rt_store(b, &ctx, i, cb, cb_data);
         written = true;
      }
   }

   if (!written) {
      /* Even if we don't write a RT, we still need to write alpha for
       * alpha-to-coverage and alpha testing. Optimize the other channels out.
       */
      for (unsigned c = 0; c < 3; c++)
         ctx.colour[FRAG_RESULT_DATA0][c] = nir_get_scalar(undef, 0);
      gather_colour_components(b, &ctx, FRAG_RESULT_DATA0, undef);

      insert_rt_store(b, &ctx, -1, cb, cb_data);
   }

   return true;
}
