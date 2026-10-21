/*
 * Copyright 2024 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "si_build_pm4.h"
#include "si_query.h"
#include "gfx/si_gfx.h"

#include "ac_barrier.h"
#include "ac_cmdbuf_cp.h"

static struct si_resource *si_get_wait_mem_scratch_bo(struct si_context *ctx,
                                                      struct radeon_cmdbuf *cs, bool is_secure)
{
   struct si_screen *sscreen = ctx->screen;

   assert(ctx->gfx_level < GFX11);

   if (likely(!is_secure)) {
      return ctx->wait_mem_scratch;
   } else {
      assert(sscreen->info.has_tmz_support);
      if (!ctx->wait_mem_scratch_tmz) {
         ctx->wait_mem_scratch_tmz =
            si_aligned_buffer_create(&sscreen->b,
                                     PIPE_RESOURCE_FLAG_UNMAPPABLE |
                                     SI_RESOURCE_FLAG_DRIVER_INTERNAL |
                                     PIPE_RESOURCE_FLAG_ENCRYPTED,
                                     PIPE_USAGE_DEFAULT, 4,
                                     sscreen->info.tcc_cache_line_size);
         si_cp_write_data(ctx, ctx->wait_mem_scratch_tmz, 0, 4, V_371_MEMORY, V_371_MICRO_ENGINE,
                          &ctx->wait_mem_number);
      }

      return ctx->wait_mem_scratch_tmz;
   }
}

static unsigned get_reduced_barrier_flags(struct si_context *ctx)
{
   unsigned flags = ctx->barrier_flags;

   if (!flags)
      return 0;

   if (!ctx->is_gfx_queue) {
      /* Only process compute flags. */
      flags &= AC_BARRIER_INV_ICACHE | AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM |
               AC_BARRIER_INV_L2 | AC_BARRIER_WB_L2 | AC_BARRIER_INV_L2_METADATA |
               AC_BARRIER_SYNC_CS;
   }

   /* Don't flush CB and DB if there have been no draw calls. */
   if (ctx->num_draw_calls == ctx->last_cb_flush_num_draw_calls &&
       ctx->num_decompress_calls == ctx->last_cb_flush_num_decompress_calls)
      flags &= ~AC_BARRIER_SYNC_AND_INV_CB;

   if (ctx->num_draw_calls == ctx->last_db_flush_num_draw_calls &&
       ctx->num_decompress_calls == ctx->last_db_flush_num_decompress_calls)
      flags &= ~AC_BARRIER_SYNC_AND_INV_DB;

   if (!ctx->compute_is_busy)
      flags &= ~AC_BARRIER_SYNC_CS;

   /* Track the last CB/DB flush. */
   if (flags & AC_BARRIER_SYNC_AND_INV_CB) {
      ctx->num_cb_cache_flushes++;
      ctx->last_cb_flush_num_draw_calls = ctx->num_draw_calls;
      ctx->last_cb_flush_num_decompress_calls = ctx->num_decompress_calls;
   }
   if (flags & AC_BARRIER_SYNC_AND_INV_DB) {
      ctx->num_db_cache_flushes++;
      ctx->last_db_flush_num_draw_calls = ctx->num_draw_calls;
      ctx->last_db_flush_num_decompress_calls = ctx->num_decompress_calls;
   }

   /* Skip VS and PS synchronization if they are idle. */
   if (ctx->num_draw_calls == ctx->last_ps_sync_num_draw_calls)
      flags &= ~AC_BARRIER_SYNC_VS & ~AC_BARRIER_SYNC_PS;
   else if (ctx->num_draw_calls == ctx->last_vs_sync_num_draw_calls)
      flags &= ~AC_BARRIER_SYNC_VS;

   /* Track the last VS/PS flush. Flushing CB or DB also waits for PS (obviously). */
   if (flags & (AC_BARRIER_SYNC_AND_INV_CB | AC_BARRIER_SYNC_AND_INV_DB | AC_BARRIER_SYNC_PS)) {
      ctx->last_ps_sync_num_draw_calls = ctx->num_draw_calls;
      ctx->last_vs_sync_num_draw_calls = ctx->num_draw_calls;
   } else if (flags & AC_BARRIER_SYNC_VS) {
      ctx->last_vs_sync_num_draw_calls = ctx->num_draw_calls;
   }

   /* We use a TS event to flush CB/DB on GFX9+. */
   bool uses_ts_event = ctx->gfx_level >= GFX9 &&
                        flags & (AC_BARRIER_SYNC_AND_INV_CB | AC_BARRIER_SYNC_AND_INV_DB);

   /* TS events wait for everything. */
   if (uses_ts_event)
      flags &= ~AC_BARRIER_SYNC_VS & ~AC_BARRIER_SYNC_PS & ~AC_BARRIER_SYNC_CS;

   /* TS events wait for compute too. */
   if (flags & AC_BARRIER_SYNC_CS || uses_ts_event)
      ctx->compute_is_busy = false;

   if (flags & AC_BARRIER_SYNC_VS)
      ctx->num_vs_flushes++;
   if (flags & AC_BARRIER_SYNC_PS)
      ctx->num_ps_flushes++;
   if (flags & AC_BARRIER_SYNC_CS)
      ctx->num_cs_flushes++;

   if (flags & AC_BARRIER_INV_L2)
      ctx->num_L2_invalidates++;
   else if (flags & AC_BARRIER_WB_L2)
      ctx->num_L2_writebacks++;

   if (flags & AC_BARRIER_PIPELINESTAT_START) {
      if (ctx->pipeline_stats_enabled != 1) {
         ctx->pipeline_stats_enabled = 1;
      } else {
         flags &= ~AC_BARRIER_PIPELINESTAT_START;
      }
   } else if (flags & AC_BARRIER_PIPELINESTAT_STOP) {
      if (ctx->pipeline_stats_enabled != 0) {
         ctx->pipeline_stats_enabled = 0;
      } else {
         flags &= ~AC_BARRIER_PIPELINESTAT_STOP;
      }
   }

   ctx->barrier_flags = 0;
   return flags;
}

static enum ac_barrier_flags si_get_ac_barrier_flags(enum amd_gfx_level gfx_level,
                                                     unsigned si_flags)
{
   enum ac_barrier_flags flags = si_flags;

   /* radeonsi has no dedicated CB/DB metadata flags: on GFX6-9 a CB/DB flush
    * always flushes the corresponding metadata (CMASK/FMASK/DCC, HTILE) too.
    */
   if (gfx_level < GFX10) {
      if (flags & AC_BARRIER_SYNC_AND_INV_CB)
         flags |= AC_BARRIER_SYNC_AND_INV_CB_META;
      if (flags & AC_BARRIER_SYNC_AND_INV_DB)
         flags |= AC_BARRIER_SYNC_AND_INV_DB_META;
   }

   return flags;
}

static void si_emit_barrier(struct si_context *sctx, struct radeon_cmdbuf *cs)
{
   enum amd_gfx_level gfx_level = sctx->gfx_level;
   enum amd_ip_type ip_type = sctx->is_gfx_queue ? AMD_IP_GFX : AMD_IP_COMPUTE;
   unsigned flags = get_reduced_barrier_flags(sctx);
   enum ac_rgp_flush_bits rgp_flush_bits = 0;
   uint64_t wait_mem_va = 0, eop_bug_va = 0;
   uint32_t *wait_mem_number = NULL;

   if (!flags)
      return;

   const uint32_t flush_cb_db = flags & (AC_BARRIER_SYNC_AND_INV_CB |
                                         AC_BARRIER_SYNC_AND_INV_DB);

   if (gfx_level >= GFX9) {
      if (gfx_level < GFX11 && flush_cb_db) {
         struct si_resource* wait_mem_scratch =
           si_get_wait_mem_scratch_bo(sctx, cs, sctx->ws->cs_is_secure(cs));

         wait_mem_va = wait_mem_scratch->gpu_address;
         wait_mem_number = &sctx->wait_mem_number;

         eop_bug_va = si_get_eop_bug_va(sctx, wait_mem_scratch, SI_NOT_QUERY);
      }

      if (si_need_emit_task_shader_query(sctx, cs) &&
          (flags & (AC_BARRIER_PIPELINESTAT_START |
                    AC_BARRIER_PIPELINESTAT_STOP))) {
         radeon_begin(cs->gang_cs);
         radeon_set_sh_reg(R_00B828_COMPUTE_PIPELINESTAT_ENABLE,
                           S_00B828_PIPELINESTAT_ENABLE(sctx->pipeline_stats_enabled));
         radeon_end();
      }
   } else if (gfx_level == GFX8 && flags & AC_BARRIER_SYNC_AND_INV_CB) {
      eop_bug_va = si_get_eop_bug_va(sctx, NULL, SI_NOT_QUERY);
   }

   struct ac_barrier_state barrier = {
      .flags = si_get_ac_barrier_flags(gfx_level, flags),
      .pws_acquire_point = AC_PWS_ACQUIRE_POINT_PFP,
      .wait_mem_va = wait_mem_va,
      .wait_mem_number = wait_mem_number,
      .eop_bug_va = eop_bug_va,
   };

   if (unlikely(sctx->sqtt_enabled))
      si_sqtt_describe_barrier_start(sctx, cs);

   ac_emit_barrier(&cs->current, gfx_level, ip_type, &barrier,
                   &sctx->context_roll, &rgp_flush_bits);

   if (unlikely(sctx->sqtt_enabled))
      si_sqtt_describe_barrier_end(sctx, cs, rgp_flush_bits);

   if (gfx_level >= GFX10_3) {
      /* Increase task wait count if not done before. */
      if (sctx->task_wait_count == sctx->last_task_wait_count)
         sctx->task_wait_count++;
   }
}

static void si_emit_barrier_as_atom(struct si_context *sctx, unsigned index)
{
   sctx->emit_barrier(sctx, &sctx->gfx_cs);
}

void si_barrier_before_internal_op(struct si_context *sctx, unsigned flags,
                                   unsigned num_buffers,
                                   const struct pipe_shader_buffer *buffers,
                                   unsigned writable_buffers_mask,
                                   unsigned num_images,
                                   const struct pipe_image_view *images)
{
   unsigned new_barriers;

   /* Invalidate the VMEM cache only. The SMEM cache isn't used by shader buffers. */
   new_barriers = AC_BARRIER_INV_VMEM;

   for (unsigned i = 0; i < num_images; i++) {
      /* The driver doesn't decompress resources automatically for internal blits, so do it manually. */
      si_decompress_subresource(&sctx->b, images[i].resource, PIPE_MASK_RGBAZS,
                                images[i].u.tex.level, images[i].u.tex.first_layer,
                                images[i].u.tex.last_layer,
                                images[i].access & PIPE_IMAGE_ACCESS_WRITE);
   }

   /* Don't sync if buffers are idle. */
   const unsigned ps_mask = SI_BIND_CONSTANT_BUFFER(MESA_SHADER_FRAGMENT) |
                            SI_BIND_SHADER_BUFFER(MESA_SHADER_FRAGMENT) |
                            SI_BIND_IMAGE_BUFFER(MESA_SHADER_FRAGMENT) |
                            SI_BIND_SAMPLER_BUFFER(MESA_SHADER_FRAGMENT);
   const unsigned cs_mask = SI_BIND_CONSTANT_BUFFER(MESA_SHADER_COMPUTE) |
                            SI_BIND_SHADER_BUFFER(MESA_SHADER_COMPUTE) |
                            SI_BIND_IMAGE_BUFFER(MESA_SHADER_COMPUTE) |
                            SI_BIND_SAMPLER_BUFFER(MESA_SHADER_COMPUTE);

   for (unsigned i = 0; i < num_buffers; i++) {
      struct si_resource *buf = si_resource(buffers[i].buffer);

      if (!buf)
         continue;

      /* We always wait for the last write. If the buffer is used for write, also wait
       * for the last read.
       */
      if (!si_is_buffer_idle(sctx, buf, RADEON_USAGE_WRITE |
                             (writable_buffers_mask & BITFIELD_BIT(i) ? RADEON_USAGE_READ : 0))) {
         if (buf->bind_history & ps_mask)
            new_barriers |= AC_BARRIER_SYNC_PS;
         else
            new_barriers |= AC_BARRIER_SYNC_VS;

         if (buf->bind_history & cs_mask)
            new_barriers |= AC_BARRIER_SYNC_CS;
      }
   }

   /* Don't sync if images are idle. */
   for (unsigned i = 0; i < num_images; i++) {
      struct si_resource *img = si_resource(images[i].resource);
      bool writable = images[i].access & PIPE_IMAGE_ACCESS_WRITE;

      /* We always wait for the last write. If the buffer is used for write, also wait
       * for the last read.
       */
      if (!si_is_buffer_idle(sctx, img, RADEON_USAGE_WRITE | (writable ? RADEON_USAGE_READ : 0))) {
         si_make_CB_shader_coherent(sctx, images[i].resource->nr_samples, true,
               ((struct si_texture*)images[i].resource)->surface.u.gfx9.color.dcc.pipe_aligned);
         new_barriers |= AC_BARRIER_SYNC_PS | AC_BARRIER_SYNC_CS;
      }
   }

   si_set_barrier_flags(sctx, new_barriers);
}

void si_barrier_after_internal_op(struct si_context *sctx, unsigned flags,
                                  unsigned num_buffers,
                                  const struct pipe_shader_buffer *buffers,
                                  unsigned writable_buffers_mask,
                                  unsigned num_images,
                                  const struct pipe_image_view *images)
{
   unsigned new_barriers = AC_BARRIER_SYNC_CS;

   if (num_images) {
      /* Make sure image stores are visible to CB, which doesn't use L2 on GFX6-8. */
      new_barriers |= sctx->gfx_level <= GFX8 ? AC_BARRIER_WB_L2 : 0;
      /* Make sure image stores are visible to all CUs. */
      new_barriers |= AC_BARRIER_INV_VMEM;
   }

   /* Make sure buffer stores are visible to all CUs and also as index/indirect buffers. */
   if (num_buffers)
      new_barriers |= AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM | AC_BARRIER_PFP_SYNC_ME;

   /* We must set L2_cache_dirty for buffers because:
    * - GFX6,12: CP DMA doesn't use L2.
    * - GFX6-7,12: Index buffer reads don't use L2.
    * - GFX6-8,12: CP doesn't use L2.
    * - GFX6-8: CB/DB don't use L2.
    *
    * L2_cache_dirty is checked explicitly when buffers are used in those cases to enforce coherency.
    */
   while (writable_buffers_mask)
      si_resource(buffers[u_bit_scan(&writable_buffers_mask)].buffer)->L2_cache_dirty = true;

   /* Make sure RBs see our DCC image stores if RBs and TCCs (L2 instances) are non-coherent. */
   if (sctx->gfx_level >= GFX10 && sctx->screen->info.tcc_rb_non_coherent) {
      for (unsigned i = 0; i < num_images; i++) {
         if (vi_dcc_enabled((struct si_texture*)images[i].resource, images[i].u.tex.level) &&
             images[i].access & PIPE_IMAGE_ACCESS_WRITE &&
             (sctx->screen->always_allow_dcc_stores ||
              images[i].access & SI_IMAGE_ACCESS_ALLOW_DCC_STORE)) {
            new_barriers |= AC_BARRIER_INV_L2;
            break;
         }
      }
   }

   si_set_barrier_flags(sctx, new_barriers);
}

static void si_set_dst_src_barrier_buffers(struct pipe_shader_buffer *buffers,
                                           struct pipe_resource *dst, struct pipe_resource *src)
{
   assert(dst);
   memset(buffers, 0, sizeof(buffers[0]) * 2);
   /* Only the "buffer" field is going to be used. */
   buffers[0].buffer = dst;
   buffers[1].buffer = src;
}

/* This is for simple buffer ops that have 1 dst and 0-1 src. */
void si_barrier_before_simple_buffer_op(struct si_context *sctx, unsigned flags,
                                        struct pipe_resource *dst, struct pipe_resource *src)
{
   struct pipe_shader_buffer barrier_buffers[2];
   si_set_dst_src_barrier_buffers(barrier_buffers, dst, src);
   si_barrier_before_internal_op(sctx, flags, src ? 2 : 1, barrier_buffers, 0x1, 0, NULL);
}

/* This is for simple buffer ops that have 1 dst and 0-1 src. */
void si_barrier_after_simple_buffer_op(struct si_context *sctx, unsigned flags,
                                       struct pipe_resource *dst, struct pipe_resource *src)
{
   struct pipe_shader_buffer barrier_buffers[2];
   si_set_dst_src_barrier_buffers(barrier_buffers, dst, src);
   si_barrier_after_internal_op(sctx, flags, src ? 2 : 1, barrier_buffers, 0x1, 0, NULL);
}

static void si_texture_barrier(struct pipe_context *ctx, unsigned flags)
{
   si_fb_barrier_after_rendering((struct si_context *)ctx, SI_FB_BARRIER_SYNC_CB);
}

/* This enforces coherency between shader stores and any past and future access. */
static void si_memory_barrier(struct pipe_context *ctx, unsigned flags)
{
   struct si_context *sctx = (struct si_context *)ctx;
   unsigned new_barriers;

   /* Ignore PIPE_BARRIER_UPDATE_BUFFER - it synchronizes against updates like buffer_subdata. */
   /* Ignore PIPE_BARRIER_UPDATE_TEXTURE - it synchronizes against updates like texture_subdata. */
   /* Ignore PIPE_BARRIER_MAPPED_BUFFER - it synchronizes against buffer_map/unmap. */
   /* Ignore PIPE_BARRIER_QUERY_BUFFER - the GL spec description is confusing, and the driver
    * always inserts barriers around get_query_result_resource.
    */
   flags &= ~PIPE_BARRIER_UPDATE_BUFFER & ~PIPE_BARRIER_UPDATE_TEXTURE &
            ~PIPE_BARRIER_MAPPED_BUFFER & ~PIPE_BARRIER_QUERY_BUFFER;

   if (!flags)
      return;

   new_barriers = AC_BARRIER_SYNC_PS | AC_BARRIER_SYNC_CS;

   if (flags & PIPE_BARRIER_CONSTANT_BUFFER)
      new_barriers |= AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM;

   /* VMEM cache contents are written back to L2 automatically at the end of waves, but
    * the contents of other VMEM caches might still be stale.
    *
    * TEXTURE and IMAGE mean sampler buffers and image buffers, respectively.
    */
   if (flags & (PIPE_BARRIER_VERTEX_BUFFER | PIPE_BARRIER_SHADER_BUFFER | PIPE_BARRIER_TEXTURE |
                PIPE_BARRIER_IMAGE | PIPE_BARRIER_STREAMOUT_BUFFER))
      new_barriers |= AC_BARRIER_INV_VMEM;

   /* Unlike LLVM, ACO may use SMEM for SSBOs and global access. */
   if (sctx->screen->use_aco && (flags & PIPE_BARRIER_SHADER_BUFFER))
      new_barriers |= AC_BARRIER_INV_SMEM;

   if (flags & (PIPE_BARRIER_INDEX_BUFFER | PIPE_BARRIER_INDIRECT_BUFFER))
      new_barriers |= AC_BARRIER_PFP_SYNC_ME;

   /* Index buffers use L2 since GFX8 */
   if (flags & PIPE_BARRIER_INDEX_BUFFER &&
       (sctx->gfx_level <= GFX7 || sctx->screen->info.cp_sdma_ge_use_system_memory_scope))
      new_barriers |= AC_BARRIER_WB_L2;

   /* Indirect buffers use L2 since GFX9. */
   if (flags & PIPE_BARRIER_INDIRECT_BUFFER &&
       (sctx->gfx_level <= GFX8 || sctx->screen->info.cp_sdma_ge_use_system_memory_scope))
      new_barriers |= AC_BARRIER_WB_L2;

   /* MSAA color images are flushed in si_decompress_textures when needed.
    * Shaders never write to depth/stencil images.
    */
   if (flags & PIPE_BARRIER_FRAMEBUFFER && sctx->framebuffer.uncompressed_cb_mask) {
      new_barriers |= AC_BARRIER_SYNC_AND_INV_CB;

      if (sctx->gfx_level >= GFX10 && sctx->gfx_level < GFX12) {
         if (sctx->screen->info.tcc_rb_non_coherent)
            new_barriers |= AC_BARRIER_INV_L2;
         else /* We don't know which shaders do image stores with DCC: */
            new_barriers |= AC_BARRIER_INV_L2_METADATA;
      } else if (sctx->gfx_level == GFX9) {
         /* We have to invalidate L2 for MSAA and when DCC can have pipe_aligned=0. */
         new_barriers |= AC_BARRIER_INV_L2;
      } else if (sctx->gfx_level <= GFX8) {
         /* CB doesn't use L2 on GFX6-8.  */
         new_barriers |= AC_BARRIER_WB_L2;
      }
   }

   si_set_barrier_flags(sctx, new_barriers);
}

static void si_set_sampler_depth_decompress_mask(struct si_context *sctx, struct si_texture *tex)
{
   assert(sctx->gfx_level < GFX12);

   /* Check all sampler bindings in all shaders where depth textures are bound, and update
    * which samplers should be decompressed.
    */
   u_foreach_bit(sh, sctx->shader_has_depth_tex) {
      u_foreach_bit(i, sctx->samplers[sh].has_depth_tex_mask) {
         if (sctx->samplers[sh].views[i]->texture == &tex->buffer.b.b) {
            sctx->samplers[sh].needs_depth_decompress_mask |= 1 << i;
            sctx->shader_needs_decompress_mask |= 1 << sh;
         }
      }
   }
}

void si_fb_barrier_before_rendering(struct si_context *sctx)
{
   /* Wait for all shaders because all image loads must finish before CB/DB can write there. */
   if (sctx->framebuffer.state.nr_cbufs || sctx->framebuffer.state.zsbuf.texture)
      si_set_barrier_flags(sctx, AC_BARRIER_SYNC_CS | AC_BARRIER_SYNC_PS);
}

void si_fb_barrier_after_rendering(struct si_context *sctx, unsigned flags)
{
   if (sctx->gfx_level < GFX12 && !sctx->decompression_enabled) {
      /* Setting dirty_level_mask should ignore SI_FB_BARRIER_SYNC_* because it triggers
       * decompression, which is not syncing.
       */
      if (sctx->framebuffer.state.zsbuf.texture) {
         struct pipe_surface *surf = &sctx->framebuffer.state.zsbuf;
         struct si_texture *tex = (struct si_texture *)surf->texture;

         tex->dirty_level_mask |= 1 << surf->level;

         if (tex->surface.has_stencil)
            tex->stencil_dirty_level_mask |= 1 << surf->level;

         si_set_sampler_depth_decompress_mask(sctx, tex);
      }

      unsigned compressed_cb_mask = sctx->framebuffer.compressed_cb_mask;
      while (compressed_cb_mask) {
         unsigned i = u_bit_scan(&compressed_cb_mask);
         struct pipe_surface *surf = &sctx->framebuffer.state.cbufs[i];
         struct si_texture *tex = (struct si_texture *)surf->texture;

         if (tex->surface.fmask_offset) {
            tex->dirty_level_mask |= 1 << surf->level;
            tex->fmask_is_identity = false;
         }
      }
   }

   if (flags & SI_FB_BARRIER_SYNC_CB) {
      /* Compressed images (MSAA with FMASK) are flushed on demand in si_decompress_textures.
       *
       * Synchronize CB only if there is actually a bound color buffer.
       */
      if (sctx->framebuffer.uncompressed_cb_mask) {
         si_make_CB_shader_coherent(sctx, sctx->framebuffer.nr_samples,
                                    sctx->framebuffer.CB_has_shader_readable_metadata,
                                    sctx->framebuffer.all_DCC_pipe_aligned);
      }
   }

   if (flags & SI_FB_BARRIER_SYNC_DB && sctx->framebuffer.state.zsbuf.texture) {
      /* DB caches are flushed on demand (using si_decompress_textures) except the cases below. */
      if (sctx->gfx_level >= GFX12) {
         si_make_DB_shader_coherent(sctx, sctx->framebuffer.nr_samples, true, false);
      } else if (sctx->generate_mipmap_for_depth) {
         /* u_blitter doesn't invoke depth decompression when it does multiple blits in a row,
          * but the only case when it matters for DB is when doing generate_mipmap, which writes Z,
          * which is always uncompressed. So here we flush DB manually between individual
          * generate_mipmap blits.
          */
         si_make_DB_shader_coherent(sctx, 1, false, sctx->framebuffer.DB_has_shader_readable_metadata);
      } else if (sctx->screen->info.family == CHIP_NAVI33) {
         struct si_texture *old_ztex = (struct si_texture *)sctx->framebuffer.state.zsbuf.texture;

         if (old_ztex->upgraded_depth) {
            /* TODO: some failures related to hyperz appeared after 969ed851 on nv33:
             * - piglit tex-miplevel-selection
             * - KHR-GL46.direct_state_access.framebuffers_texture_attachment
             * - KHR-GL46.direct_state_access.framebuffers_texture_layer_attachment
             *
             * This seems to fix them:
             */
            si_set_barrier_flags(sctx, AC_BARRIER_SYNC_AND_INV_DB | AC_BARRIER_INV_L2);
         }
      } else if (sctx->gfx_level == GFX9) {
         /* It appears that DB metadata "leaks" in a sequence of:
          *  - depth clear
          *  - DCC decompress for shader image writes (with DB disabled)
          *  - render with DEPTH_BEFORE_SHADER=1
          * Flushing DB metadata works around the problem.
          */
         si_set_barrier_flags(sctx, AC_BARRIER_SYNC_AND_INV_DB_META);
      }
   }
}

void si_barrier_before_image_fast_clear(struct si_context *sctx, unsigned types)
{
   /* Invalidate the VMEM cache because we always use compute. */
   unsigned new_barriers = AC_BARRIER_INV_VMEM;

   /* Flush caches and wait for idle. */
   if (types & (SI_CLEAR_TYPE_CMASK | SI_CLEAR_TYPE_DCC)) {
      si_make_CB_shader_coherent(sctx, sctx->framebuffer.nr_samples,
                                 sctx->framebuffer.CB_has_shader_readable_metadata,
                                 sctx->framebuffer.all_DCC_pipe_aligned);
   }

   if (types & SI_CLEAR_TYPE_HTILE) {
      si_make_DB_shader_coherent(sctx, sctx->framebuffer.nr_samples, sctx->framebuffer.has_stencil,
                                 sctx->framebuffer.DB_has_shader_readable_metadata);
   }

   /* GFX6-8: CB and DB don't use L2. */
   if (sctx->gfx_level <= GFX8)
      new_barriers |= AC_BARRIER_INV_L2;

   si_set_barrier_flags(sctx, new_barriers);
}

void si_barrier_after_image_fast_clear(struct si_context *sctx)
{
   /* Wait for idle. */
   unsigned new_barriers = AC_BARRIER_SYNC_CS;

   /* GFX6-8: CB and DB don't use L2. */
   if (sctx->gfx_level <= GFX8)
      new_barriers |= AC_BARRIER_WB_L2;

   si_set_barrier_flags(sctx, new_barriers);
}

void si_init_barrier_functions(struct si_context *sctx)
{
   sctx->emit_barrier = si_emit_barrier;

   sctx->atoms.s.barrier.emit = si_emit_barrier_as_atom;

   sctx->b.memory_barrier = si_memory_barrier;
   sctx->b.texture_barrier = si_texture_barrier;
}
