/*
 * Copyright 2024 Advanced Micro Devices, Inc.
 * Copyright 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#include "ac_barrier.h"
#include "ac_cmdbuf.h"
#include "ac_cmdbuf_cp.h"
#include "ac_rgp.h"

#include "amd_family.h"

static enum ac_barrier_flags
ac_get_reduced_barrier_flags(enum amd_gfx_level gfx_level, enum amd_ip_type ip_type,
                             enum ac_barrier_flags flags)
{
   if (ip_type == AMD_IP_COMPUTE) {
      /* Only process compute flags. */
      flags &= AC_BARRIER_INV_ICACHE | AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM |
               AC_BARRIER_INV_L2 | AC_BARRIER_WB_L2 | AC_BARRIER_INV_L2_METADATA |
               AC_BARRIER_SYNC_CS | AC_BARRIER_PIPELINESTAT_START |
               AC_BARRIER_PIPELINESTAT_STOP;
   }

   /* We use a TS event to flush CB/DB on GFX9+. */
   const bool uses_ts_event =
      gfx_level >= GFX9 && flags & (AC_BARRIER_SYNC_AND_INV_CB | AC_BARRIER_SYNC_AND_INV_DB);

   /* TS events wait for everything. */
   if (uses_ts_event)
      flags &= ~AC_BARRIER_SYNC_VS & ~AC_BARRIER_SYNC_PS & ~AC_BARRIER_SYNC_CS;

   return flags;
}

static void
ac_emit_common_barrier_flags(struct ac_cmdbuf *cs, enum amd_gfx_level gfx_level,
                             enum amd_ip_type ip_type, enum ac_barrier_flags flags)
{
   const bool is_mec = ip_type == AMD_IP_COMPUTE && gfx_level >= GFX7;

   ac_cmdbuf_begin(cs);

   if (flags & AC_BARRIER_PIPELINESTAT_START) {
      if (!is_mec) {
         ac_cmdbuf_event_write(V_028A90_PIPELINESTAT_START);
      } else {
         ac_cmdbuf_set_sh_reg(R_00B828_COMPUTE_PIPELINESTAT_ENABLE, S_00B828_PIPELINESTAT_ENABLE(1));
      }
   } else if (flags & AC_BARRIER_PIPELINESTAT_STOP) {
      if (!is_mec) {
         ac_cmdbuf_event_write(V_028A90_PIPELINESTAT_STOP);
      } else {
         ac_cmdbuf_set_sh_reg(R_00B828_COMPUTE_PIPELINESTAT_ENABLE, S_00B828_PIPELINESTAT_ENABLE(0));
      }
   }

   if (flags & AC_BARRIER_VGT_FLUSH)
      ac_cmdbuf_event_write(V_028A90_VGT_FLUSH);

   ac_cmdbuf_end();
}

static void
ac_gfx10_emit_barrier(struct ac_cmdbuf *cs, enum amd_gfx_level gfx_level,
                      enum amd_ip_type ip_type,
                      const struct ac_barrier_state *state,
                      bool *context_roll, enum ac_rgp_flush_bits *rgp_flush_bits)
{
   enum ac_barrier_flags flags =
      ac_get_reduced_barrier_flags(gfx_level, ip_type, state->flags);
   uint32_t gcr_cntl = 0;

   if (!flags)
      return;

   ac_emit_common_barrier_flags(cs, gfx_level, ip_type, flags);

   assert(!(flags & (AC_BARRIER_SYNC_AND_INV_DB_META |
                     AC_BARRIER_VGT_STREAMOUT_SYNC)));
   assert(gfx_level < GFX12 || !(flags & AC_BARRIER_INV_L2_METADATA));

   if (flags & AC_BARRIER_INV_ICACHE) {
      gcr_cntl |= S_587_GLI_INV(V_587_GLI_ALL);
      *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_ICACHE;
   }

   if (flags & AC_BARRIER_INV_SMEM) {
      gcr_cntl |= S_587_GLK_INV(1);
      *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_SMEM_L0;
   }

   if (flags & AC_BARRIER_INV_VMEM) {
      gcr_cntl |= S_587_GLV_INV(1);
      *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_VMEM_L0;
   }

   if (gfx_level < GFX12 && flags & (AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM)) {
      gcr_cntl |= S_587_GL1_INV(1);
      *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_L1;
   }

   /* The L2 cache ops are:
    * - INV: - invalidate lines that reflect memory (were loaded from memory)
    *        - don't touch lines that were overwritten (were stored by gfx clients)
    * - WB: - don't touch lines that reflect memory
    *       - write back lines that were overwritten
    * - WB | INV: - invalidate lines that reflect memory
    *             - write back lines that were overwritten
    *
    * GLM doesn't support WB alone. If WB is set, INV must be set too.
    */
   if (flags & AC_BARRIER_INV_L2) {
      gcr_cntl |= S_587_GL2_INV(1) | S_587_GL2_WB(1); /* Writeback and invalidate everything in L2. */
      *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_L2;
   } else if (flags & AC_BARRIER_WB_L2) {
      gcr_cntl |= S_587_GL2_WB(1);
      *rgp_flush_bits |= AC_RGP_FLUSH_FLUSH_L2;
   }

   /* Invalidate the metadata cache. */
   if (gfx_level < GFX12 &&
       flags & (AC_BARRIER_INV_L2 | AC_BARRIER_WB_L2 | AC_BARRIER_INV_L2_METADATA))
      gcr_cntl |= S_587_GLM_INV(1) | S_587_GLM_WB(1);

   /* Flush CB/DB. Note that this also idles all shaders, including compute shaders. */
   if (flags & (AC_BARRIER_SYNC_AND_INV_CB | AC_BARRIER_SYNC_AND_INV_DB)) {
      unsigned cb_db_event = 0;

      /* Determine the TS event that we'll use to flush CB/DB. */
      if ((flags & AC_BARRIER_SYNC_AND_INV_CB && flags & AC_BARRIER_SYNC_AND_INV_DB) ||
          /* GFX11 can't use the DB_META event and must use a full flush to flush DB_META. */
          (gfx_level == GFX11 && flags & AC_BARRIER_SYNC_AND_INV_DB)) {
         cb_db_event = V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT;
         *rgp_flush_bits |= AC_RGP_FLUSH_FLUSH_CB | AC_RGP_FLUSH_INVAL_CB |
                            AC_RGP_FLUSH_FLUSH_DB | AC_RGP_FLUSH_INVAL_DB;
      } else if (flags & AC_BARRIER_SYNC_AND_INV_CB) {
         cb_db_event = V_028A90_FLUSH_AND_INV_CB_DATA_TS;
         *rgp_flush_bits |= AC_RGP_FLUSH_FLUSH_CB | AC_RGP_FLUSH_INVAL_CB;
      } else {
         assert(flags & AC_BARRIER_SYNC_AND_INV_DB);
         cb_db_event = V_028A90_FLUSH_AND_INV_DB_DATA_TS;
         *rgp_flush_bits |= AC_RGP_FLUSH_FLUSH_DB | AC_RGP_FLUSH_INVAL_DB;
      }

      ac_cmdbuf_begin(cs);

      /* We must flush CMASK/FMASK/DCC separately if the main event only flushes CB_DATA. */
      if (gfx_level < GFX12 && cb_db_event == V_028A90_FLUSH_AND_INV_CB_DATA_TS)
         ac_cmdbuf_event_write(V_028A90_FLUSH_AND_INV_CB_META);

      /* We must flush HTILE separately if the main event only flushes DB_DATA. */
      if (gfx_level < GFX12 && cb_db_event == V_028A90_FLUSH_AND_INV_DB_DATA_TS)
         ac_cmdbuf_event_write(V_028A90_FLUSH_AND_INV_DB_META);

      ac_cmdbuf_end();

      /* First flush CB/DB, then L1/L2. */
      gcr_cntl |= S_587_SEQ(V_587_SEQ_FORWARD);

      if (gfx_level >= GFX11) {
         /* Send an event that flushes caches. */
         ac_emit_cp_release_mem_pws(cs, gfx_level, ip_type, cb_db_event,
                                    gcr_cntl & C_587_GLI_INV);

         /* The RELEASE_MEM above already flushed the data caches, so the
          * ACQUIRE only has to invalidate the I$ (GLI_INV, which RELEASE_MEM
          * can't handle).
          */
         uint32_t acquire_gcr_cntl = gcr_cntl & ~C_587_GLI_INV; /* keep only GLI_INV */

         /* Select the ACQUIRE point (PWS stage). The data caches are already
          * flushed, so ME is enough unless a PFP_SYNC_ME is pending (then
          * PFP); the barrier destination stage below can defer the wait
          * further to PRE_DEPTH.
          */
         uint32_t pws_stage = flags & AC_BARRIER_PFP_SYNC_ME ? V_581B_CP_PFP : V_581B_CP_ME;

         if (ip_type == AMD_IP_GFX) {
            enum ac_pws_acquire_point acquire_point = state->pws_acquire_point;

            /* HW limitation: GCR cache ops during an ACQUIRE can only be
             * performed at the PFP/ME stage. If the ACQUIRE still needs to
             * invalidate the I$ (GLI_INV), don't defer the wait past ME.
             */
            if (acquire_point < AC_PWS_ACQUIRE_POINT_ME && G_587_GLI_INV(acquire_gcr_cntl) != 0)
               acquire_point = AC_PWS_ACQUIRE_POINT_ME;

            switch (acquire_point) {
            case AC_PWS_ACQUIRE_POINT_PRE_DEPTH:
               pws_stage = V_581B_PRE_DEPTH;
               /* A PRE_DEPTH ACQUIRE can't carry GCR bits; GLI_INV is 0 here
                * (see the clamp above), so drop the remaining bits to make
                * the ACQUIRE a pure wait.
                */
               acquire_gcr_cntl = 0;
               break;
            case AC_PWS_ACQUIRE_POINT_ME:
               pws_stage = V_581B_CP_ME;
               break;
            default:
               break;
            }
         }

         /* Wait for the event and invalidate remaining caches if needed. */
         ac_emit_cp_acquire_mem_pws(cs, gfx_level, ip_type, cb_db_event,
                                    pws_stage, 0,
                                    gcr_cntl & ~C_587_GLI_INV /* keep only GLI_INV */);

         gcr_cntl = 0; /* all done */

         /* A PFP ACQUIRE_MEM is implemented as an ACQUIRE at the ME plus a
          * PFP_SYNC_ME, so it already syncs the PFP. A deferred
          * (ME/PRE_DEPTH) ACQUIRE does not, so only drop a pending
          * PFP_SYNC_ME when the ACQUIRE actually runs at the PFP.
          */
         if (pws_stage == V_581B_CP_PFP)
            flags &= ~AC_BARRIER_PFP_SYNC_ME;
      } else {
         /* CB/DB flush and invalidate (or possibly just a wait for a
          * meta flush) via RELEASE_MEM.
          *
          * Combine this with other cache flushes when possible; this
          * requires affected shaders to be idle, so do it after the
          * CS_PARTIAL_FLUSH before (VS/PS partial flushes are always
          * implied).
          */
         /* Get GCR_CNTL fields, because the encoding is different in RELEASE_MEM. */
         unsigned glm_wb = G_587_GLM_WB(gcr_cntl);
         unsigned glm_inv = G_587_GLM_INV(gcr_cntl);
         unsigned glv_inv = G_587_GLV_INV(gcr_cntl);
         unsigned gl1_inv = G_587_GL1_INV(gcr_cntl);
         assert(G_587_GL2_US(gcr_cntl) == 0);
         assert(G_587_GL2_RANGE(gcr_cntl) == 0);
         assert(G_587_GL2_DISCARD(gcr_cntl) == 0);
         unsigned gl2_inv = G_587_GL2_INV(gcr_cntl);
         unsigned gl2_wb = G_587_GL2_WB(gcr_cntl);
         unsigned gcr_seq = G_587_SEQ(gcr_cntl);

         gcr_cntl &=
            C_587_GLM_WB & C_587_GLM_INV & C_587_GLV_INV & C_587_GL1_INV & C_587_GL2_INV & C_587_GL2_WB; /* keep SEQ */

         assert(state->wait_mem_number);
         (*state->wait_mem_number)++;

         ac_emit_cp_release_mem(cs, gfx_level, ip_type, cb_db_event,
                                S_491_GLM_WB(glm_wb) |
                                S_491_GLM_INV(glm_inv) |
                                S_491_GLV_INV(glv_inv) |
                                S_491_GL1_INV(gl1_inv) |
                                S_491_GL2_INV(gl2_inv) |
                                S_491_GL2_WB(gl2_wb) |
                                S_491_SEQ(gcr_seq),
                                EOP_DST_SEL_MEM,
                                EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                                EOP_DATA_SEL_VALUE_32BIT,
                                state->wait_mem_va, *state->wait_mem_number,
                                state->eop_bug_va);

         ac_emit_cp_wait_mem(cs, state->wait_mem_va, *(state->wait_mem_number), 0xffffffff, WAIT_REG_MEM_EQUAL);
      }
   } else {
      /* The TS event above also makes sure that PS and CS are idle, so we
       * have to do this only if we are not flushing CB or DB.
       */

      /* Wait for graphics shaders to go idle if requested.
       *
       * On GFX10-11.7, PS_PARTIAL_FLUSH doesn't wait for GS waves that send
       * "gs_alloc_req 0", so we have to use VS_PARTIAL_FLUSH. (only tested
       * Raphael and Navi33)
       */
      ac_cmdbuf_begin(cs);

      if (flags & AC_BARRIER_SYNC_VS &&
          (gfx_level < GFX12 || !(flags & AC_BARRIER_SYNC_PS))) {
         ac_cmdbuf_event_write(V_028A90_VS_PARTIAL_FLUSH);
         *rgp_flush_bits |= AC_RGP_FLUSH_VS_PARTIAL_FLUSH;
      }

      if (flags & AC_BARRIER_SYNC_PS) {
         ac_cmdbuf_event_write(V_028A90_PS_PARTIAL_FLUSH);
         *rgp_flush_bits |= AC_RGP_FLUSH_PS_PARTIAL_FLUSH;
      }

      if (flags & AC_BARRIER_SYNC_CS) {
         ac_cmdbuf_event_write(V_028A90_CS_PARTIAL_FLUSH);
         *rgp_flush_bits |= AC_RGP_FLUSH_CS_PARTIAL_FLUSH;
      }

      ac_cmdbuf_end();
   }

   /* Ignore fields that only modify the behavior of other fields. */
   if (gcr_cntl & C_587_GL2_RANGE & C_587_SEQ & (gfx_level >= GFX12 ? ~0 : C_587_GL1_RANGE)) {
      ac_emit_cp_acquire_mem(cs, gfx_level, ip_type,
                             flags & AC_BARRIER_PFP_SYNC_ME ? V_581A_PREFETCH_PARSER : V_581A_MICRO_ENGINE,
                             gcr_cntl, context_roll, rgp_flush_bits);
   } else if (flags & AC_BARRIER_PFP_SYNC_ME && ip_type == AMD_IP_GFX) {
      /* We need to ensure that PFP waits as well. */
      ac_emit_cp_pfp_sync_me(cs, false);
      *rgp_flush_bits |= AC_RGP_FLUSH_PFP_SYNC_ME;
   }
}

static void
ac_gfx6_emit_barrier(struct ac_cmdbuf *cs, enum amd_gfx_level gfx_level,
                     enum amd_ip_type ip_type,
                     const struct ac_barrier_state *state,
                     bool *context_roll, enum ac_rgp_flush_bits *rgp_flush_bits)
{
   enum ac_barrier_flags flags =
      ac_get_reduced_barrier_flags(gfx_level, ip_type, state->flags);
   const uint32_t flush_cb_db = flags & (AC_BARRIER_SYNC_AND_INV_CB | AC_BARRIER_SYNC_AND_INV_DB);
   const bool is_mec = ip_type == AMD_IP_COMPUTE && gfx_level >= GFX7;
   uint32_t cp_coher_cntl = 0;

   if (!flags)
      return;

   ac_emit_common_barrier_flags(cs, gfx_level, ip_type, flags);

   /* GFX6 has a bug that it always flushes ICACHE and KCACHE if either bit is
    * set. An alternative way is to write SQC_CACHES, but that doesn't seem to
    * work reliably. Since the bug doesn't affect correctness (it only does
    * more work than necessary) and the performance impact is likely
    * negligible, there is no plan to add a workaround for it.
    */

   if (flags & AC_BARRIER_INV_ICACHE) {
      cp_coher_cntl |= S_0085F0_SH_ICACHE_ACTION_ENA(1);
      *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_ICACHE;
   }

   if (flags & AC_BARRIER_INV_SMEM) {
      cp_coher_cntl |= S_0085F0_SH_KCACHE_ACTION_ENA(1);
      *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_SMEM_L0;
   }

   if (gfx_level <= GFX8) {
      if (flags & AC_BARRIER_SYNC_AND_INV_CB) {
         cp_coher_cntl |= S_0085F0_CB_ACTION_ENA(1) |
                          S_0085F0_CB0_DEST_BASE_ENA(1) |
                          S_0085F0_CB1_DEST_BASE_ENA(1) |
                          S_0085F0_CB2_DEST_BASE_ENA(1) |
                          S_0085F0_CB3_DEST_BASE_ENA(1) |
                          S_0085F0_CB4_DEST_BASE_ENA(1) |
                          S_0085F0_CB5_DEST_BASE_ENA(1) |
                          S_0085F0_CB6_DEST_BASE_ENA(1) |
                          S_0085F0_CB7_DEST_BASE_ENA(1);

         /* Necessary for DCC */
         if (gfx_level == GFX8) {
            ac_emit_cp_release_mem(cs, gfx_level, ip_type,
                                   V_028A90_FLUSH_AND_INV_CB_DATA_TS, 0,
                                   EOP_DST_SEL_MEM, EOP_INT_SEL_NONE,
                                   EOP_DATA_SEL_DISCARD, 0, 0,
                                   state->eop_bug_va);
         }

         *rgp_flush_bits |= AC_RGP_FLUSH_FLUSH_CB | AC_RGP_FLUSH_INVAL_CB;
      }

      if (flags & AC_BARRIER_SYNC_AND_INV_DB) {
         cp_coher_cntl |= S_0085F0_DB_ACTION_ENA(1) | S_0085F0_DB_DEST_BASE_ENA(1);
         *rgp_flush_bits |= AC_RGP_FLUSH_FLUSH_DB | AC_RGP_FLUSH_INVAL_DB;
      }
   }

   ac_cmdbuf_begin(cs);

   /* Flush CMASK/FMASK/DCC. SURFACE_SYNC will wait for idle. */
   if (flags & AC_BARRIER_SYNC_AND_INV_CB_META)
      ac_cmdbuf_event_write(V_028A90_FLUSH_AND_INV_CB_META);

   /* Flush HTILE. SURFACE_SYNC will wait for idle. */
   if (flags & AC_BARRIER_SYNC_AND_INV_DB_META)
      ac_cmdbuf_event_write(V_028A90_FLUSH_AND_INV_DB_META);

   /* Wait for shader engines to go idle.
    * VS and PS waits are unnecessary if SURFACE_SYNC is going to wait
    * for everything including CB/DB cache flushes.
    *
    * GFX6-8: SURFACE_SYNC with CB_ACTION_ENA doesn't do anything if there are
    * no CB/DB bindings.  Reproducible with:
    * piglit/arb_framebuffer_no_attachments-atomic
    *
    * GFX9: The TS event is always written after full pipeline completion
    * regardless of CB/DB bindings.
    */
   if (gfx_level <= GFX8 || !flush_cb_db) {
      if (flags & AC_BARRIER_SYNC_PS) {
         ac_cmdbuf_event_write(V_028A90_PS_PARTIAL_FLUSH);
         *rgp_flush_bits |= AC_RGP_FLUSH_PS_PARTIAL_FLUSH;
      } else if (flags & AC_BARRIER_SYNC_VS) {
         ac_cmdbuf_event_write(V_028A90_VS_PARTIAL_FLUSH);
         *rgp_flush_bits |= AC_RGP_FLUSH_VS_PARTIAL_FLUSH;
      }
   }

   if (flags & AC_BARRIER_SYNC_CS) {
      ac_cmdbuf_event_write(V_028A90_CS_PARTIAL_FLUSH);
      *rgp_flush_bits |= AC_RGP_FLUSH_CS_PARTIAL_FLUSH;
   }

   ac_cmdbuf_end();

   /* GFX9: Wait for idle if we're flushing CB or DB. ACQUIRE_MEM doesn't
    * wait for idle on GFX9. We have to use a TS event.
    */
   if (gfx_level == GFX9 && flush_cb_db) {
      uint64_t va;
      unsigned tc_flags, cb_db_event;

      /* Set the CB/DB flush event. */
      switch (flush_cb_db) {
      case AC_BARRIER_SYNC_AND_INV_CB:
         cb_db_event = V_028A90_FLUSH_AND_INV_CB_DATA_TS;
         break;
      case AC_BARRIER_SYNC_AND_INV_DB:
         cb_db_event = V_028A90_FLUSH_AND_INV_DB_DATA_TS;
         break;
      default:
         /* both CB & DB */
         cb_db_event = V_028A90_CACHE_FLUSH_AND_INV_TS_EVENT;
      }

      /* These are the only allowed combinations. If you need to
       * do multiple operations at once, do them separately.
       * All operations that invalidate L2 also seem to invalidate
       * metadata. Volatile (VOL) and WC flushes are not listed here.
       *
       * TC    | TC_WB         = writeback & invalidate L2
       * TC    | TC_WB | TC_NC = writeback & invalidate L2 for MTYPE == NC
       *         TC_WB | TC_NC = writeback L2 for MTYPE == NC
       * TC            | TC_NC = invalidate L2 for MTYPE == NC
       * TC    | TC_MD         = writeback & invalidate L2 metadata (DCC, etc.)
       * TCL1                  = invalidate L1
       */
      tc_flags = 0;

      if (flags & AC_BARRIER_INV_L2_METADATA) {
         tc_flags = EVENT_TC_ACTION_ENA | EVENT_TC_MD_ACTION_ENA;
      }

      *rgp_flush_bits |= AC_RGP_FLUSH_FLUSH_CB | AC_RGP_FLUSH_INVAL_CB |
                         AC_RGP_FLUSH_FLUSH_DB | AC_RGP_FLUSH_INVAL_DB;

      /* Ideally flush L2 together with CB/DB. */
      if (flags & AC_BARRIER_INV_L2) {
         /* Writeback and invalidate everything in L2 & L1. */
         tc_flags = EVENT_TC_ACTION_ENA | EVENT_TC_WB_ACTION_ENA;

         /* Clear the flags. */
         flags &= ~(AC_BARRIER_INV_L2 | AC_BARRIER_WB_L2);

         *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_L2;
      }

      /* Do the flush (enqueue the event and wait for it). */
      assert(state->wait_mem_number);
      (*state->wait_mem_number)++;

      va = state->wait_mem_va;

      ac_emit_cp_release_mem(cs, gfx_level, ip_type, cb_db_event, tc_flags,
                             EOP_DST_SEL_MEM, EOP_INT_SEL_SEND_DATA_AFTER_WR_CONFIRM,
                             EOP_DATA_SEL_VALUE_32BIT, va,
                             *(state->wait_mem_number),
                             state->eop_bug_va);

      ac_emit_cp_wait_mem(cs, va, *(state->wait_mem_number), 0xffffffff, WAIT_REG_MEM_EQUAL);
   }

   /* VGT streamout state sync */
   if (flags & AC_BARRIER_VGT_STREAMOUT_SYNC) {
      ac_cmdbuf_begin(cs);
      ac_cmdbuf_event_write(V_028A90_VGT_STREAMOUT_SYNC);
      ac_cmdbuf_end();
   }

   /* GFX6-GFX8 only: When one of the CP_COHER_CNTL.DEST_BASE flags is set,
    * SURFACE_SYNC waits for idle, so it should be last.
    *
    * cp_coher_cntl should contain everything except TC flags at this point.
    *
    * GFX6-GFX7 don't support L2 write-back.
    */
   unsigned engine = flags & AC_BARRIER_PFP_SYNC_ME ? V_581A_PREFETCH_PARSER : V_581A_MICRO_ENGINE;

   if (flags & AC_BARRIER_INV_L2 || (gfx_level <= GFX7 && flags & AC_BARRIER_WB_L2)) {
      /* Invalidate L1 & L2. WB must be set on GFX8+ when TC_ACTION is set. */
      ac_emit_cp_acquire_mem(cs, gfx_level, ip_type, engine,
                             cp_coher_cntl | S_0085F0_TC_ACTION_ENA(1) | S_0085F0_TCL1_ACTION_ENA(1) |
                             S_0301F0_TC_WB_ACTION_ENA(gfx_level >= GFX8),
                             context_roll, rgp_flush_bits);

      *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_L2 | AC_RGP_FLUSH_INVAL_VMEM_L0;
   } else {
      /* L1 invalidation and L2 writeback must be done separately, because
       * both operations can't be done together.
       */
      if (flags & AC_BARRIER_WB_L2) {
         /* WB = write-back
          * NC = apply to non-coherent MTYPEs
          *      (i.e. MTYPE <= 1, which is what we use everywhere)
          *
          * WB doesn't work without NC.
          *
          * If we get here, the only flag that can't be executed together with
          * WB_L2 is VMEM cache invalidation.
          */
         const bool last_acquire_mem = !(flags & AC_BARRIER_INV_VMEM);

         ac_emit_cp_acquire_mem(cs, gfx_level, ip_type,
                                /* If this is not the last ACQUIRE_MEM, flush in ME.
                                 * We only want to synchronize with PFP in the last ACQUIRE_MEM. */
                                last_acquire_mem ? engine : V_581A_MICRO_ENGINE,
                                cp_coher_cntl | S_0301F0_TC_WB_ACTION_ENA(1) |
                                S_0301F0_TC_NC_ACTION_ENA(1),
                                context_roll, rgp_flush_bits);

         if (last_acquire_mem)
            flags &= ~AC_BARRIER_PFP_SYNC_ME;
         cp_coher_cntl = 0;

         *rgp_flush_bits |= AC_RGP_FLUSH_FLUSH_L2 | AC_RGP_FLUSH_INVAL_VMEM_L0;
      }

      if (flags & AC_BARRIER_INV_VMEM) {
         cp_coher_cntl |= S_0085F0_TCL1_ACTION_ENA(1);
         *rgp_flush_bits |= AC_RGP_FLUSH_INVAL_VMEM_L0;
      }

      /* If there are still some cache flags left. */
      if (cp_coher_cntl) {
         ac_emit_cp_acquire_mem(cs, gfx_level, ip_type, engine, cp_coher_cntl,
                                context_roll, rgp_flush_bits);
         flags &= ~AC_BARRIER_PFP_SYNC_ME;
      }

      /* This might be needed even without any cache flags, such as when doing
       * buffer stores to an index buffer.
       */
      if (flags & AC_BARRIER_PFP_SYNC_ME && !is_mec) {
         ac_emit_cp_pfp_sync_me(cs, false);
         *rgp_flush_bits |= AC_RGP_FLUSH_PFP_SYNC_ME;
      }
   }
}

void
ac_emit_barrier(struct ac_cmdbuf *cs, enum amd_gfx_level gfx_level,
                enum amd_ip_type ip_type, const struct ac_barrier_state *state,
                bool *context_roll, enum ac_rgp_flush_bits *rgp_flush_bits)
{
   if (gfx_level >= GFX10) {
      ac_gfx10_emit_barrier(cs, gfx_level, ip_type, state, context_roll, rgp_flush_bits);
   } else {
      ac_gfx6_emit_barrier(cs, gfx_level, ip_type, state, context_roll, rgp_flush_bits);
   }
}
