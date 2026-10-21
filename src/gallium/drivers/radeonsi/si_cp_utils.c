/*
 * Copyright 2024 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "si_pipe.h"
#include "gfx/si_gfx.h"
#include "ac_cmdbuf_cp.h"

void si_cp_release_acquire_mem_pws(struct si_context *sctx, struct radeon_cmdbuf *cs,
                                   unsigned event_type, unsigned gcr_cntl, unsigned stage_sel,
                                   unsigned sqtt_flush_flags)
{
   ac_emit_cp_release_mem_pws(&cs->current, sctx->gfx_level,
                              sctx->is_gfx_queue ? AMD_IP_GFX : AMD_IP_COMPUTE,
                              event_type, gcr_cntl);

   if (unlikely(sctx->sqtt_enabled))
      si_sqtt_describe_barrier_start(sctx, cs);

   ac_emit_cp_acquire_mem_pws(&cs->current, sctx->gfx_level,
                              sctx->is_gfx_queue ? AMD_IP_GFX : AMD_IP_COMPUTE,
                              event_type, stage_sel, 0, gcr_cntl);

   if (unlikely(sctx->sqtt_enabled))
      si_sqtt_describe_barrier_end(sctx, cs, sqtt_flush_flags);
}
