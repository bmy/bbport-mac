/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based on si_state.c
 * Copyright © 2015 Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "radv_cs.h"
#include "tools/radv_debug.h"
#include "tools/radv_sqtt.h"
#include "radv_buffer.h"
#include "radv_sdma.h"
#include "radv_shader.h"
#include "sid.h"

#include "ac_barrier.h"

void
radv_cs_emit_write_event_eop(struct radv_cmd_stream *cs, enum amd_gfx_level gfx_level, unsigned event,
                             unsigned event_flags, unsigned dst_sel, unsigned int_sel, unsigned data_sel, uint64_t va,
                             uint32_t new_fence, uint64_t gfx9_eop_bug_va)
{
   assert(cs->hw_ip == AMD_IP_GFX || cs->hw_ip == AMD_IP_COMPUTE);

   /* The EOP bug is specific to GFX9. Though, RadeonSI also implements it for GFX6-8 but it
    * shouldn't be necessary because it's using SURFACE_SYNC to flush L2. See
    * waEventWriteEopPrematureL2Inv in PAL.
    */
   const uint64_t eop_bug_va = gfx_level >= GFX9 ? gfx9_eop_bug_va : va;

   ac_emit_cp_release_mem(cs->b, gfx_level, cs->hw_ip, event, event_flags, dst_sel, int_sel, data_sel, va, new_fence,
                          eop_bug_va);
}

void
radv_cs_emit_cache_flush(struct radeon_winsys *ws, struct radv_cmd_stream *cs, enum amd_gfx_level gfx_level,
                         uint32_t *flush_cnt, uint64_t flush_va, enum ac_barrier_flags flush_bits,
                         enum ac_rgp_flush_bits *rgp_flush_bits, enum ac_pws_acquire_point pws_acquire_point,
                         uint64_t gfx9_eop_bug_va)
{
   /* TODO: Stop setting this barrier flags. */
   if (gfx_level >= GFX10)
      flush_bits &= ~AC_BARRIER_SYNC_AND_INV_DB_META;

   if (gfx_level >= GFX12)
      flush_bits &= ~AC_BARRIER_INV_L2_METADATA;

   struct ac_barrier_state barrier = {
      .flags = flush_bits,
      .pws_acquire_point = pws_acquire_point,
      .wait_mem_va = flush_va,
      .wait_mem_number = flush_cnt,
      .eop_bug_va = gfx9_eop_bug_va,
   };

   radeon_check_space(ws, cs->b, 128);

   ac_emit_barrier(cs->b, gfx_level, cs->hw_ip, &barrier, NULL, rgp_flush_bits);
}

void
radv_init_cmd_stream(const struct radv_device *device, struct radv_cmd_stream *cs, const enum amd_ip_type ip_type)
{
   const struct radv_physical_device *pdev = radv_device_physical(device);

   cs->buffered_sh_regs.num = 0;
   cs->hw_ip = ip_type;

   ac_init_tracked_regs(&cs->tracked_regs, &pdev->info, false);
}

VkResult
radv_create_cmd_stream(const struct radv_device *device, const enum amd_ip_type ip_type, const bool is_secondary,
                       struct radv_cmd_stream **cs_out)
{
   struct radeon_winsys *ws = device->ws;
   struct radv_cmd_stream *cs;

   cs = malloc(sizeof(*cs));
   if (!cs)
      return VK_ERROR_OUT_OF_HOST_MEMORY;

   radv_init_cmd_stream(device, cs, ip_type);

   cs->b = ws->cs_create(ws, ip_type, is_secondary);
   if (!cs->b) {
      free(cs);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }

   *cs_out = cs;
   return VK_SUCCESS;
}

void
radv_reset_cmd_stream(const struct radv_device *device, struct radv_cmd_stream *cs)
{
   struct radeon_winsys *ws = device->ws;

   radv_init_cmd_stream(device, cs, cs->hw_ip);

   ws->cs_reset(cs->b);
}

VkResult
radv_finalize_cmd_stream(const struct radv_device *device, struct radv_cmd_stream *cs)
{
   struct radeon_winsys *ws = device->ws;

   return ws->cs_finalize(cs->b);
}

void
radv_destroy_cmd_stream(const struct radv_device *device, struct radv_cmd_stream *cs)
{
   struct radeon_winsys *ws = device->ws;

   ws->cs_destroy(cs->b);
   free(cs);
}
