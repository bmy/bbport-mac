/*
 * Copyright 2024 Advanced Micro Devices, Inc.
 * Copyright 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef AC_BARRIER_H
#define AC_BARRIER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "amd_family.h"
#include "ac_rgp.h"

struct ac_cmdbuf;

#ifdef __cplusplus
extern "C" {
#endif

enum ac_barrier_flags {
   /* Instruction cache. */
   AC_BARRIER_INV_ICACHE = 1u << 0,

   /* Scalar cache. (GFX6-9: scalar L1; GFX10+: scalar L0)
    * GFX10: This also invalidates the L1 shader array cache.
    */
   AC_BARRIER_INV_SMEM = 1u << 1,

   /* Vector cache. (GFX6-9: vector L1; GFX10+: vector L0)
    * GFX10: This also invalidates the L1 shader array cache.
    */
   AC_BARRIER_INV_VMEM = 1u << 2,

   /* L2 cache + L2 metadata cache writeback & invalidate.
    * GFX6-8: Used by shaders only. GFX9+: Used by everything.
    */
   AC_BARRIER_INV_L2 = 1u << 3,

   /* L2 writeback (write dirty L2 lines to memory for non-L2 clients).
    * Only used for coherency with non-L2 clients like CB, DB, CP on GFX6-8.
    * GFX6-7 will do complete invalidation because the writeback is unsupported.
    */
   AC_BARRIER_WB_L2 = 1u << 4,

   /* Invalidate the metadata cache. To be used when the DCC/HTILE metadata
    * changed and we want to read an image from shaders.
    */
   AC_BARRIER_INV_L2_METADATA = 1u << 5,

   /* CB caches. */
   AC_BARRIER_SYNC_AND_INV_CB = 1u << 6,
   AC_BARRIER_SYNC_AND_INV_CB_META = 1u << 7,

   /* DB caches. */
   AC_BARRIER_SYNC_AND_INV_DB = 1u << 8,
   AC_BARRIER_SYNC_AND_INV_DB_META = 1u << 9,

   /* PFP waits for ME to finish. */
   AC_BARRIER_PFP_SYNC_ME = 1u << 10,

   /* VGT related. */
   AC_BARRIER_VGT_FLUSH = 1u << 11,
   AC_BARRIER_VGT_STREAMOUT_SYNC = 1u << 12,

   /* Shaders related. */
   AC_BARRIER_SYNC_VS = 1u << 13,
   AC_BARRIER_SYNC_PS = 1u << 14,
   AC_BARRIER_SYNC_CS = 1u << 15,

   /* Pipeline stats events */
   AC_BARRIER_PIPELINESTAT_START = 1u << 16,
   AC_BARRIER_PIPELINESTAT_STOP = 1u << 17,
};

#define AC_BARRIER_ALL_COMPUTE \
   (AC_BARRIER_INV_ICACHE | AC_BARRIER_INV_SMEM | AC_BARRIER_INV_VMEM | \
    AC_BARRIER_INV_L2 | AC_BARRIER_WB_L2 | AC_BARRIER_SYNC_CS)

/* PWS (Pixel Wait Sync) acquire point, i.e. the pipeline stage at which a
 * GFX11+ PWS ACQUIRE waits for a preceding RELEASE. The acquire point is
 * derived from the barrier destination stage so the wait can be deferred to
 * the latest legal pipeline stage.
 */
enum ac_pws_acquire_point {
   AC_PWS_ACQUIRE_POINT_NONE = 0,
   AC_PWS_ACQUIRE_POINT_PRE_DEPTH, /* Wait just before depth/fragment work. */
   AC_PWS_ACQUIRE_POINT_ME,        /* Wait at the CP micro-engine. */
   AC_PWS_ACQUIRE_POINT_PFP,       /* Wait at the CP prefetch parser (frontend). */
};

struct ac_barrier_state {
   enum ac_barrier_flags flags;
   enum ac_pws_acquire_point pws_acquire_point; /* GFX11+ */

   uint64_t wait_mem_va;
   uint32_t *wait_mem_number;
   uint64_t eop_bug_va;
};

void
ac_emit_barrier(struct ac_cmdbuf *cs, enum amd_gfx_level gfx_level,
                enum amd_ip_type ip_type, const struct ac_barrier_state *state,
                bool *context_roll, enum ac_rgp_flush_bits *rgp_flush_bits);

#ifdef __cplusplus
}
#endif

#endif /* AC_BARRIER_H */
