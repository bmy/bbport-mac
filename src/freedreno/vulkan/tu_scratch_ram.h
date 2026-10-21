/*
 * Copyright (c) Qualcomm Technologies, Inc. and/or its susidiaries.
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_SCRATCH_RAM_H
#define TU_SCRATCH_RAM_H

#include "tu_cs.h"

/**
 * Defines layout of CP scratch RAM for various use-cases, along with helpers
 * for accessing scratch RAM.
 */
union tu_scratch_ram {
   /**
    * For saving/restoring registers across store_3d_blit():
    */
   struct {
      uint32_t RB_CNTL;
      uint32_t RB_BUFFER_CNTL;
   } store_3d_blit;

   /**
    * Used by tu_dispatch() for unaligned indirect dispatch.
    */
   struct {
      uint32_t scratch0;
   } tu_dispatch;
};

#define tu_scratch(_field) ((struct tu_scratch_slot){ \
      offsetof(union tu_scratch_ram, _field) / 4      \
   })

inline void
tu_cs::scratch_to_reg(struct fd_reg_pair reg, struct tu_scratch_slot scratch, unsigned cnt)
{
   tu_pkt7(this, CP_SCRATCH_TO_REG, 1)
      .add(CP_SCRATCH_TO_REG_0(
         .reg = reg.reg,
         .scratch = scratch.slot,
         .cnt = cnt - 1,
      ));
}

inline void
tu_cs::reg_to_scratch(struct tu_scratch_slot scratch, struct fd_reg_pair reg, unsigned cnt)
{
   tu_pkt7(this, CP_REG_TO_SCRATCH, 1)
      .add(CP_REG_TO_SCRATCH_0(
         .reg = reg.reg,
         .scratch = scratch.slot,
         .cnt = cnt - 1,
      ));
}

inline void
tu_cs::scratch_write(struct tu_scratch_slot scratch, uint32_t *val, unsigned cnt)
{
   tu_pkt7 pkt(this, CP_SCRATCH_WRITE, cnt + 1);
   pkt.add(CP_SCRATCH_WRITE_0(
         .scratch = scratch.slot,
      ));
   for (unsigned i = 0; i < cnt; i++)
      pkt.add(val[i]);
}

/* not called anywhere, just for build time asserts */
static inline void
tu_scratch_ram_asserts(void)
{
   STATIC_ASSERT(sizeof(union tu_scratch_ram) <= 32);
}

#endif /* TU_SCRATCH_RAM_H*/
