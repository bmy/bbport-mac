/*
 * Copyright 2026 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#ifndef V3DV_UTRACE_H
#define V3DV_UTRACE_H

#include "v3dv_device.h"

struct v3dv_utrace_buffer {
   struct v3dv_bo *bo;
   uint32_t *syncs; /* one syncobj handle per uint64_t timestamp slot, 0 = none */
   bool *immediate; /* true if this slot was written directly (no GPU wait expected) */
   uint32_t count;
};

void v3dv_utrace_context_init(struct v3dv_device *device);
void v3dv_utrace_context_fini(struct v3dv_device *device);

#endif /* V3DV_UTRACE_H */
