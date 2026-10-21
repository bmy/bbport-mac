/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PVR_DRIVER_DS_H
#define PVR_DRIVER_DS_H

#include "util/macros.h"
#include "util/perf/u_trace.h"

#ifdef __cplusplus
extern "C" {
#endif

enum pvr_ds_tracepoint_flags {
   /**
    * Whether the tracepoint's timestamp must be recorded with an
    * end-of-pipe timestamp.
    */
   PVR_DS_TRACEPOINT_FLAG_END_OF_PIPE = BITFIELD_BIT(0),
   /**
    * Whether this tracepoint's timestamp is recorded on the compute pipeline.
    */
   PVR_DS_TRACEPOINT_FLAG_END_CS = BITFIELD_BIT(1),
   /**
    * Whether this tracepoint doesn't generate a timestamp but instead repeats
    * the last one.
    */
   PVR_DS_TRACEPOINT_FLAG_REPEAT_LAST = BITFIELD_BIT(2),
};

enum pvr_transfer_op {
   PVR_TRANSFER_OP_BLIT_IMAGE,
   PVR_TRANSFER_OP_COPY_IMAGE,
   PVR_TRANSFER_OP_COPY_BUFFER_TO_IMAGE,
   PVR_TRANSFER_OP_COPY_IMAGE_TO_BUFFER,
   PVR_TRANSFER_OP_COPY_BUFFER,
};

static inline const char *pvr_transfer_op_to_str(enum pvr_transfer_op op)
{
   switch (op) {
   case PVR_TRANSFER_OP_BLIT_IMAGE:
      return "blit_image";
   case PVR_TRANSFER_OP_COPY_IMAGE:
      return "copy_image";
   case PVR_TRANSFER_OP_COPY_BUFFER_TO_IMAGE:
      return "copy_buf_to_image";
   case PVR_TRANSFER_OP_COPY_IMAGE_TO_BUFFER:
      return "copy_image_to_buf";
   case PVR_TRANSFER_OP_COPY_BUFFER:
      return "copy_buffer";
   }
   return "unknown";
}

enum pvr_draw_op {
   PVR_DRAW,
   PVR_DRAW_INDEXED,
   PVR_DRAW_INDIRECT,
   PVR_DRAW_INDEXED_INDIRECT,
};

static inline const char *pvr_draw_op_to_str(enum pvr_draw_op op)
{
   switch (op) {
   case PVR_DRAW:
      return "draw";
   case PVR_DRAW_INDEXED:
      return "draw_indexed";
   case PVR_DRAW_INDIRECT:
      return "draw_indirect";
   case PVR_DRAW_INDEXED_INDIRECT:
      return "draw_indexed_indirect";
   }
   return "unknown";
}

void pvr_driver_ds_init(void);

struct pvr_ds_device {
   struct u_trace_context trace_context;
};

void pvr_ds_device_fini(struct pvr_ds_device *device);

#ifdef __cplusplus
}
#endif

#endif /* PVR_DRIVER_DS_H */
