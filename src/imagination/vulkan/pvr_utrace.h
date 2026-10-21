/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef PVR_UTRACE_H
#define PVR_UTRACE_H

#include <stdint.h>

#include "util/perf/u_trace.h"
#include "ds/pvr_driver_ds.h"
#include "ds/pvr_tracepoints.h"

#ifdef __cplusplus
extern "C" {
#endif

struct pvr_device;
struct pvr_queue;
struct pvr_cmd_buffer;

/* Initialize u_trace support for device */
void pvr_device_utrace_init(struct pvr_device *device);

/* Cleanup u_trace support */
void pvr_device_utrace_finish(struct pvr_device *device);

/* Initialize command buffer trace */
void pvr_cmd_buffer_utrace_init(struct pvr_cmd_buffer *cmd_buffer);

/* Cleanup command buffer trace */
void pvr_cmd_buffer_utrace_fini(struct pvr_cmd_buffer *cmd_buffer);

#define PVR_TRACE_ENABLED(_cmd_buffer) ((_cmd_buffer)->trace.utctx)

#define PVR_TRACE_BEGIN_COMPUTE(_cmd_buffer)                            \
   do {                                                                 \
      if (PVR_TRACE_ENABLED(_cmd_buffer))                               \
         trace_pvr_begin_compute(&(_cmd_buffer)->trace,                 \
                                 NULL,                                  \
                                 (uintptr_t) & (_cmd_buffer)->vk.base); \
   } while (0)

#define PVR_TRACE_END_COMPUTE(_cmd_buffer)                   \
   do {                                                      \
      if (PVR_TRACE_ENABLED(_cmd_buffer))                    \
         trace_pvr_end_compute(&(_cmd_buffer)->trace, NULL); \
   } while (0)

#define PVR_TRACE_BEGIN_DRAW(_cmd_buffer, _op)                      \
   do {                                                             \
      if (PVR_TRACE_ENABLED(_cmd_buffer))                           \
         trace_pvr_begin_draw(&(_cmd_buffer)->trace,                \
                              NULL,                                 \
                              (uintptr_t) & (_cmd_buffer)->vk.base, \
                              (_op));                               \
   } while (0)

#define PVR_TRACE_END_DRAW(_cmd_buffer)                   \
   do {                                                   \
      if (PVR_TRACE_ENABLED(_cmd_buffer))                 \
         trace_pvr_end_draw(&(_cmd_buffer)->trace, NULL); \
   } while (0)

#define PVR_TRACE_BEGIN_TRANSFER(_cmd_buffer, _op)                      \
   do {                                                                 \
      if (PVR_TRACE_ENABLED(_cmd_buffer))                               \
         trace_pvr_begin_transfer(&(_cmd_buffer)->trace,                \
                                  NULL,                                 \
                                  (uintptr_t) & (_cmd_buffer)->vk.base, \
                                  (_op));                               \
   } while (0)

#define PVR_TRACE_END_TRANSFER(_cmd_buffer)                   \
   do {                                                       \
      if (PVR_TRACE_ENABLED(_cmd_buffer))                     \
         trace_pvr_end_transfer(&(_cmd_buffer)->trace, NULL); \
   } while (0)

#ifdef __cplusplus
}
#endif

#endif /* PVR_UTRACE_H */
