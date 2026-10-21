/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#define PVR_BUILD_ARCH_ROGUE
#include "pvr_utrace.h"
#include "pvr_device.h"
#include "pvr_cmd_buffer.h"

#include "ds/pvr_driver_ds.h"
#include "ds/pvr_tracepoints.h"

#include <stdint.h>

/* Initialize u_trace support for device */
void pvr_device_utrace_init(struct pvr_device *device)
{
   if (!device)
      return;

   if (!device->ws)
      return;

   /* Initialize driver services */
   pvr_driver_ds_init();

   memset(&device->ds, 0, sizeof(device->ds));

   /* Initialize u_trace context */
   u_trace_context_init(&device->ds.trace_context,
                        &device->ds,
                        0, /* timestamp slot size (unused) */
                        0, /* max indirect data size (unused) */
                        NULL, /* create_buffer - not needed for markers-only */
                        NULL, /* destroy_buffer - not needed for markers-only */
                        NULL, /* record_ts - not needed for markers-only */
                        NULL, /* read_ts - not needed for markers-only */
                        NULL, /* capture_data - not needed for markers-only */
                        NULL, /* get_data - not needed for markers-only */
                        NULL); /* delete_submit - not needed for markers-only */

   /* PVR currently supports marker tracing only. */
   device->ds.trace_context.enabled_traces &= U_TRACE_TYPE_MARKERS;
}

void pvr_device_utrace_finish(struct pvr_device *device)
{
   /* Cleanup device */
   pvr_ds_device_fini(&device->ds);
}

/* Initialize trace for command buffer */
void pvr_cmd_buffer_utrace_init(struct pvr_cmd_buffer *cmd_buffer)
{
   if (!cmd_buffer)
      return;

   struct pvr_device *device = cmd_buffer->device;

   if (!device)
      return;

   u_trace_init(&cmd_buffer->trace, &device->ds.trace_context);
}

/* Cleanup trace for command buffer */
void pvr_cmd_buffer_utrace_fini(struct pvr_cmd_buffer *cmd_buffer)
{
   u_trace_fini(&cmd_buffer->trace);
}
