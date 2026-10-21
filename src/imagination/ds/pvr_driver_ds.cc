/*
 * Copyright © 2026 Imagination Technologies Ltd.
 *
 * SPDX-License-Identifier: MIT
 */

#include "pvr_driver_ds.h"

#include <stdarg.h>
#include <stdio.h>

#include "pvr_tracepoints.h"
#include "util/u_debug.h"

#ifdef HAVE_GPUVIS
/*
 * GPU Tracing via ftrace markers
 * The gpuvis backend is a ftrace marker writer that writes trace events
 * to trace_marker on Linux.
 *
 * This driver supports emitting trace events to the Linux ftrace subsystem
 * when compiled with -Dgpuvis=true. The "gpuvis backend" writes trace markers
 * to /sys/kernel/tracing/trace_marker, which can be captured and visualized.
 * The /sys/kernel/tracing/trace_pipe can be used to read the u_trace events in
 * real-time.
 */
#   include "util/perf/u_gpuvis.h"
#endif

static void pvr_driver_ds_init_once(void)
{
#ifdef HAVE_GPUVIS
   util_gpuvis_init();
#endif
}

static once_flag pvr_driver_ds_once_flag = ONCE_FLAG_INIT;

void pvr_driver_ds_init(void)
{
   call_once(&pvr_driver_ds_once_flag, pvr_driver_ds_init_once);

   pvr_gpu_tracepoint_config_variable();
}

void pvr_ds_device_fini(struct pvr_ds_device *device)
{
   u_trace_context_fini(&device->trace_context);
}

/* FINISHME: CS markers are not emitted here (see tp_markers in
 * src/util/perf/u_trace.py); this is CPU tracing only for now.
 */

static void pvr_ds_begin_marker(const char *fmt, va_list ap)
{
#ifdef HAVE_GPUVIS
   char buf[256];
   vsnprintf(buf, sizeof(buf), fmt, ap);
   util_gpuvis_begin(buf);
#endif
}

static void pvr_ds_end_marker(void)
{
#ifdef HAVE_GPUVIS
   util_gpuvis_end();
#endif
}

extern "C" {

#define PVR_DS_BEGIN_MARKER_CB(name)                              \
   void pvr_ds_begin_##name(UNUSED struct u_trace_context *utctx, \
                            UNUSED void *cs,                      \
                            UNUSED const char *fmt,               \
                            ...)                                  \
   {                                                              \
      va_list ap;                                                 \
      va_start(ap, fmt);                                          \
      pvr_ds_begin_marker(fmt, ap);                               \
      va_end(ap);                                                 \
   }

#define PVR_DS_END_MARKER_CB(name)                              \
   void pvr_ds_end_##name(UNUSED struct u_trace_context *utctx, \
                          UNUSED void *cs,                      \
                          UNUSED const char *fmt,               \
                          ...)                                  \
   {                                                            \
      pvr_ds_end_marker();                                      \
   }

PVR_DS_BEGIN_MARKER_CB(compute)
PVR_DS_END_MARKER_CB(compute)
PVR_DS_BEGIN_MARKER_CB(draw)
PVR_DS_END_MARKER_CB(draw)
PVR_DS_BEGIN_MARKER_CB(transfer)
PVR_DS_END_MARKER_CB(transfer)

#undef PVR_DS_BEGIN_MARKER_CB
#undef PVR_DS_END_MARKER_CB

} /* extern "C" */
