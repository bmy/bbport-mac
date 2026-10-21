/*
 * Copyright © 2026 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#ifndef V3DV_UTRACE_PERFETTO_H
#define V3DV_UTRACE_PERFETTO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define V3DV_UTRACE_PERFETTO_STACK_DEPTH 8

struct v3dv_device;

enum v3dv_utrace_queue {
   V3DV_UTRACE_PERFETTO_QUEUE_CL = 0,
   V3DV_UTRACE_PERFETTO_QUEUE_CSD,
   V3DV_UTRACE_PERFETTO_QUEUE_TFU,
   V3DV_UTRACE_PERFETTO_QUEUE_CPU,
   V3DV_UTRACE_PERFETTO_QUEUE_DRIVER, /* dedicated non-hw queue */
   V3DV_UTRACE_PERFETTO_QUEUE_COUNT,
};

/* FIXME: Job CL is only one stage despite having a separate render CL and bin
 * CL in the driver because the kernel does not expose a sync for "bin
 * finished" to userspace yet.
 */
enum v3dv_utrace_perfetto_stage {
   V3DV_UTRACE_PERFETTO_STAGE_JOB_CL,
   V3DV_UTRACE_PERFETTO_STAGE_JOB_CSD,
   V3DV_UTRACE_PERFETTO_STAGE_JOB_TFU,

   /* these are all CPU jobs and go on V3DV_UTRACE_PERFETTO_QUEUE_CPU */
   V3DV_UTRACE_PERFETTO_STAGE_CPU_RESET_QUERIES,
   V3DV_UTRACE_PERFETTO_STAGE_CPU_COPY_QUERY_RESULTS,
   V3DV_UTRACE_PERFETTO_STAGE_CPU_CSD_INDIRECT,
   V3DV_UTRACE_PERFETTO_STAGE_CPU_TIMESTAMP_QUERY,

   /* This is not a real job, it is used to track all the jobs
    * in a single command buffer,
    */
   V3DV_UTRACE_PERFETTO_STAGE_CMDBUF,
   V3DV_UTRACE_PERFETTO_STAGE_COUNT,
};

struct v3dv_utrace_perfetto_event {
   enum v3dv_utrace_perfetto_stage stage;
   uint64_t begin_ns;
};

struct v3dv_utrace_perfetto_queue {
   struct v3dv_utrace_perfetto_event stack[V3DV_UTRACE_PERFETTO_STACK_DEPTH];
   uint32_t stack_depth;
};

struct v3dv_utrace_perfetto {
   uint32_t gpu_clock_id;
   uint64_t device_id;

   uint64_t queue_iids[V3DV_UTRACE_PERFETTO_QUEUE_COUNT];
   uint64_t stage_iids[V3DV_UTRACE_PERFETTO_STAGE_COUNT];

   uint64_t next_clock_snapshot;
   uint64_t event_id;

   struct v3dv_utrace_perfetto_queue queues[V3DV_UTRACE_PERFETTO_QUEUE_COUNT];
};

#ifdef HAVE_PERFETTO

void v3dv_utrace_perfetto_init(struct v3dv_device *dev, uint32_t queue_count);

#else /* HAVE_PERFETTO */

static inline void
v3dv_utrace_perfetto_init(struct v3dv_device *dev, uint32_t queue_count)
{
}

#endif /* HAVE_PERFETTO */

#ifdef __cplusplus
}
#endif

#endif /* V3DV_UTRACE_PERFETTO_H */
