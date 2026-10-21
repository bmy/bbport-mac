/*
 * Copyright © 2026 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include <perfetto.h>

#include "c11/threads.h"
#include "util/macros.h"
#include "util/perf/u_perfetto.h"
#include "util/perf/u_perfetto_renderpass.h"
#include "util/timespec.h"
#include "util/u_process.h"

#include "v3dv_device.h"
#include "v3dv_tracepoints.h"
#include "v3dv_tracepoints_perfetto.h"
#include "v3dv_utrace_perfetto.h"

struct V3DVRenderpassIncrementalState {
   bool was_cleared = true;
};

struct V3DVRenderpassTraits : public perfetto::DefaultDataSourceTraits {
   using IncrementalStateType = V3DVRenderpassIncrementalState;
};

class V3DVRenderpassDataSource
    : public MesaRenderpassDataSource<V3DVRenderpassDataSource,
                                      V3DVRenderpassTraits> {};

PERFETTO_DECLARE_DATA_SOURCE_STATIC_MEMBERS(V3DVRenderpassDataSource);
PERFETTO_DEFINE_DATA_SOURCE_STATIC_MEMBERS(V3DVRenderpassDataSource);

static const char *
get_stage_name(enum v3dv_utrace_perfetto_stage stage)
{
   switch (stage) {
#define CASE(x)                                                             \
   case V3DV_UTRACE_PERFETTO_STAGE_##x:                                     \
      return #x
      CASE(JOB_CL);
      CASE(JOB_CSD);
      CASE(JOB_TFU);
      CASE(CPU_RESET_QUERIES);
      CASE(CPU_COPY_QUERY_RESULTS);
      CASE(CPU_CSD_INDIRECT);
      CASE(CPU_TIMESTAMP_QUERY);
      CASE(CMDBUF);
#undef CASE
   default:
      UNREACHABLE("bad stage");
   }
}

static enum v3dv_utrace_queue
get_queue_idx(enum v3dv_utrace_perfetto_stage stage)
{
   switch (stage) {
   case V3DV_UTRACE_PERFETTO_STAGE_JOB_CL:
      return V3DV_UTRACE_PERFETTO_QUEUE_CL;
   case V3DV_UTRACE_PERFETTO_STAGE_JOB_CSD:
      return V3DV_UTRACE_PERFETTO_QUEUE_CSD;
   case V3DV_UTRACE_PERFETTO_STAGE_JOB_TFU:
      return V3DV_UTRACE_PERFETTO_QUEUE_TFU;
   case V3DV_UTRACE_PERFETTO_STAGE_CPU_RESET_QUERIES:
   case V3DV_UTRACE_PERFETTO_STAGE_CPU_COPY_QUERY_RESULTS:
   case V3DV_UTRACE_PERFETTO_STAGE_CPU_CSD_INDIRECT:
   case V3DV_UTRACE_PERFETTO_STAGE_CPU_TIMESTAMP_QUERY:
      return V3DV_UTRACE_PERFETTO_QUEUE_CPU;
   case V3DV_UTRACE_PERFETTO_STAGE_CMDBUF:
      return V3DV_UTRACE_PERFETTO_QUEUE_DRIVER;
   default:
      UNREACHABLE("bad stage");
   }
}

static const char *
get_hw_queue_name(enum v3dv_utrace_queue queue_idx)
{
   switch (queue_idx) {
   case V3DV_UTRACE_PERFETTO_QUEUE_CL:
      return "CL";
   case V3DV_UTRACE_PERFETTO_QUEUE_CSD:
      return "CSD";
   case V3DV_UTRACE_PERFETTO_QUEUE_TFU:
      return "TFU";
   case V3DV_UTRACE_PERFETTO_QUEUE_CPU:
      return "CPU";
   case V3DV_UTRACE_PERFETTO_QUEUE_DRIVER:
      return "_Driver";
   default:
      return "unknown";
   }
}

static void
emit_interned_data_packet(struct v3dv_device *dev,
                          V3DVRenderpassDataSource::TraceContext &ctx,
                          uint64_t now)
{
   const struct v3dv_utrace_perfetto *utp = &dev->utrace.utp;

   auto packet = ctx.NewTracePacket();
   packet->set_timestamp(now);
   packet->set_sequence_flags(
      perfetto::protos::pbzero::TracePacket::SEQ_INCREMENTAL_STATE_CLEARED);

   auto interned_data = packet->set_interned_data();

   for (uint32_t i = 0; i < V3DV_UTRACE_PERFETTO_QUEUE_COUNT; i++) {
      char name[64];
      snprintf(name, sizeof(name), "%s-%s", util_get_process_name(),
               get_hw_queue_name((enum v3dv_utrace_queue)i));
      auto specs = interned_data->add_gpu_specifications();
      specs->set_iid(utp->queue_iids[i]);
      specs->set_name(name);
   }

   for (uint32_t i = 0; i < V3DV_UTRACE_PERFETTO_STAGE_COUNT; i++) {
      auto specs = interned_data->add_gpu_specifications();
      specs->set_iid(utp->stage_iids[i]);
      specs->set_name(get_stage_name((enum v3dv_utrace_perfetto_stage)i));
   }
}

static uint64_t
get_gpu_time_ns(struct v3dv_device *dev)
{
   /* v3d has no free-running GPU clock register, "GPU time" here is a CPU
    * clock sample the kernel takes via DRM_V3D_EXT_ID_CPU_TIMESTAMP_QUERY,
    * already returned in nanoseconds.
    * For clock-sync purposes we can just use the current CPU boottime.
    */
   struct timespec ts;
   clock_gettime(CLOCK_BOOTTIME, &ts);
   return (uint64_t)ts.tv_sec * NSEC_PER_SEC + ts.tv_nsec;
}

static void
emit_clock_snapshot_packet(struct v3dv_device *dev,
                           V3DVRenderpassDataSource::TraceContext &ctx)
{
   const struct v3dv_utrace_perfetto *utp = &dev->utrace.utp;
   const uint64_t gpu_ns = get_gpu_time_ns(dev);
   const uint64_t cpu_ns = perfetto::base::GetBootTimeNs().count();

   MesaRenderpassDataSource<V3DVRenderpassDataSource, V3DVRenderpassTraits>::
      EmitClockSync(ctx, cpu_ns, gpu_ns,
                    perfetto::protos::pbzero::BUILTIN_CLOCK_BOOTTIME,
                    utp->gpu_clock_id);
}

static void
emit_setup_packets(struct v3dv_device *dev,
                   V3DVRenderpassDataSource::TraceContext &ctx)
{
   struct v3dv_utrace_perfetto *utp = &dev->utrace.utp;

   const uint64_t now = perfetto::base::GetBootTimeNs().count();

   auto state = ctx.GetIncrementalState();
   if (state->was_cleared) {
      emit_interned_data_packet(dev, ctx, now);

      state->was_cleared = false;
      utp->next_clock_snapshot = 0;
   }

   if (now >= utp->next_clock_snapshot) {
      emit_clock_snapshot_packet(dev, ctx);

      utp->next_clock_snapshot = now + NSEC_PER_SEC;
   }
}

static struct v3dv_utrace_perfetto_event *
begin_event(struct v3dv_device *dev, enum v3dv_utrace_perfetto_stage stage)
{
   struct v3dv_utrace_perfetto *utp = &dev->utrace.utp;
   uint32_t queue_idx = get_queue_idx(stage);
   struct v3dv_utrace_perfetto_queue *queue = &utp->queues[queue_idx];

   if (queue->stack_depth >= V3DV_UTRACE_PERFETTO_STACK_DEPTH) {
      PERFETTO_ELOG("queue %d stage %d too deep", queue_idx, stage);
      return NULL;
   }

   struct v3dv_utrace_perfetto_event *ev = &queue->stack[queue->stack_depth++];
   ev->stage = stage;
   return ev;
}

static struct v3dv_utrace_perfetto_event *
end_event(struct v3dv_device *dev, enum v3dv_utrace_perfetto_stage stage)
{
   struct v3dv_utrace_perfetto *utp = &dev->utrace.utp;
   uint32_t queue_idx = get_queue_idx(stage);
   struct v3dv_utrace_perfetto_queue *queue = &utp->queues[queue_idx];

   if (!queue->stack_depth)
      return NULL;

   if (queue->stack_depth > V3DV_UTRACE_PERFETTO_STACK_DEPTH)
      return NULL;

   struct v3dv_utrace_perfetto_event *ev = &queue->stack[--queue->stack_depth];
   assert(ev->stage == stage);
   return ev;
}

static void
v3dv_utrace_perfetto_begin_event(struct v3dv_device *dev,
                                 enum v3dv_utrace_perfetto_stage stage,
                                 uint64_t ts_ns)
{
   struct v3dv_utrace_perfetto_event *ev = begin_event(dev, stage);
   if (!ev)
      return;

   ev->begin_ns = ts_ns;
}

static void
v3dv_utrace_perfetto_end_event(
   struct v3dv_device *dev,
   enum v3dv_utrace_perfetto_stage stage,
   uint64_t ts_ns,
   std::function<void(perfetto::protos::pbzero::GpuRenderStageEvent *)>
      emit_event_extra)
{
   const struct v3dv_utrace_perfetto_event *ev = end_event(dev, stage);
   if (!ev)
      return;

   uint32_t queue_idx = get_queue_idx(stage);

   V3DVRenderpassDataSource::Trace(
      [=](V3DVRenderpassDataSource::TraceContext ctx) {
         struct v3dv_utrace_perfetto *utp = &dev->utrace.utp;

         emit_setup_packets(dev, ctx);

         auto packet = ctx.NewTracePacket();
         packet->set_timestamp(ev->begin_ns);
         packet->set_timestamp_clock_id(utp->gpu_clock_id);

         auto event = packet->set_gpu_render_stage_event();
         event->set_event_id(utp->event_id++);
         event->set_duration(ts_ns - ev->begin_ns);
         event->set_hw_queue_iid(utp->queue_iids[queue_idx]);
         event->set_stage_iid(utp->stage_iids[stage]);
         event->set_context(utp->device_id);

         emit_event_extra(event);
      }
   );
}

#define V3DV_UTRACE_PERFETTO_PROCESS_EVENT(tp, stage)                      \
   void v3dv_utrace_perfetto_begin_##tp(                                   \
      struct v3dv_device *dev, uint64_t ts_ns, uint16_t tp_idx,            \
      const void *flush_data, const struct trace_begin_##tp *payload,      \
      const void *indirect_data)                                           \
   {                                                                       \
      v3dv_utrace_perfetto_begin_event(                                    \
         dev, V3DV_UTRACE_PERFETTO_STAGE_##stage, ts_ns);                  \
   }                                                                       \
                                                                           \
   void v3dv_utrace_perfetto_end_##tp(                                     \
      struct v3dv_device *dev, uint64_t ts_ns, uint16_t tp_idx,            \
      const void *flush_data, const struct trace_end_##tp *payload,        \
      const void *indirect_data)                                           \
   {                                                                       \
      auto emit_event_extra =                                              \
         [=](perfetto::protos::pbzero::GpuRenderStageEvent *event) {       \
            trace_payload_as_extra_end_##tp(event, payload, indirect_data);\
         };                                                                \
      v3dv_utrace_perfetto_end_event(                                      \
         dev, V3DV_UTRACE_PERFETTO_STAGE_##stage, ts_ns, emit_event_extra);\
   }

V3DV_UTRACE_PERFETTO_PROCESS_EVENT(job_cl, JOB_CL)
V3DV_UTRACE_PERFETTO_PROCESS_EVENT(job_csd, JOB_CSD)
V3DV_UTRACE_PERFETTO_PROCESS_EVENT(job_tfu, JOB_TFU)
V3DV_UTRACE_PERFETTO_PROCESS_EVENT(job_cpu_reset_queries, CPU_RESET_QUERIES)
V3DV_UTRACE_PERFETTO_PROCESS_EVENT(job_cpu_copy_query_results, CPU_COPY_QUERY_RESULTS)
V3DV_UTRACE_PERFETTO_PROCESS_EVENT(job_cpu_csd_indirect, CPU_CSD_INDIRECT)
V3DV_UTRACE_PERFETTO_PROCESS_EVENT(job_cpu_timestamp_query, CPU_TIMESTAMP_QUERY)
V3DV_UTRACE_PERFETTO_PROCESS_EVENT(cmdbuf, CMDBUF)

static uint32_t
get_gpu_clock_id(void)
{
   return _mesa_hash_string("org.freedesktop.mesa.broadcom") | 0x80000000;
}

static void
register_data_source(void)
{
   perfetto::DataSourceDescriptor dsd;
   dsd.set_name("gpu.renderstages.broadcom");
   V3DVRenderpassDataSource::Register(dsd);
}

void
v3dv_utrace_perfetto_init(struct v3dv_device *dev, uint32_t queue_count)
{
   struct v3dv_utrace_perfetto *utp = &dev->utrace.utp;

   if (queue_count > V3DV_UTRACE_PERFETTO_QUEUE_COUNT) {
      assert(!"V3DV_UTRACE_PERFETTO_QUEUE_COUNT too small");
      return;
   }

   utp->gpu_clock_id = get_gpu_clock_id();
   utp->device_id = (uintptr_t)dev;

   uint64_t next_iid = 1;
   for (uint32_t i = 0; i < ARRAY_SIZE(utp->queue_iids); i++)
      utp->queue_iids[i] = next_iid++;
   for (uint32_t i = 0; i < ARRAY_SIZE(utp->stage_iids); i++)
      utp->stage_iids[i] = next_iid++;

   util_perfetto_init();

   static once_flag register_ds_once = ONCE_FLAG_INIT;
   call_once(&register_ds_once, register_data_source);
}
