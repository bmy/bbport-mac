/*
 * Copyright 2026 Igalia S.L.
 * SPDX-License-Identifier: MIT
 */

#include "v3dv_utrace.h"
#include "v3dv_cmd_buffer.h"
#include "broadcom/common/v3d_submit_util.h"
#include "util/macros.h"
#include "util/perf/u_trace.h"
#include "vk_object.h"
#include <xf86drm.h>

static void *
create_buffer(struct u_trace_context *utctx, uint64_t size_b)
{
   struct v3dv_device *device =
      container_of(utctx, struct v3dv_device, utrace.utrace_ctx);

   struct v3dv_utrace_buffer *buf =
      vk_zalloc(&device->vk.alloc, sizeof(*buf), 8,
                VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!buf) {
      mesa_loge("v3dv utrace: buffer creation failed");
      return NULL;
   }

   buf->count = size_b / sizeof(uint64_t);
   buf->syncs = vk_zalloc(&device->vk.alloc, buf->count * sizeof(uint32_t), 8,
                          VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!buf->syncs) {
      vk_free(&device->vk.alloc, buf);
      mesa_loge("v3dv utrace: buffer creation failed");
      return NULL;
   }

   buf->immediate = vk_zalloc(&device->vk.alloc, buf->count * sizeof(bool), 8,
                              VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!buf->immediate) {
      vk_free(&device->vk.alloc, buf->syncs);
      vk_free(&device->vk.alloc, buf);
      mesa_loge("v3dv utrace: buffer creation failed");
      return NULL;
   }

   buf->bo = v3dv_bo_alloc(device, size_b, "utrace", true, VK_OBJECT_TYPE_UNKNOWN,
                           vk_object_to_u64_handle(&device->vk.base));
   if (!buf->bo) {
      vk_free(&device->vk.alloc, buf->immediate);
      vk_free(&device->vk.alloc, buf->syncs);
      vk_free(&device->vk.alloc, buf);
      mesa_loge("v3dv utrace: buffer object alloc failed");
      return NULL;
   }

   /* no mesa_loge here since v3dv_bo_map includes one already */
   if (!v3dv_bo_map(device, buf->bo, size_b)) {
      v3dv_bo_free(device, buf->bo, 0);
      vk_free(&device->vk.alloc, buf->immediate);
      vk_free(&device->vk.alloc, buf->syncs);
      vk_free(&device->vk.alloc, buf);
      return NULL;
   }

   return buf;
}

static void
delete_buffer(struct u_trace_context *utctx, void *timestamps)
{
   struct v3dv_device *device =
      container_of(utctx, struct v3dv_device, utrace.utrace_ctx);
   struct v3dv_utrace_buffer *buf = timestamps;

   v3dv_bo_free(device, buf->bo, 0);
   vk_free(&device->vk.alloc, buf->immediate);
   vk_free(&device->vk.alloc, buf->syncs);
   vk_free(&device->vk.alloc, buf);
}

/* helpers for multisync common code */
static void *
multisync_zalloc(void *mem_ctx, size_t size)
{
   struct v3dv_device *device = mem_ctx;
   return vk_zalloc(&device->vk.alloc, size, 8, VK_SYSTEM_ALLOCATION_SCOPE_COMMAND);
}

static void
multisync_free(void *mem_ctx, void *ptr)
{
   struct v3dv_device *device = mem_ctx;
   vk_free(&device->vk.alloc, ptr);
}

static bool
record_ts_wait_on(struct v3dv_device *device, struct v3dv_utrace_buffer *buf,
                   uint32_t offset_B, const uint32_t *in_syncs,
                   uint32_t in_sync_count)
{
   uint32_t idx = offset_B / sizeof(uint64_t);
   uint32_t offset = (uint32_t)offset_B;
   uint32_t out_sync;

   if (drmSyncobjCreate(device->pdevice->render_fd, 0, &out_sync)) {
      mesa_loge("v3dv utrace: Syncobj creation failed");
      return false;
   }

   struct v3d_multisync ms = {0};
   ms.ops.zalloc = multisync_zalloc;
   ms.ops.free = multisync_free;
   ms.ops.mem_ctx = device;

   if (!v3d_multisync_init(&ms, V3D_CPU, in_syncs, in_sync_count,
                            &out_sync, 1, NULL)) {
      drmSyncobjDestroy(device->pdevice->render_fd, out_sync);
      mesa_loge("v3dv utrace: Multisync init Failed");
      return false;
   }

   int ret = v3d_submit_timestamp_query_ioctl(device->pdevice->render_fd,
                                              buf->bo->handle,
                                              &offset, &out_sync, 1,
                                              &ms.ext.base);
   v3d_multisync_free(&ms);
   if (ret) {
      drmSyncobjDestroy(device->pdevice->render_fd, out_sync);
      mesa_loge("v3dv utrace: Timestamp query Failed");
      buf->syncs[idx] = 0;
      return false;
   }

   /* Stash the dedicated syncobj for this slot and let
    * read_ts (called at drain time) wait on it. This keeps the
    * submission thread free to keep issuing jobs without blocking on GPU
    * completion, which is required to observe real concurrent execution.
    */
   buf->syncs[idx] = out_sync;
   return true;
}

static bool
record_ts(struct u_trace *ut, void *cs,
          void *timestamps, uint64_t offset_B,
          uint32_t flags)
{
   struct v3dv_job *job = cs;
   struct v3dv_utrace_buffer *buf = timestamps;
   struct v3dv_queue *queue = job->queue;
   struct v3dv_device *device = queue->device;

   /* An invalid job type means that we are tracking the length of
    * the command buffer spawning GPU jobs. In this case, we need
    * to wait for all the queues to which the command buffer submitted
    * jobs, if any.
    */
   if (job->type == -1) {
      struct v3dv_cmd_buffer *cmd_buffer = job->cmd_buffer;

      /* Only bits 0..V3DV_QUEUE_COUNT-1 are ever legitimately set (see
       * queue_handle_job), anything outside that range means the mask
       * was never zeroed for this submission.
       */
      assert((cmd_buffer->trace_queue_mask &
              ~BITFIELD_MASK(V3DV_QUEUE_COUNT)) == 0);

      uint32_t in_syncs[V3DV_QUEUE_COUNT];
      uint32_t in_sync_count = 0;

      u_foreach_bit(q, cmd_buffer->trace_queue_mask) {
         assert(in_sync_count < V3DV_QUEUE_COUNT);
         in_syncs[in_sync_count++] = queue->last_job_syncs.syncs[q];
      }

      if (in_sync_count == 0) {
         /* No GPU work in this cmd_buffer, nothing to synchronize
          * against, just stamp wall-clock time directly.
          */
         uint32_t idx = offset_B / sizeof(uint64_t);
         uint64_t *ts = (uint64_t *)((char *)buf->bo->map + offset_B);
         *ts = os_time_get_nano();
         buf->immediate[idx] = true;
         return true;
      }

      return record_ts_wait_on(device, buf, offset_B, in_syncs, in_sync_count);
   }

   enum v3dv_queue_type queue_sync;
   switch (job->type) {
   case V3DV_JOB_TYPE_GPU_CL:
      queue_sync = V3DV_QUEUE_CL;
      break;
   case V3DV_JOB_TYPE_GPU_CSD:
      queue_sync = V3DV_QUEUE_CSD;
      break;
   case V3DV_JOB_TYPE_GPU_TFU:
      queue_sync = V3DV_QUEUE_TFU;
      break;
   case V3DV_JOB_TYPE_CPU_RESET_QUERIES:
   case V3DV_JOB_TYPE_CPU_COPY_QUERY_RESULTS:
   case V3DV_JOB_TYPE_CPU_CSD_INDIRECT:
   case V3DV_JOB_TYPE_CPU_TIMESTAMP_QUERY:
      queue_sync = V3DV_QUEUE_CPU;
      break;
   default:
      UNREACHABLE("unexpected job type for utrace timestamp");
   }

   uint32_t in_sync = queue->last_job_syncs.syncs[queue_sync];
   return record_ts_wait_on(device, buf, offset_B, &in_sync, 1);
}

static uint64_t
read_ts(struct u_trace_context *utctx,
        void *timestamps, uint64_t offset_B, uint32_t flags, void *flush_data)
{
   struct v3dv_utrace_buffer *buf = timestamps;
   if (!buf) {
      mesa_loge("v3dv_utrace_buffer is NULL in read_ts");
      return U_TRACE_NO_TIMESTAMP;
   }

   uint32_t idx = offset_B / sizeof(uint64_t);

   if (buf->syncs[idx]) {
      struct v3dv_device *device = flush_data;
      drmSyncobjWait(device->pdevice->render_fd, &buf->syncs[idx], 1,
                     INT64_MAX, DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL, NULL);
      drmSyncobjDestroy(device->pdevice->render_fd, buf->syncs[idx]);
      buf->syncs[idx] = 0;
   } else if (!buf->immediate[idx]) {
      return U_TRACE_NO_TIMESTAMP;
   }

   return *(uint64_t *)((char *)buf->bo->map + offset_B);
}

static void
delete_flush_data(struct u_trace_context *utctx, void *flush_data)
{
   free(flush_data);
}

void
v3dv_utrace_context_init(struct v3dv_device *device)
{
   u_trace_context_init(&device->utrace.utrace_ctx, device, sizeof(uint64_t), 0,
                        create_buffer, delete_buffer,
                        record_ts, read_ts,
                        NULL, /* capture_data */
                        NULL, /* get_data */
                        delete_flush_data);
   mtx_init(&device->utrace.process_mutex, mtx_plain);
}

void
v3dv_utrace_context_fini(struct v3dv_device *device)
{
   mtx_destroy(&device->utrace.process_mutex);
   u_trace_context_fini(&device->utrace.utrace_ctx);
}
