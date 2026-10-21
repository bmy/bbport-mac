/*
 * Copyright © 2012-2018 Rob Clark <robclark@freedesktop.org>
 * Copyright © 2026 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef MSM_COMMON_H_
#define MSM_COMMON_H_

#include "drm-uapi/msm_drm.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <xf86drm.h>

#include "util/log.h"
#include "util/os_time.h"
#include "util/timespec.h"
#include "util/u_process.h"

#include "common/freedreno_common.h"

#ifdef __cplusplus
extern "C" {
#endif

static int
msm_common_gem_new(int fd, uint64_t size, uint32_t msm_flags,
                   uint32_t *out_handle)
{
   struct drm_msm_gem_new req = {
      .size = size,
      .flags = msm_flags,
   };

   int ret = drmCommandWriteRead(fd, DRM_MSM_GEM_NEW, &req, sizeof(req));
   if (ret)
      return ret;

   *out_handle = req.handle;

   return 0;
}

static int
msm_common_gem_close(int fd, uint32_t gem_handle)
{
   struct drm_gem_close req = {
      .handle = gem_handle,
   };

   return drmIoctl(fd, DRM_IOCTL_GEM_CLOSE, &req);
}

static int
msm_common_gem_info_get(int fd, uint32_t gem_handle, uint32_t info,
                        uint64_t *value)
{
   struct drm_msm_gem_info req = {
      .handle = gem_handle,
      .info = info,
   };

   int ret = drmCommandWriteRead(fd, DRM_MSM_GEM_INFO, &req, sizeof(req));
   if (ret)
      return ret;

   *value = req.value;

   return 0;
}

static int
msm_common_get_metadata(int fd, uint32_t gem_handle, void *metadata,
                        uint32_t metadata_size)
{
   /* Zero-initialize the caller's buffer so that any bytes the kernel does
    * not fill (e.g. when no metadata was ever set on this BO) are zero
    * rather than left uninitialized, and so that we are robust to kernels
    * that don't report the actual stored length in req.len.
    */
   memset(metadata, 0, metadata_size);

   struct drm_msm_gem_info req = {
      .handle = gem_handle,
      .info = MSM_INFO_GET_METADATA,
      .value = (uintptr_t)(void *)metadata,
      .len = metadata_size,
   };

   /* drmCommandWriteRead() (not drmCommandWrite()) so that the kernel's
    * req.len -- the actual length of the stored metadata -- is copied back.
    */
   int ret = drmCommandWriteRead(fd, DRM_MSM_GEM_INFO, &req, sizeof(req));
   if (ret) {
      mesa_logw_once("Failed to get BO metadata with DRM_MSM_GEM_INFO: %d",
                     ret);
      return ret;
   }

   /* If the kernel has no metadata stored for this BO (or stored less than
    * the caller requested), signal ENODATA so callers fall back to their own
    * layout defaults instead of trusting a partial/zeroed struct.
    */
   if (req.len < metadata_size)
      return -ENODATA;

   return 0;
}

static void
msm_common_set_metadata(int fd, uint32_t gem_handle, void *metadata,
                        uint32_t metadata_size)
{
   struct drm_msm_gem_info req = {
      .handle = gem_handle,
      .info = MSM_INFO_SET_METADATA,
      .value = (uintptr_t)(void *)metadata,
      .len = metadata_size,
   };

   int ret = drmCommandWrite(fd, DRM_MSM_GEM_INFO, &req, sizeof(req));
   if (ret) {
      mesa_logw_once("Failed to set BO metadata with DRM_MSM_GEM_INFO: %d",
                     ret);
   }
}

/**
 * Sets the name in the kernel so that the contents of /debug/dri/0/gem are more
 * useful.
 */
static void
msm_common_bo_set_name(int fd, uint32_t gem_handle, const char *name,
                       size_t length)
{
   char buf[FD_MSM_GEM_NAME_LENGTH + 1];

   if (length > FD_MSM_GEM_NAME_LENGTH) {
      mesa_logd("Truncating BO name: %s", name);

      memcpy(buf, name, FD_MSM_GEM_NAME_LENGTH);
      buf[FD_MSM_GEM_NAME_LENGTH] = '\0';

      name = buf;
      length = FD_MSM_GEM_NAME_LENGTH;
   }

   struct drm_msm_gem_info req = {
      .handle = gem_handle,
      .info = MSM_INFO_SET_NAME,
      .value = (uintptr_t)(void *)name,
      .len = length,
   };

   int ret = drmCommandWrite(fd, DRM_MSM_GEM_INFO, &req, sizeof(req));
   if (ret) {
      mesa_logw_once("Failed to set BO name with DRM_MSM_GEM_INFO: %d", ret);
   }
}

static int
msm_common_get_param(int fd, uint32_t pipe, uint32_t param, uint64_t *value)
{
   struct drm_msm_param req = {
      .pipe = pipe,
      .param = param,
   };

   int ret = drmCommandWriteRead(fd, DRM_MSM_GET_PARAM, &req, sizeof(req));
   if (ret)
      return ret;

   *value = req.value;

   return 0;
}

static int
msm_common_set_param(int fd, uint32_t pipe, uint32_t param, uint64_t value,
                     uint32_t len)
{
   struct drm_msm_param req = {
      .pipe = pipe,
      .param = param,
      .value = value,
      .len = len,
   };

   return drmCommandWriteRead(fd, DRM_MSM_SET_PARAM, &req, sizeof(req));
}

static int
msm_common_get_va_prop(int fd, uint32_t pipe, uint64_t *va_start,
                       uint64_t *va_size)
{
   uint64_t value;
   int ret = msm_common_get_param(fd, pipe, MSM_PARAM_VA_START, &value);
   if (ret)
      return ret;

   *va_start = value;

   ret = msm_common_get_param(fd, pipe, MSM_PARAM_VA_SIZE, &value);
   if (ret)
      return ret;

   *va_size = value;

   return 0;
}

static void
msm_common_set_debuginfo(int fd, uint32_t pipe)
{
   const char *comm = util_get_process_name();
   if (comm)
      msm_common_set_param(fd, pipe, MSM_PARAM_COMM, (uintptr_t)comm,
                           strlen(comm));

   static char cmdline[0x1000];
   if (util_get_command_line(cmdline, sizeof(cmdline)))
      msm_common_set_param(fd, pipe, MSM_PARAM_CMDLINE, (uintptr_t)cmdline,
                           strlen(cmdline));
}

static inline void
get_abs_timeout(struct drm_msm_timespec *tv, uint64_t ns)
{
   struct timespec t;

   if (ns == OS_TIMEOUT_INFINITE)
      ns = 3600ULL * NSEC_PER_SEC; /* 1 hour timeout is almost infinite */

   clock_gettime(CLOCK_MONOTONIC, &t);
   tv->tv_sec = t.tv_sec + ns / NSEC_PER_SEC;
   tv->tv_nsec = t.tv_nsec + ns % NSEC_PER_SEC;
   if (tv->tv_nsec >= NSEC_PER_SEC) { /* handle nsec overflow */
      tv->tv_nsec -= NSEC_PER_SEC;
      tv->tv_sec++;
   }
}

static int
msm_common_wait_fence(int fd, uint32_t queue_id, int fence, uint64_t timeout_ns)
{
   struct drm_msm_wait_fence req = {
      .fence = fence,
      .queueid = queue_id,
   };

   get_abs_timeout(&req.timeout, timeout_ns);

   return drmCommandWrite(fd, DRM_MSM_WAIT_FENCE, &req, sizeof(req));
}

static bool
msm_common_is_memory_type_supported(int fd, uint64_t page_size, uint32_t flags)
{
   struct drm_msm_gem_new req = {
      .size = page_size,
      .flags = flags,
   };

   int ret = drmCommandWriteRead(fd, DRM_MSM_GEM_NEW, &req, sizeof(req));
   if (ret)
      return false;

   msm_common_gem_close(fd, req.handle);

   return true;
}

static bool
msm_common_has_preemption(int fd, uint32_t priority)
{
   struct drm_msm_submitqueue req = {
      .flags = MSM_SUBMITQUEUE_ALLOW_PREEMPT,
      .prio = priority,
   };

   int ret =
      drmCommandWriteRead(fd, DRM_MSM_SUBMITQUEUE_NEW, &req, sizeof(req));
   if (ret)
      return false;

   drmCommandWrite(fd, DRM_MSM_SUBMITQUEUE_CLOSE, &req.id, sizeof(req.id));

   return true;
}

#ifdef __cplusplus
}
#endif

#endif /* MSM_COMMON_H_ */
