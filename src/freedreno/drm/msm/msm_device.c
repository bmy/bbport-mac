/*
 * Copyright © 2012-2018 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "msm_priv.h"

static void
msm_device_destroy(struct fd_device *dev)
{
}

static const struct fd_device_funcs funcs = {
   .bo_new = msm_bo_new,
   .bo_from_handle = msm_bo_from_handle,
   .handle_from_dmabuf = fd_handle_from_dmabuf_drm,
   .bo_from_dmabuf = fd_bo_from_dmabuf_drm,
   .bo_close_handle = fd_bo_close_handle_drm,
   .pipe_new = msm_pipe_new,
   .destroy = msm_device_destroy,
};

struct fd_device *
msm_device_new(int fd, drmVersionPtr version)
{
   struct msm_device *msm_dev;
   struct fd_device *dev;

   STATIC_ASSERT(FD_BO_PREP_READ == MSM_PREP_READ);
   STATIC_ASSERT(FD_BO_PREP_WRITE == MSM_PREP_WRITE);
   STATIC_ASSERT(FD_BO_PREP_NOSYNC == MSM_PREP_NOSYNC);

   msm_dev = calloc(1, sizeof(*msm_dev));
   if (!msm_dev)
      return NULL;

   dev = &msm_dev->base;
   dev->funcs = &funcs;
   dev->version = version->version_minor;

   if (version->version_minor >= FD_VERSION_CACHED_COHERENT) {
      /* The kernel is new enough to support MSM_BO_CACHED_COHERENT,
       * but that is not a guarantee that the device we are running
       * on supports it.  So do a test allocation to find out.
       */
      dev->has_cached_coherent = msm_common_is_memory_type_supported(
         fd, os_page_size, MSM_BO_CACHED_COHERENT);
   }

   dev->bo_size = sizeof(struct msm_bo);

   return dev;
}
