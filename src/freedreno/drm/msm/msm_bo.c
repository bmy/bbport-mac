/*
 * Copyright © 2012-2018 Rob Clark <robclark@freedesktop.org>
 * SPDX-License-Identifier: MIT
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "msm_priv.h"

static int
bo_allocate(struct msm_bo *msm_bo)
{
   struct fd_bo *bo = &msm_bo->base;
   if (!msm_bo->offset) {
      /* if the buffer is already backed by pages then this
       * doesn't actually do anything (other than giving us
       * the offset)
       */
      int ret = msm_common_gem_info_get(bo->dev->fd, bo->handle,
                                        MSM_INFO_GET_OFFSET, &msm_bo->offset);
      if (ret) {
         ERROR_MSG("alloc failed: %s", strerror(errno));
         return ret;
      }
   }

   return 0;
}

static int
msm_bo_offset(struct fd_bo *bo, uint64_t *offset)
{
   struct msm_bo *msm_bo = to_msm_bo(bo);
   int ret = bo_allocate(msm_bo);
   if (ret)
      return ret;
   *offset = msm_bo->offset;
   return 0;
}

static int
msm_bo_cpu_prep(struct fd_bo *bo, struct fd_pipe *pipe, uint32_t op)
{
   struct drm_msm_gem_cpu_prep req = {
      .handle = bo->handle,
      .op = op,
   };

   get_abs_timeout(&req.timeout, OS_TIMEOUT_INFINITE);

   return drmCommandWrite(bo->dev->fd, DRM_MSM_GEM_CPU_PREP, &req, sizeof(req));
}

static int
msm_bo_madvise(struct fd_bo *bo, int willneed)
{
   struct drm_msm_gem_madvise req = {
      .handle = bo->handle,
      .madv = willneed ? MSM_MADV_WILLNEED : MSM_MADV_DONTNEED,
   };
   int ret;

   /* older kernels do not support this: */
   if (bo->dev->version < FD_VERSION_MADVISE)
      return willneed;

   ret =
      drmCommandWriteRead(bo->dev->fd, DRM_MSM_GEM_MADVISE, &req, sizeof(req));
   if (ret)
      return ret;

   return req.retained;
}

static uint64_t
msm_bo_iova(struct fd_bo *bo)
{
   uint64_t value;
   int ret = msm_common_gem_info_get(bo->dev->fd, bo->handle, MSM_INFO_GET_IOVA,
                                     &value);
   if (ret)
      return 0;

   return value;
}

static void
msm_bo_set_name(struct fd_bo *bo, const char *fmt, va_list ap)
{
   char buf[FD_MSM_GEM_NAME_LENGTH + 1];
   int sz;

   if (bo->dev->version < FD_VERSION_SOFTPIN)
      return;

   sz = vsnprintf(buf, sizeof(buf), fmt, ap);
   sz = MIN2(sz, FD_MSM_GEM_NAME_LENGTH);

   msm_common_bo_set_name(bo->dev->fd, bo->handle, buf, sz);
}

static void
msm_bo_set_metadata(struct fd_bo *bo, void *metadata, uint32_t metadata_size)
{
   msm_common_set_metadata(bo->dev->fd, bo->handle, metadata, metadata_size);
}

static int
msm_bo_get_metadata(struct fd_bo *bo, void *metadata, uint32_t metadata_size)
{
   return msm_common_get_metadata(bo->dev->fd, bo->handle, metadata,
                                  metadata_size);
}

static const struct fd_bo_funcs funcs = {
   .offset = msm_bo_offset,
   .map = fd_bo_map_os_mmap,
   .cpu_prep = msm_bo_cpu_prep,
   .madvise = msm_bo_madvise,
   .iova = msm_bo_iova,
   .set_name = msm_bo_set_name,
   .set_metadata = msm_bo_set_metadata,
   .get_metadata = msm_bo_get_metadata,
   .dmabuf = fd_bo_dmabuf_drm,
   .destroy = fd_bo_fini_common,
};

/* allocate a buffer handle: */
static int
new_handle(struct fd_device *dev, uint32_t size, uint32_t flags, uint32_t *handle)
{
   uint32_t msm_flags = 0;

   if (flags & FD_BO_SCANOUT)
      msm_flags |= MSM_BO_SCANOUT;

   if (flags & FD_BO_GPUREADONLY)
      msm_flags |= MSM_BO_GPU_READONLY;

   if (flags & FD_BO_CACHED_COHERENT)
      msm_flags |= MSM_BO_CACHED_COHERENT;
   else
      msm_flags |= MSM_BO_WC;

   return msm_common_gem_new(dev->fd, size, msm_flags, handle);
}

/* allocate a new buffer object */
struct fd_bo *
msm_bo_new(struct fd_device *dev, uint32_t size, uint32_t flags)
{
   uint32_t handle;
   int ret;

   ret = new_handle(dev, size, flags, &handle);
   if (ret)
      return NULL;

   return msm_bo_from_handle(dev, size, handle);
}

/* allocate a new buffer object from existing handle (import) */
struct fd_bo *
msm_bo_from_handle(struct fd_device *dev, uint32_t size, uint32_t handle)
{
   struct msm_bo *msm_bo;
   struct fd_bo *bo;

   msm_bo = calloc(1, sizeof(*msm_bo));
   if (!msm_bo)
      return NULL;

   bo = &msm_bo->base;
   bo->size = size;
   bo->handle = handle;
   bo->funcs = &funcs;

   fd_bo_init_common(bo, dev);

   return bo;
}
