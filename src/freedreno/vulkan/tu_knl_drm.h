/*
 * Copyright © 2018 Google, Inc.
 * Copyright © 2015 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef TU_KNL_DRM_H
#define TU_KNL_DRM_H

#include "drm-uapi/msm_drm.h"

#include "util/timespec.h"
#include "util/u_dynarray.h"
#include "vk_util.h"

#include "drm/msm/msm_common.h"
#include "tu_knl.h"

VkResult tu_allocate_userspace_iova(struct tu_device *dev,
                                    uint64_t size,
                                    uint64_t align,
                                    uint64_t client_iova,
                                    enum tu_bo_alloc_flags flags,
                                    uint64_t *iova);
int tu_drm_export_dmabuf(struct tu_device *dev, struct tu_bo *bo);

void tu_bo_list_del(struct tu_device *dev, struct tu_bo *bo);
void tu_bo_make_zombie(struct tu_device *dev, struct tu_bo *bo);

struct tu_msm_queue_submit
{
   struct util_dynarray commands;
   struct util_dynarray command_bos;
   struct util_dynarray binds;
};

void *msm_submit_create(struct tu_device *device);
void msm_submit_finish(struct tu_device *device, void *_submit);
void msm_submit_add_entries(struct tu_device *device, void *_submit,
                            struct tu_cs_entry *entries,
                            unsigned num_entries);
void msm_submit_add_bind(struct tu_device *device,
                         void *_submit,
                         struct tu_sparse_vma *vma, uint64_t vma_offset,
                         struct tu_bo *bo, uint64_t bo_offset,
                         uint64_t size);

static inline bool
fence_before(uint32_t a, uint32_t b)
{
   return (int32_t)(a - b) < 0;
}

#endif
