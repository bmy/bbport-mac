/*
 * © Copyright 2019 Alyssa Rosenzweig
 * © Copyright 2019 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef __PAN_BO_H__
#define __PAN_BO_H__

#include <time.h>
#include "util/list.h"

#include "pan_pool.h"

#include "kmod/pan_kmod.h"

/* Flags for allocated memory */

/* This memory region is executable */
#define PAN_BO_EXECUTE (1 << 0)

/* This memory region should be lazily allocated and grow-on-page-fault. Must
 * be used in conjunction with INVISIBLE */
#define PAN_BO_GROWABLE (1 << 1)

/* This memory region should not be mapped to the CPU */
#define PAN_BO_INVISIBLE (1 << 2)

/* This region may not be used immediately and will not mmap on allocate
 * (semantically distinct from INVISIBLE, which cannot never be mmaped) */
#define PAN_BO_DELAY_MMAP (1 << 3)

/* BO is shared across processes (imported or exported) and therefore cannot be
 * cached locally */
#define PAN_BO_SHARED (1 << 4)

/* BO might be exported at some point. PAN_BO_SHAREABLE does not imply
 * PAN_BO_SHARED if the BO has not been exported yet */
#define PAN_BO_SHAREABLE (1 << 5)

struct panfrost_device;

struct panfrost_bo {
   /* Must be first for casting */
   struct list_head bucket_link;

   /* Used to link the BO to the BO cache LRU list. */
   struct list_head lru_link;

   /* Store the time this BO was use last, so the BO cache logic can evict
    * stale BOs.
    */
   time_t last_used;

   /* Atomic reference count */
   int32_t refcnt;

   /* Kernel representation of a buffer object. */
   struct pan_kmod_bo *kmod_bo;

   struct panfrost_device *dev;

   /* Mapping for the entire object (all levels) */
   struct pan_ptr ptr;

   uint32_t flags;

   /* Sequence number that gets incremented every time a BO is allocated.
    * Used to identify stale data in the per-BO access tracking done at the
    * panfrost_context level.
    */
   uint32_t seqno;

   /* Driver-provided human-readable description of the BO for debugging.
    * Consists of a BO name and optional additional information, like
    * pixel format or buffer modifiers.
    */
   const char *label;

   /* Lock used to protect concurrent access to flags and label. */
   simple_mtx_t lock;
};

static inline size_t
panfrost_bo_size(struct panfrost_bo *bo)
{
   return bo->kmod_bo->size;
}

static inline size_t
panfrost_bo_handle(struct panfrost_bo *bo)
{
   return bo->kmod_bo->handle;
}

struct panfrost_bo *panfrost_bo_from_kmod_bo(struct panfrost_device *dev,
                                             struct pan_kmod_bo *kmod_bo);

/* Returns true if the BO is ready, false otherwise.
 * access_type is encoding the type of access one wants to ensure is done.
 * Waiting is always done for writers, but if wait_readers is set then readers
 * are also waited for.
 */
static inline bool
panfrost_bo_wait(struct panfrost_bo *bo, int64_t timeout_ns, bool wait_readers)
{
   PAN_TRACE_FUNC(PAN_TRACE_GL_BO);

   return pan_kmod_bo_wait(bo->kmod_bo, timeout_ns, !wait_readers);
}

void panfrost_bo_reference(struct panfrost_bo *bo);
void panfrost_bo_unreference(struct panfrost_bo *bo);
struct panfrost_bo *panfrost_bo_create(struct panfrost_device *dev, size_t size,
                                       uint32_t flags, const char *label);
int panfrost_bo_mmap(struct panfrost_bo *bo);
struct panfrost_bo *panfrost_bo_import(struct panfrost_device *dev, int fd);
int panfrost_bo_export(struct panfrost_bo *bo);
void panfrost_bo_cache_evict_all(struct panfrost_device *dev);

const char *panfrost_bo_replace_label(struct panfrost_bo *bo, const char *label,
                                      bool set_kernel_label);
static inline const char *
panfrost_bo_set_label(struct panfrost_bo *bo, const char *label)
{
   return panfrost_bo_replace_label(bo, label, true);
}

#endif /* __PAN_BO_H__ */
