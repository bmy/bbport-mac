/* Copyright © 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdint.h>

enum intel_pagefault_access {
   /** Faults caused by read access */
   INTEL_PAGEFAULT_ACCESS_READ,

   /** Faults caused by write access */
   INTEL_PAGEFAULT_ACCESS_WRITE,

   /** Faults caused by atomic access */
   INTEL_PAGEFAULT_ACCESS_ATOMIC,
};

enum intel_pagefault_type {
   /** The page or the page table was not present */
   INTEL_PAGEFAULT_TYPE_NOT_PRESENT,

   /** The page was present, but was not writable */
   INTEL_PAGEFAULT_TYPE_WRITE_ACCESS,

   /** The page was present, but was not atomic-able */
   INTEL_PAGEFAULT_TYPE_ATOMIC_ACCESS,
};

enum intel_pagefault_level {
   /** GraphicsAddress[20:12] - Page Table Entry */
   INTEL_PAGEFAULT_LEVEL_PTE,

   /** GraphicsAddress[29:21] - Page Directory Entry */
   INTEL_PAGEFAULT_LEVEL_PDE,

   /** GraphicsAddress[38:30] - Page Directory Pointer */
   INTEL_PAGEFAULT_LEVEL_PDP,

   /** GraphicsAddress[47:39] - Page Map Level 4 */
   INTEL_PAGEFAULT_LEVEL_PML4,

   /** GraphicsAddress[56:48] - Page Map Level 5 */
   INTEL_PAGEFAULT_LEVEL_PML5,
};

enum intel_pagefault_src {
   /** Access to the AMFS scratch space */
   INTEL_PAGEFAULT_SRC_AMFS_SCRATCH_ACCESS,

   /** Fetching the AMFS indirect state */
   INTEL_PAGEFAULT_SRC_AMFS_STATE,

   /** Fetching binding table indicies */
   INTEL_PAGEFAULT_SRC_BINDING_TABLE,

   /** Access performed via the blitter */
   INTEL_PAGEFAULT_SRC_BLITTER_ACCESS,

   /** Access to a compression control surface */
   INTEL_PAGEFAULT_SRC_CCS_SURFACE_ACCESS,

   /** Fetching the command stream */
   INTEL_PAGEFAULT_SRC_CS_INSTRUCTION_FETCH,

   /** Command streamer MI, indirect dispatch, postsync, context save */
   INTEL_PAGEFAULT_SRC_CS_MEMORY_ACCESS,

   /** Access performed via the LSC/HDC, sampler route-to-LSC, and raytracing */
   INTEL_PAGEFAULT_SRC_DATAPORT_ACCESS,

   /** Fetching surface states for the dataport */
   INTEL_PAGEFAULT_SRC_DATAPORT_STATE,

   /** Access to a depth buffer and HIZ */
   INTEL_PAGEFAULT_SRC_DEPTH_ACCESS,

   /** Fetching state of a depth buffer */
   INTEL_PAGEFAULT_SRC_DEPTH_STATE,

   /** Fetching shader execution unit instructions */
   INTEL_PAGEFAULT_SRC_EU_INSTRUCTION_FETCH,

   /** Access to an image/video coding bitstream and pointer */
   INTEL_PAGEFAULT_SRC_MEDIA_CODEC_BITSTREAM,

   /** Feedback from video coding (histogram, etc) */
   INTEL_PAGEFAULT_SRC_MEDIA_CODEC_FEEDBACK,

   /** Access to an image/video coding reference or output surface */
   INTEL_PAGEFAULT_SRC_MEDIA_CODEC_SURFACE,

   /** Access to an image/video enhancement state or surface */
   INTEL_PAGEFAULT_SRC_MEDIA_ENHANCE_ACCESS,

   /** Observation architecture feedback */
   INTEL_PAGEFAULT_SRC_OA_FEEDBACK,

   /** Accessing the page tables (PPGTT, GGTT, TRTT) */
   INTEL_PAGEFAULT_SRC_PAGE_WALKER,

   /** Access to a render target */
   INTEL_PAGEFAULT_SRC_PIXELPORT_ACCESS,

   /** Fetching the state of a render target */
   INTEL_PAGEFAULT_SRC_PIXELPORT_STATE,

   /** Fetching the state of the rasterization pipeline */
   INTEL_PAGEFAULT_SRC_RASTERIZER_STATE,

   /** Fetching surface and sampler states for the sampler */
   INTEL_PAGEFAULT_SRC_SAMPLER_ACCESS,

   /** Access performed via the sampler */
   INTEL_PAGEFAULT_SRC_SAMPLER_STATE,

   /** Transform feedback (stream output) */
   INTEL_PAGEFAULT_SRC_TRANSFORM_FEEDBACK,

   /** Fetching vertex data from memory */
   INTEL_PAGEFAULT_SRC_VERTEX_FETCH,

   /** Unknown or undefined source */
   INTEL_PAGEFAULT_SRC_UNKNOWN,
};

struct intel_pagefault_info {
    uint64_t address;
    uint32_t precision;
    enum intel_pagefault_access access;
    enum intel_pagefault_type type;
    enum intel_pagefault_level level;
    enum intel_pagefault_src src;
};

struct intel_pagefault_buffer {
   unsigned size;
   struct intel_pagefault_info items[];
};

const char *
intel_pagefault_access_to_string(enum intel_pagefault_access access);
const char *
intel_pagefault_type_to_string(enum intel_pagefault_type type);
const char *
intel_pagefault_level_to_string(enum intel_pagefault_level level);
const char *
intel_pagefault_src_to_string(enum intel_pagefault_src src);
