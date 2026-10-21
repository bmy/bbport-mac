/* Copyright © 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "intel_pagefault.h"

const char *
intel_pagefault_access_to_string(enum intel_pagefault_access access)
{
   static const char *const lookup[] = {
      [INTEL_PAGEFAULT_ACCESS_READ] = "Read",
      [INTEL_PAGEFAULT_ACCESS_WRITE] = "Write",
      [INTEL_PAGEFAULT_ACCESS_ATOMIC] = "Atomic",
   };
   return lookup[access];
}

const char *
intel_pagefault_type_to_string(enum intel_pagefault_type type)
{
   static const char *const lookup[] = {
      [INTEL_PAGEFAULT_TYPE_NOT_PRESENT] = "Not Present",
      [INTEL_PAGEFAULT_TYPE_WRITE_ACCESS] = "Not Writable",
      [INTEL_PAGEFAULT_TYPE_ATOMIC_ACCESS] = "Atomic",
   };
   return lookup[type];
}

const char *
intel_pagefault_level_to_string(enum intel_pagefault_level level)
{
   static const char *const lookup[] = {
      [INTEL_PAGEFAULT_LEVEL_PTE] = "PTE",
      [INTEL_PAGEFAULT_LEVEL_PDE] = "PDE",
      [INTEL_PAGEFAULT_LEVEL_PDP] = "PDP",
      [INTEL_PAGEFAULT_LEVEL_PML4] = "PML4",
      [INTEL_PAGEFAULT_LEVEL_PML5] = "PML5",
   };
   return lookup[level];
}

const char *
intel_pagefault_src_to_string(enum intel_pagefault_src src)
{
   static const char *const lookup[] = {
      [INTEL_PAGEFAULT_SRC_AMFS_SCRATCH_ACCESS] = "AMFS Scratch Access",
      [INTEL_PAGEFAULT_SRC_AMFS_STATE] = "AMFS State",
      [INTEL_PAGEFAULT_SRC_BINDING_TABLE] = "Binding Table Access",
      [INTEL_PAGEFAULT_SRC_BLITTER_ACCESS] = "Blitter Engine Access",
      [INTEL_PAGEFAULT_SRC_CCS_SURFACE_ACCESS] = "Compression Control Surface Access",
      [INTEL_PAGEFAULT_SRC_CS_INSTRUCTION_FETCH] = "Command Streamer Instruction Fetch",
      [INTEL_PAGEFAULT_SRC_CS_MEMORY_ACCESS] = "Command Streamer Memory Access",
      [INTEL_PAGEFAULT_SRC_DATAPORT_ACCESS] = "Load/Store Cache Access",
      [INTEL_PAGEFAULT_SRC_DATAPORT_STATE] = "Load/Store Cache State",
      [INTEL_PAGEFAULT_SRC_DEPTH_ACCESS] = "Depth Cache Access",
      [INTEL_PAGEFAULT_SRC_DEPTH_STATE] = "Depth Cache State",
      [INTEL_PAGEFAULT_SRC_EU_INSTRUCTION_FETCH] = "Execution Unit Instruction Fetch",
      [INTEL_PAGEFAULT_SRC_MEDIA_CODEC_BITSTREAM] = "Media Codec Bitstream",
      [INTEL_PAGEFAULT_SRC_MEDIA_CODEC_FEEDBACK] = "Media Codec Feedback",
      [INTEL_PAGEFAULT_SRC_MEDIA_CODEC_SURFACE] = "Media Codec Surface",
      [INTEL_PAGEFAULT_SRC_MEDIA_ENHANCE_ACCESS] = "Media Enhancement Access",
      [INTEL_PAGEFAULT_SRC_OA_FEEDBACK] = "OA Counter Feedback",
      [INTEL_PAGEFAULT_SRC_PAGE_WALKER] = "GAM Page Walker",
      [INTEL_PAGEFAULT_SRC_PIXELPORT_ACCESS] = "Render Cache Access",
      [INTEL_PAGEFAULT_SRC_PIXELPORT_STATE] = "Render Cache State",
      [INTEL_PAGEFAULT_SRC_RASTERIZER_STATE] = "Rasterizer State",
      [INTEL_PAGEFAULT_SRC_SAMPLER_ACCESS] = "Sampler Cache Access",
      [INTEL_PAGEFAULT_SRC_SAMPLER_STATE] = "Sampler Cache State",
      [INTEL_PAGEFAULT_SRC_TRANSFORM_FEEDBACK] = "Transform Feedback",
      [INTEL_PAGEFAULT_SRC_VERTEX_FETCH] = "Vertex Fetch",
      [INTEL_PAGEFAULT_SRC_UNKNOWN] = "Unknown",
   };
   return lookup[src];
}
