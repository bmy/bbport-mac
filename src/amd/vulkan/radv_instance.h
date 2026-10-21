/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef RADV_INSTANCE_H
#define RADV_INSTANCE_H

#include "util/bitset.h"
#include "util/macros.h"
#include "util/simple_mtx.h"
#include "radv_radeon_winsys.h"
#include "vk_instance.h"

#ifdef ANDROID_STRICT
#define RADV_API_VERSION VK_MAKE_VERSION(1, 1, VK_HEADER_VERSION)
#else
#define RADV_API_VERSION     VK_MAKE_VERSION(1, 4, VK_HEADER_VERSION)
#define RADV_API_VERSION_1_3 VK_MAKE_VERSION(1, 3, VK_HEADER_VERSION)
#endif

/* Please keep docs/envvars.rst up-to-date when you add/remove options. */
enum radv_debug_flag {
   RADV_DEBUG_NO_FAST_CLEARS = 0,
   RADV_DEBUG_NO_DCC,
   RADV_DEBUG_NO_CACHE_COMPAT,
   RADV_DEBUG_NO_CACHE,
   RADV_DEBUG_DUMP_SHADER_STATS,
   RADV_DEBUG_NO_HIZ,
   RADV_DEBUG_ALL_BOS,
   RADV_DEBUG_NO_IB_CHAINING,
   RADV_DEBUG_DUMP_SPIRV,
   RADV_DEBUG_ZERO_VRAM,
   RADV_DEBUG_SYNC_SHADERS,
   RADV_DEBUG_DUMP_PREOPT_IR,
   RADV_DEBUG_INFO,
   RADV_DEBUG_STARTUP,
   RADV_DEBUG_CHECKIR,
   RADV_DEBUG_NOBINNING,
   RADV_DEBUG_NO_NGG,
   RADV_DEBUG_DUMP_META_SHADERS,
   RADV_DEBUG_LLVM,
   RADV_DEBUG_HANG,
   RADV_DEBUG_IMG,
   RADV_DEBUG_NO_UMR,
   RADV_DEBUG_NO_DISPLAY_DCC,
   RADV_DEBUG_NO_TC_COMPAT_CMASK,
   RADV_DEBUG_NO_VRS_FLAT_SHADING,
   RADV_DEBUG_NO_ATOC_DITHERING,
   RADV_DEBUG_NO_NGGC,
   RADV_DEBUG_DUMP_PROLOGS,
   RADV_DEBUG_NO_DMA_BLIT,
   RADV_DEBUG_DUMP_EPILOGS,
   RADV_DEBUG_NO_FMASK,
   RADV_DEBUG_SHADOW_REGS,
   RADV_DEBUG_EXTRA_MD,
   RADV_DEBUG_NO_GPL,
   RADV_DEBUG_NO_RT,
   RADV_DEBUG_NO_MESH_SHADER,
   RADV_DEBUG_NO_ESO,
   RADV_DEBUG_PSO_CACHE_STATS,
   RADV_DEBUG_NIR_DEBUG_INFO,
   RADV_DEBUG_DUMP_TRAP_HANDLER,

   /* These need to be contiguous for RADV_DEBUG_DUMP_SHADERS. */
   RADV_DEBUG_DUMP_SHADERS_BEGIN,
   RADV_DEBUG_DUMP_VS = RADV_DEBUG_DUMP_SHADERS_BEGIN,
   RADV_DEBUG_DUMP_TCS,
   RADV_DEBUG_DUMP_TES,
   RADV_DEBUG_DUMP_GS,
   RADV_DEBUG_DUMP_PS,
   RADV_DEBUG_DUMP_TASK,
   RADV_DEBUG_DUMP_MESH,
   RADV_DEBUG_DUMP_CS,
   RADV_DEBUG_DUMP_NIR,
   RADV_DEBUG_DUMP_ASM,
   RADV_DEBUG_DUMP_BACKEND_IR,
   RADV_DEBUG_DUMP_SHADERS_END = RADV_DEBUG_DUMP_BACKEND_IR,

   RADV_DEBUG_PSO_HISTORY,
   RADV_DEBUG_BVH4,
   RADV_DEBUG_NO_VIDEO,
   RADV_DEBUG_VALIDATE_VAS,
   RADV_DEBUG_DUMP_BO_HISTORY,
   RADV_DEBUG_DUMP_IBS,
   RADV_DEBUG_VM,
   RADV_DEBUG_NO_SMEM_MITIGATION,
   RADV_DEBUG_FULL_SYNC,
   RADV_DEBUG_NO_TMZ,
   RADV_DEBUG_NO_HEAP,

   RADV_DEBUG_COUNT,
};

#define RADV_DEBUG(instance, flag) BITSET_TEST((instance)->debug_flags, RADV_DEBUG_##flag)

#define RADV_DEBUG_DUMP_SHADERS(instance)                                                                              \
   BITSET_TEST_RANGE((instance)->debug_flags, RADV_DEBUG_DUMP_SHADERS_BEGIN, RADV_DEBUG_DUMP_SHADERS_END)

enum {
   RADV_PERFTEST_LOCAL_BOS = 1u << 0,
   RADV_PERFTEST_DCC_MSAA = 1u << 1,
   RADV_PERFTEST_CS_WAVE_32 = 1u << 2,
   RADV_PERFTEST_PS_WAVE_32 = 1u << 3,
   RADV_PERFTEST_GE_WAVE_32 = 1u << 4,
   RADV_PERFTEST_NO_SAM = 1u << 5,
   RADV_PERFTEST_SAM = 1u << 6,
   RADV_PERFTEST_NGGC = 1u << 7,
   RADV_PERFTEST_RT_WAVE_64 = 1u << 8,
   RADV_PERFTEST_DMA_SHADERS = 1u << 9,
   RADV_PERFTEST_NIR_CACHE = 1u << 10,
   RADV_PERFTEST_NO_GTT_SPILL = 1u << 11,
   RADV_PERFTEST_RT_CPS = 1u << 12,
   RADV_PERFTEST_LOWLATENCYDEC = 1u << 13,
   RADV_PERFTEST_LOWLATENCYENC = 1u << 14,
};

enum {
   RADV_EXPERIMENTAL_EMULATE_RT = 1u << 0,
   RADV_EXPERIMENTAL_VIDEO_DECODE = 1u << 1,
   RADV_EXPERIMENTAL_TRANSFER_QUEUE = 1u << 2,
   RADV_EXPERIMENTAL_VIDEO_ENCODE = 1u << 3,
   RADV_EXPERIMENTAL_HIC = 1u << 4,
   RADV_EXPERIMENTAL_SPARSE = 1u << 5,
   RADV_EXPERIMENTAL_BFLOAT16 = 1u << 6,
   RADV_EXPERIMENTAL_MSRTSS = 1u << 7,
   RADV_EXPERIMENTAL_ELF = 1u << 8,
};

enum {
   RADV_TRAP_EXCP_MEM_VIOL = 1u << 0,
   RADV_TRAP_EXCP_FLOAT_DIV_BY_ZERO = 1u << 1,
   RADV_TRAP_EXCP_FLOAT_OVERFLOW = 1u << 2,
   RADV_TRAP_EXCP_FLOAT_UNDERFLOW = 1u << 3,
};

enum {
   RADV_QUEUE_DISABLE_GENERAL = 1u << 0,
   RADV_QUEUE_DISABLE_COMPUTE = 1u << 1,
   RADV_QUEUE_DISABLE_VIDEO_DEC = 1u << 2,
   RADV_QUEUE_DISABLE_VIDEO_ENC = 1u << 3,
   RADV_QUEUE_DISABLE_TRANSFER = 1u << 4,
   RADV_QUEUE_DISABLE_SPARSE = 1u << 5,
};

enum radv_trace_mode {
   /** Radeon GPU Profiler */
   RADV_TRACE_MODE_RGP = 1 << VK_TRACE_MODE_COUNT,

   /** Radeon Raytracing Analyzer */
   RADV_TRACE_MODE_RRA = 1 << (VK_TRACE_MODE_COUNT + 1),

   RADV_TRACE_MODE_GAMMA = 1 << (VK_TRACE_MODE_COUNT + 2),

   /** Gather context rolls of submitted command buffers */
   RADV_TRACE_MODE_CTX_ROLLS = 1 << (VK_TRACE_MODE_COUNT + 3),

   RADV_TRACE_MODE_RANGES = 1 << (VK_TRACE_MODE_COUNT + 4),
};

struct radv_instance {
   struct vk_instance vk;

   VkAllocationCallbacks alloc;

   simple_mtx_t shader_dump_mtx;

   BITSET_DECLARE(debug_flags, RADV_DEBUG_COUNT);
   uint64_t perftest_flags;
   uint64_t experimental_flags;
   uint64_t trap_excp_flags;
   uint32_t queue_disable_flags;

   enum radeon_ctx_pstate profile_pstate;

   FILE *pso_history_logfile;
};

VK_DEFINE_HANDLE_CASTS(radv_instance, vk.base, VkInstance, VK_OBJECT_TYPE_INSTANCE)

const char *radv_get_debug_option_name(int id);

const char *radv_get_perftest_option_name(int id);

static const char *
radv_bvh_stats_file()
{
   return os_get_option_secure("RADV_BVH_STATS_FILE");
}

static bool
radv_bvh_dumping_enabled(const struct radv_instance *instance)
{
   /* Gathering bvh stats uses a large part of the rra code for dumping bvhs. */
   return (instance->vk.trace_mode & (RADV_TRACE_MODE_RRA | RADV_TRACE_MODE_GAMMA)) || radv_bvh_stats_file();
}

#endif /* RADV_INSTANCE_H */
