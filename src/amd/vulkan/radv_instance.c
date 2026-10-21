/*
 * Copyright © 2016 Red Hat.
 * Copyright © 2016 Bas Nieuwenhuizen
 *
 * based in part on anv driver which is:
 * Copyright © 2015 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef HAVE_VALGRIND
#include <memcheck.h>
#include <valgrind.h>
#define VG(x) x
#else
#define VG(x) ((void)0)
#endif

#include "tools/radv_debug.h"
#include "tools/radv_debug_hang.h"
#include "radv_entrypoints.h"
#include "radv_instance.h"
#include "radv_wsi.h"

#include "util/bitset.h"
#include "util/u_debug.h"
#include "vk_instance.h"
#include "vk_log.h"

static const struct debug_control_bitset radv_debug_options[] = {
#define OPT1(name, bit)                                                                                              \
   { .string = name, .range = {bit, bit}, }
#define OPT2(name, start, end)                                                                                        \
   { .string = name, .range = {start, end}, }
   OPT1("nofastclears", RADV_DEBUG_NO_FAST_CLEARS),
   OPT1("nodcc", RADV_DEBUG_NO_DCC),
   OPT2("shaders", RADV_DEBUG_DUMP_SHADERS_BEGIN, RADV_DEBUG_DUMP_SHADERS_END),
   OPT1("nocachecompat", RADV_DEBUG_NO_CACHE_COMPAT),
   OPT1("nocache", RADV_DEBUG_NO_CACHE),
   OPT1("shaderstats", RADV_DEBUG_DUMP_SHADER_STATS),
   OPT1("nohiz", RADV_DEBUG_NO_HIZ),
   OPT1("allbos", RADV_DEBUG_ALL_BOS),
   OPT1("noibchaining", RADV_DEBUG_NO_IB_CHAINING),
   OPT1("spirv", RADV_DEBUG_DUMP_SPIRV),
   OPT1("zerovram", RADV_DEBUG_ZERO_VRAM),
   OPT1("syncshaders", RADV_DEBUG_SYNC_SHADERS),
   OPT1("preoptir", RADV_DEBUG_DUMP_PREOPT_IR),
   OPT1("info", RADV_DEBUG_INFO),
   OPT1("startup", RADV_DEBUG_STARTUP),
   OPT1("checkir", RADV_DEBUG_CHECKIR),
   OPT1("nobinning", RADV_DEBUG_NOBINNING),
   OPT1("nongg", RADV_DEBUG_NO_NGG),
   OPT1("metashaders", RADV_DEBUG_DUMP_META_SHADERS),
   OPT1("llvm", RADV_DEBUG_LLVM),
   OPT1("hang", RADV_DEBUG_HANG),
   OPT1("img", RADV_DEBUG_IMG),
   OPT1("noumr", RADV_DEBUG_NO_UMR),
   OPT1("nodisplaydcc", RADV_DEBUG_NO_DISPLAY_DCC),
   OPT1("notccompatcmask", RADV_DEBUG_NO_TC_COMPAT_CMASK),
   OPT1("novrsflatshading", RADV_DEBUG_NO_VRS_FLAT_SHADING),
   OPT1("noatocdithering", RADV_DEBUG_NO_ATOC_DITHERING),
   OPT1("nonggc", RADV_DEBUG_NO_NGGC),
   OPT1("prologs", RADV_DEBUG_DUMP_PROLOGS),
   OPT1("nodma", RADV_DEBUG_NO_DMA_BLIT),
   OPT1("epilogs", RADV_DEBUG_DUMP_EPILOGS),
   OPT1("nofmask", RADV_DEBUG_NO_FMASK),
   OPT1("shadowregs", RADV_DEBUG_SHADOW_REGS),
   OPT1("extra_md", RADV_DEBUG_EXTRA_MD),
   OPT1("nogpl", RADV_DEBUG_NO_GPL),
   OPT1("nort", RADV_DEBUG_NO_RT),
   OPT1("nomeshshader", RADV_DEBUG_NO_MESH_SHADER),
   OPT1("noeso", RADV_DEBUG_NO_ESO),
   OPT1("psocachestats", RADV_DEBUG_PSO_CACHE_STATS),
   OPT1("nirdebuginfo", RADV_DEBUG_NIR_DEBUG_INFO),
   OPT1("dump_trap_handler", RADV_DEBUG_DUMP_TRAP_HANDLER),
   OPT1("vs", RADV_DEBUG_DUMP_VS),
   OPT1("tcs", RADV_DEBUG_DUMP_TCS),
   OPT1("tes", RADV_DEBUG_DUMP_TES),
   OPT1("gs", RADV_DEBUG_DUMP_GS),
   OPT1("ps", RADV_DEBUG_DUMP_PS),
   OPT1("task", RADV_DEBUG_DUMP_TASK),
   OPT1("mesh", RADV_DEBUG_DUMP_MESH),
   OPT1("cs", RADV_DEBUG_DUMP_CS),
   OPT1("nir", RADV_DEBUG_DUMP_NIR),
   OPT1("asm", RADV_DEBUG_DUMP_ASM),
   OPT1("ir", RADV_DEBUG_DUMP_BACKEND_IR),
   OPT1("pso_history", RADV_DEBUG_PSO_HISTORY),
   OPT1("bvh4", RADV_DEBUG_BVH4),
   OPT1("novideo", RADV_DEBUG_NO_VIDEO),
   OPT1("validatevas", RADV_DEBUG_VALIDATE_VAS),
   OPT1("bo_history", RADV_DEBUG_DUMP_BO_HISTORY),
   OPT1("dumpibs", RADV_DEBUG_DUMP_IBS),
   OPT1("vm", RADV_DEBUG_VM),
   OPT1("nosmemmitigation", RADV_DEBUG_NO_SMEM_MITIGATION),
   OPT1("fullsync", RADV_DEBUG_FULL_SYNC),
   OPT1("notmz", RADV_DEBUG_NO_TMZ),
   OPT1("noheap", RADV_DEBUG_NO_HEAP),
   {
      NULL,
   }
#undef OPT1
#undef OPT2
};

const char *
radv_get_debug_option_name(int id)
{
   assert(id < RADV_DEBUG_COUNT);
   for (uint32_t i = 0; i < ARRAY_SIZE(radv_debug_options) - 1; i++) {
      /* Skip aliases like RADV_DEBUG_DUMP_SHADERS. */
      if (radv_debug_options[i].range[0] != radv_debug_options[i].range[1])
         continue;

      if (radv_debug_options[i].range[0] == (uint32_t)id)
         return radv_debug_options[i].string;
   }
   return NULL;
}

static const struct debug_control radv_perftest_options[] = {
   {"localbos", RADV_PERFTEST_LOCAL_BOS},
   {"dccmsaa", RADV_PERFTEST_DCC_MSAA},
   {"cswave32", RADV_PERFTEST_CS_WAVE_32},
   {"pswave32", RADV_PERFTEST_PS_WAVE_32},
   {"gewave32", RADV_PERFTEST_GE_WAVE_32},
   {"nosam", RADV_PERFTEST_NO_SAM},
   {"sam", RADV_PERFTEST_SAM},
   {"nggc", RADV_PERFTEST_NGGC},
   {"rtwave64", RADV_PERFTEST_RT_WAVE_64},
   {"dmashaders", RADV_PERFTEST_DMA_SHADERS},
   {"nircache", RADV_PERFTEST_NIR_CACHE},
   {"nogttspill", RADV_PERFTEST_NO_GTT_SPILL},
   {"rtcps", RADV_PERFTEST_RT_CPS},
   {"lowlatencydec", RADV_PERFTEST_LOWLATENCYDEC},
   {"lowlatencyenc", RADV_PERFTEST_LOWLATENCYENC},
   {NULL, 0},
};

static const struct debug_control radv_experimental_options[] = {
   {"emulate_rt", RADV_EXPERIMENTAL_EMULATE_RT},
   {"video_decode", RADV_EXPERIMENTAL_VIDEO_DECODE},
   {"transfer_queue", RADV_EXPERIMENTAL_TRANSFER_QUEUE},
   {"video_encode", RADV_EXPERIMENTAL_VIDEO_ENCODE},
   {"hic", RADV_EXPERIMENTAL_HIC},
   {"sparse", RADV_EXPERIMENTAL_SPARSE},
   {"bfloat16", RADV_EXPERIMENTAL_BFLOAT16},
   {"msrtss", RADV_EXPERIMENTAL_MSRTSS},
   {"elf", RADV_EXPERIMENTAL_ELF},
   {NULL, 0},
};

static const struct debug_control radv_trap_excp_options[] = {
   {"mem_viol", RADV_TRAP_EXCP_MEM_VIOL},
   {"float_div_by_zero", RADV_TRAP_EXCP_FLOAT_DIV_BY_ZERO},
   {"float_overflow", RADV_TRAP_EXCP_FLOAT_OVERFLOW},
   {"float_underflow", RADV_TRAP_EXCP_FLOAT_UNDERFLOW},
   {NULL, 0},
};

// clang-format off
static const struct debug_control radv_queue_disable_options[] = {
   {"gfx", RADV_QUEUE_DISABLE_GENERAL},
   {"compute", RADV_QUEUE_DISABLE_COMPUTE},
   {"vdec", RADV_QUEUE_DISABLE_VIDEO_DEC},
   {"venc", RADV_QUEUE_DISABLE_VIDEO_ENC},
   {"transfer", RADV_QUEUE_DISABLE_TRANSFER},
   {"sparse", RADV_QUEUE_DISABLE_SPARSE},
};
// clang-format on

const char *
radv_get_perftest_option_name(int id)
{
   assert(id < ARRAY_SIZE(radv_perftest_options));
   for (uint32_t i = 0; i < ARRAY_SIZE(radv_perftest_options); i++) {
      if (radv_perftest_options[i].flag == (1ull << id))
         return radv_perftest_options[i].string;
   }
   return NULL;
}

static const struct debug_control trace_options[] = {
   {"rgp", RADV_TRACE_MODE_RGP},
   {"rra", RADV_TRACE_MODE_RRA},
   {"gamma", RADV_TRACE_MODE_GAMMA},
   {"ctxroll", RADV_TRACE_MODE_CTX_ROLLS},
   {"ranges", RADV_TRACE_MODE_RANGES},
   {NULL, 0},
};

static const struct vk_instance_extension_table radv_instance_extensions_supported = {
   .KHR_device_group_creation = true,
   .KHR_external_fence_capabilities = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_get_physical_device_properties2 = true,
   .EXT_debug_report = true,
   .EXT_debug_utils = true,

#ifdef RADV_USE_WSI_PLATFORM
   .KHR_get_surface_capabilities2 = true,
   .KHR_surface = true,
   .KHR_surface_maintenance1 = true,
   .KHR_surface_protected_capabilities = true,
   .EXT_surface_maintenance1 = true,
   .EXT_swapchain_colorspace = true,
#endif
#ifdef VK_USE_PLATFORM_WAYLAND_KHR
   .KHR_wayland_surface = true,
#endif
#ifdef VK_USE_PLATFORM_XCB_KHR
   .KHR_xcb_surface = true,
#endif
#ifdef VK_USE_PLATFORM_XLIB_KHR
   .KHR_xlib_surface = true,
#endif
#ifdef VK_USE_PLATFORM_XLIB_XRANDR_EXT
   .EXT_acquire_xlib_display = true,
#endif
#ifdef VK_USE_PLATFORM_DISPLAY_KHR
   .KHR_display = true,
   .KHR_get_display_properties2 = true,
   .EXT_direct_mode_display = true,
   .EXT_display_surface_counter = true,
   .EXT_acquire_drm_display = true,
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
   .EXT_headless_surface = true,
#endif
};

static enum radeon_ctx_pstate
radv_parse_pstate(const char *str)
{
   if (!strcmp(str, "peak")) {
      return RADEON_CTX_PSTATE_PEAK;
   } else if (!strcmp(str, "standard")) {
      return RADEON_CTX_PSTATE_STANDARD;
   } else if (!strcmp(str, "min_sclk")) {
      return RADEON_CTX_PSTATE_MIN_SCLK;
   } else if (!strcmp(str, "min_mclk")) {
      return RADEON_CTX_PSTATE_MIN_MCLK;
   } else {
      return RADEON_CTX_PSTATE_NONE;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_CreateInstance(const VkInstanceCreateInfo *pCreateInfo, const VkAllocationCallbacks *pAllocator,
                    VkInstance *pInstance)
{
   struct radv_instance *instance;
   VkResult result;

   if (!pAllocator)
      pAllocator = vk_default_allocator();

   instance = vk_zalloc(pAllocator, sizeof(*instance), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return vk_error(NULL, VK_ERROR_OUT_OF_HOST_MEMORY);

   struct vk_instance_dispatch_table dispatch_table;
   vk_instance_dispatch_table_from_entrypoints(&dispatch_table, &radv_instance_entrypoints, true);
   vk_instance_dispatch_table_from_entrypoints(&dispatch_table, &wsi_instance_entrypoints, false);

   result =
      vk_instance_init(&instance->vk, &radv_instance_extensions_supported, &dispatch_table, pCreateInfo, pAllocator);
   if (result != VK_SUCCESS) {
      vk_free(pAllocator, instance);
      return vk_error(NULL, result);
   }

   vk_instance_add_driver_trace_modes(&instance->vk, trace_options);

   simple_mtx_init(&instance->shader_dump_mtx, mtx_plain);

   parse_debug_bitset(os_get_option("RADV_DEBUG"), radv_debug_options, instance->debug_flags);
   instance->perftest_flags = parse_debug_string(os_get_option("RADV_PERFTEST"), radv_perftest_options);
   instance->experimental_flags = parse_debug_string(os_get_option("RADV_EXPERIMENTAL"), radv_experimental_options);
   instance->trap_excp_flags = parse_debug_string(os_get_option("RADV_TRAP_HANDLER_EXCP"), radv_trap_excp_options);
   instance->profile_pstate = radv_parse_pstate(debug_get_option("RADV_PROFILE_PSTATE", "peak"));
   instance->queue_disable_flags = parse_debug_string(os_get_option("RADV_QUEUE_DISABLE"), radv_queue_disable_options);

   const bool has_shader_stage_flags =
      RADV_DEBUG(instance, DUMP_VS) || RADV_DEBUG(instance, DUMP_TCS) || RADV_DEBUG(instance, DUMP_TES) ||
      RADV_DEBUG(instance, DUMP_GS) || RADV_DEBUG(instance, DUMP_PS) || RADV_DEBUG(instance, DUMP_TASK) ||
      RADV_DEBUG(instance, DUMP_MESH) || RADV_DEBUG(instance, DUMP_CS);

   const bool has_compilation_stage_flags =
      RADV_DEBUG(instance, DUMP_SPIRV) || RADV_DEBUG(instance, DUMP_NIR) || RADV_DEBUG(instance, DUMP_PREOPT_IR) ||
      RADV_DEBUG(instance, DUMP_BACKEND_IR) || RADV_DEBUG(instance, DUMP_ASM);

   if (has_shader_stage_flags && !has_compilation_stage_flags) {
      /* When shader stages are specified but compilation stages aren't:
       * use a default set of compilation stages.
       */
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_NIR);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_BACKEND_IR);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_ASM);
   } else if (!has_shader_stage_flags && has_compilation_stage_flags) {
      /* When compilation stages are specified but shader stages aren't:
       * dump all shader stages.
       */
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_VS);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_TCS);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_TES);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_GS);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_PS);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_TASK);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_MESH);
      BITSET_SET(instance->debug_flags, RADV_DEBUG_DUMP_CS);
   }

   if (RADV_DEBUG(instance, PSO_HISTORY)) {
      const char *filename = "/tmp/radv_pso_history.log";

      instance->pso_history_logfile = fopen(filename, "w");
      if (!instance->pso_history_logfile)
         fprintf(stderr, "radv: Failed to open log file: %s.\n", filename);
   }

   instance->vk.physical_devices.try_create_for_drm = create_drm_physical_device;
   instance->vk.physical_devices.destroy = radv_physical_device_destroy;

   if (RADV_DEBUG(instance, STARTUP))
      fprintf(stderr, "radv: info: Created an instance.\n");

   VG(VALGRIND_CREATE_MEMPOOL(instance, 0, false));

   *pInstance = radv_instance_to_handle(instance);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
radv_DestroyInstance(VkInstance _instance, const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(radv_instance, instance, _instance);

   if (!instance)
      return;

   VG(VALGRIND_DESTROY_MEMPOOL(instance));

   if (instance->pso_history_logfile)
      fclose(instance->pso_history_logfile);

   simple_mtx_destroy(&instance->shader_dump_mtx);

   vk_instance_finish(&instance->vk);
   vk_free(&instance->vk.alloc, instance);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_EnumerateInstanceExtensionProperties(const char *pLayerName, uint32_t *pPropertyCount,
                                          VkExtensionProperties *pProperties)
{
   if (pLayerName)
      return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);

   return vk_enumerate_instance_extension_properties(&radv_instance_extensions_supported, pPropertyCount, pProperties);
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_EnumerateInstanceVersion(uint32_t *pApiVersion)
{
   *pApiVersion = RADV_API_VERSION;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
radv_EnumerateInstanceLayerProperties(uint32_t *pPropertyCount, VkLayerProperties *pProperties)
{
   if (pProperties == NULL) {
      *pPropertyCount = 0;
      return VK_SUCCESS;
   }

   /* None supported at this time */
   return vk_error(NULL, VK_ERROR_LAYER_NOT_PRESENT);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
radv_GetInstanceProcAddr(VkInstance _instance, const char *pName)
{
   VK_FROM_HANDLE(vk_instance, instance, _instance);
   return vk_instance_get_proc_addr(instance, &radv_instance_entrypoints, pName);
}

/* The loader wants us to expose a second GetInstanceProcAddr function
 * to work around certain LD_PRELOAD issues seen in apps.
 */
PUBLIC
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *pName)
{
   return radv_GetInstanceProcAddr(instance, pName);
}
