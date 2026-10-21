/*
 * Copyright 2015 Advanced Micro Devices, Inc.
 * Copyright 2024 Valve Corporation
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef AC_DESCRIPTORS_H
#define AC_DESCRIPTORS_H

#include "ac_gpu_info.h"
#include "ac_surface.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DUPL_16BITS_IN_DWORD(x) (((x) << 16) | (x))
#define DUPL_8BITS_IN_DWORD(x) (((x) << 24) | ((x) << 16) | ((x) << 8) | (x))
#define DUPL_4BITS_IN_DWORD(x) DUPL_8BITS_IN_DWORD((x) | ((x) << 4))

#define DCC_CODE                       DUPL_8BITS_IN_DWORD
#define CMASK_NOAA_CODE                DUPL_4BITS_IN_DWORD
#define CMASK_MSAA_CODE(fmask, color)  DUPL_4BITS_IN_DWORD((fmask) | ((color) << 2))

enum
{
   /* DCC clear codes for all generations. */
   DCC_CLEAR_0000    = DCC_CODE(0x00), /* all bits are 0 */
   DCC_UNCOMPRESSED  = DCC_CODE(0xFF),

   /* DCC clear codes for GFX8-10. */
   GFX8_DCC_CLEAR_0000     = DCC_CLEAR_0000,
   GFX8_DCC_CLEAR_0001     = DCC_CODE(0x40),
   GFX8_DCC_CLEAR_1110     = DCC_CODE(0x80),
   GFX8_DCC_CLEAR_1111     = DCC_CODE(0xC0),
   GFX8_DCC_CLEAR_REG      = DCC_CODE(0x20),
   GFX9_DCC_CLEAR_SINGLE   = DCC_CODE(0x10),

   /* DCC clear codes for GFX11. */
   GFX11_DCC_CLEAR_SINGLE     = DCC_CODE(0x01),
   GFX11_DCC_CLEAR_0000       = DCC_CLEAR_0000, /* all bits are 0 */
   GFX11_DCC_CLEAR_1111_UNORM = DCC_CODE(0x02), /* all bits are 1 */
   GFX11_DCC_CLEAR_1111_FP16  = DCC_CODE(0x04), /* all 16-bit words are 0x3c00, max 64bpp */
   GFX11_DCC_CLEAR_1111_FP32  = DCC_CODE(0x06), /* all 32-bit words are 0x3f800000 */
   /* Color bits are 0, alpha bits are 1; only 88, 8888, 16161616 */
   GFX11_DCC_CLEAR_0001_UNORM = DCC_CODE(0x08),
   /* Color bits are 1, alpha bits are 0, only 88, 8888, 16161616 */
   GFX11_DCC_CLEAR_1110_UNORM = DCC_CODE(0x0A),
};

enum {
   /* Legacy color clear and FMASK clear/compression.
    * CMASK determines whether color is cleared to a clear color in a register.
    * CMASK with MSAA also handles FMASK clears and compression.
    */
   CMASK_NOAA_COLOR_CLEAR_REG = CMASK_NOAA_CODE(0x0), /* only valid on gfx6-9, illegal with DCC */
   CMASK_NOAA_COLOR_EXPANDED  = CMASK_NOAA_CODE(0xF), /* only valid on gfx6-9 */

   /* The first value is FMASK compression code, the second value is fast clear. */
   CMASK_MSAA_FMASK_CLEAR_0_COLOR_CLEAR_REG        = CMASK_MSAA_CODE(0, 0), /* illegal with DCC */
   CMASK_MSAA_FMASK_CLEAR_0_COLOR_EXPANDED         = CMASK_MSAA_CODE(0, 3),
   /* Different MSAA modes require different CMASK codes for "FMASK uncompressed". */
   CMASK_2xMSAA_FMASK_UNCOMPRESSED_COLOR_EXPANDED  = CMASK_MSAA_CODE(1, 3),
   CMASK_4xMSAA_FMASK_UNCOMPRESSED_COLOR_EXPANDED  = CMASK_MSAA_CODE(2, 3),
   CMASK_8xMSAA_FMASK_UNCOMPRESSED_COLOR_EXPANDED  = CMASK_MSAA_CODE(3, 3),
};

/* FMASK is bandwidth compression of MSAA color images taking advantage of the fact that some
 * samples in MSAA images have equal values. FMASK is essentially a mapping of logical samples to
 * physical samples represented as an array of sample indices per pixel that can map multiple equal
 * logical samples to 1 physical sample to save bandwidth. 8x MSAA stores 8 4-bit sample indices
 * (only using range [0, 7]) in FMASK that map logical samples to physical samples, 4x MSAA stores
 * 4 2-bit sample indices (range [0, 3]) in FMASK, and 2x MSAA stores 2 1-bit sample indices (range
 * [0, 1]) in FMASK.
 *
 * FMASK uses a physical layout identical to UINT8, UINT16, and UINT32 images and it's effectively
 * an additional plane of the color image. Regardless of the element encoding in memory, FMASK is
 * always loaded to the shader as UINT32 (or UINT16 with D16). For example, FMASK value 0 means all
 * logical samples map to physical sample 0, while FMASK values 0x76543210, 0x3210, and 0x10 are
 * identity mappings for 8 samples, 4 samples, and 2 samples, respectively. The physical layout uses
 * a tighter bit packing for 2 and 4 samples, which is hidden from the shader.
 *
 * Some generations don't have FMASK and instead fully rely on DCC for MSAA compression.
 *
 * Usage:
 * - Rasterization is the only path that results in automatic optimal MSAA compression on GPUs with
 *   FMASK. Compute-shader-based image clears and copies must compare sample values, store only
 *   unique physical samples, and finally store FMASK manually to get the same bandwidth usage
 *   reduction.
 * - MSAA image stores are unaware of FMASK and always store to physical samples directly. For image
 *   stores, the driver can either "expand" FMASK to an identity mapping and store to physical samples
 *   as if they were logical samples, or store only unique physical samples and then store the FMASK
 *   value manually to set the mapping from logical samples to physical samples.
 * - MSAA image loads must first load the FMASK value, use that to remap the logical sample to
 *   a physical sample, and then load the physical sample. This indirection adds latency and can be
 *   skipped if we know in advance that FMASK is identity.
 *
 * (NOTE: FMASK image stores are untested, but we expect that they work either as-is or by
 * reinterpreting the FMASK image format as UINT8/16/32.)
 *
 * While FMASK compresses the MSAA color image, FMASK itself can also be compressed by CMASK similar
 * to how DCC compresses image data. CMASK is FMASK metadata and is cached by CB_META and
 * L2_METADATA caches similar to how DCC is color metadata. CMASK can put FMASK in the following
 * states:
 * - cleared to 0: All logical samples map to physical sample 0.
 * - compressed: CMASK compresses the FMASK image in a proprietary manner similar to DCC.
 * - decompressed: FMASK is a regular UINT8, UINT16, or UINT32 image.
 *
 * TC-compatible CMASK removes the middle "compressed" state.
 *
 * CMASK can also contain fast clear state for the MSAA color image predating DCC fast clear, but
 * that's unrelated to FMASK. If that's used, CMASK provides fast clear for both FMASK and the color
 * image simultaneously.
 *
 * Definitions:
 * - "MSAA compression" means FMASK compresses the MSAA color image.
 * - "FMASK compression" means CMASK compresses the FMASK image.
 * - "FMASK fast clear" means that CMASK is in a state that indicates that FMASK is cleared to 0.
 * - "fast color clear" means that CMASK and/or DCC are in a state that indicates that the color
 *   (MSAA) image is cleared (to some value) and FMASK is cleared to 0. (thus it includes "FMASK
 *   fast clear")
 * - "FMASK decompression" means that the FMASK image is transitioned to the decompressed state
 *   to enable shader access. Only TC-compatible CMASK doesn't require FMASK decompression before
 *   shader access since it's never in the compressed state.
 * - "FMASK expansion" means that the FMASK image is transitioned to an identity mapping, which
 *   is done by moving/duplicating the physical samples in the MSAA image to match the identity
 *   mapping, and clearing FMASK to the corresponding identity value.
 *
 * In NIR, nir_texop_fragment_mask_fetch_amd and nir_intrinsic_*_fragment_mask_load_amd load from
 * FMASK, while nir_texop_fragment_fetch_amd and ACCESS_FMASK_LOWERED_AMD (for images) load
 * physical samples from the color image. The corresponding lowering is done by nir_lower_tex and
 * nir_lower_image. If the lowering is skipped, the image coordinates end up referencing physical
 * samples instead of logical samples. Image stores and atomics are not lowered and therefore
 * always reference physical samples.
 *
 * Since Z/S don't have FMASK, ACO and ac_nir_to_llvm conditionally replace the loaded FMASK value
 * with 0x76543210 if the FMASK descriptor is NULL for tex opcodes only, thus forcing the identity
 * mapping for Z/S samplers. That still costs VMEM latency even though nothing is loaded from
 * memory. This is not done for image opcodes.
 *
 * EQAA increases quality by increasing the number of logical samples in FMASK that can map to
 * physical samples. For example, EQAA 16S 8F (16 samples, 8 fragments) means that FMASK has
 * 16 logical samples and the MSAA image has 8 physical samples. In that example, each physical
 * sample can contribute any multiple of 1/16 of itself to the resolved color value instead of only
 * any multiple of 1/8 of itself, giving the impression of 16x MSAA as long as there are at most
 * 8 unique samples. If more than 8 unique samples are needed, FMASK stores a special sample index
 * meaning "unknown" that's equal to the value of the last sample index + 1. The HW determines
 * which drawn color samples are dropped and get the "unknown" value in FMASK when the number of
 * samples that's needed exceeds the number of physical samples. Analogous logic applies to any
 * scenario where the number of physical color samples is less than the number of rasterization
 * samples, and even 1 physical sample with multiple logical samples is possible. Z/S supports
 * up to 16 samples to match rasterization samples for precise coverage determination. The MSAA
 * resolve pass can optionally ignore logical samples that say "unknown". This type of MSAA was
 * mostly popular during the DX11 era and is being phased out today. GFX10.3 is the last generation
 * that supports it.
 */
enum {
   /* Don't ever use this. Clear CMASK instead. */
   FMASK_CLEAR_0 = 0,

   /* These can be used only if FMASK is uncompressed in CMASK.
    *
    * Uncompressed doesn't mean expanded.
    * - FMASK compression only affects bandwidth, not stored values. The compression is done by CMASK.
    * - Expanded FMASK means that specific values are stored in it such that FMASK has no effect.
    *   FMASK expansion is a layout transition only required before MSAA image stores.
    * - CB_FMASK_DECOMPRESS is a layout transition required before any shader access, and does:
    *     1. FMASK decompression: Eliminating CMASK compression.
    *     2. CMASK fast color clear elimination: Writing the clear value in CB_COLORi_CLEAR_WORDj
    *        registers to cleared areas of the color image.
    *     ! It doesn't do FMASK expansion, which must be done by a compute shader.
    * - GFX8-10.3: To avoid CB_FMASK_DECOMPRESS before shader access (except MSAA image stores):
    *    - Use FMASK with TC-compatible CMASK. (enabled by FMASK_COMPRESS_1FRAG_ONLY)
    *    - Use DCC for fast MSAA color clears instead of CMASK.
    */
   FMASK_2xMSAA_EXPANDED = DUPL_8BITS_IN_DWORD(0x02),
   FMASK_4xMSAA_EXPANDED = DUPL_8BITS_IN_DWORD(0xE4),
   FMASK_8xMSAA_EXPANDED = 0x76543210,

   FMASK_EQAA_2S_1F_EXPANDED = FMASK_2xMSAA_EXPANDED,
   FMASK_EQAA_4S_1F_EXPANDED = DUPL_8BITS_IN_DWORD(0x0E),
   FMASK_EQAA_8S_1F_EXPANDED = DUPL_8BITS_IN_DWORD(0xFE),
   FMASK_EQAA_16S_1F_EXPANDED = DUPL_16BITS_IN_DWORD(0xFFFE),

   FMASK_EQAA_4S_2F_EXPANDED = DUPL_8BITS_IN_DWORD(0xA4),
   FMASK_EQAA_8S_2F_EXPANDED = DUPL_16BITS_IN_DWORD(0xAAA4),
   FMASK_EQAA_16S_2F_EXPANDED = 0xAAAAAAA4,

   FMASK_EQAA_8S_4F_EXPANDED = 0x44443210,
   FMASK_EQAA_16S_4F_EXPANDED = 0x4444444444443210ull, /* 8-byte clear value */

   /* Enums don't allow such large numbers. */
   #define FMASK_EQAA_16S_8F_EXPANDED 0x8888888876543210ull /* 8-byte clear value */
};

typedef union {
   /* Z only */
   struct {
      unsigned zmask : 4;
      unsigned minz : 14;
      unsigned maxz : 14;
   } z;

   struct {
      unsigned zmask : 4;
      /* SR0/SR1 contain stencil pretest results. */
      unsigned sr0 : 2;
      unsigned sr1 : 2;
      unsigned smem : 2;
      unsigned unused : 2;
      /* The Z Range consists of a 6-bit delta and 14-bit base.
       * ZRANGE_PRECISION determines whether zbase means minZ or maxZ.
       */
      unsigned zdelta : 6;
      unsigned zbase : 14;
   } zs;

   /* Z + VRS. VRS fields are 0-based: (0, 0) means VRS 1x1. */
   struct {
      unsigned zmask : 4;
      unsigned sr0 : 2;
      unsigned vrs_x : 2;
      unsigned smem : 2;
      unsigned vrs_y : 2;
      unsigned zdelta : 6;
      unsigned zbase : 14;
   } zs_vrs;

   uint32_t dword;
} ac_htile_dword;

#define HTILE_Z_CODE(...)        ((ac_htile_dword){.z = {__VA_ARGS__}}).dword
#define HTILE_ZS_CODE(...)       ((ac_htile_dword){.zs = {__VA_ARGS__}}).dword
#define HTILE_ZS_VRS_CODE(...)   ((ac_htile_dword){.zs_vrs = {__VA_ARGS__}}).dword

/* depth must be in [0, 1]. This only clears HiZ and sets the Z/S state to "cleared".
 * The DB register contain the full clear values.
 */
#define HTILE_Z_CLEAR_REG(depth)  HTILE_Z_CODE( \
   .zmask = 0, \
   .minz = lroundf((depth) * 0x3FFF), \
   .maxz = lroundf((depth) * 0x3FFF))

#define HTILE_ZS_CLEAR_REG(depth)  HTILE_ZS_CODE( \
   .zmask = 0, \
   .sr0 = 0x3, \
   .sr1 = 0x3, \
   .smem = 0, \
   .zdelta = 0, \
   .zbase = lroundf((depth) * 0x3FFF))

#define HTILE_ZS_VRS_CLEAR_REG(depth)  HTILE_ZS_VRS_CODE( \
   .zmask = 0, \
   .sr0 = 0x3, \
   .smem = 0, \
   .zdelta = 0, \
   .zbase = lroundf((depth) * 0x3FFF), \
   .vrs_x = 0, /* VRS = 1x1 (0-based) */ \
   .vrs_y = 0)

/* Zmask = Z uncompressed, minZ = 0, maxZ = 1. */
#define HTILE_Z_UNCOMPRESSED  HTILE_Z_CODE( \
   .zmask = 0xF, \
   .minz = 0, \
   .maxz = 0x3FFF)

/* Zmask = Z uncompressed, SR0/SR1 = Stencil pretest is unknown, Smem = Stencil uncompressed,
 * ZRange = [0, 1].
 */
#define HTILE_ZS_UNCOMPRESSED  HTILE_ZS_CODE( \
   .zmask = 0xF, \
   .sr0 = 0x3, \
   .sr1 = 0x3, \
   .smem = 0x3, \
   .zdelta = 0x3F, \
   .zbase = 0x3FFF)

/* Zmask = Z uncompressed, SR0 = Stencil pretest is unknown, Smem = Stencil uncompressed,
 * ZRange = [0, 1], VRS = 1x1 (0-based).
 */
#define HTILE_ZS_VRS_UNCOMPRESSED  HTILE_ZS_VRS_CODE( \
   .zmask = 0xF, \
   .sr0 = 0x3, \
   .smem = 0x3, \
   .zdelta = 0x3F, \
   .zbase = 0x3FFF, \
   .vrs_x = 0, /* VRS = 1x1 (0-based) */ \
   .vrs_y = 0)

void
ac_print_htile_dword(uint32_t htile_code, bool has_stencil, bool vrs, bool zrange_precision, FILE *f);

unsigned
ac_map_swizzle(unsigned swizzle);

struct ac_sampler_state {
   unsigned address_mode_u : 3;
   unsigned address_mode_v : 3;
   unsigned address_mode_w : 3;
   unsigned max_aniso_ratio : 3;
   unsigned depth_compare_func : 3;
   unsigned unnormalized_coords : 1;
   unsigned cube_wrap : 1;
   unsigned trunc_coord : 1;
   unsigned filter_mode : 2;
   unsigned mag_filter : 2;
   unsigned min_filter : 2;
   unsigned mip_filter : 2;
   unsigned aniso_single_level : 1;
   unsigned border_color_type : 2;
   unsigned border_color_ptr : 12;
   float min_lod;
   float max_lod;
   float lod_bias;
};

void
ac_build_sampler_descriptor(const enum amd_gfx_level gfx_level,
                            const struct ac_sampler_state *state,
                            uint32_t desc[4]);

struct ac_fmask_state {
   const struct radeon_surf *surf;
   uint64_t va;
   uint32_t width : 16;
   uint32_t height : 16;
   uint32_t depth : 14;
   uint32_t type : 4;
   uint32_t first_layer : 14;
   uint32_t last_layer : 13;

   uint32_t num_samples : 5;
   uint32_t num_storage_samples : 4;
   uint32_t tc_compat_cmask : 1;
};

void
ac_build_fmask_descriptor(const enum amd_gfx_level gfx_level,
                          const struct ac_fmask_state *state,
                          uint32_t desc[8]);

struct ac_texture_state {
   const struct radeon_surf *surf;
   enum pipe_format format;
   enum pipe_format img_format;
   uint32_t width : 17;
   uint32_t height : 17;
   uint32_t depth : 15;
   uint32_t type : 4;
   enum pipe_swizzle swizzle[4];
   uint32_t num_samples : 5;
   uint32_t num_storage_samples : 5;
   uint32_t first_level : 4;
   uint32_t last_level : 5;
   uint32_t num_levels : 6;
   uint32_t first_layer : 14;
   uint32_t last_layer : 13;
   float min_lod;

   struct {
      const struct ac_surf_nbc_view *nbc_view;
      uint32_t uav3d : 1;
      uint32_t upgraded_depth : 1;
   } gfx10;

   uint32_t dcc_enabled : 1;
   uint32_t tc_compat_htile_enabled : 1;
   uint32_t aniso_single_level : 1;
};

void
ac_build_texture_descriptor(const struct radeon_info *info,
                            const struct ac_texture_state *state,
                            uint32_t desc[8]);

uint32_t
ac_tile_mode_index(const struct radeon_surf *surf,
                   unsigned level,
                   bool stencil);

struct ac_mutable_tex_state {
   const struct radeon_surf *surf;
   uint64_t va;

   struct {
      const struct ac_surf_nbc_view *nbc_view;
      uint32_t write_compress_enable : 1;
      uint32_t iterate_256 : 1;
   } gfx10;

   struct {
      const struct legacy_surf_level *base_level_info;
      uint32_t base_level;
      uint32_t block_width;
   } gfx6;

   uint32_t is_stencil : 1;
   uint32_t dcc_enabled : 1;
   uint32_t tc_compat_htile_enabled : 1;
};

void
ac_set_mutable_tex_desc_fields(const struct radeon_info *info,
                               const struct ac_mutable_tex_state *state,
                               uint32_t desc[8]);

struct ac_buffer_state {
   uint64_t va;
   uint32_t size;
   enum pipe_format format;
   enum pipe_swizzle swizzle[4];
   uint32_t stride;
   uint32_t swizzle_enable : 2;
   uint32_t element_size : 2;
   uint32_t index_stride : 2;
   uint32_t add_tid : 1;
   uint32_t gfx10_oob_select : 2;
   uint32_t has_desc_resource_level : 1;

   struct {
      uint32_t compression_en : 1;
      uint32_t write_compress_enable : 1;
   } gfx12;
};

void
ac_set_buf_desc_word3(const enum amd_gfx_level gfx_level,
                      const struct ac_buffer_state *state,
                      uint32_t *rsrc_word3);

void
ac_build_buffer_descriptor(const enum amd_gfx_level gfx_level,
                           const struct ac_buffer_state *state,
                           uint32_t desc[4]);

void
ac_build_raw_buffer_descriptor(const enum amd_gfx_level gfx_level,
                               bool has_desc_resource_level,
                               uint64_t va,
                               uint32_t size,
                               uint32_t desc[4]);

void
ac_build_attr_ring_descriptor(const enum amd_gfx_level gfx_level,
                              bool has_desc_resource_level,
                              uint64_t va,
                              uint32_t size,
                              uint32_t stride,
                              uint32_t desc[4]);

struct ac_ds_state {
   const struct radeon_surf *surf;
   uint64_t va;
   enum pipe_format format;
   uint32_t width : 17;
   uint32_t height : 17;
   uint32_t level : 5;
   uint32_t num_levels : 6;
   uint32_t num_samples : 5;
   uint32_t first_layer : 14;
   uint32_t last_layer : 14;

   uint32_t allow_expclear : 1;
   uint32_t stencil_only : 1;
   uint32_t z_read_only : 1;
   uint32_t stencil_read_only : 1;

   uint32_t htile_enabled : 1;
   uint32_t htile_stencil_disabled : 1;
   uint32_t vrs_enabled : 1;
};

struct ac_ds_surface {
   uint64_t db_depth_base;
   uint64_t db_stencil_base;
   uint32_t db_depth_view;
   uint32_t db_depth_size;
   uint32_t db_z_info;
   uint32_t db_stencil_info;

   union {
      struct {
         uint64_t hiz_base;
         uint32_t hiz_info;
         uint32_t hiz_size_xy;
         uint32_t db_depth_view1;
      } gfx12;

      struct {
         uint64_t db_htile_data_base;
         uint32_t db_depth_info;
         uint32_t db_depth_slice;
         uint32_t db_htile_surface;
         uint32_t db_z_info2;
         uint32_t db_stencil_info2;
      } gfx6;
   } u;
};

void
ac_init_ds_surface(const struct radeon_info *info, const struct ac_ds_state *state, struct ac_ds_surface *ds);

struct ac_mutable_ds_state {
   const struct ac_ds_surface *ds; /* original DS surface */
   enum pipe_format format;
   uint32_t tc_compat_htile_enabled : 1;
   uint32_t zrange_precision : 1;
   uint32_t no_d16_compression : 1;
};

void
ac_set_mutable_ds_surface_fields(const struct radeon_info *info, const struct ac_mutable_ds_state *state,
                                 struct ac_ds_surface *ds);

struct ac_cb_state {
   const struct radeon_surf *surf;
   enum pipe_format format;
   uint32_t width : 17;
   uint32_t height : 17;
   uint32_t first_layer : 14;
   uint32_t last_layer : 14;
   uint32_t num_layers : 14;
   uint32_t num_samples : 5;
   uint32_t num_storage_samples : 5;
   uint32_t base_level : 5;
   uint32_t num_levels : 6;

   struct {
      struct ac_surf_nbc_view *nbc_view;
   } gfx10;
};

struct ac_cb_surface {
   uint32_t cb_color_info;
   uint32_t cb_color_view;
   uint32_t cb_color_view2;
   uint32_t cb_color_attrib;
   uint32_t cb_color_attrib2; /* GFX9+ */
   uint32_t cb_color_attrib3; /* GFX10+ */
   uint32_t cb_dcc_control;
   uint64_t cb_color_base;
   uint64_t cb_color_cmask;
   uint64_t cb_color_fmask;
   uint64_t cb_dcc_base;
   uint32_t cb_color_slice;
   uint32_t cb_color_cmask_slice;
   uint32_t cb_color_fmask_slice;
   union {
      uint32_t cb_color_pitch; /* GFX6-GFX8 */
      uint32_t cb_mrt_epitch;  /* GFX9+ */
   };
};

void
ac_init_cb_surface(const struct radeon_info *info, const struct ac_cb_state *state, struct ac_cb_surface *cb);

struct ac_mutable_cb_state {
   const struct radeon_surf *surf;
   const struct ac_cb_surface *cb; /* original CB surface */
   uint64_t va;

   uint32_t base_level : 5;
   uint32_t num_samples : 5;

   uint32_t fmask_enabled : 1;
   uint32_t cmask_enabled : 1;
   uint32_t fast_clear_enabled : 1;
   uint32_t tc_compat_cmask_enabled : 1;
   uint32_t dcc_enabled : 1;

   struct {
      struct ac_surf_nbc_view *nbc_view;
   } gfx10;
};

void
ac_set_mutable_cb_surface_fields(const struct radeon_info *info, const struct ac_mutable_cb_state *state,
                                 struct ac_cb_surface *cb);

struct ac_gfx12_hiz_state {
   const struct radeon_surf *surf;
   uint64_t va;
   uint32_t type : 4;
   uint32_t num_samples : 5;
   uint32_t first_level : 4;
   uint32_t last_level : 5;
   uint32_t num_levels : 6;
   uint32_t first_layer : 13;
   uint32_t last_layer : 13;
};

void
ac_build_gfx12_hiz_descriptor(const struct ac_gfx12_hiz_state *state, uint32_t desc[8]);

#ifdef __cplusplus
}
#endif

#endif
