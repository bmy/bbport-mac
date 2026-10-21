/*
 * Copyright (c) 2015-2023 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "nir.h"

#ifdef __cplusplus
extern "C" {
#endif

struct intel_device_info;

#define intel_nir_tess_field(b, field)                                  \
   nir_ubitfield_extract_imm(b, nir_load_tess_config_intel(b),          \
                             INTEL_TESS_CONFIG_##field##_OFFSET,        \
                             INTEL_TESS_CONFIG_##field##_SIZE)

typedef nir_def *(*intel_nir_rt_write_cb)(nir_builder *, signed rt, void *data);

void intel_nir_apply_tcs_quads_workaround(nir_shader *nir);
bool intel_nir_rebase_const_offset_ubo_loads(nir_shader *shader);
bool intel_nir_blockify_uniform_loads(nir_shader *shader,
                                      const struct intel_device_info *devinfo);
bool intel_nir_clamp_image_1d_2d_array_sizes(nir_shader *shader);
bool intel_nir_clamp_per_vertex_loads(nir_shader *shader);
bool intel_nir_cleanup_resource_intel(nir_shader *shader);

bool intel_nir_lower_fragment_outputs(nir_shader *shader,
                                      unsigned nr_colour_regions,
                                      bool replicate_alpha,
                                      intel_nir_rt_write_cb cb,
                                      void *cb_data);
bool intel_nir_lower_non_uniform_barycentric_at_sample(nir_shader *nir);
bool intel_nir_lower_non_uniform_resource_intel(nir_shader *shader);
bool intel_nir_lower_patch_vertices_in(nir_shader *shader,
                                       unsigned input_vertices);
bool intel_nir_lower_patch_vertices_tes(nir_shader *shader);

bool intel_nir_lower_shading_rate_output(nir_shader *nir);
bool intel_nir_lower_sparse_intrinsics(nir_shader *nir, bool jay);

bool intel_nir_lower_vec2_surface_sampler(nir_shader *nir);

bool intel_nir_opt_peephole_ffma(nir_shader *shader);
bool intel_nir_opt_peephole_imul32x16(nir_shader *shader);

enum intel_atomic_branch_cases {
   /* Skip atomic add/umax if value is zero. */
   INTEL_ATOMIC_BRANCH_SKIP_ON_ZERO = 1 << 0,

   /* Load memory before umax/imax and skip if value is smaller. */
   INTEL_ATOMIC_BRANCH_MAX = 1 << 1,

   /* Load memory before umin/imin and skip if value is larger. */
   INTEL_ATOMIC_BRANCH_MIN = 1 << 2,
};

bool intel_nir_opt_atomic_branch(nir_shader *shader, unsigned enabled_cases);

bool intel_nir_pulls_at_sample(nir_shader *shader);

unsigned intel_nir_split_conversions_cb(const nir_instr *instr, void *data);

bool intel_nir_lower_printf(nir_shader *nir);

#ifdef __cplusplus
}
#endif
