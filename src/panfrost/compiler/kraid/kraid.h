/* Copyright © 2026 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef KRAID_H
#define KRAID_H

#include "panfrost/compiler/pan_compiler.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t kraid_get_compiler_flags(void);

const nir_shader_compiler_options *
kraid_get_nir_shader_compiler_options(uint8_t arch, bool merge_wg);


/* State of index-driven vertex shading for current shader */
enum kraid_idvs_mode {
   /* IDVS not in use */
   KRAID_IDVS_NONE = 0,

   /* IDVS in use. Compiling a position shader */
   KRAID_IDVS_POSITION = 1,

   /* IDVS in use. Compiling a varying shader */
   KRAID_IDVS_VARYING = 2,

   /* IDVS2 in use. Compiling a deferred shader (v12+) */
   KRAID_IDVS_ALL = 3,
};

void kraid_compile_nir(nir_shader *nir,
                       const struct pan_compile_inputs *inputs,
                       struct util_dynarray *binary,
                       struct pan_shader_info *info,
                       enum kraid_idvs_mode idvs);

void kraid_disassemble(FILE *fp, const void *code, size_t size, bool verbose,
                       unsigned char arch);

#ifdef __cplusplus
} /* extern C */
#endif

#endif /* KRAID_H */
