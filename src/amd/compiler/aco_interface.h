/*
 * Copyright © 2018 Google
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ACO_INTERFACE_H
#define ACO_INTERFACE_H

#include "aco_shader_info.h"

#include "nir_defines.h"
#include "util/shader_stats.h"

#include "ac_binary.h"
#include "ac_shader_debug_info.h"
#include "amd_family.h"
#ifdef __cplusplus
extern "C" {
#endif

struct nir_parameter;
typedef struct nir_parameter nir_parameter;
struct aco_shader_info;
struct aco_vs_prolog_info;
struct aco_ps_epilog_info;
struct radeon_info;

typedef struct {
   struct ac_shader_config config;
   unsigned wave_size;

   const struct amd_stats *stats;

   const uint32_t* code;
   uint32_t code_dw;
   uint32_t exec_size;

   /* Part of [code,code+code_dw) */
   const uint8_t* constants;
   uint32_t constants_size;

   const char* ir_str;
   unsigned ir_size;

   const char* disasm_str;
   unsigned disasm_size;

   const struct aco_symbol* symbols;
   unsigned num_symbols;

   const struct ac_shader_debug_info* debug_info;
   unsigned debug_info_count;
} aco_callback_params;

typedef void(aco_callback)(void** priv_ptr, const aco_callback_params* params);

/* Note that this doesn't include the aco_symbol list or any debug info. */
size_t aco_create_elf(const struct ac_compiler_info* compiler_info,
                      const aco_callback_params* params, size_t size_before, size_t size_after,
                      void** data);

void aco_compile_shader(const struct aco_compiler_options* options,
                        const struct aco_shader_info* info, unsigned shader_count,
                        struct nir_shader* const* shaders, const struct ac_shader_args* args,
                        aco_callback* build_binary, void** binary);

void aco_compile_vs_prolog(const struct aco_compiler_options* options,
                           const struct aco_shader_info* info,
                           const struct aco_vs_prolog_info* prolog_info,
                           const struct ac_shader_args* args, aco_callback* build_prolog,
                           void** binary);

void aco_compile_ps_epilog(const struct aco_compiler_options* options,
                           const struct aco_shader_info* info,
                           const struct aco_ps_epilog_info* epilog_info,
                           const struct ac_shader_args* args, aco_callback* build_epilog,
                           void** binary);

void aco_compile_ps_prolog(const struct aco_compiler_options* options,
                           const struct aco_shader_info* info,
                           const struct aco_ps_prolog_info* pinfo,
                           const struct ac_shader_args* args, aco_callback* build_prolog,
                           void** binary);

void aco_compile_trap_handler(const struct aco_compiler_options* options,
                              const struct aco_shader_info* info, const struct ac_shader_args* args,
                              aco_callback* build_binary, void** binary);

uint64_t aco_get_codegen_flags();

bool aco_is_gpu_supported(const struct radeon_info* info);

void aco_print_asm(const struct radeon_info *info, unsigned wave_size,
                   uint32_t *binary, unsigned num_dw);

#ifdef __cplusplus
}
#endif

#endif
