/*
 * Copyright © 2026 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef HW_RUNNER_H
#define HW_RUNNER_H

#include <stdint.h>
#include "kmod/pan_kmod.h"

/* Arguments passed in FAU to the compute shader (aka CB0) */
struct hw_runner_shader_args {
   uint64_t data_addr;
   uint32_t data_stride;
   uint32_t _pad;
};

struct hw_runner_buffer_descr {
   uint64_t device_ptr;
   uint32_t size_B;
};

/* All data required to run a compute shader, every pointer is a read-only
 * pointer in CPU-space, except for data_ptr that will be uploaded to the GPU
 * for execution and copied back when the shader has run (read-write).
 */
struct hw_runner_invocation_info {
   /* This BO pointer might be null, if so the function will exit early and only
    * write the layout information so that the caller can allocate buffer object
    * and call back.
    */
   void *descr_bo_host_ptr;
   uint64_t descr_bo_device_ptr;

   void *code_ptr;
   uint64_t code_size_B;

   void *fau_ptr;
   uint64_t fau_size_B;
   /* Offset of hw_runner_shader_args in fau */
   uint64_t args_fau_offset;

   /* Simple Buffers (for DescriptorBuffers) */
   struct hw_runner_buffer_descr *simple_buffers;
   uint32_t simple_buffer_count;

   struct hw_runner_shader_args shader_args;

   uint8_t register_count;
   uint64_t register_preload;
   uint32_t invocations;
};

struct hw_runner_layout_info {
   uint64_t descr_bo_size_B;
   uint64_t cs_offset;
   uint64_t cs_size_B;
};

void hw_runner_new_cmd_stream(struct pan_kmod_dev *kdev,
                              struct hw_runner_invocation_info *info,
                              struct hw_runner_layout_info *out);

# endif /* HW_RUNNER_H */
