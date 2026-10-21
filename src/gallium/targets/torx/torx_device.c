/*
 * Copyright (c) 2025 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "pipe/p_state.h"

#include "torx_device.h"

/* Layout of the structs the Python ctypes mirror in gallium.py depends
 * on. The entries must stay in the same order as the mirror's expected
 * list; gallium.py compares them at import so a drifted mirror fails
 * loudly instead of corrupting operation structs silently. */
const uint32_t *
torx_abi_layout(unsigned *count)
{
   static const uint32_t layout[] = {
      sizeof(struct pipe_tensor),
      offsetof(struct pipe_tensor, data),
      offsetof(struct pipe_tensor, index),
      offsetof(struct pipe_tensor, dims),
      offsetof(struct pipe_tensor, rank),
      offsetof(struct pipe_tensor, scale),
      offsetof(struct pipe_tensor, scales),
      offsetof(struct pipe_tensor, zero_point),
      offsetof(struct pipe_tensor, zero_points),
      offsetof(struct pipe_tensor, is_signed),
      offsetof(struct pipe_tensor, is_constant),
      offsetof(struct pipe_tensor, is_external_output),
      offsetof(struct pipe_tensor, type_size),
      sizeof(struct pipe_ml_operation),
      offsetof(struct pipe_ml_operation, type),
      offsetof(struct pipe_ml_operation, input_tensors),
      offsetof(struct pipe_ml_operation, input_count),
      offsetof(struct pipe_ml_operation, output_tensors),
      offsetof(struct pipe_ml_operation, output_count),
      offsetof(struct pipe_ml_operation, conv.weight_tensor),
      offsetof(struct pipe_ml_operation, conv.bias_tensor),
      offsetof(struct pipe_ml_operation, conv.stride_x),
      offsetof(struct pipe_ml_operation, conv.padding_top),
      offsetof(struct pipe_ml_operation, conv.pointwise),
      offsetof(struct pipe_ml_operation, conv.activation_min),
      offsetof(struct pipe_ml_operation, conv.activation_max),
      offsetof(struct pipe_ml_operation, conv.relu),
      offsetof(struct pipe_ml_operation, conv.dilation_width_factor),
      offsetof(struct pipe_ml_operation, conv.dilation_height_factor),
      offsetof(struct pipe_ml_operation, add.relu),
      sizeof(struct pipe_ml_device),
      offsetof(struct pipe_ml_device, id),
      offsetof(struct pipe_ml_device, ml_operation_supported),
      offsetof(struct pipe_ml_device, ml_subgraph_create),
      offsetof(struct pipe_ml_device, ml_subgraph_serialize),
      offsetof(struct pipe_ml_device, ml_subgraph_destroy),
      offsetof(struct pipe_ml_device, ml_device_destroy),
      PIPE_ML_OPERATION_TYPE_CONVOLUTION,
      PIPE_ML_OPERATION_TYPE_ADD,
      PIPE_ML_OPERATION_TYPE_MINIMUM,
      PIPE_ML_OPERATION_TYPE_MEAN,
   };

   *count = ARRAY_SIZE(layout);
   return layout;
}

#ifdef GALLIUM_ETHOSU
#include "ethosu/ethosu_public.h"
#endif

struct ml_driver_descriptor {
   const char *driver_name;
   struct pipe_ml_device *(*create_device)(const char *spec);
};

static const struct ml_driver_descriptor driver_descriptors[] = {
#ifdef GALLIUM_ETHOSU
   {.driver_name = "ethosu", .create_device = ethosu_ml_device_create},
#endif
};

const struct pipe_ml_device *
torx_device_probe(void)
{
   const char *env = getenv("MESA_ML_DEVICE"); /* DRIVER-GEN-MACS-SRAM, eg. "ethosu-65-256-98304" */
   char *driver_name = NULL;
   const char *spec = NULL;
   const struct ml_driver_descriptor *dd = NULL;
   struct pipe_ml_device *ml_device = NULL;

   if (env == NULL)
      return NULL;

   spec = strchr(env, '-');
   if (!spec)
      return NULL;

   size_t len = spec - env;
   driver_name = strndup(env, len);
   spec++; // skip '-'

   if (driver_name) {
      for (int i = 0; i < ARRAY_SIZE(driver_descriptors); i++) {
         if (strcmp(driver_descriptors[i].driver_name, driver_name) == 0) {
            dd = &driver_descriptors[i];

            ml_device = dd->create_device(spec);
            if (ml_device == NULL) {
               fprintf(stderr, "Failed to create ML device for spec '%s' using driver '%s'\n", spec, driver_name);
               free(driver_name);
               return NULL;
            }

            ml_device->id = env;
            break;
         }
      }
   }

   free(driver_name);

   return ml_device;
}
