/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/macros.h"
#include "util/u_dynarray.h"
#include "util/u_inlines.h"

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <xf86drm.h>

#include "drm-uapi/ethosu_accel.h"

#include "ethosu_cmd.h"
#include "ethosu_lower.h"
#include "ethosu_ml.h"

struct ethosu_block ARCH_OFM_BLOCK_MAX = {64, 32, 128};
struct ethosu_block SUB_KERNEL_MAX = {8, 8, 65536};

unsigned
ethosu_lut_region(void)
{
   return LUT_REGION;
}

unsigned
ethosu_lut_address(struct ethosu_subgraph *subgraph,
                   unsigned activation, unsigned size)
{
   if (ethosu_ml_device(subgraph->base.device)->is_u65)
      return SHRAM_LUT_BASE(activation & 0xf);

   return ((activation >> 5) & 0x7) * size;
}

void
ethosu_dump_buffer(const uint8_t *ptr, char *name, int operation_nr,
                   int suboperation_nr, int offset, unsigned size)
{
   char buffer[255];

   snprintf(buffer, sizeof(buffer), "mesa-%s-%03u-%03u.bin", name, operation_nr,
            suboperation_nr);

   FILE *f = fopen(buffer, "wb");
   assert(f);
   fwrite(ptr + offset, 1, size, f);
   if (ferror(f)) {
      DBG("Error in writing to file: %s\n", strerror(errno));
   }
   fflush(f);
   fclose(f);
}

void
ethosu_register_tensor(struct ethosu_subgraph *subgraph,
                       const struct pipe_tensor *ptensor)
{
   struct ethosu_tensor new_tensor = {0};
   new_tensor.index = ptensor->index;
   new_tensor.shape.height = ptensor->dims[1];
   new_tensor.shape.width = ptensor->dims[2];
   new_tensor.shape.depth = ptensor->dims[3];
   new_tensor.layout = ETHOSU_LAYOUT_NHWC;
   new_tensor.batches = subgraph->batches;
   new_tensor.type_size = ptensor->type_size;
   util_dynarray_append(&subgraph->tensors, new_tensor);
}

struct ethosu_tensor *
ethosu_find_tensor(struct ethosu_subgraph *subgraph, unsigned tensor_idx)
{
   util_dynarray_foreach (&subgraph->tensors, struct ethosu_tensor, tensor) {
      if (tensor->index == tensor_idx) {
         return tensor;
      }
   }
   return NULL;
}

int
ethosu_round_up_to_multiple(int a, int b)
{
   return ((a + b - 1) / b) * b;
}

int
ethosu_round_up_divide(int a, int b)
{
   return (a + b - 1) / b;
}

int
ethosu_quantize_scale(double scale, int32_t *shift, bool reduced)
{
   int exponent = 0;
   double significand = frexp(scale, &exponent);
   int32_t quantized_scale = round(significand * (double)(1LL << 31));
   *shift = 31 - exponent;

   if (reduced) {
      quantized_scale = (quantized_scale >> 16) + (quantized_scale >> 15 & 1);
      // make sure reduced scale does not overflow
      quantized_scale = MIN2(quantized_scale, 0x7FFF);
      *shift -= 16;
   }

   if (*shift > 63) {
      if (quantized_scale > exp2(*shift - 63)) {
         quantized_scale = quantized_scale >> (*shift - 63);
         *shift = 63;
      } else {
         // Not possible to get back within bounds, set scale and shift to 0
         // as the shift would shift away all relevant bits anyway.
         quantized_scale = 0;
         *shift = 0;
      }
   } else if (*shift < 0 && quantized_scale < exp2(*shift + 32)) {
      quantized_scale = quantized_scale << (0 - *shift);
      *shift = 0;
   }

   return quantized_scale;
}

static bool
ethosu_subtract_supported(struct pipe_ml_device *pdevice,
                          const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *minuend = operation->input_tensors[0];
   const struct pipe_tensor *subtrahend = operation->input_tensors[1];
   const struct pipe_tensor *output = operation->output_tensors[0];
   bool minuend_broadcast = false;
   bool subtrahend_broadcast = false;

   for (int i = 1; i < 4; i++) {
      minuend_broadcast |= minuend->dims[i] != output->dims[i];
      subtrahend_broadcast |= subtrahend->dims[i] != output->dims[i];
   }

   /*
    * The subtrahend broadcasts freely from IFM2, so only a first operand
    * that must be broadcast constrains the lowering.
    */
   if (!minuend_broadcast)
      return true;

   if (ethosu_ml_device(pdevice)->is_u65) {
      /*
       * The first operand moves to IFM2, leaving the subtrahend in IFM,
       * which the U65 can broadcast only as a scalar. Reject a subtrahend
       * that would itself need a per-axis broadcast.
       */
      return !subtrahend_broadcast;
   }

   /*
    * The U85 broadcasts both operands independently, and a constant first
    * operand is placed in the coefficient region and broadcast from there,
    * so every shape is handled.
    */
   return true;
}

static bool
ethosu_batch_matmul_supported(const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *input = operation->input_tensors[0];
   const struct pipe_tensor *input2 = operation->input_tensors[1];
   const struct pipe_tensor *output = operation->output_tensors[0];
   unsigned rows;
   unsigned depth;
   unsigned columns;

   if (operation->batch_matmul.adj_x || input->dims[0] != 1 ||
       input2->dims[0] != 1 || output->dims[0] != 1 ||
       input->dims[1] != input2->dims[1] ||
       output->dims[1] != input->dims[1] || input->type_size != 1 ||
       input2->type_size != 1 || output->type_size != 1 ||
       !input->is_signed || !input2->is_signed || !output->is_signed)
      return false;

   rows = input->dims[2];
   depth = input->dims[3];
   columns = operation->batch_matmul.adj_y ? input2->dims[2] : input2->dims[3];

   return depth == (operation->batch_matmul.adj_y ? input2->dims[3] : input2->dims[2]) &&
          output->dims[2] == rows && output->dims[3] == columns;
}

static bool
ethosu_transpose_supported(const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *input = operation->input_tensors[0];
   const struct pipe_tensor *output = operation->output_tensors[0];
   const unsigned *perm = operation->transpose.perm;
   unsigned axes = 0;

   if (perm[0] != 0 || output->type_size != input->type_size ||
       output->is_signed != input->is_signed ||
       output->scale != input->scale ||
       output->zero_point != input->zero_point)
      return false;

   for (unsigned i = 0; i < 4; i++) {
      if (perm[i] >= 4 || (axes & BITFIELD_BIT(perm[i])) ||
          output->dims[i] != input->dims[perm[i]])
         return false;
      axes |= BITFIELD_BIT(perm[i]);
   }

   return true;
}

static bool
ethosu_pack_supported(const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *output;
   int axis = operation->conc.axis;

   if (axis < 1 || axis > 3 || !operation->input_count ||
       operation->output_count != 1)
      return false;

   output = operation->output_tensors[0];

   for (unsigned i = 0; i < operation->input_count; i++) {
      const struct pipe_tensor *pack_input = operation->input_tensors[i];

      if (pack_input->type_size != output->type_size ||
          pack_input->is_signed != output->is_signed ||
          pack_input->scale != output->scale ||
          pack_input->zero_point != output->zero_point)
         return false;

      for (unsigned dim = 0; dim < 4; dim++) {
         if (dim != axis && pack_input->dims[dim] != output->dims[dim])
            return false;
      }
   }

   return output->dims[axis] == operation->input_count;
}

static bool
ethosu_unpack_supported(const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *input;
   int axis = operation->split.axis;

   if (axis < 1 || axis > 3 || operation->input_count != 1 ||
       !operation->output_count)
      return false;

   input = operation->input_tensors[0];
   if (input->dims[axis] != operation->output_count)
      return false;

   for (unsigned i = 0; i < operation->output_count; i++) {
      const struct pipe_tensor *output = operation->output_tensors[i];

      if (input->type_size != output->type_size ||
          input->is_signed != output->is_signed ||
          input->scale != output->scale ||
          input->zero_point != output->zero_point || output->dims[axis] != 1)
         return false;

      for (unsigned dim = 0; dim < 4; dim++) {
         if (dim != axis && input->dims[dim] != output->dims[dim])
            return false;
      }
   }

   return true;
}

static bool
ethosu_resize_bilinear_supported(const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *input = operation->input_tensors[0];
   const struct pipe_tensor *output = operation->output_tensors[0];
   unsigned scale_y;
   unsigned scale_x;

   if (input->type_size != 1 || output->type_size != 1 ||
       input->is_signed != output->is_signed || input->scale != output->scale ||
       input->zero_point != output->zero_point ||
       operation->resize_bilinear.half_pixel_centers)
      return false;

   if (input->dims[1] == 1 && input->dims[2] == 1)
      return true;

   if (!operation->resize_bilinear.align_corners || input->dims[1] < 2 ||
       input->dims[2] < 2 || output->dims[1] < 2 || output->dims[2] < 2 ||
       (output->dims[1] - 1) % (input->dims[1] - 1) ||
       (output->dims[2] - 1) % (input->dims[2] - 1))
      return false;

   scale_y = (output->dims[1] - 1) / (input->dims[1] - 1);
   scale_x = (output->dims[2] - 1) / (input->dims[2] - 1);
   return scale_y == scale_x && util_is_power_of_two_nonzero(scale_y) &&
          scale_y <= 8;
}

static bool
ethosu_argmax_supported(struct pipe_ml_device *pdevice,
                        const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *input = operation->input_tensors[0];
   const struct pipe_tensor *output = operation->output_tensors[0];

   if (operation->argmax.axis != 3 || input->type_size != 1 ||
       output->type_size != 4 || input->dims[0] != 1 ||
       input->is_signed != output->is_signed || !input->dims[3] ||
       !input->dims[1] || !input->dims[2] || input->dims[2] > (1 << 16) ||
       (ethosu_ml_device(pdevice)->is_u65 && input->dims[3] > 127) ||
       (!ethosu_ml_device(pdevice)->is_u65 && input->dims[3] > (1 << 15)))
      return false;

   return output->dims[0] == input->dims[0] && output->dims[1] == 1 &&
          output->dims[2] == input->dims[1] &&
          output->dims[3] == input->dims[2];
}

static bool
ethosu_space_batch_supported(const struct pipe_ml_operation *operation)
{
   const struct pipe_tensor *input = operation->input_tensors[0];
   const struct pipe_tensor *output = operation->output_tensors[0];

   return input->type_size == 1 && output->type_size == 1 &&
          input->is_signed == output->is_signed && input->scale == output->scale &&
          input->zero_point == output->zero_point &&
          operation->space_batch.block_y > 1 &&
          operation->space_batch.block_x > 1 &&
          operation->space_batch.before_y >= 0 &&
          operation->space_batch.after_y >= 0 &&
          operation->space_batch.before_x >= 0 &&
          operation->space_batch.after_x >= 0;
}

bool
ethosu_ml_operation_supported(struct pipe_ml_device *pdevice,
                              const struct pipe_ml_operation *operation)
{
   bool supported = false;

   if (operation->input_tensors[0]->type_size == 4 ||
       (operation->output_tensors[0]->type_size == 4 &&
        operation->type != PIPE_ML_OPERATION_TYPE_ARGMAX))
      return false;

   switch (operation->type) {
   case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED: {
      struct pipe_tensor *input = operation->input_tensors[0];
      struct pipe_tensor *output = operation->output_tensors[0];
      struct pipe_tensor *weight = operation->fcon.weight_tensor;
      uint64_t input_size;
      uint64_t output_size;

      if (!weight || input->dims[0] != output->dims[0] ||
          weight->dims[0] != 1 || weight->dims[1] != 1 ||
          !weight->dims[2] || !weight->dims[3])
         break;

      input_size = (uint64_t)input->dims[1] * input->dims[2] *
                   input->dims[3];
      output_size = (uint64_t)output->dims[1] * output->dims[2] *
                    output->dims[3];
      supported = input_size % weight->dims[3] == 0 &&
                  output_size == input_size / weight->dims[3] *
                                    weight->dims[2];
      break;
   }
   case PIPE_ML_OPERATION_TYPE_BATCH_MATMUL:
      supported = ethosu_batch_matmul_supported(operation);
      break;
   case PIPE_ML_OPERATION_TYPE_PACK:
      supported = ethosu_pack_supported(operation);
      break;
   case PIPE_ML_OPERATION_TYPE_UNPACK:
      supported = ethosu_unpack_supported(operation);
      break;
   case PIPE_ML_OPERATION_TYPE_RESIZE_BILINEAR:
      supported = ethosu_resize_bilinear_supported(operation);
      break;
   case PIPE_ML_OPERATION_TYPE_ARGMAX:
      supported = ethosu_argmax_supported(pdevice, operation);
      break;
   case PIPE_ML_OPERATION_TYPE_SPACE_TO_BATCH:
   case PIPE_ML_OPERATION_TYPE_BATCH_TO_SPACE:
      supported = ethosu_space_batch_supported(operation);
      break;
   case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
      /*
       * Dilation is not yet implemented.
       */
      if (operation->conv.dilation_width_factor == 1 &&
          operation->conv.dilation_height_factor == 1 &&
          ((operation->conv.stride_x <= 3 &&
            operation->conv.stride_y <= 3) ||
           (operation->conv.padding_top == 0 &&
            operation->conv.padding_bottom == 0 &&
            operation->conv.padding_left == 0 &&
            operation->conv.padding_right == 0)))
         supported = true;

      break;
   }
   case PIPE_ML_OPERATION_TYPE_SUBTRACT:
      supported = ethosu_subtract_supported(pdevice, operation);
      break;
   case PIPE_ML_OPERATION_TYPE_MAXIMUM:
   case PIPE_ML_OPERATION_TYPE_MINIMUM:
   case PIPE_ML_OPERATION_TYPE_MUL:
   case PIPE_ML_OPERATION_TYPE_ADD:
   case PIPE_ML_OPERATION_TYPE_STRIDED_SLICE:
   case PIPE_ML_OPERATION_TYPE_LOGISTIC:
   case PIPE_ML_OPERATION_TYPE_TANH:
   case PIPE_ML_OPERATION_TYPE_HSWISH:
   case PIPE_ML_OPERATION_TYPE_LEAKY_RELU:
   case PIPE_ML_OPERATION_TYPE_QUANTIZE:
   case PIPE_ML_OPERATION_TYPE_RESHAPE:
      supported = true;
      break;
   case PIPE_ML_OPERATION_TYPE_POOLING:
      /* A stride above 3 has no valid encoding. Convolutions work around
       * this by unrolling, but pooling does not, so reject it.
       */
      supported = operation->pooling.stride_x <= 3 &&
                  operation->pooling.stride_y <= 3;
      break;
   case PIPE_ML_OPERATION_TYPE_PAD: {
      struct pipe_tensor *input = operation->input_tensors[0];
      struct pipe_tensor *output = operation->output_tensors[0];

      supported = output->dims[0] == input->dims[0] &&
                  output->dims[1] == input->dims[1] + operation->pad.before_y +
                                        operation->pad.after_y &&
                  output->dims[2] == input->dims[2] + operation->pad.before_x +
                                        operation->pad.after_x &&
                  output->dims[3] == input->dims[3] + operation->pad.before_z +
                                        operation->pad.after_z;
      break;
   }
   case PIPE_ML_OPERATION_TYPE_SCATTER_ND:
      supported = ethosu_scatter_nd_as_pad(operation, NULL);
      break;
   case PIPE_ML_OPERATION_TYPE_SOFTMAX: {
      struct pipe_tensor *input = operation->input_tensors[0];
      struct pipe_tensor *output = operation->output_tensors[0];

      supported = input->type_size == 1 &&
                  output->type_size == 1 &&
                  input->is_signed == output->is_signed &&
                  input->dims[3] > 1 &&
                  input->dims[3] <= SOFTMAX_MAX_DEPTH;
      break;
   }
   case PIPE_ML_OPERATION_TYPE_SPLIT: {
      struct pipe_tensor *input = operation->input_tensors[0];
      unsigned split_size = 0;

      if (operation->split.axis < 1 || operation->split.axis > 3 ||
          !operation->output_count)
         break;

      supported = true;
      for (unsigned i = 0; i < operation->output_count; i++) {
         struct pipe_tensor *output = operation->output_tensors[i];

         if (output->type_size != input->type_size ||
             output->is_signed != input->is_signed)
            supported = false;

         for (unsigned axis = 0; axis < 4; axis++) {
            unsigned expected = axis == operation->split.axis ? output->dims[axis] : input->dims[axis];

            if (output->dims[axis] != expected)
               supported = false;
         }

         split_size += output->dims[operation->split.axis];
      }
      supported &= split_size == input->dims[operation->split.axis];
      break;
   }
   case PIPE_ML_OPERATION_TYPE_MEAN: {
      struct pipe_tensor *input = operation->input_tensors[0];
      struct pipe_tensor *output = operation->output_tensors[0];
      unsigned hw_axes = BITFIELD_BIT(1) | BITFIELD_BIT(2);
      bool spatial_mean;
      bool depth_mean;

      spatial_mean = operation->mean.axes == hw_axes &&
                     output->dims[0] == input->dims[0] &&
                     output->dims[1] == 1 &&
                     output->dims[2] == 1 &&
                     output->dims[3] == input->dims[3] &&
                     /* This emits a single depthwise convolution and does
                      * not split large reductions, so the reduced height is
                      * capped at 64 and the kernel at 4096 elements.
                      */
                     input->dims[1] <= 64 &&
                     input->dims[1] * input->dims[2] <= 4096;
      depth_mean = operation->mean.axes == BITFIELD_BIT(3) &&
                   (input->dims[1] == 1 || input->dims[2] == 1) &&
                   output->dims[0] == input->dims[0] &&
                   output->dims[1] == input->dims[1] &&
                   output->dims[2] == input->dims[2] &&
                   output->dims[3] == 1 &&
                   /* The channel count becomes the depthwise kernel
                    * width, capped at 4096 elements. */
                   input->dims[3] <= 4096;
      supported = spatial_mean || depth_mean;
      break;
   }
   case PIPE_ML_OPERATION_TYPE_RESIZE: {
      /* NPU only supports 2x nearest neighbor upscaling */
      struct pipe_tensor *input = operation->input_tensors[0];
      struct pipe_tensor *output = operation->output_tensors[0];
      bool is_2x_height = (output->dims[1] == 2 * input->dims[1]);
      bool is_2x_width = (output->dims[2] == 2 * input->dims[2]);
      supported = is_2x_height && is_2x_width;
      break;
   }
   case PIPE_ML_OPERATION_TYPE_CONCATENATION:
      supported = operation->conc.axis <= 3 && operation->conc.axis >= -1;
      break;
   case PIPE_ML_OPERATION_TYPE_TRANSPOSE:
      supported = ethosu_transpose_supported(operation);
      break;
   case PIPE_ML_OPERATION_TYPE_RSQRT: {
      struct pipe_tensor *input = operation->input_tensors[0];
      struct pipe_tensor *output = operation->output_tensors[0];

      supported = input->type_size == 1 && output->type_size == 1 &&
                  input->is_signed && output->is_signed;
      break;
   }
   default:
      supported = false;
   }

   return supported;
}

struct pipe_ml_subgraph *
ethosu_ml_subgraph_create(struct pipe_ml_device *pdevice,
                          const struct pipe_ml_operation *poperations,
                          unsigned count)
{
   struct ethosu_subgraph *subgraph;

   subgraph = calloc(1, sizeof(*subgraph));
   subgraph->base.device = pdevice;

   subgraph->tensors = UTIL_DYNARRAY_INIT;
   subgraph->operations = UTIL_DYNARRAY_INIT;

   /* Allocate register state tracking arrays */
   subgraph->cmd0_state = calloc(ETHOSU_MAX_REG_INDEX, sizeof(*subgraph->cmd0_state));
   subgraph->cmd1_state = calloc(ETHOSU_MAX_REG_INDEX, sizeof(*subgraph->cmd1_state));
   subgraph->cmd0_valid = calloc(ETHOSU_MAX_REG_INDEX, sizeof(bool));
   subgraph->cmd1_valid = calloc(ETHOSU_MAX_REG_INDEX, sizeof(bool));
   if (!subgraph->cmd0_state || !subgraph->cmd1_state ||
       !subgraph->cmd0_valid || !subgraph->cmd1_valid) {
      free(subgraph->cmd0_state);
      free(subgraph->cmd1_state);
      free(subgraph->cmd0_valid);
      free(subgraph->cmd1_valid);
      free(subgraph);
      return NULL;
   }

   ethosu_lower_graph(subgraph, poperations, count);

   if (subgraph->failed) {
      util_dynarray_foreach (&subgraph->operations,
                             struct ethosu_operation, operation) {
         free(operation->kernel.scales);
         free(operation->kernel.zero_points);
      }
      util_dynarray_fini(&subgraph->operations);
      free(subgraph->cmd0_state);
      free(subgraph->cmd1_state);
      free(subgraph->cmd0_valid);
      free(subgraph->cmd1_valid);
      ethosu_ml_subgraph_destroy(pdevice, &subgraph->base);
      return NULL;
   }

   ethosu_emit_cmdstream(subgraph);

   if (subgraph->failed) {
      util_dynarray_foreach (&subgraph->operations,
                             struct ethosu_operation, operation) {
         free(operation->kernel.scales);
         free(operation->kernel.zero_points);
      }
      util_dynarray_fini(&subgraph->operations);
      free(subgraph->cmd0_state);
      free(subgraph->cmd1_state);
      free(subgraph->cmd0_valid);
      free(subgraph->cmd1_valid);
      ethosu_ml_subgraph_destroy(pdevice, &subgraph->base);
      return NULL;
   }

   util_dynarray_foreach (&subgraph->operations, struct ethosu_operation, operation) {
      free(operation->kernel.scales);
      free(operation->kernel.zero_points);
   }
   util_dynarray_fini(&subgraph->operations);

   free(subgraph->cmd0_state);
   free(subgraph->cmd1_state);
   free(subgraph->cmd0_valid);
   free(subgraph->cmd1_valid);

   if (DBG_ENABLED(ETHOSU_DBG_DUMP_BOS)) {
      uint64_t cmdstream_size = (subgraph->cursor - subgraph->cmdstream) *
      sizeof(*subgraph->cursor);

      ethosu_dump_buffer((uint8_t *)subgraph->cmdstream, "cmdstream", 0, 0, 0,
                         cmdstream_size);
   }

   return &subgraph->base;
}

uint8_t *
ethosu_ml_subgraph_serialize(struct pipe_ml_device *pdevice,
                             struct pipe_ml_subgraph *psubgraph,
                             size_t *size)
{
   struct ethosu_subgraph *subgraph = (struct ethosu_subgraph *)(psubgraph);
   uint64_t header_size = NUM_HEADER_FIELDS * sizeof(uint64_t);
   uint64_t tensors_size = util_dynarray_num_elements(&subgraph->tensors,
      struct ethosu_tensor) * NUM_TENSOR_FIELDS * sizeof(uint32_t);
   uint64_t cmdstream_size = (subgraph->cursor - subgraph->cmdstream) *
      sizeof(*subgraph->cursor);

   uint64_t coefs_size = subgraph->coefs_used * sizeof(*subgraph->coefs);
   uint64_t io_size = subgraph->io_used;
   uint64_t total_size = header_size + cmdstream_size + coefs_size +
      tensors_size;
   uint8_t *buffer, *cursor;

   buffer = malloc(total_size);
   if (!buffer)
      return NULL;

   cursor = buffer;

   uint64_t *header = (uint64_t *)cursor;
   header[0] = cmdstream_size;
   header[1] = coefs_size;
   header[2] = io_size;
   header[3] = tensors_size;
   cursor += header_size;

   uint32_t *tensors = (uint32_t *)cursor;
   util_dynarray_foreach(&subgraph->tensors, struct ethosu_tensor, tensor) {
      tensors[0] = tensor->index;
      tensors[1] = tensor->offset;
      tensors[2] = tensor->size;
      tensors += NUM_TENSOR_FIELDS;
   }
   cursor += tensors_size;

   memcpy(cursor, subgraph->cmdstream, cmdstream_size);
   cursor += cmdstream_size;

   if (coefs_size > 0)
      memcpy(cursor, subgraph->coefs, coefs_size);

   *size = total_size;
   return buffer;
}

static bool
prepare_for_submission(struct ethosu_subgraph *subgraph,
                       struct pipe_context *pcontext)
{
   int ret;
   subgraph->screen = ethosu_screen(pcontext->screen);
   struct ethosu_screen *screen = subgraph->screen;
   uint64_t cmdstream_size = (subgraph->cursor - subgraph->cmdstream) *
      sizeof(*subgraph->cursor);

   if (cmdstream_size) {
      struct drm_ethosu_cmdstream_bo_create cmd_bo_create = {
         .size = cmdstream_size,
         .data = (uintptr_t)subgraph->cmdstream,
      };

      ret = drmIoctl(screen->fd, DRM_IOCTL_ETHOSU_CMDSTREAM_BO_CREATE,
                     &cmd_bo_create);
      if (ret) {
         mesa_loge("ethosu: failed to create command stream BO "
                   "(size=%" PRIu64 ", errno=%d: %s)",
                   cmdstream_size, errno, strerror(errno));
         subgraph->failed = true;
         return false;
      }

      free(subgraph->cmdstream);
      subgraph->cmdstream = NULL;

      subgraph->cmdstream_bo = cmd_bo_create.handle;
   }

   DBG("subgraph->coefs_used %d\n", subgraph->coefs_used);
   if (subgraph->coefs_used > 0) {
      subgraph->coefs_rsrc = pipe_buffer_create(pcontext->screen, 0,
                                                PIPE_USAGE_DEFAULT,
                                                subgraph->coefs_used);
      if (!subgraph->coefs_rsrc) {
         mesa_loge("ethosu: failed to allocate coefficients BO "
                   "(size=%u)",
                   subgraph->coefs_used);
         subgraph->failed = true;
         return false;
      }

      pipe_buffer_write(pcontext, subgraph->coefs_rsrc, 0,
                        subgraph->coefs_used, subgraph->coefs);

      free(subgraph->coefs);
      subgraph->coefs = NULL;

      if (DBG_ENABLED(ETHOSU_DBG_DUMP_BOS)) {
         struct pipe_transfer *transfer_in;
         uint8_t *buf = pipe_buffer_map(pcontext, subgraph->coefs_rsrc,
                                        PIPE_MAP_READ, &transfer_in);
         if (!buf) {
            mesa_loge("ethosu: failed to map coefficients BO for dump");
            subgraph->failed = true;
            return false;
         }

         ethosu_dump_buffer(buf, "coefs", 0, 0, 0,
                            pipe_buffer_size(subgraph->coefs_rsrc));
         pipe_buffer_unmap(pcontext, transfer_in);
      }
   }

   subgraph->perfmon_id = 0;
   if (DBG_ENABLED(ETHOSU_DBG_DUMP_PERF)) {

      struct drm_ethosu_perfmon_create perfmon_create = {
         .counters = { 32, 35 }, /* npu-idle, npu-active */
         .ncounters = 2,
      };
      ret = drmIoctl(screen->fd, DRM_IOCTL_ETHOSU_PERFMON_CREATE, &perfmon_create);
      DBG("Perfmon create returned %d\n", ret);
      if (ret == 0) {
         subgraph->perfmon_id = perfmon_create.id;
      } else {
         DBG("Could not create perfmon: ret=%d errno=%d (%s)\n",
             ret, errno, strerror(errno));
      }
   }

   DBG("subgraph->io_used %d\n", subgraph->io_used);
   subgraph->io_rsrc = pipe_buffer_create(pcontext->screen, 0,
                                          PIPE_USAGE_DEFAULT,
                                          subgraph->io_used);
   if (!subgraph->io_rsrc) {
      mesa_loge("ethosu: failed to allocate IO BO (size=%u)",
                subgraph->io_used);
      subgraph->failed = true;
      return false;
   }

   return true;
}

struct pipe_ml_subgraph *
ethosu_ml_subgraph_deserialize(struct pipe_context *pcontext,
                               const uint8_t *data,
                               size_t size)
{
   struct ethosu_subgraph *subgraph;

   if (size < NUM_HEADER_FIELDS * sizeof(uint64_t))
      return NULL;

   subgraph = calloc(1, sizeof(*subgraph));
   if (!subgraph)
      return NULL;

   subgraph->base.device = pcontext->screen->get_ml_device(pcontext->screen);

   util_dynarray_init(&subgraph->tensors, NULL);

   const uint64_t *header = (const uint64_t *)data;
   uint64_t header_size = NUM_HEADER_FIELDS * sizeof(uint64_t);
   uint64_t cmdstream_size = header[0];
   uint64_t coefs_size = header[1];
   uint64_t io_size = header[2];
   uint64_t tensors_size = header[3];
   data += header_size;

   if (size != header_size + cmdstream_size + coefs_size + tensors_size) {
      free(subgraph);
      return NULL;
   }

   for (unsigned i = 0;
        i < tensors_size / (NUM_TENSOR_FIELDS * sizeof(uint32_t)); i++) {
      struct ethosu_tensor tensor = {0};
      const uint32_t *tdata = (const uint32_t *)data;
      tensor.index = tdata[0];
      tensor.offset = tdata[1];
      tensor.size = tdata[2];
      util_dynarray_append(&subgraph->tensors, tensor);
      data += NUM_TENSOR_FIELDS * sizeof(uint32_t);
   }

   subgraph->cmdstream_used = cmdstream_size / sizeof(*subgraph->cmdstream);
   subgraph->cmdstream = malloc(cmdstream_size);
   if (!subgraph->cmdstream) {
      util_dynarray_fini(&subgraph->tensors);
      free(subgraph);
      return NULL;
   }
   memcpy(subgraph->cmdstream, data, cmdstream_size);
   subgraph->cursor = subgraph->cmdstream + subgraph->cmdstream_used;
   data += cmdstream_size;

   subgraph->coefs_used = coefs_size;
   if (coefs_size > 0) {
      subgraph->coefs = malloc(coefs_size);
      if (!subgraph->coefs) {
         free(subgraph->cmdstream);
         util_dynarray_fini(&subgraph->tensors);
         free(subgraph);
         return NULL;
      }
      memcpy(subgraph->coefs, data, coefs_size);
   }

   subgraph->io_used = io_size;

   return &subgraph->base;
}

void
ethosu_ml_subgraph_invoke(struct pipe_context *pcontext,
                          struct pipe_ml_subgraph *psubgraph,
                          unsigned inputs_count, unsigned input_idxs[],
                          void *inputs[], bool is_signed[])
{
   struct ethosu_screen *screen = ethosu_screen(pcontext->screen);
   struct ethosu_subgraph *subgraph = (struct ethosu_subgraph *)(psubgraph);
   struct drm_ethosu_submit submit = {0};
   struct drm_ethosu_job job = {0};
   struct timespec start, end;
   int ret;

   if (subgraph->failed) {
      mesa_loge("ethosu: skipping invoke for failed subgraph");
      return;
   }

   if (subgraph->io_rsrc == NULL &&
       !prepare_for_submission(subgraph, pcontext))
      return;

   for (unsigned i = 0; i < inputs_count; i++) {
      struct ethosu_tensor *input = ethosu_find_tensor(subgraph, input_idxs[i]);
      assert(input);

      if (!inputs[i]) {
         mesa_loge("ethosu: input tensor %u has no data pointer",
                   input_idxs[i]);
         subgraph->failed = true;
         return;
      }

      if (DBG_ENABLED(ETHOSU_DBG_DUMP_BOS))
         ethosu_dump_buffer(inputs[i], "input", 0, 0, 0, input->size);

      pipe_buffer_write(pcontext, subgraph->io_rsrc, input->offset, input->size, inputs[i]);
   }

   if (DBG_ENABLED(ETHOSU_DBG_DUMP_BOS)) {
      struct pipe_transfer *transfer_in;
      uint8_t *buf = pipe_buffer_map(pcontext, subgraph->io_rsrc,
                                     PIPE_MAP_READ, &transfer_in);
      if (!buf) {
         mesa_loge("ethosu: failed to map IO BO before submit");
         subgraph->failed = true;
         return;
      }

      ethosu_dump_buffer(buf, "io-before", 0, 0, 0, pipe_buffer_size(subgraph->io_rsrc));
      pipe_buffer_unmap(pcontext, transfer_in);
   }

   if (!subgraph->cmdstream_bo)
      return;

   job.cmd_bo = subgraph->cmdstream_bo;

   if (subgraph->coefs_rsrc) {
      job.region_bo_handles[COEFS_REGION] = ethosu_resource(subgraph->coefs_rsrc)->handle;
      if (!DBG_ENABLED(ETHOSU_DBG_DISABLE_SRAM)) {
         job.region_bo_handles[SCRATCH_REGION] = 0;
         job.sram_size = ethosu_ml_device(subgraph->base.device)->sram_size;
      }
   }

   job.region_bo_handles[IO_REGION] = ethosu_resource(subgraph->io_rsrc)->handle;

   submit.jobs = (uintptr_t)&job;
   submit.job_count = 1;
   submit.perfmon_id = subgraph->perfmon_id;

   if (DBG_ENABLED(ETHOSU_DBG_MSGS))
      clock_gettime(CLOCK_MONOTONIC_RAW, &start);

   ret = drmIoctl(screen->fd, DRM_IOCTL_ETHOSU_SUBMIT, &submit);
   if (ret) {
      mesa_loge("ethosu: submit failed (errno=%d: %s)",
                errno, strerror(errno));
      subgraph->failed = true;
      return;
   }

   if (DBG_ENABLED(ETHOSU_DBG_MSGS)) {
      clock_gettime(CLOCK_MONOTONIC_RAW, &end);
      long long duration_ns = (long long)(end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
      DBG("Submission took %lld ms\n", duration_ns / 1000000);

      /* Force a sync */
      struct pipe_transfer *transfer_in;
      void *ptr = pipe_buffer_map(pcontext, subgraph->io_rsrc,
                                  PIPE_MAP_READ, &transfer_in);
      if (ptr)
         pipe_buffer_unmap(pcontext, transfer_in);
      else
         mesa_loge("ethosu: failed to map IO BO after submit");

      clock_gettime(CLOCK_MONOTONIC_RAW, &end);
      duration_ns = (long long)(end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
      DBG("Execution took %lld ms\n", duration_ns / 1000000);
   }
}

void
ethosu_ml_subgraph_read_outputs(struct pipe_context *pcontext,
                                struct pipe_ml_subgraph *psubgraph,
                                unsigned outputs_count,
                                unsigned output_idxs[], void *outputsv[],
                                bool is_signed[])
{
   struct ethosu_subgraph *subgraph = (struct ethosu_subgraph *)(psubgraph);
   uint8_t **outputs = (uint8_t **)outputsv;

   if (subgraph->failed || !subgraph->io_rsrc) {
      mesa_loge("ethosu: skipping readback for failed subgraph");
      return;
   }

   for (int i = 0; i < outputs_count; i++) {
      struct ethosu_tensor *output = ethosu_find_tensor(subgraph, output_idxs[i]);
      assert(output);

      if (!outputs[i]) {
         mesa_loge("ethosu: output tensor %u has no data pointer",
                   output_idxs[i]);
         subgraph->failed = true;
         return;
      }

      if (DBG_ENABLED(ETHOSU_DBG_DUMP_BOS)) {
         struct pipe_transfer *transfer_in;
         uint8_t *buf = pipe_buffer_map(pcontext, subgraph->io_rsrc,
                                        PIPE_MAP_READ, &transfer_in);
         if (!buf) {
            mesa_loge("ethosu: failed to map IO BO for readback");
            subgraph->failed = true;
            return;
         }

         ethosu_dump_buffer(buf, "io-after", 0, 0, 0, pipe_buffer_size(subgraph->io_rsrc));
         pipe_buffer_unmap(pcontext, transfer_in);
      }

      pipe_buffer_read(pcontext, subgraph->io_rsrc, output->offset, output->size, outputs[i]);
   }

   if (DBG_ENABLED(ETHOSU_DBG_DUMP_PERF)) {
      struct ethosu_screen *screen = ethosu_screen(pcontext->screen);
      uint64_t values[9];
      struct drm_ethosu_perfmon_get_values get_values = {
         .id = subgraph->perfmon_id,
         .values_ptr = (uintptr_t)values,
      };
      int ret;

      ret = drmIoctl(screen->fd, DRM_IOCTL_ETHOSU_PERFMON_GET_VALUES, &get_values);
      if (ret == 0) {
         mesa_logi("PMU: cycles=%" PRIu64 ", npu-active=%" PRIu64 ", npu-idle=%" PRIu64 "\n",
                   values[2], values[1], values[0]);
      } else {
         DBG("Could not read perfmon values: ret=%d errno=%d (%s)\n",
             ret, errno, strerror(errno));
      }
   }
}

void
ethosu_ml_subgraph_destroy(struct pipe_ml_device *pdevice,
                           struct pipe_ml_subgraph *psubgraph)
{
   struct ethosu_subgraph *subgraph = (struct ethosu_subgraph *)(psubgraph);
   struct ethosu_screen *screen = subgraph->screen;

   if (subgraph->io_rsrc || subgraph->coefs_rsrc || subgraph->cmdstream_bo) {
      /* Post-submission or partially prepared state: cleanup DRM resources. */
      struct drm_gem_close arg = {0};
      int ret;

      pipe_resource_reference(&subgraph->io_rsrc, NULL);
      pipe_resource_reference(&subgraph->coefs_rsrc, NULL);

      if (subgraph->cmdstream_bo && screen) {
         arg.handle = subgraph->cmdstream_bo;
         ret = drmIoctl(screen->fd, DRM_IOCTL_GEM_CLOSE, &arg);
         assert(ret >= 0);
      }
   }

   free(subgraph->cmdstream);
   free(subgraph->coefs);

   if (DBG_ENABLED(ETHOSU_DBG_DUMP_PERF)) {
      struct drm_ethosu_perfmon_destroy destroy = {
         .id = subgraph->perfmon_id,
      };
      if (screen)
         drmIoctl(screen->fd, DRM_IOCTL_ETHOSU_PERFMON_DESTROY, &destroy);
   }

   util_dynarray_fini(&subgraph->tensors);

   free(subgraph);
}
