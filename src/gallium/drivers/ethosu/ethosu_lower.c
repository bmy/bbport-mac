/*
 * Copyright (c) 2024 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include "util/u_inlines.h"

#include "ethosu_device.h"
#include "ethosu_lower.h"
#include "ethosu_coefs.h"
#include "ethosu_ml.h"
#include "ethosu_sched.h"

static bool
is_depthwise(const struct pipe_ml_operation *poperation)
{
   unsigned input_channels = poperation->input_tensors[0]->dims[3];
   unsigned output_channels = poperation->output_tensors[0]->dims[3];

   return poperation->conv.depthwise && input_channels > 1 &&
          output_channels > 1;
}

bool
ethosu_scatter_nd_as_pad(const struct pipe_ml_operation *scatter,
                         struct pipe_ml_operation *pad)
{
   const struct pipe_tensor *indices;
   const struct pipe_tensor *updates;
   const struct pipe_tensor *shape;
   const struct pipe_tensor *output;
   const int32_t *index_data;
   const int32_t *shape_data;
   unsigned updates_elements = 1;
   int start[4];

   if (scatter->input_count != 3 || scatter->output_count != 1)
      return false;

   indices = scatter->input_tensors[0];
   updates = scatter->input_tensors[1];
   shape = scatter->input_tensors[2];
   output = scatter->output_tensors[0];
   if (updates->rank != 4 || output->rank != 4 || indices->rank != 5 ||
       shape->rank != 1 || !indices->is_constant || !shape->is_constant ||
       !indices->data || !shape->data || indices->type_size != 4 ||
       shape->type_size != 4 || !indices->is_signed || !shape->is_signed ||
       indices->dims[0] != updates->dims[1] ||
       indices->dims[1] != updates->dims[2] ||
       indices->dims[2] != updates->dims[3] || indices->dims[3] != 4 ||
       updates->dims[0] != 1 || output->dims[0] != 1 ||
       updates->type_size != output->type_size ||
       updates->is_signed != output->is_signed || updates->scale != output->scale ||
       updates->zero_point != output->zero_point)
      return false;

   index_data = (const int32_t *)indices->data;
   shape_data = (const int32_t *)shape->data;
   for (unsigned axis = 0; axis < 4; axis++) {
      if (shape_data[axis] != output->dims[axis] || !updates->dims[axis])
         return false;

      updates_elements *= updates->dims[axis];
      start[axis] = index_data[axis];
      if (start[axis] < 0 ||
          start[axis] + updates->dims[axis] > output->dims[axis])
         return false;
   }

   for (unsigned i = 0; i < updates_elements; i++) {
      unsigned coordinate = i;

      for (int axis = 3; axis >= 0; axis--) {
         int expected = start[axis] + coordinate % updates->dims[axis];

         if (index_data[i * 4 + axis] != expected)
            return false;
         coordinate /= updates->dims[axis];
      }
   }

   if (start[0])
      return false;

   if (pad) {
      *pad = *scatter;
      pad->type = PIPE_ML_OPERATION_TYPE_PAD;
      pad->input_count = 1;
      pad->pad.before_y = start[1];
      pad->pad.after_y = output->dims[1] - start[1] - updates->dims[1];
      pad->pad.before_x = start[2];
      pad->pad.after_x = output->dims[2] - start[2] - updates->dims[2];
      pad->pad.before_z = start[3];
      pad->pad.after_z = output->dims[3] - start[3] - updates->dims[3];
      pad->pad.raw_zero = true;
   }

   return true;
}

static const struct pipe_ml_operation *
ethosu_find_first_producer(const struct pipe_ml_operation *poperations,
                           unsigned count,
                           unsigned tensor_index);

static unsigned
needed_total_padding(unsigned input_size, unsigned output_size,
                     unsigned stride, unsigned filter_size)
{
   int total = (output_size - 1) * stride + filter_size - input_size;

   return MAX2(total, 0);
}

static void
trim_padding_to_output(struct ethosu_operation *operation,
                       const struct pipe_tensor *input,
                       const struct pipe_tensor *output)
{
   unsigned total_y =
      needed_total_padding(input->dims[1], output->dims[1],
                           operation->kernel.stride_y,
                           operation->kernel.height);
   unsigned total_x =
      needed_total_padding(input->dims[2], output->dims[2],
                           operation->kernel.stride_x,
                           operation->kernel.width);

   operation->pad.top = MIN2(operation->pad.top, total_y);
   operation->pad.bottom = total_y - operation->pad.top;
   operation->pad.left = MIN2(operation->pad.left, total_x);
   operation->pad.right = total_x - operation->pad.left;
}

static void
set_feature_map_strides(struct ethosu_feature_map *fm, bool is_nhcwb16)
{
   unsigned elem_size = 1 << fm->precision;

   if (is_nhcwb16) {
      fm->stride.x = 16 * elem_size;
      fm->stride.c = fm->stride.x * fm->shape.width;
      fm->stride.y = elem_size * fm->shape.width * align(fm->shape.depth, 16);
   } else {
      fm->stride.c = elem_size;
      fm->stride.x = fm->shape.depth * fm->stride.c;
      fm->stride.y = fm->shape.width * fm->stride.x;
   }
}

static void
set_feature_map(struct ethosu_subgraph *subgraph,
                struct pipe_tensor *tensor,
                struct ethosu_feature_map *fm)
{
   fm->tensor = ethosu_find_tensor(subgraph, tensor->index);
   fm->region = IO_REGION;
   fm->shape.height = tensor->dims[1];
   fm->shape.width = tensor->dims[2];
   fm->shape.depth = tensor->dims[3];
   fm->zero_point = tensor->zero_point;
   fm->scale = tensor->scale;
   fm->is_signed = tensor->is_signed;
   fm->precision = log2(tensor->type_size);

   set_feature_map_strides(fm, fm->tensor->layout == ETHOSU_LAYOUT_NHCWB16);
}

static void
set_feature_maps(struct ethosu_subgraph *subgraph,
                 struct pipe_tensor *input_tensor,
                 struct pipe_tensor *output_tensor,
                 struct ethosu_operation *operation)
{
   set_feature_map(subgraph, input_tensor, &operation->ifm);
   set_feature_map(subgraph, output_tensor, &operation->ofm);
}

static const struct pipe_ml_operation *
ethosu_find_first_consumer(const struct pipe_ml_operation *poperations,
                           unsigned count,
                           unsigned tensor_index)
{
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];
      for (unsigned j = 0; j < poperation->input_count; j++)
         if (poperation->input_tensors[j]->index == tensor_index)
            return poperation;
   }

   return NULL;
}

static const struct pipe_ml_operation *
ethosu_find_only_consumer(const struct pipe_ml_operation *poperations,
                          unsigned count,
                          unsigned tensor_index)
{
   const struct pipe_ml_operation *consumer = NULL;

   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];
      for (unsigned j = 0; j < poperation->input_count; j++) {
         if (poperation->input_tensors[j]->index != tensor_index)
            continue;

         if (consumer)
            return NULL;

         consumer = poperation;
      }
   }

   return consumer;
}
static bool
ethosu_fc_needs_flatten(const struct pipe_ml_operation *poperation)
{
   const struct pipe_tensor *input;
   const struct pipe_tensor *output;
   const struct pipe_tensor *weight;

   if (poperation->type != PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED)
      return false;

   input = poperation->input_tensors[0];
   output = poperation->output_tensors[0];
   weight = poperation->fcon.weight_tensor;

   return weight &&
          (weight->dims[3] != input->dims[3] ||
           output->dims[1] != 1 ||
           output->dims[2] != input->dims[1] * input->dims[2]);
}

static unsigned
ethosu_fc_batch_width(unsigned rows)
{
   unsigned width = 1;

   while (width * width < rows)
      width++;

   width = MAX2(rows / 16, width);
   while (rows % width)
      width++;

   return width;
}

static unsigned
ethosu_fc_batch_rows(const struct pipe_ml_operation *poperation)
{
   const struct pipe_tensor *input = poperation->input_tensors[0];
   const struct pipe_tensor *weight = poperation->fcon.weight_tensor;

   if (ethosu_fc_needs_flatten(poperation))
      return input->dims[1] * input->dims[2] * input->dims[3] /
             weight->dims[3];

   return input->dims[1] * input->dims[2];
}

static bool
ethosu_fc_uses_batched_shape(const struct ethosu_subgraph *subgraph,
                             const struct pipe_ml_operation *poperation)
{
   return poperation->type == PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED &&
          ethosu_fc_batch_rows(poperation) > 1;
}

/* The input and output feature-map shapes ethosu_lower_fully_connected
 * gives a fully connected: a flatten collapses the rows into a single
 * column of the input or output channels, a conv-like case keeps the
 * spatial size, and a batched fully connected then folds the rows into a
 * near-square plane, overriding the width either set.  The layout
 * decision mirrors the lowering by reading the shapes from here. */
static void
ethosu_fc_lowered_dims(const struct ethosu_subgraph *subgraph,
                       const struct pipe_ml_operation *poperation,
                       unsigned in[4], unsigned out[4])
{
   const struct pipe_tensor *input = poperation->input_tensors[0];
   const struct pipe_tensor *output = poperation->output_tensors[0];
   const struct pipe_tensor *weight = poperation->fcon.weight_tensor;

   for (unsigned i = 0; i < 4; i++) {
      in[i] = input->dims[i];
      out[i] = output->dims[i];
   }

   if (ethosu_fc_needs_flatten(poperation)) {
      unsigned rows = input->dims[1] * input->dims[2] * input->dims[3] /
                      weight->dims[3];

      in[1] = rows;
      in[2] = 1;
      in[3] = weight->dims[3];
      out[1] = rows;
      out[2] = 1;
      out[3] = weight->dims[2];
   } else if (weight->dims[1] == 1 && weight->dims[3] == input->dims[3] &&
              output->dims[1] == 1 &&
              output->dims[2] == input->dims[1] * input->dims[2]) {
      out[1] = input->dims[1];
      out[2] = input->dims[2];
      out[3] = weight->dims[2];
   }

   if (ethosu_fc_uses_batched_shape(subgraph, poperation)) {
      unsigned rows = ethosu_fc_batch_rows(poperation);
      unsigned width = ethosu_fc_batch_width(rows);

      in[1] = rows / width;
      in[2] = width;
      out[1] = in[1];
      out[2] = width;
   }
}

/* The width and depth an operation writes or reads a tensor with, which
 * define its NHCWB16 brick order.  A fully connected writes and reads the
 * shapes ethosu_fc_lowered_dims gives it; a reshape reinterprets the
 * buffer with its output shape; every other operation uses the tensor's
 * own dimensions. */
static void
ethosu_written_wd(const struct ethosu_subgraph *subgraph,
                  const struct pipe_ml_operation *producer,
                  const struct pipe_tensor *ptensor,
                  unsigned *width, unsigned *depth)
{
   if (producer->type == PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED) {
      unsigned in[4], out[4];

      ethosu_fc_lowered_dims(subgraph, producer, in, out);
      *width = out[2];
      *depth = out[3];
      return;
   }

   *width = ptensor->dims[2];
   *depth = ptensor->dims[3];
}

static void
ethosu_read_wd(const struct ethosu_subgraph *subgraph,
               const struct pipe_ml_operation *consumer,
               const struct pipe_tensor *ptensor,
               unsigned *width, unsigned *depth)
{
   if (consumer->type == PIPE_ML_OPERATION_TYPE_RESHAPE) {
      const struct pipe_tensor *out = consumer->output_tensors[0];

      *width = out->dims[2];
      *depth = out->dims[3];
      return;
   }

   if (consumer->type == PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED &&
       consumer->input_tensors[0]->index == ptensor->index) {
      unsigned in[4], out[4];

      ethosu_fc_lowered_dims(subgraph, consumer, in, out);
      *width = in[2];
      *depth = in[3];
      return;
   }

   *width = ptensor->dims[2];
   *depth = ptensor->dims[3];
}

/* The offset along `axis` at which concat input, or split output, `index`
 * begins: the running sum of the preceding slices' extents.  The layout rule
 * and the concat and split lowerings share it. */
static unsigned
ethosu_concat_split_offset(struct pipe_tensor *const *slices,
                           unsigned index, unsigned axis)
{
   unsigned offset = 0;

   for (unsigned i = 0; i < index; i++)
      offset += slices[i]->dims[axis];

   return offset;
}

/* A channel-axis concatenation writes each input at the running sum of the
 * preceding depths.  NHCWB16 packs 16 channels per brick, so a start that is
 * not a multiple of 16 falls mid-brick and cannot be a brick base; such an
 * output must stay linear (NHWC). */
static bool
ethosu_concat_offset_needs_nhwc(const struct pipe_ml_operation *producer)
{
   if (producer->type != PIPE_ML_OPERATION_TYPE_CONCATENATION ||
       producer->conc.axis != 3)
      return false;

   for (unsigned i = 1; i < producer->input_count; i++)
      if (ethosu_concat_split_offset(producer->input_tensors, i, 3) & 15)
         return true;

   return false;
}

static bool
ethosu_consumer_uses_depth_mean(const struct pipe_ml_operation *poperation)
{
   return poperation->type == PIPE_ML_OPERATION_TYPE_MEAN &&
          poperation->mean.axes == BITFIELD_BIT(3);
}

static bool
ethosu_all_consumers_are_convolutions(const struct pipe_ml_operation *poperations,
                                      unsigned count,
                                      unsigned tensor_index)
{
   bool found_consumer = false;

   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->input_count; j++) {
         if (poperation->input_tensors[j]->index != tensor_index)
            continue;

         found_consumer = true;
         if (poperation->type != PIPE_ML_OPERATION_TYPE_CONVOLUTION)
            return false;
      }
   }

   return found_consumer;
}

static bool
ethosu_consumer_is_lut(const struct pipe_ml_operation *poperation)
{
   if (!poperation)
      return false;

   switch (poperation->type) {
   case PIPE_ML_OPERATION_TYPE_LOGISTIC:
   case PIPE_ML_OPERATION_TYPE_TANH:
   case PIPE_ML_OPERATION_TYPE_RSQRT:
   case PIPE_ML_OPERATION_TYPE_HSWISH:
   case PIPE_ML_OPERATION_TYPE_LEAKY_RELU:
      return true;
   default:
      return false;
   }
}

static const struct pipe_ml_operation *
ethosu_find_fusible_lut(const struct pipe_ml_operation *poperations,
                        unsigned count,
                        unsigned tensor_index)
{
   const struct pipe_ml_operation *consumer =
      ethosu_find_only_consumer(poperations, count, tensor_index);
   bool has_external_output;

   if (!consumer)
      return NULL;

   has_external_output = consumer->input_tensors[0]->is_external_output;

   if (consumer->type == PIPE_ML_OPERATION_TYPE_RESHAPE) {
      has_external_output |= consumer->output_tensors[0]->is_external_output;
      consumer = ethosu_find_only_consumer(poperations, count,
                                           consumer->output_tensors[0]->index);
   }

   if (!has_external_output && ethosu_consumer_is_lut(consumer) &&
       ethosu_find_first_consumer(poperations, count,
                                  consumer->output_tensors[0]->index))
      return consumer;

   return NULL;
}

static bool
ethosu_lut_is_fused(const struct pipe_ml_operation *poperations,
                    unsigned count,
                    const struct pipe_ml_operation *lut)
{
   const struct pipe_ml_operation *producer =
      ethosu_find_first_producer(poperations, count,
                                 lut->input_tensors[0]->index);

   if (producer && producer->type == PIPE_ML_OPERATION_TYPE_RESHAPE)
      producer = ethosu_find_first_producer(poperations, count,
                                            producer->input_tensors[0]->index);

   if (!producer ||
       (producer->type != PIPE_ML_OPERATION_TYPE_CONVOLUTION &&
        producer->type != PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED &&
        producer->type != PIPE_ML_OPERATION_TYPE_ADD &&
        producer->type != PIPE_ML_OPERATION_TYPE_MUL &&
        producer->type != PIPE_ML_OPERATION_TYPE_SUBTRACT &&
        producer->type != PIPE_ML_OPERATION_TYPE_MAXIMUM &&
        producer->type != PIPE_ML_OPERATION_TYPE_MINIMUM))
      return false;

   return ethosu_find_fusible_lut(
             poperations, count, producer->output_tensors[0]->index) == lut;
}

unsigned
ethosu_feature_map_span(const struct ethosu_feature_map *fm)
{
   unsigned elem_size = 1 << fm->precision;
   /* Brick format stores 16 channels contiguously, so its channel axis
    * steps in bricks of 16 and its last element spans a full brick.
    */
   bool brick = fm->tensor && fm->tensor->layout == ETHOSU_LAYOUT_NHCWB16;
   unsigned steps = brick ? align(fm->shape.depth, 16) / 16
                          : fm->shape.depth;
   uint64_t size = brick ? 16 * elem_size : elem_size;

   if (fm->shape.height)
      size += (uint64_t)(fm->shape.height - 1) * fm->stride.y;
   if (fm->shape.width)
      size += (uint64_t)(fm->shape.width - 1) * fm->stride.x;
   if (steps)
      size += (uint64_t)(steps - 1) * fm->stride.c;

   assert(size <= UINT_MAX);
   return size;
}

static unsigned
ethosu_allocate_feature_map(struct ethosu_subgraph *subgraph,
                            struct ethosu_feature_map *fm)
{
   unsigned span = ethosu_feature_map_span(fm);
   struct ethosu_tensor *tensor = fm->tensor;
   uint64_t size;

   assert(tensor);

   span = MAX2(span, tensor->required_size);
   size = (uint64_t)span * tensor->batches;
   assert(size <= UINT_MAX);

   if (tensor->size > 0) {
      assert(size <= tensor->size);
      return tensor->offset;
   }

   tensor->offset = subgraph->io_used;
   tensor->size = size;
   subgraph->io_used += ALIGN_POT(size, 16);

   return tensor->offset;
}

static void
set_internal_feature_map(struct ethosu_tensor *tensor,
                         struct ethosu_block shape,
                         float scale,
                         unsigned zero_point,
                         bool is_signed,
                         struct ethosu_feature_map *fm);

static void
ethosu_require_feature_map_span(struct ethosu_tensor *tensor,
                                struct ethosu_block shape,
                                float scale,
                                unsigned zero_point,
                                bool is_signed)
{
   struct ethosu_feature_map fm = {0};

   set_internal_feature_map(tensor, shape, scale, zero_point, is_signed, &fm);
   tensor->required_size = MAX2(tensor->required_size,
                                ethosu_feature_map_span(&fm));
}

static void
ethosu_require_softmax_spans(struct ethosu_subgraph *subgraph,
                             const struct pipe_ml_operation *poperations,
                             unsigned count)
{
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      if (poperation->type != PIPE_ML_OPERATION_TYPE_SOFTMAX)
         continue;

      struct pipe_tensor *input = poperation->input_tensors[0];
      struct pipe_tensor *output = poperation->output_tensors[0];
      unsigned spatial = input->dims[1] * input->dims[2];
      unsigned depth = input->dims[3];
      struct ethosu_block pool_shape = {depth, spatial, 1};
      struct ethosu_block vector_shape = {spatial, 1, depth};
      struct ethosu_tensor *input_tensor =
         ethosu_find_tensor(subgraph, input->index);
      struct ethosu_tensor *output_tensor =
         ethosu_find_tensor(subgraph, output->index);

      ethosu_require_feature_map_span(input_tensor, pool_shape, input->scale,
                                      input->zero_point, input->is_signed);
      ethosu_require_feature_map_span(input_tensor, vector_shape, input->scale,
                                      input->zero_point, input->is_signed);
      ethosu_require_feature_map_span(output_tensor, vector_shape, output->scale,
                                      output->zero_point, output->is_signed);
   }
}

static void
allocate_feature_maps(struct ethosu_subgraph *subgraph, struct ethosu_operation *operation)
{
   if (operation->ofm.region == IO_REGION &&
       operation->ofm.activation_storage != ETHOSU_ACTIVATION_STORAGE_CHAINED) {
      operation->ofm.tiles.addresses[0] = ethosu_allocate_feature_map(subgraph, &operation->ofm);
      operation->ofm.tiles.height_0 = operation->ofm.shape.height;
      operation->ofm.tiles.height_1 = operation->ofm.shape.height;
      operation->ofm.tiles.width_0 = operation->ofm.shape.width;
   }

   if (operation->ifm.region == IO_REGION &&
       operation->ifm.activation_storage != ETHOSU_ACTIVATION_STORAGE_CHAINED) {
      operation->ifm.tiles.addresses[0] = ethosu_allocate_feature_map(subgraph, &operation->ifm);
      operation->ifm.tiles.height_0 = operation->ifm.shape.height;
      operation->ifm.tiles.height_1 = operation->ifm.shape.height;
      operation->ifm.tiles.width_0 = operation->ifm.shape.width;
   }
}

static void
ethosu_reserve_internal_tensors(struct ethosu_subgraph *subgraph,
                                unsigned count)
{
   /* Each lowered operation may add internal tensors of its own (constant
    * fills, intermediate feature maps); reserve a conservative upper bound
    * per operation so the tensor array is not reallocated mid-lowering,
    * which would invalidate held tensor pointers. */
   unsigned internal_tensors = count * 32;

   if (!internal_tensors)
      return;

   if (!util_dynarray_ensure_cap(&subgraph->tensors,
                                 subgraph->tensors.size +
                                    internal_tensors * sizeof(struct ethosu_tensor))) {
      mesa_loge("ethosu: failed to reserve internal tensors");
      subgraph->failed = true;
   }
}

static struct ethosu_tensor *
ethosu_add_internal_tensor(struct ethosu_subgraph *subgraph,
                           struct ethosu_block shape,
                           uint8_t type_size)
{
   struct ethosu_tensor tensor = {0};

   tensor.index = BITFIELD_BIT(31) |
                  util_dynarray_num_elements(&subgraph->tensors, struct ethosu_tensor);
   tensor.shape = shape;
   tensor.layout = ETHOSU_LAYOUT_NHWC;
   tensor.batches = subgraph->batches;
   tensor.type_size = type_size;

   util_dynarray_append(&subgraph->tensors, tensor);

   return util_dynarray_element(&subgraph->tensors, struct ethosu_tensor,
                                util_dynarray_num_elements(&subgraph->tensors,
                                                           struct ethosu_tensor) -
                                   1);
}

static unsigned
ethosu_add_constant(struct ethosu_subgraph *subgraph,
                    const void *data,
                    unsigned size)
{
   unsigned address = ethosu_allocate_coefs(subgraph, size);

   memcpy(subgraph->coefs + address, data, size);

   return address;
}

static void
set_internal_feature_map(struct ethosu_tensor *tensor,
                         struct ethosu_block shape,
                         float scale,
                         unsigned zero_point,
                         bool is_signed,
                         struct ethosu_feature_map *fm)
{
   fm->tensor = tensor;
   fm->region = IO_REGION;
   fm->shape = shape;
   fm->zero_point = zero_point;
   fm->scale = scale;
   fm->is_signed = is_signed;
   fm->precision = log2(tensor->type_size);

   set_feature_map_strides(fm, tensor->layout == ETHOSU_LAYOUT_NHCWB16);
}

static void
set_constant_feature_map(struct ethosu_subgraph *subgraph,
                         int32_t value,
                         struct ethosu_feature_map *fm)
{
   struct ethosu_block scalar_shape = {1, 1, 1};
   static struct ethosu_tensor tensor = {
      .size = 4,
      .type_size = 4,
      .layout = ETHOSU_LAYOUT_NHWC,
   };

   fm->tensor = &tensor;
   fm->region = COEFS_REGION;
   fm->shape = scalar_shape;
   fm->zero_point = 0;
   fm->scale = 1.0f;
   fm->is_signed = true;
   fm->precision = 2;
   if (ethosu_ml_device(subgraph->base.device)->is_u65) {
      fm->tiles.addresses[0] =
         ethosu_add_constant(subgraph, &value, sizeof(value));
   } else {
      fm->scalar = value;
      fm->has_scalar = true;
   }
   fm->tiles.height_0 = scalar_shape.height;
   fm->tiles.height_1 = scalar_shape.height;
   fm->tiles.width_0 = scalar_shape.width;

   set_feature_map_strides(fm, false);
}

static void
set_u85_chained_feature_map(struct ethosu_subgraph *subgraph,
                            struct ethosu_feature_map *fm,
                            unsigned chain_id)
{
   if (ethosu_ml_device(subgraph->base.device)->is_u65)
      return;

   fm->activation_storage = ETHOSU_ACTIVATION_STORAGE_CHAINED;
   fm->chain_id = chain_id;
}

static void
ethosu_append_operation(struct ethosu_subgraph *subgraph,
                        struct ethosu_operation *operation)
{
   ethosu_sched_operation(subgraph, operation);
   util_dynarray_append(&subgraph->operations, *operation);
}

static bool
ethosu_needs_strided_conv_unroll(const struct ethosu_operation *operation)
{
   return operation->type == ETHOSU_OPERATION_TYPE_CONVOLUTION &&
          (operation->kernel.stride_x > 3 || operation->kernel.stride_y > 3);
}

static void
ethosu_append_strided_conv_unroll(struct ethosu_subgraph *subgraph,
                                  struct ethosu_operation *operation)
{
   const struct ethosu_block ifm_shape = operation->ifm.shape;
   const struct ethosu_block ofm_shape = operation->ofm.shape;

   assert(operation->pad.top == 0 && operation->pad.bottom == 0);
   assert(operation->pad.left == 0 && operation->pad.right == 0);
   assert(operation->kernel.dilation_x == 1 &&
          operation->kernel.dilation_y == 1);

   for (unsigned y = 0; y < ofm_shape.height; y++) {
      for (unsigned x = 0; x < ofm_shape.width; x++) {
         struct ethosu_operation unrolled = *operation;
         unsigned ifm_y = y * operation->kernel.stride_y;
         unsigned ifm_x = x * operation->kernel.stride_x;

         assert(ifm_y + operation->kernel.height <= ifm_shape.height);
         assert(ifm_x + operation->kernel.width <= ifm_shape.width);

         unrolled.ifm.shape.height = operation->kernel.height;
         unrolled.ifm.shape.width = operation->kernel.width;
         unrolled.ifm.tiles.addresses[0] += ifm_y * unrolled.ifm.stride.y +
                                            ifm_x * unrolled.ifm.stride.x;
         unrolled.ifm.tiles.height_0 = unrolled.ifm.shape.height;
         unrolled.ifm.tiles.height_1 = unrolled.ifm.shape.height;
         unrolled.ifm.tiles.width_0 = unrolled.ifm.shape.width;

         unrolled.ofm.shape.height = 1;
         unrolled.ofm.shape.width = 1;
         unrolled.ofm.tiles.addresses[0] += y * unrolled.ofm.stride.y +
                                            x * unrolled.ofm.stride.x;
         unrolled.ofm.tiles.height_0 = 1;
         unrolled.ofm.tiles.height_1 = 1;
         unrolled.ofm.tiles.width_0 = 1;

         unrolled.kernel.stride_x = 1;
         unrolled.kernel.stride_y = 1;

         /* The first command owns the per-channel quantization arrays. */
         if (x || y) {
            unrolled.kernel.scales = NULL;
            unrolled.kernel.zero_points = NULL;
         }

         util_dynarray_append(&subgraph->operations, unrolled);
      }
   }
}

static void
ethosu_prepare_feature_map(struct ethosu_subgraph *subgraph,
                           struct ethosu_feature_map *fm)
{
   if (fm->region != IO_REGION || !fm->tensor)
      return;

   if (fm->activation_storage == ETHOSU_ACTIVATION_STORAGE_CHAINED)
      return;

   fm->tiles.addresses[0] += ethosu_allocate_feature_map(subgraph, fm);
   fm->tiles.height_0 = fm->shape.height;
   fm->tiles.height_1 = fm->shape.height;
   fm->tiles.width_0 = fm->shape.width;
}

static int32_t
ethosu_read_scalar(const struct pipe_tensor *tensor)
{
   assert(tensor->data);

   if (tensor->type_size == 1)
      return tensor->is_signed ? *(int8_t *)tensor->data :
                                 *(uint8_t *)tensor->data;

   if (tensor->type_size == 2)
      return tensor->is_signed ? *(int16_t *)tensor->data :
                                 *(uint16_t *)tensor->data;

   assert(tensor->type_size == 4);
   return *(int32_t *)tensor->data;
}

static void
ethosu_append_eltwise(struct ethosu_subgraph *subgraph,
                      enum ethosu_eltwise_type type,
                      struct ethosu_feature_map *ifm,
                      struct ethosu_feature_map *ifm2,
                      struct ethosu_feature_map *ofm,
                      enum ethosu_rounding_mode round_mode,
                      bool identity_scale,
                      unsigned scale,
                      unsigned shift);

static void
ethosu_append_pool(struct ethosu_subgraph *subgraph,
                   enum ethosu_pooling_type type,
                   struct ethosu_feature_map *ifm,
                   struct ethosu_feature_map *ofm,
                   unsigned kernel_width,
                   enum ethosu_rounding_mode round_mode);

static void
ethosu_lower_lut_dma(struct ethosu_subgraph *subgraph,
                     const struct pipe_ml_operation *poperation,
                     struct ethosu_operation *pool_operation,
                     struct ethosu_operation *operation);

static void
set_constant_tensor_data_feature_map(struct ethosu_subgraph *subgraph,
                                     struct pipe_tensor *tensor,
                                     const void *data,
                                     struct ethosu_feature_map *fm);

static void
set_constant_tensor_feature_map_value(struct ethosu_subgraph *subgraph,
                                      struct pipe_tensor *tensor,
                                      int value,
                                      struct ethosu_feature_map *fm);

static void
set_constant_tensor_feature_map(struct ethosu_subgraph *subgraph,
                                struct pipe_tensor *tensor,
                                struct ethosu_feature_map *fm)
{
   set_constant_tensor_feature_map_value(subgraph, tensor,
                                         tensor->zero_point, fm);
}

static unsigned
ethosu_lut_activation(struct ethosu_subgraph *subgraph,
                      struct ethosu_feature_map *ifm,
                      struct ethosu_feature_map *ofm,
                      unsigned size,
                      bool force_int8_clip);

static void
operation_set_defaults(struct ethosu_operation *operation)
{
   memset(operation, 0, sizeof(*operation));

   operation->kernel.height = 1;
   operation->kernel.width = 1;
   operation->kernel.stride_y = 1;
   operation->kernel.stride_x = 1;
   operation->kernel.dilation_y = 1;
   operation->kernel.dilation_x = 1;
}

static void
set_full_activation_range(struct ethosu_operation *operation)
{
   if (operation->ofm.is_signed) {
      if (operation->ofm.precision == 0) {
         operation->conv.activation_min = INT8_MIN;
         operation->conv.activation_max = INT8_MAX;
      } else {
         operation->conv.activation_min = INT16_MIN;
         operation->conv.activation_max = INT16_MAX;
      }
   } else {
      operation->conv.activation_min = 0;
      if (operation->ofm.precision == 0)
         operation->conv.activation_max = UINT8_MAX;
      else
         operation->conv.activation_max = UINT16_MAX;
   }
}

static const struct pipe_ml_operation *
ethosu_find_first_producer(const struct pipe_ml_operation *poperations, unsigned count,
                           unsigned tensor_index)
{
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->output_count; j++) {
         if (poperation->output_tensors[j]->index == tensor_index)
            return poperation;
      }
   }

   return NULL;
}

static bool
ethosu_space_batch_fusion(const struct pipe_ml_operation *poperations,
                          unsigned count,
                          const struct pipe_ml_operation *conv,
                          const struct pipe_ml_operation **space_to_batch,
                          const struct pipe_ml_operation **batch_to_space)
{
   const struct pipe_ml_operation *producer =
      ethosu_find_first_producer(poperations, count,
                                 conv->input_tensors[0]->index);
   const struct pipe_ml_operation *consumer =
      ethosu_find_only_consumer(poperations, count,
                                conv->output_tensors[0]->index);

   if (!producer || !consumer ||
       ethosu_find_only_consumer(poperations, count,
                                 conv->input_tensors[0]->index) != conv ||
       producer->type != PIPE_ML_OPERATION_TYPE_SPACE_TO_BATCH ||
       consumer->type != PIPE_ML_OPERATION_TYPE_BATCH_TO_SPACE ||
       producer->space_batch.block_y != consumer->space_batch.block_y ||
       producer->space_batch.block_x != consumer->space_batch.block_x)
      return false;

   *space_to_batch = producer;
   *batch_to_space = consumer;
   return true;
}

static bool
ethosu_space_batch_is_fused(const struct pipe_ml_operation *poperations,
                            unsigned count,
                            const struct pipe_ml_operation *op)
{
   bool is_space_to_batch =
      op->type == PIPE_ML_OPERATION_TYPE_SPACE_TO_BATCH;
   const struct pipe_ml_operation *conv =
      is_space_to_batch
         ? ethosu_find_only_consumer(poperations, count,
                                     op->output_tensors[0]->index)
         : ethosu_find_first_producer(poperations, count,
                                      op->input_tensors[0]->index);
   const struct pipe_ml_operation *space_to_batch;
   const struct pipe_ml_operation *batch_to_space;

   if (!conv || conv->type != PIPE_ML_OPERATION_TYPE_CONVOLUTION ||
       !ethosu_space_batch_fusion(poperations, count, conv,
                                  &space_to_batch, &batch_to_space))
      return false;

   return op == (is_space_to_batch ? space_to_batch : batch_to_space);
}

static void
set_sparse_weight_format(struct ethosu_subgraph *subgraph,
                         struct ethosu_operation *operation,
                         const struct pipe_tensor *weight)
{
   unsigned depth = weight->dims[3];
   unsigned padded_depth = align(depth, 4);

   if (ethosu_ml_device(subgraph->base.device)->is_u65 ||
       operation->conv.depthwise)
      return;

   for (unsigned o = 0; o < weight->dims[0]; o++) {
      for (unsigned y = 0; y < weight->dims[1]; y++) {
         for (unsigned x = 0; x < weight->dims[2]; x++) {
            for (unsigned c = 0; c < padded_depth; c += 4) {
               unsigned zeros = 0;

               for (unsigned i = c; i < c + 4; i++) {
                  int value = 0;

                  if (i < depth) {
                     unsigned index = ((o * weight->dims[1] + y) *
                                       weight->dims[2] + x) * depth + i;
                     value = weight->is_signed ?
                        (int)(int8_t)weight->data[index] : weight->data[index];
                     value -= operation->kernel.zero_point;
                  }

                  if (value == 0)
                     zeros++;
                  else if (value < -127 || value > 127)
                     return;
               }

               if (zeros < 2)
                  return;
            }
         }
      }
   }

   operation->conv.weight_sparse = true;
}

static void
lower_conv_common(struct ethosu_subgraph *subgraph,
                  const struct pipe_ml_operation *poperation,
                  struct pipe_tensor *input_tensor,
                  struct pipe_tensor *weight,
                  int32_t *bias_data,
                  struct ethosu_operation *operation)
{
   operation->type = ETHOSU_OPERATION_TYPE_CONVOLUTION;

   set_feature_maps(subgraph, input_tensor, poperation->output_tensors[0], operation);

   /* Per-channel quantization support */
   unsigned num_channels = poperation->output_tensors[0]->dims[3];

   if (weight->scales != NULL) {
      operation->kernel.scales = malloc(num_channels * sizeof(float));
      memcpy(operation->kernel.scales, weight->scales, num_channels * sizeof(float));
   }

   if (weight->zero_points != NULL) {
      operation->kernel.zero_points = malloc(num_channels * sizeof(int));
      memcpy(operation->kernel.zero_points, weight->zero_points, num_channels * sizeof(int));
   }

   allocate_feature_maps(subgraph, operation);
   set_sparse_weight_format(subgraph, operation, weight);
   ethosu_sched_operation(subgraph, operation);
   fill_coefs(subgraph, operation, bias_data, weight->data,
              weight->dims[0] * weight->dims[1] * weight->dims[2] * weight->dims[3]);
}

static void
ethosu_lower_fully_connected(struct ethosu_subgraph *subgraph,
                             const struct pipe_ml_operation *poperation,
                             struct pipe_tensor *input_tensor,
                             struct ethosu_operation *operation)
{
   struct pipe_tensor flat_input = *input_tensor;
   struct pipe_tensor spatial_output = *poperation->output_tensors[0];
   struct pipe_ml_operation conv_operation = *poperation;
   struct pipe_tensor *output_tensors[1] = {&spatial_output};
   struct pipe_tensor *weight = poperation->fcon.weight_tensor;
   unsigned in[4], out[4];

   ethosu_fc_lowered_dims(subgraph, poperation, in, out);
   for (unsigned i = 0; i < 4; i++) {
      flat_input.dims[i] = in[i];
      spatial_output.dims[i] = out[i];
   }

   conv_operation.output_tensors = output_tensors;

   operation->kernel.scale = poperation->fcon.weight_tensor->scale;
   operation->kernel.zero_point = poperation->fcon.weight_tensor->zero_point;
   operation->kernel.is_signed = poperation->fcon.weight_tensor->is_signed;
   operation->kernel.scale_as_float =
      poperation->fcon.weight_tensor->scales == NULL;

   lower_conv_common(subgraph, &conv_operation, &flat_input,
                     weight,
                     (int32_t *)poperation->fcon.bias_tensor->data,
                     operation);
   set_full_activation_range(operation);
   if (poperation->fcon.relu)
      operation->conv.activation_min = operation->ofm.zero_point;
}

static uint8_t *
dilate_conv_weights(struct pipe_tensor *weight, unsigned dilation_y,
                    unsigned dilation_x)
{
   const unsigned height = weight->dims[1];
   const unsigned width = weight->dims[2];
   const unsigned dilated_height = (height - 1) * dilation_y + 1;
   const unsigned dilated_width = (width - 1) * dilation_x + 1;
   const unsigned outer = weight->dims[0];
   const unsigned channels = weight->dims[3];
   const size_t size = (size_t)outer * dilated_height * dilated_width * channels;
   uint8_t *dilated = malloc(size);

   assert(weight->type_size == 1);
   memset(dilated, weight->zero_point, size);
   for (unsigned o = 0; o < outer; o++)
      for (unsigned y = 0; y < height; y++)
         for (unsigned x = 0; x < width; x++) {
            unsigned src = ((o * height + y) * width + x) * channels;
            unsigned dst = ((o * dilated_height + y * dilation_y) *
                               dilated_width +
                            x * dilation_x) *
                           channels;
            memcpy(dilated + dst, weight->data + src, channels);
         }
   weight->dims[1] = dilated_height;
   weight->dims[2] = dilated_width;
   weight->data = dilated;
   return dilated;
}

static void
ethosu_lower_convolution(struct ethosu_subgraph *subgraph,
                         const struct pipe_ml_operation *poperation,
                         struct pipe_tensor *input_tensor,
                         struct ethosu_operation *operation)
{
   int32_t *bias_data = poperation->conv.bias_tensor ? (int32_t *)poperation->conv.bias_tensor->data : NULL;
   struct pipe_tensor conv_weight = *poperation->conv.weight_tensor;
   uint8_t *transposed_weights = NULL;
   uint8_t *dilated_weights = NULL;
   unsigned dilation_y = poperation->conv.dilation_height_factor;
   unsigned dilation_x = poperation->conv.dilation_width_factor;

   operation->conv.depthwise = is_depthwise(poperation);

   if (poperation->conv.depthwise && !operation->conv.depthwise) {
      unsigned height = conv_weight.dims[1];
      unsigned width = conv_weight.dims[2];
      unsigned output_depth = poperation->output_tensors[0]->dims[3];

      assert(input_tensor->dims[3] == 1);
      assert(conv_weight.dims[0] == 1);
      assert(conv_weight.dims[3] == output_depth);

      transposed_weights = malloc(height * width * output_depth);
      for (unsigned o = 0; o < output_depth; o++) {
         for (unsigned y = 0; y < height; y++) {
            for (unsigned x = 0; x < width; x++) {
               transposed_weights[(o * height * width) + (y * width) + x] =
                  poperation->conv.weight_tensor->data[(y * width * output_depth) +
                                                       (x * output_depth) + o];
            }
         }
      }

      conv_weight.dims[0] = output_depth;
      conv_weight.dims[3] = 1;
      conv_weight.data = transposed_weights;
   }

   if (dilation_y > 2 || dilation_x > 2) {
      unsigned manual_dilation_y;
      unsigned manual_dilation_x;
      dilation_y = dilation_y % 2 == 0 ? 2 : 1;
      dilation_x = dilation_x % 2 == 0 ? 2 : 1;
      manual_dilation_y = poperation->conv.dilation_height_factor / dilation_y;
      manual_dilation_x = poperation->conv.dilation_width_factor / dilation_x;
      dilated_weights = dilate_conv_weights(&conv_weight, manual_dilation_y,
                                            manual_dilation_x);
   }

   operation->kernel.height = conv_weight.dims[1];
   operation->kernel.width = conv_weight.dims[2];
   operation->kernel.stride_y = poperation->conv.stride_y;
   operation->kernel.stride_x = poperation->conv.stride_x;
   operation->kernel.dilation_y = dilation_y;
   operation->kernel.dilation_x = dilation_x;
   operation->kernel.depthwise = is_depthwise(poperation);
   operation->kernel.scale = conv_weight.scale;
   operation->kernel.zero_point = conv_weight.zero_point;
   operation->kernel.is_signed = conv_weight.is_signed;

   operation->pad.top = poperation->conv.padding_top;
   operation->pad.bottom = poperation->conv.padding_bottom;
   operation->pad.left = poperation->conv.padding_left;
   operation->pad.right = poperation->conv.padding_right;
   operation->conv.activation_min = poperation->conv.activation_min;
   operation->conv.activation_max = poperation->conv.activation_max;

   lower_conv_common(subgraph, poperation, input_tensor,
                     &conv_weight,
                     (int32_t *)bias_data,
                     operation);

   free(transposed_weights);
   free(dilated_weights);
}

static void
ethosu_lower_pooling(struct ethosu_subgraph *subgraph,
                     const struct pipe_ml_operation *poperation,
                     struct ethosu_operation *operation)
{
   bool avg_pool = poperation->pooling.type == PIPE_ML_POOLING_TYPE_AVG;

   operation->type = ETHOSU_OPERATION_TYPE_POOLING;

   switch (poperation->pooling.type) {
   case PIPE_ML_POOLING_TYPE_MAX:
      operation->pooling.type = ETHOSU_POOLING_TYPE_MAX;
      break;
   case PIPE_ML_POOLING_TYPE_AVG:
      if (ethosu_ml_device(subgraph->base.device)->is_u65 ||
          ((poperation->pooling.filter_height <= 8) &&
           (poperation->pooling.filter_width <= 8)))
         operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;
      else
         operation->pooling.type = ETHOSU_POOLING_TYPE_SUM;
      break;
   default:
      assert(0 && "Unsupported pooling type");
   }

   set_feature_maps(subgraph, poperation->input_tensors[0], poperation->output_tensors[0], operation);

   if (avg_pool) {
      operation->ifm.zero_point = 0;
      operation->ofm.zero_point = 0;
      operation->round_mode = ETHOSU_ROUNDING_NATURAL;
   }

   operation->kernel.height = poperation->pooling.filter_height;
   operation->kernel.width = poperation->pooling.filter_width;
   operation->kernel.stride_y = poperation->pooling.stride_y;
   operation->kernel.stride_x = poperation->pooling.stride_x;

   operation->pad.top = poperation->pooling.padding_top;
   operation->pad.bottom = poperation->pooling.padding_bottom;
   operation->pad.left = poperation->pooling.padding_left;
   operation->pad.right = poperation->pooling.padding_right;

   allocate_feature_maps(subgraph, operation);
   ethosu_sched_operation(subgraph, operation);
}

static void
ethosu_lower_argmax_u85(struct ethosu_subgraph *subgraph,
                        const struct pipe_ml_operation *poperation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];
   struct pipe_tensor *output = poperation->output_tensors[0];
   const unsigned channels = input->dims[3];
   const unsigned elements = input->dims[1] * input->dims[2];
   /* IFM and OFM height fields are 16 bits. */
   const unsigned max_elements = 1 << 16;
   unsigned input_span;
   struct ethosu_operation operation;
   struct ethosu_feature_map input_fm;
   struct ethosu_feature_map output_fm;

   operation_set_defaults(&operation);
   set_feature_maps(subgraph, input, output, &operation);

   input_fm = operation.ifm;
   output_fm = operation.ofm;
   input_span = ethosu_feature_map_span(&input_fm);
   input_fm.tensor->required_size =
      MAX2(input_fm.tensor->required_size, input_span);
   output_fm.tensor->required_size =
      MAX2(output_fm.tensor->required_size,
           ethosu_feature_map_span(&output_fm));
   output_fm.scale = 1.0f;
   output_fm.zero_point = 0;

   for (unsigned offset = 0; offset < elements; offset += max_elements) {
      const unsigned count = MIN2(max_elements, elements - offset);
      struct ethosu_feature_map chunk_input_fm = input_fm;
      struct ethosu_feature_map chunk_output_fm = output_fm;

      chunk_input_fm.shape = (struct ethosu_block){channels, count, 1};
      chunk_input_fm.stride.y = channels * input->type_size;
      chunk_input_fm.stride.x = input->type_size;
      chunk_input_fm.stride.c = input->type_size;
      chunk_input_fm.tiles.addresses[0] += offset * channels * input->type_size;

      chunk_output_fm.shape = (struct ethosu_block){1, count, 1};
      chunk_output_fm.stride.y = output->type_size;
      chunk_output_fm.stride.x = output->type_size;
      chunk_output_fm.stride.c = output->type_size;
      chunk_output_fm.tiles.addresses[0] += offset * output->type_size;

      ethosu_append_pool(subgraph, ETHOSU_POOLING_TYPE_ARGMAX_X,
                         &chunk_input_fm, &chunk_output_fm, channels,
                         ETHOSU_ROUNDING_TRUNCATE);
   }
}

static void
ethosu_lower_argmax_u65(struct ethosu_subgraph *subgraph,
                        const struct pipe_ml_operation *poperation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];
   struct pipe_tensor *output = poperation->output_tensors[0];
   const unsigned channels = input->dims[3];
   const unsigned elements = input->dims[1] * input->dims[2];
   const unsigned max_elements = 1 << 16;
   const unsigned rows_per_chunk = max_elements / input->dims[2];
   const unsigned max_chunk_elements = MIN2(elements,
                                            rows_per_chunk * input->dims[2]);
   const struct ethosu_block input_shape = {
      input->dims[2],
      input->dims[1],
      channels,
   };
   const struct ethosu_block reduced_shape = {1, max_chunk_elements, 1};
   struct ethosu_tensor *conv_tensor;
   struct ethosu_tensor *max_tensor;
   struct ethosu_tensor *lut_tensor;
   struct ethosu_feature_map input_fm = {0};
   struct ethosu_feature_map conv_fm = {0};
   struct ethosu_feature_map max_fm = {0};
   struct ethosu_feature_map lut_fm = {0};
   struct ethosu_feature_map output_fm = {0};
   struct ethosu_feature_map zero_fm = {0};
   struct ethosu_feature_map prepared_ifm;
   struct ethosu_feature_map prepared_ofm;
   struct ethosu_operation operation;
   struct ethosu_operation dma_operation;
   struct pipe_tensor zero = {
      .dims = {1, 1, 1, 1},
      .scale = 1.0f,
      .is_signed = true,
      .type_size = 2,
   };
   uint8_t weights[channels];
   int32_t biases[channels];
   uint32_t lut[512];

   set_feature_map(subgraph, input, &input_fm);
   input_fm.scale = 1.0f;
   input_fm.zero_point = 0;

   conv_tensor = ethosu_add_internal_tensor(subgraph, input_shape, 2);
   max_tensor = ethosu_add_internal_tensor(subgraph, reduced_shape, 2);
   lut_tensor = ethosu_add_internal_tensor(subgraph, reduced_shape, 2);
   set_internal_feature_map(conv_tensor, input_shape, 1.0f, 0, true, &conv_fm);
   set_internal_feature_map(max_tensor, reduced_shape, 1.0f, 0, true, &max_fm);
   set_internal_feature_map(lut_tensor, reduced_shape, 1.0f, 0, true, &lut_fm);
   set_feature_map(subgraph, output, &output_fm);
   /* Reserve the complete output before lowering it as row chunks. */
   output_fm.shape = (struct ethosu_block){1, elements, 1};
   output_fm.stride.y = output->type_size;
   output_fm.stride.x = output->type_size;
   output_fm.stride.c = output->type_size;
   output_fm.tensor->required_size =
      MAX2(output_fm.tensor->required_size, ethosu_feature_map_span(&output_fm));
   output_fm.scale = 1.0f;
   output_fm.zero_point = 0;
   set_constant_tensor_feature_map(subgraph, &zero, &zero_fm);

   memset(weights, 1 << 7, sizeof(weights));
   for (unsigned i = 0; i < channels; i++)
      biases[i] = channels - i - 1;

   operation_set_defaults(&operation);
   prepared_ifm = input_fm;
   prepared_ofm = conv_fm;
   ethosu_prepare_feature_map(subgraph, &prepared_ifm);
   ethosu_prepare_feature_map(subgraph, &prepared_ofm);
   operation.type = ETHOSU_OPERATION_TYPE_CONVOLUTION;
   operation.ifm = prepared_ifm;
   operation.ofm = prepared_ofm;
   operation.conv.depthwise = true;
   operation.kernel.depthwise = true;
   operation.kernel.scale = 1.0f;
   operation.kernel.zero_point = 0;
   operation.kernel.is_signed = false;
   set_full_activation_range(&operation);
   ethosu_sched_operation(subgraph, &operation);
   fill_coefs(subgraph, &operation, biases, weights, sizeof(weights));
   util_dynarray_append(&subgraph->operations, operation);

   for (unsigned i = 0; i < ARRAY_SIZE(lut); i++)
      lut[i] = ((uint32_t)(uint16_t)-128 << 16) + channels - 1;

   for (unsigned row = 0; row < input->dims[1]; row += rows_per_chunk) {
      const unsigned rows = MIN2(rows_per_chunk, input->dims[1] - row);
      const unsigned chunk_elements = rows * input->dims[2];
      const unsigned offset = row * input->dims[2];
      const struct ethosu_block vector_shape = {channels, chunk_elements, 1};
      const struct ethosu_block chunk_shape = {1, chunk_elements, 1};
      struct ethosu_feature_map vector_fm = conv_fm;
      struct ethosu_feature_map chunk_max_fm = max_fm;
      struct ethosu_feature_map chunk_lut_fm = lut_fm;
      struct ethosu_feature_map chunk_output_fm = output_fm;

      vector_fm.shape = vector_shape;
      vector_fm.stride.y = channels * 2;
      vector_fm.stride.x = 2;
      vector_fm.stride.c = 2;
      vector_fm.tiles.addresses[0] += offset * channels * 2;
      chunk_max_fm.shape = chunk_shape;
      chunk_lut_fm.shape = chunk_shape;
      chunk_output_fm.shape = chunk_shape;
      chunk_output_fm.stride.y = 4;
      chunk_output_fm.stride.x = 4;
      chunk_output_fm.stride.c = 4;
      chunk_output_fm.tiles.addresses[0] += offset * 4;

      ethosu_append_pool(subgraph, ETHOSU_POOLING_TYPE_MAX, &vector_fm,
                         &chunk_max_fm, channels, ETHOSU_ROUNDING_DOUBLE);

      operation_set_defaults(&operation);
      prepared_ifm = chunk_max_fm;
      prepared_ofm = chunk_lut_fm;
      ethosu_prepare_feature_map(subgraph, &prepared_ifm);
      ethosu_prepare_feature_map(subgraph, &prepared_ofm);
      operation.type = ETHOSU_OPERATION_TYPE_POOLING;
      operation.pooling.type = ETHOSU_POOLING_TYPE_AVG;
      operation.round_mode = ETHOSU_ROUNDING_NATURAL;
      operation.ifm = prepared_ifm;
      operation.ofm = prepared_ofm;
      operation.activation =
         ethosu_lut_activation(subgraph, &operation.ifm, &operation.ofm,
                               sizeof(lut), false);
      fill_lut(subgraph, &operation, lut, sizeof(lut));
      operation_set_defaults(&dma_operation);
      ethosu_lower_lut_dma(subgraph, NULL, &operation, &dma_operation);
      util_dynarray_append(&subgraph->operations, dma_operation);
      ethosu_append_operation(subgraph, &operation);

      ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_ADD,
                            &chunk_lut_fm, &zero_fm, &chunk_output_fm,
                            ETHOSU_ROUNDING_NATURAL, true, 0, 0);
   }
}

static void
ethosu_lower_argmax(struct ethosu_subgraph *subgraph,
                    const struct pipe_ml_operation *poperation)
{
   if (ethosu_ml_device(subgraph->base.device)->is_u65) {
      ethosu_lower_argmax_u65(subgraph, poperation);
      return;
   }

   ethosu_lower_argmax_u85(subgraph, poperation);
}

static void
set_constant_tensor_data_feature_map(struct ethosu_subgraph *subgraph,
                                     struct pipe_tensor *tensor,
                                     const void *data,
                                     struct ethosu_feature_map *fm)
{
   struct ethosu_block shape = {
      tensor->dims[2],
      tensor->dims[1],
      tensor->dims[3],
   };
   unsigned elements = shape.height * shape.width * shape.depth;
   unsigned size = elements * tensor->type_size;
   struct ethosu_tensor *ethosu_tensor;

   ethosu_tensor = ethosu_add_internal_tensor(subgraph, shape,
                                              tensor->type_size);
   set_internal_feature_map(ethosu_tensor, shape, tensor->scale,
                            tensor->zero_point, tensor->is_signed, fm);
   fm->region = COEFS_REGION;
   fm->tiles.addresses[0] = ethosu_add_constant(subgraph, data, size);
   fm->tiles.height_0 = shape.height;
   fm->tiles.height_1 = shape.height;
   fm->tiles.width_0 = shape.width;
}

static void
set_constant_tensor_feature_map_value(struct ethosu_subgraph *subgraph,
                                      struct pipe_tensor *tensor,
                                      int value,
                                      struct ethosu_feature_map *fm)
{
   unsigned elements = tensor->dims[1] * tensor->dims[2] * tensor->dims[3];
   unsigned size = elements * tensor->type_size;
   uint8_t *data = malloc(size);

   if (tensor->type_size == 1) {
      memset(data, value, size);
   } else {
      int16_t *data16 = (int16_t *)data;

      for (unsigned i = 0; i < elements; i++)
         data16[i] = value;
   }

   set_constant_tensor_data_feature_map(subgraph, tensor, data, fm);

   free(data);
}

static void
create_pad_constant(struct ethosu_subgraph *subgraph,
                    const struct pipe_tensor *tensor, unsigned elements,
                    int value,
                    struct ethosu_tensor **out_tensor, unsigned *out_address)
{
   struct ethosu_block shape = {elements, 1, 1};
   unsigned size = elements * tensor->type_size;
   uint8_t *data = malloc(size);

   if (tensor->type_size == 1) {
      memset(data, value, size);
   } else {
      int16_t *data16 = (int16_t *)data;

      for (unsigned i = 0; i < elements; i++)
         data16[i] = value;
   }

   *out_tensor = ethosu_add_internal_tensor(subgraph, shape,
                                            tensor->type_size);
   *out_address = ethosu_add_constant(subgraph, data, size);

   free(data);
}

static void
set_pad_ifm(struct ethosu_subgraph *subgraph,
            const struct pipe_tensor *tensor,
            struct ethosu_tensor *constant, unsigned address,
            unsigned height, unsigned width, unsigned depth,
            struct ethosu_feature_map *fm)
{
   struct ethosu_block shape = {width, height, depth};

   /* Shape the IFM to the strip this pass writes so that its stride is the
    * strip's own and the pass reads the constant as a contiguous window
    * from the front, rather than a corner of a full-output constant. */
   set_internal_feature_map(constant, shape, tensor->scale,
                            tensor->zero_point, tensor->is_signed, fm);
   fm->region = COEFS_REGION;
   fm->tiles.addresses[0] = address;
   fm->tiles.height_0 = height;
   fm->tiles.height_1 = height;
   fm->tiles.width_0 = width;
}

static void
set_pad_ofm(struct ethosu_subgraph *subgraph,
            const struct pipe_ml_operation *poperation,
            unsigned y, unsigned x, unsigned z,
            unsigned height, unsigned width, unsigned depth,
            struct ethosu_feature_map *ofm)
{
   struct pipe_tensor *output = poperation->output_tensors[0];

   set_feature_map(subgraph, output, ofm);
   ofm->tensor->required_size = MAX2(ofm->tensor->required_size,
                                     ethosu_feature_map_span(ofm));
   ofm->shape.height = height;
   ofm->shape.width = width;
   ofm->shape.depth = depth;
   ofm->tiles.addresses[0] =
      ((y * output->dims[2] + x) * output->dims[3] + z) *
      output->type_size;
   ofm->tiles.height_0 = ofm->shape.height;
   ofm->tiles.height_1 = ofm->shape.height;
   ofm->tiles.width_0 = ofm->shape.width;
}

static void
ethosu_append_pool_nop(struct ethosu_subgraph *subgraph,
                       struct ethosu_feature_map *ifm,
                       struct ethosu_feature_map *ofm)
{
   struct ethosu_operation operation;

   operation_set_defaults(&operation);

   operation.type = ETHOSU_OPERATION_TYPE_POOLING;
   operation.round_mode = ETHOSU_ROUNDING_NATURAL;
   operation.pooling.nop = true;
   if (ethosu_ml_device(subgraph->base.device)->is_u65)
      operation.pooling.type = ETHOSU_POOLING_TYPE_AVG;
   else
      operation.pooling.type = ETHOSU_POOLING_TYPE_SUM;
   operation.ifm = *ifm;
   operation.ofm = *ofm;

   /* set_pad_ofm leaves the destination's byte offset within the output
    * tensor in the OFM address; add it back to the base the allocator
    * assigns rather than pre-seeding the address for every operation.  The
    * IFM offset is kept the same way, but only for an IO_REGION input: a
    * transpose copies each row from a window into the IO input at a per-row
    * byte offset that allocate_feature_maps would otherwise overwrite with
    * the tensor base, while a constant IFM's address is absolute and the
    * allocator leaves it alone. */
   unsigned ofm_offset = operation.ofm.tiles.addresses[0];
   unsigned ifm_offset = operation.ifm.tiles.addresses[0];
   allocate_feature_maps(subgraph, &operation);
   operation.ofm.tiles.addresses[0] += ofm_offset;
   if (operation.ifm.region == IO_REGION)
      operation.ifm.tiles.addresses[0] += ifm_offset;

   ethosu_append_operation(subgraph, &operation);
}

static void
ethosu_lower_pad(struct ethosu_subgraph *subgraph,
                 const struct pipe_ml_operation *poperation,
                 struct pipe_tensor *input)
{
   struct ethosu_feature_map input_fm = {0};
   struct ethosu_feature_map inner_ofm = {0};
   struct ethosu_feature_map pad_ifm = {0};
   struct ethosu_feature_map pad_ofm = {0};
   const struct pipe_tensor *output = poperation->output_tensors[0];
   unsigned oh = output->dims[1];
   unsigned ow = output->dims[2];
   unsigned oc = output->dims[3];
   struct ethosu_tensor *constant;
   unsigned address;

   /* Every pad region is filled from one constant, read as a window shaped
    * to the region, so the constant must hold the largest such window. */
   unsigned pad_y = MAX2(poperation->pad.before_y, poperation->pad.after_y);
   unsigned pad_x = MAX2(poperation->pad.before_x, poperation->pad.after_x);
   unsigned pad_z = MAX2(poperation->pad.before_z, poperation->pad.after_z);
   unsigned elements = MAX2(MAX2(ow * oc * pad_y, oh * oc * pad_x),
                            oh * ow * pad_z);

   create_pad_constant(subgraph, output, elements,
                       poperation->pad.raw_zero ? 0 : output->zero_point,
                       &constant, &address);

   if (input->data) {
      set_constant_tensor_data_feature_map(subgraph,
                                           input, input->data,
                                           &input_fm);
   } else {
      set_feature_map(subgraph, input, &input_fm);
   }
   set_pad_ofm(subgraph, poperation,
               poperation->pad.before_y, poperation->pad.before_x,
               poperation->pad.before_z, input->dims[1], input->dims[2],
               input->dims[3], &inner_ofm);
   ethosu_append_pool_nop(subgraph, &input_fm, &inner_ofm);

   if (poperation->pad.before_y) {
      set_pad_ofm(subgraph, poperation, 0, 0, 0,
                  poperation->pad.before_y, ow, oc, &pad_ofm);
      set_pad_ifm(subgraph, output, constant, address,
                  poperation->pad.before_y, ow, oc, &pad_ifm);
      ethosu_append_pool_nop(subgraph, &pad_ifm, &pad_ofm);
   }

   if (poperation->pad.after_y) {
      set_pad_ofm(subgraph, poperation,
                  poperation->pad.before_y + input->dims[1], 0, 0,
                  poperation->pad.after_y, ow, oc, &pad_ofm);
      set_pad_ifm(subgraph, output, constant, address,
                  poperation->pad.after_y, ow, oc, &pad_ifm);
      ethosu_append_pool_nop(subgraph, &pad_ifm, &pad_ofm);
   }

   if (poperation->pad.before_x) {
      set_pad_ofm(subgraph, poperation, poperation->pad.before_y, 0, 0,
                  input->dims[1], poperation->pad.before_x, oc, &pad_ofm);
      set_pad_ifm(subgraph, output, constant, address,
                  input->dims[1], poperation->pad.before_x, oc, &pad_ifm);
      ethosu_append_pool_nop(subgraph, &pad_ifm, &pad_ofm);
   }

   if (poperation->pad.after_x) {
      set_pad_ofm(subgraph, poperation, poperation->pad.before_y,
                  poperation->pad.before_x + input->dims[2], 0,
                  input->dims[1], poperation->pad.after_x, oc, &pad_ofm);
      set_pad_ifm(subgraph, output, constant, address,
                  input->dims[1], poperation->pad.after_x, oc, &pad_ifm);
      ethosu_append_pool_nop(subgraph, &pad_ifm, &pad_ofm);
   }

   if (poperation->pad.before_z) {
      set_pad_ofm(subgraph, poperation, poperation->pad.before_y,
                  poperation->pad.before_x, 0, input->dims[1], input->dims[2],
                  poperation->pad.before_z, &pad_ofm);
      set_pad_ifm(subgraph, output, constant, address,
                  input->dims[1], input->dims[2], poperation->pad.before_z,
                  &pad_ifm);
      ethosu_append_pool_nop(subgraph, &pad_ifm, &pad_ofm);
   }

   if (poperation->pad.after_z) {
      set_pad_ofm(subgraph, poperation, poperation->pad.before_y,
                  poperation->pad.before_x,
                  poperation->pad.before_z + input->dims[3], input->dims[1],
                  input->dims[2], poperation->pad.after_z, &pad_ofm);
      set_pad_ifm(subgraph, output, constant, address,
                  input->dims[1], input->dims[2], poperation->pad.after_z,
                  &pad_ifm);
      ethosu_append_pool_nop(subgraph, &pad_ifm, &pad_ofm);
   }
}

static void
ethosu_lower_mean(struct ethosu_subgraph *subgraph,
                  const struct pipe_ml_operation *poperation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];
   struct pipe_tensor *output = poperation->output_tensors[0];
   struct pipe_tensor reshaped_input;
   struct pipe_tensor reshaped_output;
   struct ethosu_operation sum_operation;
   struct ethosu_feature_map sum_fm = {0};
   struct ethosu_feature_map output_fm = {0};
   struct ethosu_feature_map scale_fm = {0};
   struct ethosu_block sum_shape;
   struct ethosu_tensor *sum_tensor;
   unsigned kernel_height;
   unsigned kernel_width;
   unsigned kernel_elements;
   unsigned weights_size;
   uint8_t *weights;
   int32_t *biases;
   int32_t scale_shift;
   int32_t scale;
   int32_t quant_scale;
   int32_t quant_shift;
   int mean_shift = 0;

   if (poperation->mean.axes == BITFIELD_BIT(3)) {
      reshaped_input = *input;
      reshaped_output = *output;

      if (input->dims[1] == 1) {
         reshaped_input.dims[1] = input->dims[2];
         reshaped_output.dims[1] = output->dims[2];
      }

      reshaped_input.dims[2] = input->dims[3];
      reshaped_input.dims[3] = 1;
      reshaped_output.dims[2] = 1;
      reshaped_output.dims[3] = 1;
      input = &reshaped_input;
      output = &reshaped_output;
   }

   kernel_height = poperation->mean.axes == BITFIELD_BIT(3) ? 1 : input->dims[1];
   kernel_width = input->dims[2];
   sum_shape = (struct ethosu_block){output->dims[2], output->dims[1],
                                     output->dims[3]};
   sum_tensor = ethosu_add_internal_tensor(subgraph, sum_shape, 4);
   kernel_elements = kernel_height * kernel_width;
   weights_size = kernel_elements * input->dims[3];
   weights = malloc(weights_size);
   biases = calloc(output->dims[3], sizeof(*biases));
   quant_scale = ethosu_quantize_scale((double)input->scale / output->scale,
                                       &quant_shift, false);

   for (unsigned elements = kernel_elements; elements > 1; elements >>= 1)
      mean_shift++;

   mean_shift = MIN2(mean_shift, 32);
   mean_shift = MIN2(mean_shift, 62 - quant_shift);
   scale = ((int64_t)quant_scale << mean_shift) / kernel_elements;
   scale_shift = quant_shift + mean_shift;

   memset(weights, 1, weights_size);

   sum_tensor->layout = ETHOSU_LAYOUT_NHCWB16;

   operation_set_defaults(&sum_operation);
   sum_operation.type = ETHOSU_OPERATION_TYPE_CONVOLUTION;
   sum_operation.conv.depthwise = true;
   set_feature_map(subgraph, input, &sum_operation.ifm);
   set_internal_feature_map(sum_tensor, sum_shape, input->scale, 0, true,
                            &sum_operation.ofm);
   sum_operation.kernel.height = kernel_height;
   sum_operation.kernel.width = kernel_width;
   sum_operation.kernel.depthwise = true;
   sum_operation.kernel.scale = 1.0f;
   sum_operation.kernel.zero_point = 0;
   sum_operation.kernel.is_signed = true;
   sum_operation.round_mode = ETHOSU_ROUNDING_DOUBLE;
   set_full_activation_range(&sum_operation);
   allocate_feature_maps(subgraph, &sum_operation);
   ethosu_sched_operation(subgraph, &sum_operation);
   fill_coefs(subgraph, &sum_operation, biases, weights, weights_size);
   util_dynarray_append(&subgraph->operations, sum_operation);

   set_internal_feature_map(sum_tensor, sum_shape, input->scale, 0, true,
                            &sum_fm);
   set_feature_map(subgraph, output, &output_fm);
   set_constant_feature_map(subgraph, scale, &scale_fm);

   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_MUL, &sum_fm,
                         &scale_fm, &output_fm, ETHOSU_ROUNDING_DOUBLE,
                         false, 1, scale_shift);

   free(biases);
   free(weights);
}

static double
clamp_sigmoid8(double x)
{
   if (x <= -8.0)
      return 0.0;
   else if (x >= 8.0)
      return 1.0;
   else
      return (1.0 / (1.0 + exp(-x)));
}

static void
ethos_create_lut(struct ethosu_operation *operation, uint8_t *lut, double (*func)(double))
{
   double ifm_scale = operation->ifm.scale;
   double ofm_scale = operation->ofm.scale;
   int zpIn = operation->ifm.zero_point;
   int zpOut = operation->ofm.zero_point;

   int qMin = operation->ifm.is_signed ? -128 : 0;
   int qMax = operation->ifm.is_signed ? 127 : 255;

   for (int x = qMin; x <= qMax; ++x, lut++) {
      double xReal = ifm_scale * (double)(x - zpIn);
      double yReal = func(xReal);
      int lutVal = (int)round((double)zpOut + yReal / ofm_scale);
      lutVal = MIN2(qMax, MAX2(qMin, lutVal));
      *lut = lutVal;
   }
}

// Implementation from Vela and TensorFlow Lite Micro kernel
static int16_t
saturating_left_shift_16(int16_t value, int amount)
{
   int32_t result = value << amount;
   return CLAMP(result, INT16_MIN, INT16_MAX);
}

// Implementation from Vela and TensorFlow Lite Micro kernel
// Similar to ARM instruction SQDMULH.
// Similar to gemmlowp::SaturatingRoundingDoublingHighMul except
// rounding to zero instead of to nearest (SQRDMULH).
static int16_t
saturating_doubling_high_mul_16(int16_t a, int16_t b)
{
   bool overflow = a == b && a == INT16_MIN;
   int32_t a_32 = a;
   int32_t b_32 = b;
   int32_t ab_32 = a_32 * b_32;
   int16_t ab_x2_high16 = (int16_t)(ab_32 / (1 << 15));
   return overflow ? INT16_MAX : ab_x2_high16;
}

static int16_t
saturating_rounding_doubling_high_mul_16(int16_t a, int16_t b)
{
   bool overflow = a == b && a == INT16_MIN;
   int32_t a_32 = a;
   int32_t b_32 = b;
   int32_t ab_32 = a_32 * b_32;
   int16_t nudge = ab_32 >= 0 ? (1 << 14) : (1 - (1 << 14));
   int16_t ab_x2_high16 = ((ab_32 + nudge) / (1 << 15));
   return overflow ? INT16_MAX : ab_x2_high16;
}

static int16_t
rounding_divide_by_pow2_16(int16_t x, int exponent)
{
   const int16_t mask = (1 << exponent) - 1;
   const int16_t remainder = x & mask;
   const int16_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
   return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
}

static int16_t
downscale_int32_to_int16_multiplier(int32_t multiplier)
{
   return CLAMP(((multiplier / 32768) + 1) / 2, INT16_MIN, INT16_MAX);
}

static void
ethos_create_hswish_lut(struct ethosu_operation *operation, uint8_t *lut)
{
   const double ifm_scale = operation->ifm.scale;
   const double ofm_scale = operation->ofm.scale;
   const unsigned zpIn = operation->ifm.zero_point;
   const unsigned zpOut = operation->ofm.zero_point;

   const int qMin = operation->ifm.is_signed ? -128 : 0;
   const int qMax = operation->ifm.is_signed ? 127 : 255;

   const double ifmScaleHires = (1.0 / 128.0) * ifm_scale;
   const double reluMultiplier = 3.0 / 32768.0;

   int32_t out_shift;
   int32_t relu_shift;
   int32_t out_scale = ethosu_quantize_scale(ifmScaleHires / ofm_scale, &out_shift, false);
   int32_t relu_scale = ethosu_quantize_scale(ifmScaleHires / reluMultiplier, &relu_shift, false);
   int16_t outScale16 = downscale_int32_to_int16_multiplier(out_scale);
   int16_t reluScale16 = downscale_int32_to_int16_multiplier(relu_scale);
   // convert to left shift-positive notation
   int outShift = 31 - out_shift;
   int reluShift = 31 - relu_shift;

   for (int x = qMin; x <= qMax; ++x, lut++) {
      const int16_t inputValue = (int16_t)(x - zpIn);
      const int16_t inputValueOnHiresInputScale = (int16_t)(inputValue << 7);
      const int16_t inputValueOnPreshiftOutputScale = saturating_rounding_doubling_high_mul_16(inputValueOnHiresInputScale, outScale16);
      int16_t reluValue = inputValueOnHiresInputScale;

      if (reluShift > 0)
         reluValue = saturating_left_shift_16(reluValue, reluShift - 1);

      reluValue = saturating_rounding_doubling_high_mul_16(reluValue, reluScale16);

      if (reluShift > 0)
         reluValue = saturating_left_shift_16(reluValue, 1);

      // Try to get reluShift into the [-31, 0] range
      if (reluShift < -31) {
         reluValue = reluValue >> (-31 - reluShift);
         reluShift = -31;
      }

      if (reluShift < 0)
         reluValue = rounding_divide_by_pow2_16(reluValue, -reluShift);

      reluValue = (int16_t)((reluValue + (1 << 15)) >> 1);

      const int16_t preshiftOutputValue = saturating_doubling_high_mul_16(reluValue, inputValueOnPreshiftOutputScale);

      int16_t outputValue = rounding_divide_by_pow2_16(preshiftOutputValue, -outShift);

      int lutVal = outputValue + zpOut;
      lutVal = MIN2(qMax, MAX2(qMin, lutVal));
      *lut = lutVal;
   }
}

static int32_t
saturating_rounding_doubling_high_mul_32(int32_t a, int32_t b)
{
   bool overflow = a == b && a == INT32_MIN;
   int64_t a_64 = a;
   int64_t b_64 = b;
   int64_t ab_64 = a_64 * b_64;
   int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
   int32_t ab_x2_high32 = ((ab_64 + nudge) / (1ll << 31));
   return overflow ? INT32_MAX : ab_x2_high32;
}

static int32_t
rounding_divide_by_pow2_32(int32_t x, int exponent)
{
   const int32_t mask = (1 << exponent) - 1;
   const int32_t remainder = x & mask;
   const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
   return (x >> exponent) + ((remainder > threshold) ? 1 : 0);
}

// Multiplies int with QuantizedScale with rounding.
static int
multiply_by_quantized_multiplier(int x, int shift, int32_t scale)
{
   // Multiplies x (int32) by QuantizedScale (scale, shift), returns rounded result.
   // Expects the QuantizedScale to be left-shift positive.
   const int leftShift = shift > 0 ? shift : 0;
   const int rightShift = shift < 0 ? -shift : 0;
   const int32_t mul = saturating_rounding_doubling_high_mul_32(x * (1 << leftShift), scale);
   return rounding_divide_by_pow2_32(mul, rightShift);
}

/* Reference-derived reciprocal square root values in Q20 format. */
static const int32_t rsqrt_lut[] = {
   0x00000000, 0x00100000, 0x000b504e, 0x00093cd4, 0x00080000, 0x000727c9,
   0x0006882f, 0x00060c24, 0x0005a827, 0x00055555, 0x00050f45, 0x0004d2fe,
   0x00049e6a, 0x00047007, 0x000446b4, 0x00042195, 0x00040000, 0x0003e16d,
   0x0003c570, 0x0003abb0, 0x000393e5, 0x00037dd2, 0x00036945, 0x00035613,
   0x00034418, 0x00033333, 0x0003234b, 0x00031447, 0x00030612, 0x0002f89c,
   0x0002ebd3, 0x0002dfaa, 0x0002d414, 0x0002c906, 0x0002be75, 0x0002b45a,
   0x0002aaab, 0x0002a161, 0x00029875, 0x00028fe3, 0x000287a2, 0x00027fb0,
   0x00027807, 0x000270a2, 0x0002697f, 0x00026298, 0x00025bec, 0x00025577,
   0x00024f35, 0x00024925, 0x00024343, 0x00023d8e, 0x00023803, 0x000232a1,
   0x00022d65, 0x0002284e, 0x0002235a, 0x00021e87, 0x000219d5, 0x00021541,
   0x000210cb, 0x00020c70, 0x00020831, 0x0002040c, 0x00020000, 0x0001fc0c,
   0x0001f82f, 0x0001f468, 0x0001f0b7, 0x0001ed1a, 0x0001e991, 0x0001e61b,
   0x0001e2b8, 0x0001df67, 0x0001dc26, 0x0001d8f7, 0x0001d5d8, 0x0001d2c8,
   0x0001cfc8, 0x0001ccd6, 0x0001c9f2, 0x0001c71c, 0x0001c454, 0x0001c198,
   0x0001bee9, 0x0001bc46, 0x0001b9af, 0x0001b723, 0x0001b4a3, 0x0001b22d,
   0x0001afc2, 0x0001ad61, 0x0001ab0a, 0x0001a8bc, 0x0001a678, 0x0001a43e,
   0x0001a20c, 0x00019fe3, 0x00019dc2, 0x00019baa, 0x0001999a, 0x00019791,
   0x00019590, 0x00019397, 0x000191a5, 0x00018fbb, 0x00018dd7, 0x00018bfa,
   0x00018a23, 0x00018853, 0x0001868a, 0x000184c6, 0x00018309, 0x00018152,
   0x00017fa0, 0x00017df4, 0x00017c4e, 0x00017aad, 0x00017911, 0x0001777b,
   0x000175e9, 0x0001745d, 0x000172d6, 0x00017153, 0x00016fd5, 0x00016e5b,
   0x00016ce7, 0x00016b76, 0x00016a0a, 0x000168a2, 0x0001673e, 0x000165de,
   0x00016483, 0x0001632b, 0x000161d7, 0x00016087, 0x00015f3b, 0x00015df2,
   0x00015cad, 0x00015b6b, 0x00015a2d, 0x000158f2, 0x000157bb, 0x00015686,
   0x00015555, 0x00015427, 0x000152fd, 0x000151d5, 0x000150b0, 0x00014f8f,
   0x00014e70, 0x00014d54, 0x00014c3b, 0x00014b24, 0x00014a11, 0x00014900,
   0x000147f1, 0x000146e5, 0x000145dc, 0x000144d5, 0x000143d1, 0x000142cf,
   0x000141d0, 0x000140d3, 0x00013fd8, 0x00013ee0, 0x00013de9, 0x00013cf5,
   0x00013c03, 0x00013b14, 0x00013a26, 0x0001393b, 0x00013851, 0x0001376a,
   0x00013684, 0x000135a1, 0x000134bf, 0x000133e0, 0x00013302, 0x00013226,
   0x0001314c, 0x00013074, 0x00012f9e, 0x00012ec9, 0x00012df6, 0x00012d25,
   0x00012c55, 0x00012b87, 0x00012abb, 0x000129f1, 0x00012928, 0x00012860,
   0x0001279a, 0x000126d6, 0x00012613, 0x00012552, 0x00012492, 0x000123d4,
   0x00012317, 0x0001225c, 0x000121a2, 0x000120e9, 0x00012032, 0x00011f7c,
   0x00011ec7, 0x00011e14, 0x00011d62, 0x00011cb1, 0x00011c02, 0x00011b54,
   0x00011aa7, 0x000119fb, 0x00011950, 0x000118a7, 0x000117ff, 0x00011758,
   0x000116b3, 0x0001160e, 0x0001156b, 0x000114c8, 0x00011427, 0x00011387,
   0x000112e8, 0x0001124a, 0x000111ad, 0x00011111, 0x00011076, 0x00010fdc,
   0x00010f44, 0x00010eac, 0x00010e15, 0x00010d7f, 0x00010cea, 0x00010c56,
   0x00010bc4, 0x00010b32, 0x00010aa0, 0x00010a10, 0x00010981, 0x000108f3,
   0x00010865, 0x000107d9, 0x0001074d, 0x000106c2, 0x00010638, 0x000105af,
   0x00010527, 0x0001049f, 0x00010419, 0x00010393, 0x0001030e, 0x0001028a,
   0x00010206, 0x00010183, 0x00010102, 0x00010080};

static void
ethos_create_rsqrt_lut(struct ethosu_operation *operation, uint8_t *lut)
{
   const int qmin = -128;
   const int qmax = 127;
   float output_scale;
   int32_t scale_shift;
   int32_t scale;

   output_scale = sqrtf(operation->ifm.scale) * operation->ofm.scale;
   scale = ethosu_quantize_scale(1.0 / output_scale, &scale_shift, false);
   scale_shift = 31 - scale_shift - 20;

   for (int x = qmin; x <= qmax; x++, lut++) {
      int index = MAX2(0, x - operation->ifm.zero_point);
      int value = index ? operation->ofm.zero_point +
                             multiply_by_quantized_multiplier(rsqrt_lut[index],
                                                              scale_shift, scale)
                        : qmax;

      *lut = CLAMP(value, qmin, qmax);
   }
}

static int32_t
clamp_i64_to_i32(int64_t value)
{
   return CLAMP(value, INT32_MIN, INT32_MAX);
}

static int32_t
saturating_rounding_multiply_by_pow2_32(int32_t x, int exponent)
{
   if (exponent < 0)
      return rounding_divide_by_pow2_32(x, -exponent);

   int64_t shifted = (int64_t)x * (1LL << exponent);
   return clamp_i64_to_i32(shifted);
}

static int32_t
fixedpoint_mul_q0(int32_t a, int32_t b)
{
   return saturating_rounding_doubling_high_mul_32(a, b);
}

static int32_t
fixedpoint_exp_on_interval_between_negative_one_quarter_and_0(int32_t a)
{
   const int32_t constant_term = 1895147668;
   const int32_t constant_1_over_3 = 715827883;
   int32_t x = a + (1 << 28);
   int32_t x2 = fixedpoint_mul_q0(x, x);
   int32_t x3 = fixedpoint_mul_q0(x2, x);
   int32_t x4 = fixedpoint_mul_q0(x2, x2);
   int32_t x4_over_4 = saturating_rounding_multiply_by_pow2_32(x4, -2);
   int32_t t = fixedpoint_mul_q0(x4_over_4 + x3, constant_1_over_3) + x2;
   int32_t x4_over_24_plus_x3_over_6_plus_x2_over_2 =
      saturating_rounding_multiply_by_pow2_32(t, -1);

   return constant_term +
          fixedpoint_mul_q0(constant_term,
                            x + x4_over_24_plus_x3_over_6_plus_x2_over_2);
}

static int32_t
fixedpoint_exp_on_negative_values_q5(int32_t a)
{
   const int k_integer_bits = 5;
   const int k_fractional_bits = 31 - k_integer_bits;
   const int32_t k_one_quarter = 1 << (k_fractional_bits - 2);
   int32_t mask = k_one_quarter - 1;
   int32_t a_mod_quarter_minus_one_quarter = (a & mask) - k_one_quarter;
   int32_t result = fixedpoint_exp_on_interval_between_negative_one_quarter_and_0(
      saturating_rounding_multiply_by_pow2_32(a_mod_quarter_minus_one_quarter,
                                              k_integer_bits));
   int32_t remainder = a_mod_quarter_minus_one_quarter - a;

#define ETHOSU_EXP_BARREL_SHIFTER(exponent, multiplier)        \
   do {                                                        \
      const int shift_amount = k_fractional_bits + (exponent); \
      if (remainder & (1 << shift_amount))                     \
         result = fixedpoint_mul_q0(result, multiplier);       \
   } while (0)

   ETHOSU_EXP_BARREL_SHIFTER(-2, 1672461947);
   ETHOSU_EXP_BARREL_SHIFTER(-1, 1302514674);
   ETHOSU_EXP_BARREL_SHIFTER(0, 790015084);
   ETHOSU_EXP_BARREL_SHIFTER(1, 290630308);
   ETHOSU_EXP_BARREL_SHIFTER(2, 39332535);
   ETHOSU_EXP_BARREL_SHIFTER(3, 720401);
   ETHOSU_EXP_BARREL_SHIFTER(4, 242);

#undef ETHOSU_EXP_BARREL_SHIFTER

   return a == 0 ? INT32_MAX : result;
}

static float
clamp(double d)
{
   return (float)CLAMP(d, -FLT_MAX, FLT_MAX);
}

/* Calculate elementwise Mul OFM QuantizedScale */
static int32_t
elementwise_mul_scale(double inputScale, double input2Scale, double outputScale, int32_t *mul_shift)
{
   // clamp to single-point precision
   float ifm1Scale = clamp(inputScale);
   float ifm2Scale = clamp(input2Scale);
   float outScale = clamp(outputScale);

   float outputRescale = (ifm1Scale * ifm2Scale) / outScale;
   return ethosu_quantize_scale(outputRescale, mul_shift, false);
}

static void
ethos_create_leakyrelu_lut(struct ethosu_operation *operation, uint8_t *lut, float alpha)
{
   const double ifm_scale = operation->ifm.scale;
   const double ofm_scale = operation->ofm.scale;
   const int zpIn = operation->ifm.zero_point;
   const int zpOut = operation->ofm.zero_point;
   const int qMin = operation->ifm.is_signed ? -128 : 0;
   const int qMax = operation->ifm.is_signed ? 127 : 255;
   int64_t scalar = 1;
   int32_t identity_shift;
   int32_t identity_scale = elementwise_mul_scale(ifm_scale, 1.0, ofm_scale, &identity_shift);
   int32_t alpha_shift;
   int32_t alpha_scale = elementwise_mul_scale(ifm_scale, alpha, ofm_scale, &alpha_shift);

   for (int x = qMin; x <= qMax; ++x, lut++) {
      int lutResult;
      if (x < zpIn)
         lutResult = zpOut + multiply_by_quantized_multiplier((int)(scalar * (x - zpIn)), 31 - alpha_shift, alpha_scale);
      else
         lutResult = zpOut + multiply_by_quantized_multiplier((int)(x - zpIn), 31 - identity_shift, identity_scale);

      lutResult = MIN2(qMax, MAX2(qMin, lutResult));
      *lut = lutResult;
   }
}

static void
ethosu_lower_lut_dma(struct ethosu_subgraph *subgraph,
                     const struct pipe_ml_operation *poperation,
                     struct ethosu_operation *pool_operation,
                     struct ethosu_operation *operation)
{
   struct ethosu_address_range *range = &pool_operation->lut;
   unsigned activation = pool_operation->activation;

   operation->type = ETHOSU_OPERATION_TYPE_DMA;
   operation->dma.address = range->address;
   operation->dma.size = range->size;
   operation->dma.dst_region = ethosu_lut_region();
   operation->dma.dst_address =
      ethosu_lut_address(subgraph, activation, range->size);
}

static unsigned
ethosu_alloc_lut_slot(struct ethosu_subgraph *subgraph, unsigned size)
{
   unsigned slots = DIV_ROUND_UP(size, LUT_SLOT_SIZE);
   /* The LUT banks hold this many equal-sized slots. */
   unsigned max_slots = SHRAM_RESERVED_END_BANKS * SHRAM_BANK_SIZE_BYTES /
                        LUT_SLOT_SIZE;
   unsigned slot;

   if (ethosu_ml_device(subgraph->base.device)->is_u65) {
      if (size > LUT8_SIZE)
         return 0;

      return subgraph->next_lut_slot++ % max_slots;
   }

   if (slots == 0 || slots > max_slots)
      return 0;

   slot = align(subgraph->next_lut_slot, slots);
   if (slot + slots > max_slots)
      slot = 0;

   subgraph->next_lut_slot = slot + slots;
   return slot;
}

static unsigned
ethosu_lut_activation(struct ethosu_subgraph *subgraph,
                      struct ethosu_feature_map *ifm,
                      struct ethosu_feature_map *ofm,
                      unsigned size,
                      bool force_int8_clip)
{
   unsigned slot = ethosu_alloc_lut_slot(subgraph, size);

   if (ethosu_ml_device(subgraph->base.device)->is_u65)
      return ETHOSU_U65_ACTIVATION_LUT(slot) |
             (force_int8_clip ? ETHOSU_ACTIVATION_CLIP_FORCE_INT8 : 0);

   unsigned slots = DIV_ROUND_UP(size, LUT_SLOT_SIZE);
   unsigned function;

   assert(slots != 0);

   if (ifm->precision == 0 && ofm->precision == 0) {
      function = ifm->is_signed ? ETHOSU_U85_ACTIVATION_LUT_S8_S8 : ETHOSU_U85_ACTIVATION_LUT_U8_U8;
   } else if (ifm->precision == 0 && ofm->precision == 1) {
      assert(ifm->is_signed && ofm->is_signed);
      function = ETHOSU_U85_ACTIVATION_LUT_S8_S16;
   } else if (ifm->precision == 0 && ofm->precision == 2) {
      assert(ifm->is_signed && ofm->is_signed);
      function = ETHOSU_U85_ACTIVATION_LUT_S8_S32;
   } else if (ifm->precision == 1 && ofm->precision == 1) {
      assert(ifm->is_signed && ofm->is_signed);
      function = ETHOSU_U85_ACTIVATION_LUT_S16_S16;
   } else {
      assert(ifm->precision == 1 && ofm->precision == 2);
      assert(ifm->is_signed && ofm->is_signed);
      function = ETHOSU_U85_ACTIVATION_LUT_S16_S32;
   }

   return ETHOSU_U85_ACTIVATION_LUT(function, slot / slots);
}

static void
ethosu_create_simple_lut(const struct pipe_ml_operation *poperation,
                         struct ethosu_operation *operation,
                         uint8_t *lut)
{
   switch (poperation->type) {
   case PIPE_ML_OPERATION_TYPE_LOGISTIC:
      ethos_create_lut(operation, lut, clamp_sigmoid8);
      break;
   case PIPE_ML_OPERATION_TYPE_TANH:
      ethos_create_lut(operation, lut, tanh);
      break;
   case PIPE_ML_OPERATION_TYPE_RSQRT:
      ethos_create_rsqrt_lut(operation, lut);
      break;
   case PIPE_ML_OPERATION_TYPE_HSWISH:
      ethos_create_hswish_lut(operation, lut);
      break;
   case PIPE_ML_OPERATION_TYPE_LEAKY_RELU:
      ethos_create_leakyrelu_lut(operation, lut, poperation->leakyrelu.alpha);
      break;
   default:
      UNREACHABLE("Unsupported LUT operation type");
   }
}

static void
ethosu_fuse_lut(struct ethosu_subgraph *subgraph,
                const struct pipe_ml_operation *lut_operation,
                struct ethosu_operation *operation)
{
   struct ethosu_operation operation_lut = *operation;
   uint8_t lut[LUT8_SIZE];

   operation_lut.ifm = operation->ofm;
   set_feature_map(subgraph, lut_operation->output_tensors[0],
                   &operation_lut.ofm);

   ethosu_create_simple_lut(lut_operation, &operation_lut, lut);

   operation->activation =
      ethosu_lut_activation(subgraph, &operation_lut.ifm,
                            &operation_lut.ofm, LUT8_SIZE, false);
   fill_lut(subgraph, operation, lut, LUT8_SIZE);
}

static void
ethosu_lower_lut(struct ethosu_subgraph *subgraph,
                 const struct pipe_ml_operation *poperation,
                 struct ethosu_operation *operation, double (*func)(double))
{
   uint8_t lut[LUT8_SIZE];

   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->round_mode = ETHOSU_ROUNDING_NATURAL;
   operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;

   set_feature_maps(subgraph, poperation->input_tensors[0], poperation->output_tensors[0], operation);
   operation->activation =
      ethosu_lut_activation(subgraph, &operation->ifm,
                            &operation->ofm, LUT8_SIZE, false);

   ethos_create_lut(operation, lut, func);
   fill_lut(subgraph, operation, lut, LUT8_SIZE);

   /* The LUT handles 0 point and scale, so make them equal */
   operation->ofm.zero_point = operation->ifm.zero_point;
   operation->ofm.scale = operation->ifm.scale;
}

static void
ethosu_lower_hswish(struct ethosu_subgraph *subgraph,
                    const struct pipe_ml_operation *poperation,
                    struct ethosu_operation *operation)
{
   uint8_t lut[LUT8_SIZE];

   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->round_mode = ETHOSU_ROUNDING_NATURAL;
   operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;

   set_feature_maps(subgraph, poperation->input_tensors[0], poperation->output_tensors[0], operation);
   operation->activation =
      ethosu_lut_activation(subgraph, &operation->ifm,
                            &operation->ofm, LUT8_SIZE, false);

   ethos_create_hswish_lut(operation, lut);
   fill_lut(subgraph, operation, lut, LUT8_SIZE);

   /* The LUT handles 0 point and scale, so make them equal */
   operation->ofm.zero_point = operation->ifm.zero_point;
   operation->ofm.scale = operation->ifm.scale;
}

static void
ethosu_lower_rsqrt(struct ethosu_subgraph *subgraph,
                   const struct pipe_ml_operation *poperation,
                   struct ethosu_operation *operation)
{
   uint8_t lut[LUT8_SIZE];

   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->round_mode = ETHOSU_ROUNDING_NATURAL;
   operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;

   set_feature_maps(subgraph, poperation->input_tensors[0],
                    poperation->output_tensors[0], operation);
   operation->activation =
      ethosu_lut_activation(subgraph, &operation->ifm, &operation->ofm,
                            LUT8_SIZE, false);

   ethos_create_rsqrt_lut(operation, lut);
   fill_lut(subgraph, operation, lut, LUT8_SIZE);

   /* The LUT handles 0 point and scale, so make them equal. */
   operation->ofm.zero_point = operation->ifm.zero_point;
   operation->ofm.scale = operation->ifm.scale;
}

static void
ethosu_lower_leakyrelu(struct ethosu_subgraph *subgraph,
                       const struct pipe_ml_operation *poperation,
                       struct ethosu_operation *operation)
{
   uint8_t lut[LUT8_SIZE];

   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->round_mode = ETHOSU_ROUNDING_NATURAL;
   operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;

   set_feature_maps(subgraph, poperation->input_tensors[0], poperation->output_tensors[0], operation);
   operation->activation =
      ethosu_lut_activation(subgraph, &operation->ifm,
                            &operation->ofm, LUT8_SIZE, false);

   ethos_create_leakyrelu_lut(operation, lut, poperation->leakyrelu.alpha);
   fill_lut(subgraph, operation, lut, LUT8_SIZE);

   /* The LUT handles 0 point and scale, so make them equal */
   operation->ofm.zero_point = operation->ifm.zero_point;
   operation->ofm.scale = operation->ifm.scale;
}

static void
ethosu_append_lut_pool(struct ethosu_subgraph *subgraph,
                       const struct pipe_ml_operation *poperation,
                       struct ethosu_operation *operation)
{
   struct ethosu_operation dma_operation;

   allocate_feature_maps(subgraph, operation);

   operation_set_defaults(&dma_operation);
   ethosu_lower_lut_dma(subgraph, poperation, operation, &dma_operation);
   util_dynarray_append(&subgraph->operations, dma_operation);

   ethosu_sched_operation(subgraph, operation);
   util_dynarray_append(&subgraph->operations, *operation);
}

static void
ethos_create_softmax_exp_lut(float beta, float input_scale, int32_t *lut)
{
   const int k_integer_bits = 5;
   const int k_signed_bits = 31;
   const double max_real_beta = (double)((1LL << k_signed_bits) - 1);
   const double real_beta =
      MIN2((double)beta * (double)input_scale *
              (double)(1 << (k_signed_bits - k_integer_bits)),
           max_real_beta);
   int32_t quant_shift;
   int32_t quant_scale = ethosu_quantize_scale(real_beta, &quant_shift, false);
   const int left_shift = 31 - quant_shift;
   const int diff_min = -(int)floor((double)(((1 << k_integer_bits) - 1) *
                                             (1 << (k_signed_bits - k_integer_bits))) /
                                    (double)(1U << left_shift));

   for (int x = 0; x < LUT8_SIZE; x++) {
      int input_diff = x - 255;

      if (input_diff >= diff_min) {
         int32_t input_diff_rescaled = saturating_rounding_doubling_high_mul_32(
            clamp_i64_to_i32((int64_t)input_diff * (1LL << left_shift)),
            quant_scale);
         lut[x] = fixedpoint_exp_on_negative_values_q5(input_diff_rescaled);
      } else {
         lut[x] = 0;
      }
   }
}

static void
ethosu_append_pool(struct ethosu_subgraph *subgraph,
                   enum ethosu_pooling_type type,
                   struct ethosu_feature_map *ifm,
                   struct ethosu_feature_map *ofm,
                   unsigned kernel_width,
                   enum ethosu_rounding_mode round_mode)
{
   struct ethosu_operation operation;
   struct ethosu_feature_map prepared_ifm = *ifm;
   struct ethosu_feature_map prepared_ofm = *ofm;

   operation_set_defaults(&operation);
   ethosu_prepare_feature_map(subgraph, &prepared_ifm);
   ethosu_prepare_feature_map(subgraph, &prepared_ofm);

   operation.type = ETHOSU_OPERATION_TYPE_POOLING;
   operation.pooling.type = type;
   operation.round_mode = round_mode;
   operation.kernel.width = kernel_width;
   operation.ifm = prepared_ifm;
   operation.ofm = prepared_ofm;
   ethosu_append_operation(subgraph, &operation);
}

static void
ethosu_append_resize_pool(struct ethosu_subgraph *subgraph,
                          struct ethosu_feature_map *ifm,
                          struct ethosu_feature_map *ofm,
                          unsigned kernel_height,
                          unsigned kernel_width)
{
   struct ethosu_operation operation;
   struct ethosu_feature_map prepared_ifm = *ifm;
   struct ethosu_feature_map prepared_ofm = *ofm;

   operation_set_defaults(&operation);
   ethosu_prepare_feature_map(subgraph, &prepared_ifm);
   ethosu_prepare_feature_map(subgraph, &prepared_ofm);
   operation.type = ETHOSU_OPERATION_TYPE_POOLING;
   operation.pooling.type = ETHOSU_POOLING_TYPE_AVG;
   operation.round_mode = ETHOSU_ROUNDING_DOUBLE;
   operation.kernel.height = kernel_height;
   operation.kernel.width = kernel_width;
   operation.upscale = ETHOSU_UPSCALE_NEAREST;
   operation.ifm = prepared_ifm;
   operation.ofm = prepared_ofm;
   ethosu_append_operation(subgraph, &operation);
}

static void
ethosu_append_eltwise(struct ethosu_subgraph *subgraph,
                      enum ethosu_eltwise_type type,
                      struct ethosu_feature_map *ifm,
                      struct ethosu_feature_map *ifm2,
                      struct ethosu_feature_map *ofm,
                      enum ethosu_rounding_mode round_mode,
                      bool identity_scale,
                      unsigned scale,
                      unsigned shift)
{
   struct ethosu_operation operation;
   struct ethosu_feature_map prepared_ifm = *ifm;
   struct ethosu_feature_map prepared_ifm2;
   struct ethosu_feature_map prepared_ofm = *ofm;

   operation_set_defaults(&operation);
   ethosu_prepare_feature_map(subgraph, &prepared_ifm);
   if (ifm2) {
      prepared_ifm2 = *ifm2;
      ethosu_prepare_feature_map(subgraph, &prepared_ifm2);
   }
   ethosu_prepare_feature_map(subgraph, &prepared_ofm);

   /* Broadcast a single spatial vector without extending its allocation. */
   if (!prepared_ifm.has_scalar && prepared_ifm.shape.height == 1 &&
       prepared_ifm.shape.width == 1 &&
       prepared_ifm.shape.depth == prepared_ofm.shape.depth &&
       (prepared_ofm.shape.height != 1 || prepared_ofm.shape.width != 1)) {
      prepared_ifm.shape.height = prepared_ofm.shape.height;
      prepared_ifm.shape.width = prepared_ofm.shape.width;
      prepared_ifm.stride.y = 0;
      prepared_ifm.stride.x = 0;
      prepared_ifm.tiles.height_0 = prepared_ifm.shape.height;
      prepared_ifm.tiles.height_1 = prepared_ifm.shape.height;
      prepared_ifm.tiles.width_0 = prepared_ifm.shape.width;
   }

   operation.type = ETHOSU_OPERATION_TYPE_ELTWISE;
   operation.eltwise.type = type;
   operation.round_mode = round_mode;
   operation.ifm = prepared_ifm;
   if (ifm2)
      operation.ifm2 = prepared_ifm2;
   operation.ofm = prepared_ofm;
   operation.eltwise.identity_scale = identity_scale;
   operation.eltwise.raw_scale = scale != 0;
   operation.eltwise.scale = scale;
   operation.eltwise.shift = shift;
   if (ethosu_ml_device(subgraph->base.device)->is_u65) {
      operation.ofm_scale_per_channel =
         type == ETHOSU_ELTWISE_TYPE_CLZ ||
         type == ETHOSU_ELTWISE_TYPE_SHR ||
         type == ETHOSU_ELTWISE_TYPE_SHL;
   }
   ethosu_append_operation(subgraph, &operation);
}

static void
ethosu_append_softmax_lut_sub(struct ethosu_subgraph *subgraph,
                              struct ethosu_feature_map *ifm,
                              struct ethosu_feature_map *ifm2,
                              struct ethosu_feature_map *ofm,
                              int32_t *lut)
{
   struct ethosu_operation operation;
   struct ethosu_operation dma_operation;
   struct ethosu_feature_map prepared_ifm = *ifm;
   struct ethosu_feature_map prepared_ifm2 = *ifm2;
   struct ethosu_feature_map prepared_ofm = *ofm;
   struct ethosu_feature_map lut_ifm;

   operation_set_defaults(&operation);
   ethosu_prepare_feature_map(subgraph, &prepared_ifm);
   ethosu_prepare_feature_map(subgraph, &prepared_ifm2);
   ethosu_prepare_feature_map(subgraph, &prepared_ofm);
   lut_ifm = prepared_ifm;
   lut_ifm.is_signed = true;

   operation.type = ETHOSU_OPERATION_TYPE_ELTWISE;
   operation.eltwise.type = ETHOSU_ELTWISE_TYPE_SUB;
   operation.activation =
      ethosu_lut_activation(subgraph, &lut_ifm, &prepared_ofm,
                            LUT8_SIZE * sizeof(*lut), true);
   operation.eltwise.identity_scale = true;
   operation.round_mode = ETHOSU_ROUNDING_DOUBLE;
   operation.ifm = prepared_ifm;
   operation.ifm2 = prepared_ifm2;
   operation.ofm = prepared_ofm;
   fill_lut(subgraph, &operation, lut, LUT8_SIZE * sizeof(*lut));

   operation_set_defaults(&dma_operation);
   ethosu_lower_lut_dma(subgraph, NULL, &operation, &dma_operation);
   util_dynarray_append(&subgraph->operations, dma_operation);
   ethosu_append_operation(subgraph, &operation);
}

static void
ethosu_append_reversed_sub(struct ethosu_subgraph *subgraph,
                           struct ethosu_feature_map *ifm,
                           struct ethosu_feature_map *ifm2,
                           struct ethosu_feature_map *ofm,
                           enum ethosu_rounding_mode round_mode,
                           bool identity_scale,
                           unsigned scale,
                           unsigned shift)
{
   struct ethosu_operation operation;
   struct ethosu_feature_map prepared_ifm = *ifm;
   struct ethosu_feature_map prepared_ifm2 = *ifm2;
   struct ethosu_feature_map prepared_ofm = *ofm;

   operation_set_defaults(&operation);
   ethosu_prepare_feature_map(subgraph, &prepared_ifm);
   ethosu_prepare_feature_map(subgraph, &prepared_ifm2);
   ethosu_prepare_feature_map(subgraph, &prepared_ofm);

   operation.type = ETHOSU_OPERATION_TYPE_ELTWISE;
   operation.eltwise.type = ETHOSU_ELTWISE_TYPE_SUB;
   operation.round_mode = round_mode;

   if (!ethosu_ml_device(subgraph->base.device)->is_u65 &&
       prepared_ifm2.has_scalar) {
      operation.ifm = prepared_ifm2;
      operation.ifm2 = prepared_ifm;
   } else {
      operation.eltwise.ifm_reversed = true;
      operation.ifm = prepared_ifm;
      operation.ifm2 = prepared_ifm2;
   }

   operation.ofm = prepared_ofm;
   operation.eltwise.identity_scale = identity_scale;
   operation.eltwise.raw_scale = scale != 0;
   operation.eltwise.scale = scale;
   operation.eltwise.shift = shift;
   ethosu_append_operation(subgraph, &operation);
}

static void
ethosu_lower_softmax(struct ethosu_subgraph *subgraph,
                     const struct pipe_ml_operation *poperation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];
   struct pipe_tensor *output = poperation->output_tensors[0];
   unsigned spatial = input->dims[1] * input->dims[2];
   unsigned depth = input->dims[3];
   struct ethosu_block pool_shape = {depth, spatial, 1};
   struct ethosu_block pool_reduced_shape = {1, spatial, 1};
   struct ethosu_block vector_shape = {spatial, 1, depth};
   struct ethosu_block reduced_shape = {spatial, 1, 1};
   struct ethosu_tensor *input_tensor =
      ethosu_find_tensor(subgraph, input->index);
   struct ethosu_tensor *output_tensor =
      ethosu_find_tensor(subgraph, output->index);
   struct ethosu_tensor *max_tensor =
      ethosu_add_internal_tensor(subgraph, pool_reduced_shape, 1);
   struct ethosu_tensor *exp_tensor =
      ethosu_add_internal_tensor(subgraph, vector_shape, 4);
   struct ethosu_tensor *rescaled_exp_tensor =
      ethosu_add_internal_tensor(subgraph, vector_shape, 4);
   struct ethosu_tensor *sum_tensor =
      ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
   struct ethosu_tensor *headroom_plus_one_tensor =
      ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
   struct ethosu_tensor *right_shift_tensor =
      ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
   struct ethosu_tensor *headroom_tensor =
      ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
   struct ethosu_tensor *half_denominator_tensor =
      ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
   struct ethosu_tensor *nr_x_tensor =
      ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
   struct ethosu_tensor *half_denominator_times_x_tensor[3];
   struct ethosu_tensor *one_minus_half_denominator_times_x_tensor[3];
   struct ethosu_tensor *to_rescale_tensor[3];
   struct ethosu_tensor *to_add_tensor[3];
   struct ethosu_tensor *nr_x_next_tensor[3];
   struct ethosu_tensor *scaled_exp_tensor =
      ethosu_add_internal_tensor(subgraph, vector_shape, 4);
   struct ethosu_feature_map input_pool_fm = {0}, input_fm = {0};
   struct ethosu_feature_map max_fm = {0}, exp_lut_fm = {0}, exp_fm = {0};
   struct ethosu_feature_map rescaled_exp_fm = {0}, sum_fm = {0};
   struct ethosu_feature_map headroom_plus_one_fm = {0};
   struct ethosu_feature_map right_shift_fm = {0}, headroom_fm = {0};
   struct ethosu_feature_map half_denominator_fm = {0}, nr_x_fm = {0};
   struct ethosu_feature_map nr_x_mul_fm = {0};
   struct ethosu_feature_map output_fm = {0}, scaled_exp_fm = {0};
   struct ethosu_feature_map constant_fm = {0};
   int32_t lut[LUT8_SIZE];

   assert(depth > 1 && depth <= SOFTMAX_MAX_DEPTH);
   assert(input->type_size == 1 && output->type_size == 1);

   for (unsigned i = 0; i < 3; i++) {
      half_denominator_times_x_tensor[i] =
         ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
      one_minus_half_denominator_times_x_tensor[i] =
         ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
      to_rescale_tensor[i] =
         ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
      to_add_tensor[i] =
         ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
      nr_x_next_tensor[i] =
         ethosu_add_internal_tensor(subgraph, reduced_shape, 4);
   }

   max_tensor->layout = ETHOSU_LAYOUT_NHCWB16;
   exp_tensor->layout = ETHOSU_LAYOUT_NHCWB16;
   sum_tensor->layout = ETHOSU_LAYOUT_NHCWB16;
   headroom_plus_one_tensor->layout = ETHOSU_LAYOUT_NHCWB16;
   right_shift_tensor->layout = ETHOSU_LAYOUT_NHCWB16;
   headroom_tensor->layout = ETHOSU_LAYOUT_NHCWB16;
   half_denominator_tensor->layout = ETHOSU_LAYOUT_NHCWB16;
   nr_x_tensor->layout = ETHOSU_LAYOUT_NHCWB16;
   scaled_exp_tensor->layout = ETHOSU_LAYOUT_NHCWB16;

   for (unsigned i = 0; i < 3; i++) {
      half_denominator_times_x_tensor[i]->layout = ETHOSU_LAYOUT_NHCWB16;
      one_minus_half_denominator_times_x_tensor[i]->layout = ETHOSU_LAYOUT_NHCWB16;
      to_rescale_tensor[i]->layout = ETHOSU_LAYOUT_NHCWB16;
      to_add_tensor[i]->layout = ETHOSU_LAYOUT_NHCWB16;
      nr_x_next_tensor[i]->layout = ETHOSU_LAYOUT_NHCWB16;
   }

   set_internal_feature_map(input_tensor, pool_shape, input->scale,
                            input->zero_point, input->is_signed, &input_pool_fm);
   set_internal_feature_map(input_tensor, vector_shape, input->scale,
                            input->zero_point, input->is_signed, &input_fm);
   set_internal_feature_map(exp_tensor, vector_shape, 1.0f, 127, true, &exp_lut_fm);
   set_internal_feature_map(exp_tensor, vector_shape, 1.0f, 0, true, &exp_fm);
   set_internal_feature_map(rescaled_exp_tensor, vector_shape, 1.0f, 0, true, &rescaled_exp_fm);
   set_internal_feature_map(sum_tensor, reduced_shape, 1.0f, 0, true, &sum_fm);
   set_internal_feature_map(headroom_plus_one_tensor, reduced_shape, 1.0f, 0, true, &headroom_plus_one_fm);
   set_internal_feature_map(right_shift_tensor, reduced_shape, 1.0f, 0, true, &right_shift_fm);
   set_internal_feature_map(headroom_tensor, reduced_shape, 1.0f, 0, true, &headroom_fm);
   set_internal_feature_map(half_denominator_tensor, reduced_shape, 1.0f, 0, true, &half_denominator_fm);
   set_internal_feature_map(nr_x_tensor, reduced_shape, 1.0f, 0, true, &nr_x_fm);
   set_internal_feature_map(scaled_exp_tensor, vector_shape, 1.0f, 0, true, &scaled_exp_fm);
   set_internal_feature_map(output_tensor, vector_shape, output->scale,
                            output->zero_point, output->is_signed, &output_fm);

   nr_x_mul_fm = nr_x_fm;
   set_u85_chained_feature_map(subgraph, &headroom_fm, 0);
   set_u85_chained_feature_map(subgraph, &nr_x_mul_fm, 0);
   set_u85_chained_feature_map(subgraph, &scaled_exp_fm, 0);

   set_internal_feature_map(max_tensor, pool_reduced_shape, input->scale,
                            input->zero_point, input->is_signed, &max_fm);
   ethosu_append_pool(subgraph, ETHOSU_POOLING_TYPE_MAX, &input_pool_fm,
                      &max_fm, depth, ETHOSU_ROUNDING_DOUBLE);
   set_internal_feature_map(max_tensor, reduced_shape, input->scale,
                            input->zero_point, input->is_signed, &max_fm);

   ethos_create_softmax_exp_lut(poperation->softmax.beta, input->scale, lut);
   ethosu_append_softmax_lut_sub(subgraph, &input_fm, &max_fm, &exp_lut_fm, lut);

   set_constant_feature_map(subgraph, 12, &constant_fm);
   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_SHR, &exp_fm,
                         &constant_fm, &rescaled_exp_fm,
                         ETHOSU_ROUNDING_NATURAL, true, 0, 0);

   ethosu_append_pool(subgraph, ETHOSU_POOLING_TYPE_REDUCE_SUM, &rescaled_exp_fm,
                      &sum_fm, 1, ETHOSU_ROUNDING_NATURAL);

   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_CLZ, &sum_fm, NULL,
                         &headroom_plus_one_fm, ETHOSU_ROUNDING_DOUBLE,
                         true, 0, 0);

   set_constant_feature_map(subgraph, 12 + 31 - 8, &constant_fm);
   ethosu_append_reversed_sub(subgraph, &headroom_plus_one_fm, &constant_fm,
                              &right_shift_fm, ETHOSU_ROUNDING_DOUBLE,
                              true, 0, 0);

   set_constant_feature_map(subgraph, 1, &constant_fm);
   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_SUB,
                         &headroom_plus_one_fm, &constant_fm, &headroom_fm,
                         ETHOSU_ROUNDING_DOUBLE, true, 0, 0);

   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_SHL, &sum_fm,
                         &headroom_fm, &half_denominator_fm,
                         ETHOSU_ROUNDING_DOUBLE, true, 0, 0);

   set_constant_feature_map(subgraph, -((int32_t)((32ULL << 29U) / 17U)),
                            &constant_fm);
   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_MUL,
                         &half_denominator_fm, &constant_fm, &nr_x_mul_fm,
                         ETHOSU_ROUNDING_DOUBLE, false, 0x40000000, 31);

   set_constant_feature_map(subgraph, (int32_t)((48ULL << 29U) / 17U),
                            &constant_fm);
   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_ADD, &nr_x_mul_fm,
                         &constant_fm, &nr_x_fm, ETHOSU_ROUNDING_DOUBLE,
                         true, 0, 0);

   for (unsigned i = 0; i < 3; i++) {
      struct ethosu_feature_map half_denominator_times_x_fm = {0};
      struct ethosu_feature_map one_minus_half_denominator_times_x_fm = {0};
      struct ethosu_feature_map to_rescale_fm = {0};
      struct ethosu_feature_map to_add_fm = {0};
      struct ethosu_feature_map nr_x_next_fm = {0};

      set_internal_feature_map(half_denominator_times_x_tensor[i], reduced_shape,
                               1.0f, 0, true, &half_denominator_times_x_fm);
      set_internal_feature_map(one_minus_half_denominator_times_x_tensor[i], reduced_shape,
                               1.0f, 0, true, &one_minus_half_denominator_times_x_fm);
      set_internal_feature_map(to_rescale_tensor[i], reduced_shape,
                               1.0f, 0, true, &to_rescale_fm);
      set_internal_feature_map(to_add_tensor[i], reduced_shape,
                               1.0f, 0, true, &to_add_fm);
      set_internal_feature_map(nr_x_next_tensor[i], reduced_shape,
                               1.0f, 0, true, &nr_x_next_fm);

      set_u85_chained_feature_map(subgraph,
                                  &half_denominator_times_x_fm, 0);
      set_u85_chained_feature_map(subgraph,
                                  &one_minus_half_denominator_times_x_fm, 1);
      set_u85_chained_feature_map(subgraph, &to_rescale_fm, 2);

      ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_MUL, &nr_x_fm,
                            &half_denominator_fm, &half_denominator_times_x_fm,
                            ETHOSU_ROUNDING_DOUBLE, false, 0x40000000, 31);

      set_constant_feature_map(subgraph, 1 << 29, &constant_fm);
      ethosu_append_reversed_sub(subgraph, &half_denominator_times_x_fm,
                                 &constant_fm,
                                 &one_minus_half_denominator_times_x_fm,
                                 ETHOSU_ROUNDING_DOUBLE, true, 0, 0);

      ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_MUL, &nr_x_fm,
                            &one_minus_half_denominator_times_x_fm,
                            &to_rescale_fm, ETHOSU_ROUNDING_DOUBLE,
                            false, 0x40000000, 31);

      set_constant_feature_map(subgraph, 4, &constant_fm);
      ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_MUL, &to_rescale_fm,
                            &constant_fm, &to_add_fm,
                            ETHOSU_ROUNDING_DOUBLE, true, 0, 0);

      ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_ADD, &nr_x_fm,
                            &to_add_fm, &nr_x_next_fm,
                            ETHOSU_ROUNDING_DOUBLE, true, 0, 0);

      nr_x_fm = nr_x_next_fm;
   }

   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_MUL, &exp_fm, &nr_x_fm,
                         &scaled_exp_fm, ETHOSU_ROUNDING_DOUBLE,
                         false, 0x40000000, 30);

   ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_SHR, &scaled_exp_fm,
                         &right_shift_fm, &output_fm,
                         ETHOSU_ROUNDING_NATURAL, true, 0, 0);
}

static void
ethosu_lower_quantize(struct ethosu_subgraph *subgraph,
                      const struct pipe_ml_operation *poperation,
                      struct ethosu_operation *operation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];

   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->round_mode = ETHOSU_ROUNDING_DOUBLE;
   operation->pooling.nop = true;

   if (ethosu_ml_device(subgraph->base.device)->is_u65)
      operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;
   else
      operation->pooling.type = ETHOSU_POOLING_TYPE_SUM;

   set_feature_maps(subgraph, input, poperation->output_tensors[0], operation);

   if (input->data) {
      unsigned size = input->dims[1] * input->dims[2] * input->dims[3] *
         input->type_size;

      operation->ifm.region = COEFS_REGION;
      operation->ifm.tiles.addresses[0] =
         ethosu_add_constant(subgraph, input->data, size);
      operation->ifm.tiles.height_0 = operation->ifm.shape.height;
      operation->ifm.tiles.height_1 = operation->ifm.shape.height;
      operation->ifm.tiles.width_0 = operation->ifm.shape.width;
   }

   allocate_feature_maps(subgraph, operation);
   ethosu_sched_operation(subgraph, operation);
}

static void
ethosu_lower_reshape(struct ethosu_subgraph *subgraph,
                     const struct pipe_ml_operation *poperation,
                     struct ethosu_operation *operation)
{
   operation->type = ETHOSU_OPERATION_TYPE_NONE;

   set_feature_maps(subgraph, poperation->input_tensors[0], poperation->output_tensors[0], operation);
   operation->ifm.tiles.addresses[0] = ethosu_allocate_feature_map(subgraph, &operation->ifm);
   operation->ofm.tiles.addresses[0] = operation->ifm.tiles.addresses[0];

   operation->ofm.tensor->offset = operation->ifm.tensor->offset;
   operation->ofm.tensor->size = operation->ifm.tensor->size;
   operation->ofm.tensor->layout = operation->ifm.tensor->layout;
}

static void
ethosu_lower_split(struct ethosu_subgraph *subgraph,
                   const struct pipe_ml_operation *poperation,
                   unsigned output_idx,
                   struct ethosu_operation *operation)
{
   unsigned input_span;
   unsigned address_offset = 0;

   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->round_mode = ETHOSU_ROUNDING_NATURAL;
   operation->pooling.nop = true;
   if (ethosu_ml_device(subgraph->base.device)->is_u65) {
      operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;
   } else {
      operation->pooling.type = ETHOSU_POOLING_TYPE_SUM;
   }

   set_feature_maps(subgraph, poperation->input_tensors[0],
                    poperation->output_tensors[output_idx], operation);

   /* SPLIT copies quantized values without requantizing them. */
   operation->ofm.scale = operation->ifm.scale;
   operation->ofm.zero_point = operation->ifm.zero_point;

   input_span = ethosu_feature_map_span(&operation->ifm);
   operation->ifm.tensor->required_size =
      MAX2(operation->ifm.tensor->required_size, input_span);

   allocate_feature_maps(subgraph, operation);
   operation->ifm.shape = operation->ofm.shape;

   unsigned offset =
      ethosu_concat_split_offset(poperation->output_tensors, output_idx,
                                 poperation->split.axis);
   switch (poperation->split.axis) {
   case 1:
      address_offset = offset * operation->ifm.stride.y;
      break;
   case 2:
      address_offset = offset * operation->ifm.stride.x;
      break;
   case 3:
      if (operation->ifm.tensor->layout == ETHOSU_LAYOUT_NHWC) {
         address_offset = offset * operation->ifm.stride.c;
      } else if (operation->ifm.tensor->layout == ETHOSU_LAYOUT_NHCWB16) {
         unsigned elem_size = 1 << operation->ifm.precision;

         address_offset = (offset / 16) * operation->ifm.stride.c +
                          (offset % 16) * elem_size;
      } else {
         UNREACHABLE("Unsupported layout");
      }
      break;
   default:
      UNREACHABLE("Unsupported split axis");
   }

   operation->ifm.tiles.addresses[0] += address_offset;
   ethosu_sched_operation(subgraph, operation);
}

static void
ethosu_lower_concatenation(struct ethosu_subgraph *subgraph,
                           const struct pipe_ml_operation *poperation,
                           unsigned input_idx,
                           struct ethosu_operation *operation)
{
   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->round_mode = ETHOSU_ROUNDING_NATURAL;

   if (ethosu_ml_device(subgraph->base.device)->is_u65) {
      operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;
   } else
      operation->pooling.type = ETHOSU_POOLING_TYPE_SUM;
   operation->pooling.nop = true;

   set_feature_maps(subgraph, poperation->input_tensors[input_idx], poperation->output_tensors[0], operation);
   operation->ofm.tensor->required_size =
      MAX2(operation->ofm.tensor->required_size,
           ethosu_feature_map_span(&operation->ofm));

   allocate_feature_maps(subgraph, operation);

   operation->ofm.shape = operation->ifm.shape;
   unsigned offset =
      ethosu_concat_split_offset(poperation->input_tensors, input_idx,
                                 poperation->conc.axis);
   switch (poperation->conc.axis) {
   case 1:
      operation->ofm.tiles.addresses[0] += offset * operation->ofm.stride.y;
      break;
   case 2:
      operation->ofm.tiles.addresses[0] += offset * operation->ofm.stride.x;
      break;
   case 3:
      if (operation->ofm.tensor->layout == ETHOSU_LAYOUT_NHWC) {
         operation->ofm.tiles.addresses[0] +=
            offset * operation->ofm.stride.c;
      } else if (operation->ofm.tensor->layout == ETHOSU_LAYOUT_NHCWB16) {
         unsigned elem_size = 1 << operation->ofm.precision;

         operation->ofm.tiles.addresses[0] +=
            (offset / 16) * operation->ofm.stride.c +
            (offset % 16) * elem_size;
      } else {
         assert(0 && "Unsupported layout");
      }
      break;
   default:
      assert(0 && "Unsupported layout");
   }

   ethosu_sched_operation(subgraph, operation);
}

static void
ethosu_lower_unpack(struct ethosu_subgraph *subgraph,
                    const struct pipe_ml_operation *poperation,
                    unsigned output_idx,
                    struct ethosu_operation *operation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];
   unsigned input_span;

   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->round_mode = ETHOSU_ROUNDING_NATURAL;
   operation->pooling.type = ethosu_ml_device(subgraph->base.device)->is_u65 ? ETHOSU_POOLING_TYPE_AVG : ETHOSU_POOLING_TYPE_SUM;
   operation->pooling.nop = true;

   set_feature_maps(subgraph, input, poperation->output_tensors[output_idx],
                    operation);
   input_span = ethosu_feature_map_span(&operation->ifm);
   operation->ifm.tensor->required_size =
      MAX2(operation->ifm.tensor->required_size, input_span);

   allocate_feature_maps(subgraph, operation);
   operation->ifm.shape = operation->ofm.shape;

   switch (poperation->split.axis) {
   case 1:
      operation->ifm.tiles.addresses[0] +=
         output_idx * operation->ifm.shape.height * operation->ifm.stride.y;
      break;
   case 2:
      operation->ifm.tiles.addresses[0] +=
         output_idx * operation->ifm.shape.width * operation->ifm.stride.x;
      break;
   case 3:
      if (operation->ifm.tensor->layout == ETHOSU_LAYOUT_NHWC) {
         operation->ifm.tiles.addresses[0] +=
            output_idx * operation->ifm.shape.depth * operation->ifm.stride.c;
      } else if (operation->ifm.tensor->layout == ETHOSU_LAYOUT_NHCWB16) {
         unsigned depth = output_idx * operation->ifm.shape.depth;
         unsigned elem_size = 1 << operation->ifm.precision;

         operation->ifm.tiles.addresses[0] +=
            (depth / 16) * operation->ifm.stride.c +
            (depth % 16) * elem_size;
      } else {
         assert(0 && "Unsupported layout");
      }
      break;
   default:
      assert(0 && "Unsupported axis");
   }

   ethosu_sched_operation(subgraph, operation);
}

static void
ethosu_lower_resize(struct ethosu_subgraph *subgraph,
                    const struct pipe_ml_operation *poperation,
                    struct ethosu_operation *operation)
{
   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;

   set_feature_maps(subgraph, poperation->input_tensors[0], poperation->output_tensors[0], operation);

   operation->upscale = ETHOSU_UPSCALE_NEAREST;

   allocate_feature_maps(subgraph, operation);
   ethosu_sched_operation(subgraph, operation);
}

static void
ethosu_lower_resize_bilinear(struct ethosu_subgraph *subgraph,
                             const struct pipe_ml_operation *poperation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];
   struct pipe_tensor *output = poperation->output_tensors[0];
   struct ethosu_feature_map ifm = {0}, ofm = {0};
   unsigned scale;
   unsigned kernel_scale;

   set_feature_map(subgraph, input, &ifm);
   set_feature_map(subgraph, output, &ofm);

   if (input->dims[1] == 1 && input->dims[2] == 1) {
      struct pipe_tensor scalar = {
         .dims = {1, 1, 1, 1},
         .scale = input->scale,
         .zero_point = input->zero_point,
         .is_signed = input->is_signed,
         .type_size = input->type_size,
      };
      uint8_t value = input->zero_point;
      struct ethosu_feature_map zero = {0};

      scalar.data = &value;
      set_constant_tensor_feature_map(subgraph, &scalar, &zero);
      ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_ADD, &ifm, &zero,
                            &ofm, ETHOSU_ROUNDING_DOUBLE, false, 0, 0);
      return;
   }

   scale = (output->dims[1] - 1) / (input->dims[1] - 1);
   kernel_scale = scale;
   while (scale > 2) {
      struct ethosu_block shape = {
         ifm.shape.width * 2,
         ifm.shape.height * 2,
         ifm.shape.depth,
      };
      struct ethosu_tensor *tensor =
         ethosu_add_internal_tensor(subgraph, shape, input->type_size);
      struct ethosu_feature_map intermediate = {0};

      set_internal_feature_map(tensor, shape, input->scale, input->zero_point,
                               input->is_signed, &intermediate);
      ethosu_append_resize_pool(subgraph, &ifm, &intermediate, 1, 1);
      ifm = intermediate;
      scale /= 2;
   }

   ethosu_append_resize_pool(subgraph, &ifm, &ofm, kernel_scale,
                             kernel_scale);
}

static void
ethosu_swap_transpose_axes(struct ethosu_feature_map *ifm,
                           const struct ethosu_feature_map *ofm,
                           unsigned axis)
{
   unsigned *strides[] = {&ifm->stride.y, &ifm->stride.x, &ifm->stride.c};
   unsigned stride;

   assert(axis < 2);

   ifm->shape = ofm->shape;
   stride = *strides[axis];
   *strides[axis] = *strides[axis + 1];
   *strides[axis + 1] = stride;
}

static struct ethosu_block
ethosu_transposed_shape(struct ethosu_block shape, unsigned axis)
{
   unsigned *dimensions[] = {&shape.height, &shape.width, &shape.depth};
   unsigned dimension;

   assert(axis < 2);

   dimension = *dimensions[axis];
   *dimensions[axis] = *dimensions[axis + 1];
   *dimensions[axis + 1] = dimension;

   return shape;
}

static void
ethosu_append_width_depth_transpose(struct ethosu_subgraph *subgraph,
                                    const struct ethosu_feature_map *ifm,
                                    const struct ethosu_feature_map *ofm)
{
   assert(ifm->shape.height == ofm->shape.height);
   assert(ifm->shape.width == ofm->shape.depth);
   assert(ifm->shape.depth == ofm->shape.width);

   /* Each row is emitted as a smaller feature map, but shares one OFM. */
   ofm->tensor->required_size = MAX2(ofm->tensor->required_size,
                                     ethosu_feature_map_span(ofm));
   ifm->tensor->required_size = MAX2(ifm->tensor->required_size,
                                     ethosu_feature_map_span(ifm));

   for (unsigned h = 0; h < ifm->shape.height; h++) {
      struct ethosu_feature_map source = *ifm;
      struct ethosu_feature_map target = *ofm;

      /* U65 ignores non-unit NHWC channel strides.  Transpose each H row
       * as a W-by-C matrix, using only H and W strides. */
      source.shape = (struct ethosu_block){
         .width = ifm->shape.depth,
         .height = ifm->shape.width,
         .depth = 1,
      };
      source.stride.y = ifm->stride.x;
      source.stride.x = ifm->stride.c;
      source.stride.c = ifm->stride.c;
      source.tiles.addresses[0] += h * ifm->stride.y;

      target.shape = source.shape;
      target.stride.y = ofm->stride.c;
      target.stride.x = ofm->stride.x;
      target.stride.c = ofm->stride.c;
      target.tiles.addresses[0] += h * ofm->stride.y;

      ethosu_append_pool_nop(subgraph, &source, &target);
   }
}

static void
ethosu_append_height_depth_transpose(struct ethosu_subgraph *subgraph,
                                     const struct ethosu_feature_map *ifm,
                                     const struct ethosu_feature_map *ofm)
{
   assert(ifm->shape.height == ofm->shape.depth);
   assert(ifm->shape.width == ofm->shape.width);
   assert(ifm->shape.depth == ofm->shape.height);

   /* Each column is emitted as a smaller feature map, but shares one OFM. */
   ofm->tensor->required_size = MAX2(ofm->tensor->required_size,
                                     ethosu_feature_map_span(ofm));
   ifm->tensor->required_size = MAX2(ifm->tensor->required_size,
                                     ethosu_feature_map_span(ifm));

   for (unsigned w = 0; w < ifm->shape.width; w++) {
      struct ethosu_feature_map source = *ifm;
      struct ethosu_feature_map target = *ofm;

      /* U65 ignores non-unit NHWC channel strides. Transpose each W slice
       * as an H-by-C matrix, using only H and W strides. */
      source.shape = (struct ethosu_block){
         .width = ifm->shape.depth,
         .height = ifm->shape.height,
         .depth = 1,
      };
      source.stride.y = ifm->stride.y;
      source.stride.x = ifm->stride.c;
      source.stride.c = ifm->stride.c;
      source.tiles.addresses[0] += w * ifm->stride.x;

      target.shape = source.shape;
      target.stride.y = ofm->stride.c;
      target.stride.x = ofm->stride.y;
      target.stride.c = ofm->stride.c;
      target.tiles.addresses[0] += w * ofm->stride.x;

      ethosu_append_pool_nop(subgraph, &source, &target);
   }
}

static bool
ethosu_transpose_complete(const unsigned order[3], const unsigned target[3])
{
   return order[0] == target[0] && order[1] == target[1] &&
          order[2] == target[2];
}

static void
ethosu_lower_transpose(struct ethosu_subgraph *subgraph,
                       const struct pipe_ml_operation *poperation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];
   struct pipe_tensor *output = poperation->output_tensors[0];
   struct ethosu_feature_map ifm = {0};
   struct ethosu_feature_map ofm = {0};
   unsigned order[] = {1, 2, 3};
   const unsigned target[] = {
      poperation->transpose.perm[1],
      poperation->transpose.perm[2],
      poperation->transpose.perm[3],
   };

   set_feature_map(subgraph, input, &ifm);

   if (ethosu_transpose_complete(order, target)) {
      set_feature_map(subgraph, output, &ofm);
      ethosu_append_pool_nop(subgraph, &ifm, &ofm);
      return;
   }

   if (target[0] == 3 && target[1] == 1 && target[2] == 2) {
      struct ethosu_block shape = ifm.shape;
      struct ethosu_tensor *tensor;

      SWAP(shape.height, shape.depth);
      tensor = ethosu_add_internal_tensor(subgraph, shape, input->type_size);
      set_internal_feature_map(tensor, shape, input->scale, input->zero_point,
                               input->is_signed, &ofm);
      ethosu_append_height_depth_transpose(subgraph, &ifm, &ofm);

      ifm = ofm;
      set_feature_map(subgraph, output, &ofm);
      ethosu_append_width_depth_transpose(subgraph, &ifm, &ofm);
      return;
   }

   for (unsigned dst_axis = 0; dst_axis < ARRAY_SIZE(order); dst_axis++) {
      unsigned src_axis = dst_axis;

      while (order[src_axis] != target[dst_axis])
         src_axis++;

      while (src_axis > dst_axis) {
         struct ethosu_block shape =
            ethosu_transposed_shape(ifm.shape, src_axis - 1);
         struct ethosu_tensor *tensor;

         order[src_axis] = order[src_axis - 1];
         order[src_axis - 1] = target[dst_axis];

         if (ethosu_transpose_complete(order, target)) {
            set_feature_map(subgraph, output, &ofm);
         } else {
            tensor = ethosu_add_internal_tensor(subgraph, shape,
                                                input->type_size);
            set_internal_feature_map(tensor, shape, input->scale,
                                     input->zero_point, input->is_signed,
                                     &ofm);
         }

         if (src_axis - 1 == 1) {
            ethosu_append_width_depth_transpose(subgraph, &ifm, &ofm);
         } else {
            struct ethosu_feature_map transpose_ifm = ifm;

            ethosu_swap_transpose_axes(&transpose_ifm, &ofm, src_axis - 1);
            transpose_ifm.tensor->required_size =
               MAX2(transpose_ifm.tensor->required_size,
                    ethosu_feature_map_span(&transpose_ifm));
            ethosu_append_pool_nop(subgraph, &transpose_ifm, &ofm);
         }
         ifm = ofm;
         src_axis--;
      }
   }
}

static void
ethosu_lower_batch_matmul(struct ethosu_subgraph *subgraph,
                          const struct pipe_ml_operation *poperation)
{
   struct pipe_tensor *input = poperation->input_tensors[0];
   struct pipe_tensor *input2 = poperation->input_tensors[1];
   struct pipe_tensor *output = poperation->output_tensors[0];
   unsigned batches = input->dims[1];
   unsigned rows = input->dims[2];
   unsigned depth = input->dims[3];
   unsigned columns = output->dims[3];
   struct ethosu_block product_shape = {rows, batches, depth};
   struct ethosu_tensor *product_tensor;
   struct ethosu_tensor *input2_transposed_tensor;
   struct ethosu_feature_map input_fm = {0};
   struct ethosu_feature_map input2_fm = {0};
   struct ethosu_feature_map input2_transposed_fm = {0};
   struct ethosu_feature_map product_fm = {0};
   struct ethosu_feature_map output_fm = {0};
   struct ethosu_feature_map output_full_fm = {0};
   float product_scale = input->scale * input2->scale;

   product_tensor = ethosu_add_internal_tensor(subgraph, product_shape, 4);

   set_feature_map(subgraph, input, &input_fm);
   set_feature_map(subgraph, input2, &input2_fm);
   /* Allocate the complete tensor before narrowing this to one column. */
   input2_fm.tensor->required_size =
      MAX2(input2_fm.tensor->required_size,
           ethosu_feature_map_span(&input2_fm));

   if (!poperation->batch_matmul.adj_y) {
      input2_transposed_tensor =
         ethosu_add_internal_tensor(subgraph,
                                    (struct ethosu_block){columns, batches,
                                                          depth},
                                    input2->type_size);
      set_internal_feature_map(input2_transposed_tensor,
                               (struct ethosu_block){columns, batches,
                                                     depth},
                               input2->scale,
                               input2->zero_point, input2->is_signed,
                               &input2_transposed_fm);
      input2_transposed_fm.tensor->required_size =
         MAX2(input2_transposed_fm.tensor->required_size,
              ethosu_feature_map_span(&input2_transposed_fm));

      for (unsigned column = 0; column < columns; column++) {
         struct ethosu_feature_map source_fm = input2_fm;
         struct ethosu_feature_map target_fm = input2_transposed_fm;

         /* U65 ignores non-unit IFM2 channel strides for elementwise MUL. */
         source_fm.shape = (struct ethosu_block){depth, batches, 1};
         source_fm.stride.y = depth * columns * input2->type_size;
         source_fm.stride.x = columns * input2->type_size;
         source_fm.stride.c = input2->type_size;
         source_fm.tiles.addresses[0] = column * input2->type_size;
         target_fm.shape = source_fm.shape;
         target_fm.stride.y = columns * depth * input2->type_size;
         target_fm.stride.x = input2->type_size;
         target_fm.stride.c = input2->type_size;
         target_fm.tiles.addresses[0] =
            column * depth * input2->type_size;

         ethosu_append_pool_nop(subgraph, &source_fm, &target_fm);
      }

      input2_fm = input2_transposed_fm;
   }

   /* Select one right-hand matrix column with a strided feature map. */
   input2_fm.shape.height = batches;
   input2_fm.shape.width = 1;
   input2_fm.shape.depth = depth;
   input2_fm.stride.y = columns * depth * input2->type_size;
   input2_fm.stride.x = input2->type_size;
   input2_fm.stride.c = input2->type_size;
   input2_fm.tiles.height_0 = input2_fm.shape.height;
   input2_fm.tiles.height_1 = input2_fm.shape.height;
   input2_fm.tiles.width_0 = input2_fm.shape.width;

   set_internal_feature_map(product_tensor, product_shape, product_scale, 0,
                            true, &product_fm);
   /* An odd int32 depth gets an even-depth stride, as Vela lays it out. */
   product_fm.stride.x = ALIGN_POT(depth, 2) * product_tensor->type_size;
   product_fm.stride.y = rows * product_fm.stride.x;
   set_feature_map(subgraph, output, &output_full_fm);
   output_full_fm.tensor->required_size =
      MAX2(output_full_fm.tensor->required_size,
           ethosu_feature_map_span(&output_full_fm));

   for (unsigned column = 0; column < columns; column++) {
      set_feature_map(subgraph, output, &output_fm);
      output_fm.shape = (struct ethosu_block){rows, batches, 1};
      output_fm.stride.y = rows * columns * output->type_size;
      output_fm.stride.x = columns * output->type_size;
      output_fm.stride.c = output->type_size;
      output_fm.tiles.addresses[0] = column * output->type_size;
      output_fm.tiles.height_0 = output_fm.shape.height;
      output_fm.tiles.height_1 = output_fm.shape.height;
      output_fm.tiles.width_0 = output_fm.shape.width;

      ethosu_append_eltwise(subgraph, ETHOSU_ELTWISE_TYPE_MUL, &input_fm,
                            &input2_fm, &product_fm,
                            ETHOSU_ROUNDING_DOUBLE, false, 1, 0);
      ethosu_append_pool(subgraph, ETHOSU_POOLING_TYPE_REDUCE_SUM,
                         &product_fm, &output_fm, 1,
                         ETHOSU_ROUNDING_DOUBLE);

      input2_fm.tiles.addresses[0] += depth * input2->type_size;
   }
}

static void
ethosu_lower_strided_slice(struct ethosu_subgraph *subgraph,
                           const struct pipe_ml_operation *poperation,
                           struct ethosu_operation *operation)
{
   operation->type = ETHOSU_OPERATION_TYPE_POOLING;
   operation->pooling.type = ETHOSU_POOLING_TYPE_AVG;
   operation->round_mode = ETHOSU_ROUNDING_NATURAL;

   set_feature_maps(subgraph, poperation->input_tensors[0], poperation->output_tensors[0], operation);
   unsigned input_span = ethosu_feature_map_span(&operation->ifm);
   operation->ifm.tensor->required_size =
      MAX2(operation->ifm.tensor->required_size, input_span);

   allocate_feature_maps(subgraph, operation);
   operation->ifm.shape = operation->ofm.shape;

   unsigned rank = poperation->input_tensors[1]->dims[3];
   unsigned address_offset = 0;

   assert(rank <= 4);
   for (unsigned i = 0; i < rank; i++) {
      switch (i + 4 - rank) {
      case 0:
         address_offset += poperation->slice.begin[i] * input_span;
         break;
      case 1:
         address_offset += poperation->slice.begin[i] * operation->ifm.stride.y;
         break;
      case 2:
         address_offset += poperation->slice.begin[i] * operation->ifm.stride.x;
         break;
      case 3:
         address_offset += poperation->slice.begin[i] * operation->ifm.stride.c;
         break;
      }
   }

   operation->ifm.tiles.addresses[0] += address_offset;

   ethosu_sched_operation(subgraph, operation);
}

static bool
is_sub_shape(struct pipe_tensor *sub, struct pipe_tensor *super)
{
   for (int i = 1; i < 4; i++) {
      if (sub->dims[i] > super->dims[i])
         return false;
   }
   return true;
}

static void
ethosu_lower_eltwise(struct ethosu_subgraph *subgraph,
                     const struct pipe_ml_operation *poperation,
                     struct ethosu_operation *operation)
{
   operation->type = ETHOSU_OPERATION_TYPE_ELTWISE;
   int ifm_idx = 0;
   int ifm2_idx = 1;
   bool is_u65 = ethosu_ml_device(subgraph->base.device)->is_u65;
   bool is_subtract = poperation->type == PIPE_ML_OPERATION_TYPE_SUBTRACT;
   bool u85_scalar_lhs_sub =
      is_subtract && !is_u65 &&
      poperation->input_tensors[0]->data &&
      poperation->input_tensors[0]->dims[1] == 1 &&
      poperation->input_tensors[0]->dims[2] == 1 &&
      poperation->input_tensors[0]->dims[3] == 1;

   /*
    * IFM cannot be broadcast per axis on the U65, so an operand that has
    * to be broadcast must sit in IFM2. When that is the first operand,
    * exchange the inputs and set the operand-order bit, which for a
    * subtract keeps the result as minuend minus subtrahend. The U85 has
    * no operand-order bit, so exchanging a subtract's operands there
    * would negate the result; instead keep them in place and broadcast
    * the first operand through IFM, which the U85 can do.
    */
   if (!u85_scalar_lhs_sub && !(is_subtract && !is_u65) &&
       !is_sub_shape(poperation->input_tensors[1], poperation->input_tensors[0])) {
      ifm_idx = 1;
      ifm2_idx = 0;
      operation->eltwise.ifm_reversed = true;
   }

   set_feature_maps(subgraph, poperation->input_tensors[ifm_idx], poperation->output_tensors[0], operation);
   if (u85_scalar_lhs_sub) {
      operation->ifm.scalar = ethosu_read_scalar(poperation->input_tensors[ifm_idx]);
      operation->ifm.has_scalar = true;
   } else if (is_subtract && !is_u65 &&
              poperation->input_tensors[ifm_idx]->data) {
      /*
       * A non-scalar constant first operand stays in IFM as a feature map
       * in the coefficient region; the shape-derived IFM broadcast then
       * expands its size-1 axes.
       */
      struct pipe_tensor *minuend = poperation->input_tensors[ifm_idx];
      unsigned size = operation->ifm.shape.height *
                      operation->ifm.shape.width *
                      operation->ifm.shape.depth * minuend->type_size;

      operation->ifm.region = COEFS_REGION;
      operation->ifm.tiles.addresses[0] =
         ethosu_allocate_coefs(subgraph, size);
      memcpy(subgraph->coefs + operation->ifm.tiles.addresses[0],
             minuend->data, size);
      operation->ifm.tiles.height_0 = operation->ifm.shape.height;
      operation->ifm.tiles.height_1 = operation->ifm.shape.height;
      operation->ifm.tiles.width_0 = operation->ifm.shape.width;
   }

   set_feature_map(subgraph, poperation->input_tensors[ifm2_idx], &operation->ifm2);

   if (poperation->input_tensors[ifm2_idx]->data) {
      if (operation->ifm2.shape.width == 1 &&
          operation->ifm2.shape.height == 1 &&
          operation->ifm2.shape.depth == 1) {
         operation->ifm2.scalar =
            ethosu_read_scalar(poperation->input_tensors[ifm2_idx]);
         operation->ifm2.has_scalar = true;
      } else {
         size_t size = operation->ifm2.shape.height * operation->ifm2.shape.width *
                       operation->ifm2.shape.depth * poperation->input_tensors[ifm2_idx]->type_size;

         operation->ifm2.region = COEFS_REGION;
         operation->ifm2.tiles.addresses[0] =
            ethosu_allocate_coefs(subgraph, size);
         memcpy(subgraph->coefs + operation->ifm2.tiles.addresses[0],
                poperation->input_tensors[ifm2_idx]->data, size);
      }
   } else {
      operation->ifm2.region = IO_REGION;
      operation->ifm2.tiles.addresses[0] = ethosu_allocate_feature_map(subgraph, &operation->ifm2);
   }
   operation->ifm2.tiles.height_0 = operation->ifm2.shape.height;
   operation->ifm2.tiles.height_1 = operation->ifm2.shape.height;
   operation->ifm2.tiles.width_0 = operation->ifm2.shape.width;

   if (poperation->add.relu)
      operation->eltwise.activation_min = operation->ofm.zero_point;
}

static void
ethosu_lower_eltwise_with_lut(struct ethosu_subgraph *subgraph,
                              const struct pipe_ml_operation *poperation,
                              const struct pipe_ml_operation *lut,
                              enum ethosu_eltwise_type type,
                              struct ethosu_operation *operation)
{
   ethosu_lower_eltwise(subgraph, poperation, operation);
   operation->eltwise.type = type;

   if (lut) {
      struct ethosu_operation dma_operation;

      ethosu_fuse_lut(subgraph, lut, operation);
      operation_set_defaults(&dma_operation);
      ethosu_lower_lut_dma(subgraph, lut, operation, &dma_operation);
      util_dynarray_append(&subgraph->operations, dma_operation);
   }

   allocate_feature_maps(subgraph, operation);
   ethosu_sched_operation(subgraph, operation);
}

static void
ethosu_lower_dma(struct ethosu_subgraph *subgraph,
                 const struct pipe_ml_operation *poperation,
                 struct ethosu_operation *conv_operation,
                 struct ethosu_operation *operation)
{
   operation->type = ETHOSU_OPERATION_TYPE_DMA;

   operation->dma.address = conv_operation->conv.scales.address;
   operation->dma.size = conv_operation->conv.scales.size + conv_operation->conv.weights.size;
   operation->dma.dst_region = SCRATCH_REGION;

   conv_operation->conv.scales.region = SCRATCH_REGION;
   conv_operation->conv.scales.address = 0;

   conv_operation->conv.weights.region = SCRATCH_REGION;
   conv_operation->conv.weights.address = conv_operation->conv.scales.size;
}

static bool
ethosu_eltwise_fuses_lut(const struct pipe_ml_operation *poperations,
                         unsigned count,
                         const struct pipe_ml_operation *operation)
{
   switch (operation->type) {
   case PIPE_ML_OPERATION_TYPE_ADD:
   case PIPE_ML_OPERATION_TYPE_MUL:
   case PIPE_ML_OPERATION_TYPE_SUBTRACT:
   case PIPE_ML_OPERATION_TYPE_MAXIMUM:
   case PIPE_ML_OPERATION_TYPE_MINIMUM:
      return ethosu_find_fusible_lut(
         poperations, count, operation->output_tensors[0]->index);
   default:
      return false;
   }
}

/* Whether reading `tensor` through `reader` needs it linear (NHWC).  These
 * lowerings read their input as a contiguous NHWC buffer, and a channel-axis
 * split or unpack whose output starts mid-brick cannot address a brick. */
static bool
ethosu_reader_requires_nhwc(const struct pipe_ml_operation *poperations,
                            unsigned count,
                            const struct pipe_ml_operation *reader,
                            const struct pipe_tensor *tensor)
{
   switch (reader->type) {
   case PIPE_ML_OPERATION_TYPE_SOFTMAX:
   case PIPE_ML_OPERATION_TYPE_BATCH_MATMUL:
   case PIPE_ML_OPERATION_TYPE_TRANSPOSE:
   case PIPE_ML_OPERATION_TYPE_RESIZE_BILINEAR:
   case PIPE_ML_OPERATION_TYPE_ARGMAX:
      return true;
   default:
      break;
   }

   if (ethosu_consumer_uses_depth_mean(reader) ||
       ethosu_eltwise_fuses_lut(poperations, count, reader))
      return true;

   if ((reader->type == PIPE_ML_OPERATION_TYPE_SPLIT ||
        reader->type == PIPE_ML_OPERATION_TYPE_UNPACK) &&
       reader->split.axis == 3 &&
       reader->input_tensors[0]->index == tensor->index) {
      bool is_unpack = reader->type == PIPE_ML_OPERATION_TYPE_UNPACK;

      for (unsigned k = 1; k < reader->output_count; k++) {
         unsigned offset =
            is_unpack ? k * reader->output_tensors[k]->dims[3]
                      : ethosu_concat_split_offset(reader->output_tensors,
                                                   k, 3);
         if (offset & 15)
            return true;
      }
   }

   return false;
}

/* Walk the readers of `tensor_index`, following through reshapes, which alias
 * the buffer without copying.  A reshape is not itself a reader: the readers
 * behind it read the producer's buffer and are checked against what the
 * producer wrote (`width`, `depth`).  Returns false if a reader needs linear
 * format or reads a differing width or depth, or if a reshape output leaves
 * the graph, which is never brick and gets no copy.  *has_reader records that
 * a reader was found behind the reshapes. */
static bool
ethosu_readers_keep_brick(const struct ethosu_subgraph *subgraph,
                          const struct pipe_ml_operation *poperations,
                          unsigned count, unsigned tensor_index,
                          unsigned width, unsigned depth, bool *has_reader)
{
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *reader = &poperations[i];

      for (unsigned j = 0; j < reader->input_count; j++) {
         const struct pipe_tensor *in = reader->input_tensors[j];
         unsigned read_width, read_depth;

         if (in->index != tensor_index)
            continue;

         if (reader->type == PIPE_ML_OPERATION_TYPE_RESHAPE) {
            const struct pipe_tensor *out = reader->output_tensors[0];

            if (out->is_external_output)
               return false;
            if (!ethosu_readers_keep_brick(subgraph, poperations, count,
                                           out->index, width, depth,
                                           has_reader))
               return false;
            continue;
         }

         *has_reader = true;
         ethosu_read_wd(subgraph, reader, in, &read_width, &read_depth);
         if (read_width != width || read_depth != depth)
            return false;
         if (ethosu_reader_requires_nhwc(poperations, count, reader, in))
            return false;
      }
   }

   return true;
}

/* A tensor keeps brick format (NHCWB16) only if a reader exists and every
 * reader agrees on the written width and depth and reads brick-packed data.
 * Reshapes are transparent to the walk. */
static bool
ethosu_tensor_keeps_brick(const struct ethosu_subgraph *subgraph,
                          const struct pipe_ml_operation *poperations,
                          unsigned count,
                          const struct pipe_ml_operation *producer,
                          const struct pipe_tensor *ptensor)
{
   unsigned width, depth;
   bool has_reader = false;

   ethosu_written_wd(subgraph, producer, ptensor, &width, &depth);

   if (!ethosu_readers_keep_brick(subgraph, poperations, count,
                                  ptensor->index, width, depth, &has_reader))
      return false;

   return has_reader;
}

static void
register_tensors(struct ethosu_subgraph *subgraph,
                 const struct pipe_ml_operation *poperations,
                 unsigned count)
{
   subgraph->batches = 1;
   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->input_count; j++) {
         const struct pipe_tensor *ptensor = poperation->input_tensors[j];

         /* Reshapes may use any leading dimension, unlike graph inputs. */
         if (!ptensor->data &&
             !ethosu_find_first_producer(poperations, count, ptensor->index))
            subgraph->batches = MAX2(subgraph->batches, ptensor->dims[0]);
      }
   }

   for (unsigned i = 0; i < count; i++) {
      const struct pipe_ml_operation *poperation = &poperations[i];

      for (unsigned j = 0; j < poperation->input_count; j++) {
         struct pipe_tensor *ptensor = poperation->input_tensors[j];

         if (poperation->type == PIPE_ML_OPERATION_TYPE_SCATTER_ND && j != 1)
            continue;

         ethosu_register_tensor(subgraph, ptensor);
      }

      for (unsigned j = 0; j < poperation->output_count; j++) {
         struct pipe_tensor *ptensor = poperation->output_tensors[j];

         ethosu_register_tensor(subgraph, ptensor);

         if (!ptensor->is_external_output &&
             !DBG_ENABLED(ETHOSU_DBG_DISABLE_NHCWB16) &&
             poperation->type != PIPE_ML_OPERATION_TYPE_BATCH_MATMUL &&
             poperation->type != PIPE_ML_OPERATION_TYPE_SCATTER_ND &&
             poperation->type != PIPE_ML_OPERATION_TYPE_TRANSPOSE &&
             poperation->type != PIPE_ML_OPERATION_TYPE_SOFTMAX &&
             poperation->type != PIPE_ML_OPERATION_TYPE_PAD) {
            struct ethosu_tensor *tensor = ethosu_find_tensor(subgraph, ptensor->index);
            const struct pipe_ml_operation *consumer =
               ethosu_find_first_consumer(poperations, count, ptensor->index);
            if (ethosu_tensor_keeps_brick(subgraph, poperations, count,
                                          poperation, ptensor) &&
                !ethosu_concat_offset_needs_nhwc(poperation)) {
               if ((poperation->type != PIPE_ML_OPERATION_TYPE_RESHAPE &&
                    !ethosu_consumer_is_lut(poperation) &&
                    !ethosu_consumer_is_lut(consumer)) ||
                   (poperation->type == PIPE_ML_OPERATION_TYPE_CONCATENATION &&
                    ethosu_consumer_is_lut(consumer)))
                  tensor->layout = ETHOSU_LAYOUT_NHCWB16;
            }
         }
      }
   }
}

static void
ethosu_offset_feature_map_batch(struct ethosu_feature_map *fm, unsigned batch)
{
   unsigned span;

   if (fm->region != IO_REGION || !fm->tensor || fm->tensor->batches <= 1)
      return;

   span = fm->tensor->size / fm->tensor->batches;
   for (unsigned i = 0; i < ARRAY_SIZE(fm->tiles.addresses); i++)
      fm->tiles.addresses[i] += batch * span;
}

static void
ethosu_duplicate_batch_operations(struct ethosu_subgraph *subgraph)
{
   unsigned operation_count =
      util_dynarray_num_elements(&subgraph->operations, struct ethosu_operation);

   for (unsigned batch = 1; batch < subgraph->batches; batch++) {
      for (unsigned i = 0; i < operation_count; i++) {
         struct ethosu_operation operation =
            *util_dynarray_element(&subgraph->operations,
                                   struct ethosu_operation, i);

         ethosu_offset_feature_map_batch(&operation.ifm, batch);
         ethosu_offset_feature_map_batch(&operation.ifm2, batch);
         ethosu_offset_feature_map_batch(&operation.ofm, batch);

         operation.kernel.scales = NULL;
         operation.kernel.zero_points = NULL;
         util_dynarray_append(&subgraph->operations, operation);
      }
   }
}

void
ethosu_lower_graph(struct ethosu_subgraph *subgraph,
                   const struct pipe_ml_operation *poperations, unsigned count)
{
   register_tensors(subgraph, poperations, count);
   ethosu_require_softmax_spans(subgraph, poperations, count);
   ethosu_reserve_internal_tensors(subgraph, count);
   if (subgraph->failed)
      return;

   /* Lower */
   for (int i = 0; i < count; i++) {
      struct ethosu_operation operation;

      operation_set_defaults(&operation);

      switch (poperations[i].type) {

      case PIPE_ML_OPERATION_TYPE_FULLY_CONNECTED: {
         struct pipe_tensor *input_tensor = poperations[i].input_tensors[0];
         const struct pipe_ml_operation *lut =
            ethosu_find_fusible_lut(poperations, count,
                                    poperations[i].output_tensors[0]->index);

         ethosu_lower_fully_connected(subgraph, &poperations[i], input_tensor, &operation);
         if (lut) {
            struct ethosu_operation dma_operation;

            ethosu_fuse_lut(subgraph, lut, &operation);
            operation_set_defaults(&dma_operation);
            ethosu_lower_lut_dma(subgraph, lut, &operation, &dma_operation);
            util_dynarray_append(&subgraph->operations, dma_operation);
         }
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_BATCH_MATMUL:
         ethosu_lower_batch_matmul(subgraph, &poperations[i]);
         break;

      case PIPE_ML_OPERATION_TYPE_SPACE_TO_BATCH:
      case PIPE_ML_OPERATION_TYPE_BATCH_TO_SPACE:
         /* Fold the supported SpaceToBatchND/conv/BatchToSpaceND pattern. */
         assert(ethosu_space_batch_is_fused(poperations, count,
                                            &poperations[i]));
         break;

      case PIPE_ML_OPERATION_TYPE_CONVOLUTION: {
         const struct pipe_ml_operation *conv = &poperations[i];
         struct pipe_ml_operation fused_conv;
         struct pipe_tensor *fused_inputs[3];
         struct pipe_tensor *fused_outputs[1];
         const struct pipe_ml_operation *space_to_batch;
         const struct pipe_ml_operation *batch_to_space;
         struct pipe_tensor *input_tensor = conv->input_tensors[0];
         const struct pipe_ml_operation *producer = ethosu_find_first_producer(poperations, count, input_tensor->index);
         const struct pipe_ml_operation *lut;
         bool padded_input = producer &&
                             producer->type == PIPE_ML_OPERATION_TYPE_PAD &&
                             !producer->pad.raw_zero;

         if (ethosu_space_batch_fusion(poperations, count, conv,
                                       &space_to_batch, &batch_to_space)) {
            fused_conv = *conv;
            memcpy(fused_inputs, conv->input_tensors,
                   conv->input_count * sizeof(*fused_inputs));
            fused_inputs[0] = space_to_batch->input_tensors[0];
            fused_outputs[0] = batch_to_space->output_tensors[0];
            fused_conv.input_tensors = fused_inputs;
            fused_conv.output_tensors = fused_outputs;
            fused_conv.conv.dilation_height_factor *=
               space_to_batch->space_batch.block_y;
            fused_conv.conv.dilation_width_factor *=
               space_to_batch->space_batch.block_x;
            fused_conv.conv.padding_top =
               (fused_outputs[0]->dims[1] - 1) * fused_conv.conv.stride_y +
               (fused_conv.conv.weight_tensor->dims[1] - 1) *
                  fused_conv.conv.dilation_height_factor +
               1 -
               fused_inputs[0]->dims[1];
            fused_conv.conv.padding_left =
               (fused_outputs[0]->dims[2] - 1) * fused_conv.conv.stride_x +
               (fused_conv.conv.weight_tensor->dims[2] - 1) *
                  fused_conv.conv.dilation_width_factor +
               1 -
               fused_inputs[0]->dims[2];
            fused_conv.conv.padding_bottom = fused_conv.conv.padding_top -
                                             fused_conv.conv.padding_top / 2;
            fused_conv.conv.padding_top /= 2;
            fused_conv.conv.padding_right = fused_conv.conv.padding_left -
                                            fused_conv.conv.padding_left / 2;
            fused_conv.conv.padding_left /= 2;
            conv = &fused_conv;
            input_tensor = fused_inputs[0];
            padded_input = false;
         }

         lut = ethosu_find_fusible_lut(poperations, count,
                                       conv->output_tensors[0]->index);

         if (padded_input) {
            input_tensor = producer->input_tensors[0];
         }

         ethosu_lower_convolution(subgraph, conv, input_tensor, &operation);

         if (lut)
            ethosu_fuse_lut(subgraph, lut, &operation);

         if (padded_input) {
            operation.pad.top += producer->pad.before_y;
            operation.pad.bottom += producer->pad.after_y;
            operation.pad.left += producer->pad.before_x;
            operation.pad.right += producer->pad.after_x;
            trim_padding_to_output(&operation, input_tensor,
                                   conv->output_tensors[0]);
         }

         if (ethosu_needs_strided_conv_unroll(&operation)) {
            if (operation.conv.scales.size + operation.conv.weights.size <=
                ethosu_ml_device(subgraph->base.device)->sram_size) {
               struct ethosu_operation dma_operation = {0};

               ethosu_lower_dma(subgraph, conv, &operation,
                                &dma_operation);
               util_dynarray_append(&subgraph->operations, dma_operation);
            }
            if (lut) {
               struct ethosu_operation dma_operation = {0};

               ethosu_lower_lut_dma(subgraph, lut, &operation,
                                    &dma_operation);
               util_dynarray_append(&subgraph->operations, dma_operation);
            }

            ethosu_append_strided_conv_unroll(subgraph, &operation);
            break;
         }

         if (operation.conv.scales.size + operation.conv.weights.size <=
             ethosu_ml_device(subgraph->base.device)->sram_size) {
            struct ethosu_operation dma_operation = {0};
            ethosu_lower_dma(subgraph, conv, &operation, &dma_operation);

            util_dynarray_append(&subgraph->operations, dma_operation);
         }
         if (lut) {
            struct ethosu_operation dma_operation = {0};

            ethosu_lower_lut_dma(subgraph, lut, &operation,
                                 &dma_operation);
            util_dynarray_append(&subgraph->operations, dma_operation);
         }
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_ADD: {
         const struct pipe_ml_operation *lut =
            ethosu_find_fusible_lut(poperations, count,
                                    poperations[i].output_tensors[0]->index);

         ethosu_lower_eltwise_with_lut(subgraph, &poperations[i], lut,
                                       ETHOSU_ELTWISE_TYPE_ADD, &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_MUL: {
         const struct pipe_ml_operation *lut =
            ethosu_find_fusible_lut(poperations, count,
                                    poperations[i].output_tensors[0]->index);

         ethosu_lower_eltwise_with_lut(subgraph, &poperations[i], lut,
                                       ETHOSU_ELTWISE_TYPE_MUL, &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_SUBTRACT: {
         const struct pipe_ml_operation *lut =
            ethosu_find_fusible_lut(poperations, count,
                                    poperations[i].output_tensors[0]->index);

         ethosu_lower_eltwise_with_lut(subgraph, &poperations[i], lut,
                                       ETHOSU_ELTWISE_TYPE_SUB, &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_MAXIMUM: {
         const struct pipe_ml_operation *lut =
            ethosu_find_fusible_lut(poperations, count,
                                    poperations[i].output_tensors[0]->index);

         ethosu_lower_eltwise_with_lut(subgraph, &poperations[i], lut,
                                       ETHOSU_ELTWISE_TYPE_MAX, &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_MINIMUM: {
         const struct pipe_ml_operation *lut =
            ethosu_find_fusible_lut(poperations, count,
                                    poperations[i].output_tensors[0]->index);

         ethosu_lower_eltwise_with_lut(subgraph, &poperations[i], lut,
                                       ETHOSU_ELTWISE_TYPE_MIN, &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_POOLING: {
         ethosu_lower_pooling(subgraph, &poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_ARGMAX: {
         ethosu_lower_argmax(subgraph, &poperations[i]);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_MEAN: {
         ethosu_lower_mean(subgraph, &poperations[i]);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_LOGISTIC: {
         if (ethosu_lut_is_fused(poperations, count, &poperations[i])) {
            ethosu_lower_reshape(subgraph, &poperations[i], &operation);
            util_dynarray_append(&subgraph->operations, operation);
            break;
         }

         ethosu_lower_lut(subgraph, &poperations[i], &operation, clamp_sigmoid8);
         ethosu_append_lut_pool(subgraph, &poperations[i], &operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_TANH: {
         if (ethosu_lut_is_fused(poperations, count, &poperations[i])) {
            ethosu_lower_reshape(subgraph, &poperations[i], &operation);
            util_dynarray_append(&subgraph->operations, operation);
            break;
         }

         ethosu_lower_lut(subgraph, &poperations[i], &operation, tanh);
         ethosu_append_lut_pool(subgraph, &poperations[i], &operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_RSQRT: {
         if (ethosu_lut_is_fused(poperations, count, &poperations[i])) {
            ethosu_lower_reshape(subgraph, &poperations[i], &operation);
            util_dynarray_append(&subgraph->operations, operation);
            break;
         }

         ethosu_lower_rsqrt(subgraph, &poperations[i], &operation);
         ethosu_append_lut_pool(subgraph, &poperations[i], &operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_HSWISH: {
         if (ethosu_lut_is_fused(poperations, count, &poperations[i])) {
            ethosu_lower_reshape(subgraph, &poperations[i], &operation);
            util_dynarray_append(&subgraph->operations, operation);
            break;
         }

         ethosu_lower_hswish(subgraph, &poperations[i], &operation);
         ethosu_append_lut_pool(subgraph, &poperations[i], &operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_STRIDED_SLICE: {
         ethosu_lower_strided_slice(subgraph, &poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_TRANSPOSE: {
         ethosu_lower_transpose(subgraph, &poperations[i]);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_CONCATENATION:
      case PIPE_ML_OPERATION_TYPE_PACK: {
         for (int j = poperations[i].input_count - 1; j >= 0; j--) {
            operation_set_defaults(&operation);
            ethosu_lower_concatenation(subgraph, &poperations[i], j, &operation);
            util_dynarray_append(&subgraph->operations, operation);
         }
         break;
      }

      case PIPE_ML_OPERATION_TYPE_SPLIT: {
         for (unsigned j = 0; j < poperations[i].output_count; j++) {
            operation_set_defaults(&operation);
            ethosu_lower_split(subgraph, &poperations[i], j, &operation);
            util_dynarray_append(&subgraph->operations, operation);
         }
         break;
      }

      case PIPE_ML_OPERATION_TYPE_UNPACK: {
         for (int j = poperations[i].output_count - 1; j >= 0; j--) {
            operation_set_defaults(&operation);
            ethosu_lower_unpack(subgraph, &poperations[i], j, &operation);
            util_dynarray_append(&subgraph->operations, operation);
         }
         break;
      }

      case PIPE_ML_OPERATION_TYPE_RESIZE: {
         ethosu_lower_resize(subgraph, &poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_RESIZE_BILINEAR:
         ethosu_lower_resize_bilinear(subgraph, &poperations[i]);
         break;

      case PIPE_ML_OPERATION_TYPE_PAD: {
         if (!poperations[i].pad.raw_zero &&
             ethosu_all_consumers_are_convolutions(
                poperations, count, poperations[i].output_tensors[0]->index))
            break;

         ethosu_lower_pad(subgraph, &poperations[i],
                          poperations[i].input_tensors[0]);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_SCATTER_ND: {
         struct pipe_ml_operation pad;
         ASSERTED bool ok = ethosu_scatter_nd_as_pad(&poperations[i], &pad);

         assert(ok);
         ethosu_lower_pad(subgraph, &pad, poperations[i].input_tensors[1]);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_LEAKY_RELU: {
         if (ethosu_lut_is_fused(poperations, count, &poperations[i])) {
            ethosu_lower_reshape(subgraph, &poperations[i], &operation);
            util_dynarray_append(&subgraph->operations, operation);
            break;
         }

         ethosu_lower_leakyrelu(subgraph, &poperations[i], &operation);
         ethosu_append_lut_pool(subgraph, &poperations[i], &operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_QUANTIZE: {
         ethosu_lower_quantize(subgraph, &poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_RESHAPE: {
         ethosu_lower_reshape(subgraph, &poperations[i], &operation);
         util_dynarray_append(&subgraph->operations, operation);
         break;
      }

      case PIPE_ML_OPERATION_TYPE_SOFTMAX: {
         ethosu_lower_softmax(subgraph, &poperations[i]);
         break;
      }

      default:
         DBG("poperation->type %d\n", poperations[i].type);
         UNREACHABLE("Unsupported ML operation type");
      }
   }

   ethosu_duplicate_batch_operations(subgraph);
}
