/*
 * Copyright (c) 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/result.h>
#include <executorch/runtime/executor/pte_data_map.h>

#include "torx_backend.h"

namespace executorch {
namespace backends {

using executorch::ET_RUNTIME_NAMESPACE::Backend;
using executorch::ET_RUNTIME_NAMESPACE::BackendExecutionContext;
using executorch::ET_RUNTIME_NAMESPACE::BackendInitContext;
using executorch::ET_RUNTIME_NAMESPACE::CompileSpec;
using executorch::ET_RUNTIME_NAMESPACE::DelegateHandle;
using executorch::ET_RUNTIME_NAMESPACE::NamedDataMap;
using executorch::runtime::ArrayRef;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::Result;
using executorch::runtime::Span;

class TorxBackend final
    : public ::executorch::ET_RUNTIME_NAMESPACE::BackendInterface {
 private:
   mutable struct torx_backend *torx_backend = nullptr;

   static constexpr unsigned MAX_IO_TENSORS = 16;
   static constexpr unsigned INVALID_IDX = 0xFFFFFFFF;

   struct ExecutionHandle {
      struct pipe_ml_subgraph *subgraph = nullptr;
      unsigned num_inputs = 0;
      unsigned num_outputs = 0;
      unsigned inputs[MAX_IO_TENSORS];
      unsigned outputs[MAX_IO_TENSORS];
   };

   bool ensure_backend() const
   {
      if (torx_backend)
         return true;

      torx_backend = torx_backend_create();
      if (!torx_backend) {
         fprintf(stderr, "Failed to create TorxBackend\n");
         return false;
      }

      return true;
   }

 public:
   /*
    * Intentionally leak torx_backend in the static destructor.  This object
    * lives in a file-scope anonymous namespace, so its destructor runs during
    * process/library teardown.  At that point the DRM file descriptors that
    * the pipe_screen holds may already be closed, and calling
    * screen->destroy() triggers "Bad file descriptor" warnings from
    * os_same_file_description.  Per-delegate resources are already freed in
    * destroy() during normal ExecuTorch runtime shutdown.
    */
   ~TorxBackend() = default;

   TorxBackend() = default;

   bool is_available() const override
   {
      return ensure_backend();
   }

   Result<DelegateHandle *> init(
      BackendInitContext &context,
      FreeableBuffer *processed,
      ArrayRef<CompileSpec> compile_specs) const override
   {
      if (!ensure_backend())
         return Error::InvalidState;

      runtime::MemoryAllocator *allocator = context.get_runtime_allocator();
      const NamedDataMap *named_data_map = context.get_named_data_map();
      Result<FreeableBuffer> buffer = named_data_map->get_data("torx_device_id");
      if (!buffer.ok()) {
         fprintf(stderr, "TorxBackend: Missing torx_device_id in NamedDataMap\n");
         return Error::InvalidProgram;
      }

      char *torx_device_id = (char *)malloc(buffer.get().size() + 1);
      strncpy(torx_device_id, static_cast<const char *>(buffer.get().data()), buffer.get().size());
      torx_device_id[buffer.get().size()] = '\0';
      if (torx_backend->ml_dev->id == NULL) {
         fprintf(stderr, "TorxBackend: driver does not identify itself for "
                 "ahead-of-time compiled programs\n");
         free(torx_device_id);
         return Error::InvalidProgram;
      }
      if (strcmp(torx_device_id, torx_backend->ml_dev->id) != 0) {
         fprintf(stderr, "TorxBackend: Mismatched device id, expected %s got %s\n",
                 torx_backend->ml_dev->id, torx_device_id);
         free(torx_device_id);
         return Error::InvalidProgram;
      }
      free(torx_device_id);

      ExecutionHandle *handle = allocator->allocateInstance<ExecutionHandle>();
      if (handle == nullptr) {
         return Error::MemoryAllocationFailed;
      }

      /* Layout defined by torx_blob_header/torx_io_map in torx_backend.h. */
      const uint8_t *blob = static_cast<const uint8_t *>(processed->data());
      const size_t blob_size = processed->size();
      const struct torx_blob_header *hdr =
         reinterpret_cast<const struct torx_blob_header *>(blob);

      if (blob_size < sizeof(*hdr) ||
          hdr->subgraph_size > blob_size - sizeof(*hdr) ||
          hdr->io_map_size < sizeof(struct torx_io_map) ||
          hdr->io_map_size > blob_size - sizeof(*hdr) - hdr->subgraph_size) {
         fprintf(stderr, "TorxBackend: delegate payload is truncated\n");
         processed->Free();
         return Error::InvalidProgram;
      }

      if (hdr->subgraph_size > 0) {
         handle->subgraph = torx_backend->context->ml_subgraph_deserialize(
            torx_backend->context, blob + sizeof(*hdr), hdr->subgraph_size);
         if (!handle->subgraph) {
            fprintf(stderr, "TorxBackend: Failed to deserialize subgraph\n");
            processed->Free();
            return Error::InvalidProgram;
         }
      } else {
         handle->subgraph = nullptr;
      }

      const struct torx_io_map *io =
         reinterpret_cast<const struct torx_io_map *>(
            blob + sizeof(*hdr) + hdr->subgraph_size);
      handle->num_inputs = io->num_inputs;
      handle->num_outputs = io->num_outputs;
      if (handle->num_inputs > MAX_IO_TENSORS ||
          handle->num_outputs > MAX_IO_TENSORS ||
          sizeof(*io) + (handle->num_inputs + handle->num_outputs) *
             sizeof(io->gallium_idx[0]) > hdr->io_map_size) {
         fprintf(stderr, "TorxBackend: too many I/O tensors (%u in, %u out)\n",
                 handle->num_inputs, handle->num_outputs);
         processed->Free();
         return Error::InvalidProgram;
      }
      for (unsigned i = 0; i < handle->num_inputs; i++)
         handle->inputs[i] = io->gallium_idx[i];
      for (unsigned i = 0; i < handle->num_outputs; i++)
         handle->outputs[i] = io->gallium_idx[handle->num_inputs + i];

      processed->Free();

      return handle;
   }

   /* Transpose a 4-D tensor between NCHW and NHWC layouts (byte-wise). */
   static void transpose_nchw_nhwc(int8_t *dst, const int8_t *src,
                                   ssize_t N, ssize_t C, ssize_t H, ssize_t W,
                                   bool to_nhwc)
   {
      for (ssize_t n = 0; n < N; n++)
         for (ssize_t c = 0; c < C; c++)
            for (ssize_t h = 0; h < H; h++)
               for (ssize_t w = 0; w < W; w++) {
                  ssize_t nchw = ((n * C + c) * H + h) * W + w;
                  ssize_t nhwc = ((n * H + h) * W + w) * C + c;
                  if (to_nhwc)
                     dst[nhwc] = src[nchw];
                  else
                     dst[nchw] = src[nhwc];
               }
   }

   Error execute(
      BackendExecutionContext &context,
      DelegateHandle *handle,
      Span<EValue *> args) const override
   {
      auto ehandle = static_cast<ExecutionHandle *>(handle);

      /* TODO: In the future, tensors should carry information about their
       * memory layout so the backend can avoid these explicit transposes.
       */

      /* --- Prepare and transpose inputs (NCHW -> NHWC) --- */
      unsigned gallium_in_count = 0;
      unsigned input_idxs[MAX_IO_TENSORS];
      void    *input_data[MAX_IO_TENSORS];
      bool     input_signed[MAX_IO_TENSORS];
      void    *in_nhwc_bufs[MAX_IO_TENSORS];

      for (unsigned i = 0; i < ehandle->num_inputs; i++) {
         unsigned gallium_idx = ehandle->inputs[i];

         if (gallium_idx == INVALID_IDX)
            continue;

         auto &t = args[i]->toTensor();
         if (t.dim() != 4) {
            fprintf(stderr, "TorxBackend: expected 4-D input tensor, got %zd-D\n",
                    (ssize_t)t.dim());
            for (unsigned j = 0; j < gallium_in_count; j++)
               free(in_nhwc_bufs[j]);
            return Error::InvalidArgument;
         }

         ssize_t N = t.sizes()[0], C = t.sizes()[1];
         ssize_t H = t.sizes()[2], W = t.sizes()[3];
         auto *nhwc = static_cast<int8_t *>(malloc(N * C * H * W));

         /* Direct int8 — just transpose */
         const int8_t *sdata = static_cast<const int8_t *>(t.data_ptr());
         transpose_nchw_nhwc(nhwc, sdata,
                             N, C, H, W, true);

         unsigned gi = gallium_in_count++;
         input_idxs[gi] = gallium_idx;
         input_signed[gi] = true;
         in_nhwc_bufs[gi] = nhwc;
         input_data[gi] = nhwc;
      }

      /* --- Execute on NPU (skip for Q/DQ-only delegates) --- */
      if (ehandle->subgraph) {
         torx_backend->context->ml_subgraph_invoke(
            torx_backend->context, ehandle->subgraph,
            gallium_in_count, input_idxs, input_data, input_signed);
      }

      for (unsigned i = 0; i < gallium_in_count; i++)
         free(in_nhwc_bufs[i]);

      /* --- Read and transpose outputs (NHWC -> NCHW) --- */
      unsigned gallium_out_count = 0;
      unsigned output_idxs[MAX_IO_TENSORS];
      void    *output_data[MAX_IO_TENSORS];
      bool     output_signed[MAX_IO_TENSORS];
      void    *out_nhwc_bufs[MAX_IO_TENSORS];
      /* Track which output index maps to which args entry. */
      unsigned out_args_idx[MAX_IO_TENSORS];

      for (unsigned i = 0; i < ehandle->num_outputs; i++) {
         unsigned gallium_idx = ehandle->outputs[i];

         if (gallium_idx == INVALID_IDX) {
            fprintf(stderr, "TorxBackend: invalid output tensor mapping\n");
            return Error::InvalidProgram;
         }

         auto &t = args[ehandle->num_inputs + i]->toTensor();
         if (t.dim() != 4) {
            fprintf(stderr, "TorxBackend: expected 4-D output tensor, got %zd-D\n",
                    (ssize_t)t.dim());
            return Error::InvalidArgument;
         }

         ssize_t N = t.sizes()[0], C = t.sizes()[1];
         ssize_t H = t.sizes()[2], W = t.sizes()[3];

         unsigned gi = gallium_out_count++;
         output_idxs[gi] = gallium_idx;
         output_signed[gi] = true;
         out_nhwc_bufs[gi] = malloc(N * C * H * W);
         output_data[gi] = out_nhwc_bufs[gi];
         out_args_idx[gi] = i;
      }

      if (ehandle->subgraph && gallium_out_count > 0) {
         torx_backend->context->ml_subgraph_read_output(
            torx_backend->context, ehandle->subgraph,
            gallium_out_count, output_idxs, output_data, output_signed);
      }

      for (unsigned gi = 0; gi < gallium_out_count; gi++) {
         unsigned i = out_args_idx[gi];
         auto &t = args[ehandle->num_inputs + i]->toTensor();
         ssize_t N = t.sizes()[0], C = t.sizes()[1];
         ssize_t H = t.sizes()[2], W = t.sizes()[3];

         /* Direct int8 → just transpose NHWC→NCHW */
         transpose_nchw_nhwc(static_cast<int8_t *>(t.mutable_data_ptr()),
                             static_cast<const int8_t *>(out_nhwc_bufs[gi]),
                             N, C, H, W, false);
         free(out_nhwc_bufs[gi]);
      }

      return Error::Ok;
   }

   void destroy(DelegateHandle *handle) const override
   {
      auto ehandle = static_cast<ExecutionHandle *>(handle);
      if (ehandle && ehandle->subgraph && torx_backend && torx_backend->ml_dev)
         torx_backend->ml_dev->ml_subgraph_destroy(
            torx_backend->ml_dev, ehandle->subgraph);
   }
};

static bool
register_torx_backend(TorxBackend &cls)
{
   Backend backend{"TorxBackend", &cls};
   if (register_backend(backend) != Error::Ok) {
      fprintf(stderr, "Failed to register TorxBackend\n");
      return false;
   }

   return true;
}

namespace {

auto cls = TorxBackend();
bool success = register_torx_backend(cls);

} // namespace

} // namespace backends
} // namespace executorch
