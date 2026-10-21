/*
 * Copyright (c) 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#ifndef _TORX_BACKEND_H_
#define _TORX_BACKEND_H_

#include <stdint.h>

#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Layout of the delegate payload written by TorxBackend.preprocess() in
 * backend.py: the header, then subgraph_size bytes of serialized
 * subgraph, then a torx_io_map of io_map_size bytes. */
struct torx_blob_header {
   uint32_t subgraph_size;
   uint32_t io_map_size;
};

struct torx_io_map {
   uint32_t num_inputs;
   uint32_t num_outputs;
   uint32_t gallium_idx[];
};

struct torx_backend {
   struct pipe_screen *screen;
   struct pipe_context *context;
   struct pipe_ml_device *ml_dev;
};

PUBLIC struct torx_backend *torx_backend_create(void);
PUBLIC void torx_backend_destroy(struct torx_backend *backend);

#ifdef __cplusplus
}
#endif

#endif /* _TORX_BACKEND_H_ */
