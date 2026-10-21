/*
 * Copyright (c) 2026 Tomeu Vizoso <tomeu@tomeuvizoso.net>
 * SPDX-License-Identifier: MIT
 */

#include <stdlib.h>
#include <string.h>

#include "pipe-loader/pipe_loader.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"

#include "torx_backend.h"

static struct pipe_loader_device *
find_accel_device(void)
{
   struct pipe_loader_device *device = NULL;
   struct pipe_loader_device **devs;

   int n = pipe_loader_accel_probe(NULL, 0);
   if (n <= 0)
      return NULL;

   devs = (struct pipe_loader_device **)malloc(sizeof(*devs) * n);
   if (!devs)
      return NULL;
   n = pipe_loader_accel_probe(devs, n);

   for (int i = 0; i < n; i++) {
      if (!device &&
          (strstr(devs[i]->driver_name, "rocket") ||
           strstr(devs[i]->driver_name, "ethosu")))
         device = devs[i];
      else
         pipe_loader_release(&devs[i], 1);
   }
   free(devs);

   return device;
}

static struct pipe_loader_device *
find_drm_device(void)
{
   struct pipe_loader_device *device = NULL;
   struct pipe_loader_device **devs;

   int n = pipe_loader_probe(NULL, 0, false);
   if (n <= 0)
      return NULL;

   devs = (struct pipe_loader_device **)malloc(sizeof(*devs) * n);
   if (!devs)
      return NULL;
   n = pipe_loader_probe(devs, n, false);

   for (int i = 0; i < n; i++) {
      if (!device && strstr(devs[i]->driver_name, "etnaviv"))
         device = devs[i];
      else
         pipe_loader_release(&devs[i], 1);
   }
   free(devs);

   return device;
}

struct torx_backend *
torx_backend_create(void)
{
   struct torx_backend *backend = (struct torx_backend *)calloc(1, sizeof(*backend));

   struct pipe_loader_device *loader_dev = find_accel_device();

   if (!loader_dev)
      loader_dev = find_drm_device();

   if (!loader_dev)
      goto err_backend;

   backend->screen = pipe_loader_create_screen(loader_dev, false);
   if (!backend->screen)
      goto err_loader;

   backend->context = backend->screen->context_create(backend->screen, NULL, 0);
   if (!backend->context)
      goto err_screen;

   backend->ml_dev = backend->screen->get_ml_device(backend->screen);
   if (!backend->ml_dev)
      goto err_context;

   pipe_loader_release(&loader_dev, 1);

   return backend;

err_context:
   backend->context->destroy(backend->context);

err_screen:
   backend->screen->destroy(backend->screen);

err_loader:
   pipe_loader_release(&loader_dev, 1);

err_backend:
   free(backend);
   return NULL;
}

void
torx_backend_destroy(struct torx_backend *backend)
{
   if (!backend)
      return;

   if (backend->context)
      backend->context->destroy(backend->context);
   if (backend->screen)
      backend->screen->destroy(backend->screen);

   free(backend);
}
