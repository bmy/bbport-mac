/**************************************************************************
 * 
 * Copyright 2008 VMware, Inc.
 * Copyright 2010 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 * 
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 * 
 **************************************************************************/

/**
 * Polygon stipple helper module.  Drivers/GPUs which don't support polygon
 * stipple natively can use this module to simulate it.
 *
 * Basically, modify fragment shader to sample the 32x32 stipple pattern
 * texture and do a fragment kill for the 'off' bits.
 *
 * This was originally a 'draw' module stage, but since we don't need
 * vertex window coords or anything, it can be a stand-alone utility module.
 *
 * Authors:  Brian Paul
 */


#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_shader_tokens.h"
#include "util/u_inlines.h"

#include "util/format/u_format.h"
#include "util/u_memory.h"
#include "util/u_pstipple.h"
#include "util/u_sampler.h"

#include "tgsi/tgsi_transform.h"
#include "tgsi/tgsi_dump.h"
#include "tgsi/tgsi_scan.h"

/** Approx number of new tokens for instructions in pstip_transform_inst() */
#define NUM_NEW_TOKENS 53


void
util_pstipple_update_stipple_texture(struct pipe_context *pipe,
                                     struct pipe_resource *tex,
                                     const uint32_t pattern[32])
{
   static const unsigned bit31 = 1u << 31;
   struct pipe_transfer *transfer;
   uint8_t *data;
   int i, j;

   /* map texture memory */
   data = pipe_texture_map(pipe, tex, 0, 0,
                            PIPE_MAP_WRITE, 0, 0, 32, 32, &transfer);

   /*
    * Load alpha texture.
    * Note: 0 means keep the fragment, 255 means kill it.
    * We'll negate the texel value and use KILL_IF which kills if value
    * is negative.
    */
   for (i = 0; i < 32; i++) {
      for (j = 0; j < 32; j++) {
         if (pattern[i] & (bit31 >> j)) {
            /* fragment "on" */
            data[i * transfer->stride + j] = 0;
         }
         else {
            /* fragment "off" */
            data[i * transfer->stride + j] = 255;
         }
      }
   }

   /* unmap */
   pipe->texture_unmap(pipe, transfer);
}


/**
 * Create a 32x32 alpha8 texture that encodes the given stipple pattern.
 */
struct pipe_resource *
util_pstipple_create_stipple_texture(struct pipe_context *pipe,
                                     const uint32_t pattern[32])
{
   struct pipe_screen *screen = pipe->screen;
   struct pipe_resource templat, *tex;

   memset(&templat, 0, sizeof(templat));
   templat.target = PIPE_TEXTURE_2D;
   templat.format = PIPE_FORMAT_A8_UNORM;
   templat.last_level = 0;
   templat.width0 = 32;
   templat.height0 = 32;
   templat.depth0 = 1;
   templat.array_size = 1;
   templat.bind = PIPE_BIND_SAMPLER_VIEW;

   tex = screen->resource_create(screen, &templat);

   if (tex && pattern)
      util_pstipple_update_stipple_texture(pipe, tex, pattern);

   return tex;
}


/**
 * Create sampler view to sample the stipple texture.
 */
struct pipe_sampler_view *
util_pstipple_create_sampler_view(struct pipe_context *pipe,
                                  struct pipe_resource *tex)
{
   struct pipe_sampler_view templat, *sv;

   u_sampler_view_default_template(&templat, tex, tex->format);
   sv = pipe->create_sampler_view(pipe, tex, &templat);

   return sv;
}


/**
 * Create the sampler CSO that'll be used for stippling.
 */
void *
util_pstipple_create_sampler(struct pipe_context *pipe)
{
   struct pipe_sampler_state templat;
   void *s;

   memset(&templat, 0, sizeof(templat));
   templat.wrap_s = PIPE_TEX_WRAP_REPEAT;
   templat.wrap_t = PIPE_TEX_WRAP_REPEAT;
   templat.wrap_r = PIPE_TEX_WRAP_REPEAT;
   templat.min_mip_filter = PIPE_TEX_MIPFILTER_NONE;
   templat.min_img_filter = PIPE_TEX_FILTER_NEAREST;
   templat.mag_img_filter = PIPE_TEX_FILTER_NEAREST;
   templat.min_lod = 0.0f;
   templat.max_lod = 0.0f;

   s = pipe->create_sampler_state(pipe, &templat);
   return s;
}
