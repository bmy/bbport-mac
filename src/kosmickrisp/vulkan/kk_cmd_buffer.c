/*
 * Copyright © 2022 Collabora Ltd. and Red Hat Inc.
 * Copyright 2025 LunarG, Inc.
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: MIT
 */

#include "kk_cmd_buffer.h"

#include "kk_buffer.h"
#include "kk_cmd_pool.h"
#include "kk_descriptor_set_layout.h"
#include "kk_entrypoints.h"
#include "kk_format.h"
#include "kk_image_view.h"

#include "kosmickrisp/bridge/mtl_bridge.h"
#include "kosmickrisp/bridge/mtl_command_buffer.h"
#include "kosmickrisp/bridge/mtl_device.h"
#include "kosmickrisp/bridge/mtl_encoder.h"
#include "kosmickrisp/bridge/vk_to_mtl_map.h"

#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_format.h"
#include "vk_pipeline_layout.h"

static void
kk_descriptor_state_fini(struct kk_cmd_buffer *cmd,
                         struct kk_descriptor_state *desc)
{
   struct kk_cmd_pool *pool = kk_cmd_buffer_pool(cmd);

   for (unsigned i = 0; i < KK_MAX_SETS; i++) {
      vk_free(&pool->vk.alloc, desc->push[i]);
      desc->push[i] = NULL;
   }
}

static void
kk_cmd_release_resources(struct kk_device *dev, struct kk_cmd_buffer *cmd)
{
   struct kk_cmd_pool *pool = kk_cmd_buffer_pool(cmd);

   kk_cmd_release_dynamic_ds_state(cmd);
   kk_descriptor_state_fini(cmd, &cmd->state.gfx.descriptors);
   kk_descriptor_state_fini(cmd, &cmd->state.cs.descriptors);

   kk_cmd_pool_free_bo_list(pool, &cmd->uploader.bos);

   /* Release all BOs used as descriptor buffers for submissions */
   util_dynarray_foreach(&cmd->large_bos, struct kk_bo *, bo) {
      kk_destroy_bo(dev, *bo);
   }
   util_dynarray_clear(&cmd->large_bos);
   util_dynarray_clear(&cmd->post_render_writes);
   util_dynarray_clear(&cmd->ts_resolves);
   util_dynarray_clear(&cmd->ts_stage_map);
}

/* Ends Metal command buffer recording and returns allocator to pool for reuse */
static void
end_recording(struct kk_cmd_buffer *cmd)
{
   if (cmd->vk.state != MESA_VK_COMMAND_BUFFER_STATE_RECORDING)
      return;

   cs_end(cmd);
   mtl_end_command_buffer(cmd->metal.cmd_buf);
   kk_cmd_pool_return_allocator(kk_cmd_buffer_pool(cmd), cmd->metal.allocator);
   cmd->metal.allocator = NULL;
}

static void
kk_destroy_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer)
{
   struct kk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct kk_cmd_buffer, vk);
   struct kk_cmd_pool *pool = kk_cmd_buffer_pool(cmd);

   if (cmd->drawable)
      mtl_release(cmd->drawable);

   /* Ensure closed command buffer for safe returns to pool */
   end_recording(cmd);
   if (cmd->metal.cmd_buf)
      kk_cmd_pool_return_cmd_buf(pool, cmd->metal.cmd_buf);
   cmd->metal.cmd_buf = NULL;

   mtl_release(cmd->argument_table);

   vk_command_buffer_finish(&cmd->vk);
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   kk_cmd_release_resources(dev, cmd);
   util_dynarray_fini(&cmd->large_bos);
   util_dynarray_fini(&cmd->post_render_writes);
   util_dynarray_fini(&cmd->ts_resolves);
   util_dynarray_fini(&cmd->ts_stage_map);

   vk_free(&pool->vk.alloc, cmd);
}

static VkResult
kk_create_cmd_buffer(struct vk_command_pool *vk_pool,
                     VkCommandBufferLevel level,
                     struct vk_command_buffer **cmd_buffer_out)
{
   struct kk_cmd_pool *pool = container_of(vk_pool, struct kk_cmd_pool, vk);
   struct kk_device *dev = kk_cmd_pool_device(pool);
   struct kk_cmd_buffer *cmd;
   VkResult result;

   cmd = vk_zalloc(&pool->vk.alloc, sizeof(*cmd), 8,
                   VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (cmd == NULL)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   result = vk_command_buffer_init_with_params(
      &cmd->vk, &(struct vk_command_buffer_init_params){
                   .pool = &pool->vk,
                   .ops = &kk_cmd_buffer_ops,
                   .level = level,
                   .needs_cmd_queue = true,
                });
   if (result != VK_SUCCESS)
      goto alloc_fail;

   cmd->ts_resolves = UTIL_DYNARRAY_INIT;
   cmd->ts_stage_map = UTIL_DYNARRAY_INIT;
   cmd->post_render_writes = UTIL_DYNARRAY_INIT;

   {
      mtl_argument_table_descriptor *desc = mtl_new_argument_table_descriptor();
      /* Root at 0, samplers at 1 and per draw data at 2 */
      mtl_set_max_buffer_binding_count(desc, 3u);
      cmd->argument_table = mtl_new_argument_table(dev->mtl_handle, desc);
      mtl_set_address(cmd->argument_table, dev->samplers.table.bo->gpu, 1u);
      mtl_release(desc);
   }

   cmd->large_bos = UTIL_DYNARRAY_INIT;

   cmd->vk.dynamic_graphics_state.vi = &cmd->state.gfx._dynamic_vi;
   cmd->vk.dynamic_graphics_state.ms.sample_locations =
      &cmd->state.gfx._dynamic_sl;

   list_inithead(&cmd->uploader.bos);

   *cmd_buffer_out = &cmd->vk;

   return VK_SUCCESS;

alloc_fail:
   vk_free(&pool->vk.alloc, cmd);
   return result;
}

static void
kk_reset_cmd_buffer_internal(struct kk_cmd_buffer *cmd)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);

   /* Ensure closed command buffer so we can start it again */
   end_recording(cmd);
   kk_cmd_release_resources(dev, cmd);

   cmd->uploader.bo = NULL;
   cmd->uploader.offset = 0;

   memset(&cmd->state, 0, sizeof(cmd->state));
   cmd->uses_heap = false;
}

static void
kk_reset_cmd_buffer(struct vk_command_buffer *vk_cmd_buffer,
                    UNUSED VkCommandBufferResetFlags flags)
{
   struct kk_cmd_buffer *cmd =
      container_of(vk_cmd_buffer, struct kk_cmd_buffer, vk);

   kk_reset_cmd_buffer_internal(cmd);
   vk_command_buffer_reset(&cmd->vk);
   cmd->submitted = false;
   cmd->one_time_submit = false;
}

const struct vk_command_buffer_ops kk_cmd_buffer_ops = {
   .create = kk_create_cmd_buffer,
   .reset = kk_reset_cmd_buffer,
   .destroy = kk_destroy_cmd_buffer,
};

VKAPI_ATTR VkResult VKAPI_CALL
kk_BeginCommandBuffer(VkCommandBuffer commandBuffer,
                      const VkCommandBufferBeginInfo *pBeginInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   struct kk_cmd_pool *pool = kk_cmd_buffer_pool(cmd);

   /* If this is the first time starting the command buffer allocate Metal
    * resources */
   if (cmd->metal.cmd_buf == NULL) {
      cmd->metal.cmd_buf = kk_cmd_pool_get_cmd_buf(pool);

      if (cmd->metal.cmd_buf == NULL)
         goto fail_cmd;

      if (cmd->vk.base.object_name)
         mtl_command_buffer_set_label(cmd->metal.cmd_buf,
                                      cmd->vk.base.object_name);
   }

   vk_command_buffer_begin(&cmd->vk, pBeginInfo);
   cmd->one_time_submit =
      pBeginInfo->flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

   /* vk_command_buffer_begin will reset the command buffer meaning the
    * allocator may be returned to the pool. Request a new one. */
   if (cmd->metal.allocator == NULL) {
      cmd->metal.allocator = kk_cmd_pool_get_allocator(pool);

      if (cmd->metal.allocator == NULL)
         goto fail_allocator;
   }
   mtl_begin_command_buffer(cmd->metal.cmd_buf, cmd->metal.allocator);

   return VK_SUCCESS;

fail_allocator:
   kk_cmd_pool_return_cmd_buf(kk_cmd_buffer_pool(cmd), cmd->metal.cmd_buf);
   cmd->metal.cmd_buf = NULL;
fail_cmd:
   return VK_ERROR_OUT_OF_DEVICE_MEMORY;
}

VKAPI_ATTR VkResult VKAPI_CALL
kk_EndCommandBuffer(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   end_recording(cmd);

   return vk_command_buffer_end(&cmd->vk);
}

static void
kk_encoder_update_debug(struct kk_cmd_buffer *cmd, mtl_command_encoder *encoder)
{
   /* Since there are many Metal command buffers and encoders for each Vulkan
    * command buffer, we need to copy debug state from Vulkan to Metal when
    * new Metal objects are created. */
   if (cmd->vk.base.object_name)
      mtl_encoder_set_label(encoder, cmd->vk.base.object_name);

   util_dynarray_foreach(&cmd->vk.labels, VkDebugUtilsLabelEXT, label)
      mtl_encoder_push_debug_group(encoder, label->pLabelName);
}

void
cs_start_render(struct kk_cmd_buffer *cmd)
{
   struct kk_graphics_state *state = &cmd->state.gfx;
   uint32_t view_mask = state->render.view_mask;
   assert(state->render_pass_descriptor);

   /* Ensure no other active encoder, aka compute */
   cs_end(cmd);

   cmd->metal.render = mtl_new_render_command_encoder_with_descriptor(
      cmd->metal.cmd_buf, state->render_pass_descriptor);

   kk_encoder_update_debug(cmd, cmd->metal.render);
   /* Starting a new render pass means we already flushed and no barrier is
    * needed. */
   state->render.write_available = false;
   state->render.ds_write_available = false;
   state->render.storage_write_available = false;

   uint32_t layer_ids[KK_MAX_MULTIVIEW_VIEW_COUNT] = {};
   uint32_t count = 0u;
   u_foreach_bit(id, view_mask)
      layer_ids[count++] = id;
   if (view_mask == 0u) {
      layer_ids[count++] = 0;
   }
   mtl_set_vertex_amplification_count(cmd->metal.render, layer_ids, count);

   /* Argument table won't ever change */
   mtl_render_set_argument_table(
      cmd->metal.render, cmd->argument_table,
      MTL_RENDER_STAGE_VERTEX | MTL_RENDER_STAGE_FRAGMENT);

   kk_cmd_buffer_dirty_all_gfx(cmd);
}

mtl_render_encoder *
cs_get_render(struct kk_cmd_buffer *cmd)
{
   struct kk_graphics_state *gfx = &cmd->state.gfx;

   if (gfx->need_to_start_render_pass) {
      gfx->render.samples = gfx->pipeline_sample_count;
      mtl_render_pass_descriptor_set_default_raster_sample_count(
         cmd->state.gfx.render_pass_descriptor, gfx->render.samples);
      gfx->need_to_start_render_pass = false;
      cs_start_render(cmd);
   }

   return cmd->metal.render;
}

mtl_compute_encoder *
cs_get_compute(struct kk_cmd_buffer *cmd)
{
   /* If we are inside render, we need to force a restart later */
   if (cmd->metal.render) {
      kk_apply_attachment_store_ops(cmd, true);
      cs_end(cmd);
      kk_cmd_buffer_dirty_all_gfx(cmd);
      cmd->state.gfx.need_to_start_render_pass = true;
   }

   if (cmd->metal.compute == NULL) {
      cmd->metal.compute = mtl_new_compute_command_encoder(cmd->metal.cmd_buf);
      mtl_compute_set_argument_table(cmd->metal.compute, cmd->argument_table);
      kk_encoder_update_debug(cmd, cmd->metal.compute);
   }

   return cmd->metal.compute;
}

static void
end_encoder(struct kk_cmd_buffer *cmd, mtl_command_encoder *encoder)
{
   /* TODO_KOSMICKRISP This is probably overkill */
   mtl_barrier_after_stages(encoder, MTL_STAGE_ALL, MTL_STAGE_ALL);
   mtl_end_encoding(encoder);
   mtl_release(encoder);

   /* Fold the pending timestamp counter-heap resolves into `cmd_buf` */
   util_dynarray_foreach(&cmd->ts_resolves, struct kk_ts_resolve, r) {
      mtl_command_resolve_counter_heap(cmd->metal.cmd_buf, r->heap, r->index,
                                       1u, r->dst_addr);
   }

   util_dynarray_clear(&cmd->ts_resolves);
}

static void
flush_post_render_writes(struct kk_cmd_buffer *cmd)
{
   assert(cmd->metal.render == NULL);

   if (!util_dynarray_num_elements(&cmd->post_render_writes,
                                   struct libkk_imm_write))
      return;

   util_dynarray_foreach(&cmd->post_render_writes, struct libkk_imm_write, w) {
      libkk_write_u32(cmd, kk_grid_1d(1), true, (uint64_t)w->address, w->value);
   }

   util_dynarray_clear(&cmd->post_render_writes);
}

void
cs_end(struct kk_cmd_buffer *cmd)
{
   assert(cmd);

   /* Render first as it may open a compute encoder for post render writes. */
   if (cmd->metal.render) {
      end_encoder(cmd, cmd->metal.render);
      cmd->metal.render = NULL;

      /* The stage map is only valid for the current render encoder */
      util_dynarray_clear(&cmd->ts_stage_map);
      flush_post_render_writes(cmd);
   }

   if (cmd->metal.compute) {
      end_encoder(cmd, cmd->metal.compute);
      cmd->metal.compute = NULL;
   }
}

void
kk_cmd_bind_root_to_argument_table(struct kk_cmd_buffer *cmd, uint64_t addr)
{
   mtl_set_address(cmd->argument_table, addr, 0u);
   cmd->state.root_addr = addr;
}

/* Returns true if a render pass split is required. The main cases are:
 * - Texture write -> barrier -> attachment read
 * - Attachment write -> barrier -> texture read
 * - Depth/stencil write -> barrier -> depth/stencil read (Metal limitation)
 *
 * For color attachment write -> barrier -> color attachment read there is no
 * need to split the render pass.
 *
 * TODO_KOSMICKRISP Potential improvement would be to track where the
 * attachments lie in memory and check against the buffers/images provided in
 * the barrier to avoid splitting a render pass that will never overlap the
 * attachment memory. Meaning there was no write to the attachment and therefore
 * no split is needed.
 */
static bool
kk_barrier_requires_encoder_split(struct kk_cmd_buffer *cmd,
                                  const VkDependencyInfo *dep)
{
   const VkAccessFlags2 texture_read_access =
      VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT;
   const VkAccessFlags2 any_write_access =
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
      VK_ACCESS_2_MEMORY_WRITE_BIT;
   const VkAccessFlags2 ds_write_access =
      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_2_MEMORY_WRITE_BIT;
   const VkAccessFlags2 storage_write_access =
      VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
      VK_ACCESS_2_MEMORY_WRITE_BIT;
   struct kk_rendering_state *render = &cmd->state.gfx.render;
   const bool has_ds =
      render->depth_att.iview != NULL || render->stencil_att.iview != NULL;

   VkAccessFlags2 src = 0, dst = 0;
   for (uint32_t i = 0; i < dep->memoryBarrierCount; i++) {
      src |= dep->pMemoryBarriers[i].srcAccessMask;
      dst |= dep->pMemoryBarriers[i].dstAccessMask;
   }
   for (uint32_t i = 0; i < dep->bufferMemoryBarrierCount; i++) {
      src |= dep->pBufferMemoryBarriers[i].srcAccessMask;
      dst |= dep->pBufferMemoryBarriers[i].dstAccessMask;
   }
   for (uint32_t i = 0; i < dep->imageMemoryBarrierCount; i++) {
      src |= dep->pImageMemoryBarriers[i].srcAccessMask;
      dst |= dep->pImageMemoryBarriers[i].dstAccessMask;
   }

   if (src & any_write_access)
      render->write_available = true;
   if (has_ds && (src & ds_write_access))
      render->ds_write_available = true;
   if (src & storage_write_access)
      render->storage_write_available = true;

   if (render->write_available && (dst & texture_read_access))
      return true;

   return (render->ds_write_available || render->storage_write_available) &&
          (dst & VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPipelineBarrier2(VkCommandBuffer commandBuffer,
                       const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   /* TODO_KOSMICKRISP Lighten barriers according to the actual requested
    * barrier. To take advantage of this we need to remove the chaining of
    * encoders. */
   if (cmd->metal.render) {
      /* Multisample attachments require render pass split always. Then based on
       * the barrier and if we are using depth/stencil or not, we may have to
       * break the render pass. See comment in kk_barrier_requires_encoder_split
       */
      if (cmd->state.gfx.render.samples > 1 ||
          kk_barrier_requires_encoder_split(cmd, pDependencyInfo)) {
         kk_apply_attachment_store_ops(cmd, true);
         cs_end(cmd);
         cs_start_render(cmd);
      } else
         mtl_barrier_after_encoder_stages(cmd->metal.render, MTL_STAGE_VERTEX,
                                          MTL_STAGE_FRAGMENT);
   } else if (cmd->metal.compute) {
      /* We chain encoders, so an intra-encoder barrier is enough here:
       * no need to tear down and recreate the encoder.
       */
      mtl_barrier_after_encoder_stages(cmd->metal.compute,
                                       MTL_STAGE_DISPATCH | MTL_STAGE_BLIT,
                                       MTL_STAGE_DISPATCH | MTL_STAGE_BLIT);
   }
}

static void
kk_bind_descriptor_sets(struct kk_descriptor_state *desc,
                        const VkBindDescriptorSetsInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   /* From the Vulkan 1.3.275 spec:
    *
    *    "When binding a descriptor set (see Descriptor Set Binding) to
    *    set number N...
    *
    *    If, additionally, the previously bound descriptor set for set
    *    N was bound using a pipeline layout not compatible for set N,
    *    then all bindings in sets numbered greater than N are
    *    disturbed."
    *
    * This means that, if some earlier set gets bound in such a way that
    * it changes set_dynamic_buffer_start[s], this binding is implicitly
    * invalidated.
    */
   uint8_t dyn_buffer_start =
      pipeline_layout->dynamic_descriptor_offset[info->firstSet];

   uint32_t next_dyn_offset = 0;
   for (uint32_t i = 0; i < info->descriptorSetCount; ++i) {
      unsigned s = i + info->firstSet;
      VK_FROM_HANDLE(kk_descriptor_set, set, info->pDescriptorSets[i]);

      if (desc->sets[s] != set) {
         if (set != NULL) {
            desc->root.sets[s] = set->addr;
            desc->set_sizes[s] = set->size;
         } else {
            desc->root.sets[s] = 0;
            desc->set_sizes[s] = 0;
         }
         desc->sets[s] = set;

         /* Binding descriptors invalidates push descriptors */
         desc->push_dirty &= ~BITFIELD_BIT(s);
      }

      if (pipeline_layout->set_layouts[s] != NULL) {
         const struct kk_descriptor_set_layout *set_layout =
            vk_to_kk_descriptor_set_layout(pipeline_layout->set_layouts[s]);

         if (set != NULL && set_layout->vk.dynamic_descriptor_count > 0) {
            for (uint32_t j = 0; j < set_layout->vk.dynamic_descriptor_count;
                 j++) {
               struct kk_buffer_address addr = set->dynamic_buffers[j];
               addr.base_addr += info->pDynamicOffsets[next_dyn_offset + j];
               desc->root.dynamic_buffers[dyn_buffer_start + j] = addr;
            }
            next_dyn_offset += set->layout->vk.dynamic_descriptor_count;
         }

         dyn_buffer_start += set_layout->vk.dynamic_descriptor_count;
      } else {
         assert(set == NULL);
      }
   }
   assert(dyn_buffer_start <= KK_MAX_DYNAMIC_BUFFERS);
   assert(next_dyn_offset <= info->dynamicOffsetCount);

   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBindDescriptorSets2KHR(
   VkCommandBuffer commandBuffer,
   const VkBindDescriptorSetsInfoKHR *pBindDescriptorSetsInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      kk_bind_descriptor_sets(&cmd->state.gfx.descriptors,
                              pBindDescriptorSetsInfo);
   }

   if (pBindDescriptorSetsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      kk_bind_descriptor_sets(&cmd->state.cs.descriptors,
                              pBindDescriptorSetsInfo);
   }
}

static struct kk_push_descriptor_set *
kk_cmd_push_descriptors(struct kk_cmd_buffer *cmd,
                        struct kk_descriptor_state *desc,
                        struct kk_descriptor_set_layout *set_layout,
                        uint32_t set)
{
   assert(set < KK_MAX_SETS);
   if (unlikely(desc->push[set] == NULL)) {
      size_t size = sizeof(*desc->push[set]);
      desc->push[set] = vk_zalloc(&cmd->vk.pool->alloc, size, 8,
                                  VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (unlikely(desc->push[set] == NULL)) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return NULL;
      }
   }

   /* Pushing descriptors replaces whatever sets are bound */
   desc->push[set]->layout = set_layout;
   desc->sets[set] = NULL;
   desc->push_dirty |= BITFIELD_BIT(set);

   return desc->push[set];
}

static void
kk_push_descriptor_set(struct kk_cmd_buffer *cmd,
                       struct kk_descriptor_state *desc,
                       const VkPushDescriptorSetInfoKHR *info)
{
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout, info->layout);

   struct kk_descriptor_set_layout *set_layout =
      vk_to_kk_descriptor_set_layout(pipeline_layout->set_layouts[info->set]);

   struct kk_push_descriptor_set *push_set =
      kk_cmd_push_descriptors(cmd, desc, set_layout, info->set);
   if (unlikely(push_set == NULL))
      return;

   kk_push_descriptor_set_update(push_set, info->descriptorWriteCount,
                                 info->pDescriptorWrites);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPushDescriptorSet2KHR(
   VkCommandBuffer commandBuffer,
   const VkPushDescriptorSetInfoKHR *pPushDescriptorSetInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   if (pPushDescriptorSetInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
      kk_push_descriptor_set(cmd, &cmd->state.gfx.descriptors,
                             pPushDescriptorSetInfo);
   }

   if (pPushDescriptorSetInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
      kk_push_descriptor_set(cmd, &cmd->state.cs.descriptors,
                             pPushDescriptorSetInfo);
   }
}

static void
kk_push_constants(UNUSED struct kk_cmd_buffer *cmd,
                  struct kk_descriptor_state *desc,
                  const VkPushConstantsInfoKHR *info)
{
   memcpy(desc->root.push + info->offset, info->pValues, info->size);
   desc->root_dirty = true;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPushConstants2KHR(VkCommandBuffer commandBuffer,
                        const VkPushConstantsInfoKHR *pPushConstantsInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS)
      kk_push_constants(cmd, &cmd->state.gfx.descriptors, pPushConstantsInfo);

   if (pPushConstantsInfo->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      kk_push_constants(cmd, &cmd->state.cs.descriptors, pPushConstantsInfo);
}

void
kk_cmd_release_dynamic_ds_state(struct kk_cmd_buffer *cmd)
{
   if (cmd->state.gfx.is_depth_stencil_dynamic &&
       cmd->state.gfx.depth_stencil_state)
      mtl_release(cmd->state.gfx.depth_stencil_state);
   cmd->state.gfx.depth_stencil_state = NULL;
}

static VkResult
kk_cmd_buffer_alloc_bo(struct kk_cmd_buffer *cmd, struct kk_cmd_bo **bo_out)
{
   VkResult result = kk_cmd_pool_alloc_bo(kk_cmd_buffer_pool(cmd), bo_out);
   if (result != VK_SUCCESS)
      return result;

   list_addtail(&(*bo_out)->link, &cmd->uploader.bos);
   return VK_SUCCESS;
}

struct kk_ptr
kk_pool_alloc(struct kk_cmd_buffer *cmd, uint32_t size_B, uint32_t alignment_B)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_uploader *uploader = &cmd->uploader;

   /* Specially handle large allocations owned by the command buffer, e.g. used
    * for statically allocated vertex output buffers with geometry shaders.
    */
   if (size_B > KK_CMD_BO_SIZE) {
      struct kk_bo *buffer = NULL;
      const VkResult result =
         kk_alloc_bo(dev, &cmd->vk.base, size_B, alignment_B, &buffer);
      if (result != VK_SUCCESS) {
         vk_command_buffer_set_error(&cmd->vk, result);
         return (struct kk_ptr){0};
      }
      util_dynarray_append(&cmd->large_bos, buffer);

      return (struct kk_ptr){
         .gpu = buffer->gpu,
         .cpu = buffer->cpu,

         .buffer = buffer->map,
         .offset = 0u,
      };
   }

   assert(size_B <= KK_CMD_BO_SIZE);
   assert(alignment_B > 0);

   const uint32_t offset = align(uploader->offset, alignment_B);

   assert(offset <= KK_CMD_BO_SIZE);
   if (uploader->bo != NULL && size_B <= KK_CMD_BO_SIZE - offset) {
      uploader->offset = offset + size_B;

      return (struct kk_ptr){
         .gpu = uploader->bo->gpu + offset,
         .cpu = uploader->bo->cpu + offset,

         .buffer = uploader->bo->map,
         .offset = offset,
      };
   }

   struct kk_cmd_bo *bo;
   const VkResult result = kk_cmd_buffer_alloc_bo(cmd, &bo);
   if (unlikely(result != VK_SUCCESS)) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return (struct kk_ptr){0};
   }

   /* Pick whichever of the current upload BO and the new BO will have more
    * room left to be the BO for the next upload.  If our upload size is
    * bigger than the old offset, we're better off burning the whole new
    * upload BO on this one allocation and continuing on the current upload
    * BO.
    */
   if (uploader->bo == NULL || size_B < uploader->offset) {
      uploader->bo = bo->bo;
      uploader->offset = size_B;
   }

   return (struct kk_ptr){
      .gpu = bo->bo->gpu,
      .cpu = bo->bo->cpu,

      .buffer = bo->bo->map,
      .offset = 0u,
   };
}

struct kk_ptr
kk_pool_upload(struct kk_cmd_buffer *cmd, const void *data, uint32_t size,
               uint32_t alignment)
{
   struct kk_ptr T = kk_pool_alloc(cmd, size, alignment);
   if (unlikely(T.cpu == NULL))
      return (struct kk_ptr){0};

   memcpy(T.cpu, data, size);
   return T;
}

uint64_t
kk_upload_descriptor_root(struct kk_cmd_buffer *cmd,
                          VkPipelineBindPoint bind_point)
{
   struct kk_descriptor_state *desc = kk_get_descriptors_state(cmd, bind_point);
   struct kk_root_descriptor_table *root = &desc->root;
   struct kk_ptr root_ptr = kk_pool_alloc(cmd, sizeof(*root), 8u);
   if (unlikely(!root_ptr.gpu))
      return 0u;

   root->addr = root_ptr.gpu;

   memcpy(root_ptr.cpu, root, sizeof(*root));
   desc->root_dirty = false;

   return root_ptr.gpu;
}

void
kk_cmd_buffer_flush_push_descriptors(struct kk_cmd_buffer *cmd,
                                     struct kk_descriptor_state *desc)
{
   u_foreach_bit(set_idx, desc->push_dirty) {
      struct kk_push_descriptor_set *push_set = desc->push[set_idx];
      struct kk_ptr push_gpu = kk_pool_upload(
         cmd, push_set->data, sizeof(push_set->data), KK_MIN_UBO_ALIGNMENT);
      if (unlikely(!push_gpu.gpu))
         return;

      desc->root.sets[set_idx] = push_gpu.gpu;
      desc->set_sizes[set_idx] = sizeof(push_set->data);
   }

   desc->root_dirty = true;
   desc->push_dirty = 0;
}

void
kk_dispatch_precomp(struct kk_cmd_buffer *cmd, struct kk_grid grid,
                    bool pre_gfx, enum libkk_program idx, void *data,
                    size_t data_size)
{
   struct kk_device *dev = kk_cmd_buffer_device(cmd);
   struct kk_precompiled_shader *prog = &dev->precompiled_cache.shaders[idx];

   mtl_compute_encoder *encoder = cs_get_compute(cmd);
   mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_DISPATCH,
                                    MTL_STAGE_DISPATCH);

   struct kk_ptr data_gpu = kk_pool_upload(cmd, data, data_size, 8u);
   if (unlikely(!data_gpu.gpu))
      return;

   mtl_set_address(cmd->argument_table, data_gpu.gpu, 0u);
   mtl_compute_set_pipeline_state(encoder, prog->pipeline);

   struct mtl_size local_size = {
      .x = prog->info.workgroup_size[0],
      .y = prog->info.workgroup_size[1],
      .z = prog->info.workgroup_size[2],
   };

   if (grid.mode == KK_GRID_DIRECT)
      mtl_dispatch_threads(encoder, grid.size, local_size);
   else
      mtl_dispatch_threadgroups_with_indirect_buffer(encoder, grid.addr,
                                                     local_size);
   mtl_barrier_after_encoder_stages(encoder, MTL_STAGE_DISPATCH,
                                    MTL_STAGE_DISPATCH);

   /* Rebind the existing root. */
   mtl_set_address(cmd->argument_table, cmd->state.root_addr, 0u);
}

void
kk_cmd_write(struct kk_cmd_buffer *cmd, struct libkk_imm_write write)
{
   /* If we are mid render, queue the write for when we are done */
   if (cmd->metal.render)
      util_dynarray_append(&cmd->post_render_writes, write);
   else
      libkk_write_u32(cmd, kk_grid_1d(1), true, write.address, write.value);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdPushDescriptorSetWithTemplate2KHR(
   VkCommandBuffer commandBuffer, const VkPushDescriptorSetWithTemplateInfoKHR
                                     *pPushDescriptorSetWithTemplateInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);
   VK_FROM_HANDLE(vk_descriptor_update_template, template,
                  pPushDescriptorSetWithTemplateInfo->descriptorUpdateTemplate);
   VK_FROM_HANDLE(vk_pipeline_layout, pipeline_layout,
                  pPushDescriptorSetWithTemplateInfo->layout);

   struct kk_descriptor_state *desc =
      kk_get_descriptors_state(cmd, template->bind_point);
   struct kk_descriptor_set_layout *set_layout = vk_to_kk_descriptor_set_layout(
      pipeline_layout->set_layouts[pPushDescriptorSetWithTemplateInfo->set]);
   struct kk_push_descriptor_set *push_set = kk_cmd_push_descriptors(
      cmd, desc, set_layout, pPushDescriptorSetWithTemplateInfo->set);
   if (unlikely(push_set == NULL))
      return;

   kk_push_descriptor_set_update_template(
      push_set, set_layout, template,
      pPushDescriptorSetWithTemplateInfo->pData);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdBeginConditionalRendering2EXT(
   VkCommandBuffer commandBuffer,
   const VkConditionalRenderingBeginInfo2EXT *pConditionalRenderingBegin)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   cmd->state.cond_render.address =
      pConditionalRenderingBegin->addressRange.address;
   cmd->state.cond_render.inverted = pConditionalRenderingBegin->flags &
                                     VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT;
   cmd->state.cond_render.enabled = true;
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdEndConditionalRenderingEXT(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, commandBuffer);

   cmd->state.cond_render.enabled = false;
}

static enum mtl_store_action
kk_get_attachment_store_op(const struct kk_attachment *attachment,
                           bool force_store,
                           bool rendering_to_whole_framebuffer,
                           VkImageAspectFlags aspect)
{
   bool renderpass_resolve = kk_attachment_do_renderpass_resolve(
      attachment, rendering_to_whole_framebuffer, aspect);
   force_store |=
      attachment->resolve_mode != VK_RESOLVE_MODE_NONE && !renderpass_resolve;

   force_store |= (attachment->load_op == VK_ATTACHMENT_LOAD_OP_LOAD ||
                   attachment->load_op == VK_ATTACHMENT_LOAD_OP_NONE) &&
                  attachment->store_op == VK_ATTACHMENT_STORE_OP_NONE;

   enum mtl_store_action store_action =
      force_store
         ? MTL_STORE_ACTION_STORE
         : vk_attachment_store_op_to_mtl_store_action(attachment->store_op);

   if (renderpass_resolve) {
      if (store_action == MTL_STORE_ACTION_STORE)
         store_action = MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE;
      else if (store_action == MTL_STORE_ACTION_DONT_CARE)
         store_action = MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
   }

   return store_action;
}

void
kk_apply_attachment_store_ops(struct kk_cmd_buffer *cmd, bool force_store)
{
   if (!cmd->metal.render)
      return;

   struct kk_rendering_state *render = &cmd->state.gfx.render;
   mtl_render_encoder *encoder = cs_get_render(cmd);

   force_store |= render->force_attachment_store;

   for (uint32_t i = 0; i < render->color_att_count; i++) {
      uint32_t logical_index = cmd->state.gfx.render.color_map[i];

      if (render->color_att[i].iview &&
          logical_index != MESA_VK_ATTACHMENT_UNUSED) {
         enum mtl_store_action store_action = kk_get_attachment_store_op(
            &render->color_att[i], force_store,
            render->rendering_to_whole_framebuffer, VK_IMAGE_ASPECT_COLOR_BIT);

         mtl_render_set_color_store_action(encoder, store_action,
                                           logical_index);
      }
   }
   enum mtl_store_action depth_store_action = MTL_STORE_ACTION_UNKNOWN;
   enum mtl_store_action stencil_store_action = MTL_STORE_ACTION_UNKNOWN;
   if (render->depth_att.iview) {
      depth_store_action = kk_get_attachment_store_op(
         &render->depth_att, force_store,
         render->rendering_to_whole_framebuffer, VK_IMAGE_ASPECT_DEPTH_BIT);
   }
   if (render->stencil_att.iview) {
      stencil_store_action = kk_get_attachment_store_op(
         &render->stencil_att, force_store,
         render->rendering_to_whole_framebuffer, VK_IMAGE_ASPECT_STENCIL_BIT);
   }
   bool ds_resolve_dont_care =
      depth_store_action == MTL_STORE_ACTION_MULTISAMPLE_RESOLVE ||
      stencil_store_action == MTL_STORE_ACTION_MULTISAMPLE_RESOLVE;
   bool ds_store = depth_store_action == MTL_STORE_ACTION_STORE ||
                   stencil_store_action == MTL_STORE_ACTION_STORE;
   /* Metal does not allow combining MTL_STORE_ACTION_MULTISAMPLE_RESOLVE
    * (so resolve but don't care about the MS data) for depth/stencil with
    * MTL_STORE_ACTION_STORE for the other one of depth/stencil regardless of
    * whether it's the same texture or different ones. */
   bool ds_force_ms_store = ds_resolve_dont_care && ds_store;

   if (render->depth_att.iview) {
      if (depth_store_action == MTL_STORE_ACTION_MULTISAMPLE_RESOLVE &&
          ds_force_ms_store)
         depth_store_action = MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE;

      mtl_render_set_depth_store_action(encoder, depth_store_action);
   }
   if (render->stencil_att.iview) {
      if (stencil_store_action == MTL_STORE_ACTION_MULTISAMPLE_RESOLVE &&
          ds_force_ms_store)
         stencil_store_action = MTL_STORE_ACTION_STORE_AND_MULTISAMPLE_RESOLVE;

      mtl_render_set_stencil_store_action(encoder, stencil_store_action);
   }
}

bool
kk_attachment_do_renderpass_resolve(const struct kk_attachment *attachment,
                                    bool rendering_to_whole_framebuffer,
                                    VkImageAspectFlags aspect)
{
   if (!rendering_to_whole_framebuffer)
      return false;

   if (attachment->resolve_mode == VK_RESOLVE_MODE_NONE ||
       !attachment->resolve_iview || !attachment->iview)
      return false;

   /* There appears to be a synchronization bug in Metal.
    * Producer barriers do not synchronize depth render pass resolves
    * properly, even when it's STAGE_ALL -> STAGE_ALL.
    * Force meta resolves for depth and stencil for now. */
   if (!(attachment->resolve_iview->vk.aspects & VK_IMAGE_ASPECT_COLOR_BIT))
      return false;

   /* Dynamic rendering doesn't allow different formats but the Mesa Vulkan
    * runtime does it anyway when emulating legacy render passes.
    * Just fall back to meta resolves instead of creating views on the fly. */
   if (attachment->vk_format != attachment->resolve_iview->vk.view_format)
      return false;

   if (attachment->flags &
       VK_RENDERING_ATTACHMENT_RESOLVE_SKIP_TRANSFER_FUNCTION_BIT_KHR)
      return false;

   enum pipe_format p_format = vk_format_to_pipe_format(attachment->vk_format);
   const struct kk_va_format *format_features = kk_get_va_format(p_format);

   if (!format_features->resolve)
      return false;

   /* From the spec:
    * The aspectMask of any image view specified for pDepthAttachment or
    * pStencilAttachment is ignored. Instead, depth attachments are
    * automatically treated as if VK_IMAGE_ASPECT_DEPTH_BIT was specified
    * for their aspect masks, and stencil attachments are automatically
    * treated as if VK_IMAGE_ASPECT_STENCIL_BIT was specified for their
    * aspect masks.
    *
    * That's why we have an extra parameter for what we use the attachment for.
    */

   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      return attachment->resolve_mode == VK_RESOLVE_MODE_SAMPLE_ZERO_BIT ||
             attachment->resolve_mode == VK_RESOLVE_MODE_MIN_BIT ||
             attachment->resolve_mode == VK_RESOLVE_MODE_MAX_BIT;

   } else if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
      return attachment->resolve_mode == VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
   } else {
      return attachment->resolve_mode == VK_RESOLVE_MODE_AVERAGE_BIT;
   }
}

/* VK_AMD_buffer_marker */
VKAPI_ATTR void VKAPI_CALL
kk_CmdWriteMarkerToMemoryAMD(VkCommandBuffer commandBuffer,
                             const VkMemoryMarkerInfoAMD *pInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd_buffer, commandBuffer);
   struct libkk_imm_write write;

   /* If we are inside a compute encoder, we can directly encode the write but
    * we need the barrier. If we are mid render, we will queue it for when it is
    * closed. */
   if (cmd_buffer->metal.compute) {
      mtl_barrier_after_encoder_stages(cmd_buffer->metal.compute,
                                       MTL_STAGE_DISPATCH | MTL_STAGE_BLIT,
                                       MTL_STAGE_DISPATCH | MTL_STAGE_BLIT);
   }

   write.value = pInfo->marker;
   write.address = pInfo->dstRange.address;
   kk_cmd_write(cmd_buffer, write);
}

void
kk_cmd_buffer_set_label(struct kk_cmd_buffer *cmd, const char *label)
{
   if (cmd->metal.cmd_buf)
      mtl_command_buffer_set_label(cmd->metal.cmd_buf, label);

   if (cmd->metal.compute)
      mtl_encoder_set_label(cmd->metal.compute, label);

   if (cmd->metal.render)
      mtl_encoder_set_label(cmd->metal.render, label);
}

/* VK_EXT_debug_utils */
VKAPI_ATTR void VKAPI_CALL
kk_CmdBeginDebugUtilsLabelEXT(VkCommandBuffer _commandBuffer,
                              const VkDebugUtilsLabelEXT *pLabelInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, _commandBuffer);

   vk_common_CmdBeginDebugUtilsLabelEXT(_commandBuffer, pLabelInfo);

   if (cmd->metal.render)
      mtl_encoder_push_debug_group(cmd->metal.render, pLabelInfo->pLabelName);

   if (cmd->metal.compute)
      mtl_encoder_push_debug_group(cmd->metal.compute, pLabelInfo->pLabelName);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdEndDebugUtilsLabelEXT(VkCommandBuffer _commandBuffer)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, _commandBuffer);
   vk_common_CmdEndDebugUtilsLabelEXT(_commandBuffer);

   if (cmd->metal.render)
      mtl_encoder_pop_debug_group(cmd->metal.render);

   if (cmd->metal.compute)
      mtl_encoder_pop_debug_group(cmd->metal.compute);
}

VKAPI_ATTR void VKAPI_CALL
kk_CmdInsertDebugUtilsLabelEXT(VkCommandBuffer _commandBuffer,
                               const VkDebugUtilsLabelEXT *pLabelInfo)
{
   VK_FROM_HANDLE(kk_cmd_buffer, cmd, _commandBuffer);

   /* We purposely don't call the common implementation here. It makes
    * debug regions that last until the next vkCmdInsertDebugUtilsLabelEXT()
    * or vkCmdBeginDebugUtilsLabelEXT() call.
    * The Metal debug signpost does not need this, and it interferes
    * with propagating the begin/end debug regions to all of the
    * Metal command buffers and encoders. */
   if (cmd->metal.render)
      mtl_encoder_insert_debug_signpost(cmd->metal.render,
                                        pLabelInfo->pLabelName);

   if (cmd->metal.compute)
      mtl_encoder_insert_debug_signpost(cmd->metal.compute,
                                        pLabelInfo->pLabelName);
}
