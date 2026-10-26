// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal command path's frame encoder (docs/MACOS_METAL_RENDERER.md, stage 3b).
// The command layer (mtl_layer.cpp) hands it every command of a command buffer it owns
// (BB_METAL_FRAME=1); it encodes what it can in classic Metal command buffers on a queue of its
// own, and records the rest into KosmicKrisp command buffers of its own, replaying the state they
// need (the "fallback"), so that the picture stays whole while the Metal coverage grows. At
// submission the segments go to the two queues in recording order, ordered between them by shared
// events (docs/MACOS_METAL_RENDERER.md, stage 5). After Wooloomooloo2/bloodborne_mac_port's
// mtl_frame.mm (78b59f5, GPL-2.0-or-later), on KosmicKrisp's objects. Plain C++ interface; the implementation (mtl_frame.mm) is Objective-C++ and is
// built with BB_METAL_BACKEND only.

#pragma once

#include <cstdint>

#include "vulkan/vk_common.h"

namespace MetalFrame {

struct Owned;

/// Sets the encoder up with the driver's own entry points (`driver_proc`: KosmicKrisp's
/// vkGetDeviceProcAddr, not the layer's), right after the device is made: our queue with
/// KosmicKrisp's residency set (patch 0009), the shared events between the queues. False (logged)
/// when something is missing: the layer then stays a counting one.
bool Init(VkDevice device, PFN_vkGetDeviceProcAddr driver_proc);
/// bbport: before vkDestroyDevice: KosmicKrisp's residency set off our queue (the device
/// releases it; patch 0009's contract).
void Shutdown();

/// A command buffer begins (vkBeginCommandBuffer, after the driver's): its encoding state.
Owned* Begin(VkCommandBuffer command_buffer);
/// vkEndCommandBuffer (before the driver's).
void End(Owned& o);
/// The command buffer was reset or begun again without a submission.
void Discard(Owned* o);

/// One submission batch, normalized (vkQueueSubmit and vkQueueSubmit2 alike): `owned[i]` is the
/// encoding state of `command_buffers[i]`, null for one the layer does not own.
struct Batch {
    const VkSemaphoreSubmitInfo* waits{};
    uint32_t wait_count{};
    const VkCommandBuffer* command_buffers{};
    Owned* const* owned{};
    uint32_t command_buffer_count{};
    const VkSemaphoreSubmitInfo* signals{};
    uint32_t signal_count{};
};

/// Submits the batches in order: per batch its waits (a driver submission), then its command
/// buffers (an owned one as its segments: Metal command buffers committed to our queue,
/// KosmicKrisp ones submitted through the driver's vkQueueSubmit2), then its signals; the fence
/// with the last, and a driver submission after our last command buffer. Consumes the owned
/// states. Called with the queue's submissions serialized (the renderer's lock).
VkResult Submit(VkQueue queue, const Batch* batches, uint32_t count, VkFence fence);

// Objects the encoder keeps Metal counterparts for.
void SamplerCreated(VkSampler sampler, const VkSamplerCreateInfo* info);
void SamplerDestroyed(VkSampler sampler);
void ImageViewCreated(VkImageView view, const VkImageViewCreateInfo* info);
void ImageViewDestroyed(VkImageView view);
void BufferCreated(VkBuffer buffer, VkDeviceSize size);
void BufferDestroyed(VkBuffer buffer);
/// bbport: descriptor sets' contents (vkAllocateDescriptorSets: empty; vkUpdateDescriptorSets:
/// the writes, decoded; a copied-into set is unknown, its draws fall back).
void DescriptorSetsAllocated(VkDescriptorPool pool, uint32_t count, const VkDescriptorSet* sets);
/// Their contents forgotten (vkFreeDescriptorSets; a pool's reset or destruction: all of its).
void DescriptorSetsFreed(uint32_t count, const VkDescriptorSet* sets);
void DescriptorPoolReset(VkDescriptorPool pool);
void DescriptorSetsUpdated(uint32_t write_count, const VkWriteDescriptorSet* writes,
                           uint32_t copy_count, const VkCopyDescriptorSet* copies);

// Commands (the layer's names without vk).
void CmdPipelineBarrier(Owned& o, VkCommandBuffer cb, const VkDependencyInfo* info);
void CmdPipelineBarrier1(Owned& o, VkCommandBuffer cb, VkPipelineStageFlags src,
                         VkPipelineStageFlags dst, VkDependencyFlags flags, uint32_t memory_count,
                         const VkMemoryBarrier* memory, uint32_t buffer_count,
                         const VkBufferMemoryBarrier* buffers, uint32_t image_count,
                         const VkImageMemoryBarrier* images);
void CmdBeginRendering(Owned& o, const VkRenderingInfo* info);
void CmdEndRendering(Owned& o);
void CmdBindPipeline(Owned& o, VkPipelineBindPoint point, VkPipeline pipeline);
void CmdPushDescriptorSet(Owned& o, VkPipelineBindPoint point, VkPipelineLayout layout,
                          uint32_t set, uint32_t count, const VkWriteDescriptorSet* writes);
void CmdBindDescriptorSets(Owned& o, VkPipelineBindPoint point, VkPipelineLayout layout,
                           uint32_t first, uint32_t count, const VkDescriptorSet* sets,
                           uint32_t dynamic_count, const uint32_t* dynamic_offsets);
void CmdPushConstants(Owned& o, VkPipelineLayout layout, VkShaderStageFlags stages,
                      uint32_t offset, uint32_t size, const void* values);
void CmdBindVertexBuffers(Owned& o, uint32_t first, uint32_t count, const VkBuffer* buffers,
                          const VkDeviceSize* offsets, const VkDeviceSize* sizes,
                          const VkDeviceSize* strides);
void CmdBindIndexBuffer(Owned& o, VkBuffer buffer, VkDeviceSize offset, VkIndexType type);

/// Dynamic state, by kind.
enum class State : uint32_t {
    Viewport,
    Scissor,
    CullMode,
    FrontFace,
    DepthTest,
    DepthWrite,
    DepthCompare,
    DepthBoundsTest,
    DepthBounds,
    DepthBiasEnable,
    DepthBias,
    StencilTest,
    StencilOp,
    StencilCompareMask,
    StencilWriteMask,
    StencilReference,
    PrimitiveRestart,
    RasterizerDiscard,
    BlendConstants,
    ColorWriteMask,
    LineWidth,
    FeedbackLoop,
    Count,
};
void CmdSetViewports(Owned& o, uint32_t first, uint32_t count, const VkViewport* viewports,
                     bool with_count);
void CmdSetScissors(Owned& o, uint32_t first, uint32_t count, const VkRect2D* scissors,
                    bool with_count);
void CmdSetFlag(Owned& o, State state, uint32_t value); ///< cull, front face, enables, compare
void CmdSetDepthBounds(Owned& o, float min, float max);
void CmdSetDepthBias(Owned& o, float constant, float clamp, float slope);
void CmdSetStencilOp(Owned& o, VkStencilFaceFlags faces, VkStencilOp fail, VkStencilOp pass,
                     VkStencilOp depth_fail, VkCompareOp compare);
void CmdSetStencilValue(Owned& o, State state, VkStencilFaceFlags faces, uint32_t value);
void CmdSetBlendConstants(Owned& o, const float constants[4]);
void CmdSetColorWriteMask(Owned& o, uint32_t first, uint32_t count,
                          const VkColorComponentFlags* masks);
void CmdSetLineWidth(Owned& o, float width);

void CmdDraw(Owned& o, uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex,
             uint32_t first_instance);
void CmdDrawIndexed(Owned& o, uint32_t index_count, uint32_t instance_count, uint32_t first_index,
                    int32_t vertex_offset, uint32_t first_instance);
void CmdDrawIndirect(Owned& o, VkBuffer buffer, VkDeviceSize offset, uint32_t count,
                     uint32_t stride, bool indexed);
void CmdDrawIndirectCount(Owned& o, VkBuffer buffer, VkDeviceSize offset, VkBuffer count_buffer,
                          VkDeviceSize count_offset, uint32_t max_count, uint32_t stride,
                          bool indexed);
void CmdDispatch(Owned& o, uint32_t x, uint32_t y, uint32_t z);
void CmdDispatchIndirect(Owned& o, VkBuffer buffer, VkDeviceSize offset);

// Transfers, outside render passes: in Metal where they can be (BB_METAL_TRANSFERS=0: never),
// else to KosmicKrisp.
void CmdCopyBuffer(Owned& o, VkBuffer src, VkBuffer dst, uint32_t count,
                   const VkBufferCopy* regions);
void CmdFillBuffer(Owned& o, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size, uint32_t data);
void CmdUpdateBuffer(Owned& o, VkBuffer dst, VkDeviceSize offset, VkDeviceSize size,
                     const void* data);
void CmdCopyBufferToImage(Owned& o, VkBuffer src, VkImage dst, VkImageLayout layout,
                          uint32_t count, const VkBufferImageCopy* regions);
void CmdCopyImageToBuffer(Owned& o, VkImage src, VkImageLayout layout, VkBuffer dst,
                          uint32_t count, const VkBufferImageCopy* regions);
void CmdCopyImage(Owned& o, VkImage src, VkImageLayout src_layout, VkImage dst,
                  VkImageLayout dst_layout, uint32_t count, const VkImageCopy* regions);
void CmdClearColorImage(Owned& o, VkImage image, VkImageLayout layout,
                        const VkClearColorValue* color, uint32_t count,
                        const VkImageSubresourceRange* ranges);
void CmdClearDepthStencilImage(Owned& o, VkImage image, VkImageLayout layout,
                               const VkClearDepthStencilValue* value, uint32_t count,
                               const VkImageSubresourceRange* ranges);
void ImageCreated(VkImage image, const VkImageCreateInfo* info);
void ImageDestroyed(VkImage image);

/// A command the encoder does not encode in Metal and that needs no state: it goes to a
/// KosmicKrisp segment as it is (`record` calls the driver's function on the segment's command
/// buffer). `in_pass`: it may stand inside a render pass (vkCmdClearAttachments, timestamps,
/// labels); otherwise any open pass is closed first on both sides.
template <typename Record>
void Fallback(Owned& o, const char* name, bool in_pass, Record&& record);
/// The command buffer of the KosmicKrisp segment such a command goes to (opened as needed, its
/// render pass replayed when `in_pass`).
VkCommandBuffer FallbackBuffer(Owned& o, const char* name, bool in_pass);

template <typename Record>
void Fallback(Owned& o, const char* name, bool in_pass, Record&& record) {
    record(FallbackBuffer(o, name, in_pass));
}

/// An occlusion query is begun / ended (after its Fallback): while one is active, everything
/// stays in its KosmicKrisp segment (a query begins and ends in one command buffer, and one begun
/// in a render pass ends in it).
void QueryBegun(Owned& o);
void QueryEnded(Owned& o);

/// Transfers encoded in Metal so far (never reset; the frame self-test).
unsigned long long TransfersTotal();

/// Every 5 s from a submission (the layer's report): Metal and fallback counts per second.
void Report(double seconds);

} // namespace MetalFrame
