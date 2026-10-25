// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): Metal side of the Metal command path's interop test (metal_interop.h).
// Built with ARC on APPLE.

#include "metal/metal_interop.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace BbMetal {

bool InteropRound(void* queue_handle, void* texture_handle, void* buffer_handle,
                  uint64_t buffer_offset, uint32_t width, uint32_t height, const float clear[4],
                  std::string& error) {
    if (@available(macOS 26.0, *)) {
        @autoreleasepool {
            id queue_object = (__bridge id)queue_handle;
            if (![queue_object respondsToSelector:@selector(commit:count:options:)]) {
                error = "the exported queue is not a Metal 4 queue";
                return false;
            }
            id<MTL4CommandQueue> queue = queue_object;
            id<MTLTexture> texture = (__bridge id<MTLTexture>)texture_handle;
            id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_handle;
            if (!texture || !buffer) {
                error = "no exported texture or buffer";
                return false;
            }
            id<MTLDevice> device = texture.device;
            id<MTL4CommandAllocator> allocator = [device newCommandAllocator];
            id<MTL4CommandBuffer> cmd = [device newCommandBuffer];
            if (!allocator || !cmd) {
                error = "newCommandAllocator / newCommandBuffer failed";
                return false;
            }
            [cmd beginCommandBufferWithAllocator:allocator];
            cmd.label = @"bbport Metal interop test";
            const MTL4VisibilityOptions visible =
                MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias;

            // Read what Vulkan wrote: after the queue's earlier work (consumer barrier).
            id<MTL4ComputeCommandEncoder> copy = [cmd computeCommandEncoder];
            [copy barrierAfterQueueStages:MTLStageAll
                             beforeStages:MTLStageBlit
                        visibilityOptions:visible];
            [copy copyFromTexture:texture
                           sourceSlice:0
                           sourceLevel:0
                          sourceOrigin:MTLOriginMake(0, 0, 0)
                            sourceSize:MTLSizeMake(width, height, 1)
                              toBuffer:buffer
                     destinationOffset:buffer_offset
                destinationBytesPerRow:NSUInteger(width) * 4
              destinationBytesPerImage:NSUInteger(width) * height * 4];
            [copy barrierAfterStages:MTLStageBlit
                   beforeQueueStages:MTLStageAll
                   visibilityOptions:visible];
            [copy endEncoding];

            // Write what Vulkan reads next: before the queue's later work (producer barrier).
            MTL4RenderPassDescriptor* pass = [[MTL4RenderPassDescriptor alloc] init];
            pass.colorAttachments[0].texture = texture;
            pass.colorAttachments[0].loadAction = MTLLoadActionClear;
            pass.colorAttachments[0].storeAction = MTLStoreActionStore;
            pass.colorAttachments[0].clearColor =
                MTLClearColorMake(clear[0], clear[1], clear[2], clear[3]);
            pass.renderTargetWidth = width;
            pass.renderTargetHeight = height;
            id<MTL4RenderCommandEncoder> render = [cmd renderCommandEncoderWithDescriptor:pass];
            if (!render) {
                [cmd endCommandBuffer];
                error = "renderCommandEncoderWithDescriptor failed";
                return false;
            }
            [render barrierAfterQueueStages:MTLStageAll
                               beforeStages:MTLStageFragment
                          visibilityOptions:visible];
            [render barrierAfterStages:MTLStageFragment
                     beforeQueueStages:MTLStageAll
                     visibilityOptions:visible];
            [render endEncoding];
            [cmd endCommandBuffer];

            MTL4CommitOptions* options = [[MTL4CommitOptions alloc] init];
            [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
              (void)allocator; // kept until the GPU is done with the command buffer
              (void)cmd;
              if (feedback.error) {
                  std::printf("Metal interop test: the Metal command buffer failed: %s\n",
                              feedback.error.localizedDescription.UTF8String);
              }
            }];
            const id<MTL4CommandBuffer> list[1] = {cmd};
            [queue commit:list count:1 options:options];
            return true;
        }
    }
    error = "Metal 4 needs macOS 26";
    return false;
}

} // namespace BbMetal
