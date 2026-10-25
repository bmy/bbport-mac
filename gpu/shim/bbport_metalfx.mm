// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: MetalFX temporal upscaler, Metal side (bbport_metalfx.h). Built with ARC on APPLE.
// MetalFX is opened at run time (dlopen) and its classes looked up by name: nothing links
// against it, so a build or a Rosetta process without an x86-64 MetalFX still runs (unsupported).

#include "bbport_metalfx.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include <cstdio>
#include <dlfcn.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <mutex>

namespace BbMetalFx {

namespace {

Class DescriptorClass() {
    static Class cls = []() -> Class {
        // In the dyld shared cache; dlopen by path still works (macOS 11+).
        dlopen("/System/Library/Frameworks/MetalFX.framework/MetalFX", RTLD_LAZY | RTLD_LOCAL);
        return NSClassFromString(@"MTLFXTemporalScalerDescriptor");
    }();
    return cls;
}

std::string Describe(NSError* error) {
    const char* text = error ? error.localizedDescription.UTF8String : nullptr;
    return text ? text : "unknown error";
}

} // namespace

bool Supported(void* heap_handle, std::string& reason) {
    @autoreleasepool {
        Class cls = DescriptorClass();
        if (!cls) {
            reason = "MetalFX.framework did not load in this process (x86-64 under Rosetta?)";
            return false;
        }
        id<MTLHeap> heap = (__bridge id<MTLHeap>)heap_handle;
        id<MTLDevice> device = heap.device;
        if (!device) {
            reason = "the exported MTLHeap has no MTLDevice";
            return false;
        }
        if (![cls supportsDevice:device]) {
            reason = std::string{"MetalFX temporal scaling does not support "} +
                     device.name.UTF8String;
            return false;
        }
        if (heap.type != MTLHeapTypePlacement) {
            reason = "the exported MTLHeap is not a placement heap";
            return false;
        }
        return true;
    }
}

struct Scaler::Impl {
    Config config{};
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLFXTemporalScaler> scaler;
    // Linear views of the Vulkan buffers (offset 0 of each heap).
    id<MTLBuffer> color_buffer, depth_buffer, motion_buffer, output_buffer;
    // MetalFX's own textures: it requires a private output, and these have the usage it asks for.
    id<MTLTexture> color, depth, motion, output;
    NSUInteger color_bytes = 8;
    // An asynchronous frame's failure, reported by the next Encode's caller (TakeError). Shared
    // with the completion handlers, which may run after the scaler is gone.
    struct Errors {
        std::mutex mutex;
        std::string text;
        // Asynchronous command buffers committed and not yet completed (their handlers count down).
        std::atomic<int> in_flight{0};
    };
    std::shared_ptr<Errors> errors = std::make_shared<Errors>();
    // The last command buffer: waited for before the scaler goes (its handler uses this state).
    id<MTLCommandBuffer> last;
    // Queue mode (macOS 26): KosmicKrisp's id<MTL4CommandQueue>, a Metal 4 scaler
    // (id<MTL4FXTemporalScaler>) and the compiler it was built with, what its command buffers
    // make resident (MetalFX's textures; the heaps are KosmicKrisp's, resident on its queue), and
    // the fence between the copies and MetalFX's own passes. Typed as `id`: macOS 26 types.
    id queue4;
    id scaler4;
    id compiler4;
    id residency; // id<MTLResidencySet> (macOS 15)
    id<MTLFence> fence;
};

Scaler::Scaler(std::unique_ptr<Impl> impl_) : impl{std::move(impl_)} {}

namespace {

/// Queue mode: Metal 4 command buffers do not keep what they use, so the textures and the
/// residency set stay until every committed one has completed (their feedback handlers count
/// down). False after `seconds` (a stuck GPU): the caller then leaves them allocated.
bool WaitInFlight(const std::shared_ptr<Scaler::Impl::Errors>& errors, double seconds) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    while (errors->in_flight.load(std::memory_order_acquire) > 0) {
        if (std::chrono::steady_clock::now() >= until) {
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

} // namespace

Scaler::~Scaler() {
    if (impl && impl->last) {
        [impl->last waitUntilCompleted];
    }
    if (impl && impl->queue4 && !WaitInFlight(impl->errors, 10.0)) {
        std::printf("Upscaler: MetalFX's work on the driver's queue did not finish; its textures "
                    "are left allocated\n");
        (void)impl.release();
    }
}

bool Scaler::Busy() const {
    return impl->errors->in_flight.load(std::memory_order_acquire) > 0;
}

void Scaler::Abandon() {
    // The command buffers keep their buffers and textures (Metal retains what they use), so only
    // the wait goes; the Impl itself stays allocated for the handlers' sake.
    impl->last = nil;
    (void)impl.release();
}

bool Scaler::OnQueue() const {
    return impl->queue4 != nil;
}

std::unique_ptr<Scaler> Scaler::Create(const Config& config, const Heaps& heaps,
                                       std::string& error, void* mtl4_queue) {
    @autoreleasepool {
        Class cls = DescriptorClass();
        if (!cls) {
            error = "MetalFX.framework did not load";
            return nullptr;
        }
        auto impl = std::make_unique<Impl>();
        impl->config = config;
        id<MTLHeap> color_heap = (__bridge id<MTLHeap>)heaps.color;
        impl->device = color_heap.device;
        id<MTLDevice> device = impl->device;
        const float scale = float(config.out_width) / float(config.in_width);
        // macOS 14+ reports the device's range; outside it the scaler would not be created.
        if ([cls respondsToSelector:@selector(supportedInputContentMinScaleForDevice:)]) {
            const float min_scale = [cls supportedInputContentMinScaleForDevice:device];
            const float max_scale = [cls supportedInputContentMaxScaleForDevice:device];
            if (scale < min_scale - 1e-3f || scale > max_scale + 1e-3f) {
                char text[160];
                std::snprintf(text, sizeof(text),
                              "scale %.2f outside the device's MetalFX range %.2f..%.2f", scale,
                              min_scale, max_scale);
                error = text;
                return nullptr;
            }
        }
        const MTLPixelFormat color_format = config.color == ColorFormat::Rgba16Float
                                                ? MTLPixelFormatRGBA16Float
                                                : MTLPixelFormatRGBA8Unorm;
        impl->color_bytes = config.color == ColorFormat::Rgba16Float ? 8 : 4;
        MTLFXTemporalScalerDescriptor* descriptor = [[cls alloc] init];
        descriptor.inputWidth = config.in_width;
        descriptor.inputHeight = config.in_height;
        descriptor.outputWidth = config.out_width;
        descriptor.outputHeight = config.out_height;
        descriptor.colorTextureFormat = color_format;
        descriptor.depthTextureFormat = MTLPixelFormatDepth32Float;
        descriptor.motionTextureFormat = MTLPixelFormatRG16Float;
        descriptor.outputTextureFormat = MTLPixelFormatRGBA16Float;
        descriptor.autoExposureEnabled = config.hdr;
        MTLTextureUsage usages[4]{};
        if (mtl4_queue) {
            if (@available(macOS 26.0, *)) {
                NSError* failure = nil;
                id<MTL4Compiler> compiler =
                    [device newCompilerWithDescriptor:[[MTL4CompilerDescriptor alloc] init]
                                                error:&failure];
                if (!compiler) {
                    error = "newCompilerWithDescriptor failed: " + Describe(failure);
                    return nullptr;
                }
                id<MTL4FXTemporalScaler> scaler4 =
                    [descriptor newTemporalScalerWithDevice:device compiler:compiler];
                if (!scaler4) {
                    error = "newTemporalScalerWithDevice:compiler: failed";
                    return nullptr;
                }
                impl->compiler4 = compiler;
                impl->scaler4 = scaler4;
                impl->queue4 = (__bridge id<MTL4CommandQueue>)mtl4_queue;
                usages[0] = scaler4.colorTextureUsage;
                usages[1] = scaler4.depthTextureUsage;
                usages[2] = scaler4.motionTextureUsage;
                usages[3] = scaler4.outputTextureUsage;
            } else {
                error = "MetalFX on the driver's queue needs macOS 26";
                return nullptr;
            }
        } else {
            impl->scaler = [descriptor newTemporalScalerWithDevice:device];
            if (!impl->scaler) {
                error = "newTemporalScalerWithDevice failed";
                return nullptr;
            }
            usages[0] = impl->scaler.colorTextureUsage;
            usages[1] = impl->scaler.depthTextureUsage;
            usages[2] = impl->scaler.motionTextureUsage;
            usages[3] = impl->scaler.outputTextureUsage;
        }
        const auto texture = [&](MTLPixelFormat format, uint32_t w, uint32_t h,
                                 MTLTextureUsage usage) -> id<MTLTexture> {
            MTLTextureDescriptor* td =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                   width:w
                                                                  height:h
                                                               mipmapped:NO];
            td.storageMode = MTLStorageModePrivate;
            td.usage = usage;
            return [device newTextureWithDescriptor:td];
        };
        const uint32_t w = config.in_width, h = config.in_height;
        impl->color = texture(color_format, w, h, usages[0]);
        impl->depth = texture(MTLPixelFormatDepth32Float, w, h, usages[1]);
        impl->motion = texture(MTLPixelFormatRG16Float, w, h, usages[2]);
        impl->output = texture(MTLPixelFormatRGBA16Float, config.out_width, config.out_height,
                               usages[3]);
        if (!impl->color || !impl->depth || !impl->motion || !impl->output) {
            error = "MetalFX texture creation failed";
            return nullptr;
        }
        const auto buffer = [&](void* handle, NSUInteger length) -> id<MTLBuffer> {
            id<MTLHeap> heap = (__bridge id<MTLHeap>)handle;
            if (heap.device != device || heap.type != MTLHeapTypePlacement || heap.size < length) {
                return nil;
            }
            // Same storage, cache and hazard tracking modes as the heap (required).
            return [heap newBufferWithLength:length options:heap.resourceOptions offset:0];
        };
        impl->color_buffer = buffer(heaps.color, NSUInteger(w) * h * impl->color_bytes);
        impl->depth_buffer = buffer(heaps.depth, NSUInteger(w) * h * 4);
        impl->motion_buffer = buffer(heaps.motion, NSUInteger(w) * h * 4);
        impl->output_buffer =
            buffer(heaps.output, NSUInteger(config.out_width) * config.out_height * 8);
        if (!impl->color_buffer || !impl->depth_buffer || !impl->motion_buffer ||
            !impl->output_buffer) {
            error = "cannot alias the exported MTLHeaps with MTLBuffers";
            return nullptr;
        }
        if (impl->queue4) {
            if (@available(macOS 26.0, *)) {
                // Metal 4 makes nothing resident by itself: MetalFX's four textures are in a set
                // every command buffer uses. The fence orders the input copies, MetalFX's passes
                // and the output copy within the command buffer (Metal 4 tracks no hazards).
                MTLResidencySetDescriptor* rd = [[MTLResidencySetDescriptor alloc] init];
                rd.label = @"bbport MetalFX";
                rd.initialCapacity = 4;
                NSError* failure = nil;
                impl->residency = [device newResidencySetWithDescriptor:rd error:&failure];
                impl->fence = [device newFence];
                if (!impl->residency || !impl->fence) {
                    error = "MetalFX's residency set or fence: " + Describe(failure);
                    return nullptr;
                }
                for (id<MTLTexture> t : {impl->color, impl->depth, impl->motion, impl->output}) {
                    [impl->residency addAllocation:t];
                }
                [impl->residency commit];
                [impl->residency requestResidency];
                id<MTL4FXTemporalScaler> scaler4 = impl->scaler4;
                scaler4.colorTexture = impl->color;
                scaler4.depthTexture = impl->depth;
                scaler4.motionTexture = impl->motion;
                scaler4.outputTexture = impl->output;
                scaler4.fence = impl->fence;
            }
            return std::unique_ptr<Scaler>(new Scaler(std::move(impl)));
        }
        impl->queue = [device newCommandQueue];
        if (!impl->queue) {
            error = "newCommandQueue failed";
            return nullptr;
        }
        impl->queue.label = @"bbport MetalFX";
        impl->scaler.colorTexture = impl->color;
        impl->scaler.depthTexture = impl->depth;
        impl->scaler.motionTexture = impl->motion;
        impl->scaler.outputTexture = impl->output;
        return std::unique_ptr<Scaler>(new Scaler(std::move(impl)));
    }
}

bool Scaler::Encode(const Frame& frame, std::string& error) {
    @autoreleasepool {
        Impl& m = *impl;
        const NSUInteger w = m.config.in_width, h = m.config.in_height;
        const NSUInteger ow = m.config.out_width, oh = m.config.out_height;
        id<MTLCommandBuffer> cmd = [m.queue commandBuffer];
        if (!cmd) {
            error = "commandBuffer failed";
            return false;
        }
        const bool async = frame.signal_event != nullptr;
        id<MTLSharedEvent> wait_event = (__bridge id<MTLSharedEvent>)frame.wait_event;
        id<MTLSharedEvent> signal_event = (__bridge id<MTLSharedEvent>)frame.signal_event;
        if (async && wait_event) {
            // The Vulkan copies of this frame's inputs (their submission's timeline value).
            [cmd encodeWaitForEvent:wait_event value:frame.wait_value];
        }
        if (frame.test_stall) {
            // A test of the stall cutoff (BB_METALFX_TEST_STALL): this command buffer waits for an
            // event a timer sets 10 s later, so MetalFX's GPU work does not finish in that time.
            id<MTLSharedEvent> stall = [m.device newSharedEvent];
            if (stall) {
                [cmd encodeWaitForEvent:stall value:1];
                dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 10 * NSEC_PER_SEC),
                               dispatch_get_global_queue(QOS_CLASS_UTILITY, 0), ^{
                                 stall.signaledValue = 1;
                               });
            }
        }
        const MTLOrigin zero = MTLOriginMake(0, 0, 0);
        id<MTLBlitCommandEncoder> uploads = [cmd blitCommandEncoder];
        const auto upload = [&](id<MTLBuffer> source, id<MTLTexture> target, NSUInteger bytes) {
            [uploads copyFromBuffer:source
                       sourceOffset:0
                  sourceBytesPerRow:w * bytes
                sourceBytesPerImage:w * h * bytes
                         sourceSize:MTLSizeMake(w, h, 1)
                          toTexture:target
                   destinationSlice:0
                   destinationLevel:0
                  destinationOrigin:zero];
        };
        upload(m.color_buffer, m.color, m.color_bytes);
        upload(m.depth_buffer, m.depth, 4);
        upload(m.motion_buffer, m.motion, 4);
        [uploads endEncoding];

        id<MTLFXTemporalScaler> scaler = m.scaler;
        scaler.inputContentWidth = w;
        scaler.inputContentHeight = h;
        scaler.jitterOffsetX = frame.jitter_x;
        scaler.jitterOffsetY = frame.jitter_y;
        scaler.motionVectorScaleX = frame.motion_scale_x;
        scaler.motionVectorScaleY = frame.motion_scale_y;
        scaler.preExposure = 1.0f;
        scaler.reset = frame.reset;
        scaler.depthReversed = frame.depth_reversed;
        [scaler encodeToCommandBuffer:cmd];

        id<MTLBlitCommandEncoder> downloads = [cmd blitCommandEncoder];
        [downloads copyFromTexture:m.output
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:zero
                          sourceSize:MTLSizeMake(ow, oh, 1)
                            toBuffer:m.output_buffer
                   destinationOffset:0
              destinationBytesPerRow:ow * 8
            destinationBytesPerImage:ow * oh * 8];
        [downloads endEncoding];
        if (async) {
            // Vulkan's next submission waits for this value on the GPU. A failed command buffer
            // may not reach the signal: the handler sets it, so Vulkan never waits forever.
            const uint64_t value = frame.signal_value;
            [cmd encodeSignalEvent:signal_event value:value];
            std::shared_ptr<Impl::Errors> errors = m.errors;
            [cmd addCompletedHandler:^(id<MTLCommandBuffer> done) {
              if (done.status != MTLCommandBufferStatusCompleted) {
                  if (signal_event.signaledValue < value) {
                      signal_event.signaledValue = value;
                  }
                  std::scoped_lock lock{errors->mutex};
                  errors->text = "command buffer failed: " + Describe(done.error);
              }
              errors->in_flight.fetch_sub(1, std::memory_order_acq_rel);
            }];
            m.last = cmd;
            errors->in_flight.fetch_add(1, std::memory_order_acq_rel);
            [cmd commit];
            return true;
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) {
            error = "command buffer failed: " + Describe(cmd.error);
            return false;
        }
        return true;
    }
}

bool Scaler::EncodeOnQueue(const Frame& frame, std::string& error) {
    if (@available(macOS 26.0, *)) {
        @autoreleasepool {
            Impl& m = *impl;
            id<MTL4CommandQueue> queue = m.queue4;
            id<MTL4FXTemporalScaler> scaler = m.scaler4;
            if (!queue || !scaler) {
                error = "not a queue-mode scaler";
                return false;
            }
            // A new allocator and command buffer per frame: the feedback handler keeps both until
            // the GPU is done with them (an allocator must not be reset or freed before).
            id<MTL4CommandAllocator> allocator = [m.device newCommandAllocator];
            id<MTL4CommandBuffer> cmd = [m.device newCommandBuffer];
            if (!allocator || !cmd) {
                error = "newCommandAllocator / newCommandBuffer failed";
                return false;
            }
            [cmd beginCommandBufferWithAllocator:allocator];
            cmd.label = @"bbport MetalFX";
            [cmd useResidencySet:m.residency];
            const NSUInteger w = m.config.in_width, h = m.config.in_height;
            const NSUInteger ow = m.config.out_width, oh = m.config.out_height;
            const MTLOrigin zero = MTLOriginMake(0, 0, 0);
            // The staging buffers alias Vulkan's (one heap each): resource-alias visibility.
            const MTL4VisibilityOptions visible =
                MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias;

            id<MTL4ComputeCommandEncoder> uploads = [cmd computeCommandEncoder];
            // After the queue's earlier work: the Vulkan copies of the inputs.
            [uploads barrierAfterQueueStages:MTLStageAll
                                beforeStages:MTLStageBlit
                           visibilityOptions:visible];
            const auto upload = [&](id<MTLBuffer> source, id<MTLTexture> target, NSUInteger bytes) {
                [uploads copyFromBuffer:source
                           sourceOffset:0
                      sourceBytesPerRow:w * bytes
                    sourceBytesPerImage:w * h * bytes
                             sourceSize:MTLSizeMake(w, h, 1)
                              toTexture:target
                       destinationSlice:0
                       destinationLevel:0
                      destinationOrigin:zero];
            };
            upload(m.color_buffer, m.color, m.color_bytes);
            upload(m.depth_buffer, m.depth, 4);
            upload(m.motion_buffer, m.motion, 4);
            [uploads updateFence:m.fence afterEncoderStages:MTLStageBlit];
            [uploads barrierAfterStages:MTLStageBlit
                      beforeQueueStages:MTLStageAll
                      visibilityOptions:visible];
            [uploads endEncoding];

            scaler.inputContentWidth = w;
            scaler.inputContentHeight = h;
            scaler.jitterOffsetX = frame.jitter_x;
            scaler.jitterOffsetY = frame.jitter_y;
            scaler.motionVectorScaleX = frame.motion_scale_x;
            scaler.motionVectorScaleY = frame.motion_scale_y;
            scaler.preExposure = 1.0f;
            scaler.reset = frame.reset;
            scaler.depthReversed = frame.depth_reversed;
            [scaler encodeToCommandBuffer:cmd];

            id<MTL4ComputeCommandEncoder> downloads = [cmd computeCommandEncoder];
            // After MetalFX's passes (its fence, and the queue's earlier work for good measure).
            [downloads barrierAfterQueueStages:MTLStageAll
                                  beforeStages:MTLStageBlit
                             visibilityOptions:visible];
            [downloads waitForFence:m.fence beforeEncoderStages:MTLStageBlit];
            [downloads copyFromTexture:m.output
                           sourceSlice:0
                           sourceLevel:0
                          sourceOrigin:zero
                            sourceSize:MTLSizeMake(ow, oh, 1)
                              toBuffer:m.output_buffer
                     destinationOffset:0
                destinationBytesPerRow:ow * 8
              destinationBytesPerImage:ow * oh * 8];
            // Before the queue's later work: the Vulkan copy back.
            [downloads barrierAfterStages:MTLStageBlit
                        beforeQueueStages:MTLStageAll
                        visibilityOptions:visible];
            [downloads endEncoding];
            [cmd endCommandBuffer];

            MTL4CommitOptions* options = [[MTL4CommitOptions alloc] init];
            std::shared_ptr<Impl::Errors> errors = m.errors;
            id<MTLSharedEvent> signal_event = (__bridge id<MTLSharedEvent>)frame.signal_event;
            const uint64_t value = frame.signal_value;
            [options addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
              (void)allocator; // kept until the GPU is done with the command buffer
              (void)cmd;
              if (feedback.error) {
                  if (signal_event && signal_event.signaledValue < value) {
                      signal_event.signaledValue = value;
                  }
                  std::scoped_lock lock{errors->mutex};
                  errors->text = "command buffer failed: " + Describe(feedback.error);
              }
              errors->in_flight.fetch_sub(1, std::memory_order_acq_rel);
            }];
            errors->in_flight.fetch_add(1, std::memory_order_acq_rel);
            const id<MTL4CommandBuffer> list[1] = {cmd};
            [queue commit:list count:1 options:options];
            if (signal_event) {
                [queue signalEvent:signal_event value:value];
            }
            return true;
        }
    }
    error = "MetalFX on the driver's queue needs macOS 26";
    return false;
}

uint64_t EventValue(void* event) {
    if (!event) {
        return 0;
    }
    return ((__bridge id<MTLSharedEvent>)event).signaledValue;
}

void SignalEvent(void* event, uint64_t value) {
    if (!event) {
        return;
    }
    id<MTLSharedEvent> shared = (__bridge id<MTLSharedEvent>)event;
    if (shared.signaledValue < value) {
        shared.signaledValue = value;
    }
}

void Scaler::WaitIdle() {
    if (impl->last) {
        [impl->last waitUntilCompleted];
    }
    if (impl->queue4) {
        (void)WaitInFlight(impl->errors, 10.0);
    }
}

bool Scaler::TakeError(std::string& error) {
    std::scoped_lock lock{impl->errors->mutex};
    if (impl->errors->text.empty()) {
        return false;
    }
    error = std::move(impl->errors->text);
    impl->errors->text.clear();
    return true;
}

} // namespace BbMetalFx
