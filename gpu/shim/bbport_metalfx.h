// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal side of the MetalFX temporal upscaler (macOS only; bbport_metalfx.mm).
// Plain C++ so the Vulkan renderer (vk_metalfx.cpp) needs no Objective-C. Inputs and output are
// linear buffers at offset 0 of MTLHeaps that KosmicKrisp exported (VK_EXT_external_memory_metal);
// MetalFX runs on private textures on the heaps' MTLDevice, which is KosmicKrisp's own.

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace BbMetalFx {

/// MetalFX loads in this process (x86-64 under Rosetta too) and supports the MTLDevice that owns
/// `heap` (an id<MTLHeap>). Otherwise false with the reason.
bool Supported(void* heap, std::string& reason);

/// An id<MTLSharedEvent>'s value, and setting it from the CPU (never lowered).
uint64_t EventValue(void* event);
void SignalEvent(void* event, uint64_t value);

enum class ColorFormat { Rgba16Float, Rgba8Unorm };

struct Config {
    uint32_t in_width, in_height;   ///< render size: color, depth, motion
    uint32_t out_width, out_height; ///< output size
    ColorFormat color;              ///< depth is float32, motion RG16F, output RGBA16F
    bool hdr;                       ///< linear HDR color: MetalFX auto exposure
};

/// MTLHeaps holding tightly packed rows (Vulkan buffer copies with bufferRowLength 0).
struct Heaps {
    void* color;
    void* depth;
    void* motion;
    void* output;
};

struct Frame {
    float jitter_x, jitter_y;             ///< render pixels
    float motion_scale_x, motion_scale_y; ///< 1: pixels, previous minus current position
    bool reset;
    bool depth_reversed;
    /// Asynchronous (`signal_event` set): the GPU sets `signal_event` (an id<MTLSharedEvent>) to
    /// `signal_value` when the output is written; nothing waits on the CPU. With `wait_event` (a
    /// Vulkan timeline) too, the GPU first waits until it reaches `wait_value` before reading the
    /// inputs; without, the caller waited for the inputs. A failed command buffer still sets the
    /// signal (from its completion handler), and TakeError() reports it later.
    void* wait_event = nullptr;
    uint64_t wait_value = 0;
    void* signal_event = nullptr;
    uint64_t signal_value = 0;
    /// BB_METALFX_TEST_STALL (a test): the command buffer first waits for an event that a timer
    /// signals 10 s later, as MetalFX's own GPU work did when it stalled (run of 2026-10-24).
    bool test_stall = false;
};

class Scaler {
public:
    ~Scaler();
    /// With `mtl4_queue` (KosmicKrisp's id<MTL4CommandQueue>, macOS 26), MetalFX runs on that
    /// queue in its order (EncodeOnQueue); otherwise on a command queue of its own (Encode).
    static std::unique_ptr<Scaler> Create(const Config& config, const Heaps& heaps,
                                          std::string& error, void* mtl4_queue = nullptr);
    /// Copies the input buffers into textures, upscales and copies the output into its buffer.
    /// Without events in `frame`, waits until the GPU is done.
    bool Encode(const Frame& frame, std::string& error);
    /// Queue mode (created with `mtl4_queue`): copies the inputs, upscales and copies the output
    /// back in one Metal 4 command buffer committed to that queue now, then signals
    /// `frame.signal_event` to `frame.signal_value` there. The queue's earlier work (the input
    /// copies) comes first and its later work (the copy back) after, through queue barriers: no
    /// event waits. The caller serialises this with the queue's other commits (submit_mutex).
    bool EncodeOnQueue(const Frame& frame, std::string& error);
    [[nodiscard]] bool OnQueue() const;
    /// The failure of an earlier asynchronous frame, once.
    bool TakeError(std::string& error);
    /// Until the last frame handed to Metal is done (call from the thread that encodes).
    void WaitIdle();
    /// Asynchronous frames handed to Metal whose command buffers have not completed (or failed).
    /// Any thread.
    [[nodiscard]] bool Busy() const;
    /// Gives up on the last command buffer (a stalled one): the destructor no longer waits for
    /// it, and what the command buffers use is left allocated (they still hold it).
    void Abandon();

    struct Impl;

private:
    explicit Scaler(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl;
};

} // namespace BbMetalFx
