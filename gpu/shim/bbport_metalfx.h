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
};

class Scaler {
public:
    ~Scaler();
    static std::unique_ptr<Scaler> Create(const Config& config, const Heaps& heaps,
                                          std::string& error);
    /// Copies the input buffers into textures, upscales, copies the output into its buffer, and
    /// waits until the GPU is done (the first version is synchronous).
    bool Encode(const Frame& frame, std::string& error);

    struct Impl;

private:
    explicit Scaler(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl;
};

} // namespace BbMetalFx
