// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: MetalFX temporal upscaler (BB_UPSCALER=metalfx; macOS with KosmicKrisp only). The
// scene's color, depth and motion vectors are copied into buffers on exportable memory whose
// MTLHeaps (VK_EXT_external_memory_metal) MetalFX reads on KosmicKrisp's own MTLDevice
// (gpu/shim/bbport_metalfx.mm); its output comes back the same way. First version: synchronous
// (the scheduler finishes the frame so far, MetalFX runs and is waited for, Vulkan continues).
// Elsewhere it reports itself unsupported.

#pragma once

#include <array>
#include <memory>
#include <string>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace BbMetalFx {
class Scaler;
}

namespace Vulkan {

class Instance;
class Scheduler;

class MetalFxUpscaler {
public:
    MetalFxUpscaler(const Instance& instance, Scheduler& scheduler);
    ~MetalFxUpscaler();

    /// MetalFX can run: macOS, KosmicKrisp's MTLHeap export, MetalFX on that MTLDevice.
    [[nodiscard]] bool Supported() const noexcept {
        return supported;
    }
    /// Why it cannot run (now or since a failure), or null.
    [[nodiscard]] const char* Problem() const noexcept {
        return problem.empty() ? nullptr : problem.c_str();
    }
    /// The last failure will repeat (unsupported size, scaler creation): stop selecting it.
    [[nodiscard]] bool Fatal() const noexcept {
        return fatal;
    }
    /// After a menu change: a supported MetalFX may be tried again.
    void Retry() noexcept {
        if (supported) {
            fatal = false;
            problem.clear();
        }
    }

    struct Input {
        vk::Image image; ///< in General layout, TransferSrc usage
        vk::Format format;
        vk::ImageAspectFlagBits aspect;
    };
    struct Frame {
        Input color; ///< RGBA16F (HDR) or RGBA8
        Input depth; ///< D32 or D32S8 (depth aspect)
        Input motion; ///< RG16F, pixels, previous minus current position
        u32 width, height; ///< render size (the inputs' top-left)
        vk::Image output; ///< RGBA16F, output size; in General layout afterwards
        u32 out_width, out_height;
        bool hdr;
        std::array<float, 2> jitter; ///< render pixels, the FSR 3 convention
        bool reset;
    };
    /// Submits and waits for the frame so far, upscales, and records the copy of the result into
    /// `output` in the scheduler's new command buffer: callers must fetch CommandBuffer() again.
    bool Run(const Frame& frame);

private:
    struct Staging {
        vk::Buffer buffer;
        vk::DeviceMemory memory;
        vk::DeviceSize size = 0;
        void* heap = nullptr; ///< id<MTLHeap>, owned by the memory
    };
    bool CreateStaging(Staging& staging, vk::DeviceSize size);
    void DestroyStaging(Staging& staging);
    bool EnsureResources(const Frame& frame);
    void Fail(std::string reason, bool permanent);

    const Instance& instance;
    Scheduler& scheduler;
    bool supported = false;
    bool fatal = false;
    std::string problem;
    void* get_metal_handle = nullptr; ///< PFN_vkGetMemoryMetalHandleEXT
    std::array<Staging, 4> staging{}; ///< color, depth, motion, output
    std::unique_ptr<BbMetalFx::Scaler> scaler;
    u32 width = 0, height = 0, out_width = 0, out_height = 0;
    vk::Format color_format = vk::Format::eUndefined;
    bool hdr = false;
};

} // namespace Vulkan
