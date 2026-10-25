// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal side of the Metal command path's interop test (metal_interop.mm).
// Plain C++ so the Vulkan side needs no Objective-C. Every handle is an Objective-C object
// KosmicKrisp exported (VK_EXT_metal_objects, this port's patches 0007 and 0008).

#pragma once

#include <cstdint>
#include <string>

namespace BbMetal {

/// One round of the interop test, committed to KosmicKrisp's own Metal 4 queue (`queue`,
/// an id<MTL4CommandQueue>) between two Vulkan submissions: copies `texture` (an id<MTLTexture>,
/// RGBA8, `width` x `height`) into `buffer` (an id<MTLBuffer>) at `buffer_offset`, which reads
/// what the Vulkan submission before wrote; then clears `texture` to `clear` in a render pass,
/// which the Vulkan submission after reads. Queue barriers order both against the driver's work,
/// as the Metal command path's encoders will. Returns false with the reason if Metal 4 or an
/// object is missing.
bool InteropRound(void* queue, void* texture, void* buffer, uint64_t buffer_offset,
                  uint32_t width, uint32_t height, const float clear[4], std::string& error);

} // namespace BbMetal
