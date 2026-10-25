// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): this port's own chain entries for vkExportMetalObjectsEXT, as KosmicKrisp patch
// 0008 defines them (tools/macos/kosmickrisp-patches, kk_sync.c): the two definitions must stay
// the same. Their sType values decode to an unallocated extension number; another driver ignores
// them (and leaves the outputs as the caller set them). With the Vulkan validation layers on they
// may not reach the driver, or reach it with wrapped handles: use them without layers.

#pragma once

#include <cstddef>
#include <cstdint>

#include "vulkan/vk_common.h"
// Only the C types of VK_EXT_metal_objects (vulkan.hpp stays without the Metal platform define).
#include <vulkan/vulkan_metal.h>

namespace BbMetal {

inline constexpr VkStructureType ExportViewTextureType = static_cast<VkStructureType>(0x62627601);
inline constexpr VkStructureType ExportBufferObjectType = static_cast<VkStructureType>(0x62627602);

/// What an image view's texture is for (KosmicKrisp keeps one per use).
enum ViewTextureUse : uint32_t {
    ViewTextureSampled = 0, ///< with the view's swizzle and type
    ViewTextureStorage = 1, ///< for shader writes (cubes as 2D arrays)
    ViewTextureRender = 2,  ///< an attachment: `level` and `slice` go in its descriptor
    /// An input attachment; only for views with attachment usage. When it is the image's own
    /// texture, the view's level and layer are not in it (and not reported).
    ViewTextureInput = 3,
};

/// An image view's texture for one use, with the level and slice an attachment descriptor needs
/// for its render texture (0 when the texture already points at them).
struct ExportViewTexture {
    VkStructureType sType = ExportViewTextureType;
    const void* pNext = nullptr;
    VkImageView imageView = VK_NULL_HANDLE;
    uint32_t plane = 0; ///< a plane index (0-2), not an aspect bit as in VkExportMetalTextureInfoEXT
    uint32_t use = ViewTextureSampled;
    MTLTexture_id mtlTexture = nullptr; ///< out
    uint32_t level = 0;                 ///< out
    uint32_t slice = 0;                 ///< out
};

/// A VkBuffer's own id<MTLBuffer> and the offset of the buffer's start in it: a sparse buffer's
/// own (offset 0); a buffer bound to heap memory, its own sub-buffer of the heap at the bind offset
/// (offset 0; it aliases the memory's heap-wide buffer, so barriers need resource-alias
/// visibility); a buffer bound to host-imported memory, the memory's buffer (offset: the bind
/// offset).
struct ExportBufferObject {
    VkStructureType sType = ExportBufferObjectType;
    const void* pNext = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    MTLBuffer_id mtlBuffer = nullptr; ///< out
    uint64_t offset = 0;              ///< out
};

// The C structs in the patch have these layouts (LP64).
static_assert(offsetof(ExportViewTexture, imageView) == 16 && offsetof(ExportViewTexture, plane) == 24 &&
              offsetof(ExportViewTexture, use) == 28 && offsetof(ExportViewTexture, mtlTexture) == 32 &&
              offsetof(ExportViewTexture, level) == 40 && offsetof(ExportViewTexture, slice) == 44 &&
              sizeof(ExportViewTexture) == 48);
static_assert(offsetof(ExportBufferObject, buffer) == 16 &&
              offsetof(ExportBufferObject, mtlBuffer) == 24 && offsetof(ExportBufferObject, offset) == 32 &&
              sizeof(ExportBufferObject) == 40);

} // namespace BbMetal
