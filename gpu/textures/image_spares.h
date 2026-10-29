// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: spare images made ahead on a thread of their own (BB_IMAGE_SPARES).
//
// On NVIDIA under Windows (WDDM) vkBindImageMemory waits for the GPU's queue: 1-2 ms for each new
// texture and ~130 ms for a 480x272 depth image at every world entry (GTX 1050 laptop,
// 2026-10-26), all on the thread recording the draws. The texture cache's images of the shapes it
// made itself recently (this session, or the last ones: image_shapes.txt in the game's cache
// directory) are kept ready here, created and bound by the spare thread; HostImage::Create takes
// one instead of creating and binding its own. Freed images go to the recycle pool first
// (image.cpp): spares cover what nothing freed.
//
// BB_IMAGE_SPARES=1/0 forces it on/off (default: on on Windows); BB_IMAGE_SPARES_MB the memory
// for spares (default: a 32nd of the budget, 128 MiB at most; the texture cache's collector trims
// them as it trims the recycle pool).
#pragma once

#include "common/types.h"
#include "vulkan/vk_common.h"

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

namespace VideoCore::ImageSpares {

/// Starts the spare thread (once; nothing when off). `budget`: the device's memory budget.
void Start(vk::Device device, VmaAllocator allocator, u64 budget);

/// Stops it, gives every spare back and saves the shapes (the texture cache's end).
void Stop();

/// A spare of exactly `ci` from `allocator`: true with its image, memory and size (owned by the
/// caller from then on).
bool Take(VmaAllocator allocator, const vk::ImageCreateInfo& ci, vk::Image& image,
          VmaAllocation& allocation, vk::DeviceSize& size);

/// An image of `ci` (`size` bytes of memory) was made on the caller's thread: its shape is wanted.
void NoteMade(const vk::ImageCreateInfo& ci, vk::DeviceSize size);

/// At most `keep_bytes` of spares now, and no new ones past that (memory pressure).
void Trim(vk::DeviceSize keep_bytes);

/// The VMA flags HostImage::Create allocates an image of `ci` with (image.cpp): spares are made
/// with the same ones.
u32 MemoryFlags(vk::Device device, const vk::ImageCreateInfo& ci);

} // namespace VideoCore::ImageSpares
