// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: converts surfaces between the GPU's tiled layouts and linear ones on the GPU
// (textures/shaders/detile.comp): detiling for uploads, tiling for write-backs. One compute
// pipeline per tile mode, texel size, sample count and direction, made on first use.
#pragma once

#include <span>
#include <unordered_map>
#include <utility>

#include "common/types.h"
#include "gcn/tiling.h"
#include "buffers/buffer.h"

namespace Vulkan {
class Instance;
class Scheduler;
class Runtime;
} // namespace Vulkan

namespace VideoCore {

struct ImageInfo;
struct Image;

class Detiler {
public:
    Detiler(const Vulkan::Instance& instance, Vulkan::Scheduler& scheduler,
            Vulkan::Runtime& runtime, StreamBuffer& stream_buffer);
    ~Detiler();

    /// The image's levels in `buffer_copies` copied out and tiled into `out_buffer` at
    /// `out_offset` (as the guest lays them out). Tiled in VRAM first when `out_buffer` is host
    /// memory: that copy is returned (`vram_size` bytes or the image's, whichever is more; only
    /// the image's bytes are written), else {nullptr, 0}. Valid for the current submission.
    std::pair<const Buffer*, u64> TileImage(Image& image,
                                            std::span<vk::BufferImageCopy> buffer_copies,
                                            const Buffer* out_buffer, u64 out_offset,
                                            u64 vram_size = 0);
    /// The surface at `in_offset` of `in_buffer`, linear: a staging buffer with it, or the input
    /// when it is not tiled. From `first_level` on only (texture_quality's reduced images): the
    /// staging buffer then starts at that level (its linear offset is info.mips_layout[first].offset),
    /// and the GPU reads only those levels' bytes of the game's memory over the bus.
    std::pair<const Buffer*, u64> DetileImage(const Buffer* in_buffer, u64 in_offset,
                                              const ImageInfo& info, u32 first_level = 0);

private:
    struct Key {
        Gcn::TileMode tile_mode;
        u32 num_bits;
        u32 num_samples;
        bool tiler;

        bool operator==(const Key&) const = default;
    };
    struct KeyHash {
        size_t operator()(const Key& key) const {
            return size_t(key.tile_mode) | size_t(key.num_bits) << 5 |
                   size_t(key.num_samples) << 13 | size_t(key.tiler) << 18;
        }
    };

    vk::Pipeline Pipeline(const ImageInfo& info, bool tiler);
    /// The level table of the shader, in the stream buffer.
    vk::DescriptorBufferInfo LevelTable(const ImageInfo& info, u32 num_levels, u32 first_level = 0);
    void Dispatch(vk::Pipeline pipeline, const vk::DescriptorBufferInfo& tiled,
                  const vk::DescriptorBufferInfo& linear, const vk::DescriptorBufferInfo& levels,
                  u32 texels);

    const Vulkan::Instance& instance;
    Vulkan::Scheduler& scheduler;
    Vulkan::Runtime& runtime;
    StreamBuffer& stream_buffer;
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    std::unordered_map<Key, vk::UniquePipeline, KeyHash> pipelines;
};

} // namespace VideoCore
