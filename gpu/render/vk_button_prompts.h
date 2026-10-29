// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the port's button prompt icons on the GPU (gpu/shim/bbport_button_prompts.h). One image
// array, a layer per game icon; a binding of one of the game's key guide icons samples its layer
// instead while it is drawn for the device in use (Rasterizer::BindTextures). An atlas of the HUD
// with icons packed in (BbPrompts::AtlasOf) samples an uncompressed copy with the port's icons in
// its cells.
#pragma once

#include <map>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "vulkan/vk_common.h"
#include "textures/image.h"

namespace VideoCore {
class BufferCache;
struct ImageViewInfo;
} // namespace VideoCore

namespace Vulkan {

class Instance;
class Runtime;
class Scheduler;

class ButtonPrompts {
public:
    ButtonPrompts(const Instance& instance, Scheduler& scheduler, VideoCore::BufferCache& buffer_cache);
    ~ButtonPrompts();

    /// The icons for the device in use, uploaded when they changed (the draw recording thread;
    /// ends the render pass then, once per change of device or bindings).
    void Update();

    /// Whether any icon is the port's now.
    bool Active() const noexcept {
        return active;
    }

    /// The port's icon for the game's icon `icon` (Image::prompt_icon), viewed as `info` views
    /// the game's; null: the game's own.
    vk::ImageView View(u32 icon, const VideoCore::ImageViewInfo& info);

    /// The copy of the HUD atlas `image` (Image::prompt_atlas) with the port's icons in its
    /// cells, viewed as `info` views the game's; null: the game's own. Made (in stream order, out
    /// of a render pass) when the icons or the atlas's bytes changed.
    vk::ImageView AtlasView(Runtime& runtime, VideoCore::Image& image,
                            const VideoCore::ImageViewInfo& info);

private:
    struct Atlas {
        VideoCore::HostImage image;
        u32 upload = 0;
        u32 generation = 0;
        std::map<std::tuple<bool, bool, u32>, vk::UniqueImageView> views;
        vk::UniqueImageView storage_view; ///< the copy as rgba8, written by the decode
        vk::Image source;              ///< the game's image the source view is of
        vk::UniqueImageView source_view; ///< it as UNORM BC7, sampled by the decode
    };

    /// The decode of a BC atlas into its rgba8 copy (atlas_decode.comp); false when it cannot be made.
    bool DecodePipeline();

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::BufferCache& buffer_cache;
    VideoCore::HostImage image;
    u32 layers = 0;
    u32 generation = 0;
    bool active = false;
    std::vector<bool> drawn;
    std::vector<u32> small; ///< the icons at the game's size (atlas cells), a layer after another
    std::unordered_map<u64, Atlas> atlases; ///< by Image::image_uid
    std::map<std::tuple<u32, bool, bool, u32>, vk::UniqueImageView> views;
    vk::UniqueDescriptorSetLayout decode_set_layout;
    vk::UniquePipelineLayout decode_layout;
    vk::UniquePipeline decode_pipeline;
    bool decode_failed = false;
};

} // namespace Vulkan
