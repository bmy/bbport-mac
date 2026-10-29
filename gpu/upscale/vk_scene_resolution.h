// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <source_location>
#include <array>
#include <memory>
#include <optional>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "upscale/scene_resolution.h"
#include "textures/image.h"

namespace VideoCore { class TextureCache; }
namespace Vulkan {
class Instance;
class Runtime;
class Scheduler;

// Reduced raster targets, with native-size images retained for unmodified guest compute,
// integer texture loads, copies and CPU readbacks. All native access passes Runtime::Transit.
class SceneTargets {
public:
    SceneTargets(const Instance&, Scheduler&, Runtime&, VideoCore::TextureCache&);
    using Lookup = std::function<VideoCore::Image*(VideoCore::ImageId, u64)>;
    SceneTargets(const Instance&, Scheduler&, Runtime&, Lookup);
    ~SceneTargets();
    bool SetSize(SceneResolution::Size);
    SceneResolution::Size Size() const { return size; }
    bool Reduced() const { return size != native; }
    /// The game's own scene size (its targets): 1920x1080, or the startup resolution patch's
    /// (BB_RENDER_RES) when live presets run over it: the game then allocates targets of that size
    /// (aligned up by a few pixels), and proxies stand in for them only after an in-game change.
    void SetNative(SceneResolution::Size scene_native);
    SceneResolution::Size Native() const { return native; }
    /// The scene size for the shaders' FragCoord conversion, which scales by 1920x1080 / it: the
    /// proxy size relative to 1920x1080 (the same factor as proxy / native).
    SceneResolution::Size ShaderSceneSize() const;
    /// A reduced proxy can stand in for the image: a 1920x1080 scene target or, scaled by
    /// the same factor, a half-resolution (960x540) one (toggle SceneHalfRes).
    bool Eligible(const VideoCore::Image&) const;
    /// The game's depth/stencil (D32S8) can have a proxy: blittable, or resampled by a shader with
    /// stencil export. Without (GeForce GTX 10: neither) the scene cannot be reduced at all, and a
    /// reduced size only added work; resolution changes then apply at the next start.
    bool DepthProxies() const;
    /// Eligible at the full scene size (Native()): the upscaler's inputs, the scene start.
    bool EligibleScene(const VideoCore::Image&) const;
    /// The proxy size of an eligible image: the scene size divided like the native size.
    SceneResolution::Size ProxySize(const VideoCore::Image&, u32 level = 0) const;
    struct Target {
        vk::Image image;
        vk::ImageView view;
        vk::ImageLayout layout;
        vk::ImageUsageFlags usage;
    };
    Target Attachment(VideoCore::ImageId, const VideoCore::ImageViewInfo&);
    Target Read(VideoCore::ImageId, const VideoCore::ImageViewInfo&,
                vk::PipelineStageFlags2 = vk::PipelineStageFlagBits2::eComputeShader,
                vk::AccessFlags2 = vk::AccessFlagBits2::eShaderRead);
    /// A clear of `range` of a color image: the proxies it covers are not resolved first
    /// (BeginClear, before the native transition) and get the clear too (EndClear), so both
    /// copies hold the clear value and neither a resolve nor a later fill is needed.
    void BeginClear(VideoCore::Image& image, const VideoCore::SubresourceRange& range);
    void EndClear(VideoCore::Image& image, const VideoCore::SubresourceRange& range,
                  const vk::ClearColorValue& color);
    /// `caller`: for BB_SCENE_DEBUG, where the native image is accessed.
    void NativeAccess(VideoCore::Image&, vk::AccessFlags2,
                      std::source_location caller = std::source_location::current());
    /// The proxy of `image` for sampling when it holds the current content (else nullopt: the
    /// native image is current). Transitions only after other use (attachment, copy).
    /// Mipmapped targets have one proxy per level: views of one level only.
    std::optional<Target> SampleProxy(const VideoCore::Image& image,
                                      const VideoCore::ImageViewInfo& info);
    /// Whether SampleProxy would return the proxy of `image` for these levels.
    [[nodiscard]] bool ProxyCurrent(const VideoCore::Image& image, u32 level = 0,
                                    u32 levels = 1) const;
    /// SampleProxy for a view obtained before (valid while Generation() is unchanged): the
    /// layout to sample it in.
    vk::ImageLayout PrepareSample(const VideoCore::Image& image, u32 level = 0);
    /// Changes when proxies and their views are destroyed (SetSize).
    [[nodiscard]] u64 Generation() const noexcept {
        return generation;
    }
    /// Whether NativeAccess has work for this image (a reduced-size proxy exists).
    [[nodiscard]] bool Tracks(u64 image_uid) const {
        return !copying && tracked.contains(image_uid);
    }
    /// A whole-image copy from `src` to `dst` (same size and format, colour, one level) while
    /// `src`'s proxy holds its current content: the proxies are copied (dst gets one) and
    /// `dst`'s native image is left to resolve on a native access, instead of resolving `src`
    /// first (the game copies its targets every frame for the next one). False: not done.
    bool CopyProxy(VideoCore::ImageId src, VideoCore::ImageId dst);
    void ResolveAll();
    void CleanupStaleEntries(); ///< Remove proxy entries for images no longer in texture cache
    bool debug = false; ///< BB_SCENE_DEBUG frame: print resolves and fills
    /// Set around a shader's sampled or storage access (Rasterizer::TransitSampled). Those
    /// resolve the proxy first; BB_SCENE_SAMPLED_RESOLVE=0 reads the native image as it is (stale
    /// after a live preset change: black glows and lamp travel fades, so only for diagnostics).
    bool sampled_access = false;
private:
    struct Entry {
        VideoCore::ImageId source{};
        u64 uid = 0;
        u32 level = 0; ///< mip level of the native image this proxy stands for
        VideoCore::HostImage image;
        std::vector<std::pair<VideoCore::ImageViewInfo, vk::UniqueImageView>> views;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;
        SceneResolution::Coherence state;
    };
    static constexpr u64 Key(u64 uid, u32 level) {
        return uid << 4 | level;
    }
    Entry& Get(VideoCore::ImageId, u32 level = 0);
    vk::ImageView View(Entry&, const VideoCore::Image&, const VideoCore::ImageViewInfo&);
    void Copy(Entry&, VideoCore::Image&, bool to_native);
    /// CopyProxy of a depth proxy into a colour one (R32F): through this buffer, as Vulkan
    /// copies no image between depth and colour aspects.
    VkBuffer depth_copy_buffer{};
    struct VmaAllocation_T* depth_copy_allocation{};
    vk::DeviceSize depth_copy_size = 0;
    // Depth/stencil formats without blit support (D32S8 on RADV) are resampled by a
    // fullscreen draw writing gl_FragDepth and, with stencil export, the stencil value.
    vk::FormatFeatureFlags Features(vk::Format) const;
    bool Blittable(vk::Format) const;
    bool ShaderResampled(const VideoCore::Image&) const;
    void Resample(vk::Image src, vk::Image dst, const VideoCore::Image& original,
                  vk::Extent2D dst_size);
    vk::Pipeline ResamplePipeline(vk::Format, bool stencil);
    void CreateResampleResources();
    void Transition(Entry&, vk::ImageAspectFlags, vk::ImageLayout,
                    vk::PipelineStageFlags2, vk::AccessFlags2);
    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    Lookup lookup;
    SceneResolution::Size size;
    SceneResolution::Size native; ///< 1920x1080 by default (SceneResolution::Size{})
    std::unordered_map<u64, std::unique_ptr<Entry>> entries; ///< by Key(uid, level)
    std::unordered_set<u64> tracked; ///< uids with proxies
    bool copying = false;
    bool force_stencil_bits = false; ///< test the portable resampler on any driver
    /// No stencil export and no D32S8 blit (GeForce GTX 10): depth resampled alone, the stencil
    /// cleared (BB_SCENE_STENCIL_ZERO=0: no depth proxies there, the scene stays native). The
    /// game clears the stencil when the scene starts and reads it only inside the scene; one draw
    /// instead of the eight-draw stencil reconstruction (slower than no proxies at all).
    bool stencil_zero = false;
    u64 generation = 0;
    mutable std::unordered_map<vk::Format, vk::FormatFeatureFlags> format_features;
    mutable std::array<vk::FormatFeatureFlags, 256> format_table{};
    mutable std::array<bool, 256> format_known{};
    std::array<std::pair<u64, Entry*>, 8> recent{}; ///< last entries by Key
    u32 recent_next = 0;
    vk::UniqueShaderModule fs_tri_vert, depth_frag, depth_stencil_frag, stencil_bits_frag;
    vk::UniqueDescriptorSetLayout resample_set_layout;
    vk::UniquePipelineLayout resample_layout;
    std::vector<std::pair<std::pair<vk::Format, bool>, vk::UniquePipeline>> resample_pipelines;
};
} // namespace Vulkan
