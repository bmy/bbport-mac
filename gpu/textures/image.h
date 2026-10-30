// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: a guest image on the host: its Vulkan image (one per sample count it has been
// used with), its views, its layout and access state for barriers, and what the cache knows of
// it (dirty flags, the pages watched, when it was used).
#pragma once

#include <deque>
#include <optional>
#include <vector>

#include <boost/container/small_vector.hpp>
#include <boost/container/static_vector.hpp>

#include "common/enum.h"
#include "common/incremental_id.h"
#include "common/types.h"
#include "textures/image_info.h"
#include "textures/image_view.h"
#include "vulkan/vk_common.h"

namespace Vulkan {
class Instance;
class Runtime;
} // namespace Vulkan

VK_DEFINE_HANDLE(VmaAllocation)
VK_DEFINE_HANDLE(VmaAllocator)

namespace VideoCore {

enum ImageFlagBits : u32 {
    Empty = 0,
    MaybeCpuDirty = 1 << 0, ///< a page under it was written; its hash tells whether it was
    CpuDirty = 1 << 1,      ///< the guest wrote it
    GpuDirty = 1 << 2,      ///< the GPU wrote its memory as a buffer (reupload from there)
    Dirty = MaybeCpuDirty | CpuDirty | GpuDirty,
    GpuModified = 1 << 3, ///< the GPU wrote it as an image
    Registered = 1 << 6,  ///< in the cache's page table
    Picked = 1 << 7,      ///< seen in the current region walk
};
DECLARE_ENUM_FLAG_OPERATORS(ImageFlagBits)

/// A Vulkan image with its memory (VMA).
class HostImage {
public:
    HostImage() = default;
    HostImage(vk::Device device_, VmaAllocator allocator_) : device{device_}, allocator{allocator_} {}
    ~HostImage();
    HostImage(const HostImage&) = delete;
    HostImage& operator=(const HostImage&) = delete;
    HostImage(HostImage&& other) noexcept;
    HostImage& operator=(HostImage&& other) noexcept;

    void Create(const vk::ImageCreateInfo& create_info);
    void Destroy();
    /// Pooled images given back: all (`keep_bytes` 0) or down to `keep_bytes`.
    static void TrimRecycled(vk::DeviceSize keep_bytes);

    operator vk::Image() const {
        return image;
    }
    explicit operator bool() const {
        return bool(image);
    }

    vk::Device device{};
    VmaAllocator allocator{};
    VmaAllocation allocation{};
    vk::Image image{};
    vk::ImageCreateInfo image_ci{};
    vk::DeviceSize size_bytes{};
    /// The texture cache's images: a destroyed one goes to a pool of recently freed images, and a
    /// new one with the same create info takes it from there (no vkCreateImage, no allocation).
    /// The game frees and allocates the same render targets again (a menu opened: ~25 images,
    /// 106 MiB, 100-130 ms on a GTX 1050 laptop, dedicated allocations being slow on Windows).
    bool recyclable = false;
};

struct Image {
    Image(const Vulkan::Instance& instance, Vulkan::Runtime& runtime,
          Common::SlotVector<ImageView>& slot_image_views, const ImageInfo& info);
    ~Image();
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;
    Image(Image&&) = default;
    Image& operator=(Image&&) = default;

    bool Overlaps(VAddr address, size_t size) const noexcept {
        return info.guest_address < address + size && address < info.guest_address + info.guest_size;
    }
    vk::Image GetImage() const {
        return backing->image.image;
    }
    vk::DeviceSize GetHostImageSize() const {
        return backing->image.size_bytes;
    }
    bool IsTracked() const {
        return track_addr != 0 && track_addr_end != 0;
    }
    /// Written by the GPU and not changed since by anything else.
    bool SafeToDownload() const {
        return True(flags & ImageFlagBits::GpuModified) && False(flags & ImageFlagBits::Dirty);
    }
    void AssociateDepth(ImageId id, u64 uid) {
        depth_id = id;
        depth_uid = uid;
    }
    void DisassociateDepth() {
        depth_id = {};
        depth_uid = {};
    }

    /// The view with `view_info` (created on first use). `ensure_guest_samples`: switches to the
    /// backing of the guest's sample count first. A compressible backing that cannot have the
    /// view is upgraded first (Runtime::UpgradeUsage).
    ImageView& FindView(const ImageViewInfo& view_info, bool ensure_guest_samples = true);

    /// bbport (macOS, Instance::IsCompressibleImages): the current backing was made without
    /// storage usage, mutable formats and the feedback loop usage, so that Metal can compress
    /// it, and `view_info` needs one of them (a storage view, a view in another layout).
    [[nodiscard]] bool NeedsFullUsage(const ImageViewInfo& view_info) const;
    /// bbport (macOS): an image made again with this one's address, format and shape starts
    /// with its full usage (Runtime::UpgradeUsage calls it).
    void RememberFullUsage() const;
    /// bbport (macOS): a feedback loop needs the feedback loop usage, which the current backing
    /// lacks (upgraded backings get it only for a feedback loop: with it KosmicKrisp never
    /// optimizes, let alone compresses, the texture).
    [[nodiscard]] bool LacksFeedbackUsage() const {
        return backing && (full_usage & vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT) &&
               !(backing->image.image_ci.usage & vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT);
    }

    using Barriers = boost::container::small_vector<vk::ImageMemoryBarrier2, 32>;
    /// The barriers that bring `range` (else the whole image) to `dst_layout`, `dst_mask` at
    /// `dst_stage`; the state is updated as if they ran.
    void GetBarriers(Barriers& out, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                     vk::PipelineStageFlags2 dst_stage,
                     std::optional<SubresourceRange> range = {});

    // Read by every binding of the image: together, ahead of the large ImageInfo.
    ImageFlagBits flags = ImageFlagBits::Dirty;
    struct Binding {
        u32 is_bound : 1;
        u32 is_target : 1;
        u32 needs_rebind : 1;
        u32 force_general : 1;
        /// bbport (macOS): sampled in a range a target of the same draw draws to (with
        /// compressible images a target sampled at another mip is no feedback loop).
        u32 feedback : 1;
    } binding{};
    struct {
        u32 texture : 1;
        u32 storage : 1;
        u32 render_target : 1;
        u32 depth_target : 1;
        u32 vo_surface : 1;
    } usage{};
    bool scene_proxy = false; ///< SceneTargets holds a reduced-size proxy of it
    u8 prompt_icon = 0;       ///< one of the game's key guide icons (BbPrompts::IconOf + 1)
    u8 prompt_atlas = 0;      ///< one of the HUD's icon atlases (BbPrompts::AtlasOf + 1)
    u32 prompt_upload = 0;    ///< the upload that gave it its bytes (ButtonPrompts::Atlas)
    /// Its memory may hold one of the key guide icons (32x32, one level, block-compressed): the
    /// bytes are checked at each bind until they are one (a copy the GPU made is never uploaded).
    bool MaybePromptIcon() const {
        return prompt_icon == 0 && info.size.width == 32 && info.size.height == 32 &&
               info.resources.levels == 1 && info.resources.layers == 1 && info.props.is_block &&
               info.guest_size >= 1024;
    }
    VAddr track_addr = 0;     ///< the pages watched: [track_addr, track_addr_end)
    VAddr track_addr_end = 0;
    VAddr guest_begin = 0; ///< info.guest_address
    VAddr guest_end = 0;   ///< its end
    struct BackingImage;
    BackingImage* backing{};
    mutable u64 lru_touched_tick = ~0ull; ///< the GC tick of the last LRU touch
    u64 tick_accessed_last{};
    u64 gpu_write_tick{}; ///< the scheduler tick of the last binding the GPU may write
    ImageId depth_id{};
    u64 depth_uid{};
    u64 image_uid{};

    ImageInfo info;
    /// The guest's description of it (`info` without the levels texture_quality left out).
    ImageInfo guest_info;
    Vulkan::Runtime* runtime;
    Common::SlotVector<ImageView>* slot_image_views;
    vk::ImageAspectFlags aspect_mask = vk::ImageAspectFlagBits::eColor;
    vk::SampleCountFlags supported_samples = vk::SampleCountFlagBits::e1;
    vk::ImageUsageFlags usage_flags; ///< the current backing's
    vk::FormatFeatureFlags2 format_features;
    /// What every backing had before compressible images, and gets when upgraded.
    vk::ImageUsageFlags full_usage;
    vk::ImageCreateFlags full_create_flags;
    /// A kind of image made compressible (bbport, macOS; its first backing unless one at its
    /// address needed full usage before): its proxies may start compressible too.
    bool compressible = false;

    struct State {
        vk::PipelineStageFlags2 pl_stage = vk::PipelineStageFlagBits2::eAllCommands;
        vk::AccessFlags2 access_mask = vk::AccessFlagBits2::eNone;
        vk::ImageLayout layout = vk::ImageLayout::eUndefined;

        bool operator==(const State&) const = default;
    };
    struct BackingImage {
        HostImage image;
        State state;
        /// Per level and layer once parts of it were transitioned on their own.
        std::vector<State> subresource_states;
        /// The range the last per-part transition set to `settled_state` (every part in it is in
        /// that state; nothing else changes the parts): the same read-only transition again is
        /// nothing to do without walking them (a cube array of 2048 faces bound for every draw).
        SubresourceRange settled_range{};
        State settled_state{};
        bool settled = false;
        /// Every part is in `settled_state`, not only those in `settled_range`: a read of any other
        /// range in that state needs no walk either (a cube array sampled a few faces at a time by
        /// ~230 draws a frame, 48 parts walked each).
        bool settled_uniform = false;
        boost::container::small_vector<ImageViewInfo, 4> image_view_infos;
        boost::container::small_vector<ImageViewId, 4> image_view_ids;
        u32 last_view = 0; ///< index of the view FindView returned last
        u32 num_samples;  ///< 0 once retired (replaced by an upgraded backing)
        /// Made without storage usage, mutable formats and the feedback loop usage.
        bool compressible = false;
    };
    std::deque<BackingImage> backing_images;
    boost::container::static_vector<u64, 16> mip_hashes{};
    /// The texture upload budget (TextureCache::RefreshImage): its levels below this one are not
    /// uploaded yet, its sampled views start at it (min LOD). 0: all uploaded.
    u8 resident_level = 0;
    u64 lru_id{};
    u64 hash{};
    u64 corner_hash{}; ///< bbport: diagnostics (TextureCache::RefreshImage)
    u32 uploads{};     ///< bbport: BB_IMAGE_TRACE

private:
    static Common::IncrementalIdProvider<u64> next_uid;
};

} // namespace VideoCore
