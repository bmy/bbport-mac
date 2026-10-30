// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "textures/image.h"
#include "shim/bb_env.h"
#include "textures/image_spares.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <fmt/format.h>

#include <vk_mem_alloc.h>

#include "bbport_toggles.h"
#include "common/assert.h"
#include "render/gcn_to_vk.h"
#include "vulkan/instance.h"
#include "render/runtime.h"

namespace VideoCore {

namespace {

/// Every image can be copied, sampled and (but compressed ones) rendered to; color images and
/// compressed ones (through uncompressed views) are storage images too, so a compute clear never
/// needs a new image.
vk::ImageUsageFlags UsageOf(const Vulkan::Instance& instance, const ImageInfo& info) {
    vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eTransferSrc |
                                vk::ImageUsageFlagBits::eTransferDst |
                                vk::ImageUsageFlagBits::eSampled;
    if (info.props.is_block) {
        return usage | vk::ImageUsageFlagBits::eStorage;
    }
    if (info.props.is_depth) {
        return usage | vk::ImageUsageFlagBits::eDepthStencilAttachment;
    }
    usage |= vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage;
    if (instance.IsAttachmentFeedbackLoopLayoutSupported()) {
        usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
    }
    return usage;
}

/// The format features those usages need (storage left out: it is always asked for and would
/// rule out formats the other usages are fine with).
vk::FormatFeatureFlags2 FeaturesOf(vk::ImageUsageFlags usage) {
    using U = vk::ImageUsageFlagBits;
    using F = vk::FormatFeatureFlagBits2;
    vk::FormatFeatureFlags2 features{};
    const std::pair<U, F> map[] = {
        {U::eTransferSrc, F::eTransferSrc},
        {U::eTransferDst, F::eTransferDst},
        {U::eSampled, F::eSampledImage},
        {U::eColorAttachment, F::eColorAttachment},
        {U::eDepthStencilAttachment, F::eDepthStencilAttachment},
    };
    for (const auto& [use, feature] : map) {
        if (usage & use) {
            features |= feature;
        }
    }
    return features;
}

vk::ImageType VulkanType(Gcn::ImageType type) {
    switch (type) {
    case Gcn::ImageType::Color1D:
    case Gcn::ImageType::Color1DArray:
        return vk::ImageType::e1D;
    case Gcn::ImageType::Color2D:
    case Gcn::ImageType::Color2DMsaa:
    case Gcn::ImageType::Color2DArray:
        return vk::ImageType::e2D;
    case Gcn::ImageType::Color3D:
        return vk::ImageType::e3D;
    default:
        UNREACHABLE_MSG("image type {}", u64(type));
    }
}

/// Images of BB_VMA_DEDICATED_MB (4) or more get memory of their own, returned to the driver when
/// they are freed: in shared blocks the collector's evictions left holes a block only returns
/// once wholly empty (~0.9 GB on an RX 7800 XT).
vk::DeviceSize DedicatedFrom() {
    static const vk::DeviceSize bytes = [] {
        const char* env = bb_env("BB_VMA_DEDICATED_MB");
        return vk::DeviceSize(env ? std::strtoull(env, nullptr, 10) : 4) << 20;
    }();
    return bytes;
}

constexpr vk::AccessFlags2 WriteAccess = vk::AccessFlagBits2::eTransferWrite |
                                         vk::AccessFlagBits2::eShaderWrite |
                                         vk::AccessFlagBits2::eMemoryWrite;

/// Whether reaching `layout`, `access` from `state` needs a barrier: another layout or access,
/// or earlier writes to make visible.
bool NeedsBarrier(const Image::State& state, vk::ImageLayout layout, vk::AccessFlags2 access) {
    return state.layout != layout || state.access_mask != access ||
           bool(state.access_mask & WriteAccess);
}

/// bbport (macOS): the images (by address, format and shape) whose compressible backing had to be
/// upgraded (Runtime::UpgradeUsage): one made there again starts with its full usage, without
/// the copy (the game re-creates many of its compute-written post-processing images).
std::mutex full_usage_mutex;
std::unordered_set<u64> full_usage_images;

u64 FullUsageKey(const ImageInfo& info) {
    u64 key = info.guest_address;
    for (const u64 v : {u64(info.pixel_format), u64(info.size.width), u64(info.size.height),
                        u64(info.resources.levels), u64(info.resources.layers),
                        u64(info.num_samples)}) {
        key = (key ^ v) * 0x100000001b3ull;
    }
    return key;
}

bool StartsWithFullUsage(const ImageInfo& info) {
    std::scoped_lock lock{full_usage_mutex};
    return full_usage_images.contains(FullUsageKey(info));
}

} // namespace

void Image::RememberFullUsage() const {
    std::scoped_lock lock{full_usage_mutex};
    if (full_usage_images.size() < 65536) {
        full_usage_images.insert(FullUsageKey(info));
    }
}

Common::IncrementalIdProvider<u64> Image::next_uid{};

namespace {
/// Recently freed texture cache images (HostImage::recyclable), newest last.
struct Recycled {
    VmaAllocator allocator;
    VmaAllocation allocation;
    vk::Image image;
    vk::ImageCreateInfo ci;
    vk::DeviceSize size;
    std::chrono::steady_clock::time_point freed;
};
std::mutex recycle_mutex;
std::vector<Recycled> recycled;
vk::DeviceSize recycled_bytes = 0;
u64 recycle_hits = 0;
/// BB_IMAGE_RECYCLE_MB (0: off; default 256, at most an eighth of the budget, set by the texture
/// cache through TrimRecycled's caller); entries older than BB_IMAGE_RECYCLE_SECONDS (90) go.
vk::DeviceSize RecycleCap() {
    static const vk::DeviceSize cap = [] {
        const char* env = bb_env("BB_IMAGE_RECYCLE_MB");
        return vk::DeviceSize(env && *env ? std::strtoull(env, nullptr, 10) : 256) << 20;
    }();
    return cap;
}
std::chrono::seconds RecycleAge() {
    static const auto age = [] {
        const char* env = bb_env("BB_IMAGE_RECYCLE_SECONDS");
        return std::chrono::seconds(env && *env ? std::strtoull(env, nullptr, 10) : 90);
    }();
    return age;
}
bool SameImage(const vk::ImageCreateInfo& a, const vk::ImageCreateInfo& b) {
    return !a.pNext && !b.pNext && a.flags == b.flags && a.imageType == b.imageType &&
           a.format == b.format && a.extent == b.extent && a.mipLevels == b.mipLevels &&
           a.arrayLayers == b.arrayLayers && a.samples == b.samples && a.tiling == b.tiling &&
           a.usage == b.usage && a.sharingMode == b.sharingMode;
}
/// The oldest (or expired) entries destroyed until at most `keep` bytes stay; under the mutex.
void TrimLocked(vk::DeviceSize keep) {
    const auto now = std::chrono::steady_clock::now();
    size_t drop = 0;
    vk::DeviceSize bytes = recycled_bytes;
    while (drop < recycled.size() &&
           (bytes > keep || now - recycled[drop].freed > RecycleAge())) {
        bytes -= recycled[drop].size;
        ++drop;
    }
    for (size_t i = 0; i < drop; ++i) {
        BbStats::vk_image_bytes.fetch_sub(recycled[i].size, std::memory_order_relaxed);
        vmaDestroyImage(recycled[i].allocator, recycled[i].image, recycled[i].allocation);
    }
    recycled.erase(recycled.begin(), recycled.begin() + drop);
    recycled_bytes = bytes;
}
} // namespace

void HostImage::TrimRecycled(vk::DeviceSize keep_bytes) {
    std::scoped_lock lock{recycle_mutex};
    TrimLocked(keep_bytes);
}

HostImage::~HostImage() {
    Destroy();
}

HostImage::HostImage(HostImage&& other) noexcept
    : device{other.device}, allocator{std::exchange(other.allocator, VK_NULL_HANDLE)},
      allocation{std::exchange(other.allocation, VK_NULL_HANDLE)},
      image{std::exchange(other.image, VK_NULL_HANDLE)}, image_ci{other.image_ci},
      size_bytes{std::exchange(other.size_bytes, 0)}, recyclable{other.recyclable} {}

HostImage& HostImage::operator=(HostImage&& other) noexcept {
    if (this != &other) {
        Destroy();
        device = other.device;
        allocator = std::exchange(other.allocator, VK_NULL_HANDLE);
        allocation = std::exchange(other.allocation, VK_NULL_HANDLE);
        image = std::exchange(other.image, VK_NULL_HANDLE);
        image_ci = other.image_ci;
        size_bytes = std::exchange(other.size_bytes, 0);
        recyclable = other.recyclable;
    }
    return *this;
}

void HostImage::Create(const vk::ImageCreateInfo& create_info) {
    ASSERT(!image);
    image_ci = create_info;
    if (recyclable && RecycleCap() != 0) {
        std::scoped_lock lock{recycle_mutex};
        for (size_t i = recycled.size(); i-- > 0;) {
            auto& r = recycled[i];
            if (r.allocator == allocator && SameImage(r.ci, image_ci)) {
                allocation = r.allocation;
                image = r.image;
                size_bytes = r.size;
                recycled_bytes -= r.size;
                recycled.erase(recycled.begin() + i);
                if (++recycle_hits == 1 || recycle_hits % 256 == 0) {
                    std::printf("Texture cache: %llu images made again from freed ones (%llu MiB "
                                "kept for that)\n",
                                static_cast<unsigned long long>(recycle_hits),
                                static_cast<unsigned long long>(recycled_bytes >> 20));
                }
                return;
            }
        }
    }
    // A spare made ahead on the spare thread (image_spares.h): no create or bind on this one.
    if (recyclable && device && ImageSpares::Take(allocator, image_ci, image, allocation, size_bytes)) {
        return;
    }
    const VmaAllocationCreateFlags dedicated =
        ImageSpares::MemoryFlags(device, image_ci) & VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT | dedicated,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
    };
    const VkImageCreateInfo vk_create_info = image_ci;
    VkImage handle{};
    VmaAllocationInfo alloc_info{};
    BbStats::VkMemTimer timer{BbStats::t_vk_alloc, BbStats::n_vk_alloc};
    const auto split_now = [] { return std::chrono::steady_clock::now(); };
    const auto split_ns = [](auto a, auto b) {
        return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
    };
    // vmaCreateImage in its three steps, each timed (the frame stats' stall lines).
    const auto t0 = split_now();
    VkResult result = VK_SUCCESS;
    if (!device) {
        result = vmaCreateImage(allocator, &vk_create_info, &alloc_ci, &handle, &allocation, &alloc_info);
    } else {
        result = VkResult(device.createImage(&image_ci, nullptr, reinterpret_cast<vk::Image*>(&handle)));
    }
    const bool split = bool(device);
    const auto t1 = split_now();
    if (split && result == VK_SUCCESS) {
        VmaAllocationCreateInfo split_ci = alloc_ci;
        split_ci.usage = VMA_MEMORY_USAGE_GPU_ONLY; // AUTO needs the create info
        result = vmaAllocateMemoryForImage(allocator, handle, &split_ci, &allocation, &alloc_info);
        if (result != VK_SUCCESS) {
            device.destroyImage(vk::Image{handle});
        }
    }
    const auto t2 = split_now();
    if (split && result == VK_SUCCESS) {
        result = vmaBindImageMemory(allocator, allocation, handle);
    }
    const auto t3 = split_now();
    BbStats::t_img_create.fetch_add(split_ns(t0, t1), std::memory_order_relaxed);
    BbStats::t_img_alloc.fetch_add(split_ns(t1, t2), std::memory_order_relaxed);
    BbStats::t_img_bind.fetch_add(split_ns(t2, t3), std::memory_order_relaxed);
    if (static const u64 slow_us = [] {
            const char* env = bb_env("BB_SLOW_BIND_LOG"); // microseconds
            return env ? std::strtoull(env, nullptr, 10) : 0ull;
        }();
        slow_us && split_ns(t1, t3) > slow_us * 1000) {
        std::printf("Slow image: %ux%ux%u %u levels %u layers format %d usage %#x, %llu KiB%s: memory %.1f ms, bind %.1f ms\n",
                    image_ci.extent.width, image_ci.extent.height, image_ci.extent.depth,
                    image_ci.mipLevels, image_ci.arrayLayers, int(image_ci.format),
                    u32(VkImageUsageFlags(image_ci.usage)), (unsigned long long)(alloc_info.size >> 10),
                    dedicated ? " dedicated" : "", split_ns(t1, t2) / 1e6, split_ns(t2, t3) / 1e6);
    }
    ASSERT_MSG(result == VK_SUCCESS, "image allocation: {}", vk::to_string(vk::Result{result}));
    image = handle;
    size_bytes = alloc_info.size;
    BbStats::image_new_bytes.fetch_add(size_bytes, std::memory_order_relaxed);
    BbStats::vk_image_bytes.fetch_add(size_bytes, std::memory_order_relaxed);
    if (recyclable && device) {
        ImageSpares::NoteMade(image_ci, size_bytes); // its shape gets spares from now on
    }
}

u32 ImageSpares::MemoryFlags(vk::Device device, const vk::ImageCreateInfo& ci) {
    VmaAllocationCreateFlags flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
    if (device && DedicatedFrom() != 0) {
        const vk::DeviceImageMemoryRequirements query{.pCreateInfo = &ci};
        if (device.getImageMemoryRequirements(query).memoryRequirements.size >= DedicatedFrom()) {
            flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        }
    }
    return flags;
}

void HostImage::Destroy() {
    if (!image) {
        return;
    }
    if (recyclable && RecycleCap() != 0 && size_bytes <= RecycleCap()) {
        std::scoped_lock lock{recycle_mutex};
        recycled.push_back({allocator, allocation, image, image_ci, size_bytes,
                            std::chrono::steady_clock::now()});
        recycled_bytes += size_bytes;
        TrimLocked(RecycleCap());
        image = vk::Image{};
        allocation = {};
        return;
    }
    BbStats::vk_image_bytes.fetch_sub(size_bytes, std::memory_order_relaxed);
    BbStats::VkMemTimer timer{BbStats::t_vk_free, BbStats::n_vk_free};
    vmaDestroyImage(allocator, image, allocation);
    image = vk::Image{};
    allocation = {};
}

Image::Image(const Vulkan::Instance& instance, Vulkan::Runtime& runtime_,
             Common::SlotVector<ImageView>& slot_image_views_, const ImageInfo& info_)
    : guest_begin{info_.guest_address}, guest_end{info_.guest_address + info_.guest_size},
      info{info_}, guest_info{info_}, runtime{&runtime_}, slot_image_views{&slot_image_views_} {
    BbStats::Timer timer{BbStats::t_image_create};
    if (info.pixel_format == vk::Format::eUndefined) {
        return; // a placeholder (a stencil image's address)
    }
    image_uid = next_uid.Next();
    mip_hashes.resize(info.resources.levels);

    vk::ImageCreateFlags create_flags =
        vk::ImageCreateFlagBits::eMutableFormat | vk::ImageCreateFlagBits::eExtendedUsage;
    if (info.props.is_depth && instance.IsCompressibleImages()) {
        // bbport (macOS): a depth image is only ever viewed in its own format (ImageView), and
        // a mutable format would keep Metal from compressing it.
        create_flags = {};
    }
    if (info.props.is_volume) {
        create_flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
        if (instance.Is2dViewOf3dSupported()) {
            create_flags |= vk::ImageCreateFlagBits::e2DViewCompatibleEXT;
        }
    }
    if (info.props.is_block && instance.IsBlockTexelViewSupported()) {
        create_flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
    }
    usage_flags = UsageOf(instance, info);
    format_features = FeaturesOf(usage_flags);
    if (info.props.is_depth) {
        aspect_mask = vk::ImageAspectFlagBits::eDepth;
        if (info.props.has_stencil) {
            aspect_mask |= vk::ImageAspectFlagBits::eStencil;
        }
    }

    const vk::Format format = instance.GetSupportedFormat(info.pixel_format, format_features);
    const vk::ImageType vk_type = VulkanType(info.type);
    const auto query = [&] {
        return instance.GetPhysicalDevice().getImageFormatProperties2(vk::PhysicalDeviceImageFormatInfo2{
            .format = format,
            .type = vk_type,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = usage_flags,
            .flags = create_flags,
        });
    };
    auto properties = query();
    // Storage is asked for every color and compressed image; a driver may refuse it for a format
    // (AMD's Windows driver: BC6H). Such an image is made without it rather than invalid (its
    // views follow usage_flags), so the game's sampling of it stays defined.
    if (properties.result == vk::Result::eErrorFormatNotSupported &&
        (usage_flags & vk::ImageUsageFlagBits::eStorage)) {
        usage_flags &= ~vk::ImageUsageFlagBits::eStorage;
        properties = query();
        if (properties.result == vk::Result::eSuccess) {
            static std::atomic<int> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                LOG_WARNING(Render_Vulkan, "image format {} type {}: no storage use on this driver, made without it",
                            vk::to_string(format), vk::to_string(vk_type));
            }
        }
    }
    if (properties.result == vk::Result::eErrorFormatNotSupported) {
        LOG_ERROR(Render_Vulkan, "image format {} type {} unsupported (flags {}, usage {})",
                  vk::to_string(format), vk::to_string(vk_type), vk::to_string(create_flags),
                  vk::to_string(usage_flags));
    }
    supported_samples = properties.result == vk::Result::eSuccess
                            ? properties.value.imageFormatProperties.sampleCounts
                            : vk::SampleCountFlagBits::e1;

    // bbport (macOS): color images start without storage usage, mutable formats and the feedback
    // loop usage, so that Metal can compress them; a binding that needs one upgrades the image
    // (Runtime::UpgradeUsage). Not block-compressed images (never compressed by Metal), 3D ones
    // (2D views), multisampled ones (upgrades copy single-sample images only) or those in a
    // stand-in format (the blitter views them in the guest's format).
    full_usage = usage_flags;
    full_create_flags = create_flags;
    compressible = instance.IsCompressibleImages() && !info.props.is_depth &&
                   !info.props.is_block && !info.props.is_volume && info.num_samples <= 1 &&
                   format == info.pixel_format;
    // (One upgraded at this address before starts with its full usage; its proxies may still
    // start compressible: `compressible` is the kind of image.)
    const bool start_compressible = compressible && !StartsWithFullUsage(info);
    if (compressible && !start_compressible) {
        // Like an upgraded backing: without the feedback loop usage until a loop needs it
        // (with it KosmicKrisp never optimizes the texture).
        usage_flags &= ~vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
    }
    if (start_compressible) {
        create_flags &= ~(vk::ImageCreateFlagBits::eMutableFormat |
                          vk::ImageCreateFlagBits::eExtendedUsage);
        usage_flags &= ~(vk::ImageUsageFlagBits::eStorage |
                         vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT);
    }

    backing = &backing_images.emplace_back();
    backing->num_samples = info.num_samples;
    backing->compressible = start_compressible;
    backing->image = HostImage{instance.GetDevice(), instance.GetAllocator()};
    backing->image.recyclable = true;
    backing->image.Create(vk::ImageCreateInfo{
        .flags = create_flags,
        .imageType = vk_type,
        .format = format,
        .extent{info.size.width, info.size.height, info.size.depth},
        .mipLevels = info.resources.levels,
        .arrayLayers = info.resources.layers,
        .samples = Vulkan::GcnToVk::NumSamples(info.num_samples, supported_samples),
        .tiling = vk::ImageTiling::eOptimal,
        .usage = usage_flags,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    Vulkan::SetObjectName(instance.GetDevice(), GetImage(), "image {}x{}x{} {} {} {:#x}+{:#x} L{} M{} S{}",
                          info.size.width, info.size.height, info.size.depth,
                          Gcn::NameOf(info.tile_mode), vk::to_string(info.pixel_format),
                          info.guest_address, info.guest_size, info.resources.layers,
                          info.resources.levels, info.num_samples);
}

Image::~Image() = default;

ImageView& Image::FindView(const ImageViewInfo& view_info, bool ensure_guest_samples) {
    if (ensure_guest_samples && (backing->num_samples > 1) != (info.num_samples > 1)) {
        runtime->SetBackingSamples(this, info.num_samples);
    }
    if (NeedsFullUsage(view_info)) {
        runtime->UpgradeUsage(this, view_info.is_storage ? "storage view" : "view format");
    }
    auto& infos = backing->image_view_infos;
    auto& ids = backing->image_view_ids;
    if (backing->last_view < infos.size() && infos[backing->last_view] == view_info) {
        return (*slot_image_views)[ids[backing->last_view]];
    }
    if (const auto it = std::ranges::find(infos, view_info); it != infos.end()) {
        backing->last_view = u32(it - infos.begin());
        return (*slot_image_views)[ids[backing->last_view]];
    }
    const ImageViewId id = slot_image_views->insert(runtime->GetInstance(), view_info, *this);
    infos.push_back(view_info);
    ids.push_back(id);
    return (*slot_image_views)[id];
}

bool Image::NeedsFullUsage(const ImageViewInfo& view_info) const {
    if (!backing || !backing->compressible) {
        return false;
    }
    if (view_info.is_storage) {
        return true;
    }
    if (view_info.format == backing->image.image_ci.format) {
        return false; // the common case, without the format lookup
    }
    const vk::Format format =
        runtime->GetInstance().GetSupportedFormat(view_info.format, format_features);
    return !ViewableWithoutMutableFormat(format, backing->image.image_ci.format);
}

void Image::GetBarriers(Barriers& out, vk::ImageLayout dst_layout, vk::AccessFlags2 dst_mask,
                        vk::PipelineStageFlags2 dst_stage, std::optional<SubresourceRange> range) {
    auto& whole = backing->state;
    auto& parts = backing->subresource_states;
    const bool partial = range && (range->base != SubresourceBase{} ||
                                   range->extent != info.resources);
    const State target{dst_stage, dst_mask, dst_layout};
    const auto barrier = [&](const State& from, u32 level, u32 levels, u32 layer, u32 layers) {
        out.push_back(vk::ImageMemoryBarrier2{
            .srcStageMask = from.pl_stage,
            .srcAccessMask = from.access_mask,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_mask,
            .oldLayout = from.layout,
            .newLayout = dst_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = GetImage(),
            .subresourceRange{aspect_mask, level, levels, layer, layers},
        });
    };

    if (!partial && parts.empty()) {
        backing->settled = false;
        if (NeedsBarrier(whole, dst_layout, dst_mask)) {
            barrier(whole, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS);
        }
        whole = target;
        return;
    }

    // Per level and layer: the parts in the range (all of them when the whole image goes to
    // one state again), runs of layers in the same state in one barrier each.
    const u32 num_layers = info.resources.layers;
    static const bool barrier_stats = bb_env_on("BB_BARRIER_STATS");
    if (barrier_stats) {
        // BB_BARRIER_STATS=1 (diagnostics): images whose subresource states cost the most.
        struct Row {
            u64 calls = 0, partial = 0, visited = 0, fills = 0;
            std::string what;
        };
        static std::mutex mutex;
        static std::unordered_map<u64, Row> rows;
        static auto last = std::chrono::steady_clock::now();
        std::scoped_lock lk{mutex};
        auto& row = rows[info.guest_address ^ (u64(info.resources.layers) << 48)];
        if (row.what.empty()) {
            row.what = fmt::format("{:#x} {}x{}x{} {} levels {} layers {} type {}", info.guest_address,
                                   info.size.width, info.size.height, info.size.depth,
                                   vk::to_string(info.pixel_format), info.resources.levels,
                                   info.resources.layers, u32(info.type));
        }
        ++row.calls;
        row.partial += partial;
        row.fills += parts.empty() ? info.resources.levels * num_layers : 0;
        row.visited += partial ? u64(range->extent.levels) * range->extent.layers
                               : u64(info.resources.levels) * num_layers;
        if (std::chrono::steady_clock::now() - last > std::chrono::seconds(10)) {
            last = std::chrono::steady_clock::now();
            std::vector<std::pair<u64, const Row*>> top;
            for (const auto& [key, r] : rows) {
                top.emplace_back(r.visited + r.fills, &r);
            }
            std::ranges::sort(top, std::greater{}, &std::pair<u64, const Row*>::first);
            std::printf("Barrier stats (10 s, subresource walks):\n");
            for (size_t i = 0; i < top.size() && i < 10; ++i) {
                const Row& r = *top[i].second;
                std::printf("  %s: %llu calls (%llu partial), %llu visited, %llu filled\n",
                            r.what.c_str(), (unsigned long long)r.calls,
                            (unsigned long long)r.partial, (unsigned long long)r.visited,
                            (unsigned long long)r.fills);
            }
            rows.clear();
        }
    }
    const bool fresh = parts.empty();
    if (fresh) {
        parts.assign(info.resources.levels * num_layers, whole);
    }
    const SubresourceRange walk = partial ? *range : SubresourceRange{{}, info.resources};
    if (backing->settled && backing->settled_state == target &&
        (backing->settled_uniform || backing->settled_range == walk) &&
        !NeedsBarrier(target, dst_layout, dst_mask)) {
        whole = target;
        return;
    }
    // After the walk every part is in `target` when the rest was in it already.
    const bool uniform_after =
        walk == SubresourceRange{{}, info.resources} || (fresh && whole == target) ||
        (backing->settled && backing->settled_uniform && backing->settled_state == target);
    const u32 first_level = walk.base.level;
    const u32 end_level = first_level + walk.extent.levels;
    const u32 first_layer = walk.base.layer;
    const u32 end_layer = first_layer + walk.extent.layers;
    for (u32 level = first_level; level < end_level; ++level) {
        for (u32 layer = first_layer; layer < end_layer;) {
            const u32 index = level * num_layers + layer;
            ASSERT(index < parts.size());
            const State from = parts[index];
            u32 run = 1;
            while (layer + run < end_layer && parts[index + run] == from) {
                ++run;
            }
            if (NeedsBarrier(from, dst_layout, dst_mask)) {
                barrier(from, level, 1, layer, run);
            }
            std::fill_n(parts.begin() + index, run, target);
            layer += run;
        }
    }
    if (!partial) {
        parts.clear(); // all in one state: whole-image transitions again
    }
    backing->settled = partial;
    backing->settled_range = walk;
    backing->settled_state = target;
    backing->settled_uniform = partial && uniform_after;
    whole = target;
}

} // namespace VideoCore
