// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_object_motion.h"
#include "shim/bb_env.h"

#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <array>
#include <cstddef>
#include <filesystem>
#include <tuple>

#include <vk_mem_alloc.h>
#include "bbport_settings.h"
#include "common/path_util.h"
#include "bbport_toggles.h"
#include "shader/runtime_info.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"

namespace Vulkan {

namespace {

bool CreateBuffer(const Instance& instance, vk::DeviceSize size, bool host, vk::Buffer& buffer,
                  VmaAllocation& allocation, void** mapped, u64& address) {
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
    };
    VmaAllocationCreateInfo alloc_ci{
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    if (host) {
        alloc_ci.requiredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        alloc_ci.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                         VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    } else {
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    }
    VkBuffer raw{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &raw, &allocation, &info) !=
        VK_SUCCESS) {
        return false;
    }
    buffer = raw;
    if (mapped) {
        *mapped = info.pMappedData;
    }
    address = instance.GetDevice().getBufferAddress({.buffer = buffer});
    return address != 0;
}

/// The opaque capture addresses of the two buffers and their memory from the last session, kept
/// in the cache directory with the device and driver they are valid for.
struct ReplayAddresses {
    u32 magic = 0x524d4242; // "BBMR"
    u32 vendor = 0, device = 0, driver = 0;
    std::array<u8, VK_UUID_SIZE> cache_uuid{};
    u64 params_size = 0, positions_size = 0;
    u64 buffer[2]{}, memory[2]{};
};

std::filesystem::path ReplayPath() {
    return Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / "object_motion_addresses.bin";
}

ReplayAddresses ReplayIdentity(const Instance& instance, u64 params_size, u64 positions_size) {
    const auto& p = instance.GetProperties();
    ReplayAddresses r{};
    r.vendor = p.vendorID;
    r.device = p.deviceID;
    r.driver = p.driverVersion;
    std::memcpy(r.cache_uuid.data(), p.pipelineCacheUUID.data(), VK_UUID_SIZE);
    r.params_size = params_size;
    r.positions_size = positions_size;
    return r;
}

/// A buffer (and its own memory) at the address an earlier session gave it, when `buffer_address`
/// and `memory_address` are its opaque capture addresses; a new address when they are 0 or the
/// driver cannot place it there. The opaque addresses in use are returned through them.
bool CreateReplayBuffer(const Instance& instance, vk::DeviceSize size, bool host, vk::Buffer& buffer,
                        VkDeviceMemory& memory, void** mapped, u64& address, u64& buffer_address,
                        u64& memory_address) {
    const VkDevice device = instance.GetDevice();
    for (int attempt = 0; attempt < 2; ++attempt) {
        const bool replay = attempt == 0 && buffer_address && memory_address;
        if (attempt == 1 && !(buffer_address && memory_address)) {
            break; // the first attempt was already a fresh one
        }
        if (attempt == 1) {
            buffer_address = memory_address = 0;
        }
        const VkBufferOpaqueCaptureAddressCreateInfo buffer_replay{
            .sType = VK_STRUCTURE_TYPE_BUFFER_OPAQUE_CAPTURE_ADDRESS_CREATE_INFO,
            .opaqueCaptureAddress = buffer_address,
        };
        const VkBufferCreateInfo buffer_ci{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = replay ? &buffer_replay : nullptr,
            .flags = VK_BUFFER_CREATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
            .size = size,
            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        };
        VkBuffer raw{};
        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCreateBuffer(device, &buffer_ci, nullptr, &raw) != VK_SUCCESS) {
            continue;
        }
        VkMemoryRequirements req{};
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferMemoryRequirements(device, raw, &req);
        const auto& props = instance.GetMemoryProperties();
        const VkMemoryPropertyFlags want =
            host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                 : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        u32 type = UINT32_MAX;
        for (u32 i = 0; i < props.memoryTypeCount && type == UINT32_MAX; ++i) {
            if ((req.memoryTypeBits & (1u << i)) &&
                (VkMemoryPropertyFlags(props.memoryTypes[i].propertyFlags) & want) == want) {
                type = i;
            }
        }
        if (type == UINT32_MAX) {
            VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer(device, raw, nullptr);
            return false;
        }
        const VkMemoryOpaqueCaptureAddressAllocateInfo memory_replay{
            .sType = VK_STRUCTURE_TYPE_MEMORY_OPAQUE_CAPTURE_ADDRESS_ALLOCATE_INFO,
            .opaqueCaptureAddress = memory_address,
        };
        const VkMemoryAllocateFlagsInfo flags{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .pNext = replay ? &memory_replay : nullptr,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT |
                     VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT,
        };
        const VkMemoryAllocateInfo alloc{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &flags,
            .allocationSize = req.size,
            .memoryTypeIndex = type,
        };
        VkDeviceMemory mem{};
        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkAllocateMemory(device, &alloc, nullptr, &mem) != VK_SUCCESS) {
            VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer(device, raw, nullptr);
            continue;
        }
        void* ptr = nullptr;
        if (VULKAN_HPP_DEFAULT_DISPATCHER.vkBindBufferMemory(device, raw, mem, 0) != VK_SUCCESS ||
            (host && VULKAN_HPP_DEFAULT_DISPATCHER.vkMapMemory(device, mem, 0, VK_WHOLE_SIZE, 0, &ptr) != VK_SUCCESS)) {
            VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer(device, raw, nullptr);
            VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(device, mem, nullptr);
            continue;
        }
        const VkBufferDeviceAddressInfo info{.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
                                             .buffer = raw};
        const VkDeviceMemoryOpaqueCaptureAddressInfo mem_info{
            .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_OPAQUE_CAPTURE_ADDRESS_INFO, .memory = mem};
        buffer = raw;
        memory = mem;
        if (mapped) {
            *mapped = ptr;
        }
        address = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferDeviceAddress(device, &info);
        buffer_address = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetBufferOpaqueCaptureAddress(device, &info);
        memory_address = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceMemoryOpaqueCaptureAddress(device, &mem_info);
        return address != 0;
    }
    return false;
}

} // namespace

bool ObjectMotion::CreateBuffers(vk::DeviceSize params_size, vk::DeviceSize positions_size,
                                 void*& mapped, u64& params_address, u64& positions_address) {
    if (!instance.HasBufferAddressReplay()) {
        return CreateBuffer(instance, params_size, true, params_buffer, params_allocation, &mapped,
                            params_address) &&
               CreateBuffer(instance, positions_size, false, positions_buffer,
                            positions_allocation, nullptr, positions_address);
    }
    // Issue: the motion vertex shaders embed these addresses; with new ones every session they
    // were translated and their pipelines compiled again at every start (a stutter on entering
    // the level: 0.7 s on the Steam Deck). The same addresses keep them in the pipeline cache.
    const ReplayAddresses identity = ReplayIdentity(instance, params_size, positions_size);
    ReplayAddresses saved{};
    if (std::FILE* file = std::fopen(ReplayPath().string().c_str(), "rb")) {
        if (std::fread(&saved, sizeof(saved), 1, file) != 1) {
            saved = {};
        }
        std::fclose(file);
    }
    const bool usable = std::memcmp(&saved, &identity, offsetof(ReplayAddresses, buffer)) == 0;
    ReplayAddresses now = identity;
    if (usable) {
        std::memcpy(now.buffer, saved.buffer, sizeof(now.buffer));
        std::memcpy(now.memory, saved.memory, sizeof(now.memory));
    }
    if (!CreateReplayBuffer(instance, params_size, true, params_buffer, params_memory, &mapped,
                            params_address, now.buffer[0], now.memory[0]) ||
        !CreateReplayBuffer(instance, positions_size, false, positions_buffer, positions_memory,
                            nullptr, positions_address, now.buffer[1], now.memory[1])) {
        return false;
    }
    const bool same = usable && std::memcmp(now.buffer, saved.buffer, sizeof(now.buffer)) == 0 &&
                      std::memcmp(now.memory, saved.memory, sizeof(now.memory)) == 0;
    if (!same) {
        std::error_code ec;
        std::filesystem::create_directories(ReplayPath().parent_path(), ec);
        if (std::FILE* file = std::fopen(ReplayPath().string().c_str(), "wb")) {
            std::fwrite(&now, sizeof(now), 1, file);
            std::fclose(file);
        }
    }
    std::printf("Object motion: buffer addresses %s\n",
                same ? "as in the last session (motion shaders cached)"
                     : "new (motion shaders compiled this session, cached from the next)");
    return true;
}

ObjectMotion::ObjectMotion(const Instance& instance_, Scheduler& scheduler_)
    : instance{instance_}, scheduler{scheduler_},
      PositionsPerFrame{[&] {
          if (const char* env = bb_env("BB_MOTION_VERTICES_MI")) {
              return u32(std::clamp(std::strtoul(env, nullptr, 10), 1ul, 8ul)) << 20;
          }
          return instance_.GetTotalMemoryBudget() <= (u64{4} << 30) ? 1u << 20 : 4u << 20;
      }()} {
    // Pipeline selection limits the extra attachment and vertex stores to likely
    // animated draws. A setting or environment override can still disable the path.
    if (!BbSettings::Get().object_motion ||
        BbSettings::Get().upscaler == BbSettings::UpscalerOff) {
        std::printf("Object motion: off (enable in menu or BB_OBJECT_MOTION=1; setting %d, upscaler %d)\n",
                    int(BbSettings::Get().object_motion.load()), int(BbSettings::Get().upscaler.load()));
        return;
    }
    const auto& features = instance.GetPhysicalDevice().getFeatures();
    if (!features.vertexPipelineStoresAndAtomics) {
        std::printf("Object motion: vertexPipelineStoresAndAtomics unsupported, off\n");
        return;
    }
    void* mapped = nullptr;
    u64 params_address = 0, positions_address = 0;
    if (!CreateBuffers(vk::DeviceSize(1 + FrameSlots * ParamsPerFrame) * 32,
                       vk::DeviceSize(1 + 2 * PositionsPerFrame) * 16, mapped, params_address,
                       positions_address)) {
        std::printf("Object motion: buffer creation failed, off\n");
        return;
    }
    params_mapped = static_cast<u32*>(mapped);
    std::memset(params_mapped, 0, 32);
    Shader::MotionVectors::params_address = params_address;
    Shader::MotionVectors::positions_address = positions_address;
    enabled = true;
    std::printf("Object motion: on (%u vertices per frame)\n", PositionsPerFrame);
}

ObjectMotion::~ObjectMotion() {
    scheduler.Finish();
    const auto allocator = instance.GetAllocator();
    const VkDevice device = instance.GetDevice();
    for (const auto [buffer, allocation, memory] :
         {std::tuple{params_buffer, params_allocation, params_memory},
          std::tuple{positions_buffer, positions_allocation, positions_memory}}) {
        if (memory) {
            VULKAN_HPP_DEFAULT_DISPATCHER.vkDestroyBuffer(device, buffer, nullptr);
            VULKAN_HPP_DEFAULT_DISPATCHER.vkFreeMemory(device, memory, nullptr);
        } else if (buffer) {
            vmaDestroyBuffer(allocator, buffer, allocation);
        }
    }
}

void ObjectMotion::OnFrameStart() {
    if (!enabled) {
        return;
    }
    if (BbStats::enabled && (frame % 600) == 0 && history.stats.draws) {
        const auto& s = history.stats;
        std::printf("Object motion: %llu draws, %llu stored, %llu with history, %llu unmatched, "
                    "%llu capacity skips, %llu invalid, %llu still, %llu jumped (600 frames); last frame %u/%u vertices; "
                    "index ranges %llu reused, %llu scanned (%llu found changed), %zu cached\n",
                    (unsigned long long)s.draws, (unsigned long long)s.stored,
                    (unsigned long long)s.loaded, (unsigned long long)s.unmatched,
                    (unsigned long long)s.exhausted, (unsigned long long)s.invalid,
                    (unsigned long long)s.still, (unsigned long long)s.jumped,
                    history.Used(), PositionsPerFrame, (unsigned long long)index_ranges.stats.hits,
                    (unsigned long long)index_ranges.stats.scans,
                    (unsigned long long)index_ranges.stats.stale, index_ranges.Size());
        index_ranges.stats = {};
        history.stats = {};
    }
    history.NextFrame();
    ++frame;
    {
        static const char* trace = bb_env("BB_MOTION_TRACE");
        tracing = trace && std::remove(trace) == 0;
        if (tracing) {
            std::printf("Motion trace: frame %llu\n", (unsigned long long)frame);
        }
    }
    if (frame % Motion::IndexRangeCache::Unused == 0) {
        index_ranges.Trim(frame);
    }
    const u32 slot = u32(frame % FrameSlots);
    // Do not overwrite host parameters still referenced by queued GPU work.
    if (params_ticks[slot]) {
        scheduler.Wait(params_ticks[slot]);
        params_ticks[slot] = 0;
    }
    params_used = 0;
    for (auto& target : targets) {
        target.written = false;
    }
    scheduler.EndRendering();
    // Last frame's stores must be visible to this frame's loads. Also finish old reads
    // before the ping-pong half they reference is reused for writes.
    scheduler.Record([](vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eVertexShader,
            .srcAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eVertexShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
        };
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &barrier});
    });
}

u32 ObjectMotion::PrepareDraw(const DrawInfo& draw, Motion::Anchor anchor) {
    if (!enabled || BbToggle::Disabled(1u << 29) || params_used >= ParamsPerFrame) {
        return 0;
    }
    const auto allocation = history.Prepare(draw, anchor);
    if (tracing) {
        std::printf("Motion trace:   -> store %u load %u vertices %u+%u\n", allocation.store,
                    allocation.load, allocation.first_vertex, allocation.vertices);
    }
    u32 flags = (allocation.store ? Shader::MotionVectors::FlagStore : 0) |
                (allocation.load ? Shader::MotionVectors::FlagLoad : 0);
    if (!flags) {
        return 0;
    }
    const u32 slot = u32(frame % FrameSlots);
    const u32 index = 1 + slot * ParamsPerFrame + params_used++;
    u32* param = params_mapped + index * 8;
    param[0] = allocation.store;
    param[1] = allocation.load;
    param[2] = allocation.vertices;
    param[3] = flags;
    param[4] = allocation.first_vertex;
    param[5] = allocation.first_instance;
    param[6] = allocation.instances;
    param[7] = 0;
    params_ticks[slot] = scheduler.CurrentTick();
    return index;
}

ObjectMotion::Target& ObjectMotion::GetTarget(u32 width, u32 height) {
    for (auto& target : targets) {
        if (target.image && target.width == width && target.height == height) {
            target.used_frame = frame;
            return target;
        }
    }
    // A free slot, else the image used longest ago (not the one being read).
    Target* slot = nullptr;
    for (auto& target : targets) {
        if (!target.image) {
            slot = &target;
            break;
        }
        if (&target != read_target && (!slot || target.used_frame < slot->used_frame)) {
            slot = &target;
        }
    }
    if (slot->image) {
        // Destroyed once the GPU is past the work recorded so far (it may still use it).
        scheduler.DeferOperation([image = std::move(slot->image),
                                  view = std::move(slot->view)]() mutable {
            view.reset();
            image.Destroy();
        });
    }
    static u32 reports = 0;
    if (BbStats::enabled && reports < 16) {
        ++reports;
        std::printf("Object motion: motion image %ux%u%s\n", width, height,
                    slot->width ? " (replacing one)" : "");
    }
    const auto device = instance.GetDevice();
    slot->image = VideoCore::HostImage(device, instance.GetAllocator());
    slot->image.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR32G32Uint, // EmitFragmentMotion's packing
        .extent = {width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    slot->view = Check(device.createImageViewUnique({
        .image = vk::Image(slot->image),
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR32G32Uint, // EmitFragmentMotion's packing
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    }));
    slot->width = width;
    slot->height = height;
    slot->written = false;
    slot->layout = vk::ImageLayout::eUndefined;
    slot->used_frame = frame;
    return *slot;
}

void ObjectMotion::Attach(RenderState& state, u32 width, u32 height) {
    constexpr u32 slot = Shader::MotionVectors::Output;
    Target& target = GetTarget(width, height);
    auto& attachment = state.color_attachments[slot];
    attachment = {};
    attachment.image_view = *target.view;
    attachment.image_layout = vk::ImageLayout::eColorAttachmentOptimal;
    if (!target.written || target.layout != vk::ImageLayout::eColorAttachmentOptimal) {
        // Clear once per frame; later reads must not erase the validity of this history.
        scheduler.EndRendering();
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            .dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
                             vk::AccessFlagBits2::eColorAttachmentRead,
            .oldLayout = target.layout,
            .newLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .image = vk::Image(target.image),
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        // Recorded in order on the recording thread: CommandBuffer() here switched the rest of
        // the submission (most of the G-buffer) to direct recording on the GPU thread.
        scheduler.Record([barrier](vk::CommandBuffer cmd) {
            cmd.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
        });
        target.layout = vk::ImageLayout::eColorAttachmentOptimal;
        attachment.is_clear = !target.written;
        target.written = true;
    }
    state.num_color_attachments = std::max<u16>(state.num_color_attachments, slot + 1);
}

vk::ImageView ObjectMotion::PrepareRead(u32 width, u32 height, bool& valid) {
    // The image of this size; otherwise any image (the pass binds one, its vectors unused).
    Target* target = const_cast<Target*>(Find(width, height));
    if (!target) {
        for (auto& candidate : targets) {
            if (candidate.image && (!target || candidate.used_frame > target->used_frame)) {
                target = &candidate;
            }
        }
    }
    if (!target) {
        target = &GetTarget(1, 1);
    }
    valid = target->written && width == target->width && height == target->height;
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = target->layout,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = vk::Image(target->image),
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    };
    scheduler.Record([barrier](vk::CommandBuffer cmdbuf) {
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    });
    target->layout = vk::ImageLayout::eGeneral;
    read_target = target;
    return *target->view;
}

} // namespace Vulkan
