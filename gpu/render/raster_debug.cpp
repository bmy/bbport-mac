// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer diagnostics: traced shader inputs, the images of a pass, and the
// watched draws (BB_WATCH_*). Split out of rasterizer.cpp (2026-10-26).


#include <mutex>
#include "shim/bb_env.h"
#include <set>
#include <string>
#include <filesystem>
#include "render/raster_internal.h"
#include <vk_mem_alloc.h>


namespace Vulkan {

namespace {
/// BB_PASS_DUMP=addr,addr (hex guest addresses of render targets) with BB_PASS_DUMP_TRIGGER=<file>
/// (diagnostics, the flicker): while the file exists, every 8th row of those images is saved at
/// each display pass (BB_DUMP_DIR/pNNNNN_<addr>_<w>x<h>_<bits>.raw), little enough not to change
/// the frame's timing much.
std::vector<VAddr> PassDumpAddresses() {
    std::vector<VAddr> out;
    if (const char* env = bb_env("BB_PASS_DUMP")) {
        for (const char* at = env; *at;) {
            char* end = nullptr;
            out.push_back(std::strtoull(at, &end, 16));
            at = *end ? end + 1 : end;
        }
    }
    return out;
}
} // namespace

void Rasterizer::DumpTracedInputs() {
    if (traced_inputs.empty()) {
        return;
    }
    static const std::string dir = [] {
        const char* env = bb_env("BB_DUMP_DIR");
        return std::string{env && env[0] ? env : "out/dump"};
    }();
    VkDeviceSize total = 0;
    for (const auto& r : traced_inputs) total += r.size;
    const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                       .size = total,
                                       .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                        &info) != VK_SUCCESS) {
        traced_inputs.clear();
        return;
    }
    scheduler.EndRendering();
    auto ranges = traced_inputs;
    scheduler.Record([ranges, buffer](vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 before{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &before});
        VkDeviceSize at = 0;
        for (const auto& r : ranges) {
            cmd.copyBuffer(r.buffer, buffer, vk::BufferCopy{r.offset, at, r.size});
            at += r.size;
        }
    });
    std::string header;
    for (const auto& r : traced_inputs) {
        header += fmt::format("{:x}+{:x}:{} ", (u64)(VkBuffer)r.buffer, r.offset, r.size);
    }
    char path[512];
    std::snprintf(path, sizeof(path), "%s/g%06llu.bin", dir.c_str(),
                  (unsigned long long)traced_inputs_frame);
    scheduler.DeferPriorityOperation([allocator = instance.GetAllocator(), buffer, allocation, info,
                                      total, file = std::string{path}, header] {
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        if (FILE* f = std::fopen(file.c_str(), "wb")) {
            const u32 n = u32(header.size());
            std::fwrite(&n, 4, 1, f);
            std::fwrite(header.data(), 1, n, f);
            std::fwrite(info.pMappedData, 1, total, f);
            std::fclose(f);
        }
        vmaDestroyBuffer(allocator, buffer, allocation);
    });
    traced_inputs.clear();
}

void Rasterizer::DumpPassImages(char tag) {
    static const std::vector<VAddr> addresses = PassDumpAddresses();
    static const char* trigger = bb_env("BB_PASS_DUMP_TRIGGER");
    if (addresses.empty() || !trigger || !std::filesystem::exists(trigger)) {
        return;
    }
    static const std::string dir = [] {
        const char* env = bb_env("BB_DUMP_DIR");
        return std::string{env && env[0] ? env : "out/dump"};
    }();
    static u32 indices[2] = {};
    const u32 n = indices[tag == 'p' ? 0 : 1]++;
    std::printf("Pass dump %c %u: GPU frame %llu\n", tag, n,
                (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed));
    scheduler.EndRendering();
    for (const VAddr address : addresses) {
        VideoCore::Image* found = nullptr;
        texture_cache.ForEachImageInRegion(address, 1, [&](VideoCore::ImageId, VideoCore::Image& image) {
            if (image.info.guest_address == address &&
                (!found || image.info.size.width > found->info.size.width)) {
                found = &image;
            }
        });
        if (!found || found->info.num_bits == 0) {
            continue;
        }
        const u32 width = found->info.size.width, height = found->info.size.height;
        const u32 bytes = found->info.num_bits / 8;
        static const u32 step = [] {
            const char* env = bb_env("BB_PASS_DUMP_STEP");
            return env ? std::max(1, std::atoi(env)) : 8;
        }();
        const u32 rows = (height + step - 1) / step;
        const VkDeviceSize size = VkDeviceSize(width) * rows * bytes;
        const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                           .size = size,
                                           .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        const VmaAllocationCreateInfo alloc_ci{
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
        VkBuffer buffer{};
        VmaAllocation allocation{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                            &info) != VK_SUCCESS) {
            continue;
        }
        runtime.Transit(found, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
        const vk::Image image = found->GetImage();
        const vk::ImageAspectFlags aspect = found->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
        scheduler.Record([=](vk::CommandBuffer cmd) {
            std::vector<vk::BufferImageCopy> regions;
            regions.reserve(rows);
            for (u32 r = 0; r < rows; ++r) {
                regions.push_back({
                    .bufferOffset = VkDeviceSize(r) * width * bytes,
                    .imageSubresource = {aspect, 0, 0, 1},
                    .imageOffset = {0, s32(r * step), 0},
                    .imageExtent = {width, 1, 1},
                });
            }
            cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, regions);
        });
        char path[512];
        std::snprintf(path, sizeof(path), "%s/%c%05u_%llx_%ux%u_%u.raw", dir.c_str(), tag, n,
                      (unsigned long long)address, width, rows, found->info.num_bits);
        scheduler.DeferPriorityOperation([allocator = instance.GetAllocator(), buffer, allocation,
                                          info, size, file = std::string{path}] {
            vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
            if (FILE* f = std::fopen(file.c_str(), "wb")) {
                std::fwrite(info.pMappedData, 1, size, f);
                std::fclose(f);
            }
            vmaDestroyBuffer(allocator, buffer, allocation);
        });
    }
}

void Rasterizer::WatchDraw(const GraphicsPipeline* pipeline, const RenderState& state) {
    // GPU thread only. Per (vs, ps, first target): the last fields seen and how many draws in a
    // row had them. A change after a stable run is printed with the frame number, and so is the
    // change back, which marks a one-frame anomaly (docs/Flicker_video_2026-10-17.ru.md).
    const auto& regs = Regs();
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    const VAddr target = cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.guest_address : 0;
    const u64 key = vs.pgm_hash * 0x9E3779B97F4A7C15ull ^ (ps ? ps->pgm_hash : 0) ^ target;
    const auto bits = [](float f) { return u64(std::bit_cast<u32>(f)); };
    std::vector<u64> fields;
    const auto& vp = regs.viewports[0];
    fields.push_back(bits(vp.xscale) << 32 | bits(vp.xoffset));
    fields.push_back(bits(vp.yscale) << 32 | bits(vp.yoffset));
    fields.push_back(u64(std::bit_cast<u32>(regs.viewport_control)) << 32 | u64(regs.IsClipDisabled()));
    const auto& sc = regs.screen_scissor;
    fields.push_back(u64(u16(sc.top_left_x)) << 48 | u64(u16(sc.top_left_y)) << 32 |
                     u64(u16(sc.bottom_right_x)) << 16 | u16(sc.bottom_right_y));
    fields.push_back(bits(target_scale[0]) << 32 | bits(target_scale[1]));
    fields.push_back(bits(push_data.xoffset) << 32 | bits(push_data.yoffset));
    fields.push_back(u64(state.width) << 32 | state.height);
    static const bool textures = bb_env("BB_DRAW_WATCH_TEXTURES") != nullptr;
    for (const auto id : textures ? bound_images : decltype(bound_images){}) {
        const auto& info = texture_cache.GetImage(id).info;
        fields.push_back(info.guest_address);
        fields.push_back(u64(info.size.width) << 32 | info.size.height);
    }
    struct Seen {
        std::vector<u64> fields;
        std::vector<u64> stable_fields;
        u32 run = 0;
        u64 changed_frame = 0;
    };
    static std::unordered_map<u64, Seen> seen;
    static u32 printed = 0;
    auto& s = seen[key];
    const u64 frame = FrameCapture::Flips();
    if (s.fields == fields) {
        ++s.run;
        return;
    }
    const auto print = [&](const char* what) {
        if (printed++ > 4000) {
            return;
        }
        std::string text = fmt::format("Draw watch: frame {} {} vs {:016x} ps {:016x} target {:#x}:",
                                       frame, what, vs.pgm_hash, ps ? ps->pgm_hash : 0, target);
        const size_t n = std::max(fields.size(), s.fields.size());
        for (size_t i = 0; i < n; ++i) {
            const u64 now = i < fields.size() ? fields[i] : 0;
            const u64 was = i < s.fields.size() ? s.fields[i] : 0;
            if (now != was) {
                text += fmt::format(" [{}] {:#x}->{:#x}", i, was, now);
            }
        }
        std::printf("%s\n", text.c_str());
    };
    if (s.run >= 8) {
        // A stable run broke.
        s.stable_fields = s.fields;
        s.changed_frame = frame;
        print("changed");
    } else if (!s.stable_fields.empty() && fields == s.stable_fields && frame - s.changed_frame <= 2) {
        print("back after a short change");
        s.stable_fields.clear();
    }
    s.fields = std::move(fields);
    s.run = 0;
}

} // namespace Vulkan
