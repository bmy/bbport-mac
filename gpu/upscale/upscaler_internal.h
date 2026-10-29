// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what the parts of the temporal upscaler share (vk_temporal_upscaler.cpp and the
// upscaler_*.cpp files split from it): the debug dump helpers of upscaler_debug.cpp. Only those
// files include it.
#pragma once

#include <initializer_list>
#include "ffx_vk_portable.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"

namespace Vulkan::UpscalerDebug {

FfxVkPortableImage Describe(vk::Image image, vk::Format format, u32 width, u32 height,
                            vk::ImageUsageFlags usage, vk::ImageAspectFlags aspect,
                            FfxVkPortableResourceState state);

/// bbport: BB_DUMP_TRIGGER=<file> BB_DUMP_DIR=<dir> (default out/dump): creating the file dumps
/// the upscaler's images of the next BB_DUMP_FRAMES (8) frames as raw files
/// <dir>/fNNN_<name>_<w>x<h>_<format>.raw, for checking temporal stability offline.
/// Returns the frame number to dump, or -1.
int DumpFrame();

struct DumpImage {
    vk::Image image; ///< in layout General
    u32 width, height, bytes_per_pixel;
    const char* name;
    const char* format;
    vk::ImageAspectFlagBits aspect = vk::ImageAspectFlagBits::eColor;
};

/// Copies `images` into host buffers after the commands recorded so far and writes them to
/// files once the GPU is done.
void DumpImages(const Instance& instance, Scheduler& scheduler, vk::CommandBuffer cmdbuf,
                int frame, std::initializer_list<DumpImage> images);

void PrintIssues(const char* what, u64 issues);

} // namespace Vulkan::UpscalerDebug

namespace Vulkan {
using UpscalerDebug::Describe;
using UpscalerDebug::DumpFrame;
using UpscalerDebug::DumpImage;
using UpscalerDebug::DumpImages;
using UpscalerDebug::PrintIssues;
} // namespace Vulkan
