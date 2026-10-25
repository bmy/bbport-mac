// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal command path's command layer (docs/MACOS_METAL_RENDERER.md, stage 3). It sits
// in vulkan.hpp's default dispatcher, under the renderer: the renderer's command recording and
// submission go through it. After Wooloomooloo2/bloodborne_mac_port's MetalLayer (78b59f5,
// renderer_metal/mtl_vk_layer.mm, GPL-2.0-or-later).
//
// Stage 3a, BB_METAL_TRACE=1: the layer forwards every call unchanged and counts them; every 5 s
// the log has the calls per second, command buffers and submissions per second, and, once each,
// the commands and submissions libraries looked up for themselves (vkGet*ProcAddr: they get the
// layer's entry for what it covers, so those calls are counted too). Libraries that call the
// loader's functions directly (FSR 4.1.1, FSR 4 v07) are not seen. One device: a second
// dispatcher init would need the layer installed again. Plain C++: any platform.
//
// Stage 3b, BB_METAL_FRAME=1 (the Metal command path's build only, validation layers off): the
// layer owns every primary command buffer begun through it and hands its commands to the frame
// encoder (mtl_frame.h), which encodes what it can in Metal and leaves the rest to KosmicKrisp;
// submissions with owned command buffers go through the encoder. Also builds every pipeline in
// Metal (the shadow build). Neither set: nothing is installed.

#pragma once

#include "vulkan/vk_common.h"

namespace MetalLayer {

/// Installs the layer over VULKAN_HPP_DEFAULT_DISPATCHER's command functions if asked
/// (BB_METAL_TRACE=1 or BB_METAL_FRAME=1). Call once, right after the dispatcher's init with the
/// device.
void InstallIfAsked(VkDevice device);

/// Frame mode is on: libraries that record with the loader's own functions (FSR 4.1.1, FSR 4
/// v07) cannot be used (their commands would miss the owned command buffers).
bool FrameActive();

} // namespace MetalLayer
