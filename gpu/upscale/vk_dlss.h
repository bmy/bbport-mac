// SPDX-FileCopyrightText: Copyright 2026 IFreemz, bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
//
// bbport: NVIDIA DLSS Super Resolution and DLSS Frame Generation through the optional bridge
// (gpu/dlss_bridge, MIT; libbbport_dlss.so / bbport_dlss.dll), loaded at run time from next to
// bb-probe together with NVIDIA's libnvidia-ngx-dlss.so.<version> / nvngx_dlss.dll (Super
// Resolution) and libnvidia-ngx-dlssg.so.<version> / nvngx_dlssg.dll (Frame Generation, GeForce
// RTX 40 and newer). The port itself contains no NVIDIA code; without the libraries, on other
// GPUs or with BB_DLSS=0 nothing changes. Adapted from IFreemz/shadPS4-Bloodborne-DLSS-FSR
// (vk_dlss_ngx).

#pragma once

#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Dlss {
public:
    /// The process-wide instance, or null when the DLLs are absent or BB_DLSS=0.
    static Dlss* Get();
    ~Dlss();

    /// Instance and device creation: adds what NGX needs (or disables DLSS when it is missing).
    void AppendInstanceExtensions(std::vector<const char*>& enabled);
    void AppendDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical,
                                std::vector<const char*>& enabled);
    /// After the device exists.
    void Initialize(vk::Instance instance, vk::PhysicalDevice physical, vk::Device device);
    /// The device could not be created with NGX's extensions (the driver advertises them, but
    /// what they load is missing: libcuda.so.1, the UVM module): DLSS off, FSR 3.1 instead.
    void Disable(std::string_view reason);
    void Shutdown();

    [[nodiscard]] bool Available() const;
    /// Why DLSS cannot run on this system (empty when it can, or before device creation).
    [[nodiscard]] std::string Problem() const;

    struct Resource {
        vk::Image image;
        vk::ImageView view;
        vk::Format format;
        vk::ImageAspectFlags aspect;
        u32 width, height;
    };
    struct FeatureDesc {
        u32 input_width, input_height, output_width, output_height;
        int quality;
        bool hdr;
        bool operator==(const FeatureDesc&) const = default;
    };
    struct Frame {
        Resource color, depth, motion, output;
        float jitter_x, jitter_y;
        bool reset;
        float frame_ms;
        float sharpness;
    };

    /// DLSS quality mode for an output/render size ratio (0 DLAA .. 4 Ultra Performance).
    static int QualityForScale(float scale);
    [[nodiscard]] bool HasFeature(const FeatureDesc& desc) const;
    /// Records the feature creation. The GPU must be done with the previous one.
    bool CreateFeature(vk::CommandBuffer command, const FeatureDesc& desc);
    bool Evaluate(vk::CommandBuffer command, const Frame& frame);
    void ReleaseFeature();

    // DLSS Frame Generation (upscale/vk_frame_generation): frames between two finished frames.

    /// Frames it can generate per pair of rendered frames: 0 none, 1 (2x), 3 (up to 4x, GeForce
    /// RTX 50). Known after Initialize.
    [[nodiscard]] int FrameGenMaxCount() const;
    /// Why Frame Generation cannot run (empty when it can, or before device creation).
    [[nodiscard]] std::string FrameGenProblem() const;
    struct FrameGenDesc {
        u32 width, height; ///< the finished frame and the generated ones
        vk::Format format;
        u32 render_width, render_height; ///< depth and motion vectors
        bool operator==(const FrameGenDesc&) const = default;
    };
    struct FrameGenFrame {
        Resource backbuffer, depth, motion, output; ///< General layout
        Resource hudless; ///< the frame before its UI, General; image null when there is none
        float jitter_x, jitter_y;
        float camera_near, camera_far, camera_fov;
        std::array<float, 3> camera_position, camera_up, camera_right, camera_forward;
        bool depth_inverted;
        /// Row-vector clip-space matrices (camera_matrices.h); without them a still camera is
        /// assumed.
        bool has_matrices;
        std::array<float, 16> view_to_clip, clip_to_prev_clip, prev_clip_to_clip;
        bool reset;
        u32 count, index; ///< generated frames for this pair, and which one (1-based)
    };
    [[nodiscard]] bool HasFrameGen(const FrameGenDesc& desc) const;
    /// Records the feature creation. The GPU must be done with the previous one.
    bool CreateFrameGen(vk::CommandBuffer command, const FrameGenDesc& desc);
    bool EvaluateFrameGen(vk::CommandBuffer command, const FrameGenFrame& frame);
    void ReleaseFrameGen();

private:
    Dlss();
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
