// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: AMD's own FSR DLLs for Vulkan (FSR SDK 2.4, upscaler fsr_amd): the player's
// amd_fidelityfx_loader_vk.dll and amd_fidelityfx_upscaler_vk.dll (BB_FFX_DIR, <user>/fsr or
// beside bb-probe). The loader picks the provider itself, also one from AMD's driver
// (amdvkffx64.dll); BB_FFX_VERSION=<part of its name> picks another one. Windows only: the DLLs
// are not part of the package. Same inputs as the other FSR 4 paths (Fsr4Upscaler::Frame).

#pragma once

#include <memory>
#include <string>
#include "common/types.h"
#include "upscale/vk_fsr4.h"

namespace Vulkan {

class Instance;
class Scheduler;

class FsrAmd {
public:
    FsrAmd(const Instance& instance, Scheduler& scheduler);
    ~FsrAmd();

    /// The DLLs were found and loaded.
    [[nodiscard]] bool Available() const noexcept;

    /// Whether the context for this output is ready; if not, it is created on a background
    /// thread (pipelines take a while) and this answers false until it is.
    bool Prepare(u32 output_width, u32 output_height, bool auto_exposure);

    /// Records the upscale into `frame.cmdbuf` (output stays in General). On failure `problem`
    /// says why and `fatal` is set for problems that last the session.
    bool Record(const Fsr4Upscaler::Frame& frame, std::string& problem, bool& fatal);

    /// Drops the context (a retry after a menu change); one being created is kept.
    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

/// AMD's frame generation DLL (amd_fidelityfx_framegeneration_vk.dll beside the loader: FSR 4.0.2
/// ML frame generation on RDNA4, else FSR 3.1), frame_generation=fsr_amd. Without AMD's swapchain:
/// one frame generated into the port's own target, shown by its presenter. BB_FFX_FG_ML=1 makes
/// the ML provider accept any GPU, =0 refuse it (the DLL's choice otherwise).
class FgAmd {
public:
    FgAmd(const Instance& instance, Scheduler& scheduler);
    ~FgAmd();

    [[nodiscard]] bool Available() const noexcept;
    /// A context exists (Ensure succeeded and nothing destroyed it since).
    [[nodiscard]] bool Ready() const noexcept;

    /// The context for these sizes and colour format (made now when they changed: `created`).
    bool Ensure(u32 render_width, u32 render_height, u32 width, u32 height, vk::Format format,
                bool& created, std::string& problem);
    void Destroy();

    struct Frame {
        vk::CommandBuffer cmdbuf;
        Fsr4Upscaler::Image color;   ///< the frame with its UI, General
        Fsr4Upscaler::Image hudless; ///< without its UI (image null: none)
        Fsr4Upscaler::Image depth, motion;
        Fsr4Upscaler::Image output;  ///< the generated frame, General
        float jitter[2];
        float frame_ms;
        float near_plane, far_plane, vertical_fov;
        float camera_position[3], camera_up[3], camera_right[3], camera_forward[3];
        bool reset;
        u64 frame_id;
    };
    /// Configure, prepare and generate into `frame.cmdbuf`; on failure `problem` says why.
    bool Record(const Frame& frame, std::string& problem);

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace Vulkan
