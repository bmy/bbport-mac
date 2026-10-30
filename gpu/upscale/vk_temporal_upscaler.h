// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: temporal upscaling of Bloodborne's HDR scene color (docs/upscaler.md). BB_UPSCALER=fsr3
// runs FSR 3.1 (FireBurn/FSR-Vulkan, native Vulkan) on the scene color right before the
// post-processing combine pass, with the scene depth and camera motion vectors, and writes the
// result back so the game's own post, tonemap and UI continue unchanged (Native AA preset).
//
// Guest targets stay at 1920x1080. Host scene targets use output/preset dimensions,
// resized at frame boundaries. At 1080p HDR reconstruction precedes post; other outputs
// reconstruct tonemapped scene proxies before output-size UI/display composition.
// BB_RENDER_RES retains the older startup-patched guest-resolution path.

#pragma once

#include <array>
#include "shim/bb_env.h"
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/types.h"
#include "vulkan/vk_common.h"
#include "upscale/vk_dlss.h"
#include "upscale/vk_fsr4.h"
#include "upscale/vk_metalfx.h"
#include "textures/image.h"

struct FfxVkPortableUpscaleContext;

namespace VideoCore {
class TextureCache;
}

namespace Vulkan {

class FrameGeneration;

class Instance;
class Scheduler;
class Runtime;
class CameraMotion;
class SceneTargets;

/// bbport BB_FINAL_DUMP_TRIGGER: the frame as presented (after FSR and post processing) is saved
/// as final_<w>x<h> in BB_DUMP_DIR when the trigger file exists (consumed). `image` in General.
/// An image's level 0 as a raw file (block-compressed: 1 byte per texel), after the commands so far.
void DumpRaw(const Instance& instance, Scheduler& scheduler, vk::CommandBuffer cmdbuf,
             vk::Image image, u32 width, u32 height, u32 bytes_per_texel, int index, const char* name,
             const char* format);
/// One presented image as a raw file (<dir>/f<index>_<name>_...raw), after the commands so far.
void DumpPresented(const Instance& instance, Scheduler& scheduler, vk::CommandBuffer cmdbuf,
                   vk::Image image, u32 width, u32 height, vk::Format format, int index,
                   const char* name);
void DumpFinalFrameIfDue(const Instance& instance, Scheduler& scheduler, vk::CommandBuffer cmdbuf,
                         vk::Image image, u32 width, u32 height, vk::Format format);

class TemporalUpscaler {
public:
    TemporalUpscaler(const Instance& instance, Scheduler& scheduler,
                     VideoCore::TextureCache& texture_cache, Runtime& runtime,
                     CameraMotion& camera_motion, SceneTargets& scene_targets);
    ~TemporalUpscaler();

    [[nodiscard]] bool Enabled() const noexcept {
        return enabled;
    }
    /// A temporal upscaler accumulates this frame (Active), for passes whose noise it resolves
    /// (VBAO).
    [[nodiscard]] bool Accumulating() const {
        return Active();
    }
    /// Called after a scaled upscale with the output image (RGBA8, General layout, before the
    /// UI draws over it) and its size: debug views of the frame (Vbao's AO view).
    std::function<void(vk::Image, u32, u32)> on_upscaled;

    /// A draw into a full-size RGBA16F target with the scene depth: the scene color.
    void OnSceneColor(VideoCore::ImageId color);

    /// A blended (transparent) draw into the scene color that is not a full-screen pass: the
    /// first one of a frame snapshots the opaque scene for the reactive mask.
    void OnBlendedSceneDraw();

    /// A full-screen pass into the scene color: after a snapshot, the reactive mask is taken
    /// before it (the fog composite rewrites every pixel).
    void OnSceneComposite();

    /// Before a compute dispatch: the post-processing combine shader triggers the upscale.
    void OnDispatch(u64 cs_hash);

    /// Start of a frame in the command stream (display pass).
    bool OnFrameStart();
    bool RasterScaling() const;
    /// bbport: LOD bias for scene materials, log2(render / output) while a reduced scene is
    /// upscaled (0 at native size, for TAA and without an upscaler).
    [[nodiscard]] float SceneMipBias() const;
    /// bbport: the state RedirectColor/RedirectDepth/RasterScaling depend on, for memoizing a
    /// draw's render state (Rasterizer::BeginRendering).
    [[nodiscard]] u64 RedirectState() const noexcept {
        return u64(ui_phase) | u64(display_redirect) << 1 | u64(done_this_frame) << 2 |
               u64(ui_color.index) << 8 | u64(ui_depth.index) << 36;
    }


    /// This frame's sub-pixel jitter in pixels (screen x right, y down); zero when off and after
    /// the upscale (post and UI are not jittered).
    [[nodiscard]] std::array<float, 2> Jitter() const noexcept {
        return done_this_frame ? std::array<float, 2>{} : jitter;
    }

    // Scaled presets.

    /// After identifying this draw's targets, before drawing any UI pixels. Menus use
    /// spatial background copy and native UI, without needing scene depth or FSR history.
    void OnDraw(u64 vs_hash, VideoCore::ImageId color, VideoCore::ImageId depth,
                bool native_viewport);

    /// A pass's first color target: the last render-size RGBA8 target before the UI is the
    /// game's finished frame.
    void OnColorTarget(VideoCore::ImageId color);

    struct Target {
        vk::ImageView view;
        vk::ImageLayout layout;
        u32 width, height;
        bool native_ui = false;
    };
    /// Render target redirection (UI passes, display pass): false when not redirected.
    /// `view` is the game's view of the image: redirected views mirror its format (sRGB) and,
    /// for sampling, its channel swizzle.
    bool RedirectColor(VideoCore::ImageId color, const VideoCore::ImageViewInfo& view,
                       Target& target);
    bool RedirectDepth(VideoCore::ImageId depth, Target& target);
    /// Whether `address` is the UI's colour target (the one RedirectColor redirects).
    [[nodiscard]] bool IsUiColorAddress(VAddr address) const;
    /// A sampled image replaced by the upscaled frame (the display pass).
    bool RedirectSampled(VideoCore::ImageId image, const VideoCore::ImageViewInfo& info,
                         vk::ImageView& view, vk::ImageLayout& layout);
    /// Whether RedirectSampled would replace `image` (it may record a barrier then).
    [[nodiscard]] bool RedirectsSampled(VideoCore::ImageId image) const {
        return display_redirect && image == ui_color;
    }

    /// Presenter: the output-size display buffer standing in for the guest one at `address`.
    struct Display {
        vk::Image image;
        vk::Format format;
        u32 width, height;
    };
    bool DisplayOverride(VAddr address, Display& display);
    /// The game's display pass of `address` begins (recording thread): frame generation
    /// interpolates this frame now, its scene with the UI complete.
    void OnDisplayPassStart(VAddr address);
    /// Frame generation for the display pass started last, recorded after that pass's draw
    /// (Rasterizer::RecordPendingFrameGeneration). False: nothing pending.
    bool HasPendingInterpolation() const {
        return pending_interpolation;
    }
    void RunPendingInterpolation();
    bool FrameGenerationWanted() const;
    /// The frame's UI is being drawn (after the scaled scene, before the display pass).
    bool InUiPhase() const {
        return ui_phase;
    }
    /// Frame generation (null until made); the presenter takes its outputs.
    FrameGeneration* GetFrameGeneration() const {
        return frame_generation.get();
    }
    /// BB_PRESENT_DUMP_TRIGGER: whether this presented frame is to be saved (consumes the trigger).
    bool PresentDumpDue();
    /// Saves `image` (General layout, 4 bytes a pixel) as present_<w>x<h> in BB_DUMP_DIR.
    void DumpPresented(vk::Image image, u32 width, u32 height, vk::Format format);

private:
    /// bbport: views of guest images the upscaler reads, kept across frames: FSR 4 registers
    /// images by view in a registry of eight (a new view per frame filled it at Native AA).
    vk::ImageView CachedView(const VideoCore::Image& image, vk::Format format,
                             vk::ImageAspectFlags aspect);
    struct ViewEntry {
        vk::Image image;
        u64 uid = 0;
        vk::Format format{};
        vk::ImageAspectFlags aspect{};
        vk::ImageView view;
        u64 last_use = 0;
    };
    std::array<ViewEntry, 6> view_cache{};
    u64 view_uses = 0;

    void Run();
    void RunScaled();
    void RunUiOnly(VideoCore::ImageId color, VideoCore::ImageId depth);
    /// SMAA wanted on this frame's scene: anti_aliasing=smaa, no upscaler, a 3D scene.
    [[nodiscard]] bool SmaaWanted() const;
    /// SMAA of the finished scene `color` (RGBA8; `proxy`: of its proxy, at render_width x
    /// render_height): the result's image (GENERAL), or null.
    vk::Image RunSmaa(VideoCore::ImageId color, bool proxy = false);
    void EnsureUiResources(u32 width, u32 height, vk::Format color, vk::Format depth);
    void PrepareUiDepth(VideoCore::ImageId depth);
    /// Render size below the scaled-preset output size (the resolution patch is on).
    [[nodiscard]] bool Scaled() const;
    /// A target of the patched render size: the game allocates it with aligned dimensions
    /// (a 1916x1078 scene in 1916x1080 targets).
    [[nodiscard]] bool RenderTarget(u32 w, u32 h) const {
        if (!scaled_session) return w == 1920 && h == 1080;
        return w >= native_width && h >= native_height && w < native_width + 8 &&
               h < native_height + 8;
    }
    /// The scene's size inside a `w` x `h` render target: the game's viewport.
    [[nodiscard]] std::array<u32, 2> SceneSize(u32 w, u32 h) const;
    // The scene color is drawn into a reduced SceneTargets proxy (live presets).
    [[nodiscard]] bool ReducedScene(const VideoCore::Image& color) const;
    /// Available and switched on (menu setting, toggle 1 << 24).
    [[nodiscard]] bool Active() const;
    [[nodiscard]] bool ReactiveOn() const;
    bool EnsureResources(u32 width, u32 height, u32 out_width, u32 out_height, bool hdr);
    void CreatePipelines();
    /// Records the reactive mask pass; false when there is no snapshot this frame.
    bool RecordReactive(vk::ImageView color_view);
    /// FSR 4 is selected, possible in this session (not BB_RENDER_RES) and has not failed.
    [[nodiscard]] bool UseFsr4() const;
    /// bbport: DLSS selected and the bridge is ready (NVIDIA RTX, gpu/dlss_bridge).
    [[nodiscard]] bool UseDlss() const;
    /// Records DLSS into `cmdbuf` (output in General). `hdr`: linear scene color input.
    bool RecordDlss(vk::CommandBuffer cmdbuf, const Dlss::Resource& color,
                    const Dlss::Resource& depth, u32 w, u32 h, u32 ow, u32 oh, float frame_ms,
                    bool hdr);
    /// Records FSR 4 into output_image; on a permanent failure FSR 3 takes over.
    bool RecordFsr4(vk::CommandBuffer cmdbuf, Fsr4Upscaler::Image color, Fsr4Upscaler::Image depth,
                    u32 w, u32 h, u32 ow, u32 oh, float frame_ms);
    void RecordTaa(vk::CommandBuffer cmdbuf, vk::ImageView color, vk::ImageView depth);
    /// bbport: MetalFX is selected, available and has not failed (macOS, vk_metalfx).
    [[nodiscard]] bool UseMetalFx() const;
    /// MetalFX into output_image. Ends the submission: callers fetch CommandBuffer() again.
    bool RecordMetalFx(const MetalFxUpscaler::Input& color, const MetalFxUpscaler::Input& depth,
                       u32 w, u32 h, u32 ow, u32 oh, bool hdr);
    /// Sharpness above 1 for FSR 3/4 (their RCAS stops at 1): one more RCAS pass over the target
    /// (output_image, or the 8-bit UI image with ldr) in General layout after the upscaler.
    /// `applied`: the strength the upscaler applied itself (0 for MetalFX, which has no RCAS).
    void ExtraSharpen(vk::Image target, bool ldr, u32 w, u32 h, float applied = 1.0f);

    const Instance& instance;
    Scheduler& scheduler;
    VideoCore::TextureCache& texture_cache;
    Runtime& runtime;
    CameraMotion& camera_motion;
    SceneTargets& scene_targets;
    int applied_preset = -1;
    int applied_upscaler = -1;
    bool dispatched_last_frame = false;
    bool last_active = false, last_jitter = false;


    bool enabled = false;
    bool failed = false;
    /// An FSR 3 dispatch recorded on a recording thread failed; `failed` at the next frame.
    std::atomic<bool> dispatch_failed{false};
    u64 trigger_hash = 0x9a9cf8a9;
    VideoCore::ImageId scene_color{};
    bool done_this_frame = false;
    u32 preset_file_frames = 0; ///< BB_PRESET_FILE polling
    u32 applied_output = 0; ///< BbSettings::PackSize
    u32 startup_output = 0; ///< the output at launch (hybrid: the game's targets were patched for it)
    float applied_aspect = 0.0f; ///< the output's, given to the game's camera
    bool snapshot_taken = false;
    bool opaque_valid = false;
    bool mask_ready = false;
    std::array<float, 2> jitter{};
    u32 jitter_index = 0;
    bool reset = true;
    void NoteReset(u32 why);
    u64 frame_id = 0;
    std::chrono::steady_clock::time_point last_frame{};

    u32 width = 0, height = 0;             ///< render size
    u32 out_width = 0, out_height = 0;     ///< output size of the context
    bool context_hdr = true;
    u32 target_width = 1920, target_height = 1080; ///< output size of scaled presets
    u64 ui_trigger_vs = 0x34e8a281;
    VideoCore::ImageId ldr_target{};
    VideoCore::ImageId ui_color{}, ui_depth{};
    bool ui_phase = false;         ///< from the upscale to the next display pass
    const bool ui_trace_on = bb_env("BB_UI_TRACE") != nullptr; ///< diagnostics
    std::string ui_trace;
    bool display_redirect = false; ///< the display pass of an upscaled frame
    bool ui_read_barrier = false;
    VideoCore::HostImage ui_image;
    VideoCore::HostImage ui_depth_image;
    vk::UniqueImageView ui_view;
    vk::UniqueImageView ui_depth_view;
    bool depth_blit = false;
    bool scaled_session = false;
    /// Live presets over the startup patch (BB_LIVE_HYBRID=1 with BB_RENDER_RES): the game's
    /// targets have the launch preset's size (no proxies, nothing extra a frame); another preset
    /// chosen in the game is drawn through SceneTargets proxies relative to that size.
    bool hybrid = false;
    int startup_preset = -1; ///< the preset the session's patch was made for (hybrid)
    u32 ui_width = 0, ui_height = 0;
    u32 render_width = 0, render_height = 0; ///< the scene's size as drawn (proxies: theirs)
    u32 native_width = 0, native_height = 0; ///< the game's scene size (its targets)
    /// Proxies stand in for the game's scene targets (live presets).
    [[nodiscard]] bool ProxiesOn() const;
    vk::Format ui_format = vk::Format::eUndefined;
    vk::Format ui_depth_format = vk::Format::eUndefined;
    /// Views of the port's images in the formats/swizzles the game's views use.
    struct MirrorView {
        vk::Format format;
        vk::ComponentMapping mapping;
        vk::UniqueImageView view;
    };
    vk::ImageView Mirror(vk::Image image, std::vector<MirrorView>& views, vk::Format format,
                         vk::ComponentMapping mapping);
    std::vector<MirrorView> ui_views;
    struct DisplayImage {
        VideoCore::HostImage image;
        std::vector<MirrorView> views;
        vk::Format format{};
        u32 width = 0, height = 0;
        bool valid = false;
    };
    std::mutex display_mutex;
    int present_dump_remaining = 0, present_dump_index = 0;
    std::unordered_map<VAddr, DisplayImage> displays;
    FfxVkPortableUpscaleContext* context = nullptr;
    bool resources_ready = false; ///< images below match width/height/out size
    bool resources_fsr4 = false;  ///< made for FSR 4 (no FSR 3 context)
    bool resources_taa = false;
    bool resources_spatial = false;
    std::unique_ptr<Fsr4Upscaler> fsr4;
    std::unique_ptr<class FrameGeneration> frame_generation;
    /// anti_aliasing=smaa without an upscaler (RunSmaa); made on first use.
    std::unique_ptr<class Smaa> smaa;
    bool smaa_done = false; ///< this frame's SMAA ran at the game's own size (OnDraw)
    vk::Image pending_interpolation{}; ///< OnDisplayPassStart's frame, for RunPendingInterpolation
    VAddr pending_display = 0;
    bool hudless_taken = false; ///< this frame's UI-less image went to frame generation
    /// Inputs of this frame's scaled run, for frame generation (CaptureFrameGenScene).
    void CaptureFrameGenScene(vk::Image depth_image, vk::Format depth_format, u32 w, u32 h, u32 ow,
                              u32 oh, float frame_ms, bool frame_reset);
    bool fsr4_failed = false;
    std::unique_ptr<MetalFxUpscaler> metalfx;
    /// bbport (macOS): after MetalFX fails at run time (a stall, a failed command buffer) FSR 3.1
    /// stands in and MetalFX is tried again after 30 s, 2 min and 10 min; a fourth failure keeps
    /// FSR 3.1 for the session. 30 min of MetalFX after a retry starts the count over. Choosing an
    /// upscaler in a menu ends it. BB_METALFX_AUTO_RETRY=0: off (FSR 3.1 until chosen again).
    void MetalFxAutoRetry();
    void NoteMetalFxFailure();
    bool metalfx_standing_in = false; ///< the upscaler setting holds FSR 3.1 on MetalFX's behalf
    u32 metalfx_failures = 0;
    std::chrono::steady_clock::time_point metalfx_retry_at{}; ///< epoch: none due
    std::chrono::steady_clock::time_point metalfx_retried{};  ///< MetalFX on since (a retry)
    std::uint32_t metalfx_chosen_seen = 0; ///< BbSettings upscaler_chosen when the stand-in began
    bool resources_metalfx = false; ///< made for MetalFX (no FSR 3 context)
    /// The FSR 4 model for this preset and output is still being built (#213): FSR 3.1 meanwhile.
    bool fsr4_waiting = false;
    bool dlss_failed = false;
    VideoCore::HostImage motion_image;
    VideoCore::HostImage output_image;
    vk::UniqueImageView motion_view;
    vk::UniqueImageView output_view;
    VideoCore::HostImage opaque_image;   ///< scene color before the blended draws
    VideoCore::HostImage reactive_image; ///< R8 reactive mask
    vk::UniqueImageView opaque_view;
    vk::UniqueImageView reactive_view;
    vk::UniqueDescriptorSetLayout reactive_desc_layout;
    vk::UniquePipelineLayout reactive_pipeline_layout;
    vk::UniquePipeline reactive_pipeline;
    vk::UniqueDescriptorSetLayout merge_desc_layout;
    vk::UniquePipelineLayout merge_pipeline_layout;
    vk::UniquePipeline merge_pipeline;
    std::array<VideoCore::HostImage, 2> taa_history;
    std::array<vk::UniqueImageView, 2> taa_history_views;
    u32 taa_next = 0;
    vk::UniqueSampler taa_sampler;
    vk::UniqueDescriptorSetLayout taa_desc_layout;
    vk::UniquePipelineLayout taa_pipeline_layout;
    vk::UniquePipeline taa_pipeline;
    vk::UniqueDescriptorSetLayout taa_sharpen_desc_layout;
    vk::UniquePipelineLayout taa_sharpen_pipeline_layout;
    vk::UniquePipeline taa_sharpen_pipeline;
    vk::UniquePipeline taa_sharpen_ldr_pipeline;
    vk::UniquePipeline taa_sharpen_half_pipeline; // bbport (macOS): extra sharpening, RGBA16F copy
    // FSR 1 (upscaler fsr1): EASU, then the RCAS pass above (ExtraSharpen).
    vk::UniqueSampler easu_sampler;
    vk::UniqueDescriptorSetLayout easu_desc_layout;
    vk::UniquePipelineLayout easu_pipeline_layout;
    vk::UniquePipeline easu_pipeline;
    void RecordEasu(vk::CommandBuffer cmdbuf, vk::ImageView color, u32 w, u32 h, u32 iw, u32 ih,
                    u32 ow, u32 oh);
    // bbport: FSR 4 in linear light on the scaled presets (fsr4_color.comp): the decoded input
    // and the passes that decode it and encode the output again.
    VideoCore::HostImage fsr4_linear_image;
    bool fsr4_linear_frame = false; ///< this frame's FSR 4 input was decoded (and the output encoded)
    vk::UniqueImageView fsr4_linear_view;
    vk::UniqueDescriptorSetLayout fsr4_decode_desc_layout;
    vk::UniquePipelineLayout fsr4_decode_pipeline_layout;
    vk::UniquePipeline fsr4_decode_pipeline;
    vk::UniqueDescriptorSetLayout fsr4_encode_desc_layout;
    vk::UniquePipelineLayout fsr4_encode_pipeline_layout;
    vk::UniquePipeline fsr4_encode_pipeline;
    vk::UniqueSampler fsr4_linear_sampler;
    vk::UniqueDescriptorSetLayout fsr4_reactive_desc_layout;
    vk::UniquePipelineLayout fsr4_reactive_pipeline_layout;
    vk::UniquePipeline fsr4_reactive_pipeline;
    /// bbport: FSR 4 takes no reactive mask: its output is blended with `color` (the frame it
    /// upscaled, render size, General layout, same colour space as the output) where the mask
    /// marks blended effects (fsr4_reactive.comp). Recorded after FSR 4 into `cmdbuf`.
    void RecordFsr4Reactive(vk::CommandBuffer cmdbuf, vk::ImageView color, u32 w, u32 h, u32 ow,
                            u32 oh);
    // ExtraSharpen: a copy of the upscaled frame (RCAS reads neighbours) and the target views.
    VideoCore::HostImage extra_sharpen_image;
    vk::UniqueImageView extra_sharpen_view;
    u32 extra_sharpen_width = 0, extra_sharpen_height = 0;
    vk::UniqueImageView ui_storage_view;
};

} // namespace Vulkan
