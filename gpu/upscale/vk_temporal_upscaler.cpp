// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/vk_temporal_upscaler.h"
#include "shim/bb_env.h"
#include "render/vk_gpu_profiler.h"
#include "upscale/motion_history.h"
#include "upscale/vk_scene_resolution.h"
#include "upscale/ui_composition.h"

#include <array>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <vk_mem_alloc.h>

#include "bbport_guest_hooks.h"
#include "game_profile.h"
#include "bbport_settings.h"
#include "bbport_toggles.h"
#include "ffx_vk_portable.h"
#include "bbgpu/shaders/upscale_merge_comp.h"
#include "bbgpu/shaders/upscale_reactive_comp.h"
#include "bbgpu/shaders/taa_comp.h"
#include "bbgpu/shaders/taa_sharpen_comp.h"
#include "bbgpu/shaders/taa_sharpen_ldr_comp.h"
#include "bbgpu/shaders/fsr1_easu_comp.h"
#include "bbgpu/shaders/fsr4_decode_comp.h"
#include "bbgpu/shaders/fsr4_encode_comp.h"
#include "bbgpu/shaders/fsr4_reactive_comp.h"
#include "vulkan/shader_module.h"
#include "upscale/vk_camera_motion.h"
#include "upscale/vk_frame_generation.h"
#include "upscale/vk_smaa.h"
#include "render/vk_frame_capture.h"
#include "vulkan/instance.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"
#include "upscale/upscaler_internal.h"

namespace Vulkan {

namespace {
/// bbport: the menu reads the reason from another thread: keep each one alive (as FSR 4 does).
void PublishMetalFxProblem(const char* problem) {
    static std::array<std::string, 8> kept;
    static u32 next = 0;
    auto& setting = BbSettings::Get().metalfx_problem;
    if (!problem) {
        setting = nullptr;
    } else if (!setting.load() || kept[(next + kept.size() - 1) % kept.size()] != problem) {
        kept[next] = problem;
        setting = kept[next].c_str();
        next = (next + 1) % kept.size();
    }
}
} // namespace

TemporalUpscaler::TemporalUpscaler(const Instance& instance_, Scheduler& scheduler_,
                                   VideoCore::TextureCache& texture_cache_, Runtime& runtime_,
                                   CameraMotion& camera_motion_, SceneTargets& scene_targets_)
    : instance{instance_}, scheduler{scheduler_}, texture_cache{texture_cache_},
      runtime{runtime_}, camera_motion{camera_motion_}, scene_targets{scene_targets_} {
    // Reject unsupported shaders before allocating resources or recording a frame.
    metalfx = std::make_unique<MetalFxUpscaler>(instance, scheduler);
    BbSettings::ConfigureUpscalerSupport(instance.IsFsr4Int8Supported(),
                                         instance.IsFsr411Supported(), metalfx->Supported());
#ifdef __APPLE__
    PublishMetalFxProblem(metalfx->Problem());
#endif
    fsr4 = std::make_unique<Fsr4Upscaler>(instance, scheduler);
    BbSettings::ConfigureFsrAmdSupport(fsr4->AmdAvailable());
    frame_generation = std::make_unique<FrameGeneration>(instance, scheduler);
    {
        const Dlss* dlss = Dlss::Get();
        static std::string problem;
        problem = !dlss ? "the DLSS bridge and NVIDIA's DLSS library are not installed"
                        : dlss->Problem();
        BbSettings::ConfigureDlssSupport(dlss && dlss->Available(), problem.c_str());
    }
    // Available unless BB_UPSCALER=none; on/off and the parameters are the menu's settings.
    const char* env = bb_env("BB_UPSCALER");
    enabled = !(env && std::strcmp(env, "none") == 0);
    if (const char* hash = bb_env("BB_UPSCALE_BEFORE_CS")) {
        trigger_hash = std::strtoull(hash, nullptr, 16);
    }
    if (const char* hash = bb_env("BB_UI_TRIGGER_VS")) {
        ui_trigger_vs = std::strtoull(hash, nullptr, 16);
    }
    if (const char* res = bb_env("BB_OUTPUT_RES")) {
        u32 w = 0, h = 0;
        if (std::sscanf(res, "%ux%u", &w, &h) == 2 && w && h) {
            target_width = w;
            target_height = h;
        }
    }
    // Explicit resolution overrides retain the old compatibility path. Normal presets
    // keep guest allocations/UI native and change only host raster targets at frame boundaries.
    scaled_session = bb_env("BB_RENDER_RES") && bb_env("BB_RENDER_RES")[0];
    render_width = 1920;
    render_height = 1080;
    if (const char* res = bb_env("BB_RENDER_RES")) {
        u32 w = 0, h = 0;
        if (std::sscanf(res, "%ux%u", &w, &h) == 2 && w && h) {
            render_width = w;
            render_height = h;
        }
    }
    native_width = render_width;
    native_height = render_height;
    if (const char* env = bb_env("BB_LIVE_HYBRID"); scaled_session && env && env[0] == '1') {
        hybrid = true;
        startup_preset = BbSettings::Get().preset.load();
        scene_targets.SetNative({native_width, native_height});
        std::printf("Upscaler: the game's scene targets are %ux%u (startup patch); another preset "
                    "chosen in the game is drawn at its size until the next start\n",
                    native_width, native_height);
    }
    // A GPU that cannot give the game's depth a proxy (GTX 10: no D32S8 blit, no stencil
    // export) never reduced the scene: a live preset or output only added work (a GTX 1060 went
    // from its 1080p rate to below it at 720p, 2026-10-21). There they apply at the next start.
    BbSettings::Get().scene_proxies = scene_targets.DepthProxies();
    if (!scene_targets.DepthProxies()) {
        std::printf("Upscaler: this GPU cannot reduce the game's depth live (no D32S8 blit or "
                    "stencil export): presets and the output apply at the next start\n");
    }
    // Scene depth copied into the output-size UI depth by a blit (depth aspect).
    const auto features = instance.GetPhysicalDevice()
                              .getFormatProperties(vk::Format::eD32SfloatS8Uint)
                              .optimalTilingFeatures;
    depth_blit = (features & vk::FormatFeatureFlagBits::eBlitSrc) &&
                 (features & vk::FormatFeatureFlagBits::eBlitDst);
    if (enabled && !instance.IsStorageImageWriteWithoutFormatEnabled()) {
        std::printf("Upscaler: shaderStorageImageWriteWithoutFormat unsupported, FSR 3 off\n");
        enabled = false;
    }
    if (enabled) {
        std::printf("Upscaler: FSR 3.1 available (%s) on scene color before compute shader "
                    "%016llx\n",
                    BbSettings::Get().upscaler != BbSettings::UpscalerOff ? "on" : "off",
                    static_cast<unsigned long long>(trigger_hash));
    }
}

TemporalUpscaler::~TemporalUpscaler() {
    if (resources_ready) scheduler.Finish();
    if (context) {
        scheduler.Finish();
        ffxVkPortableUpscaleContextDestroy(context);
    }
}

bool TemporalUpscaler::Active() const {
    // Toggle 1 << 24 switches it off at run time (A/B); history restarts after.
    return enabled && !failed &&
           (BbSettings::Get().upscaler == BbSettings::UpscalerFsr3 ||
            BbSettings::IsFsr4(BbSettings::Get().upscaler) ||
            BbSettings::Get().upscaler == BbSettings::UpscalerTaa ||
            BbSettings::Get().upscaler == BbSettings::UpscalerFsr1 ||
            BbSettings::Get().upscaler == BbSettings::UpscalerDlss ||
            BbSettings::Get().upscaler == BbSettings::UpscalerMetalFx) &&
           !BbToggle::Disabled(1u << 24);
}

bool TemporalUpscaler::ReactiveOn() const {
    // FSR 4 takes no reactive mask itself: the mask blends its output afterwards
    // (RecordFsr4Reactive). DLSS and MetalFX take none.
    return BbSettings::Get().reactive && !BbToggle::Disabled(1u << 27) && !UseDlss() &&
           BbSettings::Get().upscaler != BbSettings::UpscalerTaa &&
           !BbSettings::IsSpatial(BbSettings::Get().upscaler) &&
           BbSettings::Get().upscaler != BbSettings::UpscalerMetalFx;
}

void TemporalUpscaler::OnSceneColor(VideoCore::ImageId color) {
    scene_color = color;
}

namespace {
float Halton(u32 index, u32 base) {
    float f = 1.0f, result = 0.0f;
    for (u32 i = index; i > 0; i /= base) {
        f /= float(base);
        result += f * float(i % base);
    }
    return result;
}
} // namespace

bool TemporalUpscaler::ProxiesOn() const {
    return !scaled_session || (hybrid && scene_targets.Reduced());
}

bool TemporalUpscaler::ReducedScene(const VideoCore::Image& color) const {
    return ProxiesOn() && scene_targets.Reduced() && scene_targets.EligibleScene(color);
}

bool TemporalUpscaler::RasterScaling() const {
    return ProxiesOn() && scene_targets.Reduced() && !done_this_frame;
}

bool TemporalUpscaler::OnFrameStart() {
    // bbport: BB_PRESET_FILE=<file> holding "preset [output [upscaler [aspect]]]" numbers, read about
    // once a second: switches them like the menu does (scripted tests of live changes).
    static const char* preset_file = bb_env("BB_PRESET_FILE");
    if (preset_file && ++preset_file_frames % 64 == 0) {
        if (FILE* f = std::fopen(preset_file, "r")) {
            int value = 0;
            int output = -1;
            int provider = -1;
            int aspect = -1;
            const int fields = std::fscanf(f, "%d %d %d %d", &value, &output, &provider, &aspect);
            if (fields >= 1 && value >= 0 &&
                value < BbSettings::PresetCount) {
                BbSettings::Get().preset.store(value);
            }
            // A fourth number: the aspect ratio (BbSettings::Aspect), before the output's number.
            if (fields == 4 && aspect >= 0 && aspect < BbSettings::AspectCount) {
                BbSettings::Get().aspect.store(aspect);
            }
            // The output: its number in the current aspect ratio's list.
            const auto outputs = BbSettings::Outputs(BbSettings::Get().aspect);
            if (fields >= 2 && output >= 0 && output < outputs.count) {
                BbSettings::Get().output_res.store(BbSettings::PackSize(outputs.sizes[output]));
            }
            if (fields == 3 && provider >= 0 && provider < BbSettings::UpscalerCount) {
                BbSettings::Get().upscaler.store(provider);
            }
            std::fclose(f);
        }
    }
    const auto& settings = BbSettings::Get();
    // The guest's startup resolution patch remains in effect until restart. Keep the FSR
    // model at the applied preset while the menu saves the requested one for run.sh.
    const int preset = BbSettings::RenderPreset();
    if (applied_preset != preset || settings.upscaler == BbSettings::UpscalerOff) failed = false;
    if (dispatch_failed.exchange(false, std::memory_order_relaxed)) {
        failed = true;
    }
    const bool active = Active();
    // FSR 1 is spatial: a jittered scene would only shimmer.
    const bool jitter_on = active && settings.jitter && !BbToggle::Disabled(1u << 25) &&
                           !BbSettings::IsSpatial(settings.upscaler);
    const int upscaler = settings.upscaler.load();
    auto output_size = BbSettings::CurrentOutput();
    // Without depth proxies the output stays the launch's too: a smaller one only rescaled the
    // same full-size frame at a cost (it applies at the next start, ResolutionNeedsRestart).
    if (!settings.scene_proxies.load(std::memory_order_relaxed) && startup_output != 0) {
        output_size = BbSettings::UnpackSize(startup_output);
    }
    if (!Game::CameraAspect() && output_size.width * 9 != output_size.height * 16) {
        // A game whose camera keeps 16:9: its picture at that shape, as high (wide) as the output
        // allows; the presenter puts it in the middle.
        if (output_size.width * 9 > output_size.height * 16) {
            output_size.width = int(std::lround(output_size.height * 16 / 9.0 / 2)) * 2;
        } else {
            output_size.height = int(std::lround(output_size.width * 9 / 16.0 / 2)) * 2;
        }
    }
    const u32 output = BbSettings::PackSize(output_size);
    // Over the startup patch (hybrid) too: the output is the port's own images (the upscaler's
    // output, the UI's, the display buffer); the scene follows it within the game's targets.
    const bool output_changed = (!scaled_session || hybrid) && applied_output != output;
    if (startup_output == 0) {
        startup_output = output; // the launch's (run.sh patched the game's targets for it)
        if (BbSettings::Get().startup_output.width == 0) {
            BbSettings::Get().startup_output = output_size; // a session without the patch
        }
    }
    if (output_changed) {
        target_width = u32(output_size.width);
        target_height = u32(output_size.height);
        std::printf("Output resolution: %ux%u (live)\n", target_width, target_height);
        failed = false;
        fsr4_failed = false;
        metalfx->Retry();
    }
    // The game's camera and HUD markers follow the output's aspect ratio (bbport_guest_hooks).
    if (const float aspect = enabled ? float(target_width) / float(target_height) : 16.0f / 9.0f;
        aspect != applied_aspect) {
        applied_aspect = aspect;
        BbGuestHooks::SetDisplayAspect(aspect);
    }
    const bool changed = output_changed || applied_preset != preset || active != last_active ||
                         jitter_on != last_jitter || applied_upscaler != upscaler;
    if (applied_upscaler != upscaler || (output_changed && BbSettings::IsFsr4(upscaler))) {
        // A failed provider keeps a fatal flag internally; a user retry gets a fresh context, and
        // so does another output (one it could not do, 7680x2160, must not keep it off at 2560x1080).
        // Reset, not a new object: that would wait for a model being built (#213).
        scheduler.Finish();
        fsr4->Reset();
        if (BbSettings::IsFsr4(upscaler)) BbSettings::Get().fsr4_problem = nullptr;
    }
    if (applied_upscaler != upscaler) fsr4_failed = false; // retry after a menu change
    if (applied_upscaler != upscaler) metalfx->Retry();
    // Dynamic scene resolution scaling (live preset switching) works on all GPUs.
    // On GPUs without D32S8 blit support, UI depth is cleared instead of copied from scene.
    if (!scaled_session) {
        scene_targets.SetSize(SceneResolution::ForPreset(active ? preset : 0,
                                                        {target_width, target_height}));
        render_width = scene_targets.Size().width;
        render_height = scene_targets.Size().height;
    } else if (hybrid) {
        // The launch preset at the launch output is the game's own size; another scene goes
        // through proxies, a larger one too (a higher preset or a larger output). Checked
        // 2026-10-20: 2560x1080 at launch, 3840x1600 chosen live, against a start at 3840x1600:
        // as sharp, the same FPS (the proxies are copied and sampled directly, not read back at
        // the game's size). BB_SCENE_UPSIZE=0: a larger scene is drawn at the largest size of
        // its shape within the game's targets and upscaled from there.
        const int wanted = active ? preset : 0;
        auto scene = SceneResolution::ForPreset(wanted, {target_width, target_height});
        // Exactly 4/3 of the proxy (Quality's targets, Performance chosen): the game's screen-door
        // dithering (a 4-pixel pattern its blur then closes) lands on every third proxy pixel in
        // both directions and the blur misses it, a grid of dark dots. 8 pixels less (under 1%)
        // keeps the pattern from lining up (1.3375 on a Steam Deck shows none).
        const auto near_four_thirds = [](u32 native, u32 proxy) {
            return proxy > 16 && std::abs(double(native) / proxy - 4.0 / 3.0) < 0.004;
        };
        if (near_four_thirds(native_width, scene.width)) scene.width -= 8;
        if (near_four_thirds(native_height, scene.height)) scene.height -= 8;
        static const bool upsize = !bb_env_off("BB_SCENE_UPSIZE");
        if (!upsize && (scene.width > native_width || scene.height > native_height)) {
            const double fit = std::min(double(native_width) / scene.width,
                                        double(native_height) / scene.height);
            scene.width = std::max(16u, u32(scene.width * fit) & ~1u);
            scene.height = std::max(16u, u32(scene.height * fit) & ~1u);
        }
        const bool native_scene = (wanted == startup_preset && output == startup_output) ||
                                  (scene.width == native_width && scene.height == native_height);
        scene_targets.SetSize(native_scene ? SceneResolution::Size{native_width, native_height}
                                           : scene);
        render_width = scene_targets.Size().width;
        render_height = scene_targets.Size().height;
    }
    // Without depth proxies the scene keeps the game's size (see the constructor).
    if (!BbSettings::Get().scene_proxies.load(std::memory_order_relaxed) && scene_targets.Reduced()) {
        scene_targets.SetSize(scene_targets.Native());
        render_width = scene_targets.Size().width;
        render_height = scene_targets.Size().height;
    }
    BbSettings::Get().active_render_width = Scaled() ? render_width : scene_targets.Size().width;
    BbSettings::Get().active_render_height = Scaled() ? render_height : scene_targets.Size().height;
    BbSettings::Get().frame_generation_restart = frame_generation && frame_generation->Wanted() && !Scaled();
    if (changed) {
        NoteReset(0);
    } else if (!dispatched_last_frame) {
        NoteReset(1);
    }
    if (changed) jitter_index = 0;
    applied_preset = preset;
    applied_output = output;
    applied_upscaler = upscaler;
    last_active = active;
    last_jitter = jitter_on;
    dispatched_last_frame = false;

    // Diagnostics BB_UI_TRACE=1: the draws of frames that ended without the output-size UI
    // (their display buffer is shown as the game drew it, at its own size), and of one that had it.
    if (ui_trace_on) {
        static int bad = 0, good = 0;
        if ((!ui_phase && Scaled() && good > 0 && bad < 12) || (ui_phase && good < 1)) {
            ++(ui_phase ? good : bad);
            std::printf("UI trace: frame %llu %s:%s\n",
                        (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed),
                        ui_phase ? "with the output-size UI" : "WITHOUT it", ui_trace.c_str());
        }
        ui_trace.clear();
    }
    // The display pass of an upscaled frame reads the upscaled UI image.
    display_redirect = ui_phase;
    ui_phase = false;
    ui_read_barrier = false;
    done_this_frame = false;
    smaa_done = false;
    ldr_target = {};
    scene_color = {};
    snapshot_taken = false;
    opaque_valid = false;
    mask_ready = false;
    // Halton(2, 3); the menu or toggle 1 << 25 disables it.
    if (!jitter_on) {
        jitter = {};
        return changed;
    }
    // CameraMotion::OnDisplayPass has already cleared Depth(). Use the context's render
    // size, which survives the frame boundary (Performance 960 -> 1920 needs 32 phases).
    const u32 phases = Scaled() ? Motion::JitterPhases(render_width, target_width)
        : Motion::JitterPhases(scene_targets.Size().width, 1920);
    jitter_index = jitter_index % phases + 1;
    jitter = {Halton(jitter_index, 2) - 0.5f, Halton(jitter_index, 3) - 0.5f};
    return changed;
}

/// Why the history was reset (the log, every 5 s while resets happen): 0 settings changed,
/// 1 the last frame did not reach the upscaler, 2 no scene color, 3 camera motion not ready,
/// 4 inactive, 5 resources made, 6 a frame of UI only (menu, loading screen).
void TemporalUpscaler::NoteReset(u32 why) {
    reset = true;
    static std::array<std::atomic<u32>, 7> counts{};
    static auto last = std::chrono::steady_clock::now();
    ++counts[why];
    const auto now = std::chrono::steady_clock::now();
    if (now - last > std::chrono::seconds(5)) {
        last = now;
        std::printf("Upscaler: history resets in 5 s: settings %u, frame without upscaler %u, no scene "
                    "color %u, camera not ready %u, inactive %u, resources %u, UI-only frame %u\n",
                    counts[0].exchange(0), counts[1].exchange(0), counts[2].exchange(0),
                    counts[3].exchange(0), counts[4].exchange(0), counts[5].exchange(0),
                    counts[6].exchange(0));
    }
}

void TemporalUpscaler::OnDispatch(u64 cs_hash) {
    if (cs_hash != trigger_hash || done_this_frame || failed || Scaled()) {
        return;
    }
    done_this_frame = true;
    if (!scene_color || !camera_motion.Ready() || !Active()) {
        NoteReset(!scene_color ? 2 : !camera_motion.Ready() ? 3 : 4);
        return;
    }
    Run();
}

void TemporalUpscaler::Run() {
    if (auto* profiler = GpuProfiler::Get()) {
        const char* label = BbSettings::Get().upscaler == BbSettings::UpscalerTaa
            ? "upscaler Run (TAA)" : "upscaler Run (FSR)";
        profiler->Mark(0xF5A0'0000ull ^ std::hash<std::string_view>{}(label),
                       [label] { return std::string{label}; });
    }
    auto& color = texture_cache.GetImage(scene_color);
    auto& depth = texture_cache.GetImage(camera_motion.Depth());
    const u32 ow = color.info.size.width, oh = color.info.size.height;
    const bool reduced = ReducedScene(color) && scene_targets.EligibleScene(depth);
    const u32 w = reduced ? scene_targets.Size().width : ow;
    const u32 h = reduced ? scene_targets.Size().height : oh;
    if (color.info.pixel_format != vk::Format::eR16G16B16A16Sfloat ||
        depth.info.size.width != ow || depth.info.size.height != oh ||
        !(color.usage_flags & vk::ImageUsageFlagBits::eStorage)) {
        return;
    }
    if (!EnsureResources(w, h, ow, oh, true)) {
        failed = true;
        return;
    }
    const auto depth_format = depth.info.pixel_format;
    const auto depth_view =
        CachedView(depth, depth_format, vk::ImageAspectFlagBits::eDepth);
    const auto color_view =
        CachedView(color, vk::Format::eR16G16B16A16Sfloat, vk::ImageAspectFlagBits::eColor);

    scheduler.EndRendering();
    runtime.Transit(&depth, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.Transit(&color, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead);
    runtime.FlushBarriers();
    vk::Image input_color = color.GetImage(), input_depth = depth.GetImage();
    vk::ImageView input_color_view = color_view, input_depth_view = depth_view;
    if (reduced) {
        VideoCore::ImageViewInfo ci;
        ci.format = color.info.pixel_format;
        const auto c = scene_targets.Read(scene_color, ci);
        ci.format = depth.info.pixel_format;
        const auto d = scene_targets.Read(camera_motion.Depth(), ci);
        input_color = c.image;
        input_depth = d.image;
        input_color_view = c.view;
        input_depth_view = d.view;
    }
    // bbport: not const: MetalFX submits the frame so far and continues in a new command buffer.
    auto cmdbuf = scheduler.CommandBuffer();
    const Breadcrumbs::Scope crumb{cmdbuf, scheduler.CrumbStream(), "TAA"};

    const auto own_barrier = [&](vk::Image image, vk::ImageLayout old_layout,
                                 vk::PipelineStageFlags2 src_stage, vk::AccessFlags2 src_access,
                                 vk::ImageLayout new_layout, vk::PipelineStageFlags2 dst_stage,
                                 vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 barrier{
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .image = image,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
        };
        cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
    };
    const auto all = vk::PipelineStageFlagBits2::eAllCommands;
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    // Previous contents are not needed: the layouts start from undefined every frame.
    own_barrier(vk::Image(motion_image), vk::ImageLayout::eUndefined, all,
                vk::AccessFlagBits2::eNone, vk::ImageLayout::eGeneral,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite);
    own_barrier(vk::Image(output_image), vk::ImageLayout::eUndefined, all,
                vk::AccessFlagBits2::eNone, vk::ImageLayout::eGeneral, all, rw);

    camera_motion.RecordMotion(input_depth_view, *motion_view, w, h);
    own_barrier(vk::Image(motion_image), vk::ImageLayout::eGeneral,
                vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
                vk::ImageLayout::eGeneral, all, vk::AccessFlagBits2::eShaderRead);

    const bool has_reactive = mask_ready || RecordReactive(input_color_view);

    const auto now = std::chrono::steady_clock::now();
    float frame_ms = std::chrono::duration<float, std::milli>(now - last_frame).count();
    if (frame_ms <= 0.0f || frame_ms > 200.0f) {
        frame_ms = 16.6f;
    }
    last_frame = now;

    bool dispatched = false;
    if (BbSettings::Get().upscaler == BbSettings::UpscalerTaa) {
        RecordTaa(cmdbuf, input_color_view, input_depth_view);
        dispatched = true;
    } else if (UseFsr4()) {
        dispatched = RecordFsr4(cmdbuf, {input_color, input_color_view, w, h, color.info.pixel_format},
                                {input_depth, input_depth_view, w, h, depth_format}, w, h, ow, oh,
                                frame_ms);
        if (dispatched && has_reactive) {
            RecordFsr4Reactive(cmdbuf, input_color_view, w, h, ow, oh);
        }
    } else if (UseMetalFx()) {
        dispatched = RecordMetalFx({input_color, color.info.pixel_format,
                                    vk::ImageAspectFlagBits::eColor},
                                   {input_depth, depth_format, vk::ImageAspectFlagBits::eDepth}, w,
                                   h, ow, oh, true);
        cmdbuf = scheduler.CommandBuffer();
    } else if (UseDlss()) {
        dispatched = RecordDlss(
            cmdbuf,
            {input_color, input_color_view, color.info.pixel_format, vk::ImageAspectFlagBits::eColor, w, h},
            {input_depth, input_depth_view, depth_format, vk::ImageAspectFlagBits::eDepth, w, h}, w, h,
            ow, oh, frame_ms, true);
    } else {
        FfxVkPortableUpscaleDispatchInfo info{};
        info.structSize = sizeof(info);
        info.commandBuffer = cmdbuf;
        info.color = Describe(input_color, color.info.pixel_format, w, h,
                              color.usage_flags, vk::ImageAspectFlagBits::eColor,
                              FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
        info.depth = Describe(input_depth, depth_format, w, h, depth.usage_flags,
                              vk::ImageAspectFlagBits::eDepth,
                              FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
        info.motionVectors = Describe(vk::Image(motion_image), vk::Format::eR16G16Sfloat, w, h,
                                      vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                                      vk::ImageAspectFlagBits::eColor,
                                      FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
        info.output = Describe(vk::Image(output_image), vk::Format::eR16G16B16A16Sfloat, ow, oh,
                               vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                                   vk::ImageUsageFlagBits::eTransferSrc,
                               vk::ImageAspectFlagBits::eColor,
                               FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS);
        // Optional inputs, absent: a described but null image.
        info.exposure.structSize = sizeof(info.exposure);
        info.reactiveMask.structSize = sizeof(info.reactiveMask);
        if (has_reactive) {
            info.reactiveMask = Describe(vk::Image(reactive_image), vk::Format::eR8Unorm, w, h,
                                         vk::ImageUsageFlagBits::eStorage |
                                             vk::ImageUsageFlagBits::eSampled,
                                         vk::ImageAspectFlagBits::eColor,
                                         FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
        }
        info.transparencyAndCompositionMask.structSize = sizeof(info.transparencyAndCompositionMask);
        // Toggle 1 << 26 (tests): the opposite sign convention for FSR.
        const float sign = BbToggle::Disabled(1u << 26) ? -1.0f : 1.0f;
        info.jitterOffset = {sign * jitter[0], sign * jitter[1]};
        info.motionVectorScale = {1.0f, 1.0f};
        info.renderSize = {w, h};
        info.outputSize = {ow, oh};
        info.frameTimeMilliseconds = frame_ms;
        info.preExposure = 1.0f;
        info.cameraNear = camera_motion.Near();
        info.cameraFar = 3000.0f;
        info.cameraVerticalFovRadians = camera_motion.VerticalFov();
        info.viewSpaceToMeters = 1.0f;
        // RCAS strength 0..1 (menu, BB_FSR_SHARPNESS); jitter at 1:1 softens the image slightly.
        // AMD's RCAS ends at 1; ExtraSharpen adds the rest of the menu's 0..2.
        info.sharpness = std::min(BbSettings::Get().sharpness.load(), 1.0f);
        info.enableSharpening = BbSettings::Get().sharpen ? VK_TRUE : VK_FALSE;
        info.reset = reset ? VK_TRUE : VK_FALSE;
        info.frameId = frame_id++;
    
        FfxVkPortableUpscaleCreateInfo create_info{};
        create_info.structSize = sizeof(create_info);
        create_info.flags = FFX_VK_PORTABLE_CONTEXT_HDR_COLOR_INPUT | FFX_VK_PORTABLE_CONTEXT_AUTO_EXPOSURE;
        create_info.maxRenderSize = {w, h};
        create_info.maxOutputSize = {ow, oh};
        if (const u64 issues = ffxVkPortableValidateUpscaleDispatchInfo(&create_info, &info)) {
            static bool printed = false;
            if (!printed) {
                printed = true;
                PrintIssues("dispatch", issues);
            }
            failed = true;
        } else if (ffxVkPortableUpscaleContextRecordDispatch(context, &info) !=
                   FFX_VK_PORTABLE_OK) {
            std::printf("Upscaler: FSR 3 dispatch failed\n");
            failed = true;
        } else {
            dispatched = true;
        }
    }
    if (dispatched) {
        reset = false;
        dispatched_last_frame = true;
        if (BbSettings::Get().upscaler != BbSettings::UpscalerTaa) {
            ExtraSharpen(vk::Image(output_image), false, ow, oh,
                         UseMetalFx() ? 0.0f : 1.0f);
        }
        // bbport: BB_DUMP_TRIGGER on this (HDR scene color) path too: the upscaler's inputs and
        // its result, for ghosting and history checks (tools/dump_view.py).
        if (const int dump = DumpFrame(); dump >= 0) {
            camera_motion.PrintState(dump);
            const vk::Format format = color.info.pixel_format;
            const bool rgba16f = format == vk::Format::eR16G16B16A16Sfloat;
            const bool r11g11b10 = format == vk::Format::eB10G11R11UfloatPack32;
            if (rgba16f || r11g11b10) {
                DumpImages(instance, scheduler, cmdbuf, dump,
                           {{input_color, w, h, rgba16f ? 8u : 4u, "input",
                             rgba16f ? "rgba16f" : "r11g11b10f"}});
            }
            DumpImages(instance, scheduler, cmdbuf, dump,
                       {{vk::Image(motion_image), w, h, 4, "motion", "rg16f"},
                        {vk::Image(output_image), ow, oh, 8, "output", "rgba16f"}});
            DumpImages(instance, scheduler, cmdbuf, dump,
                       {{input_depth, w, h, 4, "depth", "f32", vk::ImageAspectFlagBits::eDepth}});
            if (const auto object = camera_motion.ObjectMotionImage(w, h); object) {
                DumpImages(instance, scheduler, cmdbuf, dump,
                           {{object, w, h, 16, "objects", "rgba32f"}});
            }
        }
        // The result replaces the scene color's RGB (its alpha carries data for the post).
        own_barrier(vk::Image(output_image), vk::ImageLayout::eGeneral, all, rw,
                    vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                    vk::AccessFlagBits2::eShaderRead);
        runtime.Transit(&color, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
        runtime.FlushBarriers();
        const vk::DescriptorImageInfo out_info{.imageView = *output_view,
                                               .imageLayout = vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo scene_info{.imageView = color_view,
                                                 .imageLayout = vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo mask_info{.imageView = *reactive_view,
                                                .imageLayout = vk::ImageLayout::eGeneral};
        // Sampled for the debug view only; the object image falls back to the motion image.
        const vk::ImageView objects = camera_motion.ObjectMotionView();
        const vk::DescriptorImageInfo motion_info{.imageView = *motion_view,
                                                  .imageLayout = vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo objects_info{
            .imageView = objects ? objects : *motion_view,
            .imageLayout = vk::ImageLayout::eGeneral};
        const std::array<vk::WriteDescriptorSet, 5> writes = {{
            {.dstBinding = 0,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .pImageInfo = &out_info},
            {.dstBinding = 1,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .pImageInfo = &scene_info},
            {.dstBinding = 2,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage,
             .pImageInfo = &mask_info},
            {.dstBinding = 3,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .pImageInfo = &motion_info},
            {.dstBinding = 4,
             .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eSampledImage,
             .pImageInfo = &objects_info},
        }};
        // Debug (menu or toggle 1 << 28): the reactive mask in red over a darkened frame;
        // or the motion vectors (menu).
        const int debug_view = BbSettings::Get().debug_view;
        const u32 mode = has_reactive && (debug_view == BbSettings::DebugReactive ||
                                          BbToggle::Disabled(1u << 28))
                             ? 1
                         : debug_view == BbSettings::DebugMotion ? 2
                                                                 : 0;
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *merge_pipeline);
        cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *merge_pipeline_layout, 0,
                                    writes);
        cmdbuf.pushConstants(*merge_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
                             sizeof(mode), &mode);
        cmdbuf.dispatch((ow + 7) / 8, (oh + 7) / 8, 1);
    }
}

} // namespace Vulkan

namespace Vulkan {

std::array<u32, 2> TemporalUpscaler::SceneSize(u32 w, u32 h) const {
    if (ProxiesOn()) return {render_width, render_height};
    // The game's own render size (scene constants); BB_RENDER_RES until the first camera.
    auto size = camera_motion.RenderSize();
    if (size[0] == 0 || size[1] == 0) {
        size = {render_width, render_height};
    }
    return {std::min(size[0], w), std::min(size[1], h)};
}

float TemporalUpscaler::SceneMipBias() const {
    if (!Active() || BbSettings::Get().upscaler == BbSettings::UpscalerTaa ||
        BbSettings::IsSpatial(BbSettings::Get().upscaler)) return 0.0f;
    const float render = float(BbSettings::Get().active_render_width.load());
    const float output = float(Scaled() ? target_width : 1920u);
    return render > 0.0f && render < output ? std::log2(render / output) : 0.0f;
}

bool TemporalUpscaler::Scaled() const {
    // FSR 1 upscales the finished frame: the game's post runs at the scene size at 1080p too.
    return scaled_session || target_width != 1920 || target_height != 1080 ||
           BbSettings::IsSpatial(BbSettings::Get().upscaler);
}

void TemporalUpscaler::RunScaled() {
    if (auto* profiler = GpuProfiler::Get()) {
        // The chosen upscaler (FSR 4 runs FSR 3.1 until its model is built, a few seconds).
        static constexpr const char* Labels[BbSettings::UpscalerCount] = {
            "upscaler RunScaled (off)",     "upscaler RunScaled (FSR 3.1)", "upscaler RunScaled (FSR 4)",
            "upscaler RunScaled (FSR 4.1.1)", "upscaler RunScaled (TAA)",   "upscaler RunScaled (DLSS)",
            "upscaler RunScaled (FSR 1)",   "upscaler RunScaled (FSR 4, AMD's DLL)",
            "upscaler RunScaled (MetalFX)"}; // bbport: BbSettings::UpscalerMetalFx
        const int chosen = BbSettings::Get().upscaler;
        const char* label = chosen >= 0 && chosen < BbSettings::UpscalerCount ? Labels[chosen]
                                                                                : "upscaler RunScaled";
        profiler->Mark(0xF5A0'0000ull ^ std::hash<std::string_view>{}(label),
                       [label] { return std::string{label}; });
    }
    auto& color = texture_cache.GetImage(ldr_target);
    auto& depth = texture_cache.GetImage(camera_motion.Depth());
    // The scene fills the top-left w x h of its targets (iw x ih, aligned by the game).
    const u32 iw = color.info.size.width, ih = color.info.size.height;
    const auto [w, h] = SceneSize(iw, ih);
    const u32 ow = target_width, oh = target_height;
    if (depth.info.size.width != iw || depth.info.size.height != ih || w > ow || h > oh) {
        return;
    }
    if (!EnsureResources(w, h, ow, oh, false)) {
        failed = true;
        return;
    }
    EnsureUiResources(ow, oh, color.info.pixel_format, depth.info.pixel_format);
    PrepareUiDepth(camera_motion.Depth());
    const auto depth_format = depth.info.pixel_format;
    // Views kept across frames: FSR 4 registers images by view in a registry of eight.
    vk::Image depth_image = depth.GetImage(), color_image = color.GetImage();
    vk::ImageView depth_view{}, color_view{};
    u32 source_width = iw, source_height = ih;
    if (ProxiesOn()) {
        VideoCore::ImageViewInfo ci, di;
        ci.format = color.info.pixel_format;
        di.format = depth_format;
        const auto cp = scene_targets.Read(ldr_target, ci);
        const auto dp = scene_targets.Read(camera_motion.Depth(), di);
        color_image = cp.image;
        color_view = cp.view;
        depth_image = dp.image;
        depth_view = dp.view;
        source_width = w;
        source_height = h;
    } else {
        depth_view = CachedView(depth, depth_format, vk::ImageAspectFlagBits::eDepth);
        color_view = CachedView(color, color.info.pixel_format, vk::ImageAspectFlagBits::eColor);
        runtime.Transit(&depth, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.Transit(&color, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eComputeShader,
                        vk::AccessFlagBits2::eShaderRead);
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    // bbport: the motion pass and FSR 3 are recorded through the scheduler, on a recording
    // thread (recording them here cost the draw recording thread ~5% of its time, plus a wait
    // for the recording thread every frame). FSR 4 and TAA keep their frame bookkeeping on
    // this thread and record directly below.
    const auto barrier = [&](vk::Image image, vk::ImageAspectFlags aspect,
                             vk::ImageLayout old_layout, vk::PipelineStageFlags2 src_stage,
                             vk::AccessFlags2 src_access, vk::ImageLayout new_layout,
                             vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
        const vk::ImageMemoryBarrier2 b{
            .srcStageMask = src_stage,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stage,
            .dstAccessMask = dst_access,
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .image = image,
            .subresourceRange = {aspect, 0, 1, 0, 1},
        };
        scheduler.Record([b](vk::CommandBuffer cmdbuf) {
            cmdbuf.pipelineBarrier2({.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
        });
    };
    const auto all = vk::PipelineStageFlagBits2::eAllCommands;
    const auto color_access = vk::AccessFlagBits2::eColorAttachmentRead |
                              vk::AccessFlagBits2::eColorAttachmentWrite;
    const auto rw = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
    // The previous frame's display pass read it: all commands before.
    barrier(vk::Image(ui_image), vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
            all, vk::AccessFlagBits2::eNone, vk::ImageLayout::eGeneral, all, rw);
    barrier(vk::Image(motion_image), vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eUndefined,
            all, vk::AccessFlagBits2::eNone, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite);
    const bool spatial = BbSettings::IsSpatial(BbSettings::Get().upscaler);
    if (!spatial) { // FSR 1 takes no motion vectors
        camera_motion.RecordMotion(depth_view, *motion_view, w, h);
    }
    barrier(vk::Image(motion_image), vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral,
            vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite,
            vk::ImageLayout::eGeneral, all, vk::AccessFlagBits2::eShaderRead);

    const auto now = std::chrono::steady_clock::now();
    float frame_ms = std::chrono::duration<float, std::milli>(now - last_frame).count();
    if (frame_ms <= 0.0f || frame_ms > 200.0f) {
        frame_ms = 16.6f;
    }
    last_frame = now;
    const bool frame_reset = reset; // the branches below clear it

    // bbport: FSR 4, DLSS and MetalFX write their HDR-format output, copied into the output-size
    // UI image. They record directly (MetalFX also submits the frame so far and continues in a
    // new command buffer).
    const bool use_metalfx = UseMetalFx();
    if (UseFsr4() || UseDlss() || use_metalfx ||
        BbSettings::Get().upscaler == BbSettings::UpscalerTaa || spatial) {
        auto cmdbuf = scheduler.CommandBuffer(); // after the commands recorded above
        barrier(vk::Image(output_image), vk::ImageAspectFlagBits::eColor,
                vk::ImageLayout::eUndefined, all, vk::AccessFlagBits2::eNone,
                vk::ImageLayout::eGeneral, all, rw);
        bool ok4 = true;
        if (BbSettings::Get().upscaler == BbSettings::UpscalerTaa) {
            RecordTaa(cmdbuf, color_view, depth_view);
        } else if (spatial) {
            RecordEasu(cmdbuf, color_view, w, h, source_width, source_height, ow, oh);
        } else if (use_metalfx) {
            fsr4_linear_frame = false;
            ok4 = RecordMetalFx({color_image, color.info.pixel_format,
                                 vk::ImageAspectFlagBits::eColor},
                                {depth_image, depth_format, vk::ImageAspectFlagBits::eDepth}, w, h,
                                ow, oh, false);
            cmdbuf = scheduler.CommandBuffer();
        } else if (UseDlss()) {
            fsr4_linear_frame = false;
            ok4 = RecordDlss(cmdbuf,
                             {color_image, color_view, color.info.pixel_format,
                              vk::ImageAspectFlagBits::eColor, source_width, source_height},
                             {depth_image, depth_view, depth_format, vk::ImageAspectFlagBits::eDepth,
                              source_width, source_height},
                             w, h, ow, oh, frame_ms, false);
        } else {
            // bbport: FSR 4 treats its colour as linear light; this frame is the game's
            // tonemapped, sRGB-encoded one. Decoded first, the output encoded again below.
            // BB_FSR4_LINEAR=0 or toggle bit 63: the encoded frame as before (A/B).
            static const bool linear_env = [] {
                const char* env = bb_env("BB_FSR4_LINEAR");
                return !env || env[0] != '0';
            }();
            // An sRGB-format frame is decoded by its view already (and encoded by the blit).
            const bool srgb_format =
                vk::to_string(color.info.pixel_format).find("Srgb") != std::string::npos;
            fsr4_linear_frame = linear_env && fsr4_linear_image && !srgb_format &&
                                !BbToggle::Disabled(BbToggle::Fsr4EncodedInput);
            Fsr4Upscaler::Image input{color_image, color_view, source_width, source_height,
                                      color.info.pixel_format};
            {
                static int reported = -1;
                if (int(fsr4_linear_frame) != reported) {
                    reported = int(fsr4_linear_frame);
                    std::printf("Upscaler: FSR 4 input %s (%s)\n",
                                fsr4_linear_frame ? "decoded to linear light" : "as it is",
                                vk::to_string(color.info.pixel_format).c_str());
                }
            }
            if (fsr4_linear_frame) {
                const vk::ImageMemoryBarrier2 to_write{
                    .srcStageMask = all, .srcAccessMask = vk::AccessFlagBits2::eNone,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                    .oldLayout = vk::ImageLayout::eUndefined, .newLayout = vk::ImageLayout::eGeneral,
                    .image = vk::Image(fsr4_linear_image),
                    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};
                const vk::MemoryBarrier2 source_ready{
                    .srcStageMask = all, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderSampledRead};
                cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &source_ready,
                                         .imageMemoryBarrierCount = 1,
                                         .pImageMemoryBarriers = &to_write});
                const vk::DescriptorImageInfo src{.imageView = color_view,
                                                  .imageLayout = vk::ImageLayout::eGeneral};
                const vk::DescriptorImageInfo dst{.imageView = *fsr4_linear_view,
                                                  .imageLayout = vk::ImageLayout::eGeneral};
                const std::array<vk::WriteDescriptorSet, 2> writes{{
                    {.dstBinding = 0, .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eSampledImage, .pImageInfo = &src},
                    {.dstBinding = 1, .descriptorCount = 1,
                     .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &dst},
                }};
                const u32 unused = 0;
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *fsr4_decode_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
                                            *fsr4_decode_pipeline_layout, 0, writes);
                cmdbuf.pushConstants(*fsr4_decode_pipeline_layout,
                                     vk::ShaderStageFlagBits::eCompute, 0, sizeof(unused), &unused);
                cmdbuf.dispatch((source_width + 7) / 8, (source_height + 7) / 8, 1);
                const vk::MemoryBarrier2 decoded{
                    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                    .dstStageMask = all,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderRead};
                cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &decoded});
                input = {vk::Image(fsr4_linear_image), *fsr4_linear_view, source_width,
                         source_height, vk::Format::eR16G16B16A16Sfloat};
            }
            ok4 = RecordFsr4(cmdbuf, input,
                            {depth_image, depth_view, source_width, source_height, depth_format},
                            w, h, ow, oh, frame_ms);
            if (ok4 && fsr4_linear_frame) {
                const vk::MemoryBarrier2 upscaled{
                    .srcStageMask = all, .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
                    .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .dstAccessMask = vk::AccessFlagBits2::eShaderStorageRead |
                                     vk::AccessFlagBits2::eShaderStorageWrite};
                cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &upscaled});
                const vk::DescriptorImageInfo out{.imageView = *output_view,
                                                  .imageLayout = vk::ImageLayout::eGeneral};
                const vk::WriteDescriptorSet write{.dstBinding = 0, .descriptorCount = 1,
                    .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &out};
                const u32 unused = 0;
                cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *fsr4_encode_pipeline);
                cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute,
                                            *fsr4_encode_pipeline_layout, 0, write);
                cmdbuf.pushConstants(*fsr4_encode_pipeline_layout,
                                     vk::ShaderStageFlagBits::eCompute, 0, sizeof(unused), &unused);
                cmdbuf.dispatch((ow + 7) / 8, (oh + 7) / 8, 1);
                const vk::MemoryBarrier2 encoded{
                    .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
                    .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
                    .dstStageMask = all,
                    .dstAccessMask = vk::AccessFlagBits2::eMemoryRead |
                                     vk::AccessFlagBits2::eMemoryWrite};
                cmdbuf.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &encoded});
            }
            // The game's (encoded) frame, as the output is encoded again by now.
            if (ok4 && mask_ready) {
                RecordFsr4Reactive(cmdbuf, color_view, w, h, ow, oh);
            }
        }
        if (ok4) {
            if (BbSettings::Get().upscaler != BbSettings::UpscalerTaa) {
                ExtraSharpen(vk::Image(output_image), false, ow, oh, use_metalfx ? 0.0f : 1.0f);
            }
            barrier(vk::Image(output_image), vk::ImageAspectFlagBits::eColor,
                    vk::ImageLayout::eGeneral, all, rw, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferRead);
            barrier(vk::Image(ui_image), vk::ImageAspectFlagBits::eColor,
                    vk::ImageLayout::eGeneral, all, rw, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eBlit, vk::AccessFlagBits2::eTransferWrite);
            const vk::ImageBlit region{
                .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .srcOffsets = std::array{vk::Offset3D{0, 0, 0},
                                         vk::Offset3D{s32(ow), s32(oh), 1}},
                .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                .dstOffsets = std::array{vk::Offset3D{0, 0, 0},
                                         vk::Offset3D{s32(ow), s32(oh), 1}},
            };
            cmdbuf.blitImage(vk::Image(output_image), vk::ImageLayout::eGeneral,
                             vk::Image(ui_image), vk::ImageLayout::eGeneral, region,
                             vk::Filter::eNearest);
            barrier(vk::Image(ui_image), vk::ImageAspectFlagBits::eColor,
                    vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eBlit,
                    vk::AccessFlagBits2::eTransferWrite, vk::ImageLayout::eGeneral,
                    vk::PipelineStageFlagBits2::eColorAttachmentOutput, color_access);
            reset = false;
            dispatched_last_frame = true;
            if (const int dump = DumpFrame(); dump >= 0) {
                camera_motion.PrintState(dump);
                DumpImages(instance, scheduler, cmdbuf, dump,
                           {{color_image, source_width, source_height, 4, "input", "rgba"},
                            {vk::Image(motion_image), w, h, 4, "motion", "rg16f"},
                            {vk::Image(output_image), ow, oh, 8, "output", "rgba16f"}});
                DumpImages(instance, scheduler, cmdbuf, dump,
                           {{depth_image, source_width, source_height, 4, "depth", "f32",
                             vk::ImageAspectFlagBits::eDepth}});
                if (BbSettings::Get().upscaler == BbSettings::UpscalerTaa &&
                    bb_env("BB_TAA_DIAGNOSTICS")) {
                    DumpImages(instance, scheduler, cmdbuf, dump,
                               {{vk::Image(taa_history[taa_next]),ow,oh,16,
                                 "previous_history","rgba32f"},
                                {vk::Image(opaque_image),ow,oh,8,"history_sample","rgba16f"}});
                }
                // ObjectMotion's attachment can be in a different resolution from the FSR
                // inputs: only dump it when it matches, to avoid an out-of-bounds GPU copy.
                if (const auto object = camera_motion.ObjectMotionImage(w, h); object) {
                    DumpImages(instance, scheduler, cmdbuf, dump,
                               {{object, w, h, 16, "objects", "rgba32f"}});
                }
            }
        }
        if (ok4) {
            CaptureFrameGenScene(depth_image, depth_format, w, h, ow, oh, frame_ms, frame_reset);
        }
        done_this_frame = true;
        if (ok4) {
            ui_phase = true;
            hudless_taken = false;
            ui_color = ldr_target;
            ui_depth = camera_motion.Depth();
        }
        return;
    }

    const auto& settings = BbSettings::Get();
    FfxVkPortableUpscaleDispatchInfo info{};
    info.structSize = sizeof(info);
    info.color = Describe(color_image, color.info.pixel_format, source_width, source_height,
                          color.usage_flags, vk::ImageAspectFlagBits::eColor,
                          FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    info.depth = Describe(depth_image, depth_format, source_width, source_height, depth.usage_flags,
                          vk::ImageAspectFlagBits::eDepth,
                          FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    info.motionVectors = Describe(vk::Image(motion_image), vk::Format::eR16G16Sfloat, w, h,
                                  vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
                                  vk::ImageAspectFlagBits::eColor,
                                  FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    info.output = Describe(vk::Image(ui_image), vk::Format::eR8G8B8A8Unorm, ow, oh,
                           vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                               vk::ImageUsageFlagBits::eColorAttachment,
                           vk::ImageAspectFlagBits::eColor,
                           FFX_VK_PORTABLE_RESOURCE_STATE_UNORDERED_ACCESS);
    info.exposure.structSize = sizeof(info.exposure);
    info.reactiveMask.structSize = sizeof(info.reactiveMask);
    if (mask_ready && ReactiveOn()) {
        info.reactiveMask = Describe(vk::Image(reactive_image), vk::Format::eR8Unorm, w, h,
            vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled,
            vk::ImageAspectFlagBits::eColor, FFX_VK_PORTABLE_RESOURCE_STATE_GENERIC_READ);
    }

    info.transparencyAndCompositionMask.structSize = sizeof(info.transparencyAndCompositionMask);
    const float sign = BbToggle::Disabled(1u << 26) ? -1.0f : 1.0f;
    info.jitterOffset = {sign * jitter[0], sign * jitter[1]};
    info.motionVectorScale = {1.0f, 1.0f};
    info.renderSize = {w, h};
    info.outputSize = {ow, oh};
    info.frameTimeMilliseconds = frame_ms;
    info.preExposure = 1.0f;
    info.cameraNear = camera_motion.Near();
    info.cameraFar = 3000.0f;
    info.cameraVerticalFovRadians = camera_motion.VerticalFov();
    info.viewSpaceToMeters = 1.0f;
    info.sharpness = std::min(settings.sharpness.load(), 1.0f);
    info.enableSharpening = settings.sharpen ? VK_TRUE : VK_FALSE;
    info.reset = reset ? VK_TRUE : VK_FALSE;
    info.frameId = frame_id++;

    FfxVkPortableUpscaleCreateInfo create_info{};
    create_info.structSize = sizeof(create_info);
    create_info.maxRenderSize = {w, h};
    create_info.maxOutputSize = {ow, oh};
    // The command buffer is the recording thread's, known when the dispatch is recorded: the
    // check sees a placeholder.
    auto checked = info;
    checked.commandBuffer = reinterpret_cast<VkCommandBuffer>(uintptr_t{1});
    bool ok = false;
    if (const u64 issues = ffxVkPortableValidateUpscaleDispatchInfo(&create_info, &checked)) {
        PrintIssues("scaled dispatch", issues);
        failed = true;
    } else {
        // The context is only used here and destroyed after scheduler.Finish(); a dispatch
        // that fails sets `failed` at the next frame (the UI then draws over this one).
        scheduler.Record([this, ctx = context, info](vk::CommandBuffer cmdbuf) mutable {
            info.commandBuffer = cmdbuf;
            if (ffxVkPortableUpscaleContextRecordDispatch(ctx, &info) != FFX_VK_PORTABLE_OK) {
                std::printf("Upscaler: FSR 3 scaled dispatch failed\n");
                dispatch_failed.store(true, std::memory_order_relaxed);
            }
        });
        ok = true;
        reset = false;
        dispatched_last_frame = true;
        ExtraSharpen(vk::Image(ui_image), true, ow, oh);
    }
    barrier(vk::Image(ui_image), vk::ImageAspectFlagBits::eColor, vk::ImageLayout::eGeneral, all,
            rw, vk::ImageLayout::eGeneral, vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            color_access);

    if (ok) {
        CaptureFrameGenScene(depth_image, depth_format, w, h, ow, oh, frame_ms, frame_reset);
    }
    done_this_frame = true; // the UI is not jittered
    if (ok) {
        ui_phase = true;
        hudless_taken = false;
        ui_color = ldr_target;
        ui_depth = camera_motion.Depth();
    }
}

bool TemporalUpscaler::UseMetalFx() const {
    return BbSettings::Get().upscaler == BbSettings::UpscalerMetalFx && metalfx->Supported() &&
           !metalfx->Fatal();
}

bool TemporalUpscaler::RecordMetalFx(const MetalFxUpscaler::Input& color,
                                     const MetalFxUpscaler::Input& depth, u32 w, u32 h, u32 ow,
                                     u32 oh, bool hdr) {
    // Same jitter convention as FSR 3; toggle 1 << 26 flips it for tests.
    const float sign = BbToggle::Disabled(1u << 26) ? -1.0f : 1.0f;
    const bool ok = metalfx->Run({
        .color = color,
        .depth = depth,
        .motion = {vk::Image(motion_image), vk::Format::eR16G16Sfloat,
                   vk::ImageAspectFlagBits::eColor},
        .width = w,
        .height = h,
        .output = vk::Image(output_image),
        .out_width = ow,
        .out_height = oh,
        .hdr = hdr,
        .jitter = {sign * jitter[0], sign * jitter[1]},
        .reset = reset,
    });
    PublishMetalFxProblem(metalfx->Problem());
    if (!ok && metalfx->Fatal()) {
        // FSR 3.1 takes over, as for FSR 4 and DLSS (it runs on KosmicKrisp); its own failure
        // turns the upscaler off. The menu shows MetalFX's reason.
        std::printf("Upscaler: MetalFX failed; falling back to FSR 3.1 (select MetalFX again to "
                    "retry)\n");
        BbSettings::Get().upscaler = BbSettings::UpscalerFsr3;
        reset = true;
    }
    return ok;
}

} // namespace Vulkan
