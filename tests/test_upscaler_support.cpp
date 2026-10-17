// SPDX-License-Identifier: GPL-2.0-or-later
// Saved FSR 4 choices must be safe before the first rendered frame on older GPUs.
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <unistd.h>
#include "bbport_settings.h"

int main() {
    using namespace BbSettings;
    char path[] = "/tmp/bbport-upscaler-test-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    close(fd);
    setenv("BB_CONFIG", path, 1);
    unsetenv("BB_UPSCALER");
    unsetenv("BB_UPSCALE_PRESET");
    unsetenv("BB_RENDER_RES");
    auto& s = Get();
    s.upscaler = UpscalerTaa;
    s.preset = Performance;
    assert(RenderPreset() == NativeAA && s.preset == Performance);
    Save();
    s.upscaler = UpscalerOff;
    Load();
    assert(s.upscaler == UpscalerTaa && s.preset == Performance);
    assert(RenderPreset() == NativeAA && !ResolutionNeedsRestart());
    s.upscaler = UpscalerFsr3;
    assert(RenderPreset() == Performance);
    for (bool fsr4 : {false, true}) {
        for (bool fsr411 : {false, true}) {
            for (int requested = 0; requested < UpscalerCount; ++requested) {
                FILE* config = std::fopen(path, "w");
                assert(config);
                std::fprintf(config, "upscaler=%s\npreset=3\noutput_res=2560x1440\n",
                             UpscalerName(requested));
                std::fclose(config);
                s.fsr4_problem = nullptr;
                Load();
                assert(s.upscaler == requested);
                ConfigureUpscalerSupport(fsr4, fsr411);
                const bool fsr4_unsupported = (requested == UpscalerFsr4 && !fsr4) ||
                    (requested == UpscalerFsr411 && !(fsr4 && fsr411));
                // MetalFX (macOS only) is unsupported by default and falls back to off.
                const bool metalfx_unsupported = requested == UpscalerMetalFx;
                const bool unsupported = fsr4_unsupported || metalfx_unsupported;
                const int fallback = metalfx_unsupported ? UpscalerOff : UpscalerFsr3;
                assert(s.upscaler == (unsupported ? fallback : requested));
                assert(s.fsr4_supported == fsr4);
                assert(s.fsr411_supported == (fsr4 && fsr411));
                assert(bool(s.fsr4_problem.load()) == fsr4_unsupported);
                assert(s.preset == Performance && CurrentOutput() == (OutputSize{2560, 1440}));
                // Startup patch settings still describe the already applied guest patches.
                assert(s.startup_upscaler == requested);
                ConfigureUpscalerSupport(fsr4, fsr411);
                assert(s.upscaler == (unsupported ? fallback : requested));
            }
        }
    }
    // A supported MetalFX choice stays.
    s.upscaler = UpscalerMetalFx;
    ConfigureUpscalerSupport(false, false, true);
    assert(s.upscaler == UpscalerMetalFx && s.metalfx_supported);
    ConfigureUpscalerSupport(false, false, false);
    assert(s.upscaler == UpscalerOff && !s.metalfx_supported);
    s.startup_preset = Quality;
    s.startup_upscaler = UpscalerFsr3;
    s.upscaler = UpscalerFsr3;
    s.preset = Performance;
    assert(RenderPreset() == Performance && !ResolutionNeedsRestart());
    setenv("BB_RENDER_RES", "1706x960", 1);
    s.startup_output = CurrentOutput(); // Load with BB_OUTPUT_RES
    assert(RenderPreset() == Quality && ResolutionNeedsRestart());
    s.preset = Quality;
    assert(!ResolutionNeedsRestart());
    s.upscaler = UpscalerOff;
    assert(ResolutionNeedsRestart());
    s.upscaler = UpscalerFsr4;
    assert(!ResolutionNeedsRestart()); // provider change at the same patched size is live
    s.upscaler = UpscalerTaa;          // TAA needs native guest targets
    assert(ResolutionNeedsRestart());
    s.upscaler = UpscalerFsr4;
    s.output_res = PackSize({3840, 2160});
    assert(ResolutionNeedsRestart());
    s.output_res = PackSize(s.startup_output);
    assert(!ResolutionNeedsRestart());
    s.aspect = Aspect21x9; // the same height in another aspect ratio: 3440x1440
    assert(CurrentOutput() == (OutputSize{3440, 1440}) && ResolutionNeedsRestart());
    s.aspect = AspectAuto;
    setenv("BB_RENDER_RES", "", 1);
    assert(!FixedRenderSession() && !ResolutionNeedsRestart());
    unsetenv("BB_RENDER_RES");
    s.output_res = PackSize({1920, 1080});
#ifdef __APPLE__
    // bbport (macOS): Auto is 16:9 whatever the monitor, with none of its own sizes; a saved
    // 16:10 size comes back at the same height class in 16:9.
    SetMonitor(2560, 1080);
    assert(ResolvedAspect(AspectAuto) == Aspect16x9 && CurrentOutput() == (OutputSize{1920, 1080}));
    SetMonitor(1366, 768);
    assert(ResolvedAspect(AspectAuto) == Aspect16x9 && Outputs(AspectAuto).IndexOf({1366, 768}) < 0);
    s.output_res = PackSize({2560, 1600});
    SetMonitor(3024, 1964);
    assert(ResolvedAspect(AspectAuto) == Aspect16x9 && CurrentOutput() == (OutputSize{2560, 1440}));
    s.output_res = PackSize({1920, 1080});
#else
    // Auto: the monitor's group with its own size among them; the chosen height is kept.
    SetMonitor(2560, 1080);
    assert(ResolvedAspect(AspectAuto) == Aspect21x9 && CurrentOutput() == (OutputSize{2560, 1080}));
    SetMonitor(1366, 768);
    assert(ResolvedAspect(AspectAuto) == Aspect16x9 && Outputs(AspectAuto).IndexOf({1366, 768}) == 4);
    assert(CurrentOutput() == (OutputSize{1920, 1080}));
#endif
    assert(NearestOutput(Aspect32x9, {1920, 1080}) == (OutputSize{3840, 1080}));
    assert(NearestOutput(Aspect16x10, {1280, 720}) == (OutputSize{1280, 800}));
    assert(NearestOutput(Aspect4x3, {2560, 1080}) == (OutputSize{1440, 1080}));
#ifndef __APPLE__
    SetMonitor(1024, 768);
    assert(ResolvedAspect(AspectAuto) == Aspect4x3);
#endif
    std::remove(path);
    std::puts("Upscaler support: saved choices, FSR 3.1 and MetalFX fallbacks PASS");
}
