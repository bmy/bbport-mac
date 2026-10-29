// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport_settings.cpp: the menu saves its keys into bbport.ini and keeps the launcher's (controls).
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include "gpu/shim/bbport_settings.h"

static std::string Read(const char* path) {
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

int main() {
    assert(!BbSettings::IsHotEffect(6)); // SSR resizes resources on restart.
    assert(BbSettings::IsHotEffect(7));  // Bloom still switches live.
    assert(!BbSettings::IsHotEffect(8)); // Startup intros.
    char path[] = "/tmp/bbport-settings-test-XXXXXX";
    const int fd = mkstemp(path);
    assert(fd >= 0);
    const char ini[] = "# launcher\nupscaler=fsr3\nkey.cross=X, Space\npad.circle=a\nshow_fps=0\n"
                       "fullscreen_hint=1\n";
    assert(write(fd, ini, sizeof(ini) - 1) == ssize_t(sizeof(ini) - 1));
    close(fd);
    setenv("BB_CONFIG", path, 1);

    BbSettings::Load();
    auto& s = BbSettings::Get();
    assert(s.upscaler == BbSettings::UpscalerFsr3 && !s.show_fps && s.menu_x < 0.0f);
    s.show_fps = true;
    s.upscaler = BbSettings::UpscalerFsr411;
    s.menu_x = 0.625f;
    s.menu_y = 0.125f;
    BbSettings::Save();

    const std::string saved = Read(path);
    // The launcher's controls, its other keys and comments stay; the menu's keys are replaced
    // in place, new ones appended.
    assert(saved.find("# launcher\nupscaler=fsr411\nkey.cross=X, Space\npad.circle=a\nshow_fps=1\n"
                      "fullscreen_hint=1\n") == 0);
    assert(saved.find("menu_pos=0.6250,0.1250\n") != std::string::npos);
    assert(saved.find("upscaler=fsr3") == std::string::npos);

    s.menu_x = -1.0f;
    s.menu_y = -1.0f;
    BbSettings::Load();
    assert(s.menu_x == 0.625f && s.menu_y == 0.125f && s.upscaler == BbSettings::UpscalerFsr411);
    assert(saved.find("menu_language=") == std::string::npos); // only started with it (#134)

    // A language chosen in the menu is saved and wins over the launcher's from then on.
    s.menu_language = BbSettings::MenuLanguage::Russian;
    s.menu_language_chosen = true;
    BbSettings::Save();
    assert(Read(path).find("menu_language=ru\n") != std::string::npos);
    setenv("BB_MENU_LANGUAGE", "en", 1);
    s.menu_language_chosen = false;
    BbSettings::Load();
    assert(s.menu_language == BbSettings::MenuLanguage::Russian && s.menu_language_chosen);

    // Frame generation: DLSS from the file stays; FSR 3.1 only with BB_FRAME_GEN (hidden until
    // its artifacts are fixed); the environment wins; "1"/"0" of earlier versions still parse.
    const auto write_frame_gen = [&](const char* text) {
        FILE* file = std::fopen(path, "w");
        assert(file);
        std::fputs(text, file);
        std::fclose(file);
    };
    unsetenv("BB_FRAME_GEN");
    unsetenv("BB_FRAME_GEN_MULTIPLIER");
    write_frame_gen("frame_generation=dlss\nframe_generation_multiplier=3\n");
    BbSettings::Load();
    assert(s.frame_generation == BbSettings::FrameGenDlss && s.frame_generation_multiplier == 3);
    write_frame_gen("frame_generation=fsr3\nframe_generation_multiplier=9\n");
    BbSettings::Load();
    assert(s.frame_generation == BbSettings::FrameGenOff && s.frame_generation_multiplier == 4);
    setenv("BB_FRAME_GEN", "1", 1);
    BbSettings::Load();
    assert(s.frame_generation == BbSettings::FrameGenFsr3);
    setenv("BB_FRAME_GEN", "dlss", 1);
    setenv("BB_FRAME_GEN_MULTIPLIER", "2", 1);
    BbSettings::Load();
    assert(s.frame_generation == BbSettings::FrameGenDlss && s.frame_generation_multiplier == 2);
    setenv("BB_FRAME_GEN", "0", 1);
    BbSettings::Load();
    assert(s.frame_generation == BbSettings::FrameGenOff);
    unsetenv("BB_FRAME_GEN");
    unsetenv("BB_FRAME_GEN_MULTIPLIER");
    write_frame_gen("frame_generation=1\n");
    BbSettings::Load();
    assert(s.frame_generation == BbSettings::FrameGenOff); // FSR 3.1 without BB_FRAME_GEN
    s.frame_generation = BbSettings::FrameGenDlss;
    s.frame_generation_multiplier = 4;
    BbSettings::Save();
    assert(Read(path).find("frame_generation=dlss\n") != std::string::npos);
    assert(Read(path).find("frame_generation_multiplier=4\n") != std::string::npos);
    assert(std::string(BbSettings::FrameGenName(BbSettings::FrameGenFsr3)) == "fsr3");
    // Low latency (NVIDIA Reflex): on by default, names and numbers parse, the environment wins.
    unsetenv("BB_LOW_LATENCY");
    write_frame_gen("upscaler=fsr3\n");
    BbSettings::Load();
    assert(s.low_latency == BbSettings::LowLatencyOn);
    write_frame_gen("low_latency=boost\n");
    BbSettings::Load();
    assert(s.low_latency == BbSettings::LowLatencyBoost);
    write_frame_gen("low_latency=0\n");
    BbSettings::Load();
    assert(s.low_latency == BbSettings::LowLatencyOff);
    setenv("BB_LOW_LATENCY", "on", 1);
    BbSettings::Load();
    assert(s.low_latency == BbSettings::LowLatencyOn);
    unsetenv("BB_LOW_LATENCY");
    s.low_latency = BbSettings::LowLatencyBoost;
    BbSettings::Save();
    assert(Read(path).find("low_latency=boost\n") != std::string::npos);
    // With DLSS Frame Generation running, Off becomes On; Boost stays; off without it.
    s.low_latency = BbSettings::LowLatencyOff;
    s.frame_generation = BbSettings::FrameGenDlss;
    s.dlssg_max_multiplier = 1; // unavailable: the setting stands
    assert(BbSettings::EffectiveLowLatency() == BbSettings::LowLatencyOff);
    s.dlssg_max_multiplier = 2;
    assert(BbSettings::EffectiveLowLatency() == BbSettings::LowLatencyOn);
    s.low_latency = BbSettings::LowLatencyBoost;
    assert(BbSettings::EffectiveLowLatency() == BbSettings::LowLatencyBoost);
    s.frame_generation = BbSettings::FrameGenOff;
    s.low_latency = BbSettings::LowLatencyOff;
    assert(BbSettings::EffectiveLowLatency() == BbSettings::LowLatencyOff);
    unlink(path);
    std::puts("PASS: settings save keeps other keys, menu position, menu language, frame generation, low latency");
}
