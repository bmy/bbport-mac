// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "render/vk_low_latency.h"

#include <chrono>
#include <cstdio>

#include "bbport_settings.h"
#include "vulkan/instance.h"

namespace Vulkan {

LowLatency& LowLatency::Get() {
    static LowLatency instance;
    return instance;
}

void LowLatency::Initialize(const Instance& instance) {
    device = instance.GetDevice();
    backend = instance.HasLowLatency() ? Backend::Reflex
              : instance.HasAntiLag()  ? Backend::AntiLag
                                       : Backend::None;
    available = backend != Backend::None;
    auto& settings = BbSettings::Get();
    settings.low_latency_supported = available;
    settings.low_latency_backend = int(backend);
    if (!available) {
        std::printf("Low latency: neither VK_NV_low_latency2 (NVIDIA Reflex: NVIDIA's driver 550 "
                    "or newer) nor VK_AMD_anti_lag (AMD Anti-Lag 2: AMD's driver, Mesa's anti-lag "
                    "layer)\n");
        return;
    }
    if (backend == Backend::AntiLag) {
        std::printf("Low latency: AMD Anti-Lag 2 (VK_AMD_anti_lag) available, %s\n",
                    BbSettings::EffectiveLowLatency() != BbSettings::LowLatencyOff ? "on" : "off");
        return;
    }
    const vk::SemaphoreTypeCreateInfo timeline{.semaphoreType = vk::SemaphoreType::eTimeline,
                                               .initialValue = 0};
    sleep_semaphore = Check<"create the low latency semaphore">(
        device.createSemaphoreUnique({.pNext = &timeline}));
    std::printf("Low latency: NVIDIA Reflex (VK_NV_low_latency2) available, %s\n",
                BbSettings::LowLatencyName(BbSettings::EffectiveLowLatency()));
}

void LowLatency::SwapchainCreated(vk::SwapchainKHR created) {
    if (backend != Backend::Reflex) {
        return;
    }
    std::scoped_lock lock{mutex};
    swapchain = created;
    applied_mode = -1; // the mode is per swapchain
}

void LowLatency::SwapchainDestroying(vk::SwapchainKHR destroying) {
    if (backend != Backend::Reflex) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (swapchain != destroying) {
        return;
    }
    // A sleep the game's thread asked for still waits on the semaphore: the driver signals it
    // for this swapchain, so it must be done before the swapchain goes.
    if (sleep_value) {
        const vk::SemaphoreWaitInfo wait{.semaphoreCount = 1,
                                         .pSemaphores = &*sleep_semaphore,
                                         .pValues = &sleep_value};
        (void)device.waitSemaphores(wait, 100'000'000ull);
    }
    swapchain = nullptr;
    applied_mode = -1;
}

void LowLatency::ApplySettings() {
    if (!available) {
        return;
    }
    // The setting, but at least On with DLSS Frame Generation (BbSettings::EffectiveLowLatency).
    const int mode = std::clamp(BbSettings::EffectiveLowLatency(), 0, 2);
    if (backend == Backend::AntiLag) {
        // Boost is Reflex's: Anti-Lag 2 is on or off. Turned off, the driver forgets its frames.
        // Off while generated frames are shown: Mesa's layer (26.2) takes a frame slot at every present and
        // frees them only at the input stage, once per game frame; two presents a frame fill its
        // ring and it crashes on the null slot (anti_lag_QueuePresentKHR, signal 11).
        const int on = mode != 0 && !generated_presents.load(std::memory_order_relaxed);
        if (anti_lag_mode.exchange(on, std::memory_order_relaxed) == on) {
            return;
        }
        if (!on) {
            const vk::AntiLagDataAMD data{.mode = vk::AntiLagModeAMD::eOff};
            device.antiLagUpdateAMD(data);
        }
        std::printf("Low latency: AMD Anti-Lag 2 %s\n", on ? "on" : "off");
        return;
    }
    std::scoped_lock lock{mutex};
    if (!swapchain || mode == applied_mode) {
        return;
    }
    const vk::LatencySleepModeInfoNV info{
        .lowLatencyMode = mode != 0,
        .lowLatencyBoost = mode == 2,
        .minimumIntervalUs = 0,
    };
    const vk::Result result = device.setLatencySleepModeNV(swapchain, &info);
    if (result != vk::Result::eSuccess) {
        if (!told.exchange(true)) {
            std::printf("Low latency: the driver refused the sleep mode (%s)\n",
                        vk::to_string(result).c_str());
        }
        return;
    }
    applied_mode = mode;
    std::printf("Low latency: %s\n", BbSettings::LowLatencyName(mode));
}

void LowLatency::AntiLagStage(vk::AntiLagStageAMD stage, u64 id) {
    if (anti_lag_mode.load(std::memory_order_relaxed) != 1) {
        return;
    }
    const vk::AntiLagPresentationInfoAMD presentation{.stage = stage, .frameIndex = id};
    const vk::AntiLagDataAMD data{
        .mode = vk::AntiLagModeAMD::eOn, .maxFPS = 0, .pPresentationInfo = &presentation};
    if (stage != vk::AntiLagStageAMD::eInput) {
        device.antiLagUpdateAMD(data);
        return;
    }
    // The input stage is where the driver sleeps: how long, per frame, every 10 s in the log.
    const auto start = std::chrono::steady_clock::now();
    device.antiLagUpdateAMD(data);
    const auto now = std::chrono::steady_clock::now();
    slept_ns += u64(std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
    ++slept_frames;
    if (now - slept_report >= std::chrono::seconds(10)) {
        if (slept_report != std::chrono::steady_clock::time_point{}) {
            std::printf("Low latency: Anti-Lag 2 held the game's thread %.2f ms a frame (%llu "
                        "frames)\n",
                        slept_ns / 1e6 / double(slept_frames), (unsigned long long)slept_frames);
        }
        slept_report = now;
        slept_ns = slept_frames = 0;
    }
}

void LowLatency::Marker(vk::LatencyMarkerNV marker, u64 id) {
    if (backend != Backend::Reflex) {
        return;
    }
    std::scoped_lock lock{mutex};
    if (!swapchain || applied_mode <= 0) {
        return;
    }
    const vk::SetLatencyMarkerInfoNV info{.presentID = id, .marker = marker};
    device.setLatencyMarkerNV(swapchain, &info);
}

void LowLatency::Sleep() {
    if (backend == Backend::AntiLag) {
        // The driver sleeps in the input stage (the frame's ID: the simulation's).
        AntiLagStage(vk::AntiLagStageAMD::eInput, sim_id.load(std::memory_order_relaxed) + 1);
        return;
    }
    u64 value;
    {
        std::scoped_lock lock{mutex};
        if (!swapchain || applied_mode <= 0) {
            return;
        }
        value = ++sleep_value;
        const vk::LatencySleepInfoNV info{.signalSemaphore = *sleep_semaphore, .value = value};
        if (device.latencySleepNV(swapchain, &info) != vk::Result::eSuccess) {
            --sleep_value;
            return;
        }
    }
    // Outside the lock: the present thread goes on meanwhile. Bounded: a stuck driver must not
    // hold the game.
    const vk::SemaphoreWaitInfo wait{.semaphoreCount = 1,
                                     .pSemaphores = &*sleep_semaphore,
                                     .pValues = &value};
    (void)device.waitSemaphores(wait, 50'000'000ull);
}

void LowLatency::GameFrameStart() {
    if (!available) {
        return;
    }
    frame_start_hooked.store(true, std::memory_order_relaxed);
    Sleep();
    const u64 id = sim_id.fetch_add(1, std::memory_order_relaxed) + 1;
    Marker(vk::LatencyMarkerNV::eSimulationStart, id);
}

void LowLatency::GameFrameEnd() {
    if (!available) {
        return;
    }
    Marker(vk::LatencyMarkerNV::eSimulationEnd, sim_id.load(std::memory_order_relaxed));
    if (!frame_start_hooked.load(std::memory_order_relaxed)) {
        // No pacer hook (the game's own 30 FPS): the next frame begins right after the flip.
        Sleep();
        const u64 id = sim_id.fetch_add(1, std::memory_order_relaxed) + 1;
        Marker(vk::LatencyMarkerNV::eSimulationStart, id);
    }
}

void LowLatency::RenderSubmitStart() {
    if (!available || render_marked.exchange(true, std::memory_order_relaxed)) {
        return;
    }
    Marker(vk::LatencyMarkerNV::eRendersubmitStart, render_id.load(std::memory_order_relaxed));
}

u64 LowLatency::RenderSubmitEnd() {
    if (!available) {
        return 0;
    }
    const u64 id = render_id.load(std::memory_order_relaxed);
    if (!render_marked.exchange(false, std::memory_order_relaxed)) {
        Marker(vk::LatencyMarkerNV::eRendersubmitStart, id); // a frame of one submission
    }
    Marker(vk::LatencyMarkerNV::eRendersubmitEnd, id);
    render_id.store(id + 1, std::memory_order_relaxed);
    return id;
}

void LowLatency::PresentStart(u64 id, bool generated) {
    if (!available || id == 0) {
        return;
    }
    if (backend == Backend::AntiLag) {
        if (!generated) {
            AntiLagStage(vk::AntiLagStageAMD::ePresent, id);
        }
        return;
    }
    Marker(generated ? vk::LatencyMarkerNV::eOutOfBandPresentStart
                     : vk::LatencyMarkerNV::ePresentStart,
           id);
}

void LowLatency::PresentEnd(u64 id, bool generated) {
    if (!available || id == 0) {
        return;
    }
    Marker(generated ? vk::LatencyMarkerNV::eOutOfBandPresentEnd : vk::LatencyMarkerNV::ePresentEnd,
           id);
}

} // namespace Vulkan
