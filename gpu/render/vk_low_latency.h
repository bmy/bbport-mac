// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: low latency. The driver paces the CPU so that frames do not queue up in front of
// the GPU: the game's thread sleeps at the start of each of its frames until the driver wants the
// frame started. Two ways to it:
// - NVIDIA Reflex through VK_NV_low_latency2, the Vulkan form of Reflex in NVIDIA's driver (what
//   Proton gives Windows games): vkLatencySleepNV, and the frame's simulation, render
//   submissions and present marked with one ID so the driver can measure them.
// - AMD Anti-Lag 2 through VK_AMD_anti_lag: AMD's driver on Windows, Mesa's anti-lag layer on
//   Linux (any Mesa driver: it times the submissions itself and sleeps in vkAntiLagUpdateAMD at
//   the input stage). The frame's start is the input stage, its present the present stage.
// Off on GPUs and drivers without either (nothing changes).
// Settings: low_latency in bbport.ini (off, on, boost; boost is Reflex's only), the overlay menu,
// BB_LOW_LATENCY; BB_ANTI_LAG=0 leaves Anti-Lag 2 out.
#pragma once

#include <atomic>
#include <chrono>
#include <mutex>

#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

class Instance;

class LowLatency {
public:
    static LowLatency& Get();

    enum class Backend { None, Reflex, AntiLag };

    /// After the device exists: Reflex when VK_NV_low_latency2 (and VK_KHR_present_id) are
    /// enabled, else Anti-Lag 2 when VK_AMD_anti_lag is.
    void Initialize(const Instance& instance);
    [[nodiscard]] bool Available() const {
        return available;
    }
    [[nodiscard]] Backend GetBackend() const {
        return backend;
    }

    /// The swapchain's lifetime (the present thread). The driver's latency calls need no
    /// external synchronization of the swapchain (the registry): only this object's own handle
    /// is guarded, so the game's thread never waits behind a present.
    void SwapchainCreated(vk::SwapchainKHR swapchain);
    void SwapchainDestroying(vk::SwapchainKHR swapchain);
    /// The settings' mode to the driver (the present thread, each present: cheap when unchanged).
    void ApplySettings();
    /// The presenter: whether this frame also shows generated frames (Anti-Lag 2 is off then).
    void SetGeneratedPresents(bool on) {
        generated_presents.store(on, std::memory_order_relaxed);
    }

    /// The game's thread at the start of its frame (the frame-rate hooks' pacer step): the
    /// driver's sleep when low latency is on, then the simulation start marker.
    void GameFrameStart();
    /// The game's thread handed its frame to the GPU (the flip): the simulation end marker.
    /// Without the frame-start hook (30 FPS mode), the next frame starts here too.
    void GameFrameEnd();

    /// The ID of the frame being drawn (the command processor's submissions carry it).
    [[nodiscard]] u64 RenderPresentId() const {
        return render_id.load(std::memory_order_relaxed);
    }
    /// The frame's first submission (marked once per frame).
    void RenderSubmitStart();
    /// The frame's last submission (the presenter's encode): returns the frame's ID and moves on.
    u64 RenderSubmitEnd();
    /// The present of frame `id` (0: a blank frame, unmarked); `generated`: a frame generation
    /// frame shown before it (out of band for the driver).
    void PresentStart(u64 id, bool generated);
    void PresentEnd(u64 id, bool generated);

private:
    void Marker(vk::LatencyMarkerNV marker, u64 id);
    void Sleep();
    /// Anti-Lag 2: the input stage (sleeps there) or the present stage of frame `id`.
    void AntiLagStage(vk::AntiLagStageAMD stage, u64 id);

    vk::Device device{};
    bool available = false;
    Backend backend = Backend::None;
    std::atomic<int> anti_lag_mode{-1}; ///< Anti-Lag 2: the mode given to the driver
    std::atomic<bool> generated_presents{false};
    /// Anti-Lag 2's sleeps (the game's thread only): reported every 10 s.
    u64 slept_ns = 0, slept_frames = 0;
    std::chrono::steady_clock::time_point slept_report{};
    mutable std::mutex mutex; ///< the swapchain handle and the driver calls on it
    vk::SwapchainKHR swapchain{};
    vk::UniqueSemaphore sleep_semaphore; ///< timeline: the driver signals it after the sleep
    u64 sleep_value = 0;
    int applied_mode = -1; ///< BbSettings low_latency given to the driver for this swapchain

    std::atomic<u64> sim_id{0};
    std::atomic<u64> render_id{1};
    std::atomic<bool> render_marked{false};
    std::atomic<bool> frame_start_hooked{false};
    std::atomic<bool> told{false};
};

} // namespace Vulkan
