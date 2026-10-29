// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "upscale/fg_profile.h"
#include "shim/bb_env.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "vulkan/instance.h"

extern "C" void (*ffxVkJobMark)(uint32_t x, uint32_t y, uint32_t z);
#ifdef __APPLE__
// bbport: as ffxVkShaderOverride (ffx_shaders.cpp): no 0.7 patch defines it in the FFX backend.
extern "C" {
void (*ffxVkJobMark)(uint32_t x, uint32_t y, uint32_t z) = nullptr;
}
#endif

namespace FgProfile {

namespace {

constexpr u32 Slots = 4, PerSlot = 64;
struct State {
    vk::Device device{};
    vk::QueryPool pool{};
    double period_ns = 1.0;
    u32 slot = 0;
    std::vector<std::string> names[Slots];
    bool pending[Slots]{};
    vk::CommandBuffer cmd{};
    std::map<std::string, double> total_ms;
    double frame_total_ms = 0;
    u32 frames = 0;
} state;

void Mark(const char* name) {
    auto& names = state.names[state.slot];
    if (!state.cmd || names.size() + 1 >= PerSlot) {
        return;
    }
    state.cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, state.pool,
                              state.slot * PerSlot + u32(names.size()));
    names.emplace_back(name ? name : "?");
}

void JobMark(uint32_t x, uint32_t y, uint32_t z) {
    char name[64];
    std::snprintf(name, sizeof(name), "job %02zu (%ux%ux%u)", state.names[state.slot].size(), x, y, z);
    Mark(name);
}

void Collect(u32 slot) {
    auto& names = state.names[slot];
    std::vector<u64> ticks(names.size());
    if (names.size() < 2 ||
        state.device.getQueryPoolResults(state.pool, slot * PerSlot, u32(names.size()),
                                         ticks.size() * sizeof(u64), ticks.data(), sizeof(u64),
                                         vk::QueryResultFlagBits::e64) != vk::Result::eSuccess) {
        return;
    }
    for (size_t i = 0; i + 1 < names.size(); ++i) {
        const double ms = double(ticks[i + 1] - ticks[i]) * state.period_ns / 1e6;
        state.total_ms[names[i]] += ms;
    }
    state.frame_total_ms += double(ticks.back() - ticks.front()) * state.period_ns / 1e6;
    if (++state.frames == 300) {
        std::printf("FG profile: %.3f ms/frame over %u frames\n", state.frame_total_ms / state.frames,
                    state.frames);
        for (const auto& [name, ms] : state.total_ms) {
            std::printf("  %7.3f ms  %s\n", ms / state.frames, name.c_str());
        }
        state.total_ms.clear();
        state.frame_total_ms = 0;
        state.frames = 0;
    }
}

} // namespace

bool Enabled() {
    static const bool on = bb_env_on("BB_FG_PROFILE");
    return on;
}

void Begin(const Vulkan::Instance& instance, vk::CommandBuffer cmd) {
    if (!Enabled()) {
        return;
    }
    if (!state.pool) {
        state.device = instance.GetDevice();
        state.period_ns = instance.GetProperties().limits.timestampPeriod;
        auto [result, pool] = state.device.createQueryPool(
            {.queryType = vk::QueryType::eTimestamp, .queryCount = Slots * PerSlot});
        if (result != vk::Result::eSuccess) {
            return;
        }
        state.pool = pool;
        ffxVkJobMark = &JobMark;
    }
    // The slot used Slots frames ago is done by now (frames in flight are fewer).
    state.slot = (state.slot + 1) % Slots;
    if (state.pending[state.slot]) {
        Collect(state.slot);
    }
    state.names[state.slot].clear();
    cmd.resetQueryPool(state.pool, state.slot * PerSlot, PerSlot);
    state.pending[state.slot] = true;
    state.cmd = cmd;
    Mark("(start)");
}

void End(vk::CommandBuffer cmd) {
    if (!Enabled() || !state.pool) {
        return;
    }
    Mark("(end)");
    state.cmd = vk::CommandBuffer{};
    (void)cmd;
}

} // namespace FgProfile
