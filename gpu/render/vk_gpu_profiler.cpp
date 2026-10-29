// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "shim/bb_env.h"
#include <array>
#include <cstdlib>

#include "render/vk_gpu_profiler.h"
#include "vulkan/instance.h"
#include "render/scheduler.h"

namespace Vulkan {

void GpuProfiler::Init(const Instance& instance, Scheduler& scheduler) {
    const char* env = bb_env("BB_GPU_PROFILE");
    if (!env || env[0] != '1' || instance_ptr) {
        return;
    }
    instance_ptr = new GpuProfiler(instance, scheduler);
}

GpuProfiler::GpuProfiler(const Instance& instance, Scheduler& scheduler_)
    : device{instance.GetDevice()}, scheduler{scheduler_} {
    const vk::QueryPoolCreateInfo info = {
        .queryType = vk::QueryType::eTimestamp,
        .queryCount = NumSlices * SliceQueries,
    };
    pool = Check<"create timestamp query pool">(device.createQueryPoolUnique(info));
    device.resetQueryPool(*pool, 0, NumSlices * SliceQueries);
    period_ns = instance.GetPhysicalDevice().getProperties().limits.timestampPeriod;
    for (auto& k : keys) {
        k.reserve(SliceQueries);
    }
    std::printf("GPU profile: on (timestamp period %.2f ns)\n", period_ns);
}

void GpuProfiler::WriteTimestamp(u64 key, bool end_rendering) {
    if (used[slice] + 1 >= SliceQueries) {
        return; // the frame's slice is full: the rest of the frame goes to the last label
    }
    // Outside render passes: radv_CmdWriteTimestamp2 crashed inside some. Marks sit where a
    // pass, dispatch or submission ends anyway.
    if (end_rendering) {
        scheduler.EndRendering();
    }
    const u32 query = slice * SliceQueries + used[slice]++;
    keys[slice].push_back(key);
    current = key;
    scheduler.Record([pool = *pool, query](vk::CommandBuffer cmdbuf) {
        cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, query);
    });
}

void GpuProfiler::BeginFrame() {
    // Close the frame: one more timestamp without a label.
    if (used[slice] > 0 && used[slice] < SliceQueries) {
        scheduler.EndRendering();
        const u32 query = slice * SliceQueries + used[slice]++;
        scheduler.Record([pool = *pool, query](vk::CommandBuffer cmdbuf) {
            cmdbuf.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, pool, query);
        });
        pending[slice] = true;
    }
    slice = (slice + 1) % NumSlices;
    // The oldest slice: its frame was submitted four frames ago.
    if (pending[slice]) {
        Collect(slice);
    }
    if (used[slice]) {
        device.resetQueryPool(*pool, slice * SliceQueries, used[slice]);
    }
    used[slice] = 0;
    keys[slice].clear();
    pending[slice] = false;
    Print();
}

void GpuProfiler::Collect(u32 which) {
    const u32 count = used[which];
    std::vector<u64> stamps(count);
    // Four frames on this is complete unless the GPU lags that far: then wait for it.
    const auto result = device.getQueryPoolResults(
        *pool, which * SliceQueries, count, count * sizeof(u64), stamps.data(), sizeof(u64),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
    if (result != vk::Result::eSuccess) {
        return;
    }
    for (u32 i = 0; i + 1 < count; ++i) {
        const double ms = double(stamps[i + 1] - stamps[i]) * period_ns / 1e6;
        auto& total = totals[keys[which][i]];
        total.ms += ms;
        ++total.segments;
        if (keys[which][i] >= SubmissionGapKey && keys[which][i] < SubmissionGapKey + 0x20) {
            gaps.push_back(ms);
        }
    }
    ++frames;
}

void GpuProfiler::Print() {
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds(5) || frames == 0) {
        return;
    }
    window = now;
    std::vector<std::pair<u64, Total>> sorted(totals.begin(), totals.end());
    std::ranges::sort(sorted, [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    double sum = 0;
    for (const auto& [key, total] : sorted) {
        sum += total.ms;
    }
    std::printf("GPU profile: %.2f ms/frame over %llu frames, %zu labels\n", sum / frames,
                static_cast<unsigned long long>(frames), sorted.size());
    // BB_GPU_PROFILE_TOP=n: the n heaviest labels (30).
    static const size_t top = [] {
        const char* env = bb_env("BB_GPU_PROFILE_TOP");
        return env && *env ? size_t(std::strtoul(env, nullptr, 10)) : size_t(30);
    }();
    for (size_t i = 0; i < std::min<size_t>(sorted.size(), top); ++i) {
        const auto& [key, total] = sorted[i];
        std::printf("  %6.3f ms/frame %5.1f/frame  %s\n", total.ms / frames,
                    double(total.segments) / frames, described[key].c_str());
    }
    // BB_GPU_PROFILE_DRAWS=1: all labelled draws together, by what one draw of the pair costs.
    {
        struct Bin { double ms = 0; u64 draws = 0; u64 pairs = 0; };
        std::array<Bin, 5> bins{}; // < 5 us, < 20, < 50, < 150, more
        for (const auto& [key, total] : sorted) {
            if (described[key].rfind("draw vs", 0) != 0 || total.segments == 0) {
                continue;
            }
            const double us = total.ms * 1000.0 / double(total.segments);
            auto& bin = bins[us < 5 ? 0 : us < 20 ? 1 : us < 50 ? 2 : us < 150 ? 3 : 4];
            bin.ms += total.ms;
            bin.draws += total.segments;
            ++bin.pairs;
        }
        if (bins[0].draws + bins[1].draws + bins[2].draws + bins[3].draws + bins[4].draws != 0) {
            static constexpr const char* names[5] = {"< 5 us", "5-20 us", "20-50 us", "50-150 us", "> 150 us"};
            std::printf("  draws by cost of one:");
            for (size_t b = 0; b < bins.size(); ++b) {
                std::printf("  %s: %.2f ms %.0f draws (%llu pairs)", names[b], bins[b].ms / frames,
                            double(bins[b].draws) / frames, (unsigned long long)bins[b].pairs);
            }
            std::printf("\n");
        }
    }
    if (!gaps.empty()) {
        // The gaps one by one: the presenter's own submissions (outside this scheduler) fall in
        // them too, so a steady gap per frame is its work, scattered long ones are waits.
        std::ranges::sort(gaps);
        const auto at = [&](double q) { return gaps[size_t(q * double(gaps.size() - 1))]; };
        double over = 0;
        for (const double gap : gaps) {
            over += gap >= 0.25 ? gap : 0;
        }
        std::printf("  submission gaps: %zu, median %.3f ms, p90 %.3f, max %.3f; %.3f ms/frame "
                    "in gaps of 0.25 ms or more\n",
                    gaps.size(), at(0.5), at(0.9), gaps.back(), over / frames);
        gaps.clear();
    }
    totals.clear();
    frames = 0;
}

} // namespace Vulkan
