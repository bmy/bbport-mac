// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the shaders and pipelines compiled during the session, the newest kept, for the
// overlay's "Shaders" section (the idea of the SrShadowy fork's live compile log). Compiles are
// few (hundreds in a session): a mutex is enough.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <vector>

namespace BbCompileLog {

enum class Kind : std::uint8_t { Shader, Graphics, Compute, Background };

struct Event {
    std::uint64_t hash = 0;
    std::chrono::steady_clock::time_point when{};
    float ms = 0.0f;
    Kind kind = Kind::Shader;
    std::uint8_t stage = 0; ///< Shader::HwStage of a shader
    std::uint16_t permutation = 0;
};

namespace Detail {
inline std::mutex mutex;
inline std::array<Event, 256> ring{};
inline std::uint64_t count = 0;
} // namespace Detail

inline void Record(Kind kind, std::uint64_t hash, std::chrono::steady_clock::time_point start,
                   std::uint8_t stage = 0, std::uint16_t permutation = 0) {
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock{Detail::mutex};
    Detail::ring[Detail::count++ % Detail::ring.size()] = {
        .hash = hash,
        .when = now,
        .ms = std::chrono::duration<float, std::milli>(now - start).count(),
        .kind = kind,
        .stage = stage,
        .permutation = permutation,
    };
}

/// The newest `max` events, newest first; `total`: every one recorded.
inline std::vector<Event> Newest(std::size_t max, std::uint64_t& total) {
    std::scoped_lock lock{Detail::mutex};
    total = Detail::count;
    const std::size_t n = std::min<std::size_t>({max, Detail::count, Detail::ring.size()});
    std::vector<Event> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        out.push_back(Detail::ring[(Detail::count - 1 - i) % Detail::ring.size()]);
    }
    return out;
}

} // namespace BbCompileLog
