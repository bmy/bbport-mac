// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <array>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

#include "common/assert.h"
#include "platform/interrupts.h"

namespace BbIrq {
namespace {

struct LineState {
    std::mutex mutex;
    std::vector<std::pair<const void*, Handler>> listeners;
    std::deque<Handler> once;
};

/// One state per line, made up front: no lookups, no allocation on raise.
LineState& State(Line line) {
    static std::array<LineState, LineCount> lines;
    ASSERT_MSG(u32(line) < LineCount, "bad interrupt line {}", u32(line));
    return lines[u32(line)];
}

} // namespace

void Listen(Line line, const void* owner, Handler handler) {
    auto& state = State(line);
    std::scoped_lock lock{state.mutex};
    for (const auto& [listener, _] : state.listeners) {
        ASSERT_MSG(listener != owner, "interrupt {} already has this listener", u32(line));
    }
    state.listeners.emplace_back(owner, std::move(handler));
}

void Forget(Line line, const void* owner) {
    auto& state = State(line);
    std::scoped_lock lock{state.mutex};
    std::erase_if(state.listeners, [owner](const auto& entry) { return entry.first == owner; });
}

void Once(Line line, Handler handler) {
    auto& state = State(line);
    std::scoped_lock lock{state.mutex};
    state.once.push_back(std::move(handler));
}

void Raise(Line line) {
    auto& state = State(line);
    // Handlers run under the line's lock: a listener cannot be forgotten while it runs.
    std::scoped_lock lock{state.mutex};
    for (const auto& [_, handler] : state.listeners) {
        handler(line);
    }
    if (!state.once.empty()) {
        const Handler handler = std::move(state.once.front());
        state.once.pop_front();
        handler(line);
    }
    if (forward) {
        forward(line);
    }
}

} // namespace BbIrq
