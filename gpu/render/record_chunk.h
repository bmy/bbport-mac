// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: blocks of deferred Vulkan commands for the recording threads (RecordChunk),
// and the command buffer segments of one submission (SubmissionBatch).
#pragma once

#include <array>
#include <cstddef>
#include <new>
#include <utility>

#include "bbport_toggles.h"
#include "common/types.h"
#include "vulkan/vk_common.h"

namespace Vulkan {

/// The command buffer segments of one submission and their chunk counters (guarded by the
/// scheduler's recorder_mutex). With BB_ASYNC_SUBMIT several are in flight: a submission goes
/// out from a recording thread once its segments are recorded.
struct SubmissionBatch {
    static constexpr u32 MaxSegments = 64;
    std::array<vk::CommandBuffer, MaxSegments> segments{}; ///< null until begun
    std::array<u32, MaxSegments> handed{}, recorded{};
    void Reset() {
        segments.fill({});
        handed.fill(0);
        recorded.fill(0);
    }
};

/// A block of deferred Vulkan commands. Commands are closures placed in
/// fixed storage (no allocation per command) and run in order on the recording thread.
class RecordChunk {
public:
    static constexpr size_t Capacity = 128 * 1024;

    /// Returns false (and leaves `func` untouched) when the chunk has no room.
    template <typename Func>
    bool Push(Func&& func) {
        using Command = TypedCommand<std::decay_t<Func>>;
        static_assert(sizeof(Command) <= Capacity, "recorded command is too large");
        const size_t offset = (used + alignof(Command) - 1) & ~(alignof(Command) - 1);
        if (offset + sizeof(Command) > Capacity) {
            return false;
        }
        auto* command = new (storage + offset) Command(std::forward<Func>(func));
        if (last) {
            last->next = command;
        } else {
            first = command;
        }
        last = command;
        used = offset + sizeof(Command);
        PrefetchAhead();
        return true;
    }

    /// Raw storage in the chunk for a command's variable-length data; null when full.
    void* Allocate(size_t bytes, size_t align) {
        const size_t offset = (used + align - 1) & ~(align - 1);
        if (offset + bytes > Capacity) {
            return nullptr;
        }
        used = offset + bytes;
        PrefetchAhead();
        return storage + offset;
    }

    void Execute(vk::CommandBuffer cmdbuf) {
        for (CommandBase* command = first; command;) {
            CommandBase* const next = command->next;
            command->Execute(cmdbuf);
            command->~CommandBase();
            command = next;
        }
        first = last = nullptr;
        used = 0;
    }

    [[nodiscard]] bool Empty() const noexcept {
        return first == nullptr;
    }

    [[nodiscard]] size_t Size() const noexcept {
        return used;
    }

    u32 segment = 0; ///< the command buffer segment the chunk is recorded into (Scheduler)
    SubmissionBatch* batch = nullptr; ///< the submission it belongs to (Scheduler)
    /// Ordered chunks: the command chunks of `segment` (and all of the segments before it)
    /// that must be recorded before this chunk runs.
    u32 after = 0;

private:
    /// The recording thread last read these cache lines: the first write to each has to take
    /// it back from that core. Requesting ownership a few lines ahead overlaps those
    /// transfers instead of stalling on each (RecordPrefetch).
    void PrefetchAhead() const {
        if (!BbToggle::Disabled(BbToggle::RecordPrefetch)) {
            __builtin_prefetch(storage + used + 384, 1, 3);
            __builtin_prefetch(storage + used + 448, 1, 3);
        }
    }

    struct CommandBase {
        virtual ~CommandBase() = default;
        virtual void Execute(vk::CommandBuffer cmdbuf) = 0;
        CommandBase* next{};
    };
    template <typename Func>
    struct TypedCommand final : CommandBase {
        explicit TypedCommand(Func&& func_) : func{std::move(func_)} {}
        explicit TypedCommand(const Func& func_) : func{func_} {}
        void Execute(vk::CommandBuffer cmdbuf) override {
            func(cmdbuf);
        }
        Func func;
    };

    alignas(64) std::byte storage[Capacity];
    size_t used = 0;
    CommandBase* first{};
    CommandBase* last{};
};

} // namespace Vulkan
