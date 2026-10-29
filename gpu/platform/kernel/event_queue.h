// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: kernel event queues (kqueue/kevent and the sceKernel*Equeue calls). Games wait on them
// for GPU end-of-pipe events (Gnm), flips and vblanks (VideoOut), their own user events and
// timers. Each queue keeps its events in a small array under its own lock; waiting threads sleep
// on a condition variable; timers run on one timer thread.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

#include "common/types.h"

namespace Core::Loader {
class SymbolsResolver;
}

namespace BbKernel {

using EqueueHandle = s64;

/// struct kevent as PS4 games see it.
struct Event {
    enum Filter : s16 {
        None = 0,
        Read = -1,
        Write = -2,
        Aio = -3,
        Vnode = -4,
        Proc = -5,
        Signal = -6,
        Timer = -7,
        Fs = -9,
        Lio = -10,
        User = -11,
        Polling = -12,
        VideoOut = -13,
        GraphicsCore = -14,
        HrTimer = -15,
    };
    enum Flags : u16 {
        Add = 0x1,
        Delete = 0x2,
        Enable = 0x4,
        Disable = 0x8,
        OneShot = 0x10,
        Clear = 0x20, ///< untriggered (data and fflags reset) once delivered
        Receipt = 0x40,
        Dispatch = 0x80,
    };

    u64 ident = 0;
    Filter filter = None;
    u16 flags = 0;
    u32 fflags = 0;
    u64 data = 0;
    void* udata = nullptr;
};
static_assert(sizeof(Event) == 32);

class EventQueue {
public:
    EventQueue(EqueueHandle handle, std::string name);

    const std::string& Name() const {
        return name;
    }

    /// Adds `event` (ident and filter name it); one already there keeps its state and gets the
    /// new user data (and timer period). `period`: timers fire every `period` (once with OneShot).
    void Add(const Event& event, std::chrono::nanoseconds period = {});
    bool Remove(u64 ident, s16 filter);
    bool Contains(u64 ident, s16 filter);
    /// Marks the event triggered; what `data` means depends on the filter (VideoOut: the flip or
    /// vblank hint, User: new user data, GraphicsCore: the event type).
    bool Trigger(u64 ident, s16 filter, void* data);
    /// Up to `max` triggered events; `timeout_us` null: waits for one, 0: does not wait.
    int Wait(Event* out, int max, const u32* timeout_us);

    // Wait statistics (frame stats report).
    std::atomic<u64> waits{0};
    std::atomic<u64> wait_ns{0};
    std::atomic<u64> delivered{0};

private:
    struct Entry {
        Event event;
        bool triggered = false;
        std::chrono::nanoseconds period{};
        u64 timer_generation = 0; ///< a timer armed for an older generation does nothing
    };

    int TakeTriggered(Event* out, int max);
    Entry* Find(u64 ident, s16 filter);
    void ArmTimer(Entry& entry);
    friend void FireTimer(EqueueHandle, u64, s16, u64, std::chrono::steady_clock::time_point);

    EqueueHandle handle;
    std::string name;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<Entry> entries;
};

/// The queue with this handle, or null (never freed: deleted queues stay valid objects).
EventQueue* FindQueue(EqueueHandle handle);

/// The event's data word (sceKernelGetEventData: for Gnm events, the event type).
u64 PS4_SYSV_ABI sceKernelGetEventData(const Event* event);

void RegisterEventQueue(Core::Loader::SymbolsResolver* sym);
/// Frame stats: per-frame waits of the busiest queues since the last report.
void ReportEqueueWaits(double frames);

} // namespace BbKernel
