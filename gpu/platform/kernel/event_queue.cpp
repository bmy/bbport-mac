// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <queue>
#include <thread>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/rdtsc.h"
#include "common/thread.h"
#include "core/libraries/libs.h"
#include "platform/errors.h"
#include "platform/kernel.h"
#include "platform/kernel/event_queue.h"

namespace BbKernel {

namespace {

// ---- Queue handles ---------------------------------------------------------------------------
// Handles count up from 0x40000 (outside the port's file descriptors); a slot per handle, so a
// lookup is one atomic load.

constexpr EqueueHandle FirstHandle = 0x40000;
constexpr std::size_t MaxQueues = 4096;
std::array<std::atomic<EventQueue*>, MaxQueues> queues{};
std::atomic<std::size_t> queue_count{0};

/// `name` null: "kqueue<handle>".
EqueueHandle CreateQueue(const char* name) {
    const std::size_t index = queue_count.fetch_add(1, std::memory_order_relaxed);
    ASSERT_MSG(index < MaxQueues, "too many event queues");
    const EqueueHandle handle = FirstHandle + EqueueHandle(index);
    const std::string queue_name = name ? name : "kqueue" + std::to_string(handle);
    queues[index].store(new EventQueue(handle, queue_name), std::memory_order_release);
    return handle;
}

/// Display events: the data word is the flip or vblank hint the trigger brings (its low 16 bits
/// replaced by the time and a count of events since delivered).
u64 DisplayData(u64 ident, u64 data, u64 hint) {
    const u64 hint_event = hint & 0xFF;
    if (hint_event != ident || ident == 0xFE) {
        return data;
    }
    u64 count = (data >> 12) & 0xF;
    if (count != 0xF) {
        ++count;
    }
    return (Common::FencedRDTSC() & 0xFFF) | (count << 12) | (hint & 0xFFFF'FFFF'FFFF'0000ull);
}

/// A bintime (seconds, 2^-64 fractions) as kevent's HR timer data points at, in nanoseconds.
std::chrono::nanoseconds BintimeToNs(const void* bintime) {
    s64 parts[2];
    std::memcpy(parts, bintime, sizeof(parts));
    const u64 frac_ns = (1'000'000'000ull * (u64(parts[1]) >> 32)) >> 32;
    return std::chrono::nanoseconds(parts[0] * 1'000'000'000ll + s64(frac_ns));
}

} // namespace
void FireTimer(EqueueHandle handle, u64 ident, s16 filter, u64 generation,
               std::chrono::steady_clock::time_point deadline);
namespace {

// ---- Timers ----------------------------------------------------------------------------------

struct TimerItem {
    std::chrono::steady_clock::time_point at;
    EqueueHandle queue;
    u64 ident;
    s16 filter;
    u64 generation;
    bool operator>(const TimerItem& other) const {
        return at > other.at;
    }
};

std::atomic<u64> timer_generations{0};

/// One thread for every queue's timers: sleeps until the earliest deadline (the last stretch of
/// short waits spun, for HR timer precision).
class Timers {
public:
    void Arm(const TimerItem& item) {
        {
            std::scoped_lock lock{mutex};
            if (!thread.joinable()) {
                thread = std::jthread([this](std::stop_token stop) { Run(stop); });
            }
            items.push(item);
        }
        cv.notify_one();
    }

private:
    void Run(std::stop_token stop) {
        Common::SetCurrentThreadName("bb:Timers");
        constexpr auto Spin = std::chrono::microseconds(150);
        std::unique_lock lock{mutex};
        while (!stop.stop_requested()) {
            if (items.empty()) {
                cv.wait(lock, stop, [this] { return !items.empty(); });
                continue;
            }
            const auto at = items.top().at;
            if (std::chrono::steady_clock::now() < at - Spin) {
                Common::WaitUntil(cv, lock, at - Spin, [&] { return stop.stop_requested(); });
                continue; // an earlier timer may have come in
            }
            const TimerItem item = items.top();
            items.pop();
            lock.unlock();
            while (std::chrono::steady_clock::now() < item.at) {
                std::this_thread::yield();
            }
            FireTimer(item.queue, item.ident, item.filter, item.generation, item.at);
            lock.lock();
        }
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    std::priority_queue<TimerItem, std::vector<TimerItem>, std::greater<>> items;
    std::jthread thread;
};

Timers timers;

} // namespace

EventQueue* FindQueue(EqueueHandle handle) {
    const EqueueHandle index = handle - FirstHandle;
    if (index < 0 || std::size_t(index) >= MaxQueues) {
        return nullptr;
    }
    return queues[index].load(std::memory_order_acquire);
}

// ---- EventQueue ------------------------------------------------------------------------------

EventQueue::EventQueue(EqueueHandle handle_, std::string name_)
    : handle{handle_}, name{std::move(name_)} {}

EventQueue::Entry* EventQueue::Find(u64 ident, s16 filter) {
    for (auto& entry : entries) {
        if (entry.event.ident == ident && entry.event.filter == filter) {
            return &entry;
        }
    }
    return nullptr;
}

void EventQueue::Add(const Event& event, std::chrono::nanoseconds period) {
    std::scoped_lock lock{mutex};
    if (Entry* existing = Find(event.ident, event.filter)) {
        // Re-adding keeps the trigger state and data; only the user data and period change.
        existing->event.udata = event.udata;
        existing->period = period;
        return;
    }
    Entry& entry = entries.emplace_back();
    entry.event = event;
    entry.event.data = 0;
    entry.event.flags &= ~Event::Add;
    if (event.filter != Event::User) {
        entry.event.flags |= Event::Clear; // the kernel adds it to all but user events
    }
    entry.period = period;
    if (event.filter == Event::Timer || event.filter == Event::HrTimer) {
        ArmTimer(entry);
    }
}

void EventQueue::ArmTimer(Entry& entry) {
    entry.timer_generation = ++timer_generations;
    timers.Arm({std::chrono::steady_clock::now() + entry.period, handle, entry.event.ident,
                entry.event.filter, entry.timer_generation});
}

void FireTimer(EqueueHandle handle, u64 ident, s16 filter, u64 generation,
               std::chrono::steady_clock::time_point deadline) {
    EventQueue* queue = FindQueue(handle);
    if (!queue) {
        return;
    }
    {
        std::scoped_lock lock{queue->mutex};
        EventQueue::Entry* entry = queue->Find(ident, filter);
        if (!entry || entry->timer_generation != generation) {
            return; // removed (or removed and added again) since
        }
        entry->triggered = true;
        ++entry->event.data; // expirations since delivered
        // Periodic timers fire again a period after this deadline (no drift); HR timers once.
        if (filter == Event::Timer && !(entry->event.flags & Event::OneShot)) {
            entry->timer_generation = ++timer_generations;
            timers.Arm({deadline + entry->period, handle, ident, filter,
                        entry->timer_generation});
        }
    }
    queue->cv.notify_all();
}

bool EventQueue::Remove(u64 ident, s16 filter) {
    std::scoped_lock lock{mutex};
    const auto it = std::ranges::find_if(entries, [&](const Entry& entry) {
        return entry.event.ident == ident && entry.event.filter == filter;
    });
    if (it == entries.end()) {
        return false;
    }
    entries.erase(it);
    return true;
}

bool EventQueue::Contains(u64 ident, s16 filter) {
    std::scoped_lock lock{mutex};
    return Find(ident, filter) != nullptr;
}

bool EventQueue::Trigger(u64 ident, s16 filter, void* data) {
    bool found = false;
    {
        std::scoped_lock lock{mutex};
        for (auto& entry : entries) {
            if (entry.event.ident != ident || entry.event.filter != filter) {
                continue;
            }
            entry.triggered = true;
            switch (filter) {
            case Event::VideoOut:
                if (data) {
                    entry.event.data =
                        DisplayData(ident, entry.event.data, reinterpret_cast<u64>(data));
                }
                break;
            case Event::User:
                entry.event.udata = data;
                break;
            case Event::Timer:
            case Event::HrTimer:
                ++entry.event.data;
                break;
            default:
                entry.event.data = reinterpret_cast<u64>(data);
                break;
            }
            found = true;
        }
    }
    cv.notify_all();
    return found;
}

int EventQueue::TakeTriggered(Event* out, int max) {
    int count = 0;
    for (auto it = entries.begin(); it != entries.end() && count < max;) {
        if (!it->triggered) {
            ++it;
            continue;
        }
        out[count++] = it->event;
        if (it->event.flags & Event::Clear) {
            it->triggered = false;
            it->event.fflags = 0;
            it->event.data = 0;
        }
        if (it->event.flags & Event::OneShot) {
            it = entries.erase(it);
        } else {
            ++it;
        }
    }
    return count;
}

int EventQueue::Wait(Event* out, int max, const u32* timeout_us) {
    std::unique_lock lock{mutex};
    int count = 0;
    const auto ready = [&] {
        count = TakeTriggered(out, max);
        return count > 0;
    };
    if (timeout_us && *timeout_us == 0) {
        ready(); // a poll
    } else if (!timeout_us) {
        cv.wait(lock, ready);
    } else {
        Common::WaitFor(cv, lock, std::chrono::microseconds(*timeout_us), ready);
    }
    return count;
}

// ---- Exported functions ----------------------------------------------------------------------

namespace {

/// Filters the port delivers; kevent skips the others.
bool Supported(s16 filter) {
    return filter == Event::GraphicsCore || filter == Event::HrTimer || filter == Event::Timer ||
           filter == Event::User || filter == Event::VideoOut;
}

s32 PosixError(int error) {
    *Libraries::Kernel::__Error() = error;
    return BbErr::Fail;
}

struct Timespec {
    s64 tv_sec;
    s64 tv_nsec;
};

s32 PS4_SYSV_ABI posix_kqueue() {
    return s32(CreateQueue(nullptr));
}

s32 PS4_SYSV_ABI posix_kevent(s32 handle, Event* changes, u64 change_count, Event* events,
                              u64 event_count, const Timespec* timeout) {
    EventQueue* queue = FindQueue(handle);
    if (!queue) {
        return PosixError(BbErr::PosixBadFd);
    }
    for (u64 i = 0; i < change_count; i++) {
        const Event& change = changes[i];
        if (!Supported(change.filter)) {
            LOG_ERROR(Kernel_Event, "kevent: unsupported filter {}", s32(change.filter));
            continue;
        }
        if (change.flags & Event::Add) {
            std::chrono::nanoseconds period{};
            if (change.filter == Event::Timer) {
                period = std::chrono::milliseconds(change.data);
            } else if (change.filter == Event::HrTimer) {
                period = BintimeToNs(reinterpret_cast<const void*>(change.data));
            }
            queue->Add(change, period);
        }
        if ((change.flags & Event::Delete) && !queue->Remove(change.ident, change.filter)) {
            return PosixError(BbErr::PosixNoEntry);
        }
        constexpr u32 NoteTrigger = 0x1000000;
        if (change.filter == Event::User && change.fflags == NoteTrigger) {
            if (!queue->Trigger(change.ident, Event::User, change.udata)) {
                return PosixError(BbErr::PosixNoEntry);
            }
        } else if (change.fflags != 0) {
            LOG_ERROR(Kernel_Event, "kevent: filter flags {:#x} of filter {} not handled",
                      change.fflags, s32(change.filter));
        }
    }
    if (event_count == 0) {
        return 0;
    }
    if (timeout) {
        const u32 micros = u32(timeout->tv_sec * 1'000'000 + timeout->tv_nsec / 1000);
        return queue->Wait(events, int(event_count), &micros);
    }
    return queue->Wait(events, int(event_count), nullptr);
}

s32 PS4_SYSV_ABI sceKernelCreateEqueue(EqueueHandle* eq, const char* name) {
    if (!eq || !name) {
        return BbErr::KernelInvalid;
    }
    if (std::strlen(name) > 32) {
        return BbErr::KernelNameTooLong;
    }
    *eq = CreateQueue(name);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceKernelDeleteEqueue(EqueueHandle eq) {
    const EqueueHandle index = eq - FirstHandle;
    if (!FindQueue(eq)) {
        return BbErr::KernelBadFd;
    }
    // The object stays (interrupt handlers may still hold it); the handle is gone.
    queues[index].store(nullptr, std::memory_order_release);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceKernelWaitEqueue(EqueueHandle eq, Event* events, int max, int* out,
                                     u32* timeout_us) {
    EventQueue* queue = FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    if (!events) {
        return BbErr::KernelFault;
    }
    if (max < 1) {
        *out = 0;
        return BbErr::KernelInvalid;
    }
    const auto start = std::chrono::steady_clock::now();
    *out = queue->Wait(events, max, timeout_us);
    queue->waits.fetch_add(1, std::memory_order_relaxed);
    queue->wait_ns.fetch_add(u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now() - start)
                                     .count()),
                             std::memory_order_relaxed);
    queue->delivered.fetch_add(u64(*out), std::memory_order_relaxed);
    return *out == 0 ? BbErr::KernelTimedOut : BbErr::Ok;
}

s32 PS4_SYSV_ABI sceKernelAddHRTimerEvent(EqueueHandle eq, int id, const Timespec* ts, void* udata) {
    EventQueue* queue = FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    Event event{};
    event.ident = u64(id);
    event.filter = Event::HrTimer;
    event.flags = Event::Add | Event::OneShot;
    event.udata = udata;
    queue->Add(event, std::chrono::nanoseconds(ts->tv_sec * 1'000'000'000ll + ts->tv_nsec));
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceKernelDeleteHRTimerEvent(EqueueHandle eq, int id) {
    EventQueue* queue = FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    return queue->Remove(u64(id), Event::HrTimer) ? BbErr::Ok : BbErr::KernelNotFound;
}

s32 PS4_SYSV_ABI sceKernelAddTimerEvent(EqueueHandle eq, int id, u32 usec, void* udata) {
    EventQueue* queue = FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    Event event{};
    event.ident = u64(id);
    event.filter = Event::Timer;
    event.flags = Event::Add;
    event.udata = udata;
    queue->Add(event, std::chrono::milliseconds(usec / 1000));
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceKernelDeleteTimerEvent(EqueueHandle eq, int id) {
    EventQueue* queue = FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    return queue->Remove(u64(id), Event::Timer) ? BbErr::Ok : BbErr::KernelNotFound;
}

s32 AddUserEvent(EqueueHandle eq, int id, u16 flags) {
    EventQueue* queue = FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    Event event{};
    event.ident = u64(id);
    event.filter = Event::User;
    event.flags = flags;
    queue->Add(event);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceKernelAddUserEvent(EqueueHandle eq, int id) {
    return AddUserEvent(eq, id, Event::Add);
}

/// Edge-triggered: untriggered again once delivered.
s32 PS4_SYSV_ABI sceKernelAddUserEventEdge(EqueueHandle eq, int id) {
    return AddUserEvent(eq, id, Event::Add | Event::Clear);
}

s32 PS4_SYSV_ABI sceKernelTriggerUserEvent(EqueueHandle eq, int id, void* udata) {
    EventQueue* queue = FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    return queue->Trigger(u64(id), Event::User, udata) ? BbErr::Ok : BbErr::KernelNotFound;
}

s32 PS4_SYSV_ABI sceKernelDeleteUserEvent(EqueueHandle eq, int id) {
    EventQueue* queue = FindQueue(eq);
    if (!queue) {
        return BbErr::KernelBadFd;
    }
    return queue->Remove(u64(id), Event::User) ? BbErr::Ok : BbErr::KernelNotFound;
}

void* PS4_SYSV_ABI sceKernelGetEventUserData(const Event* event) {
    return event->udata;
}

u64 PS4_SYSV_ABI sceKernelGetEventId(const Event* event) {
    return event->ident;
}

s32 PS4_SYSV_ABI sceKernelGetEventFilter(const Event* event) {
    return event->filter;
}

} // namespace

u64 PS4_SYSV_ABI sceKernelGetEventData(const Event* event) {
    return event->data;
}

void ReportEqueueWaits(double frames) {
    if (frames <= 0) {
        return;
    }
    struct Stat {
        std::string name;
        u64 waits, ns, events;
    };
    std::vector<Stat> stats;
    const std::size_t count = std::min(queue_count.load(), MaxQueues);
    for (std::size_t i = 0; i < count; ++i) {
        EventQueue* queue = queues[i].load(std::memory_order_acquire);
        if (!queue) {
            continue;
        }
        const u64 waits = queue->waits.exchange(0, std::memory_order_relaxed);
        const u64 ns = queue->wait_ns.exchange(0, std::memory_order_relaxed);
        const u64 events = queue->delivered.exchange(0, std::memory_order_relaxed);
        if (waits == 0) {
            continue;
        }
        // Queues of one name are reported together.
        const auto it = std::ranges::find(stats, queue->Name(), &Stat::name);
        if (it == stats.end()) {
            stats.push_back({queue->Name(), waits, ns, events});
        } else {
            it->waits += waits;
            it->ns += ns;
            it->events += events;
        }
    }
    if (stats.empty()) {
        return;
    }
    std::ranges::sort(stats, std::greater<>{}, &Stat::ns);
    std::printf("Guest event waits per frame:");
    for (std::size_t i = 0; i < std::min<std::size_t>(stats.size(), 6); ++i) {
        const Stat& s = stats[i];
        std::printf("%s %s %.1f waits %.2f ms %.1f events", i ? ";" : "", s.name.c_str(),
                    s.waits / frames, s.ns / (frames * 1e6), s.events / frames);
    }
    std::printf("\n");
}

void RegisterEventQueue(Core::Loader::SymbolsResolver* sym) {
    struct Entry {
        const char* nid;
        const char* library;
        u64 address;
    };
    const auto fn = [](auto* function) { return reinterpret_cast<u64>(function); };
    const Entry entries[] = {
        {"nh2IFMgKTv8", "libScePosix", fn(posix_kqueue)},
        {"RW-GEfpnsqg", "libScePosix", fn(posix_kevent)},
        {"D0OdFMjp46I", "libkernel", fn(sceKernelCreateEqueue)},
        {"jpFjmgAC5AE", "libkernel", fn(sceKernelDeleteEqueue)},
        {"fzyMKs9kim0", "libkernel", fn(sceKernelWaitEqueue)},
        {"vz+pg2zdopI", "libkernel", fn(sceKernelGetEventUserData)},
        {"4R6-OvI2cEA", "libkernel", fn(sceKernelAddUserEvent)},
        {"WDszmSbWuDk", "libkernel", fn(sceKernelAddUserEventEdge)},
        {"R74tt43xP6k", "libkernel", fn(sceKernelAddHRTimerEvent)},
        {"J+LF6LwObXU", "libkernel", fn(sceKernelDeleteHRTimerEvent)},
        {"57ZK+ODEXWY", "libkernel", fn(sceKernelAddTimerEvent)},
        {"YWQFUyXIVdU", "libkernel", fn(sceKernelDeleteTimerEvent)},
        {"F6e0kwo4cnk", "libkernel", fn(sceKernelTriggerUserEvent)},
        {"LJDwdSNTnDg", "libkernel", fn(sceKernelDeleteUserEvent)},
        {"mJ7aghmgvfc", "libkernel", fn(sceKernelGetEventId)},
        {"23CPPI1tyBY", "libkernel", fn(sceKernelGetEventFilter)},
        {"kwGyyjohI50", "libkernel", fn(sceKernelGetEventData)},
    };
    for (const Entry& entry : entries) {
        sym->AddSymbol(entry.nid, entry.library, "libkernel", Core::Loader::SymbolType::Function,
                       entry.address);
    }
}

} // namespace BbKernel
