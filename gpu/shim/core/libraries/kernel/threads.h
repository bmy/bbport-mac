// bbport: host threads that may call guest code (AvPlayer allocator callbacks).
// Each thread gets a guest TCB (GS base, TLS) from the C runtime before running.
#pragma once
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>
#include "common/types.h"

extern "C" void runtime_thread_attach_host(const char* name);

namespace Libraries::Kernel {
/// A std::jthread that can be stopped/joined from any thread, including itself.
///
/// AvPlayer's decoder threads Join their own Thread object as they exit while the game thread
/// may Stop it at the same moment (sceAvPlayerStop at the end of a movie or on skip). The old
/// version checked joinable() and then joined without a lock, so the exiting thread could
/// detach itself in between and the other thread would pthread_join a detached, finished
/// thread: undefined behaviour that returns at once on glibc but waits forever on macOS (the
/// game froze after the opening movie). Now the decision is made under a lock: whoever comes
/// first either detaches (the thread itself) or takes the std::jthread out of the object and
/// joins it outside the lock, so the other side always sees "nothing to join".
class Thread {
public:
    Thread() = default;
    ~Thread() {
        Stop();
    }
    Thread(const Thread&) = delete;
    Thread& operator=(const Thread&) = delete;

    void Run(std::function<void(std::stop_token)>&& func) {
        Stop(); // a previous thread of this object finishes first
        std::scoped_lock lock{mutex};
        thread = std::jthread([func = std::move(func)](std::stop_token stop) {
            runtime_thread_attach_host("bb:hle");
            func(stop);
        });
    }

    void Join() {
        std::jthread to_join;
        {
            std::scoped_lock lock{mutex};
            if (!thread.joinable()) {
                return;
            }
            if (thread.get_id() == std::this_thread::get_id()) {
                thread.detach(); // a thread cannot join itself; it is about to return anyway
                return;
            }
            to_join = std::move(thread);
        }
        to_join.join();
    }

    bool Joinable() const {
        std::scoped_lock lock{mutex};
        return thread.joinable();
    }

    void Stop() {
        {
            std::scoped_lock lock{mutex};
            if (!thread.joinable()) {
                return;
            }
            thread.request_stop();
        }
        Join();
    }

private:
    mutable std::mutex mutex;
    std::jthread thread;
};
} // namespace Libraries::Kernel
