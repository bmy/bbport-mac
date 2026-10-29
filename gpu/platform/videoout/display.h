// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the display's threads. The vblank thread ticks at the vblank rate (BB_VBLANK_HZ),
// takes queued flips (every (flip_rate + 1) vblanks; with a frame limit or the uncapped presets,
// as soon as they arrive and their slot allows) and signals vblank events. The present thread
// (BB_PRESENT_THREAD=0 keeps it on the vblank thread) does the presenter work - acquire, blit,
// present - in order: that work blocks while the GPU is the bottleneck, and on the vblank thread
// it delayed the game's vblank events (its frame timing jumped).
#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>

#include "platform/videoout/videoout.h"

namespace Vulkan {
struct Frame;
}

namespace BbVideoOut {

/// sceVideoOutOpen's thread parameters (unused here).
struct ServiceThreadParams {
    u32 unknown;
    bool set_priority;
    u32 priority;
    bool set_affinity;
    u64 affinity;
};

class Display {
public:
    Display(u32 width, u32 height);
    ~Display();

    /// Opens the port: its handle (1), or an error.
    s32 Open();
    void Close();
    Port* GetPort(s32 handle);

    s32 RegisterBuffers(Port* port, s32 start, void* const* addresses, s32 count,
                        const BufferAttribute* attribute);
    s32 UnregisterBuffers(Port* port, s32 group);
    s32 ChangeBufferAttribute(Port* port, s32 group, const BufferAttribute* attribute);

    /// Queues a flip to buffer `index` (-1: blank). `at_eop`: from the flip interrupt at the end
    /// of a GPU frame (on the command processor's thread); else from any thread. False: the flip
    /// queue is full.
    bool SubmitFlip(Port* port, s32 index, s64 flip_arg, bool at_eop = false);

private:
    struct Request {
        Vulkan::Frame* frame = nullptr;
        Port* port = nullptr;
        s64 flip_arg = 0;
        s32 index = -1;
        bool eop = false;
    };

    void QueueFlip(Port* port, s32 index, s64 flip_arg, bool at_eop);
    bool TakeRequest(Request& request);
    void Flip(const Request& request);
    void DrawBlankFrame();
    void VblankThread(std::stop_token stop);
    void PresentThread(std::stop_token stop);
    void RunPresenter(std::function<void()> work, bool only_if_idle = false);

    std::mutex present_mutex;
    std::condition_variable_any present_cv;
    std::deque<std::function<void()>> present_work;
    bool present_busy = false;
    bool separate_present = true;
    std::jthread present_thread; ///< before vblank_thread: destroyed after it

    std::mutex request_mutex;
    std::condition_variable request_cv; ///< a flip was queued
    std::queue<Request> requests;
    Port main_port;
    std::jthread vblank_thread;
};

} // namespace BbVideoOut
