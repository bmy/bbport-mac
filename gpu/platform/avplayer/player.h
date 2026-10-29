// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a player (sceAvPlayerInit's handle): its state machine and the controller thread that
// tells the game what happens (ready, playing, buffering, stopped) through its event callback.
// With no callback, or auto start, the player enables the first (or default-language) streams
// itself and starts when ready.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string_view>

#include "core/libraries/kernel/threads.h"
#include "platform/avplayer/avplayer.h"
#include "platform/avplayer/source.h"

namespace BbAvPlayer {

class Player final : public SourceListener {
public:
    explicit Player(const AvPlayerInitData& init);
    ~Player() override;

    bool AddSource(std::string_view path);
    s32 StreamCount();
    bool StreamInfo(u32 index, AvPlayerStreamInfo& info);
    bool EnableStream(u32 index);
    bool Start();
    bool Stop();
    bool Pause();
    bool Resume();
    void SetSyncMode(AvPlayerAvSyncMode mode);
    bool SetLooping(bool looping);
    bool GetAudioData(AvPlayerFrameInfo& info);
    bool GetVideoData(AvPlayerFrameInfo& info);
    bool GetVideoData(AvPlayerFrameInfoEx& info);
    bool IsActive();
    u64 CurrentTime();

    // SourceListener (any thread).
    AvPlayerAvSyncMode SyncMode() override {
        return sync_mode;
    }
    void OnWarning(s32 code) override;
    void OnError() override;
    void OnEndOfFile() override;

private:
    enum class State {
        Unknown,
        Initial,
        AddingSource,
        Ready,
        Play,
        Stop,
        EndOfFile,
        Pause,
        Reserved8,
        Jump,
        TrickMode,
        Reserved11,
        Buffering,
        Starting,
        Error,
    };
    struct Event {
        enum class Kind { AddSource, Warning } kind;
        s32 warning = 0;
    };

    bool SetState(State state);
    bool CanEnter(State state) const;
    void Emit(AvPlayerEvents event, void* data = nullptr);
    void Announce(State state);
    void Post(Event event);
    void ControllerThread(std::stop_token stop);
    void Handle(const Event& event);
    void CheckBuffering();
    void AutoStart();

    AvPlayerInitData init;
    AvPlayerEventReplacement game_events{};
    bool auto_start = false;
    char default_language[4]{};
    std::atomic<AvPlayerAvSyncMode> sync_mode{AvPlayerAvSyncMode::Default};

    std::shared_mutex source_mutex;
    std::unique_ptr<Source> source;

    std::mutex state_mutex;
    std::atomic<State> state{State::Initial};
    std::atomic<State> previous_state{State::Initial};
    std::atomic<bool> stop_event_sent{false};

    std::mutex event_mutex;
    std::condition_variable_any event_cv;
    std::deque<Event> events;
    Libraries::Kernel::Thread controller;
};

} // namespace BbAvPlayer
