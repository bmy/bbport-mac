// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <chrono>
#include "shim/bb_env.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "common/logging/log.h"
#include "common/thread.h"
#include "platform/avplayer/player.h"

namespace BbAvPlayer {

namespace {
constexpr s32 ErrorNotSupported = s32(0x806A0004);

/// BB_AVPLAYER_TRACE=1 (avplayer.cpp's trace): what the player tells the game.
bool Tracing() {
    static const bool on = bb_env_on("BB_AVPLAYER_TRACE");
    return on;
}
} // namespace

Player::Player(const AvPlayerInitData& init_) : init{init_}, game_events{init_.event_replacement} {
    auto_start = !game_events.event_callback || init.auto_start;
    if (init.default_language) {
        std::memcpy(default_language, init.default_language, sizeof(default_language));
    }
    controller.Run([this](std::stop_token stop) { ControllerThread(stop); });
}

Player::~Player() {
    {
        std::unique_lock lock{source_mutex};
        source.reset();
    }
    controller.Stop();
}

// ---- State -----------------------------------------------------------------------------------

bool Player::CanEnter(State next) const {
    const State current = state;
    const bool over = current == State::Stop || current == State::EndOfFile ||
                      current == State::Error;
    switch (next) {
    case State::Play:
        return !over;
    case State::Pause:
        return !over && current != State::Starting;
    case State::Jump:
        return !over && current != State::TrickMode && current != State::Starting;
    case State::TrickMode:
        return !over && current != State::Jump && current != State::Starting;
    case State::Buffering:
        return !over && current != State::Pause && current != State::Starting;
    default:
        return true;
    }
}

bool Player::SetState(State next) {
    std::scoped_lock lock{state_mutex};
    if (!CanEnter(next)) {
        LOG_ERROR(Lib_AvPlayer, "no state change {} -> {}", int(state.load()), int(next));
        return false;
    }
    previous_state = state.load();
    state = next;
    return true;
}

void Player::Emit(AvPlayerEvents event, void* data) {
    if (Tracing()) {
        std::printf("AvPlayer trace: event %d to the game%s (state %d)\n", int(event),
                    auto_start && event == AvPlayerEvents::StateReady ? " (auto start instead)" : "",
                    int(state.load()));
    }
    if (auto_start && event == AvPlayerEvents::StateReady) {
        AutoStart(); // the player starts itself instead of telling the game
        return;
    }
    if (game_events.event_callback) {
        game_events.event_callback(game_events.object_ptr, event, 0, data);
        if (Tracing()) {
            std::printf("AvPlayer trace: event %d: the game's callback returned\n", int(event));
        }
    }
}

/// The event a state change gives the game.
void Player::Announce(State reached) {
    switch (reached) {
    case State::Ready:
        Emit(AvPlayerEvents::StateReady);
        break;
    case State::Play:
        Emit(AvPlayerEvents::StatePlay);
        break;
    case State::Stop:
        Emit(AvPlayerEvents::StateStop);
        break;
    case State::Pause:
        Emit(AvPlayerEvents::StatePause);
        break;
    case State::Buffering:
        Emit(AvPlayerEvents::StateBuffering);
        break;
    default:
        break;
    }
}

void Player::AutoStart() {
    const s32 count = StreamCount();
    if (count <= 0) {
        Stop();
        return;
    }
    // The first stream of each kind, or the one in the default language.
    s32 chosen[3] = {-1, -1, -1};
    for (s32 i = 0; i < count; ++i) {
        AvPlayerStreamInfo info{};
        if (!StreamInfo(u32(i), info)) {
            Stop();
            return;
        }
        const char* language = nullptr;
        switch (info.type) {
        case AvPlayerStreamType::Video:
            language = info.details.video.language_code;
            break;
        case AvPlayerStreamType::Audio:
            language = info.details.audio.language_code;
            break;
        case AvPlayerStreamType::TimedText:
            language = info.details.subs.language_code;
            break;
        default:
            continue;
        }
        s32& slot = chosen[u32(info.type)];
        if (slot == -1 || (default_language[0] != '\0' &&
                           std::strncmp(default_language, language, sizeof(default_language)) == 0)) {
            slot = i;
        }
    }
    for (const s32 index : chosen) {
        if (index != -1) {
            EnableStream(u32(index));
        }
    }
    Start();
}

// ---- Controller ------------------------------------------------------------------------------

void Player::Post(Event event) {
    {
        std::scoped_lock lock{event_mutex};
        events.push_back(event);
    }
    event_cv.notify_one();
}

void Player::ControllerThread(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:AvControl");
    while (!stop.stop_requested()) {
        Event event;
        {
            std::unique_lock lock{event_mutex};
            // Events at once; the buffering state every 5 ms otherwise.
            if (!event_cv.wait_for(lock, stop, std::chrono::milliseconds(5),
                                   [this] { return !events.empty(); })) {
                lock.unlock();
                CheckBuffering();
                continue;
            }
            event = events.front();
            events.pop_front();
        }
        Handle(event);
    }
}

void Player::Handle(const Event& event) {
    switch (event.kind) {
    case Event::Kind::Warning: {
        s32 code = event.warning;
        Emit(AvPlayerEvents::WarningId, &code);
        break;
    }
    case Event::Kind::AddSource: {
        bool found;
        {
            std::unique_lock lock{source_mutex};
            found = source && source->FindStreams();
        }
        if (Tracing()) {
            std::printf("AvPlayer trace: streams %s (%d)\n", found ? "found" : "NOT found", found ? StreamCount() : -1);
        }
        if (found) {
            SetState(State::Ready);
            Announce(State::Ready);
        } else {
            OnWarning(ErrorNotSupported);
            SetState(State::Error);
        }
        break;
    }
    }
}

void Player::CheckBuffering() {
    const State current = state;
    if (current == State::EndOfFile) {
        bool ended;
        {
            std::shared_lock lock{source_mutex};
            ended = source && !source->IsActive();
        }
        if (ended && !stop_event_sent.exchange(true)) {
            Emit(AvPlayerEvents::StateStop);
        }
        return;
    }
    if (current != State::Buffering && current != State::Play) {
        return;
    }
    bool has_frames;
    {
        std::shared_lock lock{source_mutex};
        if (!source) {
            return;
        }
        has_frames = source->HasFrames(current == State::Buffering ? 10 : 0);
    }
    if (current == State::Buffering && has_frames) {
        const State before = previous_state;
        const State next = before >= State::Reserved11 ? before : State::Play;
        SetState(next);
        Announce(next);
    } else if (current == State::Play && !has_frames) {
        SetState(State::Buffering);
        Announce(State::Buffering);
    }
}

// ---- Game calls ------------------------------------------------------------------------------

bool Player::AddSource(std::string_view path) {
    if (path.empty()) {
        return false;
    }
    {
        std::unique_lock lock{source_mutex};
        if (source) {
            LOG_ERROR(Lib_AvPlayer, "one source per player");
            return false;
        }
        source = std::make_unique<Source>(*this);
        if (!source->Open(init, path)) {
            SetState(State::Error);
            source.reset();
            return false;
        }
    }
    SetState(State::AddingSource);
    // The streams found and StateReady told before AddSource returns: the game's movie (0x28fe620:
    // one event a frame from its queue; 0x28fe720 starts only after StateReady was handled) tries
    // to start on its next frames and tears the movie down when that fails. At 200+ FPS in the
    // menus a frame came before the controller thread's StateReady (~10 ms: the stream search) and
    // the opening movie's audio thread then ran into its torn-down object (DL_PANIC "Mutex is not
    // initialized", Windows, 2026-10-27; at the PS4's 33 ms a frame it always came in time).
    // BB_AVPLAYER_ASYNC_READY=1: on the controller thread, as before.
    static const bool async_ready = bb_env_on("BB_AVPLAYER_ASYNC_READY");
    if (async_ready) {
        Post({Event::Kind::AddSource});
    } else {
        Handle({Event::Kind::AddSource});
    }
    return true;
}

s32 Player::StreamCount() {
    std::shared_lock lock{source_mutex};
    return source ? source->StreamCount() : -1;
}

bool Player::StreamInfo(u32 index, AvPlayerStreamInfo& info) {
    std::shared_lock lock{source_mutex};
    return source && source->StreamInfo(index, info);
}

bool Player::EnableStream(u32 index) {
    std::shared_lock lock{source_mutex};
    return source && source->EnableStream(index);
}

bool Player::Start() {
    {
        std::unique_lock lock{source_mutex};
        if (!source) {
            return false;
        }
        const State current = state;
        if (current == State::Unknown || current == State::Initial ||
            current == State::AddingSource || current == State::Error) {
            return false; // no streams yet, or failed
        }
        // A playback under way (or ended) starts over.
        if (current != State::Ready && current != State::Stop) {
            source->Stop();
            if (!SetState(State::Stop)) {
                return false;
            }
        }
        stop_event_sent = false;
        SetState(State::Starting);
        if (!source->Start()) {
            LOG_ERROR(Lib_AvPlayer, "playback could not start");
            SetState(State::Error);
            return false;
        }
        SetState(State::Play);
    }
    Announce(State::Play);
    return true;
}

bool Player::Stop() {
    {
        std::shared_lock lock{source_mutex};
        if (!source || state == State::Stop) {
            return false;
        }
        source->Stop();
        if (!SetState(State::Stop)) {
            return false;
        }
        stop_event_sent = true;
    }
    Announce(State::Stop);
    return true;
}

bool Player::Pause() {
    {
        std::shared_lock lock{source_mutex};
        const State current = state;
        if (current == State::EndOfFile) {
            return true;
        }
        if (!source || current == State::Pause || current == State::Ready ||
            current == State::Initial || current == State::Unknown ||
            current == State::AddingSource) {
            return false;
        }
        source->Pause();
        if (!SetState(State::Pause)) {
            return false;
        }
    }
    Announce(State::Pause);
    return true;
}

bool Player::Resume() {
    State resumed;
    {
        std::shared_lock lock{source_mutex};
        if (!source || state != State::Pause) {
            return false;
        }
        source->Resume();
        resumed = previous_state;
        SetState(resumed);
    }
    Announce(resumed);
    return true;
}

void Player::SetSyncMode(AvPlayerAvSyncMode mode) {
    sync_mode = mode;
}

bool Player::SetLooping(bool looping) {
    std::shared_lock lock{source_mutex};
    if (!source) {
        return false;
    }
    source->SetLooping(looping);
    return true;
}

bool Player::GetAudioData(AvPlayerFrameInfo& info) {
    std::shared_lock lock{source_mutex};
    return source && source->GetAudioData(info);
}

bool Player::GetVideoData(AvPlayerFrameInfo& info) {
    std::shared_lock lock{source_mutex};
    return source && source->GetVideoData(info);
}

bool Player::GetVideoData(AvPlayerFrameInfoEx& info) {
    std::shared_lock lock{source_mutex};
    return source && source->GetVideoData(info);
}

bool Player::IsActive() {
    std::shared_lock lock{source_mutex};
    const State current = state;
    return source && current != State::Stop && current != State::Error && source->IsActive();
}

u64 Player::CurrentTime() {
    std::shared_lock lock{source_mutex};
    return source ? source->CurrentTime() : 0;
}

// ---- From the source -------------------------------------------------------------------------

void Player::OnWarning(s32 code) {
    Post({Event::Kind::Warning, code});
}

void Player::OnError() {
    SetState(State::Error);
}

void Player::OnEndOfFile() {
    SetState(State::EndOfFile);
}

} // namespace BbAvPlayer
