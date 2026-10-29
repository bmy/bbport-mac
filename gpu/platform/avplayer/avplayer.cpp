// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include "shim/bb_env.h"
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string_view>

#include "common/logging/log.h"
#include "core/libraries/libs.h"
#include "platform/avplayer/avplayer.h"
#include "platform/avplayer/player.h"
#include "platform/errors.h"

namespace BbAvPlayer {

namespace {

constexpr s32 ErrorInvalidParams = s32(0x806A0001);
constexpr s32 ErrorOperationFailed = s32(0x806A0002);
constexpr s32 ErrorNotSupported = s32(0x806A0004);

using Handle = Player*;

/// BB_AVPLAYER_TRACE=1 (diagnostics): the game's calls in order, with their results and the time
/// since the first (the opening movie's crash on Windows, 2026-10-27).
void Trace(const char* format, ...) {
    static const bool on = bb_env_on("BB_AVPLAYER_TRACE");
    if (!on) {
        return;
    }
    static const auto start = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    char text[256];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    std::printf("AvPlayer trace %.1f ms: %s\n", ms, text);
}

/// The player that plays now (the game has one at a time): a key press stops it (bbgpu_video_skip).
std::mutex active_mutex;
Player* active_player = nullptr;

void NoteCreated(Player* player) {
    std::scoped_lock lock{active_mutex};
    active_player = player;
}

/// From the file name's extension (a URI's query and fragment ignored).
AvPlayerSourceType SourceTypeOf(std::string_view path) {
    if (path.find("://") != std::string_view::npos) {
        path = path.substr(0, path.find_first_of("?#"));
    }
    const auto dot = path.rfind('.');
    if (dot == std::string_view::npos) {
        return AvPlayerSourceType::Unknown;
    }
    std::string_view extension = path.substr(dot);
    extension = extension.substr(0, extension.find('/'));
    const auto is = [extension](std::string_view name) {
        return std::ranges::equal(extension, name, [](char a, char b) {
            return std::tolower(u8(a)) == std::tolower(u8(b));
        });
    };
    if (is(".mp4") || is(".m4v") || is(".m3d") || is(".m4a") || is(".mov")) {
        return AvPlayerSourceType::FileMp4;
    }
    return is(".m3u8") ? AvPlayerSourceType::Hls : AvPlayerSourceType::Unknown;
}

bool HasAllocators(const AvPlayerMemAllocator& allocator) {
    return allocator.allocate && allocator.deallocate && allocator.allocate_texture &&
           allocator.deallocate_texture;
}

s32 Result(bool ok) {
    return ok ? BbErr::Ok : ErrorOperationFailed;
}

s32 AddSource(Handle player, std::string_view path, AvPlayerSourceType type) {
    if (type == AvPlayerSourceType::Unknown) {
        type = SourceTypeOf(path);
    }
    if (type == AvPlayerSourceType::Hls) {
        LOG_ERROR(Lib_AvPlayer, "HTTP live streaming is not supported");
        return ErrorNotSupported;
    }
    return Result(player->AddSource(path));
}

Handle PS4_SYSV_ABI sceAvPlayerInit(AvPlayerInitData* data) {
    if (!data || !HasAllocators(data->memory_replacement)) {
        LOG_ERROR(Lib_AvPlayer, "sceAvPlayerInit needs all four allocators");
        return nullptr;
    }
    auto* player = new Player(*data);
    NoteCreated(player);
    Trace("Init -> %p (auto start %d, events %d)", (void*)player, int(data->auto_start),
          data->event_replacement.event_callback != nullptr);
    return player;
}

s32 PS4_SYSV_ABI sceAvPlayerInitEx(const AvPlayerInitDataEx* data, Handle* player) {
    if (!data || !player || !HasAllocators(data->memory_replacement)) {
        return ErrorInvalidParams;
    }
    AvPlayerInitData init{};
    init.memory_replacement = data->memory_replacement;
    init.file_replacement = data->file_replacement;
    init.event_replacement = data->event_replacement;
    init.default_language = data->default_language;
    init.num_output_video_framebuffers = data->num_output_video_framebuffers;
    init.auto_start = data->auto_start;
    *player = new Player(init);
    NoteCreated(*player);
    Trace("InitEx -> %p (auto start %d)", (void*)*player, int(data->auto_start));
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceAvPlayerPostInit(Handle player, AvPlayerPostInitData* data) {
    return player && data ? BbErr::Ok : ErrorInvalidParams; // decoder choices: FFmpeg's
}

s32 PS4_SYSV_ABI sceAvPlayerClose(Handle player) {
    if (!player) {
        return ErrorInvalidParams;
    }
    {
        std::scoped_lock lock{active_mutex};
        if (active_player == player) {
            active_player = nullptr;
        }
    }
    Trace("Close %p", (void*)player);
    delete player;
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceAvPlayerAddSource(Handle player, const char* path) {
    if (!player || !path) {
        return ErrorInvalidParams;
    }
    const s32 result = AddSource(player, path, AvPlayerSourceType::Unknown);
    Trace("AddSource %p %s -> %#x", (void*)player, path, unsigned(result));
    return result;
}

s32 PS4_SYSV_ABI sceAvPlayerAddSourceEx(Handle player, AvPlayerUriType uri_type,
                                        AvPlayerSourceDetails* details) {
    if (!player || uri_type != AvPlayerUriType::Source || !details) {
        return ErrorInvalidParams;
    }
    return AddSource(player, std::string_view(details->uri.name, details->uri.length),
                     details->source_type);
}

s32 PS4_SYSV_ABI sceAvPlayerStreamCount(Handle player) {
    if (!player) {
        return ErrorInvalidParams;
    }
    const s32 count = player->StreamCount();
    Trace("StreamCount %p -> %d", (void*)player, count);
    return count < 0 ? ErrorOperationFailed : count;
}

s32 PS4_SYSV_ABI sceAvPlayerGetStreamInfo(Handle player, u32 index, AvPlayerStreamInfo* info) {
    if (!player || !info) {
        return ErrorInvalidParams;
    }
    const s32 result = Result(player->StreamInfo(index, *info));
    const char* language = info->type == AvPlayerStreamType::Audio ? info->details.audio.language_code
                           : info->type == AvPlayerStreamType::Video ? info->details.video.language_code
                                                                     : info->details.subs.language_code;
    Trace("GetStreamInfo %p %u -> %#x: type %d, duration %llu, language '%.4s'", (void*)player, index,
          unsigned(result), int(info->type), (unsigned long long)info->duration, language);
    return result;
}

s32 PS4_SYSV_ABI sceAvPlayerEnableStream(Handle player, u32 index) {
    const s32 result = player ? Result(player->EnableStream(index)) : ErrorInvalidParams;
    Trace("EnableStream %p %u -> %#x", (void*)player, index, unsigned(result));
    return result;
}

/// Not supported: every enabled stream plays.
s32 PS4_SYSV_ABI sceAvPlayerDisableStream(Handle player, u32) {
    return player ? BbErr::Ok : ErrorInvalidParams;
}

s32 PS4_SYSV_ABI sceAvPlayerChangeStream() {
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceAvPlayerStart(Handle player) {
    const s32 result = player ? Result(player->Start()) : ErrorInvalidParams;
    Trace("Start %p -> %#x", (void*)player, unsigned(result));
    return result;
}

s32 PS4_SYSV_ABI sceAvPlayerStop(Handle player) {
    const s32 result = player ? Result(player->Stop()) : ErrorInvalidParams;
    Trace("Stop %p -> %#x", (void*)player, unsigned(result));
    return result;
}

s32 PS4_SYSV_ABI sceAvPlayerPause(Handle player) {
    return player ? Result(player->Pause()) : ErrorInvalidParams;
}

s32 PS4_SYSV_ABI sceAvPlayerResume(Handle player) {
    return player ? Result(player->Resume()) : ErrorInvalidParams;
}

/// Not supported: playback goes on from where it is.
s32 PS4_SYSV_ABI sceAvPlayerJumpToTime(Handle player, u64 time_ms) {
    LOG_WARNING(Lib_AvPlayer, "jumping (to {} ms) is not supported", time_ms);
    return player ? BbErr::Ok : ErrorInvalidParams;
}

s32 PS4_SYSV_ABI sceAvPlayerSetTrickSpeed(Handle player, s32 speed) {
    LOG_WARNING(Lib_AvPlayer, "trick speed {} is not supported", speed);
    return player ? BbErr::Ok : ErrorInvalidParams;
}

s32 PS4_SYSV_ABI sceAvPlayerSetAvSyncMode(Handle player, AvPlayerAvSyncMode mode) {
    if (!player) {
        return ErrorInvalidParams;
    }
    player->SetSyncMode(mode);
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceAvPlayerSetLooping(Handle player, bool looping) {
    return player ? Result(player->SetLooping(looping)) : ErrorInvalidParams;
}

bool PS4_SYSV_ABI sceAvPlayerGetAudioData(Handle player, AvPlayerFrameInfo* info) {
    return player && info && player->GetAudioData(*info);
}

bool PS4_SYSV_ABI sceAvPlayerGetVideoData(Handle player, AvPlayerFrameInfo* info) {
    return player && info && player->GetVideoData(*info);
}

bool PS4_SYSV_ABI sceAvPlayerGetVideoDataEx(Handle player, AvPlayerFrameInfoEx* info) {
    return player && info && player->GetVideoData(*info);
}

bool PS4_SYSV_ABI sceAvPlayerIsActive(Handle player) {
    const bool active = player && player->IsActive();
    // Said when it changes (the game asks every frame).
    static Handle last_player = nullptr;
    static int last = -1;
    if (player != last_player || int(active) != last) {
        last_player = player;
        last = int(active);
        Trace("IsActive %p -> %d", (void*)player, int(active));
    }
    return active;
}

u64 PS4_SYSV_ABI sceAvPlayerCurrentTime(Handle player) {
    return player ? player->CurrentTime() : u64(s64(ErrorInvalidParams));
}

/// The library's own log goes nowhere here.
s32 PS4_SYSV_ABI sceAvPlayerSetLogCallback(AvPlayerLogCallback, void*) {
    return BbErr::Ok;
}

s32 PS4_SYSV_ABI sceAvPlayerVprintf(const char*, va_list) {
    return BbErr::Ok;
}

} // namespace

void RegisterLib(Core::Loader::SymbolsResolver* sym) {
    const auto fn = [](auto* function) { return reinterpret_cast<u64>(function); };
    const std::pair<const char*, u64> exports[] = {
        {"KMcEa+rHsIo", fn(sceAvPlayerAddSource)},
        {"x8uvuFOPZhU", fn(sceAvPlayerAddSourceEx)},
        {"buMCiJftcfw", fn(sceAvPlayerChangeStream)},
        {"NkJwDzKmIlw", fn(sceAvPlayerClose)},
        {"wwM99gjFf1Y", fn(sceAvPlayerCurrentTime)},
        {"BOVKAzRmuTQ", fn(sceAvPlayerDisableStream)},
        {"ODJK2sn9w4A", fn(sceAvPlayerEnableStream)},
        {"Wnp1OVcrZgk", fn(sceAvPlayerGetAudioData)},
        {"d8FcbzfAdQw", fn(sceAvPlayerGetStreamInfo)},
        {"o3+RWnHViSg", fn(sceAvPlayerGetVideoData)},
        {"JdksQu8pNdQ", fn(sceAvPlayerGetVideoDataEx)},
        {"aS66RI0gGgo", fn(sceAvPlayerInit)},
        {"o9eWRkSL+M4", fn(sceAvPlayerInitEx)},
        {"UbQoYawOsfY", fn(sceAvPlayerIsActive)},
        {"XC9wM+xULz8", fn(sceAvPlayerJumpToTime)},
        {"9y5v+fGN4Wk", fn(sceAvPlayerPause)},
        {"HD1YKVU26-M", fn(sceAvPlayerPostInit)},
        {"w5moABNwnRY", fn(sceAvPlayerResume)},
        {"k-q+xOxdc3E", fn(sceAvPlayerSetAvSyncMode)},
        {"eBTreZ84JFY", fn(sceAvPlayerSetLogCallback)},
        {"OVths0xGfho", fn(sceAvPlayerSetLooping)},
        {"av8Z++94rs0", fn(sceAvPlayerSetTrickSpeed)},
        {"ET4Gr-Uu07s", fn(sceAvPlayerStart)},
        {"ZC17w3vB5Lo", fn(sceAvPlayerStop)},
        {"hdTyRzCXQeQ", fn(sceAvPlayerStreamCount)},
        {"yN7Jhuv8g24", fn(sceAvPlayerVprintf)},
    };
    for (const auto& [nid, address] : exports) {
        sym->AddSymbol(nid, "libSceAvPlayer", "libSceAvPlayer", Core::Loader::SymbolType::Function,
                       address);
    }
}

} // namespace BbAvPlayer

// A video playing now (the opening movie, the endings): the keyboard's Enter and Escape and the
// gamepad's Options stop it (src/runtime_pad.c). From the SrShadowy fork.
extern "C" int bbgpu_video_active(void) {
    std::scoped_lock lock{BbAvPlayer::active_mutex};
    return BbAvPlayer::active_player && BbAvPlayer::active_player->IsActive() ? 1 : 0;
}

extern "C" int bbgpu_video_skip(void) {
    std::scoped_lock lock{BbAvPlayer::active_mutex};
    if (!BbAvPlayer::active_player || !BbAvPlayer::active_player->IsActive()) {
        return 0;
    }
    LOG_INFO(Lib_AvPlayer, "video skipped by the player");
    return BbAvPlayer::active_player->Stop() ? 1 : 0;
}
