// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: one media source of the player: FFmpeg demuxing on one thread, video and audio decoding
// on one thread each, into buffers the game allocated (its allocator callbacks). Video frames are
// NV12 (pitch aligned to 64, height to 16), audio frames 16-bit PCM of up to 1024 samples.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "core/libraries/kernel/threads.h"
#include "platform/avplayer/avplayer.h"

struct AVCodecContext;
struct AVFormatContext;
struct AVFrame;
struct AVIOContext;
struct AVPacket;
struct SwrContext;
struct SwsContext;

namespace BbAvPlayer {

/// What the source tells its player.
class SourceListener {
public:
    virtual ~SourceListener() = default;
    virtual AvPlayerAvSyncMode SyncMode() = 0;
    virtual void OnWarning(s32 code) = 0;
    virtual void OnError() = 0;
    virtual void OnEndOfFile() = 0;
};

/// A buffer from the game's allocator, given back when dropped.
class GuestBuffer {
public:
    GuestBuffer(const AvPlayerMemAllocator& allocator, u32 align, u32 size, bool texture);
    ~GuestBuffer();
    GuestBuffer(GuestBuffer&& other) noexcept;
    GuestBuffer& operator=(GuestBuffer&& other) noexcept;
    GuestBuffer(const GuestBuffer&) = delete;
    GuestBuffer& operator=(const GuestBuffer&) = delete;

    u8* Data() const {
        return data;
    }
    u64 Size() const {
        return size;
    }

private:
    const AvPlayerMemAllocator* allocator;
    u8* data = nullptr;
    u64 size = 0;
    bool texture = false;
};

struct Frame {
    GuestBuffer buffer;
    AvPlayerFrameInfoEx info;
};

/// A queue shared by two threads, with a wait for "not empty".
template <typename T>
class SharedQueue {
public:
    void Push(T&& value) {
        {
            std::scoped_lock lock{mutex};
            items.push_back(std::move(value));
        }
        cv.notify_all();
    }
    std::optional<T> Pop() {
        std::scoped_lock lock{mutex};
        if (items.empty()) {
            return std::nullopt;
        }
        T value = std::move(items.front());
        items.pop_front();
        return value;
    }
    std::size_t Size() {
        std::scoped_lock lock{mutex};
        return items.size();
    }
    /// `read(first item)`, if there is one.
    template <typename F>
    auto PeekWith(F&& read) -> std::optional<decltype(read(std::declval<T&>()))> {
        std::scoped_lock lock{mutex};
        if (items.empty()) {
            return std::nullopt;
        }
        return read(items.front());
    }
    void Clear() {
        std::scoped_lock lock{mutex};
        items.clear();
    }
    /// Until an item is there (true), `give_up()` holds or `stop` is asked (false). Wake makes
    /// waiters look again.
    template <typename Pred>
    bool WaitNotEmpty(std::stop_token stop, Pred&& give_up) {
        std::unique_lock lock{mutex};
        cv.wait(lock, stop, [&] { return !items.empty() || give_up(); });
        return !items.empty();
    }
    void Wake() {
        std::scoped_lock lock{mutex};
        cv.notify_all();
    }

private:
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<T> items;
};

class Source {
public:
    explicit Source(SourceListener& listener);
    ~Source();

    /// Opens `path`: through the game's file callbacks when it gave them, else from the port's
    /// file system.
    bool Open(const AvPlayerInitData& init, std::string_view path);
    /// Reads the container's streams (the playable ones: H.264, HEVC, AAC).
    bool FindStreams();
    s32 StreamCount() const;
    bool StreamInfo(u32 index, AvPlayerStreamInfo& info) const;
    bool EnableStream(u32 index);
    void SetLooping(bool looping);
    /// More than `count` video packets are waiting (or the file ended).
    bool HasFrames(u32 count);

    bool Start();
    bool Stop();
    void Pause();
    void Resume();

    bool GetAudioData(AvPlayerFrameInfo& info);
    bool GetVideoData(AvPlayerFrameInfo& info);
    bool GetVideoData(AvPlayerFrameInfoEx& info);
    /// Playback time in milliseconds (the audio clock when there is audio).
    u64 CurrentTime();
    /// Packets or frames are still to come.
    bool IsActive();

private:
    struct PacketDeleter {
        void operator()(AVPacket* packet) const;
    };
    struct FrameDeleter {
        void operator()(AVFrame* frame) const;
    };
    using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
    using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;

    struct StreamEntry {
        int ffmpeg_index;
        AvPlayerStreamInfo info;
    };
    /// An enabled stream and its decoder.
    struct Track {
        int ffmpeg_index = -1;
        AVCodecContext* codec = nullptr;
        SharedQueue<PacketPtr> packets;
        SharedQueue<GuestBuffer> free_buffers;
        SharedQueue<Frame> frames;
        std::optional<Frame> current; ///< the frame the game holds
        Libraries::Kernel::Thread decoder;
        bool Enabled() const {
            return ffmpeg_index >= 0;
        }
    };

    bool OpenDecoder(Track& track);
    void Rewind();
    void DemuxThread(std::stop_token stop);
    void DecodeThread(std::stop_token stop, Track& track, bool video);
    bool ToFrame(Track& track, bool video, const AVFrame& decoded);
    u64 TimestampMillis(const AVFrame& frame, int ffmpeg_index) const;
    u64 DurationMillis() const;
    void WakeAll();
    void ResetPlayback();

    SourceListener& listener;
    AvPlayerMemAllocator allocator{};
    AvPlayerFileReplacement file{};
    int file_fd = -1;
    u64 file_size = 0;
    u64 file_position = 0;
    AVIOContext* avio = nullptr;
    AVFormatContext* format = nullptr;
    std::vector<StreamEntry> streams;
    u32 video_buffer_count = 2;
    u64 duration = 0;

    Track video;
    Track audio;
    SwsContext* sws = nullptr;
    SwrContext* swr = nullptr;
    Libraries::Kernel::Thread demuxer;
    std::mutex state_mutex;
    std::atomic<bool> started{false};
    std::atomic<bool> looping{false};
    std::atomic<bool> paused{false};
    std::atomic<bool> end_of_file{false};
    std::atomic<int> decoders_running{0};

    using Clock = std::chrono::steady_clock;
    std::mutex clock_mutex;
    std::optional<Clock::time_point> start_time;
    std::optional<u64> last_audio_ms;
    std::optional<Clock::time_point> last_audio_time;
    Clock::time_point pause_start{};
    Clock::duration paused_for{};

    friend int ReadFileCallback(void*, u8*, int);
    friend s64 SeekFileCallback(void*, s64, int);
};

} // namespace BbAvPlayer
