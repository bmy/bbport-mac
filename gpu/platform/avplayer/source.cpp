// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include <cstring>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libavutil/mathematics.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "core/memory.h"
#include "platform/avplayer/source.h"

extern "C" int runtime_file_translate(const char* guest, char* out, size_t size);

namespace BbAvPlayer {

namespace {

constexpr s32 WarningLoopingBack = s32(0x806A00A1);
constexpr u32 IoBufferSize = 4096;
constexpr u32 AudioBufferSize = 8 * sizeof(s32) * 1024; ///< 8 channels, 1024 samples

std::string AvError(int error) {
    char text[AV_ERROR_MAX_STRING_SIZE];
    av_make_error_string(text, sizeof(text), error);
    return text;
}

u64 ToMillis(s64 timestamp, AVRational time_base) {
    if (timestamp == AV_NOPTS_VALUE || timestamp <= 0 || time_base.num <= 0 || time_base.den <= 0) {
        return 0;
    }
    return u64(std::max<s64>(0, av_rescale_q(timestamp, time_base, AVRational{1, 1000})));
}

u64 StreamDuration(const AVFormatContext& format, const AVStream& stream) {
    const u64 duration = ToMillis(stream.duration, stream.time_base);
    return duration != 0 ? duration : ToMillis(format.duration, AVRational{1, AV_TIME_BASE});
}

bool Playable(const AVStream& stream) {
    const AVCodecID codec = stream.codecpar->codec_id;
    return codec == AV_CODEC_ID_H264 || codec == AV_CODEC_ID_HEVC || codec == AV_CODEC_ID_AAC;
}

void CopyLanguage(const AVStream& stream, char* code) {
    if (const AVDictionaryEntry* language = av_dict_get(stream.metadata, "language", nullptr, 0)) {
        std::memcpy(code, language->value, std::min<std::size_t>(std::strlen(language->value), 3));
    }
}

/// NV12 into the game's buffer: rows `pitch` bytes apart (width aligned to 64), the plane as
/// tall as the height aligned to 16 (the last row repeated into the padding).
void CopyNv12(u8* dst, u64 capacity, const AVFrame& src, u32 pitch, u32 height) {
    ASSERT_MSG(u64(pitch) * height * 3 / 2 <= capacity, "video frame does not fit its buffer");
    const u32 width = u32(src.width);
    const u32 rows = u32(src.height);
    u8* const luma = dst;
    u8* const chroma = dst + u64(pitch) * height;
    for (u32 y = 0; y < height; ++y) {
        const u32 from = std::min(y, rows - 1);
        std::memcpy(luma + u64(y) * pitch, src.data[0] + s64(from) * src.linesize[0], width);
    }
    for (u32 y = 0; y < height / 2; ++y) {
        const u32 from = std::min(y, rows / 2 - 1);
        std::memcpy(chroma + u64(y) * pitch, src.data[1] + s64(from) * src.linesize[1], width);
    }
}

} // namespace

// ---- Guest buffers ---------------------------------------------------------------------------

GuestBuffer::GuestBuffer(const AvPlayerMemAllocator& allocator_, u32 align, u32 size_,
                         bool texture_)
    : allocator{&allocator_}, size{size_}, texture{texture_} {
    void* memory = texture ? allocator->allocate_texture(allocator->object_ptr, align, size_)
                           : allocator->allocate(allocator->object_ptr, align, size_);
    data = static_cast<u8*>(memory);
    ASSERT_MSG(data, "the game's allocator gave no memory for a frame");
}

GuestBuffer::~GuestBuffer() {
    if (!data) {
        return;
    }
    if (texture) {
        allocator->deallocate_texture(allocator->object_ptr, data);
    } else {
        allocator->deallocate(allocator->object_ptr, data);
    }
}

GuestBuffer::GuestBuffer(GuestBuffer&& other) noexcept
    : allocator{other.allocator}, data{std::exchange(other.data, nullptr)}, size{other.size},
      texture{other.texture} {}

GuestBuffer& GuestBuffer::operator=(GuestBuffer&& other) noexcept {
    std::swap(allocator, other.allocator);
    std::swap(data, other.data);
    std::swap(size, other.size);
    std::swap(texture, other.texture);
    return *this;
}

void Source::PacketDeleter::operator()(AVPacket* packet) const {
    av_packet_free(&packet);
}

void Source::FrameDeleter::operator()(AVFrame* frame) const {
    av_frame_free(&frame);
}

// ---- Opening ---------------------------------------------------------------------------------

/// AVIO over the game's file callbacks.
int ReadFileCallback(void* opaque, u8* buffer, int size) {
    auto* self = static_cast<Source*>(opaque);
    if (self->file_position >= self->file_size) {
        return AVERROR_EOF;
    }
    const u32 length = u32(std::min<u64>(u64(size), self->file_size - self->file_position));
    const s32 read =
        self->file.read_offset(self->file.object_ptr, buffer, self->file_position, length);
    if (read <= 0) {
        return AVERROR_EOF;
    }
    self->file_position += u64(read);
    return read;
}

s64 SeekFileCallback(void* opaque, s64 offset, int whence) {
    auto* self = static_cast<Source*>(opaque);
    if (whence & AVSEEK_SIZE) {
        return s64(self->file_size);
    }
    s64 base = 0;
    switch (whence) {
    case SEEK_SET:
        break;
    case SEEK_CUR:
        base = s64(self->file_position);
        break;
    case SEEK_END:
        base = s64(self->file_size);
        break;
    default:
        return -1;
    }
    self->file_position = u64(std::clamp<s64>(base + offset, 0, s64(self->file_size)));
    return s64(self->file_position);
}

Source::Source(SourceListener& listener_) : listener{listener_} {}

Source::~Source() {
    Stop();
    if (video.codec) {
        avcodec_free_context(&video.codec);
    }
    if (audio.codec) {
        avcodec_free_context(&audio.codec);
    }
    sws_freeContext(sws);
    swr_free(&swr);
    if (format) {
        avformat_close_input(&format);
    }
    if (avio) {
        av_freep(&avio->buffer); // FFmpeg may have replaced the one given
        avio_context_free(&avio);
    }
    if (file_fd >= 0 && file.close) {
        file.close(file.object_ptr);
    }
}

bool Source::Open(const AvPlayerInitData& init, std::string_view path) {
    allocator = init.memory_replacement;
    video_buffer_count = u32(std::clamp(init.num_output_video_framebuffers, 2, 16));
    const std::string guest_path{path};
    format = avformat_alloc_context();
    if (!format) {
        return false;
    }
    if (init.file_replacement.open) {
        file = init.file_replacement;
        file_fd = file.open(file.object_ptr, guest_path.c_str());
        if (file_fd < 0) {
            LOG_ERROR(Lib_AvPlayer, "the game's file callbacks could not open {}", guest_path);
            return false;
        }
        file_size = file.size(file.object_ptr);
        auto* buffer = static_cast<u8*>(av_malloc(IoBufferSize));
        avio = avio_alloc_context(buffer, IoBufferSize, 0, this, &ReadFileCallback, nullptr,
                                  &SeekFileCallback);
        format->pb = avio;
        format->flags |= AVFMT_FLAG_CUSTOM_IO;
        if (avformat_open_input(&format, nullptr, nullptr, nullptr) < 0) {
            return false; // the context is freed on failure
        }
        return true;
    }
    char host[1024];
    if (runtime_file_translate(guest_path.c_str(), host, sizeof(host)) != 0) {
        LOG_ERROR(Lib_AvPlayer, "no host file for {}", guest_path);
        avformat_free_context(format);
        format = nullptr;
        return false;
    }
    LOG_INFO(Lib_AvPlayer, "playing {} ({})", guest_path, host);
    return avformat_open_input(&format, host, nullptr, nullptr) >= 0;
}

bool Source::FindStreams() {
    if (!format || avformat_find_stream_info(format, nullptr) < 0) {
        return false;
    }
    for (u32 i = 0; i < format->nb_streams; ++i) {
        const AVStream& stream = *format->streams[i];
        if (!Playable(stream)) {
            continue;
        }
        AvPlayerStreamInfo info{};
        info.duration = StreamDuration(*format, stream);
        const AVCodecParameters& codec = *stream.codecpar;
        if (codec.codec_type == AVMEDIA_TYPE_VIDEO) {
            info.type = AvPlayerStreamType::Video;
            info.details.video.aspect_ratio = f32(codec.width) / f32(codec.height);
            info.details.video.width = Common::AlignUp<u32>(u32(codec.width), 16);
            info.details.video.height = Common::AlignUp<u32>(u32(codec.height), 16);
            CopyLanguage(stream, info.details.video.language_code);
        } else {
            info.type = AvPlayerStreamType::Audio;
            info.details.audio.channel_count = u16(codec.ch_layout.nb_channels);
            info.details.audio.sample_rate = u32(codec.sample_rate);
            info.details.audio.size = 0;
            CopyLanguage(stream, info.details.audio.language_code);
        }
        streams.push_back({int(i), info});
    }
    duration = DurationMillis();
    return !streams.empty();
}

s32 Source::StreamCount() const {
    return s32(streams.size());
}

bool Source::StreamInfo(u32 index, AvPlayerStreamInfo& info) const {
    if (index >= streams.size()) {
        return false;
    }
    info = streams[index].info;
    return true;
}

bool Source::EnableStream(u32 index) {
    if (!format || index >= streams.size()) {
        return false;
    }
    const StreamEntry& entry = streams[index];
    (entry.info.type == AvPlayerStreamType::Video ? video : audio).ffmpeg_index =
        entry.ffmpeg_index;
    return true;
}

void Source::SetLooping(bool loop) {
    looping = loop;
}

bool Source::HasFrames(u32 count) {
    return video.packets.Size() > count || end_of_file;
}

u64 Source::DurationMillis() const {
    u64 longest = 0;
    for (const Track* track : {&video, &audio}) {
        if (track->Enabled()) {
            longest = std::max(longest, StreamDuration(*format, *format->streams[track->ffmpeg_index]));
        }
    }
    if (longest == 0) {
        for (u32 i = 0; i < format->nb_streams; ++i) {
            longest = std::max(longest, StreamDuration(*format, *format->streams[i]));
        }
    }
    return longest;
}

// ---- Playback --------------------------------------------------------------------------------

bool Source::OpenDecoder(Track& track) {
    if (track.codec) {
        avcodec_flush_buffers(track.codec); // started before: from the beginning
        return true;
    }
    const AVStream& stream = *format->streams[track.ffmpeg_index];
    const AVCodec* decoder = avcodec_find_decoder(stream.codecpar->codec_id);
    if (!decoder) {
        return false;
    }
    track.codec = avcodec_alloc_context3(decoder);
    if (avcodec_parameters_to_context(track.codec, stream.codecpar) < 0 ||
        avcodec_open2(track.codec, decoder, nullptr) < 0) {
        LOG_ERROR(Lib_AvPlayer, "no decoder for stream {}", track.ffmpeg_index);
        return false;
    }
    return true;
}

void Source::Rewind() {
    av_seek_frame(format, -1, 0, AVSEEK_FLAG_BACKWARD);
}

bool Source::Start() {
    std::scoped_lock lock{state_mutex};
    if (!video.Enabled() && !audio.Enabled()) {
        LOG_ERROR(Lib_AvPlayer, "nothing to play: no stream enabled");
        return false;
    }
    if (started) {
        return false;
    }
    if ((video.Enabled() && !OpenDecoder(video)) || (audio.Enabled() && !OpenDecoder(audio))) {
        return false;
    }
    Rewind();
    duration = DurationMillis();
    if (video.Enabled()) {
        const u32 pitch = Common::AlignUp<u32>(u32(video.codec->width), 64);
        const u32 height = Common::AlignUp<u32>(u32(video.codec->height), 16);
        for (u32 i = 0; i < video_buffer_count; ++i) {
            video.free_buffers.Push(GuestBuffer(allocator, 0x100, pitch * height * 3 / 2, true));
        }
    }
    if (audio.Enabled()) {
        for (u32 i = 0; i < video_buffer_count * 2; ++i) {
            audio.free_buffers.Push(GuestBuffer(allocator, 0x10, AudioBufferSize, false));
        }
    }
    end_of_file = false;
    paused = false;
    decoders_running = int(video.Enabled()) + int(audio.Enabled());
    {
        std::scoped_lock clock_lock{clock_mutex};
        start_time = Clock::now();
        last_audio_ms.reset();
        last_audio_time.reset();
        paused_for = {};
    }
    started = true;
    demuxer.Run([this](std::stop_token stop) { DemuxThread(stop); });
    if (video.Enabled()) {
        video.decoder.Run([this](std::stop_token stop) { DecodeThread(stop, video, true); });
    }
    if (audio.Enabled()) {
        audio.decoder.Run([this](std::stop_token stop) { DecodeThread(stop, audio, false); });
    }
    return true;
}

void Source::WakeAll() {
    for (Track* track : {&video, &audio}) {
        track->packets.Wake();
        track->free_buffers.Wake();
    }
}

bool Source::Stop() {
    std::scoped_lock lock{state_mutex};
    if (!started) {
        return false;
    }
    started = false;
    demuxer.Stop();
    video.decoder.Stop();
    audio.decoder.Stop();
    ResetPlayback();
    return true;
}

void Source::ResetPlayback() {
    for (Track* track : {&video, &audio}) {
        track->current.reset();
        track->frames.Clear();
        track->packets.Clear();
        track->free_buffers.Clear();
    }
    file_position = 0;
    end_of_file = false;
    paused = false;
    std::scoped_lock clock_lock{clock_mutex};
    start_time.reset();
    last_audio_ms.reset();
    last_audio_time.reset();
    paused_for = {};
}

void Source::Pause() {
    std::scoped_lock lock{clock_mutex};
    pause_start = Clock::now();
    paused = true;
}

void Source::Resume() {
    std::scoped_lock lock{clock_mutex};
    paused_for += Clock::now() - pause_start;
    paused = false;
}

void Source::DemuxThread(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:AvDemux");
    bool read_since_rewind = true; // a file that ends at once does not loop forever
    while (!stop.stop_requested()) {
        // Enough read ahead: about a second of video and a few audio packets.
        if (video.packets.Size() > 30 && (!audio.Enabled() || audio.packets.Size() > 8)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        PacketPtr packet{av_packet_alloc()};
        const int result = av_read_frame(format, packet.get());
        if (result == AVERROR_EOF && looping && read_since_rewind) {
            // From the start again; the clock restarts with it.
            listener.OnWarning(WarningLoopingBack);
            Rewind();
            read_since_rewind = false;
            std::scoped_lock lock{clock_mutex};
            start_time = Clock::now();
            paused_for = {};
            continue;
        }
        if (result < 0) {
            if (result != AVERROR_EOF) {
                LOG_ERROR(Lib_AvPlayer, "demuxing failed: {}", AvError(result));
                listener.OnError();
            }
            break;
        }
        read_since_rewind = true;
        if (packet->stream_index == video.ffmpeg_index) {
            video.packets.Push(std::move(packet));
        } else if (packet->stream_index == audio.ffmpeg_index) {
            audio.packets.Push(std::move(packet));
        }
    }
    end_of_file = true;
    WakeAll();
}

void Source::DecodeThread(std::stop_token stop, Track& track, bool is_video) {
    Common::SetCurrentThreadName(is_video ? "bb:AvVideoDec" : "bb:AvAudioDec");
    FramePtr decoded{av_frame_alloc()};
    // Takes what the decoder has ready: false on an error or once it is drained.
    const auto receive = [&]() -> bool {
        while (!stop.stop_requested()) {
            // A free buffer first: the decoder holds its frame until there is room.
            if (!track.free_buffers.WaitNotEmpty(stop, [] { return false; })) {
                return false;
            }
            const int result = avcodec_receive_frame(track.codec, decoded.get());
            if (result == AVERROR(EAGAIN)) {
                return true; // wants the next packet
            }
            if (result < 0) {
                if (result != AVERROR_EOF) {
                    LOG_ERROR(Lib_AvPlayer, "decoding failed: {}", AvError(result));
                    listener.OnError();
                }
                return false;
            }
            const bool converted = ToFrame(track, is_video, *decoded);
            av_frame_unref(decoded.get());
            if (!converted) {
                listener.OnError();
                return false;
            }
        }
        return false;
    };
    while (!stop.stop_requested()) {
        if (!track.packets.WaitNotEmpty(stop, [this] { return end_of_file.load(); })) {
            break; // stopped, or the file ended and every packet is in the decoder
        }
        const auto packet = track.packets.Pop();
        const int result = avcodec_send_packet(track.codec, packet->get());
        if (result < 0 && result != AVERROR(EAGAIN)) {
            LOG_ERROR(Lib_AvPlayer, "decoding failed: {}", AvError(result));
            listener.OnError();
            return;
        }
        if (!receive()) {
            return;
        }
    }
    if (stop.stop_requested()) {
        return;
    }
    // The frames the decoder still holds (reordered ones), then the end.
    avcodec_send_packet(track.codec, nullptr);
    receive();
    if (--decoders_running == 0) {
        listener.OnEndOfFile();
    }
}

u64 Source::TimestampMillis(const AVFrame& frame, int ffmpeg_index) const {
    s64 timestamp = frame.best_effort_timestamp;
    if (timestamp == AV_NOPTS_VALUE) {
        timestamp = frame.pts;
    }
    if (timestamp == AV_NOPTS_VALUE) {
        timestamp = frame.pkt_dts;
    }
    const AVRational time_base = format->streams[ffmpeg_index]->time_base;
    if (timestamp == AV_NOPTS_VALUE || timestamp < 0 || time_base.den <= 0) {
        return 0;
    }
    return u64(std::max<s64>(0, av_rescale_q(timestamp, time_base, AVRational{1, 1000})));
}

bool Source::ToFrame(Track& track, bool is_video, const AVFrame& decoded) {
    auto buffer = track.free_buffers.Pop();
    if (!buffer) {
        return true; // stopped meanwhile
    }
    AvPlayerFrameInfoEx info{};
    info.p_data = buffer->Data();
    info.timestamp = TimestampMillis(decoded, track.ffmpeg_index);
    if (is_video) {
        // NV12 as the decoders give it; anything else converted first.
        const AVFrame* nv12 = &decoded;
        FramePtr converted;
        if (decoded.format != AV_PIX_FMT_NV12) {
            converted.reset(av_frame_alloc());
            converted->format = AV_PIX_FMT_NV12;
            converted->width = decoded.width;
            converted->height = decoded.height;
            av_frame_get_buffer(converted.get(), 0);
            if (!sws) {
                sws = sws_getContext(decoded.width, decoded.height, AVPixelFormat(decoded.format),
                                     decoded.width, decoded.height, AV_PIX_FMT_NV12,
                                     SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
            }
            if (sws_scale(sws, decoded.data, decoded.linesize, 0, decoded.height, converted->data,
                          converted->linesize) < 0) {
                LOG_ERROR(Lib_AvPlayer, "could not convert a video frame to NV12");
                return false;
            }
            nv12 = converted.get();
        }
        const u32 pitch = Common::AlignUp<u32>(u32(decoded.width), 64);
        const u32 height = Common::AlignUp<u32>(u32(decoded.height), 16);
        CopyNv12(buffer->Data(), buffer->Size(), *nv12, pitch, height);
        // The GPU side may have this memory cached as a texture.
        Core::Memory::Instance()->InvalidateMemory(reinterpret_cast<VAddr>(buffer->Data()),
                                                   u64(pitch) * height * 3 / 2);
        auto& details = info.details.video;
        details.width = Common::AlignUp<u32>(u32(decoded.width), 16);
        details.height = height;
        details.aspect_ratio = f32(av_q2d(decoded.sample_aspect_ratio));
        details.crop_left_offset = u32(decoded.crop_left);
        details.crop_right_offset = u32(decoded.crop_right + (pitch - u32(decoded.width)));
        details.crop_top_offset = u32(decoded.crop_top);
        details.crop_bottom_offset = u32(decoded.crop_bottom + (height - u32(decoded.height)));
        details.pitch = pitch;
        details.luma_bit_depth = 8;
        details.chroma_bit_depth = 8;
    } else {
        // Interleaved 16-bit PCM.
        const AVFrame* pcm = &decoded;
        FramePtr converted;
        if (decoded.format != AV_SAMPLE_FMT_S16) {
            if (!swr) {
                swr_alloc_set_opts2(&swr, &decoded.ch_layout, AV_SAMPLE_FMT_S16,
                                    decoded.sample_rate, &decoded.ch_layout,
                                    AVSampleFormat(decoded.format), decoded.sample_rate, 0,
                                    nullptr);
                swr_init(swr);
            }
            converted.reset(av_frame_alloc());
            converted->format = AV_SAMPLE_FMT_S16;
            converted->ch_layout = decoded.ch_layout;
            converted->sample_rate = decoded.sample_rate;
            if (swr_convert_frame(swr, converted.get(), &decoded) < 0) {
                LOG_ERROR(Lib_AvPlayer, "could not convert an audio frame to 16-bit PCM");
                return false;
            }
            pcm = converted.get();
        }
        const u64 bytes = u64(pcm->ch_layout.nb_channels) * u64(pcm->nb_samples) * sizeof(s16);
        const u64 size = std::min(bytes, buffer->Size());
        std::memcpy(buffer->Data(), pcm->data[0], size);
        auto& details = info.details.audio;
        details.channel_count = u16(pcm->ch_layout.nb_channels);
        details.sample_rate = u32(pcm->sample_rate);
        details.size = u32(size);
    }
    track.frames.Push(Frame{std::move(*buffer), info});
    return true;
}

// ---- Frames for the game ---------------------------------------------------------------------

bool Source::IsActive() {
    // A decoder still at work (its last packets, the frames it holds back) counts.
    return !end_of_file || decoders_running > 0 || video.packets.Size() != 0 ||
           audio.packets.Size() != 0 || video.frames.Size() != 0 || audio.frames.Size() != 0;
}

u64 Source::CurrentTime() {
    std::scoped_lock lock{clock_mutex};
    if (!start_time) {
        return 0;
    }
    if (!IsActive()) {
        return end_of_file ? duration : 0;
    }
    const auto now = paused ? pause_start : Clock::now();
    using std::chrono::duration_cast;
    using std::chrono::milliseconds;
    if (audio.Enabled()) {
        // The audio clock: the last frame the game took, plus the time since.
        const auto since = now - last_audio_time.value_or(now) - paused_for;
        return last_audio_ms.value_or(0) + u64(std::max<s64>(0, duration_cast<milliseconds>(since).count()));
    }
    const s64 elapsed = duration_cast<milliseconds>(now - *start_time - paused_for).count();
    if (elapsed <= 0) {
        return 0;
    }
    return duration != 0 ? std::min(u64(elapsed), duration) : u64(elapsed);
}

bool Source::GetVideoData(AvPlayerFrameInfoEx& info) {
    if (!started || !IsActive() || paused) {
        return false;
    }
    const auto next = video.frames.PeekWith([](const Frame& frame) { return frame.info.timestamp; });
    if (!next) {
        return false;
    }
    // In sync, a frame waits for its time.
    if (listener.SyncMode() == AvPlayerAvSyncMode::Default && *next > CurrentTime()) {
        return false;
    }
    if (video.current) {
        video.free_buffers.Push(std::move(video.current->buffer));
        video.current.reset();
    }
    video.current = video.frames.Pop();
    info = video.current->info;
    return true;
}

bool Source::GetVideoData(AvPlayerFrameInfo& info) {
    AvPlayerFrameInfoEx ex{};
    if (!GetVideoData(ex)) {
        return false;
    }
    info = {};
    info.p_data = static_cast<u8*>(ex.p_data);
    info.timestamp = ex.timestamp;
    info.details.video.width = ex.details.video.width;
    info.details.video.height = ex.details.video.height;
    info.details.video.aspect_ratio = ex.details.video.aspect_ratio;
    return true;
}

bool Source::GetAudioData(AvPlayerFrameInfo& info) {
    if (!started || !IsActive() || paused) {
        return false;
    }
    auto frame = audio.frames.Pop();
    if (!frame) {
        return false;
    }
    if (audio.current) {
        audio.free_buffers.Push(std::move(audio.current->buffer));
        audio.current.reset();
    }
    {
        std::scoped_lock lock{clock_mutex};
        last_audio_ms = frame->info.timestamp;
        last_audio_time = Clock::now();
        paused_for = {};
    }
    info = {};
    info.p_data = static_cast<u8*>(frame->info.p_data);
    info.timestamp = frame->info.timestamp;
    info.details.audio.channel_count = frame->info.details.audio.channel_count;
    info.details.audio.sample_rate = frame->info.details.audio.sample_rate;
    info.details.audio.size = frame->info.details.audio.size;
    audio.current = std::move(frame);
    return true;
}

} // namespace BbAvPlayer
