// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, the two-stage draw pipeline: stage A (GPU command thread) packs each draw
// (dirty registers, user data, constants) and stage B (draw recording thread) records it.

#include "render/raster_internal.h"

namespace Vulkan {

// bbport: two-stage draw pipeline (vk_draw_pipe.h). Packet layout, 8-byte aligned parts:
// DrawPacket, u16 block indices, u32 block words (RegDirty::BlockWords each), then per stage a
// PacketStage with its user data and flattened user data, then (verification) the full file.
namespace {
enum class PacketKind : u32 { Draw, Task, Dispatch, Special, Indirect, IndirectDispatch };
struct TaskPacket {
    PacketKind kind;
    u32 size;
    Rasterizer::OrderedTask task;
};
struct DrawPacket {
    PacketKind kind;
    const Pipeline* pipeline;
    const PreparedDraw* prepared;
    u32 index_offset;
    u8 is_indexed;
    u8 regs_reset;
    u16 num_blocks;
    u32 num_stages;
    u32 verify;
    std::array<Gcn::CbDbExtent, Gcn::NUM_COLOR_BUFFERS> cb_extent;
    Gcn::CbDbExtent db_extent;
    u32 num_vsharps; ///< V#s of the vertex streams, after the indirect arguments
    u32 worker; ///< the resolve worker that copied constants for it (~0u: none)
    u64 ring_end; ///< constant ring position after this packet's copies
    u64 worker_ring_end; ///< the same in that worker's ring
};
struct PacketStage {
    const Shader::Info* info;
    VAddr pgm_base;
    u32 ud_size;
    u32 flat_size;
    u32 num_ring;
    u32 pad;
};
constexpr u32 AlignPacket(u32 size) {
    return (size + 7) & ~7u;
}
u32 VerifyInterval() {
    static const u32 interval = [] {
        const char* env = std::getenv("BB_PIPE_VERIFY");
        return env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    return interval;
}
} // namespace

bool Rasterizer::DrawPipeWanted() {
    const char* env = std::getenv("BB_DRAW_PIPE");
    if (env && env[0]) {
        return env[0] == '1';
    }
    // Stage B spins while draws flow. Measured ahead with 16 threads (+19%) and with 4 cores /
    // 8 threads (taskset, Steam Deck-like: +18%).
    return BbThreads::Available() >= 8;
}

bool Rasterizer::UseDrawPipe() const {
    return draw_pipe && !host_markers_enabled && !BbToggle::Disabled(BbToggle::DrawPipeline);
}

bool Rasterizer::OnStageA() const {
    return std::this_thread::get_id() == liverpool->GetGpuCommandProcessorThread();
}

namespace {
/// Stage A: cycles waited per drain site (BB_FRAME_STATS), keyed by function name and line.
std::unordered_map<const char*, std::unordered_map<u32, u64>> drain_sites;
} // namespace

void Rasterizer::DrainDrawPipe(u32 reason, u32 line, const char* function) {
    if (!draw_pipe || !OnStageA()) {
        return;
    }
    static const bool stats = std::getenv("BB_FRAME_STATS") != nullptr;
    if (!stats) {
        draw_pipe->Drain(reason);
        return;
    }
    const u64 before = draw_pipe->drain_cycles;
    draw_pipe->Drain(reason);
    if (const u64 waited = draw_pipe->drain_cycles - before) {
        drain_sites[function][line] += waited;
    }
}

bool Rasterizer::IsGpuSideThread() const {
    return OnStageA() || DrawPipe::OnStageB();
}

bool Rasterizer::IsGpuSideThreadId(u32 tid) const {
    // The userfaultfd thread handles the fault while the faulting thread waits: without locks
    // only if that thread owns the caches. The GPU command thread does not while the draw
    // recording thread may be running, so its faults take the locked path of guest threads.
    if (draw_pipe) {
        return tid == draw_pipe->StageBThreadId();
    }
    return tid == liverpool->GetGpuCommandProcessorThreadId();
}

void Rasterizer::NotePendingGpuWrite(VAddr address, u64 size) {
    if (!draw_pipe || !size) {
        return;
    }
    // Numbered by packet (a position, the packet's start, counted as reached once the packet
    // before it had run, while this one was not recorded yet), in hashed granules: a lookup per
    // constant binding instead of a scan of the writes (a list scan was ~5% of this thread).
    pending_write_table.Note(address, size, draw_pipe->NextPacket());
}

void Rasterizer::NotePendingRead(VAddr address, u64 size) {
    if (draw_pipe && size && OnStageA()) {
        pending_reads.Note(address, size, draw_pipe->NextPacket());
    }
}

namespace {
/// BB_CONSTRAM_PIPE=0: a DumpConstRam waits for all queued draws, not only for those reading its
/// bytes.
bool ConstRamPipe() {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_CONSTRAM_PIPE");
        return !env || env[0] != '0';
    }();
    return enabled;
}
} // namespace

void Rasterizer::WaitForPendingReads(VAddr address, u64 size) {
    if (!draw_pipe || !OnStageA()) {
        return;
    }
    // BB_EXP_CONSTRAM_FINISH=1 (experiment): every dump waits until the GPU has run all earlier work.
    // 2: stage B catches up and submits, no GPU wait; 3: waits for the GPU only (work submitted).
    static const int finish_all = [] {
        const char* env = std::getenv("BB_EXP_CONSTRAM_FINISH");
        return env ? std::atoi(env) : 0;
    }();
    if (finish_all == 1) {
        DrainDrawPipe(DrawPipe::ReasonConstRam);
        RunInOrder([](Rasterizer& self, const u8*) { self.Finish(); }, nullptr, 0,
                   BbToggle::PipelinedTasks, false);
        DrainDrawPipe(DrawPipe::ReasonConstRam);
        return;
    }
    if (finish_all == 2) {
        DrainDrawPipe(DrawPipe::ReasonConstRam);
        RunInOrder([](Rasterizer& self, const u8*) { self.scheduler.Flush(); }, nullptr, 0,
                   BbToggle::PipelinedTasks, false);
        DrainDrawPipe(DrawPipe::ReasonConstRam);
        return;
    }
    if (finish_all == 3) {
        auto* semaphore = scheduler.GetWorkSemaphore();
        semaphore->Refresh();
        const u64 submitted = semaphore->CurrentTick() - 1; // the last tick sent
        semaphore->Wait(submitted);
    }
    // Stage 2: the resolve workers copy (or note as read in place) the constants of the packets
    // reading these bytes before the dump replaces them.
    if (size != 0 && draw_pipe->ResolverCount() != 0) {
        if (const u64 resolving = pending_resolve.Newest(address, size); resolving != 0) {
            draw_pipe->WaitResolved(resolving);
        }
    }
    const u64 packet =
        size == 0 ? 0 : std::max(untracked_reads_packet, pending_reads.Newest(address, size));
    if (!ConstRamPipe() || !UseDrawPipe() || size == 0) {
        DrainDrawPipe(DrawPipe::ReasonConstRam);
    } else if (draw_pipe->ReachedPacket(packet)) {
        ++constram_waits_skipped;
    } else {
        draw_pipe->WaitForPacket(packet, DrawPipe::ReasonConstRam);
    }
    WaitForInPlaceReads(packet);
    if (InPlaceReads::Enabled() && size != 0) {
        auto* semaphore = scheduler.GetWorkSemaphore();
        semaphore->Refresh();
        InPlaceReads::CheckDump(address, size, [semaphore](u64 tick) { return semaphore->IsFree(tick); });
        // Images whose memory this dump writes (read by the GPU when it uploads them).
        {
            static std::atomic<u32> image_reports{0};
            texture_cache.ForEachImageInRegionLocked(address, size, [&](VideoCore::ImageId, VideoCore::Image& image) {
                if (image_reports.fetch_add(1) < 300) {
                    std::printf("In-place read check: frame %llu dump %#llx+%llu over image %#llx+%#llx %ux%u fmt %u flags %#x\n",
                                (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed),
                                (unsigned long long)address, (unsigned long long)size,
                                (unsigned long long)image.info.guest_address,
                                (unsigned long long)image.info.guest_size, image.info.size.width,
                                image.info.size.height, u32(image.info.pixel_format), u32(image.flags));
                }
            });
        }
        static std::atomic<u32> write_reports{0};
        const u64 write_packet = pending_write_table.Newest(address, size);
        if (buffer_cache.IsRegionGpuModified(address, size) && write_reports.fetch_add(1) < 400) {
            std::printf("In-place read check: frame %llu dump %#llx+%llu over a pending GPU write "
                        "(write packet %llu reached %d)\n",
                        (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed),
                        (unsigned long long)address, (unsigned long long)size,
                        (unsigned long long)write_packet, int(draw_pipe->ReachedPacket(write_packet)));
        }
    }
}

void Rasterizer::WaitForInPlaceReads(u64 packet) {
    // With the game's memory in place the draw reads these bytes when the GPU runs it, not when
    // the recording thread binds them (a copy then, with the old model): a dump while it is still
    // queued on the GPU gave it the next draw's data. Single frames showed another pass's quad:
    // the finished picture in the top-left quarter, black around it, or the scene shifted by half
    // the frame, bright around it (docs/Flicker_video_2026-10-17.ru.md). Rare: ~1 of 1000 dumps.
    // BB_CONSTRAM_GPU_WAIT=0: not waited.
    static const bool enabled = [] {
        const char* env = std::getenv("BB_CONSTRAM_GPU_WAIT");
        return VideoCore::GuestInPlace() && !(env && env[0] == '0');
    }();
    if (!enabled || packet == 0) {
        return;
    }
    const auto& entry = (*packet_ticks)[packet & 0xFFFF];
    if (entry.packet.load(std::memory_order_acquire) != packet) {
        return; // older than the table: long run
    }
    const u64 tick = entry.tick.load(std::memory_order_relaxed);
    auto* semaphore = scheduler.GetWorkSemaphore();
    if (semaphore->IsFree(tick)) {
        return;
    }
    semaphore->Refresh();
    if (semaphore->IsFree(tick)) {
        return;
    }
    // The recording thread submits the tick if it still records into it, then the GPU runs it.
    struct Data {
        u64 tick;
    };
    const Data data{tick};
    RunInOrder(
        [](Rasterizer& self, const u8* bytes) {
            Data d;
            std::memcpy(&d, bytes, sizeof(d));
            if (self.scheduler.CurrentTick() <= d.tick) {
                self.scheduler.Flush();
            }
        },
        &data, sizeof(data), BbToggle::PipelinedTasks, false);
    DrainDrawPipe(DrawPipe::ReasonConstRam);
    const u64 start = BbCpu::Ticks();
    semaphore->Wait(tick);
    ++constram_gpu_waits;
    constram_gpu_wait_cycles += BbCpu::Ticks() - start;
}

void Rasterizer::NoteCommandWriteInOrder(VAddr addr, const void* data, u64 size) {
    const bool enabled = ConstRamPipe();
    if (VideoCore::WriteTracking() || !IsMapped(addr, size)) {
        return;
    }
    if (!enabled || !UseDrawPipe() || !OnStageA() || size > 1_MB) {
        NoteCommandWrite(addr, data, size, false);
        return;
    }
    // The caches belong to the recording thread while it has packets: it updates them in order
    // (later packets see the new bytes there; the memory already holds them for this thread).
    struct Header {
        VAddr addr;
        u64 size;
    };
    thread_local std::vector<u8> payload;
    payload.resize(sizeof(Header) + size);
    const Header header{addr, size};
    std::memcpy(payload.data(), &header, sizeof(header));
    std::memcpy(payload.data() + sizeof(header), data, size);
    RunInOrder(
        [](Rasterizer& self, const u8* bytes) {
            Header write;
            std::memcpy(&write, bytes, sizeof(write));
            self.NoteCommandWrite(write.addr, bytes + sizeof(write), write.size, true);
        },
        payload.data(), static_cast<u32>(payload.size()), BbToggle::PipelinedTasks, false);
}

bool Rasterizer::PendingWriteOverlaps(VAddr address, u64 size) {
    if (!draw_pipe || size == 0) {
        return false;
    }
    const u64 packet = pending_write_table.Newest(address, size);
    return packet != 0 && !draw_pipe->ReachedPacket(packet);
}

void Rasterizer::CollectRingBindings(const Shader::Info& stage, const PreparedDraw* prepared,
                           boost::container::static_vector<RingBinding, Shader::NUM_BUFFERS>& out,
                           const Gcn::ComputeProgram* cs) {
    BB_SECTION(ARing);
    const PreparedStage* prepared_stage = nullptr;
    if (prepared && !BbToggle::Disabled(BbToggle::PreparedResources)) {
        for (u32 i = 0; i < prepared->num_stages; ++i) {
            const auto& candidate = prepared->stages[i];
            if (&candidate.program->info == &stage &&
                candidate.num_buffers == stage.buffers.size()) {
                prepared_stage = &candidate;
                break;
            }
        }
    }
    const u64 alignment = std::max<u64>(instance.StorageMinAlignment(),
                                        instance.UniformMinAlignment());
    const auto copy = [&](u32 index, const void* source, VAddr address, u64 size) {
        const auto offset = constant_ring->Allocate(size, alignment);
        if (!offset) {
            return false;
        }
        BB_SECTION(ARingCopy);
        if (BbSections::Enabled()) {
            BbSections::ring_copies.fetch_add(1, std::memory_order_relaxed);
            BbSections::ring_bytes.fetch_add(size, std::memory_order_relaxed);
        }
        u8* dst = constant_ring->Data(*offset);
        if (source) {
            std::memcpy(dst, source, size);
        } else {
            memory->CopySparseMemory(address, dst, size);
        }
        constant_ring->Flush(*offset, size);
        out.push_back({index, static_cast<u32>(size), *offset, address, 0, 0});
        return true;
    };
    // Stage 2: the checks and copies of small read-only buffers go to the resolve workers; a
    // constant RAM dump over them waits for those first (pending_resolve).
    const bool resolvers = draw_pipe && draw_pipe->ResolverCount() != 0;
    // Compute stages may run as HLE copy shaders, which read their buffers on the recording
    // thread whether or not they were copied here.
    const bool hle_candidate = stage.sw_stage == Shader::SwStage::Compute;
    // BB_CONSTANTS_IN_PLACE: nothing is copied here; the recording thread binds the game's memory
    // (or copies what is not in place, a read noted below).
    const bool constants_in_place = VideoCore::ConstantsInPlace();
    for (u32 index = 0; index < stage.buffers.size(); ++index) {
        const auto& desc = stage.buffers[index];
        if (desc.IsSpecial()) {
            if (desc.buffer_type == Shader::BufferType::Flatbuf &&
                !stage.flattened_ud_buf.empty()) {
                copy(index, stage.flattened_ud_buf.data(), 0,
                     stage.flattened_ud_buf.size() * sizeof(u32));
            }
            continue;
        }
        VAddr address;
        u64 size;
        {
            BB_SECTION(ARingSharp);
            const auto vsharp = prepared_stage ? prepared_stage->buffer_sharps[index]
                                               : desc.GetSharp(stage);
            address = vsharp.base_address;
            if (address == 0 || vsharp.GetSize() == 0) {
                continue;
            }
            size = memory->ClampRangeSize(address, vsharp.GetSize());
        }
        if (desc.is_written) {
            // The game's copy shader: what its control list names, not the tens of MiB bound (they
            // kept every constant around them from the ring: bound in place, a later constant RAM
            // dump changed them before the GPU read them, single frames drew another draw's quad).
            thread_local std::vector<std::pair<unsigned long long, unsigned long long>> dests;
            dests.clear();
            if (cs && CopyDestinationsPrecise() &&
                CopyShaderDestinations(stage, cs->dim_x, CopyDestinationGap(), dests)) {
                for (const auto& [dest, dest_size] : dests) {
                    NotePendingGpuWrite(dest, dest_size);
                }
            } else {
                NotePendingGpuWrite(address, size);
            }
            continue;
        }
        if (BbCeStats::Enabled()) {
            BbCeStats::Check(size <= VideoCore::BufferCache::STREAM_THRESHOLD
                                 ? BbCeStats::RingConstants
                                 : BbCeStats::LargeBuffer,
                             address, size);
        }
        // The stream path of BufferCache::ObtainBuffer, taken here: small, read-only, not
        // written by the GPU (now or by work still queued for the recording thread). What is not
        // copied is read on the recording thread: a CPU write over these bytes on this thread
        // waits for the packet (WaitForPendingReads).
        if (BbSections::Enabled()) {
            BbSections::ring_buffers.fetch_add(1, std::memory_order_relaxed);
        }
        if (resolvers && size != 0 && size <= VideoCore::BufferCache::STREAM_THRESHOLD &&
            !constants_in_place) {
            out.push_back({index, static_cast<u32>(size), RingPending, address, 0,
                           u32(hle_candidate)});
            pending_resolve.Note(address, size, draw_pipe->NextPacket());
            continue;
        }
        bool copyable;
        {
            BB_SECTION(ARingChecks);
            copyable = size != 0 && size <= VideoCore::BufferCache::STREAM_THRESHOLD &&
                       !constants_in_place && !buffer_cache.IsRegionGpuModified(address, size) &&
                       !PendingWriteOverlaps(address, size) &&
                       !VideoCore::BufferCache::LayerMirrored(address, size);
        }
        bool copied = false;
        if (!copyable || !(copied = copy(index, nullptr, address, size)) || hle_candidate) {
            BB_SECTION(ARingNote);
            NotePendingRead(address, size);
        }
        // BB_TRACE_CB=hash (diagnostics): how that shader's constant buffers were bound.
        static const u64 traced_cb = [] {
            const char* env = std::getenv("BB_TRACE_CB");
            return env ? (std::strcmp(env, "all") == 0 ? ~0ull : std::strtoull(env, nullptr, 16)) : 0ull;
        }();
        if (traced_cb == ~0ull && !copied && size <= VideoCore::BufferCache::STREAM_THRESHOLD) {
            static std::mutex trace_mutex;
            static std::map<std::pair<u64, u32>, u64> in_place;
            static u64 last_frame = 0;
            std::scoped_lock lock{trace_mutex};
            const u32 why = buffer_cache.IsRegionGpuModified(address, size) ? 1
                            : PendingWriteOverlaps(address, size)          ? 2
                            : VideoCore::BufferCache::LayerMirrored(address, size) ? 3
                                                                             : 4;
            ++in_place[{stage.pgm_hash, why}];
            const u64 frame = BbStats::frame_number.load(std::memory_order_relaxed);
            if (frame >= last_frame + 600) {
                last_frame = frame;
                std::printf("Trace cb: frame %llu small constants in place (hash why count):", (unsigned long long)frame);
                for (const auto& [key, count] : in_place) {
                    std::printf(" %llx/%u/%llu", (unsigned long long)key.first, key.second, (unsigned long long)count);
                }
                std::printf("\n");
                in_place.clear();
            }
        }
        // BB_TRACE_CB_LIST=hash,hash (diagnostics): their constants against the same draw's last frame.
        static const std::vector<u64> traced_list = [] {
            std::vector<u64> hashes;
            if (const char* env = std::getenv("BB_TRACE_CB_LIST")) {
                for (const char* at = env; *at;) {
                    char* end = nullptr;
                    hashes.push_back(std::strtoull(at, &end, 16));
                    at = *end ? end + 1 : end;
                }
            }
            return hashes;
        }();
        const bool listed = std::ranges::find(traced_list, stage.pgm_hash) != traced_list.end();
        if ((traced_cb && stage.pgm_hash == traced_cb) || listed) {
            static std::map<std::tuple<u64, u32, u32>, std::vector<u32>> history;
            static std::map<std::pair<u64, u32>, u32> nth;
            static u64 nth_frame = 0;
            const u64 gpu_frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
            if (gpu_frame != nth_frame) {
                nth_frame = gpu_frame;
                nth.clear();
            }
            const u32 ordinal = nth[{stage.pgm_hash, index}]++;
            auto& previous = history[{stage.pgm_hash, index, ordinal}];
            std::vector<u32> now(size / 4);
            std::memcpy(now.data(), reinterpret_cast<const void*>(address), now.size() * 4);
            std::string changes;
            u32 changed = 0;
            for (size_t i = 0; i < now.size() && i < previous.size(); ++i) {
                if (now[i] != previous[i] && changed++ < 24) {
                    changes += fmt::format(" [{}] {:x}({:g})->{:x}({:g})", i, previous[i], std::bit_cast<float>(previous[i]), now[i], std::bit_cast<float>(now[i]));
                }
            }
            std::printf("Trace cb values: gpu frame %llu %llx#%u slot %u %#llx %s changed %u:%s\n",
                        (unsigned long long)gpu_frame, (unsigned long long)stage.pgm_hash, ordinal,
                        index, (unsigned long long)address, copied ? "copied" : "in place", changed, changes.c_str());
            previous = std::move(now);
            std::printf("Trace cb: frame %llu %016llx slot %u %#llx+%llu %s gpu-mod %d pending-write %d "
                        "mirrored %d packet %llu\n",
                        (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed),
                        (unsigned long long)stage.pgm_hash, index, (unsigned long long)address,
                        (unsigned long long)size, copied ? "copied" : "IN PLACE",
                        int(buffer_cache.IsRegionGpuModified(address, size)),
                        int(PendingWriteOverlaps(address, size)),
                        int(VideoCore::BufferCache::LayerMirrored(address, size)),
                        (unsigned long long)(draw_pipe ? draw_pipe->NextPacket() : 0));
        }
        // BB_RING_REASONS=1 (diagnostics): why small read-only bindings were not copied, per 5 s.
        static const bool reasons = std::getenv("BB_RING_REASONS") != nullptr;
        if (reasons && !copyable && size != 0 && size <= VideoCore::BufferCache::STREAM_THRESHOLD) {
            static std::array<std::atomic<u64>, 5> counts{};
            static u32 samples = 0;
            static std::atomic<u64> total{0};
            static auto last = std::chrono::steady_clock::now();
            ++counts[constants_in_place                          ? 0
                     : buffer_cache.IsRegionGpuModified(address, size) ? (samples++ < 300 && std::printf("Ring GPU-modified sample %#llx+%llu\n", (unsigned long long)address, (unsigned long long)size), 1)
                     : PendingWriteOverlaps(address, size)         ? 2
                     : VideoCore::BufferCache::LayerMirrored(address, size) ? 3 : 4];
            ++total;
            if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) {
                last = std::chrono::steady_clock::now();
                std::printf("Ring reasons (5 s): in place %llu, GPU-modified %llu, pending GPU write %llu, "
                            "mirrored %llu, other %llu\n",
                            (unsigned long long)counts[0].exchange(0), (unsigned long long)counts[1].exchange(0),
                            (unsigned long long)counts[2].exchange(0), (unsigned long long)counts[3].exchange(0),
                            (unsigned long long)counts[4].exchange(0));
            }
        }
    }
}

void Rasterizer::PostDraw(const Pipeline* pipeline, const PreparedDraw* used_prepared,
                          bool is_indexed, u32 index_offset, const Gcn::ComputeProgram* cs,
                          const IndirectDraw* indirect) {
    BB_SECTION(APostDraw);
    auto& dirty = liverpool->pipe_dirty;
    if (!pipe_synced) {
        // Stage B has not run yet: its register copy starts as this one.
        pipe_regs = liverpool->regs;
        dirty.Clear();
        pipe_synced = true;
    }
    std::array<u16, Gcn::RegDirty::NumBlocks> blocks;
    u32 num_blocks = 0;
    dirty.ForEachBlock([&](size_t block) { blocks[num_blocks++] = static_cast<u16>(block); });
    const auto stages =
        pipeline ? pipeline->GetStages() : std::span<const Shader::Info* const>{};
    // Constants: copied here into the ring, the recording thread only binds them.
    thread_local std::array<boost::container::static_vector<RingBinding, Shader::NUM_BUFFERS>,
                            Shader::MaxStageTypes>
        rings;
    u32 num_stages = 0;
    bool resolve = false; // constants left to a resolve worker
    u32 size = sizeof(DrawPacket) + AlignPacket(num_blocks * sizeof(u16)) +
               num_blocks * Gcn::RegDirty::BlockWords * sizeof(u32);
    for (const auto* stage : stages) {
        if (stage) {
            auto& ring = rings[num_stages];
            if (BbCeStats::Enabled()) {
                const auto ud = stage->UserData();
                for (size_t i = 0; i + 1 < ud.size(); ++i) {
                    const u64 value = u64(ud[i]) | u64(ud[i + 1]) << 32;
                    if (value >= 0x100000000ull && value < (1ull << 40)) {
                        BbCeStats::Check(BbCeStats::UserDataPointer, value, 64);
                    }
                }
                BbCeStats::Get().checks[BbCeStats::DmaStage]++;
                BbCeStats::Get().hits[0][BbCeStats::DmaStage] += stage->uses_dma ? 1 : 0;
            }
            // bbport BB_WATCH=addr,size (diagnostics): what the translator reads from that range on
            // the CPU: user data pointing into it, constants copied from it; printed every 2 s.
            static const std::pair<u64, u64> watch = [] {
                const char* env = std::getenv("BB_WATCH");
                if (!env) {
                    return std::pair<u64, u64>{0, 0};
                }
                char* end = nullptr;
                const u64 addr = std::strtoull(env, &end, 0);
                const u64 len = end && *end == ',' ? std::strtoull(end + 1, nullptr, 0) : 0;
                return std::pair<u64, u64>{addr, len};
            }();
            if (watch.second) {
                static u64 ud_hits = 0, ud_checks = 0, ring_hits = 0, vs_hits = 0;
                static u64 code_hits = 0;
                if (stage->ProgramBase() >= watch.first && stage->ProgramBase() < watch.first + watch.second) {
                    ++code_hits;
                }
                static u64 last_hash = 0;
                static auto report = std::chrono::steady_clock::now();
                const auto ud = stage->UserData();
                ++ud_checks;
                for (size_t i = 0; i + 1 < ud.size(); ++i) {
                    const u64 value = (u64(ud[i]) | u64(ud[i + 1]) << 32) & 0xFFFFFFFFFFFFull;
                    if (value >= watch.first && value < watch.first + watch.second) {
                        ++ud_hits;
                        last_hash = stage->pgm_hash;
                    }
                }
                for (u32 index = 0; index < stage->buffers.size(); ++index) {
                    const auto& desc = stage->buffers[index];
                    if (desc.IsSpecial()) {
                        continue;
                    }
                    const auto vsharp = desc.GetSharp(*stage);
                    if (vsharp.base_address < watch.first + watch.second &&
                        watch.first < vsharp.base_address + vsharp.GetSize()) {
                        ++(vsharp.GetSize() <= VideoCore::BufferCache::STREAM_THRESHOLD ? ring_hits
                                                                                      : vs_hits);
                        last_hash = stage->pgm_hash;
                    }
                }
                if (std::chrono::steady_clock::now() - report > std::chrono::seconds(2)) {
                    report = std::chrono::steady_clock::now();
                    std::printf("Watch %#llx+%#llx: user data pointers into it %llu of %llu stages, "
                                "small buffers over it %llu, large %llu, shader code in it %llu (last shader %016llx)\n",
                                (unsigned long long)watch.first, (unsigned long long)watch.second,
                                (unsigned long long)ud_hits, (unsigned long long)ud_checks,
                                (unsigned long long)ring_hits, (unsigned long long)vs_hits,
                                (unsigned long long)code_hits,
                                (unsigned long long)last_hash);
                    ud_hits = ud_checks = ring_hits = vs_hits = code_hits = 0;
                }
            }
            // BB_EXP_DMA_UNTRACKED=1 (experiment): a stage that reads the game's memory through
            // pointers when the GPU runs it (uses_dma) counts as reading memory it does not list:
            // a constant RAM dump waits for the GPU to have run it.
            static const bool dma_untracked = std::getenv("BB_EXP_DMA_UNTRACKED") != nullptr;
            if (stage->uses_dma) {
                static std::atomic<u64> dma_stages{0};
                if ((dma_stages.fetch_add(1, std::memory_order_relaxed) & 0xFFFF) == 0) {
                    std::printf("Draw pipe: %llu stages reading memory through pointers (uses_dma), "
                                "last %016llx\n",
                                (unsigned long long)dma_stages.load(), (unsigned long long)stage->pgm_hash);
                }
                if (dma_untracked) {
                    untracked_reads_packet = draw_pipe->NextPacket();
                }
            }
            ring.clear();
            if (constant_ring && !BbToggle::Disabled(BbToggle::ConstantRing)) {
                CollectRingBindings(*stage, used_prepared, ring, cs);
                for (const auto& binding : ring) {
                    resolve |= binding.offset == RingPending;
                }
            } else {
                untracked_reads_packet = draw_pipe->NextPacket(); // bindings not listed
            }
            ++num_stages;
            size += sizeof(PacketStage) +
                    AlignPacket(u32(stage->user_data.size() + stage->flattened_ud_buf.size()) *
                                sizeof(u32)) +
                    u32(ring.size() * sizeof(RingBinding));
        }
    }
    if (cs) {
        size += AlignPacket(sizeof(Gcn::ComputeProgram));
    }
    if (indirect) {
        size += AlignPacket(sizeof(IndirectDraw));
        // The arguments are read on the recording thread.
        const u64 args_size = u64(std::max(indirect->stride, 4u)) * std::max(indirect->max_count, 1u);
        if (args_size <= 16_MB) {
            NotePendingRead(indirect->args, args_size);
        } else {
            untracked_reads_packet = draw_pipe->NextPacket();
        }
        if (indirect->count) {
            NotePendingRead(indirect->count, sizeof(u32));
        }
    }
    // bbport: the V#s of the vertex streams, read here from the tables the user data points to:
    // those tables are what the game's constant engine dumps (a ring of ~25 pages rewritten
    // ~14000 times a second). Read on the recording thread instead, a later dump had often
    // replaced them by then: depth of field drew its passes with another draw's vertices (a
    // flipped copy of the scene, streaks and blocks in the sky; issue #44). The streams' data and
    // the index buffer are still read there: a DumpConstRam over them waits for this packet.
    thread_local boost::container::static_vector<Gcn::Buffer, Shader::NUM_BUFFERS> vsharps;
    vsharps.clear();
    if (pipeline && !pipeline->IsCompute()) {
        const auto* graphics = static_cast<const GraphicsPipeline*>(pipeline);
        if (const auto& fetch = graphics->GetFetchShader();
            fetch && fetch->attributes.size() <= vsharps.capacity()) {
            const auto& vs = graphics->GetStage(Shader::SwStage::Vertex);
            for (const auto& attrib : fetch->attributes) {
                const auto buffer = attrib.GetSharp(vs);
                vsharps.push_back(buffer);
                if (buffer.base_address != 0 && buffer.GetSize() > 0) {
                    if (BbCeStats::Enabled()) {
                        BbCeStats::Check(BbCeStats::Vertex, buffer.base_address, buffer.GetSize());
                    }
                    NotePendingRead(buffer.base_address, buffer.GetSize());
                }
            }
        }
        if (is_indexed) {
            const auto& regs = liverpool->regs;
            const u32 index_size =
                regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2 : 4;
            const VAddr index_address =
                regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
            const u64 index_bytes = u64(regs.num_indices) * index_size;
            if (BbCeStats::Enabled()) {
                BbCeStats::Check(BbCeStats::Index, index_address, index_bytes);
            }
            NotePendingRead(index_address, index_bytes);
        }
    }
    size += AlignPacket(u32(vsharps.size() * sizeof(Gcn::Buffer)));
    const u32 interval = VerifyInterval();
    const bool verify = interval && (draw_pipe->packets % interval) == 0;
    if (verify) {
        size += sizeof(Gcn::Regs);
    }

    u8* out = draw_pipe->Begin(size);
    u8* const start = out;
    auto& packet = *reinterpret_cast<DrawPacket*>(out);
    packet.kind = cs         ? (indirect ? PacketKind::IndirectDispatch : PacketKind::Dispatch)
                  : indirect ? PacketKind::Indirect
                  : pipeline ? PacketKind::Draw
                             : PacketKind::Special;
    packet.pipeline = pipeline;
    packet.prepared = used_prepared;
    packet.index_offset = index_offset;
    packet.is_indexed = is_indexed;
    packet.regs_reset = dirty.reset;
    packet.num_blocks = static_cast<u16>(num_blocks);
    packet.num_stages = num_stages;
    packet.verify = verify;
    packet.cb_extent = liverpool->last_cb_extent;
    packet.db_extent = liverpool->last_db_extent;
    packet.ring_end = constant_ring ? constant_ring->Position() : 0;
    packet.num_vsharps = static_cast<u32>(vsharps.size());
    packet.worker = ~0u;
    packet.worker_ring_end = 0;
    out += sizeof(DrawPacket);
    std::memcpy(out, blocks.data(), num_blocks * sizeof(u16));
    out += AlignPacket(num_blocks * sizeof(u16));
    for (u32 i = 0; i < num_blocks; ++i) {
        std::memcpy(out,
                    liverpool->regs.reg_array.data() + blocks[i] * Gcn::RegDirty::BlockWords,
                    Gcn::RegDirty::BlockWords * sizeof(u32));
        out += Gcn::RegDirty::BlockWords * sizeof(u32);
    }
    u32 stage_index = 0;
    for (const auto* stage : stages) {
        if (!stage) {
            continue;
        }
        const auto& ring = rings[stage_index++];
        auto& header = *reinterpret_cast<PacketStage*>(out);
        header = {stage,
                  stage->pgm_base,
                  static_cast<u32>(stage->user_data.size()),
                  static_cast<u32>(stage->flattened_ud_buf.size()),
                  static_cast<u32>(ring.size()),
                  0};
        out += sizeof(PacketStage);
        std::memcpy(out, ring.data(), ring.size() * sizeof(RingBinding));
        out += ring.size() * sizeof(RingBinding);
        std::memcpy(out, stage->user_data.data(), stage->user_data.size_bytes());
        std::memcpy(out + stage->user_data.size_bytes(), stage->flattened_ud_buf.data(),
                    stage->flattened_ud_buf.size() * sizeof(u32));
        out += AlignPacket(u32(stage->user_data.size() + stage->flattened_ud_buf.size()) *
                           sizeof(u32));
    }
    if (cs) {
        std::memcpy(out, cs, sizeof(Gcn::ComputeProgram));
        out += AlignPacket(sizeof(Gcn::ComputeProgram));
    }
    if (indirect) {
        std::memcpy(out, indirect, sizeof(IndirectDraw));
        out += AlignPacket(sizeof(IndirectDraw));
    }
    std::memcpy(out, vsharps.data(), vsharps.size() * sizeof(Gcn::Buffer));
    out += AlignPacket(u32(vsharps.size() * sizeof(Gcn::Buffer)));
    if (verify) {
        std::memcpy(out, &liverpool->regs, sizeof(Gcn::Regs));
        out += sizeof(Gcn::Regs);
    }
    dirty.Clear();
    draw_pipe->Commit(static_cast<u32>(out - start), resolve);
    PrintPipeStats();
}

void Rasterizer::RetireSubmission() {
    if (!draw_pipe || !OnStageA()) {
        return;
    }
    while (!pipe_keepalive.empty() && draw_pipe->Reached(pipe_keepalive.front().first)) {
        pipe_keepalive.pop_front();
    }
    if (!draw_pipe->Idle()) {
        pipe_keepalive.emplace_back(draw_pipe->Head(), draw_prep->KeepCurrent());
    }
}

bool Rasterizer::RunInOrder(OrderedTask task, const void* data, u32 size, u64 toggle,
                            bool reads_guest_memory) {
    if (!UseDrawPipe() || BbToggle::Disabled(toggle)) {
        DrainDrawPipe(DrawPipe::ReasonRasterizer);
        task(*this, static_cast<const u8*>(data));
        return false;
    }
    if (reads_guest_memory) {
        untracked_reads_packet = draw_pipe->NextPacket();
    }
    const u32 total = sizeof(TaskPacket) + AlignPacket(size);
    u8* out = draw_pipe->Begin(total);
    *reinterpret_cast<TaskPacket*>(out) = {PacketKind::Task, size, task};
    std::memcpy(out + sizeof(TaskPacket), data, size);
    draw_pipe->Commit(total);
    return true;
}

namespace {
// BB_SECTIONS: ordered tasks by function (cycles, count), stage B only.
constexpr std::size_t TaskKinds = 16;
std::array<void*, TaskKinds> task_kinds{};
std::array<u64, TaskKinds> task_cycles{}, task_counts{};
} // namespace

void Rasterizer::RunDrawPacket(void* context, const u8* data, u32 size) {
    BB_SECTION(Packet);
    auto& self = *static_cast<Rasterizer*>(context);
    // The tick the packet ended in (a constant RAM dump over what it reads in place must wait
    // for that tick: WaitForPendingReads).
    struct TickNote {
        Rasterizer& self;
        u64 packet;
        ~TickNote() {
            auto& entry = (*self.packet_ticks)[packet & 0xFFFF];
            entry.tick.store(self.scheduler.CurrentTick(), std::memory_order_relaxed);
            entry.packet.store(packet, std::memory_order_release);
        }
    } tick_note{self, self.draw_pipe->RunningPacket()};
    if (*reinterpret_cast<const PacketKind*>(data) == PacketKind::Task) {
        BB_SECTION(Task);
        self.buffer_cache.NewPacket();
        const auto& task = *reinterpret_cast<const TaskPacket*>(data);
        if (BbSections::Enabled()) {
            const u64 t0 = BbCpu::Ticks();
            task.task(self, data + sizeof(TaskPacket));
            void* fn = reinterpret_cast<void*>(task.task);
            for (std::size_t k = 0; k < TaskKinds; ++k) {
                if (task_kinds[k] == fn || !task_kinds[k]) {
                    task_kinds[k] = fn;
                    task_cycles[k] += BbCpu::Ticks() - t0;
                    ++task_counts[k];
                    break;
                }
            }
            return;
        }
        task.task(self, data + sizeof(TaskPacket));
        return;
    }
    const auto& packet = *reinterpret_cast<const DrawPacket*>(data);
    const u8* in = data + sizeof(DrawPacket);
    const auto* blocks = reinterpret_cast<const u16*>(in);
    in += AlignPacket(packet.num_blocks * sizeof(u16));
    auto& regs = self.pipe_regs;
    if (packet.regs_reset) {
        regs.SetDefaults();
    }
    for (u32 i = 0; i < packet.num_blocks; ++i) {
        std::memcpy(regs.reg_array.data() + blocks[i] * Gcn::RegDirty::BlockWords, in,
                    Gcn::RegDirty::BlockWords * sizeof(u32));
        in += Gcn::RegDirty::BlockWords * sizeof(u32);
    }
    ASSERT(packet.num_stages <= Shader::Info::MaxUdSnapshots);
    for (u32 i = 0; i < packet.num_stages; ++i) {
        const auto& stage = *reinterpret_cast<const PacketStage*>(in);
        in += sizeof(PacketStage);
        self.ring_stages[i] = {stage.info, reinterpret_cast<const RingBinding*>(in),
                               stage.num_ring};
        in += stage.num_ring * sizeof(RingBinding);
        const auto* words = reinterpret_cast<const u32*>(in);
        Shader::Info::ud_snapshots[i] = {stage.info, words, stage.ud_size, words + stage.ud_size,
                                         stage.flat_size, stage.pgm_base};
        in += AlignPacket((stage.ud_size + stage.flat_size) * sizeof(u32));
    }
    Shader::Info::num_ud_snapshots = packet.num_stages;
    self.num_ring_stages = packet.num_stages;
    const Gcn::ComputeProgram* cs = nullptr;
    if (packet.kind == PacketKind::Dispatch || packet.kind == PacketKind::IndirectDispatch) {
        cs = reinterpret_cast<const Gcn::ComputeProgram*>(in);
        in += AlignPacket(sizeof(Gcn::ComputeProgram));
    }
    const IndirectDraw* indirect = nullptr;
    if (packet.kind == PacketKind::Indirect || packet.kind == PacketKind::IndirectDispatch) {
        indirect = reinterpret_cast<const IndirectDraw*>(in);
        in += AlignPacket(sizeof(IndirectDraw));
    }
    self.packet_vsharps = {reinterpret_cast<const Gcn::Buffer*>(in), packet.num_vsharps};
    in += AlignPacket(u32(packet.num_vsharps * sizeof(Gcn::Buffer)));
    if (packet.verify) {
        // The resource tables the GPU thread walked must read the same now: else a write
        // still queued here (WriteData, DMA) changed them after that thread read them.
        static u32 flat_reports = 0;
        for (u32 i = 0; i < packet.num_stages && flat_reports < 32; ++i) {
            const auto& snapshot = Shader::Info::ud_snapshots[i];
            const auto& srt = snapshot.info->srt_info;
            // Stages the pipeline selection did not refresh (no user data) keep their tables.
            if (!srt.walker_func || snapshot.flat_size == 0 || snapshot.user_data_size == 0) {
                continue;
            }
            // (Room for all the walker writes, should the snapshot hold fewer.)
            std::vector<u32> flat(std::max<u32>(snapshot.flat_size, srt.flattened_bufsize_dw));
            std::memcpy(flat.data(), snapshot.user_data,
                        std::min(snapshot.user_data_size, snapshot.flat_size) * sizeof(u32));
            // Pointers in the tables may be stale by now: a fault only skips the check.
            sigjmp_buf recover;
            if (sigsetjmp(recover, 0)) {
                runtime_fault_recover = nullptr;
                continue;
            }
            runtime_fault_recover = &recover;
            Shader::RunSrtWalker(srt, snapshot.user_data, flat.data());
            runtime_fault_recover = nullptr;
            if (std::memcmp(flat.data(), snapshot.flat, snapshot.flat_size * sizeof(u32)) != 0) {
                u32 first = 0, count = 0;
                for (u32 w = 0; w < snapshot.flat_size; ++w) {
                    if (flat[w] != snapshot.flat[w]) {
                        first = count++ ? first : w;
                    }
                }
                std::printf("Draw pipe: resource tables of shader %016llx changed after the GPU "
                            "thread read them (%u of %u words, first %u: %#x -> %#x, user data "
                            "%u words)\n",
                            static_cast<unsigned long long>(snapshot.info->pgm_hash), count,
                            snapshot.flat_size, first, snapshot.flat[first], flat[first],
                            snapshot.user_data_size);
                ++flat_reports;
            }
        }
        const auto* live = reinterpret_cast<const Gcn::Regs*>(in);
        in += sizeof(Gcn::Regs);
        static u32 reports = 0;
        for (u32 block = 0; block < Gcn::RegDirty::NumBlocks && reports < 32; ++block) {
            const u32 at = block * Gcn::RegDirty::BlockWords;
            if (std::memcmp(regs.reg_array.data() + at, live->reg_array.data() + at,
                            Gcn::RegDirty::BlockWords * sizeof(u32)) != 0) {
                for (u32 w = at; w < at + Gcn::RegDirty::BlockWords; ++w) {
                    if (regs.reg_array[w] != live->reg_array[w]) {
                        std::printf("Draw pipe: register copy differs at word %#x (%#x vs %#x)\n",
                                    w, regs.reg_array[w], live->reg_array[w]);
                        ++reports;
                        break;
                    }
                }
            }
        }
    }
    ASSERT(static_cast<u32>(in - data) == size);
    stage_regs = &regs;
    self.pipe_cb_extent = packet.cb_extent;
    self.pipe_db_extent = packet.db_extent;
    if (packet.kind == PacketKind::Special) {
        // A draw FilterDraw handles (fast clear elimination, resolve, depth copy, skip).
        FrameCapture::Poll();
        self.scheduler.PopPendingOperations();
        self.FilterDraw();
    } else if (cs) {
        stage_cs = cs;
        const auto* pipeline = static_cast<const ComputePipeline*>(packet.pipeline);
        if (indirect) {
            self.DispatchIndirectRecord(pipeline, indirect->args, indirect->stride);
        } else {
            self.DispatchRecord(pipeline);
        }
        stage_cs = nullptr;
    } else if (indirect) {
        self.DrawIndirectRecord(static_cast<const GraphicsPipeline*>(packet.pipeline),
                                packet.is_indexed, *indirect);
    } else {
        self.DrawRecord(static_cast<const GraphicsPipeline*>(packet.pipeline), packet.prepared,
                        packet.is_indexed, packet.index_offset);
    }
    Shader::Info::num_ud_snapshots = 0;
    self.num_ring_stages = 0;
    self.packet_vsharps = {};
    if (self.constant_ring) {
        self.constant_ring->Stamp(packet.ring_end);
    }
    if (packet.worker != ~0u) {
        self.worker_rings[packet.worker]->Stamp(packet.worker_ring_end);
    }
}

void Rasterizer::ResolvePacket(void* context, u8* data, u32, u64 number, u32 worker) {
    auto& self = *static_cast<Rasterizer*>(context);
    auto& packet = *reinterpret_cast<DrawPacket*>(data);
    u8* in = data + sizeof(DrawPacket);
    in += AlignPacket(packet.num_blocks * sizeof(u16));
    in += packet.num_blocks * Gcn::RegDirty::BlockWords * sizeof(u32);
    auto& ring = *self.worker_rings[worker];
    const u64 alignment = std::max<u64>(self.instance.StorageMinAlignment(),
                                        self.instance.UniformMinAlignment());
    bool used_ring = false;
    for (u32 i = 0; i < packet.num_stages; ++i) {
        const auto& stage = *reinterpret_cast<const PacketStage*>(in);
        in += sizeof(PacketStage);
        auto* bindings = reinterpret_cast<RingBinding*>(in);
        in += stage.num_ring * sizeof(RingBinding);
        in += AlignPacket((stage.ud_size + stage.flat_size) * sizeof(u32));
        for (u32 j = 0; j < stage.num_ring; ++j) {
            auto& binding = bindings[j];
            if (binding.offset != RingPending) {
                continue;
            }
            const VAddr address = binding.address;
            const u64 size = binding.size;
            // The checks stage A made before: the GPU has written or will write these bytes,
            // or the layer keeps a VRAM copy. Writes noted for later packets only make it
            // stay with stage B (bound in place), never the reverse.
            const bool copyable = !self.buffer_cache.IsRegionGpuModified(address, size) &&
                                  !self.PendingWriteOverlaps(address, size) &&
                                  !VideoCore::BufferCache::LayerMirrored(address, size);
            std::optional<u64> offset;
            if (copyable) {
                offset = ring.Allocate(size, alignment);
            }
            if (offset) {
                self.memory->CopySparseMemory(address, ring.Data(*offset), size);
                ring.Flush(*offset, size);
                binding.offset = *offset;
                binding.ring = 1 + worker;
                used_ring = true;
            } else {
                binding.offset = RingNotCopied;
            }
            // Read by stage B or the GPU later: a dump over it waits for the packet.
            if (!offset || binding.always_note) {
                self.pending_reads.NoteTables(address, size, number);
            }
        }
    }
    if (used_ring) {
        packet.worker = worker;
        packet.worker_ring_end = ring.Position();
    }
}

void Rasterizer::PrintPipeStats() {
    static const bool stats = std::getenv("BB_FRAME_STATS") != nullptr;
    if (!stats || (draw_pipe->packets & 1023) != 0) {
        return;
    }
    static auto window = std::chrono::steady_clock::now();
    static u64 last_packets = 0, last_drains = 0, last_drain_cycles = 0, last_busy = 0;
    static u64 last_tsc = BbCpu::Ticks();
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - window).count();
    if (seconds < 5.0) {
        return;
    }
    const u64 tsc = BbCpu::Ticks();
    const double cycles = double(tsc - last_tsc);
    const u64 busy = draw_pipe->BusyCycles();
    std::printf("Draw pipe: %.0f draws/s pipelined, %.0f drains/s that waited, stage A waited "
                "%.1f%%, stage B busy %.1f%%\n",
                (draw_pipe->packets - last_packets) / seconds,
                (draw_pipe->drains - last_drains) / seconds,
                100.0 * (draw_pipe->drain_cycles - last_drain_cycles) / cycles,
                100.0 * (busy - last_busy) / cycles);
    std::printf("  drains by reason/s:");
    for (u32 r = 0; r < DrawPipe::NumReasons; ++r) {
        if (const u64 n = draw_pipe->drains_by_reason[r]) {
            static constexpr const char* names[] = {"ConstRam", "commands", "compute",
                                                    "submission end", "draw", "rasterizer"};
            const double share = 100.0 * draw_pipe->cycles_by_reason[r] / cycles;
            draw_pipe->cycles_by_reason[r] = 0;
            if (r < 256) {
                std::printf(" %s=%.0f (%.1f%%)",
                            BbCp::Diagnostics::OpName(r).c_str(), n / seconds,
                            share);
            } else {
                std::printf(" %s=%.0f (%.1f%%)", names[r - 256], n / seconds, share);
            }
        }
    }
    if (constram_waits_skipped) {
        std::printf(" ConstRam not waited=%.0f", constram_waits_skipped / seconds);
        constram_waits_skipped = 0;
    }
    if (constram_gpu_waits) {
        std::printf(" ConstRam GPU waits=%.1f (%.2f ms/s)", constram_gpu_waits / seconds,
                    constram_gpu_wait_cycles / (cycles / seconds) * 1000.0 / seconds);
        constram_gpu_waits = constram_gpu_wait_cycles = 0;
    }
    if (BbSections::Enabled()) {
        static std::array<u64, BbSections::Count> last_sections{};
        const double packets = double(draw_pipe->packets - last_packets);
        const double tsc_hz = cycles / seconds;
        std::printf("\n  stage B us per packet:");
        for (u32 s = 0; s < BbSections::Count; ++s) {
            const u64 now_cycles = BbSections::cycles[s].load(std::memory_order_relaxed);
            if (packets > 0) {
                std::printf(" %s %.2f", BbSections::Names[s],
                            1e6 * double(now_cycles - last_sections[s]) / tsc_hz / packets);
            }
            last_sections[s] = now_cycles;
        }
        static u64 last_ring_buffers = 0, last_ring_copies = 0, last_ring_bytes = 0;
        if (packets > 0) {
            const u64 b = BbSections::ring_buffers.load(), c = BbSections::ring_copies.load(),
                      n = BbSections::ring_bytes.load();
            std::printf("\n  stage A ring per packet: %.2f buffers, %.2f copies, %.0f bytes",
                        double(b - last_ring_buffers) / packets,
                        double(c - last_ring_copies) / packets,
                        double(n - last_ring_bytes) / packets);
            last_ring_buffers = b;
            last_ring_copies = c;
            last_ring_bytes = n;
        }
        // Racy reads of stage B's counters: diagnostics only.
        std::printf("\n  stage B tasks (us per frame-second, count/s):");
        static std::array<u64, TaskKinds> last_task_cycles{}, last_task_counts{};
        for (std::size_t k = 0; k < TaskKinds && task_kinds[k]; ++k) {
            Dl_info info{};
            dladdr(task_kinds[k], &info);
            char name[32];
            std::snprintf(name, sizeof(name), "+0x%llx",
                          (unsigned long long)(reinterpret_cast<u64>(task_kinds[k]) -
                                               reinterpret_cast<u64>(info.dli_fbase)));
            const u64 c = task_cycles[k], n = task_counts[k];
            std::printf(" [%s %.0f us/s %.0f/s]", name,
                        1e6 * double(c - last_task_cycles[k]) / tsc_hz / seconds,
                        double(n - last_task_counts[k]) / seconds);
            last_task_cycles[k] = c;
            last_task_counts[k] = n;
        }
    }
    std::printf("\n  stage A waits over 0.5%%:");
    for (const auto& [function, lines] : drain_sites) {
        for (const auto& [line, waited] : lines) {
            if (waited * 200 > cycles) {
                std::printf(" %s:%u %.1f%%", function, line, 100.0 * waited / cycles);
            }
        }
    }
    std::printf("\n");
    drain_sites.clear();
    draw_pipe->drains_by_reason = {};
    window = now;
    last_tsc = tsc;
    last_packets = draw_pipe->packets;
    last_drains = draw_pipe->drains;
    last_drain_cycles = draw_pipe->drain_cycles;
    last_busy = busy;
}

} // namespace Vulkan

namespace Vulkan::InPlaceReads {
namespace {
struct Entry {
    VAddr address;
    u64 size, tick, hash;
    u32 kind;
};
constexpr u32 Count = 1u << 15;
std::array<Entry, Count> entries{};
std::atomic<u64> head{0};
std::atomic<u32> reported{0};
} // namespace

bool Enabled() {
    static const bool on = [] {
        const char* env = std::getenv("BB_INPLACE_READ_CHECK");
        return env && env[0] == '1';
    }();
    return on;
}

void Note(VAddr address, u64 size, u64 tick, u64 hash, u32 kind) {
    entries[head.fetch_add(1, std::memory_order_relaxed) & (Count - 1)] = {address, size, tick,
                                                                           hash, kind};
}

void CheckDump(VAddr address, u64 size, const std::function<bool(u64)>& tick_done) {
    static constexpr const char* kinds[] = {"buffer", "paged", "vertex", "index"};
    for (const auto& e : entries) {
        if (e.size == 0 || e.address >= address + size || address >= e.address + e.size ||
            tick_done(e.tick)) {
            continue;
        }
        if (reported.fetch_add(1, std::memory_order_relaxed) < 400) {
            std::printf("In-place read check: frame %llu dump %#llx+%llu over a pending %s read "
                        "%#llx+%llu by %016llx (tick %llu)\n",
                        (unsigned long long)BbStats::frame_number.load(std::memory_order_relaxed),
                        (unsigned long long)address, (unsigned long long)size, kinds[e.kind & 3],
                        (unsigned long long)e.address, (unsigned long long)e.size,
                        (unsigned long long)e.hash, (unsigned long long)e.tick);
        }
        return;
    }
}
} // namespace Vulkan::InPlaceReads
