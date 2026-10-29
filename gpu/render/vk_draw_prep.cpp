// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: speculative draw preparation on worker threads (see vk_draw_prep.h).

#include <algorithm>
#include "shim/bb_env.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "bbport_threads.h"
#include "bbport_toggles.h"
#include "common/thread.h"
#include "cp/command_processor.h"
#include "render/vk_draw_prep.h"
#include "render/gcn_to_vk.h"
#include "render/pipeline_cache.h"
#include "vulkan/instance.h"
#include <xxhash.h>

namespace Vulkan {

namespace {
// The workers' wait for work between checks.
constexpr std::chrono::milliseconds WorkerPoll{1}; // Common::WaitFor: no spin on Windows

bool IsDirectDraw(BbCp::Op opcode) {
    using BbCp::Op;
    return opcode == Op::DrawIndex2 || opcode == Op::DrawIndexOffset2 || opcode == Op::DrawIndexAuto;
}

/// Walks the type-3 packets of a command buffer (type-2 padding skipped).
template <typename Func>
void ForEachPacket(std::span<const u32> commands, Func&& func) {
    for (size_t at = 0; at < commands.size();) {
        const u32* packet = commands.data() + at;
        const BbCp::Header header{*packet};
        if (header.Type() == 2) {
            ++at;
            continue;
        }
        if (header.Type() != 3) {
            return;
        }
        const size_t words = header.BodyWords() + 1;
        if (at + words > commands.size()) {
            return;
        }
        func(packet);
        at += words;
    }
}

/// Copies a command buffer with the buffers it calls (INDIRECT_BUFFER) inlined where they are
/// called, as the command processor walks them: the scanner and the workers must see every
/// register write and draw of the stream, or their checksums never meet the GPU thread's.
void AppendFlattened(std::vector<u32>& out, std::span<const u32> commands, u32 depth) {
    for (size_t at = 0; at < commands.size();) {
        const u32* packet = commands.data() + at;
        const BbCp::Header header{*packet};
        if (header.Type() == 2) {
            ++at;
            continue;
        }
        const size_t words = header.Type() == 3 ? header.BodyWords() + 1 : 0;
        if (!words || at + words > commands.size()) {
            return;
        }
        if (header.Opcode() == BbCp::Op::IndirectBuffer && depth < 4) {
            const auto ib = BbCp::IndirectBuffer::From(packet + 1);
            if (ib.Address() && ib.Dwords()) {
                AppendFlattened(out, {ib.Address(), ib.Dwords()}, depth + 1);
            }
        } else {
            out.insert(out.end(), packet, packet + words);
        }
        at += words;
    }
}

u32 DefaultWorkerCount(bool integrated) {
    if (const char* env = bb_env("BB_PREP_WORKERS")) {
        return static_cast<u32>(std::clamp(std::atoi(env), 0, 16));
    }
    // Workers are SCHED_IDLE and claim whole buffers, but more of them were slower: CPU-bound
    // (docs/pipeline_4stage.ru.md, three interleaved runs each) 16 threads with 4 workers +5%
    // over 8 (and +27 % over none, 2026-10-25). A quarter of the hardware threads from 12, one
    // from 8, none below 8 or with a GPU sharing the CPU's power (Steam Deck: one worker prepared
    // only ~13 % of the draws in time, the GPU thread did the rest again, and its CPU time came
    // out of the 15 W budget: 68.6 -> 70.7 FPS, GPU clock 1345 -> 1378 MHz without it; a 4-thread
    // laptop: 30.0 -> 31.9 FPS).
    const u32 available = BbThreads::Available();
    if (available >= 12) {
        return std::min<u32>(available / 4, 8);
    }
    return available >= 8 && !integrated ? 1 : 0;
}

/// Flattened user data of one submission: chunks never move, so the GPU thread can read a
/// prepared draw's data while the worker keeps appending.
struct FlatArena {
    static constexpr size_t ChunkWords = 16 * 1024;
    std::vector<std::unique_ptr<u32[]>> chunks;
    size_t used = ChunkWords;

    const u32* Append(const std::vector<u32>& data) {
        return AppendArray(data.data(), data.size());
    }

    /// Trivially copyable elements, 8-byte aligned (sharps hold u64 bit fields).
    template <typename T>
    const T* AppendArray(const T* data, size_t count) {
        static_assert(std::is_trivially_copyable_v<T> && alignof(T) <= 8);
        const size_t words = (count * sizeof(T) + sizeof(u32) - 1) / sizeof(u32);
        used = (used + 1) & ~size_t(1);
        if (chunks.empty() || words > ChunkWords - std::min(used, ChunkWords)) {
            chunks.push_back(std::make_unique<u32[]>(std::max(ChunkWords, words)));
            used = 0;
        }
        u32* dst = chunks.back().get() + used;
        if (count) {
            std::memcpy(dst, data, count * sizeof(T));
        }
        used += words;
        return reinterpret_cast<const T*>(dst);
    }
};

/// Reads the stage's resource sharps from the worker's flattened user data, so the GPU thread
/// only binds them (the sharp fetches and texture-description hashing were ~7% of it).
void PrepareResources(const Shader::Info& info, PreparedStage& stage, FlatArena& arena) {
    thread_local std::vector<Gcn::Image> images;
    thread_local std::vector<u64> hashes;
    thread_local std::vector<Gcn::Sampler> samplers;
    thread_local std::vector<Gcn::Buffer> buffers;
    images.clear();
    hashes.clear();
    samplers.clear();
    buffers.clear();
    for (const auto& res : info.images) {
        images.push_back(res.GetSharp(info));
        hashes.push_back(ImageDescHash(images.back(), res));
    }
    for (const auto& res : info.samplers) {
        samplers.push_back(res.GetSharp(info));
    }
    for (const auto& res : info.buffers) {
        buffers.push_back(res.IsSpecial() ? Gcn::Buffer::Null() : res.GetSharp(info));
    }
    stage.image_sharps = arena.AppendArray(images.data(), images.size());
    stage.image_hashes = arena.AppendArray(hashes.data(), hashes.size());
    stage.sampler_sharps = arena.AppendArray(samplers.data(), samplers.size());
    stage.buffer_sharps = arena.AppendArray(buffers.data(), buffers.size());
    stage.num_images = static_cast<u32>(images.size());
    stage.num_samplers = static_cast<u32>(samplers.size());
    stage.num_buffers = static_cast<u32>(buffers.size());
}

/// Rasterizer::BindVertexBuffers' pure part (GraphicsPipeline::GetVertexInputs for the dynamic
/// vertex input path, and the merge of the stream memory into ranges).
void PrepareVertexInputs(const Shader::Info& vs, const Shader::Gcn::FetchShaderData& fetch,
                         const Gcn::Regs& regs, PreparedVertexInputs& out, FlatArena& arena) {
    using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
    thread_local std::vector<vk::VertexInputAttributeDescription2EXT> attributes;
    thread_local std::vector<vk::VertexInputBindingDescription2EXT> bindings;
    thread_local std::vector<Gcn::Buffer> buffers;
    thread_local std::vector<PreparedVertexInputs::Range> ranges, merged;
    thread_local std::vector<u8> range_index;
    attributes.clear();
    bindings.clear();
    buffers.clear();
    ranges.clear();
    merged.clear();
    range_index.clear();
    for (const auto& attrib : fetch.attributes) {
        const auto step_rate = attrib.GetStepRate();
        const auto buffer = attrib.GetSharp(vs);
        // The V# may come from user data the guest is still writing: an unknown format leaves
        // the inputs to the GPU thread, which reads them once the draw is submitted (the
        // assertion in SurfaceFormat stopped the process here).
        const vk::Format format =
            GcnToVk::TrySurfaceFormat(buffer.GetDataFmt(), buffer.GetNumberFmt());
        if (format == vk::Format::eUndefined &&
            buffer.GetDataFmt() != Gcn::DataFormat::FormatInvalid) {
            out.valid = false;
            return;
        }
        attributes.push_back({
            .location = attrib.semantic,
            .binding = attrib.semantic,
            .format = format,
            .offset = 0,
        });
        bindings.push_back({
            .binding = attrib.semantic,
            .stride = buffer.GetStride(),
            .inputRate = step_rate == InstanceIdType::None ? vk::VertexInputRate::eVertex
                                                           : vk::VertexInputRate::eInstance,
            .divisor = step_rate == InstanceIdType::OverStepRate0 ? regs.vgt_instance_step_rate_0
                       : step_rate == InstanceIdType::OverStepRate1
                           ? regs.vgt_instance_step_rate_1
                           : 1u,
        });
        buffers.push_back(buffer);
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            ranges.push_back({buffer.base_address, buffer.base_address + buffer.GetSize()});
        }
    }
    std::ranges::sort(ranges, {}, &PreparedVertexInputs::Range::base);
    for (const auto& range : ranges) {
        if (merged.empty() || merged.back().end < range.base) {
            merged.push_back(range);
        } else {
            merged.back().end = std::max(merged.back().end, range.end);
        }
    }
    for (const auto& buffer : buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto it = std::ranges::find_if(merged, [&](const auto& range) {
                return buffer.base_address >= range.base && buffer.base_address < range.end;
            });
            range_index.push_back(static_cast<u8>(it - merged.begin()));
        }
    }
    out.attributes = arena.AppendArray(attributes.data(), attributes.size());
    out.bindings = arena.AppendArray(bindings.data(), bindings.size());
    out.buffers = arena.AppendArray(buffers.data(), buffers.size());
    out.range_index = arena.AppendArray(range_index.data(), range_index.size());
    out.ranges = arena.AppendArray(merged.data(), merged.size());
    out.count = static_cast<u32>(attributes.size());
    out.num_ranges = static_cast<u32>(merged.size());
    out.buffers_hash =
        buffers.empty() ? 0 : XXH3_64bits(buffers.data(), buffers.size() * sizeof(Gcn::Buffer));
    out.valid = true;
}

} // namespace

DrawPreparation::DrawPreparation(PipelineCache& pipeline_cache_)
    : pipeline_cache{pipeline_cache_} {
    worker_count = DefaultWorkerCount(pipeline_cache_.GetInstance().IsIntegrated());
    if (worker_count) {
        scanner = std::jthread([this](std::stop_token stop) { ScannerLoop(stop); });
    }
    for (u32 i = 0; i < worker_count; ++i) {
        workers.emplace_back([this, i](std::stop_token stop) { WorkerLoop(stop, i); });
    }
    std::printf("GPU: draw preparation workers: %u (+1 scanner, %u hardware threads)\n",
                worker_count, BbThreads::Available());
}

DrawPreparation::~DrawPreparation() {
    for (auto& worker : workers) {
        worker.request_stop();
    }
    if (scanner.joinable()) {
        scanner.request_stop();
    }
    cv.notify_all();
    workers.clear();
    scanner = {};
}

std::shared_ptr<DrawPreparation::Submission> DrawPreparation::Build(
    std::span<const u32> commands) {
    if (!Enabled()) {
        return {};
    }
    auto submission = std::make_shared<Submission>();
    submission->commands.reserve(commands.size());
    AppendFlattened(submission->commands, commands, 0);
    ForEachPacket(submission->commands, [&](const u32* packet) {
        submission->num_draws += IsDirectDraw(BbCp::Header{*packet}.Opcode());
    });
    submission->draws = std::make_unique<PreparedDraw[]>(submission->num_draws);
    return submission;
}

void DrawPreparation::Enqueue(u64 seq, std::shared_ptr<Submission> submission) {
    if (!submission) {
        return;
    }
    submission->seq = seq;
    {
        std::scoped_lock lk{intake_mutex};
        recent[seq % RecentSlots] = submission;
        intake.push_back(std::move(submission));
    }
    // A helper deciding to sleep right now misses this: they wake up every millisecond too.
    cv.notify_all();
}

void DrawPreparation::TakeIntakeLocked() {
    std::scoped_lock lk{intake_mutex};
    for (auto& submission : intake) {
        submissions.push_back(std::move(submission));
    }
    intake.clear();
}

void DrawPreparation::BeginSubmission(u64 seq, const Gcn::Regs& regs, u64 reg_checksum) {
    current_draw = 0;
    current.reset();
    if (!Enabled()) {
        return;
    }
    gpu_seq.store(seq, std::memory_order_release);
    {
        std::scoped_lock lk{intake_mutex};
        if (const auto& submission = recent[seq % RecentSlots];
            submission && submission->seq == seq) {
            current = submission;
        }
    }
    // The first submission: the helpers' starting state (once, waiting for `mutex`). Later: the
    // rebase below only when `mutex` is free at once (it is rare, and can wait a submission).
    std::unique_lock lk{mutex, std::defer_lock};
    if (!gpu_baseline_sent) {
        lk.lock();
        gpu_baseline_sent = true;
    } else if (!lk.try_lock()) {
        return;
    }
    TakeIntakeLocked();
    if (!baseline_ready) {
        initial_regs = std::make_unique<Gcn::Regs>(regs);
        tail_regs = std::make_unique<Gcn::Regs>(regs);
        initial_checksum = tail_checksum = reg_checksum;
        baseline_seq = tail_seq = seq;
        baseline_ready = true;
        cv.notify_all();
    }
    // Idle helpers can starve on a busy CPU. Rather than queueing without bound, restart
    // them from the GPU thread's own state (one register copy) once the scanner lags.
    constexpr u64 MaxScanLag = 64;
    if (!submissions.empty() && seq > tail_seq + MaxScanLag &&
        submissions.front()->seq < seq && !submissions.front()->scanned.load()) {
        *tail_regs = regs;
        tail_seq = seq;
        tail_checksum = reg_checksum;
        while (!submissions.empty() && submissions.front()->seq < seq) {
            submissions.pop_front();
        }
        ++rebases;
        cv.notify_all();
    }
}

void DrawPreparation::EndSubmission() {
    if (current) {
        current->gpu_done.store(true, std::memory_order_release);
        {
            std::scoped_lock lk{intake_mutex};
            if (recent[current->seq % RecentSlots] == current) {
                recent[current->seq % RecentSlots].reset();
            }
        }
        current.reset();
    }
    if (Enabled()) {
        Collect();
    }
}

const PreparedDraw* DrawPreparation::NextDraw() {
    if (!current || current_draw >= current->num_draws) {
        return nullptr;
    }
    const PreparedDraw* draw = &current->draws[current_draw++];
    if (BbToggle::Disabled(BbToggle::DrawPreparation)) {
        return nullptr;
    }
    return draw;
}

void DrawPreparation::Collect() {
    std::unique_lock lk{mutex, std::try_to_lock};
    if (!lk) {
        return; // a helper has it: collected after a later submission
    }
    while (!submissions.empty()) {
        const auto& front = submissions.front();
        if (!baseline_ready || !front->gpu_done.load(std::memory_order_acquire)) {
            break;
        }
        if (front->seq >= baseline_seq) {
            // Its delta moves the restart state to the start of the next buffer.
            if (!front->scanned.load(std::memory_order_acquire)) {
                break;
            }
            front->delta.Apply(*tail_regs);
            tail_seq = front->seq + 1;
            tail_checksum = front->end_checksum;
        }
        submissions.pop_front();
    }
}

void DrawPreparation::ScannerLoop(std::stop_token stop) {
    Common::SetCurrentThreadName("bb:DrawScan");
    BbThreads::MakeBackground();
    std::unique_ptr<Gcn::Regs> regs;
    u64 checksum = 0;
    u64 next_seq = 0;
    {
        std::unique_lock lk{mutex};
        cv.wait(lk, stop, [&] { return baseline_ready; });
        if (stop.stop_requested()) {
            return;
        }
        regs = std::make_unique<Gcn::Regs>(*initial_regs);
        checksum = initial_checksum;
        next_seq = baseline_seq;
    }
    Gcn::RegDirty dirty;
    while (!stop.stop_requested()) {
        std::shared_ptr<Submission> submission;
        {
            std::unique_lock lk{mutex};
            // Woken by Enqueue, and every millisecond (its notification may come between the
            // check and the wait: it does not take `mutex`).
            while (!Common::WaitFor(cv, lk, WorkerPoll, [&] {
                if (stop.stop_requested()) {
                    return true;
                }
                TakeIntakeLocked();
                return next_seq < tail_seq ||
                       (!submissions.empty() && submissions.back()->seq >= next_seq);
            })) {
                if (stop.stop_requested()) {
                    return;
                }
            }
            if (stop.stop_requested()) {
                return;
            }
            if (next_seq < tail_seq) {
                // Rebased by the GPU thread: continue from its state.
                *regs = *tail_regs;
                checksum = tail_checksum;
                next_seq = tail_seq;
                continue;
            }
            // Unscanned buffers are only collected by a rebase, so `next_seq` is still queued.
            submission = submissions[next_seq - submissions.front()->seq];
        }
        submission->start_checksum = checksum;
        dirty.Clear();
        ForEachPacket(submission->commands, [&](const u32* packet) {
            BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(*regs, packet, checksum, &dirty);
        });
        submission->delta.Capture(*regs, dirty);
        submission->end_checksum = checksum;
        {
            std::scoped_lock lk{mutex};
            submission->scanned.store(true, std::memory_order_release);
        }
        ++next_seq;
        cv.notify_all();
    }
}

void DrawPreparation::WorkerLoop(std::stop_token stop, u32 index) {
    Common::SetCurrentThreadName(("bb:DrawPrep" + std::to_string(index)).c_str());
    BbThreads::MakeBackground();
    auto regs = std::make_unique<Gcn::Regs>();
    constexpr u64 NoPosition = ~0ull;
    u64 position = NoPosition; ///< `regs` holds the state at the start of this buffer
    PipelineSelection sel{};
    sel.regs = regs.get();
    PrepWorker prep_worker{};
    sel.worker = &prep_worker;
    FlatArena arena;
    std::vector<std::shared_ptr<Submission>> path;

    // The nearest scanned buffer the GPU thread has not reached and no worker has claimed.
    const auto claimable = [&]() -> std::shared_ptr<Submission> {
        const u64 gpu = gpu_seq.load(std::memory_order_acquire);
        for (const auto& submission : submissions) {
            if (submission->seq >= gpu && submission->seq >= tail_seq &&
                submission->scanned.load(std::memory_order_acquire) &&
                !submission->claimed.load(std::memory_order_relaxed)) {
                return submission;
            }
        }
        return {};
    };

    while (!stop.stop_requested()) {
        std::shared_ptr<Submission> target;
        {
            std::unique_lock lk{mutex};
            // gpu_seq moves without the lock: keep the candidate the predicate found.
            while (!Common::WaitFor(cv, lk, WorkerPoll, [&] {
                if (stop.stop_requested()) {
                    return true;
                }
                TakeIntakeLocked();
                target = baseline_ready ? claimable() : nullptr;
                return target != nullptr;
            })) {
                if (stop.stop_requested()) {
                    return;
                }
            }
            if (stop.stop_requested() || !target) {
                return;
            }
            target->claimed.store(true, std::memory_order_relaxed);
            // Buffers between this worker's state and the target: their deltas. A position
            // already collected (or past the target) restarts from the queue's tail state.
            if (position == NoPosition || position < tail_seq || position > target->seq) {
                *regs = *tail_regs;
                position = tail_seq;
            }
            const u64 front = submissions.front()->seq;
            path.clear();
            for (u64 seq = position; seq < target->seq; ++seq) {
                path.push_back(submissions[seq - front]);
            }
        }
        for (const auto& step : path) {
            step->delta.Apply(*regs);
        }
        path.clear();
        u64 checksum = target->start_checksum;
        u32 ordinal = 0;
        arena = FlatArena{};
        ForEachPacket(target->commands, [&](const u32* packet) {
            BbCp::CommandProcessor::ApplyGraphicsRegisterPacket(*regs, packet, checksum);
            if (!IsDirectDraw(BbCp::Header{*packet}.Opcode())) {
                return;
            }
            // Past the GPU thread the draws are no longer worth preparing.
            if (ordinal < target->num_draws &&
                target->seq >= gpu_seq.load(std::memory_order_acquire)) {
                auto& draw = target->draws[ordinal];
                prep_worker.stages.clear();
                prep_worker.failed = false;
                sel.draw_indirect_params = {};
                if (pipeline_cache.PrepareGraphicsPipeline(sel)) {
                    draw.reg_checksum = checksum;
                    draw.key = sel.graphics_key;
                    draw.num_stages = static_cast<u32>(prep_worker.stages.size());
                    for (u32 i = 0; i < draw.num_stages; ++i) {
                        const auto& stage = prep_worker.stages[i];
                        draw.stages[i] = {
                            .program = stage.program,
                            .hash = stage.hash,
                            .hw_stage = stage.hw_stage,
                            .pgm_base = stage.pgm_base,
                            .flat = arena.Append(*stage.flat),
                            .flat_size = static_cast<u32>(stage.flat->size()),
                        };
                        PrepareResources(prep_worker.infos.at(stage.program), draw.stages[i],
                                         arena);
                    }
                    draw.vertex.valid = false;
                    const auto* vs = sel.infos[static_cast<u32>(Shader::SwStage::Vertex)];
                    if (vs && sel.fetch_shader) {
                        PrepareVertexInputs(*vs, *sel.fetch_shader, *regs, draw.vertex, arena);
                    }
                    draw.state.store(PreparedDraw::Ready, std::memory_order_release);
                } else {
                    draw.state.store(PreparedDraw::Unavailable, std::memory_order_release);
                }
            }
            ++ordinal;
        });
        // The arena's chunks live as long as the submission.
        target->flat_chunks = std::move(arena.chunks);
        position = target->seq + 1;
    }
}

void DrawPreparation::Count(bool was_used, Miss why) {
    ++(was_used ? used : unused);
    if (!was_used) {
        ++misses[why];
    }
    static const bool stats = bb_env("BB_FRAME_STATS") != nullptr;
    // The clock is read every 1024 draws: per draw it was 8% of the GPU thread.
    if (!stats || ((used + unused) & 1023) != 0) {
        return;
    }
    static auto window = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds(5)) {
        return;
    }
    window = now;
    std::printf("Draw preparation: %llu prepared pipelines used, %llu not (not ready %llu, "
                "unavailable %llu, other registers %llu, other programs %llu; %u workers, "
                "%llu scanner rebases)\n",
                static_cast<unsigned long long>(used), static_cast<unsigned long long>(unused),
                static_cast<unsigned long long>(misses[NotReady]),
                static_cast<unsigned long long>(misses[Unavailable]),
                static_cast<unsigned long long>(misses[OtherRegisters]),
                static_cast<unsigned long long>(misses[OtherPrograms]), worker_count,
                static_cast<unsigned long long>(rebases));
    used = unused = 0;
    misses = {};
}

} // namespace Vulkan
