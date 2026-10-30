// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, compute dispatches (direct and indirect) and their recording.
// Split out of rasterizer.cpp (2026-10-26).


// The copy shader's history hashes ~5000 copies of ~250 bytes a frame: inlined, not a DLL call each.
#define XXH_INLINE_ALL
#include <xxhash.h>
#include <mutex>
#include "shim/bb_env.h"
#include <set>
#include <thread>
#include <unordered_map>
#include "common/thread.h"
#include <string>
#include <filesystem>
#include "render/raster_internal.h"
#include <vk_mem_alloc.h>


namespace Vulkan {

void Rasterizer::DispatchDirect() {
    BbStats::dispatches.fetch_add(1, std::memory_order_relaxed);
    // bbport: like draws, handed to the draw recording thread after the pipeline selection.
    const bool pipelined = UseDrawPipe() && !BbToggle::Disabled(BbToggle::PipelinedDispatch);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    // BB_DISPATCH_START_LOG=1 (diagnostics, the flicker): dispatches with a nonzero thread
    // group start (COMPUTE_START_X/Y/Z), which the translation does not apply.
    if (static const bool start_log = bb_env("BB_DISPATCH_START_LOG") != nullptr; start_log) {
        const auto& cs = CsRegs();
        static std::atomic<u32> printed{0};
        if ((cs.start_x | cs.start_y | cs.start_z) != 0 && printed.fetch_add(1) < 5000) {
            std::printf("Dispatch start: gpu frame %llu cs %016llx start %u,%u,%u dims %u,%u,%u\n",
                        (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                        (unsigned long long)pipeline->GetStage(Shader::SwStage::Compute).pgm_hash,
                        cs.start_x, cs.start_y, cs.start_z, cs.dim_x, cs.dim_y, cs.dim_z);
        }
    }
    if (pipelined) {
        PostDraw(pipeline, nullptr, false, 0, &CsRegs());
        return;
    }
    DispatchRecord(pipeline);
}

namespace {
/// BB_COPY_SKIP_SAME (on by default since 2026-10-28, =0 off; experiment bit 16384 inverts it
/// live): the game's copy
/// shader runs only the copies that change something. A copy is left out when the last copy into
/// exactly its destination (nothing else copied over it since) wrote the same bytes and the
/// destination already holds them (so whether that copy has run yet or not, they end up there).
/// The ~5000 copies a frame are checked on a thread of their own (bb:CopyFilter), fed by stage A
/// in packet order: on the draw recording thread they cost ~0.3 ms a frame (a weak CPU 3% slower).
/// While the draw recording thread is over 92% busy (under 85%: again) the filter is off: there
/// the CPU is the limit and its thread takes a core (8 threads: 1.5% slower; 16: 1.6% faster).
std::atomic<bool> copy_filter_room{true};

bool CopySkipSameOn() {
    static const bool on = !bb_env_off("BB_COPY_SKIP_SAME");
    return (on != BbToggle::Experiment(16384)) && copy_filter_room.load(std::memory_order_relaxed);
}

/// Stage B: copy_filter_room from its busy share, every ~0.3 s.
void UpdateCopyFilterRoom(const DrawPipe& pipe) {
    // bbport: BbCpu::Ticks, as busy_cycles (__rdtsc on x86-64; arm64: the native GPU process).
#if defined(__aarch64__)
    static const u64 interval = BbCpu::TicksFromUs(300'000);
#else
    constexpr u64 interval = 1'000'000'000ull;
#endif
    static u64 last_tsc = BbCpu::Ticks(), last_busy = 0;
    const u64 tsc = BbCpu::Ticks();
    if (tsc - last_tsc < interval) {
        return;
    }
    const u64 busy = pipe.BusyCycles();
    const double share = double(busy - last_busy) / double(tsc - last_tsc);
    last_tsc = tsc;
    last_busy = busy;
    const bool was = copy_filter_room.load(std::memory_order_relaxed);
    const bool room = share < (was ? 0.92 : 0.85);
    if (room != was) {
        copy_filter_room.store(room, std::memory_order_relaxed);
        std::printf("Copy filter: %s (draw recording thread %.0f%% busy)\n", room ? "on" : "off",
                    share * 100.0);
    }
}

using Ranges = std::vector<std::pair<unsigned long long, unsigned long long>>;

class CopyFilterThread {
public:
    struct Job {
        u64 packet = 0;
        u64 epoch = 0;
        std::vector<CopyEntry> entries;
        std::vector<u8> control; ///< the kept copies' control list entries; empty: all kept
        u32 kept = 0;
        // What DispatchRecord binds by, of all its copies (as CopyShaderDestinations and
        // CopyShaderRanges give them): worked out here, not on the draw recording thread.
        u64 base[2] = {};
        Ranges writes, ranges[2]; ///< destinations; sources, destinations by 64 KiB
    };

    static CopyFilterThread& Get() {
        static CopyFilterThread instance;
        return instance;
    }

    /// Stage A: the copies of the copy shader call in packet `packet` (taken from `entries`).
    /// A full queue posts nothing (stage B then finds no job and runs the call whole).
    void Post(u64 packet, std::vector<CopyEntry>& entries, u64 src_base, u64 dst_base) {
        const u64 at = head.load(std::memory_order_relaxed);
        // A slot is free once stage B took it and this thread checked it (either order).
        const u64 free_from = std::min(tail.load(std::memory_order_acquire),
                                       processed.load(std::memory_order_acquire));
        if (at - free_from >= jobs.size()) {
            return;
        }
        auto& job = jobs[at % jobs.size()];
        job.packet = packet;
        job.base[0] = src_base;
        job.base[1] = dst_base;
        job.entries.swap(entries);
        head.store(at + 1, std::memory_order_release);
        head.notify_one();
    }

    /// Stage B: the checked job for `packet`, or null (none: the history is dropped). Jobs of
    /// packets before it never reached here (another path ran them): dropped the same way.
    /// Not checked yet: null with `unchecked` set, taken (the call runs whole, which writes what
    /// the check will assume anyway: the history holds). Never waits for this thread.
    Job* Take(u64 packet, bool& unchecked) {
        unchecked = false;
        for (;;) {
            const u64 at = tail.load(std::memory_order_relaxed);
            if (at == head.load(std::memory_order_acquire)) {
                Invalidate();
                return nullptr;
            }
            Job& job = jobs[at % jobs.size()];
            if (job.packet == packet) {
                if (processed.load(std::memory_order_acquire) <= at) {
                    unchecked = true;
                    tail.store(at + 1, std::memory_order_release);
                    ++unchecked_calls;
                    return nullptr;
                }
                return &job;
            }
            if (job.packet > packet) {
                Invalidate();
                return nullptr;
            }
            Invalidate();
            tail.store(at + 1, std::memory_order_release);
        }
    }

    /// Stage B, after Take returned a job: done with it.
    void Release() {
        tail.store(tail.load(std::memory_order_relaxed) + 1, std::memory_order_release);
    }

    /// Stage B: a call did not run as its job said (not recorded, run whole after a check
    /// against an older history): the history starts again.
    void Invalidate() {
        epoch.fetch_add(1, std::memory_order_acq_rel);
    }
    [[nodiscard]] u64 Epoch() const {
        return epoch.load(std::memory_order_acquire);
    }

private:
    struct Last {
        u64 size, hash;
    };

    /// The copies' destination (or source) ranges merged across gaps up to `gap`, as (address,
    /// size): what BuildRanges (shader_hle.cpp) makes of the same control list.
    static void MergedRanges(const std::vector<CopyEntry>& entries, bool destinations, u64 gap,
                             Ranges& out) {
        out.clear();
        for (const auto& e : entries) {
            const u64 begin = destinations ? e.dst : e.src;
            out.emplace_back(begin, begin + e.size);
        }
        std::sort(out.begin(), out.end());
        size_t merged = 0;
        for (size_t i = 1; i < out.size(); ++i) {
            if (out[i].first <= out[merged].second + gap) {
                out[merged].second = std::max(out[merged].second, out[i].second);
            } else {
                out[++merged] = out[i];
            }
        }
        out.resize(out.empty() ? 0 : merged + 1);
        for (auto& r : out) {
            r.second -= r.first;
        }
    }

    CopyFilterThread() : thread{[this] { Run(); }} {
        thread.detach();
    }

    void Run() {
        Common::SetCurrentThreadName("bb:CopyFilter");
        for (u64 at = 0;; ++at) {
            for (u64 h = head.load(std::memory_order_acquire); h <= at;
                 h = head.load(std::memory_order_acquire)) {
                head.wait(h, std::memory_order_acquire);
            }
            Check(jobs[at % jobs.size()]);
            processed.store(at + 1, std::memory_order_release);
        }
    }

    void Check(Job& job) {
        const u64 now_epoch = Epoch();
        if (now_epoch != history_epoch) {
            last.clear();
            ranges.clear();
            history_epoch = now_epoch;
        }
        job.epoch = now_epoch;
        const auto& entries = job.entries;
        hashes.clear();
        keep.clear();
        for (const auto& e : entries) {
            const u64 hash = XXH3_64bits(reinterpret_cast<const void*>(e.src), e.size);
            hashes.push_back(hash);
            const auto it = last.find(e.dst);
            const bool unchanged = it != last.end() && it->second.size == e.size &&
                                   it->second.hash == hash &&
                                   std::memcmp(reinterpret_cast<const void*>(e.dst),
                                               reinterpret_cast<const void*>(e.src), e.size) == 0;
            if (!unchanged) {
                keep.push_back(u32(hashes.size() - 1));
            }
        }
        total += entries.size();
        left_out += entries.size() - keep.size();
        ++calls;
        // The kept ones are noted. Noted copies never overlap each other: one into exactly a noted
        // destination updates it in place (the ordered map only for new destinations).
        for (const u32 i : keep) {
            const auto& e = entries[i];
            if (const auto it = last.find(e.dst); it != last.end() && it->second.size == e.size) {
                it->second = {e.size, hashes[i]};
                continue;
            }
            auto lo = ranges.lower_bound(e.dst);
            if (lo != ranges.begin()) {
                auto prev = std::prev(lo);
                if (prev->first + prev->second > e.dst) {
                    lo = prev;
                }
            }
            while (lo != ranges.end() && lo->first < e.dst + e.size) {
                last.erase(lo->first);
                lo = ranges.erase(lo);
            }
            ranges.emplace(e.dst, e.size);
            last[e.dst] = {e.size, hashes[i]};
        }
        job.kept = u32(keep.size());
        job.control.clear();
        if (keep.size() < entries.size()) {
            job.control.resize(keep.size() * 12);
            for (size_t j = 0; j < keep.size(); ++j) {
                std::memcpy(job.control.data() + j * 12,
                            reinterpret_cast<const void*>(entries[keep[j]].control), 12);
            }
        }
        if (last.size() > 200000) {
            Invalidate();
        }
        static const u64 dest_gap = CopyDestinationGap();
        MergedRanges(entries, true, dest_gap, job.writes);
        MergedRanges(entries, false, 64_KB, job.ranges[0]);
        MergedRanges(entries, true, 64_KB, job.ranges[1]);
        const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
        if (now - printed >= 5) {
            static const bool where = bb_env_on("BB_COPY_WHERE");
            if (printed != 0 && where) {
                std::printf("Copy filter (5 s): %llu of %llu copies left out in %llu calls; "
                            "history restarted %llu times; %llu calls ran unchecked\n",
                            (unsigned long long)left_out, (unsigned long long)total,
                            (unsigned long long)calls,
                            (unsigned long long)(now_epoch - printed_epoch),
                            (unsigned long long)unchecked_calls.exchange(0));
            }
            printed = now;
            printed_epoch = now_epoch;
            total = left_out = calls = 0;
        }
    }

    std::array<Job, 512> jobs;
    std::atomic<u64> head{0};      ///< stage A posts
    std::atomic<u64> processed{0}; ///< this thread checked
    std::atomic<u64> tail{0};      ///< stage B took
    std::atomic<u64> epoch{0};
    std::atomic<u64> unchecked_calls{0};
    // This thread's.
    std::unordered_map<u64, Last> last; ///< destination -> its last copy
    std::map<u64, u64> ranges;          ///< destination -> size (the overlap check)
    std::vector<u64> hashes;
    std::vector<u32> keep;
    u64 history_epoch = 0, total = 0, left_out = 0, calls = 0, printed_epoch = 0;
    u32 printed = 0;
    std::thread thread;
};

} // namespace

void Rasterizer::PostCopyFilterJob(const Shader::Info& cs, u32 groups) {
    if (!CopySkipSameOn() || !draw_pipe) {
        return;
    }
    thread_local std::vector<CopyEntry> entries;
    entries.clear();
    if (CopyShaderEntries(cs, groups, entries)) {
        CopyFilterThread::Get().Post(draw_pipe->NextPacket(), entries,
                                     cs.buffers[1].GetSharp(cs).base_address,
                                     cs.buffers[2].GetSharp(cs).base_address);
    }
}

void Rasterizer::DispatchRecord(const ComputePipeline* pipeline) {
    BB_SECTION(DispatchRecord);
    ++work_after_early;
    buffer_cache.NewPacket();
    FrameCapture::Poll();
    gbuffer_draw = false;

    scheduler.PopPendingOperations();

    const auto& cs_program = CsRegs();

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    // After the resource binding: its transfers (image uploads, downloads) mark themselves.
    const auto mark = [&] {
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(cs.pgm_hash ^ 0xD15Aull, [&] {
                return fmt::format("dispatch cs {:016x} ({}x{}x{} groups)", cs.pgm_hash,
                                   cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
            });
        }
    };
    if (upscaler->Enabled()) {
        upscaler->OnDispatch(cs.pgm_hash);
    }
    // Experiment bit 4096 (measurement only, wrong images): the game's copy shader not run at all,
    // to see what it costs the GPU with everything around it (breaks, waits on its writes).
    if (BbToggle::Experiment(4096) && Game::BufferCopyShader() != 0 &&
        cs.pgm_hash == Game::BufferCopyShader()) {
        return;
    }
    // BB_COPY_SKIP_SAME: the copies that change something (checked by bb:CopyFilter), as the
    // shader's control list (one entry per work group) in the stream buffer, bound instead of the
    // game's (BindBuffers); none: the call is left out.
    const bool copy_shader = Game::BufferCopyShader() != 0 && cs.pgm_hash == Game::BufferCopyShader();
    static const bool copy_filter_env = !bb_env_off("BB_COPY_SKIP_SAME") || bb_env("BB_TOGGLE_FILE");
    u32 kept_groups = 0; // with a filtered control list: its work groups
    bool copy_job = false; // the call runs as its job said (else the history starts again)
    thread_local Ranges copy_writes, copy_ranges[2];
    bool ranges_given = false; // copy_writes, copy_ranges and copy_bases from its checked job
    u64 copy_bases[2] = {};
    if (copy_shader && copy_filter_env && draw_pipe) {
        static const bool gate = bb_env_on("BB_COPY_FILTER_GATE");
        if (gate) {
            UpdateCopyFilterRoom(*draw_pipe);
        }
        auto& filter = CopyFilterThread::Get();
        // Switched off (busy, or live): the calls from now on run whole, past the history.
        static bool was_on = false;
        const bool filter_on = CopySkipSameOn();
        if (was_on && !filter_on) {
            filter.Invalidate();
        }
        was_on = filter_on;
        bool unchecked = false;
        auto* job = filter_on ? filter.Take(draw_pipe->RunningPacket(), unchecked) : nullptr;
        if (job) {
            copy_writes.swap(job->writes);
            copy_ranges[0].swap(job->ranges[0]);
            copy_ranges[1].swap(job->ranges[1]);
            copy_bases[0] = job->base[0];
            copy_bases[1] = job->base[1];
            ranges_given = true;
        }
        if (unchecked) {
            copy_job = true; // runs whole (its slot taken already)
        } else if (job && job->epoch == filter.Epoch()) {
            copy_job = true;
            if (job->kept == 0) {
                filter.Release();
                return;
            }
            if (!job->control.empty()) {
                auto& stream = buffer_cache.GetStreamBuffer();
                const u64 offset = stream.Copy(job->control.data(), job->control.size(),
                                               instance.StorageMinAlignment());
                copy_control = {&stream, offset, job->control.size()};
                kept_groups = job->kept;
            }
        } else if (CopySkipSameOn()) {
            filter.Invalidate(); // run whole after a check against an older history
        }
        if (job) {
            filter.Release();
        }
    }
    bool hle;
    {
        BB_SECTION(ShaderHle);
        hle = ExecuteShaderHLE(cs, Regs(), cs_program, *this);
    }
    if (hle) {
        copy_control = {}; // the HLE ran the whole list (all its copies: the history holds)
        return;
    }
    // bbport: the game's copy shader running as itself (BB_COPY_SHADER_NATIVE): its destinations stay in
    // the game's memory. They hold data our translator reads on the CPU (next to shader code the
    // game uploads this way); a VRAM copy of them showed it a stale copy (a black scene). The
    // copy list read on the CPU (vk_shader_hle.cpp) kept them in place too.
    buffer_cache.force_writes_in_place =
        Game::BufferCopyShader() != 0 && cs.pgm_hash == Game::BufferCopyShader();
    // What it writes, from its control list: GPU-written marks on those bytes only (the bound
    // destination spans tens of MiB, marked whole it kept the constants there in place).
    // (From its checked job when it has one, the same ranges.)
    if (!ranges_given) {
        copy_writes.clear();
    }
    bool writes_known = false;
    if (buffer_cache.force_writes_in_place && CopyDestinationsPrecise() &&
        (ranges_given ||
         CopyShaderDestinations(cs, cs_program.dim_x, CopyDestinationGap(), copy_writes))) {
        buffer_cache.precise_writes = &copy_writes;
        writes_known = true;
    }
    // Its paged bindings: page table entries (and imports of the game's memory) for the ranges
    // its list names only. Bound whole, every block of the GiB it binds was published: 0.7-1 GiB
    // of never used granules imported, all zeros (2026-10-23). BB_COPY_PAGED_RANGES=0: whole.
    static const bool paged_ranges = !bb_env_off("BB_COPY_PAGED_RANGES");
    if (buffer_cache.force_writes_in_place && paged_ranges) {
        for (int i = 0; i < 2; ++i) {
            if (ranges_given) {
                buffer_cache.paged_ranges[i] = {copy_bases[i], &copy_ranges[i]};
                continue;
            }
            copy_ranges[i].clear();
            u64 base = 0;
            if (CopyShaderRanges(cs, cs_program.dim_x, i + 1, 64_KB, base, copy_ranges[i])) {
                buffer_cache.paged_ranges[i] = {base, &copy_ranges[i]};
            }
        }
    }
    static const bool copy_where = bb_env_on("BB_COPY_WHERE");
    if (ranges_given && copy_where) {
        // The job's ranges against the control list's (diagnostics).
        thread_local Ranges check[3];
        u64 base[2] = {};
        for (auto& c : check) {
            c.clear();
        }
        const bool ok =
            CopyShaderDestinations(cs, cs_program.dim_x, CopyDestinationGap(), check[0]) &&
            CopyShaderRanges(cs, cs_program.dim_x, 1, 64_KB, base[0], check[1]) &&
            CopyShaderRanges(cs, cs_program.dim_x, 2, 64_KB, base[1], check[2]);
        static std::atomic<u32> mismatches{0};
        if (!ok || check[0] != copy_writes || check[1] != copy_ranges[0] ||
            check[2] != copy_ranges[1] || base[0] != copy_bases[0] || base[1] != copy_bases[1]) {
            if (mismatches.fetch_add(1) < 20) {
                std::printf("Copy filter: ranges differ (%zu/%zu writes, %zu/%zu src, %zu/%zu dst)\n",
                            check[0].size(), copy_writes.size(), check[1].size(),
                            copy_ranges[0].size(), check[2].size(), copy_ranges[1].size());
            }
        }
    }
    if (writes_known && copy_where) {
        thread_local std::vector<std::pair<unsigned long long, unsigned long long>> src;
        src.clear();
        u64 base = 0;
        if (CopyShaderRanges(cs, cs_program.dim_x, 1, CopyDestinationGap(), base, src)) {
            buffer_cache.NoteCopyWhere(src, copy_writes, cs_program.dim_x);
        }
    }
    const bool bound = BindResources(pipeline);
    copy_control = {};
    buffer_cache.force_writes_in_place = false;
    buffer_cache.precise_writes = nullptr;
    buffer_cache.paged_ranges = {};
    if (!bound) {
        if (copy_job) {
            CopyFilterThread::Get().Invalidate(); // its copies did not run
        }
        return;
    }
    if (copy_where && writes_known) {
        // Draws and other dispatches since the previous copy shader call: 0 = back to back.
        buffer_cache.NoteCopyGap(work_total - last_copy_work);
        last_copy_work = work_total;
    } else {
        ++work_total;
    }

    // Experiment bit 128 (measurement only, unsafe): the game's copy shader without the barriers
    // around it, to see what they cost.
    const bool skip_barriers = BbToggle::Experiment(128) && Game::BufferCopyShader() != 0 &&
                               cs.pgm_hash == Game::BufferCopyShader();
    if (needs_barrier && !skip_barriers) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    mark();
    // BB_DISPATCH_LOG=1 (diagnostics): the shape of each kind of real compute dispatch, once: the
    // buffers (size, formatted, written), the groups, and what the image clear detector thinks.
    {
        static const bool log = bb_env_on("BB_DISPATCH_LOG");
        if (log) {
            static std::mutex mutex;
            static std::set<std::string> told;
            std::string key = fmt::format("{:016x} x{} y{} z{}", cs.pgm_hash, cs_program.dim_x,
                                          cs_program.dim_y, cs_program.dim_z);
            for (const auto& b : cs.buffers) {
                key += fmt::format(" [{}{}{} {}]", b.is_formatted ? "F" : "R", b.is_written ? "W" : "r",
                                   b.is_formatted ? "" : "", b.GetSharp(cs).GetSize());
            }
            if (cs.buffers.size() >= 2 && cs.buffers[1].is_formatted && cs.buffers[1].is_written) {
                const auto target = cs.buffers[1].GetSharp(cs);
                const auto id = texture_cache.FindImageFromRange(target.base_address, target.GetSize(), false);
                if (id) {
                    const auto& image = texture_cache.GetImage(id);
                    key += fmt::format(" image(size {} bits {} depth {} fmt {})", image.info.guest_size,
                                       image.info.num_bits, image.info.props.is_depth ? 1 : 0,
                                       Gcn::NumBitsPerBlock(target.GetDataFmt()));
                } else {
                    key += fmt::format(" NO-IMAGE at {:#x}", target.base_address);
                }
            }
            std::scoped_lock lock{mutex};
            if (told.size() < 400 && told.insert(key).second) {
                std::printf("Dispatch shape: %s\n", key.c_str());
            }
        }
    }
    pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                            {buffer_infos.data(), buffer_infos.size()});

    const vk::Pipeline handle = pipeline->Handle();
    const u32 dim_x = kept_groups ? kept_groups : cs_program.dim_x;
    const u32 dim_y = cs_program.dim_y, dim_z = cs_program.dim_z;
    if (FrameCapture::Active()) {
        FrameCapture::Dispatch(cs.pgm_hash, dim_x, dim_y, dim_z);
    }
    // bbport (macOS): built only with breadcrumbs on (RecordCrumbWith).
    const auto make_crumb = [&] {
        return Breadcrumbs::Crumb{
            .kind = Breadcrumbs::Kind::Dispatch,
            .hash = {cs.pgm_hash, 0},
            .program = {cs.ProgramBase(), 0},
            .count = {dim_x, dim_y, dim_z},
        };
    };
    scheduler.RecordCrumbWith(make_crumb, [=](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatch(dim_x, dim_y, dim_z);
    });

    if (skip_barriers) {
        bound_buffers.clear(); // no write marks: the draws after it wait for nothing
    }
    paged_writes_known = writes_known;
    ResetBindings(true);
    paged_writes_known = false;
    // What it wrote in the game's memory (constants of the draws after it): copied into VRAM
    // here, outside any render pass, and read there instead of over the bus.
    if (writes_known && !skip_barriers) {
        buffer_cache.ShadowGpuWrites(copy_writes);
    }
    scheduler.KickRecording();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    // bbport: handed to the recording thread like direct dispatches.
    const bool pipelined = UseDrawPipe() && !BbToggle::Disabled(BbToggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (pipelined) {
        const IndirectDraw indirect = {address + offset, 0, size, 1};
        PostDraw(pipeline, nullptr, false, 0, &CsRegs(), &indirect);
        return;
    }
    DispatchIndirectRecord(pipeline, address + offset, size);
}

void Rasterizer::DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address,
                                        u32 size) {
    ++work_after_early;
    buffer_cache.NewPacket();
    scheduler.PopPendingOperations();
    if (!BindResources(pipeline)) {
        return;
    }
    const auto [buffer, base] = buffer_cache.ObtainBuffer(address, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    // Group counts checked on the GPU first (vk_indirect_guard.h); recorded before the
    // pipeline's bindings, which the check's own pipeline would disturb.
    vk::Buffer args = buffer->Handle();
    u64 args_offset = base;
    if (IndirectGuard::Enabled()) {
        if (!indirect_guard) {
            indirect_guard = std::make_unique<IndirectGuard>(instance, scheduler);
        }
        std::tie(args, args_offset) = indirect_guard->CheckArgs(*buffer, base);
    }
    pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                            {buffer_infos.data(), buffer_infos.size()});

    const vk::Pipeline handle = pipeline->Handle();
    const vk::Buffer raw_args = buffer->Handle();
    const u64 raw_offset = base;
    const auto& cs_crumb = pipeline->GetStage(Shader::SwStage::Compute);
    const u32 args_slot = Breadcrumbs::ArgsSlot();
    // bbport (macOS): built only with breadcrumbs on (RecordCrumbWith).
    const auto make_crumb = [&] {
        return Breadcrumbs::Crumb{
            .kind = Breadcrumbs::Kind::DispatchIndirect,
            .hash = {cs_crumb.pgm_hash, 0},
            .program = {cs_crumb.ProgramBase(), 0},
            .address = address,
            .count = {buffer_cache.IsInPlace(address, size) ? 1u : 0u, 0, 0},
            .args_slot = args_slot,
        };
    };
    scheduler.RecordCrumbWith(make_crumb, [=](vk::CommandBuffer cmdbuf) {
        Breadcrumbs::CopyArgs(cmdbuf, raw_args, raw_offset, args_slot); // what the game wrote
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatchIndirect(args, args_offset);
    });

    ResetBindings(true);
    scheduler.KickRecording();
}

} // namespace Vulkan
