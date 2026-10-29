// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport cp: the command processor. Games submit command buffers (graphics: draw and constant
// engine buffers; compute: ring buffers per queue); one thread (bb:GpuComm) decodes them in order,
// keeps the register state, hands draws and dispatches to the renderer and does what the other
// packets ask (fences, memory writes, waits, predication, the constant RAM).
//
// Each submission is a walk over its buffer and the indirect buffers it calls, kept as a stack
// of positions: a packet that has to wait (WAIT_REG_MEM, MEM_SEMAPHORE, REWIND, the constant
// engine) leaves the walk where it is, and the thread goes on with the other queues' work, as the
// hardware's queues run side by side. Commands from other threads (SendCommand) run between
// packets.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <thread>
#include <vector>

#include "common/slot_vector.h"
#include "common/types.h"
#include "common/unique_function.h"
#include "cp/cp_packets.h"
#include "platform/gnm/gnm.h"
#include "gcn/regs_targets.h"
#include "gcn/regs.h"

namespace Vulkan {
class Rasterizer;
class DrawPreparation;
} // namespace Vulkan

namespace BbVideoOut {
struct Port;
}

namespace BbCp::Diagnostics {
struct SubmittedCopy;
}

namespace BbCp {

using Gcn::CbDbExtent;
using Gcn::ComputeProgram;
using Gcn::RegDirty;
using Gcn::Regs;

class CommandProcessor {
public:
    static constexpr u32 GfxQueueId = 0;
    static constexpr u32 NumGfxRings = 1;     ///< the high-priority ring is the system's
    static constexpr u32 NumComputePipes = 7; ///< the eighth is the system's
    static constexpr u32 NumQueuesPerPipe = 8;
    static constexpr u32 NumComputeRings = NumComputePipes * NumQueuesPerPipe;
    static constexpr u32 NumTotalQueues = NumGfxRings + NumComputeRings;
    /// Top-level graphics submissions are numbered for the draw preparation workers.
    static constexpr u64 NoSeq = ~0ull;

    CommandProcessor();
    ~CommandProcessor();

    /// force_copy owns transient internal commands even when guest buffer copying is disabled.
    void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb, bool force_copy = false);
    void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb);
    /// sceGnmSubmitDone. `frame`: the guest's frame number, reported finished
    /// (BbGnm::NoteFramesRetired) once the GPU has done everything submitted so far.
    void SubmitDone(u64 frame = 0) noexcept;
    void WaitGpuIdle() noexcept;
    /// Nothing to decode, and (BB_WORK_RETIRED, default) the draw recording thread and the fences
    /// it deferred are done too. sceGnmSubmitDone does not block the guest when this is true.
    bool IsGpuIdle() const;

    void SetVoPort(BbVideoOut::Port* port) {
        vo_port = port;
    }
    void BindRasterizer(Vulkan::Rasterizer* rasterizer_) {
        rasterizer = rasterizer_;
    }

    /// Runs `func` on the command processor's thread, between packets.
    template <bool wait_done = false>
    void SendCommand(auto&& func) {
        if constexpr (wait_done) {
            std::binary_semaphore done{0};
            {
                std::scoped_lock lock{submit_mutex};
                commands.emplace_back([&done, &func] {
                    func();
                    done.release();
                });
                ++num_commands;
                submit_cv.notify_one();
            }
            done.acquire();
        } else {
            std::scoped_lock lock{submit_mutex};
            commands.emplace_back(std::move(func));
            ++num_commands;
            submit_cv.notify_one();
        }
    }

    /// The compute program of the queue being decoded.
    ComputeProgram& GetCsRegs() {
        return queues[current_queue < 0 ? 0 : current_queue].cs_state;
    }

    std::thread::id GetGpuCommandProcessorThread() const {
        return thread_id;
    }

    /// A compute ring the game mapped (sceGnmMapComputeQueue).
    struct AscQueueInfo {
        static constexpr size_t Pm4BufferSize = 1024;
        VAddr map_addr;
        u32* read_addr;
        u32 ring_size_dw;
        u32 pipe_id;
        std::array<u32, Pm4BufferSize> tmp_packet; ///< a packet split by the ring's end
        u32 tmp_dwords;
    };
    Common::SlotVector<AscQueueInfo> asc_queues{};

    /// The register-writing packets' effect on `regs` (shared with the draw preparation workers,
    /// which replay the stream): the blocks written go to `dirty`, the packet into `checksum`
    /// (equal checksums at a draw mean equal register files).
    /// `touched`: also marks the blocks there (the reset of a CLEAR_STATE then restores only them).
    static void ApplyGraphicsRegisterPacket(Regs& regs, const u32* packet, u64& checksum,
                                            RegDirty* dirty = nullptr,
                                            Gcn::RegTouched* touched = nullptr, bool hash = true);
    static u64 HashRegisterPacket(u64 checksum, const u32* words, u32 count);

    Regs regs{};
    /// Render target extents from the driver's NOP hints after CB/DB setup.
    std::array<CbDbExtent, Gcn::NUM_COLOR_BUFFERS> last_cb_extent{};
    CbDbExtent last_db_extent{};
    /// Register blocks written since the last draw handed to the draw recording thread.
    RegDirty pipe_dirty;
    /// `regs`' blocks written since the last CLEAR_STATE (Gcn::ResetRegs).
    Gcn::RegTouched gfx_touched;
    /// Running checksum of graphics-register packets (ApplyGraphicsRegisterPacket).
    u64 gfx_reg_checksum{};

private:
    struct Level {
        const u32* begin;
        const u32* at;
        const u32* end;
        int depth;
        bool prepared; ///< the submission's own buffer: its direct draws were prepared ahead
        bool entered;  ///< its start seen (statistics scan, constant engine counters reset)
        const u32* prev_at = nullptr; ///< the packet walked before this one, and the header it was walked with
        u32 prev_header = 0;
        int recoveries = 0;
    };
    /// A command buffer and the indirect buffers it called, as a stack.
    struct Walk {
        std::array<Level, 10> levels;
        int top = -1;
        void Push(std::span<const u32> buffer, int depth, bool prepared = false) {
            ASSERT_MSG(top + 1 < int(levels.size()), "indirect buffers nested too deep");
            const u32* begin = buffer.data();
            levels[++top] = {begin, begin, begin + buffer.size(), depth, prepared, false, nullptr, 0, 0};
        }
        bool Done() const {
            return top < 0;
        }
    };
    /// The constant engine's counters (shared by the CCB and DCB of a submission).
    struct ConstantEngine {
        u32 ce_count = 0;
        u32 de_count = 0;
        void Reset() {
            ce_count = de_count = 0;
        }
    };
    using SubmittedCopy = Diagnostics::SubmittedCopy;
    /// One submission of any queue.
    struct Work {
        std::span<const u32> dcb; ///< graphics: the draw buffer; compute: the ring part
        std::span<const u32> ccb; ///< the constant engine's buffer
        Walk walk;
        Walk ce;
        u32 queue = 0;
        u64 seq = NoSeq;
        u64 stamp = 0; ///< BbStats::guest_submit_stamp when made
        bool started = false;
        bool ce_started = false;
        std::shared_ptr<SubmittedCopy> copy;
        /// The packet that waited, tried again next time (its registers are applied already).
        const u32* waiting_at = nullptr;
        std::chrono::steady_clock::time_point wait_start{};
        bool wait_met_at_once = false;
        /// Compute: a packet completed from the ring's end and start, and the dwords of the
        /// level it takes.
        std::vector<u32> assembled;
        u32 assembled_used = 0;
        /// Compute: DMA_DATA that patch a later DISPATCH_DIRECT's dimensions (an indirect
        /// dispatch).
        struct IndirectPatch {
            const u32* packet;
            VAddr args;
        };
        std::vector<IndirectPatch> patches;
    };
    enum class Status { Done, Wait };
    enum class Step { Next, Moved, Wait };

    struct Queue {
        std::mutex mutex;
        std::deque<std::unique_ptr<Work>> works;
        ComputeProgram cs_state{};
    };

    // Thread.
    void Run(std::stop_token stop);
    /// BB_STALL_REPORT (default on, =0 off): when the queues make no progress, or the guest
    /// submits nothing, for 5 s, prints once what each queue is at, the fences waiting for the
    /// GPU and the last labels written (a guest waiting for a label that never comes).
    void StallWatch(std::stop_token stop);
    void ReportStall(const char* why);
    void WaitForWork(std::stop_token stop);
    void RunCommands();
    void AfterBatch();
    void SignalDecodedFrames();
    void SignalGpuIdle(u64 generation);
    size_t QueueSize(u32 queue);

    // Graphics.
    Status RunGraphics(Work& work);
    void BeginGraphics(Work& work);
    void EndGraphics(Work& work);
    Step Graphics(Work& work, Level& level, const u32* packet, u32 body_words);
    Status RunConstantEngine(Work& work);
    void SetContextRegHints(const u32* body, u32 body_words);
    void Draw(Level& level, Op op, const u32* packet, u32 body_words);
    bool WaitRegMemGraphics(Work& work, const u32* body, bool retry);
    void CopyData(const u32* body);
    bool PredicationSkips(u32 header);
    void SyncForCpuRead(VAddr address, u64 size);

    // Compute.
    Status RunCompute(Work& work);
    Step Compute(Work& work, Level& level, const u32* packet, u32 body_words);
    void ReleaseMemory(const ReleaseMem& release, u32 pipe_id);

    // Fences handed to the draw recording thread: met in stream order for WAIT_REG_MEM.
    void NotePending(VAddr address, u64 value, u32 bytes, bool eop);
    void NotePendingEop(const EventWriteEop& eop);
    void NotePendingEos(const EventWriteEos& eos);
    void NotePendingWrite(VAddr address, const u32* data, u32 bytes);
    bool PendingFenceValue(VAddr address, u32& value);
    void DropReachedFences();

    Vulkan::Rasterizer* rasterizer{};
    BbVideoOut::Port* vo_port{};
    const bool guest_markers_enabled;

    std::array<Queue, NumTotalQueues> queues{};
    u32 num_mapped_queues{1}; ///< graphics, and compute up to the highest queue used
    int current_queue = -1;

    VAddr indirect_args_addr{};
    u32 num_counter_pairs = 8; ///< occlusion counter pairs (depth blocks) of a base PS4
    u64 pixel_counter{};
    ConstantEngine cblock;
    static std::array<u8, 48 * 1024> constant_ram;

    /// SET_PREDICATION: evaluated at the first predicated draw or dispatch after it, once the GPU
    /// has done the work before (the results are GPU-written).
    struct Predication {
        struct Test {
            u32 op;
            VAddr address;
        };
        std::vector<Test> tests; ///< ORed (continued predicates)
        bool active = false;
        bool evaluated = false;
        bool skip = false;
        bool draw_if_true = true;
        bool hint_no_wait = false;
        u64 skipped = 0;
    } predication;

    struct PendingFence {
        VAddr address;
        u64 data;
        u32 num_bytes;
        u64 position;
    };
    std::deque<PendingFence> pending_fences;

    std::jthread thread;
    std::jthread stall_watch;
    std::thread::id thread_id;
    u32 thread_tid = 0;
    std::mutex submit_mutex;
    std::condition_variable_any submit_cv;
    std::deque<Common::UniqueFunction<void>> commands;
    std::atomic<u32> num_commands{0};
    std::atomic<u32> num_submits{0};
    std::atomic<bool> submit_done{false};
    std::atomic<bool> work_retired{true};
    // Under submit_mutex.
    u64 submissions_total = 0;
    u64 decoded_total = 0;
    std::deque<std::pair<u64, u64>> frame_ends; ///< (submissions, frame) per sceGnmSubmitDone
    u64 gfx_submit_seq = 0;
    Work* last_decoded = nullptr; ///< BB_TIMELINE
};

} // namespace BbCp
