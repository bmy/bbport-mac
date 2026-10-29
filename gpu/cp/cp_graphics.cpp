// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport cp: graphics submissions: the draw command buffer with the indirect buffers it calls,
// and the constant engine's buffer beside it.

#include <chrono>
#include "shim/bb_env.h"
#include <thread>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/resource.h>

#include <fmt/format.h>

#include "bbport_ce_stats.h"
#include "bbport_free_check.h"
#include "bbport_toggles.h"
#include "bbport_write_log.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/memory.h"
#include "cp/command_processor.h"
#include "cp/cp_diagnostics.h"
#include "cp/cp_packets.h"
#include "cp/cp_tasks.h"
#include "cp/pm4_selftest.h"
#include "platform/interrupts.h"
#include "platform/videoout/videoout.h"
#include "render/vk_draw_pipe.h"
#include "render/vk_draw_prep.h"
#include "buffers/buffer_cache.h"
#include "render/rasterizer.h"

namespace BbCp {

namespace {

using Vulkan::DrawPipe;

constexpr u32 DbZInfo = 0xA010;
constexpr u32 CbColor0Base = 0xA318;
constexpr u32 CbColor0Cmask = 0xA31F;
constexpr u32 CbColorStride = 0xF;
constexpr u32 NopHeader = 0xC0001000; ///< a NOP with one body dword

constexpr u32 ComputeShOffset = 0x200;
constexpr u32 ComputeWords = sizeof(ComputeProgram) / 4;


bool MemSemaphoreInOrder() {
    static const bool on = !bb_env_off("BB_MEM_SEMAPHORE_ORDER");
    return on;
}

/// The command buffer this thread walks was repaired: its draws and dispatches are skipped
/// (CommandProcessor::Graphics).
thread_local bool skip_repaired_work = false;

bool Pipelined(Op op) {
    switch (op) {
    case Op::Nop:
    case Op::ContextControl:
    case Op::ClearState:
    case Op::SetConfigReg:
    case Op::SetContextReg:
    case Op::SetShReg:
    case Op::SetUconfigReg:
    case Op::IndexType:
    case Op::NumInstances:
    case Op::IndexBase:
    case Op::IndexBufferSize:
    case Op::SetBase:
    case Op::DrawIndex2:
    case Op::DrawIndexOffset2:
    case Op::DrawIndexAuto:
    case Op::IndirectBuffer:
    case Op::AcquireMem:
    case Op::PfpSyncMe:
    case Op::IncrementDeCounter:
    case Op::WaitOnCeCounter:
    case Op::EventWrite:     // occlusion results only, no rasterizer state
    case Op::EventWriteEop:  // in order on the recording thread
    case Op::EventWriteEos:
    case Op::WaitRegMem:     // waits for the recording thread only while unmet
    case Op::DispatchDirect: // handed over like draws
    case Op::DrawIndirect:
    case Op::DrawIndirectMulti:
    case Op::DrawIndexIndirect:
    case Op::DrawIndexIndirectMulti:
    case Op::DrawIndexIndirectCountMulti:
    case Op::DispatchIndirect:
    case Op::WriteData: // in order on the recording thread
    case Op::DmaData:
        return true;
    default:
        return false;
    }
}

} // namespace


namespace {
/// BB_CE_WATCH=addr (diagnostics): constant RAM dumps and WRITE_DATA packets over that address.
void WatchCommandWrite(const char* what, VAddr address, u64 bytes, const void* data) {
    static const VAddr watched = [] {
        const char* env = bb_env("BB_CE_WATCH");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    static std::atomic<u32> printed{0};
    if (!watched || watched < address || watched >= address + bytes ||
        printed.fetch_add(1, std::memory_order_relaxed) >= 3000) {
        return;
    }
    const auto* f = static_cast<const float*>(data) + (watched - address) / 4;
    std::printf("CE watch: gpu frame %llu %s %#llx+%llu: %g %g %g %g %g %g\n",
                (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed), what,
                (unsigned long long)address, (unsigned long long)bytes, f[0], f[1], f[2], f[3],
                f[4], f[5]);
}
} // namespace

// ---- Submission start and end ----------------------------------------------------------------

void CommandProcessor::BeginGraphics(Work& work) {
    work.started = true;
    const bool top = work.seq != NoSeq;
    if (top && rasterizer) {
        rasterizer->GetDrawPreparation().BeginSubmission(work.seq, regs, gfx_reg_checksum);
    }
    if (Diagnostics::DcbStatsEnabled()) {
        Diagnostics::ScanDcb(work.dcb, 0);
    }
    cblock.Reset();
    // The constant engine first, so its constants are ready.
    if (!work.ccb.empty()) {
        work.ce.Push(work.ccb, 0);
        work.ce_started = true;
        RunConstantEngine(work);
    }
    // BB_PM4_SELFTEST: test packets before and after the submission, as indirect buffers.
    Pm4SelfTest::Injection selftest{};
    if (top && Pm4SelfTest::Enabled()) {
        selftest = Pm4SelfTest::Next(
            u32(work.dcb.size()),
            {.sync_for_cpu_read = [this](VAddr address, u64 size) { SyncForCpuRead(address, size); },
             .predicated_skips = [this] { return predication.skipped; },
             .occlusion_events = [this] { return rasterizer ? rasterizer->OcclusionEvents() : 0; },
             .submit_compute = [this](VAddr ring, u32 ring_dwords, VAddr read_pointer,
                                      std::span<const u32> words) {
                 const auto slot = asc_queues.insert(ring, reinterpret_cast<u32*>(read_pointer),
                                                     ring_dwords, 1u);
                 *asc_queues[slot].read_addr = 0;
                 SubmitAsc(slot.index + 1, words);
             }});
    }
    if (!selftest.after.empty()) {
        work.walk.Push(selftest.after, 1);
    }
    work.walk.Push(work.dcb, 0, top);
    work.walk.levels[work.walk.top].entered = true; // scanned and reset above
    if (!selftest.before.empty()) {
        work.walk.Push(selftest.before, 1);
    }
}

void CommandProcessor::EndGraphics(Work& work) {
    // The constant engine's buffer to its end.
    while (work.ce_started && RunConstantEngine(work) == Status::Wait) {
    }
    if (work.seq == NoSeq) {
        return;
    }
    if (rasterizer) {
        rasterizer->RetireSubmission(); // prepared draws live until the recording thread is past
        rasterizer->GetDrawPreparation().EndSubmission();
        rasterizer->SubmitForSignals();
    }
    if (BbStats::enabled) {
        BbStats::submissions.fetch_add(1, std::memory_order_relaxed);
        if (rusage usage{}; getrusage(RUSAGE_THREAD, &usage) == 0) {
            BbStats::gpu_user_us.store(u64(usage.ru_utime.tv_sec) * 1000000 + usage.ru_utime.tv_usec,
                                       std::memory_order_relaxed);
            BbStats::gpu_sys_us.store(u64(usage.ru_stime.tv_sec) * 1000000 + usage.ru_stime.tv_usec,
                                      std::memory_order_relaxed);
            BbStats::gpu_invol_switches.store(usage.ru_nivcsw, std::memory_order_relaxed);
            BbStats::gpu_vol_switches.store(usage.ru_nvcsw, std::memory_order_relaxed);
            BbStats::gpu_minor_faults.store(usage.ru_minflt, std::memory_order_relaxed);
        }
    }
    if (work.copy && BbFreeCheck::Enabled()) {
        Diagnostics::CheckSubmittedCopy(*work.copy, work.seq);
    }
}

// ---- The walk --------------------------------------------------------------------------------

CommandProcessor::Status CommandProcessor::RunGraphics(Work& work) {
    if (!work.started) {
        BeginGraphics(work);
    }
    Walk& walk = work.walk;
    while (!walk.Done()) {
        RunCommands();
        Level& level = walk.levels[walk.top];
        if (!level.entered) {
            // An indirect (or test) buffer begins.
            level.entered = true;
            if (Diagnostics::DcbStatsEnabled()) {
                Diagnostics::ScanDcb({level.at, level.end}, level.depth);
            }
            cblock.Reset();
        }
        if (level.at >= level.end) {
            --walk.top;
            continue;
        }
        const u32 header = *level.at;
        const u32 type = header >> 30;
        if (type == 2) { // filler
            ++level.at;
            continue;
        }
        if (type != 3) {
            // The game may still be writing this buffer (a packet written after the walk began):
            // it is read again for a few milliseconds before it counts as bad. A buffer that stays
            // bad is reported as before, with the time it was waited.
            {
                static const long wait_us = [] {
                    const char* env = bb_env("BB_PM4_WAIT_US");
                    return env ? std::atol(env) : 20000L;
                }();
                const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(wait_us);
                while (*level.at >> 30 != 3 && *level.at >> 30 != 2 &&
                       std::chrono::steady_clock::now() < until) {
                    std::this_thread::sleep_for(std::chrono::microseconds(50));
                }
                if (*level.at >> 30 == 3 || *level.at >> 30 == 2) {
                    std::printf("PM4: the dword at %p was not a packet header at first, now %#x (the "
                                "game was still writing)\n", static_cast<const void*>(level.at), *level.at);
                    continue;
                }
            }
            // The packet before was walked with another header than memory holds now: the game wrote
            // it again after (or while) we read it, the tail of its body counted as the next packets.
            // A register-setting packet is walked again with the header as it is now (setting them
            // twice is the same as once). BB_PM4_RECOVER=0: stop as before.
            static const bool recover = !bb_env_off("BB_PM4_RECOVER");
            if (recover && level.prev_at && level.recoveries < 4 && *level.prev_at != level.prev_header) {
                const Op prev_op = Header{level.prev_header}.Opcode();
                if (prev_op == Op::SetShReg || prev_op == Op::SetContextReg ||
                    prev_op == Op::SetUconfigReg) {
                    std::printf("PM4: the header at %p changed after it was walked (%#x, now %#x): the packet "
                                "is walked again\n", static_cast<const void*>(level.prev_at),
                                level.prev_header, *level.prev_at);
                    ++level.recoveries;
                    level.at = level.prev_at;
                    level.prev_at = nullptr;
                    continue;
                }
            }
            Diagnostics::ReportBadPacket(reinterpret_cast<uintptr_t>(level.begin),
                                         size_t(level.end - level.begin), level.at, work.seq,
                                         level.depth, work.copy.get());
            UNREACHABLE_MSG("PM4 type {} packet ({:#x})", type, header);
        }
        level.prev_at = level.at;
        level.prev_header = header;
        const u32 body_words = Header{header}.BodyWords();
        if (level.at + 1 + body_words > level.end) {
            LOG_ERROR(Render, "PM4 packet past the end of its buffer ({} dwords left)",
                      level.end - level.at);
            level.at = level.end;
            continue;
        }
        switch (Graphics(work, level, level.at, body_words)) {
        case Step::Next:
            level.at += 1 + body_words;
            break;
        case Step::Moved:
            break;
        case Step::Wait:
            return Status::Wait;
        }
    }
    EndGraphics(work);
    return Status::Done;
}

// ---- Graphics packets ------------------------------------------------------------------------

CommandProcessor::Step CommandProcessor::Graphics(Work& work, Level& level, const u32* packet,
                                                  u32 body_words) {
    const u32 header = packet[0];
    const u32* body = packet + 1;
    const Op op = Header{header}.Opcode();
    // A repaired command buffer (RepairSubmittedCopy) draws and dispatches nothing: the corrupt
    // packets left registers with garbage (laptop: right after a repair a full-screen draw wrote to
    // address 0 and the device was lost, 2026-10-25). A frame misses some of its work instead.
    // BB_REPAIRED_DRAWS=1: they run.
    static const bool repaired_draws = bb_env_on("BB_REPAIRED_DRAWS");
    skip_repaired_work = work.copy && work.copy->repaired && !repaired_draws;
    // A packet that waited is tried again: its registers and drain are done already.
    const bool retry = work.waiting_at == packet;
    if (!retry) {
        // The checksum only matters to prepared draws (none without preparation workers).
        ApplyGraphicsRegisterPacket(regs, packet, gfx_reg_checksum, &pipe_dirty, &gfx_touched,
                                    rasterizer && rasterizer->GetDrawPreparation().Enabled());
        // Not before earlier fences either: these may write guest memory.
        if (rasterizer && !Pipelined(op) &&
            !(op == Op::DmaData && DmaData::From(body).DstLow() == 0x3022C)) {
            rasterizer->DrainDrawPipe(u32(op));
        }
    }
    work.waiting_at = nullptr;
    const auto wait = [&] {
        work.waiting_at = packet;
        return Step::Wait;
    };

    switch (op) {
    case Op::Nop: {
        if (body_words == 1) {
            break;
        }
        switch (body[0]) {
        case NopPayload::PatchedFlip:
            // After the writes before it (the display buffer's label, maybe still queued).
            if (rasterizer) {
                rasterizer->RunInOrder(&Tasks::FlipTask, packet, sizeof(u32),
                                       BbToggle::PipelinedMemoryWrites, false);
            } else {
                BbIrq::Raise(BbIrq::Line::GfxFlip);
            }
            break;
        case NopPayload::MarkerPush:
        case NopPayload::ColorMarkerPush:
            if (guest_markers_enabled && rasterizer) {
                const u32 bytes = (body_words - 1) * 2;
                const std::string_view label{reinterpret_cast<const char*>(body + 1), bytes};
                if (body[0] == NopPayload::MarkerPush) {
                    rasterizer->ScopeMarkerBegin(label, true);
                } else {
                    u32 color;
                    std::memcpy(&color, reinterpret_cast<const u8*>(body + 1) + bytes,
                                sizeof(color));
                    rasterizer->ScopedMarkerInsertColor(label, color, true);
                }
            }
            break;
        case NopPayload::MarkerPop:
            if (guest_markers_enabled && rasterizer) {
                rasterizer->ScopeMarkerEnd(true);
            }
            break;
        default:
            break;
        }
        break;
    }
    case Op::ContextControl:
    case Op::ClearState:
    case Op::SetConfigReg:
    case Op::SetUconfigReg:
    case Op::IndexType:
    case Op::NumInstances:
    case Op::IndexBase:
    case Op::IndexBufferSize:
    case Op::AcquireMem:
    case Op::PfpSyncMe:
        break; // registers only (ApplyGraphicsRegisterPacket)
    case Op::SetContextReg:
        SetContextRegHints(body, body_words);
        break;
    case Op::SetShReg: {
        const u32 offset = body[0] & 0xFFFF;
        if (offset >= ComputeShOffset && offset <= ComputeShOffset + ComputeWords) {
            const u32 bytes = (body_words - 1) * sizeof(u32);
            ASSERT(bytes <= sizeof(ComputeProgram));
            auto* state = reinterpret_cast<u32*>(&queues[GfxQueueId].cs_state);
            std::memcpy(state + (offset - ComputeShOffset), body + 1, bytes);
        }
        break;
    }
    case Op::SetPredication: {
        const auto set = SetPredication::From(body);
        if (set.Operation() == SetPredication::Clear) {
            predication.active = false;
            predication.tests.clear();
            break;
        }
        if (!set.Continue() || !predication.active) {
            predication.tests.clear();
        }
        predication.tests.push_back({u32(set.Operation()), set.Address()});
        predication.active = true;
        predication.evaluated = false;
        predication.draw_if_true = set.DrawIfTrue();
        predication.hint_no_wait = set.HintNoWait();
        break;
    }
    case Op::DrawIndex2:
    case Op::DrawIndexOffset2:
    case Op::DrawIndexAuto:
    case Op::DrawIndirect:
    case Op::DrawIndirectMulti:
    case Op::DrawIndexIndirect:
    case Op::DrawIndexIndirectMulti:
    case Op::DrawIndexIndirectCountMulti:
        Draw(level, op, packet, body_words);
        break;
    case Op::DispatchDirect: {
        if (PredicationSkips(header) || skip_repaired_work) {
            break;
        }
        auto& cs = GetCsRegs();
        cs.dim_x = body[0];
        cs.dim_y = body[1];
        cs.dim_z = body[2];
        cs.dispatch_initiator = body[3];
        if (rasterizer && (cs.dispatch_initiator & 1)) {
            const void* at = packet;
            rasterizer->ScopeMarker("gfx:{}:DispatchDirect", fmt::make_format_args(at),
                                    [&] { rasterizer->DispatchDirect(); });
        }
        break;
    }
    case Op::DispatchIndirect: {
        if (PredicationSkips(header) || skip_repaired_work) {
            break;
        }
        if (rasterizer && (GetCsRegs().dispatch_initiator & 1)) {
            const void* at = packet;
            rasterizer->ScopeMarker("gfx:{}:DispatchIndirect", fmt::make_format_args(at), [&] {
                rasterizer->DispatchIndirect(indirect_args_addr, body[0], 3 * sizeof(u32));
            });
        }
        break;
    }
    case Op::SetBase: {
        const auto set = SetBase::From(body);
        ASSERT(set.Index() == SetBase::DrawIndexIndirectPatchTable);
        indirect_args_addr = set.Address();
        break;
    }
    case Op::EventWrite: {
        const auto event = EventWrite::From(body, body_words);
        if (event.Type() == EventWrite::SoVgtStreamoutFlush ||
            event.Index() != EventWrite::IndexZpassDone ||
            (event.Type() != EventWrite::PixelPipeStatDump &&
             event.Type() != EventWrite::ZpassDone)) {
            break;
        }
        // Occlusion counters: the samples that passed, counted by Vulkan occlusion queries and
        // written by the GPU in stream order (vk_occlusion.h).
        if (rasterizer && Vulkan::Rasterizer::OcclusionTranslated()) {
            const Tasks::ZpassEvent zpass{event.Address(), num_counter_pairs};
            rasterizer->RunInOrder(&Tasks::ZpassDoneTask, &zpass, sizeof(zpass),
                                   BbToggle::PipelinedMemoryWrites, false);
            break;
        }
        // Else counters that always grow: everything visible.
        constexpr u64 Valid = 0x8000000000000000ull;
        constexpr u64 Step = 0x2FFFFFFull;
        auto* counters = reinterpret_cast<u64*>(event.Address());
        for (u32 i = 0; i < num_counter_pairs; ++i, counters += 2) {
            *counters = pixel_counter | Valid;
            if (rasterizer) {
                rasterizer->NoteLateCommandWrite(reinterpret_cast<VAddr>(counters), counters,
                                                 sizeof(u64));
            }
        }
        pixel_counter += Step;
        break;
    }
    case Op::EventWriteEos: {
        const auto eos = EventWriteEos::From(body);
        if (BbFreeCheck::Enabled()) {
            BbFreeCheck::NoteFenceDecoded(eos.Address(), eos.Value(), packet, packet, work.seq);
        }
        if (rasterizer && rasterizer->RunInOrder(&Tasks::EventWriteEosTask, &eos, sizeof(eos),
                                                 BbToggle::PipelinedTasks, false)) {
            NotePendingEos(eos);
        }
        break;
    }
    case Op::EventWriteEop: {
        const auto eop = EventWriteEop::From(body);
        if (BbFreeCheck::Enabled()) {
            BbFreeCheck::NoteFenceDecoded(eop.Address(), eop.Value(), packet, packet, work.seq);
        }
        if (!rasterizer) {
            Tasks::SignalEop(eop, 0);
        } else if (rasterizer->RunInOrder(&Tasks::EventWriteEopTask, &eop, sizeof(eop),
                                          BbToggle::PipelinedTasks, false)) {
            NotePendingEop(eop);
        }
        break;
    }
    case Op::DmaData: {
        const auto dma = DmaData::From(body);
        if (dma.DstLow() == 0x3022C || !rasterizer) {
            break; // a register the translator has no use for
        }
        // What the copy reads is known here: a later DUMP_CONST_RAM waits only if it overlaps.
        if (BbCeStats::Enabled() && dma.DstIsMemory()) {
            BbCeStats::NoteWrite(BbCeStats::Dma, dma.DstAddress(), dma.Bytes());
        }
        if (dma.SrcIsMemory()) {
            rasterizer->NotePendingRead(dma.SrcAddress(), dma.Bytes());
        }
        if (rasterizer->RunInOrder(&Tasks::DmaDataTask, &dma, sizeof(dma),
                                   BbToggle::PipelinedMemoryWrites, false) &&
            dma.DstIsMemory()) {
            rasterizer->NotePendingGpuWrite(dma.DstAddress(), dma.Bytes());
        }
        if (dma.DstIsMemory()) {
            rasterizer->NoteCopyIntoMemory(dma.DstAddress(), dma.Bytes());
        }
        break;
    }
    case Op::WriteData: {
        const u32 dst = WriteData::DstSel(body);
        ASSERT(dst == WriteData::DstMemoryAsync || dst == WriteData::DstMemory);
        ASSERT(!WriteData::OneAddress(body));
        const VAddr address = WriteData::Address(body);
        const u32 bytes = WriteData::Bytes(body_words);
        const u32* values = WriteData::Data(body);
        if (!rasterizer) {
            BbFreeCheck::Check(address, bytes, values, BbFreeCheck::WriteData);
            std::memcpy(reinterpret_cast<void*>(address), values, bytes);
            break;
        }
        // In order with the draws (on the draw recording thread when in use).
        if (BbCeStats::Enabled()) {
            BbCeStats::NoteWrite(BbCeStats::WriteData, address, bytes);
        }
        BbWriteLog::NoteIntent(address, values, bytes, BbWriteLog::WriteDataIntent);
        WatchCommandWrite("WriteData", address, bytes, values);
        if (rasterizer->RunInOrder(&Tasks::WriteDataTask, packet, (body_words + 1) * sizeof(u32),
                                   BbToggle::PipelinedMemoryWrites, false)) {
            NotePendingWrite(address, values, bytes);
            rasterizer->NotePendingGpuWrite(address, bytes);
        }
        break;
    }
    case Op::CopyData:
        CopyData(body);
        break;
    case Op::MemSemaphore: {
        const auto semaphore = MemSemaphore::From(body);
        if (semaphore.Signals()) {
            if (rasterizer && MemSemaphoreInOrder()) {
                rasterizer->RunInOrder(&Tasks::MemSemaphoreSignalTask, &semaphore,
                                       sizeof(semaphore), BbToggle::PipelinedTasks, false);
            } else {
                semaphore.Signal();
            }
            break;
        }
        while (!semaphore.Signaled()) {
            // A signal in stream order may wait for work still in the draw pipe or not submitted
            // yet; this thread would spin on it forever.
            if (rasterizer && !rasterizer->DrawPipeIdle()) {
                rasterizer->DrainDrawPipe(u32(op));
                continue;
            }
            if (rasterizer && rasterizer->HasUnsubmittedSignals()) {
                rasterizer->Flush();
            }
            return wait();
        }
        semaphore.Take();
        break;
    }
    case Op::Rewind:
        // Valid once the game has written the packets after it.
        if (rasterizer && !(*reinterpret_cast<const volatile u32*>(body) >> 31)) {
            return wait();
        }
        break;
    case Op::WaitRegMem:
        if (!WaitRegMemGraphics(work, body, retry)) {
            return wait();
        }
        break;
    case Op::IndirectBuffer: {
        const auto ib = IndirectBuffer::From(body);
        level.at += 1 + body_words;
        work.walk.Push({ib.Address(), ib.Dwords()}, level.depth + 1);
        return Step::Moved;
    }
    case Op::IncrementDeCounter:
        ++cblock.de_count;
        break;
    case Op::WaitOnCeCounter:
        while (cblock.ce_count <= cblock.de_count && work.ce_started && !work.ce.Done()) {
            RunConstantEngine(work);
        }
        // BB_CE_WAIT_LOG=1 (diagnostics): a wait the constant engine did not satisfy (its buffer
        // ended or never started): the draws after it read constants not dumped for them.
        if (cblock.ce_count <= cblock.de_count) {
            static const bool log = bb_env("BB_CE_WAIT_LOG") != nullptr;
            static std::atomic<u32> printed{0};
            if (log && printed.fetch_add(1, std::memory_order_relaxed) < 2000) {
                std::printf("CE wait unmet: gpu frame %llu ce %u de %u started %d done %d ccb %zu\n",
                            (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                            cblock.ce_count, cblock.de_count, int(work.ce_started),
                            int(work.ce.Done()), work.ccb.size());
            }
        }
        break;
    case Op::StrmoutBufferUpdate:
        LOG_WARNING(Render_Vulkan,
                    "STRMOUT_BUFFER_UPDATE not done (update memory {}, source {}, buffer {})",
                    body[0] & 1, (body[0] >> 1) & 3, (body[0] >> 8) & 3);
        break;
    case Op::GetLodStats:
        LOG_WARNING(Render_Vulkan, "GET_LOD_STATS not done");
        break;
    case Op::CondExec: {
        const auto cond = CondExec::From(body);
        if (cond.Command() != 0) {
            LOG_WARNING(Render, "COND_EXEC with a reserved command");
        }
        bool skip = !(*reinterpret_cast<const volatile u8*>(cond.Address()) != 0);
        // The condition as the command processor reads it when it gets here: a fence the
        // recording thread has yet to write (in stream order), else once the GPU has done the
        // work before this packet (this thread decodes ahead of it).
        if (rasterizer) {
            u32 value = 0;
            if (!PendingFenceValue(cond.Address(), value)) {
                SyncForCpuRead(cond.Address(), sizeof(u32));
                value = *reinterpret_cast<const volatile u32*>(cond.Address());
            }
            skip = value == 0;
        }
        if (skip) {
            level.at = std::min(level.end, level.at + 1 + body_words + cond.Dwords());
            return Step::Moved;
        }
        break;
    }
    default:
        UNREACHABLE_MSG("PM4 opcode {:#x} ({} dwords) not handled", u32(op), body_words);
    }
    return Step::Next;
}

void CommandProcessor::SetContextRegHints(const u32* body, u32 body_words) {
    // A render target's memory is in tiles: its extent is not in the CB/DB registers, and
    // computing it from pitch and slices gives larger surfaces. The driver follows the setup with
    // a NOP holding the extent; it is taken from there.
    const u32 reg = Bank::Context + (body[0] & 0xFFFF);
    const u32* values = body + 1;
    const u32 count_field = body_words - 1; // the header's count: the NOP's offset in `values`
    if (reg >= CbColor0Base && reg < CbColor0Base + 8 * CbColorStride &&
        (reg - CbColor0Base) % CbColorStride == 0) {
        const u32 target = (reg - CbColor0Base) / CbColorStride;
        if (count_field == 0x0E || count_field == 0x0D || count_field == 0x0B) {
            ASSERT_MSG(values[count_field] == NopHeader, "no extent hint after the CB setup");
            last_cb_extent[target].raw = values[count_field + 1];
        } else {
            last_cb_extent[target].raw = 0;
        }
    } else if (reg >= CbColor0Cmask && reg < CbColor0Cmask + 8 * CbColorStride &&
               (reg - CbColor0Cmask) % CbColorStride == 0) {
        if (count_field == 0x04) {
            ASSERT_MSG(values[count_field] == NopHeader, "no extent hint after the CMASK setup");
            last_cb_extent[(reg - CbColor0Cmask) / CbColorStride].raw = values[count_field + 1];
        }
    } else if (reg == DbZInfo) {
        if (count_field == 8) {
            ASSERT_MSG(values[20] == NopHeader, "no extent hint after the DB setup");
            last_db_extent.raw = values[21];
        } else {
            last_db_extent.raw = 0;
        }
    }
}

void CommandProcessor::Draw(Level& level, Op op, const u32* packet, u32 body_words) {
    const u32 header = packet[0];
    const u32* body = packet + 1;
    const void* at = packet;
    const bool direct =
        op == Op::DrawIndex2 || op == Op::DrawIndexOffset2 || op == Op::DrawIndexAuto;
    if (direct) {
        if (!rasterizer) {
            return;
        }
        // The submission's own draws were prepared ahead (taken even when skipped).
        const auto* prepared =
            level.prepared ? rasterizer->GetDrawPreparation().NextDraw() : nullptr;
        if (PredicationSkips(header) || skip_repaired_work) {
            return;
        }
        switch (op) {
        case Op::DrawIndex2:
            rasterizer->ScopeMarker("gfx:{}:DrawIndex2", fmt::make_format_args(at),
                                    [&] { rasterizer->Draw(true, 0, prepared); });
            break;
        case Op::DrawIndexOffset2:
            rasterizer->ScopeMarker("gfx:{}:DrawIndexOffset2", fmt::make_format_args(at),
                                    [&] { rasterizer->Draw(true, body[1], prepared); });
            break;
        default:
            rasterizer->ScopeMarker("gfx:{}:DrawIndexAuto", fmt::make_format_args(at),
                                    [&] { rasterizer->Draw(false, 0, prepared); });
            break;
        }
        return;
    }
    if (PredicationSkips(header) || skip_repaired_work || !rasterizer) {
        return;
    }
    const auto draw = DrawIndirect::From(body, body_words);
    const bool indexed = op != Op::DrawIndirect && op != Op::DrawIndirectMulti;
    u32 stride = indexed ? 20 : 16; // the argument records
    u32 count = 1;
    u64 count_address = 0;
    if (op == Op::DrawIndirectMulti || op == Op::DrawIndexIndirectMulti) {
        count = draw.Count();
        stride = draw.Stride();
    } else if (op == Op::DrawIndexIndirectCountMulti) {
        count = draw.MaxCount();
        stride = draw.CountStride();
        count_address = draw.CountInMemory() ? draw.CountAddress() : 0;
    }
    rasterizer->ScopeMarker("gfx:{}:DrawIndirect", fmt::make_format_args(at), [&] {
        rasterizer->DrawIndirect(indexed, indirect_args_addr, draw.DataOffset(), stride, count,
                                 count_address, draw.BaseVertexLoc(), draw.StartInstanceLoc());
    });
}

bool CommandProcessor::WaitRegMemGraphics(Work& work, const u32* body, bool retry) {
    const auto wait = WaitRegMem::From(body);
    const bool stats = Diagnostics::DcbStatsEnabled();
    // An address past any guest mapping (bits above 40) is a malformed packet the submit-time
    // repair did not catch (laptop: the next packet's NOP marker 0x68750004 as its high half):
    // passed instead of a fault reading it.
    if (wait.Memory() && (u64(wait.Address()) >> 40) != 0) {
        static std::atomic<u32> reported{0};
        if (reported.fetch_add(1, std::memory_order_relaxed) < 8) {
            std::fprintf(stderr, "WAIT_REG_MEM on a garbage address %#llx: passed\n",
                         (unsigned long long)u64(wait.Address()));
        }
        return true;
    }
    // Display buffer labels: written when the flip is done. With nothing else to run, this
    // thread sleeps until then instead of spinning.
    if (wait.Memory() && vo_port && vo_port->IsVoLabel(reinterpret_cast<const u64*>(wait.Address())) &&
        num_submits == QueueSize(GfxQueueId)) {
        // The recording thread catches up here once a frame, whether the buffer is free already
        // or not: decoding frames ahead of it (no drain when the label was free) was followed by
        // GPU hangs in another game (indirect draws with garbage arguments).
        if (rasterizer) {
            rasterizer->DrainDrawPipe(u32(Op::WaitRegMem));
        }
        if (stats) {
            Diagnostics::NoteWait(0);
        }
        vo_port->WaitVoLabel([&] { return wait.Test(regs.reg_array.data()); });
        return true;
    }
    // A fence handed to the draw recording thread counts as written: that thread runs everything
    // in stream order.
    if (u32 value; !retry && wait.Memory() && !BbToggle::Disabled(BbToggle::PendingFenceWaits) &&
                   PendingFenceValue(wait.Address(), value) && wait.TestValue(value)) {
        if (stats) {
            Diagnostics::NoteWait(1);
        }
        return true;
    }
    if (!retry) {
        work.wait_start = std::chrono::steady_clock::now();
        work.wait_met_at_once = wait.Test(regs.reg_array.data());
    }
    // A repaired buffer (RepairSubmittedCopy) may hold a wait made of a cut packet's remains,
    // waiting for a value that never comes (laptop: the queue stood still for good): it is passed
    // after BB_REPAIRED_WAIT_MS (500).
    // Likewise within 30 s of a corrupt end-of-pipe packet skipped (the label it should have
    // written never comes: the game stood still, 2026-10-26).
    const auto bad_eop_ms = BbCp::g_bad_eop_at_ms.load(std::memory_order_relaxed);
    const bool bad_eop_recent =
        bad_eop_ms != 0 &&
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count() + 1 - bad_eop_ms < 30000;
    if (((work.copy && work.copy->repaired) || bad_eop_recent) &&
        !wait.Test(regs.reg_array.data())) {
        static const auto limit = std::chrono::milliseconds([] {
            const char* env = bb_env("BB_REPAIRED_WAIT_MS");
            return env && *env ? std::atoi(env) : 500;
        }());
        if (std::chrono::steady_clock::now() - work.wait_start > limit) {
            static std::atomic<u32> told{0};
            if (told.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "WAIT_REG_MEM in a repaired command buffer gave up after %lld ms\n",
                             static_cast<long long>(limit.count()));
            }
            return true;
        }
    }
    while (!wait.Test(regs.reg_array.data())) {
        // The value may come from a fence the recording thread has yet to write: let it catch up
        // before yielding to the other queues.
        if (rasterizer && !rasterizer->DrawPipeIdle()) {
            rasterizer->DrainDrawPipe(u32(Op::WaitRegMem));
            continue;
        }
        // A fence from another queue may wait for work not submitted yet.
        if (rasterizer && rasterizer->HasUnsubmittedSignals()) {
            rasterizer->Flush();
        }
        return false;
    }
    if (stats) {
        Diagnostics::NoteWait(work.wait_met_at_once ? 2 : 3,
                              std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - work.wait_start)
                                  .count());
    }
    return true;
}

void CommandProcessor::CopyData(const u32* body) {
    // Values known when decoded (immediates, registers, the clock) are written by the GPU in
    // stream order like WRITE_DATA; copies from memory or GDS are GPU copies like DMA_DATA.
    const auto copy = BbCp::CopyData::From(body);
    const u32 bytes = copy.Bytes();
    const auto src = copy.Source();
    const auto dst = copy.Destination();
    const bool to_memory = dst == BbCp::CopyData::DstMemory || dst == BbCp::CopyData::DstL2 ||
                           dst == BbCp::CopyData::DstMemoryAsync;
    const bool from_memory = src == BbCp::CopyData::SrcMemory || src == BbCp::CopyData::SrcL2;
    std::optional<u64> value;
    switch (src) {
    case BbCp::CopyData::SrcImmediate:
        value = copy.w[1] | (bytes == 8 ? u64(copy.w[2]) << 32 : 0);
        break;
    case BbCp::CopyData::SrcRegister: {
        const u32 reg = copy.w[1] & 0x3FFFF;
        if (reg + bytes / 4 <= regs.reg_array.size()) {
            value = regs.reg_array[reg] | (bytes == 8 ? u64(regs.reg_array[reg + 1]) << 32 : 0);
        }
        break;
    }
    case BbCp::CopyData::SrcClock:
        value = GpuClock64(); // the clock of the end-of-pipe timestamps
        break;
    default:
        break;
    }
    if (src == BbCp::CopyData::SrcClock && rasterizer && to_memory && bytes == 8) {
        const VAddr target = copy.DstAddress();
        if (rasterizer->RunInOrder(&Tasks::GpuClockCopyTask, &target, sizeof(target),
                                   BbToggle::PipelinedMemoryWrites, false)) {
            rasterizer->NotePendingGpuWrite(target, bytes);
        }
        return;
    }
    const bool handled_source = value || from_memory || src == BbCp::CopyData::SrcGds;
    if (!rasterizer || !handled_source || !(to_memory || dst == BbCp::CopyData::DstGds)) {
        static bool warned = false;
        if (!std::exchange(warned, true)) {
            LOG_WARNING(Render, "COPY_DATA from {} to {} ({} bytes) not done", u32(src),
                        u32(dst), bytes);
        }
        return;
    }
    if (to_memory && value) {
        // A WRITE_DATA of the value, through its path (GPU write, fence tracking).
        alignas(8) std::array<u32, 6> packet{};
        packet[0] = (3u << 30) | ((3 + bytes / 4 - 1) << 16) | (u32(Op::WriteData) << 8);
        packet[1] = WriteData::DstMemory << 8;
        packet[2] = u32(copy.DstAddress());
        packet[3] = u32(copy.DstAddress() >> 32);
        std::memcpy(&packet[4], &*value, bytes);
        if (rasterizer->RunInOrder(&Tasks::WriteDataTask, packet.data(),
                                   (Header{packet[0]}.BodyWords() + 1) * sizeof(u32),
                                   BbToggle::PipelinedMemoryWrites, false)) {
            NotePendingWrite(copy.DstAddress(), &packet[4], bytes);
            rasterizer->NotePendingGpuWrite(copy.DstAddress(), bytes);
        }
        return;
    }
    Tasks::CopyTask task{};
    task.num_bytes = bytes;
    task.dst_gds = dst == BbCp::CopyData::DstGds;
    task.dst = task.dst_gds ? copy.w[3] : copy.DstAddress();
    if (value) {
        task.fill = true;
        task.value = *value;
    } else {
        task.src_gds = src == BbCp::CopyData::SrcGds;
        task.src = task.src_gds ? copy.w[1] : copy.SrcAddress();
        if (!task.src_gds) {
            rasterizer->NotePendingRead(task.src, bytes);
        }
    }
    rasterizer->RunInOrder(&Tasks::CopyDataTask, &task, sizeof(task),
                           BbToggle::PipelinedMemoryWrites, false);
    if (!task.dst_gds) {
        rasterizer->NotePendingGpuWrite(task.dst, bytes);
    }
}

void CommandProcessor::SyncForCpuRead(VAddr address, u64 size) {
    if (!rasterizer) {
        return;
    }
    // The command processor reads memory when it reaches the packet; the GPU work recorded before
    // it may still be queued or running (this thread decodes ahead of it).
    rasterizer->DrainDrawPipe();
    rasterizer->Finish();
    rasterizer->ReadMemory(address, size, true); // this is the thread ReadMemory would ask
}

bool CommandProcessor::PredicationSkips(u32 header) {
    if (!predication.active || !(header & 1)) {
        return false;
    }
    if (!predication.evaluated) {
        predication.evaluated = true;
        bool any = false, ready = true;
        for (const auto& test : predication.tests) {
            switch (test.op) {
            case SetPredication::ZPass: {
                SyncForCpuRead(test.address, u64(num_counter_pairs) * 16);
                // The samples between each depth block's begin and end counters.
                constexpr u64 Valid = 0x8000000000000000ull;
                const auto* counters = reinterpret_cast<const u64*>(test.address);
                u64 samples = 0;
                for (u32 i = 0; i < num_counter_pairs; ++i) {
                    const u64 begin = counters[i * 2], end = counters[i * 2 + 1];
                    if (!(begin & Valid) || !(end & Valid)) {
                        ready = false;
                        continue;
                    }
                    samples += (end & ~Valid) - (begin & ~Valid);
                }
                any |= samples != 0;
                break;
            }
            case SetPredication::Bool64:
                SyncForCpuRead(test.address, sizeof(u64));
                any |= *reinterpret_cast<const u64*>(test.address) != 0;
                break;
            case SetPredication::Bool32:
                SyncForCpuRead(test.address, sizeof(u32));
                any |= *reinterpret_cast<const u32*>(test.address) != 0;
                break;
            default:
                // PrimCount (streamout overflow): none is tracked; it held.
                any |= !predication.draw_if_true;
                break;
            }
        }
        const bool draw =
            (!ready && predication.hint_no_wait) || (predication.draw_if_true ? any : !any);
        predication.skip = !draw;
    }
    predication.skipped += predication.skip ? 1 : 0;
    return predication.skip;
}

// ---- Constant engine -------------------------------------------------------------------------

CommandProcessor::Status CommandProcessor::RunConstantEngine(Work& work) {
    Walk& walk = work.ce;
    while (!walk.Done()) {
        RunCommands();
        Level& level = walk.levels[walk.top];
        if (level.at >= level.end) {
            --walk.top;
            continue;
        }
        const u32 header = *level.at;
        if ((header >> 30) != 3) {
            UNREACHABLE_MSG("constant engine: PM4 type {} packet", header >> 30);
        }
        const u32 body_words = Header{header}.BodyWords();
        const u32* body = level.at + 1;
        switch (Header{header}.Opcode()) {
        case Op::Nop:
            break;
        case Op::WriteConstRam:
            std::memcpy(constant_ram.data() + ConstRam::WriteOffset(body), body + 1,
                        (body_words - 1) * sizeof(u32));
            break;
        case Op::DumpConstRam: {
            const VAddr address = ConstRam::DumpAddress(body);
            const u32 bytes = ConstRam::DumpBytes(body);
            const u8* from = constant_ram.data() + ConstRam::DumpOffset(body);
            if (BbCeStats::Enabled()) {
                BbCeStats::NoteDump(address, bytes);
            }
            // Earlier draws on the recording thread may still read constants here: only those
            // reading these bytes are waited for.
            if (rasterizer) {
                rasterizer->WaitForPendingReads(address, bytes);
            }
            WatchCommandWrite("DumpConstRam", address, bytes, from);
            BbFreeCheck::Check(address, bytes, from, BbFreeCheck::ConstRam);
            std::memcpy(reinterpret_cast<void*>(address), from, bytes);
            // The caches hear of it (in order, on the recording thread).
            if (rasterizer) {
                rasterizer->NoteCommandWriteInOrder(address, from, bytes);
            }
            break;
        }
        case Op::IncrementCeCounter:
            ++cblock.ce_count;
            break;
        case Op::WaitOnDeCounterDiff:
            if (cblock.de_count - cblock.ce_count >= body[0]) {
                return Status::Wait; // the draw engine catches up
            }
            break;
        case Op::IndirectBufferConst: {
            const auto ib = IndirectBuffer::From(body);
            level.at += 1 + body_words;
            walk.Push({ib.Address(), ib.Dwords()}, level.depth + 1);
            continue;
        }
        default:
            UNREACHABLE_MSG("constant engine: PM4 opcode {:#x} not handled",
                            u32(Header{header}.Opcode()));
        }
        level.at += 1 + body_words;
    }
    return Status::Done;
}

// ---- Fences handed to the draw recording thread ----------------------------------------------

void CommandProcessor::DropReachedFences() {
    while (!pending_fences.empty() &&
           rasterizer->DrawPipeReached(pending_fences.front().position) &&
           rasterizer->SignalsPublished(pending_fences.front().position)) {
        pending_fences.pop_front();
    }
}

void CommandProcessor::NotePending(VAddr address, u64 value, u32 bytes, bool eop) {
    BbWriteLog::NoteIntent(address, &value, bytes,
                           eop ? BbWriteLog::FenceIntent : BbWriteLog::EosIntent);
    pending_fences.push_back({address, value, bytes, rasterizer->DrawPipeHead()});
    rasterizer->NotePendingGpuWrite(address, bytes);
}

void CommandProcessor::NotePendingEop(const EventWriteEop& eop) {
    DropReachedFences();
    eop.Signal([&](void* address, u64 value,
                   u32 bytes) { NotePending(reinterpret_cast<VAddr>(address), value, bytes, true); },
               [] {});
}

void CommandProcessor::NotePendingEos(const EventWriteEos& eos) {
    DropReachedFences();
    if (eos.Command() == EventWriteEos::SignalFence) {
        NotePending(eos.Address(), eos.Value(), sizeof(u32), false);
    }
}

void CommandProcessor::NotePendingWrite(VAddr address, const u32* data, u32 bytes) {
    if (bytes > sizeof(u64)) {
        return; // labels are small; larger writes are not waited on
    }
    u64 value = 0;
    std::memcpy(&value, data, bytes);
    pending_fences.push_back({address, value, bytes, rasterizer->DrawPipeHead()});
}

bool CommandProcessor::PendingFenceValue(VAddr address, u32& value) {
    for (auto it = pending_fences.rbegin(); it != pending_fences.rend(); ++it) {
        if (rasterizer->DrawPipeReached(it->position) && rasterizer->SignalsPublished(it->position)) {
            break; // written already, or a queued GPU signal (below)
        }
        if (address >= it->address && address + sizeof(u32) <= it->address + it->num_bytes) {
            value = u32(it->data >> ((address - it->address) * 8));
            return true;
        }
    }
    // Past the recording thread, a fence waits for the GPU to finish the work
    // before it; in stream order it is met already.
    if (rasterizer) {
        u64 data = 0;
        if (rasterizer->PendingSignalValue(address, data)) {
            value = u32(data);
            return true;
        }
        if ((address & 7) == 4 && rasterizer->PendingSignalValue(address - 4, data)) {
            value = u32(data >> 32); // the high half of a 64-bit fence
            return true;
        }
    }
    return false;
}

size_t CommandProcessor::QueueSize(u32 queue) {
    std::scoped_lock lock{queues[queue].mutex};
    return queues[queue].works.size();
}

} // namespace BbCp
