// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport cp: compute queues. A submission is the part of a ring the game wrote since the last
// one (sceGnmDingDong); the read pointer moves as packets are done. A packet the ring's end cut in
// two waits in the queue for the rest, which the next submission brings.

#include <algorithm>
#include "shim/bb_env.h"
#include <cstring>

#include <fmt/format.h>

#include "bbport_free_check.h"
#include "bbport_toggles.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "cp/command_processor.h"
#include "cp/cp_diagnostics.h"
#include "cp/cp_packets.h"
#include "cp/cp_tasks.h"
#include "platform/interrupts.h"
#include "render/vk_draw_pipe.h"
#include "render/rasterizer.h"

namespace BbCp {

namespace {

constexpr u32 ComputeShOffset = 0x200;
constexpr u32 ComputeWords = sizeof(ComputeProgram) / 4;
constexpr u32 DispatchDimsBytes = 3 * sizeof(u32);

bool MemSemaphoreInOrder() {
    static const bool on = !bb_env_off("BB_MEM_SEMAPHORE_ORDER");
    return on;
}

/// RELEASE_MEM's write and interrupt (`rasterizer` copies GDS words; null on the GPU's signal).
void WriteRelease(const ReleaseMem& release, const auto& irq, Vulkan::Rasterizer* rasterizer) {
    const VAddr address = release.Address();
    switch (release.Data()) {
    case DataSel::None:
        break;
    case DataSel::Data32:
        *reinterpret_cast<u32*>(address) = release.Value32();
        Diagnostics::NoteLabel(Diagnostics::LabelStep::ComputeWritten, address, release.Value32(),
                               4);
        break;
    case DataSel::Data64:
        *reinterpret_cast<u64*>(address) = release.Value64();
        Diagnostics::NoteLabel(Diagnostics::LabelStep::ComputeWritten, address, release.Value64(),
                               8);
        break;
    case DataSel::GpuClock64:
        *reinterpret_cast<u64*>(address) = GpuClock64();
        break;
    case DataSel::PerfCounter:
        *reinterpret_cast<u64*>(address) = GpuPerfCounter();
        break;
    case DataSel::GdsStore:
        ASSERT_MSG(rasterizer, "RELEASE_MEM of GDS words without the renderer");
        rasterizer->CopyBuffer(address, release.GdsIndex(), release.GdsDwords() * sizeof(u32),
                               false, true);
        break;
    }
    if (release.Int() != IntSel::None) {
        irq();
    }
}

/// A RELEASE_MEM handed to the draw recording thread: its body and the queue's pipe.
struct ReleaseTask {
    ReleaseMem release;
    u32 pipe_id;
};

/// The release in stream order on the recording thread, as graphics
/// end-of-pipe events: a timestamp or a label without an interrupt written by the GPU itself, the
/// rest once the GPU has finished the work before it.
void ReleaseMemTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    ReleaseTask task;
    std::memcpy(&task, data, sizeof(task));
    const auto& release = task.release;
    const u32 pipe_id = task.pipe_id;
    const auto irq = [pipe_id] { BbIrq::Raise(static_cast<BbIrq::Line>(pipe_id)); };
    const bool interrupts = release.Int() != IntSel::None;
    rasterizer.ProcessDownloadImages();
    if (release.Data() == DataSel::GpuClock64 && !interrupts &&
        rasterizer.WriteTimestampOnGpu(release.Address(), true)) {
        return;
    }
    if ((release.Data() == DataSel::Data32 || release.Data() == DataSel::Data64) && !interrupts &&
        rasterizer.WriteLabelOnGpu(release.Address(),
                                   release.Data() == DataSel::Data32 ? release.Value32()
                                                                     : release.Value64(),
                                   release.Data() == DataSel::Data32 ? 4 : 8)) {
        Diagnostics::NoteLabel(Diagnostics::LabelStep::ComputeOnGpu, release.Address(),
                               release.Value64(), release.Data() == DataSel::Data32 ? 4 : 8);
        return;
    }
    const bool writes = release.Data() != DataSel::None;
    rasterizer.SignalAfterGpu([release, irq] { WriteRelease(release, irq, nullptr); },
                              writes ? release.Address() : 0, release.Value64());
}

/// Packets that run in stream order without the draw recording thread being idle: they only
/// change this thread's state, are handed to that thread (dispatches, writes, fences), or wait
/// for it themselves. The others use the caches here and wait for it to finish first.
bool Pipelined(Op op, const u32* body) {
    switch (op) {
    case Op::ReleaseMem:
        return ReleaseMem::From(body).Data() != DataSel::GdsStore;
    case Op::Nop:
    case Op::AcquireMem:
    case Op::EventWrite:
    case Op::IndirectBuffer:
    case Op::SetShReg:
    case Op::SetQueueReg:
    case Op::Rewind:
    case Op::DispatchDirect:
    case Op::DispatchIndirect:
    case Op::WriteData:
    case Op::DmaData:
    case Op::MemSemaphore:
    case Op::WaitRegMem:
        return true;
    default:
        return false;
    }
}

} // namespace

CommandProcessor::Status CommandProcessor::RunCompute(Work& work) {
    auto& ring = asc_queues[Common::SlotId{work.queue - 1}];
    Walk& walk = work.walk;
    if (!work.started) {
        work.started = true;
        walk.Push(work.dcb, 0);
    }
    const auto advance_ring = [&](u32 dwords) {
        *ring.read_addr = (*ring.read_addr + dwords) % ring.ring_size_dw;
    };
    while (!walk.Done()) {
        RunCommands();
        Level& level = walk.levels[walk.top];
        if (level.at >= level.end) {
            --walk.top;
            continue;
        }
        const bool on_ring = walk.top == 0;
        const u32* packet = level.at;
        u32 used = 0; ///< dwords of the level the packet takes
        if (on_ring && work.assembled_used != 0) {
            packet = work.assembled.data(); // the split packet, completed earlier (a retry)
            used = work.assembled_used;
        } else if (on_ring && ring.tmp_dwords > 0) {
            // The rest of a packet the last submission cut at the ring's end.
            const u32 words = Header{ring.tmp_packet[0]}.BodyWords() + 1;
            used = words - ring.tmp_dwords;
            std::memcpy(ring.tmp_packet.data() + ring.tmp_dwords, level.at, used * sizeof(u32));
            ring.tmp_dwords = 0;
            work.assembled.assign(ring.tmp_packet.begin(), ring.tmp_packet.begin() + words);
            work.assembled_used = used;
            packet = work.assembled.data();
        } else {
            const u32 header = *level.at;
            used = (header >> 30) == 3 ? Header{header}.BodyWords() + 1 : 1;
            if (on_ring && used > u32(level.end - level.at)) {
                // Cut by the ring's end: kept until the rest comes.
                const u32 have = u32(level.end - level.at);
                ASSERT_MSG(have <= ring.tmp_packet.size(), "split compute packet too large");
                std::memcpy(ring.tmp_packet.data(), level.at, have * sizeof(u32));
                ring.tmp_dwords = have;
                advance_ring(have);
                walk.top = -1;
                break;
            }
        }
        const u32 header = packet[0];
        if ((header >> 30) == 2) { // filler
            level.at += 1;
            if (on_ring) {
                advance_ring(1);
            }
            continue;
        }
        if ((header >> 30) != 3) {
            UNREACHABLE_MSG("compute: PM4 type {} packet ({:#x})", header >> 30, header);
        }
        const u32 body_words = Header{header}.BodyWords();
        switch (Compute(work, level, packet, body_words)) {
        case Step::Wait:
            return Status::Wait;
        case Step::Moved: // an indirect buffer pushed (the level advanced)
            if (on_ring) {
                advance_ring(used);
            }
            work.assembled_used = 0;
            continue;
        case Step::Next:
            break;
        }
        level.at += used;
        if (on_ring) {
            advance_ring(used);
        }
        work.assembled_used = 0;
    }
    return Status::Done;
}

CommandProcessor::Step CommandProcessor::Compute(Work& work, Level& level, const u32* packet,
                                                 u32 body_words) {
    const u32* body = packet + 1;
    const Op op = Header{packet[0]}.Opcode();
    auto& ring = asc_queues[Common::SlotId{work.queue - 1}];
    // The graphics queue's draws go on being recorded while compute packets are decoded (a game
    // with compute queues interleaves them every frame); only the rest waits for them.
    if (rasterizer && !Pipelined(op, body)) {
        rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCompute);
    }
    switch (op) {
    case Op::Nop:
    case Op::AcquireMem:
    case Op::EventWrite:
        break;
    case Op::IndirectBuffer: {
        const auto ib = IndirectBuffer::From(body);
        level.at += 1 + body_words;
        work.walk.Push({ib.Address(), ib.Dwords()}, level.depth + 1);
        return Step::Moved;
    }
    case Op::DmaData: {
        const auto dma = DmaData::From(body);
        if (dma.DstLow() == 0x3022C || !rasterizer) {
            break;
        }
        ASSERT(!dma.DstRegister());
        // Dispatch dimensions copied into a DISPATCH_DIRECT of this very buffer: that dispatch
        // becomes indirect (its arguments may still be written by the GPU).
        if (dma.SrcIsMemory() && dma.DstIsMemory()) {
            const VAddr dst = dma.DstAddress();
            const auto* target = reinterpret_cast<const u32*>(dst) - 1;
            if (dst >= VAddr(level.begin) && dst < VAddr(level.end) &&
                dma.Bytes() == DispatchDimsBytes && (target[0] >> 30) == 3 &&
                Header{target[0]}.Opcode() == Op::DispatchDirect) {
                work.patches.push_back({target, dma.SrcAddress()});
                break;
            }
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
    case Op::Rewind:
        if (rasterizer && !(*reinterpret_cast<const volatile u32*>(body) >> 31)) {
            return Step::Wait;
        }
        break;
    case Op::SetShReg: {
        const u32 offset = body[0] & 0xFFFF;
        const u32 bytes = (body_words - 1) * sizeof(u32);
        if (offset >= ComputeShOffset && offset <= ComputeShOffset + ComputeWords) {
            ASSERT(bytes <= sizeof(ComputeProgram));
            auto* state = reinterpret_cast<u32*>(&queues[work.queue].cs_state);
            std::memcpy(state + (offset - ComputeShOffset), body + 1, bytes);
        } else {
            std::memcpy(&regs.reg_array[Bank::Sh + offset], body + 1, bytes);
            pipe_dirty.Mark(Bank::Sh + offset, bytes / sizeof(u32)); // the recording thread's copy
            gfx_touched.Mark(Bank::Sh + offset, bytes / sizeof(u32));
            // It interleaves with the graphics stream at an unpredictable point, so the draw
            // preparation workers cannot reproduce it: their checksums must differ.
            gfx_reg_checksum = HashRegisterPacket(gfx_reg_checksum ^ 0xA5C0A5C0A5C0ull, packet,
                                                  body_words + 1);
        }
        break;
    }
    case Op::SetQueueReg:
        LOG_WARNING(Render, "compute SET_QUEUE_REG ignored (queue {}, register {:#x})",
                    (body[0] >> 26) & 0x3F, body[0] & 0xFFFF);
        break;
    case Op::DispatchDirect: {
        if (const auto it = std::ranges::find(work.patches, packet, &Work::IndirectPatch::packet);
            it != work.patches.end()) {
            rasterizer->DispatchIndirect(it->args, 0, DispatchDimsBytes);
            break;
        }
        auto& cs = GetCsRegs();
        cs.dim_x = body[0];
        cs.dim_y = body[1];
        cs.dim_z = body[2];
        cs.dispatch_initiator = body[3];
        if (rasterizer && (cs.dispatch_initiator & 1)) {
            const void* at = packet;
            const u32 vqid = work.queue - 1;
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchDirect", fmt::make_format_args(vqid, at),
                                    [&] { rasterizer->DispatchDirect(); });
        }
        break;
    }
    case Op::DispatchIndirect: {
        // The compute rings' form: the arguments' address in the packet.
        const VAddr args = body[0] | u64(body[1] & 0xFFFF) << 32;
        if (rasterizer && (GetCsRegs().dispatch_initiator & 1)) {
            const void* at = packet;
            const u32 vqid = work.queue - 1;
            rasterizer->ScopeMarker("asc[{}]:{}:DispatchIndirect",
                                    fmt::make_format_args(vqid, at),
                                    [&] { rasterizer->DispatchIndirect(args, 0, DispatchDimsBytes); });
        }
        break;
    }
    case Op::WriteData: {
        const u32 dst = WriteData::DstSel(body);
        ASSERT(dst == WriteData::DstMemoryAsync || dst == WriteData::DstMemory);
        ASSERT_MSG(!WriteData::OneAddress(body), "compute WRITE_DATA to one address");
        const VAddr address = WriteData::Address(body);
        const u32 bytes = WriteData::Bytes(body_words);
        const u32* values = WriteData::Data(body);
        if (!rasterizer) {
            BbFreeCheck::Check(address, bytes, values, BbFreeCheck::ComputeWriteData);
            std::memcpy(reinterpret_cast<void*>(address), values, bytes);
            break;
        }
        // In order with the draws and dispatches (on the recording thread when in use).
        if (rasterizer->RunInOrder(&Tasks::WriteDataTask, packet, (body_words + 1) * sizeof(u32),
                                   BbToggle::PipelinedMemoryWrites, false)) {
            NotePendingWrite(address, values, bytes);
            rasterizer->NotePendingGpuWrite(address, bytes);
        }
        break;
    }
    case Op::MemSemaphore: {
        const auto semaphore = MemSemaphore::From(body);
        if (semaphore.Signals()) {
            // In stream order: after the dispatches recorded before it.
            if (rasterizer && MemSemaphoreInOrder()) {
                rasterizer->RunInOrder(&Tasks::MemSemaphoreSignalTask, &semaphore,
                                       sizeof(semaphore), BbToggle::PipelinedTasks, false);
            } else {
                semaphore.Signal();
            }
            break;
        }
        while (!semaphore.Signaled()) {
            // A signal still on the recording thread (or waiting for unsubmitted work).
            if (rasterizer && !rasterizer->DrawPipeIdle()) {
                rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCompute);
                continue;
            }
            if (rasterizer && rasterizer->HasUnsubmittedSignals()) {
                rasterizer->Flush();
            }
            return Step::Wait;
        }
        semaphore.Take();
        break;
    }
    case Op::WaitRegMem: {
        const auto wait = WaitRegMem::From(body);
        ASSERT_MSG(!wait.Pfp(), "compute WAIT_REG_MEM on the PFP");
        // A fence handed to the recording thread counts as written (stream order).
        if (u32 value; wait.Memory() && !BbToggle::Disabled(BbToggle::PendingFenceWaits) &&
                       PendingFenceValue(wait.Address(), value) && wait.TestValue(value)) {
            break;
        }
        while (!wait.Test(regs.reg_array.data())) {
            if (rasterizer && !rasterizer->DrawPipeIdle()) {
                rasterizer->DrainDrawPipe(Vulkan::DrawPipe::ReasonCompute);
                continue;
            }
            // The value may come from a fence waiting for work not submitted.
            if (rasterizer && rasterizer->HasUnsubmittedSignals()) {
                rasterizer->Flush();
            }
            return Step::Wait;
        }
        break;
    }
    case Op::ReleaseMem: {
        const auto release = ReleaseMem::From(body);
        if (release.Data() == DataSel::Data32 || release.Data() == DataSel::Data64) {
            Diagnostics::NoteLabel(Diagnostics::LabelStep::ComputeDecoded, release.Address(),
                                   release.Value64(), release.Data() == DataSel::Data32 ? 4 : 8);
        }
        if (rasterizer && Pipelined(op, body)) {
            const ReleaseTask task{release, ring.pipe_id};
            if (rasterizer->RunInOrder(&ReleaseMemTask, &task, sizeof(task),
                                       BbToggle::PipelinedTasks, false) &&
                (release.Data() == DataSel::Data32 || release.Data() == DataSel::Data64)) {
                DropReachedFences();
                NotePending(release.Address(), release.Value64(),
                            release.Data() == DataSel::Data32 ? 4 : 8, true);
            }
            // The game polls these from the CPU (another game: its main thread spun 56% of the time
            // on a compute context's counter). The work before it goes to the GPU now instead of
            // with the rest of the frame (BB_COMPUTE_RELEASE_FLUSH=0: as before).
            static const bool flush = !bb_env_off("BB_COMPUTE_RELEASE_FLUSH");
            if (flush && release.Data() != DataSel::None && rasterizer->PipeSignals()) {
                rasterizer->RequestSignalFlush();
            }
            break;
        }
        ReleaseMemory(release, ring.pipe_id);
        break;
    }
    default:
        UNREACHABLE_MSG("compute: PM4 opcode {:#x} ({} dwords) not handled", u32(op), body_words);
    }
    return Step::Next;
}

void CommandProcessor::ReleaseMemory(const ReleaseMem& release, u32 pipe_id) {
    const auto irq = [pipe_id] { BbIrq::Raise(static_cast<BbIrq::Line>(pipe_id)); };
    const bool interrupts = release.Int() != IntSel::None;
    // Written once the GPU has finished the work before it (a GDS store needs
    // the GPU copy recorded here instead).
    if (rasterizer && release.Data() != DataSel::GdsStore) {
        rasterizer->ProcessDownloadImages();
        if (release.Data() == DataSel::GpuClock64 && !interrupts &&
            rasterizer->WriteTimestampOnGpu(release.Address(), true)) {
            return;
        }
        // A label without an interrupt is written by the GPU (end of pipe).
        if ((release.Data() == DataSel::Data32 || release.Data() == DataSel::Data64) &&
            !interrupts &&
            rasterizer->WriteLabelOnGpu(release.Address(),
                                        release.Data() == DataSel::Data32 ? release.Value32()
                                                                          : release.Value64(),
                                        release.Data() == DataSel::Data32 ? 4 : 8)) {
            Diagnostics::NoteLabel(Diagnostics::LabelStep::ComputeOnGpu, release.Address(),
                                   release.Value64(), release.Data() == DataSel::Data32 ? 4 : 8);
            return;
        }
        const bool writes = release.Data() != DataSel::None;
        rasterizer->SignalAfterGpu([release, irq] { WriteRelease(release, irq, nullptr); },
                                   writes ? release.Address() : 0, release.Value64());
        return;
    }
    // Copies and graphics fences deferred to the recording thread precede writes the guest sees.
    if (rasterizer) {
        rasterizer->WaitHostCopies();
    }
    if (release.Data() != DataSel::None && release.Data() != DataSel::GdsStore) {
        const u64 value = release.Value64();
        BbFreeCheck::Check(release.Address(), release.Data() == DataSel::Data32 ? 4 : 8, &value,
                           BbFreeCheck::ReleaseMem);
    }
    if (rasterizer) {
        rasterizer->ProcessDownloadImages();
    }
    WriteRelease(release, irq, rasterizer);
}


} // namespace BbCp
