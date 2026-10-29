// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport cp: what packets do on the draw recording thread, in stream order with the draws
// (Rasterizer::RunInOrder; here, after a drain, when the draw pipeline is off): fences, memory
// writes and copies, occlusion results, flips.

#include <chrono>
#include "shim/bb_env.h"
#include <cstdlib>
#include <cstring>
#include <thread>

#include "bbport_copy.h"
#include "bbport_free_check.h"
#include "bbport_toggles.h"
#include "bbport_write_log.h"
#include "common/assert.h"
#include "common/logging/log.h"
#include "core/memory.h"
#include "cp/cp_packets.h"
#include "cp/cp_diagnostics.h"
#include "cp/cp_tasks.h"
#include "platform/gnm/gnm.h"
#include "platform/kernel.h"
#include "platform/interrupts.h"
#include "render/vk_draw_pipe.h"
#include "buffers/buffer_cache.h"
#include "render/rasterizer.h"

namespace BbCp {

u64 GpuPerfCounter() {
    // The TSC scaled to the GPU's 800 MHz core clock.
    return u64((static_cast<unsigned __int128>(Libraries::Kernel::sceKernelReadTsc()) *
                BbGnm::GpuCoreClock) /
               Libraries::Kernel::sceKernelGetTscFrequency());
}

bool WaitRegMem::TestValue(u32 value) const {
    const u32 masked = value & Mask();
    const u32 ref = Reference();
    switch (Function()) {
    case 0:
        return true;
    case 1:
        return masked < ref;
    case 2:
        return masked <= ref;
    case 3:
        return masked == ref;
    case 4:
        return masked != ref;
    case 5:
        return masked >= ref;
    case 6:
        return masked > ref;
    default:
        UNREACHABLE_MSG("WAIT_REG_MEM function {}", Function());
    }
}

namespace Tasks {

namespace {

/// A label straight into guest memory (no backing view): one store. A memcpy of 4 or 8 bytes is
/// two overlapping stores in glibc; a thread preempted between them wrote the label again after
/// the guest had freed its block.
void StoreLabel(void* address, u64 value, u32 num_bytes) {
    const auto at = reinterpret_cast<uintptr_t>(address);
    if (num_bytes == 8 && (at & 7) == 0) {
        __atomic_store_n(static_cast<u64*>(address), value, __ATOMIC_RELEASE);
    } else if (num_bytes == 4 && (at & 3) == 0) {
        __atomic_store_n(static_cast<u32*>(address), u32(value), __ATOMIC_RELEASE);
    } else {
        auto* bytes = static_cast<u8*>(address);
        for (u32 i = 0; i < num_bytes; ++i) {
            __atomic_store_n(bytes + i, u8(value >> (8 * i)), __ATOMIC_RELEASE);
        }
    }
}

/// A fence label: through the backing mapping when the address is mapped there, and into the
/// GPU side's VRAM copy of it too.
void WriteLabel(void* address, u64 value, u32 num_bytes, BbFreeCheck::Source source, u64 seq) {
    auto* memory = Core::Memory::Instance();
    BbFreeCheck::Check(reinterpret_cast<u64>(address), num_bytes, &value, source, seq);
    BbFreeCheck::NoteFenceWriting(reinterpret_cast<u64>(address));
    BbWriteLog::Note(reinterpret_cast<u64>(address), &value, num_bytes, BbWriteLog::Fence);
    Diagnostics::NoteLabel(Diagnostics::LabelStep::GraphicsWritten, reinterpret_cast<u64>(address),
                           value, num_bytes);
    const auto start = std::chrono::steady_clock::now();
    if (!memory->TryWriteBacking(address, &value, num_bytes)) {
        StoreLabel(address, value, num_bytes);
    }
    if (auto* rasterizer = memory->GetRasterizer()) {
        rasterizer->NoteLateCommandWrite(reinterpret_cast<VAddr>(address), &value, num_bytes);
    }
    BbFreeCheck::NoteFenceWritten(
        reinterpret_cast<u64>(address),
        u64(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                                 start)
                .count()));
}

/// BB_FENCE_DELAY_TEST=1 (an experiment): every 200th fence comes 80 ms late; if the guest
/// frees label objects after a timeout, late labels corrupt its heap.
void MaybeDelay(u64 seq) {
    static const bool delay = bb_env_on("BB_FENCE_DELAY_TEST");
    if (delay && seq != 0 && seq % 200 == 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
    }
}

} // namespace

void SignalEop(const EventWriteEop& eop, u64 seq) {
    eop.Signal(
        [seq](void* address, u64 value, u32 num_bytes) {
            MaybeDelay(seq);
            WriteLabel(address, value, num_bytes, BbFreeCheck::Eop, seq);
            BbStats::eop_written.fetch_add(1, std::memory_order_relaxed);
        },
        [] { BbIrq::Raise(BbIrq::Line::GfxEop); });
}

void EventWriteEopTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    EventWriteEop eop;
    std::memcpy(&eop, data, sizeof(eop));
    if (eop.Data() != DataSel::None) {
        BbStats::eop_decoded.fetch_add(1, std::memory_order_relaxed);
    }
    const u64 seq = BbFreeCheck::NextFenceSeq(); // their order in the stream (BB_FREE_CHECK)
    rasterizer.ProcessDownloadImages();
    // Written once the GPU has finished the work before it, as the hardware does: a timestamp
    // or a label without an interrupt by the GPU itself; the rest after the GPU signals.
    if (eop.Data() == DataSel::GpuClock64 && eop.Int() == IntSel::None &&
        rasterizer.WriteTimestampOnGpu(eop.Address(), true)) {
        BbStats::eop_written.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if ((eop.Data() == DataSel::Data32 || eop.Data() == DataSel::Data64) &&
        eop.Int() == IntSel::None &&
        rasterizer.WriteLabelOnGpu(eop.Address(), eop.Value(),
                                   eop.Data() == DataSel::Data32 ? 4 : 8)) {
        Diagnostics::NoteLabel(Diagnostics::LabelStep::GraphicsOnGpu, eop.Address(),
                               eop.Value(), eop.Data() == DataSel::Data32 ? 4 : 8);
        BbStats::eop_written.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const bool writes = eop.Data() != DataSel::None;
    rasterizer.SignalAfterGpu([eop, seq] { SignalEop(eop, seq); }, writes ? eop.Address() : 0,
                              eop.Value());
}

namespace {
void WriteEosLabel(void* address, u64 value, u32 num_bytes) {
    WriteLabel(address, value, num_bytes, BbFreeCheck::Eos, 0);
}
} // namespace

void EventWriteEosTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    EventWriteEos eos;
    std::memcpy(&eos, data, sizeof(eos));
    auto* address = reinterpret_cast<void*>(eos.Address());
    if (eos.Command() == EventWriteEos::SignalFence) {
        // End of the shader work: written once the GPU has finished it.
        rasterizer.ProcessDownloadImages();
        if (rasterizer.WriteLabelOnGpu(eos.Address(), eos.Value(), sizeof(u32))) {
            Diagnostics::NoteLabel(Diagnostics::LabelStep::GraphicsOnGpu, eos.Address(),
                                   eos.Value(), sizeof(u32));
            return;
        }
        rasterizer.SignalAfterGpu([address, value = eos.Value()] {
            WriteEosLabel(address, value, sizeof(u32));
        },
                                  eos.Address(), eos.Value());
        return;
    }
    rasterizer.WaitHostCopies();
    rasterizer.ProcessDownloadImages();
    switch (eos.Command()) {
    case EventWriteEos::SignalFence:
        WriteEosLabel(address, eos.Value(), sizeof(u32));
        break;
    case EventWriteEos::GdsStore: {
        ASSERT(eos.GdsDwords() == 1);
        // A GPU copy from GDS in stream order, as the CP stores it.
        rasterizer.CopyBuffer(eos.Address(), eos.GdsIndex() * sizeof(u32), sizeof(u32), false,
                              true);
        break;
    }
    default:
        UNREACHABLE_MSG("end-of-shader command {}", eos.Command());
    }
}

void MemSemaphoreSignalTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    MemSemaphore semaphore;
    std::memcpy(&semaphore, data, sizeof(semaphore));
    rasterizer.SignalAfterGpu([semaphore] { semaphore.Signal(); });
}

void DmaDataTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    DmaData dma;
    std::memcpy(&dma, data, sizeof(dma));
    ASSERT(!dma.DstRegister());
    // A DMA into guest memory written by this thread (the GPU has not modified the range) must
    // not overtake the copies and fences deferred to the recording thread: the guest, seeing it,
    // frees objects whose labels an earlier fence still writes. Recorded as a GPU fill or copy it
    // lands after the submission goes out, after those: no wait.
    if (dma.DstIsMemory() && rasterizer.DmaMayWriteOnCpu(dma.DstAddress(), dma.Bytes())) {
        rasterizer.WaitHostCopiesFor(dma.DstAddress(), dma.Bytes());
    }
    if (dma.Src() == DmaData::Immediate && dma.Dst() == DmaData::Gds) {
        rasterizer.FillBuffer(dma.DstLow(), dma.Bytes(), dma.FillValue(), true);
    } else if (dma.SrcIsMemory() && dma.Dst() == DmaData::Gds) {
        rasterizer.CopyBuffer(dma.DstLow(), dma.SrcAddress(), dma.Bytes(), true, false);
    } else if (dma.Src() == DmaData::Immediate && dma.DstIsMemory()) {
        rasterizer.FillBuffer(dma.DstAddress(), dma.Bytes(), dma.FillValue(), false);
    } else if (dma.Src() == DmaData::Gds && dma.DstIsMemory()) {
        rasterizer.CopyBuffer(dma.DstAddress(), dma.SrcLow(), dma.Bytes(), false, true);
    } else if (dma.SrcIsMemory() && dma.DstIsMemory()) {
        rasterizer.CopyBuffer(dma.DstAddress(), dma.SrcAddress(), dma.Bytes(), false, false);
    } else {
        UNREACHABLE_MSG("DMA_DATA from {} to {}", u32(dma.Src()), u32(dma.Dst()));
    }
}

void ZpassDoneTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    ZpassEvent event;
    std::memcpy(&event, data, sizeof(event));
    rasterizer.OcclusionEvent(event.address, event.pairs);
}

void CopyDataTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    CopyTask task;
    std::memcpy(&task, data, sizeof(task));
    if (!task.dst_gds && rasterizer.DmaMayWriteOnCpu(task.dst, task.num_bytes)) {
        rasterizer.WaitHostCopiesFor(task.dst, task.num_bytes);
    }
    if (task.fill) {
        for (u32 at = 0; at < task.num_bytes; at += sizeof(u32)) {
            rasterizer.FillBuffer(task.dst + at, sizeof(u32), u32(task.value >> (at * 8)),
                                  task.dst_gds);
        }
        return;
    }
    rasterizer.CopyBuffer(task.dst, task.src, task.num_bytes, task.dst_gds, task.src_gds);
}

void FlipTask(Vulkan::Rasterizer& rasterizer, const u8*) {
    // BB_EXP_FLIP_FINISH=1 (experiment): the flip is signalled once the GPU has run the frame.
    static const bool finish = bb_env("BB_EXP_FLIP_FINISH") != nullptr;
    if (finish) {
        rasterizer.Finish();
    }
    // BB_EXP_FLIP_LAG=1 (experiment): the GPU has finished the frame before this one (one frame in
    // flight at most).
    static const bool lag = bb_env("BB_EXP_FLIP_LAG") != nullptr;
    if (lag) {
        rasterizer.WaitPreviousFrame();
    }
    BbIrq::Raise(BbIrq::Line::GfxFlip);
}

void GpuClockCopyTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    VAddr target;
    std::memcpy(&target, data, sizeof(target));
    // The GPU's own clock when it gets there (vk_timestamps.h); else the CPU's, now.
    if (rasterizer.WriteTimestampOnGpu(target, false)) {
        return;
    }
    const u64 value = GpuClock64();
    if (!rasterizer.WriteDataOnGpu(target, &value, sizeof(value))) {
        std::memcpy(reinterpret_cast<void*>(target), &value, sizeof(value));
    }
}

void WriteDataTask(Vulkan::Rasterizer& rasterizer, const u8* data) {
    const auto* packet = reinterpret_cast<const u32*>(data);
    const u32 body_words = Header{packet[0]}.BodyWords();
    const u32* body = packet + 1;
    const VAddr address = WriteData::Address(body);
    const u32 bytes = WriteData::Bytes(body_words);
    const u32* values = WriteData::Data(body);
    // The GPU writes it in stream order, as the command processor does. A CPU
    // store now lands before the GPU has run the work before it, which may still read the old
    // value (the game orders such writes after its own fences, taken as met in stream order).
    if (rasterizer.WriteDataOnGpu(address, values, bytes)) {
        return;
    }
    // Copies deferred to the recording thread that read the range precede the write; so do the
    // fences deferred to it (the guest sees this write).
    rasterizer.WaitHostCopiesFor(address, bytes);
    BbFreeCheck::Check(address, bytes, values, BbFreeCheck::WriteData);
    BbWriteLog::Note(address, values, bytes, BbWriteLog::WriteData);
    std::memcpy(reinterpret_cast<void*>(address), values, bytes);
    rasterizer.NoteCommandWrite(address, values, bytes, true);
}

} // namespace Tasks

} // namespace BbCp
