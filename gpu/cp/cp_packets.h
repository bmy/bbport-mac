// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport cp: the fields of the PM4 packets the command processor acts on, read from their body
// (the dwords after the header). Views are trivially copyable copies of the body, so a packet can
// travel to another thread as bytes (Rasterizer::RunInOrder). Layouts: the CIK command processor.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <cstring>

#include "common/assert.h"
#include "common/types.h"
#include "cp_pm4.h"

namespace BbCp {

/// A packet's body as `N` dwords, viewed as `Self`.
template <typename Self, u32 N>
struct Body {
    u32 w[N];

    static Self From(const u32* body) {
        Self b;
        std::memcpy(b.w, body, sizeof(b.w));
        return b;
    }
    /// From a body of `words` (the rest zero): packets whose forms differ in length.
    static Self From(const u32* body, u32 words) {
        Self b{};
        std::memcpy(b.w, body, std::min(words, N) * sizeof(u32));
        return b;
    }
    constexpr u32 Bits(u32 dword, u32 shift, u32 bits) const {
        return (w[dword] >> shift) & ((1u << bits) - 1);
    }
    constexpr u64 Address(u32 lo, u32 hi, u32 hi_bits = 32, u32 lo_mask = ~0u) const {
        const u64 high = hi_bits == 32 ? w[hi] : (w[hi] & ((1u << hi_bits) - 1));
        return (w[lo] & lo_mask) | high << 32;
    }
};

/// What an end-of-pipe event (and RELEASE_MEM) writes.
enum class DataSel : u32 {
    None = 0,
    Data32 = 1,
    Data64 = 2,
    GpuClock64 = 3,
    PerfCounter = 4,
    GdsStore = 5, ///< RELEASE_MEM: GDS words to memory
};

/// Whether it raises an interrupt.
enum class IntSel : u32 {
    None = 0,
    Irq = 1,
    IrqAfterWrite = 2,
    IrqUndocumented = 3, ///< RELEASE_MEM: as IrqAfterWrite
};

/// The GPU clock of end-of-pipe timestamps and COPY_DATA as the CPU gives it: the PS4's counts
/// at 100 MHz; here the host's monotonic clock in 10 ns ticks (the GPU's own timestamps are put on
/// the same time line, vk_timestamps.h).
inline u64 GpuClock64() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()) / 10;
}

/// The GPU's performance counter (its 800 MHz core clock), from the CPU's time stamp counter.
u64 GpuPerfCounter();

/// EVENT_WRITE (event type and index; occlusion and statistics events carry an address).
struct EventWrite : Body<EventWrite, 3> {
    static constexpr u32 SoVgtStreamoutFlush = 31;
    static constexpr u32 ZpassDone = 21;
    static constexpr u32 PixelPipeStatDump = 57;
    static constexpr u32 IndexZpassDone = 1;

    u32 Type() const {
        return Bits(0, 0, 6);
    }
    u32 Index() const {
        return Bits(0, 8, 4);
    }
    u64 Address() const {
        return Body::Address(1, 2);
    }
};

/// EVENT_WRITE_EOP: at the end of the pipe, write a value (or the clock) and/or interrupt.
struct EventWriteEop : Body<EventWriteEop, 5> {
    u64 Address() const {
        return Body::Address(1, 2, 16);
    }
    IntSel Int() const {
        return IntSel(Bits(2, 24, 2));
    }
    DataSel Data() const {
        return DataSel(Bits(2, 29, 3));
    }
    u32 Value32() const {
        return w[3];
    }
    u64 Value64() const {
        return w[3] | u64(w[4]) << 32;
    }
    /// The value as written (the 32-bit one widened).
    u64 Value() const {
        return Data() == DataSel::Data32 ? Value32() : Value64();
    }
    /// Writes through `write(address, value, bytes)`, then interrupts through `irq()`.
    template <typename Write, typename Irq>
    void Signal(Write&& write, Irq&& irq) const;
};

/// EVENT_WRITE_EOS: at the end of the shaders, write a value (or store GDS words).
struct EventWriteEos : Body<EventWriteEos, 4> {
    static constexpr u32 GdsStore = 1;
    static constexpr u32 SignalFence = 2;

    u64 Address() const {
        return Body::Address(1, 2, 16);
    }
    u32 Command() const {
        return Bits(2, 29, 3);
    }
    u32 Value() const {
        return w[3];
    }
    u32 GdsIndex() const {
        return Bits(3, 0, 16);
    }
    u32 GdsDwords() const {
        return Bits(3, 16, 16);
    }
};

/// RELEASE_MEM (compute rings): EOP's counterpart.
struct ReleaseMem : Body<ReleaseMem, 6> {
    IntSel Int() const {
        return IntSel(Bits(1, 24, 3));
    }
    DataSel Data() const {
        return DataSel(Bits(1, 29, 3));
    }
    u64 Address() const {
        return Body::Address(2, 3);
    }
    u32 Value32() const {
        return w[4];
    }
    u64 Value64() const {
        return w[4] | u64(w[5]) << 32;
    }
    u32 GdsIndex() const {
        return Bits(4, 0, 16);
    }
    u32 GdsDwords() const {
        return Bits(4, 16, 16);
    }
};

/// WRITE_DATA: immediate dwords to memory.
struct WriteData {
    static constexpr u32 DstMemory = 5;
    static constexpr u32 DstMemoryAsync = 2;

    static u32 DstSel(const u32* body) {
        return (body[0] >> 8) & 0xF;
    }
    static bool OneAddress(const u32* body) {
        return (body[0] >> 16) & 1;
    }
    static u64 Address(const u32* body) {
        return body[1] | u64(body[2]) << 32;
    }
    /// The data's bytes in a packet of `body_words`.
    static u32 Bytes(u32 body_words) {
        return (body_words - 3) * 4;
    }
    static const u32* Data(const u32* body) {
        return body + 3;
    }
};

/// DMA_DATA: a copy or fill between memory, GDS and immediate data.
struct DmaData : Body<DmaData, 6> {
    enum Sel : u32 { Memory = 0, Gds = 1, Immediate = 2, MemoryL2 = 3 };

    Sel Dst() const {
        return Sel(Bits(0, 20, 2));
    }
    Sel Src() const {
        return Sel(Bits(0, 29, 2));
    }
    bool DstIsMemory() const {
        return Dst() == Memory || Dst() == MemoryL2;
    }
    bool SrcIsMemory() const {
        return Src() == Memory || Src() == MemoryL2;
    }
    u32 FillValue() const {
        return w[1];
    }
    u64 SrcAddress() const {
        return Body::Address(1, 2);
    }
    u64 DstAddress() const {
        return Body::Address(3, 4);
    }
    u32 SrcLow() const {
        return w[1];
    }
    u32 DstLow() const {
        return w[3];
    }
    u32 Bytes() const {
        return Bits(5, 0, 21);
    }
    /// Destination address space: 1 = a register, which this command processor does not do.
    bool DstRegister() const {
        return Bits(5, 27, 1);
    }
};

/// WAIT_REG_MEM: wait until (value & mask) compares to the reference.
struct WaitRegMem : Body<WaitRegMem, 6> {
    bool Memory() const {
        return Bits(0, 4, 1);
    }
    bool Pfp() const {
        return Bits(0, 8, 1);
    }
    u32 Function() const {
        return Bits(0, 0, 3);
    }
    u32 Register() const {
        return Bits(1, 0, 16);
    }
    u64 Address() const {
        return Body::Address(1, 2, 32, ~3u);
    }
    u32 Reference() const {
        return w[3];
    }
    u32 Mask() const {
        return w[4];
    }
    bool TestValue(u32 value) const;
    /// Against memory or the register file.
    bool Test(const u32* registers) const {
        return TestValue(Memory() ? *reinterpret_cast<const volatile u32*>(Address())
                                  : registers[Register()]);
    }
};

/// MEM_SEMAPHORE: signal (increment or set) or wait for (and take) a 64-bit semaphore.
struct MemSemaphore : Body<MemSemaphore, 2> {
    u64* Address() const {
        return reinterpret_cast<u64*>(Body::Address(0, 1, 8, ~7u));
    }
    bool Signals() const {
        return Bits(1, 29, 3) == 6;
    }
    bool WriteOne() const {
        return Bits(1, 20, 1);
    }
    bool Signaled() const {
        return __atomic_load_n(Address(), __ATOMIC_ACQUIRE) > 0;
    }
    void Take() const {
        __atomic_fetch_sub(Address(), 1, __ATOMIC_ACQ_REL);
    }
    void Signal() const {
        if (WriteOne()) {
            __atomic_store_n(Address(), u64{1}, __ATOMIC_RELEASE);
        } else {
            __atomic_fetch_add(Address(), u64{1}, __ATOMIC_ACQ_REL);
        }
    }
};

/// COPY_DATA: one or two dwords from a register, memory, GDS, an immediate or the clock.
struct CopyData : Body<CopyData, 5> {
    enum Src : u32 {
        SrcRegister = 0,
        SrcMemory = 1,
        SrcL2 = 2,
        SrcGds = 3,
        SrcImmediate = 5,
        SrcClock = 9,
    };
    enum Dst : u32 { DstRegister = 0, DstMemory = 1, DstL2 = 2, DstGds = 3, DstMemoryAsync = 5 };

    Src Source() const {
        return Src(Bits(0, 0, 4));
    }
    Dst Destination() const {
        return Dst(Bits(0, 8, 4));
    }
    u32 Bytes() const {
        return Bits(0, 16, 1) ? 8 : 4;
    }
    u64 SrcAddress() const {
        return Body::Address(1, 2);
    }
    u64 DstAddress() const {
        return Body::Address(3, 4);
    }
};

/// SET_PREDICATION: draws and dispatches with the predicate bit run only if it holds.
struct SetPredication : Body<SetPredication, 2> {
    enum Op : u32 { Clear = 0, ZPass = 1, PrimCount = 2, Bool64 = 3, Bool32 = 4 };

    u64 Address() const {
        return Body::Address(0, 1, 8, ~3u);
    }
    bool DrawIfTrue() const {
        return Bits(1, 8, 1);
    }
    bool HintNoWait() const {
        return Bits(1, 12, 1);
    }
    Op Operation() const {
        return Op(Bits(1, 16, 3));
    }
    bool Continue() const {
        return Bits(1, 31, 1);
    }
};

/// COND_EXEC: skip the next dwords if the 32-bit value at the address is 0.
struct CondExec : Body<CondExec, 3> {
    u64 Address() const {
        return Body::Address(0, 1, 16, ~3u);
    }
    u32 Command() const {
        return Bits(1, 28, 4);
    }
    u32 Dwords() const {
        return Bits(2, 0, 14);
    }
};

/// INDIRECT_BUFFER (and _CONST): a command buffer called from this one.
struct IndirectBuffer : Body<IndirectBuffer, 3> {
    const u32* Address() const {
        return reinterpret_cast<const u32*>(Body::Address(0, 1, 16));
    }
    u32 Dwords() const {
        return Bits(2, 0, 20);
    }
};

/// SET_BASE (the indirect draw/dispatch arguments' base).
struct SetBase : Body<SetBase, 3> {
    static constexpr u32 DrawIndexIndirectPatchTable = 1;

    u32 Index() const {
        return Bits(0, 0, 4);
    }
    u64 Address() const {
        return Body::Address(1, 2, 16);
    }
};

/// The indirect draws: arguments at the base plus an offset; where the CP writes the base vertex
/// and start instance (user data registers).
struct DrawIndirect : Body<DrawIndirect, 9> {
    u32 DataOffset() const {
        return w[0];
    }
    u32 BaseVertexLoc() const {
        return Bits(1, 0, 16);
    }
    u32 StartInstanceLoc() const {
        return Bits(2, 0, 16);
    }
    // _MULTI forms.
    u32 Count() const {
        return w[3];
    }
    u32 Stride() const {
        return w[4];
    }
    // DRAW_INDEX_INDIRECT_COUNT_MULTI.
    bool CountInMemory() const {
        return Bits(3, 30, 1);
    }
    u32 MaxCount() const {
        return w[4];
    }
    u64 CountAddress() const {
        return w[5] | u64(w[6]) << 32;
    }
    u32 CountStride() const {
        return w[7];
    }
};

/// The CE's constant RAM packets.
struct ConstRam {
    /// WRITE_CONST_RAM: byte offset, dwords after it.
    static u32 WriteOffset(const u32* body) {
        return body[0] & 0xFFFF;
    }
    /// DUMP_CONST_RAM: byte offset, size, destination.
    static u32 DumpOffset(const u32* body) {
        return body[0] & 0xFFFF;
    }
    static u32 DumpBytes(const u32* body) {
        return (body[1] & 0x7FFF) * 4;
    }
    static u64 DumpAddress(const u32* body) {
        return body[2] | u64(body[3]) << 32;
    }
};

/// NOP payloads the driver and the translator agree on.
namespace NopPayload {
inline constexpr u32 MarkerPush = 0x68750001u;
inline constexpr u32 MarkerPop = 0x68750002u;
inline constexpr u32 ColorMarkerPush = 0x6875000Eu;
inline constexpr u32 PatchedFlip = 0x68750776u;
} // namespace NopPayload

/// When the last corrupt end-of-pipe packet was skipped (steady clock ms; 0: never): a wait for the
/// label it should have written gives up (WaitRegMemGraphics).
inline std::atomic<long long> g_bad_eop_at_ms{0};
inline void ReportBadEop(const char* what, u32 select) {
    g_bad_eop_at_ms.store(std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count() + 1,
                          std::memory_order_relaxed);
    static std::atomic<u32> reported{0};
    if (reported.fetch_add(1, std::memory_order_relaxed) < 16) {
        std::fprintf(stderr, "Bad PM4: end-of-pipe %s select %u skipped (corrupt packet)\n", what,
                     select);
    }
}

template <typename Write, typename Irq>
void EventWriteEop::Signal(Write&& write, Irq&& irq) const {
    auto* address = reinterpret_cast<void*>(Address());
    switch (Data()) {
    case DataSel::None:
        break;
    case DataSel::Data32:
        write(address, Value32(), 4u);
        break;
    case DataSel::Data64:
        write(address, Value64(), 8u);
        break;
    case DataSel::GpuClock64:
        write(address, GpuClock64(), 8u);
        break;
    case DataSel::PerfCounter:
        write(address, GpuPerfCounter(), 8u);
        break;
    default:
        // A corrupt packet (the game's command buffers at a character spawn on Windows, after
        // PM4 repair, 2026-10-25): no write, reported, instead of stopping the game.
        ReportBadEop("data", u32(Data()));
        return;
    }
    switch (Int()) {
    case IntSel::None:
        break;
    case IntSel::Irq:
    case IntSel::IrqAfterWrite:
        irq();
        break;
    default:
        ReportBadEop("interrupt", u32(Int()));
        break;
    }
}

} // namespace BbCp
