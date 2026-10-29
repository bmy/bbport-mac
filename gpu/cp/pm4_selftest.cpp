// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "cp/pm4_selftest.h"
#include "shim/bb_env.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cp/cp_packets.h"

extern "C" std::uintptr_t runtime_memory_resolve(const char* name);

namespace BbCp::Pm4SelfTest {

namespace {

using Clock = std::chrono::steady_clock;
constexpr u64 ScratchSize = 64 * 1024;
constexpr u64 Valid = 0x8000000000000000ULL;
constexpr u32 Pairs = 8; // depth blocks (Liverpool::num_counter_pairs on a base PS4)

// Scratch layout (offsets).
constexpr u64 OccVisible = 0x000, OccEmpty = 0x100;
constexpr u64 CopyImm32 = 0x200, CopyImm64 = 0x208, CopyMem = 0x210, CopyClock = 0x218;
constexpr u64 CondFalse = 0x300, CondSkipped = 0x304, CondTrue = 0x308, CondRun = 0x30c;
constexpr u64 SemLabel = 0x380, SemOrdered = 0x388, SemWaited = 0x390, SemAfterWait = 0x398;
constexpr u32 SemLabelValue = 0x5e5e, SemAfterValue = 0xa11;
constexpr u64 ClockEop1 = 0x3a0, ClockEop2 = 0x3a8;
// Compute queue: its writes, an end-of-pipe label, a MEM_SEMAPHORE and a WAIT_REG_MEM that the
// graphics queue releases a step later. Its ring and read pointer live in the scratch too.
constexpr u64 CmpWrite = 0x400, CmpLabel = 0x408, CmpSem = 0x410, CmpAfterSem = 0x418;
constexpr u64 CmpGate = 0x420, CmpAfterGate = 0x428, CmpReadPtr = 0x7f00, CmpRing = 0x8000;
constexpr u32 CmpRingDwords = 0x1000, CmpGateValue = 7;
constexpr u32 CmpWriteValue = 0xc0ffee, CmpLabelValue = 0x1abe1, CmpSemValue = 0x5e3,
              CmpGateAfterValue = 0x6a7e;

struct Builder {
    std::vector<u32> words;

    void Header(Op opcode, u32 words_after, bool predicated = false) {
        words.push_back(3u << 30 | (words_after - 1) << 16 | u32(opcode) << 8 |
                        (predicated ? 1u : 0u));
    }
    void Address(VAddr address) {
        words.push_back(u32(address));
        words.push_back(u32(address >> 32));
    }
    void Zpass(VAddr address) {
        Header(Op::EventWrite, 3);
        words.push_back(EventWrite::PixelPipeStatDump | EventWrite::IndexZpassDone << 8);
        Address(address);
    }
    void WriteData(VAddr address, u32 value) {
        Header(Op::WriteData, 4);
        words.push_back(5u << 8); // memory
        Address(address);
        words.push_back(value);
    }
    void CopyData(BbCp::CopyData::Src src, BbCp::CopyData::Dst dst, bool bits64, u64 source, VAddr target) {
        Header(Op::CopyData, 5);
        words.push_back(u32(src) | u32(dst) << 8 | (bits64 ? 1u << 16 : 0u));
        words.push_back(u32(source));
        words.push_back(u32(source >> 32));
        Address(target);
    }
    void CondExec(VAddr address, u32 skip_dwords) {
        Header(Op::CondExec, 3);
        words.push_back(u32(address) & ~3u);
        words.push_back(u32(address >> 32) & 0xffff);
        words.push_back(skip_dwords);
    }
    void SetPredication(BbCp::SetPredication::Op op, VAddr address, bool draw_if_true) {
        Header(Op::SetPredication, 2);
        words.push_back(u32(address));
        words.push_back((u32(address >> 32) & 0xff) | (draw_if_true ? 1u << 8 : 0u) |
                        u32(op) << 16);
    }
    void PredicatedEmptyDraw() {
        Header(Op::DrawIndexAuto, 2, true);
        words.push_back(0); // no vertices
        words.push_back(2); // auto index
    }
    // An end-of-pipe label (32-bit, or the GPU clock; no interrupt): written once the work before
    // it is done.
    void EopLabel(VAddr address, u32 value, DataSel data = DataSel::Data32) {
        constexpr u32 CacheFlushAndInvTsEvent = 20, EopIndex = 5;
        Header(Op::EventWriteEop, 5);
        words.push_back(CacheFlushAndInvTsEvent | EopIndex << 8);
        words.push_back(u32(address));
        words.push_back((u32(address >> 32) & 0xffff) | u32(data) << 29);
        words.push_back(value);
        words.push_back(0);
    }
    // RELEASE_MEM (compute queues' end-of-pipe event): a 32-bit label, no interrupt.
    void ReleaseLabel(VAddr address, u32 value) {
        constexpr u32 CsDoneEvent = 47, EopIndex = 6;
        Header(Op::ReleaseMem, 6);
        words.push_back(CsDoneEvent | EopIndex << 8);
        words.push_back(u32(DataSel::Data32) << 29);
        words.push_back(u32(address));
        words.push_back(u32(address >> 32));
        words.push_back(value);
        words.push_back(0);
    }
    // WAIT_REG_MEM on memory, the micro engine: until (value & mask) == reference.
    void WaitMemEqual(VAddr address, u32 reference) {
        Header(Op::WaitRegMem, 6);
        words.push_back(3u | 1u << 4); // equal, memory
        words.push_back(u32(address) & ~3u);
        words.push_back(u32(address >> 32));
        words.push_back(reference);
        words.push_back(~0u);
        words.push_back(10);
    }
    // MEM_SEMAPHORE (command processor client, increment): signal (6), or wait and decrement (7).
    void MemSemaphore(VAddr address, bool signal) {
        Header(Op::MemSemaphore, 2);
        words.push_back(u32(address) & ~7u);
        words.push_back((u32(address >> 32) & 0xff) | (signal ? 6u : 7u) << 29);
    }
};

struct State {
    int step = 0;
    Clock::time_point first{}, last_step{};
    VAddr scratch = 0;
    Builder before, after, compute;
    u64 skips_before = 0, skips = 0;
    u64 events_before = 0;
    bool failed_setup = false;
    // MEM_SEMAPHORE order: the label seen when the signal was first seen (probed per submission).
    bool sem_probe = false, sem_seen = false;
    u32 sem_label = 0;
    u64 clock_cpu1 = 0, clock_cpu2 = 0; // the CPU's GPU clock when the timestamps were decoded
};
State state;

VAddr AllocateScratch() {
    using Allocate = int32_t PS4_SYSV_ABI (*)(int64_t, int64_t, u64, u64, int, int64_t*);
    using Map = int32_t PS4_SYSV_ABI (*)(void**, u64, int, int, int64_t, u64);
    const auto allocate = reinterpret_cast<Allocate>(runtime_memory_resolve("rTXw65xmLIA#p#J"));
    const auto map = reinterpret_cast<Map>(runtime_memory_resolve("L-Q3LEjIbgA#p#J"));
    if (!allocate || !map) {
        return 0;
    }
    int64_t phys = 0;
    // Type 3 (GPU memory), CPU and GPU read/write.
    if (allocate(0, 0x7fffffffffffll, ScratchSize, ScratchSize, 3, &phys) != 0) {
        return 0;
    }
    void* address = nullptr;
    if (map(&address, ScratchSize, 0x33, 0, phys, ScratchSize) != 0) {
        return 0;
    }
    return reinterpret_cast<VAddr>(address);
}

template <typename T>
T Read(VAddr address) {
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(T));
    return value;
}

void Report(const Hooks& hooks) {
    const VAddr s = state.scratch;
    hooks.sync_for_cpu_read(s, 0x500);
    int failures = 0;
    const auto check = [&](const char* name, bool ok, const std::string& detail) {
        std::printf("PM4 self-test: %-34s %s (%s)\n", name, ok ? "PASS" : "FAIL", detail.c_str());
        failures += ok ? 0 : 1;
    };
    // Occlusion: draws across submissions/render-pass cuts, and nothing between two events.
    const auto samples = [&](VAddr results, bool& valid) {
        u64 total = 0;
        valid = true;
        for (u32 i = 0; i < Pairs; ++i) {
            const u64 begin = Read<u64>(results + i * 16), end = Read<u64>(results + i * 16 + 8);
            valid = valid && (begin & Valid) && (end & Valid);
            total += (end & ~Valid) - (begin & ~Valid);
        }
        return total;
    };
    bool valid_visible = false, valid_empty = false;
    const u64 visible = samples(s + OccVisible, valid_visible);
    const u64 empty = samples(s + OccEmpty, valid_empty);
    const u64 events = hooks.occlusion_events() - state.events_before;
    check("occlusion: samples across submissions", valid_visible && visible > 0,
          std::to_string(visible) + " samples, valid " + std::to_string(valid_visible) + ", " +
              std::to_string(events) + " events translated");
    check("occlusion: nothing between two events", valid_empty && empty == 0,
          std::to_string(empty) + " samples, valid " + std::to_string(valid_empty));
    char detail[128];
    std::snprintf(detail, sizeof(detail), "%#x", Read<u32>(s + CopyImm32));
    check("COPY_DATA immediate 32-bit", Read<u32>(s + CopyImm32) == 0x12345678u, detail);
    std::snprintf(detail, sizeof(detail), "%#llx", (unsigned long long)Read<u64>(s + CopyImm64));
    check("COPY_DATA immediate 64-bit", Read<u64>(s + CopyImm64) == 0x1122334455667788ull, detail);
    std::snprintf(detail, sizeof(detail), "%#x", Read<u32>(s + CopyMem));
    check("COPY_DATA memory to memory (in order)", Read<u32>(s + CopyMem) == 0x12345678u, detail);
    std::snprintf(detail, sizeof(detail), "%llu", (unsigned long long)Read<u64>(s + CopyClock));
    check("COPY_DATA GPU clock", Read<u64>(s + CopyClock) != 0, detail);
    std::snprintf(detail, sizeof(detail), "skipped write %#x, executed write %#x",
                  Read<u32>(s + CondSkipped), Read<u32>(s + CondRun));
    check("COND_EXEC false skips, true runs",
          Read<u32>(s + CondSkipped) == 0 && Read<u32>(s + CondRun) == 0x600du, detail);
    std::snprintf(detail, sizeof(detail), "signal seen %d, label then %#x",
                  int(state.sem_seen), state.sem_label);
    check("MEM_SEMAPHORE signal after earlier work",
          state.sem_seen && state.sem_label == SemLabelValue, detail);
    std::snprintf(detail, sizeof(detail), "semaphore %llu, write after the wait %#x",
                  (unsigned long long)Read<u64>(s + SemWaited), Read<u32>(s + SemAfterWait));
    check("MEM_SEMAPHORE wait and decrement",
          Read<u64>(s + SemWaited) == 0 && Read<u32>(s + SemAfterWait) == SemAfterValue, detail);
    // The GPU clock: 100 MHz, in the CPU's time line (what the GPU wrote lands within 100 ms
    // of when it was decoded), the interval between the two timestamps as the CPU saw it.
    const u64 gpu1 = Read<u64>(s + ClockEop1), gpu2 = Read<u64>(s + ClockEop2);
    const u64 copied = Read<u64>(s + CopyClock);
    const double gpu_interval = double(gpu2 - gpu1);
    const double cpu_interval = double(state.clock_cpu2 - state.clock_cpu1);
    const auto near = [](u64 gpu, u64 cpu) { // 100 MHz: from 1 ms before to 100 ms after
        return gpu + 100'000 >= cpu && gpu < cpu + 10'000'000;
    };
    std::snprintf(detail, sizeof(detail), "%.3f s apart (CPU %.3f s); %+.2f ms, copy %+.2f ms",
                  gpu_interval / 1e8, cpu_interval / 1e8,
                  (double(gpu1) - double(state.clock_cpu1)) / 1e5,
                  (double(copied) - double(state.clock_cpu1)) / 1e5);
    check("GPU clock: 100 MHz, the CPU's time line",
          gpu2 > gpu1 && std::abs(gpu_interval - cpu_interval) < 0.05 * cpu_interval &&
              near(gpu1, state.clock_cpu1) && near(copied, state.clock_cpu1),
          detail);
    std::snprintf(detail, sizeof(detail), "%llu of 4 predicated draws skipped, 2 expected",
                  (unsigned long long)state.skips);
    check("SET_PREDICATION zpass and bool32", state.skips == 2, detail);
    std::snprintf(detail, sizeof(detail), "write %#x, label %#x", Read<u32>(s + CmpWrite),
                  Read<u32>(s + CmpLabel));
    check("compute queue: WRITE_DATA, RELEASE_MEM",
          Read<u32>(s + CmpWrite) == CmpWriteValue && Read<u32>(s + CmpLabel) == CmpLabelValue,
          detail);
    std::snprintf(detail, sizeof(detail), "semaphore %llu, write after it %#x",
                  (unsigned long long)Read<u64>(s + CmpSem), Read<u32>(s + CmpAfterSem));
    check("compute queue: waits for a graphics semaphore",
          Read<u64>(s + CmpSem) == 0 && Read<u32>(s + CmpAfterSem) == CmpSemValue, detail);
    std::snprintf(detail, sizeof(detail), "write after it %#x", Read<u32>(s + CmpAfterGate));
    check("compute queue: WAIT_REG_MEM on a graphics write",
          Read<u32>(s + CmpAfterGate) == CmpGateAfterValue, detail);
    std::printf("PM4 self-test: %s, %d failure(s)\n", failures ? "FAILED" : "all passed", failures);
}

} // namespace

bool Enabled() {
    static const bool enabled = bb_env_on("BB_PM4_SELFTEST");
    return enabled;
}

Injection Next(u32 num_dwords, const Hooks& hooks) {
    auto& s = state;
    const auto now = Clock::now();
    if (s.first == Clock::time_point{}) {
        s.first = now;
    }
    s.before.words.clear();
    s.after.words.clear();
    if (s.sem_probe) {
        // The label first, the signal after it; the signal never seen before the label.
        const u64 signal = __atomic_load_n(reinterpret_cast<const u64*>(s.scratch + SemOrdered),
                                           __ATOMIC_ACQUIRE);
        if (signal != 0) {
            s.sem_label = __atomic_load_n(reinterpret_cast<const u32*>(s.scratch + SemLabel),
                                          __ATOMIC_ACQUIRE);
            s.sem_seen = true;
            s.sem_probe = false;
        }
    }
    if (s.failed_setup || s.step > 6 || now - s.first < std::chrono::seconds(40) ||
        now - s.last_step < std::chrono::seconds(1)) {
        return {};
    }
    // BB_PM4_SELFTEST_STEPS=mask (diagnostics): the steps run (bit n: step n); others skipped.
    static const u32 steps = [] {
        const char* env = bb_env("BB_PM4_SELFTEST_STEPS");
        return env ? u32(std::strtoul(env, nullptr, 0)) : ~0u;
    }();
    if (!(steps & (1u << s.step)) && s.step != 0) {
        ++s.step;
        return {};
    }
    const VAddr base = s.scratch;
    switch (s.step) {
    case 0:
        s.scratch = AllocateScratch();
        if (!s.scratch) {
            std::printf("PM4 self-test: no scratch memory, not run\n");
            s.failed_setup = true;
            return {};
        }
        std::memset(reinterpret_cast<void*>(s.scratch), 0, ScratchSize);
        std::printf("PM4 self-test: scratch at %#llx\n", (unsigned long long)s.scratch);
        s.events_before = hooks.occlusion_events();
        break;
    case 1:
        // A large PM4 submission can contain only compute/state commands. Keep this interval
        // open until the next timed step instead: actual draws and multiple pass/command-buffer
        // boundaries occur meanwhile, irrespective of how the label backend cuts rendering.
        s.before.Zpass(base + OccVisible);
        break;
    case 2:
        s.before.Zpass(base + OccVisible + 8);
        s.before.Zpass(base + OccEmpty);
        s.before.Zpass(base + OccEmpty + 8);
        break;
    case 3:
        s.after.CopyData(CopyData::SrcImmediate, CopyData::DstMemory, false, 0x12345678u,
                         base + CopyImm32);
        s.after.CopyData(CopyData::SrcImmediate, CopyData::DstMemory, true,
                         0x1122334455667788ull, base + CopyImm64);
        s.after.CopyData(CopyData::SrcMemory, CopyData::DstMemory, false, base + CopyImm32,
                         base + CopyMem);
        s.after.CopyData(CopyData::SrcClock, CopyData::DstMemory, true, 0,
                         base + CopyClock);
        s.after.WriteData(base + CondFalse, 0);
        s.after.CondExec(base + CondFalse, 5);
        s.after.WriteData(base + CondSkipped, 0xbad);
        s.after.WriteData(base + CondTrue, 1);
        s.after.CondExec(base + CondTrue, 5);
        s.after.WriteData(base + CondRun, 0x600d);
        s.clock_cpu1 = GpuClock64();
        s.after.EopLabel(base + ClockEop1, 0, DataSel::GpuClock64);
        s.after.EopLabel(base + SemLabel, SemLabelValue);
        s.after.MemSemaphore(base + SemOrdered, true);
        s.sem_probe = true;
        // The compute queue: done at once, then waiting for the graphics queue (step 5).
        if (hooks.submit_compute) {
            auto& c = s.compute;
            c.words.clear();
            c.WriteData(base + CmpWrite, CmpWriteValue);
            c.ReleaseLabel(base + CmpLabel, CmpLabelValue);
            c.MemSemaphore(base + CmpSem, false);
            c.WriteData(base + CmpAfterSem, CmpSemValue);
            c.WaitMemEqual(base + CmpGate, CmpGateValue);
            c.WriteData(base + CmpAfterGate, CmpGateAfterValue);
            auto* ring = reinterpret_cast<u32*>(base + CmpRing);
            std::memcpy(ring, c.words.data(), c.words.size() * sizeof(u32));
            hooks.submit_compute(base + CmpRing, CmpRingDwords, base + CmpReadPtr,
                                 {ring, c.words.size()});
        }
        break;
    case 4:
        s.skips_before = hooks.predicated_skips();
        s.after.SetPredication(SetPredication::ZPass, base + OccEmpty, true);
        s.after.PredicatedEmptyDraw(); // nothing visible: skipped
        s.after.SetPredication(SetPredication::ZPass, base + OccVisible, true);
        s.after.PredicatedEmptyDraw(); // drawn
        s.after.SetPredication(SetPredication::Bool32, base + CondFalse, true);
        s.after.PredicatedEmptyDraw(); // skipped
        s.after.SetPredication(SetPredication::Bool32, base + CondTrue, true);
        s.after.PredicatedEmptyDraw(); // drawn
        s.after.SetPredication(SetPredication::Clear, 0, false);
        break;
    case 5:
        s.skips = hooks.predicated_skips() - s.skips_before;
        s.clock_cpu2 = GpuClock64();
        s.after.EopLabel(base + ClockEop2, 0, DataSel::GpuClock64);
        s.after.MemSemaphore(base + SemWaited, true);
        s.after.MemSemaphore(base + SemWaited, false); // waits for the signal, then decrements
        s.after.WriteData(base + SemAfterWait, SemAfterValue);
        s.after.MemSemaphore(base + CmpSem, true); // the compute queue's waits
        s.after.WriteData(base + CmpGate, CmpGateValue);
        break;
    case 6:
        Report(hooks);
        break;
    }
    ++s.step;
    s.last_step = now;
    return {s.before.words, s.after.words};
}

} // namespace BbCp::Pm4SelfTest
