// SPDX-License-Identifier: GPL-2.0-or-later
#include "video_core/amdgpu/pm4_selftest.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "video_core/amdgpu/pm4_cmds.h"

extern "C" std::uintptr_t runtime_memory_resolve(const char* name);

namespace AmdGpu::Pm4SelfTest {

namespace {

using Clock = std::chrono::steady_clock;
constexpr u64 ScratchSize = 64 * 1024;
constexpr u64 Valid = 0x8000000000000000ULL;
constexpr u32 Pairs = 8; // depth blocks (Liverpool::num_counter_pairs on a base PS4)

// Scratch layout (offsets).
constexpr u64 OccVisible = 0x000, OccEmpty = 0x100;
constexpr u64 CopyImm32 = 0x200, CopyImm64 = 0x208, CopyMem = 0x210, CopyClock = 0x218;
constexpr u64 CondFalse = 0x300, CondSkipped = 0x304, CondTrue = 0x308, CondRun = 0x30c;

struct Builder {
    std::vector<u32> words;

    void Header(PM4ItOpcode opcode, u32 words_after, bool predicated = false) {
        words.push_back(PM4Type3Header{opcode, words_after - 1, PM4ShaderType::ShaderGraphics,
                                       predicated ? PM4Predicate::PredEnable
                                                  : PM4Predicate::PredDisable}
                            .raw);
    }
    void Address(VAddr address) {
        words.push_back(u32(address));
        words.push_back(u32(address >> 32));
    }
    void Zpass(VAddr address) {
        Header(PM4ItOpcode::EventWrite, 3);
        words.push_back(u32(EventType::PixelPipeStatDump) | u32(EventIndex::ZpassDone) << 8);
        Address(address);
    }
    void WriteData(VAddr address, u32 value) {
        Header(PM4ItOpcode::WriteData, 4);
        words.push_back(5u << 8); // memory
        Address(address);
        words.push_back(value);
    }
    void CopyData(CopyDataSrc src, CopyDataDst dst, bool bits64, u64 source, VAddr target) {
        Header(PM4ItOpcode::CopyData, 5);
        words.push_back(u32(src) | u32(dst) << 8 | (bits64 ? 1u << 16 : 0u));
        words.push_back(u32(source));
        words.push_back(u32(source >> 32));
        Address(target);
    }
    void CondExec(VAddr address, u32 skip_dwords) {
        Header(PM4ItOpcode::CondExec, 3);
        words.push_back(u32(address) & ~3u);
        words.push_back(u32(address >> 32) & 0xffff);
        words.push_back(skip_dwords);
    }
    void SetPredication(PM4CmdSetPredication::Op op, VAddr address, bool draw_if_true) {
        Header(PM4ItOpcode::SetPredication, 2);
        words.push_back(u32(address));
        words.push_back((u32(address >> 32) & 0xff) | (draw_if_true ? 1u << 8 : 0u) |
                        u32(op) << 16);
    }
    void PredicatedEmptyDraw() {
        Header(PM4ItOpcode::DrawIndexAuto, 2, true);
        words.push_back(0); // no vertices
        words.push_back(2); // auto index
    }
};

struct State {
    int step = 0;
    Clock::time_point first{}, last_step{};
    VAddr scratch = 0;
    Builder before, after;
    u64 skips_before = 0, skips = 0;
    u64 events_before = 0;
    bool failed_setup = false;
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
    hooks.sync_for_cpu_read(s, 0x400);
    int failures = 0;
    const auto check = [&](const char* name, bool ok, const std::string& detail) {
        std::printf("PM4 self-test: %-34s %s (%s)\n", name, ok ? "PASS" : "FAIL", detail.c_str());
        failures += ok ? 0 : 1;
    };
    // Occlusion: a submission with draws, and nothing between two events.
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
    check("occlusion: samples of a submission", valid_visible && visible > 0,
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
    std::snprintf(detail, sizeof(detail), "%llu of 4 predicated draws skipped, 2 expected",
                  (unsigned long long)state.skips);
    check("SET_PREDICATION zpass and bool32", state.skips == 2, detail);
    std::printf("PM4 self-test: %s, %d failure(s)\n", failures ? "FAILED" : "all passed", failures);
}

} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_PM4_SELFTEST");
        return env && env[0] == '1';
    }();
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
    if (s.failed_setup || s.step > 6 || now - s.first < std::chrono::seconds(40) ||
        now - s.last_step < std::chrono::seconds(1)) {
        return {};
    }
    // BB_PM4_SELFTEST_STEPS=mask (diagnostics): the steps run (bit n: step n); others skipped.
    static const u32 steps = [] {
        const char* env = std::getenv("BB_PM4_SELFTEST_STEPS");
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
        if (num_dwords < 1024 && now - s.last_step < std::chrono::seconds(4)) {
            return {}; // a submission with draws, if one comes
        }
        s.before.Zpass(base + OccVisible);
        s.after.Zpass(base + OccVisible + 8);
        break;
    case 2:
        s.before.Zpass(base + OccEmpty);
        s.before.Zpass(base + OccEmpty + 8);
        break;
    case 3:
        s.after.CopyData(CopyDataSrc::Immediate, CopyDataDst::MemorySync, false, 0x12345678u,
                         base + CopyImm32);
        s.after.CopyData(CopyDataSrc::Immediate, CopyDataDst::MemorySync, true,
                         0x1122334455667788ull, base + CopyImm64);
        s.after.CopyData(CopyDataSrc::Memory, CopyDataDst::MemorySync, false, base + CopyImm32,
                         base + CopyMem);
        s.after.CopyData(CopyDataSrc::GpuClock, CopyDataDst::MemorySync, true, 0,
                         base + CopyClock);
        s.after.WriteData(base + CondFalse, 0);
        s.after.CondExec(base + CondFalse, 5);
        s.after.WriteData(base + CondSkipped, 0xbad);
        s.after.WriteData(base + CondTrue, 1);
        s.after.CondExec(base + CondTrue, 5);
        s.after.WriteData(base + CondRun, 0x600d);
        break;
    case 4:
        s.skips_before = hooks.predicated_skips();
        s.after.SetPredication(PM4CmdSetPredication::Op::ZPass, base + OccEmpty, true);
        s.after.PredicatedEmptyDraw(); // nothing visible: skipped
        s.after.SetPredication(PM4CmdSetPredication::Op::ZPass, base + OccVisible, true);
        s.after.PredicatedEmptyDraw(); // drawn
        s.after.SetPredication(PM4CmdSetPredication::Op::Bool32, base + CondFalse, true);
        s.after.PredicatedEmptyDraw(); // skipped
        s.after.SetPredication(PM4CmdSetPredication::Op::Bool32, base + CondTrue, true);
        s.after.PredicatedEmptyDraw(); // drawn
        s.after.SetPredication(PM4CmdSetPredication::Op::Clear, 0, false);
        break;
    case 5:
        s.skips = hooks.predicated_skips() - s.skips_before;
        break;
    case 6:
        Report(hooks);
        break;
    }
    ++s.step;
    s.last_step = now;
    return {s.before.words, s.after.words};
}

} // namespace AmdGpu::Pm4SelfTest
