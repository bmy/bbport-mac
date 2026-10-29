// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <algorithm>
#include "shim/bb_env.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include <sys/uio.h>
#include <unistd.h>

#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>

#include "bbport_write_log.h"
#include "bbport_free_check.h"
#include "bbport_gnm_hooks.h"
#include "core/emulator_settings.h"
#include "cp/cp_diagnostics.h"
#include "cp/cp_pm4.h"

template <>
struct magic_enum::customize::enum_range<BbCp::Op> {
    static constexpr int min = 0;
    static constexpr int max = 255;
};

extern "C" int runtime_memory_vma_info(uintptr_t address, int* prot, int* type, uintptr_t* end);
extern "C" void runtime_memory_read_backing(uintptr_t address, void* data, u64 size);

namespace BbCp::Diagnostics {

std::string OpName(u32 op) {
    const auto name = magic_enum::enum_name(Op(op));
    return name.empty() ? fmt::format("{:#x}", op) : std::string(name);
}

namespace {

struct LabelRecord {
    std::atomic<u64> when_ns{0}; ///< steady clock; 0: not written yet
    u64 address;
    u64 value;
    u32 bytes;
    LabelStep step;
};
constexpr size_t LabelRingSize = 256;
std::array<LabelRecord, LabelRingSize> label_ring;
std::atomic<u64> label_next{0};

u64 NowNs() {
    return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
                   .count());
}

} // namespace

void NoteLabel(LabelStep step, u64 address, u64 value, u32 bytes) {
    LabelRecord& record = label_ring[label_next.fetch_add(1, std::memory_order_relaxed) %
                                     LabelRingSize];
    record.when_ns.store(0, std::memory_order_relaxed);
    record.address = address;
    record.value = value;
    record.bytes = bytes;
    record.step = step;
    record.when_ns.store(NowNs(), std::memory_order_release);
}

bool ReadGuest(const void* address, void* out, u32 bytes) {
    int prot, type;
    uintptr_t end;
    const auto at = reinterpret_cast<uintptr_t>(address);
    if (bytes == 0 || !runtime_memory_vma_info(at, &prot, &type, &end) || end < at + bytes) {
        return false;
    }
    runtime_memory_read_backing(at, out, bytes);
    return true;
}

std::string RecentLabels(size_t count) {
    static constexpr const char* steps[] = {"compute decoded", "compute GPU write",
                                            "compute written", "graphics GPU write",
                                            "graphics written"};
    const u64 next = label_next.load(std::memory_order_acquire);
    const u64 now = NowNs();
    count = std::min<u64>({count, next, LabelRingSize});
    std::string out;
    for (u64 i = next - count; i < next; ++i) {
        const LabelRecord& record = label_ring[i % LabelRingSize];
        const u64 when = record.when_ns.load(std::memory_order_acquire);
        if (when == 0) {
            continue;
        }
        // What the guest reads there now (unreadable: a trapped or unmapped page).
        u64 current = 0;
        const u32 bytes = std::min<u32>(record.bytes, 8);
        const bool readable =
            ReadGuest(reinterpret_cast<const void*>(record.address), &current, bytes);
        out += fmt::format("  {:>8.1f} ms ago  {:<18} {:#x} = {:#x}{}\n",
                           double(now - when) / 1e6, steps[u32(record.step) % 5], record.address,
                           record.value,
                           !readable ? " (memory unreadable now)"
                           : current != (bytes == 8 ? record.value : record.value & 0xFFFFFFFFull)
                               ? fmt::format(" (memory holds {:#x} now)", current)
                               : std::string{});
    }
    return out;
}

void DumpRolledBackLabels(size_t count) {
    const u64 next = label_next.load(std::memory_order_acquire);
    count = std::min<u64>({count, next, LabelRingSize});
    std::vector<std::pair<u64, u64>> newest; // address, value of its newest write
    for (u64 i = next - count; i < next; ++i) {
        const LabelRecord& record = label_ring[i % LabelRingSize];
        if (record.when_ns.load(std::memory_order_acquire) == 0) {
            continue;
        }
        const u64 value = record.bytes >= 8 ? record.value : record.value & 0xFFFFFFFFull;
        const auto it = std::find_if(newest.begin(), newest.end(),
                                     [&](const auto& n) { return n.first == record.address; });
        if (it != newest.end()) {
            it->second = value;
        } else {
            newest.emplace_back(record.address, value);
        }
    }
    u32 dumped = 0;
    for (const auto& [address, value] : newest) {
        u64 current = 0;
        if (dumped < 6 && ReadGuest(reinterpret_cast<const void*>(address), &current,
                                    value >> 32 ? 8 : 4) &&
            current != 0 && current != value) {
            BbFreeCheck::DumpWrites(address, 8, "label rolled back");
            ++dumped;
        }
    }
}

namespace {

struct DcbStats {
    u64 buffers[2]{}, with_draws[2]{}, draws[2]{}, dispatches[2]{}, max_draws[2]{};
    u64 clear_first[2]{}, ctx_before[2]{}, sh_before[2]{}, ibs[2]{}, dwords[2]{};
    u64 hist[2][6]{}; // draws per buffer: 0, 1-9, 10-99, 100-499, 500-1999, 2000+
    // Packets by opcode (any depth), EVENT_WRITE by event type, EOP by data and interrupt
    // select, EOS by command, WAIT_REG_MEM by memory space and engine.
    u64 opcodes[256]{}, events[64]{}, eop_data[8]{}, eop_int[4]{}, eos_cmd[8]{}, wait_mem[2][2]{};
    u64 wait_outcome[4]{};
    double wait_ms = 0;
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
};
DcbStats stats;

bool IsDraw(Op op) {
    switch (op) {
    case Op::DrawIndex2:
    case Op::DrawIndexOffset2:
    case Op::DrawIndexAuto:
    case Op::DrawIndirect:
    case Op::DrawIndirectMulti:
    case Op::DrawIndexIndirect:
    case Op::DrawIndexIndirectMulti:
    case Op::DrawIndexIndirectCountMulti:
        return true;
    default:
        return false;
    }
}

} // namespace

bool DcbStatsEnabled() {
    static const bool enabled = EmulatorSettingsImpl::Flag("BB_DCB_STATS", false);
    return enabled;
}

void NoteWait(int outcome, double ms) {
    ++stats.wait_outcome[outcome];
    stats.wait_ms += ms;
}

void ScanDcb(std::span<const u32> dcb, int depth) {
    auto& st = stats;
    const int d = depth > 0 ? 1 : 0;
    u64 draws = 0, dispatches = 0, ctx = 0, sh = 0, ibs = 0;
    bool clear_first = false;
    st.dwords[d] += dcb.size();
    for (size_t at = 0; at < dcb.size();) {
        const Header header{dcb[at]};
        if (header.Type() == 2) {
            ++at;
            continue;
        }
        if (header.Type() != 3) {
            break;
        }
        const Op op = header.Opcode();
        const u32* body = dcb.data() + at + 1;
        ++st.opcodes[u8(op)];
        if (op == Op::EventWrite) {
            ++st.events[body[0] & 63];
        } else if (op == Op::EventWriteEop) {
            ++st.eop_data[(body[2] >> 29) & 7];
            ++st.eop_int[(body[2] >> 24) & 3];
        } else if (op == Op::EventWriteEos) {
            ++st.eos_cmd[(body[2] >> 29) & 7];
        } else if (op == Op::WaitRegMem) {
            ++st.wait_mem[(body[0] >> 4) & 1][(body[0] >> 8) & 1];
        }
        if (IsDraw(op)) {
            ++draws;
        } else if (op == Op::DispatchDirect || op == Op::DispatchIndirect) {
            ++dispatches;
        } else if (op == Op::IndirectBuffer) {
            ++ibs;
        } else if (draws == 0) {
            if (op == Op::ClearState) {
                clear_first = true;
            } else if (op == Op::SetContextReg) {
                ctx += header.BodyWords() - 1;
            } else if (op == Op::SetShReg) {
                sh += header.BodyWords() - 1;
            }
        }
        at += header.BodyWords() + 1;
    }
    ++st.buffers[d];
    st.draws[d] += draws;
    st.dispatches[d] += dispatches;
    st.ibs[d] += ibs;
    st.max_draws[d] = std::max(st.max_draws[d], draws);
    if (draws) {
        ++st.with_draws[d];
        st.clear_first[d] += clear_first;
        st.ctx_before[d] += ctx;
        st.sh_before[d] += sh;
    }
    const int bucket = draws == 0 ? 0 : draws < 10 ? 1 : draws < 100 ? 2 : draws < 500 ? 3
                     : draws < 2000 ? 4 : 5;
    ++st.hist[d][bucket];

    const auto now = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(now - st.window).count();
    if (depth != 0 || secs < 5.0) {
        return;
    }
    for (int k = 0; k < 2; ++k) {
        const double with = double(std::max<u64>(st.with_draws[k], 1));
        std::printf("DCB stats %s: %.0f buffers/s (%.0f with draws), %.0f draws/s, %.0f "
                    "dispatches/s, max %llu draws in one buffer, %.0f IBs/s, %.0f kdw/s; "
                    "draw buffers: %.0f%% start with ClearState, %.0f ctx + %.0f sh reg "
                    "writes before first draw; draws/buffer hist 0:%llu 1-9:%llu 10-99:%llu "
                    "100-499:%llu 500-1999:%llu 2000+:%llu\n",
                    k ? "indirect" : "submitted", st.buffers[k] / secs, st.with_draws[k] / secs,
                    st.draws[k] / secs, st.dispatches[k] / secs,
                    (unsigned long long)st.max_draws[k], st.ibs[k] / secs,
                    st.dwords[k] / secs / 1000.0, 100.0 * st.clear_first[k] / with,
                    st.ctx_before[k] / with, st.sh_before[k] / with,
                    (unsigned long long)st.hist[k][0], (unsigned long long)st.hist[k][1],
                    (unsigned long long)st.hist[k][2], (unsigned long long)st.hist[k][3],
                    (unsigned long long)st.hist[k][4], (unsigned long long)st.hist[k][5]);
    }
    std::string line;
    for (u32 op = 0; op < 256; ++op) {
        if (st.opcodes[op]) {
            line += fmt::format(" {}:{:.0f}", OpName(op), st.opcodes[op] / secs);
        }
    }
    std::printf("PM4 packets/s:%s\n", line.c_str());
    line.clear();
    for (u32 event = 0; event < 64; ++event) {
        if (st.events[event]) {
            line += fmt::format(" {}:{:.0f}", event, st.events[event] / secs);
        }
    }
    std::printf("PM4 events/s:%s; EOP data_sel %.0f/%.0f/%.0f/%.0f/%.0f (none/32/64/clock/perf)"
                " int_sel %.0f/%.0f/%.0f/%.0f; EOS cmd %.0f/%.0f/%.0f/%.0f/%.0f; WAIT_REG_MEM "
                "reg me/pfp %.0f/%.0f mem me/pfp %.0f/%.0f\n",
                line.c_str(), st.eop_data[0] / secs, st.eop_data[1] / secs, st.eop_data[2] / secs,
                st.eop_data[3] / secs, st.eop_data[4] / secs, st.eop_int[0] / secs,
                st.eop_int[1] / secs, st.eop_int[2] / secs, st.eop_int[3] / secs,
                st.eos_cmd[0] / secs, st.eos_cmd[1] / secs, st.eos_cmd[2] / secs,
                st.eos_cmd[3] / secs, st.eos_cmd[4] / secs, st.wait_mem[0][0] / secs,
                st.wait_mem[0][1] / secs, st.wait_mem[1][0] / secs, st.wait_mem[1][1] / secs);
    std::printf("WAIT_REG_MEM outcomes/s: VO label %.0f, pending fence %.0f, met %.0f, waited "
                "%.0f (%.2f ms/s)\n",
                st.wait_outcome[0] / secs, st.wait_outcome[1] / secs, st.wait_outcome[2] / secs,
                st.wait_outcome[3] / secs, st.wait_ms / secs);
    st = DcbStats{};
}

void InspectSubmittedCopy(SubmittedCopy& copy) {
    for (size_t at = 0; at < copy.dcb_dwords;) {
        const u32 header = copy.data[at];
        if ((header >> 30) == 2) {
            ++at;
            continue;
        }
        const size_t length = ((header >> 16) & 0x3fff) + 2;
        if ((header >> 30) != 3 || length > copy.dcb_dwords - at) {
            copy.bad_dcb_at_submit = at;
            return;
        }
        at += length;
    }
}

namespace {
size_t PacketDwords(u32 header) {
    return (header >> 30) == 2 ? 1 : ((header >> 16) & 0x3fff) + 2;
}
bool PlausibleHeader(u32 header) {
    return (header >> 30) == 2 || ((header >> 30) == 3 && ((header >> 8) & 0xff) != 0);
}
/// The first malformed top-level packet from `at`, or the dcb's size.
size_t FirstBad(const SubmittedCopy& copy, size_t at) {
    while (at < copy.dcb_dwords) {
        const u32 header = copy.data[at];
        if (!PlausibleHeader(header) || PacketDwords(header) > copy.dcb_dwords - at) {
            return at;
        }
        at += PacketDwords(header);
    }
    return copy.dcb_dwords;
}
} // namespace

bool RepairSubmittedCopy(SubmittedCopy& copy, u64 number) {
    // The guest's own buffer is malformed when submitted, now and then while an area loads on
    // the 2 GB GTX 1050 laptop (~1 load in 50, right as the character appears): a packet whose
    // header promises values that are not there (the next packet follows at once), runs of
    // zeros, a header with another count (old bytes over what the game wrote?). Decoded as it
    // is, the rest is garbage: "PM4 type 0", a fault reading a WAIT_REG_MEM "address" (the next
    // packet's NOP marker 0x6875000x), a GPU fault (TEX FORMAT). Repaired here, at most 16 times:
    // a packet among the last 8 before the bad header that, cut short, leaves the rest well
    // formed (at least 32 dwords further) becomes a NOP over what it has; else zeros at the bad
    // header become 1-dword fillers. 3 of the 6 kept laptop snapshots parse to the end so.
    // BB_PM4_REPAIR=0: decoded as submitted.
    static const bool enabled = !bb_env_off("BB_PM4_REPAIR");
    if (copy.bad_dcb_at_submit == ~size_t(0) || !enabled) {
        return false;
    }
    const u32* data = copy.data.data();
    copy.repaired = true;
    // Who wrote there (BB_POOL_WATCH=1: the GPU side's writes into guest memory, copy-backs
    // among them; BB_WRITE_LOG=1: the CPU's): once per session, the first damaged buffer.
    static std::atomic<bool> dumped{false};
    if (!dumped.exchange(true)) {
        const size_t from = copy.bad_dcb_at_submit > 16 ? copy.bad_dcb_at_submit - 16 : 0;
        const auto address = reinterpret_cast<std::uint64_t>(copy.guest_dcb + from);
        BbFreeCheck::DumpWrites(address, 32 * sizeof(u32), "malformed command buffer");
        BbWriteLog::DumpRange(address, 32 * sizeof(u32));
        BbGnmHooks::ReportDriverWrites(address, 32 * sizeof(u32)); // BB_GNM_WRITES_TRACE=1
    }
    for (int repairs = 0; repairs < 16; ++repairs) {
        const size_t bad = FirstBad(copy, 0);
        if (bad >= copy.dcb_dwords) {
            std::fprintf(stderr, "PM4 repair: submission %llu well formed after %d repairs\n",
                         (unsigned long long)number, repairs);
            copy.bad_dcb_at_submit = ~size_t(0);
            return true;
        }
        std::vector<size_t> starts;
        for (size_t at = 0; at < bad; at += PacketDwords(data[at])) {
            starts.push_back(at);
        }
        bool done = false;
        for (size_t back = 0; back < std::min<size_t>(starts.size(), 8) && !done; ++back) {
            const size_t start = starts[starts.size() - 1 - back];
            const u32 header = data[start];
            if ((header >> 30) != 3) {
                continue;
            }
            const size_t length = PacketDwords(header);
            for (size_t cut = 1; cut < length && start + cut < copy.dcb_dwords; ++cut) {
                const u32 next = data[start + cut];
                if ((next >> 30) != 3 || !PlausibleHeader(next)) {
                    continue;
                }
                const size_t next_bad = FirstBad(copy, start + cut);
                if (next_bad < copy.dcb_dwords && next_bad < bad + 32) {
                    continue;
                }
                // A NOP (type 3 opcode 0x10) over the `cut` dwords written.
                copy.data[start] = cut >= 2 ? 0xC0001000u | u32(cut - 2) << 16 : 0x80000000u;
                std::fprintf(stderr,
                             "PM4 repair: submission %llu: packet at +%#llx (header %#x, %llu "
                             "dwords) cut to %llu, made a NOP (bad header at +%#llx)\n",
                             (unsigned long long)number, (unsigned long long)(start * 4), header,
                             (unsigned long long)length, (unsigned long long)cut,
                             (unsigned long long)(bad * 4));
                done = true;
                break;
            }
        }
        if (!done && data[bad] == 0) {
            size_t end = bad;
            while (end < copy.dcb_dwords && data[end] == 0) {
                copy.data[end++] = 0x80000000u;
            }
            std::fprintf(stderr, "PM4 repair: submission %llu: zeros at +%#llx..+%#llx made "
                                 "fillers\n",
                         (unsigned long long)number, (unsigned long long)(bad * 4),
                         (unsigned long long)(end * 4));
            done = true;
        }
        // A few stray dwords between whole packets (laptop, 2026-10-25: `00010000 00024004`
        // after complete SET_SH_REGs, then the next packet's NOP marker): they become fillers when
        // the buffer parses further from the next header on (more repairs follow).
        for (size_t junk = 1; !done && junk <= 8 && bad + junk < copy.dcb_dwords; ++junk) {
            const u32 next = data[bad + junk];
            if ((next >> 30) != 3 || !PlausibleHeader(next)) {
                continue;
            }
            const size_t next_bad = FirstBad(copy, bad + junk);
            if (next_bad < copy.dcb_dwords && next_bad < bad + junk + 4) {
                continue;
            }
            for (size_t i = bad; i < bad + junk; ++i) {
                copy.data[i] = 0x80000000u;
            }
            std::fprintf(stderr, "PM4 repair: submission %llu: %llu stray dwords at +%#llx "
                                 "(first %#x) made fillers\n",
                         (unsigned long long)number, (unsigned long long)junk,
                         (unsigned long long)(bad * 4), data[bad]);
            done = true;
        }
        if (!done) {
            std::fprintf(stderr, "PM4 repair: submission %llu: no repair for header %#x at +%#llx\n",
                         (unsigned long long)number, data[bad], (unsigned long long)(bad * 4));
            copy.bad_dcb_at_submit = bad;
            return false;
        }
    }
    copy.bad_dcb_at_submit = FirstBad(copy, 0);
    return false;
}

void RememberSubmittedCopy(std::shared_ptr<const SubmittedCopy> copy) {
    static const bool enabled = bb_env_on("BB_GNM_WRITES_TRACE");
    if (!enabled) return;
    struct Kept { u64 number; std::shared_ptr<const SubmittedCopy> copy; };
    static std::mutex mutex;
    static std::deque<Kept> history;
    static u64 next = 0, kept_bytes = 0;
    std::scoped_lock lock{mutex};
    const u64 number = next++;
    const auto begin = reinterpret_cast<uintptr_t>(copy->guest_dcb);
    if (copy->bad_dcb_at_submit != ~size_t(0)) {
        const auto bad = begin + copy->bad_dcb_at_submit * 4;
        std::fprintf(stderr, "PM4 preflight history #%llu: source %#llx +%#llx bytes, first bad +%#llx\n",
                     (unsigned long long)number, (unsigned long long)begin,
                     (unsigned long long)(copy->dcb_dwords * 4),
                     (unsigned long long)(copy->bad_dcb_at_submit * 4));
        unsigned shown = 0;
        for (auto it = history.rbegin(); it != history.rend() && shown < 3; ++it) {
            const auto& old = *it->copy;
            const auto old_begin = reinterpret_cast<uintptr_t>(old.guest_dcb);
            if (bad < old_begin || bad >= old_begin + old.dcb_dwords * 4) continue;
            const size_t word = (bad - old_begin) / 4;
            size_t packet = 0;
            while (packet < old.dcb_dwords) {
                const auto h = old.data[packet];
                const size_t n = (h >> 30) == 2 ? 1 : ((h >> 16) & 0x3fff) + 2;
                if (((h >> 30) != 3 && (h >> 30) != 2) ||
                    n > old.dcb_dwords - packet || packet + n > word) break;
                packet += n;
            }
            std::fprintf(stderr, "  prior #%llu source %#llx +%#llx bytes; bad address was +%#llx, "
                                 "packet at +%#llx header %#x\n",
                         (unsigned long long)it->number, (unsigned long long)old_begin,
                         (unsigned long long)(old.dcb_dwords * 4), (unsigned long long)(word * 4),
                         (unsigned long long)(packet * 4), packet < old.dcb_dwords ? old.data[packet] : 0);
            const size_t from = word > 16 ? word - 16 : 0;
            std::fprintf(stderr, "    prior words from +%#llx:", (unsigned long long)(from * 4));
            for (size_t p = from; p < std::min(old.dcb_dwords, word + 16); ++p)
                std::fprintf(stderr, p == word ? " [%08x]" : " %08x", old.data[p]);
            std::fprintf(stderr, "\n");
            const char* dir = bb_env("BB_DATA_DIR");
            const auto path = fmt::format("{}/out/pm4-history-{}-prior-{}.bin", dir ? dir : ".",
                                          number, it->number);
            if (FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(old.data.data(), sizeof(u32), old.dcb_dwords, f);
                std::fclose(f);
                std::fprintf(stderr, "    prior snapshot: %s\n", path.c_str());
            }
            ++shown;
        }
        if (!shown) std::fprintf(stderr, "  no retained submission covers the bad address\n");
        const auto from = copy->bad_dcb_at_submit > 16 ? copy->bad_dcb_at_submit - 16 : 0;
        BbGnmHooks::ReportDriverWrites(begin + from * 4,
                                       std::min<size_t>(32, copy->dcb_dwords - from) * 4);
    }
    kept_bytes += copy->data.size() * sizeof(u32);
    history.push_back({number, std::move(copy)});
    while (history.size() > 512 || kept_bytes > (64u << 20)) {
        kept_bytes -= history.front().copy->data.size() * sizeof(u32);
        history.pop_front();
    }
}

void ReportBadPacket(uintptr_t base, std::size_t dwords, const u32* at, u64 seq, int depth,
                     const SubmittedCopy* copy) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(at);
    int prot = 0, type = -1;
    uintptr_t end = 0;
    runtime_memory_vma_info(address, &prot, &type, &end);
    std::fprintf(stderr,
                 "Bad PM4 packet at %#llx: dword %#x, %#llx bytes into a %#llx-byte buffer at %#llx "
                 "(IB depth %d, submission %llu; memory type %d, prot %#x)\n",
                 (unsigned long long)address, *at, (unsigned long long)(address - base),
                 (unsigned long long)(dwords * 4), (unsigned long long)base, depth,
                 (unsigned long long)seq, type, prot);
    const u32* first = reinterpret_cast<const u32*>(std::max(base, address - 64));
    const u32* last = std::min(reinterpret_cast<const u32*>(base) + dwords, at + 16);
    for (const u32* p = first; p < last; p += 8) {
        std::fprintf(stderr, "  %#llx:", (unsigned long long)reinterpret_cast<uintptr_t>(p));
        for (const u32* q = p; q < std::min(p + 8, last); ++q) {
            std::fprintf(stderr, q == at ? " [%08x]" : " %08x", *q);
        }
        std::fprintf(stderr, "\n");
    }
    if (copy && depth == 0 &&
        base == reinterpret_cast<uintptr_t>(copy->data.data())) {
        const size_t bad = size_t(at - copy->data.data());
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copy->submitted).count();
        std::fprintf(stderr, "  submission source %p, snapshot age %.3f ms; ",
                     static_cast<const void*>(copy->guest_dcb), ms);
        if (copy->bad_dcb_at_submit == ~size_t(0)) {
            std::fprintf(stderr, "snapshot had valid packet boundaries at submit\n");
        } else {
            std::fprintf(stderr, "snapshot ALREADY invalid at submit +%#llx bytes\n",
                         (unsigned long long)(copy->bad_dcb_at_submit * 4));
        }
        const size_t from = bad > 16 ? bad - 16 : 0;
        const size_t count = std::min<size_t>(32, copy->dcb_dwords - from);
        std::array<u32, 32> source{};
        iovec local{source.data(), count * 4};
        iovec remote{const_cast<u32*>(copy->guest_dcb + from), count * 4};
        const bool read = process_vm_readv(getpid(), &local, 1, &remote, 1, 0) ==
                              ssize_t(count * 4) ||
                          ReadGuest(copy->guest_dcb + from, source.data(), u32(count * 4));
        if (read) {
            std::fprintf(stderr, "  source now at bad header: %#x (snapshot %#x)\n",
                         source[bad - from], *at);
            for (size_t p = 0; p < count; p += 8) {
                std::fprintf(stderr, "  source +%04llx:", (unsigned long long)((from + p) * 4));
                for (size_t q = p; q < std::min(p + 8, count); ++q) {
                    std::fprintf(stderr, from + q == bad ? " [%08x]" : " %08x", source[q]);
                }
                std::fprintf(stderr, "\n");
            }
        } else {
            std::fprintf(stderr, "  source memory is no longer readable\n");
        }
        // Track writes to guest memory, not the malloc'ed snapshot (which has none by design).
        BbWriteLog::DumpRange(reinterpret_cast<uintptr_t>(copy->guest_dcb), dwords * 4);
        BbGnmHooks::ReportDriverWrites(reinterpret_cast<uintptr_t>(copy->guest_dcb + from), count * 4);
        const char* data_dir = bb_env("BB_DATA_DIR");
        const std::string name = std::string(data_dir ? data_dir : ".") +
                                 "/out/pm4-bad-" + std::to_string(seq) + ".snapshot.bin";
        if (FILE* file = std::fopen(name.c_str(), "wb")) {
            const size_t written = std::fwrite(copy->data.data(), sizeof(u32), copy->dcb_dwords, file);
            std::fclose(file);
            if (written == copy->dcb_dwords) std::fprintf(stderr, "  full snapshot: %s\n", name.c_str());
        }
    } else {
        BbWriteLog::DumpRange(base, dwords * 4);
    }
    std::fflush(stderr);
}

void CheckSubmittedCopy(const SubmittedCopy& copy, u64 seq) {
    // The guest memory may be unmapped by now: read it without faulting.
    static std::atomic<u64> changed{0}, checked{0};
    const auto first_change = [](const u32* guest, const u32* kept, std::size_t dwords) -> s64 {
        std::array<u32, 1024> buf;
        for (std::size_t at = 0; at < dwords; at += buf.size()) {
            const std::size_t n = std::min(buf.size(), dwords - at);
            iovec local{buf.data(), n * 4}, remote{const_cast<u32*>(guest + at), n * 4};
            if (process_vm_readv(getpid(), &local, 1, &remote, 1, 0) != ssize_t(n * 4)) {
                // bbport: dma-buf guest memory is not readable that way; still
                // mapped, it is read directly.
                int prot = 0, type = -1;
                uintptr_t end = 0;
                const auto from = reinterpret_cast<uintptr_t>(guest + at);
                if (!runtime_memory_vma_info(from, &prot, &type, &end) || end < from + n * 4) {
                    return -2 - s64(at); // unmapped
                }
                std::memcpy(buf.data(), guest + at, n * 4);
            }
            for (std::size_t i = 0; i < n; ++i) {
                if (buf[i] != kept[at + i]) {
                    return s64(at + i);
                }
            }
        }
        return -1;
    };
    checked.fetch_add(1, std::memory_order_relaxed);
    const s64 dcb_change = first_change(copy.guest_dcb, copy.data.data(), copy.dcb_dwords);
    const s64 ccb_change =
        first_change(copy.guest_ccb, copy.data.data() + copy.dcb_dwords, copy.ccb_dwords);
    if (dcb_change == -1 && ccb_change == -1) {
        return;
    }
    const u64 n = changed.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n > 16 && n % 100 != 0) {
        return;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                copy.submitted)
                          .count();
    const s64 at = dcb_change != -1 ? dcb_change : ccb_change;
    std::printf("Command buffers: the guest changed submission %llu's %s at %p +%#llx (%s) "
                "%.3f ms after submitting it, before it was decoded; %llu of %llu so far "
                "(decoded from the copy)\n",
                (unsigned long long)seq, dcb_change != -1 ? "dcb" : "ccb",
                static_cast<const void*>(dcb_change != -1 ? copy.guest_dcb : copy.guest_ccb),
                (unsigned long long)((at < -1 ? -2 - at : at) * 4),
                at < -1 ? "unmapped" : "rewritten", ms, (unsigned long long)n,
                (unsigned long long)checked.load());
    if (n > 4 || at < 0) {
        return;
    }
    // What changed, and the packets before it (a WaitRegMem the guest releases after patching,
    // or a write of ours into its own command buffer).
    const u32* kept = dcb_change != -1 ? copy.data.data() : copy.data.data() + copy.dcb_dwords;
    const u32* guest = dcb_change != -1 ? copy.guest_dcb : copy.guest_ccb;
    const std::size_t dwords = dcb_change != -1 ? copy.dcb_dwords : copy.ccb_dwords;
    const std::size_t from = at >= 8 ? at - 8 : 0, to = std::min<std::size_t>(dwords, at + 8);
    std::array<u32, 16> now{};
    iovec local{now.data(), (to - from) * 4}, remote{const_cast<u32*>(guest + from), (to - from) * 4};
    const bool readable = process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == ssize_t((to - from) * 4);
    std::printf("  submitted:");
    for (std::size_t i = from; i < to; ++i) {
        std::printf(i == std::size_t(at) ? " [%08x]" : " %08x", kept[i]);
    }
    std::printf("\n  now:      ");
    for (std::size_t i = from; readable && i < to; ++i) {
        std::printf(i == std::size_t(at) ? " [%08x]" : " %08x", now[i - from]);
    }
    std::printf("\n  packets before it (dword offset:opcode):");
    std::array<std::pair<u32, u32>, 12> last{};
    std::size_t count = 0;
    for (std::size_t p = 0; p < std::size_t(at) && p < dwords;) {
        const u32 header = kept[p];
        if ((header >> 30) == 3) {
            last[count++ % last.size()] = {u32(p), (header >> 8) & 0xff};
            p += ((header >> 16) & 0x3fff) + 2;
        } else if ((header >> 30) == 2) {
            ++p;
        } else {
            break;
        }
    }
    for (std::size_t i = count > last.size() ? count - last.size() : 0; i < count; ++i) {
        std::printf(" %#x:%#x", last[i % last.size()].first, last[i % last.size()].second);
    }
    std::printf("\n");
}

} // namespace BbCp::Diagnostics
