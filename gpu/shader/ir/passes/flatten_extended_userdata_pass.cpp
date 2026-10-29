// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Shaders read descriptors and constants through pointers in their user data (s_load_dword from
// shader resource tables, possibly several levels deep, with computed offsets). The host shader
// reads one flat buffer instead: this pass gives every such read (ReadConst, and constant buffer
// reads that produce sharps) a slot in it after the user data registers, and generates the x86
// "SRT walker" that fills the buffer at draw time: it follows the same pointers from the user
// data, recomputing dynamic offsets from slots already copied.
//
// Reads that compute the same value (global value numbering over the pointer and offset
// arithmetic) share a slot. A walker load from unmapped memory (a stale pointer) faults; the
// fault handler supplies 0 for that invocation and resumes without changing shared code.
//
// bbport (macOS native GPU process, docs/macos-native-gpu.md): on other hosts (arm64) the walker
// is bytecode instead, the same walk with the same slots, run by RunSrtWalker; an unreadable
// load there gives 0 (srt_guest_readable).
#include <algorithm>
#include <atomic>
#include <csetjmp>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <unordered_map>
#include <vector>

#include <boost/container/flat_map.hpp>
#include <boost/container/small_vector.hpp>
#include <magic_enum/magic_enum.hpp>

#include "common/arch.h"
#ifdef ARCH_X86_64
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
#endif
#include "common/assert.h"
#include "common/decoder.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/serdes.h"
#include "common/signal_context.h"
#include "core/emulator_settings.h"
#include "core/signals.h"
#include "bbport_toggles.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/passes/srt.h"
#include "shader/ir/program.h"
#include "shader/info.h"

#ifdef ARCH_X86_64

namespace Shader {

namespace {

using namespace Xbyak::util;

/// All walkers live in one executable buffer. Published instructions are immutable.
struct WalkerCode {
    std::mutex mutex;
    Xbyak::CodeGenerator code{32_MB};
    const u8* start{};
    std::atomic<size_t> published_size{0};
};

WalkerCode& Walkers() {
    static WalkerCode walkers;
    return walkers;
}

/// Supply zero for this failed load only. Patching the shared MOV raced with other threads'
/// fault handlers and permanently discarded valid data on later draws using the same walker.
bool HandleFaultingWalkerLoad(void* context, void* fault_address) {
    WalkerCode& walkers = Walkers();
    const u8* const code = static_cast<const u8*>(Common::GetRip(context));
    const size_t published_size = walkers.published_size.load(std::memory_order_acquire);
    if (!published_size || code < walkers.start || code >= walkers.start + published_size) {
        return false;
    }
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    const ZyanStatus status = Common::Decoder::Instance()->decodeInstruction(
        instruction, operands, const_cast<u8*>(code), 15);
    if (!ZYAN_SUCCESS(status) || instruction.mnemonic != ZYDIS_MNEMONIC_MOV ||
        operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
        operands[1].type != ZYDIS_OPERAND_TYPE_MEMORY) {
        return false;
    }
    greg_t* const registers = Common::Gregs(context); // bbport (macOS): Darwin's thread state
    switch (operands[0].reg.value) {
    case ZYDIS_REGISTER_RDI:
        registers[REG_RDI] = 0;
        break;
    case ZYDIS_REGISTER_R10D:
        registers[REG_R10] = 0; // a 32-bit MOV also clears the upper half
        break;
    default:
        return false;
    }
    registers[REG_RIP] += instruction.length;
    static std::atomic<u64> faults{0};
    const u64 count = faults.fetch_add(1, std::memory_order_relaxed) + 1;
    if (count <= 5 || (count & (count - 1)) == 0) {
        LOG_WARNING(Render_Recompiler, "SRT walker read unavailable memory at {} (instruction {}, "
                    "{} reads so far); supplied zero for this draw", fault_address,
                    static_cast<const void*>(code), count);
    }
    return true;
}

/// The walker buffer, with the fault handler registered on first use (lock held).
Xbyak::CodeGenerator& WalkerBuffer(WalkerCode& walkers) {
    if (!walkers.start) {
        walkers.start = walkers.code.getCurr();
        // After the memory tracking handler.
        constexpr u32 Priority = 2;
        Core::Signals::Instance()->RegisterAccessViolationHandler(HandleFaultingWalkerLoad,
                                                                  Priority);
    }
    return walkers.code;
}

void DumpWalker(const Info& info, const u8* code, size_t size) {
    using namespace Common::FS;
    const auto dir = GetUserPath(PathType::ShaderDir) / "dumps";
    std::filesystem::create_directories(dir);
    const auto name = fmt::format("{}_{:#018x}.srtprogram.txt", info.hw_stage, info.pgm_hash);
    const IOFile file{dir / name, FileAccessMode::Create, FileType::TextFile};
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    for (u64 address = u64(code); address < u64(code) + size; address += instruction.length) {
        auto* const decoder = Common::Decoder::Instance();
        if (!ZYAN_SUCCESS(decoder->decodeInstruction(instruction, operands,
                                                     reinterpret_cast<void*>(address)))) {
            break;
        }
        file.WriteString(decoder->disassembleInst(instruction, operands, address) + "\n");
    }
}

} // namespace

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    WalkerCode& walkers = Walkers();
    std::scoped_lock lock{walkers.mutex};
    Xbyak::CodeGenerator& c = WalkerBuffer(walkers);
    const auto func = c.getCurr<PFN_SrtWalker>();
    c.db(ptr, size);
    c.ready();
    walkers.published_size.store(c.getSize(), std::memory_order_release);
    return func;
}

} // namespace Shader

#endif // ARCH_X86_64

namespace Shader::Optimization {

/// bbport: the bytecode walker's instructions, three words each (op, a, b). Pointers are 48 bits.
enum class SrtOp : u32 {
    End,
    PushPtrImm, ///< pointer stack <- pointer; pointer = load64(pointer + a * 4)
    PushPtrAcc, ///< the same at dword offset acc
    PopPtr,
    CopyImm, ///< flat[b] = load32(pointer + a * 4)
    CopyAcc, ///< flat[b] = load32(pointer + acc * 4)
    AccFlat, ///< acc = flat[a] (the user data registers, or a slot copied before)
    AccImm,  ///< acc = a
    Push,    ///< value stack <- acc
    Add,     ///< acc = op(popped, acc), to UMax
    Sub,
    Mul,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    UMin,
    UMax,
    Not, ///< acc = ~acc
    Bfe, ///< acc = bits of value from offset, count bits (value, offset popped; count = acc);
         ///< b: 1 when the count was an immediate (32 or more: all bits), as the x86 code
    Count,
};

namespace {

using IR::Opcode;
#ifdef ARCH_X86_64
using namespace Xbyak::util;
#endif

/// Value numbers of the pointer and offset arithmetic: equal numbers, equal values.
class ValueNumbering {
public:
    u32 Number(const IR::Value& value) {
        if (const auto it = numbers.find(value); it != numbers.end()) {
            return it->second;
        }
        IR::Inst* const inst = value.TryInst();
        if (!inst || !IsPure(inst->GetOpcode())) {
            return New(value);
        }
        Key key;
        key.reserve(2 + inst->NumArgs());
        key.push_back(static_cast<u32>(inst->GetOpcode()));
        key.push_back(inst->Flags<u32>());
        for (size_t i = 0; i < inst->NumArgs(); i++) {
            key.push_back(Number(inst->Arg(i)));
        }
        if (const auto it = by_key.find(key); it != by_key.end()) {
            numbers[value] = it->second;
            return it->second;
        }
        const u32 number = New(value);
        by_key.emplace(std::move(key), number);
        return number;
    }

private:
    using Key = boost::container::small_vector<u32, 8>;
    struct KeyHash {
        size_t operator()(const Key& key) const {
            u64 hash = 0;
            for (const u32 element : key) {
                hash = HashCombine(hash, u64(element));
            }
            return hash;
        }
    };

    static bool IsPure(Opcode op) {
        switch (op) {
        case Opcode::GetUserData:
        case Opcode::CompositeConstructU32x2:
        case Opcode::ReadConst:
        case Opcode::ReadConstBuffer:
        case Opcode::IAdd32:
        case Opcode::ISub32:
        case Opcode::IMul32:
        case Opcode::ShiftLeftLogical32:
        case Opcode::ShiftRightLogical32:
        case Opcode::BitwiseAnd32:
        case Opcode::BitwiseOr32:
        case Opcode::BitwiseXor32:
        case Opcode::BitwiseNot32:
        case Opcode::UMin32:
        case Opcode::UMax32:
        case Opcode::BitFieldUExtract:
            return true;
        default:
            return false;
        }
    }

    u32 New(const IR::Value& value) {
        const u32 number = next++;
        numbers[value] = number;
        return number;
    }

    std::unordered_map<IR::Value, u32> numbers;
    std::unordered_map<Key, u32, KeyHash> by_key;
    u32 next{};
};

/// Offsets read through one pointer, ordered: immediates by value, then instructions in
/// program order (blocks by address).
struct OffsetOrder {
    bool operator()(const IR::Value& a, const IR::Value& b) const {
        if (a.IsImmediate() != b.IsImmediate()) {
            return a.IsImmediate();
        }
        if (a.IsImmediate()) {
            return a.U32() < b.U32();
        }
        const IR::Inst* const a_inst = a.Inst();
        const IR::Inst* const b_inst = b.Inst();
        const IR::Block* const block = a_inst->GetParent();
        if (block != b_inst->GetParent()) {
            return block < b_inst->GetParent();
        }
        const auto& list = block->Instructions();
        for (auto it = list.iterator_to(*a_inst); it != list.end(); ++it) {
            if (&*it == b_inst) {
                return true;
            }
        }
        return false;
    }
};

using OffsetReads = boost::container::flat_map<IR::Value, IR::Inst*, OffsetOrder>;

u16 FlatbufOffset(const IR::Inst* inst) {
    switch (inst->GetOpcode()) {
    case Opcode::ReadConstBuffer:
        return inst->Flags<IR::BufferInstInfo>().flatbuf_off_dw;
    case Opcode::ReadConst:
        return inst->Flags<u16>();
    default:
        UNREACHABLE_MSG("Instruction not supported");
    }
}

void SetFlatbufOffset(IR::Inst* inst, u16 offset) {
    switch (inst->GetOpcode()) {
    case Opcode::ReadConstBuffer: {
        auto flags = inst->Flags<IR::BufferInstInfo>();
        flags.flatbuf_off_dw.Assign(offset);
        inst->SetFlags(flags);
        break;
    }
    case Opcode::ReadConst:
        inst->SetFlags(offset);
        break;
    default:
        UNREACHABLE_MSG("Instruction not supported");
    }
}

bool IsConstantMemoryValue(const IR::Value& value) {
    const IR::Inst* const inst = value.TryInst();
    if (!inst) {
        return false;
    }
    const Opcode op = inst->GetOpcode();
    return op == Opcode::GetUserData || op == Opcode::ReadConst || op == Opcode::ReadConstBuffer;
}

/// Operations a walker can recompute an offset with.
bool IsOffsetArithmetic(Opcode op) {
    switch (op) {
    case Opcode::GetUserData:
    case Opcode::ReadConst:
    case Opcode::ReadConstBuffer:
    case Opcode::IAdd32:
    case Opcode::ISub32:
    case Opcode::IMul32:
    case Opcode::ShiftLeftLogical32:
    case Opcode::ShiftRightLogical32:
    case Opcode::BitwiseAnd32:
    case Opcode::BitwiseOr32:
    case Opcode::BitwiseXor32:
    case Opcode::BitwiseNot32:
    case Opcode::UMin32:
    case Opcode::UMax32:
    case Opcode::BitFieldUExtract:
        return true;
    default:
        return false;
    }
}

#ifdef ARCH_X86_64
/// Builds one walker. Calling convention: rdi the user data (then the current table), rsi the
/// flat buffer; offsets are computed in r10d with ecx and edx as scratch.
class WalkerBuilder {
public:
    explicit WalkerBuilder(Xbyak::CodeGenerator& c_) : c{c_} {}

    ValueNumbering numbering;
    /// The reads through each pointer (by its value number's first instruction).
    std::unordered_map<IR::Inst*, OffsetReads> pointer_reads;
    /// User data pointers the walk starts from, by register.
    boost::container::small_flat_map<IR::ScalarReg, IR::Inst*, 1> roots;
    /// The first instruction of each value number.
    std::unordered_map<u32, IR::Inst*> first_of;
    u16 next_slot{};

    IR::Inst* Canonical(IR::Inst* inst) {
        return first_of.try_emplace(numbering.Number(IR::Value{inst}), inst).first->second;
    }

    /// Copies everything read through `table` (loaded from `table_off_dw` of the current table),
    /// then walks the tables among them.
    void Visit(const IR::Value& table_off_dw, IR::Inst* table) {
        if (table->GetOpcode() != Opcode::GetUserData && FlatbufOffset(table) == 0) {
            // The pointer itself was not copied.
            return;
        }
        if (!PushTable(table_off_dw)) {
            LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
            return;
        }
        OffsetReads* const reads = ReadsThrough(table);
        ASSERT(reads);
        // Contiguous in the guest table, contiguous in the flat buffer.
        for (const auto& [off_dw, read] : *reads) {
            if (off_dw.IsImmediate()) {
                c.mov(r10d, ptr[rdi + (off_dw.U32() << 2)]);
            } else {
                if (!Compute(r10d, off_dw)) {
                    LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
                    continue;
                }
                c.shl(r10d, 2);
                c.mov(r10d, r10d);
                c.mov(r10d, dword[rdi + r10]);
            }
            c.mov(ptr[rsi + (next_slot << 2)], r10d);
            SetFlatbufOffset(read, next_slot++);
        }
        for (const auto& [off_dw, read] : *reads) {
            if (ReadsThrough(read)) {
                Visit(off_dw, read);
            }
        }
        c.pop(rdi);
    }

private:
    OffsetReads* ReadsThrough(IR::Inst* inst) {
        const auto it = pointer_reads.find(inst);
        return it != pointer_reads.end() ? &it->second : nullptr;
    }

    /// Saves the current table and loads the 48-bit pointer at `off_dw` of it.
    bool PushTable(const IR::Value& off_dw) {
        c.push(rdi);
        if (off_dw.IsImmediate()) {
            c.mov(rdi, ptr[rdi + (off_dw.U32() << 2)]);
        } else {
            if (!Compute(r10d, off_dw)) {
                c.pop(rdi);
                return false;
            }
            c.shl(r10d, 2);
            c.mov(r10d, r10d);
            c.mov(rdi, ptr[rdi + r10]);
        }
        c.mov(r10, 0xFFFFFFFFFFFFULL);
        c.and_(rdi, r10);
        return true;
    }

    /// reg = a op b. `op_imm` takes an immediate, `op_mem` the other operand on the stack.
    template <typename OpImm, typename OpMem>
    bool Binary(Xbyak::Reg32 reg, const IR::Inst* inst, OpImm&& op_imm, OpMem&& op_mem) {
        const IR::Value a = inst->Arg(0);
        const IR::Value b = inst->Arg(1);
        if (a.IsImmediate()) {
            return Compute(reg, b) && (op_imm(reg, a.U32()), true);
        }
        if (b.IsImmediate()) {
            return Compute(reg, a) && (op_imm(reg, b.U32()), true);
        }
        if (!Compute(reg, a)) {
            return false;
        }
        c.push(reg.cvt64());
        if (!Compute(reg, b)) {
            c.add(rsp, 8);
            return false;
        }
        op_mem(reg, dword[rsp]);
        c.add(rsp, 8);
        return true;
    }

    /// reg = a shift b: `by_imm` shifts reg by an immediate, `by_cl` by cl.
    template <typename ByImm, typename ByCl>
    bool ShiftOp(Xbyak::Reg32 reg, const IR::Inst* inst, ByImm&& by_imm, ByCl&& by_cl) {
        const IR::Value a = inst->Arg(0);
        const IR::Value b = inst->Arg(1);
        if (b.IsImmediate()) {
            return Compute(reg, a) && (by_imm(reg, b.U32()), true);
        }
        if (a.IsImmediate()) {
            if (!Compute(reg, b)) {
                return false;
            }
            c.mov(ecx, reg);
            c.mov(reg, a.U32());
            by_cl(reg);
            return true;
        }
        if (!Compute(reg, b)) {
            return false;
        }
        c.push(reg.cvt64());
        if (!Compute(reg, a)) {
            c.add(rsp, 8);
            return false;
        }
        c.pop(rcx);
        by_cl(reg);
        return true;
    }

    bool BitFieldUExtract(Xbyak::Reg32 reg, const IR::Inst* inst) {
        const IR::Value base = inst->Arg(0);
        const IR::Value offset = inst->Arg(1);
        const IR::Value count = inst->Arg(2);
        if (base.IsImmediate()) {
            c.mov(reg, base.U32());
        } else if (!Compute(reg, base)) {
            return false;
        }
        bool on_stack = false;
        if (offset.IsImmediate()) {
            c.shr(reg, offset.U32());
        } else {
            c.push(reg.cvt64());
            on_stack = true;
            if (!Compute(reg, offset)) {
                c.add(rsp, 8);
                return false;
            }
            c.mov(ecx, reg);
            c.shr(dword[rsp], cl);
        }
        if (count.IsImmediate()) {
            if (on_stack) {
                c.pop(reg.cvt64());
            }
            if (count.U32() < 32) {
                c.and_(reg, (1U << count.U32()) - 1);
            }
            return true;
        }
        if (!on_stack) {
            c.push(reg.cvt64());
        }
        if (!Compute(reg, count)) {
            c.add(rsp, 8);
            return false;
        }
        c.mov(ecx, reg);
        c.mov(edx, 1);
        c.shl(edx, cl);
        c.dec(edx);
        c.and_(dword[rsp], edx);
        c.pop(reg.cvt64());
        return true;
    }

    /// reg = the value of `off_dw` at draw time (from user data and slots already copied).
    bool Compute(Xbyak::Reg32 reg, const IR::Value& off_dw) {
        ASSERT(reg != ecx && reg != edx);
        IR::Inst* const inst = off_dw.Inst();
        switch (inst->GetOpcode()) {
        case Opcode::GetUserData:
            c.mov(reg, ptr[rsi + (static_cast<u32>(inst->Arg(0).ScalarReg()) << 2)]);
            return true;
        case Opcode::ReadConst:
        case Opcode::ReadConstBuffer: {
            const u16 slot = FlatbufOffset(Canonical(inst));
            if (slot == 0) {
                return false;
            }
            c.mov(reg, ptr[rsi + (slot << 2)]);
            return true;
        }
        default:
            break;
        }
        if (!IsOffsetArithmetic(inst->GetOpcode())) {
            LOG_ERROR(Render_Recompiler, "Unexpected instruction for offset computation, {}",
                      magic_enum::enum_name(inst->GetOpcode()));
            return false;
        }
        ASSERT(!inst->AreAllArgsImmediates());
        switch (inst->GetOpcode()) {
        case Opcode::IAdd32:
            return Binary(reg, inst, [&](auto r, u32 imm) { c.add(r, imm); },
                          [&](auto r, const auto& m) { c.add(r, m); });
        case Opcode::ISub32: {
            const IR::Value a = inst->Arg(0);
            const IR::Value b = inst->Arg(1);
            if (b.IsImmediate()) {
                return Compute(reg, a) && (c.sub(reg, b.U32()), true);
            }
            // a - b = -b + a
            if (a.IsImmediate()) {
                return Compute(reg, b) && (c.neg(reg), c.add(reg, a.U32()), true);
            }
            if (!Compute(reg, a)) {
                return false;
            }
            c.push(reg.cvt64());
            if (!Compute(reg, b)) {
                c.add(rsp, 8);
                return false;
            }
            c.neg(reg);
            c.add(reg, dword[rsp]);
            c.add(rsp, 8);
            return true;
        }
        case Opcode::IMul32:
            return Binary(reg, inst, [&](auto r, u32 imm) { c.imul(r, r, imm); },
                          [&](auto r, const auto& m) { c.imul(r, m); });
        case Opcode::BitwiseAnd32:
            return Binary(reg, inst, [&](auto r, u32 imm) { c.and_(r, imm); },
                          [&](auto r, const auto& m) { c.and_(r, m); });
        case Opcode::BitwiseOr32:
            return Binary(reg, inst, [&](auto r, u32 imm) { c.or_(r, imm); },
                          [&](auto r, const auto& m) { c.or_(r, m); });
        case Opcode::BitwiseXor32:
            return Binary(reg, inst, [&](auto r, u32 imm) { c.xor_(r, imm); },
                          [&](auto r, const auto& m) { c.xor_(r, m); });
        case Opcode::UMin32:
            return Binary(
                reg, inst,
                [&](auto r, u32 imm) {
                    c.mov(ecx, imm);
                    c.cmp(r, ecx);
                    c.cmova(r, ecx);
                },
                [&](auto r, const auto& m) {
                    c.cmp(r, m);
                    c.cmova(r, m);
                });
        case Opcode::UMax32:
            return Binary(
                reg, inst,
                [&](auto r, u32 imm) {
                    c.mov(ecx, imm);
                    c.cmp(r, ecx);
                    c.cmovb(r, ecx);
                },
                [&](auto r, const auto& m) {
                    c.cmp(r, m);
                    c.cmovb(r, m);
                });
        case Opcode::ShiftLeftLogical32:
            return ShiftOp(reg, inst, [&](auto r, u32 imm) { c.shl(r, imm); },
                           [&](auto r) { c.shl(r, cl); });
        case Opcode::ShiftRightLogical32:
            return ShiftOp(reg, inst, [&](auto r, u32 imm) { c.shr(r, imm); },
                           [&](auto r) { c.shr(r, cl); });
        case Opcode::BitwiseNot32:
            return Compute(reg, inst->Arg(0)) && (c.not_(reg), true);
        case Opcode::BitFieldUExtract:
            return BitFieldUExtract(reg, inst);
        default:
            UNREACHABLE();
        }
    }

    Xbyak::CodeGenerator& c;
};
#else
/// bbport: builds one walker as bytecode (SrtOp): the x86 builder's walk, step for step, so the
/// slots are the same. Offsets are computed in an accumulator with a value stack for the first
/// operand; the tables on a pointer stack.
class WalkerBuilder {
public:
    ValueNumbering numbering;
    std::unordered_map<IR::Inst*, OffsetReads> pointer_reads;
    boost::container::small_flat_map<IR::ScalarReg, IR::Inst*, 1> roots;
    std::unordered_map<u32, IR::Inst*> first_of;
    u16 next_slot{};
    std::vector<u32> code;

    IR::Inst* Canonical(IR::Inst* inst) {
        return first_of.try_emplace(numbering.Number(IR::Value{inst}), inst).first->second;
    }

    void Visit(const IR::Value& table_off_dw, IR::Inst* table) {
        if (table->GetOpcode() != Opcode::GetUserData && FlatbufOffset(table) == 0) {
            return;
        }
        // A failed offset leaves nothing behind.
        std::vector<u32> push;
        if (table_off_dw.IsImmediate()) {
            Emit(push, SrtOp::PushPtrImm, table_off_dw.U32());
        } else if (Compute(push, table_off_dw)) {
            Emit(push, SrtOp::PushPtrAcc);
        } else {
            LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
            return;
        }
        code.insert(code.end(), push.begin(), push.end());
        OffsetReads* const reads = ReadsThrough(table);
        ASSERT(reads);
        for (const auto& [off_dw, read] : *reads) {
            if (off_dw.IsImmediate()) {
                Emit(code, SrtOp::CopyImm, off_dw.U32(), next_slot);
            } else {
                std::vector<u32> offset;
                if (!Compute(offset, off_dw)) {
                    LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
                    continue;
                }
                code.insert(code.end(), offset.begin(), offset.end());
                Emit(code, SrtOp::CopyAcc, 0, next_slot);
            }
            SetFlatbufOffset(read, next_slot++);
        }
        for (const auto& [off_dw, read] : *reads) {
            if (ReadsThrough(read)) {
                Visit(off_dw, read);
            }
        }
        Emit(code, SrtOp::PopPtr);
    }

private:
    static void Emit(std::vector<u32>& out, SrtOp op, u32 a = 0, u32 b = 0) {
        out.push_back(static_cast<u32>(op));
        out.push_back(a);
        out.push_back(b);
    }

    OffsetReads* ReadsThrough(IR::Inst* inst) {
        const auto it = pointer_reads.find(inst);
        return it != pointer_reads.end() ? &it->second : nullptr;
    }

    /// acc = the value (an immediate, or computed).
    bool Value(std::vector<u32>& out, const IR::Value& value) {
        if (value.IsImmediate()) {
            Emit(out, SrtOp::AccImm, value.U32());
            return true;
        }
        return Compute(out, value);
    }

    /// acc = op(a, b): a waits on the value stack while b is computed.
    bool Binary(std::vector<u32>& out, const IR::Inst* inst, SrtOp op) {
        if (!Value(out, inst->Arg(0))) {
            return false;
        }
        Emit(out, SrtOp::Push);
        if (!Value(out, inst->Arg(1))) {
            return false;
        }
        Emit(out, op);
        return true;
    }

    /// acc = the value of `off_dw` at draw time (from user data and slots already copied).
    bool Compute(std::vector<u32>& out, const IR::Value& off_dw) {
        IR::Inst* const inst = off_dw.Inst();
        switch (inst->GetOpcode()) {
        case Opcode::GetUserData:
            Emit(out, SrtOp::AccFlat, static_cast<u32>(inst->Arg(0).ScalarReg()));
            return true;
        case Opcode::ReadConst:
        case Opcode::ReadConstBuffer: {
            const u16 slot = FlatbufOffset(Canonical(inst));
            if (slot == 0) {
                return false;
            }
            Emit(out, SrtOp::AccFlat, slot);
            return true;
        }
        default:
            break;
        }
        if (!IsOffsetArithmetic(inst->GetOpcode())) {
            LOG_ERROR(Render_Recompiler, "Unexpected instruction for offset computation, {}",
                      magic_enum::enum_name(inst->GetOpcode()));
            return false;
        }
        ASSERT(!inst->AreAllArgsImmediates());
        switch (inst->GetOpcode()) {
        case Opcode::IAdd32:
            return Binary(out, inst, SrtOp::Add);
        case Opcode::ISub32:
            return Binary(out, inst, SrtOp::Sub);
        case Opcode::IMul32:
            return Binary(out, inst, SrtOp::Mul);
        case Opcode::BitwiseAnd32:
            return Binary(out, inst, SrtOp::And);
        case Opcode::BitwiseOr32:
            return Binary(out, inst, SrtOp::Or);
        case Opcode::BitwiseXor32:
            return Binary(out, inst, SrtOp::Xor);
        case Opcode::UMin32:
            return Binary(out, inst, SrtOp::UMin);
        case Opcode::UMax32:
            return Binary(out, inst, SrtOp::UMax);
        case Opcode::ShiftLeftLogical32:
            return Binary(out, inst, SrtOp::Shl);
        case Opcode::ShiftRightLogical32:
            return Binary(out, inst, SrtOp::Shr);
        case Opcode::BitwiseNot32:
            if (!Value(out, inst->Arg(0))) {
                return false;
            }
            Emit(out, SrtOp::Not);
            return true;
        case Opcode::BitFieldUExtract:
            for (u32 arg = 0; arg < 3; ++arg) {
                if (!Value(out, inst->Arg(arg))) {
                    return false;
                }
                if (arg < 2) {
                    Emit(out, SrtOp::Push);
                }
            }
            Emit(out, SrtOp::Bfe, 0, inst->Arg(2).IsImmediate() ? 1 : 0);
            return true;
        default:
            UNREACHABLE();
        }
    }
};
#endif // ARCH_X86_64

/// s_add_u32 s12, s2, off; s_addc_u32 s13, s3, 0; s_load_dword s16, s[12:13], 0: the pointer
/// becomes s[2:3] and the byte offset joins the read's dword offset.
void FoldPointerAdd(IR::Inst& inst) {
    if (inst.GetOpcode() != Opcode::ReadConst) {
        return;
    }
    IR::Inst* const pointer = inst.Arg(0).Inst();
    const IR::Inst* const lo = pointer->Arg(0).TryInst();
    const IR::Inst* const hi = pointer->Arg(1).TryInst();
    if (!lo || !hi || lo->GetOpcode() != Opcode::IAdd32 || hi->GetOpcode() != Opcode::IAdd32) {
        return;
    }
    IR::Value base;
    IR::Value byte_offset;
    if (IsConstantMemoryValue(lo->Arg(0))) {
        base = lo->Arg(0);
        byte_offset = lo->Arg(1);
    } else if (IsConstantMemoryValue(lo->Arg(1))) {
        base = lo->Arg(1);
        byte_offset = lo->Arg(0);
    } else {
        return;
    }
    // The high half adds the carry: select(carry, 1, 0).
    const IR::Inst* const carry = hi->Arg(1).TryInst();
    if (!carry || carry->GetOpcode() != Opcode::SelectU32 || !carry->Arg(1).IsImmediate() ||
        !carry->Arg(2).IsImmediate() || carry->Arg(1).U32() != 1 || carry->Arg(2).U32() != 0) {
        return;
    }
    pointer->SetArg(0, base);
    pointer->SetArg(1, hi->Arg(0));
    for (const auto& [user, operand] : pointer->Uses()) {
        ASSERT(user->GetOpcode() == Opcode::ReadConst && operand == 0);
        IR::IREmitter ir{*user->GetParent(), IR::Block::InstructionList::s_iterator_to(*user)};
        const IR::U32 offset{user->Arg(1)};
        const IR::U32 add = byte_offset.IsImmediate()
                                ? ir.Imm32(byte_offset.U32() >> 2u)
                                : IR::U32{ir.ShiftRightLogical(IR::U32{byte_offset}, ir.Imm32(2u))};
        if (offset.IsImmediate() && add.IsImmediate()) {
            user->SetArg(1, ir.Imm32(offset.U32() + add.U32()));
        } else {
            user->SetArg(1, ir.IAdd(offset, add));
        }
    }
}

using InstVector = boost::container::small_vector<IR::Inst*, 32>;

/// The reads to flatten, in reverse post order: ReadConsts and sharp-producing constant buffer
/// reads, and the other constant buffer reads their offsets are computed from.
InstVector CollectReads(IR::Program& program) {
    InstVector reads;
    InstVector visited;
    std::queue<IR::Inst*> queue;
    const auto enqueue = [&](IR::Inst* inst) {
        if (std::ranges::find(visited, inst) == visited.end()) {
            visited.push_back(inst);
            queue.push(inst);
        }
    };
    for (auto it = program.post_order_blocks.rbegin(); it != program.post_order_blocks.rend();
         ++it) {
        for (IR::Inst& inst : **it) {
            const Opcode op = inst.GetOpcode();
            if (op != Opcode::ReadConst && op != Opcode::ReadConstBuffer) {
                continue;
            }
            if (op == Opcode::ReadConstBuffer && !inst.Flags<IR::BufferInstInfo>().sharp_source) {
                continue;
            }
            FoldPointerAdd(inst);
            reads.push_back(&inst);
            const IR::Value offset = inst.Arg(1);
            if (offset.IsImmediate()) {
                continue;
            }
            queue.push(offset.Inst());
            while (!queue.empty()) {
                IR::Inst* const term = queue.front();
                queue.pop();
                const Opcode term_op = term->GetOpcode();
                if (term_op == Opcode::ReadConstBuffer) {
                    if (!term->Flags<IR::BufferInstInfo>().sharp_source.Value()) {
                        reads.push_back(term);
                        if (!term->Arg(1).IsImmediate()) {
                            enqueue(term->Arg(1).Inst());
                        }
                    }
                    continue;
                }
                if (term_op == Opcode::GetUserData || term_op == Opcode::ReadConst ||
                    !IsOffsetArithmetic(term_op)) {
                    continue;
                }
                for (size_t arg = term->NumArgs(); arg--;) {
                    if (!term->Arg(arg).IsImmediate()) {
                        enqueue(term->Arg(arg).Inst());
                    }
                }
            }
        }
    }
    return reads;
}

} // namespace

void FlattenExtendedUserdataPass(IR::Program& program) {
#ifdef ARCH_X86_64
    WalkerCode& walkers = Walkers();
    std::scoped_lock lock{walkers.mutex};
    WalkerBuilder builder{walkers.code};
#else
    WalkerBuilder builder;
#endif

    const InstVector reads = CollectReads(program);
    for (IR::Inst* const read : reads) {
        if (builder.Canonical(read) != read) {
            continue;
        }
        const IR::Inst* const pointer = read->Arg(0).Inst();
        const IR::Inst* const lo = pointer->Arg(0).TryInst();
        if (lo && lo->GetOpcode() == Opcode::ReadFirstLane) {
            continue;
        }
        ASSERT_MSG(IsConstantMemoryValue(pointer->Arg(0)),
                   "ReadConst base low not from constant memory");
        ASSERT_MSG(IsConstantMemoryValue(pointer->Arg(1)),
                   "ReadConst base high not from constant memory");
        IR::Inst* const table = builder.Canonical(pointer->Arg(0).Inst());
        builder.pointer_reads[table][read->Arg(1)] = read;
        if (table->GetOpcode() == Opcode::GetUserData) {
            builder.roots[table->Arg(0).ScalarReg()] = table;
        }
    }

    Info& info = program.info;
#ifdef ARCH_X86_64
    if (!builder.roots.empty()) {
        Xbyak::CodeGenerator& c = WalkerBuffer(walkers);
        info.srt_info.walker_func = c.getCurr<PFN_SrtWalker>();
        builder.next_slot = NUM_USER_DATA_REGS;
        ASSERT(builder.next_slot == info.srt_info.flattened_bufsize_dw);
        for (const auto& [sgpr, root] : builder.roots) {
            builder.Visit(IR::Value(static_cast<u32>(sgpr)), root);
        }
        c.ret();
        c.ready();
        walkers.published_size.store(c.getSize(), std::memory_order_release);
        const auto* const code = reinterpret_cast<const u8*>(info.srt_info.walker_func);
        info.srt_info.walker_func_size = c.getCurr() - code;
        if (EmulatorSettings.IsDumpShaders()) {
            DumpWalker(info, code, info.srt_info.walker_func_size);
        }
        info.srt_info.flattened_bufsize_dw = builder.next_slot;
    }
#else
    if (!builder.roots.empty()) {
        builder.next_slot = NUM_USER_DATA_REGS;
        ASSERT(builder.next_slot == info.srt_info.flattened_bufsize_dw);
        for (const auto& [sgpr, root] : builder.roots) {
            builder.Visit(IR::Value(static_cast<u32>(sgpr)), root);
        }
        builder.code.push_back(static_cast<u32>(SrtOp::End));
        builder.code.push_back(0);
        builder.code.push_back(0);
        info.srt_info.walker_func = reinterpret_cast<PFN_SrtWalker>(const_cast<u32*>(
            RegisterSrtBytecode(builder.code.data(), builder.code.size())));
        info.srt_info.walker_func_size = builder.code.size() * sizeof(u32);
        info.srt_info.flattened_bufsize_dw = builder.next_slot;
    }
#endif

    // Duplicates take the slot of their value number's first read.
    for (IR::Inst* const read : reads) {
        ASSERT(builder.first_of.contains(builder.numbering.Number(IR::Value{read})));
        SetFlatbufOffset(read, FlatbufOffset(builder.Canonical(read)));
    }
    info.RefreshFlatBuf();
}

} // namespace Shader::Optimization

#ifndef ARCH_X86_64

namespace Shader {

namespace {

using Optimization::SrtOp;

std::mutex bytecode_mutex;
std::vector<std::unique_ptr<u32[]>> bytecode; // kept for the run, as the x86 walkers are

/// Walker bytecode is only ever data: unknown operations, slots past the flat buffer's 16-bit
/// range or a program without its end are refused (a damaged cache entry).
bool ValidBytecode(const u32* words, size_t count) {
    if (count == 0 || count % 3 != 0 || words[count - 3] != u32(SrtOp::End)) {
        return false;
    }
    for (size_t i = 0; i < count; i += 3) {
        const u32 op = words[i];
        if (op >= u32(SrtOp::Count)) {
            return false;
        }
        if ((op == u32(SrtOp::CopyImm) || op == u32(SrtOp::CopyAcc)) && words[i + 2] > 0xFFFF) {
            return false;
        }
        if (op == u32(SrtOp::AccFlat) && words[i + 1] > 0xFFFF) {
            return false;
        }
    }
    return true;
}

/// A table pointer can be stale or garbage: an unreadable load gives 0, as the x86 walker's
/// fault handler makes it (that one patches the load for good; here each walk checks).
template <typename T>
T LoadGuest(u64 address) {
    if (srt_guest_translate) {
        address = srt_guest_translate(address);
    }
    if (srt_guest_readable && !srt_guest_readable(address, sizeof(T))) {
        return 0;
    }
    T value;
    std::memcpy(&value, reinterpret_cast<const void*>(address), sizeof(T));
    return value;
}

void InterpretSrt(const u32* program, const u32* user_data, u32* flat, u32 flat_size) {
    constexpr u64 PointerMask = 0xFFFFFFFFFFFFULL;
    constexpr u32 StackSize = 64;
    // The first level reads the user data (host memory); deeper ones guest memory.
    u64 pointer = reinterpret_cast<u64>(user_data);
    u64 pointers[StackSize];
    u32 values[StackSize];
    u32 num_pointers = 0, num_values = 0, acc = 0;
    const auto pop = [&] { return num_values ? values[--num_values] : 0u; };
    for (const u32* code = program;; code += 3) {
        switch (static_cast<SrtOp>(code[0])) {
        case SrtOp::End:
            return;
        case SrtOp::PushPtrImm:
        case SrtOp::PushPtrAcc: {
            if (num_pointers == StackSize) {
                return;
            }
            const u64 offset = code[0] == u32(SrtOp::PushPtrImm) ? u64(code[1]) << 2
                                                                 : u64(u32(acc << 2));
            pointers[num_pointers++] = pointer;
            if (num_pointers == 1) {
                // The user data registers: a pointer in them is two of the 16.
                if (offset > (NUM_USER_DATA_REGS - 2) * 4) {
                    return;
                }
                u64 value;
                std::memcpy(&value, reinterpret_cast<const u8*>(pointer) + offset, sizeof(value));
                pointer = value & PointerMask;
            } else {
                pointer = LoadGuest<u64>(pointer + offset) & PointerMask;
            }
            break;
        }
        case SrtOp::PopPtr:
            pointer = num_pointers ? pointers[--num_pointers] : pointer;
            break;
        case SrtOp::CopyImm:
        case SrtOp::CopyAcc: {
            if (code[2] >= flat_size) {
                return;
            }
            const u64 offset = code[0] == u32(SrtOp::CopyImm) ? u64(code[1]) << 2
                                                              : u64(u32(acc << 2));
            flat[code[2]] = LoadGuest<u32>(pointer + offset);
            break;
        }
        case SrtOp::AccFlat:
            if (code[1] >= flat_size) {
                return;
            }
            acc = flat[code[1]];
            break;
        case SrtOp::AccImm:
            acc = code[1];
            break;
        case SrtOp::Push:
            if (num_values == StackSize) {
                return;
            }
            values[num_values++] = acc;
            break;
        case SrtOp::Add:
            acc = pop() + acc;
            break;
        case SrtOp::Sub:
            acc = pop() - acc;
            break;
        case SrtOp::Mul:
            acc = pop() * acc;
            break;
        case SrtOp::Shl:
            acc = pop() << (acc & 31);
            break;
        case SrtOp::Shr:
            acc = pop() >> (acc & 31);
            break;
        case SrtOp::And:
            acc = pop() & acc;
            break;
        case SrtOp::Or:
            acc = pop() | acc;
            break;
        case SrtOp::Xor:
            acc = pop() ^ acc;
            break;
        case SrtOp::UMin:
            acc = std::min(pop(), acc);
            break;
        case SrtOp::UMax:
            acc = std::max(pop(), acc);
            break;
        case SrtOp::Not:
            acc = ~acc;
            break;
        case SrtOp::Bfe: {
            const u32 count = acc, offset = pop(), value = pop();
            // As the x86 code: an immediate count of 32 or more keeps every bit; a computed one
            // is taken modulo 32 (shl by cl).
            const u32 mask = code[2] ? (count >= 32 ? ~0u : (1u << count) - 1)
                                     : (1u << (count & 31)) - 1;
            acc = (value >> (offset & 31)) & mask;
            break;
        }
        default:
            return; // ValidBytecode refused anything else
        }
    }
}

} // namespace

const u32* RegisterSrtBytecode(const u32* words, size_t count) {
    ASSERT(ValidBytecode(words, count));
    auto copy = std::make_unique<u32[]>(count);
    std::memcpy(copy.get(), words, count * sizeof(u32));
    std::scoped_lock lock{bytecode_mutex};
    bytecode.push_back(std::move(copy));
    return bytecode.back().get();
}

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    // Bytecode from the pipeline cache (its own per architecture, blob_store.cpp): only
    // RunSrtWalker uses the pointer.
    std::vector<u32> words(size / sizeof(u32));
    std::memcpy(words.data(), ptr, words.size() * sizeof(u32));
    if (size % (3 * sizeof(u32)) != 0 || !ValidBytecode(words.data(), words.size())) {
        throw Serialization::CorruptData{"damaged SRT walker in the shader cache"};
    }
    return reinterpret_cast<PFN_SrtWalker>(
        const_cast<u32*>(RegisterSrtBytecode(words.data(), words.size())));
}

bool (*srt_guest_readable)(u64 address, u64 size) = nullptr;
u64 (*srt_guest_translate)(u64 address) = nullptr;

void RunSrtWalker(const PersistentSrtInfo& srt, const u32* user_data, u32* flat) {
    // A mapping removed between the check and the load faults: the walk stops there, as the x86
    // walker's patched load reads 0 (bb-gpu's handler jumps back here, runtime_fault_recover).
    sigjmp_buf recover;
    sigjmp_buf* const outer = runtime_fault_recover;
    if (sigsetjmp(recover, 0)) {
        runtime_fault_recover = outer;
        return;
    }
    runtime_fault_recover = &recover;
    InterpretSrt(reinterpret_cast<const u32*>(srt.walker_func), user_data, flat,
                 srt.flattened_bufsize_dw);
    runtime_fault_recover = outer;
}

} // namespace Shader

#endif // !ARCH_X86_64
