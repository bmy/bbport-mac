// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <unordered_map>
#include <boost/container/flat_map.hpp>
#include <queue>
#include "common/arch.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>
#ifdef ARCH_X86_64
#include <xbyak/xbyak.h>
#include <xbyak/xbyak_util.h>
#endif
#include "common/decoder.h"
#include "common/io_file.h"
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/signal_context.h"
#include "core/emulator_settings.h"
#include "core/signals.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/opcodes.h"
#include "shader_recompiler/ir/passes/srt.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/ir/reg.h"
#include "shader_recompiler/ir/srt_gvn_table.h"
#include "shader_recompiler/ir/value.h"

namespace Shader::Optimization {
/// bbport: the portable SRT walker's instructions, three words each (op, a, b).
enum class SrtOp : u32 {
    End,
    PushPtrImm, ///< pointer stack <- pointer; pointer = load64(pointer + a * 4) & 48 bits
    PushPtrAcc, ///< the same at dword offset acc
    PopPtr,
    CopyImm,    ///< flat[b] = load32(pointer + a * 4)
    CopyAcc,    ///< flat[b] = load32(pointer + acc * 4)
    AccFlat,    ///< acc = flat[a] (user data, or a value flattened before)
    AccImm,     ///< acc = a
    Push,       ///< value stack <- acc
    Add, Sub, Mul, Shl, Shr, And, Or, Xor, UMin, UMax, ///< acc = op(popped, acc)
    Not,        ///< acc = ~acc
    Bfe,        ///< acc = bitfield of value at offset, count (value, offset popped; count = acc)
};
} // namespace Shader::Optimization

#ifdef ARCH_X86_64

using namespace Xbyak::util;

static Xbyak::CodeGenerator g_srt_codegen(32_MB);
static const u8* g_srt_codegen_start = nullptr;

namespace Shader {

PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    const auto func_addr = (PFN_SrtWalker)g_srt_codegen.getCurr();
    g_srt_codegen.db(ptr, size);
    g_srt_codegen.ready();
    return func_addr;
}

} // namespace Shader

namespace {

static void DumpSrtProgram(const Shader::Info& info, const u8* code, size_t codesize) {
    using namespace Common::FS;

    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}_{:#018x}.srtprogram.txt", info.hw_stage, info.pgm_hash);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create, FileType::TextFile};

    u64 address = reinterpret_cast<u64>(code);
    u64 code_end = address + codesize;
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    ZyanStatus status = ZYAN_STATUS_SUCCESS;
    while (address < code_end && ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                                     instruction, operands, reinterpret_cast<void*>(address)))) {
        std::string s =
            Common::Decoder::Instance()->disassembleInst(instruction, operands, address);
        s += "\n";
        file.WriteString(s);
        address += instruction.length;
    }
}

static bool SrtWalkerSignalHandler(void* context, void* fault_address) {
    // Only handle if the fault address is within the SRT code range
    const u8* code_start = g_srt_codegen_start;
    const u8* code_end = code_start + g_srt_codegen.getSize();
    const void* code = Common::GetRip(context);
    if (code < code_start || code >= code_end) {
        return false; // Not in SRT code range
    }

    // Patch instruction to zero register
    ZydisDecodedInstruction instruction;
    ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
    ZyanStatus status = Common::Decoder::Instance()->decodeInstruction(instruction, operands,
                                                                       const_cast<void*>(code), 15);

    ASSERT(ZYAN_SUCCESS(status) && instruction.mnemonic == ZYDIS_MNEMONIC_MOV &&
           operands[0].type == ZYDIS_OPERAND_TYPE_REGISTER &&
           operands[1].type == ZYDIS_OPERAND_TYPE_MEMORY);

    size_t len = instruction.length;
    const size_t patch_size = 3;
    u8* code_patch = const_cast<u8*>(reinterpret_cast<const u8*>(code));

    // We can only encounter rdi or r10d as the first operand in a
    // fault memory access for SRT walker.
    switch (operands[0].reg.value) {
    case ZYDIS_REGISTER_RDI:
        // mov rdi, [rdi + (off_dw << 2)] -> xor rdi, rdi
        code_patch[0] = 0x48;
        code_patch[1] = 0x31;
        code_patch[2] = 0xFF;
        break;
    case ZYDIS_REGISTER_R10D:
        // mov r10d, [rdi + (off_dw << 2)] -> xor r10d, r10d
        code_patch[0] = 0x45;
        code_patch[1] = 0x31;
        code_patch[2] = 0xD2;
        break;
    default:
        UNREACHABLE_MSG("Unsupported register for SRT walker patch");
        return false;
    }

    // Fill nops
    memset(code_patch + patch_size, 0x90, len - patch_size);

    LOG_WARNING(Render_Recompiler, "Patched SRT walker at {}, fault address {}", code,
                fault_address);

    return true;
}

#else
// bbport: the portable SRT walker (bytecode run by RunSrtWalker) needs none of the above.
namespace {
#endif // ARCH_X86_64

using namespace Shader;

struct PassInfo {
    struct PointerListCompare {
        bool operator()(const IR::Value& a, const IR::Value& b) const {
            // Keep order with the following criteria:
            // 1. If immediate, sort by immediate value
            // 2. If not immediate, sort by instruction order
            // 3. Immediates come first, then instructions
            if (a.IsImmediate() && b.IsImmediate()) {
                return a.U32() < b.U32();
            } else if (!a.IsImmediate() && !b.IsImmediate()) {
                auto a_inst = a.Inst();
                auto b_inst = b.Inst();
                auto a_block = a_inst->GetParent();
                auto b_block = b_inst->GetParent();
                if (a_block != b_block) {
                    return a_block < b_block; // Sort by block address
                }

                auto it = a_block->Instructions().iterator_to(*a_inst);
                auto end = a_block->Instructions().end();
                for (; it != end; ++it) {
                    if (&*it == b_inst) {
                        return true;
                    }
                }
                return false;
            } else {
                return a.IsImmediate(); // Immediates come first
            }
        }
    };

    // map offset to inst
    using PtrUserList =
        boost::container::flat_map<IR::Value, Shader::IR::Inst*, PointerListCompare>;

    Optimization::SrtGvnTable gvn_table;
    // keys are GetUserData, ReadConst or ReadConstBuffer instructions that are used as pointers
    std::unordered_map<IR::Inst*, PtrUserList> pointer_uses;
    // GetUserData instructions corresponding to sgpr_base of SRT roots
    boost::container::small_flat_map<IR::ScalarReg, IR::Inst*, 1> srt_roots;

    // pick a single inst for a given value number
    std::unordered_map<u32, IR::Inst*> vn_to_inst;

    // Bumped during codegen to assign offsets to readconsts
    u16 dst_off_dw;

    PtrUserList* GetUsesAsPointer(IR::Inst* inst) {
        auto it = pointer_uses.find(inst);
        if (it != pointer_uses.end()) {
            return &it->second;
        }
        return nullptr;
    }

    // Return a single instruction that this instruction is identical to, according
    // to value number
    // The "original" is arbitrary. Here it's the first instruction found for a given value number
    IR::Inst* DeduplicateInstruction(IR::Inst* inst) {
        auto it = vn_to_inst.try_emplace(gvn_table.GetValueNumber(inst), inst);
        return it.first->second;
    }
};
} // namespace

namespace Shader::Optimization {

namespace {

static inline u16 GetFlatbufOffset(const IR::Inst* inst) {
    if (inst->GetOpcode() == IR::Opcode::ReadConstBuffer) {
        auto inst_info = inst->Flags<IR::BufferInstInfo>();
        return inst_info.flatbuf_off_dw;
    }
    if (inst->GetOpcode() == IR::Opcode::ReadConst) {
        return inst->Flags<u16>();
    }
    UNREACHABLE_MSG("Instruction not supported");
}

static inline void SetFlatbufOffset(IR::Inst* inst, u16 offset) {
    if (inst->GetOpcode() == IR::Opcode::ReadConstBuffer) {
        auto inst_info = inst->Flags<IR::BufferInstInfo>();
        inst_info.flatbuf_off_dw.Assign(offset);
        inst->SetFlags(inst_info);
        return;
    }
    if (inst->GetOpcode() == IR::Opcode::ReadConst) {
        inst->SetFlags(offset);
        return;
    }
    UNREACHABLE_MSG("Instruction not supported");
}

#ifdef ARCH_X86_64
static bool ComputeOffset(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                          const IR::Value& off_dw);

#define ABORT_ON_FAILURE(expr)                                                                     \
    if (!(expr)) {                                                                                 \
        return false;                                                                              \
    }

#define POP_ABORT_ON_FAILURE(expr)                                                                 \
    if (!(expr)) {                                                                                 \
        c.add(rsp, 8);                                                                             \
        return false;                                                                              \
    }

static bool EmitComputeOffsetIAdd32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.add(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.add(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.add(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetISub32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.neg(reg);
        c.add(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.sub(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.neg(reg);
        c.add(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetIMul32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.imul(reg, reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.imul(reg, reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.imul(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetShiftLeftLogical32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                                PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.shl(reg, inst->Arg(0).U32());
        c.mov(ecx, reg);
        c.mov(reg, inst->Arg(0).U32());
        c.shl(reg, cl);
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.shl(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.pop(rcx);
        c.shl(reg, cl);
    }
    return true;
}

static bool EmitComputeOffsetShiftRightLogical32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                                 PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.mov(ecx, reg);
        c.mov(reg, inst->Arg(0).U32());
        c.shr(reg, cl);
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.shr(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.pop(rcx);
        c.shr(reg, cl);
    }
    return true;
}

static bool EmitComputeOffsetBitwiseAnd32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                          PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.and_(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.and_(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.and_(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetBitwiseOr32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                         PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.or_(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.or_(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.or_(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetBitwiseXor32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                          PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.xor_(reg, inst->Arg(0).U32());
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.xor_(reg, inst->Arg(1).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.xor_(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetBitwiseNot32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                          PassInfo& pass_info, IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
    c.not_(reg);
    return true;
}

static bool EmitComputeOffsetUMin32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.mov(ecx, inst->Arg(0).U32());
        c.cmp(reg, ecx);
        c.cmova(reg, ecx);
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.mov(ecx, inst->Arg(1).U32());
        c.cmp(reg, ecx);
        c.cmova(reg, ecx);
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.cmp(reg, dword[rsp]);
        c.cmova(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetUMax32(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                                    IR::Inst* inst) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.mov(ecx, inst->Arg(0).U32());
        c.cmp(reg, ecx);
        c.cmovb(reg, ecx);
    } else if (inst->Arg(1).IsImmediate()) {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.mov(ecx, inst->Arg(1).U32());
        c.cmp(reg, ecx);
        c.cmovb(reg, ecx);
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
        c.push(reg.cvt64());
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.cmp(reg, dword[rsp]);
        c.cmovb(reg, dword[rsp]);
        c.add(rsp, 8);
    }
    return true;
}

static bool EmitComputeOffsetBitFieldUExtract(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg,
                                              PassInfo& pass_info, IR::Inst* inst) {
    // We asume that the count is always less than 32, Is this correct?
    ASSERT(!inst->AreAllArgsImmediates());
    if (inst->Arg(0).IsImmediate()) {
        c.mov(reg, inst->Arg(0).U32());
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(0)));
    }
    bool in_stack = false;
    if (inst->Arg(1).IsImmediate()) {
        c.shr(reg, inst->Arg(1).U32());
    } else {
        c.push(reg.cvt64());
        in_stack = true;
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(1)));
        c.mov(ecx, reg);
        c.shr(dword[rsp], cl);
    }
    if (inst->Arg(2).IsImmediate()) {
        if (in_stack) {
            c.pop(reg.cvt64());
        }
        c.and_(reg, (1U << inst->Arg(2).U32()) - 1);
    } else {
        if (!in_stack) {
            c.push(reg.cvt64());
        }
        POP_ABORT_ON_FAILURE(ComputeOffset(c, reg, pass_info, inst->Arg(2)));
        c.mov(ecx, reg);
        c.mov(edx, 1);
        c.shl(edx, cl);
        c.dec(edx);
        c.and_(dword[rsp], edx);
        c.pop(reg.cvt64());
    }
    return true;
}

#endif // ARCH_X86_64

static bool IsAllowedOffsetInstruction(const IR::Inst* inst) {
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
    case IR::Opcode::ReadConst:
    case IR::Opcode::ReadConstBuffer:
    case IR::Opcode::IAdd32:
    case IR::Opcode::ISub32:
    case IR::Opcode::IMul32:
    case IR::Opcode::ShiftLeftLogical32:
    case IR::Opcode::ShiftRightLogical32:
    case IR::Opcode::BitwiseAnd32:
    case IR::Opcode::BitwiseOr32:
    case IR::Opcode::BitwiseXor32:
    case IR::Opcode::BitwiseNot32:
    case IR::Opcode::UMin32:
    case IR::Opcode::UMax32:
    case IR::Opcode::BitFieldUExtract:
        return true;
    default:
        return false;
    }
}

#ifdef ARCH_X86_64
static bool ComputeOffset(Xbyak::CodeGenerator& c, Xbyak::Reg32 reg, PassInfo& pass_info,
                          const IR::Value& off_dw) {
    ASSERT(reg != ecx && reg != edx);
    auto inst = off_dw.Inst();
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
        c.mov(reg, ptr[rsi + (static_cast<u32>(inst->Arg(0).ScalarReg()) << 2)]);
        return true;
    case IR::Opcode::ReadConst:
    case IR::Opcode::ReadConstBuffer:
        if (u16 offset = GetFlatbufOffset(pass_info.DeduplicateInstruction(inst)); offset != 0) {
            c.mov(reg, ptr[rsi + (offset << 2)]);
            return true;
        }
        return false;
    case IR::Opcode::IAdd32:
        ABORT_ON_FAILURE(EmitComputeOffsetIAdd32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::ISub32:
        ABORT_ON_FAILURE(EmitComputeOffsetISub32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::IMul32:
        ABORT_ON_FAILURE(EmitComputeOffsetIMul32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::ShiftLeftLogical32:
        ABORT_ON_FAILURE(EmitComputeOffsetShiftLeftLogical32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::ShiftRightLogical32:
        ABORT_ON_FAILURE(EmitComputeOffsetShiftRightLogical32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitwiseAnd32:
        ABORT_ON_FAILURE(EmitComputeOffsetBitwiseAnd32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitwiseOr32:
        ABORT_ON_FAILURE(EmitComputeOffsetBitwiseOr32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitwiseXor32:
        ABORT_ON_FAILURE(EmitComputeOffsetBitwiseXor32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitwiseNot32:
        ABORT_ON_FAILURE(EmitComputeOffsetBitwiseNot32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::UMin32:
        ABORT_ON_FAILURE(EmitComputeOffsetUMin32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::UMax32:
        ABORT_ON_FAILURE(EmitComputeOffsetUMax32(c, reg, pass_info, inst));
        return true;
    case IR::Opcode::BitFieldUExtract:
        ABORT_ON_FAILURE(EmitComputeOffsetBitFieldUExtract(c, reg, pass_info, inst));
        return true;
    default:
        LOG_ERROR(Render_Recompiler, "Unexpected instruction for offset computation, {}",
                  magic_enum::enum_name(inst->GetOpcode()));
        return false;
    }
}

static inline bool PushPtr(Xbyak::CodeGenerator& c, PassInfo& pass_info, const IR::Value& off_dw) {
    c.push(rdi);
    if (off_dw.IsImmediate()) {
        c.mov(rdi, ptr[rdi + (off_dw.U32() << 2)]);
    } else {
        ABORT_ON_FAILURE(ComputeOffset(c, r10d, pass_info, off_dw));
        c.shl(r10d, 2);
        c.mov(r10d, r10d);
        c.mov(rdi, ptr[rdi + r10]);
    }
    c.mov(r10, 0xFFFFFFFFFFFFULL);
    c.and_(rdi, r10);
    return true;
}

static inline void PopPtr(Xbyak::CodeGenerator& c) {
    c.pop(rdi);
}

static void VisitPointer(const IR::Value& off_dw, IR::Inst* subtree, PassInfo& pass_info,
                         Xbyak::CodeGenerator& c) {
    if (subtree->GetOpcode() == IR::Opcode::ReadConst && subtree->Flags<u16>() == 0 ||
        subtree->GetOpcode() == IR::Opcode::ReadConstBuffer &&
            subtree->Flags<IR::BufferInstInfo>().flatbuf_off_dw == 0) {
        return;
    }

    if (!PushPtr(c, pass_info, off_dw)) {
        LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
        return;
    }
    PassInfo::PtrUserList* use_list = pass_info.GetUsesAsPointer(subtree);
    ASSERT(use_list);

    // First copy all the src data from this tree level
    // That way, all data that was contiguous in the guest SRT is also contiguous in the
    // flattened buffer.
    // TODO src and dst are contiguous. Optimize with wider loads/stores
    // TODO if this subtree is dynamically indexed, don't compact it (keep it sparse)
    for (auto [src_off_dw, use] : *use_list) {
        if (src_off_dw.IsImmediate()) {
            c.mov(r10d, ptr[rdi + (src_off_dw.U32() << 2)]);
        } else {
            if (!ComputeOffset(c, r10d, pass_info, src_off_dw)) {
                LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
                continue;
            }
            c.shl(r10d, 2);
            c.mov(r10d, r10d);
            c.mov(r10d, dword[rdi + r10]);
        }
        c.mov(ptr[rsi + (pass_info.dst_off_dw << 2)], r10d);

        SetFlatbufOffset(use, pass_info.dst_off_dw);
        pass_info.dst_off_dw++;
    }

    // Then visit any children used as pointers
    for (const auto [src_off_dw, use] : *use_list) {
        if (pass_info.GetUsesAsPointer(use)) {
            VisitPointer(src_off_dw, use, pass_info, c);
        }
    }

    PopPtr(c);
}

static void GenerateSrtProgram(Info& info, PassInfo& pass_info) {
    Xbyak::CodeGenerator& c = g_srt_codegen;

    if (pass_info.srt_roots.empty()) {
        return;
    }

    // Register the signal handler for SRT walker, if not already registered
    if (g_srt_codegen_start == nullptr) {
        g_srt_codegen_start = c.getCurr();
        auto* signals = Core::Signals::Instance();
        // Call after the memory invalidation handler
        constexpr u32 priority = 1;
        signals->RegisterAccessViolationHandler(SrtWalkerSignalHandler, priority);
    }

    info.srt_info.walker_func = c.getCurr<PFN_SrtWalker>();
    pass_info.dst_off_dw = NUM_USER_DATA_REGS;
    ASSERT(pass_info.dst_off_dw == info.srt_info.flattened_bufsize_dw);

    for (const auto& [sgpr_base, root] : pass_info.srt_roots) {
        VisitPointer(IR::Value(static_cast<u32>(sgpr_base)), root, pass_info, c);
    }

    c.ret();
    c.ready();

    info.srt_info.walker_func_size =
        c.getCurr() - reinterpret_cast<const u8*>(info.srt_info.walker_func);

    if (EmulatorSettings.IsDumpShaders()) {
        DumpSrtProgram(info, reinterpret_cast<const u8*>(info.srt_info.walker_func),
                       info.srt_info.walker_func_size);
    }

    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
}

#endif // ARCH_X86_64

// bbport: the walk as bytecode instead of x86 code, for hosts other than x86-64 (the native
// arm64 GPU process on macOS, docs/macos-native-gpu.md). On x86-64, BB_SRT_CHECK=1 also builds
// it next to the x86 code, and every walk compares the two (RunSrtWalker). Same walk, same flattened layout: the
// x86 code keeps the SRT pointer in rdi, the flat buffer in rsi, offsets in a 32-bit register
// with a stack for the second operand; here a pointer stack, an accumulator and a value stack.
// RunSrtWalker interprets it.

static void Emit(std::vector<u32>& code, SrtOp op, u32 a = 0, u32 b = 0) {
    code.push_back(static_cast<u32>(op));
    code.push_back(a);
    code.push_back(b);
}

static bool ComputeOffset(std::vector<u32>& code, PassInfo& pass_info, const IR::Value& off_dw);

/// Leaves the value in the accumulator.
static bool EmitValue(std::vector<u32>& code, PassInfo& pass_info, const IR::Value& value) {
    if (value.IsImmediate()) {
        Emit(code, SrtOp::AccImm, value.U32());
        return true;
    }
    return ComputeOffset(code, pass_info, value);
}

/// acc = op(arg0, arg1): arg0 goes to the value stack while arg1 is computed.
static bool EmitBinary(std::vector<u32>& code, PassInfo& pass_info, IR::Inst* inst, SrtOp op) {
    ASSERT(!inst->AreAllArgsImmediates());
    if (!EmitValue(code, pass_info, inst->Arg(0))) {
        return false;
    }
    Emit(code, SrtOp::Push);
    if (!EmitValue(code, pass_info, inst->Arg(1))) {
        return false;
    }
    Emit(code, op);
    return true;
}

static bool ComputeOffset(std::vector<u32>& code, PassInfo& pass_info, const IR::Value& off_dw) {
    auto inst = off_dw.Inst();
    switch (inst->GetOpcode()) {
    case IR::Opcode::GetUserData:
        // The flat buffer starts with the user data registers.
        Emit(code, SrtOp::AccFlat, static_cast<u32>(inst->Arg(0).ScalarReg()));
        return true;
    case IR::Opcode::ReadConst:
    case IR::Opcode::ReadConstBuffer:
        if (u16 offset = GetFlatbufOffset(pass_info.DeduplicateInstruction(inst)); offset != 0) {
            Emit(code, SrtOp::AccFlat, offset);
            return true;
        }
        return false;
    case IR::Opcode::IAdd32:
        return EmitBinary(code, pass_info, inst, SrtOp::Add);
    case IR::Opcode::ISub32:
        return EmitBinary(code, pass_info, inst, SrtOp::Sub);
    case IR::Opcode::IMul32:
        return EmitBinary(code, pass_info, inst, SrtOp::Mul);
    case IR::Opcode::ShiftLeftLogical32:
        return EmitBinary(code, pass_info, inst, SrtOp::Shl);
    case IR::Opcode::ShiftRightLogical32:
        return EmitBinary(code, pass_info, inst, SrtOp::Shr);
    case IR::Opcode::BitwiseAnd32:
        return EmitBinary(code, pass_info, inst, SrtOp::And);
    case IR::Opcode::BitwiseOr32:
        return EmitBinary(code, pass_info, inst, SrtOp::Or);
    case IR::Opcode::BitwiseXor32:
        return EmitBinary(code, pass_info, inst, SrtOp::Xor);
    case IR::Opcode::UMin32:
        return EmitBinary(code, pass_info, inst, SrtOp::UMin);
    case IR::Opcode::UMax32:
        return EmitBinary(code, pass_info, inst, SrtOp::UMax);
    case IR::Opcode::BitwiseNot32:
        ASSERT(!inst->AreAllArgsImmediates());
        if (!EmitValue(code, pass_info, inst->Arg(0))) {
            return false;
        }
        Emit(code, SrtOp::Not);
        return true;
    case IR::Opcode::BitFieldUExtract:
        ASSERT(!inst->AreAllArgsImmediates());
        for (u32 arg = 0; arg < 3; ++arg) {
            if (!EmitValue(code, pass_info, inst->Arg(arg))) {
                return false;
            }
            if (arg < 2) {
                Emit(code, SrtOp::Push);
            }
        }
        Emit(code, SrtOp::Bfe);
        return true;
    default:
        LOG_ERROR(Render_Recompiler, "Unexpected instruction for offset computation, {}",
                  magic_enum::enum_name(inst->GetOpcode()));
        return false;
    }
}

static void VisitPointer(const IR::Value& off_dw, IR::Inst* subtree, PassInfo& pass_info,
                         std::vector<u32>& code) {
    if (subtree->GetOpcode() == IR::Opcode::ReadConst && subtree->Flags<u16>() == 0 ||
        subtree->GetOpcode() == IR::Opcode::ReadConstBuffer &&
            subtree->Flags<IR::BufferInstInfo>().flatbuf_off_dw == 0) {
        return;
    }
    // A failed offset leaves nothing behind (the x86 path's push would stay unbalanced).
    std::vector<u32> push;
    if (off_dw.IsImmediate()) {
        Emit(push, SrtOp::PushPtrImm, off_dw.U32());
    } else if (ComputeOffset(push, pass_info, off_dw)) {
        Emit(push, SrtOp::PushPtrAcc);
    } else {
        LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
        return;
    }
    code.insert(code.end(), push.begin(), push.end());
    PassInfo::PtrUserList* use_list = pass_info.GetUsesAsPointer(subtree);
    ASSERT(use_list);

    // As on x86: this level's data first, contiguous in the flattened buffer, then children.
    for (auto [src_off_dw, use] : *use_list) {
        if (src_off_dw.IsImmediate()) {
            Emit(code, SrtOp::CopyImm, src_off_dw.U32(), pass_info.dst_off_dw);
        } else {
            std::vector<u32> offset;
            if (!ComputeOffset(offset, pass_info, src_off_dw)) {
                LOG_ERROR(Render_Recompiler, "Failed to compute offset for SRT walker");
                continue;
            }
            code.insert(code.end(), offset.begin(), offset.end());
            Emit(code, SrtOp::CopyAcc, 0, pass_info.dst_off_dw);
        }
        SetFlatbufOffset(use, pass_info.dst_off_dw);
        pass_info.dst_off_dw++;
    }
    for (const auto [src_off_dw, use] : *use_list) {
        if (pass_info.GetUsesAsPointer(use)) {
            VisitPointer(src_off_dw, use, pass_info, code);
        }
    }
    Emit(code, SrtOp::PopPtr);
}

/// The bytecode of the walk; assigns the flattened offsets exactly as the x86 generator does
/// (same traversal), so running both on one program gives one layout.
static std::vector<u32> GenerateSrtBytecode(PassInfo& pass_info) {
    pass_info.dst_off_dw = NUM_USER_DATA_REGS;
    std::vector<u32> code;
    for (const auto& [sgpr_base, root] : pass_info.srt_roots) {
        VisitPointer(IR::Value(static_cast<u32>(sgpr_base)), root, pass_info, code);
    }
    Emit(code, SrtOp::End);
    return code;
}

static bool SrtCheckEnabled() {
    static const bool on = [] {
        const char* env = std::getenv("BB_SRT_CHECK");
        return env && env[0] == '1';
    }();
    return on;
}

static void GenerateSrt(Info& info, PassInfo& pass_info) {
    if (pass_info.srt_roots.empty()) {
        return;
    }
#ifdef ARCH_X86_64
    if (SrtCheckEnabled()) {
        const u16 start = NUM_USER_DATA_REGS;
        ASSERT(start == info.srt_info.flattened_bufsize_dw);
        const auto code = GenerateSrtBytecode(pass_info);
        const u16 bytecode_end = pass_info.dst_off_dw;
        info.srt_info.check_code = RegisterSrtBytecode(code.data(), code.size());
        GenerateSrtProgram(info, pass_info); // assigns the same offsets again
        if (pass_info.dst_off_dw != bytecode_end) {
            LOG_ERROR(Render_Recompiler, "SRT check: bytecode flattens {} dwords, x86 code {}",
                      bytecode_end, pass_info.dst_off_dw);
            info.srt_info.check_code = nullptr;
        }
        return;
    }
    GenerateSrtProgram(info, pass_info);
#else
    ASSERT(NUM_USER_DATA_REGS == info.srt_info.flattened_bufsize_dw);
    const auto code = GenerateSrtBytecode(pass_info);
    info.srt_info.walker_func = RegisterWalkerCode(reinterpret_cast<const u8*>(code.data()),
                                                   code.size() * sizeof(u32));
    info.srt_info.walker_func_size = code.size() * sizeof(u32);
    info.srt_info.flattened_bufsize_dw = pass_info.dst_off_dw;
#endif
}

static bool IsReadConstSource(const IR::Value base) {
    auto* inst = base.TryInst();
    if (!inst) {
        return false;
    }
    return inst->GetOpcode() == IR::Opcode::GetUserData ||
           inst->GetOpcode() == IR::Opcode::ReadConst ||
           inst->GetOpcode() == IR::Opcode::ReadConstBuffer;
}

void SimplifyReadConstAddressAdd(IR::Inst& inst) {
    // This handles the following pattern by combining the addition with the offset
    // s_add_u32    s12, s2, <offset>
    // s_addc_u32   s13, s3, 0
    // s_load_dword s16, s[12:13], 0x0

    if (inst.GetOpcode() != IR::Opcode::ReadConst) {
        return;
    }
    IR::Inst* addr = inst.Arg(0).Inst();
    IR::Inst* hi = addr->Arg(1).TryInst();
    IR::Inst* lo = addr->Arg(0).TryInst();
    if (!hi || !lo) {
        return;
    }

    if (lo->GetOpcode() != IR::Opcode::IAdd32 || hi->GetOpcode() != IR::Opcode::IAdd32) {
        return;
    }

    IR::Value add_offset;
    IR::Value base;
    if (base = lo->Arg(0); IsReadConstSource(base)) {
        add_offset = lo->Arg(1);
    } else if (base = lo->Arg(1); IsReadConstSource(base)) {
        add_offset = lo->Arg(0);
    } else {
        return;
    }

    IR::Inst* sel = hi->Arg(1).TryInst();
    if (!sel || sel->GetOpcode() != IR::Opcode::SelectU32) {
        return;
    }

    IR::Value sel_true = sel->Arg(1);
    IR::Value sel_false = sel->Arg(2);
    if (!sel_true.IsImmediate() || !sel_false.IsImmediate() || sel_true.U32() != 1 ||
        sel_false.U32() != 0) {
        return;
    }

    addr->SetArg(0, base);
    addr->SetArg(1, hi->Arg(0));
    for (auto [user, operand] : addr->Uses()) {
        ASSERT(user->GetOpcode() == IR::Opcode::ReadConst && operand == 0);
        IR::IREmitter ir{*user->GetParent(), IR::Block::InstructionList::s_iterator_to(*user)};
        IR::U32 offset = IR::U32{user->Arg(1)};
        IR::U32 dw_add_offset =
            add_offset.IsImmediate()
                ? ir.Imm32(add_offset.U32() >> 2u)
                : IR::U32{ir.ShiftRightLogical(IR::U32{add_offset}, ir.Imm32(2u))};
        if (offset.IsImmediate() && dw_add_offset.IsImmediate()) {
            user->SetArg(1, ir.Imm32(offset.U32() + dw_add_offset.U32()));
        } else {
            user->SetArg(1, ir.IAdd(offset, dw_add_offset));
        }
    }
}

} // Anonymous namespace

void FlattenExtendedUserdataPass(IR::Program& program) {
    auto& post_order = program.post_order_blocks;
    PassInfo pass_info;

    // traverse at end and assign offsets to duplicate readconsts, using
    // vn_to_inst as the source
    boost::container::small_vector<IR::Inst*, 32> all_readconsts, visited;
    std::queue<IR::Inst*> queue;

    for (auto it = post_order.rbegin(); it != post_order.rend(); it++) {
        IR::Block* block{*it};
        for (IR::Inst& inst : *block) {
            if (inst.GetOpcode() != IR::Opcode::ReadConst &&
                inst.GetOpcode() != IR::Opcode::ReadConstBuffer) {
                continue;
            }

            if (inst.GetOpcode() == IR::Opcode::ReadConstBuffer) {
                // Only flatten ReadConstBuffer if it was marked as a sharp source in the
                // resource discovery pass
                auto inst_info = inst.Flags<IR::BufferInstInfo>();
                if (!inst_info.sharp_source) {
                    continue;
                }
            }

            SimplifyReadConstAddressAdd(inst);

            all_readconsts.push_back(&inst);

            auto offset = inst.Arg(1);
            if (offset.IsImmediate()) {
                continue;
            }

            queue.push(offset.Inst());
            while (!queue.empty()) {
                IR::Inst* inst{queue.front()};
                queue.pop();

                if (inst->GetOpcode() == IR::Opcode::ReadConstBuffer) {
                    auto buffer_inst_info = inst->Flags<IR::BufferInstInfo>();
                    if (!buffer_inst_info.sharp_source.Value()) {
                        all_readconsts.push_back(inst);
                        auto offset = inst->Arg(1);
                        if (!offset.IsImmediate()) {
                            auto arg_inst = offset.Inst();
                            if (std::ranges::find(visited, arg_inst) == visited.end()) {
                                visited.push_back(arg_inst);
                                queue.push(arg_inst);
                            }
                        }
                    }
                    continue;
                }
                if (inst->GetOpcode() == IR::Opcode::GetUserData ||
                    inst->GetOpcode() == IR::Opcode::ReadConst ||
                    !IsAllowedOffsetInstruction(inst)) {
                    continue;
                }
                for (size_t arg = inst->NumArgs(); arg--;) {
                    const IR::Value arg_value = inst->Arg(arg);
                    if (arg_value.IsImmediate()) {
                        continue;
                    }
                    IR::Inst* arg_inst = arg_value.Inst();
                    if (std::ranges::find(visited, arg_inst) == visited.end()) {
                        visited.push_back(arg_inst);
                        queue.push(arg_inst);
                    }
                }
            }
        }
    }

    for (auto inst : all_readconsts) {
        if (pass_info.DeduplicateInstruction(inst) != inst) {
            // This is a duplicate of a readconst we've already visited
            continue;
        }

        IR::Inst* base = inst->Arg(0).Inst();
        if (auto* inst = base->Arg(0).TryInst();
            inst && inst->GetOpcode() == IR::Opcode::ReadFirstLane) {
            continue;
        }
        ASSERT_MSG(IsReadConstSource(base->Arg(0)), "ReadConst base low not from constant memory");
        ASSERT_MSG(IsReadConstSource(base->Arg(1)), "ReadConst base high not from constant memory");

        IR::Inst* ptr_lo = base->Arg(0).Inst();
        ptr_lo = pass_info.DeduplicateInstruction(ptr_lo);

        auto ptr_uses_kv = pass_info.pointer_uses.try_emplace(ptr_lo, PassInfo::PtrUserList{});
        PassInfo::PtrUserList& user_list = ptr_uses_kv.first->second;

        user_list[inst->Arg(1)] = inst;

        if (ptr_lo->GetOpcode() == IR::Opcode::GetUserData) {
            IR::ScalarReg ud_reg = ptr_lo->Arg(0).ScalarReg();
            pass_info.srt_roots[ud_reg] = ptr_lo;
        }
    }

    GenerateSrt(program.info, pass_info);

    // Assign offsets to duplicate readconsts
    for (IR::Inst* readconst : all_readconsts) {
        ASSERT(pass_info.vn_to_inst.contains(pass_info.gvn_table.GetValueNumber(readconst)));
        IR::Inst* original = pass_info.DeduplicateInstruction(readconst);
        SetFlatbufOffset(readconst, GetFlatbufOffset(original));
    }

    program.info.RefreshFlatBuf();
}

} // namespace Shader::Optimization

namespace Shader {

namespace {
std::mutex g_walker_code_mutex;
std::vector<std::unique_ptr<u32[]>> g_walker_code; // never freed, as the x86 code arena
} // namespace

const u32* RegisterSrtBytecode(const u32* words, size_t count) {
    ASSERT(count % 3 == 0);
    auto copy = std::make_unique<u32[]>(count);
    std::memcpy(copy.get(), words, count * sizeof(u32));
    std::scoped_lock lock{g_walker_code_mutex};
    g_walker_code.push_back(std::move(copy));
    return g_walker_code.back().get();
}

#ifndef ARCH_X86_64
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size) {
    // Bytecode, not machine code: only RunSrtWalker uses the pointer.
    ASSERT(size % (3 * sizeof(u32)) == 0);
    std::vector<u32> words(size / sizeof(u32));
    std::memcpy(words.data(), ptr, size);
    return reinterpret_cast<PFN_SrtWalker>(
        const_cast<u32*>(RegisterSrtBytecode(words.data(), words.size())));
}
#endif

bool (*srt_guest_readable)(u64 address, u64 size) = nullptr;
u64 (*srt_guest_translate)(u64 address) = nullptr;

namespace {
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

void InterpretSrt(const u32* program, const u32* user_data, u32* flat) {
    using Optimization::SrtOp;
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
            const u64 offset = code[0] == static_cast<u32>(SrtOp::PushPtrImm)
                                   ? u64(code[1]) << 2
                                   : u64(u32(acc << 2));
            pointers[num_pointers++] = pointer;
            const u64 address = pointer + offset;
            pointer = (num_pointers == 1 ? *reinterpret_cast<const u64*>(address)
                                         : LoadGuest<u64>(address)) &
                      PointerMask;
            break;
        }
        case SrtOp::PopPtr:
            pointer = num_pointers ? pointers[--num_pointers] : pointer;
            break;
        case SrtOp::CopyImm:
            flat[code[2]] = LoadGuest<u32>(pointer + (u64(code[1]) << 2));
            break;
        case SrtOp::CopyAcc:
            flat[code[2]] = LoadGuest<u32>(pointer + u64(u32(acc << 2)));
            break;
        case SrtOp::AccFlat:
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
            acc = (value >> (offset & 31)) & (count >= 32 ? ~0u : (1u << count) - 1);
            break;
        }
        default:
            UNREACHABLE_MSG("Bad SRT walker opcode {}", code[0]);
        }
    }
}
} // namespace

void RunSrtWalker(const PersistentSrtInfo& srt, const u32* user_data, u32* flat) {
#ifdef ARCH_X86_64
    if (!srt.check_code) {
        srt.walker_func(user_data, flat);
        return;
    }
    // BB_SRT_CHECK: the bytecode on a copy, compared with the x86 code's result.
    const u32 size = srt.flattened_bufsize_dw;
    std::vector<u32> expected(flat, flat + size);
    InterpretSrt(srt.check_code, user_data, expected.data());
    srt.walker_func(user_data, flat);
    static std::atomic<u64> walks{0}, mismatches{0}, raced{0};
    const u64 walk = walks.fetch_add(1, std::memory_order_relaxed) + 1;
    bool differs = std::memcmp(expected.data(), flat, size * sizeof(u32)) != 0;
    if (differs) {
        // The game may have rewritten the table between the two walks: a second bytecode walk
        // that now agrees means that, not a difference between the walkers.
        std::vector<u32> again(flat, flat + size);
        std::memcpy(again.data(), expected.data(), NUM_USER_DATA_REGS * sizeof(u32));
        InterpretSrt(srt.check_code, user_data, again.data());
        if (std::memcmp(again.data(), flat, size * sizeof(u32)) == 0) {
            raced.fetch_add(1, std::memory_order_relaxed);
            differs = false;
        }
    }
    if (differs) {
        const u64 count = mismatches.fetch_add(1, std::memory_order_relaxed) + 1;
        if (count <= 16) {
            u32 first = 0;
            while (first < size && expected[first] == flat[first]) {
                ++first;
            }
            LOG_ERROR(Render_Recompiler,
                      "SRT check: walk {} differs at dword {} of {}: bytecode {:#x}, x86 {:#x}",
                      walk, first, size, expected[first], flat[first]);
        }
    }
    // Always in the log (LOG_INFO is filtered): a run with no line at all walked no table.
    if ((walk & (walk - 1)) == 0 && (walk == 1 || walk >= 1024)) {
        std::printf("SRT check: %llu walks compared with the x86 walker, %llu differed "
                    "(%llu more changed by the game during the check)\n",
                    static_cast<unsigned long long>(walk),
                    static_cast<unsigned long long>(mismatches.load(std::memory_order_relaxed)),
                    static_cast<unsigned long long>(raced.load(std::memory_order_relaxed)));
        std::fflush(stdout);
    }
#else
    InterpretSrt(reinterpret_cast<const u32*>(srt.walker_func), user_data, flat);
#endif
}

} // namespace Shader
