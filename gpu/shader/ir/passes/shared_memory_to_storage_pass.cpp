// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Compute shaders whose LDS does not fit the host's shared memory (or that alias several types
// in it without explicit workgroup layouts) get it as a storage buffer instead: one slice of
// shared_memory_size per workgroup, with LDS accesses turned into buffer accesses.
#include <bit>

#include "common/assert.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/program.h"
#include "shader/profile.h"
#include "shader/runtime_info.h"

namespace Shader::Optimization {

namespace {

using IR::Opcode;

/// The type an LDS access works on; Void for other instructions.
IR::Type SharedAccessType(Opcode op) {
    switch (op) {
    case Opcode::LoadSharedU16:
    case Opcode::WriteSharedU16:
        return IR::Type::U16;
    case Opcode::LoadSharedU32:
    case Opcode::WriteSharedU32:
    case Opcode::SharedAtomicIAdd32:
    case Opcode::SharedAtomicISub32:
    case Opcode::SharedAtomicSMin32:
    case Opcode::SharedAtomicUMin32:
    case Opcode::SharedAtomicSMax32:
    case Opcode::SharedAtomicUMax32:
    case Opcode::SharedAtomicInc32:
    case Opcode::SharedAtomicDec32:
    case Opcode::SharedAtomicAnd32:
    case Opcode::SharedAtomicOr32:
    case Opcode::SharedAtomicXor32:
    case Opcode::SharedAtomicCmpSwap32:
        return IR::Type::U32;
    case Opcode::LoadSharedU64:
    case Opcode::WriteSharedU64:
    case Opcode::SharedAtomicIAdd64:
    case Opcode::SharedAtomicISub64:
    case Opcode::SharedAtomicSMin64:
    case Opcode::SharedAtomicUMin64:
    case Opcode::SharedAtomicSMax64:
    case Opcode::SharedAtomicUMax64:
    case Opcode::SharedAtomicInc64:
    case Opcode::SharedAtomicDec64:
    case Opcode::SharedAtomicAnd64:
    case Opcode::SharedAtomicOr64:
    case Opcode::SharedAtomicXor64:
    case Opcode::SharedAtomicCmpSwap64:
        return IR::Type::U64;
    default:
        return IR::Type::Void;
    }
}

IR::Type UsedSharedTypes(const IR::Program& program) {
    IR::Type types = IR::Type::Void;
    for (IR::Block* const block : program.blocks) {
        for (const IR::Inst& inst : block->Instructions()) {
            types |= SharedAccessType(inst.GetOpcode());
        }
    }
    return types;
}

/// Replaces one LDS access with the buffer access at `address`.
void Lower(IR::IREmitter& ir, IR::Inst& inst, const IR::U32& handle, const IR::U32& address) {
    const IR::Value data = inst.NumArgs() > 1 ? inst.Arg(1) : IR::Value{};
    switch (const Opcode op = inst.GetOpcode()) {
    case Opcode::SharedAtomicIAdd32:
    case Opcode::SharedAtomicIAdd64:
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicIAdd(handle, address, data, {}));
        break;
    case Opcode::SharedAtomicISub32:
    case Opcode::SharedAtomicISub64:
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicISub(handle, address, data, {}));
        break;
    case Opcode::SharedAtomicSMin32:
    case Opcode::SharedAtomicUMin32:
    case Opcode::SharedAtomicSMin64:
    case Opcode::SharedAtomicUMin64: {
        const bool is_signed =
            op == Opcode::SharedAtomicSMin32 || op == Opcode::SharedAtomicSMin64;
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicIMin(handle, address, data, is_signed, {}));
        break;
    }
    case Opcode::SharedAtomicSMax32:
    case Opcode::SharedAtomicUMax32:
    case Opcode::SharedAtomicSMax64:
    case Opcode::SharedAtomicUMax64: {
        const bool is_signed =
            op == Opcode::SharedAtomicSMax32 || op == Opcode::SharedAtomicSMax64;
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicIMax(handle, address, data, is_signed, {}));
        break;
    }
    case Opcode::SharedAtomicInc32:
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicInc(handle, address, {}));
        break;
    case Opcode::SharedAtomicDec32:
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicDec(handle, address, {}));
        break;
    case Opcode::SharedAtomicAnd32:
    case Opcode::SharedAtomicAnd64:
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicAnd(handle, address, data, {}));
        break;
    case Opcode::SharedAtomicOr32:
    case Opcode::SharedAtomicOr64:
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicOr(handle, address, data, {}));
        break;
    case Opcode::SharedAtomicXor32:
    case Opcode::SharedAtomicXor64:
        inst.ReplaceUsesWithAndRemove(ir.BufferAtomicXor(handle, address, data, {}));
        break;
    case Opcode::SharedAtomicCmpSwap32:
    case Opcode::SharedAtomicCmpSwap64:
        inst.ReplaceUsesWithAndRemove(
            ir.BufferAtomicCmpSwap(handle, address, data, inst.Arg(2), {}));
        break;
    case Opcode::LoadSharedU16:
        inst.ReplaceUsesWithAndRemove(ir.LoadBufferU16(handle, address, {}));
        break;
    case Opcode::LoadSharedU32:
        inst.ReplaceUsesWithAndRemove(ir.LoadBufferU32(1, handle, address, {}));
        break;
    case Opcode::LoadSharedU64:
        inst.ReplaceUsesWithAndRemove(ir.LoadBufferU64(handle, address, {}));
        break;
    case Opcode::WriteSharedU16:
        ir.StoreBufferU16(handle, address, IR::U16{data}, {});
        inst.Invalidate();
        break;
    case Opcode::WriteSharedU32:
        ir.StoreBufferU32(1, handle, address, data, {});
        inst.Invalidate();
        break;
    case Opcode::WriteSharedU64:
        ir.StoreBufferU64(handle, address, IR::U64{data}, {});
        inst.Invalidate();
        break;
    default:
        UNREACHABLE_MSG("LDS access {} has no buffer form", op);
    }
}

} // namespace

void SharedMemoryToStoragePass(IR::Program& program, const RuntimeInfo& runtime_info,
                               const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute) {
        return;
    }
    const u32 shared_memory_size = runtime_info.hw.cs.shared_memory_size;
    const IR::Type used_types = UsedSharedTypes(program);
    if (used_types == IR::Type::Void) {
        return;
    }
    const bool fits = shared_memory_size <= profile.max_shared_memory_size;
    const bool one_type = std::popcount(static_cast<u32>(used_types)) == 1;
    if (fits && (profile.supports_workgroup_explicit_memory_layout || one_type)) {
        return;
    }

    const u32 binding = static_cast<u32>(program.info.buffers.size());
    program.info.buffers.push_back(BufferResource{
        .used_types = used_types,
        .buffer_type = BufferType::SharedMemory,
        .is_written = true,
    });
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (SharedAccessType(inst.GetOpcode()) == IR::Type::Void) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const IR::U32 handle = ir.Imm32(binding);
            const IR::U32 workgroup = ir.GetAttributeU32(IR::Attribute::WorkgroupIndex);
            const IR::U32 offset = ir.IMul(workgroup, ir.Imm32(shared_memory_size));
            const IR::U32 address = ir.IAdd(IR::U32{inst.Arg(0)}, offset);
            Lower(ir, inst, handle, address);
        }
    }
}

} // namespace Shader::Optimization
