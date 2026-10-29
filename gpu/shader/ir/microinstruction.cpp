// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/microinstruction.h"

#include <algorithm>
#include <memory>

#include "shader/ir/basic_block.h"

namespace Shader::IR {

Inst::Inst(IR::Opcode op_, u32 flags_) noexcept : op{op_}, flags{flags_} {
    if (op == Opcode::Phi) {
        std::construct_at(&phi_args);
    } else {
        std::construct_at(&args);
    }
}

Inst::Inst(const Inst& base) : op{base.op}, flags{base.flags} {
    ASSERT_MSG(base.op != Opcode::Phi, "copying a phi");
    std::construct_at(&args);
    for (size_t index = 0; index < base.NumArgs(); ++index) {
        SetArg(index, base.Arg(index));
    }
}

Inst::~Inst() {
    if (op == Opcode::Phi) {
        std::destroy_at(&phi_args);
    } else {
        std::destroy_at(&args);
    }
}

bool Inst::MayHaveSideEffects() const noexcept {
    switch (op) {
    // Control and synchronization.
    case Opcode::Barrier:
    case Opcode::WorkgroupMemoryBarrier:
    case Opcode::DeviceMemoryBarrier:
    case Opcode::ConditionRef:
    case Opcode::Prologue:
    case Opcode::Epilogue:
    case Opcode::Discard:
    case Opcode::DiscardCond:
    case Opcode::DebugPrint:
    case Opcode::EmitVertex:
    case Opcode::EmitPrimitive:
    // Outputs.
    case Opcode::SetAttribute:
    case Opcode::SetTcsGenericAttribute:
    case Opcode::SetPatch:
    // Buffer stores and atomics.
    case Opcode::StoreBufferU8:
    case Opcode::StoreBufferU16:
    case Opcode::StoreBufferU32:
    case Opcode::StoreBufferU32x2:
    case Opcode::StoreBufferU32x3:
    case Opcode::StoreBufferU32x4:
    case Opcode::StoreBufferU64:
    case Opcode::StoreBufferF32:
    case Opcode::StoreBufferF32x2:
    case Opcode::StoreBufferF32x3:
    case Opcode::StoreBufferF32x4:
    case Opcode::StoreBufferFormatF32:
    case Opcode::BufferAtomicIAdd32:
    case Opcode::BufferAtomicIAdd64:
    case Opcode::BufferAtomicISub32:
    case Opcode::BufferAtomicSMin32:
    case Opcode::BufferAtomicSMin64:
    case Opcode::BufferAtomicUMin32:
    case Opcode::BufferAtomicUMin64:
    case Opcode::BufferAtomicFMin32:
    case Opcode::BufferAtomicSMax32:
    case Opcode::BufferAtomicSMax64:
    case Opcode::BufferAtomicUMax32:
    case Opcode::BufferAtomicUMax64:
    case Opcode::BufferAtomicFMax32:
    case Opcode::BufferAtomicInc32:
    case Opcode::BufferAtomicDec32:
    case Opcode::BufferAtomicAnd32:
    case Opcode::BufferAtomicOr32:
    case Opcode::BufferAtomicXor32:
    case Opcode::BufferAtomicSwap32:
    case Opcode::BufferAtomicCmpSwap32:
    case Opcode::BufferAtomicFCmpSwap32:
    // Shared memory stores and atomics.
    case Opcode::WriteSharedU16:
    case Opcode::WriteSharedU32:
    case Opcode::WriteSharedU64:
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
    case Opcode::SharedAtomicCmpSwap32:
    case Opcode::SharedAtomicCmpSwap64:
    // Image stores and atomics.
    case Opcode::ImageWrite:
    case Opcode::ImageAtomicIAdd32:
    case Opcode::ImageAtomicSMin32:
    case Opcode::ImageAtomicUMin32:
    case Opcode::ImageAtomicSMax32:
    case Opcode::ImageAtomicUMax32:
    case Opcode::ImageAtomicFMax32:
    case Opcode::ImageAtomicFMin32:
    case Opcode::ImageAtomicInc32:
    case Opcode::ImageAtomicDec32:
    case Opcode::ImageAtomicAnd32:
    case Opcode::ImageAtomicOr32:
    case Opcode::ImageAtomicXor32:
    case Opcode::ImageAtomicExchange32:
    case Opcode::ImageAtomicCmpSwap32:
        return true;
    default:
        return false;
    }
}

bool Inst::AreAllArgsImmediates() const {
    ASSERT_MSG(op != Opcode::Phi, "immediate test of a phi's arguments");
    return std::all_of(args.begin(), args.begin() + NumArgs(),
                       [](const Value& value) { return value.IsImmediate(); });
}

IR::Type Inst::Type() const {
    switch (op) {
    case Opcode::Phi:
        return Flags<IR::Type>(); // a phi's type is in its flags
    case Opcode::GetVirtualRegister:
        return Arg(0).VirtualReg().type;
    default:
        return TypeOf(op);
    }
}

void Inst::SetArg(size_t index, Value value) {
    ASSERT_MSG(index < NumArgs(), "argument {} of {}", index, op);
    if (Inst* old = Arg(index).TryInst()) {
        RemoveUse(old, u32(index));
    }
    if (Inst* used = value.TryInst()) {
        AddUse(used, u32(index));
    }
    if (op == Opcode::Phi) {
        phi_args[index].second = value;
    } else {
        args[index] = value;
    }
}

Block* Inst::PhiBlock(size_t index) const {
    ASSERT_MSG(op == Opcode::Phi, "{} is not a phi", op);
    ASSERT_MSG(index < phi_args.size(), "phi argument {} of {}", index, phi_args.size());
    return phi_args[index].first;
}

void Inst::AddPhiOperand(Block* predecessor, const Value& value) {
    if (Inst* used = value.TryInst()) {
        AddUse(used, u32(phi_args.size()));
    }
    phi_args.emplace_back(predecessor, value);
}

void Inst::Invalidate() {
    ClearArgs();
    ReplaceOpcode(Opcode::Void);
}

void Inst::ClearArgs() {
    if (op == Opcode::Phi) {
        for (u32 i = 0; i < phi_args.size(); ++i) {
            if (Inst* used = phi_args[i].second.TryInst()) {
                RemoveUse(used, i);
            }
        }
        phi_args.clear();
        return;
    }
    for (u32 i = 0; i < args.size(); ++i) {
        if (Inst* used = args[i].TryInst()) {
            RemoveUse(used, i);
        }
    }
    std::memset(static_cast<void*>(&args), 0, sizeof(args));
}

void Inst::ReplaceUses(Value replacement) {
    // SetArg removes the use being walked: advance first.
    for (auto it = uses.begin(); it != uses.end();) {
        const auto [user, operand] = *it++;
        DEBUG_ASSERT(user->Arg(operand).Inst() == this);
        user->SetArg(operand, replacement);
    }
    Invalidate();
}

void Inst::ReplaceOpcode(IR::Opcode opcode) {
    ASSERT_MSG(opcode != Opcode::Phi, "an instruction cannot become a phi");
    if (op == Opcode::Phi) {
        std::destroy_at(&phi_args);
        std::construct_at(&args);
    }
    op = opcode;
}

void Inst::AddUse(Inst* used, u32 operand) {
    used->uses.emplace_front(this, operand);
}

void Inst::RemoveUse(Inst* used, u32 operand) {
    used->uses.remove(IR::Use{this, operand});
}

} // namespace Shader::IR
