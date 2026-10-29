// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/value.h"

#include <string_view>

#include "common/hash.h"
#include "shader/ir/microinstruction.h"

namespace Shader::IR {

bool Value::IsPhi() const noexcept {
    return type == Type::Opaque && inst->GetOpcode() == Opcode::Phi;
}

IR::Type Value::Type() const noexcept {
    return type == Type::Opaque ? inst->Type() : type;
}

u64 Value::Hash() const noexcept {
    const u64 seed = HashCombine(static_cast<u64>(type), 0ULL);
    switch (type) {
    case Type::Void:
        return seed;
    case Type::Opaque:
        return reinterpret_cast<u64>(inst);
    case Type::ScalarReg:
        return HashCombine(static_cast<u64>(sreg), seed);
    case Type::VectorReg:
        return HashCombine(static_cast<u64>(vreg), seed);
    case Type::Attribute:
        return HashCombine(static_cast<u64>(attribute), seed);
    case Type::U1:
        return HashCombine(static_cast<u64>(imm_u1), seed);
    case Type::U8:
        return HashCombine(static_cast<u64>(imm_u8), seed);
    case Type::U16:
    case Type::F16:
        return HashCombine(static_cast<u64>(imm_u16), seed);
    case Type::U32:
    case Type::F32:
        return HashCombine(static_cast<u64>(imm_u32), seed);
    case Type::U64:
    case Type::F64:
        return HashCombine(imm_u64, seed);
    default:
        UNREACHABLE_MSG("hash of a {} value", type);
    }
}

bool Value::operator==(const Value& other) const {
    if (type != other.type) {
        return false;
    }
    switch (type) {
    case Type::Void:
        return true;
    case Type::Opaque:
        return inst == other.inst;
    case Type::ScalarReg:
        return sreg == other.sreg;
    case Type::VectorReg:
        return vreg == other.vreg;
    case Type::Attribute:
        return attribute == other.attribute;
    case Type::U1:
        return imm_u1 == other.imm_u1;
    case Type::U8:
        return imm_u8 == other.imm_u8;
    case Type::U16:
    case Type::F16:
        return imm_u16 == other.imm_u16;
    case Type::U32:
    case Type::F32:
        return imm_u32 == other.imm_u32;
    case Type::U64:
    case Type::F64:
        return imm_u64 == other.imm_u64;
    case Type::StringLiteral:
        return std::string_view{string_literal} == other.string_literal;
    default:
        UNREACHABLE_MSG("comparison of {} values", type);
    }
}

} // namespace Shader::IR
