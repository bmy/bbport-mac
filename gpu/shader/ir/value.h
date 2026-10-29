// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: an operand: an instruction's result, a register, an attribute or
// patch value, or an immediate. TypedValue restricts the types a function accepts.
#pragma once

#include <functional>
#include <type_traits>

#include "common/assert.h"
#include "shader/ir/attribute.h"
#include "shader/ir/patch.h"
#include "shader/ir/reg.h"
#include "shader/ir/type.h"

namespace Shader::IR {

class Block;
class Inst;

class Value {
public:
    Value() noexcept = default;
    explicit Value(IR::Inst* value) noexcept : type{Type::Opaque}, inst{value} {}
    explicit Value(IR::ScalarReg reg) noexcept : type{Type::ScalarReg}, sreg{reg} {}
    explicit Value(IR::VectorReg reg) noexcept : type{Type::VectorReg}, vreg{reg} {}
    explicit Value(IR::VirtualReg reg_) noexcept : type{Type::VirtualReg}, reg{reg_} {}
    explicit Value(IR::Attribute value) noexcept : type{Type::Attribute}, attribute{value} {}
    explicit Value(IR::Patch value) noexcept : type{Type::Patch}, patch{value} {}
    explicit Value(bool value) noexcept : type{Type::U1}, imm_u1{value} {}
    explicit Value(u8 value) noexcept : type{Type::U8}, imm_u8{value} {}
    explicit Value(u16 value) noexcept : type{Type::U16}, imm_u16{value} {}
    explicit Value(u32 value) noexcept : type{Type::U32}, imm_u32{value} {}
    explicit Value(f32 value) noexcept : type{Type::F32}, imm_f32{value} {}
    explicit Value(u64 value) noexcept : type{Type::U64}, imm_u64{value} {}
    explicit Value(f64 value) noexcept : type{Type::F64}, imm_f64{value} {}
    explicit Value(const char* value) noexcept : type{Type::StringLiteral}, string_literal{value} {}

    /// Whether this is the result of a phi.
    bool IsPhi() const noexcept;
    /// The type; of an instruction's result, the instruction's.
    IR::Type Type() const noexcept;
    u64 Hash() const noexcept;

    bool IsEmpty() const noexcept {
        return type == Type::Void;
    }
    /// Anything but an instruction's result.
    bool IsImmediate() const noexcept {
        return type != Type::Opaque;
    }

    IR::Inst* Inst() const {
        DEBUG_ASSERT(type == Type::Opaque);
        return inst;
    }
    IR::Inst* TryInst() const {
        return type == Type::Opaque ? inst : nullptr;
    }
    IR::ScalarReg ScalarReg() const {
        DEBUG_ASSERT(type == Type::ScalarReg);
        return sreg;
    }
    IR::VectorReg VectorReg() const {
        DEBUG_ASSERT(type == Type::VectorReg);
        return vreg;
    }
    IR::VirtualReg VirtualReg() const {
        DEBUG_ASSERT(type == Type::VirtualReg);
        return reg;
    }
    IR::Attribute Attribute() const {
        DEBUG_ASSERT(type == Type::Attribute);
        return attribute;
    }
    IR::Patch Patch() const {
        DEBUG_ASSERT(type == Type::Patch);
        return patch;
    }
    bool U1() const {
        DEBUG_ASSERT(type == Type::U1);
        return imm_u1;
    }
    u8 U8() const {
        DEBUG_ASSERT(type == Type::U8);
        return imm_u8;
    }
    u16 U16() const {
        DEBUG_ASSERT(type == Type::U16);
        return imm_u16;
    }
    u32 U32() const {
        DEBUG_ASSERT(type == Type::U32);
        return imm_u32;
    }
    f32 F32() const {
        DEBUG_ASSERT(type == Type::F32);
        return imm_f32;
    }
    u64 U64() const {
        DEBUG_ASSERT(type == Type::U64);
        return imm_u64;
    }
    f64 F64() const {
        DEBUG_ASSERT(type == Type::F64);
        return imm_f64;
    }
    const char* StringLiteral() const {
        DEBUG_ASSERT(type == Type::StringLiteral);
        return string_literal;
    }

    [[nodiscard]] bool operator==(const Value& other) const;
    [[nodiscard]] bool operator!=(const Value& other) const {
        return !(*this == other);
    }

private:
    IR::Type type{};
    union {
        IR::Inst* inst{};
        IR::ScalarReg sreg;
        IR::VectorReg vreg;
        IR::VirtualReg reg;
        IR::Attribute attribute;
        IR::Patch patch;
        bool imm_u1;
        u8 imm_u8;
        u16 imm_u16;
        u32 imm_u32;
        f32 imm_f32;
        u64 imm_u64;
        f64 imm_f64;
        const char* string_literal;
    };
};
static_assert(static_cast<u32>(IR::Type::Void) == 0, "zeroed values are empty");
static_assert(std::is_trivially_copyable_v<Value>);

/// A value of one of the types in `type_` (checked when converting from a plain Value).
template <IR::Type type_>
class TypedValue : public Value {
public:
    TypedValue() = default;

    template <IR::Type other_type>
        requires((other_type & type_) != IR::Type::Void)
    TypedValue(const TypedValue<other_type>& value) : Value(value) {}

    explicit TypedValue(const Value& value) : Value(value) {
        ASSERT_MSG((value.Type() & type_) != IR::Type::Void, "Incompatible types {} and {}", type_,
                   value.Type());
    }

    explicit TypedValue(IR::Inst* inst_) : TypedValue(Value(inst_)) {}
};

using U1 = TypedValue<Type::U1>;
using U8 = TypedValue<Type::U8>;
using U16 = TypedValue<Type::U16>;
using U32 = TypedValue<Type::U32>;
using U64 = TypedValue<Type::U64>;
using F16 = TypedValue<Type::F16>;
using F32 = TypedValue<Type::F32>;
using F64 = TypedValue<Type::F64>;
using U32F32 = TypedValue<Type::U32 | Type::F32>;
using U64F64 = TypedValue<Type::U64 | Type::F64>;
using U32U64 = TypedValue<Type::U32 | Type::U64>;
using U16U32U64 = TypedValue<Type::U16 | Type::U32 | Type::U64>;
using U8U16U32U64 = TypedValue<Type::U8 | Type::U16 | Type::U32 | Type::U64>;
using F32F64 = TypedValue<Type::F32 | Type::F64>;
using F16F32F64 = TypedValue<Type::F16 | Type::F32 | Type::F64>;
using UAny = TypedValue<Type::U8 | Type::U16 | Type::U32 | Type::U64>;

} // namespace Shader::IR

template <>
struct std::hash<Shader::IR::Value> {
    std::size_t operator()(const Shader::IR::Value& v) const {
        return v.Hash();
    }
};
