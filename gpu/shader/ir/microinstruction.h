// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: an instruction: its operation, arguments (phis: one per
// predecessor block), flags, and the instructions using its result.
#pragma once

#include <array>
#include <bit>
#include <cstring>
#include <type_traits>
#include <utility>

#include <boost/container/list.hpp>
#include <boost/container/small_vector.hpp>
#include <boost/intrusive/list.hpp>

#include "common/assert.h"
#include "shader/ir/opcodes.h"
#include "shader/ir/ssa.h"
#include "shader/ir/value.h"

namespace Shader::IR {

/// An argument slot that refers to an instruction.
struct Use {
    Inst* user;
    u32 operand;

    Use() = default;
    Use(Inst* user_, u32 operand_) : user(user_), operand(operand_) {}
    bool operator==(const Use&) const noexcept = default;
};

class Inst : public boost::intrusive::list_base_hook<> {
public:
    explicit Inst(IR::Opcode op_, u32 flags_) noexcept;
    /// A copy of `base` (not of a phi) with the same arguments.
    explicit Inst(const Inst& base);
    ~Inst();

    Inst& operator=(const Inst&) = delete;
    Inst& operator=(Inst&&) = delete;
    Inst(Inst&&) = delete;

    IR::Block* GetParent() const {
        ASSERT(parent);
        return parent;
    }
    void SetParent(IR::Block* block) {
        parent = block;
    }

    int UseCount() const noexcept {
        return int(uses.size());
    }
    bool HasUses() const noexcept {
        return !uses.empty();
    }
    /// The instructions using the result, the newest use first.
    const auto& Uses() const {
        return uses;
    }

    IR::Opcode GetOpcode() const noexcept {
        return op;
    }
    size_t NumArgs() const {
        return op == IR::Opcode::Phi ? phi_args.size() : NumArgsOf(op);
    }
    Value Arg(size_t index) const noexcept {
        return op == IR::Opcode::Phi ? phi_args[index].second : args[index];
    }
    void SetArg(size_t index, Value value);

    /// Every use of the result takes `replacement`; this instruction becomes a Void.
    void ReplaceUsesWith(Value replacement) {
        ReplaceUses(replacement);
    }
    void ReplaceUsesWithAndRemove(Value replacement) {
        ReplaceUses(replacement);
    }

    template <typename FlagsType>
        requires(sizeof(FlagsType) <= sizeof(u64) && std::is_trivially_copyable_v<FlagsType>)
    [[nodiscard]] FlagsType Flags() const noexcept {
        FlagsType value;
        std::memcpy(&value, &flags, sizeof(value));
        return value;
    }
    template <typename FlagsType>
        requires(sizeof(FlagsType) <= sizeof(u64) && std::is_trivially_copyable_v<FlagsType>)
    void SetFlags(FlagsType value) noexcept {
        std::memcpy(&flags, &value, sizeof(value));
    }

    /// A 32-bit value a backend attaches (the SPIR-V id of the result).
    template <typename DefinitionType>
    void SetDefinition(DefinitionType def) {
        definition = std::bit_cast<u32>(def);
    }
    template <typename DefinitionType>
    [[nodiscard]] DefinitionType Definition() const noexcept {
        return std::bit_cast<DefinitionType>(definition);
    }

    void SetRegTag(RegTag new_tag) {
        reg_tag = new_tag;
    }
    RegTag GetRegTag() const {
        return reg_tag;
    }

    /// Writes memory, outputs or control state: never removed as unused.
    bool MayHaveSideEffects() const noexcept;
    bool AreAllArgsImmediates() const;

    IR::Type Type() const;
    /// Changes the operation (not into a phi), keeping the arguments.
    void ReplaceOpcode(IR::Opcode opcode);

    Block* PhiBlock(size_t index) const;
    void AddPhiOperand(Block* predecessor, const Value& value);

    /// Clears the arguments and becomes a Void.
    void Invalidate();
    void ClearArgs();

    u32 scc_index{}; ///< the phi's strongly connected component (phi simplification)

private:
    struct NonTriviallyDummy {
        NonTriviallyDummy() noexcept {}
    };

    void AddUse(Inst* used, u32 operand);
    void RemoveUse(Inst* used, u32 operand);
    void ReplaceUses(Value replacement);

    IR::Opcode op{};
    u32 definition{};
    u64 flags{};
    IR::Block* parent{};
    RegTag reg_tag{};
    union {
        NonTriviallyDummy dummy{};
        boost::container::small_vector<std::pair<Block*, Value>, 2> phi_args;
        std::array<Value, MaxArgs> args;
    };
    boost::container::list<IR::Use> uses;
};
static_assert(sizeof(Inst) <= 184, "Inst grew");

[[nodiscard]] inline bool IsPhi(const Inst& inst) {
    return inst.GetOpcode() == Opcode::Phi;
}

} // namespace Shader::IR
