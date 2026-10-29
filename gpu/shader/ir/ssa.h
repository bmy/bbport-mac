// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: what an instruction's result stands for (a GCN register, a goto
// variable of the structurizer, a virtual register), and the per-block values of the registers
// during SSA construction.
#pragma once

#include <array>
#include <cstring>

#include "common/assert.h"
#include "shader/ir/reg.h"
#include "shader/ir/value.h"

namespace Shader::IR {

/// Register kinds; the intrusive ones (below IntrusiveEnd) have a slot each in SsaState.
enum class RegType {
    ScalarReg,
    VectorReg = ScalarReg + NumScalarRegs,
    Scc = VectorReg + NumVectorRegs,
    Exec,
    VccLo,
    VccHi,
    M0,
    IntrusiveEnd,
    GotoVariable,
    VirtualReg,
    None,
};

struct SsaState {
    static constexpr size_t NumValues = static_cast<size_t>(RegType::IntrusiveEnd);
    std::array<Value, NumValues> values;

    void Reset() {
        std::memset(static_cast<void*>(this), 0, sizeof(SsaState));
    }
};

struct RegTag {
    RegType type{RegType::None};
    union {
        u32 index{};
        IR::ScalarReg sreg;
        IR::VectorReg vreg;
        IR::VirtualReg reg;
    };

    RegTag() = default;
    RegTag(IR::ScalarReg reg) : type{RegType::ScalarReg}, sreg{reg} {}
    RegTag(IR::VectorReg reg) : type{RegType::VectorReg}, vreg{reg} {}
    RegTag(IR::VirtualReg reg_) : type{RegType::VirtualReg}, reg{reg_} {}
    RegTag(RegType type_, u32 index_ = 0) : type{type_}, index{index_} {}

    bool IsIntrusive() const {
        return type < RegType::IntrusiveEnd;
    }
    /// The slot in SsaState.
    u32 Index() const {
        DEBUG_ASSERT(IsIntrusive());
        return static_cast<u32>(type) + index;
    }
    bool operator==(const RegTag& other) const {
        return type == other.type && index == other.index;
    }
    explicit operator bool() const {
        return type != RegType::None;
    }
};

static constexpr RegTag EMPTY_REG_TAG{};

} // namespace Shader::IR
