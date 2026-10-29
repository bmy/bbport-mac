// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: local common subexpression elimination for the few operations phi
// simplification leaves duplicated (ballots and the casts and unpacks of their results).
#pragma once

#include <algorithm>

#include <boost/container/small_vector.hpp>

#include "shader/ir/basic_block.h"
#include "shader/ir/microinstruction.h"

namespace Shader::Optimization {

inline void RunLocalCSE(IR::Block* block) {
    const auto eligible = [](const IR::Inst& inst) {
        switch (inst.GetOpcode()) {
        case IR::Opcode::Ballot:
        case IR::Opcode::UnpackUint2x32:
        case IR::Opcode::BitCastU32F32:
        case IR::Opcode::BitCastF32U32:
            return true;
        default:
            return false;
        }
    };
    const auto same = [](const IR::Inst& a, const IR::Inst& b) {
        if (a.GetOpcode() != b.GetOpcode()) {
            return false;
        }
        for (size_t i = 0; i < a.NumArgs(); ++i) {
            if (a.Arg(i) != b.Arg(i)) {
                return false;
            }
        }
        return true;
    };
    boost::container::small_vector<IR::Inst*, 8> seen;
    for (auto it = std::ranges::find_if_not(*block, IR::IsPhi); it != block->end(); ++it) {
        IR::Inst& inst = *it;
        if (!eligible(inst)) {
            continue;
        }
        const auto earlier =
            std::ranges::find_if(seen, [&](const IR::Inst* other) { return same(inst, *other); });
        if (earlier != seen.end()) {
            inst.ReplaceUsesWithAndRemove(IR::Value{*earlier});
        } else {
            seen.push_back(&inst);
        }
    }
}

} // namespace Shader::Optimization
