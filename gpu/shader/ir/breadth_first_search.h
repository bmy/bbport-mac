// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: searching the instructions a value is computed from, nearest
// first (rightmost arguments first at each step), for the first one `pred` returns a result for.
#pragma once

#include <algorithm>
#include <optional>
#include <queue>
#include <type_traits>

#include <boost/container/small_vector.hpp>

#include "shader/ir/microinstruction.h"
#include "shader/ir/value.h"

namespace Shader::IR {

/// `Instruction` is Inst or const Inst. `pred` returns an optional.
template <typename Instruction, typename Pred>
auto BreadthFirstSearch(Instruction* start, Pred&& pred)
    -> std::invoke_result_t<Pred, Instruction*> {
    if (auto result = pred(start)) {
        return result; // the common case
    }
    boost::container::small_vector<Instruction*, 2> visited;
    std::queue<Instruction*> queue;
    queue.push(start);
    while (!queue.empty()) {
        Instruction* const inst = queue.front();
        queue.pop();
        if (auto result = pred(inst)) {
            return result;
        }
        for (size_t arg = inst->NumArgs(); arg-- > 0;) {
            const Value value = inst->Arg(arg);
            if (value.IsImmediate()) {
                continue;
            }
            Instruction* const source = value.Inst();
            if (!std::ranges::contains(visited, source)) {
                visited.push_back(source);
                queue.push(source);
            }
        }
    }
    return std::nullopt;
}

template <typename Pred>
auto BreadthFirstSearch(const Value& value, Pred&& pred)
    -> std::invoke_result_t<Pred, const Inst*> {
    if (value.IsImmediate()) {
        return std::nullopt;
    }
    return BreadthFirstSearch(value.Inst(), pred);
}

template <typename Pred>
auto BreadthFirstSearch(Value value, Pred&& pred) -> std::invoke_result_t<Pred, Inst*> {
    if (value.IsImmediate()) {
        return std::nullopt;
    }
    return BreadthFirstSearch(value.Inst(), pred);
}

} // namespace Shader::IR
