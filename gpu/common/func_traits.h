// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: return and argument types of functions and lambdas.
#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace Common {

template <class Func>
struct FuncTraits {};

template <class Result, class... Args>
struct FuncTraits<Result (*)(Args...)> {
    using ReturnType = Result;
    static constexpr std::size_t NUM_ARGS = sizeof...(Args);
    template <std::size_t I>
    using ArgType = std::tuple_element_t<I, std::tuple<Args...>>;
};

template <typename Func>
struct LambdaTraits : LambdaTraits<decltype(&std::remove_reference_t<Func>::operator())> {};

template <typename Result, typename Lambda, typename... Args>
struct LambdaTraits<Result (Lambda::*)(Args...) const> {
    static constexpr std::size_t NUM_ARGS = sizeof...(Args);
    template <std::size_t I>
    using ArgType = std::tuple_element_t<I, std::tuple<Args...>>;
};

} // namespace Common
