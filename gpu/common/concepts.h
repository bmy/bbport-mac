// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: small concepts.
#pragma once

#include <concepts>
#include <iterator>
#include <type_traits>

namespace Common {

template <typename T>
concept IsContiguousContainer = std::contiguous_iterator<typename T::iterator>;

template <typename Derived, typename Base>
concept DerivedFrom = std::derived_from<Derived, Base>;

template <typename From, typename To>
concept ConvertibleTo = std::is_convertible_v<From, To>;

template <typename T>
concept IsArithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept IsIntegral = std::is_integral_v<T>;

} // namespace Common
