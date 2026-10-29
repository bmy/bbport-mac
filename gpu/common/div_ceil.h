// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: division rounding up.
#pragma once

#include <type_traits>

namespace Common {

/// number / divisor, rounded up.
template <typename N, typename D>
    requires std::is_integral_v<N> && std::is_unsigned_v<D>
[[nodiscard]] constexpr N DivCeil(N number, D divisor) {
    return static_cast<N>((static_cast<D>(number) + divisor - 1) / divisor);
}

/// value / 2^shift, rounded up.
template <typename N, typename D>
    requires std::is_integral_v<N> && std::is_unsigned_v<D>
[[nodiscard]] constexpr N DivCeilLog2(N value, D shift) {
    const D one = 1;
    return static_cast<N>((static_cast<D>(value) + (one << shift) - 1) >> shift);
}

} // namespace Common
