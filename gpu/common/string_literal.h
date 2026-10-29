// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a string usable as a template argument.
#pragma once

#include <algorithm>
#include <cstddef>

template <std::size_t N, typename C = char>
struct StringLiteral {
    static constexpr std::size_t len = N;
    C value[N]{};

    constexpr StringLiteral(const C (&text)[N]) {
        std::copy(text, text + N, value);
    }
};
