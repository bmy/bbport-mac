// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport memory: one bit per page of a tracked region (which pages the CPU or the GPU changed,
// which are protected). Ranges are half-open page indices; iterating a set of bits yields its
// runs of set pages.
#pragma once

#include <bit>
#include <cstddef>
#include <utility>

#include "common/types.h"

namespace BbMem {

template <size_t N>
class PageBits {
    static_assert(N % 64 == 0);
    static constexpr size_t Words = N / 64;

public:
    using Range = std::pair<size_t, size_t>;

    constexpr PageBits() = default;
    /// The bits of `other` within [start, end), the rest clear.
    constexpr PageBits(const PageBits& other, size_t start, size_t end) {
        if (start >= end || end > N) {
            return;
        }
        ForWords(start, end, [&](size_t i, u64 mask) { w[i] = other.w[i] & mask; });
    }
    constexpr PageBits(const PageBits&) = default;
    constexpr PageBits& operator=(const PageBits&) = default;

    constexpr void Set(size_t page) {
        w[page / 64] |= Bit(page);
    }
    constexpr void Unset(size_t page) {
        w[page / 64] &= ~Bit(page);
    }
    constexpr bool Get(size_t page) const {
        return w[page / 64] & Bit(page);
    }
    constexpr void SetRange(size_t start, size_t end) {
        ForWords(start, end, [&](size_t i, u64 mask) { w[i] |= mask; });
    }
    constexpr void UnsetRange(size_t start, size_t end) {
        ForWords(start, end, [&](size_t i, u64 mask) { w[i] &= ~mask; });
    }
    constexpr void Clear() {
        for (u64& word : w) {
            word = 0;
        }
    }
    constexpr void Fill() {
        for (u64& word : w) {
            word = ~0ull;
        }
    }
    constexpr bool Any() const {
        u64 any = 0;
        for (const u64 word : w) {
            any |= word;
        }
        return any != 0;
    }
    constexpr bool None() const {
        return !Any();
    }
    constexpr bool AnyInRange(size_t start, size_t end) const {
        bool any = false;
        ForWords(start, end, [&](size_t i, u64 mask) { any = any || (w[i] & mask); });
        return any;
    }
    constexpr bool AllInRange(size_t start, size_t end) const {
        bool all = true;
        ForWords(start, end, [&](size_t i, u64 mask) { all = all && (w[i] & mask) == mask; });
        return all;
    }

    /// The first run of set pages at or after `from` ({N, N}: none).
    constexpr Range NextRange(size_t from) const {
        if (from >= N) {
            return {N, N};
        }
        size_t i = from / 64;
        u64 word = w[i] & (~0ull << (from % 64));
        while (word == 0) {
            if (++i == Words) {
                return {N, N};
            }
            word = w[i];
        }
        const size_t start = i * 64 + std::countr_zero(word);
        // The run ends at the first clear bit after its start.
        u64 rest = ~w[i] & (~0ull << (start % 64));
        while (rest == 0) {
            if (++i == Words) {
                return {start, N};
            }
            rest = ~w[i];
        }
        return {start, i * 64 + std::countr_zero(rest)};
    }
    /// The last run of set pages ({0, 0}: none).
    constexpr Range LastRange() const {
        size_t i = Words;
        u64 word = 0;
        while (word == 0) {
            if (i == 0) {
                return {0, 0};
            }
            word = w[--i];
        }
        const size_t end = i * 64 + 64 - std::countl_zero(word);
        u64 rest = ~w[i] & (end % 64 ? (1ull << (end % 64)) - 1 : ~0ull);
        while (rest == 0) {
            if (i == 0) {
                return {0, end};
            }
            rest = ~w[--i];
        }
        return {i * 64 + 64 - std::countl_zero(rest), end};
    }
    constexpr Range FirstRange() const {
        return NextRange(0);
    }

    class Iterator {
    public:
        constexpr Iterator(const PageBits& bits_, Range range_) : bits{&bits_}, range{range_} {}
        constexpr const Range& operator*() const {
            return range;
        }
        constexpr Iterator& operator++() {
            range = bits->NextRange(range.second);
            return *this;
        }
        constexpr bool operator==(const Iterator& other) const {
            return range == other.range;
        }

    private:
        const PageBits* bits;
        Range range;
    };
    constexpr Iterator begin() const {
        return {*this, NextRange(0)};
    }
    constexpr Iterator end() const {
        return {*this, {N, N}};
    }

    constexpr PageBits& operator|=(const PageBits& o) {
        for (size_t i = 0; i < Words; ++i) {
            w[i] |= o.w[i];
        }
        return *this;
    }
    constexpr PageBits& operator&=(const PageBits& o) {
        for (size_t i = 0; i < Words; ++i) {
            w[i] &= o.w[i];
        }
        return *this;
    }
    constexpr PageBits& operator^=(const PageBits& o) {
        for (size_t i = 0; i < Words; ++i) {
            w[i] ^= o.w[i];
        }
        return *this;
    }
    constexpr PageBits operator|(const PageBits& o) const {
        PageBits r = *this;
        return r |= o;
    }
    constexpr PageBits operator&(const PageBits& o) const {
        PageBits r = *this;
        return r &= o;
    }
    constexpr PageBits operator^(const PageBits& o) const {
        PageBits r = *this;
        return r ^= o;
    }
    constexpr PageBits operator~() const {
        PageBits r;
        for (size_t i = 0; i < Words; ++i) {
            r.w[i] = ~w[i];
        }
        return r;
    }
    constexpr bool operator==(const PageBits&) const = default;

private:
    static constexpr u64 Bit(size_t page) {
        return 1ull << (page % 64);
    }
    /// Calls `f(word, mask)` for the words of [start, end) with the mask of its pages there.
    template <typename F>
    static constexpr void ForWords(size_t start, size_t end, F&& f) {
        if (start >= end) {
            return;
        }
        const size_t first = start / 64, last = (end - 1) / 64;
        const u64 head = ~0ull << (start % 64);
        const u64 tail = ~0ull >> (63 - (end - 1) % 64);
        if (first == last) {
            f(first, head & tail);
            return;
        }
        f(first, head);
        for (size_t i = first + 1; i < last; ++i) {
            f(i, ~0ull);
        }
        f(last, tail);
    }

    u64 w[Words]{};
};

} // namespace BbMem
