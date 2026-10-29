// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a move-only callable (what std::function would be without copies). Closures up to
// InlineSize bytes live inside the object: queued GPU commands and submit callbacks are created
// thousands of times a frame, and each of them was a heap allocation before.
#pragma once

#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

namespace Common {

template <typename Result, typename... Args>
class UniqueFunction {
    static constexpr std::size_t InlineSize = 48;
    static constexpr std::size_t InlineAlign = alignof(std::max_align_t);

    struct Ops {
        Result (*call)(void* object, Args&&... args);
        void (*move)(void* from, void* to); ///< construct at `to` from `from`, destroy `from`
        void (*destroy)(void* object);
    };

    template <typename F>
    static constexpr bool fits_inline = sizeof(F) <= InlineSize && alignof(F) <= InlineAlign &&
                                        std::is_nothrow_move_constructible_v<F>;

    template <typename F>
    static const Ops* InlineOps() {
        static constexpr Ops ops{
            [](void* object, Args&&... args) -> Result {
                return (*static_cast<F*>(object))(std::forward<Args>(args)...);
            },
            [](void* from, void* to) {
                new (to) F(std::move(*static_cast<F*>(from)));
                static_cast<F*>(from)->~F();
            },
            [](void* object) { static_cast<F*>(object)->~F(); },
        };
        return &ops;
    }

    template <typename F>
    static const Ops* HeapOps() {
        static constexpr Ops ops{
            [](void* object, Args&&... args) -> Result {
                return (**static_cast<F**>(object))(std::forward<Args>(args)...);
            },
            [](void* from, void* to) { *static_cast<F**>(to) = *static_cast<F**>(from); },
            [](void* object) { delete *static_cast<F**>(object); },
        };
        return &ops;
    }

public:
    UniqueFunction() = default;

    template <typename F>
        requires(!std::is_same_v<std::decay_t<F>, UniqueFunction>)
    UniqueFunction(F&& f) {
        using Fn = std::decay_t<F>;
        if constexpr (fits_inline<Fn>) {
            new (storage) Fn(std::forward<F>(f));
            ops = InlineOps<Fn>();
        } else {
            *reinterpret_cast<Fn**>(storage) = new Fn(std::forward<F>(f));
            ops = HeapOps<Fn>();
        }
    }

    UniqueFunction(UniqueFunction&& other) noexcept {
        TakeFrom(other);
    }
    UniqueFunction& operator=(UniqueFunction&& other) noexcept {
        if (this != &other) {
            Reset();
            TakeFrom(other);
        }
        return *this;
    }
    UniqueFunction(const UniqueFunction&) = delete;
    UniqueFunction& operator=(const UniqueFunction&) = delete;

    ~UniqueFunction() {
        Reset();
    }

    Result operator()(Args&&... args) const {
        return ops->call(const_cast<std::byte*>(storage), std::forward<Args>(args)...);
    }

    explicit operator bool() const noexcept {
        return ops != nullptr;
    }

private:
    void Reset() {
        if (ops) {
            ops->destroy(storage);
            ops = nullptr;
        }
    }
    void TakeFrom(UniqueFunction& other) {
        if (other.ops) {
            other.ops->move(other.storage, storage);
            ops = std::exchange(other.ops, nullptr);
        }
    }

    alignas(InlineAlign) std::byte storage[InlineSize];
    const Ops* ops = nullptr;
};

} // namespace Common
