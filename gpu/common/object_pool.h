// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: an arena for objects of one type that are all released at once (the translator's
// instructions and blocks, page table pages). Objects are placed one after another in chunks;
// ReleaseContents destroys them and keeps the chunks, so the next shader reuses the memory.
#pragma once

#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace Common {

template <typename T>
    requires std::is_destructible_v<T>
class ObjectPool {
public:
    explicit ObjectPool(std::size_t chunk_size = 1024) : chunk_objects{chunk_size ? chunk_size : 1} {}
    ObjectPool(const ObjectPool&) = delete;
    ObjectPool& operator=(const ObjectPool&) = delete;

    ~ObjectPool() {
        ReleaseContents();
    }

    template <typename... Args>
        requires std::is_constructible_v<T, Args...>
    [[nodiscard]] T* Create(Args&&... args) {
        if (used == chunk_objects || chunks.empty()) {
            NextChunk();
        }
        T* const slot = std::launder(reinterpret_cast<T*>(chunks[current].get()) + used);
        std::construct_at(slot, std::forward<Args>(args)...);
        ++used;
        return slot;
    }

    /// Destroys every object; the memory stays for the next ones.
    void ReleaseContents() {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            for (std::size_t chunk = 0; chunk < chunks.size() && chunk <= current; ++chunk) {
                T* const objects = std::launder(reinterpret_cast<T*>(chunks[chunk].get()));
                const std::size_t count = chunk == current ? used : chunk_objects;
                for (std::size_t i = 0; i < count; ++i) {
                    std::destroy_at(objects + i);
                }
            }
        }
        current = 0;
        used = chunks.empty() ? chunk_objects : 0;
    }

private:
    struct alignas(T) Slot {
        std::byte bytes[sizeof(T)];
    };

    void NextChunk() {
        if (!chunks.empty() && used == chunk_objects && current + 1 < chunks.size()) {
            ++current; // a chunk kept from before
        } else {
            chunks.push_back(std::make_unique_for_overwrite<Slot[]>(chunk_objects));
            current = chunks.size() - 1;
        }
        used = 0;
    }

    std::vector<std::unique_ptr<Slot[]>> chunks;
    std::size_t chunk_objects;
    std::size_t current = 0; ///< chunk being filled
    std::size_t used = 0;    ///< objects in it
};

} // namespace Common
