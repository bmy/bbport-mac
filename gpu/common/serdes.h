// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the byte format of the pipeline cache's entries. A value is its bytes; a container is
// its element count (size_t) followed by its elements; a string is its length (size_t) followed
// by its characters. Entries already on disk depend on exactly this layout.
#pragma once

#include <concepts>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/types.h"

namespace Serialization {

/// An entry that cannot be used: shorter than what it claims to hold (a file cut short by a
/// crash or a power loss, issue #28), or rejected by the driver while preloading. The pipeline
/// cache drops the entry and compiles it again instead of stopping the game.
struct CorruptData : std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <typename T>
concept Container = requires(T t) {
    typename T::iterator;
    { t.begin() } -> std::same_as<typename T::iterator>;
    { t.end() } -> std::same_as<typename T::iterator>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

struct Archive {
    Archive() = default;
    explicit Archive(std::vector<u8>&& bytes_) : bytes{std::move(bytes_)} {}

    [[nodiscard]] std::size_t SizeBytes() const {
        return bytes.size();
    }
    [[nodiscard]] bool IsEoS() const {
        return offset >= bytes.size();
    }
    /// The bytes written (the archive is empty afterwards).
    std::vector<u8>&& TakeOff() {
        offset = 0;
        return std::move(bytes);
    }

    /// Room for `size` more bytes (the bytes written do not change).
    void Grow(std::size_t size) {
        bytes.reserve(bytes.size() + size);
    }
    /// Reading in place: the bytes at the read position, then skipping `size` of them.
    [[nodiscard]] u8* CurrPtr() {
        return bytes.data() + offset;
    }
    void Advance(std::size_t size) {
        if (size > Left()) {
            Corrupt();
        }
        offset += size;
    }

private:
    [[noreturn]] static void Corrupt() {
        throw CorruptData{"Invalid or corrupted deserialization container/shader cache"};
    }
    std::size_t Left() const {
        return bytes.size() - offset;
    }

    std::size_t offset = 0;
    std::vector<u8> bytes;

    friend struct Writer;
    friend struct Reader;
};

struct Writer {
    explicit Writer(Archive& ar_) : ar{ar_} {}

    template <typename T>
    void Write(const T* data, std::size_t size) {
        if (ar.offset + size > ar.bytes.size()) {
            ar.bytes.resize(ar.offset + size);
        }
        std::memcpy(ar.bytes.data() + ar.offset, data, size);
        ar.offset += size;
    }

    template <typename T>
    void Write(const T& value) {
        if constexpr (std::is_same_v<T, std::string>) {
            Write(value.size());
            Write(value.data(), value.size());
        } else if constexpr (Container<T>) {
            Write(value.size());
            for (const auto& element : value) {
                Write(element);
            }
        } else {
            Write(&value, sizeof(value));
        }
    }

    Archive& ar;
};

struct Reader {
    explicit Reader(Archive& ar_) : ar{ar_} {}

    template <typename T>
    void Read(T* data, std::size_t size) {
        if (size > ar.Left()) {
            Archive::Corrupt();
        }
        std::memcpy(data, ar.bytes.data() + ar.offset, size);
        ar.offset += size;
    }

    template <typename T>
    void Read(T& value) {
        if constexpr (std::is_same_v<T, std::string>) {
            std::size_t length{};
            Read(length);
            if (length > ar.Left()) {
                Archive::Corrupt();
            }
            value.resize(length);
            Read(value.data(), length);
        } else if constexpr (Container<T>) {
            std::size_t count{};
            Read(count);
            if (count > ar.Left()) { // every element takes at least a byte
                Archive::Corrupt();
            }
            for (std::size_t i = 0; i < count; ++i) {
                value.emplace_back();
                Read(value.back());
            }
        } else {
            Read(&value, sizeof(value));
        }
    }

    Archive& ar;
};

} // namespace Serialization
