// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: a bit range of an integer, for unions describing hardware registers and packets.
//
//   union Reg {
//       u32 raw;
//       BitField<0, 4, u32> low;      // bits 0..3
//       BitField<4, 12, s32> offset;  // bits 4..15, sign-extended when read
//   };
//
// Reading converts to T (enums too); writing goes through Assign. The member is the whole
// integer, so the union stays its size, and byte-packed (the packets put fields at any offset).
#pragma once

#include <cstddef>
#include <type_traits>

#pragma pack(push, 1)
template <std::size_t Position, std::size_t Bits, typename T>
struct BitField {
    using Type = T;
    using UnderlyingType =
        typename std::conditional_t<std::is_enum_v<T>, std::underlying_type<T>, std::type_identity<T>>::type;
    using StorageType = std::make_unsigned_t<UnderlyingType>;

    static constexpr std::size_t position = Position;
    static constexpr std::size_t bits = Bits;
    static constexpr std::size_t width = 8 * sizeof(StorageType);

    static_assert(Bits > 0 && Position < width && Position + Bits <= width, "field outside the integer");
    static_assert(std::is_trivially_copyable_v<T>, "a BitField's type must be trivially copyable");

    /// The field's bits within the integer.
    static constexpr StorageType mask =
        static_cast<StorageType>((Bits == width ? ~StorageType{0} : (StorageType{1} << Bits) - 1)
                                 << Position);

    /// `value` placed at the field's bits (others zero).
    [[nodiscard]] static constexpr StorageType FormatValue(const T& value) {
        return static_cast<StorageType>(static_cast<StorageType>(value) << Position) & mask;
    }

    /// The field's value in `storage`; signed types are sign-extended from the field's top bit.
    [[nodiscard]] static constexpr T ExtractValue(const StorageType& storage) {
        if constexpr (std::is_signed_v<UnderlyingType>) {
            // Move the field to the top, then shift arithmetically back down.
            constexpr std::size_t up = width - Position - Bits;
            const auto top = static_cast<UnderlyingType>(static_cast<StorageType>(storage << up));
            return static_cast<T>(static_cast<UnderlyingType>(top >> (width - Bits)));
        } else {
            return static_cast<T>((storage & mask) >> Position);
        }
    }

    constexpr BitField() noexcept = default;
    constexpr BitField(const BitField&) noexcept = default;
    constexpr BitField(BitField&&) noexcept = default;
    constexpr BitField& operator=(const BitField&) noexcept = default;
    constexpr BitField& operator=(BitField&&) noexcept = default;
    // A value would set only this field of the union, or all of it: neither is meant. Assign.
    BitField(T) = delete;
    BitField& operator=(T) = delete;

    [[nodiscard]] constexpr T Value() const {
        return ExtractValue(storage);
    }
    [[nodiscard]] constexpr operator T() const {
        return Value();
    }
    [[nodiscard]] constexpr explicit operator bool() const {
        return (storage & mask) != 0;
    }
    constexpr void Assign(const T& value) {
        storage = static_cast<StorageType>((storage & ~mask) | FormatValue(value));
    }

private:
    StorageType storage;
};
#pragma pack(pop)
