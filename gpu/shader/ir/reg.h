// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: registers (GCN scalar and vector registers, virtual registers of
// the translator) and the flag words of texture and buffer instructions.
#pragma once

#include <cstddef>
#include <type_traits>

#include <fmt/format.h>

#include "common/assert.h"
#include "common/bit_field.h"
#include "common/enum.h"
#include "common/types.h"
#include "gcn/formats.h"
#include "shader/ir/type.h"

namespace Shader::IR {

/// V_CMP_CLASS tests.
enum class FloatClassFunc : u32 {
    SignalingNan = 1 << 0,
    QuietNan = 1 << 1,
    NegativeInfinity = 1 << 2,
    NegativeNormal = 1 << 3,
    NegativeDenorm = 1 << 4,
    NegativeZero = 1 << 5,
    PositiveZero = 1 << 6,
    PositiveDenorm = 1 << 7,
    PositiveNormal = 1 << 8,
    PositiveInfinity = 1 << 9,

    NaN = SignalingNan | QuietNan,
    Infinity = PositiveInfinity | NegativeInfinity,
    Negative = NegativeInfinity | NegativeNormal | NegativeDenorm | NegativeZero,
    Finite = NegativeNormal | NegativeDenorm | NegativeZero | PositiveNormal | PositiveDenorm |
             PositiveZero,
};
DECLARE_ENUM_FLAG_OPERATORS(FloatClassFunc)

/// The flags of image instructions.
union TextureInstInfo {
    u32 raw;
    BitField<0, 1, u32> is_depth;
    BitField<1, 1, u32> has_bias;
    BitField<2, 1, u32> has_lod_clamp;
    BitField<3, 1, u32> force_level0;
    BitField<4, 1, u32> has_lod;
    BitField<5, 1, u32> has_offset;
    BitField<6, 2, u32> gather_comp;
    BitField<8, 1, u32> has_derivatives;
    BitField<9, 1, u32> is_array;
    BitField<10, 1, u32> is_unnormalized;
    BitField<11, 1, u32> is_gather;
    BitField<12, 1, u32> is_r128;
    BitField<16, 16, u32> pc;
};

/// The flags of buffer instructions.
union BufferInstInfo {
    u64 raw;
    BitField<0, 1, u64> index_enable;
    BitField<1, 1, u64> voffset_enable;
    BitField<2, 12, u64> inst_offset;
    BitField<14, 1, u64> system_coherent;
    BitField<15, 1, u64> globally_coherent;
    BitField<16, 1, u64> typed;
    BitField<17, 4, ::Gcn::DataFormat> inst_data_fmt;
    BitField<21, 3, ::Gcn::NumberFormat> inst_num_fmt;
    BitField<24, 1, u64> sharp_source;
    BitField<32, 16, u64> pc;
    BitField<48, 16, u64> flatbuf_off_dw;
};

enum class ScalarReg : u32 {
    S0,
    Max = 104,
};
constexpr size_t NumScalarRegs = static_cast<size_t>(ScalarReg::Max);

enum class VectorReg : u32 {
    V0,
    V1,
    V2,
    V3,
    V4,
    V5,
    V6,
    Max = 256,
};
constexpr size_t NumVectorRegs = static_cast<size_t>(VectorReg::Max);

/// A register of the translator: an index and a type.
struct VirtualReg {
    explicit VirtualReg() = default;
    explicit VirtualReg(u32 index_, Type type_) : index{index_}, type{type_} {
        ASSERT(type != IR::Type::Opaque);
    }

    auto operator<=>(const VirtualReg&) const noexcept = default;

    constexpr u64 Key() const noexcept {
        return static_cast<u64>(index) | static_cast<u64>(type) << 32;
    }

    u32 index{};
    Type type{};
};
static_assert(offsetof(VirtualReg, index) == 0, "RegTag reads the index of any register");

template <class T>
concept RegT = std::is_same_v<T, ScalarReg> || std::is_same_v<T, VectorReg>;

template <RegT Reg>
[[nodiscard]] constexpr Reg operator+(Reg reg, int num) {
    const int result = static_cast<int>(reg) + num;
    ASSERT_MSG(result >= 0 && result < static_cast<int>(Reg::Max), "register {} + {} out of range",
               static_cast<int>(reg), num);
    return static_cast<Reg>(result);
}

template <RegT Reg>
[[nodiscard]] constexpr Reg operator-(Reg reg, int num) {
    return reg + (-num);
}

template <RegT Reg>
constexpr Reg operator++(Reg& reg) {
    reg = reg + 1;
    return reg;
}

template <RegT Reg>
constexpr Reg operator++(Reg& reg, int) {
    const Reg copy = reg;
    reg = reg + 1;
    return copy;
}

template <RegT Reg>
[[nodiscard]] constexpr size_t RegIndex(Reg reg) noexcept {
    return static_cast<size_t>(reg);
}

} // namespace Shader::IR

template <>
struct fmt::formatter<Shader::IR::ScalarReg> : fmt::formatter<u32> {
    auto format(Shader::IR::ScalarReg reg, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "SGPR{}", static_cast<u32>(reg));
    }
};
template <>
struct fmt::formatter<Shader::IR::VectorReg> : fmt::formatter<u32> {
    auto format(Shader::IR::VectorReg reg, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "VGPR{}", static_cast<u32>(reg));
    }
};
template <>
struct fmt::formatter<Shader::IR::VirtualReg> : fmt::formatter<u32> {
    auto format(Shader::IR::VirtualReg reg, format_context& ctx) const {
        return fmt::format_to(ctx.out(), "REG{}_{}", reg.index, reg.type);
    }
};
