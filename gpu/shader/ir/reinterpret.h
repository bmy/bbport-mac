// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: component swizzles, sRGB decoding and the number conversions of
// buffer formats (applied when reading, inverted when writing).
#pragma once

#include <limits>

#include "gcn/formats.h"
#include "shader/ir/ir_emitter.h"

namespace Shader::IR {

/// A vec4 swizzled: selectors 0 and 1 are the constants, 4..7 the components.
inline Value ApplySwizzle(IREmitter& ir, const Value& vector, const ::Gcn::CompMapping& swizzle) {
    const auto zero = ir.Imm32(0.f);
    const auto one = ir.Imm32(1.f);
    const auto constants = ir.CompositeConstruct(zero, one, zero, zero);
    return ir.CompositeShuffle(constants, vector, size_t(swizzle.r), size_t(swizzle.g),
                               size_t(swizzle.b), size_t(swizzle.a));
}

/// An sRGB-encoded value decoded to linear.
inline F32 ApplyGammaToLinear(IREmitter& ir, const F32& c) {
    const F32 curve =
        ir.FPPow(ir.FPMul(ir.FPAdd(c, ir.Imm32(0.055f)), ir.Imm32(1.0f / 1.055f)), ir.Imm32(2.4f));
    const F32 linear = ir.FPMul(c, ir.Imm32(1.0f / 12.92f));
    return F32{ir.Select(ir.FPGreaterThan(c, ir.Imm32(0.04045f)), curve, linear)};
}

/// The color channels of a vec4 decoded from sRGB (alpha is linear).
inline Value ApplyForceDegamma(IREmitter& ir, const Value& value) {
    auto x = F32{ir.CompositeExtract(value, 0)};
    auto y = F32{ir.CompositeExtract(value, 1)};
    auto z = F32{ir.CompositeExtract(value, 2)};
    auto w = F32{ir.CompositeExtract(value, 3)};
    x = ApplyGammaToLinear(ir, x);
    y = ApplyGammaToLinear(ir, y);
    z = ApplyGammaToLinear(ir, z);
    return ir.CompositeConstruct(x, y, z, w);
}

/// A loaded value converted to what the format means.
inline F32 ApplyReadNumberConversion(IREmitter& ir, const F32& value,
                                     const ::Gcn::NumberConversion& conversion) {
    using ::Gcn::NumberConversion;
    switch (conversion) {
    case NumberConversion::None:
        return value;
    case NumberConversion::UintToUscaled:
        return ir.ConvertUToF(32, 32, ir.BitCast<U32>(value));
    case NumberConversion::SintToSscaled:
        return ir.ConvertSToF(32, 32, ir.BitCast<U32>(value));
    case NumberConversion::UnormToUbnorm:
        return ir.FPSub(ir.FPMul(value, ir.Imm32(2.f)), ir.Imm32(1.f)); // 0..1 to -1..1
    case NumberConversion::Sint8ToSnormNz: {
        // (2v + 1) / 255: no value maps to zero.
        const U32 odd = ir.IAdd(ir.IMul(ir.BitCast<U32>(value), ir.Imm32(2)), ir.Imm32(1));
        const F32 numerator = ir.ConvertSToF(32, 32, odd);
        const F32 max = ir.Imm32(float(std::numeric_limits<u8>::max()));
        return ir.FPDiv(numerator, max);
    }
    case NumberConversion::Sint16ToSnormNz: {
        const U32 odd = ir.IAdd(ir.IMul(ir.BitCast<U32>(value), ir.Imm32(2)), ir.Imm32(1));
        const F32 numerator = ir.ConvertSToF(32, 32, odd);
        const F32 max = ir.Imm32(float(std::numeric_limits<u16>::max()));
        return ir.FPDiv(numerator, max);
    }
    case NumberConversion::Uint32ToUnorm: {
        const auto as_float = ir.ConvertUToF(32, 32, ir.BitCast<U32>(value));
        return ir.FPDiv(as_float, ir.Imm32(static_cast<float>(std::numeric_limits<u32>::max())));
    }
    case NumberConversion::SrgbToNorm:
        return ApplyGammaToLinear(ir, value);
    default:
        UNREACHABLE_MSG("number conversion {}", u32(conversion));
    }
}

inline Value ApplyReadNumberConversionVec4(IREmitter& ir, const Value& value,
                                           const ::Gcn::NumberConversion& conversion) {
    if (conversion == ::Gcn::NumberConversion::None) {
        return value;
    }
    const auto x = ApplyReadNumberConversion(ir, F32{ir.CompositeExtract(value, 0)}, conversion);
    const auto y = ApplyReadNumberConversion(ir, F32{ir.CompositeExtract(value, 1)}, conversion);
    const auto z = ApplyReadNumberConversion(ir, F32{ir.CompositeExtract(value, 2)}, conversion);
    const auto w = ApplyReadNumberConversion(ir, F32{ir.CompositeExtract(value, 3)}, conversion);
    return ir.CompositeConstruct(x, y, z, w);
}

/// A value to store converted to the format's encoding (kept as F32 bits for the IR).
inline F32 ApplyWriteNumberConversion(IREmitter& ir, const F32& value,
                                      const ::Gcn::NumberConversion& conversion) {
    using ::Gcn::NumberConversion;
    switch (conversion) {
    case NumberConversion::None:
        return value;
    case NumberConversion::UintToUscaled:
        return ir.BitCast<F32>(U32{ir.ConvertFToU(32, value)});
    case NumberConversion::SintToSscaled:
        return ir.BitCast<F32>(U32{ir.ConvertFToS(32, value)});
    case NumberConversion::UnormToUbnorm:
        return ir.FPDiv(ir.FPAdd(value, ir.Imm32(1.f)), ir.Imm32(2.f)); // -1..1 to 0..1
    case NumberConversion::Sint8ToSnormNz: {
        const F32 max = ir.Imm32(float(std::numeric_limits<u8>::max()));
        const F32 scaled = ir.FPMul(ir.FPClamp(value, ir.Imm32(-1.f), ir.Imm32(1.f)), max);
        const F32 shifted = ir.FPSub(scaled, ir.Imm32(1.f));
        const U32 raw = ir.ConvertFToS(32, ir.FPDiv(shifted, ir.Imm32(2.f)));
        return ir.BitCast<F32>(raw);
    }
    case NumberConversion::Sint16ToSnormNz: {
        const F32 max = ir.Imm32(float(std::numeric_limits<u16>::max()));
        const F32 scaled = ir.FPMul(ir.FPClamp(value, ir.Imm32(-1.f), ir.Imm32(1.f)), max);
        const F32 shifted = ir.FPSub(scaled, ir.Imm32(1.f));
        const U32 raw = ir.ConvertFToS(32, ir.FPDiv(shifted, ir.Imm32(2.f)));
        return ir.BitCast<F32>(raw);
    }
    case NumberConversion::Uint32ToUnorm: {
        const auto clamped = ir.FPClamp(value, ir.Imm32(0.f), ir.Imm32(1.f));
        const auto scaled =
            ir.FPMul(clamped, ir.Imm32(static_cast<float>(std::numeric_limits<u32>::max())));
        return ir.BitCast<F32>(U32{ir.ConvertFToU(32, scaled)});
    }
    default:
        UNREACHABLE_MSG("number conversion {}", u32(conversion));
    }
}

inline Value ApplyWriteNumberConversionVec4(IREmitter& ir, const Value& value,
                                            const ::Gcn::NumberConversion& conversion) {
    if (conversion == ::Gcn::NumberConversion::None) {
        return value;
    }
    const auto x = ApplyWriteNumberConversion(ir, F32{ir.CompositeExtract(value, 0)}, conversion);
    const auto y = ApplyWriteNumberConversion(ir, F32{ir.CompositeExtract(value, 1)}, conversion);
    const auto z = ApplyWriteNumberConversion(ir, F32{ir.CompositeExtract(value, 2)}, conversion);
    const auto w = ApplyWriteNumberConversion(ir, F32{ir.CompositeExtract(value, 3)}, conversion);
    return ir.CompositeConstruct(x, y, z, w);
}

} // namespace Shader::IR
