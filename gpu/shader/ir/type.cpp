// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/type.h"

#include <array>
#include <bit>

namespace Shader::IR {

std::string NameOf(Type type) {
    static constexpr std::array<const char*, 27> names{
        "Opaque", "ScalarReg", "VectorReg", "VirtualReg", "Attribute", "Patch", "U1",
        "U8",     "U16",       "U32",       "U64",        "F16",       "F32",   "F64",
        "U32x2",  "U32x3",     "U32x4",     "F16x2",      "F16x3",     "F16x4", "F32x2",
        "F32x3",  "F32x4",     "F64x2",     "F64x3",      "F64x4",     "StringLiteral"};
    u32 bits = u32(type);
    if (bits == 0) {
        return "Void";
    }
    std::string result;
    while (bits != 0) {
        const u32 bit = u32(std::countr_zero(bits));
        bits &= bits - 1;
        if (!result.empty()) {
            result += '|';
        }
        result += bit < names.size() ? names[bit] : "?";
    }
    return result;
}

} // namespace Shader::IR
