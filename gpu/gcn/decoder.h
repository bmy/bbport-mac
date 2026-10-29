// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport gcn isa: decoding GCN3 instructions.
#pragma once

#include "gcn/isa.h"

namespace GcnIsa {

/// Decodes the instruction at `code` and advances `code` past it (literals and SDWA/DPP dwords
/// included).
GcnInst Decode(const u32*& code);

/// Called before an instruction the decoder does not know stops the program (`code`: where it
/// starts, `why`: the message): who decodes it says what program that was (diagnostics).
using DecodeFailure = void (*)(const u32* code, const char* why);
inline DecodeFailure decode_failure = nullptr;

/// Whether `opcode` (VOP2 or VOP3) also writes a scalar carry or a VCC result (VOP3B).
constexpr bool IsVop3B(Opcode opcode) noexcept {
    switch (opcode) {
    case Opcode::V_ADD_I32:
    case Opcode::V_ADDC_U32:
    case Opcode::V_SUB_I32:
    case Opcode::V_SUBB_U32:
    case Opcode::V_SUBREV_I32:
    case Opcode::V_SUBBREV_U32:
    case Opcode::V_DIV_SCALE_F32:
    case Opcode::V_DIV_SCALE_F64:
    case Opcode::V_MAD_U64_U32:
    case Opcode::V_MAD_I64_I32:
        return true;
    default:
        return false;
    }
}

} // namespace GcnIsa
