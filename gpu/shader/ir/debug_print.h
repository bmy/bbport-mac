// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: DebugPrint's flags (how many of its format arguments are used).
#pragma once

#include "common/bit_field.h"
#include "common/types.h"
#include "shader/ir/opcodes.h"

namespace Shader::IR {

constexpr size_t DEBUGPRINT_NUM_FORMAT_ARGS = NumArgsOf(IR::Opcode::DebugPrint) - 1;

union DebugPrintFlags {
    u32 raw;
    BitField<0, 32, u32> num_args;
};

} // namespace Shader::IR
