// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator IR: argument positions of buffer instructions, and the parts of their
// address (index, vector offset, scalar offset: a U32x3 until resources are resolved).
#pragma once

#include "common/assert.h"
#include "shader/ir/microinstruction.h"
#include "shader/ir/value.h"

namespace Shader::IR {

namespace LoadBufferArgs {
constexpr size_t Handle = 0;
constexpr size_t Address = 1;
} // namespace LoadBufferArgs

namespace StoreBufferArgs {
constexpr size_t Handle = 0;
constexpr size_t Address = 1;
constexpr size_t Data = 2;
} // namespace StoreBufferArgs

static_assert(LoadBufferArgs::Handle == StoreBufferArgs::Handle);
static_assert(LoadBufferArgs::Address == StoreBufferArgs::Address);

inline U32 GetBufferAddressComponent(const Inst* buffer_inst, u32 component) {
    const Inst* address = buffer_inst->Arg(LoadBufferArgs::Address).Inst();
    ASSERT(address->GetOpcode() == Opcode::CompositeConstructU32x3);
    return U32{address->Arg(component)};
}
inline U32 GetBufferIndexArg(const Inst* buffer_inst) {
    return GetBufferAddressComponent(buffer_inst, 0);
}
inline U32 GetBufferVOffsetArg(const Inst* buffer_inst) {
    return GetBufferAddressComponent(buffer_inst, 1);
}
inline U32 GetBufferSOffsetArg(const Inst* buffer_inst) {
    return GetBufferAddressComponent(buffer_inst, 2);
}

} // namespace Shader::IR
