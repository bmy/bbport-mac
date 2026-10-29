// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: x86-64 instruction decoding (Zydis) for fault handlers that need to know what a
// faulting instruction did.
#pragma once

#include <string>

#include <Zydis/Zydis.h>

#include "common/singleton.h"
#include "common/types.h"

namespace Common {

class DecoderImpl {
public:
    DecoderImpl();

    std::string disassembleInst(ZydisDecodedInstruction& inst, ZydisDecodedOperand* operands,
                                u64 address);
    ZyanStatus decodeInstruction(ZydisDecodedInstruction& inst, ZydisDecodedOperand* operands,
                                 void* data, u64 size = 15);

private:
    ZydisDecoder decoder;
    ZydisFormatter formatter;
};

using Decoder = Singleton<DecoderImpl>;

} // namespace Common
