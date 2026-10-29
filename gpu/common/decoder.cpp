// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "common/decoder.h"

namespace Common {

DecoderImpl::DecoderImpl() {
    ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64);
    ZydisFormatterInit(&formatter, ZYDIS_FORMATTER_STYLE_INTEL);
}

std::string DecoderImpl::disassembleInst(ZydisDecodedInstruction& inst,
                                         ZydisDecodedOperand* operands, u64 address) {
    char text[256];
    ZydisFormatterFormatInstruction(&formatter, &inst, operands, inst.operand_count_visible, text,
                                    sizeof(text), address, ZYAN_NULL);
    return text;
}

ZyanStatus DecoderImpl::decodeInstruction(ZydisDecodedInstruction& inst,
                                          ZydisDecodedOperand* operands, void* data, u64 size) {
    return ZydisDecoderDecodeFull(&decoder, data, size, &inst, operands);
}

} // namespace Common
