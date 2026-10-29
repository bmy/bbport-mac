// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// SMRD: scalar loads through a 64-bit pointer (ReadConst) or a V# (ReadConstBuffer), and the
// clock.
#include "shader/frontend/translate/translate.h"

namespace Shader::Gcn {

namespace {

/// The offset operand is a literal.
constexpr u32 SQ_SRC_LITERAL = 0xFF;

} // namespace

void Translator::EmitScalarMemory(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::S_LOAD_DWORD:
        return S_LOAD_DWORD(1, inst);
    case Opcode::S_LOAD_DWORDX2:
        return S_LOAD_DWORD(2, inst);
    case Opcode::S_LOAD_DWORDX4:
        return S_LOAD_DWORD(4, inst);
    case Opcode::S_LOAD_DWORDX8:
        return S_LOAD_DWORD(8, inst);
    case Opcode::S_LOAD_DWORDX16:
        return S_LOAD_DWORD(16, inst);
    case Opcode::S_BUFFER_LOAD_DWORD:
        return S_BUFFER_LOAD_DWORD(1, inst);
    case Opcode::S_BUFFER_LOAD_DWORDX2:
        return S_BUFFER_LOAD_DWORD(2, inst);
    case Opcode::S_BUFFER_LOAD_DWORDX4:
        return S_BUFFER_LOAD_DWORD(4, inst);
    case Opcode::S_BUFFER_LOAD_DWORDX8:
        return S_BUFFER_LOAD_DWORD(8, inst);
    case Opcode::S_BUFFER_LOAD_DWORDX16:
        return S_BUFFER_LOAD_DWORD(16, inst);
    case Opcode::S_MEMTIME:
        return S_MEMTIME(inst);
    default:
        LogMissingOpcode(inst);
    }
}

namespace {

/// The dword offset of an SMRD access: an immediate dword offset, a literal, or an SGPR in
/// bytes.
IR::U32 SmrdDwordOffset(IR::IREmitter& ir, const GcnInst& inst) {
    const auto& smrd = inst.control.smrd;
    if (smrd.imm) {
        return ir.Imm32(smrd.offset);
    }
    if (smrd.offset == SQ_SRC_LITERAL) {
        return ir.Imm32(inst.src[1].code);
    }
    return ir.ShiftRightLogical(ir.GetScalarReg(IR::ScalarReg(smrd.offset)), ir.Imm32(2));
}

} // namespace

void Translator::S_LOAD_DWORD(int num_dwords, const GcnInst& inst) {
    const IR::ScalarReg sbase{inst.src[0].code * 2};
    const IR::U32 dword_offset = SmrdDwordOffset(ir, inst);
    const IR::Value base =
        ir.CompositeConstruct(ir.GetScalarReg(sbase), ir.GetScalarReg(sbase + 1));
    const IR::ScalarReg dst{inst.dst[0].code};
    for (u32 i = 0; i < u32(num_dwords); i++) {
        const IR::U32 index = ir.IAdd(dword_offset, ir.Imm32(i));
        ir.SetScalarReg(dst + i, ir.ReadConst(base, index));
    }
}

void Translator::S_BUFFER_LOAD_DWORD(int num_dwords, const GcnInst& inst) {
    const IR::ScalarReg sbase{inst.src[0].code * 2};
    const IR::U32 dword_offset = SmrdDwordOffset(ir, inst);
    IR::BufferInstInfo buffer_info{};
    buffer_info.pc.Assign(pc);
    const IR::Value vsharp =
        ir.CompositeConstruct(ir.GetScalarReg(sbase), ir.GetScalarReg(sbase + 1),
                              ir.GetScalarReg(sbase + 2), ir.GetScalarReg(sbase + 3));
    const IR::ScalarReg dst{inst.dst[0].code};
    for (u32 i = 0; i < u32(num_dwords); i++) {
        const IR::U32 index = ir.IAdd(dword_offset, ir.Imm32(i));
        ir.SetScalarReg(dst + i, ir.ReadConstBuffer(vsharp, index, buffer_info));
    }
}

void Translator::S_MEMTIME(const GcnInst& inst) {
    SetDst64(inst.dst[0], ir.Memtime());
}

} // namespace Shader::Gcn
