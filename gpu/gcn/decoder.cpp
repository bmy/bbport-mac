// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "gcn/decoder.h"

#include <algorithm>
#include <bit>

#include "common/logging/log.h"

namespace GcnIsa {

namespace {

/// Bits [first, last] of `value`.
template <typename T>
constexpr u32 Bits(T value, u32 first, u32 last) {
    return u32((value >> first) & ((T{1} << (last - first + 1)) - 1));
}

/// The number of `opcode` in its own encoding (the unified opcode minus the encoding's base).
constexpr u32 Number(Opcode opcode, u32 base) {
    return u32(opcode) - base;
}

/// A source register operand (VGPR codes are 256..511 in 9-bit fields).
void SetSource(InstOperand& operand, u32 code) {
    operand.field = FieldOf(code);
    operand.code = operand.field == OperandField::VectorGPR ? code - VectorGPRMin : code;
}

void Set(InstOperand& operand, OperandField field, u32 code) {
    operand.field = field;
    operand.code = code;
}

/// VOPC (and VOP2 VOP3B) results go to VCC, kept as the second destination as in VOP3.
void SetVccDestination(GcnInst& inst) {
    inst.dst[1].field = OperandField::VccLo;
    inst.dst[1].type = ScalarType::Uint64;
    inst.dst[1].code = u32(OperandField::VccLo);
}

float OutputMultiplier(u32 omod) {
    constexpr std::array multipliers{0.f, 2.0f, 4.0f, 0.5f};
    return multipliers[omod & 3];
}

template <typename Control, typename Raw>
Control ControlOf(Raw raw) {
    Control control;
    static_assert(sizeof(Control) <= sizeof(Raw));
    std::memcpy(&control, &raw, sizeof(Control));
    return control;
}

void DecodeSOP2(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::SOP2 + Bits(word, 23, 29));
    SetSource(inst.src[0], Bits(word, 0, 7));
    SetSource(inst.src[1], Bits(word, 8, 15));
    Set(inst.dst[0], FieldOf(Bits(word, 16, 22)), Bits(word, 16, 22));
    inst.dst_count = 1;
}

void DecodeSOPK(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::SOPK + Bits(word, 23, 27));
    Set(inst.dst[0], FieldOf(Bits(word, 16, 22)), Bits(word, 16, 22));
    inst.dst_count = 1;
    inst.control.sopk = ControlOf<InstControlSOPK>(word);
}

void DecodeSOP1(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::SOP1 + Bits(word, 8, 15));
    SetSource(inst.src[0], Bits(word, 0, 7));
    Set(inst.dst[0], FieldOf(Bits(word, 16, 22)), Bits(word, 16, 22));
    inst.dst_count = 1;
}

void DecodeSOPC(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::SOPC + Bits(word, 16, 22));
    SetSource(inst.src[0], Bits(word, 0, 7));
    SetSource(inst.src[1], Bits(word, 8, 15));
}

void DecodeSOPP(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::SOPP + Bits(word, 16, 22));
    inst.control.sopp = ControlOf<InstControlSOPP>(word);
}

void DecodeSMRD(GcnInst& inst, u32 word) {
    const u32 op = Bits(word, 22, 26);
    inst.opcode = Opcode(OpcodeBase::SMRD + op);
    Set(inst.src[0], OperandField::ScalarGPR, Bits(word, 9, 14));
    Set(inst.dst[0], OperandField::ScalarGPR, Bits(word, 15, 21));
    inst.dst_count = 1;
    auto& smrd = inst.control.smrd;
    smrd = ControlOf<InstControlSMRD>(word);
    constexpr u32 load_last = Number(Opcode::S_LOAD_DWORDX16, OpcodeBase::SMRD);
    constexpr u32 buffer_first = Number(Opcode::S_BUFFER_LOAD_DWORD, OpcodeBase::SMRD);
    constexpr u32 buffer_last = Number(Opcode::S_BUFFER_LOAD_DWORDX16, OpcodeBase::SMRD);
    if (op <= load_last) {
        smrd.count = 1 << op;
    } else if (op >= buffer_first && op <= buffer_last) {
        smrd.count = 1 << (op - buffer_first);
    }
    if (!smrd.imm) {
        // The offset is a register.
        const u32 code = smrd.offset;
        inst.src[1].field = FieldOf(code);
        inst.src[1].type = ScalarType::Uint32;
        inst.src[1].code = code;
    }
}

void DecodeVOP2(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::VOP2 + Bits(word, 25, 30));
    const u32 vsrc1 = Bits(word, 9, 16);
    const u32 vdst = Bits(word, 17, 24);
    SetSource(inst.src[0], Bits(word, 0, 8));
    Set(inst.src[1], OperandField::VectorGPR, vsrc1);
    Set(inst.dst[0], OperandField::VectorGPR, vdst);
    inst.dst_count = 1;
    if (inst.opcode == Opcode::V_READLANE_B32) {
        // The lane is a scalar operand, the result a scalar register.
        inst.src[1].field = FieldOf(vsrc1);
        inst.dst[0].field = FieldOf(vdst);
        inst.dst[0].type = ScalarType::Uint32;
    } else if (inst.opcode == Opcode::V_WRITELANE_B32) {
        inst.src[1].field = FieldOf(vsrc1);
    } else if (IsVop3B(inst.opcode)) {
        SetVccDestination(inst);
    }
}

void DecodeVOP1(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::VOP1 + Bits(word, 9, 16));
    const u32 vdst = Bits(word, 17, 24);
    SetSource(inst.src[0], Bits(word, 0, 8));
    Set(inst.dst[0], OperandField::VectorGPR, vdst);
    inst.dst_count = 1;
    if (inst.opcode == Opcode::V_READFIRSTLANE_B32) {
        inst.dst[0].field = FieldOf(vdst);
        inst.dst[0].type = ScalarType::Uint32;
    }
}

void DecodeVOPC(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::VOPC + Bits(word, 17, 24));
    SetSource(inst.src[0], Bits(word, 0, 8));
    Set(inst.src[1], OperandField::VectorGPR, Bits(word, 9, 16));
    SetVccDestination(inst);
}

void DecodeVINTRP(GcnInst& inst, u32 word) {
    inst.opcode = Opcode(OpcodeBase::VINTRP + Bits(word, 16, 17));
    Set(inst.src[0], OperandField::VectorGPR, Bits(word, 0, 7));
    Set(inst.dst[0], OperandField::VectorGPR, Bits(word, 18, 25));
    inst.dst_count = 1;
    inst.control.vintrp = ControlOf<InstControlVINTRP>(word);
}

void DecodeVOP3(GcnInst& inst, u64 word) {
    const u32 op = Bits(word, 17, 25) | Bits(word, 16, 16) << 9;
    const u32 vdst = Bits(word, 0, 7);
    if (op >= Vop3Form::VopcFirst && op <= Vop3Form::VopcLast) {
        inst.opcode = Opcode(OpcodeBase::VOPC + op - Vop3Form::VopcFirst);
    } else if (op >= Vop3Form::Vop2First && op <= Vop3Form::Vop2Last) {
        inst.opcode = Opcode(OpcodeBase::VOP2 + op - Vop3Form::Vop2First);
    } else if (op >= Vop3Form::Vop1First && op <= Vop3Form::Vop1Last) {
        inst.opcode = Opcode(OpcodeBase::VOP1 + op - Vop3Form::Vop1First);
    } else {
        inst.opcode = Opcode(OpcodeBase::VOP3 + op);
    }
    for (u32 i = 0; i < 3; ++i) {
        SetSource(inst.src[i], Bits(word, 32 + 9 * i, 40 + 9 * i));
    }
    Set(inst.dst[0], OperandField::VectorGPR, vdst);
    if (IsVop3B(inst.opcode)) {
        // The scalar carry (or VCC) result.
        const u32 sdst = Bits(word, 8, 14);
        inst.dst[1].field = FieldOf(sdst);
        inst.dst[1].type = ScalarType::Uint64;
        inst.dst[1].code = sdst;
    } else if (op >= Vop3Form::VopcFirst && op <= Vop3Form::VopcLast) {
        // Comparisons write a scalar register pair.
        inst.dst[1].field = FieldOf(vdst);
        inst.dst[1].type = ScalarType::Uint64;
        inst.dst[1].code = vdst;
    } else if (inst.opcode == Opcode::V_READLANE_B32 ||
               inst.opcode == Opcode::V_READFIRSTLANE_B32) {
        inst.dst[0].field = FieldOf(vdst);
        inst.dst[0].type = ScalarType::Uint32;
    }
    // Two destinations from V_ADD_I32's VOP3 number on to V_DIV_SCALE_F64's (the VOP3B range,
    // and what the numbers between hold).
    constexpr u32 vop3b_first = Vop3Form::Vop2First + Number(Opcode::V_ADD_I32, OpcodeBase::VOP2);
    constexpr u32 vop3b_last = Number(Opcode::V_DIV_SCALE_F64, OpcodeBase::VOP3);
    inst.dst_count = op >= vop3b_first && op <= vop3b_last ? 2 : 1;

    const auto control = ControlOf<InstControlVOP3>(word);
    inst.control.vop3 = control;
    auto& output = inst.dst[0].output_modifier;
    if (!IsVop3B(inst.opcode)) {
        for (u32 i = 0; i < 3; ++i) {
            inst.src[i].input_modifier.abs |= bool(control.abs & (1u << i));
        }
        output.clamp = control.clmp;
    }
    for (u32 i = 0; i < 3; ++i) {
        inst.src[i].input_modifier.neg |= bool(control.neg & (1u << i));
        inst.src[i].op_sel.op_sel |= bool(control.op_sel & (1u << i));
    }
    inst.dst[0].op_sel.op_sel = control.op_sel & (1u << 3);
    output.multiplier = OutputMultiplier(control.omod);
}

void DecodeVOP3P(GcnInst& inst, u64 word) {
    inst.opcode = Opcode(OpcodeBase::VOP3P + Bits(word, 16, 22));
    for (u32 i = 0; i < 3; ++i) {
        SetSource(inst.src[i], Bits(word, 32 + 9 * i, 40 + 9 * i));
    }
    Set(inst.dst[0], OperandField::VectorGPR, Bits(word, 0, 7));
    const auto control = ControlOf<InstControlVOP3P>(word);
    inst.control.vop3p = control;
    for (u32 i = 0; i < 3; ++i) {
        auto& src = inst.src[i];
        src.input_modifier.neg |= bool(control.neg & (1u << i));
        src.input_modifier.neg_hi |= bool(control.neg_hi & (1u << i));
        src.op_sel.op_sel |= bool(control.op_sel & (1u << i));
        src.op_sel.op_sel_hi |= control.get_op_sel_hi(int(i));
    }
    inst.dst[0].output_modifier.clamp = control.clamp;
}

/// Buffer operands: address, data, resource (4-register group), offset.
void SetBufferOperands(GcnInst& inst, u64 word) {
    Set(inst.src[0], OperandField::VectorGPR, Bits(word, 32, 39));
    Set(inst.src[1], OperandField::VectorGPR, Bits(word, 40, 47));
    Set(inst.src[2], OperandField::ScalarGPR, Bits(word, 48, 52));
    Set(inst.src[3], FieldOf(Bits(word, 56, 63)), Bits(word, 56, 63));
}

void DecodeMUBUF(GcnInst& inst, u64 word) {
    const u32 op = Bits(word, 18, 24);
    inst.opcode = Opcode(OpcodeBase::MUBUF + op);
    SetBufferOperands(inst, word);
    auto& mubuf = inst.control.mubuf;
    mubuf = ControlOf<InstControlMUBUF>(word);
    const auto in = [op](Opcode first, Opcode last) {
        return op >= Number(first, OpcodeBase::MUBUF) && op <= Number(last, OpcodeBase::MUBUF);
    };
    const auto number = [](Opcode opcode) { return Number(opcode, OpcodeBase::MUBUF); };
    const auto set = [&mubuf](u32 count, u32 size) {
        mubuf.count = count;
        mubuf.size = size;
    };
    if (in(Opcode::BUFFER_LOAD_FORMAT_X, Opcode::BUFFER_LOAD_FORMAT_XYZW)) {
        const u32 count = op - number(Opcode::BUFFER_LOAD_FORMAT_X) + 1;
        set(count, count * 4);
    } else if (in(Opcode::BUFFER_STORE_FORMAT_X, Opcode::BUFFER_STORE_FORMAT_XYZW)) {
        const u32 count = op - number(Opcode::BUFFER_STORE_FORMAT_X) + 1;
        set(count, count * 4);
    } else if (in(Opcode::BUFFER_LOAD_DWORD, Opcode::BUFFER_LOAD_DWORDX3)) {
        const u32 count = op == number(Opcode::BUFFER_LOAD_DWORDX3)
                              ? 3
                              : 1u << (op - number(Opcode::BUFFER_LOAD_DWORD));
        set(count, count * 4);
    } else if (in(Opcode::BUFFER_STORE_DWORD, Opcode::BUFFER_STORE_DWORDX3)) {
        const u32 count = op == number(Opcode::BUFFER_STORE_DWORDX3)
                              ? 3
                              : 1u << (op - number(Opcode::BUFFER_STORE_DWORD));
        set(count, count * 4);
    } else if (in(Opcode::BUFFER_LOAD_UBYTE, Opcode::BUFFER_LOAD_SSHORT)) {
        set(1, in(Opcode::BUFFER_LOAD_UBYTE, Opcode::BUFFER_LOAD_SBYTE) ? 1 : 2);
    } else if (in(Opcode::BUFFER_STORE_BYTE, Opcode::BUFFER_STORE_SHORT)) {
        set(1, op == number(Opcode::BUFFER_STORE_BYTE) ? 1 : 2);
    } else if (in(Opcode::BUFFER_ATOMIC_SWAP, Opcode::BUFFER_ATOMIC_FMAX)) {
        set(1, 4);
    } else if (in(Opcode::BUFFER_ATOMIC_SWAP_X2, Opcode::BUFFER_ATOMIC_FMAX_X2)) {
        set(2, 8);
    }
}

void DecodeMTBUF(GcnInst& inst, u64 word) {
    const u32 op = Bits(word, 16, 18);
    inst.opcode = Opcode(OpcodeBase::MTBUF + op);
    SetBufferOperands(inst, word);
    auto& mtbuf = inst.control.mtbuf;
    mtbuf = ControlOf<InstControlMTBUF>(word);
    const u32 load_x = Number(Opcode::TBUFFER_LOAD_FORMAT_X, OpcodeBase::MTBUF);
    const u32 load_xyzw = Number(Opcode::TBUFFER_LOAD_FORMAT_XYZW, OpcodeBase::MTBUF);
    const u32 store_x = Number(Opcode::TBUFFER_STORE_FORMAT_X, OpcodeBase::MTBUF);
    const u32 store_xyzw = Number(Opcode::TBUFFER_STORE_FORMAT_XYZW, OpcodeBase::MTBUF);
    if (op >= load_x && op <= load_xyzw) {
        mtbuf.count = op - load_x + 1;
    } else if (op >= store_x && op <= store_xyzw) {
        mtbuf.count = op - store_x + 1;
    }
}

void DecodeMIMG(GcnInst& inst, u64 word) {
    inst.opcode = Opcode(OpcodeBase::MIMG + Bits(word, 18, 24));
    Set(inst.src[0], OperandField::VectorGPR, Bits(word, 32, 39));
    Set(inst.src[2], OperandField::ScalarGPR, Bits(word, 48, 52));
    Set(inst.src[3], OperandField::ScalarGPR, Bits(word, 53, 57));
    Set(inst.dst[0], OperandField::VectorGPR, Bits(word, 40, 47));
    inst.control.mimg = ControlOf<InstControlMIMG>(word);
    inst.control.mimg.mod = MimgModifiersOf(inst.opcode).raw();
}

void DecodeDS(GcnInst& inst, u64 word) {
    const u32 op = Bits(word, 18, 25);
    inst.opcode = Opcode(OpcodeBase::DS + op);
    Set(inst.src[0], OperandField::VectorGPR, Bits(word, 32, 39));
    Set(inst.src[1], OperandField::VectorGPR, Bits(word, 40, 47));
    Set(inst.src[2], OperandField::VectorGPR, Bits(word, 48, 55));
    Set(inst.dst[0], OperandField::VectorGPR, Bits(word, 56, 63));
    inst.dst_count = 1;
    auto& ds = inst.control.ds;
    ds = ControlOf<InstControlDS>(word);
    const auto is = [&inst](std::initializer_list<Opcode> opcodes) {
        return std::ranges::contains(opcodes, inst.opcode);
    };
    const auto& format = FormatOf(InstEncoding::DS, op);
    ds.dual = is({Opcode::DS_WRITE2_B32, Opcode::DS_WRXCHG2_RTN_B32, Opcode::DS_READ2_B32,
                  Opcode::DS_WRITE2_B64, Opcode::DS_WRXCHG2_RTN_B64, Opcode::DS_READ2_B64});
    ds.sign = format.src_type == ScalarType::Sint32;
    ds.relative = inst.opcode >= Opcode::DS_ADD_SRC2_U32 && inst.opcode <= Opcode::DS_MAX_SRC2_F64;
    ds.stride = is({Opcode::DS_WRITE2ST64_B32, Opcode::DS_WRXCHG2ST64_RTN_B32,
                    Opcode::DS_READ2ST64_B32, Opcode::DS_WRITE2ST64_B64,
                    Opcode::DS_WRXCHG2ST64_RTN_B64, Opcode::DS_READ2ST64_B64});
    if (is({Opcode::DS_WRITE_B8, Opcode::DS_READ_I8, Opcode::DS_READ_U8})) {
        ds.size = 1;
    } else if (is({Opcode::DS_WRITE_B16, Opcode::DS_READ_I16, Opcode::DS_READ_U16})) {
        ds.size = 2;
    } else {
        switch (format.src_type) {
        case ScalarType::Sint32:
        case ScalarType::Uint32:
            ds.size = 4;
            break;
        case ScalarType::Sint64:
        case ScalarType::Uint64:
            ds.size = 8;
            break;
        default:
            ds.size = 0;
            break;
        }
    }
}

void DecodeEXP(GcnInst& inst, u64 word) {
    inst.opcode = Opcode::EXP;
    for (u32 i = 0; i < 4; ++i) {
        Set(inst.src[i], OperandField::VectorGPR, Bits(word, 32 + 8 * i, 39 + 8 * i));
    }
    inst.control.exp = ControlOf<InstControlEXP>(word);
}

/// The instruction being decoded (decode_failure).
thread_local const u32* decoding_at = nullptr;

/// Counts, types and class from the instruction's format.
void ApplyFormat(GcnInst& inst, InstEncoding encoding) {
    const auto& format = FormatOf(encoding, NumberIn(encoding, inst.opcode));
    if ((format.src_type == ScalarType::Undefined || format.dst_type == ScalarType::Undefined) &&
        decode_failure) {
        decode_failure(decoding_at, "instruction not in the format table");
    }
    ASSERT_MSG(format.src_type != ScalarType::Undefined &&
                   format.dst_type != ScalarType::Undefined,
               "instruction {} ({}) not in the format table, encoding {:#x}",
               NameOf(inst.opcode), u32(inst.opcode), u32(encoding));
    inst.inst_class = format.inst_class;
    inst.category = format.inst_category;
    inst.encoding = encoding;
    inst.src_count = format.src_count;
    inst.length = EncodingLength(encoding);
    for (u32 i = 0; i < inst.src_count; ++i) {
        if (inst.src[i].type == ScalarType::Undefined) {
            inst.src[i].type = format.src_type;
        }
    }
    if (inst.dst_count == 2 && inst.dst[1].type == ScalarType::Undefined) {
        inst.dst[1].type = ScalarType::Uint64; // VOP3B's scalar destination
    }
    if (inst.dst_count >= 1 && inst.dst[0].type == ScalarType::Undefined) {
        inst.dst[0].type = format.dst_type;
    }
}

/// A 32-bit literal after the instruction: an extra source for SOPK's S_SETREG_IMM32_B32 and
/// VOP2's V_MADMK/V_MADAK, else the value of a LiteralConst source.
void ReadLiteral(GcnInst& inst, InstEncoding encoding, const u32*& code) {
    const bool extra = (encoding == InstEncoding::SOPK &&
                        inst.opcode == Opcode::S_SETREG_IMM32_B32) ||
                       (encoding == InstEncoding::VOP2 && (inst.opcode == Opcode::V_MADMK_F32 ||
                                                           inst.opcode == Opcode::V_MADAK_F32));
    if (extra) {
        auto& src = inst.src[inst.src_count++];
        src.field = OperandField::LiteralConst;
        src.type = FormatOf(encoding, NumberIn(encoding, inst.opcode)).src_type;
        src.code = *code++;
        inst.length += 4;
        return;
    }
    const auto literal = std::ranges::find(inst.src, OperandField::LiteralConst, &InstOperand::field);
    if (literal != inst.src.end()) {
        literal->code = *code++;
        inst.length += 4;
    }
}

void SetSdwaSources(GcnInst& inst, auto sdwa) {
    auto& src0 = inst.src[0];
    src0.field = sdwa.s0 ? FieldOf(sdwa.src0) : OperandField::VectorGPR;
    src0.code = sdwa.src0;
    src0.sdwa_sel = SdwaSelector(sdwa.src0_sel);
    src0.input_modifier.neg = sdwa.src0_neg;
    src0.input_modifier.abs = sdwa.src0_abs;
    src0.input_modifier.sext = sdwa.src0_sext;
    auto& src1 = inst.src[1];
    src1.field = sdwa.s1 ? FieldOf(src1.code) : OperandField::VectorGPR;
    src1.sdwa_sel = SdwaSelector(sdwa.src1_sel);
    src1.input_modifier.neg = sdwa.src1_neg;
    src1.input_modifier.abs = sdwa.src1_abs;
    src1.input_modifier.sext = sdwa.src1_sext;
}

/// The SDWA dword (source 0 field Sdwa) of VOPC, VOP1 and VOP2.
void ReadSdwa(GcnInst& inst, InstEncoding encoding, const u32*& code) {
    if (inst.src[0].field != OperandField::Sdwa) {
        return;
    }
    const u32 word = *code++;
    inst.src[0].code = word;
    inst.length += 4;
    if (encoding == InstEncoding::VOPC) {
        const auto sdwa = ControlOf<SdwaVopc>(word);
        SetSdwaSources(inst, sdwa);
        inst.dst[0].field = sdwa.sd ? OperandField::ScalarGPR : OperandField::VccLo;
        inst.dst[0].code = sdwa.sdst;
    } else if (encoding == InstEncoding::VOP1 || encoding == InstEncoding::VOP2) {
        const auto sdwa = ControlOf<SdwaVop12>(word);
        SetSdwaSources(inst, sdwa);
        auto& dst = inst.dst[0];
        dst.sdwa_sel = SdwaSelector(sdwa.dst_sel);
        dst.sdwa_dst = SdwaDstUnused(sdwa.dst_u);
        dst.output_modifier.clamp = sdwa.clamp;
        dst.output_modifier.multiplier = OutputMultiplier(sdwa.omod);
    } else {
        LOG_WARNING(Render_Recompiler, "illegal instruction: SDWA outside VOP1/VOP2/VOPC");
    }
}

/// The DPP dword (source 0 field Dpp) of VOP1 and VOP2.
void ReadDpp(GcnInst& inst, const u32*& code) {
    if (inst.src[0].field != OperandField::Dpp) {
        return;
    }
    const u32 word = *code++;
    inst.length += 4;
    const auto dpp = ControlOf<Dpp>(word);
    inst.src[0].field = OperandField::VectorGPR;
    inst.src[0].code = dpp.src0;
    inst.src[0].input_modifier.abs |= bool(dpp.src0_abs);
    inst.src[0].input_modifier.neg |= bool(dpp.src0_neg);
    inst.src[1].input_modifier.abs |= bool(dpp.src1_abs);
    inst.src[1].input_modifier.neg |= bool(dpp.src1_neg);
    const auto op = dpp.GetOperation();
    LOG_ERROR(Render_Recompiler,
              "unhandled DPP operation: {:#x}, value {}, bc {}, row_mask {:#b}, bank_mask {:#b}",
              u32(op.op), op.value, bool(dpp.bc), u8(dpp.row_mask), u8(dpp.bank_mask));
    inst.src[0].dpp = op;
}

/// Operand types the format table does not express.
void FixOperandTypes(GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::V_MAD_U64_U32:
        inst.src[2].type = ScalarType::Uint64;
        break;
    case Opcode::V_MAD_I64_I32:
        inst.src[2].type = ScalarType::Sint64;
        break;
    case Opcode::V_ADDC_U32:
        inst.src[2].type = ScalarType::Uint64;
        break;
    case Opcode::IMAGE_GATHER4_C:
    case Opcode::IMAGE_GATHER4_C_O:
        inst.src[0].type = ScalarType::Any;
        break;
    default:
        break;
    }
}

} // namespace

GcnInst Decode(const u32*& code) {
    decoding_at = code;
    const InstEncoding encoding = EncodingOf(code[0]);
    if (encoding == InstEncoding::ILLEGAL && decode_failure) {
        decode_failure(code, "illegal instruction");
    }
    ASSERT_MSG(encoding != InstEncoding::ILLEGAL, "illegal instruction {:#x}", code[0]);
    GcnInst inst{};
    if (EncodingLength(encoding) == 4) {
        const u32 word = *code++;
        switch (encoding) {
        case InstEncoding::SOP2:
            DecodeSOP2(inst, word);
            break;
        case InstEncoding::SOPK:
            DecodeSOPK(inst, word);
            break;
        case InstEncoding::SOP1:
            DecodeSOP1(inst, word);
            break;
        case InstEncoding::SOPC:
            DecodeSOPC(inst, word);
            break;
        case InstEncoding::SOPP:
            DecodeSOPP(inst, word);
            break;
        case InstEncoding::SMRD:
            DecodeSMRD(inst, word);
            break;
        case InstEncoding::VOP2:
            DecodeVOP2(inst, word);
            break;
        case InstEncoding::VOP1:
            DecodeVOP1(inst, word);
            break;
        case InstEncoding::VOPC:
            DecodeVOPC(inst, word);
            break;
        case InstEncoding::VINTRP:
            DecodeVINTRP(inst, word);
            break;
        default:
            break;
        }
    } else {
        u64 word;
        std::memcpy(&word, code, sizeof(word));
        code += 2;
        switch (encoding) {
        case InstEncoding::VOP3:
            DecodeVOP3(inst, word);
            break;
        case InstEncoding::VOP3P:
            DecodeVOP3P(inst, word);
            break;
        case InstEncoding::MUBUF:
            DecodeMUBUF(inst, word);
            break;
        case InstEncoding::MTBUF:
            DecodeMTBUF(inst, word);
            break;
        case InstEncoding::MIMG:
            DecodeMIMG(inst, word);
            break;
        case InstEncoding::DS:
            DecodeDS(inst, word);
            break;
        case InstEncoding::EXP:
            DecodeEXP(inst, word);
            break;
        default:
            break;
        }
    }
    ApplyFormat(inst, encoding);
    // Only 32-bit encodings carry literals and SDWA/DPP dwords.
    if (EncodingLength(encoding) == 4) {
        ReadLiteral(inst, encoding, code);
        ReadSdwa(inst, encoding, code);
        ReadDpp(inst, code);
    }
    FixOperandTypes(inst);
    return inst;
}

} // namespace GcnIsa
