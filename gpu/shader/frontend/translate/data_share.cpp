// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// DS: LDS (and GDS) reads, writes, atomics, swizzles and the append/consume counters, and the
// lane reads/writes of VOP2/VOP1. Fragment shaders on GCN use LDS only as per-lane scratch
// (offsets in 256-byte lanes' slices): there it becomes scratch VGPRs.
#include "shader/frontend/translate/translate.h"
#include "shader/ir/reg.h"
#include "shader/runtime_info.h"

namespace Shader::Gcn {

void Translator::EmitDataShare(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::DS_ADD_U32:
        return DS_OP(inst, AtomicOp::Add, false);
    case Opcode::DS_ADD_U64:
        return DS_OP<IR::U64>(inst, AtomicOp::Add, false);
    case Opcode::DS_SUB_U32:
        return DS_OP(inst, AtomicOp::Sub, false);
    case Opcode::DS_INC_U32:
        return DS_OP(inst, AtomicOp::Inc, false);
    case Opcode::DS_DEC_U32:
        return DS_OP(inst, AtomicOp::Dec, false);
    case Opcode::DS_MIN_I32:
        return DS_OP(inst, AtomicOp::Smin, false);
    case Opcode::DS_MAX_I32:
        return DS_OP(inst, AtomicOp::Smax, false);
    case Opcode::DS_MIN_U32:
        return DS_OP(inst, AtomicOp::Umin, false);
    case Opcode::DS_MAX_U32:
        return DS_OP(inst, AtomicOp::Umax, false);
    case Opcode::DS_AND_B32:
        return DS_OP(inst, AtomicOp::And, false);
    case Opcode::DS_OR_B32:
        return DS_OP(inst, AtomicOp::Or, false);
    case Opcode::DS_XOR_B32:
        return DS_OP(inst, AtomicOp::Xor, false);
    case Opcode::DS_ADD_RTN_U32:
        return DS_OP(inst, AtomicOp::Add, true);
    case Opcode::DS_SUB_RTN_U32:
        return DS_OP(inst, AtomicOp::Sub, true);
    case Opcode::DS_MIN_RTN_U32:
        return DS_OP(inst, AtomicOp::Umin, true);
    case Opcode::DS_MAX_RTN_U32:
        return DS_OP(inst, AtomicOp::Umax, true);
    case Opcode::DS_AND_RTN_B32:
        return DS_OP(inst, AtomicOp::And, true);
    case Opcode::DS_OR_RTN_B32:
        return DS_OP(inst, AtomicOp::Or, true);
    case Opcode::DS_XOR_RTN_B32:
        return DS_OP(inst, AtomicOp::Xor, true);
    case Opcode::DS_CMPST_B32:
        return DS_CMPST(32, false, inst);
    case Opcode::DS_CMPST_RTN_B32:
        return DS_CMPST(32, true, inst);
    case Opcode::DS_CMPST_B64:
        return DS_CMPST(64, false, inst);
    case Opcode::DS_CMPST_RTN_B64:
        return DS_CMPST(64, true, inst);
    case Opcode::DS_WRITE_B16:
        return DS_WRITE(16, false, false, false, inst);
    case Opcode::DS_WRITE_B32:
        return DS_WRITE(32, false, false, false, inst);
    case Opcode::DS_WRITE2_B32:
        return DS_WRITE(32, false, true, false, inst);
    case Opcode::DS_WRITE2ST64_B32:
        return DS_WRITE(32, false, true, true, inst);
    case Opcode::DS_WRITE_B64:
        return DS_WRITE(64, false, false, false, inst);
    case Opcode::DS_WRITE2_B64:
        return DS_WRITE(64, false, true, false, inst);
    case Opcode::DS_WRITE2ST64_B64:
        return DS_WRITE(64, false, true, true, inst);
    case Opcode::DS_READ_U16:
        return DS_READ(16, false, false, false, inst);
    case Opcode::DS_READ_B32:
        return DS_READ(32, false, false, false, inst);
    case Opcode::DS_READ2_B32:
        return DS_READ(32, false, true, false, inst);
    case Opcode::DS_READ2ST64_B32:
        return DS_READ(32, false, true, true, inst);
    case Opcode::DS_READ_B64:
        return DS_READ(64, false, false, false, inst);
    case Opcode::DS_READ2_B64:
        return DS_READ(64, false, true, false, inst);
    case Opcode::DS_READ2ST64_B64:
        return DS_READ(64, false, true, true, inst);
    case Opcode::DS_SWIZZLE_B32:
        return DS_SWIZZLE_B32(inst);
    case Opcode::DS_CONSUME:
        return DS_CONSUME(inst);
    case Opcode::DS_APPEND:
        return DS_APPEND(inst);
    default:
        LogMissingOpcode(inst);
    }
}

void Translator::V_READFIRSTLANE_B32(const GcnInst& inst) {
    const IR::U32 value{GetSrc(inst.src[0])};
    SetDst(inst.dst[0], ir.ReadFirstLane(value));
}

void Translator::V_READLANE_B32(const GcnInst& inst) {
    ASSERT(inst.src[0].field == OperandField::VectorGPR);
    const IR::U32 value{GetSrc(inst.src[0])};
    const IR::U32 lane{GetSrc(inst.src[1])};
    SetDst(inst.dst[0], ir.ReadLane(value, lane));
}

void Translator::V_WRITELANE_B32(const GcnInst& inst) {
    const IR::VectorReg dst{inst.dst[0].code};
    const IR::U32 value{GetSrc(inst.src[0])};
    const IR::U32 lane{GetSrc(inst.src[1])};
    const IR::U32 old_value{GetSrc(inst.dst[0])};
    ir.SetVectorReg(dst, ir.WriteLane(old_value, value, lane));
}

namespace {

/// The 16-bit offset of DS instructions (offset1:offset0).
u32 DsOffset(const GcnInst& inst) {
    return (u32(inst.control.ds.offset1) << 8u) + u32(inst.control.ds.offset0);
}

/// Writes VGPR `data` (a pair for 64 bits) to LDS at `addr`.
void WriteLds(IR::IREmitter& ir, int bit_size, IR::VectorReg data, const IR::U32& addr,
              bool is_gds) {
    switch (bit_size) {
    case 64:
        ir.WriteShared(
            64,
            ir.PackUint2x32(ir.CompositeConstruct(ir.GetVectorReg(data), ir.GetVectorReg(data + 1))),
            addr, is_gds);
        break;
    case 32:
        ir.WriteShared(32, ir.GetVectorReg(data), addr, is_gds);
        break;
    case 16:
        ir.WriteShared(16, ir.UConvert(16, ir.GetVectorReg(data)), addr, is_gds);
        break;
    }
}

/// Stores an LDS read to VGPRs from `dst` on (two for 64 bits), advancing it.
void SetLdsResult(IR::IREmitter& ir, int bit_size, const IR::Value& data, IR::VectorReg& dst) {
    switch (bit_size) {
    case 64: {
        const auto halves = ir.UnpackUint2x32(IR::U64{data});
        ir.SetVectorReg(dst++, IR::U32{ir.CompositeExtract(halves, 0)});
        ir.SetVectorReg(dst++, IR::U32{ir.CompositeExtract(halves, 1)});
        break;
    }
    case 32:
        ir.SetVectorReg(dst++, IR::U32{data});
        break;
    case 16:
        ir.SetVectorReg(dst++, IR::U32{ir.UConvert(32, IR::U16{data})});
        break;
    }
}

} // namespace

template <typename T>
void Translator::DS_OP(const GcnInst& inst, AtomicOp op, bool rtn) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{GetSrc(inst.src[0])};
    const T data = [&] {
        if (op == AtomicOp::Inc || op == AtomicOp::Dec) {
            return T{};
        }
        if constexpr (std::is_same_v<T, IR::U32>) {
            return GetSrc(inst.src[1]);
        } else {
            return GetSrc64(inst.src[1]);
        }
    }();
    const IR::U32 address = ir.IAdd(addr, ir.Imm32(DsOffset(inst)));
    const T original = [&] -> T {
        switch (op) {
        case AtomicOp::Add:
            return ir.SharedAtomicIAdd(address, data, is_gds);
        case AtomicOp::Sub:
            return ir.SharedAtomicISub(address, data, is_gds);
        case AtomicOp::Umin:
        case AtomicOp::Smin:
            return ir.SharedAtomicIMin(address, data, op == AtomicOp::Smin, is_gds);
        case AtomicOp::Umax:
        case AtomicOp::Smax:
            return ir.SharedAtomicIMax(address, data, op == AtomicOp::Smax, is_gds);
        case AtomicOp::And:
            return ir.SharedAtomicAnd(address, data, is_gds);
        case AtomicOp::Or:
            return ir.SharedAtomicOr(address, data, is_gds);
        case AtomicOp::Xor:
            return ir.SharedAtomicXor(address, data, is_gds);
        case AtomicOp::Inc:
            return ir.SharedAtomicInc<T>(address, is_gds);
        case AtomicOp::Dec:
            return ir.SharedAtomicDec<T>(address, is_gds);
        default:
            UNREACHABLE();
        }
    }();
    if (!rtn) {
        return;
    }
    if constexpr (std::is_same_v<T, IR::U32>) {
        SetDst(inst.dst[0], original);
    } else {
        SetDst64(inst.dst[0], original);
    }
}

void Translator::DS_CMPST(int bit_size, bool rtn, const GcnInst& inst) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{GetSrc(inst.src[0])};
    const IR::U32 address = ir.IAdd(addr, ir.Imm32(DsOffset(inst)));
    if (bit_size == 64) {
        const IR::U64 cmp{GetSrc64(inst.src[1])};
        const IR::U64 value{GetSrc64(inst.src[2])};
        const IR::U64 original = ir.SharedAtomicCmpSwap(address, value, cmp, is_gds);
        if (rtn) {
            SetDst64(inst.dst[0], original);
        }
        return;
    }
    const IR::U32 cmp{GetSrc(inst.src[1])};
    const IR::U32 value{GetSrc(inst.src[2])};
    const IR::U32 original = ir.SharedAtomicCmpSwap(address, value, cmp, is_gds);
    if (rtn) {
        SetDst(inst.dst[0], original);
    }
}

void Translator::DS_WRITE(int bit_size, bool is_signed, bool is_pair, bool stride64,
                          const GcnInst& inst) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{ir.GetVectorReg(IR::VectorReg(inst.src[0].code))};
    const IR::VectorReg data0{inst.src[1].code};
    const IR::VectorReg data1{inst.src[2].code};
    const u32 offset = DsOffset(inst);
    if (info.hw_stage == HwStage::Fragment) {
        ASSERT_MSG(!is_pair && bit_size == 32 && offset % 256 == 0,
                   "Unexpected shared memory offset alignment: {}", offset);
        ir.SetVectorReg(GetScratchVgpr(offset), ir.GetVectorReg(data0));
        return;
    }
    if (!is_pair) {
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(offset));
        WriteLds(ir, bit_size, data0, addr0, is_gds);
        return;
    }
    // Two writes, the offsets in elements (of 64 elements with ST64).
    const u32 scale = (bit_size == 32 ? 4 : 8) * (stride64 ? 64 : 1);
    const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset0 * scale)));
    WriteLds(ir, bit_size, data0, addr0, is_gds);
    const IR::U32 addr1 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset1 * scale)));
    WriteLds(ir, bit_size, data1, addr1, is_gds);
}

void Translator::DS_READ(int bit_size, bool is_signed, bool is_pair, bool stride64,
                         const GcnInst& inst) {
    const bool is_gds = inst.control.ds.gds;
    const IR::U32 addr{ir.GetVectorReg(IR::VectorReg(inst.src[0].code))};
    IR::VectorReg dst{inst.dst[0].code};
    const u32 offset = DsOffset(inst);
    if (info.hw_stage == HwStage::Fragment) {
        ASSERT_MSG(!is_pair && bit_size == 32 && offset % 256 == 0,
                   "Unexpected shared memory offset alignment: {}", offset);
        ir.SetVectorReg(dst, ir.GetVectorReg(GetScratchVgpr(offset)));
        return;
    }
    if (!is_pair) {
        const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(offset));
        SetLdsResult(ir, bit_size, ir.LoadShared(bit_size, is_signed, addr0, is_gds), dst);
        return;
    }
    const u32 scale = (bit_size == 32 ? 4 : 8) * (stride64 ? 64 : 1);
    const IR::U32 addr0 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset0 * scale)));
    SetLdsResult(ir, bit_size, ir.LoadShared(bit_size, is_signed, addr0, is_gds), dst);
    const IR::U32 addr1 = ir.IAdd(addr, ir.Imm32(u32(inst.control.ds.offset1 * scale)));
    SetLdsResult(ir, bit_size, ir.LoadShared(bit_size, is_signed, addr1, is_gds), dst);
}

/// Quad permute mode (offset1 bit 7: four 2-bit lane selects in offset0), or bitmask mode
/// within 32 lanes (and/or/xor masks).
void Translator::DS_SWIZZLE_B32(const GcnInst& inst) {
    const u8 offset0 = inst.control.ds.offset0;
    const u8 offset1 = inst.control.ds.offset1;
    const IR::U32 src{GetSrc(inst.src[0])};
    if (offset1 & 0x80) {
        if (offset0 == 0x0 || offset0 == 0x55 || offset0 == 0xAA || offset0 == 0xFF) {
            SetDst(inst.dst[0], ir.QuadBroadcast(src, ir.Imm32(offset0 & 3)));
            return;
        }
        const IR::U32 lane_id = ir.LaneId();
        const IR::U32 id_in_quad = ir.BitwiseAnd(lane_id, ir.Imm32(3));
        const IR::U32 base = ir.ShiftLeftLogical(id_in_quad, ir.Imm32(1));
        const IR::U32 select = ir.BitFieldExtract(ir.Imm32(offset0), base, ir.Imm32(2));
        const IR::U32 quad_base = ir.BitwiseAnd(lane_id, ir.Imm32(~3u));
        SetDst(inst.dst[0], ir.Shuffle(src, ir.BitwiseOr(quad_base, select)));
        return;
    }
    const u8 and_mask = (offset0 & 0x1f) | (~u8{0} << 5);
    const u8 or_mask = (offset0 >> 5) | ((offset1 & 0x3) << 3);
    const u8 xor_mask = offset1 >> 2;
    if (or_mask == 0u && and_mask == 255u) {
        SetDst(inst.dst[0], ir.ShuffleXor(src, ir.Imm32(xor_mask)));
        return;
    }
    const IR::U32 lane_id = ir.LaneId();
    const IR::U32 local = ir.BitwiseAnd(lane_id, ir.Imm32(31u));
    const IR::U32 half = ir.BitwiseAnd(lane_id, ir.Imm32(~31u));
    const IR::U32 index = ir.BitwiseXor(
        ir.BitwiseOr(ir.BitwiseAnd(local, ir.Imm32(and_mask)), ir.Imm32(or_mask)),
        ir.Imm32(xor_mask));
    SetDst(inst.dst[0], ir.Shuffle(src, ir.BitwiseOr(index, half)));
}

/// The counter's GDS address: M0[31:16] plus the instruction offset, in dwords.
static IR::U32 GdsCounterAddress(IR::IREmitter& ir, const GcnInst& inst) {
    const IR::U32 base = ir.BitFieldExtract(ir.GetM0(), ir.Imm32(16), ir.Imm32(16));
    const IR::U32 gds_offset = ir.IAdd(base, ir.Imm32(DsOffset(inst)));
    return ir.ShiftRightLogical(gds_offset, ir.Imm32(2u));
}

void Translator::DS_APPEND(const GcnInst& inst) {
    SetDst(inst.dst[0], ir.DataAppend(GdsCounterAddress(ir, inst)));
}

void Translator::DS_CONSUME(const GcnInst& inst) {
    SetDst(inst.dst[0], ir.DataConsume(GdsCounterAddress(ir, inst)));
}

} // namespace Shader::Gcn
