// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// MUBUF/MTBUF buffer loads, stores and atomics, and MIMG image loads, stores, atomics, samples,
// gathers and queries. Resources are passed as the dwords of their sharps (found and bound by
// the resource passes); sample addresses as all the GCN address registers the modifiers may use
// (resolved once the image type is known).
#include <bit>

#include <boost/container/static_vector.hpp>

#include "shader/frontend/translate/translate.h"

namespace Shader::Gcn {

void Translator::EmitVectorMemory(const GcnInst& inst) {
    switch (inst.opcode) {
    // Buffer loads: dwords, format (typed by the instruction or the buffer), bytes and shorts.
    case Opcode::TBUFFER_LOAD_FORMAT_X:
        return BUFFER_LOAD(1, true, false, inst);
    case Opcode::TBUFFER_LOAD_FORMAT_XY:
        return BUFFER_LOAD(2, true, false, inst);
    case Opcode::TBUFFER_LOAD_FORMAT_XYZ:
        return BUFFER_LOAD(3, true, false, inst);
    case Opcode::TBUFFER_LOAD_FORMAT_XYZW:
        return BUFFER_LOAD(4, true, false, inst);
    case Opcode::BUFFER_LOAD_FORMAT_X:
        return BUFFER_LOAD(1, false, true, inst);
    case Opcode::BUFFER_LOAD_FORMAT_XY:
        return BUFFER_LOAD(2, false, true, inst);
    case Opcode::BUFFER_LOAD_FORMAT_XYZ:
        return BUFFER_LOAD(3, false, true, inst);
    case Opcode::BUFFER_LOAD_FORMAT_XYZW:
        return BUFFER_LOAD(4, false, true, inst);
    case Opcode::BUFFER_LOAD_UBYTE:
        return BUFFER_LOAD(1, false, false, inst, 8, false);
    case Opcode::BUFFER_LOAD_SBYTE:
        return BUFFER_LOAD(1, false, false, inst, 8, true);
    case Opcode::BUFFER_LOAD_USHORT:
        return BUFFER_LOAD(1, false, false, inst, 16, false);
    case Opcode::BUFFER_LOAD_SSHORT:
        return BUFFER_LOAD(1, false, false, inst, 16, true);
    case Opcode::BUFFER_LOAD_DWORD:
        return BUFFER_LOAD(1, false, false, inst);
    case Opcode::BUFFER_LOAD_DWORDX2:
        return BUFFER_LOAD(2, false, false, inst);
    case Opcode::BUFFER_LOAD_DWORDX3:
        return BUFFER_LOAD(3, false, false, inst);
    case Opcode::BUFFER_LOAD_DWORDX4:
        return BUFFER_LOAD(4, false, false, inst);
    // Buffer stores
    case Opcode::BUFFER_STORE_FORMAT_X:
        return BUFFER_STORE(1, false, true, inst);
    case Opcode::BUFFER_STORE_FORMAT_XY:
        return BUFFER_STORE(2, false, true, inst);
    case Opcode::BUFFER_STORE_FORMAT_XYZ:
        return BUFFER_STORE(3, false, true, inst);
    case Opcode::BUFFER_STORE_FORMAT_XYZW:
        return BUFFER_STORE(4, false, true, inst);
    case Opcode::TBUFFER_STORE_FORMAT_X:
        return BUFFER_STORE(1, true, false, inst);
    case Opcode::TBUFFER_STORE_FORMAT_XY:
        return BUFFER_STORE(2, true, false, inst);
    case Opcode::TBUFFER_STORE_FORMAT_XYZ:
        return BUFFER_STORE(3, true, false, inst);
    case Opcode::TBUFFER_STORE_FORMAT_XYZW:
        return BUFFER_STORE(4, true, false, inst);
    case Opcode::BUFFER_STORE_BYTE:
        return BUFFER_STORE(1, false, false, inst, 8);
    case Opcode::BUFFER_STORE_SHORT:
        return BUFFER_STORE(1, false, false, inst, 16);
    case Opcode::BUFFER_STORE_DWORD:
        return BUFFER_STORE(1, false, false, inst);
    case Opcode::BUFFER_STORE_DWORDX2:
        return BUFFER_STORE(2, false, false, inst);
    case Opcode::BUFFER_STORE_DWORDX3:
        return BUFFER_STORE(3, false, false, inst);
    case Opcode::BUFFER_STORE_DWORDX4:
        return BUFFER_STORE(4, false, false, inst);
    // Buffer atomics
    case Opcode::BUFFER_ATOMIC_ADD:
        return BUFFER_ATOMIC(AtomicOp::Add, inst);
    case Opcode::BUFFER_ATOMIC_SWAP:
        return BUFFER_ATOMIC(AtomicOp::Swap, inst);
    case Opcode::BUFFER_ATOMIC_CMPSWAP:
        return BUFFER_ATOMIC(AtomicOp::CmpSwap, inst);
    case Opcode::BUFFER_ATOMIC_SMIN:
        return BUFFER_ATOMIC(AtomicOp::Smin, inst);
    case Opcode::BUFFER_ATOMIC_SMIN_X2:
        return BUFFER_ATOMIC<IR::U64>(AtomicOp::Smin, inst);
    case Opcode::BUFFER_ATOMIC_UMIN:
        return BUFFER_ATOMIC(AtomicOp::Umin, inst);
    case Opcode::BUFFER_ATOMIC_UMIN_X2:
        return BUFFER_ATOMIC<IR::U64>(AtomicOp::Umin, inst);
    case Opcode::BUFFER_ATOMIC_SMAX:
        return BUFFER_ATOMIC(AtomicOp::Smax, inst);
    case Opcode::BUFFER_ATOMIC_SMAX_X2:
        return BUFFER_ATOMIC<IR::U64>(AtomicOp::Smax, inst);
    case Opcode::BUFFER_ATOMIC_UMAX:
        return BUFFER_ATOMIC(AtomicOp::Umax, inst);
    case Opcode::BUFFER_ATOMIC_UMAX_X2:
        return BUFFER_ATOMIC<IR::U64>(AtomicOp::Umax, inst);
    case Opcode::BUFFER_ATOMIC_AND:
        return BUFFER_ATOMIC(AtomicOp::And, inst);
    case Opcode::BUFFER_ATOMIC_OR:
        return BUFFER_ATOMIC(AtomicOp::Or, inst);
    case Opcode::BUFFER_ATOMIC_XOR:
        return BUFFER_ATOMIC(AtomicOp::Xor, inst);
    case Opcode::BUFFER_ATOMIC_INC:
        return BUFFER_ATOMIC(AtomicOp::Inc, inst);
    case Opcode::BUFFER_ATOMIC_DEC:
        return BUFFER_ATOMIC(AtomicOp::Dec, inst);
    case Opcode::BUFFER_ATOMIC_FMIN:
        return BUFFER_ATOMIC<IR::F32>(AtomicOp::Fmin, inst);
    case Opcode::BUFFER_ATOMIC_FMAX:
        return BUFFER_ATOMIC<IR::F32>(AtomicOp::Fmax, inst);
    case Opcode::BUFFER_ATOMIC_FCMPSWAP:
        return BUFFER_ATOMIC<IR::F32>(AtomicOp::FCmpSwap, inst);
    // Images
    case Opcode::IMAGE_LOAD:
        return IMAGE_LOAD(false, inst);
    case Opcode::IMAGE_LOAD_MIP:
        return IMAGE_LOAD(true, inst);
    case Opcode::IMAGE_STORE:
        return IMAGE_STORE(false, inst);
    case Opcode::IMAGE_STORE_MIP:
        return IMAGE_STORE(true, inst);
    case Opcode::IMAGE_GET_RESINFO:
        return IMAGE_GET_RESINFO(inst);
    case Opcode::IMAGE_GET_LOD:
        return IMAGE_GET_LOD(inst);
    case Opcode::IMAGE_ATOMIC_SWAP:
        return IMAGE_ATOMIC(AtomicOp::Swap, inst);
    case Opcode::IMAGE_ATOMIC_CMPSWAP:
        return IMAGE_ATOMIC(AtomicOp::CmpSwap, inst);
    case Opcode::IMAGE_ATOMIC_ADD:
        return IMAGE_ATOMIC(AtomicOp::Add, inst);
    case Opcode::IMAGE_ATOMIC_SMIN:
        return IMAGE_ATOMIC(AtomicOp::Smin, inst);
    case Opcode::IMAGE_ATOMIC_UMIN:
        return IMAGE_ATOMIC(AtomicOp::Umin, inst);
    case Opcode::IMAGE_ATOMIC_FMIN:
        return IMAGE_ATOMIC(AtomicOp::Fmin, inst);
    case Opcode::IMAGE_ATOMIC_SMAX:
        return IMAGE_ATOMIC(AtomicOp::Smax, inst);
    case Opcode::IMAGE_ATOMIC_FMAX:
        return IMAGE_ATOMIC(AtomicOp::Fmax, inst);
    case Opcode::IMAGE_ATOMIC_UMAX:
        return IMAGE_ATOMIC(AtomicOp::Umax, inst);
    case Opcode::IMAGE_ATOMIC_AND:
        return IMAGE_ATOMIC(AtomicOp::And, inst);
    case Opcode::IMAGE_ATOMIC_OR:
        return IMAGE_ATOMIC(AtomicOp::Or, inst);
    case Opcode::IMAGE_ATOMIC_XOR:
        return IMAGE_ATOMIC(AtomicOp::Xor, inst);
    case Opcode::IMAGE_ATOMIC_INC:
        return IMAGE_ATOMIC(AtomicOp::Inc, inst);
    case Opcode::IMAGE_ATOMIC_DEC:
        return IMAGE_ATOMIC(AtomicOp::Dec, inst);
    case Opcode::IMAGE_SAMPLE:
    case Opcode::IMAGE_SAMPLE_D:
    case Opcode::IMAGE_SAMPLE_L:
    case Opcode::IMAGE_SAMPLE_B:
    case Opcode::IMAGE_SAMPLE_LZ:
    case Opcode::IMAGE_SAMPLE_C:
    case Opcode::IMAGE_SAMPLE_C_LZ:
    case Opcode::IMAGE_SAMPLE_O:
    case Opcode::IMAGE_SAMPLE_D_O:
    case Opcode::IMAGE_SAMPLE_L_O:
    case Opcode::IMAGE_SAMPLE_B_O:
    case Opcode::IMAGE_SAMPLE_LZ_O:
    case Opcode::IMAGE_SAMPLE_C_O:
    case Opcode::IMAGE_SAMPLE_C_LZ_O:
    case Opcode::IMAGE_SAMPLE_CD:
        return IMAGE_SAMPLE(inst);
    case Opcode::IMAGE_GATHER4:
    case Opcode::IMAGE_GATHER4_L:
    case Opcode::IMAGE_GATHER4_LZ:
    case Opcode::IMAGE_GATHER4_C:
    case Opcode::IMAGE_GATHER4_O:
    case Opcode::IMAGE_GATHER4_C_O:
    case Opcode::IMAGE_GATHER4_C_LZ:
    case Opcode::IMAGE_GATHER4_LZ_O:
    case Opcode::IMAGE_GATHER4_C_LZ_O:
        return IMAGE_GATHER(inst);
    default:
        LogMissingOpcode(inst);
    }
}

namespace {

/// Four consecutive SGPRs (a V#, S# or half a T#).
IR::Value Sgpr4(IR::IREmitter& ir, IR::ScalarReg reg) {
    return ir.CompositeConstruct(ir.GetScalarReg(reg), ir.GetScalarReg(reg + 1),
                                 ir.GetScalarReg(reg + 2), ir.GetScalarReg(reg + 3));
}

/// Four consecutive VGPRs (an image address).
template <typename T = IR::U32>
IR::Value Vgpr4(IR::IREmitter& ir, IR::VectorReg reg) {
    return ir.CompositeConstruct(ir.GetVectorReg<T>(reg), ir.GetVectorReg<T>(reg + 1),
                                 ir.GetVectorReg<T>(reg + 2), ir.GetVectorReg<T>(reg + 3));
}

/// A T#: the low half, and the high half unless the instruction uses 128-bit sharps.
struct TSharp {
    IR::Value low;
    IR::Value high;
};

TSharp LoadTSharp(IR::IREmitter& ir, const GcnInst& inst) {
    const IR::ScalarReg reg{inst.src[2].code * 4};
    const IR::Value low = Sgpr4(ir, reg);
    const IR::Value high = inst.control.mimg.r128 ? IR::Value{} : Sgpr4(ir, reg + 4);
    return {low, high};
}

/// The addressing and coherence fields of a MUBUF instruction.
IR::BufferInstInfo MubufInfo(const GcnInst& inst, u32 pc) {
    const auto& mubuf = inst.control.mubuf;
    IR::BufferInstInfo info{};
    info.index_enable.Assign(mubuf.idxen);
    info.voffset_enable.Assign(mubuf.offen);
    info.inst_offset.Assign(mubuf.offset);
    info.globally_coherent.Assign(mubuf.glc);
    info.system_coherent.Assign(mubuf.slc);
    info.pc.Assign(pc);
    return info;
}

/// Typed accesses: the format of the instruction (MTBUF) or of the buffer.
void SetBufferFormat(IR::BufferInstInfo& info, const GcnInst& inst, bool is_inst_typed,
                     bool is_buffer_typed) {
    info.typed.Assign(is_inst_typed || is_buffer_typed);
    if (is_inst_typed) {
        const auto& mtbuf = inst.control.mtbuf;
        info.inst_data_fmt.Assign(static_cast<::Gcn::DataFormat>(mtbuf.dfmt));
        info.inst_num_fmt.Assign(static_cast<::Gcn::NumberFormat>(mtbuf.nfmt));
    } else {
        info.inst_data_fmt.Assign(::Gcn::DataFormat::FormatInvalid);
    }
}

} // namespace

IR::Value Translator::BufferAddress(const GcnInst& inst) {
    const auto& mubuf = inst.control.mubuf;
    const IR::VectorReg vaddr{inst.src[0].code};
    const IR::U32 index = mubuf.idxen ? ir.GetVectorReg(vaddr) : ir.Imm32(0);
    const IR::VectorReg voffset_vgpr = mubuf.idxen ? vaddr + 1 : vaddr;
    const IR::U32 voffset = mubuf.offen ? ir.GetVectorReg(voffset_vgpr) : ir.Imm32(0);
    const IR::U32 soffset{GetSrc(inst.src[3])};
    return ir.CompositeConstruct(index, voffset, soffset);
}

void Translator::BUFFER_LOAD(u32 num_dwords, bool is_inst_typed, bool is_buffer_typed,
                             const GcnInst& inst, u32 scalar_width, bool is_signed) {
    const IR::Value address = BufferAddress(inst);
    IR::BufferInstInfo buffer_info = MubufInfo(inst, pc);
    SetBufferFormat(buffer_info, inst, is_inst_typed, is_buffer_typed);
    const IR::Value handle = Sgpr4(ir, IR::ScalarReg(inst.src[2].code * 4));
    const IR::VectorReg dst{inst.src[1].code};
    if (buffer_info.typed) {
        const IR::Value value = ir.LoadBufferFormat(handle, address, buffer_info);
        for (u32 i = 0; i < num_dwords; i++) {
            ir.SetVectorReg(dst + i, IR::F32{ir.CompositeExtract(value, i)});
        }
        return;
    }
    IR::Value value;
    switch (scalar_width) {
    case 8: {
        const IR::U8 byte = ir.LoadBufferU8(handle, address, buffer_info);
        value = is_signed ? ir.SConvert(32, byte) : ir.UConvert(32, byte);
        break;
    }
    case 16: {
        const IR::U16 half = ir.LoadBufferU16(handle, address, buffer_info);
        value = is_signed ? ir.SConvert(32, half) : ir.UConvert(32, half);
        break;
    }
    case 32:
        value = ir.LoadBufferU32(num_dwords, handle, address, buffer_info);
        break;
    default:
        UNREACHABLE();
    }
    if (num_dwords == 1) {
        ir.SetVectorReg(dst, IR::U32{value});
        return;
    }
    for (u32 i = 0; i < num_dwords; i++) {
        ir.SetVectorReg(dst + i, IR::U32{ir.CompositeExtract(value, i)});
    }
}

void Translator::BUFFER_STORE(u32 num_dwords, bool is_inst_typed, bool is_buffer_typed,
                              const GcnInst& inst, u32 scalar_width) {
    const IR::Value address = BufferAddress(inst);
    IR::BufferInstInfo buffer_info = MubufInfo(inst, pc);
    SetBufferFormat(buffer_info, inst, is_inst_typed, is_buffer_typed);
    const IR::Value handle = Sgpr4(ir, IR::ScalarReg(inst.src[2].code * 4));
    const IR::VectorReg src{inst.src[1].code};
    const bool typed = buffer_info.typed;

    boost::container::static_vector<IR::Value, 4> comps;
    for (u32 i = 0; i < num_dwords; i++) {
        if (typed) {
            comps.push_back(ir.GetVectorReg<IR::F32>(src + i));
        } else {
            comps.push_back(ir.GetVectorReg<IR::U32>(src + i));
        }
    }
    if (typed) {
        // Formatted stores take a vec4.
        while (comps.size() < 4) {
            comps.push_back(ir.Imm32(0.f));
        }
        ir.StoreBufferFormat(handle, address, ir.CompositeConstruct(comps), buffer_info);
        return;
    }
    IR::Value value = num_dwords == 1 ? comps[0] : ir.CompositeConstruct(comps);
    if (scalar_width != 32) {
        value = ir.UConvert(scalar_width, IR::U32{value});
    }
    switch (scalar_width) {
    case 8:
        ir.StoreBufferU8(handle, address, IR::U8{value}, buffer_info);
        break;
    case 16:
        ir.StoreBufferU16(handle, address, IR::U16{value}, buffer_info);
        break;
    case 32:
        ir.StoreBufferU32(num_dwords, handle, address, value, buffer_info);
        break;
    default:
        UNREACHABLE();
    }
}

/// With GLC the previous value is returned to VDATA.
template <typename T>
void Translator::BUFFER_ATOMIC(AtomicOp op, const GcnInst& inst) {
    const IR::VectorReg vdata{inst.src[1].code};
    const IR::Value address = BufferAddress(inst);
    const IR::BufferInstInfo buffer_info = MubufInfo(inst, pc);
    const IR::Value data = [&] -> IR::Value {
        if constexpr (std::is_same_v<T, IR::U64>) {
            return ir.PackUint2x32(ir.CompositeConstruct(ir.GetVectorReg<IR::U32>(vdata),
                                                         ir.GetVectorReg<IR::U32>(vdata + 1)));
        } else {
            return ir.GetVectorReg<T>(vdata);
        }
    }();
    const IR::Value handle = Sgpr4(ir, IR::ScalarReg(inst.src[2].code * 4));
    const IR::Value original = [&] {
        switch (op) {
        case AtomicOp::Swap:
            return ir.BufferAtomicSwap(handle, address, data, buffer_info);
        case AtomicOp::CmpSwap: {
            const IR::Value cmp = ir.GetVectorReg(vdata + 1);
            return ir.BufferAtomicCmpSwap(handle, address, data, cmp, buffer_info);
        }
        case AtomicOp::FCmpSwap: {
            const IR::Value cmp = ir.GetVectorReg(vdata + 1);
            return ir.BufferAtomicFCmpSwap(handle, address, data, cmp, buffer_info);
        }
        case AtomicOp::Add:
            return ir.BufferAtomicIAdd(handle, address, data, buffer_info);
        case AtomicOp::Smin:
        case AtomicOp::Umin:
            return ir.BufferAtomicIMin(handle, address, data, op == AtomicOp::Smin, buffer_info);
        case AtomicOp::Smax:
        case AtomicOp::Umax:
            return ir.BufferAtomicIMax(handle, address, data, op == AtomicOp::Smax, buffer_info);
        case AtomicOp::And:
            return ir.BufferAtomicAnd(handle, address, data, buffer_info);
        case AtomicOp::Or:
            return ir.BufferAtomicOr(handle, address, data, buffer_info);
        case AtomicOp::Xor:
            return ir.BufferAtomicXor(handle, address, data, buffer_info);
        case AtomicOp::Inc:
            return ir.BufferAtomicInc(handle, address, buffer_info);
        case AtomicOp::Dec:
            return ir.BufferAtomicDec(handle, address, buffer_info);
        case AtomicOp::Fmin:
            return ir.BufferAtomicFMin(handle, address, data, buffer_info);
        case AtomicOp::Fmax:
            return ir.BufferAtomicFMax(handle, address, data, buffer_info);
        default:
            UNREACHABLE();
        }
    }();
    if (inst.control.mubuf.glc) {
        if constexpr (std::is_same_v<T, IR::U64>) {
            UNREACHABLE();
        } else {
            ir.SetVectorReg(vdata, T{original});
        }
    }
}

// MIMG

void Translator::IMAGE_LOAD(bool has_mip, const GcnInst& inst) {
    const auto& mimg = inst.control.mimg;
    IR::VectorReg dst{inst.dst[0].code};
    const TSharp tsharp = LoadTSharp(ir, inst);
    const IR::Value body = Vgpr4(ir, IR::VectorReg(inst.src[0].code));
    IR::TextureInstInfo info{};
    info.has_lod.Assign(has_mip);
    info.is_array.Assign(mimg.da);
    info.is_r128.Assign(mimg.r128);
    const IR::Value texel =
        ir.ImageRead(ir.ImageHandle(tsharp.low, tsharp.high), body, {}, {}, info);
    for (u32 i = 0; i < 4; i++) {
        if ((mimg.dmask >> i) & 1) {
            ir.SetVectorReg(dst++, IR::F32{ir.CompositeExtract(texel, i)});
        }
    }
}

void Translator::IMAGE_STORE(bool has_mip, const GcnInst& inst) {
    const auto& mimg = inst.control.mimg;
    IR::VectorReg data{inst.dst[0].code};
    const TSharp tsharp = LoadTSharp(ir, inst);
    const IR::Value body = Vgpr4(ir, IR::VectorReg(inst.src[0].code));
    IR::TextureInstInfo info{};
    info.has_lod.Assign(has_mip);
    info.is_array.Assign(mimg.da);
    info.is_r128.Assign(mimg.r128);
    // The DMASK components come from consecutive VGPRs; the others are 0.
    boost::container::static_vector<IR::F32, 4> comps;
    for (u32 i = 0; i < 4; i++) {
        if ((mimg.dmask >> i) & 1) {
            comps.push_back(ir.GetVectorReg<IR::F32>(data++));
        } else {
            comps.push_back(ir.Imm32(0.f));
        }
    }
    const IR::Value value = ir.CompositeConstruct(comps[0], comps[1], comps[2], comps[3]);
    ir.ImageWrite(ir.ImageHandle(tsharp.low, tsharp.high), body, {}, {}, value, info);
}

void Translator::IMAGE_GET_RESINFO(const GcnInst& inst) {
    const auto& mimg = inst.control.mimg;
    IR::VectorReg dst{inst.dst[0].code};
    const auto flags = ImageResFlags(mimg.dmask);
    const bool has_mips = flags.test(ImageResComponent::MipCount);
    const IR::U32 lod = ir.GetVectorReg(IR::VectorReg(inst.src[0].code));
    const TSharp tsharp = LoadTSharp(ir, inst);
    IR::TextureInstInfo info{};
    info.is_array.Assign(mimg.da);
    info.is_r128.Assign(mimg.r128);
    const IR::Value size = ir.ImageQueryDimension(ir.ImageHandle(tsharp.low, tsharp.high), lod,
                                                  ir.Imm1(has_mips), info);
    const std::array components{flags.test(ImageResComponent::Width),
                                flags.test(ImageResComponent::Height),
                                flags.test(ImageResComponent::Depth), has_mips};
    for (u32 i = 0; i < 4; i++) {
        if (components[i]) {
            ir.SetVectorReg(dst++, IR::U32{ir.CompositeExtract(size, i)});
        }
    }
}

void Translator::IMAGE_ATOMIC(AtomicOp op, const GcnInst& inst) {
    const auto& mimg = inst.control.mimg;
    const IR::VectorReg data{inst.dst[0].code};
    IR::TextureInstInfo info{};
    info.is_array.Assign(mimg.da);
    info.is_r128.Assign(mimg.r128);
    const IR::Value value = ir.GetVectorReg(data);
    const TSharp tsharp = LoadTSharp(ir, inst);
    const IR::Value handle = ir.ImageHandle(tsharp.low, tsharp.high);
    const IR::Value body = Vgpr4(ir, IR::VectorReg(inst.src[0].code));
    const IR::Value previous = [&] {
        switch (op) {
        case AtomicOp::Swap:
            return ir.ImageAtomicExchange(handle, body, value, {});
        case AtomicOp::CmpSwap: {
            const IR::Value cmp = ir.GetVectorReg(data + 1);
            return ir.ImageAtomicCmpSwap(handle, body, value, cmp, info);
        }
        case AtomicOp::Add:
            return ir.ImageAtomicIAdd(handle, body, value, info);
        case AtomicOp::Smin:
            return ir.ImageAtomicIMin(handle, body, value, true, info);
        case AtomicOp::Umin:
            return ir.ImageAtomicUMin(handle, body, value, info);
        case AtomicOp::Smax:
            return ir.ImageAtomicIMax(handle, body, value, true, info);
        case AtomicOp::Umax:
            return ir.ImageAtomicUMax(handle, body, value, info);
        case AtomicOp::Fmax:
            return ir.ImageAtomicFMax(handle, body, value, info);
        case AtomicOp::Fmin:
            return ir.ImageAtomicFMin(handle, body, value, info);
        case AtomicOp::And:
            return ir.ImageAtomicAnd(handle, body, value, info);
        case AtomicOp::Or:
            return ir.ImageAtomicOr(handle, body, value, info);
        case AtomicOp::Xor:
            return ir.ImageAtomicXor(handle, body, value, info);
        case AtomicOp::Inc:
            return ir.ImageAtomicInc(handle, body, value, info);
        case AtomicOp::Dec:
            return ir.ImageAtomicDec(handle, body, value, info);
        default:
            UNREACHABLE();
        }
    }();
    if (mimg.glc) {
        ir.SetVectorReg(data, IR::U32{previous});
    }
}

namespace {

/// ImageSampleRaw with the T#, S# and up to 13 address registers: the four base ones plus one
/// each for offset, bias and compare value, plus six for derivatives (all that the image type
/// could use; missing ones 0).
IR::Value EmitImageSample(IR::IREmitter& ir, const GcnInst& inst, bool gather, u32 pc) {
    const auto& mimg = inst.control.mimg;
    const auto flags = MimgModifierFlags(mimg.mod);
    const IR::VectorReg addr_reg{inst.src[0].code};
    const IR::ScalarReg sampler_reg{inst.src[3].code * 4};

    IR::TextureInstInfo info{};
    info.is_depth.Assign(flags.test(MimgModifier::Pcf));
    info.has_bias.Assign(flags.test(MimgModifier::LodBias));
    info.has_lod_clamp.Assign(flags.test(MimgModifier::LodClamp));
    info.force_level0.Assign(flags.test(MimgModifier::Level0));
    info.has_offset.Assign(flags.test(MimgModifier::Offset));
    info.has_lod.Assign(flags.any(MimgModifier::Lod));
    info.is_array.Assign(mimg.da);
    info.is_unnormalized.Assign(mimg.unrm);
    info.is_r128.Assign(mimg.r128);
    info.pc.Assign(pc);
    if (gather) {
        info.gather_comp.Assign(std::bit_width(mimg.dmask) - 1);
        info.is_gather.Assign(true);
    } else {
        info.has_derivatives.Assign(flags.test(MimgModifier::Derivative));
    }

    const TSharp tsharp = LoadTSharp(ir, inst);
    const IR::Value image_handle = ir.ImageHandle(tsharp.low, tsharp.high);
    const IR::Value sampler_handle = Sgpr4(ir, sampler_reg);

    const int num_addr_regs = 4 + int(info.has_offset) + int(info.has_bias) +
                              int(info.is_depth) + (info.has_derivatives ? 6 : 0);
    const auto addr = [&](int index) -> IR::F32 {
        if (index >= num_addr_regs) {
            return ir.Imm32(0.f);
        }
        return ir.GetVectorReg<IR::F32>(addr_reg + index);
    };
    const IR::Value address1 = ir.CompositeConstruct(addr(0), addr(1), addr(2), addr(3));
    const IR::Value address2 = ir.CompositeConstruct(addr(4), addr(5), addr(6), addr(7));
    const IR::Value address3 = ir.CompositeConstruct(addr(8), addr(9), addr(10), addr(11));
    const IR::Value address4 = addr(12);
    IR::Value texel = ir.ImageSampleRaw(image_handle, sampler_handle, address1, address2,
                                        address3, address4, info);
    if (info.is_depth && !gather) {
        // Depth compares return one value.
        texel = ir.CompositeExtract(texel, 0);
    }
    return texel;
}

} // namespace

void Translator::IMAGE_SAMPLE(const GcnInst& inst) {
    const auto& mimg = inst.control.mimg;
    IR::VectorReg dst{inst.dst[0].code};
    const bool is_depth = MimgModifierFlags(mimg.mod).test(MimgModifier::Pcf);
    const IR::Value texel = EmitImageSample(ir, inst, false, pc);
    for (u32 i = 0; i < 4; i++) {
        if (((mimg.dmask >> i) & 1) == 0) {
            continue;
        }
        IR::F32 value;
        if (is_depth) {
            // The compare result in RGB, 1 in A.
            value = i < 3 ? IR::F32{texel} : ir.Imm32(1.0f);
        } else {
            value = IR::F32{ir.CompositeExtract(texel, i)};
        }
        ir.SetVectorReg(dst++, value);
    }
}

void Translator::IMAGE_GATHER(const GcnInst& inst) {
    const auto& mimg = inst.control.mimg;
    IR::VectorReg dst{inst.dst[0].code};
    // DMASK selects the one component gathered (R for depth).
    ASSERT_MSG(std::popcount(mimg.dmask) == 1, "Unexpected bits in gather dmask");
    ASSERT(!MimgModifierFlags(mimg.mod).test(MimgModifier::Pcf) || mimg.dmask & 1);
    const IR::Value texel = EmitImageSample(ir, inst, true, pc);
    for (u32 i = 0; i < 4; i++) {
        ir.SetVectorReg(dst++, IR::F32{ir.CompositeExtract(texel, i)});
    }
}

void Translator::IMAGE_GET_LOD(const GcnInst& inst) {
    const auto& mimg = inst.control.mimg;
    IR::VectorReg dst{inst.dst[0].code};
    IR::TextureInstInfo info{};
    info.is_array.Assign(mimg.da);
    info.is_r128.Assign(mimg.r128);
    const TSharp tsharp = LoadTSharp(ir, inst);
    const IR::Value body = Vgpr4<IR::F32>(ir, IR::VectorReg(inst.src[0].code));
    const IR::Value lod = ir.ImageQueryLod(ir.ImageHandle(tsharp.low, tsharp.high), body, info);
    if (mimg.dmask & 1) {
        ir.SetVectorReg(dst++, IR::F32{ir.CompositeExtract(lod, 0)});
    }
    if (mimg.dmask & 2) {
        ir.SetVectorReg(dst++, IR::F32{ir.CompositeExtract(lod, 1)});
    }
}

} // namespace Shader::Gcn
