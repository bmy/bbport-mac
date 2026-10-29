// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "shader/ir/ir_emitter.h"

#include <array>
#include <source_location>

#include "common/assert.h"
#include "shader/ir/debug_print.h"

namespace Shader::IR {

namespace {

[[noreturn]] void ThrowInvalidType(Type type,
                                   std::source_location at = std::source_location::current()) {
    UNREACHABLE_MSG("Invalid type = {}, functionName = {}, line = {}", u32(type),
                    at.function_name(), at.line());
}

template <typename... Values>
void ExpectSameTypes(const Value& first, const Values&... rest) {
    if (((first.Type() != rest.Type()) || ...)) {
        UNREACHABLE_MSG("Mismatching types {}{}", first.Type(),
                        ((", " + NameOf(rest.Type())) + ...));
    }
}

bool Is32(Type type) {
    return type == Type::U32 || type == Type::F32;
}
bool Is64(Type type) {
    return type == Type::U64 || type == Type::F64;
}

} // namespace

template <typename T32, typename T64, typename... Args>
Value IREmitter::By32Or64(Type type, Opcode op32, Opcode op64, const Args&... args) {
    if (Is32(type)) {
        return Inst<T32>(op32, args...);
    }
    if (Is64(type)) {
        return Inst<T64>(op64, args...);
    }
    ThrowInvalidType(type);
}

// Immediates.

U1 IREmitter::Imm1(bool value) const {
    return U1{Value{value}};
}
U8 IREmitter::Imm8(u8 value) const {
    return U8{Value{value}};
}
U16 IREmitter::Imm16(u16 value) const {
    return U16{Value{value}};
}
U32 IREmitter::Imm32(u32 value) const {
    return U32{Value{value}};
}
U32 IREmitter::Imm32(s32 value) const {
    return U32{Value{static_cast<u32>(value)}};
}
F32 IREmitter::Imm32(f32 value) const {
    return F32{Value{value}};
}
U64 IREmitter::Imm64(u64 value) const {
    return U64{Value{value}};
}
U64 IREmitter::Imm64(s64 value) const {
    return U64{Value{static_cast<u64>(value)}};
}
F64 IREmitter::Imm64(f64 value) const {
    return F64{Value{value}};
}

// Bit casts.

template <>
U32 IREmitter::BitCast<U32, U1>(const U1& value) {
    return U32{Select(value, Imm32(1), Imm32(0))};
}
template <>
U32 IREmitter::BitCast<U32, F32>(const F32& value) {
    return Inst<U32>(Opcode::BitCastU32F32, value);
}
template <>
F32 IREmitter::BitCast<F32, U32>(const U32& value) {
    return Inst<F32>(Opcode::BitCastF32U32, value);
}
template <>
U16 IREmitter::BitCast<U16, F16>(const F16& value) {
    return Inst<U16>(Opcode::BitCastU16F16, value);
}
template <>
F16 IREmitter::BitCast<F16, U16>(const U16& value) {
    return Inst<F16>(Opcode::BitCastF16U16, value);
}

// Control.

U1 IREmitter::ConditionRef(const U1& value) {
    return Inst<U1>(Opcode::ConditionRef, value);
}
void IREmitter::Prologue() {
    Inst(Opcode::Prologue);
}
void IREmitter::Epilogue() {
    Inst(Opcode::Epilogue);
}
void IREmitter::Discard() {
    Inst(Opcode::Discard);
}
void IREmitter::Discard(const U1& cond) {
    Inst(Opcode::DiscardCond, cond);
}
void IREmitter::Barrier() {
    Inst(Opcode::Barrier);
}
void IREmitter::WorkgroupMemoryBarrier() {
    Inst(Opcode::WorkgroupMemoryBarrier);
}
void IREmitter::DeviceMemoryBarrier() {
    Inst(Opcode::DeviceMemoryBarrier);
}
void IREmitter::EmitVertex() {
    Inst(Opcode::EmitVertex);
}
void IREmitter::EmitPrimitive() {
    Inst(Opcode::EmitPrimitive);
}
U64 IREmitter::Memtime() {
    return Inst<U64>(Opcode::Memtime);
}

// DebugPrint: SPIR-V's NonSemantic DebugPrintf (RenderDoc and the validation layers print it).
// `fmt` is kept as a pointer: it must be a string literal.
void IREmitter::DebugPrint(const char* fmt, boost::container::small_vector<Value, 5> format_args) {
    ASSERT_MSG(format_args.size() < DEBUGPRINT_NUM_FORMAT_ARGS,
               "DebugPrint takes at most {} format arguments", DEBUGPRINT_NUM_FORMAT_ARGS);
    std::array<Value, DEBUGPRINT_NUM_FORMAT_ARGS> args;
    for (size_t i = 0; i < args.size(); ++i) {
        args[i] = i < format_args.size() ? format_args[i] : Inst(Opcode::Void);
    }
    DebugPrintFlags flags;
    flags.num_args.Assign(u32(format_args.size()));
    Inst(Opcode::DebugPrint, Flags{flags}, Value{fmt}, args[0], args[1], args[2], args[3]);
}

// Registers.

U32 IREmitter::GetUserData(IR::ScalarReg reg) {
    ASSERT(static_cast<u32>(reg) < IR::NumScalarRegs);
    return Inst<U32>(Opcode::GetUserData, reg);
}

template <>
U32 IREmitter::GetScalarReg(IR::ScalarReg reg) {
    ASSERT(static_cast<u32>(reg) < IR::NumScalarRegs);
    return Inst<U32>(Opcode::GetScalarRegister, reg);
}
template <>
F32 IREmitter::GetScalarReg(IR::ScalarReg reg) {
    return BitCast<F32>(GetScalarReg<U32>(reg));
}
template <>
U32 IREmitter::GetVectorReg(IR::VectorReg reg) {
    ASSERT(static_cast<u32>(reg) < IR::NumVectorRegs);
    return Inst<U32>(Opcode::GetVectorRegister, reg);
}
template <>
F32 IREmitter::GetVectorReg(IR::VectorReg reg) {
    return BitCast<F32>(GetVectorReg<U32>(reg));
}

void IREmitter::SetScalarReg(IR::ScalarReg reg, const U32F32& value) {
    ASSERT(reg < IR::ScalarReg::Max);
    const U32 bits = value.Type() == Type::F32 ? BitCast<U32>(F32{value}) : U32{value};
    Inst(Opcode::SetScalarRegister, reg, bits);
}
void IREmitter::SetVectorReg(IR::VectorReg reg, const U32F32& value) {
    ASSERT(reg < IR::VectorReg::Max);
    const U32 bits = value.Type() == Type::F32 ? BitCast<U32>(F32{value}) : U32{value};
    Inst(Opcode::SetVectorRegister, reg, bits);
}

Value IREmitter::GetVirtualReg(IR::VirtualReg reg) {
    return Inst(Opcode::GetVirtualRegister, reg);
}
void IREmitter::SetVirtualReg(IR::VirtualReg reg, const Value& value) {
    Inst(Opcode::SetVirtualRegister, reg, value);
}

U1 IREmitter::GetGotoVariable(u32 id) {
    return Inst<U1>(Opcode::GetGotoVariable, id);
}
void IREmitter::SetGotoVariable(u32 id, const U1& value) {
    Inst(Opcode::SetGotoVariable, id, value);
}

U1 IREmitter::GetScc() {
    return Inst<U1>(Opcode::GetScc);
}
U1 IREmitter::GetExec() {
    return Inst<U1>(Opcode::GetExec);
}
U32 IREmitter::GetVccLo() {
    return Inst<U32>(Opcode::GetVccLo);
}
U32 IREmitter::GetVccHi() {
    return Inst<U32>(Opcode::GetVccHi);
}
U32 IREmitter::GetM0() {
    return Inst<U32>(Opcode::GetM0);
}
void IREmitter::SetScc(const U1& value) {
    Inst(Opcode::SetScc, value);
}
void IREmitter::SetExec(const U1& value) {
    Inst(Opcode::SetExec, value);
}
void IREmitter::SetVccLo(const U32& value) {
    Inst(Opcode::SetVccLo, value);
}
void IREmitter::SetVccHi(const U32& value) {
    Inst(Opcode::SetVccHi, value);
}
void IREmitter::SetM0(const U32& value) {
    Inst(Opcode::SetM0, value);
}

U1 IREmitter::Condition(IR::Condition cond) {
    switch (cond) {
    case IR::Condition::False:
        return Imm1(false);
    case IR::Condition::True:
        return Imm1(true);
    case IR::Condition::Scc0:
        return LogicalNot(GetScc());
    case IR::Condition::Scc1:
        return GetScc();
    case IR::Condition::Vccz:
        return LogicalNot(InverseBallot(PackUint2x32(CompositeConstruct(GetVccLo(), GetVccHi()))));
    case IR::Condition::Vccnz:
        return InverseBallot(PackUint2x32(CompositeConstruct(GetVccLo(), GetVccHi())));
    case IR::Condition::Execz:
        return LogicalNot(GetExec());
    case IR::Condition::Execnz:
        return GetExec();
    }
    UNREACHABLE_MSG("condition {}", u32(cond));
}

// Inputs and outputs.

F32 IREmitter::GetAttribute(IR::Attribute attribute, u32 comp, u32 index) {
    return Inst<F32>(Opcode::GetAttribute, attribute, Imm32(comp), Imm32(index));
}
U1 IREmitter::GetAttributeU1(IR::Attribute attribute, u32 comp) {
    return Inst<U1>(Opcode::GetAttributeU1, attribute, Imm32(comp));
}
U32 IREmitter::GetAttributeU32(IR::Attribute attribute, u32 comp) {
    return Inst<U32>(Opcode::GetAttributeU32, attribute, Imm32(comp));
}
void IREmitter::SetAttribute(IR::Attribute attribute, const F32& value, u32 comp) {
    Inst(Opcode::SetAttribute, attribute, value, Imm32(comp));
}
F32 IREmitter::GetTessGenericAttribute(const U32& vertex_index, const U32& attr_index,
                                       const U32& comp_index) {
    return Inst<F32>(Opcode::GetTessGenericAttribute, vertex_index, attr_index, comp_index);
}
void IREmitter::SetTcsGenericAttribute(const F32& value, const U32& attr_index,
                                       const U32& comp_index) {
    Inst(Opcode::SetTcsGenericAttribute, value, attr_index, comp_index);
}
F32 IREmitter::ReadTcsGenericOuputAttribute(const U32& vertex_index, const U32& attr_index,
                                            const U32& comp_index) {
    return Inst<F32>(Opcode::ReadTcsGenericOuputAttribute, vertex_index, attr_index, comp_index);
}
U32 IREmitter::GetPcLo(const U32& pc) {
    return Inst<U32>(Opcode::GetPcLo, pc);
}
F32 IREmitter::GetPatch(Patch patch) {
    return Inst<F32>(Opcode::GetPatch, patch);
}
void IREmitter::SetPatch(Patch patch, const F32& value) {
    Inst(Opcode::SetPatch, patch, value);
}

// Shared memory.

Value IREmitter::LoadShared(int bit_size, bool, const U32& offset, bool is_gds) {
    switch (bit_size) {
    case 16:
        return Inst<U16>(Opcode::LoadSharedU16, Flags{is_gds}, offset);
    case 32:
        return Inst<U32>(Opcode::LoadSharedU32, Flags{is_gds}, offset);
    case 64:
        return Inst<U64>(Opcode::LoadSharedU64, Flags{is_gds}, offset);
    default:
        UNREACHABLE_MSG("Invalid bit size {}", bit_size);
    }
}

void IREmitter::WriteShared(int bit_size, const Value& value, const U32& offset, bool is_gds) {
    switch (bit_size) {
    case 16:
        Inst(Opcode::WriteSharedU16, Flags{is_gds}, offset, value);
        break;
    case 32:
        Inst(Opcode::WriteSharedU32, Flags{is_gds}, offset, value);
        break;
    case 64:
        Inst(Opcode::WriteSharedU64, Flags{is_gds}, offset, value);
        break;
    default:
        UNREACHABLE_MSG("Invalid bit size {}", bit_size);
    }
}

namespace {
/// The 32- and 64-bit opcodes of a shared atomic.
struct AtomicOps {
    Opcode op32;
    Opcode op64;
};
} // namespace

#define BB_SHARED_ATOMIC(name, op)                                                                \
    U32U64 IREmitter::SharedAtomic##name(const U32& address, const U32U64& data, bool is_gds) {   \
        switch (data.Type()) {                                                                    \
        case Type::U32:                                                                           \
            return Inst<U32>(Opcode::SharedAtomic##op##32, Flags{is_gds}, address, data);         \
        case Type::U64:                                                                           \
            return Inst<U64>(Opcode::SharedAtomic##op##64, Flags{is_gds}, address, data);         \
        default:                                                                                  \
            ThrowInvalidType(data.Type());                                                        \
        }                                                                                         \
    }
BB_SHARED_ATOMIC(IAdd, IAdd)
BB_SHARED_ATOMIC(ISub, ISub)
BB_SHARED_ATOMIC(And, And)
BB_SHARED_ATOMIC(Or, Or)
BB_SHARED_ATOMIC(Xor, Xor)
#undef BB_SHARED_ATOMIC

U32U64 IREmitter::SharedAtomicIMin(const U32& address, const U32U64& data, bool is_signed,
                                   bool is_gds) {
    switch (data.Type()) {
    case Type::U32:
        return Inst<U32>(is_signed ? Opcode::SharedAtomicSMin32 : Opcode::SharedAtomicUMin32,
                         Flags{is_gds}, address, data);
    case Type::U64:
        return Inst<U64>(is_signed ? Opcode::SharedAtomicSMin64 : Opcode::SharedAtomicUMin64,
                         Flags{is_gds}, address, data);
    default:
        ThrowInvalidType(data.Type());
    }
}

U32U64 IREmitter::SharedAtomicIMax(const U32& address, const U32U64& data, bool is_signed,
                                   bool is_gds) {
    switch (data.Type()) {
    case Type::U32:
        return Inst<U32>(is_signed ? Opcode::SharedAtomicSMax32 : Opcode::SharedAtomicUMax32,
                         Flags{is_gds}, address, data);
    case Type::U64:
        return Inst<U64>(is_signed ? Opcode::SharedAtomicSMax64 : Opcode::SharedAtomicUMax64,
                         Flags{is_gds}, address, data);
    default:
        ThrowInvalidType(data.Type());
    }
}

U32U64 IREmitter::SharedAtomicCmpSwap(const U32& address, const U32U64& value,
                                      const U32U64& cmp_value, bool is_gds) {
    switch (value.Type()) {
    case Type::U32:
        return Inst<U32>(Opcode::SharedAtomicCmpSwap32, Flags{is_gds}, address, value, cmp_value);
    case Type::U64:
        return Inst<U64>(Opcode::SharedAtomicCmpSwap64, Flags{is_gds}, address, value, cmp_value);
    default:
        ThrowInvalidType(value.Type());
    }
}

template <>
U32 IREmitter::SharedAtomicInc(const U32& address, bool is_gds) {
    return Inst<U32>(Opcode::SharedAtomicInc32, Flags{is_gds}, address);
}
template <>
U64 IREmitter::SharedAtomicInc(const U32& address, bool is_gds) {
    return Inst<U64>(Opcode::SharedAtomicInc64, Flags{is_gds}, address);
}
template <>
U32 IREmitter::SharedAtomicDec(const U32& address, bool is_gds) {
    return Inst<U32>(Opcode::SharedAtomicDec32, Flags{is_gds}, address);
}
template <>
U64 IREmitter::SharedAtomicDec(const U32& address, bool is_gds) {
    return Inst<U64>(Opcode::SharedAtomicDec64, Flags{is_gds}, address);
}

// Constants and buffers.

U32 IREmitter::ReadConst(const Value& base, const U32& offset) {
    return Inst<U32>(Opcode::ReadConst, base, offset);
}
U32 IREmitter::ReadConstBuffer(const Value& handle, const U32& index, BufferInstInfo info) {
    return Inst<U32>(Opcode::ReadConstBuffer, Flags{info}, handle, index);
}

U8 IREmitter::LoadBufferU8(const Value& handle, const Value& address, BufferInstInfo info) {
    return Inst<U8>(Opcode::LoadBufferU8, Flags{info}, handle, address);
}
U16 IREmitter::LoadBufferU16(const Value& handle, const Value& address, BufferInstInfo info) {
    return Inst<U16>(Opcode::LoadBufferU16, Flags{info}, handle, address);
}
U64 IREmitter::LoadBufferU64(const Value& handle, const Value& address, BufferInstInfo info) {
    return Inst<U64>(Opcode::LoadBufferU64, Flags{info}, handle, address);
}
Value IREmitter::LoadBufferFormat(const Value& handle, const Value& address, BufferInstInfo info) {
    return Inst(Opcode::LoadBufferFormatF32, Flags{info}, handle, address);
}

Value IREmitter::LoadBufferU32(int num_dwords, const Value& handle, const Value& address,
                               BufferInstInfo info) {
    static constexpr std::array ops{Opcode::LoadBufferU32, Opcode::LoadBufferU32x2,
                                    Opcode::LoadBufferU32x3, Opcode::LoadBufferU32x4};
    ASSERT_MSG(num_dwords >= 1 && num_dwords <= 4, "Invalid number of dwords {}", num_dwords);
    return Inst(ops[num_dwords - 1], Flags{info}, handle, address);
}

Value IREmitter::LoadBufferF32(int num_dwords, const Value& handle, const Value& address,
                               BufferInstInfo info) {
    static constexpr std::array ops{Opcode::LoadBufferF32, Opcode::LoadBufferF32x2,
                                    Opcode::LoadBufferF32x3, Opcode::LoadBufferF32x4};
    ASSERT_MSG(num_dwords >= 1 && num_dwords <= 4, "Invalid number of dwords {}", num_dwords);
    return Inst(ops[num_dwords - 1], Flags{info}, handle, address);
}

void IREmitter::StoreBufferU8(const Value& handle, const Value& address, const U8& data,
                              BufferInstInfo info) {
    Inst(Opcode::StoreBufferU8, Flags{info}, handle, address, data);
}
void IREmitter::StoreBufferU16(const Value& handle, const Value& address, const U16& data,
                               BufferInstInfo info) {
    Inst(Opcode::StoreBufferU16, Flags{info}, handle, address, data);
}
void IREmitter::StoreBufferU64(const Value& handle, const Value& address, const U64& data,
                               BufferInstInfo info) {
    Inst(Opcode::StoreBufferU64, Flags{info}, handle, address, data);
}
void IREmitter::StoreBufferFormat(const Value& handle, const Value& address, const Value& data,
                                  BufferInstInfo info) {
    Inst(Opcode::StoreBufferFormatF32, Flags{info}, handle, address, data);
}

void IREmitter::StoreBufferU32(int num_dwords, const Value& handle, const Value& address,
                               const Value& data, BufferInstInfo info) {
    static constexpr std::array ops{Opcode::StoreBufferU32, Opcode::StoreBufferU32x2,
                                    Opcode::StoreBufferU32x3, Opcode::StoreBufferU32x4};
    ASSERT_MSG(num_dwords >= 1 && num_dwords <= 4, "Invalid number of dwords {}", num_dwords);
    Inst(ops[num_dwords - 1], Flags{info}, handle, address, data);
}

void IREmitter::StoreBufferF32(int num_dwords, const Value& handle, const Value& address,
                               const Value& data, BufferInstInfo info) {
    static constexpr std::array ops{Opcode::StoreBufferF32, Opcode::StoreBufferF32x2,
                                    Opcode::StoreBufferF32x3, Opcode::StoreBufferF32x4};
    ASSERT_MSG(num_dwords >= 1 && num_dwords <= 4, "Invalid number of dwords {}", num_dwords);
    Inst(ops[num_dwords - 1], Flags{info}, handle, address, data);
}

Value IREmitter::BufferAtomicIAdd(const Value& handle, const Value& address, const Value& value,
                                  BufferInstInfo info) {
    switch (value.Type()) {
    case Type::U32:
        return Inst(Opcode::BufferAtomicIAdd32, Flags{info}, handle, address, value);
    case Type::U64:
        return Inst(Opcode::BufferAtomicIAdd64, Flags{info}, handle, address, value);
    default:
        ThrowInvalidType(value.Type());
    }
}

Value IREmitter::BufferAtomicIMin(const Value& handle, const Value& address, const Value& value,
                                  bool is_signed, BufferInstInfo info) {
    switch (value.Type()) {
    case Type::U32:
        return Inst(is_signed ? Opcode::BufferAtomicSMin32 : Opcode::BufferAtomicUMin32,
                    Flags{info}, handle, address, value);
    case Type::U64:
        return Inst(is_signed ? Opcode::BufferAtomicSMin64 : Opcode::BufferAtomicUMin64,
                    Flags{info}, handle, address, value);
    default:
        ThrowInvalidType(value.Type());
    }
}

Value IREmitter::BufferAtomicIMax(const Value& handle, const Value& address, const Value& value,
                                  bool is_signed, BufferInstInfo info) {
    switch (value.Type()) {
    case Type::U32:
        return Inst(is_signed ? Opcode::BufferAtomicSMax32 : Opcode::BufferAtomicUMax32,
                    Flags{info}, handle, address, value);
    case Type::U64:
        return Inst(is_signed ? Opcode::BufferAtomicSMax64 : Opcode::BufferAtomicUMax64,
                    Flags{info}, handle, address, value);
    default:
        ThrowInvalidType(value.Type());
    }
}

#define BB_BUFFER_ATOMIC(name, op)                                                                \
    Value IREmitter::BufferAtomic##name(const Value& handle, const Value& address,               \
                                        const Value& value, BufferInstInfo info) {                \
        return Inst(Opcode::BufferAtomic##op, Flags{info}, handle, address, value);              \
    }
BB_BUFFER_ATOMIC(ISub, ISub32)
BB_BUFFER_ATOMIC(FMin, FMin32)
BB_BUFFER_ATOMIC(FMax, FMax32)
BB_BUFFER_ATOMIC(And, And32)
BB_BUFFER_ATOMIC(Or, Or32)
BB_BUFFER_ATOMIC(Xor, Xor32)
BB_BUFFER_ATOMIC(Swap, Swap32)
#undef BB_BUFFER_ATOMIC

Value IREmitter::BufferAtomicInc(const Value& handle, const Value& address, BufferInstInfo info) {
    return Inst(Opcode::BufferAtomicInc32, Flags{info}, handle, address);
}
Value IREmitter::BufferAtomicDec(const Value& handle, const Value& address, BufferInstInfo info) {
    return Inst(Opcode::BufferAtomicDec32, Flags{info}, handle, address);
}
Value IREmitter::BufferAtomicCmpSwap(const Value& handle, const Value& address, const Value& vdata,
                                     const Value& cmp_value, BufferInstInfo info) {
    return Inst(Opcode::BufferAtomicCmpSwap32, Flags{info}, handle, address, vdata, cmp_value);
}
Value IREmitter::BufferAtomicFCmpSwap(const Value& handle, const Value& address, const Value& vdata,
                                      const Value& cmp_value, BufferInstInfo info) {
    return Inst(Opcode::BufferAtomicFCmpSwap32, Flags{info}, handle, address, vdata, cmp_value);
}

U32 IREmitter::DataAppend(const U32& gds_dw_offset) {
    return Inst<U32>(Opcode::DataAppend, gds_dw_offset, GetExec());
}
U32 IREmitter::DataConsume(const U32& gds_dw_offset) {
    return Inst<U32>(Opcode::DataConsume, gds_dw_offset, GetExec());
}

// Lanes and waves.

U32 IREmitter::LaneId() {
    return Inst<U32>(Opcode::LaneId);
}
U32 IREmitter::WarpId() {
    return Inst<U32>(Opcode::WarpId);
}
U32 IREmitter::QuadBroadcast(const U32& value, const U32& index) {
    return Inst<U32>(Opcode::QuadBroadcast, value, index);
}
U32 IREmitter::Shuffle(const U32& value, const U32& index) {
    return Inst<U32>(Opcode::Shuffle, value, index);
}
U32 IREmitter::ShuffleXor(const U32& value, const U32& mask) {
    return Inst<U32>(Opcode::ShuffleXor, value, mask);
}
U32 IREmitter::ReadFirstLane(const U32& value) {
    return Inst<U32>(Opcode::ReadFirstLane, value);
}
U32 IREmitter::ReadLane(const U32& value, const U32& lane) {
    return Inst<U32>(Opcode::ReadLane, value, lane);
}
U32 IREmitter::WriteLane(const U32& value, const U32& write_value, const U32& lane) {
    return Inst<U32>(Opcode::WriteLane, value, write_value, lane);
}
U64 IREmitter::Ballot(const U1& bit) {
    return Inst<U64>(Opcode::Ballot, bit);
}
U32 IREmitter::BallotFindLsb(const U64& mask) {
    return Inst<U32>(Opcode::BallotFindLsb, mask);
}
U1 IREmitter::InverseBallot(const U64& mask) {
    return Inst<U1>(Opcode::InverseBallot, mask);
}
U1 IREmitter::GroupAny(const U1& bit) {
    return Inst<U1>(Opcode::GroupAny, bit);
}

// Composites.

Value IREmitter::CompositeConstruct(const Value& e1, const Value& e2) {
    ExpectSameTypes(e1, e2);
    switch (e1.Type()) {
    case Type::U32:
        return Inst(Opcode::CompositeConstructU32x2, e1, e2);
    case Type::U32x2:
        return Inst(Opcode::CompositeConstructU32x2x2, e1, e2);
    case Type::F32:
        return Inst(Opcode::CompositeConstructF32x2, e1, e2);
    case Type::F32x2:
        return Inst(Opcode::CompositeConstructF32x2x2, e1, e2);
    default:
        ThrowInvalidType(e1.Type());
    }
}

Value IREmitter::CompositeConstruct(const Value& e1, const Value& e2, const Value& e3) {
    ExpectSameTypes(e1, e2, e3);
    switch (e1.Type()) {
    case Type::U32:
        return Inst(Opcode::CompositeConstructU32x3, e1, e2, e3);
    case Type::F32:
        return Inst(Opcode::CompositeConstructF32x3, e1, e2, e3);
    default:
        ThrowInvalidType(e1.Type());
    }
}

Value IREmitter::CompositeConstruct(const Value& e1, const Value& e2, const Value& e3,
                                    const Value& e4) {
    ExpectSameTypes(e1, e2, e3, e4);
    switch (e1.Type()) {
    case Type::U32:
        return Inst(Opcode::CompositeConstructU32x4, e1, e2, e3, e4);
    case Type::F32:
        return Inst(Opcode::CompositeConstructF32x4, e1, e2, e3, e4);
    default:
        ThrowInvalidType(e1.Type());
    }
}

Value IREmitter::CompositeConstruct(std::span<const Value> elements) {
    switch (elements.size()) {
    case 2:
        return CompositeConstruct(elements[0], elements[1]);
    case 3:
        return CompositeConstruct(elements[0], elements[1], elements[2]);
    case 4:
        return CompositeConstruct(elements[0], elements[1], elements[2], elements[3]);
    default:
        UNREACHABLE_MSG("composite of {} elements", elements.size());
    }
}

namespace {
/// The operation for a vector type among U32x2..4 and F32x2..4, and its element count.
struct VectorOp {
    Opcode op;
    size_t size;
};
template <size_t N>
VectorOp VectorOpFor(Type type, const std::array<Opcode, N>& u32_ops,
                     const std::array<Opcode, N>& f32_ops) {
    switch (type) {
    case Type::U32x2:
        return {u32_ops[0], 2};
    case Type::U32x3:
        return {u32_ops[1], 3};
    case Type::U32x4:
        return {u32_ops[2], 4};
    case Type::F32x2:
        return {f32_ops[0], 2};
    case Type::F32x3:
        return {f32_ops[1], 3};
    case Type::F32x4:
        return {f32_ops[2], 4};
    default:
        ThrowInvalidType(type);
    }
}
} // namespace

Value IREmitter::CompositeExtract(const Value& vector, size_t element) {
    const auto [op, size] = VectorOpFor<3>(
        vector.Type(),
        {Opcode::CompositeExtractU32x2, Opcode::CompositeExtractU32x3,
         Opcode::CompositeExtractU32x4},
        {Opcode::CompositeExtractF32x2, Opcode::CompositeExtractF32x3,
         Opcode::CompositeExtractF32x4});
    ASSERT_MSG(element < size, "Out of bounds element {}", element);
    return Inst(op, vector, Value{static_cast<u32>(element)});
}

Value IREmitter::CompositeInsert(const Value& vector, const Value& object, size_t element) {
    const auto [op, size] = VectorOpFor<3>(
        vector.Type(),
        {Opcode::CompositeInsertU32x2, Opcode::CompositeInsertU32x3, Opcode::CompositeInsertU32x4},
        {Opcode::CompositeInsertF32x2, Opcode::CompositeInsertF32x3,
         Opcode::CompositeInsertF32x4});
    ASSERT_MSG(element < size, "Out of bounds element {}", element);
    return Inst(op, vector, object, Value{static_cast<u32>(element)});
}

Value IREmitter::CompositeShuffle(const Value& vector1, const Value& vector2, size_t comp0,
                                  size_t comp1) {
    ExpectSameTypes(vector1, vector2);
    ASSERT_MSG(comp0 < 4 && comp1 < 4, "shuffle elements {}, {}", comp0, comp1);
    switch (vector1.Type()) {
    case Type::U32x4:
        return Inst(Opcode::CompositeShuffleU32x2, vector1, vector2, Value{u32(comp0)},
                    Value{u32(comp1)});
    case Type::F32x4:
        return Inst(Opcode::CompositeShuffleF32x2, vector1, vector2, Value{u32(comp0)},
                    Value{u32(comp1)});
    default:
        ThrowInvalidType(vector1.Type());
    }
}

Value IREmitter::CompositeShuffle(const Value& vector1, const Value& vector2, size_t comp0,
                                  size_t comp1, size_t comp2) {
    ExpectSameTypes(vector1, vector2);
    ASSERT_MSG(comp0 < 6 && comp1 < 6 && comp2 < 6, "shuffle elements {}, {}, {}", comp0, comp1,
               comp2);
    switch (vector1.Type()) {
    case Type::U32x4:
        return Inst(Opcode::CompositeShuffleU32x3, vector1, vector2, Value{u32(comp0)},
                    Value{u32(comp1)}, Value{u32(comp2)});
    case Type::F32x4:
        return Inst(Opcode::CompositeShuffleF32x3, vector1, vector2, Value{u32(comp0)},
                    Value{u32(comp1)}, Value{u32(comp2)});
    default:
        ThrowInvalidType(vector1.Type());
    }
}

Value IREmitter::CompositeShuffle(const Value& vector1, const Value& vector2, size_t comp0,
                                  size_t comp1, size_t comp2, size_t comp3) {
    ExpectSameTypes(vector1, vector2);
    ASSERT_MSG(comp0 < 8 && comp1 < 8 && comp2 < 8 && comp3 < 8,
               "shuffle elements {}, {}, {}, {}", comp0, comp1, comp2, comp3);
    switch (vector1.Type()) {
    case Type::U32x4:
        return Inst(Opcode::CompositeShuffleU32x4, vector1, vector2, Value{u32(comp0)},
                    Value{u32(comp1)}, Value{u32(comp2)}, Value{u32(comp3)});
    case Type::F32x4:
        return Inst(Opcode::CompositeShuffleF32x4, vector1, vector2, Value{u32(comp0)},
                    Value{u32(comp1)}, Value{u32(comp2)}, Value{u32(comp3)});
    default:
        ThrowInvalidType(vector1.Type());
    }
}

Value IREmitter::Select(const U1& condition, const Value& true_value, const Value& false_value) {
    ExpectSameTypes(true_value, false_value);
    switch (true_value.Type()) {
    case Type::U1:
        return Inst(Opcode::SelectU1, condition, true_value, false_value);
    case Type::U32:
        return Inst(Opcode::SelectU32, condition, true_value, false_value);
    case Type::F32:
        return Inst(Opcode::SelectF32, condition, true_value, false_value);
    case Type::U64:
        return Inst(Opcode::SelectU64, condition, true_value, false_value);
    default:
        UNREACHABLE_MSG("Invalid type {}", true_value.Type());
    }
}

// Packing.

U64 IREmitter::PackUint2x32(const Value& vector) {
    return Inst<U64>(Opcode::PackUint2x32, vector);
}
Value IREmitter::UnpackUint2x32(const U64& value) {
    return Inst<Value>(Opcode::UnpackUint2x32, value);
}
F64 IREmitter::PackDouble2x32(const Value& vector) {
    return Inst<F64>(Opcode::PackDouble2x32, vector);
}
Value IREmitter::UnpackDouble2x32(const F64& value) {
    return Inst<Value>(Opcode::UnpackDouble2x32, value);
}

namespace {
/// Operations of a packed layout by number format (Undefined... no: Phi marks "none").
struct PackOps {
    Opcode unorm = Opcode::Phi;
    Opcode snorm = Opcode::Phi;
    Opcode uint = Opcode::Phi;
    Opcode sint = Opcode::Phi;
    Opcode fp = Opcode::Phi;

    Opcode For(::Gcn::NumberFormat format, const char* layout) const {
        Opcode op = Opcode::Phi;
        switch (format) {
        case ::Gcn::NumberFormat::Unorm:
            op = unorm;
            break;
        case ::Gcn::NumberFormat::Snorm:
            op = snorm;
            break;
        case ::Gcn::NumberFormat::Uint:
            op = uint;
            break;
        case ::Gcn::NumberFormat::Sint:
            op = sint;
            break;
        case ::Gcn::NumberFormat::Float:
            op = fp;
            break;
        default:
            break;
        }
        ASSERT_MSG(op != Opcode::Phi, "Unsupported {} number format: {}", layout, u32(format));
        return op;
    }
};
} // namespace

U32 IREmitter::Pack2x16(::Gcn::NumberFormat number_format, const Value& vector) {
    static constexpr PackOps ops{Opcode::PackUnorm2x16, Opcode::PackSnorm2x16,
                                 Opcode::PackUint2x16, Opcode::PackSint2x16, Opcode::PackHalf2x16};
    return Inst<U32>(ops.For(number_format, "2x16"), vector);
}
Value IREmitter::Unpack2x16(::Gcn::NumberFormat number_format, const U32& value) {
    static constexpr PackOps ops{Opcode::UnpackUnorm2x16, Opcode::UnpackSnorm2x16,
                                 Opcode::UnpackUint2x16, Opcode::UnpackSint2x16,
                                 Opcode::UnpackHalf2x16};
    return Inst(ops.For(number_format, "2x16"), value);
}
U32 IREmitter::Pack4x8(::Gcn::NumberFormat number_format, const Value& vector) {
    static constexpr PackOps ops{Opcode::PackUnorm4x8, Opcode::PackSnorm4x8, Opcode::PackUint4x8,
                                 Opcode::PackSint4x8};
    return Inst<U32>(ops.For(number_format, "4x8"), vector);
}
Value IREmitter::Unpack4x8(::Gcn::NumberFormat number_format, const U32& value) {
    static constexpr PackOps ops{Opcode::UnpackUnorm4x8, Opcode::UnpackSnorm4x8,
                                 Opcode::UnpackUint4x8, Opcode::UnpackSint4x8};
    return Inst(ops.For(number_format, "4x8"), value);
}
U32 IREmitter::Pack10_11_11(::Gcn::NumberFormat number_format, const Value& vector) {
    static constexpr PackOps ops{.fp = Opcode::PackUfloat10_11_11};
    return Inst<U32>(ops.For(number_format, "10_11_11"), vector);
}
Value IREmitter::Unpack10_11_11(::Gcn::NumberFormat number_format, const U32& value) {
    static constexpr PackOps ops{.fp = Opcode::UnpackUfloat10_11_11};
    return Inst(ops.For(number_format, "10_11_11"), value);
}
U32 IREmitter::Pack2_10_10_10(::Gcn::NumberFormat number_format, const Value& vector) {
    static constexpr PackOps ops{Opcode::PackUnorm2_10_10_10, Opcode::PackSnorm2_10_10_10,
                                 Opcode::PackUint2_10_10_10, Opcode::PackSint2_10_10_10};
    return Inst<U32>(ops.For(number_format, "2_10_10_10"), vector);
}
Value IREmitter::Unpack2_10_10_10(::Gcn::NumberFormat number_format, const U32& value) {
    static constexpr PackOps ops{Opcode::UnpackUnorm2_10_10_10, Opcode::UnpackSnorm2_10_10_10,
                                 Opcode::UnpackUint2_10_10_10, Opcode::UnpackSint2_10_10_10};
    return Inst(ops.For(number_format, "2_10_10_10"), value);
}

// Floating point.

F32F64 IREmitter::FPAdd(const F32F64& a, const F32F64& b) {
    ExpectSameTypes(a, b);
    return F32F64{By32Or64<F32, F64>(a.Type(), Opcode::FPAdd32, Opcode::FPAdd64, a, b)};
}

F32F64 IREmitter::FPSub(const F32F64& a, const F32F64& b) {
    ExpectSameTypes(a, b);
    if (a.Type() != Type::F32) {
        ThrowInvalidType(a.Type());
    }
    return Inst<F32>(Opcode::FPSub32, a, b);
}

F32F64 IREmitter::FPMul(const F32F64& a, const F32F64& b) {
    ExpectSameTypes(a, b);
    return F32F64{By32Or64<F32, F64>(a.Type(), Opcode::FPMul32, Opcode::FPMul64, a, b)};
}

F32F64 IREmitter::FPDiv(const F32F64& a, const F32F64& b) {
    ExpectSameTypes(a, b);
    return F32F64{By32Or64<F32, F64>(a.Type(), Opcode::FPDiv32, Opcode::FPDiv64, a, b)};
}

F32F64 IREmitter::FPFma(const F32F64& a, const F32F64& b, const F32F64& c) {
    ExpectSameTypes(a, b, c);
    return F32F64{By32Or64<F32, F64>(a.Type(), Opcode::FPFma32, Opcode::FPFma64, a, b, c)};
}

#define BB_FP_UNARY(name, op)                                                                     \
    F32F64 IREmitter::name(const F32F64& value) {                                                 \
        return F32F64{                                                                            \
            By32Or64<F32, F64>(value.Type(), Opcode::op##32, Opcode::op##64, value)};             \
    }
BB_FP_UNARY(FPAbs, FPAbs)
BB_FP_UNARY(FPNeg, FPNeg)
BB_FP_UNARY(FPRecip, FPRecip)
BB_FP_UNARY(FPRecipSqrt, FPRecipSqrt)
BB_FP_UNARY(FPSaturate, FPSaturate)
BB_FP_UNARY(FPRoundEven, FPRoundEven)
BB_FP_UNARY(FPFloor, FPFloor)
BB_FP_UNARY(FPCeil, FPCeil)
BB_FP_UNARY(FPTrunc, FPTrunc)
BB_FP_UNARY(FPFract, FPFract)
BB_FP_UNARY(FPFrexpSig, FPFrexpSig)
#undef BB_FP_UNARY

F32F64 IREmitter::FPAbsNeg(const F32F64& value, bool abs, bool neg) {
    F32F64 result = value;
    if (abs) {
        result = FPAbs(result);
    }
    if (neg) {
        result = FPNeg(result);
    }
    return result;
}

F32 IREmitter::FPCos(const F32& value) {
    return Inst<F32>(Opcode::FPCos, value);
}
F32 IREmitter::FPSin(const F32& value) {
    return Inst<F32>(Opcode::FPSin, value);
}
F32 IREmitter::FPExp2(const F32& value) {
    return Inst<F32>(Opcode::FPExp2, value);
}
F32 IREmitter::FPLdexp(const F32& value, const U32& exp) {
    return Inst<F32>(Opcode::FPLdexp, value, exp);
}
F32 IREmitter::FPLog2(const F32& value) {
    return Inst<F32>(Opcode::FPLog2, value);
}
F32 IREmitter::FPPow(const F32& x, const F32& y) {
    return Inst<F32>(Opcode::FPPow, x, y);
}
F32 IREmitter::FPSqrt(const F32& value) {
    return Inst<F32>(Opcode::FPSqrt, value);
}

F32F64 IREmitter::FPClamp(const F32F64& value, const F32F64& min_value, const F32F64& max_value) {
    ExpectSameTypes(value, min_value, max_value);
    return F32F64{By32Or64<F32, F64>(value.Type(), Opcode::FPClamp32, Opcode::FPClamp64, value,
                                     min_value, max_value)};
}

U32 IREmitter::FPFrexpExp(const F32F64& value) {
    return U32{By32Or64<U32, U32>(value.Type(), Opcode::FPFrexpExp32, Opcode::FPFrexpExp64,
                                  value)};
}

#define BB_FP_COMPARE(name, op)                                                                   \
    U1 IREmitter::name(const F32F64& lhs, const F32F64& rhs, bool ordered) {                      \
        ExpectSameTypes(lhs, rhs);                                                                \
        return U1{By32Or64<U1, U1>(lhs.Type(),                                                    \
                                   ordered ? Opcode::FPOrd##op##32 : Opcode::FPUnord##op##32,     \
                                   ordered ? Opcode::FPOrd##op##64 : Opcode::FPUnord##op##64,     \
                                   lhs, rhs)};                                                    \
    }
BB_FP_COMPARE(FPEqual, Equal)
BB_FP_COMPARE(FPNotEqual, NotEqual)
BB_FP_COMPARE(FPLessThan, LessThan)
BB_FP_COMPARE(FPGreaterThan, GreaterThan)
BB_FP_COMPARE(FPLessThanEqual, LessThanEqual)
BB_FP_COMPARE(FPGreaterThanEqual, GreaterThanEqual)
#undef BB_FP_COMPARE

U1 IREmitter::FPIsNan(const F32F64& value) {
    return U1{By32Or64<U1, U1>(value.Type(), Opcode::FPIsNan32, Opcode::FPIsNan64, value)};
}
U1 IREmitter::FPIsInf(const F32F64& value) {
    return U1{By32Or64<U1, U1>(value.Type(), Opcode::FPIsInf32, Opcode::FPIsInf64, value)};
}
U1 IREmitter::FPCmpClass32(const F32& value, const U32& op) {
    return Inst<U1>(Opcode::FPCmpClass32, value, op);
}

U1 IREmitter::FPOrdered(const F32F64& lhs, const F32F64& rhs) {
    ExpectSameTypes(lhs, rhs);
    return LogicalAnd(LogicalNot(FPIsNan(lhs)), LogicalNot(FPIsNan(rhs)));
}
U1 IREmitter::FPUnordered(const F32F64& lhs, const F32F64& rhs) {
    ExpectSameTypes(lhs, rhs);
    return LogicalOr(FPIsNan(lhs), FPIsNan(rhs));
}

F32F64 IREmitter::FPMax(const F32F64& lhs, const F32F64& rhs) {
    ExpectSameTypes(lhs, rhs);
    return F32F64{By32Or64<F32, F64>(lhs.Type(), Opcode::FPMax32, Opcode::FPMax64, lhs, rhs)};
}
F32F64 IREmitter::FPMin(const F32F64& lhs, const F32F64& rhs) {
    ExpectSameTypes(lhs, rhs);
    return F32F64{By32Or64<F32, F64>(lhs.Type(), Opcode::FPMin32, Opcode::FPMin64, lhs, rhs)};
}
F32F64 IREmitter::FPMinTri(const F32F64& a, const F32F64& b, const F32F64& c) {
    return Inst<F32>(Opcode::FPMinTri32, a, b, c);
}
F32F64 IREmitter::FPMaxTri(const F32F64& a, const F32F64& b, const F32F64& c) {
    return Inst<F32>(Opcode::FPMaxTri32, a, b, c);
}
F32F64 IREmitter::FPMedTri(const F32F64& a, const F32F64& b, const F32F64& c) {
    return Inst<F32>(Opcode::FPMedTri32, a, b, c);
}

// Integers.

#define BB_INT_BINARY(name, op)                                                                   \
    U32U64 IREmitter::name(const U32U64& a, const U32U64& b) {                                    \
        ExpectSameTypes(a, b);                                                                    \
        return U32U64{By32Or64<U32, U64>(a.Type(), Opcode::op##32, Opcode::op##64, a, b)};        \
    }
BB_INT_BINARY(IAdd, IAdd)
BB_INT_BINARY(ISub, ISub)
BB_INT_BINARY(IMul, IMul)
BB_INT_BINARY(BitwiseAnd, BitwiseAnd)
BB_INT_BINARY(BitwiseOr, BitwiseOr)
BB_INT_BINARY(BitwiseXor, BitwiseXor)
#undef BB_INT_BINARY

Value IREmitter::IAddCarry(const U32& a, const U32& b) {
    ExpectSameTypes(a, b);
    if (a.Type() != Type::U32) {
        ThrowInvalidType(a.Type());
    }
    return Inst(Opcode::IAddCarry32, a, b);
}

U32 IREmitter::IMulHi(const U32& a, const U32& b, bool is_signed) {
    return Inst<U32>(is_signed ? Opcode::SMulHi : Opcode::UMulHi, a, b);
}
U32 IREmitter::IDiv(const U32& a, const U32& b, bool is_signed) {
    return Inst<U32>(is_signed ? Opcode::SDiv32 : Opcode::UDiv32, a, b);
}
U32 IREmitter::IMod(const U32& a, const U32& b, bool is_signed) {
    return Inst<U32>(is_signed ? Opcode::SMod32 : Opcode::UMod32, a, b);
}
U32U64 IREmitter::INeg(const U32U64& value) {
    return U32U64{By32Or64<U32, U64>(value.Type(), Opcode::INeg32, Opcode::INeg64, value)};
}
U32 IREmitter::IAbs(const U32& value) {
    return Inst<U32>(Opcode::IAbs32, value);
}

#define BB_SHIFT(name)                                                                            \
    U32U64 IREmitter::name(const U32U64& base, const U32& shift) {                                \
        if (shift.IsImmediate() && shift.U32() == 0) {                                            \
            return base;                                                                          \
        }                                                                                         \
        return U32U64{                                                                            \
            By32Or64<U32, U64>(base.Type(), Opcode::name##32, Opcode::name##64, base, shift)};    \
    }
BB_SHIFT(ShiftLeftLogical)
BB_SHIFT(ShiftRightLogical)
BB_SHIFT(ShiftRightArithmetic)
#undef BB_SHIFT

U32U64 IREmitter::BitFieldInsert(const U32U64& base, const U32U64& insert, const U32& offset,
                                 const U32& count) {
    ExpectSameTypes(base, insert);
    return U32U64{By32Or64<U32, U64>(base.Type(), Opcode::BitFieldInsert32,
                                     Opcode::BitFieldInsert64, base, insert, offset, count)};
}
U32 IREmitter::BitFieldExtract(const U32& base, const U32& offset, const U32& count,
                               bool is_signed) {
    return Inst<U32>(is_signed ? Opcode::BitFieldSExtract : Opcode::BitFieldUExtract, base, offset,
                     count);
}
U32 IREmitter::BitReverse(const U32& value) {
    return Inst<U32>(Opcode::BitReverse32, value);
}
U32 IREmitter::BitCount(const U32U64& value) {
    return U32{By32Or64<U32, U32>(value.Type(), Opcode::BitCount32, Opcode::BitCount64, value)};
}
U32U64 IREmitter::BitwiseNot(const U32U64& value) {
    return U32U64{
        By32Or64<U32, U64>(value.Type(), Opcode::BitwiseNot32, Opcode::BitwiseNot64, value)};
}
U32 IREmitter::MaskedBitCount(const U32& value, const U32& addend, bool hi) {
    return Inst<U32>(Opcode::MaskedBitCount32, value, addend, Imm1(hi));
}
U32 IREmitter::FindSMsb(const U32& value) {
    return Inst<U32>(Opcode::FindSMsb32, value);
}
U32 IREmitter::FindUMsb(const U32U64& value) {
    return U32{By32Or64<U32, U32>(value.Type(), Opcode::FindUMsb32, Opcode::FindUMsb64, value)};
}
U32 IREmitter::FindILsb(const U32U64& value) {
    return U32{By32Or64<U32, U32>(value.Type(), Opcode::FindILsb32, Opcode::FindILsb64, value)};
}

U32 IREmitter::SMin(const U32& a, const U32& b) {
    return Inst<U32>(Opcode::SMin32, a, b);
}
U32 IREmitter::UMin(const U32& a, const U32& b) {
    return Inst<U32>(Opcode::UMin32, a, b);
}
U32 IREmitter::IMin(const U32& a, const U32& b, bool is_signed) {
    return is_signed ? SMin(a, b) : UMin(a, b);
}
U32 IREmitter::SMax(const U32& a, const U32& b) {
    return Inst<U32>(Opcode::SMax32, a, b);
}
U32 IREmitter::UMax(const U32& a, const U32& b) {
    return Inst<U32>(Opcode::UMax32, a, b);
}
U32 IREmitter::IMax(const U32& a, const U32& b, bool is_signed) {
    return is_signed ? SMax(a, b) : UMax(a, b);
}
U32 IREmitter::SMinTri(const U32& a, const U32& b, const U32& c) {
    return Inst<U32>(Opcode::SMinTri32, a, b, c);
}
U32 IREmitter::UMinTri(const U32& a, const U32& b, const U32& c) {
    return Inst<U32>(Opcode::UMinTri32, a, b, c);
}
U32 IREmitter::IMinTri(const U32& a, const U32& b, const U32& c, bool is_signed) {
    return is_signed ? SMinTri(a, b, c) : UMinTri(a, b, c);
}
U32 IREmitter::SMaxTri(const U32& a, const U32& b, const U32& c) {
    return Inst<U32>(Opcode::SMaxTri32, a, b, c);
}
U32 IREmitter::UMaxTri(const U32& a, const U32& b, const U32& c) {
    return Inst<U32>(Opcode::UMaxTri32, a, b, c);
}
U32 IREmitter::IMaxTri(const U32& a, const U32& b, const U32& c, bool is_signed) {
    return is_signed ? SMaxTri(a, b, c) : UMaxTri(a, b, c);
}
U32 IREmitter::SMedTri(const U32& a, const U32& b, const U32& c) {
    return Inst<U32>(Opcode::SMedTri32, a, b, c);
}
U32 IREmitter::UMedTri(const U32& a, const U32& b, const U32& c) {
    return Inst<U32>(Opcode::UMedTri32, a, b, c);
}
U32 IREmitter::IMedTri(const U32& a, const U32& b, const U32& c, bool is_signed) {
    return is_signed ? SMedTri(a, b, c) : UMedTri(a, b, c);
}
U32 IREmitter::SClamp(const U32& value, const U32& min, const U32& max) {
    return Inst<U32>(Opcode::SClamp32, value, min, max);
}
U32 IREmitter::UClamp(const U32& value, const U32& min, const U32& max) {
    return Inst<U32>(Opcode::UClamp32, value, min, max);
}

#define BB_INT_COMPARE(name, signed_op, unsigned_op)                                              \
    U1 IREmitter::name(const U32U64& lhs, const U32U64& rhs, bool is_signed) {                    \
        ExpectSameTypes(lhs, rhs);                                                                \
        return U1{By32Or64<U1, U1>(lhs.Type(),                                                    \
                                   is_signed ? Opcode::signed_op##32 : Opcode::unsigned_op##32,   \
                                   is_signed ? Opcode::signed_op##64 : Opcode::unsigned_op##64,   \
                                   lhs, rhs)};                                                    \
    }
BB_INT_COMPARE(ILessThan, SLessThan, ULessThan)
BB_INT_COMPARE(ILessThanEqual, SLessThanEqual, ULessThanEqual)
BB_INT_COMPARE(IGreaterThan, SGreaterThan, UGreaterThan)
BB_INT_COMPARE(IGreaterThanEqual, SGreaterThanEqual, UGreaterThanEqual)
#undef BB_INT_COMPARE

U1 IREmitter::IEqual(const U32U64& lhs, const U32U64& rhs) {
    ExpectSameTypes(lhs, rhs);
    return U1{By32Or64<U1, U1>(lhs.Type(), Opcode::IEqual32, Opcode::IEqual64, lhs, rhs)};
}
U1 IREmitter::INotEqual(const U32U64& lhs, const U32U64& rhs) {
    ExpectSameTypes(lhs, rhs);
    return U1{By32Or64<U1, U1>(lhs.Type(), Opcode::INotEqual32, Opcode::INotEqual64, lhs, rhs)};
}

U1 IREmitter::LogicalOr(const U1& a, const U1& b) {
    return Inst<U1>(Opcode::LogicalOr, a, b);
}
U1 IREmitter::LogicalAnd(const U1& a, const U1& b) {
    return Inst<U1>(Opcode::LogicalAnd, a, b);
}
U1 IREmitter::LogicalXor(const U1& a, const U1& b) {
    return Inst<U1>(Opcode::LogicalXor, a, b);
}
U1 IREmitter::LogicalNot(const U1& value) {
    return Inst<U1>(Opcode::LogicalNot, value);
}

// Conversions. Only the pairs below exist.

U32U64 IREmitter::ConvertFToS(size_t bitsize, const F32F64& value) {
    ASSERT_MSG(bitsize == 32, "Invalid destination bitsize {}", bitsize);
    return U32U64{By32Or64<U32, U32>(value.Type(), Opcode::ConvertS32F32, Opcode::ConvertS32F64,
                                     value)};
}

U32U64 IREmitter::ConvertFToU(size_t bitsize, const F32F64& value) {
    ASSERT_MSG(bitsize == 32, "Invalid destination bitsize {}", bitsize);
    if (value.Type() != Type::F32) {
        ThrowInvalidType(value.Type());
    }
    return Inst<U32>(Opcode::ConvertU32F32, value);
}

U32U64 IREmitter::ConvertFToI(size_t bitsize, bool is_signed, const F32F64& value) {
    return is_signed ? ConvertFToS(bitsize, value) : ConvertFToU(bitsize, value);
}

F32F64 IREmitter::ConvertSToF(size_t dest_bitsize, size_t src_bitsize, const Value& value) {
    if (src_bitsize == 32 && dest_bitsize == 32) {
        return Inst<F32>(Opcode::ConvertF32S32, value);
    }
    if (src_bitsize == 32 && dest_bitsize == 64) {
        return Inst<F64>(Opcode::ConvertF64S32, value);
    }
    UNREACHABLE_MSG("Invalid bit size combination dst={} src={}", dest_bitsize, src_bitsize);
}

F32F64 IREmitter::ConvertUToF(size_t dest_bitsize, size_t src_bitsize, const Value& value) {
    if (dest_bitsize == 32 && src_bitsize == 16) {
        return Inst<F32>(Opcode::ConvertF32U16, value);
    }
    if (dest_bitsize == 32 && src_bitsize == 32) {
        return Inst<F32>(Opcode::ConvertF32U32, value);
    }
    if ((dest_bitsize == 32 || dest_bitsize == 64) && src_bitsize == 32) {
        return Inst<F64>(Opcode::ConvertF64U32, value);
    }
    UNREACHABLE_MSG("Invalid bit size combination dst={} src={}", dest_bitsize, src_bitsize);
}

F32F64 IREmitter::ConvertIToF(size_t dest_bitsize, size_t src_bitsize, bool is_signed,
                              const Value& value) {
    return is_signed ? ConvertSToF(dest_bitsize, src_bitsize, value)
                     : ConvertUToF(dest_bitsize, src_bitsize, value);
}

U8U16U32U64 IREmitter::UConvert(size_t result_bitsize, const U8U16U32U64& value) {
    // A result size falls through to the larger ones when it has no conversion of this type.
    const Type type = value.Type();
    switch (result_bitsize) {
    case 8:
        if (type == Type::U32) {
            return Inst<U8>(Opcode::ConvertU8U32, value);
        }
        [[fallthrough]];
    case 16:
        if (type == Type::U32) {
            return Inst<U16>(Opcode::ConvertU16U32, value);
        }
        [[fallthrough]];
    case 32:
        if (type == Type::U8) {
            return Inst<U32>(Opcode::ConvertU32U8, value);
        }
        if (type == Type::U16) {
            return Inst<U32>(Opcode::ConvertU32U16, value);
        }
        [[fallthrough]];
    case 64:
        if (type == Type::U32) {
            return Inst<U64>(Opcode::ConvertU64U32, value);
        }
        break;
    default:
        break;
    }
    UNREACHABLE_MSG("Conversion from {} to {} bits", type, result_bitsize);
}

U8U16U32U64 IREmitter::SConvert(size_t result_bitsize, const U8U16U32U64& value) {
    if (result_bitsize == 32 && value.Type() == Type::U8) {
        return Inst<U32>(Opcode::ConvertS32S8, value);
    }
    if (result_bitsize == 32 && value.Type() == Type::U16) {
        return Inst<U32>(Opcode::ConvertS32S16, value);
    }
    UNREACHABLE_MSG("Signed Conversion from {} to {} bits", value.Type(), result_bitsize);
}

F16F32F64 IREmitter::FPConvert(size_t result_bitsize, const F16F32F64& value) {
    const Type type = value.Type();
    switch (result_bitsize) {
    case 16:
        if (type == Type::F32) {
            return Inst<F16>(Opcode::ConvertF16F32, value);
        }
        [[fallthrough]];
    case 32:
        if (type == Type::F16) {
            return Inst<F32>(Opcode::ConvertF32F16, value);
        }
        if (type == Type::F64) {
            return Inst<F32>(Opcode::ConvertF32F64, value);
        }
        [[fallthrough]];
    case 64:
        if (type == Type::F32) {
            return Inst<F64>(Opcode::ConvertF64F32, value);
        }
        break;
    default:
        break;
    }
    UNREACHABLE_MSG("Conversion from {} to {} bits", type, result_bitsize);
}

// Images.

#define BB_IMAGE_ATOMIC(name, op)                                                                 \
    Value IREmitter::ImageAtomic##name(const Value& handle, const Value& coords,                  \
                                       const Value& value, TextureInstInfo info) {                \
        return Inst(Opcode::ImageAtomic##op, Flags{info}, handle, coords, value);                \
    }
BB_IMAGE_ATOMIC(IAdd, IAdd32)
BB_IMAGE_ATOMIC(SMin, SMin32)
BB_IMAGE_ATOMIC(UMin, UMin32)
BB_IMAGE_ATOMIC(SMax, SMax32)
BB_IMAGE_ATOMIC(UMax, UMax32)
BB_IMAGE_ATOMIC(FMax, FMax32)
BB_IMAGE_ATOMIC(FMin, FMin32)
BB_IMAGE_ATOMIC(Inc, Inc32)
BB_IMAGE_ATOMIC(Dec, Dec32)
BB_IMAGE_ATOMIC(And, And32)
BB_IMAGE_ATOMIC(Or, Or32)
BB_IMAGE_ATOMIC(Xor, Xor32)
BB_IMAGE_ATOMIC(Exchange, Exchange32)
#undef BB_IMAGE_ATOMIC

Value IREmitter::ImageAtomicIMin(const Value& handle, const Value& coords, const Value& value,
                                 bool is_signed, TextureInstInfo info) {
    return is_signed ? ImageAtomicSMin(handle, coords, value, info)
                     : ImageAtomicUMin(handle, coords, value, info);
}
Value IREmitter::ImageAtomicIMax(const Value& handle, const Value& coords, const Value& value,
                                 bool is_signed, TextureInstInfo info) {
    return is_signed ? ImageAtomicSMax(handle, coords, value, info)
                     : ImageAtomicUMax(handle, coords, value, info);
}
Value IREmitter::ImageAtomicCmpSwap(const Value& handle, const Value& coords, const Value& value,
                                    const Value& cmp_value, TextureInstInfo info) {
    return Inst(Opcode::ImageAtomicCmpSwap32, Flags{info}, handle, coords, value, cmp_value);
}

Value IREmitter::ImageHandle(const Value& tsharp_low, const Value& tsharp_high) {
    return Inst(Opcode::ImageHandle, tsharp_low, tsharp_high);
}

Value IREmitter::ImageSampleRaw(const Value& image_handle, const Value& sampler_handle,
                                const Value& address1, const Value& address2, const Value& address3,
                                const Value& address4, TextureInstInfo info) {
    return Inst(Opcode::ImageSampleRaw, Flags{info}, image_handle, sampler_handle, address1,
                address2, address3, address4);
}
Value IREmitter::ImageSampleImplicitLod(const Value& handle, const Value& coords, const F32& bias,
                                        const Value& offset, TextureInstInfo info) {
    return Inst(Opcode::ImageSampleImplicitLod, Flags{info}, handle, coords, bias, offset);
}
Value IREmitter::ImageSampleExplicitLod(const Value& handle, const Value& coords, const F32& lod,
                                        const Value& offset, TextureInstInfo info) {
    return Inst(Opcode::ImageSampleExplicitLod, Flags{info}, handle, coords, lod, offset);
}
Value IREmitter::ImageSampleDrefImplicitLod(const Value& handle, const Value& coords,
                                            const F32& dref, const F32& bias, const Value& offset,
                                            TextureInstInfo info) {
    return Inst(Opcode::ImageSampleDrefImplicitLod, Flags{info}, handle, coords, dref, bias,
                offset);
}
Value IREmitter::ImageSampleDrefExplicitLod(const Value& handle, const Value& coords,
                                            const F32& dref, const F32& lod, const Value& offset,
                                            TextureInstInfo info) {
    return Inst(Opcode::ImageSampleDrefExplicitLod, Flags{info}, handle, coords, dref, lod, offset);
}
Value IREmitter::ImageGather(const Value& handle, const Value& coords, const Value& offset,
                             TextureInstInfo info) {
    return Inst(Opcode::ImageGather, Flags{info}, handle, coords, offset);
}
Value IREmitter::ImageGatherDref(const Value& handle, const Value& coords, const Value& offset,
                                 const F32& dref, TextureInstInfo info) {
    return Inst(Opcode::ImageGatherDref, Flags{info}, handle, coords, offset, dref);
}
Value IREmitter::ImageQueryDimension(const Value& handle, const U32& lod, const U1& skip_mips,
                                     TextureInstInfo info) {
    return Inst(Opcode::ImageQueryDimensions, Flags{info}, handle, lod, skip_mips);
}
Value IREmitter::ImageQueryLod(const Value& handle, const Value& coords, TextureInstInfo info) {
    return Inst(Opcode::ImageQueryLod, Flags{info}, handle, coords);
}
Value IREmitter::ImageGradient(const Value& handle, const Value& coords,
                               const Value& derivatives_dx, const Value& derivatives_dy,
                               const Value& offset, const F32& lod_clamp, TextureInstInfo info) {
    return Inst(Opcode::ImageGradient, Flags{info}, handle, coords, derivatives_dx, derivatives_dy,
                offset, lod_clamp);
}
Value IREmitter::ImageRead(const Value& handle, const Value& coords, const U32& lod,
                           const U32& multisampling, TextureInstInfo info) {
    return Inst(Opcode::ImageRead, Flags{info}, handle, coords, lod, multisampling);
}
void IREmitter::ImageWrite(const Value& handle, const Value& coords, const U32& lod,
                           const U32& multisampling, const Value& color, TextureInstInfo info) {
    Inst(Opcode::ImageWrite, Flags{info}, handle, coords, lod, multisampling, color);
}
F32 IREmitter::CubeFaceIndex(const Value& cube_coords) {
    return Inst<F32>(Opcode::CubeFaceIndex, cube_coords);
}

} // namespace Shader::IR
