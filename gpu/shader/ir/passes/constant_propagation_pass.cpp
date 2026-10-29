// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Constant folding and local algebraic simplification: operations on immediates are evaluated,
// commutative operations get their immediate on the right (and chains of them combined), and
// inverse pairs (pack/unpack, conversions, bit casts, composite construct/extract) cancel.
// Blocks are visited in reverse post order, so operands are folded before their users.
#include <array>
#include <bit>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

#include "common/assert.h"
#include "common/func_traits.h"
#include "shader/ir/basic_block.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"

namespace Shader::Optimization {

namespace {

using IR::Opcode;

template <typename T>
T ImmArg(const IR::Value& value) {
    if constexpr (std::is_same_v<T, bool>) {
        return value.U1();
    } else if constexpr (std::is_same_v<T, u32>) {
        return value.U32();
    } else if constexpr (std::is_same_v<T, s32>) {
        return static_cast<s32>(value.U32());
    } else if constexpr (std::is_same_v<T, f32>) {
        return value.F32();
    } else if constexpr (std::is_same_v<T, u64>) {
        return value.U64();
    } else if constexpr (std::is_same_v<T, s64>) {
        return static_cast<s64>(value.U64());
    }
}

/// An instruction whose operands are all immediates becomes `func` of them (its parameter types
/// say how each operand is read, its return type the result's).
template <typename Func>
void FoldImmediates(IR::Inst& inst, Func&& func) {
    if (!inst.AreAllArgsImmediates()) {
        return;
    }
    using Traits = Common::LambdaTraits<std::remove_cvref_t<Func>>;
    const auto eval = [&]<size_t... I>(std::index_sequence<I...>) {
        return IR::Value{func(ImmArg<typename Traits::template ArgType<I>>(inst.Arg(I))...)};
    };
    inst.ReplaceUsesWithAndRemove(eval(std::make_index_sequence<Traits::NUM_ARGS>{}));
}

/// A commutative operation: evaluated on two immediates (false: the instruction is gone),
/// otherwise with the immediate moved to the right and combined with an operand that is the
/// same operation with an immediate.
template <typename T, typename Func>
bool FoldCommutative(IR::Inst& inst, Func&& func) {
    const IR::Value lhs = inst.Arg(0);
    const IR::Value rhs = inst.Arg(1);
    if (lhs.IsImmediate() && rhs.IsImmediate()) {
        inst.ReplaceUsesWithAndRemove(IR::Value{func(ImmArg<T>(lhs), ImmArg<T>(rhs))});
        return false;
    }
    if (lhs.IsImmediate()) {
        const IR::Inst* const rhs_inst = rhs.Inst();
        if (rhs_inst->GetOpcode() == inst.GetOpcode() && rhs_inst->Arg(1).IsImmediate()) {
            const auto combined = func(ImmArg<T>(lhs), ImmArg<T>(rhs_inst->Arg(1)));
            inst.SetArg(0, rhs_inst->Arg(0));
            inst.SetArg(1, IR::Value{combined});
        } else {
            inst.SetArg(0, rhs);
            inst.SetArg(1, lhs);
        }
    } else if (rhs.IsImmediate()) {
        const IR::Inst* const lhs_inst = lhs.Inst();
        if (lhs_inst->GetOpcode() == inst.GetOpcode() && lhs_inst->Arg(1).IsImmediate()) {
            const auto combined = func(ImmArg<T>(rhs), ImmArg<T>(lhs_inst->Arg(1)));
            inst.SetArg(0, lhs_inst->Arg(0));
            inst.SetArg(1, IR::Value{combined});
        }
    }
    return true;
}

/// f(g(x)) = x when the operand is the inverse operation.
void FoldInverse(IR::Inst& inst, Opcode inverse) {
    const IR::Value value = inst.Arg(0);
    if (!value.IsImmediate() && value.Inst()->GetOpcode() == inverse) {
        inst.ReplaceUsesWithAndRemove(value.Inst()->Arg(0));
    }
}

template <typename Dest, typename Source>
void FoldBitCast(IR::Inst& inst, Opcode reverse) {
    const IR::Value value = inst.Arg(0);
    if (value.IsImmediate()) {
        inst.ReplaceUsesWithAndRemove(IR::Value{std::bit_cast<Dest>(ImmArg<Source>(value))});
    } else {
        FoldInverse(inst, reverse);
    }
}

/// Unpacking a 64-bit value into two words: an immediate splits.
void FoldUnpack2x32(IR::Block& block, IR::Inst& inst, Opcode pack) {
    const IR::Value value = inst.Arg(0);
    if (!value.IsImmediate()) {
        FoldInverse(inst, pack);
        return;
    }
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const auto lo = ir.Imm32(static_cast<u32>(value.U64()));
    const auto hi = ir.Imm32(static_cast<u32>(value.U64() >> 32));
    inst.ReplaceUsesWithAndRemove(ir.CompositeConstruct(lo, hi));
}

/// The component `index` of a composite built by construct/insert instructions.
std::optional<IR::Value> ComponentOf(IR::Value composite, Opcode insert, Opcode construct,
                                     u32 index) {
    while (true) {
        const IR::Inst* const inst = composite.Inst();
        if (inst->GetOpcode() == construct) {
            return inst->Arg(index);
        }
        if (inst->GetOpcode() != insert || !inst->Arg(2).IsImmediate()) {
            return std::nullopt;
        }
        if (inst->Arg(2).U32() == index) {
            return inst->Arg(1);
        }
        composite = inst->Arg(0);
        if (composite.IsImmediate()) {
            return std::nullopt;
        }
    }
}

void FoldCompositeExtract(IR::Inst& inst, Opcode construct, Opcode insert) {
    const IR::Value composite = inst.Arg(0);
    const IR::Value index = inst.Arg(1);
    if (composite.IsImmediate() || !index.IsImmediate()) {
        return;
    }
    if (const auto component = ComponentOf(composite, insert, construct, index.U32())) {
        inst.ReplaceUsesWithAndRemove(*component);
    }
}

/// A composite rebuilt from the components of one composite, in order, is that composite.
void FoldCompositeConstruct(IR::Inst& inst, Opcode extract) {
    IR::Value source{};
    for (size_t i = 0; i < inst.NumArgs(); ++i) {
        const IR::Value value = inst.Arg(i);
        if (value.IsImmediate()) {
            return;
        }
        const IR::Inst* const element = value.Inst();
        if (element->GetOpcode() != extract || element->Arg(1).U32() != i) {
            return;
        }
        if (source.IsEmpty()) {
            source = element->Arg(0);
        } else if (source != element->Arg(0)) {
            return;
        }
    }
    inst.ReplaceUsesWithAndRemove(source);
}

void FoldLogicalAnd(IR::Inst& inst) {
    if (!FoldCommutative<bool>(inst, [](bool a, bool b) { return a && b; })) {
        return;
    }
    const IR::Value rhs = inst.Arg(1);
    if (rhs.IsImmediate()) {
        inst.ReplaceUsesWithAndRemove(rhs.U1() ? inst.Arg(0) : IR::Value{false});
    }
}

void FoldLogicalOr(IR::Inst& inst) {
    if (!FoldCommutative<bool>(inst, [](bool a, bool b) { return a || b; })) {
        return;
    }
    const IR::Value rhs = inst.Arg(1);
    if (rhs.IsImmediate()) {
        inst.ReplaceUsesWithAndRemove(rhs.U1() ? IR::Value{true} : inst.Arg(0));
    }
}

void FoldLogicalNot(IR::Inst& inst) {
    const IR::Value value = inst.Arg(0);
    if (value.IsImmediate()) {
        inst.ReplaceUsesWithAndRemove(IR::Value{!value.U1()});
    } else {
        FoldInverse(inst, Opcode::LogicalNot);
    }
}

void FoldSelect(IR::Inst& inst) {
    const IR::Value cond = inst.Arg(0);
    if (cond.IsImmediate()) {
        inst.ReplaceUsesWithAndRemove(cond.U1() ? inst.Arg(1) : inst.Arg(2));
        return;
    }
    // select(c, true, false) = c
    if (inst.GetOpcode() == Opcode::SelectU1 && inst.Arg(1).IsImmediate() && inst.Arg(1).U1() &&
        inst.Arg(2).IsImmediate() && !inst.Arg(2).U1()) {
        inst.ReplaceUsesWithAndRemove(cond);
    }
}

void FoldInverseBallot(IR::Inst& inst) {
    const IR::Value mask = inst.Arg(0);
    if (!mask.IsImmediate()) {
        FoldInverse(inst, Opcode::Ballot);
    } else if (mask.U64() == 0ULL) {
        inst.ReplaceUsesWithAndRemove(IR::Value{false});
    } else if (mask.U64() == std::numeric_limits<u64>::max()) {
        inst.ReplaceUsesWithAndRemove(IR::Value{true});
    }
}

void FoldDiscardCond(IR::Inst& inst) {
    const IR::Value cond = inst.Arg(0);
    if (!cond.IsImmediate()) {
        return;
    }
    if (cond.U1()) {
        inst.ReplaceOpcode(Opcode::Discard);
    } else {
        inst.Invalidate();
    }
}

template <typename T>
void FoldAdd(IR::Inst& inst) {
    if (FoldCommutative<T>(inst, [](T a, T b) { return a + b; })) {
        const IR::Value rhs = inst.Arg(1);
        if (rhs.IsImmediate() && ImmArg<T>(rhs) == 0) {
            inst.ReplaceUsesWithAndRemove(inst.Arg(0));
        }
    }
}

template <typename T>
void FoldMul(IR::Inst& inst) {
    if (!FoldCommutative<T>(inst, [](T a, T b) { return a * b; })) {
        return;
    }
    const IR::Value rhs = inst.Arg(1);
    if (rhs.IsImmediate() && ImmArg<T>(rhs) == 0) {
        inst.ReplaceUsesWithAndRemove(IR::Value(0u));
    } else if (rhs.IsImmediate() && ImmArg<T>(rhs) == 1) {
        inst.ReplaceUsesWithAndRemove(inst.Arg(0));
    }
}

constexpr size_t NumOpcodes = [] {
    size_t count = 0;
#define OPCODE(...) ++count;
#include "shader/ir/opcodes.inc"
#undef OPCODE
    return count;
}();

/// For each opcode, the operation it undoes (Void for none): the format packs and unpacks, the
/// half conversions, and packing two words to 64 bits.
constexpr auto InverseTable = [] {
    std::array<Opcode, NumOpcodes> table{};
    table.fill(Opcode::Void);
    const auto pair = [&](Opcode a, Opcode b) {
        table[size_t(a)] = b;
        table[size_t(b)] = a;
    };
    pair(Opcode::PackUnorm2x16, Opcode::UnpackUnorm2x16);
    pair(Opcode::PackSnorm2x16, Opcode::UnpackSnorm2x16);
    pair(Opcode::PackUint2x16, Opcode::UnpackUint2x16);
    pair(Opcode::PackSint2x16, Opcode::UnpackSint2x16);
    pair(Opcode::PackHalf2x16, Opcode::UnpackHalf2x16);
    pair(Opcode::PackUnorm4x8, Opcode::UnpackUnorm4x8);
    pair(Opcode::PackSnorm4x8, Opcode::UnpackSnorm4x8);
    pair(Opcode::PackUint4x8, Opcode::UnpackUint4x8);
    pair(Opcode::PackSint4x8, Opcode::UnpackSint4x8);
    pair(Opcode::PackUfloat10_11_11, Opcode::UnpackUfloat10_11_11);
    pair(Opcode::PackUnorm2_10_10_10, Opcode::UnpackUnorm2_10_10_10);
    pair(Opcode::PackSnorm2_10_10_10, Opcode::UnpackSnorm2_10_10_10);
    pair(Opcode::PackUint2_10_10_10, Opcode::UnpackUint2_10_10_10);
    pair(Opcode::PackSint2_10_10_10, Opcode::UnpackSint2_10_10_10);
    pair(Opcode::ConvertF32F16, Opcode::ConvertF16F32);
    // The unpacks of these fold immediates too (FoldUnpack2x32).
    table[size_t(Opcode::PackUint2x32)] = Opcode::UnpackUint2x32;
    table[size_t(Opcode::PackDouble2x32)] = Opcode::UnpackDouble2x32;
    return table;
}();

u32 BitReverse(u32 value) {
    u32 result{};
    for (int i = 0; i < 32; i++, value >>= 1) {
        result = (result << 1) | (value & 1);
    }
    return result;
}

void FoldInst(IR::Block& block, IR::Inst& inst) {
    const Opcode op = inst.GetOpcode();
    if (const Opcode inverse = InverseTable[size_t(op)]; inverse != Opcode::Void) {
        FoldInverse(inst, inverse);
        return;
    }
    switch (op) {
    case Opcode::IAdd32:
        return FoldAdd<u32>(inst);
    case Opcode::IMul32:
        return FoldMul<u32>(inst);
    case Opcode::ISub32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a - b; });
    case Opcode::UDiv32:
        return FoldImmediates(inst, [](u32 a, u32 b) {
            ASSERT_MSG(b != 0, "Folding UDiv32 with divisor 0");
            return a / b;
        });
    case Opcode::UMod32:
        return FoldImmediates(inst, [](u32 a, u32 b) {
            ASSERT_MSG(b != 0, "Folding UMod32 with modulo 0");
            return a % b;
        });
    case Opcode::UMin32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a > b ? b : a; });
    case Opcode::UMax32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a > b ? a : b; });
    case Opcode::ShiftLeftLogical32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return static_cast<u32>(a << b); });
    case Opcode::ShiftRightLogical32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return static_cast<u32>(a >> b); });
    case Opcode::ShiftRightArithmetic32:
        return FoldImmediates(inst, [](s32 a, s32 b) { return static_cast<u32>(a >> b); });
    case Opcode::ConvertF32U32:
        return FoldImmediates(inst, [](u32 a) { return static_cast<f32>(a); });
    case Opcode::FPNeg32:
        return FoldImmediates(inst, [](f32 a) { return -a; });

    case Opcode::SLessThan32:
        return FoldImmediates(inst, [](s32 a, s32 b) { return a < b; });
    case Opcode::SLessThan64:
        return FoldImmediates(inst, [](s64 a, s64 b) { return a < b; });
    case Opcode::ULessThan32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a < b; });
    case Opcode::ULessThan64:
        return FoldImmediates(inst, [](u64 a, u64 b) { return a < b; });
    case Opcode::SLessThanEqual32:
        return FoldImmediates(inst, [](s32 a, s32 b) { return a <= b; });
    case Opcode::SLessThanEqual64:
        return FoldImmediates(inst, [](s64 a, s64 b) { return a <= b; });
    case Opcode::ULessThanEqual32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a <= b; });
    case Opcode::ULessThanEqual64:
        return FoldImmediates(inst, [](u64 a, u64 b) { return a <= b; });
    case Opcode::SGreaterThan32:
        return FoldImmediates(inst, [](s32 a, s32 b) { return a > b; });
    case Opcode::SGreaterThan64:
        return FoldImmediates(inst, [](s64 a, s64 b) { return a > b; });
    case Opcode::UGreaterThan32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a > b; });
    case Opcode::UGreaterThan64:
        return FoldImmediates(inst, [](u64 a, u64 b) { return a > b; });
    case Opcode::SGreaterThanEqual32:
        return FoldImmediates(inst, [](s32 a, s32 b) { return a >= b; });
    case Opcode::SGreaterThanEqual64:
        return FoldImmediates(inst, [](s64 a, s64 b) { return a >= b; });
    case Opcode::UGreaterThanEqual32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a >= b; });
    case Opcode::UGreaterThanEqual64:
        return FoldImmediates(inst, [](u64 a, u64 b) { return a >= b; });
    case Opcode::IEqual32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a == b; });
    case Opcode::IEqual64:
        return FoldImmediates(inst, [](u64 a, u64 b) { return a == b; });
    case Opcode::INotEqual32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a != b; });
    case Opcode::INotEqual64:
        return FoldImmediates(inst, [](u64 a, u64 b) { return a != b; });

    case Opcode::BitwiseAnd32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a & b; });
    case Opcode::BitwiseAnd64:
        return FoldImmediates(inst, [](u64 a, u64 b) { return a & b; });
    case Opcode::BitwiseOr32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a | b; });
    case Opcode::BitwiseOr64:
        return FoldImmediates(inst, [](u64 a, u64 b) { return a | b; });
    case Opcode::BitwiseXor32:
        return FoldImmediates(inst, [](u32 a, u32 b) { return a ^ b; });
    case Opcode::BitwiseNot32:
        return FoldImmediates(inst, [](u32 a) { return ~a; });
    case Opcode::BitReverse32:
        return FoldImmediates(inst, [](u32 a) { return BitReverse(a); });
    case Opcode::BitCount32:
        return FoldImmediates(inst, [](u32 a) { return static_cast<u32>(std::popcount(a)); });
    case Opcode::BitCount64:
        return FoldImmediates(inst, [](u64 a) { return static_cast<u32>(std::popcount(a)); });
    case Opcode::BitFieldUExtract:
        return FoldImmediates(inst, [](u32 base, u32 shift, u32 count) {
            if (size_t(shift) + size_t(count) > 32) {
                UNREACHABLE_MSG("Undefined result in {}({}, {}, {})", Opcode::BitFieldUExtract,
                                base, shift, count);
            }
            const u32 mask = count == 32 ? ~0u : (1u << count) - 1;
            return (base >> shift) & mask;
        });
    case Opcode::BitFieldSExtract:
        return FoldImmediates(inst, [](s32 base, u32 shift, u32 count) {
            const size_t end = size_t(shift) + size_t(count);
            const size_t left_shift = 32 - end;
            const size_t right_shift = size_t(32 - count);
            if (end > 32 || left_shift >= 32 || right_shift >= 32) {
                UNREACHABLE_MSG("Undefined result in {}({}, {}, {})", Opcode::BitFieldSExtract,
                                base, shift, count);
            }
            return static_cast<u32>((base << left_shift) >> right_shift);
        });
    case Opcode::BitFieldInsert32:
        return FoldImmediates(inst, [](u32 base, u32 insert, u32 offset, u32 bits) {
            if (bits >= 32 || offset >= 32) {
                UNREACHABLE_MSG("Undefined result in {}({}, {}, {}, {})", Opcode::BitFieldInsert32,
                                base, insert, offset, bits);
            }
            return (base & ~(~(~0u << bits) << offset)) | (insert << offset);
        });
    case Opcode::BitFieldInsert64:
        return FoldImmediates(inst, [](u64 base, u64 insert, u32 offset, u32 bits) {
            if (bits >= 64 || offset >= 64) {
                UNREACHABLE_MSG("Undefined result in {}({}, {}, {}, {})", Opcode::BitFieldInsert64,
                                base, insert, offset, bits);
            }
            return (base & ~(~(~u64{0} << bits) << offset)) | (insert << offset);
        });

    case Opcode::LogicalAnd:
        return FoldLogicalAnd(inst);
    case Opcode::LogicalOr:
        return FoldLogicalOr(inst);
    case Opcode::LogicalNot:
        return FoldLogicalNot(inst);
    case Opcode::SelectU1:
    case Opcode::SelectU32:
    case Opcode::SelectF32:
        return FoldSelect(inst);

    case Opcode::BitCastF32U32:
        return FoldBitCast<f32, u32>(inst, Opcode::BitCastU32F32);
    case Opcode::BitCastU32F32:
        return FoldBitCast<u32, f32>(inst, Opcode::BitCastF32U32);
    case Opcode::UnpackUint2x32:
        return FoldUnpack2x32(block, inst, Opcode::PackUint2x32);
    case Opcode::UnpackDouble2x32:
        return FoldUnpack2x32(block, inst, Opcode::PackDouble2x32);
    case Opcode::InverseBallot:
        return FoldInverseBallot(inst);

    case Opcode::CompositeExtractU32x2:
        return FoldCompositeExtract(inst, Opcode::CompositeConstructU32x2,
                                    Opcode::CompositeInsertU32x2);
    case Opcode::CompositeExtractU32x3:
        return FoldCompositeExtract(inst, Opcode::CompositeConstructU32x3,
                                    Opcode::CompositeInsertU32x3);
    case Opcode::CompositeExtractU32x4:
        return FoldCompositeExtract(inst, Opcode::CompositeConstructU32x4,
                                    Opcode::CompositeInsertU32x4);
    case Opcode::CompositeExtractF32x2:
        return FoldCompositeExtract(inst, Opcode::CompositeConstructF32x2,
                                    Opcode::CompositeInsertF32x2);
    case Opcode::CompositeExtractF32x3:
        return FoldCompositeExtract(inst, Opcode::CompositeConstructF32x3,
                                    Opcode::CompositeInsertF32x3);
    case Opcode::CompositeExtractF32x4:
        return FoldCompositeExtract(inst, Opcode::CompositeConstructF32x4,
                                    Opcode::CompositeInsertF32x4);
    case Opcode::CompositeConstructU32x2:
        return FoldCompositeConstruct(inst, Opcode::CompositeExtractU32x2);
    case Opcode::CompositeConstructU32x3:
        return FoldCompositeConstruct(inst, Opcode::CompositeExtractU32x3);
    case Opcode::CompositeConstructU32x4:
        return FoldCompositeConstruct(inst, Opcode::CompositeExtractU32x4);

    case Opcode::DiscardCond:
        return FoldDiscardCond(inst);
    default:
        break;
    }
}

} // namespace

void ConstantPropagationPass(IR::BlockList& program) {
    for (auto it = program.rbegin(); it != program.rend(); ++it) {
        IR::Block* const block = *it;
        for (IR::Inst& inst : block->Instructions()) {
            FoldInst(*block, inst);
        }
    }
}

} // namespace Shader::Optimization
