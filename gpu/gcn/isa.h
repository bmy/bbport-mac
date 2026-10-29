// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport gcn isa: GCN3 shader instructions as the translator sees them. The encodings, the
// unified opcode (one enum over all encodings; VOP3 forms of VOPC/VOP2/VOP1 instructions share
// their opcode), each instruction's format (class, operand counts and types), and a decoded
// instruction (GcnInst: operands with their modifiers, the encoding's control fields).
#pragma once

#include <array>
#include <optional>
#include <string_view>

#include "common/assert.h"
#include "common/enum.h"
#include "common/types.h"

namespace GcnIsa {

/// An encoding by the fixed bits of its first dword (InstEncoding::VOP2 has bit 31 clear).
enum class InstEncoding : u32 {
    SOP1 = 0x17Du << 23,
    SOPP = 0x17Fu << 23,
    SOPC = 0x17Eu << 23,
    VOP1 = 0x3Fu << 25,
    VOPC = 0x3Eu << 25,
    VOP3 = 0x34u << 26,
    EXP = 0x3Eu << 26,
    VOP3P = 0x33u << 26,
    VINTRP = 0x32u << 26,
    DS = 0x36u << 26,
    MUBUF = 0x38u << 26,
    MTBUF = 0x3Au << 26,
    MIMG = 0x3Cu << 26,
    SMRD = 0x18u << 27,
    SOPK = 0xBu << 28,
    SOP2 = 0x2u << 30,
    VOP2 = 0,
    ILLEGAL,
};

/// The encoding of the instruction starting with `token` (ILLEGAL when none matches).
constexpr InstEncoding EncodingOf(u32 token) noexcept {
    // Longest fixed bit patterns first: 9, 7, 6, 5, 4, 2 and 1 bits.
    switch (InstEncoding(token & (0x1FFu << 23))) {
    case InstEncoding::SOP1:
    case InstEncoding::SOPP:
    case InstEncoding::SOPC:
        return InstEncoding(token & (0x1FFu << 23));
    default:
        break;
    }
    switch (InstEncoding(token & (0x7Fu << 25))) {
    case InstEncoding::VOP1:
    case InstEncoding::VOPC:
        return InstEncoding(token & (0x7Fu << 25));
    default:
        break;
    }
    switch (InstEncoding(token & (0x3Fu << 26))) {
    case InstEncoding::VOP3:
    case InstEncoding::VOP3P:
    case InstEncoding::EXP:
    case InstEncoding::VINTRP:
    case InstEncoding::DS:
    case InstEncoding::MUBUF:
    case InstEncoding::MTBUF:
    case InstEncoding::MIMG:
        return InstEncoding(token & (0x3Fu << 26));
    default:
        break;
    }
    if ((token & (0x1Fu << 27)) == u32(InstEncoding::SMRD)) {
        return InstEncoding::SMRD;
    }
    if ((token & (0xFu << 28)) == u32(InstEncoding::SOPK)) {
        return InstEncoding::SOPK;
    }
    if ((token & (0x3u << 30)) == u32(InstEncoding::SOP2)) {
        return InstEncoding::SOP2;
    }
    if ((token & (1u << 31)) == 0) {
        return InstEncoding::VOP2;
    }
    return InstEncoding::ILLEGAL;
}

/// Bytes of an encoding without literals or SDWA/DPP dwords.
constexpr u32 EncodingLength(InstEncoding encoding) noexcept {
    switch (encoding) {
    case InstEncoding::VOP3:
    case InstEncoding::VOP3P:
    case InstEncoding::MUBUF:
    case InstEncoding::MTBUF:
    case InstEncoding::MIMG:
    case InstEncoding::DS:
    case InstEncoding::EXP:
        return 8;
    case InstEncoding::ILLEGAL:
        return 0;
    default:
        return 4;
    }
}

enum class InstClass : u32 {
    Undefined,
    ScalarArith,
    ScalarAbs,
    ScalarMov,
    ScalarMovRel,
    ScalarCmp,
    ScalarSelect,
    ScalarBitLogic,
    ScalarBitManip,
    ScalarBitField,
    ScalarConv,
    ScalarExecMask,
    ScalarQuadMask,
    VectorRegMov,
    VectorMovRel,
    VectorLane,
    VectorBitLogic,
    VectorBitField32,
    VectorThreadMask,
    VectorBitField64,
    VectorFpArith16,
    VectorFpArith32,
    VectorFpRound32,
    VectorFpField32,
    VectorFpTran16,
    VectorFpTran32,
    VectorFpCmp16,
    VectorFpCmp32,
    VectorFpArith64,
    VectorFpRound64,
    VectorFpField64,
    VectorFpTran64,
    VectorFpCmp64,
    VectorIntArith16,
    VectorIntArith32,
    VectorIntArith64,
    VectorIntCmp16,
    VectorIntCmp32,
    VectorIntCmp64,
    VectorConv,
    VectorFpGraph32,
    VectorIntGraph,
    VectorMisc,
    ScalarProgFlow,
    ScalarSync,
    ScalarWait,
    ScalarCache,
    ScalarPrior,
    ScalarRegAccess,
    ScalarMsg,
    ScalarMemRd,
    ScalarMemUt,
    VectorMemBufNoFmt,
    VectorMemBufFmt,
    VectorMemBufAtomic,
    VectorMemImgNoSmp,
    VectorMemImgSmp,
    VectorMemImgUt,
    VectorMemL1Cache,
    DsIdxRd,
    DsIdxWr,
    DsIdxWrXchg,
    DsIdxCondXchg,
    DsIdxWrap,
    DsAtomicArith32,
    DsAtomicArith64,
    DsAtomicMinMax32,
    DsAtomicMinMax64,
    DsAtomicCmpSt32,
    DsAtomicCmpSt64,
    DsAtomicLogic32,
    DsAtomicLogic64,
    DsAppendCon,
    DsDataShareUt,
    DsDataShareMisc,
    GdsSync,
    GdsOrdCnt,
    VectorInterpFpCache,
    Exp,
    DbgProf,
};

enum class InstCategory : u32 {
    Undefined,
    ScalarALU,
    ScalarMemory,
    VectorALU,
    VectorMemory,
    FlowControl,
    DataShare,
    VectorInterpolation,
    Export,
    DebugProfile,
};

enum class ScalarType : u32 {
    Undefined,
    Any,
    Uint16,
    Uint32,
    Uint64,
    Sint16,
    Sint32,
    Sint64,
    Float16,
    Float32,
    Float64,
    Bool,
};

/// What an operand's 9-bit (or 8-bit) code names.
enum class OperandField : u32 {
    ScalarGPR = 0,
    VccLo = 106,
    VccHi,
    M0 = 124,
    ExecLo = 126,
    ExecHi,
    ConstZero,
    SignedConstIntPos = 129,
    SignedConstIntNeg = 193,
    ConstFloatPos_0_5 = 240,
    ConstFloatNeg_0_5,
    ConstFloatPos_1_0,
    ConstFloatNeg_1_0,
    ConstFloatPos_2_0,
    ConstFloatNeg_2_0,
    ConstFloatPos_4_0,
    ConstFloatNeg_4_0,
    Inv2Pi,
    Sdwa,
    Dpp,
    VccZ = 251,
    ExecZ = 252,
    Scc = 253,
    LdsDirect = 254,
    LiteralConst = 255,
    VectorGPR,

    Undefined = 0xFFFFFFFF,
};

/// Ranges of operand codes.
enum OperandFieldRange : u32 {
    ScalarGPRMin = 0,
    ScalarGPRMax = 103,
    SignedConstIntPosMin = 129,
    SignedConstIntPosMax = 192,
    SignedConstIntNegMin = 193,
    SignedConstIntNegMax = 208,
    ConstFloatMin = 240,
    VectorGPRMin = 256,
    VectorGPRMax = 511,
};

/// The field of an operand code (ranges folded to their first value).
constexpr OperandField FieldOf(u32 code) noexcept {
    if (code <= ScalarGPRMax) {
        return OperandField::ScalarGPR;
    }
    if (code >= SignedConstIntPosMin && code <= SignedConstIntPosMax) {
        return OperandField::SignedConstIntPos;
    }
    if (code >= SignedConstIntNegMin && code <= SignedConstIntNegMax) {
        return OperandField::SignedConstIntNeg;
    }
    if (code >= VectorGPRMin && code <= VectorGPRMax) {
        return OperandField::VectorGPR;
    }
    return OperandField(code);
}

/// Number of opcodes of each encoding (their places in the unified opcode).
namespace OpcodeRange {
constexpr u32 SOP2 = 45, SOPK = 22, SOP1 = 54, SOPC = 17, SOPP = 27, VOPC = 248, VOP2 = 60,
              VOP1 = 86, VOP3 = 1024, VINTRP = 3, SMRD = 32, DS = 256, MUBUF = 114, MTBUF = 8,
              MIMG = 112, EXP = 1, VOP3P = 35;
}

/// Where each encoding's opcodes start in the unified opcode.
namespace OpcodeBase {
constexpr u32 SOP2 = 0;
constexpr u32 SOPK = SOP2 + OpcodeRange::SOP2;
constexpr u32 SOP1 = SOPK + OpcodeRange::SOPK;
constexpr u32 SOPC = SOP1 + OpcodeRange::SOP1;
constexpr u32 SOPP = SOPC + OpcodeRange::SOPC;
constexpr u32 VOPC = SOPP + OpcodeRange::SOPP;
constexpr u32 VOP2 = VOPC + OpcodeRange::VOPC;
constexpr u32 VOP1 = VOP2 + OpcodeRange::VOP2;
constexpr u32 VOP3 = VOP1 + OpcodeRange::VOP1;
constexpr u32 VINTRP = VOP3 + OpcodeRange::VOP3;
constexpr u32 SMRD = VINTRP + OpcodeRange::VINTRP;
constexpr u32 DS = SMRD + OpcodeRange::SMRD;
constexpr u32 MUBUF = DS + OpcodeRange::DS;
constexpr u32 MTBUF = MUBUF + OpcodeRange::MUBUF;
constexpr u32 MIMG = MTBUF + OpcodeRange::MTBUF;
constexpr u32 EXP = MIMG + OpcodeRange::MIMG;
constexpr u32 VOP3P = EXP + OpcodeRange::EXP;
constexpr u32 End = VOP3P + OpcodeRange::VOP3P;
} // namespace OpcodeBase

/// VOP3 opcode numbers of VOPC, VOP2 and VOP1 instructions: their number plus these.
namespace Vop3Form {
constexpr u32 VopcFirst = 0, VopcLast = 247;
constexpr u32 Vop2First = 256, Vop2Last = 315;
constexpr u32 Vop1First = 384, Vop1Last = 469;
} // namespace Vop3Form

enum class Opcode : u32 {
#define GCN_OP(enc, number, name, ...) name = OpcodeBase::enc + number,
#define GCN_FORM(...)
#include "gcn/isa_table.inc"
#undef GCN_OP
#undef GCN_FORM
};

/// The opcode's name ("?" for none).
std::string_view NameOf(Opcode opcode) noexcept;

/// An instruction's format in an encoding.
struct InstFormat {
    InstClass inst_class = InstClass::Undefined;
    InstCategory inst_category = InstCategory::Undefined;
    u32 src_count = 0;
    u32 dst_count = 0;
    ScalarType src_type = ScalarType::Undefined;
    ScalarType dst_type = ScalarType::Undefined;
};

/// The format of opcode number `number` in `encoding` (all Undefined for unknown ones).
const InstFormat& FormatOf(InstEncoding encoding, u32 number) noexcept;

/// The opcode number of `opcode` in `encoding` (VOP3 numbers of VOPC/VOP2/VOP1 opcodes).
u32 NumberIn(InstEncoding encoding, Opcode opcode) noexcept;

// Operands.

/// Applied after an operand register is read.
struct InputModifiers {
    bool neg = false;
    bool neg_hi = false;
    bool abs = false;
    bool sext = false;
};

/// Applied before the result is written.
struct OutputModifiers {
    bool clamp = false;
    float multiplier = 0.f; ///< 0: none
};

/// Halves of packed 16-bit operands.
struct OperandSelection {
    bool op_sel = false;
    bool op_sel_hi = false;
};

enum class SdwaSelector : u32 {
    Byte0 = 0,
    Byte1 = 1,
    Byte2 = 2,
    Byte3 = 3,
    Word0 = 4,
    Word1 = 5,
    Dword = 6,
    Invalid = 7,
};

enum class SdwaDstUnused : u32 {
    Pad = 0,
    Sext = 1,
    Preserve = 2,
    Invalid = 3,
};

enum class DppCtrl : u16 {
    DppQuadPerm = 0,
    DppRowShl = 0x101,
    DppRowShr = 0x111,
    DppRowRor = 0x121,
    DppRowMirror = 0x140,
    DppRowHalfMirror = 0x141,
};

struct DppOperation {
    DppCtrl op;
    u16 value;
    bool bc;
    u8 row_mask;
    u8 bank_mask;
};

struct InstOperand {
    OperandField field = OperandField::Undefined;
    ScalarType type = ScalarType::Undefined;
    InputModifiers input_modifier = {};
    OutputModifiers output_modifier = {};
    OperandSelection op_sel = {}; ///< packed 16-bit operations only
    SdwaDstUnused sdwa_dst = SdwaDstUnused::Invalid;
    SdwaSelector sdwa_sel = SdwaSelector::Invalid;
    std::optional<DppOperation> dpp = {};
    u32 code = 0xFFFFFFFF; ///< register number, constant code or literal
};

// Control fields of the encodings, at their bit positions in the instruction.

struct InstControlSOPK {
    s16 simm;
};
struct InstControlSOPP {
    s16 simm;
};
struct InstControlVOP3 {
    u64 : 8;
    u64 abs : 3;
    u64 clmp : 1;
    u64 op_sel : 4;
    u64 : 43;
    u64 omod : 2;
    u64 neg : 3;
};
struct InstControlVOP3P {
    u64 : 8;
    u64 neg_hi : 3;
    u64 op_sel : 3;
    u64 op_sel_hi_2 : 1;
    u64 clamp : 1;
    u64 : 43;
    u64 op_sel_hi_01 : 2;
    u64 neg : 3;

    bool get_op_sel_hi(int index) const {
        ASSERT_MSG(index >= 0 && index < 3, "op_sel_hi of source {}", index);
        return index == 2 ? op_sel_hi_2 : (op_sel_hi_01 >> index) & 1;
    }
};
static_assert(sizeof(InstControlVOP3P) == 8);
struct InstControlSMRD {
    u32 offset : 8;
    u32 imm : 1;
    u32 count : 5; ///< dwords loaded (set by the decoder)
    u32 : 18;
};
struct InstControlMUBUF {
    u64 offset : 12;
    u64 offen : 1;
    u64 idxen : 1;
    u64 glc : 1;
    u64 addr64 : 1;
    u64 lds : 1;
    u64 : 37;
    u64 slc : 1;
    u64 tfe : 1;
    u64 count : 3; ///< dwords (set by the decoder)
    u64 size : 5;  ///< bytes (set by the decoder)
};
struct InstControlMTBUF {
    u64 offset : 12;
    u64 offen : 1;
    u64 idxen : 1;
    u64 glc : 1;
    u64 addr64 : 1;
    u64 : 3;
    u64 dfmt : 4;
    u64 nfmt : 3;
    u64 : 28;
    u64 slc : 1;
    u64 tfe : 1;
    u64 count : 3; ///< dwords (set by the decoder)
    u64 size : 5;
};
struct InstControlMIMG {
    u64 : 8;
    u64 dmask : 4;
    u64 unrm : 1;
    u64 glc : 1;
    u64 da : 1;
    u64 r128 : 1;
    u64 tfe : 1;
    u64 lwe : 1;
    u64 : 7;
    u64 slc : 1;
    u64 mod : 32; ///< MimgModifierFlags (set by the decoder)
    u64 : 6;
};
struct InstControlDS {
    u64 offset0 : 8;
    u64 offset1 : 8;
    u64 : 1;
    u64 gds : 1;
    u64 dual : 1;     ///< two addresses (set by the decoder, as the fields below)
    u64 sign : 1;     ///< signed data
    u64 relative : 1; ///< SRC2 forms
    u64 stride : 1;   ///< offsets in units of 64 elements
    u64 size : 4;     ///< bytes per element
    u64 : 38;
};
struct InstControlVINTRP {
    u32 : 8;
    u32 chan : 2;
    u32 attr : 6;
    u32 : 16;
};
struct InstControlEXP {
    u64 en : 4;
    u64 target : 6;
    u64 compr : 1;
    u64 done : 1;
    u64 vm : 1;
    u64 reserved : 51;
};

union InstControl {
    InstControlSOPK sopk;
    InstControlSOPP sopp;
    InstControlVOP3 vop3;
    InstControlVOP3P vop3p;
    InstControlSMRD smrd;
    InstControlMUBUF mubuf;
    InstControlMTBUF mtbuf;
    InstControlMIMG mimg;
    InstControlDS ds;
    InstControlVINTRP vintrp;
    InstControlEXP exp;
};

/// The SDWA dword of VOPC and of VOP1/VOP2.
struct SdwaVopc {
    u32 src0 : 8;
    u32 sdst : 7;
    u32 sd : 1;
    u32 src0_sel : 3;
    u32 src0_sext : 1;
    u32 src0_neg : 1;
    u32 src0_abs : 1;
    u32 : 1;
    u32 s0 : 1;
    u32 src1_sel : 3;
    u32 src1_sext : 1;
    u32 src1_neg : 1;
    u32 src1_abs : 1;
    u32 : 1;
    u32 s1 : 1;
};
struct SdwaVop12 {
    u32 src0 : 8;
    u32 dst_sel : 3;
    u32 dst_u : 2;
    u32 clamp : 1;
    u32 omod : 2;
    u32 src0_sel : 3;
    u32 src0_sext : 1;
    u32 src0_neg : 1;
    u32 src0_abs : 1;
    u32 : 1;
    u32 s0 : 1;
    u32 src1_sel : 3;
    u32 src1_sext : 1;
    u32 src1_neg : 1;
    u32 src1_abs : 1;
    u32 : 1;
    u32 s1 : 1;
};
union Sdwa {
    SdwaVopc vopc;
    SdwaVop12 vop12;
};

/// The DPP dword of VOP1/VOP2.
struct Dpp {
    u32 src0 : 8;
    u32 dpp_ctrl : 9;
    u32 : 1;
    u32 fi : 1;
    u32 bc : 1;
    u32 src0_neg : 1;
    u32 src0_abs : 1;
    u32 src1_neg : 1;
    u32 src1_abs : 1;
    u32 bank_mask : 4;
    u32 row_mask : 4;

    DppOperation GetOperation() const;
};

constexpr u32 GcnMaxSrcCount = 4;
constexpr u32 GcnMaxDstCount = 2;

/// A decoded instruction.
struct GcnInst {
    Opcode opcode;
    InstEncoding encoding;
    InstClass inst_class;
    InstCategory category;
    InstControl control;
    u32 length; ///< bytes, literals included
    u32 src_count;
    u32 dst_count;
    std::array<InstOperand, GcnMaxSrcCount> src;
    std::array<InstOperand, GcnMaxDstCount> dst;

    /// The target of a branch at `pc`.
    u32 BranchTarget(u32 pc) const {
        return pc + s32(control.sopp.simm) * 4 + 4;
    }
    bool IsTerminateInstruction() const {
        return IsUnconditionalBranch() || IsConditionalBranch() || IsFork() ||
               opcode == Opcode::S_ENDPGM;
    }
    bool IsUnconditionalBranch() const {
        return opcode == Opcode::S_BRANCH || opcode == Opcode::S_SETPC_B64;
    }
    bool IsConditionalBranch() const;
    bool IsFork() const {
        return opcode == Opcode::S_CBRANCH_I_FORK || opcode == Opcode::S_CBRANCH_G_FORK ||
               opcode == Opcode::S_CBRANCH_JOIN;
    }
    bool IsCmpx() const;
};

/// Addressing variants of image sampling instructions.
enum class MimgModifier : u32 {
    Lod,              ///< LOD given instead of computed
    LodBias,          ///< added to the computed LOD
    LodClamp,         ///< LOD clamped to at most this
    Derivative,       ///< derivatives given for the LOD computation
    CoarseDerivative, ///< coarse derivatives given
    Level0,           ///< level 0
    Pcf,              ///< depth comparison
    Offset,           ///< integer texel offsets (packed in one dword)
};
using MimgModifierFlags = Common::Flags<MimgModifier>;

/// The modifiers of an image sampling opcode.
MimgModifierFlags MimgModifiersOf(Opcode opcode) noexcept;

enum class ImageResComponent : u32 { Width, Height, Depth, MipCount };
using ImageResFlags = Common::Flags<ImageResComponent>;

/// S_SENDMSG's immediate.
struct SendMsgSimm {
    enum class Message : u32 {
        Interrupt = 1,
        Gs = 2,
        GsDone = 3,
        System = 15,
    };
    enum class GsOp : u32 {
        Nop = 0,
        Cut = 1,
        Emit = 2,
        EmitCut = 3,
    };
    Message msg : 4;
    GsOp op : 2;
};

} // namespace GcnIsa
