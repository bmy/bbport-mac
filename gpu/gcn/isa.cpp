// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "gcn/isa.h"

#include <span>
#include <type_traits>
#include <utility>

namespace GcnIsa {

namespace {

/// Formats by encoding and number, and names by unified opcode, from the instruction table.
struct Tables {
    std::array<InstFormat, OpcodeRange::SOP2> sop2{};
    std::array<InstFormat, OpcodeRange::SOPK> sopk{};
    std::array<InstFormat, OpcodeRange::SOP1> sop1{};
    std::array<InstFormat, OpcodeRange::SOPC> sopc{};
    std::array<InstFormat, OpcodeRange::SOPP> sopp{};
    std::array<InstFormat, OpcodeRange::VOPC> vopc{};
    std::array<InstFormat, OpcodeRange::VOP2> vop2{};
    std::array<InstFormat, OpcodeRange::VOP1> vop1{};
    std::array<InstFormat, OpcodeRange::VOP3> vop3{};
    std::array<InstFormat, OpcodeRange::VINTRP> vintrp{};
    std::array<InstFormat, OpcodeRange::SMRD> smrd{};
    std::array<InstFormat, OpcodeRange::DS> ds{};
    std::array<InstFormat, OpcodeRange::MUBUF> mubuf{};
    std::array<InstFormat, OpcodeRange::MTBUF> mtbuf{};
    std::array<InstFormat, OpcodeRange::MIMG> mimg{};
    std::array<InstFormat, OpcodeRange::EXP> exp{};
    std::array<InstFormat, OpcodeRange::VOP3P> vop3p{};
    std::array<std::string_view, OpcodeBase::End> names{};

    constexpr std::span<InstFormat> Of(InstEncoding encoding) noexcept {
        switch (encoding) {
        case InstEncoding::SOP2:
            return sop2;
        case InstEncoding::SOPK:
            return sopk;
        case InstEncoding::SOP1:
            return sop1;
        case InstEncoding::SOPC:
            return sopc;
        case InstEncoding::SOPP:
            return sopp;
        case InstEncoding::VOPC:
            return vopc;
        case InstEncoding::VOP2:
            return vop2;
        case InstEncoding::VOP1:
            return vop1;
        case InstEncoding::VOP3:
            return vop3;
        case InstEncoding::VINTRP:
            return vintrp;
        case InstEncoding::SMRD:
            return smrd;
        case InstEncoding::DS:
            return ds;
        case InstEncoding::MUBUF:
            return mubuf;
        case InstEncoding::MTBUF:
            return mtbuf;
        case InstEncoding::MIMG:
            return mimg;
        case InstEncoding::EXP:
            return exp;
        case InstEncoding::VOP3P:
            return vop3p;
        default:
            return {};
        }
    }

    constexpr std::span<const InstFormat> Of(InstEncoding encoding) const noexcept {
        switch (encoding) {
        case InstEncoding::SOP2:
            return sop2;
        case InstEncoding::SOPK:
            return sopk;
        case InstEncoding::SOP1:
            return sop1;
        case InstEncoding::SOPC:
            return sopc;
        case InstEncoding::SOPP:
            return sopp;
        case InstEncoding::VOPC:
            return vopc;
        case InstEncoding::VOP2:
            return vop2;
        case InstEncoding::VOP1:
            return vop1;
        case InstEncoding::VOP3:
            return vop3;
        case InstEncoding::VINTRP:
            return vintrp;
        case InstEncoding::SMRD:
            return smrd;
        case InstEncoding::DS:
            return ds;
        case InstEncoding::MUBUF:
            return mubuf;
        case InstEncoding::MTBUF:
            return mtbuf;
        case InstEncoding::MIMG:
            return mimg;
        case InstEncoding::EXP:
            return exp;
        case InstEncoding::VOP3P:
            return vop3p;
        default:
            return {};
        }
    }

    constexpr Tables() {
#define FORMAT(cls, cat, srcs, dsts, src_type, dst_type)                                          \
    InstFormat {                                                                                  \
        InstClass::cls, InstCategory::cat, srcs, dsts, ScalarType::src_type, ScalarType::dst_type \
    }
#define GCN_OP(enc, number, name, ...)                                                            \
    Of(InstEncoding::enc)[number] = FORMAT(__VA_ARGS__);                                          \
    names[OpcodeBase::enc + number] = #name;
#define GCN_FORM(enc, number, ...) Of(InstEncoding::enc)[number] = FORMAT(__VA_ARGS__);
#include "gcn/isa_table.inc"
#undef GCN_OP
#undef GCN_FORM
#undef FORMAT
    }
};

constinit const Tables tables{};

constexpr u32 BaseOf(InstEncoding encoding) noexcept {
    switch (encoding) {
    case InstEncoding::SOP2:
        return OpcodeBase::SOP2;
    case InstEncoding::SOPK:
        return OpcodeBase::SOPK;
    case InstEncoding::SOP1:
        return OpcodeBase::SOP1;
    case InstEncoding::SOPC:
        return OpcodeBase::SOPC;
    case InstEncoding::SOPP:
        return OpcodeBase::SOPP;
    case InstEncoding::VOPC:
        return OpcodeBase::VOPC;
    case InstEncoding::VOP2:
        return OpcodeBase::VOP2;
    case InstEncoding::VOP1:
        return OpcodeBase::VOP1;
    case InstEncoding::VOP3:
        return OpcodeBase::VOP3;
    case InstEncoding::VINTRP:
        return OpcodeBase::VINTRP;
    case InstEncoding::SMRD:
        return OpcodeBase::SMRD;
    case InstEncoding::DS:
        return OpcodeBase::DS;
    case InstEncoding::MUBUF:
        return OpcodeBase::MUBUF;
    case InstEncoding::MTBUF:
        return OpcodeBase::MTBUF;
    case InstEncoding::MIMG:
        return OpcodeBase::MIMG;
    case InstEncoding::EXP:
        return OpcodeBase::EXP;
    case InstEncoding::VOP3P:
        return OpcodeBase::VOP3P;
    default:
        return 0;
    }
}

constexpr InstFormat UndefinedFormat{};

} // namespace

std::string_view NameOf(Opcode opcode) noexcept {
    const u32 index = u32(opcode);
    return index < tables.names.size() && !tables.names[index].empty() ? tables.names[index]
                                                                         : "?";
}

const InstFormat& FormatOf(InstEncoding encoding, u32 number) noexcept {
    const auto formats = tables.Of(encoding);
    return number < formats.size() ? formats[number] : UndefinedFormat;
}

u32 NumberIn(InstEncoding encoding, Opcode opcode) noexcept {
    const u32 op = u32(opcode);
    if (encoding == InstEncoding::VOP3) {
        if (op >= OpcodeBase::VOPC && op < OpcodeBase::VOPC + OpcodeRange::VOPC) {
            return op - OpcodeBase::VOPC + Vop3Form::VopcFirst;
        }
        if (op >= OpcodeBase::VOP2 && op < OpcodeBase::VOP2 + OpcodeRange::VOP2) {
            return op - OpcodeBase::VOP2 + Vop3Form::Vop2First;
        }
        if (op >= OpcodeBase::VOP1 && op < OpcodeBase::VOP1 + OpcodeRange::VOP1) {
            return op - OpcodeBase::VOP1 + Vop3Form::Vop1First;
        }
    }
    return op - BaseOf(encoding);
}

bool GcnInst::IsConditionalBranch() const {
    switch (opcode) {
    case Opcode::S_CBRANCH_SCC0:
    case Opcode::S_CBRANCH_SCC1:
    case Opcode::S_CBRANCH_VCCZ:
    case Opcode::S_CBRANCH_VCCNZ:
    case Opcode::S_CBRANCH_EXECZ:
    case Opcode::S_CBRANCH_EXECNZ:
        return true;
    case Opcode::S_CBRANCH_CDBGSYS:
    case Opcode::S_CBRANCH_CDBGUSER:
    case Opcode::S_CBRANCH_CDBGSYS_OR_USER:
    case Opcode::S_CBRANCH_CDBGSYS_AND_USER:
        UNIMPLEMENTED_MSG("debugger branches");
        return true;
    default:
        return false;
    }
}

bool GcnInst::IsCmpx() const {
    const auto in = [this](Opcode first, Opcode last) { return opcode >= first && opcode <= last; };
    return in(Opcode::V_CMPX_F_F32, Opcode::V_CMPX_T_F32) ||
           in(Opcode::V_CMPX_F_F64, Opcode::V_CMPX_T_F64) ||
           in(Opcode::V_CMPSX_F_F32, Opcode::V_CMPSX_T_F32) ||
           in(Opcode::V_CMPSX_F_F64, Opcode::V_CMPSX_T_F64) ||
           in(Opcode::V_CMPX_F_I32, Opcode::V_CMPX_CLASS_F32) ||
           in(Opcode::V_CMPX_F_I64, Opcode::V_CMPX_CLASS_F64) ||
           in(Opcode::V_CMPX_F_U32, Opcode::V_CMPX_T_U32) ||
           in(Opcode::V_CMPX_F_U64, Opcode::V_CMPX_T_U64);
}

DppOperation Dpp::GetOperation() const {
    const auto make = [this](DppCtrl op, u16 value) {
        return DppOperation{op, value, bool(bc), u8(row_mask), u8(bank_mask)};
    };
    if (dpp_ctrl < 0x100) {
        return make(DppCtrl::DppQuadPerm, u16(dpp_ctrl & 0xFF));
    }
    if (dpp_ctrl >= 0x101 && dpp_ctrl < 0x110) {
        return make(DppCtrl::DppRowShl, u16(dpp_ctrl & 0xF));
    }
    if (dpp_ctrl >= 0x111 && dpp_ctrl < 0x120) {
        return make(DppCtrl::DppRowShr, u16(dpp_ctrl & 0xF));
    }
    if (dpp_ctrl >= 0x121 && dpp_ctrl < 0x130) {
        return make(DppCtrl::DppRowRor, u16(dpp_ctrl & 0xF));
    }
    if (dpp_ctrl == u32(DppCtrl::DppRowMirror)) {
        return make(DppCtrl::DppRowMirror, 0);
    }
    if (dpp_ctrl == u32(DppCtrl::DppRowHalfMirror)) {
        return make(DppCtrl::DppRowHalfMirror, 0);
    }
    UNREACHABLE_MSG("dpp_ctrl {:#x}", u32(dpp_ctrl));
}

namespace {

MimgModifierFlags ParseMimgModifiers(Opcode opcode) noexcept {
    using enum MimgModifier;
    MimgModifierFlags flags{};
    // Sampling: IMAGE_SAMPLE[_C][_D|_L|_B|_LZ|_CD][_CL][_O]; gathers the same without
    // derivatives. The names carry the modifiers.
    const std::string_view name = NameOf(opcode);
    const bool sample = name.starts_with("IMAGE_SAMPLE");
    const bool gather = name.starts_with("IMAGE_GATHER4");
    if (!sample && !gather) {
        return flags;
    }
    std::string_view rest = name.substr(sample ? 12 : 13);
    if (gather && rest.starts_with("_H")) {
        return flags; // IMAGE_GATHER4H: not a sampling form with modifiers
    }
    const auto take = [&rest](std::string_view part) {
        if (rest.starts_with(part) &&
            (rest.size() == part.size() || rest[part.size()] == '_')) {
            rest.remove_prefix(part.size());
            return true;
        }
        return false;
    };
    if (take("_C")) {
        flags.set(Pcf);
    }
    if (take("_D")) {
        flags.set(Derivative);
    } else if (take("_CD")) {
        flags.set(CoarseDerivative);
    } else if (take("_L")) {
        flags.set(Lod);
    } else if (take("_B")) {
        flags.set(LodBias);
    } else if (take("_LZ")) {
        flags.set(Level0);
    }
    if (take("_CL")) {
        flags.set(LodClamp);
    }
    if (take("_O")) {
        flags.set(Offset);
    }
    return rest.empty() ? flags : MimgModifierFlags{};
}

} // namespace

MimgModifierFlags MimgModifiersOf(Opcode opcode) noexcept {
    static const auto modifiers = [] {
        std::array<MimgModifierFlags, OpcodeRange::MIMG> table{};
        for (u32 i = 0; i < table.size(); ++i) {
            table[i] = ParseMimgModifiers(Opcode(OpcodeBase::MIMG + i));
        }
        return table;
    }();
    const u32 index = u32(opcode) - OpcodeBase::MIMG;
    return index < modifiers.size() ? modifiers[index] : MimgModifierFlags{};
}

} // namespace GcnIsa
