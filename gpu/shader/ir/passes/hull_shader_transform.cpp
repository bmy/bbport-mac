// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Tessellation on GCN passes control points through LDS: the local stage writes the hull
// shader's input patch there, the hull shader writes its output control points and per-patch
// constants after it, and the domain shader reads both. Tessellation factors go to a buffer the
// driver binds. Here these become stage attributes:
//
// LDS layout (all patches of the VGT group):
//   input CPs of patch 0, 1, ...  | output CPs of patch 0, 1, ...  | per-patch data of 0, 1, ...
// (a passthrough hull shader writes no control points: input and output overlap).
//
// Addresses come from the driver's tessellation constants V# (TessellationDataConstantBuffer),
// typically:
//   input CP:   PatchIdInVgt * ls_stride * input_cps + index * ls_stride + attr * 16 + comp * 4
//   output CP:  NumPatch * ls_stride * input_cps + PatchIdInVgt * output_patch_stride
//               + InvocationId * hs_cp_stride + attr * 16 + comp * 4
//   per patch:  NumPatch * ls_stride * input_cps + NumPatch * output_patch_stride
//               + PatchIdInVgt * patch_const_stride + attr * 16 + comp * 4
// Which region an access touches is told by which region-base constants (HsNumPatch,
// HsOutputBase, PatchConstBase) reach its address: each such read counts on the LDS
// instructions its value reaches (1 for the first two, 2 for PatchConstBase). The constants are
// then replaced with 0, which leaves addresses within the current patch, as Vulkan's per-patch
// arrays address them; the control point index is addr / stride, the attribute and component
// come from addr % stride. Terms of the address that are multiples of the stride are dropped
// from the modulo.
//
// The hull shader's packed V1 (InvocationId << 8 | PatchIdInVgt) is read with BFEs: the patch id
// becomes 0, the invocation id the InvocationId attribute (0 in passthrough shaders).
// Tessellation factor stores (to TessFactorsBufferBase) become the tess level patch outputs.
#include <optional>
#include <unordered_map>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/alignment.h"
#include "common/assert.h"
#include "shader/ir/breadth_first_search.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/operand_helper.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/program.h"
#include "shader/info.h"
#include "shader/runtime_info.h"

namespace Shader::Optimization {

namespace {

using IR::Opcode;

/// The instruction producing `value` when it is an `op`.
const IR::Inst* InstOf(const IR::Value& value, Opcode op) {
    const IR::Inst* const inst = value.TryInst();
    return inst && inst->GetOpcode() == op ? inst : nullptr;
}

bool IsU32(const IR::Value& value, u32 imm) {
    return value.IsImmediate() && value.Type() == IR::Type::U32 && value.U32() == imm;
}

/// GetAttributeU32 of `attribute`.
bool IsAttributeRead(const IR::Value& value, IR::Attribute attribute) {
    const IR::Inst* const inst = InstOf(value, Opcode::GetAttributeU32);
    return inst && inst->Arg(0).Type() == IR::Type::Attribute &&
           inst->Arg(0).Attribute() == attribute;
}

/// A BFE (unsigned, or either when `either_sign`) of `offset` and `count`: its base.
const IR::Inst* BitfieldExtract(const IR::Value& value, u32 offset, u32 count, bool either_sign) {
    const IR::Inst* inst = InstOf(value, Opcode::BitFieldUExtract);
    if (!inst && either_sign) {
        inst = InstOf(value, Opcode::BitFieldSExtract);
    }
    return inst && IsU32(inst->Arg(1), offset) && IsU32(inst->Arg(2), count) ? inst : nullptr;
}

struct TessSharpLocation {
    IR::ScalarReg ptr_base;
    u32 dword_off;
};

/// Where the V# of a ReadConstBuffer comes from: user data directly (ptr_base Max), or a
/// constant read through a user data pointer.
std::optional<TessSharpLocation> FindTessConstantSharp(const IR::Inst* read_const_buffer) {
    const IR::Inst* const sharp = InstOf(read_const_buffer->Arg(0), Opcode::CompositeConstructU32x4);
    if (!sharp) {
        return std::nullopt;
    }
    const IR::Value dw0 = sharp->Arg(0);
    if (const IR::Inst* const user_data = InstOf(dw0, Opcode::GetUserData)) {
        if (user_data->Arg(0).IsImmediate()) {
            return TessSharpLocation{IR::ScalarReg::Max,
                                     static_cast<u32>(user_data->Arg(0).ScalarReg())};
        }
        return std::nullopt;
    }
    const IR::Inst* const read = InstOf(dw0, Opcode::ReadConst);
    if (!read || !read->Arg(1).IsImmediate()) {
        return std::nullopt;
    }
    const IR::Inst* const pointer = InstOf(read->Arg(0), Opcode::CompositeConstructU32x2);
    if (!pointer) {
        return std::nullopt;
    }
    const IR::Inst* const base = InstOf(pointer->Arg(0), Opcode::GetUserData);
    if (!base || !base->Arg(0).IsImmediate()) {
        return std::nullopt;
    }
    return TessSharpLocation{base->Arg(0).ScalarReg(), read->Arg(1).U32()};
}

/// Counts, on the LDS accesses a region-base constant reaches through its address, which bases
/// it went through (see the top). A phi reached through another edge than on earlier walks
/// would make the region ambiguous (asserted).
class RegionBaseWalker {
public:
    void Walk(IR::Inst* read_const_buffer, TessConstantAttribute attr) {
        u32 weight;
        switch (attr) {
        case TessConstantAttribute::HsNumPatch:
        case TessConstantAttribute::HsOutputBase:
            weight = 1;
            break;
        case TessConstantAttribute::PatchConstBase:
            weight = 2;
            break;
        default:
            UNREACHABLE();
        }
        for (const IR::Use use : read_const_buffer->Uses()) {
            Visit(use, weight, false);
        }
        ++walk;
    }

private:
    struct PhiVisit {
        u32 walk;
        u32 edge;
    };

    void Visit(IR::Use use, u32 weight, bool ambiguous) {
        IR::Inst* const inst = use.user;
        switch (inst->GetOpcode()) {
        case Opcode::LoadSharedU32:
        case Opcode::LoadSharedU64:
        case Opcode::WriteSharedU32:
        case Opcode::WriteSharedU64:
            if (use.operand == 0) {
                inst->SetFlags<u32>(inst->Flags<u32>() + weight);
                ASSERT_MSG(!ambiguous, "LDS instruction {} accesses ambiguous attribute type",
                           fmt::ptr(inst));
                return;
            }
            break;
        case Opcode::Phi: {
            const auto it = phis.find(inst);
            if (it == phis.end()) {
                phis[inst] = {.walk = walk, .edge = static_cast<u16>(use.operand)};
            } else if (it->second.walk < walk) {
                it->second.walk = walk;
                ambiguous |= it->second.edge != use.operand;
            } else {
                // A cycle within this walk.
                ASSERT(it->second.walk == walk);
                return;
            }
            break;
        }
        default:
            break;
        }
        for (const IR::Use next : inst->Uses()) {
            Visit(next, weight, ambiguous);
        }
    }

    std::unordered_map<const IR::Inst*, PhiVisit> phis;
    u32 walk{1u};
};

enum class AttributeRegion : u32 { InputCP, OutputCP, PatchConst };

AttributeRegion RegionOf(const IR::Inst* lds_access, const Info& info,
                         const RuntimeInfo& runtime_info) {
    const u32 count = lds_access->Flags<u32>();
    if (count == 0) {
        return AttributeRegion::InputCP;
    }
    if (info.sw_stage == SwStage::TessellationControl && runtime_info.sw.tcs.IsPassthrough()) {
        // No output control points in LDS.
        ASSERT(count <= 1);
        return AttributeRegion::PatchConst;
    }
    ASSERT(count <= 2);
    return AttributeRegion(count);
}

bool IsMultipleOf(const IR::Value& term, u32 stride) {
    if (IsU32(term, stride)) {
        return true;
    }
    if (const IR::Inst* const bfe = BitfieldExtract(term, 0, 24, true)) {
        return IsMultipleOf(bfe->Arg(0), stride);
    }
    if (const IR::Inst* const mul = InstOf(term, Opcode::IMul32)) {
        return IsMultipleOf(mul->Arg(0), stride) || IsMultipleOf(mul->Arg(1), stride);
    }
    return false;
}

/// Splits a sum into the terms that are not multiples of `stride`; true when some were.
bool CollectNonMultiples(const IR::Value& sum, u32 stride, std::vector<IR::U32>& terms) {
    if (const IR::Inst* const add = InstOf(sum, Opcode::IAdd32)) {
        const bool dropped = CollectNonMultiples(add->Arg(0), stride, terms);
        return CollectNonMultiples(add->Arg(1), stride, terms) || dropped;
    }
    if (IsMultipleOf(sum, stride)) {
        return true;
    }
    terms.push_back(IR::U32{sum});
    return false;
}

/// The address with the terms that vanish modulo `stride` dropped:
/// (a + b) mod N = (a mod N + b mod N) mod N.
IR::U32 DropMultiples(IR::IREmitter& ir, IR::U32 addr, u32 stride) {
    std::vector<IR::U32> terms;
    if (CollectNonMultiples(addr, stride, terms)) {
        addr = ir.Imm32(0);
        for (const IR::U32& term : terms) {
            addr = ir.IAdd(addr, term);
        }
    }
    return addr;
}

/// The attribute and component an address points at within a control point of `stride`.
std::pair<IR::U32, IR::U32> AttributeAndComponent(IR::IREmitter& ir, const IR::U32& addr,
                                                  u32 stride) {
    const IR::U32 offset = ir.IMod(DropMultiples(ir, addr, stride), ir.Imm32(stride));
    const IR::U32 attr_index = ir.ShiftRightLogical(offset, ir.Imm32(4u));
    const IR::U32 comp_index =
        ir.ShiftRightLogical(ir.BitwiseAnd(offset, ir.Imm32(0xFU)), ir.Imm32(2u));
    return {attr_index, comp_index};
}

/// A control point read: a hull shader input (InputCP) or output, or a domain shader input.
IR::F32 ReadControlPoint(IR::IREmitter& ir, IR::U32 addr, u32 stride, u32 off_dw,
                         bool tcs_output) {
    if (off_dw > 0) {
        addr = ir.IAdd(addr, ir.Imm32(off_dw));
    }
    const IR::U32 control_point = ir.IDiv(addr, ir.Imm32(stride));
    const auto [attr_index, comp_index] = AttributeAndComponent(ir, addr, stride);
    if (tcs_output) {
        return ir.ReadTcsGenericOuputAttribute(control_point, attr_index, comp_index);
    }
    return ir.GetTessGenericAttribute(control_point, attr_index, comp_index);
}

/// An LDS load of one or two dwords built from per-dword reads.
template <typename ReadDword>
IR::Value LoadDwords(IR::IREmitter& ir, u32 num_dwords, ReadDword&& read) {
    if (num_dwords == 1) {
        return ir.BitCast<IR::U32>(read(0));
    }
    boost::container::static_vector<IR::Value, 4> dwords;
    for (u32 i = 0; i < num_dwords; i++) {
        const IR::F32 dword = read(i);
        dwords.push_back(ir.BitCast<IR::U32>(dword));
    }
    return ir.PackUint2x32(ir.CompositeConstruct(dwords));
}

/// The tess level a factor index of the factor buffer is: triangles store 3 edge factors and
/// the interior one, quads 4 edges and 2 interior, isolines 2 (the layout follows the domain).
IR::Patch TessFactor(const RuntimeInfo& runtime_info, u32 index) {
    switch (runtime_info.sw.tcs.tess_type) {
    case ::Gcn::TessellationType::Isoline:
        ASSERT(index < 2);
        return IR::PatchFactor(index);
    case ::Gcn::TessellationType::Triangle:
        ASSERT(index < 4);
        return index == 3 ? IR::Patch::TessellationLodInteriorU : IR::PatchFactor(index);
    case ::Gcn::TessellationType::Quad:
        ASSERT(index < 6);
        return IR::PatchFactor(index);
    default:
        UNREACHABLE();
    }
}

void LowerTessFactorStore(IR::Block& block, IR::Inst& inst, const RuntimeInfo& runtime_info) {
    if (!IsAttributeRead(IR::GetBufferSOffsetArg(&inst), IR::Attribute::TessFactorsBufferBase)) {
        return;
    }
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::Inst* const address =
        InstOf(inst.Arg(IR::StoreBufferArgs::Address), Opcode::CompositeConstructU32x3);
    ASSERT_MSG(address && IsU32(address->Arg(0), 0) && address->Arg(1).IsImmediate(),
               "unhandled pattern in tess factor store");
    const u32 first = (flags.inst_offset.Value() + address->Arg(1).U32()) >> 2;
    const IR::Value data = inst.Arg(IR::StoreBufferArgs::Data);
    const u32 num_dwords = u32(inst.GetOpcode()) - u32(Opcode::StoreBufferU32) + 1;

    const auto as_float = [&](const IR::Value& value) -> IR::F32 {
        if (const IR::Inst* const cast = InstOf(value, Opcode::BitCastU32F32)) {
            return IR::F32{cast->Arg(0)};
        }
        return ir.BitCast<IR::F32, IR::U32>(IR::U32{value});
    };
    inst.Invalidate();
    if (num_dwords == 1) {
        ir.SetPatch(TessFactor(runtime_info, first), as_float(data));
        return;
    }
    const IR::Inst* const vector = data.TryInst();
    ASSERT(vector && (vector->GetOpcode() == Opcode::CompositeConstructU32x2 ||
                      vector->GetOpcode() == Opcode::CompositeConstructU32x3 ||
                      vector->GetOpcode() == Opcode::CompositeConstructU32x4));
    for (u32 i = 0; i < num_dwords; i++) {
        ir.SetPatch(TessFactor(runtime_info, first + i), as_float(vector->Arg(i)));
    }
}

void LowerHullLdsWrite(IR::Block& block, IR::Inst& inst, const Info& info,
                       const RuntimeInfo& runtime_info) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const u32 num_dwords = inst.GetOpcode() == Opcode::WriteSharedU32 ? 1 : 2;
    const IR::U32 addr{inst.Arg(0)};
    const IR::Value data =
        num_dwords == 2 ? ir.UnpackUint2x32(IR::U64{inst.Arg(1)}) : inst.Arg(1);
    const AttributeRegion region = RegionOf(&inst, info, runtime_info);

    const auto write_dword = [&](IR::U32 dword_addr, IR::U32 value, u32 off_dw) {
        const IR::F32 component = ir.BitCast<IR::F32, IR::U32>(value);
        if (region == AttributeRegion::OutputCP) {
            if (off_dw > 0) {
                dword_addr = ir.IAdd(dword_addr, ir.Imm32(off_dw));
            }
            // The control point is the invocation's (implicit in the backend).
            const auto [attr_index, comp_index] =
                AttributeAndComponent(ir, dword_addr, runtime_info.sw.tcs.hs_output_cp_stride);
            ir.SetTcsGenericAttribute(component, attr_index, comp_index);
            return;
        }
        ASSERT(region == AttributeRegion::PatchConst);
        ASSERT_MSG(dword_addr.IsImmediate(), "patch addr non imm, inst {}",
                   fmt::ptr(dword_addr.Inst()));
        ir.SetPatch(IR::PatchGeneric((dword_addr.U32() >> 2) + off_dw), component);
    };
    if (num_dwords == 1) {
        write_dword(addr, IR::U32{data}, 0);
    } else {
        for (u32 i = 0; i < num_dwords; i++) {
            write_dword(addr, IR::U32{ir.CompositeExtract(data, i)}, i);
        }
    }
    inst.Invalidate();
}

void LowerHullLdsRead(IR::Block& block, IR::Inst& inst, const Info& info,
                      const RuntimeInfo& runtime_info) {
    IR::IREmitter ir{block, IR::Block::InstructionList::s_iterator_to(inst)};
    const IR::U32 addr{inst.Arg(0)};
    const AttributeRegion region = RegionOf(&inst, info, runtime_info);
    ASSERT_MSG(region == AttributeRegion::InputCP || region == AttributeRegion::OutputCP,
               "Unhandled read of patchconst attribute in hull shader");
    const bool tcs_output = region == AttributeRegion::OutputCP;
    const u32 stride =
        tcs_output ? runtime_info.sw.tcs.hs_output_cp_stride : runtime_info.sw.tcs.ls_stride;
    const u32 num_dwords = inst.GetOpcode() == Opcode::LoadSharedU32 ? 1 : 2;
    inst.ReplaceUsesWithAndRemove(LoadDwords(
        ir, num_dwords, [&](u32 i) { return ReadControlPoint(ir, addr, stride, i, tcs_output); }));
}

/// A passthrough hull shader copies its input control point to its output first.
void CopyInputsToOutputs(IR::Program& program, const RuntimeInfo& runtime_info) {
    IR::Block* const entry = program.blocks.front();
    auto it = std::ranges::find_if(entry->Instructions(), [](const IR::Inst& inst) {
        return inst.GetOpcode() == Opcode::Prologue;
    });
    // After Prologue and SetExec.
    ASSERT(it != entry->end());
    ++it;
    ASSERT(it != entry->end());
    ++it;
    IR::IREmitter ir{*entry, it};
    const u32 num_attributes = Common::AlignUp(runtime_info.sw.tcs.ls_stride, 16) >> 4;
    const auto invocation_id = ir.GetAttributeU32(IR::Attribute::InvocationId);
    for (u32 attr = 0; attr < num_attributes; attr++) {
        for (u32 comp = 0; comp < 4; comp++) {
            const IR::F32 value =
                ir.GetTessGenericAttribute(invocation_id, ir.Imm32(attr), ir.Imm32(comp));
            ir.SetTcsGenericAttribute(value, ir.Imm32(attr), ir.Imm32(comp));
        }
    }
    // (The patch constant part runs in every invocation; as all compute the same, that is
    // equivalent to running it in invocation 0.)
}

/// Finds the tessellation constants V# through the addresses of LDS accesses and reads it.
void FindTessConstants(IR::Program& program, RuntimeInfo& runtime_info,
                       TessellationDataConstantBuffer& constants) {
    Info& info = program.info;
    const auto find_read = [](IR::Inst* inst) -> std::optional<IR::Inst*> {
        if (inst->GetOpcode() == Opcode::ReadConstBuffer) {
            return inst;
        }
        return std::nullopt;
    };
    for (IR::Block* const block : program.blocks) {
        // The first LDS access with a constant buffer read in its address, per block.
        for (IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case Opcode::LoadSharedU32:
            case Opcode::LoadSharedU64:
            case Opcode::WriteSharedU32:
            case Opcode::WriteSharedU64:
                break;
            default:
                continue;
            }
            const auto read = IR::BreadthFirstSearch(inst.Arg(0), find_read);
            if (!read) {
                continue;
            }
            const auto location = FindTessConstantSharp(*read);
            if (!location) {
                UNREACHABLE_MSG("Failed to match tess constant sharp");
            }
            if (info.tess_consts_dword_offset >= 0) {
                // (Another constant buffer read could feed an LDS address.)
                ASSERT_MSG(static_cast<s32>(location->dword_off) == info.tess_consts_dword_offset &&
                               location->ptr_base == info.tess_consts_ptr_base,
                           "TessConstants V# is ambiguous");
            }
            info.tess_consts_ptr_base = location->ptr_base;
            info.tess_consts_dword_offset = static_cast<s32>(location->dword_off);
            info.ReadTessConstantBuffer(constants);
            runtime_info.InitFromTessConstants(constants);
            break;
        }
    }
    ASSERT(info.tess_consts_dword_offset >= 0);
}

/// Strides become constants; the region bases are counted on the LDS accesses and become 0.
void ResolveTessConstantReads(IR::Program& program, const TessellationDataConstantBuffer& constants) {
    const Info& info = program.info;
    RegionBaseWalker walker;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() != Opcode::ReadConstBuffer) {
                continue;
            }
            const auto location = FindTessConstantSharp(&inst);
            if (!location || location->ptr_base != info.tess_consts_ptr_base ||
                location->dword_off != info.tess_consts_dword_offset) {
                continue;
            }
            const IR::Value index = inst.Arg(1);
            ASSERT_MSG(index.IsImmediate(), "Tessellation constant read with dynamic index");
            ASSERT(index.U32() <= static_cast<u32>(TessConstantAttribute::FirstEdgeTessFactorIndex));
            const auto attr = static_cast<TessConstantAttribute>(index.U32());
            switch (attr) {
            case TessConstantAttribute::LsStride:
                // (Would need to be runtime state for a domain shader.)
                ASSERT(info.sw_stage == SwStage::TessellationControl);
                inst.ReplaceUsesWithAndRemove(IR::Value(constants.ls_stride));
                break;
            case TessConstantAttribute::HsCpStride:
                inst.ReplaceUsesWithAndRemove(IR::Value(constants.hs_cp_stride));
                break;
            case TessConstantAttribute::HsNumPatch:
            case TessConstantAttribute::HsOutputBase:
            case TessConstantAttribute::PatchConstBase:
                walker.Walk(&inst, attr);
                inst.ReplaceUsesWithAndRemove(IR::Value(0u));
                break;
            case TessConstantAttribute::PatchConstSize:
            case TessConstantAttribute::PatchOutputSize:
            case TessConstantAttribute::OffChipTessellationFactorThreshold:
            case TessConstantAttribute::FirstEdgeTessFactorIndex:
                break;
            default:
                UNREACHABLE_MSG("Read past end of TessConstantsBuffer");
            }
        }
    }
}

/// The hull shader's V1 fields: PatchIdInVgt (bits 0-7) becomes 0, InvocationId (bits 8-12) the
/// attribute (0 in a passthrough shader, where Bloodborne uses it in patch constant addresses).
void ResolvePackedHullIds(IR::Program& program, const RuntimeInfo& runtime_info) {
    for (IR::Block* const block : program.blocks) {
        for (auto it = block->Instructions().begin(); it != block->Instructions().end(); ++it) {
            IR::Inst& inst = *it;
            const IR::Value value{&inst};
            const auto is_v1_field = [&](u32 offset, u32 count) {
                const IR::Inst* const bfe = BitfieldExtract(value, offset, count, false);
                return bfe && IsAttributeRead(bfe->Arg(0), IR::Attribute::PackedHullInvocationInfo);
            };
            if (is_v1_field(0, 8)) {
                inst.ReplaceUsesWithAndRemove(IR::Value(0u));
            } else if (is_v1_field(8, 5)) {
                IR::IREmitter ir{*block, it};
                const IR::Value invocation_id =
                    runtime_info.sw.tcs.IsPassthrough()
                        ? IR::Value{ir.Imm32(0)}
                        : IR::Value{ir.GetAttributeU32(IR::Attribute::InvocationId)};
                inst.ReplaceUsesWithAndRemove(invocation_id);
            }
        }
    }
}

} // namespace

void HullShaderTransform(IR::Program& program, const RuntimeInfo& runtime_info) {
    const Info& info = program.info;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            switch (inst.GetOpcode()) {
            case Opcode::StoreBufferU32:
            case Opcode::StoreBufferU32x2:
            case Opcode::StoreBufferU32x3:
            case Opcode::StoreBufferU32x4:
                LowerTessFactorStore(*block, inst, runtime_info);
                break;
            case Opcode::WriteSharedU32:
            case Opcode::WriteSharedU64:
                LowerHullLdsWrite(*block, inst, info, runtime_info);
                break;
            case Opcode::LoadSharedU32:
            case Opcode::LoadSharedU64:
                LowerHullLdsRead(*block, inst, info, runtime_info);
                break;
            default:
                break;
            }
        }
    }
    if (runtime_info.sw.tcs.IsPassthrough()) {
        CopyInputsToOutputs(program, runtime_info);
    }
}

void DomainShaderTransform(const IR::Program& program, const RuntimeInfo& runtime_info) {
    const Info& info = program.info;
    for (IR::Block* const block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            const Opcode op = inst.GetOpcode();
            if (op != Opcode::LoadSharedU32 && op != Opcode::LoadSharedU64) {
                continue;
            }
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            const IR::U32 addr{inst.Arg(0)};
            const AttributeRegion region = RegionOf(&inst, info, runtime_info);
            const auto read = [&](u32 off_dw) -> IR::F32 {
                if (region == AttributeRegion::OutputCP) {
                    return ReadControlPoint(ir, addr, runtime_info.sw.tcs.hs_output_cp_stride,
                                            off_dw, false);
                }
                ASSERT(region == AttributeRegion::PatchConst);
                return ir.GetPatch(IR::PatchGeneric((addr.U32() >> 2) + off_dw));
            };
            inst.ReplaceUsesWithAndRemove(
                LoadDwords(ir, op == Opcode::LoadSharedU32 ? 1 : 2, read));
        }
    }
}

void TessellationPreprocess(IR::Program& program, RuntimeInfo& runtime_info) {
    TessellationDataConstantBuffer constants;
    FindTessConstants(program, runtime_info, constants);
    ResolveTessConstantReads(program, constants);
    if (program.info.sw_stage == SwStage::TessellationControl) {
        ResolvePackedHullIds(program, runtime_info);
    }
    ConstantPropagationPass(program.post_order_blocks);
}

} // namespace Shader::Optimization
