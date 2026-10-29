// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Stage outputs passed through memory rings on GCN become attributes: the local stage's LDS
// writes (to the hull shader), the export stage's ES ring stores (to the geometry shader), and the
// geometry shader's ring reads and GS-VS ring stores (mapped through the copy shader).
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/operand_helper.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/position.h"
#include "shader/frontend/copy_shader.h"
#include "shader/runtime_info.h"

namespace Shader::Optimization {

namespace {

/// Ring accesses are buffer accesses with both coherence bits set.
bool IsRingAccess(const IR::Inst& inst) {
    const auto flags = inst.Flags<IR::BufferInstInfo>();
    return flags.system_coherent && flags.globally_coherent;
}

template <typename Func>
void ForEachInstruction(const IR::Program& program, Func&& func) {
    for (IR::Block* block : program.blocks) {
        for (IR::Inst& inst : block->Instructions()) {
            IR::IREmitter ir{*block, IR::Block::InstructionList::s_iterator_to(inst)};
            func(ir, inst);
        }
    }
}

/// Local stage: LDS writes at a constant offset are the outputs, 16 bytes per parameter.
void LocalStage(const IR::Program& program) {
    ForEachInstruction(program, [](IR::IREmitter& ir, IR::Inst& inst) {
        const auto opcode = inst.GetOpcode();
        if (opcode != IR::Opcode::WriteSharedU32 && opcode != IR::Opcode::WriteSharedU64) {
            return;
        }
        const bool pair = opcode == IR::Opcode::WriteSharedU64;
        ASSERT(inst.Arg(0).IsImmediate());
        u32 offset = inst.Arg(0).U32();
        const IR::Value data = pair ? ir.UnpackUint2x32(IR::U64{inst.Arg(1)}) : inst.Arg(1);
        for (s32 i = 0; i < (pair ? 2 : 1); ++i) {
            const auto attribute = IR::Attribute::Param0 + (offset / 16);
            const auto comp = (offset / 4) % 4;
            const IR::U32 value = IR::U32{pair ? ir.CompositeExtract(data, i) : data};
            ir.SetAttribute(attribute, ir.BitCast<IR::F32, IR::U32>(value), comp);
            offset += 4;
        }
        inst.Invalidate();
    });
}

/// Export stage: ES ring stores; the first 16 bytes the position, then the parameters.
void ExportStage(const IR::Program& program, const RuntimeInfo& runtime_info) {
    ForEachInstruction(program, [&](IR::IREmitter& ir, IR::Inst& inst) {
        if (inst.GetOpcode() != IR::Opcode::StoreBufferU32 || !IsRingAccess(inst)) {
            return;
        }
        const auto offset = inst.Flags<IR::BufferInstInfo>().inst_offset.Value();
        ASSERT(offset < runtime_info.hw.es.vertex_data_size * 4);
        const auto data = ir.BitCast<IR::F32>(IR::U32{inst.Arg(2)});
        const auto attribute = IR::Value{offset < 16 ? IR::Attribute::Position0
                                                     : IR::Attribute::Param0 + (offset / 16 - 1)};
        const u32 comp = (offset / 4) % 4;
        inst.ReplaceOpcode(IR::Opcode::SetAttribute);
        inst.ClearArgs();
        inst.SetArg(0, attribute);
        inst.SetArg(1, data);
        inst.SetArg(2, ir.Imm32(comp));
    });
}

/// Geometry stage: ES ring reads (256 bytes per component bucket) are input attributes; GS-VS
/// ring stores are outputs, at the attributes the copy shader reads them to.
void GeometryStage(const IR::Program& program, const RuntimeInfo& runtime_info) {
    auto& info = program.info;
    const auto& gs = runtime_info.hw.gs;
    info.gs_copy_data = Shader::ParseCopyShader(gs.vs_copy);
    u32 dwords_per_vertex = gs.out_vertex_data_size;
    if (info.gs_copy_data.num_comps && info.gs_copy_data.num_comps > dwords_per_vertex) {
        LOG_WARNING(Render_Vulkan,
                    "VERT_ITEMSIZE {} is different than actual number of dwords per vertex {}",
                    dwords_per_vertex, info.gs_copy_data.num_comps);
        dwords_per_vertex = info.gs_copy_data.num_comps;
    }
    ForEachInstruction(program, [&](IR::IREmitter& ir, IR::Inst& inst) {
        switch (inst.GetOpcode()) {
        case IR::Opcode::LoadBufferU32: {
            if (!IsRingAccess(inst)) {
                break;
            }
            const auto flags = inst.Flags<IR::BufferInstInfo>();
            const auto vertex_id = (flags.index_enable ? IR::GetBufferIndexArg(&inst)
                                                       : IR::GetBufferVOffsetArg(&inst))
                                       .U32() >>
                                   2;
            const auto bucket = IR::GetBufferSOffsetArg(&inst).U32() / 256u;
            const auto attribute = bucket < 4 ? IR::Attribute::Position0
                                              : IR::Attribute::Param0 + (bucket / 4 - 1);
            const auto value = ir.GetAttribute(attribute, bucket % 4, vertex_id);
            inst.ReplaceOpcode(IR::Opcode::BitCastU32F32);
            inst.ClearArgs();
            inst.SetArg(0, value);
            break;
        }
        case IR::Opcode::StoreBufferU32: {
            if (!IsRingAccess(inst)) {
                break;
            }
            const auto offset = inst.Flags<IR::BufferInstInfo>().inst_offset.Value();
            const auto data = ir.BitCast<IR::F32>(IR::U32{inst.Arg(2)});
            const auto comp_stride = info.gs_copy_data.output_vertices * sizeof(u32);
            const auto output_size = gs.output_vertices * dwords_per_vertex * sizeof(u32);
            const auto read_offset = (((offset / comp_stride) * comp_stride) % output_size) * 16u;
            const auto it = info.gs_copy_data.attr_map.find(read_offset);
            ASSERT(it != info.gs_copy_data.attr_map.cend());
            const auto& [attribute, comp] = it->second;
            inst.Invalidate();
            if (IsPosition(attribute)) {
                ExportPosition(ir, gs, false, attribute, comp, data);
            } else {
                ir.SetAttribute(attribute, data, comp);
            }
            break;
        }
        default:
            break;
        }
    });
}

} // namespace

void RingAccessElimination(const IR::Program& program, const RuntimeInfo& runtime_info) {
    switch (program.info.hw_stage) {
    case HwStage::Local:
        LocalStage(program);
        break;
    case HwStage::Export:
        ExportStage(program, runtime_info);
        break;
    case HwStage::Geometry:
        GeometryStage(program, runtime_info);
        break;
    default:
        break;
    }
}

} // namespace Shader::Optimization
