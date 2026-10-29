// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The fixed-function user clip planes (PA_CL_UCP_*, enabled by PA_CL_CLIP_CNTL's UCP_ENA bits)
// as clip distances of the vertex shader: the plane equations change per draw and come from a
// per-draw uniform buffer (its binding is reserved for every vertex shader).
#include <algorithm>
#include <array>

#include "common/logging/log.h"
#include "shader/ir/ir_emitter.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/info.h"

namespace Shader::Optimization {

namespace {
constexpr u32 NumUserClipPlanes = 6;
}

void LowerUserClipPlanes(IR::Program& program, const RuntimeInfo& runtime_info) {
    Info& info = program.info;
    if (info.hw_stage != HwStage::Vertex) {
        return;
    }
    const u32 binding = static_cast<u32>(info.buffers.size());
    info.buffers.push_back(BufferResource{
        .used_types = IR::Type::F32,
        .buffer_type = BufferType::ClipPlanes,
    });
    const auto& vs = runtime_info.hw.vs;
    const u32 enabled = vs.user_clip_plane_mask;
    if (enabled == 0) {
        return;
    }
    // The hardware clips by user planes and exported distances independently; here both share
    // the eight distance slots and the exports own theirs: such shaders keep the exported
    // distances alone.
    for (u32 i = 0; i < vs.num_outputs; ++i) {
        for (const auto output : vs.outputs[i]) {
            if (output >= Output::ClipDist0 && output <= Output::ClipDist7) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping user clip plane lowering: shader {:#x} exports its own clip "
                            "distances",
                            info.pgm_hash);
                return;
            }
        }
    }
    // The distances are computed from the last export of the position.
    std::array<IR::Value, 4> position{};
    IR::Block* target_block{};
    IR::Block::iterator target_it{};
    for (IR::Block* const block : program.blocks) {
        for (auto inst = block->begin(); inst != block->end(); ++inst) {
            if (inst->GetOpcode() == IR::Opcode::SetAttribute &&
                inst->Arg(0).Attribute() == IR::Attribute::Position0) {
                position[inst->Arg(2).U32()] = inst->Arg(1);
                target_block = block;
                target_it = std::next(inst);
            }
        }
    }
    if (std::ranges::any_of(position, &IR::Value::IsEmpty)) {
        return;
    }
    IR::IREmitter ir{*target_block, target_it};
    const IR::U32 handle = ir.Imm32(binding);
    for (u32 plane = 0; plane < NumUserClipPlanes; ++plane) {
        if (((enabled >> plane) & 1) == 0) {
            continue;
        }
        // A plane is four dwords in the buffer: distance = dot(position, plane).
        const IR::Value equation = ir.LoadBufferF32(4, handle, ir.Imm32(plane * 4), {});
        IR::F32 distance{ir.FPMul(IR::F32{position[0]}, IR::F32{ir.CompositeExtract(equation, 0)})};
        for (u32 comp = 1; comp < 4; ++comp) {
            distance = IR::F32{ir.FPFma(IR::F32{position[comp]},
                                        IR::F32{ir.CompositeExtract(equation, comp)}, distance)};
        }
        ir.SetAttribute(IR::Attribute::ClipDistance, distance, plane);
    }
}

} // namespace Shader::Optimization
