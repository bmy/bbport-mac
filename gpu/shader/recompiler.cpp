// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "common/logging/log.h"
#include "gcn/decoder.h"
#include "shader/frontend/cfg.h"
#include "shader/frontend/structured_control_flow.h"
#include "shader/frontend/translate/translate.h"
#include "shader/ir/passes/ir_passes.h"
#include "shader/ir/post_order.h"
#include "shader/ir/program.h"
#include "shader/recompiler.h"
#include "shader/profile.h"

namespace Shader {

namespace {

/// The syntax list's blocks in order.
IR::BlockList BlocksOf(const IR::AbstractSyntaxList& syntax_list) {
    IR::BlockList blocks;
    blocks.reserve(std::ranges::count(syntax_list, IR::AbstractSyntaxNode::Type::Block,
                                      &IR::AbstractSyntaxNode::type));
    for (const auto& node : syntax_list) {
        if (node.type == IR::AbstractSyntaxNode::Type::Block) {
            blocks.push_back(node.data.block);
        }
    }
    return blocks;
}

/// Translates each CFG block into an IR block, then the prologue, the edges and the post order.
void TranslateBlocks(IR::Program& program, Pools& pools, Gcn::CFG& cfg, RuntimeInfo& runtime_info,
                     const Profile& profile) {
    Gcn::Translator translator{program.info, runtime_info, profile};
    for (auto& block : cfg) {
        const u32 start = block.begin_index;
        const u32 size = block.end_index - start + 1;
        IR::Block* const ir_block = pools.block_pool.Create(pools.inst_pool);
        ir_block->cfg_block = &block;
        block.ir_block = ir_block;
        translator.Translate(ir_block, block.begin, block.cond,
                             std::span{program.ins_list}.subspan(start, size));
        program.blocks.push_back(ir_block);
    }
    translator.EmitPrologue(program.blocks.front());
    ASSERT_MSG(!program.info.translation_failed, "Shader translation has failed");
    for (auto& block : cfg) {
        if (block.branch_true) {
            block.ir_block->AddBranch(block.branch_true->ir_block);
        }
        if (block.branch_false) {
            block.ir_block->AddBranch(block.branch_false->ir_block);
        }
    }
    program.post_order_blocks = IR::PostOrder(program.blocks.front());
}

} // namespace

IR::Program TranslateProgram(const std::span<const u32>& code, Pools& pools, Info& info,
                             RuntimeInfo& runtime_info, const Profile& profile) {
    using namespace Optimization;
    constexpr u32 MovVccHiLiteral = 0xBEEB03FF; // s_mov_b32 vcc_hi, #imm (shader start marker)
    if (code[0] != MovVccHiLiteral) {
        LOG_WARNING(Render_Recompiler, "First instruction is not s_mov_b32 vcc_hi, #imm");
    }

    IR::Program program{info};
    program.ins_list.reserve(code.size());
    for (const u32* pc = code.data(); pc < code.data() + code.size();) {
        program.ins_list.emplace_back(GcnIsa::Decode(pc));
    }
    pools.ReleaseContents();

    // The CFG's blocks (and the structurizer's) live until the end of the compilation.
    Common::ObjectPool<Gcn::Block> flow_blocks{64};
    Gcn::CFG cfg{flow_blocks, program.ins_list};
    TranslateBlocks(program, pools, cfg, runtime_info, profile);

    // Hosts whose clip distance interpolation is broken get it in the fragment shader.
    InjectClipDistanceAttributes(program, runtime_info);

    // On the unstructured CFG.
    if (!profile.support_float64) {
        LowerFp64ToFp32(program);
    }
    SsaRewritePass(program);
    ConstantPropagationPass(program.post_order_blocks);
    if (info.sw_stage == SwStage::TessellationControl) {
        TessellationPreprocess(program, runtime_info);
        HullShaderTransform(program, runtime_info);
    } else if (info.sw_stage == SwStage::TessellationEval) {
        TessellationPreprocess(program, runtime_info);
        DomainShaderTransform(program, runtime_info);
    }
    RingAccessElimination(program, runtime_info);
    ReadLaneEliminationPass(program);
    IR::DumpProgram(program, info, "pre-res-discover.");
    const auto resources = ResourceDiscoverPass(program, profile);
    FlattenExtendedUserdataPass(program);
    IR::DumpProgram(program, info, "pre-res-patch.");
    ResourcePatchingPass(program.info, resources, profile);
    LowerBufferFormatToRaw(program);
    SharedMemorySimplifyPass(program, profile);
    SharedMemoryToStoragePass(program, runtime_info, profile);
    LowerUserClipPlanes(program, runtime_info);
    PhiSimplificationPass(program);
    InverseBallotEliminationPass(program);
    LowerHardwareIntrinsics(program);
    IR::DumpProgram(program, info, "pre-lower-phi.");

    // Structurize: the edges and SSA state go, phis become goto-safe register copies.
    for (IR::Block* const block : program.blocks) {
        block->imm_predecessors.clear();
        block->imm_successors.clear();
        block->ssa_state.Reset();
    }
    LowerPhisToRegsPass(program);
    program.syntax_list = Gcn::BuildASL(pools, cfg, flow_blocks);
    program.blocks = BlocksOf(program.syntax_list);
    program.post_order_blocks = IR::PostOrder(program.syntax_list.front().data.block);

    // On the structured program.
    SsaRepairPass(program);
    SsaRewritePass(program);
    SharedMemoryBarrierPass(program, runtime_info, profile);
    DeadCodeEliminationPass(program);
    LowerWave64BallotPass(program, runtime_info, profile);
    ConstantPropagationPass(program.post_order_blocks);
    DeadCodeEliminationPass(program);
    if (info.sw_stage == SwStage::Fragment && profile.needs_integer_interpolation_fix) {
        InterpolatedIntegerPass(program);
    }
    CollectShaderInfoPass(program, profile);
    IR::DumpProgram(program, info);
    return program;
}

} // namespace Shader
