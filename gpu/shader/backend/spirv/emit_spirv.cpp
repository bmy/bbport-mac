// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// The SPIR-V backend's driver: each IR instruction goes to its Emit<Opcode> function (operands
// converted by the function's parameter types), the abstract syntax list becomes structured
// control flow, and the module gets its capabilities, entry point, execution modes and float
// controls.
#include <mutex>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <magic_enum/magic_enum.hpp>

#include "common/assert.h"
#include "common/func_traits.h"
#include "common/logging/log.h"
#include "shader/backend/spirv/emit_spirv.h"
#include "shader/ir/basic_block.h"
#include "shader/ir/program.h"
#include "shader/runtime_info.h"
#include "shader/backend/spirv/emit_spirv_instructions.h"
#include "shader/backend/spirv/spirv_emit_context.h"

namespace Shader::Backend::SPIRV {

namespace {

// Instruction dispatch.

/// An IR operand as the Emit function's parameter type.
template <typename ArgType>
ArgType Arg(EmitContext& ctx, const IR::Value& arg) {
    if constexpr (std::is_same_v<ArgType, Id>) {
        return ctx.Def(arg);
    } else if constexpr (std::is_same_v<ArgType, const IR::Value&>) {
        return arg;
    } else if constexpr (std::is_same_v<ArgType, u32>) {
        return arg.U32();
    } else if constexpr (std::is_same_v<ArgType, u64>) {
        return arg.U64();
    } else if constexpr (std::is_same_v<ArgType, bool>) {
        return arg.U1();
    } else if constexpr (std::is_same_v<ArgType, IR::Attribute>) {
        return arg.Attribute();
    } else if constexpr (std::is_same_v<ArgType, IR::ScalarReg>) {
        return arg.ScalarReg();
    } else if constexpr (std::is_same_v<ArgType, IR::VectorReg>) {
        return arg.VectorReg();
    } else if constexpr (std::is_same_v<ArgType, const char*>) {
        return arg.StringLiteral();
    } else if constexpr (std::is_same_v<ArgType, IR::Patch>) {
        return arg.Patch();
    }
    UNREACHABLE();
}

template <auto func, typename... Args>
void SetDefinition(EmitContext& ctx, IR::Inst* inst, Args... args) {
    inst->SetDefinition<Id>(func(ctx, std::forward<Args>(args)...));
}

/// Calls `func` (taking the context, optionally the instruction, then the operands); an Id
/// result becomes the instruction's definition.
template <auto func, bool takes_inst, size_t... I>
void Invoke(EmitContext& ctx, IR::Inst* inst, std::index_sequence<I...>) {
    using Traits = Common::FuncTraits<decltype(func)>;
    if constexpr (std::is_same_v<typename Traits::ReturnType, Id>) {
        if constexpr (takes_inst) {
            SetDefinition<func>(
                ctx, inst, inst,
                Arg<typename Traits::template ArgType<I + 2>>(ctx, inst->Arg(I))...);
        } else {
            SetDefinition<func>(
                ctx, inst, Arg<typename Traits::template ArgType<I + 1>>(ctx, inst->Arg(I))...);
        }
    } else {
        if constexpr (takes_inst) {
            func(ctx, inst, Arg<typename Traits::template ArgType<I + 2>>(ctx, inst->Arg(I))...);
        } else {
            func(ctx, Arg<typename Traits::template ArgType<I + 1>>(ctx, inst->Arg(I))...);
        }
    }
}

template <auto func>
void Invoke(EmitContext& ctx, IR::Inst* inst) {
    using Traits = Common::FuncTraits<decltype(func)>;
    static_assert(Traits::NUM_ARGS >= 1, "Insufficient arguments");
    if constexpr (Traits::NUM_ARGS == 1) {
        Invoke<func, false>(ctx, inst, std::make_index_sequence<0>{});
    } else {
        constexpr bool takes_inst =
            std::is_same_v<typename Traits::template ArgType<1>, IR::Inst*>;
        Invoke<func, takes_inst>(ctx, inst,
                                 std::make_index_sequence<Traits::NUM_ARGS - (takes_inst ? 2 : 1)>{});
    }
}

void EmitInst(EmitContext& ctx, IR::Inst* inst) {
    switch (inst->GetOpcode()) {
#define OPCODE(name, result_type, ...)                                                             \
    case IR::Opcode::name:                                                                         \
        return Invoke<&Emit##name>(ctx, inst);
#include "shader/ir/opcodes.inc"
#undef OPCODE
    }
    UNREACHABLE_MSG("Invalid opcode {}", inst->GetOpcode());
}

/// The SPIR-V type of a phi (its IR type is in its flags).
Id TypeId(const EmitContext& ctx, IR::Type type) {
    switch (type) {
    case IR::Type::U1:
        return ctx.U1[1];
    case IR::Type::U32:
        return ctx.U32[1];
    case IR::Type::U32x2:
        return ctx.U32[2];
    case IR::Type::U32x3:
        return ctx.U32[3];
    case IR::Type::U32x4:
        return ctx.U32[4];
    case IR::Type::F32:
        return ctx.F32[1];
    case IR::Type::U64:
        return ctx.U64;
    default:
        UNREACHABLE_MSG("Phi node type {}", type);
    }
}

// Control flow.

/// Emits the syntax list: blocks with their instructions, and the merges and branches of ifs,
/// loops, breaks and returns.
void Traverse(EmitContext& ctx, const IR::Program& program) {
    using Type = IR::AbstractSyntaxNode::Type;
    IR::Block* current_block{};
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case Type::Block: {
            const Id label{node.data.block->Definition<Id>()};
            if (current_block) {
                ctx.OpBranch(label);
            }
            current_block = node.data.block;
            ctx.AddLabel(label);
            for (IR::Inst& inst : node.data.block->Instructions()) {
                EmitInst(ctx, &inst);
            }
            // Phis name the block that ends a predecessor, which may be a later label.
            ctx.first_to_last_label_map[label.value] = ctx.last_label;
            break;
        }
        case Type::If: {
            const Id if_label{node.data.if_node.body->Definition<Id>()};
            const Id endif_label{node.data.if_node.merge->Definition<Id>()};
            ctx.OpSelectionMerge(endif_label, spv::SelectionControlMask::MaskNone);
            ctx.OpBranchConditional(ctx.Def(node.data.if_node.cond), if_label, endif_label);
            break;
        }
        case Type::Loop: {
            const Id body_label{node.data.loop.body->Definition<Id>()};
            const Id continue_label{node.data.loop.continue_block->Definition<Id>()};
            const Id endloop_label{node.data.loop.merge->Definition<Id>()};
            ctx.OpLoopMerge(endloop_label, continue_label, spv::LoopControlMask::MaskNone);
            ctx.OpBranch(body_label);
            break;
        }
        case Type::Break: {
            const Id break_label{node.data.break_node.merge->Definition<Id>()};
            const Id skip_label{node.data.break_node.skip->Definition<Id>()};
            ctx.OpBranchConditional(ctx.Def(node.data.break_node.cond), break_label, skip_label);
            break;
        }
        case Type::EndIf:
            if (current_block) {
                ctx.OpBranch(node.data.end_if.merge->Definition<Id>());
            }
            break;
        case Type::Repeat: {
            const Id cond{ctx.Def(node.data.repeat.cond)};
            const Id loop_header_label{node.data.repeat.loop_header->Definition<Id>()};
            const Id merge_label{node.data.repeat.merge->Definition<Id>()};
            ctx.OpBranchConditional(cond, loop_header_label, merge_label);
            break;
        }
        case Type::Return:
            ctx.OpReturn();
            break;
        case Type::Unreachable:
            ctx.OpUnreachable();
            break;
        }
        if (node.type != Type::Block) {
            current_block = nullptr;
        }
    }
}

Id DefineMain(EmitContext& ctx, const IR::Program& program) {
    const Id void_function{ctx.TypeFunction(ctx.void_id)};
    const Id main{ctx.OpFunction(ctx.void_id, spv::FunctionControlMask::MaskNone, void_function)};
    for (IR::Block* const block : program.blocks) {
        block->SetDefinition(ctx.OpLabel());
    }
    Traverse(ctx, program);
    ctx.OpFunctionEnd();
    return main;
}

/// Phis were emitted before their operands' definitions existed: fill them now, in order.
void PatchPhiNodes(const IR::Program& program, EmitContext& ctx) {
    auto inst{program.blocks.front()->begin()};
    size_t block_index{0};
    ctx.PatchDeferredPhi([&](u32 phi_arg, Id first_parent) {
        if (phi_arg == 0) {
            ++inst;
            while (inst == program.blocks[block_index]->end() ||
                   inst->GetOpcode() != IR::Opcode::Phi) {
                ++block_index;
                inst = program.blocks[block_index]->begin();
            }
        }
        ASSERT(inst != program.blocks[block_index]->end());
        const Id arg = ctx.Def(inst->Arg(phi_arg));
        const Id parent = ctx.first_to_last_label_map[first_parent.value];
        return std::make_pair(arg, parent);
    });
}

// Module setup.

spv::ExecutionMode InputPrimitiveMode(::Gcn::PrimitiveType type) {
    switch (type) {
    case ::Gcn::PrimitiveType::PointList:
        return spv::ExecutionMode::InputPoints;
    case ::Gcn::PrimitiveType::LineList:
    case ::Gcn::PrimitiveType::LineStrip:
        return spv::ExecutionMode::InputLines;
    case ::Gcn::PrimitiveType::TriangleList:
    case ::Gcn::PrimitiveType::TriangleStrip:
    case ::Gcn::PrimitiveType::RectList:
        return spv::ExecutionMode::Triangles;
    case ::Gcn::PrimitiveType::AdjTriangleList:
        return spv::ExecutionMode::InputTrianglesAdjacency;
    case ::Gcn::PrimitiveType::AdjLineList:
        return spv::ExecutionMode::InputLinesAdjacency;
    default:
        UNREACHABLE_MSG("Unknown input primitive type {}", u32(type));
    }
}

spv::ExecutionMode OutputPrimitiveMode(::Gcn::GsOutputPrimitiveType type) {
    switch (type) {
    case ::Gcn::GsOutputPrimitiveType::PointList:
        return spv::ExecutionMode::OutputVertices;
    case ::Gcn::GsOutputPrimitiveType::LineStrip:
        return spv::ExecutionMode::OutputLineStrip;
    case ::Gcn::GsOutputPrimitiveType::TriangleStrip:
        return spv::ExecutionMode::OutputTriangleStrip;
    default:
        UNREACHABLE_MSG("Unknown output primitive type {}", u32(type));
    }
}

spv::ExecutionMode ExecutionMode(::Gcn::TessellationType primitive) {
    switch (primitive) {
    case ::Gcn::TessellationType::Isoline:
        return spv::ExecutionMode::Isolines;
    case ::Gcn::TessellationType::Triangle:
        return spv::ExecutionMode::Triangles;
    case ::Gcn::TessellationType::Quad:
        return spv::ExecutionMode::Quads;
    }
    UNREACHABLE_MSG("Tessellation primitive {}", magic_enum::enum_name(primitive));
}

spv::ExecutionMode ExecutionMode(::Gcn::TessellationPartitioning spacing) {
    switch (spacing) {
    case ::Gcn::TessellationPartitioning::Integer:
        return spv::ExecutionMode::SpacingEqual;
    case ::Gcn::TessellationPartitioning::FracOdd:
        return spv::ExecutionMode::SpacingFractionalOdd;
    case ::Gcn::TessellationPartitioning::FracEven:
        return spv::ExecutionMode::SpacingFractionalEven;
    case ::Gcn::TessellationPartitioning::Pow2:
        // Rounds factors to powers of 2: no Vulkan equivalent, integer spacing is closest.
        LOG_WARNING(Render_Vulkan, "Tessellation partitioning Pow2 has no Vulkan equivalent, "
                                   "falling back to SpacingEqual");
        return spv::ExecutionMode::SpacingEqual;
    default:
        break;
    }
    UNREACHABLE_MSG("Tessellation spacing {}", magic_enum::enum_name(spacing));
}

void SetupCapabilities(const Info& info, const Profile& profile, const RuntimeInfo& runtime_info,
                       EmitContext& ctx) {
    using spv::Capability;
    ctx.AddCapability(Capability::Image1D);
    ctx.AddCapability(Capability::Sampled1D);
    ctx.AddCapability(Capability::ImageQuery);
    ctx.AddCapability(Capability::Int8);
    ctx.AddCapability(Capability::Int16);
    ctx.AddCapability(Capability::Int64);
    ctx.AddCapability(Capability::StorageBuffer8BitAccess);
    ctx.AddCapability(Capability::StorageBuffer16BitAccess);
    if (info.uses_fp16) {
        ctx.AddCapability(Capability::Float16);
    }
    if (info.uses_fp64) {
        ctx.AddCapability(Capability::Float64);
    }
    if (info.has_storage_images) {
        ctx.AddCapability(Capability::StorageImageExtendedFormats);
        ctx.AddCapability(Capability::StorageImageReadWithoutFormat);
        ctx.AddCapability(Capability::StorageImageWriteWithoutFormat);
        if (profile.supports_image_load_store_lod) {
            ctx.AddExtension("SPV_AMD_shader_image_load_store_lod");
            ctx.AddCapability(Capability::ImageReadWriteLodAMD);
        }
    }
    if (info.has_image_gather) {
        ctx.AddCapability(Capability::ImageGatherExtended);
    }
    if (info.has_image_query) {
        ctx.AddCapability(Capability::ImageQuery);
    }
    if ((info.uses_image_atomic_float_min_max && profile.supports_image_fp32_atomic_min_max) ||
        (info.uses_buffer_atomic_float_min_max && profile.supports_buffer_fp32_atomic_min_max)) {
        ctx.AddExtension("SPV_EXT_shader_atomic_float_min_max");
        ctx.AddCapability(Capability::AtomicFloat32MinMaxEXT);
    }
    if (info.uses_lane_id) {
        ctx.AddCapability(Capability::GroupNonUniform);
    }
    if (info.uses_group_quad) {
        ctx.AddCapability(Capability::GroupNonUniformQuad);
    }
    if (info.uses_group_shuffle) {
        ctx.AddCapability(Capability::GroupNonUniformShuffle);
    }
    if (info.uses_group_ballot || info.loads.Get(IR::Attribute::SubgroupLtMask)) {
        ctx.AddCapability(Capability::GroupNonUniformBallot);
    }
    if (info.uses_shader_clock) {
        if (ctx.profile.supports_shader_subgroup_clock) {
            ctx.AddExtension("SPV_KHR_shader_clock");
            ctx.AddCapability(Capability::ShaderClockKHR);
        } else {
            LOG_WARNING(Render_Recompiler,
                        "Shader requires support for ShaderClockKHR capability"
                        " that your Vulkan instance does not advertise. Results may vary");
        }
    }
    const auto stage = info.sw_stage;
    if (stage == SwStage::Vertex) {
        ctx.AddExtension("SPV_KHR_shader_draw_parameters");
        ctx.AddCapability(Capability::DrawParameters);
    }
    if (stage == SwStage::Geometry) {
        ctx.AddCapability(Capability::Geometry);
    }
    if (info.hw_stage == HwStage::Fragment) {
        if (profile.supports_amd_shader_explicit_vertex_parameter) {
            ctx.AddExtension("SPV_AMD_shader_explicit_vertex_parameter");
        } else if (profile.supports_fragment_shader_barycentric) {
            ctx.AddExtension("SPV_KHR_fragment_shader_barycentric");
            ctx.AddCapability(Capability::FragmentBarycentricKHR);
            ctx.AddCapability(Capability::InterpolationFunction);
        }
        const auto& addr = runtime_info.hw.fs.addr_flags;
        const bool per_sample_barycentrics =
            !profile.supports_amd_shader_explicit_vertex_parameter &&
            profile.supports_fragment_shader_barycentric &&
            info.loads.Get(IR::Attribute::BaryCoordSmoothSample);
        if (info.loads.Get(IR::Attribute::SampleIndex) || addr.linear_sample_ena ||
            addr.persp_sample_ena || per_sample_barycentrics) {
            ctx.AddCapability(Capability::SampleRateShading);
        }
        if (info.loads.GetAny(IR::Attribute::RenderTargetIndex)) {
            ctx.AddCapability(Capability::Geometry);
        }
        if (info.stores.Get(IR::Attribute::StencilRef) && profile.supports_shader_stencil_export) {
            ctx.AddExtension("SPV_EXT_shader_stencil_export");
            ctx.AddCapability(Capability::StencilExportEXT);
        }
    }
    if (stage == SwStage::TessellationControl || stage == SwStage::TessellationEval) {
        ctx.AddCapability(Capability::Tessellation);
    }
    if (stage == SwStage::Vertex || stage == SwStage::TessellationControl ||
        stage == SwStage::TessellationEval) {
        if (info.stores.GetAny(IR::Attribute::RenderTargetIndex)) {
            ctx.AddCapability(Capability::ShaderLayer);
        }
        if (info.stores.GetAny(IR::Attribute::ViewportIndex)) {
            ctx.AddCapability(Capability::ShaderViewportIndex);
        }
    } else if (stage == SwStage::Geometry && info.stores.GetAny(IR::Attribute::ViewportIndex)) {
        ctx.AddCapability(Capability::MultiViewport);
    }
    if (info.uses_dma || info.uses_paged_buffers || ctx.VertexMotion()) {
        ctx.AddCapability(Capability::PhysicalStorageBufferAddresses);
        ctx.AddExtension("SPV_KHR_physical_storage_buffer");
    }
    const auto shared_type_count = std::popcount(static_cast<u32>(info.shared_types));
    if (shared_type_count > 1 && profile.supports_workgroup_explicit_memory_layout) {
        ctx.AddExtension("SPV_KHR_workgroup_memory_explicit_layout");
        ctx.AddCapability(Capability::WorkgroupMemoryExplicitLayoutKHR);
        ctx.AddCapability(Capability::WorkgroupMemoryExplicitLayout16BitAccessKHR);
    }
    if (info.uses_buffer_int64_atomics || info.uses_shared_int64_atomics) {
        ASSERT_MSG(!info.uses_buffer_int64_atomics || ctx.profile.supports_buffer_int64_atomics,
                   "Shader requires support for atomic Int64 buffer operations that your "
                   "Vulkan instance does not advertise");
        ASSERT_MSG(!info.uses_shared_int64_atomics || ctx.profile.supports_shared_int64_atomics,
                   "Shader requires support for atomic Int64 shared memory operations that "
                   "your Vulkan instance does not advertise");
        ctx.AddCapability(Capability::Int64Atomics);
    }
}

void DefineEntryPoint(const Info& info, EmitContext& ctx, Id main) {
    const std::span interfaces(ctx.interfaces.data(), ctx.interfaces.size());
    const auto& rt = ctx.runtime_info;
    spv::ExecutionModel execution_model{};
    switch (info.sw_stage) {
    case SwStage::Compute: {
        const std::array<u32, 3> size{rt.hw.cs.workgroup_size};
        execution_model = spv::ExecutionModel::GLCompute;
        ctx.AddExecutionMode(main, spv::ExecutionMode::LocalSize, size[0], size[1], size[2]);
        break;
    }
    case SwStage::Vertex:
        execution_model = spv::ExecutionModel::Vertex;
        break;
    case SwStage::TessellationControl:
        execution_model = spv::ExecutionModel::TessellationControl;
        ctx.AddCapability(spv::Capability::Tessellation);
        ctx.AddExecutionMode(main, spv::ExecutionMode::OutputVertices,
                             rt.sw.tcs.NumOutputControlPoints());
        break;
    case SwStage::TessellationEval:
        execution_model = spv::ExecutionModel::TessellationEvaluation;
        ctx.AddExecutionMode(main, ExecutionMode(rt.sw.tes.tess_type));
        ctx.AddExecutionMode(main, ExecutionMode(rt.sw.tes.tess_partitioning));
        ctx.AddExecutionMode(main,
                             rt.sw.tes.tess_topology == ::Gcn::TessellationTopology::TriangleCcw
                                 ? spv::ExecutionMode::VertexOrderCcw
                                 : spv::ExecutionMode::VertexOrderCw);
        break;
    case SwStage::Fragment:
        execution_model = spv::ExecutionModel::Fragment;
        ctx.AddExecutionMode(main, ctx.profile.lower_left_origin_mode
                                       ? spv::ExecutionMode::OriginLowerLeft
                                       : spv::ExecutionMode::OriginUpperLeft);
        if (info.has_discard) {
            ctx.AddCapability(spv::Capability::DemoteToHelperInvocation);
        }
        if (info.stores.GetAny(IR::Attribute::Depth)) {
            ctx.AddExecutionMode(main, spv::ExecutionMode::DepthReplacing);
        }
        break;
    case SwStage::Geometry:
        execution_model = spv::ExecutionModel::Geometry;
        ctx.AddExecutionMode(main, InputPrimitiveMode(rt.hw.gs.in_primitive));
        ctx.AddExecutionMode(main, OutputPrimitiveMode(rt.hw.gs.out_primitive[0]));
        ctx.AddExecutionMode(main, spv::ExecutionMode::OutputVertices, rt.hw.gs.output_vertices);
        ctx.AddExecutionMode(main, spv::ExecutionMode::Invocations, rt.hw.gs.num_invocations);
        break;
    default:
        UNREACHABLE_MSG("Stage {}", u32(info.hw_stage));
    }
    ctx.AddEntryPoint(execution_model, main, "main", interfaces);
}

// Float controls (SPV_KHR_float_controls): the shader's denorm, rounding and inf/nan modes
// where the host supports them per width.

void WarnOnce(std::once_flag& flag, const char* message) {
    std::call_once(flag, [message] { LOG_WARNING(Render_Vulkan, "{}", message); });
}

/// The mode for `width` if supported (with its capability), else a warning once.
void AddFloatMode(EmitContext& ctx, Id main, bool supported, spv::Capability capability,
                  spv::ExecutionMode mode, u32 width, std::once_flag& warned,
                  const char* unsupported) {
    if (supported) {
        ctx.AddCapability(capability);
        ctx.AddExecutionMode(main, mode, width);
    } else {
        WarnOnce(warned, unsupported);
    }
}

void SetupDenormMode(EmitContext& ctx, const Profile& profile, const RuntimeInfo& runtime_info,
                     Id main) {
    using enum ::Gcn::FpDenormMode;
    static std::once_flag f32_flush, mixed, f16_flush, f64_flush, f16_keep, f64_keep;
    const auto mode32 = runtime_info.props.fp_denorm_mode32;
    if (mode32 == InOutFlush) {
        AddFloatMode(ctx, main, profile.support_fp32_denorm_flush,
                     spv::Capability::DenormFlushToZero, spv::ExecutionMode::DenormFlushToZero,
                     32U, f32_flush, "Float32 denorm flushing is not supported by the GPU");
    } else {
        LOG_WARNING(Render_Vulkan, "Unknown FP denorm mode {}", u32(mode32));
    }
    if (!ctx.info.uses_fp16 && !ctx.info.uses_fp64) {
        return;
    }
    const auto mode16_64 = runtime_info.props.fp_denorm_mode16_64;
    if (mode16_64 != mode32 && !profile.supports_denorm_behavior_independence) {
        WarnOnce(mixed, "Float32 and Float16/64 denorm modes are different but the GPU does "
                        "not support that");
        return;
    }
    if (mode16_64 == InOutFlush) {
        AddFloatMode(ctx, main, profile.support_fp16_denorm_flush,
                     spv::Capability::DenormFlushToZero, spv::ExecutionMode::DenormFlushToZero,
                     16U, f16_flush, "Float16 denorm flushing is not supported by the GPU");
        AddFloatMode(ctx, main, profile.support_fp64_denorm_flush,
                     spv::Capability::DenormFlushToZero, spv::ExecutionMode::DenormFlushToZero,
                     64U, f64_flush, "Float64 denorm flushing is not supported by the GPU");
    } else if (mode16_64 == InOutAllow) {
        AddFloatMode(ctx, main, profile.support_fp16_denorm_preserve,
                     spv::Capability::DenormPreserve, spv::ExecutionMode::DenormPreserve, 16U,
                     f16_keep, "Float16 denorm preserving is not supported by the GPU");
        AddFloatMode(ctx, main, profile.support_fp64_denorm_preserve,
                     spv::Capability::DenormPreserve, spv::ExecutionMode::DenormPreserve, 64U,
                     f64_keep, "Float64 denorm preserving is not supported by the GPU");
    } else {
        LOG_WARNING(Render_Vulkan, "Unknown Float16/64 denorm mode {}", u32(mode16_64));
    }
}

void SetupRoundingMode(EmitContext& ctx, const Profile& profile, const RuntimeInfo& runtime_info,
                       Id main) {
    using enum ::Gcn::FpRoundMode;
    static std::once_flag f32_rtz, mixed, f16_rtz, f64_rtz;
    const auto mode32 = runtime_info.props.fp_round_mode32;
    if (mode32 == ToZero) {
        AddFloatMode(ctx, main, profile.support_fp32_round_to_zero,
                     spv::Capability::RoundingModeRTZ, spv::ExecutionMode::RoundingModeRTZ, 32U,
                     f32_rtz, "Float32 rounding to zero is not supported by the GPU");
    } else if (mode32 != NearestEven) {
        LOG_WARNING(Render_Vulkan, "Unimplemented FP rounding mode {}",
                    magic_enum::enum_name(mode32));
    }
    if (!ctx.info.uses_fp16 && !ctx.info.uses_fp64) {
        return;
    }
    const auto mode16_64 = runtime_info.props.fp_round_mode16_64;
    if (mode16_64 != mode32 && !profile.supports_rounding_mode_independence) {
        WarnOnce(mixed, "Float32 and Float16/64 rounding modes are different but the GPU does "
                        "not support that");
        return;
    }
    if (mode16_64 == ToZero) {
        if (ctx.info.uses_fp16) {
            AddFloatMode(ctx, main, profile.support_fp16_round_to_zero,
                         spv::Capability::RoundingModeRTZ, spv::ExecutionMode::RoundingModeRTZ,
                         16U, f16_rtz, "Float16 rounding to zero is not supported by the GPU");
        }
        if (ctx.info.uses_fp64) {
            AddFloatMode(ctx, main, profile.support_fp64_round_to_zero,
                         spv::Capability::RoundingModeRTZ, spv::ExecutionMode::RoundingModeRTZ,
                         64U, f64_rtz, "Float64 rounding to zero is not supported by the GPU");
        }
    } else if (mode16_64 != NearestEven) {
        LOG_WARNING(Render_Vulkan, "Unknown Float16/64 rounding mode {}", u32(mode16_64));
    }
}

void SetupInfNanPreserveMode(EmitContext& ctx, const Profile& profile, Id main) {
    static std::once_flag f32_warned, f16_warned, f64_warned;
    if (profile.support_fp16_signed_zero_inf_nan_preserve ||
        profile.support_fp32_signed_zero_inf_nan_preserve ||
        profile.support_fp64_signed_zero_inf_nan_preserve) {
        ctx.AddCapability(spv::Capability::SignedZeroInfNanPreserve);
    }
    const auto add = [&](bool supported, u32 width, std::once_flag& warned, const char* message) {
        if (supported) {
            ctx.AddExecutionMode(main, spv::ExecutionMode::SignedZeroInfNanPreserve, width);
        } else {
            WarnOnce(warned, message);
        }
    };
    add(profile.support_fp32_signed_zero_inf_nan_preserve, 32U, f32_warned,
        "Float32 signed zero/inf/nan preserve mode is not supported by the GPU");
    if (ctx.info.uses_fp16) {
        add(profile.support_fp16_signed_zero_inf_nan_preserve, 16U, f16_warned,
            "Float16 signed zero/inf/nan preserve mode is not supported by the GPU");
    }
    if (ctx.info.uses_fp64) {
        add(profile.support_fp64_signed_zero_inf_nan_preserve, 64U, f64_warned,
            "Float64 signed zero/inf/nan preserve mode is not supported by the GPU");
    }
}

void SetupFloatMode(EmitContext& ctx, const Profile& profile, const RuntimeInfo& runtime_info,
                    Id main) {
    ctx.AddExtension("SPV_KHR_float_controls");
    SetupDenormMode(ctx, profile, runtime_info, main);
    SetupRoundingMode(ctx, profile, runtime_info, main);
    SetupInfNanPreserveMode(ctx, profile, main);
}

} // namespace

std::vector<u32> EmitSPIRV(const Profile& profile, const RuntimeInfo& runtime_info,
                           const IR::Program& program, Bindings& binding) {
    EmitContext ctx{profile, runtime_info, program.info, binding};
    const Id main{DefineMain(ctx, program)};
    DefineEntryPoint(program.info, ctx, main);
    SetupCapabilities(program.info, profile, runtime_info, ctx);
    SetupFloatMode(ctx, profile, runtime_info, main);
    PatchPhiNodes(program, ctx);
    binding.user_data += program.info.ud_mask.NumRegs();
    return ctx.Assemble();
}

// Instructions of the driver itself.

Id EmitPhi(EmitContext& ctx, IR::Inst* inst) {
    const size_t num_args{inst->NumArgs()};
    boost::container::small_vector<Id, 32> blocks;
    blocks.reserve(num_args);
    for (size_t index = 0; index < num_args; ++index) {
        blocks.push_back(inst->PhiBlock(index)->Definition<Id>());
    }
    const Id result_type{TypeId(ctx, inst->Flags<IR::Type>())};
    return ctx.DeferredOpPhi(result_type, std::span(blocks.data(), blocks.size()));
}

void EmitVoid(EmitContext&) {}

Id EmitConditionRef(EmitContext& ctx, const IR::Value& value) {
    const Id id{ctx.Def(value)};
    ASSERT_MSG(Sirit::ValidId(id), "Forward identity declaration");
    return id;
}

Id EmitGetPcLo(EmitContext& ctx, const IR::Value& value) {
    const Id id{ctx.Def(value)};
    ASSERT_MSG(Sirit::ValidId(id), "Forward identity declaration");
    return id;
}

// Register accesses are gone after SSA construction.

#define UNREACHABLE_INSTRUCTION(name)                                                              \
    void Emit##name(EmitContext&) {                                                                \
        UNREACHABLE_MSG("Unreachable instruction");                                                \
    }

UNREACHABLE_INSTRUCTION(GetScc)
UNREACHABLE_INSTRUCTION(GetExec)
UNREACHABLE_INSTRUCTION(GetVcc)
UNREACHABLE_INSTRUCTION(GetVccLo)
UNREACHABLE_INSTRUCTION(GetVccHi)
UNREACHABLE_INSTRUCTION(GetM0)
UNREACHABLE_INSTRUCTION(SetScc)
UNREACHABLE_INSTRUCTION(SetExec)
UNREACHABLE_INSTRUCTION(SetVcc)
UNREACHABLE_INSTRUCTION(SetVccLo)
UNREACHABLE_INSTRUCTION(SetVccHi)
UNREACHABLE_INSTRUCTION(SetM0)

#undef UNREACHABLE_INSTRUCTION

} // namespace Shader::Backend::SPIRV
