// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <sirit/sirit.h>

#include "shader/backend/spirv/emit_spirv_quad_rect.h"
#include "shader/runtime_info.h"

namespace Shader::Backend::SPIRV {

namespace {

using Sirit::Id;

constexpr u32 SpirvVersion1_5 = 0x00010500;

class AuxTessShader : public Sirit::Module {
public:
    explicit AuxTessShader(const HwFragmentRuntimeInfo& fs_info_)
        : Sirit::Module{SpirvVersion1_5}, fs_info{fs_info_} {
        void_id = TypeVoid();
        bool_id = TypeBool();
        float_id = TypeFloat(32);
        uint_id = TypeUInt(32U);
        int_id = TypeInt(32U, true);
        bvec2_id = TypeVector(bool_id, 2);
        vec2_id = TypeVector(float_id, 2);
        vec3_id = TypeVector(float_id, 3);
        vec4_id = TypeVector(float_id, 4);
        float_one = Constant(float_id, 1.0f);
        float_min_one = Constant(float_id, -1.0f);
        int_zero = Constant(int_id, 0);

        // gl_PerVertex
        const Id float_arr{TypeArray(float_id, Constant(uint_id, 1U))};
        per_vertex_type = TypeStruct(vec4_id, float_id, float_arr, float_arr);
        Decorate(per_vertex_type, spv::Decoration::Block);
        const std::array builtins{spv::BuiltIn::Position, spv::BuiltIn::PointSize,
                                  spv::BuiltIn::ClipDistance, spv::BuiltIn::CullDistance};
        for (u32 member = 0; member < builtins.size(); ++member) {
            MemberDecorate(per_vertex_type, member, spv::Decoration::BuiltIn,
                           static_cast<u32>(builtins[member]));
        }
    }

    /// A rect list draws rects as three vertices: the patch gets the fourth, opposite the
    /// vertex at the right angle (the one whose neighbours share its x or y), as v0 * b0 +
    /// v1 * b1 + v2 * b2 with that vertex's weight -1, the others' 1.
    void EmitRectListTCS() {
        DefineEntry(spv::ExecutionModel::TessellationControl);
        StoreUnitTessFactors();
        const Id input_vec4{TypePointer(spv::StorageClass::Input, vec4_id)};
        const Id output_vec4{TypePointer(spv::StorageClass::Output, vec4_id)};

        std::array<Id, 3> pos;
        for (int i = 0; i < 3; i++) {
            pos[i] = OpLoad(vec4_id, OpAccessChain(input_vec4, gl_in, Int(i), int_zero));
        }
        // equal(pos[i].xy, pos[i + 1].xy)
        std::array<Id, 3> xy_equal_next;
        for (int i = 0; i < 3; i++) {
            const Id left{OpVectorShuffle(vec2_id, pos[i], pos[i], 0, 1)};
            const Id right{OpVectorShuffle(vec2_id, pos[(i + 1) % 3], pos[(i + 1) % 3], 0, 1)};
            xy_equal_next[i] = OpFOrdEqual(bvec2_id, left, right);
        }
        std::array<Id, 3> weight;
        std::array<Id, 3> is_corner;
        for (int i = 0; i < 3; i++) {
            const Id prev = xy_equal_next[(i + 2) % 3];
            const Id xy_equal{OpLogicalAnd(bool_id, OpCompositeExtract(bool_id, xy_equal_next[i], 0),
                                           OpCompositeExtract(bool_id, prev, 1))};
            const Id yx_equal{OpLogicalAnd(bool_id, OpCompositeExtract(bool_id, xy_equal_next[i], 1),
                                           OpCompositeExtract(bool_id, prev, 0))};
            is_corner[i] = OpLogicalOr(bool_id, xy_equal, yx_equal);
            weight[i] = OpSelect(float_id, is_corner[i], float_min_one, float_one);
        }
        const auto interpolate = [&](Id v0, Id v1, Id v2) {
            const Id p0{OpVectorTimesScalar(vec4_id, v0, weight[0])};
            const Id p1{OpVectorTimesScalar(vec4_id, v1, weight[1])};
            const Id p2{OpVectorTimesScalar(vec4_id, v2, weight[2])};
            return OpFAdd(vec4_id, p0, OpFAdd(vec4_id, p1, p2));
        };
        // Invocations 0-2 take the vertices from the corner on, invocation 3 the new one.
        Id first{OpSelect(int_id, is_corner[2], Int(2), Int(0))};
        first = OpSelect(int_id, is_corner[1], Int(1), first);
        const Id invocation_id{OpLoad(int_id, gl_invocation_id)};
        const Id is_fourth{OpIEqual(bool_id, invocation_id, Int(3))};
        const Id index{OpSMod(int_id, OpIAdd(int_id, first, invocation_id), Int(3))};

        const Id pos3{interpolate(pos[0], pos[1], pos[2])};
        const Id in_ptr{OpAccessChain(input_vec4, gl_in, index, Int(0))};
        const Id position{OpSelect(vec4_id, is_fourth, pos3, OpLoad(vec4_id, in_ptr))};
        OpStore(OpAccessChain(output_vec4, gl_out, invocation_id, Int(0)), position);

        for (size_t i = 0; i < inputs.size(); i++) {
            const Id v0{OpLoad(vec4_id, OpAccessChain(input_vec4, inputs[i], Int(0)))};
            const Id v1{OpLoad(vec4_id, OpAccessChain(input_vec4, inputs[i], Int(1)))};
            const Id v2{OpLoad(vec4_id, OpAccessChain(input_vec4, inputs[i], Int(2)))};
            const Id fourth{interpolate(v0, v1, v2)};
            const Id param{OpLoad(vec4_id, OpAccessChain(input_vec4, inputs[i], index))};
            const Id value{OpSelect(vec4_id, is_fourth, fourth, param)};
            OpStore(OpAccessChain(output_vec4, outputs[i], invocation_id), value);
        }
        OpReturn();
        OpFunctionEnd();
    }

    /// A quad list's four vertices as control points, reordered (1, 2, 0, 3) for the quad
    /// domain's corner order.
    void EmitQuadListTCS() {
        DefineEntry(spv::ExecutionModel::TessellationControl);
        const Id array_type{TypeArray(int_id, Int(4))};
        const Id order{ConstantComposite(array_type, Int(1), Int(2), Int(0), Int(3))};
        const Id indices{AddLocalVariable(TypePointer(spv::StorageClass::Function, array_type),
                                          spv::StorageClass::Function, order)};
        StoreUnitTessFactors();
        const Id input_vec4{TypePointer(spv::StorageClass::Input, vec4_id)};
        const Id output_vec4{TypePointer(spv::StorageClass::Output, vec4_id)};
        const Id func_int{TypePointer(spv::StorageClass::Function, int_id)};
        const Id invocation_id{OpLoad(int_id, gl_invocation_id)};
        const Id index{OpLoad(int_id, OpAccessChain(func_int, indices, invocation_id))};

        const Id in_position{OpLoad(vec4_id, OpAccessChain(input_vec4, gl_in, index, Int(0)))};
        OpStore(OpAccessChain(output_vec4, gl_out, invocation_id, Int(0)), in_position);
        for (size_t i = 0; i < inputs.size(); i++) {
            const Id param{OpLoad(vec4_id, OpAccessChain(input_vec4, inputs[i], index))};
            OpStore(OpAccessChain(output_vec4, outputs[i], invocation_id), param);
        }
        OpReturn();
        OpFunctionEnd();
    }

    /// Each tessellated vertex is control point (int(v) * 2 + int(u)).
    void EmitPassthroughTES() {
        DefineEntry(spv::ExecutionModel::TessellationEvaluation);
        const Id input_float{TypePointer(spv::StorageClass::Input, float_id)};
        const Id u{OpLoad(float_id, OpAccessChain(input_float, gl_tess_coord, Int(0)))};
        const Id v{OpLoad(float_id, OpAccessChain(input_float, gl_tess_coord, Int(1)))};
        const Id index{OpIAdd(int_id, OpIMul(int_id, OpConvertFToS(int_id, v), Int(2)),
                              OpConvertFToS(int_id, u))};
        const Id input_vec4{TypePointer(spv::StorageClass::Input, vec4_id)};
        const Id output_vec4{TypePointer(spv::StorageClass::Output, vec4_id)};
        const Id position{OpLoad(vec4_id, OpAccessChain(input_vec4, gl_in, index, Int(0)))};
        OpStore(OpAccessChain(output_vec4, gl_per_vertex, Int(0)), position);
        for (size_t i = 0; i < inputs.size(); i++) {
            const Id param{OpLoad(vec4_id, OpAccessChain(input_vec4, inputs[i], index))};
            OpStore(outputs[i], param);
        }
        OpReturn();
        OpFunctionEnd();
    }

private:
    Id Int(s32 value) {
        return Constant(int_id, value);
    }

    /// Outer levels 1 (four), inner 1 (two): one patch, no subdivision.
    void StoreUnitTessFactors() {
        const Id output_float{TypePointer(spv::StorageClass::Output, float_id)};
        for (int i = 0; i < 4; i++) {
            OpStore(OpAccessChain(output_float, gl_tess_level_outer, Int(i)), float_one);
        }
        for (int i = 0; i < 2; i++) {
            OpStore(OpAccessChain(output_float, gl_tess_level_inner, Int(i)), float_one);
        }
    }

    Id AddInterface(spv::StorageClass storage, Id type) {
        const Id variable{AddGlobalVariable(TypePointer(storage, type), storage)};
        interfaces.push_back(variable);
        return variable;
    }

    void DefineEntry(spv::ExecutionModel model) {
        AddCapability(spv::Capability::Shader);
        AddCapability(spv::Capability::Tessellation);
        const Id void_function{TypeFunction(void_id)};
        main = OpFunction(void_id, spv::FunctionControlMask::MaskNone, void_function);
        if (model == spv::ExecutionModel::TessellationControl) {
            AddExecutionMode(main, spv::ExecutionMode::OutputVertices, 4U);
        } else {
            AddExecutionMode(main, spv::ExecutionMode::Quads);
            AddExecutionMode(main, spv::ExecutionMode::SpacingEqual);
            AddExecutionMode(main, spv::ExecutionMode::VertexOrderCw);
        }
        DefineInputs(model);
        DefineOutputs(model);
        AddEntryPoint(model, main, "main", interfaces);
        AddLabel(OpLabel());
    }

    void DefineInputs(spv::ExecutionModel model) {
        if (model == spv::ExecutionModel::TessellationEvaluation) {
            gl_tess_coord = AddInterface(spv::StorageClass::Input, vec3_id);
            Decorate(gl_tess_coord, spv::Decoration::BuiltIn, spv::BuiltIn::TessCoord);
        } else {
            gl_invocation_id = AddInterface(spv::StorageClass::Input, int_id);
            Decorate(gl_invocation_id, spv::Decoration::BuiltIn, spv::BuiltIn::InvocationId);
        }
        const Id per_vertex_array{TypeArray(per_vertex_type, Constant(uint_id, 32U))};
        gl_in = AddInterface(spv::StorageClass::Input, per_vertex_array);
        const Id param_array{TypeArray(vec4_id, Int(32))};
        inputs.reserve(fs_info.num_inputs);
        for (u32 i = 0; i < fs_info.num_inputs; i++) {
            const auto& input = fs_info.inputs[i];
            if (input.IsDefault()) {
                continue;
            }
            inputs.emplace_back(AddInterface(spv::StorageClass::Input, param_array));
            Decorate(inputs.back(), spv::Decoration::Location, input.param_index);
        }
    }

    void DefineOutputs(spv::ExecutionModel model) {
        const bool is_tcs = model == spv::ExecutionModel::TessellationControl;
        if (is_tcs) {
            const Id per_vertex_array{TypeArray(per_vertex_type, Constant(uint_id, 4U))};
            gl_out = AddInterface(spv::StorageClass::Output, per_vertex_array);
            const Id arr2_id{TypeArray(float_id, Constant(uint_id, 2U))};
            gl_tess_level_inner = AddInterface(spv::StorageClass::Output, arr2_id);
            Decorate(gl_tess_level_inner, spv::Decoration::BuiltIn, spv::BuiltIn::TessLevelInner);
            Decorate(gl_tess_level_inner, spv::Decoration::Patch);
            const Id arr4_id{TypeArray(float_id, Constant(uint_id, 4U))};
            gl_tess_level_outer = AddInterface(spv::StorageClass::Output, arr4_id);
            Decorate(gl_tess_level_outer, spv::Decoration::BuiltIn, spv::BuiltIn::TessLevelOuter);
            Decorate(gl_tess_level_outer, spv::Decoration::Patch);
        } else {
            gl_per_vertex = AddInterface(spv::StorageClass::Output, per_vertex_type);
        }
        outputs.reserve(fs_info.num_inputs);
        for (u32 i = 0; i < fs_info.num_inputs; i++) {
            const auto& input = fs_info.inputs[i];
            if (input.IsDefault()) {
                continue;
            }
            outputs.emplace_back(AddInterface(spv::StorageClass::Output,
                                              is_tcs ? TypeArray(vec4_id, Int(4)) : vec4_id));
            Decorate(outputs.back(), spv::Decoration::Location, input.param_index);
        }
    }

    HwFragmentRuntimeInfo fs_info;
    Id main;
    Id void_id;
    Id bool_id;
    Id float_id;
    Id uint_id;
    Id int_id;
    Id bvec2_id;
    Id vec2_id;
    Id vec3_id;
    Id vec4_id;
    Id float_one;
    Id float_min_one;
    Id int_zero;
    Id per_vertex_type;
    Id gl_in;
    Id gl_out;
    Id gl_per_vertex;
    Id gl_tess_level_inner;
    Id gl_tess_level_outer;
    Id gl_tess_coord;
    Id gl_invocation_id;
    std::vector<Id> inputs;
    std::vector<Id> outputs;
    std::vector<Id> interfaces;
};

} // namespace

std::vector<u32> EmitAuxilaryTessShader(AuxShaderType type, const HwFragmentRuntimeInfo& fs_info) {
    AuxTessShader shader{fs_info};
    switch (type) {
    case AuxShaderType::RectListTCS:
        shader.EmitRectListTCS();
        break;
    case AuxShaderType::QuadListTCS:
        shader.EmitQuadListTCS();
        break;
    case AuxShaderType::PassthroughTES:
        shader.EmitPassthroughTES();
        break;
    }
    return shader.Assemble();
}

} // namespace Shader::Backend::SPIRV
