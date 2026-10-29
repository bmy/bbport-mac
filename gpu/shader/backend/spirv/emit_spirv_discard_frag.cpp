// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include <map>
#include <set>
#include <tuple>

#include <fmt/format.h>
#include <sirit/sirit.h>

#include "common/assert.h"
#include "shader/backend/spirv/emit_spirv_discard_frag.h"

namespace Shader::Backend::SPIRV {

namespace {

using Sirit::Id;

constexpr u32 SpirvVersion1_5 = 0x00010500;

bool IsClipDistance(Output output) {
    return output >= Output::ClipDist0 && output <= Output::ClipDist7;
}

class DiscardShader : public Sirit::Module {
public:
    explicit DiscardShader(const std::array<Shader::OutputMap, 3>& outputs)
        : Sirit::Module{SpirvVersion1_5} {
        void_id = TypeVoid();
        bool_id = TypeBool();
        float_id = TypeFloat(32);
        vec4_id = TypeVector(float_id, 4);
        float_zero = Constant(float_id, 0.0f);
        for (u8 location = 0; location < 3; ++location) {
            for (u8 elem = 0; elem < 4; ++elem) {
                if (IsClipDistance(outputs.at(location).at(elem))) {
                    clip_components.emplace_back(location, elem);
                }
            }
        }
    }

    /// Reads the vec4 inputs holding clip distances; demotes the invocation for each negative
    /// one.
    void Emit() {
        std::set<u8> locations;
        for (const auto& [location, elem] : clip_components) {
            locations.insert(location);
        }
        ASSERT_MSG(locations.size() <= 2, "more than 2 clip planes locations");

        AddCapability(spv::Capability::Shader);
        AddExtension("SPV_EXT_demote_to_helper_invocation");
        AddCapability(spv::Capability::DemoteToHelperInvocation);
        SetMemoryModel(spv::AddressingModel::Logical, spv::MemoryModel::GLSL450);
        const Id void_function{TypeFunction(void_id)};
        const Id main = OpFunction(void_id, spv::FunctionControlMask::MaskNone, void_function);

        const Id input_vec4 = TypePointer(spv::StorageClass::Input, vec4_id);
        const std::vector<u8> location_list(locations.begin(), locations.end());
        std::array<Id, 2> inputs{};
        for (size_t i = 0; i < location_list.size(); ++i) {
            inputs[i] = AddGlobalVariable(input_vec4, spv::StorageClass::Input);
            Decorate(inputs[i], spv::Decoration::Location, location_list[i]);
            Name(inputs[i], fmt::format("ccdist{}_in", i));
        }
        AddEntryPoint(spv::ExecutionModel::Fragment, main, "main",
                      std::span(inputs).first(location_list.size()));
        AddExecutionMode(main, spv::ExecutionMode::OriginUpperLeft);
        AddLabel(OpLabel());

        std::map<u8, Id> values;
        for (size_t i = 0; i < location_list.size(); ++i) {
            const Id value = OpLoad(vec4_id, inputs[i]);
            values[location_list[i]] = value;
            Name(value, fmt::format("ccdist{}", i));
        }
        for (const auto& [location, elem] : clip_components) {
            const Id plane = OpCompositeExtract(float_id, values[location], elem);
            Name(plane, fmt::format("plane{}", location * 4 + elem));
            const Id clipped = OpFOrdLessThan(bool_id, plane, float_zero);
            const Id kill_label{OpLabel()};
            const Id merge_label{OpLabel()};
            OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
            OpBranchConditional(clipped, kill_label, merge_label);
            AddLabel(kill_label);
            OpDemoteToHelperInvocationEXT();
            OpBranch(merge_label);
            AddLabel(merge_label);
        }
        OpReturn();
        OpFunctionEnd();
    }

private:
    Id void_id;
    Id bool_id;
    Id float_id;
    Id vec4_id;
    Id float_zero;
    std::vector<std::tuple<u8, u8>> clip_components;
};

} // namespace

std::vector<u32> EmitDiscardFragmentShader(std::array<Shader::OutputMap, 3> outputs) {
    DiscardShader shader{outputs};
    shader.Emit();
    return shader.Assemble();
}

} // namespace Shader::Backend::SPIRV
