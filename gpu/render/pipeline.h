// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Vulkan pipelines for translated shaders. A pipeline holds its stages' Info
// (resources in binding order), a descriptor set layout (push descriptors when they fit) and a
// pipeline layout with the PushData block. Graphics pipelines are keyed by the stages and the
// fixed-function state that cannot be dynamic; compute pipelines by their shader permutation.
#pragma once

#include <array>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <boost/container/static_vector.hpp>
#include <xxhash.h>

#include "gcn/regs_raster.h"
#include "gcn/regs_targets.h"
#include "shader/frontend/fetch_shader.h"
#include "shader/profile.h"
#include "shader/runtime_info.h"
#include "vulkan/vk_common.h"

namespace Shader {
struct Info;
struct PushData;
} // namespace Shader

namespace Serialization {
struct Archive;
}

namespace Vulkan {

class Instance;
class Scheduler;
class DescriptorHeap;

static constexpr auto AllGraphicsStageBits =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eTessellationControl |
    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eGeometry |
    vk::ShaderStageFlagBits::eFragment;

static constexpr u32 MaxShaderStages = static_cast<u32>(Shader::SwStage::NumLogicalStages);
static constexpr u32 MaxVertexBufferCount = 32;

template <typename T>
using VertexInputs = boost::container::static_vector<T, MaxVertexBufferCount>;

class Pipeline {
public:
    Pipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
             const Shader::Profile& profile, bool is_compute);
    virtual ~Pipeline();

    vk::Pipeline Handle() const noexcept {
        return *pipeline;
    }

    vk::PipelineLayout GetLayout() const noexcept {
        return *pipeline_layout;
    }

    /// The stages present (the compute slot, or the graphics ones), null where absent.
    auto GetStages() const {
        static_assert(static_cast<u32>(Shader::SwStage::Compute) == Shader::MaxStageTypes - 1);
        if (is_compute) {
            return std::span{stages.cend() - 1, stages.cend()};
        }
        return std::span{stages.cbegin(), stages.cend() - 1};
    }

    const Shader::Info& GetStage(Shader::SwStage stage) const noexcept {
        return *stages[u32(stage)];
    }

    bool IsCompute() const {
        return is_compute;
    }

    using DescriptorWrites = std::vector<vk::WriteDescriptorSet>;
    /// Records the push data and the descriptors. `image_infos` and `buffer_infos`: the arrays
    /// the writes point into, if known (their descriptors are then copied for the recording
    /// thread in one piece each).
    void BindResources(DescriptorWrites& set_writes, const Shader::PushData& push_data,
                       std::span<const vk::DescriptorImageInfo> image_infos = {},
                       std::span<const vk::DescriptorBufferInfo> buffer_infos = {}) const;

protected:
    /// The descriptor set layout of the stages' resources (buffers, images, samplers per stage,
    /// in the bindings the backend assigned) and the pipeline layout.
    void CreateLayouts(vk::ShaderStageFlags push_stages, const char* kind,
                       const std::string& debug_name);

    /// The stages' shader names, for debug object names.
    [[nodiscard]] std::string GetDebugString() const;

    const Instance& instance;
    Scheduler& scheduler;
    DescriptorHeap& desc_heap;
    const Shader::Profile& profile;
    vk::UniquePipeline pipeline;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniqueDescriptorSetLayout desc_layout;
    std::array<const Shader::Info*, Shader::MaxStageTypes> stages{};
    bool uses_push_descriptors{};
    bool is_compute;
};

struct ComputePipelineKey {
    size_t value;

    friend bool operator==(const ComputePipelineKey&, const ComputePipelineKey&) = default;

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

class ComputePipeline : public Pipeline {
public:
    /// (Nothing beyond the key and shader is needed to recreate it.)
    struct SerializationSupport {
        u32 dummy{};

        void Serialize(Serialization::Archive& ar) const;
        bool Deserialize(Serialization::Archive& ar);
    };

    /// `preloading`: recreated from the pipeline cache (a driver rejection is corrupt data).
    ComputePipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
                    const Shader::Profile& profile, vk::PipelineCache pipeline_cache,
                    ComputePipelineKey compute_key, const Shader::Info& info,
                    vk::ShaderModule module, SerializationSupport& sdata, bool preloading);
    ~ComputePipeline() override;

private:
    ComputePipelineKey compute_key;
};

/// The fixed-function state baked into a graphics pipeline (compared bytewise).
struct GraphicsPipelineKey {
    std::array<size_t, MaxShaderStages> stage_hashes;
    std::array<vk::Format, MaxVertexBufferCount> vertex_buffer_formats;
    u32 patch_control_points;
    u32 num_color_attachments;
    std::array<Shader::PsColorBuffer, Gcn::NUM_COLOR_BUFFERS> color_buffers;
    std::array<Gcn::BlendControl, Gcn::NUM_COLOR_BUFFERS> blend_controls;
    std::array<vk::ColorComponentFlags, Gcn::NUM_COLOR_BUFFERS> write_masks;
    Gcn::ColorBufferMask cb_shader_mask;
    Gcn::ColorControl::LogicOp logic_op;
    u8 num_samples;
    u8 depth_samples;
    std::array<u8, Gcn::NUM_COLOR_BUFFERS> color_samples;
    u32 mrt_mask;
    u32 motion_vectors = 0;
    struct {
        Gcn::DepthBuffer::ZFormat z_format : 2;
        Gcn::DepthBuffer::StencilFormat stencil_format : 1;
        u32 depth_clamp_enable : 1;
    };
    struct {
        Gcn::PrimitiveType prim_type : 5;
        Gcn::PolygonMode polygon_mode : 2;
        Gcn::ClipSpace clip_space : 1;
        Gcn::ProvokingVtxLast provoking_vtx_last : 1;
        u32 depth_clip_enable : 1;
    };

    GraphicsPipelineKey() {
        std::memset(this, 0, sizeof(*this));
    }

    bool operator==(const GraphicsPipelineKey& key) const noexcept {
        return std::memcmp(this, &key, sizeof(key)) == 0;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

class GraphicsPipeline : public Pipeline {
public:
    /// What recreating the pipeline from the cache needs besides the key and modules: the
    /// vertex input (without dynamic vertex input), multisampling, and the generated auxiliary
    /// shaders (rect/quad tessellation, clip distance discard).
    struct SerializationSupport {
        VertexInputs<vk::VertexInputAttributeDescription> vertex_attributes{};
        VertexInputs<vk::VertexInputBindingDescription> vertex_bindings{};
        VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors{};
        vk::PipelineMultisampleStateCreateInfo multisampling{};
        std::vector<u32> tcs{};
        std::vector<u32> tes{};
        std::vector<u32> fragment{};

        void Serialize(Serialization::Archive& ar) const;
        bool Deserialize(Serialization::Archive& ar);
    };

    GraphicsPipeline(const Instance& instance, Scheduler& scheduler, DescriptorHeap& desc_heap,
                     const Shader::Profile& profile, const GraphicsPipelineKey& key,
                     vk::PipelineCache pipeline_cache,
                     std::span<const Shader::Info*, MaxShaderStages> stages,
                     std::span<const Shader::RuntimeInfo, MaxShaderStages> runtime_infos,
                     std::optional<const Shader::Gcn::FetchShaderData> fetch_shader,
                     std::span<const vk::ShaderModule> modules, SerializationSupport& sdata,
                     bool preloading);
    ~GraphicsPipeline() override;

    const std::optional<const Shader::Gcn::FetchShaderData>& GetFetchShader() const noexcept {
        return fetch_shader;
    }

    const GraphicsPipelineKey& GetGraphicsKey() const {
        return key;
    }

    /// The vertex input attributes and bindings of the fetch shader's attributes. `sharps`,
    /// when given, are their V#s as the GPU command thread read them (in the draw packet): the
    /// tables they come from may have been rewritten by a later constant engine dump by the
    /// time the recording thread gets here.
    template <typename Attribute, typename Binding>
    void GetVertexInputs(VertexInputs<Attribute>& attributes, VertexInputs<Binding>& bindings,
                         VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT>& divisors,
                         VertexInputs<Gcn::Buffer>& guest_buffers, u32 step_rate_0,
                         u32 step_rate_1, std::span<const Gcn::Buffer> sharps = {}) const;

private:
    GraphicsPipelineKey key;
    std::optional<const Shader::Gcn::FetchShaderData> fetch_shader{};
};

struct ClipDistanceShaderKey {
    std::array<std::tuple<u8, u8>, 8> clip_locations;

    bool operator==(const ClipDistanceShaderKey& key) const noexcept {
        return std::memcmp(this, &key, sizeof(key)) == 0;
    }
};

} // namespace Vulkan

template <>
struct std::hash<Vulkan::ComputePipelineKey> {
    std::size_t operator()(const Vulkan::ComputePipelineKey& key) const noexcept {
        return std::hash<size_t>{}(key.value);
    }
};

template <>
struct std::hash<Vulkan::GraphicsPipelineKey> {
    std::size_t operator()(const Vulkan::GraphicsPipelineKey& key) const noexcept {
        return XXH3_64bits(&key, sizeof(key));
    }
};

template <>
struct std::hash<Vulkan::ClipDistanceShaderKey> {
    std::size_t operator()(const Vulkan::ClipDistanceShaderKey& key) const noexcept {
        return XXH3_64bits(&key, sizeof(key));
    }
};
