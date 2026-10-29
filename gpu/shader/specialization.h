// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: what a compiled shader depends on beyond its program and runtime
// info: the bound resources' types and formats, vertex input types and divisors. The key of
// shader permutations (a module is recompiled when it differs).
#pragma once

#include <bitset>
#include <optional>

#include <boost/container/small_vector.hpp>

#include "common/types.h"
#include "shader/backend/bindings.h"
#include "shader/frontend/fetch_shader.h"
#include "shader/info.h"
#include "shader/profile.h"

namespace Shader {

struct VsAttribSpecialization {
    u32 divisor{};
    ::Gcn::NumberClass num_class{};
    ::Gcn::CompMapping dst_select{};

    bool operator==(const VsAttribSpecialization&) const = default;
};

struct BufferSpecialization {
    u32 stride : 14;
    u32 is_formatted : 1;
    u32 swizzle_enable : 1;
    u32 data_format : 6;
    u32 num_format : 4;
    u32 index_stride : 2;
    u32 element_size : 2;
    u32 paged : 1; ///< BB_LAYER_MEMORY: through the page table (IsPagedBuffer)
    ::Gcn::CompMapping dst_select{};
    ::Gcn::NumberConversion num_conversion{};

    bool operator==(const BufferSpecialization& other) const {
        return stride == other.stride && is_formatted == other.is_formatted &&
               paged == other.paged &&
               swizzle_enable == other.swizzle_enable &&
               (!is_formatted ||
                (data_format == other.data_format && num_format == other.num_format &&
                 dst_select == other.dst_select && num_conversion == other.num_conversion)) &&
               (!swizzle_enable ||
                (index_stride == other.index_stride && element_size == other.element_size));
    }
};

struct ImageSpecialization {
    ::Gcn::ImageType type = ::Gcn::ImageType::Color2D;
    bool is_integer = false;
    bool is_storage = false;
    bool is_cube = false;
    bool is_srgb = false;
    ::Gcn::CompMapping dst_select{};
    ::Gcn::NumberConversion num_conversion{};
    u32 num_bindings = 0; ///< views bound (mip fallback arrays)

    bool operator==(const ImageSpecialization&) const = default;
};

struct FMaskSpecialization {
    u32 width;
    u32 height;

    bool operator==(const FMaskSpecialization&) const = default;
};

struct SamplerSpecialization {
    u8 force_unnormalized : 1;
    u8 force_degamma : 1;

    bool operator==(const SamplerSpecialization&) const = default;
};

/// The specialization of one stage. Separate from the runtime info: the resource layout is
/// known only once the module has been compiled.
struct StageSpecialization {
    static constexpr size_t MaxStageResources = 128;

    const Info* info{};
    RuntimeInfo runtime_info{};
    std::bitset<MaxStageResources> bitset{};
    std::optional<Gcn::FetchShaderData> fetch_shader_data{};
    boost::container::small_vector<VsAttribSpecialization, 32> vs_attribs;
    boost::container::small_vector<BufferSpecialization, 16> buffers;
    boost::container::small_vector<ImageSpecialization, 16> images;
    boost::container::small_vector<FMaskSpecialization, 8> fmasks;
    boost::container::small_vector<SamplerSpecialization, 16> samplers;
    Backend::Bindings start{};

    StageSpecialization() = default;
    StageSpecialization(const Info& info_, RuntimeInfo runtime_info_, const Profile& profile_,
                        Backend::Bindings start_)
        : info{&info_}, runtime_info{runtime_info_}, start{start_} {
        fetch_shader_data = Gcn::ParseFetchShader(info_);
        if (info_.sw_stage == SwStage::Vertex && fetch_shader_data) {
            // Vertex input number types (Vulkan requires matching shader types).
            ForEachSharp(vs_attribs, fetch_shader_data->attributes,
                         [this](auto& spec, const auto& desc, ::Gcn::Buffer sharp) {
                             FillVsAttrib(spec, desc, sharp, runtime_info);
                         });
        }
        u32 binding{};
        ForEachSharp(binding, buffers, info->buffers,
                     [&profile_, this](auto& spec, const auto& desc, ::Gcn::Buffer sharp) {
                         FillBuffer(spec, desc, sharp, profile_, info->pgm_hash);
                     });
        ForEachSharp(binding, images, info->images,
                     [&](auto& spec, const auto& desc, ::Gcn::Image sharp) {
                         FillImage(spec, desc, sharp, *info);
                     });
        ForEachSharp(binding, fmasks, info->fmasks,
                     [](auto& spec, const auto& desc, ::Gcn::Image sharp) {
                         FillFmask(spec, sharp);
                     });
        ForEachSharp(samplers, info->samplers,
                     [](auto& spec, const auto& desc, ::Gcn::Sampler sharp) {
                         FillSampler(spec, sharp);
                     });

        // Runtime info found by the tessellation passes.
        if (info->sw_stage == SwStage::TessellationControl ||
            info->sw_stage == SwStage::TessellationEval) {
            TessellationDataConstantBuffer tess_constants{};
            info->ReadTessConstantBuffer(tess_constants);
            runtime_info.InitFromTessConstants(tess_constants);
        }
    }

    // bbport: each resource's part of the specialization (the constructor and Matches).
    static void FillVsAttrib(VsAttribSpecialization& spec, const auto& desc, ::Gcn::Buffer sharp,
                             const RuntimeInfo& rt) {
        using InstanceIdType = Shader::Gcn::VertexAttribute::InstanceIdType;
        if (const auto step_rate = desc.GetStepRate(); step_rate != InstanceIdType::None) {
            spec.divisor = step_rate == InstanceIdType::OverStepRate0
                               ? rt.sw.vs.step_rate_0
                               : (step_rate == InstanceIdType::OverStepRate1 ? rt.sw.vs.step_rate_1
                                                                              : 1);
        }
        spec.num_class = ::Gcn::GetNumberClass(sharp.GetNumberFmt());
        spec.dst_select = sharp.DstSelect();
    }
    static void FillBuffer(BufferSpecialization& spec, const auto& desc, ::Gcn::Buffer sharp,
                           const Profile& profile, u64 pgm_hash) {
        spec.stride = sharp.GetStride();
        spec.paged = IsPagedBuffer(profile.paged_buffers, sharp, desc.is_formatted, pgm_hash);
        spec.is_formatted = desc.is_formatted;
        spec.swizzle_enable = sharp.swizzle_enable;
        if (spec.is_formatted) {
            spec.data_format = static_cast<u32>(sharp.GetDataFmt());
            spec.num_format = static_cast<u32>(sharp.GetNumberFmt());
            spec.dst_select = sharp.DstSelect();
            spec.num_conversion = sharp.GetNumberConversion();
        }
        if (spec.swizzle_enable) {
            spec.index_stride = sharp.index_stride;
            spec.element_size = sharp.element_size;
        }
    }
    static void FillImage(ImageSpecialization& spec, const auto& desc, ::Gcn::Image sharp,
                          const Info& info_) {
        spec.type = sharp.GetViewType(desc.is_array);
        spec.is_integer = ::Gcn::IsInteger(sharp.GetNumberFmt());
        spec.is_storage = desc.is_written;
        spec.is_cube = sharp.IsCube();
        if (spec.is_storage) {
            spec.dst_select = sharp.DstSelect();
        } else {
            spec.is_srgb = sharp.GetNumberFmt() == ::Gcn::NumberFormat::Srgb;
        }
        spec.num_conversion = sharp.GetNumberConversion();
        spec.num_bindings = desc.NumBindings(info_);
    }
    static void FillFmask(FMaskSpecialization& spec, ::Gcn::Image sharp) {
        spec.width = sharp.width;
        spec.height = sharp.height;
    }
    static void FillSampler(SamplerSpecialization& spec, ::Gcn::Sampler sharp) {
        spec.force_unnormalized = sharp.force_unnormalized;
        spec.force_degamma = sharp.force_degamma;
    }

    /// bbport: whether `*this == StageSpecialization(info_, rt, profile_, start_)`, without
    /// building the right side (no copies; the first difference ends it): building and comparing
    /// it was ~11 % of the GPU command thread on the Steam Deck. `fetch`: the info's parsed fetch
    /// shader (Gcn::ParseFetchShaderRef), null without one. Not for tessellation stages (their
    /// runtime info comes from a constant buffer).
    [[nodiscard]] bool Matches(const Info& info_, const RuntimeInfo& rt, const Profile& profile_,
                               Backend::Bindings start_, const Gcn::FetchShaderData* fetch) const {
        if (!Valid()) {
            return false;
        }
        size_t attribs = 0;
        if (info_.sw_stage == SwStage::Vertex && fetch) {
            for (const auto& desc : fetch->attributes) {
                VsAttribSpecialization spec{};
                if (const auto sharp = desc.GetSharp(info_)) {
                    FillVsAttrib(spec, desc, sharp, rt);
                }
                if (attribs >= vs_attribs.size() || !(vs_attribs[attribs] == spec)) {
                    return false;
                }
                ++attribs;
            }
        }
        if (attribs != vs_attribs.size() || runtime_info != rt ||
            fetch_shader_data.has_value() != (fetch != nullptr) ||
            (fetch && !(*fetch_shader_data == *fetch))) {
            return false;
        }
        if (fmasks.size() != info_.fmasks.size() || buffers.size() != info_.buffers.size() ||
            images.size() != info_.images.size() || samplers.size() != info_.samplers.size()) {
            return false;
        }
        bool any_bound = false; // the other side's bitset is not empty
        for (size_t i = 0; i < fmasks.size(); ++i) {
            FMaskSpecialization spec{};
            if (const auto sharp = info_.fmasks[i].GetSharp(info_)) {
                FillFmask(spec, sharp);
                any_bound = true;
            }
            if (!(fmasks[i] == spec)) {
                return false;
            }
        }
        if (bitset.none() && !any_bound) {
            bool none = true;
            for (const auto& desc : info_.buffers) {
                if (desc.GetSharp(info_)) {
                    none = false;
                    break;
                }
            }
            for (size_t i = 0; none && i < info_.images.size(); ++i) {
                if (info_.images[i].GetSharp(info_)) {
                    none = false;
                }
            }
            if (none) {
                return true;
            }
        }
        if (start != start_) {
            return false;
        }
        for (size_t i = 0; i < buffers.size(); ++i) {
            const auto& desc = info_.buffers[i];
            if (const auto sharp = desc.GetSharp(info_)) {
                BufferSpecialization spec{};
                FillBuffer(spec, desc, sharp, profile_, info_.pgm_hash);
                if (buffers[i] != spec) {
                    return false;
                }
            }
        }
        for (size_t i = 0; i < images.size(); ++i) {
            const auto& desc = info_.images[i];
            if (const auto sharp = desc.GetSharp(info_)) {
                ImageSpecialization spec{};
                FillImage(spec, desc, sharp, info_);
                if (images[i] != spec) {
                    return false;
                }
            }
        }
        for (size_t i = 0; i < samplers.size(); ++i) {
            SamplerSpecialization spec{};
            if (const auto sharp = info_.samplers[i].GetSharp(info_)) {
                FillSampler(spec, sharp);
            }
            if (samplers[i] != spec) {
                return false;
            }
        }
        return true;
    }

    void ForEachSharp(auto& spec_list, auto& desc_list, auto&& func) {
        for (const auto& desc : desc_list) {
            auto& spec = spec_list.emplace_back();
            const auto sharp = desc.GetSharp(*info);
            if (!sharp) {
                continue;
            }
            func(spec, desc, sharp);
        }
    }

    void ForEachSharp(u32& binding, auto& spec_list, auto& desc_list, auto&& func) {
        for (const auto& desc : desc_list) {
            auto& spec = spec_list.emplace_back();
            const auto sharp = desc.GetSharp(*info);
            if (!sharp) {
                binding++;
                continue;
            }
            bitset.set(binding++);
            func(spec, desc, sharp);
        }
    }

    [[nodiscard]] bool Valid() const {
        return info != nullptr;
    }

    bool operator==(const StageSpecialization& other) const {
        if (!Valid()) {
            return false;
        }

        if (vs_attribs != other.vs_attribs) {
            return false;
        }

        if (runtime_info != other.runtime_info) {
            return false;
        }

        if (fetch_shader_data != other.fetch_shader_data) {
            return false;
        }

        if (fmasks != other.fmasks) {
            return false;
        }

        // Without bindings the start bindings (which follow the other stages) do not matter.
        if (bitset.none() && other.bitset.none()) {
            return true;
        }

        if (start != other.start) {
            return false;
        }

        u32 binding{};
        for (u32 i = 0; i < buffers.size(); i++) {
            if (other.bitset[binding++] && buffers[i] != other.buffers[i]) {
                return false;
            }
        }
        for (u32 i = 0; i < images.size(); i++) {
            if (other.bitset[binding++] && images[i] != other.images[i]) {
                return false;
            }
        }

        for (u32 i = 0; i < samplers.size(); i++) {
            if (samplers[i] != other.samplers[i]) {
                return false;
            }
        }
        return true;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
