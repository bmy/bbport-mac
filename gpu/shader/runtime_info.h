// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the stages, and the state from the GPU registers a shader is
// compiled for (RuntimeInfo: per hardware and software stage). Two draws with the same program
// but different runtime info need different host shaders; it is part of the shader key.
#pragma once

#include <cstddef>

#include <algorithm>
#include <array>
#include <cstring>
#include <span>

#include "common/types.h"
#include "gcn/formats.h"
#include "gcn/regs_geometry.h"
#include "gcn/regs_shader.h"
#include "shader/frontend/tessellation.h"

namespace Shader {

/// The hardware stage a program runs as (its registers: PS, VS, GS, ES, HS, LS, CS).
enum class HwStage : u32 {
    Fragment,
    Vertex,
    Geometry,
    Export,
    Hull,
    Local,
    Compute,
};

/// What the program does (Vertex after the tessellation stages: compilation order).
enum class SwStage : u32 {
    Fragment,
    TessellationControl,
    TessellationEval,
    Vertex,
    Geometry,
    Compute,
    NumLogicalStages
};

constexpr u32 MaxStageTypes = static_cast<u32>(SwStage::NumLogicalStages);
constexpr auto MaxEmulatedClipDistances = 4u;

constexpr HwStage StageFromIndex(size_t index) noexcept {
    return static_cast<HwStage>(index);
}

/// What a VS/GS misc output vector component carries (SPI_SHADER_POS_FORMAT/PA_CL_VS_OUT_CNTL).
enum class Output : u8 {
    None,
    PointSize,
    EdgeFlag,
    KillFlag,
    GsCutFlag,
    RenderTargetIndex,
    ViewportIndex,
    CullDist0,
    CullDist1,
    CullDist2,
    CullDist3,
    CullDist4,
    CullDist5,
    CullDist6,
    CullDist7,
    ClipDist0,
    ClipDist1,
    ClipDist2,
    ClipDist3,
    ClipDist4,
    ClipDist5,
    ClipDist6,
    ClipDist7,
};
using OutputMap = std::array<Output, 4>;

struct SwVertexRuntimeInfo {
    u32 step_rate_0;
    u32 step_rate_1;
    u16 vertex_sgpr_offset{};
    u16 instance_sgpr_offset{};
    bool tess_emulated_primitive{};

    bool operator==(const SwVertexRuntimeInfo& other) const noexcept = default;
};

/// Object motion vectors for temporal upscaling. G-buffer vertex shaders store their
/// clip-space positions per referenced vertex and read the previous frame's; the
/// fragment shader writes the screen-space difference to an extra color attachment.
struct MotionVectors {
    static constexpr u32 CurrentLocation = 30;  ///< varying: current clip position
    static constexpr u32 PreviousLocation = 31; ///< varying: previous clip position, z = valid
#ifdef __APPLE__
    /// bbport (macOS): the first slot after the G-buffer's five targets. At 7, after two unused
    /// slots, KosmicKrisp's passes (Metal 4) never stored a pixel of it (an empty image in every
    /// dump; our own Metal path, the same layout, did).
    static constexpr u32 Output = 5;
#else
    static constexpr u32 Output = 7;            ///< color attachment index
#endif
    /// Per-draw parameters (u32x4: store base, load base, vertices per instance, flags) and
    /// the position array (vec4 per vertex; element 0 is scratch), fixed for the session.
    static inline u64 params_address = 0;
    static inline u64 positions_address = 0;
    static constexpr u32 FlagStore = 1;
    static constexpr u32 FlagLoad = 2;
};

struct HwLocalRuntimeInfo {
    u32 ls_stride;

    bool operator==(const HwLocalRuntimeInfo& other) const noexcept = default;
};

struct HwExportRuntimeInfo {
    u32 vertex_data_size;

    bool operator==(const HwExportRuntimeInfo& other) const noexcept = default;
};

struct HwVertexRuntimeInfo {
    u32 num_outputs;
    std::array<OutputMap, 3> outputs;
    bool emulate_depth_negative_one_to_one{};
    bool clip_disable{};
    u32 user_clip_plane_mask{};
    /// Object motion vectors (G-buffer draws), see MotionVectors.
    bool motion_vectors{};

    bool operator==(const HwVertexRuntimeInfo& other) const noexcept = default;
};

struct SwTessControlRuntimeInfo {
    u32 hs_output_cp_stride;
    u32 num_input_control_points;
    u32 num_threads;
    ::Gcn::TessellationType tess_type;
    bool offchip_lds_enable;
    /// bbport (macOS), BB_TESS_CLAMP=1: the game's VGT_HOS_MAX_TESS_LEVEL, which the hardware
    /// applies to the hull shader's factors, in 1/256ths (0: not applied, as upstream). In the
    /// padding after offchip_lds_enable: RuntimeInfo keeps 0.7's layout (it is stored raw, and
    /// the memset constructor zeroed this padding in every record: the shipped recipe pack too).
    u16 max_tess_level_256;
    u32 ls_stride;
    u32 hs_output_base;

    f32 MaxTessLevel() const {
        return f32(max_tess_level_256) / 256.0f;
    }
    void SetMaxTessLevel(f32 level) { // 0 <= level < 64
        max_tess_level_256 = u16(level * 256.0f + 0.5f);
    }

    bool operator==(const SwTessControlRuntimeInfo&) const = default;

    /// The hull shader writes no control points (output = input); a shader that only looks
    /// like one would just declare outputs the domain shader does not read.
    bool IsPassthrough() const {
        return hs_output_base == 0 && ls_stride == hs_output_cp_stride && num_threads == 1;
    };

    /// LS_HS_CONFIG holds the thread count, which passthrough shaders set to 1: then the
    /// output control points are the input ones.
    u32 NumOutputControlPoints() const {
        return IsPassthrough() ? num_input_control_points : num_threads;
    }
};
// bbport: 0.7's layout (the max tess level sits in padding; see above).
static_assert(offsetof(SwTessControlRuntimeInfo, max_tess_level_256) == 18 &&
              offsetof(SwTessControlRuntimeInfo, ls_stride) == 20 &&
              sizeof(SwTessControlRuntimeInfo) == 28);

struct SwTessEvalRuntimeInfo {
    u32 hs_output_cp_stride;
    ::Gcn::TessellationType tess_type;
    ::Gcn::TessellationTopology tess_topology;
    ::Gcn::TessellationPartitioning tess_partitioning;

    bool operator==(const SwTessEvalRuntimeInfo&) const noexcept = default;
};

static constexpr auto GsMaxOutputStreams = 4u;
using GsOutputPrimTypes = std::array<::Gcn::GsOutputPrimitiveType, GsMaxOutputStreams>;
struct HwGeometryRuntimeInfo {
    u32 num_outputs;
    std::array<OutputMap, 3> outputs;
    u32 num_invocations{};
    u32 output_vertices{};
    u32 in_vertex_data_size{};
    u32 out_vertex_data_size{};
    ::Gcn::PrimitiveType in_primitive;
    GsOutputPrimTypes out_primitive;
    ::Gcn::GsScenario mode;
    std::span<const u32> vs_copy;
    u64 vs_copy_hash;

    bool operator==(const HwGeometryRuntimeInfo& other) const {
        return num_outputs == other.num_outputs && outputs == other.outputs &&
               num_invocations == other.num_invocations && output_vertices == other.output_vertices &&
               in_primitive == other.in_primitive &&
               std::ranges::equal(out_primitive, other.out_primitive) &&
               vs_copy_hash == other.vs_copy_hash;
    }
};

enum class MrtSwizzle : u8 {
    Identity = 0,
    Alt = 1,
    Reverse = 2,
    ReverseAlt = 3,
};
static constexpr u32 MaxColorBuffers = 8;

struct PsColorBuffer {
    ::Gcn::DataFormat data_format : 6;
    ::Gcn::NumberFormat num_format : 4;
    ::Gcn::NumberConversion num_conversion : 3;
    ::Gcn::ShaderExportFormat export_format : 4;
    /// GCN applies blend factors to min/max while Vulkan ignores them: for min/max(src*src,
    /// dst*dst) the shader squares its output instead (the attachment stays squared).
    u32 blend_self_scale : 1;
    ::Gcn::CompMapping swizzle;

    bool operator==(const PsColorBuffer& other) const = default;
};

struct HwFragmentRuntimeInfo {
    struct PsInput {
        u8 param_index;
        bool is_default;
        bool is_flat;
        u8 default_value;

        bool IsDefault() const {
            return is_default && !is_flat;
        }

        bool IsPassthrough() const {
            return is_default && is_flat;
        }

        bool operator==(const PsInput&) const noexcept = default;
    };
    ::Gcn::PsInput en_flags;
    ::Gcn::PsInput addr_flags;
    u32 num_inputs;
    std::array<PsInput, 32> inputs;
    std::array<PsColorBuffer, MaxColorBuffers> color_buffers;
    ::Gcn::ShaderExportFormat z_export_format;
    u8 num_samples{1};
    u8 mrtz_mask{};
    bool front_face_all_bits{false};
    bool dual_source_blending{false};
    bool clip_distance_emulation{false};
    /// Writes the object motion vector to MotionVectors::Output.
    bool motion_vectors{false};

    bool operator==(const HwFragmentRuntimeInfo& other) const noexcept {
        return std::ranges::equal(color_buffers, other.color_buffers) &&
               en_flags == other.en_flags && addr_flags == other.addr_flags &&
               num_inputs == other.num_inputs && z_export_format == other.z_export_format &&
               num_samples == other.num_samples && mrtz_mask == other.mrtz_mask &&
               front_face_all_bits == other.front_face_all_bits &&
               dual_source_blending == other.dual_source_blending &&
               clip_distance_emulation == other.clip_distance_emulation &&
               motion_vectors == other.motion_vectors &&
               std::ranges::equal(inputs.begin(), inputs.begin() + num_inputs, other.inputs.begin(),
                                  other.inputs.begin() + num_inputs);
    }
};

struct HwComputeRuntimeInfo {
    u32 shared_memory_size;
    std::array<u32, 3> workgroup_size;
    std::array<bool, 3> tgid_enable;

    bool operator==(const HwComputeRuntimeInfo& other) const noexcept {
        return workgroup_size == other.workgroup_size && tgid_enable == other.tgid_enable;
    }
};

/// The register state a shader is compiled for; compared per stage (the unions hold the
/// stage's member).
struct RuntimeInfo {
    SwStage sw_stage;
    HwStage hw_stage;
    struct Properties {
        ::Gcn::FpDenormMode fp_denorm_mode32;
        ::Gcn::FpDenormMode fp_denorm_mode16_64;
        ::Gcn::FpRoundMode fp_round_mode32;
        ::Gcn::FpRoundMode fp_round_mode16_64;
        u32 num_user_data;
        u32 num_input_vgprs;
        u32 num_allocated_vgprs;
        bool operator==(const Properties&) const noexcept = default;
    } props;
    union SwInfo {
        SwVertexRuntimeInfo vs;
        SwTessControlRuntimeInfo tcs;
        SwTessEvalRuntimeInfo tes;
    } sw;
    union HwInfo {
        HwLocalRuntimeInfo ls;
        HwExportRuntimeInfo es;
        HwGeometryRuntimeInfo gs;
        HwVertexRuntimeInfo vs;
        HwFragmentRuntimeInfo fs;
        HwComputeRuntimeInfo cs;
    } hw;

    void Initialize(HwStage stage, SwStage l_stage) {
        std::memset(this, 0, sizeof(*this));
        this->hw_stage = stage;
        this->sw_stage = l_stage;
        if (stage == HwStage::Fragment) {
            hw.fs.num_samples = 1;
        }
    }

    bool operator==(const RuntimeInfo& other) const noexcept {
        return hw_stage == other.hw_stage && sw_stage == other.sw_stage && props == other.props &&
               HasSameSwInfo(other.sw) && HasSameHwInfo(other.hw);
    }

    bool HasSameSwInfo(const SwInfo& other) const noexcept {
        switch (sw_stage) {
        case SwStage::Vertex:
            return sw.vs == other.vs;
        case SwStage::TessellationControl:
            return sw.tcs == other.tcs;
        case SwStage::TessellationEval:
            return sw.tes == other.tes;
        default:
            return true;
        }
    }

    bool HasSameHwInfo(const HwInfo& other) const noexcept {
        switch (hw_stage) {
        case HwStage::Local:
            return hw.ls == other.ls;
        case HwStage::Export:
            return hw.es == other.es;
        case HwStage::Geometry:
            return hw.gs == other.gs;
        case HwStage::Vertex:
            return hw.vs == other.vs;
        case HwStage::Fragment:
            return hw.fs == other.fs;
        case HwStage::Compute:
            return hw.cs == other.cs;
        default:
            return true;
        }
    }

    void InitFromTessConstants(Shader::TessellationDataConstantBuffer& tess_constants) {
        if (sw_stage == SwStage::TessellationControl) {
            sw.tcs.hs_output_cp_stride = tess_constants.hs_cp_stride;
            sw.tcs.ls_stride = tess_constants.ls_stride;
            sw.tcs.hs_output_base = tess_constants.hs_output_base;
        } else if (sw_stage == SwStage::TessellationEval) {
            sw.tes.hs_output_cp_stride = tess_constants.hs_cp_stride;
        }
    }
};

} // namespace Shader

template <>
struct fmt::formatter<Shader::HwStage> {
    constexpr auto parse(format_parse_context& ctx) {
        return ctx.begin();
    }
    auto format(const Shader::HwStage stage, format_context& ctx) const {
        constexpr static std::array names = {"fs", "vs", "gs", "es", "hs", "ls", "cs"};
        return fmt::format_to(ctx.out(), "{}", names[static_cast<size_t>(stage)]);
    }
};
