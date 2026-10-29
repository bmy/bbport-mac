// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport gcn: the rasterizer's registers (PA_*): clipping, culling and polygon modes, viewports
// and scissors, multisampling.
#pragma once

#include <algorithm>

#include "common/types.h"

namespace Gcn {

inline constexpr u32 NUM_VIEWPORTS = 16;
inline constexpr u32 NUM_CLIP_PLANES = 6;

enum class ClipSpace : u32 {
    MinusWToW = 0,
    ZeroToW = 1,
};

enum class PrimKillCond : u32 {
    AllVtx = 0,
    AnyVtx = 1,
};

/// PA_CL_CLIP_CNTL.
struct ClipperControl {
    u32 user_clip_plane_enable : 6;
    u32 : 10;
    u32 clip_disable : 1;
    u32 : 2;
    ClipSpace clip_space : 1;
    u32 : 1;
    PrimKillCond vtx_kill_or : 1;
    u32 dx_rasterization_kill : 1;
    u32 : 1;
    u32 dx_linear_attr_clip_enable : 1;
    u32 : 1;
    u32 zclip_near_disable : 1;
    u32 zclip_far_disable : 1;

    /// Depth clipping on at both planes (one plane alone cannot be expressed: off).
    bool ZclipEnable() const {
        return !zclip_near_disable && !zclip_far_disable;
    }
};

enum class PolygonMode : u32 {
    Point = 0,
    Line = 1,
    Fill = 2,
};

enum class ProvokingVtxLast : u32 {
    First = 0,
    Last = 1,
};

enum class CullMode : u32 {
    None = 0,
    Front = 1,
    Back = 2,
    FrontAndBack = 3,
};

enum class FrontFace : u32 {
    CounterClockwise = 0,
    Clockwise = 1,
};

/// PA_SU_SC_MODE_CNTL.
struct PolygonControl {
    u32 cull_front : 1;
    u32 cull_back : 1;
    FrontFace front_face : 1;
    u32 enable_polygon_mode : 2;
    PolygonMode polygon_mode_front : 3;
    PolygonMode polygon_mode_back : 3;
    u32 enable_polygon_offset_front : 1;
    u32 enable_polygon_offset_back : 1;
    u32 enable_polygon_offset_para : 1;
    u32 : 2;
    u32 enable_window_offset : 1;
    u32 : 2;
    ProvokingVtxLast provoking_vtx_last : 1;
    u32 persp_corr_dis : 1;
    u32 multi_prim_ib_ena : 1;

    PolygonMode PolyMode() const {
        return enable_polygon_mode ? polygon_mode_front : PolygonMode::Fill;
    }
    CullMode CullingMode() const {
        return CullMode(cull_front | cull_back << 1);
    }
    bool NeedsBias() const {
        return enable_polygon_offset_back || enable_polygon_offset_front ||
               enable_polygon_offset_para;
    }
};

/// PA_CL_VS_OUT_CNTL.
struct VsOutputControl {
    u32 clip_distance_enable : 8;
    u32 cull_distance_enable : 8;
    u32 use_vtx_point_size : 1;
    u32 use_vtx_edge_flag : 1;
    u32 use_vtx_render_target_idx : 1;
    u32 use_vtx_viewport_idx : 1;
    u32 use_vtx_kill_flag : 1;
    u32 vs_out_misc_enable : 1;
    u32 vs_out_ccdist0_enable : 1;
    u32 vs_out_ccdist1_enable : 1;
    u32 vs_out_misc_side_bus_ena : 1;
    u32 use_vtx_gs_cut_flag : 1;

    bool IsClipDistEnabled(u32 index) const {
        return (clip_distance_enable >> index) & 1;
    }
    bool IsCullDistEnabled(u32 index) const {
        return (cull_distance_enable >> index) & 1;
    }
};

/// PA_SU_LINE_CNTL: the width in 1/8 pixels.
struct LineControl {
    u32 width_fixed_point;

    float Width() const {
        return float(width_fixed_point) / 8.0f;
    }
};

/// PA_SC_MODE_CNTL_0.
struct ModeControl {
    u32 msaa_enable : 1;
    u32 vport_scissor_enable : 1;
    u32 line_stripple_enable : 1;
    u32 send_unlit_stiles_to_pkr : 1;
};

/// PA_SC_SCREEN_SCISSOR_TL/BR: signed corners.
struct Scissor {
    struct {
        s16 top_left_x;
        s16 top_left_y;
    };
    struct {
        s16 bottom_right_x;
        s16 bottom_right_y;
    };

    static u16 Clamp(s16 value) {
        return u16(std::max<s16>(0, value));
    }
    u32 GetWidth() const {
        return u32(Clamp(bottom_right_x) - Clamp(top_left_x));
    }
    u32 GetHeight() const {
        return u32(Clamp(bottom_right_y) - Clamp(top_left_y));
    }
};

/// PA_SC_WINDOW_OFFSET.
struct WindowOffset {
    s32 window_x_offset : 16;
    s32 window_y_offset : 16;
};

/// PA_SC_WINDOW_SCISSOR, PA_SC_GENERIC_SCISSOR, PA_SC_VPORT_SCISSOR_n: TL and BR registers.
struct ViewportScissor {
    struct {
        u16 top_left_x : 15;
        u16 top_left_y : 15;
        u16 window_offset_disable : 1;
    };
    struct {
        u16 bottom_right_x : 15;
        u16 bottom_right_y : 15;
    };

    u32 GetWidth() const {
        return bottom_right_x - top_left_x;
    }
    u32 GetHeight() const {
        return bottom_right_y - top_left_y;
    }
};

/// PA_SC_VPORT_ZMIN/ZMAX_n.
struct ViewportDepth {
    float zmin;
    float zmax;
};

/// PA_CL_VPORT_XSCALE_n..ZOFFSET_n.
struct ViewportBounds {
    float xscale;
    float xoffset;
    float yscale;
    float yoffset;
    float zscale;
    float zoffset;
};

/// PA_CL_VTE_CNTL.
struct ViewportControl {
    u32 xscale_enable : 1;
    u32 xoffset_enable : 1;
    u32 yscale_enable : 1;
    u32 yoffset_enable : 1;
    u32 zscale_enable : 1;
    u32 zoffset_enable : 1;
    u32 : 2;
    u32 xy_transformed : 1;
    u32 z_transformed : 1;
    u32 w_transformed : 1;
    u32 perfcounter_ref : 1;
};

/// PA_CL_UCP_n_X..W.
struct ClipUserData {
    u32 data_x;
    u32 data_y;
    u32 data_z;
    u32 data_w;
};

/// PA_SC_AA_CONFIG.
struct AaConfig {
    u32 msaa_num_samples : 3; ///< log2
    u32 : 1;
    u32 aa_mask_centroid_dtmn : 1;
    u32 : 8;
    u32 max_sample_dst : 4;
    u32 : 3;
    u32 msaa_exposed_samples : 3;
    u32 : 1;
    u32 detail_to_exposed_mode : 2;

    u32 NumSamples() const {
        return 1u << msaa_num_samples;
    }
};

} // namespace Gcn
