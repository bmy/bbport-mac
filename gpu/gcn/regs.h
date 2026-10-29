// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport gcn: the GPU's register file as the command processor keeps it: 0xD000 dwords in four
// banks (config 0x2000, SH 0x2C00, context 0xA000, uconfig 0xC000), the registers the
// translator reads named at their word index (the comments give the first register of each),
// the words between them skipped. regs.cpp checks every index.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cstring>
#include <cstring>
#include <vector>

#include "gcn/regs_geometry.h"
#include "gcn/regs_raster.h"
#include "gcn/regs_shader.h"
#include "gcn/regs_targets.h"

namespace Gcn {

#define GCN_CAT2(a, b) a##b
#define GCN_CAT(a, b) GCN_CAT2(a, b)
/// The words [from, to) of the register file, not described.
#define GCN_SKIP(from, to) [[maybe_unused]] std::array<u32, (to) - (from)> GCN_CAT(skip_, __LINE__)

union Regs {
    static constexpr u32 NumRegs = 0xD000;
    static constexpr u32 ConfigRegWordOffset = 0x2000;
    static constexpr u32 ShRegWordOffset = 0x2C00;
    static constexpr u32 ContextRegWordOffset = 0xA000;
    static constexpr u32 UconfigRegWordOffset = 0xC000;

    struct {
        // SH registers: the graphics stages (20 words each), then compute (80).
        GCN_SKIP(0, 0x2C08);
        ShaderProgram ps_program; // SPI_SHADER_PGM_LO_PS
        GCN_SKIP(0x2C1C, 0x2C48);
        ShaderProgram vs_program; // SPI_SHADER_PGM_LO_VS
        GCN_SKIP(0x2C5C, 0x2C88);
        ShaderProgram gs_program; // SPI_SHADER_PGM_LO_GS
        GCN_SKIP(0x2C9C, 0x2CC8);
        ShaderProgram es_program; // SPI_SHADER_PGM_LO_ES
        GCN_SKIP(0x2CDC, 0x2D08);
        ShaderProgram hs_program; // SPI_SHADER_PGM_LO_HS
        GCN_SKIP(0x2D1C, 0x2D48);
        ShaderProgram ls_program; // SPI_SHADER_PGM_LO_LS
        GCN_SKIP(0x2D5C, 0x2E00);
        ComputeProgram cs_program; // COMPUTE_DISPATCH_INITIATOR
        GCN_SKIP(0x2E50, 0xA000);

        // Context registers.
        DepthRenderControl depth_render_control; // DB_RENDER_CONTROL
        GCN_SKIP(0xA001, 0xA002);
        DepthView depth_view;                     // DB_DEPTH_VIEW
        DepthRenderOverride depth_render_override; // DB_RENDER_OVERRIDE
        GCN_SKIP(0xA004, 0xA005);
        Address depth_htile_data_base; // DB_HTILE_DATA_BASE
        GCN_SKIP(0xA006, 0xA008);
        float depth_bounds_min; // DB_DEPTH_BOUNDS_MIN
        float depth_bounds_max; // DB_DEPTH_BOUNDS_MAX
        u32 stencil_clear;      // DB_STENCIL_CLEAR
        float depth_clear;      // DB_DEPTH_CLEAR
        Scissor screen_scissor; // PA_SC_SCREEN_SCISSOR_TL
        GCN_SKIP(0xA00E, 0xA010);
        DepthBuffer depth_buffer; // DB_Z_INFO
        GCN_SKIP(0xA018, 0xA020);
        BorderColorBuffer ta_bc_base; // TA_BC_BASE_ADDR
        GCN_SKIP(0xA022, 0xA080);
        WindowOffset window_offset;       // PA_SC_WINDOW_OFFSET
        ViewportScissor window_scissor;   // PA_SC_WINDOW_SCISSOR_TL
        GCN_SKIP(0xA083, 0xA08E);
        ColorBufferMask color_target_mask; // CB_TARGET_MASK
        ColorBufferMask color_shader_mask; // CB_SHADER_MASK
        ViewportScissor generic_scissor;   // PA_SC_GENERIC_SCISSOR_TL
        GCN_SKIP(0xA092, 0xA094);
        std::array<ViewportScissor, NUM_VIEWPORTS> viewport_scissors; // PA_SC_VPORT_SCISSOR_0_TL
        std::array<ViewportDepth, NUM_VIEWPORTS> viewport_depths;     // PA_SC_VPORT_ZMIN_0
        GCN_SKIP(0xA0D4, 0xA102);
        u32 index_offset;            // VGT_INDX_OFFSET
        u32 primitive_restart_index; // VGT_MULTI_PRIM_IB_RESET_INDX
        GCN_SKIP(0xA104, 0xA105);
        BlendConstants blend_constants; // CB_BLEND_RED
        GCN_SKIP(0xA109, 0xA10B);
        StencilControl stencil_control;  // DB_STENCIL_CONTROL
        StencilRefMask stencil_ref_front; // DB_STENCILREFMASK
        StencilRefMask stencil_ref_back;  // DB_STENCILREFMASK_BF
        GCN_SKIP(0xA10E, 0xA10F);
        std::array<ViewportBounds, NUM_VIEWPORTS> viewports;      // PA_CL_VPORT_XSCALE
        std::array<ClipUserData, NUM_CLIP_PLANES> clip_user_data; // PA_CL_UCP_0_X
        GCN_SKIP(0xA187, 0xA191);
        std::array<PsInputControl, 32> ps_inputs; // SPI_PS_INPUT_CNTL_0
        VsOutputConfig vs_output_config;          // SPI_VS_OUT_CONFIG
        GCN_SKIP(0xA1B2, 0xA1B3);
        PsInput ps_input_ena;  // SPI_PS_INPUT_ENA
        PsInput ps_input_addr; // SPI_PS_INPUT_ADDR
        GCN_SKIP(0xA1B5, 0xA1B6);
        u32 num_interp : 6; // SPI_PS_IN_CONTROL
        GCN_SKIP(0xA1B7, 0xA1B8);
        BarycentricControl barycentric_control; // SPI_BARYC_CNTL
        GCN_SKIP(0xA1B9, 0xA1C3);
        ShaderPosFormat shader_pos_format;     // SPI_SHADER_POS_FORMAT
        ShaderExportFormat z_export_format;    // SPI_SHADER_Z_FORMAT
        ColorExportFormat color_export_format; // SPI_SHADER_COL_FORMAT
        GCN_SKIP(0xA1C6, 0xA1E0);
        std::array<BlendControl, NUM_COLOR_BUFFERS> blend_control; // CB_BLEND0_CONTROL
        GCN_SKIP(0xA1E8, 0xA1F9);
        IndexBufferBase index_base_address; // VGT_DMA_BASE_HI
        GCN_SKIP(0xA1FB, 0xA1FC);
        u32 draw_initiator; // VGT_DRAW_INITIATOR
        GCN_SKIP(0xA1FD, 0xA200);
        DepthControl depth_control; // DB_DEPTH_CONTROL
        GCN_SKIP(0xA201, 0xA202);
        ColorControl color_control;              // CB_COLOR_CONTROL
        DepthShaderControl depth_shader_control; // DB_SHADER_CONTROL
        ClipperControl clipper_control;          // PA_CL_CLIP_CNTL
        PolygonControl polygon_control;          // PA_SU_SC_MODE_CNTL
        ViewportControl viewport_control;        // PA_CL_VTE_CNTL
        VsOutputControl vs_output_control;       // PA_CL_VS_OUT_CNTL
        GCN_SKIP(0xA208, 0xA282);
        LineControl line_control; // PA_SU_LINE_CNTL
        GCN_SKIP(0xA283, 0xA287);
        TessFactorClamp hs_clamp; // VGT_HOS_MAX_TESS_LEVEL
        GCN_SKIP(0xA289, 0xA290);
        GsMode vgt_gs_mode;              // VGT_GS_MODE
        GsOnchip vgt_gs_onchip_control;  // VGT_GS_ONCHIP_CNTL
        ModeControl mode_control;        // PA_SC_MODE_CNTL_0
        GCN_SKIP(0xA293, 0xA298);
        RingOffset vgt_gsvs_ring_offset_1;       // VGT_GSVS_RING_OFFSET_1
        RingOffset vgt_gsvs_ring_offset_2;       // VGT_GSVS_RING_OFFSET_2
        RingOffset vgt_gsvs_ring_offset_3;       // VGT_GSVS_RING_OFFSET_3
        GsOutPrimitiveType vgt_gs_out_prim_type; // VGT_GS_OUT_PRIM_TYPE
        GCN_SKIP(0xA29C, 0xA29D);
        u32 index_size;                    // VGT_DMA_SIZE
        u32 max_index_size;                // VGT_DMA_MAX_SIZE
        IndexBufferType index_buffer_type; // VGT_DMA_INDEX_TYPE
        GCN_SKIP(0xA2A0, 0xA2A1);
        u32 enable_primitive_id; // VGT_PRIMITIVEID_EN
        GCN_SKIP(0xA2A2, 0xA2A5);
        u32 enable_primitive_restart; // VGT_MULTI_PRIM_IB_RESET_EN
        GCN_SKIP(0xA2A6, 0xA2A8);
        u32 vgt_instance_step_rate_0; // VGT_INSTANCE_STEP_RATE_0
        u32 vgt_instance_step_rate_1; // VGT_INSTANCE_STEP_RATE_1
        GCN_SKIP(0xA2AA, 0xA2AB);
        u32 vgt_esgs_ring_itemsize; // VGT_ESGS_RING_ITEMSIZE
        u32 vgt_gsvs_ring_itemsize; // VGT_GSVS_RING_ITEMSIZE
        GCN_SKIP(0xA2AD, 0xA2CE);
        u32 vgt_gs_max_vert_out : 11; // VGT_GS_MAX_VERT_OUT
        GCN_SKIP(0xA2CF, 0xA2D5);
        ShaderStageEnable stage_enable;   // VGT_SHADER_STAGES_EN
        LsHsConfig ls_hs_config;          // VGT_LS_HS_CONFIG
        u32 vgt_gs_vert_itemsize[4];      // VGT_GS_VERT_ITEMSIZE
        TessellationConfig tess_config;   // VGT_TF_PARAM
        GCN_SKIP(0xA2DC, 0xA2DF);
        PolygonOffset poly_offset;                           // PA_SU_POLY_OFFSET_DB_FMT_CNTL
        GsInstances vgt_gs_instance_cnt;                     // VGT_GS_INSTANCE_CNT
        StreamOutConfig vgt_strmout_config;                  // VGT_STRMOUT_CONFIG
        StreamOutBufferConfig vgt_strmout_buffer_config;     // VGT_STRMOUT_BUFFER_CONFIG
        GCN_SKIP(0xA2E7, 0xA2F8);
        AaConfig aa_config; // PA_SC_AA_CONFIG
        GCN_SKIP(0xA2F9, 0xA318);
        ColorBuffer color_buffers[NUM_COLOR_BUFFERS]; // CB_COLOR0_BASE
        GCN_SKIP(0xA390, 0xC03F);

        // Uconfig registers.
        StreamOutControl cp_strmout_cntl; // CP_STRMOUT_CNTL
        GCN_SKIP(0xC040, 0xC240);
        u32 vgt_esgs_ring_size;      // VGT_ESGS_RING_SIZE
        u32 vgt_gsvs_ring_size;      // VGT_GSVS_RING_SIZE
        PrimitiveType primitive_type; // VGT_PRIMITIVE_TYPE
        GCN_SKIP(0xC243, 0xC24C);
        u32 num_indices;               // VGT_NUM_INDICES
        VgtNumInstances num_instances; // VGT_NUM_INSTANCES
        GCN_SKIP(0xC24E, 0xC250);
        TessFactorMemoryBase vgt_tf_memory_base; // VGT_TF_MEMORY_BASE
    };
    std::array<u32, NumRegs> reg_array;

    /// The stage's program: 0 pixel, 1 vertex, 2 geometry, 3 export, 4 hull, 5 local.
    const ShaderProgram* ProgramForStage(u32 index) const {
        const ShaderProgram* const programs[] = {&ps_program, &vs_program, &gs_program,
                                                 &es_program, &hs_program, &ls_program};
        return index < std::size(programs) ? programs[index] : nullptr;
    }

    bool IsClipDisabled() const {
        return clipper_control.clip_disable || primitive_type == PrimitiveType::RectList;
    }

    /// The state after CLEAR_STATE.
    void SetDefaults();
};

#undef GCN_SKIP
#undef GCN_CAT
#undef GCN_CAT2

/// The register blocks a stretch of packets wrote. The draw preparation scanner records one per
/// submission (with RegDelta), so a worker reaches the state at the start of any later
/// submission without replaying the packets in between.
struct RegDirty {
    static constexpr u32 BlockWords = 32;
    static constexpr u32 NumBlocks = Regs::NumRegs / BlockWords;
    std::bitset<NumBlocks> blocks;
    bool reset = false; ///< CLEAR_STATE: the defaults, then only the blocks marked after it

    void Mark(u32 word, u32 count) {
        if (count == 0 || word >= Regs::NumRegs) {
            return;
        }
        const u32 last = std::min(word + count, Regs::NumRegs) - 1;
        for (u32 block = word / BlockWords; block <= last / BlockWords; ++block) {
            blocks.set(block);
        }
    }
    template <typename T>
    void MarkField(const Regs& regs, const T& field) {
        const auto bytes = reinterpret_cast<const u8*>(&field) -
                           reinterpret_cast<const u8*>(regs.reg_array.data());
        Mark(u32(bytes / sizeof(u32)), u32((sizeof(T) + sizeof(u32) - 1) / sizeof(u32)));
    }
    void Clear() {
        blocks.reset();
        reset = false;
    }
    /// Calls `f(block)` for each marked block, in order.
    template <typename F>
    void ForEachBlock(F&& f) const {
#if defined(__GLIBCXX__)
        for (size_t block = blocks._Find_first(); block < blocks.size();
             block = blocks._Find_next(block)) {
            f(block);
        }
#else
        // libc++ (Windows): no _Find_next; its bitset is its words too.
        constexpr size_t Words = (NumBlocks + 63) / 64;
        static_assert(sizeof(blocks) == Words * sizeof(u64));
        u64 words[Words];
        std::memcpy(words, &blocks, sizeof(words));
        for (size_t w = 0; w < Words; ++w) {
            for (u64 bits = words[w]; bits; bits &= bits - 1) {
                f(w * 64 + size_t(std::countr_zero(bits)));
            }
        }
#endif
    }
};

/// bbport: the register blocks written since the last ResetRegs, all that can differ from the
/// defaults. CLEAR_STATE filled the whole file (208 KiB) and set the defaults again, ~63 times a
/// frame on each pipeline stage: ~25 MiB of memory traffic a frame.
struct RegTouched {
    RegDirty marks;
    bool all = true; ///< not known (a whole copy came in): the next reset is a whole one

    void Mark(u32 word, u32 count) {
        marks.Mark(word, count);
    }
};

/// CLEAR_STATE: the touched blocks back to their defaults (BB_REGS_FULL_RESET=1: the whole file;
/// BB_REGS_CHECK=1: compared with the defaults afterwards, a block left over is reported).
void ResetRegs(Regs& regs, RegTouched& touched);

/// The values of the blocks a RegDirty marked.
struct RegDelta {
    bool reset = false;
    std::vector<u16> blocks;
    std::vector<u32> words;

    void Capture(const Regs& regs, const RegDirty& dirty) {
        reset = dirty.reset;
        blocks.clear();
        words.clear();
        for (u32 block = 0; block < RegDirty::NumBlocks; ++block) {
            if (dirty.blocks.test(block)) {
                blocks.push_back(u16(block));
                const u32* from = regs.reg_array.data() + block * RegDirty::BlockWords;
                words.insert(words.end(), from, from + RegDirty::BlockWords);
            }
        }
    }
    void Apply(Regs& regs) const {
        if (reset) {
            regs.SetDefaults();
        }
        for (size_t i = 0; i < blocks.size(); ++i) {
            std::memcpy(regs.reg_array.data() + blocks[i] * RegDirty::BlockWords,
                        words.data() + i * RegDirty::BlockWords,
                        RegDirty::BlockWords * sizeof(u32));
        }
    }
};

} // namespace Gcn
