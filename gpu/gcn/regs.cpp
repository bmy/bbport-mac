// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
#include "gcn/regs.h"
#include "shim/bb_env.h"

#include <atomic>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Gcn {

namespace {

constexpr u32 ScissorTopLeft = 0x80000000;     ///< (0, 0), window offset off
constexpr u32 ScissorBottomRight = 0x40004000; ///< (16384, 16384)
constexpr u32 One = std::bit_cast<u32>(1.0f);

} // namespace

void Regs::SetDefaults() {
    reg_array.fill(0);
    auto& r = reg_array;
    // The CLEAR_STATE values of the context registers that are not zero.
    r[0xA00D] = ScissorBottomRight; // PA_SC_SCREEN_SCISSOR_BR
    r[0xA081] = ScissorTopLeft;     // PA_SC_WINDOW_SCISSOR_TL
    r[0xA082] = ScissorBottomRight;
    r[0xA083] = 0xdeadbeef;           // PA_SC_CLIPRECT_RULE
    for (u32 rect = 0; rect < 4; ++rect) { // PA_SC_CLIPRECT_n_TL/BR
        r[0xA085 + rect * 2] = ScissorBottomRight;
    }
    r[0xA08C] = 0xaa99aaaa; // PA_SC_EDGERULE
    r[0xA08E] = 0xdeadbeef; // CB_TARGET_MASK
    r[0xA08F] = 0xdeadbeef; // CB_SHADER_MASK
    r[0xA090] = ScissorTopLeft; // PA_SC_GENERIC_SCISSOR_TL
    r[0xA091] = ScissorBottomRight;
    for (u32 vp = 0; vp < NUM_VIEWPORTS; ++vp) {
        r[0xA094 + vp * 2] = ScissorTopLeft; // PA_SC_VPORT_SCISSOR_n_TL/BR
        r[0xA095 + vp * 2] = ScissorBottomRight;
        r[0xA0B5 + vp * 2] = One; // PA_SC_VPORT_ZMAX_n (ZMIN 0)
    }
    r[0xA0D4] = 0x2a00161a; // PA_SC_RASTER_CONFIG
    r[0xA1B6] = 0x00000002; // SPI_PS_IN_CONTROL
    r[0xA204] = 0x00090000; // PA_CL_CLIP_CNTL
    r[0xA205] = 0x00000004; // PA_SU_SC_MODE_CNTL
    r[0xA295] = 0x00000100; // VGT_GS_PER_ES
    r[0xA296] = 0x00000080; // VGT_ES_PER_GS
    r[0xA297] = 0x00000002; // VGT_GS_PER_VS
    r[0xA2AA] = 0x00001000; // IA_MULTI_VGT_PARAM
    r[0xA2F7] = 0x00001000; // PA_SC_LINE_CNTL
    r[0xA2F9] = 0x00000005; // PA_SU_VTX_CNTL
    for (u32 adj = 0; adj < 4; ++adj) { // PA_CL_GB_VERT/HORZ_CLIP/DISC_ADJ
        r[0xA2FA + adj] = One;
    }
    r[0xA316] = 0x0000000e; // VGT_VERTEX_REUSE_BLOCK_CNTL
    r[0xA317] = 0x00000010; // VGT_OUT_DEALLOC_CNTL
}

// Every named register at its index.
#define AT(member, index) static_assert(offsetof(Regs, member) == (index) * sizeof(u32), #member)
AT(ps_program, 0x2C08);
AT(ps_program.settings, 0x2C0A);
AT(ps_program.user_data, 0x2C0C);
AT(vs_program, 0x2C48);
AT(vs_program.user_data, 0x2C4C);
AT(gs_program, 0x2C88);
AT(es_program, 0x2CC8);
AT(hs_program, 0x2D08);
AT(ls_program, 0x2D48);
AT(cs_program, 0x2E00);
AT(cs_program.dim_z, 0x2E03);
AT(cs_program.num_thread_x, 0x2E07);
AT(cs_program.pad1, 0x2E0E);
AT(cs_program.settings, 0x2E12);
AT(cs_program.resource_limits, 0x2E15);
AT(cs_program.user_data, 0x2E40);
AT(depth_render_control, 0xA000);
AT(depth_view, 0xA002);
AT(depth_render_override, 0xA003);
AT(depth_htile_data_base, 0xA005);
AT(depth_bounds_min, 0xA008);
AT(depth_bounds_max, 0xA009);
AT(stencil_clear, 0xA00A);
AT(depth_clear, 0xA00B);
AT(screen_scissor, 0xA00C);
AT(depth_buffer.z_info, 0xA010);
AT(depth_buffer.z_read_base, 0xA012);
AT(depth_buffer.depth_slice, 0xA017);
AT(ta_bc_base, 0xA020);
AT(window_offset, 0xA080);
AT(window_scissor, 0xA081);
AT(color_target_mask, 0xA08E);
AT(color_shader_mask, 0xA08F);
AT(generic_scissor, 0xA090);
AT(viewport_scissors, 0xA094);
AT(viewport_depths, 0xA0B4);
AT(index_offset, 0xA102);
AT(primitive_restart_index, 0xA103);
AT(blend_constants, 0xA105);
AT(stencil_control, 0xA10B);
AT(stencil_ref_front, 0xA10C);
AT(stencil_ref_back, 0xA10D);
AT(viewports, 0xA10F);
AT(clip_user_data, 0xA16F);
AT(ps_inputs, 0xA191);
AT(vs_output_config, 0xA1B1);
AT(ps_input_ena, 0xA1B3);
AT(ps_input_addr, 0xA1B4);
AT(barycentric_control, 0xA1B8);
AT(shader_pos_format, 0xA1C3);
AT(z_export_format, 0xA1C4);
AT(color_export_format, 0xA1C5);
AT(blend_control, 0xA1E0);
AT(index_base_address, 0xA1F9);
AT(draw_initiator, 0xA1FC);
AT(depth_control, 0xA200);
AT(color_control, 0xA202);
AT(depth_shader_control, 0xA203);
AT(clipper_control, 0xA204);
AT(polygon_control, 0xA205);
AT(viewport_control, 0xA206);
AT(vs_output_control, 0xA207);
AT(line_control, 0xA282);
AT(hs_clamp, 0xA287);
AT(vgt_gs_mode, 0xA290);
AT(vgt_gs_onchip_control, 0xA291);
AT(mode_control, 0xA292);
AT(vgt_gsvs_ring_offset_1, 0xA298);
AT(vgt_gsvs_ring_offset_2, 0xA299);
AT(vgt_gsvs_ring_offset_3, 0xA29A);
AT(vgt_gs_out_prim_type, 0xA29B);
AT(index_size, 0xA29D);
AT(max_index_size, 0xA29E);
AT(index_buffer_type, 0xA29F);
AT(enable_primitive_id, 0xA2A1);
AT(enable_primitive_restart, 0xA2A5);
AT(vgt_instance_step_rate_0, 0xA2A8);
AT(vgt_instance_step_rate_1, 0xA2A9);
AT(vgt_esgs_ring_itemsize, 0xA2AB);
AT(vgt_gsvs_ring_itemsize, 0xA2AC);
AT(stage_enable, 0xA2D5);
AT(ls_hs_config, 0xA2D6);
AT(vgt_gs_vert_itemsize, 0xA2D7);
AT(tess_config, 0xA2DB);
AT(poly_offset, 0xA2DF);
AT(vgt_gs_instance_cnt, 0xA2E4);
AT(vgt_strmout_config, 0xA2E5);
AT(vgt_strmout_buffer_config, 0xA2E6);
AT(aa_config, 0xA2F8);
AT(color_buffers[0].base_address, 0xA318);
AT(color_buffers[0].pitch, 0xA319);
AT(color_buffers[0].slice, 0xA31A);
AT(color_buffers[0].info, 0xA31C);
AT(color_buffers[0].attrib, 0xA31D);
AT(color_buffers[0].cmask_base_address, 0xA31F);
AT(color_buffers[0].fmask_base_address, 0xA321);
AT(color_buffers[0].clear_word0, 0xA323);
AT(color_buffers[1].base_address, 0xA327);
AT(color_buffers[7].base_address, 0xA381);
AT(cp_strmout_cntl, 0xC03F);
AT(vgt_esgs_ring_size, 0xC240);
AT(vgt_gsvs_ring_size, 0xC241);
AT(primitive_type, 0xC242);
AT(num_indices, 0xC24C);
AT(num_instances, 0xC24D);
AT(vgt_tf_memory_base, 0xC250);
#undef AT
static_assert(sizeof(Regs) == Regs::NumRegs * sizeof(u32));
static_assert(sizeof(ShaderProgram) == 20 * sizeof(u32));
static_assert(sizeof(ComputeProgram) == 80 * sizeof(u32));
static_assert(sizeof(ColorBuffer) == 15 * sizeof(u32));

} // namespace Gcn

namespace Gcn {

namespace {
/// The register file after CLEAR_STATE (built once).
const Regs& DefaultRegs() {
    static Regs defaults;
    static std::once_flag once;
    std::call_once(once, [] { defaults.SetDefaults(); });
    return defaults;
}
} // namespace

void ResetRegs(Regs& regs, RegTouched& touched) {
    static const bool full = bb_env("BB_REGS_FULL_RESET") != nullptr;
    static const bool check = bb_env("BB_REGS_CHECK") != nullptr;
    const Regs& defaults = DefaultRegs();
    if (touched.all || full) {
        regs.reg_array = defaults.reg_array;
    } else {
        touched.marks.ForEachBlock([&](size_t block) {
            const size_t at = block * RegDirty::BlockWords;
            std::memcpy(regs.reg_array.data() + at, defaults.reg_array.data() + at,
                        RegDirty::BlockWords * sizeof(u32));
        });
    }
    touched.marks.Clear();
    touched.all = false;
    if (check) {
        u32 left = 0, first = ~0u;
        for (u32 block = 0; block < RegDirty::NumBlocks; ++block) {
            const size_t at = block * RegDirty::BlockWords;
            if (std::memcmp(regs.reg_array.data() + at, defaults.reg_array.data() + at,
                            RegDirty::BlockWords * sizeof(u32)) != 0) {
                ++left;
                first = std::min(first, block);
            }
        }
        static std::atomic<u32> reports{0};
        if (left && reports.fetch_add(1) < 16) {
            std::printf("Regs: %u blocks not back to their defaults after a reset (first at word "
                        "%#x): a write was not marked\n",
                        left, first * RegDirty::BlockWords);
        }
    }
}

} // namespace Gcn
