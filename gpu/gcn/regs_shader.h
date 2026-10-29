// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport gcn: the shader stages' registers (SPI_SHADER_*, COMPUTE_*, the pixel shader's inputs
// and exports) and the header the PS4 shader compiler appends to each program.
#pragma once

#include <array>
#include <bit>
#include <span>

#include "common/assert.h"
#include "common/types.h"
#include "shader/params.h"

namespace Gcn {

inline constexpr u32 NUM_USER_DATA = 16;

using UserData = std::array<u32, NUM_USER_DATA>;

/// The "OrbShdr" block after a program's code: its length and hash.
struct BinaryInfo {
    static constexpr std::array<u8, 7> signature_ref = {'O', 'r', 'b', 'S', 'h', 'd', 'r'};

    std::array<u8, sizeof(signature_ref)> signature;
    u8 version;
    u32 pssl_or_cg : 1;
    u32 cached : 1;
    u32 type : 4;
    u32 source_type : 2;
    u32 length : 24; ///< bytes of code
    u8 chunk_usage_base_offset_in_dw;
    u8 num_input_usage_slots;
    u8 is_srt : 1;
    u8 is_srt_used_info_valid : 1;
    u8 is_extended_usage_info : 1;
    u8 : 5;
    u8 reserved3;
    u64 shader_hash;
    u32 crc32;

    bool Valid() const {
        return signature == signature_ref;
    }
};

enum class FpRoundMode : u32 {
    NearestEven = 0,
    PlusInf = 1,
    MinInf = 2,
    ToZero = 3,
};

enum class FpDenormMode : u32 {
    InOutFlush = 0,
    InAllowOutFlush = 1,
    InFlushOutAllow = 2,
    InOutAllow = 3,
};

/// SPI_SHADER_PGM_LO_xS..USER_DATA_xS_15 of one graphics stage.
struct ShaderProgram {
    u64 address : 40; ///< 256-byte units
    struct {
        u32 num_vgprs : 6;
        u32 num_sgprs : 4;
        u32 priority : 2;
        FpRoundMode fp_round_mode32 : 2;
        FpRoundMode fp_round_mode64 : 2;
        FpDenormMode fp_denorm_mode32 : 2;
        FpDenormMode fp_denorm_mode64 : 2;
        u32 : 4;
        u32 vgpr_comp_cnt : 2;
        u32 : 6;
        u32 scratch_en : 1;
        u32 num_user_regs : 5;
        u32 : 1;
        u32 oc_lds_en : 1;
    } settings;
    UserData user_data;

    template <typename T = u8*>
    T Address() const {
        return std::bit_cast<T>(u64(address) << 8);
    }
    /// VGPRs come in blocks of four.
    [[nodiscard]] u32 NumVgprs() const {
        return (settings.num_vgprs + 1) * 4;
    }
};

/// SPI_VS_OUT_CONFIG.
struct VsOutputConfig {
    u32 : 1;
    u32 export_count_min_one : 5;
    u32 half_pack : 1;

    u32 NumExports() const {
        return export_count_min_one + 1;
    }
};

/// SPI_PS_INPUT_CNTL_n.
struct PsInputControl {
    u32 input_offset : 5;
    u32 use_default : 1;
    u32 : 2;
    u32 default_value : 2;
    u32 flat_shade : 1;
};

/// SPI_PS_INPUT_ENA / SPI_PS_INPUT_ADDR.
struct PsInput {
    u32 persp_sample_ena : 1;
    u32 persp_center_ena : 1;
    u32 persp_centroid_ena : 1;
    u32 persp_pull_model_ena : 1;
    u32 linear_sample_ena : 1;
    u32 linear_center_ena : 1;
    u32 linear_centroid_ena : 1;
    u32 line_stipple_tex_ena : 1;
    u32 pos_x_float_ena : 1;
    u32 pos_y_float_ena : 1;
    u32 pos_z_float_ena : 1;
    u32 pos_w_float_ena : 1;
    u32 front_face_ena : 1;
    u32 ancillary_ena : 1;
    u32 sample_coverage_ena : 1;
    u32 pos_fixed_pt_ena : 1;

    bool operator==(const PsInput&) const = default;
};

/// SPI_BARYC_CNTL.
struct BarycentricControl {
    u32 : 24;
    u32 front_face_all_bits : 1;
};
static_assert(sizeof(BarycentricControl) == sizeof(u32));

enum class ShaderExportComp : u32 {
    None = 0,
    OneComp = 1,
    TwoComp = 2,
    FourCompCompressed = 3,
    FourComp = 4,
};

/// SPI_SHADER_POS_FORMAT.
struct ShaderPosFormat {
    ShaderExportComp pos0 : 4;
    ShaderExportComp pos1 : 4;
    ShaderExportComp pos2 : 4;
    ShaderExportComp pos3 : 4;
};

enum class ShaderExportFormat : u32 {
    Zero = 0,
    R_32 = 1,
    GR_32 = 2,
    AR_32 = 3,
    ABGR_FP16 = 4,
    ABGR_UNORM16 = 5,
    ABGR_SNORM16 = 6,
    ABGR_UINT16 = 7,
    ABGR_SINT16 = 8,
    ABGR_32 = 9,
};

/// SPI_SHADER_COL_FORMAT: four bits per color target.
struct ColorExportFormat {
    u32 raw;

    [[nodiscard]] ShaderExportFormat GetFormat(u32 buf_idx) const {
        return ShaderExportFormat((raw >> (buf_idx * 4)) & 0xf);
    }
};

/// COMPUTE_DISPATCH_INITIATOR..COMPUTE_USER_DATA_15.
struct ComputeProgram {
    u32 dispatch_initiator;
    u32 dim_x;
    u32 dim_y;
    u32 dim_z;
    u32 start_x;
    u32 start_y;
    u32 start_z;
    struct {
        u16 full;
        u16 partial;
    } num_thread_x, num_thread_y, num_thread_z;
    u32 pad0;
    u32 max_wave_id : 12;
    u64 address : 40; ///< 256-byte units
    std::array<u32, 4> pad1;
    struct {
        u32 num_vgprs : 6;
        u32 num_sgprs : 4;
        u32 : 2;
        FpRoundMode fp_round_mode32 : 2;
        FpRoundMode fp_round_mode64 : 2;
        FpDenormMode fp_denorm_mode32 : 2;
        FpDenormMode fp_denorm_mode64 : 2;
        u32 : 12;
        u32 scratch_en : 1;
        u32 num_user_regs : 5;
        u32 : 1;
        u32 tgid_enable : 3;
        u32 : 5;
        u32 lds_dwords : 9; ///< 128-dword units
    } settings;
    u32 pad2;
    u32 resource_limits;
    std::array<u32, 42> pad3;
    UserData user_data;

    template <typename T = u8*>
    T Address() const {
        return std::bit_cast<T>(u64(address) << 8);
    }
    /// Bytes.
    u32 SharedMemSize() const noexcept {
        return settings.lds_dwords * 128 * 4;
    }
    u32 NumWorkgroups() const noexcept {
        return dim_x * dim_y * dim_z;
    }
    bool IsTgidEnabled(u32 i) const noexcept {
        return (settings.tgid_enable >> i) & 1;
    }
};

/// The header of the program at `code`: after a leading "s_mov_b32 vcc_hi, length" where it
/// says, else searched for in the next 64 KiB; null when there is none (not a program yet).
inline const BinaryInfo* FindBinaryInfo(const u32* code) {
    constexpr u32 MovVccHi = 0xBEEB03FF;
    if (code[0] == MovVccHi) {
        const auto* info = reinterpret_cast<const BinaryInfo*>(code + (code[1] + 1) * 2);
        if (info->Valid()) {
            return info;
        }
    }
    for (const u32* at = code; at < code + 0x4000; ++at) {
        if (const auto* info = reinterpret_cast<const BinaryInfo*>(at); info->Valid()) {
            return info;
        }
    }
    return nullptr;
}

/// The header exactly where the leading "s_mov_b32 vcc_hi, length" puts it (null otherwise).
inline const BinaryInfo* DirectBinaryInfo(const u32* code) {
    constexpr u32 MovVccHi = 0xBEEB03FF;
    if (code[0] != MovVccHi) {
        return nullptr;
    }
    const auto* info = reinterpret_cast<const BinaryInfo*>(code + (code[1] + 1) * 2);
    return info->Valid() ? info : nullptr;
}

inline const BinaryInfo& SearchBinaryInfo(const u32* code) {
    const BinaryInfo* info = FindBinaryInfo(code);
    if (!info) {
        UNREACHABLE_MSG("no shader binary header after {}", fmt::ptr(code));
    }
    return *info;
}

/// A stage's program for the recompiler: its user data, code and hash.
inline Shader::ShaderParams GetParams(const auto& program) {
    const auto* code = program.template Address<u32*>();
    const BinaryInfo& info = SearchBinaryInfo(code);
    return {
        .user_data = program.user_data,
        .code = std::span{code, info.length / sizeof(u32)},
        .hash = info.shader_hash,
    };
}

} // namespace Gcn
