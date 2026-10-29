// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: what the GPU library knows about the running game, set once at start (BbGpuConfig):
// its serial (the pipeline cache's name), title, SDK version and param.sfo attributes.
#pragma once

#include <string>
#include <string_view>

#include "common/bit_field.h"
#include "common/types.h"

namespace Common {

/// param.sfo ATTRIBUTE.
union PSFAttributes {
    BitField<0, 1, u32> support_initial_user_logout;
    BitField<1, 1, u32> enter_button_cross;
    BitField<2, 1, u32> ps_move_warning;
    BitField<3, 1, u32> support_stereoscopic_3d;
    BitField<4, 1, u32> ps_button_suspend;
    BitField<5, 1, u32> enter_button_system;
    BitField<6, 1, u32> override_share_menu;
    BitField<8, 1, u32> special_res_ps_button_suspend;
    BitField<9, 1, u32> enable_hdcp;
    BitField<10, 1, u32> disable_hdcp_non_game;
    BitField<14, 1, u32> support_ps_vr;
    BitField<15, 1, u32> six_cpu_mode;
    BitField<16, 1, u32> seven_cpu_mode;
    BitField<23, 1, u32> support_neo_mode;
    BitField<26, 1, u32> require_ps_vr;
    BitField<29, 1, u32> support_hdr;
    BitField<31, 1, u32> display_location;
    u32 raw{};
};
static_assert(sizeof(PSFAttributes) == 4);

class ElfInfo {
public:
    // SDK versions as games report them (sceKernelGetCompiledSdkVersion).
    static constexpr u32 FW_100 = 0x1000000;
    static constexpr u32 FW_150 = 0x1500000;
    static constexpr u32 FW_170 = 0x1700000;
    static constexpr u32 FW_200 = 0x2000000;
    static constexpr u32 FW_250 = 0x2500000;
    static constexpr u32 FW_300 = 0x3000000;
    static constexpr u32 FW_350 = 0x3500000;
    static constexpr u32 FW_400 = 0x4000000;
    static constexpr u32 FW_450 = 0x4500000;
    static constexpr u32 FW_500 = 0x5000000;
    static constexpr u32 FW_550 = 0x5500000;

    static ElfInfo& Instance() {
        static ElfInfo info;
        return info;
    }

    void Set(std::string serial, std::string game_title, u32 sdk, u32 attributes) {
        game_serial = std::move(serial);
        title = std::move(game_title);
        sdk_ver = sdk;
        psf_attributes.raw = attributes;
    }

    [[nodiscard]] std::string_view GameSerial() const {
        return game_serial;
    }
    [[nodiscard]] std::string_view Title() const {
        return title;
    }
    [[nodiscard]] u32 CompiledSdkVer() const {
        return sdk_ver;
    }
    [[nodiscard]] const PSFAttributes& GetPSFAttributes() const {
        return psf_attributes;
    }

private:
    std::string game_serial{"UNKNOWN"};
    std::string title;
    u32 sdk_ver = 0;
    PSFAttributes psf_attributes{};
};

} // namespace Common
