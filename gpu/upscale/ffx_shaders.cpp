// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport upscale: FSR 3.1 frame generation passes built in this tree (CMake add_ffx_shader) in
// place of the packaged ones (the FFX Vulkan backend's ffxVkShaderOverride). The SDK's GLSL with
// its resources and bindings unchanged; the changes make them faster on the GPUs the port runs
// on. BB_FFX_OWN_SHADERS=0: the packaged ones.
#include "upscale/ffx_shaders.h"
#include "shim/bb_env.h"

#include <cstdio>
#include <cstdlib>

#include "bbgpu/shaders/ffx_fi_interpolation.h"
#include "bbgpu/shaders/ffx_fi_interpolation_check.h"
#include "bbgpu/shaders/ffx_of_v5.h"
#include "bbgpu/shaders/ffx_of_v5_16.h"
#include "bbgpu/shaders/ffx_of_v5_16_hdr.h"
#include "bbgpu/shaders/ffx_of_v5_hdr.h"
#include "bbgpu/shaders/ffx_of_v5_16_fast.h"
#include "bbgpu/shaders/ffx_of_v5_16_hdr_fast.h"
#include "bbgpu/shaders/ffx_of_v5_fast.h"
#include "bbgpu/shaders/ffx_of_v5_hdr_fast.h"

extern "C" bool (*ffxVkShaderOverride)(uint32_t effect, uint32_t pass, uint32_t permutationOptions,
                                       const uint32_t** words, size_t* size);

namespace FfxShaders {

namespace {

// ffx_types.h FfxEffect, ffx_opticalflow.h FfxOpticalflowPass, ffx_opticalflow_private.h flags.
constexpr uint32_t EffectOpticalflow = 19;
constexpr uint32_t PassComputeOpticalFlowV5 = 4;
constexpr uint32_t PermutationWave64 = 1u << 0, PermutationFp16 = 1u << 1, PermutationHdr = 1u << 2;
// ffx_frameinterpolation.h FfxFrameInterpolationPass, ffx_frameinterpolation_private.h flags. The
// interpolation pass reads none of the permutation options except wave64 (no motion vectors, no
// depth, no half types), so one build serves all.
constexpr uint32_t EffectFrameInterpolation = 18;
constexpr uint32_t PassInterpolation = 6;
constexpr uint32_t FiPermutationWave64 = 1u << 3;

template <size_t N>
bool Use(const uint32_t (&code)[N], const uint32_t** words, size_t* size) {
    *words = code;
    *size = sizeof(code);
    return true;
}

bool Override(uint32_t effect, uint32_t pass, uint32_t options, const uint32_t** words, size_t* size) {
    if (effect == EffectFrameInterpolation && pass == PassInterpolation && !(options & FiPermutationWave64)) {
        std::printf("Frame generation: interpolation pass built in the port\n");
        // BB_FI_CHECK=1: the filtered samples are drawn as their difference to the SDK's loads.
        return bb_env_on("BB_FI_CHECK") ? Use(FFX_FI_INTERPOLATION_CHECK, words, size)
                                        : Use(FFX_FI_INTERPOLATION, words, size);
    }
    if (effect != EffectOpticalflow || pass != PassComputeOpticalFlowV5 || (options & PermutationWave64)) {
        return false;
    }
    static bool told = false;
    if (!told) {
        told = true;
        std::printf("Frame generation: optical flow search built in the port (fp16 %d hdr %d)\n",
                    int(bool(options & PermutationFp16)), int(bool(options & PermutationHdr)));
    }
    const bool fp16 = options & PermutationFp16, hdr = options & PermutationHdr;
    // BB_OF_FAST=1: no search on the finest level (its vectors from the coarser one).
    static const bool fast = [] {
        const bool on = bb_env_on("BB_OF_FAST");
        if (on) {
            std::printf("Frame generation: optical flow without the finest level's search\n");
        }
        return on;
    }();
    if (fast) {
        if (fp16) {
            return hdr ? Use(FFX_OF_V5_16_HDR_FAST, words, size) : Use(FFX_OF_V5_16_FAST, words, size);
        }
        return hdr ? Use(FFX_OF_V5_HDR_FAST, words, size) : Use(FFX_OF_V5_FAST, words, size);
    }
    if (fp16) {
        return hdr ? Use(FFX_OF_V5_16_HDR, words, size) : Use(FFX_OF_V5_16, words, size);
    }
    return hdr ? Use(FFX_OF_V5_HDR, words, size) : Use(FFX_OF_V5, words, size);
}

} // namespace

void Install() {
    if (bb_env_off("BB_FFX_OWN_SHADERS")) {
        return;
    }
    ffxVkShaderOverride = &Override;
}

} // namespace FfxShaders
