// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    /// bbport BB_SRT_CHECK (x86-64): the same walk as portable bytecode, compared on every walk.
    /// Not serialized: only shaders compiled in this run have it.
    const u32* check_code{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

/// bbport: runs the walker: x86 code on x86-64, bytecode elsewhere (flatten pass).
void RunSrtWalker(const PersistentSrtInfo& srt, const u32* user_data, u32* flat);

/// bbport: set by the GPU library to check guest pointers the bytecode walker follows.
extern bool (*srt_guest_readable)(u64 address, u64 size);
/// bbport: keeps a copy of portable walker bytecode for the rest of the run.
const u32* RegisterSrtBytecode(const u32* words, size_t count);

} // namespace Shader
