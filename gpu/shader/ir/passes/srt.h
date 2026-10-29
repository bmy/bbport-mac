// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the shader resource table walker kept with a shader: generated code
// that copies the descriptors the shader reads through pointers in its user data into one flat
// buffer, and that buffer's size.
#pragma once

#include "common/arch.h"
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/);

/// Makes `size` bytes of walker code at `ptr` executable; the function to call.
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

struct PersistentSrtInfo {
    /// A fetch shader's vertex buffer descriptors kept in place (instance step rates).
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; ///< the user data registers, then the walked descriptors

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

/// bbport: runs the walker: x86 code on x86-64, bytecode elsewhere (the macOS native GPU process,
/// docs/macos-native-gpu.md; flatten pass).
#ifdef ARCH_X86_64
inline void RunSrtWalker(const PersistentSrtInfo& srt, const u32* user_data, u32* flat) {
    srt.walker_func(user_data, flat);
}
#else
void RunSrtWalker(const PersistentSrtInfo& srt, const u32* user_data, u32* flat);

/// bbport: set by the GPU library to check guest pointers the bytecode walker follows.
extern bool (*srt_guest_readable)(u64 address, u64 size);
/// bbport (native GPU process): where this process sees a guest address the walker loads from
/// (the game image's data sits elsewhere there). Null: the address itself.
extern u64 (*srt_guest_translate)(u64 address);
/// bbport: keeps a copy of walker bytecode for the rest of the run.
const u32* RegisterSrtBytecode(const u32* words, size_t count);
#endif

} // namespace Shader
