// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the resources a shader binds. Each is described by where its sharp
// is read at draw time (SharpFetch: per dword a flat user data slot or an immediate), the post
// operation the shader applied to the sharp, and how the shader uses it. The push data block
// (viewport transform, user data registers, buffer offsets) is defined here as well.
#pragma once

#include <cstdlib>
#include "shim/bb_env.h"

#include <array>
#include <cstring>
#include <limits>

#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "common/types.h"
#include "gcn/sharps.h"
#include "shader/ir/type.h"

namespace Shader {

static constexpr u32 NUM_USER_DATA_REGS = 16;
static constexpr u32 NUM_IMAGES = 64;
static constexpr u32 NUM_BUFFERS = 40;
static constexpr u32 NUM_SAMPLERS = 16;
static constexpr u32 NUM_FMASKS = 8;

/// A dword index in the flattened user data.
using SharpLocation = u16;
constexpr SharpLocation UNKNOWN_LOCATION = std::numeric_limits<u16>::max();

/// How to assemble a sharp of type T: dwords from the flat user data (load_mask) or constants.
template <typename T>
struct SharpFetch {
    static constexpr std::size_t N = sizeof(T) / sizeof(u32);
    static_assert(N <= 8);

    std::array<u32, N> immediates;
    std::array<SharpLocation, N> offsets;
    u8 load_mask;

    bool operator==(const SharpFetch&) const = default;

    /// False if a dword's location is unknown (its read was not flattened).
    template <u32 num_dwords = N>
        requires(num_dwords <= N)
    constexpr bool Fetch(const u32* flatbuf, T* out) const {
        for (u32 i = 0; i < num_dwords; i++) {
            if (offsets[i] == UNKNOWN_LOCATION) {
                return false;
            }
        }
        std::array<u32, num_dwords> dwords;
        u8 mask = load_mask;
        for (u32 i = 0; i < num_dwords; i++, mask >>= 1) {
            dwords[i] = (mask & 1) ? flatbuf[offsets[i]] : immediates[i];
        }
        std::memcpy(out, dwords.data(), sizeof(dwords));
        return true;
    }
};

/// Edits the shader made to a sharp before use (see the resource discovery pass).
enum class SharpFetchPostOp : u8 {
    None,
    // Buffers
    BitwiseOrDw1WithImm,
    OffsetByProgramBase,
    // Images
    ConvertCubeTo2DArray,
    // Samplers
    DisableAnisoIfSingleLod,
    ForceRepeatXyzClamp,
    ForceLastTexelXyClamp,
    ClearAnisoRatioAndThreshold,
};

enum class BufferType : u8 {
    Guest,
    Flatbuf,      ///< the flattened user data (ReadConst)
    BdaPagetable, ///< the page table of buffer device addresses (DMA, paged buffers)
    FaultBuffer,  ///< DMA reads of unmapped pages are reported here
    GdsBuffer,
    SharedMemory, ///< LDS as a storage buffer
    ClipPlanes,
};

struct BufferResource {
    SharpFetch<::Gcn::Buffer> sharp_fetch{};
    IR::Type used_types{};
    BufferType buffer_type{};
    bool is_written{};
    bool is_formatted{};
    SharpFetchPostOp post_op{};
    u32 post_op_dw1_mask{};

    bool IsSpecial() const noexcept {
        return buffer_type != BufferType::Guest;
    }

    /// The V# at draw time (a null buffer if it cannot be read or is invalid).
    constexpr ::Gcn::Buffer GetSharp(const auto& info) const noexcept {
        ::Gcn::Buffer buffer;
        if (!sharp_fetch.Fetch(info.FlatUserData().data(), &buffer)) {
            return ::Gcn::Buffer::Null();
        }
        if (post_op == SharpFetchPostOp::BitwiseOrDw1WithImm) {
            reinterpret_cast<u32*>(&buffer)[1] |= post_op_dw1_mask;
        } else if (post_op == SharpFetchPostOp::OffsetByProgramBase) {
            buffer.base_address += info.ProgramBase();
        }
        return buffer.Valid() ? buffer : ::Gcn::Buffer::Null();
    }
};
using BufferResourceList = boost::container::static_vector<BufferResource, NUM_BUFFERS>;

/// BB_LAYER_MEMORY: a guest buffer of 64 MiB or more (a shader given nearly all memory as one
/// buffer) is reached through the page table, not as one bound range: the game's memory is in
/// several buffers (its chunks, VRAM mirrors). docs/MEMORY_MODULE_PLAN.ru.md. BB_PAGED_MIN_MB
/// overrides (tests).
inline u64 PagedBufferMinSize() {
    static const u64 size = [] {
        const char* env = bb_env("BB_PAGED_MIN_MB");
        const u64 mb = env && *env ? std::strtoull(env, nullptr, 10) : 64;
        return (mb >= 16 && mb <= 1024 ? mb : 64) << 20;
    }();
    return size;
}
/// Lazy guest memory: whether [va, va + size) lies across imports of the game's memory (two
/// granules, BbGuestMemory): such a buffer goes through the page table too, instead of an import
/// over both granules (overlapping imports: GTT counted twice, and past their budget the
/// binding went to VRAM copies). Null: no lazy guest memory.
inline bool (*paged_range_hook)(u64 va, u64 size) = nullptr;
/// Small ones (16 KiB, the stream threshold) are not paged for crossing alone: copied when bound
/// (the game rewrites constants right after its draws), a page table would read them later, as
/// the GPU gets there. Formatted ones are (their loads are raw ones, LowerBufferFormatToRaw).
inline bool PagedFormatted() {
    static const bool on = !bb_env_off("BB_PAGED_FORMATTED");
    return on;
}
/// A program whose buffers are paged for size alone (tests: BB_PAGED_COPY_CROSSING=0 makes it
/// the game's buffer copy shader).
inline u64 paged_crossing_exclude = 0;
inline bool IsPagedBuffer(bool enabled, const ::Gcn::Buffer& sharp, bool formatted = false,
                          u64 pgm_hash = 0) {
    if (!enabled || sharp.base_address == 0) {
        return false;
    }
    const u64 size = u64(sharp.GetSize());
    return size >= PagedBufferMinSize() ||
           (size > (u64{16} << 10) && (!formatted || PagedFormatted()) && paged_range_hook &&
            (pgm_hash == 0 || pgm_hash != paged_crossing_exclude) &&
            paged_range_hook(sharp.base_address, size));
}

/// Storage access to one mip level where the host cannot address levels in image operations:
/// a constant level is bound alone, a dynamic one as an array of single-level views.
enum class MipStorageFallbackMode : u16 {
    None,
    DynamicIndex,
    ConstantIndex,
};

struct ImageResource {
    SharpFetch<::Gcn::Image> sharp_fetch{};
    bool is_depth{};
    bool is_atomic{};
    bool is_array{};
    bool is_written{};
    bool is_r128{}; ///< a 4-dword T#
    /// Read other than by normalized sampling without offsets (texel loads, size queries,
    /// offsets): needs the native-size image, not a reduced scene proxy.
    bool needs_native{};
    u8 constant_mip_index{};
    MipStorageFallbackMode mip_fallback_mode{};
    SharpFetchPostOp post_op{};

    /// The T# at draw time; a null image (depth or color) if unreadable or invalid, and for
    /// depth reads of formats other than 16/32-bit.
    constexpr ::Gcn::Image GetSharp(const auto& info) const noexcept {
        ::Gcn::Image image{};
        if (!Fetch(info.FlatUserData().data(), &image)) {
            return ::Gcn::Image::Null(is_depth);
        }
        if (post_op == SharpFetchPostOp::ConvertCubeTo2DArray) {
            image.type = u64(::Gcn::ImageType::Color2DArray);
            image.depth = (image.depth + 1) * 6 - 1;
        }
        if (!image.Valid()) {
            return ::Gcn::Image::Null(is_depth);
        }
        if (is_depth) {
            const auto format = image.GetDataFmt();
            if (format != ::Gcn::DataFormat::Format16 && format != ::Gcn::DataFormat::Format32) {
                return ::Gcn::Image::Null(true);
            }
        }
        return image;
    }

    /// A 4-dword T# has its pitch equal to the width.
    constexpr bool Fetch(const u32* flatbuf, ::Gcn::Image* out) const {
        if (!is_r128) {
            return sharp_fetch.Fetch(flatbuf, out);
        }
        if (!sharp_fetch.Fetch<4>(flatbuf, out)) {
            return false;
        }
        out->pitch = out->width;
        return true;
    }

    u32 NumBindings(const auto& info) const {
        if (mip_fallback_mode != MipStorageFallbackMode::DynamicIndex) {
            return 1;
        }
        const ::Gcn::Image tsharp = GetSharp(info);
        return tsharp.last_level - tsharp.base_level + 1;
    }
};
using ImageResourceList = boost::container::static_vector<ImageResource, NUM_IMAGES>;

struct SamplerResource {
    SharpFetch<::Gcn::Sampler> sharp_fetch{};
    SharpFetchPostOp post_op{};
    SharpLocation post_op_tsharp_dw3_off{}; ///< DisableAnisoIfSingleLod: the T#'s dword 3
    bool is_depth{};

    constexpr ::Gcn::Sampler GetSharp(const auto& info) const noexcept {
        ::Gcn::Sampler sampler{};
        sharp_fetch.Fetch(info.FlatUserData().data(), &sampler);
        switch (post_op) {
        case SharpFetchPostOp::DisableAnisoIfSingleLod: {
            const u32 tsharp_dw3 = info.FlatUserData()[post_op_tsharp_dw3_off];
            if (((tsharp_dw3 >> 12) & 0xff) == 0) {
                sampler.max_aniso = ::Gcn::AnisoRatio::One;
            }
            break;
        }
        case SharpFetchPostOp::ForceRepeatXyzClamp:
            sampler.clamp_x = ::Gcn::ClampMode::Wrap;
            sampler.clamp_y = ::Gcn::ClampMode::Wrap;
            sampler.clamp_z = ::Gcn::ClampMode::Wrap;
            break;
        case SharpFetchPostOp::ForceLastTexelXyClamp:
            sampler.clamp_x = ::Gcn::ClampMode::ClampLastTexel;
            sampler.clamp_y = ::Gcn::ClampMode::ClampLastTexel;
            sampler.clamp_z = ::Gcn::ClampMode::Wrap;
            break;
        case SharpFetchPostOp::ClearAnisoRatioAndThreshold:
            sampler.max_aniso = ::Gcn::AnisoRatio::One;
            sampler.aniso_threshold = 0;
            break;
        default:
            break;
        }
        return sampler;
    }
};
using SamplerResourceList = boost::container::static_vector<SamplerResource, NUM_SAMPLERS>;

/// An FMask T# (its size specializes the shader; reads are answered with constants).
struct FMaskResource {
    SharpLocation sharp_idx;

    constexpr ::Gcn::Image GetSharp(const auto& info) const noexcept {
        return info.template ReadUdSharp<::Gcn::Image>(sharp_idx);
    }
};
using FMaskResourceList = boost::container::static_vector<FMaskResource, NUM_FMASKS>;

/// The push constant block shared by a pipeline's stages.
struct PushData {
    static constexpr u32 XOffsetIndex = 0;
    static constexpr u32 YOffsetIndex = 1;
    static constexpr u32 XScaleIndex = 2;
    static constexpr u32 YScaleIndex = 3;
    static constexpr u32 UdRegsIndex = 4;
    static constexpr u32 BufOffsetIndex = UdRegsIndex + NUM_USER_DATA_REGS / 4;
    static constexpr u32 MotionParamIndex = BufOffsetIndex + NUM_BUFFERS / 16 + 1;
    static constexpr u32 SceneSizeIndex = MotionParamIndex + 1;

    float xoffset;
    float yoffset;
    float xscale;
    float yscale;
    std::array<u32, NUM_USER_DATA_REGS> ud_regs;
    std::array<u8, NUM_BUFFERS> buf_offsets; ///< buffer offsets below the alignment (bytes)
    u32 motion_param; ///< object motion parameter index (0: off)
    u32 scene_size;   ///< reduced width | height << 16; 0 for native pixel coordinates

    void AddOffset(u32 binding, u32 offset) {
        ASSERT(offset < 256 && binding < buf_offsets.size());
        buf_offsets[binding] = offset;
    }
};
static_assert(sizeof(PushData) <= 128, "PushData exceeds the push constant size Vulkan guarantees");

} // namespace Shader
