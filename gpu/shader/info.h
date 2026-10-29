// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: what the translator learns about a program (Info): its resources,
// the user data it reads, the inputs/outputs, the features it uses, and the flattened user data
// at draw time. InfoPersistent is the part kept with cached pipelines.
#pragma once

#include <bit>
#include <cstring>
#include <span>
#include <vector>

#include <boost/container/static_vector.hpp>

#include "common/assert.h"
#include "common/types.h"
#include "shader/backend/bindings.h"
#include "shader/frontend/copy_shader.h"
#include "shader/frontend/tessellation.h"
#include "shader/ir/attribute.h"
#include "shader/ir/passes/srt.h"
#include "shader/ir/reg.h"
#include "shader/ir/type.h"
#include "shader/params.h"
#include "shader/resource.h"
#include "shader/runtime_info.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

enum class Qualifier : u8 {
    None,
    Smooth,
    NoPerspective,
    PerVertex,
    Flat,
    Centroid,
    Sample,
};

/// BB_SHADER_CAPTURE: told of guest memory the compiler reads (size 0: a program).
inline thread_local void (*guest_read_hook)(const void* address, std::size_t size) = nullptr;
/// Shader corpus replay: the flattened user data the SRT walker produced in the game,
/// used instead of running the walker (it reads guest memory the record does not hold).
inline thread_local const std::vector<u32>* flat_buf_replay = nullptr;

struct InfoPersistent {
    BufferResourceList buffers;
    ImageResourceList images;
    SamplerResourceList samplers;
    FMaskResourceList fmasks;

    /// The user data registers the shader reads (pushed in order).
    struct UserDataMask {
        void Set(IR::ScalarReg reg) noexcept {
            mask |= 1 << static_cast<u32>(reg);
        }

        u32 Index(IR::ScalarReg reg) const noexcept {
            const u32 reg_mask = (1 << static_cast<u32>(reg)) - 1;
            return std::popcount(mask & reg_mask);
        }

        u32 NumRegs() const noexcept {
            return std::popcount(mask);
        }

        u32 mask;
    };
    UserDataMask ud_mask{};
    u32 fetch_shader_sgpr_base{};
    u32 shared_memory_scratch_size{};

    u64 pgm_hash{};

    s32 tess_consts_dword_offset = -1;
    IR::ScalarReg tess_consts_ptr_base = IR::ScalarReg::Max;
    HwStage hw_stage;
    SwStage sw_stage;

    u8 mrt_mask{};
    bool has_fetch_shader{};
    bool has_bitwise_xor{};
    bool uses_dma{};

    InfoPersistent() = default;
    InfoPersistent(HwStage hw_stage_, SwStage sw_stage_, u64 pgm_hash_)
        : hw_stage{hw_stage_}, sw_stage{sw_stage_}, pgm_hash{pgm_hash_} {}
};

struct Info : InfoPersistent {
    /// Per attribute, the components read or written.
    struct AttributeFlags {
        bool Get(IR::Attribute attrib, u32 comp = 0) const {
            return flags[Index(attrib)] & (1 << comp);
        }

        bool GetAny(IR::Attribute attrib) const {
            return flags[Index(attrib)];
        }

        void Set(IR::Attribute attrib, u32 comp = 0) {
            flags[Index(attrib)] |= (1 << comp);
        }

        u32 NumComponents(IR::Attribute attrib) const {
            return 4;
        }

        static size_t Index(IR::Attribute attrib) {
            return static_cast<size_t>(attrib);
        }

        std::array<u8, IR::NumAttributes> flags;
    };

    /// ReadConst kinds present: DMA reads (flags 0) need the page table.
    enum class ReadConstType {
        None = 0,
        Immediate = 1 << 0,
        Dynamic = 1 << 1,
    };

    struct Interpolation {
        Qualifier primary;
        Qualifier auxiliary;
    };

    std::span<const u32> user_data;
    std::vector<u32> flattened_ud_buf;
    PersistentSrtInfo srt_info;

    AttributeFlags loads{};
    AttributeFlags stores{};

    ReadConstType readconst_types{};
    CopyShaderData gs_copy_data;
    u32 uses_patches{};

    VAddr pgm_base;
    bool has_storage_images{};
    bool has_discard{};
    bool has_image_gather{};
    bool has_image_query{};
    bool has_readconst{};
    bool uses_buffer_atomic_float_min_max{};
    bool uses_image_atomic_float_min_max{};
    bool uses_lane_id{};
    bool uses_shader_clock{};
    /// BB_LAYER_MEMORY: some guest buffer goes through the page table (IsPagedBuffer).
    bool uses_paged_buffers{};
    bool uses_group_quad{};
    bool uses_group_shuffle{};
    bool uses_group_ballot{};
    IR::Type shared_types{};
    bool uses_fp16{};
    bool uses_fp64{};
    bool uses_pack_10_11_11{};
    bool uses_unpack_10_11_11{};
    bool uses_buffer_int64_atomics{};
    bool uses_shared_int64_atomics{};
    bool stores_tess_level_outer{};
    bool stores_tess_level_inner{};
    bool translation_failed{};

    std::array<Interpolation, IR::NumParams> fs_interpolation{};

    Info() = default;
    Info(HwStage stage_, SwStage l_stage_, ShaderParams params)
        : InfoPersistent(stage_, l_stage_, params.hash), pgm_base{params.Base()},
          user_data{params.user_data} {}

    /// User data of this stage as a draw saw it. The GPU command thread refreshes
    /// user_data/flattened_ud_buf for every draw it decodes, while the draw recording thread
    /// (vk_draw_pipe.h) may still work on an earlier draw: that thread installs snapshots here.
    struct UdSnapshot {
        const Info* info;
        const u32* user_data;
        u32 user_data_size;
        const u32* flat;
        u32 flat_size;
        VAddr pgm_base;
    };
    static constexpr u32 MaxUdSnapshots = 6;
    static inline thread_local UdSnapshot ud_snapshots[MaxUdSnapshots];
    static inline thread_local u32 num_ud_snapshots = 0;

    const UdSnapshot* Snapshot() const noexcept {
        for (u32 i = 0; i < num_ud_snapshots; ++i) {
            if (ud_snapshots[i].info == this) {
                return &ud_snapshots[i];
            }
        }
        return nullptr;
    }
    std::span<const u32> UserData() const noexcept {
        if (const auto* snapshot = Snapshot()) [[unlikely]] {
            return {snapshot->user_data, snapshot->user_data_size};
        }
        return user_data;
    }
    std::span<const u32> FlatUserData() const noexcept {
        if (const auto* snapshot = Snapshot()) [[unlikely]] {
            return {snapshot->flat, snapshot->flat_size};
        }
        return flattened_ud_buf;
    }

    VAddr ProgramBase() const noexcept {
        if (const auto* snapshot = Snapshot()) [[unlikely]] {
            return snapshot->pgm_base;
        }
        return pgm_base;
    }

    template <typename T>
    inline T ReadUdSharp(u32 sharp_idx) const noexcept {
        return *reinterpret_cast<const T*>(&FlatUserData()[sharp_idx]);
    }

    /// A value at `dword_offset` of the table the user data pointer at `ptr_index` points to
    /// (of the user data itself for NumScalarRegs).
    template <typename T>
    T ReadUdReg(u32 ptr_index, u32 dword_offset) const noexcept {
        T data{};
        const auto ud = UserData();
        const u32* base = ud.data();
        if (ptr_index != IR::NumScalarRegs) {
            if (ptr_index >= ud.size() || ud.size() - ptr_index < 2) {
                return data;
            }
            std::memcpy(&base, &ud[ptr_index], sizeof(base));
            base = reinterpret_cast<const u32*>(VAddr(base) & 0xFFFFFFFFFFFFULL);
            // Missing vertex tables can follow a failed game allocation. Like an unavailable
            // SRT load, a null table supplies an empty descriptor, never a host read at +offset.
            if (!base) {
                return data;
            }
            if (guest_read_hook) {
                guest_read_hook(base + dword_offset, sizeof(T));
            }
        } else if (dword_offset > ud.size() ||
                   sizeof(T) > (ud.size() - dword_offset) * sizeof(u32)) {
            return data;
        }
        std::memcpy(&data, base + dword_offset, sizeof(T));
        return data;
    }

    /// Copies the user data registers the shader reads into the push data.
    void PushUd(Backend::Bindings& bnd, PushData& push) const {
        const auto ud = UserData();
        u32 mask = ud_mask.mask;
        while (mask) {
            const u32 index = std::countr_zero(mask);
            ASSERT(bnd.user_data < NUM_USER_DATA_REGS && index < NUM_USER_DATA_REGS);
            mask &= ~(1U << index);
            push.ud_regs[bnd.user_data++] = ud[index];
        }
    }

    void AddBindings(Backend::Bindings& bnd) const {
        bnd.buffer += buffers.size();
        bnd.unified += buffers.size() + images.size() + samplers.size();
        bnd.user_data += ud_mask.NumRegs();
    }

    /// The flat user data for the current draw: the registers, then what the SRT walker copies.
    void RefreshFlatBuf() {
        if (flat_buf_replay) {
            flattened_ud_buf = *flat_buf_replay;
            return;
        }
        flattened_ud_buf.resize(srt_info.flattened_bufsize_dw);
        ASSERT(user_data.size() <= NUM_USER_DATA_REGS);
        std::memcpy(flattened_ud_buf.data(), user_data.data(), user_data.size_bytes());
        if (srt_info.walker_func) {
            RunSrtWalker(srt_info, user_data.data(), flattened_ud_buf.data());
        }
    }

    void ReadTessConstantBuffer(TessellationDataConstantBuffer& tess_constants) const {
        ASSERT(tess_consts_dword_offset >= 0); // (found by the tessellation pass)
        auto buf = ReadUdReg<::Gcn::Buffer>(static_cast<u32>(tess_consts_ptr_base),
                                             static_cast<u32>(tess_consts_dword_offset));
        VAddr tess_constants_addr = buf.base_address;
        if (guest_read_hook) {
            guest_read_hook(reinterpret_cast<const void*>(tess_constants_addr),
                            sizeof(tess_constants));
        }
        std::memcpy(&tess_constants,
               reinterpret_cast<TessellationDataConstantBuffer*>(tess_constants_addr),
               sizeof(tess_constants));
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};
DECLARE_ENUM_FLAG_OPERATORS(Info::ReadConstType);

} // namespace Shader
