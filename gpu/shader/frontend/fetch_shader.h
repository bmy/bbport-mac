// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the fetch shader a vertex shader calls (S_SWAPPC_B64) to load its
// vertex attributes. Its loads are parsed back into the attribute list (V# location, destination
// VGPRs, instancing, typed format), which become vertex input attributes on the host.
#pragma once

#include <optional>
#include <vector>

#include "common/types.h"
#include "shader/info.h"

namespace Serialization {
struct Archive;
}

namespace Shader::Gcn {

struct VertexAttribute {
    /// The index VGPR the load uses: v0 is the vertex index, v1/v2 the instance index over step
    /// rate 0/1, v3 the plain instance index.
    enum InstanceIdType : u8 {
        None = 0,
        OverStepRate0 = 1,
        OverStepRate1 = 2,
        Plain = 3,
    };

    u8 semantic;      ///< input attribute index (in load order)
    u8 dest_vgpr;     ///< first VGPR loaded
    u8 num_elements;  ///< components loaded
    u8 sgpr_base;     ///< the user data SGPR pair holding the V# table pointer
    u8 dword_offset;  ///< the V#'s dword offset in that table
    u8 instance_data; ///< InstanceIdType
    u8 inst_offset;   ///< the load's byte offset
    u8 data_format{}; ///< typed loads (TBUFFER): the instruction's format
    u8 num_format{};

    InstanceIdType GetStepRate() const {
        return static_cast<InstanceIdType>(instance_data);
    }

    /// The attribute's V#, with the load's offset and typed format applied.
    constexpr ::Gcn::Buffer GetSharp(const Shader::Info& info) const noexcept {
        auto buffer = info.ReadUdReg<::Gcn::Buffer>(sgpr_base, dword_offset);
        buffer.base_address += inst_offset;
        if (data_format) {
            buffer.data_format = data_format;
            buffer.num_format = num_format;
        }
        return buffer;
    }

    bool operator==(const VertexAttribute& other) const {
        return semantic == other.semantic && dest_vgpr == other.dest_vgpr &&
               num_elements == other.num_elements && sgpr_base == other.sgpr_base &&
               dword_offset == other.dword_offset && instance_data == other.instance_data;
    }
};

struct FetchShaderData {
    u32 size = 0; ///< code bytes up to and including S_SETPC_B64
    std::vector<VertexAttribute> attributes;
    s8 vertex_offset_sgpr = -1;   ///< SGPR the fetch shader adds to v0 (base vertex), or -1
    s8 instance_offset_sgpr = -1; ///< SGPR added to v3 (base instance), or -1

    bool operator==(const FetchShaderData& other) const {
        return attributes == other.attributes && vertex_offset_sgpr == other.vertex_offset_sgpr &&
               instance_offset_sgpr == other.instance_offset_sgpr;
    }

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& buffer);
};

/// The fetch shader's code (its address is in the user data SGPR pair at `sgpr_base`).
const u32* GetFetchShaderCode(const Info& info, u32 sgpr_base);

std::optional<FetchShaderData> ParseFetchShader(const Shader::Info& info);

/// bbport: ParseFetchShader without the copy: the calling thread's cached parse, valid until its
/// next parse; null without a fetch shader.
const FetchShaderData* ParseFetchShaderRef(const Shader::Info& info);

} // namespace Shader::Gcn
