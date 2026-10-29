// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport shader translator: the byte layout of translated shaders' metadata in the pipeline
// cache (render/pipeline_store.h). Info is stored as its persistent prefix, the flattened user
// data and the SRT walker (its machine code); a specialization field by field.

#include "common/serdes.h"
#include "shader/frontend/fetch_shader.h"
#include "shader/info.h"
#include "shader/specialization.h"

namespace Shader {

void Info::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    writer.Write(this, sizeof(InfoPersistent));
    writer.Write(flattened_ud_buf);
    srt_info.Serialize(ar);
}

bool Info::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    reader.Read(this, sizeof(InfoPersistent));
    reader.Read(flattened_ud_buf);
    return srt_info.Deserialize(ar);
}

void Gcn::FetchShaderData::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    ar.Grow(6 + attributes.size() * sizeof(VertexAttribute));
    writer.Write(size);
    writer.Write(vertex_offset_sgpr);
    writer.Write(instance_offset_sgpr);
    writer.Write(attributes);
}

bool Gcn::FetchShaderData::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    reader.Read(size);
    reader.Read(vertex_offset_sgpr);
    reader.Read(instance_offset_sgpr);
    reader.Read(attributes);
    return true;
}

void PersistentSrtInfo::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    writer.Write(this, sizeof(*this));
    if (walker_func_size) {
        writer.Write(reinterpret_cast<void*>(walker_func), walker_func_size);
    }
}

bool PersistentSrtInfo::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    reader.Read(this, sizeof(*this));
    // bbport: the stored pointer means nothing in this run, and the flat buffer holds at least
    // the user data registers (its slots are 16-bit).
    walker_func = nullptr;
    if (flattened_bufsize_dw < NUM_USER_DATA_REGS || flattened_bufsize_dw > 0x10000) {
        throw Serialization::CorruptData{"damaged SRT walker size in the shader cache"};
    }
    if (walker_func_size) {
        // The size is checked before the code is registered (it becomes executable): a file cut
        // short must not have bytes past its end run as the walker.
        const auto code = ar.CurrPtr();
        ar.Advance(walker_func_size);
        walker_func = RegisterWalkerCode(code, walker_func_size);
    }
    return true;
}

void StageSpecialization::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    writer.Write(start);
    writer.Write(runtime_info);
    writer.Write(bitset.to_string());
    if (fetch_shader_data) {
        writer.Write(sizeof(*fetch_shader_data));
        fetch_shader_data->Serialize(ar);
    } else {
        writer.Write(size_t{0});
    }
    writer.Write(vs_attribs);
    writer.Write(buffers);
    writer.Write(images);
    writer.Write(fmasks);
    writer.Write(samplers);
}

bool StageSpecialization::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    reader.Read(start);
    reader.Read(runtime_info);
    std::string bits{};
    reader.Read(bits);
    bitset = std::bitset<MaxStageResources>(bits);
    u64 fetch_data_size{};
    reader.Read(fetch_data_size);
    if (fetch_data_size) {
        Gcn::FetchShaderData fetch_data;
        fetch_data.Deserialize(ar);
        fetch_shader_data = fetch_data;
    }
    reader.Read(vs_attribs);
    reader.Read(buffers);
    reader.Read(images);
    reader.Read(fmasks);
    reader.Read(samplers);
    return true;
}

} // namespace Shader
