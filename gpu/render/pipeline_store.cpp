// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: saving to and preloading from the pipeline cache on disk (pipeline_store.h).

#include <array>
#include <cstring>
#include <unordered_set>

#include "common/serdes.h"
#include "core/emulator_settings.h"
#include "render/blob_store.h"
#include "render/pipeline_cache.h"
#include "render/pipeline_store.h"
#include "shader/info.h"
#include "shader/specialization.h"
#include "vulkan/instance.h"
#include "metal/vk_metal_shadow.h" // bbport (macOS): no-ops elsewhere

namespace Vulkan {

namespace {

BbCache::BlobStore& Store() {
    return BbCache::BlobStore::Instance();
}

std::string MetaName(u64 perm_hash) {
    return fmt::format("{:#018x}", perm_hash);
}

std::string BinaryName(u64 pgm_hash, size_t perm_idx) {
    return fmt::format("{:#018x}_{}", pgm_hash, perm_idx);
}

/// Pipeline entries: the version, graphics (0) or compute (1), the key, the support data.
template <typename Key, typename Support>
void SavePipeline(const Key& key, const Support& sdata, u32 is_compute, const std::string& name) {
    if (!Store().IsOpened()) {
        return;
    }
    Serialization::Archive ar{};
    Serialization::Writer writer{ar};
    writer.Write(PipelineStore::PipelineKeyVersion);
    writer.Write(is_compute);
    key.Serialize(ar);
    sdata.Serialize(ar);
    Store().Save(BbCache::BlobType::PipelineKey, name, ar.TakeOff());
}

/// A permutation's metadata. False when it is from another build, or a motion vertex shader:
/// those embed session-local buffer device addresses and are translated again.
bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx) {
    Serialization::Reader reader{ar};
    u32 meta_version{}, binary_version{};
    reader.Read(meta_version);
    if (meta_version != PipelineStore::ShaderMetaVersion) {
        return false;
    }
    reader.Read(binary_version);
    if (binary_version != PipelineStore::ShaderBinaryVersion) {
        return false;
    }
    u64 perm_hash{};
    reader.Read(perm_hash);
    reader.Read(perm_idx);
    spec.Deserialize(ar);
    info.Deserialize(ar);
    return !(info.hw_stage == Shader::HwStage::Vertex && spec.runtime_info.hw.vs.motion_vectors);
}

} // namespace

namespace PipelineStore {

void SaveCompute(const ComputePipelineKey& key, const ComputePipeline::SerializationSupport& sdata) {
    SavePipeline(key, sdata, 1, fmt::format("c_{:#018x}", key.value));
}

void SaveGraphics(const GraphicsPipelineKey& key, u64 hash,
                  const GraphicsPipeline::SerializationSupport& sdata) {
    SavePipeline(key, sdata, 0, fmt::format("g_{:#018x}", hash));
}

void SaveShaderMeta(const Shader::Info& info, const Shader::StageSpecialization& spec,
                    u64 perm_hash, size_t perm_idx) {
    if (!Store().IsOpened()) {
        return;
    }
    Serialization::Archive ar;
    Serialization::Writer writer{ar};
    writer.Write(ShaderMetaVersion);
    writer.Write(ShaderBinaryVersion);
    writer.Write(perm_hash);
    writer.Write(perm_idx);
    spec.Serialize(ar);
    info.Serialize(ar);
    Store().Save(BbCache::BlobType::ShaderMeta, MetaName(perm_hash), ar.TakeOff());
}

void SaveShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx) {
    if (!Store().IsOpened()) {
        return;
    }
    Store().Save(BbCache::BlobType::ShaderBinary, BinaryName(pgm_hash, perm_idx), std::move(spv));
}

} // namespace PipelineStore

void ComputePipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer{ar}.Write(value);
}

bool ComputePipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader{ar}.Read(value);
    return true;
}

void ComputePipeline::SerializationSupport::Serialize(Serialization::Archive&) const {}

bool ComputePipeline::SerializationSupport::Deserialize(Serialization::Archive&) {
    return true;
}

void GraphicsPipelineKey::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer{ar}.Write(this, sizeof(*this));
}

bool GraphicsPipelineKey::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader{ar}.Read(this, sizeof(*this));
    return true;
}

void GraphicsPipeline::SerializationSupport::Serialize(Serialization::Archive& ar) const {
    Serialization::Writer writer{ar};
    writer.Write(&vertex_attributes, sizeof(vertex_attributes));
    writer.Write(&vertex_bindings, sizeof(vertex_bindings));
    writer.Write(&divisors, sizeof(divisors));
    writer.Write(multisampling);
    writer.Write(tcs);
    writer.Write(tes);
    writer.Write(fragment);
}

bool GraphicsPipeline::SerializationSupport::Deserialize(Serialization::Archive& ar) {
    Serialization::Reader reader{ar};
    reader.Read(&vertex_attributes, sizeof(vertex_attributes));
    reader.Read(&vertex_bindings, sizeof(vertex_bindings));
    reader.Read(&divisors, sizeof(divisors));
    reader.Read(multisampling);
    reader.Read(tcs);
    reader.Read(tes);
    // The clip distance discard shader is stored since 2026-10-13; older entries end before it
    // (their pipelines are recreated without it, as they were before).
    if (!ar.IsEoS()) {
        reader.Read(fragment);
    }
    return true;
}

bool PipelineCache::LoadPipelineStage(Serialization::Archive& ar, size_t stage) {
    auto program = std::make_unique<Program>();
    Shader::StageSpecialization spec{};
    spec.info = &program->info;
    size_t perm_idx{};
    if (!LoadShaderMeta(ar, program->info, spec, perm_idx)) {
        return false;
    }
    if (spec.fetch_shader_data) {
        sel.fetch_shader = spec.fetch_shader_data; // the vertex stage's (as when selected)
    }

    std::vector<u32> spv{};
    Store().Load(BbCache::BlobType::ShaderBinary, BinaryName(program->info.pgm_hash, perm_idx),
                 spv);
    // SPIR-V starts with its magic number and a 5-word header; anything else is a damaged file
    // (a crash or power loss while it was written) the driver must not see.
    if (spv.size() < 5 || spv[0] != 0x07230203u) {
        if (!spv.empty()) {
            throw Serialization::CorruptData{"damaged SPIR-V in the shader cache"};
        }
        return false;
    }
    // A module the driver rejects is a damaged entry (the cache is rebuilt), not a fatal error.
    const auto compile = [&] {
        auto [result, created] = instance.GetDevice().createShaderModule(
            {.codeSize = spv.size() * sizeof(u32), .pCode = spv.data()});
        if (result != vk::Result::eSuccess) {
            throw Serialization::CorruptData{"cached SPIR-V rejected by the driver"};
        }
        if (MetalShadow::Enabled()) { // bbport (macOS): docs/MACOS_METAL_RENDERER.md, stage 2
            MetalShadow::RegisterModule(created, spv);
        }
        return created;
    };

    vk::ShaderModule module{};
    auto [it_pgm, new_program] = program_cache.try_emplace(program->info.pgm_hash);
    if (new_program) {
        module = compile();
        it_pgm.value() = std::move(program);
    } else {
        auto& modules = it_pgm.value()->modules;
        const auto it = std::ranges::find(modules, spec, &Program::Module::spec);
        if (it == modules.end()) {
            module = compile();
        } else {
            // A permutation is valid only at its original index (its hash includes it): another
            // index means entries of more than one cache generation, so the pipeline is left to
            // compile at run time.
            const size_t idx = std::distance(modules.begin(), it);
            if (perm_idx != idx) {
                LOG_WARNING(Render_Vulkan,
                            "Cached permutation {} of {}_{:x} conflicts with index {}, skipping "
                            "preload",
                            perm_idx, program->info.hw_stage, program->info.pgm_hash, idx);
                return false;
            }
            module = it->module;
        }
    }
    it_pgm.value()->InsertPermut(module, std::move(spec), perm_idx);
    sel.infos[stage] = &it_pgm.value()->info;
    sel.modules[stage] = module;
    return true;
}

bool PipelineCache::LoadComputePipeline(Serialization::Archive& ar) {
    sel.infos.fill(nullptr);
    sel.modules.fill(nullptr);
    compute_key.Deserialize(ar);
    ComputePipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);

    std::vector<u8> meta;
    Store().Load(BbCache::BlobType::ShaderMeta, MetaName(compute_key.value), meta);
    if (meta.empty()) {
        return false;
    }
    Serialization::Archive meta_ar{std::move(meta)};
    if (!LoadPipelineStage(meta_ar, 0)) {
        return false;
    }
    // Built before it is inserted: a pipeline the driver rejects throws
    // (Serialization::CorruptData) and leaves no empty entry behind.
    auto pipeline =
        std::make_unique<ComputePipeline>(instance, scheduler, desc_heap, profile, *pipeline_cache,
                                          compute_key, *sel.infos[0], sel.modules[0], sdata, true);
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    ASSERT(is_new);
    it.value() = std::move(pipeline);
    sel.infos.fill(nullptr);
    sel.modules.fill(nullptr);
    return true;
}

bool PipelineCache::LoadGraphicsPipeline(Serialization::Archive& ar) {
    // An entry left half loaded must not lend its stages to this one.
    sel.infos.fill(nullptr);
    sel.modules.fill(nullptr);
    sel.fetch_shader.reset();
    sel.graphics_key.Deserialize(ar);
    GraphicsPipeline::SerializationSupport sdata{};
    sdata.Deserialize(ar);
    // bbport: a pipeline with a geometry stage, cached while the driver had one, is skipped
    // (stale, not damaged) when it has none now: BB_GEOMETRY_SHADERS=0, or a branch on a driver
    // without geometry shaders. Building it there is invalid; the driver's rejection wiped the
    // whole cache.
    if (sel.graphics_key.stage_hashes[static_cast<u32>(Shader::SwStage::Geometry)] &&
        !instance.IsGeometryStageSupported()) {
        return false;
    }

    for (u32 stage = 0; stage < MaxShaderStages; ++stage) {
        const u64 hash = sel.graphics_key.stage_hashes[stage];
        if (!hash) {
            continue;
        }
        std::vector<u8> meta;
        Store().Load(BbCache::BlobType::ShaderMeta, MetaName(hash), meta);
        if (meta.empty()) {
            return false;
        }
        Serialization::Archive meta_ar{std::move(meta)};
        if (!LoadPipelineStage(meta_ar, stage)) {
            return false;
        }
    }
    auto pipeline = std::make_unique<GraphicsPipeline>(
        instance, scheduler, desc_heap, profile, sel.graphics_key, *pipeline_cache, sel.infos,
        sel.runtime_infos, sel.fetch_shader, sel.modules, sdata, true);
    const auto [it, is_new] = graphics_pipelines.try_emplace(sel.graphics_key);
    ASSERT(is_new);
    it.value() = std::move(pipeline);
    sel.infos.fill(nullptr);
    sel.modules.fill(nullptr);
    sel.fetch_shader.reset();
    return true;
}

namespace {
/// The cache's header: a tag and the versions; the device profile follows.
constexpr std::array<u32, 4> CacheVersions{0x42425043u, PipelineStore::ShaderBinaryVersion,
                                           PipelineStore::ShaderMetaVersion,
                                           PipelineStore::PipelineKeyVersion};
constexpr size_t HeaderSize = sizeof(CacheVersions);
} // namespace

void PipelineCache::SaveProfile() {
    std::vector<u8> data(HeaderSize + sizeof(profile));
    std::memcpy(data.data(), CacheVersions.data(), HeaderSize);
    std::memcpy(data.data() + HeaderSize, &profile, sizeof(profile));
    Store().Save(BbCache::BlobType::ShaderProfile, "profile", std::move(data));
}

void PipelineCache::DropPreloaded() {
    // Modules of a damaged entry may sit in programs that later lookups would reuse: nothing
    // preloaded is kept.
    graphics_pipelines.clear();
    compute_pipelines.clear();
    std::unordered_set<VkShaderModule> modules;
    for (const auto& [_, program] : program_cache) {
        if (!program) {
            continue;
        }
        for (const auto& permutation : program->modules) {
            if (permutation.module) {
                modules.insert(VkShaderModule(permutation.module));
            }
        }
    }
    for (const VkShaderModule module : modules) {
        instance.GetDevice().destroyShaderModule(vk::ShaderModule{module});
    }
    program_cache.clear();
}

void PipelineCache::WarmUp() {
    if (!EmulatorSettings.IsPipelineCacheEnabled()) {
        return;
    }
    Store().Open();
    const auto rebuild = [&] {
        Store().Clear();
        Store().FinishPreload();
        SaveProfile();
    };

    // Metadata and SPIR-V share permutation indices: blobs of another build next to this one's
    // could pair a valid metadata entry with another binary. The whole cache is rebuilt when the
    // versions or the device profile differ (or are missing: unknown compatibility). Before,
    // such a cache was only ignored and every later session compiled every shader again.
    std::vector<u8> header{};
    Store().Load(BbCache::BlobType::ShaderProfile, "profile", header);
    if (header.empty()) {
        rebuild();
        return;
    }
    bool compatible = header.size() == HeaderSize + sizeof(Shader::Profile) &&
                      std::memcmp(header.data(), CacheVersions.data(), HeaderSize) == 0;
    if (compatible) {
        Shader::Profile cached{};
        std::memcpy(&cached, header.data() + HeaderSize, sizeof(cached));
        compatible = cached == profile;
    }
    if (!compatible) {
        LOG_WARNING(Render,
                    "Pipeline cache isn't compatible with current compiler/system: rebuilding it");
        rebuild();
        return;
    }

    u32 num_pipelines = 0, num_total = 0, num_damaged = 0;
    Store().ForEachBlob(BbCache::BlobType::PipelineKey, [&](std::vector<u8>&& data) {
        ++num_total;
        // A damaged entry (cut short by a crash or a power loss, or rejected by the driver) used
        // to stop the game at every start until the cache was deleted by hand (issues #28, #38,
        // #39): it is counted and the cache rebuilt below.
        try {
            Serialization::Archive ar{std::move(data)};
            Serialization::Reader reader{ar};
            u32 version{}, is_compute{};
            reader.Read(version);
            if (version != PipelineStore::PipelineKeyVersion) {
                return;
            }
            reader.Read(is_compute);
            if (is_compute ? LoadComputePipeline(ar) : LoadGraphicsPipeline(ar)) {
                ++num_pipelines;
            }
        } catch (const std::exception& e) {
            if (num_damaged++ == 0) {
                LOG_WARNING(Render, "Pipeline cache: damaged entry ({})", e.what());
            }
            sel.infos.fill(nullptr);
            sel.modules.fill(nullptr);
            sel.fetch_shader.reset();
        }
    });

    if (num_damaged) {
        LOG_WARNING(Render, "Pipeline cache: {} damaged entries, rebuilding it", num_damaged);
        DropPreloaded();
        rebuild();
        return;
    }
    LOG_INFO(Render, "Preloaded {} pipelines", num_pipelines);
    if (num_total > num_pipelines) {
        LOG_WARNING(Render, "{} stale pipelines were found. Consider re-generating the cache",
                    num_total - num_pipelines);
    }
    Store().FinishPreload();
}

void PipelineCache::Sync() {
    Store().Close();
}

} // namespace Vulkan
