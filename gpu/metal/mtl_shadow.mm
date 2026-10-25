// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal shadow build (vk_metal_shadow.h; docs/MACOS_METAL_RENDERER.md, stage 2).
// From Wooloomooloo2/bloodborne_mac_port (78b59f5, GPL-2.0-or-later), renderer_metal/mtl_shadow.mm,
// without its MoltenVK layer's generic (host pass) pipelines. Every pipeline the Vulkan path
// creates is built again as a Metal pipeline state from the same key, SPIR-V and blend states;
// the counts show how much of the game the Metal command path's pipelines cover.
// BB_METAL_SHADOW=1.

#import <Metal/Metal.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "common/path_util.h"
#include "shader/runtime_info.h"
#include "metal/mtl_device.h"
#include "metal/mtl_pso_cache.h"
#include "metal/mtl_shader.h"
#include "metal/mtl_shadow.h"
#include "metal/mtl_spirv_rectlist.h"
#include "metal/mtl_geometry.h"
#include "metal/mtl_spirv_geometry.h"
#include "metal/vk_to_mtl.h"
#include "render/pipeline.h"
#include "metal/vk_metal_shadow.h"

namespace MetalShadow {
namespace {

/// To the log (stdout, as the rest of the GPU host's lines), flushed: a hang still shows it.
__attribute__((format(printf, 1, 2))) void Print(const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
    std::fflush(stdout);
}

struct ShaderEntry {
    std::once_flag once;
    std::unique_ptr<Metal::CompiledShader> compiled;
    std::once_flag rect_once; ///< the RectList rewrite of a vertex shader (mtl_spirv_rectlist.h)
    std::unique_ptr<Metal::CompiledShader> rect;
    std::mutex variant_mutex; ///< tessellation variants (Metal::CompileVariant)
    std::vector<std::pair<Metal::CompileVariant, std::unique_ptr<Metal::CompiledShader>>> variants;
};

struct State {
    /// The system's default Metal device: on Apple silicon (one GPU) the same object as
    /// KosmicKrisp's. Building needs no queue. Stage 3, which draws with these pipelines, takes
    /// the device KosmicKrisp exports (patch 0008) and checks it is this one.
    Metal::Device device{};
    std::filesystem::path dir;
    Metal::ShaderCompiler compiler;
    Metal::PsoCache pso;
    Metal::Geometry::PassThroughCache pass_through{device};
    std::mutex mutex;
    std::unordered_map<u64, std::shared_ptr<std::vector<u32>>> spirv; ///< by VkShaderModule
    std::unordered_map<u64, std::shared_ptr<ShaderEntry>> shaders;          ///< by VkShaderModule

    std::atomic<u32> render_ok{}, render_failed{}, compute_ok{}, compute_failed{};
    std::atomic<u32> shader_failed{}, no_spirv{};
    std::atomic<u32> rect_ok{}, rect_failed{};
    std::atomic<u32> tess_ok{}, tess_failed{}, tess_unsupported{};
    std::atomic<u32> geometry_ok{}, geometry_failed{}, geometry_unsupported{};
    std::atomic<u32> skipped_geometry{}, skipped_quad{},
        skipped_aux_fragment{};
    std::atomic<u32> seen{};

    std::mutex pipelines_mutex;
    std::unordered_map<VkPipeline, GraphicsEntry> graphics;
    std::unordered_map<VkPipeline, const Metal::ComputePipeline*> compute;

    static std::filesystem::path Dir() {
        auto dir = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / "metal-shadow";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        return dir;
    }
    State() : dir{Dir()}, compiler{device, dir}, pso{device, dir} {
        Print("Metal shadow: building every pipeline as a Metal pipeline state too (%s, %s)\n",
                    device.Name().c_str(), dir.string().c_str());
    }

    const Metal::CompiledShader* Get(vk::ShaderModule module, u64 hash, const char* what,
                                     bool rect_list = false) {
        const u64 handle = u64(VkShaderModule(module));
        std::shared_ptr<std::vector<u32>> words;
        std::shared_ptr<ShaderEntry> shader;
        {
            std::scoped_lock lock{mutex};
            auto it = spirv.find(handle);
            if (it == spirv.end()) {
                ++no_spirv;
                return nullptr;
            }
            words = it->second;
            auto& slot = shaders[handle];
            if (!slot) {
                slot = std::make_shared<ShaderEntry>();
            }
            shader = slot;
        }
        if (rect_list) {
            std::call_once(shader->rect_once, [&] {
                const auto rewrite = Metal::RewriteRectListVertex(*words);
                if (!rewrite.ok) {
                    Print("Metal shadow: rect list vertex shader %#llx not rewritable: %s\n",
                                (unsigned long long)hash, rewrite.error.c_str());
                    return;
                }
                auto result = compiler.Compile(rewrite.spirv, hash ^ 0x5265637443ull, "rect");
                if (result.status == Metal::CompileStatus::Ok) {
                    shader->rect = std::move(result.shader);
                }
            });
            return shader->rect.get();
        }
        std::call_once(shader->once, [&] {
            auto result = compiler.Compile(*words, hash, what);
            if (result.status == Metal::CompileStatus::Ok) {
                shader->compiled = std::move(result.shader);
            } else {
                ++shader_failed;
            }
        });
        return shader->compiled.get();
    }

    /// The module's SPIR-V and its cache entry, or false when it was never registered.
    bool Entry(vk::ShaderModule module, std::shared_ptr<std::vector<u32>>& words,
               std::shared_ptr<ShaderEntry>& entry) {
        const u64 handle = u64(VkShaderModule(module));
        std::scoped_lock lock{mutex};
        auto it = spirv.find(handle);
        if (it == spirv.end()) {
            ++no_spirv;
            return false;
        }
        words = it->second;
        auto& slot = shaders[handle];
        if (!slot) {
            slot = std::make_shared<ShaderEntry>();
        }
        entry = slot;
        return true;
    }

    const Metal::CompiledShader* Variant(vk::ShaderModule module, u64 hash,
                                         const Metal::CompileVariant& variant) {
        std::shared_ptr<std::vector<u32>> words;
        std::shared_ptr<ShaderEntry> entry;
        if (!Entry(module, words, entry)) {
            return nullptr;
        }
        std::scoped_lock lock{entry->variant_mutex};
        for (const auto& [known, shader] : entry->variants) {
            if (known == variant) {
                return shader.get();
            }
        }
        auto result = compiler.Compile(*words, hash, "variant", variant);
        auto& made = entry->variants.emplace_back(variant, std::move(result.shader));
        return made.second.get();
    }

    std::shared_ptr<std::vector<u32>> Words(vk::ShaderModule module) {
        std::shared_ptr<std::vector<u32>> words;
        std::shared_ptr<ShaderEntry> entry;
        return Entry(module, words, entry) ? words : nullptr;
    }

    void Report() {
        const u32 n = ++seen;
        if (n != 1 && n % 50 != 0) {
            return;
        }
        Print("Metal shadow: %u pipelines: render %u ok / %u failed, compute %u ok / %u "
                    "failed, geometry %u ok / %u failed / %u unsupported, rect lists %u ok / %u failed, tessellation %u ok / %u failed / %u not "
                    "possible in Metal; shaders failed %u, no SPIR-V %u; not yet: geometry %u, quad "
                    "lists %u, clip-emulation fragment %u\n",
                    n, render_ok.load(), render_failed.load(), compute_ok.load(),
                    compute_failed.load(), geometry_ok.load(), geometry_failed.load(),
                    geometry_unsupported.load(), rect_ok.load(), rect_failed.load(), tess_ok.load(),
                    tess_failed.load(), tess_unsupported.load(), shader_failed.load(),
                    no_spirv.load(), skipped_geometry.load(),
                    skipped_quad.load(), skipped_aux_fragment.load());
    }
};

State& Get() {
    static State state;
    return state;
}

} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* shadow = std::getenv("BB_METAL_SHADOW");
        return shadow && shadow[0] == '1';
    }();
    return enabled;
}

const GraphicsEntry* FindGraphics(VkPipeline pipeline) {
    auto& s = Get();
    std::scoped_lock lock{s.pipelines_mutex};
    auto it = s.graphics.find(pipeline);
    return it == s.graphics.end() ? nullptr : &it->second;
}

const Metal::ComputePipeline* FindCompute(VkPipeline pipeline) {
    auto& s = Get();
    std::scoped_lock lock{s.pipelines_mutex};
    auto it = s.compute.find(pipeline);
    return it == s.compute.end() ? nullptr : it->second;
}

void* DepthStencil(const Metal::DepthStencilKey& key) {
    return Get().pso.GetDepthStencilState(key);
}

void Forget(VkPipeline pipeline) {
    if (!Enabled()) {
        return;
    }
    auto& s = Get();
    std::scoped_lock lock{s.pipelines_mutex};
    s.graphics.erase(pipeline);
    s.compute.erase(pipeline);
}

void RegisterModule(vk::ShaderModule module, std::span<const u32> spirv) {
    auto& s = Get();
    auto words = std::make_shared<std::vector<u32>>(spirv.begin(), spirv.end());
    std::scoped_lock lock{s.mutex};
    s.spirv[u64(VkShaderModule(module))] = std::move(words);
    // A handle the driver reused: the shaders compiled from the module it was before are stale.
    s.shaders.erase(u64(VkShaderModule(module)));
}

void Graphics(const Vulkan::GraphicsPipelineKey& key, std::span<const vk::ShaderModule> modules,
              std::span<const bool> has_stage, std::span<const VertexInput> inputs,
              std::span<const vk::PipelineColorBlendAttachmentState> blend, bool aux_fragment,
              vk::Pipeline vk_pipeline) {
    @autoreleasepool { // pipeline-creation threads drain no pool of their own
    using Stage = ::Shader::SwStage;
    auto& s = Get();
    {
        // A handle the driver reused: what an earlier pipeline left must not stand for this one.
        std::scoped_lock lock{s.pipelines_mutex};
        s.graphics.erase(VkPipeline(vk_pipeline));
    }
    auto has = [&](Stage stage) { return has_stage[u32(stage)]; };
    if (has(Stage::TessellationControl) || has(Stage::TessellationEval)) {
        // As Huax-1839's PipelineCache::TessellationShaders: the vertex shader as a kernel per
        // index type, the control shader as a kernel, the evaluation shader as the
        // post-tessellation vertex function.
        using Kind = Metal::CompileVariant::Kind;
        const auto vs_module = modules[u32(Stage::Vertex)];
        const auto tcs_module = modules[u32(Stage::TessellationControl)];
        const auto tes_module = modules[u32(Stage::TessellationEval)];
        const auto tcs_words = s.Words(tcs_module);
        const auto tes_words = s.Words(tes_module);
        bool ok = has(Stage::Vertex) && has(Stage::TessellationControl) &&
                  has(Stage::TessellationEval) && tcs_words && tes_words;
        Metal::TessellationInput input;
        std::vector<Metal::VertexAttribute> attributes;
        for (const auto& in : inputs) {
            attributes.push_back({in.location, in.per_instance, in.divisor});
        }
        input.vertex_attributes = attributes;
        input.blend = blend;
        if (ok) {
            const auto tes_info = Metal::ReadTessellationInfo(*tes_words);
            const auto tcs_info = Metal::ReadTessellationInfo(*tcs_words);
            if ((tes_info.domain != Metal::TessDomain::Triangles &&
                 tes_info.domain != Metal::TessDomain::Quads) ||
                tes_info.point_mode) {
                ++s.tess_unsupported;
                s.Report();
                return;
            }
            ok = tcs_info.output_vertices != 0;
            for (u32 index_type = 0; ok && index_type < 3; ++index_type) {
                input.vertex[index_type] =
                    s.Variant(vs_module, key.stage_hashes[u32(Stage::Vertex)],
                              {.kind = Kind::TessVertex, .index_type = index_type});
                ok = input.vertex[index_type] != nullptr;
            }
            if (ok) {
                input.control = s.Variant(tcs_module, key.stage_hashes[u32(Stage::TessellationControl)],
                                          {.kind = Kind::TessControl, .domain = tes_info.domain});
                input.eval = s.Variant(tes_module, key.stage_hashes[u32(Stage::TessellationEval)],
                                       {.kind = Kind::TessEval,
                                        .output_vertices = tcs_info.output_vertices});
                ok = input.control && input.eval;
            }
        }
        const Metal::CompiledShader* fs = nullptr;
        if (ok && has(Stage::Fragment)) {
            fs = s.Get(modules[u32(Stage::Fragment)], key.stage_hashes[u32(Stage::Fragment)],
                       "fragment");
            ok = fs != nullptr;
        }
        const Metal::TessellationPipeline* tess =
            ok ? s.pso.GetTessellationPipeline(key, input, fs) : nullptr;
        ok = tess != nullptr;
        ++(ok ? s.tess_ok : s.tess_failed);
        if (ok) {
            std::scoped_lock lock{s.pipelines_mutex};
            s.graphics[VkPipeline(vk_pipeline)] = {.render = tess->render, .tess = tess};
        }
    } else if (has(Stage::Geometry)) {
        // As Huax-1839's PipelineCache::GeometryShaders: the export shader as a kernel per index
        // type, the geometry shader as a kernel, a generated pass-through vertex function.
        using Kind = Metal::CompileVariant::Kind;
        const auto es_module = modules[u32(Stage::Vertex)];
        const auto gs_module = modules[u32(Stage::Geometry)];
        const auto gs_words = s.Words(gs_module);
        std::string why;
        const Metal::GeometryPipeline* geo = nullptr;
        if (!has(Stage::Vertex) || !gs_words) {
            why = "a stage is missing";
        } else {
            const auto analysis = Metal::AnalyzeGeometryShader(*gs_words);
            const bool matches =
                analysis.ok &&
                ((analysis.info.input == Metal::GsInputPrimitive::Points &&
                  key.prim_type == Gcn::PrimitiveType::PointList) ||
                 (analysis.info.input == Metal::GsInputPrimitive::Lines &&
                  key.prim_type == Gcn::PrimitiveType::LineList) ||
                 (analysis.info.input == Metal::GsInputPrimitive::Triangles &&
                  key.prim_type == Gcn::PrimitiveType::TriangleList));
            if (!analysis.ok || !matches) {
                ++s.geometry_unsupported;
                why = analysis.ok ? "primitive type differs from the shader's input"
                                  : analysis.error;
            } else {
                Metal::GeometryPipelineInput input;
                std::vector<Metal::VertexAttribute> attributes;
                for (const auto& in : inputs) {
                    attributes.push_back({in.location, in.per_instance, in.divisor});
                }
                input.vertex_attributes = attributes;
                input.blend = blend;
                bool ok = true;
                for (u32 index_type = 0; ok && index_type < 3; ++index_type) {
                    input.es[index_type] =
                        s.Variant(es_module, key.stage_hashes[u32(Stage::Vertex)],
                                  {.kind = Kind::TessVertex, .index_type = index_type});
                    ok = input.es[index_type] != nullptr;
                }
                if (ok) {
                    Metal::CompileVariant variant{.kind = Kind::GeometryKernel};
                    variant.es.stride = input.es[0]->tess_layout.output;
                    variant.es.members = input.es[0]->tess_layout.members;
                    input.gs = s.Variant(gs_module, key.stage_hashes[u32(Stage::Geometry)], variant);
                    ok = input.gs != nullptr;
                }
                const Metal::CompiledShader* fs = nullptr;
                if (ok && has(Stage::Fragment)) {
                    fs = s.Get(modules[u32(Stage::Fragment)],
                               key.stage_hashes[u32(Stage::Fragment)], "fragment");
                    ok = fs != nullptr;
                }
                if (ok) {
                    std::span<const Metal::StageInput> fs_inputs;
                    if (fs) {
                        fs_inputs = fs->stage_inputs;
                    }
                    input.pass_through = s.pass_through.Get(
                        input.gs->geometry, fs_inputs,
                        key.clip_space == Gcn::ClipSpace::MinusWToW, why);
                    ok = input.pass_through != nullptr;
                }
                geo = ok ? s.pso.GetGeometryPipeline(key, input, fs) : nullptr;
                if (!geo && why.empty()) {
                    why = "a shader variant or the pipeline state failed";
                }
            }
        }
        ++(geo ? s.geometry_ok : s.geometry_failed);
        if (geo) {
            std::scoped_lock lock{s.pipelines_mutex};
            s.graphics[VkPipeline(vk_pipeline)] = {.render = geo->render, .geometry = geo};
        } else {
            static std::atomic<u32> logged{0};
            if (logged++ < 10) {
                Print("Metal shadow: geometry pipeline not built: %s\n", why.c_str());
            }
        }
    } else if (key.prim_type == Gcn::PrimitiveType::QuadList) {
        ++s.skipped_quad;
    } else if (aux_fragment) {
        ++s.skipped_aux_fragment;
    } else {
        // RectList: the vertex shader rewritten to emit two triangles per rectangle (MoltenVK
        // emulates it with a tessellation control shader instead).
        const bool rect_list = key.prim_type == Gcn::PrimitiveType::RectList;
        const auto* vs = s.Get(modules[u32(Stage::Vertex)], key.stage_hashes[u32(Stage::Vertex)],
                               "vertex", rect_list);
        const Metal::CompiledShader* fs = nullptr;
        bool ok = vs != nullptr;
        if (ok && has(Stage::Fragment)) {
            fs = s.Get(modules[u32(Stage::Fragment)], key.stage_hashes[u32(Stage::Fragment)],
                       "fragment");
            ok = fs != nullptr;
        }
        if (ok) {
            std::vector<Metal::VertexAttribute> attributes;
            for (const auto& in : inputs) {
                attributes.push_back({in.location, in.per_instance, in.divisor});
            }
            const Metal::RenderPipeline* render = s.pso.GetRenderPipeline(
                key, vs, fs, {.vertex_attributes = attributes, .blend = blend, .rect_list_triangles = rect_list});
            ok = render != nullptr;
            if (ok) {
                std::scoped_lock lock{s.pipelines_mutex};
                s.graphics[VkPipeline(vk_pipeline)] = {.render = render, .rect_list = rect_list};
            }
        }
        if (rect_list) {
            ++(ok ? s.rect_ok : s.rect_failed);
        } else {
            ++(ok ? s.render_ok : s.render_failed);
        }
    }
    s.Report();
    }
}

void Compute(const Vulkan::ComputePipelineKey& key, vk::ShaderModule module,
             vk::Pipeline vk_pipeline) {
    @autoreleasepool { // pipeline-creation threads drain no pool of their own
    auto& s = Get();
    {
        std::scoped_lock lock{s.pipelines_mutex};
        s.compute.erase(VkPipeline(vk_pipeline));
    }
    const auto* cs = s.Get(module, key.value, "compute");
    const Metal::ComputePipeline* pipeline = cs ? s.pso.GetComputePipeline(key, cs) : nullptr;
    const bool ok = pipeline != nullptr;
    if (ok) {
        std::scoped_lock lock{s.pipelines_mutex};
        s.compute[VkPipeline(vk_pipeline)] = pipeline;
    }
    ++(ok ? s.compute_ok : s.compute_failed);
    s.Report();
    }
}

} // namespace MetalShadow
