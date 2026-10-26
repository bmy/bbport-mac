// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal shadow build (vk_metal_shadow.h; docs/MACOS_METAL_RENDERER.md, stage 2).
// From Wooloomooloo2/bloodborne_mac_port (78b59f5, GPL-2.0-or-later), renderer_metal/mtl_shadow.mm,
// without its MoltenVK layer's generic (host pass) pipelines. Every pipeline the Vulkan path
// creates is built again as a Metal pipeline state from the same key, SPIR-V and blend states;
// the counts show how much of the game the Metal command path's pipelines cover.
// BB_METAL_SHADOW=1.

#import <Metal/Metal.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <pthread.h>
#include <thread>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <xxhash.h>
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

/// One SPIR-V module's translations, by its content (never dropped: pipeline states keep pointers
/// to them, and a module the driver gives an old handle again is another entry).
struct ShaderEntry {
    std::shared_ptr<const std::vector<u32>> words;
    std::once_flag once;
    std::unique_ptr<Metal::CompiledShader> compiled;
    std::once_flag rect_once; ///< the RectList rewrite of a vertex shader (mtl_spirv_rectlist.h)
    std::unique_ptr<Metal::CompiledShader> rect;
    /// bbport: the same two with clip space z converted from [-w, w] (CompileVariant::clip_fixup)
    std::once_flag clip_once, rect_clip_once;
    std::unique_ptr<Metal::CompiledShader> clip, rect_clip;
    std::mutex variant_mutex; ///< tessellation variants (Metal::CompileVariant)
    std::vector<std::pair<Metal::CompileVariant, std::unique_ptr<Metal::CompiledShader>>> variants;
};

struct State {
    /// The system's default Metal device: on Apple silicon (one GPU) the same object as
    /// KosmicKrisp's. Building needs no queue. Stage 3, which draws with these pipelines, takes
    /// the device KosmicKrisp exports (patch 0008) and checks it is this one.
    Metal::Device device{nullptr, nullptr, false}; // pipelines only
    std::filesystem::path dir;
    Metal::ShaderCompiler compiler;
    Metal::PsoCache pso;
    Metal::Geometry::PassThroughCache pass_through{device};
    std::mutex mutex;
    /// By VkShaderModule: the module registered last with that handle.
    std::unordered_map<u64, std::shared_ptr<ShaderEntry>> modules;
    /// By the SPIR-V's hash (XXH3); compared in full on a match.
    std::unordered_multimap<u64, std::shared_ptr<ShaderEntry>> shaders;

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

    /// `clip`: a vertex shader with clip space z converted from [-w, w] (the pipeline's key).
    const Metal::CompiledShader* Get(const std::shared_ptr<ShaderEntry>& shader, u64 hash,
                                     const char* what, bool rect_list = false, bool clip = false) {
        if (!shader) {
            ++no_spirv;
            return nullptr;
        }
        const auto& words = shader->words;
        const Metal::CompileVariant variant{.clip_fixup = clip};
        if (rect_list) {
            auto& once = clip ? shader->rect_clip_once : shader->rect_once;
            auto& slot = clip ? shader->rect_clip : shader->rect;
            std::call_once(once, [&] {
                const auto rewrite = Metal::RewriteRectListVertex(*words);
                if (!rewrite.ok) {
                    Print("Metal shadow: rect list vertex shader %#llx not rewritable: %s\n",
                                (unsigned long long)hash, rewrite.error.c_str());
                    return;
                }
                auto result =
                    compiler.Compile(rewrite.spirv, hash ^ 0x5265637443ull, "rect", variant);
                if (result.status == Metal::CompileStatus::Ok) {
                    slot = std::move(result.shader);
                }
            });
            return slot.get();
        }
        auto& once = clip ? shader->clip_once : shader->once;
        auto& slot = clip ? shader->clip : shader->compiled;
        std::call_once(once, [&] {
            auto result = compiler.Compile(*words, hash, what, variant);
            if (result.status == Metal::CompileStatus::Ok) {
                slot = std::move(result.shader);
            } else {
                ++shader_failed;
            }
        });
        return slot.get();
    }

    /// The module's entry (at pipeline creation: the handle may stand for another module by
    /// the time the build runs), null when it was never registered.
    std::shared_ptr<ShaderEntry> Lookup(vk::ShaderModule module) {
        if (!module) {
            return nullptr;
        }
        std::scoped_lock lock{mutex};
        auto it = modules.find(u64(VkShaderModule(module)));
        return it == modules.end() ? nullptr : it->second;
    }

    const Metal::CompiledShader* Variant(const std::shared_ptr<ShaderEntry>& entry, u64 hash,
                                         const Metal::CompileVariant& variant) {
        if (!entry) {
            ++no_spirv;
            return nullptr;
        }
        const auto& words = entry->words;
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

    std::shared_ptr<const std::vector<u32>> Words(const std::shared_ptr<ShaderEntry>& entry) {
        if (!entry) {
            ++no_spirv;
        }
        return entry ? entry->words : nullptr;
    }

    // ---- The builds run on worker threads of their own (off the pipeline-creation path) ----

    /// The build each pipeline handle is waiting for; a result is kept only while it still is.
    std::unordered_map<VkPipeline, u64> pending;
    u64 next_build = 0;

    /// Starts a build for `pipeline` (under pipelines_mutex): what an earlier pipeline with the
    /// handle left is dropped.
    u64 Expect(VkPipeline pipeline) {
        graphics.erase(pipeline);
        compute.erase(pipeline);
        const u64 id = ++next_build;
        pending[pipeline] = id;
        return id;
    }
    bool Current(VkPipeline pipeline, u64 id) {
        const auto it = pending.find(pipeline);
        if (it == pending.end() || it->second != id) {
            return false; // destroyed (Forget) or created again since
        }
        pending.erase(it);
        return true;
    }
    void PublishGraphics(VkPipeline pipeline, u64 id, const GraphicsEntry& entry) {
        std::scoped_lock lock{pipelines_mutex};
        if (Current(pipeline, id)) {
            graphics[pipeline] = entry;
        }
    }
    void PublishCompute(VkPipeline pipeline, u64 id, const Metal::ComputePipeline* p) {
        std::scoped_lock lock{pipelines_mutex};
        if (Current(pipeline, id)) {
            compute[pipeline] = p;
        }
    }
    void Abandon(VkPipeline pipeline, u64 id) {
        std::scoped_lock lock{pipelines_mutex};
        Current(pipeline, id);
    }

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<std::function<void()>> queue;
    std::vector<std::thread> workers;

    void Post(std::function<void()> task) {
        {
            std::scoped_lock lock{queue_mutex};
            queue.push_back(std::move(task));
            if (workers.empty()) {
                // Two: the Metal compiler runs its own threads; more only add memory.
                for (int i = 0; i < 2; ++i) {
                    workers.emplace_back([this] {
                        pthread_setname_np("bb:MetalShadow");
                        for (;;) {
                            std::function<void()> next;
                            {
                                std::unique_lock lock{queue_mutex};
                                queue_cv.wait(lock, [this] { return !queue.empty(); });
                                next = std::move(queue.front());
                                queue.pop_front();
                            }
                            @autoreleasepool {
                                try {
                                    next(); // a build that throws is abandoned (its Done)
                                } catch (...) {
                                    Print("Metal shadow: a build failed with an exception\n");
                                }
                            }
                        }
                    });
                    workers.back().detach(); // the state is never destroyed
                }
            }
        }
        queue_cv.notify_one();
    }

    void Report() {
        const u32 n = ++seen;
        size_t waiting = 0, queued = 0;
        {
            std::scoped_lock lock{pipelines_mutex};
            waiting = pending.size();
        }
        {
            std::scoped_lock lock{queue_mutex};
            queued = queue.size();
        }
        if (n % 200 == 0) {
            pso.Flush(); // the binary archive, for the next run (the state is never destroyed)
        }
        if (n != 1 && n % 50 != 0) {
            return;
        }
        Print("Metal shadow: %u pipelines: render %u ok / %u failed, compute %u ok / %u "
                    "failed, geometry %u ok / %u failed / %u unsupported, rect lists %u ok / %u failed, tessellation %u ok / %u failed / %u not "
                    "possible in Metal; shaders failed %u, no SPIR-V %u; not yet: geometry %u, quad "
                    "lists %u, clip-emulation fragment %u; builds waiting %zu (queued %zu)\n",
                    n, render_ok.load(), render_failed.load(), compute_ok.load(),
                    compute_failed.load(), geometry_ok.load(), geometry_failed.load(),
                    geometry_unsupported.load(), rect_ok.load(), rect_failed.load(), tess_ok.load(),
                    tess_failed.load(), tess_unsupported.load(), shader_failed.load(),
                    no_spirv.load(), skipped_geometry.load(),
                    skipped_quad.load(), skipped_aux_fragment.load(), waiting, queued);
    }
};

/// Never destroyed: pipeline-creation threads may still use it while the process exits.
State& Get() {
    static State* const state = new State;
    return *state;
}

} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* shadow = std::getenv("BB_METAL_SHADOW");
        const char* frame = std::getenv("BB_METAL_FRAME"); // its draws need the pipelines
        return (shadow && shadow[0] == '1') || (frame && frame[0] == '1');
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

void Forget(vk::Pipeline vk_pipeline) {
    const VkPipeline pipeline = vk_pipeline;
    if (!Enabled()) {
        return;
    }
    auto& s = Get();
    std::scoped_lock lock{s.pipelines_mutex};
    s.graphics.erase(pipeline);
    s.compute.erase(pipeline);
    s.pending.erase(pipeline); // a build still running for it is dropped when done
}

void RegisterModule(vk::ShaderModule module, std::span<const u32> spirv) {
    auto& s = Get();
    const u64 hash = XXH3_64bits(spirv.data(), spirv.size_bytes());
    std::scoped_lock lock{s.mutex};
    std::shared_ptr<ShaderEntry> entry;
    const auto [first, last] = s.shaders.equal_range(hash);
    for (auto it = first; it != last; ++it) {
        if (std::ranges::equal(*it->second->words, spirv)) {
            entry = it->second;
            break;
        }
    }
    if (!entry) {
        entry = std::make_shared<ShaderEntry>();
        entry->words = std::make_shared<const std::vector<u32>>(spirv.begin(), spirv.end());
        s.shaders.emplace(hash, entry);
    }
    // A handle the driver reused now stands for this module.
    s.modules[u64(VkShaderModule(module))] = std::move(entry);
}

namespace {
using ModuleRefs = std::array<std::shared_ptr<ShaderEntry>, Vulkan::MaxShaderStages>;

/// One graphics pipeline's build (a worker thread): what Graphics was given, kept.
void BuildGraphics(const Vulkan::GraphicsPipelineKey& key, const ModuleRefs& modules,
                   std::span<const bool> has_stage, std::span<const VertexInput> inputs,
                   std::span<const vk::PipelineColorBlendAttachmentState> blend,
                   bool aux_fragment, VkPipeline vk_pipeline, u64 id) {
    using Stage = ::Shader::SwStage;
    auto& s = Get();
    bool published = false;
    const auto publish = [&](const GraphicsEntry& entry) {
        s.PublishGraphics(vk_pipeline, id, entry);
        published = true;
    };
    struct Done { // a build without a result leaves the handle waiting for nothing
        State& s;
        bool& published;
        VkPipeline pipeline;
        u64 id;
        ~Done() {
            if (!published) {
                s.Abandon(pipeline, id);
            }
        }
    } done{s, published, vk_pipeline, id};
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
                                        .output_vertices = tcs_info.output_vertices,
                                        .clip_fixup = key.clip_space ==
                                                      Gcn::ClipSpace::MinusWToW});
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
            publish({.render = tess->render, .tess = tess});
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
            publish({.render = geo->render, .geometry = geo});
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
                               "vertex", rect_list,
                               key.clip_space == Gcn::ClipSpace::MinusWToW);
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
                publish({.render = render, .rect_list = rect_list});
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
} // namespace

void Graphics(const Vulkan::GraphicsPipelineKey& key, std::span<const vk::ShaderModule> modules,
              std::span<const bool> has_stage, std::span<const VertexInput> inputs,
              std::span<const vk::PipelineColorBlendAttachmentState> blend, bool aux_fragment,
              vk::Pipeline vk_pipeline) {
    auto& s = Get();
    // What the build needs, kept: the modules' entries now (a handle can stand for another
    // module by the time it runs), the rest by value.
    ModuleRefs refs{};
    for (size_t i = 0; i < refs.size() && i < modules.size(); ++i) {
        refs[i] = s.Lookup(modules[i]);
    }
    std::array<bool, Vulkan::MaxShaderStages> stages{};
    std::copy_n(has_stage.begin(), std::min(stages.size(), has_stage.size()), stages.begin());
    std::vector<VertexInput> input_list(inputs.begin(), inputs.end());
    std::vector<vk::PipelineColorBlendAttachmentState> blend_list(blend.begin(), blend.end());
    const VkPipeline pipeline = vk_pipeline;
    u64 id;
    {
        std::scoped_lock lock{s.pipelines_mutex};
        id = s.Expect(pipeline);
    }
    s.Post([key, refs, stages, input_list = std::move(input_list),
            blend_list = std::move(blend_list), aux_fragment, pipeline, id] {
        BuildGraphics(key, refs, stages, input_list, blend_list, aux_fragment, pipeline, id);
    });
}

void Compute(const Vulkan::ComputePipelineKey& key, vk::ShaderModule module,
             vk::Pipeline vk_pipeline) {
    auto& s = Get();
    auto ref = s.Lookup(module);
    const VkPipeline pipeline = vk_pipeline;
    u64 id;
    {
        std::scoped_lock lock{s.pipelines_mutex};
        id = s.Expect(pipeline);
    }
    s.Post([key, ref = std::move(ref), pipeline, id] {
        auto& s = Get();
        const auto* cs = s.Get(ref, key.value, "compute");
        const Metal::ComputePipeline* p = cs ? s.pso.GetComputePipeline(key, cs) : nullptr;
        if (p) {
            s.PublishCompute(pipeline, id, p);
        } else {
            s.Abandon(pipeline, id);
        }
        ++(p ? s.compute_ok : s.compute_failed);
        s.Report();
    });
}

} // namespace MetalShadow
