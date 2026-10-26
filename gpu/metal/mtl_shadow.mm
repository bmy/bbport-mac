// SPDX-License-Identifier: GPL-2.0-or-later
// bbport (macOS): the Metal shadow build (vk_metal_shadow.h; docs/MACOS_METAL_RENDERER.md, stage 2).
// From Wooloomooloo2/bloodborne_mac_port (78b59f5, GPL-2.0-or-later), renderer_metal/mtl_shadow.mm,
// with its generic (host pass) pipelines, which its MoltenVK layer built from Vulkan create infos. Every pipeline the Vulkan path
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
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <vector>
#include <xxhash.h>
#include "common/path_util.h"
#include "shader/runtime_info.h"
#include "metal/mtl_device.h"
#include "metal/mtl_layer.h"
#include "metal/mtl_device_impl.h"
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
    /// bbport: pending builds a draw waits for rather than falling back (FindGraphicsWaiting):
    /// tessellated and geometry-shader pipelines, whose fallback is KosmicKrisp's emulation.
    std::unordered_set<VkPipeline> waitable;
    std::condition_variable built_cv; ///< a waitable build was published, abandoned or dropped

    /// Starts a build for `pipeline` (under pipelines_mutex): what an earlier pipeline with the
    /// handle left is dropped.
    u64 Expect(VkPipeline pipeline) {
        graphics.erase(pipeline);
        compute.erase(pipeline);
        if (waitable.erase(pipeline)) {
            built_cv.notify_all(); // (a waiter for the handle's earlier pipeline)
        }
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
        if (waitable.erase(pipeline)) {
            built_cv.notify_all();
        }
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

    /// `first`: ahead of the queued builds (one a draw waits for).
    void Post(std::function<void()> task, bool first = false) {
        {
            std::scoped_lock lock{queue_mutex};
            if (first) {
                queue.push_front(std::move(task));
            } else {
                queue.push_back(std::move(task));
            }
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
    static State* const state = [] {
        State* s = new State;
        // bbport: the binary archive is written every 200 pipelines (Report); the rest at exit
        // (the state is never destroyed, so ~PsoCache's flush never runs).
        std::atexit([] { Get().pso.Flush(); });
        return s;
    }();
    return *state;
}

} // namespace

bool Enabled() {
    static const bool shadow_asked = [] {
        const char* shadow = std::getenv("BB_METAL_SHADOW");
        return shadow && shadow[0] == '1';
    }();
    static const bool frame_asked = [] {
        const char* frame = std::getenv("BB_METAL_FRAME"); // its draws need the pipelines
        return frame && frame[0] == '1';
    }();
    // bbport: for the frame encoder only while it is on (not when the layer fell back to
    // counting: validation layers, an Init that failed); the layer is installed at the device's
    // creation, before any pipeline.
    return shadow_asked || (frame_asked && MetalLayer::FrameActive());
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
    if (s.waitable.erase(pipeline)) {
        s.built_cv.notify_all();
    }
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
} // namespace

/// bbport: a plain game pipeline's build inputs (GraphicsEntry::source); the module entries keep
/// the compiled shaders alive.
struct VariantSource {
    Vulkan::GraphicsPipelineKey key;
    ModuleRefs modules;
    const Metal::CompiledShader* vs{};
    const Metal::CompiledShader* fs{};
    std::vector<Metal::VertexAttribute> attributes;
    std::vector<vk::PipelineColorBlendAttachmentState> blend;
    bool rect_list{};
};

namespace {

/// One graphics pipeline's build (a worker thread): what Graphics was given, kept.
/// bbport: whether a fragment shader has flat inputs (OpDecorate id Flat, OpMemberDecorate
/// struct member Flat) other than built-ins (the recompiler decorates the sample index, layer and
/// subgroup built-ins Flat: they do not come from a vertex).
bool HasFlat(const std::vector<u32>& w) {
    constexpr u32 Flat = 14, BuiltIn = 11;
    std::vector<u32> flat_ids, builtin_ids;
    std::vector<std::pair<u32, u32>> flat_members, builtin_members;
    for (size_t i = 5; i < w.size();) {
        const u32 count = w[i] >> 16, op = w[i] & 0xffff;
        if (!count || i + count > w.size() || op == 54) { // OpFunction: decorations are before
            break;
        }
        if (op == 71 && count >= 3) {
            if (w[i + 2] == Flat) {
                flat_ids.push_back(w[i + 1]);
            } else if (w[i + 2] == BuiltIn) {
                builtin_ids.push_back(w[i + 1]);
            }
        } else if (op == 72 && count >= 4) {
            if (w[i + 3] == Flat) {
                flat_members.emplace_back(w[i + 1], w[i + 2]);
            } else if (w[i + 3] == BuiltIn) {
                builtin_members.emplace_back(w[i + 1], w[i + 2]);
            }
        }
        i += count;
    }
    for (const u32 id : flat_ids) {
        if (std::find(builtin_ids.begin(), builtin_ids.end(), id) == builtin_ids.end()) {
            return true;
        }
    }
    for (const auto& m : flat_members) {
        if (std::find(builtin_members.begin(), builtin_members.end(), m) == builtin_members.end()) {
            return true;
        }
    }
    return false;
}

void BuildGraphics(const Vulkan::GraphicsPipelineKey& key, const ModuleRefs& modules,
                   std::span<const bool> has_stage, std::span<const VertexInput> inputs,
                   std::span<const vk::PipelineColorBlendAttachmentState> blend,
                   bool aux_fragment, VkPipeline vk_pipeline, u64 id) {
    using Stage = ::Shader::SwStage;
    auto& s = Get();
    bool published = false;
    // (Not for tessellated draws, rect lists among them on the Vulkan side: the provoking
    // vertex does not decide their flat values.)
    bool flat_from_last = false;
    if (key.provoking_vtx_last != Gcn::ProvokingVtxLast::First &&
        has_stage[u32(Stage::Fragment)] && !has_stage[u32(Stage::TessellationControl)] &&
        !has_stage[u32(Stage::TessellationEval)] &&
        key.prim_type != Gcn::PrimitiveType::RectList) {
        const auto words = s.Words(modules[u32(Stage::Fragment)]);
        flat_from_last = !words || HasFlat(*words); // (unknown: as if it had)
    }
    const auto publish = [&](const GraphicsEntry& built) {
        GraphicsEntry entry = built;
        entry.flat_from_last = flat_from_last;
        entry.hashes = {u64(key.stage_hashes[u32(Stage::Vertex)]),
                        u64(key.stage_hashes[u32(Stage::TessellationControl)]),
                        u64(key.stage_hashes[u32(Stage::TessellationEval)]),
                        u64(key.stage_hashes[u32(Stage::Geometry)]),
                        u64(key.stage_hashes[u32(Stage::Fragment)])};
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
        // The guest's [-w, w] clip space: the Vulkan pipeline has VK_EXT_depth_clip_control for
        // it (KosmicKrisp has the extension, so the recompiler does not convert z itself:
        // Profile emulate_depth_negative_one_to_one); the Metal vertex function converts it.
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
                auto source = std::make_shared<VariantSource>();
                source->key = key;
                source->modules = modules;
                source->vs = vs;
                source->fs = fs;
                source->attributes = std::move(attributes);
                source->blend.assign(blend.begin(), blend.end());
                source->rect_list = rect_list;
                publish({.render = render, .rect_list = rect_list, .source = std::move(source)});
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
    using Stage = ::Shader::SwStage;
    const bool emulated = stages[u32(Stage::TessellationControl)] ||
                          stages[u32(Stage::TessellationEval)] || stages[u32(Stage::Geometry)];
    u64 id;
    {
        std::scoped_lock lock{s.pipelines_mutex};
        id = s.Expect(pipeline);
        if (emulated) {
            s.waitable.insert(pipeline);
        } else {
            s.waitable.erase(pipeline);
        }
    }
    s.Post(
        [key, refs, stages, input_list = std::move(input_list),
         blend_list = std::move(blend_list), aux_fragment, pipeline, id] {
            BuildGraphics(key, refs, stages, input_list, blend_list, aux_fragment, pipeline, id);
        },
        emulated);
}

const GraphicsEntry* FindGraphicsWaiting(VkPipeline pipeline, u32 timeout_ms) {
    auto& s = Get();
    std::unique_lock lock{s.pipelines_mutex};
    if (timeout_ms && s.waitable.contains(pipeline)) {
        s.built_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                            [&] { return !s.waitable.contains(pipeline); });
    }
    const auto it = s.graphics.find(pipeline);
    return it == s.graphics.end() ? nullptr : &it->second;
}

namespace {
/// FormatVariant's builds, by (base pipeline, depth format, stencil format); never dropped (the
/// pipeline cache keeps the pipelines too).
struct VariantKey {
    const Metal::RenderPipeline* base;
    u32 depth, stencil;
    bool operator==(const VariantKey&) const = default;
};
struct VariantKeyHash {
    size_t operator()(const VariantKey& k) const noexcept {
        return std::hash<const void*>{}(k.base) ^ (size_t(k.depth) * 0x9e3779b97f4a7c15ull) ^
               (size_t(k.stencil) << 17);
    }
};
struct VariantSlot {
    std::atomic<bool> done{};
    const Metal::RenderPipeline* pipeline{}; ///< written before done
};
/// Never destroyed (as State): the detached workers may still finish a build while the process
/// exits.
std::mutex& variants_mutex = *new std::mutex;
auto& variants = *new std::unordered_map<VariantKey, std::unique_ptr<VariantSlot>, VariantKeyHash>;
std::atomic<u32> variant_ok{0}, variant_failed{0};
} // namespace

const Metal::RenderPipeline* FormatVariant(const GraphicsEntry& entry, u32 depth, u32 stencil,
                                           bool& building) {
    building = false;
    if (!entry.render || !entry.source) {
        return nullptr;
    }
    VariantSlot* slot;
    bool post = false;
    {
        std::scoped_lock lock{variants_mutex};
        auto& owned = variants[{entry.render, depth, stencil}];
        if (!owned) {
            owned = std::make_unique<VariantSlot>();
            post = true;
        }
        slot = owned.get();
    }
    if (slot->done.load(std::memory_order_acquire)) {
        return slot->pipeline;
    }
    building = true;
    if (post) {
        Get().Post([source = entry.source, slot, depth, stencil] {
            auto& s = Get();
            const Metal::RenderPipeline* p = nullptr;
            try {
                p = s.pso.GetRenderPipeline(source->key, source->vs, source->fs,
                                            {.vertex_attributes = source->attributes,
                                             .blend = source->blend,
                                             .rect_list_triangles = source->rect_list,
                                             .formats_override = true,
                                             .depth_format = depth,
                                             .stencil_format = stencil});
            } catch (...) {
                p = nullptr;
            }
            slot->pipeline = p;
            slot->done.store(true, std::memory_order_release);
            const u32 n = ++(p ? variant_ok : variant_failed);
            if (n == 1 || n % 50 == 0) {
                Print("Metal shadow: depth/stencil format variants %u ok / %u failed\n",
                      variant_ok.load(), variant_failed.load());
            }
        });
    }
    return nullptr;
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

// ---- Generic pipelines (the renderer's own passes): built from their Vulkan create infos ----
// After Wooloomooloo2/bloodborne_mac_port's mtl_shadow.mm at 78b59f5 (GPL-2.0-or-later).

namespace {
thread_local int game_build_depth = 0;

/// Fixes the SPIR-V specialization constants at the values given (or their defaults) and turns
/// them into plain constants, so SPIRV-Cross emits no Metal function constants (a pipeline state
/// would then need a specialized function): OpSpecConstant's literal, OpSpecConstantTrue/False by
/// value, the SpecId decorations dropped. OpSpecConstantOp stays (none in the host shaders).
std::vector<u32> Specialize(const std::vector<u32>& words, const VkSpecializationInfo* info) {
    if (words.size() < 5) {
        return words;
    }
    std::unordered_map<u32, u32> spec_ids; // result id -> SpecId
    for (size_t i = 5; i < words.size();) {
        const u32 count = words[i] >> 16, op = words[i] & 0xffff;
        if (!count || i + count > words.size()) {
            break;
        }
        if (op == 71 && count >= 4 && words[i + 2] == 1) { // OpDecorate id SpecId n
            spec_ids[words[i + 1]] = words[i + 3];
        }
        i += count;
    }
    std::vector<u32> out(words.begin(), words.begin() + 5);
    out.reserve(words.size());
    for (size_t i = 5; i < words.size();) {
        const u32 count = words[i] >> 16, op = words[i] & 0xffff;
        if (!count || i + count > words.size()) {
            break;
        }
        if (op == 71 && count >= 4 && words[i + 2] == 1) {
            i += count; // SpecId on what is now a plain constant
            continue;
        }
        const size_t at = out.size();
        out.insert(out.end(), words.begin() + i, words.begin() + i + count);
        // OpSpecConstantTrue 48, False 49, OpSpecConstant 50, Composite 51 -> 41, 42, 43, 44
        if ((op == 48 || op == 49 || op == 50 || op == 51) && count >= 3) {
            u32 new_op = op == 48 ? 41u : op == 49 ? 42u : op == 50 ? 43u : 44u;
            const auto it = spec_ids.find(words[i + 2]);
            if (info && it != spec_ids.end()) {
                for (u32 e = 0; e < info->mapEntryCount; ++e) {
                    const auto& entry = info->pMapEntries[e];
                    if (entry.constantID != it->second ||
                        size_t(entry.offset) + entry.size > info->dataSize) {
                        continue;
                    }
                    const u8* data = static_cast<const u8*>(info->pData) + entry.offset;
                    if (op == 50) {
                        for (u32 w = 0; w + 3 < count && w * 4 < entry.size; ++w) {
                            u32 v = 0;
                            std::memcpy(&v, data + w * 4, std::min<size_t>(4, entry.size - w * 4));
                            out[at + 3 + w] = v;
                        }
                    } else if (op == 48 || op == 49) {
                        bool value = false;
                        for (size_t b = 0; b < entry.size; ++b) {
                            value |= data[b] != 0;
                        }
                        new_op = value ? 41u : 42u;
                    }
                }
            }
            out[at] = (count << 16) | new_op;
        }
        i += count;
    }
    return out;
}

std::mutex generic_mutex;
/// By the specialized SPIR-V's hash: translated once, kept (pipeline states point at them).
std::unordered_map<u64, std::unique_ptr<Metal::CompiledShader>> generic_shaders;
std::vector<std::unique_ptr<Metal::RenderPipeline>> generic_pipelines;
std::atomic<u32> generic_ok{0}, generic_failed{0};

const Metal::CompiledShader* CompileStage(State& s, const VkPipelineShaderStageCreateInfo& stage,
                                          const char* what) {
    std::shared_ptr<ShaderEntry> entry = s.Lookup(vk::ShaderModule(stage.module));
    if (!entry) {
        ++s.no_spirv;
        return nullptr;
    }
    const std::vector<u32> spec = Specialize(*entry->words, stage.pSpecializationInfo);
    const u64 hash = XXH3_64bits(spec.data(), spec.size() * sizeof(u32));
    {
        std::scoped_lock lock{generic_mutex};
        if (const auto it = generic_shaders.find(hash); it != generic_shaders.end()) {
            return it->second.get();
        }
    }
    auto result = s.compiler.Compile(spec, hash, what);
    std::scoped_lock lock{generic_mutex};
    auto& slot = generic_shaders[hash];
    if (!slot && result.status == Metal::CompileStatus::Ok) {
        slot = std::move(result.shader);
    }
    return slot.get();
}

void NoteGeneric(bool ok, const std::string& why) {
    const u32 n = ++(ok ? generic_ok : generic_failed);
    if (!ok && n <= 20) {
        Print("Metal shadow: a pipeline of the renderer's own not built in Metal: %s\n",
              why.c_str());
    }
    if (ok && (n == 1 || n % 25 == 0)) {
        Print("Metal shadow: %u pipelines of the renderer's own built in Metal, %u not\n", n,
              generic_failed.load());
    }
}

bool GenericWanted(VkPipeline pipeline) {
    static const bool off = [] {
        const char* env = std::getenv("BB_METAL_GENERIC");
        const bool no = env && env[0] == '0';
        if (no) {
            Print("Metal shadow: the renderer's own pipelines stay on KosmicKrisp "
                  "(BB_METAL_GENERIC=0)\n");
        }
        return no;
    }();
    return pipeline && !game_build_depth && !off && Enabled();
}
} // namespace

GameBuildScope::GameBuildScope() {
    ++game_build_depth;
}
GameBuildScope::~GameBuildScope() {
    --game_build_depth;
}

void ModuleCreated(VkShaderModule module, const VkShaderModuleCreateInfo* info) {
    if (!Enabled() || !module || !info || !info->pCode) {
        return;
    }
    RegisterModule(vk::ShaderModule(module),
                   std::span<const u32>(info->pCode, info->codeSize / sizeof(u32)));
}

void PipelineDestroyed(VkPipeline pipeline) {
    Forget(vk::Pipeline(pipeline));
}

void ComputeCreated(const VkComputePipelineCreateInfo* info, VkPipeline pipeline) {
    if (!GenericWanted(pipeline)) {
        return;
    }
    auto& s = Get();
    u64 build;
    {
        std::scoped_lock lock{s.pipelines_mutex};
        build = s.Expect(pipeline);
    }
    // A work-group size given by specialization constants (the WorkgroupSize built-in, or
    // LocalSizeId): the translation reads only the literal LocalSize. Left to KosmicKrisp.
    bool spec_size = false;
    if (const auto entry = s.Lookup(vk::ShaderModule(info->stage.module))) {
        const auto& w = *entry->words;
        for (size_t i = 5; i < w.size();) {
            const u32 count = w[i] >> 16, op = w[i] & 0xffff;
            if (!count || i + count > w.size()) {
                break;
            }
            spec_size |= op == 71 && count >= 4 && w[i + 2] == 11 && w[i + 3] == 25; // BuiltIn
            spec_size |= op == 331;                                                 // LocalSizeId
            i += count;
        }
    }
    if (spec_size) {
        s.Abandon(pipeline, build);
        NoteGeneric(false, "a compute work-group size from specialization constants");
        return;
    }
    const auto* cs = CompileStage(s, info->stage, "host compute");
    const Metal::ComputePipeline* p = nullptr;
    if (cs) {
        Vulkan::ComputePipelineKey key{};
        key.value = size_t(XXH3_64bits(&cs, sizeof cs)); // one pipeline per translated shader
        p = s.pso.GetComputePipeline(key, cs);
    }
    if (p) {
        s.PublishCompute(pipeline, build, p);
    } else {
        s.Abandon(pipeline, build);
    }
    NoteGeneric(p != nullptr, cs ? "its compute pipeline state" : "its compute shader");
}

void GraphicsCreated(const VkGraphicsPipelineCreateInfo* info, VkPipeline pipeline) {
    if (!GenericWanted(pipeline)) {
        return;
    }
    using namespace Metal::VkToMtl;
    auto& s = Get();
    u64 build;
    {
        std::scoped_lock lock{s.pipelines_mutex};
        build = s.Expect(pipeline);
    }
    const auto fail = [&](const std::string& why) {
        s.Abandon(pipeline, build);
        NoteGeneric(false, why);
    };
    const Metal::CompiledShader* vs = nullptr;
    const Metal::CompiledShader* fs = nullptr;
    for (u32 i = 0; i < info->stageCount; ++i) {
        const auto& stage = info->pStages[i];
        if (stage.stage == VK_SHADER_STAGE_VERTEX_BIT) {
            vs = CompileStage(s, stage, "host vertex");
        } else if (stage.stage == VK_SHADER_STAGE_FRAGMENT_BIT) {
            fs = CompileStage(s, stage, "host fragment");
            if (!fs) {
                return fail("its fragment shader");
            }
        } else {
            return fail("a stage other than vertex and fragment");
        }
    }
    if (!vs) {
        return fail("its vertex shader");
    }
    auto entry = std::make_unique<Metal::RenderPipeline>();
    GraphicsEntry ge;
    ge.generic = true;
    @autoreleasepool {
        MTLRenderPipelineDescriptor* d = [MTLRenderPipelineDescriptor new];
        d.vertexFunction = (__bridge id<MTLFunction>)vs->function;
        if (fs) {
            d.fragmentFunction = (__bridge id<MTLFunction>)fs->function;
        }
        const VkPipelineRenderingCreateInfo* rendering = nullptr;
        for (auto* n = static_cast<const VkBaseInStructure*>(info->pNext); n; n = n->pNext) {
            if (n->sType == VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO) {
                rendering = reinterpret_cast<const VkPipelineRenderingCreateInfo*>(n);
            }
        }
        if (!rendering) {
            return fail("no dynamic rendering formats (a render pass object)");
        }
        const auto* blend = info->pColorBlendState;
        if (rendering->colorAttachmentCount > entry->color_pixel_formats.size()) {
            return fail("more color attachments than Metal has");
        }
        for (u32 i = 0; i < rendering->colorAttachmentCount; ++i) {
            const VkFormat vk_format = rendering->pColorAttachmentFormats[i];
            if (vk_format == VK_FORMAT_UNDEFINED) {
                continue;
            }
            const u32 format = PixelFormat(vk::Format(vk_format));
            if (!format || IsDepthPixelFormat(format)) {
                return fail("color attachment " + std::to_string(i) + " format " +
                            vk::to_string(vk::Format(vk_format)));
            }
            if (!(FormatSwizzle(vk::Format(vk_format)) == IdentitySwizzle)) {
                // Stored swizzled: the frame encoder renders only into identity formats.
                return fail("color attachment " + std::to_string(i) + " stored swizzled");
            }
            entry->color_pixel_formats[i] = format;
            d.colorAttachments[i].pixelFormat = MTLPixelFormat(format);
            if (blend && i < blend->attachmentCount) {
                const auto& a = blend->pAttachments[i];
                entry->key.write_masks[i] = vk::ColorComponentFlags(a.colorWriteMask);
                d.colorAttachments[i].writeMask =
                    MTLColorWriteMask(ColorWriteMask(vk::ColorComponentFlags(a.colorWriteMask)));
                d.colorAttachments[i].blendingEnabled = a.blendEnable;
                if (a.blendEnable) {
                    d.colorAttachments[i].sourceRGBBlendFactor =
                        MTLBlendFactor(BlendFactor(vk::BlendFactor(a.srcColorBlendFactor)));
                    d.colorAttachments[i].destinationRGBBlendFactor =
                        MTLBlendFactor(BlendFactor(vk::BlendFactor(a.dstColorBlendFactor)));
                    d.colorAttachments[i].rgbBlendOperation =
                        MTLBlendOperation(BlendOp(vk::BlendOp(a.colorBlendOp)));
                    d.colorAttachments[i].sourceAlphaBlendFactor =
                        MTLBlendFactor(BlendFactor(vk::BlendFactor(a.srcAlphaBlendFactor)));
                    d.colorAttachments[i].destinationAlphaBlendFactor =
                        MTLBlendFactor(BlendFactor(vk::BlendFactor(a.dstAlphaBlendFactor)));
                    d.colorAttachments[i].alphaBlendOperation =
                        MTLBlendOperation(BlendOp(vk::BlendOp(a.alphaBlendOp)));
                }
            } else {
                entry->key.write_masks[i] = vk::ColorComponentFlags(0xf);
            }
        }
        entry->num_color_attachments = rendering->colorAttachmentCount;
        if (rendering->depthAttachmentFormat) {
            entry->depth_pixel_format = PixelFormat(vk::Format(rendering->depthAttachmentFormat));
            d.depthAttachmentPixelFormat = MTLPixelFormat(entry->depth_pixel_format);
        }
        if (rendering->stencilAttachmentFormat) {
            entry->stencil_pixel_format =
                PixelFormat(vk::Format(rendering->stencilAttachmentFormat));
            d.stencilAttachmentPixelFormat = MTLPixelFormat(entry->stencil_pixel_format);
        }
        entry->sample_count = 1;
        if (info->pMultisampleState) {
            d.rasterSampleCount = info->pMultisampleState->rasterizationSamples;
            entry->sample_count = info->pMultisampleState->rasterizationSamples;
        }
        // Vertex input: binding b at buffer slot kVertexBufferBase - b, dynamic stride.
        if (const auto* vi = info->pVertexInputState; vi && vi->vertexAttributeDescriptionCount) {
            MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
            for (u32 i = 0; i < vi->vertexAttributeDescriptionCount; ++i) {
                const auto& a = vi->pVertexAttributeDescriptions[i];
                const u32 format = VertexFormat(vk::Format(a.format));
                if (!format || a.binding >= Metal::kMaxVertexBuffers || a.location >= 31) {
                    return fail("vertex attribute format " + vk::to_string(vk::Format(a.format)));
                }
                vd.attributes[a.location].format = MTLVertexFormat(format);
                vd.attributes[a.location].offset = a.offset;
                vd.attributes[a.location].bufferIndex = Metal::kVertexBufferBase - a.binding;
            }
            for (u32 i = 0; i < vi->vertexBindingDescriptionCount; ++i) {
                const auto& b = vi->pVertexBindingDescriptions[i];
                if (b.binding >= Metal::kMaxVertexBuffers) {
                    return fail("a vertex binding beyond Metal's slots");
                }
                const u32 slot = Metal::kVertexBufferBase - b.binding;
                vd.layouts[slot].stride = MTLBufferLayoutStrideDynamic;
                vd.layouts[slot].stepFunction = b.inputRate == VK_VERTEX_INPUT_RATE_INSTANCE
                                                    ? MTLVertexStepFunctionPerInstance
                                                    : MTLVertexStepFunctionPerVertex;
                ge.strides[b.binding] = b.stride;
                auto& vslot = entry->vertex_slots[entry->num_vertex_slots++];
                vslot.location = b.binding;
                vslot.slot = slot;
                for (auto* n = static_cast<const VkBaseInStructure*>(vi->pNext); n; n = n->pNext) {
                    if (n->sType != VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_DIVISOR_STATE_CREATE_INFO) {
                        continue;
                    }
                    const auto* div =
                        reinterpret_cast<const VkPipelineVertexInputDivisorStateCreateInfo*>(n);
                    for (u32 k = 0; k < div->vertexBindingDivisorCount; ++k) {
                        const auto& bd = div->pVertexBindingDivisors[k];
                        if (bd.binding != b.binding) {
                            continue;
                        }
                        if (bd.divisor == 0) {
                            vd.layouts[slot].stepFunction = MTLVertexStepFunctionConstant;
                            vd.layouts[slot].stepRate = 0;
                            vslot.constant = true;
                        } else {
                            vd.layouts[slot].stepRate = bd.divisor;
                        }
                    }
                }
            }
            d.vertexDescriptor = vd;
        }
        const auto topology =
            info->pInputAssemblyState
                ? PrimitiveType(vk::PrimitiveTopology(info->pInputAssemblyState->topology))
                : PrimitiveType(vk::PrimitiveTopology::eTriangleList);
        entry->primitive_type = topology.type;
        entry->primitive_emulated = topology.needs_emulation;
        d.inputPrimitiveTopology = topology.type == MTLPrimitiveTypePoint ? MTLPrimitiveTopologyClassPoint
                                   : topology.type == MTLPrimitiveTypeLine ||
                                           topology.type == MTLPrimitiveTypeLineStrip
                                       ? MTLPrimitiveTopologyClassLine
                                       : MTLPrimitiveTopologyClassTriangle;
        NSError* error = nil;
        id<MTLRenderPipelineState> state =
            [s.device.GetImpl().device newRenderPipelineStateWithDescriptor:d error:&error];
        if (!state) {
            return fail(std::string("Metal: ") +
                        (error ? error.localizedDescription.UTF8String : "unknown error"));
        }
        entry->state = (__bridge_retained void*)state;
        entry->vertex_function = (__bridge_retained void*)d.vertexFunction;
        if (fs) {
            entry->fragment_function = (__bridge_retained void*)d.fragmentFunction;
        }
    }
    entry->vs = vs;
    entry->fs = fs;
    entry->fill_mode = MTLTriangleFillModeFill;
    entry->depth_clip = true;
    // The state Vulkan fixes unless it is dynamic.
    std::vector<VkDynamicState> dynamic;
    if (info->pDynamicState) {
        dynamic.assign(info->pDynamicState->pDynamicStates,
                       info->pDynamicState->pDynamicStates + info->pDynamicState->dynamicStateCount);
    }
    const auto is_dynamic = [&](VkDynamicState d) {
        return std::find(dynamic.begin(), dynamic.end(), d) != dynamic.end();
    };
    if (const auto* r = info->pRasterizationState) {
        ge.fixed.cull = !is_dynamic(VK_DYNAMIC_STATE_CULL_MODE);
        ge.fixed.cull_mode = r->cullMode;
        ge.fixed.front_face = !is_dynamic(VK_DYNAMIC_STATE_FRONT_FACE);
        ge.fixed.front = r->frontFace;
        ge.fixed.depth_bias = !is_dynamic(VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE);
        ge.fixed.depth_bias_enable = r->depthBiasEnable;
        entry->depth_clip = !r->depthClampEnable;
        if (r->polygonMode == VK_POLYGON_MODE_LINE) {
            entry->fill_mode = MTLTriangleFillModeLines;
        } else if (r->polygonMode == VK_POLYGON_MODE_POINT) {
            entry->fill_point = true; // Metal has none: its draws fall back
        }
    }
    ge.fixed.write_masks = !is_dynamic(VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT);
    ge.fixed.strides = !is_dynamic(VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE);
    if (const auto* cb = info->pColorBlendState; cb && !is_dynamic(VK_DYNAMIC_STATE_BLEND_CONSTANTS)) {
        ge.fixed.blend_constants_fixed = true;
        std::copy_n(cb->blendConstants, 4, ge.fixed.blend_constants);
    }
    if (const auto* ds = info->pDepthStencilState) {
        ge.fixed.depth = !is_dynamic(VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE);
        ge.fixed.depth_write_fixed = !is_dynamic(VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE);
        ge.fixed.depth_compare_fixed = !is_dynamic(VK_DYNAMIC_STATE_DEPTH_COMPARE_OP);
        ge.fixed.depth_test = ds->depthTestEnable;
        ge.fixed.depth_write = ds->depthWriteEnable;
        ge.fixed.depth_compare = ds->depthCompareOp;
        ge.fixed.stencil = !is_dynamic(VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE);
        ge.fixed.stencil_test = ds->stencilTestEnable;
        ge.fixed.stencil_op = !is_dynamic(VK_DYNAMIC_STATE_STENCIL_OP);
        ge.fixed.stencil_compare_mask = !is_dynamic(VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK);
        ge.fixed.stencil_write_mask = !is_dynamic(VK_DYNAMIC_STATE_STENCIL_WRITE_MASK);
        ge.fixed.stencil_reference = !is_dynamic(VK_DYNAMIC_STATE_STENCIL_REFERENCE);
        const auto face = [](const VkStencilOpState& f) {
            return FixedState::Face{f.failOp,      f.passOp,    f.depthFailOp, f.compareOp,
                                    f.compareMask, f.writeMask, f.reference};
        };
        ge.fixed.stencil_front = face(ds->front);
        ge.fixed.stencil_back = face(ds->back);
        ge.fixed.depth_bounds = !is_dynamic(VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE);
        ge.fixed.depth_bounds_test = ds->depthBoundsTestEnable;
        ge.fixed.depth_bounds_values = !is_dynamic(VK_DYNAMIC_STATE_DEPTH_BOUNDS);
        ge.fixed.min_depth_bounds = ds->minDepthBounds;
        ge.fixed.max_depth_bounds = ds->maxDepthBounds;
    } else {
        // No depth/stencil state: depth, stencil and depth bounds tests all off.
        ge.fixed.depth = ge.fixed.depth_write_fixed = ge.fixed.depth_compare_fixed = true;
        ge.fixed.stencil = ge.fixed.depth_bounds = true;
    }
    ge.render = entry.get();
    {
        std::scoped_lock lock{generic_mutex};
        generic_pipelines.push_back(std::move(entry));
    }
    s.PublishGraphics(pipeline, build, ge);
    NoteGeneric(true, "");
}

} // namespace MetalShadow
