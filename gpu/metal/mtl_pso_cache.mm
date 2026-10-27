// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: from Huax-1839/bloodborne-macos (ee3120a, GPL-2.0-or-later), renderer_metal/.
// Taken from Wooloomooloo2/bloodborne_mac_port at 78b59f5, its includes and names adapted.

#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <xxhash.h>

#include "common/assert.h"
#include "common/logging/log.h"
#include "metal/mtl_device_impl.h"
#include "metal/mtl_pso_cache.h"
#include "metal/mtl_tessellation.h"
#include "metal/vk_to_mtl.h"
#include "render/gcn_to_vk.h"

namespace Metal {

namespace {

using Clock = std::chrono::steady_clock;

u64 UsSince(Clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
}

std::string FirstLine(NSError* error) {
    if (!error) {
        return "unknown error";
    }
    std::string text = error.localizedDescription.UTF8String;
    const auto end = text.find('\n');
    return end == std::string::npos ? text : text.substr(0, end);
}


/// The attributes field by field (VertexAttribute has padding after `per_instance`).
u64 HashAttributes(std::span<const VertexAttribute> attributes) {
    u64 h = attributes.size();
    for (const VertexAttribute& a : attributes) {
        const u64 fields[3] = {a.location, a.per_instance ? 1u : 0u, a.divisor};
        h = XXH3_64bits_withSeed(fields, sizeof(fields), h);
    }
    return h;
}

/// A vertex attribute's format as the Vulkan pipeline uses it: the key keeps Undefined for a V#
/// whose format is impossible, which GraphicsPipeline::GetVertexInputs reads as 32-bit floats.
vk::Format VertexBufferFormat(const Vulkan::GraphicsPipelineKey& key, u32 i) {
    const vk::Format stored = key.vertex_buffer_formats[i];
    return stored == vk::Format::eUndefined ? vk::Format::eR32G32B32A32Sfloat : stored;
}

struct RenderMapKey {
    Vulkan::GraphicsPipelineKey key;
    const void* vs{}; ///< the MTLFunction objects (the entry retains them: no address reuse)
    const void* fs{};
    u64 attributes_hash{};
    bool rect_list_triangles{};
    bool tessellation{};
    bool geometry{};
    u64 formats{}; ///< bbport: RenderPipelineInput's override (flag, depth, stencil), 0: none

    bool operator==(const RenderMapKey& other) const noexcept {
        return vs == other.vs && fs == other.fs && attributes_hash == other.attributes_hash &&
               formats == other.formats &&
               rect_list_triangles == other.rect_list_triangles &&
               tessellation == other.tessellation && geometry == other.geometry &&
               key == other.key;
    }
};

struct RenderMapHash {
    size_t operator()(const RenderMapKey& k) const noexcept {
        u64 h = XXH3_64bits(&k.key, sizeof(k.key));
        const u64 rest[5] = {reinterpret_cast<u64>(k.vs), reinterpret_cast<u64>(k.fs),
                             k.attributes_hash,
                             u64(k.rect_list_triangles) | (u64(k.tessellation) << 1) | (u64(k.geometry) << 2),
                             k.formats};
        return XXH3_64bits_withSeed(rest, sizeof(rest), h);
    }
};

/// A tessellation pipeline: the stages' functions name the compiled variants.
struct TessMapKey {
    Vulkan::GraphicsPipelineKey key;
    const void* vertex{}; ///< the vertex kernel's function without an index buffer
    const void* control{};
    const void* eval{};
    const void* fs{};
    u64 attributes_hash{};

    bool operator==(const TessMapKey& other) const noexcept {
        return vertex == other.vertex && control == other.control && eval == other.eval &&
               fs == other.fs && attributes_hash == other.attributes_hash && key == other.key;
    }
};

struct TessMapHash {
    size_t operator()(const TessMapKey& k) const noexcept {
        u64 h = XXH3_64bits(&k.key, sizeof(k.key));
        const u64 rest[5] = {reinterpret_cast<u64>(k.vertex), reinterpret_cast<u64>(k.control),
                             reinterpret_cast<u64>(k.eval), reinterpret_cast<u64>(k.fs),
                             k.attributes_hash};
        return XXH3_64bits_withSeed(rest, sizeof(rest), h);
    }
};

/// A geometry pipeline: the stages' functions name the compiled variants.
struct GeometryMapKey {
    Vulkan::GraphicsPipelineKey key;
    const void* es{}; ///< the export kernel's function without an index buffer
    const void* gs{};
    const void* pass_through{};
    const void* fs{};
    u64 attributes_hash{};

    bool operator==(const GeometryMapKey& other) const noexcept {
        return es == other.es && gs == other.gs && pass_through == other.pass_through &&
               fs == other.fs && attributes_hash == other.attributes_hash && key == other.key;
    }
};

struct GeometryMapHash {
    size_t operator()(const GeometryMapKey& k) const noexcept {
        u64 h = XXH3_64bits(&k.key, sizeof(k.key));
        const u64 rest[5] = {reinterpret_cast<u64>(k.es), reinterpret_cast<u64>(k.gs),
                             reinterpret_cast<u64>(k.pass_through), reinterpret_cast<u64>(k.fs),
                             k.attributes_hash};
        return XXH3_64bits_withSeed(rest, sizeof(rest), h);
    }
};

struct ComputeMapKey {
    size_t value{};
    const void* function{};
    bool operator==(const ComputeMapKey&) const = default;
};

struct ComputeMapHash {
    size_t operator()(const ComputeMapKey& k) const noexcept {
        const u64 data[2] = {k.value, reinterpret_cast<u64>(k.function)};
        return XXH3_64bits(data, sizeof(data));
    }
};

struct DepthStencilHash {
    size_t operator()(const DepthStencilKey& k) const noexcept {
        const auto face = [](const DepthStencilKey::Face& f) {
            return (u64(f.fail_op) << 0) | (u64(f.pass_op) << 4) | (u64(f.depth_fail_op) << 8) |
                   (u64(f.compare_op) << 12) | (u64(f.compare_mask) << 16) |
                   (u64(f.write_mask) << 24);
        };
        const u64 data[3] = {u64(k.depth_test) | (u64(k.depth_write) << 1) |
                                 (u64(k.depth_compare) << 2) | (u64(k.stencil_test) << 6),
                             face(k.front), face(k.back)};
        return XXH3_64bits(data, sizeof(data));
    }
};

std::string StageHashes(const Vulkan::GraphicsPipelineKey& key) {
    return fmt::format("vs {:#x} fs {:#x}", key.stage_hashes[u32(Shader::SwStage::Vertex)],
                       key.stage_hashes[u32(Shader::SwStage::Fragment)]);
}

} // namespace

struct PsoCache::Impl {
    id<MTLDevice> mtl_device;

    std::shared_mutex mutex; ///< guards the three maps
    std::unordered_map<RenderMapKey, std::unique_ptr<RenderPipeline>, RenderMapHash> render;
    std::unordered_map<ComputeMapKey, std::unique_ptr<ComputePipeline>, ComputeMapHash> compute;
    std::unordered_map<TessMapKey, std::unique_ptr<TessellationPipeline>, TessMapHash> tessellation;
    std::unordered_map<GeometryMapKey, std::unique_ptr<GeometryPipeline>, GeometryMapHash> geometry;
    std::unordered_map<DepthStencilKey, id<MTLDepthStencilState>, DepthStencilHash> depth_stencil;

    std::mutex debug_mutex; ///< BB_GS_DEBUG: the magenta fragment functions by source
    std::map<std::string, std::unique_ptr<CompiledShader>> debug_fragments;

    std::mutex archive_mutex; ///< MTLBinaryArchive is not documented as thread-safe
    id<MTLBinaryArchive> archive; ///< nil: none (yet / any more)
    bool archive_dirty{};
    bool archive_error_logged{};
};

RenderPipeline::~RenderPipeline() {
    // Balance the +1 of __bridge_retained; ARC releases at the end of the scope.
    for (void* object : {state, vertex_function, fragment_function}) {
        if (object) {
            id released = (__bridge_transfer id)object;
            (void)released;
        }
    }
}

TessKernel::~TessKernel() {
    for (void* object : {state, function}) {
        if (object) {
            id released = (__bridge_transfer id)object;
            (void)released;
        }
    }
}

ComputePipeline::~ComputePipeline() {
    for (void* object : {state, function}) {
        if (object) {
            id released = (__bridge_transfer id)object;
            (void)released;
        }
    }
}

PsoCache::PsoCache(Device& device_, std::filesystem::path cache_dir)
    : device{device_}, impl{std::make_unique<Impl>()} {
    impl->mtl_device = device.GetImpl().device;
    if (cache_dir.empty()) {
        cache_dir = ShaderCompiler::DefaultCacheDir();
    }
    archive_path = cache_dir / "pipelines.binaryarchive";

    @autoreleasepool {
        std::error_code ec;
        std::filesystem::create_directories(cache_dir, ec);
        const bool exists = std::filesystem::exists(archive_path, ec);
        MTLBinaryArchiveDescriptor* descriptor = [MTLBinaryArchiveDescriptor new];
        if (exists) {
            descriptor.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:archive_path.c_str()]];
        }
        NSError* error = nil;
        id<MTLBinaryArchive> archive =
            [impl->mtl_device newBinaryArchiveWithDescriptor:descriptor error:&error];
        if (archive && exists) {
            archive_loaded = true;
        } else if (!archive && exists) {
            // Another device or OS build, or a damaged file: start over.
            LOG_WARNING(Render, "Metal binary archive {} unusable ({}), starting a new one",
                        archive_path.string(), FirstLine(error));
            std::filesystem::remove(archive_path, ec);
            descriptor.url = nil;
            error = nil;
            archive = [impl->mtl_device newBinaryArchiveWithDescriptor:descriptor error:&error];
        }
        if (!archive) {
            LOG_WARNING(Render, "Metal binary archive not available ({}): pipelines are compiled "
                                "on every start",
                        FirstLine(error));
        }
        impl->archive = archive;
    }
}

PsoCache::~PsoCache() {
    Flush();
}

bool PsoCache::Flush() {
    std::scoped_lock lock{impl->archive_mutex};
    if (!impl->archive || !impl->archive_dirty) {
        return true;
    }
    @autoreleasepool {
        std::error_code ec;
        const std::filesystem::path temp = archive_path.string() + ".tmp";
        std::filesystem::remove(temp, ec);
        NSError* error = nil;
        const BOOL ok = [impl->archive
            serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:temp.c_str()]]
                     error:&error];
        if (ok) {
            std::filesystem::rename(temp, archive_path, ec);
        }
        if (!ok || ec) {
            if (!impl->archive_error_logged) {
                impl->archive_error_logged = true;
                LOG_WARNING(Render, "Metal binary archive {} could not be written: {}",
                            archive_path.string(), ok ? ec.message() : FirstLine(error));
            }
            std::filesystem::remove(temp, ec);
            return false;
        }
    }
    impl->archive_dirty = false;
    return true;
}

namespace {

/// Makes `archive`'s handling of one descriptor uniform for render and compute: a lookup with
/// FailOnBinaryArchiveMiss first (an archive hit skips the compiler), the regular creation
/// otherwise, then the descriptor is added to the archive.
/// bbport: `archivable` false keeps the descriptor out of the archive: a render pipeline without
/// a fragment function adds without an error, but then the whole archive fails to serialize
/// ("expecting 'fragment' stage in pipeline no. N"), so nothing was ever saved for the next run.
template <typename Descriptor, typename Create, typename Add>
id CreateWithArchive(PsoCache::Impl& impl, bool archive_loaded, Descriptor* descriptor,
                     Create create, Add add, bool& from_archive, std::string& error,
                     bool archivable = true) {
    id<MTLBinaryArchive> archive = archivable ? impl.archive : nil;
    from_archive = false;
    NSError* ns_error = nil;
    if (archive) {
        descriptor.binaryArchives = @[ archive ];
        if (archive_loaded) {
            id state = create(MTLPipelineOptionFailOnBinaryArchiveMiss, &ns_error);
            if (state) {
                from_archive = true;
                return state;
            }
            ns_error = nil;
        }
    }
    id state = create(MTLPipelineOptionNone, &ns_error);
    if (!state) {
        error = FirstLine(ns_error);
        return nil;
    }
    if (archive) {
        std::scoped_lock lock{impl.archive_mutex};
        NSError* add_error = nil;
        if (add(archive, descriptor, &add_error)) {
            impl.archive_dirty = true;
        } else if (!impl.archive_error_logged) {
            impl.archive_error_logged = true;
            LOG_WARNING(Render, "Metal binary archive: adding a pipeline failed ({}), continuing "
                                "without further additions",
                        FirstLine(add_error));
        }
    }
    return state;
}

} // namespace

const RenderPipeline* PsoCache::GetRenderPipeline(const Vulkan::GraphicsPipelineKey& key,
                                                       const CompiledShader* vs,
                                                       const CompiledShader* fs,
                                                       const RenderPipelineInput& input) {
    ASSERT(vs && vs->function);
    const RenderMapKey map_key{
        .key = key,
        .vs = vs->function,
        .fs = fs ? fs->function : nullptr,
        .attributes_hash = HashAttributes(input.vertex_attributes),
        .rect_list_triangles = input.rect_list_triangles,
        .tessellation = input.tessellation_eval,
        .geometry = input.geometry,
        .formats = input.formats_override
                       ? (1ull << 63) | (u64(input.depth_format) << 32) | input.stencil_format
                       : 0,
    };
    {
        std::shared_lock lock{impl->mutex};
        const auto it = impl->render.find(map_key);
        if (it != impl->render.end()) {
            return it->second->state ? it->second.get() : nullptr;
        }
    }

    namespace L = Vulkan::GcnToVk;
    auto pipeline = std::make_unique<RenderPipeline>();
    pipeline->key = key;
    pipeline->vs = vs;
    pipeline->fs = fs;
    std::string error;
    bool from_archive = false;
    u64 elapsed_us = 0;
    @autoreleasepool {
        do {
            // ---- primitive and encoder state ----
            const auto topology = L::PrimitiveType(key.prim_type);
            VkToMtl::PrimitiveInfo primitive = VkToMtl::PrimitiveType(topology);
            if (input.rect_list_triangles) {
                // The rewritten vertex shader emits the two triangles of each rectangle.
                primitive = {MTLPrimitiveTypeTriangle, false};
            }
            const bool tessellation = input.tessellation_eval;
            if (tessellation) {
                // `vs` is the evaluation shader: the tessellator makes the triangles, drawPatches
                // is the draw (the guest's patch list is not an emulated primitive).
                if (vs->variant.kind != CompileVariant::Kind::TessEval) {
                    error = "the vertex stage of a tessellation pipeline is not an evaluation shader";
                    break;
                }
                primitive = {MTLPrimitiveTypeTriangle, false};
                pipeline->tessellation = true;
                pipeline->patch_control_points = vs->variant.output_vertices;
            }
            if (input.geometry) {
                // `vs` is the pass-through function: it draws the strips of the geometry kernel.
                primitive = {static_cast<u32>(input.geometry_output == GsOutputPrimitive::LineStrip
                                                  ? MTLPrimitiveTypeLineStrip
                                                  : MTLPrimitiveTypeTriangleStrip),
                             false};
                pipeline->geometry = true;
            }
            pipeline->primitive_type = primitive.type;
            pipeline->primitive_emulated = primitive.needs_emulation;
            pipeline->rect_list = key.prim_type == Gcn::PrimitiveType::RectList;
            pipeline->quad_list = key.prim_type == Gcn::PrimitiveType::QuadList;
            const VkToMtl::FillInfo fill = VkToMtl::PolygonMode(L::PolygonMode(key.polygon_mode));
            pipeline->fill_mode = fill.mode;
            pipeline->fill_point = fill.needs_emulation;
            pipeline->depth_clip = key.depth_clip_enable;
            pipeline->provoking_vertex_last =
                key.provoking_vtx_last != Gcn::ProvokingVtxLast::First;
            pipeline->clip_minus_w_to_w = key.clip_space == Gcn::ClipSpace::MinusWToW;
            pipeline->clip_fixed = vs->variant.clip_fixup;
            pipeline->num_color_attachments = key.num_color_attachments;

            MTLRenderPipelineDescriptor* desc = [MTLRenderPipelineDescriptor new];
            desc.label = [NSString stringWithUTF8String:StageHashes(key).c_str()];
            desc.vertexFunction = (__bridge id<MTLFunction>)vs->function;
            if (pipeline->fs) {
                desc.fragmentFunction = (__bridge id<MTLFunction>)fs->function;
            }
            if (tessellation) {
                // The tessellator's state comes from the evaluation shader's execution modes:
                // equal / fractional spacing, the order of the triangles it makes, a half
                // precision factor per patch (the control kernel's structs).
                const TessellationInfo& modes = vs->tess;
                desc.tessellationPartitionMode = static_cast<MTLTessellationPartitionMode>(
                    Tessellation::PartitionMode(modes.spacing));
                desc.tessellationFactorFormat = MTLTessellationFactorFormatHalf;
                desc.tessellationFactorStepFunction = MTLTessellationFactorStepFunctionPerPatch;
                desc.tessellationControlPointIndexType = MTLTessellationControlPointIndexTypeNone;
                desc.tessellationOutputWindingOrder = static_cast<MTLWinding>(
                    Tessellation::OutputWinding(modes.clockwise));
                desc.maxTessellationFactor = Tessellation::kMaxTessellationFactor;
            } else {
                switch (primitive.type) {
                case MTLPrimitiveTypePoint:
                    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassPoint;
                    break;
                case MTLPrimitiveTypeLine:
                case MTLPrimitiveTypeLineStrip:
                    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassLine;
                    break;
                default:
                    desc.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
                    break;
                }
            }

            // ---- sample count ----
            u32 samples = std::max<u32>(key.num_samples, 1);
            while (samples > 1 && ![impl->mtl_device supportsTextureSampleCount:samples]) {
                samples >>= 1;
            }
            pipeline->sample_count = samples;
            desc.rasterSampleCount = samples;

            // ---- attachments ----
            {
                for (u32 i = 0; i < key.num_color_attachments; ++i) {
                    if (!(key.mrt_mask & (1u << i))) {
                        continue; // a hole: not written by the fragment shader
                    }
                    const auto& cb = key.color_buffers[i];
                    const vk::Format format = L::TrySurfaceFormat(cb.data_format, cb.num_format);
                    if (format == vk::Format::eUndefined) {
                        continue; // written by the shader but not bound: Vulkan leaves it unused
                    }
                    const u32 pixel_format = VkToMtl::PixelFormat(format);
                    if (!pixel_format) {
                        error = fmt::format("color attachment {} format {} has no Metal pixel "
                                            "format",
                                            i, vk::to_string(format));
                        break;
                    }
                    pipeline->color_pixel_formats[i] = pixel_format;
                    MTLRenderPipelineColorAttachmentDescriptor* att = desc.colorAttachments[i];
                    att.pixelFormat = static_cast<MTLPixelFormat>(pixel_format);
                    att.writeMask = static_cast<MTLColorWriteMask>(
                        VkToMtl::ColorWriteMask(key.write_masks[i]));
                    const bool blends = i < input.blend.size() && input.blend[i].blendEnable;
                    att.blendingEnabled = blends;
                    if (blends) {
                        const vk::PipelineColorBlendAttachmentState& blend = input.blend[i];
                        att.sourceRGBBlendFactor = static_cast<MTLBlendFactor>(
                            VkToMtl::BlendFactor(blend.srcColorBlendFactor));
                        att.destinationRGBBlendFactor = static_cast<MTLBlendFactor>(
                            VkToMtl::BlendFactor(blend.dstColorBlendFactor));
                        att.rgbBlendOperation = static_cast<MTLBlendOperation>(
                            VkToMtl::BlendOp(blend.colorBlendOp));
                        att.sourceAlphaBlendFactor = static_cast<MTLBlendFactor>(
                            VkToMtl::BlendFactor(blend.srcAlphaBlendFactor));
                        att.destinationAlphaBlendFactor = static_cast<MTLBlendFactor>(
                            VkToMtl::BlendFactor(blend.dstAlphaBlendFactor));
                        att.alphaBlendOperation = static_cast<MTLBlendOperation>(
                            VkToMtl::BlendOp(blend.alphaBlendOp));
                    }
                }
                if (!error.empty()) {
                    break;
                }
                const vk::Format depth_format = L::DepthFormat(key.z_format, key.stencil_format);
                const u32 depth_pixel = VkToMtl::PixelFormat(depth_format);
                if (depth_format != vk::Format::eUndefined && !depth_pixel) {
                    error = fmt::format("depth format {} has no Metal pixel format",
                                        vk::to_string(depth_format));
                    break;
                }
                if (input.formats_override) {
                    // bbport: the pass's formats; an aspect the key has none of stays unused.
                    pipeline->depth_pixel_format = input.depth_format;
                    pipeline->stencil_pixel_format = input.stencil_format;
                    desc.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(input.depth_format);
                    desc.stencilAttachmentPixelFormat =
                        static_cast<MTLPixelFormat>(input.stencil_format);
                    pipeline->depth_ignored = key.z_format == Gcn::DepthBuffer::ZFormat::Invalid;
                    pipeline->stencil_ignored =
                        key.stencil_format == Gcn::DepthBuffer::StencilFormat::Invalid;
                } else {
                    if (key.z_format != Gcn::DepthBuffer::ZFormat::Invalid) {
                        pipeline->depth_pixel_format = depth_pixel;
                        desc.depthAttachmentPixelFormat = static_cast<MTLPixelFormat>(depth_pixel);
                    }
                    if (key.stencil_format != Gcn::DepthBuffer::StencilFormat::Invalid) {
                        pipeline->stencil_pixel_format = depth_pixel;
                        desc.stencilAttachmentPixelFormat =
                            static_cast<MTLPixelFormat>(depth_pixel);
                    }
                }
            }

            // ---- vertex input ----
            const auto attributes = input.vertex_attributes;
            if (attributes.size() > kMaxVertexBuffers) {
                error = fmt::format("{} vertex attributes, at most {} fit", attributes.size(),
                                    kMaxVertexBuffers);
                break;
            }
            if (!attributes.empty()) {
                MTLVertexDescriptor* vd = [MTLVertexDescriptor vertexDescriptor];
                for (u32 i = 0; i < attributes.size(); ++i) {
                    const VertexAttribute& attribute = attributes[i];
                    const vk::Format format = VertexBufferFormat(key, i);
                    const u32 vertex_format = VkToMtl::VertexFormat(format);
                    if (!vertex_format || attribute.location >= 31) {
                        error = fmt::format("vertex attribute {} (location {}) format {} has no "
                                            "Metal vertex format",
                                            i, attribute.location, vk::to_string(format));
                        break;
                    }
                    const u32 slot = kVertexBufferBase - i;
                    const bool constant = attribute.per_instance && attribute.divisor == 0;
                    MTLVertexAttributeDescriptor* a = vd.attributes[attribute.location];
                    a.format = static_cast<MTLVertexFormat>(vertex_format);
                    a.offset = 0;
                    a.bufferIndex = slot;
                    MTLVertexBufferLayoutDescriptor* layout = vd.layouts[slot];
                    if (constant) {
                        layout.stepFunction = MTLVertexStepFunctionConstant;
                        layout.stepRate = 0;
                        layout.stride = MTLBufferLayoutStrideDynamic; // Metal rejects stride 0
                    } else {
                        layout.stepFunction = attribute.per_instance
                                                  ? MTLVertexStepFunctionPerInstance
                                                  : MTLVertexStepFunctionPerVertex;
                        layout.stepRate = attribute.per_instance ? attribute.divisor : 1;
                        layout.stride = MTLBufferLayoutStrideDynamic;
                    }
                    pipeline->vertex_slots[i] = {attribute.location, slot, constant};
                }
                if (!error.empty()) {
                    break;
                }
                pipeline->num_vertex_slots = static_cast<u32>(attributes.size());
                desc.vertexDescriptor = vd;
            }

            // ---- create ----
            static std::atomic<bool> logic_op_warned{};
            if (key.logic_op != Gcn::ColorControl::LogicOp::Copy &&
                !logic_op_warned.exchange(true)) {
                LOG_WARNING(Render, "Metal has no logic ops; the pipelines ignore them");
            }
            const auto start = Clock::now();
            id state = CreateWithArchive(
                *impl, archive_loaded, desc,
                [&](MTLPipelineOption options, NSError** err) {
                    return [impl->mtl_device newRenderPipelineStateWithDescriptor:desc
                                                                          options:options
                                                                       reflection:nil
                                                                            error:err];
                },
                [](id<MTLBinaryArchive> archive, MTLRenderPipelineDescriptor* d, NSError** err) {
                    return [archive addRenderPipelineFunctionsWithDescriptor:d error:err];
                },
                from_archive, error, desc.fragmentFunction != nil);
            elapsed_us = UsSince(start);
            if (!state) {
                break;
            }
            pipeline->state = (__bridge_retained void*)state;
            pipeline->vertex_function = (__bridge_retained void*)desc.vertexFunction;
            if (desc.fragmentFunction) {
                pipeline->fragment_function = (__bridge_retained void*)desc.fragmentFunction;
            }
        } while (false);
    }

    if (!pipeline->state) {
        LOG_ERROR(Render, "Metal render pipeline ({}): {}", StageHashes(key), error);
        // The retained function pointers must exist for the identity check; a failed entry has
        // none and is found by its (never reused while we hold the shaders) key only. Keep the
        // functions alive by retaining them here.
        pipeline->vertex_function = (__bridge_retained void*)(__bridge id)vs->function;
        if (fs) {
            pipeline->fragment_function = (__bridge_retained void*)(__bridge id)fs->function;
        }
        ++counters.failures;
    }
    if (elapsed_us >= 500000 && !from_archive) {
        static std::atomic<u32> slow_logged{0};
        if (slow_logged++ < 30) {
            LOG_WARNING(Render, "Metal render pipeline ({}{}): {} ms to build", StageHashes(key),
                        input.tessellation_eval ? ", tessellation" : "", elapsed_us / 1000);
        }
    }
    ++counters.render_pipelines;
    counters.render_us += elapsed_us;
    counters.archive_hits += from_archive;
    if (!from_archive && pipeline->state && impl->archive && pipeline->fragment_function) {
        ++counters.archive_adds; // (depth-only pipelines stay out of the archive)
    }

    std::unique_lock lock{impl->mutex};
    const auto [it, inserted] = impl->render.try_emplace(map_key, std::move(pipeline));
    if (!inserted) {
        --counters.render_pipelines; // another thread built it first; this one is dropped
        if (!it->second->state) {
            --counters.failures;
        }
    }
    return it->second->state ? it->second.get() : nullptr;
}

const ComputePipeline* PsoCache::GetComputePipeline(const Vulkan::ComputePipelineKey& key,
                                                         const CompiledShader* cs) {
    ASSERT(cs && cs->function);
    const ComputeMapKey map_key{key.value, cs->function};
    {
        std::shared_lock lock{impl->mutex};
        const auto it = impl->compute.find(map_key);
        if (it != impl->compute.end()) {
            return it->second->state ? it->second.get() : nullptr;
        }
    }

    auto pipeline = std::make_unique<ComputePipeline>();
    pipeline->shader = cs;
    pipeline->key_value = key.value;
    pipeline->threads_per_group = cs->local_size;
    pipeline->function = (__bridge_retained void*)(__bridge id)cs->function;
    std::string error;
    bool from_archive = false;
    u64 elapsed_us = 0;
    @autoreleasepool {
        do {
            const u32 total = cs->local_size[0] * cs->local_size[1] * cs->local_size[2];
            if (total == 0) {
                error = "the shader declares no LocalSize";
                break;
            }
            MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
            desc.label = [NSString stringWithFormat:@"cs %#llx", (unsigned long long)key.value];
            desc.computeFunction = (__bridge id<MTLFunction>)cs->function;
            if (total <= 1024) {
                desc.maxTotalThreadsPerThreadgroup = total;
            }
            const auto start = Clock::now();
            id state = CreateWithArchive(
                *impl, archive_loaded, desc,
                [&](MTLPipelineOption options, NSError** err) {
                    return [impl->mtl_device newComputePipelineStateWithDescriptor:desc
                                                                           options:options
                                                                        reflection:nil
                                                                             error:err];
                },
                [](id<MTLBinaryArchive> archive, MTLComputePipelineDescriptor* d, NSError** err) {
                    return [archive addComputePipelineFunctionsWithDescriptor:d error:err];
                },
                from_archive, error);
            elapsed_us = UsSince(start);
            if (!state) {
                break;
            }
            id<MTLComputePipelineState> compute_state = state;
            pipeline->max_total_threads = static_cast<u32>(compute_state.maxTotalThreadsPerThreadgroup);
            pipeline->thread_execution_width = static_cast<u32>(compute_state.threadExecutionWidth);
            if (total > pipeline->max_total_threads) {
                error = fmt::format("LocalSize {}x{}x{} exceeds the pipeline's limit of {} threads",
                                    cs->local_size[0], cs->local_size[1], cs->local_size[2],
                                    pipeline->max_total_threads);
                break;
            }
            pipeline->state = (__bridge_retained void*)state;
        } while (false);
    }

    if (!pipeline->state) {
        LOG_ERROR(Render, "Metal compute pipeline cs {:#x}: {}", key.value, error);
        ++counters.failures;
    }
    ++counters.compute_pipelines;
    counters.compute_us += elapsed_us;
    counters.archive_hits += from_archive;
    if (!from_archive && pipeline->state && impl->archive) {
        ++counters.archive_adds;
    }

    std::unique_lock lock{impl->mutex};
    const auto [it, inserted] = impl->compute.try_emplace(map_key, std::move(pipeline));
    if (!inserted) {
        --counters.compute_pipelines;
        if (!it->second->state) {
            --counters.failures;
        }
    }
    return it->second->state ? it->second.get() : nullptr;
}

namespace {

/// One of the tessellation path's compute pipelines; `stage_in` (nil: none) describes the vertex
/// kernel's attributes. False with `error` set.
bool MakeKernel(PsoCache::Impl& impl, bool archive_loaded, TessKernel& kernel,
                const CompiledShader& shader, NSString* label,
                MTLStageInputOutputDescriptor* stage_in, bool& from_archive, std::string& error) {
    kernel.shader = &shader;
    kernel.function = (__bridge_retained void*)(__bridge id)shader.function;
    MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
    desc.label = label;
    desc.computeFunction = (__bridge id<MTLFunction>)shader.function;
    desc.stageInputDescriptor = stage_in;
    bool archived = false;
    id state = CreateWithArchive(
        impl, archive_loaded, desc,
        [&](MTLPipelineOption options, NSError** err) {
            return [impl.mtl_device newComputePipelineStateWithDescriptor:desc
                                                                  options:options
                                                               reflection:nil
                                                                    error:err];
        },
        [](id<MTLBinaryArchive> archive, MTLComputePipelineDescriptor* d, NSError** err) {
            return [archive addComputePipelineFunctionsWithDescriptor:d error:err];
        },
        archived, error);
    from_archive = from_archive && archived;
    if (!state) {
        return false;
    }
    id<MTLComputePipelineState> compute_state = state;
    kernel.max_total_threads = static_cast<u32>(compute_state.maxTotalThreadsPerThreadgroup);
    kernel.thread_execution_width = static_cast<u32>(compute_state.threadExecutionWidth);
    kernel.state = (__bridge_retained void*)state;
    return true;
}

/// The stage-input descriptors of a vertex kernel (one per index type) for the guest's
/// attributes, and the slots they use. False with `error`.
bool BuildStageInputs(const Vulkan::GraphicsPipelineKey& key,
                      std::span<const VertexAttribute> attributes,
                      std::array<MTLStageInputOutputDescriptor*, 3>& stage_in,
                      std::array<VertexBufferSlot, Vulkan::MaxVertexBufferCount>& vertex_slots,
                      u32& num_vertex_slots, std::string& error) {
    if (attributes.size() > kMaxVertexBuffers) {
        error = fmt::format("{} vertex attributes, at most {} fit", attributes.size(),
                            kMaxVertexBuffers);
        return false;
    }
    for (u32 index_type = 0; index_type < 3 && !attributes.empty(); ++index_type) {
        MTLStageInputOutputDescriptor* sid =
            [MTLStageInputOutputDescriptor stageInputOutputDescriptor];
        for (u32 i = 0; i < attributes.size(); ++i) {
            const VertexAttribute& attribute = attributes[i];
            const vk::Format format = VertexBufferFormat(key, i);
            const u32 vertex_format = VkToMtl::VertexFormat(format);
            if (!vertex_format || attribute.location >= 31) {
                error = fmt::format("vertex attribute {} (location {}) format {} has no "
                                    "Metal vertex format",
                                    i, attribute.location, vk::to_string(format));
                return false;
            }
            const u32 slot = kVertexBufferBase - i;
            const bool constant = attribute.per_instance && attribute.divisor == 0;
            // (MTLAttributeFormat has the values of MTLVertexFormat.)
            MTLAttributeDescriptor* a = sid.attributes[attribute.location];
            a.format = static_cast<MTLAttributeFormat>(vertex_format);
            a.offset = 0;
            a.bufferIndex = slot;
            MTLBufferLayoutDescriptor* layout = sid.layouts[slot];
            layout.stride = MTLBufferLayoutStrideDynamic;
            if (constant) {
                layout.stepFunction = MTLStepFunctionConstant;
                layout.stepRate = 0;
            } else if (attribute.per_instance) {
                layout.stepFunction = index_type != 0
                                          ? MTLStepFunctionThreadPositionInGridYIndexed
                                          : MTLStepFunctionThreadPositionInGridY;
                layout.stepRate = attribute.divisor;
            } else {
                layout.stepFunction = index_type != 0
                                          ? MTLStepFunctionThreadPositionInGridXIndexed
                                          : MTLStepFunctionThreadPositionInGridX;
                layout.stepRate = 1;
            }
            vertex_slots[i] = {attribute.location, slot, constant};
        }
        if (index_type != 0) {
            sid.indexType = index_type == 1 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
            sid.indexBufferIndex = kTessIndexBuffer;
        }
        stage_in[index_type] = sid;
    }
    num_vertex_slots = static_cast<u32>(attributes.size());
    return true;
}

} // namespace

const TessellationPipeline* PsoCache::GetTessellationPipeline(
    const Vulkan::GraphicsPipelineKey& key, const TessellationInput& input,
    const CompiledShader* fs) {
    ASSERT(input.vertex[0] && input.control && input.eval);
    const TessMapKey map_key{
        .key = key,
        .vertex = input.vertex[0]->function,
        .control = input.control->function,
        .eval = input.eval->function,
        .fs = fs ? fs->function : nullptr,
        .attributes_hash = HashAttributes(input.vertex_attributes),
    };
    {
        std::shared_lock lock{impl->mutex};
        const auto it = impl->tessellation.find(map_key);
        if (it != impl->tessellation.end()) {
            return it->second->render ? it->second.get() : nullptr;
        }
    }

    auto pipeline = std::make_unique<TessellationPipeline>();
    std::string error;
    bool from_archive = true;
    u64 elapsed_us = 0;
    @autoreleasepool {
        do {
            const CompiledShader& tcs = *input.control;
            const CompiledShader& tes = *input.eval;
            // The buffers between the stages are indexed with each stage's own struct size: the
            // interfaces must agree (the recompiler's stages exchange one array of vec4s).
            const auto& vs_layout = input.vertex[0]->tess_layout;
            if (tcs.tess_layout.input != 0 && tcs.tess_layout.input != vs_layout.output) {
                error = fmt::format("the control shader reads {} bytes per vertex, the vertex shader "
                                    "writes {}",
                                    tcs.tess_layout.input, vs_layout.output);
                break;
            }
            if (tes.tess_layout.input != 0 && tes.tess_layout.input != tcs.tess_layout.output) {
                error = fmt::format("the evaluation shader reads {} bytes per control point, the "
                                    "control shader writes {}",
                                    tes.tess_layout.input, tcs.tess_layout.output);
                break;
            }
            if (tes.tess_layout.patch_input != 0 &&
                tes.tess_layout.patch_input != tcs.tess_layout.patch_output) {
                error = fmt::format("the evaluation shader reads {} bytes of patch constants, the "
                                    "control shader writes {}",
                                    tes.tess_layout.patch_input, tcs.tess_layout.patch_output);
                break;
            }
            if (tcs.tess.output_vertices == 0 || tcs.tess.output_vertices != tes.variant.output_vertices ||
                key.patch_control_points == 0) {
                error = "the control shader declares no OutputVertices (or the patch size is 0)";
                break;
            }
            pipeline->input_control_points = key.patch_control_points;
            pipeline->output_control_points = tcs.tess.output_vertices;
            pipeline->triangles = tes.tess.domain == TessDomain::Triangles;
            pipeline->vertex_stride = vs_layout.output;
            pipeline->control_point_stride = tcs.tess_layout.output;
            pipeline->patch_stride = tcs.tess_layout.patch_output;
            pipeline->factor_stride = Tessellation::FactorStride(pipeline->triangles);

            // The vertex kernel's stage input: the attributes of the guest's fetch shader, in the
            // slots of the render path's vertex descriptor (data is read by thread position, or
            // through the index buffer for indexed draws; strides are set at each draw).
            std::array<MTLStageInputOutputDescriptor*, 3> stage_in{};
            if (!BuildStageInputs(key, input.vertex_attributes, stage_in, pipeline->vertex_slots,
                                  pipeline->num_vertex_slots, error)) {
                break;
            }

            // The render pipeline (the evaluation shader is its vertex function) and the kernels.
            const RenderPipelineInput render_input{.blend = input.blend, .tessellation_eval = true};
            pipeline->render = GetRenderPipeline(key, input.eval, fs, render_input);
            if (!pipeline->render) {
                error = "the render pipeline of the evaluation shader failed (see above)";
                break;
            }
            const auto start = Clock::now();
            for (u32 index_type = 0; index_type < 3; ++index_type) {
                if (!input.vertex[index_type] ||
                    input.vertex[index_type]->variant.index_type != index_type) {
                    error = "a vertex kernel variant is missing";
                    break;
                }
                NSString* label = [NSString
                    stringWithFormat:@"tess vs %#llx idx%u",
                                     (unsigned long long)key.stage_hashes[u32(Shader::SwStage::Vertex)],
                                     index_type];
                if (!MakeKernel(*impl, archive_loaded, pipeline->vertex[index_type],
                                *input.vertex[index_type], label, stage_in[index_type],
                                from_archive, error)) {
                    break;
                }
            }
            if (!error.empty()) {
                break;
            }
            NSString* label = [NSString
                stringWithFormat:@"tess hs %#llx",
                                 (unsigned long long)key.stage_hashes[u32(Shader::SwStage::TessellationControl)]];
            if (!MakeKernel(*impl, archive_loaded, pipeline->control, tcs, label, nil,
                            from_archive, error)) {
                break;
            }
            elapsed_us = UsSince(start);
        } while (false);
    }

    if (!error.empty()) {
        LOG_ERROR(Render, "Metal tessellation pipeline ({}): {}", StageHashes(key), error);
        pipeline->render = nullptr; // failed entries are remembered by their key
        ++counters.failures;
    }
    ++counters.tessellation_pipelines;
    counters.compute_us += elapsed_us;
    counters.archive_hits += from_archive && error.empty();

    std::unique_lock lock{impl->mutex};
    const auto [it, inserted] = impl->tessellation.try_emplace(map_key, std::move(pipeline));
    if (!inserted) {
        --counters.tessellation_pipelines; // another thread built it first; this one is dropped
        if (!it->second->render) {
            --counters.failures;
        }
    }
    return it->second->render ? it->second.get() : nullptr;
}

bool GeometryDebugEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("BB_GS_DEBUG");
        return value && value[0] == '1';
    }();
    return enabled;
}

namespace {

/// BB_GS_DEBUG (P7.1c): the key with blending off and every colour attachment but the first
/// float one masked off, and a fragment function that writes magenta to that one (zero to the
/// integer attachments). Null with `error` when the function does not compile.
const CompiledShader* DebugFragment(PsoCache::Impl& impl, Vulkan::GraphicsPipelineKey& key,
                                    std::string& error) {
    std::string outputs, body;
    bool magenta_done = false;
    for (u32 i = 0; i < key.num_color_attachments; ++i) {
        key.blend_controls[i].enable = false;
        if (!(key.mrt_mask & (1u << i))) {
            continue;
        }
        const auto format = key.color_buffers[i].num_format;
        const bool integer = format == Gcn::NumberFormat::Uint || format == Gcn::NumberFormat::Sint;
        const char* type = format == Gcn::NumberFormat::Uint   ? "uint4"
                           : format == Gcn::NumberFormat::Sint ? "int4"
                                                                  : "float4";
        outputs += fmt::format("    {} c{} [[color({})]];\n", type, i, i);
        if (!integer && !magenta_done) {
            magenta_done = true;
            body += fmt::format("    o.c{} = float4(1.0, 0.0, 1.0, 1.0);\n", i);
        } else {
            body += fmt::format("    o.c{} = {}(0);\n", i, type);
            key.write_masks[i] = {};
        }
    }
    const std::string source = "#include <metal_stdlib>\nusing namespace metal;\n\nstruct FOut {\n" +
                               outputs + "};\n\nfragment FOut bb_gs_debug() {\n    FOut o;\n" +
                               body + "    return o;\n}\n";
    std::scoped_lock lock{impl.debug_mutex};
    if (const auto it = impl.debug_fragments.find(source); it != impl.debug_fragments.end()) {
        return it->second.get();
    }
    @autoreleasepool {
        NSError* ns_error = nil;
        id<MTLLibrary> library =
            [impl.mtl_device newLibraryWithSource:[NSString stringWithUTF8String:source.c_str()]
                                          options:nil
                                            error:&ns_error];
        id<MTLFunction> function = library ? [library newFunctionWithName:@"bb_gs_debug"] : nil;
        if (!function) {
            error = fmt::format("debug fragment function: {}", FirstLine(ns_error));
            return nullptr;
        }
        auto shader = std::make_unique<CompiledShader>();
        shader->stage = ShaderStage::Fragment;
        shader->entry = "bb_gs_debug";
        shader->bindings.push_back({~0u, ~0u, ShaderBinding::Kind::PushConstants,
                                    kPushConstantBuffer, 1});
        shader->num_buffers = 1;
        shader->library = (__bridge_retained void*)library;
        shader->function = (__bridge_retained void*)function;
        return impl.debug_fragments.emplace(source, std::move(shader)).first->second.get();
    }
}

} // namespace

const GeometryPipeline* PsoCache::GetGeometryPipeline(const Vulkan::GraphicsPipelineKey& key,
                                                      const GeometryPipelineInput& input,
                                                      const CompiledShader* fs) {
    ASSERT(input.es[0] && input.gs && input.pass_through);
    const GeometryMapKey map_key{
        .key = key,
        .es = input.es[0]->function,
        .gs = input.gs->function,
        .pass_through = input.pass_through->function,
        .fs = fs ? fs->function : nullptr,
        .attributes_hash = HashAttributes(input.vertex_attributes),
    };
    {
        std::shared_lock lock{impl->mutex};
        const auto it = impl->geometry.find(map_key);
        if (it != impl->geometry.end()) {
            return it->second->render ? it->second.get() : nullptr;
        }
    }

    auto pipeline = std::make_unique<GeometryPipeline>();
    std::string error;
    bool from_archive = true;
    u64 elapsed_us = 0;
    @autoreleasepool {
        do {
            const CompiledShader& gs = *input.gs;
            pipeline->info = gs.geometry;
            pipeline->es_stride = input.es[0]->tess_layout.output;
            if (pipeline->es_stride == 0 || pipeline->info.max_vertices == 0) {
                error = "the export kernel has no output struct or the geometry shader no vertices";
                break;
            }
            std::array<MTLStageInputOutputDescriptor*, 3> stage_in{};
            if (!BuildStageInputs(key, input.vertex_attributes, stage_in, pipeline->vertex_slots,
                                  pipeline->num_vertex_slots, error)) {
                break;
            }
            RenderPipelineInput render_input{.blend = input.blend,
                                             .geometry = true,
                                             .geometry_output = pipeline->info.output};
            Vulkan::GraphicsPipelineKey render_key = key;
            const CompiledShader* render_fs = fs;
            if (GeometryDebugEnabled() && fs) { // BB_GS_DEBUG
                // bbport: no blending (the pipeline's blend states come from `blend`, not the
                // key's blend controls DebugFragment clears).
                render_input.blend = {};
                render_fs = DebugFragment(*impl, render_key, error);
                if (!render_fs) {
                    break;
                }
            }
            pipeline->render =
                GetRenderPipeline(render_key, input.pass_through, render_fs, render_input);
            if (!pipeline->render) {
                error = "the render pipeline of the pass-through function failed (see above)";
                break;
            }
            const auto start = Clock::now();
            for (u32 index_type = 0; index_type < 3; ++index_type) {
                if (!input.es[index_type] ||
                    input.es[index_type]->variant.index_type != index_type) {
                    error = "an export kernel variant is missing";
                    break;
                }
                NSString* label = [NSString
                    stringWithFormat:@"gs es %#llx idx%u",
                                     (unsigned long long)key.stage_hashes[u32(Shader::SwStage::Vertex)],
                                     index_type];
                if (!MakeKernel(*impl, archive_loaded, pipeline->es[index_type],
                                *input.es[index_type], label, stage_in[index_type], from_archive,
                                error)) {
                    break;
                }
            }
            if (!error.empty()) {
                break;
            }
            NSString* label = [NSString
                stringWithFormat:@"gs %#llx",
                                 (unsigned long long)key.stage_hashes[u32(Shader::SwStage::Geometry)]];
            if (!MakeKernel(*impl, archive_loaded, pipeline->gs, gs, label, nil, from_archive,
                            error)) {
                break;
            }
            elapsed_us = UsSince(start);
        } while (false);
    }

    if (!error.empty()) {
        LOG_ERROR(Render, "Metal geometry pipeline ({}): {}", StageHashes(key), error);
        pipeline->render = nullptr; // failed entries are remembered by their key
        ++counters.failures;
    }
    ++counters.geometry_pipelines;
    counters.compute_us += elapsed_us;
    counters.archive_hits += from_archive && error.empty();

    std::unique_lock lock{impl->mutex};
    const auto [it, inserted] = impl->geometry.try_emplace(map_key, std::move(pipeline));
    if (!inserted) {
        --counters.geometry_pipelines; // another thread built it first; this one is dropped
        if (!it->second->render) {
            --counters.failures;
        }
    }
    return it->second->render ? it->second.get() : nullptr;
}

void* PsoCache::GetDepthStencilState(const DepthStencilKey& requested) {
    // Ignored parts are zeroed so that equal states share one object.
    DepthStencilKey key = requested;
    if (!key.depth_test) {
        key.depth_write = false;
        key.depth_compare = vk::CompareOp::eAlways;
    }
    if (!key.stencil_test) {
        key.front = {};
        key.back = {};
    }
    {
        std::shared_lock lock{impl->mutex};
        const auto it = impl->depth_stencil.find(key);
        if (it != impl->depth_stencil.end()) {
            return (__bridge void*)it->second;
        }
    }
    @autoreleasepool {
        MTLDepthStencilDescriptor* desc = [MTLDepthStencilDescriptor new];
        desc.depthCompareFunction = static_cast<MTLCompareFunction>(
            key.depth_test ? VkToMtl::CompareFunction(key.depth_compare) : MTLCompareFunctionAlways);
        desc.depthWriteEnabled = key.depth_test && key.depth_write;
        if (key.stencil_test) {
            const auto make = [](const DepthStencilKey::Face& f) {
                MTLStencilDescriptor* s = [MTLStencilDescriptor new];
                s.stencilCompareFunction =
                    static_cast<MTLCompareFunction>(VkToMtl::CompareFunction(f.compare_op));
                s.stencilFailureOperation =
                    static_cast<MTLStencilOperation>(VkToMtl::StencilOp(f.fail_op));
                s.depthFailureOperation =
                    static_cast<MTLStencilOperation>(VkToMtl::StencilOp(f.depth_fail_op));
                s.depthStencilPassOperation =
                    static_cast<MTLStencilOperation>(VkToMtl::StencilOp(f.pass_op));
                s.readMask = f.compare_mask;
                s.writeMask = f.write_mask;
                return s;
            };
            desc.frontFaceStencil = make(key.front);
            desc.backFaceStencil = make(key.back);
        }
        id<MTLDepthStencilState> state = [impl->mtl_device newDepthStencilStateWithDescriptor:desc];
        if (!state) {
            LOG_ERROR(Render, "Metal depth-stencil state could not be created");
            ++counters.failures;
            return nullptr;
        }
        std::unique_lock lock{impl->mutex};
        const auto [it, inserted] = impl->depth_stencil.try_emplace(key, state);
        if (inserted) {
            ++counters.depth_stencil_states;
        }
        return (__bridge void*)it->second;
    }
}

} // namespace Metal
