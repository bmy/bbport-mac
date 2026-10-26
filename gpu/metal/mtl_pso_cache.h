// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: from Huax-1839/bloodborne-macos (ee3120a, GPL-2.0-or-later), renderer_metal/.
// Taken from Wooloomooloo2/bloodborne_mac_port at 78b59f5, its includes and names adapted.

#pragma once

#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <span>
#include <string>

#include "common/types.h"
#include "metal/mtl_shader.h"
#include "render/pipeline.h"

namespace Metal {

class Device;

/// One vertex attribute of a draw, in the order of the fetch shader's attributes (the order of
/// GraphicsPipelineKey::vertex_buffer_formats, which holds the attribute's format). The Vulkan
/// key does not carry the locations or the step rates: GraphicsPipeline::GetVertexInputs derives
/// them from the fetch shader (`location = attrib.semantic`, instance rate and divisor from
/// GetStepRate and the VS's step_rate_0/1), the Metal draw path passes the same values here.
struct VertexAttribute {
    u32 location{};          ///< the VS input location (the fetch shader's semantic)
    bool per_instance{};     ///< vk::VertexInputRate::eInstance
    u32 divisor{1};          ///< per_instance: instances per step; 0 = the value is constant
    bool operator==(const VertexAttribute&) const = default;
};

/// What a render pipeline needs besides the Vulkan key and the shaders.
struct RenderPipelineInput {
    std::span<const VertexAttribute> vertex_attributes;
    /// The color attachments' blend states as the Vulkan pipeline made them (its
    /// BlendAttachment, with GCN's differences already emulated); an attachment past the end
    /// does not blend.
    std::span<const vk::PipelineColorBlendAttachmentState> blend;
    /// A RectList pipeline whose vertex shader is the rewritten one (mtl_spirv_rectlist): it
    /// draws two plain triangles per rectangle, so the pipeline is a triangle list that needs no
    /// emulation (`primitive_type` Triangle, `primitive_emulated` false, `rect_list` true).
    bool rect_list_triangles{};
    /// The vertex shader given to GetRenderPipeline is a tessellation evaluation shader, compiled
    /// as a post-tessellation vertex function (CompileVariant::Kind::TessEval): the pipeline
    /// carries the tessellator's state (partition mode, factor format, winding, patch size from
    /// the shader's execution modes) and takes no vertex attributes (P4.4).
    bool tessellation_eval{};
    /// The vertex shader given to GetRenderPipeline is the geometry draw's pass-through function
    /// (mtl_geometry.h): the pipeline draws the strips the geometry kernel wrote, primitive type
    /// and topology class come from `geometry_output`, not from the key's (point list etc.).
    bool geometry{};
    GsOutputPrimitive geometry_output{GsOutputPrimitive::TriangleStrip};
    // Rasterizer discard (Vulkan: dynamic rasterizerDiscardEnable, regs.clipper_control.
    // dx_rasterization_kill) is not a pipeline option here: Metal accepts rasterizationEnabled =
    // NO only for a vertex function returning void, and the recompiler's vertex shaders return
    // their outputs. The draw recorder skips such draws instead.
};

/// Where the vertex buffer of one attribute goes. The layout stride is dynamic
/// (MTLBufferLayoutStrideDynamic): bind with setVertexBuffer:offset:attributeStride:atIndex:
/// using the V#'s stride. `constant` layouts (per-instance with divisor 0, step function
/// Constant) are dynamic too (Metal rejects stride 0): pass any non-zero stride, e.g. the
/// attribute's size; every vertex and instance reads the data at the buffer offset.
struct VertexBufferSlot {
    u32 location{};
    u32 slot{};            ///< Metal buffer index: kVertexBufferBase - index of the attribute
    bool constant{};       ///< per-instance with divisor 0
};

struct RenderPipeline {
    RenderPipeline() = default;
    ~RenderPipeline(); // releases the retained Metal objects
    RenderPipeline(const RenderPipeline&) = delete;
    RenderPipeline& operator=(const RenderPipeline&) = delete;

    /// id<MTLRenderPipelineState> (+1); the function pointers keep the vertex and fragment
    /// functions alive so that the cache's identity check stays valid. For .mm files:
    /// (__bridge id<...>).
    void* state{};
    void* vertex_function{};
    void* fragment_function{};

    // ---- What the draw has to do with it (everything the key says that is not in the state) ----
    /// MTLPrimitiveType of the encoded draw; `primitive_emulated`: the guest topology has no
    /// Metal equivalent (fan, adjacency, patch lists incl. RectList/QuadList) and the backend
    /// rewrites the indices/vertices first (VkToMtl::PrimitiveType).
    u32 primitive_type{};
    bool primitive_emulated{};
    bool rect_list{};
    bool quad_list{};
    /// Encoder state taken from the key: setTriangleFillMode, setDepthClipMode.
    u32 fill_mode{};         ///< MTLTriangleFillMode
    bool fill_point{};       ///< polygon mode point: Metal has none (draws filled)
    bool depth_clip{};       ///< key.depth_clip_enable: MTLDepthClipModeClip, else Clamp
    bool provoking_vertex_last{};
    bool tessellation{};         ///< `vs` is the evaluation shader (RenderPipelineInput)
    bool geometry{};             ///< `vs` is a geometry draw's pass-through function
    u32 patch_control_points{};  ///< tessellation: the patch size of the draw (drawPatches)
    bool clip_minus_w_to_w{}; ///< clip space z in [-w, w] (Metal's is [0, w])
    /// bbport: the vertex function converts it (CompileVariant::clip_fixup).
    bool clip_fixed{};
    u32 sample_count{1};
    u32 num_color_attachments{}; ///< the key's (holes have pixel format Invalid)
    u32 depth_pixel_format{};    ///< MTLPixelFormat of the depth / stencil attachment, 0 = none
    u32 stencil_pixel_format{};
    std::array<u32, Gcn::NUM_COLOR_BUFFERS> color_pixel_formats{};

    /// One entry per attribute (RenderPipelineInput order). Slots are 30, 29, ...
    std::array<VertexBufferSlot, Vulkan::MaxVertexBufferCount> vertex_slots{};
    u32 num_vertex_slots{};

    /// Owned by the caller's shader cache, which keeps them as long as the pipeline cache lives;
    /// they hold the binding maps (push constants, buffer/texture/sampler slots).
    const CompiledShader* vs{};
    const CompiledShader* fs{}; ///< null for a depth-only pipeline

    /// Not mapped to Metal (see mtl_pso_cache.mm): logic op, depth bounds test, provoking
    /// vertex last, per-attachment sample counts that differ.
    Vulkan::GraphicsPipelineKey key;
};

/// One compute pipeline of the tessellation path (the vertex kernel, the control kernel).
struct TessKernel {
    TessKernel() = default;
    ~TessKernel();
    TessKernel(const TessKernel&) = delete;
    TessKernel& operator=(const TessKernel&) = delete;

    void* state{};    ///< id<MTLComputePipelineState> (+1)
    void* function{}; ///< id<MTLFunction> (+1)
    u32 max_total_threads{};
    u32 thread_execution_width{};
    const CompiledShader* shader{};
};

/// What PsoCache::GetTessellationPipeline builds from (P4.4).
struct TessellationInput {
    std::span<const VertexAttribute> vertex_attributes;
    std::span<const vk::PipelineColorBlendAttachmentState> blend; ///< as RenderPipelineInput's
    /// The vertex shader compiled as a kernel, by index type (CompileVariant::index_type).
    std::array<const CompiledShader*, 3> vertex{};
    const CompiledShader* control{}; ///< the control shader's kernel
    const CompiledShader* eval{};    ///< the evaluation shader's post-tessellation function
};

/// A tessellated draw's pipelines (mtl_tessellation.h says what runs when): the vertex shader and
/// the control shader as compute kernels, the evaluation shader and the fragment shader as the
/// render pipeline.
struct TessellationPipeline {
    /// The render pipeline of the evaluation shader (`vs`) and the fragment shader.
    const RenderPipeline* render{};
    std::array<TessKernel, 3> vertex; ///< by index type: none, uint16, uint32
    TessKernel control;
    /// The vertex kernel's attributes (RenderPipelineInput order), the buffer slots of its stage
    /// input descriptor (kVertexBufferBase - index, as the render pipelines').
    std::array<VertexBufferSlot, Vulkan::MaxVertexBufferCount> vertex_slots{};
    u32 num_vertex_slots{};
    u32 input_control_points{};  ///< the key's patch size: vertices per patch
    u32 output_control_points{}; ///< the control shader's OutputVertices
    bool triangles{};            ///< the triangle domain (factors: 8 bytes), else quads (12)
    /// Bytes per element of the buffers between the stages (at least 16 each).
    u32 vertex_stride{};
    u32 control_point_stride{};
    u32 patch_stride{};
    u32 factor_stride{};
};

/// What PsoCache::GetGeometryPipeline builds from (P7.1).
struct GeometryPipelineInput {
    std::span<const VertexAttribute> vertex_attributes;
    std::span<const vk::PipelineColorBlendAttachmentState> blend; ///< as RenderPipelineInput's
    /// The export shader (the pipeline's vertex stage) as a kernel, by index type.
    std::array<const CompiledShader*, 3> es{};
    const CompiledShader* gs{};           ///< the geometry kernel
    const CompiledShader* pass_through{}; ///< the render pipeline's vertex function
};

/// A geometry draw's pipelines (mtl_geometry.h says what runs when): the export shader and the
/// geometry shader as compute kernels, the pass-through function and the fragment shader as the
/// render pipeline (an indexed strip draw over the indices the geometry kernel wrote).
struct GeometryPipeline {
    const RenderPipeline* render{};
    std::array<TessKernel, 3> es; ///< by index type: none, uint16, uint32
    TessKernel gs;
    /// The export kernel's attributes (RenderPipelineInput order) and their stage-input slots.
    std::array<VertexBufferSlot, Vulkan::MaxVertexBufferCount> vertex_slots{};
    u32 num_vertex_slots{};
    GeometryInfo info;  ///< the geometry kernel's shape (CompiledShader::geometry)
    u32 es_stride{};    ///< bytes of one export output element
};

struct ComputePipeline {
    ComputePipeline() = default;
    ~ComputePipeline();
    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    void* state{};    ///< id<MTLComputePipelineState> (+1)
    void* function{}; ///< id<MTLFunction> (+1)
    /// The shader's LocalSize: threadsPerThreadgroup of a dispatch (the grid is a number of
    /// groups, as in Vulkan: dispatchThreadgroups).
    std::array<u32, 3> threads_per_group{1, 1, 1};
    u32 max_total_threads{}; ///< the state's limit; threads_per_group never exceeds it
    u32 thread_execution_width{};
    const CompiledShader* shader{};
};

/// The depth/stencil part that Vulkan sets dynamically (vk_rasterizer.cpp UpdateDepthStencilState):
/// not in GraphicsPipelineKey, a separate MTLDepthStencilState. The stencil references are encoder
/// state (setStencilFrontReferenceValue:backReferenceValue:), the depth bounds test does not exist
/// in Metal. With the test disabled the compare op / ops are ignored (and zeroed by Normalize).
struct DepthStencilKey {
    struct Face {
        vk::StencilOp fail_op{};
        vk::StencilOp pass_op{};
        vk::StencilOp depth_fail_op{};
        vk::CompareOp compare_op{};
        u8 compare_mask{0xff}; ///< the Vulkan compare mask (stencil is 8 bits)
        u8 write_mask{0xff};
        bool operator==(const Face&) const = default;
    };
    bool depth_test{};
    bool depth_write{};
    vk::CompareOp depth_compare{vk::CompareOp::eAlways};
    bool stencil_test{};
    Face front;
    Face back;
    bool operator==(const DepthStencilKey&) const = default;
};

/// The PSO builder (P3.2): Metal render/compute pipeline states from the pipeline cache core's
/// keys (D2, D5, D7). Metal::PipelineCache (mtl_pipeline_cache.h) is the core's cache and owns one.
/// Thread-safe: lookups take a shared lock; a miss builds the pipeline outside the lock (two
/// threads missing on the same key both build, the second result is dropped). Pointers returned
/// stay valid until the cache is destroyed. A failed build is remembered (the error is logged
/// once, with the shader hashes and the Metal error) and returns nullptr every time.
/// BB_GS_DEBUG=1 (read once): geometry draws use a constant magenta fragment function, no
/// blending, no depth write, and the draws log their first vertex records (P7.1c).
bool GeometryDebugEnabled();

class PsoCache {
public:
    /// `cache_dir` is the folder of the MSL cache (default: ShaderCompiler::DefaultCacheDir());
    /// the binary archive lives in it. Without a writable folder only the archive is lost.
    explicit PsoCache(Device& device, std::filesystem::path cache_dir = {});
    ~PsoCache(); // flushes the archive

    /// `vs` is required, `fs` may be null (depth-only pass). The key must be the one the Vulkan
    /// cache computed for the same stages (stage_hashes name the shaders in the logs).
    const RenderPipeline* GetRenderPipeline(const Vulkan::GraphicsPipelineKey& key,
                                            const CompiledShader* vs, const CompiledShader* fs,
                                            const RenderPipelineInput& input);

    const ComputePipeline* GetComputePipeline(const Vulkan::ComputePipelineKey& key,
                                              const CompiledShader* cs);

    /// The pipelines of a tessellated draw (P4.4), or nullptr (logged once per key): the shader
    /// variants must agree on the structs between the stages (their TessLayout), the render
    /// pipeline is GetRenderPipeline's with `RenderPipelineInput::tessellation_eval`.
    const TessellationPipeline* GetTessellationPipeline(const Vulkan::GraphicsPipelineKey& key,
                                                        const TessellationInput& input,
                                                        const CompiledShader* fs);

    /// The pipelines of a geometry draw (P7.1), or nullptr (logged once per key). The render
    /// pipeline is GetRenderPipeline's with `RenderPipelineInput::geometry`.
    const GeometryPipeline* GetGeometryPipeline(const Vulkan::GraphicsPipelineKey& key,
                                                const GeometryPipelineInput& input,
                                                const CompiledShader* fs);

    /// id<MTLDepthStencilState> (owned by the cache), nullptr on failure.
    void* GetDepthStencilState(const DepthStencilKey& key);

    /// Writes the binary archive when pipelines were added since the last write (atomically:
    /// temp file + rename). Cheap to call when nothing changed; call it every few new pipelines
    /// and at shutdown (the destructor does).
    bool Flush();

    struct Counters {
        std::atomic<u64> render_pipelines{};   ///< distinct render pipelines built (incl. failed)
        std::atomic<u64> compute_pipelines{};
        std::atomic<u64> tessellation_pipelines{}; ///< distinct tessellation pipelines built (incl. failed)
        std::atomic<u64> geometry_pipelines{};     ///< distinct geometry pipelines built (incl. failed)
        std::atomic<u64> failures{};           ///< render + compute builds that failed
        std::atomic<u64> depth_stencil_states{};
        std::atomic<u64> archive_hits{};       ///< pipelines taken from the binary archive
        std::atomic<u64> archive_adds{};
        /// Microseconds spent in the Metal pipeline-state calls.
        std::atomic<u64> render_us{};
        std::atomic<u64> compute_us{};
    };
    const Counters& GetCounters() const {
        return counters;
    }
    const std::filesystem::path& ArchivePath() const {
        return archive_path;
    }
    bool ArchiveLoaded() const {
        return archive_loaded;
    }

    struct Impl; ///< Objective-C state (mtl_pso_cache.mm)

private:
    Device& device;
    std::filesystem::path archive_path;
    bool archive_loaded{}; ///< an existing archive file was opened
    Counters counters;
    std::unique_ptr<Impl> impl;
};

} // namespace Metal
