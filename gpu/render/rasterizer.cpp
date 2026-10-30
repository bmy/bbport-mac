// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, the draw and dispatch entry points: pipeline selection, then recording here or on the
// draw recording thread (raster_pipe.cpp), and the per-frame bookkeeping.

#include <chrono>
#include <cstring>
#include <mutex>
#include "shim/bb_env.h"
#include <set>
#include <string>
#include <filesystem>
#include "render/raster_internal.h"
#include <vk_mem_alloc.h>

namespace Vulkan {

Rasterizer::Rasterizer(const Instance& instance_, Scheduler& scheduler_, Runtime& runtime_,
                       BbCp::CommandProcessor* liverpool_)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_}, page_watch{this},
      buffer_cache{instance, scheduler, runtime, liverpool_, texture_cache, page_watch},
      texture_cache{instance, scheduler, runtime, liverpool_, buffer_cache, page_watch},
      liverpool{liverpool_}, memory{Core::Memory::Instance()},
      pipeline_cache{instance, scheduler, liverpool, buffer_cache.GetSparsePageShift()},
      host_markers_enabled{EmulatorSettings.IsVkHostMarkersEnabled()},
      guest_markers_enabled{EmulatorSettings.IsVkGuestMarkersEnabled()} {
    // Before the rasterizer is bound: Liverpool enqueues buffers only once it sees it.
    draw_prep = std::make_unique<DrawPreparation>(pipeline_cache);
    // A program (or fetch shader) that does not look like one yet: the game may have copied it
    // into place with the GPU (DMA) in work recorded but not run yet (another game: all-zero
    // programs, the load hung). Called on the GPU command thread, inside a draw: the work
    // handed to the recording thread and recorded before it runs, so a
    // program the GPU copies into place (in place in guest memory) is there. No read back of
    // GPU-modified ranges here: downloading 512 KiB windows at draws overwrote what the CPU had
    // written there with older GPU data (another game: shader code zeroed, characters missing,
    // 5 FPS), and asking this very thread for one waited forever.
    pipeline_cache.SetProgramWriteSync(
        [this](VAddr address, u64 size) { SyncProgramWrites(address, size); });
    pipeline_cache.SetProgramMemorySync([this](VAddr, u64) {
        DrainDrawPipe(); // a copy into place may still be a packet for the recording thread
        Finish();
    });
    scene_targets = std::make_unique<SceneTargets>(instance, scheduler, runtime, texture_cache);
    // Object motion first: it fixes the buffer addresses the motion shader variants embed.
    object_motion = std::make_unique<ObjectMotion>(instance, scheduler);
    pipeline_cache.Preload();
    camera_motion = std::make_unique<CameraMotion>(instance, scheduler, texture_cache, runtime);
    camera_motion->SetObjectMotion(object_motion.get());
    upscaler = std::make_unique<TemporalUpscaler>(instance, scheduler, texture_cache, runtime,
                                                  *camera_motion, *scene_targets);
    vbao = std::make_unique<Vbao>(instance, scheduler, texture_cache, runtime, *camera_motion);
    upscaler->on_upscaled = [this](vk::Image output, u32 width, u32 height) {
        vbao->OnUpscaled(output, width, height);
    };
    prompts = std::make_unique<ButtonPrompts>(instance, scheduler, buffer_cache);
    if (!EmulatorSettings.IsNullGPU()) {
        liverpool->BindRasterizer(this);
    }
    BbGuestMemory::Install(instance); // bbport: before the runtime maps direct memory
    memory->SetRasterizer(this);

    GpuProfiler::Init(instance, scheduler);
    if (DrawPipeWanted()) {
        constant_ring = std::make_unique<ConstantRing>(instance, scheduler);
        draw_pipe = std::make_unique<DrawPipe>(&RunDrawPacket, this);
        buffer_cache.in_place_read_context = this;
        buffer_cache.in_place_read_hook = [](void* context, VAddr address, u64 size) {
            auto& self = *static_cast<Rasterizer*>(context);
            if (DrawPipe::OnStageB()) {
                self.gpu_reads.Note(address, size, self.stage_b_packet);
            }
        };
        // BB_RESOLVE_WORKERS=n (stage 2, docs/pipeline_4stage.ru.md): threads that check and copy
        // the constants of draws between the GPU command thread and the recording thread.
        static const u32 resolvers = [] {
            const char* env = bb_env("BB_RESOLVE_WORKERS");
            return env ? u32(std::clamp(std::atoi(env), 0, 4)) : 0u;
        }();
        // Without resolve workers stage A is the only thread noting reads (PendingReads::Raise).
        pending_reads.single_writer = resolvers == 0;
        pending_resolve.single_writer = true;
        if (resolvers != 0) {
            for (u32 i = 0; i < resolvers; ++i) {
                worker_rings.push_back(std::make_unique<ConstantRing>(instance, scheduler));
            }
            draw_pipe->StartResolvers(&ResolvePacket, resolvers);
            std::printf("GPU: %u constant resolve workers (BB_RESOLVE_WORKERS)\n", resolvers);
        }
    }
    // bbport: this thread joins the texture binding helper before it changes image state.
    // bbport (macOS): only with the helper (BB_TEXTURE_HELPER=1); without it the hook was an
    // indirect call that did nothing on every image access.
    if (bind_helper.Available()) {
        runtime.SetImageAccessHook(&JoinBindHelper, this);
    }
    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });
}

Rasterizer::~Rasterizer() = default;

namespace {
// bbport: the pipeline's own type (optional<const ...>): the other one made each draw copy it,
// its attribute list included (a malloc per draw).
std::pair<u32, u32> GetDrawOffsets(
    const Gcn::Regs& regs, const Shader::Info& info,
    const std::optional<const Shader::Gcn::FetchShaderData>& fetch_shader) {
    u32 vertex_offset = regs.index_offset;
    u32 instance_offset = 0;
    if (fetch_shader) {
        if (vertex_offset == 0 && fetch_shader->vertex_offset_sgpr != -1) {
            vertex_offset = info.UserData()[fetch_shader->vertex_offset_sgpr];
        }
        if (fetch_shader->instance_offset_sgpr != -1) {
            instance_offset = info.UserData()[fetch_shader->instance_offset_sgpr];
        }
    }
    return {vertex_offset, instance_offset};
}
} // namespace

#ifdef __APPLE__
namespace {

/// bbport (macOS), BB_SKIP_TESS_DRAWS=1 (a test): draws with the game's own tessellation are not
/// recorded, to tell whether they are what makes the GPU slow (missing from the picture meanwhile).
bool SkipTessDraws() {
    static const bool skip = [] {
        const bool on = bb_env_on("BB_SKIP_TESS_DRAWS");
        if (on) {
            std::printf("GPU: draws with the game's tessellation are skipped (BB_SKIP_TESS_DRAWS=1, "
                        "a test: they are missing from the picture)\n");
        }
        return on;
    }();
    return skip;
}

/// Counts a recorded draw by kind; true when it is to be skipped (BB_SKIP_TESS_DRAWS). The first
/// draw with each pipeline that uses the game's tessellation is logged with its stages.
bool CountTessellation(const Vulkan::GraphicsPipeline* pipeline) {
    const auto& key = pipeline->GetGraphicsKey();
    if (key.stage_hashes[u32(Shader::SwStage::TessellationControl)] == 0) {
        if (key.prim_type == Gcn::PrimitiveType::RectList ||
            key.prim_type == Gcn::PrimitiveType::QuadList) {
            BbStats::patch_list_draws.fetch_add(1, std::memory_order_relaxed);
        }
        return false;
    }
    BbStats::tess_draws.fetch_add(1, std::memory_order_relaxed);
    static std::mutex seen_mutex;
    static std::set<const Vulkan::GraphicsPipeline*> seen;
    {
        std::scoped_lock lock{seen_mutex};
        if (seen.size() < 4096 && seen.insert(pipeline).second) {
            std::printf("GPU: a pipeline with the game's tessellation drawn (%zu so far; vertex "
                        "%#llx, hull %#llx, domain %#llx, pixel %#llx, geometry %#llx, %u control "
                        "points)\n",
                        seen.size(),
                        (unsigned long long)key.stage_hashes[u32(Shader::SwStage::Vertex)],
                        (unsigned long long)key.stage_hashes[u32(Shader::SwStage::TessellationControl)],
                        (unsigned long long)key.stage_hashes[u32(Shader::SwStage::TessellationEval)],
                        (unsigned long long)key.stage_hashes[u32(Shader::SwStage::Fragment)],
                        (unsigned long long)key.stage_hashes[u32(Shader::SwStage::Geometry)],
                        key.patch_control_points);
            std::fflush(stdout);
        }
    }
    return SkipTessDraws();
}

/// bbport (macOS) experiment bits of BB_TOGGLE_FILE (its second number), to measure what draws
/// drawn as patches cost on drivers that tessellate in software (KosmicKrisp): 1 << 18 leaves out
/// the game's tessellated draws, 1 << 19 the rect and quad lists (drawn as patches; 0.6's 128 and
/// 256 are 0.7's own experiments). The picture is wrong while either is set: for GPU time
/// measurements only.
bool SkippedPatchDraw(const GraphicsPipeline* pipeline) {
    const bool skip_tess = BbToggle::Experiment(1u << 18);
    const bool skip_lists = BbToggle::Experiment(1u << 19);
    if (!skip_tess && !skip_lists) {
        return false;
    }
    const auto prim = pipeline->GetGraphicsKey().prim_type;
    const bool list = prim == Gcn::PrimitiveType::RectList || prim == Gcn::PrimitiveType::QuadList;
    return (skip_lists && list) || (skip_tess && prim == Gcn::PrimitiveType::PatchPrimitive);
}

/// bbport (macOS) diagnostics: the game's tessellated draws (direct ones) every 5 s, per frame,
/// with the tess level clamp the game sets (VGT_HOS_MAX/MIN_TESS_LEVEL, 0xA286/0xA287, which
/// the renderer does not apply): what KosmicKrisp's software tessellation is given.
void NoteTessellatedDraw(const GraphicsPipeline* pipeline, const Gcn::Regs& regs) {
    const auto& key = pipeline->GetGraphicsKey();
    if (key.prim_type != Gcn::PrimitiveType::PatchPrimitive) {
        return;
    }
    static std::mutex mutex;
    static auto window = std::chrono::steady_clock::now();
    static u64 frames_at = BbStats::gpu_frames.load(std::memory_order_relaxed);
    static u64 draws = 0, patches = 0;
    static std::set<std::pair<u32, u32>> clamps;
    std::scoped_lock lock{mutex};
    ++draws;
    const u32 points = std::max(1u, u32(key.patch_control_points));
    patches += u64(regs.num_indices / points) * std::max(1u, regs.num_instances.NumInstances());
    if (clamps.size() < 8) {
        clamps.emplace(regs.reg_array[0xA286], regs.reg_array[0xA287]);
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - window < std::chrono::seconds(5)) {
        return;
    }
    const u64 frames_now = BbStats::gpu_frames.load(std::memory_order_relaxed);
    const double frames = double(std::max<u64>(1, frames_now - frames_at));
    std::string levels;
    for (const auto& [max_bits, min_bits] : clamps) {
        float max_level, min_level;
        std::memcpy(&max_level, &max_bits, sizeof(float));
        std::memcpy(&min_level, &min_bits, sizeof(float));
        levels += fmt::format("{}{}..{}", levels.empty() ? "" : ", ", min_level, max_level);
    }
    std::printf("Tessellation (game): %.1f draws/frame, %.0f patches/frame; tess level clamp "
                "the game sets (min..max): %s\n",
                double(draws) / frames, double(patches) / frames, levels.c_str());
    window = now;
    frames_at = frames_now;
    draws = patches = 0;
    clamps.clear();
}
} // namespace
#endif // __APPLE__

void Rasterizer::Draw(bool is_indexed, u32 index_offset, const PreparedDraw* prepared) {
    BB_SECTION(ADraw);
#ifdef __APPLE__
    static const bool skip_note = SkipTessDraws(); // says so at the first draw of the run
    (void)skip_note;
#endif
    BbStats::draws.fetch_add(1, std::memory_order_relaxed);

    // bbport: with the draw pipeline this thread only selects the pipeline and hands the draw
    // to the recording thread (DrawRecord there); draws FilterDraw handles itself run here.
    const bool pipelined = UseDrawPipe();
    if (pipelined && !FilterDrawPasses() &&
        !BbToggle::Disabled(BbToggle::PipelinedMemoryWrites)) {
        PostDraw(nullptr, nullptr, false, 0);
        return;
    }
    if (!pipelined || !FilterDrawPasses()) {
        DrainDrawPipe(DrawPipe::ReasonDraw);
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
        if (!FilterDraw()) {
            return;
        }
    }
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline({}, prepared);
    const PreparedDraw* used_prepared = pipeline_cache.UsedPrepared();
    if (prepared) {
        const u32 state = prepared->state.load(std::memory_order_acquire);
        draw_prep->Count(used_prepared != nullptr,
                         state == PreparedDraw::Pending       ? DrawPreparation::NotReady
                         : state == PreparedDraw::Unavailable ? DrawPreparation::Unavailable
                         : prepared->reg_checksum != liverpool->gfx_reg_checksum
                             ? DrawPreparation::OtherRegisters
                             : DrawPreparation::OtherPrograms);
    }
    if (!pipeline) {
        return;
    }
    if (pipelined && FilterDrawPasses()) {
        PostDraw(pipeline, used_prepared, is_indexed, index_offset);
        return;
    }
    DrawRecord(pipeline, used_prepared, is_indexed, index_offset);
}

/// The guest memory a draw reads, in its breadcrumb: a GPU stuck in the draw is reported with it
/// (a stalled read of the game's memory on NVIDIA, whose hangs show no fault).
static void NoteCrumbReads(Breadcrumbs::Crumb& crumb, const GraphicsPipeline* pipeline,
                           const auto& regs, bool is_indexed, u32 index_offset) {
    u32 n = 0;
    const auto add = [&](char kind, u64 address, u32 size) {
        if (address && n < Breadcrumbs::Crumb::MaxReads) {
            crumb.read_kinds[n] = kind;
            crumb.reads[n] = address;
            crumb.read_sizes[n++] = size;
        }
    };
    if (is_indexed) {
        const u32 index_size = regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2u : 4u;
        add('i', regs.index_base_address.template Address<VAddr>() + u64(index_offset) * index_size,
            regs.num_indices * index_size);
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    if (const auto& fetch = pipeline->GetFetchShader()) {
        for (const auto& attribute : fetch->attributes) {
            const auto sharp = attribute.GetSharp(vs);
            add('v', sharp.base_address, sharp.GetSize());
        }
    }
    for (const auto* stage : {&vs, pipeline->GetStages()[u32(Shader::SwStage::Fragment)]}) {
        if (!stage) {
            continue;
        }
        for (const auto& buffer : stage->buffers) {
            if (!buffer.IsSpecial()) {
                const auto sharp = buffer.GetSharp(*stage);
                add('b', sharp.base_address, sharp.GetSize());
            }
        }
        for (const auto& image : stage->images) {
            add('t', image.GetSharp(*stage).Address(), 0);
        }
    }
}

/// BB_FG_EARLY_RETIRE (default 1, 0 off): with frame generation the game's frame counts as finished
/// (BbGnm::NoteFramesRetired, which the game's next frame waits for) once the GPU has run its
/// display pass, before the interpolation: that reads only the port's own images, so the game
/// goes on with its next frame meanwhile instead of waiting ~3 ms more (Deck: 35.1 -> 38.8 FPS).
/// The frame's last work is its display pass (checked: guest work recorded after it and before
/// the frame's end is counted and reported).
static bool EarlyRetire() {
    static const bool on = [] {
        const char* env = bb_env("BB_FG_EARLY_RETIRE");
        return !env || env[0] != '0';
    }();
    return on;
}

void Rasterizer::NoteFrameEndTask(u64 frame) {
    if (early_retired != 0 && frame == early_retired && work_after_early > 0) {
        ++early_unsafe;
    }
    if (early_retired != 0 && (early_frames & (early_frames - 1)) == 0 && early_frames >= 64) {
        std::printf("Frame generation: %llu frames retired before the interpolation, %llu of them "
                    "with game work after their display pass\n",
                    static_cast<unsigned long long>(early_frames),
                    static_cast<unsigned long long>(early_unsafe));
    }
    frame_end_task = frame;
    early_retired = 0;
}

void Rasterizer::RecordPendingFrameGeneration() {
    if (!upscaler->HasPendingInterpolation()) {
        return;
    }
    // The frame after the last frame end seen: this display pass's. Not before one was seen
    // (frame numbers start at the first sceGnmSubmitDone).
    if (EarlyRetire() && frame_end_task != 0 && early_retired == 0 &&
        upscaler->FrameGenerationWanted()) {
        const u64 frame = frame_end_task + 1;
        SignalAfterGpu([frame] { BbGnm::NoteFramesRetired(frame); });
        Flush(); // its own submission: the interpolation's is the next one
        early_retired = frame;
        work_after_early = 0;
        ++early_frames;
    }
    upscaler->RunPendingInterpolation();
}

namespace {
/// A draw that changes nothing (#352): no color target it may write, no depth or stencil test,
/// no pixel shader, a vertex shader writing no memory. The game issues ~14 a frame (vs 89cda9e4),
/// each a render pass of its own of 32768x32768 with nothing attached. Off by default: leaving
/// them out changed nothing measurable (RX 7800 XT, 2026-10-27: 137-139 against 137-142 FPS);
/// BB_SKIP_NOOP_DRAWS=1: left out.
bool IsNoopDraw(const Gcn::Regs& regs, const Vulkan::GraphicsPipeline* pipeline) {
    static const bool skip = bb_env_on("BB_SKIP_NOOP_DRAWS");
    if (!skip || pipeline->GetStages()[u32(Shader::SwStage::Fragment)] != nullptr) {
        return false;
    }
    const auto& key = pipeline->GetGraphicsKey();
    if (regs.color_control.mode != Gcn::ColorControl::OperationMode::Disable) {
        for (s32 cb = 0; cb < std::bit_width(key.mrt_mask); ++cb) {
            if (regs.color_buffers[cb] && regs.color_target_mask.GetMask(cb) &&
                (key.mrt_mask & (1u << cb)) != 0) {
                return false;
            }
        }
    }
    if ((regs.depth_control.depth_enable && regs.depth_buffer.DepthValid()) ||
        (regs.depth_control.stencil_enable && regs.depth_buffer.StencilValid())) {
        return false;
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    if (vs.uses_dma || std::ranges::any_of(vs.buffers, [](const auto& b) { return b.is_written; }) ||
        std::ranges::any_of(vs.images, [](const auto& i) { return i.is_written; })) {
        return false;
    }
    static std::atomic<u32> told{0};
    if (told.fetch_add(1, std::memory_order_relaxed) < 4) {
        std::printf("GPU: draws without any output left out (vs %016llx; BB_SKIP_NOOP_DRAWS=1)\n",
                    static_cast<unsigned long long>(vs.pgm_hash));
    }
    return true;
}
} // namespace

void Rasterizer::NoteInPlaceVertexReads(const GraphicsPipeline* pipeline, bool is_indexed,
                                        u32 index_offset) {
    const auto& regs = Regs();
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto [base_vertex, first_instance] = GetDrawOffsets(regs, vs, pipeline->GetFetchShader());
    Motion::VertexRange range{base_vertex, regs.num_indices};
    if (is_indexed) {
        const u32 index_size =
            regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2u : 4u;
        const VAddr address = regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
        const u64 bytes = u64(regs.num_indices) * index_size;
        range = {};
        if (address && memory->ClampRangeSize(address, bytes) == bytes) {
            const auto* data = reinterpret_cast<const void*>(address);
            const bool restart = regs.enable_primitive_restart != 0;
            range = index_size == 2
                        ? Motion::IndexedRange(std::span(static_cast<const u16*>(data), regs.num_indices),
                                               s32(base_vertex), restart)
                        : Motion::IndexedRange(std::span(static_cast<const u32*>(data), regs.num_indices),
                                               s32(base_vertex), restart);
        }
    }
    static u64 bound = 0, read = 0, draws = 0, frames_start = 0;
    static std::unordered_set<u64> pages; // 4 KiB pages read, this frame
    static u64 unique = 0, pages_frame = 0;
    const u64 frame = BbStats::frame_number.load(std::memory_order_relaxed);
    if (frame != pages_frame) {
        unique += pages.size();
        pages.clear();
        pages_frame = frame;
    }
    ++draws;
    for (const auto& [base, stride, size] : in_place_streams) {
        bound += size;
        const u64 begin = std::min<u64>(u64(range.first) * stride, size);
        const u64 end = std::min<u64>(u64(range.first + range.count) * stride, size);
        read += end - begin;
        for (u64 page = (base + begin) >> 12; end > begin && page <= (base + end - 1) >> 12; ++page) {
            pages.insert(page);
        }
    }
    if (frames_start == 0) {
        frames_start = frame;
    }
    static u32 printed = 0;
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed >= 2 && frame > frames_start) {
        const double f = double(frame - frames_start);
        std::printf("Vertex in place reads (KiB/frame): %.0f draws, bound %.0f, read by indices "
                    "%.0f, unique pages %.0f\n",
                    draws / f, bound / f / 1024.0, read / f / 1024.0, unique * 4.0 / f);
        printed = now;
        bound = read = draws = unique = 0;
        frames_start = frame;
    }
}

void Rasterizer::DrawRecord(const GraphicsPipeline* pipeline, const PreparedDraw* used_prepared,
                            bool is_indexed, u32 index_offset) {
    BB_SECTION(DrawRecord);
#ifdef __APPLE__
    if (CountTessellation(pipeline)) {
        return; // BB_SKIP_TESS_DRAWS (a test)
    }
#endif
    buffer_cache.NewPacket();
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
    }
    const auto& regs = Regs();
    ++work_after_early;
    // Frame generation goes after the display pass's draw, so on leaving this function.
    struct AfterDisplayPass {
        Rasterizer* self = nullptr;
        ~AfterDisplayPass() {
            if (self) {
                self->RecordPendingFrameGeneration();
            }
        }
    } after_display_pass;
    // bbport: the pass copying a finished frame to a display buffer; the previous draw's
    // target is that frame.
    if (camera_motion->Enabled() && regs.color_buffers[0] &&
        FrameCapture::IsDisplayBuffer(regs.color_buffers[0].Address())) {
        after_display_pass.self = this;
        scene_started = false;
        DumpTracedInputs();
        DumpPassImages('p');
        upscaler->OnDisplayPassStart(regs.color_buffers[0].Address());
        vbao->OnDisplayPass(cb_descs[0].first);
        camera_motion->OnDisplayPass(cb_descs[0].first);
        if (upscaler->OnFrameStart()) {
            object_motion->InvalidateHistory();
            camera_motion->InvalidateHistory();
        }
        camera_motion->SetJitter(upscaler->Jitter());
        object_motion->OnFrameStart();
        NoteFrameStart();
        static const char* scene_debug = bb_env("BB_SCENE_DEBUG");
        scene_debug_frame = scene_debug && std::remove(scene_debug) == 0;
        scene_targets->debug = scene_debug_frame;
    }
    // BB_SKIP_PS=hash[,hash...] (diagnostics): draws with these pixel shaders are left out.
    static const std::vector<u64> skipped_ps = [] {
        std::vector<u64> hashes;
        if (const char* env = bb_env("BB_SKIP_PS")) {
            for (const char* at = env; *at;) {
                char* end = nullptr;
                hashes.push_back(std::strtoull(at, &end, 16));
                at = *end ? end + 1 : end;
            }
        }
        return hashes;
    }();
    if (!skipped_ps.empty()) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        if (ps && std::ranges::find(skipped_ps, ps->pgm_hash) != skipped_ps.end()) {
            return;
        }
    }
    if (IsNoopDraw(regs, pipeline)) {
        return;
    }
#ifdef __APPLE__
    if (SkippedPatchDraw(pipeline)) {
        return;
    }
    NoteTessellatedDraw(pipeline, regs);
#endif
    // bbport: the key guide icons for the device in use (a change: uploaded before this draw).
    prompts->Update();
    bind_prepared = used_prepared;
    motion_draw = pipeline->GetGraphicsKey().motion_vectors;
    motion_geometry = 0;

    // BB_EXP_FINISH_BEFORE_TARGET=addr (experiment, the flicker): the GPU runs everything before
    // a draw into that color target before the draw is recorded.
    static const VAddr finish_target = [] {
        const char* env = bb_env("BB_EXP_FINISH_BEFORE_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (finish_target && Regs().color_buffers[0] &&
        Regs().color_buffers[0].Address() == finish_target) {
        scheduler.Finish();
    }
    // BB_PASS_DUMP_AT=addr (diagnostics): BB_PASS_DUMP's images also before the first draw of a
    // pass into that color target ('q' files).
    static const VAddr dump_at = [] {
        const char* env = bb_env("BB_PASS_DUMP_AT");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (dump_at && Regs().color_buffers[0] && Regs().color_buffers[0].Address() == dump_at &&
        !scheduler.IsRendering()) {
        const auto* q_ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        std::printf("Pass dump q stages vs %08x ps %08x\n",
                    u32(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash), u32(q_ps ? q_ps->pgm_hash : 0));
        DumpPassImages('q');
    }
    // BB_EXP_TRANSIT_TARGET=addr (experiment): the target image goes through TransferSrcOptimal
    // before the first draw of a pass into it (what BB_PASS_DUMP_AT does, without the copy).
    static const VAddr transit_target = [] {
        const char* env = bb_env("BB_EXP_TRANSIT_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (transit_target && Regs().color_buffers[0] &&
        Regs().color_buffers[0].Address() == transit_target && !scheduler.IsRendering()) {
        VideoCore::Image* found = nullptr;
        texture_cache.ForEachImageInRegion(transit_target, 1, [&](VideoCore::ImageId, VideoCore::Image& image) {
            if (image.info.guest_address == transit_target) found = &image;
        });
        if (found) {
            runtime.Transit(found, vk::ImageLayout::eTransferSrcOptimal,
                            vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
            runtime.FlushBarriers();
        }
    }
    // BB_EXP_BARRIER_TARGET=addr (experiment): a full memory barrier before each draw into that
    // color target (its render pass is ended first).
    static const VAddr barrier_target = [] {
        const char* env = bb_env("BB_EXP_BARRIER_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (barrier_target && Regs().color_buffers[0] &&
        Regs().color_buffers[0].Address() == barrier_target) {
        scheduler.EndRendering();
        scheduler.Record([](vk::CommandBuffer cmd) {
            const vk::MemoryBarrier2 all{
                .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
                .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
                .dstAccessMask = vk::AccessFlagBits2::eMemoryWrite | vk::AccessFlagBits2::eMemoryRead,
            };
            cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &all});
        });
    }
    PrepareRenderState(pipeline);
    // BB_SKIP_EMPTY_DRAWS=1 (experiment): a draw without a pixel shader, colour targets or a depth
    // or stencil test writes nothing here (the game's depth decompress and resummarize passes:
    // full-screen, a 16384x16384 area; 1.75 ms a frame on a Steam Deck). Their samples still
    // reach the occlusion count otherwise; skipped, they do not.
    static const bool skip_empty = bb_env_on("BB_SKIP_EMPTY_DRAWS");
    const bool empty_draw =
        !pipeline->GetStages()[u32(Shader::SwStage::Fragment)] && !db_desc.first &&
        std::ranges::none_of(std::span{cb_descs}.first(std::bit_width(
                                 pipeline->GetGraphicsKey().mrt_mask)),
                             [](const auto& cb) { return bool(cb.first); });
    // The game's depth decompress passes (DB_RENDER_CONTROL compress disables, resummarize,
    // decompress): Vulkan keeps the depth compression itself, so they are drawn without the depth
    // buffer, i.e. without any output, over a 16384x16384 area: 0.5-1.7 ms a frame of rasterizing
    // on a Steam Deck, more with a reduced preset live. Skipped (BB_KEEP_DECOMPRESS_DRAWS=1: drawn).
    static const bool keep_decompress = bb_env_on("BB_KEEP_DECOMPRESS_DRAWS");
    const auto& render_control = Regs().depth_render_control;
    const bool decompress_pass = render_control.depth_compress_disable ||
                                 render_control.stencil_compress_disable ||
                                 render_control.resummarize_enable || render_control.decompress_enable;
    if (empty_draw && (skip_empty || (decompress_pass && !keep_decompress))) {
        bind_prepared = nullptr;
        return;
    }
    NoteUpscalerDraw(pipeline);
    const PreparedDraw* draw_prepared = bind_prepared;
    // bbport: vertex and index buffers are resolved while the helper binds textures (their
    // commands are recorded after BeginRendering, as before).
    draw_inputs = {pipeline, draw_prepared, index_offset, is_indexed, true, false};
    read_draw = {pipeline, index_offset, is_indexed};
    ui_trace_draw = {index_offset, is_indexed};
    const bool bound = BindResources(pipeline);
    bind_prepared = nullptr; // indirect draws and dispatches bind without prepared sharps
    const bool inputs_resolved = draw_inputs.resolved;
    draw_inputs.pending = false;
    if (!bound) {
        read_draw = {};
        return;
    }
    // BB_EXP_EXTRA_BREAKS=N (experiment): the render pass ended before every Nth draw, to measure
    // what a break costs.
    static const u32 extra_breaks = [] {
        const char* env = bb_env("BB_EXP_EXTRA_BREAKS");
        return env ? u32(std::strtoul(env, nullptr, 10)) : 0u;
    }();
    // Experiment bit 256: a break before every 10th draw, switched while the game runs.
    const u32 every = BbToggle::Experiment(256) ? 10u : extra_breaks;
    if (every) {
        static u32 draws = 0;
        if (++draws % every == 0) {
            scheduler.EndRendering();
        }
    }
    ++work_total;
    const auto state = BeginRendering(pipeline);

    if (!inputs_resolved) {
        ResolveVertexBuffers(pipeline, draw_prepared);
        if (is_indexed) {
            ResolveIndexBuffer(index_offset);
        }
    }
    read_draw = {};
    EmitVertexBuffers();
    if (is_indexed) {
        EmitIndexBuffer();
    }
    if (VertexInPlaceDiag() && !in_place_streams.empty()) {
        NoteInPlaceVertexReads(pipeline, is_indexed, index_offset);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    ScaleToTarget(pipeline);
    if (motion_draw && motion_geometry) {
        BB_SECTION(Motion);
        const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
        const auto [base_vertex, first_instance] =
            GetDrawOffsets(regs, vs, pipeline->GetFetchShader());
        const u32 index_size =
            regs.index_buffer_type.index_type == Gcn::IndexType::Index16 ? 2u : 4u;
        const VAddr index_address = is_indexed
            ? regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size : 0;
        // Without a character-size skeleton the draw is gated on its bone palette changing
        // since last frame (model constants follow the camera, the palette does not). The
        // check comes first: most gated draws are static world pieces with long index lists.
        static const bool all_motion = bb_env_on("BB_OBJECT_MOTION_ALL");
        // Instanced draws (foliage) hash their per-instance matrices into the draw's identity and
        // every draw anchors its object by the first bone's translation (Motion::Draw, Anchor).
        const u32 instances = regs.num_instances.NumInstances();
        bool gated = !all_motion;
        u64 palette = 0, instance_data = 0;
        Motion::Anchor anchor{};
#ifdef __APPLE__
        // bbport (macOS): the anchor from the most skeleton-like buffer (a character palette over
        // a large one over a small one), not the first: a 144-byte constant buffer of some
        // character shaders (a small palette by size) reads 0 0 1 one frame and 0 0 0 the next,
        // a jump every other frame and no history for those draws.
        const auto anchor_rank = [](Motion::BufferRole role) {
            return role == Motion::BufferRole::Skeleton        ? 3
                   : role == Motion::BufferRole::LargeSkeleton ? 2
                                                               : 1;
        };
        int anchor_from = 0;
#endif
        {
        BB_SECTION(MotionHash);
        for (const auto& resource : vs.buffers) {
            if (resource.IsSpecial()) continue;
            const auto buffer = resource.GetSharp(vs);
            const u32 size = buffer.GetSize();
            if (!buffer.Valid() || buffer.GetStride() != 16) continue;
            const auto role = Motion::ClassifyBuffer(size);
            if (role == Motion::BufferRole::Other) continue;
            const VAddr address = buffer.base_address;
            if (!memory->IsValidGpuMapping(address, 0) ||
                memory->ClampRangeSize(address, size) != size) {
                continue;
            }
            const void* data = reinterpret_cast<const void*>(address);
#ifdef __APPLE__
            if (anchor_rank(role) > anchor_from) {
                anchor_from = anchor_rank(role);
#else
            if (!anchor.valid) {
#endif
                const float* bone = static_cast<const float*>(data);
                anchor = {bone[3], bone[7], bone[11], true};
            }
            // The instances' 3x4 world matrices (a 48-byte stride buffer) up to the instance count;
            // past them and in the other buffers the game leaves per-frame data.
            if (instances > 1 && size % 48 == 0) {
                instance_data = XXH3_64bits_withSeed(data, std::min<u64>(size, u64(instances) * 48),
                                                     instance_data | 1);
            }
            if (role == Motion::BufferRole::Skeleton) {
                gated = false;
            } else if (gated) {
                palette = XXH3_64bits_withSeed(data, size, palette);
            }
        }
        }
        if (object_motion->Tracing()) {
            std::printf("Motion trace: vs %016llx geometry %llx indices %llx count %u inst %u gated %d",
                        (unsigned long long)vs.pgm_hash, (unsigned long long)motion_geometry,
                        (unsigned long long)index_address, regs.num_indices,
                        regs.num_instances.NumInstances(), int(gated));
            for (const auto& resource : vs.buffers) {
                if (resource.IsSpecial()) continue;
                const auto buffer = resource.GetSharp(vs);
                if (!buffer.Valid() || buffer.GetStride() != 16) continue;
                const u32 size = buffer.GetSize();
                const VAddr address = buffer.base_address;
                std::printf(" | buf %u @%llx", size, (unsigned long long)address);
                if (size >= 48 && memory->IsValidGpuMapping(address, 0) &&
                    memory->ClampRangeSize(address, 48) == 48) {
                    const float* m = reinterpret_cast<const float*>(address);
                    std::printf(" t0 %.2f %.2f %.2f", m[3], m[7], m[11]);
                    const u32 head = std::min<u32>(size, regs.num_instances.NumInstances() * 48);
                    std::printf(" h %08x/%08x", u32(XXH3_64bits(m, size)), u32(XXH3_64bits(m, head)));
                }
            }
            std::printf("\n");
        }
        push_data.motion_param = 0;
        bool moving = !gated;
        if (!moving) {
            BB_SECTION(MotionGate);
            moving = object_motion->Moving({.shader = vs.pgm_hash,
                                             .geometry = motion_geometry,
                                             .indices = index_address,
                                             .index_count = regs.num_indices,
                                             .instances = instances,
                                             .first_instance = first_instance},
                                            palette, anchor);
        }
        if (moving) {
            Motion::VertexRange range{base_vertex, regs.num_indices};
            u64 topology = 0;
            if (is_indexed) {
                range = {};
                const u64 bytes = u64(regs.num_indices) * index_size;
                if (index_address && memory->IsValidGpuMapping(index_address, 0) &&
                    memory->ClampRangeSize(index_address, bytes) == bytes) {
                    const bool restart = regs.enable_primitive_restart != 0;
                    const u32 count = regs.num_indices;
                    BB_SECTION(MotionIndex);
                    const auto scanned = object_motion->IndexRange(
                        {index_address, count, index_size, s32(base_vertex), restart}, [&] {
                            const auto* data = reinterpret_cast<const void*>(index_address);
                            const auto scan_range =
                                index_size == 2
                                    ? Motion::IndexedRange(
                                          std::span(static_cast<const u16*>(data), count),
                                          s32(base_vertex), restart)
                                    : Motion::IndexedRange(
                                          std::span(static_cast<const u32*>(data), count),
                                          s32(base_vertex), restart);
                            return Motion::IndexRangeCache::Result{scan_range,
                                                                   XXH3_64bits(data, bytes)};
                        });
                    range = scanned.range;
                    topology = scanned.topology;
                }
            }
            BB_SECTION(MotionPrepare);
            push_data.motion_param = object_motion->PrepareDraw({
                .shader = vs.pgm_hash,
                .geometry = motion_geometry,
                .indices = index_address,
                .topology = topology,
                .instance_data = instance_data,
                .index_count = regs.num_indices,
                .instances = instances,
                .first_instance = first_instance,
                .vertices = range,
            }, anchor);
        }
    }
    // BB_TRACE_SHADER with BB_GPU_PROFILE: a mark before the traced vertex shader's draws too.
    if (auto* profiler = GpuProfiler::Get()) {
        static const char* traced = bb_env("BB_TRACE_SHADER");
        const u64 vs_hash = pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash;
        if (traced && std::strtoull(traced, nullptr, 16) == vs_hash) {
            profiler->Mark(vs_hash ^ 0xBEF0'0000ull, [&] {
                return fmt::format("the draw of vs {:016x} (from before its pass begins)", vs_hash);
            });
        }
    }
    // Geometry drawn with the scene depth is jittered, full-screen passes are not (a shifted
    // full-screen quad leaves an edge column unwritten).
    // BB_TRACE_GPU_INPUTS=vs hash (diagnostics, the flicker): the buffer ranges this draw reads
    // (descriptors and vertex streams), copied by the GPU at the next display pass (DumpPassImages).
    if (static const u64 trace_inputs = [] {
            const char* env = bb_env("BB_TRACE_GPU_INPUTS");
            return env ? std::strtoull(env, nullptr, 16) : 0ull;
        }();
        trace_inputs && pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash == trace_inputs) {
        traced_inputs.clear();
        for (const auto& info : buffer_infos) {
            if (info.buffer && info.range != VK_WHOLE_SIZE && info.range <= 65536) {
                traced_inputs.push_back({info.buffer, info.offset, info.range});
            }
        }
        for (u32 i = 0; i < vertex_binds.num_buffers; ++i) {
            if (vertex_binds.host_buffers[i] && vertex_binds.host_sizes[i] <= 65536) {
                traced_inputs.push_back({vertex_binds.host_buffers[i], vertex_binds.host_offsets[i],
                                         vertex_binds.host_sizes[i]});
            }
        }
        traced_inputs_frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    }
    BeginDraw(pipeline, state, is_indexed,
              regs.num_indices > 6 || regs.num_instances.NumInstances() > 1);

    const auto& vs_info = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto& fetch_shader = pipeline->GetFetchShader();
    const auto [vertex_offset, instance_offset] = GetDrawOffsets(regs, vs_info, fetch_shader);

    const vk::Pipeline handle = pipeline->Handle();
    const u32 num_indices = regs.num_indices;
    // BB_TRACE_PIPELINE=vs hash (diagnostics, the flicker): its pipeline handle and push data,
    // printed when they differ from its previous draw.
    if (static const u64 traced_pipeline = [] {
            const char* env = bb_env("BB_TRACE_PIPELINE");
            return env ? std::strtoull(env, nullptr, 16) : 0ull;
        }();
        traced_pipeline && vs_info.pgm_hash == traced_pipeline) {
        static VkPipeline last_handle{};
        static std::array<float, 4> last_push{};
        const std::array<float, 4> push{push_data.xscale, push_data.xoffset, push_data.yscale,
                                        push_data.yoffset};
        static u32 prints = 0;
        if ((VkPipeline(handle) != last_handle || push != last_push) && prints++ < 2000) {
            std::printf("Pipeline trace: gpu frame %llu vs %016llx handle %p push %g %g %g %g "
                        "clipoff %d vte %#x\n",
                        (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                        (unsigned long long)vs_info.pgm_hash, (void*)VkPipeline(handle), push[0],
                        push[1], push[2], push[3], int(regs.IsClipDisabled()),
                        std::bit_cast<u32>(regs.viewport_control));
            last_handle = handle;
            last_push = push;
        }
    }
    const u32 num_instances = regs.num_instances.NumInstances();
    const u32 first_vertex = vertex_offset;
    const u32 first_instance = instance_offset;
    const auto* ps_info = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    // On NVIDIA by default (its hangs give no fault address): it costs the recording thread a
    // few T# and V# reads a draw. BB_BREADCRUMB_READS=1 / 0: on / off everywhere.
    static const bool crumb_reads = [&] {
        const char* env = bb_env("BB_BREADCRUMB_READS");
        return env ? env[0] == '1' : instance.GetVendorID() == 0x10de;
    }();
    // bbport (macOS): built only with breadcrumbs on (RecordCrumbWith).
    const auto make_crumb = [&] {
        Breadcrumbs::Crumb crumb{
            .kind = is_indexed ? Breadcrumbs::Kind::DrawIndexed : Breadcrumbs::Kind::Draw,
            .hash = {vs_info.pgm_hash, ps_info ? ps_info->pgm_hash : 0},
            .program = {vs_info.ProgramBase(), ps_info ? ps_info->ProgramBase() : 0},
            .count = {num_indices, num_instances, 0},
        };
        if (crumb_reads) {
            NoteCrumbReads(crumb, pipeline, regs, is_indexed, index_offset);
        }
        return crumb;
    };
    // BB_GPU_PROFILE=1 with BB_GPU_PROFILE_DRAWS=1: a timestamp before each draw labelled by its
    // shaders (the profile lists the dearest draws by shader pair).
    if (auto* profiler = GpuProfiler::Get()) {
        static const bool per_draw = bb_env_on("BB_GPU_PROFILE_DRAWS");
        if (per_draw) {
            const u64 vs = vs_info.pgm_hash, ps = ps_info ? ps_info->pgm_hash : 0;
            const u64 key = (vs * 0x9E3779B97F4A7C15ull) ^ (ps + 0x1234567ull);
            profiler->MarkInPass(key, [&] {
                return fmt::format("draw vs {:016x} ps {:016x}", vs, ps);
            });
        }
    }
    const auto draw = [=](vk::CommandBuffer cmdbuf) {
        if (is_indexed) {
            cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex), first_instance);
        } else {
            cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
        }
    };
    if (DynCheck::Enabled()) {
        const Scissors check_scissors = scheduler.GetDynamicState().CurrentScissors();
        const Viewports check_viewports = scheduler.GetDynamicState().CurrentViewports();
        const u64 check_vs = vs_info.pgm_hash, check_ps = ps_info ? ps_info->pgm_hash : 0;
        scheduler.RecordCrumbWith(make_crumb, [=](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
            DynCheck::Draw(cmdbuf, check_viewports, check_scissors, check_vs, check_ps);
            draw(cmdbuf);
        });
    } else {
        // bbport (macOS): without BB_DYNSTATE_CHECK the closure no longer carries its
        // viewports and scissors (about 650 bytes a draw).
        scheduler.RecordCrumbWith(make_crumb, [=](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
            draw(cmdbuf);
        });
    }
    // BB_TRACE_SHADER with BB_GPU_PROFILE: a mark right after the traced vertex shader's draws,
    // so the profile splits their pass into the draw and what follows it unlabeled.
    if (auto* profiler = GpuProfiler::Get()) {
        static const char* traced = bb_env("BB_TRACE_SHADER");
        if (traced && std::strtoull(traced, nullptr, 16) == vs_info.pgm_hash) {
            profiler->Mark(vs_info.pgm_hash ^ 0xAF7E'0000ull, [&] {
                return fmt::format("after the draw of vs {:016x}", vs_info.pgm_hash);
            });
        }
    }
    // bbport: the game's AO is drawn (either path); VBAO overwrites its target (docs/vbao.md).
    if (ps_info && Vbao::AoShader(ps_info->pgm_hash)) {
        // The G-buffer normals the game's SSAO reads (its texture 1): the normal-map detail.
        VideoCore::ImageId normals{};
        if (ps_info->images.size() > 1) {
            const auto& res = ps_info->images[1];
            const auto& entry = CachedImageDescEntry(res.GetSharp(*ps_info), res);
            if (entry.found_generation == texture_cache.RegistryGeneration()) {
                normals = entry.found_id;
            }
        }
        vbao->OnAoPass(cb_descs[0].first, normals, upscaler->Accumulating());
    }
    if (FrameCapture::Active()) {
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        FrameCapture::Draw(vs_info.pgm_hash, ps ? ps->pgm_hash : 0, num_indices, num_instances);
        if (const auto& bc = regs.blend_control[0]; bc.enable && regs.color_buffers[0]) {
            char note[128];
            std::snprintf(note, sizeof(note), "\n  blend ps %08x idx %u: src %u dst %u func %u z %u%u",
                          u32(ps ? ps->pgm_hash : 0), num_indices, u32(bc.color_src_factor),
                          u32(bc.color_dst_factor), u32(bc.color_func),
                          u32(regs.depth_control.depth_enable),
                          u32(regs.depth_control.depth_write_enable));
            FrameCapture::Note(note);
        }
        // Shadow maps and clears: where the draw lands in its depth target and how it clears.
        if (db_desc.first && (texture_cache.GetImage(db_desc.first).info.size.width >= 2048 ||
                              regs.depth_render_control.depth_clear_enable ||
                              regs.depth_render_control.stencil_clear_enable)) {
            const auto& vp = regs.viewports[0];
            const auto rect = ClearDrawRect(16384, 16384);
            char note[320];
            std::snprintf(note, sizeof(note),
                          "\n  depth draw idx %u: vp off %g,%g scale %g,%g z %g+%g rect %d,%d %ux%u "
                          "dclr %u sclr %u load-clear %u htile %#llx slice %u func %u "
                          "z %u%u clip-off %u",
                          num_indices, vp.xoffset, vp.yoffset, vp.xscale, vp.yscale, vp.zoffset,
                          vp.zscale, rect.offset.x, rect.offset.y, rect.extent.width,
                          rect.extent.height, u32(regs.depth_render_control.depth_clear_enable),
                          u32(regs.depth_render_control.stencil_clear_enable),
                          u32(state.depth_stencil_attachment.depth_clear),
                          (unsigned long long)regs.depth_htile_data_base.GetAddress(),
                          u32(db_desc.second.view_info.range.base.layer),
                          u32(regs.depth_control.depth_func),
                          u32(regs.depth_control.depth_enable),
                          u32(regs.depth_control.depth_write_enable),
                          u32(regs.IsClipDisabled()));
            FrameCapture::Note(note);
        }
    }

    static const bool watch = bb_env_on("BB_DRAW_WATCH");
    if (watch) {
        WatchDraw(pipeline, state);
    }

    BB_SECTION(Kick);
    ResetBindings(false);
    scheduler.KickRecording();
}

void Rasterizer::NoteUpscalerDraw(const GraphicsPipeline* pipeline) {
    // Single-target passes: the upscaler looks for the scene's and the UI's (UiComposition).
    if (upscaler->Enabled() && std::popcount(pipeline->GetGraphicsKey().mrt_mask) == 1) {
        const auto& viewport = Regs().viewports[0];
        upscaler->OnDraw(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash,
                         cb_descs[0].first, db_desc.first,
                         UiComposition::NativeViewport(viewport.xscale * 2, viewport.yscale * 2));
    }
}

void Rasterizer::ScaleToTarget(const GraphicsPipeline* pipeline) {
    UiEdgeStretch(pipeline);
    // Screen-space (clip disabled) draws into the upscaler's output-size images.
    push_data.xscale *= target_scale[0];
    push_data.xoffset *= target_scale[0];
    push_data.yscale *= target_scale[1];
    push_data.yoffset *= target_scale[1];
}

void Rasterizer::BeginDraw(const GraphicsPipeline* pipeline, const RenderState& state,
                           bool is_indexed, bool jitter_eligible) {
    // The upscaler's sub-pixel jitter, for geometry drawn with the scene depth.
    draw_jitter = {};
    if (jitter_eligible && upscaler->Enabled() && db_desc.first &&
        db_desc.first == camera_motion->Depth()) {
        draw_jitter = upscaler->Jitter();
    }
    // A new render pass sets its viewports and scissors again, even when the state tracker has
    // them already: single frames were drawn with the 960x540 viewport of the passes before (the
    // final picture in the top-left quarter, black around it; docs/Flicker_video_2026-10-17.ru.md).
    // BB_DYNAMIC_STATE_ALWAYS=1 (diagnostics): all dynamic state again at every draw.
    static const bool dynamic_always = bb_env_on("BB_DYNAMIC_STATE_ALWAYS");
    if (dynamic_always) {
        scheduler.GetDynamicState().Invalidate();
    } else if (scheduler.WillBeginRendering(state)) {
        scheduler.GetDynamicState().InvalidateViewports();
    }
    UpdateDynamicState(pipeline, is_indexed);
    // BB_DISPLAY_WATCH=1 (diagnostics): every draw into a display buffer, with its sizes and state.
    static const bool display_watch = bb_env("BB_DISPLAY_WATCH") != nullptr;
    // BB_DISPLAY_WATCH_TARGET=addr: also every draw into that color target.
    static const VAddr watch_target = [] {
        const char* env = bb_env("BB_DISPLAY_WATCH_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (display_watch && Regs().color_buffers[0] &&
        (FrameCapture::IsDisplayBuffer(Regs().color_buffers[0].Address()) ||
         Regs().color_buffers[0].Address() == watch_target)) {
        const auto& rvp = Regs().viewports[0];
        std::printf("Display watch regs: vp0 %g %g %g %g vte %#x jitter %g,%g new pass %d\n", rvp.xscale,
                    rvp.xoffset, rvp.yscale, rvp.yoffset, std::bit_cast<u32>(Regs().viewport_control),
                    draw_jitter[0], draw_jitter[1], int(scheduler.WillBeginRendering(state)));
        const auto& ds = scheduler.GetDynamicState();
        const auto& vps = ds.CurrentViewports();
        const auto& scs = ds.CurrentScissors();
        const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
        const auto& image = texture_cache.GetImage(cb_descs[0].first).info;
        std::printf("Display watch: gpu frame %llu flips %llu vs %08x ps %08x pass %ux%u image %ux%u scale %g,%g "
                    "vp %g,%g %gx%g sc %d,%d %ux%u clipoff %d push %g,%g,%g,%g\n",
                    (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                    (unsigned long long)FrameCapture::Flips(),
                    u32(pipeline->GetStage(Shader::SwStage::Vertex).pgm_hash), u32(ps ? ps->pgm_hash : 0),
                    state.width, state.height, image.size.width, image.size.height, target_scale[0],
                    target_scale[1], vps.empty() ? 0.f : vps[0].x, vps.empty() ? 0.f : vps[0].y,
                    vps.empty() ? 0.f : vps[0].width, vps.empty() ? 0.f : vps[0].height,
                    scs.empty() ? 0 : scs[0].offset.x, scs.empty() ? 0 : scs[0].offset.y,
                    scs.empty() ? 0u : scs[0].extent.width, scs.empty() ? 0u : scs[0].extent.height,
                    int(Regs().IsClipDisabled()), push_data.xscale, push_data.xoffset,
                    push_data.yscale, push_data.yoffset);
    }
    MarkPass(pipeline, state);
    const bool pass_begins = scheduler.WillBeginRendering(state);
    if (occlusion) {
        occlusion->SetPassScale(target_scale[0] * target_scale[1]); // samples at the game's size
    }
    scheduler.BeginRendering(state);
    // BB_GPU_PROFILE_DRAWS=1: the pass's beginning (what its boundary recorded, its load ops) apart
    // from its draws.
    if (auto* profiler = GpuProfiler::Get(); profiler && pass_begins) {
        static const bool per_draw = bb_env_on("BB_GPU_PROFILE_DRAWS");
        if (per_draw) {
            profiler->MarkInPass(0xB361'0000ull, [] { return std::string{"(a pass begun: boundary, load ops)"}; });
        }
    }
    // Push constants after the pass begins: beginning it may record the occlusion results' compute
    // dispatch (OcclusionQueries::Suspend), and the driver keeps one set of push constants for
    // compute and graphics. Its 8 bytes over xoffset/yoffset drew single frames shifted by half the
    // screen (docs/Flicker_2026-10-18.ru.md).
    {
        BB_SECTION(PipelineBind);
        pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                                {buffer_infos.data(), buffer_infos.size()});
    }
}

void Rasterizer::NoteCopyIntoMemory(VAddr address, u64 size) {
    if (size == 0 || !OnStageA()) {
        return;
    }
    copies_into_memory[num_copies_into_memory++ % copies_into_memory.size()] = {address,
                                                                               address + size};
    if (num_copies_into_memory % copies_into_memory.size() == 0) {
        // Once around the ring: the bounds of the copies kept (the ones pushed out no longer).
        copies_begin = ~VAddr{0};
        copies_end = 0;
        for (const auto& copy : copies_into_memory) {
            copies_begin = std::min(copies_begin, copy.begin);
            copies_end = std::max(copies_end, copy.end);
        }
        return;
    }
    copies_begin = std::min(copies_begin, address);
    copies_end = std::max<VAddr>(copies_end, address + size);
}

// Program code the GPU copies into place (DMA in the stream before the draw): this thread decodes
// the draw ahead of the copy and read the code that was there before (another program: a
// character's vertices stretched into streaks in another game; or zeros). Code under a copy noted
// since the GPU last ran everything: everything before it runs first. Rare (shader uploads); the
// last 64 copies are kept (a game copies far more a frame: a wait for each program check after
// the ring filled cost a quarter of this thread).
void Rasterizer::SyncProgramWrites(VAddr address, u64 size) {
    if (address + size <= copies_begin || address >= copies_end || !OnStageA()) {
        return;
    }
    bool overlaps = false;
    const u32 kept = std::min<u32>(num_copies_into_memory, u32(copies_into_memory.size()));
    for (u32 i = 0; !overlaps && i < kept; ++i) {
        const auto& copy = copies_into_memory[i];
        overlaps = copy.begin < address + size && address < copy.end;
    }
    if (!overlaps) {
        return;
    }
    DrainDrawPipe();
    Finish();
    num_copies_into_memory = 0;
    copies_begin = ~VAddr{0};
    copies_end = 0;
}

void Rasterizer::DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 stride,
                              u32 max_count, VAddr count_address, u16 vertex_sgpr_offset,
                              u16 instance_sgpr_offset) {
    // bbport: like direct draws, handed to the recording thread after the pipeline selection
    // (else the GPU thread waited here for a whole frame of queued draws).
    const bool pipelined = UseDrawPipe() && FilterDrawPasses() &&
                           !BbToggle::Disabled(BbToggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
        scheduler.PopPendingOperations();
        if (!FilterDraw()) {
            return;
        }
    }

    const DrawIndirectParams params = {
        .vertex_sgpr_offset = vertex_sgpr_offset,
        .instance_sgpr_offset = instance_sgpr_offset,
    };
    const GraphicsPipeline* pipeline = pipeline_cache.GetGraphicsPipeline(params, nullptr, true);
    if (!pipeline) {
        return;
    }
    const IndirectDraw indirect = {arg_address + offset, count_address, stride, max_count};
    if (pipelined) {
        PostDraw(pipeline, nullptr, is_indexed, 0, nullptr, &indirect);
        return;
    }
    DrawIndirectRecord(pipeline, is_indexed, indirect);
}

void Rasterizer::DrawIndirectRecord(const GraphicsPipeline* pipeline, bool is_indexed,
                                    const IndirectDraw& indirect) {
#ifdef __APPLE__
    if (CountTessellation(pipeline)) {
        return; // BB_SKIP_TESS_DRAWS (a test)
    }
#endif
    ++work_after_early;
    ++work_total;
    buffer_cache.NewPacket();
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
    }
    if (IsNoopDraw(Regs(), pipeline)) {
        return;
    }
#ifdef __APPLE__
    if (SkippedPatchDraw(pipeline)) {
        return;
    }
#endif
    const VAddr count_address = indirect.count;
    const u32 stride = indirect.stride;
    const u32 max_count = indirect.max_count;

    // Indirect arguments may be GPU-written: leave these draws on camera fallback.
    motion_draw = false;
    motion_geometry = 0;
    PrepareRenderState(pipeline);
    NoteUpscalerDraw(pipeline);
    if (!BindResources(pipeline)) {
        return;
    }
    const auto [buffer, base] =
        buffer_cache.ObtainBuffer(indirect.args, stride * max_count, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, stride * max_count);
    // The arguments checked on the GPU first (vk_indirect_guard.h): a dispatch, outside the
    // render pass.
    vk::Buffer checked_args = buffer->Handle();
    u64 checked_offset = base;
    if (IndirectGuard::Enabled()) {
        if (!indirect_guard) {
            indirect_guard = std::make_unique<IndirectGuard>(instance, scheduler);
        }
        if (needs_barrier) {
            runtime.FlushBarriers();
            needs_barrier = false;
        }
        scheduler.EndRendering();
        std::tie(checked_args, checked_offset) =
            indirect_guard->CheckDrawArgs(*buffer, base, stride, max_count, is_indexed);
    }
    const auto state = BeginRendering(pipeline);

    BindVertexBuffers(pipeline);
    if (is_indexed) {
        BindIndexBuffer();
    }

    const VideoCore::Buffer* count_buffer;
    u64 count_offset;
    if (count_address != 0) {
        std::tie(count_buffer, count_offset) = buffer_cache.ObtainBuffer(count_address, 4, false);
        needs_barrier |= runtime.IsBufferAccessed(count_buffer, count_offset, 4);
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    ScaleToTarget(pipeline);
    BeginDraw(pipeline, state, is_indexed, true);

    ASSERT(stride == (is_indexed ? sizeof(VkDrawIndexedIndirectCommand)
                                 : sizeof(VkDrawIndirectCommand)));
    const vk::Pipeline handle = pipeline->Handle();
    const vk::Buffer args = checked_args;
    const u64 args_offset = checked_offset;
    const vk::Buffer counts = count_address != 0 ? count_buffer->Handle() : vk::Buffer{};
    const u64 counts_offset = count_address != 0 ? count_offset : 0;
    const auto* vs_crumb = pipeline->GetStages()[u32(Shader::SwStage::Vertex)];
    const auto* ps_crumb = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    // bbport (macOS): built only with breadcrumbs on (RecordCrumbWith).
    const auto make_crumb = [&] {
        return Breadcrumbs::Crumb{
            .kind = Breadcrumbs::Kind::DrawIndirect,
            .hash = {vs_crumb ? vs_crumb->pgm_hash : 0, ps_crumb ? ps_crumb->pgm_hash : 0},
            .program = {vs_crumb ? vs_crumb->ProgramBase() : 0,
                        ps_crumb ? ps_crumb->ProgramBase() : 0},
            .address = indirect.args,
            .count = {max_count, stride, 0},
        };
    };
    if (auto* profiler = GpuProfiler::Get()) {
        static const bool per_draw = bb_env_on("BB_GPU_PROFILE_DRAWS");
        if (per_draw) {
            const u64 vs = vs_crumb ? vs_crumb->pgm_hash : 0, ps = ps_crumb ? ps_crumb->pgm_hash : 0;
            const u64 key = (vs * 0x9E3779B97F4A7C15ull) ^ (ps + 0x7654321ull);
            profiler->MarkInPass(key, [&] {
                return fmt::format("indirect draw vs {:016x} ps {:016x} max {}{}", vs, ps, max_count,
                                   counts ? " counted" : "");
            });
        }
    }
    scheduler.RecordCrumbWith(make_crumb, [=](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        if (is_indexed) {
            if (counts) {
                cmdbuf.drawIndexedIndirectCount(args, args_offset, counts, counts_offset, max_count,
                                                stride);
            } else {
                cmdbuf.drawIndexedIndirect(args, args_offset, max_count, stride);
            }
        } else if (counts) {
            cmdbuf.drawIndirectCount(args, args_offset, counts, counts_offset, max_count, stride);
        } else {
            cmdbuf.drawIndirect(args, args_offset, max_count, stride);
        }
    });

    ResetBindings(false);
    scheduler.KickRecording();
}

namespace {
/// BB_TIMELINE: the address FlushCaller events are relative to (symbolized with the binary).
[[gnu::noinline]] void FlushAnchor() {
    asm volatile("");
}
} // namespace

[[gnu::noinline]] u64 Rasterizer::Flush() {
    BB_SECTION(Flush);
    BbTimeline::Note(BbTimeline::FlushCaller, 0,
                     u64(__builtin_return_address(0)) - u64(&FlushAnchor));
    DrainDrawPipe();
    PublishSignals(); // the recording thread's (it is idle here, or this is it)
    last_flush_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                        std::memory_order_relaxed);
    const u64 current_tick = scheduler.CurrentTick();
    SubmitInfo info{};
    scheduler.Flush(info);
    BbTimeline::Note(BbTimeline::VulkanSubmit, current_tick, DrawPipe::OnStageB() ? 1 : 0);
    return current_tick;
}

void Rasterizer::Finish() {
    DrainDrawPipe();
    PublishSignals();
    scheduler.Finish();
}

void Rasterizer::ProcessDeadImagesInOrder() {
    // BB_DEAD_IMAGES_LATE=1: at the frame's end only, as before (to compare).
    static const bool late = bb_env("BB_DEAD_IMAGES_LATE") != nullptr;
    if (late || !buffer_cache.DeadImagesPending()) {
        return;
    }
    const u64 started = BbStats::started_submit_stamp.load(std::memory_order_acquire);
    RunInOrder(
        [](Rasterizer& self, const u8* data) {
            u64 started_stamp;
            std::memcpy(&started_stamp, data, sizeof(started_stamp));
            self.buffer_cache.ProcessDeadImages(started_stamp);
        },
        &started, sizeof(started), BbToggle::PipelinedTasks, false);
}

void Rasterizer::NewGuestWorkInOrder() {
    // The copies of the guest submission before it serve no draw of this one (the game may have
    // written the data again in between).
    // (Live toggles: always, so copies made before a switch-off never serve after a switch-on.)
    static const bool may_copy = bb_env_on("BB_HOT_VERTEX_COPY") ||
                                 bb_env_on("BB_VERTEX_READ_COPY") || bb_env("BB_TOGGLE_FILE");
    if (!may_copy) {
        return;
    }
    RunInOrder([](Rasterizer& self, const u8*) { self.buffer_cache.ClearHotCopies(); }, nullptr, 0,
               BbToggle::PipelinedTasks, false);
}

void Rasterizer::OnSubmit(u64 started_stamp) {
    DrainDrawPipe();
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
    buffer_cache.ProcessDeadImages(started_stamp);
    texture_cache.RunGarbageCollector();
    runtime.TickFrame();
}

void Rasterizer::MarkPass(const GraphicsPipeline* pipeline, const RenderState& state) {
    auto* profiler = GpuProfiler::Get();
    if (!profiler || !scheduler.WillBeginRendering(state)) {
        return;
    }
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    std::array<u64, Gcn::NUM_COLOR_BUFFERS + 6> parts{};
    parts[0] = state.width | u64(state.height) << 32;
    parts[1] = vs.pgm_hash;
    parts[2] = ps ? ps->pgm_hash : 0;
    parts[3] = db_desc.first ? u64(texture_cache.GetImage(db_desc.first).info.pixel_format) : 0;
    for (u32 cb = 0; cb < state.num_color_attachments && cb < Gcn::NUM_COLOR_BUFFERS; ++cb) {
        parts[4 + cb] = cb_descs[cb].first
                            ? u64(texture_cache.GetImage(cb_descs[cb].first).info.pixel_format) + 1
                            : 0;
    }
    const u64 key = XXH3_64bits(parts.data(), sizeof(parts));
    profiler->Mark(key, [&] {
        std::string targets;
        for (u32 cb = 0; cb < state.num_color_attachments; ++cb) {
            targets += cb_descs[cb].first
                           ? vk::to_string(texture_cache.GetImage(cb_descs[cb].first).info.pixel_format)
                           : std::string{"-"};
            targets += ' ';
        }
        return fmt::format("pass {}x{} [{}] depth {} vs {:016x} ps {:016x}", state.width,
                           state.height, targets,
                           db_desc.first ? vk::to_string(
                                               texture_cache.GetImage(db_desc.first).info.pixel_format)
                                         : std::string{"-"},
                           vs.pgm_hash, ps ? ps->pgm_hash : 0);
    });
}

void Rasterizer::NoteFrameStart() {
    if (auto* profiler = GpuProfiler::Get()) {
        profiler->BeginFrame();
    }
    const u64 frame = BbStats::gpu_frames.fetch_add(1, std::memory_order_relaxed) + 1;
    // BB_BUFFER_STATS=1: how many earlier frames the GPU has not finished when the GPU thread
    // starts a frame — the lag that decides whether guest memory can be read in place.
    static const bool stats = bb_env_on("BB_BUFFER_STATS");
    if (!stats) {
        return;
    }
    static std::array<u64, 16> frame_ticks{};
    static std::array<u64, 17> lag_histogram{};
    u32 lag = 0;
    for (u32 back = 1; back < frame_ticks.size() && back < frame; ++back) {
        const u64 tick = frame_ticks[(frame - back) % frame_ticks.size()];
        if (tick && !scheduler.IsFree(tick)) {
            lag = back;
        }
    }
    ++lag_histogram[lag];
    frame_ticks[frame % frame_ticks.size()] = scheduler.CurrentTick();
    if (frame % 600 == 0) {
        std::printf("GPU lag at frame start (frames not finished: count):");
        for (u32 i = 0; i < lag_histogram.size(); ++i) {
            if (lag_histogram[i]) {
                std::printf(" %u:%llu", i, static_cast<unsigned long long>(lag_histogram[i]));
            }
        }
        std::printf("\n");
        lag_histogram = {};
    }
}

} // namespace Vulkan
