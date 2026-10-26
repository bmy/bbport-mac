// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, the draw and dispatch entry points: pipeline selection, then recording here or on the
// draw recording thread (raster_pipe.cpp), and the per-frame bookkeeping.

#include <mutex>
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
    camera_motion = std::make_unique<CameraMotion>(instance, scheduler, texture_cache, runtime);
    camera_motion->SetObjectMotion(object_motion.get());
    upscaler = std::make_unique<TemporalUpscaler>(instance, scheduler, texture_cache, runtime,
                                                  *camera_motion, *scene_targets);
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
        // BB_RESOLVE_WORKERS=n (stage 2, docs/pipeline_4stage.ru.md): threads that check and copy
        // the constants of draws between the GPU command thread and the recording thread.
        static const u32 resolvers = [] {
            const char* env = std::getenv("BB_RESOLVE_WORKERS");
            return env ? u32(std::clamp(std::atoi(env), 0, 4)) : 0u;
        }();
        if (resolvers != 0) {
            for (u32 i = 0; i < resolvers; ++i) {
                worker_rings.push_back(std::make_unique<ConstantRing>(instance, scheduler));
            }
            draw_pipe->StartResolvers(&ResolvePacket, resolvers);
            std::printf("GPU: %u constant resolve workers (BB_RESOLVE_WORKERS)\n", resolvers);
        }
    }
    // bbport: this thread joins the texture binding helper before it changes image state.
    runtime.SetImageAccessHook(&JoinBindHelper, this);
    scheduler.SetSubmitCallback([this](Vulkan::SubmitInfo& info) {
        runtime.FlushBarriers();
        buffer_cache.SubmitPendingArenaBinds(info);
    });
}

Rasterizer::~Rasterizer() = default;

namespace {
/// BB_PASS_DUMP=addr,addr (hex guest addresses of render targets) with BB_PASS_DUMP_TRIGGER=<file>
/// (diagnostics, the flicker): while the file exists, every 8th row of those images is saved at
/// each display pass (BB_DUMP_DIR/pNNNNN_<addr>_<w>x<h>_<bits>.raw), little enough not to change
/// the frame's timing much.
std::vector<VAddr> PassDumpAddresses() {
    std::vector<VAddr> out;
    if (const char* env = std::getenv("BB_PASS_DUMP")) {
        for (const char* at = env; *at;) {
            char* end = nullptr;
            out.push_back(std::strtoull(at, &end, 16));
            at = *end ? end + 1 : end;
        }
    }
    return out;
}
} // namespace

void Rasterizer::DumpTracedInputs() {
    if (traced_inputs.empty()) {
        return;
    }
    static const std::string dir = [] {
        const char* env = std::getenv("BB_DUMP_DIR");
        return std::string{env && env[0] ? env : "out/dump"};
    }();
    VkDeviceSize total = 0;
    for (const auto& r : traced_inputs) total += r.size;
    const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                       .size = total,
                                       .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    const VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                        &info) != VK_SUCCESS) {
        traced_inputs.clear();
        return;
    }
    scheduler.EndRendering();
    auto ranges = traced_inputs;
    scheduler.Record([ranges, buffer](vk::CommandBuffer cmd) {
        const vk::MemoryBarrier2 before{
            .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
            .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
            .dstAccessMask = vk::AccessFlagBits2::eTransferRead,
        };
        cmd.pipelineBarrier2({.memoryBarrierCount = 1, .pMemoryBarriers = &before});
        VkDeviceSize at = 0;
        for (const auto& r : ranges) {
            cmd.copyBuffer(r.buffer, buffer, vk::BufferCopy{r.offset, at, r.size});
            at += r.size;
        }
    });
    std::string header;
    for (const auto& r : traced_inputs) {
        header += fmt::format("{:x}+{:x}:{} ", (u64)(VkBuffer)r.buffer, r.offset, r.size);
    }
    char path[512];
    std::snprintf(path, sizeof(path), "%s/g%06llu.bin", dir.c_str(),
                  (unsigned long long)traced_inputs_frame);
    scheduler.DeferPriorityOperation([allocator = instance.GetAllocator(), buffer, allocation, info,
                                      total, file = std::string{path}, header] {
        vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
        if (FILE* f = std::fopen(file.c_str(), "wb")) {
            const u32 n = u32(header.size());
            std::fwrite(&n, 4, 1, f);
            std::fwrite(header.data(), 1, n, f);
            std::fwrite(info.pMappedData, 1, total, f);
            std::fclose(f);
        }
        vmaDestroyBuffer(allocator, buffer, allocation);
    });
    traced_inputs.clear();
}

void Rasterizer::DumpPassImages(char tag) {
    static const std::vector<VAddr> addresses = PassDumpAddresses();
    static const char* trigger = std::getenv("BB_PASS_DUMP_TRIGGER");
    if (addresses.empty() || !trigger || !std::filesystem::exists(trigger)) {
        return;
    }
    static const std::string dir = [] {
        const char* env = std::getenv("BB_DUMP_DIR");
        return std::string{env && env[0] ? env : "out/dump"};
    }();
    static u32 indices[2] = {};
    const u32 n = indices[tag == 'p' ? 0 : 1]++;
    std::printf("Pass dump %c %u: GPU frame %llu\n", tag, n,
                (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed));
    scheduler.EndRendering();
    for (const VAddr address : addresses) {
        VideoCore::Image* found = nullptr;
        texture_cache.ForEachImageInRegion(address, 1, [&](VideoCore::ImageId, VideoCore::Image& image) {
            if (image.info.guest_address == address &&
                (!found || image.info.size.width > found->info.size.width)) {
                found = &image;
            }
        });
        if (!found || found->info.num_bits == 0) {
            continue;
        }
        const u32 width = found->info.size.width, height = found->info.size.height;
        const u32 bytes = found->info.num_bits / 8;
        static const u32 step = [] {
            const char* env = std::getenv("BB_PASS_DUMP_STEP");
            return env ? std::max(1, std::atoi(env)) : 8;
        }();
        const u32 rows = (height + step - 1) / step;
        const VkDeviceSize size = VkDeviceSize(width) * rows * bytes;
        const VkBufferCreateInfo buffer_ci{.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                           .size = size,
                                           .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        const VmaAllocationCreateInfo alloc_ci{
            .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
        VkBuffer buffer{};
        VmaAllocation allocation{};
        VmaAllocationInfo info{};
        if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer, &allocation,
                            &info) != VK_SUCCESS) {
            continue;
        }
        runtime.Transit(found, vk::ImageLayout::eTransferSrcOptimal,
                        vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
        runtime.FlushBarriers();
        const vk::Image image = found->GetImage();
        const vk::ImageAspectFlags aspect = found->aspect_mask & ~vk::ImageAspectFlagBits::eStencil;
        scheduler.Record([=](vk::CommandBuffer cmd) {
            std::vector<vk::BufferImageCopy> regions;
            regions.reserve(rows);
            for (u32 r = 0; r < rows; ++r) {
                regions.push_back({
                    .bufferOffset = VkDeviceSize(r) * width * bytes,
                    .imageSubresource = {aspect, 0, 0, 1},
                    .imageOffset = {0, s32(r * step), 0},
                    .imageExtent = {width, 1, 1},
                });
            }
            cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, regions);
        });
        char path[512];
        std::snprintf(path, sizeof(path), "%s/%c%05u_%llx_%ux%u_%u.raw", dir.c_str(), tag, n,
                      (unsigned long long)address, width, rows, found->info.num_bits);
        scheduler.DeferPriorityOperation([allocator = instance.GetAllocator(), buffer, allocation,
                                          info, size, file = std::string{path}] {
            vmaInvalidateAllocation(allocator, allocation, 0, VK_WHOLE_SIZE);
            if (FILE* f = std::fopen(file.c_str(), "wb")) {
                std::fwrite(info.pMappedData, 1, size, f);
                std::fclose(f);
            }
            vmaDestroyBuffer(allocator, buffer, allocation);
        });
    }
}


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

namespace {
/// bbport (macOS) experiment bits of BB_TOGGLE_FILE (its second number), to measure what draws
/// drawn as patches cost on drivers that tessellate in software (KosmicKrisp): 128 leaves out
/// the game's tessellated draws, 256 the rect and quad lists (drawn as patches). The picture is
/// wrong while either is set: for GPU time measurements only.
bool SkippedPatchDraw(const GraphicsPipeline* pipeline) {
    const bool skip_tess = BbToggle::Experiment(128), skip_lists = BbToggle::Experiment(256);
    if (!skip_tess && !skip_lists) {
        return false;
    }
    const auto prim = pipeline->GetGraphicsKey().prim_type;
    const bool list = prim == Gcn::PrimitiveType::RectList || prim == Gcn::PrimitiveType::QuadList;
    return (skip_lists && list) || (skip_tess && prim == Gcn::PrimitiveType::PatchPrimitive);
}
} // namespace

void Rasterizer::Draw(bool is_indexed, u32 index_offset, const PreparedDraw* prepared) {
    BB_SECTION(ADraw);
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

void Rasterizer::DrawRecord(const GraphicsPipeline* pipeline, const PreparedDraw* used_prepared,
                            bool is_indexed, u32 index_offset) {
    BB_SECTION(DrawRecord);
    buffer_cache.NewPacket();
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
    }
    // bbport (BB_PREUPLOAD): once a frame, guest GPU memory the CPU wrote and then left alone is
    // uploaded ahead of its use, BB_PREUPLOAD_MB (16) at most per frame.
    if (const u32 frame = BbStats::frame_number.load(std::memory_order_relaxed);
        frame != preupload_frame) {
        preupload_frame = frame;
        BB_SECTION(Preupload);
        static const u64 budget = [] {
            const char* env = std::getenv("BB_PREUPLOAD_MB");
            return (env ? std::strtoull(env, nullptr, 10) : 16ull) << 20;
        }();
        buffer_cache.Preupload(budget);
    }
    const auto& regs = Regs();
    // bbport: the pass copying a finished frame to a display buffer; the previous draw's
    // target is that frame.
    if (camera_motion->Enabled() && regs.color_buffers[0] &&
        FrameCapture::IsDisplayBuffer(regs.color_buffers[0].Address())) {
        scene_started = false;
        DumpTracedInputs();
        DumpPassImages('p');
        upscaler->OnDisplayPassStart(regs.color_buffers[0].Address());
        camera_motion->OnDisplayPass(cb_descs[0].first);
        if (upscaler->OnFrameStart()) {
            object_motion->InvalidateHistory();
            camera_motion->InvalidateHistory();
        }
        camera_motion->SetJitter(upscaler->Jitter());
        object_motion->OnFrameStart();
        NoteFrameStart();
        static const char* scene_debug = std::getenv("BB_SCENE_DEBUG");
        scene_debug_frame = scene_debug && std::remove(scene_debug) == 0;
        scene_targets->debug = scene_debug_frame;
    }
    // BB_SKIP_PS=hash[,hash...] (diagnostics): draws with these pixel shaders are left out.
    static const std::vector<u64> skipped_ps = [] {
        std::vector<u64> hashes;
        if (const char* env = std::getenv("BB_SKIP_PS")) {
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
    if (SkippedPatchDraw(pipeline)) {
        return;
    }
    // bbport: the key guide icons for the device in use (a change: uploaded before this draw).
    prompts->Update();
    bind_prepared = used_prepared;
    motion_draw = pipeline->GetGraphicsKey().motion_vectors;
    motion_geometry = 0;

    // BB_EXP_FINISH_BEFORE_TARGET=addr (experiment, the flicker): the GPU runs everything before
    // a draw into that color target before the draw is recorded.
    static const VAddr finish_target = [] {
        const char* env = std::getenv("BB_EXP_FINISH_BEFORE_TARGET");
        return env ? VAddr(std::strtoull(env, nullptr, 16)) : VAddr{0};
    }();
    if (finish_target && Regs().color_buffers[0] &&
        Regs().color_buffers[0].Address() == finish_target) {
        scheduler.Finish();
    }
    // BB_PASS_DUMP_AT=addr (diagnostics): BB_PASS_DUMP's images also before the first draw of a
    // pass into that color target ('q' files).
    static const VAddr dump_at = [] {
        const char* env = std::getenv("BB_PASS_DUMP_AT");
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
        const char* env = std::getenv("BB_EXP_TRANSIT_TARGET");
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
        const char* env = std::getenv("BB_EXP_BARRIER_TARGET");
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
    NoteUpscalerDraw(pipeline);
    const PreparedDraw* draw_prepared = bind_prepared;
    // bbport: vertex and index buffers are resolved while the helper binds textures (their
    // commands are recorded after BeginRendering, as before).
    draw_inputs = {pipeline, draw_prepared, index_offset, is_indexed, true, false};
    const bool bound = BindResources(pipeline);
    bind_prepared = nullptr; // indirect draws and dispatches bind without prepared sharps
    const bool inputs_resolved = draw_inputs.resolved;
    draw_inputs.pending = false;
    if (!bound) {
        return;
    }
    const auto state = BeginRendering(pipeline);

    if (!inputs_resolved) {
        ResolveVertexBuffers(pipeline, draw_prepared);
        if (is_indexed) {
            ResolveIndexBuffer(index_offset);
        }
    }
    EmitVertexBuffers();
    if (is_indexed) {
        EmitIndexBuffer();
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
        static const bool all_motion = [] {
            const char* value = std::getenv("BB_OBJECT_MOTION_ALL");
            return value && value[0] == '1';
        }();
        bool gated = !all_motion;
        u64 palette = 0;
        for (const auto& resource : vs.buffers) {
            if (resource.IsSpecial()) continue;
            const auto buffer = resource.GetSharp(vs);
            const u32 size = buffer.GetSize();
            if (!buffer.Valid() || buffer.GetStride() != 16) continue;
            const auto role = Motion::ClassifyBuffer(size);
            if (role == Motion::BufferRole::Skeleton) {
                gated = false;
                break;
            }
            const VAddr address = buffer.base_address;
            if (role == Motion::BufferRole::SmallSkeleton &&
                memory->IsValidGpuMapping(address, 0) &&
                memory->ClampRangeSize(address, size) == size) {
                palette = XXH3_64bits_withSeed(reinterpret_cast<const void*>(address), size,
                                               palette);
            }
        }
        push_data.motion_param = 0;
        if (!gated || object_motion->Moving({.shader = vs.pgm_hash,
                                             .geometry = motion_geometry,
                                             .indices = index_address,
                                             .index_count = regs.num_indices,
                                             .instances = regs.num_instances.NumInstances(),
                                             .first_instance = first_instance},
                                            palette)) {
            Motion::VertexRange range{base_vertex, regs.num_indices};
            u64 topology = 0;
            if (is_indexed) {
                range = {};
                const u64 bytes = u64(regs.num_indices) * index_size;
                if (index_address && memory->IsValidGpuMapping(index_address, 0) &&
                    memory->ClampRangeSize(index_address, bytes) == bytes) {
                    const bool restart = regs.enable_primitive_restart != 0;
                    const u32 count = regs.num_indices;
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
            push_data.motion_param = object_motion->PrepareDraw({
                .shader = vs.pgm_hash,
                .geometry = motion_geometry,
                .indices = index_address,
                .topology = topology,
                .index_count = regs.num_indices,
                .instances = regs.num_instances.NumInstances(),
                .first_instance = first_instance,
                .vertices = range,
            });
        }
    }
    // BB_TRACE_SHADER with BB_GPU_PROFILE: a mark before the traced vertex shader's draws too.
    if (auto* profiler = GpuProfiler::Get()) {
        static const char* traced = std::getenv("BB_TRACE_SHADER");
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
            const char* env = std::getenv("BB_TRACE_GPU_INPUTS");
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
            const char* env = std::getenv("BB_TRACE_PIPELINE");
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
    Breadcrumbs::Crumb crumb{
        .kind = is_indexed ? Breadcrumbs::Kind::DrawIndexed : Breadcrumbs::Kind::Draw,
        .hash = {vs_info.pgm_hash, ps_info ? ps_info->pgm_hash : 0},
        .program = {vs_info.ProgramBase(), ps_info ? ps_info->ProgramBase() : 0},
        .count = {num_indices, num_instances, 0},
    };
    // On NVIDIA by default (its hangs give no fault address): it costs the recording thread a
    // few T# and V# reads a draw. BB_BREADCRUMB_READS=1 / 0: on / off everywhere.
    static const bool crumb_reads = [&] {
        const char* env = std::getenv("BB_BREADCRUMB_READS");
        return env ? env[0] == '1' : instance.GetVendorID() == 0x10de;
    }();
    if (crumb_reads && Breadcrumbs::Enabled()) {
        NoteCrumbReads(crumb, pipeline, regs, is_indexed, index_offset);
    }
    const Scissors check_scissors =
        DynCheck::Enabled() ? scheduler.GetDynamicState().CurrentScissors() : Scissors{};
    const Viewports check_viewports =
        DynCheck::Enabled() ? scheduler.GetDynamicState().CurrentViewports() : Viewports{};
    const u64 check_vs = vs_info.pgm_hash, check_ps = ps_info ? ps_info->pgm_hash : 0;
    scheduler.RecordCrumb(crumb, [=](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics, handle);
        if (DynCheck::Enabled()) {
            DynCheck::Draw(cmdbuf, check_viewports, check_scissors, check_vs, check_ps);
        }
        if (is_indexed) {
            cmdbuf.drawIndexed(num_indices, num_instances, 0, s32(first_vertex), first_instance);
        } else {
            cmdbuf.draw(num_indices, num_instances, first_vertex, first_instance);
        }
    });
    // BB_TRACE_SHADER with BB_GPU_PROFILE: a mark right after the traced vertex shader's draws,
    // so the profile splits their pass into the draw and what follows it unlabeled.
    if (auto* profiler = GpuProfiler::Get()) {
        static const char* traced = std::getenv("BB_TRACE_SHADER");
        if (traced && std::strtoull(traced, nullptr, 16) == vs_info.pgm_hash) {
            profiler->Mark(vs_info.pgm_hash ^ 0xAF7E'0000ull, [&] {
                return fmt::format("after the draw of vs {:016x}", vs_info.pgm_hash);
            });
        }
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
    }

    static const bool watch = [] {
        const char* env = std::getenv("BB_DRAW_WATCH");
        return env && env[0] == '1';
    }();
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
    static const bool dynamic_always = [] {
        const char* env = std::getenv("BB_DYNAMIC_STATE_ALWAYS");
        return env && env[0] == '1';
    }();
    if (dynamic_always) {
        scheduler.GetDynamicState().Invalidate();
    } else if (scheduler.WillBeginRendering(state)) {
        scheduler.GetDynamicState().InvalidateViewports();
    }
    UpdateDynamicState(pipeline, is_indexed);
    // BB_DISPLAY_WATCH=1 (diagnostics): every draw into a display buffer, with its sizes and state.
    static const bool display_watch = std::getenv("BB_DISPLAY_WATCH") != nullptr;
    // BB_DISPLAY_WATCH_TARGET=addr: also every draw into that color target.
    static const VAddr watch_target = [] {
        const char* env = std::getenv("BB_DISPLAY_WATCH_TARGET");
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
    if (occlusion) {
        occlusion->SetPassScale(target_scale[0] * target_scale[1]); // samples at the game's size
    }
    scheduler.BeginRendering(state);
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
    buffer_cache.NewPacket();
    if (DrawPipe::OnStageB()) {
        FrameCapture::Poll();
        scheduler.PopPendingOperations();
    }
    if (SkippedPatchDraw(pipeline)) {
        return;
    }
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
    const Breadcrumbs::Crumb crumb{
        .kind = Breadcrumbs::Kind::DrawIndirect,
        .hash = {vs_crumb ? vs_crumb->pgm_hash : 0, ps_crumb ? ps_crumb->pgm_hash : 0},
        .program = {vs_crumb ? vs_crumb->ProgramBase() : 0,
                    ps_crumb ? ps_crumb->ProgramBase() : 0},
        .address = indirect.args,
        .count = {max_count, stride, 0},
    };
    scheduler.RecordCrumb(crumb, [=](vk::CommandBuffer cmdbuf) {
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

void Rasterizer::DispatchDirect() {
    BbStats::dispatches.fetch_add(1, std::memory_order_relaxed);
    // bbport: like draws, handed to the draw recording thread after the pipeline selection.
    const bool pipelined = UseDrawPipe() && !BbToggle::Disabled(BbToggle::PipelinedDispatch);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    // BB_DISPATCH_START_LOG=1 (diagnostics, the flicker): dispatches with a nonzero thread
    // group start (COMPUTE_START_X/Y/Z), which the translation does not apply.
    if (static const bool start_log = std::getenv("BB_DISPATCH_START_LOG") != nullptr; start_log) {
        const auto& cs = CsRegs();
        static std::atomic<u32> printed{0};
        if ((cs.start_x | cs.start_y | cs.start_z) != 0 && printed.fetch_add(1) < 5000) {
            std::printf("Dispatch start: gpu frame %llu cs %016llx start %u,%u,%u dims %u,%u,%u\n",
                        (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                        (unsigned long long)pipeline->GetStage(Shader::SwStage::Compute).pgm_hash,
                        cs.start_x, cs.start_y, cs.start_z, cs.dim_x, cs.dim_y, cs.dim_z);
        }
    }
    if (pipelined) {
        PostDraw(pipeline, nullptr, false, 0, &CsRegs());
        return;
    }
    DispatchRecord(pipeline);
}

void Rasterizer::DispatchRecord(const ComputePipeline* pipeline) {
    BB_SECTION(DispatchRecord);
    buffer_cache.NewPacket();
    FrameCapture::Poll();
    gbuffer_draw = false;

    scheduler.PopPendingOperations();

    const auto& cs_program = CsRegs();

    const auto& cs = pipeline->GetStage(Shader::SwStage::Compute);
    // After the resource binding: its transfers (image uploads, downloads) mark themselves.
    const auto mark = [&] {
        if (auto* profiler = GpuProfiler::Get()) {
            profiler->Mark(cs.pgm_hash ^ 0xD15Aull, [&] {
                return fmt::format("dispatch cs {:016x} ({}x{}x{} groups)", cs.pgm_hash,
                                   cs_program.dim_x, cs_program.dim_y, cs_program.dim_z);
            });
        }
    };
    if (upscaler->Enabled()) {
        upscaler->OnDispatch(cs.pgm_hash);
    }
    bool hle;
    {
        BB_SECTION(ShaderHle);
        hle = ExecuteShaderHLE(cs, Regs(), cs_program, *this);
    }
    if (hle) {
        return;
    }
    // bbport: the game's copy shader running as itself (BB_COPY_SHADER_NATIVE): its destinations stay in
    // the game's memory. They hold data our translator reads on the CPU (next to shader code the
    // game uploads this way); a VRAM copy of them showed it a stale copy (a black scene). The
    // copy list read on the CPU (vk_shader_hle.cpp) kept them in place too.
    buffer_cache.force_writes_in_place =
        Game::BufferCopyShader() != 0 && cs.pgm_hash == Game::BufferCopyShader();
    // What it writes, from its control list: GPU-written marks on those bytes only (the bound
    // destination spans tens of MiB, marked whole it kept the constants there in place).
    thread_local std::vector<std::pair<unsigned long long, unsigned long long>> copy_writes;
    copy_writes.clear();
    if (buffer_cache.force_writes_in_place && CopyDestinationsPrecise() &&
        CopyShaderDestinations(cs, cs_program.dim_x, CopyDestinationGap(), copy_writes)) {
        buffer_cache.precise_writes = &copy_writes;
    }
    const bool bound = BindResources(pipeline);
    buffer_cache.force_writes_in_place = false;
    buffer_cache.precise_writes = nullptr;
    if (!bound) {
        return;
    }

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    mark();
    // BB_DISPATCH_LOG=1 (diagnostics): the shape of each kind of real compute dispatch, once: the
    // buffers (size, formatted, written), the groups, and what the image clear detector thinks.
    {
        static const bool log = [] {
            const char* env = std::getenv("BB_DISPATCH_LOG");
            return env && env[0] == '1';
        }();
        if (log) {
            static std::mutex mutex;
            static std::set<std::string> told;
            std::string key = fmt::format("{:016x} x{} y{} z{}", cs.pgm_hash, cs_program.dim_x,
                                          cs_program.dim_y, cs_program.dim_z);
            for (const auto& b : cs.buffers) {
                key += fmt::format(" [{}{}{} {}]", b.is_formatted ? "F" : "R", b.is_written ? "W" : "r",
                                   b.is_formatted ? "" : "", b.GetSharp(cs).GetSize());
            }
            if (cs.buffers.size() >= 2 && cs.buffers[1].is_formatted && cs.buffers[1].is_written) {
                const auto target = cs.buffers[1].GetSharp(cs);
                const auto id = texture_cache.FindImageFromRange(target.base_address, target.GetSize(), false);
                if (id) {
                    const auto& image = texture_cache.GetImage(id);
                    key += fmt::format(" image(size {} bits {} depth {} fmt {})", image.info.guest_size,
                                       image.info.num_bits, image.info.props.is_depth ? 1 : 0,
                                       Gcn::NumBitsPerBlock(target.GetDataFmt()));
                } else {
                    key += fmt::format(" NO-IMAGE at {:#x}", target.base_address);
                }
            }
            std::scoped_lock lock{mutex};
            if (told.size() < 400 && told.insert(key).second) {
                std::printf("Dispatch shape: %s\n", key.c_str());
            }
        }
    }
    pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                            {buffer_infos.data(), buffer_infos.size()});

    const vk::Pipeline handle = pipeline->Handle();
    const u32 dim_x = cs_program.dim_x, dim_y = cs_program.dim_y, dim_z = cs_program.dim_z;
    if (FrameCapture::Active()) {
        FrameCapture::Dispatch(cs.pgm_hash, dim_x, dim_y, dim_z);
    }
    const Breadcrumbs::Crumb crumb{
        .kind = Breadcrumbs::Kind::Dispatch,
        .hash = {cs.pgm_hash, 0},
        .program = {cs.ProgramBase(), 0},
        .count = {dim_x, dim_y, dim_z},
    };
    scheduler.RecordCrumb(crumb, [=](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatch(dim_x, dim_y, dim_z);
    });

    ResetBindings(true);
    scheduler.KickRecording();
}

void Rasterizer::DispatchIndirect(VAddr address, u32 offset, u32 size) {
    // bbport: handed to the recording thread like direct dispatches.
    const bool pipelined = UseDrawPipe() && !BbToggle::Disabled(BbToggle::PipelinedIndirectDraws);
    if (!pipelined) {
        DrainDrawPipe();
    }
    const ComputePipeline* pipeline = pipeline_cache.GetComputePipeline();
    if (!pipeline) {
        return;
    }
    if (pipelined) {
        const IndirectDraw indirect = {address + offset, 0, size, 1};
        PostDraw(pipeline, nullptr, false, 0, &CsRegs(), &indirect);
        return;
    }
    DispatchIndirectRecord(pipeline, address + offset, size);
}

void Rasterizer::DispatchIndirectRecord(const ComputePipeline* pipeline, VAddr address,
                                        u32 size) {
    buffer_cache.NewPacket();
    scheduler.PopPendingOperations();
    if (!BindResources(pipeline)) {
        return;
    }
    const auto [buffer, base] = buffer_cache.ObtainBuffer(address, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, base, size);

    if (needs_barrier) {
        runtime.FlushBarriers();
    }

    scheduler.EndRendering();
    // Group counts checked on the GPU first (vk_indirect_guard.h); recorded before the
    // pipeline's bindings, which the check's own pipeline would disturb.
    vk::Buffer args = buffer->Handle();
    u64 args_offset = base;
    if (IndirectGuard::Enabled()) {
        if (!indirect_guard) {
            indirect_guard = std::make_unique<IndirectGuard>(instance, scheduler);
        }
        std::tie(args, args_offset) = indirect_guard->CheckArgs(*buffer, base);
    }
    pipeline->BindResources(set_writes, push_data, {image_infos.data(), image_infos.size()},
                            {buffer_infos.data(), buffer_infos.size()});

    const vk::Pipeline handle = pipeline->Handle();
    const vk::Buffer raw_args = buffer->Handle();
    const u64 raw_offset = base;
    const auto& cs_crumb = pipeline->GetStage(Shader::SwStage::Compute);
    const u32 args_slot = Breadcrumbs::ArgsSlot();
    const Breadcrumbs::Crumb crumb{
        .kind = Breadcrumbs::Kind::DispatchIndirect,
        .hash = {cs_crumb.pgm_hash, 0},
        .program = {cs_crumb.ProgramBase(), 0},
        .address = address,
        .count = {buffer_cache.IsInPlace(address, size) ? 1u : 0u, 0, 0},
        .args_slot = args_slot,
    };
    scheduler.RecordCrumb(crumb, [=](vk::CommandBuffer cmdbuf) {
        Breadcrumbs::CopyArgs(cmdbuf, raw_args, raw_offset, args_slot); // what the game wrote
        cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, handle);
        cmdbuf.dispatchIndirect(args, args_offset);
    });

    ResetBindings(true);
    scheduler.KickRecording();
}

u64 Rasterizer::Flush() {
    BB_SECTION(Flush);
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

void Rasterizer::OnSubmit() {
    DrainDrawPipe();
    if (fault_process_pending) {
        fault_process_pending = false;
        buffer_cache.ProcessFaultBuffer();
    }
    texture_cache.ProcessDownloadImages();
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
    static const bool stats = [] {
        const char* value = std::getenv("BB_BUFFER_STATS");
        return value && value[0] == '1';
    }();
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

namespace Vulkan {

void Rasterizer::WatchDraw(const GraphicsPipeline* pipeline, const RenderState& state) {
    // GPU thread only. Per (vs, ps, first target): the last fields seen and how many draws in a
    // row had them. A change after a stable run is printed with the frame number, and so is the
    // change back, which marks a one-frame anomaly (docs/Flicker_video_2026-10-17.ru.md).
    const auto& regs = Regs();
    const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
    const auto* ps = pipeline->GetStages()[u32(Shader::SwStage::Fragment)];
    const VAddr target = cb_descs[0].first ? texture_cache.GetImage(cb_descs[0].first).info.guest_address : 0;
    const u64 key = vs.pgm_hash * 0x9E3779B97F4A7C15ull ^ (ps ? ps->pgm_hash : 0) ^ target;
    const auto bits = [](float f) { return u64(std::bit_cast<u32>(f)); };
    std::vector<u64> fields;
    const auto& vp = regs.viewports[0];
    fields.push_back(bits(vp.xscale) << 32 | bits(vp.xoffset));
    fields.push_back(bits(vp.yscale) << 32 | bits(vp.yoffset));
    fields.push_back(u64(std::bit_cast<u32>(regs.viewport_control)) << 32 | u64(regs.IsClipDisabled()));
    const auto& sc = regs.screen_scissor;
    fields.push_back(u64(u16(sc.top_left_x)) << 48 | u64(u16(sc.top_left_y)) << 32 |
                     u64(u16(sc.bottom_right_x)) << 16 | u16(sc.bottom_right_y));
    fields.push_back(bits(target_scale[0]) << 32 | bits(target_scale[1]));
    fields.push_back(bits(push_data.xoffset) << 32 | bits(push_data.yoffset));
    fields.push_back(u64(state.width) << 32 | state.height);
    static const bool textures = std::getenv("BB_DRAW_WATCH_TEXTURES") != nullptr;
    for (const auto id : textures ? bound_images : decltype(bound_images){}) {
        const auto& info = texture_cache.GetImage(id).info;
        fields.push_back(info.guest_address);
        fields.push_back(u64(info.size.width) << 32 | info.size.height);
    }
    struct Seen {
        std::vector<u64> fields;
        std::vector<u64> stable_fields;
        u32 run = 0;
        u64 changed_frame = 0;
    };
    static std::unordered_map<u64, Seen> seen;
    static u32 printed = 0;
    auto& s = seen[key];
    const u64 frame = FrameCapture::Flips();
    if (s.fields == fields) {
        ++s.run;
        return;
    }
    const auto print = [&](const char* what) {
        if (printed++ > 4000) {
            return;
        }
        std::string text = fmt::format("Draw watch: frame {} {} vs {:016x} ps {:016x} target {:#x}:",
                                       frame, what, vs.pgm_hash, ps ? ps->pgm_hash : 0, target);
        const size_t n = std::max(fields.size(), s.fields.size());
        for (size_t i = 0; i < n; ++i) {
            const u64 now = i < fields.size() ? fields[i] : 0;
            const u64 was = i < s.fields.size() ? s.fields[i] : 0;
            if (now != was) {
                text += fmt::format(" [{}] {:#x}->{:#x}", i, was, now);
            }
        }
        std::printf("%s\n", text.c_str());
    };
    if (s.run >= 8) {
        // A stable run broke.
        s.stable_fields = s.fields;
        s.changed_frame = frame;
        print("changed");
    } else if (!s.stable_fields.empty() && fields == s.stable_fields && frame - s.changed_frame <= 2) {
        print("back after a short change");
        s.stable_fields.clear();
    }
    s.fields = std::move(fields);
    s.run = 0;
}

} // namespace Vulkan
