// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, resource binding: push data, buffers, vertex and index buffers, and the texture
// binding helper thread.

#include "bbport_button_prompts.h"
#include "render/raster_internal.h"

namespace Vulkan {

namespace {
/// The push data's viewport transform: the first viewport's (as the vertex shader applies it
/// with clipping off; one viewport, see ViewportIndex).
Shader::PushData MakeUserData(const Gcn::Regs& regs) {
    const auto& ctl = regs.viewport_control;
    const auto& vp = regs.viewports[0];
    Shader::PushData push_data{};
    push_data.xoffset = ctl.xoffset_enable ? vp.xoffset : 0.f;
    push_data.xscale = ctl.xscale_enable ? vp.xscale : 1.f;
    push_data.yoffset = ctl.yoffset_enable ? vp.yoffset : 0.f;
    push_data.yscale = ctl.yscale_enable ? vp.yscale : 1.f;
    return push_data;
}
} // namespace

namespace {
void TraceShaderBinding(const Shader::Info& stage, VAddr addr, u64 size, bool written,
                        const VideoCore::BufferCache& cache, const VideoCore::Buffer* bound);
} // namespace

const PreparedStage* Rasterizer::FindPreparedStage(const Shader::Info& stage) const {
    // Sharps a draw-preparation worker read from the same flattened user data.
    if (!bind_prepared || BbToggle::Disabled(BbToggle::PreparedResources)) {
        return nullptr;
    }
    for (u32 i = 0; i < bind_prepared->num_stages; ++i) {
        const auto& candidate = bind_prepared->stages[i];
        if (&candidate.program->info == &stage &&
            candidate.num_images == stage.images.size() &&
            candidate.num_samplers == stage.samplers.size() &&
            candidate.num_buffers == stage.buffers.size()) {
            return &candidate;
        }
    }
    return nullptr;
}

bool Rasterizer::BindHelperWanted() {
    // A spinning helper pays off only with cores to spare (the recording thread spins too).
    const char* env = std::getenv("BB_TEXTURE_HELPER");
    if (env && env[0]) {
        return env[0] == '1';
    }
    // Measured neutral on 16 threads (docs/parallel_gpu.md): opt-in.
    return false;
}

bool Rasterizer::HelperEligible(const Pipeline* pipeline) const {
    if (pipeline->IsCompute() || !bind_helper.Available() || FrameCapture::Active() ||
        BbToggle::Disabled(BbToggle::TextureBindHelper) ||
        BbToggle::Disabled(BbToggle::TextureBindingMemo) ||
        BbToggle::Disabled(BbToggle::UpdateImageFastPath)) {
        return false;
    }
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        // Mip arrays make the descriptor layout depend on the T#s.
        for (const auto& image : stage->images) {
            if (image.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
                return false;
            }
        }
        // Storage writes invalidate cached images from the buffer side.
        for (const auto& buffer : stage->buffers) {
            if (!buffer.IsSpecial() && buffer.is_written) {
                return false;
            }
        }
    }
    return true;
}

bool Rasterizer::TexturesBindableOnHelper(const Shader::Info& stage,
                                          const PreparedStage* prepared) {
    const u64 generation = texture_cache.RegistryGeneration();
    for (u32 image_index = 0; image_index < stage.images.size(); ++image_index) {
        const auto& image_desc = stage.images[image_index];
        if (image_desc.is_written) {
            return false;
        }
        const auto tsharp =
            prepared ? prepared->image_sharps[image_index] : image_desc.GetSharp(stage);
        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == Gcn::DataFormat::FormatInvalid ||
            !memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !IsKnownFormat(data_fmt, num_fmt)) {
            continue; // null descriptor
        }
        const auto& entry =
            prepared ? CachedImageDescEntry(tsharp, image_desc, prepared->image_hashes[image_index])
                     : CachedImageDescEntry(tsharp, image_desc);
        if (entry.found_generation != generation) {
            return false;
        }
        VideoCore::ImageId image_id = entry.found_id;
        auto* image = &texture_cache.GetImage(image_id);
        if (const auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
            image_id = depth_image_id;
            image = &texture_cache.GetImage(image_id);
        }
        // A key guide icon the GPU copied in (never uploaded, so never recognised there).
        if (image->MaybePromptIcon() && prompts->Active()) {
            image->prompt_icon =
                u8(BbPrompts::IconOf(reinterpret_cast<const void*>(image->info.guest_address), 1024));
        }
        // A compressible image this binding or a feedback loop needs in full usage: upgraded
        // on the GPU thread (BindTextures).
        // (FindView may first switch to another sample count's backing, which may need it.)
        if (image->compressible && image->backing &&
            ((image->backing->num_samples > 1) != (image->info.num_samples > 1) ||
             (image->binding.is_target && image->LacksFeedbackUsage()) ||
             (image->backing->compressible &&
              (image->binding.is_target ||
               image->NeedsFullUsage(entry.found_desc.view_info))))) {
            return false;
        }
        if (image->binding.needs_rebind || !texture_cache.IsUpToDate(image_id) ||
            scene_targets->Tracks(image->image_uid) || ((image->prompt_icon || image->prompt_atlas) && prompts->Active()) ||
            (upscaler->Enabled() && upscaler->RedirectsSampled(image_id))) {
            return false;
        }
    }
    return true;
}

void Rasterizer::RunTextureTask(void* context) {
    auto& self = *static_cast<Rasterizer*>(context);
    auto& task = self.texture_task;
    for (u32 i = 0; i < task.count; ++i) {
        auto& stage = task.stages[i];
        if (!self.TexturesBindableOnHelper(*stage.info, stage.prepared)) {
            return;
        }
        auto binding = stage.binding;
        u32 write_index = stage.write_index;
        self.BindTextures(*stage.info, stage.prepared, binding, write_index, task.barrier, true);
        task.completed = i + 1;
    }
}

void Rasterizer::JoinBindHelper(void* context) {
    if (!BindHelper::OnHelper()) {
        static_cast<Rasterizer*>(context)->bind_helper.Join();
    }
}

namespace {
// bbport: BB_RES_STATS=1 (diagnostics) — what the bound shaders read their resource descriptors
// from: user data registers (state the API call set) or tables in memory (SRT walker), by kind,
// per draw and dispatch; printed every 5 s. Measures the step "descriptors read by the GPU"
// (docs/EMULATION_REMOVAL_PLAN.ru.md).
struct ResStats {
    u64 calls = 0, stages = 0, walker_stages = 0, readconst_stages = 0, dma_stages = 0;
    u64 flat_dwords = 0;
    // [kind][source]: kind 0 guest buffers, 1 images, 2 samplers; source 0 registers or
    // immediates, 1 memory (SRT), 2 mixed.
    u64 sharps[3][3]{};
    u64 buffers_written = 0, buffers_formatted = 0, images_written = 0, special_buffers = 0;
    std::unordered_set<u64> shaders, walker_shaders;
};
template <typename Fetch>
int SharpSource(const Fetch& fetch) {
    bool reg = false, mem = false;
    u8 mask = fetch.load_mask;
    for (u32 i = 0; i < fetch.offsets.size(); ++i, mask >>= 1) {
        if (!(mask & 1)) {
            continue;
        }
        (fetch.offsets[i] < Shader::NUM_USER_DATA_REGS ? reg : mem) = true;
    }
    return mem ? (reg ? 2 : 1) : 0;
}
void NoteResStats(const Pipeline* pipeline) {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_RES_STATS");
        return env && env[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    static std::mutex mutex;
    static ResStats s;
    static auto window = std::chrono::steady_clock::now();
    std::scoped_lock lk{mutex};
    ++s.calls;
    for (const auto* stage : pipeline->GetStages()) {
        if (!stage) {
            continue;
        }
        ++s.stages;
        s.shaders.insert(stage->pgm_hash);
        if (stage->srt_info.walker_func) {
            ++s.walker_stages;
            s.walker_shaders.insert(stage->pgm_hash);
        }
        s.readconst_stages += stage->has_readconst ? 1 : 0;
        s.dma_stages += stage->uses_dma ? 1 : 0;
        s.flat_dwords += stage->srt_info.flattened_bufsize_dw;
        for (const auto& b : stage->buffers) {
            if (b.IsSpecial()) {
                ++s.special_buffers;
                continue;
            }
            ++s.sharps[0][SharpSource(b.sharp_fetch)];
            s.buffers_written += b.is_written;
            s.buffers_formatted += b.is_formatted;
        }
        for (const auto& i : stage->images) {
            ++s.sharps[1][SharpSource(i.sharp_fetch)];
            s.images_written += i.is_written;
        }
        for (const auto& smp : stage->samplers) {
            ++s.sharps[2][SharpSource(smp.sharp_fetch)];
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - window).count();
    if (seconds < 5.0) {
        return;
    }
    window = now;
    const double c = std::max<double>(1, s.calls);
    std::printf("Resource stats: %.0f binds/s, %.2f stages/bind, %.2f with SRT walker (%zu of %zu "
                "shaders), readconst %.2f, dma %.2f, flat %.1f dw/bind; per bind: buffers "
                "reg %.2f mem %.2f mixed %.2f (written %.2f, formatted %.2f, special %.2f), "
                "images reg %.2f mem %.2f mixed %.2f (written %.2f), samplers reg %.2f mem %.2f "
                "mixed %.2f\n",
                s.calls / seconds, s.stages / c, s.walker_stages / c, s.walker_shaders.size(),
                s.shaders.size(), s.readconst_stages / c, s.dma_stages / c, s.flat_dwords / c,
                s.sharps[0][0] / c, s.sharps[0][1] / c, s.sharps[0][2] / c, s.buffers_written / c,
                s.buffers_formatted / c, s.special_buffers / c, s.sharps[1][0] / c,
                s.sharps[1][1] / c, s.sharps[1][2] / c, s.images_written / c, s.sharps[2][0] / c,
                s.sharps[2][1] / c, s.sharps[2][2] / c);
    s = {};
}
} // namespace

bool Rasterizer::BindResources(const Pipeline* pipeline) {
    BB_SECTION(BindResources);
    NoteResStats(pipeline);
    // BB_EXP_BARRIER_ALL=1 (experiment): a full memory barrier before every draw and dispatch.
    static const bool barrier_all = std::getenv("BB_EXP_BARRIER_ALL") != nullptr;
    if (barrier_all) {
        runtime.AccessGlobal(vk::PipelineStageFlagBits2::eAllCommands, vk::AccessFlagBits2::eShaderWrite);
        needs_barrier = true;
    }
    {
        // bbport BB_HLE_STATS=1 (diagnostics): how often the compute shader substitutions fire.
        static const bool hle_stats = std::getenv("BB_HLE_STATS") != nullptr;
        // BB_HLE_OFF=mask (experiment): 1 image copy, 2 metadata clear, 4 image clear run as the
        // game's shaders.
        static const u32 hle_off = [] {
            const char* env = std::getenv("BB_HLE_OFF");
            return env ? u32(std::strtoul(env, nullptr, 0)) : 0u;
        }();
        static const bool copy_log = std::getenv("BB_IMAGE_COPY_LOG") != nullptr;
        if ((hle_off & 1) && copy_log) {
            IsComputeImageCopy(pipeline, true);
        }
        const int which = !(hle_off & 1) && IsComputeImageCopy(pipeline)   ? 1
                          : !(hle_off & 2) && IsComputeMetaClear(pipeline) ? 2
                          : !(hle_off & 4) && IsComputeImageClear(pipeline) ? 3
                                                                             : 0;
        if (hle_stats) {
            static std::array<u64, 4> counts{};
            static auto last = std::chrono::steady_clock::now();
            ++counts[which];
            if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) {
                last = std::chrono::steady_clock::now();
                std::printf("HLE stats (5 s): image copy %llu, meta clear %llu, image clear %llu, "
                            "other binds %llu\n",
                            (unsigned long long)counts[1], (unsigned long long)counts[2],
                            (unsigned long long)counts[3], (unsigned long long)counts[0]);
                counts = {};
            }
        }
        if (which) {
            return false;
        }
    }

    set_write_index = 0;
    set_writes.clear();
    buffer_infos.clear();
    image_infos.clear();
    // bbport: G-buffer passes sample material textures for a scene rendered below the output
    // size; the temporal upscaler restores the detail of the output's mip level (FSR guide:
    // log2(render / output)). Shadows, post-processing and UI keep the guest's bias.
    sampler_lod_bias = 0.0f;
    pipeline_is_compute = pipeline->IsCompute();
    if (!pipeline->IsCompute() && !BbToggle::Disabled(BbToggle::SceneMipBias) &&
        std::popcount(static_cast<const GraphicsPipeline*>(pipeline)->GetGraphicsKey().mrt_mask &
                      0xff) >= 5) {
        sampler_lod_bias = upscaler->SceneMipBias();
        static float reported = 0.0f;
        if (sampler_lod_bias != reported) {
            reported = sampler_lod_bias;
            std::printf("Upscaler: scene texture LOD bias %.2f\n", sampler_lod_bias);
        }
    }

    bool uses_dma = false;

    const bool stats = BbStats::enabled; // bbport (macOS): was on for any value, even 0
    if (stats && ((helper_full + helper_partial + helper_serial) & 1023) == 0) {
        static auto window = std::chrono::steady_clock::now();
        const auto now = std::chrono::steady_clock::now();
        if (now - window >= std::chrono::seconds(5)) {
            window = now;
            const double forks = std::max<double>(1, helper_full + helper_partial);
            std::printf("Texture helper: %llu binds in parallel, %llu partly, %llu serial; "
                        "per fork %.0f cycles helper task, %.0f cycles GPU thread wait\n",
                        static_cast<unsigned long long>(helper_full),
                        static_cast<unsigned long long>(helper_partial),
                        static_cast<unsigned long long>(helper_serial),
                        bind_helper.task_cycles.exchange(0) / forks,
                        bind_helper.wait_cycles / forks);
            bind_helper.wait_cycles = 0;
            helper_full = helper_partial = helper_serial = 0;
        }
    }

    // Bind resource buffers and textures.
    Shader::Backend::Bindings binding{};
    push_data = MakeUserData(Regs());
    if (!HelperEligible(pipeline)) {
        ++helper_serial;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            set_writes.resize(set_writes.size() + stage->buffers.size() + stage->images.size() +
                              stage->samplers.size());
            stage->PushUd(binding, push_data);
            const PreparedStage* prepared = FindPreparedStage(*stage);
            BindBuffers(*stage, prepared, binding, push_data, set_write_index);
            BindTextures(*stage, prepared, binding, set_write_index, needs_barrier);
            uses_dma |= stage->uses_dma;
        }
    } else {
        // bbport: textures on the helper, buffers here. Without mip arrays every stage's
        // descriptors are [buffers][images][samplers], one each, so both sides know their
        // binding numbers and descriptor write slots up front.
        struct BufferStage {
            const Shader::Info* info;
            const PreparedStage* prepared;
            Shader::Backend::Bindings binding;
            u32 write_index;
        };
        std::array<BufferStage, Shader::MaxStageTypes> buffer_stages;
        auto& task = texture_task;
        task.count = 0;
        task.completed = 0;
        task.barrier = false;
        for (const auto* stage : pipeline->GetStages()) {
            if (!stage) {
                continue;
            }
            const u32 num_buffers = static_cast<u32>(stage->buffers.size());
            const u32 num_descriptors =
                num_buffers + static_cast<u32>(stage->images.size() + stage->samplers.size());
            set_writes.resize(set_writes.size() + num_descriptors);
            stage->PushUd(binding, push_data);
            const PreparedStage* prepared = FindPreparedStage(*stage);
            buffer_stages[task.count] = {stage, prepared, binding, set_write_index};
            auto texture_binding = binding;
            texture_binding.buffer += num_buffers;
            texture_binding.unified += num_buffers;
            task.stages[task.count] = {stage, prepared, texture_binding,
                                       set_write_index + num_buffers};
            ++task.count;
            binding.buffer += num_buffers;
            binding.unified += num_descriptors;
            set_write_index += num_descriptors;
            uses_dma |= stage->uses_dma;
        }
        bind_helper.Fork(&RunTextureTask, this);
        for (u32 i = 0; i < task.count; ++i) {
            auto& stage = buffer_stages[i];
            BindBuffers(*stage.info, stage.prepared, stage.binding, push_data, stage.write_index);
        }
        if (draw_inputs.pending && !BbToggle::Disabled(BbToggle::EarlyDrawInputs)) {
            ResolveVertexBuffers(draw_inputs.pipeline, draw_inputs.prepared);
            if (draw_inputs.is_indexed) {
                ResolveIndexBuffer(draw_inputs.index_offset);
            }
            draw_inputs.resolved = true;
        }
        bind_helper.Join();
        needs_barrier |= task.barrier;
        ++(task.completed == task.count ? helper_full
                                        : task.completed ? helper_partial : helper_serial);
        // Stages the helper left (an image needing work that records commands).
        for (u32 i = task.completed; i < task.count; ++i) {
            auto& stage = task.stages[i];
            u32 write_index = stage.write_index;
            BindTextures(*stage.info, stage.prepared, stage.binding, write_index, needs_barrier);
        }
    }

    if (uses_dma) {
        buffer_cache.SynchronizeDmaBuffers();
        fault_process_pending = true;
    }

    return true;
}

void Rasterizer::BindVertexBuffers(const GraphicsPipeline* pipeline, const PreparedDraw* prepared) {
    ResolveVertexBuffers(pipeline, prepared);
    EmitVertexBuffers();
}

void Rasterizer::EmitVertexBuffers() {
    BB_SECTION(EmitVertexBuffers);
    // bbport: only the used entries go into the recording chunk (the static vectors hold 32 of
    // each: ~3 KiB copied per draw).
    auto& v = vertex_binds;
    const bool dynamic_input = instance.IsVertexInputDynamicState();
    const u32 num_buffers = v.num_buffers;
    scheduler.ReserveRecordData(
        v.bindings.size() * sizeof(v.bindings[0]) + v.attributes.size() * sizeof(v.attributes[0]) +
        num_buffers * (sizeof(vk::Buffer) + 3 * sizeof(vk::DeviceSize)) + 128);
    if (dynamic_input) {
        scheduler.Record([bindings = scheduler.RecordData(std::span<const vk::VertexInputBindingDescription2EXT>{v.bindings.data(), v.bindings.size()}),
                          attributes = scheduler.RecordData(std::span<const vk::VertexInputAttributeDescription2EXT>{
                              v.attributes.data(), v.attributes.size()})](
                             vk::CommandBuffer cmdbuf) {
            cmdbuf.setVertexInputEXT(bindings, attributes);
        });
    }
    if (num_buffers == 0) {
        return;
    }
    const auto buffers = scheduler.RecordData(std::span<const vk::Buffer>{v.host_buffers.data(), num_buffers});
    const auto offsets = scheduler.RecordData(std::span<const vk::DeviceSize>{v.host_offsets.data(), num_buffers});
    if (dynamic_input) {
        scheduler.Record([num_buffers, buffers, offsets](vk::CommandBuffer cmdbuf) {
            cmdbuf.bindVertexBuffers(0, num_buffers, buffers.data(), offsets.data());
        });
        return;
    }
    const auto sizes = scheduler.RecordData(std::span<const vk::DeviceSize>{v.host_sizes.data(), num_buffers});
    const auto strides = scheduler.RecordData(std::span<const vk::DeviceSize>{v.host_strides.data(), num_buffers});
    scheduler.Record([num_buffers, buffers, offsets, sizes, strides](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindVertexBuffers2(0, num_buffers, buffers.data(), offsets.data(), sizes.data(),
                                  strides.data());
    });
}

void Rasterizer::ResolveVertexBuffers(const GraphicsPipeline* pipeline,
                                      const PreparedDraw* prepared) {
    BB_SECTION(ResolveVertexBuffers);
    const auto& regs = Regs();
    auto& v = vertex_binds;
    v.num_buffers = 0;
    v.host_buffers.clear();
    v.host_offsets.clear();
    v.host_sizes.clear();
    v.host_strides.clear();
    auto& attributes = v.attributes;
    auto& bindings = v.bindings;
    attributes.clear();
    bindings.clear();
    VertexInputs<vk::VertexInputBindingDivisorDescriptionEXT> divisors;
    VertexInputs<Gcn::Buffer> guest_buffers;
    // Inputs a draw-preparation worker read from the same registers and (verified) user data.
    const auto& fetch = pipeline->GetFetchShader();
    const PreparedVertexInputs* ready =
        prepared && prepared->vertex.valid && instance.IsVertexInputDynamicState() && fetch &&
                fetch->attributes.size() == prepared->vertex.count && !FrameCapture::Active() &&
                !BbToggle::Disabled(BbToggle::PreparedResources)
            ? &prepared->vertex
            : nullptr;
    if (ready) {
        attributes.assign(ready->attributes, ready->attributes + ready->count);
        bindings.assign(ready->bindings, ready->bindings + ready->count);
        guest_buffers.assign(ready->buffers, ready->buffers + ready->count);
        if (BbCeStats::Enabled() && packet_vsharps.size() == ready->count) {
            static u64 compared = 0, differed = 0;
            ++compared;
            differed += std::memcmp(ready->buffers, packet_vsharps.data(),
                                    ready->count * sizeof(Gcn::Buffer)) != 0;
            if ((compared & 0xFFFF) == 0) {
                std::printf("CE stats: prepared V#s differed from the packet's in %llu of %llu "
                            "draws\n",
                            (unsigned long long)differed, (unsigned long long)compared);
            }
        }
    } else {
        pipeline->GetVertexInputs(attributes, bindings, divisors, guest_buffers,
                                  regs.vgt_instance_step_rate_0, regs.vgt_instance_step_rate_1,
                                  packet_vsharps);
        if (BbCeStats::Enabled() && !packet_vsharps.empty()) {
            // What reading the V# tables here would have given instead (the old path).
            static u64 compared = 0, differed = 0;
            const auto& fetch = pipeline->GetFetchShader();
            const auto& vs = pipeline->GetStage(Shader::SwStage::Vertex);
            for (u32 i = 0; fetch && i < fetch->attributes.size() && i < packet_vsharps.size(); ++i) {
                ++compared;
                const auto now = fetch->attributes[i].GetSharp(vs);
                differed += std::memcmp(&now, &packet_vsharps[i], sizeof(now)) != 0;
            }
            if ((compared & 0xFFFF) == 0) {
                std::printf("CE stats: V#s read on the recording thread differed from the "
                            "packet's in %llu of %llu\n",
                            (unsigned long long)differed, (unsigned long long)compared);
            }
        }
    }

    if (bindings.empty()) {
        // If there are no bindings, there is nothing further to do.
        return;
    }
    if (motion_draw && !guest_buffers.empty()) {
        // Include every stream: identical position buffers can be paired with different
        // skinning/instance data. Index topology and base offsets are matched separately.
        motion_geometry = ready ? ready->buffers_hash
                                : XXH3_64bits(guest_buffers.data(),
                                              guest_buffers.size() * sizeof(Gcn::Buffer));
    }
    // Object motion research: vertex streams of G-buffer draws.
    if (FrameCapture::Active() && gbuffer_draw) {
        for (const auto& vb : guest_buffers) {
            char note[128];
            std::snprintf(note, sizeof(note), "\n  gb vb at %#llx size %u stride %u",
                          (unsigned long long)vb.base_address, u32(vb.GetSize()),
                          u32(vb.GetStride()));
            FrameCapture::Note(note);
        }
    }

    // The streams' memory in as few host ranges as possible: overlapping and touching streams
    // share one (a draw-preparation worker merged them already).
    struct Range {
        VAddr begin;
        VAddr end;
        const VideoCore::Buffer* buffer = nullptr;
        u64 offset = 0;
    };
    VertexInputs<Range> ranges;
    if (ready) {
        for (u32 i = 0; i < ready->num_ranges; ++i) {
            ranges.push_back({ready->ranges[i].base, ready->ranges[i].end});
        }
    } else {
        VertexInputs<Range> streams;
        for (const auto& buffer : guest_buffers) {
            if (buffer.base_address != 0 && buffer.GetSize() > 0) {
                streams.push_back({buffer.base_address, buffer.base_address + buffer.GetSize()});
            }
        }
        std::ranges::sort(streams, {}, &Range::begin);
        for (const auto& stream : streams) {
            if (ranges.empty() || ranges.back().end < stream.begin) {
                ranges.push_back(stream);
            } else {
                ranges.back().end = std::max(ranges.back().end, stream.end);
            }
        }
    }
    for (auto& range : ranges) {
        const u64 size = memory->ClampRangeSize(range.begin, range.end - range.begin);
        std::tie(range.buffer, range.offset) = buffer_cache.ObtainBuffer(range.begin, size, false);
        if (InPlaceReads::Enabled()) {
            InPlaceReads::Note(range.begin, size, scheduler.CurrentTick(), 0, 2);
        }
        if (const auto* vs = pipeline ? pipeline->GetStages()[u32(Shader::SwStage::Vertex)] : nullptr) {
            TraceShaderBinding(*vs, range.begin, size, false, buffer_cache, range.buffer);
            // BB_TRACE_SHADER: the draw's counts and streams too, once a second.
            static const char* traced = std::getenv("BB_TRACE_SHADER");
            // BB_TRACE_VERTEX_DATA=1 with BB_TRACE_SHADER: the stream bytes whenever they differ
            // from the traced shader's previous draw (what it reads when this copy is taken).
            static const bool trace_data = std::getenv("BB_TRACE_VERTEX_DATA") != nullptr;
            if (traced && trace_data && std::strtoull(traced, nullptr, 16) == vs->pgm_hash &&
                size <= 256) {
                static std::array<u8, 256> last{};
                static u32 data_prints = 0;
                std::array<u8, 256> now{};
                std::memcpy(now.data(), reinterpret_cast<const void*>(range.begin), size);
                if (now != last && data_prints++ < 3000) {
                    std::string text;
                    const auto* f = reinterpret_cast<const float*>(now.data());
                    for (u64 i = 0; i < size / 4 && i < 24; ++i) {
                        text += fmt::format(" {:g}", f[i]);
                    }
                    std::printf("Vertex data: gpu frame %llu %#llx+%#llx:%s\n",
                                (unsigned long long)BbStats::gpu_frames.load(std::memory_order_relaxed),
                                (unsigned long long)range.begin, (unsigned long long)size,
                                text.c_str());
                    last = now;
                }
            }
            static std::atomic<u32> printed{0};
            const u32 second = BbStats::coarse_second.load(std::memory_order_relaxed);
            if (traced && std::strtoull(traced, nullptr, 16) == vs->pgm_hash &&
                printed.exchange(second) != second) {
                std::printf("Draw vs %016llx: %u indices, %u instances, range %#llx+%#llx",
                            (unsigned long long)vs->pgm_hash, regs.num_indices,
                            regs.num_instances.NumInstances(), (unsigned long long)range.begin,
                            (unsigned long long)size);
                for (const auto& b : guest_buffers) {
                    std::printf("; stream %#llx size %#llx stride %u",
                                (unsigned long long)b.base_address,
                                (unsigned long long)b.GetSize(), b.GetStride());
                }
                std::printf("%s\n", ready ? " (prepared)" : "");
            }
        }
        needs_barrier |= runtime.IsBufferAccessed(range.buffer, range.offset, size);
    }

    // Each stream at its offset in its range's buffer (null for streams without memory).
    u32 stream_with_memory = 0;
    for (const auto& buffer : guest_buffers) {
        if (buffer.base_address != 0 && buffer.GetSize() > 0) {
            const auto range =
                ready ? ranges.begin() + ready->range_index[stream_with_memory++]
                      : std::ranges::find_if(ranges, [&](const Range& r) {
                            return buffer.base_address >= r.begin && buffer.base_address < r.end;
                        });
            ASSERT(range != ranges.end());
            v.host_buffers.push_back(range->buffer->Handle());
            v.host_offsets.push_back(range->offset + buffer.base_address - range->begin);
        } else {
            v.host_buffers.push_back(VK_NULL_HANDLE);
            v.host_offsets.push_back(0);
        }
        v.host_sizes.push_back(buffer.GetSize());
        v.host_strides.push_back(buffer.GetStride());
    }
    v.num_buffers = static_cast<u32>(guest_buffers.size());
}

void Rasterizer::BindIndexBuffer(u32 index_offset) {
    ResolveIndexBuffer(index_offset);
    EmitIndexBuffer();
}

void Rasterizer::EmitIndexBuffer() {
    scheduler.Record([handle = index_bind.handle, offset = index_bind.offset,
                      type = index_bind.type](vk::CommandBuffer cmdbuf) {
        cmdbuf.bindIndexBuffer(handle, offset, type);
    });
}

void Rasterizer::ResolveIndexBuffer(u32 index_offset) {
    BB_SECTION(ResolveIndexBuffer);
    const auto& regs = Regs();
    const bool index16 = regs.index_buffer_type.index_type == Gcn::IndexType::Index16;
    const u32 index_size = index16 ? sizeof(u16) : sizeof(u32);
    const VAddr address = regs.index_base_address.Address<VAddr>() + u64(index_offset) * index_size;
    const u32 size = regs.num_indices * index_size;
    const auto [buffer, offset] = buffer_cache.ObtainBuffer(address, size, false);
    needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size);
    if (InPlaceReads::Enabled()) {
        InPlaceReads::Note(address, size, scheduler.CurrentTick(), 0, 3);
    }
    // Tracked as a graphics read after the draw (ResetBindings): an upload into a mirrored index
    // range must not overtake the draw's fixed-function index fetches.
    bound_buffers.emplace_back(buffer, offset, size, false);
    index_bind = {buffer->Handle(), offset, index16 ? vk::IndexType::eUint16 : vk::IndexType::eUint32};
}

void Rasterizer::ResetBindings(bool is_compute) {
    for (auto& image_id : bound_images) {
        texture_cache.GetImage(image_id).binding = {};
    }
    runtime.ReleaseRetired(); // the draw recorded: images it replaced can go after it

    for (const auto [buffer, offset, size, is_written] : bound_buffers) {
        const auto dst_stage = is_compute ? vk::PipelineStageFlagBits2::eComputeShader
                                          : vk::PipelineStageFlagBits2::eAllGraphics;
        const auto write_flag =
            is_written ? vk::AccessFlagBits2::eShaderWrite : vk::AccessFlagBits2::eNone;
        if (!buffer) {
            // bbport BB_LAYER_MEMORY: a paged binding (any buffer of the game's memory).
            runtime.AccessGlobal(dst_stage, vk::AccessFlagBits2::eShaderRead | write_flag);
            continue;
        }
        runtime.AccessBuffer(buffer, offset, size, dst_stage,
                             vk::AccessFlagBits2::eShaderRead | write_flag);
    }
    bound_images.clear();
    bound_buffers.clear();
    needs_barrier = false;
}

namespace {
// bbport: BB_TRACE_SHADER=hash[,hash...] (diagnostics) — the buffers those shaders bind: address,
// size, written or read, where the data is (in place: the game's memory over PCIe on a discrete
// GPU; VRAM; partly), how often; printed every 5 s.
void TraceShaderBinding(const Shader::Info& stage, VAddr addr, u64 size, bool written,
                        const VideoCore::BufferCache& cache, const VideoCore::Buffer* bound) {
    static const std::vector<u64> hashes = [] {
        std::vector<u64> out;
        const char* env = std::getenv("BB_TRACE_SHADER");
        for (const char* at = env; at && *at;) {
            char* end = nullptr;
            out.push_back(std::strtoull(at, &end, 16));
            at = end && *end == ',' ? end + 1 : nullptr;
        }
        return out;
    }();
    if (hashes.empty() || std::ranges::find(hashes, stage.pgm_hash) == hashes.end()) {
        return;
    }
    const bool in_place = cache.IsInPlace(addr, size);
    const bool any_in_place = cache.IsAnyInPlace(addr, size);
    struct Key {
        u64 hash, addr, size;
        bool written;
        const char* bound;
        auto operator<=>(const Key&) const = default;
    };
    struct Entry {
        u64 count = 0;
        int place = 0; // 0 VRAM, 1 in place, 2 partly
    };
    static std::mutex mutex;
    static std::map<Key, Entry> entries;
    static auto report = std::chrono::steady_clock::now();
    std::scoped_lock lk{mutex};
    auto& entry = entries[{stage.pgm_hash, addr, size, written, cache.BoundPlace(bound)}];
    ++entry.count;
    entry.place = in_place ? 1 : any_in_place ? 2 : 0;
    const auto now = std::chrono::steady_clock::now();
    if (now - report < std::chrono::seconds(5)) {
        return;
    }
    report = now;
    static constexpr const char* Places[] = {"VRAM", "in place", "partly in place"};
    for (const auto& [key, value] : entries) {
        std::printf("Shader %016llx binds %#llx+%#llx %s, %s, bound: %s, x%llu (uses_dma %d)\n",
                    (unsigned long long)key.hash, (unsigned long long)key.addr,
                    (unsigned long long)key.size, key.written ? "written" : "read",
                    Places[value.place], key.bound, (unsigned long long)value.count, int(stage.uses_dma));
    }
    entries.clear();
}
} // namespace

namespace {
/// BB_LAYER_PAGED_STATS=1 (diagnostics): paged bindings per shader, every 5 s.
void NotePagedBinding(u64 pgm_hash, const Gcn::Buffer& vsharp, bool written) {
    static const bool enabled = std::getenv("BB_LAYER_PAGED_STATS") != nullptr;
    if (!enabled) {
        return;
    }
    static std::map<std::pair<u64, u64>, std::pair<u64, u64>> counts;
    static u32 printed = 0;
    auto& count = counts[{pgm_hash, (vsharp.base_address << 1) | u64(written)}];
    ++count.first;
    count.second = vsharp.GetSize();
    const u32 now = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (now - printed < 5) {
        return;
    }
    printed = now;
    for (const auto& [key, value] : counts) {
        std::printf("Paged binding: shader %016llx buffer %#llx+%#llx %s: %llu in 5 s\n",
                    (unsigned long long)key.first, (unsigned long long)(key.second >> 1),
                    (unsigned long long)value.second, (key.second & 1) ? "written" : "read",
                    (unsigned long long)value.first);
    }
    counts.clear();
}
} // namespace

void Rasterizer::BindSpecialBuffer(const Shader::BufferResource& desc, const Shader::Info& stage,
                                   u32 slot, u64 alignment) {
    switch (desc.buffer_type) {
    case Shader::BufferType::GdsBuffer: {
        const auto* gds = buffer_cache.GetGdsBuffer();
        buffer_infos.emplace_back(gds->Handle(), 0, gds->SizeBytes());
        needs_barrier |= runtime.IsBufferAccessed(gds, 0, gds->SizeBytes());
        break;
    }
    case Shader::BufferType::Flatbuf: {
        // The flattened user data (ReadConst), copied into the stream buffer.
        const auto flat = stage.FlatUserData();
        const u32 size = flat.size() * sizeof(u32);
        if (FrameCapture::Active()) {
            FrameCapture::Buffer(stage.pgm_hash, slot, 0, flat.data(), size);
        }
        auto& stream = buffer_cache.GetStreamBuffer();
        const u64 offset = stream.Copy(flat.data(), size, alignment);
        buffer_infos.emplace_back(stream.Handle(), offset, size);
        break;
    }
    case Shader::BufferType::ClipPlanes: {
        // Permutations compiled without enabled planes never read the buffer: a null
        // descriptor satisfies the binding.
        if (Regs().clipper_control.user_clip_plane_enable == 0) {
            buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
            break;
        }
        std::array<float, Gcn::NUM_CLIP_PLANES * 4> planes{};
        for (u32 i = 0; i < Gcn::NUM_CLIP_PLANES; ++i) {
            const auto& plane = Regs().clip_user_data[i];
            planes[i * 4 + 0] = std::bit_cast<float>(plane.data_x);
            planes[i * 4 + 1] = std::bit_cast<float>(plane.data_y);
            planes[i * 4 + 2] = std::bit_cast<float>(plane.data_z);
            planes[i * 4 + 3] = std::bit_cast<float>(plane.data_w);
        }
        auto& stream = buffer_cache.GetStreamBuffer();
        const u32 size = static_cast<u32>(sizeof(planes));
        const u64 offset = stream.Copy(planes.data(), size, alignment);
        buffer_infos.emplace_back(stream.Handle(), offset, size);
        break;
    }
    case Shader::BufferType::BdaPagetable: {
        const auto* table = buffer_cache.GetBdaPageTableBuffer();
        buffer_infos.emplace_back(table->Handle(), 0, table->SizeBytes());
        break;
    }
    case Shader::BufferType::FaultBuffer: {
        const auto* faults = buffer_cache.GetFaultBuffer();
        buffer_infos.emplace_back(faults->Handle(), 0, faults->SizeBytes());
        break;
    }
    case Shader::BufferType::SharedMemory: {
        // LDS as a storage buffer: zeroed, one block per workgroup.
        auto& stream = buffer_cache.GetStreamBuffer();
        const auto& cs = CsRegs();
        const auto size = cs.SharedMemSize() * cs.NumWorkgroups();
        const auto [data, offset] = stream.Map(size, alignment);
        std::memset(data, 0, size);
        stream.Commit();
        buffer_infos.emplace_back(stream.Handle(), offset, size);
        break;
    }
    default:
        UNREACHABLE_MSG("Unexpected buffer type {}", u32(desc.buffer_type));
    }
}

void Rasterizer::BindGuestBuffer(const Shader::BufferResource& desc, const Gcn::Buffer& vsharp,
                                 const Shader::Info& stage, u32 slot, Shader::PushData& push_data,
                                 u64 alignment) {
    const VAddr address = vsharp.base_address;
    if (vsharp.GetSize() == 864 && gbuffer_draw && memory->IsValidGpuMapping(address, 0)) {
        camera_motion->OnConstants(reinterpret_cast<const float*>(address)); // the main camera
    }
    if (FrameCapture::Active()) {
        const bool mapped = address != 0 && memory->IsValidGpuMapping(address, 0);
        // Object motion research: the vertex shader buffers of G-buffer draws.
        if (gbuffer_draw && stage.sw_stage == Shader::SwStage::Vertex && mapped) {
            const auto* f = reinterpret_cast<const float*>(address);
            char note[192];
            std::snprintf(note, sizeof(note),
                          "\n  gb vs %08x slot %u at %#llx size %u stride %u: %g %g %g %g",
                          u32(stage.pgm_hash), slot, (unsigned long long)address,
                          u32(vsharp.GetSize()), u32(vsharp.GetStride()), f[0], f[1], f[2], f[3]);
            FrameCapture::Note(note);
        }
        if (mapped && vsharp.GetSize() != 0) {
            FrameCapture::Buffer(stage.pgm_hash, slot, address,
                                 reinterpret_cast<const void*>(address),
                                 memory->ClampRangeSize(address, vsharp.GetSize()));
        }
    }
    if (address == 0 || vsharp.GetSize() == 0) {
        buffer_infos.emplace_back(VK_NULL_HANDLE, 0, VK_WHOLE_SIZE);
        return;
    }
    if (Shader::IsPagedBuffer(VideoCore::BufferCache::LayerPagedActive(), vsharp)) {
        // BB_LAYER_MEMORY: nearly all memory as one buffer. The shader reaches it through the
        // page table and the descriptor holds its record; it may touch any buffer of the game's
        // memory: barriers before it, and after it if it writes.
        const auto [record, record_offset] =
            buffer_cache.LayerPagedRecord(address, vsharp.GetSize(), desc.is_written);
        NotePagedBinding(stage.pgm_hash, vsharp, desc.is_written);
        if (!desc.is_written && InPlaceReads::Enabled()) {
            InPlaceReads::Note(address, vsharp.GetSize(), scheduler.CurrentTick(), stage.pgm_hash, 1);
        }
        push_data.AddOffset(slot, 0);
        buffer_infos.emplace_back(record->Handle(), record_offset, 32);
        bound_buffers.emplace_back(nullptr, 0, 0, desc.is_written);
        if (desc.is_written) {
            texture_cache.InvalidateMemoryFromGPU(
                address, memory->ClampRangeSize(address, vsharp.GetSize()));
        }
        needs_barrier = true;
        return;
    }
    const u64 size = memory->ClampRangeSize(address, vsharp.GetSize());
    if (size != vsharp.GetSize()) {
        LOG_ERROR(Render, "Clamped size from {} to {} for stage {:#x}", vsharp.GetSize(), size,
                  stage.pgm_hash);
    }
    const auto [buffer, offset] =
        buffer_cache.ObtainBuffer(address, size, desc.is_written, desc.is_formatted);
    if (!desc.is_written && InPlaceReads::Enabled()) {
        InPlaceReads::Note(address, size, scheduler.CurrentTick(), stage.pgm_hash, 0);
    }
    TraceShaderBinding(stage, address, size, desc.is_written, buffer_cache, buffer);
    // Bound at the storage alignment below; the shader adds the rest (PushData::buf_offsets).
    const u64 offset_aligned = Common::AlignDown(offset, alignment);
    const u64 adjust = offset - offset_aligned;
    if (adjust % 4 != 0) {
        LOG_WARNING(Render_Vulkan, "Buffer binding in shader {:#x} isn't dword aligned",
                    stage.pgm_hash);
    }
    push_data.AddOffset(slot, adjust);
    // BB_LAYER_MEMORY: a huge binding may end at the end of its guest chunk's buffer.
    buffer_infos.emplace_back(buffer->Handle(), offset_aligned,
                              std::min<u64>(size + adjust, buffer->SizeBytes() - offset_aligned));
    bound_buffers.emplace_back(buffer, offset, size, desc.is_written);
    if (desc.is_written) {
        // Raw storage-buffer writes can also make an aliased cached image stale.
        texture_cache.InvalidateMemoryFromGPU(address, size);
    }
    needs_barrier |= runtime.IsBufferAccessed(buffer, offset, size, desc.is_written);
}

void Rasterizer::BindBuffers(const Shader::Info& stage, const PreparedStage* prepared,
                             Shader::Backend::Bindings& binding,
                             Shader::PushData& push_data, u32& write_index) {
    BB_SECTION(BindBuffers);
    const u64 alignment = instance.StorageMinAlignment();
    for (u32 index = 0; index < stage.buffers.size(); ++index) {
        const auto& desc = stage.buffers[index];
        if (const RingBinding* ring = num_ring_stages ? FindRingBinding(stage, index) : nullptr) {
            // Copied into the constant ring by the GPU command thread (read-only).
            if (!desc.IsSpecial()) {
                if (ring->size == 864 && gbuffer_draw &&
                    memory->IsValidGpuMapping(ring->address, 0)) {
                    camera_motion->OnConstants(reinterpret_cast<const float*>(ring->address));
                }
                push_data.AddOffset(binding.buffer, 0);
            }
            buffer_infos.emplace_back(RingHandle(ring->ring), ring->offset, ring->size);
        } else if (desc.IsSpecial()) {
            BindSpecialBuffer(desc, stage, binding.buffer, alignment);
        } else {
            const auto vsharp = prepared ? prepared->buffer_sharps[index] : desc.GetSharp(stage);
            BindGuestBuffer(desc, vsharp, stage, binding.buffer, push_data, alignment);
        }
        set_writes[write_index++] = vk::WriteDescriptorSet{
            .dstBinding = binding.unified++,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pBufferInfo = &buffer_infos.back(),
        };
        ++binding.buffer;
    }
}

} // namespace Vulkan
