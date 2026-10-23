// SPDX-License-Identifier: GPL-2.0-or-later
// bbport render: Rasterizer, texture and sampler binding with the memoized descriptions, lookups and texture
// sets.

#include <atomic>
#include <cstring>
#include <unordered_map>
#include <chrono>
#include "bbport_button_prompts.h"
#include <unordered_set>
#include <mutex>
#include "render/raster_internal.h"

namespace Vulkan {

Rasterizer::ImageDescCacheEntry& Rasterizer::CachedImageDescEntry(const Gcn::Image& sharp,
                                                                   const Shader::ImageResource& res,
                                                                   u64 hash) {
    std::array<u64, 4> key;
    std::memcpy(key.data(), &sharp, sizeof(key));
    const u32 flags = u32(res.is_written) | u32(res.is_depth) << 1 | u32(res.is_array) << 2;
    // Two-way set associative: a frame binds a few thousand distinct T#s.
    auto* set = &image_desc_cache[(hash % (image_desc_cache.size() / 2)) * 2];
    const auto matches = [&](const ImageDescCacheEntry& e) {
        return e.flags == flags && e.sharp == key;
    };
    const bool disabled = BbToggle::Disabled(BbToggle::ImageDescCache);
    ImageDescCacheEntry* slot = nullptr;
    if (!disabled && matches(set[0])) {
        slot = &set[0];
    } else if (!disabled && matches(set[1])) {
        slot = &set[1];
    } else {
        // Replace the least recently used way that no binding of this call points at.
        slot = set[0].last_use <= set[1].last_use ? &set[0] : &set[1];
        if (slot->pinned == bind_epoch) {
            slot = slot == &set[0] ? &set[1] : &set[0];
        }
        if (slot->pinned == bind_epoch) {
            slot = &image_desc_overflow.emplace_back();
        }
    }
    slot->last_use = ++desc_use_counter;
    auto& entry = *slot;
    if (entry.flags != flags || entry.sharp != key ||
        BbToggle::Disabled(BbToggle::ImageDescCache)) {
        entry.desc = VideoCore::TextureCache::ImageDesc{sharp, res};
        entry.sharp = key;
        entry.flags = flags;
        entry.found_generation = ~0ULL;
        entry.view_memo = {};
    }
    return entry;
}

bool Rasterizer::OverlapsTarget(VideoCore::ImageId image_id,
                                const VideoCore::SubresourceRange& range) const {
    const auto overlaps = [&](const VideoCore::SubresourceRange& target) {
        return range.base.level < target.base.level + target.extent.levels &&
               target.base.level < range.base.level + range.extent.levels &&
               range.base.layer < target.base.layer + target.extent.layers &&
               target.base.layer < range.base.layer + range.extent.layers;
    };
    // (Slots past this draw's targets may be stale: at worst a loop is assumed.)
    for (const auto& [id, desc] : cb_descs) {
        if (id == image_id && overlaps(desc.view_info.range)) {
            return true;
        }
    }
    return db_desc.first == image_id && overlaps(db_desc.second.view_info.range);
}

bool Rasterizer::TransitSampled(VideoCore::Image& image,
                                const VideoCore::TextureCache::ImageDesc& desc,
                                const VideoCore::Image::Binding& binding, bool is_storage) {
    // Also a storage image of this draw (its own descriptor) or a render target (a feedback
    // loop): the whole image in the general or feedback loop layout. Depth images are never
    // storage, and a feedback loop makes no sense for them.
    if ((binding.force_general || binding.is_target) && !image.info.props.is_depth) {
        if (instance.IsAttachmentFeedbackLoopLayoutSupported() && binding.is_target) {
            return runtime.Transit(&image, vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT,
                                   vk::PipelineStageFlagBits2::eAllGraphics,
                                   vk::AccessFlagBits2::eShaderRead);
        }
        return runtime.Transit(&image, vk::ImageLayout::eGeneral,
                               vk::PipelineStageFlagBits2::eAllCommands,
                               vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite);
    }
    if (is_storage) {
        return runtime.Transit(&image, vk::ImageLayout::eGeneral,
                               vk::PipelineStageFlagBits2::eAllCommands,
                               vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
                               desc.view_info.range);
    }
    return runtime.Transit(&image,
                           image.info.props.is_depth ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                                     : vk::ImageLayout::eShaderReadOnlyOptimal,
                           vk::PipelineStageFlagBits2::eAllCommands,
                           vk::AccessFlagBits2::eShaderRead, desc.view_info.range);
}

void Rasterizer::BindTextures(const Shader::Info& stage, const PreparedStage* prepared,
                              Shader::Backend::Bindings& binding, u32& write_index,
                              bool& barrier, bool on_helper) {
    BB_SECTION(BindTextures);
    const u32 first_image_idx = image_infos.size();
    TextureSet* set_slot = nullptr;
    if (!on_helper && BindTexturesFromSet(stage, prepared, first_image_idx, barrier, set_slot)) {
        // Descriptor writes: one per image (no mip arrays in a memoized set).
        for (u32 i = 0; i < stage.images.size(); ++i) {
            set_writes[write_index++] = vk::WriteDescriptorSet{
                .dstBinding = binding.unified++,
                .descriptorCount = 1,
                .descriptorType = vk::DescriptorType::eSampledImage,
                .pImageInfo = &image_infos[first_image_idx + i],
            };
        }
        BindSamplers(stage, prepared, binding, write_index);
        return;
    }
    image_bindings.clear();
    image_binding_entries.clear();
    image_desc_overflow.clear();
    image_desc_storage.clear();
    ++bind_epoch;
    // To emulate storing to explicit mip levels, build a descriptor array with each mip level.
    boost::container::small_vector<u32, 8> image_descriptor_array_sizes;
    // TextureSetMemo: what this call resolves, to remember when the set qualifies.
    std::array<TextureSetEntry, TextureSet::MaxImages> resolved{};
    bool set_ok = set_slot != nullptr;
    // A proxy upgraded while binding (SceneTargets::UpgradeProxy) leaves earlier bindings with
    // views of the replaced proxy: such a set is not remembered.
    const u64 scene_generation = scene_targets->Generation();
    // SampleSceneProxies: bindings that may read a reduced scene proxy instead of the native
    // image (normalized sampling only; see Shader::ImageResource::needs_native).
    boost::container::small_vector<bool, 16> binding_proxy_ok;
    const bool sample_proxies = !BbToggle::Disabled(BbToggle::SampleSceneProxies) &&
                                !on_helper && scene_targets->Reduced();
    // An invalid T# binds a null descriptor.
    const auto bind_null = [&] {
        image_bindings.emplace_back(VideoCore::ImageId{}, &image_desc_storage.emplace_back());
        image_binding_entries.push_back(nullptr);
        binding_proxy_ok.push_back(false);
        image_descriptor_array_sizes.push_back(1);
    };

    for (u32 image_index = 0; image_index < stage.images.size(); ++image_index) {
        const auto& image_desc = stage.images[image_index];
        const auto tsharp =
            prepared ? prepared->image_sharps[image_index] : image_desc.GetSharp(stage);
        // bbport: a hash lookup per texture per draw for a diagnostic only.
        static const bool warn_meta = std::getenv("BB_WARN_META_TEXTURE") != nullptr;
        if (warn_meta && texture_cache.IsMeta(tsharp.Address())) {
            LOG_WARNING(Render_Vulkan, "Unexpected metadata read by a shader (texture)");
        }

        const auto data_fmt = tsharp.GetDataFmt();
        const auto num_fmt = tsharp.GetNumberFmt();
        if (tsharp.Address() == 0 || data_fmt == Gcn::DataFormat::FormatInvalid) {
            bind_null();
            continue;
        }

        if (!memory->IsValidGpuMapping(tsharp.Address(), 0) ||
            !IsKnownFormat(data_fmt, num_fmt)) {
            LOG_WARNING(Render_Vulkan,
                        "Rejecting invalid T# address={:#x}, pitch={}, width={}, "
                        "data_format={}, num_format={}",
                        tsharp.Address(), tsharp.pitch, tsharp.width, static_cast<u32>(data_fmt),
                        static_cast<u32>(num_fmt));
            bind_null();
            continue;
        }

        const Shader::MipStorageFallbackMode mip_fallback_mode = image_desc.mip_fallback_mode;
        // Same as image_desc.NumBindings(stage), without fetching the T# again.
        const u32 num_bindings =
            mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex
                ? static_cast<u32>(tsharp.last_level - tsharp.base_level + 1)
                : 1u;
        const bool proxy_candidate = sample_proxies && num_bindings == 1 &&
                                     !image_desc.needs_native && !image_desc.is_written &&
                                     mip_fallback_mode == Shader::MipStorageFallbackMode::None;

        auto& desc_entry =
            prepared ? CachedImageDescEntry(tsharp, image_desc, prepared->image_hashes[image_index])
                     : CachedImageDescEntry(tsharp, image_desc);
        // BB_SCENE_DEBUG: proxied scene targets this binding reads at the native size, and why.
        const auto debug_native = [&](const VideoCore::Image& image) {
            if (!scene_debug_frame || proxy_candidate || !image.scene_proxy) return;
            std::printf("Scene native read: %s %016llx %s %ux%u needs_native %d written %d "
                        "bindings %u mip_fallback %d\n",
                        stage.sw_stage == Shader::SwStage::Compute ? "cs" : "gfx",
                        (unsigned long long)stage.pgm_hash,
                        vk::to_string(image.info.pixel_format).c_str(), image.info.size.width,
                        image.info.size.height, int(image_desc.needs_native),
                        int(image_desc.is_written), num_bindings, int(mip_fallback_mode));
        };
        // The image bound: a stencil image with an associated depth image is read through the
        // depth-stencil image; one bound twice and written is used in the general layout.
        const auto mark_bound = [&](VideoCore::ImageId& image_id) {
            auto* image = &texture_cache.GetImage(image_id);
            if (const auto depth_image_id = texture_cache.GetAssociatedDepth(*image)) {
                image_id = depth_image_id;
                image = &texture_cache.GetImage(image_id);
            }
            debug_native(*image);
            if (image->binding.is_bound) {
                image->binding.force_general |= image_desc.is_written;
            }
            image->binding.is_bound = 1u;
        };
        for (u32 i = 0; i < num_bindings; i++) {
            // bbport: a plain binding (no mip override) of the same T# resolves to the same image
            // while no image was registered or unregistered.
            if (mip_fallback_mode == Shader::MipStorageFallbackMode::None &&
                desc_entry.found_generation == texture_cache.RegistryGeneration() &&
                !BbToggle::Disabled(BbToggle::TextureBindingMemo)) {
                desc_entry.pinned = bind_epoch;
                auto& [image_id, _] =
                    image_bindings.emplace_back(desc_entry.found_id, &desc_entry.found_desc);
                image_binding_entries.push_back(&desc_entry);
                binding_proxy_ok.push_back(proxy_candidate);
                texture_cache.MarkFound(image_id);
                mark_bound(image_id);
                continue;
            }
            auto& desc = image_desc_storage.emplace_back(desc_entry.desc);
            auto& [image_id, _] = image_bindings.emplace_back(VideoCore::ImageId{}, &desc);
            image_binding_entries.push_back(nullptr);
            binding_proxy_ok.push_back(proxy_candidate);

            if (mip_fallback_mode == Shader::MipStorageFallbackMode::ConstantIndex) {
                ASSERT(num_bindings == 1);
                desc.view_info.range.base.level += image_desc.constant_mip_index;
                desc.view_info.range.extent.levels = 1;
            } else if (mip_fallback_mode == Shader::MipStorageFallbackMode::DynamicIndex) {
                desc.view_info.range.base.level += i;
                desc.view_info.range.extent.levels = 1;
            }

            const u64 generation = texture_cache.RegistryGeneration();
            image_id = texture_cache.FindImage(desc);
            if (mip_fallback_mode == Shader::MipStorageFallbackMode::None &&
                generation == texture_cache.RegistryGeneration() &&
                desc_entry.pinned != bind_epoch) {
                desc_entry.found_generation = generation;
                desc_entry.found_id = image_id;
                desc_entry.found_desc = desc;
                desc_entry.view_memo = {};
            }
            mark_bound(image_id);
        }

        image_descriptor_array_sizes.push_back(num_bindings);
    }

    // bbport (macOS): compressible images this stage needs in full usage (a storage or another
    // format's view, or a feedback loop with a target of this draw) are upgraded before any
    // binding takes a view, so that no binding of the stage holds a view of a replaced backing.
    // (The helper never gets such stages: TexturesBindableOnHelper.)
    if (!on_helper && instance.IsCompressibleImages()) {
        for (const auto& [image_id, desc_ptr] : image_bindings) {
            if (!image_id) {
                continue;
            }
            auto& image = texture_cache.GetImage(image_id);
            if (!image.backing) {
                continue;
            }
            if (image.binding.is_target && !pipeline_is_compute &&
                OverlapsTarget(image_id, desc_ptr->view_info.range) &&
                (image.backing->compressible || image.LacksFeedbackUsage())) {
                runtime.UpgradeUsage(&image, "feedback loop", true);
            } else if (image.NeedsFullUsage(desc_ptr->view_info)) {
                runtime.UpgradeUsage(&image, desc_ptr->view_info.is_storage ? "storage view"
                                                                           : "view format");
            }
        }
    }

    // Second pass to re-bind images that were updated after binding
    for (u32 binding_index = 0; binding_index < image_bindings.size(); ++binding_index) {
        auto& [image_id, desc_ptr] = image_bindings[binding_index];
        auto* memo_entry = image_binding_entries[binding_index];
        const auto& desc = *desc_ptr;
        bool is_storage = desc.type == VideoCore::TextureCache::BindingType::Storage;
        if (!image_id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        } else {
            if (auto& old_image = texture_cache.GetImage(image_id);
                old_image.binding.needs_rebind) {
                old_image.binding = {};
                auto& rebind_desc = image_desc_storage.emplace_back(desc);
                desc_ptr = &rebind_desc;
                image_id = texture_cache.FindImage(rebind_desc);
                memo_entry = nullptr;
                set_ok = false;
            }

            bound_images.emplace_back(image_id);

            auto& image = texture_cache.GetImage(image_id);
            auto& image_view = texture_cache.FindTexture(
                image_id, desc, memo_entry ? &memo_entry->view_memo : nullptr, !on_helper);
            const auto binding = image.binding;

            // BB_PROMPT_TRACE=1: a 32x32 texture holding a key guide icon's bytes, and the image
            // the cache found for it (once per address).
            if (static const bool trace = std::getenv("BB_PROMPT_TRACE") != nullptr;
                trace && desc.info.size.width <= 128 && desc.info.size.height <= 128) {
                static std::mutex traced_mutex;
                static std::unordered_set<VAddr> traced;
                const VAddr address = desc.info.guest_address;
                std::scoped_lock lock{traced_mutex};
                if (traced.insert(address).second) {
                    // Uncompressed ones are saved as they are in memory (out/present/tex_*.raw).
                    if (!desc.info.props.is_block && desc.info.num_bits == 32 &&
                        desc.info.size.width <= 64 && desc.info.size.height <= 64) {
                        char path[256];
                        std::snprintf(path, sizeof(path), "out/present/tex_%llx_%ux%u_t%u.raw",
                                      (unsigned long long)address, desc.info.size.width,
                                      desc.info.size.height, u32(desc.info.tile_mode));
                        if (FILE* f = std::fopen(path, "wb")) {
                            std::fwrite(reinterpret_cast<const void*>(address), 1,
                                        std::min<u32>(desc.info.guest_size, 64 * 64 * 4), f);
                            std::fclose(f);
                        }
                    }
                    std::printf("Prompt trace: bound %#llx %ux%u %s (icon %u) -> image %#llx %ux%u "
                                "levels %u flag %u at %llu ms\n",
                                (unsigned long long)address, desc.info.size.width,
                                desc.info.size.height, vk::to_string(desc.info.pixel_format).c_str(),
                                desc.info.guest_size >= 1024 ?
                                BbPrompts::IconOf(reinterpret_cast<const void*>(address), 1024) : 0,
                                (unsigned long long)image.info.guest_address, image.info.size.width,
                                image.info.size.height, image.info.resources.levels, u32(image.prompt_icon),
                                (unsigned long long)(std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch()).count()));
                }
            }
            // BB_UI_TEXTURE_LOG=<file>: when the file appears, every texture the UI draws of the
            // next 2 s bind (once each), for finding what a HUD element samples.
            if (static const char* ui_log = std::getenv("BB_UI_TEXTURE_LOG");
                ui_log && upscaler->Enabled() && upscaler->InUiPhase()) {
                static std::atomic<u64> until_ms{0};
                static std::mutex ui_mutex;
                static std::unordered_set<VAddr> ui_seen;
                const u64 now_ms = u64(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
                if (until_ms.load() == 0 && std::remove(ui_log) == 0) {
                    until_ms = now_ms + 2000;
                }
                if (now_ms < until_ms.load()) {
                    std::scoped_lock lock{ui_mutex};
                    if (ui_seen.insert(desc.info.guest_address).second) {
                        // BC7 atlases' contents as the GPU holds them (out/present).
                        if (desc.info.pixel_format == vk::Format::eBc7UnormBlock &&
                            image.info.size.width >= 512) {
                            static int dumped = 0;
                            scheduler.EndRendering();
                            const vk::Image vk_image = image.GetImage();
                            const u32 w = image.info.size.width, h = image.info.size.height;
                            const int index = dumped++;
                            scheduler.Record([this, vk_image, w, h, index](vk::CommandBuffer cmdbuf) {
                                Vulkan::DumpRaw(instance, scheduler, cmdbuf, vk_image, w, h, 1, index,
                                                "uiatlas", "bc7");
                            });
                        }
                        std::printf("UI texture: %#llx %ux%u levels %u %s tile %u -> image %#llx %ux%u\n",
                                    (unsigned long long)desc.info.guest_address, desc.info.size.width,
                                    desc.info.size.height, desc.info.resources.levels,
                                    vk::to_string(desc.info.pixel_format).c_str(),
                                    u32(desc.info.tile_mode),
                                    (unsigned long long)image.info.guest_address,
                                    image.info.size.width, image.info.size.height);
                    }
                }
            }
            // BB_PROMPT_TRACE=1: larger block-compressed textures rescanned for an icon's micro
            // tile (the game may copy icons into an atlas on the GPU), every 5 s per address.
            if (static const bool trace_atlas = std::getenv("BB_PROMPT_TRACE") != nullptr;
                trace_atlas && desc.info.props.is_block) {
                static std::mutex scan_mutex;
                static std::unordered_map<VAddr, u64> scanned;
                const VAddr address = desc.info.guest_address;
                const u64 now_ms = u64(std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count());
                bool scan = false;
                {
                    std::scoped_lock lock{scan_mutex};
                    auto& last = scanned[address];
                    if (now_ms - last > 5000) {
                        last = now_ms;
                        scan = true;
                    }
                }
                if (scan) {
                    const auto* bytes = reinterpret_cast<const u8*>(address);
                    // Any 16-byte block of the cross icon (KG_Cancel, 2) with colour in it: blocks
                    // stay whole whatever the layout.
                    if (const u8* cancel = BbPrompts::IconBytes(2)) {
                        u32 hits = 0, first = 0;
                        for (u32 offset = 0; offset + 16 <= desc.info.guest_size; offset += 16) {
                            for (u32 b : {27u, 28u, 35u, 36u}) {
                                if (std::memcmp(bytes + offset, cancel + b * 16, 16) == 0) {
                                    if (!hits++) first = offset;
                                }
                            }
                        }
                        if (hits) {
                            std::printf("Prompt trace: %u cross blocks in bound %#llx %ux%u %s from %#x\n",
                                        hits, (unsigned long long)address, desc.info.size.width,
                                        desc.info.size.height,
                                        vk::to_string(desc.info.pixel_format).c_str(), first);
                        }
                    }
                    for (u32 offset = 0; offset + 1024 <= desc.info.guest_size; offset += 1024) {
                        if (const u32 icon = BbPrompts::IconOf(bytes + offset, 1024)) {
                            std::printf("Prompt trace: icon %u in bound %#llx %ux%u %s at %#x\n", icon,
                                        (unsigned long long)address, desc.info.size.width,
                                        desc.info.size.height,
                                        vk::to_string(desc.info.pixel_format).c_str(), offset);
                        }
                    }
                }
            }
            // bbport: one of the game's key guide icons, drawn by the port for the device in use
            // (vk_button_prompts.h). Not remembered in the texture set: the device may change.
            // A copy the GPU made (DMA) is recognised here, by its bytes in memory.
            if (image.MaybePromptIcon() && prompts->Active()) {
                image.prompt_icon =
                    u8(BbPrompts::IconOf(reinterpret_cast<const void*>(image.info.guest_address), 1024));
            }
            if (image.prompt_atlas && !is_storage && prompts->Active()) {
                if (const vk::ImageView atlas = prompts->AtlasView(runtime, image, desc.view_info)) {
                    image.usage.texture = 1u;
                    image_infos.emplace_back(VK_NULL_HANDLE, atlas, vk::ImageLayout::eShaderReadOnlyOptimal);
                    set_ok = false;
                    continue;
                }
            }
            if (image.prompt_icon && !is_storage && prompts->Active()) {
                if (const vk::ImageView icon = prompts->View(image.prompt_icon, desc.view_info)) {
                    image.usage.texture = 1u;
                    image_infos.emplace_back(VK_NULL_HANDLE, icon, vk::ImageLayout::eShaderReadOnlyOptimal);
                    set_ok = false;
                    continue;
                }
            }

            // bbport: a proxied scene image sampled with normalized coordinates reads the proxy:
            // no resample to the native size (the native image is not touched).
            if (binding_proxy_ok[binding_index] && !is_storage && !binding.force_general &&
                !binding.is_target &&
                !(upscaler->Enabled() && upscaler->RedirectsSampled(image_id))) {
                if (const auto proxy = scene_targets->SampleProxy(image, desc.view_info)) {
                    image.usage.texture = 1u;
                    image_infos.emplace_back(VK_NULL_HANDLE, proxy->view, proxy->layout);
                    if (set_ok && binding_index < resolved.size()) {
                        resolved[binding_index] = {image_id, proxy->view, image.backing,
                                                   desc.view_info.range, true};
                    }
                    ++proxy_samples;
                    continue;
                }
            }
            // bbport (macOS): with compressible images a target of this draw sampled at another
            // mip or layer is no feedback loop (each range in its own layout).
            auto transit_binding = binding;
            if (binding.is_target && instance.IsCompressibleImages() && !pipeline_is_compute) {
                const bool feedback = OverlapsTarget(image_id, desc.view_info.range);
                transit_binding.is_target = feedback;
                image.binding.feedback |= feedback;
            }
            barrier |= TransitSampled(image, desc, transit_binding, is_storage);
            image.usage.storage |= is_storage;
            image.usage.texture |= !is_storage;
            if (FrameCapture::Active()) {
                FrameCapture::Sampled(image.info, is_storage);
            }

            vk::ImageView view = *image_view.image_view;
            vk::ImageLayout layout = image.backing->state.layout;
            if (upscaler->Enabled()) {
                // bbport: the display pass reads the upscaled frame (scaled presets).
                upscaler->RedirectSampled(image_id, image_view.info, view, layout);
            }
            image_infos.emplace_back(VK_NULL_HANDLE, view, layout);
            if (set_ok && binding_index < resolved.size()) {
                resolved[binding_index] = {image_id, *image_view.image_view, image.backing,
                                           desc.view_info.range};
                set_ok = !is_storage && !binding.force_general && !binding.is_target;
            }
        }
    }
    if (set_ok && image_bindings.size() == stage.images.size() &&
        scene_targets->Generation() == scene_generation) {
        // Remember the set: the same T#s resolve the same way while no image is (un)registered.
        set_slot->generation = texture_cache.RegistryGeneration();
        set_slot->scene_generation = scene_targets->Generation();
        set_slot->count = static_cast<u32>(stage.images.size());
        std::copy_n(resolved.begin(), set_slot->count, set_slot->entries.begin());
    } else if (set_slot) {
        set_slot->key = 0;
    }

    // One descriptor write per image resource: an array of single-level views for dynamic mip
    // indices.
    u32 info_index = first_image_idx;
    u32 binding_index = 0;
    for (const u32 array_size : image_descriptor_array_sizes) {
        const bool is_storage = image_bindings[binding_index].second->type ==
                                VideoCore::TextureCache::BindingType::Storage;
        set_writes[write_index++] = vk::WriteDescriptorSet{
            .dstBinding = binding.unified,
            .descriptorCount = array_size,
            .descriptorType =
                is_storage ? vk::DescriptorType::eStorageImage : vk::DescriptorType::eSampledImage,
            .pImageInfo = &image_infos[info_index],
        };
        info_index += array_size;
        binding_index += array_size;
        binding.unified += array_size;
    }

    BindSamplers(stage, prepared, binding, write_index);
}

// bbport: the PS4 build samples most scene textures with little or no anisotropic filtering.
// Behind the temporal upscaler the output shows the detail of the output's mip level, and
// surfaces seen at a grazing angle (stairs, floors, walls along the view) stay blurred.
// Mipmapped linear samplers of graphics draws get BB_ANISO (16; 0 = the game's) times.
void Rasterizer::ForceAnisotropy(Gcn::Sampler& sharp, bool is_depth) const {
    static const int forced = [] {
        const char* value = std::getenv("BB_ANISO");
        return value ? std::atoi(value) : 16;
    }();
    if (!is_depth && !pipeline_is_compute) {
        // The game's filters, once per kind: min filter, mip filter, ratio.
        static std::atomic<u64> seen{0};
        const u32 kind = u32(sharp.xy_min_filter) * 3 * 5 + u32(sharp.mip_filter) * 5 +
                         u32(sharp.max_aniso);
        // bbport (macOS): a relaxed load first; the read-modify-write (for every sampler of
        // every draw) only for a kind not printed yet.
        if (kind < 64 && !(seen.load(std::memory_order_relaxed) & (1ull << kind)) &&
            !(seen.fetch_or(1ull << kind, std::memory_order_relaxed) & (1ull << kind))) {
            std::printf("Sampler: game uses min filter %u, mip filter %u, anisotropy %.0fx\n",
                        u32(sharp.xy_min_filter), u32(sharp.mip_filter),
                        sharp.MaxAniso());
        }
    }
    if (forced <= 1 || is_depth || pipeline_is_compute || BbToggle::Disabled(BbToggle::ForcedAniso) ||
        sharp.mip_filter == Gcn::MipFilter::None ||
        (sharp.xy_min_filter != Gcn::Filter::Bilinear &&
         sharp.xy_min_filter != Gcn::Filter::AnisoLinear)) {
        return;
    }
    const auto ratio = forced >= 16 ? Gcn::AnisoRatio::Sixteen
        : forced >= 8 ? Gcn::AnisoRatio::Eight
        : forced >= 4 ? Gcn::AnisoRatio::Four : Gcn::AnisoRatio::Two;
    if (Gcn::IsAnisoFilter(sharp.xy_min_filter) && sharp.MaxAniso() >= forced) {
        return;
    }
    sharp.xy_min_filter = Gcn::Filter::AnisoLinear;
    if (sharp.xy_mag_filter == Gcn::Filter::Bilinear) {
        sharp.xy_mag_filter = Gcn::Filter::AnisoLinear;
    }
    sharp.max_aniso = ratio;
}

void Rasterizer::BindSamplers(const Shader::Info& stage, const PreparedStage* prepared,
                              Shader::Backend::Bindings& binding, u32& write_index) {
    for (u32 sampler_index = 0; sampler_index < stage.samplers.size(); ++sampler_index) {
        const auto& sampler = stage.samplers[sampler_index];
        auto ssharp =
            prepared ? prepared->sampler_sharps[sampler_index] : sampler.GetSharp(stage);
        ForceAnisotropy(ssharp, sampler.is_depth);
        const auto vk_sampler = texture_cache.GetSampler(
            ssharp, Regs().ta_bc_base, sampler.is_depth, sampler.is_depth ? 0.0f : sampler_lod_bias);
        image_infos.emplace_back(vk_sampler, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
        auto& set_write = set_writes[write_index++];
        set_write.dstSet = VK_NULL_HANDLE;
        set_write.dstBinding = binding.unified++;
        set_write.dstArrayElement = 0;
        set_write.descriptorCount = 1;
        set_write.descriptorType = vk::DescriptorType::eSampler;
        set_write.pImageInfo = &image_infos.back();
    }
}

bool Rasterizer::BindTexturesFromSet(const Shader::Info& stage, const PreparedStage* prepared,
                                     u32 first_image_idx, bool& barrier, TextureSet*& slot) {
    const u32 count = static_cast<u32>(stage.images.size());
    if (!prepared || count == 0 || count > TextureSet::MaxImages ||
        BbToggle::Disabled(BbToggle::TextureSetMemo) || FrameCapture::Active()) {
        return false;
    }
    for (const auto& image : stage.images) {
        if (image.is_written || image.mip_fallback_mode != Shader::MipStorageFallbackMode::None) {
            return false;
        }
    }
    u64 key = XXH3_64bits_withSeed(prepared->image_hashes, count * sizeof(u64),
                                   reinterpret_cast<u64>(&stage));
    key |= 1; // 0 marks an empty slot
    auto& set = texture_sets[key % texture_sets.size()];
    slot = &set;
    const u64 generation = texture_cache.RegistryGeneration();
    const bool match = set.key == key && set.stage == &stage && set.count == count &&
                       set.generation == generation &&
                       set.scene_generation == scene_targets->Generation() &&
                       std::equal(set.hashes.begin(), set.hashes.begin() + count,
                                  prepared->image_hashes);
    if (!match) {
        ++texture_set_why[set.key != key ? 0 : set.generation != generation ? 1 : 3];
        ++texture_set_misses;
        set.key = key;
        set.stage = &stage;
        set.count = 0;
        set.generation = ~0ull;
        std::copy_n(prepared->image_hashes, count, set.hashes.begin());
        return false;
    }
    // Every image still has the backing its view belongs to and needs no refresh.
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (!entry.id) {
            continue;
        }
        auto& image = texture_cache.GetImage(entry.id);
        // A key guide icon the GPU copied in, recognised by its bytes (then not from the set).
        if (image.MaybePromptIcon() && prompts->Active()) {
            image.prompt_icon =
                u8(BbPrompts::IconOf(reinterpret_cast<const void*>(image.info.guest_address), 1024));
        }
        // Proxy entries need a current proxy; native entries of a proxied image go through
        // BindTextures, which may sample the proxy instead.
        const bool proxies_on = !BbToggle::Disabled(BbToggle::SampleSceneProxies);
        if (image.backing != entry.backing || image.binding.needs_rebind ||
            image.binding.is_target || !texture_cache.IsUpToDate(entry.id) ||
            ((image.prompt_icon || image.prompt_atlas) && prompts->Active()) ||
            (entry.proxy ? !proxies_on ||
                               !scene_targets->ProxyCurrent(image, entry.range.base.level,
                                                            entry.range.extent.levels)
                         : image.scene_proxy && proxies_on) ||
            (upscaler->Enabled() && upscaler->RedirectsSampled(entry.id))) {
            ++texture_set_why[2];
            ++texture_set_misses;
            set.generation = ~0ull;
            return false;
        }
    }
    ++texture_set_hits;
    slot = nullptr;
    for (u32 i = 0; i < count; ++i) {
        const auto& entry = set.entries[i];
        if (!entry.id) {
            image_infos.emplace_back(VK_NULL_HANDLE, VK_NULL_HANDLE, vk::ImageLayout::eGeneral);
            continue;
        }
        texture_cache.MarkFound(entry.id);
        auto& image = texture_cache.GetImage(entry.id);
        image.binding.is_bound = 1u;
        bound_images.emplace_back(entry.id);
        image.usage.texture = 1u;
        if (entry.proxy) {
            image_infos.emplace_back(VK_NULL_HANDLE, entry.view,
                                     scene_targets->PrepareSample(image, entry.range.base.level));
            ++proxy_samples;
            continue;
        }
        const auto new_layout = image.info.props.is_depth
                                    ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
                                    : vk::ImageLayout::eShaderReadOnlyOptimal;
        barrier |= runtime.Transit(&image, new_layout, vk::PipelineStageFlagBits2::eAllCommands,
                                   vk::AccessFlagBits2::eShaderRead, entry.range);
        image_infos.emplace_back(VK_NULL_HANDLE, entry.view, image.backing->state.layout);
    }
    (void)first_image_idx;
    return true;
}

} // namespace Vulkan
