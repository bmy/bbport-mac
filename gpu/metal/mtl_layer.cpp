// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal command path's command layer (mtl_layer.h): counting (stage 3a) and, with
// the frame encoder, owning command buffers (stage 3b). The hook template and the function list
// follow Wooloomooloo2/bloodborne_mac_port's mtl_vk_layer.mm (78b59f5).

#include "metal/mtl_layer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/emulator_settings.h"
#include "vulkan/vk_common.h"
#ifdef BB_METAL_BACKEND
#include "metal/mtl_frame.h"
#endif

namespace MetalLayer {
namespace {

// What the renderer records and submits (vulkan.hpp's CommandBuffer and Queue members it calls,
// with the extension aliases vulkan.hpp may have filled instead of the core names), the objects
// the frame encoder keeps Metal counterparts of, and the lookups libraries make for themselves.
#define BB_LAYER_FUNCTIONS(X)                                                                      \
    X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer) X(vkQueueSubmit)         \
    X(vkQueueSubmit2) X(vkQueueSubmit2KHR) X(vkCmdPipelineBarrier) X(vkCmdPipelineBarrier2)        \
    X(vkCmdPipelineBarrier2KHR) X(vkCmdBindPipeline) X(vkCmdPushDescriptorSetKHR)                  \
    X(vkCmdPushConstants) X(vkCmdBindDescriptorSets) X(vkCmdDispatch) X(vkCmdDispatchIndirect)     \
    X(vkCmdBeginRendering) X(vkCmdBeginRenderingKHR) X(vkCmdEndRendering)                          \
    X(vkCmdEndRenderingKHR) X(vkCmdDraw) X(vkCmdDrawIndexed) X(vkCmdDrawIndirect)                  \
    X(vkCmdDrawIndexedIndirect) X(vkCmdDrawIndirectCount) X(vkCmdDrawIndirectCountKHR)             \
    X(vkCmdDrawIndexedIndirectCount) X(vkCmdDrawIndexedIndirectCountKHR)                           \
    X(vkCmdBindVertexBuffers) X(vkCmdBindVertexBuffers2) X(vkCmdBindVertexBuffers2EXT)             \
    X(vkCmdBindIndexBuffer) X(vkCmdCopyImage) X(vkCmdCopyBuffer) X(vkCmdCopyImageToBuffer)         \
    X(vkCmdCopyBufferToImage) X(vkCmdBlitImage) X(vkCmdResolveImage) X(vkCmdClearColorImage)       \
    X(vkCmdClearDepthStencilImage) X(vkCmdClearAttachments) X(vkCmdFillBuffer)                     \
    X(vkCmdUpdateBuffer) X(vkCmdSetViewport) X(vkCmdSetScissor) X(vkCmdSetViewportWithCount)       \
    X(vkCmdSetViewportWithCountEXT) X(vkCmdSetScissorWithCount) X(vkCmdSetScissorWithCountEXT)     \
    X(vkCmdSetDepthTestEnable) X(vkCmdSetDepthTestEnableEXT) X(vkCmdSetDepthWriteEnable)           \
    X(vkCmdSetDepthWriteEnableEXT) X(vkCmdSetDepthCompareOp) X(vkCmdSetDepthCompareOpEXT)          \
    X(vkCmdSetDepthBoundsTestEnable) X(vkCmdSetDepthBoundsTestEnableEXT) X(vkCmdSetDepthBounds)    \
    X(vkCmdSetDepthBiasEnable) X(vkCmdSetDepthBiasEnableEXT) X(vkCmdSetDepthBias)                  \
    X(vkCmdSetStencilTestEnable) X(vkCmdSetStencilTestEnableEXT) X(vkCmdSetStencilOp)              \
    X(vkCmdSetStencilOpEXT) X(vkCmdSetStencilReference) X(vkCmdSetStencilWriteMask)                \
    X(vkCmdSetStencilCompareMask) X(vkCmdSetPrimitiveRestartEnable)                                \
    X(vkCmdSetPrimitiveRestartEnableEXT) X(vkCmdSetRasterizerDiscardEnable)                        \
    X(vkCmdSetRasterizerDiscardEnableEXT) X(vkCmdSetCullMode) X(vkCmdSetCullModeEXT)               \
    X(vkCmdSetFrontFace) X(vkCmdSetFrontFaceEXT) X(vkCmdSetBlendConstants)                         \
    X(vkCmdSetColorWriteMaskEXT) X(vkCmdSetLineWidth) X(vkCmdSetVertexInputEXT)                    \
    X(vkCmdSetAttachmentFeedbackLoopEnableEXT) X(vkCmdBeginDebugUtilsLabelEXT)                     \
    X(vkCmdEndDebugUtilsLabelEXT) X(vkCmdInsertDebugUtilsLabelEXT) X(vkCmdWriteTimestamp)          \
    X(vkCmdWriteTimestamp2) X(vkCmdWriteTimestamp2KHR) X(vkCmdWriteBufferMarkerAMD)                \
    X(vkCmdBeginQuery) X(vkCmdEndQuery) X(vkCmdResetQueryPool) X(vkCmdCopyQueryPoolResults)        \
    X(vkCmdExecuteCommands) X(vkFreeCommandBuffers) X(vkCreateSampler) X(vkDestroySampler)         \
    X(vkCreateImageView) X(vkDestroyImageView) X(vkCreateBuffer) X(vkDestroyBuffer)                \
    X(vkCreateImage) X(vkDestroyImage) X(vkGetDeviceProcAddr) X(vkGetInstanceProcAddr)

enum Id : unsigned {
#define BB_ID(name) Id_##name,
    BB_LAYER_FUNCTIONS(BB_ID)
#undef BB_ID
        Count
};
constexpr const char* Names[] = {
#define BB_NAME(name) #name,
    BB_LAYER_FUNCTIONS(BB_NAME)
#undef BB_NAME
};

__attribute__((format(printf, 1, 2))) void Print(const char* format, ...) {
    va_list args;
    va_start(args, format);
    std::vprintf(format, args);
    va_end(args);
    std::fflush(stdout);
}

bool trace{}; ///< BB_METAL_TRACE=1
bool frame{}; ///< BB_METAL_FRAME=1 and the frame encoder is up

std::array<std::atomic<unsigned long long>, Count> calls{};
std::atomic<unsigned long long> submitted_buffers{0}; ///< command buffers in submissions
std::atomic<unsigned long long> submissions{0};       ///< vkQueueSubmit* calls
std::chrono::steady_clock::time_point since = std::chrono::steady_clock::now();
std::mutex report_mutex;

/// The layer's entry points by Id (those installed), for vkGet*ProcAddr.
std::array<PFN_vkVoidFunction, Count> hooked{};

/// Every 5 s, from a submission: the calls per second, most first.
void Report() {
    const auto now = std::chrono::steady_clock::now();
    std::unique_lock lock{report_mutex, std::try_to_lock};
    if (!lock || now - since < std::chrono::seconds(5)) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - since).count();
    since = now;
#ifdef BB_METAL_BACKEND
    if (frame) {
        MetalFrame::Report(seconds);
    }
#endif
    std::vector<std::pair<unsigned long long, unsigned>> top;
    unsigned long long commands = 0;
    for (unsigned i = 0; i < Count; ++i) {
        if (const auto n = calls[i].exchange(0, std::memory_order_relaxed)) {
            top.emplace_back(n, i);
            if (std::strncmp(Names[i], "vkCmd", 5) == 0) {
                commands += n;
            }
        }
    }
    if (!trace) {
        return;
    }
    std::sort(top.rbegin(), top.rend());
    std::string line;
    for (const auto& [n, i] : top) {
        char item[96];
        std::snprintf(item, sizeof item, "%s%s %.0f", line.empty() ? "" : ", ", Names[i] + 2,
                      double(n) / seconds);
        line += item;
    }
    const auto buffers = submitted_buffers.exchange(0, std::memory_order_relaxed);
    const auto subs = submissions.exchange(0, std::memory_order_relaxed);
    Print("Metal layer: per second %.0f commands, %.0f submissions of %.0f command buffers "
          "(%.0f commands each); calls per second: %s\n",
          double(commands) / seconds, double(subs) / seconds, double(buffers) / seconds,
          buffers ? double(commands) / double(buffers) : 0.0, line.c_str());
}

/// Once per name: a library looked a command up for itself.
void NoteLookup(const char* name, bool covered) {
    static std::mutex mutex;
    static std::vector<std::string> seen;
    std::scoped_lock lock{mutex};
    if (std::find(seen.begin(), seen.end(), name) != seen.end()) {
        return;
    }
    seen.emplace_back(name);
    if (!covered && frame) {
        Print("Metal layer: %s looked up, not covered: in owned command buffers its calls would "
              "go out of order (BB_METAL_FRAME=1)\n",
              name);
        return;
    }
    Print("Metal layer: %s looked up %s\n", name,
          covered ? "(the layer's entry)" : "(not covered: its calls bypass the layer)");
}

PFN_vkVoidFunction HookedByName(const char* name) {
    for (unsigned i = 0; i < Count; ++i) {
        if (hooked[i] && std::strcmp(Names[i], name) == 0) {
            return hooked[i];
        }
    }
    return nullptr;
}

#ifdef BB_METAL_BACKEND
// ---- Owned command buffers (frame mode) ----

std::mutex owned_mutex;
std::unordered_map<VkCommandBuffer, MetalFrame::Owned*> owned;
std::atomic<unsigned> owned_count{0}; ///< owned.size(): the lock is skipped while it is zero
/// Bumped whenever an owned command buffer appears or goes: a thread's cached lookup is valid
/// while it is unchanged (a command buffer is recorded by one thread at a time).
std::atomic<unsigned long long> owned_generation{0};

MetalFrame::Owned* FindOwned(VkCommandBuffer cb) {
    if (owned_count.load(std::memory_order_acquire) == 0) {
        return nullptr;
    }
    thread_local VkCommandBuffer cached_cb = VK_NULL_HANDLE;
    thread_local MetalFrame::Owned* cached = nullptr;
    thread_local unsigned long long cached_generation = ~0ull;
    const auto generation = owned_generation.load(std::memory_order_acquire);
    if (cb == cached_cb && generation == cached_generation) {
        return cached;
    }
    std::scoped_lock lock{owned_mutex};
    auto it = owned.find(cb);
    cached_cb = cb;
    cached = it == owned.end() ? nullptr : it->second;
    cached_generation = owned_generation.load(std::memory_order_acquire);
    return cached;
}

/// Takes the command buffer's state out of the map (null when not owned).
MetalFrame::Owned* TakeOwned(VkCommandBuffer cb) {
    if (owned_count.load(std::memory_order_acquire) == 0) {
        return nullptr;
    }
    std::scoped_lock lock{owned_mutex};
    auto it = owned.find(cb);
    if (it == owned.end()) {
        return nullptr;
    }
    MetalFrame::Owned* o = it->second;
    owned.erase(it);
    owned_count.store(unsigned(owned.size()), std::memory_order_release);
    ++owned_generation;
    return o;
}

void Own(VkCommandBuffer cb) {
    MetalFrame::Discard(TakeOwned(cb)); // begun again without a submission
    MetalFrame::Owned* o = MetalFrame::Begin(cb);
    std::scoped_lock lock{owned_mutex};
    owned[cb] = o;
    owned_count.store(unsigned(owned.size()), std::memory_order_release);
    ++owned_generation;
}

/// Submits through the frame encoder when the submission has an owned command buffer.
template <typename Info>
bool SubmitOwned(VkQueue queue, uint32_t count, const Info* submits, VkFence fence,
                 VkResult& result) {
    constexpr bool v2 = std::is_same_v<Info, VkSubmitInfo2>;
    bool any = false;
    for (uint32_t s = 0; s < count && !any; ++s) {
        uint32_t n;
        if constexpr (v2) {
            n = submits[s].commandBufferInfoCount;
        } else {
            n = submits[s].commandBufferCount;
        }
        for (uint32_t i = 0; i < n && !any; ++i) {
            VkCommandBuffer cb;
            if constexpr (v2) {
                cb = submits[s].pCommandBufferInfos[i].commandBuffer;
            } else {
                cb = submits[s].pCommandBuffers[i];
            }
            any = FindOwned(cb) != nullptr;
        }
    }
    if (!any) {
        return false;
    }
    struct Storage {
        std::vector<VkSemaphoreSubmitInfo> waits, signals;
        std::vector<VkCommandBuffer> cbs;
        std::vector<MetalFrame::Owned*> owned;
    };
    std::vector<Storage> storage(count);
    std::vector<MetalFrame::Batch> batches(count);
    for (uint32_t s = 0; s < count; ++s) {
        const Info& in = submits[s];
        Storage& st = storage[s];
        if constexpr (v2) {
            st.waits.assign(in.pWaitSemaphoreInfos,
                            in.pWaitSemaphoreInfos + in.waitSemaphoreInfoCount);
            st.signals.assign(in.pSignalSemaphoreInfos,
                              in.pSignalSemaphoreInfos + in.signalSemaphoreInfoCount);
            for (uint32_t i = 0; i < in.commandBufferInfoCount; ++i) {
                st.cbs.push_back(in.pCommandBufferInfos[i].commandBuffer);
            }
        } else {
            const VkTimelineSemaphoreSubmitInfo* values = nullptr;
            for (auto* p = static_cast<const VkBaseInStructure*>(in.pNext); p; p = p->pNext) {
                if (p->sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO) {
                    values = reinterpret_cast<const VkTimelineSemaphoreSubmitInfo*>(p);
                }
            }
            for (uint32_t i = 0; i < in.waitSemaphoreCount; ++i) {
                st.waits.push_back({
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .semaphore = in.pWaitSemaphores[i],
                    .value = values && i < values->waitSemaphoreValueCount
                                 ? values->pWaitSemaphoreValues[i]
                                 : 0,
                    .stageMask = in.pWaitDstStageMask
                                     ? VkPipelineStageFlags2(in.pWaitDstStageMask[i])
                                     : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                });
            }
            for (uint32_t i = 0; i < in.signalSemaphoreCount; ++i) {
                st.signals.push_back({
                    .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
                    .semaphore = in.pSignalSemaphores[i],
                    .value = values && i < values->signalSemaphoreValueCount
                                 ? values->pSignalSemaphoreValues[i]
                                 : 0,
                    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                });
            }
            st.cbs.assign(in.pCommandBuffers, in.pCommandBuffers + in.commandBufferCount);
        }
        for (VkCommandBuffer cb : st.cbs) {
            st.owned.push_back(TakeOwned(cb));
        }
        batches[s] = {
            .waits = st.waits.data(),
            .wait_count = uint32_t(st.waits.size()),
            .command_buffers = st.cbs.data(),
            .owned = st.owned.data(),
            .command_buffer_count = uint32_t(st.cbs.size()),
            .signals = st.signals.data(),
            .signal_count = uint32_t(st.signals.size()),
        };
    }
    result = MetalFrame::Submit(queue, batches.data(), count, fence);
    return true;
}
#endif

/// One forwarding function per dispatcher member: counts, encodes for an owned command buffer
/// (frame mode), else calls the driver.
template <unsigned I, typename F>
struct Hook;
template <unsigned I, typename R, typename... A>
struct Hook<I, R (*)(A...)> {
    static inline R (*original)(A...) = nullptr;

    static R Call(A... args) {
        calls[I].fetch_add(1, std::memory_order_relaxed);
        if constexpr (I == Id_vkQueueSubmit || I == Id_vkQueueSubmit2 ||
                      I == Id_vkQueueSubmit2KHR) {
            const auto [queue, count, submits, fence] = std::tuple{args...};
            submissions.fetch_add(1, std::memory_order_relaxed);
            for (uint32_t s = 0; s < count; ++s) {
                if constexpr (I == Id_vkQueueSubmit) {
                    submitted_buffers.fetch_add(submits[s].commandBufferCount,
                                                std::memory_order_relaxed);
                } else {
                    submitted_buffers.fetch_add(submits[s].commandBufferInfoCount,
                                                std::memory_order_relaxed);
                }
            }
            Report();
#ifdef BB_METAL_BACKEND
            if (frame) {
                VkResult result = VK_SUCCESS;
                if (SubmitOwned(queue, count, submits, fence, result)) {
                    return result;
                }
            }
#endif
            return original(args...);
        } else if constexpr (I == Id_vkGetDeviceProcAddr || I == Id_vkGetInstanceProcAddr) {
            // What the driver answers, replaced by the layer's entry for a function it covers
            // (the lookup function itself too, so later lookups come back here). Not where the
            // driver has none (a core name of an extension vulkan.hpp filled from its alias).
            const char* name = std::get<1>(std::tuple{args...});
            const R driver = original(args...);
            if (!driver || !name) {
                return driver;
            }
            PFN_vkVoidFunction ours = HookedByName(name);
            if (std::strncmp(name, "vkCmd", 5) == 0 ||
                std::strncmp(name, "vkQueueSubmit", 13) == 0) {
                NoteLookup(name, ours != nullptr);
            }
            return ours ? ours : driver;
        } else {
#ifdef BB_METAL_BACKEND
            if (frame) {
                if constexpr (FrameHandles()) {
                    return Frame(args...);
                }
            }
#endif
            return original(args...);
        }
    }

#ifdef BB_METAL_BACKEND
    /// The functions frame mode handles itself (the rest go straight to the driver).
    static constexpr bool FrameHandles() {
        return std::string_view{Names[I]}.starts_with("vkCmd") || I == Id_vkBeginCommandBuffer ||
               I == Id_vkEndCommandBuffer || I == Id_vkResetCommandBuffer ||
               I == Id_vkFreeCommandBuffers || I == Id_vkCreateSampler ||
               I == Id_vkDestroySampler || I == Id_vkCreateImageView ||
               I == Id_vkDestroyImageView || I == Id_vkCreateBuffer ||
               I == Id_vkDestroyBuffer || I == Id_vkCreateImage || I == Id_vkDestroyImage;
    }

    static R Frame(A... args) {
        namespace F = MetalFrame;
        const auto t = std::tuple{args...};
        if constexpr (I == Id_vkBeginCommandBuffer) {
            const R result = original(args...);
            const VkCommandBufferBeginInfo* info = std::get<1>(t);
            if (result == VK_SUCCESS && !info->pInheritanceInfo) { // secondaries stay the driver's
                Own(std::get<0>(t));
            } else {
                // A handle that was owned before (its pool reset or destroyed): no longer.
                MetalFrame::Discard(TakeOwned(std::get<0>(t)));
            }
            return result;
        } else if constexpr (I == Id_vkEndCommandBuffer) {
            if (F::Owned* o = FindOwned(std::get<0>(t))) {
                F::End(*o);
            }
            return original(args...);
        } else if constexpr (I == Id_vkResetCommandBuffer) {
            F::Discard(TakeOwned(std::get<0>(t)));
            return original(args...);
        } else if constexpr (I == Id_vkFreeCommandBuffers) {
            const auto [dev, pool, count, cbs] = t;
            for (uint32_t i = 0; i < count; ++i) {
                F::Discard(TakeOwned(cbs[i]));
            }
            return original(args...);
        } else if constexpr (I == Id_vkCreateSampler) {
            const R result = original(args...);
            if (result == VK_SUCCESS) {
                F::SamplerCreated(*std::get<3>(t), std::get<1>(t));
            }
            return result;
        } else if constexpr (I == Id_vkDestroySampler) {
            F::SamplerDestroyed(std::get<1>(t));
            return original(args...);
        } else if constexpr (I == Id_vkCreateImageView) {
            const R result = original(args...);
            if (result == VK_SUCCESS) {
                F::ImageViewCreated(*std::get<3>(t), std::get<1>(t));
            }
            return result;
        } else if constexpr (I == Id_vkDestroyImageView) {
            F::ImageViewDestroyed(std::get<1>(t));
            return original(args...);
        } else if constexpr (I == Id_vkCreateBuffer) {
            const R result = original(args...);
            if (result == VK_SUCCESS) {
                F::BufferCreated(*std::get<3>(t), std::get<1>(t)->size);
            }
            return result;
        } else if constexpr (I == Id_vkCreateImage) {
            const R result = original(args...);
            if (result == VK_SUCCESS) {
                F::ImageCreated(*std::get<3>(t), std::get<1>(t));
            }
            return result;
        } else if constexpr (I == Id_vkDestroyImage) {
            F::ImageDestroyed(std::get<1>(t));
            return original(args...);
        } else if constexpr (I == Id_vkDestroyBuffer) {
            F::BufferDestroyed(std::get<1>(t));
            return original(args...);
        } else {
            // vkCmd*: the first argument is the command buffer.
            F::Owned* o = FindOwned(std::get<0>(t));
            if (!o) {
                return original(args...);
            }
            Encode(*o, args...);
            if constexpr (!std::is_void_v<R>) {
                return R{};
            }
        }
    }

    /// One command of an owned command buffer.
    static void Encode(MetalFrame::Owned& o, A... args) {
        namespace F = MetalFrame;
        using S = F::State;
        const auto t = std::tuple{args...};
        const auto fallback = [&](bool in_pass) {
            F::Fallback(o, Names[I], in_pass, [&](VkCommandBuffer cb) {
                if (cb) {
                    std::apply(
                        [&](VkCommandBuffer, auto... rest) { original(cb, rest...); }, t);
                }
            });
        };
        if constexpr (I == Id_vkCmdPipelineBarrier) {
            std::apply([&](auto... a) { F::CmdPipelineBarrier1(o, a...); }, t);
        } else if constexpr (I == Id_vkCmdPipelineBarrier2 || I == Id_vkCmdPipelineBarrier2KHR) {
            F::CmdPipelineBarrier(o, std::get<0>(t), std::get<1>(t));
        } else if constexpr (I == Id_vkCmdBeginRendering || I == Id_vkCmdBeginRenderingKHR) {
            F::CmdBeginRendering(o, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdEndRendering || I == Id_vkCmdEndRenderingKHR) {
            F::CmdEndRendering(o);
        } else if constexpr (I == Id_vkCmdBindPipeline) {
            F::CmdBindPipeline(o, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdPushDescriptorSetKHR) {
            F::CmdPushDescriptorSet(o, std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                    std::get<4>(t), std::get<5>(t));
        } else if constexpr (I == Id_vkCmdBindDescriptorSets) {
            F::CmdBindDescriptorSets(o, std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                     std::get<4>(t), std::get<5>(t), std::get<6>(t),
                                     std::get<7>(t));
        } else if constexpr (I == Id_vkCmdPushConstants) {
            F::CmdPushConstants(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t),
                                std::get<5>(t));
        } else if constexpr (I == Id_vkCmdBindVertexBuffers) {
            F::CmdBindVertexBuffers(o, std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                    std::get<4>(t), nullptr, nullptr);
        } else if constexpr (I == Id_vkCmdBindVertexBuffers2 ||
                             I == Id_vkCmdBindVertexBuffers2EXT) {
            F::CmdBindVertexBuffers(o, std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                    std::get<4>(t), std::get<5>(t), std::get<6>(t));
        } else if constexpr (I == Id_vkCmdBindIndexBuffer) {
            F::CmdBindIndexBuffer(o, std::get<1>(t), std::get<2>(t), std::get<3>(t));
        } else if constexpr (I == Id_vkCmdSetViewport) {
            F::CmdSetViewports(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), false);
        } else if constexpr (I == Id_vkCmdSetViewportWithCount ||
                             I == Id_vkCmdSetViewportWithCountEXT) {
            F::CmdSetViewports(o, 0, std::get<1>(t), std::get<2>(t), true);
        } else if constexpr (I == Id_vkCmdSetScissor) {
            F::CmdSetScissors(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), false);
        } else if constexpr (I == Id_vkCmdSetScissorWithCount ||
                             I == Id_vkCmdSetScissorWithCountEXT) {
            F::CmdSetScissors(o, 0, std::get<1>(t), std::get<2>(t), true);
        } else if constexpr (I == Id_vkCmdSetCullMode || I == Id_vkCmdSetCullModeEXT) {
            F::CmdSetFlag(o, S::CullMode, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetFrontFace || I == Id_vkCmdSetFrontFaceEXT) {
            F::CmdSetFlag(o, S::FrontFace, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetDepthTestEnable ||
                             I == Id_vkCmdSetDepthTestEnableEXT) {
            F::CmdSetFlag(o, S::DepthTest, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetDepthWriteEnable ||
                             I == Id_vkCmdSetDepthWriteEnableEXT) {
            F::CmdSetFlag(o, S::DepthWrite, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetDepthCompareOp || I == Id_vkCmdSetDepthCompareOpEXT) {
            F::CmdSetFlag(o, S::DepthCompare, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetDepthBoundsTestEnable ||
                             I == Id_vkCmdSetDepthBoundsTestEnableEXT) {
            F::CmdSetFlag(o, S::DepthBoundsTest, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetDepthBounds) {
            F::CmdSetDepthBounds(o, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdSetDepthBiasEnable ||
                             I == Id_vkCmdSetDepthBiasEnableEXT) {
            F::CmdSetFlag(o, S::DepthBiasEnable, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetDepthBias) {
            F::CmdSetDepthBias(o, std::get<1>(t), std::get<2>(t), std::get<3>(t));
        } else if constexpr (I == Id_vkCmdSetStencilTestEnable ||
                             I == Id_vkCmdSetStencilTestEnableEXT) {
            F::CmdSetFlag(o, S::StencilTest, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetStencilOp || I == Id_vkCmdSetStencilOpEXT) {
            F::CmdSetStencilOp(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t),
                               std::get<5>(t));
        } else if constexpr (I == Id_vkCmdSetStencilReference) {
            F::CmdSetStencilValue(o, S::StencilReference, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdSetStencilWriteMask) {
            F::CmdSetStencilValue(o, S::StencilWriteMask, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdSetStencilCompareMask) {
            F::CmdSetStencilValue(o, S::StencilCompareMask, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdSetPrimitiveRestartEnable ||
                             I == Id_vkCmdSetPrimitiveRestartEnableEXT) {
            F::CmdSetFlag(o, S::PrimitiveRestart, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetRasterizerDiscardEnable ||
                             I == Id_vkCmdSetRasterizerDiscardEnableEXT) {
            F::CmdSetFlag(o, S::RasterizerDiscard, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetBlendConstants) {
            F::CmdSetBlendConstants(o, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetColorWriteMaskEXT) {
            F::CmdSetColorWriteMask(o, std::get<1>(t), std::get<2>(t), std::get<3>(t));
        } else if constexpr (I == Id_vkCmdSetLineWidth) {
            F::CmdSetLineWidth(o, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdSetAttachmentFeedbackLoopEnableEXT) {
            F::CmdSetFlag(o, S::FeedbackLoop, std::get<1>(t));
        } else if constexpr (I == Id_vkCmdDraw) {
            F::CmdDraw(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t));
        } else if constexpr (I == Id_vkCmdDrawIndexed) {
            F::CmdDrawIndexed(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t),
                              std::get<5>(t));
        } else if constexpr (I == Id_vkCmdDrawIndirect || I == Id_vkCmdDrawIndexedIndirect) {
            F::CmdDrawIndirect(o, std::get<1>(t), std::get<2>(t), std::get<3>(t), std::get<4>(t),
                               I == Id_vkCmdDrawIndexedIndirect);
        } else if constexpr (I == Id_vkCmdDrawIndirectCount || I == Id_vkCmdDrawIndirectCountKHR ||
                             I == Id_vkCmdDrawIndexedIndirectCount ||
                             I == Id_vkCmdDrawIndexedIndirectCountKHR) {
            F::CmdDrawIndirectCount(o, std::get<1>(t), std::get<2>(t), std::get<3>(t),
                                    std::get<4>(t), std::get<5>(t), std::get<6>(t),
                                    I == Id_vkCmdDrawIndexedIndirectCount ||
                                        I == Id_vkCmdDrawIndexedIndirectCountKHR);
        } else if constexpr (I == Id_vkCmdDispatch) {
            F::CmdDispatch(o, std::get<1>(t), std::get<2>(t), std::get<3>(t));
        } else if constexpr (I == Id_vkCmdDispatchIndirect) {
            F::CmdDispatchIndirect(o, std::get<1>(t), std::get<2>(t));
        } else if constexpr (I == Id_vkCmdCopyBuffer) {
            std::apply([&](VkCommandBuffer, auto... a) { F::CmdCopyBuffer(o, a...); }, t);
        } else if constexpr (I == Id_vkCmdFillBuffer) {
            std::apply([&](VkCommandBuffer, auto... a) { F::CmdFillBuffer(o, a...); }, t);
        } else if constexpr (I == Id_vkCmdUpdateBuffer) {
            std::apply([&](VkCommandBuffer, auto... a) { F::CmdUpdateBuffer(o, a...); }, t);
        } else if constexpr (I == Id_vkCmdCopyBufferToImage) {
            std::apply([&](VkCommandBuffer, auto... a) { F::CmdCopyBufferToImage(o, a...); }, t);
        } else if constexpr (I == Id_vkCmdCopyImageToBuffer) {
            std::apply([&](VkCommandBuffer, auto... a) { F::CmdCopyImageToBuffer(o, a...); }, t);
        } else if constexpr (I == Id_vkCmdCopyImage) {
            std::apply([&](VkCommandBuffer, auto... a) { F::CmdCopyImage(o, a...); }, t);
        } else if constexpr (I == Id_vkCmdClearColorImage) {
            std::apply([&](VkCommandBuffer, auto... a) { F::CmdClearColorImage(o, a...); }, t);
        } else if constexpr (I == Id_vkCmdClearDepthStencilImage) {
            std::apply([&](VkCommandBuffer, auto... a) { F::CmdClearDepthStencilImage(o, a...); },
                       t);
        } else if constexpr (I == Id_vkCmdBeginDebugUtilsLabelEXT ||
                             I == Id_vkCmdEndDebugUtilsLabelEXT ||
                             I == Id_vkCmdInsertDebugUtilsLabelEXT) {
            // Labels are dropped (they would have to balance in every segment).
        } else if constexpr (I == Id_vkCmdBeginQuery) {
            fallback(true);
            F::QueryBegun(o);
        } else if constexpr (I == Id_vkCmdEndQuery) {
            fallback(true);
            F::QueryEnded(o);
        } else if constexpr (I == Id_vkCmdClearAttachments || I == Id_vkCmdWriteTimestamp ||
                             I == Id_vkCmdWriteTimestamp2 || I == Id_vkCmdWriteTimestamp2KHR ||
                             I == Id_vkCmdWriteBufferMarkerAMD) {
            fallback(true); // may stand in a render pass
        } else if constexpr (I == Id_vkCmdSetVertexInputEXT) {
            static std::atomic<bool> noted{};
            if (!noted.exchange(true)) {
                Print("Metal layer: vkCmdSetVertexInputEXT in an owned command buffer is not "
                      "kept (KosmicKrisp has no dynamic vertex input; frame mode assumes it)\n");
            }
            fallback(true);
        } else {
            // Transfers, query resets and copies, secondary command buffers: outside passes.
            fallback(false);
        }
    }
#endif

    static void Install(R (*&slot)(A...)) {
        if (slot && slot != &Call) {
            original = slot;
            slot = &Call;
        }
    }
};

} // namespace

bool FrameActive() {
    return frame;
}

void InstallIfAsked(VkDevice device) {
    const char* trace_env = std::getenv("BB_METAL_TRACE");
    const char* frame_env = std::getenv("BB_METAL_FRAME");
    const bool want_frame = frame_env && frame_env[0] == '1';
    trace = trace_env && trace_env[0] == '1';
    if (!trace && !want_frame) {
        return;
    }
    static std::once_flag once;
    std::call_once(once, [&] {
        auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
        if (want_frame) {
#ifdef BB_METAL_BACKEND
            if (EmulatorSettings.IsVkValidationEnabled()) {
                Print("Metal layer: frame mode needs the Vulkan validation layers off; counting "
                      "only\n");
            } else {
                frame = MetalFrame::Init(device, d.vkGetDeviceProcAddr);
            }
#else
            (void)device;
            Print("Metal layer: BB_METAL_FRAME=1 needs the Metal command path's build (arm64 macOS "
                  "with SPIRV-Cross); counting only\n");
#endif
        }
        unsigned installed = 0;
#define BB_INSTALL(name)                                                                           \
    Hook<Id_##name, decltype(d.name)>::Install(d.name);                                            \
    if (Hook<Id_##name, decltype(d.name)>::original) {                                             \
        hooked[Id_##name] = reinterpret_cast<PFN_vkVoidFunction>(d.name);                          \
        ++installed;                                                                               \
    }
        BB_LAYER_FUNCTIONS(BB_INSTALL)
#undef BB_INSTALL
        Print("Metal layer: over %u Vulkan functions (%s)\n", installed,
              frame ? "BB_METAL_FRAME=1: command buffers encoded in Metal where they can be"
                    : "BB_METAL_TRACE=1: counting; calls go to the driver unchanged");
    });
}

} // namespace MetalLayer
