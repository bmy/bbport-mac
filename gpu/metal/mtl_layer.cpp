// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the Metal command path's command layer, stage 3a (mtl_layer.h). The hook template and
// the function list follow Wooloomooloo2/bloodborne_mac_port's mtl_vk_layer.mm (78b59f5).

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
#include <tuple>
#include <utility>
#include <vector>

#include "vulkan/vk_common.h"

namespace MetalLayer {
namespace {

using Dispatcher = std::remove_reference_t<decltype(VULKAN_HPP_DEFAULT_DISPATCHER)>;

// What the renderer records and submits (vulkan.hpp's CommandBuffer and Queue members it calls,
// with the extension aliases vulkan.hpp may have filled instead of the core names), and the
// lookups libraries make for themselves.
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
    X(vkCmdExecuteCommands) X(vkGetDeviceProcAddr) X(vkGetInstanceProcAddr)

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
    Print("Metal layer: %s looked up %s\n", name,
          covered ? "(the layer's entry)" : "(not covered: its calls bypass the layer)");
}

PFN_vkVoidFunction HookedByName(const char* name) {
    for (unsigned i = 0; i < Count; ++i) {
        if (hooked[i] && i != Id_vkGetDeviceProcAddr && i != Id_vkGetInstanceProcAddr &&
            std::strcmp(Names[i], name) == 0) {
            return hooked[i];
        }
    }
    return nullptr;
}

/// One forwarding function per dispatcher member: counts, then calls the driver.
template <unsigned I, typename F>
struct Hook;
template <unsigned I, typename R, typename... A>
struct Hook<I, R (*)(A...)> {
    static inline R (*original)(A...) = nullptr;

    static R Call(A... args) {
        calls[I].fetch_add(1, std::memory_order_relaxed);
        if constexpr (I == Id_vkQueueSubmit) {
            const auto [queue, count, submits, fence] = std::tuple{args...};
            submissions.fetch_add(1, std::memory_order_relaxed);
            for (uint32_t s = 0; s < count; ++s) {
                submitted_buffers.fetch_add(submits[s].commandBufferCount,
                                            std::memory_order_relaxed);
            }
            Report();
        } else if constexpr (I == Id_vkQueueSubmit2 || I == Id_vkQueueSubmit2KHR) {
            const auto [queue, count, submits, fence] = std::tuple{args...};
            submissions.fetch_add(1, std::memory_order_relaxed);
            for (uint32_t s = 0; s < count; ++s) {
                submitted_buffers.fetch_add(submits[s].commandBufferInfoCount,
                                            std::memory_order_relaxed);
            }
            Report();
        } else if constexpr (I == Id_vkGetDeviceProcAddr || I == Id_vkGetInstanceProcAddr) {
            const char* name = std::get<1>(std::tuple{args...});
            if (name && std::strncmp(name, "vkCmd", 5) == 0) {
                PFN_vkVoidFunction ours = HookedByName(name);
                NoteLookup(name, ours != nullptr);
                if (ours) {
                    return ours;
                }
            }
        }
        return original(args...);
    }

    static void Install(R (*&slot)(A...)) {
        if (slot && slot != &Call) {
            original = slot;
            slot = &Call;
        }
    }
};

} // namespace

void InstallIfAsked() {
    const char* env = std::getenv("BB_METAL_TRACE");
    if (!env || env[0] != '1') {
        return;
    }
    static std::once_flag once;
    std::call_once(once, [] {
        auto& d = VULKAN_HPP_DEFAULT_DISPATCHER;
        unsigned installed = 0;
#define BB_INSTALL(name)                                                                           \
    Hook<Id_##name, decltype(d.name)>::Install(d.name);                                            \
    if (Hook<Id_##name, decltype(d.name)>::original) {                                             \
        hooked[Id_##name] = reinterpret_cast<PFN_vkVoidFunction>(d.name);                          \
        ++installed;                                                                               \
    }
        BB_LAYER_FUNCTIONS(BB_INSTALL)
#undef BB_INSTALL
        Print("Metal layer: counting %u Vulkan functions (BB_METAL_TRACE=1; calls go to the driver "
              "unchanged)\n",
              installed);
    });
}

} // namespace MetalLayer
