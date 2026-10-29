// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "shim/bb_env.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include "render/vk_breadcrumbs.h"
#include "vulkan/instance.h"

extern "C" int runtime_memory_vma_info(uintptr_t address, int* prot, int* type, uintptr_t* end);

#include <vk_mem_alloc.h>

namespace Vulkan::Breadcrumbs {

namespace {

constexpr u32 MaxStreams = 16;
constexpr u32 StreamStride = 64; // bytes of the marker buffer per stream (a cache line)
constexpr u64 RingSize = 1u << 15;
constexpr u32 ArgSlots = 4096;                     // indirect dispatch arguments, 16 bytes each
constexpr u64 ArgsOffset = MaxStreams * StreamStride; // where they start in the buffer

struct Entry {
    u64 id;
    Crumb crumb;
};

struct Stream {
    const char* name = nullptr;
    std::atomic<u64> next{1};
    std::unique_ptr<Entry[]> ring;
};

std::array<Stream, MaxStreams> streams;
std::atomic<u32> num_streams{0};
bool enabled = false;
std::once_flag init_once;
vk::Buffer marker_buffer{};
VmaAllocation marker_allocation{};
volatile u32* markers = nullptr;
volatile u32* args = nullptr;
std::atomic<u32> next_args_slot{0};
/// What each argument slot was last used for.
struct ArgsNote {
    u64 id;
    u32 stream;
    u64 hash;
};
std::array<ArgsNote, ArgSlots> args_notes{};

const Instance* fault_instance = nullptr;

void Init(const Instance& instance) {
    if (instance.IsDeviceFaultSupported()) {
        fault_instance = &instance;
    }
#ifdef __APPLE__
    // bbport (macOS): off unless BB_BREADCRUMBS=1. KosmicKrisp writes each marker with a
    // compute dispatch of its own (hundreds of thousands a minute), queued to the end of every
    // render pass.
    if (!bb_env_on("BB_BREADCRUMBS")) {
        return;
    }
#endif
    if (bb_env_off("BB_BREADCRUMBS")) {
        return;
    }
    if (!instance.IsBufferMarkerSupported()) {
        std::printf("GPU breadcrumbs: VK_AMD_buffer_marker unavailable, GPU hangs will not be "
                    "located\n");
        return;
    }
    const VkBufferCreateInfo buffer_ci = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = ArgsOffset + ArgSlots * 16,
        .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    const VmaAllocationCreateInfo alloc_ci = {
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    };
    VkBuffer buffer{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(instance.GetAllocator(), &buffer_ci, &alloc_ci, &buffer,
                        &marker_allocation, &info) != VK_SUCCESS ||
        !info.pMappedData) {
        std::printf("GPU breadcrumbs: no host-visible buffer, GPU hangs will not be located\n");
        return;
    }
    marker_buffer = vk::Buffer{buffer};
    markers = static_cast<volatile u32*>(info.pMappedData);
    for (u32 i = 0; i < (ArgsOffset + ArgSlots * 16) / sizeof(u32); ++i) {
        markers[i] = 0;
    }
    args = markers + ArgsOffset / sizeof(u32);
    enabled = true;
}

/// Guest memory that may be gone by now: read only where the runtime has it mapped (its direct
/// memory is a GPU driver mapping, which process_vm_readv cannot read).
size_t ReadGuest(u64 address, void* out, size_t size) {
    int prot = 0, type = 0;
    uintptr_t end = 0;
    if (!address || !runtime_memory_vma_info(address, &prot, &type, &end) || end <= address) {
        return 0;
    }
    size = std::min<size_t>(size, end - address);
    std::memcpy(out, reinterpret_cast<const void*>(address), size);
    return size;
}

const char* KindName(Kind kind) {
    switch (kind) {
    case Kind::Pass:
        return "pass";
    case Kind::Draw:
        return "draw";
    case Kind::DrawIndexed:
        return "indexed draw";
    case Kind::DrawIndirect:
        return "indirect draw";
    case Kind::Dispatch:
        return "dispatch";
    case Kind::DispatchIndirect:
        return "indirect dispatch";
    }
    return "?";
}

void PrintEntry(const Stream& stream, u64 id, const char* label) {
    const Entry& entry = stream.ring[id & (RingSize - 1)];
    if (entry.id != id) {
        std::printf("    #%llu %s: no longer kept\n", (unsigned long long)id, label);
        return;
    }
    const Crumb& c = entry.crumb;
    std::printf("    #%llu %s: %s", (unsigned long long)id, label, KindName(c.kind));
    switch (c.kind) {
    case Kind::Pass:
        std::printf(" %s", c.name ? c.name : "?");
        break;
    case Kind::Draw:
    case Kind::DrawIndexed:
        std::printf(" vs %016llx ps %016llx, %u %s x %u instances", (unsigned long long)c.hash[0],
                    (unsigned long long)c.hash[1], c.count[0],
                    c.kind == Kind::Draw ? "vertices" : "indices", c.count[1]);
        break;
    case Kind::DrawIndirect: {
        std::printf(" vs %016llx ps %016llx, up to %u draws, stride %u, arguments at %#llx",
                    (unsigned long long)c.hash[0], (unsigned long long)c.hash[1], c.count[0],
                    c.count[1], (unsigned long long)c.address);
        u32 now[5]{};
        if (ReadGuest(c.address, now, sizeof(now)) == sizeof(now)) {
            std::printf(" (guest memory now %u %u %u %u %u)", now[0], now[1], now[2], now[3], now[4]);
        }
        break;
    }
    case Kind::Dispatch:
        std::printf(" cs %016llx, %ux%ux%u groups", (unsigned long long)c.hash[0], c.count[0],
                    c.count[1], c.count[2]);
        break;
    case Kind::DispatchIndirect: {
        std::printf(" cs %016llx, arguments at %#llx", (unsigned long long)c.hash[0],
                    (unsigned long long)c.address);
        if (c.args_slot < ArgSlots) {
            const volatile u32* read = args + c.args_slot * 4;
            std::printf(" (the GPU read %ux%ux%u groups)", read[0], read[1], read[2]);
        }
        std::printf(c.count[0] ? ", in place" : ", a copy");
        u32 now[3]{};
        if (ReadGuest(c.address, now, sizeof(now)) == sizeof(now)) {
            std::printf(" (guest memory now %ux%ux%u)", now[0], now[1], now[2]);
        }
        break;
    }
    }
    for (u32 i = 0; i < 2; ++i) {
        if (c.program[i]) {
            std::printf(", code %#llx", (unsigned long long)c.program[i]);
        }
    }
    std::printf("\n");
    // What a stuck draw reads, and whether the runtime still has it mapped (prot, memory type).
    if (std::strcmp(label, "STUCK") == 0 && c.reads[0]) {
        std::printf("      reads:");
        for (u32 i = 0; i < Crumb::MaxReads && c.reads[i]; ++i) {
            int prot = 0, type = 0;
            uintptr_t end = 0;
            const bool mapped = runtime_memory_vma_info(c.reads[i], &prot, &type, &end) && end > c.reads[i];
            std::printf(" %c %#llx", c.read_kinds[i], (unsigned long long)c.reads[i]);
            if (c.read_sizes[i]) {
                std::printf("+%#x", c.read_sizes[i]);
            }
            if (mapped) {
                std::printf(" (prot %d type %d%s)", prot, type,
                            c.read_sizes[i] && end < c.reads[i] + c.read_sizes[i] ? ", mapping ends inside" : "");
            } else {
                std::printf(" (not mapped)");
            }
            std::printf(";");
        }
        std::printf("\n");
    }
}

/// The guest code of the shaders of the stuck command, for disassembly (up to 64 KiB each:
/// the program and what follows, its OrbShdr header among it).
void DumpPrograms(const Stream& stream, u64 id) {
    const Entry& entry = stream.ring[id & (RingSize - 1)];
    if (entry.id != id) {
        return;
    }
    const char* dir = access("out", W_OK) == 0 ? "out/" : "";
    for (u32 i = 0; i < 2; ++i) {
        const u64 program = entry.crumb.program[i];
        if (!program) {
            continue;
        }
        std::array<u8, 64 * 1024> code{};
        size_t size = 0;
        // Page by page: the read stops at the first page that is not mapped.
        while (size < code.size()) {
            const size_t chunk = std::min<size_t>(4096 - ((program + size) & 4095), code.size() - size);
            const size_t read = ReadGuest(program + size, code.data() + size, chunk);
            size += read;
            if (read != chunk) {
                break;
            }
        }
        char path[256];
        std::snprintf(path, sizeof(path), "%shang_shader_%016llx.bin", dir,
                      (unsigned long long)entry.crumb.hash[i]);
        if (FILE* file = std::fopen(path, "wb")) {
            std::fwrite(code.data(), 1, size, file);
            std::fclose(file);
            std::printf("    shader %016llx code (%zu bytes from %#llx) saved to %s\n",
                        (unsigned long long)entry.crumb.hash[i], size,
                        (unsigned long long)program, path);
        }
    }
}

} // Anonymous namespace

u32 NewStream(const Instance& instance, const char* name) {
    std::call_once(init_once, [&] { Init(instance); });
    const u32 index = num_streams.fetch_add(1, std::memory_order_relaxed);
    if (index >= MaxStreams) {
        return MaxStreams;
    }
    streams[index].name = name;
    streams[index].ring = std::make_unique<Entry[]>(RingSize);
    return index;
}

bool Enabled() noexcept {
    return enabled;
}

u32 Note(u32 stream_index, const Crumb& crumb) {
    if (stream_index >= MaxStreams) {
        return 0;
    }
    Stream& stream = streams[stream_index];
    const u64 id = stream.next.fetch_add(1, std::memory_order_relaxed);
    Entry& entry = stream.ring[id & (RingSize - 1)];
    entry.crumb = crumb;
    entry.id = id;
    if (crumb.args_slot < ArgSlots) {
        args_notes[crumb.args_slot] = {id, stream_index, crumb.hash[0]};
    }
    return u32(id);
}

void Mark(vk::CommandBuffer cmdbuf, u32 stream, u32 id, bool after) {
    if (!enabled || stream >= MaxStreams) {
        return;
    }
    cmdbuf.writeBufferMarkerAMD(vk::PipelineStageFlagBits::eBottomOfPipe, marker_buffer,
                                stream * StreamStride, id * 2 + (after ? 1 : 0));
}

u32 ArgsSlot() {
    return enabled ? next_args_slot.fetch_add(1, std::memory_order_relaxed) % ArgSlots : ~0u;
}

void CopyArgs(vk::CommandBuffer cmdbuf, vk::Buffer source, u64 offset, u32 slot) {
    if (!enabled || slot >= ArgSlots) {
        return;
    }
    cmdbuf.copyBuffer(source, marker_buffer,
                      vk::BufferCopy{offset, ArgsOffset + u64(slot) * 16, 3 * sizeof(u32)});
}

namespace {
void Report(const char* title, const char* where);

/// VK_EXT_device_fault: the driver's own account of the fault (NVIDIA: the faulting address and
/// a description), once per device lost.
void PrintDeviceFault() {
    if (!fault_instance) {
        return;
    }
    const VkDevice device = fault_instance->GetDevice();
    const auto get = reinterpret_cast<PFN_vkGetDeviceFaultInfoEXT>(
        VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr(device, "vkGetDeviceFaultInfoEXT"));
    if (!get) {
        return;
    }
    VkDeviceFaultCountsEXT counts{.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    const VkResult counted = get(device, &counts, nullptr);
    std::vector<VkDeviceFaultAddressInfoEXT> addresses(counts.addressInfoCount);
    std::vector<VkDeviceFaultVendorInfoEXT> vendor(counts.vendorInfoCount);
    counts.vendorBinarySize = 0;
    VkDeviceFaultInfoEXT info{.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT,
                              .pAddressInfos = addresses.data(),
                              .pVendorInfos = vendor.data()};
    const VkResult result = get(device, &counts, &info);
    std::printf("GPU fault info (%d/%d): %s\n", int(counted), int(result), info.description);
    static constexpr const char* kinds[] = {"none",
                                            "read invalid",
                                            "write invalid",
                                            "execute invalid",
                                            "instruction pointer unknown",
                                            "instruction pointer invalid",
                                            "instruction pointer fault"};
    for (u32 i = 0; i < counts.addressInfoCount; ++i) {
        const auto& a = addresses[i];
        const u32 kind = u32(a.addressType);
        std::printf("  address %s: 0x%llx (+-0x%llx)\n", kind < 7 ? kinds[kind] : "?",
                    (unsigned long long)a.reportedAddress, (unsigned long long)a.addressPrecision);
    }
    for (u32 i = 0; i < counts.vendorInfoCount; ++i) {
        const auto& v = vendor[i];
        std::printf("  vendor: %s (code 0x%llx, data 0x%llx)\n", v.description,
                    (unsigned long long)v.vendorFaultCode, (unsigned long long)v.vendorFaultData);
    }
    std::fflush(stdout);
}
} // namespace

void ReportDeviceLost(const char* where) {
    static std::atomic_flag reported = ATOMIC_FLAG_INIT;
    static std::atomic<bool> done{false};
    if (!enabled && !fault_instance) {
        return;
    }
    if (!reported.test_and_set()) {
        PrintDeviceFault();
        if (enabled) {
            Report("at device lost", where);
        }
        done.store(true);
        return;
    }
    // Another thread is printing the report: wait for it (up to 3 s), as the caller stops the
    // game next and the report was cut off (laptop, TEX FORMAT device losts).
    for (int i = 0; i < 300 && !done.load(); ++i) {
        usleep(10000);
    }
}

void ReportStuck(const char* where) {
    static std::atomic_flag reported = ATOMIC_FLAG_INIT;
    if (enabled && !reported.test_and_set()) {
        Report("while the GPU makes no progress", where);
    }
}

namespace {
void Report(const char* title, const char* where) {
    std::printf("GPU breadcrumbs %s (%s): what each command stream finished\n", title, where);
    const u32 count = std::min(num_streams.load(), MaxStreams);
    for (u32 s = 0; s < count; ++s) {
        const Stream& stream = streams[s];
        const u64 last = stream.next.load() - 1;
        const u32 marker = markers[s * StreamStride / sizeof(u32)];
        std::printf("  stream %u (%s): %llu commands noted", s, stream.name,
                    (unsigned long long)last);
        if (last == 0) {
            std::printf("\n");
            continue;
        }
        if (marker == 0) {
            std::printf(", the GPU finished none\n");
            continue;
        }
        // The marker holds the low 32 bits of 2 * id + after.
        const u64 newest = last * 2 + 1;
        const u64 value = newest - u32(u32(newest) - marker);
        const u64 id = value / 2;
        if (value & 1) {
            if (id == last) {
                std::printf(", the GPU finished all\n");
                continue;
            }
            std::printf(", the GPU finished #%llu but did not start #%llu: stuck in the work "
                        "recorded between them (copies, clears, barriers) or before it\n",
                        (unsigned long long)id, (unsigned long long)(id + 1));
            PrintEntry(stream, id, "finished");
            PrintEntry(stream, id + 1, "next");
        } else {
            std::printf(", the GPU started #%llu and did not finish it (%llu noted after it)\n",
                        (unsigned long long)id, (unsigned long long)(last - id));
            if (id > 1) {
                PrintEntry(stream, id - 1, "finished");
            }
            PrintEntry(stream, id, "STUCK");
            std::fflush(stdout);
            DumpPrograms(stream, id);
        }
        for (u64 next = id + 2; next <= std::min(last, id + 4); ++next) {
            PrintEntry(stream, next, "later");
        }
    }
    std::fflush(stdout);
}
} // Anonymous namespace


void PrintProgress() {
    if (!enabled) {
        return;
    }
    std::printf("GPU breadcrumbs:");
    const u32 count = std::min(num_streams.load(), MaxStreams);
    std::array<u64, MaxStreams> finished{};
    for (u32 s = 0; s < count; ++s) {
        const u64 last = streams[s].next.load(std::memory_order_relaxed) - 1;
        if (last == 0) {
            continue;
        }
        const u32 marker = markers[s * StreamStride / sizeof(u32)];
        const u64 newest = last * 2 + 1;
        const u64 value = marker ? newest - u32(u32(newest) - marker) : 0;
        // 2n + 1: n finished; 2n: n started, n - 1 finished.
        finished[s] = value ? value / 2 - (value & 1 ? 0 : 1) : 0;
        std::printf(" stream %u (%s) %llu noted, %llu unfinished;", s, streams[s].name,
                    (unsigned long long)last, (unsigned long long)(last - finished[s]));
    }
    // The largest indirect dispatches among the last ArgSlots the GPU finished, one per shader:
    // counts that only grow from frame to frame end in a dispatch that never finishes.
    struct Largest {
        u64 hash;
        u64 groups;
        u32 dims[3];
    };
    std::array<Largest, 3> largest{};
    for (u32 slot = 0; slot < ArgSlots; ++slot) {
        const ArgsNote note = args_notes[slot];
        if (note.id == 0 || note.stream >= count || note.id > finished[note.stream]) {
            continue;
        }
        const u32 x = args[slot * 4], y = args[slot * 4 + 1], z = args[slot * 4 + 2];
        const u64 groups = u64(x) * y * z;
        auto it = std::find_if(largest.begin(), largest.end(),
                               [&](const Largest& l) { return l.hash == note.hash; });
        if (it == largest.end()) {
            it = std::min_element(largest.begin(), largest.end(),
                                  [](const Largest& a, const Largest& b) { return a.groups < b.groups; });
            if (it->groups >= groups) {
                continue;
            }
            *it = {note.hash, 0, {}};
        }
        if (groups >= it->groups) {
            *it = {note.hash, groups, {x, y, z}};
        }
    }
    std::sort(largest.begin(), largest.end(),
              [](const Largest& a, const Largest& b) { return a.groups > b.groups; });
    std::printf(" largest indirect dispatches:");
    for (const auto& l : largest) {
        if (l.groups != 0) {
            std::printf(" cs %016llx %ux%ux%u;", (unsigned long long)l.hash, l.dims[0], l.dims[1],
                        l.dims[2]);
        }
    }
    std::printf("\n");
}

} // namespace Vulkan::Breadcrumbs
