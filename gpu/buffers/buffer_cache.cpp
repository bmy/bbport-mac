// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include "shim/bb_env.h"
#include <array>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <bit>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <magic_enum/magic_enum.hpp>
#include "shader/profile.h"
#include "bbport_copy.h"
#include "bbport_toggles.h"
#include "bbport_sections.h"
#include "bbport_free_check.h"
#include "bbport_guest_memory.h"
#include "game_profile.h"
#include "bblayer_gpu_memory.h"
#include "bblayer_write_traps.h"
#include "common/alignment.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "cp/command_processor.h"
#include "buffers/buffer.h"
#include "buffers/buffer_cache.h"
#include "memory/dirty_tracker.h"
#include "render/pipeline.h"
#include "vulkan/instance.h"
#include "vulkan/platform.h"
#include "render/runtime.h"
#include "render/scheduler.h"
#include "textures/texture_cache.h"
#include "buffers/buffer_cache_internal.h"

#include <mutex>
#include <pthread.h>
#include "bbport_cpu.h"
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <fmt/format.h>
#include <sys/mman.h>
#ifdef __APPLE__
#include <sys/ucontext.h> // <ucontext.h> requires _XOPEN_SOURCE on macOS
#else
#include <ucontext.h>
#endif
#include "common/signal_context.h"
#include "core/signals.h"
#include "remote/bb_remote.h"
#include <vk_mem_alloc.h>

/// The layer's memory module asks the runtime where direct memory is mapped.
static bool LayerTranslate(uint64_t va, uint64_t* phys, uint64_t* end) {
    uintptr_t map_end = 0;
    if (!runtime_memory_direct_phys(va, phys, &map_end)) {
        return false;
    }
    *end = map_end;
    return true;
}

namespace VideoCore {

/// On an integrated GPU (Steam Deck) VRAM is the same memory: everything stays in place there.
bool integrated_gpu = false;
/// bbport: Garlic memory (type 3) is cached in VRAM, Onion in place. Optimizations, for A/B: VRAM copies uploaded by GPU copies from the
/// guest chunks (BB_CHUNK_UPLOADS=0: CPU staging), texture data bound in place whatever its type
/// (BB_INPLACE_TEXTURES=1; off: by type).
bool ChunkUploads() {
    static const bool on = !bb_env_off("BB_CHUNK_UPLOADS");
    return on;
}
bool ChunkTextures() {
    static const bool on = !bb_env_off("BB_CHUNK_TEXTURES");
    return on;
}
bool InPlaceTextures() {
    static const bool on = [] {
        // Off by default: sampled data looked smeared and ghosted with it (2026-10-04).
        return bb_env_on("BB_INPLACE_TEXTURES");
    }();
    return on;
}

/// bbport: the arena keeps the first binding of every block while the game runs (imported host
/// memory, NVIDIA): game data is read in place over the bus, never moved to VRAM and back, and
/// the few blocks outside the guest chunks get system memory, not VRAM. On NVIDIA (GTX 1660 Ti,
/// 615.71) the first blocks moved to VRAM held the GPU for 10 s and more, the desktop with it.
/// BB_FIXED_ARENA=1/0 forces it on/off (tests). Valid once the BufferCache checked the chunks.
bool FixedArena() {
    // Only sparse rebinding needs this NVIDIA workaround. The layer has ordinary chunk
    // buffers and VRAM mirrors, so fixing an arena that it does not use would disable all
    // its promotions on host-import GPUs (GTX 1660 Ti: zero mirrors and 10-15 FPS).
    if (BbGuestMemory::LayerMemory()) {
        return false;
    }
    static const bool on = [] {
        const char* env = bb_env("BB_FIXED_ARENA");
        const bool fixed = env && *env ? env[0] != '0' : BbGuestMemory::HostImported();
        if (fixed) {
            std::printf("Guest memory: the arena keeps its first bindings (game data read in place, "
                        "no copies in VRAM)\n");
        }
        return fixed;
    }();
    return on;
}

bool MixStats() {
    static const bool on = bb_env_on("BB_RESIDENCY_MIX");
    return on;
}

/// bbport BB_LAYER_MEMORY=1: the layer's memory module (gpu/layer) binds the game's memory: a
/// range in place is a range of its chunk's buffer, with separate VRAM mirrors and no sparse
/// arena. FixedArena must not prohibit those mirrors.
bool LayerMode() {
    return BbGuestMemory::LayerMemory();
}

bool GarlicInVram() {
    static const bool on = [] {
        const char* env = bb_env("BB_GARLIC_VRAM");
        return env ? env[0] != '0' : !integrated_gpu;
    }();
    return on && !FixedArena();
}

static constexpr size_t GDS_BUFFER_SIZE = 64_KB;
/// The stream ring: 128 MiB, 32 on GPUs with 3 GiB of VRAM or less (a frame streams about
/// 0.4 MiB, BB_STREAM_STATS); BB_STREAM_BUFFER_MB another size.
static size_t StreamBufferSize(const Vulkan::Instance& instance) {
    const char* env = bb_env("BB_STREAM_BUFFER_MB");
    const u64 mb = env ? std::strtoull(env, nullptr, 10) : 0;
    const bool small = !instance.IsIntegrated() && instance.GetTotalMemoryBudget() <= (3ull << 30);
    return (mb ? mb : small ? 32ull : 128ull) << 20;
}

static constexpr auto ARENA_USAGE =
    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;

std::optional<u32> FindMemoryType(const vk::PhysicalDeviceMemoryProperties& properties,
                                  vk::MemoryPropertyFlags wanted, u32 memory_type_bits) {
    for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
        if (((memory_type_bits >> i) & 1) == 0) {
            continue;
        }
        const auto flags = properties.memoryTypes[i].propertyFlags;
        if ((flags & wanted) == wanted) {
            return i;
        }
    }
    return std::nullopt;
}

BufferCache::BufferCache(const Vulkan::Instance& instance_, Vulkan::Scheduler& scheduler_,
                         Vulkan::Runtime& runtime_, BbCp::CommandProcessor* liverpool_,
                         TextureCache& texture_cache_, BbMem::PageWatch& tracker)
    : instance{instance_}, scheduler{scheduler_}, runtime{runtime_},
      staging_pool{runtime_.GetStagingPool()}, liverpool{liverpool_},
      memory{Core::Memory::Instance()}, texture_cache{texture_cache_},
      memory_tracker{std::make_unique<BbMem::DirtyTracker>(tracker)},
      stream_buffer{instance, scheduler, MemoryType::Stream, StreamBufferSize(instance)},
      gds_buffer{instance, 0, GDS_BUFFER_SIZE, MemoryType::Stream, "GDS Buffer"},
      memory_semaphore{instance} {
    integrated_gpu = instance.IsIntegrated();
    // bbport: the memory model needs the game's direct memory in chunks the GPU can use in place
    // (dma-buf export or host memory import: BbGuestMemory::Usable).
    if (!BbGuestMemory::MemoryModuleFits(instance) || !BbGuestMemory::Usable(instance)) {
        GuestMemoryUnusable("neither a dma-buf of system memory nor imported host memory passed "
                            "the startup checks (see the \"Guest memory:\" lines above)");
    }
    // Hooks in the game's own code, from its profile (games/). BB_GUEST_HOOKS=0: none, as for a
    // game the port knows nothing about.
    const Game::Profile* game = Game::Active();
    if (game && game->install_hooks && BbRemote::BackActive()) {
        // bbport (macOS native GPU process): the game's code is in the game process, which
        // installs them (bbgpu_init there).
    } else if (game && game->install_hooks && !bb_env_off("BB_GUEST_HOOKS")) {
        static BufferCache* hooked_cache = this;
        game->install_hooks(Game::CoreServices{
            .fresh_range = +[](u64 address, u64 size) {
                hooked_cache->NoteFreshRange(address, size);
            },
            .dead_range = +[](u64 address, u64 size) {
                hooked_cache->LayerDomainDead(address, size); // queued: the GPU thread forgets it
            },
            .live_range = +[](u64 address, u64 size) { hooked_cache->LayerDomainAdd(address, size); },
        });
    } else if (game && game->install_hooks) {
        std::printf("Guest hooks: off (BB_GUEST_HOOKS=0): nothing in the game's own code is hooked\n");
    }
    if (LayerMode()) {
        BbLayer::GpuMemory::Get().SetTranslate(&LayerTranslate);
        std::printf("Guest memory: the layer's memory module binds the game's memory in place "
                    "(chunk buffers, no sparse arena for it; BB_LAYER_MEMORY=1)\n");
    }
    const std::array<u32, 2> families = {instance.GetGraphicsQueueFamilyIndex(),
                                         instance.GetReadbackQueueFamilyIndex()};
    const bool shared = bool(instance.GetReadbackQueue()); // as Buffer creates the arenas
    const vk::BufferCreateInfo probe_ci = {
        .flags =
            vk::BufferCreateFlagBits::eSparseBinding | vk::BufferCreateFlagBits::eSparseResidency,
        .size = ARENA_PAGE_SIZE,
        .usage = ARENA_USAGE,
        .sharingMode = shared ? vk::SharingMode::eConcurrent : vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = shared ? 2u : 0u,
        .pQueueFamilyIndices = shared ? families.data() : nullptr,
    };
    const vk::DeviceBufferMemoryRequirements req_info = {
        .pCreateInfo = &probe_ci,
    };
    const auto device = instance.GetDevice();
    const auto reqs = device.getBufferMemoryRequirements(req_info).memoryRequirements;
    block_size = Common::AlignUp(std::max<u64>(reqs.alignment, MIN_BLOCK_SIZE), reqs.alignment);
    ASSERT_MSG(std::popcount(block_size) == 1, "Sparse block size {} is not a power of 2",
               block_size);
    block_shift = std::bit_width(block_size) - 1;
    blocks_per_arena_page = ARENA_PAGE_SIZE / block_size;
    blocks_per_arena_page_shift = ARENA_PAGE_BITS - block_shift;
    BbLayer::GpuMemory::Get().SetBlockShift(static_cast<u32>(block_shift));
    group_use.assign(u64{1} << (ADDRESS_SPACE_BITS - USE_GROUP_BITS), 0);
    group_demoted.assign(group_use.size(), 0);
    layer_group_use.assign(group_use.size(), 0);
    arena_memory_type_index =
        FindMemoryType(instance.GetMemoryProperties(), vk::MemoryPropertyFlagBits::eDeviceLocal,
                       reqs.memoryTypeBits)
            .value();
    arena_memory_type_bits = reqs.memoryTypeBits;
    // BB_ARENA_VRAM_ONLY=1 (tests): the arena takes VRAM only, as NVIDIA's sparse buffers do
    // (with BB_RESIDENCY_VRAM_LIMIT_MB: VRAM running out on such a card).
    if (bb_env_on("BB_ARENA_VRAM_ONLY")) {
        arena_memory_type_bits = 1u << arena_memory_type_index;
    } else {
        const auto& properties = instance.GetMemoryProperties();
        for (u32 i = 0; i < properties.memoryTypeCount; ++i) {
            const auto flags = properties.memoryTypes[i].propertyFlags;
            if (((reqs.memoryTypeBits >> i) & 1) && !(flags & vk::MemoryPropertyFlagBits::eDeviceLocal) &&
                (flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
                arena_fallback_type_index = i;
                break;
            }
        }
    }

    // The window of guest addresses the table covers (Shader::PageTableBase, PageTableSpan).
    const u64 bda_pagetable_size = (Shader::PageTableSpan >> block_shift) * sizeof(vk::DeviceAddress) *
                                   (LayerActive() ? 2 : 1);
    fault_manager = std::make_unique<FaultBuffer>(instance, scheduler, *this, block_shift,
                                                   blocks_per_arena_page * NUM_ARENA_PAGES);
    bda_pagetable_buffer = std::make_unique<Buffer>(
        instance, 0, bda_pagetable_size, MemoryType::DeviceLocal, "BDA Page Table Buffer");
    runtime.FillBuffer(bda_pagetable_buffer.get(), 0u, bda_pagetable_size, 0u);
}

BufferCache::~BufferCache() = default;

void BufferCache::InvalidateMemory(VAddr device_addr, u64 size, bool assume_locks) {
    if (LayerMode()) {
        LayerNoteCpuWrite(device_addr, size);
    }
    memory_tracker->InvalidateRegion(device_addr, size, [this, device_addr, size, assume_locks] {
        ReadMemory(device_addr, size, true, assume_locks);
    });
}

// bbport: the game fills its per-frame buffers (constants, skinning output) sequentially and
// every 4 KiB page cost a protection fault (~110k/s in Hunter's Nightmare, a fifth of each
// render worker's time in the kernel). A fault unprotects the aligned window around it instead;
// pages marked CPU-modified without being written only cost an upload when bound.
// Item: context = BufferCache, source = guest address, destination = host pointer, size,
// extra = the Buffer whose mapping holds the destination (flushed after the copy).
void BufferCache::RunGuestCopy(const BbCopy::Item& item) {
    auto* cache = static_cast<BufferCache*>(item.context);
    auto* dst = reinterpret_cast<u8*>(item.destination);
    cache->memory->CopySparseMemory(item.source, dst, item.size);
    auto* buffer = reinterpret_cast<Buffer*>(item.extra);
    buffer->Flush(dst - buffer->mapped_data.data(), item.size);
}

// Small guest copies run on the recording thread (it spins for work: no wakeup, and it is
// idle most of the time); PoolSmallCopies (toggle 524288) batches them for the copy threads.
// Each source is noted once the copy is queued (a WaitHostCopies started after the note covers it).
void BufferCache::SmallGuestCopy(const BbCopy::Item& item) {
    if (!scheduler.IsRecordingDeferred()) {
        // bbport: commands recorded on this thread (direct mode: the draw or dispatch after the
        // upscaler's passes; or BB_VK_RECORD_THREAD=0): the copy runs now, before the commands
        // that read it. In this thread's copy batch it ran at the next submission at the
        // earliest, possibly after a WRITE_DATA had changed its source (that wait runs only the
        // GpuComm thread's own batch). Suspected in the vertex explosions with
        // BB_VK_RECORD_THREAD=0 (issue #39).
        item.run(item);
    } else if (!BbToggle::Disabled(BbToggle::PoolSmallCopies)) {
        scheduler.RecordHostCopy([item] { item.run(item); });
    } else {
        BbCopy::QueueCopy(item);
    }
    scheduler.NoteHostCopySource(item.source, item.size);
}

void BufferCache::ExtendWriteFault(VAddr device_addr, u64 guest_rip) {
    NoteCpuWrite(device_addr, guest_rip);
    static const u64 window = [] {
        const char* env = bb_env("BB_FAULT_WINDOW");
        const u64 kib = env ? std::strtoull(env, nullptr, 10) : 256;
        return std::bit_ceil(std::clamp<u64>(kib, 4, 1024)) * 1024;
    }();
    if (window <= BbMem::PageSize || BbToggle::Disabled(BbToggle::FaultWindow)) {
        return;
    }
    memory_tracker->ExtendWriteFault(Common::AlignDown(device_addr, window), window);
}

namespace {
// bbport: BB_BUFFER_STATS=1 — how buffer bindings reach the GPU, by guest region (256 MiB),
// printed every 5 s: small read-only copies into the stream buffer, arena bindings, and the
// bytes those re-upload after CPU writes. Input for engine-level short paths.
struct BufferStats {
    struct Region {
        u64 stream_count{}, stream_bytes{}, arena_count{}, arena_bytes{}, upload_bytes{};
    };
    std::map<u64, Region> regions;
    /// Hot window: frame of the last binding per 64 KiB block, and the distribution of the
    /// frames between uses (1, 2, 3, 4, 5-8, 9-16, 17+): the ring's reuse distance.
    std::unordered_map<u64, u64> last_use;
    std::array<u64, 7> reuse{};
    std::chrono::steady_clock::time_point window = std::chrono::steady_clock::now();
    u64 calls = 0;
};
/// 256 MiB regions; 1 MiB inside the hot 0x104xxxxxxx window (the engine's frame data).
BufferStats& Stats();
void NoteHotUse(VAddr address, u32 size) {
    if ((address >> 28) != 0x104) {
        return;
    }
    auto& st = Stats();
    const u64 frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    for (u64 block = address >> 16; block <= (address + size - 1) >> 16; ++block) {
        auto [it, inserted] = st.last_use.try_emplace(block, frame);
        if (!inserted && it->second != frame) {
            const u64 d = frame - it->second;
            ++st.reuse[d <= 4 ? d - 1 : d <= 8 ? 4 : d <= 16 ? 5 : 6];
            it->second = frame;
        }
    }
}
u64 RegionKey(VAddr address) {
    return (address >> 28) == 0x104 ? (address >> 20) | (1ull << 40) : address >> 28;
}
bool BufferStatsEnabled() {
    static const bool enabled = bb_env_on("BB_BUFFER_STATS");
    return enabled;
}
BufferStats& Stats() {
    static BufferStats stats;
    return stats;
}
void PrintBufferStats() {
    auto& st = Stats();
    if ((++st.calls & 4095) != 0) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(now - st.window).count();
    if (seconds < 5.0) {
        return;
    }
    std::printf("Buffer stats (%.1f s, per second): region, stream copies/bytes, arena bindings, "
                "re-uploaded bytes\n", seconds);
    std::printf("  hot window reuse distance in frames (1,2,3,4,5-8,9-16,17+): %llu %llu %llu %llu "
                "%llu %llu %llu\n",
                (unsigned long long)st.reuse[0], (unsigned long long)st.reuse[1],
                (unsigned long long)st.reuse[2], (unsigned long long)st.reuse[3],
                (unsigned long long)st.reuse[4], (unsigned long long)st.reuse[5],
                (unsigned long long)st.reuse[6]);
    st.reuse = {};
    for (const auto& [region, r] : st.regions) {
        const u64 base = (region >> 40) ? (region & ((1ull << 40) - 1)) << 20 : region << 28;
        if ((region >> 40) && r.stream_bytes + r.upload_bytes < 1e6 * seconds) {
            continue; // quiet MiB of the hot window
        }
        std::printf("  %#012llx: %8.0f copies %8.2f MB, %8.0f arena %8.2f MB bound, %8.2f MB uploads\n",
                    static_cast<unsigned long long>(base), r.stream_count / seconds,
                    r.stream_bytes / seconds / 1e6, r.arena_count / seconds,
                    r.arena_bytes / seconds / 1e6,
                    r.upload_bytes / seconds / 1e6);
    }
    st.regions.clear();
    st.window = now;
}
} // namespace

std::pair<const Buffer*, u64> BufferCache::ObtainBufferImpl(VAddr device_addr, u32 size,
                                                            bool is_written, bool is_texel_buffer) {
    BB_SECTION(ObtainBuffer);
    const bool stats = BufferStatsEnabled();
    if (stats) {
        PrintBufferStats();
        NoteHotUse(device_addr, size);
    }
    // bbport: buffers in the game's direct memory are used where they are. Small
    // read-only ones (constants) are still copied when bound (the stream path below): commands
    // decoded later (WriteData, constant RAM dumps) change them on the CPU in decode order, and
    // a draw decoded before must keep its values (buffer renaming). GPU writes are noted for
    // that: a range the GPU has yet to write is not copied.
    if (is_written && (!shadows.empty() || !write_shadows.empty())) {
        if (precise_writes && force_writes_in_place) {
            // The copy shader: it writes only the ranges its list names, not the whole
            // destination it binds (dropping by the binding wiped the copies of the dispatches
            // before it, back to back, before their draws read them).
            for (const auto& [addr, len] : *precise_writes) {
                DropShadows(addr, len);
            }
        } else {
            DropShadows(device_addr, size);
        }
    }
    // bbport: a read-only binding of blocks that were resident and in place the last time, with
    // the residency unchanged since (most bindings in place, constants with
    // BB_CONSTANTS_IN_PLACE): the arena where it is, without the residency walk.
    struct InPlaceMemo {
        u64 first = ~0ULL, last = 0, generation = 0;
        const Buffer* arena = nullptr;
    };
    // bbport BB_LAYER_MEMORY: the game's memory in place through its chunk buffer (the module
    // says which); what is not whole in one chunk takes the paths below (VRAM copies).
    // Small read-only bindings the layer serves in place (constants the GPU wrote into the game's
    // memory, read over PCIe by every wave) from the VRAM copy made after their writer
    // (ShadowGpuWrites), when there is one.
    const auto small_to_vram = [&](std::pair<const Buffer*, u64> bound) {
        if (is_written || is_texel_buffer || size > STREAM_THRESHOLD || write_shadows.empty()) {
            return bound;
        }
        if (std::string_view{BoundPlace(bound.first)} != "game memory") {
            return bound;
        }
        const auto shadow = GpuWriteShadow(device_addr, size);
        if (!shadow) {
            BbStats::write_shadow_misses.fetch_add(1, std::memory_order_relaxed);
            return bound;
        }
        BbStats::write_shadow_hits.fetch_add(1, std::memory_order_relaxed);
        return *shadow;
    };
    if (LayerMode() && !is_written && !is_texel_buffer && !stats && BindMemoOn()) {
        // A reading binding resolved before, with nothing changed since (BindMemo).
        if (maintained_epoch != packet_epoch) {
            Maintain();
        }
        LayerProcessTraps();
        const BindMemo& memo = BindMemoSlot(device_addr, size);
        if (memo.address == device_addr && memo.size == size &&
            memo.generation == layer_generation &&
            memo.second == BbStats::coarse_second.load(std::memory_order_relaxed) &&
            memo.epoch == BbMem::cpu_change_epoch.load(std::memory_order_acquire)) {
            (memo.vram ? BbStats::bound_vram_bytes : BbStats::bound_in_place_bytes)
                .fetch_add(size, std::memory_order_relaxed);
            return small_to_vram({memo.buffer, memo.offset});
        }
    }
    if (LayerMode() && (is_written || size > STREAM_THRESHOLD || ConstantsInPlace() ||
                        IsRegionGpuModified(device_addr, size) || LayerMirrored(device_addr, size))) {
        if (const auto bound = LayerBind(device_addr, size, is_written, is_texel_buffer)) {
            return small_to_vram(*bound);
        }
        if (!is_written && !is_texel_buffer) {
            if (const auto stitched = LayerStitch(device_addr, size)) {
                return *stitched;
            }
        }
        NoteLayerMiss(device_addr, size, is_written);
    }
    thread_local std::array<InPlaceMemo, 256> in_place_reads{};
    const u64 memo_first = device_addr >> block_shift;
    const u64 memo_last = (device_addr + size - 1) >> block_shift;
    auto& read_memo = in_place_reads[memo_first & (in_place_reads.size() - 1)];
    const bool memo_candidate = !LayerMode() && !is_written &&
                                !is_texel_buffer && !stats &&
                                (size > STREAM_THRESHOLD || ConstantsInPlace());
    if (memo_candidate && maintained_epoch != packet_epoch) {
        Maintain(); // what EnsureResident does first: residency changes queued since
    }
    if (memo_candidate && read_memo.first == memo_first && read_memo.last == memo_last &&
        read_memo.generation == ResidencyGeneration()) {
        NoteUse(memo_first << block_shift, (memo_last - memo_first + 1) << block_shift);
        BbStats::bound_in_place_bytes.fetch_add(size, std::memory_order_relaxed);
        return {read_memo.arena, read_memo.arena->Offset(device_addr)};
    }
    if ((is_written || size > STREAM_THRESHOLD || ConstantsInPlace() ||
                           IsRegionGpuModified(device_addr, size))) {
        const u64 first_block = device_addr >> block_shift;
        const u64 last_block = (device_addr + size - 1) >> block_shift;
        const auto* arena = GetArena(first_block, last_block);
        // GPU-written buffers: in place with write tracking (CPU reads of them stay coherent, as
        // before). Without it the GPU is a writer we hear of, like the loaders: VRAM wherever whole
        // blocks are known data, as a default heap. The CPU does not read what the GPU wrote there
        // (no readbacks: BB_READBACKS was relaxed before too); BB_GPU_WRITES_IN_PLACE=1 keeps them
        // in place (PCIe, ~9 ms a frame on an RX 7800 XT).
        static const bool writes_in_place_forced = bb_env_on("BB_GPU_WRITES_IN_PLACE");
        const bool writes_in_place = writes_in_place_forced || force_writes_in_place;
        if (is_written && !writes_in_place) {
            NoteGpuWrite(device_addr, size);
        }
        EnsureResident(arena, first_block, last_block, is_written && writes_in_place);
        if (MixStats()) {
            NoteResidencyMix(first_block, last_block, size, is_written);
        }
        if (is_written && size > 64_KB && BbStats::enabled) {
            NoteLargeGpuWrite(device_addr, size);
        }
        if (IsInPlace(device_addr, size)) {
            if (is_written) {
                WatchGpuWrite(device_addr, size, __LINE__), WriteTicks().Note(device_addr, size, scheduler.CurrentTick());
            }
            BbStats::bound_in_place_bytes.fetch_add(size, std::memory_order_relaxed);
            if (is_written) {
                BbStats::bound_in_place_written_bytes.fetch_add(size, std::memory_order_relaxed);
            }
            std::pair<const Buffer*, u64> vram_copy{};
            if (is_texel_buffer && !is_written) {
                SynchronizeMemoryFromImage(arena, arena->Offset(device_addr), device_addr, size,
                                           &vram_copy);
            }
            TraceBinding(device_addr, size, is_written, is_written ? 0 : 2);
            if (vram_copy.first) {
                return vram_copy; // the image tiled in VRAM (SynchronizeMemoryFromImage)
            }
            if (!is_written && !is_texel_buffer) {
                if (const auto shadow = ShadowCopy(device_addr, size)) {
                    return *shadow;
                }
            }
            if (memo_candidate) {
                read_memo = {memo_first, memo_last, ResidencyGeneration(), arena};
            }
            return {arena, arena->Offset(device_addr)};
        }
    }
    // BB_EXP_NO_STREAM=1 (experiment, the flicker): never through the stream buffer.
    static const bool no_stream = bb_env("BB_EXP_NO_STREAM") != nullptr;
    // For read-only buffers use device local stream buffer to reduce renderpass breaks.
    if (!is_written && size <= STREAM_THRESHOLD && !IsRegionGpuModified(device_addr, size) &&
        !no_stream) {
        BB_SECTION(ObtainStream);
        if (stats) {
            auto& region = Stats().regions[RegionKey(device_addr)];
            ++region.stream_count;
            region.stream_bytes += size;
        }
        // bbport: the guest data is copied on a copy thread, started now; submission and
        // guest-visible fences wait for it (Scheduler::WaitHostCopies).
        if (!stream_buffer.mapped_data.empty() &&
            !BbToggle::Disabled(BbToggle::DeferredStreamCopies)) {
            if (const auto offset = stream_buffer.Reserve(size, instance.UniformMinAlignment())) {
                SmallGuestCopy({
                    .run = &RunGuestCopy,
                    .context = this,
                    .source = device_addr,
                    .destination = reinterpret_cast<u64>(stream_buffer.mapped_data.data() + *offset),
                    .size = size,
                    .extra = reinterpret_cast<u64>(static_cast<Buffer*>(&stream_buffer)),
                });
                TraceBinding(device_addr, size, false, 3);
                return {&stream_buffer, *offset};
            }
        }
        const auto [data, offset] = stream_buffer.Map(size, instance.UniformMinAlignment());
        memory->CopySparseMemory(device_addr, data, size);
        stream_buffer.Commit();
        TraceBinding(device_addr, size, false, 3);
        return {&stream_buffer, offset};
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    BB_SECTION(ObtainVram);
    if (!is_written) {
    }
    const u64 uploaded_before = BbStats::buffer_upload_bytes.load(std::memory_order_relaxed);
    BbStats::bound_vram_bytes.fetch_add(size, std::memory_order_relaxed);
    SynchronizeMemory(arena, device_addr, size, is_written, is_texel_buffer);
    if (stats) {
        auto& region = Stats().regions[RegionKey(device_addr)];
        ++region.arena_count;
        region.arena_bytes += size;
        region.upload_bytes +=
            BbStats::buffer_upload_bytes.load(std::memory_order_relaxed) - uploaded_before;
    }
    if (is_written) {
        gpu_modified_ranges.Add(device_addr, size);
        WatchGpuWrite(device_addr, size, __LINE__), WriteTicks().Note(device_addr, size, scheduler.CurrentTick());
        if (Readbacks()) {
            written_bindings[written_head.fetch_add(1) % written_bindings.size()] =
                WrittenBinding{device_addr, size, scheduler.CurrentTick()};
        }
    }
    TraceBinding(device_addr, size, is_written, is_written ? 1 : 4);
    return {arena, arena->Offset(device_addr)};
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::CommandWriteMirror(VAddr device_addr,
                                                                           u32 size) {
    if (!LayerMode() || size == 0) {
        return std::nullopt;
    }
    const auto resolution = BbLayer::GpuMemory::Get().Resolve(device_addr, size);
    if (resolution.kind != BbLayer::Resolution::Kind::Mirror) {
        return std::nullopt;
    }
    // Not over volatile blocks (refreshed from the game's memory anyway) or data only the GPU has
    // there (copied back before a demotion; a write beside it keeps the rules simple).
    const u64 first = device_addr >> block_shift, end = ((device_addr + size - 1) >> block_shift) + 1;
    if (layer_volatile.Overlaps(first, end) || gpu_modified_ranges.Intersects(device_addr, size)) {
        return std::nullopt;
    }
    auto* mirror = static_cast<Buffer*>(resolution.span.source->owner);
    return std::pair<const Buffer*, u64>{mirror, mirror->Offset(device_addr)};
}

std::optional<std::pair<const Buffer*, u64>> BufferCache::CommandWriteTarget(VAddr device_addr,
                                                                           u32 size) {
    if (size == 0) {
        return std::nullopt;
    }
    if (LayerMode()) {
        // Written by the GPU in place: a VRAM copy of it would be stale (step 3a), unless the
        // range is whole in one mirror: then the caller writes that copy too (CommandWriteMirror).
        // Demoting it made watched blocks (BB_LAYER_WATCH) go back and forth between VRAM and in
        // place (WRITE_DATA beside the data shaders read: 2 GiB a minute).
        if (!CommandWriteMirror(device_addr, size)) {
            LayerDemote(device_addr >> block_shift, ((device_addr + size - 1) >> block_shift) + 1,
                        true, 2);
        }
        const auto target = LayerInPlace(device_addr, size);
        if (target) {
            NoteCommandWriteTick(device_addr, size, scheduler.CurrentTick());
        }
        return target;
    }
    const u64 first_block = device_addr >> block_shift;
    const u64 last_block = (device_addr + size - 1) >> block_shift;
    const auto* arena = GetArena(first_block, last_block);
    EnsureResident(arena, first_block, last_block);
    if (!IsInPlace(device_addr, size)) {
        return std::nullopt;
    }
    NoteCommandWriteTick(device_addr, size, scheduler.CurrentTick());
    return std::pair<const Buffer*, u64>{arena, arena->Offset(device_addr)};
}

std::pair<const Buffer*, u64> BufferCache::ObtainBufferForImage(VAddr device_addr, u32 size) {
    // bbport: texture data is read by the GPU from the game's memory itself:
    // through the arena where it is bound in place, else (BB_CHUNK_TEXTURES) from its guest
    // memory chunk.
    if (LayerMode()) {
        // From its mirror where the data is in VRAM (the GPU may have written it there), else
        // in place (a mixed range: the GPU's data in mirrors copied back first, LayerBind). No
        // mirror is made for an upload: the image holds the data in VRAM.
        if (const auto source = LayerBind(device_addr, size, false, false, false)) {
            TraceBinding(device_addr, size, false, 6);
            return *source;
        }
        if (const auto stitched = LayerStitch(device_addr, size)) {
            return *stitched;
        }
        NoteLayerMiss(device_addr, size, false);
    } else {
        const u64 first_block = device_addr >> block_shift;
        const u64 last_block = (device_addr + size - 1) >> block_shift;
        if (InPlaceTextures()) {
            EnsureResident(GetArena(first_block, last_block), first_block, last_block, true);
        }
        if (IsInPlace(device_addr, size)) {
            const auto* arena = GetArena(first_block, last_block);
            TraceBinding(device_addr, size, false, 5);
            return {arena, arena->Offset(device_addr)};
        }
        if (ChunkTextures() && !IsRegionGpuModified(device_addr, size)) {
            if (const auto source = GuestChunkSource(device_addr, size)) {
                TraceBinding(device_addr, size, false, 6);
                    return *source;
            }
        }
    }
    if (IsRegionGpuModified(device_addr, size)) {
        TraceBinding(device_addr, size, false, 7);
        return ObtainBuffer(device_addr, size, false);
    }
    TraceBinding(device_addr, size, false, 9);
    const auto staging = staging_pool.Request(size, VideoCore::MemoryType::HostUncached,
                                              instance.StorageMinAlignment());
    if (!BbToggle::Disabled(BbToggle::DeferredUploads)) {
        // bbport: texture data is copied on the copy threads (streaming: 100+ MB per frame);
        // the upload reads the staging only after submission, which waits for the copies.
        constexpr u64 Chunk = 1_MB;
        for (u64 offset = 0; offset < staging.size; offset += Chunk) {
            const BbCopy::Item item{
                .run = &RunGuestCopy,
                .context = this,
                .source = device_addr + offset,
                .destination = reinterpret_cast<u64>(staging.mapped + offset),
                .size = std::min(Chunk, staging.size - offset),
                .extra = reinterpret_cast<u64>(staging.buffer),
            };
            if (staging.size < Chunk) {
                SmallGuestCopy(item);
            } else {
                BbCopy::Async([item] { item.run(item); });
                scheduler.NoteHostCopySource(item.source, item.size);
            }
        }
        return {staging.buffer, staging.offset};
    }
    memory->CopySparseMemory(device_addr, staging.mapped, staging.size);
    staging.Flush();
    return {staging.buffer, staging.offset};
}

bool BufferCache::IsRegionCpuModified(VAddr addr, size_t size) {
    return memory_tracker->IsRegionCpuModified(addr, size);
}

bool BufferCache::IsRegionGpuModified(VAddr addr, size_t size) {
    if (memory_tracker->IsRegionGpuModified(addr, size)) {
        return true;
    }
    // bbport: GPU writes are not tracked per page; one not done yet counts.
    // So does what the GPU wrote into a VRAM copy (NoteGpuWrite: its data, as a default heap):
    // the game's memory under it is older (issue #122). BB_GPU_DATA_CHECK=0: not counted.
    static const bool gpu_data_check = !bb_env_off("BB_GPU_DATA_CHECK");
    if (gpu_data_check && !LayerMode() && gpu_written_bytes.Overlaps(addr, addr + size) &&
        !IsInPlace(addr, size)) {
        return true;
    }
    const u64 tick = NewestGpuWrite(addr, size);
    if (tick == 0 || scheduler.GetWorkSemaphore()->IsFree(tick)) {
        return false;
    }
    // Asking the driver is a system call, and the GPU command thread asks for thousands of
    // constant bindings a frame: at most every ~250 us per thread (until then the write
    // counts as not done, which only takes the slower, always correct path; the GPU signal
    // thread refreshes the known tick on every fence anyway).
    thread_local u64 last_refresh = 0;
    if (const u64 now = BbCpu::Ticks(); now - last_refresh > BbCpu::TicksFromUs(250)) {
        last_refresh = now;
        scheduler.GetWorkSemaphore()->Refresh();
    }
    return !scheduler.GetWorkSemaphore()->IsFree(tick);
}

u32 BufferCache::GpuModifiedWhy(VAddr addr, size_t size) {
    if (memory_tracker->IsRegionGpuModified(addr, size)) {
        return 1;
    }
    if (!LayerMode() && gpu_written_bytes.Overlaps(addr, addr + size) && !IsInPlace(addr, size)) {
        return 2;
    }
    const u64 tick = NewestGpuWrite(addr, size);
    if (tick != 0 && !scheduler.GetWorkSemaphore()->IsFree(tick)) {
        static std::atomic<u64> seen{0};
        if (const u64 n = seen.fetch_add(1); n % 20000 == 0 && n < 20000 * 40) {
            std::printf("GPU-modified why: %#llx+%zu page tick %llu precise tick %llu (now done %llu)\n",
                        (unsigned long long)addr, size,
                        (unsigned long long)WriteTicks().Newest(addr, size),
                        (unsigned long long)PreciseTicks().Newest(addr, size),
                        (unsigned long long)scheduler.GetWorkSemaphore()->KnownGpuTick());
        }
        return 3;
    }
    return 0;
}

void BufferCache::ProcessFaultBuffer() {
    fault_manager->ProcessFaultBuffer();
}

void BufferCache::SynchronizeDmaBuffers() {
    for (const auto& range : resident_ranges) {
        const u64 page = range.start >> (ARENA_PAGE_BITS - block_shift);
        const VAddr device_addr = range.start << block_shift;
        const u64 size = (range.end - range.start) << block_shift;
        SynchronizeMemory(address_space[page], device_addr, size, false, false);
    }
}

const Buffer* BufferCache::GetArena(u64 first_block, u64 last_block) {
    // Arenas cover whole 4 GiB pages of the address space; a binding lies in at most two pages.
    const u64 first_page = first_block >> blocks_per_arena_page_shift;
    const u64 last_page = last_block >> blocks_per_arena_page_shift;
    ASSERT_MSG(last_page - first_page <= 1, "a binding over more than two arena pages");
    const Buffer* first = address_space[first_page];
    const Buffer* last = address_space[last_page];
    if (first == last) {
        if (!first) {
            first = &arenas.emplace_back(instance, first_page << ARENA_PAGE_BITS,
                                         (last_page - first_page + 1) << ARENA_PAGE_BITS,
                                         MemoryType::Sparse);
            address_space[first_page] = first;
            address_space[last_page] = first;
            if (bb_env("BB_BDA_LOG")) {
                std::printf("BDA: arena %#llx+%#llx at GPU %#llx\n",
                            (unsigned long long)first->cpu_addr,
                            (unsigned long long)first->size_bytes,
                            (unsigned long long)first->BufferDeviceAddress());
            }
        }
        return first;
    }

    // Different arenas (or one and none): one arena over both, their resident blocks bound into
    // it again.
    LOG_WARNING(Render, "Migrating arena");
    const VAddr base = first ? first->cpu_addr : first_page << ARENA_PAGE_BITS;
    const u64 first_size = first ? first->size_bytes : ARENA_PAGE_SIZE;
    const u64 last_size = last ? last->size_bytes : ARENA_PAGE_SIZE;
    const u64 base_block = base >> block_shift;
    const u64 end_block = (base + first_size + last_size) >> block_shift;
    const Buffer* merged =
        &arenas.emplace_back(instance, base, first_size + last_size, MemoryType::Sparse);
    auto* binds = BindsForArena(merged);
    resident_ranges.ForEachInRange(base_block, end_block, [&](const Backing& backing) {
        const u64 start = std::max(base_block, backing.start);
        const u64 end = std::min(end_block, backing.end);
        binds->binds.push_back(vk::SparseMemoryBind{
            .resourceOffset = (start - base_block) << block_shift,
            .size = (end - start) << block_shift,
            .memory = backing.memory,
            .memoryOffset = (backing.offset + (start - backing.start)) << block_shift,
        });
    });
    const auto assign = [&](u64 page, u64 bytes) {
        for (u64 i = 0; i < (bytes >> ARENA_PAGE_BITS); ++i) {
            address_space[page + i] = merged;
        }
    };
    assign(base >> ARENA_PAGE_BITS, first_size);
    assign(last_page, last_size);
    return merged;
}

} // namespace VideoCore

namespace VideoCore {
void BufferCache::NoteWriteTick(VAddr address, u64 size) {
    // BB_TICK_LOG=1 (diagnostics): the ranges marked written by the GPU most often, per 5 s.
    static const bool tick_log = bb_env("BB_TICK_LOG") != nullptr;
    if (tick_log) {
        static std::mutex mutex;
        static std::unordered_map<u64, u64> counts;
        static auto last = std::chrono::steady_clock::now();
        std::scoped_lock lk{mutex};
        ++counts[address << 8 ^ std::min<u64>(size, 0xff)];
        if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) {
            last = std::chrono::steady_clock::now();
            std::vector<std::pair<u64, u64>> top(counts.begin(), counts.end());
            std::ranges::sort(top, std::greater{}, [](const auto& e) { return e.second; });
            std::string text;
            for (size_t i = 0; i < std::min<size_t>(top.size(), 12); ++i) {
                text += fmt::format(" {:#x}(sz~{}):{}", top[i].first >> 8, top[i].first & 0xff, top[i].second);
            }
            std::printf("GPU write ticks (5 s, %zu ranges):%s\n", counts.size(), text.c_str());
            counts.clear();
        }
    }
    // The same range again in the same submission changes nothing: the copy shader's paged
    // bindings over nearly all memory (~60 a frame, ~10 a submission) were a store per MiB each.
    thread_local VAddr last_address = 0;
    thread_local u64 last_size = 0, last_tick = 0;
    const u64 tick = scheduler.CurrentTick();
    if (address == last_address && size == last_size && tick == last_tick) {
        return;
    }
    last_address = address;
    last_size = size;
    last_tick = tick;
    WatchGpuWrite(address, size, __LINE__), WriteTicks().Note(address, size, tick);
}
} // namespace VideoCore

namespace VideoCore {
bool BufferCache::GarlicInVramForLayer() {
    return GarlicInVram();
}
} // namespace VideoCore

namespace VideoCore {
u64 BufferCache::LayerMirrorFloor(u64 budget) {
    // BB_LAYER_MIRROR_FLOOR_PERCENT of the driver's VRAM budget: the mirrors' share that the
    // texture collector's mark does not take from them. Past it they yield to images; under it
    // images yield (the collector evicts more), up to the budget's BB_VRAM_PROMOTE_PERCENT. Reading
    // the game's memory over the bus costs far more on NVIDIA. 25 %; 40 % on cards of 4 GB or
    // less (GTX 1050, 2 GB: the G-buffer read its vertices over PCIe with 25 %).
    static const u64 forced = [] {
        const char* v = bb_env("BB_LAYER_MIRROR_FLOOR_PERCENT");
        return v && *v ? std::min<u64>(std::strtoull(v, nullptr, 10), 90ull) : 0ull;
    }();
    const u64 percent = forced ? forced : budget <= (u64{4} << 30) ? 40 : 25;
    return budget / 100 * percent;
}

u64 BufferCache::LayerMirrorHeadroom(u64 budget) {
    // What the texture collector keeps free for the mirrors under their floor: room for an
    // area's copies to come at once, not the whole floor. The mirrors seldom fill it (only known
    // data moves): on the 2 GB laptop 180-300 MiB of a 658 MiB floor, so the collector held the
    // images at 670-920 MiB and evicted 50-200 every 5 s with ~600 MiB of VRAM unused.
    // BB_LAYER_MIRROR_HEADROOM_MB=N; 0 = the whole floor (as before).
    static const s64 forced = [] {
        const char* v = bb_env("BB_LAYER_MIRROR_HEADROOM_MB");
        return v && *v ? s64(std::strtoull(v, nullptr, 10)) : s64(-1);
    }();
    if (forced == 0) {
        return ~0ull;
    }
    return forced > 0 ? u64(forced) << 20 : budget / 8;
}

u64 BufferCache::PromoteCriticalMargin() const {
    return std::min<u64>(384_MB, instance.GetTotalMemoryBudget() / 12);
}

u64 BufferCache::StaleHeadroom() const {
    return std::min<u64>(256_MB, instance.GetTotalMemoryBudget() / 16);
}

bool BufferCache::VramPromotionsPaused() {
    // VRAM nearly full (past BB_VRAM_PROMOTE_PERCENT of the driver's budget, default 90): no new
    // copies; the candidates wait, read in place meanwhile (#32: a card with 8 GB). Checked at
    // most once a second (the query goes to the driver).
    if (!instance.CanReportMemoryUsage()) {
        return false;
    }
    static const u64 percent = [] {
        const char* v = bb_env("BB_VRAM_PROMOTE_PERCENT");
        return v && *v ? std::strtoull(v, nullptr, 10) : 90ull;
    }();
    static s64 checked_second = -1;
    static bool full = false;
    const s64 second = BbStats::coarse_second.load(std::memory_order_relaxed);
    if (second != checked_second) {
        checked_second = second;
        const u64 usage = instance.GetDeviceMemoryUsage();
        const u64 budget = instance.GetDeviceMemoryBudgetNow();
        // Other programs taking video memory shrink the budget (laptop: 1.83 GB with nothing else
        // open, 1.52 GB with a browser: VRAM past it, Windows paging it out, 8 s frames, 2026-10-26).
        static u64 most_budget = 0;
        static u32 lost_printed = 0;
        most_budget = std::max(most_budget, budget);
        const u64 lost = most_budget > budget ? most_budget - budget : 0;
        BbStats::vram_budget_lost_mb.store(u32(lost >> 20), std::memory_order_relaxed);
        if (lost >= (u64{150} << 20) && second - lost_printed >= 30) {
            lost_printed = u32(second);
            std::printf("Guest memory: the VRAM budget is %llu MiB, %llu MiB less than earlier in this "
                        "session: other programs use video memory (browsers with hardware "
                        "acceleration, video players, overlays)\n",
                        (unsigned long long)(budget >> 20), (unsigned long long)(lost >> 20));
        }
        // Also short of the texture collector's critical mark: the collector compares all of our
        // VRAM and can only free images, so copies past it had it evict textures in use (6 GB
        // GTX 1660 Ti: mark 3927 MiB, 1.8 GiB of copies, 200-1250 images evicted per 5 s).
        // Not below the mirrors' own share (LayerMirrorFloor), though: on a 4 GB card images and
        // the rest already held the usage past that point, no copy was made again once the old
        // ones went idle, and everything was read over the bus (5 FPS after 4 minutes).
        const u64 critical = BbStats::gc_critical_bytes.load(std::memory_order_relaxed);
        const bool under_floor = LayerMode() && BbLayer::GpuMemory::Get().ValidBytes() <
                                                    LayerMirrorFloor(budget);
        const bool near_critical =
            critical != 0 && usage + PromoteCriticalMargin() >= critical && !under_floor;
        // Under their share the copies go on up to 95 % (the collector makes room for them).
        const u64 limit = under_floor ? std::max<u64>(percent, 95) : percent;
        const bool now_full = (budget != 0 && usage * 100 > budget * limit) || near_critical;
        if (now_full != full) {
            std::printf("Guest memory: VRAM %llu of %llu MiB in use (texture collector critical at "
                        "%llu): copies of loaded blocks %s\n",
                        (unsigned long long)(usage >> 20), (unsigned long long)(budget >> 20),
                        (unsigned long long)(critical >> 20),
                        now_full ? "wait (read in place meanwhile)" : "resume");
        }
        full = now_full;
    }
    return full;
}
} // namespace VideoCore

namespace VideoCore {
void BufferCache::NoteBindingWriteTick(VAddr address, u64 size) {
    if (!precise_writes || !force_writes_in_place) {
        NoteWriteTick(address, size);
        return;
    }
    const VAddr end = address + size;
    const auto& ranges = *precise_writes;
    auto it = std::lower_bound(ranges.begin(), ranges.end(), address,
                               [](const auto& r, VAddr a) { return r.first + r.second <= a; });
    for (; it != ranges.end() && it->first < end; ++it) {
        const VAddr from = std::max<VAddr>(it->first, address);
        const VAddr to = std::min<VAddr>(it->first + it->second, end);
        if (from < to) {
            WatchGpuWrite(from, to - from, __LINE__), PreciseTicks().Note(from, to - from, scheduler.CurrentTick());
        }
    }
}
} // namespace VideoCore
