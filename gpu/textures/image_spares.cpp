// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport textures: spare images made ahead (image_spares.h).
#include "textures/image_spares.h"
#include "shim/bb_env.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <vk_mem_alloc.h>

#include "bbport_toggles.h"
#include "common/elf_info.h"
#include "common/path_util.h"
#include "common/thread.h"

namespace VideoCore::ImageSpares {

namespace {

/// What makes two images interchangeable: their create info (no pNext, exclusive sharing).
struct Shape {
    u32 flags, type, format, width, height, depth, mips, layers, samples, tiling, usage, layout;
    bool operator==(const Shape&) const = default;
};
struct ShapeHash {
    size_t operator()(const Shape& s) const noexcept {
        u64 h = 0xcbf29ce484222325ull;
        const auto* bytes = reinterpret_cast<const u8*>(&s);
        for (size_t i = 0; i < sizeof(s); ++i) {
            h = (h ^ bytes[i]) * 0x100000001b3ull;
        }
        return size_t(h);
    }
};

bool ShapeOf(const vk::ImageCreateInfo& ci, Shape& shape) {
    if (ci.pNext || ci.sharingMode != vk::SharingMode::eExclusive || ci.queueFamilyIndexCount) {
        return false;
    }
    shape = {u32(VkImageCreateFlags(ci.flags)), u32(ci.imageType), u32(ci.format), ci.extent.width,
             ci.extent.height, ci.extent.depth, ci.mipLevels, ci.arrayLayers, u32(ci.samples),
             u32(ci.tiling), u32(VkImageUsageFlags(ci.usage)), u32(ci.initialLayout)};
    return true;
}

vk::ImageCreateInfo CreateInfoOf(const Shape& s) {
    return vk::ImageCreateInfo{
        .flags = vk::ImageCreateFlags(s.flags),
        .imageType = vk::ImageType(s.type),
        .format = vk::Format(s.format),
        .extent = {s.width, s.height, s.depth},
        .mipLevels = s.mips,
        .arrayLayers = s.layers,
        .samples = vk::SampleCountFlagBits(s.samples),
        .tiling = vk::ImageTiling(s.tiling),
        .usage = vk::ImageUsageFlags(s.usage),
        .sharingMode = vk::SharingMode::eExclusive,
        .initialLayout = vk::ImageLayout(s.layout),
    };
}

struct Spare {
    vk::Image image;
    VmaAllocation allocation;
    vk::DeviceSize size;
};

struct Entry {
    std::vector<Spare> spares;
    vk::DeviceSize size = 0; ///< memory of one image (0: not known yet)
    u32 made = 0;            ///< made on the caller's thread this session
    u32 taken = 0;           ///< spares taken this session
    u32 saved = 0;           ///< demand of the last sessions (image_shapes.txt)
    std::chrono::steady_clock::time_point failed{}; ///< the last allocation that failed
};

/// How many spares a shape gets: by its demand (this session, then the last ones, halved).
u32 Wanted(const Entry& e) {
    const u32 demand = std::max(e.made + e.taken, e.saved / 2);
    return demand >= 32 ? 4 : demand >= 8 ? 2 : demand >= 2 ? 1 : 0;
}

struct State {
    std::mutex mutex;
    std::condition_variable cv;
    std::thread thread;
    bool stop = false, dirty = false, changed = false;
    vk::Device device{};
    VmaAllocator allocator{};
    vk::DeviceSize cap = 0, limit = 0, bytes = 0; ///< cap: the setting; limit: now (Trim)
    std::unordered_map<Shape, Entry, ShapeHash> shapes;
    std::filesystem::path file;
    u64 hits = 0, made_ahead = 0;
    std::chrono::steady_clock::time_point saved_at{};
};
State* state = nullptr;

bool Enabled() {
    static const bool on = [] {
        const char* env = bb_env("BB_IMAGE_SPARES");
#ifdef _WIN32
        return !(env && env[0] == '0');
#else
        return env && env[0] == '1';
#endif
    }();
    return on;
}

void DestroySpare(State& s, const Spare& spare) {
    vmaDestroyImage(s.allocator, VkImage(spare.image), spare.allocation);
    BbStats::vk_image_bytes.fetch_sub(spare.size, std::memory_order_relaxed);
    s.bytes -= spare.size;
}

/// The shapes of a list; seed (the list shipped with the port, BB_IMAGE_SHAPES): only shapes this
/// player's own list lacks, its memory sizes (another GPU's) an estimate until one is made here.
/// The number read.
u32 Load(State& s, const std::filesystem::path& path, bool seed) {
    FILE* f = std::fopen(path.string().c_str(), "r");
    if (!f) {
        return 0;
    }
    Shape sh{};
    unsigned long long size = 0;
    u32 demand = 0, count = 0;
    while (std::fscanf(f, "%u %u %u %u %u %u %u %u %u %u %u %u %llu %u", &sh.flags, &sh.type,
                       &sh.format, &sh.width, &sh.height, &sh.depth, &sh.mips, &sh.layers,
                       &sh.samples, &sh.tiling, &sh.usage, &sh.layout, &size, &demand) == 14) {
        if (seed && s.shapes.contains(sh)) {
            continue;
        }
        auto& e = s.shapes[sh];
        e.size = size;
        e.saved = demand;
        ++count;
    }
    std::fclose(f);
    return count;
}

/// Under the mutex.
void Save(State& s) {
    std::error_code ec;
    std::filesystem::create_directories(s.file.parent_path(), ec);
    const auto tmp = s.file.string() + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) {
        return;
    }
    for (const auto& [sh, e] : s.shapes) {
        // This session's demand plus a decaying share of the earlier ones.
        const u32 demand = std::min<u32>(e.made + e.taken + e.saved / 2, 1u << 20);
        if (demand >= 2 && e.size) {
            std::fprintf(f, "%u %u %u %u %u %u %u %u %u %u %u %u %llu %u\n", sh.flags, sh.type,
                         sh.format, sh.width, sh.height, sh.depth, sh.mips, sh.layers, sh.samples,
                         sh.tiling, sh.usage, sh.layout, (unsigned long long)e.size, demand);
        }
    }
    std::fclose(f);
    std::filesystem::rename(tmp, s.file, ec);
    s.changed = false;
    s.saved_at = std::chrono::steady_clock::now();
}

/// The shape most in want of a spare that fits, or nullptr (under the mutex).
const Shape* Next(State& s) {
    const auto now = std::chrono::steady_clock::now();
    const Shape* best = nullptr;
    u32 best_gap = 0;
    for (const auto& [sh, e] : s.shapes) {
        const u32 want = Wanted(e);
        if (e.spares.size() >= want || now - e.failed < std::chrono::seconds(5)) {
            continue;
        }
        // An unknown size is learnt by making the first one; none larger than a quarter of the cap.
        if (e.size && (s.bytes + e.size > s.limit || e.size > s.cap / 4)) {
            continue;
        }
        const u32 gap = want - u32(e.spares.size());
        if (!best || gap > best_gap) {
            best = &sh;
            best_gap = gap;
        }
    }
    return best;
}

void Run(State& s) {
    Common::SetCurrentThreadName("ImageSpares");
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    std::unique_lock lock{s.mutex};
    while (!s.stop) {
        const Shape* next = Next(s);
        if (!next) {
            if (s.changed && std::chrono::steady_clock::now() - s.saved_at > std::chrono::seconds(30)) {
                Save(s);
            }
            s.dirty = false;
            Common::WaitFor(s.cv, lock, std::chrono::seconds(5), [&] { return s.stop || s.dirty; });
            continue;
        }
        const Shape shape = *next;
        const vk::ImageCreateInfo ci = CreateInfoOf(shape);
        lock.unlock();
        // As HostImage::Create makes them: the image, its memory, the bind (the slow part on WDDM,
        // here off the recording thread).
        Spare spare{};
        VkImage handle{};
        bool ok = VkResult(s.device.createImage(&ci, nullptr, reinterpret_cast<vk::Image*>(&handle))) ==
                  VK_SUCCESS;
        VmaAllocationInfo info{};
        if (ok) {
            const VmaAllocationCreateInfo aci{.flags = MemoryFlags(s.device, ci),
                                              .usage = VMA_MEMORY_USAGE_GPU_ONLY};
            ok = vmaAllocateMemoryForImage(s.allocator, handle, &aci, &spare.allocation, &info) ==
                 VK_SUCCESS;
            if (!ok) {
                s.device.destroyImage(vk::Image{handle});
            }
        }
        if (ok && vmaBindImageMemory(s.allocator, spare.allocation, handle) != VK_SUCCESS) {
            vmaDestroyImage(s.allocator, handle, spare.allocation);
            ok = false;
        }
        lock.lock();
        auto& e = s.shapes[shape];
        if (!ok) {
            e.failed = std::chrono::steady_clock::now(); // over the budget: later
            continue;
        }
        spare.image = vk::Image{handle};
        spare.size = info.size;
        e.size = info.size;
        BbStats::vk_image_bytes.fetch_add(spare.size, std::memory_order_relaxed);
        s.bytes += spare.size;
        ++s.made_ahead;
        if (s.stop || s.bytes > s.limit) {
            DestroySpare(s, spare);
            continue;
        }
        e.spares.push_back(spare);
    }
}

} // namespace

void Start(vk::Device device, VmaAllocator allocator, u64 budget) {
    if (state || !Enabled() || !device) {
        return;
    }
    const char* env = bb_env("BB_IMAGE_SPARES_MB");
    const vk::DeviceSize cap = env && *env ? vk::DeviceSize(std::strtoull(env, nullptr, 10)) << 20
                                           : std::min<vk::DeviceSize>(budget / 32, 128ull << 20);
    if (cap == 0) {
        return;
    }
    state = new State;
    auto& s = *state;
    s.device = device;
    s.allocator = allocator;
    s.cap = s.limit = cap;
    s.file = Common::FS::GetUserPath(Common::FS::PathType::CacheDir);
    if (const auto serial = Common::ElfInfo::Instance().GameSerial(); !serial.empty()) {
        s.file /= std::string{serial};
    }
    s.file /= "image_shapes.txt";
    Load(s, s.file, false);
    // The port's list (run.py: the package's image_shapes.txt): a first session has them too.
    const char* seed = bb_env("BB_IMAGE_SHAPES");
    const u32 seeded = seed && *seed ? Load(s, seed, true) : 0;
    u32 ready = 0;
    for (const auto& [sh, e] : s.shapes) {
        ready += Wanted(e) != 0;
    }
    std::printf("Texture cache: spare images made ahead on their own thread (%llu MiB; %u shapes "
                "from the last sessions, %u of them from the port's list; BB_IMAGE_SPARES=0: off)\n",
                (unsigned long long)(cap >> 20), ready, seeded);
    s.dirty = true;
    s.thread = std::thread([&s] { Run(s); });
}

void Stop() {
    if (!state) {
        return;
    }
    auto& s = *state;
    {
        std::scoped_lock lock{s.mutex};
        s.stop = true;
    }
    s.cv.notify_all();
    s.thread.join();
    std::scoped_lock lock{s.mutex};
    for (auto& [sh, e] : s.shapes) {
        for (const auto& spare : e.spares) {
            DestroySpare(s, spare);
        }
        e.spares.clear();
    }
    Save(s);
    std::printf("Texture cache: %llu images taken from spares, %llu made ahead\n",
                (unsigned long long)s.hits, (unsigned long long)s.made_ahead);
    delete state;
    state = nullptr;
}

bool Take(VmaAllocator allocator, const vk::ImageCreateInfo& ci, vk::Image& image,
          VmaAllocation& allocation, vk::DeviceSize& size) {
    Shape shape;
    if (!state || allocator != state->allocator || !ShapeOf(ci, shape)) {
        return false;
    }
    auto& s = *state;
    std::scoped_lock lock{s.mutex};
    const auto it = s.shapes.find(shape);
    if (it == s.shapes.end() || it->second.spares.empty()) {
        return false;
    }
    const Spare spare = it->second.spares.back();
    it->second.spares.pop_back();
    ++it->second.taken;
    s.bytes -= spare.size; // the image's from now on (HostImage counts it in vk_image_bytes)
    image = spare.image;
    allocation = spare.allocation;
    size = spare.size;
    s.changed = s.dirty = true;
    if (++s.hits == 1 || s.hits % 256 == 0) {
        std::printf("Texture cache: %llu images taken from spares (no create or bind on this "
                    "thread; %llu MiB ready)\n",
                    (unsigned long long)s.hits, (unsigned long long)(s.bytes >> 20));
    }
    s.cv.notify_one();
    return true;
}

void NoteMade(const vk::ImageCreateInfo& ci, vk::DeviceSize size) {
    Shape shape;
    if (!state || !ShapeOf(ci, shape)) {
        return;
    }
    auto& s = *state;
    std::scoped_lock lock{s.mutex};
    auto& e = s.shapes[shape];
    ++e.made;
    e.size = size;
    s.changed = s.dirty = true;
    s.cv.notify_one();
}

void Trim(vk::DeviceSize keep_bytes) {
    if (!state) {
        return;
    }
    auto& s = *state;
    std::scoped_lock lock{s.mutex};
    const vk::DeviceSize before = s.limit;
    s.limit = std::min(s.cap, keep_bytes);
    // The least wanted shapes' spares go first.
    while (s.bytes > s.limit) {
        Entry* victim = nullptr;
        for (auto& [sh, e] : s.shapes) {
            if (!e.spares.empty() && (!victim || Wanted(e) < Wanted(*victim))) {
                victim = &e;
            }
        }
        if (!victim) {
            break;
        }
        DestroySpare(s, victim->spares.back());
        victim->spares.pop_back();
    }
    if (s.limit > before) { // room again: the thread may make what is missing
        s.dirty = true;
        s.cv.notify_one();
    }
}

} // namespace VideoCore::ImageSpares
