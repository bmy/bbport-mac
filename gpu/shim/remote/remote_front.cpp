// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game process's side of the native GPU process split (docs/macos-native-gpu.md).
// With the native GPU process (macOS default where bb-gpu is built), bb-probe (x86-64 under Rosetta) starts bb-gpu (native arm64), shares the
// guest's memory pool with it and forwards what reaches the GPU core: command buffer submissions,
// the VideoOut driver's calls, page-tracking faults and the runtime's memory events. bb-gpu sends
// back interrupts, flip and vblank status, retired frames and page protection changes.
#include <algorithm>
#include <array>
#include <bit>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <new>
#include <spawn.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <thread>
#include <type_traits>
#include <unistd.h>
#include <vector>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include "bb_env.h"
#include "bbport_frame_rate.h"
#include "bbport_game_menu.h"
#include "bbport_guest_hooks.h"
#include "bbport_portable.h"
#include "bbport_settings.h"
#include "bbport_threads.h"
#include "bbport_toggles.h"
#include "common/rdtsc.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "platform/gnm/gnm.h"
#include "platform/interrupts.h"
#include "platform/kernel/event_queue.h"
#include "platform/videoout/videoout.h"
#include "remote/bb_control.h"
#include "remote/bb_protocol.h"
#include "remote/bb_remote.h"

extern char** environ;

extern "C" {
int runtime_memory_host_pool(uint64_t host_bytes, int* fd, uint64_t* host_offset, uint64_t* total);
typedef void (*RuntimeMirrorHook)(int op, uintptr_t address, uint64_t size, uint64_t phys,
                                  int kind, int prot, int type, int shared);
void runtime_memory_set_mirror_hook(RuntimeMirrorHook hook);
int runtime_memory_is_mapped(uintptr_t address, uint64_t size);
int runtime_memory_lock_held(void);
int runtime_memory_exclude(uintptr_t start, uintptr_t end);
void runtime_memory_gpu_protect(uintptr_t address, uint64_t size, int read, int write);
void runtime_memory_trap(uintptr_t address, uint64_t size, unsigned reason, int on);
void runtime_memory_trap_seq(uintptr_t address, uint64_t size, unsigned reason, int on,
                             uint64_t seq, int lift);
unsigned runtime_memory_trap_reasons(uintptr_t address);
typedef void (*RuntimeGpuRange)(uintptr_t address, uint64_t size);
void runtime_memory_set_gpu_hooks(RuntimeGpuRange map, RuntimeGpuRange unmap,
                                  RuntimeGpuRange invalidate);
void runtime_memory_set_note_write_hook(RuntimeGpuRange note);
uint64_t runtime_process_time_counter(void);
uint64_t runtime_tsc_frequency(void);
void runtime_restart(void);
void runtime_wait_report(double frames);
// runtime_pad.c / runtime_file.c: the button prompts' state (bb-gpu draws them).
int runtime_pad_prompt(const char* name, int prefer_mouse, char* label, int size);
unsigned runtime_pad_prompt_generation(void);
int runtime_pad_prompt_device(void);
int runtime_pad_mouse_look(void);
int runtime_file_host_path(const char* guest, char* out, size_t size);
__attribute__((weak)) int runtime_pad_steam_only(void); // weak as in window.cpp
// The memory model's chunks (bbport 0.7): made by bb-gpu's GPU, asked for by this runtime.
void runtime_memory_set_guest_chunk_allocator(int (*alloc)(uint64_t phys, uint64_t size));
void runtime_memory_set_guest_chunk_free(void (*free_chunk)(uint64_t phys, uint64_t size));
void runtime_memory_set_guest_chunk_whole(int whole);
void runtime_memory_set_guest_lazy(int lazy);
void runtime_memory_note_pinned(uint64_t phys, uint64_t size);
}

namespace BbRemote {

namespace VO = BbVideoOut;
using VideoOutPort = BbVideoOut::Port;

static_assert(sizeof(VO::FlipStatus) == FlipStatusBytes);
static_assert(sizeof(VO::VblankStatus) == VblankStatusBytes);
static_assert(sizeof(VO::ResolutionStatus) == ResolutionBytes);
static_assert(sizeof(VO::BufferAttribute) == BufferAttributeBytes);
static_assert(sizeof(VO::DisplayBuffer) == VideoOutBufferBytes);
static_assert(sizeof(VO::BufferAttributeGroup) == AttributeGroupBytes);
static_assert(std::is_trivially_copyable_v<VO::FlipStatus> &&
              std::is_trivially_copyable_v<VO::VblankStatus> &&
              std::is_trivially_copyable_v<VO::ResolutionStatus> &&
              std::is_trivially_copyable_v<VO::DisplayBuffer> &&
              std::is_trivially_copyable_v<VO::BufferAttributeGroup>);
static_assert(VO::MaxDisplayBuffers == MaxDisplayBuffers &&
              VO::MaxDisplayBufferGroups == MaxDisplayBufferGroups);

bool Requested() {
    if (back_active) {
        return false;
    }
    // macOS: on by default where bb-gpu is built (the native branches with deps-arm64, and
    // their releases); a build without it runs the GPU work in this process. BB_NATIVE_GPU=0: in
    // this process anyway; BB_NATIVE_GPU=1: bb-gpu or nothing (a missing one stops the game).
    static const bool requested = [] {
        const char* env = std::getenv("BB_NATIVE_GPU");
        if (env && env[0] == '1') {
            return true;
        }
        if (env && env[0] == '0') {
            std::printf("GPU process: off (BB_NATIVE_GPU=0): the GPU work runs in this process\n");
            return false;
        }
#ifdef __APPLE__
        if (!Front::Built()) {
            std::printf("GPU process: not built here (tools/macos/build_gpu_native.sh): the GPU "
                        "work runs in this process\n");
            return false;
        }
        return true;
#else
        return false;
#endif
    }();
    return requested;
}

namespace Front {
namespace {

constexpr u64 SharePage = 16384; // shared mappings are whole 16 KiB pages, even under Rosetta

struct State {
    ControlBlock* control = nullptr;
    std::unique_ptr<Channel> channel, protect, fresh, locked;
    bool fresh_sync = false; ///< BB_REMOTE_FRESH_SYNC=1: FreshRange is a call (ControlBlock::fresh)
    int pool_fd = -1;
    u64 pool_bytes = 0, host_offset = 0, host_next = 0, host_end = 0;
    pid_t child = -1;
    std::atomic<VideoOutPort*> port{nullptr}; ///< set by the display, read by the reader thread
    u64 flip_version = 0; ///< newest flip status applied (under port->status_mutex)
    std::mutex asc_mutex;
    std::array<AscQueue, 64> asc{};
    std::mutex share_mutex;
    std::vector<std::pair<u64, u64>> shared_ranges; ///< ShareRange: [begin, end)
};
State g;

template <typename T>
T Payload(const Message& message) {
    T value{};
    std::memcpy(&value, message.data, std::min<std::size_t>(sizeof(T), message.size));
    return value;
}

u64 MonotonicNs() {
    timespec now{};
    clock_gettime(CLOCK_MONOTONIC, &now);
    return u64(now.tv_sec) * 1'000'000'000ull + u64(now.tv_nsec);
}

void CopyString(char* out, std::size_t size, const char* in) {
    std::snprintf(out, size, "%s", in ? in : "");
}

// ---- the runtime's memory events -----------------------------------------------------------

void MirrorHook(int op, uintptr_t address, uint64_t size, uint64_t phys, int kind, int prot,
                int type, int shared) {
    switch (op) {
    case 0: { // map (runtime_memory.c: KIND_DIRECT 2, KIND_FLEXIBLE 3)
        const MapArgs args{address,
                           size,
                           phys,
                           kind == 2 ? u32(MapDirect) : u32(MapFlexible),
                           prot,
                           kind == 2 ? type : -1,
                           shared ? 0u : u32(MapUnshared)};
        g.channel->Send(MsgMapMemory, &args, sizeof(args));
        break;
    }
    case 1: {
        const RangeArgs args{address, size};
        g.channel->Send(MsgUnmapMemory, &args, sizeof(args));
        break;
    }
    case 2: {
        const MirrorProtectArgs args{address, size, prot, type};
        g.channel->Send(MsgProtectMemory, &args, sizeof(args));
        break;
    }
    case 3: { // the range's GPU memory changed, its mappings did not (a heap piece's chunk)
        const RangeArgs args{address, size};
        g.channel->Send(MsgGpuRemap, &args, sizeof(args));
        break;
    }
    default:
        break;
    }
}

void SendRange(u32 type, uintptr_t address, uint64_t size) {
    const RangeArgs args{address, size};
    g.channel->Send(type, &args, sizeof(args));
}
void InvalidateHook(uintptr_t address, uint64_t size) {
    SendRange(MsgInvalidate, address, size);
}

/// The runtime's chunk allocator: bb-gpu's GPU imports the pool's pages there (CHUNK_IMPORTED,
/// -3), else -1 (the chunk stays plain pool memory). On ControlBlock::protect, whose bb-gpu side
/// never waits for this process: the runtime may call this with its lock held.
int RemoteChunkAlloc(uint64_t phys, uint64_t size) {
    const ChunkArgs args{phys, size};
    ResultReply result{-1, 0};
    g.protect->Call(MsgChunkAlloc, &args, sizeof(args), &result, sizeof(result));
    // Only an import keeps the game's mappings on the pool (a dma-buf would be bb-gpu's memory).
    return result.result == -3 ? -3 : -1;
}

void RemoteChunkFree(uint64_t phys, uint64_t size) {
    const ChunkArgs args{phys, size};
    ResultReply result{};
    g.protect->Call(MsgChunkFree, &args, sizeof(args), &result, sizeof(result));
}

/// The runtime places the game's direct memory in the chunks bb-gpu's GPU can use in place, as
/// BbGuestMemory::Install set them up there (bbport 0.7: the only memory model).
bool InstallChunks() {
    GuestMemoryReply memory{};
    g.channel->Call(MsgGuestMemory, nullptr, 0, &memory, sizeof(memory));
    if (memory.private_regions) {
        std::fprintf(stderr, "GPU process: the GPU takes the game's memory as private regions "
                             "(BB_GUEST_MEMORY=private), which only one process can have\n");
        return false;
    }
    if (!memory.allocator || !memory.host_import) {
        std::fprintf(stderr, "GPU process: its GPU does not import the game's memory from the "
                             "shared pool (host memory import), which this split needs\n");
        return false;
    }
    if (memory.lazy) {
        runtime_memory_set_guest_lazy(1);
    }
    if (memory.free_chunks) {
        runtime_memory_set_guest_chunk_free(&RemoteChunkFree);
    }
    runtime_memory_set_guest_chunk_whole(memory.whole ? 1 : 0);
    runtime_memory_set_guest_chunk_allocator(&RemoteChunkAlloc);
    std::printf("GPU process: the game's direct memory is imported by its GPU in place (%s%s)\n",
                memory.whole ? "a chunk per allocation" : "256 MiB chunks",
                memory.lazy ? ", lazily" : "");
    return true;
}
void NoteWriteHook(uintptr_t address, uint64_t size) {
    SendRange(MsgNoteWrite, address, size);
}

/// Guest memory bb-gpu sees at the same address: a guest mapping or a shared range.
bool Shared(const void* data, u64 bytes) {
    const auto address = reinterpret_cast<u64>(data);
    if (runtime_memory_is_mapped(address, bytes)) {
        return true;
    }
    std::scoped_lock lock{g.share_mutex};
    for (const auto& [begin, end] : g.shared_ranges) {
        if (address >= begin && address + bytes <= end) {
            return true;
        }
    }
    return false;
}

// ---- VideoOut port --------------------------------------------------------------------------

/// As display.cpp's TriggerEvents: the hint is the event id in the low byte, the flip argument or
/// vblank count from bit 16.
void TriggerEvents(const std::vector<s64>& queues, VO::InternalEvent event, u64 payload) {
    for (const s64 handle : queues) {
        if (auto* queue = BbKernel::FindQueue(handle)) {
            queue->Trigger(u64(event), BbKernel::Event::VideoOut,
                           reinterpret_cast<void*>(u64(event) | (payload << 16)));
        }
    }
}

/// The port's state from bb-gpu (status_mutex taken here).
void ApplyVoState(const VoState& state) {
    VideoOutPort* port = g.port;
    if (!port) {
        return;
    }
    std::scoped_lock lock{port->status_mutex};
    if (state.version > g.flip_version) {
        g.flip_version = state.version;
        std::memcpy(&port->flip_status, state.flip_status, FlipStatusBytes);
    }
    std::memcpy(port->buffers.data(), state.slots, sizeof(state.slots));
    // Field by field: is_occupied is a bool, and another process's byte may be anything.
    for (std::size_t i = 0; i < MaxDisplayBufferGroups; ++i) {
        const u8* group = state.groups + i * AttributeGroupBytes;
        port->groups[i].is_occupied = group[offsetof(VO::BufferAttributeGroup, is_occupied)] != 0;
        std::memcpy(&port->groups[i].attrib, group + offsetof(VO::BufferAttributeGroup, attrib),
                    sizeof(VO::BufferAttribute));
    }
    std::memcpy(&port->resolution, state.resolution, ResolutionBytes);
    port->is_open = state.is_open != 0;
}

void ApplyFlipStatus(const VoStatusArgs& args) {
    VideoOutPort* port = g.port;
    if (!port) {
        return;
    }
    {
        std::scoped_lock lock{port->status_mutex};
        if (args.version > g.flip_version) {
            g.flip_version = args.version;
            std::memcpy(&port->flip_status, args.flip_status, FlipStatusBytes);
        }
    }
    if (!args.flip_event) {
        return;
    }
    // As Display::Flip does in-process.
    TriggerEvents(port->flip_queues, VO::InternalEvent::Flip, u64(args.flip_arg));
}

void ApplyVblank(const VoVblankArgs& args) {
    VideoOutPort* port = g.port;
    if (!port) {
        return;
    }
    // As Display::VblankThread does in-process.
    std::scoped_lock lock{port->vblank_mutex};
    TriggerEvents(port->vblank_queues, VO::InternalEvent::Vblank, args.event_count);
    std::memcpy(&port->vblank_status, args.status, VblankStatusBytes);
    port->vblank_cv.notify_all();
}

// ---- messages from bb-gpu -------------------------------------------------------------------

/// The button prompts' label for an input (runtime_pad_prompt), for bb-gpu. On the reader thread,
/// not the protection thread: a guest thread holding the pad's or the file table's lock may wait
/// on a write fault, whose protections that thread applies.
void PadPrompt(const Message& message, auto&& reply) {
    auto args = Payload<PadPromptArgs>(message);
    args.name[sizeof(args.name) - 1] = 0;
    PadPromptReply out{};
    out.kind = runtime_pad_prompt(args.name, args.prefer_mouse, out.label, int(sizeof(out.label)));
    out.label[sizeof(out.label) - 1] = 0;
    reply(&out, sizeof(out));
}

/// Where a game file is on this computer (runtime_file_host_path), for bb-gpu: only the game's
/// own files (/app0/), nothing else of the guest's file system.
void FileHostPath(const Message& message, auto&& reply) {
    auto args = Payload<PathArgs>(message);
    args.path[sizeof(args.path) - 1] = 0;
    PathReply out{-1, 0, {}};
    if (std::strncmp(args.path, "/app0/", 6) == 0 && !std::strstr(args.path, "..")) {
        out.result = runtime_file_host_path(args.path, out.path, sizeof(out.path));
        out.path[sizeof(out.path) - 1] = 0;
    }
    reply(&out, sizeof(out));
}


/// The reader thread never calls bb-gpu itself.
void Handle(const Message& message, auto&& reply) {
    switch (message.type) {
    case MsgIrq: {
        const u64 line = Payload<ValueArgs>(message).value;
        if (line < BbIrq::LineCount) {
            BbIrq::Raise(BbIrq::Line(line));
        }
        break;
    }
    case MsgFramesRetired:
        BbGnm::NoteFramesRetired(Payload<ValueArgs>(message).value);
        break;
    case MsgReleaseSubmissionLock:
        BbGnm::ReleaseSubmissionLock();
        break;
    case MsgVoFlipStatus:
        ApplyFlipStatus(Payload<VoStatusArgs>(message));
        break;
    case MsgVoVblank:
        ApplyVblank(Payload<VoVblankArgs>(message));
        break;
    case MsgRestart:
        std::printf("GPU process: restart requested\n");
        runtime_restart();
        break;
    case MsgGameStats: {
        // The frame stats come from bb-gpu; the guest's threads and waits are this process's.
        const auto args = Payload<StatsArgs>(message);
        if (args.frames > 0) {
            const double frames = args.frames;
            std::printf("Gnm: %.1f sceGnmSubmitDone/frame, %.1f sceGnmAreSubmitsAllowed/frame, "
                        "%.1f refused/frame; guest waited for the previous frame %.1fx %.2f "
                        "ms/frame\n",
                        BbStats::submit_done_calls.exchange(0) / frames,
                        BbStats::submits_allowed_queries.exchange(0) / frames,
                        BbStats::submits_refused.exchange(0) / frames,
                        BbStats::gnm_frame_waits.exchange(0) / frames,
                        BbStats::gnm_frame_wait_ns.exchange(0) / (1e6 * frames));
        }
        BbKernel::ReportEqueueWaits(args.frames);
        runtime_wait_report(args.frames);
#ifdef __APPLE__
        if (const std::string threads = BbThreads::ReportProcessThreads(args.window_s);
            !threads.empty()) {
            std::printf("Game process: %s\n", threads.c_str());
        }
#endif
        break;
    }
    case MsgPadPrompt:
        PadPrompt(message, reply);
        break;
    case MsgFileHostPath:
        FileHostPath(message, reply);
        break;
    case MsgGameEffects: {
        // bb-gpu's settings menu switched effects: the hot ones are switched in the game's code
        // here (BbGuestHooks::ApplyGameEffects compares with what it applied last).
        const u64 mask = Payload<ValueArgs>(message).value;
        auto& settings = BbSettings::Get();
        for (int e = 0; e < BbSettings::EffectCount; ++e) {
            settings.effects[e] = ((mask >> e) & 1) != 0;
        }
        settings.fxaa_forced = ((mask >> 63) & 1) != 0; // bb-gpu's SyncAntiAliasing
        BbGuestHooks::ApplyGameEffects();
        break;
    }
    case MsgFrameLimit:
        BbFrameRate::SetLimit(int(std::min<u64>(Payload<ValueArgs>(message).value, 1000)));
        break;
    case MsgDisplayAspect: {
        // SetDisplayAspect takes only 0.5..8 as another aspect ratio (else the game's 16:9).
        const u32 bits = u32(Payload<ValueArgs>(message).value);
        float aspect = 0.0f;
        std::memcpy(&aspect, &bits, sizeof(aspect));
        BbGuestHooks::SetDisplayAspect(std::isfinite(aspect) ? aspect : 0.0f);
        break;
    }
    case MsgMonitor: {
        const auto size = BbSettings::UnpackSize(u32(Payload<ValueArgs>(message).value));
        if (size.width >= 320 && size.height >= 200) {
            BbSettings::NoteMonitor(size.width, size.height);
        }
        break;
    }
    case MsgGameSettings: {
        // bb-gpu's settings menu; the game's code here reads them (the camera hook, the light grid).
        const auto args = Payload<GameSettingsArgs>(message);
        auto& settings = BbSettings::Get();
        for (int i = 0; i < 4; ++i) {
            const float speed = args.camera_speed[i];
            if (std::isfinite(speed)) {
                settings.camera_speed[i].store(std::clamp(speed, 0.25f, 2.0f),
                                               std::memory_order_relaxed);
            }
        }
        settings.light_grid_opt.store(args.light_grid_opt != 0, std::memory_order_relaxed);
        break;
    }
    case MsgGpuFailed: {
        const auto args = Payload<TextArgs>(message);
        std::fprintf(stderr, "GPU process: %.*s\n", int(sizeof(args.text)), args.text);
        break;
    }
    default:
        std::fprintf(stderr, "GPU process: unknown message %u\n", message.type);
        break;
    }
}

void ReaderThread() {
    Common::SetCurrentThreadName("bb:remote");
    for (;;) {
        g.channel->ReceiveOne([](const Message& message, auto&& reply) { Handle(message, reply); },
                              [] { return false; });
    }
}

/// A protection or write trap from bb-gpu (ProtectArgs, ProtectTrap).
void ApplyProtect(const ProtectArgs& args) {
    if (args.read & ProtectTrap) {
        // runtime_memory_trap trusts its caller (in-process, the GPU library): a range that wraps
        // or reaches past its table would write outside it.
        constexpr u64 TrapLimit = 1ull << 40;
        if (!args.size || args.address >= TrapLimit || args.size > TrapLimit - args.address) {
            std::fprintf(stderr, "GPU process: refused a write trap at %#llx+%#llx\n",
                         static_cast<unsigned long long>(args.address),
                         static_cast<unsigned long long>(args.size));
            return;
        }
        // bbport (protocol 12): in bb-gpu's order (bb_trap_copy.h); a lift only where no newer
        // change reached the page.
        runtime_memory_trap_seq(args.address, args.size, args.read & ~(ProtectTrap | ProtectLift),
                                args.write ? 1 : 0, args.seq, (args.read & ProtectLift) ? 1 : 0);
    } else {
        runtime_memory_gpu_protect(args.address, args.size, int(args.read), int(args.write));
    }
}

/// bb-gpu's page protection calls (ControlBlock::protect): only the runtime's lock is taken here.
void ProtectThread() {
    Common::SetCurrentThreadName("bb:remote-protect");
    for (;;) {
        g.protect->ReceiveOne(
            [](const Message& message, auto&& reply) {
                if (message.type == MsgProtect) {
                    ApplyProtect(Payload<ProtectArgs>(message));
                } else if (message.type == MsgNotePinned) {
                    // bb-gpu's lazy import of these pages (pin_mutex only, no runtime lock).
                    const auto args = Payload<ChunkArgs>(message);
                    runtime_memory_note_pinned(args.phys, args.size);
                }
                const ResultReply result{0, 0};
                reply(&result, sizeof(result));
            },
            [] { return false; });
    }
}

/// runtime_pad.c's prompt state for bb-gpu's button prompts, and whether a Steam pad echoes keys
/// for its window (SharedState): once a frame.
void PublishPadState() {
    if (!g.control) {
        return;
    }
    auto& s = g.control->state;
    s.prompt_device.store(runtime_pad_prompt_device(), std::memory_order_relaxed);
    s.pad_mouse_look.store(runtime_pad_mouse_look(), std::memory_order_relaxed);
    s.steam_only.store(runtime_pad_steam_only ? runtime_pad_steam_only() : 0,
                       std::memory_order_relaxed);
    s.prompt_generation.store(runtime_pad_prompt_generation(), std::memory_order_release);
}

/// Ends the game when bb-gpu ends (its window was closed, or it failed).
void WatchThread() {
    Common::SetCurrentThreadName("bb:remote-watch");
    int status = 0;
    while (waitpid(g.child, &status, 0) < 0 && errno == EINTR) {
    }
    int code = 1;
    if (WIFEXITED(status)) {
        code = WEXITSTATUS(status);
        std::printf("GPU process exited (%d)\n", code);
    } else if (WIFSIGNALED(status)) {
        std::fprintf(stderr, "GPU process ended by signal %d\n", WTERMSIG(status));
    }
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(code);
}

// ---- starting bb-gpu ------------------------------------------------------------------------

std::string ExecutableDir() {
    char path[4096] = {};
#ifdef __APPLE__
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        return ".";
    }
#else
    const ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (n <= 0) {
        return ".";
    }
    path[n] = 0;
#endif
    std::string dir = path;
    const auto slash = dir.rfind('/');
    return slash == std::string::npos ? "." : dir.substr(0, slash);
}

/// A path for the log: the home folder as ~ (logs get shared).
std::string ForLog(const std::string& path) {
    const char* home = std::getenv("HOME");
    const std::size_t n = home ? std::strlen(home) : 0;
    if (n > 1 && path.compare(0, n, home) == 0 && (path.size() == n || path[n] == '/')) {
        return "~" + path.substr(n);
    }
    return path;
}

bool Exists(const std::string& path) {
    struct stat info{};
    return stat(path.c_str(), &info) == 0;
}

std::string DepsDir() {
    if (const char* env = std::getenv("BB_GPU_DEPS")) {
        return env;
    }
    return ExecutableDir() + "/../deps-arm64";
}

std::string Program() {
    if (const char* env = std::getenv("BB_GPU_PROCESS")) {
        return env;
    }
    return ExecutableDir() + "/gpu-arm64/bb-gpu";
}

bool Spawn() {
    const std::string dir = ExecutableDir();
    std::string program = Program(); // argv wants it writable
    if (!Exists(program)) {
        std::fprintf(stderr,
                     "GPU process: %s missing (build it with tools/macos/build_gpu_native.sh)\n",
                     ForLog(program).c_str());
        return false;
    }
    // Its own libraries: the arm64 dependency prefix instead of the game process's x86-64 one.
    const std::string deps = DepsDir();
    const bool own_deps = Exists(deps + "/lib");
    std::vector<std::string> env_strings;
    for (char** e = environ; e && *e; ++e) {
        const std::string_view entry{*e};
        // The game's Vulkan loader and driver are x86-64: bb-gpu gets its own (the release app
        // names the loader by path, BB_VULKAN_LIBRARY, as DYLD_LIBRARY_PATH doesn't survive bash).
        if (own_deps && (entry.starts_with("DYLD_LIBRARY_PATH=") ||
                         entry.starts_with("VK_DRIVER_FILES=") ||
                         entry.starts_with("VK_ICD_FILENAMES=") ||
                         entry.starts_with("BB_VULKAN_LIBRARY=") ||
                         entry.starts_with("SDL_VULKAN_LIBRARY="))) {
            continue;
        }
        env_strings.emplace_back(entry);
    }
    if (own_deps) {
        // A checkout's libraries are found through this; the release app's through @rpath
        // (package.sh), where it would only let them stand in for the system's by name.
        if (!Exists(deps + "/../.packaged")) {
            env_strings.push_back("DYLD_LIBRARY_PATH=" + deps + "/lib");
        }
        env_strings.push_back("VK_DRIVER_FILES=" + deps +
                              "/lib/kosmickrisp/kosmickrisp_mesa_icd.json");
        if (const std::string loader = deps + "/lib/libvulkan.1.dylib"; Exists(loader)) {
            env_strings.push_back("BB_VULKAN_LIBRARY=" + loader);
            env_strings.push_back("SDL_VULKAN_LIBRARY=" + loader);
        }
    }
    std::vector<char*> envp;
    for (auto& entry : env_strings) {
        envp.push_back(entry.data());
    }
    envp.push_back(nullptr);

    std::string fd_arg = std::to_string(g.pool_fd), bytes_arg = std::to_string(g.pool_bytes),
                offset_arg = std::to_string(g.host_offset);
    std::vector<char*> argv{program.data(), const_cast<char*>("--pool-fd"), fd_arg.data(),
                            const_cast<char*>("--pool-bytes"), bytes_arg.data(),
                            const_cast<char*>("--control-offset"), offset_arg.data(), nullptr};
#ifdef __APPLE__
    // bb-gpu gets the pool (by its number) and the standard streams, and no other descriptor of
    // this process (game files, sockets, the log pipe's other copies).
    posix_spawnattr_t attr;
    posix_spawn_file_actions_t actions;
    posix_spawnattr_init(&attr);
    posix_spawn_file_actions_init(&actions);
    int error = posix_spawnattr_setflags(&attr, POSIX_SPAWN_CLOEXEC_DEFAULT);
    for (const int fd : {0, 1, 2, g.pool_fd}) {
        // A standard stream the game was started without is simply not passed on.
        if (error == 0 && (fd == g.pool_fd || fcntl(fd, F_GETFD) != -1)) {
            error = posix_spawn_file_actions_addinherit_np(&actions, fd);
        }
    }
    if (error == 0) {
        error = posix_spawn(&g.child, program.c_str(), &actions, &attr, argv.data(), envp.data());
    }
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
#else
    // shm_open sets close-on-exec: bb-gpu inherits the pool by its number.
    if (fcntl(g.pool_fd, F_SETFD, 0) != 0) {
        std::perror("GPU process: pool descriptor");
        return false;
    }
    const int error = posix_spawn(&g.child, program.c_str(), nullptr, nullptr, argv.data(),
                                  envp.data());
#endif
    if (error != 0) {
        std::fprintf(stderr, "GPU process: cannot start %s: %s\n", ForLog(program).c_str(),
                     std::strerror(error));
        return false;
    }
    std::printf("GPU process: started %s (pid %d)%s\n",
                program.substr(program.find_last_of('/') + 1).c_str(), int(g.child),
                own_deps ? "" : " without an arm64 dependency prefix");
    return true;
}

} // namespace

bool Built() {
    // The default only where it can run: an executable bb-gpu and its arm64 libraries.
    const std::string program = Program();
    return Exists(program) && access(program.c_str(), X_OK) == 0 && Exists(DepsDir() + "/lib");
}

bool Init(const BbGpuConfig& config) {
    int fd = -1;
    u64 host_offset = 0, total = 0;
    if (runtime_memory_host_pool(HostSpanBytes, &fd, &host_offset, &total) != 0) {
        std::fprintf(stderr, "GPU process: the game's memory pool exists already\n");
        return false;
    }
    g.pool_fd = fd;
    g.pool_bytes = total;
    g.host_offset = host_offset;
    g.host_next = host_offset + ControlBytes;
    g.host_end = host_offset + HostSpanBytes;

    // The control block, at the same address in both processes (no MAP_FIXED: only if free).
    void* at = mmap(reinterpret_cast<void*>(ControlAddress), ControlBytes, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, static_cast<off_t>(host_offset));
    if (at == MAP_FAILED || at != reinterpret_cast<void*>(ControlAddress)) {
        std::fprintf(stderr, "GPU process: cannot map the control block at %#llx\n",
                     static_cast<unsigned long long>(ControlAddress));
        if (at != MAP_FAILED) {
            munmap(at, ControlBytes);
        }
        return false;
    }
    g.control = static_cast<ControlBlock*>(at);
    Channel::Init(&g.control->channel);
    Channel::Init(&g.control->protect);
    Channel::Init(&g.control->fresh);
    Channel::Init(&g.control->locked);
    new (&g.control->state) SharedState{};
    g.channel = std::make_unique<Channel>(&g.control->channel, Side::Frontend);
    g.protect = std::make_unique<Channel>(&g.control->protect, Side::Frontend);
    g.fresh = std::make_unique<Channel>(&g.control->fresh, Side::Frontend);
    g.locked = std::make_unique<Channel>(&g.control->locked, Side::Frontend);
    if (bb_env_on("BB_REMOTE_FRESH_SYNC")) {
        g.fresh_sync = true;
        std::printf("GPU process: fresh ranges are calls (BB_REMOTE_FRESH_SYNC=1)\n");
    }

    if (!Spawn()) {
        return false;
    }
    std::thread(ReaderThread).detach();
    std::thread(ProtectThread).detach();
    std::thread(WatchThread).detach();

    HelloArgs hello{};
    hello.protocol = ProtocolVersion;
    hello.page_size = u32(sysconf(_SC_PAGESIZE));
    hello.pool_size = g.pool_bytes;
    hello.guest_begin = GuestBegin;
    hello.guest_end = GuestEnd;
    hello.low_begin = LowBegin;
    hello.low_end = LowEnd;
    const u64 counter = runtime_process_time_counter();
    hello.start_monotonic_ns = MonotonicNs() - counter;
    hello.tsc_frequency = runtime_tsc_frequency();
    hello.tsc_sample = Common::FencedRDTSC();
    hello.tsc_sample_monotonic_ns = MonotonicNs();
    HelloReply welcome{};
    g.channel->Call(MsgHello, &hello, sizeof(hello), &welcome, sizeof(welcome));
    if (welcome.error != 0 || welcome.protocol != ProtocolVersion) {
        std::fprintf(stderr, "GPU process: refused to start (error %d, protocol %u)\n",
                     welcome.error, welcome.protocol);
        return false;
    }
    // bb-gpu's own memory inside the guest range: the guest's mappings go around it.
    u64 taken_bytes = 0;
    for (u32 i = 0; i < std::min(welcome.taken_count, MaxTakenRanges); ++i) {
        const u64 begin = welcome.taken[i][0], end = welcome.taken[i][1];
        if (runtime_memory_exclude(begin, end) != 0) {
            std::fprintf(stderr, "GPU process: the game has memory at %#llx-%#llx already\n",
                         static_cast<unsigned long long>(begin),
                         static_cast<unsigned long long>(end));
            return false;
        }
        taken_bytes += end - begin;
    }
    if (taken_bytes) {
        std::printf("GPU process: %llu MiB of the guest range kept for its own memory\n",
                    static_cast<unsigned long long>(taken_bytes >> 20));
    }

    // From here on bb-gpu follows the guest's mappings (the existing ones first).
    runtime_memory_set_mirror_hook(MirrorHook);
    runtime_memory_set_gpu_hooks(nullptr, nullptr, InvalidateHook);
    runtime_memory_set_note_write_hook(NoteWriteHook);

    InitArgs init{};
    init.sdk_version = config.sdk_version;
    init.psf_attributes = config.psf_attributes;
    init.width = config.width;
    init.height = config.height;
    CopyString(init.title, sizeof(init.title), config.title);
    CopyString(init.serial, sizeof(init.serial), config.serial);
    CopyString(init.user_dir, sizeof(init.user_dir), config.user_dir);
    ResultReply result{-1, 0};
    g.channel->Call(MsgInit, &init, sizeof(init), &result, sizeof(result));
    if (result.result != 0) {
        std::fprintf(stderr, "GPU process: GPU initialization failed (%d)\n", result.result);
        return false;
    }
    if (!InstallChunks()) {
        return false;
    }
    front_active = true;
    std::printf("GPU process: native GPU core ready (page size %u here, %u there)\n",
                hello.page_size, welcome.page_size);
    return true;
}

// ---- GnmDriver ------------------------------------------------------------------------------

bool SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb, bool force_copy) {
    struct {
        SubmitGfxArgs args;
        u32 words[(MaxPayload - sizeof(SubmitGfxArgs)) / 4];
    } message{};
    message.args.force_copy = force_copy ? 1 : 0;
    std::size_t words = 0;
    // Host data (the init sequences) travels inline; guest command buffers by address.
    const auto add = [&](std::span<const u32> span, u64& address, u32& dwords, u32& inline_flag) {
        dwords = u32(span.size());
        if (span.empty()) {
            return true;
        }
        // force_copy words always travel inline (bb-gpu refuses them by address).
        if (!force_copy && Shared(span.data(), span.size_bytes())) {
            address = reinterpret_cast<u64>(span.data());
            return true;
        }
        if (words + span.size() > std::size(message.words)) {
            return false;
        }
        std::memcpy(message.words + words, span.data(), span.size_bytes());
        words += span.size();
        inline_flag = 1;
        return true;
    };
    if (!add(dcb, message.args.dcb, message.args.dcb_dwords, message.args.inline_dcb) ||
        !add(ccb, message.args.ccb, message.args.ccb_dwords, message.args.inline_ccb)) {
        std::fprintf(stderr, "GPU process: a %zu+%zu dword submission in host memory is too "
                     "large to forward; dropped\n", dcb.size(), ccb.size());
        return false;
    }
    g.channel->Send(MsgSubmitGfx, &message, u32(sizeof(SubmitGfxArgs) + words * 4));
    return true;
}

void SubmitAsc(u32 gnm_vqid, std::span<const u32> acb) {
    if (!acb.empty() && !Shared(acb.data(), acb.size_bytes())) {
        std::fprintf(stderr, "GPU process: compute submission outside shared memory; dropped\n");
        return;
    }
    const SubmitAscArgs args{reinterpret_cast<u64>(acb.data()), gnm_vqid, u32(acb.size())};
    g.channel->Send(MsgSubmitAsc, &args, sizeof(args));
}

void SubmitDone(u64 frame) {
    PublishPadState();
    const ValueArgs args{frame};
    g.channel->Send(MsgSubmitDone, &args, sizeof(args));
    // The game's System menu pages live in this process (their hooks are in the game's code), the
    // renderer's settings in bb-gpu: a change there is saved to bbport.ini and bb-gpu reloads it.
    // Once a frame, on the game's thread (bb-gpu's presenter polls in-process builds).
    if (BbGameMenu::Poll()) {
        g.channel->Send(MsgSettingsChanged, nullptr, 0);
    }
}

bool IsGpuIdle() {
    ResultReply result{};
    g.channel->Call(MsgIsGpuIdle, nullptr, 0, &result, sizeof(result));
    return result.result != 0;
}

s32 MapComputeQueue(u64 ring_base, u32* read_ptr, u32 ring_size_dw, u32 pipe_id) {
    const ComputeQueueArgs args{ring_base, reinterpret_cast<u64>(read_ptr), ring_size_dw, pipe_id};
    ResultReply result{-1, 0};
    g.channel->Call(MsgMapComputeQueue, &args, sizeof(args), &result, sizeof(result));
    if (result.result < 0) {
        return -1;
    }
    if (std::size_t(result.result) >= g.asc.size()) {
        UnmapComputeQueue(u32(result.result)); // not expected: bb-gpu numbers fewer queues
        return -1;
    }
    {
        std::scoped_lock lock{g.asc_mutex};
        g.asc[result.result] = {ring_base, ring_size_dw};
    }
    return result.result;
}

void UnmapComputeQueue(u32 index) {
    const ValueArgs args{index};
    g.channel->Send(MsgUnmapComputeQueue, &args, sizeof(args));
    if (index < g.asc.size()) {
        std::scoped_lock lock{g.asc_mutex};
        g.asc[index] = {};
    }
}

AscQueue GetAscQueue(u32 index) {
    std::scoped_lock lock{g.asc_mutex};
    return index < g.asc.size() ? g.asc[index] : AscQueue{};
}

// ---- VideoOut -------------------------------------------------------------------------------

void SetVideoOutPort(VideoOutPort* port) {
    g.port = port;
}

u64* VideoOutLabels() {
    return g.control ? g.control->state.vo_labels : nullptr;
}

int VoOpen() {
    VoReply reply{};
    g.channel->Call(MsgVoOpen, nullptr, 0, &reply, sizeof(reply));
    ApplyVoState(reply.state);
    return reply.result;
}

void VoClose(s32 handle) {
    const VoArgs args{handle, 0};
    ResultReply result{};
    g.channel->Call(MsgVoClose, &args, sizeof(args), &result, sizeof(result));
}

int VoRegisterBuffers(s32 start, void* const* addresses, s32 count,
                      const BbVideoOut::BufferAttribute* attribute) {
    VoRegisterArgs args{};
    args.handle = 1;
    args.start = start;
    args.count = count;
    std::memcpy(args.attribute, attribute, BufferAttributeBytes);
    for (s32 i = 0; i < std::min<s32>(count, s32(MaxDisplayBuffers)); ++i) {
        args.addresses[i] = reinterpret_cast<u64>(addresses[i]);
    }
    VoReply reply{};
    g.channel->Call(MsgVoRegisterBuffers, &args, sizeof(args), &reply, sizeof(reply));
    ApplyVoState(reply.state);
    return reply.result;
}

int VoUnregisterBuffers(s32 attribute_index) {
    const VoArgs args{1, attribute_index};
    VoReply reply{};
    g.channel->Call(MsgVoUnregisterBuffers, &args, sizeof(args), &reply, sizeof(reply));
    ApplyVoState(reply.state);
    return reply.result;
}

int VoChangeBufferAttribute(s32 attribute_index, const BbVideoOut::BufferAttribute* attribute) {
    VoAttributeArgs args{};
    args.handle = 1;
    args.index = attribute_index;
    std::memcpy(args.attribute, attribute, BufferAttributeBytes);
    VoReply reply{};
    g.channel->Call(MsgVoChangeAttribute, &args, sizeof(args), &reply, sizeof(reply));
    ApplyVoState(reply.state);
    return reply.result;
}

bool VoSubmitFlip(s32 index, s64 flip_arg) {
    const VoFlipArgs args{1, index, 1, 0, flip_arg};
    VoReply reply{};
    g.channel->Call(MsgVoSubmitFlip, &args, sizeof(args), &reply, sizeof(reply));
    ApplyVoState(reply.state);
    return reply.result != 0;
}

s32 VoSubmitEopFlip(s32 handle, u32 buf_id, u32 mode, s64 flip_arg) {
    const VoFlipArgs args{handle, s32(buf_id), mode, 0, flip_arg};
    VoReply reply{};
    g.channel->Call(MsgVoSubmitEopFlip, &args, sizeof(args), &reply, sizeof(reply));
    return reply.result;
}

void VoSetFlipRate(s32 handle, s32 rate) {
    const VoArgs args{handle, rate};
    g.channel->Send(MsgVoSetFlipRate, &args, sizeof(args));
}

void VoSetHdr(s32 handle, bool hdr) {
    const VoArgs args{handle, hdr ? 1 : 0};
    g.channel->Send(MsgVoSetHdr, &args, sizeof(args));
}

bool VoIsHdrSupported() {
    ResultReply result{};
    g.channel->Call(MsgVoIsHdrSupported, nullptr, 0, &result, sizeof(result));
    return result.result != 0;
}

void VoSetGamma(float gamma) {
    const GammaArgs args{gamma, 0};
    g.channel->Send(MsgVoSetGamma, &args, sizeof(args));
}

// ---- faults and memory ----------------------------------------------------------------------

int HandleFault(void* context, void* address) {
    const auto addr = reinterpret_cast<u64>(address);
    // Only the guest's own mappings are tracked; anything else is a real crash.
    if (!runtime_memory_is_mapped(addr & ~u64(4095), 1)) {
        return 0;
    }
    const bool lock_held = runtime_memory_lock_held() != 0;
    const FaultArgs args{addr,
                         BbPortable::Rip(context),
                         Common::IsWriteError(context) ? 1u : 0u,
                         lock_held ? 1u : 0u,
                         runtime_memory_trap_reasons(addr),
                         0};
    FaultReply reply{};
    // bbport: the protections bb-gpu's handlers decide for this fault are applied here, by this
    // thread while it waits, each before the handler goes on (asked over the call's reply slot),
    // as this thread's own handler does in one process; with the runtime's lock held here too
    // (the protection thread would wait for it). See FaultReply. bbport (protocol 12): holding
    // that lock, on a channel of its own (remote_back.cpp FaultThread): on the main one it could
    // wait behind a fault whose thread waits for this one's lock.
    Channel& channel = lock_held ? *g.locked : *g.channel;
    channel.Call(MsgWriteFault, &args, sizeof(args), &reply, sizeof(reply),
                 [](const Message& message, auto&& answer) {
                     if (message.type == MsgProtect) {
                         ApplyProtect(Payload<ProtectArgs>(message));
                     }
                     const ResultReply result{0, 0};
                     answer(&result, sizeof(result));
                 });
    return reply.handled;
}

void InvalidateMemory(u64 address, u64 size) {
    SendRange(MsgInvalidate, address, size);
}

void FreshRange(u64 address, u64 size) {
    // bbport: BB_REMOTE_FRESH_SYNC=1: back once bb-gpu's buffer cache has the range (its GPU data
    // there no longer GPU-modified), before this thread writes the new block, as in one process;
    // a send lets a readback of the old data land after those writes. Served by bb-gpu's
    // FreshThread (deadlock-free: see there); this thread runs the game's code (no runtime or HLE
    // lock held).
    if (g.fresh_sync) {
        // After what this process sent bb-gpu before it (FreshThread waits for that).
        const FreshArgs args{address, size, g.channel->Written()};
        ResultReply result{};
        g.fresh->Call(MsgFreshRange, &args, sizeof(args), &result, sizeof(result));
        return;
    }
    SendRange(MsgFreshRange, address, size);
}

void DeadRange(u64 address, u64 size) {
    SendRange(MsgDeadRange, address, size);
}

void LiveRange(u64 address, u64 size) {
    SendRange(MsgLiveRange, address, size);
}

int ShareRange(void* address, u64 size, int prot) {
    const auto begin = reinterpret_cast<u64>(address);
    if (begin % SharePage || !size) {
        std::fprintf(stderr, "GPU process: cannot share %#llx (not on a 16 KiB page)\n",
                     static_cast<unsigned long long>(begin));
        return 0; // not fatal: bb-gpu reads zeros there
    }
    const u64 bytes = (size + SharePage - 1) & ~(SharePage - 1);
    u64 offset = 0;
    {
        std::scoped_lock lock{g.share_mutex};
        if (g.host_next + bytes > g.host_end) {
            std::fprintf(stderr, "GPU process: no room to share %llu MiB of the game's data\n",
                         static_cast<unsigned long long>(bytes >> 20));
            return -1;
        }
        offset = g.host_next;
        g.host_next += bytes;
    }
    // Same contents, now in the pool: copy out, map the shared pages over it, copy back.
    std::vector<u8> contents(bytes);
    std::memcpy(contents.data(), address, bytes);
    void* at = mmap(address, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, g.pool_fd,
                    static_cast<off_t>(offset));
    if (at == MAP_FAILED) {
        std::fprintf(stderr, "GPU process: sharing %#llx+%#llx failed: %s\n",
                     static_cast<unsigned long long>(begin),
                     static_cast<unsigned long long>(bytes), std::strerror(errno));
        return -1;
    }
    std::memcpy(address, contents.data(), bytes);
    if (mprotect(address, bytes, prot) != 0) {
        std::perror("GPU process: protecting shared game data");
        return -1;
    }
    {
        std::scoped_lock lock{g.share_mutex};
        g.shared_ranges.emplace_back(begin, begin + bytes);
    }
    const int guest_prot = ((prot & PROT_READ) ? 0x1 : 0) | ((prot & PROT_WRITE) ? 0x2 : 0);
    const MapArgs args{begin, bytes, offset, u32(MapShared), guest_prot, -1, 0};
    g.channel->Send(MsgMapMemory, &args, sizeof(args));
    std::printf("GPU process: shared %llu KiB of the game's data at %#llx\n",
                static_cast<unsigned long long>(bytes >> 10),
                static_cast<unsigned long long>(begin));
    return 0;
}

// ---- window -----------------------------------------------------------------------------------

void ShowTrophy(char grade, const char* name, const char* icon, float delay) {
    TrophyArgs args{};
    CopyString(args.name, sizeof(args.name), name);
    CopyString(args.icon, sizeof(args.icon), icon);
    args.delay = delay;
    args.grade = grade;
    g.channel->Send(MsgShowTrophy, &args, sizeof(args));
}

int WorldActive() {
    return g.control && g.control->state.world_active.load(std::memory_order_relaxed) ? 1 : 0;
}

unsigned long long WorldUs() {
    return g.control ? g.control->state.world_us.load(std::memory_order_relaxed) : 0;
}

int OverlayCapturesInput() {
    return g.control && g.control->state.overlay_captures_input.load(std::memory_order_acquire)
               ? 1
               : 0;
}

const bool* KeyboardState() {
    static_assert(sizeof(bool) == 1);
    return g.control ? reinterpret_cast<const bool*>(g.control->state.keyboard) : nullptr;
}

void TakeKeyLatch(volatile std::uint8_t* latch, int count) {
    if (!g.control || !latch) {
        return;
    }
    auto& words = g.control->state.key_latch;
    for (int w = 0; w < int(std::size(words)); ++w) {
        u64 bits = words[w].exchange(0, std::memory_order_relaxed);
        while (bits) {
            const int code = w * 64 + std::countr_zero(bits);
            bits &= bits - 1;
            if (code < count) {
                latch[code] = 1;
            }
        }
    }
}

void MouseTake(double* dx, double* dy, int* wheel_up, int* wheel_down) {
    *dx = *dy = 0;
    *wheel_up = *wheel_down = 0;
    if (!g.control) {
        return;
    }
    auto& s = g.control->state;
    *dx = double(s.mouse_dx.exchange(0, std::memory_order_relaxed)) / 1024.0;
    *dy = double(s.mouse_dy.exchange(0, std::memory_order_relaxed)) / 1024.0;
    // At most a second of notches (the pad treats each as a short press).
    *wheel_up = int(std::min<u32>(s.wheel_up.exchange(0, std::memory_order_relaxed), 64));
    *wheel_down = int(std::min<u32>(s.wheel_down.exchange(0, std::memory_order_relaxed), 64));
}

int MouseCaptured() {
    return g.control && g.control->state.mouse_captured.load(std::memory_order_relaxed) ? 1 : 0;
}

u32 MouseButtons() {
    return g.control ? g.control->state.mouse_buttons.load(std::memory_order_relaxed) : 0;
}

void MouseCapture() {
    if (g.control) {
        g.control->state.mouse_capture_requests.fetch_add(1, std::memory_order_relaxed);
    }
}

void MouseLookEnable(int enabled) {
    if (g.control) {
        g.control->state.mouse_look.store(enabled ? 2 : 1, std::memory_order_relaxed);
    }
}

int TextInputBegin(const char* initial, const char* prompt) {
    TextInputArgs args{};
    CopyString(args.initial, sizeof(args.initial), initial);
    CopyString(args.prompt, sizeof(args.prompt), prompt);
    ResultReply result{};
    g.channel->Call(MsgTextInputBegin, &args, sizeof(args), &result, sizeof(result));
    return result.result;
}

int TextInputPoll(char* out, u64 size) {
    TextInputReply reply{};
    reply.state = 2;
    g.channel->Call(MsgTextInputPoll, nullptr, 0, &reply, sizeof(reply));
    if (size) {
        const std::size_t n = std::min<std::size_t>(strnlen(reply.text, sizeof(reply.text)),
                                                     size - 1);
        std::memcpy(out, reply.text, n);
        out[n] = 0;
    }
    return reply.state;
}

} // namespace Front
} // namespace BbRemote
