// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the game process's side of the native GPU process split (docs/macos-native-gpu.md).
// With BB_NATIVE_GPU=1, bb-probe (x86-64 under Rosetta) starts bb-gpu (native arm64), shares the
// guest's memory pool with it and forwards what reaches the GPU core: command buffer submissions,
// the VideoOut driver's calls, page-tracking faults and the runtime's memory events. bb-gpu sends
// back interrupts, flip and vblank status, retired frames and page protection changes.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
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
#include <unistd.h>
#include <vector>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include "bbport_game_menu.h"
#include "bbport_portable.h"
#include "bbport_threads.h"
#include "bbport_toggles.h"
#include "common/rdtsc.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/libraries/kernel/equeue.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/video_out.h"
#include "core/platform.h"
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
typedef void (*RuntimeGpuRange)(uintptr_t address, uint64_t size);
void runtime_memory_set_gpu_hooks(RuntimeGpuRange map, RuntimeGpuRange unmap,
                                  RuntimeGpuRange invalidate);
void runtime_memory_set_note_write_hook(RuntimeGpuRange note);
uint64_t runtime_process_time_counter(void);
uint64_t runtime_tsc_frequency(void);
void runtime_restart(void);
void runtime_wait_report(double frames);
}

namespace Libraries::GnmDriver {
void NoteFramesRetired(u64 frames);
void ReleaseSubmissionLock();
} // namespace Libraries::GnmDriver

namespace BbRemote {

using Libraries::VideoOut::VideoOutPort;
namespace VO = Libraries::VideoOut;

static_assert(sizeof(VO::FlipStatus) == FlipStatusBytes);
static_assert(sizeof(VO::SceVideoOutVblankStatus) == VblankStatusBytes);
static_assert(sizeof(VO::SceVideoOutResolutionStatus) == ResolutionBytes);
static_assert(sizeof(VO::BufferAttribute) == BufferAttributeBytes);
static_assert(sizeof(VO::VideoOutBuffer) == VideoOutBufferBytes);
static_assert(sizeof(VO::BufferAttributeGroup) == AttributeGroupBytes);
static_assert(VO::MaxDisplayBuffers == MaxDisplayBuffers &&
              VO::MaxDisplayBufferGroups == MaxDisplayBufferGroups);

bool Requested() {
    static const bool requested = [] {
        const char* env = std::getenv("BB_NATIVE_GPU");
        return env && env[0] == '1';
    }();
    return requested && !back_active;
}

namespace Front {
namespace {

constexpr u64 SharePage = 16384; // shared mappings are whole 16 KiB pages, even under Rosetta

struct State {
    ControlBlock* control = nullptr;
    std::unique_ptr<Channel> channel, protect;
    int pool_fd = -1;
    u64 pool_bytes = 0, host_offset = 0, host_next = 0, host_end = 0;
    pid_t child = -1;
    VideoOutPort* port = nullptr;
    u64 flip_version = 0; ///< newest flip status applied (under port->port_mutex)
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

/// The port's state from bb-gpu (port_mutex taken here).
void ApplyVoState(const VoState& state) {
    VideoOutPort* port = g.port;
    if (!port) {
        return;
    }
    std::scoped_lock lock{port->port_mutex};
    if (state.version > g.flip_version) {
        g.flip_version = state.version;
        std::memcpy(&port->flip_status, state.flip_status, FlipStatusBytes);
    }
    std::memcpy(port->buffer_slots.data(), state.slots, sizeof(state.slots));
    std::memcpy(port->groups.data(), state.groups, sizeof(state.groups));
    std::memcpy(&port->resolution, state.resolution, ResolutionBytes);
    port->is_open = state.is_open != 0;
}

void ApplyFlipStatus(const VoStatusArgs& args) {
    VideoOutPort* port = g.port;
    if (!port) {
        return;
    }
    {
        std::scoped_lock lock{port->port_mutex};
        if (args.version > g.flip_version) {
            g.flip_version = args.version;
            std::memcpy(&port->flip_status, args.flip_status, FlipStatusBytes);
        }
    }
    if (!args.flip_event) {
        return;
    }
    // As VideoOutDriver::Flip does in-process.
    for (auto event : port->flip_events) {
        if (auto* equeue = Libraries::Kernel::GetEqueue(event)) {
            equeue->TriggerEvent(
                static_cast<u64>(VO::OrbisVideoOutInternalEventId::Flip),
                Libraries::Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(VO::OrbisVideoOutInternalEventId::Flip) |
                                        (args.flip_arg << 16)));
        }
    }
}

void ApplyVblank(const VoVblankArgs& args) {
    VideoOutPort* port = g.port;
    if (!port) {
        return;
    }
    // As VideoOutDriver::PresentThread does in-process.
    std::scoped_lock lock{port->vo_mutex};
    for (auto event : port->vblank_events) {
        if (auto* equeue = Libraries::Kernel::GetEqueue(event)) {
            equeue->TriggerEvent(
                static_cast<u64>(VO::OrbisVideoOutInternalEventId::Vblank),
                Libraries::Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(VO::OrbisVideoOutInternalEventId::Vblank) |
                                        (args.event_count << 16)));
        }
    }
    std::memcpy(&port->vblank_status, args.status, VblankStatusBytes);
    port->vblank_cv.notify_all();
}

// ---- messages from bb-gpu -------------------------------------------------------------------

/// The reader thread never calls bb-gpu itself.
void Handle(const Message& message, auto&& reply) {
    switch (message.type) {
    case MsgIrq:
        Platform::IrqC::Instance()->Signal(
            static_cast<Platform::InterruptId>(Payload<ValueArgs>(message).value));
        break;
    case MsgFramesRetired:
        Libraries::GnmDriver::NoteFramesRetired(Payload<ValueArgs>(message).value);
        break;
    case MsgReleaseSubmissionLock:
        Libraries::GnmDriver::ReleaseSubmissionLock();
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
        Libraries::Kernel::ReportEqueueWaits(args.frames);
        runtime_wait_report(args.frames);
#ifdef __APPLE__
        if (const std::string threads = BbThreads::ReportProcessThreads(args.window_s);
            !threads.empty()) {
            std::printf("Game process: %s\n", threads.c_str());
        }
#endif
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
        runtime_memory_trap(args.address, args.size, args.read & ~ProtectTrap, args.write ? 1 : 0);
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
                }
                const ResultReply result{0, 0};
                reply(&result, sizeof(result));
            },
            [] { return false; });
    }
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

bool Exists(const std::string& path) {
    struct stat info{};
    return stat(path.c_str(), &info) == 0;
}

bool Spawn() {
    const std::string dir = ExecutableDir();
    std::string program = dir + "/gpu-arm64/bb-gpu";
    if (const char* env = std::getenv("BB_GPU_PROCESS")) {
        program = env;
    }
    if (!Exists(program)) {
        std::fprintf(stderr,
                     "GPU process: %s missing (build it with tools/macos/build_gpu_native.sh)\n",
                     program.c_str());
        return false;
    }
    // Its own libraries: the arm64 dependency prefix instead of the game process's x86-64 one.
    std::string deps = dir + "/../deps-arm64";
    if (const char* env = std::getenv("BB_GPU_DEPS")) {
        deps = env;
    }
    const bool own_deps = Exists(deps + "/lib");
    std::vector<std::string> env_strings;
    for (char** e = environ; e && *e; ++e) {
        const std::string_view entry{*e};
        if (own_deps && (entry.starts_with("DYLD_LIBRARY_PATH=") ||
                         entry.starts_with("VK_DRIVER_FILES=") ||
                         entry.starts_with("VK_ICD_FILENAMES="))) {
            continue;
        }
        env_strings.emplace_back(entry);
    }
    if (own_deps) {
        env_strings.push_back("DYLD_LIBRARY_PATH=" + deps + "/lib");
        env_strings.push_back("VK_DRIVER_FILES=" + deps +
                              "/lib/kosmickrisp/kosmickrisp_mesa_icd.json");
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
        if (error == 0) {
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
        std::fprintf(stderr, "GPU process: cannot start %s: %s\n", program.c_str(),
                     std::strerror(error));
        return false;
    }
    std::printf("GPU process: started %s (pid %d)%s\n", program.c_str(), int(g.child),
                own_deps ? "" : " without an arm64 dependency prefix");
    return true;
}

} // namespace

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
    new (&g.control->state) SharedState{};
    g.channel = std::make_unique<Channel>(&g.control->channel, Side::Frontend);
    g.protect = std::make_unique<Channel>(&g.control->protect, Side::Frontend);

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
    front_active = true;
    std::printf("GPU process: native GPU core ready (page size %u here, %u there)\n",
                hello.page_size, welcome.page_size);
    return true;
}

// ---- GnmDriver ------------------------------------------------------------------------------

void SubmitGfx(std::span<const u32> dcb, std::span<const u32> ccb) {
    struct {
        SubmitGfxArgs args;
        u32 words[(MaxPayload - sizeof(SubmitGfxArgs)) / 4];
    } message{};
    std::size_t words = 0;
    // Host data (the init sequences) travels inline; guest command buffers by address.
    const auto add = [&](std::span<const u32> span, u64& address, u32& dwords, u32& inline_flag) {
        dwords = u32(span.size());
        if (span.empty()) {
            return true;
        }
        if (Shared(span.data(), span.size_bytes())) {
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
        return;
    }
    g.channel->Send(MsgSubmitGfx, &message, u32(sizeof(SubmitGfxArgs) + words * 4));
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
    if (result.result >= 0 && std::size_t(result.result) < g.asc.size()) {
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
                      const VO::BufferAttribute* attribute) {
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

int VoChangeBufferAttribute(s32 attribute_index, const VO::BufferAttribute* attribute) {
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
    const FaultArgs args{addr, BbPortable::Rip(context), Common::IsWriteError(context) ? 1u : 0u,
                         runtime_memory_lock_held() ? 1u : 0u};
    FaultReply reply{};
    g.channel->Call(MsgWriteFault, &args, sizeof(args), &reply, sizeof(reply));
    // Only when this thread holds the runtime's lock (see FaultReply).
    for (u32 i = 0; i < std::min(reply.count, MaxFaultProtects); ++i) {
        ApplyProtect(reply.protects[i]);
    }
    return reply.handled;
}

void InvalidateMemory(u64 address, u64 size) {
    SendRange(MsgInvalidate, address, size);
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

int OverlayCapturesInput() {
    return g.control && g.control->state.overlay_captures_input.load(std::memory_order_acquire)
               ? 1
               : 0;
}

const bool* KeyboardState() {
    static_assert(sizeof(bool) == 1);
    return g.control ? reinterpret_cast<const bool*>(g.control->state.keyboard) : nullptr;
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
