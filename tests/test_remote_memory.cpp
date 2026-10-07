// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the native GPU process's view of guest memory (gpu/shim/remote/bb_memory_mirror.h)
// across two real processes, as bb-probe and bb-gpu will run: the GPU side is a fresh process
// (exec, not just fork) that reserves the guest range, mirrors the game side's mappings of one
// shared pool at the same addresses, sees its writes and is seen, follows remaps and aliases, and
// answers a real write fault on a write-protected guest page with a protection call back
// (signal handler -> call -> nested call -> the game side's reader unprotects -> retry).
//   c++ -std=c++20 -O2 -pthread -Igpu/shim tests/test_remote_memory.cpp -o /tmp/memory-test
//   /tmp/memory-test
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#ifdef __APPLE__
#include <mach/machine.h>
#include <sys/sysctl.h>
#endif
#include <spawn.h>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include "remote/bb_channel.h"
#include "remote/bb_memory_mirror.h"
#include "remote/bb_protocol.h"

using namespace BbRemote;

extern char** environ;

namespace {
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed (errno %d: %s)\n", __FILE__, __LINE__, \
                         #x, errno, std::strerror(errno));                                        \
            std::_Exit(1);                                                                         \
        }                                                                                          \
    } while (0)

// Test-only calls.
enum : std::uint32_t { TestRead = 1000, TestWrite = 1001 };
struct Access {
    std::uint64_t address, value;
};

constexpr std::uint64_t GuestBegin = 0x7000000000ull, GuestEnd = 0x7040000000ull;
constexpr std::uint64_t PoolBytes = 64ull << 20;

/// A shared memory object of `bytes` (macOS allows one ftruncate per object).
int CreateShared(const char* what, std::uint64_t bytes) {
#ifdef __linux__
    const int fd = memfd_create(what, 0);
#else
    char name[64];
    std::snprintf(name, sizeof(name), "/%s-%d", what, int(getpid()));
    const int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
    shm_unlink(name);
#endif
    CHECK(fd >= 0 && ftruncate(fd, static_cast<off_t>(bytes)) == 0);
    // shm_open sets close-on-exec (POSIX): the GPU process inherits the descriptor only without it.
    CHECK(fcntl(fd, F_SETFD, 0) == 0);
    return fd;
}

/// "x86-64 under Rosetta", "arm64" or "x86-64".
const char* ThisArch() {
#if defined(__aarch64__)
    return "arm64";
#elif defined(__APPLE__)
    int translated = 0;
    size_t size = sizeof(translated);
    if (sysctlbyname("sysctl.proc_translated", &translated, &size, nullptr, 0) == 0 && translated) {
        return "x86-64 under Rosetta";
    }
    return "x86-64";
#else
    return "x86-64";
#endif
}

// ---- the GPU process --------------------------------------------------------------------------

int GpuProcess(int channel_fd, int pool_fd) {
    void* shared = mmap(nullptr, ChannelBytes, PROT_READ | PROT_WRITE, MAP_SHARED, channel_fd, 0);
    CHECK(shared != MAP_FAILED);
    Channel channel{shared, Side::Backend};
    CHECK(channel.Valid());
    MemoryMirror mirror;
    bool quit = false;
    while (!quit && channel.ReceiveOne(
                        [&](const Message& m, auto&& reply) {
                            switch (m.type) {
                            case MsgHello: {
                                HelloArgs args;
                                std::memcpy(&args, m.data, sizeof(args));
                                HelloReply out{ProtocolVersion, 0, 0, 0};
                                std::fprintf(stderr, "GPU process: %s\n", ThisArch());
                                std::string_view error;
                                if (args.protocol != ProtocolVersion ||
                                    !mirror.Reserve(args.guest_begin, args.guest_end, error)) {
                                    std::fprintf(stderr, "GPU process: %.*s\n", int(error.size()),
                                                 error.data());
                                    out.error = 1;
                                }
                                out.page_size = std::uint32_t(mirror.PageSize());
                                reply(&out, sizeof(out));
                                break;
                            }
                            case MsgMapMemory: {
                                MapArgs args;
                                std::memcpy(&args, m.data, sizeof(args));
                                CHECK(mirror.Map(args.address, args.size, pool_fd, args.offset));
                                break;
                            }
                            case MsgUnmapMemory: {
                                RangeArgs args;
                                std::memcpy(&args, m.data, sizeof(args));
                                CHECK(mirror.Unmap(args.address, args.size));
                                break;
                            }
                            case MsgWriteFault: {
                                // As libbbgpu will: mark the page written, then lift the
                                // protection in the game process before it retries the write.
                                RangeArgs page;
                                std::memcpy(&page, m.data, sizeof(page));
                                ProtectArgs protect{page.address, page.size, 1, 1};
                                ResultReply result{};
                                CHECK(channel.Call(MsgProtect, &protect, sizeof(protect), &result,
                                                   sizeof(result)) == sizeof(result));
                                reply(&result, sizeof(result));
                                break;
                            }
                            case TestRead: {
                                Access a;
                                std::memcpy(&a, m.data, sizeof(a));
                                a.value = *reinterpret_cast<volatile std::uint64_t*>(a.address);
                                reply(&a, sizeof(a));
                                break;
                            }
                            case TestWrite: {
                                Access a;
                                std::memcpy(&a, m.data, sizeof(a));
                                // Release: the game side reads it after the reply.
                                reinterpret_cast<std::atomic<std::uint64_t>*>(a.address)
                                    ->store(a.value, std::memory_order_release);
                                reply(&a, sizeof(a));
                                break;
                            }
                            case MsgShutdown:
                                reply(nullptr, 0);
                                quit = true;
                                break;
                            }
                        },
                        [] { return false; })) {
    }
    return 0;
}

// ---- the game process -------------------------------------------------------------------------

Channel* g_channel;
std::atomic<int> g_faults{0};

void OnFault(int, siginfo_t* info, void*) {
    const auto address = reinterpret_cast<std::uint64_t>(info->si_addr);
    if (address < GuestBegin || address >= GuestEnd) {
        std::_Exit(3);
    }
    const std::uint64_t page = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    RangeArgs args{address & ~(page - 1), page};
    ResultReply result{};
    g_channel->Call(MsgWriteFault, &args, sizeof(args), &result, sizeof(result));
    g_faults.fetch_add(1);
}

std::uint64_t Read(Channel& channel, std::uint64_t address) {
    Access a{address, 0};
    CHECK(channel.Call(TestRead, &a, sizeof(a), &a, sizeof(a)) == sizeof(a));
    return a.value;
}

void Write(Channel& channel, std::uint64_t address, std::uint64_t value) {
    Access a{address, value};
    CHECK(channel.Call(TestWrite, &a, sizeof(a), &a, sizeof(a)) == sizeof(a));
}

void MapGuest(Channel& channel, int pool, std::uint64_t address, std::uint64_t size,
              std::uint64_t offset) {
    CHECK(mmap(reinterpret_cast<void*>(address), size, PROT_READ | PROT_WRITE,
               MAP_SHARED | MAP_FIXED, pool, static_cast<off_t>(offset)) ==
          reinterpret_cast<void*>(address));
    MapArgs args{address, size, offset};
    channel.Send(MsgMapMemory, &args, sizeof(args));
}

int GameProcess(const char* self) {
    const int pool = CreateShared("bb-test-pool", PoolBytes);
    const int channel_fd = CreateShared("bb-test-channel", ChannelBytes);
    void* shared = mmap(nullptr, ChannelBytes, PROT_READ | PROT_WRITE, MAP_SHARED, channel_fd, 0);
    CHECK(shared != MAP_FAILED);
    Channel::Init(shared);
    Channel channel{shared, Side::Frontend};
    g_channel = &channel;

    // The GPU process: a fresh image, the two shared objects inherited by number.
    const std::string channel_arg = std::to_string(channel_fd), pool_arg = std::to_string(pool);
    char* argv[] = {const_cast<char*>(self), const_cast<char*>("gpu"),
                    const_cast<char*>(channel_arg.c_str()), const_cast<char*>(pool_arg.c_str()),
                    nullptr};
    pid_t gpu = 0;
    posix_spawnattr_t attributes;
    posix_spawnattr_init(&attributes);
#ifdef __APPLE__
    // A universal build run with `arch -x86_64`: the GPU side starts as native arm64, as bb-gpu
    // will (BB_TEST_SAME_ARCH=1 keeps the parent's architecture).
    if (!std::getenv("BB_TEST_SAME_ARCH")) {
        cpu_type_t preferred[] = {CPU_TYPE_ARM64};
        size_t count = 0;
        posix_spawnattr_setbinpref_np(&attributes, 1, preferred, &count);
    }
#endif
    CHECK(posix_spawn(&gpu, self, nullptr, &attributes, argv, environ) == 0);
    posix_spawnattr_destroy(&attributes);

    // This side's reader: applies the GPU process's protection changes.
    std::atomic<bool> stop{false};
    std::atomic<int> protects{0};
    std::thread reader([&] {
        while (channel.ReceiveOne(
            [&](const Message& m, auto&& reply) {
                if (m.type == MsgProtect) {
                    ProtectArgs args;
                    std::memcpy(&args, m.data, sizeof(args));
                    const int prot =
                        (args.read ? PROT_READ : 0) | (args.write ? PROT_WRITE | PROT_READ : 0);
                    ResultReply result{mprotect(reinterpret_cast<void*>(args.address), args.size,
                                                prot)};
                    protects.fetch_add(1);
                    reply(&result, sizeof(result));
                }
            },
            [&] { return stop.load(); })) {
        }
    });

    // Reserve the guest range here too (the runtime's job in bb-probe).
    void* reserved = mmap(reinterpret_cast<void*>(GuestBegin), GuestEnd - GuestBegin, PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    CHECK(reserved == reinterpret_cast<void*>(GuestBegin));

    HelloArgs hello{ProtocolVersion, std::uint32_t(sysconf(_SC_PAGESIZE)), PoolBytes, GuestBegin,
                    GuestEnd};
    HelloReply welcome{};
    CHECK(channel.Call(MsgHello, &hello, sizeof(hello), &welcome, sizeof(welcome)) ==
          sizeof(welcome));
    CHECK(welcome.error == 0 && welcome.protocol == ProtocolVersion);

    // Two mappings and an alias of the first one's pool memory.
    const std::uint64_t a = GuestBegin + (16ull << 20), b = GuestBegin + (256ull << 20);
    const std::uint64_t alias = GuestBegin + (512ull << 20);
    MapGuest(channel, pool, a, 8ull << 20, 0);
    MapGuest(channel, pool, b, 4ull << 20, 16ull << 20);
    MapGuest(channel, pool, alias, 1ull << 20, 0);

    // Game writes, GPU reads; GPU writes, game reads; through the alias too.
    *reinterpret_cast<volatile std::uint64_t*>(a + 64) = 0x1111;
    CHECK(Read(channel, a + 64) == 0x1111);
    CHECK(Read(channel, alias + 64) == 0x1111);
    Write(channel, b + 4096, 0x2222);
    CHECK(*reinterpret_cast<volatile std::uint64_t*>(b + 4096) == 0x2222);
    Write(channel, alias + 128, 0x3333);
    CHECK(*reinterpret_cast<volatile std::uint64_t*>(a + 128) == 0x3333);

    // Remap b to other pool memory: the GPU side follows.
    CHECK(munmap(reinterpret_cast<void*>(b), 4ull << 20) == 0);
    RangeArgs gone{b, 4ull << 20};
    channel.Send(MsgUnmapMemory, &gone, sizeof(gone));
    MapGuest(channel, pool, b, 4ull << 20, 32ull << 20);
    *reinterpret_cast<volatile std::uint64_t*>(b + 4096) = 0x4444;
    CHECK(Read(channel, b + 4096) == 0x4444);

    // A write fault on a write-protected guest page, answered by the GPU process.
    struct sigaction action {};
    action.sa_sigaction = OnFault;
    action.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &action, nullptr);
    sigaction(SIGBUS, &action, nullptr);
    CHECK(mprotect(reinterpret_cast<void*>(a), 8ull << 20, PROT_READ) == 0);
    *reinterpret_cast<volatile std::uint64_t*>(a + 8192) = 0x5555; // faults once
    CHECK(g_faults.load() == 1 && protects.load() == 1);
    CHECK(Read(channel, a + 8192) == 0x5555);
    *reinterpret_cast<volatile std::uint64_t*>(a + 8200) = 0x6666; // same page: no fault
    CHECK(g_faults.load() == 1);

    CHECK(channel.Call(MsgShutdown, nullptr, 0, nullptr, 0) == 0);
    int status = 0;
    waitpid(gpu, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    stop = true;
    channel.WakeReader();
    reader.join();
    std::printf("memory mirror: shared mappings, alias, remap and a write fault round trip OK "
                "(game side %s, GPU page size %u)\n",
                ThisArch(), welcome.page_size);
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 4 && std::string_view{argv[1]} == "gpu") {
        return GpuProcess(std::atoi(argv[2]), std::atoi(argv[3]));
    }
    return GameProcess(argv[0]);
}
