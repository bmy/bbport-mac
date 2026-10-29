// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the order in which the game process applies bb-gpu's write traps (remote_back.cpp
// WriteFault, LiftStaleTraps, MemoryTrap; remote_front.cpp HandleFault, ProtectThread), modelled
// with two real processes and the real channel (gpu/shim/remote/bb_channel.h):
// - the GPU side's trap owners (the layer's VRAM copies, the texture cache: two reasons) arm
//   pages under their own locks, each trap a call on the protect channel that the game side's
//   protection thread applies;
// - the game side's threads fault on trapped pages; on the GPU side the owners' handlers run in
//   order until one claims the page (Core::Signals) and disarm it under the owner's lock, the
//   trap change asked of the faulting thread (Reply::Ask), which applies it; reasons the game
//   holds and the GPU side does not (a trap that crossed a remap, injected here) are lifted
//   first, checked under the GPU side's table lock (LiftStaleTraps).
// Afterwards every trapped page is faulted on until none is: a page an owner still counts as
// armed while the game's page is not trapped for it (CPU writes unseen, a stale VRAM copy) fails.
// `collect` and `unlocked-lift` run protocol 10's ways (the fault's protections applied after the
// reply; the stale check outside the lock) and report what they get wrong. The threads sleep now
// and then in the windows that matter (Preempt), in every mode alike.
//   c++ -std=c++20 -O2 -pthread -Igpu/shim tests/test_remote_protect_order.cpp -o /tmp/order-test
//   /tmp/order-test [collect | unlocked-lift] [rounds]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string_view>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "remote/bb_channel.h"

using namespace BbRemote;

namespace {
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        if (!(x)) {                                                                                \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #x);          \
            std::_Exit(1);                                                                         \
        }                                                                                          \
    } while (0)

enum : std::uint32_t { MsgFault = 1, MsgProtect = 2, MsgStopOwners = 3, MsgQuit = 4 };
enum class Mode : std::uint32_t { Ask = 0, Collect = 1, UnlockedLift = 2 };

constexpr std::uint32_t Pages = 32;
constexpr std::uint32_t Reasons = 2; ///< reason r: bit 1 << r, its own owner and handler
constexpr int OwnersPerReason = 2, Faulters = 3;

struct TrapArgs {
    std::uint32_t page, reasons, on;
};
struct FaultArgs {
    std::uint32_t page, reasons; ///< the game's reasons for the page when it faulted
};
constexpr std::uint32_t MaxCollected = 64;
struct FaultReply {
    std::uint32_t handled, count;
    TrapArgs collected[MaxCollected]; ///< Mode::Collect only
};
struct QuitReply {
    std::uint32_t armed, trapped; ///< (page, reason)s the GPU side still counts as armed, trapped
};

struct Shared {
    ChannelLayout main, protect;
};

/// A thread descheduled now and then.
void Preempt() {
    thread_local std::minstd_rand random{
        std::uint32_t(std::hash<std::thread::id>{}(std::this_thread::get_id()))};
    if (random() % 4 == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(random() % 40));
    }
}

// ---- the GPU process ---------------------------------------------------------------------------

struct Gpu {
    Mode mode;
    Channel main, protect;
    std::mutex owner_mutex[Reasons]; ///< each owner's lock (layer_trap_mutex, ...)
    bool armed[Reasons][Pages] = {};
    std::mutex trap_mutex; ///< remote_back.cpp trap_mutex: this side's copy of the reasons
    std::atomic<std::uint8_t> table[Pages] = {};
    std::atomic<bool> stop{false};

    static thread_local Channel::Reply* fault_reply;
    static thread_local std::vector<TrapArgs>* collected;

    Gpu(Shared* shared, Mode mode_)
        : mode{mode_}, main{&shared->main, Side::Backend}, protect{&shared->protect, Side::Backend} {}

    // remote_back.cpp ForwardProtect.
    void Forward(const TrapArgs& args) {
        std::uint32_t result = 0;
        if (collected) {
            collected->push_back(args);
        } else if (fault_reply) {
            CHECK(fault_reply->Ask(MsgProtect, &args, sizeof(args), &result, sizeof(result)) == 4);
        } else {
            CHECK(protect.Call(MsgProtect, &args, sizeof(args), &result, sizeof(result)) == 4);
        }
    }

    // remote_back.cpp MemoryTrap: the table under its lock, then sent.
    void Trap(std::uint32_t page, std::uint32_t reasons, bool on) {
        {
            std::scoped_lock lock{trap_mutex};
            const std::uint8_t old = table[page].load(std::memory_order_relaxed);
            table[page].store(std::uint8_t(on ? old | reasons : old & ~reasons),
                              std::memory_order_release);
        }
        Forward({page, reasons, on ? 1u : 0u});
    }

    // remote_back.cpp LiftStaleTraps (Mode::UnlockedLift: protocol 10's WriteFault).
    void LiftStale(std::uint32_t page, std::uint32_t reasons) {
        if (mode == Mode::UnlockedLift) {
            if (const std::uint32_t stale = reasons & ~table[page].load(std::memory_order_acquire)) {
                Preempt();
                Trap(page, stale, false);
            }
            return;
        }
        std::scoped_lock lock{trap_mutex};
        if (const std::uint32_t stale = reasons & ~table[page].load(std::memory_order_relaxed)) {
            Preempt();
            Forward({page, stale, 0});
        }
    }

    // An owner's handler (LayerTrapHandler): its trap lifted under its lock.
    bool Handler(std::uint32_t reason, std::uint32_t page) {
        std::scoped_lock lock{owner_mutex[reason]};
        if (!armed[reason][page]) {
            return false;
        }
        armed[reason][page] = false;
        Trap(page, 1u << reason, false);
        return true;
    }

    void Owner(std::uint32_t reason, unsigned seed) {
        std::minstd_rand random{seed};
        while (!stop.load(std::memory_order_relaxed)) {
            {
                std::scoped_lock lock{owner_mutex[reason]};
                const std::uint32_t page = random() % Pages;
                // Armed again once its handler lifted it (LayerProcessTraps after a hit), never
                // lifted otherwise: a trap lost in the game process stays lost to the end.
                if (!armed[reason][page]) {
                    armed[reason][page] = true;
                    Trap(page, 1u << reason, true);
                }
            }
            // The first reason's owner keeps its pages armed nearly all the time.
            std::this_thread::sleep_for(std::chrono::microseconds(random() % (reason ? 64 : 8)));
        }
    }

    void Fault(const Message& m, Channel::Reply& reply) {
        FaultArgs args;
        CHECK(m.size == sizeof(args));
        std::memcpy(&args, m.data, sizeof(args));
        CHECK(args.page < Pages);
        std::vector<TrapArgs> list;
        collected = mode == Mode::Collect ? &list : nullptr;
        fault_reply = mode != Mode::Collect && reply.CanAsk() ? &reply : nullptr;
        LiftStale(args.page, args.reasons);
        FaultReply out{};
        // Core::Signals: the handlers in order, until one claims the fault.
        for (std::uint32_t r = 0; r < Reasons && !out.handled; ++r) {
            out.handled = Handler(r, args.page) ? 1 : 0;
        }
        fault_reply = nullptr;
        collected = nullptr;
        CHECK(list.size() <= MaxCollected);
        out.count = std::uint32_t(list.size());
        std::copy(list.begin(), list.end(), out.collected);
        reply(&out, sizeof(out));
    }

    int Run() {
        CHECK(main.Valid() && protect.Valid());
        std::vector<std::thread> owners;
        for (std::uint32_t r = 0; r < Reasons; ++r) {
            for (int i = 0; i < OwnersPerReason; ++i) {
                owners.emplace_back([this, r, i] { Owner(r, 1000 + 10 * r + i); });
            }
        }
        bool quit = false;
        while (!quit && main.ReceiveOne(
                            [&](const Message& m, auto&& reply) {
                                if (m.type == MsgFault) {
                                    Fault(m, reply);
                                } else if (m.type == MsgStopOwners) {
                                    stop = true;
                                    for (auto& t : owners) {
                                        t.join();
                                    }
                                    reply(nullptr, 0);
                                } else if (m.type == MsgQuit) {
                                    QuitReply out{};
                                    for (std::uint32_t r = 0; r < Reasons; ++r) {
                                        std::scoped_lock lock{owner_mutex[r]};
                                        for (std::uint32_t p = 0; p < Pages; ++p) {
                                            out.armed += armed[r][p] ? 1 : 0;
                                            out.trapped += (table[p].load() >> r) & 1;
                                        }
                                    }
                                    reply(&out, sizeof(out));
                                    quit = true;
                                }
                            },
                            [] { return false; })) {
        }
        return 0;
    }
};
thread_local Channel::Reply* Gpu::fault_reply = nullptr;
thread_local std::vector<TrapArgs>* Gpu::collected = nullptr;

// ---- the game process --------------------------------------------------------------------------

struct Game {
    Channel main, protect;
    std::mutex trap_mutex; ///< runtime_memory.c trap_mutex
    std::atomic<std::uint8_t> bits[Pages] = {};
    std::atomic<std::uint64_t> faults{0}, asked{0}, applied_late{0};

    explicit Game(Shared* shared)
        : main{&shared->main, Side::Frontend}, protect{&shared->protect, Side::Frontend} {}

    // runtime_memory_trap.
    void Apply(const TrapArgs& args) {
        CHECK(args.page < Pages);
        std::scoped_lock lock{trap_mutex};
        const std::uint8_t old = bits[args.page].load(std::memory_order_relaxed);
        bits[args.page].store(std::uint8_t(args.on ? old | args.reasons : old & ~args.reasons),
                              std::memory_order_release);
    }

    // remote_front.cpp HandleFault: what is asked applied while waiting (Mode::Collect: the
    // reply's, after it).
    void Fault(std::uint32_t page) {
        const FaultArgs args{page, bits[page].load(std::memory_order_acquire)};
        FaultReply reply{};
        main.Call(MsgFault, &args, sizeof(args), &reply, sizeof(reply),
                  [&](const Message& m, auto&& answer) {
                      CHECK(m.type == MsgProtect && m.size == sizeof(TrapArgs));
                      TrapArgs trap;
                      std::memcpy(&trap, m.data, sizeof(trap));
                      Preempt();
                      Apply(trap);
                      asked.fetch_add(1, std::memory_order_relaxed);
                      const std::uint32_t result = 0;
                      answer(&result, sizeof(result));
                  });
        for (std::uint32_t i = 0; i < std::min(reply.count, MaxCollected); ++i) {
            Preempt();
            Apply(reply.collected[i]);
            applied_late.fetch_add(1, std::memory_order_relaxed);
        }
        faults.fetch_add(1, std::memory_order_relaxed);
    }
};

std::uint32_t Round(Mode mode, std::uint64_t faults_per_thread) {
    void* memory = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                        -1, 0);
    CHECK(memory != MAP_FAILED);
    auto* shared = static_cast<Shared*>(memory);
    Channel::Init(&shared->main);
    Channel::Init(&shared->protect);
    const pid_t child = fork();
    if (child == 0) {
        auto* gpu = new Gpu{shared, mode};
        std::_Exit(gpu->Run());
    }
    auto* game = new Game{shared};
    // The protection thread (remote_front.cpp ProtectThread).
    std::atomic<bool> stop{false};
    std::thread protect_thread([&] {
        while (game->protect.ReceiveOne(
            [&](const Message& m, auto&& reply) {
                if (m.type == MsgProtect) {
                    TrapArgs trap;
                    CHECK(m.size == sizeof(trap));
                    std::memcpy(&trap, m.data, sizeof(trap));
                    game->Apply(trap);
                }
                const std::uint32_t result = 0;
                reply(&result, sizeof(result));
            },
            [&] { return stop.load(); })) {
        }
    });
    std::vector<std::thread> faulters;
    for (int t = 0; t < Faulters; ++t) {
        faulters.emplace_back([&, t] {
            std::minstd_rand random{std::uint32_t(77 + t)};
            for (std::uint64_t done = 0; done < faults_per_thread;) {
                const std::uint32_t page = random() % Pages;
                if (random() % 32 == 0) {
                    // A trap that crossed a remap: the game holds a reason the GPU side does not
                    // (the second one's: its handler runs only when the first one's does not
                    // claim the page).
                    game->Apply({page, 2, 1});
                }
                if (game->bits[page].load(std::memory_order_acquire)) {
                    game->Fault(page); // an access to a trapped page
                    ++done;
                }
            }
        });
    }
    for (auto& t : faulters) {
        t.join();
    }
    CHECK(game->main.Call(MsgStopOwners, nullptr, 0, nullptr, 0) == 0);
    // Every trapped page accessed once more, until none is.
    for (bool any = true; any;) {
        any = false;
        for (std::uint32_t p = 0; p < Pages; ++p) {
            if (game->bits[p].load()) {
                game->Fault(p);
                any = true;
            }
        }
    }
    QuitReply result{};
    CHECK(game->main.Call(MsgQuit, nullptr, 0, &result, sizeof(result)) == sizeof(result));
    int status = 0;
    waitpid(child, &status, 0);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    stop = true;
    game->protect.WakeReader();
    protect_thread.join();
    // Nothing is trapped in the game process now: whatever the GPU side counts is lost there.
    const std::uint32_t lost = std::max(result.armed, result.trapped);
    std::printf("  %llu faults, %llu protections asked of the faulting thread, %llu applied after "
                "the reply; armed but not trapped: %u\n",
                (unsigned long long)game->faults.load(), (unsigned long long)game->asked.load(),
                (unsigned long long)game->applied_late.load(), lost);
    munmap(memory, sizeof(Shared));
    return lost;
}
} // namespace

int main(int argc, char** argv) {
    Mode mode = Mode::Ask;
    int rounds = 6;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "collect") {
            mode = Mode::Collect;
        } else if (arg == "unlocked-lift") {
            mode = Mode::UnlockedLift;
        } else {
            rounds = std::atoi(argv[i]);
        }
    }
    const char* names[] = {"asked of the faulting thread", "collected, applied after the reply",
                           "asked, stale reasons checked outside the lock"};
    std::printf("protect order (%s), %d rounds:\n", names[int(mode)], rounds);
    std::uint32_t lost = 0;
    for (int r = 0; r < rounds; ++r) {
        lost += Round(mode, 4000);
    }
    if (mode != Mode::Ask) {
        std::printf("protect order: %u traps lost in the game process (protocol 10's way)\n", lost);
        return 0;
    }
    CHECK(lost == 0);
    std::printf("protect order: every armed page trapped in the game process\n");
    return 0;
}
