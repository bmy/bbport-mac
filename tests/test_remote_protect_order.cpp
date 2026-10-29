// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: the order in which the game process applies bb-gpu's write traps, with two real
// processes, the real channel (gpu/shim/remote/bb_channel.h), bb-gpu's real copy of the reasons
// (bb_trap_copy.h) and the game's real trap table (runtime_memory.c runtime_memory_trap_seq).
// Modelled after remote_back.cpp (WriteFault, LiftStaleTraps, MemoryTrap, FaultThread) and
// remote_front.cpp (HandleFault, ProtectThread), which need the GPU library:
// - bb-gpu's trap owners (the layer's VRAM copies, the texture cache: two reasons) arm pages
//   under their own locks, each trap a call on the protect channel the game's protection thread
//   applies;
// - the game's threads fault on trapped pages; bb-gpu's handlers run in order until one claims
//   the page (Core::Signals) and disarm it under the owner's lock, the change asked of the
//   faulting thread (Reply::Ask), which applies it; reasons the game holds and bb-gpu does not (a
//   trap that crossed a remap, injected here) are lifted first.
// Afterwards every trapped page is faulted on until none is: a page an owner still counts as
// armed while the game's page is not trapped for it (CPU writes unseen, a stale VRAM copy) fails.
// `collect` and `unlocked-lift` run protocol 10's ways (the fault's protections applied after the
// reply; the stale check outside the lock, the lift unconditional) and report what they lose:
// `collect` some traps per run; `unlocked-lift` rarely (its window is a few instructions: about
// one lost trap in 10-20 rounds here). The threads sleep now and then in the windows that matter
// (Preempt), in every mode alike.
// Then a fault from a thread holding the runtime's lock (modelled: a lock the game's applies take
// shared) while another fault's protection waits for that lock (protocol 12's ControlBlock::locked
// and FaultThread): it must finish; `one-reader` serves both on one reader, as protocol 11 did, and
// reports the deadlock.
//   cc -c -std=c11 -D_GNU_SOURCE -w -Isrc -I. src/runtime_memory.c -o /tmp/rm.o
//   c++ -std=c++20 -O2 -pthread -Igpu/shim tests/test_remote_protect_order.cpp /tmp/rm.o -o /tmp/order-test
//   /tmp/order-test [collect | unlocked-lift | one-reader] [rounds]
// ThreadSanitizer reports runtime_memory.c's own trap table race (trap_change's plain byte store,
// runtime_memory_trap_reasons' lock-free read: upstream's, not this test's).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <signal.h>
#include <string_view>
#include <sys/mman.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "remote/bb_channel.h"
#include "remote/bb_trap_copy.h"

extern "C" {
// runtime_memory.c's own imports (as tests/test_remote_concurrent.cpp).
typedef struct {
    const char* name;
    void* function;
} RuntimeExport;
uintptr_t runtime_lookup(const RuntimeExport* table, size_t count, const char* nid) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::strcmp(table[i].name, nid)) {
            return reinterpret_cast<uintptr_t>(table[i].function);
        }
    }
    return 0;
}
int32_t* runtime_errno(void) {
    static int32_t value;
    return &value;
}
void runtime_guest_call_sites(uint64_t sites[3]) {
    sites[0] = sites[1] = sites[2] = 0;
}
int bbgpu_gpu_touched(uint64_t) {
    return 0;
}
void bbgpu_straddle_report(void) {}
void runtime_memory_trap(uintptr_t address, uint64_t size, unsigned reason, int on);
void runtime_memory_trap_seq(uintptr_t address, uint64_t size, unsigned reason, int on,
                             uint64_t seq, int lift);
unsigned runtime_memory_trap_reasons(uintptr_t address);
}

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
enum class Mode : std::uint32_t { Ask = 0, Collect = 1, UnlockedLift = 2, OneReader = 3 };

constexpr std::uint32_t Pages = 32;
constexpr std::uint64_t Base = 1ull << 32; ///< page p: Base + p * 4096 (below TrapCopy::Limit)
constexpr std::uint32_t Reasons = 2;       ///< reason r: bit 1 << r, its own owner and handler
constexpr int OwnersPerReason = 2, Faulters = 3;

std::uint64_t Address(std::uint32_t page) {
    return Base + std::uint64_t(page) * 4096;
}

struct TrapArgs {
    std::uint32_t page, reasons, on, lift;
    std::uint64_t seq;
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
    std::uint32_t armed, trapped; ///< (page, reason)s bb-gpu still counts as armed, or trapped
};

struct Shared {
    ChannelLayout main, protect, locked;
};

/// A thread descheduled now and then.
void Preempt() {
    thread_local std::minstd_rand random{
        std::uint32_t(std::hash<std::thread::id>{}(std::this_thread::get_id()))};
    if (random() % 4 == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(random() % 40));
    }
}

// ---- bb-gpu ------------------------------------------------------------------------------------

struct Gpu {
    Mode mode;
    Channel main, protect, locked;
    std::mutex owner_mutex[Reasons]; ///< each owner's lock (layer_trap_mutex, ...)
    bool armed[Reasons][Pages] = {};
    TrapCopy traps; ///< remote_back.cpp `traps`
    std::atomic<bool> stop{false};

    static thread_local Channel::Reply* fault_reply;
    static thread_local std::vector<TrapArgs>* collected;

    Gpu(Shared* shared, Mode mode_)
        : mode{mode_}, main{&shared->main, Side::Backend}, protect{&shared->protect, Side::Backend},
          locked{&shared->locked, Side::Backend} {}

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

    // remote_back.cpp MemoryTrap.
    void Trap(std::uint32_t page, std::uint32_t reasons, bool on) {
        const TrapCopy::Order order = traps.Change(Address(page), 4096, reasons, on);
        CHECK(order.size == 4096);
        Forward({page, reasons, on ? 1u : 0u, 0, order.seq});
    }

    // remote_back.cpp LiftStaleTraps (Mode::UnlockedLift: protocol 10's WriteFault).
    void LiftStale(std::uint32_t page, std::uint32_t reasons) {
        if (mode == Mode::UnlockedLift) {
            if (const std::uint32_t stale = reasons & ~traps.Reasons(Address(page))) {
                Preempt();
                Trap(page, stale, false);
            }
            return;
        }
        const TrapCopy::Lift lift = traps.Stale(Address(page), reasons);
        if (lift.reasons) {
            Preempt();
            Forward({page, lift.reasons, 0, 1, lift.seq});
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

    // remote_back.cpp WriteFault.
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
        CHECK(main.Valid() && protect.Valid() && locked.Valid());
        std::vector<std::thread> owners;
        for (std::uint32_t r = 0; r < Reasons; ++r) {
            for (int i = 0; i < OwnersPerReason; ++i) {
                owners.emplace_back([this, r, i] { Owner(r, 1000 + 10 * r + i); });
            }
        }
        // remote_back.cpp FaultThread: faults from lock-holding threads.
        std::thread([this] {
            for (;;) {
                locked.ReceiveOne([&](const Message& m, auto&& reply) { Fault(m, reply); },
                                  [] { return false; });
            }
        }).detach();
        bool quit = false;
        while (!quit && main.ReceiveOne(
                            [&](const Message& m, auto&& reply) {
                                if (m.type == MsgFault) {
                                    Fault(m, reply);
                                } else if (m.type == MsgStopOwners) {
                                    stop = true;
                                    for (auto& t : owners) {
                                        if (t.joinable()) {
                                            t.join();
                                        }
                                    }
                                    reply(nullptr, 0);
                                } else if (m.type == MsgQuit) {
                                    QuitReply out{};
                                    for (std::uint32_t r = 0; r < Reasons; ++r) {
                                        std::scoped_lock lock{owner_mutex[r]};
                                        for (std::uint32_t p = 0; p < Pages; ++p) {
                                            out.armed += armed[r][p] ? 1 : 0;
                                            out.trapped += (traps.Reasons(Address(p)) >> r) & 1;
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

/// The runtime's lock, as the applies take it (runtime_memory_trap: read_lock, which a thread
/// holding the lock exclusively skips).
std::shared_mutex runtime_lock;
thread_local bool holds_runtime_lock = false;

struct Game {
    Channel main, protect, locked;
    std::atomic<std::uint64_t> faults{0}, asked{0}, applied_late{0};

    explicit Game(Shared* shared)
        : main{&shared->main, Side::Frontend}, protect{&shared->protect, Side::Frontend},
          locked{&shared->locked, Side::Frontend} {}

    // remote_front.cpp ApplyProtect.
    static void Apply(const TrapArgs& args) {
        CHECK(args.page < Pages);
        std::shared_lock<std::shared_mutex> lock{runtime_lock, std::defer_lock};
        if (!holds_runtime_lock) {
            lock.lock();
        }
        runtime_memory_trap_seq(Address(args.page), 4096, args.reasons, int(args.on), args.seq,
                                int(args.lift));
    }

    static unsigned Reasons(std::uint32_t page) {
        return runtime_memory_trap_reasons(Address(page));
    }

    // remote_front.cpp HandleFault: what is asked applied while waiting (Mode::Collect: the
    // reply's, after it); from a thread holding the runtime's lock on its own channel.
    void Fault(std::uint32_t page) {
        const FaultArgs args{page, Reasons(page)};
        FaultReply reply{};
        Channel& channel = holds_runtime_lock && mode != Mode::OneReader ? locked : main;
        channel.Call(MsgFault, &args, sizeof(args), &reply, sizeof(reply),
                     [&](const Message& m, auto&& answer) {
                         CHECK(m.type == MsgProtect && m.size == sizeof(TrapArgs));
                         TrapArgs trap;
                         std::memcpy(&trap, m.data, sizeof(trap));
                         Preempt();
                         if (on_apply) {
                             on_apply();
                         }
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

    Mode mode = Mode::Ask;
    static thread_local std::function<void()> on_apply; ///< the scenario's hook
};
thread_local std::function<void()> Game::on_apply;

struct Session {
    void* memory = nullptr;
    pid_t child = -1;
    Game* game = nullptr;
    std::atomic<bool> stop{false};
    std::thread protect_thread;

    explicit Session(Mode mode) {
        memory = mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                      -1, 0);
        CHECK(memory != MAP_FAILED);
        auto* shared = static_cast<Shared*>(memory);
        Channel::Init(&shared->main);
        Channel::Init(&shared->protect);
        Channel::Init(&shared->locked);
        std::fflush(stdout);
        child = fork();
        if (child == 0) {
            auto* gpu = new Gpu{shared, mode};
            std::_Exit(gpu->Run());
        }
        game = new Game{shared};
        game->mode = mode;
        // The protection thread (remote_front.cpp ProtectThread).
        protect_thread = std::thread([this] {
            while (game->protect.ReceiveOne(
                [&](const Message& m, auto&& reply) {
                    if (m.type == MsgProtect) {
                        TrapArgs trap;
                        CHECK(m.size == sizeof(trap));
                        std::memcpy(&trap, m.data, sizeof(trap));
                        Game::Apply(trap);
                    }
                    const std::uint32_t result = 0;
                    reply(&result, sizeof(result));
                },
                [&] { return stop.load(); })) {
            }
        });
    }

    /// Owners stopped, every trapped page faulted on until none is: what bb-gpu still counts.
    std::uint32_t Finish() {
        CHECK(game->main.Call(MsgStopOwners, nullptr, 0, nullptr, 0) == 0);
        for (bool any = true; any;) {
            any = false;
            for (std::uint32_t p = 0; p < Pages; ++p) {
                if (Game::Reasons(p)) {
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
        munmap(memory, sizeof(Shared));
        return std::max(result.armed, result.trapped);
    }
};

std::uint32_t Round(Mode mode, std::uint64_t faults_per_thread) {
    Session session{mode};
    Game* game = session.game;
    std::vector<std::thread> faulters;
    for (int t = 0; t < Faulters; ++t) {
        faulters.emplace_back([&, t] {
            std::minstd_rand random{std::uint32_t(77 + t)};
            for (std::uint64_t done = 0; done < faults_per_thread;) {
                const std::uint32_t page = random() % Pages;
                if (random() % 32 == 0) {
                    // A trap that crossed a remap: the game holds a reason bb-gpu does not (the
                    // second one's: its handler runs only when the first one's does not claim it).
                    runtime_memory_trap(Address(page), 4096, 2, 1);
                }
                if (Game::Reasons(page)) {
                    game->Fault(page); // an access to a trapped page
                    ++done;
                }
            }
        });
    }
    for (auto& t : faulters) {
        t.join();
    }
    const std::uint64_t faults = game->faults.load(), asked = game->asked.load(),
                        late = game->applied_late.load();
    const std::uint32_t lost = session.Finish();
    std::printf("  %llu faults, %llu protections asked of the faulting thread, %llu applied after "
                "the reply; armed but not trapped: %u\n",
                (unsigned long long)faults, (unsigned long long)asked, (unsigned long long)late,
                lost);
    return lost;
}

/// A fault from a thread holding the runtime's lock while another fault's protection waits for
/// that lock in its faulting thread (on bb-gpu's main reader). True when both finished.
bool LockHeldFault(Mode mode) {
    Session session{mode};
    Game* game = session.game;
    // No owner arming meanwhile: one waiting in the protection thread for the lock, holding its
    // owner lock, would stop the lock holder's handler as in one process (not this test's case).
    CHECK(game->main.Call(MsgStopOwners, nullptr, 0, nullptr, 0) == 0);
    // Two stale traps: each fault's lift is asked of its faulting thread.
    runtime_memory_trap(Address(1), 4096, 2, 1);
    runtime_memory_trap(Address(2), 4096, 2, 1);
    std::atomic<int> stage{0};
    std::thread holder([&] {
        std::unique_lock<std::shared_mutex> lock{runtime_lock};
        holds_runtime_lock = true;
        stage = 1;
        while (stage.load() < 2) { // the other fault's apply waits for this lock now
            std::this_thread::yield();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        game->Fault(2); // inside the runtime, holding its lock
        holds_runtime_lock = false;
        stage = 3;
    });
    std::thread other([&] {
        while (stage.load() < 1) {
            std::this_thread::yield();
        }
        Game::on_apply = [&] { stage = 2; };
        game->Fault(1);
        Game::on_apply = nullptr;
    });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (stage.load() < 3 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (stage.load() < 3) {
        kill(session.child, SIGKILL);
        std::printf("deadlocked: the lock holder's fault waits behind the other one\n");
        std::fflush(stdout);
        std::_Exit(mode == Mode::OneReader ? 0 : 1);
    }
    holder.join();
    other.join();
    CHECK(Game::Reasons(1) == 0 && Game::Reasons(2) == 0);
    session.Finish();
    return true;
}
} // namespace

/// `run()` in a process of its own (the game's tables and bb-gpu's order start over); its result.
template <typename Run>
std::uint32_t Isolated(Run&& run) {
    int pipe_fds[2];
    CHECK(pipe(pipe_fds) == 0);
    std::fflush(stdout);
    const pid_t child = fork();
    if (child == 0) {
        const std::uint32_t result = run();
        CHECK(write(pipe_fds[1], &result, sizeof(result)) == sizeof(result));
        std::fflush(stdout);
        std::_Exit(0);
    }
    close(pipe_fds[1]);
    std::uint32_t result = ~0u;
    const bool got = read(pipe_fds[0], &result, sizeof(result)) == sizeof(result);
    int status = 0;
    waitpid(child, &status, 0);
    close(pipe_fds[0]);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    return got ? result : ~0u;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    Mode mode = Mode::Ask;
    int rounds = 6;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg{argv[i]};
        if (arg == "collect") {
            mode = Mode::Collect;
        } else if (arg == "unlocked-lift") {
            mode = Mode::UnlockedLift;
        } else if (arg == "one-reader") {
            mode = Mode::OneReader;
        } else {
            rounds = std::atoi(argv[i]);
        }
    }
    if (mode == Mode::OneReader) {
        std::printf("lock-held fault on the main reader (protocol 11): ");
        std::fflush(stdout);
        if (Isolated([&] { return LockHeldFault(mode) ? 1u : 0u; }) == 1) {
            std::printf("finished (not expected)\n");
        }
        return 0;
    }
    const char* names[] = {"asked of the faulting thread, lifts ordered",
                           "collected, applied after the reply",
                           "asked, stale reasons checked outside the lock, lifts unconditional"};
    std::printf("protect order (%s), %d rounds:\n", names[int(mode)], rounds);
    std::uint32_t lost = 0;
    for (int r = 0; r < rounds; ++r) {
        lost += Isolated([&] { return Round(mode, 4000); });
    }
    if (mode != Mode::Ask) {
        std::printf("protect order: %u traps lost in the game process (protocol 10's way)\n", lost);
        return 0;
    }
    CHECK(lost == 0);
    std::printf("protect order: every armed page trapped in the game process\n");
    CHECK(Isolated([&] { return LockHeldFault(mode) ? 1u : 0u; }) == 1);
    std::printf("lock-held fault while another waits for that lock: both finished (ControlBlock::"
                "locked)\n");
    return 0;
}
