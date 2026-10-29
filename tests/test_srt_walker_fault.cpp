// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// Real walker faults must be local to an invocation, including concurrent SIGSEGVs.
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>
#include <signal.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>

#include "shader/ir/passes/srt.h"
#include "shader/info.h"

extern "C" int bbgpu_handle_fault(void*, void*);

namespace {
std::atomic<unsigned> faults{0};
std::atomic<unsigned> rendezvous{0};
std::atomic<bool> synchronize{false};
static_assert(std::atomic<unsigned>::is_always_lock_free);

void OnFault(int, siginfo_t* info, void* context) {
    faults.fetch_add(1, std::memory_order_relaxed);
    if (synchronize.load(std::memory_order_relaxed)) {
        // Force two threads to fault before either handler can alter anything.
        rendezvous.fetch_add(1, std::memory_order_relaxed);
        while (rendezvous.load(std::memory_order_relaxed) < 2) {}
    }
    if (!bbgpu_handle_fault(context, info->si_addr)) _exit(90);
}
}

int main() {
    // The host specialization path reads vertex descriptors without executing a walker.
    Shader::Info info{};
    std::array<u32, 16> ud{};
    info.user_data = ud;
    assert(info.ReadUdReg<u64>(2, 2) == 0); // the crash's null table + 8
    assert(info.ReadUdReg<u64>(15, 0) == 0); // truncated pointer pair
    assert(info.ReadUdReg<u64>(Shader::IR::NumScalarRegs, 15) == 0);
    const std::array<u32, 4> table{7, 11, 13, 17};
    const std::uintptr_t table_ptr = reinterpret_cast<std::uintptr_t>(table.data());
    std::memcpy(ud.data() + 2, &table_ptr, sizeof(table_ptr));
    assert(info.ReadUdReg<u32>(2, 2) == 13);
    assert(info.ReadUdReg<u32>(Shader::IR::NumScalarRegs, 0) == 0);

    struct sigaction action{};
    action.sa_sigaction = OnFault;
    action.sa_flags = SA_SIGINFO;
    sigemptyset(&action.sa_mask);
    assert(sigaction(SIGSEGV, &action, nullptr) == 0);
    void* unavailable = mmap(nullptr, 4096, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(unavailable != MAP_FAILED);
    // mov rdi,[rdi]; mov r10d,[rdi]; mov [rsi],r10d; ret
    constexpr std::array<u8, 10> code{0x48, 0x8b, 0x3f, 0x44, 0x8b,
                                     0x17, 0x44, 0x89, 0x16, 0xc3};
    const auto walker = Shader::RegisterWalkerCode(code.data(), code.size());
    ucontext_t context{};
    context.uc_mcontext.gregs[REG_RIP] = reinterpret_cast<greg_t>(walker) + 3;
    context.uc_mcontext.gregs[REG_R10] = -1;
    context.uc_mcontext.gregs[REG_EFL] = 0x247;
    assert(bbgpu_handle_fault(&context, unavailable));
    assert(context.uc_mcontext.gregs[REG_R10] == 0);
    assert(context.uc_mcontext.gregs[REG_RIP] == reinterpret_cast<greg_t>(walker) + 6);
    assert(context.uc_mcontext.gregs[REG_EFL] == 0x247);
    // A store in the walker is not an unavailable resource read and must not be swallowed.
    assert(!bbgpu_handle_fault(&context, unavailable));
    u32 value = 0x12345678;
    const std::uintptr_t valid = reinterpret_cast<std::uintptr_t>(&value);
    const std::uintptr_t invalid = reinterpret_cast<std::uintptr_t>(unavailable);
    u32 result = 1;
    walker(reinterpret_cast<const u32*>(&valid), &result);
    assert(result == value);
    walker(reinterpret_cast<const u32*>(&invalid), &result); // R10D load faults
    assert(result == 0);
    walker(static_cast<const u32*>(unavailable), &result); // RDI and then R10D fault
    assert(result == 0);
    walker(reinterpret_cast<const u32*>(&valid), &result);
    assert(result == value); // earlier failures must not poison this walker
    assert(std::memcmp(reinterpret_cast<const void*>(walker), code.data(), code.size()) == 0);

    synchronize.store(true);
    auto simultaneous = [&] {
        u32 out = 1;
        walker(reinterpret_cast<const u32*>(&invalid), &out);
        assert(out == 0);
    };
    std::thread a(simultaneous), b(simultaneous);
    a.join(); b.join();
    synchronize.store(false);
    std::vector<std::thread> workers;
    for (unsigned i = 0; i < 8; ++i) workers.emplace_back([&] {
        for (unsigned j = 0; j < 64; ++j) {
            u32 out = 1;
            walker(reinterpret_cast<const u32*>(&invalid), &out);
            assert(out == 0);
            walker(reinterpret_cast<const u32*>(&valid), &out);
            assert(out == value);
        }
    });
    for (auto& worker : workers) worker.join();
    assert(faults.load() == 3 + 2 + 8 * 64);
    assert(std::memcmp(reinterpret_cast<const void*>(walker), code.data(), code.size()) == 0);
    assert(munmap(unavailable, 4096) == 0);
    std::printf("SRT faults: %u recovered, concurrent reads and later valid data passed; code unchanged\n",
                faults.load());
}
