// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: observers in a game's own code (any game). At a byte-checked run of whole instructions
// (at least 5 bytes, nothing relative to the instruction pointer) a jump goes to a trampoline that
// saves every general register, the flags, MXCSR and the YMM registers into a snapshot, calls a
// host function with it, puts the (possibly changed) registers back, runs the replaced
// instructions and jumps back. With a skip flag (function entries only) the hook may set it to
// return from the game's function at once.
//
// The interface is shadPS4's (core/cpu_patches.h of the shadp2p/BBSC forks), so game modules
// written against it (BBSC's bloodborne_re.cpp) build unchanged.
#pragma once

#include <array>
#include <cstdint>
#include <span>

namespace Core {

struct alignas(16) GuestRegisterSnapshot {
    std::uint64_t rax;
    std::uint64_t rbx;
    std::uint64_t rcx;
    std::uint64_t rdx;
    std::uint64_t rsi;
    std::uint64_t rdi;
    std::uint64_t rbp;
    std::uint64_t rsp;
    std::uint64_t r8;
    std::uint64_t r9;
    std::uint64_t r10;
    std::uint64_t r11;
    std::uint64_t r12;
    std::uint64_t r13;
    std::uint64_t r14;
    std::uint64_t r15;
    std::uint64_t rflags;
    std::uint32_t mxcsr;
    alignas(16) std::array<std::array<std::uint8_t, 32>, 16> ymm;
};

/// Called on the game's thread, on its stack (keep it small), with the System V ABI.
using GuestCodeHook = void(__attribute__((sysv_abi)) *)(std::uint64_t tag,
                                                         const GuestRegisterSnapshot* registers);

/// Installs a hook at `address` over `expected_instructions` (checked byte for byte). False: the
/// bytes differ (another build of the game, or a port patch there), the span is not whole
/// position-independent instructions, it overlaps a hook, or no trampoline memory is in reach.
bool InstallGuestCodeHook(void* address, std::span<const std::uint8_t> expected_instructions,
                          std::uint64_t tag, GuestCodeHook hook,
                          const std::uint8_t* skip_flag = nullptr);

/// The protection the game's code goes back to after the port patches it: read and execute, or
/// read, write and execute once a game module writes its own code (BBSC's byte patches, some
/// switched while the game runs) and calls KeepGuestCodeWritable.
int GuestCodeProtection();
void KeepGuestCodeWritable();

} // namespace Core
