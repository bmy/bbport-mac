// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: machine code for the native detours in the game's code (bbport_guest_hooks,
// bbport_frame_rate): a small x86-64 emitter for the stubs the detours jump to.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>

namespace BbGuestCode {
using u64 = std::uint64_t;
using u8 = std::uint8_t;

/// A stub's bytes, built in place.
struct Emitter {
    std::array<u8, 512> code{};
    std::size_t size = 0;
    bool overflow = false; ///< more than `code` holds: PlaceStub refuses it
    void Bytes(std::initializer_list<u8> bytes) {
        for (const u8 b : bytes) {
            if (size < code.size()) code[size] = b;
            ++size;
        }
        overflow = overflow || size > code.size();
    }
    void U32(std::uint32_t v) {
        u8 b[4];
        std::memcpy(b, &v, 4);
        Bytes({b[0], b[1], b[2], b[3]});
    }
    void U64(u64 v) {
        U32(std::uint32_t(v));
        U32(std::uint32_t(v >> 32));
    }
    /// Saves the caller-saved registers (the handler may change them), aligns the stack.
    void SaveVolatile() {
        Bytes({0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53});
        Bytes({0x41, 0x54});             // push r12 (holds the unaligned stack pointer)
        Bytes({0x49, 0x89, 0xe4});       // mov r12, rsp
        Bytes({0x48, 0x83, 0xe4, 0xf0}); // and rsp, -16
    }
    void CallAbs(const void* fn) {
        Bytes({0x48, 0xb8}); // movabs rax, fn
        U64(reinterpret_cast<u64>(fn));
        Bytes({0xff, 0xd0}); // call rax
    }
    /// xmm0..15 below the aligned stack (after SaveVolatile): a function's float arguments.
    void SaveXmm() {
        Bytes({0x48, 0x81, 0xec}); // sub rsp, 256
        U32(256);
        for (u8 i = 0; i < 16; ++i) { // movdqu [rsp + 16 i], xmm_i
            Bytes({0xf3});
            if (i >= 8) Bytes({0x44});
            Bytes({0x0f, 0x7f, u8(0x84 | (i & 7) << 3), 0x24});
            U32(16u * i);
        }
    }
    void RestoreXmm() {
        for (u8 i = 0; i < 16; ++i) { // movdqu xmm_i, [rsp + 16 i]
            Bytes({0xf3});
            if (i >= 8) Bytes({0x44});
            Bytes({0x0f, 0x6f, u8(0x84 | (i & 7) << 3), 0x24});
            U32(16u * i);
        }
        Bytes({0x48, 0x81, 0xc4}); // add rsp, 256
        U32(256);
    }
    /// jmp [rip]: to `target` anywhere.
    void JmpAbs(u64 target) {
        Bytes({0xff, 0x25, 0x00, 0x00, 0x00, 0x00});
        U64(target);
    }
    /// The 32-bit value at `address` into the stack frame's [rbp + offset] (rax kept).
    void StoreToFrame(const void* address, std::int8_t offset) {
        Bytes({0x50, 0x48, 0xb8}); // push rax; movabs rax, address
        U64(reinterpret_cast<u64>(address));
        Bytes({0x8b, 0x00});                          // mov eax, [rax]
        Bytes({0x89, 0x45, u8(offset), 0x58});        // mov [rbp + offset], eax; pop rax
    }
    void RestoreVolatile() {
        Bytes({0x4c, 0x89, 0xe4}); // mov rsp, r12
        Bytes({0x41, 0x5c});       // pop r12
        Bytes({0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x58});
    }
};

} // namespace BbGuestCode
