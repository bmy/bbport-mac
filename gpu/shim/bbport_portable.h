// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: Linux/macOS differences for the diagnostics code: registers of a signal context, and
// reading this process's own memory without faulting (process_vm_readv on Linux, Mach on macOS).
#pragma once

#include <cstddef>
#include <cstdint>
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/ucontext.h>
#else
#include <sys/uio.h>
#include <ucontext.h>
#include <unistd.h>
#endif

namespace BbPortable {

/// Copies `size` bytes at `address` into `out`; false when any of it is unmapped (no fault).
inline bool ReadSelf(void* out, const void* address, std::size_t size) {
#ifdef __APPLE__
    mach_vm_size_t got = 0;
    return mach_vm_read_overwrite(mach_task_self(),
                                  static_cast<mach_vm_address_t>(reinterpret_cast<std::uintptr_t>(address)),
                                  size, static_cast<mach_vm_address_t>(reinterpret_cast<std::uintptr_t>(out)),
                                  &got) == KERN_SUCCESS &&
           got == size;
#else
    iovec local{out, size}, remote{const_cast<void*>(address), size};
    return process_vm_readv(getpid(), &local, 1, &remote, 1, 0) == static_cast<ssize_t>(size);
#endif
}

/// How many bytes of [address, address + size) could be read into `out` from the start (stops at
/// the first unmapped page).
inline std::size_t ReadSelfPrefix(void* out, const void* address, std::size_t size) {
#ifdef __APPLE__
    const std::uintptr_t start = reinterpret_cast<std::uintptr_t>(address);
    std::size_t done = 0;
    while (done < size) {
        const std::uintptr_t at = start + done;
        const std::size_t room = 4096 - (at & 4095);
        const std::size_t n = size - done < room ? size - done : room;
        if (!ReadSelf(static_cast<unsigned char*>(out) + done, reinterpret_cast<const void*>(at), n)) {
            break;
        }
        done += n;
    }
    return done;
#else
    iovec local{out, size}, remote{const_cast<void*>(address), size};
    const ssize_t got = process_vm_readv(getpid(), &local, 1, &remote, 1, 0);
    return got > 0 ? static_cast<std::size_t>(got) : 0;
#endif
}

inline std::uint64_t Rip(const void* context) {
    const auto* uc = static_cast<const ucontext_t*>(context);
#ifdef __APPLE__
    return uc->uc_mcontext->__ss.__rip;
#else
    return static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RIP]);
#endif
}
inline std::uint64_t Rsp(const void* context) {
    const auto* uc = static_cast<const ucontext_t*>(context);
#ifdef __APPLE__
    return uc->uc_mcontext->__ss.__rsp;
#else
    return static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RSP]);
#endif
}
inline std::uint64_t Rbp(const void* context) {
    const auto* uc = static_cast<const ucontext_t*>(context);
#ifdef __APPLE__
    return uc->uc_mcontext->__ss.__rbp;
#else
    return static_cast<std::uint64_t>(uc->uc_mcontext.gregs[REG_RBP]);
#endif
}

} // namespace BbPortable
