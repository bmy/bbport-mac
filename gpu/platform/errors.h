// SPDX-FileCopyrightText: Copyright 2026 bbport contributors
// SPDX-License-Identifier: GPL-2.0-or-later
// bbport: error codes the PS4 system libraries return (the values games compare against).
#pragma once

#include "common/types.h"

namespace BbErr {

inline constexpr s32 Ok = 0;
inline constexpr s32 Fail = -1;

// Kernel (errno values with the 0x8002 prefix).
inline constexpr s32 KernelBadFd = s32(0x80020009);    // EBADF
inline constexpr s32 KernelInvalid = s32(0x80020016);  // EINVAL
inline constexpr s32 KernelNotFound = s32(0x80020002); // ENOENT
inline constexpr s32 KernelNoMemory = s32(0x8002000C); // ENOMEM
inline constexpr s32 KernelBusy = s32(0x80020010);     // EBUSY
inline constexpr s32 KernelTimedOut = s32(0x8002003C); // ETIMEDOUT
inline constexpr s32 KernelAgain = s32(0x80020023);    // EAGAIN
inline constexpr s32 KernelFault = s32(0x8002000E);    // EFAULT
inline constexpr s32 KernelNameTooLong = s32(0x8002003F); // ENAMETOOLONG

// errno values (posix_* calls return -1 and set these).
inline constexpr int PosixNoEntry = 2;  // ENOENT
inline constexpr int PosixBadFd = 9;    // EBADF
inline constexpr int PosixNoMemory = 12; // ENOMEM

// libSceGnmDriver.
inline constexpr s32 GnmFailure = s32(0x8EEE00FF);
inline constexpr s32 GnmInvalid = s32(0x80D11000); ///< bad submission arguments
inline constexpr s32 GnmFlipQueueFull = s32(0x80D11081);
inline constexpr s32 GnmValidationOff = s32(0x80D13FFF);
inline constexpr s32 GnmRazorNotLoaded = s32(0x80D15001);
inline constexpr s32 GnmCaptureFailed = s32(0x80D1500F);
inline constexpr s32 GnmQueueBadPipe = s32(0x80D17000);
inline constexpr s32 GnmQueueBadQueue = s32(0x80D17001);
inline constexpr s32 GnmQueueBadRingSize = s32(0x80D17002);
inline constexpr s32 GnmQueueBadRingBase = s32(0x80D17003);
inline constexpr s32 GnmQueueBadReadPtr = s32(0x80D17004);

} // namespace BbErr
